use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, mem};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use super::{Alloc, Error, Local, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::atomic::{AStack, AStackDbg, Atomic};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
use crate::table::Mapping;
use crate::util::Page;

const PUTS_RESERVE: usize = 4;

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// This allocator splits its memory range into 1G chunks.
/// Giant pages are directly allocated in it.
/// For smaller pages, however, the 1G chunk is handed over to the
/// lower allocator, managing these smaller allocations.
/// These 1G chunks are, due to the inner workins of the lower allocator,
/// called 1G *subtrees*.
///
/// This allocator stores the level three entries (subtree roots) in a
/// packed array.
/// The subtree reservation is speed up using free lists for
/// empty and partially empty subtrees.
/// These free lists are implemented as atomic linked lists with their next
/// pointers stored inside the level 3 entries.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[repr(align(64))]
pub struct ArrayAtomicAlloc<L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    subtrees: Box<[Atomic<Entry3>]>,
    /// CPU local data
    local: Box<[Local<PUTS_RESERVE>]>,
    /// Metadata of the lower alloc
    lower: L,

    /// List of idx to subtrees that are not allocated at all
    empty: AStack<Entry3>,
    /// List of idx to subtrees that are partially allocated with small pages
    partial_l1: AStack<Entry3>,
    /// List of idx to subtrees that are partially allocated with huge pages
    partial_l0: AStack<Entry3>,
}

unsafe impl<L: LowerAlloc> Send for ArrayAtomicAlloc<L> {}
unsafe impl<L: LowerAlloc> Sync for ArrayAtomicAlloc<L> {}

impl<L: LowerAlloc> fmt::Debug for ArrayAtomicAlloc<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;
        for (i, entry) in self.subtrees.iter().enumerate() {
            let pte = entry.load();
            writeln!(f, "    {i:>3}: {pte:?}")?;
        }
        writeln!(f, "    empty: {:?}", AStackDbg(&self.empty, self))?;
        writeln!(f, "    partial_l0: {:?}", AStackDbg(&self.partial_l0, self))?;
        writeln!(f, "    partial_l1: {:?}", AStackDbg(&self.partial_l1, self))?;
        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: L0 {:?}", local.pte(false))?;
            writeln!(f, "         L1 {:?}", local.pte(true))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Index<usize> for ArrayAtomicAlloc<L> {
    type Output = Atomic<Entry3>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.subtrees[index]
    }
}

impl<L: LowerAlloc> Alloc for ArrayAtomicAlloc<L> {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < Self::MAPPING.span(2) * cores {
            error!("memory {} < {}", memory.len(), MIN_PAGES * cores);
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        let (memory, rem) = memory.split_at_mut((memory.len() - 1).min(MAX_PAGES));
        let meta = rem[0].cast_mut::<Meta>();
        self.meta = meta;

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Local::new);
        self.local = local.into();

        // Create lower allocator
        self.lower = L::new(cores, memory);

        // Array with all pte3
        let pte3_num = Self::MAPPING.num_pts(2, self.lower.pages());
        let mut subtrees = Vec::with_capacity(pte3_num);
        subtrees.resize_with(pte3_num, || Atomic::new(Entry3::new()));
        self.subtrees = subtrees.into();

        self.empty = AStack::default();
        self.partial_l0 = AStack::default();
        self.partial_l1 = AStack::default();

        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            info!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            self.recover(deep)?;
        } else {
            info!("Setup allocator state p={}", self.pages());
            self.setup();

            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        meta.active.store(1, Ordering::SeqCst);
        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        self.get_lower(core, size == Size::L1)
            .map(|p| unsafe { self.lower.memory().start.add(p as _) } as u64)
    }

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<Size> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let i = page / Self::MAPPING.span(2);
        let pte = self[i].load();
        self.put_lower(core, page, pte)
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let pte = self[i].load();
            pages -= pte.free();
        }
        // Pages allocated in reserved subtrees
        for local in self.local.iter() {
            pages -= local.pte(false).free();
            pages -= local.pte(true).free();
        }
        pages
    }
}

impl<L: LowerAlloc> Drop for ArrayAtomicAlloc<L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}
impl<L: LowerAlloc> Default for ArrayAtomicAlloc<L> {
    fn default() -> Self {
        Self {
            meta: null_mut(),
            subtrees: Box::new([]),
            local: Box::new([]),
            lower: L::default(),
            empty: AStack::default(),
            partial_l1: AStack::default(),
            partial_l0: AStack::default(),
        }
    }
}

impl<L: LowerAlloc> ArrayAtomicAlloc<L> {
    const MAPPING: Mapping<3> = Mapping([512]).with_lower(&L::MAPPING);
    const ALMOST_FULL: usize = Self::MAPPING.span(2) / 64;

    /// Setup a new allocator.
    #[cold]
    fn setup(&mut self) {
        self.lower.clear();

        // Add all entries to the empty list
        let pte3_num = Self::MAPPING.num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            self[i].store(Entry3::empty(Self::MAPPING.span(2)));
            self.empty.push(self, i);
        }

        // The last one may be cut off
        let max =
            (self.pages() - (pte3_num - 1) * Self::MAPPING.span(2)).min(Self::MAPPING.span(2));
        self[pte3_num - 1].store(Entry3::new().with_free(max));

        if max == Self::MAPPING.span(2) {
            self.empty.push(self, pte3_num - 1);
        } else if max > Self::ALMOST_FULL {
            self.partial_l0.push(self, pte3_num - 1);
        }
    }

    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover(&self, deep: bool) -> Result<usize> {
        if deep {
            warn!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let page = i * Self::MAPPING.span(2);
            let (pages, size) = self.lower.recover(page, deep)?;

            self[i].store(Entry3::new_table(pages, size, false));

            // Add to lists
            if pages == Self::MAPPING.span(2) {
                self.empty.push(self, i);
            } else if pages > Self::ALMOST_FULL {
                self.partial(size == Size::L1).push(self, i);
            }
            total += pages;
        }
        Ok(total)
    }

    fn partial(&self, huge: bool) -> &AStack<Entry3> {
        if huge {
            &self.partial_l1
        } else {
            &self.partial_l0
        }
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees,
    /// and allocates a page from it in one step.
    fn reserve(&self, huge: bool) -> Result<(usize, Entry3)> {
        while let Some((i, r)) = self.partial(huge).pop_update(self, |v| {
            v.reserve_partial(huge, Self::ALMOST_FULL, Self::MAPPING.span(2))
        }) {
            info!("reserve partial {i}");
            match r {
                Ok(pte) => {
                    let pte = pte
                        .dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
                        .unwrap()
                        .with_idx(i);
                    return Ok((i * Self::MAPPING.span(2), pte));
                }
                Err(pte) => {
                    // Skip empty entries
                    if !pte.reserved() && pte.free() == Self::MAPPING.span(2) {
                        self.empty.push(self, i);
                    }
                }
            }
        }

        while let Some((i, r)) = self
            .empty
            .pop_update(self, |v| v.reserve_empty(huge, Self::MAPPING.span(2)))
        {
            info!("reserve empty {i}");
            if let Ok(pte) = r {
                let pte = pte
                    .dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
                    .unwrap()
                    .with_idx(i);
                return Ok((i * Self::MAPPING.span(2), pte));
            }
        }

        error!("No memory {self:?}");
        Err(Error::Memory)
    }

    /// Swap the current reserved subtree out replacing it with a new one.
    /// The old subtree is unreserved and added back to the lists.
    fn swap_reserved(&self, huge: bool, new_pte: Entry3, pte_a: &mut Entry3) -> Result<()> {
        let pte = mem::replace(pte_a, new_pte);
        let i = pte.idx();
        if i >= Entry3::IDX_MAX {
            return Ok(());
        }

        let max = (self.pages() - i * Self::MAPPING.span(2)).min(Self::MAPPING.span(2));
        if let Ok(v) = self[i].update(|v| v.unreserve_add(huge, pte, max, Self::MAPPING.span(2))) {
            // Only if not already in list
            if v.idx() == Entry3::IDX_MAX {
                // Add to list if new counter is small enough
                let new_pages = v.free() + pte.free();
                if new_pages == max {
                    self.empty.push(self, i);
                } else if new_pages > Self::ALMOST_FULL {
                    self.partial(huge).push(self, i);
                }
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Allocate a small or huge page from the lower alloc.
    fn get_lower(&self, core: usize, huge: bool) -> Result<usize> {
        let start_a = self.local[core].start(huge);
        let pte_a = self.local[core].pte(huge);
        let mut start = *start_a;

        if start == usize::MAX {
            let (s, pte) = self.reserve(huge)?;
            *pte_a = pte;
            start = s;
        } else {
            // Incremet or clear (atomic sync with put dec)
            if let Some(pte) = pte_a.dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2)) {
                *pte_a = pte;
            } else {
                let (s, new_pte) = self.reserve(huge)?;
                self.swap_reserved(huge, new_pte, pte_a)?;
                start = s;
            }
        }

        let page = self.lower.get(core, huge, start)?;
        *start_a = page;
        Ok(page)
    }

    /// Free a small or huge page from the lower alloc.
    fn put_lower(&self, core: usize, page: usize, pte: Entry3) -> Result<Size> {
        let max = self
            .pages()
            .saturating_sub(Self::MAPPING.round(2, page))
            .min(Self::MAPPING.span(2));
        if pte.free() >= max {
            error!("Not allocated {page} (i{})", page / Self::MAPPING.span(2));
            return Err(Error::Address);
        }

        let i = page / Self::MAPPING.span(2);
        let huge = self.lower.put(page)?;
        let size = if huge { Size::L1 } else { Size::L0 };

        let local = &self.local[core];
        let _push = local.defer_frees_push(i);

        // Try decrement own subtree first
        let pte_a = local.pte(huge);
        if let Some(pte) = pte_a.inc_idx(Self::MAPPING.span(huge as _), i, max) {
            *pte_a = pte;
            return Ok(size);
        }

        // Subtree not owned by us
        if let Ok(pte) = self[i].update(|v| v.inc(Self::MAPPING.span(huge as _), max)) {
            if !pte.reserved() {
                let new_pages = pte.free() + Self::MAPPING.span(huge as _);

                // check if recent frees also operated in this subtree
                if new_pages > Self::ALMOST_FULL && local.frees_related(i) {
                    // Try to reserve it for bulk frees
                    if let Ok(pte) = self[i]
                        .update(|v| v.reserve(Self::MAPPING.span(huge as _), Self::MAPPING.span(2)))
                    {
                        let pte = pte.with_idx(i);
                        // warn!("put reserve {i}");
                        self.swap_reserved(huge, pte, pte_a)?;
                        *local.start(huge) = page;
                        return Ok(size);
                    }
                }

                // Add to partially free list
                // Only if not already in list
                if pte.idx() == Entry3::IDX_MAX
                    && pte.free() <= Self::ALMOST_FULL
                    && new_pages > Self::ALMOST_FULL
                {
                    self.partial(huge).push(self, i);
                }
            }
            Ok(size)
        } else {
            error!("Corruption l3 i{i} p=-{huge:?}");
            Err(Error::Corruption)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::lower::{CacheLower, FixedLower};

    use super::ArrayAtomicAlloc;

    #[test]
    fn correct_sizes() {
        assert_eq!(ArrayAtomicAlloc::<FixedLower>::ALMOST_FULL, 8 * 512);
        assert_eq!(ArrayAtomicAlloc::<CacheLower<512>>::ALMOST_FULL, 8 * 512);
    }
}
