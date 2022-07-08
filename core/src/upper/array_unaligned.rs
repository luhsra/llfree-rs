use core::fmt;
use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use crate::{Error, Result, Size};
use super::{Alloc, Local, MAGIC, MAX_PAGES};
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

/// This allocator is equivalent to the `ArrayAligned` allocator,
/// except for the entry array not beeing cache-line aligned.
/// It sole purpose is to show the effect of false-sharing on this array.
#[repr(align(64))]
pub struct ArrayUnalignedAlloc<L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    subtrees: Box<[Aligned]>,
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

/// *Not* cache-line aligned, to test false-sharing
#[repr(transparent)]
struct Aligned(Atomic<Entry3>);

impl<L: LowerAlloc> Index<usize> for ArrayUnalignedAlloc<L> {
    type Output = Atomic<Entry3>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.subtrees[index].0
    }
}

unsafe impl<L: LowerAlloc> Send for ArrayUnalignedAlloc<L> {}
unsafe impl<L: LowerAlloc> Sync for ArrayUnalignedAlloc<L> {}

impl<L: LowerAlloc> fmt::Debug for ArrayUnalignedAlloc<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;
        for (i, entry) in self.subtrees.iter().enumerate() {
            let pte = entry.0.load();
            writeln!(f, "    {i:>3}: {pte:?}")?;
        }
        writeln!(f, "    empty: {:?}", AStackDbg(&self.empty, self))?;
        writeln!(f, "    partial_l0: {:?}", AStackDbg(&self.partial_l0, self))?;
        writeln!(f, "    partial_l1: {:?}", AStackDbg(&self.partial_l1, self))?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Alloc for ArrayUnalignedAlloc<L> {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < Self::MAPPING.span(2) * cores {
            error!("memory {} < {}", memory.len(), Self::MAPPING.span(2) * cores);
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
        subtrees.resize_with(pte3_num, || Aligned(Atomic::new(Entry3::new())));
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
    fn put(&self, _core: usize, addr: u64) -> Result<Size> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let i = page / Self::MAPPING.span(2);
        let pte = self[i].load();
        self.put_lower(page, pte)
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let pte = self[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.free();
        }
        pages
    }
}

impl<L: LowerAlloc> Drop for ArrayUnalignedAlloc<L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}

impl<L: LowerAlloc> Default for ArrayUnalignedAlloc<L> {
    fn default() -> Self {
        Self {
            meta: null_mut(),
            lower: L::default(),
            local: Box::new([]),
            subtrees: Box::new([]),
            empty: AStack::default(),
            partial_l1: AStack::default(),
            partial_l0: AStack::default(),
        }
    }
}

impl<L: LowerAlloc> ArrayUnalignedAlloc<L> {
    const MAPPING: Mapping<3> = Mapping([512]).with_lower(&L::MAPPING);
    const PTE3_FULL: usize = 8 * Self::MAPPING.span(1);

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
        } else if max > Self::PTE3_FULL {
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
            } else if pages > Self::PTE3_FULL {
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
    fn reserve(&self, huge: bool) -> Result<usize> {
        while let Some((i, r)) = self.partial(huge).pop_update(self, |v| {
            // Skip empty entries
            if v.free() < Self::MAPPING.span(2) {
                v.dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
            } else {
                None
            }
        }) {
            if r.is_ok() {
                info!("reserve partial {i}");
                return Ok(i * Self::MAPPING.span(2));
            }
            self.empty.push(self, i);
        }

        if let Some((i, r)) = self.empty.pop_update(self, |v| {
            v.dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
        }) {
            debug_assert!(r.is_ok());
            info!("reserve empty {i}");
            Ok(i * Self::MAPPING.span(2))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    /// Allocate a small or huge page from the lower alloc.
    fn get_lower(&self, core: usize, huge: bool) -> Result<usize> {
        let start_a = self.local[core].start(huge);
        let mut start = *start_a;

        if start == usize::MAX {
            start = self.reserve(huge)?;
        } else {
            let i = start / Self::MAPPING.span(2);
            if self[i]
                .update(|v| v.dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2)))
                .is_err()
            {
                start = self.reserve(huge)?;
                if self[i].update(Entry3::unreserve).is_err() {
                    panic!("Unreserve failed");
                }
            }
        }

        let page = self.lower.get(core, huge, start)?;
        *start_a = page;
        Ok(page)
    }

    /// Free a small or huge page from the lower alloc.
    fn put_lower(&self, page: usize, pte: Entry3) -> Result<Size> {
        let max = self
            .pages()
            .saturating_sub(Self::MAPPING.round(2, page))
            .min(Self::MAPPING.span(2));
        if pte.free() >= max {
            error!("Not allocated 0x{page:x}, {:x} >= {max:x}", pte.free());
            return Err(Error::Address);
        }

        let i = page / Self::MAPPING.span(2);
        let huge = self.lower.put(page)?;
        if let Ok(pte3) = self[i].update(|v| v.inc(Self::MAPPING.span(huge as _), max)) {
            if !pte3.reserved() {
                let new_pages = pte3.free() + Self::MAPPING.span(huge as _);
                if pte3.free() <= Self::PTE3_FULL && new_pages > Self::PTE3_FULL {
                    // Add back to partial
                    self.partial(huge).push(self, i);
                }
            }
            Ok(if huge { Size::L1 } else { Size::L0 })
        } else {
            error!("Corruption l3 i{i} p=-{huge:?}");
            Err(Error::Corruption)
        }
    }
}
