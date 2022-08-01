use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, hint};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use super::{Alloc, Local, MAGIC, MAX_PAGES};
use crate::atomic::{ANode, AStack, AStackDbg, Atomic};
use crate::entry::{Entry2, Entry3};
use crate::lower::LowerAlloc;
use crate::table::Mapping;
use crate::upper::CAS_RETRIES;
use crate::util::{log2, Page};
use crate::{Error, Result};

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
    /// List of idx to subtrees that are partially allocated with huge pages
    partial: AStack<Entry3>,
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
        writeln!(f, "    partial: {:?}", AStackDbg(&self.partial, self))?;
        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.pte.load())?;
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
    fn init(&mut self, mut cores: usize, mut memory: &mut [Page], persistent: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < Self::MAPPING.span(2) * cores {
            warn!(
                "memory {} < {}",
                memory.len(),
                Self::MAPPING.span(2) * cores
            );
            cores = 1.max(memory.len() / Self::MAPPING.span(2));
        }

        if persistent {
            // Last frame is reserved for metadata
            let (m, rem) = memory.split_at_mut((memory.len() - 1).min(MAX_PAGES));
            let meta = rem[0].cast_mut::<Meta>();
            self.meta = meta;
            memory = m;
        }

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Local::new);
        self.local = local.into();

        // Create lower allocator
        self.lower = L::new(cores, memory, persistent);

        // Array with all pte3
        let pte3_num = Self::MAPPING.num_pts(2, self.lower.pages());
        let mut pte3s = Vec::with_capacity(pte3_num);
        pte3s.resize_with(pte3_num, || Atomic::new(Entry3::new()));
        self.subtrees = pte3s.into();

        self.empty = AStack::default();
        self.partial = AStack::default();

        Ok(())
    }

    fn recover(&self) -> Result<()> {
        let meta = unsafe { &mut *self.meta };
        if meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            info!("recover p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            self.recover_inner(deep)?;
            meta.active.store(1, Ordering::SeqCst);
            Ok(())
        } else {
            Err(Error::Initialization)
        }
    }

    fn free_all(&self) -> Result<()> {
        info!("free all p={}", self.pages());
        self.lower.free_all();

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

        self.enqueue(pte3_num - 1, max);

        if !self.meta.is_null() {
            let meta = unsafe { &mut *self.meta };
            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
            meta.active.store(1, Ordering::SeqCst);
        }
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        info!("reserve all p={}", self.pages());
        self.lower.reserve_all();

        // Set all entries to zero
        let pte3_num = Self::MAPPING.num_pts(2, self.pages());
        for i in 0..pte3_num {
            self[i].store(Entry3::new());
        }
        // Clear the lists
        self.empty.set(Entry3::default().with_next(None));
        self.partial.set(Entry3::default().with_next(None));

        if !self.meta.is_null() {
            let meta = unsafe { &mut *self.meta };
            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
            meta.active.store(1, Ordering::SeqCst);
        }
        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order > L::MAX_ORDER {
            error!("invalid order: !{order} <= {}", L::MAX_ORDER);
            return Err(Error::Memory);
        }

        // Select local data (which can be shared between cores if we do not have enough memory)
        let c = core % self.local.len();
        let start_a = &self.local[c].start;
        let pte_a = &self.local[c].pte;

        // Incremet or clear (atomic sync with put dec)
        let mut pte_res = pte_a.update(|v| v.dec(1 << order));

        for _ in 0..CAS_RETRIES {
            let (pte, mut start) = match pte_res {
                Ok(pte) => (pte, start_a.load()),
                Err(pte) => {
                    // Try reserve new subtree
                    let (s, new_pte) = match self.reserve(order) {
                        Ok((s, pte)) => (s, pte),
                        Err(Error::Memory) => {
                            hint::spin_loop();
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                    match self.cas_reserved(pte_a, pte, new_pte) {
                        Err(Error::CAS) => {
                            let max = self
                                .pages()
                                .saturating_sub(Self::MAPPING.round(2, s))
                                .min(Self::MAPPING.span(2));

                            // Rollback reservation & and counter
                            if let Ok(pte) = self[new_pte.idx()]
                                .update(|v| v.unreserve_add(new_pte.free() + (1 << order), max))
                            {
                                debug_assert!(pte.idx() == Entry3::IDX_MAX);
                                self.enqueue(new_pte.idx(), pte.free() + (1 << order));
                            } else {
                                error!("get - rollback reservation failed");
                                return Err(Error::Corruption);
                            }
                            // CAS -> retry
                            pte_res = pte_a.update(|v| v.dec(1 << order));
                            hint::spin_loop();
                            continue;
                        }
                        Err(e) => return Err(e),
                        Ok(_) => (),
                    }
                    start_a.store(s);
                    (new_pte, s)
                }
            };

            // Load starting idx
            if start / Self::MAPPING.span(2) != pte.idx() {
                start = pte.idx() * Self::MAPPING.span(2)
            }

            // Allocate in subtree
            match self.lower.get(start, order) {
                Ok(page) => {
                    // small pages
                    if order < log2(64) {
                        start_a.store(page);
                    }
                    return Ok(unsafe { self.lower.memory().start.add(page as _) } as u64);
                }
                Err(Error::Memory) => {
                    // Counter reset and retry
                    warn!("alloc failed o={order} => retry");
                    let max = self
                        .pages()
                        .saturating_sub(Self::MAPPING.round(2, start))
                        .min(Self::MAPPING.span(2));
                    // Increment global to prevent race condition with concurrent reservation
                    match self[pte.idx()].update(|v| v.inc(1 << order, max)) {
                        Ok(pte) => pte_res = Err(pte), // -> reserve new
                        Err(pte) => {
                            error!(
                                "Counter reset failed o={order} {}: {pte:?}",
                                start / Self::MAPPING.span(2)
                            );
                            return Err(Error::Corruption);
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }

        error!("Exceeding retries {self:?}");
        Err(Error::Memory)
    }

    #[inline(never)]
    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()> {
        if order > L::MAX_ORDER {
            error!("invalid order: !{order} <= {}", L::MAX_ORDER);
            return Err(Error::Memory);
        }

        let num_pages = 1 << order;

        if addr % (num_pages * Page::SIZE) as u64 != 0
            || !self.lower.memory().contains(&(addr as _))
        {
            error!(
                "invalid addr 0x{addr:x} r={:?} o={order}",
                self.lower.memory()
            );
            return Err(Error::Address);
        }

        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        self.lower.put(page, order)?;

        let i = page / Self::MAPPING.span(2);
        // Save the modified subtree id for the push-reserve heuristic
        let c = core % self.local.len();
        let local = &self.local[c];

        let max = self
            .pages()
            .saturating_sub(Self::MAPPING.round(2, page))
            .min(Self::MAPPING.span(2));

        // Try decrement own subtree first
        let l_pte = if let Err(pte) = local.pte.update(|v| v.inc_idx(num_pages, i, max)) {
            if pte.idx() == i {
                error!("inc failed L{i}: {pte:?} o={order}");
                return Err(Error::Corruption);
            }
            pte
        } else {
            if c == core {
                local.frees_push(i);
            }
            return Ok(());
        };

        // Subtree not owned by us
        match self[i].update(|v| v.inc(num_pages, max)) {
            Ok(pte) => {
                let new_pages = pte.free() + num_pages;
                if !pte.reserved() && new_pages > Self::ALMOST_FULL {
                    // check if recent frees also operated in this subtree
                    if core == c && local.frees_related(i) {
                        // Try to reserve it for bulk frees
                        if let Ok(new_pte) = self[i].update(|v| v.reserve(Self::ALMOST_FULL)) {
                            match self.cas_reserved(&local.pte, l_pte, new_pte.with_idx(i)) {
                                Err(Error::CAS) => {
                                    warn!("rollback {i}");
                                    // Rollback reservation
                                    let max = (self.pages() - i * Self::MAPPING.span(2))
                                        .min(Self::MAPPING.span(2));
                                    if let Err(_) =
                                        self[i].update(|v| v.unreserve_add(new_pte.free(), max))
                                    {
                                        error!("put - reservation rollback failed");
                                        return Err(Error::Corruption);
                                    }
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {
                                    local.start.store(i * Self::MAPPING.span(2));
                                    return Ok(());
                                }
                            }
                        }
                    }

                    // Add to partially free list
                    // Only if not already in list
                    if pte.idx() == Entry3::IDX_MAX && pte.free() <= Self::ALMOST_FULL {
                        self.partial.push(self, i);
                    }
                }
                if c == core {
                    local.frees_push(i);
                }
                Ok(())
            }
            Err(pte) => {
                error!("inc failed i{i}: {pte:?} o={order}");
                Err(Error::Corruption)
            }
        }
    }

    #[cold]
    fn pages_needed(&self, cores: usize) -> usize {
        Self::MAPPING.span(2) * cores
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_for_each_pte2(&self, f: fn(Entry2)) {
        self.lower.dbg_for_each_pte2(f)
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
            pages -= local.pte.load().free();
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
            partial: AStack::default(),
        }
    }
}

impl<L: LowerAlloc> ArrayAtomicAlloc<L> {
    const MAPPING: Mapping<3> = Mapping([512]).with_lower(&L::MAPPING);
    const ALMOST_FULL: usize = 1 << L::MAX_ORDER;

    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover_inner(&self, deep: bool) -> Result<usize> {
        if deep {
            warn!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let page = i * Self::MAPPING.span(2);
            let pages = self.lower.recover(page, deep)?;

            self[i].store(Entry3::new_table(pages, false));

            // Add to lists
            self.enqueue(i, pages);
            total += pages;
        }
        Ok(total)
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees,
    /// and allocates a page from it in one step.
    fn reserve(&self, order: usize) -> Result<(usize, Entry3)> {
        while let Some((i, r)) = self.partial.pop_update(self, |v| {
            v.reserve_partial(Self::ALMOST_FULL, Self::MAPPING.span(2))
        }) {
            info!("reserve partial {i}");
            match r {
                Ok(pte) => {
                    let pte = pte.with_idx(i).dec(1 << order).unwrap();
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
            .pop_update(self, |v| v.reserve_empty(Self::MAPPING.span(2)))
        {
            info!("reserve empty {i}");
            if let Ok(pte) = r {
                let pte = pte.with_idx(i).dec(1 << order).unwrap();
                return Ok((i * Self::MAPPING.span(2), pte));
            }
        }

        error!("reserve: no memory");
        Err(Error::Memory)
    }

    fn enqueue(&self, i: usize, new_pages: usize) {
        // Add to list if new counter is small enough
        if new_pages == Self::MAPPING.span(2) {
            self.empty.push(self, i);
        } else if new_pages > Self::ALMOST_FULL {
            self.partial.push(self, i);
        }
    }

    /// Swap the current reserved subtree out replacing it with a new one.
    /// The old subtree is unreserved and added back to the lists.
    fn cas_reserved(&self, pte_a: &Atomic<Entry3>, old_pte: Entry3, new_pte: Entry3) -> Result<()> {
        let pte = pte_a
            .compare_exchange(old_pte, new_pte)
            .map_err(|_| Error::CAS)?;
        let i = pte.idx();
        if i >= Entry3::IDX_MAX {
            return Ok(());
        }

        let max = (self.pages() - i * Self::MAPPING.span(2)).min(Self::MAPPING.span(2));
        if let Ok(v) = self[i].update(|v| v.unreserve_add(pte.free(), max)) {
            // Only if not already in list
            if v.idx() == Entry3::IDX_MAX {
                // Add to list
                self.enqueue(i, v.free() + pte.free());
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }
}
