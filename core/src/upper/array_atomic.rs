use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, hint};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use super::{Alloc, Local, MAGIC, MAX_PAGES};
use crate::atomic::{ANode, AStack, AStackDbg, Atomic};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
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
pub struct ArrayAtomic<L: LowerAlloc> {
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

unsafe impl<L: LowerAlloc> Send for ArrayAtomic<L> {}
unsafe impl<L: LowerAlloc> Sync for ArrayAtomic<L> {}

impl<L: LowerAlloc> fmt::Debug for ArrayAtomic<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;

        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;
        writeln!(f, "    empty: {:?}", AStackDbg(&self.empty, self))?;
        writeln!(f, "    partial: {:?}", AStackDbg(&self.partial, self))?;
        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.pte.load())?;
        }

        for (i, entry) in self.subtrees.iter().enumerate() {
            let pte = entry.load();
            writeln!(f, "    {i:>3}: {pte:?}")?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Index<usize> for ArrayAtomic<L> {
    type Output = Atomic<Entry3>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.subtrees[index]
    }
}

impl<L: LowerAlloc> Alloc for ArrayAtomic<L> {
    #[cold]
    fn init(&mut self, mut cores: usize, mut memory: &mut [Page], persistent: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < self.pages_needed(cores) {
            warn!("memory {} < {}", memory.len(), self.pages_needed(cores));
            cores = 1.max(memory.len() / L::MAPPING.span(2));
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
        let pte3_num = L::MAPPING.num_pts(2, self.lower.pages());
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
        let pte3_num = L::MAPPING.num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            self[i].store(Entry3::empty(L::MAPPING.span(2)));
            self.empty.push(self, i);
        }

        // The last one may be cut off
        let max =
            (self.pages() - (pte3_num - 1) * L::MAPPING.span(2)).min(L::MAPPING.span(2));
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
        let pte3_num = L::MAPPING.num_pts(2, self.pages());
        for i in 0..pte3_num {
            self[i].store(Entry3::new().with_next(None));
        }
        // Clear the lists
        self.empty.set(Entry3::new().with_next(None));
        self.partial.set(Entry3::new().with_next(None));

        if !self.meta.is_null() {
            let meta = unsafe { &mut *self.meta };
            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
            meta.active.store(1, Ordering::SeqCst);
        }
        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order > L::MAX_ORDER {
            error!("invalid order: !{order} <= {}", L::MAX_ORDER);
            return Err(Error::Memory);
        }

        // Select local data (which can be shared between cores if we do not have enough memory)
        let c = core % self.local.len();
        let local = &self.local[c];

        // Incremet or clear (atomic sync with put dec)
        let mut pte_res = local.pte.update(|v| v.dec(1 << order));

        'outer: for _ in 0..CAS_RETRIES {
            let (pte, mut start) = match pte_res {
                Ok(pte) => (pte, local.start.load()),
                Err(pte) => {
                    // Try reserving new subtree
                    if let Some(ret) = self.get_reserve(local, pte, order)? {
                        ret
                    } else {
                        pte_res = local.pte.update(|v| v.dec(1 << order));
                        continue 'outer;
                    }
                }
            };

            // Load starting idx
            if start / L::MAPPING.span(2) != pte.idx() {
                start = pte.idx() * L::MAPPING.span(2)
            }

            // Allocate in subtree
            match self.lower.get(start, order) {
                Ok(page) => {
                    // small pages
                    if order < log2(64) {
                        local.start.store(page);
                    }
                    return Ok(unsafe { self.lower.memory().start.add(page as _) } as u64);
                }
                Err(Error::Memory) => {
                    // Counter reset and retry
                    warn!("alloc failed o={order} => retry");
                    let max = self
                        .pages()
                        .saturating_sub(L::MAPPING.round(2, start))
                        .min(L::MAPPING.span(2));
                    // Increment global to prevent race condition with concurrent reservation
                    match self[pte.idx()].update(|v| v.inc(1 << order, max)) {
                        Ok(pte) => pte_res = Err(pte), // -> reserve new
                        Err(pte) => {
                            error!("Counter reset failed o={order} {}: {pte:?}", pte.idx());
                            return Err(Error::Corruption);
                        }
                    }
                    continue 'outer;
                }
                Err(e) => return Err(e),
            }
        }

        error!("Exceeding retries");
        Err(Error::Memory)
    }

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

        let i = page / L::MAPPING.span(2);
        // Save the modified subtree id for the push-reserve heuristic
        let c = core % self.local.len();
        let local = &self.local[c];

        let max = self
            .pages()
            .saturating_sub(L::MAPPING.round(2, page))
            .min(L::MAPPING.span(2));

        // Try decrement own subtree first
        if let Err(pte) = local.pte.update(|v| v.inc_idx(num_pages, i, max)) {
            if pte.idx() == i {
                error!("inc failed L{i}: {pte:?} o={order}");
                return Err(Error::Corruption);
            }
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
                    // Try to reserve subtree if recent frees were part of it
                    if core == c && local.frees_related(i) {
                        if self.put_reserve(local, i)? {
                            return Ok(());
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

    fn pages_needed(&self, cores: usize) -> usize {
        L::MAPPING.span(2) * cores
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_for_each_huge_page(&self, f: fn(usize)) {
        self.lower.dbg_for_each_huge_page(f)
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..L::MAPPING.num_pts(2, self.pages()) {
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

impl<L: LowerAlloc> Drop for ArrayAtomic<L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}
impl<L: LowerAlloc> Default for ArrayAtomic<L> {
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

impl<L: LowerAlloc> ArrayAtomic<L> {
    const ALMOST_FULL: usize = 1 << L::MAX_ORDER;

    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover_inner(&self, deep: bool) -> Result<usize> {
        if deep {
            warn!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..L::MAPPING.num_pts(2, self.pages()) {
            let page = i * L::MAPPING.span(2);
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
    fn reserve(&self, order: usize) -> Result<(Entry3, usize)> {
        while let Some((i, r)) = self.partial.pop_update(self, |v| {
            v.reserve_partial(Self::ALMOST_FULL, L::MAPPING.span(2))
        }) {
            info!("reserve partial {i}");
            match r {
                Ok(pte) => {
                    let pte = pte.with_idx(i).dec(1 << order).unwrap();
                    return Ok((pte, i * L::MAPPING.span(2)));
                }
                Err(pte) => {
                    // Skip empty entries
                    if !pte.reserved() && pte.free() == L::MAPPING.span(2) {
                        self.empty.push(self, i);
                    }
                }
            }
        }

        while let Some((i, r)) = self
            .empty
            .pop_update(self, |v| v.reserve_empty(L::MAPPING.span(2)))
        {
            info!("reserve empty {i}");
            if let Ok(pte) = r {
                let pte = pte.with_idx(i).dec(1 << order).unwrap();
                return Ok((pte, i * L::MAPPING.span(2)));
            }
        }

        error!("reserve: no memory");
        Err(Error::Memory)
    }

    fn enqueue(&self, i: usize, new_pages: usize) {
        // Add to list if new counter is small enough
        if new_pages == L::MAPPING.span(2) {
            self.empty.push(self, i);
        } else if new_pages > Self::ALMOST_FULL {
            self.partial.push(self, i);
        }
    }

    /// Swap the current reserved subtree out replacing it with a new one.
    /// The old subtree is unreserved and added back to the lists.
    fn cas_reserved(
        &self,
        pte_a: &Atomic<Entry3>,
        expect_reserved: bool,
        new_pte: Entry3,
    ) -> Result<()> {
        debug_assert!(!new_pte.reserved());

        let pte = pte_a
            .update(|v| (v.reserved() == expect_reserved).then_some(new_pte))
            .map_err(|_| Error::CAS)?;
        let i = pte.idx();
        if i >= Entry3::IDX_MAX {
            return Ok(());
        }

        let max = (self.pages() - i * L::MAPPING.span(2)).min(L::MAPPING.span(2));
        if let Ok(v) = self[i].update(|v| v.unreserve_add(pte.free(), max)) {
            // Only if not already in list
            if !v.is_valid() {
                // Add to list
                self.enqueue(i, v.free() + pte.free());
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Try to reserve a new subtree or wait for concurrent reservations to finish
    fn get_reserve(
        &self,
        local: &Local<PUTS_RESERVE>,
        pte: Entry3,
        order: usize,
    ) -> Result<Option<(Entry3, usize)>> {
        // Set the reserved flag, locking the reservation
        if !pte.reserved()
            && local
                .pte
                .update(|v| (!v.reserved()).then_some(v.with_reserved(true)))
                .is_ok()
        {
            // Try reserve new subtree
            let (new_pte, start) = match self.reserve(order) {
                Ok(ret) => ret,
                Err(e) => {
                    // Clear reserve flag
                    if local
                        .pte
                        .update(|v| v.reserved().then_some(v.with_reserved(false)))
                        .is_err()
                    {
                        error!("unexpected reserve state");
                        return Err(Error::Corruption);
                    }
                    return Err(e);
                }
            };
            match self.cas_reserved(&local.pte, true, new_pte) {
                Ok(_) => {}
                Err(Error::CAS) => {
                    error!("unexpected reserve state");
                    return Err(Error::Corruption);
                }
                Err(e) => return Err(e),
            }
            local.start.store(start);
            Ok(Some((new_pte, start)))
        } else {
            // Wait for reservation to end
            for _ in 0..(2 * CAS_RETRIES) {
                if local.pte.load().reserved() {
                    hint::spin_loop()
                } else {
                    return Ok(None);
                }
            }
            error!("Timeout reservation wait");
            return Err(Error::Corruption);
        }
    }

    fn put_reserve(&self, local: &Local<PUTS_RESERVE>, i: usize) -> Result<bool> {
        // Try to reserve it for bulk frees
        if let Ok(new_pte) = self[i].update(|v| v.reserve(Self::ALMOST_FULL)) {
            match self.cas_reserved(&local.pte, false, new_pte.with_idx(i)) {
                Ok(_) => {
                    local.start.store(i * L::MAPPING.span(2));
                    Ok(true)
                }
                Err(Error::CAS) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.pages() - i * L::MAPPING.span(2)).min(L::MAPPING.span(2));
                    if let Err(_) = self[i].update(|v| v.unreserve_add(new_pte.free(), max)) {
                        error!("put - reservation rollback failed");
                        return Err(Error::Corruption);
                    }
                    Ok(false)
                }
                Err(e) => Err(e),
            }
        } else {
            Ok(false)
        }
    }
}