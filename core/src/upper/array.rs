use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, hint};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use super::{Alloc, Local, MAGIC, MAX_PAGES};
use crate::atomic::{ANode, Atomic, Next};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
use crate::upper::CAS_RETRIES;
use crate::util::{align_down, Page};
use crate::{Error, Result};

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// This allocator splits its memory range into chunks.
/// Giant pages are directly allocated in it.
/// For smaller pages, however, the chunk is handed over to the
/// lower allocator, managing these smaller allocations.
/// These chunks are, due to the inner workins of the lower allocator,
/// called *subtrees*.
///
/// This allocator stores the level three entries (subtree roots) in a
/// packed array.
/// For the reservation, the allocator simply scans the array for free entries,
/// while prioritizing partially empty chunks.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[repr(align(64))]
pub struct Array<const PR: usize, L: LowerAlloc>
where
    [(); L::N]:,
{
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// CPU local data (only shared between CPUs if the memory area is too small)
    local: Box<[Local<PR>]>,
    /// Metadata of the lower alloc
    lower: L,
    /// Manages the allocators subtrees
    trees: Trees<{ L::N }>,
}

unsafe impl<const PR: usize, L: LowerAlloc> Send for Array<PR, L> where [(); L::N]: {}
unsafe impl<const PR: usize, L: LowerAlloc> Sync for Array<PR, L> where [(); L::N]: {}

impl<const PR: usize, L: LowerAlloc> fmt::Debug for Array<PR, L>
where
    [(); L::N]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;

        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;

        writeln!(f, "    subtrees: {:?} ({} pages)", self.trees, L::N)?;
        let free_pages = self.dbg_free_pages();
        let free_huge_pages = self.dbg_free_huge_pages();
        writeln!(
            f,
            "    free pages: {free_pages} ({free_huge_pages} huge, {} trees)",
            free_pages.div_ceil(L::N)
        )?;

        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.pte.load())?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

impl<const PR: usize, L: LowerAlloc> Alloc for Array<PR, L>
where
    [(); L::N]:,
{
    #[cold]
    fn init(&mut self, mut cores: usize, mut memory: &mut [Page], persistent: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < L::N * cores {
            warn!("memory {} < {}", memory.len(), L::N * cores);
            cores = 1.max(memory.len() / L::N);
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
        let pte3_num = self.pages().div_ceil(L::N);
        self.trees.init(pte3_num);

        Ok(())
    }

    fn recover(&self) -> Result<()> {
        if let Some(meta) = unsafe { self.meta.as_ref() } {
            if meta.pages.load(Ordering::SeqCst) == self.pages()
                && meta.magic.load(Ordering::SeqCst) == MAGIC
            {
                info!("recover p={}", self.pages());
                let deep = meta.active.load(Ordering::SeqCst) != 0;
                self.recover_inner(deep)?;
                meta.active.store(1, Ordering::SeqCst);
                Ok(())
            } else {
                error!("No metadata found");
                Err(Error::Initialization)
            }
        } else {
            error!("Allocator not persistent");
            Err(Error::Initialization)
        }
    }

    fn free_all(&self) -> Result<()> {
        info!("free all p={}", self.pages());
        self.lower.free_all();

        // Add all entries to the empty list
        let pte3_num = self.pages().div_ceil(L::N);
        for i in 0..pte3_num - 1 {
            self.trees[i].store(Entry3::empty(L::N).with_next(Next::Outside));
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * L::N).min(L::N);
        self.trees[pte3_num - 1].store(Entry3::new().with_free(max).with_next(Next::Outside));

        if let Some(meta) = unsafe { self.meta.as_ref() } {
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
        self.trees.clear();

        if let Some(meta) = unsafe { self.meta.as_ref() } {
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

        for _ in 0..CAS_RETRIES {
            match self.get_inner(core, order) {
                Err(Error::CAS) => continue,
                Ok(addr) => return Ok(addr),
                Err(e) => return Err(e),
            }
        }

        error!("Exceeding retries");
        Err(Error::Memory)
    }

    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()> {
        let page = self.addr_to_page(addr, order)?;

        self.lower.put(page, order)?;

        let i = page / L::N;
        // Save the modified subtree id for the push-reserve heuristic
        let c = core % self.local.len();
        let local = &self.local[c];

        let max = (self.pages() - i * L::N).min(L::N);

        // Try decrement own subtree first
        let num_pages = 1 << order;
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
        match self.trees[i].update(|v| v.inc(num_pages, max)) {
            Ok(pte) => {
                let new_pages = pte.free() + num_pages;
                if !pte.reserved() && new_pages > Trees::<{ L::N }>::almost_full() {
                    // put-reserve optimization:
                    // Try to reserve the subtree that was targeted by the recent frees
                    if core == c && local.frees_related(i) && self.reserve_entry(&local.pte, i)? {
                        return Ok(());
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

    fn is_free(&self, addr: u64, order: usize) -> bool {
        if let Ok(page) = self.addr_to_page(addr, order) {
            self.lower.is_free(page, order)
        } else {
            false
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_free_pages(&self) -> usize {
        let mut pages = 0;
        for i in 0..self.pages().div_ceil(L::N) {
            let pte = self.trees[i].load();
            pages += pte.free();
        }
        // Pages allocated in reserved subtrees
        for local in self.local.iter() {
            pages += local.pte.load().free();
        }
        pages
    }

    #[cold]
    fn dbg_free_huge_pages(&self) -> usize {
        let mut counter = 0;
        self.lower.dbg_for_each_huge_page(|c| {
            if c == (1 << L::HUGE_ORDER) {
                counter += 1;
            }
        });
        counter
    }

    #[cold]
    fn dbg_for_each_huge_page(&self, f: fn(usize)) {
        self.lower.dbg_for_each_huge_page(f)
    }
}

impl<const PR: usize, L: LowerAlloc> Drop for Array<PR, L>
where
    [(); L::N]:,
{
    fn drop(&mut self) {
        if let Some(meta) = unsafe { self.meta.as_mut() } {
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}
impl<const PR: usize, L: LowerAlloc> Default for Array<PR, L>
where
    [(); L::N]:,
{
    fn default() -> Self {
        Self {
            meta: null_mut(),
            trees: Default::default(),
            local: Default::default(),
            lower: Default::default(),
        }
    }
}

impl<const PR: usize, L: LowerAlloc> Array<PR, L>
where
    [(); L::N]:,
{
    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover_inner(&self, deep: bool) -> Result<usize> {
        if deep {
            warn!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..self.pages().div_ceil(L::N) {
            let page = i * L::N;
            let pages = self.lower.recover(page, deep)?;

            self.trees[i].store(Entry3::new_table(pages, false));

            total += pages;
        }
        Ok(total)
    }

    fn addr_to_page(&self, addr: u64, order: usize) -> Result<usize> {
        if order > L::MAX_ORDER {
            error!("invalid order: {order} > {}", L::MAX_ORDER);
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
        Ok(page)
    }

    fn get_inner(&self, core: usize, order: usize) -> Result<u64> {
        // Select local data (which can be shared between cores if we do not have enough memory)
        let c = core % self.local.len();
        let local = &self.local[c];

        match local.pte.update(|v| v.dec(1 << order)) {
            Ok(pte) => {
                let mut start = local.start.load();
                if start / L::N != pte.idx() {
                    start = pte.idx() * L::N
                }
                match self.lower.get(start, order) {
                    Ok(page) => {
                        if order < 64usize.ilog2() as usize {
                            local.start.store(page);
                        }
                        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
                    }
                    Err(Error::Memory) => {
                        // counter reset
                        info!("alloc failed o={order} => retry");
                        let max = (self.pages() - align_down(start, L::N)).min(L::N);
                        // Increment global to prevent race condition with concurrent reservation
                        if let Err(pte) = self.trees[pte.idx()].update(|v| v.inc(1 << order, max)) {
                            error!("Counter reset failed o={order} {pte:?}");
                            Err(Error::Corruption)
                        } else {
                            // reserve new, pushing the old entry to the end of the partial list
                            self.reserve_or_wait(&local.pte, pte, true)?;
                            Err(Error::CAS)
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            Err(pte) => {
                // Try sync with global
                self.try_sync_with_global(&local.pte, pte)?;

                // reserve new
                self.reserve_or_wait(&local.pte, pte, false)?;
                Err(Error::CAS)
            }
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// If successful returns `Error::CAS` -> retry.
    /// Returns Ok if the global counter was not large enough -> fallback to normal reservation.
    fn try_sync_with_global(&self, pte_a: &Atomic<Entry3>, old: Entry3) -> Result<()> {
        let i = old.idx();
        if i < self.trees.entries.len()
            && old.free() + self.trees[i].load().free() > Trees::<{ L::N }>::almost_full()
        {
            if let Ok(pte) = self.trees[i].update(|e| e.reserved().then_some(e.with_free(0))) {
                if pte_a
                    .update(|e| (e.idx() == i).then_some(e.with_free(e.free() + pte.free())))
                    .is_ok()
                {
                    // Sync successfull -> retry allocation
                    return Err(Error::CAS);
                } else {
                    // undo global change
                    if self.trees[i]
                        .update(|e| Some(e.with_free(e.free() + pte.free())))
                        .is_err()
                    {
                        error!("Failed undo sync");
                        return Err(Error::Corruption);
                    }
                }
            }
        }
        Ok(())
    }

    /// Try to reserve a new subtree or wait for concurrent reservations to finish.
    ///
    /// If `retry`, tries to reserve a less fragmented subtree
    fn reserve_or_wait(&self, pte_a: &Atomic<Entry3>, old: Entry3, retry: bool) -> Result<()> {
        // Set the reserved flag, locking the reservation
        if !old.reserved() && pte_a.update(|v| v.toggle_reserve(true)).is_ok() {
            // Try reserve new subtree
            let start = if old.has_idx() { old.idx() } else { 0 };
            let new_pte = match self.trees.reserve(start, retry) {
                Ok(pte) => pte,
                Err(e) => {
                    // Clear reserve flag
                    if pte_a.update(|v| v.toggle_reserve(false)).is_err() {
                        error!("unexpected reserve state");
                        return Err(Error::Corruption);
                    }
                    return Err(e);
                }
            };
            match self.cas_reserved(pte_a, new_pte, true) {
                Ok(_) => Ok(()),
                Err(Error::CAS) => {
                    error!("unexpected reserve state");
                    Err(Error::Corruption)
                }
                Err(e) => Err(e),
            }
        } else {
            // Wait for concurrent reservation to end
            for _ in 0..(2 * CAS_RETRIES) {
                let new_pte = pte_a.load();
                if !new_pte.reserved() {
                    return Ok(());
                }
                hint::spin_loop(); // pause cpu
            }
            error!("Timeout reservation wait");
            Err(Error::Corruption)
        }
    }

    fn reserve_entry(&self, pte_a: &Atomic<Entry3>, i: usize) -> Result<bool> {
        // Try to reserve it for bulk frees
        if let Ok(new_pte) =
            self.trees[i].update(|v| v.reserve_min(Trees::<{ L::N }>::almost_full()))
        {
            match self.cas_reserved(pte_a, new_pte.with_idx(i), false) {
                Ok(_) => Ok(true),
                Err(Error::CAS) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.pages() - i * L::N).min(L::N);
                    if self.trees[i]
                        .update(|v| v.unreserve_add(new_pte.free(), max))
                        .is_err()
                    {
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

    /// Swap the current reserved subtree out replacing it with a new one.
    /// The old subtree is unreserved and added back to the lists.
    ///
    /// If `enqueue_back`, the old unreserved entry is added to the back of the partial list.
    fn cas_reserved(
        &self,
        pte_a: &Atomic<Entry3>,
        new_pte: Entry3,
        expect_reserved: bool,
    ) -> Result<()> {
        debug_assert!(!new_pte.reserved());

        let pte = pte_a
            .update(|v| (v.reserved() == expect_reserved).then_some(new_pte))
            .map_err(|_| Error::CAS)?;

        self.trees.unreserve(pte, self.pages())
    }
}

#[derive(Default)]
struct Trees<const LN: usize> {
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    entries: Box<[Atomic<Entry3>]>,
}

impl<const LN: usize> Index<usize> for Trees<LN> {
    type Output = Atomic<Entry3>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl<const LN: usize> fmt::Debug for Trees<LN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut empty = 0;
        let mut partial = 0;
        for e in &*self.entries {
            let free = e.load().free();
            if free == LN {
                empty += 1;
            } else if free > Self::almost_full() {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, empty: {empty}, partial: {partial})")?;
        Ok(())
    }
}

impl<const LN: usize> Trees<LN> {
    fn init(&mut self, pte3_num: usize) {
        let mut pte3s = Vec::with_capacity(pte3_num);
        pte3s.resize_with(pte3_num, || Atomic::new(Entry3::new()));

        self.entries = pte3s.into();
    }

    /// Almost no free pages left
    const fn almost_full() -> usize {
        1 << 10 // MAX_ORDER
    }

    /// Almost all pages are free
    const fn almost_empty() -> usize {
        LN - (1 << 10) // MAX_ORDER
    }

    fn clear(&self) {
        // Set all entries to zero
        for pte in &self.entries[..] {
            pte.store(Entry3::new().with_next(Next::Outside));
        }
    }

    fn unreserve(&self, pte: Entry3, pages: usize) -> Result<()> {
        let i = pte.idx();
        if i >= Entry3::IDX_MAX {
            return Ok(());
        }

        let max = (pages - i * LN).min(LN);
        if let Ok(_) = self[i].update(|v| v.unreserve_add(pte.free(), max)) {
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Find and reserve an empty tree
    fn reserve_empty(&self, start: usize) -> Result<Entry3> {
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(pte) = self[i].update(|v| v.reserve_min(Self::almost_empty())) {
                return Ok(pte.with_idx(i));
            }
        }
        warn!("no empty tree {self:?}");
        Err(Error::Memory)
    }

    /// Find and reserve a partially filled tree in the vicinity
    fn reserve_partial(&self, start: usize) -> Result<Entry3> {
        // rechecking previous entries reduces fragmentation
        //  -> start with the previous cache line
        let start = align_down(
            start + self.entries.len().saturating_sub(self.entries.len() / 32),
            8,
        );

        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(pte) =
                self[i].update(|v| v.reserve_partial(Self::almost_full()..Self::almost_empty()))
            {
                return Ok(pte.with_idx(i));
            }
        }
        Err(Error::Memory)
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees.
    fn reserve(&self, start: usize, prioritize_empty: bool) -> Result<Entry3> {
        info!("reserve prio={prioritize_empty}");
        if prioritize_empty {
            match self.reserve_empty(start) {
                Err(Error::Memory) => self.reserve_partial(start),
                r => r,
            }
        } else {
            match self.reserve_partial(start) {
                Err(Error::Memory) => self.reserve_empty(start),
                r => r,
            }
        }
    }
}
