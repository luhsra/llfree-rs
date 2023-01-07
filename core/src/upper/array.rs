use core::fmt;
use core::mem::{size_of, align_of};
use core::ops::Index;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crossbeam_utils::atomic::AtomicCell;
use log::{error, info, warn};

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::{Alloc, Init, Local, MAGIC, MAX_PAGES};
use crate::entry::{ReservedTree, Tree};
use crate::lower::LowerAlloc;
use crate::upper::CAS_RETRIES;
use crate::util::{align_down, spin_wait, CacheLine, Page};
use crate::{Error, Result};

/// Non-Volatile global metadata
#[repr(align(0x1000))]
struct Meta {
    /// A magic number used to check if the persistent memory contains the allocator state
    magic: AtomicUsize,
    /// Number of pages managed by the persistent allocator
    pages: AtomicUsize,
    /// Flag that stores if the system has crashed or was shutdown correctly
    crashed: AtomicBool,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// This allocator splits its memory range into chunks.
/// These chunks are reserved by CPUs to reduce sharing.
/// Allocations/frees within the chunk are handed over to the
/// lower allocator.
/// These chunks are, due to the inner workins of the lower allocator,
/// called *subtrees*.
///
/// This allocator stores the level three entries (subtree roots) in a
/// packed array.
/// For the reservation, the allocator simply scans the array for free entries,
/// while prioritizing partially empty chunks to avoid fragmentation.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[repr(align(64))]
pub struct Array<const F: usize, L: LowerAlloc>
where
    [(); L::N]:,
{
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: Option<&'static Meta>,
    /// CPU local data (only shared between CPUs if the memory area is too small)
    local: Box<[Local<F>]>,
    /// Metadata of the lower alloc
    lower: L,
    /// Manages the allocators subtrees
    trees: Trees<{ L::N }>,
}

unsafe impl<const F: usize, L: LowerAlloc> Send for Array<F, L> where [(); L::N]: {}
unsafe impl<const F: usize, L: LowerAlloc> Sync for Array<F, L> where [(); L::N]: {}

impl<const F: usize, L: LowerAlloc> fmt::Debug for Array<F, L>
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
            writeln!(f, "    L{t:>2}: {:?}", local.reserved.load())?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

impl<const F: usize, L: LowerAlloc> Alloc for Array<F, L>
where
    [(); L::N]:,
{
    #[cold]
    fn init(
        &mut self,
        mut cores: usize,
        mut memory: &mut [Page],
        init: Init,
        free_all: bool,
    ) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < L::N * cores {
            warn!("memory {} < {}", memory.len(), L::N * cores);
            if memory.len() < 1 << L::HUGE_ORDER {
                error!("Expecting at least {} pages", 1 << L::HUGE_ORDER);
                return Err(Error::Memory);
            }
            cores = 1.max(memory.len() / L::N);
        }

        if init != Init::Volatile {
            // Last frame is reserved for metadata
            let (m, rem) = memory.split_at_mut((memory.len() - 1).min(MAX_PAGES));
            self.meta = Some(unsafe { &*rem.as_ptr().cast::<Meta>() });
            memory = m;
        }

        // Init per-cpu data
        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Local::new);
        self.local = local.into();

        // Create lower allocator
        self.lower = L::new(cores, memory, init, free_all);

        if init == Init::Recover {
            match self.recover() {
                // If the recovery fails, continue with initializing a new allocator instead
                Err(Error::Initialization) => {}
                r => return r,
            }
        }

        self.trees.init(self.pages(), free_all);

        if let Some(meta) = self.meta {
            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
            meta.crashed.store(true, Ordering::SeqCst);
        }

        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order > L::MAX_ORDER {
            error!("invalid order: !{order} <= {}", L::MAX_ORDER);
            return Err(Error::Memory);
        }

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..CAS_RETRIES {
            match self.get_inner(core, order) {
                Err(Error::Retry) => continue,
                Ok(addr) => return Ok(addr),
                Err(e) => return Err(e),
            }
        }

        error!("Exceeding retries");
        Err(Error::Memory)
    }

    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()> {
        let page = self.addr_to_page(addr, order)?;

        // First free the page in the lower allocator
        self.lower.put(page, order)?;

        // Then update local / global counters
        let i = page / L::N;
        let local = &self.local[core % self.local.len()];
        let max = (self.pages() - i * L::N).min(L::N);

        // Try update own tree first
        let num_pages = 1 << order;
        if let Err(tree) = local
            .reserved
            .fetch_update(|v| v.inc(num_pages, max, |s| s / L::N == i))
        {
            if tree.start() / L::N == i {
                error!("inc failed L{i}: {tree:?} o={order}");
                return Err(Error::Corruption);
            }
        } else {
            // Save the modified tree id for the push-reserve heuristic
            local.frees_push(i);
            return Ok(());
        };

        // Tree not owned by us -> update global
        match self.trees[i].fetch_update(|v| v.inc(num_pages, max)) {
            Ok(tree) => {
                let new_pages = tree.free() + num_pages;
                if !tree.reserved() && new_pages > Trees::<{ L::N }>::almost_allocated() {
                    // put-reserve optimization:
                    // Try to reserve the tree that was targeted by the recent frees
                    if local.frees_in_tree(i) && self.reserve_entry(&local.reserved, i)? {
                        return Ok(());
                    }
                }
                local.frees_push(i);
                Ok(())
            }
            Err(tree) => {
                error!("inc failed i{i}: {tree:?} o={order}");
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

    fn drain(&self, core: usize) -> Result<()> {
        let local = &self.local[core % self.local.len()];
        match self.cas_reserved(&local.reserved, ReservedTree::default(), false) {
            Err(Error::Retry) => Ok(()), // ignore cas errors
            r => r,
        }
    }

    fn dbg_free_pages(&self) -> usize {
        let mut pages = 0;
        // Global array
        for tree in self.trees.entries.iter() {
            pages += tree.load().free();
        }
        // Pages allocated in reserved subtrees
        for local in self.local.iter() {
            pages += local.reserved.load().free();
        }
        pages
    }

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

impl<const F: usize, L: LowerAlloc> Drop for Array<F, L>
where
    [(); L::N]:,
{
    fn drop(&mut self) {
        if let Some(meta) = self.meta {
            meta.crashed.store(false, Ordering::SeqCst);
        }
    }
}
impl<const F: usize, L: LowerAlloc> Default for Array<F, L>
where
    [(); L::N]:,
{
    fn default() -> Self {
        Self {
            meta: None,
            trees: Default::default(),
            local: Default::default(),
            lower: Default::default(),
        }
    }
}

impl<const F: usize, L: LowerAlloc> Array<F, L>
where
    [(); L::N]:,
{
    /// Recover the allocator from NVM after reboot.
    /// If crashed then the level 1 page tables are traversed and diverging counters are corrected.
    fn recover(&mut self) -> Result<()> {
        if let Some(meta) = self.meta {
            if meta.pages.load(Ordering::SeqCst) == self.pages()
                && meta.magic.load(Ordering::SeqCst) == MAGIC
            {
                info!("recover p={}", self.pages());
                // The active flag is set on boot and reset on a successful shutdown
                // If it is already set, the allocator has been crashed
                // In this case, we have to initiate a deep recovery, correcting all the counters
                let deep = meta.crashed.load(Ordering::SeqCst);
                if deep {
                    warn!("Try recover crashed allocator!");
                }

                let mut trees = Vec::with_capacity(self.pages().div_ceil(L::N));
                // Recover each subtree one-by-one
                for i in 0..self.pages().div_ceil(L::N) {
                    let page = i * L::N;
                    let pages = self.lower.recover(page, deep)?;
                    trees.push(AtomicCell::new(Tree::new_with(pages, false)));
                }
                self.trees.entries = trees.into();

                meta.crashed.store(true, Ordering::SeqCst);
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

    /// Convert an address to the page index
    fn addr_to_page(&self, addr: u64, order: usize) -> Result<usize> {
        if order > L::MAX_ORDER {
            error!("invalid order: {order} > {}", L::MAX_ORDER);
            return Err(Error::Memory);
        }

        // Check alignment and if this addr is within our address range
        if addr % ((1 << order) * Page::SIZE) as u64 != 0
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

    /// Try to allocate a page with the given order
    fn get_inner(&self, core: usize, order: usize) -> Result<u64> {
        // Select local data (which can be shared between cores if we do not have enough memory)
        let c = core % self.local.len();
        let local = &self.local[c];
        // Update the upper counters first
        match local.reserved.fetch_update(|v| v.dec(1 << order)) {
            Ok(old) => {
                // The start point for the search
                let start = old.start();
                // Try allocating with the lower allocator
                match self.lower.get(start, order) {
                    Ok(page) => {
                        // Success
                        if order < 64usize.ilog2() as usize
                            && align_down(start, 64) != align_down(page, 64)
                        {
                            // Save start index for lower allocations
                            let new = old.dec(1 << order).unwrap();
                            let ret = local.reserved.compare_exchange(new, new.with_start(page));
                            debug_assert!(ret.is_ok());
                        }
                        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
                    }
                    Err(Error::Memory) => {
                        // Failure (e.g. due to fragmentation)
                        // Reset counters, reserve new entry and retry allocation
                        info!("alloc failed o={order} => retry");
                        let max = (self.pages() - align_down(start, L::N)).min(L::N);
                        // Increment global to prevent race condition with concurrent reservation
                        if let Err(old) =
                            self.trees[start / L::N].fetch_update(|v| v.inc(1 << order, max))
                        {
                            error!("Counter reset failed o={order} {old:?}");
                            Err(Error::Corruption)
                        } else {
                            self.reserve_or_wait(core, order, &local.reserved, old, true)?;
                            Err(Error::Retry)
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            Err(old) => {
                // If the local counter is large enough we do not have to reserve a new subtree
                // Just update the local counter and reuse the current subtree
                self.try_sync_with_global(&local.reserved, old)?;

                // The local subtree is full -> reserve a new one
                self.reserve_or_wait(core, order, &local.reserved, old, false)?;

                // TODO: Steal from other CPUs on Error::Memory
                // Stealing in general should not only be done after the whole array has been searched,
                // due to the terrible performance.
                // We probably need a stealing mode where a CPU steals the next N pages from another CPU.

                // Reservation successfull -> retry the allocation
                Err(Error::Retry)
            }
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// If successful returns `Error::CAS` -> retry.
    /// Returns Ok if the global counter was not large enough -> fallback to normal reservation.
    fn try_sync_with_global(
        &self,
        local: &AtomicCell<ReservedTree>,
        old: ReservedTree,
    ) -> Result<()> {
        let i = old.start() / L::N;
        if i < self.trees.entries.len()
            && old.free() + self.trees[i].load().free() > Trees::<{ L::N }>::almost_allocated()
        {
            if let Ok(entry) =
                self.trees[i].fetch_update(|e| e.reserved().then_some(e.with_free(0)))
            {
                if local
                    .fetch_update(|e| {
                        (e.start() / L::N == i).then_some(e.with_free(e.free() + entry.free()))
                    })
                    .is_ok()
                {
                    // Sync successfull -> retry allocation
                    return Err(Error::Retry);
                } else {
                    // undo global change
                    if self.trees[i]
                        .fetch_update(|e| Some(e.with_free(e.free() + entry.free())))
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
    fn reserve_or_wait(
        &self,
        core: usize,
        order: usize,
        local: &AtomicCell<ReservedTree>,
        old: ReservedTree,
        retry: bool,
    ) -> Result<()> {
        // Set the reserved flag, locking the reservation
        if !old.locked() && local.fetch_update(|v| v.toggle_locked(true)).is_ok() {
            // Try reserve new subtree
            let start = if old.has_start() {
                old.start() / L::N
            } else {
                // Different initial starting point for every core
                self.trees.entries.len() / self.local.len() * core
                // TODO: Reset start periodically to space CPUs more evenly over the memory zone
            };
            let new = match self.trees.reserve(order, self.local.len(), start, retry) {
                Ok(entry) => entry,
                Err(e) => {
                    // Clear reserve flag
                    if local.fetch_update(|v| v.toggle_locked(false)).is_err() {
                        error!("unexpected reserve state");
                        return Err(Error::Corruption);
                    }
                    return Err(e);
                }
            };
            match self.cas_reserved(local, new, true) {
                Ok(_) => Ok(()),
                Err(Error::Retry) => {
                    error!("unexpected reserve state");
                    Err(Error::Corruption)
                }
                Err(e) => Err(e),
            }
        } else {
            // Wait for concurrent reservation to end
            if spin_wait(2 * CAS_RETRIES, || !local.load().locked()) {
                Ok(())
            } else {
                error!("Timeout reservation wait");
                Err(Error::Corruption)
            }
        }
    }

    // Reserve an entry for bulk frees
    fn reserve_entry(&self, local: &AtomicCell<ReservedTree>, i: usize) -> Result<bool> {
        if let Ok(entry) =
            self.trees[i].fetch_update(|v| v.reserve(Trees::<{ L::N }>::almost_allocated()..))
        {
            let entry = ReservedTree::new_with(entry.free(), i * L::N);
            match self.cas_reserved(local, entry, false) {
                Ok(_) => Ok(true),
                Err(Error::Retry) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.pages() - i * L::N).min(L::N);
                    if self.trees[i]
                        .fetch_update(|v| v.unreserve_add(entry.free(), max))
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
        local: &AtomicCell<ReservedTree>,
        new: ReservedTree,
        expect_reserved: bool,
    ) -> Result<()> {
        debug_assert!(!new.locked());

        let old = local
            .fetch_update(|v| (v.locked() == expect_reserved).then_some(new))
            .map_err(|_| Error::Retry)?;

        self.trees.unreserve(old, self.pages())
    }
}

#[derive(Default)]
struct Trees<const LN: usize> {
    /// Array of level 3 entries, which are the roots of the subtrees
    entries: Box<[AtomicCell<Tree>]>,
}

impl<const LN: usize> Index<usize> for Trees<LN> {
    type Output = AtomicCell<Tree>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl<const LN: usize> fmt::Debug for Trees<LN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut free = 0;
        let mut partial = 0;
        for e in &*self.entries {
            let f = e.load().free();
            if f == LN {
                free += 1;
            } else if f > Self::almost_allocated() {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, free: {free}, partial: {partial})")?;
        Ok(())
    }
}

impl<const LN: usize> Trees<LN> {
    /// Initialize the subtree array
    fn init(&mut self, pages: usize, free_all: bool) {
        let len = pages.div_ceil(LN);
        let mut entries = Vec::with_capacity(len);
        if free_all {
            entries.resize_with(len - 1, || AtomicCell::new(Tree::new_with(LN, false)));
            // The last one might be cut off
            let max = ((pages - 1) % LN) + 1;
            entries.push(AtomicCell::new(Tree::new_with(max, false)));
        } else {
            entries.resize_with(len, || AtomicCell::new(Tree::new()));
        }
        self.entries = entries.into();
    }

    /// Almost no free pages left
    const fn almost_allocated() -> usize {
        1 << 10 // MAX_ORDER
    }

    /// Almost all pages are free
    const fn almost_free() -> usize {
        LN - (1 << 10) // MAX_ORDER
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    fn unreserve(&self, entry: ReservedTree, pages: usize) -> Result<()> {
        if !entry.has_start() {
            return Ok(());
        }

        let i = entry.start() / LN;
        let max = (pages - i * LN).min(LN);
        if let Ok(_) = self[i].fetch_update(|v| v.unreserve_add(entry.free(), max)) {
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Find and reserve a free tree
    fn reserve_free(&self, start: usize) -> Option<ReservedTree> {
        // Just search linearly through the array
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(Self::almost_free()..)) {
                return Some(ReservedTree::new_with(entry.free(), i * LN));
            }
        }
        warn!("no full tree");
        None
    }

    /// Find and reserve a partial tree in the vicinity
    fn reserve_partial(&self, cores: usize, start: usize) -> Option<ReservedTree> {
        const ENTRIES_PER_CACHELINE: usize = align_of::<CacheLine>() / size_of::<Tree>();
        // One quater of the per-CPU memory
        let vicinity = ((self.entries.len() / cores) / 4).max(1) as isize;

        // Positive modulo and cacheline alignment
        let start = align_down(start + self.entries.len(), ENTRIES_PER_CACHELINE) as isize;

        // Search the the array for a partially or entirely free subtree
        // This speeds up the search drastically if many subtrees are free
        for i in 1..vicinity {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(Self::almost_allocated()..)) {
                return Some(ReservedTree::new_with(entry.free(), i * LN));
            }
        }

        // Search the rest of the array for a partially but not entirely free subtree
        for i in vicinity..=self.entries.len() as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) =
                self[i].fetch_update(|v| v.reserve(Self::almost_allocated()..Self::almost_free()))
            {
                return Some(ReservedTree::new_with(entry.free(), i * LN));
            }
        }
        None
    }

    /// Fallback to search for any suitable subtree
    fn reserve_any(&self, start: usize, order: usize) -> Option<ReservedTree> {
        // Just search linearly through the array
        let num_pages = 1 << order;
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(num_pages..)) {
                return Some(ReservedTree::new_with(entry.free(), i * LN));
            }
        }
        warn!("no pages left");
        None
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees.
    fn reserve(
        &self,
        order: usize,
        cores: usize,
        start: usize,
        prioritize_free: bool,
    ) -> Result<ReservedTree> {
        if prioritize_free {
            // try free subtrees first
            self.reserve_free(start)
                .or_else(|| self.reserve_partial(cores, start))
                .or_else(|| self.reserve_any(start, order))
                .ok_or(Error::Memory)
        } else {
            self.reserve_partial(cores, start)
                .or_else(|| self.reserve_free(start))
                .or_else(|| self.reserve_any(start, order))
                .ok_or(Error::Memory)
        }
    }
}
