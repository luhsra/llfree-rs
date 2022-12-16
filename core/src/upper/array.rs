use core::fmt;
use core::mem::size_of;
use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use super::{Alloc, Init, Local, MAGIC, MAX_PAGES};
use crate::atomic::Atomic;
use crate::entry::{Entry3, SEntry3};
use crate::lower::LowerAlloc;
use crate::upper::CAS_RETRIES;
use crate::util::{align_down, spin_wait, CacheLine, Page};
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
pub struct Array<const F: usize, L: LowerAlloc>
where
    [(); L::N]:,
{
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
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
            writeln!(f, "    L{t:>2}: {:?}", local.entry.load())?;
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
            cores = 1.max(memory.len() / L::N);
        }

        if init != Init::Volatile {
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
        self.lower = L::new(cores, memory, init, free_all);

        if init == Init::Recover {
            match self.recover() {
                Err(Error::Initialization) => {}
                r => return r,
            }
        }

        self.trees.init(self.pages(), free_all);

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
        if let Err(entry) = local.entry.update(|v| v.inc_idx(num_pages, i, max)) {
            if entry.idx() == i {
                error!("inc failed L{i}: {entry:?} o={order}");
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
            Ok(entry) => {
                let new_pages = entry.free() + num_pages;
                if !entry.reserved() && new_pages > Trees::<{ L::N }>::almost_allocated() {
                    // put-reserve optimization:
                    // Try to reserve the subtree that was targeted by the recent frees
                    if core == c && local.frees_related(i) && self.reserve_entry(&local.entry, i)? {
                        return Ok(());
                    }
                }
                if c == core {
                    local.frees_push(i);
                }
                Ok(())
            }
            Err(entry) => {
                error!("inc failed i{i}: {entry:?} o={order}");
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
    fn drain(&self, core: usize) -> Result<()> {
        let c = core % self.local.len();
        let local = &self.local[c];
        match self.cas_reserved(&local.entry, Entry3::new().with_idx(Entry3::IDX_MAX), false) {
            Err(Error::CAS) => Ok(()), // ignore cas errors
            r => r,
        }
    }

    #[cold]
    fn dbg_free_pages(&self) -> usize {
        let mut pages = 0;
        // Global array
        for i in 0..self.pages().div_ceil(L::N) {
            pages += self.trees[i].load().free();
        }
        // Pages allocated in reserved subtrees
        for local in self.local.iter() {
            pages += local.entry.load().free();
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

impl<const F: usize, L: LowerAlloc> Drop for Array<F, L>
where
    [(); L::N]:,
{
    fn drop(&mut self) {
        if let Some(meta) = unsafe { self.meta.as_mut() } {
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}
impl<const F: usize, L: LowerAlloc> Default for Array<F, L>
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

impl<const F: usize, L: LowerAlloc> Array<F, L>
where
    [(); L::N]:,
{
    /// Recover the allocator from NVM after reboot.
    /// If crashed then the level 1 page tables are traversed and diverging counters are corrected.
    fn recover(&mut self) -> Result<()> {
        if let Some(meta) = unsafe { self.meta.as_ref() } {
            if meta.pages.load(Ordering::SeqCst) == self.pages()
                && meta.magic.load(Ordering::SeqCst) == MAGIC
            {
                info!("recover p={}", self.pages());
                let deep = meta.active.load(Ordering::SeqCst) != 0;
                if deep {
                    warn!("Try recover crashed allocator!");
                }

                let mut trees = Vec::with_capacity(self.pages().div_ceil(L::N));

                for i in 0..self.pages().div_ceil(L::N) {
                    let page = i * L::N;
                    let pages = self.lower.recover(page, deep)?;
                    trees.push(Atomic::new(SEntry3::new_table(pages, false)));
                }
                self.trees.entries = trees.into();

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

        match local.entry.update(|v| v.dec(1 << order)) {
            Ok(old) => {
                let mut start = local.start.load();
                if start / L::N != old.idx() {
                    start = old.idx() * L::N
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
                        if let Err(old) = self.trees[old.idx()].update(|v| v.inc(1 << order, max)) {
                            error!("Counter reset failed o={order} {old:?}");
                            Err(Error::Corruption)
                        } else {
                            // reserve new, pushing the old entry to the end of the partial list
                            self.reserve_or_wait(core, &local.entry, old, true)?;
                            Err(Error::CAS)
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            Err(old) => {
                // Try sync with global
                self.try_sync_with_global(&local.entry, old)?;

                // reserve new
                self.reserve_or_wait(core, &local.entry, old, false)?;
                Err(Error::CAS)
            }
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// If successful returns `Error::CAS` -> retry.
    /// Returns Ok if the global counter was not large enough -> fallback to normal reservation.
    fn try_sync_with_global(&self, local: &Atomic<Entry3>, old: Entry3) -> Result<()> {
        let i = old.idx();
        if i < self.trees.entries.len()
            && old.free() + self.trees[i].load().free() > Trees::<{ L::N }>::almost_allocated()
        {
            if let Ok(entry) = self.trees[i].update(|e| e.reserved().then_some(e.with_free(0))) {
                if local
                    .update(|e| (e.idx() == i).then_some(e.with_free(e.free() + entry.free())))
                    .is_ok()
                {
                    // Sync successfull -> retry allocation
                    return Err(Error::CAS);
                } else {
                    // undo global change
                    if self.trees[i]
                        .update(|e| Some(e.with_free(e.free() + entry.free())))
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
        local: &Atomic<Entry3>,
        old: Entry3,
        retry: bool,
    ) -> Result<()> {
        // Set the reserved flag, locking the reservation
        if !old.reserved() && local.update(|v| v.toggle_reserve(true)).is_ok() {
            // Try reserve new subtree
            let start = if old.has_idx() {
                old.idx()
            } else {
                // Different starting point for every core
                self.trees.entries.len() / self.local.len() * core
            };
            let new = match self.trees.reserve(start, retry) {
                Ok(entry) => entry,
                Err(e) => {
                    // Clear reserve flag
                    if local.update(|v| v.toggle_reserve(false)).is_err() {
                        error!("unexpected reserve state");
                        return Err(Error::Corruption);
                    }
                    return Err(e);
                }
            };
            match self.cas_reserved(local, new, true) {
                Ok(_) => Ok(()),
                Err(Error::CAS) => {
                    error!("unexpected reserve state");
                    Err(Error::Corruption)
                }
                Err(e) => Err(e),
            }
        } else {
            // Wait for concurrent reservation to end
            if spin_wait(2 * CAS_RETRIES, || !local.load().reserved()) {
                Ok(())
            } else {
                error!("Timeout reservation wait");
                Err(Error::Corruption)
            }
        }
    }

    fn reserve_entry(&self, local: &Atomic<Entry3>, i: usize) -> Result<bool> {
        // Try to reserve it for bulk frees
        if let Ok(entry) =
            self.trees[i].update(|v| v.reserve(Trees::<{ L::N }>::almost_allocated()..))
        {
            let entry = Entry3::from(entry).with_idx(i);
            match self.cas_reserved(local, entry, false) {
                Ok(_) => Ok(true),
                Err(Error::CAS) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.pages() - i * L::N).min(L::N);
                    if self.trees[i]
                        .update(|v| v.unreserve_add(entry.free(), max))
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
        local: &Atomic<Entry3>,
        new: Entry3,
        expect_reserved: bool,
    ) -> Result<()> {
        debug_assert!(!new.reserved());

        let old = local
            .update(|v| (v.reserved() == expect_reserved).then_some(new))
            .map_err(|_| Error::CAS)?;

        self.trees.unreserve(old, self.pages())
    }
}

#[derive(Default)]
struct Trees<const LN: usize> {
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    entries: Box<[Atomic<SEntry3>]>,
}

impl<const LN: usize> Index<usize> for Trees<LN> {
    type Output = Atomic<SEntry3>;

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
            } else if free > Self::almost_allocated() {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, empty: {empty}, partial: {partial})")?;
        Ok(())
    }
}

impl<const LN: usize> Trees<LN> {
    fn init(&mut self, pages: usize, free_all: bool) {
        let len = pages.div_ceil(LN);
        let mut entries = Vec::with_capacity(len);
        if free_all {
            entries.resize_with(len - 1, || Atomic::new(SEntry3::new_table(LN, false)));
            // The last one might be cut off
            let max = ((pages - 1) % LN) + 1;
            entries.push(Atomic::new(SEntry3::new_table(max, false)));
        } else {
            entries.resize_with(len, || Atomic::new(SEntry3::new()));
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

    fn unreserve(&self, entry: Entry3, pages: usize) -> Result<()> {
        if !entry.has_idx() {
            return Ok(());
        }

        let i = entry.idx();
        let max = (pages - i * LN).min(LN);
        if let Ok(_) = self[i].update(|v| v.unreserve_add(entry.free(), max)) {
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Find and reserve an empty tree
    fn reserve_empty(&self, start: usize) -> Result<Entry3> {
        // Just search linearly through the array
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].update(|v| v.reserve(Self::almost_free()..)) {
                return Ok(Entry3::from(entry).with_idx(i));
            }
        }
        warn!("no empty tree {self:?}");
        Err(Error::Memory)
    }

    /// Find and reserve a partially filled tree in the vicinity
    fn reserve_partial(&self, start: usize) -> Result<Entry3> {
        const ENTRIES_PER_CACHELINE: usize = size_of::<CacheLine>() / size_of::<SEntry3>();
        const VICINITY: usize = ENTRIES_PER_CACHELINE * 2 - 1;

        // Positive modulo and cacheline alignment
        let start = align_down(start + self.entries.len(), ENTRIES_PER_CACHELINE) as isize;

        if VICINITY < self.entries.len() {
            // Find the best subtree in the close neighborhood
            let mut best = None;
            let mut best_free: usize = usize::MAX;
            for i in 1..VICINITY as isize {
                let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
                let i = (start + off) as usize % self.entries.len();
                let free = self[i].load().free();
                if (Self::almost_allocated()..Self::almost_free()).contains(&free)
                    && free < best_free
                {
                    best = Some(i);
                    best_free = free;
                }
            }
            if let Some(i) = best {
                if let Ok(entry) =
                    self[i].update(|v| v.reserve(Self::almost_allocated()..Self::almost_free()))
                {
                    return Ok(Entry3::from(entry).with_idx(i));
                }
            }
        }

        // Search the rest of the array for a partially filled subtrees
        for i in 1..=self.entries.len() as isize {
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) =
                self[i].update(|v| v.reserve(Self::almost_allocated()..Self::almost_free()))
            {
                return Ok(Entry3::from(entry).with_idx(i));
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
