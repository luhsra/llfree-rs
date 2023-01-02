#![allow(deprecated)]

use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::{fmt, hint};

use crossbeam_utils::atomic::AtomicCell;
use log::{error, info, warn};
use spin::Mutex;

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::{Alloc, Init, Local, MAGIC, MAX_PAGES};
use crate::atomic::{ANode, BufferList, Next};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
use crate::upper::CAS_RETRIES;
use crate::util::{align_down, Page};
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
/// The subtree reservation is speed up using free lists for
/// empty and partially empty subtrees.
/// These free lists are implemented as atomic linked lists with their next
/// pointers stored inside the level 3 entries.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[repr(align(64))]
#[deprecated = "Replaced by the improved Array allocator."]
pub struct ArrayList<const PR: usize, L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// CPU local data (only shared between CPUs if the memory area is too small)
    local: Box<[Local<PR>]>,
    /// Metadata of the lower alloc
    lower: L,
    /// Manages the allocators subtrees
    trees: Trees,
}

unsafe impl<const PR: usize, L: LowerAlloc> Send for ArrayList<PR, L> {}
unsafe impl<const PR: usize, L: LowerAlloc> Sync for ArrayList<PR, L> {}

impl<const PR: usize, L: LowerAlloc> fmt::Debug for ArrayList<PR, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;

        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;

        writeln!(f, "    subtrees: {:?}", self.trees)?;
        let free_pages = self.dbg_free_pages();
        writeln!(
            f,
            "    free pages: {free_pages} ({} trees)",
            free_pages.div_ceil(L::N)
        )?;

        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.reserved.load())?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

impl<const PR: usize, L: LowerAlloc> Alloc for ArrayList<PR, L> {
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

        self.trees = Default::default();
        self.trees.init(self.pages(), L::N, free_all);

        if let Some(meta) = unsafe { self.meta.as_ref() } {
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

        self.lower.put(page, order)?;

        let i = page / L::N;
        // Save the modified subtree id for the push-reserve heuristic
        let c = core % self.local.len();
        let local = &self.local[c];

        let max = (self.pages() - i * L::N).min(L::N);

        // Try decrement own subtree first
        let num_pages = 1 << order;
        if let Err(entry) = local.reserved.fetch_update(|v| {
            if v.idx() / L::N == i {
                v.inc(num_pages, max)
            } else {
                None
            }
        }) {
            if entry.idx() / L::N == i {
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
        match self.trees[i].fetch_update(|v| v.inc(num_pages, max)) {
            Ok(entry) => {
                let new_pages = entry.free() + num_pages;
                if !entry.reserved() && new_pages > Trees::almost_full() {
                    // Try to reserve the subtree that was targeted by the recent frees
                    if core == c
                        && local.frees_eq_to(i)
                        && self.reserve_entry(&local.reserved, i)?
                    {
                        return Ok(());
                    }

                    // Add to partially free list
                    // Only if not already in list
                    if entry.next() == Next::Outside {
                        self.trees.push(i, new_pages, L::N);
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
        match self.cas_reserved(
            &local.reserved,
            Entry3::new().with_idx(Entry3::IDX_MAX),
            false,
            false,
        ) {
            Err(Error::Retry) => Ok(()), // ignore cas errors
            r => r,
        }
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
            pages += local.reserved.load().free();
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

impl<const PR: usize, L: LowerAlloc> Drop for ArrayList<PR, L> {
    fn drop(&mut self) {
        if let Some(meta) = unsafe { self.meta.as_mut() } {
            meta.crashed.store(false, Ordering::SeqCst);
        }
    }
}
impl<const PR: usize, L: LowerAlloc> Default for ArrayList<PR, L> {
    fn default() -> Self {
        Self {
            meta: null_mut(),
            trees: Default::default(),
            local: Default::default(),
            lower: Default::default(),
        }
    }
}

impl<const PR: usize, L: LowerAlloc> ArrayList<PR, L> {
    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    fn recover(&mut self) -> Result<()> {
        if let Some(meta) = unsafe { self.meta.as_ref() } {
            if meta.pages.load(Ordering::SeqCst) == self.pages()
                && meta.magic.load(Ordering::SeqCst) == MAGIC
            {
                info!("recover p={}", self.pages());
                let deep = meta.crashed.load(Ordering::SeqCst);
                if deep {
                    warn!("Try recover crashed allocator!");
                }
                let mut trees = Vec::with_capacity(self.pages().div_ceil(L::N));
                trees.resize_with(self.pages().div_ceil(L::N), || {
                    AtomicCell::new(Entry3::new())
                });
                self.trees.entries = trees.into();

                for i in 0..self.pages().div_ceil(L::N) {
                    let page = i * L::N;
                    let pages = self.lower.recover(page, deep)?;

                    self.trees[i].store(Entry3::new_table(pages, false));
                    // Add to lists
                    self.trees.push(i, pages, L::N);
                }

                meta.crashed.store(true, Ordering::SeqCst);
                Ok(())
            } else {
                Err(Error::Initialization)
            }
        } else {
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

        match local.reserved.fetch_update(|v| v.dec(1 << order)) {
            Ok(old) => {
                let start = old.idx();
                match self.lower.get(start, order) {
                    Ok(page) => {
                        if order < 64usize.ilog2() as usize {
                            // Save start index for lower allocations
                            let _ = local.reserved.compare_exchange(old, old.with_idx(page));
                        }
                        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
                    }
                    Err(Error::Memory) => {
                        // counter reset
                        warn!("alloc failed o={order} => retry");
                        let max = (self.pages() - align_down(start, L::N)).min(L::N);
                        // Increment global to prevent race condition with concurrent reservation
                        if let Err(old) =
                            self.trees[start / L::N].fetch_update(|v| v.inc(1 << order, max))
                        {
                            error!("Counter reset failed o={order} {old:?}");
                            Err(Error::Corruption)
                        } else {
                            // reserve new, pushing the old entry to the end of the partial list
                            self.reserve_or_wait(&local.reserved, old, true)?;
                            Err(Error::Retry)
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            Err(entry) => {
                // TODO: try sync with global

                // reserve new
                self.reserve_or_wait(&local.reserved, entry, false)?;
                Err(Error::Retry)
            }
        }
    }

    /// Try to reserve a new subtree or wait for concurrent reservations to finish.
    ///
    /// If `retry`, tries to reserve a less fragmented subtree
    fn reserve_or_wait(
        &self,
        local: &AtomicCell<Entry3>,
        old: Entry3,
        retry: bool,
    ) -> Result<Entry3> {
        // Set the reserved flag, locking the reservation
        if !old.reserved() && local.fetch_update(|v| v.toggle_reserve(true)).is_ok() {
            // Try reserve new subtree
            let new = match self.trees.reserve_from_list(L::N, retry) {
                Ok(ret) => ret,
                Err(e) => {
                    // Clear reserve flag
                    if local.fetch_update(|v| v.toggle_reserve(false)).is_err() {
                        error!("unexpected reserve state");
                        return Err(Error::Corruption);
                    }
                    return Err(e);
                }
            };
            match self.cas_reserved(local, new, true, retry) {
                Ok(_) => Ok(new),
                Err(Error::Retry) => {
                    error!("unexpected reserve state");
                    Err(Error::Corruption)
                }
                Err(e) => Err(e),
            }
        } else {
            // Wait for concurrent reservation to end
            for _ in 0..(2 * CAS_RETRIES) {
                let new = local.load();
                if !new.reserved() {
                    return Ok(new);
                }
                hint::spin_loop(); // pause cpu
            }
            error!("Timeout reservation wait");
            Err(Error::Corruption)
        }
    }

    fn reserve_entry(&self, local: &AtomicCell<Entry3>, i: usize) -> Result<bool> {
        // Try to reserve it for bulk frees
        if let Ok(new) = self.trees[i].fetch_update(|v| v.reserve_min(Trees::almost_full())) {
            match self.cas_reserved(local, new.with_idx(i * L::N), false, false) {
                Ok(_) => Ok(true),
                Err(Error::Retry) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.pages() - i * L::N).min(L::N);
                    if self.trees[i]
                        .fetch_update(|v| v.unreserve_add(new.free(), max))
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
        local: &AtomicCell<Entry3>,
        new: Entry3,
        expect_reserved: bool,
        enqueue_back: bool,
    ) -> Result<()> {
        debug_assert!(!new.reserved());

        let local = local
            .fetch_update(|v| (v.reserved() == expect_reserved).then_some(new))
            .map_err(|_| Error::Retry)?;
        if !local.has_idx() {
            return Ok(());
        }

        let i = local.idx() / L::N;
        let max = (self.pages() - i * L::N).min(L::N);
        if let Ok(global) = self.trees[i].fetch_update(|v| v.unreserve_add(local.free(), max)) {
            // Only if not already in list
            if global.next() == Next::Outside {
                if enqueue_back {
                    self.trees.push_back(i, global.free() + local.free(), L::N);
                } else {
                    self.trees.push(i, global.free() + local.free(), L::N);
                }
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }
}

#[derive(Default)]
struct Trees {
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    entries: Box<[AtomicCell<Entry3>]>,
    /// List of idx to subtrees
    lists: Mutex<Lists>,
}

#[derive(Default)]
struct Lists {
    /// List of subtrees where all pages are free
    empty: BufferList<Entry3>,
    /// List of subtrees that are partially allocated.
    /// This list may also include reserved or 'empty' subtrees, which should be skipped.
    partial: BufferList<Entry3>,
}

impl Index<usize> for Trees {
    type Output = AtomicCell<Entry3>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl fmt::Debug for Trees {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let (empty, partial) = {
            let lists = self.lists.lock();
            (
                lists.empty.iter(self).take(max + 1).count(),
                lists.partial.iter(self).take(max + 1).count(),
            )
        };

        write!(f, "(total: {max}, empty: ")?;
        if empty <= max {
            write!(f, "{empty}")?;
        } else {
            write!(f, "!!")?;
        }
        write!(f, ", partial: ")?;
        if partial <= max {
            write!(f, "{partial}")?;
        } else {
            write!(f, "!!")?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Trees {
    fn init(&mut self, pages: usize, span: usize, free_all: bool) {
        let len = pages.div_ceil(span);
        let mut entries = Vec::with_capacity(len);

        if free_all {
            let len = pages.div_ceil(span);
            entries.resize_with(len - 1, || {
                AtomicCell::new(Entry3::empty(span).with_next(Next::Outside))
            });

            // The last one might be cut off
            let max = ((pages - 1) % span) + 1;
            entries.push(AtomicCell::new(
                Entry3::new().with_free(max).with_next(Next::Outside),
            ));

            self.entries = entries.into();

            self.push_empty_all(0..len - 1);
            self.push(len - 1, max, span);
        } else {
            entries.resize_with(len, || {
                AtomicCell::new(Entry3::new().with_next(Next::Outside))
            });
            self.lists = Default::default();
            self.entries = entries.into();
        }
    }

    const fn almost_full() -> usize {
        1 << 10 // MAX_ORDER
    }

    fn push_empty_all(&self, entries: impl Iterator<Item = usize>) {
        let mut lists = self.lists.lock();
        for entry in entries {
            lists.empty.push(self, entry);
        }
    }

    fn push(&self, i: usize, new_pages: usize, span: usize) {
        // Add to list if new counter is small enough
        if new_pages == span {
            self.lists.lock().empty.push(self, i);
        } else if new_pages > Self::almost_full() {
            self.lists.lock().partial.push(self, i);
        }
    }

    fn push_back(&self, i: usize, new_pages: usize, span: usize) {
        if new_pages == span {
            self.lists.lock().empty.push(self, i);
        } else if new_pages > Self::almost_full() {
            self.lists.lock().partial.push_back(self, i);
        }
    }

    fn reserve_empty(&self, span: usize) -> Result<Entry3> {
        if let Some(i) = {
            let mut lists = self.lists.lock();
            let r = lists.empty.pop(self);
            drop(lists); // unlock immediately
            r
        } {
            info!("reserve empty {i}");
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve_min(span)) {
                Ok(entry.with_idx(i * span))
            } else {
                error!("reserve empty failed");
                Err(Error::Corruption)
            }
        } else {
            warn!("no empty tree {self:?}");
            Err(Error::Memory)
        }
    }

    fn reserve_partial(&self, span: usize) -> Result<Entry3> {
        let mut skipped_empty = None;
        loop {
            if let Some(i) = {
                let mut lists = self.lists.lock();
                let r = lists.partial.pop(self);
                drop(lists); // unlock immediately
                r
            } {
                info!("reserve partial {i}");

                match self[i].fetch_update(|v| v.reserve_partial(Self::almost_full()..span)) {
                    Ok(entry) => {
                        if let Some(empty) = skipped_empty {
                            self.lists.lock().empty.push(self, empty);
                        }
                        return Ok(entry.with_idx(i * span));
                    }
                    Err(entry) => {
                        // Skip reserved and empty entries
                        // They might be reserved by the put-reserve optimization
                        if !entry.reserved() && entry.free() == span {
                            if let Some(empty) = skipped_empty.replace(i) {
                                self.lists.lock().empty.push(self, empty);
                            }
                        }
                    }
                }
            } else if let Some(i) = skipped_empty {
                // Reserve the last skipped empty entry instead
                return if let Ok(entry) = self[i].fetch_update(|v| v.reserve_min(span)) {
                    Ok(entry.with_idx(i * span))
                } else {
                    error!("reserve empty failed");
                    Err(Error::Corruption)
                };
            } else {
                return Err(Error::Memory);
            }
        }
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees.
    fn reserve_from_list(&self, span: usize, prioritize_empty: bool) -> Result<Entry3> {
        info!("reserve prio={prioritize_empty}");
        if prioritize_empty {
            match self.reserve_empty(span) {
                Err(Error::Memory) => self.reserve_partial(span),
                r => r,
            }
        } else {
            match self.reserve_partial(span) {
                Err(Error::Memory) => self.reserve_empty(span),
                r => r,
            }
        }
    }
}
