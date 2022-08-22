use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, hint};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};
use spin::Mutex;

use super::{Alloc, Local, MAGIC, MAX_PAGES};
use crate::atomic::{ANode, Atomic, BufList};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
use crate::upper::CAS_RETRIES;
use crate::util::{align_down, Page};
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
pub struct ArrayList<L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// CPU local data (only shared between CPUs if the memory area is too small)
    local: Box<[Local<PUTS_RESERVE>]>,
    /// Metadata of the lower alloc
    lower: L,
    /// Manages the allocators subtrees
    subtrees: Subtrees,
}

unsafe impl<L: LowerAlloc> Send for ArrayList<L> {}
unsafe impl<L: LowerAlloc> Sync for ArrayList<L> {}

impl<L: LowerAlloc> fmt::Debug for ArrayList<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;

        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;
        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.pte.load())?;
        }

        write!(f, "{:?}", self.subtrees)?;
        write!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Alloc for ArrayList<L> {
    #[cold]
    fn init(&mut self, mut cores: usize, mut memory: &mut [Page], persistent: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < self.pages_needed(cores) {
            warn!("memory {} < {}", memory.len(), self.pages_needed(cores));
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
        self.subtrees.init(pte3_num);

        Ok(())
    }

    fn recover(&self) -> Result<()> {
        if let Some(meta) = unsafe { self.meta.as_ref() }
            && meta.pages.load(Ordering::SeqCst) == self.pages()
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
        let pte3_num = self.pages().div_ceil(L::N);
        for i in 0..pte3_num - 1 {
            self.subtrees[i].store(Entry3::empty(L::N));
        }
        self.subtrees.push_empty_all((0..pte3_num - 1).into_iter());

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * L::N).min(L::N);
        self.subtrees[pte3_num - 1].store(Entry3::new().with_free(max));

        self.subtrees.push(pte3_num - 1, max, L::N);

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
        self.subtrees.clear();

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

        let i = page / L::N;
        // Save the modified subtree id for the push-reserve heuristic
        let c = core % self.local.len();
        let local = &self.local[c];

        let max = (self.pages() - i * L::N).min(L::N);

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
        match self.subtrees[i].update(|v| v.inc(num_pages, max)) {
            Ok(pte) => {
                let new_pages = pte.free() + num_pages;
                if !pte.reserved() && new_pages > Subtrees::almost_full(L::N) {
                    // Try to reserve subtree if recent frees were part of it
                    if core == c && local.frees_related(i) && self.reserve_entry(&local.pte, i)? {
                        return Ok(());
                    }

                    // Add to partially free list
                    // Only if not already in list
                    if pte.idx() == Entry3::IDX_MAX && pte.free() <= Subtrees::almost_full(L::N) {
                        self.subtrees.push_partial(i);
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
        L::N * cores
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
        for i in 0..self.pages().div_ceil(L::N) {
            let pte = self.subtrees[i].load();
            pages -= pte.free();
        }
        // Pages allocated in reserved subtrees
        for local in self.local.iter() {
            pages -= local.pte.load().free();
        }
        pages
    }
}

impl<L: LowerAlloc> Drop for ArrayList<L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}
impl<L: LowerAlloc> Default for ArrayList<L> {
    fn default() -> Self {
        Self {
            meta: null_mut(),
            subtrees: Default::default(),
            local: Default::default(),
            lower: Default::default(),
        }
    }
}

impl<L: LowerAlloc> ArrayList<L> {
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

            self.subtrees[i].store(Entry3::new_table(pages, false));

            // Add to lists
            self.subtrees.push(i, pages, L::N);
            total += pages;
        }
        Ok(total)
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
                        warn!("alloc failed o={order} => retry");
                        let max = (self.pages() - align_down(start, L::N)).min(L::N);
                        // Increment global to prevent race condition with concurrent reservation
                        if let Err(pte) =
                            self.subtrees[pte.idx()].update(|v| v.inc(1 << order, max))
                        {
                            error!("Counter reset failed o={order} {}: {pte:?}", pte.idx());
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
                // TODO: try sync with global

                // reserve new
                self.reserve_or_wait(&local.pte, pte, false)?;
                Err(Error::CAS)
            }
        }
    }

    /// Try to reserve a new subtree or wait for concurrent reservations to finish.
    ///
    /// If `retry`, tries to reserve a less fragmented subtree
    fn reserve_or_wait(&self, pte_a: &Atomic<Entry3>, old: Entry3, retry: bool) -> Result<Entry3> {
        // Set the reserved flag, locking the reservation
        if !old.reserved()
            && pte_a
                .update(|v| (!v.reserved()).then_some(v.with_reserved(true)))
                .is_ok()
        {
            // Try reserve new subtree
            let new_pte = match self.subtrees.reserve_from_list(L::N, retry) {
                Ok(ret) => ret,
                Err(e) => {
                    // Clear reserve flag
                    if pte_a
                        .update(|v| v.reserved().then_some(v.with_reserved(false)))
                        .is_err()
                    {
                        error!("unexpected reserve state");
                        return Err(Error::Corruption);
                    }
                    return Err(e);
                }
            };
            match self.cas_reserved(pte_a, new_pte, true, retry) {
                Ok(_) => Ok(new_pte),
                Err(Error::CAS) => {
                    error!("unexpected reserve state");
                    Err(Error::Corruption)
                }
                Err(e) => Err(e),
            }
        } else {
            // Wait for reservation to end
            for _ in 0..(2 * CAS_RETRIES) {
                let new_pte = pte_a.load();
                if new_pte.reserved() {
                    hint::spin_loop() // pause cpu
                } else {
                    return Ok(new_pte);
                }
            }
            error!("Timeout reservation wait");
            Err(Error::Corruption)
        }
    }

    fn reserve_entry(&self, pte_a: &Atomic<Entry3>, i: usize) -> Result<bool> {
        // Try to reserve it for bulk frees
        if let Ok(new_pte) = self.subtrees[i].update(|v| v.reserve(Subtrees::almost_full(L::N))) {
            match self.cas_reserved(pte_a, new_pte.with_idx(i), false, false) {
                Ok(_) => Ok(true),
                Err(Error::CAS) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.pages() - i * L::N).min(L::N);
                    if self.subtrees[i]
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
        enqueue_back: bool,
    ) -> Result<()> {
        debug_assert!(!new_pte.reserved());

        let pte = pte_a
            .update(|v| (v.reserved() == expect_reserved).then_some(new_pte))
            .map_err(|_| Error::CAS)?;
        let i = pte.idx();
        if i >= Entry3::IDX_MAX {
            return Ok(());
        }

        let max = (self.pages() - i * L::N).min(L::N);
        if let Ok(v) = self.subtrees[i].update(|v| v.unreserve_add(pte.free(), max)) {
            // Only if not already in list
            if !v.is_valid() {
                // Add to list
                if enqueue_back {
                    self.subtrees.push_back(i, v.free() + pte.free(), L::N);
                } else {
                    self.subtrees.push(i, v.free() + pte.free(), L::N);
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
struct Subtrees {
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    entries: Box<[Atomic<Entry3>]>,
    /// List of idx to subtrees that are not allocated at all
    empty: Mutex<BufList<Entry3>>,
    /// List of idx to subtrees that are partially allocated with huge pages
    partial: Mutex<BufList<Entry3>>,
}

impl Index<usize> for Subtrees {
    type Output = Atomic<Entry3>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl fmt::Debug for Subtrees {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let empty_count = self.empty.lock().iter(self).count();
        let partial_count = self.partial.lock().iter(self).count();

        writeln!(f, "    total: {}", self.entries.len())?;
        writeln!(f, "    empty: {empty_count}")?;
        writeln!(f, "    partial: {partial_count}")?;

        let mut free = 0;
        for pte in &self.entries[..] {
            free += pte.load().free();
        }
        writeln!(f, "    free pages: {free}")?;
        Ok(())
    }
}

impl Subtrees {
    fn init(&mut self, pte3_num: usize) {
        let mut pte3s = Vec::with_capacity(pte3_num);
        pte3s.resize_with(pte3_num, || Atomic::new(Entry3::new()));

        self.entries = pte3s.into();
        self.empty = Default::default();
        self.partial = Default::default();
    }

    fn almost_full(span: usize) -> usize {
        span / 8
    }

    fn clear(&self) {
        // Set all entries to zero
        for pte in &self.entries[..] {
            pte.store(Entry3::new().with_next(None));
        }
        // Clear the lists
        self.empty.lock().clear(self);
        self.partial.lock().clear(self);
    }

    fn push_empty_all(&self, entries: impl Iterator<Item = usize>) {
        let mut empty = self.empty.lock();
        for entry in entries {
            empty.push(self, entry);
        }
    }

    fn push(&self, i: usize, new_pages: usize, span: usize) {
        // Add to list if new counter is small enough
        if new_pages == span {
            self.empty.lock().push(self, i);
        } else if new_pages > Self::almost_full(span) {
            self.partial.lock().push(self, i);
        }
    }

    fn push_back(&self, i: usize, new_pages: usize, span: usize) {
        if new_pages == span {
            self.empty.lock().push(self, i);
        } else if new_pages > Self::almost_full(span) {
            self.partial.lock().push_back(self, i);
        }
    }

    fn push_partial(&self, i: usize) {
        self.partial.lock().push(self, i);
    }

    fn reserve_empty(&self, span: usize) -> Result<Entry3> {
        if let Some(i) = self.empty.lock().pop(self) {
            info!("reserve empty {i}");
            if let Ok(pte) = self[i].update(|v| v.reserve_empty(span)) {
                Ok(pte.with_idx(i))
            } else {
                error!("reserve empty failed");
                Err(Error::Corruption)
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn reserve_partial(&self, span: usize) -> Result<Entry3> {
        while let Some(i) = self.partial.lock().pop(self) {
            info!("reserve partial {i}");

            match self[i].update(|v| v.reserve_partial(Self::almost_full(span), span)) {
                Ok(pte) => return Ok(pte.with_idx(i)),
                Err(pte) => {
                    // Skip empty entries
                    if !pte.reserved() && pte.free() == span {
                        self.empty.lock().push(self, i);
                    }
                }
            }
        }
        Err(Error::Memory)
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees.
    fn reserve_from_list(&self, span: usize, prioritize_empty: bool) -> Result<Entry3> {
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
