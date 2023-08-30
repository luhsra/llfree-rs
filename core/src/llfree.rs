//! Upper allocator implementation

use core::ffi::c_void;
use core::fmt;
use core::hint::spin_loop;
use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use core::unreachable;

use alloc::boxed::Box;
use alloc::vec::Vec;

use bitfield_struct::bitfield;
use log::{error, info, warn};

use crate::atomic::{Atom, Atomic};
use crate::entry::{ReservedTree, Tree};
use crate::frame::{Frame, PFNRange, PFN};
use crate::lower::Lower;
use crate::util::{align_down, Align};
use crate::CAS_RETRIES;
use crate::{Error, Result};

use super::{trees::Trees, Alloc, Init};

/// This allocator splits its memory range into chunks.
/// These chunks are reserved by CPUs to reduce sharing.
/// Allocations/frees within the chunk are handed over to the
/// lower allocator.
/// These chunks are, due to the inner workins of the lower allocator,
/// called *trees*.
///
/// This allocator stores the tree entries in a packed array.
/// For reservations, the allocator simply scans the array for free entries,
/// while prioritizing partially empty already fragmented chunks to avoid
/// further fragmentation.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[derive(Default)]
#[repr(align(64))]
pub struct LLFree {
    /// Pointer to the metadata frame at the end of the allocators persistent memory range
    meta: Option<&'static Meta>,
    /// CPU local data (only shared between CPUs if the memory area is too small)
    local: Box<[Align<Local>]>,
    /// Metadata of the lower alloc
    lower: Lower,
    /// Manages the allocators trees
    trees: Trees<{ Lower::N }>,
}

unsafe impl Send for LLFree {}
unsafe impl Sync for LLFree {}

/// Last frees heuristic that reserves trees where a lot of frees happen
/// to reduce false sharing on the tree counters.
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
struct LastFrees {
    /// Tree index where the last free occurred
    #[bits(48)]
    tree_index: usize,
    /// Number of consecutive frees that happened in the same `tree_index`
    #[bits(16)]
    count: usize,
}
impl Atomic for LastFrees {
    type I = AtomicU64;
}

/// Core-local data
#[derive(Default)]
struct Local {
    /// Local copy of the reserved tree entry
    preferred: Atom<ReservedTree>,
    /// Last frees heuristic
    last_frees: Atom<LastFrees>,
}

impl Local {
    /// Threshold for the number of frees after which a tree is reserved
    const F: usize = 4;

    /// Add a tree index to the history.
    fn frees_push(&self, tree_index: usize) {
        // If the update of this heuristic fails, ignore it
        // Relaxed ordering is enough, as this is not shared between CPUs
        let _ = self.last_frees.fetch_update(|v| {
            if v.tree_index() == tree_index {
                (v.count() < Self::F).then_some(v.with_count(v.count() + 1))
            } else {
                Some(LastFrees::new().with_tree_index(tree_index).with_count(1))
            }
        });
    }
    /// Checks if the previous `count` frees had the same tree index.
    fn frees_in_tree(&self, tree_index: usize) -> bool {
        let lf = self.last_frees.load();
        lf.tree_index() == tree_index && lf.count() >= Self::F
    }
}

impl fmt::Debug for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("reserved", &self.preferred.load())
            .field("frees", &self.last_frees.load())
            .finish()
    }
}

/// Non-Volatile metadata that is used to recover the allocator at reboot
#[repr(align(0x1000))]
struct Meta {
    /// A magic number used to check if the persistent memory contains the allocator state
    magic: AtomicUsize,
    /// Number of frames managed by the persistent allocator
    frames: AtomicUsize,
    /// Flag that stores if the system has crashed or was shutdown correctly
    crashed: AtomicBool,
}
impl Meta {
    /// Magic marking the meta frame.
    const MAGIC: usize = 0x_dead_beef;
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Frame::SIZE);

impl fmt::Debug for LLFree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;

        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.frames()
        )?;

        writeln!(f, "    trees: {:?} ({} frames)", self.trees, Lower::N)?;
        let free_frames = self.free_frames();
        let free_huge_frames = self.free_huge_frames();
        writeln!(
            f,
            "    free frames: {free_frames} ({free_huge_frames} huge, {} trees)",
            free_frames.div_ceil(Lower::N)
        )?;

        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.preferred.load())?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

impl Alloc for LLFree {
    /// Return the name of the allocator.
    #[cold]
    fn name(&self) -> &'static str {
        "LLFree"
    }

    /// Initialize the allocator.
    #[cold]
    fn init(
        &mut self,
        mut cores: usize,
        mut memory: Range<PFN>,
        init: Init,
        free_all: bool,
    ) -> Result<()> {
        info!("initializing c={cores} {memory:?} {}", memory.len());

        if memory.start.0 % (1 << Lower::MAX_ORDER) != 0 {
            warn!("Unexpected memory alignment {:x}", memory.start.0);
            memory.start.0 = align_down(memory.start.0, 1 << Lower::MAX_ORDER);
        }

        if memory.len() < Lower::N * cores {
            warn!("memory {} < {}", memory.len(), Lower::N * cores);
            if memory.len() < 2 {
                error!("Expecting at least {} frames", 2);
                return Err(Error::Memory);
            }
            cores = memory.len().div_ceil(Lower::N);
        }

        if init != Init::Volatile {
            // Last frame is reserved for metadata
            let m = memory.start..PFN(memory.end.0 - 1);
            self.meta = Some(unsafe { &*m.end.as_ptr().cast() });
            memory = m;
        }

        // Create lower allocator
        self.lower = Lower::new(cores, memory.start, memory.len(), init, free_all);

        // Init per-cpu data
        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Default::default);
        self.local = local.into();

        if init == Init::Recover {
            self.recover()?;
        } else {
            self.trees.init(self.frames(), free_all);
        }

        if let Some(meta) = self.meta {
            meta.frames.store(self.frames(), Ordering::SeqCst);
            meta.magic.store(Meta::MAGIC, Ordering::SeqCst);
            meta.crashed.store(true, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Allocate a new frame of `order` on the given `core`.
    fn get(&self, core: usize, order: usize) -> Result<PFN> {
        if order > Lower::MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..CAS_RETRIES {
            match self.get_inner(core, order) {
                Ok(frame) => return Ok(self.lower.begin().off(frame)),
                Err(Error::Retry) => continue,
                Err(Error::Memory) => return self.drain_and_steal(core, order),
                Err(e) => return Err(e),
            }
        }
        error!("Exceeding retries");
        Err(Error::Memory)
    }

    /// Free the `frame` of `order` on the given `core`..
    fn put(&self, core: usize, addr: PFN, order: usize) -> Result<()> {
        let frame = self.addr_to_frame(addr, order)?;

        // First free the frame in the lower allocator
        self.lower.put(frame, order)?;

        // Then update local / global counters
        let i = frame / Lower::N;
        let local = &self.local[core % self.local.len()];
        let max = (self.frames() - i * Lower::N).min(Lower::N);

        // Try update own tree first
        let num_frames = 1 << order;
        if let Err(tree) = local
            .preferred
            .fetch_update(|v| v.inc(num_frames, max, |s| s / Lower::N == i))
        {
            if tree.start() / Lower::N == i {
                error!("inc failed L{i}: {tree:?} o={order}");
                return Err(Error::Corruption);
            }
        } else {
            // Save the modified tree id for the push-reserve heuristic
            local.frees_push(i);
            return Ok(());
        };

        let mut reserved = false;
        // Tree not owned by us -> update global
        match self.trees[i].fetch_update(|v| {
            let mut v = v.inc(num_frames, max)?;
            if !v.reserved() && v.free() > Trees::<{ Lower::N }>::MIN_FREE && local.frees_in_tree(i)
            {
                // put-reserve optimization:
                // Reserve the tree that was targeted by the recent frees
                v = v.with_free(0).with_reserved(true);
                reserved = true;
            }
            Some(v)
        }) {
            Ok(tree) => {
                // Update preferred tree if reserved
                let free = tree.free() + num_frames;
                if !reserved || !self.reserve_for_put(free, &local.preferred, i)? {
                    local.frees_push(i);
                }
                Ok(())
            }
            Err(tree) => {
                error!("inc failed i{i}: {tree:?} o={order}");
                Err(Error::Corruption)
            }
        }
    }

    /// Returns if `frame` is free. This might be racy!
    fn is_free(&self, addr: PFN, order: usize) -> bool {
        if let Ok(frame) = self.addr_to_frame(addr, order) {
            self.lower.is_free(frame, order)
        } else {
            false
        }
    }

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize {
        self.lower.frames()
    }

    /// Return the number of allocated frames.
    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }

    /// Unreserve cpu-local frames
    fn drain(&self, core: usize) -> Result<()> {
        let local = &self.local[core % self.local.len()];
        match self.cas_reserved(&local.preferred, ReservedTree::default(), false) {
            Err(Error::Retry) => Ok(()), // ignore cas errors
            r => r,
        }
    }

    /// Return the number of free frames.
    fn free_frames(&self) -> usize {
        let mut frames = 0;
        // Global array
        for tree in self.trees.iter() {
            frames += tree.load().free();
        }
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            frames += local.preferred.load().free();
        }
        frames
    }

    /// Return the number of free huge frames or 0 if the allocator cannot allocate huge frames.
    fn free_huge_frames(&self) -> usize {
        let mut counter = 0;
        self.lower.for_each_huge_frame(|_, c| {
            if c == (1 << Lower::HUGE_ORDER) {
                counter += 1;
            }
        });
        counter
    }

    fn for_each_huge_frame(&self, ctx: *mut c_void, f: fn(*mut c_void, PFN, usize)) {
        self.lower.for_each_huge_frame(|frame, c| f(ctx, frame, c))
    }
}

impl LLFree {
    /// Recover the allocator from NVM after reboot.
    /// If crashed then the level 1 frame tables are traversed and diverging counters are corrected.
    fn recover(&mut self) -> Result<()> {
        if let Some(meta) = self.meta {
            if meta.frames.load(Ordering::SeqCst) == self.frames()
                && meta.magic.load(Ordering::SeqCst) == Meta::MAGIC
            {
                info!("recover p={}", self.frames());
                // The active flag is set on boot and reset on a successful shutdown
                // If it is already set, the allocator has been crashed
                // In this case, we have to initiate a deep recovery, correcting all the counters
                let deep = meta.crashed.load(Ordering::SeqCst);
                if deep {
                    warn!("Try recover crashed allocator!");
                }

                let mut trees = Vec::with_capacity(self.frames().div_ceil(Lower::N));
                // Recover each tree one-by-one
                for i in 0..self.frames().div_ceil(Lower::N) {
                    let frame = i * Lower::N;
                    let frames = self.lower.recover(frame, deep)?;
                    trees.push(Atom::new(Tree::new_with(frames, false)));
                }
                self.trees = trees.into();

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

    /// Convert an address to the frame index
    fn addr_to_frame(&self, addr: PFN, order: usize) -> Result<usize> {
        if order > Lower::MAX_ORDER {
            error!("invalid order: {order} > {}", Lower::MAX_ORDER);
            return Err(Error::Memory);
        }

        // Check alignment and if this addr is within our address range
        if addr.0 % (1 << order) != 0 || !self.lower.memory().contains(&addr) {
            error!("invalid addr {addr} r={:?} o={order}", self.lower.memory());
            return Err(Error::Address);
        }

        Ok(addr.0 - self.lower.begin().0)
    }

    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, order: usize) -> Result<usize> {
        // Select local data (which can be shared between cores if we do not have enough memory)
        let local = &self.local[core];

        // Update the upper counters first
        let old = match local.preferred.fetch_update(|v| v.dec(1 << order)) {
            Ok(old) => old,
            Err(old) => {
                // If the local counter is large enough we do not have to reserve a new tree
                // Just update the local counter and reuse the current tree
                if self.try_sync_with_global(&local.preferred, old)? {
                    // Success -> Retry allocation
                    return Err(Error::Retry);
                }

                // The local tree is full -> reserve a new one
                return self.reserve_or_wait(core, order, old, false);
            }
        };

        // The start point for the search
        let start = old.start();
        // Try allocating with the lower allocator
        match self.get_lower(old, order) {
            Ok((reserved, frame)) => {
                // Success
                if order < 6 && start != reserved.start() {
                    // Save start index for lower allocations
                    let _ = local
                        .preferred
                        .compare_exchange(reserved.with_start(start), reserved);
                }
                Ok(frame)
            }
            Err(Error::Memory) => {
                // Failure (e.g. due to fragmentation)
                // Reset counters, reserve new entry and retry allocation
                warn!("alloc failed o={order} => retry");
                // Increment global to prevent race condition with concurrent reservation
                if let Err(_) =
                    self.trees[start / Lower::N].fetch_update(|v| v.inc(1 << order, Lower::N))
                {
                    error!("Counter reset failed");
                    Err(Error::Corruption)
                } else {
                    self.reserve_or_wait(core, order, old, true)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Allocate a frame in the lower allocator
    fn get_lower(&self, reserved: ReservedTree, order: usize) -> Result<(ReservedTree, usize)> {
        let frame = self.lower.get(reserved.start(), order)?;
        Ok((
            reserved
                .with_free(reserved.free() - (1 << order))
                .with_start(frame),
            frame,
        ))
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn try_sync_with_global(&self, local: &Atom<ReservedTree>, old: ReservedTree) -> Result<bool> {
        let i = old.start() / Lower::N;
        if i >= self.trees.len() {
            return Ok(false);
        }

        if let Ok(entry) = self.trees[i].fetch_update(|e| {
            (e.reserved() && e.free() + old.free() > Trees::<{ Lower::N }>::MIN_FREE)
                .then_some(e.with_free(0))
        }) {
            debug_assert!(old.free() + entry.free() <= Lower::N);

            if local
                .compare_exchange(old, old.with_free(old.free() + entry.free()))
                .is_ok()
            {
                // Sync successfull -> retry allocation
                return Ok(true);
            } else {
                // undo global change
                if self.trees[i]
                    .fetch_update(|e| Some(e.with_free(e.free() + entry.free())))
                    .is_err()
                {
                    unreachable!("Failed undo sync");
                }
            }
        }

        Ok(false)
    }

    /// Try to reserve a new tree or wait for concurrent reservations to finish.
    /// We directly try to allocate in the tree and continues the search if it is fragmented.
    ///
    /// If `fragmented`, prioritize less fragmented trees
    fn reserve_or_wait(
        &self,
        core: usize,
        order: usize,
        old: ReservedTree,
        fragmented: bool,
    ) -> Result<usize> {
        let local = &self.local[core].preferred;

        // Set the reserved flag, locking the reservation
        if !old.locked() {
            if let Ok(old) = local.fetch_update(|v| v.toggle_locked(true)) {
                // Try reserve new tree
                return self.reserve_and_get(core, order, old, fragmented);
            }
        }

        // Wait for concurrent reservation to end
        for _ in 0..CAS_RETRIES {
            let reserved = local.load();
            if !reserved.locked() {
                // Try allocation on new tree
                return match self.lower.get(reserved.start(), order) {
                    Err(Error::Memory) => Err(Error::Retry),
                    r => r,
                };
            }
            spin_loop()
        }
        warn!("Timeout reservation wait");
        Err(Error::Retry)
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(
        &self,
        core: usize,
        order: usize,
        old: ReservedTree,
        fragmented: bool,
    ) -> Result<usize> {
        let local = &self.local[core].preferred;

        // Try reserve new tree
        let start = if old.has_start() {
            old.start() / Lower::N
        } else {
            // Different initial starting point for every core
            self.trees.len() / self.local.len() * core
        };

        // Reserved a new tree an allocate a frame in it
        let cores = self.local.len();
        let (new, frame) = match self.trees.reserve(order, cores, start, fragmented, |r| {
            self.get_lower(r, order)
        }) {
            Ok(entry) => entry,
            Err(e) => {
                // Rollback: Clear reserve flag
                local
                    .fetch_update(|v| v.toggle_locked(false))
                    .map_err(|_| Error::Corruption)?;
                return Err(e);
            }
        };

        match self.cas_reserved(local, new, true) {
            Ok(_) => Ok(frame),
            Err(Error::Retry) => {
                error!("unexpected reserve state");
                Err(Error::Corruption)
            }
            Err(e) => Err(e),
        }
    }

    /// Fallback if all trees are full
    fn drain_and_steal(&self, core: usize, order: usize) -> Result<PFN> {
        // Drain all
        for core in 0..self.local.len() {
            self.drain(core)?;
        }

        // Steal drained trees
        let start = self.trees.len() / self.local.len() * core;
        let (reserved, frame) = self
            .trees
            .reserve_far(start, (1 << order).., |r| self.get_lower(r, order))?;

        let local = &self.local[core].preferred;
        match self.cas_reserved(local, reserved, false) {
            Ok(_) => Ok(self.lower.begin().off(frame)),
            Err(Error::Retry) => {
                error!("unexpected reserve state");
                Err(Error::Corruption)
            }
            Err(e) => Err(e),
        }
    }

    /// Reserve an entry for bulk frees
    fn reserve_for_put(&self, free: usize, local: &Atom<ReservedTree>, i: usize) -> Result<bool> {
        let entry = ReservedTree::new_with(free, i * Lower::N);
        match self.cas_reserved(local, entry, false) {
            Ok(_) => Ok(true),
            Err(Error::Retry) => {
                warn!("rollback {i}");
                // Rollback reservation
                let max = (self.frames() - i * Lower::N).min(Lower::N);
                if let Err(_) = self.trees[i].fetch_update(|v| v.unreserve_add(free, max)) {
                    error!("put - reservation rollback failed");
                    return Err(Error::Corruption);
                }
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    fn cas_reserved(
        &self,
        local: &Atom<ReservedTree>,
        new: ReservedTree,
        expect_locked: bool,
    ) -> Result<()> {
        debug_assert!(!new.locked());

        let old = local
            .fetch_update(|v| (v.locked() == expect_locked).then_some(new))
            .map_err(|_| Error::Retry)?;

        self.trees.unreserve(old, self.frames())
    }
}

impl Drop for LLFree {
    fn drop(&mut self) {
        if let Some(meta) = self.meta {
            meta.crashed.store(false, Ordering::SeqCst);
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use super::Local;

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        let local = Local::default();
        let frame1 = 43;
        let i1 = frame1 / (512 * 512);
        assert!(!local.frees_in_tree(i1));
        local.frees_push(i1);
        local.frees_push(i1);
        local.frees_push(i1);
        assert!(!local.frees_in_tree(i1));
        local.frees_push(i1);
        assert!(local.frees_in_tree(i1));
        let frame2 = 512 * 512 + 43;
        let i2 = frame2 / (512 * 512);
        assert_ne!(i1, i2);
        local.frees_push(i2);
        assert!(!local.frees_in_tree(i1));
        assert!(!local.frees_in_tree(i2));
    }
}
