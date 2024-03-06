//! Upper allocator implementation

use core::mem::align_of;
use core::sync::atomic::AtomicU64;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::{error, info, warn};

use super::trees::Trees;
use super::{Alloc, Init};
use crate::atomic::{Atom, Atomic};
use crate::entry::{LocalTree, Preferred};
use crate::lower::Lower;
use crate::util::{size_of_slice, spin_wait, Align};
use crate::{Error, MetaSize, Result, CAS_RETRIES};

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
#[repr(align(64))]
pub struct LLFree<'a> {
    /// CPU local data (only shared between CPUs if the memory area is too small)
    local: &'a [Align<Local>],
    /// Metadata of the lower alloc
    pub lower: Lower<'a>,
    /// Manages the allocators trees
    pub trees: Trees<'a, { Lower::N }>,
}

unsafe impl Send for LLFree<'_> {}
unsafe impl Sync for LLFree<'_> {}

/// Size of the dynamic metadata
struct Metadata {
    local_size: usize,
    tree_size: usize,
}

impl Metadata {
    fn new(cores: usize, frames: usize) -> Self {
        let cores = cores.clamp(1, frames.div_ceil(Lower::N));
        let local_size = size_of_slice::<Align<Local>>(cores);
        let tree_size = Trees::<{ Lower::N }>::metadata_size(frames);
        Self {
            local_size,
            tree_size,
        }
    }
}

impl<'a> Alloc<'a> for LLFree<'a> {
    const MAX_ORDER: usize = Lower::MAX_ORDER;
    const HUGE_ORDER: usize = Lower::HUGE_ORDER;

    /// Return the name of the allocator.
    #[cold]
    #[cfg(feature = "16K")]
    fn name() -> &'static str {
        "LLFree16K"
    }

    #[cfg(not(feature = "16K"))]
    fn name() -> &'static str {
        "LLFree"
    }

    /// Initialize the allocator.
    #[cold]
    fn new(
        mut cores: usize,
        frames: usize,
        init: Init,
        primary: &'a mut [u8],
        secondary: &'a mut [u8],
    ) -> Result<Self> {
        info!(
            "initializing c={cores} f={frames} {:?} {:?}",
            primary.as_ptr_range(),
            secondary.as_ptr_range()
        );

        if frames < Lower::N * cores {
            warn!("memory {} < {}", frames, Lower::N * cores);
            cores = frames.div_ceil(Lower::N);
        }

        let m = Metadata::new(cores, frames);
        if !secondary.as_ptr().is_aligned_to(align_of::<Align>())
            || secondary.len() < m.local_size + m.tree_size
        {
            error!("secondary metadata");
            return Err(Error::Memory);
        }

        // Create lower allocator
        let lower = Lower::new(frames, init, primary)?;

        let (local, trees) = secondary.split_at_mut(m.local_size);

        // Init per-cpu data
        let local = unsafe { slice::from_raw_parts_mut(local.as_mut_ptr().cast(), cores) };
        local.fill_with(Default::default);

        // Init tree array
        let trees = Trees::new(frames, trees, |start| lower.free_in_tree(start));

        Ok(Self {
            local,
            lower,
            trees,
        })
    }

    fn metadata_size(cores: usize, frames: usize) -> MetaSize {
        let m = Metadata::new(cores, frames);
        MetaSize {
            primary: Lower::metadata_size(frames),
            secondary: m.local_size + m.tree_size,
        }
    }

    fn metadata(&mut self) -> (&'a mut [u8], &'a mut [u8]) {
        let m = Self::metadata_size(self.local.len(), self.lower.frames());
        let secondary = unsafe {
            slice::from_raw_parts_mut(self.local.as_ptr().cast_mut().cast(), m.secondary)
        };
        (self.lower.metadata(), secondary)
    }

    fn get(&self, core: usize, order: usize) -> Result<usize> {
        if order > Lower::MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..CAS_RETRIES {
            //info!("Retrying allocation {i} for order o={order}. local Allocator State: {:?}", self.local);
            match self.get_inner(core, order) {
                Ok(frame) => return Ok(frame),
                Err(Error::Retry) => continue,
                Err(e) => return Err(e),
            }
        }
        error!("Exceeding retries");
        Err(Error::Memory)
    }

    fn put(&self, core: usize, frame: usize, order: usize) -> Result<()> {
        if frame >= self.lower.frames() {
            error!("invalid frame number");
            return Err(Error::Memory);
        }

        // First free the frame in the lower allocator
        self.lower.put(frame, order)?;

        // Then update local / global counters
        let i = frame / Lower::N;
        let local = &self.local[core % self.local.len()];

        // Update the put-reserve heuristic
        let may_reserve = local.frees_push(i);

        // Try update own tree first
        let num_frames = 1 << order;
        if let Ok(_) = local
            .preferred
            .fetch_update(|v| v.inc(num_frames, Lower::N, |s| s / Lower::N == i))
        {
            return Ok(());
        }

        let mut reserved = false;
        // Tree not owned by us -> update global
        let min = Trees::<{ Lower::N }>::MIN_FREE;
        // Increment or reserve the tree
        let tree = self.trees[i]
            .fetch_update(|v| {
                let v = v.inc(num_frames, Lower::N);
                if may_reserve && !v.reserved() && v.free() > min {
                    // Reserve the tree that was targeted by the last N frees
                    reserved = true;
                    Some(v.with_free(0).with_reserved(true))
                } else {
                    reserved = false; // <- This is very important if CAS fails!
                    Some(v)
                }
            })
            .unwrap();

        if reserved {
            // Change preferred tree if to speedup future frees
            let free = tree.free() + num_frames;
            let entry = Preferred::tree(i * Lower::N, free, false);
            if !self.swap_reserved(&local.preferred, entry, false) {
                // Rollback reservation
                self.trees[i]
                    .fetch_update(|v| v.unreserve_add(free, Lower::N))
                    .expect("rollback failed");
            }
            Ok(())
        } else {
            // Update free statistic
            local.frees_push(i);
            Ok(())
        }
    }

    fn is_free(&self, frame: usize, order: usize) -> bool {
        if frame < self.lower.frames() {
            self.lower.is_free(frame, order)
        } else {
            false
        }
    }

    fn frames(&self) -> usize {
        self.lower.frames()
    }

    fn cores(&self) -> usize {
        self.local.len()
    }

    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }

    fn drain(&self, core: usize) -> Result<()> {
        let local = &self.local[core % self.local.len()];
        // ignore cas errors
        self.swap_reserved(&local.preferred, Preferred::default(), false);
        Ok(())
    }

    fn free_frames(&self) -> usize {
        let mut frames = 0;
        // Global array
        for tree in self.trees.iter() {
            frames += tree.load().free();
        }
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            if let Some(LocalTree { free, .. }) = local.preferred.load().tree {
                frames += free;
            }
        }
        frames
    }

    fn free_huge_frames(&self) -> usize {
        let mut counter = 0;
        self.lower.for_each_huge_frame(|_, c| {
            if c == (1 << Lower::HUGE_ORDER) {
                counter += 1;
            }
        });
        counter
    }

    fn free_at(&self, frame: usize, order: usize) -> usize {
        if order == Lower::N {
            let tree = self.trees[frame / Lower::N].load();
            if tree.reserved() {
                for local in self.local {
                    let local = local.preferred.load();
                    if let Some(tree) = local.tree
                        && tree.frame / Lower::N == frame / Lower::N
                    {
                        return tree.free;
                    }
                }
                0
            } else {
                tree.free()
            }
        } else if order <= Self::MAX_ORDER {
            self.lower.free_at(frame, order)
        } else {
            0
        }
    }
}

impl LLFree<'_> {
    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, order: usize) -> Result<usize> {
        // Select local data (which can be shared between cores if we do not have enough memory)
        let local = &self.local[core];

        // Try decrementing the local counter
        let mut locked = false;
        let old = local
            .preferred
            .fetch_update(|v| v.dec_or_lock(1 << order, &mut locked));

        // If decrement succeeded
        if !locked
            && let Ok(preferred) = old
            && let Some(tree) = preferred.tree
        {
            assert!(tree.free >= (1 << order));

            // Try allocating with the lower allocator
            match self.get_lower(tree, order) {
                Ok(LocalTree { frame, free }) => {
                    if order < 6 && tree.frame / 64 != frame / 64 {
                        // Save start index for small allocations
                        if let Err(_) = local.preferred.compare_exchange(
                            Preferred::tree(tree.frame, free, false),
                            Preferred::tree(frame, free, false),
                        ) {
                            info!("start update failed");
                        }
                    }
                    Ok(frame)
                }
                Err(Error::Memory) => {
                    // Failure due to fragmentation
                    // Reset counters, reserve new entry and retry allocation
                    warn!("alloc failed o={order} => retry");
                    info!("Allocator State: {:?}", &self);
                    // Increment global to prevent race condition with concurrent reservation
                    self.trees[tree.frame / Lower::N]
                        .fetch_update(|v| Some(v.inc(1 << order, Lower::N)))
                        .unwrap();

                    self.reserve_or_wait(core, order, preferred)
                }
                Err(e) => Err(e),
            }
        } else if let Ok(preferred) = old
            && locked
        {
            // Local tree is now locked by us

            // Try sync with global counter
            if let Some(tree) = preferred.tree {
                if self.sync_with_global(&local.preferred, tree, order) {
                    warn!("sync success");
                    // Success -> Retry allocation
                    return Err(Error::Retry);
                }
                //warn!("sync failed");
            }

            // The local tree is full -> reserve a new one
            self.reserve_and_get(core, order, preferred)
        } else {
            warn!("wait");
            // Old tree is already locked
            // Wait for concurrent reservation to end
            if !spin_wait(CAS_RETRIES, || !local.preferred.load().locked) {
                warn!("Timeout reservation wait");
            }
            Err(Error::Retry)
        }
    }

    /// Allocate a frame in the lower allocator
    fn get_lower(&self, tree: LocalTree, order: usize) -> Result<LocalTree> {
        let frame = self.lower.get(tree.frame, order)?;
        Ok(LocalTree {
            frame,
            free: tree.free - (1 << order),
        })
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn sync_with_global(&self, local: &Atom<Preferred>, tree: LocalTree, order: usize) -> bool {
        let i = tree.frame / Lower::N;
        let min = 1 << order;
        if let Ok(entry) = self.trees[i].fetch_update(|e| e.sync_steal(tree.free, min)) {
            debug_assert!(tree.free + entry.free() <= Lower::N);
            local
                .fetch_update(|v| v.inc_unlock(entry.free(), Lower::N))
                .expect("Sync failed");
            true
        } else {
            false
        }
    }

    /// Try to reserve a new tree or wait for concurrent reservations to finish.
    fn reserve_or_wait(&self, core: usize, order: usize, old: Preferred) -> Result<usize> {
        let preferred = &self.local[core].preferred;

        // Set the reserved flag, locking the reservation
        if !old.locked {
            if let Ok(old) =
                preferred.fetch_update(|v| (!v.locked).then_some(Preferred { locked: true, ..v }))
                // not already locked
                && !old.locked
            {
                // Try reserve new tree
                return self.reserve_and_get(core, order, old);
            }
        }

        // Wait for concurrent reservation to end
        if !spin_wait(CAS_RETRIES, || !preferred.load().locked) {
            warn!("Timeout reservation wait");
        }
        Err(Error::Retry)
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, core: usize, order: usize, old: Preferred) -> Result<usize> {
        assert!(!old.locked); // No reservation in progress
        let preferred = &self.local[core].preferred;

        // Try reserve new tree
        let start = if let Some(LocalTree { frame: start, .. }) = old.tree {
            start / Lower::N
        } else {
            // Different initial starting point for every core
            self.trees.len() / self.local.len() * core
        };

        let drain_fn = || {
            for off in 1..self.local.len() {
                self.drain(core + off)?;
            }
            Ok(())
        };

        // Reserved a new tree an allocate a frame in it
        let cores = self.local.len();
        match self
            .trees
            .reserve(order, cores, start, |t| self.get_lower(t, order), drain_fn)
        {
            Ok(LocalTree { frame, free }) => {
                let new = Preferred::tree(frame, free, false);
                let success = self.swap_reserved(preferred, new, true);
                assert!(success);
                Ok(frame)
            }
            Err(e) => {
                warn!("Reserve failed {e:?}");
                // Rollback: Clear reserve flag
                let success = self.swap_reserved(preferred, Preferred::default(), true);
                assert!(success);
                Err(e)
            }
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    /// Returns false if the swap failed.
    fn swap_reserved(&self, local: &Atom<Preferred>, new: Preferred, expect_locked: bool) -> bool {
        assert!(!new.locked);

        if let Ok(old) = local.fetch_update(|v| (v.locked == expect_locked).then_some(new)) {
            if let Some(LocalTree { frame: start, free }) = old.tree {
                self.trees.unreserve(start / Lower::N, free);
            }
            true
        } else {
            false
        }
    }
}

impl fmt::Debug for LLFree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", Self::name())?;

        writeln!(f, "    frames: {}", self.lower.frames())?;

        writeln!(f, "    trees: {:?} ({} framesize)", self.trees, Lower::N)?;
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

/// Core-local data
#[derive(Default)]
struct Local {
    /// Local copy of the reserved tree entry
    preferred: Atom<Preferred>,
    /// Last frees heuristic
    last_frees: Atom<LastFrees>,
}

impl Local {
    /// Threshold for the number of frees after which a tree is reserved
    const F: usize = 4;

    /// Add a tree index to the history, returing if there are enough frees
    fn frees_push(&self, tree_index: usize) -> bool {
        let res = self.last_frees.fetch_update(|v| {
            if v.tree_index() == tree_index {
                // fails if there are already enough frees
                (v.count() < Self::F).then_some(v.with_count(v.count() + 1))
            } else {
                Some(LastFrees::new().with_tree_index(tree_index).with_count(1))
            }
        });
        res.is_err() // no update -> enough frees
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

#[cfg(all(test, feature = "std"))]
mod test {
    use super::Local;

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        let local = Local::default();
        let frame1 = 43;
        let i1 = frame1 / (512 * 512);
        assert!(!local.frees_push(i1));
        assert!(!local.frees_push(i1));
        assert!(!local.frees_push(i1));
        assert!(!local.frees_push(i1));
        assert!(local.frees_push(i1));
        assert!(local.frees_push(i1));
        let frame2 = 512 * 512 + 43;
        let i2 = frame2 / (512 * 512);
        assert_ne!(i1, i2);
        assert!(!local.frees_push(i2));
        assert!(!local.frees_push(i2));
        assert!(!local.frees_push(i1));
    }
}
