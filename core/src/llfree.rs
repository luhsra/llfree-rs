//! Upper allocator implementation

use core::hint::spin_loop;
use core::mem::{align_of, size_of};
use core::sync::atomic::AtomicU64;
use core::unreachable;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::{error, info, warn};

use crate::atomic::{Atom, Atomic};
use crate::entry::{LocalTree, Preferred};
use crate::lower::Lower;
use crate::util::{align_up, Align};
use crate::{Error, Result};
use crate::{MetaSize, CAS_RETRIES};

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

impl<'a> Alloc<'a> for LLFree<'a> {
    const MAX_ORDER: usize = Lower::MAX_ORDER;

    /// Return the name of the allocator.
    #[cold]
    #[cfg(feature="16K")]
    fn name() -> &'static str {
        "LLFree16K"
    }

    #[cfg(not(feature="16K"))]
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

        let local_size = cores * size_of::<Align<Local>>();
        let trees_size = Trees::<{ Lower::N }>::metadata_size(frames);
        if secondary.as_ptr() as usize % align_of::<Align>() != 0
            || secondary.len() < local_size + trees_size
        {
            error!("secondary metadata");
            return Err(Error::Memory);
        }

        if frames < Lower::N * cores {
            warn!("memory {} < {}", frames, Lower::N * cores);
            cores = frames.div_ceil(Lower::N);
        }

        // Create lower allocator
        let lower = Lower::new(frames, init, primary)?;
        //info!("Lower Allocater: {lower:#?}");
        let (local, trees) = secondary.split_at_mut(local_size);

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
        let local = cores * size_of::<Align<Local>>();
        let trees = align_up(
            Trees::<{ Lower::N }>::metadata_size(frames),
            align_of::<Align>(),
        );
        MetaSize {
            primary: Lower::metadata_size(frames),
            secondary: local + trees,
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
        let max = (self.frames() - i * Lower::N).min(Lower::N);

        // Try update own tree first
        let num_frames = 1 << order;
        if let Ok(_) = local
            .preferred
            .fetch_update(|v| v.inc(num_frames, max, |s| s / Lower::N == i))
        {
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
                // Reserve the tree that was targeted by the last N frees
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
            Err(tree) => panic!("inc failed i{i}: {tree:?} o={order}"),
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

    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }

    fn drain(&self, core: usize) -> Result<()> {
        let local = &self.local[core % self.local.len()];
        match self.swap_reserved(&local.preferred, Preferred::default(), false) {
            Err(Error::Retry) => Ok(()), // ignore cas errors
            r => r,
        }
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

    fn for_each_huge_frame<F: FnMut(usize, usize)>(&self, f: F) {
        self.lower.for_each_huge_frame(f)
    }
}

impl LLFree<'_> {
    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, order: usize) -> Result<usize> {
        // Select local data (which can be shared between cores if we do not have enough memory)
        let local = &self.local[core];

        // Update the upper counters first
        let tree = match local.preferred.fetch_update(|v| v.dec(1 << order)) {
            Ok(Preferred {
                tree: Some(tree), ..
            }) => tree,
            Ok(_) => unreachable!("Unexpected preferred"),
            Err(old) => {
                // If the local counter is large enough we do not have to reserve a new tree
                // Just update the local counter and reuse the current tree
                if let Some(tree) = old.tree {
                    if self.try_sync_with_global(&local.preferred, tree)? {
                        // Success -> Retry allocation
                        return Err(Error::Retry);
                    }
                }

                // The local tree is full -> reserve a new one
                return self.reserve_or_wait(core, order);
            }
        };

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
                    .fetch_update(|v| v.inc(1 << order, Lower::N))
                    .expect("Undo failed");

                self.reserve_or_wait(core, order)
            }
            Err(e) => Err(e),
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
    fn try_sync_with_global(&self, local: &Atom<Preferred>, tree: LocalTree) -> Result<bool> {
        let i = tree.frame / Lower::N;
        if i >= self.trees.len() {
            return Ok(false);
        }

        if let Ok(entry) = self.trees[i].fetch_update(|e| {
            (e.reserved() && e.free() + tree.free > Trees::<{ Lower::N }>::MIN_FREE)
                .then_some(e.with_free(0))
        }) {
            debug_assert!(tree.free + entry.free() <= Lower::N);

            if let Ok(_) =
                local.fetch_update(|v| v.inc(entry.free(), Lower::N, |s| s / Lower::N == i))
            {
                // Sync successfull -> retry allocation
                return Ok(true);
            } else {
                // undo global change
                self.trees[i]
                    .fetch_update(|e| Some(e.with_free(e.free() + entry.free())))
                    .expect("Undo failed");
            }
        }

        Ok(false)
    }

    /// Try to reserve a new tree or wait for concurrent reservations to finish.
    /// We directly try to allocate in the tree and continues the search if it is fragmented.
    ///
    /// If `fragmented`, prioritize less fragmented trees
    fn reserve_or_wait(&self, core: usize, order: usize) -> Result<usize> {
        let preferred = &self.local[core].preferred;

        // Set the reserved flag, locking the reservation
        if let Ok(old) =
            preferred.fetch_update(|v| (!v.locked).then_some(Preferred { locked: true, ..v }))
        {
            // Try reserve new tree
            return self.reserve_and_get(core, order, old);
        }

        // Wait for concurrent reservation to end
        for _ in 0..CAS_RETRIES {
            let old = preferred.load();
            if !old.locked {
                let Ok(old) = preferred.fetch_update(|v| v.dec(1 << order)) else {
                    error!("Decrement failed {old:?}");
                    return Err(Error::Retry);
                };

                // Try allocation on new tree
                let Some(LocalTree { frame: start, .. }) = old.tree else {
                    unreachable!("invalid preferred");
                };
                return match self.lower.get(start, order) {
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
    fn reserve_and_get(&self, core: usize, order: usize, old: Preferred) -> Result<usize> {
        let local = &self.local[core].preferred;

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
        let LocalTree { frame, free } =
            match self
                .trees
                .reserve(order, cores, start, |t| self.get_lower(t, order), drain_fn)
            {
                Ok(entry) => entry,
                Err(e) => {
                    // Rollback: Clear reserve flag
                    local
                        .fetch_update(|v| v.locked.then_some(Preferred { locked: false, ..v }))
                        .expect("unexpected unlock");
                    return Err(e);
                }
            };

        match self.swap_reserved(local, Preferred::tree(frame, free, false), true) {
            Ok(_) => Ok(frame),
            Err(Error::Retry) => panic!("unexpected unlock"),

            Err(e) => Err(e),
        }
    }

    /// Reserve an entry for bulk frees
    fn reserve_for_put(&self, free: usize, local: &Atom<Preferred>, i: usize) -> Result<bool> {
        let entry = Preferred::tree(i * Lower::N, free, false);
        match self.swap_reserved(local, entry, false) {
            Ok(_) => Ok(true),
            Err(Error::Retry) => {
                warn!("rollback {i}");
                // Rollback reservation
                let max = (self.frames() - i * Lower::N).min(Lower::N);
                self.trees[i]
                    .fetch_update(|v| v.unreserve_add(free, max))
                    .expect("put - reservation rollback failed");
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    fn swap_reserved(
        &self,
        local: &Atom<Preferred>,
        new: Preferred,
        expect_locked: bool,
    ) -> Result<()> {
        debug_assert!(!new.locked);

        let old = local
            .fetch_update(|v| (v.locked == expect_locked).then_some(new))
            .map_err(|_| Error::Retry)?;

        if let Some(LocalTree { frame: start, free }) = old.tree {
            self.trees
                .unreserve(start / Lower::N, free, self.frames())?;
        }
        Ok(())
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
