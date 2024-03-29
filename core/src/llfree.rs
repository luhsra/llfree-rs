//! Upper allocator implementation

use core::mem::align_of;
use core::{fmt, slice};

use log::{error, info, warn};
use spin::mutex::SpinMutex;

use super::trees::Trees;
use super::{Alloc, Init};
use crate::entry::LocalTree;
use crate::lower::Lower;
use crate::util::{size_of_slice, Align};
use crate::{Error, Flags, MetaSize, Result, CAS_RETRIES};

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
    /// CPU local data
    ///
    /// Other CPUs can access this if they drain cores.
    /// Also, these are shared between CPUs if we have more cores than trees.
    local: &'a [Align<SpinMutex<Local>>],
    /// Metadata of the lower alloc
    pub lower: Lower<'a>,
    /// Manages the allocators trees
    pub trees: Trees<'a, { Lower::N }>,
}

unsafe impl Send for LLFree<'_> {}
unsafe impl Sync for LLFree<'_> {}

/// Size of the dynamic metadata
struct Metadata {
    local: usize,
    tree: usize,
}

impl Metadata {
    fn new(cores: usize, frames: usize) -> Self {
        let cores = cores.clamp(1, frames.div_ceil(Lower::N));
        let local = size_of_slice::<Align<SpinMutex<Local>>>(cores);
        let tree = Trees::<{ Lower::N }>::metadata_size(frames);
        Self { local, tree }
    }
}

impl<'a> Alloc<'a> for LLFree<'a> {
    const MAX_ORDER: usize = Lower::MAX_ORDER;
    const HUGE_ORDER: usize = Lower::HUGE_ORDER;

    /// Return the name of the allocator.
    #[cold]
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
            || secondary.len() < m.local + m.tree
        {
            error!("secondary metadata");
            return Err(Error::Memory);
        }

        // Create lower allocator
        let lower = Lower::new(frames, init, primary)?;

        let (local, trees) = secondary.split_at_mut(m.local);

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
            secondary: m.local + m.tree,
        }
    }

    fn metadata(&mut self) -> (&'a mut [u8], &'a mut [u8]) {
        let m = Self::metadata_size(self.local.len(), self.lower.frames());
        let secondary = unsafe {
            slice::from_raw_parts_mut(self.local.as_ptr().cast_mut().cast(), m.secondary)
        };
        (self.lower.metadata(), secondary)
    }

    fn get(&self, core: usize, flags: Flags) -> Result<usize> {
        if flags.order() > Lower::MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..CAS_RETRIES {
            match self.get_inner(core, flags) {
                Ok(frame) => return Ok(frame),
                Err(Error::Retry) => continue,
                Err(e) => return Err(e),
            }
        }
        error!("Exceeding retries");
        Err(Error::Memory)
    }

    fn put(&self, core: usize, frame: usize, mut flags: Flags) -> Result<()> {
        if frame >= self.lower.frames() {
            error!("invalid frame number");
            return Err(Error::Memory);
        }
        // Ignore on small memory sizes
        if flags.movable() && self.trees.len() <= self.local.len() / 2 {
            flags.set_movable(false);
        }

        // First free the frame in the lower allocator
        self.lower.put(frame, flags)?;

        // Then update local / global counters
        let i = frame / Lower::N;
        let mut local = self.local[core % self.local.len()].lock();

        // Update the put-reserve heuristic
        let may_reserve = local.frees_push(i);

        // Try update own trees first
        let num_frames = 1usize << flags.order();
        if let Some(preferred) = &mut local.movable
            && preferred.frame / Lower::N == i
        {
            preferred.free += num_frames as u16;
            assert!(preferred.free <= Lower::N as u16);
            return Ok(());
        }
        if let Some(preferred) = &mut local.immovable
            && preferred.frame / Lower::N == i
        {
            preferred.free += num_frames as u16;
            assert!(preferred.free <= Lower::N as u16);
            return Ok(());
        }

        // Increment or reserve the tree
        if let Some(tree) = self.trees.inc_or_reserve(i, num_frames, may_reserve) {
            // Change preferred tree to speedup future frees
            let entry = LocalTree {
                frame: i * Lower::N,
                free: (tree.free() + num_frames) as _,
            };
            self.swap_reserved(local.preferred(tree.movable()), Some(entry), tree.movable());
        }
        Ok(())
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
        if let Some(mut local) = self.local[core % self.local.len()].try_lock() {
            self.swap_reserved(&mut local.movable, None, true);
            self.swap_reserved(&mut local.immovable, None, false);
        }
        Ok(())
    }

    fn free_frames(&self) -> usize {
        // Global array
        let mut frames = self.trees.total_free();
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            if let Some(local) = local.try_lock() {
                if let Some(tree) = local.movable {
                    frames += tree.free as usize;
                }
                if let Some(tree) = local.immovable {
                    frames += tree.free as usize;
                }
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
            let tree = self.trees.get(frame / Lower::N);
            if tree.reserved() {
                for local in self.local {
                    let local = local.lock();
                    if let Some(tree) = local.movable
                        && tree.frame / Lower::N == frame / Lower::N
                    {
                        return tree.free as _;
                    }
                    if let Some(tree) = local.immovable
                        && tree.frame / Lower::N == frame / Lower::N
                    {
                        return tree.free as _;
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
    /// Steal a tree from another core
    fn steal_tree(&self, core: usize, flags: Flags) -> Result<LocalTree> {
        for i in 1..self.local.len() {
            let target_core = (core + i) % self.local.len();
            if let Some(mut target) = self.local[target_core].try_lock()
                && let Some(tree) = target.preferred(flags.movable())
                && tree.free >= (1 << flags.order())
                && let Ok(frame) = self.lower.get(tree.frame, flags)
            {
                assert!(frame / Lower::N == tree.frame / Lower::N);
                let free = tree.free;
                *target.preferred(flags.movable()) = None;
                return Ok(LocalTree::new(frame, free));
            }
        }
        Err(Error::Memory)
    }

    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, flags: Flags) -> Result<usize> {
        let mut local = self.local[core].lock();

        // Try decrementing the local counter
        if let Some(tree) = local.preferred(flags.movable())
            && tree.free >= (1 << flags.order()) as u16
        {
            match self.lower.get(tree.frame, flags) {
                Ok(frame) => {
                    assert!(frame / Lower::N == tree.frame / Lower::N);
                    tree.free -= 1 << flags.order();
                    tree.frame = frame;
                    Ok(frame)
                }
                Err(Error::Memory) => {
                    // Failure due to fragmentation
                    // Reset counters, reserve new entry and retry allocation
                    info!("alloc failed {flags:?} => retry");
                    self.reserve_and_get(&mut local, core, flags)
                }
                Err(e) => Err(e),
            }
        } else {
            // Try sync with global counter
            if let Some(tree) = local.preferred(flags.movable()) {
                if self.sync_with_global(tree) {
                    // Success -> Retry allocation
                    return Err(Error::Retry);
                }
            }

            // The local tree is full -> reserve a new one
            self.reserve_and_get(&mut local, core, flags)
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn sync_with_global(&self, tree: &mut LocalTree) -> bool {
        let i = tree.frame / Lower::N;
        let min = Trees::<{ Lower::N }>::MIN_FREE;
        if let Some(free) = self.trees.sync(i, min.saturating_sub(tree.free as _)) {
            tree.free += free as u16;
            assert!(tree.free <= Lower::N as u16);
            true
        } else {
            false
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, local: &mut Local, core: usize, flags: Flags) -> Result<usize> {
        // Try reserve new tree
        let preferred = local.preferred(flags.movable());
        let start = if let Some(tree) = *preferred {
            tree.frame / Lower::N
        } else {
            // Different initial starting point for every core
            self.trees.len() / self.local.len() * core
        };

        // Reserved a new tree an allocate a frame in it
        let cores = self.local.len();
        match self.trees.reserve(
            cores,
            start,
            flags,
            |s, f| self.lower.get(s, f),
            |f| self.steal_tree(core, f),
        ) {
            Ok(mut new) => {
                new.free -= 1 << flags.order();
                self.swap_reserved(preferred, Some(new), flags.movable());
                Ok(new.frame)
            }
            Err(e) => Err(e),
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    /// Returns false if the swap failed.
    fn swap_reserved(
        &self,
        preferred: &mut Option<LocalTree>,
        new: Option<LocalTree>,
        movable: bool,
    ) {
        let old_tree = core::mem::replace(preferred, new);
        if let Some(LocalTree { frame, free }) = old_tree {
            self.trees.unreserve(frame / Lower::N, free as _, movable);
        }
    }
}

impl fmt::Debug for LLFree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", Self::name())?;
        let huge = self.frames() / (1 << Lower::HUGE_ORDER);
        writeln!(f, "    managed: {} frames ({huge} huge)", self.frames())?;
        let free = self.free_frames();
        let free_huge = self.free_huge_frames();
        writeln!(f, "    free: {free} frames ({free_huge} huge)")?;
        writeln!(f, "    trees: {:?} (N={})", self.trees, Lower::N)?;
        for (t, local) in self.local.iter().enumerate() {
            let local = local.lock();
            writeln!(
                f,
                "    L{t:>2}: m={:?} f={:?}",
                local.movable, local.immovable
            )?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

/// Core-local data
#[derive(Default, Debug)]
struct Local {
    /// Reserved tree for movable (user) allocations
    movable: Option<LocalTree>,
    /// Reserved tree for immovable (kernel) allocations
    immovable: Option<LocalTree>,
    /// Tree index of the last freed frame
    last_idx: usize,
    /// Last frees counter
    last_frees: u8,
}

impl Local {
    /// Threshold for the number of frees after which a tree is reserved
    const F: u8 = 4;

    fn preferred(&mut self, movable: bool) -> &mut Option<LocalTree> {
        if movable {
            &mut self.movable
        } else {
            &mut self.immovable
        }
    }

    /// Add a tree index to the history, returing if there are enough frees
    fn frees_push(&mut self, tree_idx: usize) -> bool {
        if self.last_idx == tree_idx {
            if self.last_frees >= Self::F {
                return true;
            }
            self.last_frees += 1;
        } else {
            self.last_idx = tree_idx;
            self.last_frees = 0;
        }
        false
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use super::Local;

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        let mut local = Local::default();
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
