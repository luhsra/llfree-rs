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
        let local = size_of_slice::<Align<Local>>(cores);
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

    fn get(&self, core: usize, order: usize) -> Result<usize> {
        if order > Lower::MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..CAS_RETRIES {
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
        let mut local = self.local[core % self.local.len()].lock();

        // Update the put-reserve heuristic
        let may_reserve = local.frees_push(i);

        // Try update own tree first
        let num_frames = 1usize << order;
        if let Some(preferred) = &mut local.preferred
            && preferred.frame / Lower::N == i
        {
            preferred.free += num_frames as u16;
            assert!(preferred.free <= Lower::N as u16);
            return Ok(());
        }

        // Increment or reserve the tree
        if let Some(free) = self.trees.inc_or_reserve(i, num_frames, may_reserve) {
            // Change preferred tree to speedup future frees
            let entry = LocalTree {
                frame: i * Lower::N,
                free: (free + num_frames) as _,
            };
            self.swap_reserved(&mut local.preferred, Some(entry));
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
            self.swap_reserved(&mut local.preferred, None);
        }
        Ok(())
    }

    fn free_frames(&self) -> usize {
        // Global array
        let mut frames = self.trees.total_free();
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            if let Some(local) = local.try_lock() {
                if let Some(tree) = local.preferred {
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
                    if let Some(tree) = local.preferred
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
    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, order: usize) -> Result<usize> {
        let mut local = self.local[core].lock();

        // Try decrementing the local counter
        if let Some(tree) = &mut local.preferred
            && tree.free >= (1 << order) as u16
        {
            match self.lower.get(tree.frame, order) {
                Ok(frame) => {
                    assert!(frame / Lower::N == tree.frame / Lower::N);
                    tree.free -= 1 << order;
                    tree.frame = frame;
                    Ok(frame)
                }
                Err(Error::Memory) => {
                    // Failure due to fragmentation
                    // Reset counters, reserve new entry and retry allocation
                    warn!("alloc failed o={order} => retry");
                    self.reserve_and_get(&mut local.preferred, core, order)
                }
                Err(e) => Err(e),
            }
        } else {
            // Try sync with global counter
            if let Some(tree) = &mut local.preferred {
                if self.sync_with_global(tree) {
                    // Success -> Retry allocation
                    return Err(Error::Retry);
                }
            }

            // The local tree is full -> reserve a new one
            self.reserve_and_get(&mut local.preferred, core, order)
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
    fn reserve_and_get(
        &self,
        preferred: &mut Option<LocalTree>,
        core: usize,
        order: usize,
    ) -> Result<usize> {
        // Try reserve new tree
        let start = if let Some(tree) = *preferred {
            tree.frame / Lower::N
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
            .reserve(order, cores, start, |t| self.lower.get(t, order), drain_fn)
        {
            Ok(mut new) => {
                new.free -= 1 << order;
                self.swap_reserved(preferred, Some(new));
                Ok(new.frame)
            }
            Err(e) => {
                // Rollback: Clear reserve flag
                self.swap_reserved(preferred, None);
                Err(e)
            }
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    /// Returns false if the swap failed.
    fn swap_reserved(&self, preferred: &mut Option<LocalTree>, new: Option<LocalTree>) {
        let old_tree = core::mem::replace(preferred, new);
        if let Some(LocalTree { frame, free }) = old_tree {
            self.trees.unreserve(frame / Lower::N, free as _);
        }
    }
}

impl fmt::Debug for LLFree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", Self::name())?;

        writeln!(f, "    frames: {}", self.lower.frames())?;

        writeln!(f, "    trees: {:?} ({} frames)", self.trees, Lower::N)?;
        let free_frames = self.free_frames();
        let free_huge_frames = self.free_huge_frames();
        writeln!(
            f,
            "    free frames: {free_frames} ({free_huge_frames} huge, {} trees)",
            free_frames.div_ceil(Lower::N)
        )?;

        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.lock().preferred)?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

/// Core-local data
#[derive(Default, Debug)]
struct Local {
    /// Local copy of the reserved tree entry
    preferred: Option<LocalTree>,
    /// Tree index of the last freed frame
    last_idx: usize,
    /// Last frees counter
    last_frees: u8,
}

impl Local {
    /// Threshold for the number of frees after which a tree is reserved
    const F: u8 = 4;

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
