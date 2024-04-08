//! Upper allocator implementation

use core::mem::align_of;
use core::{fmt, slice};

use log::{error, info, warn};
use spin::mutex::SpinMutex;

use crate::entry::{Kind, LocalTree};
use crate::lower::Lower;
use crate::trees::Trees;
use crate::util::{size_of_slice, Align, FmtFn};
use crate::{
    Alloc, Error, Flags, Init, MetaSize, Result, CAS_RETRIES, HUGE_FRAMES, HUGE_ORDER, MAX_ORDER,
    TREE_FRAMES, TREE_HUGE,
};

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
    pub trees: Trees<'a>,
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
        let cores = cores.clamp(1, frames.div_ceil(TREE_FRAMES));
        let local = size_of_slice::<Align<SpinMutex<Local>>>(cores);
        let tree = Trees::metadata_size(frames);
        Self { local, tree }
    }
}

impl<'a> Alloc<'a> for LLFree<'a> {
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

        if frames < TREE_FRAMES * cores {
            warn!("memory {} < {}", frames, TREE_FRAMES * cores);
            cores = frames.div_ceil(TREE_FRAMES);
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
        if flags.order() > MAX_ORDER {
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
        // Put usually does not know about movability
        flags.set_movable(false);

        // First free the frame in the lower allocator
        let huge = self.lower.put(frame, flags)?;
        // Could be multiple huge frames depending on the allocation size
        let huge = (huge as usize).max((1 << flags.order()) / HUGE_FRAMES);

        // Then update local / global counters
        let i = frame / TREE_FRAMES;
        let mut local = self.local[core % self.local.len()].lock();

        // Update the put-reserve heuristic
        let may_reserve = local.frees_push(i);

        // Try update own trees first
        let num_frames = 1usize << flags.order();
        if flags.order() >= HUGE_ORDER {
            if let Some(preferred) = &mut local.huge
                && preferred.frame / TREE_FRAMES == i
            {
                preferred.free += num_frames;
                preferred.huge += huge;
                assert!(preferred.free <= TREE_FRAMES && preferred.huge <= TREE_HUGE);
                return Ok(());
            }
        } else {
            // Might be movable or fixed
            for kind in [Kind::Movable, Kind::Fixed] {
                if let Some(preferred) = &mut local.preferred(kind)
                    && preferred.frame / TREE_FRAMES == i
                {
                    preferred.free += num_frames;
                    preferred.huge += huge;
                    assert!(preferred.free <= TREE_FRAMES && preferred.huge <= TREE_HUGE);
                    return Ok(());
                }
            }
        }

        // Increment or reserve the tree
        if let Some(tree) = self.trees.inc_or_reserve(i, num_frames, huge, may_reserve) {
            // Change preferred tree to speedup future frees
            let entry = LocalTree::new(
                i * TREE_FRAMES,
                tree.free() + num_frames,
                tree.huge() + huge,
            );
            let kind = flags.with_movable(tree.kind() == Kind::Movable).into();
            self.swap_reserved(local.preferred(kind), Some(entry), kind);
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
            self.swap_reserved(&mut local.fixed, None, Kind::Fixed);
            self.swap_reserved(&mut local.movable, None, Kind::Movable);
            self.swap_reserved(&mut local.huge, None, Kind::Huge);
        }
        Ok(())
    }

    fn free_frames(&self) -> usize {
        // Global array
        let mut frames = self.trees.free_frames();
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            if let Some(local) = local.try_lock() {
                for tree in [local.movable, local.fixed, local.huge] {
                    if let Some(tree) = tree {
                        frames += tree.free as usize;
                    }
                }
            }
        }
        frames
    }

    fn free_huge(&self) -> usize {
        // Global array
        let mut huge = self.trees.free_huge();
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            if let Some(local) = local.try_lock() {
                for tree in [local.movable, local.fixed, local.huge] {
                    if let Some(tree) = tree {
                        huge += tree.huge as usize;
                    }
                }
            }
        }
        huge
    }

    fn free_at(&self, frame: usize, order: usize) -> usize {
        if order == TREE_FRAMES {
            let global = self.trees.get(frame / TREE_FRAMES);
            if global.reserved() {
                for local in self.local {
                    if let Some(local) = local.try_lock() {
                        for tree in [local.movable, local.fixed, local.huge] {
                            if let Some(tree) = tree {
                                if tree.frame / TREE_FRAMES == frame / TREE_FRAMES {
                                    return global.free() + tree.free;
                                }
                            }
                        }
                    }
                }
            }
            global.free()
        } else if order <= MAX_ORDER {
            self.lower.free_at(frame, order)
        } else {
            0
        }
    }

    fn validate(&self) {
        warn!("validate");
        assert_eq!(self.free_frames(), self.lower.free_frames());
        assert_eq!(self.free_huge(), self.lower.free_huge());
        let mut reserved = 0;
        for (i, tree) in self.trees.entries.iter().enumerate() {
            let tree = tree.load();
            if !tree.reserved() {
                let (free, huge) = self.lower.free_in_tree(i * TREE_FRAMES);
                assert_eq!(tree.free(), free);
                assert_eq!(tree.huge(), huge);
            } else {
                reserved += 1;
            }
        }
        for local in self.local {
            let local = local.lock();
            for tree in [local.movable, local.fixed, local.huge] {
                if let Some(tree) = tree {
                    let global = self.trees.get(tree.frame / TREE_FRAMES);
                    let (free, huge) = self.lower.free_in_tree(tree.frame);
                    assert_eq!(tree.free + global.free(), free);
                    assert_eq!(tree.huge + global.huge(), huge);
                    reserved -= 1;
                }
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn lower_get(&self, mut tree: LocalTree, flags: Flags) -> Result<LocalTree> {
        let (frame, huge) = self.lower.get(tree.frame, flags)?;
        tree.frame = frame;
        tree.free -= 1 << flags.order();
        let huge = (huge as usize).max((1 << flags.order()) / HUGE_FRAMES);
        if huge > tree.huge {
            assert!(self.sync_with_global(&mut tree, flags.order()));
        }
        tree.huge -= huge;
        Ok(tree)
    }

    /// Steal a tree from another core
    fn steal_tree(&self, core: usize, flags: Flags) -> Result<LocalTree> {
        for i in 1..self.local.len() {
            let target_core = (core + i) % self.local.len();
            if let Some(mut target) = self.local[target_core].try_lock()
                && let Some(tree) = target.preferred(flags.into())
                && tree.free >= (1 << flags.order())
                && tree.huge >= (1 << flags.order()) / HUGE_FRAMES
                && let Ok(new) = self.lower_get(*tree, flags)
            {
                assert!(new.frame / TREE_FRAMES == tree.frame / TREE_FRAMES);
                *target.preferred(flags.into()) = None;
                return Ok(new);
            }
        }
        Err(Error::Memory)
    }

    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, flags: Flags) -> Result<usize> {
        let mut local = self.local[core].lock();

        let min_huge = (1 << flags.order()) / HUGE_FRAMES;

        // Try decrementing the local counter
        if let Some(tree) = local.preferred(flags.into())
            && tree.free >= 1 << flags.order()
            && tree.huge >= min_huge
        {
            match self.lower_get(*tree, flags) {
                Ok(new) => {
                    assert!(new.frame / TREE_FRAMES == tree.frame / TREE_FRAMES);
                    *tree = new;
                    Ok(new.frame)
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
            if let Some(tree) = local.preferred(flags.into()) {
                if self.sync_with_global(tree, flags.order()) {
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
    fn sync_with_global(&self, tree: &mut LocalTree, order: usize) -> bool {
        let i = tree.frame / TREE_FRAMES;
        let min = Trees::MIN_FREE.saturating_sub(tree.free);
        let min_huge = ((1 << order) / HUGE_FRAMES).saturating_sub(tree.huge);
        if let Some(global) = self.trees.sync(i, min, min_huge) {
            tree.free += global.free();
            tree.huge += global.huge();
            assert!(tree.free <= TREE_FRAMES && tree.huge <= TREE_HUGE);
            true
        } else {
            false
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, local: &mut Local, core: usize, flags: Flags) -> Result<usize> {
        // Try reserve new tree
        let preferred = local.preferred(flags.into());
        let start = if let Some(tree) = *preferred {
            tree.frame / TREE_FRAMES
        } else {
            // Different initial starting point for every core
            self.trees.len() / self.local.len() * core
        };

        // Reserved a new tree an allocate a frame in it
        let cores = self.local.len();
        match self
            .trees
            .reserve(cores, start, flags, |t, f| self.lower_get(t, f))
        {
            Ok(new) => {
                self.swap_reserved(preferred, Some(new), flags.into());
                Ok(new.frame)
            }
            Err(Error::Memory) => {
                // Fall back to stealing from other cores
                let new = self.steal_tree(core, flags)?;
                self.swap_reserved(preferred, Some(new), flags.into());
                Ok(new.frame)
            }
            Err(e) => Err(e),
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    /// Returns false if the swap failed.
    fn swap_reserved(&self, preferred: &mut Option<LocalTree>, new: Option<LocalTree>, kind: Kind) {
        let old_tree = core::mem::replace(preferred, new);
        if let Some(LocalTree { frame, free, huge }) = old_tree {
            self.trees.unreserve(frame / TREE_FRAMES, free, huge, kind);
        }
    }
}

impl fmt::Debug for LLFree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let huge = self.frames() / (1 << HUGE_ORDER);
        let free = self.free_frames();
        let free_huge = self.free_huge();

        f.debug_struct(Self::name())
            .field(
                "managed",
                &FmtFn(|f| write!(f, "{} frames ({huge} huge)", self.frames())),
            )
            .field(
                "free",
                &FmtFn(|f| write!(f, "{free} frames ({free_huge} huge)")),
            )
            .field(
                "trees",
                &FmtFn(|f| write!(f, "{:?} (N={})", self.trees, TREE_FRAMES)),
            )
            .field(
                "locals",
                &FmtFn(|f| {
                    let mut f = f.debug_list();
                    for local in self.local.iter() {
                        let local = local.lock();
                        f.entry(&local);
                    }
                    f.finish()
                }),
            )
            .finish()?;
        Ok(())
    }
}

/// Core-local data
#[derive(Default, Debug)]
struct Local {
    /// Reserved tree for movable (user) allocations
    movable: Option<LocalTree>,
    /// Reserved tree for immovable (kernel) allocations
    fixed: Option<LocalTree>,
    /// Reserved tree for huge allocations
    huge: Option<LocalTree>,
    /// Tree index of the last freed frame
    last_idx: usize,
    /// Last frees counter
    last_frees: u8,
}

impl Local {
    /// Threshold for the number of frees after which a tree is reserved
    const F: u8 = 4;

    fn preferred(&mut self, kind: Kind) -> &mut Option<LocalTree> {
        match kind {
            Kind::Huge => &mut self.huge,
            Kind::Movable => &mut self.movable,
            Kind::Fixed => &mut self.fixed,
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
