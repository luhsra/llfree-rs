//! Upper allocator implementation

use core::{fmt, slice};

use log::{error, info, warn};
use spin::mutex::SpinMutex;

use crate::local::{Local, LocalTree};
use crate::lower::Lower;
use crate::trees::{Kind, Trees};
use crate::util::{size_of_slice, Align, FmtFn};
use crate::{
    Alloc, Error, Flags, Init, MetaData, MetaSize, Result, HUGE_FRAMES, HUGE_ORDER, MAX_ORDER,
    RETRIES, TREE_FRAMES,
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

impl<'a> Alloc<'a> for LLFree<'a> {
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
    fn new(mut cores: usize, frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self> {
        info!(
            "initializing c={cores} f={frames} {:?} {:?} {:?}",
            meta.local.as_ptr_range(),
            meta.trees.as_ptr_range(),
            meta.lower.as_ptr_range()
        );
        assert!(meta.valid(Self::metadata_size(cores, frames)));

        if frames < TREE_FRAMES * cores {
            warn!("memory {} < {}", frames, TREE_FRAMES * cores);
            cores = frames.div_ceil(TREE_FRAMES);
        }

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Init per-cpu data
        let local = unsafe { slice::from_raw_parts_mut(meta.local.as_mut_ptr().cast(), cores) };
        local.fill_with(Default::default);

        // Init tree array
        let trees = Trees::new(frames, meta.trees, |start| lower.free_in_tree(start));

        Ok(Self {
            local,
            lower,
            trees,
        })
    }

    fn metadata_size(cores: usize, frames: usize) -> MetaSize {
        let cores = cores.clamp(1, frames.div_ceil(TREE_FRAMES));
        MetaSize {
            local: size_of_slice::<Align<SpinMutex<Local>>>(cores),
            trees: Trees::metadata_size(frames),
            lower: Lower::metadata_size(frames),
        }
    }

    fn metadata(&mut self) -> MetaData<'a> {
        let m = Self::metadata_size(self.local.len(), self.lower.frames());
        MetaData {
            local: unsafe {
                slice::from_raw_parts_mut(self.local.as_ptr().cast_mut().cast(), m.local)
            },
            trees: self.trees.metadata(),
            lower: self.lower.metadata(),
        }
    }

    fn get(&self, core: usize, flags: Flags) -> Result<usize> {
        if flags.order() > MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..RETRIES {
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
            if let Some(preferred) = local.preferred_mut(Kind::Huge)
                && preferred.frame() / TREE_FRAMES == i
            {
                preferred.set_free(preferred.free() + num_frames);
                preferred.set_huge(preferred.huge() + huge);
                return Ok(());
            }
        } else {
            // Might be movable or fixed
            for kind in [Kind::Movable, Kind::Fixed] {
                if let Some(preferred) = &mut local.preferred_mut(kind)
                    && preferred.frame() / TREE_FRAMES == i
                {
                    preferred.set_free(preferred.free() + num_frames);
                    preferred.set_huge(preferred.huge() + huge);
                    return Ok(());
                }
            }
        }

        // Increment or reserve the tree
        if let Some(tree) = self.trees.inc_or_reserve(i, num_frames, huge, may_reserve) {
            // Change preferred tree to speedup future frees
            let entry = LocalTree::with(
                i * TREE_FRAMES,
                tree.free() + num_frames,
                tree.huge() + huge,
            );
            let kind = flags.with_movable(tree.kind() == Kind::Movable).into();
            self.swap_reserved(local.preferred_mut(kind), Some(entry), kind);
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
            for kind in [Kind::Fixed, Kind::Movable, Kind::Huge] {
                self.swap_reserved(&mut local.preferred_mut(kind), None, kind);
            }
        }
        Ok(())
    }

    fn free_frames(&self) -> usize {
        // Global array
        let mut frames = self.trees.free_frames();
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            if let Some(local) = local.try_lock() {
                for kind in [Kind::Fixed, Kind::Movable, Kind::Huge] {
                    if let Some(tree) = local.preferred(kind) {
                        frames += tree.free();
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
                for kind in [Kind::Fixed, Kind::Movable, Kind::Huge] {
                    if let Some(tree) = local.preferred(kind) {
                        huge += tree.huge();
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
                        for kind in [Kind::Fixed, Kind::Movable, Kind::Huge] {
                            if let Some(tree) = local.preferred(kind) {
                                if tree.frame() / TREE_FRAMES == frame / TREE_FRAMES {
                                    return global.free() + tree.free();
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
            for kind in [Kind::Movable, Kind::Fixed, Kind::Huge] {
                if let Some(tree) = local.preferred(kind) {
                    let global = self.trees.get(tree.frame() / TREE_FRAMES);
                    let (free, huge) = self.lower.free_in_tree(tree.frame());
                    assert_eq!(tree.free() + global.free(), free);
                    assert_eq!(tree.huge() + global.huge(), huge);
                    reserved -= 1;
                }
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn lower_get(&self, mut tree: LocalTree, flags: Flags) -> Result<LocalTree> {
        let (frame, huge) = self.lower.get(tree.frame(), flags)?;
        tree.set_frame(frame);
        tree.set_free(tree.free() - (1 << flags.order()));
        let huge = (huge as usize).max((1 << flags.order()) / HUGE_FRAMES);
        if huge > tree.huge() {
            assert!(self.sync_with_global(&mut tree, flags.order()));
        }
        tree.set_huge(tree.huge() - huge);
        Ok(tree)
    }

    /// Steal a tree from another core
    fn steal_tree(&self, core: usize, flags: Flags) -> Result<LocalTree> {
        for i in 1..self.local.len() {
            let target_core = (core + i) % self.local.len();
            if let Some(mut target) = self.local[target_core].try_lock()
                && let Some(tree) = target.preferred_mut(flags.into())
                && tree.free() >= (1 << flags.order())
                && tree.huge() >= (1 << flags.order()) / HUGE_FRAMES
                && let Ok(new) = self.lower_get(*tree, flags)
            {
                assert!(new.frame() / TREE_FRAMES == tree.frame() / TREE_FRAMES);
                *target.preferred_mut(flags.into()) = None;
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
        if let Some(tree) = local.preferred_mut(flags.into())
            && tree.free() >= 1 << flags.order()
            && tree.huge() >= min_huge
        {
            match self.lower_get(*tree, flags) {
                Ok(new) => {
                    assert!(new.frame() / TREE_FRAMES == tree.frame() / TREE_FRAMES);
                    *tree = new;
                    Ok(new.frame())
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
            if let Some(tree) = local.preferred_mut(flags.into()) {
                if self.sync_with_global(tree, flags.order()) {
                    // Success -> Retry allocation
                    return Err(Error::Retry);
                }
                //warn!("sync failed");
            }

            // The local tree is full -> reserve a new one
            self.reserve_and_get(&mut local, core, flags)
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn sync_with_global(&self, tree: &mut LocalTree, order: usize) -> bool {
        let i = tree.frame() / TREE_FRAMES;
        let min = Trees::MIN_FREE.saturating_sub(tree.free());
        let min_huge = ((1 << order) / HUGE_FRAMES).saturating_sub(tree.huge());
        if let Some(global) = self.trees.sync(i, min, min_huge) {
            tree.set_free(tree.free() + global.free());
            tree.set_huge(tree.huge() + global.huge());
            true
        } else {
            false
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, local: &mut Local, core: usize, flags: Flags) -> Result<usize> {
        // Try reserve new tree
        let preferred = local.preferred_mut(flags.into());
        let start = if let Some(tree) = *preferred {
            tree.frame() / TREE_FRAMES
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
                Ok(new.frame())
            }
            Err(Error::Memory) => {
                // Fall back to stealing from other cores
                let new = self.steal_tree(core, flags)?;
                self.swap_reserved(preferred, Some(new), flags.into());
                Ok(new.frame())
            }
            Err(e) => Err(e),
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    /// Returns false if the swap failed.
    fn swap_reserved(&self, preferred: &mut Option<LocalTree>, new: Option<LocalTree>, kind: Kind) {
        let old_tree = core::mem::replace(preferred, new);
        if let Some(tree) = old_tree {
            self.trees
                .unreserve(tree.frame() / TREE_FRAMES, tree.free(), tree.huge(), kind);
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
