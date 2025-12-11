//! Upper allocator implementation

use core::ops::Range;
use core::{fmt, slice};

use log::{error, info, warn};

use crate::atomic::Atom;
use crate::local::{Local, LocalTree};
use crate::lower::Lower;
use crate::trees::{Kind, Tree, Trees};
use crate::util::{Align, FmtFn, align_down, size_of_slice};
use crate::{
    Alloc, BITFIELD_ROW, Error, Flags, HUGE_FRAMES, HUGE_ORDER, Init, MAX_ORDER, MetaData,
    MetaSize, RETRIES, Result, Stats, TREE_FRAMES, TREE_ORDER,
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
    local: &'a [Align<Local>],
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
    fn name() -> &'static str {
        if cfg!(feature = "16K") {
            "LLFree16K"
        } else {
            "LLFree"
        }
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
        if cores > Self::max_cores(frames) {
            warn!(
                "Reducing core count to {}, due to insufficient CPU-local trees",
                Self::max_cores(frames)
            );
            cores = Self::max_cores(frames);
        }
        ensure!(
            Error::Initialization;
            meta.valid(Self::metadata_size(cores, frames)),
            "Invalid metadata"
        );

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Init per-cpu data
        let local = unsafe { slice::from_raw_parts_mut(meta.local.as_mut_ptr().cast(), cores) };
        local.fill_with(Default::default);

        // Init tree array
        let tree_init = if init != Init::None {
            Some(|start| lower.stats_at(start, TREE_FRAMES.ilog2() as _).free_frames)
        } else {
            None
        };
        let trees = Trees::new(frames, meta.trees, tree_init);

        Ok(Self {
            local,
            lower,
            trees,
        })
    }

    fn metadata_size(cores: usize, frames: usize) -> MetaSize {
        let cores = cores.min(Self::max_cores(frames));
        MetaSize {
            local: size_of_slice::<Align<Local>>(cores),
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

    fn get(&self, core: usize, frame: Option<usize>, flags: Flags) -> Result<usize> {
        self.check(0, flags.order())?;
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Different starting points for each core
        let mut start_idx = self.trees.len() / self.local.len() * core;

        // Try local reservation first (if enough memory)
        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..RETRIES {
            match self.get_from_local(core, flags.into(), flags.order(), frame) {
                Ok(frame) => return Ok(frame),
                Err((Error::Retry, _)) => {}
                Err((Error::Memory, old)) => {
                    if old.present() {
                        start_idx = old.frame() / TREE_FRAMES;
                    }
                    break;
                }
                Err((e, _)) => return Err(e),
            }
        }

        // Global search
        if let Some(frame) = frame {
            // Do not reserve trees if a specific frame is allocated
            if let Ok(_) = self.trees.entries[frame / TREE_FRAMES]
                .fetch_update(|v| v.dec_force(1 << flags.order(), flags.into()))
            {
                match self.lower.get_at(frame, flags.order()) {
                    Err(Error::Memory) => {
                        self.trees.entries[frame / TREE_FRAMES]
                            .fetch_update(|v| Some(v.inc(1 << flags.order())))
                            .expect("Undo failed");
                        // try next
                    }
                    Ok(_) => return Ok(frame),
                    Err(e) => return Err(e),
                };
            }
        } else {
            match self.reserve_and_get(core, flags, start_idx) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }

        for _ in 0..RETRIES {
            match self.steal_from_reserved(core, flags, frame) {
                Err(Error::Retry) => {}
                Err(Error::Memory) => break,
                r => return r,
            }
        }

        // Global search
        if frame.is_none() {
            // Fallback to global allocation (ignoring local reservations)
            return self.get_any_global(start_idx, flags);
        }
        Err(Error::Memory)
    }

    fn put(&self, core: usize, frame: usize, mut flags: Flags) -> Result<()> {
        self.check(frame, flags.order())?;
        // Put usually does not know about movability
        flags.set_movable(false);

        // First free the frame in the lower allocator
        self.lower.put(frame, flags.order())?;

        // Then update local / global counters
        let i = frame / TREE_FRAMES;
        let local = &self.local[core % self.local.len()];
        // Update the put-reserve heuristic
        let may_reserve = self.cores() > 1 && local.frees_push(i);

        // Try update own trees first
        let num_frames = 1usize << flags.order();
        if flags.order() >= HUGE_ORDER {
            let preferred = local.preferred(Kind::Huge);
            if let Ok(_) = preferred.fetch_update(|v| v.inc(frame, 1 << flags.order())) {
                return Ok(());
            }
        } else {
            // Might be movable or fixed
            for kind in [Kind::Movable, Kind::Fixed] {
                let preferred = local.preferred(kind);
                if let Ok(_) = preferred.fetch_update(|v| v.inc(frame, 1 << flags.order())) {
                    return Ok(());
                }
            }
        }

        // Increment or reserve globally
        if let Some(tree) = self.trees.inc_or_reserve(i, num_frames, may_reserve) {
            // Change preferred tree to speedup future frees
            let entry = LocalTree::with(i * TREE_FRAMES, tree.free() + num_frames);
            let kind = flags.with_movable(tree.kind() == Kind::Movable).into();
            self.swap_reserved(local.preferred(kind), entry, kind);
        }
        Ok(())
    }

    fn is_free(&self, frame: usize, order: usize) -> bool {
        if self.check(frame, order).is_ok() {
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

    fn drain(&self, core: usize) -> Result<()> {
        for kind in Kind::all() {
            let preferred = self.local[core % self.local.len()].preferred(kind);
            self.swap_reserved(preferred, LocalTree::none(), kind);
        }
        Ok(())
    }

    fn fast_stats(&self) -> Stats {
        let mut stats = self.trees.stats();
        for local in self.local {
            for kind in Kind::all() {
                let preferred = local.preferred(kind).load();
                if preferred.present() {
                    stats.free_frames += preferred.free();
                    if kind == Kind::Huge || stats.free_frames == TREE_FRAMES {
                        stats.free_huge += preferred.free() / HUGE_FRAMES;
                    }
                }
            }
        }
        stats
    }

    fn stats(&self) -> Stats {
        self.lower.stats()
    }

    fn fast_stats_at(&self, frame: usize, order: usize) -> Stats {
        if order == TREE_ORDER {
            let tree = self.trees.get(frame / TREE_FRAMES);
            if tree.reserved() {
                for local in self.local {
                    for kind in Kind::all() {
                        let preferred = local.preferred(kind).load();
                        if preferred.present()
                            && preferred.frame() / TREE_FRAMES == frame / TREE_FRAMES
                        {
                            return Stats {
                                free_frames: tree.free() + preferred.free(),
                                free_huge: if kind == Kind::Huge || tree.free() == TREE_FRAMES {
                                    (tree.free() + preferred.free()) / HUGE_FRAMES
                                } else {
                                    0
                                },
                                free_trees: (tree.free() + preferred.free()) / TREE_FRAMES,
                            };
                        }
                    }
                }
                Stats::default()
            } else {
                Stats {
                    free_frames: tree.free(),
                    free_huge: if tree.kind() == Kind::Huge || tree.free() == TREE_FRAMES {
                        tree.free() / HUGE_FRAMES
                    } else {
                        0
                    },
                    free_trees: tree.free() / TREE_FRAMES,
                }
            }
        } else {
            self.stats_at(frame, order)
        }
    }

    fn stats_at(&self, frame: usize, order: usize) -> Stats {
        self.lower.stats_at(frame, order)
    }

    fn validate(&self) {
        warn!("validate");
        let fast_stats = self.fast_stats();
        let full_stats = self.stats();
        assert_eq!(fast_stats.free_frames, full_stats.free_frames);
        assert!(fast_stats.free_huge <= full_stats.free_huge); // under-approximation
        let mut reserved = 0;
        for (i, tree) in self.trees.entries.iter().enumerate() {
            let tree = tree.load();
            if !tree.reserved() {
                let free = self
                    .lower
                    .stats_at(i * TREE_FRAMES, TREE_FRAMES.ilog2() as _)
                    .free_frames;
                assert_eq!(tree.free(), free);
            } else {
                reserved += 1;
            }
        }
        for local in self.local {
            for kind in Kind::all() {
                let tree = local.preferred(kind).load();
                if tree.present() {
                    let global = self.trees.get(tree.frame() / TREE_FRAMES);
                    let free = self
                        .lower
                        .stats_at(tree.frame(), TREE_FRAMES.ilog2() as _)
                        .free_frames;
                    assert_eq!(tree.free() + global.free(), free);
                    reserved -= 1;
                }
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn check(&self, frame: usize, order: usize) -> Result<()> {
        ensure!(order <= MAX_ORDER, "Invalid order {order}");
        ensure!(
            frame + (1 << order) <= self.lower.frames(),
            "Frame {frame} out of bounds"
        );
        ensure!(frame.is_multiple_of(1 << order), "Frame {frame} misaligned");
        Ok(())
    }

    /// The number of cores, we sufficiently support.
    /// For only a few frames, we cannot really reserve enough CPU-local trees.
    fn max_cores(frames: usize) -> usize {
        (frames.div_ceil(TREE_FRAMES) / Kind::LEN).max(1)
    }

    fn lower_get(
        &self,
        mut tree: LocalTree,
        order: usize,
        frame: Option<usize>,
    ) -> Result<LocalTree> {
        let (frame, _huge) = match frame {
            Some(frame) => self.lower.get_at(frame, order).map(|h| (frame, h))?,
            None => self.lower.get(tree.frame(), order)?,
        };
        tree.set_frame(frame);
        tree.set_free(tree.free() - (1 << order));
        Ok(tree)
    }

    fn get_from_local(
        &self,
        core: usize,
        kind: Kind,
        order: usize,
        frame: Option<usize>,
    ) -> core::result::Result<usize, (Error, LocalTree)> {
        let preferred = self.local[core].preferred(kind);

        match preferred.fetch_update(|v| v.dec(frame, 1 << order)) {
            Ok(old) => match self.lower_get(old, order, frame) {
                Ok(new) => {
                    if old.frame() / BITFIELD_ROW != new.frame() / BITFIELD_ROW {
                        let _ = preferred.fetch_update(|v| v.set_start(new.frame(), false));
                    }
                    Ok(new.frame())
                }
                Err(e) => {
                    self.trees.entries[old.frame() / TREE_FRAMES]
                        .fetch_update(|v| Some(v.inc(1 << order)))
                        .expect("Undo failed");
                    Err((e, old))
                }
            },
            Err(old) => {
                if old.present() && self.sync_with_global(preferred, order, old) {
                    return Err((Error::Retry, old));
                }
                Err((Error::Memory, old))
            }
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn sync_with_global(&self, preferred: &Atom<LocalTree>, order: usize, old: LocalTree) -> bool {
        if !old.present() || old.free() >= 1 << order {
            return false;
        }

        let i = old.frame() / TREE_FRAMES;
        let min = (1 << order) - old.free();

        if let Some(global) = self.trees.sync(i, min) {
            let new = LocalTree::with(old.frame(), old.free() + global.free());
            if preferred.compare_exchange(old, new).is_ok() {
                info!(
                    "sync success idx={i} kind={:?} free={}",
                    global.kind(),
                    new.free()
                );
                return true;
            }
            self.trees.entries[i]
                .fetch_update(|v| Some(v.inc(global.free())))
                .expect("Undo failed");
            false
        } else {
            false
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, core: usize, flags: Flags, start: usize) -> Result<usize> {
        // Try reserve new tree
        let reserve = |i: usize, range: Range<usize>| {
            let range = (1 << flags.order()).max(range.start)..range.end;
            if let Ok(old) =
                self.trees.entries[i].fetch_update(|v| v.reserve(range.clone(), flags.into()))
            {
                match self.lower_get(
                    LocalTree::with(i * TREE_FRAMES, old.free()),
                    flags.order(),
                    None,
                ) {
                    Ok(new) => {
                        let preferred = self.local[core].preferred(flags.into());
                        self.swap_reserved(preferred, new, flags.into());
                        Ok(new.frame())
                    }
                    Err(e) => {
                        self.trees.unreserve(i, old.free(), flags.into());
                        Err(e)
                    }
                }
            } else {
                Err(Error::Memory)
            }
        };
        let filter =
            |t: Tree, range: Range<usize>| t.reserve(range.clone(), flags.into()).is_some();

        const CL: usize = align_of::<Align>() / size_of::<Tree>();
        // Why does 16 work so well? Are there better values?
        let near = (self.trees.len() / 16).max(CL / 4);
        // Why does align twice near help?
        // This leaves some space between starting points...
        let start = align_down(start, (2 * near).next_power_of_two());

        // Find best fit in fragmented trees
        if flags.order() < HUGE_ORDER {
            let range = 0..TREE_FRAMES;
            match self.trees.search_best::<4>(
                start,
                1,
                near,
                |t| filter(t, range.clone()),
                |i| reserve(i, range.clone()),
            ) {
                Err(Error::Memory) => {}
                r => return r,
            }
            // Not free
            match self.trees.search(start, near, self.trees.len(), |i| {
                reserve(i, 0..TREE_FRAMES)
            }) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }

        // Any
        self.trees
            .search(start, 0, self.trees.len(), |i| reserve(i, 0..usize::MAX))
    }

    /// Steal a tree from another core
    fn steal_from_reserved(
        &self,
        core: usize,
        flags: Flags,
        frame: Option<usize>,
    ) -> Result<usize> {
        let kind = Kind::from(flags);
        for i in 1..self.local.len() {
            let target_core = (core + i) % self.local.len();
            for o_kind in 0..Kind::LEN {
                let t_kind = Kind::from((kind as usize + o_kind) % Kind::LEN);

                if t_kind.accepts(kind) {
                    // Less strict kind, just allocate
                    match self.get_from_local(target_core, t_kind, flags.order(), frame) {
                        Err((Error::Memory, _)) => {}
                        r => return r.map_err(|e| e.0),
                    }
                } else {
                    // More strict kind, steal and convert tree
                    match self.steal_tree(target_core, t_kind, flags.order(), frame) {
                        Ok(stolen) => {
                            self.swap_reserved(self.local[core].preferred(kind), stolen, kind);
                            return Ok(stolen.frame());
                        }
                        Err(Error::Memory) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        Err(Error::Memory)
    }

    fn steal_tree(
        &self,
        core: usize,
        kind: Kind,
        order: usize,
        frame: Option<usize>,
    ) -> Result<LocalTree> {
        let preferred = self.local[core].preferred(kind);
        match preferred.fetch_update(|v| v.steal(1 << order, frame)) {
            Ok(stolen) => match self.lower_get(stolen, order, frame) {
                Ok(stolen) => Ok(stolen),
                Err(e) => {
                    assert!(stolen.present());
                    let i = stolen.frame() / TREE_FRAMES;
                    self.trees.unreserve(i, stolen.free(), kind);
                    Err(e)
                }
            },
            _ => Err(Error::Memory),
        }
    }

    fn get_any_global(&self, start_idx: usize, flags: Flags) -> Result<usize> {
        self.trees.search(start_idx, 0, self.trees.len(), |i| {
            let old = self.trees.entries[i]
                .fetch_update(|v| v.dec_force(1 << flags.order(), flags.into()))
                .map_err(|_| Error::Memory)?;

            match self.lower.get(i * TREE_FRAMES, flags.order()) {
                Ok((frame, _)) => Ok(frame),
                Err(e) => {
                    let exp = old.dec_force(1 << flags.order(), flags.into()).unwrap();
                    if self.trees.entries[i].compare_exchange(exp, old).is_err() {
                        self.trees.entries[i]
                            .fetch_update(|v| Some(v.inc(1 << flags.order())))
                            .expect("Undo failed");
                    }
                    Err(e)
                }
            }
        })
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    /// Returns false if the swap failed.
    fn swap_reserved(&self, preferred: &Atom<LocalTree>, new: LocalTree, kind: Kind) {
        let old = preferred.swap(new);
        if old.present() {
            self.trees
                .unreserve(old.frame() / TREE_FRAMES, old.free(), kind);
        }
    }
}

impl fmt::Debug for LLFree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let huge = self.frames() / (1 << HUGE_ORDER);
        let Stats {
            free_frames,
            free_huge,
            free_trees: _,
        } = self.stats();

        f.debug_struct(Self::name())
            .field(
                "managed",
                &FmtFn(|f| write!(f, "{} frames ({huge} huge)", self.frames())),
            )
            .field(
                "free",
                &FmtFn(|f| write!(f, "{free_frames} frames ({free_huge} huge)")),
            )
            .field(
                "trees",
                &FmtFn(|f| write!(f, "{:?} (N={})", self.trees, TREE_FRAMES)),
            )
            .field("locals", &self.local)
            .finish()?;
        Ok(())
    }
}
