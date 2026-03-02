//! Upper allocator implementation

use core::fmt;
use core::ops::Range;

use log::{debug, info, warn};

use crate::local::Locals;
use crate::lower::Lower;
use crate::trees::{Prio, Tree, TreeId, Trees};
use crate::util::{Align, FmtFn, align_down, spin_wait};
use crate::{
    Alloc, Error, Flags, FrameId, HUGE_FRAMES, HUGE_ORDER, Init, Kind, KindDesc, MAX_ORDER,
    MetaData, MetaSize, RETRIES, Result, Stats, TREE_FRAMES, TREE_ORDER,
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
    locals: Locals<'a>,
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

    #[cold]
    fn new(kinds: &[KindDesc], frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self> {
        info!(
            "initializing k={kinds:?} f={frames} {:?} {:?} {:?}",
            meta.local.as_ptr_range(),
            meta.trees.as_ptr_range(),
            meta.lower.as_ptr_range()
        );
        ensure!(
            Error::Initialization;
            meta.valid(Self::metadata_size(kinds, frames)),
            "Invalid metadata"
        );
        ensure!(
            kinds.iter().any(|k| k.0 != Kind::HUGE),
            "At least one kind for small pages"
        );
        ensure!(
            kinds.iter().any(|k| k.0 == Kind::HUGE),
            "At least one kind for huge pages"
        );

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Init per-cpu data
        let locals = Locals::new(meta.local, kinds)?;

        // Init tree array
        let tree_init = if init != Init::None {
            Some(|start| lower.stats_at(FrameId(start), TREE_ORDER).free_frames)
        } else {
            None
        };
        let trees = Trees::new(frames, meta.trees, tree_init);

        Ok(Self {
            locals,
            lower,
            trees,
        })
    }

    fn metadata_size(kinds: &[KindDesc], frames: usize) -> MetaSize {
        MetaSize {
            local: Locals::metadata_size(kinds),
            trees: Trees::metadata_size(frames),
            lower: Lower::metadata_size(frames),
        }
    }

    unsafe fn metadata(&mut self) -> MetaData<'a> {
        unsafe {
            MetaData {
                local: self.locals.metadata(),
                trees: self.trees.metadata(),
                lower: self.lower.metadata(),
            }
        }
    }

    fn get(&self, frame: Option<FrameId>, flags: Flags) -> Result<FrameId> {
        self.check(FrameId(0), flags.order())?;
        // Different starting points for each core
        let mut start_idx = TreeId(self.trees.len() / self.locals.len() * flags.local());

        // Try local reservation first (if enough memory)
        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..RETRIES {
            match self.get_from_local(flags, frame) {
                Ok(frame) => return Ok(frame),
                Err((Error::Retry, _)) => continue,
                Err((Error::Memory, old)) => {
                    if let Some(frame) = old {
                        start_idx = frame.as_tree();
                    }
                }
                Err((e, _)) => return Err(e),
            }
            // Try reserve new tree if no specific frame is requested
            if frame.is_none() {
                match self.reserve_and_get(flags, start_idx) {
                    Err(Error::Memory) => {}
                    r => return r,
                }
            }

            // few local trees -> high probability that they are shared
            if self.locals.len() < 4 {
                // If reservation fails, there might be another concurrent update -> retry
                spin_wait(8 * RETRIES, || {
                    self.locals.can_get(flags.local(), frame, flags.frames())
                });
            }
        }

        // Global search
        if let Some(frame) = frame {
            let kind = self.locals.kind(flags.local());
            // Do not reserve trees if a specific frame is allocated
            if let Ok(_) = self
                .trees
                .at(frame.as_tree())
                .fetch_update(|v| v.dec_force(flags.frames(), kind))
            {
                match self.lower.get_at(frame, flags.order()) {
                    Err(Error::Memory) => {
                        self.trees
                            .at(frame.as_tree())
                            .fetch_update(|v| Some(v.inc(flags.frames())))
                            .expect("Undo failed");
                        // try next
                    }
                    Ok(_) => return Ok(frame),
                    Err(e) => return Err(e),
                };
            }
        }

        // OOM steal from other cores
        for _ in 0..RETRIES {
            match self.steal_from_reserved(flags, frame) {
                Err(Error::Retry) => continue,
                Err(Error::Memory) => break,
                r => return r,
            }
        }

        // Last resort: Global search
        if frame.is_none() {
            // Fallback to global allocation (ignoring local reservations)
            return self.get_any_global(start_idx, flags);
        }
        Err(Error::Memory)
    }

    fn put(&self, frame: FrameId, flags: Flags) -> Result<()> {
        self.check(frame, flags.order())?;

        // First free the frame in the lower allocator
        self.lower.put(frame, flags.order())?;

        // Then update local / global counters
        let i = frame.as_tree();
        // Update the put-reserve heuristic
        #[cfg(feature = "free_reserve")]
        let may_reserve = self.locals.len() > 1 && self.locals.frees_push(flags.local(), i);
        #[cfg(not(feature = "free_reserve"))]
        let may_reserve = false;

        // Try update own trees first
        if let Ok(_) = self.locals.put(flags.local(), frame, flags.frames()) {
            return Ok(());
        }

        let kind = self.locals.kind(flags.local());
        // Increment or reserve globally
        if let Some(tree) = self
            .trees
            .inc_or_reserve(i, flags.frames(), kind, may_reserve)
        {
            warn!(
                "free reserved tree idx={i} kind={:?} free={}",
                tree.kind(),
                tree.free()
            );
            // Change preferred tree to speedup future frees
            if let Some((frame, free)) =
                self.locals
                    .swap(flags.local(), i.as_frame(), tree.free() + flags.frames())
            {
                self.trees.unreserve(frame.as_tree(), free, kind);
            }
        }
        Ok(())
    }

    fn is_free(&self, frame: FrameId, order: usize) -> bool {
        if self.check(frame, order).is_ok() {
            self.lower.is_free(frame, order)
        } else {
            false
        }
    }

    fn frames(&self) -> usize {
        self.lower.frames()
    }

    fn drain(&self, local: usize) -> Result<()> {
        self.locals.drain(local);
        Ok(())
    }

    fn fast_stats(&self) -> Stats {
        let t_stats = self.trees.stats();
        let l_stats = self.locals.stats();
        t_stats + l_stats
    }

    fn stats(&self) -> Stats {
        self.lower.stats()
    }

    fn fast_stats_at(&self, frame: FrameId, order: usize) -> Stats {
        if order == TREE_ORDER {
            let tree = self.trees.get(frame.as_tree());
            if tree.reserved() {
                self.locals.stats_at(frame, tree.free())
            } else {
                Stats {
                    free_frames: tree.free(),
                    free_huge: if tree.kind().is_huge() || tree.free() == TREE_FRAMES {
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

    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        self.lower.stats_at(frame, order)
    }

    fn validate(&self) {
        debug!("validate");
        let fast_stats = self.fast_stats();
        let full_stats = self.stats();
        assert_eq!(fast_stats.free_frames, full_stats.free_frames);
        assert!(fast_stats.free_huge <= full_stats.free_huge); // under-approximation
        let mut reserved = 0;
        for i in (0..self.trees.len()).map(TreeId) {
            let tree = self.trees.get(i);
            if !tree.reserved() {
                let free = self
                    .lower
                    .stats_at(i.as_frame(), TREE_FRAMES.ilog2() as _)
                    .free_frames;
                assert_eq!(tree.free(), free);
            } else {
                reserved += 1;
            }
        }
        for local in 0..self.locals.len() {
            if let Some((frame, free)) = self.locals.load(local) {
                let global = self.trees.get(frame.as_tree());
                let e_free = self
                    .lower
                    .stats_at(frame, TREE_FRAMES.ilog2() as _)
                    .free_frames;
                assert_eq!(free + global.free(), e_free);
                reserved -= 1;
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn check(&self, frame: FrameId, order: usize) -> Result<()> {
        ensure!(order <= MAX_ORDER, "Invalid order {order}");
        ensure!(
            frame.0 + (1 << order) <= self.lower.frames(),
            "Frame {} out of bounds",
            frame.0
        );
        ensure!(
            frame.0.is_multiple_of(1 << order),
            "Frame {} misaligned",
            frame.0
        );
        Ok(())
    }

    fn lower_get(&self, frame: FrameId, order: usize, specific: bool) -> Result<FrameId> {
        let (frame, _huge) = if specific {
            self.lower.get_at(frame, order).map(|h| (frame, h))?
        } else {
            self.lower.get(frame, order)?
        };
        Ok(frame)
    }

    fn get_from_local(
        &self,
        flags: Flags,
        frame: Option<FrameId>,
    ) -> core::result::Result<FrameId, (Error, Option<FrameId>)> {
        match self.locals.get(flags.local(), frame, flags.frames()) {
            Ok(t_frame) => {
                match self.lower_get(frame.unwrap_or(t_frame), flags.order(), frame.is_some()) {
                    Ok(frame) => {
                        if t_frame.as_row() != frame.as_row() {
                            self.locals.set_start(flags.local(), frame);
                        }
                        Ok(frame)
                    }
                    Err(e) => {
                        self.trees
                            .at(t_frame.as_tree())
                            .fetch_update(|v| Some(v.inc(flags.frames())))
                            .expect("Undo failed");
                        Err((e, Some(t_frame)))
                    }
                }
            }
            Err(Some((t_frame, free))) => {
                let e = if self.sync_with_global(flags, t_frame, free) {
                    Error::Retry
                } else {
                    Error::Memory
                };
                Err((e, Some(t_frame)))
            }
            Err(None) => Err((Error::Memory, None)),
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn sync_with_global(&self, flags: Flags, frame: FrameId, free: usize) -> bool {
        let i = frame.as_tree();
        let min = flags.frames() - free;

        if let Some(global) = self.trees.sync(i, min) {
            if self.locals.put(flags.local(), frame, global.free()).is_ok() {
                debug!(
                    "sync success {i} {:?} free={}",
                    global.kind(),
                    free + global.free()
                );
                true
            } else {
                self.trees
                    .at(i)
                    .fetch_update(|v| Some(v.inc(global.free())))
                    .expect("Undo failed");
                false
            }
        } else {
            false
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, flags: Flags, start: TreeId) -> Result<FrameId> {
        let kind = self.locals.kind(flags.local());
        // Try reserve new tree
        let reserve = |i: TreeId, range: Range<usize>| {
            let range = (flags.frames()).max(range.start)..range.end;
            if let Ok(old) = self
                .trees
                .at(i)
                .fetch_update(|v| v.reserve(range.clone(), kind))
            {
                match self.lower_get(i.as_frame(), flags.order(), false) {
                    Ok(new) => {
                        if let Some((frame, free)) =
                            self.locals
                                .swap(flags.local(), new, old.free() - flags.frames())
                        {
                            self.trees.unreserve(frame.as_tree(), free, kind);
                        }
                        Ok(new)
                    }
                    Err(e) => {
                        self.trees.unreserve(i, old.free(), kind);
                        Err(e)
                    }
                }
            } else {
                Err(Error::Memory)
            }
        };

        const CL: usize = align_of::<Align>() / size_of::<Tree>();
        // Why does 16 work so well? Are there better values?
        let near = (self.trees.len() / 16).max(CL / 4);
        // Why does align twice near help?
        // This leaves some space between starting points...
        let start = TreeId(align_down(start.0, (2 * near).next_power_of_two()));

        // Find best fit in fragmented trees
        if flags.order() < HUGE_ORDER {
            let prio = |tree: Tree| -> Prio {
                match tree.free() {
                    0 | TREE_FRAMES => Prio::None,              // entirely allocated
                    f if f >= TREE_FRAMES / 2 => Prio::Good(2), // half free
                    f if f >= TREE_FRAMES / 64 => Prio::Best,   // almost allocated
                    _ => Prio::Good(1), // low free count -> causes frequent reservations
                }
            };

            let range = 0..TREE_FRAMES;
            match self
                .trees
                .search_best::<2>(start, 1, near, prio, |i| reserve(i, range.clone()))
            {
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
    fn steal_from_reserved(&self, flags: Flags, frame: Option<FrameId>) -> Result<FrameId> {
        // info!("try steal local={}", flags.local());
        if let Some(t_frame) = self.locals.steal(flags.local(), frame, flags.frames()) {
            match self.lower_get(frame.unwrap_or(t_frame), flags.order(), frame.is_some()) {
                Err(Error::Memory) => {
                    // undo counter decrement
                    self.trees
                        .at(t_frame.as_tree())
                        .fetch_update(|v| Some(v.inc(flags.frames())))
                        .expect("Undo failed");
                }
                r => return r,
            }
        }

        if let Some((t_frame, old)) =
            self.locals
                .steal_downgrade(flags.local(), frame, flags.frames())
        {
            let kind = self.locals.kind(flags.local());
            if let Some((frame, free)) = old {
                self.trees.unreserve(frame.as_tree(), free, kind);
            }

            match self.lower_get(frame.unwrap_or(t_frame), flags.order(), frame.is_some()) {
                Err(Error::Memory) => {
                    // undo counter decrement
                    self.trees
                        .at(t_frame.as_tree())
                        .fetch_update(|v| Some(v.inc(flags.frames())))
                        .expect("Undo failed");
                }
                r => return r,
            }
        }
        Err(Error::Retry)
    }

    fn get_any_global(&self, start_idx: TreeId, flags: Flags) -> Result<FrameId> {
        let kind = self.locals.kind(flags.local());
        self.trees.search(start_idx, 0, self.trees.len(), |i| {
            let old = self
                .trees
                .at(i)
                .fetch_update(|v| v.dec_force(flags.frames(), kind))
                .map_err(|_| Error::Memory)?;

            match self.lower.get(i.as_frame(), flags.order()) {
                Ok((frame, _)) => Ok(frame),
                Err(e) => {
                    let exp = old.dec_force(flags.frames(), kind).unwrap();
                    if self.trees.at(i).compare_exchange(exp, old).is_err() {
                        self.trees
                            .at(i)
                            .fetch_update(|v| Some(v.inc(flags.frames())))
                            .expect("Undo failed");
                    }
                    Err(e)
                }
            }
        })
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
            .field("locals", &self.locals)
            .finish()?;
        Ok(())
    }
}
