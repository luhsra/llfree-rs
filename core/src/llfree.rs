//! Upper allocator implementation

use core::fmt;
use core::ops::Range;

use log::{debug, info, warn};

use crate::local::Locals;
use crate::lower::Lower;
use crate::trees::{TreeId, Trees};
use crate::util::{Align, align_down, spin_wait};
use crate::*;

/// Return [`Error::Address`] if condition is not met.
#[allow(unused_macros)]
macro_rules! ensure {
    ($cond:expr, $($args:expr),*) => {
        if !($cond) {
            log::error!($($args),*);
            return Err(Error::Address);
        }
    };
    ($err:expr; $cond:expr, $($args:expr),*) => {
        if !($cond) {
            log::error!($($args),*);
            return Err($err);
        }
    };
}

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
    /// Policy for accessing tree tiers
    policy: PolicyFn,
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
    fn new(frames: usize, init: Init, tiering: &Tiering, meta: MetaData<'a>) -> Result<Self> {
        info!("initializing f={frames} {tiering:?} {meta:?}");
        ensure!(
            Error::Initialization;
            meta.valid(Self::metadata_size(tiering, frames)),
            "Invalid metadata"
        );

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Init per-cpu data
        let locals = Locals::new(meta.local, tiering)?;

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
            policy: tiering.policy,
        })
    }

    fn metadata_size(tiering: &Tiering, frames: usize) -> MetaSize {
        MetaSize {
            local: Locals::metadata_size(tiering),
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

    fn get(&self, frame: Option<FrameId>, request: Request) -> Result<(Tier, FrameId)> {
        self.check(frame.unwrap_or(FrameId(0)), &request)?;
        // Different starting points for each core
        let mut start_idx =
            TreeId(self.trees.len() / self.locals.len() * request.local.unwrap_or_default());

        // Try local reservation first (if enough memory)
        // Retry allocation up to n times if it fails due to a concurrent update
        if let Some(local) = request.local {
            for _ in 0..RETRIES {
                match self.get_from_local(request.order, local, frame) {
                    Ok(res) => return Ok(res),
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
                    match self.reserve_and_get(request.order, local, start_idx) {
                        Err(Error::Memory) => {}
                        r => return r,
                    }
                }

                // few local trees -> high probability that they are shared
                if self.locals.len() < 4 {
                    // If reservation fails, there might be another concurrent update -> retry
                    spin_wait(8 * RETRIES, || {
                        self.locals
                            .can_get(local, frame.map(FrameId::as_tree), request.frames())
                    });
                }
            }
        }

        // Global search
        if let Some(frame) = frame {
            // Do not reserve trees if a specific frame is allocated
            if let Some(demoted) =
                self.trees
                    .get_demote(frame.as_tree(), request.frames(), request.tier, self.policy)
            {
                match self.lower.get(frame.as_row(), request.order, Some(frame)) {
                    Err(Error::Memory) => {
                        self.trees.put(frame.as_tree(), request.frames());
                        // try next
                    }
                    Ok(_) => return Ok((demoted, frame)),
                    Err(e) => return Err(e),
                };
            }
        }

        // OOM steal from other cores
        for _ in 0..RETRIES {
            match self.steal_from_local(&request, frame) {
                Err(Error::Retry) => continue,
                Err(Error::Memory) => break,
                r => return r,
            }
        }

        // Last resort: Global search
        if frame.is_none() {
            // Fallback to global allocation (ignoring local reservations)
            return self.get_any_global(start_idx, request.order, request.tier);
        }
        Err(Error::Memory)
    }

    fn put(&self, frame: FrameId, request: Request) -> Result<()> {
        self.check(frame, &request)?;

        // First free the frame in the lower allocator
        self.lower.put(frame, request.order)?;

        // Then update local / global counters
        let i = frame.as_tree();

        // Try update own trees first
        if let Some(local) = request.local {
            // Update the put-reserve heuristic
            #[cfg(feature = "free_reserve")]
            let may_reserve = self.locals.frees_push(local, i);
            #[cfg(not(feature = "free_reserve"))]
            let may_reserve = false;

            if self.locals.put(local, frame.as_tree(), request.frames()) {
                return Ok(());
            }

            let tier = self.locals.tier(local);
            // Increment or reserve globally
            if let Some(free) = self
                .trees
                .put_or_reserve(i, request.frames(), tier, may_reserve)
            {
                warn!("free reserved tree idx={i} tier={tier:?} free={free}");
                // Change preferred tree to speedup future frees
                if let Some((frame, free)) = self.locals.swap(local, i, free + request.frames()) {
                    self.trees
                        .unreserve(frame.as_tree(), free, tier, self.policy);
                }
            }
        } else {
            self.trees.put(i, request.frames());
        }

        Ok(())
    }

    fn is_free(&self, frame: FrameId, order: usize) -> bool {
        if self
            .check(
                frame,
                &Request {
                    order,
                    tier: Tier(0),
                    local: None,
                },
            )
            .is_ok()
        {
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

    fn tree_stats(&self, tiers: &mut [TierStats]) -> TreeStats {
        let t_stats = self.trees.stats(tiers);
        let l_stats = self.locals.stats(tiers);
        TreeStats {
            free_frames: t_stats.free_frames + l_stats.free_frames,
            free_trees: t_stats.free_trees + l_stats.free_trees,
        }
    }

    fn stats(&self) -> Stats {
        self.lower.stats()
    }

    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        self.lower.stats_at(frame, order)
    }

    fn change_tree(&self, matcher: TreeMatch, change: TreeChange) -> Result<()> {
        self.trees.change(matcher, change, |i| {
            self.stats_at(i.as_frame(), TREE_ORDER).free_frames
        })
    }

    fn validate(&self) {
        debug!("validate");
        let fast_stats = self.tree_stats(&mut []);
        let full_stats = self.stats();
        assert_eq!(fast_stats.free_frames, full_stats.free_frames);
        let mut reserved = 0;
        for i in (0..self.trees.len()).map(TreeId) {
            match self.trees.stats_at(i) {
                (_tier, free, false) => {
                    let l_free = self
                        .lower
                        .stats_at(i.as_frame(), TREE_FRAMES.ilog2() as _)
                        .free_frames;
                    assert_eq!(free, l_free);
                }
                _ => {
                    reserved += 1;
                }
            }
        }
        for local in 0..self.locals.len() {
            if let Some((row, free)) = self.locals.load(local) {
                // Tree is expected to be reserved!
                let (_tier, g_free, res) = self.trees.stats_at(row.as_tree());
                assert!(res);
                let l_free = self
                    .lower
                    .stats_at(row.as_frame(), TREE_FRAMES.ilog2() as _)
                    .free_frames;
                assert_eq!(free + g_free, l_free);
                reserved -= 1;
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn check(&self, frame: FrameId, request: &Request) -> Result<()> {
        ensure!(request.order <= MAX_ORDER, "Invalid order {request:?}");
        ensure!(
            frame.0 + (1 << request.order) <= self.lower.frames(),
            "Frame {} out of bounds",
            frame.0
        );
        ensure!(
            frame.0.is_multiple_of(1 << request.order),
            "Frame {} misaligned",
            frame.0
        );
        if let Some(local) = request.local {
            ensure!(
                self.locals.tier(local) == request.tier,
                "Invalid local {} for tier {:?}",
                local,
                request.tier
            );
        }
        Ok(())
    }

    fn get_from_local(
        &self,
        order: usize,
        local: usize,
        frame: Option<FrameId>,
    ) -> core::result::Result<(Tier, FrameId), (Error, Option<FrameId>)> {
        match self
            .locals
            .get(local, frame.map(FrameId::as_tree), 1 << order)
        {
            Ok(row) => match self.lower.get(row, order, frame) {
                Ok(frame) => {
                    if row != frame.as_row() {
                        self.locals.set_start(local, frame.as_row());
                    }
                    Ok((self.locals.tier(local), frame))
                }
                Err(e) => {
                    self.trees.put(row.as_tree(), 1 << order);
                    Err((e, Some(row.as_frame())))
                }
            },
            Err(Some((row, free))) => {
                let e = if self.sync_with_global(order, local, row.as_tree(), free) {
                    Error::Retry
                } else {
                    Error::Memory
                };
                Err((e, Some(row.as_frame())))
            }
            Err(None) => Err((Error::Memory, None)),
        }
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// Returns if the global counter was large enough
    fn sync_with_global(&self, order: usize, local: usize, tree: TreeId, free: usize) -> bool {
        let min = (1 << order) - free;
        let tier = self.locals.tier(local);

        if let Some(free) = self.trees.sync(tree, min) {
            if self.locals.put(local, tree, free) {
                debug!("sync success {tree} {tier:?} free={}", free + free);
                true
            } else {
                self.trees.put(tree, free);
                false
            }
        } else {
            false
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(
        &self,
        order: usize,
        local: usize,
        start: TreeId,
    ) -> Result<(Tier, FrameId)> {
        let tier = self.locals.tier(local);
        // Try reserve new tree
        let reserve = |i: TreeId, range: Range<usize>| {
            let range = (1 << order).max(range.start)..range.end;
            if let Some(free) = self.trees.reserve(i, range.clone(), tier) {
                match self.lower.get(i.as_row(), order, None) {
                    Ok(new) => {
                        if let Some((row, free)) =
                            self.locals.swap(local, new.as_tree(), free - (1 << order))
                        {
                            self.trees.unreserve(row.as_tree(), free, tier, self.policy);
                        }
                        Ok(new)
                    }
                    Err(e) => {
                        self.trees.unreserve(i, free, tier, self.policy);
                        Err(e)
                    }
                }
            } else {
                Err(Error::Memory)
            }
        };

        const CL: usize = align_of::<Align>() / 4;
        // Why does 16 work so well? Are there better values?
        let near = (self.trees.len() / 16).max(CL / 4);
        // Why does align twice near help?
        // This leaves some space between starting points...
        let start = TreeId(align_down(start.0, (2 * near).next_power_of_two()));

        // Find best fit in fragmented trees
        if order < HUGE_ORDER {
            let range = 0..TREE_FRAMES;
            match self
                .trees
                .search_best::<2>(start, 1, near, tier, self.policy, |i| {
                    reserve(i, range.clone())
                }) {
                Err(Error::Memory) => {}
                r => return r.map(|frame| (tier, frame)),
            }
            // Not free
            match self.trees.search(start, near, self.trees.len(), |i| {
                reserve(i, 0..TREE_FRAMES)
            }) {
                Err(Error::Memory) => {}
                r => return r.map(|frame| (tier, frame)),
            }
        }

        // Any
        self.trees
            .search(start, 0, self.trees.len(), |i| reserve(i, 0..usize::MAX))
            .map(|frame| (tier, frame))
    }

    /// Steal from a local reservation and possibly demote or drain it
    fn steal_from_local(
        &self,
        request: &Request,
        frame: Option<FrameId>,
    ) -> Result<(Tier, FrameId)> {
        if let Some((tier, row)) = self.locals.steal(
            request.tier,
            request.local,
            frame.map(FrameId::as_tree),
            request.frames(),
            self.policy,
        ) {
            match self.lower.get(row, request.order, frame) {
                Err(Error::Memory) => {
                    // undo counter decrement
                    self.trees.put(row.as_tree(), request.frames());
                }
                r => return r.map(|frame| (tier, frame)),
            }
        }
        if let Some((row, old)) = self.locals.steal_demote(
            request.tier,
            request.local,
            frame.map(FrameId::as_tree),
            request.frames(),
            self.policy,
        ) {
            if let Some((frame, free)) = old {
                self.trees
                    .unreserve(frame.as_tree(), free, request.tier, self.policy);
            }

            match self.lower.get(row, request.order, frame) {
                Err(Error::Memory) => {
                    // undo counter decrement
                    self.trees.put(row.as_tree(), request.frames());
                }
                r => return r.map(|frame| (request.tier, frame)),
            }
        }
        Err(Error::Retry)
    }

    fn get_any_global(
        &self,
        start_idx: TreeId,
        order: usize,
        tier: Tier,
    ) -> Result<(Tier, FrameId)> {
        self.trees.search(start_idx, 0, self.trees.len(), |i| {
            if let Some(tier) = self.trees.get_demote(i, 1 << order, tier, self.policy) {
                match self.lower.get(i.as_row(), order, None) {
                    Ok(frame) => Ok((tier, frame)),
                    Err(e) => {
                        self.trees.put(i, 1 << order);
                        Err(e)
                    }
                }
            } else {
                Err(Error::Memory)
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
                &fmt::from_fn(|f| write!(f, "{} frames ({huge} huge)", self.frames())),
            )
            .field(
                "free",
                &fmt::from_fn(|f| write!(f, "{free_frames} frames ({free_huge} huge)")),
            )
            .field(
                "trees",
                &fmt::from_fn(|f| write!(f, "{:?} (N={})", self.trees, TREE_FRAMES)),
            )
            .field("locals", &self.locals)
            .finish()?;
        Ok(())
    }
}

impl MetaData<'_> {
    /// Check for alignment and overlap
    fn valid(&self, m: MetaSize) -> bool {
        fn overlap(a: Range<*const u8>, b: Range<*const u8>) -> bool {
            a.contains(&b.start)
                || a.contains(&unsafe { b.end.sub(1) })
                || b.contains(&a.start)
                || b.contains(&unsafe { a.end.sub(1) })
        }
        self.local.len() >= m.local
            && self.trees.len() >= m.trees
            && self.lower.len() >= m.lower
            && self.local.as_ptr().align_offset(align_of::<Align>()) == 0
            && self.trees.as_ptr().align_offset(align_of::<Align>()) == 0
            && self.lower.as_ptr().align_offset(align_of::<Align>()) == 0
            && !overlap(self.local.as_ptr_range(), self.trees.as_ptr_range())
            && !overlap(self.trees.as_ptr_range(), self.lower.as_ptr_range())
            && !overlap(self.lower.as_ptr_range(), self.local.as_ptr_range())
    }
}
