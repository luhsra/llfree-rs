//! Upper allocator implementation

use core::fmt;
use core::ops::Range;

use log::{debug, info};

use crate::local::{Locals, Reservation};
use crate::lower::Lower;
use crate::trees::{TreeId, Trees};
use crate::util::{Align, align_down};
use crate::*;

/// Return [`Error::Argument`] if condition is not met.
#[allow(unused_macros)]
macro_rules! ensure {
    ($cond:expr, $($args:expr),*) => {
        if !($cond) {
            log::error!($($args),*);
            return Err(Error::Argument);
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
/// Allocations/frees within a chunk are handed over to the
/// lower allocator.
/// These chunks are, due to the inner workings of the lower allocator,
/// called *trees*.
/// This allocator stores these tree entries in a [packed array][Trees].
///
/// Additionally, the allocator manages user-provided [clusters][Cluster].
/// Clusters are used to separate trees into different groups.
/// The user also has to provide a [policy function][PolicyFn] that defines
/// how to access the clusters.
///
/// Each cluster can have a different number of [local reservations][Locals],
/// which are used to reduce contention on the tree array.
///
/// If an allocation for a certain cluster cannot be fulfilled,
/// the allocator falls back on stealing from other clusters or
/// demoting the request to a lower cluster, depending on the [policy][Policy].
#[repr(align(64))]
pub struct LLFree<'a> {
    /// CPU local data
    ///
    /// Other CPUs can access this if they drain cores.
    /// Also, these are shared between CPUs if we have more cores than trees.
    locals: Locals<'a>,
    /// Metadata of the lower alloc
    pub lower: Lower<'a>,
    /// Manages the allocator's trees.
    pub trees: Trees<'a>,
    /// Policy for accessing tree clusters
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
    fn new(frames: usize, init: Init, clustering: &Clustering, meta: MetaData<'a>) -> Result<Self> {
        info!("initializing f={frames} {clustering:?} {meta:?}");
        ensure!(
            Error::Initialization;
            meta.valid(&Self::metadata_size(clustering, frames)),
            "Invalid metadata"
        );

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Initialize per-CPU data
        let locals = Locals::new(meta.local, clustering)?;

        // Init tree array
        let tree_init = if init == Init::None {
            None
        } else {
            Some(|start| lower.stats_at(FrameId(start), TREE_ORDER).free_frames)
        };
        let trees = Trees::new(frames, meta.trees, tree_init, clustering.default);

        Ok(Self {
            locals,
            lower,
            trees,
            policy: clustering.policy,
        })
    }

    fn metadata_size(clustering: &Clustering, frames: usize) -> MetaSize {
        MetaSize {
            local: Locals::metadata_size(clustering),
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

    fn get(&self, frame: Option<FrameId>, request: Request) -> Result<(FrameId, Cluster)> {
        self.check(frame.unwrap_or(FrameId(0)), &request)?;

        // Try reserving a specific frame
        if let Some(frame) = frame {
            return self.get_at(frame, request);
        }

        let len = self.locals.cluster_locals(request.cluster).unwrap_or(0);
        // Different starting points for each core
        let mut start_idx =
            TreeId(self.trees.len().checked_div(len).unwrap_or(0) * request.local.unwrap_or(0));

        // Use local reservation if possible
        if let Some(local) = request.local
            && self
                .locals
                .cluster_locals(request.cluster)
                .is_some_and(|len| len > 0 && len < self.trees.len())
        {
            match self.get_local(request.order, request.cluster, local, None, true) {
                Err((Error::Memory, Some(start))) => start_idx = start,
                Err((Error::Memory, _)) => {}
                Err((e, _)) => return Err(e),
                Ok(r) => return Ok(r),
            }
            // Try reserving new tree
            match self.search_and_reserve(request.order, request.cluster, local, start_idx) {
                Err(Error::Memory) => {}
                r => return r,
            }
        } else {
            // Global search
            // Rate how good the tree fulfills the allocation
            let rate = |t, free| {
                if free < request.frames() {
                    return Policy::Invalid;
                }
                (self.policy)(request.cluster, t, free)
            };
            // Any frame
            match self
                .trees
                .search_best::<8, _>(start_idx, 0, self.trees.len(), rate, |i| {
                    self.steal_global(i, request.cluster, request.order, None)
                }) {
                Err(Error::Memory) => {} // continue
                r => return r,
            }
        }

        // -- Out of memory handling ---
        debug!("OOM {:?}", request.cluster);

        // Try stealing from other local reservations
        match self.steal_local(&request, None) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }
        // Fallback to demoting local reservations
        match self.demote_local(&request, None) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }

        Err(Error::Memory)
    }

    fn put(&self, frame: FrameId, request: Request) -> Result<()> {
        self.check(frame, &request)?;

        // First free the frame in the lower allocator
        self.lower.put(frame, request.order)?;

        // Then update local / global counters
        let i = frame.as_tree();

        // Try updating own trees first
        let c = request.cluster;
        if let Some(local) = request.local
            && self.locals.put(c, local, frame.as_tree(), request.frames())
        {
            return Ok(());
        }

        // FIXME: Handle the case where huge pages are split
        // -> This would ideally demote the huge tree to the next lower cluster

        // Increment globally
        self.trees.put(i, request.frames(), self.policy);

        Ok(())
    }

    fn frames(&self) -> usize {
        self.lower.frames()
    }

    fn drain(&self) {
        self.locals.drain(|row, cluster, free| {
            self.trees
                .unreserve(row.as_tree(), free, cluster, self.policy);
        });
    }

    fn tree_stats(&self) -> TreeStats {
        let mut stats = self.trees.stats();
        let l_stats = self.locals.stats();
        stats.free_frames += l_stats.free_frames;
        stats.free_trees += l_stats.free_trees;
        for (c, lc) in stats.clusters.iter_mut().zip(l_stats.clusters.iter()) {
            c.free_frames += lc.free_frames;
            c.alloc_frames += lc.alloc_frames;
        }
        stats
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
        let fast_stats = self.tree_stats();
        let full_stats = self.stats();
        assert_eq!(fast_stats.free_frames, full_stats.free_frames);
        let mut reserved = 0;
        for i in (0..self.trees.len()).map(TreeId) {
            match self.trees.stats_at(i) {
                (_cluster, free, false) => {
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
        for cluster in (0..Cluster::LEN).map(Cluster) {
            if let Some(len) = self.locals.cluster_locals(cluster) {
                for local in 0..len {
                    if let Some(Reservation { row, free, .. }) = self.locals.load(cluster, local) {
                        // Tree is expected to be reserved!
                        let (_cluster, g_free, res) = self.trees.stats_at(row.as_tree());
                        assert!(res);
                        let l_free = self
                            .lower
                            .stats_at(row.as_frame(), TREE_FRAMES.ilog2() as _)
                            .free_frames;
                        assert_eq!(free + g_free, l_free);
                        reserved -= 1;
                    }
                }
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn check(&self, frame: FrameId, request: &Request) -> Result<()> {
        ensure!(request.order <= TREE_ORDER, "Invalid order {request:?}");
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
        ensure!(
            self.locals.cluster_locals(request.cluster).is_some(),
            "Invalid cluster {:?}",
            request.cluster
        );
        Ok(())
    }

    fn reserve_or_steal(
        &self,
        i: TreeId,
        order: usize,
        cluster: Cluster,
        local: usize,
    ) -> Result<(FrameId, Cluster)> {
        if let Some((reserved, free, target_cluster)) =
            self.trees
                .reserve_or_steal(i, cluster, 1 << order, self.policy)
        {
            let cluster_len = self
                .locals
                .cluster_locals(target_cluster)
                .expect("Invalid cluster");
            // Target might have less locals or none
            assert!(cluster_len > 0, "No locals for cluster {target_cluster:?}");
            let local = local % cluster_len;

            // Perform lower alloc, if it fails undo reservation
            match self.lower.get(i.as_row(), order, None) {
                Ok(frame) => {
                    // Swap and unreserve old tree
                    if reserved
                        && let Some(Reservation { row, free, .. }) = self.locals.swap(
                            target_cluster,
                            local,
                            frame.as_tree(),
                            free - (1 << order),
                        )
                    {
                        self.trees
                            .unreserve(row.as_tree(), free, target_cluster, self.policy);
                    }
                    Ok((frame, target_cluster))
                }
                Err(e) => {
                    if reserved {
                        self.trees.unreserve(i, free, target_cluster, self.policy);
                    } else {
                        self.trees.put(i, 1 << order, self.policy);
                    }
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn steal_global(
        &self,
        i: TreeId,
        cluster: Cluster,
        order: usize,
        frame: Option<FrameId>,
    ) -> Result<(FrameId, Cluster)> {
        if let Some(cluster) = self.trees.steal(i, cluster, 1 << order, self.policy) {
            match self.lower.get(i.as_row(), order, frame) {
                Ok(frame) => Ok((frame, cluster)),
                Err(e) => {
                    self.trees.put(i, 1 << order, self.policy);
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn get_at(&self, frame: FrameId, request: Request) -> Result<(FrameId, Cluster)> {
        // Try local reservation first
        if let Some(local) = request.local {
            match self.get_local(request.order, request.cluster, local, Some(frame), true) {
                Err((Error::Memory, _)) => {} // continue with global
                Err((e, _)) => return Err(e),
                Ok(r) => return Ok(r),
            }
        }

        // Fallback to global reservation
        match self.steal_global(frame.as_tree(), request.cluster, request.order, Some(frame)) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }

        // Last resort, steal or downgrade any local reservation
        match self.steal_local(&request, Some(frame)) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }

        self.demote_local(&request, Some(frame))
    }

    /// Try decrementing the local reservation for the given cluster and local index.
    fn get_local(
        &self,
        order: usize,
        cluster: Cluster,
        local: usize,
        frame: Option<FrameId>,
        sync: bool,
    ) -> core::result::Result<(FrameId, Cluster), (Error, Option<TreeId>)> {
        match self
            .locals
            .get(cluster, local, frame.map(FrameId::as_tree), 1 << order)
        {
            Ok(row) => match self.lower.get(row, order, frame) {
                Ok(frame) => {
                    // Update current bitfield row if it changed
                    if row != frame.as_row() {
                        self.locals.set_start(cluster, local, frame.as_row());
                    }
                    Ok((frame, cluster))
                }
                Err(e) => {
                    debug!("local_lower failed {cluster:?} {}", row.as_tree());
                    self.trees.put(row.as_tree(), 1 << order, self.policy);
                    Err((e, Some(row.as_tree())))
                }
            },
            Err(Some(Reservation { row, free, .. })) => {
                // Sync with global tree
                if sync {
                    let min = (1 << order) - free;
                    if let Some(free) = self.trees.sync(row.as_tree(), min) {
                        if self.locals.put(cluster, local, row.as_tree(), free) {
                            // retry
                            return self.get_local(order, cluster, local, frame, false);
                        } else {
                            // undo tree change
                            self.trees.put(row.as_tree(), free, self.policy);
                        }
                    }
                }
                Err((Error::Memory, Some(row.as_tree())))
            }
            Err(None) => Err((Error::Memory, None)),
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn search_and_reserve(
        &self,
        order: usize,
        cluster: Cluster,
        local: usize,
        start: TreeId,
    ) -> Result<(FrameId, Cluster)> {
        debug!("reserve {cluster:?} start");
        // Rate how if and how good the tree fulfills the allocation
        let rate = |t, free| {
            if free >= (1 << order) {
                (self.policy)(cluster, t, free)
            } else {
                // Skip if not in range
                Policy::Invalid
            }
        };

        // Try to reserve a new tree
        let reserve_or_steal = |i| self.reserve_or_steal(i, order, cluster, local);

        const CL: usize = align_of::<Align>() / 4;

        // Why does 16 work so well? Are there better values?
        let near = (self.trees.len() / 16).max(CL / 4);

        // Why does align twice near help?
        // It leaves some space between starting points...
        let start = TreeId(align_down(start.0, (2 * near).next_power_of_two()));

        // Find best fit in neighborhood
        if order < HUGE_ORDER {
            match self.trees.search_best::<3, _>(
                start,
                1,
                near,
                |t, f| match rate(t, f) {
                    // Only match or empty
                    p @ Policy::Match(_) => p,
                    p @ Policy::Demote if f == TREE_FRAMES => p,
                    _ => Policy::Invalid,
                },
                reserve_or_steal,
            ) {
                Err(Error::Memory) => {}
                r => return r,
            }
            debug!("reserve {cluster:?} no near");
        }

        // Global search
        self.trees.search_best::<8, _>(
            start,
            0,
            self.trees.len(),
            |t, f| match rate(t, f) {
                // Direct allocation of matching or empty
                Policy::Match(_) => Policy::Match(u8::MAX),
                Policy::Demote if f == TREE_FRAMES => Policy::Match(u8::MAX),
                p => p,
            },
            reserve_or_steal,
        )
    }

    /// Steal from a local reservation and possibly demote or drain it
    fn demote_local(
        &self,
        request: &Request,
        frame: Option<FrameId>,
    ) -> Result<(FrameId, Cluster)> {
        if let Some((row, old)) = self.locals.demote_any(
            request.cluster,
            request.local,
            frame.map(FrameId::as_tree),
            request.frames(),
            self.policy,
        ) {
            // Unreserve the old reservation (or the demoted tree if request.local is None)
            if let Some(Reservation { row, cluster, free }) = old {
                self.trees
                    .unreserve(row.as_tree(), free, cluster, self.policy);
            }

            match self.lower.get(row, request.order, frame) {
                Err(Error::Memory) => {
                    // undo counter decrement
                    self.trees.put(row.as_tree(), request.frames(), self.policy);
                }
                r => return r.map(|frame| (frame, request.cluster)),
            }
        }
        Err(Error::Memory)
    }

    fn steal_local(&self, request: &Request, frame: Option<FrameId>) -> Result<(FrameId, Cluster)> {
        if let Some(reservation) = self.locals.steal_any(
            request.cluster,
            request.local,
            frame.map(FrameId::as_tree),
            request.frames(),
            self.policy,
        ) {
            match self.lower.get(reservation.row, request.order, frame) {
                Err(Error::Memory) => {
                    self.trees
                        .put(reservation.row.as_tree(), 1 << request.order, self.policy);
                }
                r => return r.map(|frame| (frame, reservation.cluster)),
            }
        }
        Err(Error::Memory)
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
    fn valid(&self, m: &MetaSize) -> bool {
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
