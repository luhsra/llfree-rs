//! Upper allocator implementation

use core::fmt;
use core::ops::Range;

use log::{debug, info, warn};

use crate::local::{Locals, Reservation};
use crate::lower::Lower;
use crate::trees::{TreeId, Trees};
use crate::util::{Align, align_down};
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
/// Allocations/frees within a chunk are handed over to the
/// lower allocator.
/// These chunks are, due to the inner workings of the lower allocator,
/// called *trees*.
/// This allocator stores these tree entries in a [packed array][Trees].
///
/// Additionally, the allocator manages user-provided [tiers][Tier].
/// Tiers are used to separate trees into different groups.
/// The users also has to provide a [policy function][PolicyFn] that defines
/// how to access the tiers.
///
/// Each tier can have a different number of [local reservations][Locals],
/// which are used to reduce contention on the tree array.
///
/// If an allocation for a certain tier cannot be fulfilled,
/// the allocator falls back on stealing from other tiers or
/// demoting the request to a lower tier, depending on the [policy][Policy].
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
            meta.valid(&Self::metadata_size(tiering, frames)),
            "Invalid metadata"
        );

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Initialize per-CPU data
        let locals = Locals::new(meta.local, tiering)?;

        // Init tree array
        let tree_init = if init == Init::None {
            None
        } else {
            Some(|start| lower.stats_at(FrameId(start), TREE_ORDER).free_frames)
        };
        let trees = Trees::new(frames, meta.trees, tree_init, tiering.default);

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

    fn get(&self, frame: Option<FrameId>, request: Request) -> Result<(FrameId, Tier)> {
        self.check(frame.unwrap_or(FrameId(0)), &request)?;
        // Different starting points for each core

        if let Some(frame) = frame {
            return self.get_at(frame, request);
        }

        let len = self.locals.tier_locals(request.tier).unwrap_or_default();
        let mut start_idx = TreeId(
            self.trees.len().checked_div(len).unwrap_or(0) * request.local.unwrap_or_default(),
        );

        // Try local reservation first
        match self.get_matching(&request, &mut start_idx) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }

        // Demote local / global
        match self.get_demoting(&request, &mut start_idx) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }

        // -- Out of memory handling ---
        warn!("OOM");

        // Downgrade request until successful
        for t in 1..Tier::LEN {
            // Go downwards
            let tier = Tier((Tier::LEN + request.tier.0 - t) % Tier::LEN);
            if let Some(locals) = self.locals.tier_locals(tier) {
                let request = Request {
                    tier,
                    order: request.order,
                    local: request.local.map(|v| v % locals), // adjust to fit
                };
                match self.get_matching(&request, &mut start_idx) {
                    Err(Error::Memory) => {} // try next tier
                    r => return r,
                }
            }
        }
        // Try stealing from local reservations
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

        // Try update own trees first
        if let Some(local) = request.local {
            // Update the put-reserve heuristic
            let may_reserve = cfg_select! {
                feature = "free_reserve" => self.locals.frees_push(local, i),
                _ => false,
            };

            let tier = request.tier;
            if self
                .locals
                .put(tier, local, frame.as_tree(), request.frames())
            {
                return Ok(());
            }

            // Increment or reserve globally
            if let Some(free) =
                self.trees
                    .put_or_reserve(i, request.frames(), tier, may_reserve, self.policy)
            {
                warn!("free reserved tree idx={i} tier={tier:?} free={free}");
                // Change preferred tree to speedup future frees
                if let Some(Reservation { row, free, .. }) =
                    self.locals.swap(tier, local, i, free + request.frames())
                {
                    self.trees.unreserve(row.as_tree(), free, tier, self.policy);
                }
            }
        } else {
            self.trees.put(i, request.frames(), self.policy);
        }

        Ok(())
    }

    fn frames(&self) -> usize {
        self.lower.frames()
    }

    fn drain(&self) {
        self.locals.drain(|row, tier, free| {
            self.trees.unreserve(row.as_tree(), free, tier, self.policy);
        });
    }

    fn tree_stats(&self) -> TreeStats {
        let mut stats = self.trees.stats();
        let l_stats = self.locals.stats();
        stats.free_frames += l_stats.free_frames;
        stats.free_trees += l_stats.free_trees;
        for (t, lt) in stats.tiers.iter_mut().zip(l_stats.tiers.iter()) {
            t.free_frames += lt.free_frames;
            t.alloc_frames += lt.alloc_frames;
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
        for tier in (0..Tier::LEN).map(Tier) {
            if let Some(len) = self.locals.tier_locals(tier) {
                for local in 0..len {
                    if let Some(Reservation { row, free, .. }) = self.locals.load(tier, local) {
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
        ensure!(
            self.locals.tier_locals(request.tier).is_some(),
            "Invalid tier {:?}",
            request.tier
        );
        Ok(())
    }

    fn get_reserve(
        &self,
        order: usize,
        i: TreeId,
        local: usize,
        check: impl Fn(Tier, usize) -> Option<Tier>,
    ) -> Result<(FrameId, Tier)> {
        if let Some((free, target_tier)) = self.trees.reserve(i, check) {
            let tier_len = self.locals.tier_locals(target_tier).expect("Invalid tier");
            // Target might have less locals or none
            assert!(tier_len > 0, "No locals for tier {target_tier:?}");
            let local = local % tier_len;

            // Perform lower alloc, if it fails undo reservation
            match self.lower.get(i.as_row(), order, None) {
                Ok(frame) => {
                    // Swap and unreserve old tree
                    if let Some(Reservation { row, free, .. }) =
                        self.locals
                            .swap(target_tier, local, frame.as_tree(), free - (1 << order))
                    {
                        self.trees
                            .unreserve(row.as_tree(), free, target_tier, self.policy);
                    }
                    Ok((frame, target_tier))
                }
                Err(e) => {
                    self.trees.unreserve(i, free, target_tier, self.policy);
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn get_global(
        &self,
        i: TreeId,
        order: usize,
        frame: Option<FrameId>,
        check: impl Fn(Tier, usize) -> Option<Tier>,
    ) -> Result<(FrameId, Tier)> {
        if let Some(tier) = self.trees.get(i, 1 << order, check) {
            match self.lower.get(i.as_row(), order, frame) {
                Ok(frame) => Ok((frame, tier)),
                Err(e) => {
                    self.trees.put(i, 1 << order, self.policy);
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn get_at(&self, frame: FrameId, request: Request) -> Result<(FrameId, Tier)> {
        // Try local reservation first
        if let Some(local) = request.local {
            let mut start_idx = TreeId(0);
            for _ in 0..RETRIES {
                match self.get_matching_local(
                    request.order,
                    request.tier,
                    local,
                    Some(frame),
                    &mut start_idx,
                ) {
                    Err(Error::Retry) => {}
                    Err(Error::Memory) => break, // continue with global
                    r => return r,
                }
            }
        }

        // Fallback to global reservation
        match self.get_global(frame.as_tree(), request.order, Some(frame), |t, f| {
            if f >= request.frames() {
                return match (self.policy)(request.tier, t, f) {
                    Policy::Match(_) | Policy::Steal => Some(t),
                    Policy::Demote => Some(request.tier),
                    Policy::Invalid => None,
                };
            }
            None
        }) {
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

    fn get_matching(&self, request: &Request, start_idx: &mut TreeId) -> Result<(FrameId, Tier)> {
        if let Some(local) = request.local
            && self
                .locals
                .tier_locals(request.tier)
                .is_some_and(|len| len > 0 && len < self.trees.len())
        {
            for _ in 0..RETRIES {
                match self.get_matching_local(request.order, request.tier, local, None, start_idx) {
                    Err(Error::Retry) => {}
                    Err(Error::Memory) => break, // continue with global
                    r => return r,
                }
            }
            // Try reserve new tree if no specific frame is requested
            self.get_matching_reserve(request.order, request.tier, local, *start_idx)
        } else {
            let check = |t: Tier, f: usize| {
                if f < request.frames() {
                    return None;
                }
                match (self.policy)(request.tier, t, f) {
                    Policy::Match(_) => Some(t),
                    Policy::Demote if f == TREE_FRAMES => Some(request.tier),
                    _ => None,
                }
            };
            // Any frame
            self.trees.search(*start_idx, 0, self.trees.len(), |i| {
                self.get_global(i, request.order, None, check)
            })
        }
    }

    fn get_matching_local(
        &self,
        order: usize,
        tier: Tier,
        local: usize,
        frame: Option<FrameId>,
        start_idx: &mut TreeId,
    ) -> core::result::Result<(FrameId, Tier), Error> {
        match self
            .locals
            .get(tier, local, frame.map(FrameId::as_tree), 1 << order)
        {
            Ok(row) => match self.lower.get(row, order, frame) {
                Ok(frame) => {
                    if row != frame.as_row() {
                        self.locals.set_start(tier, local, frame.as_row());
                    }
                    Ok((frame, tier))
                }
                Err(e) => {
                    self.trees.put(row.as_tree(), 1 << order, self.policy);
                    *start_idx = row.as_tree();
                    Err(e)
                }
            },
            Err(Some(Reservation { row, free, .. })) => {
                *start_idx = row.as_tree();
                // Sync with global tree
                let min = (1 << order) - free;
                if let Some(free) = self.trees.sync(row.as_tree(), min) {
                    if self.locals.put(tier, local, row.as_tree(), free) {
                        Err(Error::Retry)
                    } else {
                        self.trees.put(row.as_tree(), free, self.policy);
                        Err(Error::Memory)
                    }
                } else {
                    Err(Error::Memory)
                }
            }
            Err(None) => Err(Error::Memory),
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn get_matching_reserve(
        &self,
        order: usize,
        tier: Tier,
        local: usize,
        start: TreeId,
    ) -> Result<(FrameId, Tier)> {
        // Rate how if and how good the tree fulfills the allocation
        let rate = |t, free, range: Range<usize>| {
            if range.contains(&free) {
                (self.policy)(tier, t, free)
            } else {
                // Skip if not in range
                Policy::Invalid
            }
        };

        // Try to reserve a new tree
        let reserve = |i: TreeId, range: Range<usize>| {
            let check = |t: Tier, f: usize| {
                match rate(t, f, range.clone()) {
                    Policy::Match(_) => Some(tier),
                    // allow demotion if tree is empty
                    Policy::Demote if f == TREE_FRAMES => Some(tier),
                    _ => None,
                }
            };
            self.get_reserve(order, i, local, check)
        };

        const CL: usize = align_of::<Align>() / 4;

        // Why does 16 work so well? Are there better values?
        let near = (self.trees.len() / 16).max(CL / 4);

        // Why does align twice near help?
        // It leaves some space between starting points...
        let start = TreeId(align_down(start.0, (2 * near).next_power_of_two()));

        // Find best fit in fragmented trees
        if order < HUGE_ORDER {
            let range = 1 << order..TREE_FRAMES;

            match self.trees.search_best::<3, _>(
                start,
                1,
                near,
                |t, f| rate(t, f, range.clone()),
                |i| reserve(i, range.clone()),
            ) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }

        // Any
        self.trees.search(start, 0, self.trees.len(), |i| {
            reserve(i, 1 << order..usize::MAX)
        })
    }

    /// Allocate with demotion if needed, but without downgrading the request
    fn get_demoting(&self, request: &Request, start_idx: &mut TreeId) -> Result<(FrameId, Tier)> {
        let check = |tier, free| {
            if free < request.frames() {
                return None;
            }
            match (self.policy)(request.tier, tier, free) {
                Policy::Demote => Some(request.tier),
                _ => None,
            }
        };

        if let Some(local) = request.local
            && self
                .locals
                .tier_locals(request.tier)
                .is_some_and(|len| len > 0 && len < self.trees.len())
        {
            // Try reserving new tree
            self.trees.search(*start_idx, 0, self.trees.len(), |i| {
                self.get_reserve(request.order, i, local, check)
            })
        } else {
            // Try global allocation
            self.trees.search(*start_idx, 0, self.trees.len(), |i| {
                self.get_global(i, request.order, None, check)
            })
        }
    }

    /// Steal from a local reservation and possibly demote or drain it
    fn demote_local(&self, request: &Request, frame: Option<FrameId>) -> Result<(FrameId, Tier)> {
        if let Some((row, old)) = self.locals.demote_any(
            request.tier,
            request.local,
            frame.map(FrameId::as_tree),
            request.frames(),
            self.policy,
        ) {
            // Unreserve the old reservation (or the demoted tree if request.local is None)
            if let Some(Reservation { row, tier, free }) = old {
                self.trees.unreserve(row.as_tree(), free, tier, self.policy);
            }

            match self.lower.get(row, request.order, frame) {
                Err(Error::Memory) => {
                    // undo counter decrement
                    self.trees.put(row.as_tree(), request.frames(), self.policy);
                }
                r => return r.map(|frame| (frame, request.tier)),
            }
        }
        Err(Error::Memory)
    }

    fn steal_local(&self, request: &Request, frame: Option<FrameId>) -> Result<(FrameId, Tier)> {
        if let Some(reservation) = self.locals.steal_any(
            request.tier,
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
                r => return r.map(|frame| (frame, reservation.tier)),
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
