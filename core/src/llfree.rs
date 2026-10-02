//! Upper allocator implementation

use core::fmt;
use core::slice;

use log::{debug, info};
use spin::mutex::SpinMutex;

use crate::cache::{Align, Aligned, Invalidate};
use crate::cxl::{CXLock, CXLockGuard};
use crate::local::{FreeClone, Locals, Reservation, TreeData};
use crate::lower::LowerTree;
use crate::trees::{TreeId, Trees};
use crate::util::align_down;
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
/// Additionally, the allocator manages user-provided [classes][Class].
/// Classes are used to separate trees into different groups.
/// The user also has to provide a [policy function][PolicyFn] that defines
/// how to access the classes.
///
/// Each class can have a different number of [local reservations][Locals],
/// which are used to reduce contention on the tree array.
///
/// If an allocation for a certain class cannot be fulfilled,
/// the allocator falls back on stealing from other classes or
/// demoting the request to a lower class, depending on the [policy][Policy].
///
/// Statistics include shared free capacity and this host's reservation/free clones.
/// Other hosts' private clones are not visible. Concurrent local operations can
/// make statistics approximate; validation requires quiescent local operations.
#[repr(align(64))]
pub struct LLFree<'a> {
    /// Number of managed frames
    frames: usize,
    /// Policy for accessing tree classes
    policy: PolicyFn,
    /// CPU local data
    ///
    /// Other CPUs can access this if they drain cores.
    /// Also, these are shared between CPUs if we have more cores than trees.
    locals: Locals<'a>,
    /// Lock for the uncached data
    cxl_lock: SpinMutex<CXLock<'a>>,
    /// Manages the allocator's trees.
    ///
    /// Protected by the `cxl_lock`.
    pub trees: Trees<'a>,
    /// Metadata of the lower alloc
    ///
    /// Protected by the `cxl_lock`.
    pub lower: &'a [Uncached<Align<LowerTree>>],
}

unsafe impl Send for LLFree<'_> {}
unsafe impl Sync for LLFree<'_> {}

struct CXLMetaSize {
    lock: usize,
    lower: usize,
    trees: usize,
}
impl CXLMetaSize {
    fn new(hosts: usize, frames: usize) -> Self {
        let num_trees = frames.div_ceil(TREE_FRAMES);
        Self {
            lock: CXLock::metadata_size(hosts),
            lower: num_trees * size_of::<Align<LowerTree>>().div_ceil(CacheLine::SIZE),
            trees: Trees::metadata_size(frames),
        }
    }
    fn sum(&self) -> usize {
        self.lock + self.lower + self.trees
    }
}

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
    fn new(
        hosts: usize,
        host_id: usize,
        frames: usize,
        init: Init,
        classing: &Classing,
        meta: MetaData<'a>,
    ) -> Result<Self> {
        info!("initializing f={frames} {classing:?} {meta:?}");
        ensure!(
            Error::Initialization;
            meta.valid(&Self::metadata_size(hosts, frames, classing)),
            "Invalid metadata"
        );

        let num_trees = frames.div_ceil(TREE_FRAMES);

        // Initialize host-local per-CPU data
        let locals = Locals::new(meta.local, classing)?;

        // Initialize CXL data
        let cms = CXLMetaSize::new(hosts, frames);
        let buffer = meta.remote;
        let (lock, buffer) = buffer.split_at(cms.lock);
        let (trees, lower) = buffer.split_at(cms.trees);

        // CXL Lock
        let cxl_lock = SpinMutex::new(match init {
            Init::AllocAll | Init::FreeAll => CXLock::init(host_id, hosts, lock)?,
            Init::Recover | Init::None => CXLock::join(host_id, hosts, lock)?,
        });

        let lower = unsafe { lower.cast_slice::<Align<LowerTree>>(num_trees)?.transpose() };
        // Init lower allocator
        let tree_init = if matches!(init, Init::AllocAll | Init::FreeAll) {
            for (i, ltree) in lower.iter().enumerate() {
                unsafe {
                    ltree
                        .inner()
                        .init((frames - i * TREE_FRAMES).min(TREE_FRAMES), init)
                };
            }
            Some(|start: TreeId| unsafe { lower[start.0].inner().stats().free_frames })
        } else {
            None
        };

        // Init tree array
        let trees = Trees::new(frames, trees, tree_init, classing.default)?;
        if matches!(init, Init::AllocAll | Init::FreeAll) {
            lower.flush_invalidate();
        }

        Ok(Self {
            frames,
            policy: classing.policy,
            locals,
            cxl_lock,
            trees,
            lower,
        })
    }

    fn metadata_size(hosts: usize, frames: usize, classing: &Classing) -> MetaSize {
        MetaSize {
            local: Locals::metadata_size(classing),
            remote: CXLMetaSize::new(hosts, frames).sum(),
        }
    }

    unsafe fn metadata(&mut self) -> MetaData<'a> {
        let lock = self.cxl_lock.lock();
        unsafe {
            MetaData {
                local: self.locals.metadata(),
                remote: Uncached::from_ref(slice::from_raw_parts(
                    lock.metadata().as_ptr(),
                    CXLMetaSize::new(lock.hosts(), self.frames).sum(),
                )),
            }
        }
    }

    fn get(&self, frame: Option<FrameId>, request: Request) -> Result<(FrameId, Class)> {
        self.check(frame.unwrap_or(FrameId(0)), &request)?;

        // Try reserving a specific frame
        if let Some(frame) = frame {
            return self.get_at(frame, request);
        }

        let len = self.locals.class_locals(request.class).unwrap_or(0);
        let trees_len = self.frames.div_ceil(TREE_FRAMES);
        // Different starting points for each core
        let mut start_idx =
            TreeId(trees_len.checked_div(len).unwrap_or(0) * request.local.unwrap_or(0));

        // Use local reservation if possible
        if let Some(local) = request.local
            && self
                .locals
                .class_locals(request.class)
                .is_some_and(|len| len > 0 && len < trees_len)
        {
            match self.get_local(request.order, request.class, local, None, true) {
                Err((Error::Memory, Some(start))) => start_idx = start,
                Err((Error::Memory, _)) => {}
                Err((e, _)) => return Err(e),
                Ok(r) => return Ok((r, request.class)),
            }
            // Try reserving new tree
            match self.search_and_reserve(request.order, request.class, local, start_idx) {
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
                (self.policy)(request.class, t, free)
            };
            // Any frame

            let mut inner = self.cxl_lock.lock();
            let guard = inner.lock();

            match self
                .trees
                .search_best::<8, _>(&guard, start_idx, 0, trees_len, rate, |i| {
                    self.steal_global(&guard, i, request.class, request.order, None)
                }) {
                Err(Error::Memory) => {} // continue
                r => return r,
            }
        }

        // -- Out of memory handling ---
        debug!("OOM {:?}", request.class);

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

    fn put(&self, frame: FrameId, request: Request, free_idx: usize) -> Result<()> {
        self.check(frame, &request)?;

        match self
            .locals
            .put(request.class, request.local, frame, request.order, free_idx)
        {
            Err(Error::Memory) => {}
            r => return r,
        }

        // Create new free clone
        let clone = FreeClone::new(
            TreeData::new(frame.as_row(), request.frames()),
            LowerTree::default(),
        );
        clone.lower.init(TREE_FRAMES, Init::AllocAll);
        clone.lower.put(frame.inside_tree(), request.order).unwrap();
        let old = self.locals.switch_free_clone(free_idx, Some(clone));

        if let Some(FreeClone { data, lower }) = old {
            let mut inner = self.cxl_lock.lock();
            let mut _guard = inner.lock();
            unsafe {
                let i = data.row.as_tree();
                // First free the frame in the lower allocator
                self.lower[i.0].borrow().merge_frees(lower);
                // Increment globally
                self.trees
                    .put(&_guard, data.row.as_tree(), data.free, self.policy);
            }
        }
        Ok(())
    }

    fn frames(&self) -> usize {
        self.frames
    }

    fn drain(&self) {
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();

        self.locals.drain(
            |Reservation { class, data, lower }| {
                let l = unsafe { self.lower[data.row.as_tree().0].borrow() };
                l.merge_frees(lower);
                drop(l);
                self.trees
                    .unreserve(&guard, data.row.as_tree(), data.free, class, self.policy);
            },
            |FreeClone { data, lower }| {
                let l = unsafe { self.lower[data.row.as_tree().0].borrow() };
                l.merge_frees(lower);
                self.trees
                    .put(&guard, data.row.as_tree(), data.free, self.policy);
            },
        );
    }

    fn tree_stats(&self) -> TreeStats {
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();
        self.tree_stats_locked(&guard)
    }

    fn stats(&self) -> Stats {
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();
        (0..self.lower.len())
            .map(TreeId)
            .fold(Stats::default(), |stats, tree| {
                stats + self.detailed_tree_stats(&guard, tree)
            })
    }

    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();
        self.stats_at_locked(&guard, frame, order)
    }

    fn change_tree(&self, matcher: TreeMatch, change: TreeChange) -> Result<()> {
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();
        self.trees.change(&guard, matcher, change, |i| {
            let lower = unsafe { self.lower[i.0].borrow() };
            lower.stats().free_frames
        })
    }

    fn validate(&self) {
        debug!("validate");
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();
        let fast_stats = self.tree_stats_locked(&guard);
        let mut full_stats = Stats::default();
        for tree in (0..self.trees.len()).map(TreeId) {
            let (_, free, _) = self.trees.stats_at(tree);
            let lower = unsafe { self.lower[tree.0].borrow() };
            assert!(lower.verify());
            assert_eq!(free, lower.stats().free_frames);
            drop(lower);
            self.locals.for_each_lower(tree, |free, lower| {
                assert!(lower.verify());
                assert_eq!(free, lower.stats().free_frames);
            });
            let stats = self.detailed_tree_stats(&guard, tree);
            assert!(stats.free_frames <= (self.frames - tree.as_frame().0).min(TREE_FRAMES));
            full_stats = full_stats + stats;
        }
        self.locals.for_each_tree_data(|tree, _, class| {
            if class.is_some() {
                assert!(self.trees.stats_at(tree).2);
            }
        });
        assert_eq!(fast_stats.free_frames, full_stats.free_frames);
        assert_eq!(fast_stats.free_trees, full_stats.free_trees);
    }
}

impl<'a> LLFree<'a> {
    fn tree_stats_locked(&self, _guard: &CXLockGuard<'_, 'a>) -> TreeStats {
        let mut stats = self.trees.stats();
        let tail = self.frames % TREE_FRAMES;
        if tail != 0 {
            let (class, _, _) = self.trees.stats_at(TreeId(self.trees.len() - 1));
            stats.classes[class.0 as usize].alloc_frames -= TREE_FRAMES - tail;
        }
        self.locals
            .for_each_tree_data(|tree, local_free, local_class| {
                let (global_class, global_free, _) = self.trees.stats_at(tree);
                let capacity = (self.frames - tree.as_frame().0).min(TREE_FRAMES);

                let class = local_class.unwrap_or(global_class);
                if class != global_class {
                    let global = &mut stats.classes[global_class.0 as usize];
                    global.free_frames = global.free_frames.saturating_sub(global_free);
                    global.alloc_frames =
                        global.alloc_frames.saturating_sub(capacity - global_free);
                    let local = &mut stats.classes[class.0 as usize];
                    local.free_frames += global_free;
                    local.alloc_frames += capacity - global_free;
                }
                stats.free_frames += local_free;
                stats.free_trees += usize::from(
                    global_free < TREE_FRAMES && global_free + local_free == TREE_FRAMES,
                );
                let class = &mut stats.classes[class.0 as usize];
                class.free_frames += local_free;
                class.alloc_frames = class.alloc_frames.saturating_sub(local_free);
            });
        stats
    }

    fn detailed_tree_stats(&self, _guard: &CXLockGuard<'_, 'a>, tree: TreeId) -> Stats {
        let lower = unsafe { self.lower[tree.0].borrow() };
        let mut huge_free = core::array::from_fn::<_, TREE_HUGE, _>(|i| {
            lower
                .stats_at(FrameId(i * HUGE_FRAMES), HUGE_ORDER)
                .free_frames
        });
        drop(lower);
        self.locals.for_each_lower(tree, |_, lower| {
            for (i, free) in huge_free.iter_mut().enumerate() {
                *free += lower
                    .stats_at(FrameId(i * HUGE_FRAMES), HUGE_ORDER)
                    .free_frames;
            }
        });
        let free_frames = huge_free.iter().sum();
        Stats {
            free_frames,
            free_huge: huge_free
                .iter()
                .filter(|&&free| free == HUGE_FRAMES)
                .count(),
            free_trees: usize::from(free_frames == TREE_FRAMES),
        }
    }

    fn stats_at_locked(&self, guard: &CXLockGuard<'_, 'a>, frame: FrameId, order: usize) -> Stats {
        let tree = frame.as_tree();
        if order == TREE_ORDER {
            return self.detailed_tree_stats(guard, tree);
        }
        if order != HUGE_ORDER && order != 0 {
            return Stats::default();
        }
        let lower = unsafe { self.lower[tree.0].borrow() };
        let mut free_frames = lower.stats_at(frame.inside_tree(), order).free_frames;
        drop(lower);
        self.locals.for_each_lower(tree, |_, lower| {
            free_frames += lower.stats_at(frame.inside_tree(), order).free_frames;
        });
        Stats {
            free_frames,
            free_huge: usize::from(order == HUGE_ORDER && free_frames == HUGE_FRAMES),
            free_trees: 0,
        }
    }

    fn check(&self, frame: FrameId, request: &Request) -> Result<()> {
        ensure!(request.order <= TREE_ORDER, "Invalid order {request:?}");
        ensure!(
            frame.0 + (1 << request.order) <= self.frames(),
            "Frame {} out of bounds",
            frame.0
        );
        ensure!(
            frame.0.is_multiple_of(1 << request.order),
            "Frame {} misaligned",
            frame.0
        );
        ensure!(
            self.locals.class_locals(request.class).is_some(),
            "Invalid class {:?}",
            request.class
        );
        Ok(())
    }

    fn reserve_or_steal(
        &self,
        guard: &CXLockGuard<'_, 'a>,
        i: TreeId,
        order: usize,
        class: Class,
        local: usize,
    ) -> Result<(FrameId, Class)> {
        if let Some((reserved, free, target_class)) =
            self.trees
                .reserve_or_steal(guard, i, class, 1 << order, self.policy)
        {
            let lower = unsafe { self.lower[i.0].borrow() };
            if !reserved {
                return match lower.get(FrameId(0).as_row(), order, None) {
                    Ok(frame) => Ok((i.as_frame() + frame, target_class)),
                    Err(e) => {
                        self.trees.put(guard, i, 1 << order, self.policy);
                        Err(e)
                    }
                };
            }

            let class_len = self
                .locals
                .class_locals(target_class)
                .expect("Invalid class");
            assert!(class_len > 0, "No locals for class {target_class:?}");
            let local = local % class_len;
            let clone = lower.get_all_copy();

            match clone.get(FrameId(0).as_row(), order, None) {
                Ok(frame) => {
                    let frame = i.as_frame() + frame;
                    let new = Reservation::new(
                        target_class,
                        TreeData::new(frame.as_row(), free - (1 << order)),
                        clone,
                    );
                    // Flush the emptied source before publishing its clone.
                    drop(lower);
                    if let Some(Reservation { class, data, lower }) = self
                        .locals
                        .switch_reservation(target_class, local, Some(new))
                    {
                        let l = unsafe { self.lower[data.row.as_tree().0].borrow() };
                        l.merge_frees(lower);
                        drop(l);

                        self.trees.unreserve(
                            guard,
                            data.row.as_tree(),
                            data.free,
                            class,
                            self.policy,
                        );
                    }
                    Ok((frame, target_class))
                }
                Err(e) => {
                    lower.merge_frees(clone);
                    drop(lower);
                    self.trees
                        .unreserve(guard, i, free, target_class, self.policy);
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn steal_global(
        &self,
        guard: &CXLockGuard<'_, 'a>,
        i: TreeId,
        class: Class,
        order: usize,
        frame: Option<FrameId>,
    ) -> Result<(FrameId, Class)> {
        if let Some(class) = self.trees.steal(guard, i, class, 1 << order, self.policy) {
            let lower = unsafe { self.lower[i.0].borrow() };

            match lower.get(FrameId(0).as_row(), order, frame.map(FrameId::inside_tree)) {
                Ok(frame) => Ok((i.as_frame() + frame, class)),
                Err(e) => {
                    self.trees.put(guard, i, 1 << order, self.policy);
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn get_at(&self, frame: FrameId, request: Request) -> Result<(FrameId, Class)> {
        // Try local reservation first
        if let Some(local) = request.local {
            match self.get_local(request.order, request.class, local, Some(frame), true) {
                Err((Error::Memory, _)) => {} // continue with global
                Err((e, _)) => return Err(e),
                Ok(r) => return Ok((r, request.class)),
            }
        }

        // Fallback to global reservation
        {
            let mut inner = self.cxl_lock.lock();
            let guard = inner.lock();
            match self.steal_global(
                &guard,
                frame.as_tree(),
                request.class,
                request.order,
                Some(frame),
            ) {
                Err(Error::Memory) => {} // continue
                r => return r,
            }
        }

        // Last resort, steal or downgrade any local reservation
        match self.steal_local(&request, Some(frame)) {
            Err(Error::Memory) => {} // continue
            r => return r,
        }

        self.demote_local(&request, Some(frame))
    }

    /// Try decrementing the local reservation for the given class and local index.
    fn get_local(
        &self,
        order: usize,
        class: Class,
        local: usize,
        frame: Option<FrameId>,
        sync: bool,
    ) -> core::result::Result<FrameId, (Error, Option<TreeId>)> {
        match self.locals.get(class, local, frame, order) {
            Ok(frame) => Ok(frame),
            Err(Some(TreeData { row, free })) => {
                // Sync with global tree
                if sync {
                    let mut inner = self.cxl_lock.lock();
                    let guard = inner.lock();
                    let min = (1usize << order).saturating_sub(free);
                    if self.locals.sync(class, local, row.as_tree(), || {
                        let free = self.trees.sync(&guard, row.as_tree(), min)?;
                        let lower = unsafe { self.lower[row.as_tree().0].borrow() };
                        Some((lower.get_all_copy(), free))
                    }) {
                        // retry after both lower state and counters have transferred
                        return self.get_local(order, class, local, frame, false);
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
        class: Class,
        local: usize,
        start: TreeId,
    ) -> Result<(FrameId, Class)> {
        debug!("reserve {class:?} start");
        // Rate how if and how good the tree fulfills the allocation
        let rate = |t, free| {
            if free >= (1 << order) {
                (self.policy)(class, t, free)
            } else {
                // Skip if not in range
                Policy::Invalid
            }
        };

        // Try to reserve a new tree
        let mut inner = self.cxl_lock.lock();
        let guard = inner.lock();
        let reserve_or_steal = |i| self.reserve_or_steal(&guard, i, order, class, local);

        const CL: usize = align_of::<Align>() / 4;

        // Why does 16 work so well? Are there better values?
        let near = (self.trees.len() / 16).max(CL / 4);

        // Why does align twice near help?
        // It leaves some space between starting points...
        let start = TreeId(align_down(start.0, (2 * near).next_power_of_two()));

        // Find best fit in neighborhood
        if order < HUGE_ORDER {
            match self.trees.search_best::<3, _>(
                &guard,
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
            debug!("reserve {class:?} no near");
        }

        // Global search
        self.trees.search_best::<8, _>(
            &guard,
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
    fn demote_local(&self, request: &Request, frame: Option<FrameId>) -> Result<(FrameId, Class)> {
        if let Some((frame, old)) = self.locals.demote_any(
            request.class,
            request.local,
            frame,
            request.order,
            self.policy,
        ) {
            // Unreserve the old reservation (or the demoted tree if request.local is None)
            if let Some(Reservation { class, data, lower }) = old {
                let mut inner = self.cxl_lock.lock();
                let guard = inner.lock();
                // Merge into global
                let global = unsafe { self.lower[data.row.as_tree().0].borrow() };
                global.merge_frees(lower);
                drop(global);
                // Update counters
                self.trees
                    .unreserve(&guard, data.row.as_tree(), data.free, class, self.policy);
            }
            return Ok((frame, request.class));
        }
        Err(Error::Memory)
    }

    fn steal_local(&self, request: &Request, frame: Option<FrameId>) -> Result<(FrameId, Class)> {
        self.locals
            .steal_any(
                request.class,
                request.local,
                frame,
                request.order,
                self.policy,
            )
            .ok_or(Error::Memory)
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
        fn overlap(a: impl Aligned, b: impl Aligned) -> bool {
            if a.cache_lines() == 0 || b.cache_lines() == 0 {
                return false;
            }
            let a_start = unsafe { a.cache_line() } as usize;
            let b_start = unsafe { b.cache_line() } as usize;
            let a_end = a_start + a.cache_lines() * CacheLine::SIZE;
            let b_end = b_start + b.cache_lines() * CacheLine::SIZE;
            a_start < b_end && b_start < a_end
        }
        self.local.len() >= m.local
            && self.remote.len() >= m.remote
            && self.local.as_ptr().align_offset(align_of::<Align>()) == 0
            && self.remote.as_ptr().align_offset(align_of::<Align>()) == 0
            && !overlap(self.local, self.remote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocator(frames: usize, init: Init) -> LLFree<'static> {
        let (classing, _) = Classing::simple(1);
        let meta = MetaData::alloc(&LLFree::metadata_size(1, frames, &classing));
        LLFree::new(1, 0, frames, init, &classing, meta).unwrap()
    }

    fn assert_statistics(
        alloc: &LLFree<'_>,
        free_frames: usize,
        free_huge: usize,
        free_trees: usize,
    ) {
        let detailed = alloc.stats();
        assert_eq!(detailed.free_frames, free_frames);
        assert_eq!(detailed.free_huge, free_huge);
        assert_eq!(detailed.free_trees, free_trees);
        let trees = alloc.tree_stats();
        assert_eq!(trees.free_frames, free_frames);
        assert_eq!(trees.free_trees, free_trees);
        assert_eq!(
            trees
                .classes
                .iter()
                .map(|class| class.free_frames)
                .sum::<usize>(),
            free_frames
        );
        assert_eq!(
            trees
                .classes
                .iter()
                .map(|class| class.alloc_frames)
                .sum::<usize>(),
            alloc.frames() - free_frames
        );
        alloc.validate();
    }

    #[test]
    fn statistics_include_pending_frees_in_a_partial_tree() {
        let frames = 2 * TREE_FRAMES + 3;
        let alloc = allocator(frames, Init::FreeAll);
        assert_statistics(&alloc, frames, 2 * TREE_HUGE, 2);
        let frame = TreeId(2).as_frame() + FrameId(1);
        let request = Request::new(0, Class(0), None);
        alloc.get(Some(frame), request).unwrap();
        assert_statistics(&alloc, frames - 1, 2 * TREE_HUGE, 2);
        assert_eq!(alloc.stats_at(frame, 0).free_frames, 0);
        alloc.put(frame, request, 0).unwrap();
        assert_statistics(&alloc, frames, 2 * TREE_HUGE, 2);
        assert_eq!(alloc.stats_at(frame, 0).free_frames, 1);
        assert_eq!(alloc.stats_at(frame, HUGE_ORDER).free_frames, 3);
        assert_eq!(alloc.stats_at(frame, HUGE_ORDER).free_huge, 0);
        assert_eq!(alloc.stats_at(frame, TREE_ORDER).free_frames, 3);
        assert_eq!(alloc.stats_at(frame, TREE_ORDER).free_trees, 0);
        assert_eq!(unsafe { alloc.lower[2].inner() }.stats().free_frames, 2);
        alloc.drain();
        assert_statistics(&alloc, frames, 2 * TREE_HUGE, 2);
    }

    #[test]
    fn statistics_combine_free_regions_split_across_clones() {
        let frames = 3 * TREE_FRAMES;
        let (classing, _) = Classing::simple(2);
        let meta = MetaData::alloc(&LLFree::metadata_size(1, frames, &classing));
        let alloc = LLFree::new(1, 0, frames, Init::FreeAll, &classing, meta).unwrap();
        let tree = TreeId(1);
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        let first = alloc
            .reserve_or_steal(&guard, tree, 0, Class(0), 0)
            .unwrap()
            .0;
        drop(guard);
        drop(inner);
        let second = tree.as_frame() + FrameId(HUGE_FRAMES);
        alloc
            .get(Some(second), Request::new(0, Class(0), Some(0)))
            .unwrap();
        assert_statistics(&alloc, frames - 2, 3 * TREE_HUGE - 2, 2);
        alloc
            .put(first, Request::new(0, Class(0), None), 0)
            .unwrap();
        assert_statistics(&alloc, frames - 1, 3 * TREE_HUGE - 1, 2);
        alloc
            .put(second, Request::new(0, Class(0), None), 1)
            .unwrap();
        assert_statistics(&alloc, frames, 3 * TREE_HUGE, 3);
        let stats = alloc.stats_at(tree.as_frame(), TREE_ORDER);
        assert_eq!(stats.free_frames, TREE_FRAMES);
        assert_eq!(stats.free_huge, TREE_HUGE);
        assert_eq!(stats.free_trees, 1);
        for frame in [first, second] {
            assert_eq!(alloc.stats_at(frame, 0).free_frames, 1);
            assert_eq!(alloc.stats_at(frame, HUGE_ORDER).free_huge, 1);
        }
        let stats = alloc.tree_stats();
        assert_eq!(stats.classes[0].free_frames, TREE_FRAMES);
        assert_eq!(stats.classes[0].alloc_frames, 0);
        assert_eq!(stats.classes[1].free_frames, 2 * TREE_FRAMES);
        assert_eq!(
            unsafe { alloc.lower[tree.0].inner() }.stats().free_frames,
            0
        );
        alloc.drain();
        assert_statistics(&alloc, frames, 3 * TREE_HUGE, 3);
    }

    #[test]
    fn statistics_use_the_demoted_reservation_class() {
        let alloc = allocator(3 * TREE_FRAMES, Init::FreeAll);
        let tree = TreeId(1);
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        let first = alloc
            .reserve_or_steal(&guard, tree, 0, Class(1), 0)
            .unwrap()
            .0;
        drop(guard);
        drop(inner);
        let second = tree.as_frame() + FrameId(1);
        alloc
            .demote_local(&Request::new(0, Class(0), Some(0)), Some(second))
            .unwrap();
        assert_eq!(alloc.trees.stats_at(tree).0, Class(1));
        assert_statistics(&alloc, alloc.frames() - 2, 3 * TREE_HUGE - 1, 2);
        let stats = alloc.tree_stats();
        assert_eq!(stats.classes[0].free_frames, TREE_FRAMES - 2);
        assert_eq!(stats.classes[0].alloc_frames, 2);
        assert_eq!(stats.classes[1].free_frames, 2 * TREE_FRAMES);
        assert_eq!(stats.classes[1].alloc_frames, 0);
        alloc
            .put(first, Request::new(0, Class(0), None), 0)
            .unwrap();
        alloc
            .put(second, Request::new(0, Class(0), Some(0)), 0)
            .unwrap();
        assert_statistics(&alloc, alloc.frames(), 3 * TREE_HUGE, 3);
        alloc.drain();
        assert_statistics(&alloc, alloc.frames(), 3 * TREE_HUGE, 3);
    }

    #[test]
    fn online_does_not_publish_private_free_clone_capacity() {
        let alloc = allocator(2 * TREE_FRAMES, Init::AllocAll);
        let tree = TreeId(1);
        let frame = tree.as_frame();
        alloc
            .put(frame, Request::new(0, Class(0), None), 0)
            .unwrap();
        alloc
            .change_tree(
                TreeMatch {
                    id: Some(tree),
                    ..TreeMatch::default()
                },
                TreeChange {
                    class: None,
                    operation: Some(TreeOperation::Online),
                },
            )
            .unwrap();
        assert_eq!(alloc.trees.stats_at(tree).1, 0);
        assert_statistics(&alloc, 1, 0, 0);
        alloc.drain();
        assert_eq!(alloc.trees.stats_at(tree).1, 1);
        assert_statistics(&alloc, 1, 0, 0);
    }

    #[test]
    fn statistics_exclude_other_hosts_private_clones() {
        let frames = 3 * TREE_FRAMES;
        let (classing, _) = Classing::simple(1);
        let sizes = LLFree::metadata_size(2, frames, &classing);
        let meta = MetaData::alloc(&sizes);
        let remote = meta.remote;
        let owner = LLFree::new(2, 0, frames, Init::FreeAll, &classing, meta).unwrap();
        let peer = LLFree::new(
            2,
            1,
            frames,
            Init::None,
            &classing,
            MetaData {
                local: crate::util::aligned_buf(sizes.local),
                remote,
            },
        )
        .unwrap();
        let mut inner = owner.cxl_lock.lock();
        let guard = inner.lock();
        let frame = owner
            .reserve_or_steal(&guard, TreeId(1), 0, Class(0), 0)
            .unwrap()
            .0;
        drop(guard);
        drop(inner);
        assert_statistics(&owner, frames - 1, 3 * TREE_HUGE - 1, 2);
        assert_statistics(&peer, 2 * TREE_FRAMES, 2 * TREE_HUGE, 2);
        owner
            .put(frame, Request::new(0, Class(0), None), 0)
            .unwrap();
        assert_statistics(&owner, frames, 3 * TREE_HUGE, 3);
        assert_statistics(&peer, 2 * TREE_FRAMES, 2 * TREE_HUGE, 2);
        assert_eq!(peer.stats_at(frame, 0).free_frames, 0);
        owner.drain();
        assert_statistics(&peer, frames, 3 * TREE_HUGE, 3);
        assert_eq!(peer.stats_at(frame, 0).free_frames, 1);
    }

    #[test]
    fn online_reads_statistics_without_relocking_cxl() {
        let alloc = allocator(2 * TREE_FRAMES, Init::FreeAll);
        let matcher = TreeMatch {
            id: Some(TreeId(1)),
            ..TreeMatch::default()
        };
        for operation in [TreeOperation::Offline, TreeOperation::Online] {
            alloc
                .change_tree(
                    matcher.clone(),
                    TreeChange {
                        class: None,
                        operation: Some(operation),
                    },
                )
                .unwrap();
        }
        assert_statistics(&alloc, alloc.frames(), 2 * TREE_HUGE, 2);
    }

    #[test]
    fn global_allocations_preserve_nonzero_tree_addresses() {
        for order in [0, HUGE_ORDER, TREE_ORDER] {
            let alloc = allocator(3 * TREE_FRAMES, Init::FreeAll);
            let request = Request::new(order, Class(0), None);
            let target = TreeId(2).as_frame();
            assert_eq!(alloc.get(Some(target), request).unwrap().0, target);
            assert_eq!(alloc.get(Some(target), request), Err(Error::Memory));
            assert_eq!(alloc.stats_at(target, order).free_frames, 0);
            assert_eq!(alloc.stats_at(FrameId(0), order).free_frames, 1 << order);
            alloc.put(target, request, 0).unwrap();
            alloc.drain();
            assert_eq!(alloc.stats().free_frames, alloc.frames());
            alloc.validate();
        }
    }

    #[test]
    fn reservation_allocates_from_clone_and_replacement_restores_source() {
        let alloc = allocator(3 * TREE_FRAMES, Init::FreeAll);
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        let first = alloc
            .reserve_or_steal(&guard, TreeId(1), 0, Class(0), 0)
            .unwrap()
            .0;
        assert_eq!(first.as_tree(), TreeId(1));
        assert_eq!(
            alloc.locals.load(Class(0), 0).unwrap().free,
            TREE_FRAMES - 1
        );
        assert_eq!(unsafe { alloc.lower[1].inner() }.stats().free_frames, 0);
        let second = alloc
            .reserve_or_steal(&guard, TreeId(2), 0, Class(0), 0)
            .unwrap()
            .0;
        assert_eq!(second.as_tree(), TreeId(2));
        assert_eq!(alloc.trees.stats_at(TreeId(1)).1, TREE_FRAMES - 1);
        assert!(!alloc.trees.stats_at(TreeId(1)).2);
        assert_eq!(
            unsafe { alloc.lower[1].inner() }.stats().free_frames,
            TREE_FRAMES - 1
        );
        drop(guard);
        drop(inner);
        let next = alloc.locals.get(Class(0), 0, None, 0).unwrap();
        assert_eq!(next.as_tree(), TreeId(2));
        assert_ne!(next, second);
        alloc
            .put(next, Request::new(0, Class(0), Some(0)), 0)
            .unwrap();
        alloc.drain();
        assert_eq!(alloc.stats().free_frames, alloc.frames() - 2);
        alloc.validate();
    }

    #[test]
    fn failed_reservation_restores_fragmented_capacity() {
        let alloc = allocator(2 * TREE_FRAMES, Init::AllocAll);
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        let lower = unsafe { alloc.lower[1].borrow() };
        lower.put(FrameId(0), 0).unwrap();
        lower.put(FrameId(2), 0).unwrap();
        drop(lower);
        alloc.trees.put(&guard, TreeId(1), 2, alloc.policy);
        assert_eq!(
            alloc.reserve_or_steal(&guard, TreeId(1), 1, Class(0), 0),
            Err(Error::Memory)
        );
        assert_eq!(alloc.trees.stats_at(TreeId(1)).1, 2);
        assert!(!alloc.trees.stats_at(TreeId(1)).2);
        assert!(alloc.locals.load(Class(0), 0).is_none());
        let lower = unsafe { alloc.lower[1].borrow() };
        assert_eq!(lower.stats().free_frames, 2);
        assert!(lower.is_free(FrameId(0), 0));
        assert!(lower.is_free(FrameId(2), 0));
        assert!(lower.verify());
        drop(lower);
        assert_eq!(
            alloc.reserve_or_steal(&guard, TreeId(1), 1, Class(1), 0),
            Err(Error::Memory)
        );
        assert_eq!(alloc.trees.stats_at(TreeId(1)).1, 2);
        assert_eq!(unsafe { alloc.lower[1].inner() }.stats().free_frames, 2);
    }

    #[test]
    fn stealing_does_not_transfer_a_tree_into_a_clone() {
        let alloc = allocator(2 * TREE_FRAMES, Init::FreeAll);
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        alloc
            .reserve_or_steal(&guard, TreeId(1), 0, Class(0), 0)
            .unwrap();
        drop(guard);
        drop(inner);
        alloc.drain();
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        let (frame, class) = alloc
            .reserve_or_steal(&guard, TreeId(1), 0, Class(1), 0)
            .unwrap();
        assert_eq!(frame.as_tree(), TreeId(1));
        assert_eq!(class, Class(0));
        assert!(alloc.locals.load(Class(0), 0).is_none());
        assert!(alloc.locals.load(Class(1), 0).is_none());
        assert_eq!(alloc.trees.stats_at(TreeId(1)).1, TREE_FRAMES - 2);
        assert_eq!(
            unsafe { alloc.lower[1].inner() }.stats().free_frames,
            TREE_FRAMES - 2
        );
    }

    #[test]
    fn allocation_waits_for_exhausted_reservation_synchronization() {
        use std::sync::mpsc;
        use std::time::Duration;

        let alloc = allocator(2 * TREE_FRAMES, Init::FreeAll);
        let tree = TreeId(1);
        let mut inner = alloc.cxl_lock.lock();
        let guard = inner.lock();
        let frame = alloc
            .reserve_or_steal(&guard, tree, TREE_ORDER, Class(0), 0)
            .unwrap()
            .0;
        let lower = unsafe { alloc.lower[tree.0].borrow() };
        lower.put(FrameId(0), 0).unwrap();
        drop(lower);
        alloc.trees.put(&guard, tree, 1, alloc.policy);
        drop(guard);
        drop(inner);

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (started_tx, started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let alloc = &alloc;
            let sync = scope.spawn(move || {
                let mut inner = alloc.cxl_lock.lock();
                let guard = inner.lock();
                assert!(alloc.locals.sync(Class(0), 0, tree, || {
                    let free = alloc.trees.sync(&guard, tree, 1).unwrap();
                    let lower = unsafe { alloc.lower[tree.0].borrow() };
                    let clone = lower.get_all_copy();
                    entered_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    Some((clone, free))
                }));
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let get = scope.spawn(move || {
                started_tx.send(()).unwrap();
                result_tx
                    .send(alloc.get(Some(frame), Request::new(0, Class(0), Some(0))))
                    .unwrap();
            });
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(
                result_rx.recv_timeout(Duration::from_millis(20)),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
            release_tx.send(()).unwrap();
            assert_eq!(
                result_rx
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap()
                    .unwrap()
                    .0,
                frame
            );
            sync.join().unwrap();
            get.join().unwrap();
        });
        assert_eq!(alloc.locals.load(Class(0), 0).unwrap().free, 0);
        alloc.drain();
        assert_eq!(alloc.stats().free_frames, TREE_FRAMES);
        alloc.validate();
    }

    #[test]
    fn synchronization_transfers_cross_host_frees_with_exact_capacity() {
        let frames = 2 * TREE_FRAMES;
        let (classing, _) = Classing::simple(1);
        let sizes = LLFree::metadata_size(2, frames, &classing);
        let meta = MetaData::alloc(&sizes);
        let remote = meta.remote;
        let owner = LLFree::new(2, 0, frames, Init::FreeAll, &classing, meta).unwrap();
        let peer = LLFree::new(
            2,
            1,
            frames,
            Init::None,
            &classing,
            MetaData {
                local: crate::util::aligned_buf(sizes.local),
                remote,
            },
        )
        .unwrap();
        let mut inner = owner.cxl_lock.lock();
        let guard = inner.lock();
        let frame = owner
            .reserve_or_steal(&guard, TreeId(1), TREE_ORDER, Class(0), 0)
            .unwrap()
            .0;
        drop(guard);
        drop(inner);
        assert_eq!(owner.locals.load(Class(0), 0).unwrap().free, 0);
        peer.put(frame, Request::new(TREE_ORDER, Class(0), None), 0)
            .unwrap();
        peer.drain();
        assert_eq!(owner.trees.stats_at(TreeId(1)).1, TREE_FRAMES);
        assert_eq!(
            owner
                .get_local(TREE_ORDER, Class(0), 0, Some(frame), true)
                .unwrap(),
            frame
        );
        assert_eq!(owner.locals.load(Class(0), 0).unwrap().free, 0);
        assert_eq!(owner.trees.stats_at(TreeId(1)).1, 0);
        assert_eq!(unsafe { owner.lower[1].inner() }.stats().free_frames, 0);
        assert_eq!(
            owner.get(Some(frame), Request::new(0, Class(0), None)),
            Err(Error::Memory)
        );
        owner.drain();
        assert_eq!(owner.stats().free_frames, TREE_FRAMES);
        owner.validate();
    }
}
