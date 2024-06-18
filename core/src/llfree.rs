//! Upper allocator implementation

use core::fmt;
use core::ops::Range;

use log::{error, info, warn};

use crate::local::{Local, LocalTree};
use crate::lower::Lower;
use crate::trees::{Kind, Tree, Trees};
use crate::util::{Align, FmtFn};
use crate::{
    Alloc, Error, Flags, Init, MetaData, MetaSize, Result, HUGE_ORDER, MAX_ORDER, RETRIES,
    TREE_FRAMES,
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
    /// Reservations
    local: Local,
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
        "LLFree"
    }

    /// Initialize the allocator.
    #[cold]
    fn new(frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self> {
        info!(
            "initializing f={frames} {:?} {:?}",
            meta.trees.as_ptr_range(),
            meta.lower.as_ptr_range()
        );
        assert!(meta.valid(Self::metadata_size(frames)));

        // Create lower allocator
        let lower = Lower::new(frames, init, meta.lower)?;

        // Init tree array
        let tree_init = if init != Init::None {
            Some(|start| lower.free_in_tree(start))
        } else {
            None
        };
        let trees = Trees::new(frames, meta.trees, tree_init);

        Ok(Self {
            local: Local::default(),
            lower,
            trees,
        })
    }

    fn metadata_size(frames: usize) -> MetaSize {
        MetaSize {
            trees: Trees::metadata_size(frames),
            lower: Lower::metadata_size(frames),
        }
    }

    fn metadata(&mut self) -> MetaData<'a> {
        MetaData {
            trees: self.trees.metadata(),
            lower: self.lower.metadata(),
        }
    }

    fn get(&self, flags: Flags) -> Result<usize> {
        if flags.order() > MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }

        let mut old = LocalTree::none();
        // Try local reservation first (if enough memory)
        if self.trees.len() > 3 {
            // Retry allocation up to n times if it fails due to a concurrent update
            for _ in 0..RETRIES {
                old = self.local.preferred(flags.into()).load();
                match self.get_from_local(old, flags.into(), flags.order()) {
                    Err(Error::Retry) => {}
                    Err(Error::Memory) => break,
                    r => return r,
                }
            }

            match self.reserve_and_get(flags, old) {
                Err(Error::Memory) => {}
                r => return r,
            }

            for _ in 0..RETRIES {
                match self.steal_from_reserved(flags) {
                    Err(Error::Retry) => {}
                    Err(Error::Memory) => break,
                    r => return r,
                }
            }
        }
        // Fallback to global allocation (ignoring local reservations)
        let start = if old.present() { old.frame() } else { 0 };
        self.get_any_global(start / TREE_FRAMES, flags)
    }

    fn put(&self, frame: usize, flags: Flags) -> Result<()> {
        if frame >= self.lower.frames() {
            error!("invalid frame number");
            return Err(Error::Memory);
        }
        if flags.order() > MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // First free the frame in the lower allocator
        self.lower.put(frame, flags.order())?;

        // Increment or reserve globally
        let num_frames = 1usize << flags.order();
        self.trees.entries[frame / TREE_FRAMES]
            .fetch_update(|v| Some(v.inc(num_frames)))
            .expect("Inc failed");
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

    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }

    fn drain(&self) -> Result<()> {
        for kind in [Kind::Fixed, Kind::Movable, Kind::Huge] {
            let old = self.local.preferred(kind).swap(LocalTree::none());
            if old.present() {
                self.trees.unreserve(old.frame() / TREE_FRAMES, kind, 0);
            }
        }
        Ok(())
    }

    fn free_frames(&self) -> usize {
        self.trees.free_frames()
    }

    fn free_huge(&self) -> usize {
        self.lower.free_huge()
    }

    fn free_at(&self, frame: usize, order: usize) -> usize {
        if order == TREE_FRAMES {
            self.trees.get(frame / TREE_FRAMES).free()
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
                let free = self.lower.free_in_tree(i * TREE_FRAMES);
                assert_eq!(tree.free(), free);
            } else {
                reserved += 1;
            }
        }
        for kind in [Kind::Movable, Kind::Fixed, Kind::Huge] {
            let tree = self.local.preferred(kind).load();
            if tree.present() {
                let global = self.trees.get(tree.frame() / TREE_FRAMES);
                let free = self.lower.free_in_tree(tree.frame());
                assert_eq!(global.free(), free);
                reserved -= 1;
            }
        }
        assert!(reserved == 0);
    }
}

impl LLFree<'_> {
    fn get_from_local(&self, old: LocalTree, kind: Kind, order: usize) -> Result<usize> {
        let preferred = self.local.preferred(kind);
        let i = old.frame() / TREE_FRAMES;

        if let Ok(_) = self.trees.entries[i].fetch_update(|v| v.dec(kind, 1 << order)) {
            match self.lower.get(old.frame(), order) {
                Ok((frame, _)) => {
                    if old.frame() / 64 != frame / 64 {
                        let _ = preferred.fetch_update(|v| v.set_start(frame, false));
                    }
                    Ok(frame)
                }
                Err(e) => {
                    self.trees.entries[i]
                        .fetch_update(|v| Some(v.inc(1 << order)))
                        .expect("Undo failed");
                    Err(e)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(&self, flags: Flags, old: LocalTree) -> Result<usize> {
        // Try reserve new tree
        let start = if old.present() {
            old.frame() / TREE_FRAMES
        } else {
            0
        };
        const CL: usize = align_of::<Align>() / size_of::<Tree>();
        let near = (self.trees.len() / 4).clamp(CL / 4, CL * 2);

        let reserve = |i: usize, range: Range<usize>| {
            let range = (1 << flags.order()).max(range.start)..range.end;
            if let Ok(_) = self.trees.entries[i]
                .fetch_update(|v| v.reserve(range.clone(), flags.into(), 1 << flags.order()))
            {
                self.try_get_reserve(flags.into(), flags.order(), i)
            } else {
                Err(Error::Memory)
            }
        };

        // Over half filled trees
        let range = TREE_FRAMES / 16..TREE_FRAMES / 2;
        match self
            .trees
            .search(start, 1, near, |i| reserve(i, range.clone()))
        {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Partially filled
        let range = TREE_FRAMES / 64..TREE_FRAMES - TREE_FRAMES / 16;
        match self
            .trees
            .search(start, 1, near, |i| reserve(i, range.clone()))
        {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Not free
        let range = 0..TREE_FRAMES;
        match self
            .trees
            .search(start, 1, near, |i| reserve(i, range.clone()))
        {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Any
        let range = 0..usize::MAX;
        self.trees
            .search(start, 1, near, |i| reserve(i, range.clone()))
    }

    /// Steal a tree from another core
    fn steal_from_reserved(&self, flags: Flags) -> Result<usize> {
        let kind = Kind::from(flags);
        for o_kind in 0..Kind::LEN {
            let t_kind = Kind::from((kind as usize + o_kind) % Kind::LEN);

            if t_kind.accepts(kind) {
                let old = self.local.preferred(t_kind).load();
                // Less strict kind, just allocate
                match self.get_from_local(old, t_kind, flags.order()) {
                    Err(Error::Memory) => {}
                    r => return r,
                }
            } else {
                // More strict kind, steal and convert tree
                match self.steal_tree(t_kind, flags.order()) {
                    Err(Error::Memory) => {}
                    r => return r,
                }
            }
        }
        Err(Error::Memory)
    }

    fn steal_tree(&self, kind: Kind, order: usize) -> Result<usize> {
        let preferred = self.local.preferred(kind);
        if let Ok(stolen) = preferred.fetch_update(|v| v.steal()) {
            assert!(stolen.present());

            let i = stolen.frame() / TREE_FRAMES;
            if self.trees.entries[i]
                .fetch_update(|v| v.dec_force(kind, 1 << order))
                .is_ok()
            {
                self.try_get_reserve(kind, order, stolen.frame() / TREE_FRAMES)
            } else {
                self.trees.unreserve(i, kind, 0);
                Err(Error::Memory)
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn get_any_global(&self, start_idx: usize, flags: Flags) -> Result<usize> {
        self.trees.search(start_idx, 0, self.trees.len(), |i| {
            let old = self.trees.entries[i]
                .fetch_update(|v| v.dec_force(flags.into(), 1 << flags.order()))
                .map_err(|_| Error::Memory)?;

            match self.lower.get(i * TREE_FRAMES, flags.order()) {
                Ok((frame, _)) => Ok(frame),
                Err(e) => {
                    let exp = old.dec_force(flags.into(), 1 << flags.order()).unwrap();
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

    fn try_get_reserve(&self, kind: Kind, order: usize, i: usize) -> Result<usize> {
        match self.lower.get(i * TREE_FRAMES, order) {
            Ok((frame, _)) => {
                let old = self.local.preferred(kind).swap(LocalTree::with(frame));
                if old.present() {
                    self.trees.unreserve(old.frame() / TREE_FRAMES, kind, 0);
                }
                Ok(frame)
            }
            Err(e) => {
                self.trees.unreserve(i, kind, 1 << order);
                Err(e)
            }
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
            .field("locals", &self.local)
            .finish()?;
        Ok(())
    }
}
