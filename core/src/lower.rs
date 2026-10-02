//! Lower allocator implementations

use core::fmt;
use core::sync::atomic::AtomicU16;

use bitfield_struct::bitfield;
use log::{error, info, warn};

use crate::atomic::{Atom, Atomic, AtomicSlice};
use crate::bitfield::{Bitfield, RowId};
use crate::cache::Align;
use crate::trees::TreeId;
use crate::util::{align_down, spin_wait};
use crate::{
    CacheLine, Error, FrameId, HUGE_FRAMES, HUGE_ORDER, Init, RETRIES, Result, Stats, TREE_FRAMES,
    TREE_HUGE, TREE_ORDER,
};

const _: () = assert!(Bitfield::LEN == HUGE_FRAMES);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HugeId(pub usize);
impl HugeId {
    pub fn as_frame(self) -> FrameId {
        FrameId(self.0 * HUGE_FRAMES)
    }
    pub fn as_tree(self) -> TreeId {
        self.as_frame().as_tree()
    }
    pub fn as_row(self) -> RowId {
        self.as_frame().as_row()
    }
    pub fn child_idx(self) -> usize {
        self.0 % TREE_HUGE
    }
}
impl core::ops::Add<Self> for HugeId {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl fmt::Display for HugeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hx{:x}", self.0)
    }
}

impl fmt::Debug for HugeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Lower-level cache-aligned tree structure
#[repr(C)] // <- enforce field order
pub struct LowerTree {
    children: [Atom<HugeEntry>; TREE_HUGE],
    pub bitfield: Align<[Bitfield; TREE_HUGE]>,
}
impl Default for LowerTree {
    fn default() -> Self {
        Self {
            children: core::array::from_fn(|_| Atom::default()),
            bitfield: Align(core::array::from_fn(|_| Bitfield::default())),
        }
    }
}
impl fmt::Debug for LowerTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LowerTree")
            .field("children", &self.children)
            .finish_non_exhaustive()
    }
}

impl LowerTree {
    /// Size of one tree's metadata in cache lines, including partial trees.
    pub const fn metadata_size() -> usize {
        size_of::<Self>().div_ceil(CacheLine::SIZE)
    }

    /// Initialize a tree managing `frames` tree-relative frames.
    ///
    /// `Recover` and `None` require metadata previously initialized for the same frame count.
    /// The caller must exclusively own the metadata while initializing it.
    pub fn init(&self, frames: usize, init: Init) {
        debug_assert!(frames <= TREE_FRAMES);
        match init {
            Init::FreeAll | Init::AllocAll => {
                for (i, (entry, bitfield)) in
                    self.children.iter().zip(self.bitfield.iter()).enumerate()
                {
                    let free = frames.saturating_sub(i * HUGE_FRAMES).min(HUGE_FRAMES);
                    if init == Init::FreeAll {
                        entry.store(HugeEntry::new_with(free));
                        bitfield.fill(true);
                        if free > 0 {
                            bitfield.set(FrameId(0)..FrameId(free), false);
                        }
                    } else {
                        let included = free == HUGE_FRAMES;
                        entry.store(if included {
                            HugeEntry::new_huge()
                        } else {
                            HugeEntry::new_with(0)
                        });
                        bitfield.fill(!included);
                    }
                }
            }
            Init::Recover => {
                self.recover();
            }
            Init::None => {}
        }
    }

    /// Try allocating a new `frame` in the [`TREE_FRAMES`] sized chunk at `start`.
    ///
    /// Returns the allocated frame and whether a new huge frame was fragmented.
    pub fn get(&self, start: RowId, order: usize, frame: Option<FrameId>) -> Result<FrameId> {
        assert!(order <= TREE_ORDER);
        assert!(start.as_frame().0 < TREE_FRAMES);

        if let Some(frame) = frame {
            return self.get_at(frame, order).map(|()| frame);
        }

        let child_off = start.as_huge().child_idx();

        if let Some(h_order) = order.checked_sub(HUGE_ORDER) {
            let h_num = 1 << h_order;
            let child_off = align_down(child_off, h_num);
            for i in (0..TREE_HUGE).step_by(h_num) {
                let i = (child_off + i) % TREE_HUGE;
                if self.children[i..i + h_num]
                    .compare_exchange_all(HugeEntry::new_with(Bitfield::LEN), HugeEntry::new_huge())
                    .is_ok()
                {
                    return Ok(FrameId(HugeId(i).as_frame().0));
                }
            }
        } else {
            for j in 0..TREE_HUGE {
                let i = (child_off + j) % TREE_HUGE;
                if let Ok(_) = self.children[i].try_update(|v| v.dec(1 << order)) {
                    // start with the bitfield row from the last allocation
                    if let Ok(offset) = self.bitfield[i].set_first_zeros(start.row_idx(), order) {
                        return Ok(HugeId(i).as_frame() + offset);
                    }
                    self.children[i]
                        .try_update(|v| v.inc(1 << order))
                        .expect("Undo failed");
                }
            }
        }
        Err(Error::Memory)
    }

    /// Try allocating a specific `frame`.
    fn get_at(&self, frame: FrameId, order: usize) -> Result<()> {
        debug_assert!(order <= TREE_ORDER);
        debug_assert!(frame.is_aligned(order));
        assert!(frame.0 + (1 << order) <= TREE_FRAMES);

        let i = frame.as_huge().child_idx();

        if let Some(h_order) = order.checked_sub(HUGE_ORDER) {
            let children = &self.children[i..i + (1 << h_order)];
            match children
                .compare_exchange_all(HugeEntry::new_with(Bitfield::LEN), HugeEntry::new_huge())
            {
                Ok(_) => Ok(()),
                Err(_) => Err(Error::Memory),
            }
        } else {
            if let Ok(_) = self.children[i].try_update(|v| v.dec(1 << order)) {
                if let Ok(()) = self.bitfield[i].toggle(frame, order, false) {
                    return Ok(());
                }
                // Undo decrement
                self.children[i].try_update(|v| v.inc(1 << order)).unwrap();
            }
            Err(Error::Memory)
        }
    }

    /// Free single frame, returning whether a whole huge page has become free.
    pub fn put(&self, frame: FrameId, order: usize) -> Result<()> {
        assert!(frame.is_aligned(order));
        assert!(frame.0 + (1 << order) <= TREE_FRAMES);

        let i = frame.as_huge().child_idx();

        if let Some(h_order) = order.checked_sub(HUGE_ORDER) {
            let children = &self.children[i..i + (1 << h_order)];
            match children
                .compare_exchange_all(HugeEntry::new_huge(), HugeEntry::new_with(Bitfield::LEN))
            {
                Ok(_) => Ok(()),
                Err(_) => Err(Error::Memory),
            }
        } else {
            let old = self.children[i].load();
            if old.huge() {
                self.partial_put_huge(old, frame, order)
            } else if old.free() <= Bitfield::LEN - (1 << order) {
                self.put_small(frame, order)
            } else {
                error!("Addr {frame:?} o={order} {old:?}");
                Err(Error::Memory)
            }
        }
    }
    fn put_small(&self, frame: FrameId, order: usize) -> Result<()> {
        debug_assert!(order < HUGE_ORDER);
        debug_assert!(frame.is_aligned(order));
        debug_assert!(frame.0 + (1 << order) <= TREE_FRAMES);

        let i = frame.as_huge().child_idx();
        let bitfield = &self.bitfield[i];
        if bitfield.toggle(frame, order, true).is_err() {
            error!(
                "L1 put failed o={order} i={} p={frame:?}",
                frame.0 % Bitfield::LEN
            );
            return Err(Error::Memory);
        }

        match self.children[i].try_update(|v| v.inc(1 << order)) {
            Ok(_) => Ok(()),
            Err(entry) => panic!("Inc failed i{i} p={frame:?} {entry:?}"),
        }
    }

    fn partial_put_huge(&self, old: HugeEntry, frame: FrameId, order: usize) -> Result<()> {
        info!("partial free of huge frame {frame:?} o={order}");
        let i = frame.as_huge().child_idx();
        self.split_huge(old, i);
        self.put_small(frame, order)
    }

    fn split_huge(&self, old: HugeEntry, i: usize) {
        let bitfield = &self.bitfield[i];
        // Try filling the whole bitfield
        if bitfield.toggle(FrameId(0), Bitfield::ORDER, false).is_ok() {
            self.children[i]
                .compare_exchange(old, HugeEntry::new())
                .expect("Failed partial clear");
        }
        // Wait for parallel partial_put_huge to finish
        else if !spin_wait(RETRIES, || !self.children[i].load().huge()) {
            panic!("Exceeding retries");
        }
    }

    /// Merges all free frames from other into self.
    pub fn merge_frees(&self, other: Self) {
        for (i, source) in other.children.iter().enumerate() {
            if source.load().free() == 0 {
                continue;
            }
            let old = self.children[i].load();
            if old.huge() {
                self.split_huge(old, i);
            }
            let freed = self.bitfield[i].merge_frees(&other.bitfield[i]);
            if freed != 0 {
                self.children[i]
                    .try_update(|v| v.inc(freed))
                    .expect("Failed to increment merged free frames");
            }
        }
    }

    /// Returns a copy of the tree and reserve it entirely.
    pub fn get_all_copy(&self) -> Self {
        let mut result = Self::default();
        for i in 0..TREE_HUGE {
            let huge =
                self.children[i].update(|v| if v.huge() { v } else { HugeEntry::new_with(0) });
            result.children[i] = Atom::new(huge);
            if !huge.huge() {
                result.bitfield[i] = self.bitfield[i].fill_and_copy(true);
            }
        }
        result
    }

    /// Returns if the frame is free. This might be racy!
    pub fn is_free(&self, frame: FrameId, order: usize) -> bool {
        assert!(frame.is_aligned(order));
        assert!(frame.0 + (1 << order) <= TREE_FRAMES);
        assert!(order <= TREE_ORDER, "Order {order} is not supported");

        let i = frame.as_huge().child_idx();

        if let Some(h_order) = order.checked_sub(Bitfield::ORDER) {
            self.children[i..i + (1 << h_order)]
                .iter()
                .all(|e| e.load().free() == Bitfield::LEN)
        } else {
            let child = self.children[i].load();
            if child.free() < (1 << order) {
                false
            } else if child.free() == Bitfield::LEN {
                true
            } else {
                self.bitfield[i].is_zero(frame, order)
            }
        }
    }

    /// Returns statistics.
    pub fn stats(&self) -> Stats {
        let mut stats = Stats::default();
        for entry in self.children.iter() {
            let f = entry.load().free();
            stats.free_frames += f;
            stats.free_huge += (f == HUGE_FRAMES) as usize;
        }
        stats.free_trees = (stats.free_frames == TREE_FRAMES) as _;
        stats
    }

    /// Returns statistics at a specific frame, huge frame, or tree.
    pub fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        const TREE_ORDER: usize = TREE_FRAMES.ilog2() as usize;
        let i = frame.as_huge().child_idx();
        match order {
            0 => Stats {
                free_frames: (self.children[i].load().free() > 0
                    && self.bitfield[i].is_zero(frame, 0)) as usize,
                free_huge: 0,
                free_trees: 0,
            },
            HUGE_ORDER => {
                let free = self.children[i].load().free();
                Stats {
                    free_frames: free,
                    free_huge: (free == HUGE_FRAMES) as usize,
                    free_trees: 0,
                }
            }
            TREE_ORDER => self.stats(),
            _ => Stats::default(),
        }
    }
    /// Resolves all invalid huge entry counters, returning if all entries were valid.
    pub fn recover(&self) -> bool {
        let mut matching = true;
        for (huge, bitfield) in self.children.iter().zip(self.bitfield.iter()) {
            let entry = huge.load();
            let zeros = bitfield.count_zeros();
            if entry.huge() {
                // Check that underlying bitfield is empty
                if zeros != Bitfield::LEN {
                    warn!("Recover huge entry: h != {zeros}");
                    bitfield.fill(false);
                    matching = false;
                }
            } else {
                // Check the bitfield has the same number of zero bits
                if entry.free() != zeros {
                    warn!("Recover huge entry: {} != {zeros}", entry.free());
                    huge.store(HugeEntry::new_with(zeros));
                    matching = false;
                }
            }
        }
        matching
    }
    /// Verifies the internal state of the lower tree.
    pub fn verify(&self) -> bool {
        for (i, (child, bitfield)) in self.children.iter().zip(self.bitfield.iter()).enumerate() {
            let (h, bf) = (child.load(), bitfield.count_zeros());
            if bf != if h.huge() { Bitfield::LEN } else { h.free() } {
                warn!("Verify {i}: {h:?} != {bf}");
                return false;
            }
        }
        true
    }

    #[cfg(any(test, feature = "std"))]
    #[allow(dead_code)]
    pub fn dump(&self) {
        use std::fmt::Write;

        let mut out = std::string::String::new();
        writeln!(out, "Dumping lower tree").unwrap();
        let entries = &self.children;
        for (i, entry) in entries.iter().enumerate() {
            let entry = entry.load();
            let indent = 4;
            let bitfield = &self.bitfield[i];
            writeln!(out, "{:indent$}l2 i={i}: {entry:?}\t{bitfield:?}", "").unwrap();
            if !entry.huge() && bitfield.count_zeros() != entry.free() {
                error!("Invalid free counter i={i}");
            }
        }
        warn!("{out}");
    }
}

/// Manages huge frame, that can be allocated as base frames.
#[bitfield(u16)]
#[derive(PartialEq, Eq)]
struct HugeEntry {
    /// Number of free 4K frames or [`u16::MAX`] for a huge frame.
    count: u16,
}
impl Atomic for HugeEntry {
    type I = AtomicU16;
}
impl HugeEntry {
    /// Creates an entry marked as allocated huge frame.
    fn new_huge() -> Self {
        Self::new().with_count(u16::MAX)
    }
    /// Creates a new entry with the given free counter.
    fn new_with(free: usize) -> Self {
        Self::new().with_count(free as _)
    }
    /// Returns wether this entry is allocated as huge frame.
    fn huge(self) -> bool {
        self.count() == u16::MAX
    }
    /// Returns the free frames counter
    fn free(self) -> usize {
        if self.huge() { 0 } else { self.count() as _ }
    }
    /// Decrement the free frames counter.
    fn dec(self, num_frames: usize) -> Option<Self> {
        if !self.huge() && self.free() >= num_frames {
            Some(Self::new_with(self.free() - num_frames))
        } else {
            None
        }
    }
    /// Increments the free frames counter.
    fn inc(self, num_frames: usize) -> Option<Self> {
        if !self.huge() && self.free() <= Bitfield::LEN - num_frames {
            Some(Self::new_with(self.free() + num_frames))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use core::array::from_fn;
    use std::sync::Barrier;
    use std::vec::Vec;

    use super::Bitfield;
    use super::{HugeEntry, LowerTree};
    use crate::atomic::Atom;
    use crate::cache::Align;
    use crate::util::{WyRand, logging, parallel};
    use crate::{
        Error, FrameId, HUGE_FRAMES, HUGE_ORDER, HugeId, Init, TREE_FRAMES, TREE_HUGE, TREE_ORDER,
    };

    fn lower_tree(frames: usize, init: Init) -> LowerTree {
        let tree = LowerTree {
            children: from_fn(|_| Atom::new(HugeEntry::new())),
            bitfield: Align(from_fn(|_| Bitfield::default())),
        };
        tree.init(frames, init);
        tree
    }

    #[test]
    fn default_alloc_all_accepts_frees() {
        for order in [0, HUGE_ORDER] {
            let tree = LowerTree::default();
            assert!(tree.children.iter().all(|entry| entry.load().count() == 0));
            assert!(
                tree.bitfield
                    .iter()
                    .all(|bitfield| bitfield.count_zeros() == HUGE_FRAMES)
            );

            tree.init(TREE_FRAMES, Init::AllocAll);
            assert_eq!(tree.stats().free_frames, 0);
            assert!(tree.verify());

            tree.put(FrameId(0), order).unwrap();
            assert!(tree.is_free(FrameId(0), order));
            assert_eq!(tree.stats().free_frames, 1 << order);
            assert_eq!(tree.stats().free_huge, (order == HUGE_ORDER) as usize);
            if order == 0 {
                assert!(!tree.is_free(FrameId(1), 0));
            }
            assert!(tree.verify());
        }
    }

    #[test]
    fn merge_frees() {
        let tree = lower_tree(TREE_FRAMES, Init::AllocAll);
        let other = lower_tree(TREE_FRAMES, Init::AllocAll);
        tree.put(FrameId(0), 0).unwrap();
        tree.put(FrameId(HUGE_FRAMES), 0).unwrap();
        other.put(FrameId(0), HUGE_ORDER).unwrap();
        other.put(FrameId(2 * HUGE_FRAMES), 0).unwrap();

        tree.merge_frees(other);

        assert!(tree.is_free(FrameId(0), HUGE_ORDER));
        assert!(tree.is_free(FrameId(HUGE_FRAMES), 0));
        assert!(tree.is_free(FrameId(2 * HUGE_FRAMES), 0));
        assert!(!tree.is_free(FrameId(2 * HUGE_FRAMES + 1), 0));
        assert_eq!(tree.stats().free_frames, HUGE_FRAMES + 2);
        assert_eq!(tree.stats().free_huge, 1);
        assert!(tree.verify());
    }

    #[test]
    fn merge_frees_partial_tree() {
        let tree = lower_tree(3, Init::AllocAll);
        tree.merge_frees(lower_tree(3, Init::FreeAll));
        tree.merge_frees(lower_tree(3, Init::FreeAll));

        assert_eq!(tree.stats().free_frames, 3);
        for i in 0..TREE_FRAMES {
            assert_eq!(tree.is_free(FrameId(i), 0), i < 3);
        }
        assert!(tree.verify());
    }

    #[test]
    fn tree_initialization() {
        let frames = [
            0,
            3,
            HUGE_FRAMES - 1,
            HUGE_FRAMES,
            TREE_FRAMES - 1,
            TREE_FRAMES,
        ];
        let init = [Init::FreeAll, Init::AllocAll];
        for (frames, init) in frames
            .iter()
            .flat_map(|&f| init.iter().map(move |&i| (f, i)))
        {
            let tree = lower_tree(frames, init);
            assert_eq!(
                tree.stats().free_frames,
                if init == Init::FreeAll { frames } else { 0 }
            );
            assert!(tree.verify());

            for i in frames..TREE_FRAMES {
                assert!(!tree.is_free(FrameId(i), 0));
            }
            if frames > 0 {
                if init == Init::FreeAll {
                    tree.get(FrameId(0).as_row(), 0, Some(FrameId(0))).unwrap();
                } else {
                    tree.put(FrameId(0), 0).unwrap();
                }
            }
            let stats = tree.stats();
            tree.init(frames, Init::None);
            assert_eq!(tree.stats().free_frames, stats.free_frames);
            let free = tree.bitfield[0].count_zeros();
            tree.children[0].store(HugeEntry::new_with((free + 1) % (HUGE_FRAMES + 1)));
            assert!(!tree.verify());
            tree.init(frames, Init::Recover);
            assert!(tree.verify());
            assert_eq!(tree.stats().free_frames, stats.free_frames);
            assert!(tree.recover());
            tree.dump();
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "frames <= TREE_FRAMES")]
    fn invalid_tree_frames() {
        lower_tree(TREE_FRAMES + 1, Init::FreeAll);
    }

    #[test]
    fn recover_huge_bitfield() {
        let tree = lower_tree(TREE_FRAMES, Init::AllocAll);
        tree.bitfield[0].fill(true);
        assert!(!tree.recover());
        assert!(tree.verify());
        assert_eq!(tree.stats().free_frames, 0);
        assert!(tree.recover());
    }

    #[test]
    fn colocated_multiple_trees() {
        let trees: Vec<LowerTree> = (0..3)
            .map(|i| lower_tree(if i == 2 { 3 } else { TREE_FRAMES }, Init::FreeAll))
            .collect();
        for order in [0, HUGE_ORDER, TREE_ORDER] {
            let frame = FrameId(0);
            assert_eq!(trees[1].get(frame.as_row(), order, Some(frame)), Ok(frame));
            assert!(!trees[1].is_free(frame, order));
            assert_eq!(trees[0].stats().free_frames, TREE_FRAMES);
            assert_eq!(trees[2].stats().free_frames, 3);
            trees[1].put(frame, order).unwrap();
            assert!(trees[1].is_free(frame, order));
            assert_eq!(trees[1].get(frame.as_row(), order, None), Ok(frame));
            trees[1].put(frame, order).unwrap();
        }
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            2 * TREE_FRAMES + 3
        );
        assert_eq!(trees.iter().map(|t| t.stats().free_trees).sum::<usize>(), 2);
        assert_eq!(trees[2].stats_at(FrameId(0), TREE_ORDER).free_frames, 3);
        for _ in 0..3 {
            let frame = trees[2].get(FrameId(0).as_row(), 0, None).unwrap();
            assert!(frame.0 < 3);
        }
        assert_eq!(
            trees[2].get(FrameId(0).as_row(), 0, None),
            Err(Error::Memory)
        );
        for tree in &trees {
            assert!(tree.recover());
        }
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            2 * TREE_FRAMES
        );
    }

    #[test]
    fn alloc_normal() {
        logging();

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);
        lower.get(FrameId(0).as_row(), 0, None).unwrap();

        parallel(0..2, |_| {
            let frame = lower.get(FrameId(0).as_row(), 0, None).unwrap().0;
            assert!(frame < TREE_FRAMES);
        });

        assert_eq!(lower.children[0].load().free(), Bitfield::LEN - 3);
        assert_eq!(
            lower.stats_at(HugeId(0).as_frame(), HUGE_ORDER).free_frames,
            Bitfield::LEN - 3
        );
    }

    #[test]
    fn alloc_first() {
        logging();

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);

        parallel(0..2, |_| {
            lower.get(FrameId(0).as_row(), 0, None).unwrap();
        });

        let entry2 = lower.children[0].load();
        assert_eq!(entry2.free(), Bitfield::LEN - 2);
        assert_eq!(
            lower.stats_at(HugeId(0).as_frame(), HUGE_ORDER).free_frames,
            Bitfield::LEN - 2
        );
    }

    #[test]
    #[cfg(not(feature = "tree_huge_1"))]
    fn alloc_last() {
        logging();

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);

        for _ in 0..Bitfield::LEN - 1 {
            lower.get(FrameId(0).as_row(), 0, None).unwrap();
        }

        parallel(0..2, |_| {
            lower.get(FrameId(0).as_row(), 0, None).unwrap();
        });

        let table = &lower.children;
        assert_eq!(table[0].load().free(), 0);
        assert_eq!(table[1].load().free(), Bitfield::LEN - 1);
        assert_eq!(
            lower.stats_at(HugeId(1).as_frame(), HUGE_ORDER).free_frames,
            Bitfield::LEN - 1
        );
    }

    #[test]
    fn free_normal() {
        logging();

        let mut frames = [0; 2];

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);

        frames[0] = lower.get(FrameId(0).as_row(), 0, None).unwrap().0;
        frames[1] = lower.get(FrameId(0).as_row(), 0, None).unwrap().0;

        parallel(0..2, |t| {
            lower.put(FrameId(frames[t]), 0).unwrap();
        });

        assert_eq!(lower.children[0].load().free(), Bitfield::LEN);
    }

    #[test]
    fn free_last() {
        logging();

        let mut frames = [0; Bitfield::LEN];

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);

        for frame in &mut frames {
            *frame = lower.get(FrameId(0).as_row(), 0, None).unwrap().0;
        }

        parallel(0..2, |t| {
            lower.put(FrameId(frames[t]), 0).unwrap();
        });

        let table = &lower.children;
        assert_eq!(table[0].load().free(), 2);
        assert_eq!(
            lower.stats_at(HugeId(0).as_frame(), HUGE_ORDER).free_frames,
            2
        );
    }

    #[test]
    #[cfg(not(feature = "tree_huge_1"))]
    fn realloc_last() {
        logging();

        let mut frames = [0; Bitfield::LEN];

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);

        for frame in &mut frames[..Bitfield::LEN - 1] {
            *frame = lower.get(FrameId(0).as_row(), 0, None).unwrap().0;
        }

        std::thread::scope(|s| {
            s.spawn(|| {
                lower.get(FrameId(0).as_row(), 0, None).unwrap();
            });

            lower.put(FrameId(frames[0]), 0).unwrap();
        });

        let table = &lower.children;
        if table[0].load().free() == 1 {
            assert_eq!(
                lower.stats_at(HugeId(0).as_frame(), HUGE_ORDER).free_frames,
                1
            );
        } else {
            // Table entry skipped
            assert_eq!(table[0].load().free(), 2);
            assert_eq!(
                lower.stats_at(HugeId(0).as_frame(), HUGE_ORDER).free_frames,
                2
            );
            assert_eq!(table[1].load().free(), Bitfield::LEN - 1);
            assert_eq!(
                lower.stats_at(HugeId(1).as_frame(), HUGE_ORDER).free_frames,
                Bitfield::LEN - 1
            );
        }
    }

    #[test]
    fn alloc_normal_large() {
        logging();

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);
        lower.get(FrameId(0).as_row(), 0, None).unwrap();

        parallel(0..2, |t| {
            let order = t + 1; // order 1 and 2
            let frame = lower.get(FrameId(0).as_row(), order, None).unwrap().0;
            assert!(frame < TREE_FRAMES);
        });

        let allocated = 1 + 2 + 4;
        assert_eq!(lower.children[0].load().free(), Bitfield::LEN - allocated);
        assert_eq!(
            lower.stats_at(HugeId(0).as_frame(), HUGE_ORDER).free_frames,
            Bitfield::LEN - allocated
        );
    }

    #[test]
    fn free_normal_large() {
        logging();

        let mut frames = [0; 2];

        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);

        frames[0] = lower.get(FrameId(0).as_row(), 1, None).unwrap().0;
        frames[1] = lower.get(FrameId(0).as_row(), 2, None).unwrap().0;

        assert_eq!(lower.children[0].load().free(), Bitfield::LEN - 2 - 4);

        parallel(0..2, |t| {
            lower.put(FrameId(frames[t]), t + 1).unwrap();
        });

        assert_eq!(lower.children[0].load().free(), Bitfield::LEN);
    }

    #[test]
    #[cfg(not(feature = "tree_huge_1"))]
    fn different_orders() {
        logging();
        const MAX_ORDER: usize = HUGE_ORDER + 1;
        const FRAMES: usize = (MAX_ORDER + 2) << MAX_ORDER;
        let trees: Vec<LowerTree> = (0..FRAMES.div_ceil(TREE_FRAMES))
            .map(|i| lower_tree((FRAMES - i * TREE_FRAMES).min(TREE_FRAMES), Init::FreeAll))
            .collect();
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            FRAMES
        );
        let mut rng = WyRand::new(42);
        let mut num_frames = 0;
        let mut frames = Vec::new();
        for order in 0..=MAX_ORDER {
            for _ in 0..1usize << (MAX_ORDER - order) {
                frames.push((order, 0, FrameId(0)));
                num_frames += 1 << order;
            }
        }
        rng.shuffle(&mut frames);
        assert!(FRAMES >= num_frames);
        let mut tree_idx = 0;
        'outer: for (order, tree, frame) in &mut frames {
            for offset in 0..trees.len() {
                let i = (offset + tree_idx) % trees.len();
                match trees[i].get(FrameId(0).as_row(), *order, None) {
                    Ok(free) => {
                        *frame = free;
                        *tree = i;
                        tree_idx = i;
                        continue 'outer;
                    }
                    Err(Error::Memory) => {}
                    Err(e) => panic!("{e:?}"),
                }
            }
            panic!("Fragmented!");
        }
        assert_eq!(
            FRAMES - trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            num_frames
        );
        for (order, tree, frame) in &frames {
            trees[*tree].put(*frame, *order).unwrap();
        }
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            FRAMES
        );
    }

    #[test]
    fn init_reserved_max_order() {
        logging();
        const FRAMES: usize = 24 * TREE_FRAMES;
        let trees: Vec<LowerTree> = (0..24)
            .map(|_| lower_tree(TREE_FRAMES, Init::AllocAll))
            .collect();
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            0
        );
        for tree in &trees {
            tree.put(FrameId(0), TREE_ORDER).unwrap();
        }
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            FRAMES
        );
    }

    #[test]
    fn partial_put_huge() {
        logging();

        let lower = lower_tree(TREE_FRAMES - 1, Init::AllocAll);

        assert_eq!(lower.stats().free_frames, 0);

        lower.put(FrameId(0), 0).unwrap();

        assert_eq!(lower.stats().free_frames, 1);
    }

    #[test]
    fn alloc_large_orders() {
        logging();
        const FRAMES: usize = 4 * TREE_FRAMES;
        let trees: Vec<LowerTree> = (0..4)
            .map(|_| lower_tree(TREE_FRAMES, Init::FreeAll))
            .collect();
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            FRAMES
        );
        let orders = [HUGE_ORDER, HUGE_ORDER + 1, HUGE_ORDER + 2, TREE_ORDER];
        let mut allocations = Vec::new();
        let mut tree_id = 0;
        let mut allocated = 0;
        for order in orders {
            if order >= TREE_ORDER {
                continue;
            }
            let frame = match trees[tree_id].get(FrameId(0).as_row(), order, None) {
                Ok(frame) => frame,
                Err(Error::Memory) => {
                    tree_id += 1;
                    assert!(tree_id < trees.len());
                    trees[tree_id]
                        .get(FrameId(0).as_row(), order, None)
                        .unwrap()
                }
                Err(e) => panic!("Unexpected error: {e:?}"),
            };
            allocations.push((tree_id, frame, order));
            assert!(frame.is_aligned(order));
            assert!(frame.0 + (1 << order) <= TREE_FRAMES);
            allocated += 1 << order;
            assert_eq!(
                trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
                FRAMES - allocated
            );
        }
        for (tree, frame, order) in allocations {
            trees[tree].put(frame, order).unwrap();
            allocated -= 1 << order;
            assert_eq!(
                trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
                FRAMES - allocated
            );
        }
        assert_eq!(
            trees.iter().map(|t| t.stats().free_frames).sum::<usize>(),
            FRAMES
        );
        assert_eq!(
            trees.iter().map(|t| t.stats().free_huge).sum::<usize>(),
            FRAMES / HUGE_FRAMES
        );
    }

    #[test]
    #[ignore]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 6;
        const FRAMES: usize = TREE_FRAMES;

        for _ in 0..8 {
            let lower = lower_tree(FRAMES, Init::FreeAll);
            assert_eq!(lower.stats().free_frames, FRAMES);

            let barrier = Barrier::new(THREADS);
            parallel(0..THREADS, |_| {
                barrier.wait();

                let mut frames = [FrameId(0); 4];
                for p in &mut frames {
                    *p = lower.get(FrameId(0).as_row(), 0, None).unwrap();
                }
                frames.reverse();
                for p in frames {
                    lower.put(p, 0).unwrap();
                }
            });

            assert_eq!(lower.stats().free_frames, FRAMES);
        }
    }

    #[test]
    #[ignore]
    fn rand_realloc_last() {
        logging();

        const THREADS: usize = 6;
        const FRAMES: usize = TREE_FRAMES;
        let mut frames = [0; HUGE_FRAMES];

        for _ in 0..8 {
            let lower = lower_tree(FRAMES, Init::FreeAll);
            assert_eq!(lower.stats().free_frames, FRAMES);

            for frame in &mut frames[..HUGE_FRAMES - 3] {
                *frame = lower.get(FrameId(0).as_row(), 0, None).unwrap().0;
            }

            let barrier = Barrier::new(THREADS);
            parallel(0..THREADS, |t| {
                barrier.wait();

                if t < THREADS / 2 {
                    lower.put(FrameId(frames[t]), 0).unwrap();
                } else {
                    lower.get(FrameId(0).as_row(), 0, None).unwrap();
                }
            });

            assert_eq!(TREE_FRAMES - lower.stats().free_frames, HUGE_FRAMES - 3);
        }
    }

    #[test]
    fn alloc_stress_huge() {
        logging();

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        const ITER: usize = 50;
        const THREADS: usize = 4;
        let lower = lower_tree(TREE_FRAMES, Init::FreeAll);
        let barrier = Barrier::new(THREADS);

        parallel(0..THREADS, |t| {
            let mut frames = Vec::with_capacity(TREE_FRAMES);

            barrier.wait();

            let mut rng = WyRand::new(seed + t as u64);

            for _ in 0..ITER {
                let target = rng.range(0..(2 * TREE_FRAMES / THREADS) as _) as usize;

                while frames.len() != target {
                    if target < frames.len() {
                        lower.put(frames.pop().unwrap(), 0).unwrap();
                    } else {
                        match lower.get(FrameId(0).as_row(), 0, None) {
                            Ok(frame) => {
                                frames.push(frame);
                            }
                            Err(Error::Memory) => break,
                            Err(e) => panic!("{e:?}"),
                        }
                    }
                }
                rng.shuffle(&mut frames);
            }
            for frame in frames {
                lower.put(frame, 0).unwrap();
            }
        });

        assert_eq!(lower.stats().free_frames, TREE_FRAMES);
        assert_eq!(lower.stats().free_huge, TREE_HUGE);
    }
}
