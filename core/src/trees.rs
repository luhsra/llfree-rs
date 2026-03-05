use core::mem::align_of;
use core::ops::RangeBounds;
use core::sync::atomic::AtomicU32;
use core::{fmt, slice};

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::lower::HugeId;
use crate::util::{Align, size_of_slice};
use crate::{Error, FrameId, Tier, KindStats, Policy, Result, TREE_FRAMES, TreeStats};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TreeId(pub usize);
impl TreeId {
    pub const fn as_frame(self) -> FrameId {
        FrameId(self.0 * TREE_FRAMES)
    }
    pub const fn as_huge(self) -> HugeId {
        self.as_frame().as_huge()
    }
    pub const fn as_row(self) -> RowId {
        self.as_frame().as_row()
    }
    pub const fn from_bits(value: u64) -> Self {
        Self(value as _)
    }
    pub const fn into_bits(self) -> u64 {
        self.0 as _
    }
}
impl core::ops::Add<Self> for TreeId {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl fmt::Display for TreeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

impl fmt::Debug for TreeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Default)]
pub struct Trees<'a> {
    /// Array of level 3 entries, which are the roots of the trees
    entries: &'a [Atom<Tree>],
}

impl fmt::Debug for Trees<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut free = 0;
        let mut partial = 0;
        for e in self.entries {
            let f = e.load().free();
            if f == TREE_FRAMES {
                free += 1;
            } else if f > Self::MIN_FREE {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, free: {free}, partial: {partial})")?;
        Ok(())
    }
}

impl<'a> Trees<'a> {
    pub const MIN_FREE: usize = TREE_FRAMES / 16;

    pub const fn metadata_size(frames: usize) -> usize {
        // Event thought the elements are not cache aligned, the whole array should be
        size_of_slice::<Atom<Tree>>(frames.div_ceil(TREE_FRAMES))
            .next_multiple_of(align_of::<Align>())
    }

    pub unsafe fn metadata(&mut self) -> &'a mut [u8] {
        let len = Self::metadata_size(self.len() * TREE_FRAMES);
        unsafe { slice::from_raw_parts_mut(self.entries.as_ptr().cast_mut().cast(), len) }
    }

    /// Initialize the tree array
    pub fn new(
        frames: usize,
        buffer: &'a mut [u8],
        tree_init: Option<impl Fn(usize) -> usize>,
    ) -> Self {
        assert!(buffer.len() >= Self::metadata_size(frames));

        let len = frames.div_ceil(TREE_FRAMES);
        let entries: &mut [Atom<Tree>] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) };

        if let Some(tree_init) = tree_init {
            for (i, e) in entries.iter_mut().enumerate() {
                let frames = tree_init(i * TREE_FRAMES);
                *e = Atom::new(Tree::with(frames, false, Tier(0)));
            }
        }

        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn stats(&self, kinds: &mut [KindStats]) -> TreeStats {
        self.entries
            .iter()
            .fold(TreeStats::default(), |mut acc, e| {
                let tree = e.load();
                acc.free_frames += tree.free();
                if tree.free() != TREE_FRAMES
                    && let Some(kind) = kinds.get_mut(tree.kind().0 as usize)
                {
                    kind.free += tree.free();
                    kind.alloc += TREE_FRAMES - tree.free();
                }
                acc.free_trees += tree.free() / TREE_FRAMES;
                acc
            })
    }

    pub fn stats_at(&self, i: TreeId) -> (Tier, usize, bool) {
        let tree = self.entries[i.0].load();
        (tree.kind(), tree.free(), tree.reserved())
    }

    /// Return the number of entirely free trees
    pub fn free(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.load().free() == TREE_FRAMES)
            .count()
    }
    /// Return the total sum of the tree counters
    pub fn free_frames(&self) -> usize {
        self.entries.iter().map(|e| e.load().free()).sum()
    }
    /// Sync with the global tree, stealing its counters
    pub fn sync(&self, i: TreeId, min: usize) -> Option<usize> {
        self.entries[i.0]
            .fetch_update(|e| e.sync_steal(min))
            .map(|tree| tree.free())
            .ok()
    }

    pub fn get(&self, i: TreeId, free: usize, kind: Tier) -> bool {
        self.entries[i.0]
            .fetch_update(|e| e.get(free, kind))
            .is_ok()
    }

    pub fn get_demote(
        &self,
        i: TreeId,
        free: usize,
        other: Tier,
        reserve: fn(Tier, usize) -> Policy,
    ) -> Option<Tier> {
        self.entries[i.0]
            .fetch_update(|e| e.get_demote(free, other, reserve))
            .ok()
            .map(|e| {
                if reserve(other, e.free()) == Policy::Demotes {
                    other
                } else {
                    e.kind()
                }
            })
    }

    pub fn put(&self, i: TreeId, free: usize) {
        self.entries[i.0]
            .fetch_update(|v| Some(v.put(free)))
            .expect("Put failed");
    }

    /// Increment or reserve the tree, returning the old free counter if it was reserved
    pub fn put_or_reserve(
        &self,
        i: TreeId,
        free: usize,
        kind: Tier,
        may_reserve: bool,
    ) -> Option<usize> {
        let mut reserved = false;
        let tree = self.entries[i.0]
            .fetch_update(|v| {
                let v = v.put(free);
                if may_reserve && !v.reserved() && v.kind() == kind && v.free() > Self::MIN_FREE {
                    // Reserve the tree that was targeted by the last N frees
                    reserved = true;
                    Some(v.with_free(0).with_reserved(true))
                } else {
                    reserved = false; // <- This one is very important if CAS fails!
                    Some(v)
                }
            })
            .unwrap();

        if reserved { Some(tree.free()) } else { None }
    }

    pub fn reserve(
        &self,
        i: TreeId,
        free: impl RangeBounds<usize> + Clone,
        kind: Tier,
    ) -> Option<usize> {
        self.entries[i.0]
            .fetch_update(|v| v.reserve(free.clone(), kind))
            .map(|v| v.free())
            .ok()
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: TreeId, free: usize, kind: Tier, reserve: fn(Tier, usize) -> Policy) {
        self.entries[i.0]
            .fetch_update(|v| v.unreserve_add(free, kind, reserve))
            .expect("Unreserve failed");
    }

    /// Iterate through all trees, trying to find the best N fits, then trying to `access` them
    pub fn search_best<const N: usize>(
        &self,
        start: TreeId,
        offset: usize,
        len: usize,
        reserve: fn(Tier, usize) -> Policy,
        access: impl Fn(TreeId) -> Result<FrameId>,
    ) -> Result<FrameId> {
        #[derive(Clone, Copy)]
        struct Best {
            i: TreeId,
            prio: u8,
        }
        let mut best: [Option<Best>; N] = [None; N];

        for i in offset..len {
            // Alternating between before and after start
            let off = if i.is_multiple_of(2) {
                (i / 2) as isize
            } else {
                -(i.div_ceil(2) as isize)
            };
            let s = (start.0 + self.entries.len()) as isize;
            let i = TreeId((s + off) as usize % self.entries.len());

            let tree = self.entries[i.0].load();
            match reserve(tree.kind(), tree.free()) {
                Policy::Good(p) => {
                    let pos = best.iter().position(|e| match e {
                        None => true,
                        Some(best) => p > best.prio,
                    });
                    if let Some(pos) = pos {
                        for j in pos..N - 1 {
                            best[j + 1] = best[j];
                        }
                        best[pos] = Some(Best { i, prio: p });
                    }
                }
                Policy::Best => match access(i) {
                    Err(Error::Memory) => {}
                    r => return r,
                },
                _ => {}
            }
        }

        for Best { i, .. } in best.into_iter().flatten() {
            match access(i) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }

        Err(Error::Memory)
    }

    /// Iterate through all trees as long `access` returns `Error::Memory`
    pub fn search<R>(
        &self,
        start: TreeId,
        offset: usize,
        len: usize,
        access: impl Fn(TreeId) -> Result<R>,
    ) -> Result<R> {
        for i in offset..len {
            // Alternating between before and after start
            let off = if i.is_multiple_of(2) {
                (i / 2) as isize
            } else {
                -(i.div_ceil(2) as isize)
            };
            let s = (start.0 + self.entries.len()) as isize;
            let i = TreeId((s + off) as usize % self.entries.len());
            match access(i) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }
        Err(Error::Memory)
    }
}

/// Tree entry for 4K frames
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
struct Tree {
    /// Number of free 4K frames.
    #[bits(28)]
    free: usize,
    /// If this subtree is reserved by a CPU.
    reserved: bool,
    /// Are the frames movable?
    #[bits(3)]
    kind: Tier,
}

const _: () = assert!(1 << Tree::FREE_BITS >= TREE_FRAMES);
const _: () = assert!(Tree::KIND_BITS == Tier::BITS);

impl Atomic for Tree {
    type I = AtomicU32;
}
impl Tree {
    /// Creates a new entry.
    fn with(free: usize, reserved: bool, kind: Tier) -> Self {
        assert!(free <= TREE_FRAMES);
        Self::new()
            .with_free(free)
            .with_reserved(reserved)
            .with_kind(kind)
    }
    /// Increments the free frames counter.
    fn put(self, free: usize) -> Self {
        let free = self.free() + free;
        assert!(free <= TREE_FRAMES, "{free}");
        self.with_free(free)
    }

    fn get(self, free: usize, kind: Tier) -> Option<Self> {
        if !self.reserved() && self.free() >= free && self.kind() == kind {
            Some(self.with_free(self.free() - free))
        } else {
            None
        }
    }

    fn get_demote(
        mut self,
        free: usize,
        other: Tier,
        reserve: fn(Tier, usize) -> Policy,
    ) -> Option<Self> {
        if self.free() >= free {
            if reserve(other, self.free()) == Policy::Demotes {
                self.set_kind(other);
            }
            Some(self.with_free(self.free() - free))
        } else {
            None
        }
    }

    /// Reserves this entry if its frame count is in `range`.
    fn reserve(self, free: impl RangeBounds<usize>, kind: Tier) -> Option<Self> {
        if !self.reserved()
            && free.contains(&self.free())
            && (kind == self.kind() || self.free() == TREE_FRAMES)
        {
            Some(Self::with(0, true, kind))
        } else {
            None
        }
    }
    /// Add the frames from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    fn unreserve_add(
        self,
        free: usize,
        kind: Tier,
        reserve: fn(Tier, usize) -> Policy,
    ) -> Option<Self> {
        if self.reserved() {
            Some(
                self.with_reserved(false)
                    .with_kind(if reserve(kind, free) == Policy::Demotes {
                        kind
                    } else {
                        self.kind()
                    })
                    .put(free),
            )
        } else {
            None
        }
    }
    /// Set the free counter to zero if it is large enough for synchronization
    fn sync_steal(self, min: usize) -> Option<Self> {
        if self.reserved() && self.free() > min {
            Some(self.with_free(0))
        } else {
            None
        }
    }
}
