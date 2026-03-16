use core::mem::align_of;
use core::sync::atomic::AtomicU32;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::warn;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::lower::HugeId;
use crate::util::{Align, size_of_slice};
use crate::*;

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

pub struct Trees<'a> {
    /// Array of level 3 entries, which are the roots of the trees
    entries: &'a [Atom<Tree>],
    /// Default tier for new trees or entirely free trees,
    default: Tier,
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
        default: Tier,
    ) -> Self {
        assert!(buffer.len() >= Self::metadata_size(frames));

        let len = frames.div_ceil(TREE_FRAMES);
        let entries: &mut [Atom<Tree>] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) };

        if let Some(tree_init) = tree_init {
            for (i, e) in entries.iter_mut().enumerate() {
                let frames = tree_init(i * TREE_FRAMES);
                *e = Atom::new(Tree::with(frames, false, default));
            }
        }

        Self { entries, default }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn stats(&self) -> TreeStats {
        let mut stats = TreeStats::default();
        for entry in self.entries {
            let tree = entry.load();
            stats.free_frames += tree.free();
            stats.free_trees += tree.free() / TREE_FRAMES;

            let tier = &mut stats.tiers[tree.tier().0 as usize];
            tier.free_frames += tree.free();
            tier.alloc_frames += TREE_FRAMES - tree.free();
        }
        stats
    }

    pub fn stats_at(&self, i: TreeId) -> (Tier, usize, bool) {
        let tree = self.entries[i.0].load();
        (tree.tier(), tree.free(), tree.reserved())
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

    pub fn get(
        &self,
        i: TreeId,
        free: usize,
        check: impl Fn(Tier, usize) -> Option<Tier>,
    ) -> Option<Tier> {
        let mut tier = None;
        self.entries[i.0]
            .fetch_update(|e| {
                e.get(free, |t, f| {
                    tier = check(t, f);
                    tier
                })
            })
            .ok()
            .map(|_| tier.unwrap())
    }

    pub fn put(&self, i: TreeId, free: usize, policy: PolicyFn) {
        self.entries[i.0]
            .fetch_update(|v| Some(v.put(free, policy, self.default)))
            .expect("Put failed");
    }

    /// Increment or reserve the tree, returning the old free counter if it was reserved
    pub fn put_or_reserve(
        &self,
        i: TreeId,
        free: usize,
        tier: Tier,
        may_reserve: bool,
        policy: PolicyFn,
    ) -> Option<usize> {
        let mut reserved = false;
        let tree = self.entries[i.0]
            .fetch_update(|v| {
                let v = v.put(free, policy, self.default);
                if may_reserve && !v.reserved() && v.tier() == tier && v.free() > Self::MIN_FREE {
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
        check: impl Fn(Tier, usize) -> Option<Tier>,
    ) -> Option<(usize, Tier)> {
        let mut tier = None;
        self.entries[i.0]
            .fetch_update(|v| {
                v.reserve(|t, f| {
                    tier = check(t, f);
                    tier
                })
            })
            .map(|v| (v.free(), tier.unwrap()))
            .ok()
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: TreeId, free: usize, tier: Tier, policy: PolicyFn) {
        self.entries[i.0]
            .fetch_update(|v| v.unreserve_add(free, tier, policy, self.default))
            .expect("Unreserve failed");
    }

    /// Iterate through all trees, trying to find the best N fits, then trying to `access` them
    pub fn search_best<const N: usize, R>(
        &self,
        start: TreeId,
        offset: usize,
        len: usize,
        rate: impl Fn(Tier, usize) -> Policy,
        access: impl Fn(TreeId) -> Result<R>,
    ) -> Result<R> {
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
            if tree.reserved() {
                continue;
            }
            match rate(tree.tier(), tree.free()) {
                Policy::Match(u8::MAX) => match access(i) {
                    Err(Error::Memory) => {}
                    r => return r,
                },
                Policy::Match(p) => {
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

    pub fn change(
        &self,
        matcher: TreeMatch,
        change: TreeChange,
        fetch_free: impl Fn(TreeId) -> usize,
    ) -> Result<()> {
        if let Some(i) = matcher.id {
            self.change_at(i, matcher.tier, matcher.free, change, || fetch_free(i))
        } else {
            self.search(TreeId(0), 0, self.len(), |i| {
                self.change_at(i, matcher.tier, matcher.free, change.clone(), || {
                    fetch_free(i)
                })
            })
        }
    }

    fn change_at(
        &self,
        id: TreeId,
        tier: Option<Tier>,
        free: usize,
        change: TreeChange,
        fetch_free: impl Fn() -> usize + Copy,
    ) -> Result<()> {
        match self.entries[id.0].fetch_update(|e| e.change(tier, free, change.clone(), fetch_free))
        {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Memory),
        }
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
    tier: Tier,
}

const _: () = assert!(1 << Tree::FREE_BITS > TREE_FRAMES);
const _: () = assert!(Tree::TIER_BITS == Tier::BITS);

impl Atomic for Tree {
    type I = AtomicU32;
}
impl Tree {
    /// Creates a new entry.
    fn with(free: usize, reserved: bool, tier: Tier) -> Self {
        assert!(free <= TREE_FRAMES);
        Self::new()
            .with_free(free)
            .with_reserved(reserved)
            .with_tier(tier)
    }
    /// Increments the free frames counter.
    fn put(mut self, free: usize, policy: PolicyFn, default: Tier) -> Self {
        let free = self.free() + free;
        assert!(free <= TREE_FRAMES, "{free}");

        // Check if transition is allowed by policy
        if free == TREE_FRAMES && policy(self.tier(), default, free) != Policy::Invalid {
            self.set_tier(default);
        }
        self.with_free(free)
    }
    /// Decrements the free frames counter if it is large enough
    fn get(self, free: usize, mut check: impl FnMut(Tier, usize) -> Option<Tier>) -> Option<Self> {
        if self.free() >= free
            && let Some(tier) = check(self.tier(), self.free())
        {
            Some(self.with_free(self.free() - free).with_tier(tier))
        } else {
            None
        }
    }
    /// Reserves this entry if its frame count is in `range`.
    fn reserve(self, mut check: impl FnMut(Tier, usize) -> Option<Tier>) -> Option<Self> {
        if !self.reserved()
            && let Some(tier) = check(self.tier(), self.free())
        {
            Some(Self::with(0, true, tier))
        } else {
            None
        }
    }
    /// Add the frames from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    fn unreserve_add(
        self,
        free: usize,
        tier: Tier,
        policy: PolicyFn,
        default: Tier,
    ) -> Option<Self> {
        if self.reserved() {
            Some(
                self.with_reserved(false)
                    .with_tier(if policy(tier, self.tier(), free) == Policy::Demote {
                        tier
                    } else {
                        self.tier()
                    })
                    .put(free, policy, default),
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
    /// Change the entry if it is not reserved and the tier and free counter conditions match
    fn change(
        mut self,
        tier: Option<Tier>,
        free: usize,
        change: TreeChange,
        fetch_free: impl Fn() -> usize,
    ) -> Option<Self> {
        if !self.reserved() && tier.is_none_or(|k| k == self.tier()) && self.free() >= free {
            if let Some(tier) = change.tier {
                self.set_tier(tier);
            }
            match change.operation {
                Some(TreeOperation::Offline) => self.set_free(0),
                Some(TreeOperation::Online) => {
                    if self.free() != 0 {
                        warn!("Online non-empty tree: tier={:?}", self.tier());
                        return None;
                    }
                    self.set_free(fetch_free());
                }
                None => {}
            }
            Some(self)
        } else {
            None
        }
    }
}
