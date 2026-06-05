use core::mem::align_of;
use core::sync::atomic::AtomicU32;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::warn;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::lower::HugeId;
use crate::util::{Align, OrdBy, SortedBuffer, size_of_slice};
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
    /// Default cluster for new trees or entirely free trees,
    default: Cluster,
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
        default: Cluster,
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

            let cluster = &mut stats.clusters[tree.cluster().0 as usize];
            cluster.free_frames += tree.free();
            cluster.alloc_frames += TREE_FRAMES - tree.free();
        }
        stats
    }

    pub fn stats_at(&self, i: TreeId) -> (Cluster, usize, bool) {
        let tree = self.entries[i.0].load();
        (tree.cluster(), tree.free(), tree.reserved())
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
            .try_update(|e| e.sync_steal(min))
            .map(|tree| tree.free())
            .ok()
    }

    pub fn steal(
        &self,
        i: TreeId,
        cluster: Cluster,
        free: usize,
        policy: PolicyFn,
    ) -> Option<Cluster> {
        let mut new_cluster = None;
        self.entries[i.0]
            .try_update(|e| {
                let e = e.steal(cluster, free, policy);
                new_cluster = e.map(|e| e.cluster());
                e
            })
            .ok()
            .map(|_| new_cluster.unwrap())
    }

    pub fn put(&self, i: TreeId, free: usize, policy: PolicyFn) {
        self.entries[i.0].update(|v| v.put(free, policy, self.default));
    }

    pub fn reserve_or_steal(
        &self,
        i: TreeId,
        cluster: Cluster,
        free: usize,
        policy: PolicyFn,
    ) -> Option<(bool, usize, Cluster)> {
        let mut new = None;
        self.entries[i.0]
            .try_update(|v| {
                let v = v.reserve_or_steal(free, policy, cluster);
                new = v.map(|v| (v.reserved(), v.cluster()));
                v
            })
            .map(|v| (new.unwrap().0, v.free(), new.unwrap().1))
            .ok()
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: TreeId, free: usize, cluster: Cluster, policy: PolicyFn) {
        self.entries[i.0]
            .try_update(|v| v.unreserve_add(free, cluster, policy, self.default))
            .expect("Unreserve failed");
    }

    /// Iterate through all trees, trying to find the best N fits, then trying to `access` them
    pub fn search_best<const N: usize, R>(
        &self,
        start: TreeId,
        offset: usize,
        len: usize,
        rate: impl Fn(Cluster, usize) -> Policy,
        access: impl Fn(TreeId) -> Result<R>,
    ) -> Result<R> {
        let mut best = SortedBuffer::<N, OrdBy<(Policy, bool), TreeId>>::new();

        for i in offset..len {
            // Alternating between before and after start
            let off = if i.is_multiple_of(2) {
                (i / 2).cast_signed()
            } else {
                -i.div_ceil(2).cast_signed()
            };
            let s = (start.0 + self.entries.len()).cast_signed();
            let i = TreeId((s + off).cast_unsigned() % self.entries.len());

            let tree = self.entries[i.0].load();
            if tree.reserved() {
                continue;
            }
            match rate(tree.cluster(), tree.free()) {
                // Try accessing perfect matches directly
                Policy::Match(u8::MAX) => match access(i) {
                    Err(Error::Memory) => {}
                    r => return r,
                },
                // Skip invalid matches
                Policy::Invalid => {}
                // Cache the best matches
                p => best.add(OrdBy((p, tree.free() == TREE_FRAMES), i)),
            }
        }

        // Try accessing the best matches
        for OrdBy(_prio, i) in best.iter().rev() {
            match access(*i) {
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
            self.change_at(i, matcher.cluster, matcher.free, change, || fetch_free(i))
        } else {
            self.search(TreeId(0), 0, self.len(), |i| {
                self.change_at(i, matcher.cluster, matcher.free, change.clone(), || {
                    fetch_free(i)
                })
            })
        }
    }

    fn change_at(
        &self,
        id: TreeId,
        cluster: Option<Cluster>,
        free: usize,
        change: TreeChange,
        fetch_free: impl Fn() -> usize + Copy,
    ) -> Result<()> {
        match self.entries[id.0].try_update(|e| e.change(cluster, free, change.clone(), fetch_free))
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
    cluster: Cluster,
}

const _: () = assert!(1 << Tree::FREE_BITS > TREE_FRAMES);
const _: () = assert!(Tree::CLUSTER_BITS == Cluster::BITS);

impl Atomic for Tree {
    type I = AtomicU32;
}
impl Tree {
    /// Creates a new entry.
    fn with(free: usize, reserved: bool, cluster: Cluster) -> Self {
        assert!(free <= TREE_FRAMES);
        Self::new()
            .with_free(free)
            .with_reserved(reserved)
            .with_cluster(cluster)
    }
    /// Increments the free frames counter.
    fn put(mut self, free: usize, policy: PolicyFn, default: Cluster) -> Self {
        let free = self.free() + free;
        assert!(free <= TREE_FRAMES, "{free}");

        // Check if transition is allowed by policy
        if free == TREE_FRAMES && policy(self.cluster(), default, free) != Policy::Invalid {
            self.set_cluster(default);
        }
        self.with_free(free)
    }
    /// Decrements the free frames counter if it is large enough
    fn steal(self, cluster: Cluster, free: usize, policy: PolicyFn) -> Option<Self> {
        if self.free() >= free && !self.reserved() {
            let new_cluster = match (policy)(cluster, self.cluster(), free) {
                Policy::Match(_) => cluster,
                // Cannot demote reserved trees (requires changing local entries)
                Policy::Demote if self.reserved() => return None,
                Policy::Demote => cluster,
                Policy::Steal => self.cluster(),
                Policy::Invalid => return None,
            };
            Some(self.with_free(self.free() - free).with_cluster(new_cluster))
        } else {
            None
        }
    }
    /// Reserve or steal frames from this entry.
    fn reserve_or_steal(self, free: usize, policy: PolicyFn, cluster: Cluster) -> Option<Self> {
        if self.free() >= free && !self.reserved() {
            match (policy)(cluster, self.cluster(), free) {
                // Reserve the entry if it is not reserved, possibly demoting it.
                Policy::Match(_) | Policy::Demote if !self.reserved() => {
                    Some(Self::with(0, true, cluster))
                }
                // Steal frames from matching entries, even if they are reserved.
                Policy::Match(_) => Some(self.with_free(self.free() - free)),
                // Cannot demote reserved trees (requires changing local entries)
                Policy::Demote => None,
                Policy::Steal => Some(self.with_free(self.free() - free)),
                Policy::Invalid => None,
            }
        } else {
            None
        }
    }
    /// Add the frames from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    fn unreserve_add(
        self,
        free: usize,
        cluster: Cluster,
        policy: PolicyFn,
        default: Cluster,
    ) -> Option<Self> {
        if self.reserved() {
            Some(
                self.with_reserved(false)
                    .with_cluster(match policy(cluster, self.cluster(), free) {
                        Policy::Match(_) => self.cluster(),
                        Policy::Demote => cluster,
                        Policy::Steal | Policy::Invalid => panic!("unreserve invalid cluster"),
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
    /// Change the entry if it is not reserved and the cluster and free counter conditions match
    fn change(
        mut self,
        cluster: Option<Cluster>,
        free: usize,
        change: TreeChange,
        fetch_free: impl Fn() -> usize,
    ) -> Option<Self> {
        if !self.reserved() && cluster.is_none_or(|k| k == self.cluster()) && self.free() >= free {
            if let Some(cluster) = change.cluster {
                self.set_cluster(cluster);
            }
            match change.operation {
                Some(TreeOperation::Offline) => self.set_free(0),
                Some(TreeOperation::Online) if self.free() == 0 => self.set_free(fetch_free()),
                Some(TreeOperation::Online) => {
                    warn!("Online non-empty tree: cluster={:?}", self.cluster());
                    return None;
                }
                None => {}
            }
            Some(self)
        } else {
            None
        }
    }
}
