use core::sync::atomic::AtomicU32;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::warn;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::cache::{Align, Aligned, Invalidate};
use crate::cxl::CXLockGuard;
use crate::lower::HugeId;
use crate::util::{OrdBy, SortedBuffer};
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

type TreeChunk = Align<[Atom<Tree>; Trees::CHUNK_TREES]>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkId(pub usize);
impl ChunkId {
    pub const fn new(tree: TreeId) -> Self {
        Self(tree.0 / Trees::CHUNK_TREES)
    }
    pub const fn as_tree(self, i: usize) -> TreeId {
        TreeId(self.0 * Trees::CHUNK_TREES + i)
    }
}

pub struct Trees<'a> {
    /// Number of trees
    len: usize,
    /// Array of level 3 entries, which are the roots of the trees
    chunks: &'a [Uncached<TreeChunk>],
    /// Default class for new trees or entirely free trees,
    default: Class,
}
unsafe impl<'a> Aligned for Trees<'a> {
    unsafe fn cache_line(&self) -> *const CacheLine {
        self.chunks.as_ptr() as _
    }
    fn cache_lines(&self) -> usize {
        self.chunks.len()
    }
}

impl fmt::Debug for Trees<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.len;
        let mut free = 0;
        let mut partial = 0;
        for (i, chunk) in self.chunks.iter().enumerate() {
            chunk.flush_invalidate(); // -> access reads from memory
            // Safety: read-only access
            let chunk = unsafe { chunk.inner() };
            let len = (self.len - i * Self::CHUNK_TREES).min(Self::CHUNK_TREES);
            for e in chunk[..len].iter() {
                let f = e.load().free();
                if f == TREE_FRAMES {
                    free += 1;
                } else if f > Self::MIN_FREE {
                    partial += 1;
                }
            }
        }
        write!(f, "(total: {max}, free: {free}, partial: {partial})")?;
        Ok(())
    }
}

impl<'a> Trees<'a> {
    pub const MIN_FREE: usize = TREE_FRAMES / 16;
    const CHUNK_TREES: usize = CacheLine::SIZE / size_of::<Atom<Tree>>();

    /// Cache lines required for the metadata
    pub const fn metadata_size(frames: usize) -> usize {
        // Event thought the elements are not cache aligned, the whole array should be
        (size_of::<Atom<Tree>>() * frames.div_ceil(TREE_FRAMES)).div_ceil(CacheLine::SIZE)
    }

    pub unsafe fn metadata(&mut self) -> &'a Uncached<[UnsafeCacheLine]> {
        const _: () = assert!(size_of::<TreeChunk>() == CacheLine::SIZE);
        unsafe {
            Uncached::from_ref(slice::from_raw_parts(
                self.chunks.as_ptr().cast(),
                self.chunks.len(),
            ))
        }
    }

    /// Initialize the tree array
    pub fn new(
        frames: usize,
        buffer: &'a Uncached<[UnsafeCacheLine]>,
        tree_init: Option<impl Fn(TreeId) -> usize>,
        default: Class,
    ) -> Result<Self> {
        if buffer.len() < Self::metadata_size(frames) {
            return Err(Error::Initialization);
        }

        let len = frames.div_ceil(TREE_FRAMES);
        let entries = unsafe {
            buffer
                .cast_slice::<TreeChunk>(len.div_ceil(Self::CHUNK_TREES))?
                .transpose()
        };

        if let Some(tree_init) = tree_init {
            for (i, chunk) in entries.iter().enumerate() {
                let chunk = unsafe { chunk.borrow() };
                let chunk_len = (len - i * Self::CHUNK_TREES).min(Self::CHUNK_TREES);
                for (j, e) in chunk[..chunk_len].iter().enumerate() {
                    let frames = tree_init(ChunkId(i).as_tree(j));
                    e.store(Tree::with(frames, false, default));
                }
            }
        }

        Ok(Self {
            len,
            chunks: entries,
            default,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn stats(&self) -> TreeStats {
        let mut stats = TreeStats::default();
        for (i, chunk) in self.chunks.iter().enumerate() {
            chunk.flush_invalidate(); // -> access reads from memory
            // Safety: read-only access
            let chunk = unsafe { chunk.inner() };
            let len = (self.len - i * Self::CHUNK_TREES).min(Self::CHUNK_TREES);
            for entry in chunk[..len].iter() {
                let tree = entry.load();
                stats.free_frames += tree.free();
                stats.free_trees += tree.free() / TREE_FRAMES;

                let class = &mut stats.classes[tree.class().0 as usize];
                class.free_frames += tree.free();
                class.alloc_frames += TREE_FRAMES - tree.free();
            }
        }
        stats
    }

    pub fn stats_at(&self, i: TreeId) -> (Class, usize, bool) {
        let chunk = &self.chunks[ChunkId::new(i).0];
        chunk.flush_invalidate();
        let tree = unsafe { chunk.inner()[i.0 % Self::CHUNK_TREES].load() };
        (tree.class(), tree.free(), tree.reserved())
    }

    /// Sync with the global tree, stealing its counters
    pub fn sync(&self, _guard: &CXLockGuard, i: TreeId, min: usize) -> Option<usize> {
        let chunk = &self.chunks[ChunkId::new(i).0];
        let chunk = unsafe { chunk.borrow() };
        chunk[i.0 % Self::CHUNK_TREES]
            .try_update(|e| e.sync_steal(min))
            .map(|tree| tree.free())
            .ok()
    }

    pub fn steal(
        &self,
        _guard: &CXLockGuard<'_, 'a>,
        i: TreeId,
        class: Class,
        free: usize,
        policy: PolicyFn,
    ) -> Option<Class> {
        assert!(i.0 < self.len);
        let chunk = &self.chunks[ChunkId::new(i).0];
        let chunk = unsafe { chunk.borrow() };
        let mut new_class = None;
        chunk[i.0 % Self::CHUNK_TREES]
            .try_update(|e| {
                let e = e.steal(class, free, policy);
                new_class = e.map(|e| e.class());
                e
            })
            .ok()
            .map(|_| new_class.unwrap())
    }

    pub fn put(&self, _guard: &CXLockGuard<'_, 'a>, i: TreeId, free: usize, policy: PolicyFn) {
        assert!(i.0 < self.len);
        let chunk = &self.chunks[ChunkId::new(i).0];
        let chunk = unsafe { chunk.borrow() };
        chunk[i.0 % Self::CHUNK_TREES].update(|v| v.put(free, policy, self.default));
    }

    pub fn reserve_or_steal(
        &self,
        _guard: &CXLockGuard<'_, 'a>,
        i: TreeId,
        class: Class,
        free: usize,
        policy: PolicyFn,
    ) -> Option<(bool, usize, Class)> {
        assert!(i.0 < self.len);
        let chunk = &self.chunks[ChunkId::new(i).0];
        let chunk = unsafe { chunk.borrow() };
        let mut new = None;
        chunk[i.0 % Self::CHUNK_TREES]
            .try_update(|v| {
                let v = v.reserve_or_steal(free, policy, class);
                new = v.map(|v| (v.reserved(), v.class()));
                v
            })
            .map(|v| (new.unwrap().0, v.free(), new.unwrap().1))
            .ok()
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(
        &self,
        _guard: &CXLockGuard<'_, 'a>,
        i: TreeId,
        free: usize,
        class: Class,
        policy: PolicyFn,
    ) {
        assert!(i.0 < self.len);
        let chunk = &self.chunks[ChunkId::new(i).0];
        let chunk = unsafe { chunk.borrow() };
        chunk[i.0 % Self::CHUNK_TREES]
            .try_update(|v| v.unreserve_add(free, class, policy, self.default))
            .expect("Unreserve failed");
    }

    /// Iterate through all trees, trying to find the best N fits, then trying to `access` them
    pub fn search_best<const N: usize, R>(
        &self,
        _guard: &CXLockGuard<'_, 'a>,
        start: TreeId,
        offset: usize,
        len: usize,
        rate: impl Fn(Class, usize) -> Policy,
        access: impl Fn(TreeId) -> Result<R>,
    ) -> Result<R> {
        let mut best = SortedBuffer::<N, OrdBy<(Policy, bool), TreeId>>::new();
        let offset = offset / Self::CHUNK_TREES;
        let len = len.div_ceil(Self::CHUNK_TREES);

        for i in offset..len {
            // Alternating between before and after start
            let off = if i.is_multiple_of(2) {
                (i / 2).cast_signed()
            } else {
                -i.div_ceil(2).cast_signed()
            };
            let s = (ChunkId::new(start).0 + self.chunks.len()).cast_signed();
            let i = ChunkId((s + off).cast_unsigned() % self.chunks.len());

            let chunk = &self.chunks[i.0];
            chunk.flush_invalidate(); // <- load from memory

            let chunk = unsafe { chunk.inner() }; // SAFETY: read-only access
            for (j, tree) in chunk.iter().enumerate() {
                let tree_id = i.as_tree(j);
                if tree_id.0 >= self.len {
                    continue;
                }

                let tree = tree.load();
                if tree.reserved() {
                    continue;
                }
                match rate(tree.class(), tree.free()) {
                    // Try accessing perfect matches directly
                    Policy::Match(u8::MAX) => match access(tree_id) {
                        Err(Error::Memory) => {}
                        r => return r,
                    },
                    // Skip invalid matches
                    Policy::Invalid => {}
                    // Cache the best matches
                    p => best.add(OrdBy((p, tree.free() == TREE_FRAMES), tree_id)),
                }
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
        _guard: &CXLockGuard<'_, 'a>,
        start: TreeId,
        offset: usize,
        len: usize,
        access: impl Fn(TreeId) -> Result<R>,
    ) -> Result<R> {
        let offset = offset / Self::CHUNK_TREES;
        let len = len.div_ceil(Self::CHUNK_TREES);

        for i in offset..len {
            // Alternating between before and after start
            let off = if i.is_multiple_of(2) {
                (i / 2) as isize
            } else {
                -(i.div_ceil(2) as isize)
            };
            let s = (ChunkId::new(start).0 + self.chunks.len()) as isize;
            let i = ChunkId((s + off) as usize % self.chunks.len());

            let len = (self.len - i.0 * Self::CHUNK_TREES).min(Self::CHUNK_TREES);
            for j in 0..len {
                let tree_id = i.as_tree(j);
                match access(tree_id) {
                    Err(Error::Memory) => {}
                    r => return r,
                }
            }
        }
        Err(Error::Memory)
    }

    pub fn change(
        &self,
        _guard: &CXLockGuard<'_, 'a>,
        matcher: TreeMatch,
        change: TreeChange,
        fetch_free: impl Fn(TreeId) -> usize,
    ) -> Result<()> {
        if let Some(i) = matcher.id {
            self.change_at(_guard, i, matcher.class, matcher.free, change, || {
                fetch_free(i)
            })
        } else {
            self.search(_guard, TreeId(0), 0, self.len, |i| {
                self.change_at(
                    _guard,
                    i,
                    matcher.class,
                    matcher.free,
                    change.clone(),
                    || fetch_free(i),
                )
            })
        }
    }

    fn change_at(
        &self,
        _guard: &CXLockGuard<'_, 'a>,
        id: TreeId,
        class: Option<Class>,
        free: usize,
        change: TreeChange,
        fetch_free: impl Fn() -> usize + Copy,
    ) -> Result<()> {
        assert!(id.0 < self.len);
        let chunk = &self.chunks[ChunkId::new(id).0];
        let chunk = unsafe { chunk.borrow() };
        match chunk[id.0 % Self::CHUNK_TREES]
            .try_update(|e| e.change(class, free, change.clone(), fetch_free))
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
    class: Class,
}

const _: () = assert!(1 << Tree::FREE_BITS > TREE_FRAMES);
const _: () = assert!(Tree::CLASS_BITS == Class::BITS);

impl Atomic for Tree {
    type I = AtomicU32;
}
impl Tree {
    /// Creates a new entry.
    fn with(free: usize, reserved: bool, class: Class) -> Self {
        assert!(free <= TREE_FRAMES);
        Self::new()
            .with_free(free)
            .with_reserved(reserved)
            .with_class(class)
    }
    /// Increments the free frames counter.
    fn put(mut self, free: usize, policy: PolicyFn, default: Class) -> Self {
        let free = self.free() + free;
        assert!(free <= TREE_FRAMES, "{free}");

        // Check if transition is allowed by policy
        if free == TREE_FRAMES && policy(self.class(), default, free) != Policy::Invalid {
            self.set_class(default);
        }
        self.with_free(free)
    }
    /// Decrements the free frames counter if it is large enough
    fn steal(self, class: Class, free: usize, policy: PolicyFn) -> Option<Self> {
        if self.free() >= free && !self.reserved() {
            let new_class = match (policy)(class, self.class(), free) {
                Policy::Match(_) => class,
                // Cannot demote reserved trees (requires changing local entries)
                Policy::Demote if self.reserved() => return None,
                Policy::Demote => class,
                Policy::Steal => self.class(),
                Policy::Invalid => return None,
            };
            Some(self.with_free(self.free() - free).with_class(new_class))
        } else {
            None
        }
    }
    /// Reserve or steal frames from this entry.
    fn reserve_or_steal(self, free: usize, policy: PolicyFn, class: Class) -> Option<Self> {
        if self.free() >= free && !self.reserved() {
            match (policy)(class, self.class(), free) {
                // Reserve the entry if it is not reserved, possibly demoting it.
                Policy::Match(_) | Policy::Demote if !self.reserved() => {
                    Some(Self::with(0, true, class))
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
        class: Class,
        policy: PolicyFn,
        default: Class,
    ) -> Option<Self> {
        if self.reserved() {
            Some(
                self.with_reserved(false)
                    .with_class(match policy(class, self.class(), free) {
                        Policy::Match(_) => self.class(),
                        Policy::Demote => class,
                        Policy::Steal | Policy::Invalid => panic!("unreserve invalid class"),
                    })
                    .put(free, policy, default),
            )
        } else {
            None
        }
    }
    /// Set the free counter to zero if it is large enough for synchronization
    fn sync_steal(self, min: usize) -> Option<Self> {
        if self.reserved() && self.free() > 0 && self.free() >= min {
            Some(self.with_free(0))
        } else {
            None
        }
    }
    /// Change the entry if it is not reserved and the class and free counter conditions match
    fn change(
        mut self,
        class: Option<Class>,
        free: usize,
        change: TreeChange,
        fetch_free: impl Fn() -> usize,
    ) -> Option<Self> {
        if !self.reserved() && class.is_none_or(|k| k == self.class()) && self.free() >= free {
            if let Some(class) = change.class {
                self.set_class(class);
            }
            match change.operation {
                Some(TreeOperation::Offline) => self.set_free(0),
                Some(TreeOperation::Online) if self.free() == 0 => self.set_free(fetch_free()),
                Some(TreeOperation::Online) => {
                    warn!("Online non-empty tree: class={:?}", self.class());
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

#[cfg(test)]
mod tests {
    use core::cell::{Cell, UnsafeCell};

    use super::*;
    use crate::cxl::CXLock;

    #[test]
    fn initialization_and_stats_exclude_padding() {
        for count in [1, Trees::CHUNK_TREES, Trees::CHUNK_TREES + 1] {
            let buffer = [const { UnsafeCell::new(CacheLine([0xff; CacheLine::SIZE])) }; 2];
            let frames = count * TREE_FRAMES;
            let buffer = &buffer[..Trees::metadata_size(frames)];
            let initialized = Cell::new(0);
            let trees = Trees::new(
                frames,
                Uncached::from_ref(buffer),
                Some(|id: TreeId| {
                    assert_eq!(id.0, initialized.get());
                    assert!(id.0 < count);
                    initialized.set(initialized.get() + 1);
                    TREE_FRAMES
                }),
                Class(0),
            )
            .unwrap();

            assert_eq!(trees.len(), count);
            assert_eq!(trees.chunks.len(), count.div_ceil(Trees::CHUNK_TREES));
            assert_eq!(initialized.get(), count);
            let stats = trees.stats();
            assert_eq!(stats.free_frames, frames);
            assert_eq!(stats.free_trees, count);
            assert_eq!(stats.classes[0].free_frames, frames);
            assert_eq!(stats.classes[0].alloc_frames, 0);
        }
    }

    #[test]
    fn searches_exclude_padding() {
        let buffer = [const { UnsafeCell::new(CacheLine([0; CacheLine::SIZE])) }; 2];
        let count = Trees::CHUNK_TREES + 1;
        let trees = Trees::new(
            count * TREE_FRAMES,
            Uncached::from_ref(&buffer[..]),
            Some(|_| TREE_FRAMES),
            Class(0),
        )
        .unwrap();
        let lock_buffer = [const { UnsafeCell::new(CacheLine([0; CacheLine::SIZE])) }; 2];
        let mut lock = CXLock::init(0, 1, Uncached::from_ref(&lock_buffer[..])).unwrap();
        let guard = lock.lock();
        let visited = Cell::new(0u64);
        let access = |id: TreeId| -> Result<()> {
            assert!(id.0 < count);
            assert_eq!(visited.get() & (1 << id.0), 0);
            visited.set(visited.get() | (1 << id.0));
            Err(Error::Memory)
        };

        assert_eq!(
            trees.search(&guard, TreeId(0), 0, count, access),
            Err(Error::Memory)
        );
        assert_eq!(visited.get(), (1 << count) - 1);
        visited.set(0);
        assert_eq!(
            trees.search_best::<1, _>(
                &guard,
                TreeId(0),
                0,
                count,
                |_, _| Policy::Match(u8::MAX),
                access,
            ),
            Err(Error::Memory)
        );
        assert_eq!(visited.get(), (1 << count) - 1);
    }

    #[test]
    fn searches_start_at_requested_tree_chunk() {
        let buffer = [const { UnsafeCell::new(CacheLine([0; CacheLine::SIZE])) }; 4];
        let count = 3 * Trees::CHUNK_TREES + 1;
        let trees = Trees::new(
            count * TREE_FRAMES,
            Uncached::from_ref(&buffer[..]),
            Some(|_| TREE_FRAMES),
            Class(0),
        )
        .unwrap();
        let lock_buffer = [const { UnsafeCell::new(CacheLine([0; CacheLine::SIZE])) }; 2];
        let mut lock = CXLock::init(0, 1, Uncached::from_ref(&lock_buffer[..])).unwrap();
        let guard = lock.lock();
        let start = TreeId(Trees::CHUNK_TREES + 3);
        let expected = TreeId(Trees::CHUNK_TREES);

        assert_eq!(
            trees.search(&guard, start, 0, Trees::CHUNK_TREES, Ok),
            Ok(expected)
        );
        assert_eq!(
            trees.search_best::<1, _>(
                &guard,
                start,
                0,
                Trees::CHUNK_TREES,
                |_, _| Policy::Match(u8::MAX),
                Ok,
            ),
            Ok(expected)
        );
    }
}
