//! Packed entries for the allocators data structures

use core::mem::{align_of, size_of};
use core::ops::RangeBounds;
use core::sync::atomic::Ordering::Release;
use core::sync::atomic::{self, AtomicU16, AtomicU32, AtomicU64};

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};

pub trait AtomicArray<T: Copy, const L: usize> {
    /// Overwrite the content of the whole array non-atomically.
    ///
    /// This is faster than atomics but does not handle race conditions.
    fn atomic_fill(&self, e: T);
}

impl<T: Atomic, const L: usize> AtomicArray<T, L> for [Atom<T>; L] {
    fn atomic_fill(&self, e: T) {
        // cast to raw memory to let the compiler use vector instructions
        #[allow(invalid_reference_casting)]
        let mem = unsafe { &mut *(self.as_ptr() as *mut [T; L]) };
        mem.fill(e);
        // memory ordering has to be enforced with a memory barrier
        atomic::fence(Release);
    }
}

/// Level 3 entry
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Preferred {
    /// If this subtree locked for a reservation.
    pub locked: bool,
    /// The local tree copy.
    pub tree: Option<LocalTree>,
}

/// Local tree copy
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct LocalTree {
    pub frame: usize,
    pub free: usize,
}
impl Atomic for Preferred {
    type I = AtomicU64;
}

impl Preferred {
    pub fn tree(start: usize, free: usize, locked: bool) -> Self {
        Self {
            locked,
            tree: Some(LocalTree { frame: start, free }),
        }
    }

    /// Decrement or lock the tree if decrement fails
    pub fn dec_or_lock(self, num_frames: usize) -> Option<Self> {
        if let Some(t) = self.tree
            && t.free >= num_frames
        {
            Some(Preferred::tree(t.frame, t.free - num_frames, self.locked))
        } else {
            Some(Preferred {
                locked: true,
                ..self
            })
        }
    }

    pub fn unlock(self) -> Option<Self> {
        self.locked.then_some(Self {
            locked: false,
            ..self
        })
    }

    /// Increments the free frames counter.
    pub fn inc(
        self,
        num_frames: usize,
        max: usize,
        check_start: impl FnOnce(usize) -> bool,
    ) -> Option<Self> {
        if let Some(LocalTree { frame: start, free }) = self.tree
            && check_start(start)
        {
            let frames = free + num_frames;
            assert!(frames <= max, "inc failed {self:?}");
            Some(Preferred::tree(start, frames, self.locked))
        } else {
            None
        }
    }

    /// Increment the free counter and unlock.
    pub fn inc_unlock(
        self,
        num_frames: usize,
        max: usize,
        check_start: impl FnOnce(usize) -> bool,
    ) -> Option<Self> {
        if let Some(LocalTree { frame: start, free }) = self.tree
            && self.locked
            && check_start(start)
        {
            let frames = free + num_frames;
            assert!(frames <= max, "inc failed {self:?}");
            Some(Preferred::tree(start, frames, false))
        } else {
            None
        }
    }
}

#[bitfield(u64)]
struct PTreeBits {
    #[bits(16)]
    free: usize,
    locked: bool,
    reserved: bool,
    #[bits(46)]
    start: usize,
}
impl From<u64> for Preferred {
    fn from(value: u64) -> Self {
        let bits = PTreeBits::from(value);
        if bits.reserved() {
            Self {
                locked: bits.locked(),
                tree: Some(LocalTree {
                    frame: bits.start() * 64,
                    free: bits.free(),
                }),
            }
        } else {
            Self {
                locked: bits.locked(),
                tree: None,
            }
        }
    }
}
impl From<Preferred> for u64 {
    fn from(value: Preferred) -> Self {
        match value.tree {
            Some(LocalTree { free, frame: start }) => PTreeBits::new()
                .with_reserved(true)
                .with_locked(value.locked)
                .with_free(free)
                .with_start(start / 64)
                .into(),
            None => PTreeBits::new().with_locked(value.locked).into(),
        }
    }
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
pub struct Tree {
    /// Number of free 4K frames.
    #[bits(15)]
    pub free: usize,
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
}
impl Atomic for Tree {
    type I = AtomicU16;
}
impl Tree {
    pub fn empty(span: usize) -> Self {
        debug_assert!(span < (1 << 15));
        Self::new().with_free(span)
    }
    /// Creates a new entry.
    pub fn new_with(frames: usize, reserved: bool) -> Self {
        debug_assert!(frames < (1 << 15));
        Self::new().with_free(frames).with_reserved(reserved)
    }
    /// Increments the free frames counter.
    pub fn inc(self, num_frames: usize, max: usize) -> Option<Self> {
        let frames = self.free() + num_frames;
        if frames <= max {
            Some(self.with_free(frames))
        } else {
            None
        }
    }
    /// Reserves this entry if its frame count is in `range`.
    pub fn reserve<R: RangeBounds<usize>>(self, free: R) -> Option<Self> {
        if !self.reserved() && free.contains(&self.free()) {
            Some(self.with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Add the frames from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    pub fn unreserve_add(self, add: usize, max: usize) -> Option<Self> {
        let frames = self.free() + add;
        if self.reserved() && frames <= max {
            Some(self.with_free(frames).with_reserved(false))
        } else {
            None
        }
    }
    /// Set the free counter to zero if it is large enough for synchronization
    pub fn sync_steal(self, free: usize, min: usize) -> Option<Self> {
        if self.reserved() && self.free() + free > min {
            Some(self.with_free(0))
        } else {
            None
        }
    }
}

/// Manages huge frame, that can be allocated as base frames.
#[bitfield(u16)]
#[derive(PartialEq, Eq)]
pub struct HugeEntry {
    /// Number of free 4K frames or u16::MAX for a huge frame.
    count: u16,
}
impl Atomic for HugeEntry {
    type I = AtomicU16;
}
impl HugeEntry {
    /// Creates an entry marked as allocated huge frame.
    pub fn new_huge() -> Self {
        Self::new().with_count(u16::MAX)
    }
    /// Creates a new entry with the given free counter.
    pub fn new_free(free: usize) -> Self {
        Self::new().with_count(free as _)
    }
    /// Returns wether this entry is allocated as huge frame.
    pub fn huge(self) -> bool {
        self.count() == u16::MAX
    }
    /// Returns the free frames counter
    pub fn free(self) -> usize {
        if !self.huge() {
            self.count() as _
        } else {
            0
        }
    }
    /// Try to allocate this entry as huge frame.
    pub fn mark_huge(self, span: usize) -> Option<Self> {
        if self.free() == span {
            Some(Self::new_huge())
        } else {
            None
        }
    }
    /// Decrement the free frames counter.
    pub fn dec(self, num_frames: usize) -> Option<Self> {
        if !self.huge() && self.free() >= num_frames {
            Some(Self::new_free(self.free() - num_frames))
        } else {
            None
        }
    }
    /// Increments the free frames counter.
    pub fn inc(self, span: usize, num_frames: usize) -> Option<Self> {
        if !self.huge() && self.free() <= span - num_frames {
            Some(Self::new_free(self.free() + num_frames))
        } else {
            None
        }
    }
}

/// Pair of huge entries that can be changed at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(align(4))]
pub struct HugePair(pub HugeEntry, pub HugeEntry);
impl Atomic for HugePair {
    type I = AtomicU32;
}

const _: () = assert!(size_of::<HugePair>() == 2 * size_of::<HugeEntry>());
const _: () = assert!(align_of::<HugePair>() == size_of::<HugePair>());

impl HugePair {
    /// Apply `f` to both entries.
    pub fn map<F: Fn(HugeEntry) -> Option<HugeEntry>>(self, f: F) -> Option<HugePair> {
        Some(HugePair(f(self.0)?, f(self.1)?))
    }
    /// Check if `f` is true for both entries.
    pub fn all<F: Fn(HugeEntry) -> bool>(self, f: F) -> bool {
        f(self.0) && f(self.1)
    }
}
impl From<u32> for HugePair {
    fn from(value: u32) -> Self {
        let [a, b, c, d] = value.to_ne_bytes();
        Self(
            HugeEntry(u16::from_ne_bytes([a, b])),
            HugeEntry(u16::from_ne_bytes([c, d])),
        )
    }
}
impl From<HugePair> for u32 {
    fn from(value: HugePair) -> Self {
        let ([a, b], [c, d]) = (value.0 .0.to_ne_bytes(), value.1 .0.to_ne_bytes());
        u32::from_ne_bytes([a, b, c, d])
    }
}

/// Next element of a list
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Next {
    #[default]
    Outside,
    End,
    Some(usize),
}

impl Next {
    pub fn some(self) -> Option<usize> {
        match self {
            Next::Some(i) => Some(i),
            Next::End => None,
            Next::Outside => panic!("invalid list element"),
        }
    }
}
impl From<Option<usize>> for Next {
    fn from(v: Option<usize>) -> Self {
        match v {
            Some(i) => Self::Some(i),
            None => Self::End,
        }
    }
}
impl From<u64> for Next {
    fn from(value: u64) -> Self {
        const MAX_SUB: u64 = u64::MAX - 1;
        match value {
            u64::MAX => Next::Outside,
            MAX_SUB => Next::End,
            _ => Next::Some(value as _),
        }
    }
}
impl From<Next> for u64 {
    fn from(value: Next) -> Self {
        match value {
            Next::Outside => u64::MAX,
            Next::End => u64::MAX - 1,
            Next::Some(v) => v as _,
        }
    }
}
impl Atomic for Next {
    type I = AtomicU64;
}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::sync::atomic::AtomicU64;

    use crate::atomic::Atom;
    use crate::entry::Preferred;
    use crate::frame::PT_LEN;

    #[test]
    fn pt() {
        let pt: [Atom<u64>; PT_LEN] = [const { Atom(AtomicU64::new(0)) }; PT_LEN];
        pt[0].compare_exchange(0, 42).unwrap();
        pt[0].fetch_update(|v| Some(v + 1)).unwrap();
        assert_eq!(pt[0].load(), 43);
    }

    #[test]
    fn preferred() {
        println!("{:#?}", Preferred::from(200354));
    }
}
