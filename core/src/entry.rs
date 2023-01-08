use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::{Range, RangeBounds};
use core::sync::atomic::{AtomicU16, AtomicU32, AtomicU64};

use bitfield_struct::bitfield;
use log::error;

use crate::atomic::Atomic;

/// Level 3 entry
#[bitfield(u64)]
#[derive(Default, PartialEq, Eq)]
pub struct TreeNode {
    /// Number of free 4K pages.
    #[bits(20)]
    pub free: usize,
    /// Index of the next tree node (linked list).
    #[bits(43)]
    pub idx: usize,
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
}
impl Atomic for TreeNode {
    type I = AtomicU64;
}
impl TreeNode {
    pub const IDX_MAX: usize = (1 << Self::IDX_BITS) - 1;
    pub const IDX_END: usize = (1 << Self::IDX_BITS) - 2;

    pub fn empty(span: usize) -> Self {
        Self::new().with_free(span)
    }
    /// Creates a new entry referring to a level 2 page table.
    pub fn new_with(free: usize, idx: usize) -> Self {
        Self::new().with_free(free).with_idx(idx)
    }
    /// Increments the free pages counter.
    pub fn inc(self, num_pages: usize, max: usize) -> Option<Self> {
        let pages = self.free() + num_pages;
        if pages <= max {
            Some(self.with_free(pages))
        } else {
            None
        }
    }
    /// Reserves this entry if it has at least `min` pages.
    pub fn reserve_min(self, min: usize) -> Option<Self> {
        if !self.reserved() && self.free() >= min {
            Some(self.with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Reserves this entry if its page count is in `range`.
    pub fn reserve_partial(self, range: Range<usize>) -> Option<Self> {
        if !self.reserved() && range.contains(&self.free()) {
            Some(self.with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Add the pages from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    pub fn unreserve_add(self, add: usize, max: usize) -> Option<Self> {
        let pages = self.free() + add;
        if self.reserved() && pages <= max {
            Some(self.with_free(pages).with_reserved(false))
        } else {
            error!("{self:?} + {add}, {pages} <= {max}");
            None
        }
    }
}

/// Level 3 entry
#[bitfield(u64, debug = false)]
#[derive(PartialEq, Eq)]
pub struct ReservedTree {
    /// Number of free 4K pages.
    #[bits(16)]
    pub free: usize,
    /// If this subtree is locked by a CPU.
    pub locked: bool,
    /// Start pfn / 64 within this reserved tree.
    #[bits(47)]
    start_raw: usize,
}
impl Atomic for ReservedTree {
    type I = AtomicU64;
}
impl Default for ReservedTree {
    fn default() -> Self {
        Self::new().with_start_raw(Self::START_RAW_MAX)
    }
}
impl ReservedTree {
    const START_RAW_MAX: usize = (1 << Self::START_RAW_BITS) - 1;

    /// Creates a new entry referring to a level 2 page table.
    pub fn new_with(free: usize, start: usize) -> Self {
        Self::new().with_free(free).with_start(start)
    }
    /// If this entry has a valid start pfn.
    pub fn has_start(self) -> bool {
        self.start_raw() < Self::START_RAW_MAX
    }
    /// Start page frame number.
    #[inline(always)]
    pub fn start(self) -> usize {
        self.start_raw() * 64
    }
    #[inline(always)]
    pub fn with_start(self, start: usize) -> Self {
        let raw = start / 64;
        debug_assert!(raw < (1 << Self::START_RAW_BITS));
        self.with_start_raw(raw)
    }
    #[inline(always)]
    pub fn set_start(&mut self, start: usize) {
        *self = self.with_start(start);
    }

    /// Decrements the free pages counter.
    pub fn dec(self, num_pages: usize) -> Option<Self> {
        if self.has_start() && self.free() >= num_pages {
            Some(self.with_free(self.free() - num_pages))
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    pub fn inc<F: FnOnce(usize) -> bool>(
        self,
        num_pages: usize,
        max: usize,
        check_start: F,
    ) -> Option<Self> {
        if !check_start(self.start()) {
            return None;
        }
        let pages = self.free() + num_pages;
        if pages <= max {
            Some(self.with_free(pages))
        } else {
            None
        }
    }
    /// Updates the reserve flag to `new` if `old != new`.
    pub fn toggle_locked(self, new: bool) -> Option<Self> {
        (self.locked() != new).then_some(self.with_locked(new))
    }
}

impl fmt::Debug for ReservedTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReservedTree")
            .field("free", &self.free())
            .field("locked", &self.locked())
            .field("start", &self.start())
            .finish()
    }
}

#[bitfield(u16)]
#[derive(Default, PartialEq, Eq)]
pub struct Tree {
    /// Number of free 4K pages.
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
    /// Creates a new entry referring to a level 2 page table.
    pub fn new_with(pages: usize, reserved: bool) -> Self {
        debug_assert!(pages < (1 << 15));
        Self::new().with_free(pages).with_reserved(reserved)
    }
    /// Increments the free pages counter.
    pub fn inc(self, num_pages: usize, max: usize) -> Option<Self> {
        let pages = self.free() + num_pages;
        if pages <= max {
            Some(self.with_free(pages))
        } else {
            None
        }
    }
    /// Reserves this entry if its page count is in `range`.
    pub fn reserve<R: RangeBounds<usize>>(self, free: R) -> Option<Self> {
        if !self.reserved() && free.contains(&self.free()) {
            Some(self.with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Add the pages from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    pub fn unreserve_add(self, add: usize, max: usize) -> Option<Self> {
        let pages = self.free() + add;
        if self.reserved() && pages <= max {
            Some(self.with_free(pages).with_reserved(false))
        } else {
            error!("{self:?} + {add}, {pages} <= {max}");
            None
        }
    }
}

#[bitfield(u16)]
#[derive(Default, PartialEq, Eq)]
pub struct Child {
    /// Number of free 4K pages.
    #[bits(15)]
    pub free: usize,
    /// If this entry is reserved by a CPU.
    pub page: bool,
}
impl Atomic for Child {
    type I = AtomicU16;
}
impl Child {
    pub fn new_page() -> Self {
        Self::new().with_page(true)
    }
    pub fn new_free(free: usize) -> Self {
        Self::new().with_free(free)
    }
    pub fn mark_page(self, span: usize) -> Option<Self> {
        if self.free() == span {
            Some(Self::new_page())
        } else {
            None
        }
    }
    /// Decrement the free pages counter.
    pub fn dec(self, num_pages: usize) -> Option<Self> {
        if !self.page() && self.free() >= num_pages {
            Some(Self::new_free(self.free() - num_pages))
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    pub fn inc(self, span: usize, num_pages: usize) -> Option<Self> {
        if !self.page() && self.free() <= span - num_pages {
            Some(Self::new_free(self.free() + num_pages))
        } else {
            None
        }
    }
}

/// Pair of level 2 entries that can be changed at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(align(4))]
pub struct ChildPair(pub Child, pub Child);
impl Atomic for ChildPair {
    type I = AtomicU32;
}

const _: () = assert!(size_of::<ChildPair>() == 2 * size_of::<Child>());
const _: () = assert!(align_of::<ChildPair>() == size_of::<ChildPair>());

impl ChildPair {
    pub fn map<F: Fn(Child) -> Option<Child>>(self, f: F) -> Option<ChildPair> {
        match (f(self.0), f(self.1)) {
            (Some(a), Some(b)) => Some(ChildPair(a, b)),
            _ => None,
        }
    }
    pub fn all<F: Fn(Child) -> bool>(self, f: F) -> bool {
        f(self.0) && f(self.1)
    }
}
impl From<u32> for ChildPair {
    fn from(value: u32) -> Self {
        unsafe { core::mem::transmute(value) }
    }
}
impl From<ChildPair> for u32 {
    fn from(value: ChildPair) -> Self {
        unsafe { core::mem::transmute(value) }
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::sync::atomic::AtomicU64;

    use crate::atomic::Atom;
    use crate::table::PT_LEN;

    #[test]
    fn pt() {
        const A: Atom<u64> = Atom::raw(AtomicU64::new(0));
        let pt: [Atom<u64>; PT_LEN] = [A; PT_LEN];
        pt[0].compare_exchange(0, 42).unwrap();
        pt[0].fetch_update(|v| Some(v + 1)).unwrap();
        assert_eq!(pt[0].load(), 43);
    }
}
