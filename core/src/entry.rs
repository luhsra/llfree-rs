use core::fmt;
use core::mem::{align_of, size_of};

use bitfield_struct::bitfield;
use log::error;

use crate::atomic::AtomicValue;

#[bitfield(u64)]
pub struct Entry3 {
    /// Number of free 4K pages.
    #[bits(20)]
    pub free: usize,
    /// Metadata for the higher level allocators.
    #[bits(43)]
    pub idx: usize,
    // TODO: huge-page counter?
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
}

impl Default for Entry3 {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicValue for Entry3 {
    type V = u64;
}

impl Entry3 {
    pub const IDX_MAX: usize = (1 << 41) - 1;

    #[inline]
    pub fn empty(span: usize) -> Entry3 {
        Entry3::new().with_free(span)
    }
    /// Creates a new entry referring to a level 2 page table.
    #[inline]
    pub fn new_table(pages: usize, reserved: bool) -> Entry3 {
        Entry3::new().with_free(pages).with_reserved(reserved)
    }
    /// Decrements the free pages counter.
    #[inline]
    pub fn dec(self, num_pages: usize) -> Option<Entry3> {
        if self.idx() != Self::IDX_MAX && self.free() >= num_pages {
            Some(self.with_free(self.free() - num_pages).with_reserved(true))
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    #[inline]
    pub fn inc(self, num_pages: usize, max: usize) -> Option<Entry3> {
        let pages = self.free() + num_pages;
        if pages <= max {
            Some(self.with_free(pages))
        } else {
            None
        }
    }
    /// Increments the free pages counter and checks for `idx` to match.
    #[inline]
    pub fn inc_idx(self, num_pages: usize, idx: usize, max: usize) -> Option<Entry3> {
        if self.idx() == idx {
            self.inc(num_pages, max)
        } else {
            None
        }
    }
    /// Reserves this entry.
    #[inline]
    pub fn reserve(self, min: usize) -> Option<Entry3> {
        if !self.reserved() && self.free() > min {
            Some(self.with_free(0).with_reserved(true))
        } else {
            None
        }
    }
    /// Reserves this entry if it is partially filled.
    #[inline]
    pub fn reserve_partial(self, min: usize, span: usize) -> Option<Entry3> {
        if !self.reserved() && self.free() > min && self.free() < span {
            Some(self.with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Reserves this entry if it is completely empty.
    #[inline]
    pub fn reserve_empty(self, span: usize) -> Option<Entry3> {
        if !self.reserved() && self.free() == span {
            Some(self.with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Clears the reserve flag of this entry.
    #[inline]
    pub fn unreserve(self) -> Option<Entry3> {
        if self.reserved() {
            Some(self.with_reserved(false))
        } else {
            None
        }
    }
    /// Add the pages from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    #[inline]
    pub fn unreserve_add(self, add: usize, max: usize) -> Option<Entry3> {
        let pages = self.free() + add;
        if self.reserved() && pages <= max {
            Some(self.with_free(pages).with_reserved(false))
        } else {
            error!("{self:?} + {add}, {pages} <= {max}");
            None
        }
    }
}

impl fmt::Debug for Entry3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry3")
            .field("free", &self.free())
            .field("idx", &self.idx())
            .field("reserved", &self.reserved())
            .finish()
    }
}

#[bitfield(u16)]
pub struct Entry2 {
    /// Number of free pages.
    #[bits(15)]
    pub free: usize,
    /// If this is allocated as large page.
    pub page: bool,
}

impl AtomicValue for Entry2 {
    type V = u16;
}

impl Entry2 {
    #[inline]
    pub fn new_page() -> Self {
        Self::new().with_page(true)
    }
    #[inline]
    pub fn new_free(free: usize) -> Self {
        Self::new().with_free(free)
    }

    #[inline]
    pub fn mark_huge(self, span: usize) -> Option<Self> {
        if !self.page() && self.free() == span {
            Some(Self::new_page())
        } else {
            None
        }
    }
    /// Decrement the free pages counter.
    #[inline]
    pub fn dec(self, num_pages: usize) -> Option<Self> {
        if !self.page() && self.free() >= num_pages {
            Some(Self::new_free(self.free() - num_pages))
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    #[inline]
    pub fn inc(self, span: usize, num_pages: usize) -> Option<Self> {
        if !self.page() && self.free() <= span - num_pages {
            Some(Self::new_free(self.free() + num_pages))
        } else {
            None
        }
    }
}

impl fmt::Debug for Entry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry2")
            .field("free", &self.free())
            .field("page", &self.page())
            .finish()
    }
}

/// Pair of level 2 entries that can be changed at once.
#[derive(Debug, Clone, Copy)]
#[repr(align(4))]
pub struct Entry2Pair(pub Entry2, pub Entry2);

const _: () = assert!(size_of::<Entry2Pair>() == 2 * size_of::<Entry2>());
const _: () = assert!(align_of::<Entry2Pair>() == size_of::<Entry2Pair>());

impl Entry2Pair {
    pub fn both<F: Fn(Entry2) -> Option<Entry2>>(self, f: F) -> Option<Entry2Pair> {
        match (f(self.0), f(self.1)) {
            (Some(a), Some(b)) => Some(Entry2Pair(a, b)),
            _ => None,
        }
    }
}

impl From<u32> for Entry2Pair {
    fn from(v: u32) -> Self {
        Entry2Pair((v as u16).into(), ((v >> 16) as u16).into())
    }
}

impl From<Entry2Pair> for u32 {
    fn from(v: Entry2Pair) -> Self {
        u16::from(v.0) as u32 | ((u16::from(v.1) as u32) << 16)
    }
}

impl AtomicValue for Entry2Pair {
    type V = u32;
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::table::{ATable, PT_LEN};

    #[test]
    fn pt() {
        type Table = ATable<u64, PT_LEN>;

        let pt: Table = Table::empty();
        pt.cas(0, 0, 42).unwrap();
        pt.update(0, |v| Some(v + 1)).unwrap();
        assert_eq!(pt.get(0), 43);
    }
}
