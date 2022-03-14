use core::cmp::Ordering;
use core::fmt;

use bitfield_struct::bitfield;
use log::error;

use crate::table::Table;
use crate::Size;

#[bitfield(u64)]
pub struct Entry {
    /// Number of subtrees where no page is allocated.
    #[bits(20)]
    pub empty: usize,
    /// Number of subtrees where at least one l0 page is allocated.
    #[bits(20)]
    pub partial_l0: usize,
    /// Number of subtrees where at least one l1 page is allocated.
    #[bits(20)]
    pub partial_l1: usize,
    #[bits(4)]
    _p: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    None,
    IncEmpty,
    IncPartialL0,
    IncPartialL1,
    DecPartialL0,
    DecPartialL1,
}
impl Change {
    #[inline]
    pub fn p_inc(huge: bool) -> Change {
        if huge {
            Change::IncPartialL1
        } else {
            Change::IncPartialL0
        }
    }
    #[inline]
    pub fn p_dec(huge: bool) -> Change {
        if huge {
            Change::DecPartialL1
        } else {
            Change::DecPartialL0
        }
    }
}

impl Entry {
    #[inline]
    pub fn dec_partial(self, huge: bool) -> Option<Self> {
        if !huge && self.partial_l0() > 0 {
            Some(self.with_partial_l0(self.partial_l0() - 1))
        } else if huge && self.partial_l1() > 0 {
            Some(self.with_partial_l1(self.partial_l1() - 1))
        } else {
            None
        }
    }
    #[inline]
    pub fn dec_empty(self) -> Option<Self> {
        (self.empty() > 0).then(|| self.with_empty(self.empty() - 1))
    }
    #[inline]
    pub fn change(self, dec: Change) -> Option<Self> {
        match dec {
            Change::IncEmpty => Some(self.with_empty(self.empty() + 1)),
            Change::IncPartialL0 => Some(self.with_partial_l0(self.partial_l0() + 1)),
            Change::IncPartialL1 => Some(self.with_partial_l1(self.partial_l1() + 1)),
            Change::DecPartialL0 if self.partial_l0() > 0 => Some(
                self.with_partial_l0(self.partial_l0() - 1)
                    .with_empty(self.empty() + 1),
            ),
            Change::DecPartialL1 if self.partial_l1() > 0 => Some(
                self.with_partial_l1(self.partial_l1() - 1)
                    .with_empty(self.empty() + 1),
            ),
            _ => None,
        }
    }
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry")
            .field("empty", &self.empty())
            .field("partial_l0", &self.partial_l0())
            .field("partial_l1", &self.partial_l1())
            .finish()
    }
}

#[bitfield(u64)]
pub struct Entry3 {
    /// Number of free 4K pages.
    #[bits(20)]
    pub free: usize,
    /// Metadata for the higher level allocators.
    #[bits(41)]
    pub idx: usize,
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
    /// If this subtree contains allocated huge pages.
    pub huge: bool,
    /// If this subtree is allocated as giant page.
    pub page: bool,
}

impl Entry3 {
    pub const IDX_MAX: usize = (1 << 41) - 1;
    /// Creates a new entry where this is allocated as a single giant page.
    #[inline]
    pub fn new_giant() -> Entry3 {
        Entry3::new().with_page(true)
    }
    /// Creates a new entry referring to a layer 2 page table.
    #[inline]
    pub fn new_table(pages: usize, size: Size, reserved: bool) -> Entry3 {
        Entry3::new()
            .with_free(pages)
            .with_huge(size == Size::L1)
            .with_reserved(reserved)
    }
    /// Decrements the free pages counter.
    #[inline]
    pub fn dec(self, huge: bool) -> Option<Entry3> {
        let sub = Table::span(huge as _);
        if !self.page()
            && (self.huge() == huge || self.free() == Table::span(2))
            && self.free() >= sub
        {
            Some(
                self.with_free(self.free() - sub)
                    .with_huge(huge)
                    .with_reserved(true),
            )
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    #[inline]
    pub fn inc(self, huge: bool, max: usize) -> Option<Entry3> {
        if self.page() || self.huge() != huge {
            return None;
        }
        let pages = self.free() + Table::span(huge as _);
        match pages.cmp(&max) {
            Ordering::Greater => None,
            Ordering::Equal => Some(self.with_free(max).with_huge(false)),
            Ordering::Less => Some(self.with_free(pages)),
        }
    }
    /// Increments the free pages counter and checks for `idx` to match.
    #[inline]
    pub fn inc_idx(self, huge: bool, idx: usize, max: usize) -> Option<Entry3> {
        if self.page() || self.huge() != huge || self.idx() != idx {
            return None;
        }
        let pages = self.free() + Table::span(huge as _);
        match pages.cmp(&max) {
            Ordering::Greater => None,
            Ordering::Equal => Some(self.with_free(max).with_huge(false)),
            Ordering::Less => Some(self.with_free(pages)),
        }
    }
    /// Reserves this entry.
    #[inline]
    pub fn reserve(self, huge: bool) -> Option<Entry3> {
        if !self.page()
            && !self.reserved()
            && (self.free() == Table::span(2) || self.huge() == huge)
            && self.free() >= Table::span(huge as _)
        {
            Some(self.with_free(0).with_huge(huge).with_reserved(true))
        } else {
            None
        }
    }
    /// Reserves this entry if it is partially filled.
    #[inline]
    pub fn reserve_partial(self, huge: bool, min: usize) -> Option<Entry3> {
        if !self.page()
            && !self.reserved()
            && self.free() > min
            && self.free() < Table::span(2)
            && self.huge() == huge
        {
            Some(self.with_huge(huge).with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Reserves this entry if it is completely empty.
    #[inline]
    pub fn reserve_empty(self, huge: bool) -> Option<Entry3> {
        if !self.page() && !self.reserved() && self.free() == Table::span(2) {
            Some(self.with_huge(huge).with_reserved(true).with_free(0))
        } else {
            None
        }
    }
    /// Clears the reserve flag of this entry.
    #[inline]
    pub fn unreserve(self) -> Option<Entry3> {
        if !self.page() && self.reserved() {
            Some(self.with_reserved(false))
        } else {
            None
        }
    }
    /// Add the pages from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    #[inline]
    pub fn unreserve_add(self, huge: bool, other: Entry3, max: usize) -> Option<Entry3> {
        let pages = self.free() + other.free();
        if !self.page()
            && self.reserved()
            && pages <= max
            && (self.huge() == huge || self.free() == Table::span(2))
            && (other.huge() == huge || other.free() == Table::span(2))
        {
            Some(
                self.with_free(pages)
                    .with_huge(huge && pages < max)
                    .with_reserved(false),
            )
        } else {
            error!("{self:?} + {other:?}, {pages} <= {max}");
            None
        }
    }
}

impl fmt::Debug for Entry3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry3")
            .field("pages", &self.free())
            .field("idx", &self.idx())
            .field("reserved", &self.reserved())
            .field("huge", &self.huge())
            .field("page", &self.page())
            .finish()
    }
}

#[bitfield(u64)]
pub struct Entry2 {
    /// Number of free pages.
    #[bits(10)]
    pub free: usize,
    /// Index of the layer one page table.
    #[bits(9)]
    pub i1: usize,
    #[bits(43)]
    _p: u64,
    /// If the whole page table area is allocated as giant page.
    pub giant: bool,
    /// If this is allocated as large page.
    pub page: bool,
}

impl Entry2 {
    /// Creates a new entry referencing a layer one page table.
    #[inline]
    pub fn new_table(pages: usize, i1: usize) -> Self {
        Self::new().with_free(pages).with_i1(i1)
    }
    #[inline]
    pub fn mark_huge(self) -> Option<Self> {
        if !self.giant() && !self.page() && self.free() == Table::span(Size::L1 as _) {
            Some(Entry2::new().with_free(0).with_i1(0).with_page(true))
        } else {
            None
        }
    }
    /// Decrement the free pages counter.
    #[inline]
    pub fn dec(self, i1: usize) -> Option<Self> {
        if !self.page() && !self.giant() && self.i1() == i1 && self.free() > 0 {
            Some(self.with_free(self.free() - 1))
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    #[inline]
    pub fn inc_partial(self, i1: usize) -> Option<Self> {
        if !self.giant()
            && !self.page()
            && i1 == self.i1()
            && self.free() > 0 // is the child pt already initialized?
            && self.free() < Table::LEN
        {
            Some(self.with_free(self.free() + 1))
        } else {
            None
        }
    }
    /// Increments the free pages counter.
    #[inline]
    pub fn inc(self) -> Option<Self> {
        if !self.giant() && !self.page() && self.free() < Table::LEN {
            Some(self.with_free(self.free() + 1))
        } else {
            None
        }
    }
}

impl fmt::Debug for Entry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry2")
            .field("pages", &self.free())
            .field("i1", &self.i1())
            .field("page", &self.page())
            .field("giant", &self.giant())
            .finish()
    }
}

/// Level 1 page table entry
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u64)]
pub enum Entry1 {
    Empty = 0,
    Page = 1,
}

impl From<u64> for Entry1 {
    fn from(v: u64) -> Self {
        match v {
            0 => Entry1::Empty,
            _ => Entry1::Page,
        }
    }
}

impl From<Entry1> for u64 {
    fn from(v: Entry1) -> u64 {
        v as u64
    }
}

#[cfg(test)]
mod test {
    use crate::table::Table;

    #[test]
    fn pt() {
        let pt: Table<u64> = Table::empty();
        pt.cas(0, 0, 42).unwrap();
        pt.update(0, |v| Some(v + 1)).unwrap();
        assert_eq!(pt.get(0), 43);
    }
}
