use core::fmt;
use std::cmp::Ordering;

use bitfield_struct::bitfield;
use log::error;

use crate::table::Table;
use crate::Size;

#[bitfield(u64)]
pub struct Entry {
    #[bits(20)]
    pub empty: usize,
    #[bits(20)]
    pub partial_l0: usize,
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
    pub pages: usize,
    /// Metadata for the higher level allocators.
    #[bits(41)]
    pub idx: usize,
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
    /// If this subtree contains allocated huge pages.
    pub huge: bool,
    /// If this subtree is allocated as giant page.
    pub giant: bool,
}

impl Entry3 {
    pub const IDX_MAX: usize = (1 << 41) - 1;

    #[inline]
    pub fn new_giant() -> Entry3 {
        Entry3::new().with_giant(true)
    }
    #[inline]
    pub fn new_table(pages: usize, size: Size, reserved: bool) -> Entry3 {
        Entry3::new()
            .with_pages(pages)
            .with_huge(size == Size::L1)
            .with_reserved(reserved)
    }
    /// Decrements the free pages counter and sets the size and reserved bits.
    #[inline]
    pub fn dec(self, huge: bool) -> Option<Entry3> {
        let sub = Table::span(huge as _);
        if !self.giant()
            && (self.huge() == huge || self.pages() == Table::span(2))
            && self.pages() >= sub
        {
            Some(
                self.with_pages(self.pages() - sub)
                    .with_huge(huge)
                    .with_reserved(true),
            )
        } else {
            None
        }
    }
    /// Increments the free pages counter and clears the size flag if empty.
    #[inline]
    pub fn inc(self, huge: bool, max: usize) -> Option<Entry3> {
        if self.giant() || self.huge() != huge {
            return None;
        }
        let pages = self.pages() + Table::span(huge as _);
        match pages.cmp(&max) {
            Ordering::Greater => None,
            Ordering::Equal => Some(self.with_pages(max).with_huge(false)),
            Ordering::Less => Some(self.with_pages(pages)),
        }
    }
    /// Increments the free pages counter and clears the size flag if empty.
    /// Additionally checks for `idx` to match.
    #[inline]
    pub fn inc_idx(self, huge: bool, idx: usize, max: usize) -> Option<Entry3> {
        if self.giant() || self.huge() != huge || self.idx() != idx {
            return None;
        }
        let pages = self.pages() + Table::span(huge as _);
        match pages.cmp(&max) {
            Ordering::Greater => None,
            Ordering::Equal => Some(self.with_pages(max).with_huge(false)),
            Ordering::Less => Some(self.with_pages(pages)),
        }
    }
    /// Sets the size and reserved bits and the page counter to it's max value.
    #[inline]
    pub fn reserve(self, huge: bool) -> Option<Entry3> {
        if !self.giant()
            && !self.reserved()
            && (self.pages() == Table::span(2) || self.huge() == huge)
            && self.pages() >= Table::span(huge as _)
        {
            Some(self.with_pages(0).with_huge(huge).with_reserved(true))
        } else {
            None
        }
    }
    #[inline]
    pub fn reserve_partial(self, huge: bool, min: usize) -> Option<Entry3> {
        if !self.giant()
            && !self.reserved()
            && self.pages() > min
            && self.pages() < Table::span(2)
            && self.huge() == huge
        {
            Some(self.with_huge(huge).with_reserved(true).with_pages(0))
        } else {
            None
        }
    }
    #[inline]
    pub fn reserve_empty(self, huge: bool) -> Option<Entry3> {
        if !self.giant() && !self.reserved() && self.pages() == Table::span(2) {
            Some(self.with_huge(huge).with_reserved(true).with_pages(0))
        } else {
            None
        }
    }
    #[inline]
    pub fn unreserve(self) -> Option<Entry3> {
        if !self.giant() && self.reserved() {
            Some(self.with_reserved(false).with_idx(0))
        } else {
            None
        }
    }
    /// Add the pages from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    #[inline]
    pub fn unreserve_add(self, huge: bool, other: Entry3, max: usize) -> Option<Entry3> {
        let pages = self.pages() + other.pages();
        if !self.giant()
            && self.reserved()
            && pages <= max
            && (self.huge() == huge || self.pages() == Table::span(2))
            && (other.huge() == huge || other.pages() == Table::span(2))
        {
            Some(
                self.with_pages(pages)
                    .with_huge(huge && pages < max)
                    .with_reserved(false)
                    .with_idx(0),
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
            .field("pages", &self.pages())
            .field("idx", &self.idx())
            .field("reserved", &self.reserved())
            .field("huge", &self.huge())
            .field("giant", &self.giant())
            .finish()
    }
}

#[bitfield(u64)]
pub struct Entry2 {
    #[bits(10)]
    pub pages: usize,
    #[bits(9)]
    pub i1: usize,
    pub giant: bool,
    #[bits(43)]
    _p: u64,
    pub page: bool,
}

impl Entry2 {
    #[inline]
    pub fn new_table(pages: usize, i1: usize) -> Self {
        Self::new().with_pages(pages).with_i1(i1)
    }
    #[inline]
    pub fn mark_huge(self) -> Option<Self> {
        if !self.giant() && !self.page() && self.pages() == Table::span(Size::L1 as _) {
            Some(Entry2::new().with_pages(0).with_i1(0).with_page(true))
        } else {
            None
        }
    }
    #[inline]
    pub fn dec(self, i1: usize) -> Option<Self> {
        if !self.page() && !self.giant() && self.i1() == i1 && self.pages() > 0 {
            Some(self.with_pages(self.pages() - 1))
        } else {
            None
        }
    }
    #[inline]
    pub fn inc(self, i1: usize) -> Option<Self> {
        if !self.giant()
            && !self.page()
            && i1 == self.i1()
            && self.pages() > 0 // is the child pt already initialized?
            && self.pages() < Table::LEN
        {
            Some(self.with_pages(self.pages() + 1))
        } else {
            None
        }
    }
}

impl fmt::Debug for Entry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry2")
            .field("pages", &self.pages())
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
