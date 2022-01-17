use core::fmt;
use std::cmp::Ordering;

use bitfield_struct::bitfield;

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

impl Entry {
    pub fn reserve_partial(self, size: Size) -> Option<Self> {
        match size {
            Size::L0 if self.partial_l0() > 0 => Some(self.with_partial_l0(self.partial_l0() - 1)),
            Size::L1 if self.partial_l1() > 0 => Some(self.with_partial_l1(self.partial_l1() - 1)),
            _ => None,
        }
    }
    pub fn dec_empty(self) -> Option<Self> {
        if self.empty() > 0 {
            Some(self.with_empty(self.empty() - 1))
        } else {
            None
        }
    }
    pub fn inc(self, dec: Change) -> Option<Self> {
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
    #[bits(2)]
    pub size_n: u8,
    pub reserved: bool,
    #[bits(41)]
    pub idx: usize,
}

impl Entry3 {
    #[inline(always)]
    pub fn new_giant() -> Entry3 {
        Entry3::new().with_size_n(3)
    }
    #[inline(always)]
    pub fn new_table(pages: usize, size: Size, reserved: bool) -> Entry3 {
        Entry3::new()
            .with_pages(pages)
            .with_size_n(size as u8 + 1)
            .with_reserved(reserved)
    }
    pub fn size(self) -> Option<Size> {
        let s = self.size_n();
        match s {
            1..=3 => Some(unsafe { std::mem::transmute(s - 1) }),
            _ => None,
        }
    }
    /// Decrements the free pages counter and sets the size and reserved bits.
    #[inline(always)]
    pub fn dec(self, size: Size) -> Option<Entry3> {
        if self.size_n() != 3
            && (self.size() == None || self.size() == Some(size))
            && self.pages() >= Table::span(size as _)
        {
            Some(
                self.with_pages(self.pages() - Table::span(size as usize))
                    .with_size_n(size as u8 + 1)
                    .with_reserved(true),
            )
        } else {
            None
        }
    }
    /// Increments the free pages counter and clears the size flag if empty.
    #[inline(always)]
    pub fn inc(self, size: Size, max: usize) -> Option<Entry3> {
        if self.size_n() == 3 || self.size() != Some(size) {
            return None;
        }
        let pages = self.pages() + Table::span(size as _);
        match pages.cmp(&max) {
            Ordering::Greater => None,
            Ordering::Equal => Some(self.with_pages(max).with_size_n(0)),
            Ordering::Less => Some(self.with_pages(pages)),
        }
    }
    /// Increments the free pages counter and clears the size flag if empty.
    /// Additionally checks for `idx` to match.
    #[inline(always)]
    pub fn inc_idx(self, size: Size, idx: usize, max: usize) -> Option<Entry3> {
        if self.size_n() == 3 || self.size() != Some(size) || self.idx() != idx {
            return None;
        }

        let pages = self.pages() + Table::span(size as _);
        match pages.cmp(&max) {
            Ordering::Greater => None,
            Ordering::Equal => Some(self.with_pages(max).with_size_n(0)),
            Ordering::Less => Some(self.with_pages(pages)),
        }
    }
    /// Sets the size and reserved bits and the page counter to it's max value.
    #[inline(always)]
    pub fn reserve_take(self, size: Size) -> Option<Entry3> {
        if self.size_n() != 3
            && (self.size() == None || self.size() == Some(size))
            && self.pages() >= Table::span(size as _)
        {
            Some(
                self.with_pages(0)
                    .with_size_n(size as u8 + 1)
                    .with_reserved(true),
            )
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn reserve_partial(self, size: Size) -> Option<Entry3> {
        if !self.reserved()
            && self.pages() >= Table::span(size as _)
            && self.pages() < Table::span(2)
            && (self.size() == None || self.size() == Some(size))
        {
            Some(
                self.with_size_n(size as u8 + 1)
                    .with_reserved(true)
                    .with_pages(0),
            )
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn reserve_empty(self, size: Size) -> Option<Entry3> {
        if !self.reserved()
            && self.pages() >= Table::span(size as _)
            && self.pages() == Table::span(2)
            && self.size_n() != 3
        {
            Some(
                self.with_size_n(size as u8 + 1)
                    .with_reserved(true)
                    .with_pages(0),
            )
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn unreserve(self) -> Option<Entry3> {
        if self.size_n() != 3 && self.reserved() {
            Some(self.with_reserved(false).with_idx(0))
        } else {
            None
        }
    }
    /// Clear reserve flag and own free pages from `other`.
    #[inline(always)]
    pub fn unreserve_add(self, other: Entry3, max: usize) -> Option<Entry3> {
        let pages = self.pages() + other.pages();
        if self.reserved() && self.size_n() != 3 && self.size_n() == other.size_n() && pages <= max
        {
            Some(self.with_pages(pages).with_reserved(false).with_idx(0))
        } else {
            None
        }
    }
}

impl fmt::Debug for Entry3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry3")
            .field("pages", &self.pages())
            .field("size", &self.size())
            .field("reserved", &self.reserved())
            .field("idx", &self.idx())
            .finish()
    }
}

#[bitfield(u64)]
pub struct Entry2 {
    #[bits(10)]
    pub pages: usize,
    #[bits(9)]
    pub i1: usize,
    pub page: bool,
    pub giant: bool,
    #[bits(43)]
    _p: u64,
}

impl Entry2 {
    #[inline(always)]
    pub fn new_table(pages: usize, i1: usize) -> Self {
        Self::new().with_pages(pages).with_i1(i1)
    }
    #[inline(always)]
    pub fn mark_huge(self) -> Option<Self> {
        if !self.giant() && !self.page() && self.pages() == Table::span(Size::L1 as _) {
            Some(Entry2::new().with_pages(0).with_i1(0).with_page(true))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec(self, i1: usize) -> Option<Self> {
        if !self.page() && !self.giant() && self.i1() == i1 && self.pages() > 0 {
            Some(self.with_pages(self.pages() - 1))
        } else {
            None
        }
    }
    #[inline(always)]
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
        assert!(v <= Self::Page as u64);
        unsafe { std::mem::transmute(v) }
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
