use core::fmt;
use std::sync::atomic::AtomicU64;
use std::{convert::TryInto, mem::size_of};

use modular_bitfield::{bitfield, prelude::*};
use static_assertions::{const_assert, const_assert_eq};

use crate::table::span;
use crate::{
    table::{self, LAYERS, PTE_SIZE, PT_LEN, PT_LEN_BITS},
    Size,
};

#[bitfield(bits = 64)]
pub struct Entry {
    pub pages: B64,
}

impl Entry {
    pub fn inc(self, size: Size, layer: usize, max: usize) -> Option<Self> {
        let pages = self.pages() + span(size as _) as u64;
        if pages <= span(layer) as u64 && pages <= max as u64 {
            Some(Entry::new().with_pages(pages))
        } else {
            None
        }
    }
    pub fn dec(self, size: Size) -> Option<Self> {
        if self.pages() >= span(size as _) as u64 {
            Some(Entry::new().with_pages(self.pages() - span(size as _) as u64))
        } else {
            None
        }
    }
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry")
            .field("pages", &self.pages())
            .finish()
    }
}

#[derive(BitfieldSpecifier, PartialEq, Eq, Debug)]
#[bits = 2]
pub enum ESize {
    None = 0,
    L0 = 1,
    L1 = 2,
    L2 = 3,
}

impl From<Size> for ESize {
    fn from(s: Size) -> Self {
        match s {
            Size::L0 => Self::L0,
            Size::L1 => Self::L1,
            Size::L2 => Self::L2,
        }
    }
}

impl TryInto<Size> for ESize {
    type Error = ();
    fn try_into(self) -> Result<Size, Self::Error> {
        match self {
            ESize::None => Err(()),
            ESize::L0 => Ok(Size::L0),
            ESize::L1 => Ok(Size::L1),
            ESize::L2 => Ok(Size::L2),
        }
    }
}

impl PartialEq<Size> for ESize {
    fn eq(&self, other: &Size) -> bool {
        Self::from(*other) == *self
    }
}

#[bitfield(bits = 64)]
pub struct Entry3 {
    pub pages: B44,
    pub size: ESize,
    pub usage: B18,
}

impl Entry3 {
    #[inline(always)]
    pub fn new_giant() -> Entry3 {
        Entry3::new().with_size(ESize::L2)
    }
    #[inline(always)]
    pub fn new_table(pages: usize, size: Size, usage: usize) -> Entry3 {
        Entry3::new()
            .with_pages(pages as _)
            .with_size(size.into())
            .with_usage(usage as _)
    }
    #[inline(always)]
    pub fn inc(self, size: Size, max: usize) -> Option<Entry3> {
        if self.size() == ESize::L2 || (self.pages() != 0 && self.size() != ESize::from(size)) {
            return None;
        }

        let pages = self.pages() + span(size as usize) as u64;
        if pages < span(2) as u64 && pages < max as u64 {
            Some(self.with_pages(pages))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec(self, size: Size) -> Option<Entry3> {
        if self.size() == ESize::L2 || (self.pages() != 0 && self.size() != ESize::from(size)) {
            return None;
        }

        if self.pages() >= span(size as usize) as u64 {
            Some(self.with_pages(self.pages() - span(size as usize) as u64))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn inc_usage(self, size: Size, max: usize) -> Option<Entry3> {
        if self.size() == ESize::L2 || (self.pages() != 0 && self.size() != ESize::from(size)) {
            return None;
        }

        if self.pages() as usize + span(size as usize) < span(2) && (self.usage() as usize) < max {
            Some(self.with_usage(self.usage() + 1))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec_usage(self) -> Option<Entry3> {
        if self.size() != ESize::L2 && self.usage() > 0 {
            Some(self.with_usage(self.usage() - 1))
        } else {
            None
        }
    }
}

impl From<u64> for Entry3 {
    fn from(v: u64) -> Self {
        Self::from_bytes(v.to_le_bytes())
    }
}

impl Into<u64> for Entry3 {
    fn into(self) -> u64 {
        u64::from_le_bytes(self.into_bytes())
    }
}

impl fmt::Debug for Entry3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry3")
            .field("pages", &self.pages())
            .field("size", &self.size())
            .field("usage", &self.usage())
            .finish()
    }
}

#[bitfield(bits = 64, filled = false)]
pub struct Entry2 {
    pub pages: B10,
    pub i1: B9,
    pub page: bool,
    pub giant: bool,
}

impl Entry2 {
    #[inline(always)]
    pub fn new_table(pages: usize, i1: usize) -> Self {
        Self::new().with_pages(pages as _).with_i1(i1 as _)
    }
    #[inline(always)]
    pub fn inc(self, i1: usize) -> Option<Self> {
        if !self.page()
            && !self.giant()
            && self.i1() as usize == i1
            && (self.pages() as usize) < PT_LEN
        {
            Some(self.with_pages(pages + 1))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec(self, i1: usize) -> Option<Self> {
        if !self.giant()
            && !self.page()
            && i1 == self.i1() as usize
            && self.pages() > 0
            && (self.pages() as usize) < PT_LEN
        {
            Some(self.with_pages(self.pages() - 1))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        !self.giant() && !self.page() && self.pages() == 0
    }
}

impl From<u64> for Entry2 {
    fn from(v: u64) -> Self {
        Self::from_bytes(v.to_le_bytes()).unwrap()
    }
}

impl Into<u64> for Entry2 {
    fn into(self) -> u64 {
        u64::from_le_bytes(self.into_bytes())
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
enum Entry1 {
    Empty = 0,
    Page = 1,
    Reserved = 2,
}

impl From<u64> for Entry1 {
    fn from(v: u64) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

impl Into<u64> for Entry1 {
    fn into(self) -> u64 {
        self as u64
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
