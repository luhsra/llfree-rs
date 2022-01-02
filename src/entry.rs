use core::fmt;

use bitfield_struct::bitfield;

use crate::table::span;
use crate::{table::PT_LEN, Size};

#[bitfield(u64)]
pub struct Entry {
    pub pages: usize,
}

impl Entry {
    pub fn inc(self, size: Size, layer: usize, max: usize) -> Option<Self> {
        let pages = self.pages() + span(size as _);
        if pages <= span(layer) && pages <= max {
            Some(Entry::new().with_pages(pages))
        } else {
            None
        }
    }
    pub fn dec(self, size: Size) -> Option<Self> {
        if self.pages() >= span(size as _) {
            Some(Entry::new().with_pages(self.pages() - span(size as _)))
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

#[bitfield(u64)]
pub struct Entry3 {
    #[bits(44)]
    pub pages: usize,
    #[bits(2)]
    pub size_n: u8,
    #[bits(18)]
    pub usage: usize,
}

impl Entry3 {
    #[inline(always)]
    pub fn new_giant() -> Entry3 {
        Entry3::new().with_size_n(3)
    }
    #[inline(always)]
    pub fn new_table(pages: usize, size: Size, usage: usize) -> Entry3 {
        Entry3::new()
            .with_pages(pages)
            .with_size_n(size as u8 + 1)
            .with_usage(usage)
    }
    pub fn size(self) -> Option<Size> {
        let s = self.size_n();
        match s {
            1..=3 => Some(unsafe { std::mem::transmute(s - 1) }),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn inc(self, size: Size, max: usize) -> Option<Entry3> {
        if self.size_n() == 3 || (self.pages() != 0 && self.size() != Some(size)) {
            return None;
        }

        let pages = self.pages() + span(size as usize);
        if pages < span(2) && pages < max {
            Some(self.with_pages(pages).with_size_n(size as u8 + 1))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec(self, size: Size) -> Option<Entry3> {
        if self.size_n() == 3 || (self.pages() != 0 && self.size() != Some(size)) {
            return None;
        }

        if self.pages() >= span(size as usize) {
            Some(self.with_pages(self.pages() - span(size as usize)))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn inc_usage(self, size: Size, max: usize) -> Option<Entry3> {
        if self.size_n() == 3 || (self.pages() != 0 && self.size() != Some(size)) {
            return None;
        }

        if self.pages() as usize + span(size as usize) < span(2) && (self.usage() as usize) < max {
            Some(
                self.with_usage(self.usage() + 1)
                    .with_size_n(size as u8 + 1),
            )
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec_usage(self) -> Option<Entry3> {
        if self.size_n() != 3 && self.usage() > 0 {
            Some(self.with_usage(self.usage() - 1))
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
            .field("usage", &self.usage())
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
    pub fn inc(self, i1: usize) -> Option<Self> {
        if !self.page() && !self.giant() && self.i1() == i1 && self.pages() < PT_LEN {
            Some(self.with_pages(self.pages() + 1))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec(self, i1: usize) -> Option<Self> {
        if !self.giant()
            && !self.page()
            && i1 == self.i1()
            && self.pages() > 0
            && self.pages() < PT_LEN
        {
            Some(self.with_pages(self.pages() - 1))
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
