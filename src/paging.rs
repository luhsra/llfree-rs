use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use static_assertions::{const_assert, const_assert_eq};

pub const PAGE_SIZE_BITS: usize = 12; // 2^12 => 4KiB
pub const PAGE_SIZE: usize = 1 << PAGE_SIZE_BITS;
pub const PTE_SIZE_BITS: usize = 3; // 2^3 => 8B => 64b
pub const PTE_SIZE: usize = 1 << PTE_SIZE_BITS;
pub const PT_LEN_BITS: usize = PAGE_SIZE_BITS - PTE_SIZE_BITS;
pub const PT_LEN: usize = 1 << PT_LEN_BITS;

pub const LAYERS: usize = 4;

/// Page table with atomic entries
#[repr(align(32))]
pub struct Table {
    entries: [AtomicU64; PT_LEN],
}

const_assert_eq!(size_of::<AtomicU64>(), PTE_SIZE);
const_assert_eq!(size_of::<Table>(), PAGE_SIZE);
const_assert_eq!(size_of::<usize>(), size_of::<u64>());

impl Table {
    /// Area in bytes that a page table covers
    #[inline(always)]
    pub const fn span(layer: usize) -> usize {
        Self::p_span(layer) << PAGE_SIZE_BITS
    }

    /// Area in pages that a page table covers
    #[inline(always)]
    pub const fn p_span(layer: usize) -> usize {
        1 << (PT_LEN_BITS * layer)
    }

    /// Returns pt index that contains the `page`
    #[inline(always)]
    pub fn p_idx(layer: usize, page: usize) -> usize {
        (page >> (PT_LEN_BITS * (layer - 1))) & (PT_LEN - 1)
    }

    pub fn get(&self, i: usize) -> Entry {
        Entry(self.entries[i].load(Ordering::SeqCst))
    }

    pub fn set(&self, i: usize, e: Entry) {
        self.entries[i].store(e.0, Ordering::SeqCst);
    }

    pub fn clear(&self) {
        for i in 0..PT_LEN {
            self.entries[i].store(0, Ordering::SeqCst);
        }
    }

    pub fn cas(&self, i: usize, expected: Entry, new: Entry) -> Result<Entry, Entry> {
        match self.entries[i].compare_exchange(
            expected.0,
            new.0,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(v) => Ok(Entry(v)),
            Err(v) => Err(Entry(v)),
        }
    }

    pub fn insert_page(&self, i: usize) -> Result<Entry, Entry> {
        match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            if Entry(v).is_empty() {
                Some(Entry::page().0)
            } else {
                None
            }
        }) {
            Ok(v) => Ok(Entry(v)),
            Err(v) => Err(Entry(v)),
        }
    }

    pub fn reserve(&self, i: usize, reserved: bool) -> Result<Entry, Entry> {
        match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            let pte = Entry(v);
            if pte.is_reserved() == reserved {
                return None;
            } else {
                if pte.is_page() {
                    return Some(Entry::page_reserved().0);
                } else {
                    return Some(Entry::table(pte.pages(), pte.nonempty(), pte.i1(), reserved).0);
                }
            }
        }) {
            Ok(v) => Ok(Entry(v)),
            Err(v) => Err(Entry(v)),
        }
    }

    pub fn inc(&self, i: usize, pages: usize, nonempty: usize) -> Result<Entry, Entry> {
        match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            let pte = Entry(v);
            if !pte.is_table() && !pte.is_empty() {
                return None;
            }
            let pages = pte.pages() + pages;
            let nonempty = pte.nonempty() + nonempty;
            if pages < nonempty {
                return None;
            }
            if pages < PTE_PAGES_MASK as _ && nonempty < PTE_NONEMPTY_MASK as _ {
                Some(Entry::table(pages, nonempty, pte.i1(), pte.is_reserved()).0)
            } else {
                None
            }
        }) {
            Ok(v) => Ok(Entry(v)),
            Err(v) => Err(Entry(v)),
        }
    }
    pub fn dec(&self, i: usize, pages: usize, nonempty: usize) -> Result<Entry, Entry> {
        match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            let pte = Entry(v);
            if !pte.is_table() {
                return None;
            }

            if pte.pages() >= pages && pte.nonempty() >= nonempty {
                let pages = pte.pages() - pages;
                let nonempty = pte.nonempty() - nonempty;
                if pages < nonempty {
                    return None;
                }
                Some(Entry::table(pages, nonempty, pte.i1(), pte.is_reserved()).0)
            } else {
                None
            }
        }) {
            Ok(v) => Ok(Entry(v)),
            Err(v) => Err(Entry(v)),
        }
    }
}

const PTE_PAGES_OFF: usize = 0;
const PTE_PAGES_BITS: usize = (PT_LEN_BITS + 1) * LAYERS;
const PTE_PAGES_MASK: u64 = ((1 << PTE_PAGES_BITS) - 1) << PTE_PAGES_OFF;

const PTE_NONEMPTY_OFF: usize = PTE_PAGES_OFF + PTE_PAGES_BITS;
const PTE_NONEMPTY_BITS: usize = PT_LEN_BITS + 1;
const PTE_NONEMPTY_MASK: u64 = ((1 << PTE_NONEMPTY_BITS) - 1) << PTE_NONEMPTY_OFF;

const PTE_I1_OFF: usize = PTE_NONEMPTY_OFF + PTE_NONEMPTY_BITS;
const PTE_I1_BITS: usize = PT_LEN_BITS;
const PTE_I1_MASK: u64 = ((1 << PTE_I1_BITS) - 1) << PTE_I1_OFF;

const PTE_TABLE_OFF: usize = PTE_I1_OFF + PTE_I1_BITS;
const PTE_TABLE: u64 = 1 << PTE_TABLE_OFF;

const PTE_RESERVED_OFF: usize = PTE_TABLE_OFF + 1;
const PTE_RESERVED: u64 = 1 << PTE_RESERVED_OFF;

const_assert!(PTE_RESERVED_OFF < PTE_SIZE * 8);

/// PTE states:
/// - `pages`: num of allocated 4KiB pages
/// - `nonempty`: num of nonempty entries
/// - (PT2) `pt1idx`: Index of the page where the pt1 is stored
/// - `table`: Has child table
/// - (optional) `reserved`: Currently reserved by another thread...
///
/// ```text
/// [ .. | reserved | table | i1 | nonempty | pages ]
///    3       1        1      9      10       40
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Entry(pub u64);

impl Entry {
    #[inline(always)]
    pub fn empty() -> Entry {
        Entry(0)
    }
    #[inline(always)]
    pub fn page() -> Entry {
        Entry(PTE_PAGES_MASK | PTE_NONEMPTY_MASK)
    }
    #[inline(always)]
    pub fn page_reserved() -> Entry {
        Entry(PTE_PAGES_MASK | PTE_NONEMPTY_MASK | PTE_RESERVED)
    }
    #[inline(always)]
    pub fn table(pages: usize, nonempty: usize, i1: usize, reserved: bool) -> Entry {
        Entry(
            PTE_TABLE
                | (((pages as u64) << PTE_PAGES_OFF) & PTE_PAGES_MASK)
                | (((nonempty as u64) << PTE_NONEMPTY_OFF) & PTE_NONEMPTY_MASK)
                | (((i1 as u64) << PTE_I1_OFF) & PTE_I1_MASK)
                | ((reserved as u64) << PTE_RESERVED_OFF),
        )
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        !self.is_table() && self.pages() == 0
    }

    #[inline(always)]
    pub fn is_table(self) -> bool {
        self.0 & PTE_TABLE != 0
    }

    #[inline(always)]
    pub fn is_page(self) -> bool {
        !self.is_table() && self.pages() != 0
    }

    #[inline(always)]
    pub fn pages(self) -> usize {
        ((self.0 & PTE_PAGES_MASK) >> PTE_PAGES_OFF) as _
    }

    #[inline(always)]
    pub fn nonempty(self) -> usize {
        ((self.0 & PTE_NONEMPTY_MASK) >> PTE_NONEMPTY_OFF) as _
    }

    #[inline(always)]
    pub fn i1(self) -> usize {
        ((self.0 & PTE_I1_MASK) >> PTE_I1_OFF) as _
    }

    #[inline(always)]
    pub fn is_reserved(self) -> bool {
        self.0 & PTE_RESERVED != 0
    }
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[ .. | {} | {} | {} | {} ]",
            self.is_reserved(),
            self.i1(),
            self.nonempty(),
            self.pages()
        )
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::AtomicU64;

    use crate::paging::{Entry, Table, PAGE_SIZE, PT_LEN};

    #[test]
    fn pt_size() {
        assert_eq!(Table::span(0), PAGE_SIZE);
        assert_eq!(Table::span(1), PAGE_SIZE * PT_LEN);
        assert_eq!(Table::span(2), PAGE_SIZE * PT_LEN * PT_LEN);

        assert_eq!(Table::p_span(0), 1);
        assert_eq!(Table::p_span(1), PT_LEN);
        assert_eq!(Table::p_span(2), PT_LEN * PT_LEN);
    }

    #[test]
    fn pt() {
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let pt = Table {
            entries: [DEFAULT; PT_LEN],
        };
        pt.cas(0, Entry::empty(), Entry::table(0, 0, 0, true))
            .unwrap();
        pt.inc(0, 42, 1).unwrap();
        assert_eq!(pt.get(0), Entry::table(42, 1, 0, true));
    }
}
