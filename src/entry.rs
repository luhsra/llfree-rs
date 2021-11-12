use core::fmt;

use static_assertions::const_assert;

use crate::table::{LAYERS, PTE_SIZE, PT_LEN, PT_LEN_BITS};

const PTE_PAGES_OFF: usize = 0;
const PTE_PAGES_BITS: usize = (PT_LEN_BITS + 1) * LAYERS;
const PTE_PAGES_MASK: u64 = ((1 << PTE_PAGES_BITS) - 1) << PTE_PAGES_OFF;

const PTE_NONEMPTY_OFF: usize = PTE_PAGES_OFF + PTE_PAGES_BITS;
const PTE_NONEMPTY_BITS: usize = PT_LEN_BITS + 1;
const PTE_NONEMPTY_MASK: u64 = ((1 << PTE_NONEMPTY_BITS) - 1) << PTE_NONEMPTY_OFF;

const PTE_PAGE_OFF: usize = PTE_NONEMPTY_OFF + PTE_NONEMPTY_BITS;
const PTE_PAGE: u64 = 1 << PTE_PAGE_OFF;

const PTE_RESERVED_OFF: usize = PTE_PAGE_OFF + 1;
const PTE_RESERVED: u64 = 1 << PTE_RESERVED_OFF;

const_assert!(PTE_RESERVED_OFF < PTE_SIZE * 8);

/// PTE states:
/// - `pages`: num of allocated 4KiB pages
/// - `nonempty`: num of nonempty entries
/// - `reserved`: Currently reserved by another thread...
///
/// ```text
/// [ .. | reserved | page | nonempty | pages ]
///           1        1        10       40
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
        Entry(PTE_PAGE)
    }
    #[inline(always)]
    pub fn table(pages: usize, nonempty: usize, reserved: bool) -> Entry {
        Entry(
            (((pages as u64) << PTE_PAGES_OFF) & PTE_PAGES_MASK)
                | (((nonempty as u64) << PTE_NONEMPTY_OFF) & PTE_NONEMPTY_MASK)
                | ((reserved as u64) << PTE_RESERVED_OFF),
        )
    }
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.0 & (PTE_PAGE | PTE_NONEMPTY_MASK | PTE_PAGES_MASK) == 0
    }
    #[inline(always)]
    pub fn is_page(self) -> bool {
        self.0 & PTE_PAGE != 0
    }
    #[inline(always)]
    pub fn is_reserved(self) -> bool {
        self.0 & PTE_RESERVED != 0
    }
    #[inline(always)]
    pub fn pages(self) -> usize {
        ((self.0 & PTE_PAGES_MASK) >> PTE_PAGES_OFF) as _
    }
    #[inline(always)]
    pub fn nonempty(self) -> usize {
        ((self.0 & PTE_NONEMPTY_MASK) >> PTE_NONEMPTY_OFF) as _
    }
}

impl From<u64> for Entry {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl Into<u64> for Entry {
    fn into(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ r={} | p={} | nonempty={} | pages={} }}",
            self.is_reserved(),
            self.is_page(),
            self.nonempty(),
            self.pages()
        )
    }
}

const PTE2_PAGES_OFF: usize = 0;
const PTE2_PAGES_BITS: usize = PT_LEN_BITS + 1;
const PTE2_PAGES_MASK: u64 = ((1 << PTE2_PAGES_BITS) - 1) << PTE2_PAGES_OFF;

const PTE2_USAGE_OFF: usize = PTE2_PAGES_OFF + PTE2_PAGES_BITS;
const PTE2_USAGE_BITS: usize = PT_LEN_BITS + 1;
const PTE2_USAGE_MASK: u64 = ((1 << PTE2_USAGE_BITS) - 1) << PTE2_USAGE_OFF;

const PTE2_I1_OFF: usize = PTE2_USAGE_OFF + PTE2_USAGE_BITS;
const PTE2_I1_BITS: usize = PT_LEN_BITS;
const PTE2_I1_MASK: u64 = ((1 << PTE2_I1_BITS) - 1) << PTE2_I1_OFF;

const PTE2_PAGE_OFF: usize = PTE2_I1_OFF + PTE2_I1_BITS;
const PTE2_PAGE: u64 = 1 << PTE2_PAGE_OFF;

const PTE2_RESERVED_OFF: usize = PTE2_PAGE_OFF + 1;
const PTE2_RESERVED: u64 = 1 << PTE2_RESERVED_OFF;

const PTE2_HUGE_OFF: usize = PTE2_PAGE_OFF + 1;
const PTE2_HUGE: u64 = 1 << PTE2_HUGE_OFF;

const_assert!(PTE2_HUGE_OFF < PTE_SIZE * 8);

/// Level 2 page table entry:
/// - `pages`: num of allocated 4KiB pages
/// - `usage`: num of concurrently running frees
/// - `i1`: Index of the page where the pt1 is stored
/// - `page`: Is page allocated
/// - `reserved`: Currently reserved (prevent parallel init of pt1)
/// - `huge`: is allocated as huge page (1G)
///
/// ```text
/// [ .. | huge | reserved | page | i1 | usage | pages ]
///         1        1        1      9     10      10
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct L2Entry(pub u64);

impl L2Entry {
    #[inline(always)]
    pub fn empty() -> Self {
        Self(0)
    }
    #[inline(always)]
    pub fn huge() -> Self {
        Self(PTE2_HUGE)
    }
    #[inline(always)]
    pub fn page() -> Self {
        Self(PTE2_PAGE)
    }
    #[inline(always)]
    pub fn page_reserved() -> Self {
        Self(PTE2_PAGE | PTE2_RESERVED)
    }
    #[inline(always)]
    pub fn table(pages: usize, usage: usize, i1: usize, reserved: bool) -> Self {
        Self(
            ((pages << PTE2_PAGES_OFF) as u64 & PTE2_PAGES_MASK)
                | ((usage << PTE2_USAGE_OFF) as u64 & PTE2_USAGE_MASK)
                | ((i1 << PTE2_I1_OFF) as u64 & PTE2_I1_MASK)
                | ((reserved as u64) << PTE2_RESERVED_OFF),
        )
    }
    #[inline(always)]
    pub fn inc_usage(self) -> Option<Self> {
        if self.has_i1() {
            Some(L2Entry::table(
                self.pages(),
                self.usage() + 1,
                self.i1(),
                self.is_reserved(),
            ))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec_usage(self) -> Option<Self> {
        if !self.is_page() && !self.is_huge() && !self.is_reserved() {
            Some(L2Entry::table(
                self.pages(),
                self.usage() + 1,
                self.i1(),
                self.is_reserved(),
            ))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn dec_all(self, i1: usize) -> Option<Self> {
        if !self.is_huge()
            && !self.is_page()
            && !self.is_reserved()
            && i1 == self.i1()
            && self.pages() > 0
            && self.usage() > 0
        {
            Some(L2Entry::table(
                self.pages() - 1,
                self.usage() - 1,
                self.i1(),
                self.is_reserved(),
            ))
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.0 & (PTE2_HUGE | PTE2_PAGE | PTE2_PAGES_MASK) == 0
    }
    #[inline(always)]
    pub fn is_page(self) -> bool {
        self.0 & PTE2_PAGE != 0
    }
    #[inline(always)]
    pub fn is_huge(self) -> bool {
        self.0 & PTE2_HUGE != 0
    }
    #[inline(always)]
    pub fn is_reserved(self) -> bool {
        self.0 & PTE2_RESERVED != 0
    }
    #[inline(always)]
    pub fn has_i1(self) -> bool {
        self.0 & (PTE2_RESERVED | PTE2_HUGE | PTE2_PAGE) == 0
            && 0 < self.pages()
            && self.pages() < PT_LEN
    }
    #[inline(always)]
    pub fn pages(self) -> usize {
        ((self.0 & PTE2_PAGES_MASK) >> PTE2_PAGES_OFF) as _
    }
    #[inline(always)]
    pub fn usage(self) -> usize {
        ((self.0 & PTE2_USAGE_MASK) >> PTE2_USAGE_OFF) as _
    }
    #[inline(always)]
    pub fn i1(self) -> usize {
        ((self.0 & PTE2_I1_MASK) >> PTE2_I1_OFF) as _
    }
}

impl From<u64> for L2Entry {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl Into<u64> for L2Entry {
    fn into(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for L2Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ h={} | r={} | p={} | i1={} | usage={} | pages={} }}",
            self.is_huge(),
            self.is_reserved(),
            self.is_page(),
            self.i1(),
            self.pages(),
            self.usage()
        )
    }
}

const PTE1_PAGE: u64 = 0b01;
const PTE1_RESERVED: u64 = 0b10;

/// Level 1 page table entry:
/// - `page`: Is page allocated
/// - `reserved`: Is reserved for the pt1
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct L1Entry(pub u64);

impl L1Entry {
    #[inline(always)]
    pub fn empty() -> Self {
        Self(0)
    }
    #[inline(always)]
    pub fn page() -> Self {
        Self(PTE1_PAGE)
    }
    #[inline(always)]
    pub fn reserved() -> Self {
        Self(PTE1_RESERVED)
    }
}

impl From<u64> for L1Entry {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl Into<u64> for L1Entry {
    fn into(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for L1Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ r={} | p={} }}",
            self == &Self::reserved(),
            self == &Self::page()
        )
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::AtomicU64;

    use crate::table::Table;

    #[test]
    fn pt() {
        let pt: Table<u64> = Table::empty();
        pt.cas(0, 0, 42).unwrap();
        pt.update(0, |v| Some(v + 1)).unwrap();
        assert_eq!(pt.get(0), 43);
    }
}
