use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
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
#[repr(align(128))]
pub struct Table<T> {
    entries: [AtomicU64; PT_LEN],
    phantom: PhantomData<T>,
}

const_assert_eq!(size_of::<AtomicU64>(), PTE_SIZE);
const_assert_eq!(size_of::<Table<u64>>(), PAGE_SIZE);
const_assert_eq!(size_of::<usize>(), size_of::<u64>());

/// Area in bytes that a page table covers
#[inline(always)]
pub const fn m_span(layer: usize) -> usize {
    span(layer) << PAGE_SIZE_BITS
}

/// Area in pages that a page table covers
#[inline(always)]
pub const fn span(layer: usize) -> usize {
    1 << (PT_LEN_BITS * layer)
}

/// Returns pt index that contains the `page`
#[inline(always)]
pub const fn idx(layer: usize, page: usize) -> usize {
    (page >> (PT_LEN_BITS * (layer - 1))) & (PT_LEN - 1)
}

/// Computes the index range for the given page range
#[inline(always)]
pub fn range(layer: usize, pages: Range<usize>) -> Range<usize> {
    let bits = PT_LEN_BITS * (layer - 1);
    let start = pages.start >> bits;
    let end = (pages.end >> bits) + (pages.end.trailing_zeros() < bits as _) as usize;

    let end = (end.saturating_sub(start & !(PT_LEN - 1))).min(PT_LEN);
    let start = start & (PT_LEN - 1);

    start..end
}

impl<T: Sized + From<u64> + Into<u64>> Table<T> {
    #[cfg(test)]
    pub fn empty() -> Self {
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        Self {
            entries: [DEFAULT; PT_LEN],
            phantom: PhantomData,
        }
    }

    pub fn get(&self, i: usize) -> T {
        T::from(self.entries[i].load(Ordering::SeqCst))
    }

    pub fn set(&self, i: usize, e: T) {
        self.entries[i].store(e.into(), Ordering::SeqCst);
    }

    pub fn clear(&self) {
        for i in 0..PT_LEN {
            self.entries[i].store(0, Ordering::SeqCst);
        }
    }

    pub fn cas(&self, i: usize, expected: T, new: T) -> Result<T, T> {
        match self.entries[i].compare_exchange(
            expected.into(),
            new.into(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(v) => Ok(T::from(v)),
            Err(v) => Err(T::from(v)),
        }
    }

    pub fn update<F: FnMut(T) -> Option<T>>(&self, i: usize, mut f: F) -> Result<T, T> {
        match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            f(T::from(v)).map(T::into)
        }) {
            Ok(v) => Ok(T::from(v)),
            Err(v) => Err(T::from(v)),
        }
    }

    // pub fn insert_page(&self, i: usize) -> Result<Entry, Entry> {
    //     match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
    //         if Entry(v).is_empty() {
    //             Some(Entry::page().0)
    //         } else {
    //             None
    //         }
    //     }) {
    //         Ok(v) => Ok(Entry(v)),
    //         Err(v) => Err(Entry(v)),
    //     }
    // }

    // pub fn reserve(&self, i: usize, reserved: bool) -> Result<Entry, Entry> {
    //     match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
    //         let pte = Entry(v);
    //         if pte.is_reserved() == reserved {
    //             None
    //         } else if pte.is_page() {
    //             Some(Entry::page_reserved().0)
    //         } else {
    //             Some(Entry::table(pte.pages(), pte.nonempty(), pte.i1(), reserved).0)
    //         }
    //     }) {
    //         Ok(v) => Ok(Entry(v)),
    //         Err(v) => Err(Entry(v)),
    //     }
    // }

    // pub fn inc(&self, i: usize, pages: usize, nonempty: usize, i1: usize) -> Result<Entry, Entry> {
    //     match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
    //         let pte = Entry(v);
    //         if !pte.is_table() || i1 != pte.i1() {
    //             return None;
    //         }
    //         let pages = pte.pages() + pages;
    //         let nonempty = pte.nonempty() + nonempty;
    //         if pages < nonempty {
    //             return None;
    //         }
    //         if pages < PTE_PAGES_MASK as _ && nonempty < PTE_NONEMPTY_MASK as _ {
    //             Some(Entry::table(pages, nonempty, pte.i1(), pte.is_reserved()).0)
    //         } else {
    //             None
    //         }
    //     }) {
    //         Ok(v) => Ok(Entry(v)),
    //         Err(v) => Err(Entry(v)),
    //     }
    // }

    // pub fn dec(&self, i: usize, pages: usize, nonempty: usize, i1: usize) -> Result<Entry, Entry> {
    //     match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
    //         let pte = Entry(v);
    //         if !pte.is_table() || i1 != pte.i1() {
    //             return None;
    //         }

    //         if pte.pages() >= pages && pte.nonempty() >= nonempty {
    //             let pages = pte.pages() - pages;
    //             let nonempty = pte.nonempty() - nonempty;
    //             if pages < nonempty {
    //                 return None;
    //             }
    //             Some(Entry::table(pages, nonempty, pte.i1(), pte.is_reserved()).0)
    //         } else {
    //             None
    //         }
    //     }) {
    //         Ok(v) => Ok(Entry(v)),
    //         Err(v) => Err(Entry(v)),
    //     }
    // }

    // pub fn dec_nofull(&self, i: usize, pages: usize, nonempty: usize) -> Result<Entry, Entry> {
    //     match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
    //         let pte = Entry(v);
    //         if !pte.is_table() || pte.nonempty() >= PT_LEN {
    //             return None;
    //         }

    //         if pte.pages() >= pages && pte.nonempty() >= nonempty {
    //             let pages = pte.pages() - pages;
    //             let nonempty = pte.nonempty() - nonempty;
    //             if pages < nonempty {
    //                 return None;
    //             }
    //             Some(Entry::table(pages, nonempty, pte.i1(), pte.is_reserved()).0)
    //         } else {
    //             None
    //         }
    //     }) {
    //         Ok(v) => Ok(Entry(v)),
    //         Err(v) => Err(Entry(v)),
    //     }
    // }
}

#[cfg(test)]
mod test {
    use crate::table::{self, PAGE_SIZE, PT_LEN};

    #[test]
    fn pt_size() {
        assert_eq!(table::m_span(0), PAGE_SIZE);
        assert_eq!(table::m_span(1), PAGE_SIZE * PT_LEN);
        assert_eq!(table::m_span(2), PAGE_SIZE * PT_LEN * PT_LEN);

        assert_eq!(table::span(0), 1);
        assert_eq!(table::span(1), PT_LEN);
        assert_eq!(table::span(2), PT_LEN * PT_LEN);
    }

    #[test]
    fn indexing() {
        assert_eq!(table::range(1, 0..PT_LEN), 0..PT_LEN);
        assert_eq!(table::range(1, 0..0), 0..0);
        assert_eq!(table::range(1, 0..PT_LEN + 1), 0..PT_LEN);
        assert_eq!(table::range(1, PT_LEN..PT_LEN - 1), 0..0);

        // L2
        assert_eq!(table::range(2, 0..table::span(1)), 0..1);
        assert_eq!(table::range(2, table::span(1)..3 * table::span(1)), 1..3);
        assert_eq!(table::range(2, 0..table::span(2)), 0..PT_LEN);

        // L3
        assert_eq!(table::range(3, 0..table::span(2)), 0..1);
        assert_eq!(table::range(3, table::span(2)..3 * table::span(2)), 1..3);
        assert_eq!(table::range(3, 0..table::span(3)), 0..PT_LEN);

        assert_eq!(table::range(3, 0..1), 0..1);
    }
}
