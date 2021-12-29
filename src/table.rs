use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

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

const _: () = assert!(size_of::<AtomicU64>() == PTE_SIZE);
const _: () = assert!(size_of::<Table<u64>>() == PAGE_SIZE);
const _: () = assert!(size_of::<usize>() == size_of::<u64>());

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

/// Returns the starting page of the corresponding page table
#[inline(always)]
pub const fn round(layer: usize, page: usize) -> usize {
    page & !((1 << (PT_LEN_BITS * layer)) - 1)
}

/// Returns the page at the given index `i`
#[inline(always)]
pub const fn page(layer: usize, start: usize, i: usize) -> usize {
    round(layer, start) + i * span(layer - 1)
}

pub const fn num_pts(layer: usize, pages: usize) -> usize {
    (pages + span(layer) - 1) / span(layer)
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
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }

    pub fn update<F: FnMut(T) -> Option<T>>(&self, i: usize, mut f: F) -> Result<T, T> {
        match self.entries[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            f(v.into()).map(T::into)
        }) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
}


impl<T: Sized + From<u64> + Into<u64>> Clone for Table<T> {
    fn clone(&self) -> Self {
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let entries = [DEFAULT; PT_LEN];
        for i in 0..PT_LEN {
            entries[i].store(self.entries[i].load(Ordering::Relaxed), Ordering::Relaxed);
        }
        Self {
            entries,
            phantom: self.phantom,
        }
    }
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

        assert_eq!(table::round(1, 15), 0);
        assert_eq!(table::round(1, PT_LEN), PT_LEN);
        assert_eq!(table::round(1, table::span(2)), table::span(2));
        assert_eq!(table::round(2, table::span(2)), table::span(2));
        assert_eq!(table::round(3, table::span(2)), 0);
        assert_eq!(table::round(3, 2 * table::span(3)), 2 * table::span(3));

        assert_eq!(table::page(1, 15, 2), 2);
        assert_eq!(table::page(1, PT_LEN, 2), PT_LEN + 2);
        assert_eq!(table::page(1, table::span(2), 0), table::span(2));
        assert_eq!(
            table::page(2, table::span(2), 1),
            table::span(2) + table::span(1)
        );
    }
}
