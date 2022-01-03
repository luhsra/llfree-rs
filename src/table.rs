use std::alloc::Layout;
use std::fmt;
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
#[repr(align(0x1000))]
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
pub fn range(layer: usize, pages: Range<usize>) -> Range<usize> {
    let bits = PT_LEN_BITS * (layer - 1);
    let start = pages.start >> bits;
    let end = (pages.end >> bits) + (pages.end.trailing_zeros() < bits as _) as usize;

    let end = end.saturating_sub(start & !(PT_LEN - 1)).min(PT_LEN);
    let start = start & (PT_LEN - 1);

    start..end
}

/// Iterates over the table pages beginning with `start`.
/// It wraps around the end and ends one before `start`.
pub fn iterate(layer: usize, start: usize, pages: usize) -> impl Iterator<Item = usize> {
    assert!(layer >= 1 && start < pages);

    let bits = PT_LEN_BITS * (layer - 1);
    let pt_start = round(layer, start);
    let max = (pages.saturating_sub(pt_start) >> bits).min(PT_LEN);
    let offset = (start >> bits) % PT_LEN;
    std::iter::once(start).chain((1..max).into_iter().map(move |v| (((offset + v) % max) << bits) + pt_start))
}

impl<T: Sized + From<u64> + Into<u64>> Table<T> {
    pub fn empty() -> Self {
        Self {
            entries: unsafe { std::mem::zeroed() },
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
        let entries: [AtomicU64; PT_LEN] = unsafe { std::mem::zeroed() };
        for i in 0..PT_LEN {
            entries[i].store(self.entries[i].load(Ordering::Relaxed), Ordering::Relaxed);
        }
        Self {
            entries,
            phantom: self.phantom,
        }
    }
}

impl<T: fmt::Debug + From<u64>> fmt::Debug for Table<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Table {{")?;
        for (i, entry) in self.entries.iter().enumerate() {
            writeln!(
                f,
                "    {:>3}; {:?},",
                i,
                T::from(entry.load(Ordering::SeqCst))
            )?;
        }
        writeln!(f, "}}")
    }
}

/// Correctly sized and aligned page.
#[derive(Clone)]
#[repr(align(0x1000))]
pub struct Page {
    _data: [u8; PAGE_SIZE],
}
const _: () = assert!(Layout::new::<Page>().size() == PAGE_SIZE);
const _: () = assert!(Layout::new::<Page>().align() == PAGE_SIZE);
impl Page {
    pub const fn new() -> Self {
        Self {
            _data: [0; PAGE_SIZE],
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        table::{self, PAGE_SIZE, PT_LEN},
        util::logging,
    };

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

    #[test]
    fn iterate() {
        logging();
        // 5 -> 5, 6, .., 499, 0, 1, 2, 3, 4,
        let mut iter = table::iterate(1, 5, 500).enumerate();
        assert_eq!(iter.next(), Some((0, 5)));
        assert_eq!(iter.next(), Some((1, 6)));
        assert_eq!(iter.last(), Some((499, 4)));

        let mut iter = table::iterate(1, 5 + 2 * table::span(1), 500 + 2 * table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 + 2 * table::span(1))));
        assert_eq!(iter.next(), Some((1, 6 + 2 * table::span(1))));
        assert_eq!(iter.last(), Some((499, 4 + 2 * table::span(1))));

        let mut iter = table::iterate(2, 5 * table::span(1), 500 * table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 * table::span(1))));
        assert_eq!(iter.last(), Some((499, 4 * table::span(1))));

        let mut iter = table::iterate(2, 0, 500 * table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.last(), Some((499, 499 * table::span(1))));

        let mut iter = table::iterate(2, 500, 500 * table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 500)));
        assert_eq!(iter.next(), Some((1, table::span(1))));
        assert_eq!(iter.last(), Some((499, 499 * table::span(1))));

        let mut iter = table::iterate(2, 499 * table::span(1), 500 * table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 499 * table::span(1))));
        assert_eq!(iter.last(), Some((499, 498 * table::span(1))));

        let mut iter = table::iterate(2, 499 * table::span(1), 1000 * table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 499 * table::span(1))));
        assert_eq!(iter.last(), Some((511, 498 * table::span(1))));
    }
}
