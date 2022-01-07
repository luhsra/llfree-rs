use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Page;

/// Page table with atomic entries
#[repr(align(0x1000))]
pub struct Table<T = u64> {
    entries: [AtomicU64; Table::LEN],
    phantom: PhantomData<T>,
}

const _: () = assert!(size_of::<AtomicU64>() == Table::PTE_SIZE);
const _: () = assert!(size_of::<Table>() == Page::SIZE);
const _: () = assert!(size_of::<usize>() == size_of::<u64>());

impl Table {
    pub const PTE_SIZE_BITS: usize = 3; // 2^3 => 8B => 64b
    pub const PTE_SIZE: usize = 1 << Self::PTE_SIZE_BITS;
    pub const LEN_BITS: usize = Page::SIZE_BITS - Self::PTE_SIZE_BITS;
    pub const LEN: usize = 1 << Self::LEN_BITS;

    pub const LAYERS: usize = 4;

    /// Area in bytes that a page table covers
    #[inline(always)]
    pub const fn m_span(layer: usize) -> usize {
        Self::span(layer) << Page::SIZE_BITS
    }

    /// Area in pages that a page table covers
    #[inline(always)]
    pub const fn span(layer: usize) -> usize {
        1 << (Self::LEN_BITS * layer)
    }

    /// Returns pt index that contains the `page`
    #[inline(always)]
    pub const fn idx(layer: usize, page: usize) -> usize {
        (page >> (Self::LEN_BITS * (layer - 1))) & (Self::LEN - 1)
    }

    /// Returns the starting page of the corresponding page table
    #[inline(always)]
    pub const fn round(layer: usize, page: usize) -> usize {
        page & !((1 << (Self::LEN_BITS * layer)) - 1)
    }

    /// Returns the page at the given index `i`
    #[inline(always)]
    pub const fn page(layer: usize, start: usize, i: usize) -> usize {
        Self::round(layer, start) + i * Self::span(layer - 1)
    }

    pub const fn num_pts(layer: usize, pages: usize) -> usize {
        (pages + Self::span(layer) - 1) / Self::span(layer)
    }

    /// Computes the index range for the given page range
    pub fn range(layer: usize, pages: Range<usize>) -> Range<usize> {
        let bits = Self::LEN_BITS * (layer - 1);
        let start = pages.start >> bits;
        let end = (pages.end >> bits) + (pages.end.trailing_zeros() < bits as _) as usize;

        let end = end.saturating_sub(start & !(Self::LEN - 1)).min(Self::LEN);
        let start = start & (Self::LEN - 1);

        start..end
    }

    /// Iterates over the table pages beginning with `start`.
    /// It wraps around the end and ends one before `start`.
    pub fn iterate(layer: usize, start: usize, pages: usize) -> impl Iterator<Item = usize> {
        assert!(layer >= 1 && start < pages);

        let bits = Self::LEN_BITS * (layer - 1);
        let pt_start = Self::round(layer, start);
        let max =
            ((pages.saturating_sub(pt_start) + Self::span(layer - 1) - 1) >> bits).min(Self::LEN);
        let offset = (start >> bits) % Self::LEN;
        std::iter::once(start).chain(
            (1..max)
                .into_iter()
                .map(move |v| (((offset + v) % max) << bits) + pt_start),
        )
    }
}

impl<T: Sized + From<u64> + Into<u64>> Table<T> {
    pub fn empty() -> Self {
        Self {
            entries: unsafe { std::mem::zeroed() },
            phantom: PhantomData,
        }
    }

    pub fn clear(&self) {
        for i in 0..Table::LEN {
            self.entries[i].store(0, Ordering::SeqCst);
        }
    }
}

impl<T: Sized + From<u64> + Into<u64>> AtomicBuffer<T> for Table<T> {
    fn entry(&self, i: usize) -> &AtomicU64 {
        &self.entries[i]
    }
}

pub trait AtomicBuffer<T: Sized + From<u64> + Into<u64>> {
    fn entry(&self, i: usize) -> &AtomicU64;

    fn get(&self, i: usize) -> T {
        T::from(self.entry(i).load(Ordering::SeqCst))
    }

    fn set(&self, i: usize, e: T) {
        self.entry(i).store(e.into(), Ordering::SeqCst);
    }

    fn cas(&self, i: usize, expected: T, new: T) -> Result<T, T> {
        match self.entry(i).compare_exchange(
            expected.into(),
            new.into(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }

    fn update<F: FnMut(T) -> Option<T>>(&self, i: usize, mut f: F) -> Result<T, T> {
        match self
            .entry(i)
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                f(v.into()).map(T::into)
            }) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
}

impl<T: Sized + From<u64> + Into<u64>> Clone for Table<T> {
    fn clone(&self) -> Self {
        let entries: [AtomicU64; Table::LEN] = unsafe { std::mem::zeroed() };
        for i in 0..Table::LEN {
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

#[cfg(test)]
mod test {
    use crate::table::Table;
    use crate::util::logging;
    use crate::Page;

    #[test]
    fn pt_size() {
        assert_eq!(Table::m_span(0), Page::SIZE);
        assert_eq!(Table::m_span(1), Page::SIZE * Table::LEN);
        assert_eq!(Table::m_span(2), Page::SIZE * Table::LEN * Table::LEN);

        assert_eq!(Table::span(0), 1);
        assert_eq!(Table::span(1), Table::LEN);
        assert_eq!(Table::span(2), Table::LEN * Table::LEN);
    }

    #[test]
    fn indexing() {
        assert_eq!(Table::range(1, 0..Table::LEN), 0..Table::LEN);
        assert_eq!(Table::range(1, 0..0), 0..0);
        assert_eq!(Table::range(1, 0..Table::LEN + 1), 0..Table::LEN);
        assert_eq!(Table::range(1, Table::LEN..Table::LEN - 1), 0..0);

        // L2
        assert_eq!(Table::range(2, 0..Table::span(1)), 0..1);
        assert_eq!(Table::range(2, Table::span(1)..3 * Table::span(1)), 1..3);
        assert_eq!(Table::range(2, 0..Table::span(2)), 0..Table::LEN);

        // L3
        assert_eq!(Table::range(3, 0..Table::span(2)), 0..1);
        assert_eq!(Table::range(3, Table::span(2)..3 * Table::span(2)), 1..3);
        assert_eq!(Table::range(3, 0..Table::span(3)), 0..Table::LEN);

        assert_eq!(Table::range(3, 0..1), 0..1);

        assert_eq!(Table::round(1, 15), 0);
        assert_eq!(Table::round(1, Table::LEN), Table::LEN);
        assert_eq!(Table::round(1, Table::span(2)), Table::span(2));
        assert_eq!(Table::round(2, Table::span(2)), Table::span(2));
        assert_eq!(Table::round(3, Table::span(2)), 0);
        assert_eq!(Table::round(3, 2 * Table::span(3)), 2 * Table::span(3));

        assert_eq!(Table::page(1, 15, 2), 2);
        assert_eq!(Table::page(1, Table::LEN, 2), Table::LEN + 2);
        assert_eq!(Table::page(1, Table::span(2), 0), Table::span(2));
        assert_eq!(
            Table::page(2, Table::span(2), 1),
            Table::span(2) + Table::span(1)
        );
    }

    #[test]
    fn iterate() {
        logging();
        // 5 -> 5, 6, .., 499, 0, 1, 2, 3, 4,
        let mut iter = Table::iterate(1, 5, 500).enumerate();
        assert_eq!(iter.next(), Some((0, 5)));
        assert_eq!(iter.next(), Some((1, 6)));
        assert_eq!(iter.last(), Some((499, 4)));

        let mut iter =
            Table::iterate(1, 5 + 2 * Table::span(1), 500 + 2 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 + 2 * Table::span(1))));
        assert_eq!(iter.next(), Some((1, 6 + 2 * Table::span(1))));
        assert_eq!(iter.last(), Some((499, 4 + 2 * Table::span(1))));

        let mut iter = Table::iterate(2, 5 * Table::span(1), 500 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 * Table::span(1))));
        assert_eq!(iter.last(), Some((499, 4 * Table::span(1))));

        let mut iter = Table::iterate(2, 0, 500 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.last(), Some((499, 499 * Table::span(1))));

        let mut iter = Table::iterate(2, 500, 500 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 500)));
        assert_eq!(iter.next(), Some((1, Table::span(1))));
        assert_eq!(iter.last(), Some((499, 499 * Table::span(1))));

        let mut iter = Table::iterate(2, 499 * Table::span(1), 500 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 499 * Table::span(1))));
        assert_eq!(iter.last(), Some((499, 498 * Table::span(1))));

        let mut iter = Table::iterate(2, 499 * Table::span(1), 1000 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 499 * Table::span(1))));
        assert_eq!(iter.last(), Some((511, 498 * Table::span(1))));

        let mut iter = Table::iterate(2, 0, 2 * Table::span(1) + 1).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next(), Some((1, 1 * Table::span(1))));
        assert_eq!(iter.next(), Some((2, 2 * Table::span(1))));
        assert_eq!(iter.next(), None);
    }
}
