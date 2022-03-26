use core::fmt;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ops::Range;
use core::sync::atomic::{self, AtomicU64, Ordering};

use crate::atomic::Atomic;
use crate::Page;

/// Page table with atomic entries
#[repr(align(0x1000))]
pub struct Table<T: Sized + From<u64> + Into<u64> = u64> {
    entries: [Atomic<T>; Table::LEN],
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

    pub const LEVELS: usize = 4;

    /// Area in bytes that a page table covers
    #[inline]
    pub const fn m_span(level: usize) -> usize {
        Self::span(level) << Page::SIZE_BITS
    }

    /// Area in pages that a page table covers
    #[inline]
    pub const fn span(level: usize) -> usize {
        1 << (Self::LEN_BITS * level)
    }

    /// Returns pt index that contains the `page`
    #[inline]
    pub const fn idx(level: usize, page: usize) -> usize {
        (page >> (Self::LEN_BITS * (level - 1))) & (Self::LEN - 1)
    }

    /// Returns the starting page of the corresponding page table
    #[inline]
    pub const fn round(level: usize, page: usize) -> usize {
        page & !((1 << (Self::LEN_BITS * level)) - 1)
    }

    /// Returns the page at the given index `i`
    #[inline]
    pub const fn page(level: usize, start: usize, i: usize) -> usize {
        Self::round(level, start) + i * Self::span(level - 1)
    }

    #[inline]
    pub const fn num_pts(level: usize, pages: usize) -> usize {
        (pages + Self::span(level) - 1) / Self::span(level)
    }

    /// Computes the index range for the given page range
    pub fn range(level: usize, pages: Range<usize>) -> Range<usize> {
        let bits = Self::LEN_BITS * (level - 1);
        let start = pages.start >> bits;
        let end = (pages.end >> bits) + (pages.end.trailing_zeros() < bits as _) as usize;

        let end = end.saturating_sub(start & !(Self::LEN - 1)).min(Self::LEN);
        let start = start & (Self::LEN - 1);

        start..end
    }

    /// Iterates over the table pages beginning with `start`.
    /// It wraps around the end and ends one before `start`.
    pub fn iterate(level: usize, start: usize) -> impl Iterator<Item = usize> {
        debug_assert!(level >= 1);

        let bits = Self::LEN_BITS * (level - 1);
        let pt_start = Self::round(level, start);
        let offset = (start >> bits) % Self::LEN;
        std::iter::once(start).chain(
            (1..Table::LEN)
                .into_iter()
                .map(move |v| (((offset + v) % Table::LEN) << bits) + pt_start),
        )
    }
}

impl<T: Sized + From<u64> + Into<u64> + Clone> Table<T> {
    pub fn empty() -> Self {
        Self {
            entries: unsafe { std::mem::zeroed() },
            phantom: PhantomData,
        }
    }
    pub fn fill(&self, e: T) {
        // cast to raw memory to let the compiler use vector instructions
        #[allow(clippy::cast_ref_to_mut)]
        let mem = unsafe { &mut *(&self.entries as *const _ as *mut [T; Table::LEN]) };
        mem.fill(e);
        // memory ordering has to be enforced with a memory barrier
        atomic::fence(Ordering::SeqCst);
    }
    #[inline]
    pub fn get(&self, i: usize) -> T {
        self.entries[i].load()
    }
    #[inline]
    pub fn set(&self, i: usize, e: T) {
        self.entries[i].store(e);
    }
    #[inline]
    pub fn cas(&self, i: usize, expected: T, new: T) -> Result<T, T> {
        self.entries[i].compare_exchange(expected, new)
    }
    #[inline]
    pub fn update<F: FnMut(T) -> Option<T>>(&self, i: usize, f: F) -> Result<T, T> {
        self.entries[i].update(f)
    }
}

impl<T: fmt::Debug + Sized + From<u64> + Into<u64>> fmt::Debug for Table<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Table {{")?;
        for (i, entry) in self.entries.iter().enumerate() {
            writeln!(f, "    {i:>3}; {:?},", entry.load())?;
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
        let mut iter = Table::iterate(1, 0).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.last(), Some((511, 511)));

        // 5 -> 5, 6, .., 511, 0, 1, 2, 3, 4,
        let mut iter = Table::iterate(1, 5).enumerate();
        assert_eq!(iter.next(), Some((0, 5)));
        assert_eq!(iter.next(), Some((1, 6)));
        assert_eq!(iter.last(), Some((511, 4)));

        let mut iter = Table::iterate(1, 5 + 2 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 + 2 * Table::span(1))));
        assert_eq!(iter.next(), Some((1, 6 + 2 * Table::span(1))));
        assert_eq!(iter.last(), Some((511, 4 + 2 * Table::span(1))));

        let mut iter = Table::iterate(2, 5 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 * Table::span(1))));
        assert_eq!(iter.last(), Some((511, 4 * Table::span(1))));

        let mut iter = Table::iterate(2, 0).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next(), Some((1, 1 * Table::span(1))));
        assert_eq!(iter.last(), Some((511, 511 * Table::span(1))));

        let mut iter = Table::iterate(2, 500).enumerate();
        assert_eq!(iter.next(), Some((0, 500)));
        assert_eq!(iter.next(), Some((1, Table::span(1))));
        assert_eq!(iter.next(), Some((2, 2 * Table::span(1))));
        assert_eq!(iter.last(), Some((511, 511 * Table::span(1))));

        let mut iter = Table::iterate(2, 499 * Table::span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 499 * Table::span(1))));
        assert_eq!(iter.last(), Some((511, 498 * Table::span(1))));
    }
}
