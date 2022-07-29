use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::Range;
use core::ptr::addr_of;
use core::sync::atomic::{self, AtomicU64, Ordering};

use crate::atomic::{Atomic, AtomicValue};
use crate::util::Page;
use crate::util::{align_down, align_up, log2, CacheLine};
use crate::Error;

pub const PT_ORDER: usize = 9;
pub const PT_LEN: usize = 1 << PT_ORDER;

/// Page table with atomic entries
#[repr(align(64))]
pub struct ATable<T: AtomicValue, const LEN: usize = { PT_LEN }> {
    entries: [Atomic<T>; LEN],
}

// Sanity checks
const _: () = assert!(size_of::<u64>() == ATable::<u64, PT_LEN>::PTE_SIZE);
const _: () = assert!(size_of::<ATable<u64, PT_LEN>>() == Page::SIZE);
const _: () = assert!(align_of::<ATable<u64, PT_LEN>>() == CacheLine::SIZE);

const _: () = assert!(ATable::<u64, PT_LEN>::SIZE == Page::SIZE);
const _: () = assert!(ATable::<u64, PT_LEN>::LEN == 512);
const _: () = assert!(ATable::<u64, PT_LEN>::ORDER == 9);

const _: () = assert!(ATable::<u16, 64>::PTE_SIZE == size_of::<u16>());
const _: () = assert!(ATable::<u16, 64>::SIZE == 128);

impl<T: AtomicValue, const LEN: usize> ATable<T, LEN> {
    pub const LEN: usize = LEN;
    pub const ORDER: usize = log2(LEN);
    pub const PTE_SIZE: usize = size_of::<T>();
    pub const SIZE: usize = LEN * Self::PTE_SIZE;

    pub fn empty() -> Self {
        Self {
            entries: unsafe { core::mem::zeroed() },
        }
    }
    pub fn fill(&self, e: T) {
        // cast to raw memory to let the compiler use vector instructions
        #[allow(clippy::cast_ref_to_mut)]
        let mem = unsafe { &mut *(&self.entries as *const _ as *mut [T; LEN]) };
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

impl<T: AtomicValue + fmt::Debug, const LEN: usize> fmt::Debug for ATable<T, LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Table {{")?;
        for (i, entry) in self.entries.iter().enumerate() {
            writeln!(f, "    {i:>3}; {:?},", entry.load())?;
        }
        writeln!(f, "}}")
    }
}

/// Bitfield replacing the level one-page table.
#[repr(align(64))]
pub struct Bitfield {
    data: [AtomicU64; Self::ENTRIES],
}

const _: () = assert!(size_of::<Bitfield>() == Bitfield::SIZE);
const _: () = assert!(size_of::<Bitfield>() == CacheLine::SIZE);
const _: () = assert!(Bitfield::LEN % Bitfield::ENTRY_BITS == 0);
const _: () = assert!(1 << Bitfield::ORDER == Bitfield::LEN);

impl Default for Bitfield {
    fn default() -> Self {
        const D: AtomicU64 = AtomicU64::new(0);
        Self {
            data: [D; Self::ENTRIES],
        }
    }
}

impl fmt::Debug for Bitfield {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitfield( ")?;
        for d in &self.data {
            write!(f, "{:016x} ", d.load(Ordering::Relaxed))?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Bitfield {
    pub const ENTRY_BITS: usize = 64;
    pub const ENTRIES: usize = Self::LEN / Self::ENTRY_BITS;
    pub const LEN: usize = PT_LEN;
    pub const ORDER: usize = log2(Self::LEN);
    pub const SIZE: usize = Self::LEN / 8;

    pub fn set(&self, i: usize, v: bool) {
        let di = i / Self::ENTRY_BITS;
        let bit = 1 << (i % Self::ENTRY_BITS);
        if v {
            self.data[di].fetch_or(bit, Ordering::SeqCst);
        } else {
            self.data[di].fetch_and(!bit, Ordering::SeqCst);
        }
    }

    pub fn get(&self, i: usize) -> bool {
        let di = i / Self::ENTRY_BITS;
        let bit = 1 << (i % Self::ENTRY_BITS);
        self.data[di].load(Ordering::SeqCst) & bit != 0
    }

    pub fn toggle(&self, i: usize, order: usize, expected: bool) -> Result<(), ()> {
        let di = i / Self::ENTRY_BITS;
        let num_pages = 1 << order;
        debug_assert!(num_pages <= u64::BITS);
        let mask = if num_pages >= u64::BITS {
            u64::MAX
        } else {
            ((1 << num_pages) - 1) << (i % Self::ENTRY_BITS)
        };
        match self.data[di].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |e| {
            if expected {
                e & mask == (mask * expected as u64)
            } else {
                e & mask == 0
            }
            .then(|| if expected { e & !mask } else { e | mask })
        }) {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }

    /// Set the first 0 bit to 1 returning its bit index.
    pub fn set_first_zero(&self, i: usize) -> Result<usize, Error> {
        for j in 0..self.data.len() {
            let i = (j + i) % self.data.len();

            #[cfg(feature = "stop")]
            {
                // Skip full entries for the tests
                if self.data[i].load(Ordering::SeqCst) == u64::MAX {
                    continue;
                }
                crate::stop::stop().unwrap();
            }

            if let Ok(e) = self.data[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |e| {
                let off = e.trailing_ones() as usize;
                (off < Self::ENTRY_BITS).then(|| e | (1 << off))
            }) {
                return Ok(i * Self::ENTRY_BITS + e.trailing_ones() as usize);
            }
        }
        Err(Error::Memory)
    }

    pub fn set_first_zeros(&self, i: usize, order: usize) -> Result<usize, Error> {
        if order == 0 {
            return self.set_first_zero(i);
        }

        debug_assert!(order <= 6);

        for j in 0..self.data.len() {
            let i = (j + i) % self.data.len();

            #[cfg(feature = "stop")]
            {
                // Skip full entries for the tests
                if self.data[i].load(Ordering::SeqCst) == u64::MAX {
                    continue;
                }
                crate::stop::stop().unwrap();
            }

            let mut offset = 0;
            if self.data[i]
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |e| {
                    if let Some((val, o)) = first_zeros_aligned(e, order) {
                        offset = o;
                        Some(val)
                    } else {
                        None
                    }
                })
                .is_ok()
            {
                return Ok(i * Self::ENTRY_BITS + offset);
            }
        }
        Err(Error::Memory)
    }

    pub fn fill(&self, v: bool) {
        let v = if v { u64::MAX } else { 0 };
        // cast to raw memory to let the compiler use vector instructions
        #[allow(clippy::cast_ref_to_mut)]
        let mem = unsafe { &mut *(addr_of!(self.data) as *mut [u64; Self::ENTRIES]) };
        mem.fill(v);
        // memory ordering has to be enforced with a memory barrier
        atomic::fence(Ordering::SeqCst);
    }
}

#[inline]
fn first_zeros_aligned(v: u64, order: usize) -> Option<(u64, usize)> {
    let num_pages = 1 << order;
    debug_assert!(num_pages <= u64::BITS as usize);
    if num_pages >= u64::BITS as usize {
        return if v == 0 { Some((u64::MAX, 0)) } else { None };
    }
    for i in 0..64 / num_pages {
        let i = i * num_pages;
        let mask = ((1 << num_pages) - 1) << i;
        if v & mask == 0 {
            return Some((v | mask, i as usize));
        }
    }
    None
}

/// Specifies the different table sizes from level 1 to N.
/// Level 0 are the pages below the first level of page tables.
/// Each entry contains the number of bits that are used to index the table.
/// The total sum of bits has to be less than the systems pointer size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping<const L: usize>(pub [usize; L]);

impl<const L: usize> Mapping<L> {
    pub const LEVELS: usize = L;

    pub const fn submap<const S: usize>(&self) -> Mapping<S> {
        let mut levels = [0; S];
        let mut l = 0;
        while l < S {
            levels[l] = self.0[l];
            l += 1;
        }
        Mapping(levels)
    }

    pub const fn with_lower<const S: usize>(self, lower: &Mapping<S>) -> Mapping<{ L + S }> {
        let mut levels = [0; L + S];
        let mut l = 0;
        while l < S {
            levels[l] = lower.0[l];
            l += 1;
        }
        while l < L + S {
            levels[l] = self.0[l - S];
            l += 1;
        }
        Mapping(levels)
    }

    pub const fn levels(&self) -> usize {
        L
    }

    pub const fn len(&self, level: usize) -> usize {
        if level == 0 {
            1
        } else {
            1 << self.0[level - 1]
        }
    }

    /// Memory bytes that the `level` covers.
    /// 0 is always 1 page.
    pub const fn m_span(&self, level: usize) -> usize {
        self.span(level) << Page::SIZE_BITS
    }

    /// Number of pages that the `level` covers.
    /// 0 is always 1 page.
    pub const fn span(&self, level: usize) -> usize {
        debug_assert!(level <= Self::LEVELS);

        1 << self.order(level)
    }

    /// Log2 of the number of pages that the `level` covers.
    pub const fn order(&self, level: usize) -> usize {
        debug_assert!(level <= Self::LEVELS);

        let mut res = 0;
        let mut l = 0;
        while l < level {
            res += self.0[l];
            l += 1;
        }
        res
    }

    pub const fn max_order(&self) -> usize {
        self.order(Self::LEVELS)
    }

    /// Returns pt index that contains the `page`
    pub const fn idx(&self, level: usize, page: usize) -> usize {
        debug_assert!(0 < level && level <= Self::LEVELS);

        (page / self.span(level - 1)) % self.len(level)
    }

    /// Returns the starting page of the corresponding page table
    pub const fn round(&self, level: usize, page: usize) -> usize {
        align_down(page, self.span(level))
    }

    /// Returns the page at the given index `i`
    pub const fn page(&self, level: usize, start: usize, i: usize) -> usize {
        debug_assert!(0 < level && level <= Self::LEVELS);

        self.round(level, start) + i * self.span(level - 1)
    }

    /// Returns the number of page tables needed to manage the number of `pages`
    pub const fn num_pts(&self, level: usize, pages: usize) -> usize {
        debug_assert!(0 < level && level <= Self::LEVELS);

        (pages + self.span(level) - 1) / self.span(level)
    }

    /// Computes the index range for the given page range
    pub fn range(&self, level: usize, pages: Range<usize>) -> Range<usize> {
        debug_assert!(0 < level && level <= Self::LEVELS);

        if pages.start < pages.end {
            let span_m1 = self.span(level - 1);
            let start_d = pages.start / span_m1;
            let end_d = align_up(pages.end, span_m1) / span_m1;

            let entries = self.len(level);
            let max = align_down(start_d, entries) + entries;
            let start = start_d % entries;
            let end = if end_d >= max {
                entries
            } else {
                end_d % entries
            };
            start..end
        } else {
            0..0
        }
    }

    /// Iterates over the table pages beginning with `start`.
    /// It wraps around the end and ends one before `start`.
    pub fn iterate(&self, level: usize, start: usize) -> impl Iterator<Item = usize> {
        debug_assert!(0 < level && level <= Self::LEVELS);

        let span_m1 = self.span(level - 1);
        let span = self.span(level);
        let rounded = self.round(level, start);
        let offset = self.round(level - 1, start - rounded);
        core::iter::once(start).chain(
            (1..self.len(level))
                .into_iter()
                .map(move |v| rounded + (v * span_m1 + offset) % span),
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::table::Mapping;
    use crate::util::Page;

    #[test]
    fn pt_size() {
        const MAPPING: Mapping<3> = Mapping([9, 9, 9]);

        assert_eq!(MAPPING.m_span(0), Page::SIZE);
        assert_eq!(MAPPING.m_span(1), Page::SIZE * 512);
        assert_eq!(MAPPING.m_span(2), Page::SIZE * 512 * 512);

        assert_eq!(MAPPING.span(0), 1);
        assert_eq!(MAPPING.span(1), 512);
        assert_eq!(MAPPING.span(2), 512 * 512);

        assert_eq!(MAPPING.num_pts(1, 0), 0);
        assert_eq!(MAPPING.num_pts(1, MAPPING.span(1)), 1);
        assert_eq!(MAPPING.num_pts(1, 2 * MAPPING.span(1) + 1), 3);
        assert_eq!(
            MAPPING.num_pts(1, MAPPING.span(3)),
            MAPPING.len(2) * MAPPING.len(3)
        );
    }

    #[test]
    fn pt_size_verying() {
        const MAPPING: Mapping<3> = Mapping([9, 6, 5]);

        assert_eq!(MAPPING.m_span(0), Page::SIZE);
        assert_eq!(MAPPING.m_span(1), Page::SIZE * 512);
        assert_eq!(MAPPING.m_span(2), Page::SIZE * 512 * 64);

        assert_eq!(MAPPING.span(0), 1);
        assert_eq!(MAPPING.span(1), 512);
        assert_eq!(MAPPING.span(2), 512 * 64);
        assert_eq!(MAPPING.span(3), 512 * 64 * 32);

        assert_eq!(MAPPING.num_pts(1, 0), 0);
        assert_eq!(MAPPING.num_pts(1, MAPPING.span(1)), 1);
        assert_eq!(MAPPING.num_pts(1, 2 * MAPPING.span(1) + 1), 3);
        assert_eq!(
            MAPPING.num_pts(1, MAPPING.span(3)),
            MAPPING.len(2) * MAPPING.len(3)
        );
    }

    #[test]
    fn rounding() {
        const MAPPING: Mapping<3> = Mapping([9, 9, 9]);

        assert_eq!(MAPPING.round(1, 15), 0);
        assert_eq!(MAPPING.round(1, 512), 512);
        assert_eq!(MAPPING.round(1, MAPPING.span(2)), MAPPING.span(2));
        assert_eq!(MAPPING.round(2, MAPPING.span(2)), MAPPING.span(2));
        assert_eq!(MAPPING.round(3, MAPPING.span(2)), 0);
        assert_eq!(MAPPING.round(3, 2 * MAPPING.span(3)), 2 * MAPPING.span(3));

        assert_eq!(MAPPING.page(1, 15, 2), 2);
        assert_eq!(MAPPING.page(1, 512, 2), 512 + 2);
        assert_eq!(MAPPING.page(1, MAPPING.span(2), 0), MAPPING.span(2));
        assert_eq!(
            MAPPING.page(2, MAPPING.span(2), 1),
            MAPPING.span(2) + MAPPING.span(1)
        );

        assert_eq!(MAPPING.idx(1, 3), 3);
        assert_eq!(MAPPING.idx(2, 3), 0);
        assert_eq!(MAPPING.idx(2, 3 * MAPPING.span(1)), 3);
        assert_eq!(MAPPING.idx(3, 3), 0);
        assert_eq!(MAPPING.idx(3, 3 * MAPPING.span(2)), 3);
    }

    #[test]
    fn rounding_verying() {
        const MAPPING: Mapping<3> = Mapping([9, 6, 5]);

        assert_eq!(MAPPING.round(1, 15), 0);
        assert_eq!(MAPPING.round(1, 512), 512);
        assert_eq!(MAPPING.round(1, MAPPING.span(2)), MAPPING.span(2));
        assert_eq!(MAPPING.round(2, MAPPING.span(2)), MAPPING.span(2));
        assert_eq!(MAPPING.round(3, MAPPING.span(2)), 0);
        assert_eq!(MAPPING.round(3, 2 * MAPPING.span(3)), 2 * MAPPING.span(3));

        assert_eq!(MAPPING.page(1, 15, 2), 2);
        assert_eq!(MAPPING.page(1, 512, 2), 512 + 2);
        assert_eq!(MAPPING.page(1, MAPPING.span(2), 0), MAPPING.span(2));
        assert_eq!(
            MAPPING.page(2, MAPPING.span(2), 1),
            MAPPING.span(2) + MAPPING.span(1)
        );

        assert_eq!(MAPPING.idx(1, 3), 3);
        assert_eq!(MAPPING.idx(2, 3), 0);
        assert_eq!(MAPPING.idx(2, 3 * MAPPING.span(1)), 3);
        assert_eq!(MAPPING.idx(3, 3), 0);
        assert_eq!(MAPPING.idx(3, 3 * MAPPING.span(2)), 3);
    }

    #[test]
    fn range() {
        const MAPPING: Mapping<3> = Mapping([9, 9, 9]);

        assert_eq!(MAPPING.range(1, 0..512), 0..512);
        assert_eq!(MAPPING.range(1, 0..0), 0..0);
        assert_eq!(MAPPING.range(1, 0..512 + 1), 0..512);
        assert_eq!(MAPPING.range(1, 512..512 - 1), 0..0);

        // L2
        assert_eq!(MAPPING.range(2, 0..MAPPING.span(1)), 0..1);
        assert_eq!(MAPPING.range(2, MAPPING.span(1)..3 * MAPPING.span(1)), 1..3);
        assert_eq!(MAPPING.range(2, 0..MAPPING.span(2)), 0..MAPPING.len(2));

        // L3
        assert_eq!(MAPPING.range(3, 0..MAPPING.span(2)), 0..1);
        assert_eq!(MAPPING.range(3, MAPPING.span(2)..3 * MAPPING.span(2)), 1..3);
        assert_eq!(MAPPING.range(3, 0..MAPPING.span(3)), 0..MAPPING.len(3));

        assert_eq!(MAPPING.range(3, 0..1), 0..1);
    }

    #[test]
    fn range_verying() {
        const MAPPING: Mapping<3> = Mapping([9, 6, 5]);

        assert_eq!(MAPPING.range(1, 0..MAPPING.len(1)), 0..MAPPING.len(1));
        assert_eq!(MAPPING.range(1, 0..0), 0..0);
        assert_eq!(MAPPING.range(1, 0..MAPPING.len(1) + 1), 0..MAPPING.len(1));
        assert_eq!(MAPPING.range(1, MAPPING.len(1)..MAPPING.len(1) - 1), 0..0);

        // L2
        assert_eq!(MAPPING.range(2, 0..MAPPING.span(1)), 0..1);
        assert_eq!(MAPPING.range(2, MAPPING.span(1)..3 * MAPPING.span(1)), 1..3);
        assert_eq!(MAPPING.range(2, 0..MAPPING.span(2)), 0..MAPPING.len(2));

        // L3
        assert_eq!(MAPPING.range(3, 0..MAPPING.span(2)), 0..1);
        assert_eq!(MAPPING.range(3, MAPPING.span(2)..3 * MAPPING.span(2)), 1..3);
        assert_eq!(MAPPING.range(3, 0..MAPPING.span(3)), 0..MAPPING.len(3));

        assert_eq!(MAPPING.range(3, 0..1), 0..1);
    }

    #[test]
    fn iterate() {
        const MAPPING: Mapping<3> = Mapping([9, 9, 9]);

        let mut iter = MAPPING.iterate(1, 0).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.last(), Some((511, 511)));

        // 5 -> 5, 6, .., 511, 0, 1, 2, 3, 4,
        let mut iter = MAPPING.iterate(1, 5).enumerate();
        assert_eq!(iter.next(), Some((0, 5)));
        assert_eq!(iter.next(), Some((1, 6)));
        assert_eq!(iter.last(), Some((511, 4)));

        let mut iter = MAPPING.iterate(1, 5 + 2 * MAPPING.span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 + 2 * MAPPING.span(1))));
        assert_eq!(iter.next(), Some((1, 6 + 2 * MAPPING.span(1))));
        assert_eq!(iter.last(), Some((511, 4 + 2 * MAPPING.span(1))));

        let mut iter = MAPPING.iterate(2, 5 * MAPPING.span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 * MAPPING.span(1))));
        assert_eq!(iter.last(), Some((511, 4 * MAPPING.span(1))));

        let mut iter = MAPPING.iterate(2, 0).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next(), Some((1, MAPPING.span(1))));
        assert_eq!(iter.last(), Some((511, 511 * MAPPING.span(1))));

        let mut iter = MAPPING.iterate(2, 500).enumerate();
        assert_eq!(iter.next(), Some((0, 500)));
        assert_eq!(iter.next(), Some((1, MAPPING.span(1))));
        assert_eq!(iter.next(), Some((2, 2 * MAPPING.span(1))));
        assert_eq!(iter.last(), Some((511, 511 * MAPPING.span(1))));

        let mut iter = MAPPING.iterate(2, 499 * MAPPING.span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 499 * MAPPING.span(1))));
        assert_eq!(iter.last(), Some((511, 498 * MAPPING.span(1))));
    }

    #[test]
    fn iterate_varying() {
        const MAPPING: Mapping<3> = Mapping([9, 6, 5]);

        let mut iter = MAPPING.iterate(1, 0).enumerate();
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.last(), Some((511, 511)));

        // 5 -> 5, 6, .., 511, 0, 1, 2, 3, 4,
        let mut iter = MAPPING.iterate(1, 5).enumerate();
        assert_eq!(iter.next(), Some((0, 5)));
        assert_eq!(iter.next(), Some((1, 6)));
        assert_eq!(iter.last(), Some((511, 4)));

        let mut iter = MAPPING.iterate(1, 5 + 2 * MAPPING.span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 + 2 * MAPPING.span(1))));
        assert_eq!(iter.next(), Some((1, 6 + 2 * MAPPING.span(1))));
        assert_eq!(iter.last(), Some((511, 4 + 2 * MAPPING.span(1))));

        let mut iter = MAPPING.iterate(2, 5 * MAPPING.span(1)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 * MAPPING.span(1))));
        assert_eq!(iter.last(), Some((63, 4 * MAPPING.span(1))));

        let mut iter = MAPPING.iterate(3, 5 * MAPPING.span(2)).enumerate();
        assert_eq!(iter.next(), Some((0, 5 * MAPPING.span(2))));
        assert_eq!(iter.last(), Some((31, 4 * MAPPING.span(2))));
    }

    #[test]
    fn first_zeros_aligned() {
        assert_eq!(super::first_zeros_aligned(0b0, 0), Some((0b1, 0)));
        assert_eq!(super::first_zeros_aligned(0b0, 1), Some((0b11, 0)));
        assert_eq!(super::first_zeros_aligned(0b0, 2), Some((0b1111, 0)));
        assert_eq!(super::first_zeros_aligned(0b0, 3), Some((0xff, 0)));
        assert_eq!(super::first_zeros_aligned(0b0, 4), Some((0xffff, 0)));
        assert_eq!(super::first_zeros_aligned(0b0, 5), Some((0xffffffff, 0)));
        assert_eq!(
            super::first_zeros_aligned(0b0, 6),
            Some((0xffffffffffffffff, 0))
        );

        assert_eq!(super::first_zeros_aligned(0b1, 0), Some((0b11, 1)));
        assert_eq!(super::first_zeros_aligned(0b1, 1), Some((0b1101, 2)));
        assert_eq!(super::first_zeros_aligned(0b1, 2), Some((0xf1, 4)));
        assert_eq!(super::first_zeros_aligned(0b1, 3), Some((0xff01, 8)));
        assert_eq!(super::first_zeros_aligned(0b1, 4), Some((0xffff0001, 16)));
        assert_eq!(
            super::first_zeros_aligned(0b1, 5),
            Some((0xffffffff00000001, 32))
        );
        assert_eq!(super::first_zeros_aligned(0b1, 6), None);

        assert_eq!(super::first_zeros_aligned(0b101, 0), Some((0b111, 1)));
        assert_eq!(super::first_zeros_aligned(0b10011, 1), Some((0b11111, 2)));
        assert_eq!(super::first_zeros_aligned(0x10f, 2), Some((0x1ff, 4)));
        assert_eq!(super::first_zeros_aligned(0x100ff, 3), Some((0x1ffff, 8)));
        assert_eq!(
            super::first_zeros_aligned(0x10000ffff, 4),
            Some((0x1ffffffff, 16))
        );
        assert_eq!(
            super::first_zeros_aligned(0x00000000ff00ff0f, 5),
            Some((0xffffffffff00ff0f, 32))
        );
        assert_eq!(
            super::first_zeros_aligned(0b1111_0000_1100_0011_1000_1111, 2),
            Some((0b1111_1111_1100_0011_1000_1111, 16))
        );
    }
}
