use core::mem::size_of;
use core::ops::{Not, Range};
use core::sync::atomic::{self, AtomicU64, Ordering::*};
use core::{fmt, mem};

use log::error;

use crate::atomic::{Atom, Atomic};
use crate::util::align_down;
use crate::Frame;
use crate::{Error, Result};

pub const PT_ORDER: usize = 9;
pub const PT_LEN: usize = 1 << PT_ORDER;

pub trait AtomicArray<T: Copy, const L: usize> {
    /// Overwrite the content of the whole array non-atomically.
    ///
    /// This is faster than atomics but does not handle race conditions.
    fn atomic_fill(&self, e: T);
}

impl<T: Atomic, const L: usize> AtomicArray<T, L> for [Atom<T>; L] {
    fn atomic_fill(&self, e: T) {
        // cast to raw memory to let the compiler use vector instructions
        #[allow(clippy::cast_ref_to_mut)]
        let mem = unsafe { &mut *(self.as_ptr() as *mut [T; L]) };
        mem.fill(e);
        // memory ordering has to be enforced with a memory barrier
        atomic::fence(Release);
    }
}

/// Bitfield replacing the level one table.
pub struct Bitfield<const N: usize> {
    data: [Atom<u64>; N],
}

const _: () = assert!(size_of::<Bitfield<64>>() >= 8);
const _: () = assert!(Bitfield::<64>::LEN % Bitfield::<64>::ENTRY_BITS == 0);
const _: () = assert!(1 << Bitfield::<64>::ORDER == Bitfield::<64>::LEN);
const _: () = assert!(Bitfield::<2>::ORDER == 7);

impl<const N: usize> Default for Bitfield<N> {
    fn default() -> Self {
        const A: Atom<u64> = Atom::raw(AtomicU64::new(0));
        Self { data: [A; N] }
    }
}

impl<const N: usize> fmt::Debug for Bitfield<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitfield( ")?;
        for d in &self.data {
            write!(f, "{:016x} ", d.load())?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl<const N: usize> Bitfield<N>
where
    [(); N * size_of::<u64>() / size_of::<u8>()]:, // Validate generic const expressions
    [(); N * size_of::<u64>() / size_of::<u16>()]:,
    [(); N * size_of::<u64>() / size_of::<u32>()]:,
{
    pub const ENTRY_BITS: usize = 64;
    pub const ENTRIES: usize = N;
    pub const LEN: usize = N * Self::ENTRY_BITS;
    pub const ORDER: usize = Self::LEN.ilog2() as _;

    /// Overwrite the `range` of bits with `v`
    pub fn set(&self, range: Range<usize>, v: bool) {
        assert!(range.start <= range.end && range.end <= Self::LEN);

        if range.start != range.end {
            for ei in range.start / Self::ENTRY_BITS..=(range.end - 1) / Self::ENTRY_BITS {
                let bit_off = ei * Self::ENTRY_BITS;
                let bit_start = range.start.saturating_sub(bit_off);
                let bit_end = (range.end - bit_off).min(Self::ENTRY_BITS);
                let bits = bit_end - bit_start;
                let byte = (u64::MAX >> (Self::ENTRY_BITS - bits)) << bit_start;
                if v {
                    self.data[ei].fetch_or(byte);
                } else {
                    self.data[ei].fetch_and(!byte);
                }
            }
        }
    }

    /// Return if the `i`-th bit is set
    pub fn get(&self, i: usize) -> bool {
        let di = i / Self::ENTRY_BITS;
        let bit = 1 << (i % Self::ENTRY_BITS);
        self.data[di].load() & bit != 0
    }

    /// Return the  `i`-th entry
    pub fn get_entry(&self, i: usize) -> u64 {
        self.data[i].load()
    }

    /// Toggle 2^`order` bits at the `i`-th place if they are all zero or one as expected
    ///
    /// # Warning
    /// Orders above 6 need multiple CAS operations, which might lead to race conditions!
    pub fn toggle(&self, i: usize, order: usize, expected: bool) -> Result<()> {
        let num_bits = 1 << order;
        debug_assert!(i % num_bits == 0, "not aligned");
        match order {
            0..=2 => {
                // Updates within a single entry
                let mask = (u64::MAX >> (Self::ENTRY_BITS - num_bits)) << (i % Self::ENTRY_BITS);
                let di = i / Self::ENTRY_BITS;
                match self.data[di].fetch_update(|e| {
                    if expected {
                        (e & mask == mask).then_some(e & !mask)
                    } else {
                        (e & mask == 0).then_some(e | mask)
                    }
                }) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(Error::Retry),
                }
            }
            3 => self.toggle_int::<u8>(i, expected),
            4 => self.toggle_int::<u16>(i, expected),
            5 => self.toggle_int::<u32>(i, expected),
            _ => {
                // Update multiple entries
                let num_entries = num_bits / Self::ENTRY_BITS;
                let di = i / Self::ENTRY_BITS;
                for i in di..di + num_entries {
                    let expected = if expected { !0 } else { 0 };
                    if let Err(_) = self.data[i].compare_exchange(expected, !expected) {
                        // Undo changes
                        for j in (di..i).rev() {
                            if let Err(_) = self.data[j].compare_exchange(!expected, expected) {
                                error!("Failed undo toggle");
                                return Err(Error::Corruption);
                            }
                        }
                        return Err(Error::Retry);
                    }
                }
                Ok(())
            }
        }
    }

    /// Toggle multiple bits with a single correctly sized compare exchange operation
    ///
    /// Note: This only seems to make a difference between a 64 bit fetch_update on Intel Optane
    fn toggle_int<I: Atomic + Default + Not<Output = I>>(&self, i: usize, e: bool) -> Result<()>
    where
        [(); N * size_of::<u64>() / size_of::<I>()]:,
    {
        let idx = i / (8 * size_of::<I>());
        let val = if e { !I::default() } else { I::default() };
        // Cast to int type, keeping the same byte size
        let data: &[Atom<I>; N * size_of::<u64>() / size_of::<I>()] =
            unsafe { mem::transmute(&self.data) };
        match data[idx].compare_exchange(val, !val) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Retry),
        }
    }

    /// Set the first aligned 2^`order` zero bits, returning the bit offset
    ///
    /// # Warning
    /// Orders above 6 need multiple CAS operations, which might lead to race conditions!
    pub fn set_first_zeros(&self, start_i: usize, order: usize) -> Result<usize> {
        let offset = start_i / Self::ENTRY_BITS;
        debug_assert!(offset < Self::ENTRIES);

        if order > Self::ENTRY_BITS.ilog2() as usize {
            return self.set_first_zero_entries(order);
        }

        for i in 0..self.data.len() {
            let i = (i + offset) % self.data.len();

            #[cfg(all(test, feature = "stop"))]
            {
                // Skip full entries for the tests
                if self.data[i].load() == u64::MAX {
                    continue;
                }
                crate::stop::stop().unwrap();
            }

            let mut offset = 0;
            if self.data[i]
                .fetch_update(|e| {
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

    /// Allocate multiple entries with multiple CAS
    ///
    /// # Warning
    /// Using multiple CAS operations might lead to race conditions!
    fn set_first_zero_entries(&self, order: usize) -> Result<usize> {
        debug_assert!(order > Self::ENTRY_BITS.ilog2() as usize);
        debug_assert!(order <= Self::ORDER);
        let num_entries = 1 << (order - Self::ENTRY_BITS.ilog2() as usize);

        for (i, chunk) in self.data.chunks(num_entries).enumerate() {
            // Check that these entries are free
            if chunk.iter().all(|e| e.load() == 0) {
                for (j, entry) in chunk.iter().enumerate() {
                    if let Err(_) = entry.compare_exchange(0, u64::MAX) {
                        // Undo previous updates
                        for k in (0..j).rev() {
                            if let Err(_) = chunk[k].compare_exchange(u64::MAX, 0) {
                                error!("Failed undo search");
                                return Err(Error::Corruption);
                            }
                        }
                        break;
                    }
                }
                return Ok(i * num_entries * Self::ENTRY_BITS);
            }
        }
        Err(Error::Memory)
    }

    /// Fill this bitset with `v` ignoring any previous data.
    ///
    /// This is faster than atomics but does not handle race conditions.
    pub fn fill(&self, v: bool) {
        let v = if v { u64::MAX } else { 0 };
        // cast to raw memory to let the compiler use vector instructions
        #[allow(clippy::cast_ref_to_mut)]
        let mem = unsafe { &mut *(self.data.as_ptr() as *mut [u64; N]) };
        mem.fill(v);
        // memory ordering has to be enforced with a memory barrier
        atomic::fence(SeqCst);
    }

    /// Fills the bitset using atomic compare exchage, returning false on failure
    ///
    /// # Warning
    /// If false is returned, the bitfield might be corrupted!
    pub fn fill_cas(&self, v: bool) -> bool {
        let v = if v { u64::MAX } else { 0 };

        for e in &self.data {
            if e.compare_exchange(!v, v).is_err() {
                return false;
            }
        }
        true
    }

    /// Returns the number of zeros in this bitfield
    pub fn count_zeros(&self) -> usize {
        self.data
            .iter()
            .map(|v| v.load().count_zeros() as usize)
            .sum()
    }
}

/// Set the first aligned 2^`order` zero bits, returning the bit offset
fn first_zeros_aligned(v: u64, order: usize) -> Option<(u64, usize)> {
    match order {
        0 => {
            let off = v.trailing_ones();
            (off < u64::BITS).then(|| (v | (0b1 << off), off as _))
        }
        1 => {
            let mask = 0xaaaa_aaaa_aaaa_aaaa_u64;
            let or = (v | (v >> 1)) | mask;
            let off = or.trailing_ones();
            (off < u64::BITS).then(|| (v | (0b11 << off), off as _))
        }
        2 => {
            let mask = 0x1111_1111_1111_1111_u64;
            let off = (((v.wrapping_sub(mask) & !v) >> 3) & mask).trailing_zeros();
            (off < u64::BITS).then(|| (v | (0b1111 << off), off as _))
        }
        3 => {
            // https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
            let mask = 0x0101_0101_0101_0101_u64;
            let off = (((v.wrapping_sub(mask) & !v) >> 7) & mask).trailing_zeros();
            (off < u64::BITS).then(|| (v | (0xff << off), off as _))
        }
        4 => {
            let mask = 0x0001_0001_0001_0001_u64;
            let off = (((v.wrapping_sub(mask) & !v) >> 15) & mask).trailing_zeros();
            (off < u64::BITS).then(|| (v | (0xffff << off), off as _))
        }
        5 => {
            let mask = 0xffff_ffff_u64;
            if v as u32 == 0 {
                Some((v | mask, 0))
            } else if v >> 32 == 0 {
                Some((v | (mask << 32), 32))
            } else {
                None
            }
        }
        6 => {
            if v == 0 {
                Some((u64::MAX, 0))
            } else {
                None
            }
        }
        // All other orders are handled differently
        _ => unreachable!(),
    }
}

/// Specifies the different table sizes from level 1 to N.
/// Level 0 are the frames below the first level of tables.
/// Each entry contains the number of bits that are used to index the table.
/// The total sum of bits has to be less than the systems pointer size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping<const L: usize>(pub [usize; L]);

impl<const L: usize> Mapping<L> {
    pub const LEVELS: usize = L;

    #[inline(always)]
    pub const fn levels(&self) -> usize {
        L
    }

    #[inline(always)]
    pub const fn len(&self, level: usize) -> usize {
        if level == 0 {
            1
        } else {
            1 << self.0[level - 1]
        }
    }

    /// Memory bytes that the `level` covers.
    /// 0 is always 1 frame.
    #[inline(always)]
    pub const fn m_span(&self, level: usize) -> usize {
        self.span(level) << Frame::SIZE_BITS
    }

    /// Number of frames that the `level` covers.
    /// 0 is always 1 frame.
    #[inline(always)]
    pub const fn span(&self, level: usize) -> usize {
        debug_assert!(level <= Self::LEVELS);

        1 << self.order(level)
    }

    /// Log2 of the number of frames that the `level` covers.
    #[inline(always)]
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

    #[inline(always)]
    pub const fn max_order(&self) -> usize {
        self.order(Self::LEVELS)
    }

    /// Returns pt index that contains the `frame`
    #[inline(always)]
    pub const fn idx(&self, level: usize, frame: usize) -> usize {
        debug_assert!(0 < level && level <= Self::LEVELS);

        (frame / self.span(level - 1)) % self.len(level)
    }

    /// Returns the starting frame of the corresponding table
    #[inline(always)]
    pub const fn round(&self, level: usize, frame: usize) -> usize {
        align_down(frame, self.span(level))
    }

    /// Returns the frame at the given index `i`
    #[inline(always)]
    pub const fn frame(&self, level: usize, start: usize, i: usize) -> usize {
        debug_assert!(0 < level && level <= Self::LEVELS);

        self.round(level, start) + i * self.span(level - 1)
    }

    /// Returns the number of tables needed to manage the number of `frames`
    #[inline(always)]
    pub const fn num_tables(&self, level: usize, frames: usize) -> usize {
        debug_assert!(0 < level && level <= Self::LEVELS);

        frames.div_ceil(self.span(level))
    }

    /// Computes the index range for the given frame range
    #[inline(always)]
    pub const fn range(&self, level: usize, frames: Range<usize>) -> Range<usize> {
        debug_assert!(0 < level && level <= Self::LEVELS);

        if frames.start < frames.end {
            let span_m1 = self.span(level - 1);
            let start_d = frames.start / span_m1;
            let end_d = frames.end.div_ceil(span_m1);

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
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::table::Mapping;
    use crate::Frame;

    #[test]
    fn pt_size() {
        const MAPPING: Mapping<3> = Mapping([9, 9, 9]);

        assert_eq!(MAPPING.m_span(0), Frame::SIZE);
        assert_eq!(MAPPING.m_span(1), Frame::SIZE * 512);
        assert_eq!(MAPPING.m_span(2), Frame::SIZE * 512 * 512);

        assert_eq!(MAPPING.span(0), 1);
        assert_eq!(MAPPING.span(1), 512);
        assert_eq!(MAPPING.span(2), 512 * 512);

        assert_eq!(MAPPING.num_tables(1, 0), 0);
        assert_eq!(MAPPING.num_tables(1, MAPPING.span(1)), 1);
        assert_eq!(MAPPING.num_tables(1, 2 * MAPPING.span(1) + 1), 3);
        assert_eq!(
            MAPPING.num_tables(1, MAPPING.span(3)),
            MAPPING.len(2) * MAPPING.len(3)
        );
    }

    #[test]
    fn pt_size_verying() {
        const MAPPING: Mapping<3> = Mapping([9, 6, 5]);

        assert_eq!(MAPPING.m_span(0), Frame::SIZE);
        assert_eq!(MAPPING.m_span(1), Frame::SIZE * 512);
        assert_eq!(MAPPING.m_span(2), Frame::SIZE * 512 * 64);

        assert_eq!(MAPPING.span(0), 1);
        assert_eq!(MAPPING.span(1), 512);
        assert_eq!(MAPPING.span(2), 512 * 64);
        assert_eq!(MAPPING.span(3), 512 * 64 * 32);

        assert_eq!(MAPPING.num_tables(1, 0), 0);
        assert_eq!(MAPPING.num_tables(1, MAPPING.span(1)), 1);
        assert_eq!(MAPPING.num_tables(1, 2 * MAPPING.span(1) + 1), 3);
        assert_eq!(
            MAPPING.num_tables(1, MAPPING.span(3)),
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

        assert_eq!(MAPPING.frame(1, 15, 2), 2);
        assert_eq!(MAPPING.frame(1, 512, 2), 512 + 2);
        assert_eq!(MAPPING.frame(1, MAPPING.span(2), 0), MAPPING.span(2));
        assert_eq!(
            MAPPING.frame(2, MAPPING.span(2), 1),
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

        assert_eq!(MAPPING.frame(1, 15, 2), 2);
        assert_eq!(MAPPING.frame(1, 512, 2), 512 + 2);
        assert_eq!(MAPPING.frame(1, MAPPING.span(2), 0), MAPPING.span(2));
        assert_eq!(
            MAPPING.frame(2, MAPPING.span(2), 1),
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
    fn bit_set() {
        let bitfield = super::Bitfield::<2>::default();
        bitfield.set(0..0, true);
        assert_eq!(bitfield.get_entry(0), 0);
        assert_eq!(bitfield.get_entry(1), 0);
        bitfield.set(0..1, true);
        assert_eq!(bitfield.get_entry(0), 0b1);
        assert_eq!(bitfield.get_entry(1), 0b0);
        bitfield.set(0..1, false);
        assert_eq!(bitfield.get_entry(0), 0b0);
        assert_eq!(bitfield.get_entry(1), 0b0);
        bitfield.set(0..2, true);
        assert_eq!(bitfield.get_entry(0), 0b11);
        assert_eq!(bitfield.get_entry(1), 0b0);
        bitfield.set(2..56, true);
        assert_eq!(bitfield.get_entry(0), 0x00ff_ffff_ffff_ffff);
        assert_eq!(bitfield.get_entry(1), 0b0);
        bitfield.set(60..73, true);
        assert_eq!(bitfield.get_entry(0), 0xf0ff_ffff_ffff_ffff);
        assert_eq!(bitfield.get_entry(1), 0x01ff);
        bitfield.set(96..128, true);
        assert_eq!(bitfield.get_entry(0), 0xf0ff_ffff_ffff_ffff);
        assert_eq!(bitfield.get_entry(1), 0xffff_ffff_0000_01ff);
        bitfield.set(0..128, false);
        assert_eq!(bitfield.get_entry(0), 0);
        assert_eq!(bitfield.get_entry(1), 0);
    }

    #[test]
    fn bit_toggle() {
        let bitfield = super::Bitfield::<2>::default();
        bitfield.toggle(8, 3, false).unwrap();
        assert_eq!(bitfield.get_entry(0), 0xff00);
        assert_eq!(bitfield.get_entry(1), 0);
        bitfield.toggle(16, 2, false).unwrap();
        assert_eq!(bitfield.get_entry(0), 0xfff00);
        assert_eq!(bitfield.get_entry(1), 0);
        bitfield.toggle(20, 2, false).unwrap();
        assert_eq!(bitfield.get_entry(0), 0xffff00);
        assert_eq!(bitfield.get_entry(1), 0);
        bitfield.toggle(8, 3, false).expect_err("");
        bitfield.toggle(8, 3, true).unwrap();
        bitfield.toggle(16, 3, true).unwrap();
        assert_eq!(bitfield.get_entry(0), 0);
        assert_eq!(bitfield.get_entry(1), 0);
        bitfield.toggle(0, 6, false).unwrap();
        assert_eq!(bitfield.get_entry(0), u64::MAX);
        assert_eq!(bitfield.get_entry(1), 0);
        bitfield.toggle(64, 6, false).unwrap();
        assert_eq!(bitfield.get_entry(0), u64::MAX);
        assert_eq!(bitfield.get_entry(1), u64::MAX);
        bitfield.toggle(0, 7, true).unwrap();
        assert_eq!(bitfield.get_entry(0), 0);
        assert_eq!(bitfield.get_entry(1), 0);
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

    #[test]
    fn first_zeros_aligned_1() {
        assert_eq!(super::first_zeros_aligned(0b0, 1), Some((0b11, 0)));
        assert_eq!(super::first_zeros_aligned(0b1, 1), Some((0b1101, 2)));
        assert_eq!(super::first_zeros_aligned(0b10011, 1), Some((0b11111, 2)));
        assert_eq!(
            super::first_zeros_aligned(0b0001_1001_1011, 1),
            Some((0b1101_1001_1011, 10))
        );
    }

    #[test]
    fn first_zeros_aligned_2() {
        assert_eq!(super::first_zeros_aligned(0b0, 2), Some((0b1111, 0)));
        assert_eq!(super::first_zeros_aligned(0b1, 2), Some((0b11110001, 4)));
        assert_eq!(
            super::first_zeros_aligned(0b11_0000_0001, 2),
            Some((0b11_1111_0001, 4))
        );
        assert_eq!(
            super::first_zeros_aligned(0b0000_0100_1000_0001_0010, 2),
            Some((0b1111_0100_1000_0001_0010, 16))
        );
    }

    #[test]
    fn first_zero_entries() {
        let bitfield = super::Bitfield::<8>::default();

        // 9
        assert!(bitfield.data.iter().all(|e| e.load() == 0));
        assert_eq!(0, bitfield.set_first_zeros(0, 9).unwrap());
        assert!(bitfield.data.iter().all(|e| e.load() == u64::MAX));
        bitfield.toggle(0, 9, true).unwrap();
        assert!(bitfield.data.iter().all(|e| e.load() == 0));

        assert_eq!(0, bitfield.set_first_zeros(0, 7).unwrap());
        assert!(bitfield.data[0..2].iter().all(|e| e.load() == u64::MAX));

        assert_eq!(4 * 64, bitfield.set_first_zeros(0, 8).unwrap());
        assert!(bitfield.data[4..8].iter().all(|e| e.load() == u64::MAX));

        assert_eq!(2 * 64, bitfield.set_first_zeros(0, 6).unwrap());
        assert!(bitfield.get_entry(2) == u64::MAX);
        assert_eq!(3 * 64, bitfield.set_first_zeros(0, 6).unwrap());
        assert!(bitfield.get_entry(3) == u64::MAX);

        bitfield.set_first_zeros(0, 9).expect_err("no mem");
        bitfield.set_first_zeros(0, 8).expect_err("no mem");
        bitfield.set_first_zeros(0, 7).expect_err("no mem");
        bitfield.set_first_zeros(0, 6).expect_err("no mem");
    }
}
