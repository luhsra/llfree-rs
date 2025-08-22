//! Atomic bitfield

use core::fmt;
use core::mem::size_of;
use core::ops::{Not, Range};
use core::sync::atomic::AtomicU64;

use log::warn;

use crate::atomic::{Atom, Atomic};
use crate::{Error, Result};

/// Bitfield replacing the level one table.
pub struct Bitfield<const N: usize> {
    data: [Atom<u64>; N],
}

const _: () = assert!(size_of::<Bitfield<64>>() >= 8);
const _: () = assert!(Bitfield::<64>::LEN.is_multiple_of(Bitfield::<64>::ROW_BITS));
const _: () = assert!(1 << Bitfield::<64>::ORDER == Bitfield::<64>::LEN);
const _: () = assert!(Bitfield::<2>::ORDER == 7);

impl<const N: usize> Default for Bitfield<N> {
    fn default() -> Self {
        Self {
            data: [const { Atom(AtomicU64::new(0)) }; N],
        }
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

impl<const N: usize> Bitfield<N> {
    pub const ROW_BITS: usize = 64;
    pub const ROWS: usize = N;
    pub const LEN: usize = N * Self::ROW_BITS;
    pub const ORDER: usize = Self::LEN.ilog2() as _;

    /// Overwrite the `range` of bits with `v`
    pub fn set(&self, range: Range<usize>, v: bool) {
        assert!(range.start <= range.end && range.end <= Self::LEN);

        if range.start != range.end {
            for ei in range.start / Self::ROW_BITS..=(range.end - 1) / Self::ROW_BITS {
                let bit_off = ei * Self::ROW_BITS;
                let bit_start = range.start.saturating_sub(bit_off);
                let bit_end = (range.end - bit_off).min(Self::ROW_BITS);
                let bits = bit_end - bit_start;
                let byte = (u64::MAX >> (Self::ROW_BITS - bits)) << bit_start;
                if v {
                    self.data[ei].fetch_or(byte);
                } else {
                    self.data[ei].fetch_and(!byte);
                }
            }
        }
    }

    /// Return the  `i`-th row
    pub fn get_row(&self, i: usize) -> u64 {
        self.data[i].load()
    }

    /// Toggle 2^`order` bits at the `i`-th place if they are all zero or one as expected
    ///
    /// # Warning
    /// Orders above 6 need multiple CAS operations, which might lead to race conditions!
    pub fn toggle(&self, i: usize, order: usize, expected: bool) -> Result<()> {
        let num_bits = 1 << order;
        debug_assert!(i.is_multiple_of(num_bits), "not aligned");
        match order {
            0..=2 => {
                // Updates within a single row
                let mask = (u64::MAX >> (Self::ROW_BITS - num_bits)) << (i % Self::ROW_BITS);
                let di = i / Self::ROW_BITS;
                match self.data[di].fetch_update(|e| {
                    if expected {
                        (e & mask == mask).then_some(e & !mask)
                    } else {
                        (e & mask == 0).then_some(e | mask)
                    }
                }) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(Error::Address),
                }
            }
            3 => self.toggle_int::<u8>(i, expected),
            4 => self.toggle_int::<u16>(i, expected),
            5 => self.toggle_int::<u32>(i, expected),
            6 => self.toggle_int::<u64>(i, expected),
            _ => {
                // Update multiple rows
                let num_rows = num_bits / Self::ROW_BITS;
                let di = i / Self::ROW_BITS;
                for i in di..di + num_rows {
                    let expected = if expected { !0 } else { 0 };
                    if let Err(e) = self.data[i].compare_exchange(expected, !expected) {
                        warn!("Toggle failed {e:x} != {expected:x}");

                        // Undo changes
                        for j in (di..i).rev() {
                            self.data[j]
                                .compare_exchange(!expected, expected)
                                .expect("Failed undo toggle");
                        }
                        return Err(Error::Address);
                    }
                }
                Ok(())
            }
        }
    }

    /// Toggle multiple bits with a single correctly sized compare exchange operation
    ///
    /// Note: This only seems to make a difference between a 64 bit fetch_update on Intel Optane
    fn toggle_int<I: Atomic + Default + Not<Output = I>>(&self, i: usize, e: bool) -> Result<()> {
        assert!(i < Self::LEN);
        debug_assert!(size_of::<I>() <= Self::ROW_BITS / 8);

        let idx = i / (8 * size_of::<I>());
        let val = if e { !I::default() } else { I::default() };
        // Safety: I is guaranteed to be smaller than u64 and i is in bounds, so is idx
        debug_assert!(idx * size_of::<I>() <= size_of::<Self>());
        // Safety: Cast to smaller type atomic, keeping the same total bitfield size
        let atom = unsafe { &*self.data.as_ptr().cast::<Atom<I>>().add(idx) };
        match atom.compare_exchange(val, !val) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Retry),
        }
    }

    pub fn is_zero(&self, i: usize, order: usize) -> bool {
        let num_bits = 1 << order;
        debug_assert!(i < Self::LEN && order <= Self::ORDER);
        debug_assert!(i.is_multiple_of(num_bits), "not aligned");

        let row_i = i / Self::ROW_BITS;
        if num_bits > Self::ROW_BITS {
            let end_i = (i + num_bits) / Self::ROW_BITS;
            (row_i..end_i).all(|i| self.get_row(i) == 0)
        } else {
            let row = self.get_row(row_i);
            let mask = (u64::MAX >> (u64::BITS as usize - num_bits)) << (i % Self::ROW_BITS);
            (row & mask) == 0
        }
    }

    /// Set the first aligned 2^`order` zero bits, returning the bit offset
    ///
    /// # Warning
    /// Orders above 6 need multiple CAS operations, which might lead to race conditions!
    pub fn set_first_zeros(&self, start_row: usize, order: usize) -> Result<usize> {
        debug_assert!(start_row < Self::ROWS);

        if order > Self::ROW_BITS.ilog2() as usize {
            return self.set_first_zero_rows(order);
        }

        for i in 0..self.data.len() {
            let i = (i + start_row) % self.data.len();

            let mut offset = 0;
            if let Ok(_) = self.data[i].fetch_update(|e| {
                let (val, o) = first_zeros_aligned(e, order)?;
                offset = o;
                Some(val)
            }) {
                return Ok(i * Self::ROW_BITS + offset);
            }
        }
        Err(Error::Memory)
    }

    /// Allocate multiple rows with multiple CAS
    ///
    /// # Warning
    /// Using multiple CAS operations might lead to race conditions!
    fn set_first_zero_rows(&self, order: usize) -> Result<usize> {
        debug_assert!(order > Self::ROW_BITS.ilog2() as usize);
        debug_assert!(order <= Self::ORDER);

        let num_rows = 1 << (order - Self::ROW_BITS.ilog2() as usize);

        for (i, rows) in self.data.chunks(num_rows).enumerate() {
            // Check that these rows are free
            if rows.iter().all(|e| e.load() == 0) {
                for (j, row) in rows.iter().enumerate() {
                    if let Err(_) = row.compare_exchange(0, u64::MAX) {
                        // Undo previous updates
                        for k in (0..j).rev() {
                            rows[k]
                                .compare_exchange(u64::MAX, 0)
                                .expect("Failed undo search");
                        }
                        break;
                    }
                }
                return Ok(i * num_rows * Self::ROW_BITS);
            }
        }
        Err(Error::Memory)
    }

    /// Fill this bitset with `v` ignoring any previous data.
    pub fn fill(&self, v: bool) {
        let v = if v { u64::MAX } else { 0 };
        for row in &self.data {
            row.store(v);
        }
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
///
/// - See <https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord>
fn first_zeros_aligned(v: u64, order: usize) -> Option<(u64, usize)> {
    match order {
        0 => {
            let off = v.trailing_ones();
            (off < u64::BITS).then(|| (v | (0b1 << off), off as _))
        }
        1 => {
            let mask = 0xaaaa_aaaa_aaaa_aaaa_u64;
            let off = ((v | (v >> 1)) | mask).trailing_ones();
            (off < u64::BITS).then(|| (v | (0b11 << off), off as _))
        }
        2 => {
            let mask = 0x1111_1111_1111_1111_u64;
            let off = (((v.wrapping_sub(mask) & !v) >> 3) & mask).trailing_zeros();
            (off < u64::BITS).then(|| (v | (0b1111 << off), off as _))
        }
        3 => {
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
        6 => (v == 0).then_some((u64::MAX, 0)),
        // All other orders are handled differently
        _ => unreachable!(),
    }
}

#[cfg(all(test, feature = "std"))]
mod test {

    #[test]
    fn bit_set() {
        let bitfield = super::Bitfield::<2>::default();
        bitfield.set(0..0, true);
        assert_eq!(bitfield.get_row(0), 0);
        assert_eq!(bitfield.get_row(1), 0);
        bitfield.set(0..1, true);
        assert_eq!(bitfield.get_row(0), 0b1);
        assert_eq!(bitfield.get_row(1), 0b0);
        bitfield.set(0..1, false);
        assert_eq!(bitfield.get_row(0), 0b0);
        assert_eq!(bitfield.get_row(1), 0b0);
        bitfield.set(0..2, true);
        assert_eq!(bitfield.get_row(0), 0b11);
        assert_eq!(bitfield.get_row(1), 0b0);
        bitfield.set(2..56, true);
        assert_eq!(bitfield.get_row(0), 0x00ff_ffff_ffff_ffff);
        assert_eq!(bitfield.get_row(1), 0b0);
        bitfield.set(60..73, true);
        assert_eq!(bitfield.get_row(0), 0xf0ff_ffff_ffff_ffff);
        assert_eq!(bitfield.get_row(1), 0x01ff);
        bitfield.set(96..128, true);
        assert_eq!(bitfield.get_row(0), 0xf0ff_ffff_ffff_ffff);
        assert_eq!(bitfield.get_row(1), 0xffff_ffff_0000_01ff);
        bitfield.set(0..128, false);
        assert_eq!(bitfield.get_row(0), 0);
        assert_eq!(bitfield.get_row(1), 0);
    }

    #[test]
    fn bit_toggle() {
        let bitfield = super::Bitfield::<2>::default();

        assert!(bitfield.is_zero(8, 3));
        bitfield.toggle(8, 3, false).unwrap();
        assert_eq!(bitfield.get_row(0), 0xff00);
        assert_eq!(bitfield.get_row(1), 0);
        assert!(!bitfield.is_zero(8, 3));

        assert!(bitfield.is_zero(16, 2));
        bitfield.toggle(16, 2, false).unwrap();
        assert_eq!(bitfield.get_row(0), 0xfff00);
        assert_eq!(bitfield.get_row(1), 0);
        assert!(!bitfield.is_zero(16, 2));

        assert!(bitfield.is_zero(20, 2));
        bitfield.toggle(20, 2, false).unwrap();
        assert_eq!(bitfield.get_row(0), 0xffff00);
        assert_eq!(bitfield.get_row(1), 0);
        assert!(!bitfield.is_zero(16, 2));

        assert!(!bitfield.is_zero(8, 3));
        bitfield.toggle(8, 3, false).expect_err("");
        bitfield.toggle(8, 3, true).unwrap();
        bitfield.toggle(16, 3, true).unwrap();
        assert_eq!(bitfield.get_row(0), 0);
        assert_eq!(bitfield.get_row(1), 0);
        assert!(bitfield.is_zero(0, super::Bitfield::<2>::ORDER));

        bitfield.toggle(0, 6, false).unwrap();
        assert_eq!(bitfield.get_row(0), u64::MAX);
        assert_eq!(bitfield.get_row(1), 0);
        bitfield.toggle(64, 6, false).unwrap();
        assert_eq!(bitfield.get_row(0), u64::MAX);
        assert_eq!(bitfield.get_row(1), u64::MAX);
        bitfield.toggle(0, 7, true).unwrap();
        assert_eq!(bitfield.get_row(0), 0);
        assert_eq!(bitfield.get_row(1), 0);
    }

    #[test]
    fn first_zeros_aligned() {
        use super::first_zeros_aligned as fza;

        assert_eq!(fza(0b0, 0), Some((0b1, 0)));
        assert_eq!(fza(0b0, 1), Some((0b11, 0)));
        assert_eq!(fza(0b0, 2), Some((0b1111, 0)));
        assert_eq!(fza(0b0, 3), Some((0xff, 0)));
        assert_eq!(fza(0b0, 4), Some((0xffff, 0)));
        assert_eq!(fza(0b0, 5), Some((0xffffffff, 0)));
        assert_eq!(fza(0b0, 6), Some((0xffffffffffffffff, 0)));

        assert_eq!(fza(0b1, 0), Some((0b11, 1)));
        assert_eq!(fza(0b1, 1), Some((0b1101, 2)));
        assert_eq!(fza(0b1, 2), Some((0xf1, 4)));
        assert_eq!(fza(0b1, 3), Some((0xff01, 8)));
        assert_eq!(fza(0b1, 4), Some((0xffff0001, 16)));
        assert_eq!(fza(0b1, 5), Some((0xffffffff00000001, 32)));
        assert_eq!(fza(0b1, 6), None);

        assert_eq!(fza(0b101, 0), Some((0b111, 1)));
        assert_eq!(fza(0b10011, 1), Some((0b11111, 2)));
        assert_eq!(fza(0x10f, 2), Some((0x1ff, 4)));
        assert_eq!(fza(0x100ff, 3), Some((0x1ffff, 8)));
        assert_eq!(fza(0x10000ffff, 4), Some((0x1ffffffff, 16)));
        assert_eq!(fza(0x00000000ff00ff0f, 5), Some((0xffffffffff00ff0f, 32)));
        assert_eq!(
            fza(0b1111_0000_1100_0011_1000_1111, 2),
            Some((0b1111_1111_1100_0011_1000_1111, 16))
        );

        // Upper bound
        assert_eq!(fza(0x7fffffffffffffff, 0), Some((0xffffffffffffffff, 63)));
        assert_eq!(fza(0xffffffffffffffff, 0), None);

        assert_eq!(fza(0x3fffffffffffffff, 1), Some((0xffffffffffffffff, 62)));
        assert_eq!(fza(0x7fffffffffffffff, 1), None);

        assert_eq!(fza(0x0fffffffffffffff, 2), Some((0xffffffffffffffff, 60)));
        assert_eq!(fza(0x1fffffffffffffff, 2), None);
        assert_eq!(fza(0x3fffffffffffffff, 2), None);

        assert_eq!(fza(0x00ffffffffffffff, 3), Some((0xffffffffffffffff, 56)));
        assert_eq!(fza(0x0fffffffffffffff, 3), None);
        assert_eq!(fza(0x1fffffffffffffff, 3), None);

        assert_eq!(fza(0x0000ffffffffffff, 4), Some((0xffffffffffffffff, 48)));
        assert_eq!(fza(0x0001ffffffffffff, 4), None);
        assert_eq!(fza(0x00ffffffffffffff, 4), None);

        assert_eq!(fza(0x00000000ffffffff, 5), Some((0xffffffffffffffff, 32)));
        assert_eq!(fza(0x00000001ffffffff, 5), None);
        assert_eq!(fza(0x0000ffffffffffff, 5), None);

        assert_eq!(fza(0, 6), Some((0xffffffffffffffff, 0)));
        assert_eq!(fza(1, 6), None);
        assert_eq!(fza(0xa000000000000000, 6), None);
    }

    #[test]
    fn first_zero_rows() {
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
        assert!(bitfield.get_row(2) == u64::MAX);
        assert_eq!(3 * 64, bitfield.set_first_zeros(0, 6).unwrap());
        assert!(bitfield.get_row(3) == u64::MAX);

        bitfield.set_first_zeros(0, 9).expect_err("no mem");
        bitfield.set_first_zeros(0, 8).expect_err("no mem");
        bitfield.set_first_zeros(0, 7).expect_err("no mem");
        bitfield.set_first_zeros(0, 6).expect_err("no mem");
    }
}
