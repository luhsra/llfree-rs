//! Page frame utilities

use core::mem::{align_of, size_of, transmute};

pub const PT_ORDER: usize = 9;
pub const PT_LEN: usize = 1 << PT_ORDER;

/// Correctly sized and aligned page frame.
#[derive(Clone)]
#[repr(align(0x1000))]
pub struct Frame {
    _data: [u8; Self::SIZE],
}
const _: () = assert!(size_of::<Frame>() == Frame::SIZE);
const _: () = assert!(align_of::<Frame>() == Frame::SIZE);
impl Frame {
    pub const SIZE: usize = 0x1000;
    pub const SIZE_BITS: usize = Self::SIZE.ilog2() as _;
    pub const fn new() -> Self {
        Self {
            _data: [0; Self::SIZE],
        }
    }
    pub fn cast<T>(&self) -> &T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { transmute(self) }
    }
    pub fn cast_mut<T>(&mut self) -> &mut T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { transmute(self) }
    }
}
