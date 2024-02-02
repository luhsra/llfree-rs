//! Page frame utilities

use core::mem::{align_of, size_of, transmute};

use crate::lower::Lower;

//pub const PT_ORDER: usize = 9;
pub const PT_LEN: usize = 1 << Lower::HUGE_ORDER;//PT_ORDER;

/// Correctly sized and aligned page frame.
#[derive(Clone)]
// for 16K needs to be 0x4000, for 4K needs to be 0x1000 
#[cfg_attr(feature="16K", repr(align(0x4000)))]
#[cfg_attr(not(feature="16K"), repr(align(0x1000)))]
//#[repr(align(0x4000))] 

pub struct Frame {
    _data: [u8; Self::SIZE],
}
const _: () = assert!(size_of::<Frame>() == Frame::SIZE);
//const _: () = assert!(align_of::<Frame>() == Frame::SIZE);
impl Frame {
    #[cfg(not(feature="16K"))]
    pub const SIZE: usize = 0x1000; // 4KiB
    #[cfg(feature="16K")]
    pub const SIZE: usize = 0x4000; // 16KiB 

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
