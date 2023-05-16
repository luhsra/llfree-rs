use core::fmt;
use core::mem::{align_of, size_of, transmute};
use core::ops::Range;

/// Page frame number that is convertible to pointers
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct PFN(pub usize);

impl PFN {
    pub fn from_ptr(ptr: *const Frame) -> Self {
        Self(ptr as usize / Frame::SIZE)
    }
    pub fn as_ptr(self) -> *const Frame {
        (self.0 * Frame::SIZE) as _
    }
    pub fn as_ptr_mut(self) -> *mut Frame {
        (self.0 * Frame::SIZE) as _
    }
    pub fn off(self, offset: usize) -> Self {
        Self(self.0 + offset)
    }
}

impl From<usize> for PFN {
    fn from(value: usize) -> Self {
        Self(value)
    }
}
impl From<PFN> for usize {
    fn from(value: PFN) -> Self {
        value.0
    }
}

impl fmt::Display for PFN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}
impl fmt::Debug for PFN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// This wrapper exists as its unfortunately impossible to implement iterator for custom range types
pub trait PFNRange: Sized {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn as_range(&self) -> Range<usize>;
    fn as_ptr_range(&self) -> Range<*const Frame> {
        let Range { start, end } = self.as_range();
        (start * Frame::SIZE) as _..(end * Frame::SIZE) as _
    }
}

pub fn pfn_range(slice: &[Frame]) -> Range<PFN> {
    let Range { start, end } = slice.as_ptr_range();
    PFN::from_ptr(start)..PFN::from_ptr(end)
}

impl PFNRange for Range<PFN> {
    fn len(&self) -> usize {
        self.end.0.saturating_sub(self.start.0)
    }
    fn as_range(&self) -> Range<usize> {
        self.start.into()..self.end.into()
    }
}

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
