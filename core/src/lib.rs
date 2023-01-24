//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.
#![no_std]
#![feature(new_uninit)]
#![feature(int_roundings)]
#![feature(array_windows)]
#![feature(generic_const_exprs)]
#![feature(associated_type_bounds)]
// Don't warn for compile-time checks
#![allow(clippy::assertions_on_constants)]

use core::fmt;
use core::mem::{align_of, size_of, transmute};
use core::ops::Range;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
pub mod mmap;
#[cfg(feature = "std")]
pub mod thread;

pub mod atomic;
pub mod entry;
pub mod lower;
pub mod table;
pub mod upper;
pub mod util;

#[cfg(all(test, feature = "stop"))]
mod stop;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory = 1,
    /// Failed atomic operation, retry procedure
    Retry = 2,
    /// Invalid address
    Address = 3,
    /// Allocator not initialized or initialization failed
    Initialization = 4,
    /// Corrupted allocator state
    Corruption = 5,
}

pub type Result<T> = core::result::Result<T, Error>;

/// Page frame number that is convertible to pointers
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct PFN(pub usize);

impl PFN {
    pub fn from_ptr(ptr: *const Page) -> Self {
        Self(ptr as usize / Page::SIZE)
    }
    pub fn as_ptr(self) -> *const Page {
        (self.0 * Page::SIZE) as _
    }
    pub fn as_ptr_mut(self) -> *mut Page {
        (self.0 * Page::SIZE) as _
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
impl From<*const Page> for PFN {
    fn from(value: *const Page) -> Self {
        Self::from_ptr(value)
    }
}
impl From<PFN> for *const Page {
    fn from(value: PFN) -> Self {
        value.as_ptr()
    }
}

impl fmt::Display for PFN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}
impl fmt::Debug for PFN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}

/// This wrapper exists as its unfortunately impossible to implement iterator for custom range types
pub trait PFNRange: Sized {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn as_range(&self) -> Range<usize>;
    fn as_ptr_range(&self) -> Range<*const Page> {
        let Range { start, end } = self.as_range();
        (start * Page::SIZE) as _..(end * Page::SIZE) as _
    }
}

pub fn pfn_range(slice: &[Page]) -> Range<PFN> {
    let Range { start, end } = slice.as_ptr_range();
    start.into()..end.into()
}

impl PFNRange for Range<PFN> {
    fn len(&self) -> usize {
        self.end.0.saturating_sub(self.start.0)
    }
    fn as_range(&self) -> Range<usize> {
        self.start.into()..self.end.into()
    }
}

/// Correctly sized and aligned page.
#[derive(Clone)]
#[repr(align(0x1000))]
pub struct Page {
    _data: [u8; Self::SIZE],
}
const _: () = assert!(size_of::<Page>() == Page::SIZE);
const _: () = assert!(align_of::<Page>() == Page::SIZE);
impl Page {
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
