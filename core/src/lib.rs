//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.

#![no_std]
#![feature(new_uninit)]
#![feature(int_roundings)]
#![feature(array_windows)]
#![feature(generic_const_exprs)]
#![feature(associated_type_bounds)]
#![feature(inline_const)]
#![feature(allocator_api)]
#![feature(let_chains)]
// Don't warn for compile-time checks
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::redundant_pattern_matching)]

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
pub mod bitfield;
pub mod entry;
pub mod frame;
pub mod lower;
pub mod upper;
pub mod util;

/// Allocation error
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

/// Allocation result
pub type Result<T> = core::result::Result<T, Error>;
