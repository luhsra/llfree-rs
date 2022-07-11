//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.
#![no_std]
#![feature(generic_const_exprs)]
#![feature(new_uninit)]

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

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

#[cfg(feature = "stop")]
mod stop;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory = 1,
    /// Failed comapare and swap operation
    CAS = 2,
    /// Invalid address
    Address = 3,
    /// Allocator not initialized or initialization failed
    Initialization = 4,
    /// Corrupted allocator state
    Corruption = 5,
}

pub type Result<T> = core::result::Result<T, Error>;
