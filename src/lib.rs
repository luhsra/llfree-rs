//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.
#![no_std]
#![feature(generic_const_exprs)]

// Custom out of memory handler
#![cfg_attr(not(feature = "std"), feature(alloc_error_handler))]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

#[macro_use]
extern crate alloc;

#[cfg(any(test, feature = "std"))]
pub mod mmap;
#[cfg(any(test, feature = "thread"))]
pub mod thread;

pub mod atomic;
pub mod entry;
pub mod lower;
pub mod table;
pub mod upper;
pub mod util;

#[cfg(feature = "stop")]
pub mod stop;

#[cfg(not(any(test, feature = "std")))]
mod linux;

use core::ffi::c_void;

use alloc::boxed::Box;
use upper::{Alloc, Error, Size};
use util::Page;

pub type Allocator = upper::ArrayAtomicAlloc<lower::CacheLower<128>>;
static mut ALLOC: Option<Box<Allocator>> = None;

pub fn init(cores: usize, memory: &mut [Page], overwrite: bool) -> upper::Result<()> {
    let mut alloc = Allocator::default();
    alloc.init(cores, memory, overwrite)?;
    unsafe { ALLOC = Some(Box::new(alloc)) };
    Ok(())
}

pub fn instance<'a>() -> &'a dyn Alloc {
    unsafe { ALLOC.as_ref().unwrap().as_ref() }
}

pub fn uninit() {
    unsafe { ALLOC.take().unwrap() };
}

// C bindings

/// Initialize the allocator for the given memory range.
/// If `overwrite` is nonzero no existing allocator state is recovered.
#[no_mangle]
pub extern "C" fn nvalloc_init(cores: u32, addr: *mut c_void, pages: u64, overwrite: u32) -> i64 {
    let memory = unsafe { core::slice::from_raw_parts_mut(addr.cast(), pages as _) };
    match init(cores as _, memory, overwrite != 0) {
        Ok(_) => 0,
        Err(e) => -(e as usize as i64),
    }
}

/// Shut down the allocator normally.
#[no_mangle]
pub extern "C" fn nvalloc_uninit() {
    uninit();
}

/// Allocate a page of the given `size` on the given cpu `core`.
#[no_mangle]
pub extern "C" fn nvalloc_get(core: u32, size: u32) -> i64 {
    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        _ => return -(Error::Memory as usize as i64),
    };

    match instance().get(core as _, size) {
        Ok(addr) => addr as i64,
        Err(e) => -(e as usize as i64),
    }
}

/// Frees the given page.
#[no_mangle]
pub extern "C" fn nvalloc_put(core: u32, addr: u64) -> i64 {
    match instance().put(core as _, addr) {
        Ok(_) => 0,
        Err(e) => -(e as usize as i64),
    }
}

#[cfg(test)]
mod test {
    use core::sync::atomic::{AtomicU64, Ordering};

    use log::info;

    use crate::mmap::MMap;
    use crate::table::PT_LEN;
    use crate::thread::parallel;
    use crate::util::logging;
    use crate::{init, instance, uninit};
    use crate::{Page, Size};

    #[test]
    fn threading() {
        logging();

        const THREADS: usize = 8;
        const PAGES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        let mut mapping: MMap<Page> = MMap::anon(0x0000_1000_0000_0000, PAGES).unwrap();

        info!("mmap {} bytes", mapping.len());

        info!("init alloc");

        const DEFAULT: AtomicU64 = AtomicU64::new(0);

        init(THREADS, &mut mapping[..], true).unwrap();

        parallel(THREADS, |t| {
            let pages = [DEFAULT; PT_LEN];
            for addr in &pages {
                addr.store(instance().get(t, Size::L0).unwrap(), Ordering::SeqCst);
            }

            for addr in &pages {
                instance().put(t, addr.load(Ordering::SeqCst)).unwrap();
            }
        });

        uninit();

        info!("Finish");
    }
}
