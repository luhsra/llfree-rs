//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.
#![feature(panic_info_message)]
#![feature(const_maybe_uninit_zeroed)]

pub mod alloc;
pub mod entry;
mod leaf_alloc;
pub mod mmap;
pub mod table;
pub mod thread;
pub mod util;

#[cfg(feature = "wait")]
pub mod wait;

use std::ffi::c_void;
use std::sync::atomic::AtomicU64;

use alloc::{Alloc, Allocator, Error, Size};
use util::Page;

#[no_mangle]
pub extern "C" fn nvalloc_init(cores: u32, addr: *mut c_void, pages: u64, overwrite: u32) -> i64 {
    let memory = unsafe { std::slice::from_raw_parts_mut(addr as *mut Page, pages as _) };
    match Allocator::init(cores as _, memory, overwrite != 0) {
        Ok(_) => 0,
        Err(e) => -(e as usize as i64),
    }
}

#[no_mangle]
pub extern "C" fn nvalloc_uninit() {
    Allocator::uninit();
}

#[no_mangle]
pub extern "C" fn nvalloc_get(core: u32, size: u32) -> i64 {
    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        2 => Size::L2,
        _ => return -(Error::Memory as usize as i64),
    };

    let alloc = Allocator::instance();
    match alloc.get(core as _, size) {
        Ok(addr) => addr as i64,
        Err(e) => -(e as usize as i64),
    }
}

#[no_mangle]
pub extern "C" fn nvalloc_get_cas(
    core: u32,
    size: u32,
    dst: *const u64,
    translate: extern "C" fn(u64) -> u64,
    expected: u64,
) -> i64 {
    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        2 => Size::L2,
        _ => return -(Error::Memory as usize as i64),
    };

    let dst = unsafe { &*(dst as *const AtomicU64) };
    let alloc = Allocator::instance();

    match alloc.get_cas(core as _, size, dst, |p| translate(p), expected) {
        Ok(_) => 0,
        Err(e) => -(e as usize as i64),
    }
}

#[no_mangle]
pub extern "C" fn nvalloc_put(core: u32, addr: u64) -> i64 {
    match Allocator::instance().put(core as _, addr) {
        Ok(_) => 0,
        Err(e) => -(e as usize as i64),
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicU64, Ordering};

    use log::info;

    use crate::mmap::MMap;
    use crate::table::Table;
    use crate::thread::parallel;
    use crate::util::logging;
    use crate::{Alloc, Allocator, Page, Size};

    #[test]
    fn threading() {
        logging();

        const THREADS: usize = 8;

        let mut mapping: MMap<Page> = MMap::anon(0x1000_0000_0000_u64 as _, 20 << 18).unwrap();

        info!("mmap {} bytes", mapping.len());

        info!("init alloc");

        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        Allocator::init(THREADS, &mut mapping[..], true).unwrap();

        parallel(THREADS, |t| {
            let pages = [DEFAULT; Table::LEN];
            for addr in &pages {
                Allocator::instance()
                    .get_cas(t, Size::L0, addr, |v| v, 0)
                    .unwrap();
            }

            for addr in &pages {
                Allocator::instance()
                    .put(t, addr.load(Ordering::SeqCst))
                    .unwrap();
            }
        });

        info!("Finish");
    }
}
