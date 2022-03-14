//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.
pub mod alloc;
pub mod entry;
pub mod lower;
pub mod mmap;
pub mod table;
pub mod thread;
pub mod util;

#[cfg(feature = "stop")]
pub mod stop;

use core::ffi::c_void;
use core::sync::atomic::AtomicU64;
use std::sync::Arc;

use alloc::{Alloc, Error, Size};
use lower::dynamic::DynamicLower;
use util::Page;

pub type Allocator = alloc::array_atomic::ArrayAtomicAlloc<DynamicLower>;
static mut ALLOC: Option<Arc<dyn Alloc>> = None;

pub fn init(cores: usize, memory: &mut [Page], overwrite: bool) -> alloc::Result<()> {
    let mut alloc = Allocator::new();
    alloc.init(cores, memory, overwrite)?;
    unsafe { ALLOC = Some(Arc::new(alloc)) };
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
    let memory = unsafe { std::slice::from_raw_parts_mut(addr as *mut Page, pages as _) };
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
        2 => Size::L2,
        _ => return -(Error::Memory as usize as i64),
    };

    match instance().get(core as _, size) {
        Ok(addr) => addr as i64,
        Err(e) => -(e as usize as i64),
    }
}

/// Allocate and atomically insert the page into the given `dst`.
/// Returns an error if the allocation or the CAS operation fail.
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

    match alloc::get_cas(instance(), core as _, size, dst, |p| translate(p), expected) {
        Ok(_) => 0,
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
    use std::sync::atomic::{AtomicU64, Ordering};

    use log::info;

    use crate::mmap::MMap;
    use crate::table::Table;
    use crate::thread::parallel;
    use crate::util::logging;
    use crate::{alloc, init, instance, uninit};
    use crate::{Page, Size};

    #[test]
    fn threading() {
        logging();

        const THREADS: usize = 8;

        let mut mapping: MMap<Page> = MMap::anon(0x1000_0000_0000_u64 as _, 20 << 18).unwrap();

        info!("mmap {} bytes", mapping.len());

        info!("init alloc");

        const DEFAULT: AtomicU64 = AtomicU64::new(0);

        init(THREADS, &mut mapping[..], true).unwrap();

        parallel(THREADS, |t| {
            let pages = [DEFAULT; Table::LEN];
            for addr in &pages {
                alloc::get_cas(instance(), t, Size::L0, addr, |v| v, 0).unwrap();
            }

            for addr in &pages {
                instance().put(t, addr.load(Ordering::SeqCst)).unwrap();
            }
        });

        uninit();

        info!("Finish");
    }
}
