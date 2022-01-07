//! Simple reduced alloc example.
#![feature(asm)]
#![feature(panic_info_message)]

use std::{alloc::Layout, sync::atomic::AtomicUsize};

mod alloc;
pub mod entry;
pub use alloc::{Alloc, Allocator};
mod leaf_alloc;
pub mod mmap;
pub mod table;
pub mod thread;
pub mod util;

#[cfg(test)]
mod wait;

use table::Table;

pub const MAGIC: usize = 0xdeadbeef;
pub const MIN_PAGES: usize = 2 * Table::span(2);
pub const MAX_PAGES: usize = Table::span(Table::LAYERS);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory = 1,
    /// Failed comapare and swap operation
    CAS = 2,
    /// Invalid address
    Address = 3,
    /// Allocator not initialized
    Uninitialized = 4,
    /// Corrupted allocator state
    Corruption = 5,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Size {
    /// 4KiB
    L0 = 0,
    /// 2MiB
    L1 = 1,
    /// 1GiB
    L2 = 2,
}

pub mod raw {
    use std::{ffi::c_void, sync::atomic::AtomicU64};

    use log::error;

    use crate::{Alloc, Allocator, Error, Page, Size};

    #[no_mangle]
    pub extern "C" fn nvalloc_init(cores: usize, addr: *mut c_void, pages: usize) -> i64 {
        let memory = unsafe { std::slice::from_raw_parts_mut(addr as *mut Page, pages) };
        match Allocator::init(cores, memory) {
            Ok(_) => 0,
            Err(e) => -(e as usize as i64),
        }
    }

    pub fn nvalloc_uninit() {
        Allocator::uninit();
    }

    pub fn nvalloc_get(core: usize, size: Size) -> i64 {
        let alloc = Allocator::instance();
        let begin = alloc.begin();
        match alloc.get(core, size) {
            Ok(page) => (page * Page::SIZE + begin) as i64,
            Err(e) => -(e as usize as i64),
        }
    }

    pub fn nvalloc_get_cas(
        core: usize,
        size: Size,
        dst: *const u64,
        translate: fn(u64) -> u64,
        expected: u64,
    ) -> i64 {
        let dst = unsafe { &*(dst as *const AtomicU64) };
        let alloc = Allocator::instance();
        let begin = alloc.begin();
        match alloc.get_cas(
            core,
            size,
            dst,
            |p| translate(p * Page::SIZE as u64 + begin as u64),
            expected,
        ) {
            Ok(_) => 0,
            Err(e) => -(e as usize as i64),
        }
    }

    pub fn nvalloc_put(core: usize, addr: u64) -> i64 {
        if addr % Page::SIZE as u64 != 0 {
            error!("Invalid align {addr:x}");
            return -(Error::Address as usize as i64);
        }
        let page = addr as usize / Page::SIZE;
        match Allocator::instance().put(core, page) {
            Ok(_) => 0,
            Err(e) => -(e as usize as i64),
        }
    }
}

/// Correctly sized and aligned page.
#[derive(Clone)]
#[repr(align(0x1000))]
pub struct Page {
    _data: [u8; Page::SIZE],
}
const _: () = assert!(Layout::new::<Page>().size() == Page::SIZE);
const _: () = assert!(Layout::new::<Page>().align() == Page::SIZE);
impl Page {
    pub const SIZE_BITS: usize = 12; // 2^12 => 4KiB
    pub const SIZE: usize = 1 << Page::SIZE_BITS;
    pub const fn new() -> Self {
        Self {
            _data: [0; Page::SIZE],
        }
    }
}
/// Non-Volatile global metadata
pub struct Meta {
    pub magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

enum Init {
    None,
    Initializing,
    Ready,
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

        let mut mapping: MMap<'_, Page> = MMap::anon(0x1000_0000_0000_u64 as _, 20 << 18).unwrap();

        info!("mmap {} bytes", mapping.len());

        info!("init alloc");

        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        Allocator::init(THREADS, &mut mapping[..]).unwrap();

        parallel(THREADS, |t| {
            let pages = [DEFAULT; Table::LEN];
            for page in &pages {
                Allocator::instance()
                    .get_cas(t, Size::L0, page, |v| v, 0)
                    .unwrap();
            }

            for page in &pages {
                Allocator::instance()
                    .put(t, page.load(Ordering::SeqCst) as usize)
                    .unwrap();
            }
        });

        info!("Finish");
    }
}
