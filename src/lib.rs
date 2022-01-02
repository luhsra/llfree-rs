//! Simple reduced alloc example.
#![feature(asm)]
#![feature(panic_info_message)]

use std::sync::atomic::{AtomicU64, Ordering};

mod alloc;
mod cpu;
mod entry;
mod leaf_alloc;
pub mod mmap;
mod table;
mod util;

#[cfg(test)]
mod wait;

use alloc::{alloc, Allocator};
use table::{Page, PAGE_SIZE};
use util::{align_down, align_up};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory,
    /// Failed comapare and swap operation
    CAS,
    /// Invalid address
    Address,
    /// Corrupted allocator state
    Corruption(usize, usize, u64),
    /// Allocator not initialized
    Uninitialized,
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

pub fn init(cores: usize, addr: *mut (), size: usize) -> Result<()> {
    let begin = align_up(addr as usize, PAGE_SIZE) as *mut Page;
    let size = align_down(addr as usize + size, PAGE_SIZE).saturating_sub(begin as usize);
    let memory = unsafe { std::slice::from_raw_parts_mut(begin, size / PAGE_SIZE) };

    Allocator::init(cores, memory)
}

pub fn uninit() {
    Allocator::uninit();
}

pub fn get<F: FnOnce(u64) -> u64>(
    core: usize,
    size: Size,
    dst: &AtomicU64,
    translate: F,
    expected: u64,
) -> Result<()> {
    let page = alloc().get(core, size)?;
    let new = translate((page * PAGE_SIZE) as u64);
    match dst.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => Ok(()),
        Err(_) => {
            alloc().put(core, page).unwrap();
            Err(Error::CAS)
        }
    }
}

pub fn put(core: usize, addr: u64) -> Result<()> {
    if addr % PAGE_SIZE as u64 != 0 {
        return Err(Error::Address);
    }
    let page = addr as usize / PAGE_SIZE;

    alloc().put(core, page).map(|_| ())
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicU64, Ordering};

    use log::info;

    use crate::mmap::MMap;
    use crate::table::{Page, PT_LEN};
    use crate::util::{logging, parallel};
    use crate::{get, init, put, Size};

    #[test]
    fn threading() {
        logging();

        const THREADS: usize = 8;

        let mapping: MMap<'_, Page> = MMap::anon(0x1000_0000_0000_u64 as _, 20 << 18).unwrap();

        info!("mmap {} bytes", mapping.len());

        info!("init alloc");

        let addr = mapping.as_ptr() as usize;
        let size = mapping.len();

        info!("init finished");
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        init(THREADS, addr as _, size).unwrap();

        parallel(THREADS, |t| {
            let pages = [DEFAULT; PT_LEN];
            for page in &pages {
                get(t, Size::L0, page, |v| v, 0).unwrap();
            }

            for page in &pages {
                put(t, page.load(Ordering::SeqCst)).unwrap();
            }
        });

        info!("Finish");
    }
}
