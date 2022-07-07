//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.
#![no_std]
#![feature(generic_const_exprs)]

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

use alloc::boxed::Box;
use upper::{Alloc, Size};
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
