//! Simple reduced alloc example.

use std::{cell::RefCell, sync::atomic::AtomicU64};

mod alloc;
pub mod mmap;
mod paging;
mod util;
mod cpu;

#[cfg(test)]
mod sync;

use alloc::{Allocator, Error, Size};

thread_local! {
    static ALLOC: RefCell<Option<Allocator>> = RefCell::new(None);
}

pub fn init(addr: *mut (), size: usize) -> alloc::Result<()> {
    ALLOC.with(|a| {
        *a.borrow_mut() = Some(Allocator::init(addr as usize, size)?);
        Ok(())
    })?;

    Ok(())
}

pub fn get<F: FnOnce(u64) -> u64>(
    size: Size,
    dst: &AtomicU64,
    translate: F,
    expected: u64,
) -> alloc::Result<()> {
    ALLOC.with(|a| {
        let mut a = a.borrow_mut();

        if let Some(a) = a.as_mut() {
            a.get(size, dst, translate, expected)
        } else {
            Err(Error::Uninitialized)
        }
    })?;

    Ok(())
}

pub fn put(addr: u64, size: Size) -> alloc::Result<()> {
    ALLOC.with(|a| {
        let mut a = a.borrow_mut();

        if let Some(a) = a.as_mut() {
            a.put(addr, size)
        } else {
            Err(Error::Uninitialized)
        }
    })?;

    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;

    use log::info;

    use crate::alloc::Size;
    use crate::mmap::MMap;
    use crate::paging::PT_LEN;
    use crate::util::logging;
    use crate::{get, init, put};

    #[test]
    fn threading() {
        logging();

        let mapping = MMap::anon(0x1000_0000_0000_u64 as _, 20 << 30).unwrap();

        info!("mmap {} bytes", mapping.slice.len());

        info!("init alloc");

        let addr = mapping.slice.as_ptr() as usize;
        let size = mapping.slice.len();

        info!("init finished");
        const DEFAULT: AtomicU64 = AtomicU64::new(0);

        let threads = (0..10)
            .into_iter()
            .map(|_| {
                thread::spawn(move || {
                    init(addr as _, size).unwrap();

                    let pages = [DEFAULT; PT_LEN];
                    for page in &pages {
                        get(Size::Page, page, |v| v, 0).unwrap();
                    }

                    for page in &pages {
                        put(page.load(Ordering::SeqCst), Size::Page).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        thread::sleep(Duration::from_secs(1));

        for t in threads {
            t.join().unwrap();
        }

        info!("Finish");
    }
}
