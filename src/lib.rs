//! Simple reduced alloc example.

use std::{cell::RefCell, sync::atomic::AtomicU64};

mod alloc;
mod mmap;
mod paging;
mod util;

use alloc::{Allocator, ChunkSize, Error};

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
    size: ChunkSize,
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

pub fn put(addr: u64, size: ChunkSize) -> alloc::Result<()> {
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
    use std::{slice, sync::atomic::AtomicU64, thread};

    use crate::{
        alloc::{ChunkSize, MAX_SIZE},
        get, init,
        mmap::c_mmap_fixed,
    };

    #[test]
    fn threading() {
        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, MAX_SIZE) };

        println!("prepare file");

        let f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("memfile")
            .unwrap();
        f.set_len(data.len() as _).unwrap();
        f.sync_all().unwrap();

        println!("alloc {} bytes", data.len());

        c_mmap_fixed(data, f).unwrap();

        println!("init alloc");

        let addr = data.as_ptr() as _;
        let size = data.len();

        init(addr, size).unwrap();

        println!("init finished");

        let addr = addr as usize;
        let handle = thread::spawn(move || {
            init(addr as _, size).unwrap();

            for _ in 0..10 {
                let dst = AtomicU64::new(0);
                get(ChunkSize::Page, &dst, |v| v, 0).unwrap();
            }
        });
        for _ in 0..10 {
            let dst = AtomicU64::new(0);
            get(ChunkSize::Page, &dst, |v| v, 0).unwrap();
        }

        handle.join().unwrap();

        println!("Finish");
        super::ALLOC.with(|a| {
            let mut a = a.borrow_mut();
            if let Some(a) = a.as_mut() {
                a.dump();
            }
        });
    }
}
