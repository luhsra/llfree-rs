//! Barebones linux mmap wrapper

use std::ffi::CString;
use std::fs::File;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::os::unix::prelude::AsRawFd;

pub fn perror(s: &str) {
    let s = CString::new(s).unwrap();
    unsafe { libc::perror(s.as_ptr()) }
}

/// Chunk of mapped memory.
pub struct MMap<'a, T> {
    slice: &'a mut [T],
}

impl<'a, T> MMap<'a, T> {
    pub fn file(begin: usize, len: usize, file: File) -> Result<MMap<'a, T>, ()> {
        assert!(len > 0);

        let fd = file.as_raw_fd();
        let addr = unsafe {
            libc::mmap(
                begin as _,
                (len * size_of::<T>()) as _,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if addr != libc::MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, len) },
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0" as *const _ as _) };
            Err(())
        }
    }

    #[cfg(target_os = "linux")]
    pub fn dax(begin: usize, len: usize, file: File) -> Result<MMap<'a, T>, ()> {
        assert!(len > 0);

        let fd = file.as_raw_fd();
        let addr = unsafe {
            libc::mmap(
                begin as _,
                (len * size_of::<T>()) as _,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED_VALIDATE | libc::MAP_SYNC,
                fd,
                0,
            )
        };
        if addr != libc::MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, len) },
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0" as *const _ as _) };
            Err(())
        }
    }

    pub fn anon(begin: usize, len: usize) -> Result<MMap<'a, T>, ()> {
        if len == 0 {
            return Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(begin as _, len) },
            });
        }

        let addr = unsafe {
            libc::mmap(
                begin as _,
                (len * size_of::<T>()) as _,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if addr != libc::MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, len) },
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0" as *const _ as _) };
            Err(())
        }
    }
}

impl<'a, T> Deref for MMap<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T> DerefMut for MMap<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

impl<'a, T> Drop for MMap<'a, T> {
    fn drop(&mut self) {
        if self.len() > 0 {
            let ret = unsafe { libc::munmap(self.slice.as_ptr() as _, self.slice.len() as _) };
            if ret != 0 {
                unsafe { libc::perror(b"munmap failed\0" as *const _ as _) };
                panic!();
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::util::logging;
    use std::thread;

    use log::info;

    use crate::mmap::MMap;
    use crate::Page;

    #[test]
    fn mapping() {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("memfile")
            .unwrap();
        f.set_len(Page::SIZE as _).unwrap();

        let mut mapping = MMap::file(0x0000_1000_0000_0000, Page::SIZE, f).unwrap();

        mapping[0] = 42;
        assert_eq!(mapping[0], 42);
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn dax() {
        use crate::util::_mm_clwb;
        use core::arch::x86_64::_mm_sfence;

        logging();

        let file = std::env::var("NVM_DAX").unwrap_or("/dev/dax0.1".into());

        info!("MMap file {} l={}G", file, 1);

        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file)
            .unwrap();

        let mut mapping = MMap::dax(0x0000_1000_0000_0000, 1 << 30, f).unwrap();

        info!("previously {}", mapping[0]);

        mapping[0] = 42;
        unsafe { _mm_clwb(mapping.as_ptr() as _) };
        unsafe { _mm_sfence() };

        assert_eq!(mapping[0], 42);
    }

    #[test]
    fn anonymous() {
        logging();

        let mut mapping = MMap::anon(0x0000_1000_0000_0000, Page::SIZE).unwrap();

        mapping[0] = 42;
        assert_eq!(mapping[0], 42);

        let addr = mapping.as_ptr() as usize;
        info!("check own thread");
        assert_eq!(unsafe { *(addr as *const u8) }, 42);
        thread::spawn(move || {
            let data = unsafe { &mut *(addr as *mut u8) };
            info!("check multithreading");
            assert_eq!(*data, 42);
            *data = 43;
            info!("success");
        })
        .join()
        .unwrap();

        assert_eq!(mapping[0], 43);
    }
}
