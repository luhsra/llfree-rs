//! Barebones linux mmap wrapper

use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use std::fs::File;
use std::os::unix::prelude::AsRawFd;

#[allow(unused)]
use alloc::boxed::Box;

use crate::util::align_down;
use crate::Frame;

/// Chunk of mapped memory.
/// Wrapper for POSIX mmap syscalls.
///
/// Tested on Linux and MacOS.
pub struct MMap<T: 'static> {
    slice: &'static mut [T],
    fd: Option<i32>,
}

impl<T> MMap<T> {
    #[cfg(target_family = "unix")]
    pub fn file(begin: usize, len: usize, file: File) -> Result<MMap<T>, ()> {
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
                slice: unsafe { std::slice::from_raw_parts_mut(addr.cast(), len) },
                fd: Some(fd),
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0".as_ptr().cast()) };
            Err(())
        }
    }

    #[cfg(target_os = "linux")]
    pub fn dax(begin: usize, len: usize, file: File) -> Result<MMap<T>, ()> {
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
                fd: Some(fd),
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0".as_ptr().cast()) };
            Err(())
        }
    }

    #[cfg(target_family = "unix")]
    pub fn anon(begin: usize, len: usize, shared: bool, populate: bool) -> Result<MMap<T>, ()> {
        if len == 0 {
            return Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(begin as _, len) },
                fd: None,
            });
        }

        let visibility = if shared {
            libc::MAP_SHARED
        } else {
            libc::MAP_PRIVATE
        };

        let addr = unsafe {
            libc::mmap(
                begin as _,
                (len * size_of::<T>()) as _,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | visibility | if populate { libc::MAP_POPULATE } else { 0 },
                -1,
                0,
            )
        };
        if addr != libc::MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, len) },
                fd: None,
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0".as_ptr().cast()) };
            Err(())
        }
    }

    /// Fallback for systems without mmap
    #[cfg(not(target_family = "unix"))]
    pub fn anon(_begin: usize, len: usize, _shared: bool, _populate: bool) -> Result<MMap<T>, ()> {
        Ok(Self {
            slice: unsafe { Box::leak(Box::new_uninit_slice(len).assume_init()) },
            fd: None,
        })
    }

    #[cfg(target_family = "unix")]
    pub fn f_sync(&self) {
        unsafe {
            if let Some(fd) = self.fd {
                if libc::fsync(fd) != 0 {
                    libc::perror(b"fsync failed\0".as_ptr().cast());
                }
            }
        }
    }
}

#[cfg(target_family = "unix")]
pub fn m_async<T>(slice: &mut [T]) {
    unsafe {
        if libc::msync(
            align_down(slice.as_mut_ptr() as usize, Frame::SIZE) as *mut _,
            slice.len() * size_of::<T>(),
            libc::MS_ASYNC,
        ) != 0
        {
            libc::perror(b"fsync failed\0".as_ptr().cast());
        }
    }
}

impl<T> Deref for MMap<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<T> DerefMut for MMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

impl<T> Drop for MMap<T> {
    #[cfg(target_family = "unix")]
    fn drop(&mut self) {
        if self.len() > 0 {
            let ret = unsafe {
                libc::munmap(
                    self.slice.as_ptr() as _,
                    (self.slice.len() * size_of::<T>()) as _,
                )
            };
            if ret != 0 {
                unsafe { libc::perror(b"munmap failed\0".as_ptr().cast()) };
                panic!(
                    "{:?} l={} (0x{:x})",
                    self.slice.as_ptr(),
                    self.slice.len(),
                    self.slice.len() * size_of::<T>()
                );
            }
        }
    }
    #[cfg(not(target_family = "unix"))]
    fn drop(&mut self) {
        if self.len() > 0 {
            drop(unsafe { Box::from_raw(self.slice) });
        }
    }
}

#[cfg(target_family = "unix")]
#[repr(i32)]
pub enum MAdvise {
    Normal = libc::MADV_NORMAL,
    Random = libc::MADV_RANDOM,
    Sequential = libc::MADV_SEQUENTIAL,
    WillNeed = libc::MADV_WILLNEED,
    /// Warn: This is not an advise. This will free the memory range immediately!
    DontNeed = libc::MADV_DONTNEED,
    Free = libc::MADV_FREE,
    Remove = libc::MADV_REMOVE,
    DontFork = libc::MADV_DONTFORK,
    DoFork = libc::MADV_DOFORK,
    Mergeable = libc::MADV_MERGEABLE,
    Unmergeable = libc::MADV_UNMERGEABLE,
    Hugepage = libc::MADV_HUGEPAGE,
    NoHugepage = libc::MADV_NOHUGEPAGE,
    DontDump = libc::MADV_DONTDUMP,
    DoDump = libc::MADV_DODUMP,
    HwPoison = libc::MADV_HWPOISON,

    /// see /usr/include/bits/mman-linux.h
    Cold = 20,
    /// see /usr/include/bits/mman-linux.h
    PageOut = 21,
}

#[cfg(target_family = "unix")]
pub fn madvise(mem: &mut [Frame], advise: MAdvise) {
    let ret = unsafe {
        libc::madvise(
            mem.as_mut_ptr() as *mut _,
            Frame::SIZE * mem.len(),
            advise as _,
        )
    };
    if ret != 0 {
        unsafe { libc::perror(b"madvice\0".as_ptr().cast()) };
        panic!("madvice {ret}");
    }
}

#[cfg(test)]
pub fn test_mapping(begin: usize, length: usize) -> core::result::Result<MMap<Frame>, ()> {
    use log::warn;

    #[cfg(target_os = "linux")]
    if let Ok(file) = std::env::var("NVM_FILE") {
        warn!("MMap file {file} l={}G", (length * Frame::SIZE) >> 30);
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file)
            .unwrap();
        return MMap::dax(begin, length, f);
    }
    MMap::anon(begin, length, false, true)
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::util::logging;
    use std::thread;

    use log::info;

    use crate::mmap::MMap;
    use crate::Frame;

    #[cfg(target_family = "unix")]
    #[test]
    fn file() {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("memfile")
            .unwrap();
        f.set_len(Frame::SIZE as _).unwrap();

        let mut mapping = MMap::file(0x0000_1000_0000_0000, Frame::SIZE, f).unwrap();

        mapping[0] = 42;
        assert_eq!(mapping[0], 42);
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn dax() {
        use core::arch::x86_64::_mm_sfence;

        logging();

        let file = std::env::var("NVM_DAX").unwrap_or_else(|_| "/dev/dax0.1".into());

        info!("MMap file {file} l=1G");

        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file)
            .unwrap();

        let mut mapping = MMap::dax(0x0000_1000_0000_0000, 1 << 30, f).unwrap();

        info!("previously {}", mapping[0]);

        mapping[0] = 42;
        unsafe { _mm_sfence() };

        assert_eq!(mapping[0], 42);
    }

    #[test]
    fn anonymous() {
        logging();

        let mut mapping = MMap::anon(0x0000_1000_0000_0000, Frame::SIZE, false, true).unwrap();

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
