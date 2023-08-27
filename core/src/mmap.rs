//! Barebones linux mmap wrapper

use alloc::boxed::Box;
use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use std::fs::File;
use std::os::unix::prelude::AsRawFd;

use crate::frame::Frame;

/// Create an private anonymous mapping
pub fn anon<T>(begin: usize, len: usize, shared: bool, populate: bool) -> Box<[T], MMap> {
    unsafe { Box::new_uninit_slice_in(len, MMap::anon(begin, shared, populate)).assume_init() }
}
/// Create an file backed mapping (optionally DAX)
pub fn file<T>(begin: usize, len: usize, path: &str, dax: bool) -> Box<[T], MMap> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    unsafe { Box::new_uninit_slice_in(len, MMap::file(begin, file, dax)).assume_init() }
}

/// Wrapper for POSIX mmap syscalls.
///
/// Tested on Linux and MacOS.
pub struct MMap {
    begin: usize,
    shared: bool,
    #[allow(unused)]
    populate: bool,
    file: Option<(File, bool)>,
}

impl MMap {
    pub fn anon(begin: usize, shared: bool, populate: bool) -> Self {
        Self {
            begin,
            shared,
            populate,
            file: None,
        }
    }

    #[cfg(target_family = "unix")]
    pub fn file(begin: usize, file: File, dax: bool) -> Self {
        Self {
            begin,
            shared: true,
            populate: false,
            file: Some((file, dax)),
        }
    }
}

#[cfg(target_family = "unix")]
unsafe impl Allocator for MMap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let begin = if layout.align() != 0 {
            self.begin.next_multiple_of(layout.align())
        } else {
            self.begin
        };

        if layout.size() == 0 {
            return Ok(unsafe { std::slice::from_raw_parts(begin as _, 0) }.into());
        }

        let addr = if let Some((file, _dax)) = &self.file {
            let fd = file.as_raw_fd();

            #[allow(unused_mut)]
            let mut flags = libc::MAP_SHARED;

            #[cfg(target_os = "linux")]
            if *_dax {
                flags = libc::MAP_SHARED_VALIDATE | libc::MAP_SYNC;
            }

            unsafe {
                libc::mmap(
                    begin as _,
                    layout.size() as _,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags,
                    fd,
                    0,
                )
            }
        } else {
            let visibility = if self.shared {
                libc::MAP_SHARED
            } else {
                libc::MAP_PRIVATE
            };

            #[allow(unused_mut)]
            let mut populate = 0;
            #[cfg(target_os = "linux")]
            if self.populate {
                populate = libc::MAP_POPULATE
            };

            unsafe {
                libc::mmap(
                    begin as _,
                    layout.size() as _,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANONYMOUS | visibility | populate,
                    -1,
                    0,
                )
            }
        };

        if addr != libc::MAP_FAILED {
            Ok(unsafe { std::slice::from_raw_parts(addr.cast(), layout.size()) }.into())
        } else {
            unsafe { libc::perror(b"mmap failed\0".as_ptr().cast()) };
            Err(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() > 0 {
            let ret = unsafe { libc::munmap(ptr.as_ptr() as _, layout.size() as _) };
            if ret != 0 {
                unsafe { libc::perror(b"munmap failed\0".as_ptr().cast()) };
                panic!("unmap {layout:?}");
            }
        }
    }
}

// Fallback for non-unix systems
#[cfg(not(target_family = "unix"))]
unsafe impl Allocator for MMap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { std::alloc::alloc_zeroed(layout) }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

#[cfg(target_family = "unix")]
pub fn m_async<T>(slice: &[T]) {
    use core::mem::size_of_val;

    unsafe {
        let ret = libc::msync(slice.as_ptr() as _, size_of_val(slice), libc::MS_ASYNC);
        if ret != 0 {
            libc::perror(b"fsync failed\0".as_ptr().cast());
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

    #[cfg(target_os = "linux")]
    Remove = libc::MADV_REMOVE,
    #[cfg(target_os = "linux")]
    DontFork = libc::MADV_DONTFORK,
    #[cfg(target_os = "linux")]
    DoFork = libc::MADV_DOFORK,
    #[cfg(target_os = "linux")]
    Mergeable = libc::MADV_MERGEABLE,
    #[cfg(target_os = "linux")]
    Unmergeable = libc::MADV_UNMERGEABLE,
    #[cfg(target_os = "linux")]
    Hugepage = libc::MADV_HUGEPAGE,
    #[cfg(target_os = "linux")]
    NoHugepage = libc::MADV_NOHUGEPAGE,
    #[cfg(target_os = "linux")]
    DontDump = libc::MADV_DONTDUMP,
    #[cfg(target_os = "linux")]
    DoDump = libc::MADV_DODUMP,
    #[cfg(target_os = "linux")]
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
pub fn test_mapping(begin: usize, length: usize) -> Box<[Frame], MMap> {
    #[cfg(target_os = "linux")]
    if let Ok(f) = std::env::var("NVM_FILE") {
        use log::warn;
        warn!("MMap file {f} l={}G", (length * Frame::SIZE) >> 30);
        return file(begin, length, &f, true);
    }
    anon(begin, length, false, true)
}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::thread;

    use crate::frame::Frame;
    use crate::util::logging;

    use log::info;

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
        drop(f);

        let mut mapping = super::file(0x0000_1000_0000_0000, Frame::SIZE, "memfile", false);

        mapping[0] = 42u8;
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

        let mut mapping = super::file(0x0000_1000_0000_0000, 1 << 30, &file, true);

        info!("previously {}", mapping[0]);

        mapping[0] = 42u8;
        unsafe { _mm_sfence() };

        assert_eq!(mapping[0], 42);
    }

    #[test]
    fn anonymous() {
        logging();

        let mut mapping = super::anon(0x1000_0000_0000, 1024, false, true);
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
