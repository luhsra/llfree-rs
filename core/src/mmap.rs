//! Barebones linux mmap wrapper

use core::alloc::Layout;
use core::error::Error;
use core::fmt;
use core::num::NonZero;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::fs::File;
use std::os::unix::prelude::AsRawFd;
use std::path::Path;

use crate::frame::Frame;

#[derive(Debug)]
pub struct AllocError;
impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllocError")
    }
}
impl Error for AllocError {}

pub struct Mapping<T> {
    ptr: NonNull<T>,
    size: NonZero<usize>,
}

impl<T> Mapping<T> {
    pub fn anon(
        begin: usize,
        len: usize,
        shared: bool,
        populate: bool,
    ) -> Result<Self, AllocError> {
        let (begin, size) = calc_layout::<T>(begin, len)?;

        let visibility = if shared {
            libc::MAP_SHARED
        } else {
            libc::MAP_PRIVATE
        };

        #[allow(unused_mut)]
        let mut populate_f = 0;
        #[cfg(target_os = "linux")]
        if populate {
            populate_f = libc::MAP_POPULATE
        };

        let addr = unsafe {
            libc::mmap(
                begin as _,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | visibility | populate_f,
                -1,
                0,
            )
        };

        if addr != libc::MAP_FAILED {
            // This non-null slice is somewhat cursed
            Ok(Self {
                ptr: NonNull::new(addr.cast()).ok_or(AllocError)?,
                size: NonZero::new(size).ok_or(AllocError)?,
            })
        } else {
            unsafe { libc::perror(c"mmap failed".as_ptr()) };
            Err(AllocError)
        }
    }

    pub fn file(
        begin: usize,
        len: usize,
        file: impl AsRef<Path>,
        dax: bool,
    ) -> Result<Self, AllocError> {
        let (begin, size) = calc_layout::<T>(begin, len)?;

        #[allow(unused_mut)]
        let mut flags = libc::MAP_SHARED;

        #[cfg(target_os = "linux")]
        if dax {
            flags = libc::MAP_SHARED_VALIDATE | libc::MAP_SYNC;
        }

        let file = File::options()
            .read(true)
            .write(true)
            .open(file)
            .map_err(|_| AllocError)?;
        let fd = file.as_raw_fd();
        let addr = unsafe {
            libc::mmap(
                begin as _,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                fd,
                0,
            )
        };

        if addr != libc::MAP_FAILED {
            Ok(Self {
                ptr: NonNull::new(addr.cast()).ok_or(AllocError)?,
                size: NonZero::new(size).ok_or(AllocError)?,
            })
        } else {
            unsafe { libc::perror(c"mmap failed".as_ptr()) };
            Err(AllocError)
        }
    }
}

impl<T> Drop for Mapping<T> {
    fn drop(&mut self) {
        let ret = unsafe { libc::munmap(self.ptr.as_ptr() as _, self.size.get()) };
        if ret != 0 {
            unsafe { libc::perror(c"munmap failed".as_ptr()) };
            panic!("unmap failed");
        }
    }
}

impl<T> Deref for Mapping<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        let layout = Layout::new::<T>();
        unsafe {
            core::slice::from_raw_parts(
                self.ptr.as_ptr(),
                self.size.get() / layout.size().next_multiple_of(layout.align()),
            )
        }
    }
}
impl<T> DerefMut for Mapping<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let layout = Layout::new::<T>();
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.size.get() / layout.size().next_multiple_of(layout.align()),
            )
        }
    }
}

fn calc_layout<T>(begin: usize, len: usize) -> Result<(usize, usize), AllocError> {
    let layout = Layout::new::<T>();
    if layout.size() == 0 || layout.align() == 0 {
        return Err(AllocError);
    }
    let begin = begin
        .next_multiple_of(layout.align())
        .next_multiple_of(Frame::SIZE);
    let size = (len * layout.size())
        .next_multiple_of(layout.align())
        .next_multiple_of(Frame::SIZE);
    Ok((begin, size))
}

#[cfg(target_family = "unix")]
pub fn m_async<T>(slice: &[T]) {
    use core::mem::size_of_val;

    unsafe {
        let ret = libc::msync(slice.as_ptr() as _, size_of_val(slice), libc::MS_ASYNC);
        if ret != 0 {
            libc::perror(c"fsync failed".as_ptr());
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
    #[cfg(target_os = "linux")]
    Cold = 20,
    /// see /usr/include/bits/mman-linux.h
    #[cfg(target_os = "linux")]
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
        unsafe { libc::perror(c"madvice".as_ptr()) };
        panic!("madvice {ret}");
    }
}

#[cfg(test)]
pub fn test_mapping(begin: usize, length: usize) -> Mapping<Frame> {
    #[cfg(target_os = "linux")]
    if let Ok(f) = std::env::var("NVM_FILE") {
        use log::warn;
        warn!("MMap file {f} l={}G", (length * Frame::SIZE) >> 30);
        return Mapping::file(begin, length, &f, true).unwrap();
    }
    Mapping::anon(begin, length, false, true).unwrap()
}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::thread;

    use log::info;

    use super::Mapping;
    use crate::frame::Frame;
    use crate::util::logging;

    #[ignore]
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

        let mut mapping =
            Mapping::file(0x0000_1000_0000_0000, Frame::SIZE, "memfile", false).unwrap();

        mapping[0] = 42u8;
        assert_eq!(mapping[0], 42);

        std::fs::remove_file("memfile").unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn dax() {
        use core::arch::x86_64::_mm_sfence;

        logging();

        let file = std::env::var("NVM_DAX").unwrap_or_else(|_| "/dev/dax0.1".into());

        info!("MMap file {file} l=1G");

        let mut mapping = Mapping::file(0x0000_1000_0000_0000, 1 << 30, &file, true).unwrap();

        info!("previously {}", mapping[0]);

        mapping[0] = 42u8;
        unsafe { _mm_sfence() };

        assert_eq!(mapping[0], 42);
    }

    #[test]
    fn anonymous() {
        logging();

        let mut mapping = Mapping::anon(0x1000_0000_0000, 1024, false, true).unwrap();
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
