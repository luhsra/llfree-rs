//! Barebones linux mmap wrapper

#![allow(dead_code)]

use std::ffi::{c_void, CString};
use std::fs::File;
use std::os::raw::{c_char, c_int, c_long, c_ulong};
use std::os::unix::prelude::AsRawFd;

/// Page can be read.
pub const PROT_READ: c_int = 0x1;
/// Page can be written.
pub const PROT_WRITE: c_int = 0x2;
/// Page can be executed.
pub const PROT_EXEC: c_int = 0x4;
/// Page can not be accessed.
pub const PROT_NONE: c_int = 0x0;
/// Extend change to start of growsdown vma (mprotect only).
pub const PROT_GROWSDOWN: c_int = 0x01000000;
/// Extend change to start of growsup vma (mprotect only).
pub const PROT_GROWSUP: c_int = 0x02000000;

/// Interpret addr exactly.
pub const MAP_FIXED: c_int = 0x10;

// Sharing types (must choose one and only one of these).
/// Share changes.
pub const MAP_SHARED: c_int = 0x01;
/// Changes are private.
pub const MAP_PRIVATE: c_int = 0x02;
/// Share changes and validate extension flags.
pub const MAP_SHARED_VALIDATE: c_int = 0x03;
/// Mask for type of mapping.
pub const MAP_TYPE: c_int = 0x0f;

/// Don't use a file.
pub const MAP_ANONYMOUS: c_int = 0x20;
/// When MAP_HUGETLB is set bits [26:31] encode the log2 of the huge page size.
pub const MAP_HUGE_SHIFT: c_int = 26;
pub const MAP_HUGE_MASK: c_int = 0x3f;

/// Return value of `mmap' in case of an error.
pub const MAP_FAILED: *mut c_void = -1 as _;

// generic flags

/// Stack-like segment.
pub const MAP_GROWSDOWN: c_int = 0x00100;
/// ETXTBSY.
pub const MAP_DENYWRITE: c_int = 0x00800;
/// Mark it as an executable.
pub const MAP_EXECUTABLE: c_int = 0x01000;
/// Lock the mapping.
pub const MAP_LOCKED: c_int = 0x02000;
/// Don't check for reservations.
pub const MAP_NORESERVE: c_int = 0x04000;
/// Populate (prefault) pagetables.
pub const MAP_POPULATE: c_int = 0x08000;
/// Do not block on IO.
pub const MAP_NONBLOCK: c_int = 0x10000;
/// Allocation is for a stack.
pub const MAP_STACK: c_int = 0x20000;
/// Create huge page mapping.
pub const MAP_HUGETLB: c_int = 0x40000;
/// Perform synchronous page faults for the mapping.
pub const MAP_SYNC: c_int = 0x80000;
/// MAP_FIXED but do not unmap underlying mapping.
pub const MAP_FIXED_NOREPLACE: c_int = 0x100000;

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: c_ulong,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: c_long,
    ) -> *mut c_void;

    fn munmap(addr: *mut c_void, len: c_ulong) -> c_int;

    fn perror(s: *const c_char);
}

pub fn c_perror(s: &str) {
    let ref this = CString::new(s).unwrap();
    unsafe { perror(this.as_ptr()) }
}

pub fn c_mmap_fixed(slice: &[u8], file: File) -> Result<(), ()> {
    let fd = file.as_raw_fd();
    let addr = unsafe {
        mmap(
            slice.as_ptr() as _,
            slice.len() as _,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_FIXED_NOREPLACE,
            fd,
            0,
        )
    };
    if addr == MAP_FAILED {
        unsafe { perror(b"mmap failed\0" as *const _ as _) };
        Err(())
    } else {
        Ok(())
    }
}

pub fn c_mmap_anon(slice: &[u8]) -> Result<(), ()> {
    let addr = unsafe {
        mmap(
            slice.as_ptr() as _,
            slice.len() as _,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };
    if addr == MAP_FAILED {
        unsafe { perror(b"mmap failed\0" as *const _ as _) };
        Err(())
    } else {
        Ok(())
    }
}

pub fn c_munmap(slice: &[u8]) -> Result<(), ()> {
    let ret = unsafe { munmap(slice.as_ptr() as _, slice.len() as _) };
    if ret != 0 {
        unsafe { perror(b"munmap failed\0" as *const _ as _) };
        Err(())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use core::slice;

    use crate::mmap::{c_mmap_anon, c_mmap_fixed, c_munmap};
    use crate::paging::PAGE_SIZE;

    #[test]
    fn mapping() {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("memfile")
            .unwrap();
        f.set_len(PAGE_SIZE as _).unwrap();
        f.sync_all().unwrap();

        let slice =
            unsafe { slice::from_raw_parts_mut(0x0000_1000_0000_0000_u64 as *mut u8, PAGE_SIZE) };

        c_mmap_fixed(slice, f).unwrap();

        slice[0] = 42;
        assert_eq!(slice[0], 42);

        c_munmap(slice).unwrap();
    }

    #[test]
    fn anonymous() {
        let slice =
            unsafe { slice::from_raw_parts_mut(0x0000_1000_0000_0000_u64 as *mut u8, PAGE_SIZE) };
        c_mmap_anon(slice).unwrap();

        slice[0] = 42;
        assert_eq!(slice[0], 42);

        c_munmap(slice).unwrap();
    }
}
