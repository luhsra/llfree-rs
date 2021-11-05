//! Barebones linux mmap wrapper

use std::ffi::{c_void, CString};
use std::fs::File;
use std::os::raw::c_int;
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

/// Don't interpret addr as a hint: place the mapping at  exactly  that  address.   addr  must  be  suitably
/// aligned:  for  most architectures a multiple of the page size is sufficient; however, some architectures
/// may impose additional restrictions.  If the memory region specified by addr and len  overlaps  pages  of
/// any  existing mapping(s), then the overlapped part of the existing mapping(s) will be discarded.  If the
/// specified address cannot be used, mmap() will fail.
pub const MAP_FIXED: c_int = 0x10;

// Sharing types (must choose one and only one of these).

/// Share this mapping.  Updates to the mapping are visible to other processes mapping the same region,  and
/// (in the case of file-backed mappings) are carried through to the underlying file.  (To precisely control
/// when updates are carried through to the underlying file requires the use of msync(2).)
pub const MAP_SHARED: c_int = 0x01;
/// Create  a private copy-on-write mapping.  Updates to the mapping are not visible to other processes map‐
/// ping the same file, and are not carried through to the  underlying  file.   It  is  unspecified  whether
/// changes made to the file after the mmap() call are visible in the mapped region.
pub const MAP_PRIVATE: c_int = 0x02;
/// This flag provides the same behavior as MAP_SHARED except that MAP_SHARED mappings ignore unknown  flags
/// in  flags.   By  contrast,  when  creating  a mapping using MAP_SHARED_VALIDATE, the kernel verifies all
/// passed flags are known and fails the mapping with the error EOPNOTSUPP for unknown flags.  This  mapping
/// type is also required to be able to use some mapping flags (e.g., MAP_SYNC).
pub const MAP_SHARED_VALIDATE: c_int = 0x03;
/// Mask for type of mapping.
pub const MAP_TYPE: c_int = 0x0f;

/// The mapping is not backed by any file; its contents are initialized to zero.  The  fd  argument  is  ig‐
/// nored;  however,  some  implementations require fd to be -1 if MAP_ANONYMOUS (or MAP_ANON) is specified,
/// and portable applications should ensure  this.   The  offset  argument  should  be  zero.   The  use  of
/// MAP_ANONYMOUS in conjunction with MAP_SHARED is supported on Linux only since kernel 2.4.
pub const MAP_ANONYMOUS: c_int = 0x20;

/// When MAP_HUGETLB is set bits [26:31] encode the log2 of the huge page size.
/// Used  in  conjunction  with MAP_HUGETLB to select alternative hugetlb page sizes (respectively, 2 MB and
/// 1 GB) on systems that support multiple hugetlb page sizes.
///
/// More generally, the desired huge page size can be configured by encoding the base-2 logarithm of the de‐
/// sired  page  size in the six bits at the offset MAP_HUGE_SHIFT.  (A value of zero in this bit field pro‐
/// vides the default huge page size; the default huge page size can  be  discovered  via  the  Hugepagesize
/// field exposed by /proc/meminfo.)  Thus, the above two constants are defined as:
///
///     #define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
///     #define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
///
/// The range of huge page sizes that are supported by the system can be discovered by listing the subdirec‐
/// tories in /sys/kernel/mm/hugepages.
pub const MAP_HUGE_SHIFT: c_int = 26;
pub const MAP_HUGE_MASK: c_int = 0x3f;

/// Return value of `mmap' in case of an error.
pub const MAP_FAILED: *mut c_void = -1 as _;

// generic flags

/// Stack-like segment.
pub const MAP_GROWSDOWN: c_int = 0x00100;
/// This  flag is ignored.  (Long ago—Linux 2.0 and earlier—it signaled that attempts to write to the under‐
/// lying file should fail with ETXTBUSY.  But this was a source of denial-of-service attacks.)
#[deprecated]
pub const MAP_DENYWRITE: c_int = 0x00800;
/// Mark it as an executable.
pub const MAP_EXECUTABLE: c_int = 0x01000;
/// Mark the mapped region to be locked in the same way as mlock(2).  This implementation will try to  popu‐
/// late  (prefault)  the whole range but the mmap() call doesn't fail with ENOMEM if this fails.  Therefore
/// major faults might happen later on.  So the semantic is not as  strong  as  mlock(2).   One  should  use
/// mmap()  plus mlock(2) when major faults are not acceptable after the initialization of the mapping.  The
/// MAP_LOCKED flag is ignored in older kernels.
pub const MAP_LOCKED: c_int = 0x02000;
/// Do  not reserve swap space for this mapping.  When swap space is reserved, one has the guarantee that it
/// is possible to modify the mapping.  When swap space is not reserved one might get SIGSEGV upon  a  write
/// if  no physical memory is available.  See also the discussion of the file /proc/sys/vm/overcommit_memory
/// in proc(5).  In kernels before 2.6, this flag had effect only for private writable mappings.
pub const MAP_NORESERVE: c_int = 0x04000;
/// Populate (prefault) page tables for a mapping.  For a file mapping, this causes read-ahead on the  file.
/// This  will help to reduce blocking on page faults later.  MAP_POPULATE is supported for private mappings
/// only since Linux 2.6.23.
pub const MAP_POPULATE: c_int = 0x08000;
/// This flag is meaningful only in conjunction with MAP_POPULATE.  Don't perform  read-ahead:  create  page
/// tables  entries  only  for  pages that are already present in RAM.  Since Linux 2.6.23, this flag causes
/// MAP_POPULATE to do nothing.  One day, the combination of MAP_POPULATE and MAP_NONBLOCK may  be  reimple‐
/// mented.
pub const MAP_NONBLOCK: c_int = 0x10000;
/// Allocate the mapping at an address suitable for a process or thread stack.
pub const MAP_STACK: c_int = 0x20000;
/// Allocate the mapping using  "huge  pages."   See  the  Linux  kernel  source  file  Documentation/admin-
/// guide/mm/hugetlbpage.rst for further information, as well as NOTES, below.
pub const MAP_HUGETLB: c_int = 0x40000;
/// This flag is available only with the MAP_SHARED_VALIDATE mapping type; mappings of type MAP_SHARED  will
/// silently ignore this flag.  This flag is supported only for files supporting DAX (direct mapping of per‐
/// sistent memory).  For other files, creating a mapping with this flag results in an EOPNOTSUPP error.
///
/// Shared file mappings with this flag provide the guarantee that while some memory is writably  mapped  in
/// the  address space of the process, it will be visible in the same file at the same offset even after the
/// system crashes or is rebooted.  In conjunction with the use of appropriate CPU instructions,  this  pro‐
/// vides users of such mappings with a more efficient way of making data modifications persistent.
pub const MAP_SYNC: c_int = 0x80000;
/// MAP_FIXED but do not unmap underlying mapping.
pub const MAP_FIXED_NOREPLACE: c_int = 0x100000;

pub fn perror(s: &str) {
    let ref s = CString::new(s).unwrap();
    unsafe { libc::perror(s.as_ptr()) }
}

pub struct MMap<'a> {
    pub slice: &'a mut [u8],
}

impl<'a> MMap<'a> {
    /// Create a file based mapping.
    pub fn file(begin: usize, length: usize, file: File) -> Result<MMap<'a>, ()> {
        let fd = file.as_raw_fd();
        let addr = unsafe {
            libc::mmap(
                begin as _,
                length as _,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED_NOREPLACE,
                fd,
                0,
            )
        };
        if addr != MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, length) },
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0" as *const _ as _) };
            Err(())
        }
    }
    /// Create a mapping to the given Direct Memory Access device.
    pub fn dax(begin: usize, length: usize, file: File) -> Result<MMap<'a>, ()> {
        let fd = file.as_raw_fd();
        let addr = unsafe {
            libc::mmap(
                begin as _,
                length as _,
                PROT_READ | PROT_WRITE,
                MAP_SHARED_VALIDATE | MAP_SYNC,
                fd,
                0,
            )
        };
        if addr != MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, length) },
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0" as *const _ as _) };
            Err(())
        }
    }
    /// Create an anonymous mapping without a file.
    pub fn anon(begin: usize, length: usize) -> Result<MMap<'a>, ()> {
        let addr = unsafe {
            libc::mmap(
                begin as _,
                length as _,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        if addr != MAP_FAILED {
            Ok(MMap {
                slice: unsafe { std::slice::from_raw_parts_mut(addr as _, length) },
            })
        } else {
            unsafe { libc::perror(b"mmap failed\0" as *const _ as _) };
            Err(())
        }
    }
}

impl<'a> Drop for MMap<'a> {
    fn drop(&mut self) {
        let ret = unsafe { libc::munmap(self.slice.as_ptr() as _, self.slice.len() as _) };
        if ret != 0 {
            unsafe { libc::perror(b"munmap failed\0" as *const _ as _) };
            panic!();
        }
    }
}

#[cfg(test)]
mod test {
    use core::arch::x86_64::_mm_sfence;
    use std::thread;

    use log::info;

    use crate::mmap::MMap;
    use crate::paging::PAGE_SIZE;
    use crate::util::{_mm_clwb, logging};

    #[test]
    fn mapping() {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("memfile")
            .unwrap();
        f.set_len(PAGE_SIZE as _).unwrap();

        let mapping = MMap::file(0x0000_1000_0000_0000, PAGE_SIZE, f).unwrap();

        mapping.slice[0] = 42;
        assert_eq!(mapping.slice[0], 42);
    }

    #[test]
    fn dax() {
        logging();

        let file = std::env::var("NVM_DAX").unwrap_or("/dev/dax0.1".into());

        info!("MMap file {} l={}G", file, 1);

        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file)
            .unwrap();

        let mapping = MMap::dax(0x0000_1000_0000_0000, 1 << 30, f).unwrap();

        info!("previously {}", mapping.slice[0]);

        mapping.slice[0] = 42;
        unsafe { _mm_clwb(mapping.slice.as_ptr() as _) };
        unsafe { _mm_sfence() };

        assert_eq!(mapping.slice[0], 42);
    }

    #[test]
    fn anonymous() {
        logging();

        let mapping = MMap::anon(0x0000_1000_0000_0000, PAGE_SIZE).unwrap();

        mapping.slice[0] = 42;
        assert_eq!(mapping.slice[0], 42);

        let addr = mapping.slice.as_ptr() as usize;
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

        assert_eq!(mapping.slice[0], 43);
    }
}
