use core::ffi::{c_char, c_size_t, c_void, CStr};
use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::Range;
use core::ptr::addr_of_mut;

use alloc::alloc::{alloc, dealloc, Layout};
use log::info;

use super::{Alloc, Init};
use crate::frame::PFNRange;
use crate::util::Align;
use crate::{Error, Result, PFN};

/// C implementation of LLFree
///
/// Note: This abstraction assumes that the state is movable and smaller than two cache lines!
#[repr(transparent)]
pub struct LLC {
    raw: [u8; 2 * align_of::<Align>()],
}

unsafe impl Send for LLC {}
unsafe impl Sync for LLC {}

impl Alloc for LLC {
    fn name() -> &'static str {
        "LLC"
    }

    fn new(cores: usize, area: Range<PFN>, init: Init, free_all: bool) -> Result<Self> {
        let mut raw = [0u8; size_of::<Self>()];

        let ret = unsafe {
            llfree_init(
                raw.as_mut_ptr().cast(),
                cores as _,
                area.start.0 as _,
                area.len() as _,
                init as usize as _,
                free_all,
            )
        };
        ret.ok().map(|_| LLC { raw })
    }

    fn get(&self, core: usize, order: usize) -> Result<PFN> {
        let ret = unsafe { llfree_get(self.raw.as_ptr().cast(), core as _, order as _) };
        Ok(PFN(ret.ok()? as _))
    }

    fn put(&self, core: usize, frame: PFN, order: usize) -> Result<()> {
        let ret = unsafe {
            llfree_put(
                self.raw.as_ptr().cast(),
                core as _,
                frame.0 as _,
                order as _,
            )
        };
        ret.ok().map(|_| ())
    }

    fn is_free(&self, frame: PFN, order: usize) -> bool {
        unsafe { llfree_is_free(self.raw.as_ptr().cast(), frame.0 as _, order as _) }
    }

    fn drain(&self, core: usize) -> Result<()> {
        unsafe {
            llfree_drain(self.raw.as_ptr().cast(), core as _)
                .ok()
                .map(|_| ())
        }
    }

    fn frames(&self) -> usize {
        unsafe { llfree_frames(self.raw.as_ptr().cast()) as _ }
    }

    fn free_frames(&self) -> usize {
        unsafe { llfree_free_frames(self.raw.as_ptr().cast()) as _ }
    }

    fn for_each_huge_frame<F: FnMut(PFN, usize)>(&self, mut f: F) {
        extern "C" fn wrapper<F: FnMut(PFN, usize)>(context: *mut c_void, pfn: u64, free: u64) {
            let f: &mut F = unsafe { &mut *context.cast() };
            f(PFN(pfn as usize), free as usize)
        }
        unsafe {
            llfree_for_each_huge(
                self.raw.as_ptr().cast(),
                addr_of_mut!(f).cast(),
                wrapper::<F>,
            )
        }
    }
}

impl Drop for LLC {
    fn drop(&mut self) {
        unsafe { llfree_drop(self.raw.as_mut_ptr().cast()) };
    }
}

impl fmt::Debug for LLC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // wrapper function that is called by the c implementation
        extern "C" fn writer(arg: *mut c_void, msg: *const c_char) {
            let f = unsafe { &mut *arg.cast::<fmt::Formatter<'_>>() };
            let c_str = unsafe { CStr::from_ptr(msg) };
            write!(f, "{}", c_str.to_str().unwrap()).unwrap();
        }

        unsafe {
            llfree_print_debug(
                self.raw.as_ptr().cast(),
                writer,
                (f as *mut fmt::Formatter).cast(),
            )
        };

        Ok(())
    }
}

#[repr(C)]
#[allow(non_camel_case_types)]
struct result_t {
    val: i64,
}

impl result_t {
    fn ok(self) -> Result<u64> {
        match self.val {
            val if val >= 0 => Ok(val as _),
            -1 => Err(Error::Memory),
            -2 => Err(Error::Retry),
            -3 => Err(Error::Address),
            -4 => Err(Error::Initialization),
            -5 => Err(Error::Corruption),
            _ => unreachable!("invalid return code"),
        }
    }
}

#[link(name = "llc", kind = "static")]
extern "C" {
    /// Initializes the allocator for the given memory region, returning 0 on success or a negative error code
    fn llfree_init(
        this: *mut c_void,
        cores: c_size_t,
        offset: u64,
        len: c_size_t,
        init: u8,
        free_all: bool,
    ) -> result_t;

    /// Destructs the allocator
    fn llfree_drop(this: *mut c_void);

    /// Allocates a frame and returns its address, or a negative error code
    fn llfree_get(this: *const c_void, core: c_size_t, order: c_size_t) -> result_t;

    /// Frees a frame, returning 0 on success or a negative error code
    fn llfree_put(this: *const c_void, core: c_size_t, frame: u64, order: c_size_t) -> result_t;

    /// Checks if a frame is allocated, returning 0 if not
    fn llfree_is_free(this: *const c_void, frame: u64, order: c_size_t) -> bool;

    /// Frees a frame, returning 0 on success or a negative error code
    fn llfree_drain(this: *const c_void, core: c_size_t) -> result_t;

    /// Returns the total number of frames the allocator can allocate
    fn llfree_frames(this: *const c_void) -> u64;

    /// Returns number of currently free frames
    fn llfree_free_frames(this: *const c_void) -> u64;

    /// Prints the allocators state for debugging
    fn llfree_print_debug(
        this: *const c_void,
        writer: extern "C" fn(*mut c_void, *const c_char),
        arg: *mut c_void,
    );

    fn llfree_for_each_huge(
        this: *const c_void,
        context: *mut c_void,
        f: extern "C" fn(*mut c_void, u64, u64),
    );
}

/// Allocate metadata function
#[no_mangle]
pub extern "C" fn llfree_ext_alloc(align: c_size_t, size: c_size_t) -> *mut c_void {
    info!("alloc a={align} {size}");
    unsafe { alloc(Layout::from_size_align(size as _, align as _).unwrap()) as _ }
}
/// Free metadata function
#[no_mangle]
pub extern "C" fn llfree_ext_free(align: c_size_t, size: c_size_t, addr: *mut c_void) {
    info!("free a={align} {size}");
    unsafe { dealloc(addr as _, Layout::from_size_align(size, align).unwrap()) }
}

#[cfg(test)]
mod test {
    use crate::{frame::PFN, Alloc, Init};

    use super::LLC;

    #[test]
    fn test_debug() {
        let alloc = LLC::new(1, PFN(0)..PFN(512), Init::Volatile, true).unwrap();
        println!("{alloc:?}");
    }
}
