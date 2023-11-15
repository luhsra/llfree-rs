use core::ffi::{c_char, c_size_t, c_void, CStr};
use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::Range;
use core::ptr::addr_of_mut;

use alloc::alloc::{alloc, dealloc, Layout};

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
            llc_init(
                raw.as_mut_ptr().cast(),
                cores as _,
                area.start.0 as _,
                area.len() as _,
                init as usize as _,
                free_all as _,
            )
        };
        to_result(ret).map(|_| LLC { raw })
    }

    fn get(&self, core: usize, order: usize) -> Result<PFN> {
        let ret = unsafe { llc_get(self.raw.as_ptr().cast(), core as _, order as _) };
        Ok(PFN(to_result(ret)? as _))
    }

    fn put(&self, core: usize, frame: PFN, order: usize) -> Result<()> {
        let ret = unsafe {
            llc_put(
                self.raw.as_ptr().cast(),
                core as _,
                frame.0 as _,
                order as _,
            )
        };
        to_result(ret).map(|_| ())
    }

    fn is_free(&self, frame: PFN, order: usize) -> bool {
        let ret = unsafe { llc_is_free(self.raw.as_ptr().cast(), frame.0 as _, order as _) };
        ret != 0
    }

    fn frames(&self) -> usize {
        unsafe { llc_frames(self.raw.as_ptr().cast()) as _ }
    }

    fn free_frames(&self) -> usize {
        unsafe { llc_free_frames(self.raw.as_ptr().cast()) as _ }
    }

    fn for_each_huge_frame<F: FnMut(PFN, usize)>(&self, mut f: F) {
        extern "C" fn wrapper<F: FnMut(PFN, usize)>(context: *mut c_void, pfn: u64, free: u64) {
            let f: &mut F = unsafe { &mut *context.cast() };
            f(PFN(pfn as usize), free as usize)
        }
        unsafe {
            llc_for_each_huge(
                self.raw.as_ptr().cast(),
                addr_of_mut!(f).cast(),
                wrapper::<F>,
            )
        }
    }
}

impl Drop for LLC {
    fn drop(&mut self) {
        unsafe { llc_drop(self.raw.as_mut_ptr().cast()) };
    }
}

/// Converting return codes to errors
fn to_result(code: i64) -> Result<u64> {
    if code >= 0 {
        Ok(code as _)
    } else {
        match code {
            -1 => Err(Error::Memory),
            -2 => Err(Error::Retry),
            -3 => Err(Error::Address),
            -4 => Err(Error::Initialization),
            -5 => Err(Error::Corruption),
            _ => unreachable!("invalid return code"),
        }
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
            llc_debug(
                self.raw.as_ptr().cast(),
                writer,
                (f as *mut fmt::Formatter).cast(),
            )
        };

        Ok(())
    }
}

#[link(name = "llc", kind = "static")]
extern "C" {
    /// Initializes the allocator for the given memory region, returning 0 on success or a negative error code
    fn llc_init(
        this: *mut c_void,
        cores: u64,
        start_pfn: u64,
        len: u64,
        init: u8,
        free_all: u8,
    ) -> i64;

    /// Destructs the allocator
    fn llc_drop(this: *mut c_void);

    /// Allocates a frame and returns its address, or a negative error code
    fn llc_get(this: *const c_void, core: u64, order: u64) -> i64;

    /// Frees a frame, returning 0 on success or a negative error code
    fn llc_put(this: *const c_void, core: u64, frame: u64, order: u64) -> i64;

    /// Checks if a frame is allocated, returning 0 if not
    fn llc_is_free(this: *const c_void, frame: u64, order: u64) -> u8;

    /// Returns the total number of frames the allocator can allocate
    fn llc_frames(this: *const c_void) -> u64;

    /// Returns number of currently free frames
    fn llc_free_frames(this: *const c_void) -> u64;

    /// Prints the allocators state for debugging
    fn llc_debug(
        this: *const c_void,
        writer: extern "C" fn(*mut c_void, *const c_char),
        arg: *mut c_void,
    );

    fn llc_for_each_huge(
        this: *const c_void,
        context: *mut c_void,
        f: extern "C" fn(*mut c_void, u64, u64),
    );
}

/// Allocate metadata function
#[no_mangle]
pub extern "C" fn llc_ext_alloc(align: c_size_t, size: c_size_t) -> *mut c_void {
    unsafe { alloc(Layout::from_size_align(size as _, align as _).unwrap()) as _ }
}
/// Free metadata function
#[no_mangle]
pub extern "C" fn llc_ext_free(align: c_size_t, size: c_size_t, addr: *mut c_void) {
    unsafe { dealloc(addr as _, Layout::from_size_align(size, align).unwrap()) }
}

#[cfg(test)]
mod test {
    use crate::{frame::PFN, Alloc, Init};

    use super::LLC;

    #[test]
    fn test_debug() {
        let alloc = LLC::new(1, PFN(0)..PFN(2048), Init::Volatile, true).unwrap();
        println!("{alloc:?}");
    }
}
