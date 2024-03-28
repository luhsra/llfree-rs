use core::ffi::{c_char, c_size_t, c_void, CStr};
use core::mem::{align_of, size_of};
use core::{fmt, slice};

use super::{Alloc, Init};
use crate::util::Align;
use crate::{Error, Flags, Result};

/// C implementation of LLFree
///
/// Note: This abstraction assumes that the state is movable and smaller than two cache lines!
#[repr(transparent)]
pub struct LLC {
    raw: [u8; 2 * align_of::<Align>()],
}

/// Opaque type of the internal allocator
#[allow(non_camel_case_types)]
type llfree_t = c_void;

unsafe impl Send for LLC {}
unsafe impl Sync for LLC {}

impl<'a> Alloc<'a> for LLC {
    fn name() -> &'static str {
        "LLC"
    }

    fn metadata_size(cores: usize, frames: usize) -> crate::MetaSize {
        let m = unsafe { llfree_metadata_size(cores as _, frames as _) };
        crate::MetaSize {
            primary: m.primary as _,
            secondary: m.secondary as _,
        }
    }

    fn metadata(&mut self) -> (&'a mut [u8], &'a mut [u8]) {
        let cores = unsafe { llfree_cores(self.raw.as_ptr().cast()) };
        let ms = Self::metadata_size(cores, self.frames());
        let m = unsafe { llfree_metadata(self.raw.as_mut_ptr().cast()) };
        let primary = unsafe { slice::from_raw_parts_mut(m.primary.cast(), ms.primary) };
        let secondary = unsafe { slice::from_raw_parts_mut(m.secondary.cast(), ms.secondary) };
        (primary, secondary)
    }

    fn new(
        cores: usize,
        frames: usize,
        init: Init,
        primary: &'a mut [u8],
        secondary: &'a mut [u8],
    ) -> Result<Self> {
        let mut raw = [0u8; size_of::<Self>()];

        let init = match init {
            Init::FreeAll => 0,
            Init::AllocAll => 1,
            Init::Recover(false) => 2,
            Init::Recover(true) => 3,
        };

        let m = Self::metadata_size(cores, frames);
        assert!(primary.len() >= m.primary);
        assert!(secondary.len() >= m.secondary);

        let ret = unsafe {
            llfree_init(
                raw.as_mut_ptr().cast(),
                cores as _,
                frames,
                init,
                primary.as_mut_ptr(),
                secondary.as_mut_ptr(),
            )
        };
        ret.ok().map(|_| LLC { raw })
    }

    fn get(&self, core: usize, flags: Flags) -> Result<usize> {
        let ret = unsafe { llfree_get(self.raw.as_ptr().cast(), core as _, flags.into()) };
        Ok(ret.ok()? as _)
    }

    fn put(&self, core: usize, frame: usize, flags: Flags) -> Result<()> {
        let ret = unsafe {
            llfree_put(
                self.raw.as_ptr().cast(),
                core as _,
                frame as _,
                flags.into(),
            )
        };
        ret.ok().map(|_| ())
    }

    fn is_free(&self, frame: usize, order: usize) -> bool {
        unsafe { llfree_is_free(self.raw.as_ptr().cast(), frame as _, order as _) }
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

    fn cores(&self) -> usize {
        unsafe { llfree_cores(self.raw.as_ptr().cast()) as _ }
    }

    fn free_frames(&self) -> usize {
        unsafe { llfree_free_frames(self.raw.as_ptr().cast()) as _ }
    }
    fn free_huge(&self) -> usize {
        unsafe { llfree_free_huge(self.raw.as_ptr().cast()) as _ }
    }

    fn free_at(&self, frame: usize, order: usize) -> usize {
        unsafe { llfree_free_at(self.raw.as_ptr().cast(), frame as _, order) }
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
struct flags_t {
    order: u8,
    flags: u8,
}
impl From<Flags> for flags_t {
    fn from(flags: Flags) -> Self {
        flags_t {
            order: flags.order() as _,
            flags: flags.movable() as _,
        }
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
            _ => unreachable!("invalid return code"),
        }
    }
}

#[repr(C)]
struct MetaSize {
    primary: c_size_t,
    secondary: c_size_t,
}

#[repr(C)]
struct Meta {
    primary: *mut u8,
    secondary: *mut u8,
}

#[link(name = "llc", kind = "static")]
extern "C" {
    /// Initializes the allocator for the given memory region, returning 0 on success or a negative error code
    fn llfree_init(
        this: *mut llfree_t,
        cores: c_size_t,
        frames: c_size_t,
        init: u8,
        primary: *mut u8,
        secondary: *mut u8,
    ) -> result_t;

    /// Returns the size of the metadata buffers required for initialization
    fn llfree_metadata_size(cores: c_size_t, frames: c_size_t) -> MetaSize;

    /// Returns the metadata
    fn llfree_metadata(this: *mut llfree_t) -> Meta;

    /// Allocates a frame and returns its address, or a negative error code
    fn llfree_get(this: *const llfree_t, core: c_size_t, flags: flags_t) -> result_t;
    /// Frees a frame, returning 0 on success or a negative error code
    fn llfree_put(this: *const llfree_t, core: c_size_t, frame: u64, flags: flags_t) -> result_t;

    /// Frees a frame, returning 0 on success or a negative error code
    fn llfree_drain(this: *const llfree_t, core: c_size_t) -> result_t;

    /// Returns the number of cores this allocator was initialized with
    fn llfree_cores(this: *const llfree_t) -> c_size_t;
    /// Returns the total number of frames the allocator can allocate
    fn llfree_frames(this: *const llfree_t) -> c_size_t;

    /// Checks if a frame is allocated, returning 0 if not
    fn llfree_is_free(this: *const llfree_t, frame: u64, order: c_size_t) -> bool;
    /// Returns the number of frames in the given chunk.
    /// This is only implemented for 0, HUGE_ORDER and TREE_ORDER.
    fn llfree_free_at(this: *const llfree_t, frame: u64, order: c_size_t) -> c_size_t;

    /// Returns number of currently free frames
    fn llfree_free_frames(this: *const llfree_t) -> c_size_t;
    /// Returns number of currently free huge frames
    fn llfree_free_huge(this: *const llfree_t) -> c_size_t;

    /// Prints the allocators state for debugging
    fn llfree_print_debug(
        this: *const llfree_t,
        writer: extern "C" fn(*mut c_void, *const c_char),
        arg: *mut c_void,
    );
}

#[cfg(test)]
mod test {
    use super::super::test::TestAlloc;
    use super::LLC;
    use crate::Init;

    #[test]
    fn test_debug() {
        let alloc = TestAlloc::<LLC>::create(1, 1024, Init::FreeAll).unwrap();
        println!("{alloc:?}");
    }
}
