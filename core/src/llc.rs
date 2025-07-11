use core::ffi::{c_char, c_void, CStr};
use core::mem::{align_of, size_of};
use core::{fmt, slice};

use bitfield_struct::bitfield;

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

const _: () = assert!(size_of::<Flags>() == 2);

unsafe impl Send for LLC {}
unsafe impl Sync for LLC {}

impl<'a> Alloc<'a> for LLC {
    fn name() -> &'static str {
        "LLC"
    }

    fn metadata_size(cores: usize, frames: usize) -> crate::MetaSize {
        let m = unsafe { llfree_metadata_size(cores as _, frames as _) };
        crate::MetaSize {
            local: m.local,
            trees: m.trees,
            lower: m.lower,
        }
    }

    fn metadata(&mut self) -> super::MetaData<'a> {
        let cores = unsafe { llfree_cores(self.raw.as_ptr().cast()) };
        let ms = Self::metadata_size(cores, self.frames());
        let m = unsafe { llfree_metadata(self.raw.as_mut_ptr().cast()) };
        super::MetaData {
            local: unsafe { slice::from_raw_parts_mut(m.local, ms.local) },
            trees: unsafe { slice::from_raw_parts_mut(m.trees, ms.trees) },
            lower: unsafe { slice::from_raw_parts_mut(m.lower, ms.lower) },
        }
    }

    fn new(cores: usize, frames: usize, init: Init, meta: super::MetaData<'a>) -> Result<Self> {
        let mut raw = [0u8; size_of::<Self>()];

        let init = match init {
            Init::FreeAll => 0,
            Init::AllocAll => 1,
            Init::Recover(false) => 2,
            Init::Recover(true) => 3,
            Init::None => 4,
        };

        let m = unsafe { llfree_metadata_size(cores as _, frames as _) };
        assert!(size_of::<Self>() >= m.llfree);

        assert!(meta.valid(Self::metadata_size(cores, frames)));
        let meta = Meta {
            local: meta.local.as_mut_ptr(),
            trees: meta.trees.as_mut_ptr(),
            lower: meta.lower.as_mut_ptr(),
        };

        let ret = unsafe { llfree_init(raw.as_mut_ptr().cast(), cores as _, frames, init, meta) };
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

    fn validate(&self) {
        unsafe { llfree_validate(self.raw.as_ptr().cast()) }
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

#[bitfield(u64)]
#[allow(non_camel_case_types)]
struct result_t {
    #[bits(55)]
    frame: u64,
    reclaimed: bool,
    error: u8,
}

impl result_t {
    fn ok(self) -> Result<u64> {
        match self.error() {
            0 => Ok(self.frame()),
            1 => Err(Error::Memory),
            2 => Err(Error::Retry),
            3 => Err(Error::Address),
            4 => Err(Error::Initialization),
            _ => unreachable!("invalid return code"),
        }
    }
}

#[repr(C)]
struct MetaSize {
    llfree: usize,
    local: usize,
    trees: usize,
    lower: usize,
}

#[repr(C)]
struct Meta {
    local: *mut u8,
    trees: *mut u8,
    lower: *mut u8,
}

#[link(name = "llc", kind = "static")]
unsafe extern "C" {
    /// Initializes the allocator for the given memory region, returning 0 on success or a negative error code
    fn llfree_init(
        this: *mut llfree_t,
        cores: usize,
        frames: usize,
        init: u8,
        meta: Meta,
    ) -> result_t;

    /// Returns the size of the metadata buffers required for initialization
    fn llfree_metadata_size(cores: usize, frames: usize) -> MetaSize;

    /// Returns the metadata
    fn llfree_metadata(this: *mut llfree_t) -> Meta;

    /// Allocates a frame and returns its address, or a negative error code
    fn llfree_get(this: *const llfree_t, core: usize, flags: Flags) -> result_t;
    /// Frees a frame, returning 0 on success or a negative error code
    fn llfree_put(this: *const llfree_t, core: usize, frame: u64, flags: Flags) -> result_t;

    /// Frees a frame, returning 0 on success or a negative error code
    fn llfree_drain(this: *const llfree_t, core: usize) -> result_t;

    /// Returns the number of cores this allocator was initialized with
    fn llfree_cores(this: *const llfree_t) -> usize;
    /// Returns the total number of frames the allocator can allocate
    fn llfree_frames(this: *const llfree_t) -> usize;

    /// Checks if a frame is allocated, returning 0 if not
    fn llfree_is_free(this: *const llfree_t, frame: u64, order: usize) -> bool;
    /// Returns the number of frames in the given chunk.
    /// This is only implemented for 0, HUGE_ORDER and TREE_ORDER.
    fn llfree_free_at(this: *const llfree_t, frame: u64, order: usize) -> usize;

    /// Returns number of currently free frames
    fn llfree_free_frames(this: *const llfree_t) -> usize;
    /// Returns number of currently free huge frames
    fn llfree_free_huge(this: *const llfree_t) -> usize;

    /// Prints the allocators state for debugging
    fn llfree_print_debug(
        this: *const llfree_t,
        writer: extern "C" fn(*mut c_void, *const c_char),
        arg: *mut c_void,
    );

    /// Validates the allocator's state
    fn llfree_validate(this: *const llfree_t);
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
