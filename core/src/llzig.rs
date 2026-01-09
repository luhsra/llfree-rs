use core::cell::UnsafeCell;
use core::ffi::{CStr, c_char, c_void};
use core::mem::{align_of, size_of};
use core::{fmt, slice};

use super::{Alloc, Init};
use crate::util::Align;
use crate::{Flags, HUGE_ORDER, Result, Stats, TREE_FRAMES, TREE_HUGE};

/// C implementation of LLFree
///
/// Note: This abstraction assumes that the state is movable and smaller than two cache lines!
#[repr(transparent)]
pub struct LLZig {
    raw: UnsafeCell<[u8; 2 * align_of::<Align>()]>,
}

const _: () = assert!(size_of::<Flags>() == 2);

unsafe impl Send for LLZig {}
unsafe impl Sync for LLZig {}

impl<'a> Alloc<'a> for LLZig {
    fn name() -> &'static str {
        "LLC"
    }

    fn metadata_size(cores: usize, frames: usize) -> crate::MetaSize {
        let m = unsafe { bindings::llzig_metadata_size(cores as _, frames as _) };
        crate::MetaSize {
            local: m.local,
            trees: m.trees,
            lower: m.lower,
        }
    }

    fn metadata(&mut self) -> super::MetaData<'a> {
        let cores = unsafe { bindings::llzig_cores(self.raw.get().cast()) };
        let ms = Self::metadata_size(cores, self.frames());
        let m = unsafe { bindings::llzig_metadata(self.raw.get().cast()) };
        fn to_slice<'a>(ptr: *mut u8, len: usize) -> &'a mut [u8] {
            unsafe {
                ptr.as_mut()
                    .map_or(&mut [], |p| slice::from_raw_parts_mut(p, len))
            }
        }
        super::MetaData {
            local: to_slice(m.local, ms.local),
            trees: to_slice(m.trees, ms.trees),
            lower: to_slice(m.lower, ms.lower),
        }
    }

    fn new(cores: usize, frames: usize, init: Init, meta: super::MetaData<'a>) -> Result<Self> {
        let raw = UnsafeCell::new([0u8; size_of::<Self>()]);

        let init = match init {
            Init::FreeAll => 0,
            Init::AllocAll => 1,
            Init::Recover(false) => 2,
            Init::Recover(true) => 3,
            Init::None => 4,
        };

        let m = unsafe { bindings::llzig_metadata_size(cores as _, frames as _) };
        assert!(size_of::<Self>() >= m.llfree);

        assert!(meta.valid(Self::metadata_size(cores, frames)));
        let meta = bindings::llzig_meta {
            local: meta.local.as_mut_ptr(),
            trees: meta.trees.as_mut_ptr(),
            lower: meta.lower.as_mut_ptr(),
        };

        let ret = unsafe { bindings::llzig_init(raw.get().cast(), cores as _, frames, init, meta) };
        ret.ok().map(|_| LLZig { raw })
    }

    fn get(&self, core: usize, frame: Option<usize>, flags: Flags) -> Result<usize> {
        let ret = if let Some(frame) = frame {
            unsafe {
                bindings::llzig_get_at(self.raw.get().cast(), core as _, frame as _, flags.into())
            }
        } else {
            unsafe { bindings::llzig_get(self.raw.get().cast(), core as _, flags.into()) }
        };
        Ok(ret.ok()? as _)
    }

    fn put(&self, core: usize, frame: usize, flags: Flags) -> Result<()> {
        let ret = unsafe {
            bindings::llzig_put(self.raw.get().cast(), core as _, frame as _, flags.into())
        };
        ret.ok().map(|_| ())
    }

    fn is_free(&self, frame: usize, order: usize) -> bool {
        let stats =
            unsafe { bindings::llzig_full_stats_at(self.raw.get().cast(), frame as _, order as _) };
        order == 0 && stats.free_frames == 1
            || order == HUGE_ORDER && stats.free_huge == 1
            || order == TREE_FRAMES.ilog2() as usize && stats.free_huge == TREE_HUGE
    }

    fn drain(&self, core: usize) -> Result<()> {
        unsafe {
            bindings::llzig_drain(self.raw.get().cast(), core as _)
                .ok()
                .map(|_| ())
        }
    }

    fn frames(&self) -> usize {
        unsafe { bindings::llzig_frames(self.raw.get().cast()) as _ }
    }

    fn cores(&self) -> usize {
        unsafe { bindings::llzig_cores(self.raw.get().cast()) as _ }
    }

    fn fast_stats(&self) -> crate::Stats {
        unsafe { bindings::llzig_stats(self.raw.get().cast()).into() }
    }

    fn fast_stats_at(&self, frame: usize, order: usize) -> crate::Stats {
        unsafe { bindings::llzig_stats_at(self.raw.get().cast(), frame as _, order as _).into() }
    }

    fn stats(&self) -> crate::Stats {
        unsafe { bindings::llzig_full_stats(self.raw.get().cast()).into() }
    }

    fn stats_at(&self, frame: usize, order: usize) -> crate::Stats {
        unsafe {
            bindings::llzig_full_stats_at(self.raw.get().cast(), frame as _, order as _).into()
        }
    }

    fn validate(&self) {
        unsafe { bindings::llzig_validate(self.raw.get().cast()) }
    }
}

impl fmt::Debug for LLZig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // wrapper function that is called by the c implementation
        extern "C" fn writer(arg: *mut c_void, msg: *const c_char) {
            let f = unsafe { &mut *arg.cast::<fmt::Formatter<'_>>() };
            let c_str = unsafe { CStr::from_ptr(msg) };
            write!(f, "{}", c_str.to_str().unwrap()).unwrap();
        }

        unsafe {
            bindings::llzig_print_debug(
                self.raw.get().cast(),
                Some(writer),
                (f as *mut fmt::Formatter).cast(),
            )
        };

        Ok(())
    }
}

mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(unnecessary_transmutes)]

    include!(concat!(env!("OUT_DIR"), "/llzig.rs"));

    impl From<super::Flags> for llflags_t {
        fn from(flags: super::Flags) -> Self {
            llflags_t {
                order: flags.order() as _,
                movable: flags.movable(),
                zeroed: false,
                long_living: false,
            }
        }
    }

    impl llzig_result_t {
        pub fn ok(self) -> super::Result<u64> {
            match self.err {
                LLZIG_ERR_OK => Ok(self.frame),
                LLZIG_ERR_MEMORY => Err(crate::Error::Memory),
                LLZIG_ERR_RETRY => Err(crate::Error::Retry),
                LLZIG_ERR_ADDRESS => Err(crate::Error::Address),
                LLZIG_ERR_INIT => Err(crate::Error::Initialization),
                _ => unreachable!("invalid return code"),
            }
        }
    }

    impl From<ll_stats> for super::Stats {
        fn from(val: ll_stats) -> Self {
            super::Stats {
                free_frames: val.free_frames,
                free_huge: val.free_huge,
                free_trees: 0,
            }
        }
    }
}

#[cfg(all(test, feature = "std", feature = "llzig"))]
mod test {
    use super::super::alloc_test::TestAlloc;
    use super::LLZig;
    use crate::Init;

    #[test]
    fn test_debug() {
        let alloc = TestAlloc::<LLZig>::create(1, 1024, Init::FreeAll).unwrap();
        println!("{alloc:?}");
    }
}
