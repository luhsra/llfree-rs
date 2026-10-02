//! CPU-cache abstractions

use core::arch;
use core::cell::UnsafeCell;
use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::Deref;
use core::ops::DerefMut;
#[cfg(test)]
use core::sync::atomic::AtomicUsize;

#[cfg(test)]
use crate::atomic::Atom;

#[repr(align(64))]
pub struct CacheLine(pub [u8; 64]);
impl CacheLine {
    pub const SIZE: usize = 64;
}
pub type UnsafeCacheLine = UnsafeCell<CacheLine>;

/// Cache alignment for T
#[derive(Clone, Default, Hash, PartialEq, Eq)]
#[repr(align(64))]
pub struct Align<T = ()>(pub T);

const _: () = assert!(align_of::<Align>() == 64);
const _: () = assert!(align_of::<Align<usize>>() == 64);
const _: () = assert!(size_of::<Align<usize>>() == 64);
const _: () = assert!(align_of::<CacheLine>() == align_of::<Align>());

impl<T> Deref for Align<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T> DerefMut for Align<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
impl<T: fmt::Debug> fmt::Debug for Align<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// Cache-aligned data where we can guarantee no false sharing.
///
/// # Safety
/// Implementations must ensure that `cache_line()` returns a non-null,
/// cache-line-aligned pointer to storage that remains valid while `self` is borrowed.
/// The `cache_lines()` consecutive cache lines starting at that pointer must lie
/// within the backing allocation and must not include unrelated data. The pointer
/// and line count must describe the same storage, including for forwarding wrappers.
///
/// Implementing this trait requires an explicit unsafe implementation:
///
/// ```compile_fail,E0200
/// use llfree::cache::{Aligned, CacheLine};
///
/// struct Metadata;
/// impl Aligned for Metadata {
///     unsafe fn cache_line(&self) -> *const CacheLine {
///         core::ptr::null()
///     }
/// }
/// ```
pub unsafe trait Aligned {
    /// Returns a pointer to the first cache line containing this data.
    unsafe fn cache_line(&self) -> *const CacheLine;
    /// Returns the number of cache lines this data spans.
    fn cache_lines(&self) -> usize {
        size_of_val(self).div_ceil(size_of::<CacheLine>())
    }
}
unsafe impl<T: ?Sized + Aligned> Aligned for &T {
    unsafe fn cache_line(&self) -> *const CacheLine {
        unsafe { (**self).cache_line() }
    }
    fn cache_lines(&self) -> usize {
        (**self).cache_lines()
    }
}
unsafe impl<T: Sized> Aligned for Align<T> {
    unsafe fn cache_line(&self) -> *const CacheLine {
        (self as *const Self).cast()
    }
}
unsafe impl<T: ?Sized + Aligned> Aligned for UnsafeCell<T> {
    unsafe fn cache_line(&self) -> *const CacheLine {
        self.get().cast_const().cast()
    }
}
unsafe impl<T: Sized + Aligned> Aligned for [T] {
    unsafe fn cache_line(&self) -> *const CacheLine {
        self.as_ptr().cast()
    }
}
unsafe impl<const N: usize, T: Sized + Aligned> Aligned for [T; N] {
    unsafe fn cache_line(&self) -> *const CacheLine {
        self.as_ptr().cast()
    }
}
unsafe impl Aligned for CacheLine {
    unsafe fn cache_line(&self) -> *const CacheLine {
        self
    }
}

/// Trait for types that contain data will be flushed manually (invalidate)
pub trait Invalidate: Aligned {
    /// Flush and invalidate the cache lines
    fn flush_invalidate(&self) {
        unsafe { for_cache_lines(self, |cl| clflushopt(cl)) };
        // Sync clflushopts
        unsafe { memory_fence() };
    }
    /// Write back cache lines, keeping them valid
    ///
    /// This is just for optimization purposes, and does not synchronize.
    fn write_back(&self) {
        unsafe { for_cache_lines(self, |cl| clwb(cl)) };
    }

    /// Pre-fetch the cache line for reading
    fn prefetch_read(&self) {
        unsafe { for_cache_lines(self, |cl| prefetch_read(cl)) };
    }
    /// Pre-fetch the cache line for writing
    fn prefetch_write(&self) {
        unsafe { for_cache_lines(self, |cl| prefetch_write(cl)) };
    }
}
impl Invalidate for CacheLine {}
impl Invalidate for [CacheLine] {}
impl<T> Invalidate for Align<T> {}
impl<T> Invalidate for [Align<T>] {}

#[inline(always)]
pub unsafe fn prefetch_read(cl: *const CacheLine) {
    const _: () = assert!(arch::x86_64::_MM_HINT_T1 == 2);
    unsafe { arch::x86_64::_mm_prefetch::<2>(cl.cast()) };
}
#[inline(always)]
pub unsafe fn prefetch_write(cl: *const CacheLine) {
    const _: () = assert!(arch::x86_64::_MM_HINT_ET1 == 6);
    unsafe { arch::x86_64::_mm_prefetch::<6>(cl.cast()) };
}

#[inline(always)]
pub unsafe fn clwb(cl: *const CacheLine) {
    unsafe { arch::asm!("clwb [{0}]", in(reg) cl, options(nostack)) };
}

#[cfg(test)]
thread_local! {
    pub (crate) static FLUSH_COUTER: Atom<usize> = const { Atom(AtomicUsize::new(0)) };
}

#[inline(always)]
pub unsafe fn clflushopt(cl: *const CacheLine) {
    #[cfg(test)]
    FLUSH_COUTER.with(|c| c.fetch_add(1));

    unsafe { arch::asm!("clflushopt [{0}]", in(reg) cl, options(nostack)) };
}
#[inline(always)]
pub unsafe fn clflush(cl: *const CacheLine) {
    unsafe { arch::x86_64::_mm_clflush(cl.cast()) };
}

#[inline(always)]
pub unsafe fn load_fence() {
    unsafe { arch::x86_64::_mm_lfence() };
}
#[inline(always)]
pub unsafe fn store_fence() {
    unsafe { arch::x86_64::_mm_sfence() };
}
#[inline(always)]
pub unsafe fn memory_fence() {
    unsafe { arch::x86_64::_mm_mfence() };
}

/// Applies `f` to each cache line of `aligned`.
#[inline(always)]
pub unsafe fn for_cache_lines<T: ?Sized + Aligned>(aligned: &T, f: impl Fn(*const CacheLine)) {
    unsafe {
        for i in 0..aligned.cache_lines() {
            f(aligned.cache_line().add(i));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn aligned_references_forward_to_pointee() {
        let data = [Align(0u8), Align(1u8)];
        let reference = &data[..];
        assert_eq!(Aligned::cache_lines(&reference), 2);
        assert_eq!(
            unsafe { Aligned::cache_line(&reference) },
            data.as_ptr().cast()
        );
    }

    #[test]
    fn test_aligned() {
        fn cl<T: Aligned + ?Sized>(data: &T) -> usize {
            data.cache_lines()
        }

        assert_eq!(cl(&Align(())), 0);
        assert_eq!(cl(&Align(0u8)), 1);
        assert_eq!(cl(&Align([0u8; 65])), 2);

        assert_eq!(cl(&UnsafeCell::new(Align(()))), 0);
        assert_eq!(cl(&UnsafeCell::new(Align(0u8))), 1);
        assert_eq!(cl(&UnsafeCell::new(Align([0u8; 65]))), 2);

        assert_eq!(cl(&CacheLine([0; 64])), 1);

        assert_eq!(cl(&[] as &[Align<u8>]), 0);
        assert_eq!(cl(&[Align(0u8)][..]), 1);
        assert_eq!(cl(&[Align(0u8), Align(1u8)][..]), 2);
        assert_eq!(cl(&[CacheLine([0; 64]), CacheLine([0; 64])][..]), 2);
    }
}
