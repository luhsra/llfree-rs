//! Generic atomics

use core::cell::UnsafeCell;
use core::fmt;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::Ordering::*;
use core::sync::atomic::*;

use log::debug;

use crate::util::Align;

/// Atomic value
///
/// See [core::sync::atomic::AtomicU64] for the documentation.
#[repr(transparent)]
pub struct Atom<T: Atomic>(pub T::I);

impl<T: Atomic> Atom<T> {
    pub fn new(v: T) -> Self {
        Self(T::I::new(v.into()))
    }
    #[cfg_attr(feature = "log_debug", track_caller)]
    pub fn load(&self) -> T {
        debug!("{} load", core::panic::Location::caller());
        self.0.load().into()
    }
    #[cfg_attr(feature = "log_debug", track_caller)]
    pub fn store(&self, v: T) {
        debug!("{} store", core::panic::Location::caller());
        self.0.store(v.into())
    }
    #[cfg_attr(feature = "log_debug", track_caller)]
    pub fn swap(&self, v: T) -> T {
        debug!("{} swap", core::panic::Location::caller());
        self.0.swap(v.into()).into()
    }
    #[cfg_attr(feature = "log_debug", track_caller)]
    pub fn compare_exchange(&self, current: T, new: T) -> Result<T, T> {
        debug!("{} cmpxchg", core::panic::Location::caller());
        match self.0.compare_exchange(current.into(), new.into()) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    #[cfg_attr(feature = "log_debug", track_caller)]
    pub fn compare_exchange_weak(&self, current: T, new: T) -> Result<T, T> {
        debug!("{} cmpxchgw", core::panic::Location::caller());
        match self.0.compare_exchange_weak(current.into(), new.into()) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    #[cfg_attr(feature = "log_debug", track_caller)]
    pub fn fetch_update<F: FnMut(T) -> Option<T>>(&self, mut f: F) -> Result<T, T> {
        debug!("{} update", core::panic::Location::caller());
        match self.0.fetch_update(|v| f(v.into()).map(|v| v.into())) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
}
impl<T: Atomic + Default> Default for Atom<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}
impl<T: Atomic + fmt::Debug> fmt::Debug for Atom<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.load().fmt(f)
    }
}

/// Types that can be converted from/into atomics
///
/// # Note
/// For compare_exchange and fetch_update, equality on this type has to be
/// the same as when they are converted into the underlying integers.
///
/// `a == b <-> a.into() == b.into()`
pub trait Atomic:
    Sized + Copy + Into<<Self::I as AtomicImpl>::V> + From<<Self::I as AtomicImpl>::V>
{
    type I: AtomicImpl;
}

/// Implementation of the atomic values
pub trait AtomicImpl: Sized {
    type V: Sized + Eq + Copy;
    fn new(v: Self::V) -> Self;
    fn load(&self) -> Self::V;
    fn store(&self, v: Self::V);
    fn swap(&self, v: Self::V) -> Self::V;
    fn compare_exchange(&self, current: Self::V, new: Self::V) -> Result<Self::V, Self::V>;
    fn compare_exchange_weak(&self, current: Self::V, new: Self::V) -> Result<Self::V, Self::V>;
    fn fetch_update<F: FnMut(Self::V) -> Option<Self::V>>(&self, f: F) -> Result<Self::V, Self::V>;

    fn fetch_min(&self, v: Self::V) -> Self::V;
    fn fetch_max(&self, v: Self::V) -> Self::V;
    fn fetch_add(&self, v: Self::V) -> Self::V;
    fn fetch_sub(&self, v: Self::V) -> Self::V;
    fn fetch_and(&self, v: Self::V) -> Self::V;
    fn fetch_or(&self, v: Self::V) -> Self::V;
    fn fetch_xor(&self, v: Self::V) -> Self::V;
    fn fetch_nand(&self, v: Self::V) -> Self::V;
}

macro_rules! atomic_trivial {
    ($($name:ident),+) => {
        $(
            fn $name(&self, v: Self::V) -> Self::V {
                self.$name(v.into(), AcqRel).into()
            }
        )+
    };
}

macro_rules! fn_trivial {
    ($ty:ident ; $($name:ident),+) => {
        $(
            pub fn $name(&self, v: $ty) -> $ty {
                AtomicImpl::$name(&self.0, v)
            }
        )+
    };
}

macro_rules! atomic_impl {
    ($ty:ident, $atomic:ident) => {
        impl Atomic for $ty {
            type I = $atomic;
        }
        impl AtomicImpl for $atomic {
            type V = $ty;
            fn new(v: Self::V) -> Self {
                Self::new(v)
            }
            fn load(&self) -> Self::V {
                self.load(Acquire)
            }
            fn store(&self, v: Self::V) {
                self.store(v, Release)
            }
            fn compare_exchange(&self, current: Self::V, new: Self::V) -> Result<Self::V, Self::V> {
                self.compare_exchange(current, new, AcqRel, Acquire)
            }
            fn compare_exchange_weak(
                &self,
                current: Self::V,
                new: Self::V,
            ) -> Result<Self::V, Self::V> {
                self.compare_exchange_weak(current, new, AcqRel, Acquire)
            }
            fn fetch_update<F: FnMut(Self::V) -> Option<Self::V>>(
                &self,
                f: F,
            ) -> Result<Self::V, Self::V> {
                self.fetch_update(AcqRel, Acquire, f)
            }
            atomic_trivial![
                swap, fetch_min, fetch_max, fetch_add, fetch_sub, fetch_and, fetch_or, fetch_xor, fetch_nand
            ];
        }

        impl Atom<$ty> {
            fn_trivial![
                $ty; fetch_min, fetch_max, fetch_add, fetch_sub, fetch_and, fetch_or, fetch_xor, fetch_nand
            ];
        }
    };
}

atomic_impl!(u8, AtomicU8);
atomic_impl!(u16, AtomicU16);
atomic_impl!(u32, AtomicU32);
atomic_impl!(u64, AtomicU64);
atomic_impl!(usize, AtomicUsize);

pub trait AtomArray<T: Copy, const L: usize> {
    /// Overwrite the content of the whole array non-atomically.
    ///
    /// This is faster than atomics but does not handle race conditions.
    fn atomic_fill(&self, e: T);
}

impl<T: Atomic, const L: usize> AtomArray<T, L> for [Atom<T>; L] {
    fn atomic_fill(&self, e: T) {
        // cast to raw memory to let the compiler use vector instructions
        #[allow(invalid_reference_casting)]
        let mem = unsafe { &mut *(self.as_ptr() as *mut [T; L]) };
        mem.fill(e);
        // memory ordering has to be enforced with a memory barrier
        fence(Release);
    }
}

/// Very simple spin lock implementation
pub struct Spin<T> {
    lock: AtomicBool,
    /// Cache aligned value -> no races with lock
    value: Align<UnsafeCell<T>>,
}

impl<T> Spin<T> {
    pub const fn new(value: T) -> Self {
        Self {
            lock: AtomicBool::new(false),
            value: Align(UnsafeCell::new(value)),
        }
    }
    pub fn lock(&self) -> SpinGuard<T> {
        while let Err(_) = self
            .lock
            .compare_exchange_weak(false, true, Acquire, Relaxed)
        {
            core::hint::spin_loop();
        }
        SpinGuard { spin: self }
    }
    pub fn try_lock(&self) -> Option<SpinGuard<T>> {
        if let Ok(_) = self.lock.compare_exchange(false, true, Acquire, Relaxed) {
            Some(SpinGuard { spin: self })
        } else {
            None
        }
    }
}
impl<T: Default> Default for Spin<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

pub struct SpinGuard<'a, T> {
    spin: &'a Spin<T>,
}
impl<T> Deref for SpinGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.spin.value.get() }
    }
}
impl<T> DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.spin.value.get() }
    }
}
impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) {
        self.spin.lock.store(false, Release);
    }
}
