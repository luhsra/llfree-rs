use core::cell::UnsafeCell;
use core::fmt;
use core::ops::Deref;
use core::slice;

use log::error;

use crate::atomic::{Atom, Atomic};
use crate::cache::{Align, Aligned, CacheLine, Invalidate, UnsafeCacheLine};
use crate::{Error, Result};

/// Wrapper around non cache-coherent CXL data
///
/// Due to the missing cache coherency, the data *has* to be cache aligned and
/// externally synchronized.
/// Any (false-)sharing effects lead to inconsistent data.
#[repr(transparent)]
pub struct Uncached<T: ?Sized + Aligned>(T);

impl<T: ?Sized + Aligned> Invalidate for Uncached<T> {}
impl<T: Aligned> Invalidate for &Uncached<T> {}
impl<T: Sized + Aligned> Invalidate for [Uncached<T>] {}

unsafe impl<T: ?Sized + Aligned> Aligned for Uncached<T> {
    unsafe fn cache_line(&self) -> *const CacheLine {
        unsafe { self.0.cache_line() }
    }
    fn cache_lines(&self) -> usize {
        self.0.cache_lines()
    }
}

impl<T: Sized + Aligned> Uncached<T> {
    pub fn new(data: T) -> Self {
        Self(data)
    }
}
impl<T: ?Sized + Aligned> Uncached<T> {
    /// Create a new reference to the given data
    pub fn from_ref(data: &T) -> &Self {
        // SAFETY: `Uncached<T>` has the same memory layout as `T` due to #[repr(transparent)].
        unsafe { &*(data as *const T as *const Self) }
    }

    /// Unsynchronized access to the internal data
    pub unsafe fn inner(&self) -> &T {
        &self.0
    }

    /// Cast this uncached slice into another aligned type.
    /// `len` is the number of elements in the new type.
    ///
    /// Errors if the new slice is larger than this slice.
    pub unsafe fn cast<R: Aligned>(&self) -> Result<&Uncached<R>> {
        if self.cache_lines() >= size_of::<R>().div_ceil(CacheLine::SIZE) {
            Ok(unsafe { &*(self as *const Self as *const Uncached<R>) })
        } else {
            Err(Error::Initialization)
        }
    }
    /// Cast this uncached slice into another aligned type.
    /// `len` is the number of elements in the new type.
    ///
    /// Errors if the new slice is larger than this slice.
    pub unsafe fn cast_slice<R: Aligned>(&self, len: usize) -> Result<&Uncached<[R]>> {
        if len
            .checked_mul(size_of::<R>())
            .is_some_and(|size| size <= core::mem::size_of_val(self))
        {
            Ok(Uncached::from_ref(unsafe {
                slice::from_raw_parts(self.cache_line().cast::<R>(), len)
            }))
        } else {
            Err(Error::Initialization)
        }
    }

    /// Take ownership of the data, assuming no other host is modifying it.
    ///
    /// # Safety
    /// The caller must ensure that no other host is modifying the data.
    pub unsafe fn borrow<'a>(&'a self) -> Cached<'a, T> {
        self.flush_invalidate();
        Cached(self)
    }
}

// Specialization for atomic values -> need only inter host sync
impl<T: Atomic> Uncached<Align<Atom<T>>> {
    /// Specialized read method for [`Atom`] values.
    pub fn load(&self) -> T {
        self.flush_invalidate();
        // Invalidate before load to avoid reading pre-fetched data.
        self.0.load()
    }
    /// Specialized write method for [`Atom`] values.
    pub fn store(&self, value: T) {
        self.0.store(value);
        self.flush_invalidate();
    }
}

// Internal mutability as unsafe cell -> need inner AND inter host sync
impl<T: Copy> Uncached<Align<UnsafeCell<T>>> {
    /// Read `value` from memory, assuming the cache was previously invalidated,
    /// as in [`Self::write`].
    ///
    /// # Safety
    /// x86 has no cache-bypassing cache, thus we must avoid having modified
    /// cachelines while other hosts modify the memory.
    pub unsafe fn read(&self) -> T {
        // Invalidate as we do not know if something has been pre-fetched?
        // Here we assume that non-dirty cachelines are not flushed,
        // and do not overwrite the (externally changed) memory.
        self.flush_invalidate();
        unsafe { self.0.get().read_volatile() }
    }

    /// Commit `data` to memory and invalidate the cache.
    ///
    /// x86 has no way to invalidate cachelines without flushing them.
    /// Consequently, we have to avoid having modified cachelines while
    /// other hosts modify the memory.
    ///
    /// Also we have to assume that clflush(opt) does not flush cachelines
    /// that are only shared/exclusive but not modified.
    /// This is essential as the pre-fetcher might load any mapped memory
    /// into the cache at any time.
    /// This is also the reason why we have to invalidate again before reading.
    ///
    /// # Safety
    /// x86 has no cache-bypassing cache, thus we must avoid having modified
    /// cachelines while other hosts modify the memory.
    pub unsafe fn write(&self, value: T) {
        unsafe { self.0.get().write_volatile(value) };
        // Question: CLWB is not enough here as it does not guarantee that the
        // cachelines are transitioned from modified to shared/exclusive?
        self.flush_invalidate();
    }
}

// Specialization for arrays/slices!
impl<T: Aligned> Uncached<[T]> {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn at(&self, index: usize) -> &Uncached<T> {
        unsafe { core::mem::transmute(&self.0[index]) }
    }
    pub fn slice(&self, range: core::ops::Range<usize>) -> &Uncached<[T]> {
        unsafe { core::mem::transmute(&self.0[range]) }
    }
    pub fn iter(&self) -> impl Iterator<Item = &Uncached<T>> {
        self.0.iter().map(|t| unsafe { core::mem::transmute(t) })
    }
    #[allow(clippy::missing_transmute_annotations)]
    pub fn split_at(&self, index: usize) -> (&Uncached<[T]>, &Uncached<[T]>) {
        let (l, r) = self.0.split_at(index);
        unsafe { (core::mem::transmute(l), core::mem::transmute(r)) }
    }
    pub fn transpose(&self) -> &[Uncached<T>] {
        unsafe { core::mem::transmute(&self.0) }
    }
}

/// Temporary (host-)unique ownership of an [`Uncached`] value.
///
/// The data is flushed and invalidated on drop.
pub struct Cached<'a, T: ?Sized + Aligned>(&'a Uncached<T>);
impl<'a, T: ?Sized + Aligned> Deref for Cached<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.0
    }
}
impl<'a, T: Copy> Cached<'a, Align<UnsafeCell<T>>> {
    pub fn read(&self) -> T {
        unsafe { self.0.0.get().read_volatile() }
    }
    pub fn write(&mut self, value: T) {
        unsafe { self.0.0.get().write_volatile(value) }
    }
}
impl<'a, T: ?Sized + Aligned> Drop for Cached<'a, T> {
    fn drop(&mut self) {
        self.0.flush_invalidate()
    }
}
impl<'a, T: ?Sized + Aligned + fmt::Debug> fmt::Debug for Cached<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0.0, f)
    }
}

/// A lock that uses the Bakery algorithm to synchronize access
/// across multiple hosts with non-coherent memory.
///
/// It additionally needs inner host synchronization, like atomics or spinlocks.
pub struct CXLock<'a> {
    host_id: usize,
    entering: &'a Uncached<[Align<Atom<u64>>]>,
    number: &'a Uncached<[Align<Atom<u64>>]>,
}

impl<'a> CXLock<'a> {
    /// Size of the metadata buffer required for `hosts` in cache lines.
    pub fn metadata_size(hosts: usize) -> usize {
        2 * hosts
    }
    pub fn metadata(&self) -> &'a Uncached<[UnsafeCacheLine]> {
        unsafe {
            Uncached::from_ref(slice::from_raw_parts(
                self.entering.as_ptr().cast(),
                Self::metadata_size(self.hosts()),
            ))
        }
    }

    /// Returns the number of hosts participating in this lock.
    pub fn hosts(&self) -> usize {
        self.entering.len()
    }

    /// Initialize a new lock with the given host ID, number of hosts, and buffer.
    pub fn init(
        host_id: usize,
        hosts: usize,
        buffer: &'a Uncached<[UnsafeCacheLine]>,
    ) -> Result<Self> {
        let lock = Self::join(host_id, hosts, buffer)?;
        for i in 0..hosts {
            lock.entering.at(i).store(0);
            lock.number.at(i).store(0);
        }
        Ok(lock)
    }

    /// Join a lock that was created by another host.
    pub fn join(
        host_id: usize,
        hosts: usize,
        buffer: &'a Uncached<[UnsafeCacheLine]>,
    ) -> Result<Self> {
        if hosts == 0 || host_id >= hosts {
            return Err(Error::Initialization);
        }
        if buffer.cache_lines() < Self::metadata_size(hosts) {
            error!(
                "buffer {}CL too small for {hosts} hosts",
                buffer.cache_lines(),
            );
            return Err(Error::Initialization);
        }
        let buffer = unsafe { buffer.cast_slice::<Align<Atom<u64>>>(2 * hosts)? };
        let (entering, number) = buffer.split_at(hosts);
        assert!(entering.len() == hosts && number.len() == hosts);
        Ok(CXLock {
            entering,
            number,
            host_id,
        })
    }

    /// Acquire the lock, returning a guard that will unlock it when dropped.
    pub fn lock<'b>(&'b mut self) -> CXLockGuard<'b, 'a> {
        unsafe { self.lock_raw() };
        CXLockGuard(self)
    }

    /// Blocking call to get the lock.
    pub unsafe fn lock_raw(&mut self) {
        self.entering.at(self.host_id).store(1);

        let mut m = 0;
        for i in 0..self.number.len() {
            let n = self.number.at(i).load();
            if n > m {
                m = n;
            }
        }
        m += 1;
        self.number.at(self.host_id).store(m);

        self.entering.at(self.host_id).store(0);
        for i in 0..self.entering.len() {
            while self.entering.at(i).load() != 0 {}
            let mut num = self.number.at(i).load();
            while (num != 0) && ((num < m) || ((num == m) && (i < self.host_id))) {
                num = self.number.at(i).load();
            }
        }
    }

    /// Unlock the lock.
    pub unsafe fn unlock_raw(&mut self) {
        self.number.at(self.host_id).store(0);
    }
}

pub struct CXLockGuard<'b, 'a: 'b>(&'b mut CXLock<'a>);
impl<'a, 'b: 'a> Drop for CXLockGuard<'a, 'b> {
    fn drop(&mut self) {
        unsafe { self.0.unlock_raw() };
    }
}

#[cfg(test)]
mod tests {
    use std::boxed::Box;
    use std::cell::UnsafeCell;

    use crate::Error;
    use crate::atomic::Atom;
    use crate::cache::{Align, FLUSH_COUTER};

    use super::{Invalidate, Uncached};

    #[test]
    fn lock_rejects_invalid_host() {
        let data =
            unsafe { Box::<[crate::cache::UnsafeCacheLine]>::new_zeroed_slice(2).assume_init() };
        let buffer = Uncached::from_ref(&*data);
        assert!(matches!(
            super::CXLock::join(0, 0, buffer),
            Err(Error::Initialization)
        ));
        assert!(matches!(
            super::CXLock::join(1, 1, buffer),
            Err(Error::Initialization)
        ));
        assert!(super::CXLock::init(0, 1, buffer).is_ok());
    }

    #[test]
    fn flush_counter() {
        FLUSH_COUTER.with(|c| c.store(0));

        let data = unsafe { Box::<[Align<Atom<u64>>]>::new_zeroed_slice(8).assume_init() };
        let uncached = Uncached::from_ref(&*data);
        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 0);

        uncached.flush_invalidate();
        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 8);
        uncached.slice(0..4).flush_invalidate();
        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 8 + 4);
    }

    #[test]
    fn atomic_simple() {
        FLUSH_COUTER.with(|c| c.store(0));

        let data = unsafe { Box::<[Align<Atom<u64>>]>::new_zeroed_slice(8).assume_init() };

        let uncached = Uncached::from_ref(&*data);
        uncached.at(0).store(42);
        assert_eq!(uncached.at(0).load(), 42);

        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 2);
    }

    #[test]
    fn cell_simple() {
        FLUSH_COUTER.with(|c| c.store(0));

        let data = unsafe { Box::<[Align<UnsafeCell<u64>>]>::new_zeroed_slice(8).assume_init() };
        let uncached = Uncached::from_ref(&*data);
        unsafe {
            uncached.at(0).write(42);
            assert_eq!(uncached.at(0).read(), 42);
        }

        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 2);
    }

    #[test]
    fn cast_slice_length_and_bounds() {
        let data = [Align(1u64), Align(2), Align(3)];
        let uncached = Uncached::from_ref(&data[..]);

        let cast = unsafe { uncached.cast_slice::<Align<u64>>(2) }.unwrap();
        assert_eq!(cast.len(), 2);
        assert_eq!(unsafe { cast.at(1).inner().0 }, 2);
        assert_eq!(
            unsafe { uncached.cast_slice::<Align<u64>>(0) }
                .unwrap()
                .len(),
            0
        );
        assert!(matches!(
            unsafe { uncached.cast_slice::<Align<u64>>(4) },
            Err(Error::Initialization)
        ));
        assert!(matches!(
            unsafe { uncached.cast_slice::<Align<u64>>(usize::MAX) },
            Err(Error::Initialization)
        ));
    }

    #[test]
    fn atomic_borrow_flushes_on_drop() {
        FLUSH_COUTER.with(|c| c.store(0));
        let data = Align(Atom::new(0u64));
        let uncached = Uncached::from_ref(&data);

        {
            let cached = unsafe { uncached.borrow() };
            assert_eq!(FLUSH_COUTER.with(|c| c.load()), 1);
            cached.store(42);
            assert_eq!(cached.load(), 42);
        }
        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 2);
        assert_eq!(uncached.load(), 42);
        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 3);
    }

    #[test]
    fn cell_borrow_flushes_on_drop() {
        FLUSH_COUTER.with(|c| c.store(0));
        let data = Align(UnsafeCell::new(0u64));
        let uncached = Uncached::from_ref(&data);

        unsafe {
            {
                let cached = uncached.borrow();
                assert_eq!(FLUSH_COUTER.with(|c| c.load()), 1);
                cached.get().cast::<u64>().write_volatile(42);
                assert_eq!(cached.get().cast::<u64>().read_volatile(), 42);
            }
            assert_eq!(FLUSH_COUTER.with(|c| c.load()), 2);
            assert_eq!(uncached.read(), 42);
        }
        assert_eq!(FLUSH_COUTER.with(|c| c.load()), 3);
    }
}
