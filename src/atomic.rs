use core::fmt;
use std::marker::PhantomData;
use std::ops::Index;
use std::sync::atomic::{AtomicU64, Ordering};

use log::error;

use crate::entry::Entry3;

/// Atomic wrapper for 64bit-sized values.
#[repr(transparent)]
pub struct Atomic<T: From<u64> + Into<u64>>(pub AtomicU64, PhantomData<T>);

const _: () = assert!(std::mem::size_of::<Atomic<u64>>() == 8);

impl<T: From<u64> + Into<u64>> Atomic<T> {
    #[inline]
    pub fn new(v: T) -> Self {
        Self(AtomicU64::new(v.into()), PhantomData)
    }
    #[inline]
    pub fn compare_exchange(&self, current: T, new: T) -> Result<T, T> {
        match self.0.compare_exchange(
            current.into(),
            new.into(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    #[inline]
    pub fn update<F: FnMut(T) -> Option<T>>(&self, mut f: F) -> Result<T, T> {
        match self
            .0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                f(v.into()).map(T::into)
            }) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    #[inline]
    pub fn load(&self) -> T {
        self.0.load(Ordering::SeqCst).into()
    }
    #[inline]
    pub fn store(&self, v: T) {
        self.0.store(v.into(), Ordering::SeqCst);
    }
    #[inline]
    pub fn swap(&self, v: T) -> T {
        self.0.swap(v.into(), Ordering::SeqCst).into()
    }
}

/// Node of an atomic stack
pub trait ANode: Copy + From<u64> + Into<u64> {
    fn next(self) -> Option<usize>;
    fn with_next(self, next: Option<usize>) -> Self;
}

impl ANode for Entry3 {
    fn next(self) -> Option<usize> {
        match self.idx() {
            Entry3::IDX_MAX => None,
            v => Some(v),
        }
    }

    fn with_next(self, next: Option<usize>) -> Self {
        self.with_idx(next.unwrap_or(Entry3::IDX_MAX))
    }
}

/// Simple atomic stack with atomic entries.
/// It is constructed over an already existing fixed size buffer.
#[repr(align(64))] // Just to be sure
pub struct AStack<T: ANode> {
    start: Atomic<T>,
}

unsafe impl<T: ANode> Send for AStack<T> {}
unsafe impl<T: ANode> Sync for AStack<T> {}

impl<T: ANode> Default for AStack<T> {
    fn default() -> Self {
        Self {
            start: Atomic::new(T::from(0).with_next(None)),
        }
    }
}

impl<T: ANode> AStack<T> {
    /// Pushes the element at `idx` to the front of the stack.
    pub fn push<B>(&self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        let mut start = self.start.load();
        let elem = &buf[idx];
        loop {
            if elem.update(|v| Some(v.with_next(start.next()))).is_err() {
                panic!();
            }
            match self
                .start
                .compare_exchange(start, start.with_next(Some(idx)))
            {
                Ok(_) => return,
                Err(s) => start = s,
            }
        }
    }

    /// Poping the first element and updating it in place.
    pub fn pop_update<B, F>(&self, buf: &B, mut f: F) -> Option<(usize, Result<T, T>)>
    where
        B: Index<usize, Output = Atomic<T>>,
        F: FnMut(T) -> Option<T>,
    {
        let mut start = self.start.load();
        loop {
            let idx = start.next()?;
            let next = buf[idx].load().next();
            match self.start.compare_exchange(start, start.with_next(next)) {
                Ok(old) => {
                    let i = old.next()?;
                    return Some((i, buf[i].update(|v| f(v).map(|v| v.with_next(None)))));
                }
                Err(s) => start = s,
            }
        }
    }

    /// Poping the first element returning its index.
    pub fn pop<B>(&self, buf: &B) -> Option<usize>
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        self.pop_update(buf, |v| Some(v)).map(|v| v.0)
    }
}

#[allow(dead_code)]
pub struct AStackDbg<'a, T, B>(pub &'a AStack<T>, pub &'a B)
where
    T: ANode,
    B: Index<usize, Output = Atomic<T>>;

impl<'a, T, B> fmt::Debug for AStackDbg<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atomic<T>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_list();

        match self.0.start.load().next() {
            None => {}
            Some(i) => {
                let mut i = i as usize;
                let mut ended = false;
                for _ in 0..1000 {
                    dbg.entry(&i);
                    let elem = self.1[i].load();
                    if let Some(next) = elem.next() {
                        i = next;
                    } else {
                        ended = true;
                        break;
                    }
                }
                if !ended {
                    error!("Circular List!");
                }
            }
        }

        dbg.finish()
    }
}

#[cfg(test)]
mod test {
    use core::marker::PhantomData;
    use core::sync::atomic::AtomicU64;
    use std::sync::Arc;

    use spin::Barrier;

    use crate::{thread, util::black_box};

    use super::{ANode, AStack, AStackDbg, Atomic};

    const DATA_V: Atomic<u64> = Atomic(AtomicU64::new(0), PhantomData);
    const N: usize = 64;
    static mut DATA: [Atomic<u64>; N] = [DATA_V; N];

    #[test]
    fn atomic_stack() {
        impl ANode for u64 {
            fn next(self) -> Option<usize> {
                (self != u64::MAX).then(|| self as _)
            }
            fn with_next(self, next: Option<usize>) -> Self {
                next.map(|v| v as u64).unwrap_or(u64::MAX)
            }
        }

        let stack = AStack::default();
        stack.push(unsafe { &DATA }, 0);
        stack.push(unsafe { &DATA }, 1);

        println!("{:?}", AStackDbg(&stack, unsafe { &DATA }));

        assert_eq!(stack.pop(unsafe { &DATA }), Some(1));
        assert_eq!(stack.pop(unsafe { &DATA }), Some(0));
        assert_eq!(stack.pop(unsafe { &DATA }), None);

        // Stress test

        const THREADS: usize = 8;
        const I: usize = N / THREADS;
        let barrier = Arc::new(Barrier::new(THREADS));
        let stack = Arc::new(stack);
        let copy = stack.clone();
        thread::parallel(THREADS, move |t| {
            thread::pin(t);
            let mut idx: [usize; I] = [0; I];
            for i in 0..I {
                idx[i] = t * I + i;
            }
            barrier.wait();

            for _ in 0..1000 {
                for &i in &idx {
                    stack.push(unsafe { &DATA }, i);
                }
                idx = black_box(idx);
                for (i, &a) in idx.iter().enumerate() {
                    for (j, &b) in idx.iter().enumerate() {
                        assert!(i == j || a != b);
                    }
                }
                for i in &mut idx {
                    *i = stack.pop(unsafe { &DATA }).unwrap();
                }
            }
        });
        assert_eq!(copy.pop(unsafe { &DATA }), None);
    }
}
