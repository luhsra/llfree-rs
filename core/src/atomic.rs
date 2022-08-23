use core::fmt;
use core::marker::PhantomData;
use core::ops::Index;
use core::sync::atomic::*;

use log::error;

use crate::entry::Entry3;

/// Atomic wrapper for the different integer sizes and values that can be converted into them.
#[repr(transparent)]
pub struct Atomic<T: AtomicValue>(pub <<T as AtomicValue>::V as AtomicT>::A);

const _: () = assert!(core::mem::size_of::<Atomic<u64>>() == 8);

impl<T: AtomicValue> Atomic<T> {
    pub fn new(v: T) -> Self {
        Self(T::V::atomic(v.into()))
    }
    pub fn compare_exchange(&self, current: T, new: T) -> Result<T, T> {
        match T::V::atomic_compare_exchange(&self.0, current.into(), new.into()) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    pub fn compare_exchange_weak(&self, current: T, new: T) -> Result<T, T> {
        match T::V::atomic_compare_exchange_weak(&self.0, current.into(), new.into()) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    pub fn update<F: FnMut(T) -> Option<T>>(&self, mut f: F) -> Result<T, T> {
        match T::V::atomic_update(&self.0, |v| f(v.into()).map(T::into)) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    pub fn load(&self) -> T {
        T::V::atomic_load(&self.0).into()
    }
    pub fn store(&self, v: T) {
        T::V::atomic_store(&self.0, v.into());
    }
    pub fn swap(&self, v: T) -> T {
        T::V::atomic_swap(&self.0, v.into()).into()
    }
}

/// Value that can be converted into an atomic type.
pub trait AtomicValue: From<Self::V> + Into<Self::V> + Clone + Copy {
    type V: AtomicT;
}

/// An atomic type with atomic functions.
pub trait AtomicT: Sized + Clone + Copy {
    type A;
    /// Specifies the memory ordering used by this type.
    const ORDER: Ordering = Ordering::SeqCst;

    fn atomic(v: Self) -> Self::A;
    fn atomic_compare_exchange(atomic: &Self::A, current: Self, new: Self) -> Result<Self, Self>;
    fn atomic_compare_exchange_weak(
        atomic: &Self::A,
        current: Self,
        new: Self,
    ) -> Result<Self, Self>;
    fn atomic_update<F: FnMut(Self) -> Option<Self>>(atomic: &Self::A, f: F) -> Result<Self, Self>;
    fn atomic_load(atomic: &Self::A) -> Self;
    fn atomic_store(atomic: &Self::A, v: Self);
    fn atomic_swap(atomic: &Self::A, v: Self) -> Self;
}

macro_rules! impl_atomic {
    ($value:ident, $atomic:ident) => {
        impl AtomicValue for $value {
            type V = $value;
        }

        impl AtomicT for $value {
            type A = $atomic;

            fn atomic(v: Self) -> Self::A {
                Self::A::new(v)
            }

            fn atomic_compare_exchange(
                atomic: &Self::A,
                current: Self,
                new: Self,
            ) -> Result<Self, Self> {
                Self::A::compare_exchange(atomic, current, new, Self::ORDER, Self::ORDER)
            }

            fn atomic_compare_exchange_weak(
                atomic: &Self::A,
                current: Self,
                new: Self,
            ) -> Result<Self, Self> {
                Self::A::compare_exchange_weak(atomic, current, new, Self::ORDER, Self::ORDER)
            }

            fn atomic_update<F: FnMut(Self) -> Option<Self>>(
                atomic: &Self::A,
                f: F,
            ) -> Result<Self, Self> {
                Self::A::fetch_update(atomic, Self::ORDER, Self::ORDER, f)
            }

            fn atomic_load(atomic: &Self::A) -> Self {
                Self::A::load(atomic, Self::ORDER)
            }

            fn atomic_store(atomic: &Self::A, v: Self) {
                Self::A::store(atomic, v, Self::ORDER)
            }

            fn atomic_swap(atomic: &Self::A, v: Self) -> Self {
                Self::A::swap(atomic, v, Self::ORDER)
            }
        }
    };
}

impl_atomic!(usize, AtomicUsize);
impl_atomic!(u64, AtomicU64);
impl_atomic!(u32, AtomicU32);
impl_atomic!(u16, AtomicU16);
impl_atomic!(u8, AtomicU8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Next {
    Some(usize),
    End,
    Outside,
}

impl Next {
    fn opt(v: Option<usize>) -> Self {
        match v {
            Some(i) => Self::Some(i),
            None => Self::End,
        }
    }
    fn some(self) -> Option<usize> {
        match self {
            Next::Some(i) => Some(i),
            Next::End => None,
            Next::Outside => unreachable!(),
        }
    }
}

/// Node of an atomic stack
pub trait ANode: AtomicValue + Default {
    fn next(self) -> Next;
    fn with_next(self, next: Next) -> Self;
    fn enqueue(self, next: Next) -> Option<Self> {
        (self.next() == Next::Outside).then_some(self.with_next(next))
    }
}

impl ANode for Entry3 {
    fn next(self) -> Next {
        match self.idx() {
            Entry3::IDX_MAX => Next::Outside,
            Entry3::IDX_END => Next::End,
            i => Next::Some(i),
        }
    }

    fn with_next(self, next: Next) -> Self {
        self.with_idx(match next {
            Next::Some(i) => i,
            Next::End => Entry3::IDX_END,
            Next::Outside => Entry3::IDX_MAX,
        })
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
            start: Atomic::new(T::default().with_next(Next::End)),
        }
    }
}

impl<T: ANode> AStack<T> {
    pub fn set(&self, v: T) {
        self.start.store(v)
    }

    /// Pushes the element at `idx` to the front of the stack.
    pub fn push<B>(&self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        let mut prev = self.start.load();
        let elem = &buf[idx];
        loop {
            if elem.update(|v| Some(v.with_next(prev.next()))).is_err() {
                panic!();
            }
            // CAS weak is important for fetch-update!
            match self
                .start
                .compare_exchange_weak(prev, prev.with_next(Next::Some(idx)))
            {
                Ok(_) => return,
                Err(s) => prev = s,
            }
        }
    }

    /// Poping the first element and updating it in place.
    pub fn pop_update<B, F>(&self, buf: &B, mut f: F) -> Option<(usize, Result<T, T>)>
    where
        B: Index<usize, Output = Atomic<T>>,
        F: FnMut(T) -> Option<T>,
    {
        let mut prev = self.start.load();
        loop {
            let idx = prev.next().some()?;
            let next = buf[idx].load().next();
            // CAS weak is important for fetch-update!
            match self.start.compare_exchange_weak(prev, prev.with_next(next)) {
                Ok(old) => {
                    let i = old.next().some()?;
                    return Some((
                        i,
                        buf[i].update(|v| f(v).map(|v| v.with_next(Next::Outside))),
                    ));
                }
                Err(s) => prev = s,
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

/// Debug printer for the [AStack].
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
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut dbg = f.debug_list();

        if let Next::Some(i) = self.0.start.load().next() {
            let mut i = i as usize;
            let mut ended = false;
            for _ in 0..1000 {
                dbg.entry(&i);
                let elem = self.1[i].load();
                if let Next::Some(next) = elem.next() {
                    if i == next {
                        break;
                    }
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

        dbg.finish()
    }
}

/// Simple linked list over a buffer of atomic entries.
#[derive(Default)]
pub struct BufList<T: ANode> {
    start: Option<usize>,
    end: Option<usize>,
    _phantom: PhantomData<T>,
}

impl<T: ANode> BufList<T> {
    pub fn clear<B>(&mut self, buf: &B)
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        while self.pop(buf).is_some() {}
    }

    pub fn push<B>(&mut self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        if buf[idx]
            .update(|v| v.enqueue(Next::opt(self.start)))
            .is_err()
        {
            return;
        }

        if self.start.is_none() {
            self.end = Some(idx);
        }
        self.start = Some(idx);
    }

    pub fn push_back<B>(&mut self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        if buf[idx].update(|v| v.enqueue(Next::End)).is_err() {
            return;
        }

        if let Some(end) = self.end {
            if buf[end]
                .update(|v| Some(v.with_next(Next::Some(idx))))
                .is_err()
            {
                unreachable!();
            }
        }
        self.end = Some(idx);
        if self.start.is_none() {
            self.start = Some(idx);
        }
    }

    /// Poping the first element and updating it in place.
    pub fn pop<B>(&mut self, buf: &B) -> Option<usize>
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        let start = self.start?;
        if let Ok(pte) = buf[start].update(|v| Some(v.with_next(Next::Outside))) {
            self.start = pte.next().some();
            if self.start.is_none() {
                self.end = None;
            }
            Some(start)
        } else {
            unreachable!()
        }
    }

    pub fn iter<'a, B>(&'a self, buf: &'a B) -> BufListIter<'a, T, B>
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        BufListIter {
            next: self.start,
            buf,
        }
    }
}

pub struct BufListIter<'a, T: ANode, B: Index<usize, Output = Atomic<T>>> {
    next: Option<usize>,
    buf: &'a B,
}

impl<'a, T, B> Iterator for BufListIter<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atomic<T>>,
{
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.next {
            let ret = self.buf[next].load();
            self.next = ret.next().some();
            Some((next, ret))
        } else {
            None
        }
    }
}

impl<'a, T, B> fmt::Debug for BufListIter<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atomic<T>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut dbg = f.debug_list();
        let iter = BufListIter {
            next: self.next,
            buf: self.buf,
        };
        for (i, _) in iter.take(100) {
            dbg.entry(&i);
        }
        dbg.finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::sync::atomic::AtomicU64;
    use std::sync::Arc;

    use spin::Barrier;

    use crate::{atomic::BufList, thread, util::black_box};

    use super::{ANode, AStack, AStackDbg, Atomic, Next};

    impl ANode for u64 {
        fn next(self) -> Next {
            const END: u64 = u64::MAX - 1;
            match self {
                u64::MAX => Next::Outside,
                END => Next::End,
                v => Next::Some(v as _),
            }
        }
        fn with_next(self, next: Next) -> Self {
            match next {
                Next::Some(i) => i as _,
                Next::End => u64::MAX - 1,
                Next::Outside => u64::MAX,
            }
        }
    }

    #[test]
    fn atomic_stack() {
        const DATA_V: Atomic<u64> = Atomic(AtomicU64::new(0));
        const N: usize = 64;
        let data: [Atomic<u64>; N] = [DATA_V; N];

        let stack = AStack::default();
        stack.push(&data, 0);
        stack.push(&data, 1);

        println!("{:?}", AStackDbg(&stack, &data));

        assert_eq!(stack.pop(&data), Some(1));
        assert_eq!(stack.pop(&data), Some(0));
        assert_eq!(stack.pop(&data), None);

        // Stress test

        const THREADS: usize = 6;
        const I: usize = N / THREADS;
        let barrier = Arc::new(Barrier::new(THREADS));
        let stack = Arc::new(stack);
        let copy = stack.clone();
        thread::parallel(THREADS, |t| {
            thread::pin(t);
            let mut idx: [usize; I] = [0; I];
            for i in 0..I {
                idx[i] = t * I + i;
            }
            barrier.wait();

            for _ in 0..1000 {
                for &i in &idx {
                    stack.push(&data, i);
                }
                idx = black_box(idx);
                for (i, &a) in idx.iter().enumerate() {
                    for (j, &b) in idx.iter().enumerate() {
                        assert!(i == j || a != b);
                    }
                }
                for i in &mut idx {
                    *i = stack.pop(&data).unwrap();
                }
            }
        });
        assert_eq!(copy.pop(&data), None);
    }

    #[test]
    fn buf_list() {
        const DATA_V: Atomic<u64> = Atomic(AtomicU64::new(u64::MAX));
        const N: usize = 64;
        let data: [Atomic<u64>; N] = [DATA_V; N];

        let mut list = BufList::default();
        assert_eq!(list.pop(&data), None);
        list.push(&data, 0);
        list.push(&data, 1);
        list.push_back(&data, 63);
        list.push_back(&data, 62);

        assert_eq!(list.pop(&data), Some(1));
        assert_eq!(list.pop(&data), Some(0));
        assert_eq!(list.pop(&data), Some(63));
        assert_eq!(list.pop(&data), Some(62));
        assert_eq!(list.pop(&data), None);
    }
}
