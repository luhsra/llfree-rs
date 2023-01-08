use core::fmt::{self, Debug};
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index};
use core::sync::atomic::{
    AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering::*,
};

use log::error;

use crate::entry::TreeNode;
use crate::util::CacheLine;

/// Atomic value
#[repr(transparent)]
pub struct Atom<T: Atomic>(T::I);

impl<T: Atomic> Atom<T> {
    pub const fn raw(v: T::I) -> Self {
        Self(v)
    }
    pub fn new(v: T) -> Self {
        Self(T::I::new(v.into()))
    }
    pub fn load(&self) -> T {
        self.0.load().into()
    }
    pub fn store(&self, v: T) {
        self.0.store(v.into())
    }
    pub fn swap(&self, v: T) -> T {
        self.0.swap(v.into()).into()
    }
    pub fn compare_exchange(&self, current: T, new: T) -> Result<T, T> {
        match self.0.compare_exchange(current.into(), new.into()) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    pub fn fetch_update<F: FnMut(T) -> Option<T>>(&self, mut f: F) -> Result<T, T> {
        match self.0.fetch_update(|v| f(v.into()).map(|v| v.into())) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
}

impl<T: Atomic + fmt::Debug> fmt::Debug for Atom<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.load().fmt(f)
    }
}

macro_rules! fn_trivial {
    ($($name:ident),+) => {
        $(
            pub fn $name(&self, v: T) -> T {
                self.0.$name(v.into()).into()
            }
        )+
    };
}

// For all types that are the same as the underlying implementation
impl<T: Atomic<I: AtomicImpl<V = T>>> Atom<T> {
    fn_trivial![
        fetch_min, fetch_max, fetch_add, fetch_sub, fetch_and, fetch_or, fetch_xor, fetch_nand
    ];
}

/// Types that can be converted from/into atomics
pub trait Atomic:
    Sized + Eq + Copy + Into<<Self::I as AtomicImpl>::V> + From<<Self::I as AtomicImpl>::V>
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
            fn fetch_update<F: FnMut(Self::V) -> Option<Self::V>>(
                &self,
                f: F,
            ) -> Result<Self::V, Self::V> {
                self.fetch_update(AcqRel, Acquire, f)
            }
            atomic_trivial![
                swap, fetch_min, fetch_max, fetch_add, fetch_sub, fetch_and, fetch_or, fetch_xor,
                fetch_nand
            ];
        }
    };
}

atomic_impl!(u8, AtomicU8);
atomic_impl!(u16, AtomicU16);
atomic_impl!(u32, AtomicU32);
atomic_impl!(u64, AtomicU64);
atomic_impl!(usize, AtomicUsize);

/// Very simple spin lock implementation
pub struct Spin<T> {
    lock: AtomicBool,
    /// Cache aligned value -> no races with lock
    value: CacheLine<T>,
}

impl<T> Spin<T> {
    pub const fn new(value: T) -> Self {
        Self {
            lock: AtomicBool::new(false),
            value: CacheLine(value),
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
impl<'a, T> Deref for SpinGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.spin.value
    }
}
impl<'a, T> DerefMut for SpinGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(&self.spin.value as *const _ as *mut _) }
    }
}
impl<'a, T> Drop for SpinGuard<'a, T> {
    fn drop(&mut self) {
        self.spin.lock.store(false, Release);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Next {
    Some(usize),
    End,
    Outside,
}

impl Next {
    fn some(self) -> Option<usize> {
        match self {
            Next::Some(i) => Some(i),
            Next::End => None,
            Next::Outside => panic!("invalid list element"),
        }
    }
}

impl From<Option<usize>> for Next {
    fn from(v: Option<usize>) -> Self {
        match v {
            Some(i) => Self::Some(i),
            None => Self::End,
        }
    }
}

/// Node of an atomic stack
pub trait ANode: Atomic + Default {
    fn next(self) -> Next;
    fn with_next(self, next: Next) -> Self;
    fn enqueue(self, next: Next) -> Option<Self> {
        (self.next() == Next::Outside).then_some(self.with_next(next))
    }
}

impl ANode for TreeNode {
    fn next(self) -> Next {
        match self.idx() {
            TreeNode::IDX_MAX => Next::Outside,
            TreeNode::IDX_END => Next::End,
            i => Next::Some(i),
        }
    }

    fn with_next(self, next: Next) -> Self {
        self.with_idx(match next {
            Next::Some(i) => i,
            Next::End => TreeNode::IDX_END,
            Next::Outside => TreeNode::IDX_MAX,
        })
    }
}

/// Simple atomic stack with atomic entries.
/// It is constructed over an already existing fixed size buffer.
#[repr(align(64))] // Just to be sure
pub struct AtomicStack<T: ANode> {
    start: Atom<T>,
}

impl<T: ANode> Default for AtomicStack<T> {
    fn default() -> Self {
        Self {
            start: Atom::new(T::default().with_next(Next::End)),
        }
    }
}

impl<T: ANode> AtomicStack<T> {
    pub fn set(&self, v: T) {
        self.start.store(v)
    }

    /// Pushes the element at `idx` to the front of the stack.
    pub fn push<B>(&self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atom<T>>,
    {
        let mut prev = self.start.load();
        let elem = &buf[idx];
        loop {
            if elem
                .fetch_update(|v| Some(v.with_next(prev.next())))
                .is_err()
            {
                panic!();
            }
            // CAS weak is important for fetch-update!
            match self
                .start
                .compare_exchange(prev, prev.with_next(Next::Some(idx)))
            {
                Ok(_) => return,
                Err(s) => prev = s,
            }
        }
    }

    /// Poping the first element and updating it in place.
    pub fn pop_update<B, F>(&self, buf: &B, mut f: F) -> Option<(usize, Result<T, T>)>
    where
        B: Index<usize, Output = Atom<T>>,
        F: FnMut(T) -> Option<T>,
    {
        let mut prev = self.start.load();
        loop {
            let idx = prev.next().some()?;
            let next = buf[idx].load().next();
            // CAS weak is important for fetch-update!
            match self.start.compare_exchange(prev, prev.with_next(next)) {
                Ok(old) => {
                    let i = old.next().some()?;
                    return Some((
                        i,
                        buf[i].fetch_update(|v| f(v).map(|v| v.with_next(Next::Outside))),
                    ));
                }
                Err(s) => prev = s,
            }
        }
    }

    /// Poping the first element returning its index.
    pub fn pop<B>(&self, buf: &B) -> Option<usize>
    where
        B: Index<usize, Output = Atom<T>>,
    {
        self.pop_update(buf, |v| Some(v)).map(|v| v.0)
    }
}

/// Debug printer for the [AStack].
#[allow(dead_code)]
pub struct AtomicStackDbg<'a, T, B>(pub &'a AtomicStack<T>, pub &'a B)
where
    T: ANode,
    B: Index<usize, Output = Atom<T>>;

impl<'a, T, B> fmt::Debug for AtomicStackDbg<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atom<T>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut dbg = f.debug_list();

        if let Next::Some(mut i) = self.0.start.load().next() {
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
pub struct BufferList<T: ANode> {
    start: Option<usize>,
    end: Option<usize>,
    _phantom: PhantomData<T>,
}

impl<T: ANode> fmt::Debug for BufferList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufList")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish()
    }
}

impl<T: ANode + Debug> BufferList<T> {
    pub fn clear(&mut self) {
        self.start = None;
        self.end = None;
    }

    pub fn push<B>(&mut self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atom<T>>,
    {
        if buf[idx]
            .fetch_update(|v| v.enqueue(self.start.into()))
            .is_err()
        {
            return;
        }

        self.start = Some(idx);
        if self.end.is_none() {
            self.end = Some(idx);
        }
    }

    pub fn push_back<B>(&mut self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atom<T>>,
    {
        if buf[idx].fetch_update(|v| v.enqueue(Next::End)).is_err() {
            return;
        }

        if let Some(end) = self.end {
            if buf[end]
                .fetch_update(|v| Some(v.with_next(Next::Some(idx))))
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
        B: Index<usize, Output = Atom<T>>,
    {
        let start = self.start?;
        if let Ok(pte) = buf[start].fetch_update(|v| Some(v.with_next(Next::Outside))) {
            self.start = pte.next().some();
            if self.start.is_none() {
                self.end = None;
            }
            Some(start)
        } else {
            unreachable!()
        }
    }

    pub fn iter<'a, B>(&'a self, buf: &'a B) -> BufferListIter<'a, T, B>
    where
        B: Index<usize, Output = Atom<T>>,
    {
        BufferListIter {
            next: self.start,
            buf,
        }
    }
}

pub struct BufferListIter<'a, T: ANode, B: Index<usize, Output = Atom<T>>> {
    next: Option<usize>,
    buf: &'a B,
}

impl<'a, T, B> Iterator for BufferListIter<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atom<T>>,
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

impl<'a, T, B> fmt::Debug for BufferListIter<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atom<T>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut dbg = f.debug_list();
        let iter = BufferListIter {
            next: self.next,
            buf: self.buf,
        };
        for (i, _) in iter.take(10) {
            dbg.entry(&i);
        }
        dbg.finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::hint::black_box;
    use core::sync::atomic::AtomicU64;
    use std::sync::Arc;

    use std::sync::Barrier;

    use crate::atomic::{Atom, BufferList};
    use crate::thread;

    use super::{ANode, AtomicStack, AtomicStackDbg, Next};

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
        const DATA_V: Atom<u64> = Atom::raw(AtomicU64::new(0));
        const N: usize = 64;
        let data: [Atom<u64>; N] = [DATA_V; N];

        let stack = AtomicStack::default();
        stack.push(&data, 0);
        stack.push(&data, 1);

        println!("{:?}", AtomicStackDbg(&stack, &data));

        assert_eq!(stack.pop(&data), Some(1));
        assert_eq!(stack.pop(&data), Some(0));
        assert_eq!(stack.pop(&data), None);

        // Stress test

        const THREADS: usize = 6;
        const I: usize = N / THREADS;
        let barrier = Arc::new(Barrier::new(THREADS));
        let stack = Arc::new(stack);
        let copy = stack.clone();
        thread::parallel(0..THREADS, |t| {
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
        const DATA_V: Atom<u64> = Atom::raw(AtomicU64::new(u64::MAX));
        const N: usize = 64;
        let data: [Atom<u64>; N] = [DATA_V; N];

        let mut list = BufferList::default();
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
