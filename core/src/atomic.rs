use core::fmt::{self, Debug};
use core::marker::PhantomData;
use core::ops::Index;

use crossbeam_utils::atomic::AtomicCell;
use log::error;

use crate::entry::Entry3;

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
pub trait ANode: Eq + Copy + Default {
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
pub struct AtomicStack<T: ANode> {
    start: AtomicCell<T>,
}

impl<T: ANode> Default for AtomicStack<T> {
    fn default() -> Self {
        Self {
            start: AtomicCell::new(T::default().with_next(Next::End)),
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
        B: Index<usize, Output = AtomicCell<T>>,
    {
        let mut prev = self.start.load();
        let elem = &buf[idx];
        loop {
            if elem.fetch_update(|v| Some(v.with_next(prev.next()))).is_err() {
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
        B: Index<usize, Output = AtomicCell<T>>,
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
        B: Index<usize, Output = AtomicCell<T>>,
    {
        self.pop_update(buf, |v| Some(v)).map(|v| v.0)
    }
}

/// Debug printer for the [AStack].
#[allow(dead_code)]
pub struct AtomicStackDbg<'a, T, B>(pub &'a AtomicStack<T>, pub &'a B)
where
    T: ANode,
    B: Index<usize, Output = AtomicCell<T>>;

impl<'a, T, B> fmt::Debug for AtomicStackDbg<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = AtomicCell<T>>,
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
        B: Index<usize, Output = AtomicCell<T>>,
    {
        if buf[idx].fetch_update(|v| v.enqueue(self.start.into())).is_err() {
            return;
        }

        self.start = Some(idx);
        if self.end.is_none() {
            self.end = Some(idx);
        }
    }

    pub fn push_back<B>(&mut self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = AtomicCell<T>>,
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
        B: Index<usize, Output = AtomicCell<T>>,
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
        B: Index<usize, Output = AtomicCell<T>>,
    {
        BufferListIter {
            next: self.start,
            buf,
        }
    }
}

pub struct BufferListIter<'a, T: ANode, B: Index<usize, Output = AtomicCell<T>>> {
    next: Option<usize>,
    buf: &'a B,
}

impl<'a, T, B> Iterator for BufferListIter<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = AtomicCell<T>>,
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
    B: Index<usize, Output = AtomicCell<T>>,
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
    use std::sync::Arc;

    use crossbeam_utils::atomic::AtomicCell;
    use spin::Barrier;

    use crate::{atomic::BufferList, thread};

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
        const DATA_V: AtomicCell<u64> = AtomicCell::new(0);
        const N: usize = 64;
        let data: [AtomicCell<u64>; N] = [DATA_V; N];

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
        const DATA_V: AtomicCell<u64> = AtomicCell::new(u64::MAX);
        const N: usize = 64;
        let data: [AtomicCell<u64>; N] = [DATA_V; N];

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
