use core::fmt;
use core::ops::{Index, Range};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use bitfield_struct::bitfield;
use log::{error, info};

use super::{Alloc, Init, MIN_PAGES};
use crate::atomic::{Atom, Atomic};
use crate::entry::Next;
use crate::frame::{PFNRange, PFN};
use crate::{Error, Result};

/// Simple volatile 4K frame allocator that uses a single shared linked lists
/// protected by a ticked lock.
/// The linked list pointers are stored similar to Linux's in the struct pages.
///
/// As expected the contention on the ticket lock is very high.
#[repr(align(64))]
pub struct ListCAS {
    begin: PFN,
    len: usize,
    frames: Box<[PageFrame]>,
    /// CPU local metadata
    local: Box<[LocalCounter]>,
    /// Per frame metadata
    list: AtomicStack,
}

#[repr(align(64))]
struct LocalCounter {
    counter: AtomicUsize,
}
const _: () = assert!(core::mem::align_of::<LocalCounter>() == 64);

unsafe impl Send for ListCAS {}
unsafe impl Sync for ListCAS {}

impl fmt::Debug for ListCAS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        for (t, l) in self.local.iter().enumerate() {
            writeln!(f, "    L {t:>2} C={}", l.counter.load(Ordering::Relaxed))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Default for ListCAS {
    fn default() -> Self {
        Self {
            len: 0,
            begin: PFN(0),
            list: AtomicStack::default(),
            local: Box::new([]),
            frames: Box::new([]),
        }
    }
}

impl Alloc for ListCAS {
    #[cold]
    fn init(&mut self, cores: usize, memory: Range<PFN>, init: Init, free_all: bool) -> Result<()> {
        debug_assert!(init == Init::Volatile);
        info!("initializing c={cores} {memory:?} {}", memory.len());
        if memory.len() < cores * MIN_PAGES {
            error!("Not enough memory {} < {}", memory.len(), cores * MIN_PAGES);
            return Err(Error::Memory);
        }

        self.begin = memory.start;
        self.len = memory.len();

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || LocalCounter {
            counter: AtomicUsize::new(0),
        });
        self.local = local.into();

        let mut struct_pages = Vec::with_capacity(memory.len());
        struct_pages.resize_with(memory.len(), PageFrame::new);
        self.frames = struct_pages.into();

        if free_all {
            self.free_all()?;
        } else {
            self.reserve_all()?;
        }

        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<PFN> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }

        if let Some(pfn) = self.list.pop(self) {
            self.local[core].counter.fetch_add(1, Ordering::Relaxed);
            Ok(self.from_frame(pfn))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: PFN, order: usize) -> Result<()> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }
        let pfn = self.to_frame(addr)?;

        self.list.push(self, pfn);
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn is_free(&self, _addr: PFN, _order: usize) -> bool {
        false
    }

    fn frames(&self) -> usize {
        self.len
    }

    fn free_frames(&self) -> usize {
        self.frames()
            - self
                .local
                .iter()
                .map(|c| c.counter.load(Ordering::SeqCst))
                .sum::<usize>()
    }
}

impl ListCAS {
    #[cold]
    fn free_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local.counter.store(0, Ordering::Relaxed);
        }

        // build free lists
        for i in 1..self.frames() {
            self.frames[i - 1].next.store(Next::Some(i));
        }
        self.frames[self.frames() - 1].next.store(Next::End);

        self.list.start.store(ANext::with(Next::Some(0), 0));
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local
                .counter
                .store(self.frames() / self.local.len(), Ordering::Relaxed);
        }
        self.list.start.store(ANext::with(Next::End, 0));
        Ok(())
    }

    #[inline]
    fn to_frame(&self, addr: PFN) -> Result<usize> {
        if let Some(pfn) = addr.0.checked_sub(self.begin.0) {
            if pfn < self.len {
                Ok(pfn as _)
            } else {
                Err(Error::Address)
            }
        } else {
            Err(Error::Address)
        }
    }
    #[inline]
    fn from_frame(&self, frame: usize) -> PFN {
        debug_assert!(frame < self.len);
        self.begin.off(frame)
    }
}

impl Index<usize> for ListCAS {
    type Output = Atom<Next>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.frames[index].next
    }
}

/// Representing Linux `struct page`
#[repr(align(64))]
struct PageFrame {
    /// Next page frame number
    next: Atom<Next>,
}
const _: () = assert!(core::mem::align_of::<PageFrame>() == 64);

impl PageFrame {
    fn new() -> Self {
        Self {
            next: Atom::new(Next::Outside),
        }
    }
}

#[bitfield(u64)]
struct ANext {
    index: u32,
    /// Counts the number of successful writes to prevent ABA
    #[bits(32)]
    tag: usize,
}

impl ANext {
    const OUTSIDE: u32 = ((1u64 << Self::INDEX_BITS) - 1) as _;
    const END: u32 = ((1u64 << Self::INDEX_BITS) - 2) as _;
    fn with(n: Next, tag: usize) -> Self {
        Self::new()
            .with_index(match n {
                Next::Outside => Self::OUTSIDE,
                Next::End => Self::END,
                Next::Some(i) => i as _,
            })
            .with_tag(tag)
    }
    fn next(self) -> Next {
        match self.index() {
            Self::OUTSIDE => Next::Outside,
            Self::END => Next::End,
            i => Next::Some(i as _),
        }
    }
}
impl Atomic for ANext {
    type I = AtomicU64;
}

/// Simple atomic stack with atomic entries.
/// It is constructed over an already existing fixed size buffer.
#[repr(align(64))] // Just to be sure
pub struct AtomicStack {
    start: Atom<ANext>,
}

impl Default for AtomicStack {
    fn default() -> Self {
        Self {
            start: Atom::new(ANext::with(Next::End, 0)),
        }
    }
}

impl AtomicStack {
    /// Pushes the element at `idx` to the front of the stack.
    pub fn push<B>(&self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atom<Next>>,
    {
        let mut prev = Next::Outside;
        let mut top = self.start.load();
        let elem = &buf[idx];
        loop {
            if elem.compare_exchange(prev, top.next()).is_err() {
                error!("invalid list element");
                panic!()
            }

            match self
                .start
                .compare_exchange(top, ANext::with(Next::Some(idx), top.tag().wrapping_add(1)))
            {
                Ok(_) => return,
                Err(new_top) => {
                    prev = top.next();
                    top = new_top;
                }
            }
        }
    }

    /// Poping the first element and updating it in place.
    pub fn pop<B>(&self, buf: &B) -> Option<usize>
    where
        B: Index<usize, Output = Atom<Next>>,
    {
        let mut top = self.start.load();
        loop {
            let top_idx = top.next().some()?;
            let next = buf[top_idx].load();
            match self
                .start
                .compare_exchange(top, ANext::with(next, top.tag().wrapping_add(1)))
            {
                Ok(_) => {
                    if buf[top_idx].compare_exchange(next, Next::Outside).is_err() {
                        error!("invalid list element");
                        panic!();
                    }
                    return Some(top_idx);
                }
                Err(new_top) => top = new_top,
            }
        }
    }
}

/// Debug printer for the [AStack].
#[allow(dead_code)]
pub struct AtomicStackDbg<'a, B>(pub &'a AtomicStack, pub &'a B)
where
    B: Index<usize, Output = Atom<Next>>;

impl<'a, B> fmt::Debug for AtomicStackDbg<'a, B>
where
    B: Index<usize, Output = Atom<Next>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut dbg = f.debug_list();

        if let Next::Some(mut i) = self.0.start.load().next() {
            let mut ended = false;
            for _ in 0..1000 {
                dbg.entry(&i);
                let elem = self.1[i].load();
                if let Next::Some(next) = elem {
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

#[cfg(test)]
mod test {
    use core::hint::black_box;
    use core::sync::atomic::AtomicU64;
    use std::sync::Barrier;

    use alloc::vec::Vec;
    use log::{info, warn};

    use crate::atomic::Atom;
    use crate::frame::{pfn_range, Frame};
    use crate::mmap::test_mapping;
    use crate::table::PT_LEN;
    use crate::upper::list_cas::{AtomicStack, AtomicStackDbg, Next};
    use crate::upper::{Alloc, AllocExt, Init};
    use crate::util::{self, logging};
    use crate::{thread, Error};

    use super::ListCAS;

    type Allocator = ListCAS;

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mapping = test_mapping(0x1000_0000_0000, MEM_SIZE / Frame::SIZE);
        let area = pfn_range(&mapping);

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        assert_eq!(alloc.free_frames(), alloc.frames());

        warn!("start alloc...");
        let small = alloc.get(0, 0).unwrap();

        assert_eq!(alloc.allocated_frames(), 1, "{alloc:?}");
        warn!("stress test...");

        // Stress test
        let mut frames = Vec::new();
        loop {
            match alloc.get(0, 0) {
                Ok(frame) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + frames.len());
        warn!("check...");

        assert_eq!(alloc.allocated_frames(), 1 + frames.len());
        assert_eq!(alloc.allocated_frames(), alloc.frames());
        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for i in 0..frames.len() - 1 {
            let p1 = frames[i];
            let p2 = frames[i + 1];
            assert!(area.contains(&p1));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = PT_LEN * PT_LEN - 10;
        for frame in &frames[..FREE_NUM] {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(
            alloc.allocated_frames(),
            1 + frames.len() - FREE_NUM,
            "{alloc:?}"
        );

        // Realloc
        for frame in &mut frames[..FREE_NUM] {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        alloc.put(0, small, 0).unwrap();
        // Free all
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn atomic_stack() {
        util::logging();
        const DATA_V: Atom<Next> = Atom(AtomicU64::new(u64::MAX));
        const N: usize = 640;
        let data: [Atom<Next>; N] = [DATA_V; N];

        let stack = AtomicStack::default();
        stack.push(&data, 0);
        stack.push(&data, 1);

        warn!("{:?}", AtomicStackDbg(&stack, &data));

        assert_eq!(stack.pop(&data), Some(1));
        assert_eq!(stack.pop(&data), Some(0));
        assert_eq!(stack.pop(&data), None);

        // Stress test
        warn!("parallel:");

        const THREADS: usize = 6;
        const I: usize = N / THREADS;
        let barrier = Barrier::new(THREADS);
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
        assert_eq!(stack.pop(&data), None);
    }

    #[test]
    fn atomic_stack_repeat() {
        util::logging();
        const THREADS: usize = 6;
        const DATA_V: Atom<Next> = Atom(AtomicU64::new(u64::MAX));
        let data: [Atom<Next>; THREADS] = [DATA_V; THREADS];

        let stack = AtomicStack::default();
        // Stress test
        warn!("parallel:");

        let barrier = Barrier::new(THREADS);
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut idx = t;

            barrier.wait();

            for _ in 0..1000 {
                stack.push(&data, idx);
                idx = stack.pop(&data).unwrap();
            }
        });
        assert_eq!(stack.pop(&data), None);
    }
}
