use core::fmt;
use core::ops::{Index, Range};
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};

use super::{Alloc, Init, MIN_PAGES};
use crate::atomic::{Atom, Spin};
use crate::entry::Next;
use crate::frame::{PFNRange, PFN};
use crate::{Error, Result};

/// Simple volatile 4K frame allocator that uses a single shared linked lists
/// protected by a ticked lock.
/// The linked list pointers are stored similar to Linux's in the struct pages.
///
/// As expected the contention on the ticket lock is very high.
#[repr(align(64))]
pub struct ListLocked {
    offset: PFN,
    len: usize,
    frames: Box<[PageFrame]>,
    /// CPU local metadata
    local: Box<[LocalCounter]>,
    /// Per frame metadata
    next: Spin<Node>,
}

#[repr(align(64))]
struct LocalCounter {
    counter: AtomicUsize,
}
const _: () = assert!(core::mem::align_of::<LocalCounter>() == 64);

unsafe impl Send for ListLocked {}
unsafe impl Sync for ListLocked {}

impl fmt::Debug for ListLocked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        for (t, l) in self.local.iter().enumerate() {
            writeln!(f, "    L {t:>2} C={}", l.counter.load(Ordering::Relaxed))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Default for ListLocked {
    fn default() -> Self {
        Self {
            len: 0,
            offset: PFN(0),
            next: Spin::new(Node::new(Next::End)),
            local: Box::new([]),
            frames: Box::new([]),
        }
    }
}

impl Alloc for ListLocked {
    #[cold]
    fn init(&mut self, cores: usize, memory: Range<PFN>, init: Init, free_all: bool) -> Result<()> {
        debug_assert!(init == Init::Volatile);
        info!("initializing c={cores} {:?} {}", memory, memory.len());
        if memory.len() < cores * MIN_PAGES {
            error!("Not enough memory {} < {}", memory.len(), cores * MIN_PAGES);
            return Err(Error::Memory);
        }

        self.offset = memory.start;
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

        if let Some(frame) = self.next.lock().pop(self) {
            self.local[core].counter.fetch_add(1, Ordering::Relaxed);
            Ok(self.from_frame(frame))
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

        self.next.lock().push(self, pfn);
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

impl ListLocked {
    #[cold]
    fn free_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local.counter.store(0, Ordering::Relaxed);
        }

        let mut next = self.next.lock(); // lock here to prevent races

        // build free lists
        for i in 1..self.frames() {
            self.frames[i - 1].next.store(Next::Some(i));
        }
        self.frames[self.frames() - 1].next.store(Next::End);

        next.set(Next::Some(0));
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local
                .counter
                .store(self.frames() / self.local.len(), Ordering::Relaxed);
        }
        self.next.lock().set(Next::End);
        Ok(())
    }

    #[inline]
    fn to_frame(&self, addr: PFN) -> Result<usize> {
        if let Some(pfn) = addr.0.checked_sub(self.offset.0) && pfn < self.len {
            Ok(pfn as _)
        } else {
            Err(Error::Address)
        }
    }
    #[inline]
    fn from_frame(&self, frame: usize) -> PFN {
        debug_assert!(frame < self.len);
        self.offset.off(frame)
    }
}

impl Index<usize> for ListLocked {
    type Output = Atom<Next>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.frames[index].next
    }
}

/// Representing Linux `struct page`
#[repr(align(64))]
struct PageFrame {
    /// Next pointers are a bit ugly, but they are used heavily in linux
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

struct Node {
    start: Next,
}

impl Node {
    fn new(start: Next) -> Self {
        Self { start }
    }
    fn set(&mut self, start: Next) {
        self.start = start;
    }
    fn push<B>(&mut self, buf: &B, next: usize)
    where
        B: Index<usize, Output = Atom<Next>>,
    {
        buf[next].store(self.start);
        self.start = Next::Some(next);
    }
    fn pop<B>(&mut self, buf: &B) -> Option<usize>
    where
        B: Index<usize, Output = Atom<Next>>,
    {
        if let Some(idx) = self.start.some() {
            self.start = buf[idx].swap(Next::Outside);
            Some(idx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;
    use log::{info, warn};

    use crate::frame::{pfn_range, Frame};
    use crate::mmap::test_mapping;
    use crate::table::PT_LEN;
    use crate::upper::{Alloc, AllocExt, Init};
    use crate::util::logging;
    use crate::Error;

    use super::ListLocked;

    type Allocator = ListLocked;

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
}
