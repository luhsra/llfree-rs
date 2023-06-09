use core::cell::UnsafeCell;
use core::fmt;
use core::ops::{Index, Range};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};

use super::{Alloc, Init, MIN_PAGES};
use crate::atomic::Atom;
use crate::entry::Next;
use crate::frame::{PFNRange, PFN};
use crate::{Error, Result};

/// Simple volatile 4K frame allocator that uses CPU-local linked lists.
/// During initialization allocators memory is split into frames
/// and evenly distributed to the cores.
/// The linked lists are build directly within the frames,
/// storing the next pointers at the beginning of the free frames.
///
/// No extra load balancing is made, if a core runs out of memory,
/// the allocation fails.
#[repr(align(64))]
pub struct ListLocal {
    frames: Box<[PageFrame]>,
    /// CPU local metadata
    local: Box<[UnsafeCell<Local>]>,
    begin: PFN,
    len: usize,
}

unsafe impl Send for ListLocal {}
unsafe impl Sync for ListLocal {}

impl fmt::Debug for ListLocal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        for (t, l) in self.local.iter().enumerate() {
            let local = unsafe { &*l.get() };
            writeln!(f, "    L {t:>2} C={}", local.counter)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Default for ListLocal {
    fn default() -> Self {
        Self {
            frames: Box::new([]),
            local: Box::new([]),
            begin: PFN(0),
            len: 0,
        }
    }
}

impl Alloc for ListLocal {
    #[cold]
    fn init(&mut self, cores: usize, memory: Range<PFN>, init: Init, free_all: bool) -> Result<()> {
        debug_assert!(init == Init::Volatile);
        info!("initializing c={cores} {memory:?} {}", memory.len());
        if memory.len() < cores * MIN_PAGES {
            error!("Not enough memory {} < {}", memory.len(), cores * MIN_PAGES);
            return Err(Error::Memory);
        }

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || UnsafeCell::new(Local::default()));

        let mut struct_pages = Vec::with_capacity(memory.len());
        struct_pages.resize_with(memory.len(), PageFrame::new);
        self.frames = struct_pages.into();

        self.len = (memory.len() / cores) * cores;
        self.begin = memory.start;
        self.local = local.into();

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

        let local = self.local(core);
        if let Some(index) = local.next.pop(self) {
            local.counter += 1;
            Ok(self.from_frame(index))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: PFN, order: usize) -> Result<()> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Address);
        }

        let pfn = self.to_frame(addr)?;
        let local = self.local(core);
        local.next.push(self, pfn);
        local.counter -= 1;
        Ok(())
    }

    fn is_free(&self, _addr: PFN, _order: usize) -> bool {
        false
    }

    fn frames(&self) -> usize {
        self.len
    }

    fn free_frames(&self) -> usize {
        let mut frames = self.len;
        for local in self.local.iter() {
            let local = unsafe { &*local.get() };
            frames -= local.counter;
        }
        frames
    }
}

impl ListLocal {
    #[allow(clippy::mut_from_ref)]
    fn local(&self, core: usize) -> &mut Local {
        unsafe { &mut *self.local[core].get() }
    }

    #[cold]
    fn free_all(&self) -> Result<()> {
        let cores = self.local.len();
        // build core local free lists
        let p_core = self.len / cores;
        for core in 0..cores {
            let l = self.local(core);
            // build linked list
            for pfn in core * p_core + 1..(core + 1) * p_core {
                self.frames[pfn - 1].next.store(Next::Some(pfn));
            }
            self.frames[(core + 1) * p_core - 1].next.store(Next::End);
            l.next.set(Next::Some(core * p_core));
            l.counter = 0;
        }
        Ok(())
    }

    #[cold]
    fn reserve_all(&self) -> Result<()> {
        let cores = self.local.len();

        for core in 0..cores {
            let l = self.local(core);
            l.next.set(Next::Outside);
            l.counter = self.frames() / cores;
        }
        Ok(())
    }

    #[inline]
    fn to_frame(&self, addr: PFN) -> Result<usize> {
        if let Some(pfn) = addr.0.checked_sub(self.begin.0) && pfn < self.len {
            Ok(pfn as _)
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

impl Index<usize> for ListLocal {
    type Output = Atom<Next>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.frames[index].next
    }
}

#[repr(align(64))]
struct Local {
    next: Node,
    counter: usize,
}

impl Default for Local {
    fn default() -> Self {
        Self {
            next: Node::new(Next::End),
            counter: 0,
        }
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

    use crate::bitfield::PT_LEN;
    use crate::frame::{pfn_range, Frame};
    use crate::mmap::test_mapping;
    use crate::upper::{Alloc, AllocExt, Init};
    use crate::util::logging;
    use crate::Error;

    use super::ListLocal;

    type Allocator = ListLocal;

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
