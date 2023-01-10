use core::fmt;
use core::ops::Index;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};

use super::{Alloc, Init, MIN_PAGES};
use crate::atomic::{Atom, Spin};
use crate::entry::Next;
use crate::util::Page;
use crate::{Error, Result};

/// Simple volatile 4K page allocator that uses a single shared linked lists
/// protected by a ticked lock.
/// The linked list pointers are stored similar to Linux's in the struct pages.
///
/// As expected the contention on the ticket lock is very high.
#[repr(align(64))]
pub struct ListLocked {
    offset: u64,
    len: usize,
    frames: Box<[PageFrame]>,
    /// CPU local metadata
    local: Box<[LocalCounter]>,
    /// Per page metadata
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
            offset: 0,
            next: Spin::new(Node::new(Next::End)),
            local: Box::new([]),
            frames: Box::new([]),
        }
    }
}

impl Alloc for ListLocked {
    #[cold]
    fn init(
        &mut self,
        cores: usize,
        memory: &mut [Page],
        init: Init,
        free_all: bool,
    ) -> Result<()> {
        debug_assert!(init == Init::Volatile);
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < cores * MIN_PAGES {
            error!("Not enough memory {} < {}", memory.len(), cores * MIN_PAGES);
            return Err(Error::Memory);
        }

        self.offset = memory.as_ptr() as u64 / Page::SIZE as u64;
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

    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }

        if let Some(pfn) = self.next.lock().pop(self) {
            self.local[core].counter.fetch_add(1, Ordering::Relaxed);
            Ok(self.from_pfn(pfn))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }
        let pfn = self.to_pfn(addr)?;

        self.next.lock().push(self, pfn);
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn is_free(&self, _addr: u64, _order: usize) -> bool {
        false
    }

    fn pages(&self) -> usize {
        self.len
    }

    fn dbg_free_pages(&self) -> usize {
        self.pages()
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
        for i in 1..self.pages() {
            self.frames[i - 1].next.store(Next::Some(i));
        }
        self.frames[self.pages() - 1].next.store(Next::End);

        next.set(Next::Some(0));
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local
                .counter
                .store(self.pages() / self.local.len(), Ordering::Relaxed);
        }
        self.next.lock().set(Next::End);
        Ok(())
    }

    #[inline]
    fn to_pfn(&self, addr: u64) -> Result<usize> {
        if addr % Page::SIZE as u64 != 0 {
            return Err(Error::Address);
        }
        let off = addr / (Page::SIZE as u64);
        if let Some(pfn) = off.checked_sub(self.offset) {
            if (pfn as usize) < self.len {
                Ok(pfn as _)
            } else {
                Err(Error::Address)
            }
        } else {
            Err(Error::Address)
        }
    }
    #[inline]
    fn from_pfn(&self, pfn: usize) -> u64 {
        debug_assert!(pfn < self.len);
        (self.offset + pfn as u64) * Page::SIZE as u64
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

    use crate::mmap::test_mapping;
    use crate::table::PT_LEN;
    use crate::upper::{Alloc, AllocExt, Init};
    use crate::util::{logging, Page};
    use crate::Error;

    use super::ListLocked;

    type Allocator = ListLocked;

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mut mapping = test_mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Allocator::new(1, &mut mapping, Init::Volatile, true).unwrap();

        assert_eq!(alloc.dbg_free_pages(), alloc.pages());

        warn!("start alloc...");
        let small = alloc.get(0, 0).unwrap();

        assert_eq!(alloc.dbg_allocated_pages(), 1, "{alloc:?}");
        warn!("stress test...");

        // Stress test
        let mut pages = Vec::new();
        loop {
            match alloc.get(0, 0) {
                Ok(page) => pages.push(page),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + pages.len());
        warn!("check...");

        assert_eq!(alloc.dbg_allocated_pages(), 1 + pages.len());
        assert_eq!(alloc.dbg_allocated_pages(), alloc.pages());
        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = PT_LEN * PT_LEN - 10;
        for page in &pages[..FREE_NUM] {
            alloc.put(0, *page, 0).unwrap();
        }

        assert_eq!(
            alloc.dbg_allocated_pages(),
            1 + pages.len() - FREE_NUM,
            "{alloc:?}"
        );

        // Realloc
        for page in &mut pages[..FREE_NUM] {
            *page = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        alloc.put(0, small, 0).unwrap();
        // Free all
        for page in &pages {
            alloc.put(0, *page, 0).unwrap();
        }

        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }
}
