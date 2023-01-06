use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use crossbeam_utils::atomic::AtomicCell;
use log::{error, info};
use spin::Mutex;

use super::{Alloc, Init, MIN_PAGES};
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
    next: Mutex<Node>,
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
            next: Mutex::new(Node::new(None)),
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
        struct_pages.resize_with(memory.len(), || PageFrame::new());
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

        if let Some(pfn) = self.next.lock().pop(|i| self.frames[i].get_next()) {
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

        self.next
            .lock()
            .push(pfn, |i, n| self.frames[i].set_next(n));
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn is_free(&self, _addr: u64, _order: usize) -> bool {
        false
    }

    fn pages(&self) -> usize {
        self.len
    }

    #[cold]
    fn dbg_for_each_huge_page(&self, _f: fn(usize)) {}

    #[cold]
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
            self.frames[i - 1].set_next(Some(i));
        }
        self.frames[self.pages() - 1].set_next(None);

        next.set(Some(0));
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local
                .counter
                .store(self.pages() / self.local.len(), Ordering::Relaxed);
        }
        self.next.lock().set(None);
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

/// Representing Linux `struct page`
#[repr(align(64))]
struct PageFrame {
    /// Next page frame number
    next: AtomicCell<usize>,
}
const _: () = assert!(core::mem::align_of::<PageFrame>() == 64);
const _: () = assert!(AtomicCell::<usize>::is_lock_free());

impl PageFrame {
    fn new() -> Self {
        Self {
            next: AtomicCell::new(usize::MAX),
        }
    }
    fn get_next(&self) -> Option<usize> {
        let val = self.next.load();
        (val < usize::MAX).then_some(val)
    }
    fn set_next(&self, next: Option<usize>) {
        self.next.store(next.unwrap_or(usize::MAX));
    }
}

struct Node {
    start: Option<usize>,
}

impl Node {
    fn new(start: Option<usize>) -> Self {
        Self { start }
    }
    fn set(&mut self, start: Option<usize>) {
        self.start = start;
    }
    fn push<F>(&mut self, next: usize, set_next: F)
    where
        F: FnOnce(usize, Option<usize>),
    {
        set_next(next, self.start);
        self.start = Some(next);
    }
    fn pop<F>(&mut self, get_next: F) -> Option<usize>
    where
        F: FnOnce(usize) -> Option<usize>,
    {
        if let Some(idx) = self.start {
            self.start = get_next(idx);
            Some(idx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use log::{info, warn};

    use crate::mmap::test_mapping;
    use crate::table::PT_LEN;
    use crate::upper::{Alloc, Init};
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

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, Init::Volatile, true).unwrap();
            a
        });

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
