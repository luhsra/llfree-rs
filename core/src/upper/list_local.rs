use core::cell::UnsafeCell;
use core::fmt;

use alloc::boxed::Box;
use alloc::vec::Vec;
use crossbeam_utils::atomic::AtomicCell;
use log::{error, info};

use super::{Alloc, Init, MIN_PAGES};
use crate::util::Page;
use crate::{Error, Result};

/// Simple volatile 4K page allocator that uses CPU-local linked lists.
/// During initialization allocators memory is split into pages
/// and evenly distributed to the cores.
/// The linked lists are build directly within the pages,
/// storing the next pointers at the beginning of the free pages.
///
/// No extra load balancing is made, if a core runs out of memory,
/// the allocation fails.
#[repr(align(64))]
pub struct ListLocal {
    frames: Box<[PageFrame]>,
    /// CPU local metadata
    local: Box<[UnsafeCell<Local>]>,
    offset: u64,
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
            offset: 0,
            len: 0,
        }
    }
}

impl Alloc for ListLocal {
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

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || UnsafeCell::new(Local::default()));

        let mut struct_pages = Vec::with_capacity(memory.len());
        struct_pages.resize_with(memory.len(), PageFrame::new);
        self.frames = struct_pages.into();

        self.len = (memory.len() / cores) * cores;
        self.offset = memory.as_ptr() as u64 / Page::SIZE as u64;
        self.local = local.into();

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

        let local = self.local(core);
        if let Some(index) = local.next.pop(|i| self.frames[i].get_next()) {
            local.counter += 1;
            Ok(self.from_pfn(index))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Address);
        }

        let pfn = self.to_pfn(addr)?;
        let local = self.local(core);
        local.next.push(pfn, |i, n| self.frames[i].set_next(n));
        local.counter -= 1;
        Ok(())
    }

    fn is_free(&self, _addr: u64, _order: usize) -> bool {
        false
    }

    fn pages(&self) -> usize {
        self.len
    }

    fn dbg_free_pages(&self) -> usize {
        let mut pages = self.len;
        for local in self.local.iter() {
            let local = unsafe { &*local.get() };
            pages -= local.counter;
        }
        pages
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
                self.frames[pfn - 1].set_next(Some(pfn));
            }
            self.frames[(core + 1) * p_core - 1].set_next(None);
            l.next.set(Some(core * p_core));
            l.counter = 0;
        }
        Ok(())
    }

    #[cold]
    fn reserve_all(&self) -> Result<()> {
        let cores = self.local.len();

        for core in 0..cores {
            let l = self.local(core);
            l.next.set(None);
            l.counter = self.pages() / cores;
        }
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

#[repr(align(64))]
struct Local {
    next: Node,
    counter: usize,
}

impl Default for Local {
    fn default() -> Self {
        Self {
            next: Node::new(None),
            counter: 0,
        }
    }
}

/// Representing Linux `struct page`
#[repr(align(64))]
struct PageFrame {
    /// Next pointers are a bit ugly, but they are used heavily in linux
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
    use alloc::vec::Vec;
    use log::{info, warn};

    use crate::mmap::test_mapping;
    use crate::table::PT_LEN;
    use crate::upper::{Alloc, AllocExt, Init};
    use crate::util::{logging, Page};
    use crate::Error;

    use super::ListLocal;

    type Allocator = ListLocal;

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
