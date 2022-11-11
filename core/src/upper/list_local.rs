use core::cell::UnsafeCell;
use core::fmt;
use core::ops::Range;
use core::ptr::{null, null_mut};

use alloc::boxed::Box;
use alloc::slice;
use alloc::vec::Vec;
use log::{error, info};

use super::{Alloc, MIN_PAGES};
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
    memory: Range<*const Page>,
    struct_pages: Box<[StructPage]>,
    /// CPU local metadata
    local: Box<[UnsafeCell<Local>]>,
    pages: usize,
}

unsafe impl Send for ListLocal {}
unsafe impl Sync for ListLocal {}

impl fmt::Debug for ListLocal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        for (t, l) in self.local.iter().enumerate() {
            let local = unsafe { &mut *l.get() };
            writeln!(f, "    L {t:>2} C={}", local.counter)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Default for ListLocal {
    fn default() -> Self {
        Self {
            memory: null()..null(),
            struct_pages: Box::new([]),
            local: Box::new([]),
            pages: 0,
        }
    }
}

impl Alloc for ListLocal {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], _persistent: bool) -> Result<()> {
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
        struct_pages.resize_with(memory.len(), || StructPage { next: null_mut() });
        self.struct_pages = struct_pages.into();

        self.pages = (memory.len() / cores) * cores;
        self.memory = memory[..self.pages].as_ptr_range();
        self.local = local.into();
        Ok(())
    }

    #[cold]
    fn free_all(&self) -> Result<()> {
        let cores = self.local.len();
        let struct_pages = unsafe {
            slice::from_raw_parts_mut(self.struct_pages.as_ptr() as *mut StructPage, self.pages)
        };

        let begin = self.struct_pages.as_ptr();
        // build core local free lists
        let p_core = self.pages / cores;
        for core in 0..cores {
            let l = unsafe { &mut *self.local[core].get() };
            // build linked list
            for i in core * p_core + 1..(core + 1) * p_core {
                struct_pages[i - 1].next = unsafe { begin.add(i).cast_mut() };
            }
            struct_pages[(core + 1) * p_core - 1].next = null_mut();
            l.next.set(unsafe { begin.add(core * p_core).cast_mut() });
            l.counter = 0;
        }
        Ok(())
    }

    #[cold]
    fn reserve_all(&self) -> Result<()> {
        let cores = self.local.len();

        for core in 0..cores {
            let l = unsafe { &mut *self.local[core].get() };
            l.next.set(null_mut());
            l.counter = self.pages() / cores;
        }
        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }

        let local = unsafe { &mut *self.local[core].get() };
        if let Some(node) = local.next.pop() {
            local.counter += 1;
            let addr = self.from_struct_page(node as *mut _) as u64;
            debug_assert!(addr % Page::SIZE as u64 == 0 && self.memory.contains(&(addr as _)));
            Ok(addr)
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) || order != 0 {
            error!("invalid addr");
            return Err(Error::Address);
        }

        let local = unsafe { &mut *self.local[core].get() };
        let struct_page = self.to_struct_page(addr as *mut Page);
        local.next.push(unsafe { &mut *struct_page });
        local.counter -= 1;
        Ok(())
    }

    fn is_free(&self, _addr: u64, _order: usize) -> bool {
        false
    }

    fn pages(&self) -> usize {
        self.pages
    }

    #[cold]
    fn dbg_for_each_huge_page(&self, _f: fn(usize)) {}

    #[cold]
    fn dbg_free_pages(&self) -> usize {
        let mut pages = self.pages;
        for local in self.local.iter() {
            let local = unsafe { &*local.get() };
            pages -= local.counter;
        }
        pages
    }
}

impl ListLocal {
    #[inline]
    fn to_struct_page(&self, v: *mut Page) -> *mut StructPage {
        debug_assert!(self.memory.contains(&(v as *const _)));
        let idx = unsafe { v.offset_from(self.memory.start) };
        unsafe { self.struct_pages.as_ptr().add(idx as _) as _ }
    }
    #[inline]
    fn from_struct_page(&self, v: *mut StructPage) -> *mut Page {
        debug_assert!(self.struct_pages.as_ptr_range().contains(&(v as *const _)));
        let idx = unsafe { v.offset_from(self.struct_pages.as_ptr()) };
        unsafe { self.memory.start.add(idx as _) as _ }
    }
}

#[repr(align(64))]
struct Local {
    next: Node<StructPage>,
    counter: usize,
}

impl Default for Local {
    fn default() -> Self {
        Self {
            next: Node::new(null_mut()),
            counter: 0,
        }
    }
}

/// Representing Linux `struct page`
#[repr(align(64))]
struct StructPage {
    /// Next pointers are a bit ugly, but they are used heavily in linux
    next: *mut StructPage,
}
const _: () = assert!(core::mem::align_of::<StructPage>() == 64);

impl HasNext for StructPage {
    fn next(&mut self) -> &mut *mut Self {
        &mut self.next
    }
}

trait HasNext {
    fn next(&mut self) -> &mut *mut Self;
}

struct Node<T: HasNext>(*mut T);

impl<T: HasNext> Node<T> {
    fn new(v: *mut T) -> Self {
        Self(v)
    }
    fn set(&mut self, v: *mut T) {
        self.0 = v;
    }
    fn push(&mut self, v: &mut T) {
        *v.next() = self.0;
        self.0 = v;
    }
    fn pop(&mut self) -> Option<&mut T> {
        if let Some(curr) = unsafe { self.0.as_mut() } {
            self.0 = *curr.next();
            Some(curr)
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
    use crate::upper::Alloc;
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

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, false).unwrap();
            a.free_all().unwrap();
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
