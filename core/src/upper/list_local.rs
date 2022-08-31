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

        self.pages = (memory.len() / cores) * cores;
        self.memory = memory[..self.pages].as_ptr_range();
        self.local = local.into();
        Ok(())
    }

    #[cold]
    fn free_all(&self) -> Result<()> {
        let begin = self.memory.start as usize;
        let cores = self.local.len();
        let memory = unsafe { slice::from_raw_parts_mut(begin as *mut Page, self.pages) };

        // build core local free lists
        let p_core = self.pages / cores;
        for core in 0..cores {
            let l = unsafe { &mut *self.local[core].get() };
            // build linked list
            for i in core * p_core + 1..(core + 1) * p_core {
                memory[i - 1]
                    .cast_mut::<Node>()
                    .set((begin + i * Page::SIZE) as *mut _);
            }
            memory[(core + 1) * p_core - 1]
                .cast_mut::<Node>()
                .set(null_mut());
            l.next.set((begin + core * p_core * Page::SIZE) as *mut _);
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
            let addr = node as *mut _ as u64;
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
        local.next.push(unsafe { &mut *(addr as *mut Node) });
        local.counter -= 1;
        Ok(())
    }

    fn is_free(&self, _addr: u64, _order: usize) -> bool {
        false
    }

    fn pages(&self) -> usize {
        self.pages
    }

    fn pages_needed(&self, cores: usize) -> usize {
        cores
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

#[repr(align(64))]
struct Local {
    next: Node,
    counter: usize,
}

impl Default for Local {
    fn default() -> Self {
        Self {
            next: Node::new(),
            counter: 0,
        }
    }
}

struct Node(*mut Node);

impl Node {
    fn new() -> Self {
        Self(null_mut())
    }
    fn set(&mut self, v: *mut Node) {
        self.0 = v;
    }
    fn push(&mut self, v: &mut Node) {
        v.0 = self.0;
        self.0 = v;
    }
    fn pop(&mut self) -> Option<&mut Node> {
        if !self.0.is_null() {
            let curr = unsafe { &mut *self.0 };
            self.0 = curr.0;
            Some(curr)
        } else {
            None
        }
    }
}
