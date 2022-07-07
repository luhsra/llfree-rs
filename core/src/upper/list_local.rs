use core::fmt;
use core::ops::Range;
use core::ptr::{null, null_mut};
use core::cell::UnsafeCell;

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};

use super::{Alloc, Error, Result, Size, MIN_PAGES};
use crate::util::Page;

/// Simple volatile 4K page allocator that uses CPU-local linked lists.
/// During initialization allocators memory is split into pages
/// and evenly distributed to the cores.
/// The linked lists are build directly within the pages,
/// storing the next pointers at the beginning of the free pages.
///
/// No extra load balancing is made, if a core runs out of memory,
/// the allocation fails.
#[repr(align(64))]
pub struct ListLocalAlloc {
    memory: Range<*const Page>,
    /// CPU local metadata
    local: Box<[UnsafeCell<Local>]>,
    pages: usize,
}

unsafe impl Send for ListLocalAlloc {}
unsafe impl Sync for ListLocalAlloc {}

impl fmt::Debug for ListLocalAlloc {
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

impl Default for ListLocalAlloc {
    fn default() -> Self {
        Self {
            memory: null()..null(),
            local: Box::new([]),
            pages: 0,
        }
    }
}

impl Alloc for ListLocalAlloc {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], _overwrite: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < cores * MIN_PAGES {
            error!("Not enough memory {} < {}", memory.len(), cores * MIN_PAGES);
            return Err(Error::Memory);
        }

        let begin = memory.as_ptr() as usize;
        let pages = memory.len();

        // build core local free lists
        let mut local = Vec::with_capacity(cores);
        let p_core = pages / cores;
        for core in 0..cores {
            let mut l = Local::default();
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
            local.push(UnsafeCell::new(l));
        }

        self.pages = p_core * cores;
        self.memory = memory[..p_core * cores].as_ptr_range();
        self.local = local.into();
        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("{size:?} not supported");
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

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<Size> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Address);
        }

        let local = unsafe { &mut *self.local[core].get() };
        local.next.push(unsafe { &mut *(addr as *mut Node) });
        local.counter -= 1;
        Ok(Size::L0)
    }

    fn pages(&self) -> usize {
        self.pages
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = 0;
        for local in self.local.iter() {
            let local = unsafe { &*local.get() };
            pages += local.counter;
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
