use core::fmt;
use core::ops::Range;
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::slice;
use alloc::vec::Vec;
use log::{error, info};
use spin::Mutex;

use super::{Alloc, MIN_PAGES};
use crate::util::Page;
use crate::{Error, Result};

/// Simple volatile 4K page allocator that uses a single shared linked lists
/// protected by a ticked lock.
/// The linked list is build directly within the pages,
/// storing the next pointers at the beginning of the free pages.
///
/// As expected the contention on the ticket lock is very high.
#[repr(align(64))]
pub struct ListLocked {
    memory: Range<*const Page>,
    next: Mutex<Node>,
    /// CPU local metadata
    local: Box<[LocalCounter]>,
}

#[repr(align(64))]
struct LocalCounter {
    counter: AtomicUsize,
}

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
            memory: null()..null(),
            next: Mutex::new(Node::new()),
            local: Box::new([]),
        }
    }
}

impl Alloc for ListLocked {
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

        self.memory = memory.as_ptr_range();

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || LocalCounter {
            counter: AtomicUsize::new(0),
        });
        self.local = local.into();

        Ok(())
    }

    #[cold]
    fn free_all(&self) -> Result<()> {
        let begin = self.memory.start as usize;
        let memory =
            unsafe { slice::from_raw_parts_mut(self.memory.start as *mut Page, self.pages()) };
        // build free lists
        for i in 1..self.pages() {
            memory[i - 1]
                .cast_mut::<Node>()
                .set((begin + i * Page::SIZE) as *mut _);
        }
        memory[self.pages() - 1].cast_mut::<Node>().set(null_mut());

        *self.next.lock() = Node(begin as _);
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local
                .counter
                .store(self.pages() / self.local.len(), Ordering::Relaxed);
        }
        *self.next.lock() = Node::new();
        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }

        if let Some(node) = self.next.lock().pop() {
            self.local[core].counter.fetch_add(1, Ordering::Relaxed);
            let addr = node as *mut _ as u64;
            debug_assert!(addr % Page::SIZE as u64 == 0 && self.memory.contains(&(addr as _)),);
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

        self.next.lock().push(unsafe { &mut *(addr as *mut Node) });
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn pages(&self) -> usize {
        unsafe { self.memory.end.offset_from(self.memory.start) as usize }
    }

    fn pages_needed(&self, cores: usize) -> usize {
        cores
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
