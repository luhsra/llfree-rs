use core::fmt;
use core::ops::Range;
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};
use spin::mutex::TicketMutex;

use super::{Alloc, Error, Result, Size, MIN_PAGES};
use crate::util::Page;

/// Simple volatile 4K page allocator that uses a single shared linked lists
/// protected by a ticked lock.
/// The linked list is build directly within the pages,
/// storing the next pointers at the beginning of the free pages.
///
/// As expected the contention on the ticket lock is very high.
#[repr(align(64))]
pub struct ListLockedAlloc {
    memory: Range<*const Page>,
    next: TicketMutex<Node>,
    /// CPU local metadata
    local: Box<[LocalCounter]>,
}

#[repr(align(64))]
struct LocalCounter {
    counter: AtomicUsize,
}

unsafe impl Send for ListLockedAlloc {}
unsafe impl Sync for ListLockedAlloc {}

impl fmt::Debug for ListLockedAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        for (t, l) in self.local.iter().enumerate() {
            writeln!(f, "    L {t:>2} C={}", l.counter.load(Ordering::Relaxed))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Default for ListLockedAlloc {
    fn default() -> Self {
        Self {
            memory: null()..null(),
            next: TicketMutex::new(Node::new()),
            local: Box::new([]),
        }
    }
}

impl Alloc for ListLockedAlloc {
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

        // build free lists
        for i in 1..pages {
            memory[i - 1]
                .cast_mut::<Node>()
                .set((begin + i * Page::SIZE) as *mut _);
        }
        memory[pages - 1].cast_mut::<Node>().set(null_mut());

        self.memory = memory.as_ptr_range();
        self.next = TicketMutex::new(Node(begin as _));

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || LocalCounter {
            counter: AtomicUsize::new(0),
        });
        self.local = local.into();

        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("{size:?} not supported");
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

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<Size> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Address);
        }

        self.next.lock().push(unsafe { &mut *(addr as *mut Node) });
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(Size::L0)
    }

    fn pages(&self) -> usize {
        unsafe { self.memory.end.offset_from(self.memory.start) as usize }
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        self.local
            .iter()
            .map(|c| c.counter.load(Ordering::SeqCst))
            .sum()
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
