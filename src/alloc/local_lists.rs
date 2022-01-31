use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, warn};

use super::{Alloc, Error, Result, Size, MIN_PAGES};
use crate::util::Page;

#[repr(align(64))]
pub struct LocalListAlloc {
    memory: Range<*const Page>,
    local: Vec<Local>,
}

const INITIALIZING: *mut LocalListAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<LocalListAlloc> = AtomicPtr::new(null_mut());

impl Alloc for LocalListAlloc {
    #[cold]
    fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < cores * MIN_PAGES {
            error!("Not enough memory {} < {}", memory.len(), cores * MIN_PAGES);
            return Err(Error::Memory);
        }

        if unsafe {
            SHARED
                .compare_exchange(null_mut(), INITIALIZING, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        } {
            return Err(Error::Uninitialized);
        }

        let begin = memory.as_ptr() as usize;
        let pages = memory.len();

        // build core local free lists
        let mut local = Vec::with_capacity(cores);
        let p_core = pages / cores;
        for core in 0..cores {
            let l = Local::new();
            // build linked list
            for i in core * p_core + 1..(core + 1) * p_core {
                memory[i - 1]
                    .cast::<Node>()
                    .set((begin + i * Page::SIZE) as *mut _);
            }
            memory[(core + 1) * p_core - 1]
                .cast::<Node>()
                .set(null_mut());
            l.next.set((begin + core * p_core * Page::SIZE) as *mut _);
            local.push(l);
        }

        let alloc = Box::new(LocalListAlloc {
            memory: memory.as_ptr_range(),
            local,
        });
        let alloc = Box::leak(alloc);

        unsafe { SHARED.store(alloc, Ordering::SeqCst) };
        Ok(())
    }

    #[cold]
    fn uninit() {
        let ptr = unsafe { SHARED.swap(INITIALIZING, Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");

        let alloc = unsafe { &mut *ptr };

        drop(unsafe { Box::from_raw(alloc) });
        unsafe { SHARED.store(null_mut(), Ordering::SeqCst) };
    }

    #[cold]
    fn destroy() {
        Self::uninit();
    }

    fn instance<'a>() -> &'a Self {
        let ptr = unsafe { SHARED.load(Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");
        unsafe { &*ptr }
    }

    fn get(&self, core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("{size:?} not supported");
            return Err(Error::Memory);
        }

        let l = &self.local[core];
        if let Some(node) = l.next.pop() {
            l.counter.fetch_add(1, Ordering::Relaxed);
            let addr = node as *mut _ as u64;
            debug_assert!(
                addr % Page::SIZE as u64 == 0 && self.memory.contains(&(addr as _)),
                "{:x}",
                addr
            );
            Ok(addr)
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Address);
        }

        let l = &self.local[core];
        l.next.push(unsafe { &mut *(addr as *mut Node) });
        l.counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = 0;
        for l in &self.local {
            pages += l.counter.load(Ordering::Relaxed);
        }
        pages
    }
}

#[repr(align(64))]
struct Local {
    next: Node,
    counter: AtomicUsize,
}

impl Local {
    fn new() -> Self {
        Self {
            next: Node::new(),
            counter: AtomicUsize::new(0),
        }
    }
}

struct Node(AtomicPtr<Node>);

impl Node {
    fn new() -> Self {
        Self(AtomicPtr::new(null_mut()))
    }
    fn set(&self, v: *mut Node) {
        self.0.store(v, Ordering::Relaxed);
    }
    fn push(&self, v: &mut Node) {
        let next = self.0.load(Ordering::Relaxed);
        v.0.store(next, Ordering::Relaxed);
        self.0.store(v, Ordering::Relaxed);
    }
    fn pop(&self) -> Option<&mut Node> {
        let curr = self.0.load(Ordering::Relaxed);
        if !curr.is_null() {
            let curr = unsafe { &mut *curr };
            let next = curr.0.load(Ordering::Relaxed);
            self.0.store(next, Ordering::Relaxed);
            Some(curr)
        } else {
            None
        }
    }
}
