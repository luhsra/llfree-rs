use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, warn};
use spin::mutex::TicketMutex;

use super::{Alloc, Error, Result, Size, MIN_PAGES};
use crate::util::Page;

#[repr(align(64))]
pub struct LockedListAlloc {
    memory: Range<*const Page>,
    next: TicketMutex<Node>,
    counter: AtomicUsize,
}

const INITIALIZING: *mut LockedListAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<LockedListAlloc> = AtomicPtr::new(null_mut());

impl Alloc for LockedListAlloc {
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

        // build free lists
        for i in 1..pages {
            memory[i - 1]
                .cast::<Node>()
                .set((begin + i * Page::SIZE) as *mut _);
        }
        memory[pages - 1].cast::<Node>().set(null_mut());

        let alloc = Box::new(LockedListAlloc {
            memory: memory.as_ptr_range(),
            next: TicketMutex::new(Node(AtomicPtr::new(begin as _))),
            counter: AtomicUsize::new(0),
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

    fn get(&self, _core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("{size:?} not supported");
            return Err(Error::Memory);
        }

        if let Some(node) = self.next.lock().pop() {
            self.counter.fetch_add(1, Ordering::Relaxed);
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

    fn put(&self, _core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Address);
        }

        self.next.lock().push(unsafe { &mut *(addr as *mut Node) });
        self.counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        self.counter.load(Ordering::SeqCst)
    }
}

struct Node(AtomicPtr<Node>);

impl Node {
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
