use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, warn};

use super::{Alloc, Error, Result, Size};
use crate::table::Table;
use crate::util::Page;

pub struct MallocAlloc {
    local: Vec<Local>,
}

#[repr(align(64))]
struct Local {
    counter: AtomicUsize,
}

const INITIALIZING: *mut MallocAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<MallocAlloc> = AtomicPtr::new(null_mut());

impl Alloc for MallocAlloc {
    fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if !memory.is_empty() {
            warn!("This allocator uses its own memory range");
        }

        if unsafe {
            SHARED
                .compare_exchange(null_mut(), INITIALIZING, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        } {
            return Err(Error::Uninitialized);
        }

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || Local {
            counter: AtomicUsize::new(0),
        });

        let alloc = Box::new(MallocAlloc { local });
        let alloc = Box::leak(alloc);

        unsafe { SHARED.store(alloc, Ordering::SeqCst) };
        Ok(())
    }

    fn uninit() {
        let ptr = unsafe { SHARED.swap(INITIALIZING, Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");

        let alloc = unsafe { &mut *ptr };

        drop(unsafe { Box::from_raw(alloc) });
        unsafe { SHARED.store(null_mut(), Ordering::SeqCst) };
    }

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
            error!("Invalid size {size:?}");
            return Err(Error::Memory);
        }

        let ptr = unsafe { libc::malloc(Table::m_span(0)) };
        if !ptr.is_null() {
            self.local[core]
                .counter
                .fetch_add(Table::span(0), Ordering::Relaxed);
            Ok(ptr as u64)
        } else {
            Err(Error::Memory)
        }
    }

    fn put(&self, core: usize, addr: u64) -> Result<()> {
        unsafe { libc::free(addr as *mut _) };
        self.local[core]
            .counter
            .fetch_sub(Table::span(0), Ordering::Relaxed);
        Ok(())
    }

    fn allocated_pages(&self) -> usize {
        self.local
            .iter()
            .map(|l| l.counter.load(Ordering::Relaxed))
            .sum()
    }
}
