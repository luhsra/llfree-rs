use std::alloc::Layout;
use std::ops::Range;
use std::ptr::{null_mut, NonNull};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

use log::{error, warn};

use crate::{table::Table, util::Page};

use super::{Alloc, Error, Result, Size};

#[repr(align(64))]
pub struct BuddyAlloc {
    memory: Range<*const Page>,
    inner: Mutex<buddy_system_allocator::Heap<36>>,
}

const INITIALIZING: *mut BuddyAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<BuddyAlloc> = AtomicPtr::new(null_mut());

impl Alloc for BuddyAlloc {
    fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if unsafe {
            SHARED
                .compare_exchange(null_mut(), INITIALIZING, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        } {
            return Err(Error::Uninitialized);
        }

        let mut heap = buddy_system_allocator::Heap::new();
        unsafe { heap.add_to_heap(memory.as_ptr() as _, memory.as_ptr_range().end as _) };
        let bytes = heap.stats_total_bytes();
        warn!("avaliable bytes {bytes} ({})", bytes / Page::SIZE);

        let alloc = Self {
            memory: memory.as_ptr_range(),
            inner: Mutex::new(heap),
        };
        let alloc = Box::leak(Box::new(alloc));

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

    fn instance<'a>() -> &'a Self {
        let ptr = unsafe { SHARED.load(Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");
        unsafe { &*ptr }
    }

    fn destroy() {
        Self::uninit();
    }

    fn get(&self, _core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("{size:?} not supported");
            return Err(Error::Memory);
        }

        let size = Table::m_span(Size::L0 as _);
        let layout = unsafe { Layout::from_size_align_unchecked(size, Page::SIZE) };
        match self.inner.lock().unwrap().alloc(layout) {
            Ok(val) => Ok(val.as_ptr() as u64),
            Err(_) => Err(Error::Memory),
        }
    }

    fn put(&self, _core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }

        let size = Table::m_span(Size::L0 as _);
        let layout = unsafe { Layout::from_size_align_unchecked(size, Page::SIZE) };
        let ptr = unsafe { NonNull::new_unchecked(addr as *mut u8) };
        self.inner.lock().unwrap().dealloc(ptr, layout);
        Ok(())
    }

    fn allocated_pages(&self) -> usize {
        self.inner.lock().unwrap().stats_alloc_actual() / Page::SIZE
    }
}
