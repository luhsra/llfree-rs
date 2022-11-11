use core::fmt;
use core::ops::Range;
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};
use spin::Mutex;

use super::{Alloc, MIN_PAGES};
use crate::util::Page;
use crate::{Error, Result};

/// Simple volatile 4K page allocator that uses a single shared linked lists
/// protected by a ticked lock.
/// The linked list pointers are stored similar to Linux's in the struct pages.
///
/// As expected the contention on the ticket lock is very high.
#[repr(align(64))]
pub struct ListLocked {
    memory: Range<*const Page>,
    struct_pages: Box<[StructPage]>,
    /// CPU local metadata
    local: Box<[LocalCounter]>,
    /// Per page metadata
    next: Mutex<Node<StructPage>>,
}

#[repr(align(64))]
struct LocalCounter {
    counter: AtomicUsize,
}
const _: () = assert!(core::mem::align_of::<LocalCounter>() == 64);

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
            next: Mutex::new(Node::new(null_mut())),
            local: Box::new([]),
            struct_pages: Box::new([]),
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

        let mut struct_pages = Vec::with_capacity(memory.len());
        struct_pages.resize_with(memory.len(), || StructPage { next: null_mut() });
        self.struct_pages = struct_pages.into();

        Ok(())
    }

    #[cold]
    fn free_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local.counter.store(0, Ordering::Relaxed);
        }

        let mut next = self.next.lock(); // lock here to prevent races

        let begin = self.struct_pages.as_ptr();

        // Safety: The next pointer is locked
        let struct_pages = unsafe {
            core::slice::from_raw_parts_mut(
                self.struct_pages.as_ptr().cast_mut(),
                self.struct_pages.len(),
            )
        };
        // build free lists
        for i in 1..self.pages() {
            struct_pages[i - 1].next = unsafe { begin.add(i) as _ };
        }
        struct_pages[self.pages() - 1].next = null_mut();

        *next = Node::new(begin as _);
        Ok(())
    }

    fn reserve_all(&self) -> Result<()> {
        for local in self.local.iter() {
            local
                .counter
                .store(self.pages() / self.local.len(), Ordering::Relaxed);
        }
        *self.next.lock() = Node::new(null_mut());
        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order != 0 {
            error!("order {order:?} not supported");
            return Err(Error::Memory);
        }

        if let Some(node) = self.next.lock().pop() {
            self.local[core].counter.fetch_add(1, Ordering::Relaxed);
            let addr = self.from_struct_page(node as *mut _) as u64;
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

        let struct_page = self.to_struct_page(addr as *mut Page);
        self.next.lock().push(unsafe { &mut *struct_page });
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn is_free(&self, _addr: u64, _order: usize) -> bool {
        false
    }

    fn pages(&self) -> usize {
        unsafe { self.memory.end.offset_from(self.memory.start) as usize }
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

impl ListLocked {
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

    use super::ListLocked;

    type Allocator = ListLocked;

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
