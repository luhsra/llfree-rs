use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use log::{error, info, warn};

use super::Alloc;
use crate::entry::Entry3;
use crate::leaf_alloc::LeafAllocator;
use crate::table::{AtomicBuffer, Table};
use crate::{Error, Meta, Page, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};

const PTE3_FULL: usize = Table::span(2) - 2 * Table::span(1);

/// Volatile shared metadata
#[repr(align(64))]
pub struct AllocStack {
    begin: usize,
    pages: usize,
    meta: *mut Meta,
    pub local: Vec<LeafAllocator>,
    entries: Entry3Buf,

    empty: Mutex<Vec<usize>>,
    partial_l1: Mutex<Vec<usize>>,
    partial_l0: Mutex<Vec<usize>>,
}

struct Entry3Buf(Vec<AtomicU64>);
impl Entry3Buf {
    fn new(len: usize) -> Self {
        let mut entries = Vec::with_capacity(len);
        entries.resize_with(len, || AtomicU64::new(0));
        Self(entries)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl AtomicBuffer<Entry3> for Entry3Buf {
    fn entry(&self, i: usize) -> &AtomicU64 {
        &self.0[i]
    }
}

const INITIALIZING: *mut AllocStack = usize::MAX as _;
static mut SHARED: AtomicPtr<AllocStack> = AtomicPtr::new(null_mut());

impl Alloc for AllocStack {
    fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        warn!("initializing c={cores} {:?}", memory.as_ptr_range());
        if unsafe {
            SHARED
                .compare_exchange(null_mut(), INITIALIZING, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        } {
            return Err(Error::Uninitialized);
        }

        let alloc = Self::new(cores, memory)?;
        let alloc = Box::leak(Box::new(alloc));
        let meta = alloc.meta();

        warn!("init");
        if meta.pages.load(Ordering::SeqCst) == alloc.pages
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", alloc.pages);
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Allocator unexpectedly terminated");
            }
            let pages = alloc.recover(deep)?;
            warn!("Recovered pages {pages}");
        } else {
            warn!("Setup allocator state p={}", alloc.pages);
            alloc.local[0].clear();
            // Add all entries to the empty list
            alloc.empty.lock().unwrap().extend(0..alloc.entries.len());

            meta.pages.store(alloc.pages, Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        // init all leaf_allocs
        for (i, leaf) in alloc.local.iter().enumerate() {
            leaf.start_l0
                .store(alloc.reserve(Size::L0)?, Ordering::Relaxed);
            leaf.start_l1
                .store(alloc.reserve(Size::L1)?, Ordering::Relaxed);
            info!(
                "init {i} small={} huge={}",
                leaf.start_l0.load(Ordering::Relaxed),
                leaf.start_l1.load(Ordering::Relaxed),
            );
        }

        meta.active.store(1, Ordering::SeqCst);
        unsafe { SHARED.store(alloc, Ordering::SeqCst) };
        Ok(())
    }

    fn uninit() {
        let ptr = unsafe { SHARED.swap(INITIALIZING, Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");

        let alloc = unsafe { &mut *ptr };
        let meta = unsafe { &*alloc.meta };
        meta.active.store(0, Ordering::SeqCst);

        drop(unsafe { Box::from_raw(alloc) });
        unsafe { SHARED.store(null_mut(), Ordering::SeqCst) };
    }

    fn instance<'a>() -> &'a Self {
        let ptr = unsafe { SHARED.load(Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");
        unsafe { &*ptr }
    }

    fn begin(&self) -> usize {
        self.begin
    }

    fn pages(&self) -> usize {
        self.pages
    }

    fn meta<'a>(&self) -> &'a Meta {
        unsafe { &*self.meta }
    }

    fn get(&self, core: usize, size: Size) -> Result<usize> {
        if Size::L2 == size {
            return self.get_giant(core);
        }
        let local = &self.local[core];
        let start_a = if Size::L0 == size {
            &local.start_l0
        } else {
            &local.start_l1
        };
        let start = start_a.load(Ordering::Relaxed);
        let i = start / Table::span(2);
        let max = self.pages - Table::round(2, start);
        let new = match self.entries.update(i, |v| v.inc(size, max)) {
            Ok(_) => start,
            Err(pte3) => {
                info!("Try reserve new {pte3:?}");
                // reserve *and* increment next
                let new = self.reserve_inc(size)?;
                self.unreserve(start);
                new
            }
        };
        let page = if size == Size::L0 {
            local.get(new)?
        } else {
            local.get_huge(new)?
        };
        start_a.store(page, Ordering::Relaxed);
        Ok(page)
    }

    fn put(&self, core: usize, page: usize) -> Result<Size> {
        let i = page / Table::span(2);
        let pte3 = self.entries.get(i);
        match pte3.size() {
            Some(Size::L2) => self.put_giant(core, page, pte3),
            _ => self.put_small(core, page, pte3),
        }
    }

    fn allocated_pages(&self) -> usize {
        let mut pages = 0;
        for i in 0..Table::num_pts(2, self.pages) {
            let pte = self.entries.get(i);
            // warn!("{i:>3}: {pte:?}");
            if pte.size() == Some(Size::L2) {
                pages += Table::span(2);
            } else {
                pages += pte.pages();
            }
        }
        pages
    }
}

impl AllocStack {
    fn new(cores: usize, memory: &[Page]) -> Result<Self> {
        if memory.len() < MIN_PAGES * cores {
            error!("memory {} < {}", memory.len(), MIN_PAGES * cores);
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        let pages = (memory.len() - 1).min(MAX_PAGES);
        let begin = memory.as_ptr() as usize;

        let meta = (begin + pages * Page::SIZE) as *mut Meta;
        let meta = unsafe { &mut *meta };

        // level 2 tables are stored at the end of the NVM
        let pages = pages - Table::num_pts(2, pages);

        // Array with all pte3
        let pte3_num = Table::num_pts(2, pages);
        warn!("alloc entries {pte3_num}");
        let entries = Entry3Buf::new(pte3_num);

        warn!("alloc local {cores}");
        let local = vec![LeafAllocator::new(begin, pages); cores];

        warn!("alloc lists {pte3_num}");
        let empty = Mutex::new(Vec::with_capacity(pte3_num));
        let partial_l0 = Mutex::new(Vec::with_capacity(pte3_num));
        let partial_l1 = Mutex::new(Vec::with_capacity(pte3_num));

        Ok(Self {
            begin,
            pages,
            meta,
            local,
            entries,
            empty,
            partial_l1,
            partial_l0,
        })
    }

    fn recover(&self, deep: bool) -> Result<usize> {
        let mut total = 0;
        for i in 0..Table::num_pts(2, self.pages) {
            let page = i * Table::span(2);
            let (pages, size) = self.local[0].recover(page, deep)?;
            if size == Size::L2 {
                self.entries.set(i, Entry3::new_giant());
            } else {
                self.entries.set(i, Entry3::new_table(pages, size, false));

                // Add to lists
                if pages == 0 {
                    self.empty.lock().unwrap().push(i);
                } else if pages < PTE3_FULL {
                    self.partial(size).push(i);
                }
            }
            total += pages;
        }
        Ok(total)
    }

    fn partial<'a, 'b: 'a>(&'b self, size: Size) -> MutexGuard<'a, Vec<usize>> {
        if size == Size::L0 {
            &self.partial_l0
        } else {
            &self.partial_l1
        }
        .lock()
        .unwrap()
    }

    fn reserve(&self, size: Size) -> Result<usize> {
        {
            let mut partial = self.partial(size);
            if let Some(i) = partial.pop() {
                let max = self.pages - i * Table::span(2);
                self.entries.update(i, |v| v.reserve(size, max)).unwrap();
                warn!("reserved partial {i}");
                return Ok(i * Table::span(2));
            }
        };
        {
            let mut empty = self.empty.lock().unwrap();
            if let Some(i) = empty.pop() {
                let max = self.pages - i * Table::span(2);
                self.entries.update(i, |v| v.reserve(size, max)).unwrap();
                warn!("reserved empty {i}");
                return Ok(i * Table::span(2));
            }
        };
        error!("No memory");
        Err(Error::Memory)
    }

    fn reserve_inc(&self, size: Size) -> Result<usize> {
        {
            let mut partial = self.partial(size);
            if let Some(i) = partial.pop() {
                let max = self.pages - i * Table::span(2);
                self.entries.update(i, |v| v.inc(size, max)).unwrap();
                warn!("reserved partial {i}");
                return Ok(i * Table::span(2));
            }
        };
        {
            let mut empty = self.empty.lock().unwrap();
            if let Some(i) = empty.pop() {
                let max = self.pages - i * Table::span(2);
                self.entries.update(i, |v| v.inc(size, max)).unwrap();
                warn!("reserved empty {i}");
                return Ok(i * Table::span(2));
            }
        };
        error!("No memory");
        Err(Error::Memory)
    }

    fn unreserve(&self, start: usize) {
        let i = start / Table::span(2);
        if self.entries.update(i, Entry3::unreserve).is_err() {
            panic!("Unreserve failed")
        }
    }

    fn get_giant(&self, core: usize) -> Result<usize> {
        if let Some(i) = self.empty.lock().unwrap().pop() {
            match self.entries.cas(i, Entry3::new(), Entry3::new_giant()) {
                Ok(_) => {
                    self.local[core].persist(i * Table::span(2));
                    Ok(i * Table::span(2))
                }
                Err(pte3) => {
                    error!("Corruption i{i} {pte3:?}");
                    Err(Error::Corruption)
                }
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn put_giant(&self, core: usize, page: usize, pte3: Entry3) -> Result<Size> {
        let i = page / Table::span(2);
        if page % Table::span(2) != 0 {
            error!("Invalid align {page:x}");
            return Err(Error::Address);
        }
        self.local[core].clear_giant(page);

        match self.entries.cas(i, pte3, Entry3::new()) {
            Ok(_) => {
                // Add to empty list
                self.empty.lock().unwrap().push(i);
                Ok(Size::L2)
            }
            Err(_) => {
                error!("CAS invalid i{i}");
                Err(Error::Address)
            }
        }
    }

    fn put_small(&self, core: usize, page: usize, pte3: Entry3) -> Result<Size> {
        if pte3.pages() == 0 {
            error!("Not allocated {page}");
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let size = self.local[core].put(page)?;
        match self.entries.update(i, |v| v.dec(size)) {
            Ok(pte3) => {
                let new_pages = pte3.pages() - Table::span(size as _);

                if pte3.pages() >= PTE3_FULL && new_pages < PTE3_FULL {
                    // Add back to partial
                    self.partial(size).push(i);
                } else if new_pages == 0 {
                    // Add back to empty
                    let mut partial = self.partial(size);
                    if let Some(idx) = partial.iter().copied().position(|v| v == i) {
                        partial.swap_remove(idx);
                    }
                    self.empty.lock().unwrap().push(i);
                }

                Ok(size)
            }
            Err(_) => {
                error!("Corruption l3 i{i} p=-{size:?}");
                Err(Error::Corruption)
            }
        }
    }
}
