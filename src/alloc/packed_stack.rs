use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use log::{error, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::Entry3;
use crate::leaf_alloc::{LeafAllocator, Leafs};
use crate::table::Table;
use crate::util::{Atomic, Page};

const PTE3_FULL: usize = Table::span(2) - 4 * Table::span(1);

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// Volatile shared metadata
#[repr(align(64))]
pub struct PackedStackAlloc {
    memory: Range<*const Page>,
    meta: *mut Meta,
    local: Vec<LeafAllocator<Self>>,
    entries: Vec<Atomic<Entry3>>,

    empty: Mutex<Vec<usize>>,
    partial_l1: Mutex<Vec<usize>>,
    partial_l0: Mutex<Vec<usize>>,
}

const INITIALIZING: *mut PackedStackAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<PackedStackAlloc> = AtomicPtr::new(null_mut());

impl Leafs for PackedStackAlloc {
    fn leafs<'a>() -> &'a [LeafAllocator<Self>] {
        &Self::instance().local
    }
}

impl Alloc for PackedStackAlloc {
    fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < MIN_PAGES * cores {
            error!("memory {} < {}", memory.len(), MIN_PAGES * cores);
            return Err(Error::Memory);
        }

        if unsafe {
            SHARED
                .compare_exchange(null_mut(), INITIALIZING, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        } {
            return Err(Error::Uninitialized);
        }

        let alloc = Self::new(cores, memory)?;
        let alloc = Box::leak(Box::new(alloc));
        let meta = unsafe { &mut *alloc.meta };

        warn!("init");
        if meta.pages.load(Ordering::SeqCst) == alloc.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", alloc.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            let pages = alloc.recover(deep)?;
            warn!("Recovered pages {pages}");
        } else {
            warn!("Setup allocator state p={}", alloc.pages());
            alloc.local[0].clear();
            // Add all entries to the empty list
            // TODO: handle last partially cut off subtree!
            alloc.empty.lock().unwrap().extend(0..alloc.entries.len());

            meta.pages.store(alloc.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
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

    fn destroy() {
        let alloc = Self::instance();
        let meta = unsafe { &*alloc.meta };
        meta.magic.store(0, Ordering::SeqCst);
        Self::uninit();
    }

    fn get(&self, core: usize, size: Size) -> Result<u64> {
        match size {
            Size::L2 => self.get_giant(core),
            _ => self.get_small(core, size),
        }
        .map(|p| unsafe { self.memory.start.add(p as _) } as u64)
    }

    fn put(&self, core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.memory.start) } as usize;

        let i = page / Table::span(2);
        let pte3 = self.entries[i].load();
        match pte3.size() {
            Some(Size::L2) => self.put_giant(core, page, pte3),
            _ => self.put_small(core, page, pte3),
        }
    }

    fn allocated_pages(&self) -> usize {
        let mut pages = 0;
        let mut sub = 0;
        for i in 0..Table::num_pts(2, self.pages()) {
            let pte = self.entries[i].load();
            warn!("{i:>3}: {pte:?}");
            if pte.size() == Some(Size::L2) {
                pages += Table::span(2);
            } else if pte.reserved() {
                // Pages freed by other cores
                let max = (self.pages() - i * Table::span(2) - 1).min(Table::span(2));
                sub += max - pte.pages();
            } else {
                pages += pte.pages();
            }
        }
        // Pages allocated in reserved subtrees
        for local in &self.local {
            pages += local.pte(Size::L0).load().pages();
            pages += local.pte(Size::L1).load().pages();
        }
        pages - sub
    }
}

impl PackedStackAlloc {
    fn new(cores: usize, memory: &mut [Page]) -> Result<Self> {
        // Last frame is reserved for metadata
        let pages = (memory.len() - 1).min(MAX_PAGES);
        let (memory, rem) = memory.split_at_mut(pages);
        let meta = rem[0].cast::<Meta>();

        // level 2 tables are stored at the end of the NVM
        let pages = pages - Table::num_pts(2, pages);
        let (memory, _pt2) = memory.split_at_mut(pages);

        // Array with all pte3
        let pte3_num = Table::num_pts(2, pages);

        let mut entries = Vec::with_capacity(pte3_num);
        entries.resize_with(pte3_num, || Atomic::new(Entry3::new()));
        let empty = Mutex::new(Vec::with_capacity(pte3_num));
        let partial_l0 = Mutex::new(Vec::with_capacity(pte3_num));
        let partial_l1 = Mutex::new(Vec::with_capacity(pte3_num));

        let local = vec![LeafAllocator::new(memory.as_ptr() as usize, pages); cores];

        Ok(Self {
            memory: memory.as_ptr_range(),
            meta,
            local,
            entries,
            empty,
            partial_l1,
            partial_l0,
        })
    }

    fn pages(&self) -> usize {
        (self.memory.end as usize - self.memory.start as usize) / Page::SIZE
    }

    fn recover(&self, deep: bool) -> Result<usize> {
        if deep {
            error!("Allocator unexpectedly terminated");
        }
        let mut total = 0;
        for i in 0..Table::num_pts(2, self.pages()) {
            let page = i * Table::span(2);
            let (pages, size) = self.local[0].recover(page, deep)?;
            if size == Size::L2 {
                self.entries[i].store(Entry3::new_giant());
            } else {
                self.entries[i].store(Entry3::new_table(pages, size, false));

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

    fn reserve_inc(&self, size: Size) -> Result<(usize, Entry3)> {
        if let Some(i) = self
            .partial(size)
            .pop()
            .or_else(|| self.empty.lock().unwrap().pop())
        {
            let max = self.pages() - i * Table::span(2) - 1;
            let pte = self.entries[i]
                .fetch_update(|v| v.reserve_fill(size, max))
                .unwrap();
            let mut pte = pte.inc(size, max).unwrap();
            pte.set_idx(i);

            warn!("reserved {i}");
            return Ok((i * Table::span(2), pte));
        }

        error!("No memory");
        Err(Error::Memory)
    }

    fn get_small(&self, core: usize, size: Size) -> Result<usize> {
        let local = &self.local[core];
        let start_a = local.start(size);
        let pte_a = local.pte(size);
        let mut start = start_a.load(Ordering::SeqCst);

        if start == usize::MAX {
            warn!("Try reserve first");
            let (s, pte) = self.reserve_inc(size)?;
            pte_a.store(pte);
            start = s
        } else {
            let i = start / Table::span(2);
            let max = self.pages() - Table::round(2, start) - 1;

            // Incremet or clear (atomic sync with put dec)
            if pte_a.fetch_update(|v| v.inc(size, max)).is_err() {
                warn!("Try reserve next");
                let (s, new_pte) = self.reserve_inc(size)?;
                start = s;
                let pte = pte_a.swap(new_pte);

                if let Ok(v) = self.entries[i].fetch_update(|v| v.unreserve_sub(pte, max)) {
                    let pte = v.unreserve_sub(pte, max).unwrap();
                    // Add to list if new counter is small enough
                    if pte.pages() == 0 {
                        self.empty.lock().unwrap().push(i);
                    } else if pte.pages() < PTE3_FULL {
                        self.partial(size).push(i);
                    }
                } else {
                    error!("Unreserve failed");
                    return Err(Error::Corruption);
                }
            }
        }

        let page = match size {
            Size::L0 => local.get(start)?,
            _ => local.get_huge(start)?,
        };
        start_a.store(page, Ordering::SeqCst);
        Ok(page)
    }

    fn get_giant(&self, core: usize) -> Result<usize> {
        if let Some(i) = self.empty.lock().unwrap().pop() {
            if self.entries[i]
                .compare_exchange(Entry3::new(), Entry3::new_giant())
                .is_ok()
            {
                self.local[core].persist(i * Table::span(2));
                Ok(i * Table::span(2))
            } else {
                error!("CAS invalid i{i}");
                Err(Error::Corruption)
            }
        } else {
            Err(Error::Memory)
        }
    }

    fn put_giant(&self, core: usize, page: usize, pte3: Entry3) -> Result<()> {
        let i = page / Table::span(2);
        if page % Table::span(2) != 0 {
            error!("Invalid align {page:x}");
            return Err(Error::Address);
        }
        self.local[core].clear_giant(page);

        if self.entries[i]
            .compare_exchange(pte3, Entry3::new())
            .is_ok()
        {
            // Add to empty list
            self.empty.lock().unwrap().push(i);
            Ok(())
        } else {
            error!("CAS invalid i{i}");
            Err(Error::Address)
        }
    }

    fn put_small(&self, core: usize, page: usize, pte3: Entry3) -> Result<()> {
        if pte3.pages() == 0 {
            error!("Not allocated {page}");
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let local = &self.local[core];
        let size = local.put(page)?;

        // Try decrement own pte first
        if local.pte(size).fetch_update(|v| v.dec_idx(size, i)).is_ok() {
            return Ok(());
        }

        // Subtree not owned by us
        if let Ok(pte3) = self.entries[i].fetch_update(|v| v.dec(size)) {
            // Add to free list if not reserved
            if !pte3.reserved() {
                let new_pages = pte3.pages() - Table::span(size as _);
                if pte3.pages() >= PTE3_FULL && new_pages < PTE3_FULL {
                    // Add back to partial
                    self.partial(size).push(i);
                } else if new_pages == 0 {
                    // Add back to empty (and remove from partial)
                    let mut partial = self.partial(size);
                    if let Some(idx) = partial.iter().copied().position(|v| v == i) {
                        partial.swap_remove(idx);
                    }
                    drop(partial);
                    self.empty.lock().unwrap().push(i);
                }
            }
            Ok(())
        } else {
            error!("Corruption l3 i{i} p=-{size:?}");
            Err(Error::Corruption)
        }
    }
}
