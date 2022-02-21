use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, warn};
use spin::{Mutex, MutexGuard};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::Entry3;
use crate::lower_alloc::{LowerAlloc, LowerAccess};
use crate::table::Table;
use crate::util::{Atomic, Page};

const PTE3_FULL: usize = 8 * Table::span(1);

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// Volatile shared metadata
#[repr(align(64))]
pub struct ArrayLockedAlloc {
    memory: Range<*const Page>,
    meta: *mut Meta,
    local: Vec<LowerAlloc<Self>>,
    entries: Vec<Atomic<Entry3>>,

    empty: Mutex<Vec<usize>>,
    partial_l1: Mutex<Vec<usize>>,
    partial_l0: Mutex<Vec<usize>>,
}

const INITIALIZING: *mut ArrayLockedAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<ArrayLockedAlloc> = AtomicPtr::new(null_mut());

impl LowerAccess for ArrayLockedAlloc {
    fn lower_allocs<'a>() -> &'a [LowerAlloc<Self>] {
        &Self::instance().local
    }
}

impl Alloc for ArrayLockedAlloc {
    #[cold]
    fn init(cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
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
            return Err(Error::Initialization);
        }

        let alloc = Self::new(cores, memory)?;
        let alloc = Box::leak(Box::new(alloc));
        let meta = unsafe { &mut *alloc.meta };

        warn!("init");
        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == alloc.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", alloc.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            let pages = alloc.recover(deep)?;
            warn!("Recovered pages {pages}");
        } else {
            warn!("Setup allocator state p={}", alloc.pages());
            alloc.setup();

            meta.pages.store(alloc.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        meta.active.store(1, Ordering::SeqCst);
        unsafe { SHARED.store(alloc, Ordering::SeqCst) };
        Ok(())
    }

    #[cold]
    fn uninit() {
        let ptr = unsafe { SHARED.swap(INITIALIZING, Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");

        let alloc = unsafe { &mut *ptr };
        let meta = unsafe { &*alloc.meta };
        meta.active.store(0, Ordering::SeqCst);

        drop(unsafe { Box::from_raw(alloc) });
        unsafe { SHARED.store(null_mut(), Ordering::SeqCst) };
    }

    #[cold]
    fn destroy() {
        let alloc = Self::instance();
        let meta = unsafe { &*alloc.meta };
        meta.magic.store(0, Ordering::SeqCst);
        Self::uninit();
    }

    fn instance<'a>() -> &'a Self {
        let ptr = unsafe { SHARED.load(Ordering::SeqCst) };
        assert!(!ptr.is_null() && ptr != INITIALIZING, "Not initialized");
        unsafe { &*ptr }
    }

    fn get(&self, core: usize, size: Size) -> Result<u64> {
        match size {
            Size::L2 => self.get_giant(core),
            _ => self.get_small(core, size == Size::L1),
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
        let pte = self.entries[i].load();
        if pte.page() {
            self.put_giant(core, page)
        } else {
            self.put_small(core, page, pte)
        }
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Table::num_pts(2, self.pages()) {
            let pte = self.entries[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.free();
        }
        // Pages allocated in reserved subtrees
        for local in &self.local {
            pages -= local.pte(false).load().free();
            pages -= local.pte(true).load().free();
        }
        pages
    }
}

impl ArrayLockedAlloc {
    #[cold]
    fn new(cores: usize, memory: &mut [Page]) -> Result<Self> {
        // Last frame is reserved for metadata
        let pages = (memory.len() - 1).min(MAX_PAGES);
        let (memory, rem) = memory.split_at_mut(pages);
        let meta = rem[0].cast_mut::<Meta>();

        // level 2 tables are stored at the end of the NVM
        let pages = pages - Table::num_pts(2, pages);
        let (memory, _pt2) = memory.split_at_mut(pages);

        // Array with all pte3
        let pte3_num = Table::num_pts(2, pages);
        let mut entries = Vec::with_capacity(pte3_num);
        entries.resize_with(pte3_num, || Atomic::new(Entry3::new()));

        let local = vec![LowerAlloc::new(memory.as_ptr() as usize, pages); cores];

        Ok(Self {
            memory: memory.as_ptr_range(),
            meta,
            local,
            entries,
            empty: Mutex::new(Vec::with_capacity(pte3_num)),
            partial_l1: Mutex::new(Vec::with_capacity(pte3_num)),
            partial_l0: Mutex::new(Vec::with_capacity(pte3_num)),
        })
    }

    fn pages(&self) -> usize {
        (self.memory.end as usize - self.memory.start as usize) / Page::SIZE
    }

    #[cold]
    fn setup(&mut self) {
        self.local[0].clear();

        // Add all entries to the empty list
        let mut empty = self.empty.lock();

        let pte3_num = Table::num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            empty.push(i);
            self.entries[i] = Atomic::new(Entry3::new().with_free(Table::span(2)));
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * Table::span(2)).min(Table::span(2));
        self.entries[pte3_num - 1] = Atomic::new(Entry3::new().with_free(max));

        if max == Table::span(2) {
            empty.push(pte3_num - 1);
        } else if max > PTE3_FULL {
            self.partial_l0.lock().push(pte3_num - 1);
        }
    }

    #[cold]
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
                if pages == Table::span(2) {
                    self.empty.lock().push(i);
                } else if pages > PTE3_FULL {
                    self.partial(size == Size::L1).push(i);
                }
            }
            total += pages;
        }
        Ok(total)
    }

    fn partial<'a, 'b: 'a>(&'b self, huge: bool) -> MutexGuard<'a, Vec<usize>> {
        if huge {
            &self.partial_l1
        } else {
            &self.partial_l0
        }
        .lock()
    }

    fn reserve_dec(&self, huge: bool) -> Result<(usize, Entry3)> {
        while let Some(i) = self.partial(huge).pop() {
            warn!("reserve partial {i}");
            match self.entries[i].update(|v| v.reserve_partial(huge, PTE3_FULL)) {
                Ok(pte) => {
                    let pte = pte.dec(huge).unwrap().with_idx(i);
                    return Ok((i * Table::span(2), pte));
                }
                Err(pte) => {
                    // Skip empty entries
                    if !pte.reserved() && pte.free() == Table::span(2) {
                        self.empty.lock().push(i);
                    }
                }
            }
        }

        while let Some(i) = self.empty.lock().pop() {
            warn!("reserve empty {i}");
            if let Ok(pte) = self.entries[i].update(|v| v.reserve_empty(huge)) {
                let pte = pte.dec(huge).unwrap().with_idx(i);
                return Ok((i * Table::span(2), pte));
            }
        }

        error!("No memory");
        Err(Error::Memory)
    }

    fn swap_reserved(&self, huge: bool, new_pte: Entry3, pte_a: &Atomic<Entry3>) -> Result<()> {
        let pte = pte_a.swap(new_pte);
        let i = pte.idx();
        let max = (self.pages() - i * Table::span(2)).min(Table::span(2));

        if let Ok(v) = self.entries[i].update(|v| v.unreserve_add(huge, pte, max)) {
            // Add to list if new counter is small enough
            let new_pages = v.free() + pte.free();
            if new_pages == max {
                self.empty.lock().push(i);
            } else if new_pages > PTE3_FULL {
                self.partial(huge).push(i);
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    fn get_small(&self, core: usize, huge: bool) -> Result<usize> {
        let local = &self.local[core];
        let start_a = local.start(huge);
        let pte_a = local.pte(huge);
        let mut start = start_a.load(Ordering::SeqCst);

        if start == usize::MAX {
            warn!("Try reserve first");
            let (s, pte) = self.reserve_dec(huge)?;
            pte_a.store(pte);
            start = s
        } else {
            // Incremet or clear (atomic sync with put dec)
            if pte_a.update(|v| v.dec(huge)).is_err() {
                warn!("Try reserve next");
                let (s, new_pte) = self.reserve_dec(huge)?;
                self.swap_reserved(huge, new_pte, pte_a)?;
                start = s;
            }
        }

        let page = if huge {
            local.get_huge(start)?
        } else {
            local.get(start)?
        };
        start_a.store(page, Ordering::SeqCst);
        Ok(page)
    }

    fn get_giant(&self, core: usize) -> Result<usize> {
        if let Some(i) = self.empty.lock().pop() {
            if self.entries[i]
                .compare_exchange(
                    Entry3::new().with_free(Table::span(2)),
                    Entry3::new_giant(),
                )
                .is_ok()
            {
                self.local[core].persist(i * Table::span(2));
                Ok(i * Table::span(2))
            } else {
                error!("CAS invalid i{i}");
                Err(Error::Corruption)
            }
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn put_giant(&self, core: usize, page: usize) -> Result<()> {
        let i = page / Table::span(2);
        if page % Table::span(2) != 0 {
            error!("Invalid align {page:x}");
            return Err(Error::Address);
        }
        self.local[core].clear_giant(page);

        if self.entries[i]
            .compare_exchange(
                Entry3::new_giant(),
                Entry3::new().with_free(Table::span(2)),
            )
            .is_ok()
        {
            // Add to empty list
            self.empty.lock().push(i);
            Ok(())
        } else {
            error!("CAS invalid i{i}");
            Err(Error::Address)
        }
    }

    fn put_small(&self, core: usize, page: usize, pte: Entry3) -> Result<()> {
        let max = self
            .pages()
            .saturating_sub(Table::round(2, page))
            .min(Table::span(2));
        if pte.free() == max {
            error!("Not allocated {page} (i{})", page / Table::span(2));
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let local = &self.local[core];
        let size = local.put(page)?;

        // Try decrement own pte first
        let pte_a = local.pte(size);
        if pte_a.update(|v| v.inc_idx(size, i, max)).is_ok() {
            return Ok(());
        }

        // Subtree not owned by us
        if let Ok(pte) = self.entries[i].update(|v| v.inc(size, max)) {
            if !pte.reserved() {
                let new_pages = pte.free() + Table::span(size as _);
                if pte.free() <= PTE3_FULL && new_pages > PTE3_FULL {
                    // Try to reserve it for bulk frees
                    if let Ok(pte) = self.entries[i].update(|v| v.reserve(size)) {
                        let pte = pte.with_idx(i);
                        warn!("put reserve {i}");
                        self.swap_reserved(size, pte, pte_a)?;
                        local.start(size).store(page, Ordering::SeqCst);
                    } else {
                        // Add to partially free list
                        self.partial(size).push(i);
                    }
                }
            }
            Ok(())
        } else {
            error!("Corruption l3 i{i} p=-{size:?}");
            Err(Error::Corruption)
        }
    }
}
