use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::Entry3;
use crate::leaf_alloc::{LeafAllocator, Leafs};
use crate::table::Table;
use crate::util::{AStack, Atomic, Page};

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
pub struct ArrayAtomicAlloc {
    memory: Range<*const Page>,
    meta: *mut Meta,
    local: Vec<LeafAllocator<Self>>,
    entries: Vec<Atomic<Entry3>>,

    empty: AStack<Entry3>,
    partial_l1: AStack<Entry3>,
    partial_l0: AStack<Entry3>,
}

const INITIALIZING: *mut ArrayAtomicAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<ArrayAtomicAlloc> = AtomicPtr::new(null_mut());

impl Leafs for ArrayAtomicAlloc {
    fn leafs<'a>() -> &'a [LeafAllocator<Self>] {
        &Self::instance().local
    }
}

impl Alloc for ArrayAtomicAlloc {
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
            return Err(Error::Uninitialized);
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
            Some(Size::L2) => self.put_giant(core, page),
            _ => self.put_small(core, page, pte3),
        }
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Table::num_pts(2, self.pages()) {
            let pte = self.entries[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.pages();
        }
        // Pages allocated in reserved subtrees
        for local in &self.local {
            pages -= local.pte(Size::L0).load().pages();
            pages -= local.pte(Size::L1).load().pages();
        }
        pages
    }
}

impl ArrayAtomicAlloc {
    #[cold]
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

        let local = vec![LeafAllocator::new(memory.as_ptr() as usize, pages); cores];

        Ok(Self {
            memory: memory.as_ptr_range(),
            meta,
            local,
            entries,
            empty: AStack::new(),
            partial_l1: AStack::new(),
            partial_l0: AStack::new(),
        })
    }

    fn pages(&self) -> usize {
        (self.memory.end as usize - self.memory.start as usize) / Page::SIZE
    }

    #[cold]
    fn setup(&mut self) {
        self.local[0].clear();

        // Add all entries to the empty list
        let pte3_num = Table::num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            self.entries[i] = Atomic::new(Entry3::new().with_pages(Table::span(2)));
            self.empty.push(&self.entries, i);
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * Table::span(2)).min(Table::span(2));
        self.entries[pte3_num - 1] = Atomic::new(Entry3::new().with_pages(max));

        if max == Table::span(2) {
            self.empty.push(&self.entries, pte3_num - 1);
        } else if max > PTE3_FULL {
            self.partial_l0.push(&self.entries, pte3_num - 1);
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
                    self.empty.push(&self.entries, i);
                } else if pages > PTE3_FULL {
                    self.partial(size).push(&self.entries, i);
                }
            }
            total += pages;
        }
        Ok(total)
    }

    fn partial(&self, size: Size) -> &AStack<Entry3> {
        if size == Size::L0 {
            &self.partial_l0
        } else {
            &self.partial_l1
        }
    }

    fn reserve_dec(&self, size: Size) -> Result<(usize, Entry3)> {
        while let Some(i) = self.partial(size).pop(&self.entries) {
            // Skip empty entries
            if self.entries[i].load().pages() < Table::span(2) {
                warn!("reserve partial {i}");
                let pte = self.entries[i].update(|v| v.reserve_take(size)).unwrap();
                let pte = pte.dec(size).unwrap().with_idx(i);
                return Ok((i * Table::span(2), pte));
            } else {
                self.empty.push(&self.entries, i);
            }
        }

        if let Some(i) = self.empty.pop(&self.entries) {
            warn!("reserve empty {i}");
            let pte = self.entries[i].update(|v| v.reserve_take(size)).unwrap();
            let pte = pte.dec(size).unwrap().with_idx(i);
            return Ok((i * Table::span(2), pte));
        }

        error!("No memory");
        Err(Error::Memory)
    }

    fn swap_reserved(&self, size: Size, new_pte: Entry3, pte_a: &Atomic<Entry3>) -> Result<()> {
        let pte = pte_a.swap(new_pte);
        let i = pte.idx();
        let max = (self.pages() - i * Table::span(2)).min(Table::span(2));

        if let Ok(v) = self.entries[i].update(|v| v.unreserve_add(pte, max)) {
            // Add to list if new counter is small enough
            let new_pages = v.pages() + pte.pages();
            if new_pages == max {
                self.empty.push(&self.entries, i);
            } else if new_pages > PTE3_FULL {
                self.partial(size).push(&self.entries, i);
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            return Err(Error::Corruption);
        }
    }

    fn get_small(&self, core: usize, size: Size) -> Result<usize> {
        let local = &self.local[core];
        let start_a = local.start(size);
        let pte_a = local.pte(size);
        let mut start = start_a.load(Ordering::SeqCst);

        if start == usize::MAX {
            warn!("Try reserve first");
            let (s, pte) = self.reserve_dec(size)?;
            pte_a.store(pte);
            start = s
        } else {
            // Incremet or clear (atomic sync with put dec)
            if pte_a.update(|v| v.dec(size)).is_err() {
                warn!("Try reserve next");
                let (s, new_pte) = self.reserve_dec(size)?;
                self.swap_reserved(size, new_pte, pte_a)?;
                start = s;
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
        if let Some(i) = self.empty.pop(&self.entries) {
            if self.entries[i]
                .update(|v| (v.pages() == Table::span(2)).then(Entry3::new_giant))
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
                Entry3::new().with_pages(Table::span(2)),
            )
            .is_ok()
        {
            // Add to empty list
            self.empty.push(&self.entries, i);
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
        if pte.pages() == max {
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
                let new_pages = pte.pages() + Table::span(size as _);
                if pte.pages() <= PTE3_FULL && new_pages > PTE3_FULL {
                    // Try to reserve it for bulk frees
                    if let Ok(pte) = self.entries[i].update(|v| v.reserve_take(size)) {
                        let pte = pte.with_idx(i);
                        warn!("put reserve {i}");
                        self.swap_reserved(size, pte, pte_a)?;
                        local.start(size).store(page, Ordering::SeqCst);
                    } else {
                        // Add to partially free list
                        self.partial(size).push(&self.entries, i);
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
