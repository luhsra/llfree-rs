use std::ops::{Index, Range};
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
pub struct ArrayAlignedAlloc {
    memory: Range<*const Page>,
    meta: *mut Meta,
    local: Vec<LeafAllocator<Self>>,
    entries: Vec<Aligned>,

    empty: AStack<Entry3>,
    partial_l1: AStack<Entry3>,
    partial_l0: AStack<Entry3>,
}

#[repr(align(64))]
struct Aligned(Atomic<Entry3>);

const INITIALIZING: *mut ArrayAlignedAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<ArrayAlignedAlloc> = AtomicPtr::new(null_mut());

impl Leafs for ArrayAlignedAlloc {
    fn leafs<'a>() -> &'a [LeafAllocator<Self>] {
        &Self::instance().local
    }
}

impl Index<usize> for ArrayAlignedAlloc {
    type Output = Atomic<Entry3>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index].0
    }
}

impl Alloc for ArrayAlignedAlloc {
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
        let pte3 = self[i].load();
        match pte3.size() {
            Some(Size::L2) => self.put_giant(core, page),
            Some(_) => self.put_small(core, page, pte3),
            None => Err(Error::Memory),
        }
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Table::num_pts(2, self.pages()) {
            let pte = self[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.pages();
        }
        pages
    }
}

impl ArrayAlignedAlloc {
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
        entries.resize_with(pte3_num, || Aligned(Atomic::new(Entry3::new())));

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
            self.entries[i] = Aligned(Atomic::new(Entry3::new().with_pages(Table::span(2))));
            self.empty.push(self, i);
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * Table::span(2)).min(Table::span(2));
        self.entries[pte3_num - 1] = Aligned(Atomic::new(Entry3::new().with_pages(max)));

        if max == Table::span(2) {
            self.empty.push(self, pte3_num - 1);
        } else if max > PTE3_FULL {
            self.partial_l0.push(self, pte3_num - 1);
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
                self[i].store(Entry3::new_giant());
            } else {
                self.entries[i]
                    .0
                    .store(Entry3::new_table(pages, size, false));

                // Add to lists
                if pages == Table::span(2) {
                    self.empty.push(self, i);
                } else if pages > PTE3_FULL {
                    self.partial(size).push(self, i);
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

    fn reserve_dec(&self, size: Size) -> Result<usize> {
        while let Some(i) = self.partial(size).pop(self) {
            // Skip empty entries
            if self[i].load().pages() < Table::span(2) {
                warn!("reserve partial {i}");
                self[i].update(|v| v.dec(size)).unwrap();
                return Ok(i * Table::span(2));
            } else {
                self.empty.push(self, i);
            }
        }

        if let Some(i) = self.empty.pop(self) {
            warn!("reserve empty {i}");
            self[i].update(|v| v.dec(size)).unwrap();
            Ok(i * Table::span(2))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn get_small(&self, core: usize, size: Size) -> Result<usize> {
        let local = &self.local[core];
        let start_a = local.start(size);
        let mut start = start_a.load(Ordering::Relaxed);

        if start == usize::MAX {
            warn!("Try reserve first");
            start = self.reserve_dec(size)?;
        } else {
            let i = start / Table::span(2);
            if self[i].update(|v| v.dec(size)).is_err() {
                start = self.reserve_dec(size)?;
                if self[i].update(Entry3::unreserve).is_err() {
                    panic!("Unreserve failed")
                }
            }
        }

        let page = match size {
            Size::L0 => local.get(start)?,
            _ => local.get_huge(start)?,
        };
        start_a.store(page, Ordering::Relaxed);
        Ok(page)
    }

    fn get_giant(&self, core: usize) -> Result<usize> {
        if let Some(i) = self.empty.pop(self) {
            match self[i].update(|v| (v.pages() == Table::span(2)).then(Entry3::new_giant)) {
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

        match self[i].compare_exchange(
            Entry3::new_giant(),
            Entry3::new().with_pages(Table::span(2)),
        ) {
            Ok(_) => {
                // Add to empty list
                self.empty.push(self, i);
                Ok(())
            }
            Err(_) => {
                error!("CAS invalid i{i}");
                Err(Error::Address)
            }
        }
    }

    fn put_small(&self, core: usize, page: usize, pte3: Entry3) -> Result<()> {
        let max = self
            .pages()
            .saturating_sub(Table::round(2, page))
            .min(Table::span(2));
        if pte3.pages() == max {
            error!("Not allocated {page}");
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let local = &self.local[core];
        let size = local.put(page)?;
        if let Ok(pte3) = self[i].update(|v| v.inc(size, max)) {
            if !pte3.reserved() {
                let new_pages = pte3.pages() + Table::span(size as _);
                if pte3.pages() <= PTE3_FULL && new_pages > PTE3_FULL {
                    // Add back to partial
                    self.partial(size).push(self, i);
                }
            }
            Ok(())
        } else {
            error!("Corruption l3 i{i} p=-{size:?}");
            Err(Error::Corruption)
        }
    }
}
