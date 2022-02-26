use std::ops::Index;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::Entry3;
use crate::lower_alloc::LowerAlloc;
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
pub struct ArrayUnalignedAlloc {
    meta: *mut Meta,
    lower: LowerAlloc,
    entries: Vec<Atomic<Entry3>>,

    empty: AStack<Entry3>,
    partial_l1: AStack<Entry3>,
    partial_l0: AStack<Entry3>,
}

impl Index<usize> for ArrayUnalignedAlloc {
    type Output = Atomic<Entry3>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

unsafe impl Send for ArrayUnalignedAlloc {}
unsafe impl Sync for ArrayUnalignedAlloc {}

impl Alloc for ArrayUnalignedAlloc {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < MIN_PAGES * cores {
            error!("memory {} < {}", memory.len(), MIN_PAGES * cores);
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        let (memory, rem) = memory.split_at_mut((memory.len() - 1).min(MAX_PAGES));
        let meta = rem[0].cast_mut::<Meta>();
        self.meta = meta;

        // Create lower allocator
        self.lower = LowerAlloc::new(cores, memory);

        // Array with all pte3
        let pte3_num = Table::num_pts(2, self.lower.pages);
        self.entries = Vec::with_capacity(pte3_num);
        self.entries
            .resize_with(pte3_num, || Atomic::new(Entry3::new()));

        self.empty = AStack::new();
        self.partial_l0 = AStack::new();
        self.partial_l1 = AStack::new();

        warn!("init");
        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            let pages = self.recover(deep)?;
            warn!("Recovered pages {pages}");
        } else {
            warn!("Setup allocator state p={}", self.pages());
            self.setup();

            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        meta.active.store(1, Ordering::SeqCst);
        Ok(())
    }

    fn get(&self, core: usize, size: Size) -> Result<u64> {
        match size {
            Size::L2 => self.get_giant(),
            _ => self.get_small(core, size == Size::L1),
        }
        .map(|p| unsafe { self.lower.memory().start.add(p as _) } as u64)
    }

    fn put(&self, _core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let i = page / Table::span(2);
        let pte3 = self[i].load();
        if pte3.page() {
            self.put_giant(page)
        } else {
            self.put_small(page, pte3)
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Table::num_pts(2, self.pages()) {
            let pte = self[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.free();
        }
        pages
    }
}

impl Drop for ArrayUnalignedAlloc {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}

impl ArrayUnalignedAlloc {
    #[cold]
    pub fn new() -> Self {
        Self {
            meta: null_mut(),
            lower: LowerAlloc::default(),
            entries: Vec::new(),
            empty: AStack::new(),
            partial_l1: AStack::new(),
            partial_l0: AStack::new(),
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages
    }

    #[cold]
    fn setup(&mut self) {
        self.lower.clear();

        // Add all entries to the empty list
        let pte3_num = Table::num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            self[i].store(Entry3::new().with_free(Table::span(2)));
            self.empty.push(self, i);
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * Table::span(2)).min(Table::span(2));
        self[pte3_num - 1].store(Entry3::new().with_free(max));

        if max == Table::span(2) {
            self.empty.push(self, pte3_num - 1);
        } else if max > PTE3_FULL {
            self.partial_l0.push(self, pte3_num - 1);
        }
    }

    #[cold]
    fn recover(&self, deep: bool) -> Result<usize> {
        if deep {
            error!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..Table::num_pts(2, self.pages()) {
            let page = i * Table::span(2);
            let (pages, size) = self.lower.recover(page, deep)?;
            if size == Size::L2 {
                self[i].store(Entry3::new_giant());
            } else {
                self[i].store(Entry3::new_table(pages, size, false));

                // Add to lists
                if pages == Table::span(2) {
                    self.empty.push(self, i);
                } else if pages > PTE3_FULL {
                    self.partial(size == Size::L1).push(self, i);
                }
            }
            total += pages;
        }
        Ok(total)
    }

    fn partial(&self, huge: bool) -> &AStack<Entry3> {
        if huge {
            &self.partial_l1
        } else {
            &self.partial_l0
        }
    }

    fn reserve_dec(&self, huge: bool) -> Result<usize> {
        while let Some(i) = self.partial(huge).pop(self) {
            // Skip empty entries
            if self[i].load().free() < Table::span(2) {
                warn!("reserve partial {i}");
                self[i].update(|v| v.dec(huge)).unwrap();
                return Ok(i * Table::span(2));
            } else {
                self.empty.push(self, i);
            }
        }

        if let Some(i) = self.empty.pop(self) {
            warn!("reserve empty {i}");
            self[i].update(|v| v.dec(huge)).unwrap();
            Ok(i * Table::span(2))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    fn get_small(&self, core: usize, huge: bool) -> Result<usize> {
        let start_a = self.lower[core].start(huge);
        let mut start = start_a.load(Ordering::Relaxed);

        if start == usize::MAX {
            warn!("Try reserve first");
            start = self.reserve_dec(huge)?;
        } else {
            let i = start / Table::span(2);
            if self[i].update(|v| v.dec(huge)).is_err() {
                start = self.reserve_dec(huge)?;
                if self[i].update(Entry3::unreserve).is_err() {
                    panic!("Unreserve failed")
                }
            }
        }

        let page = if huge {
            self.lower.get_huge(start)?
        } else {
            self.lower.get(core, start)?
        };
        start_a.store(page, Ordering::Relaxed);
        Ok(page)
    }

    fn get_giant(&self) -> Result<usize> {
        if let Some(i) = self.empty.pop(self) {
            match self[i].update(|v| (v.free() == Table::span(2)).then(Entry3::new_giant)) {
                Ok(_) => {
                    self.lower.persist(i * Table::span(2));
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

    fn put_giant(&self, page: usize) -> Result<()> {
        let i = page / Table::span(2);
        if page % Table::span(2) != 0 {
            error!("Invalid align {page:x}");
            return Err(Error::Address);
        }
        self.lower.clear_giant(page);

        match self[i].compare_exchange(Entry3::new_giant(), Entry3::new().with_free(Table::span(2)))
        {
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

    fn put_small(&self, page: usize, pte3: Entry3) -> Result<()> {
        let max = self
            .pages()
            .saturating_sub(Table::round(2, page))
            .min(Table::span(2));
        if pte3.free() == max {
            error!("Not allocated {page}");
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let size = self.lower.put(page)?;
        if let Ok(pte3) = self[i].update(|v| v.inc(size, max)) {
            if !pte3.reserved() {
                let new_pages = pte3.free() + Table::span(size as _);
                if pte3.free() <= PTE3_FULL && new_pages > PTE3_FULL {
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
