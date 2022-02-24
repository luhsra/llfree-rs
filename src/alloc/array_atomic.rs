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
pub struct ArrayAtomicAlloc {
    meta: *mut Meta,
    lower: LowerAlloc,
    entries: Vec<Atomic<Entry3>>,

    empty: AStack<Entry3>,
    partial_l1: AStack<Entry3>,
    partial_l0: AStack<Entry3>,
}

unsafe impl Send for ArrayAtomicAlloc {}
unsafe impl Sync for ArrayAtomicAlloc {}

impl Alloc for ArrayAtomicAlloc {
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

    fn put(&self, core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let i = page / Table::span(2);
        let pte = self.entries[i].load();
        if pte.page() {
            self.put_giant(page)
        } else {
            self.put_small(core, page, pte)
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages
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
        for local in self.lower.iter() {
            pages -= local.pte(false).load().free();
            pages -= local.pte(true).load().free();
        }
        pages
    }
}

impl Drop for ArrayAtomicAlloc {
    fn drop(&mut self) {
        let meta = unsafe { &*self.meta };
        meta.active.store(0, Ordering::SeqCst);
    }
}

impl ArrayAtomicAlloc {
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
            self.entries[i] = Atomic::new(Entry3::new().with_free(Table::span(2)));
            self.empty.push(&self.entries, i);
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * Table::span(2)).min(Table::span(2));
        self.entries[pte3_num - 1] = Atomic::new(Entry3::new().with_free(max));

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
            let (pages, size) = self.lower.recover(page, deep)?;
            if size == Size::L2 {
                self.entries[i].store(Entry3::new_giant());
            } else {
                self.entries[i].store(Entry3::new_table(pages, size, false));

                // Add to lists
                if pages == Table::span(2) {
                    self.empty.push(&self.entries, i);
                } else if pages > PTE3_FULL {
                    self.partial(size == Size::L1).push(&self.entries, i);
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

    fn reserve_dec(&self, huge: bool) -> Result<(usize, Entry3)> {
        while let Some(i) = self.partial(huge).pop(&self.entries) {
            warn!("reserve partial {i}");
            match self.entries[i].update(|v| v.reserve_partial(huge, PTE3_FULL)) {
                Ok(pte) => {
                    let pte = pte.dec(huge).unwrap().with_idx(i);
                    return Ok((i * Table::span(2), pte));
                }
                Err(pte) => {
                    // Skip empty entries
                    if !pte.reserved() && pte.free() == Table::span(2) {
                        self.empty.push(&self.entries, i);
                    }
                }
            }
        }

        while let Some(i) = self.empty.pop(&self.entries) {
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
                self.empty.push(&self.entries, i);
            } else if new_pages > PTE3_FULL {
                self.partial(huge).push(&self.entries, i);
            }
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    fn get_small(&self, core: usize, huge: bool) -> Result<usize> {
        let start_a = self.lower[core].start(huge);
        let pte_a = self.lower[core].pte(huge);
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
            self.lower.get_huge(start)?
        } else {
            self.lower.get(core, start)?
        };
        start_a.store(page, Ordering::SeqCst);
        Ok(page)
    }

    fn get_giant(&self) -> Result<usize> {
        if let Some(i) = self.empty.pop(&self.entries) {
            if self.entries[i]
                .update(|v| (v.free() == Table::span(2)).then(Entry3::new_giant))
                .is_ok()
            {
                self.lower.persist(i * Table::span(2));
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

    fn put_giant(&self, page: usize) -> Result<()> {
        let i = page / Table::span(2);
        if page % Table::span(2) != 0 {
            error!("Invalid align {page:x}");
            return Err(Error::Address);
        }
        self.lower.clear_giant(page);

        if self.entries[i]
            .compare_exchange(Entry3::new_giant(), Entry3::new().with_free(Table::span(2)))
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
        if pte.free() == max {
            error!("Not allocated {page} (i{})", page / Table::span(2));
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let huge = self.lower.put(page)?;

        // Try decrement own pte first
        let pte_a = self.lower[core].pte(huge);
        if pte_a.update(|v| v.inc_idx(huge, i, max)).is_ok() {
            return Ok(());
        }

        // Subtree not owned by us
        if let Ok(pte) = self.entries[i].update(|v| v.inc(huge, max)) {
            if !pte.reserved() {
                let new_pages = pte.free() + Table::span(huge as _);
                if pte.free() <= PTE3_FULL && new_pages > PTE3_FULL {
                    // Try to reserve it for bulk frees
                    if let Ok(pte) = self.entries[i].update(|v| v.reserve(huge)) {
                        let pte = pte.with_idx(i);
                        warn!("put reserve {i}");
                        self.swap_reserved(huge, pte, pte_a)?;
                        self.lower[core].start(huge).store(page, Ordering::SeqCst);
                    } else {
                        // Add to partially free list
                        self.partial(huge).push(&self.entries, i);
                    }
                }
            }
            Ok(())
        } else {
            error!("Corruption l3 i{i} p=-{huge:?}");
            Err(Error::Corruption)
        }
    }
}
