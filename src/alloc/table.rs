//! Simple reduced non-volatile memory allocator.
use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, info, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::{Change, Entry, Entry3};
use crate::leaf_alloc::{LeafAllocator, Leafs};
use crate::table::Table;
use crate::util::Page;

const PTE3_FULL: usize = 4 * Table::span(1);

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// Volatile shared metadata
#[repr(align(64))]
pub struct TableAlloc {
    memory: Range<*const Page>,
    meta: *mut Meta,
    local: Vec<LeafAllocator<Self>>,
    tables: Vec<Table<Entry>>,
}

const INITIALIZING: *mut TableAlloc = usize::MAX as _;
static mut SHARED: AtomicPtr<TableAlloc> = AtomicPtr::new(null_mut());

impl Leafs for TableAlloc {
    fn leafs<'a>() -> &'a [LeafAllocator<Self>] {
        &Self::instance().local
    }
}

impl Alloc for TableAlloc {
    #[cold]
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

        if meta.pages.load(Ordering::SeqCst) == alloc.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", alloc.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Allocator unexpectedly terminated");
            }
            let pages = alloc.recover_rec(Table::LAYERS, 0, deep)?;
            warn!("Recovered {pages:?}");
        } else {
            warn!("Setup allocator state p={}", alloc.pages());
            alloc.local[0].clear();

            alloc.setup_rec(Table::LAYERS, 0);

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
        // Start at the reserved memory chunk for this thread
        let page = if size == Size::L2 {
            loop {
                match self.get_giant(core, Table::LAYERS, 0) {
                    Ok(page) => break page,
                    Err(Error::CAS) => warn!("CAS: retry alloc"),
                    Err(e) => return Err(e),
                }
            }
        } else {
            self.get_small(core, size)?
        };

        Ok(unsafe { self.memory.start.add(page as _) } as u64)
    }

    fn put(&self, core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.memory.contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.memory.start) } as usize;

        loop {
            match self.put_rec(core, page) {
                Ok(_) => return Ok(()),
                Err(Error::CAS) => warn!("CAS: retry free"),
                Err(e) => return Err(e),
            }
        }
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = self.allocated_pages_rec(Table::LAYERS, 0);
        // Pages allocated in reserved subtrees
        for local in &self.local {
            pages += local.pte(Size::L0).load().pages();
            pages += local.pte(Size::L1).load().pages();
        }
        self.pages() - pages
    }
}

impl TableAlloc {
    #[cold]
    fn new(cores: usize, memory: &mut [Page]) -> Result<Self> {
        // Last frame is reserved for metadata
        let mut pages = (memory.len() - 1).min(MAX_PAGES);
        let (memory, rem) = memory.split_at_mut(pages);
        let meta = rem[0].cast::<Meta>();

        // level 2 tables are stored at the end of the NVM
        pages -= Table::num_pts(2, pages);
        let (memory, _pt2) = memory.split_at_mut(pages);

        let mut num_pt = 0;
        for layer in 3..=Table::LAYERS {
            num_pt += Table::num_pts(layer, pages);
        }
        let tables = vec![Table::empty(); num_pt];
        let local = vec![LeafAllocator::new(memory.as_ptr() as _, pages); cores];

        Ok(Self {
            memory: memory.as_ptr_range(),
            meta,
            local,
            tables,
        })
    }

    fn allocated_pages_rec(&self, layer: usize, start: usize) -> usize {
        let mut pages = 0;
        if layer == 3 {
            let pt3 = self.pt3(start);
            for i in Table::range(3, 0..self.pages()) {
                let pte3 = pt3.get(i);
                // warn!(" - {i:>3}: {pte3:?}");
                pages += pte3.pages();
            }
        } else {
            let pt = self.pt(layer, start);
            for i in Table::range(layer, 0..self.pages()) {
                let pte = pt.get(i);
                // warn!("{i:>3}: {pte:?}");
                if pte.empty() > 0 || pte.partial_l0() > 0 || pte.partial_l1() > 0 {
                    pages += self.allocated_pages_rec(layer - 1, Table::page(layer, start, i));
                }
            }
        }
        pages
    }

    fn pages(&self) -> usize {
        (self.memory.end as usize - self.memory.start as usize) / Page::SIZE
    }

    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// ```
    fn pt(&self, layer: usize, page: usize) -> &Table<Entry> {
        assert!((4..=Table::LAYERS).contains(&layer));

        let i = page >> (Table::LEN_BITS * layer);
        let offset: usize = (layer..Table::LAYERS)
            .map(|i| Table::num_pts(i, self.pages()))
            .sum();
        &self.tables[offset + i]
    }

    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// ```
    fn pt3(&self, page: usize) -> &Table<Entry3> {
        let i = page >> (Table::LEN_BITS * 3);
        let offset: usize = (3..Table::LAYERS)
            .map(|i| Table::num_pts(i, self.pages()))
            .sum();
        unsafe { &*(&self.tables[offset + i] as *const _ as *const Table<Entry3>) }
    }

    #[cold]
    fn setup_rec(&self, layer: usize, start: usize) {
        for i in 0..Table::LEN {
            let page = Table::page(layer, start, i);
            if page > self.pages() {
                break;
            }

            if layer == 3 {
                let pt = self.pt3(page);
                let max = (self.pages() - page).min(Table::span(2));
                pt.set(i, Entry3::new().with_pages(max));
            } else {
                let pt = self.pt(layer, page);
                let max = (self.pages() - page).min(Table::span(layer));
                let empty = max / Table::span(2);

                if max - empty * Table::span(2) > PTE3_FULL {
                    pt.set(i, Entry::new().with_empty(empty).with_partial_l0(1));
                } else {
                    pt.set(i, Entry::new().with_empty(empty));
                }
                self.setup_rec(layer - 1, page);
            }
        }
    }

    #[cold]
    fn recover_rec(&self, layer: usize, start: usize, deep: bool) -> Result<(usize, usize, usize)> {
        let mut empty = 0;
        let mut partial_l0 = 0;
        let mut partial_l1 = 0;
        let pt = self.pt(layer, start);
        for i in 0..Table::LEN {
            let page = Table::page(layer, start, i);
            if page > self.pages() {
                break;
            }

            let (c_empty, c_pl0, c_pl1) = if layer - 1 == 3 {
                self.recover_l3(page, deep)?
            } else {
                self.recover_rec(layer - 1, page, deep)?
            };

            pt.set(
                i,
                Entry::new()
                    .with_empty(c_empty)
                    .with_partial_l0(c_pl0)
                    .with_partial_l1(c_pl1),
            );
            empty += c_empty;
            partial_l0 += c_pl0;
            partial_l1 += c_pl1;
        }

        Ok((empty, partial_l0, partial_l1))
    }

    fn recover_l3(&self, start: usize, deep: bool) -> Result<(usize, usize, usize)> {
        let mut empty = 0;
        let mut partial_l0 = 0;
        let mut partial_l1 = 0;
        let pt = self.pt3(start);

        for i in 0..Table::LEN {
            let page = Table::page(3, start, i);
            if page > self.pages() {
                break;
            }

            let (pages, size) = self.local[0].recover(page, deep)?;
            if size == Size::L2 {
                pt.set(i, Entry3::new_giant());
            } else if pages > 0 {
                pt.set(i, Entry3::new_table(pages, size, false));
                if pages == Table::span(2) {
                    empty += 1;
                } else if pages > PTE3_FULL {
                    if size == Size::L0 {
                        partial_l0 += 1;
                    } else {
                        partial_l1 += 1;
                    }
                }
            } else {
                pt.set(i, Entry3::new());
            }
        }
        Ok((empty, partial_l0, partial_l1))
    }

    #[cold]
    fn reserve_rec_empty(&self, layer: usize, start: usize, size: Size) -> Result<(usize, Entry3)> {
        if layer == 3 {
            return self.reserve_l3_empty(start, size);
        }

        for page in Table::iterate(layer, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(layer, page);
            let pt = self.pt(layer, start);
            if pt.update(i, |v| v.dec_empty()).is_ok() {
                if let Ok(result) = self.reserve_rec_empty(layer - 1, page, size) {
                    return Ok(result);
                }
            }
        }
        Err(Error::Memory)
    }

    fn reserve_l3_empty(&self, start: usize, size: Size) -> Result<(usize, Entry3)> {
        for page in Table::iterate(3, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(3, page);
            let pt = self.pt3(start);
            if let Ok(pte) = pt.update(i, |v| v.reserve_empty(size)) {
                return Ok((page, pte.dec(size).unwrap().with_idx(page / Table::span(2))));
            }
        }
        Err(Error::Memory)
    }

    #[cold]
    fn reserve_rec_partial(
        &self,
        layer: usize,
        start: usize,
        size: Size,
    ) -> Result<(usize, Entry3)> {
        if layer == 3 {
            return self.reserve_l3_partial(start, size);
        }

        for page in Table::iterate(layer, start) {
            if page > self.pages() {
                continue;
            }

            let i = Table::idx(layer, page);
            let pt = self.pt(layer, start);
            if pt.update(i, |v| v.reserve_partial(size)).is_ok() {
                if let Ok(result) = self.reserve_rec_partial(layer - 1, page, size) {
                    return Ok(result);
                }
            }
        }
        Err(Error::Memory)
    }

    fn reserve_l3_partial(&self, start: usize, size: Size) -> Result<(usize, Entry3)> {
        for page in Table::iterate(3, start) {
            if page > self.pages() {
                continue;
            }

            let i = Table::idx(3, page);
            let pt = self.pt3(start);
            if let Ok(pte) = pt.update(i, |v| v.reserve_partial(size)) {
                return Ok((page, pte.dec(size).unwrap().with_idx(page / Table::span(2))));
            }
        }
        Err(Error::Memory)
    }

    #[cold]
    fn update_parents(&self, page: usize, change: Change) -> Result<()> {
        for layer in 4..=Table::LAYERS {
            let pt = self.pt(layer, page);
            let i = Table::idx(layer, page);
            if let Err(pte) = pt.update(i, |v| v.change(change)) {
                error!("Update failed l{layer} i{i} {pte:?} {change:?}");
                return Err(Error::Corruption);
            }
        }
        Ok(())
    }

    #[cold]
    fn unreserve(&self, page: usize, old: Entry3) -> Result<()> {
        let pt = self.pt3(page);
        let i = Table::idx(3, page);
        let max = (self.pages() - Table::round(2, page)).min(Table::span(2));
        match pt.update(i, |v| v.unreserve_add(old, max)) {
            Err(pte) => {
                error!("Unreserve failed {pte:?}");
                Err(Error::Corruption)
            }
            Ok(pte) => {
                // Update parents
                let new_pages = old.pages() + pte.pages();
                if new_pages == Table::span(2) {
                    self.update_parents(page, Change::IncEmpty)
                } else if new_pages > PTE3_FULL {
                    let change = if old.size() == Some(Size::L1) {
                        Change::IncPartialL0
                    } else {
                        Change::IncPartialL1
                    };
                    self.update_parents(page, change)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn get_giant(&self, core: usize, layer: usize, start: usize) -> Result<usize> {
        if layer == 3 {
            return self.get_giant_l3(core, start);
        }

        let pt = self.pt(layer, start);
        for i in 0..Table::LEN {
            let page = Table::page(layer, start, i);
            if page > self.pages() {
                break;
            }

            if let Err(pte) = pt.update(i, |pte| pte.dec_empty()) {
                warn!("giant update failed {pte:?}");
                continue;
            }

            return self.get_giant(core, layer - 1, page);
        }

        error!("Nothing found l{layer} s={start}");
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn get_giant_l3(&self, core: usize, start: usize) -> Result<usize> {
        let pt = self.pt3(start);

        for i in 0..Table::LEN {
            if pt
                .cas(
                    i,
                    Entry3::new().with_pages(Table::span(2)),
                    Entry3::new_giant(),
                )
                .is_ok()
            {
                let page = Table::page(3, start, i);
                info!("allocated l3 i{i} p={page} s={start}");
                self.local[core].persist(page);
                return Ok(page);
            }
        }
        error!("Nothing found l3 s={start}");
        Err(Error::Memory)
    }

    fn get_small(&self, core: usize, size: Size) -> Result<usize> {
        let leaf = &self.local[core];
        let pte_a = leaf.pte(size);
        let start_a = leaf.start(size);
        let mut start = start_a.load(Ordering::SeqCst);

        if start == usize::MAX {
            warn!("try reserve first");
            let (s, pte) = self
                .reserve_rec_partial(Table::LAYERS, 0, size)
                .or_else(|_| self.reserve_rec_empty(Table::LAYERS, 0, size))?;
            warn!("reserved {}", s / Table::span(2));
            start = s;
            pte_a.store(pte);
        } else {
            // Increment or reserve new if full
            if pte_a.update(|v| v.dec(size)).is_err() {
                warn!("try reserve next");
                let (s, pte) = self
                    .reserve_rec_partial(Table::LAYERS, start, size)
                    .or_else(|_| self.reserve_rec_empty(Table::LAYERS, start, size))?;
                warn!("reserved {}", s / Table::span(2));

                let old = pte_a.swap(pte);
                self.unreserve(start, old)?;

                start = s;
            }
        }

        let page = if size == Size::L0 {
            leaf.get(start)?
        } else {
            leaf.get_huge(start)?
        };
        start_a.store(page, Ordering::SeqCst);
        Ok(page)
    }

    fn put_rec(&self, core: usize, page: usize) -> Result<()> {
        // Check parents
        for layer in (4..=Table::LAYERS).rev() {
            let pt = self.pt(layer, page);
            let i = Table::idx(layer, page);
            let pte = pt.get(i);

            let max = (Table::num_pts(2, self.pages()) - i * Table::span(layer - 2))
                .min(Table::span(layer - 2));
            if pte.empty() == max {
                error!("No table found l{layer} {pte:?}");
                return Err(Error::Address);
            }
        }

        let pt = self.pt3(page);
        let i = Table::idx(3, page);
        let pte = pt.get(i);

        if pte.size() == Some(Size::L2) {
            warn!("free giant l3 i{i}");
            return self.put_giant(core, page);
        }

        let max = (self.pages() - Table::round(2, page)).min(Table::span(2));
        if pte.size().is_none() || pte.pages() == max {
            error!("Invalid address l3 i{i}");
            return Err(Error::Address);
        }

        let leaf = &self.local[core];

        let size = leaf.put(page)?;
        let idx = page / Table::span(2);
        if leaf.pte(size).update(|v| v.inc_idx(size, idx, max)).is_ok() {
            return Ok(());
        }

        match pt.update(i, |v| v.inc(size, max)) {
            Err(pte) => {
                error!("Corruption l3 i{i} {size:?} {pte:?}");
                Err(Error::Corruption)
            }
            Ok(pte) => {
                if pte.reserved() {
                    Ok(())
                } else if pte.pages() + Table::span(size as _) == Table::span(2) {
                    let change = if size == Size::L0 {
                        Change::DecPartialL0
                    } else {
                        Change::DecPartialL1
                    };
                    self.update_parents(page, change)
                } else if pte.pages() < PTE3_FULL
                    && pte.pages() + Table::span(size as _) >= PTE3_FULL
                {
                    let change = if size == Size::L0 {
                        Change::IncPartialL0
                    } else {
                        Change::IncPartialL1
                    };
                    self.update_parents(page, change)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn put_giant(&self, core: usize, page: usize) -> Result<()> {
        if (page % Table::span(Size::L2 as _)) != 0 {
            error!(
                "Invalid alignment p={page:x} a={:x}",
                Table::span(Size::L2 as _)
            );
            return Err(Error::Address);
        }

        // Clear pt1's & remove pt2 flag
        self.local[core].clear_giant(page);

        let pt = self.pt3(page);
        let i = Table::idx(3, page);

        info!("free l3 i{}", i);
        match pt.cas(
            i,
            Entry3::new_giant(),
            Entry3::new().with_pages(Table::span(2)),
        ) {
            Ok(_) => self.update_parents(page, Change::IncEmpty),
            _ => {
                error!("Invalid {page}");
                Err(Error::Address)
            }
        }
    }
}
