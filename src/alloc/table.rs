//! Simple reduced non-volatile memory allocator.
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::{Change, Entry, Entry3};
use crate::lower_alloc::LowerAlloc;
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
pub struct TableAlloc {
    meta: *mut Meta,
    lower: LowerAlloc,
    tables: Vec<Page>,
}

unsafe impl Send for TableAlloc {}
unsafe impl Sync for TableAlloc {}

impl Alloc for TableAlloc {
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

        let mut num_pt = 0;
        for layer in 3..=Table::LAYERS {
            num_pt += Table::num_pts(layer, self.lower.pages);
        }

        self.tables = vec![Page::new(); num_pt];

        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Allocator unexpectedly terminated");
            }
            let pages = self.recover_rec(Table::LAYERS, 0, deep)?;
            warn!("Recovered {pages:?}");
        } else {
            warn!("Setup allocator state p={}", self.pages());
            self.lower.clear();

            self.setup_rec(Table::LAYERS, 0);

            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        meta.active.store(1, Ordering::SeqCst);
        Ok(())
    }

    #[cold]
    fn destroy(&mut self) {
        let meta = unsafe { &*self.meta };
        meta.active.store(0, Ordering::SeqCst);
        meta.magic.store(0, Ordering::SeqCst);
    }

    fn get(&self, core: usize, size: Size) -> Result<u64> {
        // Start at the reserved memory chunk for this thread
        let page = match size {
            Size::L2 => self.get_giant(Table::LAYERS, 0)?,
            _ => self.get_small(core, size == Size::L1)?,
        };

        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
    }

    fn put(&self, core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        self.put_rec(core, page)
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        let mut pages = self.allocated_pages_rec(Table::LAYERS, 0);
        // Pages allocated in reserved subtrees
        for (_t, local) in self.lower.iter().enumerate() {
            let pte = local.pte(false).load();
            // warn!("L {t:>2}: L0 {pte:?}");
            pages += pte.free();
            let pte = local.pte(true).load();
            // warn!("L {t:>2}: L1 {pte:?}");
            pages += pte.free();
        }
        self.pages() - pages
    }
}

impl Drop for TableAlloc {
    fn drop(&mut self) {
        warn!("drop");
        let meta = unsafe { &*self.meta };
        meta.active.store(0, Ordering::SeqCst);
    }
}

impl TableAlloc {
    #[cold]
    pub fn new() -> Self {
        Self {
            meta: null_mut(),
            lower: LowerAlloc::default(),
            tables: Vec::new(),
        }
    }

    fn allocated_pages_rec(&self, layer: usize, start: usize) -> usize {
        let mut pages = 0;
        if layer == 3 {
            let pt3 = self.pt3(start);
            for i in Table::range(3, 0..self.pages()) {
                let pte3 = pt3.get(i);
                // warn!(" - {i:>3}: {pte3:?}");
                pages += pte3.free();
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
        self.lower.pages
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
        self.tables[offset + i].cast()
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
        self.tables[offset + i].cast()
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
                pt.set(i, Entry3::new().with_free(max));
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

            let (pages, size) = self.lower.recover(page, deep)?;
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
    fn reserve_rec_empty(&self, layer: usize, start: usize, huge: bool) -> Result<Entry3> {
        if layer == 3 {
            return self.reserve_l3_empty(start, huge);
        }

        for page in Table::iterate(layer, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(layer, page);
            let pt = self.pt(layer, start);
            if pt.update(i, |v| v.dec_empty()).is_ok() {
                return self.reserve_rec_empty(layer - 1, page, huge);
            }
        }
        error!("No memory l{layer}");
        Err(Error::Memory)
    }

    fn reserve_l3_empty(&self, start: usize, huge: bool) -> Result<Entry3> {
        for page in Table::iterate(3, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(3, page);
            let pt = self.pt3(start);
            if let Ok(pte) = pt.update(i, |v| v.reserve_empty(huge)) {
                return Ok(pte.dec(huge).unwrap().with_idx(page / Table::span(2)));
            }
        }
        error!("No memory l3");
        Err(Error::Memory)
    }

    #[cold]
    fn reserve_rec_partial(&self, layer: usize, start: usize, huge: bool) -> Result<Entry3> {
        if layer == 3 {
            return self.reserve_l3_partial(start, huge);
        }

        for page in Table::iterate(layer, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(layer, page);
            let pt = self.pt(layer, start);
            if pt.update(i, |v| v.dec_partial(huge)).is_ok() {
                return self.reserve_rec_partial(layer - 1, page, huge);
            }
        }
        Err(Error::Memory)
    }

    fn reserve_l3_partial(&self, start: usize, huge: bool) -> Result<Entry3> {
        for page in Table::iterate(3, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(3, page);
            let pt = self.pt3(start);
            if let Ok(pte) = pt.update(i, |v| v.reserve_partial(huge, PTE3_FULL)) {
                return Ok(pte.dec(huge).unwrap().with_idx(page / Table::span(2)));
            }
        }
        Err(Error::Memory)
    }

    #[cold]
    fn update_parents(&self, page: usize, change: Change) -> Result<()> {
        for layer in 4..=Table::LAYERS {
            let pt = self.pt(layer, page);
            let i = Table::idx(layer, page);
            if pt.update(i, |v| v.change(change)).is_err() {
                error!("Update failed l{layer} i{i} {change:?}");
                return Err(Error::Corruption);
            }
        }
        Ok(())
    }

    #[cold]
    fn swap_reserved(&self, huge: bool, new_pte: Entry3, pte_a: &Atomic<Entry3>) -> Result<()> {
        let old = pte_a.swap(new_pte);
        let start = old.idx() * Table::span(2);
        let i = Table::idx(3, start);
        let pt = self.pt3(start);
        let max = (self.pages() - start).min(Table::span(2));

        if let Ok(pte) = pt.update(i, |v| v.unreserve_add(huge, old, max)) {
            // Update parents
            let new_pages = old.free() + pte.free();
            if new_pages == Table::span(2) {
                self.update_parents(start, Change::IncEmpty)
            } else if new_pages > PTE3_FULL {
                self.update_parents(start, Change::p_inc(huge))
            } else {
                Ok(())
            }
        } else {
            error!("Unreserve failed {i}");
            Err(Error::Corruption)
        }
    }

    fn get_giant(&self, layer: usize, start: usize) -> Result<usize> {
        if layer == 3 {
            return self.get_giant_l3(start);
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

            return self.get_giant(layer - 1, page);
        }

        error!("Nothing found l{layer} s={start}");
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn get_giant_l3(&self, start: usize) -> Result<usize> {
        let pt = self.pt3(start);

        for i in 0..Table::LEN {
            if pt
                .cas(
                    i,
                    Entry3::new().with_free(Table::span(2)),
                    Entry3::new_giant(),
                )
                .is_ok()
            {
                let page = Table::page(3, start, i);
                info!("allocated l3 i{i} p={page} s={start}");
                self.lower.persist(page);
                return Ok(page);
            }
        }
        error!("Nothing found l3 s={start}");
        Err(Error::Memory)
    }

    fn get_small(&self, core: usize, huge: bool) -> Result<usize> {
        let pte_a = self.lower[core].pte(huge);
        let start_a = self.lower[core].start(huge);
        let mut start = start_a.load(Ordering::SeqCst);

        if start == usize::MAX {
            warn!("try reserve first");
            let pte = self
                .reserve_rec_partial(Table::LAYERS, 0, huge)
                .or_else(|_| self.reserve_rec_empty(Table::LAYERS, 0, huge))?;
            warn!("reserved {}", pte.idx());
            pte_a.store(pte);
            start = pte.idx() * Table::span(2);
        } else {
            // Increment or reserve new if full
            if pte_a.update(|v| v.dec(huge)).is_err() {
                warn!("try reserve next");
                let pte = self
                    .reserve_rec_partial(Table::LAYERS, start, huge)
                    .or_else(|_| self.reserve_rec_empty(Table::LAYERS, start, huge))?;
                warn!("reserved {}", pte.idx());
                self.swap_reserved(huge, pte, pte_a)?;
                start = pte.idx() * Table::span(2);
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

    fn put_rec(&self, core: usize, page: usize) -> Result<()> {
        let pt = self.pt3(page);
        let i = Table::idx(3, page);
        let pte = pt.get(i);

        if pte.page() {
            warn!("free giant l3 i{i}");
            return self.put_giant(page);
        }

        let max = (self.pages() - Table::round(2, page)).min(Table::span(2));
        if pte.free() == max {
            error!("Invalid address l3 i{i}");
            return Err(Error::Address);
        }

        let size = self.lower.put(page)?;
        let idx = page / Table::span(2);
        let pte_a = self.lower[core].pte(size);
        if pte_a.update(|v| v.inc_idx(size, idx, max)).is_ok() {
            return Ok(());
        }

        if let Ok(pte) = pt.update(i, |v| v.inc(size, max)) {
            let new_pages = pte.free() + Table::span(size as _);
            if pte.reserved() {
                Ok(())
            } else if new_pages == Table::span(2) {
                self.update_parents(page, Change::p_dec(size))
            } else if pte.free() <= PTE3_FULL && new_pages > PTE3_FULL {
                // reserve for bulk put
                if let Ok(pte) = pt.update(i, |v| v.reserve(size)) {
                    warn!("put reserve {i}");
                    self.swap_reserved(size, pte.with_idx(i), pte_a)?;
                    self.lower[core].start(size).store(page, Ordering::SeqCst);
                    Ok(())
                } else {
                    self.update_parents(page, Change::p_inc(size))
                }
            } else {
                Ok(())
            }
        } else {
            error!("Corruption l3 i{i} {size:?}");
            Err(Error::Corruption)
        }
    }

    fn put_giant(&self, page: usize) -> Result<()> {
        if (page % Table::span(Size::L2 as _)) != 0 {
            error!(
                "Invalid alignment p={page:x} a={:x}",
                Table::span(Size::L2 as _)
            );
            return Err(Error::Address);
        }

        // Clear pt1's & remove pt2 flag
        self.lower.clear_giant(page);

        let pt = self.pt3(page);
        let i = Table::idx(3, page);

        info!("free l3 i{i}");
        match pt.cas(
            i,
            Entry3::new_giant(),
            Entry3::new().with_free(Table::span(2)),
        ) {
            Ok(_) => self.update_parents(page, Change::IncEmpty),
            _ => {
                error!("Invalid {page}");
                Err(Error::Address)
            }
        }
    }
}
