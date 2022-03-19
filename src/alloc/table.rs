//! Simple reduced non-volatile memory allocator.
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, mem};

use log::{error, info, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::{Change, Entry, Entry3};
use crate::lower::LowerAlloc;
use crate::table::Table;
use crate::util::Page;

const PTE3_FULL: usize = 8 * Table::span(1);

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// This allocator splits its memory range into 1G chunks.
/// Giant pages are directly allocated in it.
/// For smaller pages, however, the 1G chunk is handed over to the
/// lower allocator, managing these smaller allocations.
/// These 1G chunks are, due to the inner workins of the lower allocator,
/// called 1G *subtrees*.
///
/// This allocator uses page tables to manage the empty and partially
/// allocated subtrees.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[repr(align(64))]
pub struct TableAlloc<L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// Metadata of the lower alloc
    lower: L,
    /// Array of the volatile level 3-5 page tables
    /// ```text
    /// DRAM: [ 1*PT5 | n*PT4 | m*PT3 ]
    /// ```
    tables: Vec<Page>,
}

unsafe impl<L: LowerAlloc> Send for TableAlloc<L> {}
unsafe impl<L: LowerAlloc> Sync for TableAlloc<L> {}

impl<L: LowerAlloc> fmt::Debug for TableAlloc<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;

        fn dump_rec<L: LowerAlloc>(
            this: &TableAlloc<L>,
            f: &mut fmt::Formatter<'_>,
            level: usize,
            start: usize,
        ) -> fmt::Result {
            if level == 3 {
                let pt3 = this.pt3(start);
                for i in Table::range(3, start..this.pages()) {
                    let pte3 = pt3.get(i);
                    let l = 7 + (Table::LAYERS - level) * 4;
                    writeln!(f, "{i:>l$} {pte3:?}")?;
                }
            } else {
                let pt = this.pt(level, start);
                for i in Table::range(level, start..this.pages()) {
                    let pte = pt.get(i);
                    let l = 7 + (Table::LAYERS - level) * 4;
                    writeln!(f, "{i:>l$} {pte:?}")?;
                    dump_rec(this, f, level - 1, Table::page(level, start, i))?;
                }
            }
            Ok(())
        }
        dump_rec(self, f, Table::LAYERS, 0)?;

        for (t, local) in self.lower.iter().enumerate() {
            let pte = local.pte(false);
            writeln!(f, "    L{t:>2}: L0 {pte:?}")?;
            let pte = local.pte(true);
            writeln!(f, "         L1 {pte:?}")?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Alloc for TableAlloc<L> {
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
        self.lower = L::new(cores, memory);

        let mut num_pt = 0;
        for level in 3..=Table::LAYERS {
            num_pt += Table::num_pts(level, self.lower.pages());
        }

        self.tables = vec![Page::new(); num_pt];

        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Try recover crashed allocator!");
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

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        // Start at the reserved memory chunk for this thread
        let page = match size {
            Size::L2 => self.get_giant(Table::LAYERS, 0)?,
            _ => self.get_small(core, size == Size::L1)?,
        };

        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
    }

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        self.put_rec(core, page)
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.allocated_pages_rec(Table::LAYERS, 0);
        // Pages allocated in reserved subtrees
        for (_t, local) in self.lower.iter().enumerate() {
            let pte = local.pte(false);
            // warn!("L {t:>2}: L0 {pte:?}");
            pages += pte.free();
            let pte = local.pte(true);
            // warn!("L {t:>2}: L1 {pte:?}");
            pages += pte.free();
        }
        self.pages() - pages
    }
}

impl<L: LowerAlloc> Drop for TableAlloc<L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}
impl<L: LowerAlloc> Default for TableAlloc<L> {
    fn default() -> Self {
        Self {
            meta: null_mut(),
            lower: L::default(),
            tables: Vec::new(),
        }
    }
}

impl<L: LowerAlloc> TableAlloc<L> {
    fn allocated_pages_rec(&self, level: usize, start: usize) -> usize {
        let mut pages = 0;
        if level == 3 {
            let pt3 = self.pt3(start);
            for i in Table::range(3, start..self.pages()) {
                let pte3 = pt3.get(i);
                pages += pte3.free();
            }
        } else {
            for i in Table::range(level, start..self.pages()) {
                pages += self.allocated_pages_rec(level - 1, Table::page(level, start, i));
            }
        }
        pages
    }

    /// Returns the page table of the given `level` that contains the `page`.
    fn pt(&self, level: usize, page: usize) -> &Table<Entry> {
        assert!((4..=Table::LAYERS).contains(&level));

        let i = page >> (Table::LEN_BITS * level);
        let offset: usize = (level..Table::LAYERS)
            .map(|i| Table::num_pts(i, self.pages()))
            .sum();
        self.tables[offset + i].cast()
    }

    /// Returns the page table of the given `level` that contains the `page`.
    fn pt3(&self, page: usize) -> &Table<Entry3> {
        let i = page >> (Table::LEN_BITS * 3);
        let offset: usize = (3..Table::LAYERS)
            .map(|i| Table::num_pts(i, self.pages()))
            .sum();
        self.tables[offset + i].cast()
    }

    /// Setup a new allocator.
    #[cold]
    fn setup_rec(&self, level: usize, start: usize) {
        for i in 0..Table::LEN {
            let page = Table::page(level, start, i);
            if page > self.pages() {
                break;
            }

            if level == 3 {
                let pt = self.pt3(page);
                let max = (self.pages() - page).min(Table::span(2));
                pt.set(i, Entry3::new().with_free(max));
            } else {
                let pt = self.pt(level, page);
                let max = (self.pages() - page).min(Table::span(level));
                let empty = max / Table::span(2);

                if max - empty * Table::span(2) > PTE3_FULL {
                    pt.set(i, Entry::new().with_empty(empty).with_partial_l0(1));
                } else {
                    pt.set(i, Entry::new().with_empty(empty));
                }
                self.setup_rec(level - 1, page);
            }
        }
    }

    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover_rec(&self, level: usize, start: usize, deep: bool) -> Result<(usize, usize, usize)> {
        let mut empty = 0;
        let mut partial_l0 = 0;
        let mut partial_l1 = 0;
        let pt = self.pt(level, start);
        for i in 0..Table::LEN {
            let page = Table::page(level, start, i);
            if page > self.pages() {
                break;
            }

            let (c_empty, c_pl0, c_pl1) = if level - 1 == 3 {
                self.recover_l3(page, deep)?
            } else {
                self.recover_rec(level - 1, page, deep)?
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

    /// Try to reserve an empty subtree.
    #[cold]
    fn reserve_rec_empty(&self, level: usize, start: usize, huge: bool) -> Result<Entry3> {
        if level == 3 {
            return self.reserve_l3_empty(start, huge);
        }

        for page in Table::iterate(level, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(level, page);
            let pt = self.pt(level, start);
            if pt.update(i, |v| v.dec_empty()).is_ok() {
                return self.reserve_rec_empty(level - 1, page, huge);
            }
        }
        error!("No memory l{level}");
        if level == Table::LAYERS {
            error!("{self:?}");
        }
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

    /// Try to reserve a partially filled subtree.
    #[cold]
    fn reserve_rec_partial(&self, level: usize, start: usize, huge: bool) -> Result<Entry3> {
        if level == 3 {
            return self.reserve_l3_partial(start, huge);
        }

        for page in Table::iterate(level, start) {
            if page > self.pages() {
                continue;
            }
            let i = Table::idx(level, page);
            let pt = self.pt(level, start);
            if pt.update(i, |v| v.dec_partial(huge)).is_ok() {
                return self.reserve_rec_partial(level - 1, page, huge);
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

    /// Propagates counter updates up to the root.
    #[cold]
    fn update_parents(&self, page: usize, change: Change) -> Result<()> {
        for level in 4..=Table::LAYERS {
            let pt = self.pt(level, page);
            let i = Table::idx(level, page);
            if pt.update(i, |v| v.change(change)).is_err() {
                error!("Update failed l{level} i{i} {change:?}");
                return Err(Error::Corruption);
            }
        }
        Ok(())
    }

    #[cold]
    fn swap_reserved(&self, huge: bool, new_pte: Entry3, pte_a: &mut Entry3) -> Result<()> {
        let old = mem::replace(pte_a, new_pte);
        if old.idx() >= Entry3::IDX_MAX {
            return Ok(());
        }

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

    fn get_giant(&self, level: usize, start: usize) -> Result<usize> {
        if level == 3 {
            return self.get_giant_l3(start);
        }

        let pt = self.pt(level, start);
        for i in 0..Table::LEN {
            let page = Table::page(level, start, i);
            if page > self.pages() {
                break;
            }

            if let Err(pte) = pt.update(i, |pte| pte.dec_empty()) {
                warn!("giant update failed {pte:?}");
                continue;
            }

            return self.get_giant(level - 1, page);
        }

        error!("Nothing found l{level} s={start}");
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
                self.lower.set_giant(page);
                return Ok(page);
            }
        }
        error!("Nothing found l3 s={start}");
        Err(Error::Memory)
    }

    fn get_small(&self, core: usize, huge: bool) -> Result<usize> {
        let pte_a = self.lower[core].pte(huge);
        let start_a = self.lower[core].start(huge);
        let mut start = *start_a;

        if start == usize::MAX {
            warn!("try reserve first");
            let pte = self
                .reserve_rec_partial(Table::LAYERS, 0, huge)
                .or_else(|_| self.reserve_rec_empty(Table::LAYERS, 0, huge))?;
            warn!("reserved {}", pte.idx());
            *pte_a = pte;
            start = pte.idx() * Table::span(2);
        } else {
            // Decrement or reserve new if full
            if let Some(pte) = pte_a.dec(huge) {
                *pte_a = pte;
            } else {
                warn!("try reserve next");
                let pte = self
                    .reserve_rec_partial(Table::LAYERS, start, huge)
                    .or_else(|_| self.reserve_rec_empty(Table::LAYERS, start, huge))?;
                warn!("reserved {}", pte.idx());
                self.swap_reserved(huge, pte, pte_a)?;
                start = pte.idx() * Table::span(2);
            }
        }

        let page = self.lower.get(core, huge, start)?;
        *start_a = page;
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

        let huge = self.lower.put(page)?;

        let lower = &self.lower[core];
        lower.frees_push(page);

        let idx = page / Table::span(2);
        let pte_a = lower.pte(huge);
        if let Some(pte) = pte_a.inc_idx(huge, idx, max) {
            *pte_a = pte;
            return Ok(());
        }

        if let Ok(pte) = pt.update(i, |v| v.inc(huge, max)) {
            let new_pages = pte.free() + Table::span(huge as _);
            if pte.reserved() {
                Ok(())
            } else if new_pages == Table::span(2) {
                self.update_parents(page, Change::p_dec(huge))
            } else if new_pages > PTE3_FULL && lower.frees_related(page) {
                // Reserve for bulk put
                if let Ok(pte) = pt.update(i, |v| v.reserve(huge)) {
                    warn!("put reserve {i}");
                    self.swap_reserved(huge, pte.with_idx(i), pte_a)?;

                    // Decrement partial counter if previously updated
                    if pte.free() <= PTE3_FULL {
                        self.update_parents(page, Change::p_dec(huge))?;
                    }

                    *lower.start(huge) = page;
                    Ok(())
                } else {
                    self.update_parents(page, Change::p_inc(huge))
                }
            } else {
                Ok(())
            }
        } else {
            error!("Corruption l3 i{i} {huge}");
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

    #[allow(unused)]
    fn dump(&self) {
        fn dump_rec<L: LowerAlloc>(this: &TableAlloc<L>, level: usize, start: usize) {
            if level == 3 {
                let pt3 = this.pt3(start);
                for i in Table::range(3, start..this.pages()) {
                    let pte3 = pt3.get(i);
                    let l = 3 + (Table::LAYERS - level) * 4;
                    error!("{i:>l$} {pte3:?}");
                }
            } else {
                let pt = this.pt(level, start);
                for i in Table::range(level, start..this.pages()) {
                    let pte = pt.get(i);
                    let l = 3 + (Table::LAYERS - level) * 4;
                    error!("{i:>l$} {pte:?}");
                    this.allocated_pages_rec(level - 1, Table::page(level, start, i));
                }
            }
        }

        for (t, local) in self.lower.iter().enumerate() {
            let pte = local.pte(false);
            error!("L {t:>2}: L0 {pte:?}");
            let pte = local.pte(true);
            error!("L {t:>2}: L1 {pte:?}");
        }
    }
}
