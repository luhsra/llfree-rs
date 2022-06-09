//! Simple reduced non-volatile memory allocator.
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{fmt, mem};

use log::{error, info, warn};

use super::{Alloc, Error, Local, Result, Size, CAS_RETRIES, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::{Change, Entry, Entry3};
use crate::lower::LowerAlloc;
use crate::table::{AtomicTable, Mapping};
use crate::util::Page;

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
    /// Array of the volatile level 3-5 page tables
    /// ```text
    /// DRAM: [ 1*PT5 | n*PT4 | m*PT3 ]
    /// ```
    tables: Box<[Page]>,
    /// CPU local data
    local: Box<[Local]>,
    /// Metadata of the lower alloc
    lower: L,
}

unsafe impl<L: LowerAlloc> Send for TableAlloc<L> {}
unsafe impl<L: LowerAlloc> Sync for TableAlloc<L> {}

impl<L: LowerAlloc> fmt::Debug for TableAlloc<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn dump_rec<L: LowerAlloc>(
            this: &TableAlloc<L>,
            f: &mut fmt::Formatter<'_>,
            level: usize,
            start: usize,
        ) -> fmt::Result {
            if level == 3 {
                let pt3 = this.pt3(start);
                for i in TableAlloc::<L>::MAPPING.range(3, start..this.pages()) {
                    let pte3 = pt3.get(i);
                    let l = 7 + (TableAlloc::<L>::MAPPING.levels() - level) * 4;
                    writeln!(f, "{i:>l$} {pte3:?}")?;
                }
            } else {
                let pt = this.pt(level, start);
                for i in TableAlloc::<L>::MAPPING.range(level, start..this.pages()) {
                    let pte = pt.get(i);
                    let l = 7 + (TableAlloc::<L>::MAPPING.levels() - level) * 4;
                    writeln!(f, "{i:>l$} {pte:?}")?;
                    dump_rec(
                        this,
                        f,
                        level - 1,
                        TableAlloc::<L>::MAPPING.page(level, start, i),
                    )?;
                }
            }
            Ok(())
        }

        writeln!(f, "{} {{", self.name())?;
        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;

        dump_rec(self, f, Self::MAPPING.levels(), 0)?;

        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: L0 {:?}", local.pte(false))?;
            writeln!(f, "         L1 {:?}", local.pte(true))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Alloc for TableAlloc<L> {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
        info!(
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

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Local::new);
        self.local = local.into();

        // Create lower allocator
        self.lower = L::new(cores, memory);

        let mut num_pt = 0;
        for level in 3..=Self::MAPPING.levels() {
            num_pt += Self::MAPPING.num_pts(level, self.lower.pages());
        }

        self.tables = vec![Page::new(); num_pt].into();

        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            info!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                warn!("Try recover crashed allocator!");
            }
            self.recover_rec(Self::MAPPING.levels(), 0, deep)?;
        } else {
            info!("Setup allocator state p={}", self.pages());
            self.lower.clear();

            self.setup_rec(Self::MAPPING.levels(), 0);

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
            Size::L2 => self.get_giant(Self::MAPPING.levels(), 0)?,
            _ => self.get_lower(core, size == Size::L1)?,
        };

        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
    }

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<Size> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let pt = self.pt3(page);
        let i = Self::MAPPING.idx(3, page);
        let pte = pt.get(i);

        if pte.page() {
            return self.put_giant(page).map(|_| Size::L2);
        }

        let max = (self.pages() - Self::MAPPING.round(2, page)).min(Self::MAPPING.span(2));
        if pte.free() >= max {
            error!("Invalid address l3 i{i}");
            return Err(Error::Address);
        }

        let huge = self.lower.put(page)?;
        let size = if huge { Size::L1 } else { Size::L0 };

        let local = &self.local[core];
        local.frees_push(page);

        // Try updating local copy
        let idx = page / Self::MAPPING.span(2);
        let pte_a = local.pte(huge);
        if let Some(pte) = pte_a.inc_idx(Self::MAPPING.span(huge as _), idx, max) {
            *pte_a = pte;
            return Ok(size);
        }
        // Fallback to global page table
        if let Ok(pte) = pt.update(i, |v| v.inc(Self::MAPPING.span(huge as _), max)) {
            let new_pages = pte.free() + Self::MAPPING.span(huge as _);
            if pte.reserved() {
                return Ok(size);
            } else if new_pages == Self::MAPPING.span(2) {
                self.update_parents(page, Change::p_dec(huge))?;
                return Ok(size);
            }

            // Reserve for bulk put
            if new_pages > Self::PTE3_FULL && local.frees_related(page) {
                // Try reserve this subtree
                match self.reserve_tree(huge, page, pte.free() > Self::PTE3_FULL) {
                    Ok(pte) => {
                        info!("put reserve {i}");
                        self.swap_reserved(huge, pte.with_idx(i), pte_a)?;
                        *local.start(huge) = page;
                        return Ok(size);
                    }
                    Err(Error::Memory) => (),
                    Err(e) => return Err(e),
                }
            }

            if pte.free() <= Self::PTE3_FULL && new_pages > Self::PTE3_FULL {
                // Increment parents if exceeding treshold
                self.update_parents(page, Change::p_inc(huge))?;
            }
            Ok(size)
        } else {
            error!("Corruption l3 i{i} {huge}");
            Err(Error::Corruption)
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.allocated_pages_rec(Self::MAPPING.levels(), 0);
        // Pages allocated in reserved subtrees
        for (_t, local) in self.local.iter().enumerate() {
            let pte = local.pte(false);
            // warn!("L {t:>2}: L0 {pte:?}");
            pages += pte.free();
            let pte = local.pte(true);
            // warn!("L {t:>2}: L1 {pte:?}");
            pages += pte.free();
        }
        self.pages() - pages
    }

    fn span(&self, size: Size) -> usize {
        Self::MAPPING.span(size as _)
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
            tables: Box::new([]),
            local: Box::new([]),
            lower: L::default(),
        }
    }
}

impl<L: LowerAlloc> TableAlloc<L> {
    const MAPPING: Mapping<4> = Mapping([512; 2]).with_lower(&L::MAPPING);
    const PTE3_FULL: usize = 8 * Self::MAPPING.span(1);

    fn allocated_pages_rec(&self, level: usize, start: usize) -> usize {
        let mut pages = 0;
        if level == 3 {
            let pt3 = self.pt3(start);
            for i in Self::MAPPING.range(3, start..self.pages()) {
                let pte3 = pt3.get(i);
                pages += pte3.free();
            }
        } else {
            for i in Self::MAPPING.range(level, start..self.pages()) {
                pages += self.allocated_pages_rec(level - 1, Self::MAPPING.page(level, start, i));
            }
        }
        pages
    }

    /// Returns the page table of the given `level` that contains the `page`.
    fn pt(&self, level: usize, page: usize) -> &AtomicTable<Entry> {
        assert!((4..=Self::MAPPING.levels()).contains(&level));

        let i = page / Self::MAPPING.span(level);
        let offset: usize = (level..Self::MAPPING.levels())
            .map(|i| Self::MAPPING.num_pts(i, self.pages()))
            .sum();
        self.tables[offset + i].cast()
    }

    /// Returns the page table of the given `level` that contains the `page`.
    fn pt3(&self, page: usize) -> &AtomicTable<Entry3> {
        let i = page / Self::MAPPING.span(3);
        let offset: usize = (3..Self::MAPPING.levels())
            .map(|i| Self::MAPPING.num_pts(i, self.pages()))
            .sum();
        self.tables[offset + i].cast()
    }

    /// Setup a new allocator.
    #[cold]
    fn setup_rec(&self, level: usize, start: usize) {
        for i in 0..Self::MAPPING.len(level) {
            let page = Self::MAPPING.page(level, start, i);
            if page > self.pages() {
                break;
            }

            if level == 3 {
                let pt = self.pt3(page);
                let max = (self.pages() - page).min(Self::MAPPING.span(2));
                pt.set(i, Entry3::new().with_free(max));
            } else {
                let pt = self.pt(level, page);
                let max = (self.pages() - page).min(Self::MAPPING.span(level));
                let empty = max / Self::MAPPING.span(2);

                if max - empty * Self::MAPPING.span(2) > Self::PTE3_FULL {
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
        for i in 0..Self::MAPPING.len(level) {
            let page = Self::MAPPING.page(level, start, i);
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

        for i in 0..Self::MAPPING.len(3) {
            let page = Self::MAPPING.page(3, start, i);
            if page > self.pages() {
                break;
            }

            let (pages, size) = self.lower.recover(page, deep)?;
            if size == Size::L2 {
                pt.set(i, Entry3::new_giant());
            } else if pages > 0 {
                pt.set(i, Entry3::new_table(pages, size, false));
                if pages == Self::MAPPING.span(2) {
                    empty += 1;
                } else if pages > Self::PTE3_FULL {
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

    /// Updates the page tables from top to bottom,
    /// returning the subtree that was successfully changed.
    /// The global index is returned as part of the entry (`idx`).
    fn find_rec<F, G>(&self, level: usize, start: usize, mut fx: F, mut f3: G) -> Result<Entry3>
    where
        F: FnMut(Entry) -> Option<Entry>,
        G: FnMut(Entry3) -> Option<Entry3>,
    {
        for _ in 0..CAS_RETRIES {
            if level == 3 {
                for page in Self::MAPPING.iterate(3, start) {
                    if page > self.pages() {
                        continue;
                    }
                    let i = Self::MAPPING.idx(3, page);
                    let pt = self.pt3(start);
                    if let Ok(pte) = pt.update(i, |v| f3(v)) {
                        return Ok(pte.with_idx(page / Self::MAPPING.span(2)));
                    }
                }
            } else {
                for page in Self::MAPPING.iterate(level, start) {
                    if page > self.pages() {
                        continue;
                    }
                    let i = Self::MAPPING.idx(level, page);
                    let pt = self.pt(level, start);
                    if pt.update(i, |v| fx(v)).is_ok() {
                        return self.find_rec(level - 1, page, fx, f3);
                    }
                }
            }

            if level == Self::MAPPING.levels() {
                return Err(Error::Memory);
            }
            core::hint::spin_loop(); // pause cpu
        }

        error!("Missing l{level}!\n{self:?}");
        Err(Error::Corruption)
    }

    /// Try reserving new subtree, prioritizing partially filled ones.
    #[cold]
    fn reserve(&self, huge: bool) -> Result<Entry3> {
        match self.find_rec(
            Self::MAPPING.levels(),
            0,
            |v| v.dec_partial(huge),
            |v| v.reserve_partial(huge, Self::PTE3_FULL, Self::MAPPING.span(2)),
        ) {
            Ok(v) => v
                .dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
                .ok_or(Error::Corruption),
            Err(Error::Memory) => match self.find_rec(
                Self::MAPPING.levels(),
                0,
                |v| v.dec_empty(),
                |v| v.reserve_empty(huge, Self::MAPPING.span(2)),
            ) {
                Ok(v) => v
                    .dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
                    .ok_or(Error::Corruption),
                Err(Error::Memory) => {
                    error!("No memory {self:?}");
                    Err(Error::Memory)
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        }
    }

    /// Try reserving a subtree, if `parents` the parent entries are updated first.
    /// Any changes to parent entries are reverted if the reservation fails.
    fn reserve_tree(&self, huge: bool, page: usize, parents: bool) -> Result<Entry3> {
        // Try reserve top -> bottom
        if parents {
            for level in (4..=Self::MAPPING.levels()).rev() {
                let i = Self::MAPPING.idx(level, page);
                let pt = self.pt(level, page);
                if pt.update(i, |v| v.dec_partial(huge)).is_err() {
                    // Undo reservation
                    for level in level + 1..=Self::MAPPING.levels() {
                        let i = Self::MAPPING.idx(level, page);
                        let pt = self.pt(level, page);
                        if pt.update(i, |v| v.change(Change::p_inc(huge))).is_err() {
                            error!("Unable to undo reservation\n{self:?}");
                            return Err(Error::Corruption);
                        }
                    }
                    return Err(Error::Memory);
                }
            }
        }
        // Reserve the level 3 entry
        let i = Self::MAPPING.idx(3, page);
        let pt = self.pt3(page);
        if let Ok(pte) = pt.update(i, |v| {
            v.reserve(Self::MAPPING.span(huge as _), Self::MAPPING.span(2))
        }) {
            return Ok(pte);
        } else if parents {
            // Undo reservation
            self.update_parents(page, Change::p_inc(huge))?;
        }

        Err(Error::Memory)
    }

    /// Propagates counter updates up to the root.
    #[cold]
    fn update_parents(&self, page: usize, change: Change) -> Result<()> {
        for level in 4..=Self::MAPPING.levels() {
            let pt = self.pt(level, page);
            let i = Self::MAPPING.idx(level, page);
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

        let start = old.idx() * Self::MAPPING.span(2);
        let i = Self::MAPPING.idx(3, start);
        let pt = self.pt3(start);
        let max = (self.pages() - start).min(Self::MAPPING.span(2));

        if let Ok(pte) = pt.update(i, |v| {
            v.unreserve_add(huge, old, max, Self::MAPPING.span(2))
        }) {
            // Update parents
            let new_pages = old.free() + pte.free();
            if new_pages == Self::MAPPING.span(2) {
                self.update_parents(start, Change::IncEmpty)
            } else if new_pages > Self::PTE3_FULL {
                self.update_parents(start, Change::p_inc(huge))
            } else {
                Ok(())
            }
        } else {
            error!("Unreserve failed {i}");
            Err(Error::Corruption)
        }
    }

    fn get_lower(&self, core: usize, huge: bool) -> Result<usize> {
        let pte_a = self.local[core].pte(huge);
        let start_a = self.local[core].start(huge);
        let mut start = *start_a;

        if start == usize::MAX {
            let pte = self.reserve(huge)?;
            info!("reserved {}", pte.idx());
            *pte_a = pte;
            start = pte.idx() * Self::MAPPING.span(2);
        } else {
            // Decrement or reserve new if full
            if let Some(pte) = pte_a.dec(Self::MAPPING.span(huge as _), Self::MAPPING.span(2)) {
                *pte_a = pte;
            } else {
                let pte = self.reserve(huge)?;
                info!("reserved {}", pte.idx());
                self.swap_reserved(huge, pte, pte_a)?;
                start = pte.idx() * Self::MAPPING.span(2);
            }
        }

        let page = self.lower.get(core, huge, start)?;
        *start_a = page;
        Ok(page)
    }

    fn get_giant(&self, level: usize, start: usize) -> Result<usize> {
        let pte = self.find_rec(level, start, Entry::dec_empty, |v| {
            (v.free() == Self::MAPPING.span(2)).then(Entry3::new_giant)
        })?;
        let page = pte.idx() * Self::MAPPING.span(2);
        self.lower.set_giant(page);
        Ok(page)
    }

    fn put_giant(&self, page: usize) -> Result<()> {
        if (page % Self::MAPPING.span(Size::L2 as _)) != 0 {
            error!(
                "Invalid alignment p={page:x} a={:x}",
                Self::MAPPING.span(Size::L2 as _)
            );
            return Err(Error::Address);
        }

        let pt = self.pt3(page);
        let i = Self::MAPPING.idx(3, page);
        if let Ok(_) = pt.cas(
            i,
            Entry3::new_giant(),
            Entry3::new().with_free(Self::MAPPING.span(2)),
        ) {
            self.lower.clear_giant(page);
            self.update_parents(page, Change::IncEmpty)
        } else {
            error!("Invalid {page}");
            Err(Error::Address)
        }
    }
}
