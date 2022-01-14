//! Simple reduced non-volatile memory allocator.
use std::ops::Range;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use log::{error, info, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::{Dec, Entry, Entry3};
use crate::leaf_alloc::{LeafAllocator, Leafs};
use crate::table::Table;
use crate::util::Page;

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
            let leaf = &self.local[core];
            let start_a = leaf.start(size);
            let mut start = start_a.load(Ordering::SeqCst);

            if start == usize::MAX {
                warn!("try reserve first");
                start = self
                    .reserve_rec_partial(Table::LAYERS, 0, size)
                    .or_else(|_| self.reserve_rec_empty(Table::LAYERS, 0, size))?;
            } else {
                // Increment or reserve new if full
                let pt = self.pt3(start);
                let i = Table::idx(3, start);
                let max = self.pages() - Table::round(2, start) - 1;
                if pt.update(i, |v| v.inc(size, max)).is_err() {
                    warn!("try reserve next");
                    start = self
                        .reserve_rec_partial(Table::LAYERS, start, size)
                        .or_else(|_| self.reserve_rec_empty(Table::LAYERS, start, size))?;
                    if pt.update(i, Entry3::unreserve).is_err() {
                        panic!("Unreserve failed")
                    }
                }
            }
            assert!(start < self.pages());

            let page = match size {
                Size::L0 => leaf.get(start)?,
                Size::L1 => leaf.get_huge(start)?,
                Size::L2 => panic!(),
            };
            start_a.store(page, Ordering::SeqCst);
            page
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
            match self.put_rec(core, Table::LAYERS, page) {
                Ok(_) => return Ok(()),
                Err(Error::CAS) => warn!("CAS: retry free"),
                Err(e) => return Err(e),
            }
        }
    }

    fn allocated_pages(&self) -> usize {
        self.allocated_pages_rec(Table::LAYERS, 0)
    }
}

impl TableAlloc {
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
            for i in Table::range(layer, 0..self.pages()) {
                let pte3 = pt3.get(i);
                // warn!("{i:>3}: {pte3:?}");
                match pte3.size() {
                    Some(Size::L2) => pages += Table::span(2),
                    Some(_) => pages += pte3.pages(),
                    _ => {}
                }
            }
        } else {
            let pt = self.pt(layer, start);
            for i in Table::range(layer, 0..self.pages()) {
                let pte = pt.get(i);
                warn!("{i:>3}: {pte:?}");
                if pte.full() > 0 || pte.partial_l0() > 0 || pte.partial_l1() > 0 {
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

    fn recover_rec(&self, layer: usize, start: usize, deep: bool) -> Result<(usize, usize, usize)> {
        let mut full = 0;
        let mut partial_l0 = 0;
        let mut partial_l1 = 0;
        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages()) {
            let page = Table::page(layer, start, i);

            let (c_full, c_pl0, c_pl1) = if layer - 1 == 3 {
                self.recover_l3(page, deep)?
            } else {
                self.recover_rec(layer - 1, page, deep)?
            };

            pt.set(
                i,
                Entry::new()
                    .with_full(c_full)
                    .with_partial_l0(c_pl0)
                    .with_partial_l1(c_pl1),
            );
            full += c_full;
            partial_l0 += c_pl0;
            partial_l1 += c_pl1;
        }

        Ok((full, partial_l0, partial_l1))
    }

    fn recover_l3(&self, start: usize, deep: bool) -> Result<(usize, usize, usize)> {
        let mut full = 0;
        let mut partial_l0 = 0;
        let mut partial_l1 = 0;
        let pt = self.pt3(start);

        for i in Table::range(3, start..self.pages()) {
            let page = Table::page(3, start, i);

            let (pages, size) = self.local[0].recover(page, deep)?;
            if size == Size::L2 {
                pt.set(i, Entry3::new_giant());
                full += 1;
            } else if pages > 0 {
                pt.set(i, Entry3::new_table(pages, size, false));
                if pages < PTE3_FULL {
                    if size == Size::L0 {
                        partial_l0 += 1;
                    } else {
                        partial_l1 += 1;
                    }
                } else {
                    full += 1;
                }
            } else {
                pt.set(i, Entry3::new());
            }
        }
        Ok((full, partial_l0, partial_l1))
    }

    fn reserve_rec_empty(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        if layer == 3 {
            return self.reserve_l3_empty(start, size);
        }

        for page in Table::iterate(layer, start, self.pages()) {
            let i = Table::idx(layer, page);

            let pt = self.pt(layer, start);
            let max =
                (self.pages() - Table::round(layer, page) + Table::span(2) - 1) / Table::span(2);
            if pt.update(i, |v| v.inc_full(max)).is_ok() {
                if let Ok(result) = self.reserve_rec_empty(layer - 1, page, size) {
                    return Ok(result);
                }
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn reserve_l3_empty(&self, start: usize, size: Size) -> Result<usize> {
        for page in Table::iterate(3, start, self.pages()) {
            let i = Table::idx(3, page);

            let pt = self.pt3(start);
            let max = self.pages() - page - 1;
            if pt.update(i, |v| v.reserve_inc_empty(size, max)).is_ok() {
                return Ok(page);
            }
        }
        Err(Error::Memory)
    }

    fn reserve_rec_partial(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        if layer == 3 {
            return self.reserve_l3_partial(start, size);
        }

        for page in Table::iterate(layer, start, self.pages()) {
            let i = Table::idx(layer, page);

            let pt = self.pt(layer, start);
            let max =
                (self.pages() - Table::round(layer, page) + Table::span(2) - 1) / Table::span(2);
            if pt.update(i, |v| v.reserve_partial(size, max)).is_ok() {
                if let Ok(result) = self.reserve_rec_partial(layer - 1, page, size) {
                    return Ok(result);
                }
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn reserve_l3_partial(&self, start: usize, size: Size) -> Result<usize> {
        for page in Table::iterate(3, start, self.pages()) {
            let i = Table::idx(3, page);

            let pt = self.pt3(start);
            let max = self.pages() - page - 1;
            if pt.update(i, |v| v.reserve_inc_partial(size, max)).is_ok() {
                return Ok(page);
            }
        }
        Err(Error::Memory)
    }

    fn get_giant(&self, core: usize, layer: usize, start: usize) -> Result<usize> {
        if layer == 3 {
            return self.get_giant_l3(core, start);
        }

        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages()) {
            let page = Table::page(layer, start, i);

            let max = (self.pages() - page) / Table::span(2);
            if let Err(pte) = pt.update(i, |pte| pte.inc_full(max)) {
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

        for i in Table::range(3, start..self.pages()) {
            if pt.cas(i, Entry3::new(), Entry3::new_giant()).is_ok() {
                let page = Table::page(3, start, i);
                info!("allocated l3 i{} p={} s={}", i, page, start);
                self.local[core].persist(page);
                return Ok(page);
            }
        }
        error!("Nothing found l3 s={start}");
        Err(Error::Memory)
    }

    fn put_rec(&self, core: usize, layer: usize, page: usize) -> Result<Dec> {
        if layer == 3 {
            return self.put_l3(core, page);
        }

        let pt = self.pt(layer, page);
        let i = Table::idx(layer, page);
        let pte = pt.get(i);

        if pte.full() == 0 && pte.partial_l0() == 0 && pte.partial_l1() == 0 {
            error!("No table found l{} {:?}", layer, pte);
            return Err(Error::Address);
        }

        let dec = self.put_rec(core, layer - 1, page)?;
        if dec == Dec::None {
            return Ok(dec);
        }

        match pt.update(i, |v| v.dec(dec)) {
            Ok(_) => Ok(dec),
            Err(pte) => {
                error!("Corruption: l{layer} i{i} {pte:?} {dec:?}");
                Err(Error::Corruption)
            }
        }
    }

    fn put_l3(&self, core: usize, page: usize) -> Result<Dec> {
        let pt = self.pt3(page);
        let i3 = Table::idx(3, page);
        let pte3 = pt.get(i3);

        if pte3.size() == Some(Size::L2) {
            warn!("free giant l3 i{}", i3);
            self.put_giant(core, page)?;
            return Ok(Dec::Full);
        }

        if pte3.pages() == 0 {
            error!("Invalid address l3 i{}", i3);
            return Err(Error::Address);
        }

        let size = self.local[core].put(page)?;

        match pt.update(i3, |v| v.dec(size)) {
            Err(pte3) => {
                error!("Corruption l3 i{} p={} - {:?}", i3, pte3.pages(), size);
                Err(Error::Corruption)
            }
            Ok(pte3) => {
                if pte3.reserved() {
                    Ok(Dec::None)
                } else if pte3.pages() >= PTE3_FULL
                    && pte3.pages() - Table::span(size as _) < PTE3_FULL
                {
                    if size == Size::L0 {
                        Ok(Dec::FullPartialL0)
                    } else {
                        Ok(Dec::FullPartialL1)
                    }
                } else if pte3.pages() == Table::span(size as _) {
                    if size == Size::L0 {
                        Ok(Dec::PartialL0)
                    } else {
                        Ok(Dec::PartialL1)
                    }
                } else {
                    Ok(Dec::None)
                }
            }
        }
    }

    fn put_giant(&self, core: usize, page: usize) -> Result<Size> {
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
        match pt.cas(i, Entry3::new_giant(), Entry3::new()) {
            Ok(_) => Ok(Size::L2),
            _ => {
                error!("Invalid {page}");
                Err(Error::Address)
            }
        }
    }
}
