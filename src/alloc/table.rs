//! Simple reduced non-volatile memory allocator.
use std::ops::Range;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use super::{Alloc, Error, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::entry::{Entry, Entry3};
use crate::leaf_alloc::{LeafAllocator, Leafs};
use crate::table::{AtomicBuffer, Table};
use crate::util::Page;

/// Non-Volatile global metadata
pub struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

enum Init {
    None,
    Initializing,
    Ready,
}

/// Volatile shared metadata
#[repr(align(64))]
pub struct TableAlloc {
    memory: Range<*const Page>,
    meta: *mut Meta,
    pub local: Vec<LeafAllocator<Self>>,
    initialized: AtomicUsize,
    tables: Vec<Table<Entry>>,
}

static mut SHARED: TableAlloc = TableAlloc {
    memory: null()..null(),
    meta: core::ptr::null_mut(),
    local: Vec::new(),
    initialized: AtomicUsize::new(0),
    tables: Vec::new(),
};

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

        let alloc = unsafe { &mut SHARED };

        if alloc
            .initialized
            .compare_exchange(
                Init::None as _,
                Init::Initializing as _,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_err()
        {
            return Err(Error::Uninitialized);
        }

        // Last frame is reserved for metadata
        let mut pages = (memory.len() - 1).min(MAX_PAGES);
        let (memory, rem) = memory.split_at_mut(pages);

        let meta = rem[0].cast::<Meta>();
        alloc.meta = meta;

        // level 2 tables are stored at the end of the NVM
        pages -= Table::num_pts(2, pages);
        let (memory, _pt2) = memory.split_at_mut(pages);
        alloc.memory = memory.as_ptr_range();

        let mut num_pt = 0;
        for layer in 3..=Table::LAYERS {
            num_pt += Table::num_pts(layer, pages);
        }
        alloc.tables = vec![Table::empty(); num_pt];
        alloc.local = vec![LeafAllocator::new(memory.as_ptr() as _, pages); cores];

        if meta.pages.load(Ordering::SeqCst) == pages && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", pages);
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Allocator unexpectedly terminated");
            }
            let pages = alloc.recover_rec(Table::LAYERS, 0, deep)?;
            warn!("Recovered pages {}", pages);
        } else {
            warn!("Setup allocator state p={}", pages);
            alloc.local[0].clear();
            meta.pages.store(pages, Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        // init all leaf_allocs
        let mut start = 0;
        for (i, leaf) in alloc.local.iter().enumerate() {
            start = alloc.reserve_pt2(Table::LAYERS, start, Size::L0)?;
            leaf.start(Size::L0).store(start, Ordering::Relaxed);
            start = alloc.reserve_pt2(Table::LAYERS, start, Size::L1)?;
            leaf.start(Size::L1).store(start, Ordering::Relaxed);
            info!(
                "init {} small={} huge={}",
                i,
                leaf.start(Size::L0).load(Ordering::Relaxed),
                leaf.start(Size::L1).load(Ordering::Relaxed),
            );
        }

        meta.active.store(1, Ordering::SeqCst);
        alloc.initialized.store(Init::Ready as _, Ordering::SeqCst);
        Ok(())
    }

    fn uninit() {
        let alloc = unsafe { &mut SHARED };

        alloc
            .initialized
            .compare_exchange(
                Init::Ready as _,
                Init::Initializing as _,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .unwrap();
        unsafe { &*alloc.meta }.active.store(0, Ordering::SeqCst);
        alloc.memory = null()..null();
        alloc.tables.clear();
        alloc.meta = null_mut();
        alloc.local.clear();
        alloc.initialized.store(Init::None as _, Ordering::SeqCst);
    }

    fn instance<'a>() -> &'a Self {
        let alloc = unsafe { &SHARED };
        assert!(
            alloc.initialized.load(Ordering::SeqCst) == Init::Ready as usize,
            "Not initialized"
        );
        alloc
    }

    fn destroy() {
        let alloc = unsafe { &mut SHARED };
        unsafe { &*alloc.meta }.magic.store(0, Ordering::SeqCst);
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
            let start = leaf.start(size);
            let old = start.load(Ordering::SeqCst);
            let new = self.increment_parents(Table::LAYERS, old, size)?;

            let page = match size {
                Size::L0 => leaf.get(new)?,
                Size::L1 => leaf.get_huge(new)?,
                Size::L2 => panic!(),
            };
            start.store(page, Ordering::SeqCst);
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
        assert_eq!(
            self.initialized.load(Ordering::SeqCst),
            Init::Ready as usize
        );

        let mut pages = 0;
        let pt = self.pt(Table::LAYERS, 0);
        for i in Table::range(Table::LAYERS, 0..self.pages()) {
            let pte = pt.get(i);
            warn!("{i:>3} {pte:?}");
            pages += pte.pages();
        }
        pages
    }
}

impl TableAlloc {
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

    fn recover_rec(&self, layer: usize, start: usize, deep: bool) -> Result<usize> {
        let mut pages = 0;
        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages()) {
            let page = Table::page(layer, start, i);

            let c_pages = if layer - 1 == 3 {
                self.recover_l3(page, deep)?
            } else {
                self.recover_rec(layer - 1, page, deep)?
            };

            pt.set(i, Entry::new().with_pages(c_pages));
            pages += c_pages;
        }

        Ok(pages)
    }

    fn recover_l3(&self, start: usize, deep: bool) -> Result<usize> {
        let mut pages = 0;
        let pt = self.pt3(start);

        for i in Table::range(3, start..self.pages()) {
            let page = Table::page(3, start, i);

            let (c_pages, size) = self.local[0].recover(page, deep)?;
            if size == Size::L2 {
                pt.set(i, Entry3::new_giant());
            } else if c_pages > 0 {
                pt.set(i, Entry3::new_table(c_pages, size, false));
            } else {
                pt.set(i, Entry3::new());
            }
            pages += c_pages;
        }
        Ok(pages)
    }

    fn reserve_pt2(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        for i in 0..Table::LEN {
            let i = (Table::idx(layer, start) + i) % Table::LEN;
            let page = Table::page(layer, start, i);

            if layer > 3 {
                if let Ok(result) = self.reserve_pt2(layer - 1, page, size) {
                    return Ok(result);
                }
            } else {
                let pt = self.pt3(start);
                let max = self.pages() - page;
                if pt.update(i, |v| v.reserve(size, max)).is_ok() {
                    return Ok(page);
                }
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn increment_parents(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        if layer <= 3 {
            return self.increment_parents_l3(start, size);
        }

        // Increment parents
        let pt = self.pt(layer, start);
        for page in Table::iterate(layer, start, self.pages()) {
            let i = Table::idx(layer, page);

            let max = self.pages() - Table::round(layer, page);
            if pt.update(i, |v| v.inc(size, layer, max)).is_ok() {
                match self.increment_parents(layer - 1, page, size) {
                    Ok(result) => return Ok(result),
                    Err(Error::Memory) => {
                        // TODO: special behavior on fragmentation
                        pt.update(i, |v| v.dec(size)).unwrap();
                    }
                    Err(e) => return Err(e),
                }
            };
        }
        Err(Error::Memory)
    }

    fn increment_parents_l3(&self, start: usize, size: Size) -> Result<usize> {
        let pt = self.pt3(start);
        for page in Table::iterate(3, start, self.pages()) {
            let i = Table::idx(3, page);

            let max = self.pages() - Table::round(2, page);
            if page == start {
                if pt.update(i, |v| v.inc(size, max)).is_ok() {
                    return Ok(page);
                }
                warn!("try reserve next {i}");
            } else if pt.update(i, |v| v.inc_reserve(size, max)).is_ok() {
                warn!("reserved {i}");
                pt.update(Table::idx(3, start), Entry3::unreserve).unwrap();
                return Ok(page);
            }
        }
        Err(Error::Memory)
    }

    fn get_giant(&self, core: usize, layer: usize, start: usize) -> Result<usize> {
        if layer == 3 {
            return self.get_giant_page(core, start);
        }

        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages()) {
            let page = Table::page(layer, start, i);
            let pte = pt.get(i);

            // Already allocated or reserved
            if pte.pages() > Table::span(layer - 1) - Table::span(Size::L2 as _) {
                warn!(
                    "giant no space {} > {} - {}",
                    pte.pages(),
                    Table::span(layer - 1),
                    Table::span(Size::L2 as _)
                );
                continue;
            }

            let max = self.pages() - page;
            if let Err(pte) = pt.update(i, |pte| pte.inc(Size::L2, layer, max)) {
                warn!("giant update failed {pte:?}");
                continue;
            }

            return match self.get_giant(core, layer - 1, page) {
                Ok(page) => Ok(page),
                Err(Error::Memory) => match pt.update(i, |pte| pte.dec(Size::L2)) {
                    Ok(_) => Err(Error::Memory),
                    Err(pte) => {
                        error!("revocation failed {layer} {pte:?}");
                        Err(Error::Corruption)
                    }
                },
                Err(e) => Err(e),
            };
        }

        error!("Nothing found l{layer} s={start}");
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn get_giant_page(&self, core: usize, start: usize) -> Result<usize> {
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

    fn put_rec(&self, core: usize, layer: usize, page: usize) -> Result<Size> {
        if layer == 3 {
            return self.put_l3(core, page);
        }

        let pt = self.pt(layer, page);
        let i = Table::idx(layer, page);
        let pte = pt.get(i);

        if pte.pages() == 0 {
            error!("No table found l{} {:?}", layer, pte);
            return Err(Error::Address);
        }

        let size = self.put_rec(core, layer - 1, page)?;

        match pt.update(i, |v| v.dec(size)) {
            Ok(_) => Ok(size),
            Err(pte) => {
                error!("Corruption: l{} i{} {:?}", layer, i, pte);
                Err(Error::Corruption)
            }
        }
    }

    fn put_l3(&self, core: usize, page: usize) -> Result<Size> {
        let pt = self.pt3(page);
        let i3 = Table::idx(3, page);
        let pte3 = pt.get(i3);

        if pte3.size() == Some(Size::L2) {
            warn!("free giant l3 i{}", i3);
            return self.put_giant(core, page);
        }

        if pte3.pages() == 0 {
            error!("Invalid address l3 i{}", i3);
            return Err(Error::Address);
        }

        let size = self.local[core].put(page)?;

        if let Err(pte3) = pt.update(i3, |v| v.dec(size)) {
            error!("Corruption l3 i{} p={} - {:?}", i3, pte3.pages(), size);
            return Err(Error::Corruption);
        }
        Ok(size)
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
