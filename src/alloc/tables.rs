//! Simple reduced non-volatile memory allocator.
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use super::Alloc;
use crate::entry::{Entry, Entry3};
use crate::leaf_alloc::LeafAllocator;
use crate::table::{AtomicBuffer, Table};
use crate::{Error, Init, Meta, Page, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};

/// Volatile shared metadata
#[repr(align(64))]
pub struct AllocTables {
    begin: usize,
    pages: usize,
    meta: *mut Meta,
    pub local: Vec<LeafAllocator>,
    initialized: AtomicUsize,
    tables: Vec<Table<Entry>>,
}

static mut SHARED: AllocTables = AllocTables {
    begin: 0,
    pages: 0,
    meta: core::ptr::null_mut(),
    local: Vec::new(),
    initialized: AtomicUsize::new(0),
    tables: Vec::new(),
};

impl Alloc for AllocTables {
    fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        info!(
            "init cores={} mem={:?} ({})",
            cores,
            memory.as_ptr_range(),
            memory.len()
        );

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

        if memory.len() < MIN_PAGES * cores {
            error!("memory {} < {}", memory.len(), MIN_PAGES * cores);
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        alloc.pages = (memory.len() - 1).min(MAX_PAGES);
        alloc.begin = memory.as_ptr() as usize;

        alloc.meta = (alloc.begin + alloc.pages * Page::SIZE) as *mut Meta;
        let meta = unsafe { &mut *alloc.meta };

        // level 2 tables are stored at the end of the NVM
        alloc.pages -= Table::num_pts(2, alloc.pages);

        let mut num_pt = 0;
        for layer in 3..=Table::LAYERS {
            num_pt += Table::num_pts(layer, alloc.pages);
        }
        alloc.tables = vec![Table::empty(); num_pt];
        alloc.local = vec![LeafAllocator::new(alloc.begin, alloc.pages); cores];

        if meta.pages.load(Ordering::SeqCst) == alloc.pages
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", alloc.pages);
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Allocator unexpectedly terminated");
            }
            let pages = alloc.recover_rec(Table::LAYERS, 0, deep)?;
            warn!("Recovered pages {}", pages);
        } else {
            warn!("Setup allocator state p={}", alloc.pages);
            alloc.local[0].clear();
            meta.pages.store(alloc.pages, Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        // init all leaf_allocs
        let mut start = 0;
        for (i, leaf) in alloc.local.iter().enumerate() {
            start = alloc.reserve_pt2(Table::LAYERS, start, Size::L0)?;
            leaf.start_l0.store(start, Ordering::Relaxed);
            start = alloc.reserve_pt2(Table::LAYERS, start, Size::L1)?;
            leaf.start_l1.store(start, Ordering::Relaxed);
            info!(
                "init {} small={} huge={}",
                i,
                leaf.start_l0.load(Ordering::Relaxed),
                leaf.start_l1.load(Ordering::Relaxed),
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
        alloc.meta().active.store(0, Ordering::SeqCst);
        alloc.begin = 0;
        alloc.pages = 0;
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

    fn begin(&self) -> usize {
        self.begin
    }

    fn pages(&self) -> usize {
        self.pages
    }

    fn meta<'a>(&self) -> &'a Meta {
        unsafe { &*self.meta }
    }

    fn get(&self, core: usize, size: Size) -> Result<usize> {
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

            match size {
                Size::L0 => {
                    let start = leaf.start_l0.load(Ordering::SeqCst);
                    let new = self.increment_parents(Table::LAYERS, start, size)?;
                    let page = leaf.get(new)?;
                    leaf.start_l0.store(page, Ordering::SeqCst);
                    page
                }
                Size::L1 => {
                    let start = leaf.start_l1.load(Ordering::SeqCst);
                    let new = self.increment_parents(Table::LAYERS, start, size)?;
                    let page = leaf.get_huge(new)?;
                    leaf.start_l1.store(page, Ordering::SeqCst);
                    page
                }
                Size::L2 => panic!(),
            }
        };

        Ok(page)
    }

    fn put(&self, core: usize, page: usize) -> Result<Size> {
        loop {
            match self.put_rec(core, Table::LAYERS, page) {
                Err(Error::CAS) => warn!("CAS: retry free"),
                r => return r,
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
        for i in 0..Table::LEN {
            pages += pt.get(i).pages();
        }
        pages
    }
}

impl AllocTables {
    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// ```
    fn pt(&self, layer: usize, page: usize) -> &Table<Entry> {
        assert!((4..=Table::LAYERS).contains(&layer));

        let i = page >> (Table::LEN_BITS * layer);
        let offset: usize = (layer..Table::LAYERS)
            .map(|i| Table::num_pts(i, self.pages))
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
            .map(|i| Table::num_pts(i, self.pages))
            .sum();
        unsafe { &*(&self.tables[offset + i] as *const _ as *const Table<Entry3>) }
    }

    fn recover_rec(&self, layer: usize, start: usize, deep: bool) -> Result<usize> {
        let mut pages = 0;
        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages) {
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

        for i in Table::range(3, start..self.pages) {
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
                let max = self.pages - page;
                if pt.update(i, |v| v.reserve(size, max)).is_ok() {
                    return Ok(page);
                }
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn unreserve_pt2(&self, start: usize) {
        let pt = self.pt3(start);
        let i = Table::idx(3, start);
        if pt.update(i, Entry3::unreserve).is_err() {
            panic!("Unreserve failed")
        }
    }

    fn increment_parents(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        if layer <= 3 {
            return self.increment_parents_l3(start, size);
        }

        // Increment parents
        let pt = self.pt(layer, start);
        for page in Table::iterate(layer, start, self.pages) {
            let i = Table::idx(layer, page);

            let max = self.pages - Table::round(layer, page);
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
        for page in Table::iterate(3, start, self.pages) {
            let i = Table::idx(3, page);

            let max = self.pages - Table::round(2, page);
            if page == start {
                if pt.update(i, |v| v.inc(size, max)).is_ok() {
                    return Ok(page);
                }
                warn!("try reserve next {i}");
                pt.update(i, Entry3::unreserve).unwrap();
            } else if pt.update(i, |v| v.inc_reserve(size, max)).is_ok() {
                warn!("reserved {i}");
                return Ok(page);
            }
        }
        Err(Error::Memory)
    }

    fn get_giant(&self, core: usize, layer: usize, start: usize) -> Result<usize> {
        info!("alloc l{}, s={}", layer, start);

        if layer == 3 {
            return self.get_giant_page(core, start);
        }

        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages) {
            info!("get giant l{} i{}", layer, i);

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

            if pt
                .update(i, |pte| pte.inc(Size::L2, layer, self.pages - page))
                .is_err()
            {
                warn!("giant update failed");
                continue;
            }

            return self.get_giant(core, layer - 1, page);
        }

        error!("Nothing found l{} s={}", layer, start);
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn get_giant_page(&self, core: usize, start: usize) -> Result<usize> {
        let pt = self.pt3(start);

        for i in Table::range(3, start..self.pages) {
            if pt.cas(i, Entry3::new(), Entry3::new_giant()).is_ok() {
                let page = Table::page(3, start, i);
                warn!("allocated l3 i{} p={} s={}", i, page, start);
                self.local[core].persist(page);
                return Ok(page);
            }
        }
        error!("Nothing found s={}", start);
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

    #[allow(dead_code)]
    pub fn dump(&self) {
        self.dump_rec(Table::LAYERS, 0);
    }

    fn dump_rec(&self, layer: usize, start: usize) {
        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages) {
            let start = Table::page(layer, start, i);

            let pte = pt.get(i);

            info!(
                "{:1$}l{5} i={2} 0x{3:x}: {4:?}",
                "",
                (Table::LAYERS - layer) * 4,
                i,
                start * Page::SIZE,
                pte,
                layer
            );

            if pte.pages() > 0 {
                if layer > 3 {
                    self.dump_rec(layer - 1, start);
                } else {
                    self.dump_l3(start);
                }
            }
        }
    }

    fn dump_l3(&self, start: usize) {
        let pt = self.pt3(start);
        for i in Table::range(3, start..self.pages) {
            let start = Table::page(3, start, i);

            let pte = pt.get(i);

            info!(
                "{:1$}l3 i={2} 0x{3:x}: {4:?}",
                "",
                (Table::LAYERS - 3) * 4,
                i,
                start * Page::SIZE,
                pte,
            );

            match pte.size() {
                Some(Size::L0 | Size::L1) if pte.pages() > 0 => self.local[0].dump(start),
                _ => {}
            }
        }
    }
}

impl Drop for AllocTables {
    fn drop(&mut self) {
        for local in &self.local {
            self.unreserve_pt2(local.start_l0.load(Ordering::SeqCst));
            self.unreserve_pt2(local.start_l1.load(Ordering::SeqCst));
        }
        self.meta().active.store(0, Ordering::SeqCst);
    }
}
