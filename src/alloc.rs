//! Simple reduced non-volatile memory allocator.

use std::alloc::{alloc_zeroed, Layout};
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use log::{error, info, warn};
use static_assertions::const_assert;

use crate::paging::{Entry, Table, LAYERS, PAGE_SIZE, PT_LEN, PT_LEN_BITS};
use crate::util::align_down;
use crate::util::align_up;

const MAGIC: usize = 0xdeadbeef;
pub const MIN_SIZE: usize = Table::span(2) * 2;
pub const MAX_SIZE: usize = Table::span(LAYERS);

/// Volatile per thread metadata
pub struct Allocator {
    begin: usize,
    pages: usize,
    volatile: *mut Table,
    small_start: usize,
    large_start: usize,
}

static VOLATILE: AtomicPtr<Table> = AtomicPtr::new(std::ptr::null_mut());

/// Non-Volatile global metadata
pub struct Meta {
    magic: AtomicUsize,
    length: AtomicUsize,
}
const_assert!(size_of::<Meta>() <= PAGE_SIZE);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory,
    /// Failed comapare and swap operation
    CAS,
    /// Invalid address
    Address,
    /// Corrupted allocator state
    Corruption,
    /// Allocator not initialized
    Uninitialized,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Size {
    Page = 0, // 4KiB
    L1 = 1,   // 2MiB
    L2 = 2,   // 1GiB
}

impl Allocator {
    /// Returns the metadata page, that contains size information and checksums
    pub fn meta(&self) -> &Meta {
        unsafe { &mut *((self.begin + self.pages * PAGE_SIZE) as *mut Meta) }
    }

    /// Returns the root page table. Shorthand for `self.pt(LAYERS, 0)`.
    fn root(&self) -> &Table {
        unsafe { &mut *self.volatile }
    }

    /// Returns the number of allocated pages.
    pub fn allocated_pages(&self) -> usize {
        let mut pages = 0;
        for i in 0..PT_LEN {
            pages += self.root().get(i).pages();
        }
        pages
    }

    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// NVRAM: [ ... | PT2 | Meta ]
    /// ```
    fn pt(&self, layer: usize, page: usize) -> &Table {
        if layer < 2 || layer > LAYERS {
            panic!("layer has to be in 2..{}", LAYERS);
        }
        let i = page >> (PT_LEN_BITS * layer);
        if layer == 2 {
            // Located in NVRAM
            // The l2 tables are stored after this area
            let pt2 = (self.begin + self.pages * PAGE_SIZE) as *mut Table;
            unsafe { &mut *pt2.add(i) }
        } else {
            // Located in DRAM
            let mut offset = 0;
            for i in layer..LAYERS {
                let span = Table::p_span(i);
                offset += (self.pages + span - 1) / span;
            }
            unsafe { &mut *self.volatile.add(offset + i) }
        }
    }

    fn page_alloc(&self) -> PageAllocator {
        PageAllocator::new(self.begin, self.pages)
    }

    /// Allows init from multiple threads.
    pub fn init(begin: usize, length: usize) -> Result<Allocator> {
        let end = align_down(begin + length, PAGE_SIZE);
        let begin = align_up(begin, PAGE_SIZE);
        if begin + MIN_SIZE > end {
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        let length = (end - begin - PAGE_SIZE).min(MAX_SIZE);
        info!(
            "Alloc: {:?}-{:?} - {} pages",
            begin as *const (),
            (begin + length) as *const (),
            length / PAGE_SIZE
        );

        let meta = unsafe { &mut *((begin + length) as *mut Meta) };

        let volatile = VOLATILE.load(Ordering::SeqCst);
        if !volatile.is_null() {
            let pages = length / PAGE_SIZE;
            let num_pt2 = (pages + (PT_LEN * PT_LEN) - 1) / (PT_LEN * PT_LEN);
            // Remaining number of pages
            let pages = pages - num_pt2;
            let mut alloc = Allocator {
                begin,
                pages,
                volatile,
                small_start: 0,
                large_start: 0,
            };
            alloc.small_start = alloc.reserve_pt2(LAYERS, 0).unwrap();
            alloc.large_start = alloc.reserve_pt2(LAYERS, alloc.small_start).unwrap();
            info!("Alloc already initialized small={} large={}", alloc.small_start, alloc.large_start);
            return Ok(alloc);
        }

        if meta.length.load(Ordering::SeqCst) == length
            && meta.magic.load(Ordering::SeqCst) == MAGIC
            && false
        {
            // TODO: check if power was lost and recovery is necessary
            info!("Found allocator state. Recovery...");
            Self::recover(begin, length)
        } else {
            info!("Create new allocator state.");
            let alloc = Self::setup(begin, length)?;
            meta.length.store(length, Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
            VOLATILE.store(alloc.volatile, Ordering::SeqCst);
            Ok(alloc)
        }
    }

    fn setup(begin: usize, length: usize) -> Result<Allocator> {
        let pages = length / PAGE_SIZE;
        let num_pt2 = (pages + (PT_LEN * PT_LEN) - 1) / (PT_LEN * PT_LEN);
        let num_pt1 = ((pages - num_pt2) + PT_LEN - 1) / PT_LEN;
        let pt2 = (begin + length - (num_pt2 * PAGE_SIZE)) as *mut Table;
        // Remaining number pages
        let pages = pages - num_pt2;

        info!(
            "pages={}, #pt2={}, #pt1={}, area=[0x{:x}|{:?}-0x{:x}]",
            pages,
            num_pt2,
            num_pt1,
            begin,
            pt2,
            begin + length
        );

        // pt1's are initialized on demand

        info!("Init pt2");

        // Init pt2
        for i in 0..num_pt2 {
            let pt2 = unsafe { &*pt2.add(i) };
            pt2.clear();
        }

        let mut higher_level_pts = 0;
        for i in 3..=LAYERS {
            let span = Table::p_span(i);
            higher_level_pts += (pages + span - 1) / span;
        }

        info!("#higher level pts = {}", higher_level_pts);

        // the high level page table are initialized with zero
        // -> all counters and flags are zero
        let volatile = unsafe {
            alloc_zeroed(Layout::from_size_align_unchecked(
                higher_level_pts * PAGE_SIZE,
                PAGE_SIZE,
            ))
        } as *mut Table;
        if volatile.is_null() {
            return Err(Error::Memory);
        }

        let alloc = Allocator {
            begin,
            pages,
            volatile,
            small_start: 0,
            large_start: Table::p_span(2),
        };
        alloc.pt(3, alloc.small_start).set(
            Table::p_idx(3, alloc.small_start),
            Entry::table(0, 0, 0, true),
        );
        alloc.pt(3, alloc.large_start).set(
            Table::p_idx(3, alloc.large_start),
            Entry::table(0, 0, 0, true),
        );

        info!(
            "reserved small={} ({:?}), large={} ({:?})",
            alloc.small_start,
            alloc
                .pt(3, alloc.small_start)
                .get(Table::p_idx(3, alloc.small_start)),
            alloc.large_start,
            alloc
                .pt(3, alloc.large_start)
                .get(Table::p_idx(3, alloc.large_start))
        );

        Ok(alloc)
    }

    fn recover(begin: usize, length: usize) -> Result<Allocator> {
        let pages = length / PAGE_SIZE;
        let num_pt2 = (pages + Table::p_span(2) - 1) / Table::p_span(2);
        let pages = pages - num_pt2;

        let mut higher_level_pts = 0;
        for i in 3..=LAYERS {
            let span = Table::p_span(i);
            higher_level_pts += (pages + span - 1) / span;
        }
        info!("#higher level pts = {}", higher_level_pts);

        // Init ptn - pt3
        let volatile = unsafe {
            alloc_zeroed(Layout::from_size_align_unchecked(
                higher_level_pts * PAGE_SIZE,
                PAGE_SIZE,
            ))
        } as *mut Table;
        if volatile.is_null() {
            return Err(Error::Memory);
        }

        let mut alloc = Allocator {
            begin,
            pages,
            volatile,
            small_start: 0,
            large_start: 0,
        };

        // TODO recreate ptn-pt3 mapping
        let (pages, nonempty) = alloc.recover_rec(alloc.root(), LAYERS, 0);

        info!("Recovered pages={}, nonempty={}", pages, nonempty);

        alloc.small_start = alloc.reserve_pt2(LAYERS, 0).unwrap();
        alloc.large_start = alloc.reserve_pt2(LAYERS, alloc.small_start).unwrap();

        Ok(alloc)
    }

    fn recover_rec(&self, pt: &Table, layer: usize, start: usize) -> (usize, usize) {
        let mut pages = 0;
        let mut nonemtpy = 0;

        for i in 0..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                return (pages, nonemtpy);
            }

            if layer > 2 {
                let child_pt = self.pt(layer - 1, start);
                let (child_pages, child_nonempty) = self.recover_rec(child_pt, layer - 1, start);

                if child_pages > 0 {
                    pt.set(i, Entry::table(child_pages, child_nonempty, 0, false));
                    nonemtpy += 1;
                } else {
                    pt.set(i, Entry::empty());
                }
                pages += child_pages;
            } else {
                let pte = pt.get(i);
                pages += pte.pages();
                if pte.pages() > 0 {
                    nonemtpy += 1;
                }
            }
        }

        info!(
            "recovered pt{}={:?} p={} n={}",
            layer, pt as *const _, pages, nonemtpy
        );

        (pages, nonemtpy)
    }

    fn reserve_new(&mut self, size: Size) -> Result<usize> {
        let result = self.reserve_pt2(LAYERS, 0)?;
        info!("reserved new {} ({:?})", result, size);
        match size {
            Size::Page => {
                self.unreserve_pt2(self.small_start);
                self.small_start = result;
            }
            Size::L1 => {
                self.unreserve_pt2(self.large_start);
                self.large_start = result;
            }
            _ => panic!("invalid reserve size"),
        }
        Ok(result)
    }

    fn reserve_pt2(&self, layer: usize, start: usize) -> Result<usize> {
        let pt = self.pt(layer, start);
        for i in Table::p_idx(layer, start)..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                break;
            }

            if layer == 3 {
                let pte3 = pt.get(i);
                if pte3.is_reserved() || pte3.pages() >= Table::p_span(layer - 1) {
                    continue;
                }

                match pt.reserve(i, true) {
                    Ok(_) => return Ok(start),
                    Err(_) => {}
                }
            } else {
                if let Ok(result) = self.reserve_pt2(layer - 1, start) {
                    return Ok(result);
                }
            }
        }
        Err(Error::Memory)
    }

    fn unreserve_pt2(&self, start: usize) {
        let pt = self.pt(3, start);
        let i = Table::p_idx(3, start);
        if let Err(_) = pt.reserve(i, false) {
            error!("Unreserve failed! {}", start);
            panic!()
        }
    }

    /// Search free page table entry.
    fn alloc_child_page(&self, layer: usize, start: usize) -> Result<(usize, bool)> {
        let pt = self.pt(layer, start);

        for i in 0..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                break;
            }

            if let Ok(_) = pt.insert_page(i) {
                return Ok((start, true));
            }
        }
        Err(Error::Memory)
    }

    fn alloc(&self, layer: usize, size: Size, start: usize) -> Result<(usize, bool)> {
        assert!(layer > 1);
        assert!((size as usize) < layer);
        assert!(start < self.pages);

        info!("alloc l{}, s={}", layer, start);

        if layer == 2 && size == Size::Page {
            return self.page_alloc().alloc(start);
        }

        let pt = self.pt(layer, start);
        if layer - 1 == size as usize {
            return self.alloc_child_page(layer, start);
        }

        assert!(layer > 2);

        for i in 0..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                break;
            }

            let pte = pt.get(i);

            if pte.is_page() {
                continue;
            }

            // Large / Huge Pages or enough pages in child pt
            if (size as usize == layer - 2 && pte.nonempty() < PT_LEN)
                || ((size as usize) < layer - 2
                    && Table::p_span(size as usize) <= Table::p_span(layer - 1) - pte.pages())
            {
                if let Ok((page, newentry)) = self.alloc(layer - 1, size, start) {
                    match pt.inc(i, Table::p_span(size as _), newentry as _) {
                        Ok(pte) => return Ok((page, pte.pages() == 0)),
                        Err(pte) => {
                            error!("CAS: inc failed {:?}", pte);
                            return Err(Error::Corruption);
                        }
                    }
                }
            }
        }

        Err(Error::Memory)
    }

    pub fn get<F: FnOnce(u64) -> u64>(
        &mut self,
        size: Size,
        dst: &AtomicU64,
        translate: F,
        expected: u64,
    ) -> Result<()> {
        let mut start = match size {
            Size::Page => self.small_start,
            Size::L1 => self.large_start,
            _ => panic!("Huge pages are currently not supported!"),
        };

        let (page, mut newentry) = loop {
            // TODO: check pte3 pages / entries & reserve next before alloc

            match self.alloc(2, size, start) {
                Ok(result) => break result,
                Err(Error::CAS) => warn!("CAS: retry alloc"),
                Err(Error::Memory) => {
                    warn!("MEM: reserve & retry alloc");
                    start = self.reserve_new(size)?;
                }
                Err(e) => return Err(e),
            }
        };

        // Increment parents
        for layer in 3..=LAYERS {
            let pt = self.pt(layer, page);
            let i = Table::p_idx(layer, page);
            match pt.inc(i, Table::p_span(size as _), newentry as _) {
                Ok(pte) => newentry = pte.pages() == 0,
                Err(pte) => {
                    error!("CAS: inc failed {:?}", pte);
                    return Err(Error::Corruption);
                }
            }
        }

        let addr = (page * PAGE_SIZE) as u64 + self.begin as u64;
        let new = translate(addr);
        match dst.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => {
                error!("CAS: get failed -> free");
                self.free(LAYERS, size, page).unwrap();
                Err(Error::CAS)
            }
        }
    }

    fn free(&self, layer: usize, size: Size, page: usize) -> Result<bool> {
        if size as usize >= layer {
            return Err(Error::Memory);
        }
        let pt = self.pt(layer, page);
        let i = Table::p_idx(layer, page);

        info!("free l{} i{} p={}", layer, i, page);

        // TODO: create pt1 after freeing large/huge pages!
        // Or create pt1 on demand in alloc

        if (size as usize) < layer - 1 {
            if layer == 2 {
                return self.page_alloc().free(page);
            }

            let pte = pt.get(i);

            if pte.is_page() {
                error!("No table found l{} {:?}", layer, pte);
                return Err(Error::Address);
            }

            let cleared = self.free(layer - 1, size, page)?;

            match pt.dec(i, Table::p_span(size as usize), cleared as _) {
                Ok(pte) => {
                    info!(
                        "free dec l{} i{} {:?} -{} -{}",
                        layer,
                        i,
                        pte,
                        Table::p_span(size as usize),
                        cleared as usize
                    );
                    Ok(pte.pages() == Table::p_span(size as usize))
                }
                Err(pte) => {
                    error!("CAS: free dec l{} {:?}", layer, pte);
                    Err(Error::Corruption)
                }
            }
        } else {
            info!("free l{} i={}", layer, i);
            match pt.cas(i, Entry::page(), Entry::empty()) {
                Ok(_) => Ok(true),
                Err(pte) => {
                    error!("CAS: free unexpected {:?}", pte);
                    Err(Error::Address)
                }
            }
        }
    }

    pub fn put(&mut self, addr: u64, size: Size) -> Result<()> {
        if size > Size::L1 {
            panic!("Huge pages are currently not supported!");
        }

        let addr = addr as usize;

        if addr % Table::span(size as usize) != 0
            || addr < self.begin
            || addr >= self.begin + self.pages * PAGE_SIZE
        {
            return Err(Error::Address);
        }
        let page = (addr - self.begin) / PAGE_SIZE;

        loop {
            match self.free(LAYERS, size, page) {
                Err(Error::CAS) => warn!("CAS: retry free"),
                Err(e) => return Err(e),
                Ok(_) => break,
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn dump(&self) {
        self.dump_rec(LAYERS, 0);
    }

    fn dump_rec(&self, layer: usize, start: usize) {
        let pt = self.pt(layer, start);
        for i in 0..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                return;
            }

            let pte = pt.get(i);
            info!(
                "{:1$}l{5} i={2} 0x{3:x}: {4:?}",
                "",
                (LAYERS - layer) * 4,
                i,
                start * PAGE_SIZE,
                pte,
                layer
            );

            if !pte.is_page() && pte.pages() > 0 {
                if layer > 2 {
                    self.dump_rec(layer - 1, start);
                } else {
                    self.page_alloc().dump(start);
                }
            }
        }
    }
}

/// Layer 2 page allocator.
struct PageAllocator {
    begin: usize,
    pages: usize,
}

impl PageAllocator {
    fn new(begin: usize, pages: usize) -> Self {
        Self { begin, pages }
    }

    /// Returns the l1 page table that contains the `page`.
    fn pt1(&self, pte2: Entry, page: usize) -> Option<&Table> {
        if pte2.is_table() && pte2.pages() < PT_LEN {
            let start = page & !(Table::p_span(1) - 1);
            Some(unsafe { &*((self.begin + (start + pte2.i1()) * PAGE_SIZE) as *const Table) })
        } else {
            None
        }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ ... | PT2 | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table {
        let i = page >> (PT_LEN_BITS * 2);
        // The l2 tables are stored after this area
        let pt2 = (self.begin + self.pages * PAGE_SIZE) as *mut Table;
        unsafe { &mut *pt2.add(i) }
    }

    /// Allocate a single page
    fn alloc(&self, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);

        for i2 in 0..PT_LEN {
            let start = start + i2 * PT_LEN;
            let pte2 = pt2.get(i2);

            if pte2.pages() >= PT_LEN || pte2.is_reserved() {
                continue;
            }

            if pte2.pages() == PT_LEN - 1 {
                return self.alloc_last(pte2, start);
            }

            if let Some(pt1) = self.pt1(pte2, start) {
                return self.alloc_pt(pt1, start);
            }

            assert!(pte2.is_empty());
            return self.alloc_first(start);
        }
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn alloc_pt(&self, pt1: &Table, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);

        for i in 0..PT_LEN {
            let page = start + i;
            if page >= self.pages {
                break;
            }
            if let Ok(_) = pt1.cas(i, Entry::empty(), Entry::page()) {
                info!("alloc l1 i={}: {}", i, page);

                let i2 = Table::p_idx(2, page);
                info!("alloc inc l2 i{} {:?} +1 +1", i2, pt2.get(i2));
                return match pt2.inc(i2, 1, 1) {
                    Ok(pte) => Ok((page, pte.pages() == 0)),
                    Err(pte) => {
                        error!("CAS: inc alloc pt {:?}", pte);
                        Err(Error::Memory)
                    }
                };
            }
        }
        Err(Error::Memory)
    }

    /// Split and alloc new page table.
    fn alloc_first(&self, start: usize) -> Result<(usize, bool)> {
        info!("alloc init pt1 s={}", start);
        assert!(start % PT_LEN == 0);

        let pt2 = self.pt2(start);
        let i2 = Table::p_idx(2, start);

        // store pt1 at i=1, so that the allocated page is at i=0
        let pt1 = unsafe { &*((self.begin + (start + 1) * PAGE_SIZE) as *const Table) };
        pt1.set(0, Entry::page());
        pt1.set(1, Entry::page_reserved());
        for i1 in 2..PT_LEN {
            pt1.set(i1, Entry::empty());
        }

        match pt2.cas(i2, Entry::empty(), Entry::table(1, 1, 1, false)) {
            Ok(_) => Ok((start, true)),
            Err(pte) => {
                warn!("CAS: init pt1 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    /// Allocate the last page (the pt1 is reused as last page).
    fn alloc_last(&self, pte2: Entry, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);
        let i2 = Table::p_idx(2, start);

        assert!(pte2.is_table() && pte2.pages() == PT_LEN - 1);

        info!("alloc last {} s={}", pte2.i1(), start);

        // Only modify pt2 as pt1 is returned as page.
        match pt2.cas(i2, pte2, Entry::table(PT_LEN, PT_LEN, 0, false)) {
            Ok(_) => Ok((start + pte2.i1(), false)),
            Err(pte) => {
                warn!("CAS: alloc last pt2 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    /// Free single page
    fn free(&self, page: usize) -> Result<bool> {
        info!("free leaf page {}", page);

        let pt2 = self.pt2(page);
        let i2 = Table::p_idx(2, page);

        let pte2 = pt2.get(i2);
        if pte2.is_page() {
            return Err(Error::Address);
        } else if pte2.pages() < PT_LEN {
            if let Some(pt1) = self.pt1(pte2, page) {
                let i1 = Table::p_idx(1, page);
                info!("free l1 i={}", i1);

                match pt1.cas(i1, Entry::page(), Entry::empty()) {
                    Ok(_) => {}
                    Err(pte) => {
                        error!("CAS: free unexpected l1 {:?}", pte);
                        return Err(Error::Address);
                    }
                }
            } else {
                error!("free no pt1");
                return Err(Error::Address);
            }

            // i2 is expected to be unchanged.
            // If not the page table has been moved in between.
            match pt2.cas(
                i2,
                pte2,
                Entry::table(pte2.pages() - 1, pte2.nonempty() - 1, pte2.i1(), false),
            ) {
                Ok(pte) => Ok(pte.pages() == 1),
                Err(pte) => {
                    warn!("CAS: free dec l2 {:?}", pte);
                    Err(Error::CAS)
                }
            }
        } else {
            // free last page of pt1 & rebuild pt1
            self.free_last(page)
        }
    }

    /// Free last page & rebuild pt1 in it
    fn free_last(&self, page: usize) -> Result<bool> {
        // The new pt
        let pt2 = self.pt2(page);
        let i = Table::p_idx(2, page);

        let pt1 = unsafe { &*((self.begin + page * PAGE_SIZE) as *const Table) };
        info!("free: init last pt1 {}", page);

        for j in 0..PT_LEN {
            if j == page % PT_LEN {
                pt1.set(j, Entry::page_reserved());
            } else {
                pt1.set(j, Entry::page());
            }
        }

        match pt2.cas(
            i,
            Entry::table(PT_LEN, PT_LEN, 0, false),
            Entry::table(PT_LEN - 1, PT_LEN - 1, page % PT_LEN, false),
        ) {
            Ok(_) => Ok(false),
            Err(pte) => {
                warn!("CAS: create pt1 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    fn dump(&self, start: usize) {
        let pt2 = self.pt2(start);
        for i2 in 0..PT_LEN {
            let start = start + i2 * PT_LEN;
            if start >= self.pages {
                return;
            }

            let pte2 = pt2.get(i2);
            info!(
                "{:1$}l2 i={2} 0x{3:x}: {4:?}",
                "",
                (LAYERS - 2) * 4,
                i2,
                start * PAGE_SIZE,
                pte2
            );
            if let Some(pt1) = self.pt1(pte2, start) {
                for i1 in 0..PT_LEN {
                    let pte1 = pt1.get(i1);
                    info!(
                        "{:1$}l1 i={2} 0x{3:x}: {4:?}",
                        "",
                        (LAYERS - 1) * 4,
                        i1,
                        (start + i1) * PAGE_SIZE,
                        pte1
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use core::slice;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use log::{info, warn};

    use crate::alloc::{Error, Size};
    use crate::util::logging;
    use crate::mmap::c_mmap_anon;
    use crate::paging::{PAGE_SIZE, PT_LEN};

    use super::{Allocator, MAX_SIZE};

    #[test]
    fn init() {
        logging();
        // 8GiB
        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, 8 << 30) };
        c_mmap_anon(data).unwrap();

        info!("init alloc");

        let mut alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        warn!("start alloc...");

        let small = AtomicU64::new(0);
        alloc.get(Size::Page, &small, |v| v, 0).unwrap();

        let large = AtomicU64::new(0);
        alloc.get(Size::L1, &large, |v| v, 0).unwrap();

        // Unexpected value
        let small = AtomicU64::new(5);
        assert_eq!(alloc.get(Size::Page, &small, |v| v, 0), Err(Error::CAS));

        // Stress test
        let mut pages = Vec::with_capacity(PT_LEN * PT_LEN);
        pages.resize_with(PT_LEN * PT_LEN, AtomicU64::default);
        for page in &pages {
            alloc.get(Size::Page, page, |v| v, 0).unwrap();
        }

        // alloc.dump();
        alloc.page_alloc().dump(0);

        warn!("check...");

        assert_eq!(alloc.allocated_pages(), 1 + PT_LEN + pages.len());

        pages.sort_unstable_by(|a, b| a.load(Ordering::Relaxed).cmp(&b.load(Ordering::Relaxed)));

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i].load(Ordering::Relaxed) as *mut u8;
            let p2 = pages[i + 1].load(Ordering::Relaxed) as *mut u8;
            info!("addr {}={:?}", i, p1);
            assert!(p1 as usize % PAGE_SIZE == 0 && data.contains(unsafe { &mut *p1 }));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        for page in &pages[10..PT_LEN + 10] {
            let addr = page.swap(0, Ordering::SeqCst);
            alloc.put(addr, Size::Page).unwrap();
        }

        alloc.put(large.load(Ordering::SeqCst), Size::L1).unwrap();

        // Realloc
        for page in &pages[10..PT_LEN + 10] {
            alloc.get(Size::Page, page, |v| v, 0).unwrap();
        }

        warn!("free...");

        // Free all
        for page in &pages {
            let addr = page.swap(0, Ordering::SeqCst);
            alloc.put(addr, Size::Page).unwrap();
        }
    }

    #[test]
    fn parallel_alloc() {
        logging();

        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, 20 << 30) };
        c_mmap_anon(data).unwrap();

        info!("init alloc");

        let alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        // Stress test
        const ALLOC_PER_THREAD: usize = PT_LEN * 2;
        const THREADS: usize = 4;
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let pages = [DEFAULT; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));

        let handles = (0..THREADS)
            .into_iter()
            .map(|t| {
                let pages_begin = pages.as_ptr() as usize;
                let begin = data.as_ptr() as usize;
                let size = data.len();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let mut alloc = Allocator::init(begin, size).unwrap();
                    barrier.wait();

                    for i in 0..ALLOC_PER_THREAD {
                        let dst = unsafe {
                            &*(pages_begin as *const AtomicU64).add(t * ALLOC_PER_THREAD + i)
                        };
                        alloc.get(Size::Page, dst, |v| v, 0).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        thread::sleep(Duration::from_millis(1000));

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(alloc.allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        for i in 0..pages.len() {
            let p1 = pages[i].load(Ordering::SeqCst) as *mut u8;
            info!("addr {}={:?}", i, p1);
            assert!(p1 as usize % PAGE_SIZE == 0 && data.contains(unsafe { &mut *p1 }));
            for j in (i + 1)..pages.len() {
                let p2 = pages[j].load(Ordering::SeqCst) as *mut u8;
                assert_ne!(
                    p1,
                    p2,
                    "{}=={} ({})",
                    i,
                    j,
                    ((p1 as usize) - alloc.begin) / PAGE_SIZE
                );
            }
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, 20 << 30) };
        c_mmap_anon(data).unwrap();

        info!("init alloc");

        let alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        // Stress test
        const ALLOC_PER_THREAD: usize = PT_LEN * 2;
        const THREADS: usize = 4;
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let pages = [DEFAULT; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));

        let handles = (0..THREADS)
            .into_iter()
            .map(|t| {
                let pages_begin = pages.as_ptr() as usize;
                let begin = data.as_ptr() as usize;
                let size = data.len();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let mut alloc = Allocator::init(begin, size).unwrap();
                    warn!("t{} wait", t);
                    barrier.wait();

                    for i in 0..ALLOC_PER_THREAD {
                        let dst = unsafe {
                            &*(pages_begin as *const AtomicU64).add(t * ALLOC_PER_THREAD + i)
                        };
                        alloc.get(Size::Page, dst, |v| v, 0).unwrap();
                    }

                    for i in 0..ALLOC_PER_THREAD {
                        let dst = unsafe {
                            &*(pages_begin as *const AtomicU64).add(t * ALLOC_PER_THREAD + i)
                        };
                        alloc.put(dst.load(Ordering::SeqCst), Size::Page).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(alloc.allocated_pages(), 0);
    }

    #[test]
    fn last_page() {
        logging();

        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, 8 << 30) };

        info!("mmap {} bytes", data.len());
        c_mmap_anon(data).unwrap();

        let mut alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let pages = Arc::new([DEFAULT; PT_LEN]);
        let barrier = Arc::new(Barrier::new(2));

        for page in pages.iter() {
            alloc.get(Size::Page, page, |v| v, 0).unwrap();
        }

        let handle = {
            let pages = pages.clone();
            let barrier = barrier.clone();
            let begin = data.as_ptr() as usize;
            let size = data.len();

            thread::spawn(move || {
                let mut alloc = Allocator::init(begin, size).unwrap();

                barrier.wait();
                for _ in 0..10 {
                    let addr = pages[14].load(Ordering::SeqCst);
                    alloc.put(addr, Size::Page).unwrap();
                    alloc.get(Size::Page, &pages[14], |v| v, addr).unwrap();
                }
            })
        };

        barrier.wait();
        for _ in 0..10 {
            let addr = pages[13].load(Ordering::SeqCst);
            alloc.put(addr, Size::Page).unwrap();
            alloc.get(Size::Page, &pages[13], |v| v, addr).unwrap();
        }

        handle.join().unwrap();

        alloc.dump();
    }
}
