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
    Memory,
    CAS,
    Address,
    Uninitialized,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ChunkSize {
    Page = 0, // 4KiB
    L1 = 1,   // 2MiB
    L2 = 2,   // 1GiB
}

#[derive(Debug, Clone)]
struct SearchResult {
    size: ChunkSize,
    last_page: bool,
    page: usize,
}

impl SearchResult {
    fn new(size: ChunkSize, page: usize, last_page: bool) -> Self {
        Self {
            size,
            page,
            last_page,
        }
    }
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

    fn l2(&self) -> L2Alloc {
        L2Alloc::at(self.begin, self.pages)
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
            info!("Alloc already initialized");
            let pages = length / PAGE_SIZE;
            let num_pt2 = (pages + (PT_LEN * PT_LEN) - 1) / (PT_LEN * PT_LEN);
            // Remaining number of pages
            let pages = pages - num_pt2;
            return Ok(Allocator {
                begin,
                pages,
                volatile,
            });
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

        Ok(Allocator {
            begin,
            pages,
            volatile,
        })
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

        let alloc = Allocator {
            begin,
            pages,
            volatile,
        };

        // TODO recreate ptn-pt3 mapping
        let (pages, nonempty) = alloc.recover_rec(alloc.root(), LAYERS, 0);

        info!("Recovered pages={}, nonempty={}", pages, nonempty);

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

    /// Search free page table entry.
    fn search_pt_page(&self, layer: usize, start: usize) -> Result<usize> {
        let pt = self.pt(layer, start);
        for i in 0..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                break;
            }
            let pte = pt.get(i);
            if pte.is_empty() {
                info!("search found i={}: {} - {:?}", i, start, pte);
                return Ok(start);
            }
        }
        Err(Error::Memory)
    }

    fn search(&self, layer: usize, size: ChunkSize, start: usize) -> Result<SearchResult> {
        assert!(layer > 1);
        assert!((size as usize) < layer);
        assert!(start < self.pages);

        info!("search {:?}, l{}, s={}", size, layer, start);

        if layer == 2 && size == ChunkSize::Page {
            return self.l2().search(start);
        }

        let pt = self.pt(layer, start);
        if layer - 1 == size as usize {
            let page = self.search_pt_page(layer, start)?;
            return Ok(SearchResult::new(size, page, false));
        }

        assert!(layer > 2);

        for i in 0..PT_LEN {
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                return Err(Error::Memory);
            }

            let pte = pt.get(i);

            if pte.is_page() {
                continue;
            }

            if size as usize == layer - 2 {
                // Large / Huge Pages
                if pte.nonempty() < PT_LEN {
                    if let Ok(result) = self.search(layer - 1, size, start) {
                        return Ok(result);
                    }
                }
            } else if Table::p_span(size as usize) <= Table::p_span(layer - 1) - pte.pages() {
                // Enough pages in child pt
                if let Ok(result) = self.search(layer - 1, size, start) {
                    return Ok(result);
                }
            }
        }

        Err(Error::Memory)
    }

    fn alloc(&self, layer: usize, target: SearchResult) -> Result<bool> {
        if target.size as usize >= layer {
            return Err(Error::Memory);
        }
        let pt = self.pt(layer, target.page);
        let i = Table::p_idx(layer, target.page);

        info!(
            "alloc {:?} pt{}={:?} i={} p={}",
            target.size, layer, pt as *const _, i, target.page
        );

        if (target.size as usize) < layer - 1 {
            if layer == 2 {
                return self.l2().alloc(target);
            }

            let newentry = self.alloc(layer - 1, target.clone())?;

            match pt.inc(i, Table::p_span(target.size as usize), newentry as _) {
                Ok(pte) => Ok(pte.is_empty()),
                Err(pte) => {
                    error!("CAS: inc failed {:?}", pte);
                    Err(Error::Memory)
                }
            }
        } else {
            info!("alloc l{} i={}", layer, i);

            match pt.insert_page(i) {
                Ok(_) => Ok(true),
                Err(pte) => {
                    warn!("CAS: alloc unexpected {:?}", pte);
                    Err(Error::CAS)
                }
            }
        }
    }

    pub fn get<F: FnOnce(u64) -> u64>(
        &mut self,
        size: ChunkSize,
        dst: &AtomicU64,
        translate: F,
        expected: u64,
    ) -> Result<()> {
        if size > ChunkSize::L1 {
            panic!("Huge pages are currently not supported!");
        }

        let mut target;
        loop {
            target = self.search(LAYERS, size, 0)?;
            info!("page found {}, last={:?}", target.page, target.last_page);

            match self.alloc(LAYERS, target.clone()) {
                Ok(_) => break,
                Err(Error::CAS) => warn!("CAS: retry alloc"),
                Err(e) => return Err(e),
            }
        }

        let addr = (target.page * PAGE_SIZE) as u64 + self.begin as u64;
        let new = translate(addr);
        match dst.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => {
                error!("CAS: get failed -> free");
                self.free(LAYERS, size, target.page).unwrap();
                Err(Error::CAS)
            }
        }
    }

    fn free(&self, layer: usize, size: ChunkSize, page: usize) -> Result<bool> {
        if size as usize >= layer {
            return Err(Error::Memory);
        }
        let pt = self.pt(layer, page);
        let i = Table::p_idx(layer, page);

        info!(
            "free {:?} pt{}={:?} i={} p={}",
            size, layer, pt as *const _, i, page
        );

        // TODO: create pt1 after freeing large/huge pages!
        // Or create pt1 on demand in alloc

        if (size as usize) < layer - 1 {
            if layer == 2 {
                return self.l2().free(page);
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
                        "free dec l{} i={} pages={} cleared={} from={:?}",
                        layer,
                        i,
                        Table::p_span(size as usize),
                        cleared as usize,
                        pte
                    );
                    Ok(pte.pages() == Table::p_span(size as usize))
                }
                Err(pte) => {
                    error!("CAS: free dec l{} {:?}", layer, pte);
                    Err(Error::Address)
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

    pub fn put(&mut self, addr: u64, size: ChunkSize) -> Result<()> {
        if size > ChunkSize::L1 {
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
        self.dump_rec(self.root(), LAYERS, 0);
    }

    fn dump_rec(&self, pt: &Table, layer: usize, start: usize) {
        for i in 0..PT_LEN {
            let pte = pt.get(i);
            let start = start + i * Table::p_span(layer - 1);
            if start >= self.pages {
                return;
            }

            info!(
                "{:1$} i={2} 0x{3:x}: {4:?}",
                "",
                (LAYERS - layer) * 4,
                i,
                start * PAGE_SIZE,
                pte
            );

            if !pte.is_page() && !pte.is_empty() {
                if layer > 2 {
                    let child_pt = self.pt(layer - 1, start);
                    self.dump_rec(child_pt, layer - 1, start);
                } else {
                    self.l2().dump(start);
                }
            }
        }
    }
}

struct L2Alloc {
    begin: usize,
    pages: usize,
}

impl L2Alloc {
    fn at(begin: usize, pages: usize) -> Self {
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

    /// Search free page table entry.
    fn search_pt_page(&self, start: usize, pt: &Table) -> Result<usize> {
        for i in 0..PT_LEN {
            let start = start + i;
            if start >= self.pages {
                break;
            }
            let pte = pt.get(i);
            if pte.is_empty() {
                info!("search found i={}: {} - {:?}", i, start, pte);
                return Ok(start);
            }
        }
        Err(Error::Memory)
    }

    fn search(&self, start: usize) -> Result<SearchResult> {
        let pt2 = self.pt2(start);

        for i2 in 0..PT_LEN {
            let start = start + i2 * Table::p_span(1);
            let pte2 = pt2.get(i2);

            if pte2.pages() >= PT_LEN {
                continue;
            }

            if pte2.pages() == PT_LEN - 1 {
                info!("search pt2 full: {}, {:?}", pte2.i1(), pte2);
                return Ok(SearchResult::new(ChunkSize::Page, start + pte2.i1(), true));
            }

            if let Some(pt1) = self.pt1(pte2, start) {
                if let Ok(page) = self.search_pt_page(start, pt1) {
                    return Ok(SearchResult::new(ChunkSize::Page, page, false));
                }
            } else {
                assert!(pte2.is_empty());
                info!("search pt2 empty {:?}", pte2);
                return Ok(SearchResult::new(ChunkSize::Page, start, false));
            }
        }
        Err(Error::Memory)
    }

    /// Allocate single page
    fn alloc(&self, target: SearchResult) -> Result<bool> {
        if target.last_page {
            return self.alloc_last(target.page);
        }

        let pt2 = self.pt2(target.page);
        let i2 = Table::p_idx(2, target.page);
        let pte2 = pt2.get(i2);

        // Check before modifying pt1 as it's page allocated by the user.
        if pte2.pages() >= PT_LEN {
            warn!("CAS: page full");
            return Err(Error::CAS);
        }

        // Is there already a page table
        if let Some(pt1) = self.pt1(pte2, target.page) {
            let i1 = Table::p_idx(1, target.page);
            match pt1.cas(i1, Entry::empty(), Entry::page()) {
                Err(pte) => {
                    warn!("CAS: alloc leaf pt1 {:?} (pt2={:?})", pte, pte2);
                    return Err(Error::CAS);
                }
                _ => {}
            }
        } else if pte2.is_empty() {
            return self.alloc_first(target.page);
        } else {
            error!("alloc unexpected pte2 {:?}", pte2);
            panic!();
        }

        match pt2.inc(i2, 1, 1) {
            Ok(pte) => Ok(pte.is_empty()),
            Err(pte) => {
                error!("CAS: alloc leaf pt2 {:?}", pte);
                Err(Error::Memory)
            }
        }
    }

    /// Split and alloc new page table.
    fn alloc_first(&self, start: usize) -> Result<bool> {
        info!("alloc init pt1");

        let start = (start & !(PT_LEN - 1)) + 1;
        // store pt1 at i=1, so that the allocated page is at i=0
        let pt1 = unsafe { &*((self.begin + start * PAGE_SIZE) as *const Table) };
        pt1.set(0, Entry::page());
        pt1.set(1, Entry::page_reserved());
        for i1 in 2..PT_LEN {
            pt1.set(i1, Entry::empty());
        }

        let pt2 = self.pt2(start);
        let i2 = Table::p_idx(2, start);
        info!(
            "alloc init pt1 inc pt2={:?} i2={} s={}",
            pt2 as *const _, i2, start
        );
        match pt2.cas(i2, Entry::empty(), Entry::table(1, 1, 1, false)) {
            Ok(_) => Ok(true),
            Err(pte) => {
                warn!("CAS: init pt1 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    /// Allocate the last page (the pt1 is reused as last page).
    fn alloc_last(&self, page: usize) -> Result<bool> {
        let pt2 = self.pt2(page);
        let i2 = Table::p_idx(2, page);
        let i1 = Table::p_idx(1, page);

        // Only modify pt2 as pt1 is returned as page.
        match pt2.cas(
            i2,
            Entry::table(PT_LEN - 1, PT_LEN - 1, i1, false),
            Entry::table(PT_LEN, PT_LEN, 0, false),
        ) {
            Ok(_) => Ok(false),
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
                    error!("CAS: free dec l2 {:?}", pte);
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
            let pte2 = pt2.get(i2);
            info!(
                "{:1$}i={2} 0x{3:x}: {4:?}",
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
                        "{:1$}i={2} 0x{3:x}: {4:?}",
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

    use log::info;

    use crate::alloc::{ChunkSize, Error};
    use crate::logging;
    use crate::mmap::c_mmap_anon;
    use crate::paging::{PAGE_SIZE, PT_LEN};

    use super::{Allocator, MAX_SIZE};

    #[test]
    fn init() {
        logging();

        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, MAX_SIZE) };

        info!("mmap {} bytes", data.len());
        c_mmap_anon(data).unwrap();

        info!("init alloc");

        let mut alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        info!("get");

        let small = AtomicU64::new(0);
        alloc.get(ChunkSize::Page, &small, |v| v, 0).unwrap();

        let large = AtomicU64::new(0);
        alloc.get(ChunkSize::L1, &large, |v| v, 0).unwrap();

        // Unexpected value
        let small = AtomicU64::new(5);
        assert_eq!(
            alloc.get(ChunkSize::Page, &small, |v| v, 0),
            Err(Error::CAS)
        );

        // Stress test
        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let pages = [DEFAULT; PT_LEN * 5];
        for page in &pages {
            alloc.get(ChunkSize::Page, page, |v| v, 0).unwrap();
        }

        alloc.dump();

        assert_eq!(alloc.allocated_pages(), 1 + PT_LEN + pages.len());
        // Check that the same page was not allocated twice
        for i in 0..pages.len() {
            let p1 = pages[i].load(Ordering::SeqCst) as *mut u8;
            info!("addr {}={:?}", i, p1);
            assert!(p1 as usize % PAGE_SIZE == 0 && data.contains(unsafe { &mut *p1 }));
            for j in (i + 1)..pages.len() {
                let p2 = pages[j].load(Ordering::SeqCst) as *mut u8;
                assert_ne!(p1, p2, "{}=={}", i, j);
            }
        }

        // Free some
        for page in &pages[5..10] {
            let addr = page.swap(0, Ordering::SeqCst);
            alloc.put(addr, ChunkSize::Page).unwrap();
        }

        // Realloc
        for page in &pages[5..10] {
            alloc.get(ChunkSize::Page, page, |v| v, 0).unwrap();
        }

        // Free all
        for page in &pages {
            let addr = page.swap(0, Ordering::SeqCst);
            alloc.put(addr, ChunkSize::Page).unwrap();
        }
    }

    #[test]
    fn last_page() {
        logging();

        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, MAX_SIZE) };

        info!("mmap {} bytes", data.len());
        c_mmap_anon(data).unwrap();

        let mut alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        const DEFAULT: AtomicU64 = AtomicU64::new(0);
        let pages = Arc::new([DEFAULT; PT_LEN]);
        let barrier = Arc::new(Barrier::new(2));

        for page in pages.iter() {
            alloc.get(ChunkSize::Page, page, |v| v, 0).unwrap();
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
                    alloc.put(addr, ChunkSize::Page).unwrap();
                    alloc.get(ChunkSize::Page, &pages[14], |v| v, addr).unwrap();
                }
            })
        };

        barrier.wait();
        for _ in 0..10 {
            let addr = pages[13].load(Ordering::SeqCst);
            alloc.put(addr, ChunkSize::Page).unwrap();
            alloc.get(ChunkSize::Page, &pages[13], |v| v, addr).unwrap();
        }

        handle.join().unwrap();

        alloc.dump();
    }
}
