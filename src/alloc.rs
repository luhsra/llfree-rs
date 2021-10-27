//! Simple reduced non-volatile memory allocator.

use std::alloc::{alloc_zeroed, Layout};
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use log::{error, info, warn};
use static_assertions::const_assert;

use crate::paging::*;
use crate::util::align_down;
use crate::util::align_up;

const MAGIC: usize = 0xdeadbeef;
pub const MIN_SIZE: usize = PageTable::span(2) * 2;
pub const MAX_SIZE: usize = PageTable::span(LAYERS);

/// Volatile per thread metadata
pub struct Allocator {
    begin: usize,
    pages: usize,
    pt2: *mut PageTable,
    volatile: *mut PageTable,
}

static VOLATILE: AtomicPtr<PageTable> = AtomicPtr::new(std::ptr::null_mut());

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
    fn root(&self) -> &PageTable {
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
    fn pt(&self, layer: usize, page: usize) -> &PageTable {
        if layer < 2 || layer > LAYERS {
            panic!("layer has to be in 2..{}", LAYERS);
        }
        let i = page >> (PT_LEN_BITS * layer);
        if layer == 2 {
            // Located in NVRAM
            unsafe { &mut *self.pt2.add(i) }
        } else {
            // Located in DRAM
            let mut offset = 0;
            for i in layer..LAYERS {
                let span = PageTable::p_span(i);
                offset += (self.pages + span - 1) / span;
            }
            unsafe { &mut *self.volatile.add(offset + i) }
        }
    }

    /// Returns the according l1 page table
    fn pt1(&self, pte2: Entry, page: usize) -> Option<&PageTable> {
        if pte2.is_table() {
            let start = page & !(PageTable::p_span(1) - 1);
            Some(unsafe { &*((self.begin + (start + pte2.i1()) * PAGE_SIZE) as *const PageTable) })
        } else {
            None
        }
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

        let volatile = VOLATILE.load(Ordering::Acquire);
        if !volatile.is_null() {
            info!("Alloc already initialized");
            let pages = length / PAGE_SIZE;
            let num_pt2 = (pages + (PT_LEN * PT_LEN) - 1) / (PT_LEN * PT_LEN);
            let pt2 = (begin + length - (num_pt2 * PAGE_SIZE)) as *mut PageTable;
            // Remaining number of pages
            let pages = pages - num_pt2;
            return Ok(Allocator {
                begin,
                pages,
                pt2,
                volatile,
            });
        }

        if meta.length.load(Ordering::Acquire) == length
            && meta.magic.load(Ordering::Acquire) == MAGIC
            && false
        {
            // TODO: check if power was lost and recovery is necessary
            info!("Found allocator state. Recovery...");
            Self::recover(begin, length)
        } else {
            info!("Create new allocator state.");
            let alloc = Self::setup(begin, length)?;
            meta.length.store(length, Ordering::Release);
            meta.magic.store(MAGIC, Ordering::Release);
            VOLATILE.store(alloc.volatile, Ordering::Release);
            Ok(alloc)
        }
    }

    fn setup(begin: usize, length: usize) -> Result<Allocator> {
        let pages = length / PAGE_SIZE;
        let num_pt2 = (pages + (PT_LEN * PT_LEN) - 1) / (PT_LEN * PT_LEN);
        let num_pt1 = ((pages - num_pt2) + PT_LEN - 1) / PT_LEN;
        let pt2 = (begin + length - (num_pt2 * PAGE_SIZE)) as *mut PageTable;
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

        // Init pt1
        info!("Init pt1");
        for i in 0..num_pt1 {
            let addr = begin + i * PAGE_SIZE * PT_LEN;
            let pt1 = unsafe { &mut *(addr as *mut PageTable) };
            pt1.clear();
            pt1.set(0, Entry::page_reserved());
        }

        info!("Init pt2");
        // Init pt2
        for i in 0..num_pt2 {
            let pt2 = unsafe { &*pt2.add(i) };
            pt2.clear();
        }

        let mut higher_level_pts = 0;
        for i in 3..=LAYERS {
            let span = PageTable::p_span(i);
            higher_level_pts += (pages + span - 1) / span;
        }
        info!("#higher level pts = {}", higher_level_pts);

        // Init ptn - pt3
        let volatile = unsafe {
            alloc_zeroed(Layout::from_size_align_unchecked(
                higher_level_pts * PAGE_SIZE,
                PAGE_SIZE,
            ))
        } as *mut PageTable;
        if volatile.is_null() {
            return Err(Error::Memory);
        }

        // the high level page table are now initialized with zero
        // -> all counters and flags are zero

        Ok(Allocator {
            begin,
            pages,
            pt2,
            volatile,
        })
    }

    fn recover(begin: usize, length: usize) -> Result<Allocator> {
        let pages = length / PAGE_SIZE;
        let num_pt2 = (pages + PageTable::p_span(2) - 1) / PageTable::p_span(2);
        let pt2 = (begin + length - (num_pt2 * PAGE_SIZE)) as *mut PageTable;
        let pages = pages - num_pt2;

        let mut higher_level_pts = 0;
        for i in 3..=LAYERS {
            let span = PageTable::p_span(i);
            higher_level_pts += (pages + span - 1) / span;
        }
        info!("#higher level pts = {}", higher_level_pts);

        // Init ptn - pt3
        let volatile = unsafe {
            alloc_zeroed(Layout::from_size_align_unchecked(
                higher_level_pts * PAGE_SIZE,
                PAGE_SIZE,
            ))
        } as *mut PageTable;
        if volatile.is_null() {
            return Err(Error::Memory);
        }

        let alloc = Allocator {
            begin,
            pages,
            pt2,
            volatile,
        };

        // TODO recreate ptn-pt3 mapping
        let (pages, nonempty) = alloc.recover_rec(alloc.root(), LAYERS, 0);

        info!("Recovered pages={}, nonempty={}", pages, nonempty);

        Ok(alloc)
    }

    fn recover_rec(&self, pt: &PageTable, layer: usize, start: usize) -> (usize, usize) {
        let mut pages = 0;
        let mut nonemtpy = 0;

        for i in 0..PT_LEN {
            let start = start + i * PageTable::p_span(layer - 1);
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
    fn search_pt_page(&self, layer: usize, start: usize, pt: &PageTable) -> Result<usize> {
        for i in 0..PT_LEN {
            let start = start + i * PageTable::p_span(layer - 1);
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

    fn search_leaf_page(&self, start: usize) -> Result<SearchResult> {
        let pt2 = self.pt(2, start);

        for i2 in 0..PT_LEN {
            let start = start + i2 * PageTable::p_span(1);
            let pte2 = pt2.get(i2);

            if pte2.pages() >= PT_LEN {
                continue;
            }

            if pte2.pages() == PT_LEN - 1 {
                info!("search >> child pt full: {}, {:?}", start + pte2.i1(), pte2);
                return Ok(SearchResult::new(ChunkSize::Page, start + pte2.i1(), true));
            }

            if let Some(pt1) = self.pt1(pte2, start) {
                if let Ok(page) = self.search_pt_page(1, start, pt1) {
                    return Ok(SearchResult::new(ChunkSize::Page, page, false));
                }
            } else {
                assert!(pte2.is_empty());
                return Ok(SearchResult::new(ChunkSize::Page, start, false));
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
            return self.search_leaf_page(start);
        }

        let pt = self.pt(layer, start);
        if layer - 1 == size as usize {
            let pt = self.pt(layer, start);
            let page = self.search_pt_page(layer, start, pt)?;
            return Ok(SearchResult::new(size, page, false));
        }

        assert!(layer > 2);

        for i in 0..PT_LEN {
            let start = start + i * PageTable::p_span(layer - 1);
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
            } else if PageTable::p_span(size as usize) <= PageTable::p_span(layer - 1) - pte.pages()
            {
                // Enough pages in child pt
                if let Ok(result) = self.search(layer - 1, size, start) {
                    return Ok(result);
                }
            }
        }

        Err(Error::Memory)
    }

    fn alloc_last_pt1(&self, page: usize) -> Result<bool> {
        let pt2 = self.pt(2, page);
        let i2 = PageTable::p_idx(2, page);
        let i1 = PageTable::p_idx(1, page);

        // Only modify pt2 as pt1 is returned as page.

        match pt2.cas(
            i2,
            Entry::table(PT_LEN - 1, PT_LEN - 1, i1, false),
            Entry::table(PT_LEN, PT_LEN, 0, false),
        ) {
            Ok(_) => Ok(true),
            Err(pte) => {
                warn!("CAS: alloc last pt2 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    fn alloc_init_pt1(&self, start: usize) -> Result<bool> {
        info!("alloc init pt1");
        let start = start & !(PageTable::p_span(1) - 1);

        let pt1 = unsafe { &*((self.begin + start * PAGE_SIZE) as *const PageTable) };
        pt1.set(0, Entry::page_reserved());
        pt1.set(0, Entry::page());
        for i1 in 2..PT_LEN {
            pt1.set(i1, Entry::empty());
        }

        let pt2 = self.pt(2, start);
        let i2 = PageTable::p_idx(2, start);
        match pt2.cas(i2, Entry::empty(), Entry::table(1, 1, 0, false)) {
            Ok(_) => Ok(true),
            Err(pte) => {
                warn!("CAS: init pt1 {:?}", pte);
                Err(Error::CAS)
            },
        }
    }

    fn alloc_leaf_page(&self, target: SearchResult) -> Result<bool> {
        if target.last_page {
            return self.alloc_last_pt1(target.page);
        }

        let pt2 = self.pt(2, target.page);
        let i2 = PageTable::p_idx(2, target.page);
        let pte2 = pt2.get(i2);

        // Check before modifying pt1 as it's page allocated by the user.
        if pte2.pages() >= PT_LEN {
            warn!("CAS: page full");
            return Err(Error::CAS);
        }

        // Is there already a page table
        if let Some(pt1) = self.pt1(pte2, target.page) {
            let i1 = PageTable::p_idx(1, target.page);
            match pt1.cas(i1, Entry::empty(), Entry::page()) {
                Err(pte) => {
                    warn!("CAS: alloc leaf pt1 {:?}", pte);
                    return Err(Error::CAS);
                }
                _ => {}
            }
        } else if pte2.is_empty() {
            // Split and alloc new page table.
            return self.alloc_init_pt1(target.page);
        } else {
            error!("alloc unexpected pte2 {:?}", pte2);
            panic!();
        }

        let new = Entry::table(
            pte2.pages() + 1,
            pte2.nonempty() + 1,
            pte2.i1(),
            pte2.is_reserved(),
        );
        match pt2.cas(i2, pte2, new) {
            Ok(pte) => Ok(pte.is_empty()),
            Err(pte) => {
                warn!("CAS: alloc leaf pt2 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    fn alloc(&self, layer: usize, target: SearchResult) -> Result<bool> {
        if target.size as usize >= layer {
            return Err(Error::Memory);
        }
        let pt = self.pt(layer, target.page);
        let i = PageTable::p_idx(layer, target.page);

        info!(
            "alloc {:?} pt{}={:?} i={} p={}",
            target.size, layer, pt as *const _, i, target.page
        );

        if (target.size as usize) < layer - 1 {
            if layer == 2 {
                return self.alloc_leaf_page(target);
            }

            let newentry = self.alloc(layer - 1, target.clone())?;

            info!(
                "alloc l{} i={} -> +{}, +{}",
                layer,
                i,
                PageTable::p_span(target.size as usize),
                newentry as usize
            );

            match pt.inc(i, PageTable::p_span(target.size as usize), newentry as _) {
                Ok(pte) => Ok(pte.is_empty()),
                Err(pte) => {
                    error!("CAS: inc failed {:?}", pte);
                    Err(Error::CAS)
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
        match dst.compare_exchange(expected, new, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => Ok(()),
            Err(_) => {
                error!("CAS: get failed -> free");
                self.free(LAYERS, size, target.page).unwrap();
                Err(Error::CAS)
            }
        }
    }

    fn free_and_create_pt1(&self, page: usize) -> Result<bool> {
        // The new pt
        let pt2 = self.pt(2, page);
        let i = PageTable::p_idx(2, page);

        let child_pt = unsafe { &*((self.begin + page * PAGE_SIZE) as *const PageTable) };
        info!("free: init last pt1 {}", page);

        for j in 0..PT_LEN {
            if j == page % PT_LEN {
                child_pt.set(j, Entry::page_reserved());
            } else {
                child_pt.set(j, Entry::page());
            }
        }

        // memory barrier
        unsafe { core::arch::x86_64::_mm_sfence() };

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

    fn free_leaf_page(&self, page: usize) -> Result<bool> {
        info!("free leaf page {}", page);

        let pt2 = self.pt(2, page);
        let i2 = PageTable::p_idx(2, page);

        let pte2 = pt2.get(i2);
        if pte2.is_page() {
            return Err(Error::Address);
        } else if pte2.pages() < PT_LEN {
            if let Some(pt1) = self.pt1(pte2, page) {
                let i1 = PageTable::p_idx(1, page);
                info!("free l1 i={}", i1);

                match pt1.cas(i1, Entry::page(), Entry::empty()) {
                    Ok(_) => {}
                    Err(pte) => {
                        error!("CAS: free unexpected {:?}", pte);
                        return Err(Error::Address);
                    }
                }
            } else {
                error!("free no pt1");
                return Err(Error::Address);
            }

            match pt2.dec(i2, 1, 1) {
                Ok(pte) => Ok(pte.pages() == 1),
                Err(pte) => {
                    error!("CAS: free dec l2 {:?}", pte);
                    Err(Error::Address)
                }
            }
        } else {
            // free last page of pt1 & rebuild pt1
            self.free_and_create_pt1(page)
        }
    }

    fn free(&self, layer: usize, size: ChunkSize, page: usize) -> Result<bool> {
        if size as usize >= layer {
            return Err(Error::Memory);
        }
        let pt = self.pt(layer, page);
        let i = PageTable::p_idx(layer, page);

        info!(
            "free {:?} pt{}={:?} i={} p={}",
            size, layer, pt as *const _, i, page
        );

        // TODO: create pt1 after freeing large/huge pages!
        // Or create pt1 on demand in alloc

        if (size as usize) < layer - 1 {
            if layer == 2 {
                return self.free_leaf_page(page);
            }

            let pte = pt.get(i);

            if pte.is_page() {
                error!("No table found l{} {:?}", layer, pte);
                return Err(Error::Address);
            }

            let cleared = self.free(layer - 1, size, page)?;

            match pt.dec(i, PageTable::p_span(size as usize), cleared as _) {
                Ok(pte) => Ok(pte.pages() == PageTable::p_span(size as usize)),
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

        if addr % PageTable::span(size as usize) != 0
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

    fn dump_rec(&self, pt: &PageTable, layer: usize, start: usize) {
        for i in 0..PT_LEN {
            let pte = pt.get(i);
            let start = start + i * PageTable::p_span(layer - 1);
            if start >= self.pages {
                return;
            }

            info!(
                "{:1$} i={2} {3}: {4:?}",
                "",
                (LAYERS - layer) * 4,
                i,
                start,
                pte
            );

            if layer > 1 && !pte.is_page() && !pte.is_empty() {
                let child_pt = if layer == 2 {
                    self.pt1(pt.get(i), start).unwrap()
                } else {
                    self.pt(layer - 1, start)
                };
                self.dump_rec(child_pt, layer - 1, start);
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
    use crate::paging::PT_LEN;

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
        let pages = [DEFAULT; PT_LEN];
        for page in &pages {
            alloc.get(ChunkSize::Page, page, |v| v, 0).unwrap();
        }

        assert_eq!(alloc.allocated_pages(), 1 + PT_LEN + PT_LEN);

        // Check that the same page was not allocated twice
        for i in 0..pages.len() {
            for j in i + 1..pages.len() {
                assert_ne!(
                    pages[i].load(Ordering::Acquire),
                    pages[j].load(Ordering::Acquire)
                );
            }
        }
        // Free some
        for page in &pages[5..10] {
            let addr = page.swap(0, Ordering::AcqRel);
            alloc.put(addr, ChunkSize::Page).unwrap();
        }

        // Realloc
        for page in &pages[5..10] {
            alloc.get(ChunkSize::Page, page, |v| v, 0).unwrap();
        }

        // Free all
        for page in &pages {
            let addr = page.swap(0, Ordering::AcqRel);
            alloc.put(addr, ChunkSize::Page).unwrap();
        }

        alloc.dump();

        alloc
            .put(large.load(Ordering::Acquire), ChunkSize::L1)
            .unwrap();
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
                    let addr = pages[14].load(Ordering::Acquire);
                    alloc.put(addr, ChunkSize::Page).unwrap();
                    alloc.get(ChunkSize::Page, &pages[14], |v| v, addr).unwrap();
                }
            })
        };

        barrier.wait();
        for _ in 0..10 {
            let addr = pages[13].load(Ordering::Acquire);
            alloc.put(addr, ChunkSize::Page).unwrap();
            alloc.get(ChunkSize::Page, &pages[13], |v| v, addr).unwrap();
        }

        handle.join().unwrap();

        alloc.dump();
    }
}
