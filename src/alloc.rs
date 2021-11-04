//! Simple reduced non-volatile memory allocator.

use std::alloc::{alloc_zeroed, Layout};
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use log::{error, info, warn};
use static_assertions::const_assert;

use crate::page_alloc::PageAllocator;
use crate::paging::{Entry, Table, LAYERS, PAGE_SIZE, PT_LEN, PT_LEN_BITS};
use crate::util::{align_down, align_up};
use crate::{Error, Result, Size};

const MAGIC: usize = 0xdeadbeef;
pub const MIN_SIZE: usize = Table::m_span(2) * 2;
pub const MAX_SIZE: usize = Table::m_span(LAYERS);

/// Volatile per thread metadata
pub struct Allocator {
    begin: usize,
    pages: usize,
    volatile: *mut Table,
    small_start: usize,
    large_start: usize,
    meta: *mut Meta,
}

static VOLATILE: AtomicPtr<Table> = AtomicPtr::new(std::ptr::null_mut());

/// Non-Volatile global metadata
pub struct Meta {
    magic: AtomicUsize,
    length: AtomicUsize,
    active: AtomicUsize,
}
const_assert!(size_of::<Meta>() <= PAGE_SIZE);

impl Allocator {
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

        const INITIALIZING: *mut Table = usize::MAX as *mut _;
        loop {
            match VOLATILE.compare_exchange(
                std::ptr::null_mut(),
                usize::MAX as *mut _,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    if meta.length.load(Ordering::SeqCst) == length
                        && meta.magic.load(Ordering::SeqCst) == MAGIC
                    {
                        info!("Found allocator state. Recovery...");
                        return Self::recover(begin, length);
                    } else {
                        info!("Create new allocator state.");
                        let alloc = Self::setup(begin, length)?;
                        alloc.meta().length.store(length, Ordering::SeqCst);
                        alloc.meta().magic.store(MAGIC, Ordering::SeqCst);
                        VOLATILE.store(alloc.volatile, Ordering::SeqCst);
                        return Ok(alloc);
                    }
                }
                Err(INITIALIZING) => {
                    // TODO: passive waiting
                }
                Err(volatile) => {
                    return Self::new(begin, length, Some(unsafe { &mut *volatile }));
                }
            }
        }
    }

    fn new(begin: usize, length: usize, volatile: Option<&mut Table>) -> Result<Allocator> {
        let pages = length / PAGE_SIZE;
        let num_pt2 = (pages + (PT_LEN * PT_LEN) - 1) / (PT_LEN * PT_LEN);
        // Remaining number of pages
        let pages = pages - num_pt2;

        let volatile = if let Some(volatile) = volatile {
            volatile
        } else {
            // Allocate new page tables
            let mut higher_level_pts = 0;
            for i in 3..=LAYERS {
                let span = Table::span(i);
                higher_level_pts += (pages + span - 1) / span;
            }
            // the high level page table are initialized with zero
            // -> all counters and flags are zero
            unsafe {
                alloc_zeroed(Layout::from_size_align_unchecked(
                    higher_level_pts * PAGE_SIZE,
                    PAGE_SIZE,
                )) as *mut Table
            }
        };
        if volatile.is_null() {
            return Err(Error::Memory);
        }

        let mut alloc = Allocator {
            begin,
            pages,
            volatile,
            small_start: 0,
            large_start: 0,
            meta: (begin + length) as *mut Meta,
        };
        alloc.small_start = alloc.reserve_pt2(LAYERS, 0, Size::Page)?;
        alloc.large_start = alloc.reserve_pt2(LAYERS, 0, Size::L1)?;

        warn!(
            "pages={}, #pt2={}, area=[0x{:x}|{:x}-0x{:x}] small={} large={}",
            pages,
            num_pt2,
            begin,
            begin + pages * PT_LEN,
            begin + length,
            alloc.small_start / Table::span(2),
            alloc.large_start / Table::span(2)
        );

        alloc.meta().active.fetch_add(1, Ordering::SeqCst);

        return Ok(alloc);
    }

    fn setup(begin: usize, length: usize) -> Result<Allocator> {
        let alloc = Self::new(begin, length, None)?;

        // Init pt2
        for i in 0..alloc.num_pt(2) {
            let pt2 = alloc.pt(2, i * Table::span(2));
            pt2.clear();
        }
        // pt1's are initialized on demand

        Ok(alloc)
    }

    fn recover(begin: usize, length: usize) -> Result<Allocator> {
        let alloc = Self::new(begin, length, None)?;

        let meta = alloc.meta();

        if meta.active.load(Ordering::SeqCst) != 1 {
            error!("Allocator unexpectedly terminated");
            meta.active.store(1, Ordering::SeqCst);
        }

        let (pages, nonempty) = alloc.recover_rec(LAYERS, 0);
        info!("Recovered pages={}, nonempty={}", pages, nonempty);
        Ok(alloc)
    }

    /// Returns the metadata page, that contains size information and checksums
    pub fn meta(&self) -> &Meta {
        unsafe { &mut *self.meta }
    }

    /// Returns the number of allocated pages.
    pub fn allocated_pages(&self) -> usize {
        let mut pages = 0;
        let pt = self.pt(LAYERS, 0);
        for i in 0..PT_LEN {
            pages += pt.get(i).pages();
        }
        pages
    }

    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// NVRAM: [ Pages & PT1 | PT2 | Meta ]
    /// ```
    fn pt(&self, layer: usize, page: usize) -> &Table {
        assert!(layer >= 2 && layer <= LAYERS);

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
                offset += self.num_pt(i);
            }
            unsafe { &mut *self.volatile.add(offset + i) }
        }
    }

    /// Returns the number of page tables for the given `layer`
    #[inline(always)]
    fn num_pt(&self, layer: usize) -> usize {
        let span = Table::span(layer);
        (self.pages + span - 1) / span
    }

    fn page_alloc(&self) -> PageAllocator {
        PageAllocator::new(self.begin, self.pages)
    }

    fn recover_rec(&self, layer: usize, start: usize) -> (usize, usize) {
        let pt = self.pt(layer, start);

        let mut pages = 0;
        let mut nonemtpy = 0;

        for i in Table::range(layer, start..self.pages) {
            let start = start + i * Table::span(layer - 1);

            if layer > 2 {
                let (child_pages, child_nonempty) = self.recover_rec(layer - 1, start);

                if child_pages > 0 {
                    pt.set(i, Entry::table(child_pages, child_nonempty, 0, false));
                    nonemtpy += 1;
                } else {
                    pt.set(i, Entry::empty());
                }
                pages += child_pages;
            } else {
                let pte = pt.get(i);
                nonemtpy += !pte.is_empty() as usize;
                if pte.is_page() {
                    pages += PT_LEN;
                } else if pte.is_table() {
                    pages += pte.pages();
                }
            }
        }

        (pages, nonemtpy)
    }

    fn reserve_new(&mut self, size: Size) -> Result<usize> {
        let result = self.reserve_pt2(LAYERS, 0, size)?;
        warn!("reserved new {} ({:?})", result / Table::span(2), size);
        match size {
            Size::Page => {
                self.unreserve_pt2(self.small_start);
                self.small_start = result;
            }
            Size::L1 => {
                self.unreserve_pt2(self.large_start);
                self.large_start = result;
            }
            _ => panic!("Invalid reserve size"),
        }
        Ok(result)
    }

    fn reserve_pt2(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        let pt = self.pt(layer, start);
        for i in Table::range(layer, start..self.pages) {
            let start = start + i * Table::span(layer - 1);

            if layer == 3 {
                let pte3 = pt.get(i);

                // Is reserved or full?
                if pte3.is_reserved()
                    || (size == Size::Page && pte3.pages() >= Table::span(layer - 1))
                    || (size == Size::L1 && pte3.nonempty() >= PT_LEN)
                {
                    continue;
                }

                if let Ok(_) = pt.reserve(i, true) {
                    return Ok(start);
                }
            } else {
                if let Ok(result) = self.reserve_pt2(layer - 1, start, size) {
                    return Ok(result);
                }
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn unreserve_pt2(&self, start: usize) {
        let pt = self.pt(3, start);
        let i = Table::idx(3, start);
        if let Err(_) = pt.reserve(i, false) {
            error!("Unreserve failed! {}", start);
            panic!()
        }
    }

    /// Search free page table entry.
    fn alloc_child_page(&self, layer: usize, start: usize) -> Result<(usize, bool)> {
        let pt = self.pt(layer, start);

        for i in Table::range(layer, start..self.pages) {
            let start = start + i * Table::span(layer - 1);

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

        for i in Table::range(layer, start..self.pages) {
            let start = start + i * Table::span(layer - 1);

            let pte = pt.get(i);

            if pte.is_page() {
                continue;
            }

            // Large / Huge Pages or enough pages in child pt
            if (size as usize == layer - 2 && pte.nonempty() < PT_LEN)
                || ((size as usize) < layer - 2
                    && Table::span(size as usize) <= Table::span(layer - 1) - pte.pages())
            {
                if let Ok((page, newentry)) = self.alloc(layer - 1, size, start) {
                    match pt.inc(i, Table::span(size as _), newentry as _, 0) {
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
            // TODO: check pte3 pages / entries & reserve next before alloc
            _ => panic!("Huge pages are currently not supported!"),
        };

        let (page, mut newentry) = loop {
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
            let i = Table::idx(layer, page);
            match pt.inc(i, Table::span(size as _), newentry as _, 0) {
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
        let i = Table::idx(layer, page);

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

            match pt.dec(i, Table::span(size as usize), cleared as _, 0) {
                Ok(pte) => Ok(pte.pages() == Table::span(size as usize)),
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

        if addr % Table::m_span(size as usize) != 0
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
        for i in Table::range(layer, start..self.pages) {
            let start = start + i * Table::span(layer - 1);

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

impl Drop for Allocator {
    fn drop(&mut self) {
        self.unreserve_pt2(self.small_start);
        self.unreserve_pt2(self.large_start);
        self.meta().active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod test {

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Instant;

    use log::{info, warn};

    use crate::alloc::{Error, Size, VOLATILE};
    use crate::mmap::MMap;
    use crate::paging::{Table, PAGE_SIZE, PT_LEN};
    use crate::util::logging;

    use super::Allocator;

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mapping = MMap::anon(0x1000_0000_0000, MEM_SIZE).unwrap();
        let begin = mapping.slice.as_ptr() as usize;

        info!("mmap {} bytes at {:?}", MEM_SIZE, begin as *const u8);

        info!("init alloc");

        let mut alloc = Allocator::init(begin, MEM_SIZE).unwrap();

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
            assert!(
                p1 as usize % PAGE_SIZE == 0
                    && p1 as usize >= begin
                    && p1 as usize - begin < MEM_SIZE
            );
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

        const THREADS: usize = 10;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * Table::m_span(2);

        warn!("mapping {}", MEM_SIZE);

        let mapping = MMap::anon(0x1000_0000_0000, MEM_SIZE).unwrap();
        let begin = mapping.slice.as_ptr() as usize;
        let size = mapping.slice.len();

        info!("init alloc");
        drop(Allocator::init(begin, size).unwrap());

        // Stress test
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD * THREADS);
        pages.resize_with(ALLOC_PER_THREAD * THREADS, AtomicU64::default);

        let barrier = Arc::new(Barrier::new(THREADS + 1));

        let handles = (0..THREADS)
            .into_iter()
            .map(|t| {
                let pages_begin = pages.as_ptr() as usize;
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

        barrier.wait();
        let timer = Instant::now();

        for handle in handles {
            handle.join().unwrap();
        }

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        let alloc = Allocator::init(mapping.slice.as_ptr() as _, mapping.slice.len()).unwrap();
        assert_eq!(alloc.allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable_by(|a, b| a.load(Ordering::Relaxed).cmp(&b.load(Ordering::Relaxed)));

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i].load(Ordering::Relaxed) as *mut u8;
            let p2 = pages[i + 1].load(Ordering::Relaxed) as *mut u8;
            info!("addr {}={:?}", i, p1);
            assert!(p1 as usize % PAGE_SIZE == 0 && mapping.slice.contains(unsafe { &mut *p1 }));
            assert!(p1 != p2);
        }
    }

    #[test]
    fn parallel_malloc() {
        logging();

        const THREADS: usize = 10;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD * THREADS);
        pages.resize_with(ALLOC_PER_THREAD * THREADS, AtomicU64::default);

        let barrier = Arc::new(Barrier::new(THREADS + 1));

        let handles = (0..THREADS)
            .into_iter()
            .map(|t| {
                let pages_begin = pages.as_ptr() as usize;
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();

                    for i in 0..ALLOC_PER_THREAD {
                        let dst = unsafe {
                            &*(pages_begin as *const AtomicU64).add(t * ALLOC_PER_THREAD + i)
                        };
                        let val = unsafe {
                            std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                                PAGE_SIZE, PAGE_SIZE,
                            ))
                        } as u64;
                        dst.store(val, Ordering::SeqCst)
                    }
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        let timer = Instant::now();

        for handle in handles {
            handle.join().unwrap();
        }

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable_by(|a, b| a.load(Ordering::Relaxed).cmp(&b.load(Ordering::Relaxed)));

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i].load(Ordering::Relaxed) as *mut u8;
            let p2 = pages[i + 1].load(Ordering::Relaxed) as *mut u8;
            info!("addr {}={:?}", i, p1);
            assert!(p1 as usize % PAGE_SIZE == 0);
            assert!(p1 != p2);
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        const THREADS: usize = 10;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * Table::m_span(2);

        let mapping = MMap::anon(0x1000_0000_0000, MEM_SIZE).unwrap();
        let begin = mapping.slice.as_ptr() as usize;
        let size = mapping.slice.len();

        // Stress test
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD * THREADS);
        pages.resize_with(ALLOC_PER_THREAD * THREADS, AtomicU64::default);
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;

        let handles = (0..THREADS)
            .into_iter()
            .map(|t| {
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let mut alloc = Allocator::init(begin, size).unwrap();
                    warn!("t{} wait", t);
                    barrier.wait();

                    let pages = unsafe {
                        std::slice::from_raw_parts_mut(
                            (pages_begin as *mut AtomicU64).add(t * ALLOC_PER_THREAD),
                            ALLOC_PER_THREAD,
                        )
                    };

                    for page in pages.iter_mut() {
                        alloc.get(Size::Page, page, |v| v, 0).unwrap();
                    }

                    for page in pages {
                        alloc.put(page.load(Ordering::SeqCst), Size::Page).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        let alloc = Allocator::init(mapping.slice.as_ptr() as _, mapping.slice.len()).unwrap();
        assert_eq!(alloc.allocated_pages(), 0);
    }

    #[test]
    fn alloc_free() {
        logging();

        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10);

        let mapping = MMap::anon(0x1000_0000_0000, 8 << 30).unwrap();
        let begin = mapping.slice.as_ptr() as usize;
        let size = mapping.slice.len();

        info!("mmap {} bytes", mapping.slice.len());

        let mut alloc = Allocator::init(begin, size).unwrap();

        let barrier = Arc::new(Barrier::new(2));

        // Alloc on first thread
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD);
        pages.resize_with(ALLOC_PER_THREAD, AtomicU64::default);
        for page in pages.iter() {
            alloc.get(Size::Page, page, |v| v, 0).unwrap();
        }

        let handle = {
            let barrier = barrier.clone();

            thread::spawn(move || {
                let mut alloc = Allocator::init(begin, size).unwrap();

                barrier.wait();
                // Free on another thread
                for page in pages.iter() {
                    let addr = page.load(Ordering::SeqCst);
                    alloc.put(addr, Size::Page).unwrap();
                }
            })
        };

        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD);
        pages.resize_with(ALLOC_PER_THREAD, AtomicU64::default);

        barrier.wait();

        // Simultaneously alloc on first thread
        for page in pages.iter() {
            alloc.get(Size::Page, page, |v| v, 0).unwrap();
        }

        handle.join().unwrap();

        assert_eq!(alloc.allocated_pages(), ALLOC_PER_THREAD);

        alloc.dump();
    }

    #[test]
    fn recover() {
        logging();

        let mapping = MMap::anon(0x1000_0000_0000, 8 << 30).unwrap();

        info!("mmap {} bytes", mapping.slice.len());

        let mut alloc = Allocator::init(mapping.slice.as_ptr() as _, mapping.slice.len()).unwrap();

        for _ in 0..PT_LEN + 2 {
            let small = AtomicU64::new(0);
            alloc.get(Size::Page, &small, |v| v, 0).unwrap();
            let large = AtomicU64::new(0);
            alloc.get(Size::L1, &large, |v| v, 0).unwrap();
        }

        assert_eq!(alloc.allocated_pages(), PT_LEN + 2 + PT_LEN * (PT_LEN + 2));

        VOLATILE.store(std::ptr::null_mut(), Ordering::SeqCst);
        let alloc = Allocator::init(mapping.slice.as_ptr() as _, mapping.slice.len()).unwrap();

        assert_eq!(alloc.allocated_pages(), PT_LEN + 2 + PT_LEN * (PT_LEN + 2));
    }
}
