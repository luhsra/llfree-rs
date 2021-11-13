//! Simple reduced non-volatile memory allocator.

use std::alloc::{alloc_zeroed, Layout};
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use log::{error, info, warn};
use static_assertions::const_assert;

use crate::entry::{Entry, L2Entry};
use crate::leaf_alloc::LeafAllocator;
use crate::table::{self, Table, LAYERS, PAGE_SIZE, PT_LEN, PT_LEN_BITS};
use crate::util::{align_down, align_up};
use crate::{Error, Result, Size};

const MAGIC: usize = 0xdeadbeef;
pub const MIN_SIZE: usize = table::m_span(2) * 2;
pub const MAX_SIZE: usize = table::m_span(LAYERS);

/// Volatile per thread metadata
pub struct Allocator {
    begin: usize,
    pages: usize,
    volatile: *mut Table<Entry>,
    meta: *mut Meta,
    small_start: usize,
    large_start: usize,
}

static VOLATILE: AtomicPtr<Table<Entry>> = AtomicPtr::new(std::ptr::null_mut());

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

        const INITIALIZING: *mut Table<Entry> = usize::MAX as *mut _;
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
                        && false
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

    fn new(begin: usize, length: usize, volatile: Option<&mut Table<Entry>>) -> Result<Allocator> {
        let pages = length / PAGE_SIZE;
        let num_pt2 = (pages + table::span(2) - 1) / table::span(2);
        // Remaining number of pages
        let pages = pages - num_pt2;

        let volatile = if let Some(volatile) = volatile {
            volatile
        } else {
            // Allocate new page tables
            let pts: usize = (3..=LAYERS)
                .map(|i| (pages + table::span(i) - 1) / table::span(i))
                .sum();

            // the high level page table are initialized with zero
            // -> all counters and flags are zero
            unsafe {
                alloc_zeroed(Layout::from_size_align_unchecked(
                    pts * PAGE_SIZE,
                    PAGE_SIZE,
                )) as *mut Table<Entry>
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
            "p={} #pt2={} [0x{:x}|{:x}-0x{:x}] small={} large={}",
            pages,
            num_pt2,
            begin,
            begin + pages * PT_LEN,
            begin + length,
            alloc.small_start / table::span(2),
            alloc.large_start / table::span(2)
        );

        alloc.meta().active.fetch_add(1, Ordering::SeqCst);

        Ok(alloc)
    }

    fn setup(begin: usize, length: usize) -> Result<Allocator> {
        let alloc = Self::new(begin, length, None)?;

        // Init pt2
        for i in 0..alloc.num_pt(2) {
            let pt2 = alloc.pt2(i * table::span(2));
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

    #[cfg(test)]
    /// Returns the number of allocated pages.
    pub fn allocated_pages(&self) -> usize {
        let mut pages = 0;
        let pt = self.pt(LAYERS, 0);
        for i in 0..PT_LEN {
            pages += pt.get(i).pages();
        }
        pages
    }

    /// Returns the layer 2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages & PT1 | PT2 | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table<L2Entry> {
        let i = page >> (PT_LEN_BITS * 2);
        // Located in NVRAM
        let pt2 = (self.begin + self.pages * PAGE_SIZE) as *mut Table<L2Entry>;
        unsafe { &mut *pt2.add(i) }
    }

    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// ```
    fn pt(&self, layer: usize, page: usize) -> &Table<Entry> {
        assert!((3..=LAYERS).contains(&layer));

        let i = page >> (PT_LEN_BITS * layer);
        // Located in DRAM
        let offset: usize = (layer..LAYERS).map(|i| self.num_pt(i)).sum();
        unsafe { &mut *self.volatile.add(offset + i) }
    }

    /// Returns the number of page tables for the given `layer`
    #[inline(always)]
    fn num_pt(&self, layer: usize) -> usize {
        let span = table::span(layer);
        (self.pages + span - 1) / span
    }

    #[inline(always)]
    fn leaf_alloc(&self) -> LeafAllocator {
        LeafAllocator::new(self.begin, self.pages)
    }

    fn recover_rec(&self, layer: usize, start: usize) -> (usize, usize) {
        let mut pages = 0;
        let mut nonemtpy = 0;

        if layer > 2 {
            let pt = self.pt(layer, start);

            for i in table::range(layer, start..self.pages) {
                let page = table::page(layer, start, i);

                let (child_pages, child_nonempty) = self.recover_rec(layer - 1, page);

                if child_pages > 0 {
                    pt.set(i, Entry::table(child_pages, child_nonempty, false));
                    nonemtpy += 1;
                } else {
                    pt.set(i, Entry::empty());
                }
                pages += child_pages;
            }
        } else {
            let pt = self.pt2(start);

            for i in table::range(layer, start..self.pages) {
                let pte = pt.get(i);
                nonemtpy += !pte.is_empty() as usize;
                if pte.is_huge() {
                    todo!();
                } else if pte.is_page() {
                    pages += PT_LEN;
                } else {
                    pages += pte.pages();
                }
            }
        }

        (pages, nonemtpy)
    }

    fn reserve_new(&mut self, size: Size) -> Result<usize> {
        let result = self.reserve_pt2(LAYERS, 0, size)?;
        warn!("reserved new {} ({:?})", result / table::span(2), size);
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
        for i in 0..PT_LEN {
            let i = (table::idx(layer, start) + i) % PT_LEN;
            let page = table::page(layer, start, i);

            if layer == 3 {
                let pte3 = pt.get(i);

                if pt
                    .update(i, |v| {
                        // Is reserved or full?
                        if !pte3.is_reserved()
                            && !pte3.is_page()
                            && ((size == Size::Page && pte3.pages() < table::span(layer - 1))
                                || (size == Size::L1 && pte3.nonempty() < PT_LEN))
                        {
                            Some(Entry::table(v.pages(), v.nonempty(), true))
                        } else {
                            None
                        }
                    })
                    .is_ok()
                {
                    return Ok(page);
                }
            } else if let Ok(result) = self.reserve_pt2(layer - 1, page, size) {
                return Ok(result);
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn unreserve_pt2(&self, start: usize) {
        let pt = self.pt(3, start);
        let i = table::idx(3, start);
        if pt
            .update(i, |v| Some(Entry::table(v.pages(), v.nonempty(), false)))
            .is_err()
        {
            panic!("Unreserve failed")
        }
    }

    /// Search free page table entry.
    fn alloc_huge_page(&self, layer: usize, start: usize) -> Result<(usize, bool)> {
        let pt = self.pt(layer, start);

        for i in table::range(layer, start..self.pages) {
            if pt.cas(i, Entry::empty(), Entry::page()).is_ok() {
                let page = table::page(layer, start, i);
                warn!("allocated l{} i{} p={} s={}", layer, i, page, start);
                self.pt2(page).set(0, L2Entry::huge()); // Persist
                return Ok((page, true));
            }
        }
        Err(Error::Memory)
    }

    fn alloc_l2(&self, size: Size, start: usize) -> Result<(usize, bool)> {
        return if size == Size::Page {
            self.leaf_alloc().alloc(start)
        } else {
            let pt = self.pt2(start);

            for i in table::range(2, start..self.pages) {
                if pt.cas(i, L2Entry::empty(), L2Entry::page()).is_ok() {
                    let page = table::page(2, start, i);
                    return Ok((page, true));
                }
            }

            Err(Error::Memory)
        };
    }

    fn alloc_huge(&self, layer: usize, size: Size, start: usize) -> Result<(usize, bool)> {
        assert!(layer > 2);
        assert!(size >= Size::L2);
        assert!((size as usize) < layer);
        assert!(start < self.pages);

        info!("alloc l{}, s={}", layer, start);

        if layer - 1 == size as usize {
            return self.alloc_huge_page(layer, start);
        }

        let pt = self.pt(layer, start);
        for i in table::range(layer, start..self.pages) {
            let page = table::page(layer, start, i);

            let pte = pt.get(i);

            // Already allocated or reserved
            if pte.is_page() || pte.is_reserved() {
                continue;
            }

            // Not enough space in child table
            if (size as usize == layer - 2 && pte.nonempty() >= PT_LEN)
                || ((size as usize) < layer - 2
                    && table::span(size as _) > table::span(layer - 1) - pte.pages())
            {
                continue;
            }

            return match self.alloc_huge(layer - 1, size, page) {
                Ok((page, newentry)) => {
                    match pt.update(i, |v| v.inc(table::span(size as _), newentry as _)) {
                        Ok(pte) => Ok((page, pte.pages() == 0)),
                        Err(pte) => panic!("Corruption: l{} i{} {:?}", layer, i, pte),
                    }
                }
                Err(Error::Memory) => continue,
                Err(e) => Err(e),
            };
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
        // Start at the reserved memory chunk for this thread
        let page = if size == Size::L2 {
            loop {
                match self.alloc_huge(LAYERS, size, self.small_start.max(self.large_start)) {
                    Ok((page, _)) => break page,
                    Err(Error::CAS) => warn!("CAS: retry alloc"),
                    Err(e) => return Err(e),
                }
            }
        } else {
            let mut start = match size {
                Size::Page => self.small_start,
                Size::L1 => self.large_start,
                _ => panic!(),
            };

            // Allocate small / large pages in the reserved memory
            let (page, mut newentry) = loop {
                match self.alloc_l2(size, start) {
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
                let i = table::idx(layer, page);

                match pt.update(i, |v| v.inc(table::span(size as _), newentry as _)) {
                    Ok(pte) => newentry = pte.pages() == 0,
                    Err(pte) => panic!("Corruption: l{} i{} {:?}", layer, i, pte),
                }
            }
            page
        };

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
        assert!((size as usize) < layer);

        if layer == 2 {
            return self.free_l2(size, page);
        }

        if size as usize == layer - 1 {
            return self.free_huge(layer, page);
        }

        let pt = self.pt(layer, page);
        let i = table::idx(layer, page);
        let pte = pt.get(i);

        if pte.is_page() {
            error!("No table found l{} {:?}", layer, pte);
            return Err(Error::Address);
        }

        let cleared = self.free(layer - 1, size, page)?;

        match pt.update(i, |v| v.dec(table::span(size as _), cleared as _)) {
            Ok(pte) => Ok(pte.pages() == table::span(size as usize)),
            Err(pte) => panic!("Corruption: l{} i{} {:?}", layer, i, pte),
        }
    }

    fn free_huge(&self, layer: usize, page: usize) -> Result<bool> {
        assert!(layer >= 3);

        let pt = self.pt(layer, page);
        let i = table::idx(layer, page);

        info!("free l{} i={}", layer, i);
        match pt.cas(i, Entry::page(), Entry::empty()) {
            Ok(_) => {
                // Clear persistance
                self.pt2(page).set(0, L2Entry::empty());
                Ok(true)
            }
            Err(pte) => {
                error!("free unexpected l{} i{} p={} {:?}", layer, i, page, pte);
                Err(Error::Address)
            }
        }
    }

    fn free_l2(&self, size: Size, page: usize) -> Result<bool> {
        if size == Size::Page {
            return self.leaf_alloc().free(page);
        } else {
            let pt = self.pt2(page);
            let i = table::idx(2, page);
            info!("free l2 i={}", i);
            return match pt.cas(i, L2Entry::page(), L2Entry::empty()) {
                Ok(_) => Ok(true),
                Err(_) => Err(Error::Address),
            };
        }
    }

    pub fn put(&mut self, addr: u64, size: Size) -> Result<()> {
        let addr = addr as usize;

        if addr % table::m_span(size as usize) != 0
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
                Ok(_) => break Ok(()),
            }
        }
    }

    #[allow(dead_code)]
    pub fn dump(&self) {
        self.dump_rec(LAYERS, 0);
    }

    fn dump_rec(&self, layer: usize, start: usize) {
        let pt = self.pt(layer, start);
        for i in table::range(layer, start..self.pages) {
            let start = table::page(layer, start, i);

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
                    self.leaf_alloc().dump(start);
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

    use std::os::raw::c_int;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Instant;

    use log::{info, warn};

    use crate::alloc::{Error, Size, VOLATILE};
    use crate::mmap::MMap;
    use crate::table::{self, PAGE_SIZE, PT_LEN};
    use crate::util::{logging, parallel};

    use super::Allocator;

    fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<'a>, c_int> {
        if let Ok(file) = std::env::var("NVM_FILE") {
            warn!("MMap file {} l={}G", file, length >> 30);
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(file)
                .unwrap();
            MMap::dax(begin, length, f)
        } else {
            MMap::anon(begin, length)
        }
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mapping = mapping(0x1000_0000_0000, MEM_SIZE).unwrap();
        let begin = mapping.as_ptr() as usize;

        info!("mmap {} bytes at {:?}", MEM_SIZE, begin as *const u8);

        info!("init alloc");

        let mut alloc = Allocator::init(begin, MEM_SIZE).unwrap();

        warn!("start alloc...");

        let small = AtomicU64::new(0);
        alloc.get(Size::Page, &small, |v| v, 0).unwrap();

        let large = AtomicU64::new(0);
        alloc.get(Size::L1, &large, |v| v, 0).unwrap();

        let huge = AtomicU64::new(0);
        alloc.get(Size::L2, &huge, |v| v, 0).unwrap();

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
        alloc.leaf_alloc().dump(0);

        warn!("check...");

        assert_eq!(
            alloc.allocated_pages(),
            1 + PT_LEN + table::span(2) + pages.len()
        );

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
        alloc.put(huge.load(Ordering::SeqCst), Size::L2).unwrap();

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
        const MEM_SIZE: usize = 2 * THREADS * table::m_span(2);

        let mapping = mapping(0x1000_0000_0000, MEM_SIZE).unwrap();
        let begin = mapping.as_ptr() as usize;
        let size = mapping.len();

        info!("init alloc");
        drop(Allocator::init(begin, size).unwrap());

        // Stress test
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD * THREADS);
        pages.resize_with(ALLOC_PER_THREAD * THREADS, AtomicU64::default);

        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;

        let timer = Instant::now();

        parallel(THREADS as _, move |t| {
            let mut alloc = Allocator::init(begin, size).unwrap();
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe {
                    &*(pages_begin as *const AtomicU64).add(t as usize * ALLOC_PER_THREAD + i)
                };
                alloc.get(Size::Page, dst, |v| v, 0).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        let alloc = Allocator::init(mapping.as_ptr() as _, mapping.len()).unwrap();
        assert_eq!(alloc.allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable_by(|a, b| a.load(Ordering::Relaxed).cmp(&b.load(Ordering::Relaxed)));

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i].load(Ordering::Relaxed) as usize;
            let p2 = pages[i + 1].load(Ordering::Relaxed) as usize;
            assert!(p1 % PAGE_SIZE == 0 && p1 >= begin && p1 - begin < MEM_SIZE);
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
        let pages_begin = pages.as_ptr() as usize;

        let barrier = Arc::new(Barrier::new(THREADS));

        let timer = Instant::now();

        parallel(THREADS as _, move |t| {
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe {
                    &*(pages_begin as *const AtomicU64).add(t as usize * ALLOC_PER_THREAD + i)
                };
                let val = unsafe { libc::malloc(PAGE_SIZE) } as u64;
                dst.store(val, Ordering::SeqCst)
            }
        });

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable_by(|a, b| a.load(Ordering::Relaxed).cmp(&b.load(Ordering::Relaxed)));

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i].load(Ordering::Relaxed);
            let p2 = pages[i + 1].load(Ordering::Relaxed);
            assert!(p1 != p2);
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        const THREADS: usize = 10;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * table::m_span(2);

        let mapping = mapping(0x1000_0000_0000, MEM_SIZE).unwrap();
        let begin = mapping.as_ptr() as usize;
        let size = mapping.len();

        // Stress test
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD * THREADS);
        pages.resize_with(ALLOC_PER_THREAD * THREADS, AtomicU64::default);
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;

        parallel(THREADS as _, move |t| {
            let mut alloc = Allocator::init(begin, size).unwrap();
            barrier.wait();

            let pages = unsafe {
                std::slice::from_raw_parts_mut(
                    (pages_begin as *mut AtomicU64).add(t as usize * ALLOC_PER_THREAD),
                    ALLOC_PER_THREAD,
                )
            };

            for page in pages.iter_mut() {
                alloc.get(Size::Page, page, |v| v, 0).unwrap();
            }

            for page in pages {
                alloc.put(page.load(Ordering::SeqCst), Size::Page).unwrap();
            }
        });

        let alloc = Allocator::init(mapping.as_ptr() as _, mapping.len()).unwrap();
        assert_eq!(alloc.allocated_pages(), 0);
    }

    #[test]
    fn alloc_free() {
        logging();

        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10);

        let mapping = mapping(0x1000_0000_0000, 8 << 30).unwrap();
        let begin = mapping.as_ptr() as usize;
        let size = mapping.len();

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

        let mapping = mapping(0x1000_0000_0000, 8 << 30).unwrap();

        let mut alloc = Allocator::init(mapping.as_ptr() as _, mapping.len()).unwrap();

        for _ in 0..PT_LEN + 2 {
            let small = AtomicU64::new(0);
            alloc.get(Size::Page, &small, |v| v, 0).unwrap();
            let large = AtomicU64::new(0);
            alloc.get(Size::L1, &large, |v| v, 0).unwrap();
        }

        assert_eq!(alloc.allocated_pages(), PT_LEN + 2 + PT_LEN * (PT_LEN + 2));

        VOLATILE.store(std::ptr::null_mut(), Ordering::SeqCst);
        let alloc = Allocator::init(mapping.as_ptr() as _, mapping.len()).unwrap();

        assert_eq!(alloc.allocated_pages(), PT_LEN + 2 + PT_LEN * (PT_LEN + 2));
    }
}
