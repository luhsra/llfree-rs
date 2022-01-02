//! Simple reduced non-volatile memory allocator.
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use crate::entry::{Entry, Entry3};
use crate::leaf_alloc::LeafAllocator;
use crate::table::{self, Page, Table, LAYERS, PAGE_SIZE, PT_LEN, PT_LEN_BITS};
use crate::{Error, Result, Size};

const MAGIC: usize = 0xdeadbeef;
pub const MIN_SIZE: usize = 2 * table::m_span(2);
pub const MAX_SIZE: usize = table::m_span(LAYERS);

/// Non-Volatile global metadata
pub struct Meta {
    pub magic: AtomicUsize,
    length: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(size_of::<Meta>() <= PAGE_SIZE);

enum Init {
    None,
    Initializing,
    Ready,
}

/// Volatile shared metadata
#[repr(align(64))]
pub struct Allocator {
    begin: usize,
    pages: usize,
    tables: Vec<Table<Entry>>,
    meta: *mut Meta,
    pub local: Vec<LeafAllocator>,
    initialized: AtomicUsize,
}

static mut SHARED: Allocator = Allocator {
    begin: 0,
    pages: 0,
    tables: Vec::new(),
    meta: core::ptr::null_mut(),
    local: Vec::new(),
    initialized: AtomicUsize::new(0),
};

pub fn alloc<'a>() -> &'a Allocator {
    let alloc = unsafe { &SHARED };
    if alloc.initialized.load(Ordering::SeqCst) != Init::Ready as usize {
        panic!("Not initialized!");
    }
    alloc
}

impl Allocator {
    /// Allows init from multiple threads.
    pub fn init(cores: usize, memory: &mut [Page]) -> Result<()> {
        info!("init cores={} mem={:?}", cores, memory.as_ptr_range());
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

        if memory.len() < (MIN_SIZE * cores) / PAGE_SIZE {
            error!(
                "memory size {} < {}",
                memory.len(),
                (MIN_SIZE * cores) / PAGE_SIZE
            );
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        let length = (memory.len() * PAGE_SIZE - PAGE_SIZE).min(MAX_SIZE);
        let begin = memory.as_ptr() as usize;
        info!(
            "Alloc: {:?}-{:?} - {} pages",
            begin as *const (),
            (begin + length) as *const (),
            length / PAGE_SIZE
        );

        alloc.meta = (begin + length) as *mut Meta;
        let meta = unsafe { &mut *alloc.meta };

        alloc.begin = begin;
        alloc.pages = length / PAGE_SIZE;
        // level 2 tables are stored at the end of the NVM
        alloc.pages -= table::num_pts(2, alloc.pages);

        let mut num_pt = 0;
        for layer in 3..=LAYERS {
            num_pt += table::num_pts(layer, alloc.pages);
        }
        alloc.tables = vec![Table::empty(); num_pt];
        alloc.local = vec![LeafAllocator::new(alloc.begin, alloc.pages); cores];

        if meta.length.load(Ordering::SeqCst) == length
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", alloc.pages);
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            if deep {
                error!("Allocator unexpectedly terminated");
            }
            let pages = alloc.recover_rec(LAYERS, 0, deep);
            warn!("Recovered pages {}", pages);
        } else {
            warn!("Setup allocator state p={}", alloc.pages);
            alloc.local[0].clear();
            meta.length.store(length, Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        // init all leaf_allocs
        let mut start = 0;
        for (i, leaf) in alloc.local.iter().enumerate() {
            start = alloc.reserve_pt2(LAYERS, start, Size::L0)?;
            leaf.start_l0.store(start, Ordering::Relaxed);
            start = alloc.reserve_pt2(LAYERS, start, Size::L1)?;
            leaf.start_l1.store(start, Ordering::Relaxed);
            info!(
                "init {} small={} huge={}",
                i,
                leaf.start_l0.load(Ordering::Relaxed),
                leaf.start_l1.load(Ordering::Relaxed),
            );
        }

        info!("{:?}", alloc.pt3(0));

        meta.active.store(1, Ordering::SeqCst);
        alloc.initialized.store(Init::Ready as _, Ordering::SeqCst);
        Ok(())
    }

    pub fn uninit() {
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

    /// Returns the metadata page, that contains size information and checksums
    pub fn meta(&self) -> &Meta {
        unsafe { &mut *self.meta }
    }

    #[cfg(test)]
    /// Returns the number of allocated pages.
    pub fn allocated_pages(&self) -> usize {
        assert_eq!(self.initialized.load(Ordering::SeqCst), Init::Ready as _);

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
    /// ```
    fn pt(&self, layer: usize, page: usize) -> &Table<Entry> {
        assert!((4..=LAYERS).contains(&layer));

        let i = page >> (PT_LEN_BITS * layer);
        let offset: usize = (layer..LAYERS).map(|i| table::num_pts(i, self.pages)).sum();
        &self.tables[offset + i]
    }

    /// Returns the page table of the given `layer` that contains the `page`.
    /// ```text
    /// DRAM: [ PT4 | n*PT3 (| ...) ]
    /// ```
    fn pt3(&self, page: usize) -> &Table<Entry3> {
        let i = page >> (PT_LEN_BITS * 3);
        let offset: usize = (3..LAYERS).map(|i| table::num_pts(i, self.pages)).sum();
        unsafe { &*(&self.tables[offset + i] as *const _ as *const Table<Entry3>) }
    }

    fn recover_rec(&self, layer: usize, start: usize, deep: bool) -> usize {
        let mut pages = 0;
        let pt = self.pt(layer, start);
        for i in table::range(layer, start..self.pages) {
            let page = table::page(layer, start, i);

            let c_pages = if layer - 1 == 3 {
                self.recover_l3(page, deep)
            } else {
                self.recover_rec(layer - 1, page, deep)
            };

            pt.set(i, Entry::new().with_pages(c_pages));
            pages += c_pages;
        }

        pages
    }

    fn recover_l3(&self, start: usize, deep: bool) -> usize {
        let mut pages = 0;
        let pt = self.pt3(start);

        for i in table::range(3, start..self.pages) {
            let page = table::page(3, start, i);

            let (c_pages, size) = self.local[0].recover(page, deep);
            if size == Size::L2 {
                pt.set(i, Entry3::new_giant());
            } else if c_pages > 0 {
                pt.set(i, Entry3::new_table(c_pages, size, 0));
            } else {
                pt.set(i, Entry3::new());
            }
            pages += c_pages;
        }
        pages
    }

    fn reserve_pt2(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        for i in 0..PT_LEN {
            let i = (table::idx(layer, start) + i) % PT_LEN;
            let page = table::page(layer, start, i);

            if layer > 3 {
                if let Ok(result) = self.reserve_pt2(layer - 1, page, size) {
                    return Ok(result);
                }
            } else {
                let pt = self.pt3(start);
                if pt.update(i, |v| v.inc_usage(size, 1)).is_ok() {
                    return Ok(page);
                }
            }
        }
        error!("Reserve failed!");
        Err(Error::Memory)
    }

    fn unreserve_pt2(&self, start: usize) {
        let pt = self.pt3(start);
        let i = table::idx(3, start);
        if pt.update(i, Entry3::dec_usage).is_err() {
            panic!("Unreserve failed")
        }
    }

    pub fn get(&self, core: usize, size: Size) -> Result<usize> {
        // Start at the reserved memory chunk for this thread
        let page = if size == Size::L2 {
            loop {
                match self.get_giant(core, LAYERS, 0) {
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
                    let new = self.increment_parents(LAYERS, start, size)?;
                    if start != new {
                        leaf.start_l0.store(new, Ordering::SeqCst);
                    }
                    leaf.get(new)
                }
                Size::L1 => {
                    let start = leaf.start_l1.load(Ordering::SeqCst);
                    let new = self.increment_parents(LAYERS, start, size)?;
                    if start != new {
                        leaf.start_l1.store(new, Ordering::SeqCst);
                    }
                    leaf.get_huge(new)
                }
                Size::L2 => panic!(),
            }
        };

        Ok(page)
    }

    fn increment_parents(&self, layer: usize, start: usize, size: Size) -> Result<usize> {
        if layer <= 3 {
            return self.increment_parents_l3(start, size);
        }

        // Increment parents
        let pt = self.pt(layer, start);
        for i in table::range(layer, start..self.pages) {
            let page = table::page(layer, start, i).max(start);

            if let Ok(_) = pt.update(i, |v| {
                v.inc(size, layer, self.pages - table::round(layer, page))
            }) {
                match self.increment_parents(layer - 1, page, size) {
                    Ok(result) => return Ok(result),
                    Err(_) => todo!("special behavior on fragmentation"),
                }
            };
        }
        Err(Error::Memory)
    }

    fn increment_parents_l3(&self, start: usize, size: Size) -> Result<usize> {
        let pt = self.pt3(start);
        for i in table::range(3, start..self.pages) {
            let page = table::page(3, start, i);

            if page != start {
                if pt.update(i, |v| v.inc_usage(size, 1)).is_err() {
                    continue;
                }
            }

            match pt.update(i, |v| v.inc(size, self.pages - table::round(3, start))) {
                Ok(_) => return Ok(page),
                Err(pte) => {
                    warn!("try reserve new i{} {:?}", i, pte);
                    pt.update(i, Entry3::dec_usage).unwrap()
                }
            };
        }
        Err(Error::Memory)
    }

    fn get_giant(&self, core: usize, layer: usize, start: usize) -> Result<usize> {
        info!("alloc l{}, s={}", layer, start);

        if layer == 3 {
            return self.get_giant_page(core, start);
        }

        let pt = self.pt(layer, start);
        for i in table::range(layer, start..self.pages) {
            info!("get giant l{} i{}", layer, i);

            let page = table::page(layer, start, i);
            let pte = pt.get(i);

            // Already allocated or reserved
            if pte.pages() > table::span(layer - 1) - table::span(Size::L2 as _) {
                warn!(
                    "giant no space {} > {} - {}",
                    pte.pages(),
                    table::span(layer - 1),
                    table::span(Size::L2 as _)
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

        for i in table::range(3, start..self.pages) {
            if pt.cas(i, Entry3::new(), Entry3::new_giant()).is_ok() {
                let page = table::page(3, start, i);
                warn!("allocated l3 i{} p={} s={}", i, page, start);
                self.local[core].persist(page);
                return Ok(page);
            }
        }
        error!("Nothing found s={}", start);
        Err(Error::Memory)
    }

    pub fn put(&self, core: usize, page: usize) -> Result<Size> {
        loop {
            match self.put_rec(core, LAYERS, page) {
                Err(Error::CAS) => warn!("CAS: retry free"),
                r => return r,
            }
        }
    }

    fn put_rec(&self, core: usize, layer: usize, page: usize) -> Result<Size> {
        if layer == 3 {
            return self.put_l3(core, page);
        }

        let pt = self.pt(layer, page);
        let i = table::idx(layer, page);
        let pte = pt.get(i);

        if pte.pages() == 0 {
            error!("No table found l{} {:?}", layer, pte);
            return Err(Error::Address);
        }

        let size = self.put_rec(core, layer - 1, page)?;

        match pt.update(i, |v| v.dec(size)) {
            Ok(_) => Ok(size),
            Err(pte) => panic!("Corruption: l{} i{} {:?}", layer, i, pte),
        }
    }

    fn put_l3(&self, core: usize, page: usize) -> Result<Size> {
        let pt = self.pt3(page);
        let i3 = table::idx(3, page);
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
            panic!("Corruption l3 i{} p={} - {:?}", i3, pte3.pages(), size)
        }
        Ok(size)
    }

    fn put_giant(&self, core: usize, page: usize) -> Result<Size> {
        if (page % table::span(Size::L2 as _)) != 0 {
            error!(
                "Invalid alignment p={:x} a={:x}",
                page,
                table::span(Size::L2 as _)
            );
            return Err(Error::Address);
        }

        // Clear pt1's & remove pt2 flag
        self.local[core].clear_giant(page);

        let pt = self.pt3(page);
        let i = table::idx(3, page);

        info!("free l3 i{}", i);
        match pt.cas(i, Entry3::new_giant(), Entry3::new()) {
            Ok(_) => Ok(Size::L2),
            _ => {
                error!("Invalid {}", page);
                Err(Error::Address)
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
        for i in table::range(3, start..self.pages) {
            let start = table::page(3, start, i);

            let pte = pt.get(i);

            info!(
                "{:1$}l3 i={2} 0x{3:x}: {4:?}",
                "",
                (LAYERS - 3) * 4,
                i,
                start * PAGE_SIZE,
                pte,
            );

            match pte.size() {
                Some(Size::L0 | Size::L1) if pte.pages() > 0 => self.local[0].dump(start),
                _ => {}
            }
        }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        for local in &self.local {
            self.unreserve_pt2(local.start_l0.load(Ordering::SeqCst));
            self.unreserve_pt2(local.start_l1.load(Ordering::SeqCst));
        }
        self.meta().active.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod test {

    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Instant;

    use log::{info, warn};

    use crate::alloc::{alloc, Allocator, Size};
    use crate::cpu;
    use crate::mmap::MMap;
    use crate::table::Page;
    use crate::table::{self, PAGE_SIZE, PT_LEN};
    use crate::util::{logging, parallel};

    #[cfg(target_os = "linux")]
    fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<'a, Page>, ()> {
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
    #[cfg(not(target_os = "linux"))]
    fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<'a, Page>, ()> {
        MMap::anon(begin, length)
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / PAGE_SIZE).unwrap();

        info!("mmap {} bytes at {:?}", MEM_SIZE, mapping.as_ptr());

        info!("init alloc");

        Allocator::init(1, &mut mapping).unwrap();

        warn!("start alloc...");
        let small = alloc().get(0, Size::L0).unwrap();
        let large = alloc().get(0, Size::L1).unwrap();
        let huge = alloc().get(0, Size::L2).unwrap();

        // Stress test
        let mut pages = vec![0; PT_LEN * PT_LEN];
        for page in &mut pages {
            *page = alloc().get(0, Size::L0).unwrap();
        }

        warn!("check...");

        assert_eq!(
            alloc().allocated_pages(),
            1 + PT_LEN + table::span(2) + pages.len()
        );

        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            info!("addr {}={:x}", i, p1);
            assert!(p1 < mapping.len());
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        for page in &pages[10..PT_LEN + 10] {
            alloc().put(0, *page).unwrap();
        }

        alloc().put(0, small).unwrap();
        alloc().put(0, large).unwrap();
        alloc().put(0, huge).unwrap();

        // Realloc
        for page in &mut pages[10..PT_LEN + 10] {
            *page = alloc().get(0, Size::L0).unwrap();
        }

        warn!("free...");

        // Free all
        for page in &pages {
            alloc().put(0, *page).unwrap();
        }

        Allocator::uninit();
    }

    #[test]
    fn parallel_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const PAGES: usize = 2 * THREADS * table::span(2);

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        info!("init alloc");
        Allocator::init(THREADS, &mut mapping).unwrap();

        // Stress test
        let mut pages = vec![0usize; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;
        let timer = Instant::now();

        parallel(THREADS as _, move |t| {
            cpu::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst =
                    unsafe { &mut *(pages_begin as *mut usize).add(t * ALLOC_PER_THREAD + i) };
                *dst = alloc().get(t, Size::L0).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc().allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(p1 < PAGES && p2 < PAGES);
            assert!(p1 != p2);
        }

        Allocator::uninit();
    }

    #[test]
    fn parallel_malloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let pages_begin = pages.as_ptr() as usize;
        let barrier = Arc::new(Barrier::new(THREADS));
        let timer = Instant::now();

        parallel(THREADS as _, move |t| {
            cpu::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = unsafe { libc::malloc(PAGE_SIZE) } as u64;
                assert!(*dst != 0);
            }
        });

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();

        // Check that the same page was not allocated twice
        let mut last = None;
        for p in pages {
            assert!(last != Some(p));
            unsafe { libc::free(p as _) };
            last = Some(p);
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        const THREADS: usize = 8;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * table::m_span(2);

        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / PAGE_SIZE).unwrap();

        Allocator::init(THREADS, &mut mapping).unwrap();

        // Stress test
        let mut pages = Vec::with_capacity(ALLOC_PER_THREAD * THREADS);
        pages.resize_with(ALLOC_PER_THREAD * THREADS, AtomicU64::default);
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;

        parallel(THREADS as _, move |t| {
            cpu::pin(t);
            barrier.wait();

            let pages = unsafe {
                std::slice::from_raw_parts_mut(
                    (pages_begin as *mut usize).add(t as usize * ALLOC_PER_THREAD),
                    ALLOC_PER_THREAD,
                )
            };

            for page in pages.iter_mut() {
                *page = alloc().get(t, Size::L0).unwrap();
            }

            for page in pages {
                alloc().put(t, *page).unwrap();
            }
        });

        assert_eq!(alloc().allocated_pages(), 0);
        Allocator::uninit();
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10) / 2;

        let mut mapping = mapping(0x1000_0000_0000, 4 * table::span(2)).unwrap();

        Allocator::init(THREADS, &mut mapping).unwrap();

        let barrier = Arc::new(Barrier::new(THREADS));

        // Alloc on first thread
        cpu::pin(0);
        let mut pages = vec![0; ALLOC_PER_THREAD];
        for page in &mut pages {
            *page = alloc().get(0, Size::L0).unwrap();
        }

        let handle = {
            let barrier = barrier.clone();

            thread::spawn(move || {
                cpu::pin(1);
                barrier.wait();
                // Free on another thread
                for page in &pages {
                    alloc().put(1, *page).unwrap();
                }
            })
        };

        let mut pages = vec![0; ALLOC_PER_THREAD];

        barrier.wait();

        // Simultaneously alloc on first thread
        for page in &mut pages {
            *page = alloc().get(0, Size::L0).unwrap();
        }

        handle.join().unwrap();

        assert_eq!(alloc().allocated_pages(), ALLOC_PER_THREAD);

        Allocator::uninit();
    }

    #[test]
    fn recover() {
        logging();

        let mut mapping = mapping(0x1000_0000_0000, 8 << 18).unwrap();

        {
            Allocator::init(1, &mut mapping).unwrap();

            for _ in 0..PT_LEN + 2 {
                alloc().get(0, Size::L0).unwrap();
                alloc().get(0, Size::L1).unwrap();
            }

            alloc().get(0, Size::L2).unwrap();

            assert_eq!(
                alloc().allocated_pages(),
                table::span(2) + PT_LEN + 2 + PT_LEN * (PT_LEN + 2)
            );
            Allocator::uninit();
        }

        Allocator::init(1, &mut mapping).unwrap();

        assert_eq!(
            alloc().allocated_pages(),
            table::span(2) + PT_LEN + 2 + PT_LEN * (PT_LEN + 2)
        );
        Allocator::uninit();
    }
}
