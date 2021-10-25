//! Simple reduced non-volatile memory allocator.

use std::alloc::{alloc_zeroed, Layout};
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

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
#[repr(usize)]
pub enum ChunkSize {
    Page = 0, // 4KiB
    L1 = 1,   // 2MiB
    L2 = 2,   // 1GiB
    L3 = 3,   // 512GiB
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
    fn pt1(&self, pte2: Entry, page: usize) -> &PageTable {
        let start = page & !(PageTable::p_span(1) - 1);
        unsafe { &*((self.begin + (start + pte2.i1()) * PAGE_SIZE) as *const PageTable) }
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
        println!(
            "Alloc: {:?}-{:?} - {} pages",
            begin as *const (),
            (begin + length) as *const (),
            length / PAGE_SIZE
        );

        let meta = unsafe { &mut *((begin + length) as *mut Meta) };

        let volatile = VOLATILE.load(Ordering::Acquire);
        if !volatile.is_null() {
            println!("Alloc already initialized");
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
        {
            // TODO: check if power was lost and recovery is necessary
            println!("Found allocator state. Recovery...");
            Self::recover(begin, length)
        } else {
            println!("Create new allocator state.");
            let alloc = Self::setup(begin, length)?;
            meta.length.store(length, Ordering::Relaxed);
            meta.magic.store(MAGIC, Ordering::Acquire);
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

        println!(
            "pages={}, #pt2={}, #pt1={}, area=[0x{:x}|{:?}-0x{:x}]",
            pages,
            num_pt2,
            num_pt1,
            begin,
            pt2,
            begin + length
        );

        // Init pt1
        println!("Init pt1");
        for i in 0..num_pt1 {
            let addr = begin + i * PAGE_SIZE * PT_LEN;
            let pt1 = unsafe { &mut *(addr as *mut PageTable) };
            pt1.clear();
            pt1.set(0, Entry::page());
        }

        println!("Init pt2");
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
        println!("#higher level pts = {}", higher_level_pts);

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
        println!("#higher level pts = {}", higher_level_pts);

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

        println!("Recovered pages={}, nonempty={}", pages, nonempty);

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

        println!(
            "recovered pt{}={:?} p={} n={}",
            layer, pt as *const _, pages, nonemtpy
        );

        (pages, nonemtpy)
    }

    fn search(&self, size: ChunkSize, pt: &PageTable, layer: usize, start: usize) -> Result<usize> {
        if size as usize >= layer {
            return Err(Error::Memory);
        }

        println!(
            "search {:?}, pt{}={:?}, s={}",
            size, layer, pt as *const _, start
        );

        if size as usize == layer - 1 {
            // Search in leaf page table
            for i in 0..PT_LEN {
                let start = start + i * PageTable::p_span(layer - 1);
                if start >= self.pages {
                    return Err(Error::Memory);
                }
                let pte = pt.get(i);
                if pte.is_empty() {
                    println!("search found i={}: {} - {:?}", i, start, pte);
                    return Ok(start);
                }
            }
        } else {
            for i in 0..PT_LEN {
                let start = start + i * PageTable::p_span(layer - 1);
                if start >= self.pages {
                    return Err(Error::Memory);
                }
                let pte = pt.get(i);
                let child_pt = if layer == 2 {
                    self.pt1(pte, start)
                } else {
                    self.pt(layer - 1, start)
                };

                if layer > 2 && size as usize == layer - 2 {
                    // Large / Huge Pages
                    if pte.nonempty() < PT_LEN {
                        if let Ok(result) = self.search(size, child_pt, layer - 1, start) {
                            return Ok(result);
                        }
                    }
                } else if PageTable::p_span(size as usize)
                    < PageTable::p_span(layer - 1) - pte.pages()
                {
                    if layer == 2 && pte.pages() == PT_LEN - 1 {
                        println!("")
                    }

                    // Enough pages in child pt
                    if let Ok(result) = self.search(size, child_pt, layer - 1, start) {
                        return Ok(result);
                    }
                }
            }
        }

        Err(Error::Memory)
    }

    fn alloc(&self, size: ChunkSize, pt: &PageTable, layer: usize, page: usize) -> Result<bool> {
        if size as usize >= layer {
            return Err(Error::Memory);
        }
        let i = PageTable::p_idx(layer, page);

        // TODO: alloc last page of pt1

        println!(
            "alloc {:?} pt{}={:?} i={} p={}",
            size, layer, pt as *const _, i, page
        );

        if (size as usize) < layer - 1 {
            let child_pt = if layer == 2 {
                self.pt1(pt.get(i), page)
            } else {
                self.pt(layer - 1, page)
            };

            let newentry = self.alloc(size, child_pt, layer - 1, page)?;

            println!(
                "alloc update pt{}={:?} i={} -> +{}, +{}",
                layer,
                pt as *const _,
                i,
                PageTable::p_span(size as usize),
                newentry as usize
            );

            match pt.inc(i, PageTable::p_span(size as usize), newentry as _) {
                Ok(pte) => Ok(pte.is_empty()),
                Err(_) => Err(Error::CAS),
            }
        } else {
            println!("alloc l{} insert i={}", layer, i);
            if pt.cas(i, Entry::empty(), Entry::page()) {
                Ok(true)
            } else {
                Err(Error::CAS)
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

        let mut page;
        loop {
            page = self.search(size, self.root(), LAYERS, 0)?;
            println!("page found {}", page);

            match self.alloc(size, self.root(), LAYERS, page) {
                Ok(_) => {
                    println!("allocated {:?}", self.root().get(0));
                    break;
                }
                Err(Error::CAS) => {
                    println!(">> CAS ERROR: retry alloc")
                }
                Err(e) => return Err(e),
            }
        }

        let addr = (page * PAGE_SIZE) as u64 + self.begin as u64;
        let new = translate(addr);
        match dst.compare_exchange(expected, new, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => Ok(()),
            Err(_) => {
                self.free(size, self.root(), LAYERS, page).unwrap();
                Err(Error::CAS)
            }
        }
    }

    fn free(&self, size: ChunkSize, pt: &PageTable, layer: usize, page: usize) -> Result<bool> {
        if size as usize >= layer {
            return Err(Error::Memory);
        }
        let i = PageTable::p_idx(layer, page);
        println!(
            "free {:?} pt{}={:?} i={} p={}",
            size, layer, pt as *const _, i, page
        );

        // TODO: free last page of pt1 & rebuild pt1

        if (size as usize) < layer - 1 {
            let child_pt = if layer == 2 {
                self.pt1(pt.get(i), page)
            } else {
                self.pt(layer - 1, page)
            };

            let cleared = self.free(size, child_pt, layer - 1, page)?;

            println!(
                "free update pt{}={:?} i={} -> -{}, -{}",
                layer,
                pt as *const _,
                i,
                PageTable::p_span(size as usize),
                cleared as usize
            );

            match pt.dec(i, PageTable::p_span(size as usize), cleared as _) {
                Ok(pte) => Ok(pte.is_page() || pte.pages() == PageTable::p_span(size as usize)),
                Err(_) => Err(Error::CAS),
            }
        } else {
            println!("free l{} insert i={}", layer, i);
            if pt.cas(i, Entry::page(), Entry::empty()) {
                Ok(true)
            } else {
                println!("cas err: found {:?}", pt.get(i));
                Err(Error::CAS)
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

        self.free(size, self.root(), LAYERS, page)?;

        println!("allocated {:?}", self.root().get(0));

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

            print!("{:1$}", "", (LAYERS - layer) * 4);
            println!("i={} {}: {:?}", i, start, pte);

            if layer > 1 && pte.is_table() {
                let child_pt = if layer == 2 {
                    self.pt1(pt.get(i), start)
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

    use crate::alloc::{ChunkSize, Error};
    use crate::mmap::c_mmap_fixed;

    use super::{Allocator, MAX_SIZE};

    #[test]
    fn initialization() {
        let data = unsafe { slice::from_raw_parts(0x1000_0000_0000_u64 as _, MAX_SIZE) };

        println!("prepare file");

        let f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("memfile")
            .unwrap();
        f.set_len(data.len() as _).unwrap();
        f.sync_all().unwrap();

        println!("alloc {} bytes", data.len());

        c_mmap_fixed(data, f).unwrap();

        println!("init alloc");

        let mut alloc = Allocator::init(data.as_ptr() as _, data.len()).unwrap();

        println!("get");

        let small = AtomicU64::new(0);
        alloc.get(ChunkSize::Page, &small, |v| v, 0).unwrap();

        let large = AtomicU64::new(0);
        alloc.get(ChunkSize::L1, &large, |v| v, 0).unwrap();

        let small = AtomicU64::new(0);
        alloc.get(ChunkSize::Page, &small, |v| v, 0).unwrap();
        let small = AtomicU64::new(0);
        alloc.get(ChunkSize::Page, &small, |v| v, 0).unwrap();

        let small = AtomicU64::new(5);

        assert_eq!(
            alloc.get(ChunkSize::Page, &small, |v| v, 0),
            Err(Error::CAS)
        );

        alloc.dump();

        alloc
            .put(large.load(Ordering::Acquire), ChunkSize::L1)
            .unwrap();
    }
}
