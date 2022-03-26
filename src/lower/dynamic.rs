use core::fmt;
use core::ops::Range;
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use crate::alloc::{Error, Result, Size};
use crate::entry::{Entry1, Entry2};
use crate::table::Table;
use crate::util::Page;

use super::LowerAlloc;
use crate::alloc::CAS_RETRIES;

/// Level 2 page allocator.
/// ```text
/// NVRAM: [ Pages & PT1s | PT2s | Meta ]
/// ```
#[derive(Default)]
pub struct DynamicLower {
    pub begin: usize,
    pub pages: usize,
    shared: Box<[Shared]>,
}

#[repr(align(64))]
struct Shared {
    /// Flag used to determine if a cpu is still updating a level 1 page table
    pub alloc_pt1: AtomicUsize,
}

impl fmt::Debug for DynamicLower {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicLower")
            .field("begin", &self.begin)
            .field("pages", &self.pages)
            .finish()
    }
}

impl LowerAlloc for DynamicLower {
    fn new(cores: usize, memory: &mut [Page]) -> Self {
        let mut shared = Vec::with_capacity(cores);
        shared.resize_with(cores, || Shared {
            alloc_pt1: AtomicUsize::new(0),
        });

        Self {
            begin: memory.as_ptr() as usize,
            // level 2 tables are stored at the end of the NVM
            pages: memory.len() - Table::num_pts(2, memory.len()),
            shared: shared.into(),
        }
    }

    fn pages(&self) -> usize {
        self.pages
    }

    fn memory(&self) -> Range<*const Page> {
        self.begin as *const Page..(self.begin + self.pages * Page::SIZE) as *const Page
    }

    fn clear(&self) {
        // Init pt2
        for i in 0..Table::num_pts(2, self.pages) {
            let pt2 = self.pt2(i * Table::span(2));
            if i + 1 < Table::num_pts(2, self.pages) {
                pt2.fill(Entry2::new().with_free(Table::span(1)));
            } else {
                for j in 0..Table::LEN {
                    let page = i * Table::span(2) + j * Table::span(1);
                    let max = Table::span(1).min(self.pages.saturating_sub(page));
                    pt2.set(j, Entry2::new().with_free(max));
                }
            }
        }
        // Init pt1
        for i in 0..Table::num_pts(1, self.pages) {
            // Within first page of own area
            let pt1 = unsafe { &*((self.begin + i * Table::m_span(1)) as *const Table<Entry1>) };

            if i + 1 < Table::num_pts(1, self.pages) {
                pt1.fill(Entry1::Empty);
            } else {
                for j in 0..Table::LEN {
                    let page = i * Table::span(1) + j;
                    if page < self.pages {
                        pt1.set(j, Entry1::Empty);
                    } else {
                        pt1.set(j, Entry1::Page);
                    }
                }
            }
        }
    }

    fn recover(&self, start: usize, deep: bool) -> Result<(usize, Size)> {
        let mut pages = 0;
        let mut size = Size::L0;

        let pt = self.pt2(start);
        for i in 0..Table::LEN {
            let start = Table::page(2, start, i);
            if start > self.pages {
                pt.set(i, Entry2::new());
            }

            let pte = pt.get(i);
            if pte.giant() {
                return Ok((0, Size::L2));
            } else if pte.page() {
                size = Size::L1;
            } else if deep && pte.free() > 0 && size == Size::L0 {
                let p = self.recover_l1(start, pte)?;
                if pte.free() != p {
                    warn!("Invalid PTE2 start=0x{start:x} i{i}: {} != {p}", pte.free());
                    pt.set(i, pte.with_free(p));
                }
                pages += p;
            } else {
                pages += pte.free();
            }
        }

        Ok((pages, size))
    }

    fn get(&self, core: usize, huge: bool, start: usize) -> Result<usize> {
        if !huge {
            return self.get_small(core, start);
        }

        let pt = self.pt2(start);
        for _ in 0..CAS_RETRIES {
            for page in Table::iterate(2, start) {
                let i = Table::idx(2, page);
                if pt.update(i, Entry2::mark_huge).is_ok() {
                    return Ok(page);
                }
            }
        }
        error!("Exceeding retries {}", start / Table::span(2));
        Err(Error::Corruption)
    }

    /// Free single page and returns if the page was huge
    fn put(&self, page: usize) -> Result<bool> {
        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);

        stop!();

        let mut old = pt2.get(i2);
        if old.page() {
            // Free huge page
            if page % Table::span(Size::L1 as _) != 0 {
                error!("Invalid address {page}");
                return Err(Error::Address);
            }

            let pt1 = self.pt1(page, 0);
            pt1.fill(Entry1::Empty);

            match pt2.cas(i2, old, Entry2::new_table(Table::LEN, 0)) {
                Ok(_) => Ok(true),
                Err(_) => {
                    error!("Corruption l2 i{i2}");
                    Err(Error::Corruption)
                }
            }
        } else if !old.giant() && old.free() < Table::LEN {
            for _ in 0..CAS_RETRIES {
                match self.put_small(old, page) {
                    Err(Error::CAS) => old = pt2.get(i2),
                    Err(e) => return Err(e),
                    Ok(_) => return Ok(false),
                }
            }
            error!("Exceeding retries {} {old:?}", page / Table::span(2));
            Err(Error::CAS)
        } else {
            error!("Not allocated 0x{page:x} {old:?}");
            Err(Error::Address)
        }
    }

    fn set_giant(&self, page: usize) {
        self.pt2(page).set(0, Entry2::new().with_giant(true));
    }
    fn clear_giant(&self, page: usize) {
        // Clear all level 1 page tables in this area
        for i in Table::range(2, page..self.pages) {
            let start = Table::page(2, page, i);

            // i1 is initially 0
            let pt1 = self.pt1(start, 0);
            pt1.fill(Entry1::Empty);
        }
        // Clear the persist flag
        self.pt2(page).set(0, Entry2::new_table(Table::LEN, 0));
    }

    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages;
        for i in 0..Table::num_pts(2, self.pages) {
            let start = i * Table::span(2);
            let pt2 = self.pt2(start);
            for i2 in Table::range(2, start..self.pages) {
                let start = Table::page(2, start, i2);
                let pte2 = pt2.get(i2);

                assert!(!pte2.giant());

                pages -= if pte2.page() {
                    Table::span(1)
                } else {
                    let pt1 = self.pt1(start, pte2.i1());
                    let mut child_pages = 0;
                    for i1 in Table::range(1, start..self.pages) {
                        child_pages += (pt1.get(i1) == Entry1::Empty) as usize;
                    }
                    assert_eq!(child_pages, pte2.free());
                    child_pages
                }
            }
        }
        pages
    }
}

impl DynamicLower {
    /// Returns the l1 page table that contains the `page`.
    fn pt1(&self, page: usize, i1: usize) -> &Table<Entry1> {
        let page = Table::page(1, page, i1);
        unsafe { &*((self.begin + page * Page::SIZE) as *const Table<Entry1>) }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages & PT1s | PT2s | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table<Entry2> {
        let i = page >> (Table::LEN_BITS * 2);
        unsafe { &*((self.begin + (self.pages + i) * Page::SIZE) as *mut Table<Entry2>) }
    }

    fn recover_l1(&self, start: usize, pte2: Entry2) -> Result<usize> {
        let pt = self.pt1(start, pte2.i1());
        let mut pages = 0;
        for i in 0..Table::LEN {
            if Table::page(1, start, i) > self.pages {
                break;
            }

            if pt.get(i) == Entry1::Empty {
                pages += 1;
            }
        }
        if pt.get(pte2.i1()) != Entry1::Empty {
            error!("Missing pt1 not found i1={}", pte2.i1());
            return Err(Error::Corruption);
        }
        Ok(pages)
    }

    /// Allocate a single page
    fn get_small(&self, core: usize, start: usize) -> Result<usize> {
        let pt2 = self.pt2(start);

        for _ in 0..CAS_RETRIES {
            for newstart in Table::iterate(2, start) {
                let i2 = Table::idx(2, newstart);

                stop!();

                let pte2 = pt2.get(i2);

                if pte2.page() || pte2.free() == 0 {
                    continue;
                }

                self.shared[core]
                    .alloc_pt1
                    .store(!Table::page(1, start, pte2.i1()), Ordering::SeqCst);

                stop!();

                if let Ok(pte2) = pt2.update(i2, |v| v.dec(pte2.i1())) {
                    let page = if pte2.free() == 1 {
                        self.get_last(core, pte2, newstart)
                    } else {
                        self.get_table(pte2, newstart)
                    };
                    self.shared[core].alloc_pt1.store(0, Ordering::SeqCst);
                    return page;
                }
                self.shared[core].alloc_pt1.store(0, Ordering::SeqCst);
            }
        }
        error!("Exceeding retries {start} {}", start / Table::span(2));
        Err(Error::Corruption)
    }

    /// Search free page table entry.
    fn get_table(&self, pte2: Entry2, start: usize) -> Result<usize> {
        let pt1 = self.pt1(start, pte2.i1());

        for _ in 0..CAS_RETRIES {
            for page in Table::iterate(1, start) {
                let i = Table::idx(1, page);
                if i == pte2.i1() {
                    continue;
                }

                #[cfg(feature = "stop")]
                if pt1.get(i) != Entry1::Empty {
                    continue;
                } else {
                    stop!();
                }

                if pt1.cas(i, Entry1::Empty, Entry1::Page).is_ok() {
                    return Ok(page);
                }
            }

            warn!("Nothing found, retry {}", start / Table::span(2));
            stop!();
        }
        error!("Exceeding retries {} {pte2:?}", start / Table::span(2));
        Err(Error::Corruption)
    }

    /// Allocate the last page (the pt1 is reused as last page).
    fn get_last(&self, core: usize, pte2: Entry2, start: usize) -> Result<usize> {
        stop!();
        info!("alloc last {} s={start}", pte2.i1());

        let pt1 = self.pt1(start, pte2.i1());
        let alloc_p1 = !Table::page(1, start, pte2.i1());

        // Wait for others to finish
        for (i, shared) in self.shared.iter().enumerate() {
            if i != core {
                while shared.alloc_pt1.load(Ordering::SeqCst) == alloc_p1 {
                    #[cfg(not(feature = "stop"))]
                    core::hint::spin_loop(); // pause CPU while waiting
                    stop!();
                }
            }
        }

        if pt1.cas(pte2.i1(), Entry1::Empty, Entry1::Page).is_err() {
            error!("Corruption l1 i{} {pte2:?}", pte2.i1());
            return Err(Error::Corruption);
        }

        Ok(Table::page(1, start, pte2.i1()))
    }

    fn put_small(&self, pte2: Entry2, page: usize) -> Result<()> {
        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);

        if pte2.free() == 0 {
            return self.put_full(pte2, page);
        }

        stop!();

        let pt1 = self.pt1(page, pte2.i1());
        let i1 = Table::idx(1, page);
        let pte1 = pt1.get(i1);

        if pte1 != Entry1::Page {
            error!("Invalid Addr l1 i{i1} p={page}");
            return Err(Error::Address);
        }

        stop!();

        // Update pt2 first, to avoid write in freed pt1
        if let Err(pte2) = pt2.update(i2, |pte| pte.inc_partial(pte2.i1())) {
            return if pte2.free() == Table::LEN {
                error!("Invalid Addr l1 i{i1} p={page}");
                Err(Error::Address)
            } else {
                Err(Error::CAS)
            };
        }

        stop!();

        if pt1.cas(i1, Entry1::Page, Entry1::Empty).is_err() {
            error!("Corruption l1 i{i1}");
            return Err(Error::Corruption);
        }

        Ok(())
    }

    /// Free last page & rebuild pt1 in it
    fn put_full(&self, pte2: Entry2, page: usize) -> Result<()> {
        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);
        let i1 = Table::idx(1, page);

        stop!();

        // The freed page becomes the new pt
        let pt1 = self.pt1(page, i1);
        info!("free: init last pt1 {page} (i{i1})");

        pt1.fill(Entry1::Page);
        pt1.set(i1, Entry1::Empty);

        match pt2.cas(i2, pte2, Entry2::new_table(1, i1)) {
            Ok(_) => Ok(()),
            Err(pte) => {
                warn!("CAS: create pt1 {pte:?}");
                Err(Error::CAS)
            }
        }
    }

    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        let mut out = String::new();
        writeln!(out, "Dumping pt {}", start / Table::span(2)).unwrap();

        let pt2 = self.pt2(start);
        for i2 in 0..Table::LEN {
            let start = Table::page(2, start, i2);
            if start > self.pages {
                return;
            }

            let pte2 = pt2.get(i2);
            let indent = (Table::LEVELS - 2) * 4;
            let addr = start * Page::SIZE;
            writeln!(out, "{:indent$}l2 i={i2} 0x{addr:x}: {pte2:?}", "").unwrap();
        }
        warn!("{out}");
    }
}

#[cfg(feature = "stop")]
#[cfg(test)]
mod test {
    use std::sync::Arc;

    use log::warn;

    use super::DynamicLower;
    use crate::entry::Entry1;
    use crate::lower::LowerAlloc;
    use crate::stop::{StopVec, Stopper};
    use crate::table::Table;
    use crate::thread;
    use crate::util::{logging, Page};

    fn count(pt: &Table<Entry1>) -> usize {
        let mut pages = 0;
        for i in 0..Table::LEN {
            pages += (pt.get(i) == Entry1::Empty) as usize;
        }
        pages
    }

    #[test]
    fn alloc_normal() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 0, 1, 1, 1, 0, 0],
            vec![1, 1, 0, 0, 0, 1, 1],
            vec![1, 0, 1, 0, 1, 0, 0],
        ];

        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(DynamicLower::new(2, &mut buffer));
            lower.clear();
            lower.get(0, false, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let page = l.get(t, false, 0).unwrap();
                drop(key);
                assert!(page != 0);
            });

            assert_eq!(lower.pt2(0).get(0).free(), Table::LEN - 3);
            assert_eq!(
                count(lower.pt1(0, lower.pt2(0).get(0).i1())),
                Table::LEN - 3
            );
        }
    }

    #[test]
    fn alloc_first() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 1, 1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0, 1, 1],
            vec![0, 1, 1, 1, 0, 0],
        ];

        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(DynamicLower::new(2, &mut buffer));
            lower.clear();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(t, false, 0).unwrap();
            });

            let pte2 = lower.pt2(0).get(0);
            assert_eq!(pte2.free(), Table::LEN - 2);
            assert_eq!(count(lower.pt1(0, pte2.i1())), Table::LEN - 2);
        }
    }

    #[test]
    fn alloc_last() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1, 1, 1],
            vec![0, 0, 1, 1, 0, 1, 1, 0], // wait for other cpu
            vec![1, 0, 0, 1, 1, 1, 1, 0],
            vec![1, 1, 0, 1, 0, 0, 0],
        ];

        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(DynamicLower::new(2, &mut buffer));
            lower.clear();

            for _ in 0..Table::LEN - 1 {
                lower.get(0, false, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(t, false, 0).unwrap();
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), 0);
            assert_eq!(pt2.get(1).free(), Table::LEN - 1);
            assert_eq!(
                count(lower.pt1(Table::LEN, pt2.get(1).i1())),
                Table::LEN - 1
            );
        }
    }

    #[test]
    fn free_normal() {
        logging();

        let orders = [
            vec![0, 0, 0, 0, 1, 1, 1, 1], // first 0, then 1
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 1, 1, 0, 0],
        ];

        let mut pages = [0; 2];
        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(DynamicLower::new(2, &mut buffer));
            lower.clear();

            pages[0] = lower.get(0, false, 0).unwrap();
            pages[1] = lower.get(0, false, 0).unwrap();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, {
                let pages = pages.clone();
                move |t| {
                    let _stopper = Stopper::init(stop, t as _);

                    l.put(pages[t as usize]).unwrap();
                }
            });

            assert_eq!(lower.pt2(0).get(0).free(), Table::LEN);
        }
    }

    #[test]
    fn free_last() {
        logging();

        let orders = [
            vec![0, 0, 1, 1, 1, 1],    // first 0, then 1
            vec![0, 1, 0, 1, 1, 1, 1], // 1 fails cas
            vec![0, 1, 1, 0, 0, 0, 0], // 0 fails cas
        ];

        let mut pages = [0; Table::LEN];
        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(DynamicLower::new(2, &mut buffer));
            lower.clear();

            for page in &mut pages {
                *page = lower.get(0, false, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                let _stopper = Stopper::init(stop, t as _);

                l.put(pages[t as usize]).unwrap();
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), 2);
            assert_eq!(count(lower.pt1(0, pt2.get(0).i1())), 2);
        }
    }

    #[test]
    fn realloc_last() {
        logging();

        let orders = [
            vec![0, 0, 0, 0, 1, 1, 1], // free then alloc
            vec![1, 1, 1, 0, 0],       // alloc last then free last
            vec![0, 1, 1, 1, 0, 0, 0], // 1 skips table
            vec![0, 1, 0, 1, 0, 1, 0], // 1 skips table
            vec![0, 0, 1, 0, 1, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 0, 1, 1], // nothing found & retry
        ];

        let mut pages = [0; Table::LEN];
        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(DynamicLower::new(2, &mut buffer));
            lower.clear();

            for page in &mut pages[..Table::LEN - 1] {
                *page = lower.get(0, false, 0).unwrap();
            }
            let stop = StopVec::new(2, order);

            let handle = std::thread::spawn({
                let stop = Arc::clone(&stop);
                let lower = lower.clone();
                move || {
                    let _stopper = Stopper::init(stop, 1);

                    lower.get(1, false, 0).unwrap();
                }
            });

            {
                let _stopper = Stopper::init(stop, 0);

                lower.put(pages[0]).unwrap();
            }

            handle.join().unwrap();

            let pt2 = lower.pt2(0);
            if pt2.get(0).free() == 1 {
                assert_eq!(count(lower.pt1(0, pt2.get(0).i1())), 1);
            } else {
                // Table entry skipped
                assert_eq!(pt2.get(0).free(), 2);
                assert_eq!(count(lower.pt1(0, pt2.get(0).i1())), 2);
                assert_eq!(pt2.get(1).free(), Table::LEN - 1);
                assert_eq!(
                    count(lower.pt1(Table::LEN, pt2.get(1).i1())),
                    Table::LEN - 1
                );
            }
        }
    }
}
