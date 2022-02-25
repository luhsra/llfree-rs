use std::ops::{Deref, Range};
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use crate::alloc::{Error, Result, Size};
use crate::entry::{Entry1, Entry2, Entry3};
use crate::table::Table;
use crate::util::Atomic;
use crate::Page;

const CAS_RETRIES: usize = 4096;

#[cfg(all(test, feature = "stop"))]
macro_rules! stop {
    () => {
        crate::stop::stop().unwrap()
    };
}
#[cfg(not(all(test, feature = "stop")))]
macro_rules! stop {
    () => {};
}

/// Layer 2 page allocator.
#[repr(align(64))]
pub struct LowerAlloc {
    pub begin: usize,
    pub pages: usize,
    local: Vec<Local>,
}

/// Per core data.
#[repr(align(64))]
pub struct Local {
    start: [AtomicUsize; 2],
    pte: [Atomic<Entry3>; 2],
    frees: [AtomicUsize; 4],
    frees_i: AtomicUsize,
    alloc_pt1: AtomicUsize,
}

impl Local {
    fn new() -> Self {
        const F: AtomicUsize = AtomicUsize::new(usize::MAX);
        Self {
            alloc_pt1: AtomicUsize::new(0),
            start: [AtomicUsize::new(usize::MAX), AtomicUsize::new(usize::MAX)],
            pte: [Atomic::new(Entry3::new()), Atomic::new(Entry3::new())],
            frees_i: AtomicUsize::new(0),
            frees: [F; 4],
        }
    }
    pub fn start(&self, huge: bool) -> &AtomicUsize {
        &self.start[huge as usize]
    }
    pub fn pte(&self, huge: bool) -> &Atomic<Entry3> {
        &self.pte[huge as usize]
    }
    pub fn frees_push(&self, page: usize) {
        let i = self.frees_i.fetch_add(1, Ordering::Relaxed) % self.frees.len();
        self.frees[i].store(page, Ordering::Relaxed);
    }
    pub fn frees_related(&self, page: usize) -> bool {
        let n = page / Table::span(2);
        self.frees
            .iter()
            .all(|p| p.load(Ordering::Relaxed) / Table::span(2) == n)
    }
}

impl Deref for LowerAlloc {
    type Target = [Local];

    fn deref(&self) -> &Self::Target {
        &self.local
    }
}

impl LowerAlloc {
    pub fn default() -> Self {
        Self {
            begin: 0,
            pages: 0,
            local: Vec::new(),
        }
    }

    pub fn new(cores: usize, memory: &mut [Page]) -> Self {
        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Local::new);
        Self {
            begin: memory.as_ptr() as usize,
            // level 2 tables are stored at the end of the NVM
            pages: memory.len() - Table::num_pts(2, memory.len()),
            local,
        }
    }

    pub fn memory(&self) -> Range<*const Page> {
        self.begin as *const Page..(self.begin + self.pages * Page::SIZE) as *const Page
    }

    pub fn clear(&self) {
        // Init pt2
        for i in 0..Table::num_pts(2, self.pages) {
            let pt2 = self.pt2(i * Table::span(2));
            for j in 0..Table::LEN {
                let page = i * Table::span(2) + j * Table::span(1);
                let max = Table::span(1).min(self.pages.saturating_sub(page));
                pt2.set(j, Entry2::new().with_free(max));
            }
        }
        // Init pt1
        for i in 0..Table::num_pts(1, self.pages) {
            // Within first page of own area
            let pt1 = unsafe { &*((self.begin + i * Table::m_span(1)) as *const Table<Entry1>) };

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

    /// Returns the l1 page table that contains the `page`.
    fn pt1(&self, pte2: Entry2, page: usize) -> &Table<Entry1> {
        let page = Table::page(1, page, pte2.i1());
        unsafe { &*((self.begin + page * Page::SIZE) as *const Table<Entry1>) }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages & PT1 | PT2 | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table<Entry2> {
        let i = page >> (Table::LEN_BITS * 2);
        unsafe { &*((self.begin + (self.pages + i) * Page::SIZE) as *mut Table<Entry2>) }
    }

    pub fn recover(&self, start: usize, deep: bool) -> Result<(usize, Size)> {
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

    fn recover_l1(&self, start: usize, pte2: Entry2) -> Result<usize> {
        let pt = self.pt1(pte2, start);
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
    pub fn get(&self, core: usize, start: usize) -> Result<usize> {
        let pt2 = self.pt2(start);

        for _ in 0..CAS_RETRIES {
            for newstart in Table::iterate(2, start) {
                let i2 = Table::idx(2, newstart);

                stop!();

                let pte2 = pt2.get(i2);

                if pte2.page() || pte2.free() == 0 {
                    continue;
                }

                self[core]
                    .alloc_pt1
                    .store(!Table::page(1, start, pte2.i1()), Ordering::SeqCst);

                stop!();

                if let Ok(pte2) = pt2.update(i2, |v| v.dec(pte2.i1())) {
                    let page = if pte2.free() == 1 {
                        self.get_last(core, pte2, newstart)
                    } else {
                        self.get_table(pte2, newstart)
                    };
                    self[core].alloc_pt1.store(0, Ordering::SeqCst);
                    return page;
                }
                self[core].alloc_pt1.store(0, Ordering::SeqCst);
            }
        }
        error!("Exceeding retries {start} {}", start / Table::span(2));
        Err(Error::Corruption)
    }

    /// Search free page table entry.
    fn get_table(&self, pte2: Entry2, start: usize) -> Result<usize> {
        let pt1 = self.pt1(pte2, start);

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
        info!("alloc last {} s={}", pte2.i1(), start);

        let pt1 = self.pt1(pte2, start);
        let alloc_p1 = !Table::page(1, start, pte2.i1());

        // Wait for others to finish
        for (i, leaf) in self.iter().enumerate() {
            if i != core {
                while leaf.alloc_pt1.load(Ordering::SeqCst) == alloc_p1 {
                    warn!("Waiting for cpu {i} on {core}");
                    stop!();
                }
            }
        }

        if pt1.cas(pte2.i1(), Entry1::Empty, Entry1::Page).is_err() {
            error!("Corruption l1 i{} {:?}", pte2.i1(), pte2);
            return Err(Error::Corruption);
        }

        Ok(Table::page(1, start, pte2.i1()))
    }

    pub fn get_huge(&self, start: usize) -> Result<usize> {
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

    pub fn persist(&self, page: usize) {
        self.pt2(page).set(0, Entry2::new().with_giant(true));
    }

    /// Free single page and returns if the page was huge
    pub fn put(&self, page: usize) -> Result<bool> {
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

            let pt1 = unsafe { &*((self.begin + page * Page::SIZE) as *const Table<Entry1>) };
            pt1.clear();

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
            Err(Error::Address)
        }
    }

    fn put_small(&self, pte2: Entry2, page: usize) -> Result<()> {
        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);

        if pte2.free() == 0 {
            return self.put_full(pte2, page);
        }

        stop!();

        let pt1 = self.pt1(pte2, page);
        let i1 = Table::idx(1, page);
        let pte1 = pt1.get(i1);

        if pte1 != Entry1::Page {
            error!("Invalid Addr l1 i{i1} p={page}");
            return Err(Error::Address);
        }

        stop!();

        if let Err(pte2) = pt2.update(i2, |pte| pte.inc(pte2.i1())) {
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
        let pt1 = unsafe { &*((self.begin + page * Page::SIZE) as *const Table<Entry1>) };
        info!("free: init last pt1 {page} (i{i1})");

        for j in 0..Table::LEN {
            if j == i1 {
                pt1.set(j, Entry1::Empty);
            } else {
                pt1.set(j, Entry1::Page);
            }
        }

        match pt2.cas(i2, pte2, Entry2::new_table(1, i1)) {
            Ok(_) => Ok(()),
            Err(pte) => {
                warn!("CAS: create pt1 {pte:?}");
                Err(Error::CAS)
            }
        }
    }

    pub fn clear_giant(&self, page: usize) {
        // Clear all layer 1 page tables in this area
        for i in 0..Table::LEN {
            // i1 is initially 0
            let pt1 = unsafe {
                &*((self.begin + (page + i * Table::span(1)) * Page::SIZE) as *const Table<Entry1>)
            };
            pt1.clear();
        }
        // Clear the persist flag
        self.pt2(page).set(0, Entry2::new_table(Table::LEN, 0));
    }

    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        let pt2 = self.pt2(start);
        for i2 in 0..Table::LEN {
            let start = Table::page(2, start, i2);
            if start > self.pages {
                return;
            }

            let pte2 = pt2.get(i2);
            info!(
                "{:1$}l2 i={i2} 0x{2:x}: {pte2:?}",
                "",
                (Table::LAYERS - 2) * 4,
                start * Page::SIZE,
            );
            if !pte2.giant() && !pte2.page() && pte2.free() > 0 && pte2.free() < Table::LEN {
                let pt1 = self.pt1(pte2, start);
                for i1 in 0..Table::LEN {
                    let page = Table::page(1, start, i1);
                    let pte1 = pt1.get(i1);
                    info!(
                        "{:1$}l1 i={2} 0x{3:x}: {4:?}",
                        "",
                        (Table::LAYERS - 1) * 4,
                        i1,
                        page * Page::SIZE,
                        pte1
                    );
                }
            }
        }
    }
}

#[cfg(feature = "stop")]
#[cfg(test)]
mod test {
    use std::sync::Arc;

    use log::warn;

    use crate::entry::Entry1;
    use crate::lower_alloc::LowerAlloc;
    use crate::stop::{StopRand, StopVec, Stopper};
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
            let lower = Arc::new(LowerAlloc::new(2, &mut buffer));
            lower.clear();
            lower.get(0, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let page = l.get(t, 0).unwrap();
                drop(key);
                assert!(page != 0);
            });

            assert_eq!(lower.pt2(0).get(0).free(), Table::LEN - 3);
            assert_eq!(count(lower.pt1(lower.pt2(0).get(0), 0)), Table::LEN - 3);
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
            let lower = Arc::new(LowerAlloc::new(2, &mut buffer));
            lower.clear();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(t, 0).unwrap();
            });

            assert_eq!(lower.pt2(0).get(0).free(), Table::LEN - 2);
            assert_eq!(count(lower.pt1(lower.pt2(0).get(0), 0)), Table::LEN - 2);
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
            let lower = Arc::new(LowerAlloc::new(2, &mut buffer));
            lower.clear();

            for _ in 0..Table::LEN - 1 {
                lower.get(0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(t, 0).unwrap();
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), 0);
            assert_eq!(pt2.get(1).free(), Table::LEN - 1);
            assert_eq!(count(lower.pt1(pt2.get(1), Table::LEN)), Table::LEN - 1);
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
            let lower = Arc::new(LowerAlloc::new(2, &mut buffer));
            lower.clear();

            pages[0] = lower.get(0, 0).unwrap();
            pages[1] = lower.get(0, 0).unwrap();

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
            let lower = Arc::new(LowerAlloc::new(2, &mut buffer));
            lower.clear();

            for page in &mut pages {
                *page = lower.get(0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                let _stopper = Stopper::init(stop, t as _);

                l.put(pages[t as usize]).unwrap();
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), 2);
            assert_eq!(count(lower.pt1(pt2.get(0), 0)), 2);
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
            let lower = Arc::new(LowerAlloc::new(2, &mut buffer));
            lower.clear();

            for page in &mut pages[..Table::LEN - 1] {
                *page = lower.get(0, 0).unwrap();
            }
            let stop = StopVec::new(2, order);

            let handle = std::thread::spawn({
                let stop = Arc::clone(&stop);
                let lower = lower.clone();
                move || {
                    let _stopper = Stopper::init(stop, 1);

                    lower.get(1, 0).unwrap();
                }
            });

            {
                let _stopper = Stopper::init(stop, 0);

                lower.put(pages[0]).unwrap();
            }

            handle.join().unwrap();

            let pt2 = lower.pt2(0);
            if pt2.get(0).free() == 1 {
                assert_eq!(count(lower.pt1(pt2.get(0), 0)), 1);
            } else {
                // Table entry skipped
                assert_eq!(pt2.get(0).free(), 2);
                assert_eq!(count(lower.pt1(pt2.get(0), 0)), 2);
                assert_eq!(pt2.get(1).free(), Table::LEN - 1);
                assert_eq!(count(lower.pt1(pt2.get(1), Table::LEN)), Table::LEN - 1);
            }
        }
    }

    #[test]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 12;
        let mut buffer = vec![Page::new(); 2 * THREADS * Table::span(2)];

        for _ in 0..64 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(LowerAlloc::new(THREADS, &mut buffer));
            lower.clear();

            let stop = StopRand::new(THREADS, seed);
            let l = lower.clone();
            thread::parallel(THREADS, move |t| {
                let _stopper = Stopper::init(stop, t);

                let mut pages = [0; 4];
                for p in &mut pages {
                    *p = l.get(t, 0).unwrap();
                }
                pages.reverse();
                for p in pages {
                    l.put(p).unwrap();
                }
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), Table::LEN);
            assert_eq!(pt2.get(1).free(), Table::LEN);
            assert_eq!(pt2.get(0).free(), count(lower.pt1(pt2.get(0), 0)));
            assert_eq!(pt2.get(1).free(), count(lower.pt1(pt2.get(1), Table::LEN)));
        }
    }

    #[test]
    fn rand_realloc_last() {
        logging();

        const THREADS: usize = 12;
        let mut pages = [0; Table::LEN];
        let mut buffer = vec![Page::new(); 2 * THREADS * Table::span(2)];

        for _ in 0..64 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(LowerAlloc::new(THREADS, &mut buffer));
            lower.clear();
            for page in &mut pages[..Table::LEN - 3] {
                *page = lower.get(0, 0).unwrap();
            }

            let stop = StopRand::new(THREADS, seed);
            let l = lower.clone();
            thread::parallel(THREADS, move |t| {
                let _stopper = Stopper::init(stop, t);

                if t < THREADS / 2 {
                    l.put(pages[t]).unwrap();
                } else {
                    l.get(t, 0).unwrap();
                }
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free() + pt2.get(1).free(), 3 + Table::LEN);
            assert_eq!(pt2.get(0).free(), count(lower.pt1(pt2.get(0), 0)));
            assert_eq!(pt2.get(1).free(), count(lower.pt1(pt2.get(1), Table::LEN)));
        }
    }
}
