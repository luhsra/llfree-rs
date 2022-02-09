use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use crate::alloc::{Error, Result, Size};
use crate::entry::{Entry1, Entry2, Entry3};
use crate::table::Table;
use crate::util::Atomic;
use crate::Page;

const CAS_RETRIES: usize = 4;

#[cfg(all(test, feature = "wait"))]
macro_rules! wait {
    () => {
        if let Err(e) = crate::wait::wait() {
            panic!("{:?}", e);
        }
    };
}
#[cfg(not(all(test, feature = "wait")))]
macro_rules! wait {
    () => {};
}

pub trait Leafs: Sized {
    fn leafs<'a>() -> &'a [LeafAllocator<Self>];
}

/// Layer 2 page allocator, per core.
#[repr(align(64))]
pub struct LeafAllocator<A: Leafs> {
    pub begin: usize,
    pub pages: usize,
    alloc_pt1: AtomicUsize,
    start_l0: AtomicUsize,
    pte_l0: Atomic<Entry3>,
    start_l1: AtomicUsize,
    pte_l1: Atomic<Entry3>,
    _phantom: PhantomData<A>,
}

impl<A: Leafs> Clone for LeafAllocator<A> {
    fn clone(&self) -> Self {
        Self {
            begin: self.begin,
            pages: self.pages,
            alloc_pt1: AtomicUsize::new(0),
            start_l0: AtomicUsize::new(usize::MAX),
            pte_l0: Atomic::new(Entry3::new()),
            start_l1: AtomicUsize::new(usize::MAX),
            pte_l1: Atomic::new(Entry3::new()),
            _phantom: PhantomData,
        }
    }
}

impl<A: Leafs> LeafAllocator<A> {
    pub fn new(begin: usize, pages: usize) -> Self {
        Self {
            begin,
            pages,
            alloc_pt1: AtomicUsize::new(0),
            start_l0: AtomicUsize::new(usize::MAX),
            pte_l0: Atomic::new(Entry3::new()),
            start_l1: AtomicUsize::new(usize::MAX),
            pte_l1: Atomic::new(Entry3::new()),
            _phantom: PhantomData,
        }
    }

    pub fn clear(&self) {
        // Init pt2
        for i in 0..Table::num_pts(2, self.pages) {
            let pt2 = self.pt2(i * Table::span(2));
            for j in 0..Table::LEN {
                let page = i * Table::span(2) + j * Table::span(1);
                let max = Table::span(1).min(self.pages.saturating_sub(page));
                pt2.set(j, Entry2::new().with_pages(max));
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

    #[inline]
    pub fn pte(&self, size: Size) -> &Atomic<Entry3> {
        if size == Size::L0 {
            &self.pte_l0
        } else {
            &self.pte_l1
        }
    }

    pub fn start(&self, size: Size) -> &AtomicUsize {
        if size == Size::L0 {
            &self.start_l0
        } else {
            &self.start_l1
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
            if Table::page(2, start, i) > self.pages {
                pt.set(i, Entry2::new());
            }

            let pte = pt.get(i);
            if pte.giant() {
                return Ok((0, Size::L2));
            } else if pte.page() {
                size = Size::L1;
            } else if deep && pte.pages() > 0 && size == Size::L0 {
                let p = self.recover_l1(Table::page(1, start, i), pte)?;
                if pte.pages() != p {
                    warn!(
                        "Invalid PTE2 start=0x{start:x} i{i}: {} != {p}",
                        pte.pages(),
                    );
                    pt.set(i, pte.with_pages(p));
                }
                pages += p;
            } else {
                pages += pte.pages();
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
    pub fn get(&self, start: usize) -> Result<usize> {
        let pt2 = self.pt2(start);

        for _ in 0..CAS_RETRIES {
            for newstart in Table::iterate(2, start) {
                let i2 = Table::idx(2, newstart);

                wait!();

                let pte2 = pt2.get(i2);

                if pte2.page() || pte2.pages() == 0 {
                    continue;
                }

                self.alloc_pt1
                    .store(!Table::page(1, start, pte2.i1()), Ordering::SeqCst);

                wait!();

                if let Ok(pte2) = pt2.update(i2, |v| v.dec(pte2.i1())) {
                    let page = if pte2.pages() == 1 {
                        self.get_last(pte2, newstart)
                    } else {
                        self.get_table(pte2, newstart)
                    };
                    self.alloc_pt1.store(0, Ordering::SeqCst);
                    return page;
                }
                self.alloc_pt1.store(0, Ordering::SeqCst);
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

                #[cfg(feature = "wait")]
                if pt1.get(i) != Entry1::Empty {
                    continue;
                } else {
                    wait!();
                }

                if pt1.cas(i, Entry1::Empty, Entry1::Page).is_ok() {
                    return Ok(page);
                }
            }

            warn!("Nothing found, retry");
            wait!();
        }
        error!(
            "Exceeding retries {}-{}: {pte2:?}",
            start / Table::span(2),
            Table::idx(2, start)
        );
        Err(Error::Corruption)
    }

    /// Allocate the last page (the pt1 is reused as last page).
    fn get_last(&self, pte2: Entry2, start: usize) -> Result<usize> {
        wait!();
        info!("alloc last {} s={}", pte2.i1(), start);

        let pt1 = self.pt1(pte2, start);
        let alloc_p1 = !Table::page(1, start, pte2.i1());

        // Wait for others to finish
        for (i, leaf) in A::leafs().iter().enumerate() {
            if leaf as *const _ != self as *const _ {
                while leaf.alloc_pt1.load(Ordering::SeqCst) == alloc_p1 {
                    warn!("Waiting for cpu {} on {}", i, unsafe {
                        (self as *const Self).offset_from(&A::leafs()[0] as *const _)
                    });
                    wait!();
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

    /// Free single page
    pub fn put(&self, page: usize) -> Result<Size> {
        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);

        wait!();

        let mut old = pt2.get(i2);
        if old.page() {
            // Free huge page
            if page % Table::span(Size::L1 as _) != 0 {
                error!("Invalid address {}", page);
                return Err(Error::Address);
            }

            let pt1 = unsafe { &*((self.begin + page * Page::SIZE) as *const Table<Entry1>) };
            pt1.clear();

            match pt2.cas(i2, old, Entry2::new_table(Table::LEN, 0)) {
                Ok(_) => Ok(Size::L1),
                Err(_) => {
                    error!("Corruption l2 i{}", i2);
                    Err(Error::Corruption)
                }
            }
        } else if !old.giant() && old.pages() < Table::LEN {
            for _ in 0..CAS_RETRIES {
                match self.put_small(old, page).map(|_| Size::L0) {
                    Err(Error::CAS) => old = pt2.get(i2),
                    e => return e,
                }
            }
            error!("Exceeding retries {page:x}");
            Err(Error::Corruption)
        } else {
            Err(Error::Address)
        }
    }

    fn put_small(&self, pte2: Entry2, page: usize) -> Result<()> {
        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);

        if pte2.pages() == 0 {
            return self.put_full(pte2, page);
        }

        wait!();

        let pt1 = self.pt1(pte2, page);
        let i1 = Table::idx(1, page);
        let pte1 = pt1.get(i1);

        if pte1 != Entry1::Page {
            error!("Invalid Addr l1 i{i1} p={page}");
            return Err(Error::Address);
        }

        wait!();

        if let Err(pte2) = pt2.update(i2, |pte| pte.inc(pte2.i1())) {
            return if pte2.pages() == Table::LEN {
                error!("Invalid Addr l1 i{i1} p={page}");
                Err(Error::Address)
            } else {
                Err(Error::CAS)
            };
        }

        wait!();

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

        wait!();

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
        for i in 0..Table::LEN {
            // i1 is initially 0
            let pt1 = unsafe {
                &*((self.begin + (page + i * Table::span(1)) * Page::SIZE) as *const Table<Entry1>)
            };
            pt1.clear(); // set all to 0 -> empty
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
                "{:1$}l2 i={2} 0x{3:x}: {4:?}",
                "",
                (Table::LAYERS - 2) * 4,
                i2,
                start * Page::SIZE,
                pte2
            );
            if !pte2.giant() && !pte2.page() && pte2.pages() > 0 && pte2.pages() < Table::LEN {
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

#[cfg(feature = "wait")]
#[cfg(test)]
mod test {
    use std::sync::Arc;

    use log::warn;

    use super::Leafs;
    use crate::alloc::{array_aligned::ArrayAlignedAlloc, Alloc};
    use crate::entry::Entry1;
    use crate::table::Table;
    use crate::thread;
    use crate::util::{logging, Page};
    use crate::wait::{DbgWait, DbgWaitKey};

    type Allocator = ArrayAlignedAlloc;

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
            Allocator::init(2, &mut buffer, true).unwrap();
            Allocator::leafs()[0].get(0).unwrap();

            let wait = DbgWait::setup(2, order);

            thread::parallel(2, move |t| {
                thread::pin(t);
                let key = DbgWaitKey::init(wait, t as _);
                let page_alloc = &Allocator::leafs()[t];

                let page = page_alloc.get(0).unwrap();
                drop(key);
                assert!(page != 0);
            });

            let page_alloc = &Allocator::leafs()[0];
            assert_eq!(page_alloc.pt2(0).get(0).pages(), Table::LEN - 3);
            assert_eq!(
                count(page_alloc.pt1(page_alloc.pt2(0).get(0), 0)),
                Table::LEN - 3
            );

            Allocator::destroy()
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
            Allocator::init(2, &mut buffer, true).unwrap();

            let wait = DbgWait::setup(2, order);

            thread::parallel(2, move |t| {
                thread::pin(t);
                let _key = DbgWaitKey::init(wait, t as _);
                let page_alloc = &Allocator::leafs()[t];

                page_alloc.get(0).unwrap();
            });

            let page_alloc = &Allocator::leafs()[0];
            assert_eq!(page_alloc.pt2(0).get(0).pages(), Table::LEN - 2);
            assert_eq!(
                count(page_alloc.pt1(page_alloc.pt2(0).get(0), 0)),
                Table::LEN - 2
            );

            Allocator::destroy()
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
            Allocator::init(2, &mut buffer, true).unwrap();
            let page_alloc = &Allocator::leafs()[0];
            for _ in 0..Table::LEN - 1 {
                page_alloc.get(0).unwrap();
            }
            let wait = DbgWait::setup(2, order);

            thread::parallel(2, move |t| {
                thread::pin(t);
                let _key = DbgWaitKey::init(wait, t as _);
                let page_alloc = &Allocator::leafs()[t];

                page_alloc.get(0).unwrap();
            });

            let page_alloc = &Allocator::leafs()[0];
            let pt2 = page_alloc.pt2(0);
            assert_eq!(pt2.get(0).pages(), 0);
            assert_eq!(pt2.get(1).pages(), Table::LEN - 1);
            assert_eq!(
                count(page_alloc.pt1(pt2.get(1), Table::LEN)),
                Table::LEN - 1
            );

            Allocator::destroy()
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
            Allocator::init(2, &mut buffer, true).unwrap();
            let page_alloc = &Allocator::leafs()[0];
            pages[0] = page_alloc.get(0).unwrap();
            pages[1] = page_alloc.get(0).unwrap();
            let wait = DbgWait::setup(2, order);

            thread::parallel(2, {
                let pages = pages.clone();
                move |t| {
                    let _key = DbgWaitKey::init(wait, t as _);
                    let page_alloc = &Allocator::leafs()[t];

                    page_alloc.put(pages[t as usize]).unwrap();
                }
            });

            let page_alloc = &Allocator::leafs()[0];
            assert_eq!(page_alloc.pt2(0).get(0).pages(), Table::LEN);

            Allocator::destroy()
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
            Allocator::init(2, &mut buffer, true).unwrap();
            let page_alloc = &Allocator::leafs()[0];
            for page in &mut pages {
                *page = page_alloc.get(0).unwrap();
            }
            let wait = DbgWait::setup(2, order);

            thread::parallel(2, {
                let pages = pages.clone();
                move |t| {
                    let _key = DbgWaitKey::init(wait, t as _);
                    let page_alloc = &Allocator::leafs()[t];

                    page_alloc.put(pages[t as usize]).unwrap();
                }
            });

            let page_alloc = &Allocator::leafs()[0];
            let pt2 = page_alloc.pt2(0);
            assert_eq!(pt2.get(0).pages(), 2);
            assert_eq!(count(page_alloc.pt1(pt2.get(0), 0)), 2);

            Allocator::destroy()
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
            Allocator::init(2, &mut buffer, true).unwrap();
            let page_alloc = &Allocator::leafs()[0];
            for page in &mut pages[..Table::LEN - 1] {
                *page = page_alloc.get(0).unwrap();
            }
            let wait = DbgWait::setup(2, order);

            let wait_clone = Arc::clone(&wait);
            let handle = std::thread::spawn(move || {
                let _key = DbgWaitKey::init(wait_clone, 1);
                let page_alloc = &Allocator::leafs()[1];

                page_alloc.get(0).unwrap();
            });

            {
                let _key = DbgWaitKey::init(wait, 0);
                let page_alloc = &Allocator::leafs()[0];

                page_alloc.put(pages[0]).unwrap();
            }

            handle.join().unwrap();

            let page_alloc = &Allocator::leafs()[0];
            let pt2 = page_alloc.pt2(0);
            if pt2.get(0).pages() == 1 {
                assert_eq!(count(page_alloc.pt1(pt2.get(0), 0)), 1);
            } else {
                // Table entry skipped
                assert_eq!(pt2.get(0).pages(), 2);
                assert_eq!(count(page_alloc.pt1(pt2.get(0), 0)), 2);
                assert_eq!(pt2.get(1).pages(), Table::LEN - 1);
                assert_eq!(
                    count(page_alloc.pt1(pt2.get(1), Table::LEN)),
                    Table::LEN - 1
                );
            }

            Allocator::destroy()
        }
    }
}
