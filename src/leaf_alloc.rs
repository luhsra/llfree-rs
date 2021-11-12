use log::{error, info, warn};

use crate::entry::{L1Entry, L2Entry};
use crate::table::{self, Table, LAYERS, PAGE_SIZE, PT_LEN, PT_LEN_BITS};

use crate::{Error, Result};

#[cfg(test)]
macro_rules! wait {
    () => {
        if let Err(e) = crate::sync::wait() {
            error!("{:?}", e);
            panic!("{:?}", e);
        }
    };
}
#[cfg(not(test))]
macro_rules! wait {
    () => {};
}

/// Layer 2 page allocator.
pub struct LeafAllocator {
    begin: usize,
    pages: usize,
}

impl LeafAllocator {
    pub fn new(begin: usize, pages: usize) -> Self {
        Self { begin, pages }
    }

    /// Returns the l1 page table that contains the `page`.
    fn pt1(&self, pte2: L2Entry, page: usize) -> Option<&Table<L1Entry>> {
        if pte2.has_i1() {
            let start = page & !(PT_LEN - 1);
            Some(unsafe {
                &*((self.begin + (start + pte2.i1()) * PAGE_SIZE) as *const Table<L1Entry>)
            })
        } else {
            None
        }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages & PT1 | PT2 | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table<L2Entry> {
        let i = page >> (PT_LEN_BITS * 2);
        // The l2 tables are stored after this area
        let pt2 = (self.begin + (self.pages + i) * PAGE_SIZE) as *mut Table<L2Entry>;
        unsafe { &mut *pt2 }
    }

    /// Allocate a single page
    pub fn alloc(&self, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);

        for i2 in table::range(2, start..self.pages) {
            let start = start + i2 * PT_LEN;

            wait!();

            let pte2 = pt2.get(i2);
            info!("i={} pte2={:?}", i2, pte2);

            if pte2.pages() >= PT_LEN {
                continue;
            }

            if pte2.pages() == 0 {
                return if pte2.is_reserved() {
                    Err(Error::CAS)
                } else {
                    self.alloc_first(start)
                };
            }

            if pte2.pages() == PT_LEN - 1 {
                if pte2.usage() > 0 {
                    continue; // Skip if free is in progress
                }
                return self.alloc_last(pte2, start);
            }
            match self.alloc_pt(pte2, start) {
                Err(Error::Memory) => {} // table full before inc
                r => return r,
            }
        }
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn alloc_pt(&self, pte2: L2Entry, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);
        let pt1 = self.pt1(pte2, start).ok_or(Error::Memory)?;

        for i in table::range(1, start..self.pages) {
            let page = start + i;

            wait!();

            if pt1.cas(i, L1Entry::empty(), L1Entry::page()).is_ok() {
                info!("alloc l1 i={}: {}", i, page);

                let i2 = table::idx(2, page);

                wait!();

                return match pt2.update(i2, |v| {
                    if v.has_i1() && v.pages() < PT_LEN {
                        Some(L2Entry::table(
                            v.pages() + 1,
                            v.usage(),
                            v.i1(),
                            v.is_reserved(),
                        ))
                    } else {
                        None
                    }
                }) {
                    Ok(pte) => Ok((page, pte.pages() == 0)),
                    Err(pte) => panic!("Corruption l2 i{} {:?}", i2, pte),
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
        let i2 = table::idx(2, start);

        wait!();

        match pt2.cas(i2, L2Entry::empty(), L2Entry::page_reserved()) {
            Ok(_) => {}
            Err(pte) => {
                warn!("CAS: init pt1 {:?}", pte);
                return Err(Error::CAS);
            }
        }

        // store pt1 at i=1, so that the allocated page is at i=0
        let pt1 = unsafe { &*((self.begin + (start + 1) * PAGE_SIZE) as *const Table<L1Entry>) };
        pt1.set(0, L1Entry::page());
        pt1.set(1, L1Entry::reserved());
        for i1 in 2..PT_LEN {
            pt1.set(i1, L1Entry::empty());
        }

        wait!();

        match pt2.cas(i2, L2Entry::page_reserved(), L2Entry::table(1, 0, 1, false)) {
            Ok(_) => Ok((start, true)),
            Err(pte) => Err(Error::Corruption(2, i2, pte.into())),
        }
    }

    /// Allocate the last page (the pt1 is reused as last page).
    fn alloc_last(&self, pte2: L2Entry, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);
        let i2 = table::idx(2, start);

        assert!(pte2.has_i1() && pte2.pages() == PT_LEN - 1);

        wait!();
        info!("alloc last {} s={}", pte2.i1(), start);

        // Only modify pt2 as pt1 is returned as page.
        match pt2.cas(i2, pte2, L2Entry::table(PT_LEN, 0, 0, false)) {
            Ok(_) => Ok((start + pte2.i1(), false)),
            Err(pte) => {
                warn!("CAS: alloc last pt2 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    /// Free single page
    pub fn free(&self, page: usize) -> Result<bool> {
        info!("free leaf page {}", page);

        let pt2 = self.pt2(page);
        let i2 = table::idx(2, page);

        wait!();

        match pt2.update(i2, L2Entry::inc_usage) {
            Ok(pte2) => {
                let pt1 = self.pt1(pte2, page).unwrap();
                let i1 = table::idx(1, page);

                wait!();

                match pt1.cas(i1, L1Entry::page(), L1Entry::empty()) {
                    Ok(_) => {
                        wait!();
                        info!("free dec i={} {:?}", i2, pte2);

                        // i1 is expected to be unchanged.
                        // If not the page table has been moved in between, which should not be possible!
                        match pt2.update(i2, |v| v.dec_all(pte2.i1())) {
                            Ok(pte) => Ok(pte.pages() == 1),
                            Err(pte) => panic!("Corruption: l2 i{} {:?}", i2, pte),
                        }
                    }
                    Err(_) => {
                        error!("free invalid {}", page);
                        pt2.update(i2, L2Entry::dec_usage).unwrap();
                        Err(Error::Address)
                    }
                }
            }
            // Free last page of pt1 & rebuild pt1
            Err(pte2) if pte2.pages() == PT_LEN => self.free_full(page),
            // Large / huge page
            Err(_) => Err(Error::Address),
        }
    }

    /// Free last page & rebuild pt1 in it
    fn free_full(&self, page: usize) -> Result<bool> {
        // The new pt
        let pt2 = self.pt2(page);
        let i = table::idx(2, page);

        wait!();

        let pt1 = unsafe { &*((self.begin + page * PAGE_SIZE) as *const Table<L1Entry>) };
        info!("free: init last pt1 {}", page);

        for j in 0..PT_LEN {
            if j == page % PT_LEN {
                pt1.set(j, L1Entry::reserved());
            } else {
                pt1.set(j, L1Entry::page());
            }
        }

        match pt2.cas(
            i,
            L2Entry::table(PT_LEN, 0, 0, false),
            L2Entry::table(PT_LEN - 1, 0, page % PT_LEN, false),
        ) {
            Ok(_) => Ok(false),
            Err(pte) => {
                warn!("CAS: create pt1 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    pub fn dump(&self, start: usize) {
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
    use std::alloc::{alloc_zeroed, Layout};
    use std::thread;

    use log::warn;

    use crate::entry::L1Entry;
    use crate::sync;
    use crate::table::{Table, PAGE_SIZE, PT_LEN};
    use crate::util::{logging, parallel};

    use super::LeafAllocator;

    fn aligned_buffer(size: usize) -> Vec<u8> {
        let buffer = unsafe {
            Vec::from_raw_parts(
                alloc_zeroed(Layout::from_size_align_unchecked(size, PAGE_SIZE)),
                size,
                size,
            )
        };
        assert!(buffer.as_ptr() as usize % PAGE_SIZE == 0);
        buffer
    }

    fn count(pt: &Table<L1Entry>) -> usize {
        let mut pages = 0;
        for i in 0..PT_LEN {
            pages += (pt.get(i) == L1Entry::page()) as usize;
        }
        pages
    }

    #[test]
    fn alloc_normal() {
        logging();
        const MEM_SIZE: usize = (PT_LEN + 1) * PAGE_SIZE;
        let buffer = aligned_buffer(MEM_SIZE);

        // init
        {
            let page_alloc = LeafAllocator::new(buffer.as_ptr() as _, PT_LEN);
            let (page, newentry) = page_alloc.alloc(0).unwrap();
            warn!("setup single alloc {} {}", page, newentry);
        }

        let orders = [
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], // first 0 complete, 1 fails cas
            vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], // 0 first cas, but inc out of order
            vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], // 1 first cas, but inc out of order
            vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0], // 1 first cas, but inc out of order
        ];

        for order in orders {
            warn!("order: {:?}", order);
            let copy = buffer.clone();
            let begin = copy.as_ptr() as usize;
            sync::setup(2, order);

            parallel(2, move |t| {
                sync::init(t).unwrap();
                let page_alloc = LeafAllocator::new(begin, PT_LEN);

                let (page, newentry) = page_alloc.alloc(0).unwrap();
                sync::end().unwrap();
                assert!(page != 0 && !newentry);
            });

            let page_alloc = LeafAllocator::new(begin, PT_LEN);
            assert_eq!(page_alloc.pt2(0).get(0).pages(), 3);
            assert_eq!(
                count(page_alloc.pt1(page_alloc.pt2(0).get(0), 0).unwrap()),
                3
            );
        }
    }

    #[test]
    fn alloc_first() {
        logging();
        const MEM_SIZE: usize = (PT_LEN + 1) * PAGE_SIZE;
        let buffer = aligned_buffer(MEM_SIZE);

        let orders = [
            vec![0, 0, 0, 1, 1, 1, 1, 1],       // first 0 complete, 1 normal
            vec![0, 1, 0, 0, 1, 1, 1, 1, 1, 1], // 1 fails cas
            vec![1, 0, 1, 1, 0, 0, 0, 0, 0, 0], // 0 fails cas
            vec![1, 0, 1, 0, 1, 0, 0, 0, 0, 0], // 0 fails cas
        ];

        for order in orders {
            warn!("order: {:?}", order);
            let copy = buffer.clone();
            let begin = copy.as_ptr() as usize;
            sync::setup(2, order);

            parallel(2, move |t| {
                sync::init(t).unwrap();
                let page_alloc = LeafAllocator::new(begin, PT_LEN);

                match page_alloc.alloc(0) {
                    Err(crate::Error::CAS) => {
                        page_alloc.alloc(0).unwrap();
                    }
                    Err(e) => panic!("{:?}", e),
                    Ok(_) => {}
                }
                sync::end().unwrap();
            });

            let page_alloc = LeafAllocator::new(begin, PT_LEN);
            assert_eq!(page_alloc.pt2(0).get(0).pages(), 2);
            assert_eq!(
                count(page_alloc.pt1(page_alloc.pt2(0).get(0), 0).unwrap()),
                2
            );
        }
    }

    #[test]
    fn alloc_last() {
        logging();
        const MEM_SIZE: usize = 2 * (PT_LEN + 1) * PAGE_SIZE;
        let buffer = aligned_buffer(MEM_SIZE);

        let orders = [
            vec![0, 0, 1, 1, 1, 1],       // first 0 complete, 1 normal
            vec![0, 1, 0, 1, 1, 1, 1, 1], // 1 fails cas
            vec![1, 0, 0, 1, 1, 1, 1, 1], // 1 fails cas
            vec![1, 0, 1, 0, 0, 0, 0, 0], // 0 fails cas
        ];

        // init
        {
            let page_alloc = LeafAllocator::new(buffer.as_ptr() as _, 2 * PT_LEN);
            for _ in 0..PT_LEN - 1 {
                page_alloc.alloc(0).unwrap();
            }
            warn!("setup single alloc");
        }

        for order in orders {
            warn!("order: {:?}", order);
            let copy = buffer.clone();
            let begin = copy.as_ptr() as usize;
            sync::setup(2, order);

            parallel(2, move |t| {
                sync::init(t).unwrap();
                let page_alloc = LeafAllocator::new(begin, 2 * PT_LEN);

                match page_alloc.alloc(0) {
                    Err(crate::Error::CAS) => {
                        page_alloc.alloc(0).unwrap();
                    }
                    Err(e) => panic!("{:?}", e),
                    Ok(_) => {}
                }
                sync::end().unwrap();
            });

            let page_alloc = LeafAllocator::new(begin, 2 * PT_LEN);
            let pt2 = page_alloc.pt2(0);
            assert_eq!(pt2.get(0).pages(), PT_LEN);
            assert_eq!(pt2.get(1).pages(), 1);
            assert_eq!(count(page_alloc.pt1(pt2.get(1), PT_LEN).unwrap()), 1);
        }
    }

    #[test]
    fn free_normal() {
        logging();
        const MEM_SIZE: usize = (PT_LEN + 1) * PAGE_SIZE;
        let buffer = aligned_buffer(MEM_SIZE);

        let orders = [
            vec![0, 0, 0, 1, 1, 1], // first 0, then 1
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 1, 0],
        ];

        let mut pages = [0; 2];

        // init
        {
            let page_alloc = LeafAllocator::new(buffer.as_ptr() as _, PT_LEN);
            pages[0] = page_alloc.alloc(0).unwrap().0;
            pages[1] = page_alloc.alloc(0).unwrap().0;
            warn!("setup single alloc");
        }

        for order in orders {
            warn!("order: {:?}", order);
            let copy = buffer.clone();
            let begin = copy.as_ptr() as usize;
            sync::setup(2, order);

            parallel(2, {
                let pages = pages.clone();
                move |t| {
                    sync::init(t).unwrap();
                    let page_alloc = LeafAllocator::new(begin, PT_LEN);

                    match page_alloc.free(pages[t as usize]) {
                        Err(crate::Error::CAS) => {
                            page_alloc.free(pages[t as usize]).unwrap();
                        }
                        Err(e) => panic!("{:?}", e),
                        Ok(_) => {}
                    }
                    sync::end().unwrap();
                }
            });

            let page_alloc = LeafAllocator::new(begin, PT_LEN);
            assert_eq!(page_alloc.pt2(0).get(0).pages(), 0);
        }
    }

    #[test]
    fn free_last() {
        logging();
        const MEM_SIZE: usize = (PT_LEN + 1) * PAGE_SIZE;
        let buffer = aligned_buffer(MEM_SIZE);

        let orders = [
            vec![0, 0, 1, 1, 1],       // first 0, then 1
            vec![0, 1, 0, 1, 1, 1, 1], // 1 fails cas
            vec![0, 1, 1, 0, 0, 0, 0], // 0 fails cas
        ];

        let mut pages = [0; PT_LEN];

        // init
        {
            let page_alloc = LeafAllocator::new(buffer.as_ptr() as _, PT_LEN);
            for page in &mut pages {
                *page = page_alloc.alloc(0).unwrap().0;
            }
            warn!("setup single alloc");
        }

        for order in orders {
            warn!("order: {:?}", order);
            let buffer = buffer.clone();
            let begin = buffer.as_ptr() as usize;
            sync::setup(2, order);

            parallel(2, {
                let pages = pages.clone();
                move |t| {
                    sync::init(t).unwrap();
                    let page_alloc = LeafAllocator::new(begin, PT_LEN);

                    match page_alloc.free(pages[t as usize]) {
                        Err(crate::Error::CAS) => {
                            page_alloc.free(pages[t as usize]).unwrap();
                        }
                        Err(e) => panic!("{:?}", e),
                        Ok(_) => {}
                    }
                    sync::end().unwrap();
                }
            });

            let page_alloc = LeafAllocator::new(begin, PT_LEN);
            let pt2 = page_alloc.pt2(0);
            assert_eq!(pt2.get(0).pages(), PT_LEN - 2);
            assert_eq!(count(page_alloc.pt1(pt2.get(0), 0).unwrap()), PT_LEN - 2);
        }
    }

    #[test]
    fn alloc_free_last() {
        logging();

        const MEM_SIZE: usize = 2 * (PT_LEN + 1) * PAGE_SIZE;
        let buffer = aligned_buffer(MEM_SIZE);

        let orders = [
            vec![0, 0, 0, 1, 1, 1],       // 0 free then 1 alloc
            vec![1, 1, 0, 0],             // 1 alloc last then 0 free last
            vec![0, 1, 1, 1, 1, 0, 0],    // 1 skips table
            vec![1, 0, 1, 0, 0, 1, 1, 1], // 1 fails cas
        ];

        let mut pages = [0; PT_LEN];

        {
            let page_alloc = LeafAllocator::new(buffer.as_ptr() as _, 2 * PT_LEN);
            for page in &mut pages[..PT_LEN - 1] {
                *page = page_alloc.alloc(0).unwrap().0;
            }
            warn!("setup single alloc");
        }

        for order in orders {
            warn!("order: {:?}", order);
            let buffer = buffer.clone();
            let begin = buffer.as_ptr() as usize;
            sync::setup(2, order);

            let handle = thread::spawn(move || {
                sync::init(1).unwrap();
                let page_alloc = LeafAllocator::new(begin, 2 * PT_LEN);

                match page_alloc.alloc(0) {
                    Err(crate::Error::CAS) => {
                        page_alloc.alloc(0).unwrap();
                    }
                    Err(e) => panic!("{:?}", e),
                    Ok(_) => {}
                }
                sync::end().unwrap();
            });

            {
                sync::init(0).unwrap();
                let page_alloc = LeafAllocator::new(begin, 2 * PT_LEN);

                match page_alloc.free(pages[0]) {
                    Err(crate::Error::CAS) => {
                        page_alloc.free(pages[0]).unwrap();
                    }
                    Err(e) => panic!("{:?}", e),
                    Ok(_) => {}
                }
                sync::end().unwrap();
            }

            handle.join().unwrap();

            let page_alloc = LeafAllocator::new(begin, 2 * PT_LEN);
            let pt2 = page_alloc.pt2(0);
            if pt2.get(0).pages() == PT_LEN - 1 {
                assert_eq!(count(page_alloc.pt1(pt2.get(0), 0).unwrap()), PT_LEN - 1);
            } else {
                // Table entry skipped
                assert_eq!(pt2.get(0).pages(), PT_LEN - 2);
                assert_eq!(count(page_alloc.pt1(pt2.get(0), 0).unwrap()), PT_LEN - 2);
                assert_eq!(pt2.get(1).pages(), 1);
                assert_eq!(count(page_alloc.pt1(pt2.get(1), PT_LEN).unwrap()), 1);
            }
        }
    }
}
