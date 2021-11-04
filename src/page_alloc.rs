use log::{error, info, warn};

use crate::paging::{Entry, Table, LAYERS, PAGE_SIZE, PT_LEN, PT_LEN_BITS};

use crate::{Error, Result};

/// Layer 2 page allocator.
pub struct PageAllocator {
    begin: usize,
    pages: usize,
}

impl PageAllocator {
    pub fn new(begin: usize, pages: usize) -> Self {
        Self { begin, pages }
    }

    /// Returns the l1 page table that contains the `page`.
    fn pt1(&self, pte2: Entry, page: usize) -> Option<&Table> {
        if pte2.is_table() && pte2.pages() < PT_LEN {
            let start = page & !(PT_LEN - 1);
            Some(unsafe { &*((self.begin + (start + pte2.i1()) * PAGE_SIZE) as *const Table) })
        } else {
            None
        }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages & PT1 | PT2 | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table {
        let i = page >> (PT_LEN_BITS * 2);
        // The l2 tables are stored after this area
        let pt2 = (self.begin + self.pages * PAGE_SIZE) as *mut Table;
        unsafe { &mut *pt2.add(i) }
    }

    /// Allocate a single page
    pub fn alloc(&self, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);

        for i2 in Table::range(2, start..self.pages) {
            let start = start + i2 * PT_LEN;
            let pte2 = pt2.get(i2);
            info!("pte2={:?}", pte2);

            if pte2.pages() >= PT_LEN {
                continue;
            }

            if pte2.pages() == PT_LEN - 1 {
                return self.alloc_last(pte2, start);
            }
            if pte2.is_empty() {
                return self.alloc_first(start);
            }
            return self.alloc_pt(pte2, start);
        }
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn alloc_pt(&self, pte2: Entry, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);
        let pt1 = self.pt1(pte2, start).ok_or(Error::Memory)?;

        for i in Table::range(1, start..self.pages) {
            let page = start + i;

            if let Ok(_) = pt1.cas(i, Entry::empty(), Entry::page()) {
                info!("alloc l1 i={}: {}", i, page);

                let i2 = Table::idx(2, page);
                return match pt2.inc(i2, 1, 1, pte2.i1()) {
                    Ok(pte) => Ok((page, pte.pages() == 0)),
                    Err(pte) => {
                        error!("CAS: inc alloc pt {:?}", pte);
                        Err(Error::Memory)
                    }
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
        let i2 = Table::idx(2, start);

        // store pt1 at i=1, so that the allocated page is at i=0
        let pt1 = unsafe { &*((self.begin + (start + 1) * PAGE_SIZE) as *const Table) };
        pt1.set(0, Entry::page());
        pt1.set(1, Entry::page_reserved());
        for i1 in 2..PT_LEN {
            pt1.set(i1, Entry::empty());
        }

        match pt2.cas(i2, Entry::empty(), Entry::table(1, 1, 1, false)) {
            Ok(_) => Ok((start, true)),
            Err(pte) => {
                warn!("CAS: init pt1 {:?}", pte);
                Err(Error::CAS)
            }
        }
    }

    /// Allocate the last page (the pt1 is reused as last page).
    fn alloc_last(&self, pte2: Entry, start: usize) -> Result<(usize, bool)> {
        let pt2 = self.pt2(start);
        let i2 = Table::idx(2, start);

        assert!(pte2.is_table() && pte2.pages() == PT_LEN - 1);

        info!("alloc last {} s={}", pte2.i1(), start);

        // Only modify pt2 as pt1 is returned as page.
        match pt2.cas(i2, pte2, Entry::table(PT_LEN, PT_LEN, 0, false)) {
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
        let i2 = Table::idx(2, page);

        let pte2 = pt2.get(i2);
        if pte2.is_page() {
            return Err(Error::Address);
        } else if pte2.pages() < PT_LEN {
            if let Some(pt1) = self.pt1(pte2, page) {
                let i1 = Table::idx(1, page);
                if let Err(pte) = pt1.cas(i1, Entry::page(), Entry::empty()) {
                    error!("CAS: free unexpected l1 {:?}", pte);
                    return Err(Error::Address);
                }
            } else {
                error!("free no pt1");
                return Err(Error::Address);
            }

            // i2 is expected to be unchanged.
            // If not the page table has been moved in between, which should not be possible!
            match pt2.dec(i2, 1, 1, pte2.i1()) {
                Ok(pte) => Ok(pte.pages() == 1),
                Err(pte) => {
                    error!("CAS: free dec l2 {:?} (i1={})", pte, pte2.i1());
                    Err(Error::Address)
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
        let i = Table::idx(2, page);

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
