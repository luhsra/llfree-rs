use core::fmt::Write;
use core::ops::Range;

use alloc::string::String;
use log::{error, warn};

use crate::entry::SmallEntry2;
use crate::table::{ATable, Bitfield, Mapping};
use crate::upper::CAS_RETRIES;
use crate::util::{align_up, div_ceil, Page};
use crate::{Error, Result};

use super::LowerAlloc;

/// Level 2 page allocator.
/// ```text
/// NVRAM: [ Pages | PT1s + padding | PT2s | Meta ]
/// ```
#[derive(Default, Debug)]
pub struct CacheLower<const T2N: usize> {
    pub begin: usize,
    pub pages: usize,
}

impl<const T2N: usize> LowerAlloc for CacheLower<T2N> {
    const MAPPING: Mapping<2> = Mapping([Bitfield::ORDER, ATable::<SmallEntry2, T2N>::ORDER]);
    const HUGE_ORDER: usize = Self::MAPPING.order(1);

    fn new(_cores: usize, memory: &mut [Page]) -> Self {
        let s1 = Self::MAPPING.num_pts(1, memory.len()) * Bitfield::SIZE;
        let s1 = align_up(s1, ATable::<SmallEntry2, T2N>::SIZE); // correct alignment
        let s2 = Self::MAPPING.num_pts(2, memory.len()) * ATable::<SmallEntry2, T2N>::SIZE;
        let pages = div_ceil(s1 + s2, Page::SIZE);
        Self {
            begin: memory.as_ptr() as usize,
            // level 1 and 2 tables are stored at the end of the NVM
            pages: memory.len() - pages,
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
        for i in 0..Self::MAPPING.num_pts(2, self.pages) {
            let pt2 = self.pt2(i * Self::MAPPING.span(2));
            if i + 1 < Self::MAPPING.num_pts(2, self.pages) {
                pt2.fill(SmallEntry2::new().with_free(Self::MAPPING.span(1)));
            } else {
                for j in 0..Self::MAPPING.len(2) {
                    let page = i * Self::MAPPING.span(2) + j * Self::MAPPING.span(1);
                    let max = Self::MAPPING.span(1).min(self.pages.saturating_sub(page));
                    pt2.set(j, SmallEntry2::new().with_free(max));
                }
            }
        }
        // Init pt1
        for i in 0..Self::MAPPING.num_pts(1, self.pages) {
            let pt1 = self.pt1(i * Self::MAPPING.span(1));

            if i + 1 < Self::MAPPING.num_pts(1, self.pages) {
                pt1.fill(false);
            } else {
                for j in 0..Self::MAPPING.len(1) {
                    let page = i * Self::MAPPING.span(1) + j;
                    pt1.set(j, page >= self.pages);
                }
            }
        }
    }

    fn recover(&self, start: usize, deep: bool) -> Result<(usize, bool)> {
        let mut pages = 0;
        let mut huge = false;

        let pt = self.pt2(start);
        for i in 0..Self::MAPPING.len(2) {
            let start = Self::MAPPING.page(2, start, i);
            if start > self.pages {
                pt.set(i, SmallEntry2::new());
            }

            let pte = pt.get(i);
            if pte.page() {
                huge = true;
            } else if deep && pte.free() > 0 && !huge {
                let p = self.recover_l1(start);
                if pte.free() != p {
                    warn!("Invalid PTE2 start=0x{start:x} i{i}: {} != {p}", pte.free());
                    pt.set(i, pte.with_free(p));
                }
                pages += p;
            } else {
                pages += pte.free();
            }
        }

        Ok((pages, huge))
    }

    fn get(&self, _core: usize, order: usize, start: usize) -> Result<usize> {
        debug_assert!(order <= Self::MAPPING.order(1));

        if order > 0 {
            // allocate huge page
            let pt = self.pt2(start);
            for _ in 0..CAS_RETRIES {
                for page in Self::MAPPING.iterate(2, start) {
                    let i = Self::MAPPING.idx(2, page);
                    if pt.update(i, |v| v.mark_huge(Self::MAPPING.span(1))).is_ok() {
                        return Ok(page);
                    }
                }
            }
            error!("Nothing found {}", start / Self::MAPPING.span(2));
            Err(Error::Corruption)
        } else {
            self.get_small(start)
        }
    }

    /// Free single page and returns if the page was huge
    fn put(&self, page: usize, order: usize) -> Result<()> {
        debug_assert!(page < self.pages);
        stop!();

        let pt2 = self.pt2(page);
        let i2 = Self::MAPPING.idx(2, page);
        if order == 0 {
            let old = pt2.get(i2);
            if old.free() <= Self::MAPPING.span(1) - (1 << order) {
                self.put_small(page)
            } else {
                error!("Addr p={page:x} o={order} {old:?}");
                Err(Error::Address)
            }
        } else {
            // try free huge
            if let Err(old) = pt2.cas(
                i2,
                SmallEntry2::new().with_page(true),
                SmallEntry2::new().with_free(Self::MAPPING.span(1)),
            ) {
                error!("Addr {page:x} o={order} {old:?}");
                Err(Error::Address)
            } else {
                Ok(())
            }
        }
    }

    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages;
        for i in 0..Self::MAPPING.num_pts(2, self.pages) {
            let start = i * Self::MAPPING.span(2);
            let pt2 = self.pt2(start);
            for i2 in Self::MAPPING.range(2, start..self.pages) {
                let start = Self::MAPPING.page(2, start, i2);
                let pte2 = pt2.get(i2);

                pages -= if pte2.page() {
                    Self::MAPPING.span(1)
                } else {
                    let pt1 = self.pt1(start);
                    let mut child_pages = 0;
                    for i1 in Self::MAPPING.range(1, start..self.pages) {
                        child_pages += !pt1.get(i1) as usize;
                    }
                    assert_eq!(child_pages, pte2.free());
                    child_pages
                }
            }
        }
        pages
    }
}

impl<const T2N: usize> CacheLower<T2N> {
    /// Returns the l1 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | padding | PT1s | PT2s | Meta ]
    /// ```
    fn pt1(&self, page: usize) -> &Bitfield {
        let mut offset = self.begin + self.pages * Page::SIZE;

        let i = page / Self::MAPPING.span(1);
        debug_assert!(i < Self::MAPPING.num_pts(1, self.pages));
        offset += i * Bitfield::SIZE;
        unsafe { &*(offset as *const Bitfield) }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | padding | PT1s | PT2s | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &ATable<SmallEntry2, T2N> {
        let mut offset = self.begin + self.pages * Page::SIZE;
        offset += Self::MAPPING.num_pts(1, self.pages) * Bitfield::SIZE;
        offset = align_up(offset, ATable::<SmallEntry2, T2N>::SIZE); // correct alignment

        let i = page / Self::MAPPING.span(2);
        debug_assert!(i < Self::MAPPING.num_pts(2, self.pages));
        offset += i * ATable::<SmallEntry2, T2N>::SIZE;
        unsafe { &*(offset as *mut ATable<SmallEntry2, T2N>) }
    }

    fn recover_l1(&self, start: usize) -> usize {
        let pt = self.pt1(start);
        let mut pages = 0;
        for i in Self::MAPPING.range(1, start..self.pages) {
            pages += !pt.get(i) as usize;
        }
        pages
    }

    /// Allocate a single page
    fn get_small(&self, start: usize) -> Result<usize> {
        let pt2 = self.pt2(start);

        for _ in 0..CAS_RETRIES {
            for newstart in Self::MAPPING.iterate(2, start) {
                let i2 = Self::MAPPING.idx(2, newstart);

                #[cfg(feature = "stop")]
                {
                    let pte2 = pt2.get(i2);
                    if pte2.page() || pte2.free() == 0 {
                        continue;
                    }
                    stop!();
                }

                if pt2.update(i2, |v| v.dec()).is_ok() {
                    return self.get_table(newstart);
                }
            }
        }
        error!("Nothing found {}", start / Self::MAPPING.span(2));
        Err(Error::Corruption)
    }

    /// Search free page table entry.
    fn get_table(&self, start: usize) -> Result<usize> {
        let i = Self::MAPPING.idx(1, start);
        let pt1 = self.pt1(start);

        for _ in 0..CAS_RETRIES {
            if let Ok(i) = pt1.set_first_zero(i) {
                return Ok(Self::MAPPING.page(1, start, i));
            }
            stop!();
        }
        error!("Nothing found {}", start / Self::MAPPING.span(2));
        Err(Error::Corruption)
    }

    fn put_small(&self, page: usize) -> Result<()> {
        stop!();

        let pt1 = self.pt1(page);
        let i1 = Self::MAPPING.idx(1, page);
        if pt1.toggle(i1, true).is_err() {
            error!("Invalid Addr l1 i{i1} p={page}");
            return Err(Error::Address);
        }

        stop!();

        let pt2 = self.pt2(page);
        let i2 = Self::MAPPING.idx(2, page);
        if let Err(pte2) = pt2.update(i2, |v| v.inc(Self::MAPPING.span(1))) {
            error!("Invalid Addr l1 i{i1} p={page} {pte2:?}");
            return Err(Error::Address);
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        let mut out = String::new();
        writeln!(out, "Dumping pt {}", start / Self::MAPPING.span(2)).unwrap();
        let pt2 = self.pt2(start);
        for i2 in 0..Self::MAPPING.len(2) {
            let start = Self::MAPPING.page(2, start, i2);
            if start > self.pages {
                return;
            }

            let pte2 = pt2.get(i2);
            let indent = (Self::MAPPING.levels() - 2) * 4;
            let addr = start * Page::SIZE;
            writeln!(out, "{:indent$}l2 i={i2} 0x{addr:x}: {pte2:?}", "").unwrap();
        }
        warn!("{out}");
    }
}

#[cfg(feature = "stop")]
#[cfg(all(test, feature = "std"))]
mod test {
    use std::sync::Arc;

    use log::warn;

    use super::CacheLower;
    use crate::lower::LowerAlloc;
    use crate::stop::{StopVec, Stopper};
    use crate::table::{Bitfield, Mapping};
    use crate::thread;
    use crate::util::{logging, Page};

    const T2N: usize = 512;
    const MAPPING: Mapping<2> = CacheLower::<T2N>::MAPPING;

    fn count(pt: &Bitfield) -> usize {
        let mut pages = 0;
        for i in 0..Bitfield::LEN {
            pages += !pt.get(i) as usize;
        }
        pages
    }

    #[test]
    fn alloc_normal() {
        logging();

        let orders = [
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1, 1, 0, 0],
            vec![1, 1, 0, 0, 0, 1, 1],
            vec![1, 0, 1, 0, 0],
            vec![1, 1, 0, 0],
        ];

        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(CacheLower::<T2N>::new(2, &mut buffer));
            lower.clear();
            lower.get(0, 0, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let page = l.get(t, 0, 0).unwrap();
                drop(key);
                assert!(page != 0);
            });

            assert_eq!(lower.pt2(0).get(0).free(), MAPPING.span(1) - 3);
            assert_eq!(count(lower.pt1(0)), MAPPING.span(1) - 3);
        }
    }

    #[test]
    fn alloc_first() {
        logging();

        let orders = [
            vec![0, 0, 1, 1],
            vec![0, 1, 1, 0, 0],
            vec![0, 1, 0, 1, 1],
            vec![1, 1, 0, 0],
        ];

        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(CacheLower::<T2N>::new(2, &mut buffer));
            lower.clear();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(t, 0, 0).unwrap();
            });

            let pte2 = lower.pt2(0).get(0);
            assert_eq!(pte2.free(), MAPPING.span(1) - 2);
            assert_eq!(count(lower.pt1(0)), MAPPING.span(1) - 2);
        }
    }

    #[test]
    fn alloc_last() {
        logging();

        let orders = [
            vec![0, 0, 1, 1, 1],
            vec![0, 1, 1, 0, 1, 1, 0],
            vec![1, 0, 0, 1, 0],
            vec![1, 1, 0, 0, 0],
        ];

        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(CacheLower::<T2N>::new(2, &mut buffer));
            lower.clear();

            for _ in 0..MAPPING.span(1) - 1 {
                lower.get(0, 0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(t, 0, 0).unwrap();
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), 0);
            assert_eq!(pt2.get(1).free(), MAPPING.span(1) - 1);
            assert_eq!(count(lower.pt1(MAPPING.span(1))), MAPPING.span(1) - 1);
        }
    }

    #[test]
    fn free_normal() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1, 1], // first 0, then 1
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 1, 0, 0],
        ];

        let mut pages = [0; 2];
        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(CacheLower::<T2N>::new(2, &mut buffer));
            lower.clear();

            pages[0] = lower.get(0, 0, 0).unwrap();
            pages[1] = lower.get(0, 0, 0).unwrap();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, {
                let pages = pages.clone();
                move |t| {
                    let _stopper = Stopper::init(stop, t as _);

                    l.put(pages[t as usize], 0).unwrap();
                }
            });

            assert_eq!(lower.pt2(0).get(0).free(), MAPPING.span(1));
        }
    }

    #[test]
    fn free_last() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 1, 1, 0, 0, 1, 1, 0],
        ];

        let mut pages = [0; MAPPING.span(1)];
        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(CacheLower::<T2N>::new(2, &mut buffer));
            lower.clear();

            for page in &mut pages {
                *page = lower.get(0, 0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                let _stopper = Stopper::init(stop, t as _);

                l.put(pages[t as usize], 0).unwrap();
            });

            let pt2 = lower.pt2(0);
            assert_eq!(pt2.get(0).free(), 2);
            assert_eq!(count(lower.pt1(0)), 2);
        }
    }

    #[test]
    fn realloc_last() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1], // free then alloc
            vec![1, 1, 0, 0, 0], // alloc last then free last
            vec![0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0],
            vec![1, 0, 1, 0, 0],
            vec![0, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 1],
            vec![1, 0, 0, 0, 1],
        ];

        let mut pages = [0; MAPPING.span(1)];
        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(CacheLower::<T2N>::new(2, &mut buffer));
            lower.clear();

            for page in &mut pages[..MAPPING.span(1) - 1] {
                *page = lower.get(0, 0, 0).unwrap();
            }
            let stop = StopVec::new(2, order);

            let handle = std::thread::spawn({
                let stop = Arc::clone(&stop);
                let lower = lower.clone();
                move || {
                    let _stopper = Stopper::init(stop, 1);

                    lower.get(1, 0, 0).unwrap();
                }
            });

            {
                let _stopper = Stopper::init(stop, 0);

                lower.put(pages[0], 0).unwrap();
            }

            handle.join().unwrap();

            let pt2 = lower.pt2(0);
            if pt2.get(0).free() == 1 {
                assert_eq!(count(lower.pt1(0)), 1);
            } else {
                // Table entry skipped
                assert_eq!(pt2.get(0).free(), 2);
                assert_eq!(count(lower.pt1(0)), 2);
                assert_eq!(pt2.get(1).free(), MAPPING.span(1) - 1);
                assert_eq!(count(lower.pt1(MAPPING.span(1))), MAPPING.span(1) - 1);
            }
        }
    }
}
