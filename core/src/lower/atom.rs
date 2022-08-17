use core::fmt::Write;
use core::ops::Range;

use alloc::boxed::Box;
use alloc::slice;
use alloc::string::String;
use log::{error, info, warn};

use crate::entry::{SEntry2, SEntry2T2, SEntry2T4, SEntry2T8, SEntry2Tuple};
use crate::table::{ATable, Bitfield, Mapping};
use crate::upper::CAS_RETRIES;
use crate::util::{align_up, div_ceil, log2, Page};
use crate::{Error, Result};

use super::LowerAlloc;

type Table1 = Bitfield<2>;

/// Subtree allocator which is able to allocate order 0..11 pages.
///
/// The generic parameter `T2N` configures the level 2 table size.
/// It has to be a multiple of 8!
///
/// ## Memory Layout
/// **persistent:**
/// ```text
/// NVRAM: [ Pages | PT1s + padding | PT2s | Meta ]
/// ```
/// **volatile:**
/// ```text
/// RAM: [ Pages ], PT1s and PT2s are allocated elswhere
/// ```
#[derive(Default, Debug)]
pub struct Atom<const T2N: usize> {
    pub begin: usize,
    pub pages: usize,
    l1: Box<[Table1]>,
    l2: Box<[ATable<SEntry2, T2N>]>,
    persistent: bool,
}

impl<const T2N: usize> LowerAlloc for Atom<T2N>
where
    [(); T2N / 1]:,
    [(); T2N / 2]:,
    [(); T2N / 4]:,
    [(); T2N / 8]:,
{
    const MAPPING: Mapping<2> = Mapping([Table1::ORDER, ATable::<SEntry2, T2N>::ORDER]);
    const MAX_ORDER: usize = Self::MAPPING.order(1) + 3;

    fn new(_cores: usize, memory: &mut [Page], persistent: bool) -> Self {
        let n1 = Self::MAPPING.num_pts(1, memory.len());
        let n2 = Self::MAPPING.num_pts(2, memory.len());
        if persistent {
            let s1 = n1 * Table1::SIZE;
            let s1 = align_up(s1, ATable::<SEntry2, T2N>::SIZE); // correct alignment
            let s2 = n1 * ATable::<SEntry2, T2N>::SIZE;

            let pages = div_ceil(s1 + s2, Page::SIZE);

            assert!(pages < memory.len());

            let mut offset = memory.as_ptr() as usize + (memory.len() - pages) * Page::SIZE;

            let l1 = unsafe { Box::from_raw(slice::from_raw_parts_mut(offset as *mut _, n1)) };

            offset += n1 * Table1::SIZE;
            offset = align_up(offset, ATable::<SEntry2, T2N>::SIZE); // correct alignment

            let l2 = unsafe { Box::from_raw(slice::from_raw_parts_mut(offset as *mut _, n2)) };

            Self {
                begin: memory.as_ptr() as usize,
                pages: memory.len() - pages,
                // level 1 and 2 tables are stored at the end of the NVM
                l1,
                l2,
                persistent,
            }
        } else {
            Self {
                begin: memory.as_ptr() as usize,
                pages: memory.len(),
                // Allocate in volatile memory
                l1: unsafe { Box::new_zeroed_slice(n1).assume_init() },
                l2: unsafe { Box::new_zeroed_slice(n2).assume_init() },
                persistent,
            }
        }
    }

    fn pages(&self) -> usize {
        self.pages
    }

    fn memory(&self) -> Range<*const Page> {
        self.begin as *const Page..(self.begin + self.pages * Page::SIZE) as *const Page
    }

    fn free_all(&self) {
        // Init pt2
        for i in 0..Self::MAPPING.num_pts(2, self.pages) {
            let pt2 = self.pt2(i * Self::MAPPING.span(2));
            if i + 1 < Self::MAPPING.num_pts(2, self.pages) {
                pt2.fill(SEntry2::new().with_free(Self::MAPPING.span(1)));
            } else {
                for j in 0..Self::MAPPING.len(2) {
                    let page = i * Self::MAPPING.span(2) + j * Self::MAPPING.span(1);
                    let max = Self::MAPPING.span(1).min(self.pages.saturating_sub(page));
                    pt2.set(j, SEntry2::new().with_free(max));
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

    fn reserve_all(&self) {
        // Init pt2
        for i in 0..Self::MAPPING.num_pts(2, self.pages) {
            let pt2 = self.pt2(i * Self::MAPPING.span(2));
            let end = (i + 1) * Self::MAPPING.span(2);
            if self.pages >= end {
                // Table is fully included in the memory range
                // -> Allocated as full pages
                pt2.fill(SEntry2::new_page());
            } else {
                // Table is only partially included in the memory range
                for j in 0..Self::MAPPING.len(2) {
                    let end = i * Self::MAPPING.span(2) + (j + 1) * Self::MAPPING.span(1);
                    if self.pages >= end {
                        pt2.set(j, SEntry2::new_page());
                    } else {
                        // Remainder is allocated as small pages
                        pt2.set(j, SEntry2::new_free(0));
                    }
                }
            }
        }
        // Init pt1
        for i in 0..Self::MAPPING.num_pts(1, self.pages) {
            let pt1 = self.pt1(i * Self::MAPPING.span(1));
            let end = (i + 1) * Self::MAPPING.span(1);
            if self.pages >= end {
                // Table is fully included in the memory range
                pt1.fill(false);
            } else {
                // Table is only partially included in the memory range
                pt1.fill(true);
            }
        }
    }

    fn recover(&self, start: usize, deep: bool) -> Result<usize> {
        let mut pages = 0;

        let pt = self.pt2(start);
        for i in 0..Self::MAPPING.len(2) {
            let start = Self::MAPPING.page(2, start, i);
            if start > self.pages {
                pt.set(i, SEntry2::new());
                continue;
            }

            let pte = pt.get(i);
            if deep && pte.free() > 0 {
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

        Ok(pages)
    }

    fn get(&self, start: usize, order: usize) -> Result<usize> {
        debug_assert!(order <= Self::MAX_ORDER);

        if order < Self::MAPPING.order(1) {
            self.get_small(start, order)
        } else if order == Self::MAPPING.order(1) {
            self.get_huge_n::<SEntry2>(start)
        } else if order == Self::MAPPING.order(1) + 1 {
            self.get_huge_n::<SEntry2T2>(start)
        } else if order == Self::MAPPING.order(1) + 2 {
            self.get_huge_n::<SEntry2T4>(start)
        } else if order == Self::MAPPING.order(1) + 3 {
            self.get_huge_n::<SEntry2T8>(start)
        } else {
            Err(Error::Address)
        }
    }

    /// Free single page and returns if the page was huge
    fn put(&self, page: usize, order: usize) -> Result<()> {
        debug_assert!(order <= Self::MAX_ORDER);
        debug_assert!(page < self.pages);
        stop!();

        if order >= Self::MAPPING.order(1) {
            return if order == Self::MAPPING.order(1) {
                self.put_huge_n::<SEntry2>(page)
            } else if order == Self::MAPPING.order(1) + 1 {
                self.put_huge_n::<SEntry2T2>(page)
            } else if order == Self::MAPPING.order(1) + 2 {
                self.put_huge_n::<SEntry2T4>(page)
            } else if order == Self::MAPPING.order(1) + 3 {
                self.put_huge_n::<SEntry2T8>(page)
            } else {
                Err(Error::Address)
            };
        } else {
            let pt2 = self.pt2(page);
            let i2 = Self::MAPPING.idx(2, page);
            let old = pt2.get(i2);
            if old.page() {
                // TODO: handle partial free of huge page
                // Order 6 should work -> set other half to 1, clear page flag
                warn!("ignore partial free of huge page {page:x} o={order}");
                Ok(())
            } else if old.free() <= Self::MAPPING.span(1) - (1 << order) {
                self.put_small(page, order)
            } else {
                error!("Addr p={page:x} o={order} {old:?}");
                Err(Error::Address)
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
                    0
                } else {
                    let pt1 = self.pt1(start);
                    let mut free = 0;
                    for i1 in Self::MAPPING.range(1, start..self.pages) {
                        free += !pt1.get(i1) as usize;
                    }
                    assert_eq!(free, pte2.free(), "{pte2:?}: {pt1:?}");
                    free
                }
            }
        }
        pages
    }

    fn dbg_for_each_huge_page<F: FnMut(usize)>(&self, mut f: F) {
        for i2 in 0..(self.pages / Self::MAPPING.span(2)) {
            let start = i2 * Self::MAPPING.span(2);
            let pt2 = self.pt2_t::<SEntry2T4>(start);
            for i1 in Self::MAPPING.range(2, start..self.pages).step_by(4) {
                let SEntry2T4([e0, e1, e2, e3]) = pt2.get(i1 / 4);
                f(e0.free() + e1.free() + e2.free() + e3.free());
            }
        }
    }
}

impl<const T2N: usize> Atom<T2N>
where
    [(); T2N / 1]:,
    [(); T2N / 2]:,
    [(); T2N / 4]:,
    [(); T2N / 8]:,
{
    /// Returns the l1 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | padding | PT1s | PT2s | Meta ]
    /// ```
    fn pt1(&self, page: usize) -> &Table1 {
        let i = page / Self::MAPPING.span(1);
        debug_assert!(i < Self::MAPPING.num_pts(1, self.pages));
        &self.l1[i]
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | padding | PT1s | PT2s | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &ATable<SEntry2, T2N> {
        let i = page / Self::MAPPING.span(2);
        debug_assert!(i < Self::MAPPING.num_pts(2, self.pages));
        &self.l2[i]
    }

    /// Returns the l2 page table with pair entries that can be updated at once.
    fn pt2_t<T: SEntry2Tuple>(&self, page: usize) -> &ATable<T, { T2N / T::N }>
    where
        [(); T2N / T::N]:,
    {
        let pt2 = self.pt2(page);
        unsafe { &*((pt2 as *const ATable<SEntry2, T2N>) as *const ATable<T, { T2N / T::N }>) }
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
    fn get_small(&self, start: usize, order: usize) -> Result<usize> {
        let pt2 = self.pt2(start);

        for _ in 0..CAS_RETRIES {
            for newstart in Self::MAPPING.iterate(2, start) {
                let i2 = Self::MAPPING.idx(2, newstart);

                #[cfg(feature = "stop")]
                {
                    let pte2 = pt2.get(i2);
                    if pte2.page() || pte2.free() < 1 << order {
                        continue;
                    }
                    stop!();
                }

                if pt2.update(i2, |v| v.dec(1 << order)).is_ok() {
                    match self.get_table(newstart, order) {
                        // Revert conter
                        Err(Error::Memory) => {
                            if let Err(e) =
                                pt2.update(i2, |v| v.inc(Self::MAPPING.span(1), 1 << order))
                            {
                                error!("Rollback failed {e:?}");
                                return Err(Error::Corruption);
                            }
                        }
                        ret => return ret,
                    }
                }
            }
        }
        info!("Nothing found o={order}");
        Err(Error::Memory)
    }

    /// Search free page table entry.
    fn get_table(&self, start: usize, order: usize) -> Result<usize> {
        let i = Self::MAPPING.idx(1, start);
        let pt1 = self.pt1(start);

        for _ in 0..CAS_RETRIES {
            if let Ok(i) = pt1.set_first_zeros(i, order) {
                return Ok(Self::MAPPING.page(1, start, i));
            }
            stop!();
        }
        info!("Nothing found o={order}");
        Err(Error::Memory)
    }

    /// Allocate multiple huge pages
    fn get_huge_n<T: SEntry2Tuple>(&self, start: usize) -> Result<usize>
    where
        [(); T2N / T::N]:,
    {
        let order = Table1::ORDER + log2(T::N);
        let pt2 = self.pt2_t::<T>(start);
        for _ in 0..CAS_RETRIES {
            for page in Self::MAPPING.iterate(2, start).step_by(T::N) {
                let i2 = Self::MAPPING.idx(2, page) / T::N;
                if pt2
                    .update(i2, |v| v.all(|v| v.mark_huge(Self::MAPPING.span(1))))
                    .is_ok()
                {
                    return Ok(Self::MAPPING.page(2, start, i2 * T::N));
                }
            }
        }
        info!("Nothing found o={order}");
        Err(Error::Memory)
    }

    fn put_small(&self, page: usize, order: usize) -> Result<()> {
        stop!();

        let pt1 = self.pt1(page);
        let i1 = Self::MAPPING.idx(1, page);
        if pt1.toggle(i1, order, true).is_err() {
            error!("L1 put failed p={page} o={order}");
            return Err(Error::Address);
        }

        stop!();

        let pt2 = self.pt2(page);
        let i2 = Self::MAPPING.idx(2, page);
        if let Err(pte2) = pt2.update(i2, |v| v.inc(Self::MAPPING.span(1), 1 << order)) {
            error!("Inc failed i{i1} p={page} {pte2:?}");
            return Err(Error::Corruption);
        }

        Ok(())
    }

    pub fn put_huge_n<T: SEntry2Tuple>(&self, page: usize) -> Result<()>
    where
        [(); T2N / T::N]:,
    {
        let order = Table1::ORDER + log2(T::N);
        let pt2 = self.pt2_t::<T>(page);
        let i2 = Self::MAPPING.idx(2, page) / T::N;
        info!("Put o={order} i={i2}");

        // TODO: set pt1s to 1?

        if let Err(old) = pt2.cas(
            i2,
            T::new(SEntry2::new_page()),
            T::new(SEntry2::new_free(Self::MAPPING.span(1))),
        ) {
            error!("Addr {page:x} o={order} {old:?} i={i2}");
            Err(Error::Address)
        } else {
            Ok(())
        }
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
            let pt1 = self.pt1(start);
            writeln!(out, "{:indent$}l2 i={i2}: {pte2:?}\t{pt1:?}", "").unwrap();
        }
        warn!("{out}");
    }
}

impl<const T2N: usize> Drop for Atom<T2N> {
    fn drop(&mut self) {
        if self.persistent {
            Box::leak(core::mem::take(&mut self.l1));
            Box::leak(core::mem::take(&mut self.l2));
        }
    }
}

#[cfg(feature = "stop")]
#[cfg(all(test, feature = "std"))]
mod test {
    use std::sync::Arc;

    use alloc::vec::Vec;
    use log::{debug, info, warn};

    use super::{Atom, Table1};
    use crate::lower::LowerAlloc;
    use crate::stop::{StopVec, Stopper};
    use crate::table::Mapping;
    use crate::thread;
    use crate::util::{logging, Page, WyRand};

    type Allocator = Atom<128>;
    const MAPPING: Mapping<2> = Allocator::MAPPING;

    fn count(pt: &Table1) -> usize {
        let mut pages = 0;
        for i in 0..Table1::LEN {
            pages += !pt.get(i) as usize;
        }
        pages
    }

    #[test]
    fn alloc_normal() {
        logging();

        println!("{:?}", log::STATIC_MAX_LEVEL);
        println!("{:?}", log::max_level());
        debug!("debug");
        info!("info");
        warn!("warn");

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
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();
            lower.get(0, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let page = l.get(0, 0).unwrap();
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(0, 0).unwrap();
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();

            for _ in 0..MAPPING.span(1) - 1 {
                lower.get(0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(0, 0).unwrap();
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
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();

            pages[0] = lower.get(0, 0).unwrap();
            pages[1] = lower.get(0, 0).unwrap();

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
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();

            for page in &mut pages {
                *page = lower.get(0, 0).unwrap();
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
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();

            for page in &mut pages[..MAPPING.span(1) - 1] {
                *page = lower.get(0, 0).unwrap();
            }
            let stop = StopVec::new(2, order);

            let handle = std::thread::spawn({
                let stop = Arc::clone(&stop);
                let lower = lower.clone();
                move || {
                    let _stopper = Stopper::init(stop, 1);

                    lower.get(0, 0).unwrap();
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

    #[test]
    fn alloc_normal_large() {
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();
            lower.get(0, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let order = t + 1; // order 1 and 2
                let page = l.get(0, order).unwrap();
                drop(key);
                assert!(page != 0);
            });

            let allocated = 1 + 2 + 4;
            assert_eq!(lower.pt2(0).get(0).free(), MAPPING.span(1) - allocated);
            assert_eq!(count(lower.pt1(0)), MAPPING.span(1) - allocated);
        }
    }

    #[test]
    fn free_normal_large() {
        logging();

        let orders = [
            vec![0, 0, 0, 1, 1, 1], // first 0, then 1
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 1, 0, 0],
        ];

        let mut pages = [0; 2];
        let mut buffer = vec![Page::new(); 4 * MAPPING.span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, true));
            lower.free_all();

            pages[0] = lower.get(0, 1).unwrap();
            pages[1] = lower.get(0, 2).unwrap();

            assert_eq!(lower.pt2(0).get(0).free(), MAPPING.span(1) - 2 - 4);

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(2, {
                let pages = pages.clone();
                move |t| {
                    let _stopper = Stopper::init(stop, t as _);

                    l.put(pages[t as usize], t + 1).unwrap();
                }
            });

            assert_eq!(lower.pt2(0).get(0).free(), MAPPING.span(1));
        }
    }

    #[test]
    fn different_orders() {
        logging();

        const MAX_ORDER: usize = Allocator::MAX_ORDER;
        let mut buffer = vec![Page::new(); MAPPING.span(2)];

        thread::pin(0);
        let lower = Arc::new(Allocator::new(1, &mut buffer, true));
        lower.free_all();

        assert_eq!(lower.dbg_allocated_pages(), 0);

        let mut rng = WyRand::new(42);

        let mut num_pages = 0;
        let mut pages = Vec::new();
        for order in 0..=MAX_ORDER {
            for _ in 0..1usize << (MAX_ORDER - order) {
                pages.push((order, 0));
                num_pages += 1 << order;
            }
        }
        rng.shuffle(&mut pages);
        warn!("allocate {num_pages} pages up to order {MAX_ORDER}");

        for (order, page) in &mut pages {
            *page = lower.get(0, *order).unwrap();
        }

        lower.dump(0);
        assert_eq!(lower.dbg_allocated_pages(), num_pages);

        for (order, page) in &pages {
            lower.put(*page, *order).unwrap();
        }

        assert_eq!(lower.dbg_allocated_pages(), 0);
    }

    #[test]
    fn init_reserved() {
        logging();

        const MAX_ORDER: usize = Allocator::MAX_ORDER;

        let mut buffer = vec![Page::new(); MAPPING.span(2) - 1];

        let lower = Arc::new(Allocator::new(1, &mut buffer, false));
        lower.reserve_all();

        assert_eq!(lower.dbg_allocated_pages(), MAPPING.span(2) - 1);

        for (i, _) in buffer.chunks_exact(1 << MAX_ORDER).enumerate() {
            lower.put(i * (1 << MAX_ORDER), MAX_ORDER).unwrap();
        }

        assert_eq!(lower.dbg_allocated_pages(), (1 << MAX_ORDER) - 1);
    }
}