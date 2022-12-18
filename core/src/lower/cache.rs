use core::fmt::{self, Write};
use core::mem::size_of;
use core::ops::Range;

use crossbeam_utils::atomic::AtomicCell;
use log::{error, info, warn};

use alloc::boxed::Box;
use alloc::slice;
use alloc::string::String;

use crate::entry::{Entry2, Entry2Pair};
use crate::table::{AtomicArray, Bitfield, Mapping};
use crate::upper::{Init, CAS_RETRIES};
use crate::util::{align_up, spin_wait, Page};
use crate::{Error, Result};

use super::LowerAlloc;

type Table1 = Bitfield<8>;

/// Subtree allocator which is able to allocate order 0..11 pages.
///
/// Here the bitfields are 512bit large -> strong focus on huge pages.
///
/// The generic parameter `T2N` configures the level 2 table size.
/// It has to be a multiple of 2!
///
/// ## Memory Layout
/// **persistent:**
/// ```text
/// NVRAM: [ Pages | Entry1s | Entry2s | Meta ]
/// ```
/// **volatile:**
/// ```text
/// RAM: [ Pages ], Entry1s and Entry2s are allocated elswhere
/// ```
pub struct Cache<const T2N: usize> {
    area: &'static mut [Page],
    l1: Box<[Table1]>,
    l2: Box<[[AtomicCell<Entry2>; T2N]]>,
    persistent: bool,
}

impl<const T2N: usize> Default for Cache<T2N> {
    fn default() -> Self {
        Self {
            area: &mut [],
            l1: Default::default(),
            l2: Default::default(),
            persistent: Default::default(),
        }
    }
}

impl<const T2N: usize> fmt::Debug for Cache<T2N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cache")
            .field("area", &self.area.as_ptr_range())
            .field("l1", &self.l1)
            .field("l2", &self.l2)
            .field("persistent", &self.persistent)
            .finish()
    }
}

unsafe impl<const T2N: usize> Send for Cache<T2N> {}
unsafe impl<const T2N: usize> Sync for Cache<T2N> {}

impl<const T2N: usize> LowerAlloc for Cache<T2N>
where
    [(); T2N / 2]:,
{
    const N: usize = Self::MAPPING.span(2);
    const MAX_ORDER: usize = Self::MAPPING.order(1) + 1;
    const HUGE_ORDER: usize = Self::MAPPING.order(1);

    fn new(_cores: usize, area: &mut [Page], init: Init, free_all: bool) -> Self {
        assert!(T2N < (1 << (u16::BITS as usize - Self::MAPPING.order(1))));

        // FIXME: Lifetime hack!
        let area = unsafe { slice::from_raw_parts_mut(area.as_mut_ptr(), area.len()) };
        let n1 = Self::MAPPING.num_pts(1, area.len());
        let n2 = Self::MAPPING.num_pts(2, area.len());
        let alloc = if init != Init::Volatile {
            // Reserve memory within the managed NVM for the l1 and l2 tables
            // These tables are stored at the end of the NVM
            let s1 = n1 * size_of::<Table1>();
            let s1 = align_up(s1, size_of::<[Entry2; T2N]>()); // correct alignment
            let s2 = n1 * size_of::<[Entry2; T2N]>();
            // Num of pages occupied by the tables
            let pages = (s1 + s2).div_ceil(Page::SIZE);

            assert!(pages < area.len());
            let (area, tables) = area.split_at_mut(area.len() - pages);

            let tables: *mut u8 = tables.as_mut_ptr().cast();

            // Start of the l1 table array
            let l1 = unsafe { Box::from_raw(slice::from_raw_parts_mut(tables.cast(), n1)) };

            let mut offset = n1 * size_of::<Table1>();
            offset = align_up(offset, size_of::<[Entry2; T2N]>()); // correct alignment

            // Start of the l2 table array
            let l2 =
                unsafe { Box::from_raw(slice::from_raw_parts_mut(tables.add(offset).cast(), n2)) };

            Self {
                area,
                l1,
                l2,
                persistent: init != Init::Volatile,
            }
        } else {
            // Allocate l1 and l2 tables in volatile memory
            Self {
                area,
                l1: unsafe { Box::new_uninit_slice(n1).assume_init() },
                l2: unsafe { Box::new_uninit_slice(n2).assume_init() },
                persistent: init != Init::Volatile,
            }
        };
        // Skip for manual recovery using `Self::recover`
        if init != Init::Recover {
            if free_all {
                alloc.free_all();
            } else {
                alloc.reserve_all();
            }
        }
        alloc
    }

    fn pages(&self) -> usize {
        self.area.len()
    }

    fn memory(&self) -> Range<*const Page> {
        self.area.as_ptr_range()
    }

    fn recover(&self, start: usize, deep: bool) -> Result<usize> {
        let mut pages = 0;

        let pt = self.table2(start);
        for i in 0..Self::MAPPING.len(2) {
            let start = Self::MAPPING.page(2, start, i);
            if start > self.pages() {
                pt[i].store(Entry2::new());
                continue;
            }

            let entry = pt[i].load();
            if deep && entry.free() > 0 {
                // Deep recovery updates the counter
                let p = self.table1(start).count_zeros();

                if entry.free() != p {
                    warn!("Invalid L2 start=0x{start:x} i{i}: {} != {p}", entry.free());
                    pt[i].store(entry.with_free(p));
                }
                pages += p;
            } else {
                pages += entry.free();
            }
        }

        Ok(pages)
    }

    fn get(&self, start: usize, order: usize) -> Result<usize> {
        debug_assert!(order <= Self::MAX_ORDER);

        if order == Self::MAX_ORDER {
            self.get_max(start)
        } else if order == Self::HUGE_ORDER {
            self.get_huge(start)
        } else if order < Self::HUGE_ORDER {
            self.get_small(start, order)
        } else {
            error!("Invalid order");
            Err(Error::Corruption)
        }
    }

    /// Free single page and returns if the page was huge
    fn put(&self, page: usize, order: usize) -> Result<()> {
        debug_assert!(order <= Self::MAX_ORDER);
        debug_assert!(page < self.pages());
        stop!();

        if order == Self::MAX_ORDER {
            self.put_max(page)
        } else if order == Self::HUGE_ORDER {
            let i2 = Self::MAPPING.idx(2, page);
            let table2 = self.table2(page);

            if let Err(old) = table2[i2]
                .compare_exchange(Entry2::new_page(), Entry2::new_free(Self::MAPPING.span(1)))
            {
                error!("Addr p={page:x} o={order} {old:?}");
                Err(Error::Address)
            } else {
                Ok(())
            }
        } else if order < Self::HUGE_ORDER {
            let i2 = Self::MAPPING.idx(2, page);
            let table2 = self.table2(page);

            let old = table2[i2].load();
            if old.page() {
                self.partial_put_huge(old, page, order)
            } else if old.free() <= Self::MAPPING.span(1) - (1 << order) {
                self.put_small(page, order)
            } else {
                error!("Addr p={page:x} o={order} {old:?}");
                Err(Error::Address)
            }
        } else {
            error!("Invalid order!");
            Err(Error::Corruption)
        }
    }

    fn is_free(&self, page: usize, order: usize) -> bool {
        debug_assert!(page % (1 << order) == 0);
        if order > Self::MAX_ORDER || page + (1 << order) > self.pages() {
            return false;
        }

        if order > Self::MAPPING.order(1) {
            // multiple hugepages
            let i2 = Self::MAPPING.idx(2, page);
            self.table2_pair(page)[i2 / 2]
                .load()
                .all(|e| !e.page() && e.free() == Self::MAPPING.span(1))
        } else {
            let table2 = self.table2(page);
            let i2 = Self::MAPPING.idx(2, page);
            let entry2 = table2[i2].load();
            let num_pages = 1 << order;

            if entry2.free() < num_pages {
                return false;
            }
            if entry2.free() == Self::MAPPING.span(1) {
                return true;
            }

            if num_pages > u64::BITS as usize {
                // larger than 64 pages (often allocated as huge page)
                let pt = self.table1(page);
                let start = page / Table1::ENTRY_BITS;
                let end = (page + num_pages) / Table1::ENTRY_BITS;
                (start..end).all(|i| pt.get_entry(i) == 0)
            } else {
                // small allocations
                let pt = self.table1(page);
                let entry = pt.get_entry(page / Table1::ENTRY_BITS);
                let mask =
                    (u64::MAX >> (u64::BITS as usize - num_pages)) << (page % Table1::ENTRY_BITS);
                (entry & mask) == 0
            }
        }
    }

    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let start = i * Self::MAPPING.span(2);
            let table2 = self.table2(start);
            for i2 in Self::MAPPING.range(2, start..self.pages()) {
                let start = Self::MAPPING.page(2, start, i2);
                let entry2 = table2[i2].load();

                pages -= if entry2.page() {
                    0
                } else {
                    let table1 = self.table1(start);
                    let mut free = 0;
                    for i1 in Self::MAPPING.range(1, start..self.pages()) {
                        free += !table1.get(i1) as usize;
                    }
                    assert_eq!(free, entry2.free(), "{entry2:?}: {table1:?}");
                    free
                }
            }
        }
        pages
    }

    fn dbg_for_each_huge_page<F: FnMut(usize)>(&self, mut f: F) {
        for i3 in 0..(self.pages() / Self::MAPPING.span(2)) {
            let start = i3 * Self::MAPPING.span(2);
            let table2 = self.table2(start);
            for i2 in Self::MAPPING.range(2, start..self.pages()) {
                f(table2[i2].load().free());
            }
        }
    }

    fn size_per_gib() -> usize {
        let pages = 1usize << (30 - Page::SIZE_BITS);
        let s1 = pages.div_ceil(8); // 1 bit per page
        let s2 = size_of::<Entry2>() * pages.div_ceil(Table1::LEN);
        s1 + s2
    }
}

impl<const T2N: usize> Cache<T2N>
where
    [(); T2N / 2]:,
{
    const MAPPING: Mapping<2> = Mapping([Table1::ORDER, T2N.ilog2() as _]);

    /// Returns the l1 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | padding | PT1s | PT2s | Meta ]
    /// ```
    fn table1(&self, page: usize) -> &Table1 {
        let i = page / Self::MAPPING.span(1);
        debug_assert!(i < Self::MAPPING.num_pts(1, self.pages()));
        &self.l1[i]
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | padding | PT1s | PT2s | Meta ]
    /// ```
    fn table2(&self, page: usize) -> &[AtomicCell<Entry2>; T2N] {
        let i = page / Self::MAPPING.span(2);
        debug_assert!(i < Self::MAPPING.num_pts(2, self.pages()));
        &self.l2[i]
    }

    /// Returns the l2 page table with pair entries that can be updated at once.
    fn table2_pair(&self, page: usize) -> &[AtomicCell<Entry2Pair>; T2N / 2] {
        let table2 = self.table2(page);
        unsafe { &*(table2.as_ptr().cast()) }
    }

    fn free_all(&self) {
        // Init table2
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let table2 = self.table2(i * Self::MAPPING.span(2));
            if i + 1 < Self::MAPPING.num_pts(2, self.pages()) {
                unsafe { table2.atomic_fill(Entry2::new_free(Self::MAPPING.span(1))) };
            } else {
                for j in 0..Self::MAPPING.len(2) {
                    let page = i * Self::MAPPING.span(2) + j * Self::MAPPING.span(1);
                    let max = Self::MAPPING.span(1).min(self.pages().saturating_sub(page));
                    table2[j].store(Entry2::new_free(max));
                }
            }
        }
        // Init table1
        for i in 0..Self::MAPPING.num_pts(1, self.pages()) {
            let table1 = self.table1(i * Self::MAPPING.span(1));

            if i + 1 < Self::MAPPING.num_pts(1, self.pages()) {
                table1.fill(false);
            } else {
                let end = self.pages() - i * Self::MAPPING.span(1);
                table1.set(0..end, false);
                table1.set(end..Self::MAPPING.len(1), true);
            }
        }
    }

    fn reserve_all(&self) {
        // Init table2
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let table2 = self.table2(i * Self::MAPPING.span(2));
            let end = (i + 1) * Self::MAPPING.span(2);
            if self.pages() >= end {
                // Table is fully included in the memory range
                // -> Allocated as full pages
                unsafe { table2.atomic_fill(Entry2::new_page()) };
            } else {
                // Table is only partially included in the memory range
                for j in 0..Self::MAPPING.len(2) {
                    let end = i * Self::MAPPING.span(2) + (j + 1) * Self::MAPPING.span(1);
                    if self.pages() >= end {
                        table2[j].store(Entry2::new_page());
                    } else {
                        // Remainder is allocated as small pages
                        table2[j].store(Entry2::new_free(0));
                    }
                }
            }
        }
        // Init table1
        for i in 0..Self::MAPPING.num_pts(1, self.pages()) {
            let table1 = self.table1(i * Self::MAPPING.span(1));
            let end = (i + 1) * Self::MAPPING.span(1);
            if self.pages() >= end {
                // Table is fully included in the memory range
                table1.fill(false);
            } else {
                // Table is only partially included in the memory range
                table1.fill(true);
            }
        }
    }

    /// Allocate pages up to order 8
    fn get_small(&self, start: usize, order: usize) -> Result<usize> {
        debug_assert!(order < Self::MAPPING.order(1));

        let table2 = self.table2(start);

        for _ in 0..CAS_RETRIES {
            // Begin iteration by start idx
            let off = Self::MAPPING.idx(2, start);
            for j2 in 0..Self::MAPPING.len(2) {
                let i2 = (j2 + off) % Self::MAPPING.len(2);

                let newstart = if j2 == 0 {
                    start // Don't round the start pfn
                } else {
                    Self::MAPPING.page(2, start, i2)
                };

                #[cfg(feature = "stop")]
                {
                    let entry2 = table2[i2].load();
                    if entry2.page() || entry2.free() < (1 << order) {
                        continue;
                    }
                    stop!();
                }

                if table2[i2].fetch_update(|v| v.dec(1 << order)).is_ok() {
                    let i1 = Self::MAPPING.idx(1, newstart);
                    let table1 = self.table1(newstart);

                    if let Ok(offset) = table1.set_first_zeros(i1, order) {
                        return Ok(Self::MAPPING.page(1, newstart, offset));
                    } else {
                        // Revert conter
                        if let Err(_) =
                            table2[i2].fetch_update(|v| v.inc(Self::MAPPING.span(1), 1 << order))
                        {
                            error!("Undo failed");
                            return Err(Error::Corruption);
                        }
                    }
                }
            }
        }
        info!("Nothing found o={order}");
        Err(Error::Memory)
    }

    /// Allocate huge page
    fn get_huge(&self, start: usize) -> Result<usize> {
        let table2 = self.table2(start);
        let start_i = Self::MAPPING.idx(2, start);
        for _ in 0..CAS_RETRIES {
            for i2 in 0..Self::MAPPING.len(2) {
                let i2 = (start_i + i2) % Self::MAPPING.len(2);
                if table2[i2]
                    .fetch_update(|v| v.mark_huge(Self::MAPPING.span(1)))
                    .is_ok()
                {
                    return Ok(Self::MAPPING.page(2, start, i2));
                }
            }
            core::hint::spin_loop();
        }
        info!("Nothing found o=9");
        Err(Error::Memory)
    }

    /// Allocate multiple huge pages
    fn get_max(&self, start: usize) -> Result<usize> {
        let table2_pair = self.table2_pair(start);
        let start_i = Self::MAPPING.idx(2, start) / 2;
        for _ in 0..CAS_RETRIES {
            for i2 in 0..Self::MAPPING.len(2) / 2 {
                let i2 = (start_i + i2) % (Self::MAPPING.len(2) / 2);
                if table2_pair[i2]
                    .fetch_update(|v| v.map(|v| v.mark_huge(Self::MAPPING.span(1))))
                    .is_ok()
                {
                    return Ok(Self::MAPPING.page(2, start, i2 * 2));
                }
            }
            core::hint::spin_loop();
        }
        info!("Nothing found o=10");
        Err(Error::Memory)
    }

    fn put_small(&self, page: usize, order: usize) -> Result<()> {
        debug_assert!(order < Self::HUGE_ORDER);

        stop!();

        let table1 = self.table1(page);
        let i1 = Self::MAPPING.idx(1, page);
        if table1.toggle(i1, order, true).is_err() {
            error!("L1 put failed i{i1} p={page}");
            return Err(Error::Address);
        }

        stop!();

        let table2 = self.table2(page);
        let i2 = Self::MAPPING.idx(2, page);
        if let Err(entry2) = table2[i2].fetch_update(|v| v.inc(Self::MAPPING.span(1), 1 << order)) {
            error!("Inc failed i{i1} p={page} {entry2:?}");
            return Err(Error::Corruption);
        }

        Ok(())
    }

    pub fn put_max(&self, page: usize) -> Result<()> {
        let table2_pair = self.table2_pair(page);
        let i2 = Self::MAPPING.idx(2, page) / 2;

        if let Err(old) = table2_pair[i2].compare_exchange(
            Entry2Pair(Entry2::new_page(), Entry2::new_page()),
            Entry2Pair(
                Entry2::new_free(Self::MAPPING.span(1)),
                Entry2::new_free(Self::MAPPING.span(1)),
            ),
        ) {
            error!("Addr {page:x} o={} {old:?} i={i2}", Self::MAX_ORDER);
            Err(Error::Address)
        } else {
            Ok(())
        }
    }

    fn partial_put_huge(&self, old: Entry2, page: usize, order: usize) -> Result<()> {
        warn!("partial free of huge page {page:x} o={order}");
        let i2 = Self::MAPPING.idx(2, page);
        let table2 = self.table2(page);
        let table1 = self.table1(page);

        // Try filling the whole bitfield
        if table1.fill_cas(true) {
            if table2[i2].compare_exchange(old, Entry2::new()).is_err() {
                error!("Failed partial clear");
                return Err(Error::Corruption);
            }
        }
        // Wait for parallel partial_put_huge to finish
        else if !spin_wait(CAS_RETRIES, || !table2[i2].load().page()) {
            error!("Exceeding retries");
            return Err(Error::Corruption);
        }

        self.put_small(page, order)
    }

    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        let mut out = String::new();
        writeln!(out, "Dumping pt {}", start / Self::MAPPING.span(2)).unwrap();
        let table2 = self.table2(start);
        for i2 in 0..Self::MAPPING.len(2) {
            let start = Self::MAPPING.page(2, start, i2);
            if start > self.pages() {
                break;
            }

            let entry2 = table2[i2].load();
            let indent = (Self::MAPPING.levels() - 2) * 4;
            let table1 = self.table1(start);
            writeln!(out, "{:indent$}l2 i={i2}: {entry2:?}\t{table1:?}", "").unwrap();
        }
        warn!("{out}");
    }
}

impl<const T2N: usize> Drop for Cache<T2N> {
    fn drop(&mut self) {
        if self.persistent {
            // The chunks are part of the allocators managed memory
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
    use log::warn;

    use super::{Cache, Table1};
    use crate::lower::LowerAlloc;
    use crate::stop::{StopVec, Stopper};
    use crate::table::Mapping;
    use crate::thread;
    use crate::upper::Init;
    use crate::util::{logging, Page, WyRand};

    type Allocator = Cache<64>;
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));
            lower.get(0, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(0..2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let page = l.get(0, 0).unwrap();
                drop(key);
                assert!(page != 0);
            });

            assert_eq!(lower.table2(0)[0].load().free(), MAPPING.span(1) - 3);
            assert_eq!(count(lower.table1(0)), MAPPING.span(1) - 3);
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(0, 0).unwrap();
            });

            let entry2 = lower.table2(0)[0].load();
            assert_eq!(entry2.free(), MAPPING.span(1) - 2);
            assert_eq!(count(lower.table1(0)), MAPPING.span(1) - 2);
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            for _ in 0..MAPPING.span(1) - 1 {
                lower.get(0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(0, 0).unwrap();
            });

            let table2 = lower.table2(0);
            assert_eq!(table2[0].load().free(), 0);
            assert_eq!(table2[1].load().free(), MAPPING.span(1) - 1);
            assert_eq!(count(lower.table1(MAPPING.span(1))), MAPPING.span(1) - 1);
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            pages[0] = lower.get(0, 0).unwrap();
            pages[1] = lower.get(0, 0).unwrap();

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, {
                let pages = pages.clone();
                move |t| {
                    let _stopper = Stopper::init(stop, t as _);

                    l.put(pages[t as usize], 0).unwrap();
                }
            });

            assert_eq!(lower.table2(0)[0].load().free(), MAPPING.span(1));
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            for page in &mut pages {
                *page = lower.get(0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, move |t| {
                let _stopper = Stopper::init(stop, t as _);

                l.put(pages[t as usize], 0).unwrap();
            });

            let table2 = lower.table2(0);
            assert_eq!(table2[0].load().free(), 2);
            assert_eq!(count(lower.table1(0)), 2);
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            for page in &mut pages[..MAPPING.span(1) - 1] {
                *page = lower.get(0, 0).unwrap();
            }
            let stop = StopVec::new(2, order);

            std::thread::scope(|s| {
                let st = stop.clone();
                s.spawn(|| {
                    let _stopper = Stopper::init(st, 1);

                    lower.get(0, 0).unwrap();
                });

                let _stopper = Stopper::init(stop, 0);

                lower.put(pages[0], 0).unwrap();
            });

            let table2 = lower.table2(0);
            if table2[0].load().free() == 1 {
                assert_eq!(count(lower.table1(0)), 1);
            } else {
                // Table entry skipped
                assert_eq!(table2[0].load().free(), 2);
                assert_eq!(count(lower.table1(0)), 2);
                assert_eq!(table2[1].load().free(), MAPPING.span(1) - 1);
                assert_eq!(count(lower.table1(MAPPING.span(1))), MAPPING.span(1) - 1);
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));
            lower.get(0, 0).unwrap();

            let stop = StopVec::new(2, order);

            let l = lower.clone();
            thread::parallel(0..2, move |t| {
                thread::pin(t);
                let key = Stopper::init(stop, t as _);

                let order = t + 1; // order 1 and 2
                let page = l.get(0, order).unwrap();
                drop(key);
                assert!(page != 0);
            });

            let allocated = 1 + 2 + 4;
            assert_eq!(
                lower.table2(0)[0].load().free(),
                MAPPING.span(1) - allocated
            );
            assert_eq!(count(lower.table1(0)), MAPPING.span(1) - allocated);
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
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            pages[0] = lower.get(0, 1).unwrap();
            pages[1] = lower.get(0, 2).unwrap();

            assert_eq!(lower.table2(0)[0].load().free(), MAPPING.span(1) - 2 - 4);

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, {
                let pages = pages.clone();
                move |t| {
                    let _stopper = Stopper::init(stop, t as _);

                    l.put(pages[t as usize], t + 1).unwrap();
                }
            });

            assert_eq!(lower.table2(0)[0].load().free(), MAPPING.span(1));
        }
    }

    #[test]
    fn different_orders() {
        logging();

        const MAX_ORDER: usize = Allocator::MAX_ORDER;
        let mut buffer = vec![Page::new(); MAPPING.span(2)];

        thread::pin(0);
        let lower = Arc::new(Allocator::new(1, &mut buffer, Init::Overwrite, true));

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
        let num_max_pages = buffer.len() / (1 << MAX_ORDER);

        let lower = Arc::new(Allocator::new(1, &mut buffer, Init::Volatile, false));

        assert_eq!(lower.dbg_allocated_pages(), MAPPING.span(2) - 1);

        for i in 0..num_max_pages {
            lower.put(i * (1 << MAX_ORDER), MAX_ORDER).unwrap();
        }

        assert_eq!(lower.dbg_allocated_pages(), (1 << MAX_ORDER) - 1);
    }

    #[test]
    fn partial_put_huge() {
        logging();

        let mut buffer = vec![Page::new(); MAPPING.span(2) - 1];

        let lower = Arc::new(Allocator::new(1, &mut buffer, Init::Volatile, false));

        assert_eq!(lower.dbg_allocated_pages(), MAPPING.span(2) - 1);

        lower.put(0, 0).unwrap();

        assert_eq!(lower.dbg_allocated_pages(), MAPPING.span(2) - 2);

        lower.dump(0);
    }
}
