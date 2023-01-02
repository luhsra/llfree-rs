use core::fmt::{self, Write};
use core::mem::size_of;
use core::ops::Range;

use crossbeam_utils::atomic::AtomicCell;
use log::{error, info, warn};

use alloc::boxed::Box;
use alloc::slice;
use alloc::string::String;

use crate::entry::{Entry2, Entry2Pair};
use crate::table::AtomicArray;
use crate::upper::{Init, CAS_RETRIES};
use crate::util::{align_down, align_up, spin_wait, Page};
use crate::{Error, Result};

use super::LowerAlloc;

type Bitfield = crate::table::Bitfield<8>;

/// Tree allocator which is able to allocate order 0..11 pages.
///
/// Here the bitfields are 512 bit large -> strong focus on huge pages.
/// Upon that is a table for each tree, with an entry per bitfield.
///
/// The parameter `HP` configures the number of table entries (huge pages per tree).
/// It has to be a multiple of 2!
///
/// ## Memory Layout
/// **persistent:**
/// ```text
/// NVRAM: [ Pages | Bitfields | Tables | Zone ]
/// ```
/// **volatile:**
/// ```text
/// RAM: [ Pages ], Bitfields and Tables are allocated elswhere
/// ```
pub struct Cache<const HP: usize> {
    area: &'static mut [Page],
    bitfields: Box<[Bitfield]>,
    tables: Box<[[AtomicCell<Entry2>; HP]]>,
    persistent: bool,
}

impl<const HP: usize> Default for Cache<HP> {
    fn default() -> Self {
        Self {
            area: &mut [],
            bitfields: Default::default(),
            tables: Default::default(),
            persistent: Default::default(),
        }
    }
}

impl<const HP: usize> fmt::Debug for Cache<HP> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cache")
            .field("area", &self.area.as_ptr_range())
            .field("bitfields", &self.bitfields)
            .field("tables", &self.tables)
            .field("persistent", &self.persistent)
            .finish()
    }
}

unsafe impl<const HP: usize> Send for Cache<HP> {}
unsafe impl<const HP: usize> Sync for Cache<HP> {}

impl<const HP: usize> LowerAlloc for Cache<HP>
where
    [(); HP / 2]:,
{
    const N: usize = HP * Bitfield::LEN;
    const HUGE_ORDER: usize = Bitfield::ORDER;
    const MAX_ORDER: usize = Self::HUGE_ORDER + 1;

    fn new(_cores: usize, area: &mut [Page], init: Init, free_all: bool) -> Self {
        debug_assert!(HP < (1 << (u16::BITS as usize - Self::HUGE_ORDER)));

        // FIXME: Lifetime hack!
        let area = unsafe { slice::from_raw_parts_mut(area.as_mut_ptr(), area.len()) };
        let num_bitfields = area.len().div_ceil(Bitfield::LEN);
        let num_tables = area.len().div_ceil(Self::N);
        let alloc = if init != Init::Volatile {
            // Reserve memory within the managed NVM for the l1 and l2 tables
            // These tables are stored at the end of the NVM
            let size_bitfields = num_bitfields * size_of::<Bitfield>();
            let size_bitfields = align_up(size_bitfields, size_of::<[Entry2; HP]>()); // correct alignment
            let size_tables = num_bitfields * size_of::<[Entry2; HP]>();
            // Num of pages occupied by the tables
            let metadata_pages = (size_bitfields + size_tables).div_ceil(Page::SIZE);

            debug_assert!(metadata_pages < area.len());
            let (area, tables) = area.split_at_mut(area.len() - metadata_pages);

            let tables: *mut u8 = tables.as_mut_ptr().cast();

            // Start of the l1 table array
            let bitfields =
                unsafe { Box::from_raw(slice::from_raw_parts_mut(tables.cast(), num_bitfields)) };

            let mut offset = num_bitfields * size_of::<Bitfield>();
            offset = align_up(offset, size_of::<[Entry2; HP]>()); // correct alignment

            // Start of the l2 table array
            let tables = unsafe {
                Box::from_raw(slice::from_raw_parts_mut(
                    tables.add(offset).cast(),
                    num_tables,
                ))
            };

            Self {
                area,
                bitfields,
                tables,
                persistent: init != Init::Volatile,
            }
        } else {
            // Allocate l1 and l2 tables in volatile memory
            Self {
                area,
                bitfields: unsafe { Box::new_uninit_slice(num_bitfields).assume_init() },
                tables: unsafe { Box::new_uninit_slice(num_tables).assume_init() },
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

        let table = &self.tables[start / Self::N];
        for (i, a_entry) in table.iter().enumerate() {
            let start = align_down(start, Self::N) + i * Bitfield::LEN;
            let i = start / Bitfield::LEN;

            if start > self.pages() {
                a_entry.store(Entry2::new());
                continue;
            }

            let entry = a_entry.load();
            let free = entry.free();
            if deep {
                // Deep recovery updates the counter
                if entry.page() {
                    // Check that underlying bitfield is empty
                    let p = self.bitfields[i].count_zeros();
                    if p != Bitfield::LEN {
                        warn!("Invalid L2 start=0x{start:x} i{i}: h != {p}");
                        self.bitfields[i].fill(false);
                    }
                } else if free == Bitfield::LEN {
                    // Skip entirely free entries
                    // This is possible because the counter is decremented first
                    // for allocations and afterwards for frees
                    pages += Bitfield::LEN;
                } else {
                    // Check if partially filled bitfield has the same free count
                    let p = self.bitfields[i].count_zeros();
                    if free != p {
                        warn!("Invalid L2 start=0x{start:x} i{i}: {free} != {p}");
                        a_entry.store(entry.with_free(p));
                    }
                    pages += p;
                }
            } else {
                pages += free;
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
            let i = (page / Bitfield::LEN) % HP;
            let table = &self.tables[page / Self::N];

            if let Err(old) =
                table[i].compare_exchange(Entry2::new_page(), Entry2::new_free(Bitfield::LEN))
            {
                error!("Addr p={page:x} o={order} {old:?}");
                Err(Error::Address)
            } else {
                Ok(())
            }
        } else if order < Self::HUGE_ORDER {
            let i = (page / Bitfield::LEN) % HP;
            let table = &self.tables[page / Self::N];

            let old = table[i].load();
            if old.page() {
                self.partial_put_huge(old, page, order)
            } else if old.free() <= Bitfield::LEN - (1 << order) {
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

        if order > Bitfield::ORDER {
            // multiple hugepages
            let i = (page / Bitfield::LEN) % HP;
            self.table_pair(page)[i / 2]
                .load()
                .all(|e| e.free() == Bitfield::LEN)
        } else {
            let table = &self.tables[page / Self::N];
            let i = (page / Bitfield::LEN) % HP;
            let entry = table[i].load();
            let num_pages = 1 << order;

            if entry.free() < num_pages {
                return false;
            }
            if entry.free() == Bitfield::LEN {
                return true;
            }

            if num_pages > u64::BITS as usize {
                // larger than 64 pages
                let bitfield = &self.bitfields[page / Bitfield::LEN];
                let start = page / Bitfield::ENTRY_BITS;
                let end = (page + num_pages) / Bitfield::ENTRY_BITS;
                (start..end).all(|i| bitfield.get_entry(i) == 0)
            } else {
                // small allocations
                let bitfield = &self.bitfields[page / Bitfield::LEN];
                let entry = bitfield.get_entry(page / Bitfield::ENTRY_BITS);
                let mask =
                    (u64::MAX >> (u64::BITS as usize - num_pages)) << (page % Bitfield::ENTRY_BITS);
                (entry & mask) == 0
            }
        }
    }

    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for table in &*self.tables {
            for entry in table {
                pages -= entry.load().free();
            }
        }
        pages
    }

    fn dbg_for_each_huge_page<F: FnMut(usize)>(&self, mut f: F) {
        for table in &*self.tables {
            for entry in table {
                f(entry.load().free())
            }
        }
    }

    fn size_per_gib() -> usize {
        let pages = 1usize << (30 - Page::SIZE_BITS);
        let size_bitfields = pages.div_ceil(8); // 1 bit per page
        let size_tables = size_of::<Entry2>() * pages.div_ceil(Bitfield::LEN);
        size_bitfields + size_tables
    }
}

impl<const HP: usize> Cache<HP>
where
    [(); HP / 2]:,
{
    /// Returns the table with pair entries that can be updated at once.
    fn table_pair(&self, page: usize) -> &[AtomicCell<Entry2Pair>; HP / 2] {
        let table = &self.tables[page / Self::N];
        unsafe { &*table.as_ptr().cast() }
    }

    fn free_all(&self) {
        // Init tables
        let (last, tables) = self.tables.split_last().unwrap();
        // Table is fully included in the memory range
        for table in tables {
            unsafe { table.atomic_fill(Entry2::new_free(Bitfield::LEN)) };
        }
        // Table is only partially included in the memory range
        for (i, entry) in last.iter().enumerate() {
            let page = tables.len() * Self::N + i * Bitfield::LEN;
            let free = self.pages().saturating_sub(page).min(Bitfield::LEN);
            entry.store(Entry2::new_free(free));
        }

        // Init bitfields
        let last_i = self.pages() / Bitfield::LEN;
        let (included, mut remainder) = self.bitfields.split_at(last_i);
        // Bitfield is fully included in the memory range
        for bitfield in included {
            bitfield.fill(false);
        }
        // Bitfield might be only partially included in the memory range
        if let Some((last, excluded)) = remainder.split_first() {
            let end = self.pages() - included.len() * Bitfield::LEN;
            debug_assert!(end <= Bitfield::LEN);
            last.set(0..end, false);
            last.set(end..Bitfield::LEN, true);
            remainder = excluded;
        }
        // Not part of the final memory range
        for bitfield in remainder {
            bitfield.fill(true);
        }
    }

    fn reserve_all(&self) {
        // Init table
        let (last, tables) = self.tables.split_last().unwrap();
        // Table is fully included in the memory range
        for table in tables {
            unsafe { table.atomic_fill(Entry2::new_page()) };
        }
        // Table is only partially included in the memory range
        let last_i = (self.pages() / Bitfield::LEN) - tables.len() * HP;
        let (included, remainder) = last.split_at(last_i);
        for entry in included {
            entry.store(Entry2::new_page());
        }
        // Remainder is allocated as small pages
        for entry in remainder {
            entry.store(Entry2::new_free(0));
        }

        // Init bitfields
        let last_i = self.pages() / Bitfield::LEN;
        let (included, remainder) = self.bitfields.split_at(last_i);
        // Bitfield is fully included in the memory range
        for bitfield in included {
            bitfield.fill(false);
        }
        // Bitfield might be only partially included in the memory range
        for bitfield in remainder {
            bitfield.fill(true);
        }
    }

    /// Allocate pages up to order 8
    fn get_small(&self, start: usize, order: usize) -> Result<usize> {
        debug_assert!(order < Bitfield::ORDER);

        let table = &self.tables[start / Self::N];

        for _ in 0..CAS_RETRIES {
            // Begin iteration by start idx
            let off = (start / Bitfield::LEN) % HP;
            for j in 0..HP {
                let i = (j + off) % HP;

                let newstart = if j == 0 {
                    start // Don't round the start pfn
                } else {
                    align_down(start, Self::N) + i * Bitfield::LEN
                };

                #[cfg(feature = "stop")]
                {
                    let entry = table[i].load();
                    if entry.page() || entry.free() < (1 << order) {
                        continue;
                    }
                    stop!();
                }

                if table[i].fetch_update(|v| v.dec(1 << order)).is_ok() {
                    let bi = newstart % Bitfield::LEN;
                    let bitfield = &self.bitfields[newstart / Bitfield::LEN];

                    if let Ok(offset) = bitfield.set_first_zeros(bi, order) {
                        return Ok(align_down(newstart, Bitfield::LEN) + offset);
                    } else {
                        // Revert conter
                        if let Err(_) = table[i].fetch_update(|v| v.inc(Bitfield::LEN, 1 << order))
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
        let table = &self.tables[start / Self::N];
        let offset = (start / Bitfield::LEN) % HP;
        for _ in 0..CAS_RETRIES {
            for i in 0..HP {
                let i = (offset + i) % HP;
                if table[i]
                    .fetch_update(|v| v.mark_page(Bitfield::LEN))
                    .is_ok()
                {
                    return Ok(align_down(start, Self::N) + i * Bitfield::LEN);
                }
            }
            core::hint::spin_loop();
        }
        info!("Nothing found o=9");
        Err(Error::Memory)
    }

    /// Allocate multiple huge pages
    fn get_max(&self, start: usize) -> Result<usize> {
        let table_pair = self.table_pair(start);
        let offset = ((start / Bitfield::LEN) % HP) / 2;
        for _ in 0..CAS_RETRIES {
            for i in 0..HP / 2 {
                let i = (offset + i) % (HP / 2);
                if table_pair[i]
                    .fetch_update(|v| v.map(|v| v.mark_page(Bitfield::LEN)))
                    .is_ok()
                {
                    return Ok(align_down(start, Self::N) + 2 * i * Bitfield::LEN);
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

        let bitfield = &self.bitfields[page / Bitfield::LEN];
        let i = page % Bitfield::LEN;
        if bitfield.toggle(i, order, true).is_err() {
            error!("L1 put failed i{i} p={page}");
            return Err(Error::Address);
        }

        stop!();

        let table = &self.tables[page / Self::N];
        let i = (page / Bitfield::LEN) % HP;
        if let Err(entry) = table[i].fetch_update(|v| v.inc(Bitfield::LEN, 1 << order)) {
            error!("Inc failed i{i} p={page} {entry:?}");
            return Err(Error::Corruption);
        }

        Ok(())
    }

    pub fn put_max(&self, page: usize) -> Result<()> {
        let table_pair = self.table_pair(page);
        let i = ((page / Bitfield::LEN) % HP) / 2;

        if let Err(old) = table_pair[i].compare_exchange(
            Entry2Pair(Entry2::new_page(), Entry2::new_page()),
            Entry2Pair(
                Entry2::new_free(Bitfield::LEN),
                Entry2::new_free(Bitfield::LEN),
            ),
        ) {
            error!("Addr {page:x} o={} {old:?} i={i}", Self::MAX_ORDER);
            Err(Error::Address)
        } else {
            Ok(())
        }
    }

    fn partial_put_huge(&self, old: Entry2, page: usize, order: usize) -> Result<()> {
        warn!("partial free of huge page {page:x} o={order}");
        let i = (page / Bitfield::LEN) % HP;
        let table = &self.tables[page / Self::N];
        let bitfield = &self.bitfields[page / Bitfield::LEN];

        // Try filling the whole bitfield
        if bitfield.fill_cas(true) {
            if table[i].compare_exchange(old, Entry2::new()).is_err() {
                error!("Failed partial clear");
                return Err(Error::Corruption);
            }
        }
        // Wait for parallel partial_put_huge to finish
        else if !spin_wait(CAS_RETRIES, || !table[i].load().page()) {
            error!("Exceeding retries");
            return Err(Error::Corruption);
        }

        self.put_small(page, order)
    }

    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        let mut out = String::new();
        writeln!(out, "Dumping pt {}", start / Self::N).unwrap();
        let table = &self.tables[start / Self::N];
        for (i, entry) in table.iter().enumerate() {
            let start = align_down(start, Self::N) + i * Bitfield::LEN;
            if start > self.pages() {
                break;
            }

            let entry = entry.load();
            let indent = 4;
            let bitfield = &self.bitfields[start / Bitfield::LEN];
            writeln!(out, "{:indent$}l2 i={i}: {entry:?}\t{bitfield:?}", "").unwrap();
        }
        warn!("{out}");
    }
}

impl<const HP: usize> Drop for Cache<HP> {
    fn drop(&mut self) {
        if self.persistent {
            // The chunks are part of the allocators managed memory
            Box::leak(core::mem::take(&mut self.bitfields));
            Box::leak(core::mem::take(&mut self.tables));
        }
    }
}

#[cfg(feature = "stop")]
#[cfg(all(test, feature = "std"))]
mod test {
    use std::sync::Arc;

    use alloc::vec::Vec;
    use log::warn;

    use super::{Bitfield, Cache};
    use crate::lower::LowerAlloc;
    use crate::stop::{StopVec, Stopper};
    use crate::thread;
    use crate::upper::Init;
    use crate::util::{logging, Page, WyRand};

    type Allocator = Cache<64>;

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

        let mut buffer = vec![Page::new(); 4 * Allocator::N];

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

            assert_eq!(lower.tables[0][0].load().free(), Bitfield::LEN - 3);
            assert_eq!(count(&lower.bitfields[0]), Bitfield::LEN - 3);
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

        let mut buffer = vec![Page::new(); 4 * Allocator::N];

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

            let entry2 = lower.tables[0][0].load();
            assert_eq!(entry2.free(), Bitfield::LEN - 2);
            assert_eq!(count(&lower.bitfields[0]), Bitfield::LEN - 2);
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

        let mut buffer = vec![Page::new(); 4 * Allocator::N];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            for _ in 0..Bitfield::LEN - 1 {
                lower.get(0, 0).unwrap();
            }

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, move |t| {
                thread::pin(t);
                let _stopper = Stopper::init(stop, t as _);

                l.get(0, 0).unwrap();
            });

            let table = &lower.tables[0];
            assert_eq!(table[0].load().free(), 0);
            assert_eq!(table[1].load().free(), Bitfield::LEN - 1);
            assert_eq!(count(&lower.bitfields[1]), Bitfield::LEN - 1);
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
        let mut buffer = vec![Page::new(); 4 * Allocator::N];

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

            assert_eq!(lower.tables[0][0].load().free(), Bitfield::LEN);
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

        let mut pages = [0; Bitfield::LEN];
        let mut buffer = vec![Page::new(); 4 * Allocator::N];

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

            let table = &lower.tables[0];
            assert_eq!(table[0].load().free(), 2);
            assert_eq!(count(&lower.bitfields[0]), 2);
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

        let mut pages = [0; Bitfield::LEN];
        let mut buffer = vec![Page::new(); 4 * Allocator::N];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            for page in &mut pages[..Bitfield::LEN - 1] {
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

            let table = &lower.tables[0];
            if table[0].load().free() == 1 {
                assert_eq!(count(&lower.bitfields[0]), 1);
            } else {
                // Table entry skipped
                assert_eq!(table[0].load().free(), 2);
                assert_eq!(count(&lower.bitfields[0]), 2);
                assert_eq!(table[1].load().free(), Bitfield::LEN - 1);
                assert_eq!(count(&lower.bitfields[1]), Bitfield::LEN - 1);
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

        let mut buffer = vec![Page::new(); 4 * Allocator::N];

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
            assert_eq!(lower.tables[0][0].load().free(), Bitfield::LEN - allocated);
            assert_eq!(count(&lower.bitfields[0]), Bitfield::LEN - allocated);
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
        let mut buffer = vec![Page::new(); 4 * Allocator::N];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(Allocator::new(2, &mut buffer, Init::Overwrite, true));

            pages[0] = lower.get(0, 1).unwrap();
            pages[1] = lower.get(0, 2).unwrap();

            assert_eq!(lower.tables[0][0].load().free(), Bitfield::LEN - 2 - 4);

            let stop = StopVec::new(2, order);
            let l = lower.clone();
            thread::parallel(0..2, {
                let pages = pages.clone();
                move |t| {
                    let _stopper = Stopper::init(stop, t as _);

                    l.put(pages[t as usize], t + 1).unwrap();
                }
            });

            assert_eq!(lower.tables[0][0].load().free(), Bitfield::LEN);
        }
    }

    #[test]
    fn different_orders() {
        logging();

        const MAX_ORDER: usize = Allocator::MAX_ORDER;
        let mut buffer = vec![Page::new(); Allocator::N];

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

        let mut buffer = vec![Page::new(); Allocator::N - 1];
        let num_max_pages = buffer.len() / (1 << MAX_ORDER);

        let lower = Arc::new(Allocator::new(1, &mut buffer, Init::Volatile, false));

        assert_eq!(lower.dbg_allocated_pages(), Allocator::N - 1);

        for i in 0..num_max_pages {
            lower.put(i * (1 << MAX_ORDER), MAX_ORDER).unwrap();
        }

        assert_eq!(lower.dbg_allocated_pages(), (1 << MAX_ORDER) - 1);
    }

    #[test]
    fn partial_put_huge() {
        logging();

        let mut buffer = vec![Page::new(); Allocator::N - 1];

        let lower = Arc::new(Allocator::new(1, &mut buffer, Init::Volatile, false));

        assert_eq!(lower.dbg_allocated_pages(), Allocator::N - 1);

        lower.put(0, 0).unwrap();

        assert_eq!(lower.dbg_allocated_pages(), Allocator::N - 2);

        lower.dump(0);
    }
}
