use core::fmt;
use core::ops::Range;
use std::fmt::Write;
use std::mem::size_of;
use std::sync::atomic::{self, AtomicU8, Ordering};

use log::{error, warn};

use crate::alloc::{Error, Result, Size, CAS_RETRIES};
use crate::entry::Entry2;
use crate::table::Table;
use crate::util::{align_up, Page};

use super::LowerAlloc;

/// Level 2 page allocator.
/// ```text
/// NVRAM: [ Pages | PT1s | PT2s | Meta ]
/// ```
#[derive(Default, Debug)]
pub struct PackedLower {
    pub begin: usize,
    pub pages: usize,
}

impl LowerAlloc for PackedLower {
    fn new(_cores: usize, memory: &mut [Page]) -> Self {
        let n2_pages =
            (Table::num_pts(2, memory.len()) * Bitfield::SIZE + Page::SIZE - 1) / Page::SIZE;
        Self {
            begin: memory.as_ptr() as usize,
            // level 1 and 2 tables are stored at the end of the NVM
            pages: memory.len() - n2_pages - Table::num_pts(1, memory.len()),
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
            let pt1 = self.pt1(i * Table::span(1));

            if i + 1 < Table::num_pts(1, self.pages) {
                pt1.fill(false);
            } else {
                for j in 0..Bitfield::LEN {
                    let page = i * Table::span(1) + j;
                    if page < self.pages {
                        pt1.set(j, false);
                    } else {
                        pt1.set(j, true);
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
                let p = self.recover_l1(start)?;
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

    fn get(&self, _core: usize, huge: bool, start: usize) -> Result<usize> {
        if !huge {
            return self.get_small(start);
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
        error!("Nothing found {}", start / Table::span(2));
        Err(Error::Corruption)
    }

    /// Free single page and returns if the page was huge
    fn put(&self, page: usize) -> Result<bool> {
        debug_assert!(page < self.pages);
        stop!();

        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);
        // try free huge
        if let Err(old) = pt2.cas(
            i2,
            Entry2::new().with_page(true),
            Entry2::new_table(Table::LEN, 0),
        ) {
            if !old.giant() && old.free() < Table::LEN {
                self.put_small(page).map(|_| false)
            } else {
                error!("Addr {page:x} {old:?}");
                Err(Error::Address)
            }
        } else {
            Ok(true)
        }
    }

    fn set_giant(&self, page: usize) {
        self.pt2(page).set(0, Entry2::new().with_giant(true));
    }
    fn clear_giant(&self, page: usize) {
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
                    let pt1 = self.pt1(start);
                    let mut child_pages = 0;
                    for i1 in Table::range(1, start..self.pages) {
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

impl PackedLower {
    /// Returns the l1 page table that contains the `page`.
    fn pt1(&self, page: usize) -> &Bitfield {
        let mut offset = self.begin + self.pages * Page::SIZE;
        let i = page >> Table::LEN_BITS;
        debug_assert!(i < Table::num_pts(1, self.pages));
        offset += i * Bitfield::SIZE;
        unsafe { &*(offset as *const Bitfield) }
    }

    /// Returns the l2 page table that contains the `page`.
    /// ```text
    /// NVRAM: [ Pages | PT1s | PT2s | Meta ]
    /// ```
    fn pt2(&self, page: usize) -> &Table<Entry2> {
        let mut offset = self.begin + self.pages * Page::SIZE;
        offset += Table::num_pts(1, self.pages) * Bitfield::SIZE;
        offset = align_up(offset, Page::SIZE);
        let i = page >> (Table::LEN_BITS * 2);
        debug_assert!(i < Table::num_pts(2, self.pages));
        offset += i * Page::SIZE;
        unsafe { &*(offset as *mut Table<Entry2>) }
    }

    fn recover_l1(&self, start: usize) -> Result<usize> {
        let pt = self.pt1(start);
        let mut pages = 0;
        for i in Table::range(1, start..self.pages) {
            pages += pt.get(i) as usize;
        }
        Ok(pages)
    }

    /// Allocate a single page
    fn get_small(&self, start: usize) -> Result<usize> {
        let pt2 = self.pt2(start);

        for _ in 0..CAS_RETRIES {
            for newstart in Table::iterate(2, start) {
                let i2 = Table::idx(2, newstart);

                #[cfg(feature = "stop")]
                {
                    let pte2 = pt2.get(i2);
                    if pte2.page() || pte2.free() == 0 {
                        continue;
                    }
                    stop!();
                }

                if pt2.update(i2, |v| v.dec(0)).is_ok() {
                    return self.get_table(newstart);
                }
            }
        }
        error!("Nothing found {}", start / Table::span(2));
        Err(Error::Corruption)
    }

    /// Search free page table entry.
    fn get_table(&self, start: usize) -> Result<usize> {
        let i = Table::idx(1, start);
        let pt1 = self.pt1(start);

        for _ in 0..CAS_RETRIES {
            if let Ok(i) = pt1.search_set(i) {
                return Ok(Table::page(1, start, i));
            }
            stop!();
        }
        error!("Nothing found {}", start / Table::span(2));
        Err(Error::Corruption)
    }

    fn put_small(&self, page: usize) -> Result<()> {
        stop!();

        let pt1 = self.pt1(page);
        let i1 = Table::idx(1, page);
        if pt1.toggle(i1, true).is_err() {
            error!("Invalid Addr l1 i{i1} p={page}");
            return Err(Error::Address);
        }

        stop!();

        let pt2 = self.pt2(page);
        let i2 = Table::idx(2, page);
        if let Err(pte2) = pt2.update(i2, |pte| pte.inc()) {
            error!("Invalid Addr l1 i{i1} p={page} {pte2:?}");
            return Err(Error::Address);
        }

        Ok(())
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

/// Bitfield replacing the level one-page table.
#[repr(align(64))]
struct Bitfield {
    data: [AtomicU8; Table::LEN / Self::ENTRY_BITS],
}

const _: () = assert!(size_of::<Bitfield>() == Bitfield::SIZE);

impl Default for Bitfield {
    fn default() -> Self {
        const D: AtomicU8 = AtomicU8::new(0);
        Self {
            data: [D; Table::LEN / Self::ENTRY_BITS],
        }
    }
}

impl fmt::Debug for Bitfield {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitfield(")?;
        for (i, d) in self.data.iter().enumerate() {
            if i % 4 == 0 && i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{:02x}", d.load(Ordering::Relaxed))?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Bitfield {
    const ENTRY_BITS: usize = 8;
    const SIZE: usize = Table::LEN / 8;
    const LEN: usize = Table::LEN;

    fn set(&self, i: usize, v: bool) {
        let di = i / Self::ENTRY_BITS;
        let bit = 1 << (i % Self::ENTRY_BITS);
        let _ = self.data[di].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |e| {
            Some(if v { e | bit } else { e & !bit })
        });
    }

    fn get(&self, i: usize) -> bool {
        let di = i / Self::ENTRY_BITS;
        let bit = 1 << (i % Self::ENTRY_BITS);
        self.data[di].load(Ordering::SeqCst) & bit != 0
    }

    fn toggle(&self, i: usize, expected: bool) -> core::result::Result<bool, bool> {
        let di = i / Self::ENTRY_BITS;
        let bit = 1 << (i % Self::ENTRY_BITS);
        match self.data[di].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |e| {
            ((e & bit != 0) == expected).then(|| if !expected { e | bit } else { e & !bit })
        }) {
            Ok(e) => Ok(e & bit != 0),
            Err(e) => Err(e & bit != 0),
        }
    }

    /// Set the first 0 bit to 1 returning its bit index.
    fn search_set(&self, i: usize) -> core::result::Result<usize, ()> {
        for j in 0..self.data.len() {
            let i = (j + i) % self.data.len();

            #[cfg(feature = "stop")]
            {
                // Skip full entries for the tests
                if self.data[i].load(Ordering::SeqCst) == u8::MAX {
                    continue;
                }
                stop!();
            }

            if let Ok(e) = self.data[i].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |e| {
                let off = e.trailing_ones() as usize;
                (off < Self::ENTRY_BITS).then(|| e | (1 << off))
            }) {
                return Ok(i * Self::ENTRY_BITS + e.trailing_ones() as usize);
            }
        }
        Err(())
    }

    fn fill(&self, v: bool) {
        let v = if v { u8::MAX } else { 0 };
        // cast to raw memory to let the compiler use vector instructions
        #[allow(clippy::cast_ref_to_mut)]
        let mem = unsafe { &mut *(&self.data as *const _ as *mut [u8; Self::SIZE]) };
        mem.fill(v);
        // memory ordering has to be enforced with a memory barrier
        atomic::fence(Ordering::SeqCst);
    }
}

#[cfg(feature = "stop")]
#[cfg(test)]
mod test {
    use std::sync::Arc;

    use log::warn;

    use super::{Bitfield, PackedLower};
    use crate::lower::LowerAlloc;
    use crate::stop::{StopVec, Stopper};
    use crate::table::Table;
    use crate::thread;
    use crate::util::{logging, Page};

    fn count(pt: &Bitfield) -> usize {
        warn!("{pt:?}");
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

        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(PackedLower::new(2, &mut buffer));
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
            assert_eq!(count(lower.pt1(0)), Table::LEN - 3);
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

        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(PackedLower::new(2, &mut buffer));
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
            assert_eq!(count(lower.pt1(0)), Table::LEN - 2);
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

        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {order:?}");
            let lower = Arc::new(PackedLower::new(2, &mut buffer));
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
            assert_eq!(count(lower.pt1(Table::LEN)), Table::LEN - 1);
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
        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(PackedLower::new(2, &mut buffer));
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
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0, 1, 1, 0, 0, 1, 1, 0],
        ];

        let mut pages = [0; Table::LEN];
        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(PackedLower::new(2, &mut buffer));
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

        let mut pages = [0; Table::LEN];
        let mut buffer = vec![Page::new(); 4 * Table::span(2)];

        for order in orders {
            warn!("order: {:?}", order);
            let lower = Arc::new(PackedLower::new(2, &mut buffer));
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
                assert_eq!(count(lower.pt1(0)), 1);
            } else {
                // Table entry skipped
                assert_eq!(pt2.get(0).free(), 2);
                assert_eq!(count(lower.pt1(0)), 2);
                assert_eq!(pt2.get(1).free(), Table::LEN - 1);
                assert_eq!(count(lower.pt1(Table::LEN)), Table::LEN - 1);
            }
        }
    }
}
