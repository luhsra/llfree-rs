use core::fmt;
use core::ops::Range;
use std::cell::UnsafeCell;
use std::ops::Deref;

use crate::alloc::{Result, Size};
use crate::entry::Entry3;
use crate::table::Table;
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

pub mod dynamic;
pub mod fixed;

/// Level 2 page allocator.
pub trait LowerAlloc: Default + fmt::Debug + Deref<Target = [Local]> {
    fn new(cores: usize, memory: &mut [Page]) -> Self;

    fn pages(&self) -> usize;
    fn memory(&self) -> Range<*const Page>;

    fn clear(&self);
    fn recover(&self, start: usize, deep: bool) -> Result<(usize, Size)>;

    fn get(&self, core: usize, huge: bool, start: usize) -> Result<usize>;
    fn put(&self, page: usize) -> Result<bool>;

    fn set_giant(&self, page: usize);
    fn clear_giant(&self, page: usize);

    /// Debug function, returning the number of free pages and performing internal checks.
    fn dbg_allocated_pages(&self) -> usize;
}

/// Per core data.
/// # Safety
/// This should only be accessed from the corresponding (virtual) CPU core!
#[repr(align(64))]
pub struct Local(UnsafeCell<Inner>);
#[repr(align(64))]
struct Inner {
    start: [usize; 2],
    pte: [Entry3; 2],
    frees: [usize; 4],
    frees_i: usize,
}

unsafe impl Send for Local {}
unsafe impl Sync for Local {}

impl Local {
    fn new() -> Self {
        Self(UnsafeCell::new(Inner {
            start: [usize::MAX, usize::MAX],
            pte: [
                Entry3::new().with_idx(Entry3::IDX_MAX),
                Entry3::new().with_idx(Entry3::IDX_MAX),
            ],
            frees_i: 0,
            frees: [usize::MAX; 4],
        }))
    }
    fn p(&self) -> &mut Inner {
        unsafe { &mut *self.0.get() }
    }
    pub fn start(&self, huge: bool) -> &mut usize {
        &mut self.p().start[huge as usize]
    }
    pub fn pte(&self, huge: bool) -> &mut Entry3 {
        &mut self.p().pte[huge as usize]
    }
    pub fn frees_push(&self, page: usize) {
        self.p().frees_i = (self.p().frees_i + 1) % self.p().frees.len();
        self.p().frees[self.p().frees_i] = page;
    }
    pub fn frees_related(&self, page: usize) -> bool {
        let n = page / Table::span(2);
        self.p().frees.iter().all(|p| p / Table::span(2) == n)
    }
}

#[cfg(all(test, feature = "stop"))]
mod test {
    use std::sync::Arc;

    use log::warn;

    use crate::lower::LowerAlloc;
    use crate::{
        stop::{StopRand, Stopper},
        table::Table,
        thread,
        util::{logging, Page},
    };

    type Lower = super::fixed::FixedLower;

    #[test]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 12;
        let mut buffer = vec![Page::new(); 2 * THREADS * Table::span(2)];

        for _ in 0..32 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(Lower::new(THREADS, &mut buffer));
            lower.clear();
            assert_eq!(lower.dbg_allocated_pages(), 0);

            let stop = StopRand::new(THREADS, seed);
            let l = lower.clone();
            thread::parallel(THREADS, move |t| {
                let _stopper = Stopper::init(stop, t);

                let mut pages = [0; 4];
                for p in &mut pages {
                    *p = l.get(t, false, 0).unwrap();
                }
                pages.reverse();
                for p in pages {
                    l.put(p).unwrap();
                }
            });

            assert_eq!(lower.dbg_allocated_pages(), 0);
        }
    }

    #[test]
    fn rand_realloc_last() {
        logging();

        const THREADS: usize = 12;
        let mut pages = [0; Table::LEN];
        let mut buffer = vec![Page::new(); 2 * THREADS * Table::span(2)];

        for _ in 0..32 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(Lower::new(THREADS, &mut buffer));
            lower.clear();
            assert_eq!(lower.dbg_allocated_pages(), 0);

            for page in &mut pages[..Table::LEN - 3] {
                *page = lower.get(0, false, 0).unwrap();
            }

            let stop = StopRand::new(THREADS, seed);
            let l = lower.clone();
            thread::parallel(THREADS, move |t| {
                let _stopper = Stopper::init(stop, t);

                if t < THREADS / 2 {
                    l.put(pages[t]).unwrap();
                } else {
                    l.get(t, false, 0).unwrap();
                }
            });

            assert_eq!(lower.dbg_allocated_pages(), Table::LEN - 3);
        }
    }
}
