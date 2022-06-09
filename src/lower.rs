use core::fmt;
use core::ops::Range;

use crate::alloc::{Result, Size};
use crate::Page;
use crate::table::Mapping;

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

// mod dense;
// pub use dense::DenseLower;
mod dynamic;
pub use dynamic::DynamicLower;
mod fixed;
pub use fixed::FixedLower;
mod packed;
pub use packed::PackedLower;

/// Level 2 page allocator.
pub trait LowerAlloc: Default + fmt::Debug {
    const MAPPING: Mapping<2>;

    /// Create a new lower allocator.
    fn new(cores: usize, memory: &mut [Page]) -> Self;

    fn pages(&self) -> usize;
    fn memory(&self) -> Range<*const Page>;

    /// Clear all metadata.
    fn clear(&self);
    /// Recover the level 2 page table at `start`.
    /// If deep, the level 1 pts are also traversed and false counters are corrected.
    fn recover(&self, start: usize, deep: bool) -> Result<(usize, Size)>;

    /// Try allocating a new `huge` page at the subtree at `start`.
    fn get(&self, core: usize, huge: bool, start: usize) -> Result<usize>;
    /// Try freeing a page. Returns if it was huge.
    fn put(&self, page: usize) -> Result<bool>;

    /// Mark the page as giant allocation.
    fn set_giant(&self, page: usize);
    /// Unmark the page, clearing any affected page tables.
    fn clear_giant(&self, page: usize);

    /// Debug function, returning the number of free pages and performing internal checks.
    fn dbg_allocated_pages(&self) -> usize;
}

#[cfg(all(test, feature = "stop"))]
mod test {
    use std::sync::Arc;

    use log::warn;

    use crate::lower::LowerAlloc;
    use crate::stop::{StopRand, Stopper};
    use crate::table::PT_LEN;
    use crate::thread;
    use crate::util::{logging, Page};

    type Lower = super::fixed::FixedLower;

    #[test]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 12;
        let mut buffer = vec![Page::new(); 2 * THREADS * PT_LEN * PT_LEN];

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
        let mut pages = [0; PT_LEN];
        let mut buffer = vec![Page::new(); 2 * THREADS * PT_LEN * PT_LEN];

        for _ in 0..32 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(Lower::new(THREADS, &mut buffer));
            lower.clear();
            assert_eq!(lower.dbg_allocated_pages(), 0);

            for page in &mut pages[..PT_LEN - 3] {
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

            assert_eq!(lower.dbg_allocated_pages(), PT_LEN - 3);
        }
    }
}
