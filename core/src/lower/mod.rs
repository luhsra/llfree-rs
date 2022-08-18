use core::fmt;
use core::ops::Range;

use crate::util::Page;
use crate::Result;

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

mod cache;
pub use cache::Cache;
mod atom;
pub use atom::Atom;

/// Level 2 page allocator.
pub trait LowerAlloc: Default + fmt::Debug {
    /// Pages per subtree
    const N: usize;
    /// The maximal allowed order of this allocator
    const MAX_ORDER: usize;

    /// Create a new lower allocator.
    fn new(cores: usize, memory: &mut [Page], persistent: bool) -> Self;

    fn pages(&self) -> usize;
    fn memory(&self) -> Range<*const Page>;

    /// Free all pages.
    fn free_all(&self);
    /// Reserve all pages.
    fn reserve_all(&self);
    /// Recover the level 2 page table at `start`.
    /// If deep, the level 1 pts are also traversed and false counters are corrected.
    /// Returns the number of recovered pages.
    fn recover(&self, start: usize, deep: bool) -> Result<usize>;

    /// Try allocating a new `huge` page in the subtree at `start`.
    fn get(&self, start: usize, order: usize) -> Result<usize>;
    /// Try freeing a page. Returns if it was huge.
    fn put(&self, page: usize, order: usize) -> Result<()>;

    /// Debug function, returning the number of allocated pages and performing internal checks.
    fn dbg_allocated_pages(&self) -> usize;
    /// Debug function returning number of free pages in each order 9 chunk
    fn dbg_for_each_huge_page<F: FnMut(usize)>(&self, f: F);
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

    type Lower = super::atom::Atom<512>;

    #[test]
    #[ignore]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 6;
        let mut buffer = vec![Page::new(); 2 * THREADS * PT_LEN * PT_LEN];

        for _ in 0..8 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(Lower::new(THREADS, &mut buffer, true));
            lower.free_all();
            assert_eq!(lower.dbg_allocated_pages(), 0);

            let stop = StopRand::new(THREADS, seed);
            let l = lower.clone();
            thread::parallel(THREADS, move |t| {
                let _stopper = Stopper::init(stop, t);

                let mut pages = [0; 4];
                for p in &mut pages {
                    *p = l.get(0, 0).unwrap();
                }
                pages.reverse();
                for p in pages {
                    l.put(p, 0).unwrap();
                }
            });

            assert_eq!(lower.dbg_allocated_pages(), 0);
        }
    }

    #[test]
    #[ignore]
    fn rand_realloc_last() {
        logging();

        const THREADS: usize = 6;
        let mut pages = [0; PT_LEN];
        let mut buffer = vec![Page::new(); 2 * THREADS * PT_LEN * PT_LEN];

        for _ in 0..8 {
            let seed = unsafe { libc::rand() } as u64;
            warn!("order: {seed:x}");

            let lower = Arc::new(Lower::new(THREADS, &mut buffer, true));
            lower.free_all();
            assert_eq!(lower.dbg_allocated_pages(), 0);

            for page in &mut pages[..PT_LEN - 3] {
                *page = lower.get(0, 0).unwrap();
            }

            let stop = StopRand::new(THREADS, seed);
            let l = lower.clone();
            thread::parallel(THREADS, move |t| {
                let _stopper = Stopper::init(stop, t);

                if t < THREADS / 2 {
                    l.put(pages[t], 0).unwrap();
                } else {
                    l.get(0, 0).unwrap();
                }
            });

            assert_eq!(lower.dbg_allocated_pages(), PT_LEN - 3);
        }
    }
}
