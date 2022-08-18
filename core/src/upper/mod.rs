use core::any::type_name;
use core::cell::UnsafeCell;
use core::fmt;

use alloc::string::String;

use crate::atomic::Atomic;
use crate::entry::Entry3;
use crate::table::{Mapping, PT_LEN};
use crate::util::Page;
use crate::{Error, Result};

mod array_aligned;
pub use array_aligned::{ArrayAligned, CacheAligned, Unaligned};
mod array_atomic;
pub use array_atomic::ArrayAtomic;
mod array_list;
pub use array_list::ArrayList;
mod list_local;
pub use list_local::ListLocal;
mod list_locked;
pub use list_locked::ListLocked;

pub const CAS_RETRIES: usize = 8;
pub const MAGIC: usize = 0xdead_beef;
pub const MIN_PAGES: usize = PT_LEN * PT_LEN;
pub const MAX_PAGES: usize = Mapping([9; 4]).span(4);

pub trait Alloc: Sync + Send + fmt::Debug {
    /// Initialize the allocator.
    /// When `persistent` is set all level 1 and 2 page tables are allocated
    /// at the end of `memory` together with an meta page.
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], persistent: bool) -> Result<()>;

    /// Init as entirely free area.
    #[cold]
    fn free_all(&self) -> Result<()>;

    /// Init as fully reserved area.
    #[cold]
    fn reserve_all(&self) -> Result<()>;

    /// Recover the allocator from persistent memory.
    #[cold]
    fn recover(&self) -> Result<()> {
        Err(Error::Initialization)
    }

    /// Allocate a new page.
    fn get(&self, core: usize, order: usize) -> Result<u64>;
    /// Free the given page.
    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()>;

    /// Return the number of pages that can be allocated.
    fn pages(&self) -> usize;

    /// Returns the minimal number of pages the allocator needs.
    fn pages_needed(&self, cores: usize) -> usize;
    /// Return the number of allocated pages.
    #[cold]
    fn dbg_allocated_pages(&self) -> usize;
    #[cold]
    fn dbg_for_each_huge_page(&self, f: fn(usize));
    #[cold]
    fn name(&self) -> String {
        name::<Self>()
    }
}

#[must_use]
pub fn name<A: Alloc + ?Sized>() -> String {
    let name = type_name::<A>();
    // Add first letter of generic type as suffix
    let (name, first, second, size) = if let Some((prefix, suffix)) = name.split_once('<') {
        // Aligned / Unaligned allocator
        let (first, suffix) = if let Some((first, suffix)) = suffix.split_once(", ") {
            // Strip path
            let first = first.rsplit_once(':').map_or(first, |s| s.1);
            (&first[0..1], suffix)
        } else {
            ("", suffix)
        };

        // Strip path
        let suffix = suffix.rsplit_once(':').map_or(suffix, |s| s.1);
        // Lower allocator size
        let size = suffix
            .split_once('<')
            .map(|(_, s)| s.split_once('>').map_or(s, |s| s.0))
            .unwrap_or_default();
        (prefix, first, &suffix[0..1], size)
    } else {
        (name, "", "", "")
    };

    // Strip namespaces
    let name = name.rsplit_once(':').map_or(name, |s| s.1);
    let name = name.strip_suffix("Alloc").unwrap_or(name);
    format!("{name}{first}{second}{size}")
}

/// Per core data.
pub struct Local<const L: usize> {
    start: Atomic<usize>,
    pte: Atomic<Entry3>,
    /// # Safety
    /// This should only be accessed from the corresponding (virtual) CPU core!
    inner: UnsafeCell<Inner<L>>,
}

#[repr(align(64))]
struct Inner<const L: usize> {
    frees: [usize; L],
    frees_i: usize,
}

impl<const L: usize> Local<L> {
    fn new() -> Self {
        Self {
            start: Atomic::new(usize::MAX),
            pte: Atomic::new(Entry3::new().with_idx(Entry3::IDX_MAX)),
            inner: UnsafeCell::new(Inner {
                frees_i: 0,
                frees: [usize::MAX; L],
            }),
        }
    }
    #[allow(clippy::mut_from_ref)]
    fn p(&self) -> &mut Inner<L> {
        unsafe { &mut *self.inner.get() }
    }
    /// Add a chunk (subtree) id to the history of chunks.
    pub fn frees_push(&self, chunk: usize) {
        self.p().frees_i = (self.p().frees_i + 1) % self.p().frees.len();
        self.p().frees[self.p().frees_i] = chunk;
    }
    /// Calls frees_push on exiting scope.
    /// NOTE: Bind the return value to a variable!
    #[must_use]
    pub fn defer_frees_push(&self, chunk: usize) -> LocalFreePush<'_, L> {
        LocalFreePush(self, chunk)
    }
    /// Checks if the previous frees were in the given chunk.
    pub fn frees_related(&self, chunk: usize) -> bool {
        self.p().frees.iter().all(|p| *p == chunk)
    }
}

impl<const L: usize> fmt::Debug for Local<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("start", &self.start.load())
            .field("pte", &self.pte.load())
            .field("frees", &self.p().frees)
            .field("frees_i", &self.p().frees_i)
            .finish()
    }
}

/// Calls `frees_push` on drop.
pub struct LocalFreePush<'a, const L: usize>(&'a Local<L>, usize);
impl<'a, const L: usize> Drop for LocalFreePush<'a, L> {
    fn drop(&mut self) {
        self.0.frees_push(self.1);
    }
}

#[cfg(all(test, feature = "std"))]
mod test {

    use core::any::type_name;
    use core::ptr::null_mut;
    use std::sync::Arc;
    use std::time::Instant;

    use alloc::vec::Vec;
    use spin::Barrier;

    use log::{info, warn};

    use super::Local;
    use crate::lower::*;
    use crate::mmap::MMap;
    use crate::table::PT_LEN;
    use crate::thread;
    use crate::upper::array_aligned::{CacheAligned, Unaligned};
    use crate::upper::*;
    use crate::util::{logging, Page, WyRand};
    use crate::Error;

    type Lower = Atom<128>;
    type Allocator = ArrayList<Lower>;

    fn mapping(begin: usize, length: usize) -> core::result::Result<MMap<Page>, ()> {
        #[cfg(target_os = "linux")]
        if let Ok(file) = std::env::var("NVM_FILE") {
            warn!("MMap file {file} l={}G", (length * Page::SIZE) >> 30);
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(file)
                .unwrap();
            return MMap::dax(begin, length, f);
        }
        MMap::anon(begin, length)
    }

    #[test]
    fn names() {
        println!(
            "{}\n -> {}",
            type_name::<ArrayAtomic<Lower>>(),
            super::name::<ArrayAtomic<Lower>>()
        );
        println!(
            "{}\n -> {}",
            type_name::<ListLocal>(),
            super::name::<ListLocal>()
        );
        println!(
            "{}\n -> {}",
            type_name::<ArrayAligned<Unaligned, Lower>>(),
            super::name::<ArrayAligned<Unaligned, Lower>>()
        );
        println!(
            "{}\n -> {}",
            type_name::<ArrayAligned<CacheAligned, Lower>>(),
            super::name::<ArrayAligned<CacheAligned, Lower>>()
        );
    }

    /// Testing the related pages heuristic for frees
    #[test]
    fn related_pages() {
        let local = Local::<4>::new();
        let page1 = 43;
        let i1 = page1 / (512 * 512);
        assert!(!local.frees_related(i1));
        local.frees_push(i1);
        local.frees_push(i1);
        local.frees_push(i1);
        assert!(!local.frees_related(i1));
        local.frees_push(i1);
        assert!(local.frees_related(i1));
        let page2 = 512 * 512 + 43;
        let i2 = page2 / (512 * 512);
        assert_ne!(i1, i2);
        local.frees_push(i2);
        assert!(!local.frees_related(i1));
        assert!(!local.frees_related(i2));

        {
            let _push1 = local.defer_frees_push(i1);
            local.frees_push(i1);
            local.frees_push(i1);
            let _push2 = local.defer_frees_push(i1);
            assert!(!local.frees_related(i1));
        };
        assert!(local.frees_related(i1));
    }

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        warn!("start alloc...");
        let small = alloc.get(0, 0).unwrap();

        assert_eq!(alloc.dbg_allocated_pages(), 1, "{alloc:?}");
        warn!("stress test...");

        // Stress test
        let mut pages = Vec::new();
        loop {
            match alloc.get(0, 0) {
                Ok(page) => pages.push(page),
                Err(Error::Memory) => break,
                Err(e) => panic!("{:?}", e),
            }
        }

        warn!("allocated {}", 1 + pages.len());
        warn!("check...");

        assert_eq!(alloc.dbg_allocated_pages(), 1 + pages.len());
        assert_eq!(alloc.dbg_allocated_pages(), alloc.pages());
        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = PT_LEN * PT_LEN - 10;
        for page in &pages[..FREE_NUM] {
            alloc.put(0, *page, 0).unwrap();
        }

        assert_eq!(
            alloc.dbg_allocated_pages(),
            1 + pages.len() - FREE_NUM,
            "{alloc:?}"
        );

        // Realloc
        for page in &mut pages[..FREE_NUM] {
            *page = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        alloc.put(0, small, 0).unwrap();
        // Free all
        for page in &pages {
            alloc.put(0, *page, 0).unwrap();
        }

        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn rand() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 4 << 30;
        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        warn!("start alloc...");
        const ALLOCS: usize = MEM_SIZE / Page::SIZE / 4 * 3;
        let mut pages = Vec::with_capacity(ALLOCS);
        for _ in 0..ALLOCS {
            pages.push(alloc.get(0, 0).unwrap());
        }
        warn!("allocated {}", pages.len());

        warn!("check...");
        assert_eq!(alloc.dbg_allocated_pages(), pages.len());
        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("reallocate rand...");
        let mut rng = WyRand::new(100);
        rng.shuffle(&mut pages);

        for _ in 0..pages.len() {
            let i = rng.range(0..pages.len() as _) as usize;
            alloc.put(0, pages[i], 0).unwrap();
            pages[i] = alloc.get(0, 0).unwrap();
        }

        warn!("check...");
        assert_eq!(alloc.dbg_allocated_pages(), pages.len());
        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("free...");
        rng.shuffle(&mut pages);
        for page in &pages {
            alloc.put(0, *page, 0).unwrap();
        }
        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn multirand() {
        const THREADS: usize = 4;
        const MEM_SIZE: usize = (8 << 30) / Page::SIZE;
        const ALLOCS: usize = ((MEM_SIZE / THREADS) / 4) * 3;

        logging();
        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE).unwrap();
        let range = mapping.as_ptr_range();
        info!("mmap {MEM_SIZE} bytes at {range:?}");
        let range = range.start as u64..range.end as u64;

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        let a = alloc.clone();
        let barrier = Arc::new(Barrier::new(THREADS));
        thread::parallel(THREADS, move |t| {
            thread::pin(t);

            barrier.wait();
            warn!("start alloc...");
            let mut pages = Vec::with_capacity(ALLOCS);
            for _ in 0..ALLOCS {
                pages.push(alloc.get(t, 0).unwrap());
            }
            warn!("allocated {}", pages.len());

            warn!("check...");
            // Check that the same page was not allocated twice
            pages.sort_unstable();
            for i in 0..pages.len() - 1 {
                let p1 = pages[i];
                let p2 = pages[i + 1];
                assert!(range.contains(&p1), "{} not in {:?}", p1, range);
                assert!(p1 != p2, "{}", p1);
            }

            barrier.wait();
            warn!("reallocate rand...");
            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut pages);

            for _ in 0..pages.len() {
                let i = rng.range(0..pages.len() as _) as usize;
                alloc.put(t, pages[i], 0).unwrap();
                pages[i] = alloc.get(t, 0).unwrap();
            }

            warn!("check...");
            // Check that the same page was not allocated twice
            pages.sort_unstable();
            for i in 0..pages.len() - 1 {
                let p1 = pages[i];
                let p2 = pages[i + 1];
                assert!(range.contains(&p1), "{} not in {:?}", p1, range);
                assert!(p1 != p2, "{}", p1);
            }

            warn!("free...");
            rng.shuffle(&mut pages);
            for page in &pages {
                alloc.put(t, *page, 0).unwrap();
            }
        });

        assert_eq!(a.dbg_allocated_pages(), 0);
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        assert_eq!(alloc.dbg_allocated_pages(), 0);

        warn!("start alloc");
        let small = alloc.get(0, 0).unwrap();
        let huge = alloc.get(0, 9).unwrap();

        let expected_pages = 1 + (1 << 9);
        assert_eq!(alloc.dbg_allocated_pages(), expected_pages);
        assert!(small != huge);

        warn!("start stress test");

        // Stress test
        let mut pages = vec![0; PT_LEN * PT_LEN];
        for page in &mut pages {
            *page = alloc.get(0, 0).unwrap();
        }

        warn!("check");

        assert_eq!(alloc.dbg_allocated_pages(), expected_pages + pages.len());

        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("free some...");

        // Free some
        for page in &pages[10..PT_LEN + 10] {
            alloc.put(0, *page, 0).unwrap();
        }

        warn!("free special...");

        alloc.put(0, small, 0).unwrap();
        alloc.put(0, huge, 9).unwrap();

        warn!("realloc...");

        // Realloc
        for page in &mut pages[10..PT_LEN + 10] {
            *page = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        // Free all
        for page in &pages {
            alloc.put(0, *page, 0).unwrap();
        }

        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn parallel_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const PAGES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;
        let timer = Instant::now();

        let a = alloc.clone();
        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = a.get(t, 0).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.dbg_allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }
    }

    #[test]
    #[ignore]
    fn less_mem() {
        logging();

        const THREADS: usize = 4;
        const PAGES: usize = 4096;
        const ALLOC_PER_THREAD: usize = PAGES / THREADS - THREADS;

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;
        let timer = Instant::now();

        let a = alloc.clone();
        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = a.get(t, 0).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.dbg_allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        let barrier = Arc::new(Barrier::new(THREADS));
        let a = alloc.clone();
        thread::parallel(THREADS, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let addr = unsafe { *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                a.put(t, addr, 0).unwrap();
            }
        });

        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn parallel_huge_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN - 1;
        // additional space for the allocators metadata
        const PAGES: usize = THREADS * (ALLOC_PER_THREAD + 2) * PT_LEN;

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;
        let timer = Instant::now();

        let a = alloc.clone();
        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = a.get(t, 9).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.dbg_allocated_pages(), pages.len() * PT_LEN);
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }
    }

    #[ignore]
    #[test]
    fn parallel_malloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let pages_begin = pages.as_ptr() as usize;
        let barrier = Arc::new(Barrier::new(THREADS));
        let timer = Instant::now();

        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = unsafe { libc::malloc(Page::SIZE) } as u64;
                assert!(*dst != 0);
            }
        });

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        let mut last = None;
        for p in pages {
            assert!(last != Some(p));
            unsafe { libc::free(p as _) };
            last = Some(p);
        }
    }

    #[ignore]
    #[test]
    fn parallel_mmap() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let pages_begin = pages.as_ptr() as usize;
        let barrier = Arc::new(Barrier::new(THREADS));
        let timer = Instant::now();

        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = unsafe {
                    libc::mmap(
                        null_mut(),
                        Page::SIZE,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                } as u64;
                assert!(*dst != 0);
            }
        });

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        let mut last = None;
        for p in pages {
            assert!(last != Some(p));
            unsafe { libc::munmap(p as _, Page::SIZE) };
            last = Some(p);
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * PT_LEN * PT_LEN * Page::SIZE;

        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        // Stress test
        let barrier = Arc::new(Barrier::new(THREADS));

        let a = alloc.clone();
        thread::parallel(THREADS, move |t| {
            thread::pin(t);
            barrier.wait();

            let mut pages = vec![0; ALLOC_PER_THREAD];

            for page in &mut pages {
                *page = alloc.get(t, 0).unwrap();
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut pages);

            for page in pages {
                alloc.put(t, page, 0).unwrap();
            }
        });

        warn!("check");
        assert_eq!(a.dbg_allocated_pages(), 0);
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10) / 2;

        let mut mapping = mapping(0x1000_0000_0000, 4 * PT_LEN * PT_LEN).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });

        let barrier = Arc::new(Barrier::new(THREADS));

        // Alloc on first thread
        thread::pin(0);
        let mut pages = vec![0; ALLOC_PER_THREAD];
        for page in &mut pages {
            *page = alloc.get(0, 0).unwrap();
        }

        let handle = {
            let barrier = barrier.clone();
            let alloc = alloc.clone();
            std::thread::spawn(move || {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for page in &pages {
                    alloc.put(1, *page, 0).unwrap();
                }
            })
        };

        let mut pages = vec![0; ALLOC_PER_THREAD];

        barrier.wait();

        // Simultaneously alloc on first thread
        for page in &mut pages {
            *page = alloc.get(0, 0).unwrap();
        }

        handle.join().unwrap();

        warn!("check");
        assert_eq!(alloc.dbg_allocated_pages(), ALLOC_PER_THREAD);
    }

    #[test]
    fn recover() {
        logging();

        let mut mapping = mapping(0x1000_0000_0000, 8 << 18).unwrap();
        thread::pin(0);

        let expected_pages = (PT_LEN + 2) * (1 + (1 << 9));

        {
            let alloc = Arc::new({
                let mut a = Allocator::default();
                a.init(1, &mut mapping, true).unwrap();
                a.free_all().unwrap();
                a
            });

            for _ in 0..PT_LEN + 2 {
                alloc.get(0, 0).unwrap();
                alloc.get(0, 9).unwrap();
            }

            assert_eq!(alloc.dbg_allocated_pages(), expected_pages);

            // leak
            let _ = Arc::into_raw(alloc);
        }

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, true).unwrap();
            a.recover().unwrap();
            a
        });

        assert_eq!(alloc.dbg_allocated_pages(), expected_pages);
    }

    #[test]
    fn different_orders() {
        const MAX_ORDER: usize = Lower::MAX_ORDER;
        const THREADS: usize = 4;

        logging();

        let mut mapping = mapping(0x1000_0000_0000, Lower::N * (THREADS * 2 + 1)).unwrap();
        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.free_all().unwrap();
            a
        });
        let a = alloc.clone();

        let barrier = Arc::new(Barrier::new(THREADS));

        thread::parallel(THREADS, move |t| {
            thread::pin(t);
            let mut rng = WyRand::new(42 + t as u64);
            let mut num_pages = 0;
            let mut pages = Vec::new();
            for order in 0..=MAX_ORDER {
                for _ in 0..1 << (MAX_ORDER - order) {
                    pages.push((order, 0));
                    num_pages += 1 << order;
                }
            }
            rng.shuffle(&mut pages);

            warn!("allocate {num_pages} pages up to order {MAX_ORDER}");
            barrier.wait();

            for (order, page) in &mut pages {
                *page = match alloc.get(t, *order) {
                    Ok(page) => page,
                    Err(e) => panic!("{e:?} o={order} {alloc:?}"),
                };
                assert!(
                    *page % ((1 << *order) * Page::SIZE) as u64 == 0,
                    "{page:x} {:x}",
                    (1 << *order) * Page::SIZE
                );
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut pages);

            for (order, page) in pages {
                match alloc.put(t, page, order) {
                    Ok(_) => {}
                    Err(e) => panic!("{e:?} o={order} {alloc:?}"),
                }
            }
        });

        assert_eq!(a.dbg_allocated_pages(), 0);
    }

    #[test]
    fn init_reserved() {
        logging();

        const THREADS: usize = 2;
        const PAGES: usize = 8 << 18;

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, false).unwrap();
            a.reserve_all().unwrap();
            a
        });
        assert_eq!(alloc.dbg_allocated_pages(), PAGES);

        for pages in mapping.chunks_exact(1 << Lower::MAX_ORDER) {
            alloc.put(0, pages.as_ptr() as _, Lower::MAX_ORDER).unwrap();
        }
        assert_eq!(alloc.dbg_allocated_pages(), 0);

        let a = alloc.clone();
        thread::parallel(THREADS, move |core| {
            thread::pin(core);
            for _ in 0..(PAGES / THREADS) / (1 << Lower::MAX_ORDER) {
                alloc.get(core, Lower::MAX_ORDER).unwrap();
            }
        });
        assert_eq!(a.dbg_allocated_pages(), PAGES);
    }
}
