use core::any::type_name;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use bitfield_struct::bitfield;

use crate::atomic::Atom;
use crate::entry::ReservedTree;
use crate::table::PT_LEN;
use crate::util::{CacheLine, Page};
use crate::Result;

mod array;
pub use array::Array;
mod array_list;
#[allow(deprecated)]
pub use array_list::ArrayList;
mod list_local;
pub use list_local::ListLocal;
mod list_locked;
pub use list_locked::ListLocked;

/// Number of retries if an atomic operation fails.
pub const CAS_RETRIES: usize = 16;
/// Magic marking the meta page.
pub const MAGIC: usize = 0x_dead_beef;
/// Minimal number of pages an allocator needs (1G).
pub const MIN_PAGES: usize = PT_LEN * PT_LEN;
/// Maximal number of pages an allocator can manage (about 256TiB).
pub const MAX_PAGES: usize = PT_LEN * PT_LEN * PT_LEN * PT_LEN;

/// The general interface of the allocator implementations.
pub trait Alloc: Sync + Send + fmt::Debug {
    /// Initialize the allocator.
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], init: Init, free_all: bool)
        -> Result<()>;

    /// Allocate a new page.
    fn get(&self, core: usize, order: usize) -> Result<u64>;
    /// Free the given page.
    fn put(&self, core: usize, addr: u64, order: usize) -> Result<()>;
    /// Returns if the page is free. This might be racy!
    fn is_free(&self, addr: u64, order: usize) -> bool;

    /// Return the total number of pages the allocator manages.
    fn pages(&self) -> usize;

    /// Unreserve cpu-local pages
    fn drain(&self, _core: usize) -> Result<()> {
        Ok(())
    }

    /// Return the number of allocated pages.
    fn dbg_allocated_pages(&self) -> usize {
        self.pages() - self.dbg_free_pages()
    }
    /// Return the number of free pages.
    fn dbg_free_pages(&self) -> usize;
    /// Return the number of free huge pages or 0 if the allocator cannot allocate huge pages.
    fn dbg_free_huge_pages(&self) -> usize {
        0
    }
    /// Execute f for each huge page with the number of free pages
    /// in this huge page as parameter.
    #[cold]
    fn dbg_for_each_huge_page(&self, _f: fn(usize)) {}
    /// Return the name of the allocator.
    #[cold]
    fn name(&self) -> AllocName {
        AllocName::new::<Self>()
    }
}

/// Defines if the allocator should be allocated persistently
/// and if it in that case should try to recover from the persistent memory.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Init {
    /// Not persistent
    Volatile,
    /// Persistent and try recovery
    Recover,
    /// Overwrite the persistent memory
    Overwrite,
}

/// Wrapper for creating a new allocator instance
pub trait AllocExt: Sized + Alloc + Default {
    /// Create and initialize the allocator.
    #[cold]
    fn new(cores: usize, memory: &mut [Page], init: Init, free_all: bool) -> Result<Self> {
        let mut a = Self::default();
        a.init(cores, memory, init, free_all)?;
        Ok(a)
    }
}
// Implement for all default initializable allocators
impl<A: Sized + Alloc + Default> AllocExt for A {}

/// The short name of an allocator.
///
/// E.g.: `Array4C32` for
/// `nvalloc::upper::array::Array<4, nvalloc::lower::cache::Cache<32>>`
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct AllocName([&'static str; 4]);

impl fmt::Display for AllocName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for part in self.0 {
            write!(f, "{part}")?;
        }
        Ok(())
    }
}

impl AllocName {
    pub fn new<A: Alloc + ?Sized>() -> Self {
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
        let base = name.rsplit_once(':').map_or(name, |s| s.1);
        let base = base.strip_suffix("Alloc").unwrap_or(base);
        Self([base, first, second, size])
    }
}

impl PartialEq<&str> for AllocName {
    fn eq(&self, other: &&str) -> bool {
        let mut remainder = *other;
        for part in self.0 {
            if let Some(r) = remainder.strip_prefix(part) {
                remainder = r;
            } else {
                return false;
            };
        }
        remainder.is_empty()
    }
}

impl PartialEq<AllocName> for &str {
    fn eq(&self, other: &AllocName) -> bool {
        other.eq(self)
    }
}

#[bitfield(u64)]
#[derive(Default, PartialEq, Eq)]
pub struct LastFrees {
    #[bits(47)]
    tree_index: usize,
    #[bits(17)]
    count: usize,
}

/// Per core data.
pub struct Local<const F: usize> {
    /// Local copy of the reserved level 3 entry
    reserved: CacheLine<Atom<ReservedTree>>,
    /// Last frees
    last_frees: CacheLine<AtomicU64>,
}

impl<const F: usize> Local<F> {
    fn new() -> Self {
        Self {
            reserved: CacheLine(Atom::new(ReservedTree::default())),
            last_frees: CacheLine(AtomicU64::new(LastFrees::default().into())),
        }
    }
    /// Add a tree index to the history.
    pub fn frees_push(&self, tree_index: usize) {
        // If the update of this heuristic fails, ignore it
        let _ = self
            .last_frees
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                let v = LastFrees::from(v);
                if v.tree_index() == tree_index {
                    if v.count() < F {
                        Some(v.with_count(v.count() + 1).into())
                    } else {
                        None
                    }
                } else {
                    Some(
                        LastFrees::new()
                            .with_tree_index(tree_index)
                            .with_count(1)
                            .into(),
                    )
                }
            });
    }
    /// Calls frees_push on scope exit.
    /// # Note
    /// Bind the return value to a **named** variable!
    #[must_use]
    pub fn defer_frees_push(&self, tree_index: usize) -> LocalFreePush<'_, F> {
        LocalFreePush(self, tree_index)
    }
    /// Checks if the previous `count` frees had the same tree index.
    pub fn frees_in_tree(&self, tree_index: usize) -> bool {
        let lf = LastFrees::from(self.last_frees.load(Ordering::Relaxed));
        lf.tree_index() == tree_index && lf.count() >= F
    }
}

impl<const F: usize> fmt::Debug for Local<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("reserved", &self.reserved.load())
            .field(
                "frees",
                &LastFrees::from(self.last_frees.load(Ordering::Relaxed)),
            )
            .finish()
    }
}

/// Calls `frees_push` on scope exit.
pub struct LocalFreePush<'a, const F: usize>(&'a Local<F>, usize);
impl<'a, const F: usize> Drop for LocalFreePush<'a, F> {
    fn drop(&mut self) {
        self.0.frees_push(self.1);
    }
}

#[cfg(all(test, feature = "std"))]
mod test {

    use core::any::type_name;
    use core::mem::size_of;
    use core::ptr::null_mut;
    use std::time::Instant;

    use alloc::vec::Vec;
    use std::sync::Barrier;

    use log::{info, warn};

    use crate::lower::*;
    use crate::mmap::test_mapping;
    use crate::table::PT_LEN;
    use crate::thread;
    use crate::upper::*;
    use crate::util::{logging, Page, WyRand};
    use crate::Error;

    type Lower = Cache<32>;
    type Allocator = Array<4, Lower>;

    #[test]
    fn names() {
        println!(
            "{}\n -> {}",
            type_name::<Array<4, Lower>>(),
            AllocName::new::<Array<4, Lower>>()
        );
        assert_eq!("Array4C32", AllocName::new::<Array<4, Lower>>());
        println!(
            "{}\n -> {}",
            type_name::<ListLocal>(),
            AllocName::new::<ListLocal>()
        );
        assert_eq!("ListLocal", AllocName::new::<ListLocal>());
    }

    #[test]
    fn sizes() {
        type C32 = Cache<32>;
        type A4C32 = Array<4, C32>;
        println!("{}:", AllocName::new::<A4C32>());
        println!("  Static size {}B", size_of::<A4C32>());
        println!("  Size per CPU: {}B", size_of::<Local<4>>());
        println!("  Size per GiB: {}B", C32::size_per_gib());
    }

    /// Testing the related pages heuristic for frees
    #[test]
    fn last_frees() {
        let local = Local::<4>::new();
        let page1 = 43;
        let i1 = page1 / (512 * 512);
        assert!(!local.frees_in_tree(i1));
        local.frees_push(i1);
        local.frees_push(i1);
        local.frees_push(i1);
        assert!(!local.frees_in_tree(i1));
        local.frees_push(i1);
        assert!(local.frees_in_tree(i1));
        let page2 = 512 * 512 + 43;
        let i2 = page2 / (512 * 512);
        assert_ne!(i1, i2);
        local.frees_push(i2);
        assert!(!local.frees_in_tree(i1));
        assert!(!local.frees_in_tree(i2));

        {
            let _push1 = local.defer_frees_push(i1);
            local.frees_push(i1);
            local.frees_push(i1);
            let _push2 = local.defer_frees_push(i1);
            assert!(!local.frees_in_tree(i1));
        };
        assert!(local.frees_in_tree(i1));
    }

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        let mut mapping = test_mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();
        let range = mapping.as_ptr_range();
        let range = range.start as u64..range.end as u64;

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Allocator::new(1, &mut mapping, Init::Volatile, true).unwrap();

        assert_eq!(alloc.dbg_free_pages(), alloc.pages());

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
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + pages.len());
        warn!("check...");

        assert_eq!(alloc.dbg_allocated_pages(), 1 + pages.len());
        assert_eq!(alloc.dbg_allocated_pages(), alloc.pages());
        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
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
        let mut mapping = test_mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();
        let range = mapping.as_ptr_range();
        let range = range.start as u64..range.end as u64;

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Allocator::new(1, &mut mapping, Init::Volatile, true).unwrap();

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
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
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
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
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
        let mut mapping = test_mapping(0x1000_0000_0000, MEM_SIZE).unwrap();
        let range = mapping.as_ptr_range();
        info!("mmap {MEM_SIZE} bytes at {range:?}");
        let range = range.start as u64..range.end as u64;

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();

        let barrier = Barrier::new(THREADS);
        thread::parallel(0..THREADS, |t| {
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
            for &[a, b] in pages.array_windows() {
                assert_ne!(a, b);
                assert!(range.contains(&a) && range.contains(&b));
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
            for &[a, b] in pages.array_windows() {
                assert_ne!(a, b);
                assert!(range.contains(&a) && range.contains(&b));
            }

            warn!("free...");
            rng.shuffle(&mut pages);
            for page in &pages {
                alloc.put(t, *page, 0).unwrap();
            }
        });

        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mut mapping = test_mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();
        let range = mapping.as_ptr_range();
        let range = range.start as u64..range.end as u64;

        info!("mmap {MEM_SIZE} bytes at {:?}", mapping.as_ptr());

        let alloc = Allocator::new(1, &mut mapping, Init::Volatile, true).unwrap();

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
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
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

        let mut mapping = test_mapping(0x1000_0000_0000, PAGES).unwrap();
        let range = mapping.as_ptr_range();
        let range = range.start as u64..range.end as u64;

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            pages.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, pages)| {
                thread::pin(t);
                barrier.wait();

                for page in pages {
                    *page = alloc.get(t, 0).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.dbg_allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
        }
    }

    #[test]
    #[ignore]
    fn less_mem() {
        logging();

        const THREADS: usize = 4;
        const PAGES: usize = 4096;
        const ALLOC_PER_THREAD: usize = PAGES / THREADS - THREADS;

        let mut mapping = test_mapping(0x1000_0000_0000, PAGES).unwrap();
        let range = mapping.as_ptr_range();
        let range = range.start as u64..range.end as u64;

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            pages.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, pages)| {
                thread::pin(t);
                barrier.wait();

                for page in pages {
                    *page = alloc.get(t, 0).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.dbg_allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
        }

        thread::parallel(pages.chunks(ALLOC_PER_THREAD).enumerate(), |(t, pages)| {
            thread::pin(t);
            barrier.wait();

            for page in pages {
                alloc.put(t, *page, 0).unwrap();
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

        let mut mapping = test_mapping(0x1000_0000_0000, PAGES).unwrap();
        let range = mapping.as_ptr_range();
        let range = range.start as u64..range.end as u64;

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            pages.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, pages)| {
                thread::pin(t);
                barrier.wait();

                for page in pages {
                    *page = alloc.get(t, 9).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.dbg_allocated_pages(), pages.len() * PT_LEN);
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for &[a, b] in pages.array_windows() {
            assert_ne!(a, b);
            assert!(range.contains(&a) && range.contains(&b));
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
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            pages.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, pages)| {
                thread::pin(t);
                barrier.wait();

                for page in pages {
                    *page = unsafe { libc::malloc(Page::SIZE) } as u64;
                    assert!(*page != 0);
                }
            },
        );

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
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            pages.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, pages)| {
                thread::pin(t);
                barrier.wait();

                for page in pages {
                    *page = unsafe {
                        libc::mmap(
                            null_mut(),
                            Page::SIZE,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                            -1,
                            0,
                        )
                    } as u64;
                    assert!(*page != 0);
                }
            },
        );

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

        let mut mapping = test_mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();
        let barrier = Barrier::new(THREADS);

        // Stress test
        thread::parallel(0..THREADS, |t| {
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
        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10) / 2;

        let mut mapping = test_mapping(0x1000_0000_0000, 4 * PT_LEN * PT_LEN).unwrap();

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();

        // Alloc on first thread
        thread::pin(0);
        let mut pages = vec![0; ALLOC_PER_THREAD];
        for page in &mut pages {
            *page = alloc.get(0, 0).unwrap();
        }

        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            s.spawn(|| {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for page in &pages {
                    alloc.put(1, *page, 0).unwrap();
                }
            });

            let mut pages = vec![0; ALLOC_PER_THREAD];

            barrier.wait();

            // Simultaneously alloc on first thread
            for page in &mut pages {
                *page = alloc.get(0, 0).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.dbg_allocated_pages(), ALLOC_PER_THREAD);
    }

    #[test]
    fn recover() {
        logging();

        let mut mapping = test_mapping(0x1000_0000_0000, 8 << 18).unwrap();
        thread::pin(0);

        let expected_pages = (PT_LEN + 2) * (1 + (1 << 9));

        {
            let alloc = Allocator::new(1, &mut mapping, Init::Overwrite, true).unwrap();

            for _ in 0..PT_LEN + 2 {
                alloc.get(0, 0).unwrap();
                alloc.get(0, 9).unwrap();
            }

            assert_eq!(alloc.dbg_allocated_pages(), expected_pages);

            // leak (crash)
            std::mem::forget(alloc);
        }

        let mut alloc = Allocator::default();
        alloc.init(1, &mut mapping, Init::Recover, true).unwrap();
        assert_eq!(alloc.dbg_allocated_pages(), expected_pages);
    }

    #[test]
    fn different_orders() {
        const MAX_ORDER: usize = Lower::MAX_ORDER;
        const THREADS: usize = 4;

        logging();

        let mut mapping = test_mapping(0x1000_0000_0000, Lower::N * (THREADS * 2 + 1)).unwrap();
        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, true).unwrap();

        let barrier = Barrier::new(THREADS);

        thread::parallel(0..THREADS, |t| {
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

            warn!("allocate {num_pages} pages up to order <{MAX_ORDER}");
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

        assert_eq!(alloc.dbg_allocated_pages(), 0);
    }

    #[test]
    fn init_reserved() {
        logging();

        const THREADS: usize = 2;
        const PAGES: usize = 8 << 18;

        let mut mapping = test_mapping(0x1000_0000_0000, PAGES).unwrap();
        let pages = mapping
            .chunks_exact(1 << Lower::MAX_ORDER)
            .map(|p| p.as_ptr() as u64)
            .collect::<Vec<_>>();

        let alloc = Allocator::new(THREADS, &mut mapping, Init::Volatile, false).unwrap();
        assert_eq!(alloc.dbg_allocated_pages(), PAGES);

        for page in pages {
            alloc.put(0, page, Lower::MAX_ORDER).unwrap();
        }
        assert_eq!(alloc.dbg_allocated_pages(), 0);

        thread::parallel(0..THREADS, |core| {
            thread::pin(core);
            for _ in 0..(PAGES / THREADS) / (1 << Lower::MAX_ORDER) {
                alloc.get(core, Lower::MAX_ORDER).unwrap();
            }
        });
        assert_eq!(alloc.dbg_allocated_pages(), PAGES);
    }

    #[test]
    fn fragmentation_retry() {
        logging();

        let mut mapping = test_mapping(0x1000_0000_0000, Lower::N * 2).unwrap();
        let alloc = Allocator::new(1, &mut mapping, Init::Volatile, true).unwrap();

        // Alloc a whole subtree
        let mut pages = Vec::with_capacity(Lower::N / 2);
        for i in 0..Lower::N {
            if i % 2 == 0 {
                pages.push(alloc.get(0, 0).unwrap());
            } else {
                alloc.get(0, 0).unwrap();
            }
        }
        // Free every second one -> fragmentation
        for page in pages {
            alloc.put(0, page, 0).unwrap();
        }

        let huge = alloc.get(0, 9).unwrap();
        warn!("huge = {huge}");
        warn!("{alloc:?}");
    }

    #[test]
    fn drain() {
        let mut mapping = test_mapping(0x1000_0000_0000, Lower::N * 2).unwrap();
        let alloc = Allocator::new(2, &mut mapping, Init::Volatile, true).unwrap();
        // should not change anything
        alloc.drain(0).unwrap();
        alloc.drain(1).unwrap();

        // allocate on second core => reserve a subtree
        alloc.get(1, 0).unwrap();

        // completely the subtree of the first core
        for _ in 0..Lower::N {
            alloc.get(0, 0).unwrap();
        }
        // next allocation should trigger drain+reservation (no subtree left)
        println!("{:?}", alloc.get(0, 0));
    }
}
