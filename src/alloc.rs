use core::any::type_name;
use core::cell::UnsafeCell;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use log::error;

use crate::entry::Entry3;
use crate::table::{Mapping, PT_LEN};
use crate::util::Page;

mod array_aligned;
pub use array_aligned::ArrayAlignedAlloc;
mod array_atomic;
pub use array_atomic::ArrayAtomicAlloc;
mod array_locked;
pub use array_locked::ArrayLockedAlloc;
mod array_unaligned;
pub use array_unaligned::ArrayUnalignedAlloc;
mod list_local;
pub use list_local::ListLocalAlloc;
mod list_locked;
pub use list_locked::ListLockedAlloc;
mod malloc;
pub use malloc::MallocAlloc;
mod table;
pub use table::TableAlloc;

pub const CAS_RETRIES: usize = 4096;
pub const MAGIC: usize = 0xdead_beef;
pub const MIN_PAGES: usize = 2 * PT_LEN * PT_LEN;
pub const MAX_PAGES: usize = Mapping([512; 4]).span(4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory = 1,
    /// Failed comapare and swap operation
    CAS = 2,
    /// Invalid address
    Address = 3,
    /// Allocator not initialized or initialization failed
    Initialization = 4,
    /// Corrupted allocator state
    Corruption = 5,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Size {
    /// 4KiB
    L0 = 0,
    /// 2MiB
    L1 = 1,
}

impl Size {
    pub fn span(self) -> usize {
        match self {
            Size::L0 => 1,
            Size::L1 => PT_LEN,
        }
    }
}

pub trait Alloc: Sync + Send + fmt::Debug {
    /// Initialize the allocator.
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()>;

    /// Allocate a new page.
    fn get(&self, core: usize, size: Size) -> Result<u64>;
    /// Free the given page.
    fn put(&self, core: usize, addr: u64) -> Result<Size>;

    /// Return the number of pages that can be allocated.
    fn pages(&self) -> usize;
    /// Return the number of allocated pages.
    #[cold]
    fn dbg_allocated_pages(&self) -> usize;
    #[cold]
    fn name(&self) -> String {
        name::<Self>()
    }
}

#[must_use]
pub fn name<A: Alloc + ?Sized>() -> String {
    let name = type_name::<A>();
    // Add first letter of generic type as suffix
    let (name, suffix, size) = if let Some((prefix, suffix)) = name.split_once('<') {
        let suffix = suffix.rsplit_once(':').map_or(suffix, |s| s.1);
        let size = suffix
            .split_once('<')
            .map(|(_, s)| s.split_once('>').map_or(s, |s| s.0))
            .unwrap_or_default();
        (prefix, &suffix[0..1], size)
    } else {
        (name, "", "")
    };

    // Strip namespaces
    let name = name.rsplit_once(':').map_or(name, |s| s.1);
    let name = name.strip_suffix("Alloc").unwrap_or(name);
    format!("{name}{suffix}{size}")
}

/// Allocates a new page and writes the value after translation into `dst`.
/// If `dst` has an other value than `expected` the operation is revoked and `Error::CAS` is returned.
pub fn get_cas<F: FnOnce(u64) -> u64>(
    alloc: &dyn Alloc,
    core: usize,
    size: Size,
    dst: &AtomicU64,
    translate: F,
    expected: u64,
) -> Result<()> {
    let page = alloc.get(core, size)?;
    let new = translate(page);
    if let Err(_) = dst.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
        error!("CAS");
        alloc.put(core, page).unwrap();
        Err(Error::CAS)
    } else {
        Ok(())
    }
}

/// Per core data.
/// # Safety
/// This should only be accessed from the corresponding (virtual) CPU core!
#[repr(transparent)]
pub struct Local<const L: usize>(UnsafeCell<Inner<L>>);
#[repr(align(64))]
struct Inner<const L: usize> {
    start: [usize; 2],
    pte: [Entry3; 2],
    frees: [usize; L],
    frees_i: usize,
}

unsafe impl<const L: usize> Send for Local<L> {}
unsafe impl<const L: usize> Sync for Local<L> {}

impl<const L: usize> Local<L> {
    fn new() -> Self {
        Self(UnsafeCell::new(Inner {
            start: [usize::MAX, usize::MAX],
            pte: [
                Entry3::new().with_idx(Entry3::IDX_MAX),
                Entry3::new().with_idx(Entry3::IDX_MAX),
            ],
            frees_i: 0,
            frees: [usize::MAX; L],
        }))
    }
    #[allow(clippy::mut_from_ref)]
    fn p(&self) -> &mut Inner<L> {
        unsafe { &mut *self.0.get() }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn start(&self, huge: bool) -> &mut usize {
        &mut self.p().start[huge as usize]
    }
    #[allow(clippy::mut_from_ref)]
    pub fn pte(&self, huge: bool) -> &mut Entry3 {
        &mut self.p().pte[huge as usize]
    }
    /// Add a chunk (subtree) id to the history of chunks.
    pub fn frees_push(&self, chunk: usize) {
        self.p().frees_i = (self.p().frees_i + 1) % self.p().frees.len();
        self.p().frees[self.p().frees_i] = chunk;
    }
    /// Calls frees_push on exiting scope. WARN: bin the return value to a variable!
    pub fn frees_push_on_drop<'a>(&'a self, chunk: usize) -> LocalFreePush<'a, L> {
        LocalFreePush(self, chunk)
    }
    /// Checks if the previous frees were in the given chunk.
    pub fn frees_related(&self, chunk: usize) -> bool {
        self.p().frees.iter().all(|p| *p == chunk)
    }
}

pub struct LocalFreePush<'a, const L: usize>(&'a Local<L>, usize);
impl<'a, const L: usize> Drop for LocalFreePush<'a, L> {
    fn drop(&mut self) {
        self.0.frees_push(self.1);
    }
}

#[cfg(test)]
mod test {

    use core::ptr::null_mut;
    use std::sync::Arc;
    use std::time::Instant;

    use spin::Barrier;

    use log::{info, warn};

    use super::Error;
    use super::Local;
    use crate::alloc::Alloc;
    use crate::alloc::MIN_PAGES;
    use crate::lower::*;
    use crate::mmap::MMap;
    use crate::table::PT_LEN;
    use crate::util::{logging, Page, WyRand};
    use crate::{thread, Size};

    type Lower = CacheLower<256>;
    type Allocator = super::ArrayAtomicAlloc<Lower>;

    fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<Page>, ()> {
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
    fn correctly_sized_huge_pages() {
        assert_eq!(Lower::MAPPING.span(1), Size::L1.span())
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
            let _push1 = local.frees_push_on_drop(i1);
            local.frees_push(i1);
            local.frees_push(i1);
            let _push2 = local.frees_push_on_drop(i1);
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
            a.init(1, &mut mapping, true).unwrap();
            a
        });

        warn!("start alloc...");
        let small = alloc.get(0, Size::L0).unwrap();

        assert_eq!(alloc.dbg_allocated_pages(), Size::L0.span());

        // Stress test
        let mut pages = Vec::new();
        loop {
            match alloc.get(0, Size::L0) {
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
            alloc.put(0, *page).unwrap();
        }

        assert_eq!(alloc.dbg_allocated_pages(), 1 + pages.len() - FREE_NUM);

        // Realloc
        for page in &mut pages[..FREE_NUM] {
            *page = alloc.get(0, Size::L0).unwrap();
        }

        warn!("free...");

        alloc.put(0, small).unwrap();
        // Free all
        for page in &pages {
            alloc.put(0, *page).unwrap();
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
            a.init(1, &mut mapping, true).unwrap();
            a
        });

        warn!("start alloc...");
        const ALLOCS: usize = MEM_SIZE / Page::SIZE / 4 * 3;
        let mut pages = Vec::with_capacity(ALLOCS);
        for _ in 0..ALLOCS {
            pages.push(alloc.get(0, Size::L0).unwrap());
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
            alloc.put(0, pages[i]).unwrap();
            pages[i] = alloc.get(0, Size::L0).unwrap();
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
            alloc.put(0, *page).unwrap();
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
            a.init(THREADS, &mut mapping, true).unwrap();
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
                pages.push(alloc.get(t, Size::L0).unwrap());
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
                alloc.put(t, pages[i]).unwrap();
                pages[i] = alloc.get(t, Size::L0).unwrap();
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
                alloc.put(t, *page).unwrap();
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
            a.init(1, &mut mapping, true).unwrap();
            a
        });

        assert_eq!(alloc.dbg_allocated_pages(), 0);

        warn!("start alloc");
        let small = alloc.get(0, Size::L0).unwrap();
        let huge = alloc.get(0, Size::L1).unwrap();

        let expected_pages = Size::L0.span() + Size::L1.span();
        assert_eq!(alloc.dbg_allocated_pages(), expected_pages);
        assert!(small != huge);

        warn!("start stress test");

        // Stress test
        let mut pages = vec![0; PT_LEN * PT_LEN];
        for page in &mut pages {
            *page = alloc.get(0, Size::L0).unwrap();
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
            alloc.put(0, *page).unwrap();
        }

        warn!("free special...");

        alloc.put(0, small).unwrap();
        alloc.put(0, huge).unwrap();

        warn!("realloc...");

        // Realloc
        for page in &mut pages[10..PT_LEN + 10] {
            *page = alloc.get(0, Size::L0).unwrap();
        }

        warn!("free...");

        // Free all
        for page in &pages {
            alloc.put(0, *page).unwrap();
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
            a.init(THREADS, &mut mapping, true).unwrap();
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
                *dst = a.get(t, Size::L0).unwrap();
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
    fn parallel_huge_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN - 1;
        const PAGES: usize = THREADS * MIN_PAGES;

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(THREADS, &mut mapping, true).unwrap();
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
                *dst = a.get(t, Size::L1).unwrap();
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
            a.init(THREADS, &mut mapping, true).unwrap();
            a
        });

        // Stress test
        let barrier = Arc::new(Barrier::new(THREADS));

        let a = alloc.clone();
        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            let mut pages = vec![0; ALLOC_PER_THREAD];

            for page in &mut pages {
                *page = alloc.get(t, Size::L0).unwrap();
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut pages);

            for page in pages {
                alloc.put(t, page).unwrap();
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
            a.init(THREADS, &mut mapping, true).unwrap();
            a
        });

        let barrier = Arc::new(Barrier::new(THREADS));

        // Alloc on first thread
        thread::pin(0);
        let mut pages = vec![0; ALLOC_PER_THREAD];
        for page in &mut pages {
            *page = alloc.get(0, Size::L0).unwrap();
        }

        let handle = {
            let barrier = barrier.clone();
            let alloc = alloc.clone();
            std::thread::spawn(move || {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for page in &pages {
                    alloc.put(1, *page).unwrap();
                }
            })
        };

        let mut pages = vec![0; ALLOC_PER_THREAD];

        barrier.wait();

        // Simultaneously alloc on first thread
        for page in &mut pages {
            *page = alloc.get(0, Size::L0).unwrap();
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

        let expected_pages = (PT_LEN + 2) * (Size::L0.span() + Size::L1.span());

        {
            let alloc = Arc::new({
                let mut a = Allocator::default();
                a.init(1, &mut mapping, true).unwrap();
                a
            });

            for _ in 0..PT_LEN + 2 {
                alloc.get(0, Size::L0).unwrap();
                alloc.get(0, Size::L1).unwrap();
            }

            assert_eq!(alloc.dbg_allocated_pages(), expected_pages);

            // leak
            let _ = Arc::into_raw(alloc);
        }

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, false).unwrap();
            a
        });

        assert_eq!(alloc.dbg_allocated_pages(), expected_pages);
    }
}
