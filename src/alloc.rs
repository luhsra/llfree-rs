use core::any::type_name;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use log::error;

use crate::table::Table;
use crate::util::Page;

pub mod array_aligned;
pub mod array_atomic;
pub mod array_locked;
pub mod array_unaligned;
pub mod list_local;
pub mod list_locked;
pub mod malloc;
pub mod table;

pub const MAGIC: usize = 0xdeadbeef;
pub const MIN_PAGES: usize = 2 * Table::span(2);
pub const MAX_PAGES: usize = Table::span(Table::LAYERS);

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
    /// 1GiB
    L2 = 2,
}

pub trait Alloc: Sync + Send + fmt::Debug {
    /// Initialize the allocator.
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()>;

    /// Allocate a new page.
    fn get(&self, core: usize, size: Size) -> Result<u64>;
    /// Free the given page.
    fn put(&self, core: usize, addr: u64) -> Result<()>;

    /// Return the number of pages that can be allocated.
    fn pages(&self) -> usize;
    /// Return the number of allocated pages.
    #[cold]
    fn dbg_allocated_pages(&self) -> usize;
    #[cold]
    fn name(&self) -> String {
        let name = type_name::<Self>();
        let (name, suffix) = if let Some((prefix, suffix)) = name.split_once('<') {
            let suffix = suffix.rsplit_once('>').map(|s| s.0).unwrap_or(suffix);
            let suffix = suffix.rsplit_once(':').map(|s| s.1).unwrap_or(suffix);
            (prefix, &suffix[0..1])
        } else {
            (name, "")
        };
        let name = name.rsplit_once(':').map(|s| s.1).unwrap_or(name);
        format!("{}{}", name.strip_suffix("Alloc").unwrap_or(name), suffix)
    }
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
    match dst.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => Ok(()),
        Err(_) => {
            error!("CAS");
            alloc.put(core, page).unwrap();
            Err(Error::CAS)
        }
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
    use crate::alloc::Alloc;
    use crate::alloc::MIN_PAGES;
    use crate::lower::fixed::FixedLower;
    use crate::mmap::MMap;
    use crate::table::Table;
    use crate::util::{logging, Page, WyRand};
    use crate::{thread, Size};

    type Allocator = super::array_atomic::ArrayAtomicAlloc<FixedLower>;

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

        assert_eq!(alloc.dbg_allocated_pages(), 1);

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
            info!("addr {i}={p1:x}");
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        for page in &pages[..Table::span(2) - 10] {
            alloc.put(0, *page).unwrap();
        }

        assert_eq!(
            alloc.dbg_allocated_pages(),
            1 + pages.len() - Table::span(2) + 10
        );

        // Realloc
        for page in &mut pages[..Table::span(2) - 10] {
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
            info!("addr {i}={p1:x}");
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
            info!("addr {i}={p1:x}");
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
        let giant = alloc.get(0, Size::L2).unwrap();

        assert_eq!(alloc.dbg_allocated_pages(), 1 + Table::LEN + Table::span(2));
        assert!(small != huge && small != giant && huge != giant);

        warn!("start stress test");

        // Stress test
        let mut pages = vec![0; Table::LEN * Table::LEN];
        for page in &mut pages {
            *page = alloc.get(0, Size::L0).unwrap();
        }

        warn!("check");

        assert_eq!(
            alloc.dbg_allocated_pages(),
            1 + Table::LEN + Table::span(2) + pages.len()
        );

        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            info!("addr {i}={p1:x}");
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("free some...");

        // Free some
        for page in &pages[10..Table::LEN + 10] {
            alloc.put(0, *page).unwrap();
        }

        warn!("free special...");

        alloc.put(0, small).unwrap();
        alloc.put(0, huge).unwrap();
        alloc.put(0, giant).unwrap();

        warn!("realloc...");

        // Realloc
        for page in &mut pages[10..Table::LEN + 10] {
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
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 2 * THREADS);
        const PAGES: usize = 2 * THREADS * Table::span(2);

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
        const ALLOC_PER_THREAD: usize = Table::LEN - 1;
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

        assert_eq!(alloc.dbg_allocated_pages(), pages.len() * Table::span(1));
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
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 2 * THREADS);

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
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 2 * THREADS);

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
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * Table::m_span(2);

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
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 10) / 2;

        let mut mapping = mapping(0x1000_0000_0000, 4 * Table::span(2)).unwrap();

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

        {
            let alloc = Arc::new({
                let mut a = Allocator::default();
                a.init(1, &mut mapping, true).unwrap();
                a
            });

            for _ in 0..Table::LEN + 2 {
                alloc.get(0, Size::L0).unwrap();
                alloc.get(0, Size::L1).unwrap();
            }

            alloc.get(0, Size::L2).unwrap();

            assert_eq!(
                alloc.dbg_allocated_pages(),
                Table::span(2) + Table::LEN + 2 + Table::LEN * (Table::LEN + 2)
            );

            // leak
            let _ = Arc::into_raw(alloc);
        }

        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(1, &mut mapping, false).unwrap();
            a
        });

        assert_eq!(
            alloc.dbg_allocated_pages(),
            Table::span(2) + Table::LEN + 2 + Table::LEN * (Table::LEN + 2)
        );
    }
}
