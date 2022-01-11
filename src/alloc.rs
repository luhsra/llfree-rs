use std::sync::atomic::{AtomicU64, Ordering};

use log::error;

use crate::table::Table;
use crate::util::Page;

pub mod buddy;
pub mod local_lists;
pub mod malloc;
pub mod stack;
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
    /// Allocator not initialized
    Uninitialized = 4,
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

pub type Allocator = table::TableAlloc;

pub trait Alloc {
    /// Initialize the allocator.
    fn init(cores: usize, memory: &mut [Page]) -> Result<()>;
    /// Uninitialize the allocator. The persistent data remains.
    fn uninit();
    /// Clear the persistent memory pool.
    fn destroy();

    /// Return the initialized allocator instance (or panic if it is not initialized)
    fn instance<'a>() -> &'a Self;

    /// Allocate a new page.
    fn get(&self, core: usize, size: Size) -> Result<u64>;
    /// Allocates a new page and writes the value after translation into `dst`.
    /// If `dst` has an other value than `expected` the operation is revoked and `Error::CAS` is returned.
    fn get_cas<F: FnOnce(u64) -> u64>(
        &self,
        core: usize,
        size: Size,
        dst: &AtomicU64,
        translate: F,
        expected: u64,
    ) -> Result<()> {
        let page = self.get(core, size)?;
        let new = translate(page);
        match dst.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => {
                error!("CAS");
                self.put(core, page).unwrap();
                Err(Error::CAS)
            }
        }
    }
    /// Free the given page.
    fn put(&self, core: usize, addr: u64) -> Result<()>;

    /// Return the number of allocated pages.
    fn allocated_pages(&self) -> usize;
}

#[cfg(test)]
mod test {

    use std::ptr::null_mut;
    use std::sync::{Arc, Barrier};
    use std::time::Instant;

    use log::{info, warn};

    use super::{Alloc, Allocator, Error};
    use crate::alloc::MIN_PAGES;
    use crate::mmap::MMap;
    use crate::table::Table;
    use crate::util::{logging, Page};
    use crate::{thread, Size};

    fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<'a, Page>, ()> {
        #[cfg(target_os = "linux")]
        if let Ok(file) = std::env::var("NVM_FILE") {
            warn!("MMap file {} l={}G", file, (length * Page::SIZE) >> 30);
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

        info!("mmap {} bytes at {:?}", MEM_SIZE, mapping.as_ptr());

        info!("init alloc");

        Allocator::init(1, &mut mapping).unwrap();

        warn!("start alloc...");
        let small = Allocator::instance().get(0, Size::L0).unwrap();

        assert_eq!(Allocator::instance().allocated_pages(), 1);

        // Stress test
        let mut pages = Vec::new();
        loop {
            match Allocator::instance().get(0, Size::L0) {
                Ok(page) => pages.push(page),
                Err(Error::Memory) => break,
                Err(e) => panic!("{:?}", e),
            }
        }

        warn!("allocated {}", pages.len());
        warn!("check...");

        assert_eq!(Allocator::instance().allocated_pages(), 1 + pages.len());
        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            info!("addr {}={:x}", i, p1);
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        for page in &pages[..Table::span(2) - 10] {
            Allocator::instance().put(0, *page).unwrap();
        }

        assert_eq!(
            Allocator::instance().allocated_pages(),
            1 + pages.len() - Table::span(2) + 10
        );

        // Realloc
        for page in &mut pages[..Table::span(2) - 10] {
            *page = Allocator::instance().get(0, Size::L0).unwrap();
        }

        warn!("free...");

        Allocator::instance().put(0, small).unwrap();
        // Free all
        for page in &pages {
            Allocator::instance().put(0, *page).unwrap();
        }

        assert_eq!(Allocator::instance().allocated_pages(), 0);

        Allocator::uninit();
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        info!("mmap {} bytes at {:?}", MEM_SIZE, mapping.as_ptr());

        info!("init alloc");

        Allocator::init(1, &mut mapping).unwrap();

        warn!("start alloc...");
        let small = Allocator::instance().get(0, Size::L0).unwrap();
        let huge = Allocator::instance().get(0, Size::L1).unwrap();
        let giant = Allocator::instance().get(0, Size::L2).unwrap();

        assert_eq!(
            Allocator::instance().allocated_pages(),
            1 + Table::LEN + Table::span(2)
        );
        assert!(small != huge && small != giant && huge != giant);

        // Stress test
        let mut pages = vec![0; Table::LEN * Table::LEN];
        for page in &mut pages {
            *page = Allocator::instance().get(0, Size::L0).unwrap();
        }

        warn!("check...");

        assert_eq!(
            Allocator::instance().allocated_pages(),
            1 + Table::LEN + Table::span(2) + pages.len()
        );

        pages.sort_unstable();

        // Check that the same page was not allocated twice
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            info!("addr {}={:x}", i, p1);
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        warn!("realloc...");

        // Free some
        for page in &pages[10..Table::LEN + 10] {
            Allocator::instance().put(0, *page).unwrap();
        }

        Allocator::instance().put(0, small).unwrap();
        Allocator::instance().put(0, huge).unwrap();
        Allocator::instance().put(0, giant).unwrap();

        // Realloc
        for page in &mut pages[10..Table::LEN + 10] {
            *page = Allocator::instance().get(0, Size::L0).unwrap();
        }

        warn!("free...");

        // Free all
        for page in &pages {
            Allocator::instance().put(0, *page).unwrap();
        }

        Allocator::uninit();
    }

    #[test]
    fn parallel_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 2 * THREADS);
        const PAGES: usize = 2 * THREADS * Table::span(2);

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        info!("init alloc");
        Allocator::init(THREADS, &mut mapping).unwrap();

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;
        let timer = Instant::now();

        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = Allocator::instance().get(t, Size::L0).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(Allocator::instance().allocated_pages(), pages.len());
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        Allocator::uninit();
    }

    #[test]
    fn parallel_huge_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = Table::LEN - 1;
        const PAGES: usize = THREADS * MIN_PAGES;

        let mut mapping = mapping(0x1000_0000_0000, PAGES).unwrap();

        info!("init alloc");
        Allocator::init(THREADS, &mut mapping).unwrap();

        // Stress test
        let mut pages = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Arc::new(Barrier::new(THREADS));
        let pages_begin = pages.as_ptr() as usize;
        let timer = Instant::now();

        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            for i in 0..ALLOC_PER_THREAD {
                let dst = unsafe { &mut *(pages_begin as *mut u64).add(t * ALLOC_PER_THREAD + i) };
                *dst = Allocator::instance().get(t, Size::L1).unwrap();
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            Allocator::instance().allocated_pages(),
            pages.len() * Table::span(1)
        );
        warn!("allocated pages: {}", pages.len());

        // Check that the same page was not allocated twice
        pages.sort_unstable();
        for i in 0..pages.len() - 1 {
            let p1 = pages[i];
            let p2 = pages[i + 1];
            assert!(mapping.as_ptr_range().contains(&(p1 as _)));
            assert!(p1 != p2);
        }

        Allocator::uninit();
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

        const THREADS: usize = 8;
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * Table::m_span(2);

        let mut mapping = mapping(0x1000_0000_0000, MEM_SIZE / Page::SIZE).unwrap();

        Allocator::init(THREADS, &mut mapping).unwrap();

        // Stress test
        let barrier = Arc::new(Barrier::new(THREADS));

        thread::parallel(THREADS as _, move |t| {
            thread::pin(t);
            barrier.wait();

            let mut pages = vec![0; ALLOC_PER_THREAD];

            for page in &mut pages {
                *page = Allocator::instance().get(t, Size::L0).unwrap();
            }

            for page in pages {
                Allocator::instance().put(t, page).unwrap();
            }
        });

        assert_eq!(Allocator::instance().allocated_pages(), 0);
        Allocator::uninit();
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = Table::LEN * (Table::LEN - 10) / 2;

        let mut mapping = mapping(0x1000_0000_0000, 4 * Table::span(2)).unwrap();

        Allocator::init(THREADS, &mut mapping).unwrap();

        let barrier = Arc::new(Barrier::new(THREADS));

        // Alloc on first thread
        thread::pin(0);
        let mut pages = vec![0; ALLOC_PER_THREAD];
        for page in &mut pages {
            *page = Allocator::instance().get(0, Size::L0).unwrap();
        }

        let handle = {
            let barrier = barrier.clone();

            std::thread::spawn(move || {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for page in &pages {
                    Allocator::instance().put(1, *page).unwrap();
                }
            })
        };

        let mut pages = vec![0; ALLOC_PER_THREAD];

        barrier.wait();

        // Simultaneously alloc on first thread
        for page in &mut pages {
            *page = Allocator::instance().get(0, Size::L0).unwrap();
        }

        handle.join().unwrap();

        assert_eq!(Allocator::instance().allocated_pages(), ALLOC_PER_THREAD);

        Allocator::uninit();
    }

    #[test]
    fn recover() {
        logging();

        let mut mapping = mapping(0x1000_0000_0000, 8 << 18).unwrap();
        thread::pin(0);

        {
            Allocator::init(1, &mut mapping).unwrap();

            for _ in 0..Table::LEN + 2 {
                Allocator::instance().get(0, Size::L0).unwrap();
                Allocator::instance().get(0, Size::L1).unwrap();
            }

            Allocator::instance().get(0, Size::L2).unwrap();

            assert_eq!(
                Allocator::instance().allocated_pages(),
                Table::span(2) + Table::LEN + 2 + Table::LEN * (Table::LEN + 2)
            );
            Allocator::uninit();
        }

        Allocator::init(1, &mut mapping).unwrap();

        assert_eq!(
            Allocator::instance().allocated_pages(),
            Table::span(2) + Table::LEN + 2 + Table::LEN * (Table::LEN + 2)
        );
        Allocator::uninit();
    }
}
