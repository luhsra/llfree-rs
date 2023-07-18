//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.

#![no_std]
#![feature(new_uninit)]
#![feature(int_roundings)]
#![feature(array_windows)]
#![feature(generic_const_exprs)]
#![feature(inline_const)]
#![feature(allocator_api)]
#![feature(let_chains)]
// Don't warn for compile-time checks
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::redundant_pattern_matching)]

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
pub mod mmap;
#[cfg(feature = "std")]
pub mod thread;

pub mod atomic;
pub mod frame;
pub use llfree::LLFree;
pub mod util;

mod bitfield;
mod entry;
mod llfree;
mod lower;

use core::ffi::c_void;
use core::fmt;
use core::ops::Range;

use frame::PFN;

/// Allocation error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory = 1,
    /// Failed atomic operation, retry procedure
    Retry = 2,
    /// Invalid address
    Address = 3,
    /// Allocator not initialized or initialization failed
    Initialization = 4,
    /// Corrupted allocator state
    Corruption = 5,
}

/// Allocation result
pub type Result<T> = core::result::Result<T, Error>;

/// Number of retries if an atomic operation fails.
pub const CAS_RETRIES: usize = 4;

/// The general interface of the allocator implementations.
pub trait Alloc: Sync + Send + fmt::Debug {
    /// Return the name of the allocator.
    #[cold]
    fn name(&self) -> &'static str {
        "Unknown"
    }

    /// Initialize the allocator.
    #[cold]
    fn init(&mut self, cores: usize, area: Range<PFN>, init: Init, free_all: bool) -> Result<()>;

    /// Allocate a new frame of `order` on the given `core`.
    fn get(&self, core: usize, order: usize) -> Result<PFN>;
    /// Free the `frame` of `order` on the given `core`..
    fn put(&self, core: usize, frame: PFN, order: usize) -> Result<()>;
    /// Returns if `frame` is free. This might be racy!
    fn is_free(&self, frame: PFN, order: usize) -> bool;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;

    /// Unreserve cpu-local frames
    fn drain(&self, _core: usize) -> Result<()> {
        Ok(())
    }

    /// Return the number of allocated frames.
    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }
    /// Return the number of free frames.
    fn free_frames(&self) -> usize;
    /// Return the number of free huge frames or 0 if the allocator cannot allocate huge frames.
    fn free_huge_frames(&self) -> usize {
        0
    }
    /// Execute f for each huge frame with the number of free frames
    /// in this huge frame as parameter.
    fn for_each_huge_frame(&self, _ctx: *mut c_void, _f: fn(*mut c_void, PFN, usize)) {}
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

/// Extending the dynamic [Alloc] interface
pub trait AllocExt: Sized + Alloc + Default {
    /// Create and initialize the allocator.
    #[cold]
    fn new(cores: usize, area: Range<PFN>, init: Init, free_all: bool) -> Result<Self> {
        let mut a = Self::default();
        a.init(cores, area, init, free_all)?;
        Ok(a)
    }
    /// Calls f for every huge page
    fn each_huge_frame<F: FnMut(PFN, usize)>(&self, mut f: F) {
        self.for_each_huge_frame((&mut f) as *mut F as *mut c_void, |ctx, pfn, free| {
            let f = unsafe { &mut *ctx.cast::<F>() };
            f(pfn, free)
        })
    }
}
// Implement for all default initializable allocators
impl<A: Sized + Alloc + Default> AllocExt for A {}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::ptr::null_mut;
    use std::time::Instant;

    use alloc::vec::Vec;
    use std::sync::Barrier;

    use log::{info, warn};

    use crate::frame::{pfn_range, Frame, PFNRange, PFN, PT_LEN};
    use crate::lower::Lower;
    use crate::mmap::test_mapping;
    use crate::thread;
    use crate::util::{logging, WyRand};
    use crate::Error;
    use crate::{Alloc, AllocExt, Init};

    use super::LLFree;

    type Allocator = LLFree;

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        assert_eq!(alloc.free_frames(), alloc.frames());

        warn!("start alloc...");
        let small = alloc.get(0, 0).unwrap();

        assert_eq!(alloc.allocated_frames(), 1, "{alloc:?}");
        warn!("stress test...");

        // Stress test
        let mut frames = Vec::new();
        loop {
            match alloc.get(0, 0) {
                Ok(frame) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + frames.len());
        warn!("check...");

        assert_eq!(alloc.allocated_frames(), 1 + frames.len());
        assert_eq!(alloc.allocated_frames(), alloc.frames());
        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = PT_LEN * PT_LEN - 10;
        for frame in &frames[..FREE_NUM] {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(
            alloc.allocated_frames(),
            1 + frames.len() - FREE_NUM,
            "{alloc:?}"
        );

        // Realloc
        for frame in &mut frames[..FREE_NUM] {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        alloc.put(0, small, 0).unwrap();
        // Free all
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn rand() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 4 << 30;
        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        warn!("start alloc...");
        const ALLOCS: usize = MEM_SIZE / Frame::SIZE / 4 * 3;
        let mut frames = Vec::with_capacity(ALLOCS);
        for _ in 0..ALLOCS {
            frames.push(alloc.get(0, 0).unwrap());
        }
        warn!("allocated {}", frames.len());

        warn!("check...");
        assert_eq!(alloc.allocated_frames(), frames.len());
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("reallocate rand...");
        let mut rng = WyRand::new(100);
        rng.shuffle(&mut frames);

        for _ in 0..frames.len() {
            let i = rng.range(0..frames.len() as _) as usize;
            alloc.put(0, frames[i], 0).unwrap();
            frames[i] = alloc.get(0, 0).unwrap();
        }

        warn!("check...");
        assert_eq!(alloc.allocated_frames(), frames.len());
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("free...");
        rng.shuffle(&mut frames);
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }
        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn multirand() {
        const THREADS: usize = 4;
        const MEM_SIZE: usize = (8 << 30) / Frame::SIZE;
        const ALLOCS: usize = ((MEM_SIZE / THREADS) / 4) * 3;

        logging();
        let area = PFN(0)..PFN(MEM_SIZE);
        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        let barrier = Barrier::new(THREADS);
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);

            barrier.wait();
            warn!("start alloc...");
            let mut frames = Vec::with_capacity(ALLOCS);
            for _ in 0..ALLOCS {
                frames.push(alloc.get(t, 0).unwrap());
            }
            warn!("allocated {}", frames.len());

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for &[a, b] in frames.array_windows() {
                assert_ne!(a, b);
                assert!(area.contains(&a) && area.contains(&b));
            }

            barrier.wait();
            warn!("reallocate rand...");
            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for _ in 0..frames.len() {
                let i = rng.range(0..frames.len() as _) as usize;
                alloc.put(t, frames[i], 0).unwrap();
                frames[i] = alloc.get(t, 0).unwrap();
            }

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for &[a, b] in frames.array_windows() {
                assert_ne!(a, b);
                assert!(area.contains(&a) && area.contains(&b));
            }

            warn!("free...");
            rng.shuffle(&mut frames);
            for frame in &frames {
                alloc.put(t, *frame, 0).unwrap();
            }
        });

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        assert_eq!(alloc.allocated_frames(), 0);

        warn!("start alloc");
        let small = alloc.get(0, 0).unwrap();
        let huge = alloc.get(0, 9).unwrap();

        let expected_frames = 1 + (1 << 9);
        assert_eq!(alloc.allocated_frames(), expected_frames);
        assert!(small != huge);

        warn!("start stress test");

        // Stress test
        let mut frames = vec![PFN(0); PT_LEN * PT_LEN];
        for frame in &mut frames {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("check");

        assert_eq!(alloc.allocated_frames(), expected_frames + frames.len());

        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("free some...");

        // Free some
        for frame in &frames[10..PT_LEN + 10] {
            alloc.put(0, *frame, 0).unwrap();
        }

        warn!("free special...");

        alloc.put(0, small, 0).unwrap();
        alloc.put(0, huge, 9).unwrap();

        warn!("realloc...");

        // Realloc
        for frame in &mut frames[10..PT_LEN + 10] {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        // Free all
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn parallel_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const PAGES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        // Stress test
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, 0).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.allocated_frames(), frames.len());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }
    }

    #[test]
    #[ignore]
    fn less_mem() {
        logging();

        const THREADS: usize = 4;
        const PAGES: usize = 4096;
        const ALLOC_PER_THREAD: usize = PAGES / THREADS - THREADS;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        // Stress test
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, 0).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.allocated_frames(), frames.len());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        thread::parallel(
            frames.chunks(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    alloc.put(t, *frame, 0).unwrap();
                }
            },
        );

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn parallel_huge_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN - 1;
        // additional space for the allocators metadata
        const PAGES: usize = THREADS * (ALLOC_PER_THREAD + 2) * PT_LEN;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        // Stress test
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, 9).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.allocated_frames(), frames.len() * PT_LEN);
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }
    }

    #[ignore]
    #[test]
    fn parallel_malloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut frames = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = unsafe { libc::malloc(Frame::SIZE) } as u64;
                    assert!(*frame != 0);
                }
            },
        );

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        let mut last = None;
        for p in frames {
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
        let mut frames = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = unsafe {
                        libc::mmap(
                            null_mut(),
                            Frame::SIZE,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                            -1,
                            0,
                        )
                    } as u64;
                    assert!(*frame != 0);
                }
            },
        );

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        let mut last = None;
        for p in frames {
            assert!(last != Some(p));
            unsafe { libc::munmap(p as _, Frame::SIZE) };
            last = Some(p);
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * PT_LEN * PT_LEN * Frame::SIZE;

        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        let alloc = Allocator::new(THREADS, area, Init::Volatile, true).unwrap();
        let barrier = Barrier::new(THREADS);

        // Stress test
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            barrier.wait();

            let mut frames = vec![PFN(0); ALLOC_PER_THREAD];

            for frame in &mut frames {
                *frame = alloc.get(t, 0).unwrap();
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for frame in frames {
                alloc.put(t, frame, 0).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10) / 2;

        let area = PFN(0)..PFN(4 * PT_LEN * PT_LEN);

        let alloc = Allocator::new(THREADS, area, Init::Volatile, true).unwrap();

        // Alloc on first thread
        thread::pin(0);
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD];
        for frame in &mut frames {
            *frame = alloc.get(0, 0).unwrap();
        }

        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            s.spawn(|| {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for frame in &frames {
                    alloc.put(1, *frame, 0).unwrap();
                }
            });

            let mut frames = vec![PFN(0); ALLOC_PER_THREAD];

            barrier.wait();

            // Simultaneously alloc on first thread
            for frame in &mut frames {
                *frame = alloc.get(0, 0).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.allocated_frames(), ALLOC_PER_THREAD);
    }

    #[test]
    fn recover() {
        logging();

        let mapping = test_mapping(0x1000_0000_0000, 8 << 18);
        let area = pfn_range(&mapping);
        warn!("{area:?}");
        thread::pin(0);

        let expected_frames = (PT_LEN + 2) * (1 + (1 << 9));

        {
            let alloc = Allocator::new(1, area.clone(), Init::Overwrite, true).unwrap();

            for _ in 0..PT_LEN + 2 {
                alloc.get(0, 0).unwrap();
                alloc.get(0, 9).unwrap();
            }

            assert_eq!(alloc.allocated_frames(), expected_frames);

            // leak (crash)
            std::mem::forget(alloc);
        }

        let mut alloc = Allocator::default();
        alloc.init(1, area, Init::Recover, true).unwrap();
        assert_eq!(alloc.allocated_frames(), expected_frames);
    }

    #[test]
    fn different_orders() {
        const MAX_ORDER: usize = Lower::MAX_ORDER;
        const THREADS: usize = 4;

        logging();

        let area = PFN(0)..PFN(Lower::N * (THREADS * 2 + 1));
        let alloc = Allocator::new(THREADS, area, Init::Volatile, true).unwrap();

        let barrier = Barrier::new(THREADS);

        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut rng = WyRand::new(42 + t as u64);
            let mut num_frames = 0;
            let mut frames = Vec::new();
            for order in 0..=MAX_ORDER {
                for _ in 0..1 << (MAX_ORDER - order) {
                    frames.push((order, PFN(0)));
                    num_frames += 1 << order;
                }
            }
            rng.shuffle(&mut frames);

            warn!("allocate {num_frames} frames up to order <{MAX_ORDER}");
            barrier.wait();

            for (order, frame) in &mut frames {
                *frame = match alloc.get(t, *order) {
                    Ok(frame) => frame,
                    Err(e) => panic!("{e:?} o={order} {alloc:?}"),
                };
                assert!(frame.0 % (1 << *order) == 0, "{frame} {:x}", 1 << *order);
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for (order, frame) in frames {
                match alloc.put(t, frame, order) {
                    Ok(_) => {}
                    Err(e) => panic!("{e:?} o={order} {alloc:?}"),
                }
            }
        });

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn init_reserved() {
        logging();

        const THREADS: usize = 2;
        const PAGES: usize = 8 << 18;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, false).unwrap();
        assert_eq!(alloc.frames(), PAGES);
        assert_eq!(alloc.allocated_frames(), PAGES);

        for frame in area.as_range().step_by(1 << Lower::HUGE_ORDER).map(PFN) {
            alloc.put(0, frame, Lower::HUGE_ORDER).unwrap();
        }
        assert_eq!(alloc.allocated_frames(), 0);

        thread::parallel(0..THREADS, |core| {
            thread::pin(core);
            for _ in 0..(PAGES / THREADS) / (1 << Lower::HUGE_ORDER) {
                alloc.get(core, Lower::HUGE_ORDER).unwrap();
            }
        });
        assert_eq!(alloc.allocated_frames(), PAGES);
    }

    #[test]
    fn fragmentation_retry() {
        logging();

        let area = PFN(0)..PFN(Lower::N * 2);
        let alloc = Allocator::new(1, area, Init::Volatile, true).unwrap();

        // Alloc a whole subtree
        let mut frames = Vec::with_capacity(Lower::N / 2);
        for i in 0..Lower::N {
            if i % 2 == 0 {
                frames.push(alloc.get(0, 0).unwrap());
            } else {
                alloc.get(0, 0).unwrap();
            }
        }
        // Free every second one -> fragmentation
        for frame in frames {
            alloc.put(0, frame, 0).unwrap();
        }

        let huge = alloc.get(0, 9).unwrap();
        warn!("huge = {huge}");
        warn!("{alloc:?}");
    }

    #[test]
    fn drain() {
        let area = PFN(0)..PFN(Lower::N * 2);
        let alloc = Allocator::new(2, area, Init::Volatile, true).unwrap();
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
