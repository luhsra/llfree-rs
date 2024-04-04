//! # Persistent non-volatile memory allocator
//!
//! This project contains multiple allocator designs for NVM and benchmarks comparing them.

#![no_std]
#![cfg_attr(feature = "std", feature(new_uninit))]
#![feature(int_roundings)]
#![feature(array_windows)]
#![feature(inline_const)]
#![feature(allocator_api)]
#![feature(c_size_t)]
#![feature(let_chains)]
#![feature(pointer_is_aligned_to)]
// Don't warn for compile-time checks
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::redundant_pattern_matching)]

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[cfg(feature = "std")]
pub mod mmap;
#[cfg(feature = "std")]
pub mod thread;

pub mod atomic;
pub mod frame;
pub mod util;
pub mod wrapper;

mod bitfield;
mod entry;
mod llfree;
pub use llfree::LLFree;

#[cfg(feature = "llc")]
mod llc;
#[cfg(feature = "llc")]
pub use llc::LLC;

mod lower;
mod trees;

use core::fmt;

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
}

/// Allocation result
pub type Result<T> = core::result::Result<T, Error>;

/// Number of retries if an atomic operation fails.
pub const CAS_RETRIES: usize = 8;

/// The general interface of the allocator implementations.
pub trait Alloc<'a>: Sized + Sync + Send + fmt::Debug {
    /// Maximum allocation size order.
    const MAX_ORDER: usize;
    /// Maximum allocation size order.
    const HUGE_ORDER: usize;

    /// Return the name of the allocator.
    #[cold]
    fn name() -> &'static str;

    /// Initialize the allocator.
    ///
    /// The metadata is stored into the primary (optionally persistant) and secondary buffers.
    #[cold]
    fn new(
        cores: usize,
        frames: usize,
        init: Init,
        primary: &'a mut [u8],
        secondary: &'a mut [u8],
    ) -> Result<Self>;

    /// Returns the size of the metadata buffers required for initialization.
    #[cold]
    fn metadata_size(cores: usize, frames: usize) -> MetaSize;
    /// Returns the metadata buffers.
    #[cold]
    fn metadata(&mut self) -> (&'a mut [u8], &'a mut [u8]);

    /// Allocate a new frame of `order` on the given `core`.
    fn get(&self, core: usize, order: usize) -> Result<usize>;
    /// Free the `frame` of `order` on the given `core`..
    fn put(&self, core: usize, frame: usize, order: usize) -> Result<()>;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;
    /// Return the core count the allocator was initialized with.
    fn cores(&self) -> usize;

    /// Return the number of free frames.
    fn free_frames(&self) -> usize;
    /// Return the number of free huge frames or 0 if the allocator cannot allocate huge frames.
    fn free_huge_frames(&self) -> usize {
        0
    }

    /// Returns if `frame` is free. This might be racy!
    fn is_free(&self, frame: usize, order: usize) -> bool;
    /// Free frames in the given chunk. Only TREE_ORDER and HUGE_ORDER are supported.
    fn free_at(&self, frame: usize, order: usize) -> usize;

    /// Return the number of allocated frames.
    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }
    /// Unreserve cpu-local frames
    fn drain(&self, _core: usize) -> Result<()> {
        Ok(())
    }
}

/// Size of the required metadata
pub struct MetaSize {
    /// Size of the optionally persistent data.
    pub primary: usize,
    /// Size of the volatile data.
    pub secondary: usize,
}

/// Defines if the allocator should be allocated persistently
/// and if it in that case should try to recover from the persistent memory.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Init {
    /// Clear the allocator marking all frames as free
    FreeAll,
    /// Clear the allocator marking all frames as allocated
    AllocAll,
    /// Try recovering all frames from persistent memory
    Recover(bool),
}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::mem::ManuallyDrop;
    use core::ops::Deref;
    use core::ptr::null_mut;
    use std::sync::Barrier;
    use std::time::Instant;
    use std::vec::Vec;

    use log::{error, warn};

    use super::*;
    use crate::frame::{Frame, PT_LEN};
    use crate::lower::Lower;
    use crate::util::{aligned_buf, logging, WyRand};
    use crate::wrapper::NvmAlloc;

    #[cfg(feature = "llc")]
    type Allocator = TestAlloc<LLC>;
    #[cfg(not(feature = "llc"))]
    type Allocator = TestAlloc<LLFree<'static>>;

    pub struct TestAlloc<A: Alloc<'static>>(ManuallyDrop<A>);

    impl<A: Alloc<'static>> TestAlloc<A> {
        pub fn create(cores: usize, frames: usize, init: Init) -> Result<Self> {
            let MetaSize { primary, secondary } = A::metadata_size(cores, frames);
            let primary = aligned_buf(primary).leak();
            let secondary = aligned_buf(secondary).leak();
            Ok(Self(ManuallyDrop::new(A::new(
                cores, frames, init, primary, secondary,
            )?)))
        }
    }
    impl<A: Alloc<'static>> Drop for TestAlloc<A> {
        fn drop(&mut self) {
            let (primary, secondary) = self.0.metadata();
            unsafe {
                // drop first
                drop(ManuallyDrop::take(&mut self.0));
                // free metadata buffers
                Vec::from_raw_parts(primary.as_mut_ptr(), primary.len(), primary.len());
                Vec::from_raw_parts(secondary.as_mut_ptr(), secondary.len(), secondary.len());
            }
        }
    }
    impl<A: Alloc<'static>> Deref for TestAlloc<A> {
        type Target = A;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<A: Alloc<'static>> fmt::Debug for TestAlloc<A> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            A::fmt(&self, f)
        }
    }

    #[test]
    fn minimal() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        let frames = MEM_SIZE / Frame::SIZE;

        warn!("init");
        let alloc = Allocator::create(1, frames, Init::FreeAll).unwrap();
        warn!("finit");

        assert_eq!(alloc.free_frames(), alloc.frames());

        warn!("get >>>");
        let frame1 = alloc.get(0, 0).unwrap();
        warn!("get <<<");
        warn!("get >>>");
        let frame2 = alloc.get(0, 0).unwrap();
        warn!("get <<<");

        warn!("put >>>");
        alloc.put(0, frame2, 0).unwrap();
        warn!("put <<<");
        warn!("put >>>");
        alloc.put(0, frame1, 0).unwrap();
        warn!("put <<<");
    }

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        const FRAMES: usize = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

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
            assert!(a < FRAMES && b < FRAMES);
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = FRAMES / 3;
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
        const FRAMES: usize = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        warn!("start alloc...");
        const ALLOCS: usize = MEM_SIZE / Frame::SIZE / 2;
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
            assert!(a < FRAMES && b < FRAMES);
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
            assert!(a < FRAMES && b < FRAMES);
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
        const FRAMES: usize = (8 << 30) / Frame::SIZE;
        const ALLOCS: usize = ((FRAMES / THREADS) / 4) * 3;

        logging();

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

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
                assert!(a < FRAMES && b < FRAMES);
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
                assert!(a < FRAMES && b < FRAMES);
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
        const FRAMES: usize = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        assert_eq!(alloc.allocated_frames(), 0);

        warn!("start alloc");
        let small = alloc.get(0, 0).unwrap();
        let huge = alloc.get(0, 9).unwrap();

        let expected_frames = 1 + (1 << 9);
        assert_eq!(alloc.allocated_frames(), expected_frames);
        assert!(small != huge);

        warn!("start stress test");

        // Stress test
        //TODO: try and really allocate ALL pages
        let mut frames = vec![0; FRAMES / 2]; // 0 to #frames that are accessible (- prev n this trest allocated pages)
        for frame in &mut frames {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("check");

        assert_eq!(alloc.allocated_frames(), expected_frames + frames.len());

        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
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
        const FRAMES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = vec![0; ALLOC_PER_THREAD * THREADS];
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
            assert!(a < FRAMES && b < FRAMES);
        }
    }

    #[test]
    fn alloc_all() {
        logging();

        //create 2 GiB memory
        const FRAMES: usize = (2 * (1 << 30)) / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = Vec::new();
        let timer = Instant::now();

        loop {
            match alloc.get(0, 0) {
                Ok(frame) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("{alloc:?}");

        assert_eq!(alloc.allocated_frames(), frames.len());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }
    }

    #[test]
    fn parallel_alloc_all() {
        logging();

        const THREADS: usize = 4;
        const FRAMES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = vec![Vec::new(); THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(frames.iter_mut().enumerate(), |(t, frames)| {
            thread::pin(t);
            barrier.wait();

            loop {
                match alloc.get(t, 0) {
                    Ok(frame) => frames.push(frame),
                    Err(Error::Memory) => break,
                    Err(e) => panic!("{e:?}"),
                }
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("{alloc:?}");

        let mut frames = frames.into_iter().flatten().collect::<Vec<_>>();

        assert_eq!(alloc.allocated_frames(), frames.len());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }
    }

    #[test]
    fn less_mem() {
        logging();

        const THREADS: usize = 4;
        const FRAMES: usize = 4096;
        const ALLOC_PER_THREAD: usize = FRAMES / THREADS - THREADS;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = vec![0; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for (i, frame) in frames.iter_mut().enumerate() {
                    if let Ok(f) = alloc.get(t, 0) {
                        *frame = f;
                    } else {
                        error!("OOM: {i}: {alloc:?}");
                        panic!()
                    }
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
            assert!(a < FRAMES && b < FRAMES);
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
        const FRAMES: usize = THREADS * (ALLOC_PER_THREAD + 2) * PT_LEN;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = vec![0; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, Lower::HUGE_ORDER).unwrap();
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
            assert!(a < FRAMES && b < FRAMES);
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

        let area = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(THREADS, area, Init::FreeAll).unwrap();
        let barrier = Barrier::new(THREADS);

        // Stress test
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            barrier.wait();

            let mut frames = vec![0; ALLOC_PER_THREAD];

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
        const FRAMES: usize = 4 * PT_LEN * PT_LEN;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

        // Alloc on first thread
        thread::pin(0);
        let mut frames = vec![0; ALLOC_PER_THREAD];
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

            let mut frames = vec![0; ALLOC_PER_THREAD];

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
        #[cfg(feature = "llc")]
        type Allocator<'a> = NvmAlloc<'a, LLC>;
        #[cfg(not(feature = "llc"))]
        type Allocator<'a> = NvmAlloc<'a, LLFree<'a>>;

        logging();

        const FRAMES: usize = 8 << 18;

        thread::pin(0);

        let expected_frames = (PT_LEN + 2) * (1 + (1 << 9));

        let mut zone = mmap::anon(0x1000_0000_0000, FRAMES, false, false);
        let secondary = aligned_buf(Allocator::metadata_size(1, FRAMES).secondary).leak();

        {
            let alloc = Allocator::create(1, &mut zone, false, secondary).unwrap();

            for _ in 0..PT_LEN + 2 {
                alloc.get(0, 0).unwrap();
                alloc.get(0, 9).unwrap();
            }

            assert_eq!(alloc.allocated_frames(), expected_frames);

            // leak (crash)
            std::mem::forget(alloc);
        }

        let secondary = aligned_buf(secondary.len()).leak();
        let alloc = Allocator::create(1, &mut zone, true, secondary).unwrap();
        assert_eq!(alloc.allocated_frames(), expected_frames);
    }

    #[test]
    fn different_orders() {
        const MAX_ORDER: usize = Lower::MAX_ORDER;
        const THREADS: usize = 4;
        const FRAMES: usize = (1 << MAX_ORDER) * (MAX_ORDER + 2) * THREADS; // 6 GiB for 16K

        logging();

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
        warn!("Created Allocator \n {:?}", alloc);
        let barrier = Barrier::new(THREADS);

        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut rng = WyRand::new(42 + t as u64);
            let mut num_frames = 0;
            let mut frames = Vec::new();
            for order in 0..=MAX_ORDER {
                for _ in 0..1 << (MAX_ORDER - order) {
                    frames.push((order, 0));
                    num_frames += 1 << order;
                }
            }
            rng.shuffle(&mut frames);

            warn!("allocate {num_frames} frames up to order <{MAX_ORDER}");
            barrier.wait();

            for (order, frame) in &mut frames {
                *frame = match alloc.get(t, *order) {
                    Ok(frame) => frame,
                    Err(e) => panic!("{e:?} o={order} {alloc:?} on core {t}"),
                };
                assert!(*frame % (1 << *order) == 0, "{frame} {:x}", 1 << *order);
                if *order > 8 {
                    //info!("allocated order {order}, {alloc:?} on core {t}");
                }
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
        const FRAMES: usize = 8 << 18;

        let alloc = Allocator::create(THREADS, FRAMES, Init::AllocAll).unwrap();
        assert_eq!(alloc.frames(), FRAMES);
        assert_eq!(alloc.allocated_frames(), FRAMES);

        for frame in (0..FRAMES).step_by(1 << Lower::HUGE_ORDER) {
            alloc.put(0, frame, Lower::HUGE_ORDER).unwrap();
        }
        assert_eq!(alloc.allocated_frames(), 0);

        thread::parallel(0..THREADS, |core| {
            thread::pin(core);
            for _ in 0..(FRAMES / THREADS) / (1 << Lower::HUGE_ORDER) {
                alloc.get(core, Lower::HUGE_ORDER).unwrap();
            }
        });
        assert_eq!(alloc.allocated_frames(), FRAMES);
    }

    #[test]
    fn fragmentation_retry() {
        logging();

        const FRAMES: usize = Lower::N * 2;
        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

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
        const FRAMES: usize = Lower::N * 2;
        let alloc = Allocator::create(2, FRAMES, Init::FreeAll).unwrap();
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
