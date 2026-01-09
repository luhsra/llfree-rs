// Include readme as documentation
#![doc = include_str!("../../README.md")]
// Disable standard library
#![no_std]
// Don't warn for compile-time checks
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::redundant_pattern_matching)]
#![debugger_visualizer(gdb_script_file = "../../scripts/gdb.py")]

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

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
/// Return [`Error::Address`] if condition is not met.
#[allow(unused_macros)]
macro_rules! ensure {
    ($cond:expr, $($args:expr),*) => {
        if !($cond) {
            error!($($args),*);
            return Err(Error::Address);
        }
    };
    ($err:expr; $cond:expr, $($args:expr),*) => {
        if !($cond) {
            error!($($args),*);
            return Err($err);
        }
    };
}

#[cfg(feature = "std")]
pub mod mmap;
#[cfg(feature = "std")]
pub mod thread;

pub mod atomic;
pub mod frame;
pub mod util;
pub mod wrapper;

mod bitfield;
mod llfree;
use bitfield_struct::bitfield;
pub use llfree::LLFree;

#[cfg(feature = "llc")]
mod llc;
#[cfg(feature = "llc")]
pub use llc::LLC;
#[cfg(feature = "llzig")]
mod llzig;
#[cfg(feature = "llzig")]
pub use llzig::LLZig;
use util::Align;

mod local;
mod lower;
mod trees;

use core::fmt;
use core::mem::align_of;
use core::ops::Range;

/// Order of a physical frame
pub const FRAME_SIZE: usize = if cfg!(feature = "16K") {
    0x4000
} else {
    0x1000
};
/// Number of huge frames in tree
pub const TREE_HUGE: usize = 8;
/// Number of small frames in tree
pub const TREE_FRAMES: usize = TREE_HUGE << HUGE_ORDER;
/// Order of an entire tree
pub const TREE_ORDER: usize = TREE_FRAMES.ilog2() as usize;
/// Order for huge frames
pub const HUGE_ORDER: usize = if cfg!(feature = "16K") { 11 } else { 9 };
/// Number of small frames in huge frame
pub const HUGE_FRAMES: usize = 1 << HUGE_ORDER;
/// Maximum order the llfree supports
pub const MAX_ORDER: usize = HUGE_ORDER + 1;
/// Bit size of the atomic ints that comprise the bitfields
pub const BITFIELD_ROW: usize = 64;

/// Number of retries if an atomic operation fails.
pub const RETRIES: usize = 4;

/// Allocation result
pub type Result<T> = core::result::Result<T, Error>;

/// The general interface of the allocator implementations.
pub trait Alloc<'a>: Sized + Sync + Send + fmt::Debug {
    /// Return the name of the allocator.
    #[cold]
    fn name() -> &'static str;

    /// Initialize the allocator.
    ///
    /// The metadata is stored into the primary (optionally persistent) and secondary buffers.
    #[cold]
    fn new(cores: usize, frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self>;

    /// Returns the size of the metadata buffers required for initialization.
    #[cold]
    fn metadata_size(cores: usize, frames: usize) -> MetaSize;
    /// Returns the metadata buffers.
    #[cold]
    fn metadata(&mut self) -> MetaData<'a>;

    /// Allocate a new frame of `order` on the given `core`.
    /// If specified try allocating the given `frame`.
    fn get(&self, core: usize, frame: Option<usize>, flags: Flags) -> Result<usize>;
    /// Free the `frame` of `order` on the given `core`.
    fn put(&self, core: usize, frame: usize, flags: Flags) -> Result<()>;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;
    /// Return the core count the allocator was initialized with.
    fn cores(&self) -> usize;

    /// Quickly retrieve allocator statistics, where `free_huge` is an under-approximation.
    fn fast_stats(&self) -> Stats;
    /// Quickly retrieve allocator statistics, where `free_huge` is an under-approximation.
    /// Only TREE_ORDER, HUGE_ORDER and 0 are supported.
    fn fast_stats_at(&self, frame: usize, order: usize) -> Stats;

    /// Retrieve detailed allocator statistics.
    fn stats(&self) -> Stats;
    /// Retrieve detailed allocator statistics.
    /// Only TREE_ORDER, HUGE_ORDER and 0 are supported.
    fn stats_at(&self, frame: usize, order: usize) -> Stats;

    /// Returns if `frame` is free. This might be racy!
    fn is_free(&self, frame: usize, order: usize) -> bool;

    /// Unreserve cpu-local frames
    fn drain(&self, _core: usize) -> Result<()> {
        Ok(())
    }

    /// Validate the internal state
    #[cold]
    fn validate(&self) {}
}

/// Size of the required metadata
#[derive(Debug)]
pub struct MetaSize {
    /// Size of the volatile CPU-local data.
    pub local: usize,
    /// Size of the volatile trees.
    pub trees: usize,
    /// Size of the optionally persistent data.
    pub lower: usize,
}

// The dynamic metadata of the allocator
pub struct MetaData<'a> {
    pub local: &'a mut [u8],
    pub trees: &'a mut [u8],
    pub lower: &'a mut [u8],
}
#[cfg(feature = "std")]
impl MetaData<'_> {
    pub fn alloc(m: MetaSize) -> Self {
        use util::aligned_buf;
        Self {
            local: aligned_buf(m.local),
            trees: aligned_buf(m.trees),
            lower: aligned_buf(m.lower),
        }
    }
}

impl MetaData<'_> {
    /// Check for alignment and overlap
    fn valid(&self, m: MetaSize) -> bool {
        fn overlap(a: Range<*const u8>, b: Range<*const u8>) -> bool {
            a.contains(&b.start)
                || a.contains(&unsafe { b.end.sub(1) })
                || b.contains(&a.start)
                || b.contains(&unsafe { a.end.sub(1) })
        }
        self.local.len() >= m.local
            && self.trees.len() >= m.trees
            && self.lower.len() >= m.lower
            && self.local.as_ptr().align_offset(align_of::<Align>()) == 0
            && self.trees.as_ptr().align_offset(align_of::<Align>()) == 0
            && self.lower.as_ptr().align_offset(align_of::<Align>()) == 0
            && !overlap(self.local.as_ptr_range(), self.trees.as_ptr_range())
            && !overlap(self.trees.as_ptr_range(), self.lower.as_ptr_range())
            && !overlap(self.lower.as_ptr_range(), self.local.as_ptr_range())
    }
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
    /// Assume that the allocator is already initialized
    None,
}

#[bitfield(u16)]
pub struct Flags {
    #[bits(8)]
    pub order: usize,
    pub movable: bool,
    #[bits(7)]
    __: (),
}
impl Flags {
    pub const fn o(order: usize) -> Self {
        Self::new().with_order(order)
    }
}

/// Statistics about the allocator's state
#[derive(Debug, Default)]
pub struct Stats {
    /// Number of free frames
    pub free_frames: usize,
    /// Number of entirely free huge frames
    pub free_huge: usize,
    /// Number of entirely free trees
    pub free_trees: usize,
}

#[cfg(all(test, feature = "std"))]
mod alloc_test {
    use core::mem::ManuallyDrop;
    use core::ops::Deref;
    use core::ptr::null_mut;
    use std::sync::Barrier;
    use std::time::Instant;
    use std::vec::Vec;

    use log::{error, warn};

    use super::*;
    use crate::frame::Frame;
    use crate::util::{WyRand, aligned_buf, logging};
    use crate::wrapper::NvmAlloc;

    #[cfg(all(feature = "llc", not(feature = "llzig")))]
    type Allocator = TestAlloc<LLC>;
    #[cfg(feature = "llzig")]
    type Allocator = TestAlloc<LLZig>;
    #[cfg(not(any(feature = "llc", feature = "llzig")))]
    type Allocator = TestAlloc<LLFree<'static>>;

    pub struct TestAlloc<A: Alloc<'static>>(ManuallyDrop<A>);

    impl<A: Alloc<'static>> TestAlloc<A> {
        pub fn create(cores: usize, frames: usize, init: Init) -> Result<Self> {
            let MetaSize {
                local,
                trees,
                lower,
            } = A::metadata_size(cores, frames);
            let meta = MetaData {
                local: aligned_buf(local),
                trees: aligned_buf(trees),
                lower: aligned_buf(lower),
            };
            Ok(Self(ManuallyDrop::new(A::new(cores, frames, init, meta)?)))
        }
    }
    impl<A: Alloc<'static>> Drop for TestAlloc<A> {
        fn drop(&mut self) {
            let MetaData {
                local,
                trees,
                lower,
            } = self.0.metadata();
            unsafe {
                // drop first
                drop(ManuallyDrop::take(&mut self.0));
                // free metadata buffers
                Vec::from_raw_parts(local.as_mut_ptr(), local.len(), local.len());
                Vec::from_raw_parts(trees.as_mut_ptr(), trees.len(), trees.len());
                Vec::from_raw_parts(lower.as_mut_ptr(), lower.len(), lower.len());
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
            A::fmt(self, f)
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

        assert_eq!(alloc.fast_stats().free_frames, alloc.frames());

        warn!("get >>>");
        let frame1 = alloc.get(0, None, Flags::o(0)).unwrap();
        warn!("get <<<");
        warn!("get >>>");
        let frame2 = alloc.get(0, None, Flags::o(0)).unwrap();
        warn!("get <<<");

        warn!("put >>>");
        alloc.put(0, frame2, Flags::o(0)).unwrap();
        warn!("put <<<");
        warn!("put >>>");
        alloc.put(0, frame1, Flags::o(0)).unwrap();
        warn!("put <<<");
        alloc.validate();
    }

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 * (1 << 30);
        const FRAMES: usize = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        assert_eq!(alloc.fast_stats().free_frames, alloc.frames());

        warn!("start alloc...");
        let small = alloc.get(0, None, Flags::o(0)).unwrap();

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            1,
            "{alloc:?}"
        );
        warn!("stress test...");

        // Stress test
        let mut frames = Vec::new();
        loop {
            match alloc.get(0, None, Flags::o(0)) {
                Ok(frame) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + frames.len());
        warn!("check...");
        alloc.validate();

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            1 + frames.len()
        );
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            alloc.frames()
        );
        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = HUGE_FRAMES - 10;
        for frame in &frames[..FREE_NUM] {
            alloc.put(0, *frame, Flags::o(0)).unwrap();
        }

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            1 + frames.len() - FREE_NUM,
            "{alloc:?}"
        );
        alloc.validate();

        // Realloc
        for frame in &mut frames[..FREE_NUM] {
            *frame = alloc.get(0, None, Flags::o(0)).unwrap();
        }

        warn!("free...");

        alloc.put(0, small, Flags::o(0)).unwrap();
        // Free all
        for frame in &frames {
            alloc.put(0, *frame, Flags::o(0)).unwrap();
        }

        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn alloc_at() {
        logging();
        const MEM_SIZE: usize = 1 << 30;
        let frames = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, frames, Init::FreeAll).unwrap();

        assert_eq!(alloc.fast_stats().free_frames, alloc.frames());

        alloc.get(0, Some(1), Flags::o(0)).unwrap();
        alloc.get(0, Some(2), Flags::o(0)).unwrap();
        alloc
            .get(0, Some(HUGE_FRAMES), Flags::o(HUGE_ORDER))
            .unwrap();

        // Test normal allocation
        let frame = alloc.get(0, None, Flags::o(0)).unwrap();
        assert!(frame != 1 && frame != 2 && frame != HUGE_FRAMES);

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            3 + HUGE_FRAMES
        );
        alloc.validate();

        alloc.put(0, HUGE_FRAMES, Flags::o(HUGE_ORDER)).unwrap();
        alloc.put(0, 2, Flags::o(0)).unwrap();
        alloc.put(0, 1, Flags::o(0)).unwrap();
        alloc.put(0, frame, Flags::o(0)).unwrap();

        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate();
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
            frames.push(alloc.get(0, None, Flags::o(0)).unwrap());
        }
        warn!("allocated {}", frames.len());

        warn!("check...");
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len()
        );
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }

        warn!("reallocate rand...");
        let mut rng = WyRand::new(100);
        rng.shuffle(&mut frames);

        for _ in 0..frames.len() {
            let i = rng.range(0..frames.len() as _) as usize;
            alloc.put(0, frames[i], Flags::o(0)).unwrap();
            frames[i] = alloc.get(0, None, Flags::o(0)).unwrap();
        }

        warn!("check...");
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len()
        );
        alloc.validate();
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }

        warn!("free...");
        rng.shuffle(&mut frames);
        for frame in &frames {
            alloc.put(0, *frame, Flags::o(0)).unwrap();
        }
        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate()
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
                frames.push(alloc.get(t, None, Flags::o(0)).unwrap());
            }
            warn!("allocated {}", frames.len());

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
                assert_ne!(a, b);
                assert!(a < FRAMES && b < FRAMES);
            }

            barrier.wait();
            warn!("reallocate rand...");
            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for _ in 0..frames.len() {
                let i = rng.range(0..frames.len() as _) as usize;
                alloc.put(t, frames[i], Flags::o(0)).unwrap();
                frames[i] = alloc.get(t, None, Flags::o(0)).unwrap();
            }

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
                assert_ne!(a, b);
                assert!(a < FRAMES && b < FRAMES);
            }

            if barrier.wait().is_leader() {
                alloc.validate();
            }
            barrier.wait();

            warn!("free...");
            rng.shuffle(&mut frames);
            for frame in &frames {
                alloc.put(t, *frame, Flags::o(0)).unwrap();
            }
        });

        assert_eq!(alloc.stats().free_huge, FRAMES / HUGE_FRAMES);
        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 2 << 30;
        const FRAMES: usize = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);

        warn!("start alloc");
        let small = alloc.get(0, None, Flags::o(0)).unwrap();
        let huge = alloc.get(0, None, Flags::o(HUGE_ORDER)).unwrap();

        let expected_frames = 1 + HUGE_FRAMES;
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            expected_frames
        );
        assert!(small != huge);

        warn!("start stress test");

        // Stress test
        let mut frames = vec![0; FRAMES / 2];
        for frame in &mut frames {
            *frame = alloc.get(0, None, Flags::o(0)).unwrap();
        }

        warn!("check");
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            expected_frames + frames.len()
        );
        alloc.validate();

        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }

        warn!("free some...");

        // Free some
        for frame in &frames[10..HUGE_FRAMES + 10] {
            alloc.put(0, *frame, Flags::o(0)).unwrap();
        }
        alloc.validate();

        warn!("free special...");

        alloc.put(0, small, Flags::o(0)).unwrap();
        alloc.put(0, huge, Flags::o(HUGE_ORDER)).unwrap();
        alloc.validate();

        warn!("realloc...");

        // Realloc
        for frame in &mut frames[10..HUGE_FRAMES + 10] {
            *frame = alloc.get(0, None, Flags::o(0)).unwrap();
        }
        alloc.validate();

        warn!("free...");
        // Free all
        for frame in &frames {
            alloc.put(0, *frame, Flags::o(0)).unwrap();
        }

        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        assert_eq!(alloc.stats().free_huge, FRAMES / HUGE_FRAMES);
        alloc.validate();
    }

    #[test]
    fn parallel_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 2 * THREADS);
        const FRAMES: usize = 2 * THREADS * HUGE_FRAMES * HUGE_FRAMES;

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
                    *frame = alloc.get(t, None, Flags::o(0)).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len()
        );
        warn!("allocated frames: {}", frames.len());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }
    }

    #[test]
    fn alloc_all() {
        logging();

        const FRAMES: usize = 2 * HUGE_FRAMES * HUGE_FRAMES;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = Vec::new();
        let timer = Instant::now();

        loop {
            match alloc.get(0, None, Flags::o(0)) {
                Ok(frame) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("{alloc:?}");

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len()
        );
        assert_eq!(alloc.fast_stats().free_frames, 0);
        assert_eq!(alloc.stats().free_huge, 0);
        warn!("allocated frames: {}", frames.len());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }
    }

    #[test]
    fn parallel_alloc_all() {
        logging();

        const THREADS: usize = 4;
        const FRAMES: usize = 2 * THREADS * HUGE_FRAMES * HUGE_FRAMES;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

        // Stress test
        let mut frames = vec![Vec::new(); THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(frames.iter_mut().enumerate(), |(t, frames)| {
            thread::pin(t);
            barrier.wait();

            loop {
                match alloc.get(t, None, Flags::o(0)) {
                    Ok(frame) => frames.push(frame),
                    Err(Error::Memory) => break,
                    Err(e) => panic!("{e:?}"),
                }
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("{alloc:?}");

        let mut frames = frames.into_iter().flatten().collect::<Vec<_>>();

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len()
        );
        warn!("allocated frames: {}/{}", frames.len(), alloc.frames());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
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
                    if let Ok(f) = alloc.get(t, None, Flags::o(0)) {
                        *frame = f;
                    } else {
                        error!("OOM: {i}: {alloc:?}");
                        panic!()
                    }
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len()
        );
        warn!("allocated frames: {}", frames.len());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }

        thread::parallel(
            frames.chunks(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    alloc.put(t, *frame, Flags::o(0)).unwrap();
                }
            },
        );

        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn parallel_huge_alloc() {
        logging();

        const THREADS: usize = 4;
        const FRAMES: usize = (8 << 30) / Frame::SIZE; // 1GiB
        const ALLOC_PER_THREAD: usize = FRAMES / THREADS / HUGE_FRAMES;
        // additional space for the allocators metadata

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
                    *frame = alloc.get(t, None, Flags::o(HUGE_ORDER)).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            frames.len() * HUGE_FRAMES
        );
        warn!(
            "allocated frames: {}/{}",
            frames.len(),
            alloc.frames() / HUGE_FRAMES
        );
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a < FRAMES && b < FRAMES);
        }
    }

    #[ignore]
    #[test]
    fn parallel_malloc() {
        logging();

        const THREADS: usize = 4;
        const FRAMES: usize = (8 << 30) / Frame::SIZE; // 1GiB
        const ALLOC_PER_THREAD: usize = FRAMES / THREADS / HUGE_FRAMES;

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
        const FRAMES: usize = (8 << 30) / Frame::SIZE; // 1GiB
        const ALLOC_PER_THREAD: usize = FRAMES / THREADS / HUGE_FRAMES;

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
        const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * HUGE_FRAMES * HUGE_FRAMES * Frame::SIZE;

        let area = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(THREADS, area, Init::FreeAll).unwrap();
        let barrier = Barrier::new(THREADS);

        // Stress test
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            barrier.wait();

            let mut frames = vec![0; ALLOC_PER_THREAD];

            for frame in &mut frames {
                *frame = alloc.get(t, None, Flags::o(0)).unwrap();
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for frame in frames {
                alloc.put(t, frame, Flags::o(0)).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 10) / 2;
        const FRAMES: usize = 4 * HUGE_FRAMES * HUGE_FRAMES;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
        warn!("{alloc:?}");

        // Alloc on first thread
        thread::pin(0);
        let mut frames = vec![0; ALLOC_PER_THREAD];
        for frame in &mut frames {
            *frame = alloc.get(0, None, Flags::o(0)).unwrap();
        }
        alloc.validate();

        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            s.spawn(|| {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for frame in &frames {
                    alloc.put(1, *frame, Flags::o(0)).unwrap();
                }
            });

            let mut frames = vec![0; ALLOC_PER_THREAD];

            barrier.wait();

            // Simultaneously alloc on first thread
            for frame in &mut frames {
                *frame = alloc.get(0, None, Flags::o(0)).unwrap();
            }
        });

        warn!("check {alloc:?}");
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            ALLOC_PER_THREAD
        );
        alloc.validate();
    }

    #[test]
    fn recover() {
        #[cfg(feature = "llc")]
        type Allocator<'a> = NvmAlloc<'a, LLC>;
        #[cfg(not(feature = "llc"))]
        type Allocator<'a> = NvmAlloc<'a, LLFree<'a>>;

        logging();

        const FRAMES: usize = 8 * (1 << 30) / FRAME_SIZE;

        thread::pin(0);

        let expected_frames = 128 * (1 + (1 << HUGE_ORDER));

        let mut zone = mmap::Mapping::anon(0x1000_0000_0000, FRAMES, false, false).unwrap();
        let m = Allocator::metadata_size(1, FRAMES);
        let local = aligned_buf(m.local);
        let trees = aligned_buf(m.trees);

        {
            let alloc = Allocator::create(1, &mut zone, false, local, trees).unwrap();

            let mut _allocated_frames = 0;
            for _ in 0..128 {
                alloc.get(0, None, Flags::o(0)).unwrap();
                _allocated_frames = alloc.frames() - alloc.fast_stats().free_frames;
                alloc.get(0, None, Flags::o(HUGE_ORDER)).unwrap();
                _allocated_frames = alloc.frames() - alloc.fast_stats().free_frames;
            }

            assert_eq!(
                alloc.frames() - alloc.fast_stats().free_frames,
                expected_frames
            );
            alloc.validate();

            // leak (crash)
            std::mem::forget(alloc);
        }

        let local = aligned_buf(m.local);
        let trees = aligned_buf(m.trees);
        let alloc = Allocator::create(1, &mut zone, true, local, trees).unwrap();
        assert_eq!(
            alloc.frames() - alloc.fast_stats().free_frames,
            expected_frames
        );
        alloc.validate();
    }

    #[test]
    fn different_orders() {
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

            // reallocate all
            for (order, frame) in &mut frames {
                *frame = match alloc.get(t, None, Flags::o(*order)) {
                    Ok(frame) => frame,
                    Err(e) => panic!("{e:?} o={order} {alloc:?} on core {t}"),
                };
                assert!(*frame % (1 << *order) == 0, "{frame} {:x}", 1 << *order);
            }

            rng.shuffle(&mut frames);

            // free all
            for (order, frame) in frames {
                if let Err(e) = alloc.put(t, frame, Flags::o(order)) {
                    panic!("{e:?} o={order} {alloc:#?}")
                }
            }
        });

        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn init_reserved() {
        logging();

        const THREADS: usize = 2;
        const FRAMES: usize = 8 << 18;

        let alloc = Allocator::create(THREADS, FRAMES, Init::AllocAll).unwrap();
        assert_eq!(alloc.frames(), FRAMES);
        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, FRAMES);

        for frame in (0..FRAMES).step_by(1 << HUGE_ORDER) {
            alloc.put(0, frame, Flags::o(HUGE_ORDER)).unwrap();
        }
        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, 0);

        thread::parallel(0..THREADS, |core| {
            thread::pin(core);
            for _ in (0..FRAMES / THREADS).step_by(1 << HUGE_ORDER) {
                alloc.get(core, None, Flags::o(HUGE_ORDER)).unwrap();
            }
        });
        assert_eq!(alloc.frames() - alloc.fast_stats().free_frames, FRAMES);
        alloc.validate();
    }

    #[test]
    fn fragmentation_retry() {
        logging();

        const FRAMES: usize = TREE_FRAMES * 2;
        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

        // Alloc a whole subtree
        let mut frames = Vec::with_capacity(TREE_FRAMES / 2);
        for i in 0..TREE_FRAMES {
            if i % 2 == 0 {
                frames.push(alloc.get(0, None, Flags::o(0)).unwrap());
            } else {
                alloc.get(0, None, Flags::o(0)).unwrap();
            }
        }
        // Free every second one -> fragmentation
        for frame in frames {
            alloc.put(0, frame, Flags::o(0)).unwrap();
        }

        let huge = alloc.get(0, None, Flags::o(9)).unwrap();
        warn!("huge = {huge}");
        warn!("{alloc:?}");
        alloc.validate();
    }

    #[test]
    fn drain() {
        logging();

        const FRAMES: usize = TREE_FRAMES * 8;
        let alloc = Allocator::create(2, FRAMES, Init::FreeAll).unwrap();
        // should not change anything
        alloc.drain(0).unwrap();
        alloc.drain(1).unwrap();

        // allocate on second core => reserve a subtree
        alloc.get(1, None, Flags::o(0)).unwrap();

        // completely the subtree of the first core
        for _ in 0..FRAMES - TREE_FRAMES {
            alloc.get(0, None, Flags::o(0)).unwrap();
        }
        // next allocation should trigger drain+reservation (no subtree left)
        println!("{:?}", alloc.get(0, None, Flags::o(0)));
        alloc.validate();
    }

    #[test]
    fn stress() {
        const THREADS: usize = 4;
        const FRAMES: usize = (1 << 30) / Frame::SIZE;
        const ITER: usize = 100;

        logging();

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
        alloc.validate();

        let rand = unsafe { libc::rand() as u64 };
        let barrier = Barrier::new(THREADS);

        let allocated = thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut rng = WyRand::new(rand + t as u64);
            let mut frames = Vec::with_capacity(FRAMES / THREADS);
            barrier.wait();

            for _ in 0..ITER {
                let target = rng.range(0..(FRAMES / THREADS) as _) as usize;
                while frames.len() != target {
                    if frames.len() < target {
                        match alloc.get(t, None, Flags::o(0)) {
                            Ok(frame) => frames.push(frame),
                            Err(Error::Memory) => break,
                            Err(e) => panic!("{e:?}"),
                        }
                    } else {
                        alloc.put(t, frames.pop().unwrap(), Flags::o(0)).unwrap();
                    }
                }
            }
            frames.len()
        });

        assert_eq!(
            allocated.into_iter().sum::<usize>(),
            alloc.frames() - alloc.fast_stats().free_frames
        );
        alloc.validate();
    }
}
