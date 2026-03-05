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
            log::error!($($args),*);
            return Err(Error::Address);
        }
    };
    ($err:expr; $cond:expr, $($args:expr),*) => {
        if !($cond) {
            log::error!($($args),*);
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
use bitfield::RowId;
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
pub use lower::HugeId;
mod trees;
pub use trees::TreeId;

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
    /// The `kind` descritors define how many local trees we have for each kind.
    /// We need at least one for small and one for huge pages,
    /// but we can have more to reduce sharing or optimize fragmentation.
    ///
    /// The metadata is stored into the primary (optionally persistent) and secondary buffers.
    #[cold]
    fn new(kinds: &[TierConfig], frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self>;

    /// Returns the size of the metadata buffers required for initialization.
    #[cold]
    fn metadata_size(kinds: &[TierConfig], frames: usize) -> MetaSize;
    /// Returns the metadata buffers.
    ///
    /// # Safety
    /// These buffers must must not be freed or used for other purposes.
    #[cold]
    unsafe fn metadata(&mut self) -> MetaData<'a>;

    /// Allocate a new frame of `order` on the given `local`.
    /// If specified try allocating the given `frame`.
    fn get(&self, frame: Option<FrameId>, flags: Flags) -> Result<(Tier, FrameId)>;
    /// Free the `frame` of `order` on the given `local`.
    fn put(&self, frame: FrameId, flags: Flags) -> Result<()>;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;

    /// Quickly retrieve tree statistics.
    fn tree_stats(&self, kinds: &mut [KindStats]) -> TreeStats;

    /// Retrieve detailed allocator statistics.
    fn stats(&self) -> Stats;
    /// Retrieve detailed allocator statistics.
    /// Only TREE_ORDER, HUGE_ORDER and 0 are supported.
    fn stats_at(&self, frame: FrameId, order: usize) -> Stats;

    /// Returns if `frame` is free. This might be racy!
    fn is_free(&self, frame: FrameId, order: usize) -> bool;

    /// Unreserve a local tree
    fn drain(&self, _local: usize) -> Result<()> {
        Ok(())
    }

    /// Validate the internal state
    #[cold]
    fn validate(&self) {}
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameId(pub usize);
impl FrameId {
    const fn as_tree(self) -> TreeId {
        TreeId(self.0 / TREE_FRAMES)
    }
    const fn as_huge(self) -> HugeId {
        HugeId(self.0 / HUGE_FRAMES)
    }
    const fn as_row(self) -> RowId {
        RowId(self.0 / BITFIELD_ROW)
    }
    const fn row_bit_idx(self) -> usize {
        self.0 % BITFIELD_ROW
    }
    const fn is_aligned(self, order: usize) -> bool {
        self.0 & ((1 << order) - 1) == 0
    }
    pub const fn into_bits(self) -> u64 {
        self.0 as u64
    }
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits as usize)
    }
}
impl core::ops::Add<Self> for FrameId {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fx{:x}", self.0)
    }
}
impl fmt::Debug for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
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

pub struct Tiering<'a> {
    /// Specifies the tiers and number of local reservations for each tier.
    pub tiers: &'a [TierConfig],
    /// Policy for accessing a tree with a `target` rank and `free` counter.
    pub evaluate: fn(current: Tier, target: Tier, free: usize) -> Policy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Match, where higher is better, [u8::MAX] is a perfect match
    Match(u8),
    /// Can steal from the target, but not demote it
    Stealing,
    /// Would demote the target to the current kind
    Demotes,
    /// Can't be used
    Invalid,
}

/// Specifies how many reservations of each rank the allocator should maintain
#[derive(Clone, Copy, Debug)]
pub struct TierConfig {
    /// Opaque rank of a tree, used for reservation and stealing policies
    pub tier: Tier,
    /// The number of local reservations of this rank or 0 for none.
    pub count: u8,
}


/// Opaque rank of a tree
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tier(pub u8);
impl Tier {
    pub const BITS: usize = 3;
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
    pub const fn into_bits(self) -> u8 {
        self.0
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
    /// Allocation order (#frames = 1 << order)
    #[bits(4)]
    pub order: usize,
    /// Local tree index (local trees are configured during initialization)
    ///
    /// The policy for mapping locals to cores has to be provided by the user,
    /// but a common approach is to have one local tree per core.
    #[bits(12)]
    pub local: usize,
}
impl Flags {
    pub fn with(order: usize, local: usize) -> Self {
        Self::new().with_order(order).with_local(local)
    }
}
impl Flags {
    pub const fn frames(self) -> usize {
        1 << self.order()
    }
}
const _: () = assert!(Flags::ORDER_BITS >= MAX_ORDER.ilog2() as usize);

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

#[derive(Debug, Default)]
pub struct TreeStats {
    /// Number of total free frames
    pub free_frames: usize,
    /// Number of free trees
    pub free_trees: usize,
}

pub struct KindStats {
    /// Number of free frames
    pub free: usize,
    /// Number of allocated frames
    pub alloc: usize,
}

#[cfg(all(test, feature = "std"))]
mod alloc_test {
    use core::mem::ManuallyDrop;
    use core::ops::Deref;
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
            let kinds = TierConfig::cores(cores);
            let MetaSize {
                local,
                trees,
                lower,
            } = A::metadata_size(&kinds, frames);
            let meta = MetaData {
                local: aligned_buf(local),
                trees: aligned_buf(trees),
                lower: aligned_buf(lower),
            };
            Ok(Self(ManuallyDrop::new(A::new(&kinds, frames, init, meta)?)))
        }
    }
    impl<A: Alloc<'static>> Drop for TestAlloc<A> {
        fn drop(&mut self) {
            unsafe {
                let MetaData {
                    local,
                    trees,
                    lower,
                } = self.0.metadata();
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

    fn reserve(kind: Tier, other: Tier, free: usize) -> Policy {
        if kind.0 > other.0 {
            return Policy::Stealing;
        }
        if kind.0 < other.0 {
            return Policy::Demotes;
        }
        match free {
            TREE_FRAMES => Policy::Good(1), // fully free -> fragmentation
            f if f >= TREE_FRAMES / 2 => Policy::Good(3), // half free
            f if f >= TREE_FRAMES / 64 => Policy::Best, // almost allocated
            _ => Policy::Good(2),           // low free count -> causes frequent reservations
        }
    }

    // Test setup: 2 trees per core, one for small and one for huge frames
    impl TierConfig {
        pub const fn cores(cores: usize) -> [Self; 2] {
            [
                Self {
                    tier: Tier(0),
                    count: cores as _,
                    reserve: |other, free| reserve(Tier(0), other, free),
                },
                Self {
                    tier: Tier(1),
                    count: cores as _,
                    reserve: |other, free| reserve(Tier(1), other, free),
                },
            ]
        }
    }
    fn flags(order: usize, core: usize, cores: usize) -> Flags {
        Flags::new()
            .with_order(order)
            .with_local(core % cores + if order >= HUGE_ORDER { cores } else { 0 })
    }

    #[test]
    fn minimal() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        let frames = MEM_SIZE / Frame::SIZE;

        warn!("init");
        let alloc = Allocator::create(1, frames, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, 1);
        warn!("finit");

        assert_eq!(alloc.tree_stats(&mut []).free_frames, alloc.frames());

        warn!("get >>>");
        let (_, frame1) = alloc.get(None, llf(0, 0)).unwrap();
        warn!("get <<<");
        warn!("get >>>");
        let (_, frame2) = alloc.get(None, llf(0, 0)).unwrap();
        warn!("get <<<");

        warn!("put >>>");
        alloc.put(frame2, llf(0, 0)).unwrap();
        warn!("put <<<");
        warn!("put >>>");
        alloc.put(frame1, llf(0, 0)).unwrap();
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
        let llf = |order, core| flags(order, core, 1);

        assert_eq!(alloc.tree_stats(&mut []).free_frames, alloc.frames());

        warn!("start alloc...");
        let (_, small) = alloc.get(None, llf(0, 0)).unwrap();

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            1,
            "{alloc:?}"
        );
        warn!("stress test...");

        // Stress test
        let mut frames = Vec::new();
        loop {
            match alloc.get(None, llf(0, 0)) {
                Ok((_, frame)) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + frames.len());
        warn!("check...");
        alloc.validate();

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            1 + frames.len()
        );
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            alloc.frames()
        );
        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = HUGE_FRAMES - 10;
        for frame in &frames[..FREE_NUM] {
            alloc.put(*frame, llf(0, 0)).unwrap();
        }

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            1 + frames.len() - FREE_NUM,
            "{alloc:?}"
        );
        alloc.validate();

        // Realloc
        for frame in &mut frames[..FREE_NUM] {
            *frame = alloc.get(None, llf(0, 0)).unwrap().1;
        }

        warn!("free...");

        alloc.put(small, llf(0, 0)).unwrap();
        // Free all
        for frame in &frames {
            alloc.put(*frame, llf(0, 0)).unwrap();
        }

        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn alloc_at() {
        logging();
        const MEM_SIZE: usize = 1 << 30;
        let frames = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, frames, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, 1);

        assert_eq!(alloc.tree_stats(&mut []).free_frames, alloc.frames());

        alloc.get(Some(FrameId(1)), llf(0, 0)).unwrap();
        alloc.get(Some(FrameId(2)), llf(0, 0)).unwrap();
        alloc
            .get(Some(FrameId(HUGE_FRAMES)), llf(HUGE_ORDER, 0))
            .unwrap();

        // Test normal allocation
        let (_, frame) = alloc.get(None, llf(0, 0)).unwrap();
        assert!(frame != FrameId(1) && frame != FrameId(2) && frame != FrameId(HUGE_FRAMES));

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            3 + HUGE_FRAMES
        );
        alloc.validate();

        alloc.put(FrameId(HUGE_FRAMES), llf(HUGE_ORDER, 0)).unwrap();
        alloc.put(FrameId(2), llf(0, 0)).unwrap();
        alloc.put(FrameId(1), llf(0, 0)).unwrap();
        alloc.put(frame, llf(0, 0)).unwrap();

        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn rand() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 4 << 30;
        const FRAMES: usize = MEM_SIZE / Frame::SIZE;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, 1);

        warn!("start alloc...");
        const ALLOCS: usize = MEM_SIZE / Frame::SIZE / 2;
        let mut frames = Vec::with_capacity(ALLOCS);
        for _ in 0..ALLOCS {
            frames.push(alloc.get(None, llf(0, 0)).unwrap().1);
        }
        warn!("allocated {}", frames.len());

        warn!("check...");
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            frames.len()
        );
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }

        warn!("reallocate rand...");
        let mut rng = WyRand::new(100);
        rng.shuffle(&mut frames);

        for _ in 0..frames.len() {
            let i = rng.range(0..frames.len() as _) as usize;
            alloc.put(frames[i], llf(0, 0)).unwrap();
            frames[i] = alloc.get(None, llf(0, 0)).unwrap().1;
        }

        warn!("check...");
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            frames.len()
        );
        alloc.validate();
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }

        warn!("free...");
        rng.shuffle(&mut frames);
        for frame in &frames {
            alloc.put(*frame, llf(0, 0)).unwrap();
        }
        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
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
            let llf = |order, core| flags(order, core, THREADS);

            barrier.wait();
            warn!("start alloc...");
            let mut frames = Vec::with_capacity(ALLOCS);
            for _ in 0..ALLOCS {
                frames.push(alloc.get(None, llf(0, t)).unwrap().1);
            }
            warn!("allocated {}", frames.len());

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
                assert_ne!(a, b);
                assert!(a.0 < FRAMES && b.0 < FRAMES);
            }

            barrier.wait();
            warn!("reallocate rand...");
            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for _ in 0..frames.len() {
                let i = rng.range(0..frames.len() as _) as usize;
                alloc.put(frames[i], llf(0, t)).unwrap();
                frames[i] = alloc.get(None, llf(0, t)).unwrap().1;
            }

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
                assert_ne!(a, b);
                assert!(a.0 < FRAMES && b.0 < FRAMES);
            }

            if barrier.wait().is_leader() {
                alloc.validate();
            }
            barrier.wait();

            warn!("free...");
            rng.shuffle(&mut frames);
            for frame in &frames {
                alloc.put(*frame, llf(0, t)).unwrap();
            }
        });

        assert_eq!(alloc.stats().free_huge, FRAMES / HUGE_FRAMES);
        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
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
        let mut frames = vec![FrameId(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                let llf = |order, core| flags(order, core, THREADS);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(None, llf(0, t)).unwrap().1;
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            frames.len()
        );
        warn!("allocated frames: {}", frames.len());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }
    }

    #[test]
    fn alloc_all() {
        logging();

        const FRAMES: usize = 2 * HUGE_FRAMES * HUGE_FRAMES;

        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, 1);

        // Stress test
        let mut frames = Vec::new();
        let timer = Instant::now();

        loop {
            match alloc.get(None, llf(0, 0)) {
                Ok((_, frame)) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("{alloc:?}");

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            frames.len()
        );
        assert_eq!(alloc.tree_stats(&mut []).free_frames, 0);
        assert_eq!(alloc.stats().free_huge, 0);
        warn!("allocated frames: {}", frames.len());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
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
            let llf = |order, core| flags(order, core, THREADS);
            barrier.wait();

            loop {
                match alloc.get(None, llf(0, t)) {
                    Ok((_, frame)) => frames.push(frame),
                    Err(Error::Memory) => break,
                    Err(e) => panic!("{e:?}"),
                }
            }
        });
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("{alloc:?}");

        let mut frames = frames.into_iter().flatten().collect::<Vec<_>>();

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            frames.len()
        );
        warn!("allocated frames: {}/{}", frames.len(), alloc.frames());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
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
        let mut frames = vec![FrameId(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                let llf = |order, core| flags(order, core, THREADS);
                barrier.wait();

                for (i, frame) in frames.iter_mut().enumerate() {
                    if let Ok((_, f)) = alloc.get(None, llf(0, t)) {
                        *frame = f;
                    } else {
                        error!("OOM: {i}: {alloc:#?}");
                        panic!()
                    }
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            frames.len()
        );
        warn!("allocated frames: {}", frames.len());
        alloc.validate();

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }

        thread::parallel(
            frames.chunks(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                let llf = |order, core| flags(order, core, THREADS);
                barrier.wait();

                for frame in frames {
                    alloc.put(*frame, llf(0, t)).unwrap();
                }
            },
        );

        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
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
        let mut frames = vec![FrameId(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                let llf = |order, core| flags(order, core, THREADS);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(None, llf(HUGE_ORDER, t)).unwrap().1;
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
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
            assert!(a.0 < FRAMES && b.0 < FRAMES);
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
            let llf = |order, core| flags(order, core, THREADS);
            barrier.wait();

            let mut frames = vec![FrameId(0); ALLOC_PER_THREAD];

            for frame in &mut frames {
                *frame = alloc.get(None, llf(0, t)).unwrap().1;
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for frame in frames {
                alloc.put(frame, llf(0, t)).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 10) / 2;
        const FRAMES: usize = 4 * HUGE_FRAMES * HUGE_FRAMES;

        let alloc = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, THREADS);
        warn!("{alloc:?}");

        // Alloc on first thread
        thread::pin(0);
        let mut frames = vec![FrameId(0); ALLOC_PER_THREAD];
        for frame in &mut frames {
            *frame = alloc.get(None, llf(0, 0)).unwrap().1;
        }
        alloc.validate();

        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            s.spawn(|| {
                thread::pin(1);
                let llf = |order, core| flags(order, core, THREADS);
                barrier.wait();
                // Free on another thread
                for frame in &frames {
                    alloc.put(*frame, llf(0, 1)).unwrap();
                }
            });

            let mut frames = vec![FrameId(0); ALLOC_PER_THREAD];
            let llf = |order, core| flags(order, core, THREADS);

            barrier.wait();

            // Simultaneously alloc on first thread
            for frame in &mut frames {
                *frame = alloc.get(None, llf(0, 0)).unwrap().1;
            }
        });

        warn!("check {alloc:?}");
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
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
        let m = Allocator::metadata_size(&TierConfig::cores(1), FRAMES);
        let local = aligned_buf(m.local);
        let trees = aligned_buf(m.trees);

        {
            let alloc =
                Allocator::create(&TierConfig::cores(1), &mut zone, false, local, trees).unwrap();
            let llf = |order, core| flags(order, core, 1);

            let mut _allocated_frames = 0;
            for _ in 0..128 {
                alloc.get(None, llf(0, 0)).unwrap();
                _allocated_frames = alloc.frames() - alloc.tree_stats(&mut []).free_frames;
                alloc.get(None, llf(HUGE_ORDER, 0)).unwrap();
                _allocated_frames = alloc.frames() - alloc.tree_stats(&mut []).free_frames;
            }

            assert_eq!(
                alloc.frames() - alloc.tree_stats(&mut []).free_frames,
                expected_frames
            );
            alloc.validate();

            // leak (crash)
            std::mem::forget(alloc);
        }

        let local = aligned_buf(m.local);
        let trees = aligned_buf(m.trees);
        let alloc =
            Allocator::create(&TierConfig::cores(1), &mut zone, true, local, trees).unwrap();
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
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
        let llf = |order, core| flags(order, core, THREADS);
        warn!("Created Allocator \n {:?}", alloc);
        let barrier = Barrier::new(THREADS);

        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut rng = WyRand::new(42 + t as u64);
            let mut num_frames = 0;
            let mut frames = Vec::new();
            for order in 0..=MAX_ORDER {
                for _ in 0..1 << (MAX_ORDER - order) {
                    frames.push((order, FrameId(0)));
                    num_frames += 1 << order;
                }
            }
            rng.shuffle(&mut frames);

            warn!("allocate {num_frames} frames up to order <= {MAX_ORDER}");
            barrier.wait();

            // reallocate all
            for (order, frame) in &mut frames {
                *frame = match alloc.get(None, llf(*order, t)) {
                    Ok((_, frame)) => frame,
                    Err(e) => panic!("{e:?} o={order} {alloc:?} on core {t}"),
                };
                assert!(frame.is_aligned(*order), "{frame:?} {:x}", 1 << *order);
            }

            rng.shuffle(&mut frames);

            // free all
            for (order, frame) in frames {
                if let Err(e) = alloc.put(frame, llf(order, t)) {
                    panic!("{e:?} o={order} {alloc:#?}")
                }
            }
        });

        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);
        alloc.validate();
    }

    #[test]
    fn init_reserved() {
        logging();

        const THREADS: usize = 2;
        const FRAMES: usize = 8 << 18;

        let alloc = Allocator::create(THREADS, FRAMES, Init::AllocAll).unwrap();
        let llf = |order, core| flags(order, core, THREADS);
        assert_eq!(alloc.frames(), FRAMES);
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            FRAMES
        );

        for frame in (0..FRAMES).step_by(1 << HUGE_ORDER) {
            alloc.put(FrameId(frame), llf(HUGE_ORDER, 0)).unwrap();
        }
        assert_eq!(alloc.frames() - alloc.tree_stats(&mut []).free_frames, 0);

        thread::parallel(0..THREADS, |core| {
            thread::pin(core);
            let llf = |order, core| flags(order, core, THREADS);
            for _ in (0..FRAMES / THREADS).step_by(1 << HUGE_ORDER) {
                alloc.get(None, llf(HUGE_ORDER, core)).unwrap();
            }
        });
        assert_eq!(
            alloc.frames() - alloc.tree_stats(&mut []).free_frames,
            FRAMES
        );
        alloc.validate();
    }

    #[test]
    fn fragmentation_retry() {
        logging();

        const FRAMES: usize = TREE_FRAMES * 2;
        let alloc = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, 1);

        // Alloc a whole subtree
        let mut frames = Vec::with_capacity(TREE_FRAMES / 2);
        for i in 0..TREE_FRAMES {
            if i % 2 == 0 {
                frames.push(alloc.get(None, llf(0, 0)).unwrap().1);
            } else {
                alloc.get(None, llf(0, 0)).unwrap().1;
            }
        }
        // Free every second one -> fragmentation
        for frame in frames {
            alloc.put(frame, llf(0, 0)).unwrap();
        }

        let huge = alloc.get(None, llf(9, 0)).unwrap();
        warn!("huge = {:?}", huge);
        warn!("{alloc:?}");
        alloc.validate();
    }

    #[test]
    fn drain() {
        logging();

        const FRAMES: usize = TREE_FRAMES * 8;
        let alloc = Allocator::create(2, FRAMES, Init::FreeAll).unwrap();
        let llf = |order, core| flags(order, core, 2);
        // should not change anything
        alloc.drain(0).unwrap();
        alloc.drain(1).unwrap();

        // allocate on second core => reserve a subtree
        alloc.get(None, llf(0, 1)).unwrap();

        // completely the subtree of the first core
        for _ in 0..FRAMES - TREE_FRAMES {
            alloc.get(None, llf(0, 0)).unwrap();
        }
        // next allocation should trigger drain+reservation (no subtree left)
        println!("{:?}", alloc.get(None, llf(0, 0)));
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
            let llf = |order, core| flags(order, core, THREADS);
            let mut rng = WyRand::new(rand + t as u64);
            let mut frames = Vec::with_capacity(FRAMES / THREADS);
            barrier.wait();

            for _ in 0..ITER {
                let target = rng.range(0..(FRAMES / THREADS) as _) as usize;
                while frames.len() != target {
                    if frames.len() < target {
                        match alloc.get(None, llf(0, t)) {
                            Ok((_, frame)) => frames.push(frame),
                            Err(Error::Memory) => break,
                            Err(e) => panic!("{e:?}"),
                        }
                    } else {
                        alloc.put(frames.pop().unwrap(), llf(0, t)).unwrap();
                    }
                }
            }
            frames.len()
        });

        assert_eq!(
            allocated.into_iter().sum::<usize>(),
            alloc.frames() - alloc.tree_stats(&mut []).free_frames
        );
        alloc.validate();
    }
}
