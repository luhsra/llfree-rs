// Include readme as documentation
#![doc = include_str!("../../README.md")]
// Disable standard library
#![no_std]
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

mod atomic;
pub mod frame;
pub mod util;
pub mod wrapper;

mod bitfield;
use bitfield::RowId;
mod llfree;
pub use llfree::LLFree;

#[cfg(feature = "llc")]
mod llc;
#[cfg(feature = "llc")]
pub use llc::LLC;
#[cfg(feature = "llzig")]
mod llzig;
#[cfg(feature = "llzig")]
pub use llzig::LLZig;

mod local;
mod lower;
pub use lower::HugeId;
mod trees;
pub use trees::TreeId;

use core::fmt;
use core::mem::align_of;

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

/// The general interface of the allocator implementations.
pub trait Alloc<'a>: Sized + Sync + Send + fmt::Debug {
    /// Return the name of the allocator.
    #[cold]
    fn name() -> &'static str;

    /// Initialize the allocator for `frames`.
    ///
    /// The `init` parameter defines how the allocator should be initialized,
    /// e.g. if it should try recovering from the persistent memory or
    /// just mark all frames as free or allocated.
    ///
    /// The `tiering` config define how many local trees we have for each tier
    /// and the `policy` for accessing them.
    ///
    /// The `meta` data contains buffers for the data structures of the allocator.
    /// It has to outlive the allocator and must be properly aligned and sized (`metadata_size`).
    #[cold]
    fn new(frames: usize, init: Init, tiering: &Tiering, meta: MetaData<'a>) -> Result<Self>;

    /// Returns the size of the metadata buffers required for initialization.
    #[cold]
    fn metadata_size(tiering: &Tiering, frames: usize) -> MetaSize;
    /// Returns the metadata buffers.
    ///
    /// # Safety
    /// These buffers must must not be freed or used for other purposes.
    #[cold]
    unsafe fn metadata(&mut self) -> MetaData<'a>;

    /// Allocate a new frame of `order` on the given `local`.
    /// If specified try allocating the given `frame`.
    fn get(&self, frame: Option<FrameId>, flags: Request) -> Result<(Tier, FrameId)>;
    /// Free the `frame` of `order` on the given `local`.
    fn put(&self, frame: FrameId, flags: Request) -> Result<()>;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;

    /// Quickly retrieve tree statistics.
    fn tree_stats(&self, tiers: &mut [TierStats]) -> TreeStats;

    /// Retrieve detailed allocator statistics.
    /// Takes more time than `tree_stats`.
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

    /// Change the tree matching `matching` to `change`.
    /// This can be used for promotion/demotion or offlining.
    ///
    /// Fails if the tree does not match or is currently reserved.
    fn change_tree(&self, _matcher: TreeMatch, _change: TreeChange) -> Result<()> {
        Err(Error::Memory)
    }

    /// Validate the internal state
    #[cold]
    fn validate(&self) {}
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameId(pub usize);
impl FrameId {
    pub const fn into_bits(self) -> u64 {
        self.0 as u64
    }
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits as usize)
    }

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
    pub const fn is_aligned(self, order: usize) -> bool {
        self.0 & ((1 << order) - 1) == 0
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
        Self {
            local: util::aligned_buf(m.local),
            trees: util::aligned_buf(m.trees),
            lower: util::aligned_buf(m.lower),
        }
    }
}

impl fmt::Debug for MetaData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetaData")
            .field("local", &self.local.as_ptr_range())
            .field("trees", &self.trees.as_ptr_range())
            .field("lower", &self.lower.as_ptr_range())
            .finish()
    }
}

#[derive(Debug)]
/// Defines the tiers and policy for the allocator.
pub struct Tiering<'a> {
    /// Specifies the tiers and number of local reservations for each tier.
    pub tiers: &'a [(Tier, usize)],
    /// Default tier for initialization or if a tree becomes completely free.
    pub default: Tier,
    /// Policy for accessing a `target` tree with `free` frames.
    pub policy: PolicyFn,
}
pub type PolicyFn = fn(requested: Tier, target: Tier, free: usize) -> Policy;

impl<'a> Tiering<'a> {
    /// Simple policy with two tiers, `0` for small and `1` for huge frames, and local reservations for each core.
    #[cfg(feature = "std")]
    pub fn simple(cores: usize) -> (Self, impl Fn(usize, usize) -> Request) {
        let tiers = vec![
            (Tier(0), cores), // small frames
            (Tier(1), cores), // huge frames
        ];

        fn policy(requested: Tier, target: Tier, free: usize) -> Policy {
            if requested.0 > target.0 {
                return Policy::Steal;
            } else if requested.0 < target.0 {
                return Policy::Demote;
            }
            match free {
                f if f >= TREE_FRAMES / 2 => Policy::Match(1), // half free
                f if f >= TREE_FRAMES / 64 => Policy::Match(u8::MAX), // almost allocated
                _ => Policy::Match(0), // low free count -> causes frequent reservations
            }
        }

        fn request(order: usize, core: usize, cores: usize) -> Request {
            if order >= HUGE_ORDER {
                Request::new(order, Tier(1), Some(core % cores + cores))
            } else {
                Request::new(order, Tier(0), Some(core % cores))
            }
        }

        (
            Self {
                tiers: tiers.leak(),
                default: Tier(0),
                policy,
            },
            move |order, core| request(order, core, cores),
        )
    }

    /// Policy with three tiers, `0` for small immovable frames,
    /// `1` for small movable frames and `2` for huge frames,
    /// and local reservations for each core and tier.
    #[cfg(feature = "std")]
    pub fn movable(cores: usize) -> (Self, impl Fn(usize, usize, bool) -> Request) {
        let tiers = vec![
            (Tier(0), cores), // immovable frames
            (Tier(1), cores), // movable frames
            (Tier(2), cores), // huge frames
        ];

        fn policy(requested: Tier, target: Tier, free: usize) -> Policy {
            if requested.0 > target.0 {
                return Policy::Steal;
            } else if requested.0 < target.0 {
                return Policy::Demote;
            }
            match free {
                f if f >= TREE_FRAMES / 2 => Policy::Match(1), // half free
                f if f >= TREE_FRAMES / 64 => Policy::Match(u8::MAX), // almost allocated
                _ => Policy::Match(2), // low free count -> causes frequent reservations
            }
        }

        fn request(order: usize, core: usize, cores: usize, movable: bool) -> Request {
            if order >= HUGE_ORDER {
                Request::new(order, Tier(2), Some(core % cores + 2 * cores))
            } else if movable {
                Request::new(order, Tier(1), Some(core % cores + cores))
            } else {
                Request::new(order, Tier(0), Some(core % cores))
            }
        }

        (
            Self {
                tiers: tiers.leak(),
                default: Tier(0),
                policy,
            },
            move |order, core, movable| request(order, core, cores, movable),
        )
    }
}

/// Policy for accessing a `target` tree with `free` frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Match, where higher is better, with [u8::MAX] as perfect match and 0 as bad match.
    /// The allocated frame will have the `target` tier (usually the same as `requested`).
    Match(u8),
    /// Can steal from the `target` tree, but does not demote it.
    /// The allocated frame will have the `target` tier (self-demotion).
    Steal,
    /// Would demote the `target` tree to the `requested` tier.
    /// The allocated frame will have the `requested` tier.
    Demote,
    /// Can't be used.
    Invalid,
}

/// Opaque tier of a tree
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
    /// Try recovering all frames from persistent memory, reinitialize the trees and children.
    Recover,
    /// Assume that the allocator is already initialized
    None,
}

/// A request for memory allocation.
#[derive(Debug, Clone, Copy)]
pub struct Request {
    /// Allocation order (#frames = 1 << order)
    pub order: usize,
    /// The requested tier.
    pub tier: Tier,
    /// The local reservation index or None for global allocation.
    pub local: Option<usize>,
}
impl Request {
    pub fn new(order: usize, tier: Tier, local: Option<usize>) -> Self {
        Self { order, tier, local }
    }
}
impl Request {
    const fn frames(&self) -> usize {
        1 << self.order
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

#[derive(Debug, Default)]
pub struct TreeStats {
    /// Number of total free frames
    pub free_frames: usize,
    /// Number of free trees
    pub free_trees: usize,
}
pub struct TierStats {
    /// Number of free frames
    pub free: usize,
    /// Number of allocated frames
    pub alloc: usize,
}

/// Match a tree for `change_tree`
#[derive(Debug, Clone, Default)]
pub struct TreeMatch {
    /// Match a specific tree
    pub id: Option<TreeId>,
    /// Match a specific tier
    pub tier: Option<Tier>,
    /// Require at least `free` frames in the tree
    pub free: usize,
}

/// Change for `change_tree`
#[derive(Debug, Clone)]
pub struct TreeChange {
    /// Change the tier
    pub tier: Option<Tier>,
    /// Transform the tree
    pub operation: Option<TreeOperation>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeOperation {
    /// Offline the tree, i.e. make it unavailable for allocation.
    Online,
    /// Online the tree, i.e. make it available for allocation.
    Offline,
}
