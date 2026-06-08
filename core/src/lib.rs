// Include readme as documentation
#![doc = include_str!("../../README.md")]
// Disable standard library
#![no_std]
// Don't warn for compile-time checks
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::redundant_pattern_matching)]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

pub mod atomic;
pub mod frame;
pub mod util;
pub mod wrapper;

mod bitfield;
use bitfield::RowId;
mod llfree;
pub use llfree::LLFree;

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
///
/// This is a classical accuracy vs speed tradeoff
pub const TREE_HUGE: usize = cfg_select! {
    feature = "tree_huge_1" => 1,
    feature = "tree_huge_2" => 2,
    feature = "tree_huge_4" => 4,
    feature = "tree_huge_8" => 8,
    feature = "tree_huge_16" => 16,
    feature = "tree_huge_32" => 32,
    feature = "tree_huge_64" => 64,
    feature = "tree_huge_128" => 128,
    feature = "tree_huge_256" => 256,
    feature = "tree_huge_512" => 512,
    _ => 4,
};
/// Number of small frames in tree
pub const TREE_FRAMES: usize = TREE_HUGE << HUGE_ORDER;
/// Order of an entire tree
pub const TREE_ORDER: usize = TREE_FRAMES.ilog2() as usize;
/// Order for huge frames
pub const HUGE_ORDER: usize = if cfg!(feature = "16K") { 11 } else { 9 };
/// Number of small frames in huge frame
pub const HUGE_FRAMES: usize = 1 << HUGE_ORDER;
/// Bit size of the atomic ints that comprise the bitfields
pub const BITFIELD_ROW: usize = 64;

/// Number of retries if an atomic operation fails.
const RETRIES: usize = 4;

/// Allocation error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    Memory = 1,
    /// Invalid argument
    Argument = 3,
    /// Allocator not initialized or initialization failed
    Initialization = 4,
}

/// Allocation result
pub type Result<T> = core::result::Result<T, Error>;

/// The general interface of the allocator implementation.
pub trait Alloc<'a>: Sized + Sync + Send + fmt::Debug {
    /// Return the name of the allocator.
    fn name() -> &'static str;

    /// Initialize the allocator for `frames`.
    ///
    /// The `init` parameter defines how the allocator should be initialized,
    /// e.g. if it should try recovering from the persistent memory or
    /// just mark all frames as free or allocated.
    ///
    /// The `classing` config defines how many local trees we have for each class
    /// and the `policy` for accessing them.
    ///
    /// The `meta` data contains buffers for the data structures of the allocator.
    /// It has to outlive the allocator and must be properly aligned and sized (`metadata_size`).
    fn new(frames: usize, init: Init, classing: &Classing, meta: MetaData<'a>) -> Result<Self>;

    /// Returns the size of the metadata buffers required for initialization.
    fn metadata_size(classing: &Classing, frames: usize) -> MetaSize;
    /// Returns the metadata buffers.
    ///
    /// # Safety
    /// These buffers must not be freed or used for other purposes.
    unsafe fn metadata(&mut self) -> MetaData<'a>;

    /// Allocate a new frame of `order` on the given `local`.
    /// If specified try allocating the given `frame`.
    fn get(&self, frame: Option<FrameId>, flags: Request) -> Result<(FrameId, Class)>;
    /// Free the `frame` of `order` on the given `local`.
    fn put(&self, frame: FrameId, flags: Request) -> Result<()>;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;

    /// Quickly retrieve tree statistics.
    fn tree_stats(&self) -> TreeStats;

    /// Retrieve detailed allocator statistics.
    /// Takes more time than `tree_stats`.
    fn stats(&self) -> Stats;
    /// Retrieve detailed allocator statistics.
    /// Only [`TREE_ORDER`], [`HUGE_ORDER`] and `0` are supported.
    fn stats_at(&self, frame: FrameId, order: usize) -> Stats;

    /// Unreserve a local tree
    fn drain(&self) {}

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

#[cfg(any(test, feature = "std"))]
impl MetaData<'_> {
    #[must_use]
    pub fn alloc(m: &MetaSize) -> Self {
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

/// Defines the classes and policy for the allocator.
pub struct Classing {
    /// Specifies the classes and number of local reservations for each class.
    classes_raw: [(Class, usize); 1 << Class::BITS],
    /// Number of classes in use, i.e. the length of the `classes` array.
    classes_len: usize,
    /// Default class for initialization or if a tree becomes completely free.
    pub default: Class,
    /// Policy for accessing a `target` tree with `free` frames.
    pub policy: PolicyFn,
}
pub type PolicyFn = fn(requested: Class, target: Class, free: usize) -> Policy;

impl fmt::Debug for Classing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Classing")
            .field("classes", &self.classes())
            .field("default", &self.default)
            .finish()
    }
}

impl Classing {
    pub fn new(classes: &[(Class, usize)], default: Class, policy: PolicyFn) -> Self {
        assert!(classes.len() <= 1 << Class::BITS);
        let mut classes_raw = [const { (Class(0), 0) }; 1 << Class::BITS];
        classes_raw[..classes.len()].copy_from_slice(classes);
        Self {
            classes_raw,
            classes_len: classes.len(),
            default,
            policy,
        }
    }

    pub fn classes(&self) -> &[(Class, usize)] {
        &self.classes_raw[..self.classes_len]
    }

    /// Simple policy with two classes, `0` for small and `1` for huge frames, and local reservations for each core.
    pub fn simple(cores: usize) -> (Self, impl Fn(usize, usize) -> Request) {
        let classes = [
            (Class(0), cores), // small frames
            (Class(1), cores), // huge frames
        ];

        fn policy(requested: Class, target: Class, free: usize) -> Policy {
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
            Request::new(order, Class((order >= HUGE_ORDER) as _), Some(core % cores))
        }

        (Self::new(&classes, Class(1), policy), move |order, core| {
            request(order, core, cores)
        })
    }

    /// Policy with three classes, `0` for small immovable frames,
    /// `1` for small movable frames and `2` for huge frames,
    /// and local reservations for each core and class.
    pub fn movable(cores: usize) -> (Self, impl Fn(usize, usize, bool) -> Request) {
        let classes = [
            (Class(0), cores), // immovable frames
            (Class(1), cores), // movable frames
            (Class(2), cores), // huge frames
        ];

        fn policy(requested: Class, target: Class, free: usize) -> Policy {
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
                Request::new(order, Class(2), Some(core % cores))
            } else if movable {
                Request::new(order, Class(1), Some(core % cores))
            } else {
                Request::new(order, Class(0), Some(core % cores))
            }
        }

        (
            Self::new(&classes, Class(2), policy),
            move |order, core, movable| request(order, core, cores, movable),
        )
    }
}

/// Policy for accessing a `target` tree with `free` frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Policy {
    /// Match, where higher is better, with [u8::MAX] as perfect match and 0 as bad match.
    /// The allocated frame will have the `target` class (usually the same as `requested`).
    Match(u8),
    /// Would demote the `target` tree to the `requested` class.
    /// The allocated frame will have the `requested` class.
    /// The `bool` indicates whether the `target` tree is entirely empty.
    Demote,
    /// Can steal from the `target` tree, but does not demote it.
    /// The allocated frame will have the `target` class (self-demotion).
    Steal,
    /// Can't be used.
    Invalid,
}

/// Opaque class of a tree
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Class(pub u8);
impl Class {
    pub const BITS: usize = 3;
    pub const LEN: u8 = 1 << Self::BITS;
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
    pub const fn into_bits(self) -> u8 {
        self.0
    }
}
impl fmt::Debug for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}", self.0)
    }
}

/// Defines whether the allocator should use persistent memory
/// and, in that case, whether it should try to recover prior state.
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
    /// The requested class.
    pub class: Class,
    /// The local reservation index for this class or None for global allocation.
    pub local: Option<usize>,
}
impl Request {
    pub fn new(order: usize, class: Class, local: Option<usize>) -> Self {
        Self {
            order,
            class,
            local,
        }
    }
}
impl Request {
    const fn frames(&self) -> usize {
        1 << self.order
    }
}

/// Allocation statistics of allocator
#[derive(Debug, Default)]
pub struct Stats {
    /// Number of free frames
    pub free_frames: usize,
    /// Number of entirely free huge frames
    pub free_huge: usize,
    /// Number of entirely free trees
    pub free_trees: usize,
}

/// Statistics about the trees of the allocator
#[derive(Debug, Default)]
pub struct TreeStats {
    /// Number of total free frames
    pub free_frames: usize,
    /// Number of free trees
    pub free_trees: usize,
    /// Stats per class
    pub classes: [ClassStats; 1 << Class::BITS],
}
/// Statistics about a class of the allocator
#[derive(Debug, Default)]
pub struct ClassStats {
    /// Number of free frames
    pub free_frames: usize,
    /// Number of allocated frames
    pub alloc_frames: usize,
}

/// Match a tree for `change_tree`
#[derive(Debug, Clone, Default)]
pub struct TreeMatch {
    /// Match a specific tree
    pub id: Option<TreeId>,
    /// Match a specific class
    pub class: Option<Class>,
    /// Require at least `free` frames in the tree
    pub free: usize,
}

/// Change for `change_tree`
#[derive(Debug, Clone)]
pub struct TreeChange {
    /// Change the class
    pub class: Option<Class>,
    /// Transform the tree
    pub operation: Option<TreeOperation>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeOperation {
    /// Online the tree, i.e. make it available for allocation.
    Online,
    /// Offline the tree, i.e. make it unavailable for allocation.
    Offline,
}

#[cfg(test)]
mod test {
    use log::warn;

    use crate::frame::Frame;
    use crate::util::logging;
    use crate::{Alloc, Classing, Init, LLFree, MetaData};

    #[test]
    fn minimal() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        let frames = MEM_SIZE / Frame::SIZE;

        warn!("init");
        // Specify classing policy
        let (classing, request) = Classing::simple(1);
        // Allocate the metadata buffers
        let ms = LLFree::metadata_size(&classing, frames);
        let meta = MetaData::alloc(&ms);
        // Initialize the allocator
        let alloc = LLFree::new(frames, Init::FreeAll, &classing, meta).unwrap();
        warn!("finit");
        assert_eq!(alloc.tree_stats().free_frames, alloc.frames());

        warn!("get >>>");
        let (frame1, _) = alloc.get(None, request(0, 0)).unwrap();
        warn!("get <<<");
        warn!("get >>>");
        let (frame2, _) = alloc.get(None, request(0, 0)).unwrap();
        warn!("get <<<");

        assert_eq!(alloc.stats().free_frames, alloc.frames() - 2);
        alloc.validate();

        warn!("put >>>");
        alloc.put(frame2, request(0, 0)).unwrap();
        warn!("put <<<");
        warn!("put >>>");
        alloc.put(frame1, request(0, 0)).unwrap();
        warn!("put <<<");
        alloc.validate();
    }
}
