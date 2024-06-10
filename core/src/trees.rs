use core::mem::{align_of, size_of};
use core::ops::{RangeBounds, RangeInclusive};
use core::sync::atomic::AtomicU32;
use core::{fmt, slice};

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};
use crate::local::LocalTree;
use crate::util::{align_down, size_of_slice, Align};
use crate::{Error, Flags, Result, HUGE_FRAMES, HUGE_ORDER, TREE_FRAMES, TREE_HUGE};

#[derive(Default)]
pub struct Trees<'a> {
    /// Array of level 3 entries, which are the roots of the trees
    pub entries: &'a [Atom<Tree>],
}

impl<'a> fmt::Debug for Trees<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut free = 0;
        let mut partial = 0;
        for e in self.entries {
            let f = e.load().free();
            if f == TREE_FRAMES {
                free += 1;
            } else if f > Self::MIN_FREE {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, free: {free}, partial: {partial})")?;
        Ok(())
    }
}

impl<'a> Trees<'a> {
    pub const MIN_FREE: usize = TREE_FRAMES / 16;

    pub fn metadata_size(frames: usize) -> usize {
        // Event thought the elements are not cache aligned, the whole array should be
        size_of_slice::<Atom<Tree>>(frames.div_ceil(TREE_FRAMES))
            .next_multiple_of(align_of::<Align>())
    }

    pub fn metadata(&mut self) -> &'a mut [u8] {
        let len = Self::metadata_size(self.len() * TREE_FRAMES);
        unsafe { slice::from_raw_parts_mut(self.entries.as_ptr().cast_mut().cast(), len) }
    }

    /// Initialize the tree array
    pub fn new<F: Fn(usize) -> (usize, usize)>(
        frames: usize,
        buffer: &'a mut [u8],
        tree_init: Option<F>,
    ) -> Self {
        assert!(buffer.len() >= Self::metadata_size(frames));

        let len = frames.div_ceil(TREE_FRAMES);
        let entries: &mut [Atom<Tree>] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) };

        if let Some(tree_init) = tree_init {
            for (i, e) in entries.iter_mut().enumerate() {
                let (frames, huge) = tree_init(i * TREE_FRAMES);
                *e = Atom::new(Tree::with(frames, huge, false, Kind::Fixed));
            }
        }

        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&self, i: usize) -> Tree {
        self.entries[i].load()
    }

    /// Return the number of entirely free trees
    pub fn free(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.load().free() == TREE_FRAMES)
            .count()
    }
    /// Return the total sum of the tree counters
    pub fn free_frames(&self) -> usize {
        self.entries.iter().map(|e| e.load().free()).sum()
    }
    /// Return the total sum of the huge counters
    pub fn free_huge(&self) -> usize {
        self.entries.iter().map(|e| e.load().huge()).sum()
    }
    /// Sync with the global tree, stealing its counters
    pub fn sync(&self, i: usize, min: usize, min_huge: usize) -> Option<Tree> {
        self.entries[i]
            .fetch_update(|e| e.sync_steal(min, min_huge))
            .ok()
    }

    /// Increment or reserve the tree
    pub fn inc_or_reserve(
        &self,
        i: usize,
        free: usize,
        huge: usize,
        may_reserve: bool,
    ) -> Option<Tree> {
        let mut reserved = false;
        let tree = self.entries[i]
            .fetch_update(|v| {
                let v = v.inc(free, huge);
                if may_reserve && !v.reserved() && v.free() > Self::MIN_FREE {
                    // Reserve the tree that was targeted by the last N frees
                    reserved = true;
                    Some(v.with_free(0).with_huge(0).with_reserved(true))
                } else {
                    reserved = false; // <- This one is very important if CAS fails!
                    Some(v)
                }
            })
            .unwrap();

        if reserved {
            Some(tree)
        } else {
            None
        }
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: usize, free: usize, huge: usize, kind: Kind) {
        self.entries[i]
            .fetch_update(|v| v.unreserve_add(free, huge, kind))
            .expect("Unreserve failed");
    }

    /// Find and reserve a free tree
    pub fn reserve_matching(
        &self,
        start: usize,
        flags: Flags,
        offset: usize,
        len: usize,
        free: RangeInclusive<usize>,
        mut get_lower: impl FnMut(LocalTree, Flags) -> Result<LocalTree>,
    ) -> Result<LocalTree> {
        // There has to be enough space for the current allocation
        let free = (1 << flags.order()).max(*free.start())..=*free.end();
        let min_huge = (1 << flags.order()) / HUGE_FRAMES;

        let start = (start + self.entries.len()) as isize;
        for i in offset as isize..len as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) =
                self.entries[i].fetch_update(|v| v.reserve(free.clone(), min_huge, flags.into()))
            {
                let tree = LocalTree::with(i * TREE_FRAMES, entry.free(), entry.huge());
                match get_lower(tree, flags) {
                    Ok(tree) => return Ok(tree),
                    Err(Error::Memory) => {
                        self.unreserve(i, entry.free(), entry.huge(), flags.into())
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        Err(Error::Memory)
    }

    /// Reserves a new tree, prioritizing partially filled trees.
    pub fn reserve(
        &self,
        cores: usize,
        start: usize,
        flags: Flags,
        get_lower: impl FnMut(LocalTree, Flags) -> Result<LocalTree> + Copy,
    ) -> Result<LocalTree> {
        const CACHELINE: usize = align_of::<Align>() / size_of::<Tree>();
        let start = align_down(start, CACHELINE);

        // Search near trees
        let near = (self.len() / cores / 4).clamp(CACHELINE / 4, CACHELINE * 2);

        // Over half filled trees
        let half = TREE_FRAMES / 16..=TREE_FRAMES / 2;
        match self.reserve_matching(start, flags, 1, near, half, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Partially filled trees
        let partial = TREE_FRAMES / 64..=TREE_FRAMES - TREE_FRAMES / 16;
        match self.reserve_matching(start, flags, 1, 2 * near, partial, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Not free trees
        match self.reserve_matching(start, flags, 1, self.len(), 0..=TREE_FRAMES - 1, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Any tree
        self.reserve_matching(start, flags, 0, self.len(), 0..=TREE_FRAMES, get_lower)
    }

    #[allow(unused)]
    pub fn dump(&'a self) -> TreeDbg<'a> {
        TreeDbg(self)
    }
}

pub struct TreeDbg<'a>(&'a Trees<'a>);
impl fmt::Debug for TreeDbg<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for entry in self.0.entries {
            writeln!(f, "    {:?}", entry.load())?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Tree entry
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct Tree {
    /// Number of free 4K frames.
    #[bits(13)]
    pub free: usize,
    /// Number of free 4K frames.
    #[bits(4)]
    pub huge: usize,
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
    /// Are the frames movable?
    #[bits(2)]
    pub kind: Kind,
    #[bits(12)]
    __: (),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Huge,
    Movable,
    Fixed,
}

impl Kind {
    pub const LEN: usize = 3;

    const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Huge,
            1 => Self::Movable,
            2 => Self::Fixed,
            _ => unreachable!(),
        }
    }
    const fn into_bits(self) -> u8 {
        match self {
            Self::Huge => 0,
            Self::Movable => 1,
            Self::Fixed => 2,
        }
    }
}
impl From<Flags> for Kind {
    fn from(flags: Flags) -> Self {
        if flags.order() >= HUGE_ORDER {
            Self::Huge
        } else if flags.movable() {
            Self::Movable
        } else {
            Self::Fixed
        }
    }
}

const _: () = assert!(1 << Tree::FREE_BITS >= TREE_FRAMES);
const _: () = assert!(1 << Tree::HUGE_BITS >= TREE_HUGE);

impl Atomic for Tree {
    type I = AtomicU32;
}
impl Tree {
    /// Creates a new entry.
    pub fn with(free: usize, huge: usize, reserved: bool, kind: Kind) -> Self {
        assert!(free <= TREE_FRAMES && huge <= TREE_HUGE);
        Self::new()
            .with_free(free)
            .with_huge(huge)
            .with_reserved(reserved)
            .with_kind(kind)
    }
    /// Increments the free frames counter.
    pub fn inc(self, free: usize, huge: usize) -> Self {
        let free = self.free() + free;
        let huge = self.huge() + huge;
        assert!(free <= TREE_FRAMES && huge <= TREE_HUGE);
        self.with_free(free).with_huge(huge)
    }
    /// Reserves this entry if its frame count is in `range`.
    pub fn reserve(
        self,
        free: impl RangeBounds<usize>,
        min_huge: usize,
        kind: Kind,
    ) -> Option<Self> {
        if !self.reserved()
            && free.contains(&self.free())
            && self.huge() >= min_huge
            && (kind == self.kind() || self.free() == TREE_FRAMES)
        {
            Some(Self::with(0, 0, true, kind))
        } else {
            None
        }
    }
    /// Add the frames from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    pub fn unreserve_add(self, free: usize, huge: usize, kind: Kind) -> Option<Self> {
        if self.reserved() {
            let free = self.free() + free;
            let huge = self.huge() + huge;
            assert!(free <= TREE_FRAMES && huge <= TREE_HUGE);
            Some(Self::with(free, huge, false, kind))
        } else {
            None
        }
    }
    /// Set the free counter to zero if it is large enough for synchronization
    pub fn sync_steal(self, min: usize, min_huge: usize) -> Option<Self> {
        if self.reserved() && self.free() > min && self.huge() >= min_huge {
            Some(self.with_free(0).with_huge(0))
        } else {
            None
        }
    }
}
