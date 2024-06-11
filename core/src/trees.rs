use core::mem::align_of;
use core::ops::RangeBounds;
use core::sync::atomic::AtomicU16;
use core::{fmt, slice};

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};
use crate::util::{size_of_slice, Align};
use crate::{Error, Flags, Result, HUGE_ORDER, TREE_FRAMES};

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
    pub fn new(
        frames: usize,
        buffer: &'a mut [u8],
        tree_init: Option<impl Fn(usize) -> usize>,
    ) -> Self {
        assert!(buffer.len() >= Self::metadata_size(frames));

        let len = frames.div_ceil(TREE_FRAMES);
        let entries: &mut [Atom<Tree>] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) };

        if let Some(tree_init) = tree_init {
            for (i, e) in entries.iter_mut().enumerate() {
                let frames = tree_init(i * TREE_FRAMES);
                *e = Atom::new(Tree::with(frames, false, Kind::Fixed));
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
    /// Sync with the global tree, stealing its counters
    pub fn sync(&self, i: usize, min: usize) -> Option<Tree> {
        self.entries[i].fetch_update(|e| e.sync_steal(min)).ok()
    }

    /// Increment or reserve the tree
    pub fn inc_or_reserve(&self, i: usize, free: usize, may_reserve: bool) -> Option<Tree> {
        let mut reserved = false;
        let tree = self.entries[i]
            .fetch_update(|v| {
                let v = v.inc(free);
                if may_reserve && !v.reserved() && v.free() > Self::MIN_FREE {
                    // Reserve the tree that was targeted by the last N frees
                    reserved = true;
                    Some(v.with_free(0).with_reserved(true))
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
    pub fn unreserve(&self, i: usize, free: usize, kind: Kind) {
        self.entries[i]
            .fetch_update(|v| v.unreserve_add(free, kind))
            .expect("Unreserve failed");
    }

    /// Iterate through all trees as long `f` returns `Error::Memory`
    pub fn search(
        &self,
        start: usize,
        offset: usize,
        len: usize,
        mut f: impl FnMut(usize) -> Result<usize> + Clone,
    ) -> Result<usize> {
        // There has to be enough space for the current allocation
        let start = (start + self.entries.len()) as isize;
        for i in offset as isize..len as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            match f(i) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }
        Err(Error::Memory)
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
#[bitfield(u16)]
#[derive(PartialEq, Eq)]
pub struct Tree {
    /// Number of free 4K frames.
    #[bits(13)]
    pub free: usize,
    /// If this subtree is reserved by a CPU.
    pub reserved: bool,
    /// Are the frames movable?
    #[bits(2)]
    pub kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Fixed,
    Movable,
    Huge,
}

impl Kind {
    pub const LEN: usize = 3;

    pub fn accepts(self, other: Kind) -> bool {
        (other as usize) >= (self as usize)
    }

    const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Fixed,
            1 => Self::Movable,
            2 => Self::Huge,
            _ => Self::Fixed,
        }
    }
    const fn into_bits(self) -> u8 {
        match self {
            Self::Fixed => 0,
            Self::Movable => 1,
            Self::Huge => 2,
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
impl From<usize> for Kind {
    fn from(value: usize) -> Self {
        Self::from_bits(value as _)
    }
}

const _: () = assert!(1 << Tree::FREE_BITS >= TREE_FRAMES);

impl Atomic for Tree {
    type I = AtomicU16;
}
impl Tree {
    /// Creates a new entry.
    pub fn with(free: usize, reserved: bool, kind: Kind) -> Self {
        assert!(free <= TREE_FRAMES);
        Self::new()
            .with_free(free)
            .with_reserved(reserved)
            .with_kind(kind)
    }
    /// Increments the free frames counter.
    pub fn inc(self, free: usize) -> Self {
        let free = self.free() + free;
        assert!(free <= TREE_FRAMES);
        self.with_free(free)
    }
    pub fn dec_force(mut self, free: usize, kind: Kind) -> Option<Self> {
        if !self.reserved() && self.free() >= free {
            if !self.kind().accepts(kind) {
                self.set_kind(kind);
            }
            Some(self.with_free(self.free() - free))
        } else {
            None
        }
    }

    /// Reserves this entry if its frame count is in `range`.
    pub fn reserve(self, free: impl RangeBounds<usize>, kind: Kind) -> Option<Self> {
        if !self.reserved()
            && free.contains(&self.free())
            && (kind == self.kind() || self.free() == TREE_FRAMES)
        {
            Some(Self::with(0, true, kind))
        } else {
            None
        }
    }
    /// Add the frames from the `other` entry to the reserved `self` entry and unreserve it.
    /// `self` is the entry in the global array / table.
    pub fn unreserve_add(self, free: usize, kind: Kind) -> Option<Self> {
        if self.reserved() {
            let free = self.free() + free;
            assert!(free <= TREE_FRAMES);
            Some(Self::with(free, false, kind))
        } else {
            None
        }
    }
    /// Set the free counter to zero if it is large enough for synchronization
    pub fn sync_steal(self, min: usize) -> Option<Self> {
        if self.reserved() && self.free() > min {
            Some(self.with_free(0))
        } else {
            None
        }
    }
}
