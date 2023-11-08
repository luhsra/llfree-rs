use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::{Index, RangeBounds};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info};

use crate::atomic::Atom;
use crate::entry::{ReservedTree, Tree};
use crate::util::{align_down, Align};
use crate::{Error, Result};

#[derive(Default)]
pub struct Trees<const LN: usize> {
    /// Array of level 3 entries, which are the roots of the trees
    entries: Box<[Atom<Tree>]>,
}

impl<const LN: usize> Index<usize> for Trees<LN> {
    type Output = Atom<Tree>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl<const LN: usize> fmt::Debug for Trees<LN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut free = 0;
        let mut partial = 0;
        for e in &*self.entries {
            let f = e.load().free();
            if f == LN {
                free += 1;
            } else if f > Self::MIN_FREE {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, free: {free}, partial: {partial})")?;
        Ok(())
    }
}

impl<const LN: usize> Trees<LN> {
    pub const MIN_FREE: usize = 1 << 10;
    pub const MAX_FREE: usize = LN - (1 << 10);

    /// Initialize the tree array
    pub fn new(frames: usize, free_all: bool) -> Self {
        let len = frames.div_ceil(LN);
        let mut entries = Vec::with_capacity(len);
        if free_all {
            entries.resize_with(len - 1, || Atom::new(Tree::new_with(LN, false)));
            // The last one might be cut off
            let max = ((frames - 1) % LN) + 1;
            entries.push(Atom::new(Tree::new_with(max, false)));
        } else {
            entries.resize_with(len, || Atom::new(Tree::new()));
        }
        Self {
            entries: entries.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Atom<Tree>> {
        self.entries.iter()
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, entry: ReservedTree, frames: usize) -> Result<()> {
        if !entry.has_start() {
            return Ok(());
        }

        let i = entry.start() / LN;
        let max = (frames - i * LN).min(LN);
        if let Ok(_) = self[i].fetch_update(|v| v.unreserve_add(entry.free(), max)) {
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Find and reserve a free tree
    pub fn reserve_far(
        &self,
        start: usize,
        free: impl RangeBounds<usize> + Clone,
        mut get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)>,
    ) -> Result<(ReservedTree, usize)> {
        // Just search linearly through the array
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(free.clone())) {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }
        Err(Error::Memory)
    }

    /// Find and reserve partial tree or one that is close
    fn reserve_partial(
        &self,
        cores: usize,
        start: usize,
        mut get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)>,
    ) -> Result<(ReservedTree, usize)> {
        // One quater of the per-CPU memory
        let near = ((self.entries.len() / cores) / 4).max(1) as isize;

        // Positive modulo and cacheline alignment
        const CACHELINE: usize = align_of::<Align>() / size_of::<Tree>();
        let start = align_down(start + self.entries.len(), CACHELINE) as isize;

        // Search the the array for a partially or entirely free tree
        // This speeds up the search drastically if many trees are free
        for i in 1..near {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(Self::MIN_FREE..)) {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }

        // Search the rest of the array for a partially but not entirely free tree
        for i in near..=self.entries.len() as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(Self::MIN_FREE..Self::MAX_FREE)) {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }
        Err(Error::Memory)
    }

    /// Reserves a new tree, prioritizing partially filled trees.
    pub fn reserve(
        &self,
        order: usize,
        cores: usize,
        start: usize,
        fragmented: bool,
        get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)> + Copy,
        drain: impl FnOnce() -> Result<()>,
    ) -> Result<(ReservedTree, usize)> {
        if fragmented {
            // search for a free tree
            match self.reserve_far(start, Self::MAX_FREE.., get_lower) {
                Err(Error::Memory) => {}
                r => return r,
            }
        } else {
            // search for a partially filled tree
            match self.reserve_partial(cores, start, get_lower) {
                Err(Error::Memory) => {}
                r => return r,
            }
        }
        // fallback to any tree
        match self.reserve_far(start, (1 << order).., get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // fallback to draining all reservations from other cores
        drain()?;
        self.reserve_far(start, (1 << order).., get_lower)
    }
}

impl<const LN: usize> From<Vec<Atom<Tree>>> for Trees<LN> {
    fn from(value: Vec<Atom<Tree>>) -> Self {
        Self {
            entries: value.into(),
        }
    }
}
