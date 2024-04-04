use core::mem::{align_of, size_of};
use core::ops::{Index, RangeBounds};
use core::{fmt, slice};

use log::{error, info};

use crate::atomic::Atom;
use crate::entry::{LocalTree, Tree};
use crate::util::{align_down, size_of_slice, Align};
use crate::{Error, Result};

#[derive(Default)]
pub struct Trees<'a, const LN: usize> {
    /// Array of level 3 entries, which are the roots of the trees
    entries: &'a [Atom<Tree>],
}

impl<'a, const LN: usize> Index<usize> for Trees<'a, LN> {
    type Output = Atom<Tree>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl<'a, const LN: usize> fmt::Debug for Trees<'a, LN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut free = 0;
        let mut partial = 0;
        for e in self.entries {
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

impl<'a, const LN: usize> Trees<'a, LN> {
    pub const MIN_FREE: usize = 1 << 11; //This is the 12.5% threshold that is descrobed in the paper, but for 16K bf
    pub const MAX_FREE: usize = LN - (1 << 11);

    pub fn metadata_size(frames: usize) -> usize {
        // Event thought the elements are not cache aligned, the whole array should be
        size_of_slice::<Atom<Tree>>(frames.div_ceil(LN)).next_multiple_of(align_of::<Align>())
    }

    /// Initialize the tree array
    pub fn new<F: Fn(usize) -> usize>(
        frames: usize,
        buffer: &'a mut [u8],
        free_in_tree: F,
    ) -> Self {
        assert!(buffer.len() >= Self::metadata_size(frames));

        let len = frames.div_ceil(LN);
        let entries: &mut [Atom<Tree>] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) };

        for (i, e) in entries.iter_mut().enumerate() {
            let frames = free_in_tree(i * LN);
            *e = Atom::new(Tree::with(frames, false));
        }

        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Atom<Tree>> {
        self.entries.iter()
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: usize, free: usize) {
        if let Err(t) = self[i].fetch_update(|v| v.unreserve_add(free, LN)) {
            error!("Unreserve failed i{i} {t:?} + {free}");
            panic!()
        }
    }

    /// Find and reserve a free tree
    pub fn reserve_far(
        &self,
        start: usize,
        free: impl RangeBounds<usize> + Clone,
        mut get_lower: impl FnMut(LocalTree) -> Result<LocalTree>,
    ) -> Result<LocalTree> {
        // Just search linearly through the array
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(free.clone())) {
                match get_lower(LocalTree {
                    frame: i * LN,
                    free: entry.free(),
                }) {
                    Err(Error::Memory) => {
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .expect("Rollback failed");
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
        mut get_lower: impl FnMut(LocalTree) -> Result<LocalTree>,
    ) -> Result<LocalTree> {
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
                match get_lower(LocalTree {
                    frame: i * LN,
                    free: entry.free(),
                }) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .expect("Rollback failed");
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
                match get_lower(LocalTree {
                    frame: i * LN,
                    free: entry.free(),
                }) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .expect("Rollback failed");
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
        get_lower: impl FnMut(LocalTree) -> Result<LocalTree> + Copy,
        drain: impl FnOnce() -> Result<()>,
    ) -> Result<LocalTree> {
        // search for a partially filled tree
        match self.reserve_partial(cores, start, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
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
