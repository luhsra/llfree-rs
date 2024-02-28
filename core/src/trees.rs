use core::mem::{align_of, size_of};
use core::ops::RangeBounds;
use core::{fmt, slice};

use crate::atomic::Atom;
use crate::entry::{LocalTree, Tree};
use crate::util::{align_down, size_of_slice, Align};
use crate::{Error, Result};

#[derive(Default)]
pub struct Trees<'a, const LN: usize> {
    /// Array of level 3 entries, which are the roots of the trees
    entries: &'a [Atom<Tree>],
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
    pub const MIN_FREE: usize = 1 << 10;
    pub const MAX_FREE: usize = LN - (1 << 10);

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

    pub fn get(&self, i: usize) -> Tree {
        self.entries[i].load()
    }

    /// Return the total sum of the tree counters
    pub fn total_free(&self) -> usize {
        self.entries.iter().map(|e| e.load().free()).sum()
    }

    pub fn sync(&self, i: usize, min: usize) -> Option<usize> {
        match self.entries[i].fetch_update(|e| e.sync_steal(min)) {
            Ok(e) => Some(e.free()),
            Err(_) => None,
        }
    }

    /// Increment or reserve the tree
    pub fn inc_or_reserve(&self, i: usize, num_frames: usize, may_reserve: bool) -> Option<usize> {
        let mut reserved = false;
        let tree = self.entries[i]
            .fetch_update(|v| {
                let v = v.inc(num_frames, LN);
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
            Some(tree.free())
        } else {
            None
        }
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: usize, free: usize) {
        self.entries[i]
            .fetch_update(|v| v.unreserve_add(free, LN))
            .expect("Unreserve failed");
    }

    /// Find and reserve a free tree
    pub fn reserve_far(
        &self,
        start: usize,
        free: impl RangeBounds<usize> + Clone,
        mut get_lower: impl FnMut(usize) -> Result<usize>,
    ) -> Result<LocalTree> {
        // Just search linearly through the array
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self.entries[i].fetch_update(|v| v.reserve(free.clone())) {
                match get_lower(i * LN) {
                    Ok(frame) => return Ok(LocalTree::new(frame, entry.free() as _)),
                    Err(Error::Memory) => self.unreserve(i, entry.free()),
                    Err(e) => return Err(e),
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
        mut get_lower: impl FnMut(usize) -> Result<usize>,
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
            if let Ok(entry) = self.entries[i].fetch_update(|v| v.reserve(Self::MIN_FREE..)) {
                assert!(!entry.reserved());
                match get_lower(i * LN) {
                    Ok(frame) => return Ok(LocalTree::new(frame, entry.free() as _)),
                    Err(Error::Memory) => self.unreserve(i, entry.free()),
                    Err(e) => return Err(e),
                }
            }
        }

        // Search the rest of the array for a partially but not entirely free tree
        for i in near..=self.entries.len() as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) =
                self.entries[i].fetch_update(|v| v.reserve(Self::MIN_FREE..Self::MAX_FREE))
            {
                assert!(!entry.reserved());
                match get_lower(i * LN) {
                    Ok(frame) => return Ok(LocalTree::new(frame, entry.free() as _)),
                    Err(Error::Memory) => self.unreserve(i, entry.free()),
                    Err(e) => return Err(e),
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
        get_lower: impl FnMut(usize) -> Result<usize> + Copy,
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
