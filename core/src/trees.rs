use core::mem::{align_of, size_of};
use core::ops::RangeInclusive;
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
    pub const MIN_FREE: usize = LN / 16;

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
    pub fn reserve_matching(
        &self,
        start: usize,
        order: usize,
        offset: usize,
        len: usize,
        free: RangeInclusive<usize>,
        mut get_lower: impl FnMut(usize, usize) -> Result<usize>,
    ) -> Result<LocalTree> {
        // There has to be enough space for the current allocation
        let free = (1 << order).max(*free.start())..=*free.end();

        let start = (start + self.entries.len()) as isize;
        for i in offset as isize..len as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) = self.entries[i].fetch_update(|v| v.reserve(free.clone())) {
                match get_lower(i * LN, order) {
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
        cores: usize,
        start: usize,
        order: usize,
        get_lower: impl FnMut(usize, usize) -> Result<usize> + Copy,
        steal: impl FnOnce(usize) -> Result<LocalTree>,
    ) -> Result<LocalTree> {
        const CACHELINE: usize = align_of::<Align>() / size_of::<Tree>();
        let start = align_down(start, CACHELINE);

        let half = (4 << order).max(LN / 32)..=LN / 2;
        let partial = (2 << order).max(LN / 128)..=LN - LN / 32;

        // Search near trees
        let near = (self.len() / cores / 4).clamp(CACHELINE / 4, CACHELINE * 2);

        // Over half filled trees
        match self.reserve_matching(start, order, 1, near, half.clone(), get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Partially filled trees
        match self.reserve_matching(start, order, 1, near, partial.clone(), get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Not free trees
        match self.reserve_matching(start, order, 1, near, 0..=LN - 4, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Any tree
        match self.reserve_matching(start, order, 1, near, 0..=LN, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }

        // Search globally

        // Over half filled trees
        match self.reserve_matching(start, order, near, self.len(), half, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Partially filled trees
        match self.reserve_matching(start, order, near, self.len(), partial, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Not free trees
        match self.reserve_matching(start, order, near, self.len(), 0..=LN - 4, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }

        // Any tree
        match self.reserve_matching(start, order, 0, self.len(), 0..=LN, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // steal from another core
        steal(order)
    }
}
