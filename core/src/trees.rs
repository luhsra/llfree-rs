use core::mem::{align_of, size_of};
use core::ops::RangeInclusive;
use core::{fmt, slice};

use crate::atomic::Atom;
use crate::entry::{LocalTree, Tree};
use crate::util::{align_down, size_of_slice, Align};
use crate::{Error, Flags, Result};

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
            *e = Atom::new(Tree::with(frames, false, false));
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
    pub fn inc_or_reserve(&self, i: usize, num_frames: usize, may_reserve: bool) -> Option<Tree> {
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
            Some(tree)
        } else {
            None
        }
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    pub fn unreserve(&self, i: usize, free: usize, movable: bool) {
        self.entries[i]
            .fetch_update(|v| v.unreserve_add(free, LN, movable))
            .expect("Unreserve failed");
    }

    pub fn reserve_best(
        &self,
        cores: usize,
        start: usize,
        flags: Flags,
        mut get_lower: impl FnMut(usize, Flags) -> Result<usize>,
    ) -> Result<LocalTree> {
        const CACHELINE: usize = align_of::<Align>() / size_of::<Tree>();
        let near = (self.len() / cores / 4).clamp(CACHELINE / 2, CACHELINE * 4);

        let mut best = [None, None];

        fn better(a: Tree, b: Option<LocalTree>, order: usize) -> bool {
            let min_pages = 4.max(2 << order);
            if a.free() < min_pages {
                return false;
            }
            match b {
                Some(b) => a.free() < b.free as usize,
                None => true,
            }
        }

        let mut total_free = 0;
        let start = (start + self.entries.len()) as isize;
        for j in 1..=align_down(self.len() - 1, near) as isize {
            // Alternating between before and after this entry
            let off = if j % 2 == 0 { j / 2 } else { -j.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();

            let tree = self.entries[i].load();
            total_free += tree.free();

            if better(tree, best[0], flags.order()) {
                best[1] = best[0];
                best[0] = Some(LocalTree::new(i * LN, tree.free() as _));
            } else if better(tree, best[1], flags.order()) {
                best[1] = Some(LocalTree::new(i * LN, tree.free() as _));
            }

            if !(j as usize % near == 0) {
                continue;
            }

            // Try allocate best trees
            let average_free = total_free / j as usize;
            let max_free = average_free; // + LN / (self.len() / (j + 1) as usize);
            for tree in &best {
                if let Some(tree) = tree
                    && tree.free <= max_free as u16
                {
                    let i = tree.frame / LN;
                    if let Ok(entry) = self.entries[i].fetch_update(|v| {
                        v.reserve(1 << flags.order()..max_free, LN, flags.movable())
                    }) {
                        match get_lower(i * LN, flags) {
                            Ok(frame) => return Ok(LocalTree::new(frame, entry.free() as _)),
                            Err(Error::Memory) => self.unreserve(i, entry.free(), flags.movable()),
                            Err(e) => return Err(e),
                        }
                    }
                }
            }
            // Allocation failed, reset best
            best = [None, None];
        }

        Err(Error::Memory)
    }

    /// Find and reserve a free tree
    pub fn reserve_matching(
        &self,
        start: usize,
        flags: Flags,
        offset: usize,
        len: usize,
        free: RangeInclusive<usize>,
        mut get_lower: impl FnMut(usize, Flags) -> Result<usize>,
    ) -> Result<LocalTree> {
        // There has to be enough space for the current allocation
        let free = (1 << flags.order()).max(*free.start())..=*free.end();

        let start = (start + self.entries.len()) as isize;
        for i in offset as isize..len as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) =
                self.entries[i].fetch_update(|v| v.reserve(free.clone(), LN, flags.movable()))
            {
                match get_lower(i * LN, flags) {
                    Ok(frame) => return Ok(LocalTree::new(frame, entry.free() as _)),
                    Err(Error::Memory) => self.unreserve(i, entry.free(), flags.movable()),
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
        get_lower: impl FnMut(usize, Flags) -> Result<usize> + Copy,
        steal: impl FnOnce(Flags) -> Result<LocalTree>,
    ) -> Result<LocalTree> {
        const CACHELINE: usize = align_of::<Align>() / size_of::<Tree>();
        let start = align_down(start, CACHELINE);

        // Search near trees
        let near = (self.len() / cores / 4).clamp(CACHELINE / 4, CACHELINE * 2);

        // Over half filled trees
        let half = (4 << flags.order()).max(LN / 16)..=LN / 2;
        match self.reserve_matching(start, flags, 1, near, half, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Partially filled trees
        let partial = (2 << flags.order()).max(LN / 64)..=LN - LN / 16;
        match self.reserve_matching(start, flags, 1, 2 * near, partial, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Not free trees
        match self.reserve_matching(start, flags, 1, 4 * near, 0..=LN - 4, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // Any tree
        match self.reserve_matching(start, flags, 0, self.len(), 0..=LN, get_lower) {
            Err(Error::Memory) => {}
            r => return r,
        }
        // steal from another core
        steal(flags)
    }
}
