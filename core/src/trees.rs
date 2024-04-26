use core::mem::{align_of, size_of};
use core::ops::RangeInclusive;
use core::{fmt, slice};

use crate::atomic::Atom;
use crate::entry::{Kind, LocalTree, Tree};
use crate::util::{align_down, size_of_slice, Align};
use crate::{Error, Flags, Result, HUGE_FRAMES, TREE_FRAMES};

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

    /// Initialize the tree array
    pub fn new<F: Fn(usize) -> (usize, usize)>(
        frames: usize,
        buffer: &'a mut [u8],
        free_in_tree: F,
    ) -> Self {
        assert!(buffer.len() >= Self::metadata_size(frames));

        let len = frames.div_ceil(TREE_FRAMES);
        let entries: &mut [Atom<Tree>] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) };

        for (i, e) in entries.iter_mut().enumerate() {
            let (frames, huge) = free_in_tree(i * TREE_FRAMES);
            *e = Atom::new(Tree::with(frames, huge, false, Kind::Fixed));
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
                let tree = LocalTree::new(i * TREE_FRAMES, entry.free(), entry.huge());
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
