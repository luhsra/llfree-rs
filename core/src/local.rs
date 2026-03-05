use core::mem::size_of;
use core::sync::atomic::AtomicU64;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::debug;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::util::size_of_slice;
use crate::{Error, Tier, TierConfig, KindStats, Policy, TreeStats};
use crate::{TREE_FRAMES, TreeId};

pub struct Locals<'a> {
    /// Local reservations for each [Kind]
    local: &'a [Local],
}

impl fmt::Debug for Locals<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.local.fmt(f)
    }
}

impl<'a> Locals<'a> {
    pub fn metadata_size(kinds: &[TierConfig]) -> usize {
        size_of_slice::<Local>(kinds.iter().map(|k| k.count as usize).sum())
    }
    pub unsafe fn metadata(&mut self) -> &'a mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self.local.as_ptr().cast_mut().cast(),
                size_of_slice::<Local>(self.local.len()),
            )
        }
    }

    pub fn new(buffer: &'a mut [u8], kinds: &[TierConfig]) -> Result<Self, Error> {
        let len = buffer.len() / size_of::<Local>().next_multiple_of(align_of::<Local>());
        let kind_len = kinds.iter().map(|k| k.count as usize).sum::<usize>();
        if len < kind_len {
            return Err(Error::Initialization);
        }
        let local: &mut [Local] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), kind_len) };

        let mut offset = 0;
        for TierConfig {
            tier: kind,
            count,
            reserve,
        } in kinds
        {
            for local in &mut local[offset..][..*count as usize] {
                *local = Local {
                    kind: *kind,
                    reserve: *reserve,
                    preferred: Atom::new(LocalTree::none()),
                    #[cfg(feature = "free_reserve")]
                    frees: Atom::new(FreeHistory::default()),
                };
            }
            offset += *count as usize;
        }
        Ok(Self { local })
    }

    pub fn len(&self) -> usize {
        self.local.len()
    }

    pub fn can_get(&self, index: usize, tree: Option<TreeId>, free: usize) -> bool {
        let local = self.local[index].preferred.load();
        local.get(tree, free).is_some()
    }

    pub fn kind(&self, index: usize) -> Tier {
        self.local[index].kind
    }

    pub fn can_reserve(&self, index: usize) -> fn(Tier, usize) -> Policy {
        self.local[index].reserve
    }

    pub fn get(
        &self,
        index: usize,
        tree: Option<TreeId>,
        free: usize,
    ) -> Result<RowId, Option<(RowId, usize)>> {
        let local = &self.local[index];
        match local.preferred.fetch_update(|v| v.get(tree, free)) {
            Ok(old) => Ok(old.row()),
            Err(old) => Err(old.present().then_some((old.row(), old.free()))),
        }
    }

    /// Steal from another compatible local tree (no downgrade necessary)
    pub fn steal(&self, index: usize, tree: Option<TreeId>, free: usize) -> Option<(Tier, RowId)> {
        let kind = self.local[index].kind;
        // Steal from same kind first
        for i in 0..self.local.len() {
            let i = (index + i) % self.local.len();
            let local = &self.local[i];
            if local.kind == kind
                && let Ok(row) = self.get(i, tree, free)
            {
                return Some((kind, row));
            }
        }
        // Fallback to stealing kinds that would not demote
        for i in 1..self.local.len() {
            let i = (index + i) % self.local.len();
            let local = &self.local[i];
            if self.local[index].try_reserve(local.kind, free) == Policy::Stealing
                && let Ok(row) = self.get(i, tree, free)
            {
                return Some((local.kind, row));
            }
        }
        None
    }

    /// Find higher kind and steal and downgrade it to the current kind
    pub fn steal_downgrade(
        &self,
        index: usize,
        tree: Option<TreeId>,
        free: usize,
    ) -> Option<((Tier, RowId), Option<(RowId, usize)>)> {
        for i in 1..self.local.len() {
            let i = (index + i) % self.local.len();
            let other = &self.local[i];
            if self.local[index].try_reserve(other.kind, free) == Policy::Demotes
                && let Ok(old) = other
                    .preferred
                    .fetch_update(|v| v.get(tree, free).map(|_| LocalTree::none()))
            {
                let new = old.get(tree, free).unwrap();
                // Replace local tree and return its free count for unreservation
                let old = self.local[index].preferred.swap(new);
                return Some((
                    (other.kind, new.row()),
                    old.present().then_some((old.row(), old.free())),
                ));
            }
        }
        None
    }

    pub fn put(&self, index: usize, tree: TreeId, free: usize) -> bool {
        let local = &self.local[index];
        local.preferred.fetch_update(|v| v.put(tree, free)).is_ok()
    }

    pub fn swap(&self, index: usize, tree: TreeId, free: usize) -> Option<(RowId, usize)> {
        let local = &self.local[index];
        let old = local.preferred.swap(LocalTree::with(tree.as_row(), free));
        old.present().then_some((old.row(), old.free()))
    }

    pub fn drain(&self, index: usize) -> Option<(RowId, usize)> {
        let local = &self.local[index];
        let old = local.preferred.swap(LocalTree::none());
        old.present().then_some((old.row(), old.free()))
    }

    pub fn set_start(&self, index: usize, row: RowId) {
        let local = &self.local[index];
        let _ = local.preferred.fetch_update(|v| v.set_start(row, false));
    }

    #[allow(dead_code)]
    #[cfg(feature = "free_reserve")]
    pub fn frees_push(&self, index: usize, tree_idx: TreeId) -> bool {
        self.local[index].frees_push(tree_idx)
    }

    pub fn stats(&self, kinds: &mut [KindStats]) -> TreeStats {
        let mut s = TreeStats::default();
        for local in self.local {
            let preferred = local.preferred.load();
            if preferred.present() {
                s.free_frames += preferred.free();
                if preferred.free() != TREE_FRAMES
                    && let Some(kind) = kinds.get_mut(local.kind.0 as usize)
                {
                    kind.free += preferred.free();
                    kind.alloc += TREE_FRAMES - preferred.free();
                }
                s.free_trees += preferred.free() / TREE_FRAMES;
            }
        }
        s
    }

    pub fn load(&self, local: usize) -> Option<(RowId, usize)> {
        let preferred = self.local[local].preferred.load();
        preferred
            .present()
            .then_some((preferred.row(), preferred.free()))
    }
}

/// Core-local data
#[derive(Debug)]
#[repr(align(64))]
struct Local {
    /// Kind of the local tree
    kind: Tier,
    /// Reserve function
    reserve: fn(kind: Tier, free: usize) -> Policy,
    /// Reserved trees for each [Kind]
    preferred: Atom<LocalTree>,
    #[cfg(feature = "free_reserve")]
    /// Recent frees
    frees: Atom<FreeHistory>,
}

impl Local {
    /// Add a tree index to the history, returning if there are enough frees
    #[cfg(feature = "free_reserve")]
    fn frees_push(&self, tree_idx: TreeId) -> bool {
        let mut success = false;
        let _ = self.frees.fetch_update(|mut v| {
            success = v.push(tree_idx);
            (!success).then_some(v)
        });
        success
    }

    fn try_reserve(&self, other: Tier, free: usize) -> Policy {
        (self.reserve)(other, free)
    }
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
struct LocalTree {
    #[bits(48)]
    row: RowId,
    #[bits(15)]
    free: usize,
    /// Reserved for present bit...
    present: bool,
}
impl Atomic for LocalTree {
    type I = AtomicU64;
}
impl LocalTree {
    fn with(row: RowId, free: usize) -> Self {
        Self::new().with_row(row).with_free(free).with_present(true)
    }
    fn none() -> Self {
        Self::new().with_present(false)
    }
    fn get(self, tree: Option<TreeId>, free: usize) -> Option<Self> {
        if self.present() && tree.is_none_or(|i| self.row().as_tree() == i) {
            Some(self.with_free(self.free().checked_sub(free)?))
        } else {
            None
        }
    }
    fn put(self, tree: TreeId, free: usize) -> Option<Self> {
        if self.present() && self.row().as_tree() == tree {
            assert!(self.free() + free <= TREE_FRAMES);
            Some(self.with_free(self.free() + free))
        } else {
            None
        }
    }
    fn set_start(self, row: RowId, force: bool) -> Option<Self> {
        if force || (self.present() && self.row().as_tree() == row.as_tree()) {
            Some(self.with_row(row))
        } else {
            None
        }
    }
}

#[bitfield(u64)]
struct FreeHistory {
    #[bits(48)]
    idx: TreeId,
    #[bits(16)]
    counter: usize,
}

#[allow(dead_code)]
impl FreeHistory {
    /// Threshold for the number of frees after which a tree is reserved
    const F: usize = 4;

    /// Add a tree index to the history, returing if there are enough frees
    fn push(&mut self, tree_idx: TreeId) -> bool {
        debug!("Pushing {tree_idx:?} to {self:?}");
        if self.idx() == tree_idx {
            if self.counter() >= Self::F {
                return true;
            }
            self.set_counter(self.counter() + 1);
        } else {
            self.set_idx(tree_idx);
            self.set_counter(0);
        }
        false
    }
}
impl Atomic for FreeHistory {
    type I = AtomicU64;
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::{FrameId, util};

    use super::FreeHistory;

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        util::logging();
        let mut history = FreeHistory::default();
        let frame1 = FrameId(43);
        let i1 = frame1.as_tree();
        assert!(!history.push(i1), "{history:?}");
        assert!(!history.push(i1), "{history:?}");
        assert!(!history.push(i1), "{history:?}");
        assert!(!history.push(i1), "{history:?}");
        assert!(history.push(i1), "{history:?}");
        assert!(history.push(i1), "{history:?}");
        let frame2 = FrameId(512 * 512 + 43);
        let i2 = frame2.as_tree();
        assert_ne!(i1, i2);
        assert!(!history.push(i2), "{history:?}");
        assert!(!history.push(i2), "{history:?}");
        assert!(!history.push(i1), "{history:?}");
    }
}
