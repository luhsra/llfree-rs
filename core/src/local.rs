use core::sync::atomic::AtomicU64;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::debug;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::util::size_of_slice;
use crate::{Error, Policy, PolicyFn, Tier, Tiering, TreeStats};
use crate::{TREE_FRAMES, TreeId};

pub struct Locals<'a> {
    /// Local reservations for each tier
    tiers: [Option<&'a [Local]>; 1 << Tier::BITS],
}

impl fmt::Debug for Locals<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.tiers.fmt(f)
    }
}

impl<'a> Locals<'a> {
    pub fn metadata_size(tiering: &Tiering) -> usize {
        size_of_slice::<Local>(tiering.tiers().iter().map(|&(_, count)| count).sum())
    }
    pub unsafe fn metadata(&mut self) -> &'a mut [u8] {
        let Some(first) = self.tiers.iter().find_map(|local| *local) else {
            return &mut [];
        };
        let Some(last) = self.tiers.iter().rev().find_map(|local| *local) else {
            return &mut [];
        };
        let start = first.as_ptr();
        let end = last.as_ptr_range().end;
        unsafe {
            slice::from_raw_parts_mut(start.cast_mut().cast(), end.offset_from(start) as usize)
        }
    }

    /// Initialize the locals from a buffer
    pub fn new(buffer: &'a mut [u8], tiering: &Tiering) -> Result<Self, Error> {
        if buffer.len() < Self::metadata_size(tiering) {
            return Err(Error::Initialization);
        }

        let mut offset = 0;
        let mut tiers = [None::<&'a [Local]>; 1 << Tier::BITS];
        for &(tier, count) in tiering.tiers() {
            let local = unsafe {
                slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<Local>().add(offset), count)
            };
            offset += count;
            tiers[tier.0 as usize] = Some(local);
        }
        Ok(Self { tiers })
    }

    /// Get the number of locals for a tier, or None if the tier is not configured
    pub fn tier_locals(&self, tier: Tier) -> Option<usize> {
        self.tiers[tier.0 as usize].map(|local| local.len())
    }

    /// Try allocating from a local, returning the row id if successful, or the current reservation if not
    pub fn get(
        &self,
        tier: Tier,
        local: usize,
        tree: Option<TreeId>,
        free: usize,
    ) -> Result<RowId, Option<(RowId, usize)>> {
        let Some(locals) = &self.tiers[tier.0 as usize] else {
            return Err(None);
        };
        match locals[local].tree.fetch_update(|v| v.get(tree, free)) {
            Ok(old) => Ok(old.row()),
            Err(old) => Err(old.present().then_some((old.row(), old.free()))),
        }
    }

    /// Steal without demoting the target, but the request might be downgraded to a lower tier
    pub fn steal_any(
        &self,
        tier: Tier,
        index: Option<usize>,
        tree: Option<TreeId>,
        free: usize,
        policy: PolicyFn,
    ) -> Option<(RowId, Tier)> {
        let index = index.unwrap_or(0);
        for i in 0..self.tiers.len() {
            let target_tier = Tier(((i as u8) + tier.0) % self.tiers.len() as u8);

            let Some(target) = &self.tiers[target_tier.0 as usize] else {
                continue;
            };
            if !matches!(
                policy(tier, target_tier, free),
                Policy::Steal | Policy::Match(_)
            ) {
                continue;
            }

            for j in 0..target.len() {
                // Start at same local index to improve cache locality
                let j = (index + j) % target.len();

                if let Ok(row) = self.get(target_tier, j, tree, free) {
                    return Some((row, target_tier));
                }
            }
        }
        None
    }

    /// Steal from another tier and demote it to the current tier
    pub fn demote_any(
        &self,
        tier: Tier,
        index: Option<usize>,
        tree: Option<TreeId>,
        free: usize,
        policy: PolicyFn,
    ) -> Option<(RowId, Option<(RowId, Tier, usize)>)> {
        let Some(locals) = &self.tiers[tier.0 as usize] else {
            return None;
        };

        for i in 1..self.tiers.len() {
            let target_tier = Tier(((i as u8) + tier.0) % self.tiers.len() as u8);

            let Some(target) = &self.tiers[target_tier.0 as usize] else {
                continue;
            };
            if policy(tier, target_tier, free) != Policy::Demote {
                continue;
            }

            for j in 0..target.len() {
                // Start at same local index to improve cache locality
                let j = (index.unwrap_or(0) + j) % target.len();

                if let Ok(old) = target[j]
                    .tree
                    .fetch_update(|v| v.get(tree, free).map(|_| LocalTree::none()))
                {
                    let new = old.get(tree, free).unwrap();

                    let old = if let Some(index) = index {
                        // Replace local tree and return its free count for unreservation
                        let old = locals[index].tree.swap(new);
                        old.present().then_some((old.row(), tier, old.free()))
                    } else {
                        // Just return the new tree, without replacing any local tree
                        None
                    };
                    return Some((new.row(), old));
                }
            }
        }
        None
    }

    pub fn put(&self, tier: Tier, local: usize, tree: TreeId, free: usize) -> bool {
        let Some(locals) = &self.tiers[tier.0 as usize] else {
            return false;
        };

        let local = &locals[local];
        local.tree.fetch_update(|v| v.put(tree, free)).is_ok()
    }

    pub fn swap(
        &self,
        tier: Tier,
        local: usize,
        tree: TreeId,
        free: usize,
    ) -> Option<(RowId, usize)> {
        let Some(locals) = &self.tiers[tier.0 as usize] else {
            panic!("Invalid tier");
        };

        let local = &locals[local];
        let old = local.tree.swap(LocalTree::with(tree.as_row(), free));
        old.present().then_some((old.row(), old.free()))
    }

    pub fn drain(&self, unreserve: impl Fn(RowId, Tier, usize)) {
        for (i, locals) in self.tiers.iter().copied().enumerate() {
            if let Some(locals) = locals {
                let tier = Tier(i as u8);
                for local in locals {
                    let old = local.tree.swap(LocalTree::none());
                    if old.present() {
                        unreserve(old.row(), tier, old.free());
                    }
                }
            }
        }
    }

    pub fn set_start(&self, tier: Tier, index: usize, row: RowId) {
        let Some(locals) = &self.tiers[tier.0 as usize] else {
            return;
        };
        let local = &locals[index];
        let _ = local.tree.fetch_update(|v| v.set_start(row));
    }

    #[allow(dead_code)]
    #[cfg(feature = "free_reserve")]
    pub fn frees_push(&self, index: usize, tree_idx: TreeId) -> bool {
        self.local[index].frees_push(tree_idx)
    }

    pub fn stats(&self) -> TreeStats {
        let mut s = TreeStats::default();
        for (i, locals) in self.tiers.iter().copied().enumerate() {
            if let Some(locals) = locals {
                let tier = Tier(i as u8);
                for local in locals {
                    let tree = local.tree.load();
                    if tree.present() {
                        s.free_frames += tree.free();
                        s.free_trees += tree.free() / TREE_FRAMES;
                        s.tiers[tier.0 as usize].free_frames += tree.free();
                    }
                }
            }
        }
        s
    }

    pub fn load(&self, tier: Tier, local: usize) -> Option<(RowId, usize)> {
        let Some(locals) = &self.tiers[tier.0 as usize] else {
            return None;
        };
        let tree = locals[local].tree.load();
        tree.present().then_some((tree.row(), tree.free()))
    }
}

/// Core-local data
#[derive(Debug)]
#[repr(align(64))]
struct Local {
    /// Reserved trees for each [Tier]
    tree: Atom<LocalTree>,
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
    fn set_start(self, row: RowId) -> Option<Self> {
        if self.present() && self.row().as_tree() == row.as_tree() && self.row() != row {
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

#[cfg(test)]
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
