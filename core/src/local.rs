use core::sync::atomic::AtomicU64;

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};
use crate::trees::Kind;
use crate::{FrameId, TREE_FRAMES, TreeId};

/// Core-local data
#[derive(Default, Debug)]
pub struct Local {
    /// Reserved trees for each [Kind]
    preferred: [Atom<LocalTree>; Kind::LEN],
    /// Recent frees
    frees: Atom<FreeHistory>,
}

impl Local {
    pub fn preferred(&self, kind: Kind) -> &Atom<LocalTree> {
        &self.preferred[kind as usize]
    }

    /// Add a tree index to the history, returing if there are enough frees
    pub fn frees_push(&self, tree_idx: TreeId) -> bool {
        let mut success = false;
        let _ = self.frees.fetch_update(|mut v| {
            success = v.push(tree_idx);
            Some(v)
        });
        success
    }
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct LocalTree {
    #[bits(48)]
    pub frame: FrameId,
    #[bits(15)]
    pub free: usize,
    /// Reserved for present bit...
    pub present: bool,
}
impl Atomic for LocalTree {
    type I = AtomicU64;
}
impl LocalTree {
    pub fn with(frame: FrameId, free: usize) -> Self {
        Self::new()
            .with_frame(frame)
            .with_free(free)
            .with_present(true)
    }
    pub fn none() -> Self {
        Self::new().with_present(false)
    }
    pub fn dec(self, frame: Option<FrameId>, free: usize) -> Option<Self> {
        if self.present() && frame.is_none_or(|i| self.frame().as_tree() == i.as_tree()) {
            Some(self.with_free(self.free().checked_sub(free)?))
        } else {
            None
        }
    }
    pub fn inc(self, frame: FrameId, free: usize) -> Option<Self> {
        if self.present() && self.frame().as_tree() == frame.as_tree() {
            assert!(self.free() + free <= TREE_FRAMES);
            Some(self.with_free(self.free() + free))
        } else {
            None
        }
    }
    pub fn set_start(self, frame: FrameId, force: bool) -> Option<Self> {
        if force || (self.present() && self.frame().as_tree() == frame.as_tree()) {
            Some(self.with_frame(frame))
        } else {
            None
        }
    }
    pub fn steal(self, free: usize, frame: Option<FrameId>) -> Option<Self> {
        if self.present()
            && self.free() >= free
            && frame.is_none_or(|i| self.frame().as_tree() == i.as_tree())
        {
            Some(Self::none())
        } else {
            None
        }
    }
}

#[bitfield(u64)]
pub struct FreeHistory {
    #[bits(48)]
    pub idx: TreeId,
    #[bits(16)]
    pub counter: usize,
}

impl FreeHistory {
    /// Threshold for the number of frees after which a tree is reserved
    const F: usize = 4;

    /// Add a tree index to the history, returing if there are enough frees
    pub fn push(&mut self, tree_idx: TreeId) -> bool {
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
    use crate::trees::TreeId;

    use super::Local;

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        let local = Local::default();
        let frame1 = 43;
        let i1 = frame1 / (512 * 512);
        assert!(!local.frees_push(TreeId(i1)));
        assert!(!local.frees_push(TreeId(i1)));
        assert!(!local.frees_push(TreeId(i1)));
        assert!(!local.frees_push(TreeId(i1)));
        assert!(local.frees_push(TreeId(i1)));
        assert!(local.frees_push(TreeId(i1)));
        let frame2 = 512 * 512 + 43;
        let i2 = frame2 / (512 * 512);
        assert_ne!(i1, i2);
        assert!(!local.frees_push(TreeId(i2)));
        assert!(!local.frees_push(TreeId(i2)));
        assert!(!local.frees_push(TreeId(i1)));
    }
}
