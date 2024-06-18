use core::sync::atomic::AtomicU64;

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};
use crate::trees::Kind;
use crate::TREE_FRAMES;

/// Core-local data
#[derive(Default, Debug)]
pub struct Local {
    /// Reserved trees for each [Kind]
    preferred: [Atom<LocalTree>; Kind::LEN],
}

impl Local {
    pub fn preferred(&self, kind: Kind) -> &Atom<LocalTree> {
        &self.preferred[kind as usize]
    }
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct LocalTree {
    #[bits(63)]
    pub frame: usize,
    /// Reserved for present bit...
    pub present: bool,
}
impl Atomic for LocalTree {
    type I = AtomicU64;
}
impl LocalTree {
    pub fn with(frame: usize) -> Self {
        Self::new()
            .with_frame(frame)
            .with_present(true)
    }
    pub fn none() -> Self {
        Self::new().with_present(false)
    }
    pub fn set_start(self, frame: usize, force: bool) -> Option<Self> {
        if force || (self.present() && self.frame() / TREE_FRAMES == frame / TREE_FRAMES) {
            Some(self.with_frame(frame))
        } else {
            None
        }
    }
    pub fn steal(self) -> Option<Self> {
        if self.present() {
            Some(Self::none())
        } else {
            None
        }
    }
}
