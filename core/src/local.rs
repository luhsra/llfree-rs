use core::mem::size_of;
use core::sync::atomic::AtomicU64;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::debug;

use crate::atomic::{Atom, Atomic};
use crate::util::size_of_slice;
use crate::{Error, Kind, KindDesc, Stats};
use crate::{FrameId, HUGE_FRAMES, TREE_FRAMES, TreeId};

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
    pub fn metadata_size(kinds: &[KindDesc]) -> usize {
        size_of_slice::<Local>(kinds.iter().map(|k| k.1 as usize).sum())
    }
    pub unsafe fn metadata(&mut self) -> &'a mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self.local.as_ptr().cast_mut().cast(),
                size_of_slice::<Local>(self.local.len()),
            )
        }
    }

    pub fn new(buffer: &'a mut [u8], kinds: &[KindDesc]) -> Result<Self, Error> {
        let len = buffer.len() / size_of::<Local>().next_multiple_of(align_of::<Local>());
        let kind_len = kinds.iter().map(|k| k.1 as usize).sum::<usize>();
        if len < kind_len {
            return Err(Error::Initialization);
        }
        let local: &mut [Local] =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), kind_len) };
        let mut offset = 0;
        for KindDesc(kind, count) in kinds {
            for local in &mut local[offset..][..*count as usize] {
                *local = Local {
                    kind: *kind,
                    ..Default::default()
                };
            }
            offset += *count as usize;
        }
        Ok(Self { local })
    }

    pub fn len(&self) -> usize {
        self.local.len()
    }

    pub fn can_get(&self, index: usize, frame: Option<FrameId>, free: usize) -> bool {
        let local = self.local[index].preferred.load();
        local.dec(frame, free).is_some()
    }

    pub fn kind(&self, index: usize) -> Kind {
        self.local[index].kind
    }

    pub fn get(
        &self,
        index: usize,
        frame: Option<FrameId>,
        free: usize,
    ) -> Result<FrameId, Option<(FrameId, usize)>> {
        let local = &self.local[index];
        match local.preferred.fetch_update(|v| v.dec(frame, free)) {
            Ok(old) => Ok(old.frame()),
            Err(old) => Err(old.present().then_some((old.frame(), old.free()))),
        }
    }

    pub fn steal(&self, index: usize, frame: Option<FrameId>, free: usize) -> Option<FrameId> {
        let kind = self.local[index].kind;
        // Steal from same kind first
        for i in 0..self.local.len() {
            let i = (index + i) % self.local.len();
            let local = &self.local[i];
            if local.kind == kind
                && let Ok(frame) = self.get(i, frame, free)
            {
                return Some(frame);
            }
        }
        // Fallback to stealing from lower kinds (no downgrade necessary)
        for i in 1..self.local.len() {
            let i = (index + i) % self.local.len();
            let local = &self.local[i];
            if local.kind < kind
                && let Ok(frame) = self.get(i, frame, free)
            {
                return Some(frame);
            }
        }
        None
    }

    // Find higher kind and steal and downgrade it to the current kind
    pub fn steal_downgrade(
        &self,
        index: usize,
        frame: Option<FrameId>,
        free: usize,
    ) -> Option<(FrameId, Option<(FrameId, usize)>)> {
        let kind = self.local[index].kind;
        for i in 1..self.local.len() {
            let i = (index + i) % self.local.len();
            let local = &self.local[i];
            if local.kind > kind
                && let Ok(old) = local
                    .preferred
                    .fetch_update(|v| v.dec(frame, free).map(|_| LocalTree::none()))
            {
                let new = old.dec(frame, free).unwrap();
                // Replace local tree and return its free count for unreservation
                let old = self.local[index].preferred.swap(new);
                return Some((
                    new.frame(),
                    old.present().then_some((old.frame(), old.free())),
                ));
            }
        }
        None
    }

    pub fn put(&self, index: usize, frame: FrameId, free: usize) -> Result<(), Option<FrameId>> {
        let local = &self.local[index];
        match local.preferred.fetch_update(|v| v.inc(frame, free)) {
            Ok(_) => Ok(()),
            Err(old) => Err(old.present().then_some(old.frame())),
        }
    }

    pub fn swap(&self, index: usize, frame: FrameId, free: usize) -> Option<(FrameId, usize)> {
        let local = &self.local[index];
        let old = local.preferred.swap(LocalTree::with(frame, free));
        old.present().then_some((old.frame(), old.free()))
    }

    pub fn drain(&self, index: usize) -> Option<(FrameId, usize)> {
        let local = &self.local[index];
        let old = local.preferred.swap(LocalTree::none());
        old.present().then_some((old.frame(), old.free()))
    }

    pub fn set_start(&self, index: usize, frame: FrameId) {
        let local = &self.local[index];
        let _ = local.preferred.fetch_update(|v| v.set_start(frame, false));
    }

    #[allow(dead_code)]
    #[cfg(feature = "free_reserve")]
    pub fn frees_push(&self, index: usize, tree_idx: TreeId) -> bool {
        self.local[index].frees_push(tree_idx)
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats::default();
        for local in self.local {
            let preferred = local.preferred.load();
            if preferred.present() {
                s.free_frames += preferred.free();
                if local.kind.is_huge() || preferred.free() == TREE_FRAMES {
                    s.free_huge += preferred.free() / HUGE_FRAMES;
                }
            }
        }
        s
    }

    pub fn stats_at(&self, frame: FrameId, free: usize) -> Stats {
        for local in self.local {
            let preferred = local.preferred.load();
            if preferred.present() && preferred.frame().as_tree() == frame.as_tree() {
                return Stats {
                    free_frames: free + preferred.free(),
                    free_huge: if local.kind.is_huge() || free == TREE_FRAMES {
                        (free + preferred.free()) / HUGE_FRAMES
                    } else {
                        0
                    },
                    free_trees: (free + preferred.free()) / TREE_FRAMES,
                };
            }
        }
        Stats::default()
    }

    pub fn load(&self, local: usize) -> Option<(FrameId, usize)> {
        let preferred = self.local[local].preferred.load();
        preferred
            .present()
            .then_some((preferred.frame(), preferred.free()))
    }
}

/// Core-local data
#[derive(Default, Debug)]
#[repr(align(64))]
struct Local {
    /// Kind of the local tree
    kind: Kind,
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
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
struct LocalTree {
    #[bits(48)]
    frame: FrameId,
    #[bits(15)]
    free: usize,
    /// Reserved for present bit...
    present: bool,
}
impl Atomic for LocalTree {
    type I = AtomicU64;
}
impl LocalTree {
    fn with(frame: FrameId, free: usize) -> Self {
        Self::new()
            .with_frame(frame)
            .with_free(free)
            .with_present(true)
    }
    fn none() -> Self {
        Self::new().with_present(false)
    }
    fn dec(self, frame: Option<FrameId>, free: usize) -> Option<Self> {
        if self.present() && frame.is_none_or(|i| self.frame().as_tree() == i.as_tree()) {
            Some(self.with_free(self.free().checked_sub(free)?))
        } else {
            None
        }
    }
    fn inc(self, frame: FrameId, free: usize) -> Option<Self> {
        if self.present() && self.frame().as_tree() == frame.as_tree() {
            assert!(self.free() + free <= TREE_FRAMES);
            Some(self.with_free(self.free() + free))
        } else {
            None
        }
    }
    fn set_start(self, frame: FrameId, force: bool) -> Option<Self> {
        if force || (self.present() && self.frame().as_tree() == frame.as_tree()) {
            Some(self.with_frame(frame))
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
