use bitfield_struct::bitfield;

use crate::trees::Kind;

/// Core-local data
#[derive(Default, Debug)]
pub struct Local {
    /// Reserved trees for each [Kind]
    preferred: [Option<LocalTree>; Kind::LEN],
    /// Tree index of the last freed frame
    last_idx: usize,
    /// Last frees counter
    last_frees: u8,
}

impl Local {
    /// Threshold for the number of frees after which a tree is reserved
    const F: u8 = 4;

    pub fn preferred(&self, kind: Kind) -> Option<LocalTree> {
        self.preferred[kind as usize]
    }
    pub fn preferred_mut(&mut self, kind: Kind) -> &mut Option<LocalTree> {
        &mut self.preferred[kind as usize]
    }

    /// Add a tree index to the history, returing if there are enough frees
    pub fn frees_push(&mut self, tree_idx: usize) -> bool {
        if self.last_idx == tree_idx {
            if self.last_frees >= Self::F {
                return true;
            }
            self.last_frees += 1;
        } else {
            self.last_idx = tree_idx;
            self.last_frees = 0;
        }
        false
    }
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct LocalTree {
    #[bits(45)]
    pub frame: usize,
    #[bits(15)]
    pub free: usize,
    #[bits(4)]
    pub huge: usize,
}
impl LocalTree {
    pub fn with(frame: usize, free: usize, huge: usize) -> Self {
        Self::new()
            .with_frame(frame)
            .with_free(free)
            .with_huge(huge)
    }
}
#[cfg(all(test, feature = "std"))]
mod test {
    use super::Local;

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        let mut local = Local::default();
        let frame1 = 43;
        let i1 = frame1 / (512 * 512);
        assert!(!local.frees_push(i1));
        assert!(!local.frees_push(i1));
        assert!(!local.frees_push(i1));
        assert!(!local.frees_push(i1));
        assert!(local.frees_push(i1));
        assert!(local.frees_push(i1));
        let frame2 = 512 * 512 + 43;
        let i2 = frame2 / (512 * 512);
        assert_ne!(i1, i2);
        assert!(!local.frees_push(i2));
        assert!(!local.frees_push(i2));
        assert!(!local.frees_push(i1));
    }
}
