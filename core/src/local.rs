use core::sync::atomic::AtomicU64;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::debug;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::util::{OffsetSlice, size_of_slice};
use crate::{Class, Classing, Error, Policy, PolicyFn, TreeStats};
use crate::{TREE_FRAMES, TreeId};

pub struct Locals<'a> {
    buffer: &'a mut [u8],
    /// Local reservations for each class
    classes: [Option<OffsetSlice<Local>>; 1 << Class::BITS],
}

impl fmt::Debug for Locals<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_map();
        for (i, locals) in self.classes.iter().enumerate() {
            if let Some(locals) = locals {
                f.entry(&Class(i as u8), &locals.as_slice(self.buffer));
            }
        }
        f.finish()
    }
}

impl<'a> Locals<'a> {
    pub fn metadata_size(classing: &Classing) -> usize {
        size_of_slice::<Local>(classing.classes().iter().map(|&(_, count)| count).sum())
    }
    pub unsafe fn metadata(&mut self) -> &'a mut [u8] {
        // Lifetime hack: internal buffer outlives instance!
        unsafe { slice::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.len()) }
    }

    /// Initialize the locals from a buffer
    pub fn new(buffer: &'a mut [u8], classing: &Classing) -> Result<Self, Error> {
        if buffer.len() < Self::metadata_size(classing)
            || buffer.as_ptr().align_offset(align_of::<Local>()) != 0
        {
            return Err(Error::Initialization);
        }

        let mut offset = 0;
        let mut classes = [const { None }; 1 << Class::BITS];
        for &(class, count) in classing.classes() {
            let local = OffsetSlice::new(offset, count);
            offset += size_of_slice::<Local>(count);
            classes[class.0 as usize] = Some(local);
        }
        Ok(Self { buffer, classes })
    }

    /// Get the number of locals for a class, or None if the class is not configured
    pub fn class_locals(&self, class: Class) -> Option<usize> {
        self.classes[class.0 as usize]
            .as_ref()
            .map(|local| local.len())
    }

    /// Try allocating from a local, returning the row id if successful, or the current reservation if not
    pub fn get(
        &self,
        class: Class,
        local: usize,
        tree: Option<TreeId>,
        free: usize,
    ) -> Result<RowId, Option<Reservation>> {
        let Some(locals) = &self.locals(class) else {
            return Err(None);
        };
        match locals[local].tree.try_update(|v| v.get(tree, free)) {
            Ok(old) => Ok(old.row()),
            Err(old) => Err(old.present().then_some(old.as_reservation(class))),
        }
    }

    /// Steal without demoting the target, but the request might be downgraded to a lower class
    pub fn steal_any(
        &self,
        class: Class,
        index: Option<usize>,
        tree: Option<TreeId>,
        free: usize,
        policy: PolicyFn,
    ) -> Option<Reservation> {
        let index = index.unwrap_or(0);
        for i in 0..self.classes.len() {
            let target_class = Class(((i as u8) + class.0) % self.classes.len() as u8);

            let Some(target) = &self.classes[target_class.0 as usize] else {
                continue;
            };
            if !matches!(
                policy(class, target_class, free),
                Policy::Steal | Policy::Match(_)
            ) {
                continue;
            }

            for j in 0..target.len() {
                // Start at same local index to improve cache locality
                let j = (index + j) % target.len();

                if let Ok(row) = self.get(target_class, j, tree, free) {
                    return Some(Reservation::new(row, target_class, 0));
                }
            }
        }
        None
    }

    /// Steal from another class and demote it to the current class
    pub fn demote_any(
        &self,
        class: Class,
        local: Option<usize>,
        tree: Option<TreeId>,
        free: usize,
        policy: PolicyFn,
    ) -> Option<(RowId, Option<Reservation>)> {
        let Some(locals) = &self.locals(class) else {
            return None;
        };

        for i in 1..self.classes.len() {
            let target_class = Class(((i as u8) + class.0) % self.classes.len() as u8);

            let Some(target) = &self.locals(target_class) else {
                continue;
            };
            if policy(class, target_class, free) != Policy::Demote {
                continue;
            }

            for j in 0..target.len() {
                // Start at same local index to improve cache locality
                let j = (local.unwrap_or(0) + j) % target.len();

                if let Ok(old) = target[j]
                    .tree
                    .try_update(|v| v.get(tree, free).map(|_| LocalTree::none()))
                {
                    let new = old.get(tree, free).unwrap();

                    let old = if let Some(local) = local {
                        // Replace local tree and return the old tree for unreservation
                        let old = locals[local].tree.swap(new);
                        old.present().then_some(old.as_reservation(class))
                    } else {
                        // Or return (and unreserve) the demoted tree
                        Some(new.as_reservation(class))
                    };
                    return Some((new.row(), old));
                }
            }
        }
        None
    }

    pub fn put(&self, class: Class, local: usize, tree: TreeId, free: usize) -> bool {
        let Some(locals) = &self.locals(class) else {
            return false;
        };

        let local = &locals[local];
        local.tree.try_update(|v| v.put(tree, free)).is_ok()
    }

    pub fn swap(
        &self,
        class: Class,
        local: usize,
        tree: TreeId,
        free: usize,
    ) -> Option<Reservation> {
        debug!("swap tree");
        let Some(locals) = &self.locals(class) else {
            panic!("Invalid class");
        };

        let local = &locals[local];
        let old = local.tree.swap(LocalTree::with(tree.as_row(), free));
        old.present().then_some(old.as_reservation(class))
    }

    pub fn drain(&self, unreserve: impl Fn(RowId, Class, usize)) {
        for (i, locals) in self.classes.iter().enumerate() {
            if let Some(locals) = locals {
                let locals = locals.as_slice(self.buffer);
                let class = Class(i as u8);
                for local in locals {
                    let old = local.tree.swap(LocalTree::none());
                    if old.present() {
                        unreserve(old.row(), class, old.free());
                    }
                }
            }
        }
    }

    pub fn set_start(&self, class: Class, index: usize, row: RowId) {
        let Some(locals) = &self.locals(class) else {
            return;
        };
        let local = &locals[index];
        let _ = local.tree.try_update(|v| v.set_start(row));
    }

    pub fn stats(&self) -> TreeStats {
        let mut s = TreeStats::default();
        for (i, locals) in self.classes.iter().enumerate() {
            if let Some(locals) = locals {
                let locals = locals.as_slice(self.buffer);
                let class = Class(i as u8);
                for local in locals {
                    let tree = local.tree.load();
                    if tree.present() {
                        s.free_frames += tree.free();
                        s.free_trees += tree.free() / TREE_FRAMES;
                        s.classes[class.0 as usize].free_frames += tree.free();
                    }
                }
            }
        }
        s
    }

    pub fn load(&self, class: Class, local: usize) -> Option<Reservation> {
        let Some(locals) = &self.locals(class) else {
            return None;
        };
        let tree = locals[local].tree.load();
        tree.present().then_some(tree.as_reservation(class))
    }

    fn locals(&'a self, class: Class) -> Option<&'a [Local]> {
        self.classes[class.0 as usize]
            .as_ref()
            .map(|locals| locals.as_slice(self.buffer))
    }
}

#[derive(Debug, Clone)]
pub struct Reservation {
    pub row: RowId,
    pub class: Class,
    pub free: usize,
}
impl Reservation {
    pub fn new(row: RowId, class: Class, free: usize) -> Self {
        Self { row, class, free }
    }
}

/// Core-local data
#[derive(Debug)]
#[repr(align(64))]
struct Local {
    /// Reserved trees for each class
    tree: Atom<LocalTree>,
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
struct LocalTree {
    #[bits(44)]
    row: RowId,
    #[bits(19)]
    free: usize,
    /// Reserved for present bit...
    present: bool,
}

const _: () = assert!(1usize << LocalTree::FREE_BITS > TREE_FRAMES);

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

    fn as_reservation(self, class: Class) -> Reservation {
        Reservation::new(self.row(), class, self.free())
    }
}
