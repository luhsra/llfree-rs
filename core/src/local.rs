use core::cell::UnsafeCell;
use core::ops::Range;
use core::sync::atomic::AtomicU64;
use core::{fmt, slice};

use bitfield_struct::bitfield;
use log::debug;

use crate::atomic::{Atom, Atomic};
use crate::bitfield::RowId;
use crate::cache::{Align, Aligned, UnsafeCacheLine};
use crate::lower::LowerTree;
use crate::{CacheLine, Class, Classing, Error, FrameId, Policy, PolicyFn, TREE_FRAMES, TreeId};

/// The data associated with a local tree.
#[derive(Debug, Clone, Copy)]
pub struct TreeData {
    pub row: RowId,
    pub free: usize,
}
impl TreeData {
    pub fn new(row: RowId, free: usize) -> Self {
        Self { row, free }
    }
}

trait HasLocalTree {
    fn local_tree(&self) -> &Atom<LocalTree>;
}

/// Copy of a local tree.
struct TreeClone {
    tree: Atom<LocalTree>,
    lower: UnsafeCell<LowerTree>,
}
impl fmt::Debug for TreeClone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeClone")
            .field("tree", &self.tree)
            .finish_non_exhaustive()
    }
}
impl HasLocalTree for TreeClone {
    fn local_tree(&self) -> &Atom<LocalTree> {
        &self.tree
    }
}

struct SingleAccess;
struct MultiAccess;

/// An tree change operation.
trait TreeChange {
    type Access;
    type CommitArgs: Clone;
    fn try_start(&mut self, tree: LocalTree, retry: &mut bool) -> Option<LocalTree>;
    fn commit(&self, tree: LocalTree, args: Self::CommitArgs) -> LocalTree;
    fn abort(&self, tree: LocalTree) -> LocalTree;
}

/// Allocate frames from a local tree.
pub struct TreeGet {
    tree: Option<TreeId>,
    free: usize,
}
impl TreeGet {
    fn new(tree: Option<TreeId>, free: usize) -> Self {
        Self { tree, free }
    }
}
impl TreeChange for TreeGet {
    type Access = MultiAccess;
    type CommitArgs = RowId;
    fn try_start(&mut self, tree: LocalTree, retry: &mut bool) -> Option<LocalTree> {
        *retry = tree.switching();
        if *retry {
            return None;
        }
        if self.tree.is_none_or(|t| tree.row().as_tree() == t) {
            let tree = tree
                .is_present(self.tree)?
                .with_free(tree.free().checked_sub(self.free)?);
            if let Some(tree) = tree.try_acquire() {
                Some(tree)
            } else {
                *retry = true;
                None
            }
        } else {
            None
        }
    }
    fn commit(&self, tree: LocalTree, args: Self::CommitArgs) -> LocalTree {
        tree.release().with_row(args)
    }
    fn abort(&self, tree: LocalTree) -> LocalTree {
        tree.release().with_free(tree.free() + self.free)
    }
}

/// Free frames into a local tree.
pub struct TreePut {
    tree: TreeId,
    free: usize,
}
impl TreePut {
    pub fn new(tree: TreeId, free: usize) -> Self {
        Self { tree, free }
    }
}
impl TreeChange for TreePut {
    type Access = MultiAccess;
    type CommitArgs = ();
    fn try_start(&mut self, tree: LocalTree, retry: &mut bool) -> Option<LocalTree> {
        *retry = false;
        if self.tree == tree.row().as_tree() {
            assert!(tree.free() + self.free <= TREE_FRAMES);
            let tree = tree.is_present(Some(self.tree))?;
            if let Some(tree) = tree.try_acquire() {
                Some(tree)
            } else {
                *retry = true;
                None
            }
        } else {
            None
        }
    }
    fn commit(&self, tree: LocalTree, _args: Self::CommitArgs) -> LocalTree {
        tree.release().with_free(tree.free() + self.free)
    }
    fn abort(&self, tree: LocalTree) -> LocalTree {
        tree.release()
    }
}

pub struct TreeSwitch;
impl TreeChange for TreeSwitch {
    type Access = SingleAccess;
    type CommitArgs = Option<TreeData>;
    fn try_start(&mut self, tree: LocalTree, retry: &mut bool) -> Option<LocalTree> {
        *retry = false;
        if let Some(t) = tree.try_switch() {
            Some(t)
        } else {
            *retry = true;
            None
        }
    }
    fn commit(&self, _tree: LocalTree, args: Self::CommitArgs) -> LocalTree {
        match args {
            Some(data) => LocalTree::with(data.row, data.free),
            None => LocalTree::none(),
        }
    }
    fn abort(&self, tree: LocalTree) -> LocalTree {
        tree.with_switching(false)
    }
}

pub struct TreeDrain {
    tree: Option<TreeId>,
    free: usize,
}
impl TreeDrain {
    pub fn new(tree: Option<TreeId>, free: usize) -> Self {
        Self { tree, free }
    }
}
impl TreeChange for TreeDrain {
    type Access = SingleAccess;
    type CommitArgs = ();
    fn try_start(&mut self, tree: LocalTree, retry: &mut bool) -> Option<LocalTree> {
        *retry = false;
        if self.tree.is_none_or(|t| tree.row().as_tree() == t) {
            let tree = tree
                .is_present(self.tree)?
                .with_free(tree.free().checked_sub(self.free)?);
            if let Some(tree) = tree.try_switch() {
                Some(tree)
            } else {
                *retry = true;
                None
            }
        } else {
            None
        }
    }
    fn commit(&self, _tree: LocalTree, _args: Self::CommitArgs) -> LocalTree {
        LocalTree::none()
    }
    fn abort(&self, tree: LocalTree) -> LocalTree {
        assert!(self.tree.is_none_or(|t| tree.row().as_tree() == t));
        assert!(tree.switching());
        tree.with_free(tree.free() + self.free)
            .with_switching(false)
    }
}

/// Two-part access to a local tree.
///
/// This covers two use cases:
/// - Modification: usage counter is increased, preventing a switch
/// - Switching: state is changed, preventing modification
struct TreeAccess<'a, T: HasLocalTree, C: TreeChange> {
    tree: &'a T,
    change: C,
}
impl<'a, T: HasLocalTree, C: TreeChange> TreeAccess<'a, T, C> {
    fn try_start(
        tree: &'a T,
        mut change: C,
    ) -> ::core::result::Result<(LocalTree, Self), LocalTree> {
        // Spin while waiting for incompatible changes to be finished
        loop {
            let mut retry = false;
            match tree
                .local_tree()
                .try_update(|t| change.try_start(t, &mut retry))
            {
                Ok(t) => return Ok((t, Self { tree, change })),
                Err(tree) if !retry => return Err(tree),
                _ => core::hint::spin_loop(),
            }
        }
    }
    fn commit(self, args: C::CommitArgs) -> LocalTree {
        let res = self
            .tree
            .local_tree()
            .update(|t| self.change.commit(t, args.clone()));
        // Do not run the default drop
        core::mem::forget(self);
        res.with_switching(false)
    }
}
impl<'a, C: TreeChange<Access = MultiAccess>> TreeAccess<'a, TreeClone, C> {
    pub fn lower(&self) -> &'_ LowerTree {
        unsafe { &*self.tree.lower.get() }
    }
}
impl<'a, C: TreeChange<Access = SingleAccess>> TreeAccess<'a, TreeClone, C> {
    pub fn lower_mut(&mut self) -> &mut LowerTree {
        unsafe { &mut *self.tree.lower.get() }
    }
}
impl<'a, T: HasLocalTree, C: TreeChange> Drop for TreeAccess<'a, T, C> {
    fn drop(&mut self) {
        self.tree.local_tree().update(|t| self.change.abort(t));
    }
}

/// Local tree reservations for each class.
pub struct Locals<'a> {
    /// Per-class reservations used for allocations, each class can have multiple reservations
    ///
    /// - None for not-defined classes
    /// - 0..0 for defined classes with no reservations
    /// - .. for defined classes with reservations
    ///
    /// Frees also prioritize local classes over `free_copies`
    classes: [Option<Range<usize>>; Class::LEN as usize],
    /// Tree reservations for each class, indexed by `classes`
    reservations: &'a [Align<TreeClone>],
    /// Tree copies used for batching frees
    ///
    /// Len is the max number of reservations of all classes.
    /// Replacement strategy: LRU
    free_clones: &'a [Align<TreeClone>],
}

impl fmt::Debug for Locals<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_map();
        for (i, local) in self.classes.iter().enumerate() {
            if let Some(local) = local {
                let slice = &self.reservations[local.clone()];
                f.entry(&Class(i as u8), &slice);
            }
        }
        f.finish()
    }
}

impl<'a> Locals<'a> {
    /// Metadata size in cache lines
    pub fn metadata_size(classing: &Classing) -> usize {
        let classes: usize = classing.classes().iter().map(|&(_, count)| count).sum();
        let classes = classes * size_of::<Align<TreeClone>>().div_ceil(CacheLine::SIZE);
        let free_clones =
            classing.free_clones * size_of::<Align<TreeClone>>().div_ceil(CacheLine::SIZE);
        classes + free_clones
    }
    pub unsafe fn metadata(&mut self) -> &'a [UnsafeCacheLine] {
        unsafe {
            // Lifetime hack: internal buffer outlives instance!
            slice::from_raw_parts(
                self.reservations.cache_line().cast(),
                self.reservations.cache_lines() + self.free_clones.cache_lines(),
            )
        }
    }

    /// Initialize the locals from a buffer
    pub fn new(buffer: &'a [UnsafeCacheLine], classing: &Classing) -> Result<Self, Error> {
        if buffer.len() < Self::metadata_size(classing) {
            return Err(Error::Initialization);
        }

        let num_classes = classing.classes().iter().map(|c| c.1).sum::<usize>();
        let reservations = unsafe {
            slice::from_raw_parts(buffer.as_ptr().cast::<Align<TreeClone>>(), num_classes)
        };
        let buffer = &buffer[reservations.cache_lines()..];
        let free_clones = unsafe {
            slice::from_raw_parts(
                buffer.as_ptr().cast::<Align<TreeClone>>(),
                classing.free_clones,
            )
        };
        assert!(buffer.len() >= free_clones.cache_lines());

        let mut offset = 0;
        let mut classes = [const { None }; Class::LEN as usize];
        for &(class, count) in classing.classes() {
            classes[class.0 as usize] = Some(offset..offset + count);
            offset += count;
        }
        Ok(Self {
            reservations,
            classes,
            free_clones,
        })
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
        frame: Option<FrameId>,
        order: usize,
    ) -> Result<FrameId, Option<TreeData>> {
        let Some(locals) = &self.locals(class) else {
            return Err(None);
        };
        match TreeAccess::try_start(
            &locals[local].0,
            TreeGet::new(frame.map(FrameId::as_tree), 1 << order),
        ) {
            Ok((t, a)) => match a.lower().get(
                t.row().as_frame().inside_tree().as_row(),
                order,
                frame.map(FrameId::inside_tree),
            ) {
                Ok(frame) => {
                    let frame = t.row().as_tree().as_frame() + frame;
                    a.commit(frame.as_row());
                    Ok(frame)
                }
                Err(_) => Err(t.as_data()),
            },
            Err(old) => Err(old.as_data()),
        }
    }

    /// Steal without demoting the target, but the request might be downgraded to a lower class
    pub fn steal_any(
        &self,
        class: Class,
        index: Option<usize>,
        frame: Option<FrameId>,
        order: usize,
        policy: PolicyFn,
    ) -> Option<(FrameId, Class)> {
        let index = index.unwrap_or(0);
        for i in 0..self.classes.len() {
            let target_class = Class(((i as u8) + class.0) % self.classes.len() as u8);

            let Some(target) = &self.classes[target_class.0 as usize] else {
                continue;
            };
            if !matches!(
                policy(class, target_class, 1 << order),
                Policy::Steal | Policy::Match(_)
            ) {
                continue;
            }

            for j in 0..target.len() {
                // Start at same local index to improve cache locality
                let j = (index + j) % target.len();

                if let Ok(frame) = self.get(target_class, j, frame, order) {
                    return Some((frame, target_class));
                }
            }
        }
        None
    }

    /// Steal from another class and demote it to the current class, and claiming the reservation
    pub fn demote_any(
        &self,
        class: Class,
        local: Option<usize>,
        frame: Option<FrameId>,
        order: usize,
        policy: PolicyFn,
    ) -> Option<(FrameId, Option<Reservation>)> {
        let Some(locals) = &self.locals(class) else {
            return None;
        };

        for i in 1..self.classes.len() {
            let target_class = Class(((i as u8) + class.0) % self.classes.len() as u8);

            let Some(target) = &self.locals(target_class) else {
                continue;
            };
            if policy(class, target_class, 1 << order) != Policy::Demote {
                continue;
            }

            for j in 0..target.len() {
                // Start at same local index to improve cache locality
                let j = (local.unwrap_or(0) + j) % target.len();

                if let Ok((old, mut access)) = TreeAccess::try_start(
                    &target[j].0,
                    TreeDrain::new(frame.map(FrameId::as_tree), 1 << order),
                )
                    && let Ok(frame) = access.lower_mut().get(
                        old.row().as_frame().inside_tree().as_row(),
                        order,
                        frame.map(FrameId::inside_tree),
                    ) {
                        let frame = old.row().as_tree().as_frame() + frame;
                        // Claim the tree
                        let claimed_lower =
                            core::mem::take(access.lower_mut());
                        let mut claimed_tree = access.commit(()).as_data().unwrap();
                        claimed_tree.row = frame.as_row();

                        let res = if let Some(local) = local
                            && let Ok((_, mut access)) =
                                TreeAccess::try_start(&locals[local].0, TreeSwitch)
                        {
                            // Replace local tree and return the old tree for unreservation
                            let lower = core::mem::replace(access.lower_mut(), claimed_lower);
                            let old = access.commit(Some(claimed_tree));
                            old.as_data().map(|old| Reservation::new(class, old, lower))
                        } else {
                            // Or return (and unreserve) the demoted tree
                            Some(Reservation::new(class, claimed_tree, claimed_lower))
                        };

                        return Some((frame, res));
                    }
            }
        }
        None
    }

    pub fn put(
        &self,
        class: Class,
        local: Option<usize>,
        frame: FrameId,
        order: usize,
        free_idx: usize,
    ) -> Result<(), Error> {
        // Try reservation first, to directly supply frames to the allocation path
        if let Some(local) = local {
            let Some(locals) = self.locals(class) else {
                return Err(Error::Memory);
            };
            if let Ok((_, access)) =
                TreeAccess::try_start(&locals[local].0, TreePut::new(frame.as_tree(), 1 << order))
            {
                access.lower().put(frame.inside_tree(), order)?;
                access.commit(());
                return Ok(());
            }
        }
        // Batch free in local clone
        if let Ok((_, access)) = TreeAccess::try_start(
            &self.free_clones[free_idx].0,
            TreePut::new(frame.as_tree(), 1 << order),
        ) {
            access.lower().put(frame.inside_tree(), order)?;
            access.commit(());
            return Ok(());
        }
        Err(Error::Memory)
    }

    /// Merge transferred free frames into the reservation while exclusively accessing it.
    /// The caller must hold the CXL lock while `take` runs.
    pub fn sync(
        &self,
        class: Class,
        local: usize,
        tree: TreeId,
        take: impl FnOnce() -> Option<(LowerTree, usize)>,
    ) -> bool {
        let Some(locals) = self.locals(class) else {
            return false;
        };
        let Ok((old, mut access)) = TreeAccess::try_start(&locals[local].0, TreeSwitch) else {
            return false;
        };
        if !old.present() || old.row().as_tree() != tree {
            return false;
        }
        let Some((lower, added)) = take() else {
            return false;
        };
        access.lower_mut().merge_frees(lower);
        access.commit(Some(TreeData::new(old.row(), old.free() + added)));
        true
    }

    pub fn switch_reservation(
        &self,
        class: Class,
        local: usize,
        tree: Option<Reservation>,
    ) -> Option<Reservation> {
        debug!("swap alloc tree");
        let Some(locals) = &self.locals(class) else {
            panic!("Invalid class");
        };
        let local = &locals[local];
        if let Ok((_, mut access)) = TreeAccess::try_start(&local.0, TreeSwitch) {
            let (data, lower) = tree.map(|t| (t.data, t.lower)).unzip();
            let old_lower = core::mem::replace(access.lower_mut(), lower.unwrap_or_default());
            let old = access.commit(data);
            old.as_data()
                .map(|old| Reservation::new(class, old, old_lower))
        } else {
            None
        }
    }

    pub fn switch_free_clone(&self, free_idx: usize, tree: Option<FreeClone>) -> Option<FreeClone> {
        debug!("swap free tree");
        let free_clone = &self.free_clones[free_idx];
        if let Ok((_, mut access)) = TreeAccess::try_start(&free_clone.0, TreeSwitch) {
            let (data, lower) = tree.map(|t| (t.data, t.lower)).unzip();
            let old_lower = core::mem::replace(access.lower_mut(), lower.unwrap_or_default());
            let old = access.commit(data);
            old.as_data().map(|old| FreeClone::new(old, old_lower))
        } else {
            None
        }
    }

    pub fn drain(&'a self, unreserve: impl Fn(Reservation), merge: impl Fn(FreeClone)) {
        for (i, locals) in self.classes.iter().enumerate() {
            if let Some(locals) = locals {
                let locals = &self.reservations[locals.clone()];
                let class = Class(i as u8);
                for local in locals {
                    if let Ok((_, mut access)) = TreeAccess::try_start(&local.0, TreeSwitch) {
                        let lower = core::mem::take(access.lower_mut());
                        let old = access.commit(None);
                        if old.present() {
                            unreserve(Reservation::new(class, old.as_data().unwrap(), lower));
                        }
                    }
                }
            }
        }
        for free_clone in self.free_clones.iter() {
            if let Ok((_, mut access)) = TreeAccess::try_start(&free_clone.0, TreeSwitch) {
                let old_lower = core::mem::take(access.lower_mut());
                let old = access.commit(None);
                if let Some(old) = old.as_data() {
                    merge(FreeClone::new(old, old_lower));
                }
            }
        }
    }

    /// Visit approximate counter snapshots, grouped by tree across all local clones.
    pub fn for_each_tree_data(&self, mut visit: impl FnMut(TreeId, usize, Option<Class>)) {
        for (index, (_, clone)) in self.clones().enumerate() {
            let snapshot = clone.tree.load();
            if !snapshot.present() {
                continue;
            }
            let tree = snapshot.row().as_tree();
            if self.clones().take(index).any(|(_, clone)| {
                let snapshot = clone.tree.load();
                snapshot.present() && snapshot.row().as_tree() == tree
            }) {
                continue;
            }
            let mut free = 0;
            let mut class = None;
            for (host, clone) in self.clones() {
                let snapshot = clone.tree.load();
                if snapshot.present() && snapshot.row().as_tree() == tree {
                    free += snapshot.free();
                    if host.is_some() {
                        class = host;
                    }
                }
            }
            visit(tree, free, class);
        }
    }

    /// Visit each matching lower clone while exclusively accessing its slot.
    /// The caller must hold the CXL lock to prevent global transfers.
    pub fn for_each_lower(&self, tree: TreeId, mut visit: impl FnMut(usize, &LowerTree)) {
        for (_, clone) in self.clones() {
            let snapshot = clone.tree.load();
            if !snapshot.present() || snapshot.row().as_tree() != tree {
                continue;
            }
            if let Ok((old, mut access)) = TreeAccess::try_start(clone, TreeSwitch)
                && old.present()
                && old.row().as_tree() == tree
            {
                visit(old.free(), access.lower_mut());
            }
        }
    }

    fn clones(&self) -> impl Iterator<Item = (Option<Class>, &TreeClone)> {
        self.classes
            .iter()
            .enumerate()
            .flat_map(move |(i, locals)| {
                locals.iter().flat_map(move |locals| {
                    self.reservations[locals.clone()]
                        .iter()
                        .map(move |clone| (Some(Class(i as u8)), &clone.0))
                })
            })
            .chain(self.free_clones.iter().map(|clone| (None, &clone.0)))
    }

    #[cfg(test)]
    pub fn load(&self, class: Class, local: usize) -> Option<TreeData> {
        self.locals(class)
            .and_then(|locals| locals.get(local))
            .and_then(|l| l.tree.load().as_data())
    }

    fn locals(&self, class: Class) -> Option<&'_ [Align<TreeClone>]> {
        self.classes[class.0 as usize]
            .clone()
            .map(|locals| &self.reservations[locals])
    }
}

#[derive(Debug)]
pub struct Reservation {
    pub class: Class,
    pub data: TreeData,
    pub lower: LowerTree,
}
impl Reservation {
    pub fn new(class: Class, data: TreeData, lower: LowerTree) -> Self {
        Self { class, data, lower }
    }
}

pub struct FreeClone {
    pub data: TreeData,
    pub lower: LowerTree,
}
impl FreeClone {
    pub fn new(data: TreeData, lower: LowerTree) -> Self {
        Self { data, lower }
    }
}

/// Local tree copy
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
struct LocalTree {
    #[bits(32)]
    row: RowId,
    #[bits(19)]
    free: usize,
    /// Is this tree slot currently present
    present: bool,
    /// Is a core currently switching to this tree
    ///
    /// This is only allowed when no other cores are using this tree.
    /// Similar to the single writer of an RwLock.
    switching: bool,
    /// Number of cores currently modifying this tree
    ///
    /// Concurrent modifications are allowed but no switching operation.
    /// Similar to the multiple readers of an RwLock.
    #[bits(11)]
    users: usize,
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
        Self::new()
    }

    /// A single core can switch trees if no other users are using it.
    fn try_switch(self) -> Option<Self> {
        if !self.switching() && self.users() == 0 {
            Some(self.with_switching(true))
        } else {
            None
        }
    }
    /// Multiple cores can acquire a tree if it is not being switched.
    fn try_acquire(self) -> Option<Self> {
        if !self.switching() {
            Some(self.with_users(self.users() + 1))
        } else {
            None
        }
    }

    fn is_present(self, tree: Option<TreeId>) -> Option<Self> {
        if self.present() && tree.is_none_or(|i| self.row().as_tree() == i) {
            Some(self)
        } else {
            None
        }
    }

    fn release(self) -> Self {
        assert!(self.present());
        self.with_users(self.users() - 1)
    }

    fn as_data(self) -> Option<TreeData> {
        assert!(!self.switching());
        if self.present() {
            Some(TreeData::new(self.row(), self.free()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clone_with(free: usize) -> TreeClone {
        TreeClone {
            tree: Atom::new(LocalTree::with(TreeId(0).as_row(), free)),
            lower: UnsafeCell::new(LowerTree::default()),
        }
    }

    fn clone_at(tree: TreeId, free: usize) -> TreeClone {
        let lower = LowerTree::default();
        lower.init(free, crate::Init::FreeAll);
        TreeClone {
            tree: Atom::new(LocalTree::with(tree.as_row(), free)),
            lower: UnsafeCell::new(lower),
        }
    }

    fn test_locals<'a>(
        reservations: &'a [Align<TreeClone>],
        free_clones: &'a [Align<TreeClone>],
    ) -> Locals<'a> {
        let mut classes = [const { None }; Class::LEN as usize];
        for (i, _) in reservations.iter().enumerate() {
            classes[i] = Some(i..i + 1);
        }
        Locals {
            classes,
            reservations,
            free_clones,
        }
    }

    #[test]
    fn get_nonzero_tree_relative_addresses() {
        let tree = TreeId(3);
        let reservations = [Align(clone_at(tree, 256))];
        let locals = test_locals(&reservations, &[]);
        let target = tree.as_frame() + FrameId(128);
        assert_eq!(locals.get(Class(0), 0, Some(target), 0).unwrap(), target);
        assert_eq!(locals.load(Class(0), 0).unwrap().row, target.as_row());
        let frame = locals.get(Class(0), 0, None, 0).unwrap();
        assert_eq!(frame.as_tree(), tree);
        assert_eq!(frame.inside_tree().as_row(), target.inside_tree().as_row());
        let reservation = locals.switch_reservation(Class(0), 0, None).unwrap();
        assert_eq!(reservation.data.free, 254);
        assert_eq!(reservation.lower.stats().free_frames, 254);
        assert!(!reservation.lower.is_free(target.inside_tree(), 0));
        assert!(!reservation.lower.is_free(frame.inside_tree(), 0));
    }

    #[test]
    fn put_nonzero_tree_reservation_and_free_clone() {
        let tree = TreeId(4);
        let reservations = [Align(clone_at(tree, 0))];
        let free_clones = [Align(clone_at(TreeId(5), 0))];
        let locals = test_locals(&reservations, &free_clones);
        let frame = tree.as_frame() + FrameId(128);
        locals.put(Class(0), Some(0), frame, 0, 0).unwrap();
        let reservation = locals.switch_reservation(Class(0), 0, None).unwrap();
        assert_eq!(reservation.data.free, 1);
        assert_eq!(reservation.lower.stats().free_frames, 1);
        assert!(reservation.lower.is_free(frame.inside_tree(), 0));
        let frame = TreeId(5).as_frame() + FrameId(192);
        locals.put(Class(0), Some(0), frame, 0, 0).unwrap();
        let clone = locals.switch_free_clone(0, None).unwrap();
        assert_eq!(clone.data.free, 1);
        assert_eq!(clone.lower.stats().free_frames, 1);
        assert!(clone.lower.is_free(frame.inside_tree(), 0));
    }

    #[test]
    fn demote_nonzero_tree_relative_addresses() {
        for local in [Some(0), None] {
            for targeted in [true, false] {
                let tree = TreeId(6);
                let reservations = [Align(clone_at(TreeId(0), 0)), Align(clone_at(tree, 256))];
                let locals = test_locals(&reservations, &[]);
                let target = tree.as_frame() + FrameId(128);
                let (frame, returned) = locals
                    .demote_any(Class(0), local, targeted.then_some(target), 0, |_, _, _| {
                        Policy::Demote
                    })
                    .unwrap();
                assert_eq!(frame.as_tree(), tree);
                if targeted {
                    assert_eq!(frame, target);
                }
                assert!(locals.load(Class(1), 0).is_none());
                let reservation = if local.is_some() {
                    assert_eq!(returned.unwrap().data.free, 0);
                    locals.switch_reservation(Class(0), 0, None).unwrap()
                } else {
                    returned.unwrap()
                };
                assert_eq!(reservation.data.row, frame.as_row());
                assert_eq!(reservation.data.free, 255);
                assert_eq!(reservation.lower.stats().free_frames, 255);
                assert!(!reservation.lower.is_free(frame.inside_tree(), 0));
            }
        }
    }

    #[test]
    fn allocation_waits_for_switch_before_capacity_or_identity_checks() {
        let tree = LocalTree::with(TreeId(7).as_row(), 0).with_switching(true);
        for target in [None, Some(TreeId(7)), Some(TreeId(8))] {
            let mut change = TreeGet::new(target, 1);
            let mut retry = false;
            assert!(change.try_start(tree, &mut retry).is_none());
            assert!(retry);
        }
    }

    #[test]
    fn sync_mismatch_does_not_transfer() {
        let tree = TreeId(7);
        let reservations = [Align(clone_at(tree, 1))];
        let locals = test_locals(&reservations, &[]);
        let source = clone_at(tree, 2);
        let called = core::cell::Cell::new(false);
        assert!(!locals.sync(Class(0), 0, TreeId(8), || {
            called.set(true);
            Some((unsafe { &*source.lower.get() }.get_all_copy(), 2))
        }));
        assert!(!called.get());
        assert_eq!(unsafe { &*source.lower.get() }.stats().free_frames, 2);
        assert_eq!(locals.load(Class(0), 0).unwrap().free, 1);
        assert!(!reservations[0].tree.load().switching());
        locals.switch_reservation(Class(0), 0, None).unwrap();
        assert!(!locals.sync(Class(0), 0, tree, || panic!(
            "empty reservation called take"
        )));
        assert!(!reservations[0].tree.load().switching());
    }

    #[test]
    fn sync_no_transfer_restores_reservation() {
        let tree = TreeId(7);
        let reservations = [Align(clone_at(tree, 1))];
        let locals = test_locals(&reservations, &[]);
        assert!(!locals.sync(Class(0), 0, tree, || None));
        assert!(!reservations[0].tree.load().switching());
        let reservation = locals.switch_reservation(Class(0), 0, None).unwrap();
        assert_eq!(reservation.data.row, tree.as_row());
        assert_eq!(reservation.data.free, 1);
        assert_eq!(reservation.lower.stats().free_frames, 1);
    }

    #[test]
    fn sync_credit_matches_merged_lower() {
        let tree = TreeId(9);
        let reservations = [Align(clone_at(tree, 1))];
        let locals = test_locals(&reservations, &[]);
        let source = LowerTree::default();
        source.init(0, crate::Init::FreeAll);
        source.put(FrameId(128), 1).unwrap();
        assert!(locals.sync(Class(0), 0, tree, || {
            assert!(reservations[0].tree.load().switching());
            Some((source.get_all_copy(), 2))
        }));
        assert_eq!(source.stats().free_frames, 0);
        assert!(!reservations[0].tree.load().switching());
        let reservation = locals.switch_reservation(Class(0), 0, None).unwrap();
        assert_eq!(reservation.data.row, tree.as_row());
        assert_eq!(reservation.data.free, 3);
        assert_eq!(reservation.lower.stats().free_frames, 3);
        assert!(reservation.lower.is_free(FrameId(0), 0));
        assert!(reservation.lower.is_free(FrameId(128), 1));
    }

    #[test]
    fn tree_data_groups_reservations_and_free_clones() {
        let tree = TreeId(9);
        let free_only = TreeId(10);
        let reservations = [Align(clone_at(tree, 4)), Align(clone_at(tree, 5))];
        let free_clones = [
            Align(clone_at(tree, 3)),
            Align(clone_at(free_only, 2)),
            Align(clone_at(free_only, 1)),
            Align(clone_at(TreeId(11), 0)),
        ];
        let mut locals = test_locals(&reservations, &free_clones);
        locals.classes[0] = None;
        locals.classes[1] = Some(0..2);
        locals.switch_free_clone(3, None).unwrap();
        let mut snapshots = std::vec::Vec::new();
        locals.for_each_tree_data(|tree, free, class| snapshots.push((tree, free, class)));
        assert_eq!(
            snapshots,
            [(tree, 12, Some(Class(1))), (free_only, 3, None)]
        );
    }

    #[test]
    fn tree_data_reads_switching_snapshots() {
        let tree = TreeId(9);
        let reservations = [Align(clone_at(tree, 4))];
        let free_clones = [Align(clone_at(tree, 3))];
        let locals = test_locals(&reservations, &free_clones);
        let (_, reservation_access) =
            TreeAccess::try_start(&reservations[0].0, TreeSwitch).unwrap();
        let (_, free_access) = TreeAccess::try_start(&free_clones[0].0, TreeSwitch).unwrap();
        let mut calls = 0;
        locals.for_each_tree_data(|visited, free, class| {
            assert_eq!(visited, tree);
            assert_eq!(free, 7);
            assert_eq!(class, Some(Class(0)));
            calls += 1;
        });
        assert_eq!(calls, 1);
        drop(reservation_access);
        drop(free_access);
    }

    #[test]
    fn lower_reader_matches_synced_counters_and_restores_slots() {
        let tree = TreeId(9);
        let reservations = [Align(clone_at(tree, 1)), Align(clone_at(TreeId(10), 2))];
        let free_clones = [Align(clone_at(tree, 4)), Align(clone_at(tree, 0))];
        let locals = test_locals(&reservations, &free_clones);
        let source = LowerTree::default();
        source.init(0, crate::Init::FreeAll);
        source.put(FrameId(128), 1).unwrap();
        assert!(locals.sync(Class(0), 0, tree, || Some((source.get_all_copy(), 2))));
        locals.switch_free_clone(1, None).unwrap();
        let before: std::vec::Vec<_> = locals
            .clones()
            .map(|(_, clone)| clone.tree.load())
            .collect();
        let mut counters = std::vec::Vec::new();
        locals.for_each_lower(tree, |free, lower| {
            assert_eq!(free, lower.stats().free_frames);
            assert!(
                locals
                    .clones()
                    .any(|(_, clone)| clone.tree.load().switching())
            );
            counters.push(free);
        });
        assert_eq!(counters, [3, 4]);
        let after: std::vec::Vec<_> = locals
            .clones()
            .map(|(_, clone)| clone.tree.load())
            .collect();
        assert_eq!(before, after);
        assert_eq!(locals.load(Class(0), 0).unwrap().free, 3);
    }

    #[test]
    fn allocation_abort_preserves_other_changes() {
        let tree = clone_with(100);
        let (_, aborted) = TreeAccess::try_start(&tree, TreeGet::new(None, 8)).unwrap();
        let (_, allocated) = TreeAccess::try_start(&tree, TreeGet::new(None, 4)).unwrap();
        allocated.commit(TreeId(0).as_row());
        let (_, freed) = TreeAccess::try_start(&tree, TreePut::new(TreeId(0), 2)).unwrap();
        freed.commit(());
        drop(aborted);

        let current = tree.tree.load();
        assert_eq!(current.free(), 98);
        assert_eq!(current.users(), 0);
    }

    #[test]
    fn switch_commit_returns_normalized_snapshot() {
        let tree = clone_with(100);
        let (_, access) = TreeAccess::try_start(&tree, TreeSwitch).unwrap();
        let old = access.commit(None);
        assert!(!old.switching());
        assert_eq!(old.as_data().unwrap().free, 100);
        assert!(!tree.tree.load().present());

        let (_, access) = TreeAccess::try_start(&tree, TreeSwitch).unwrap();
        let old = access.commit(Some(TreeData::new(TreeId(0).as_row(), 50)));
        assert!(old.as_data().is_none());
        assert_eq!(tree.tree.load().as_data().unwrap().free, 50);
    }

    #[test]
    fn drain_commit_returns_remaining_frames() {
        let tree = clone_with(100);
        let (_, access) = TreeAccess::try_start(&tree, TreeDrain::new(None, 8)).unwrap();
        let old = access.commit(());
        assert!(!old.switching());
        assert_eq!(old.as_data().unwrap().free, 92);
        assert!(!tree.tree.load().present());
    }

    #[test]
    fn switch_and_drain_local_slots() {
        let reservations = [Align(clone_with(100))];
        let free_clones = [Align(clone_with(3))];
        let mut classes = [const { None }; Class::LEN as usize];
        classes[0] = Some(0..1);
        let locals = Locals {
            classes,
            reservations: &reservations,
            free_clones: &free_clones,
        };

        let old = locals.switch_reservation(Class(0), 0, None).unwrap();
        assert_eq!(old.data.free, 100);
        assert!(locals.switch_reservation(Class(0), 0, Some(old)).is_none());
        let old = locals.switch_free_clone(0, None).unwrap();
        assert_eq!(old.data.free, 3);
        assert!(locals.switch_free_clone(0, Some(old)).is_none());

        let reserved_free = core::cell::Cell::new(0);
        let batched_free = core::cell::Cell::new(0);
        locals.drain(
            |reservation| reserved_free.set(reservation.data.free),
            |clone| batched_free.set(clone.data.free),
        );
        assert_eq!(reserved_free.get(), 100);
        assert_eq!(batched_free.get(), 3);
        assert!(locals.load(Class(0), 0).is_none());
        assert!(!free_clones[0].tree.load().present());
    }
}
