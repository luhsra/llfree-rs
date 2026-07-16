//! Reimplementation of Linux's the buddy allocator system

use core::cell::UnsafeCell;
use core::mem::{size_of, size_of_val};
use core::{fmt, slice};

use llfree::{
    Alloc, Class, ClassStats, Error, FrameId, HUGE_FRAMES, Init, MetaData, MetaSize, Policy,
    PolicyFn, Request, Stats, TREE_ORDER, TreeChange, TreeMatch, TreeOperation, TreeStats,
};
use log::error;
use spin::mutex::SpinMutex;

/// Reimplementation of Linux's buddy allocator system without the PCP lists.
pub struct Buddy<'a> {
    /// Access to the struct pages is unsafe, similar to Linux's buddy allocator
    pages: &'a [UnsafeCell<StructPage>],
    /// Free lists are indexed by order and Linux migrate type.
    free_area: SpinMutex<FreeAreas>,
    /// Selects migrate type
    policy: PolicyFn,
}

unsafe impl<'a> Send for Buddy<'a> {}
unsafe impl<'a> Sync for Buddy<'a> {}

#[repr(align(64))]
#[derive(Default)]
struct StructPage {
    /// Zero means free, non-zero means allocated.
    private: u8,
    /// The order is meaningful for the head of a free block.
    order: u8,
    free_class: u8,
    in_list: bool,
    offline: bool,
    buddy_list: ListHead,
}
impl StructPage {
    const fn new() -> Self {
        Self {
            private: 0,
            order: 0,
            free_class: 0,
            in_list: false,
            offline: false,
            buddy_list: ListHead::new(),
        }
    }
}

#[derive(Copy, Clone, Default)]
struct FreeArea {
    free_list: ListHead,
    /// Number of blocks, as in Linux's `nr_free`.
    nr_free: usize,
}

/// Maximum order representable by a `usize` frame index.
const MAX_ORDER: usize = TREE_ORDER;
const LIST_NONE: usize = usize::MAX;
type FreeAreas = [[Option<FreeArea>; Class::LEN as usize]; MAX_ORDER + 1];

/// Doubly linked list head for free areas.
/// This contains indices into the struct pages.
#[derive(Copy, Clone, Default)]
struct ListHead {
    next: usize,
    prev: usize,
}

impl ListHead {
    const fn new() -> Self {
        Self {
            next: LIST_NONE,
            prev: LIST_NONE,
        }
    }
}

impl<'a> Alloc<'a> for Buddy<'a> {
    fn name() -> &'static str {
        "Buddy"
    }

    fn new(
        frames: usize,
        init: Init,
        classing: &llfree::Classing,
        meta: MetaData<'a>,
    ) -> llfree::Result<Self> {
        let pages: &mut [UnsafeCell<StructPage>] = unsafe {
            slice::from_raw_parts_mut(
                meta.trees.as_mut_ptr().cast(),
                size_of_val(meta.trees) / size_of::<StructPage>(),
            )
        };
        if pages.len() < frames {
            return Err(Error::Argument);
        }

        let mut free_area: FreeAreas = [[const { None }; _]; _];
        for order in &mut free_area {
            for (class, _count) in classing.classes() {
                order[class.0 as usize] = Some(FreeArea {
                    free_list: ListHead::new(),
                    nr_free: 0,
                });
            }
        }
        for page in &pages[..frames] {
            unsafe { page.get().write(StructPage::new()) };
        }
        if matches!(init, Init::FreeAll | Init::AllocAll) {
            for page in &pages[..frames] {
                unsafe { (*page.get()).private = 1 };
            }
        }

        let this = Self {
            pages: &pages[..frames],
            free_area: SpinMutex::new(free_area),
            policy: classing.policy,
        };

        match init {
            Init::None | Init::AllocAll => {}
            Init::Recover => {
                error!("Invalid init mode");
                return Err(Error::Initialization);
            }
            Init::FreeAll => {
                for frame in 0..frames {
                    this.put(FrameId(frame), Request::new(0, classing.default, None))
                        .unwrap();
                }
            }
        }
        Ok(this)
    }

    fn metadata_size(_classing: &llfree::Classing, frames: usize) -> MetaSize {
        MetaSize {
            local: 0,
            lower: 0,
            trees: size_of::<StructPage>() * frames,
        }
    }

    unsafe fn metadata(&mut self) -> MetaData<'a> {
        MetaData {
            local: &mut [],
            lower: &mut [],
            trees: unsafe {
                slice::from_raw_parts_mut(
                    self.pages.as_ptr().cast_mut().cast(),
                    size_of_val(self.pages),
                )
            },
        }
    }

    /// Return the frame and class for the given `frame` and `order`.
    fn get(
        &self,
        frame: Option<FrameId>,
        flags: Request,
    ) -> llfree::Result<(FrameId, llfree::Class)> {
        if flags.order > MAX_ORDER
            || (1usize << flags.order) > self.frames()
            || !self.valid_class(flags.class)
        {
            return Err(Error::Argument);
        }
        let wanted = 1usize << flags.order;
        let mut areas = self.free_area.lock();
        let mut candidate = None;

        for order in flags.order..=MAX_ORDER {
            let size = 1usize << order;
            for class in 0..Class::LEN as usize {
                let Some(area) = areas[order][class].as_ref() else {
                    continue;
                };

                let policy = (self.policy)(flags.class, Class(class as u8), area.nr_free * size);
                let score = match policy {
                    Policy::Match(score) => score as usize + 1,
                    Policy::Demote | Policy::Steal => 1,
                    Policy::Invalid => continue,
                };
                let mut current = area.free_list.next;
                while current != LIST_NONE {
                    if frame.is_none_or(|f| current <= f.0 && f.0 - current < size) {
                        let better = candidate.is_none_or(|(_, _, old_order, old_score)| {
                            score > old_score || (score == old_score && order < old_order)
                        });
                        if better {
                            candidate = Some((current, Class(class as u8), order, score));
                        }
                        if frame.is_some() {
                            break;
                        }
                    }
                    current = unsafe { &*self.pages[current].get() }.buddy_list.next;
                }
            }
            if candidate.is_some() && frame.is_none() {
                break;
            }
        }

        let Some((mut block, target, mut order, _)) = candidate else {
            return Err(Error::Memory);
        };
        let policy = (self.policy)(
            flags.class,
            target,
            areas[order][target.0 as usize].as_ref().unwrap().nr_free * (1usize << order),
        );
        let result_class = if matches!(policy, Policy::Demote) {
            flags.class
        } else {
            target
        };

        unsafe {
            self.list_del(areas[order][target.0 as usize].as_mut().unwrap(), block);
            while order > flags.order {
                order -= 1;
                let other = block + (1usize << order);
                let (left, split) = if frame.is_some_and(|f| f.0 >= other) {
                    (other, block)
                } else {
                    (block, other)
                };
                let page = &mut *self.pages[split].get();
                page.private = 0;
                page.order = order as u8;
                self.list_add(
                    areas[order][target.0 as usize].as_mut().unwrap(),
                    split,
                    target,
                );
                block = left;
            }
            for i in block..block + wanted {
                (*self.pages[i].get()).private = 1;
                (*self.pages[i].get()).order = 0;
                (*self.pages[i].get()).free_class = result_class.0;
            }
        }
        Ok((FrameId(block), result_class))
    }

    fn put(&self, frame: FrameId, flags: Request) -> llfree::Result<()> {
        if flags.order > MAX_ORDER || !self.valid_class(flags.class) {
            return Err(Error::Argument);
        }
        let size = 1usize << flags.order;
        if frame.0 >= self.frames()
            || !frame.is_aligned(flags.order)
            || frame
                .0
                .checked_add(size)
                .is_none_or(|end| end > self.frames())
        {
            return Err(Error::Argument);
        }
        let mut areas = self.free_area.lock();
        if (frame.0..frame.0 + size).any(|i| unsafe { &*self.pages[i].get() }.private == 0) {
            return Err(Error::Argument);
        }
        let class = flags.class;
        let mut block = frame.0;
        let mut order = flags.order;
        unsafe {
            for i in block..block + size {
                (*self.pages[i].get()).private = 0;
            }
            while order < MAX_ORDER {
                let buddy = block ^ (1usize << order);
                if buddy + (1usize << order) > self.frames() {
                    break;
                }
                let Some(buddy_class) = self.buddy_class(buddy, order) else {
                    break;
                };
                self.list_del(
                    areas[order][buddy_class.0 as usize].as_mut().unwrap(),
                    buddy,
                );
                block = block.min(buddy);
                order += 1;
            }
            let head = &mut *self.pages[block].get();
            head.order = order as u8;
            self.list_add(
                areas[order][class.0 as usize].as_mut().unwrap(),
                block,
                class,
            );
        }
        Ok(())
    }

    fn change_tree(&self, matcher: TreeMatch, change: TreeChange) -> llfree::Result<()> {
        if matcher.class.is_some_and(|class| !self.valid_class(class))
            || change.class.is_some_and(|class| !self.valid_class(class))
        {
            return Err(Error::Argument);
        }
        let tree_frames = 1usize << TREE_ORDER;
        let mut areas = self.free_area.lock();
        let mut found = None;

        if matches!(change.operation, Some(TreeOperation::Online)) {
            for tree in 0..self.frames().div_ceil(tree_frames) {
                let base = tree * tree_frames;
                if base >= self.frames() {
                    continue;
                }
                let page = unsafe { &*self.pages[base].get() };
                if !page.offline
                    || matcher.id.is_some_and(|id| id.0 != tree)
                    || matcher
                        .class
                        .is_some_and(|class| class != Class(page.free_class))
                {
                    continue;
                }
                found = Some((base, change.class.unwrap_or(Class(page.free_class))));
                break;
            }
            let Some((base, class)) = found else {
                return Err(Error::Memory);
            };
            unsafe {
                self.list_add(
                    areas[TREE_ORDER][class.0 as usize].as_mut().unwrap(),
                    base,
                    class,
                );
            }
            return Ok(());
        }

        for class in 0..Class::LEN as usize {
            let Some(area) = areas[TREE_ORDER][class].as_ref() else {
                continue;
            };
            let mut current = area.free_list.next;
            while current != LIST_NONE {
                let tree = current / tree_frames;
                if matcher.id.is_none_or(|id| id.0 == tree)
                    && matcher
                        .class
                        .is_none_or(|wanted| wanted == Class(class as u8))
                    && matcher.free <= tree_frames
                {
                    found = Some((current, Class(class as u8)));
                    break;
                }
                current = unsafe { &*self.pages[current].get() }.buddy_list.next;
            }
            if found.is_some() {
                break;
            }
        }
        let Some((base, class)) = found else {
            return Err(Error::Memory);
        };
        if let Some(TreeOperation::Offline) = change.operation {
            unsafe {
                self.list_del(areas[TREE_ORDER][class.0 as usize].as_mut().unwrap(), base);
                (*self.pages[base].get()).offline = true;
            }
            return Ok(());
        }
        if let Some(new_class) = change.class {
            unsafe {
                self.list_del(areas[TREE_ORDER][class.0 as usize].as_mut().unwrap(), base);
                self.list_add(
                    areas[TREE_ORDER][new_class.0 as usize].as_mut().unwrap(),
                    base,
                    new_class,
                );
            }
            return Ok(());
        }
        Err(Error::Argument)
    }

    fn frames(&self) -> usize {
        self.pages.len()
    }

    fn tree_stats(&self) -> TreeStats {
        let mut free_frames = 0;
        let mut classes = [const {
            ClassStats {
                free_frames: 0,
                alloc_frames: 0,
            }
        }; 1 << Class::BITS];
        let areas = self.free_area.lock();
        for (order, order_areas) in areas.iter().enumerate() {
            for (class, area) in order_areas
                .iter()
                .enumerate()
                .filter_map(|(class, area)| area.as_ref().map(|area| (class, area)))
            {
                let frames = area.nr_free * (1usize << order);
                free_frames += frames;
                classes[class].free_frames += frames;
            }
        }
        for page in self.pages {
            let page = unsafe { &*page.get() };
            if page.private != 0 {
                classes[page.free_class as usize].alloc_frames += 1;
            }
        }
        for tree in 0..self.frames().div_ceil(1usize << TREE_ORDER) {
            let page = unsafe { &*self.pages[tree << TREE_ORDER].get() };
            if page.offline {
                classes[page.free_class as usize].alloc_frames +=
                    (1usize << TREE_ORDER).min(self.frames() - (tree << TREE_ORDER));
            }
        }
        TreeStats {
            free_frames,
            free_trees: 0,
            classes,
        }
    }

    fn stats(&self) -> Stats {
        let mut free = 0;
        let mut huge = 0;
        let mut is_huge = false;
        for (i, page) in self.pages.iter().enumerate() {
            if unsafe { &*page.get() }.private == 0 {
                free += 1;
                if i.is_multiple_of(HUGE_FRAMES) {
                    is_huge = true;
                }
                if (i + 1).is_multiple_of(HUGE_FRAMES) && is_huge {
                    huge += 1;
                }
            } else {
                is_huge = false;
            }
        }
        Stats {
            free_frames: free,
            free_huge: huge,
            free_trees: 0,
        }
    }

    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        let num = 1 << order;
        assert!(frame.0.is_multiple_of(num));
        let mut free = 0;
        let mut huge = 0;
        let mut is_huge = false;
        for i in frame.0..frame.0 + num {
            if unsafe { &*self.pages[i].get() }.private == 0 {
                free += 1;
                if i.is_multiple_of(HUGE_FRAMES) {
                    is_huge = true;
                }
                if (i + 1).is_multiple_of(HUGE_FRAMES) && is_huge {
                    huge += 1;
                }
            } else {
                is_huge = false;
            }
        }
        Stats {
            free_frames: free,
            free_huge: huge,
            free_trees: 0,
        }
    }
}

impl Buddy<'_> {
    fn valid_class(&self, class: Class) -> bool {
        class.0 < Class::LEN && self.free_area.lock()[0][class.0 as usize].is_some()
    }

    unsafe fn list_add(&self, area: &mut FreeArea, index: usize, class: Class) {
        let next = area.free_list.next;
        let page = unsafe { &mut *self.pages[index].get() };
        page.buddy_list.next = next;
        page.buddy_list.prev = LIST_NONE;
        page.free_class = class.0;
        page.in_list = true;
        page.offline = false;
        if next == LIST_NONE {
            area.free_list.prev = index;
        } else {
            unsafe { &mut *self.pages[next].get() }.buddy_list.prev = index;
        }
        area.free_list.next = index;
        area.nr_free += 1;
    }

    unsafe fn list_del(&self, area: &mut FreeArea, index: usize) {
        let page = unsafe { &*self.pages[index].get() };
        let next = page.buddy_list.next;
        let prev = page.buddy_list.prev;
        if prev == LIST_NONE {
            area.free_list.next = next;
        } else {
            unsafe { &mut *self.pages[prev].get() }.buddy_list.next = next;
        }
        if next == LIST_NONE {
            area.free_list.prev = prev;
        } else {
            unsafe { &mut *self.pages[next].get() }.buddy_list.prev = prev;
        }
        let removed = unsafe { &mut *self.pages[index].get() };
        removed.buddy_list = ListHead::new();
        removed.in_list = false;
        area.nr_free -= 1;
    }

    fn buddy_class(&self, frame: usize, order: usize) -> Option<Class> {
        if frame >= self.frames() {
            return None;
        }
        let page = unsafe { &*self.pages[frame].get() };
        if page.private != 0 || page.order as usize != order || !page.in_list {
            return None;
        }
        Some(Class(page.free_class))
    }
}

impl fmt::Debug for Buddy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buddy")
            .field("pages", &self.pages.len())
            .field("free_area", &self.free_area)
            .finish()
    }
}
impl fmt::Debug for FreeArea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeArea")
            .field("nr_free", &self.nr_free)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocator(frames: usize) -> Buddy<'static> {
        let (classing, _) = llfree::Classing::simple(1);
        let meta = Box::leak(Box::new(MetaData::alloc(&Buddy::metadata_size(
            &classing, frames,
        ))));
        Buddy::new(frames, Init::FreeAll, &classing, unsafe {
            core::ptr::read(meta)
        })
        .unwrap()
    }

    #[test]
    fn allocates_and_splits_buddy_blocks() {
        let alloc = allocator(16);
        let request = Request::new(2, Class(0), None);
        let (first, _) = alloc.get(None, request).unwrap();
        assert!(first.is_aligned(2));
        assert_eq!(alloc.stats().free_frames, 12);
        assert_eq!(alloc.tree_stats().free_frames, 12);
    }

    #[test]
    fn freeing_coalesces_back_to_one_block() {
        let alloc = allocator(16);
        let request = Request::new(2, Class(0), None);
        let (frame, _) = alloc.get(None, request).unwrap();
        alloc.put(frame, request).unwrap();
        assert_eq!(alloc.stats().free_frames, 16);
        assert_eq!(alloc.tree_stats().free_frames, 16);
        assert_eq!(
            alloc.get(None, Request::new(4, Class(0), None)).unwrap().0,
            FrameId(0)
        );
    }

    #[test]
    fn requested_frame_is_honored() {
        let alloc = allocator(16);
        let request = Request::new(1, Class(0), None);
        let (frame, _) = alloc.get(Some(FrameId(8)), request).unwrap();
        assert_eq!(frame, FrameId(8));
    }

    #[test]
    fn invalid_class_returns_argument_error() {
        let alloc = allocator(16);
        let request = Request::new(0, Class(Class::LEN), None);
        assert_eq!(alloc.get(None, request), Err(Error::Argument));
        assert_eq!(alloc.put(FrameId(0), request), Err(Error::Argument));
    }
}
