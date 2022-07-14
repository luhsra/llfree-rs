use core::fmt;
use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::{error, info, warn};

use super::{Alloc, Local, MAGIC, MAX_PAGES};
use crate::atomic::{AStack, AStackDbg, Atomic};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
use crate::table::Mapping;
use crate::util::Page;
use crate::{Error, Result};

const PUTS_RESERVE: usize = 4;

/// Non-Volatile global metadata
struct Meta {
    magic: AtomicUsize,
    pages: AtomicUsize,
    active: AtomicUsize,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Page::SIZE);

/// This allocator splits its memory range into 1G chunks.
/// Giant pages are directly allocated in it.
/// For smaller pages, however, the 1G chunk is handed over to the
/// lower allocator, managing these smaller allocations.
/// These 1G chunks are, due to the inner workins of the lower allocator,
/// called 1G *subtrees*.
///
/// This allocator uses a cache-line aligned array to store the subtrees
/// (level 3 entries).
/// The subtree reservation is speed up using free lists for
/// empty and partially empty subtrees.
/// These free lists are implemented as atomic linked lists with their next
/// pointers stored inside the level 3 entries.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[repr(align(64))]
pub struct ArrayAlignedAlloc<A: Entry, L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    subtrees: Box<[A]>,
    /// CPU local data
    local: Box<[Local<PUTS_RESERVE>]>,
    /// Metadata of the lower alloc
    lower: L,

    /// List of idx to subtrees that are not allocated at all
    empty: AStack<Entry3>,
    /// List of idx to subtrees that are partially allocated with small pages
    partial_l1: AStack<Entry3>,
    /// List of idx to subtrees that are partially allocated with huge pages
    partial_l0: AStack<Entry3>,
}

pub trait Entry: Sized {
    fn new(v: Atomic<Entry3>) -> Self;
    fn as_ref(&self) -> &Atomic<Entry3>;
}

/// Cache line aligned entries to prevent false-sharing.
#[repr(align(64))]
pub struct CacheAligned(Atomic<Entry3>);
impl Entry for CacheAligned {
    fn new(v: Atomic<Entry3>) -> Self {
        Self(v)
    }
    fn as_ref(&self) -> &Atomic<Entry3> {
        &self.0
    }
}

/// *Not* cache-line aligned, to test false-sharing
#[repr(transparent)]
pub struct Unaligned(Atomic<Entry3>);
impl Entry for Unaligned {
    fn new(v: Atomic<Entry3>) -> Self {
        Self(v)
    }
    fn as_ref(&self) -> &Atomic<Entry3> {
        &self.0
    }
}

impl<A: Entry, L: LowerAlloc> Index<usize> for ArrayAlignedAlloc<A, L> {
    type Output = Atomic<Entry3>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.subtrees[index].as_ref()
    }
}

unsafe impl<A: Entry, L: LowerAlloc> Send for ArrayAlignedAlloc<A, L> {}
unsafe impl<A: Entry, L: LowerAlloc> Sync for ArrayAlignedAlloc<A, L> {}

impl<A: Entry, L: LowerAlloc> fmt::Debug for ArrayAlignedAlloc<A, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;
        for (i, entry) in self.subtrees.iter().enumerate() {
            let pte = entry.as_ref().load();
            writeln!(f, "    {i:>3}: {pte:?}")?;
        }
        writeln!(f, "    empty: {:?}", AStackDbg(&self.empty, self))?;
        writeln!(f, "    partial_l0: {:?}", AStackDbg(&self.partial_l0, self))?;
        writeln!(f, "    partial_l1: {:?}", AStackDbg(&self.partial_l1, self))?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<A: Entry, L: LowerAlloc> Alloc for ArrayAlignedAlloc<A, L> {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < Self::MAPPING.span(2) * cores {
            error!(
                "memory {} < {}",
                memory.len(),
                Self::MAPPING.span(2) * cores
            );
            return Err(Error::Memory);
        }

        // Last frame is reserved for metadata
        let (memory, rem) = memory.split_at_mut((memory.len() - 1).min(MAX_PAGES));
        let meta = rem[0].cast_mut::<Meta>();
        self.meta = meta;

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Local::new);
        self.local = local.into();

        // Create lower allocator
        self.lower = L::new(cores, memory);

        // Array with all pte3
        let pte3_num = Self::MAPPING.num_pts(2, self.lower.pages());
        let mut subtrees = Vec::with_capacity(pte3_num);
        subtrees.resize_with(pte3_num, || A::new(Atomic::new(Entry3::new())));
        self.subtrees = subtrees.into();

        self.empty = AStack::default();
        self.partial_l0 = AStack::default();
        self.partial_l1 = AStack::default();

        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            info!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            self.recover(deep)?;
        } else {
            info!("Setup allocator state p={}", self.pages());
            self.setup();

            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        meta.active.store(1, Ordering::SeqCst);
        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, order: usize) -> Result<u64> {
        if order > Self::MAX_ORDER {
            error!("invalid order: !{order} <= {}", Self::MAX_ORDER);
            return Err(Error::Memory);
        }
        let huge = order >= Self::HUGE_ORDER;

        let start_a = self.local[core].start(huge);
        let mut start = *start_a;

        if start == usize::MAX {
            start = self.reserve(order)?;
        } else {
            let i = start / Self::MAPPING.span(2);
            if self[i]
                .update(|v| v.dec(huge, 1 << order, Self::MAPPING.span(2)))
                .is_err()
            {
                start = self.reserve(order)?;
                if self[i].update(Entry3::unreserve).is_err() {
                    error!("Unreserve failed");
                    return Err(Error::Corruption);
                }
            }
        }

        let page = loop {
            match self.lower.get(start, order) {
                Ok(page) => break page,
                Err(Error::Memory) => {
                    let i = start / Self::MAPPING.span(2);
                    let max = self
                        .pages()
                        .saturating_sub(Self::MAPPING.round(2, start))
                        .min(Self::MAPPING.span(2));
                    if let Err(e) = self[i].update(|v| v.inc(huge, 1 << order, max)) {
                        error!("Counter reset failed o={order} {i}: {e:?}");
                        return Err(Error::Corruption);
                    } else {
                        start = self.reserve(order)?;
                        if self[i].update(Entry3::unreserve).is_err() {
                            error!("Unreserve failed");
                            return Err(Error::Corruption);
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        };

        *start_a = page;
        Ok(unsafe { self.lower.memory().start.add(page as _) } as u64)
    }

    #[inline(never)]
    fn put(&self, _core: usize, addr: u64, order: usize) -> Result<()> {
        if order > Self::MAX_ORDER {
            error!("invalid order: !{order} <= {}", Self::MAX_ORDER);
            return Err(Error::Memory);
        }
        let num_pages = 1 << order;
        let huge = order >= Self::HUGE_ORDER;

        if addr % (num_pages * Page::SIZE) as u64 != 0
            || !self.lower.memory().contains(&(addr as _))
        {
            error!("invalid addr {addr:x}");
            return Err(Error::Memory);
        }

        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let i = page / Self::MAPPING.span(2);
        let pte3 = self[i].load();

        let max = self
            .pages()
            .saturating_sub(Self::MAPPING.round(2, page))
            .min(Self::MAPPING.span(2));

        if pte3.free() > max - num_pages {
            error!("Not allocated 0x{page:x}, {:x} >= {max:x}", pte3.free());
            return Err(Error::Address);
        }

        self.lower.put(page, order)?;
        if let Ok(pte3) = self[i].update(|v| v.inc(huge, num_pages, max)) {
            if !pte3.reserved() {
                let new_pages = pte3.free() + num_pages;
                if pte3.free() <= Self::ALMOST_FULL && new_pages > Self::ALMOST_FULL {
                    // Add back to partial
                    self.partial(huge).push(self, i);
                }
            }
            Ok(())
        } else {
            error!("Corruption l3 i{i} o={order}");
            Err(Error::Corruption)
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let pte = self[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.free();
        }
        pages
    }
}

impl<A: Entry, L: LowerAlloc> Drop for ArrayAlignedAlloc<A, L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}

impl<A: Entry, L: LowerAlloc> Default for ArrayAlignedAlloc<A, L> {
    fn default() -> Self {
        Self {
            meta: null_mut(),
            lower: L::default(),
            local: Box::new([]),
            subtrees: Box::new([]),
            empty: AStack::default(),
            partial_l1: AStack::default(),
            partial_l0: AStack::default(),
        }
    }
}

impl<A: Entry, L: LowerAlloc> ArrayAlignedAlloc<A, L> {
    const MAPPING: Mapping<2> = L::MAPPING;
    const ALMOST_FULL: usize = Self::MAPPING.span(2) / 64;
    const HUGE_ORDER: usize = L::HUGE_ORDER;
    const MAX_ORDER: usize = L::MAX_ORDER;

    /// Setup a new allocator.
    #[cold]
    fn setup(&mut self) {
        self.lower.clear();

        // Add all entries to the empty list
        let pte3_num = Self::MAPPING.num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            self[i].store(Entry3::empty(Self::MAPPING.span(2)));
            self.empty.push(self, i);
        }

        // The last one may be cut off
        let max =
            (self.pages() - (pte3_num - 1) * Self::MAPPING.span(2)).min(Self::MAPPING.span(2));
        self[pte3_num - 1].store(Entry3::new().with_free(max));

        if max == Self::MAPPING.span(2) {
            self.empty.push(self, pte3_num - 1);
        } else if max > Self::ALMOST_FULL {
            self.partial_l0.push(self, pte3_num - 1);
        }
    }

    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover(&self, deep: bool) -> Result<usize> {
        if deep {
            warn!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..Self::MAPPING.num_pts(2, self.pages()) {
            let page = i * Self::MAPPING.span(2);
            let (pages, huge) = self.lower.recover(page, deep)?;

            self[i].store(Entry3::new_table(pages, huge, false));

            // Add to lists
            if pages == Self::MAPPING.span(2) {
                self.empty.push(self, i);
            } else if pages > Self::ALMOST_FULL {
                self.partial(huge).push(self, i);
            }
            total += pages;
        }
        Ok(total)
    }

    fn partial(&self, huge: bool) -> &AStack<Entry3> {
        if huge {
            &self.partial_l1
        } else {
            &self.partial_l0
        }
    }

    /// Reserves a new subtree, prioritizing partially filled subtrees,
    /// and allocates a page from it in one step.
    fn reserve(&self, order: usize) -> Result<usize> {
        let huge = order >= Self::HUGE_ORDER;
        while let Some((i, r)) = self.partial(huge).pop_update(self, |v| {
            // Skip empty entries
            if v.free() < Self::MAPPING.span(2) {
                v.dec(huge, 1 << order, Self::MAPPING.span(2))
            } else {
                None
            }
        }) {
            if r.is_ok() {
                info!("reserve partial {i}");
                return Ok(i * Self::MAPPING.span(2));
            }
            self.empty.push(self, i);
        }

        if let Some((i, r)) = self
            .empty
            .pop_update(self, |v| v.dec(huge, 1 << order, Self::MAPPING.span(2)))
        {
            debug_assert!(r.is_ok());
            info!("reserve empty {i}");
            Ok(i * Self::MAPPING.span(2))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }
}
