use core::fmt;
use core::ops::Index;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

use log::{error, warn};

use super::{Alloc, Error, Local, Result, Size, MAGIC, MAX_PAGES, MIN_PAGES};
use crate::atomic::{AStack, AStackDbg, Atomic};
use crate::entry::Entry3;
use crate::lower::LowerAlloc;
use crate::table::Table;
use crate::util::Page;

const PTE3_FULL: usize = 8 * Table::span(1);

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
pub struct ArrayAlignedAlloc<L: LowerAlloc> {
    /// Pointer to the metadata page at the end of the allocators persistent memory range
    meta: *mut Meta,
    /// Array of level 3 entries, the roots of the 1G subtrees, the lower alloc manages
    subtrees: Box<[Aligned]>,
    /// CPU local data
    local: Box<[Local]>,
    /// Metadata of the lower alloc
    lower: L,

    /// List of idx to subtrees that are not allocated at all
    empty: AStack<Entry3>,
    /// List of idx to subtrees that are partially allocated with small pages
    partial_l1: AStack<Entry3>,
    /// List of idx to subtrees that are partially allocated with huge pages
    partial_l0: AStack<Entry3>,
}

/// Cache line aligned entries to prevent false-sharing.
#[repr(align(64))]
struct Aligned(Atomic<Entry3>);

impl<L: LowerAlloc> Index<usize> for ArrayAlignedAlloc<L> {
    type Output = Atomic<Entry3>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.subtrees[index].0
    }
}

unsafe impl<L: LowerAlloc> Send for ArrayAlignedAlloc<L> {}
unsafe impl<L: LowerAlloc> Sync for ArrayAlignedAlloc<L> {}

impl<L: LowerAlloc> fmt::Debug for ArrayAlignedAlloc<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.pages()
        )?;
        for (i, entry) in self.subtrees.iter().enumerate() {
            let pte = entry.0.load();
            writeln!(f, "    {i:>3}: {pte:?}")?;
        }
        writeln!(f, "    empty: {:?}", AStackDbg(&self.empty, self))?;
        writeln!(f, "    partial_l0: {:?}", AStackDbg(&self.partial_l0, self))?;
        writeln!(f, "    partial_l1: {:?}", AStackDbg(&self.partial_l1, self))?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<L: LowerAlloc> Alloc for ArrayAlignedAlloc<L> {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], overwrite: bool) -> Result<()> {
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if memory.len() < MIN_PAGES * cores {
            error!("memory {} < {}", memory.len(), MIN_PAGES * cores);
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
        let pte3_num = Table::num_pts(2, self.lower.pages());
        let mut subtrees = Vec::with_capacity(pte3_num);
        subtrees.resize_with(pte3_num, || Aligned(Atomic::new(Entry3::new())));
        self.subtrees = subtrees.into();

        self.empty = AStack::default();
        self.partial_l0 = AStack::default();
        self.partial_l1 = AStack::default();

        warn!("init");
        if !overwrite
            && meta.pages.load(Ordering::SeqCst) == self.pages()
            && meta.magic.load(Ordering::SeqCst) == MAGIC
        {
            warn!("Recover allocator state p={}", self.pages());
            let deep = meta.active.load(Ordering::SeqCst) != 0;
            let pages = self.recover(deep)?;
            warn!("Recovered pages {pages}");
        } else {
            warn!("Setup allocator state p={}", self.pages());
            self.setup();

            meta.pages.store(self.pages(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
        }

        meta.active.store(1, Ordering::SeqCst);
        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        match size {
            Size::L2 => self.get_giant(),
            _ => self.get_lower(core, size == Size::L1),
        }
        .map(|p| unsafe { self.lower.memory().start.add(p as _) } as u64)
    }

    #[inline(never)]
    fn put(&self, _core: usize, addr: u64) -> Result<()> {
        if addr % Page::SIZE as u64 != 0 || !self.lower.memory().contains(&(addr as _)) {
            error!("invalid addr");
            return Err(Error::Memory);
        }
        let page = unsafe { (addr as *const Page).offset_from(self.lower.memory().start) } as usize;

        let i = page / Table::span(2);
        let pte = self[i].load();
        if pte.page() {
            self.put_giant(page)
        } else {
            self.put_lower(page, pte)
        }
    }

    fn pages(&self) -> usize {
        self.lower.pages()
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        let mut pages = self.pages();
        for i in 0..Table::num_pts(2, self.pages()) {
            let pte = self[i].load();
            // warn!("{i:>3}: {pte:?}");
            pages -= pte.free();
        }
        pages
    }
}

impl<L: LowerAlloc> Drop for ArrayAlignedAlloc<L> {
    fn drop(&mut self) {
        if !self.meta.is_null() {
            let meta = unsafe { &*self.meta };
            meta.active.store(0, Ordering::SeqCst);
        }
    }
}

impl<L: LowerAlloc> Default for ArrayAlignedAlloc<L> {
    #[cold]
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

impl<L: LowerAlloc> ArrayAlignedAlloc<L> {
    /// Setup a new allocator.
    #[cold]
    fn setup(&mut self) {
        self.lower.clear();

        // Add all entries to the empty list
        let pte3_num = Table::num_pts(2, self.pages());
        for i in 0..pte3_num - 1 {
            self[i].store(Entry3::new().with_free(Table::span(2)));
            self.empty.push(self, i);
        }

        // The last one may be cut off
        let max = (self.pages() - (pte3_num - 1) * Table::span(2)).min(Table::span(2));
        self[pte3_num - 1].store(Entry3::new().with_free(max));

        if max == Table::span(2) {
            self.empty.push(self, pte3_num - 1);
        } else if max > PTE3_FULL {
            self.partial_l0.push(self, pte3_num - 1);
        }
    }

    /// Recover the allocator from NVM after reboot.
    /// If `deep` then the level 1 page tables are traversed and diverging counters are corrected.
    #[cold]
    fn recover(&self, deep: bool) -> Result<usize> {
        if deep {
            error!("Try recover crashed allocator!");
        }
        let mut total = 0;
        for i in 0..Table::num_pts(2, self.pages()) {
            let page = i * Table::span(2);
            let (pages, size) = self.lower.recover(page, deep)?;
            if size == Size::L2 {
                self[i].store(Entry3::new_giant());
            } else {
                self[i].store(Entry3::new_table(pages, size, false));

                // Add to lists
                if pages == Table::span(2) {
                    self.empty.push(self, i);
                } else if pages > PTE3_FULL {
                    self.partial(size == Size::L1).push(self, i);
                }
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
    fn reserve(&self, huge: bool) -> Result<usize> {
        while let Some(i) = self.partial(huge).pop(self) {
            // Skip empty entries
            if self[i].load().free() < Table::span(2) {
                warn!("reserve partial {i}");
                self[i].update(|v| v.dec(huge)).unwrap();
                return Ok(i * Table::span(2));
            } else {
                self.empty.push(self, i);
            }
        }

        if let Some(i) = self.empty.pop(self) {
            warn!("reserve empty {i}");
            self[i].update(|v| v.dec(huge)).unwrap();
            Ok(i * Table::span(2))
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    /// Allocate a small or huge page from the lower alloc.
    fn get_lower(&self, core: usize, huge: bool) -> Result<usize> {
        let start_a = self.local[core].start(huge);
        let mut start = *start_a;

        if start == usize::MAX {
            warn!("Try reserve first");
            start = self.reserve(huge)?;
        } else {
            let i = start / Table::span(2);
            if self[i].update(|v| v.dec(huge)).is_err() {
                start = self.reserve(huge)?;
                if self[i].update(Entry3::unreserve).is_err() {
                    panic!("Unreserve failed")
                }
            }
        }

        let page = self.lower.get(core, huge, start)?;
        *start_a = page;
        Ok(page)
    }

    /// Free a small or huge page from the lower alloc.
    fn put_lower(&self, page: usize, pte: Entry3) -> Result<()> {
        let max = self
            .pages()
            .saturating_sub(Table::round(2, page))
            .min(Table::span(2));
        if pte.free() >= max {
            error!("Not allocated 0x{page:x}, {:x} >= {max:x}", pte.free());
            return Err(Error::Address);
        }

        let i = page / Table::span(2);
        let huge = self.lower.put(page)?;
        if let Ok(pte3) = self[i].update(|v| v.inc(huge, max)) {
            if !pte3.reserved() {
                let new_pages = pte3.free() + Table::span(huge as _);
                if pte3.free() <= PTE3_FULL && new_pages > PTE3_FULL {
                    // Add back to partial
                    self.partial(huge).push(self, i);
                }
            }
            Ok(())
        } else {
            error!("Corruption l3 i{i} p=-{huge:?}");
            Err(Error::Corruption)
        }
    }

    /// Allocate a giant page.
    fn get_giant(&self) -> Result<usize> {
        if let Some(i) = self.empty.pop(self) {
            if let Err(pte) =
                self[i].update(|v| (v.free() == Table::span(2)).then(Entry3::new_giant))
            {
                error!("Corruption i{i} {pte:?}");
                Err(Error::Corruption)
            } else {
                self.lower.set_giant(i * Table::span(2));
                Ok(i * Table::span(2))
            }
        } else {
            error!("No memory");
            Err(Error::Memory)
        }
    }

    /// Free a giant page.
    fn put_giant(&self, page: usize) -> Result<()> {
        let i = page / Table::span(2);
        if page % Table::span(2) != 0 {
            error!("Invalid align {page:x}");
            return Err(Error::Address);
        }

        if let Err(pte) =
            self[i].compare_exchange(Entry3::new_giant(), Entry3::new().with_free(Table::span(2)))
        {
            error!("Not allocated i{i} {pte:?}");
            Err(Error::Address)
        } else {
            self.lower.clear_giant(page);
            // Add to empty list
            self.empty.push(self, i);
            Ok(())
        }
    }
}
