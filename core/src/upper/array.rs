use core::fmt;
use core::hint::spin_loop;
use core::mem::{align_of, size_of};
use core::ops::{Index, Range};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use log::{error, info, warn};

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::{Alloc, Init, Local, MAGIC};
use crate::atomic::Atom;
use crate::entry::{ReservedTree, Tree};
use crate::lower::LowerAlloc;
use crate::upper::CAS_RETRIES;
use crate::util::{align_down, CacheLine};
use crate::{Error, Frame, PFNRange, Result, PFN};

/// Non-Volatile global metadata
#[repr(align(0x1000))]
pub struct Meta {
    /// A magic number used to check if the persistent memory contains the allocator state
    magic: AtomicUsize,
    /// Number of frames managed by the persistent allocator
    frames: AtomicUsize,
    /// Flag that stores if the system has crashed or was shutdown correctly
    crashed: AtomicBool,
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Frame::SIZE);

/// This allocator splits its memory range into chunks.
/// These chunks are reserved by CPUs to reduce sharing.
/// Allocations/frees within the chunk are handed over to the
/// lower allocator.
/// These chunks are, due to the inner workins of the lower allocator,
/// called *trees*.
///
/// This allocator stores the level three entries (tree roots) in a
/// packed array.
/// For the reservation, the allocator simply scans the array for free entries,
/// while prioritizing partially empty chunks to avoid fragmentation.
///
/// This volatile shared metadata is rebuild on boot from
/// the persistent metadata of the lower allocator.
#[derive(Default)]
#[repr(align(64))]
pub struct Array<const F: usize, L: LowerAlloc>
where
    [(); L::N]:,
{
    /// Pointer to the metadata frame at the end of the allocators persistent memory range
    pub meta: Option<&'static Meta>,
    /// CPU local data (only shared between CPUs if the memory area is too small)
    pub local: Box<[CacheLine<Local<F>>]>,
    /// Metadata of the lower alloc
    pub lower: L,
    /// Manages the allocators trees
    pub trees: Trees<{ L::N }>,
}

unsafe impl<const F: usize, L: LowerAlloc> Send for Array<F, L> where [(); L::N]: {}
unsafe impl<const F: usize, L: LowerAlloc> Sync for Array<F, L> where [(); L::N]: {}

impl<const F: usize, L: LowerAlloc> fmt::Debug for Array<F, L>
where
    [(); L::N]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;

        writeln!(
            f,
            "    memory: {:?} ({})",
            self.lower.memory(),
            self.lower.frames()
        )?;

        writeln!(f, "    trees: {:?} ({} frames)", self.trees, L::N)?;
        let free_frames = self.free_frames();
        let free_huge_frames = self.free_huge_frames();
        writeln!(
            f,
            "    free frames: {free_frames} ({free_huge_frames} huge, {} trees)",
            free_frames.div_ceil(L::N)
        )?;

        for (t, local) in self.local.iter().enumerate() {
            writeln!(f, "    L{t:>2}: {:?}", local.preferred.load())?;
        }

        write!(f, "}}")?;
        Ok(())
    }
}

impl<const F: usize, L: LowerAlloc> Alloc for Array<F, L>
where
    [(); L::N]:,
{
    #[cold]
    fn init(
        &mut self,
        mut cores: usize,
        mut memory: Range<PFN>,
        init: Init,
        free_all: bool,
    ) -> Result<()> {
        info!("initializing c={cores} {memory:?} {}", memory.len());
        if memory.start.0 % (1 << L::MAX_ORDER) != 0 {
            error!("Unexpected memory alignment");
            return Err(Error::Initialization);
        }

        if memory.len() < L::N * cores {
            warn!("memory {} < {}", memory.len(), L::N * cores);
            if memory.len() < 1 << L::HUGE_ORDER {
                error!("Expecting at least {} frames", 1 << L::HUGE_ORDER);
                return Err(Error::Memory);
            }
            cores = 1.max(memory.len() / L::N);
        }

        if init != Init::Volatile {
            // Last frame is reserved for metadata
            let m = memory.start..PFN(memory.end.0 - 1);
            self.meta = Some(unsafe { &*m.end.as_ptr().cast() });
            memory = m;
        }

        // Init per-cpu data
        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, Default::default);
        self.local = local.into();

        // Create lower allocator
        self.lower = L::new(cores, memory.start, memory.len(), init, free_all);

        if init == Init::Recover {
            match self.recover() {
                // If the recovery fails, continue with initializing a new allocator instead
                Err(Error::Initialization) => {}
                r => return r,
            }
        }

        self.trees.init(self.frames(), free_all);

        if let Some(meta) = self.meta {
            meta.frames.store(self.frames(), Ordering::SeqCst);
            meta.magic.store(MAGIC, Ordering::SeqCst);
            meta.crashed.store(true, Ordering::SeqCst);
        }

        Ok(())
    }

    fn get(&self, core: usize, order: usize) -> Result<PFN> {
        if order > L::MAX_ORDER {
            error!("invalid order");
            return Err(Error::Memory);
        }
        // We might have more cores than cpu-local data
        let core = core % self.local.len();

        // Retry allocation up to n times if it fails due to a concurrent update
        for _ in 0..CAS_RETRIES {
            match self.get_inner(core, order) {
                Ok(frame) => return Ok(self.lower.begin().off(frame)),
                Err(Error::Retry) => continue,
                Err(Error::Memory) => return self.drain_and_steal(core, order),
                Err(e) => return Err(e),
            }
        }
        error!("Exceeding retries");
        Err(Error::Memory)
    }

    fn put(&self, core: usize, addr: PFN, order: usize) -> Result<()> {
        let frame = self.addr_to_frame(addr, order)?;

        // First free the frame in the lower allocator
        self.lower.put(frame, order)?;

        // Then update local / global counters
        let i = frame / L::N;
        let local = &self.local[core % self.local.len()];
        let max = (self.frames() - i * L::N).min(L::N);

        // Try update own tree first
        let num_frames = 1 << order;
        if let Err(tree) = local
            .preferred
            .fetch_update(|v| v.inc(num_frames, max, |s| s / L::N == i))
        {
            if tree.start() / L::N == i {
                error!("inc failed L{i}: {tree:?} o={order}");
                return Err(Error::Corruption);
            }
        } else {
            // Save the modified tree id for the push-reserve heuristic
            local.frees_push(i);
            return Ok(());
        };

        // Tree not owned by us -> update global
        match self.trees[i].fetch_update(|v| v.inc(num_frames, max)) {
            Ok(tree) => {
                let new_frames = tree.free() + num_frames;
                if !tree.reserved() && new_frames > Trees::<{ L::N }>::almost_allocated() {
                    // put-reserve optimization:
                    // Try to reserve the tree that was targeted by the recent frees
                    if local.frees_in_tree(i) && self.reserve_for_put(&local.preferred, i)? {
                        return Ok(());
                    }
                }
                local.frees_push(i);
                Ok(())
            }
            Err(tree) => {
                error!("inc failed i{i}: {tree:?} o={order}");
                Err(Error::Corruption)
            }
        }
    }

    fn is_free(&self, addr: PFN, order: usize) -> bool {
        if let Ok(frame) = self.addr_to_frame(addr, order) {
            self.lower.is_free(frame, order)
        } else {
            false
        }
    }

    fn frames(&self) -> usize {
        self.lower.frames()
    }

    fn drain(&self, core: usize) -> Result<()> {
        let local = &self.local[core % self.local.len()];
        match self.cas_reserved(&local.preferred, ReservedTree::default(), false) {
            Err(Error::Retry) => Ok(()), // ignore cas errors
            r => r,
        }
    }

    fn free_frames(&self) -> usize {
        let mut frames = 0;
        // Global array
        for tree in self.trees.entries.iter() {
            frames += tree.load().free();
        }
        // Frames allocated in reserved trees
        for local in self.local.iter() {
            frames += local.preferred.load().free();
        }
        frames
    }

    fn free_huge_frames(&self) -> usize {
        let mut counter = 0;
        self.lower.for_each_huge_frame(|_, c| {
            if c == (1 << L::HUGE_ORDER) {
                counter += 1;
            }
        });
        counter
    }

    #[cold]
    fn for_each_huge_frame(&self, f: fn(PFN, usize)) {
        self.lower
            .for_each_huge_frame(|frame, c| f(self.lower.begin().off(frame), c))
    }
}

impl<const F: usize, L: LowerAlloc> Drop for Array<F, L>
where
    [(); L::N]:,
{
    fn drop(&mut self) {
        if let Some(meta) = self.meta {
            meta.crashed.store(false, Ordering::SeqCst);
        }
    }
}

impl<const F: usize, L: LowerAlloc> Array<F, L>
where
    [(); L::N]:,
{
    /// Recover the allocator from NVM after reboot.
    /// If crashed then the level 1 frame tables are traversed and diverging counters are corrected.
    fn recover(&mut self) -> Result<()> {
        if let Some(meta) = self.meta {
            if meta.frames.load(Ordering::SeqCst) == self.frames()
                && meta.magic.load(Ordering::SeqCst) == MAGIC
            {
                info!("recover p={}", self.frames());
                // The active flag is set on boot and reset on a successful shutdown
                // If it is already set, the allocator has been crashed
                // In this case, we have to initiate a deep recovery, correcting all the counters
                let deep = meta.crashed.load(Ordering::SeqCst);
                if deep {
                    warn!("Try recover crashed allocator!");
                }

                let mut trees = Vec::with_capacity(self.frames().div_ceil(L::N));
                // Recover each tree one-by-one
                for i in 0..self.frames().div_ceil(L::N) {
                    let frame = i * L::N;
                    let frames = self.lower.recover(frame, deep)?;
                    trees.push(Atom::new(Tree::new_with(frames, false)));
                }
                self.trees.entries = trees.into();

                meta.crashed.store(true, Ordering::SeqCst);
                Ok(())
            } else {
                error!("No metadata found");
                Err(Error::Initialization)
            }
        } else {
            error!("Allocator not persistent");
            Err(Error::Initialization)
        }
    }

    /// Convert an address to the frame index
    fn addr_to_frame(&self, addr: PFN, order: usize) -> Result<usize> {
        if order > L::MAX_ORDER {
            error!("invalid order: {order} > {}", L::MAX_ORDER);
            return Err(Error::Memory);
        }

        // Check alignment and if this addr is within our address range
        if addr.0 % (1 << order) != 0 || !self.lower.memory().contains(&addr) {
            error!("invalid addr {addr} r={:?} o={order}", self.lower.memory());
            return Err(Error::Address);
        }

        Ok(addr.0 - self.lower.begin().0)
    }

    /// Try to allocate a frame with the given order
    fn get_inner(&self, core: usize, order: usize) -> Result<usize> {
        // Select local data (which can be shared between cores if we do not have enough memory)
        let local = &self.local[core];

        // Update the upper counters first
        let old = match local.preferred.fetch_update(|v| v.dec(1 << order)) {
            Ok(old) => old,
            Err(old) => {
                // If the local counter is large enough we do not have to reserve a new tree
                // Just update the local counter and reuse the current tree
                self.try_sync_with_global(&local.preferred, old)?;

                // The local tree is full -> reserve a new one
                return self.reserve_or_wait(core, order, old, false);
            }
        };

        // The start point for the search
        let start = old.start();
        // Try allocating with the lower allocator
        match self.get_lower(old, order) {
            Ok((reserved, frame)) => {
                // Success
                if order < 6 && start != reserved.start() {
                    // Save start index for lower allocations
                    let _ = local
                        .preferred
                        .compare_exchange(reserved.with_start(start), reserved);
                }
                Ok(frame)
            }
            Err(Error::Memory) => {
                // Failure (e.g. due to fragmentation)
                // Reset counters, reserve new entry and retry allocation
                warn!("alloc failed o={order} => retry");
                // Increment global to prevent race condition with concurrent reservation
                if let Err(_) = self.trees[start / L::N].fetch_update(|v| v.inc(1 << order, L::N)) {
                    error!("Counter reset failed");
                    Err(Error::Corruption)
                } else {
                    self.reserve_or_wait(core, order, old, true)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Allocate a frame in the lower allocator
    fn get_lower(&self, reserved: ReservedTree, order: usize) -> Result<(ReservedTree, usize)> {
        let frame = self.lower.get(reserved.start(), order)?;
        Ok((
            reserved
                .with_free(reserved.free() - (1 << order))
                .with_start(frame),
            frame,
        ))
    }

    /// Frees from other CPUs update the global entry -> sync free counters.
    ///
    /// If successful returns `Error::CAS` -> retry.
    /// Returns Ok if the global counter was not large enough -> fallback to normal reservation.
    fn try_sync_with_global(&self, local: &Atom<ReservedTree>, old: ReservedTree) -> Result<()> {
        let i = old.start() / L::N;
        if i < self.trees.entries.len()
            && old.free() + self.trees[i].load().free() > Trees::<{ L::N }>::almost_allocated()
        {
            if let Ok(entry) =
                self.trees[i].fetch_update(|e| e.reserved().then_some(e.with_free(0)))
            {
                if local
                    .fetch_update(|e| {
                        (e.start() / L::N == i).then_some(e.with_free(e.free() + entry.free()))
                    })
                    .is_ok()
                {
                    // Sync successfull -> retry allocation
                    return Err(Error::Retry);
                } else {
                    // undo global change
                    if self.trees[i]
                        .fetch_update(|e| Some(e.with_free(e.free() + entry.free())))
                        .is_err()
                    {
                        error!("Failed undo sync");
                        return Err(Error::Corruption);
                    }
                }
            }
        }
        Ok(())
    }

    /// Try to reserve a new tree or wait for concurrent reservations to finish.
    /// We directly try to allocate in the tree and continues the search if it is fragmented.
    ///
    /// If `fragmented`, prioritize less fragmented trees
    fn reserve_or_wait(
        &self,
        core: usize,
        order: usize,
        old: ReservedTree,
        fragmented: bool,
    ) -> Result<usize> {
        let local = &self.local[core].preferred;

        // Set the reserved flag, locking the reservation
        if !old.locked() && local.fetch_update(|v| v.toggle_locked(true)).is_ok() {
            // Try reserve new tree
            self.reserve_and_get(core, order, old, fragmented)
        } else {
            // Wait for concurrent reservation to end
            for _ in 0..CAS_RETRIES {
                let reserved = local.load();
                if !reserved.locked() {
                    // Try allocation on new tree
                    return match self.lower.get(reserved.start(), order) {
                        Err(Error::Memory) => Err(Error::Retry),
                        r => r,
                    };
                }
                spin_loop()
            }
            warn!("Timeout reservation wait");
            Err(Error::Retry)
        }
    }

    /// Reserve a new tree and allocate the frame in it
    fn reserve_and_get(
        &self,
        core: usize,
        order: usize,
        old: ReservedTree,
        fragmented: bool,
    ) -> Result<usize> {
        let local = &self.local[core].preferred;

        // Try reserve new tree
        let start = if old.has_start() {
            old.start() / L::N
        } else {
            // Different initial starting point for every core
            self.trees.entries.len() / self.local.len() * core
        };

        // Reserved a new tree an allocate a frame in it
        let cores = self.local.len();
        let (new, frame) = match self.trees.reserve(order, cores, start, fragmented, |r| {
            self.get_lower(r, order)
        }) {
            Ok(entry) => entry,
            Err(e) => {
                // Rollback: Clear reserve flag
                local
                    .fetch_update(|v| v.toggle_locked(false))
                    .map_err(|_| Error::Corruption)?;
                return Err(e);
            }
        };

        match self.cas_reserved(local, new, true) {
            Ok(_) => Ok(frame),
            Err(Error::Retry) => {
                error!("unexpected reserve state");
                Err(Error::Corruption)
            }
            Err(e) => Err(e),
        }
    }

    /// Fallback if all trees are full
    fn drain_and_steal(&self, core: usize, order: usize) -> Result<PFN> {
        // Drain all
        for core in 0..self.local.len() {
            self.drain(core)?;
        }
        // Steal drained trees
        let start = self.trees.entries.len() / self.local.len() * core;
        let (reserved, frame) = self
            .trees
            .reserve_any(start, order, |r| self.get_lower(r, order))?;

        let local = &self.local[core].preferred;
        match self.cas_reserved(local, reserved, false) {
            Ok(_) => Ok(self.lower.begin().off(frame)),
            Err(Error::Retry) => {
                error!("unexpected reserve state");
                Err(Error::Corruption)
            }
            Err(e) => Err(e),
        }
    }

    /// Reserve an entry for bulk frees
    fn reserve_for_put(&self, local: &Atom<ReservedTree>, i: usize) -> Result<bool> {
        if let Ok(entry) =
            self.trees[i].fetch_update(|v| v.reserve(Trees::<{ L::N }>::almost_allocated()..))
        {
            let entry = ReservedTree::new_with(entry.free(), i * L::N);
            match self.cas_reserved(local, entry, false) {
                Ok(_) => Ok(true),
                Err(Error::Retry) => {
                    warn!("rollback {i}");
                    // Rollback reservation
                    let max = (self.frames() - i * L::N).min(L::N);
                    if self.trees[i]
                        .fetch_update(|v| v.unreserve_add(entry.free(), max))
                        .is_err()
                    {
                        error!("put - reservation rollback failed");
                        return Err(Error::Corruption);
                    }
                    Ok(false)
                }
                Err(e) => Err(e),
            }
        } else {
            Ok(false)
        }
    }

    /// Swap the current reserved tree out replacing it with a new one.
    /// The old tree is unreserved.
    fn cas_reserved(
        &self,
        local: &Atom<ReservedTree>,
        new: ReservedTree,
        expect_locked: bool,
    ) -> Result<()> {
        debug_assert!(!new.locked());

        let old = local
            .fetch_update(|v| (v.locked() == expect_locked).then_some(new))
            .map_err(|_| Error::Retry)?;

        self.trees.unreserve(old, self.frames())
    }
}

#[derive(Default)]
pub struct Trees<const LN: usize> {
    /// Array of level 3 entries, which are the roots of the trees
    entries: Box<[Atom<Tree>]>,
}

impl<const LN: usize> Index<usize> for Trees<LN> {
    type Output = Atom<Tree>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl<const LN: usize> fmt::Debug for Trees<LN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max = self.entries.len();
        let mut free = 0;
        let mut partial = 0;
        for e in &*self.entries {
            let f = e.load().free();
            if f == LN {
                free += 1;
            } else if f > Self::almost_allocated() {
                partial += 1;
            }
        }
        write!(f, "(total: {max}, free: {free}, partial: {partial})")?;
        Ok(())
    }
}

impl<const LN: usize> Trees<LN> {
    /// Initialize the tree array
    fn init(&mut self, frames: usize, free_all: bool) {
        let len = frames.div_ceil(LN);
        let mut entries = Vec::with_capacity(len);
        if free_all {
            entries.resize_with(len - 1, || Atom::new(Tree::new_with(LN, false)));
            // The last one might be cut off
            let max = ((frames - 1) % LN) + 1;
            entries.push(Atom::new(Tree::new_with(max, false)));
        } else {
            entries.resize_with(len, || Atom::new(Tree::new()));
        }
        self.entries = entries.into();
    }

    /// Almost no free frames left
    const fn almost_allocated() -> usize {
        1 << 10 // MAX_ORDER
    }

    /// Almost all frames are free
    const fn almost_free() -> usize {
        LN - (1 << 10) // MAX_ORDER
    }

    /// Unreserve an entry, adding the local entry counter to the global one
    fn unreserve(&self, entry: ReservedTree, frames: usize) -> Result<()> {
        if !entry.has_start() {
            return Ok(());
        }

        let i = entry.start() / LN;
        let max = (frames - i * LN).min(LN);
        if let Ok(_) = self[i].fetch_update(|v| v.unreserve_add(entry.free(), max)) {
            Ok(())
        } else {
            error!("Unreserve failed i{i}");
            Err(Error::Corruption)
        }
    }

    /// Find and reserve a free tree
    fn reserve_free(
        &self,
        start: usize,
        mut get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)>,
    ) -> Result<(ReservedTree, usize)> {
        // Just search linearly through the array
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(Self::almost_free()..)) {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }
        Err(Error::Memory)
    }

    /// Find and reserve a partial tree in the vicinity
    fn reserve_partial(
        &self,
        cores: usize,
        start: usize,
        mut get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)>,
    ) -> Result<(ReservedTree, usize)> {
        const ENTRIES_PER_CACHELINE: usize = align_of::<CacheLine>() / size_of::<Tree>();
        // One quater of the per-CPU memory
        let vicinity = ((self.entries.len() / cores) / 4).max(1) as isize;

        // Positive modulo and cacheline alignment
        let start = align_down(start + self.entries.len(), ENTRIES_PER_CACHELINE) as isize;

        // Search the the array for a partially or entirely free tree
        // This speeds up the search drastically if many trees are free
        for i in 1..vicinity {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(Self::almost_allocated()..)) {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }

        // Search the rest of the array for a partially but not entirely free tree
        for i in vicinity..=self.entries.len() as isize {
            // Alternating between before and after this entry
            let off = if i % 2 == 0 { i / 2 } else { -i.div_ceil(2) };
            let i = (start + off) as usize % self.entries.len();
            if let Ok(entry) =
                self[i].fetch_update(|v| v.reserve(Self::almost_allocated()..Self::almost_free()))
            {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }
        Err(Error::Memory)
    }

    /// Fallback to search for any suitable tree
    fn reserve_any(
        &self,
        start: usize,
        order: usize,
        mut get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)>,
    ) -> Result<(ReservedTree, usize)> {
        // Just search linearly through the array
        let num_frames = 1 << order;
        for i in 0..self.entries.len() {
            let i = (i + start) % self.entries.len();
            if let Ok(entry) = self[i].fetch_update(|v| v.reserve(num_frames..)) {
                match get_lower(ReservedTree::new_with(entry.free(), i * LN)) {
                    Err(Error::Memory) => {
                        info!("Fragmentation -> continue reservation");
                        self[i]
                            .fetch_update(|v| v.unreserve_add(entry.free(), LN))
                            .map_err(|_| Error::Corruption)?;
                    }
                    r => return r,
                }
            }
        }
        info!("no frames left");
        Err(Error::Memory)
    }

    /// Reserves a new tree, prioritizing partially filled trees.
    fn reserve(
        &self,
        order: usize,
        cores: usize,
        start: usize,
        fragmented: bool,
        get_lower: impl FnMut(ReservedTree) -> Result<(ReservedTree, usize)> + Copy,
    ) -> Result<(ReservedTree, usize)> {
        if fragmented {
            // try free trees first
            match self.reserve_free(start, get_lower) {
                Err(Error::Memory) => match self.reserve_partial(cores, start, get_lower) {
                    Err(Error::Memory) => self.reserve_any(start, order, get_lower),
                    r => r,
                },
                r => r,
            }
        } else {
            match self.reserve_partial(cores, start, get_lower) {
                Err(Error::Memory) => match self.reserve_free(start, get_lower) {
                    Err(Error::Memory) => self.reserve_any(start, order, get_lower),
                    r => r,
                },
                r => r,
            }
        }
    }
}
