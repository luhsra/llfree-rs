//! Lower allocator implementations

use core::mem::{align_of, size_of};
use core::slice;
use core::sync::atomic::{AtomicU16, AtomicU32};

use bitfield_struct::bitfield;
use log::{error, info, warn};

use crate::atomic::{Atom, AtomArray, Atomic};
use crate::util::{Align, Deferred, align_down, size_of_slice, spin_wait};
use crate::{
    Error, HUGE_FRAMES, HUGE_ORDER, Init, MAX_ORDER, RETRIES, Result, Stats, TREE_FRAMES, TREE_HUGE,
};

const CHILDREN: usize = HUGE_FRAMES / crate::bitfield::Bitfield::<1>::ROW_BITS;
type Bitfield = crate::bitfield::Bitfield<CHILDREN>;
const _: () = assert!(Bitfield::LEN == HUGE_FRAMES);

/// Lower-level frame allocator.
///
/// This level implements the actual allocation/free operations.
/// Each allocation/free is limited to a chunk of [LowerAlloc::N] frames.
///
/// Here the bitfields are 512 bit large -> strong focus on huge frames.
/// Upon that is a table for each tree, with an entry per bitfield.
///
/// The parameter `HP` configures the number of table entries (huge frames per tree).
/// It has to be a multiple of 2!
///
/// ## Memory Layout
/// **persistent:**
/// ```text
/// NVRAM: [ Frames | Bitfields | Tables | Zone ]
/// ```
/// **volatile:**
/// ```text
/// RAM: [ Frames ], Bitfields and Tables are allocated elswhere
/// ```
#[derive(Default, Debug)]
pub struct Lower<'a> {
    len: usize,
    bitfields: &'a [Align<Bitfield>],
    children: &'a [Align<[Atom<HugeEntry>; TREE_HUGE]>],
}

unsafe impl Send for Lower<'_> {}
unsafe impl Sync for Lower<'_> {}

const _: () = assert!(TREE_HUGE < (1 << (u16::BITS as usize - HUGE_ORDER)));

/// Size of the dynamic metadata
struct Metadata {
    bitfield_len: usize,
    bitfield_size: usize,
    table_len: usize,
    table_size: usize,
}

impl Metadata {
    fn new(frames: usize) -> Self {
        let bitfield_len = frames.div_ceil(Bitfield::LEN);
        let table_len = frames.div_ceil(TREE_FRAMES);
        Self {
            bitfield_len,
            // This also respects the cache line alignment
            bitfield_size: size_of_slice::<Bitfield>(bitfield_len),
            table_len,
            table_size: size_of_slice::<Align<[HugeEntry; TREE_HUGE]>>(table_len),
        }
    }
}

impl<'a> Lower<'a> {
    pub fn metadata_size(frames: usize) -> usize {
        let m = Metadata::new(frames);
        m.bitfield_size + m.table_size
    }

    /// Create a new lower allocator.
    pub fn new(frames: usize, init: Init, primary: &'a mut [u8]) -> Result<Self> {
        let m = Metadata::new(frames);

        if primary.len() < m.bitfield_size + m.table_size
            || !(primary.as_ptr() as usize).is_multiple_of(align_of::<Align>())
        {
            error!("primary metadata");
            return Err(Error::Initialization);
        }
        let (bitfields, children) = primary.split_at_mut(m.bitfield_size);

        // Start of the l1 table array
        let bitfields =
            unsafe { slice::from_raw_parts_mut(bitfields.as_mut_ptr().cast(), m.bitfield_len) };

        // Start of the l2 table array
        let children =
            unsafe { slice::from_raw_parts_mut(children.as_mut_ptr().cast(), m.table_len) };

        let alloc = Self {
            len: frames,
            bitfields,
            children,
        };

        match init {
            Init::FreeAll => alloc.free_all(),
            Init::AllocAll => alloc.reserve_all(),
            Init::Recover(false) | Init::None => {} // skip, assuming everything is valid
            Init::Recover(true) => alloc.recover(),
        }
        Ok(alloc)
    }

    pub fn frames(&self) -> usize {
        self.len
    }

    pub fn metadata(&mut self) -> &'a mut [u8] {
        let len = Self::metadata_size(self.frames());
        unsafe { slice::from_raw_parts_mut(self.bitfields.as_ptr().cast_mut().cast(), len) }
    }

    /// Recovers the data structures for the [LowerAlloc::N] sized chunk at `start`.
    /// This corrects any data corrupted by a crash.
    pub fn recover(&self) {
        for (i, table) in self.children.iter().enumerate() {
            for (j, a_entry) in table.iter().enumerate() {
                let start = i * TREE_FRAMES + j * Bitfield::LEN;
                let entry = a_entry.load();

                if entry.huge() {
                    // Check that underlying bitfield is empty
                    let p = self.bitfields[start / Bitfield::LEN].count_zeros();
                    if p != Bitfield::LEN {
                        warn!("Invalid L2 start=0x{start:x} i{i}: h != {p}");
                        self.bitfields[start / Bitfield::LEN].fill(false);
                    }
                } else {
                    // Check the bitfield has the same number of zero bits
                    let zeros = self.bitfields[start / Bitfield::LEN].count_zeros();
                    if entry.free() != zeros {
                        warn!(
                            "Invalid L2 start=0x{start:x} i{i}: {} != {zeros}",
                            entry.free()
                        );
                        a_entry.store(HugeEntry::new_with(zeros));
                    }
                }
            }
        }
    }

    /// Try allocating a new `frame` in the [LowerAlloc::N] sized chunk at `start`.
    ///
    /// Returns the allocated frame and whether a new huge frame was fragmented.
    pub fn get(&self, start: usize, order: usize) -> Result<(usize, bool)> {
        debug_assert!(start < self.frames());

        let tree_start = align_down(start, TREE_FRAMES);
        let child_off = (start / Bitfield::LEN) % TREE_HUGE;
        let children = &self.children[start / TREE_FRAMES];

        if order == MAX_ORDER {
            let table_pair = self.child_pairs(start);
            for i in 0..TREE_HUGE / 2 {
                let i = (child_off / 2 + i) % (TREE_HUGE / 2);
                if let Ok(_) = table_pair[i].fetch_update(|v| v.map(|v| v.mark_huge())) {
                    return Ok((tree_start + 2 * i * Bitfield::LEN, true));
                }
            }
        } else if order == HUGE_ORDER {
            for i in 0..TREE_HUGE {
                let i = (child_off + i) % TREE_HUGE;
                if let Ok(_) = children[i].fetch_update(|v| v.mark_huge()) {
                    return Ok((tree_start + i * Bitfield::LEN, true));
                }
            }
        } else if order <= Bitfield::ORDER {
            let first_child = tree_start / Bitfield::LEN;
            let start_row = (start / Bitfield::ROW_BITS) % Bitfield::ROWS;

            for j in 0..TREE_HUGE {
                let i = (child_off + j) % TREE_HUGE;
                if let Ok(child) = children[i].fetch_update(|v| v.dec(1 << order)) {
                    let undo = Deferred::new(move || {
                        children[i]
                            .fetch_update(|v| v.inc(1 << order))
                            .expect("Undo failed");
                    });

                    // start with the bitfield row from the last allocation
                    let row = if j == 0 { start_row } else { 0 };

                    let bf_i = first_child + i;
                    if let Ok(offset) = self.bitfields[bf_i].set_first_zeros(row, order) {
                        undo.cancel();
                        return Ok((bf_i * Bitfield::LEN + offset, child.free() == Bitfield::LEN));
                    }
                }
            }
        } else {
            unreachable!("Order {order} is not supported")
        }
        info!("Nothing found o={order}");
        Err(Error::Memory)
    }

    /// Try allocating a specific `frame`.
    pub fn get_at(&self, frame: usize, order: usize) -> Result<bool> {
        debug_assert!(frame.is_multiple_of(1 << order));
        debug_assert!(frame + (1 << order) <= self.frames());

        let i = (frame / Bitfield::LEN) % TREE_HUGE;
        let children = &self.children[frame / TREE_FRAMES][i];

        if order == MAX_ORDER {
            if let Ok(_) = self.child_pairs(frame)[i / 2].fetch_update(|v| v.map(|v| v.mark_huge()))
            {
                return Ok(true);
            }
        } else if order == HUGE_ORDER {
            if let Ok(_) = children.fetch_update(|v| v.mark_huge()) {
                return Ok(true);
            }
        } else if order <= Bitfield::ORDER {
            if let Ok(old) = children.fetch_update(|v| v.dec(1 << order)) {
                let deferred = Deferred::new(move || {
                    children.fetch_update(|v| v.inc(1 << order)).unwrap();
                });

                if let Ok(_) = self.bitfields[frame / Bitfield::LEN].toggle(
                    frame % Bitfield::LEN,
                    order,
                    false,
                ) {
                    deferred.cancel();
                    return Ok(old.free() == Bitfield::LEN);
                }
            }
        } else {
            unreachable!("Order {order} is not supported")
        }
        Err(Error::Memory)
    }

    /// Free single frame, returning whether a whole huge page has become free.
    pub fn put(&self, frame: usize, order: usize) -> Result<bool> {
        debug_assert!(frame.is_multiple_of(1 << order));
        debug_assert!(frame + (1 << order) <= self.frames());

        let i = (frame / Bitfield::LEN) % TREE_HUGE;
        let children = &self.children[frame / TREE_FRAMES];

        if order == MAX_ORDER {
            let table_pair = self.child_pairs(frame);
            if let Err(old) = table_pair[i / 2].compare_exchange(
                HugePair(HugeEntry::new_huge(), HugeEntry::new_huge()),
                HugePair(
                    HugeEntry::new_with(Bitfield::LEN),
                    HugeEntry::new_with(Bitfield::LEN),
                ),
            ) {
                error!("Addr {frame:x} o={order} {old:?}");
                Err(Error::Address)
            } else {
                Ok(true)
            }
        } else if order == HUGE_ORDER {
            if let Err(old) = children[i]
                .compare_exchange(HugeEntry::new_huge(), HugeEntry::new_with(Bitfield::LEN))
            {
                error!("Addr {frame:x} o={order} {old:?}");
                Err(Error::Address)
            } else {
                Ok(true)
            }
        } else if order <= Bitfield::ORDER {
            let old = children[i].load();
            if old.huge() {
                self.partial_put_huge(old, frame, order)
            } else if old.free() <= Bitfield::LEN - (1 << order) {
                self.put_small(frame, order)
            } else {
                error!("Addr {frame:x} o={order} {old:?}");
                Err(Error::Address)
            }
        } else {
            unreachable!("Order {order} is not supported")
        }
    }

    /// Returns if the frame is free. This might be racy!
    pub fn is_free(&self, frame: usize, order: usize) -> bool {
        debug_assert!(frame.is_multiple_of(1 << order));
        debug_assert!(frame + (1 << order) <= self.frames());

        let i = (frame / Bitfield::LEN) % TREE_HUGE;
        let children = &self.children[frame / TREE_FRAMES];

        if 1 << order == TREE_FRAMES {
            children.iter().all(|e| e.load().free() == Bitfield::LEN)
        } else if order == MAX_ORDER {
            // multiple huge frames
            self.child_pairs(frame)[i / 2]
                .load()
                .all(|e| e.free() == Bitfield::LEN)
        } else if order == HUGE_ORDER {
            children[i].load().free() == Bitfield::LEN
        } else if order <= Bitfield::ORDER {
            let child = children[i].load();
            if child.free() < (1 << order) {
                false
            } else if child.free() == Bitfield::LEN {
                true
            } else {
                let bitfield = &self.bitfields[frame / Bitfield::LEN];
                bitfield.is_zero(frame % Bitfield::LEN, order)
            }
        } else {
            unreachable!("Order {order} is not supported")
        }
    }

    /// Debug function returning number of free frames in each order 9 chunk
    pub fn for_each_child<F: FnMut(usize, usize)>(&self, mut f: F) {
        for (ti, table) in self.children.iter().enumerate() {
            for (ci, child) in table.iter().enumerate() {
                f(ti * TREE_HUGE + ci, child.load().free())
            }
        }
    }

    /// Returns statistics.
    pub fn stats(&self) -> Stats {
        let mut stats = Stats::default();
        self.for_each_child(|_, f| {
            stats.free_frames += f;
            stats.free_huge += (f == HUGE_FRAMES) as usize;
            stats.free_trees += (f == TREE_FRAMES) as usize;
        });
        stats
    }

    /// Returns statistics at a specific frame, huge frame, or tree.
    pub fn stats_at(&self, frame: usize, order: usize) -> Stats {
        const TREE_ORDER: usize = TREE_FRAMES.ilog2() as usize;
        let children = &self.children[frame / TREE_FRAMES];
        let i = frame / Bitfield::LEN;
        match order {
            0 => Stats {
                free_frames: (children[i % TREE_HUGE].load().free() > 0
                    && self.bitfields[i].is_zero(frame % Bitfield::LEN, 0))
                    as usize,
                free_huge: 0,
                free_trees: 0,
            },
            HUGE_ORDER => {
                let free = children[i % TREE_HUGE].load().free();
                Stats {
                    free_frames: free,
                    free_huge: free / HUGE_FRAMES,
                    free_trees: 0,
                }
            }
            TREE_ORDER => {
                let mut stats = children.iter().fold(Stats::default(), |mut acc, e| {
                    let f = e.load().free();
                    acc.free_frames += f;
                    acc.free_huge += f / HUGE_FRAMES;
                    acc
                });
                stats.free_trees = stats.free_frames / TREE_FRAMES;
                stats
            }
            _ => Stats::default(),
        }
    }

    /// Returns the table with pair entries that can be updated at once.
    fn child_pairs(&self, frame: usize) -> &[Atom<HugePair>; TREE_HUGE / 2] {
        let children = &self.children[frame / TREE_FRAMES];
        unsafe { &*children.as_ptr().cast() }
    }

    fn free_all(&self) {
        // Init tables
        let (last, tables) = self.children.split_last().unwrap();
        // Table is fully included in the memory range
        for table in tables {
            table.atomic_fill(HugeEntry::new_with(Bitfield::LEN));
        }
        // Table is only partially included in the memory range
        for (i, entry) in last.iter().enumerate() {
            let frame = tables.len() * TREE_FRAMES + i * Bitfield::LEN;
            let free = self.frames().saturating_sub(frame).min(Bitfield::LEN);
            entry.store(HugeEntry::new_with(free));
        }

        // Init bitfields
        let last_i = self.frames() / Bitfield::LEN;
        let (included, mut remainder) = self.bitfields.split_at(last_i);
        // Bitfield is fully included in the memory range
        for bitfield in included {
            bitfield.fill(false);
        }
        // Bitfield might be only partially included in the memory range
        if let Some((last, excluded)) = remainder.split_first() {
            let end = self.frames() - included.len() * Bitfield::LEN;
            debug_assert!(end <= Bitfield::LEN);
            last.set(0..end, false);
            last.set(end..Bitfield::LEN, true);
            remainder = excluded;
        }
        // Not part of the final memory range
        for bitfield in remainder {
            bitfield.fill(true);
        }
    }

    fn reserve_all(&self) {
        // Init table
        let (last, tables) = self.children.split_last().unwrap();
        // Table is fully included in the memory range
        for table in tables {
            table.atomic_fill(HugeEntry::new_huge());
        }
        // Table is only partially included in the memory range
        let last_i = (self.frames() / Bitfield::LEN) - tables.len() * TREE_HUGE;
        let (included, remainder) = last.split_at(last_i);
        for entry in included {
            entry.store(HugeEntry::new_huge());
        }
        // Remainder is allocated as small frames
        for entry in remainder {
            entry.store(HugeEntry::new_with(0));
        }

        // Init bitfields
        let last_i = self.frames() / Bitfield::LEN;
        let (included, remainder) = self.bitfields.split_at(last_i);
        // Bitfield is fully included in the memory range
        for bitfield in included {
            bitfield.fill(false);
        }
        // Bitfield might be only partially included in the memory range
        for bitfield in remainder {
            bitfield.fill(true);
        }
    }

    fn put_small(&self, frame: usize, order: usize) -> Result<bool> {
        debug_assert!(order < HUGE_ORDER);

        let bitfield = &self.bitfields[frame / Bitfield::LEN];
        let i = frame % Bitfield::LEN;
        if bitfield.toggle(i, order, true).is_err() {
            error!("L1 put failed o={order} i={i} p={frame}");
            return Err(Error::Address);
        }

        let children = &self.children[frame / TREE_FRAMES];
        let i = (frame / Bitfield::LEN) % TREE_HUGE;
        match children[i].fetch_update(|v| v.inc(1 << order)) {
            Err(entry) => panic!("Inc failed i{i} p={frame} {entry:?}"),
            Ok(entry) => Ok(entry.free() + (1 << order) == Bitfield::LEN),
        }
    }

    fn partial_put_huge(&self, old: HugeEntry, frame: usize, order: usize) -> Result<bool> {
        info!("partial free of huge frame {frame:x} o={order}");
        let i = (frame / Bitfield::LEN) % TREE_HUGE;
        let children = &self.children[frame / TREE_FRAMES];
        let bitfield = &self.bitfields[frame / Bitfield::LEN];

        // Try filling the whole bitfield
        if bitfield.toggle(0, Bitfield::ORDER, false).is_ok() {
            children[i]
                .compare_exchange(old, HugeEntry::new())
                .expect("Failed partial clear");
        }
        // Wait for parallel partial_put_huge to finish
        else if !spin_wait(RETRIES, || !children[i].load().huge()) {
            panic!("Exceeding retries");
        }

        self.put_small(frame, order)
    }

    #[cfg(feature = "std")]
    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        use std::fmt::Write;

        let mut out = std::string::String::new();
        writeln!(out, "Dumping pt {}", start / TREE_FRAMES).unwrap();
        let table = &self.children[start / TREE_FRAMES];
        for (i, entry) in table.iter().enumerate() {
            let start = align_down(start, TREE_FRAMES) + i * Bitfield::LEN;
            if start >= self.frames() {
                break;
            }

            let entry = entry.load();
            let indent = 4;
            let bitfield = &self.bitfields[start / Bitfield::LEN];
            writeln!(out, "{:indent$}l2 i={i}: {entry:?}\t{bitfield:?}", "").unwrap();
            if !entry.huge() {
                assert_eq!(bitfield.count_zeros(), entry.free());
            }
        }
        warn!("{out}");
    }
}

/// Manages huge frame, that can be allocated as base frames.
#[bitfield(u16)]
#[derive(PartialEq, Eq)]
struct HugeEntry {
    /// Number of free 4K frames or u16::MAX for a huge frame.
    count: u16,
}
impl Atomic for HugeEntry {
    type I = AtomicU16;
}
impl HugeEntry {
    /// Creates an entry marked as allocated huge frame.
    fn new_huge() -> Self {
        Self::new().with_count(u16::MAX)
    }
    /// Creates a new entry with the given free counter.
    fn new_with(free: usize) -> Self {
        Self::new().with_count(free as _)
    }
    /// Returns wether this entry is allocated as huge frame.
    fn huge(self) -> bool {
        self.count() == u16::MAX
    }
    /// Returns the free frames counter
    fn free(self) -> usize {
        if !self.huge() { self.count() as _ } else { 0 }
    }
    /// Try to allocate this entry as huge frame.
    fn mark_huge(self) -> Option<Self> {
        if self.free() == Bitfield::LEN {
            Some(Self::new_huge())
        } else {
            None
        }
    }
    /// Decrement the free frames counter.
    fn dec(self, num_frames: usize) -> Option<Self> {
        if !self.huge() && self.free() >= num_frames {
            Some(Self::new_with(self.free() - num_frames))
        } else {
            None
        }
    }
    /// Increments the free frames counter.
    fn inc(self, num_frames: usize) -> Option<Self> {
        if !self.huge() && self.free() <= Bitfield::LEN - num_frames {
            Some(Self::new_with(self.free() + num_frames))
        } else {
            None
        }
    }
}

/// Pair of huge entries that can be changed at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(4))]
struct HugePair(HugeEntry, HugeEntry);
impl Atomic for HugePair {
    type I = AtomicU32;
}

const _: () = assert!(size_of::<HugePair>() == 2 * size_of::<HugeEntry>());
const _: () = assert!(align_of::<HugePair>() == size_of::<HugePair>());

impl HugePair {
    /// Apply `f` to both entries.
    fn map(self, f: impl Fn(HugeEntry) -> Option<HugeEntry>) -> Option<HugePair> {
        Some(HugePair(f(self.0)?, f(self.1)?))
    }
    /// Check if `f` is true for both entries.
    fn all(self, f: impl Fn(HugeEntry) -> bool) -> bool {
        f(self.0) && f(self.1)
    }
}
impl From<u32> for HugePair {
    fn from(value: u32) -> Self {
        let [a, b, c, d] = value.to_ne_bytes();
        Self(
            HugeEntry(u16::from_ne_bytes([a, b])),
            HugeEntry(u16::from_ne_bytes([c, d])),
        )
    }
}
impl From<HugePair> for u32 {
    fn from(value: HugePair) -> Self {
        let ([a, b], [c, d]) = (value.0.0.to_ne_bytes(), value.1.0.to_ne_bytes());
        u32::from_ne_bytes([a, b, c, d])
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use core::mem::ManuallyDrop;
    use core::ops::Deref;
    use std::sync::Barrier;
    use std::vec::Vec;

    use log::warn;

    use super::Bitfield;
    use crate::lower::Lower;
    use crate::util::{WyRand, aligned_buf, logging};
    use crate::{Error, HUGE_FRAMES, Init, MAX_ORDER, Result, TREE_FRAMES, TREE_HUGE, thread};

    struct LowerTest<'a>(ManuallyDrop<Lower<'a>>);

    impl LowerTest<'_> {
        fn create(frames: usize, init: Init) -> Result<Self> {
            let primary = aligned_buf(Lower::metadata_size(frames));
            Ok(Self(ManuallyDrop::new(Lower::new(frames, init, primary)?)))
        }
    }
    impl<'a> Deref for LowerTest<'a> {
        type Target = Lower<'a>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl Drop for LowerTest<'_> {
        fn drop(&mut self) {
            let meta = self.0.metadata();
            unsafe {
                drop(ManuallyDrop::take(&mut self.0));
                Vec::from_raw_parts(meta.as_mut_ptr(), meta.len(), meta.len());
            }
        }
    }

    #[test]
    fn alloc_normal() {
        logging();

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();
        lower.get(0, 0).unwrap();

        thread::parallel(0..2, |t| {
            thread::pin(t);

            let frame = lower.get(0, 0).unwrap().0;
            assert!(frame < lower.frames());
        });

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN - 3);
        assert_eq!(lower.bitfields[0].count_zeros(), Bitfield::LEN - 3);
    }

    #[test]
    fn alloc_first() {
        logging();

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.get(0, 0).unwrap();
        });

        let entry2 = lower.children[0][0].load();
        assert_eq!(entry2.free(), Bitfield::LEN - 2);
        assert_eq!(lower.bitfields[0].count_zeros(), Bitfield::LEN - 2);
    }

    #[test]
    fn alloc_last() {
        logging();

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();

        for _ in 0..Bitfield::LEN - 1 {
            lower.get(0, 0).unwrap();
        }

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.get(0, 0).unwrap();
        });

        let table = &lower.children[0];
        assert_eq!(table[0].load().free(), 0);
        assert_eq!(table[1].load().free(), Bitfield::LEN - 1);
        assert_eq!(lower.bitfields[1].count_zeros(), Bitfield::LEN - 1);
    }

    #[test]
    fn free_normal() {
        logging();

        let mut frames = [0; 2];

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();

        frames[0] = lower.get(0, 0).unwrap().0;
        frames[1] = lower.get(0, 0).unwrap().0;

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.put(frames[t], 0).unwrap();
        });

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN);
    }

    #[test]
    fn free_last() {
        logging();

        let mut frames = [0; Bitfield::LEN];

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();

        for frame in &mut frames {
            *frame = lower.get(0, 0).unwrap().0;
        }

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.put(frames[t], 0).unwrap();
        });

        let table = &lower.children[0];
        assert_eq!(table[0].load().free(), 2);
        assert_eq!(lower.bitfields[0].count_zeros(), 2);
    }

    #[test]
    fn realloc_last() {
        logging();

        let mut frames = [0; Bitfield::LEN];

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();

        for frame in &mut frames[..Bitfield::LEN - 1] {
            *frame = lower.get(0, 0).unwrap().0;
        }

        std::thread::scope(|s| {
            s.spawn(|| {
                thread::pin(0);

                lower.get(0, 0).unwrap();
            });
            thread::pin(1);

            lower.put(frames[0], 0).unwrap();
        });

        let table = &lower.children[0];
        if table[0].load().free() == 1 {
            assert_eq!(lower.bitfields[0].count_zeros(), 1);
        } else {
            // Table entry skipped
            assert_eq!(table[0].load().free(), 2);
            assert_eq!(lower.bitfields[0].count_zeros(), 2);
            assert_eq!(table[1].load().free(), Bitfield::LEN - 1);
            assert_eq!(lower.bitfields[1].count_zeros(), Bitfield::LEN - 1);
        }
    }

    #[test]
    fn alloc_normal_large() {
        logging();

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();
        lower.get(0, 0).unwrap();

        thread::parallel(0..2, |t| {
            thread::pin(t);

            let order = t + 1; // order 1 and 2
            let frame = lower.get(0, order).unwrap().0;
            assert!(frame < lower.frames());
        });

        let allocated = 1 + 2 + 4;
        assert_eq!(
            lower.children[0][0].load().free(),
            Bitfield::LEN - allocated
        );
        assert_eq!(lower.bitfields[0].count_zeros(), Bitfield::LEN - allocated);
    }

    #[test]
    fn free_normal_large() {
        logging();

        let mut frames = [0; 2];

        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();

        frames[0] = lower.get(0, 1).unwrap().0;
        frames[1] = lower.get(0, 2).unwrap().0;

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN - 2 - 4);

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.put(frames[t], t + 1).unwrap();
        });

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN);
    }

    #[test]
    fn different_orders() {
        logging();

        const FRAMES: usize = (MAX_ORDER + 2) << MAX_ORDER;

        let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();

        assert_eq!(lower.stats().free_frames, lower.frames());
        assert_eq!(lower.stats().free_frames, FRAMES);

        let mut rng = WyRand::new(42);

        let mut num_frames = 0;
        let mut frames = Vec::new();
        for order in 0..=MAX_ORDER {
            for _ in 0..1usize << (MAX_ORDER - order) {
                frames.push((order, 0));
                num_frames += 1 << order;
            }
        }
        rng.shuffle(&mut frames);
        assert!(lower.frames() >= num_frames);
        warn!(
            "allocate {num_frames}/{} frames up to order {MAX_ORDER}",
            lower.frames()
        );

        let mut tree_idx = 0;
        'outer: for (order, frame) in &mut frames {
            for i in 0..lower.frames().div_ceil(TREE_FRAMES) {
                // fall back to other chunks
                let i = (i + tree_idx) % lower.frames().div_ceil(TREE_FRAMES);
                match lower.get(i * TREE_FRAMES, *order) {
                    Ok((free, _huge)) => {
                        *frame = free;
                        tree_idx = free / TREE_FRAMES;
                        continue 'outer;
                    }
                    Err(Error::Memory) => {}
                    Err(e) => panic!("{e:?}"),
                }
            }
            panic!("Fragmented!");
        }

        assert_eq!(lower.frames() - lower.stats().free_frames, num_frames);

        for (order, frame) in &frames {
            lower.put(*frame, *order).unwrap();
        }

        assert_eq!(lower.stats().free_frames, lower.frames());
    }

    #[test]
    fn init_reserved() {
        logging();

        const FRAMES: usize = (TREE_FRAMES - 1) / (1 << MAX_ORDER);

        let lower = LowerTest::create(TREE_FRAMES - 1, Init::AllocAll).unwrap();

        assert_eq!(lower.stats().free_frames, 0);

        for i in 0..FRAMES {
            lower.put(i * (1 << MAX_ORDER), MAX_ORDER).unwrap();
        }

        assert_eq!(
            lower.frames() - lower.stats().free_frames,
            (1 << MAX_ORDER) - 1
        );
    }

    #[test]
    fn partial_put_huge() {
        logging();

        let lower = LowerTest::create(TREE_FRAMES - 1, Init::AllocAll).unwrap();

        assert_eq!(lower.stats().free_frames, 0);

        lower.put(0, 0).unwrap();

        assert_eq!(lower.stats().free_frames, 1);
    }

    #[test]
    #[ignore]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 6;
        const FRAMES: usize = 2 * THREADS * TREE_FRAMES;

        for _ in 0..8 {
            let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();
            assert_eq!(lower.stats().free_frames, FRAMES);

            let barrier = Barrier::new(THREADS);
            thread::parallel(0..THREADS, |t| {
                thread::pin(t);
                barrier.wait();

                let mut frames = [0; 4];
                for p in &mut frames {
                    *p = lower.get(0, 0).unwrap().0;
                }
                frames.reverse();
                for p in frames {
                    lower.put(p, 0).unwrap();
                }
            });

            assert_eq!(lower.stats().free_frames, FRAMES);
        }
    }

    #[test]
    #[ignore]
    fn rand_realloc_last() {
        logging();

        const THREADS: usize = 6;
        const FRAMES: usize = 2 * THREADS * TREE_FRAMES;
        let mut frames = [0; HUGE_FRAMES];

        for _ in 0..8 {
            let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();
            assert_eq!(lower.stats().free_frames, FRAMES);

            for frame in &mut frames[..HUGE_FRAMES - 3] {
                *frame = lower.get(0, 0).unwrap().0;
            }

            let barrier = Barrier::new(THREADS);
            thread::parallel(0..THREADS, |t| {
                thread::pin(t);
                barrier.wait();

                if t < THREADS / 2 {
                    lower.put(frames[t], 0).unwrap();
                } else {
                    lower.get(0, 0).unwrap();
                }
            });

            assert_eq!(lower.frames() - lower.stats().free_frames, HUGE_FRAMES - 3);
        }
    }

    #[test]
    fn alloc_track_huge() {
        logging();

        const THREADS: usize = 4;
        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();
        let barrier = Barrier::new(THREADS);

        let huge = thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut frames = Vec::with_capacity(TREE_FRAMES);

            barrier.wait();

            let mut rng = WyRand::new(t as u64);

            let mut get = 0;
            let mut put = 0;
            loop {
                match lower.get(0, 0) {
                    Ok((frame, huge)) => {
                        get += huge as usize;
                        frames.push(frame);
                    }
                    Err(Error::Memory) => break,
                    Err(e) => panic!("{e:?}"),
                }
            }

            barrier.wait();

            rng.shuffle(&mut frames);
            while let Some(frame) = frames.pop() {
                put += lower.put(frame, 0).unwrap() as usize;
            }

            (get, put)
        });
        let (get, put) = huge
            .iter()
            .fold((0, 0), |acc, x| (acc.0 + x.0, acc.1 + x.1));

        assert_eq!(get, put);
        assert_eq!(get, TREE_HUGE);
        assert_eq!(lower.stats().free_frames, TREE_FRAMES);
        assert_eq!(lower.stats().free_huge, TREE_HUGE);
    }

    #[test]
    fn alloc_stress_huge() {
        logging();

        let rand = unsafe { libc::rand() as u64 };

        const ITER: usize = 50;
        const THREADS: usize = 4;
        let lower = LowerTest::create(TREE_FRAMES, Init::FreeAll).unwrap();
        let barrier = Barrier::new(THREADS);

        let huge = thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut frames = Vec::with_capacity(TREE_FRAMES);

            barrier.wait();

            let mut rng = WyRand::new(rand + t as u64);

            let mut get = 0;
            let mut put = 0;
            for _ in 0..ITER {
                let target = rng.range(0..(2 * TREE_FRAMES / THREADS) as _) as usize;

                while frames.len() != target {
                    if target < frames.len() {
                        put += lower.put(frames.pop().unwrap(), 0).unwrap() as usize;
                    } else {
                        match lower.get(0, 0) {
                            Ok((frame, huge)) => {
                                get += huge as usize;
                                frames.push(frame);
                            }
                            Err(Error::Memory) => break,
                            Err(e) => panic!("{e:?}"),
                        }
                    }
                }
                rng.shuffle(&mut frames);
            }
            for frame in frames {
                put += lower.put(frame, 0).unwrap() as usize;
            }

            (get, put)
        });
        let (get, put) = huge
            .iter()
            .fold((0, 0), |acc, x| (acc.0 + x.0, acc.1 + x.1));

        assert_eq!(get, put);
        assert_eq!(lower.stats().free_frames, TREE_FRAMES);
        assert_eq!(lower.stats().free_huge, TREE_HUGE);
    }
}
