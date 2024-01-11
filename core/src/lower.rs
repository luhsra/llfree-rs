//! Lower allocator implementations

use core::mem::{align_of, size_of};
use core::slice;

use log::{error, info, warn};

use crate::atomic::Atom;
use crate::entry::{AtomicArray, HugeEntry, HugePair};
use crate::util::{align_down, align_up, spin_wait, Align};
use crate::{Error, Init, Result, CAS_RETRIES};

type Bitfield = crate::bitfield::Bitfield<8>;

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
    children: &'a [Align<[Atom<HugeEntry>; Lower::HP]>],
}

unsafe impl Send for Lower<'_> {}
unsafe impl Sync for Lower<'_> {}

const _: () = assert!(Lower::HP < (1 << (u16::BITS as usize - Lower::HUGE_ORDER)));

impl<'a> Lower<'a> {
    /// Number of huge pages managed by a chunk
    pub const HP: usize = 32;
    /// Pages per chunk. Every alloc only searches in a chunk of this size.
    pub const N: usize = Self::HP * Bitfield::LEN;
    /// The maximal allowed order of this allocator
    pub const HUGE_ORDER: usize = Bitfield::ORDER;
    pub const MAX_ORDER: usize = Self::HUGE_ORDER + 1;

    pub fn metadata_size(frames: usize) -> usize {
        let bitfields = frames.div_ceil(Bitfield::LEN);
        let bitfields_size = bitfields * size_of::<Bitfield>();
        let bitfields_size = align_up(bitfields_size, size_of::<[HugeEntry; Lower::HP]>());

        let tables = frames.div_ceil(Self::N);
        let tables_size = tables * align_up(size_of::<[HugeEntry; Lower::HP]>(), align_of::<Align>());
        bitfields_size + tables_size
    }

    /// Create a new lower allocator.
    pub fn new(frames: usize, init: Init, primary: &'a mut [u8]) -> Result<Self> {
        let bitfields_num = frames.div_ceil(Bitfield::LEN);
        let bitfields_size = bitfields_num * size_of::<Bitfield>();
        let bitfields_size = align_up(bitfields_size, size_of::<[HugeEntry; Lower::HP]>());

        let tables_num = frames.div_ceil(Self::N);
        let tables_size = tables_num * size_of::<[HugeEntry; Lower::HP]>();

        if primary.len() < bitfields_size + tables_size
            || primary.as_ptr() as usize % align_of::<Align>() != 0
        {
            error!("primary metadata");
            return Err(Error::Initialization);
        }
        let (bitfields, children) = primary.split_at_mut(bitfields_size);

        // Start of the l1 table array
        let bitfields =
            unsafe { slice::from_raw_parts_mut(bitfields.as_mut_ptr().cast(), bitfields_num) };

        // Start of the l2 table array
        let children =
            unsafe { slice::from_raw_parts_mut(children.as_mut_ptr().cast(), tables_num) };

        let alloc = Self {
            len: frames,
            bitfields,
            children,
        };

        match init {
            Init::FreeAll => alloc.free_all(),
            Init::AllocAll => alloc.reserve_all(),
            Init::Recover(false) => {} // skip, assuming everything is valid
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
                let start = i * Self::N + j * Bitfield::LEN;
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
                        a_entry.store(HugeEntry::new_free(zeros));
                    }
                }
            }
        }
    }

    /// Return the number of free frames in the tree at `start`.
    pub fn free_in_tree(&self, start: usize) -> usize {
        assert!(start < self.frames());
        let mut free = 0;
        for entry in self.children[start / Self::N].iter() {
            free += entry.load().free();
        }
        free
    }

    /// Try allocating a new `frame` in the [LowerAlloc::N] sized chunk at `start`.
    pub fn get(&self, start: usize, order: usize) -> Result<usize> {
        debug_assert!(order <= Self::MAX_ORDER);
        debug_assert!(start < self.frames());

        match order {
            Self::MAX_ORDER => self.get_max(start),
            Self::HUGE_ORDER => self.get_huge(start),
            _ => self.get_small(start, order),
        }
    }

    /// Free single frame
    pub fn put(&self, frame: usize, order: usize) -> Result<()> {
        debug_assert!(order <= Self::MAX_ORDER);
        debug_assert!(frame < self.frames());

        if order == Self::MAX_ORDER {
            self.put_max(frame)
        } else if order == Self::HUGE_ORDER {
            let i = (frame / Bitfield::LEN) % Self::HP;
            let table = &self.children[frame / Self::N];

            if let Err(old) =
                table[i].compare_exchange(HugeEntry::new_huge(), HugeEntry::new_free(Bitfield::LEN))
            {
                error!("Addr p={frame:x} o={order} {old:?}");
                Err(Error::Address)
            } else {
                Ok(())
            }
        } else {
            let i = (frame / Bitfield::LEN) % Self::HP;
            let table = &self.children[frame / Self::N];

            let old = table[i].load();
            if old.huge() {
                self.partial_put_huge(old, frame, order)
            } else if old.free() <= Bitfield::LEN - (1 << order) {
                self.put_small(frame, order)
            } else {
                error!("Addr p={frame:x} o={order} {old:?}");
                Err(Error::Address)
            }
        }
    }

    /// Returns if the frame is free. This might be racy!
    pub fn is_free(&self, frame: usize, order: usize) -> bool {
        debug_assert!(frame % (1 << order) == 0);
        if order > Self::MAX_ORDER || frame + (1 << order) > self.frames() {
            return false;
        }

        if order > Bitfield::ORDER {
            // multiple huge frames
            let i = (frame / Bitfield::LEN) % Self::HP;
            self.table_pair(frame)[i / 2]
                .load()
                .all(|e| e.free() == Bitfield::LEN)
        } else {
            let table = &self.children[frame / Self::N];
            let i = (frame / Bitfield::LEN) % Self::HP;
            let entry = table[i].load();

            if entry.free() < (1 << order) {
                false
            } else if entry.free() == Bitfield::LEN {
                true
            } else {
                let bitfield = &self.bitfields[frame / Bitfield::LEN];
                bitfield.is_zero(frame % Bitfield::LEN, order)
            }
        }
    }

    /// Debug function, returning the number of allocated frames and performing internal checks.
    #[allow(unused)]
    pub fn allocated_frames(&self) -> usize {
        let mut frames = self.frames();
        for table in &*self.children {
            for entry in table.iter() {
                frames -= entry.load().free();
            }
        }
        frames
    }

    /// Debug function returning number of free frames in each order 9 chunk
    pub fn for_each_huge_frame<F: FnMut(usize, usize)>(&self, mut f: F) {
        for (ti, table) in self.children.iter().enumerate() {
            for (ci, child) in table.iter().enumerate() {
                f(ti * Self::HP + ci, child.load().free())
            }
        }
    }

    /// Returns the table with pair entries that can be updated at once.
    fn table_pair(&self, frame: usize) -> &[Atom<HugePair>; Lower::HP / 2] {
        let table = &self.children[frame / Self::N];
        unsafe { &*table.as_ptr().cast() }
    }

    fn free_all(&self) {
        // Init tables
        let (last, tables) = self.children.split_last().unwrap();
        // Table is fully included in the memory range
        for table in tables {
            table.atomic_fill(HugeEntry::new_free(Bitfield::LEN));
        }
        // Table is only partially included in the memory range
        for (i, entry) in last.iter().enumerate() {
            let frame = tables.len() * Self::N + i * Bitfield::LEN;
            let free = self.frames().saturating_sub(frame).min(Bitfield::LEN);
            entry.store(HugeEntry::new_free(free));
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
        let last_i = (self.frames() / Bitfield::LEN) - tables.len() * Self::HP;
        let (included, remainder) = last.split_at(last_i);
        for entry in included {
            entry.store(HugeEntry::new_huge());
        }
        // Remainder is allocated as small frames
        for entry in remainder {
            entry.store(HugeEntry::new_free(0));
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

    /// Allocate frames up to order 8
    fn get_small(&self, start: usize, order: usize) -> Result<usize> {
        debug_assert!(order < Bitfield::ORDER);

        let first_bf_i = align_down(start / Bitfield::LEN, Self::HP);
        let start_bf_e = (start / Bitfield::ENTRY_BITS) % Bitfield::ENTRIES;
        let table = &self.children[start / Self::N];
        let offset = (start / Bitfield::LEN) % Self::HP;

        for j in 0..Self::HP {
            let i = (j + offset) % Self::HP;

            if table[i].fetch_update(|v| v.dec(1 << order)).is_ok() {
                let bf_i = first_bf_i + i;
                // start with the previous bitfield entry
                let bf_e = if j == 0 { start_bf_e } else { 0 };

                if let Ok(offset) = self.bitfields[bf_i].set_first_zeros(bf_e, order) {
                    return Ok(bf_i * Bitfield::LEN + offset);
                }

                // Revert conter
                table[i]
                    .fetch_update(|v| v.inc(Bitfield::LEN, 1 << order))
                    .expect("undo failed");
            }
        }

        info!("Nothing found o={order}");
        Err(Error::Memory)
    }

    /// Allocate huge frame
    fn get_huge(&self, start: usize) -> Result<usize> {
        let table = &self.children[start / Self::N];
        let offset = (start / Bitfield::LEN) % Self::HP;

        for i in 0..Self::HP {
            let i = (offset + i) % Self::HP;
            if let Ok(_) = table[i].fetch_update(|v| v.mark_huge(Bitfield::LEN)) {
                return Ok(align_down(start, Self::N) + i * Bitfield::LEN);
            }
        }

        info!("Nothing found o=9");
        Err(Error::Memory)
    }

    /// Allocate multiple huge frames
    fn get_max(&self, start: usize) -> Result<usize> {
        let table_pair = self.table_pair(start);
        let offset = ((start / Bitfield::LEN) % Self::HP) / 2;

        for i in 0..Self::HP / 2 {
            let i = (offset + i) % (Self::HP / 2);
            if let Ok(_) = table_pair[i].fetch_update(|v| v.map(|v| v.mark_huge(Bitfield::LEN))) {
                return Ok(align_down(start, Self::N) + 2 * i * Bitfield::LEN);
            }
        }

        info!("Nothing found o=10");
        Err(Error::Memory)
    }

    fn put_small(&self, frame: usize, order: usize) -> Result<()> {
        debug_assert!(order < Self::HUGE_ORDER);

        let bitfield = &self.bitfields[frame / Bitfield::LEN];
        let i = frame % Bitfield::LEN;
        if bitfield.toggle(i, order, true).is_err() {
            error!("L1 put failed i{i} p={frame}");
            return Err(Error::Address);
        }

        let table = &self.children[frame / Self::N];
        let i = (frame / Bitfield::LEN) % Self::HP;
        if let Err(entry) = table[i].fetch_update(|v| v.inc(Bitfield::LEN, 1 << order)) {
            panic!("Inc failed i{i} p={frame} {entry:?}");
        }

        Ok(())
    }

    pub fn put_max(&self, frame: usize) -> Result<()> {
        let table_pair = self.table_pair(frame);
        let i = ((frame / Bitfield::LEN) % Self::HP) / 2;

        if let Err(old) = table_pair[i].compare_exchange(
            HugePair(HugeEntry::new_huge(), HugeEntry::new_huge()),
            HugePair(
                HugeEntry::new_free(Bitfield::LEN),
                HugeEntry::new_free(Bitfield::LEN),
            ),
        ) {
            error!("Addr {frame} o={} {old:?} i={i}", Self::MAX_ORDER);
            Err(Error::Address)
        } else {
            Ok(())
        }
    }

    fn partial_put_huge(&self, old: HugeEntry, frame: usize, order: usize) -> Result<()> {
        info!("partial free of huge frame {frame:x} o={order}");
        let i = (frame / Bitfield::LEN) % Self::HP;
        let table = &self.children[frame / Self::N];
        let bitfield = &self.bitfields[frame / Bitfield::LEN];

        // Try filling the whole bitfield
        if bitfield.toggle(0, Bitfield::ORDER, false).is_ok() {
            table[i]
                .compare_exchange(old, HugeEntry::new())
                .expect("Failed partial clear");
        }
        // Wait for parallel partial_put_huge to finish
        else if !spin_wait(CAS_RETRIES, || !table[i].load().huge()) {
            panic!("Exceeding retries");
        }

        self.put_small(frame, order)
    }

    #[cfg(feature = "std")]
    #[allow(dead_code)]
    pub fn dump(&self, start: usize) {
        use std::fmt::Write;

        let mut out = std::string::String::new();
        writeln!(out, "Dumping pt {}", start / Self::N).unwrap();
        let table = &self.children[start / Self::N];
        for (i, entry) in table.iter().enumerate() {
            let start = align_down(start, Self::N) + i * Bitfield::LEN;
            if start > self.frames() {
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

#[cfg(all(test, feature = "std"))]
mod test {
    use core::mem::ManuallyDrop;
    use core::ops::Deref;
    use std::sync::Barrier;
    use std::vec::Vec;

    use crate::Result;
    use log::warn;

    use super::Bitfield;
    use crate::frame::PT_LEN;
    use crate::lower::Lower;
    use crate::thread;
    use crate::util::{aligned_buf, logging, WyRand};
    use crate::Init;

    struct LowerTest(ManuallyDrop<Lower<'static>>);

    impl LowerTest {
        fn create(frames: usize, init: Init) -> Result<Self> {
            let primary = aligned_buf(Lower::metadata_size(frames)).leak();
            Ok(Self(ManuallyDrop::new(Lower::new(frames, init, primary)?)))
        }
    }
    impl Deref for LowerTest {
        type Target = Lower<'static>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl Drop for LowerTest {
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

        const FRAMES: usize = 4 * Lower::N;
        let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();
        lower.get(0, 0).unwrap();

        thread::parallel(0..2, |t| {
            thread::pin(t);

            let frame = lower.get(0, 0).unwrap();
            assert!(frame < FRAMES);
        });

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN - 3);
        assert_eq!(lower.bitfields[0].count_zeros(), Bitfield::LEN - 3);
    }

    #[test]
    fn alloc_first() {
        logging();

        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

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

        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

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

        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

        frames[0] = lower.get(0, 0).unwrap();
        frames[1] = lower.get(0, 0).unwrap();

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.put(frames[t as usize], 0).unwrap();
        });

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN);
    }

    #[test]
    fn free_last() {
        logging();

        let mut frames = [0; Bitfield::LEN];

        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

        for frame in &mut frames {
            *frame = lower.get(0, 0).unwrap();
        }

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.put(frames[t as usize], 0).unwrap();
        });

        let table = &lower.children[0];
        assert_eq!(table[0].load().free(), 2);
        assert_eq!(lower.bitfields[0].count_zeros(), 2);
    }

    #[test]
    fn realloc_last() {
        logging();

        let mut frames = [0; Bitfield::LEN];

        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

        for frame in &mut frames[..Bitfield::LEN - 1] {
            *frame = lower.get(0, 0).unwrap();
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

        const FRAMES: usize = 4 * Lower::N;
        let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();
        lower.get(0, 0).unwrap();

        thread::parallel(0..2, |t| {
            thread::pin(t);

            let order = t + 1; // order 1 and 2
            let frame = lower.get(0, order).unwrap();
            assert!(frame < FRAMES);
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

        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

        frames[0] = lower.get(0, 1).unwrap();
        frames[1] = lower.get(0, 2).unwrap();

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN - 2 - 4);

        thread::parallel(0..2, |t| {
            thread::pin(t);

            lower.put(frames[t as usize], t + 1).unwrap();
        });

        assert_eq!(lower.children[0][0].load().free(), Bitfield::LEN);
    }

    #[test]
    fn different_orders() {
        logging();

        const MAX_ORDER: usize = Lower::MAX_ORDER;

        thread::pin(0);
        let lower = LowerTest::create(4 * Lower::N, Init::FreeAll).unwrap();

        assert_eq!(lower.allocated_frames(), 0);

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
        warn!("allocate {num_frames} frames up to order {MAX_ORDER}");

        for (order, frame) in &mut frames {
            *frame = lower.get(0, *order).unwrap();
        }

        assert_eq!(lower.allocated_frames(), num_frames);

        for (order, frame) in &frames {
            lower.put(*frame, *order).unwrap();
        }

        assert_eq!(lower.allocated_frames(), 0);
    }

    #[test]
    fn init_reserved() {
        logging();

        const MAX_ORDER: usize = Lower::MAX_ORDER;

        let num_max_frames = (Lower::N - 1) / (1 << MAX_ORDER);

        let lower = LowerTest::create(Lower::N - 1, Init::AllocAll).unwrap();

        assert_eq!(lower.allocated_frames(), Lower::N - 1);

        for i in 0..num_max_frames {
            lower.put(i * (1 << MAX_ORDER), MAX_ORDER).unwrap();
        }

        assert_eq!(lower.allocated_frames(), (1 << MAX_ORDER) - 1);
    }

    #[test]
    fn partial_put_huge() {
        logging();

        let lower = LowerTest::create(Lower::N - 1, Init::AllocAll).unwrap();

        assert_eq!(lower.allocated_frames(), Lower::N - 1);

        lower.put(0, 0).unwrap();

        assert_eq!(lower.allocated_frames(), Lower::N - 2);
    }

    #[test]
    #[ignore]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 6;
        const FRAMES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        for _ in 0..8 {
            let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();
            assert_eq!(lower.allocated_frames(), 0);

            let barrier = Barrier::new(THREADS);
            thread::parallel(0..THREADS, |t| {
                thread::pin(t);
                barrier.wait();

                let mut frames = [0; 4];
                for p in &mut frames {
                    *p = lower.get(0, 0).unwrap();
                }
                frames.reverse();
                for p in frames {
                    lower.put(p, 0).unwrap();
                }
            });

            assert_eq!(lower.allocated_frames(), 0);
        }
    }

    #[test]
    #[ignore]
    fn rand_realloc_last() {
        logging();

        const THREADS: usize = 6;
        const FRAMES: usize = 2 * THREADS * PT_LEN * PT_LEN;
        let mut frames = [0; PT_LEN];

        for _ in 0..8 {
            let lower = LowerTest::create(FRAMES, Init::FreeAll).unwrap();
            assert_eq!(lower.allocated_frames(), 0);

            for frame in &mut frames[..PT_LEN - 3] {
                *frame = lower.get(0, 0).unwrap();
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

            assert_eq!(lower.allocated_frames(), PT_LEN - 3);
        }
    }
}
