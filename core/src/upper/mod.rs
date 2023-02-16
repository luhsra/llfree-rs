use core::any::type_name;
use core::fmt;
use core::ops::Range;
use core::sync::atomic::AtomicU64;

use bitfield_struct::bitfield;

use crate::atomic::{Atom, Atomic};
use crate::entry::ReservedTree;
use crate::{Result, PFN};

mod array;
pub use array::Array;
mod list_local;
pub use list_local::ListLocal;
mod list_locked;
pub use list_locked::ListLocked;
mod list_cas;
pub use list_cas::ListCAS;

/// Number of retries if an atomic operation fails.
pub const CAS_RETRIES: usize = 16;
/// Magic marking the meta frame.
pub const MAGIC: usize = 0x_dead_beef;

/// Minimal number of frames an allocator needs (1G).
pub const MIN_PAGES: usize = 1 << 9;
/// Maximal number of frames an allocator can manage (about 256TiB).
pub const MAX_PAGES: usize = 1 << (4 * 9);

/// The general interface of the allocator implementations.
pub trait Alloc: Sync + Send + fmt::Debug {
    /// Initialize the allocator.
    #[cold]
    fn init(&mut self, cores: usize, area: Range<PFN>, init: Init, free_all: bool) -> Result<()>;

    /// Allocate a new frame of `order` on the given `core`.
    fn get(&self, core: usize, order: usize) -> Result<PFN>;
    /// Free the `frame` of `order` on the given `core`..
    fn put(&self, core: usize, frame: PFN, order: usize) -> Result<()>;
    /// Returns if `frame` is free. This might be racy!
    fn is_free(&self, frame: PFN, order: usize) -> bool;

    /// Return the total number of frames the allocator manages.
    fn frames(&self) -> usize;

    /// Unreserve cpu-local frames
    fn drain(&self, _core: usize) -> Result<()> {
        Ok(())
    }

    /// Return the number of allocated frames.
    fn allocated_frames(&self) -> usize {
        self.frames() - self.free_frames()
    }
    /// Return the number of free frames.
    fn free_frames(&self) -> usize;
    /// Return the number of free huge frames or 0 if the allocator cannot allocate huge frames.
    fn free_huge_frames(&self) -> usize {
        0
    }
    /// Execute f for each huge frame with the number of free frames
    /// in this huge frame as parameter.
    #[cold]
    fn for_each_huge_frame(&self, _f: fn(PFN, usize)) {}
    /// Return the name of the allocator.
    #[cold]
    fn name(&self) -> AllocName {
        AllocName::new::<Self>()
    }
}

/// Defines if the allocator should be allocated persistently
/// and if it in that case should try to recover from the persistent memory.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Init {
    /// Not persistent
    Volatile,
    /// Persistent and try recovery
    Recover,
    /// Overwrite the persistent memory
    Overwrite,
}

/// Wrapper for creating a new allocator instance
pub trait AllocExt: Sized + Alloc + Default {
    /// Create and initialize the allocator.
    #[cold]
    fn new(cores: usize, area: Range<PFN>, init: Init, free_all: bool) -> Result<Self> {
        let mut a = Self::default();
        a.init(cores, area, init, free_all)?;
        Ok(a)
    }
}
// Implement for all default initializable allocators
impl<A: Sized + Alloc + Default> AllocExt for A {}

/// The short name of an allocator.
///
/// E.g.: `Array4C32` for
/// `nvalloc::upper::array::Array<4, nvalloc::lower::cache::Cache<32>>`
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct AllocName([&'static str; 4]);

impl fmt::Display for AllocName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for part in self.0 {
            write!(f, "{part}")?;
        }
        Ok(())
    }
}

impl AllocName {
    pub fn new<A: Alloc + ?Sized>() -> Self {
        let name = type_name::<A>();
        // Add first letter of generic type as suffix
        let (name, first, second, size) = if let Some((prefix, suffix)) = name.split_once('<') {
            // Aligned / Unaligned allocator
            let (first, suffix) = if let Some((first, suffix)) = suffix.split_once(", ") {
                // Strip path
                let first = first.rsplit_once(':').map_or(first, |s| s.1);
                (&first[0..1], suffix)
            } else {
                ("", suffix)
            };

            // Strip path
            let suffix = suffix.rsplit_once(':').map_or(suffix, |s| s.1);
            // Lower allocator size
            let size = suffix
                .split_once('<')
                .map(|(_, s)| s.split_once('>').map_or(s, |s| s.0))
                .unwrap_or_default();
            (prefix, first, &suffix[0..1], size)
        } else {
            (name, "", "", "")
        };

        // Strip namespaces
        let base = name.rsplit_once(':').map_or(name, |s| s.1);
        let base = base.strip_suffix("Alloc").unwrap_or(base);
        Self([base, first, second, size])
    }
}

impl PartialEq<&str> for AllocName {
    fn eq(&self, other: &&str) -> bool {
        let mut remainder = *other;
        for part in self.0 {
            if let Some(r) = remainder.strip_prefix(part) {
                remainder = r;
            } else {
                return false;
            };
        }
        remainder.is_empty()
    }
}

impl PartialEq<AllocName> for &str {
    fn eq(&self, other: &AllocName) -> bool {
        other.eq(self)
    }
}

#[bitfield(u64)]
#[derive(Default, PartialEq, Eq)]
pub struct LastFrees {
    #[bits(48)]
    tree_index: usize,
    #[bits(16)]
    count: usize,
}
impl Atomic for LastFrees {
    type I = AtomicU64;
}

#[derive(Default)]
/// Per core data.
pub struct Local<const F: usize> {
    /// Local copy of the reserved level 3 entry
    pub preferred: Atom<ReservedTree>,
    /// Last frees
    pub last_frees: Atom<LastFrees>,
}

impl<const F: usize> Local<F> {
    /// Add a tree index to the history.
    pub fn frees_push(&self, tree_index: usize) {
        // If the update of this heuristic fails, ignore it
        // Relaxed ordering is enough, as this is not shared between CPUs
        let _ = self.last_frees.fetch_update(|v| {
            let v = LastFrees::from(v);
            if v.tree_index() == tree_index {
                (v.count() < F).then_some(v.with_count(v.count() + 1))
            } else {
                Some(
                    LastFrees::new()
                        .with_tree_index(tree_index)
                        .with_count(1)
                        .into(),
                )
            }
        });
    }
    /// Checks if the previous `count` frees had the same tree index.
    pub fn frees_in_tree(&self, tree_index: usize) -> bool {
        let lf = LastFrees::from(self.last_frees.load());
        lf.tree_index() == tree_index && lf.count() >= F
    }
}

impl<const F: usize> fmt::Debug for Local<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("reserved", &self.preferred.load())
            .field("frees", &LastFrees::from(self.last_frees.load()))
            .finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod test {

    use core::any::type_name;
    use core::mem::size_of;
    use core::ptr::null_mut;
    use std::time::Instant;

    use alloc::vec::Vec;
    use std::sync::Barrier;

    use log::{info, warn};

    use crate::lower::*;
    use crate::mmap::test_mapping;
    use crate::pfn_range;
    use crate::table::PT_LEN;
    use crate::thread;
    use crate::upper::*;
    use crate::util::{logging, CacheLine, WyRand};
    use crate::PFNRange;
    use crate::{Error, Frame, PFN};

    type Lower = Cache<32>;
    type Allocator = Array<4, Lower>;

    #[test]
    fn names() {
        println!(
            "{}\n -> {}",
            type_name::<Array<4, Lower>>(),
            AllocName::new::<Array<4, Lower>>()
        );
        assert_eq!("Array4C32", AllocName::new::<Array<4, Lower>>());
        println!(
            "{}\n -> {}",
            type_name::<ListLocal>(),
            AllocName::new::<ListLocal>()
        );
        assert_eq!("ListLocal", AllocName::new::<ListLocal>());
    }

    #[test]
    fn sizes() {
        type C32 = Cache<32>;
        type A4C32 = Array<4, C32>;
        println!("{}:", AllocName::new::<A4C32>());
        println!("  Static size {}B", size_of::<A4C32>());
        println!("  Size per CPU: {}B", size_of::<CacheLine<Local<4>>>());
        println!("  Size per GiB: {}B", C32::size_per_gib());
    }

    /// Testing the related frames heuristic for frees
    #[test]
    fn last_frees() {
        let local = Local::<4>::default();
        let frame1 = 43;
        let i1 = frame1 / (512 * 512);
        assert!(!local.frees_in_tree(i1));
        local.frees_push(i1);
        local.frees_push(i1);
        local.frees_push(i1);
        assert!(!local.frees_in_tree(i1));
        local.frees_push(i1);
        assert!(local.frees_in_tree(i1));
        let frame2 = 512 * 512 + 43;
        let i2 = frame2 / (512 * 512);
        assert_ne!(i1, i2);
        local.frees_push(i2);
        assert!(!local.frees_in_tree(i1));
        assert!(!local.frees_in_tree(i2));
    }

    #[test]
    fn simple() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 1 << 30;
        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        assert_eq!(alloc.free_frames(), alloc.frames());

        warn!("start alloc...");
        let small = alloc.get(0, 0).unwrap();

        assert_eq!(alloc.allocated_frames(), 1, "{alloc:?}");
        warn!("stress test...");

        // Stress test
        let mut frames = Vec::new();
        loop {
            match alloc.get(0, 0) {
                Ok(frame) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }

        warn!("allocated {}", 1 + frames.len());
        warn!("check...");

        assert_eq!(alloc.allocated_frames(), 1 + frames.len());
        assert_eq!(alloc.allocated_frames(), alloc.frames());
        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("realloc...");

        // Free some
        const FREE_NUM: usize = PT_LEN * PT_LEN - 10;
        for frame in &frames[..FREE_NUM] {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(
            alloc.allocated_frames(),
            1 + frames.len() - FREE_NUM,
            "{alloc:?}"
        );

        // Realloc
        for frame in &mut frames[..FREE_NUM] {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        alloc.put(0, small, 0).unwrap();
        // Free all
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn rand() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 4 << 30;
        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        warn!("start alloc...");
        const ALLOCS: usize = MEM_SIZE / Frame::SIZE / 4 * 3;
        let mut frames = Vec::with_capacity(ALLOCS);
        for _ in 0..ALLOCS {
            frames.push(alloc.get(0, 0).unwrap());
        }
        warn!("allocated {}", frames.len());

        warn!("check...");
        assert_eq!(alloc.allocated_frames(), frames.len());
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("reallocate rand...");
        let mut rng = WyRand::new(100);
        rng.shuffle(&mut frames);

        for _ in 0..frames.len() {
            let i = rng.range(0..frames.len() as _) as usize;
            alloc.put(0, frames[i], 0).unwrap();
            frames[i] = alloc.get(0, 0).unwrap();
        }

        warn!("check...");
        assert_eq!(alloc.allocated_frames(), frames.len());
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("free...");
        rng.shuffle(&mut frames);
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }
        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn multirand() {
        const THREADS: usize = 4;
        const MEM_SIZE: usize = (8 << 30) / Frame::SIZE;
        const ALLOCS: usize = ((MEM_SIZE / THREADS) / 4) * 3;

        logging();
        let area = PFN(0)..PFN(MEM_SIZE);
        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        let barrier = Barrier::new(THREADS);
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);

            barrier.wait();
            warn!("start alloc...");
            let mut frames = Vec::with_capacity(ALLOCS);
            for _ in 0..ALLOCS {
                frames.push(alloc.get(t, 0).unwrap());
            }
            warn!("allocated {}", frames.len());

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for &[a, b] in frames.array_windows() {
                assert_ne!(a, b);
                assert!(area.contains(&a) && area.contains(&b));
            }

            barrier.wait();
            warn!("reallocate rand...");
            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for _ in 0..frames.len() {
                let i = rng.range(0..frames.len() as _) as usize;
                alloc.put(t, frames[i], 0).unwrap();
                frames[i] = alloc.get(t, 0).unwrap();
            }

            warn!("check...");
            // Check that the same frame was not allocated twice
            frames.sort_unstable();
            for &[a, b] in frames.array_windows() {
                assert_ne!(a, b);
                assert!(area.contains(&a) && area.contains(&b));
            }

            warn!("free...");
            rng.shuffle(&mut frames);
            for frame in &frames {
                alloc.put(t, *frame, 0).unwrap();
            }
        });

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn init() {
        logging();
        // 8GiB
        const MEM_SIZE: usize = 8 << 30;
        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        info!("mmap {MEM_SIZE} bytes at {:?}", area.as_ptr_range());

        let alloc = Allocator::new(1, area.clone(), Init::Volatile, true).unwrap();

        assert_eq!(alloc.allocated_frames(), 0);

        warn!("start alloc");
        let small = alloc.get(0, 0).unwrap();
        let huge = alloc.get(0, 9).unwrap();

        let expected_frames = 1 + (1 << 9);
        assert_eq!(alloc.allocated_frames(), expected_frames);
        assert!(small != huge);

        warn!("start stress test");

        // Stress test
        let mut frames = vec![PFN(0); PT_LEN * PT_LEN];
        for frame in &mut frames {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("check");

        assert_eq!(alloc.allocated_frames(), expected_frames + frames.len());

        frames.sort_unstable();

        // Check that the same frame was not allocated twice
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        warn!("free some...");

        // Free some
        for frame in &frames[10..PT_LEN + 10] {
            alloc.put(0, *frame, 0).unwrap();
        }

        warn!("free special...");

        alloc.put(0, small, 0).unwrap();
        alloc.put(0, huge, 9).unwrap();

        warn!("realloc...");

        // Realloc
        for frame in &mut frames[10..PT_LEN + 10] {
            *frame = alloc.get(0, 0).unwrap();
        }

        warn!("free...");

        // Free all
        for frame in &frames {
            alloc.put(0, *frame, 0).unwrap();
        }

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn parallel_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const PAGES: usize = 2 * THREADS * PT_LEN * PT_LEN;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        // Stress test
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, 0).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.allocated_frames(), frames.len());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }
    }

    #[test]
    #[ignore]
    fn less_mem() {
        logging();

        const THREADS: usize = 4;
        const PAGES: usize = 4096;
        const ALLOC_PER_THREAD: usize = PAGES / THREADS - THREADS;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        // Stress test
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, 0).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.allocated_frames(), frames.len());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }

        thread::parallel(
            frames.chunks(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    alloc.put(t, *frame, 0).unwrap();
                }
            },
        );

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn parallel_huge_alloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN - 1;
        // additional space for the allocators metadata
        const PAGES: usize = THREADS * (ALLOC_PER_THREAD + 2) * PT_LEN;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, true).unwrap();

        // Stress test
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = alloc.get(t, 9).unwrap();
                }
            },
        );
        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

        assert_eq!(alloc.allocated_frames(), frames.len() * PT_LEN);
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for &[a, b] in frames.array_windows() {
            assert_ne!(a, b);
            assert!(area.contains(&a) && area.contains(&b));
        }
    }

    #[ignore]
    #[test]
    fn parallel_malloc() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut frames = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = unsafe { libc::malloc(Frame::SIZE) } as u64;
                    assert!(*frame != 0);
                }
            },
        );

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        let mut last = None;
        for p in frames {
            assert!(last != Some(p));
            unsafe { libc::free(p as _) };
            last = Some(p);
        }
    }

    #[ignore]
    #[test]
    fn parallel_mmap() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);

        // Stress test
        let mut frames = vec![0u64; ALLOC_PER_THREAD * THREADS];
        let barrier = Barrier::new(THREADS);
        let timer = Instant::now();

        thread::parallel(
            frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
            |(t, frames)| {
                thread::pin(t);
                barrier.wait();

                for frame in frames {
                    *frame = unsafe {
                        libc::mmap(
                            null_mut(),
                            Frame::SIZE,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                            -1,
                            0,
                        )
                    } as u64;
                    assert!(*frame != 0);
                }
            },
        );

        warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
        warn!("allocated frames: {}", frames.len());

        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        let mut last = None;
        for p in frames {
            assert!(last != Some(p));
            unsafe { libc::munmap(p as _, Frame::SIZE) };
            last = Some(p);
        }
    }

    #[test]
    fn parallel_free() {
        logging();

        const THREADS: usize = 4;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 2 * THREADS);
        const MEM_SIZE: usize = 2 * THREADS * PT_LEN * PT_LEN * Frame::SIZE;

        let area = PFN(0)..PFN(MEM_SIZE / Frame::SIZE);

        let alloc = Allocator::new(THREADS, area, Init::Volatile, true).unwrap();
        let barrier = Barrier::new(THREADS);

        // Stress test
        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            barrier.wait();

            let mut frames = vec![PFN(0); ALLOC_PER_THREAD];

            for frame in &mut frames {
                *frame = alloc.get(t, 0).unwrap();
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for frame in frames {
                alloc.put(t, frame, 0).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn alloc_free() {
        logging();
        const THREADS: usize = 2;
        const ALLOC_PER_THREAD: usize = PT_LEN * (PT_LEN - 10) / 2;

        let area = PFN(0)..PFN(4 * PT_LEN * PT_LEN);

        let alloc = Allocator::new(THREADS, area, Init::Volatile, true).unwrap();

        // Alloc on first thread
        thread::pin(0);
        let mut frames = vec![PFN(0); ALLOC_PER_THREAD];
        for frame in &mut frames {
            *frame = alloc.get(0, 0).unwrap();
        }

        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            s.spawn(|| {
                thread::pin(1);
                barrier.wait();
                // Free on another thread
                for frame in &frames {
                    alloc.put(1, *frame, 0).unwrap();
                }
            });

            let mut frames = vec![PFN(0); ALLOC_PER_THREAD];

            barrier.wait();

            // Simultaneously alloc on first thread
            for frame in &mut frames {
                *frame = alloc.get(0, 0).unwrap();
            }
        });

        warn!("check");
        assert_eq!(alloc.allocated_frames(), ALLOC_PER_THREAD);
    }

    #[test]
    fn recover() {
        logging();

        let mapping = test_mapping(0x1000_0000_0000, 8 << 18).unwrap();
        let area = pfn_range(&mapping);
        warn!("{area:?}");
        thread::pin(0);

        let expected_frames = (PT_LEN + 2) * (1 + (1 << 9));

        {
            let alloc = Allocator::new(1, area.clone(), Init::Overwrite, true).unwrap();

            for _ in 0..PT_LEN + 2 {
                alloc.get(0, 0).unwrap();
                alloc.get(0, 9).unwrap();
            }

            assert_eq!(alloc.allocated_frames(), expected_frames);

            // leak (crash)
            std::mem::forget(alloc);
        }

        let mut alloc = Allocator::default();
        alloc.init(1, area, Init::Recover, true).unwrap();
        assert_eq!(alloc.allocated_frames(), expected_frames);
    }

    #[test]
    fn different_orders() {
        const MAX_ORDER: usize = Lower::MAX_ORDER;
        const THREADS: usize = 4;

        logging();

        let area = PFN(0)..PFN(Lower::N * (THREADS * 2 + 1));
        let alloc = Allocator::new(THREADS, area, Init::Volatile, true).unwrap();

        let barrier = Barrier::new(THREADS);

        thread::parallel(0..THREADS, |t| {
            thread::pin(t);
            let mut rng = WyRand::new(42 + t as u64);
            let mut num_frames = 0;
            let mut frames = Vec::new();
            for order in 0..=MAX_ORDER {
                for _ in 0..1 << (MAX_ORDER - order) {
                    frames.push((order, PFN(0)));
                    num_frames += 1 << order;
                }
            }
            rng.shuffle(&mut frames);

            warn!("allocate {num_frames} frames up to order <{MAX_ORDER}");
            barrier.wait();

            for (order, frame) in &mut frames {
                *frame = match alloc.get(t, *order) {
                    Ok(frame) => frame,
                    Err(e) => panic!("{e:?} o={order} {alloc:?}"),
                };
                assert!(frame.0 % (1 << *order) == 0, "{frame} {:x}", 1 << *order);
            }

            let mut rng = WyRand::new(t as _);
            rng.shuffle(&mut frames);

            for (order, frame) in frames {
                match alloc.put(t, frame, order) {
                    Ok(_) => {}
                    Err(e) => panic!("{e:?} o={order} {alloc:?}"),
                }
            }
        });

        assert_eq!(alloc.allocated_frames(), 0);
    }

    #[test]
    fn init_reserved() {
        logging();

        const THREADS: usize = 2;
        const PAGES: usize = 8 << 18;

        let area = PFN(0)..PFN(PAGES);

        let alloc = Allocator::new(THREADS, area.clone(), Init::Volatile, false).unwrap();
        assert_eq!(alloc.frames(), PAGES);
        assert_eq!(alloc.allocated_frames(), PAGES);

        for frame in area.as_range().step_by(1 << Lower::HUGE_ORDER).map(PFN) {
            alloc.put(0, frame, Lower::HUGE_ORDER).unwrap();
        }
        assert_eq!(alloc.allocated_frames(), 0);

        thread::parallel(0..THREADS, |core| {
            thread::pin(core);
            for _ in 0..(PAGES / THREADS) / (1 << Lower::HUGE_ORDER) {
                alloc.get(core, Lower::HUGE_ORDER).unwrap();
            }
        });
        assert_eq!(alloc.allocated_frames(), PAGES);
    }

    #[test]
    fn fragmentation_retry() {
        logging();

        let area = PFN(0)..PFN(Lower::N * 2);
        let alloc = Allocator::new(1, area, Init::Volatile, true).unwrap();

        // Alloc a whole subtree
        let mut frames = Vec::with_capacity(Lower::N / 2);
        for i in 0..Lower::N {
            if i % 2 == 0 {
                frames.push(alloc.get(0, 0).unwrap());
            } else {
                alloc.get(0, 0).unwrap();
            }
        }
        // Free every second one -> fragmentation
        for frame in frames {
            alloc.put(0, frame, 0).unwrap();
        }

        let huge = alloc.get(0, 9).unwrap();
        warn!("huge = {huge}");
        warn!("{alloc:?}");
    }

    #[test]
    fn drain() {
        let area = PFN(0)..PFN(Lower::N * 2);
        let alloc = Allocator::new(2, area, Init::Volatile, true).unwrap();
        // should not change anything
        alloc.drain(0).unwrap();
        alloc.drain(1).unwrap();

        // allocate on second core => reserve a subtree
        alloc.get(1, 0).unwrap();

        // completely the subtree of the first core
        for _ in 0..Lower::N {
            alloc.get(0, 0).unwrap();
        }
        // next allocation should trigger drain+reservation (no subtree left)
        println!("{:?}", alloc.get(0, 0));
    }
}
