use core::fmt;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use std::sync::Barrier;
use std::time::Instant;
use std::vec::Vec;

#[cfg(all(feature = "llc", not(feature = "llzig")))]
use llfree_eval::LLC;
#[cfg(all(feature = "llzig"))]
use llfree_eval::LLZig;
use llfree_eval::{mmap, thread};
use log::{error, warn};

use llfree::frame::Frame;
use llfree::util::{WyRand, aligned_buf, logging};
use llfree::wrapper::NvmAlloc;
use llfree::*;

#[cfg(all(feature = "llc", not(feature = "llzig")))]
type Allocator = TestAlloc<LLC>;
#[cfg(feature = "llzig")]
type Allocator = TestAlloc<LLZig>;
#[cfg(not(any(feature = "llc", feature = "llzig")))]
type Allocator = TestAlloc<LLFree<'static>>;

pub struct TestAlloc<A: Alloc<'static>>(ManuallyDrop<A>);

impl<A: Alloc<'static>> TestAlloc<A> {
    pub fn create(
        cores: usize,
        frames: usize,
        init: Init,
    ) -> Result<(Self, impl Fn(usize, usize) -> Request)> {
        let (tiering, request) = Tiering::simple(cores);
        let MetaSize {
            local,
            trees,
            lower,
        } = A::metadata_size(&tiering, frames);
        let meta = MetaData {
            local: aligned_buf(local),
            trees: aligned_buf(trees),
            lower: aligned_buf(lower),
        };
        Ok((
            Self(ManuallyDrop::new(A::new(frames, init, &tiering, meta)?)),
            request,
        ))
    }
}
impl<A: Alloc<'static>> Drop for TestAlloc<A> {
    fn drop(&mut self) {
        unsafe {
            let MetaData {
                local,
                trees,
                lower,
            } = self.0.metadata();
            // drop first
            drop(ManuallyDrop::take(&mut self.0));
            // free metadata buffers
            Vec::from_raw_parts(local.as_mut_ptr(), local.len(), local.len());
            Vec::from_raw_parts(trees.as_mut_ptr(), trees.len(), trees.len());
            Vec::from_raw_parts(lower.as_mut_ptr(), lower.len(), lower.len());
        }
    }
}
impl<A: Alloc<'static>> Deref for TestAlloc<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<A: Alloc<'static>> fmt::Debug for TestAlloc<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        A::fmt(self, f)
    }
}

#[test]
fn minimal() {
    logging();
    // 8GiB
    const MEM_SIZE: usize = 1 << 30;
    let frames = MEM_SIZE / Frame::SIZE;

    warn!("init");
    let (alloc, request) = Allocator::create(1, frames, Init::FreeAll).unwrap();
    warn!("finit");

    assert_eq!(alloc.tree_stats().free_frames, alloc.frames());

    warn!("get >>>");
    let (frame1, _) = alloc.get(None, request(0, 0)).unwrap();
    warn!("get <<<");
    warn!("get >>>");
    let (frame2, _) = alloc.get(None, request(0, 0)).unwrap();
    warn!("get <<<");

    warn!("put >>>");
    alloc.put(frame2, request(0, 0)).unwrap();
    warn!("put <<<");
    warn!("put >>>");
    alloc.put(frame1, request(0, 0)).unwrap();
    warn!("put <<<");
    alloc.validate();
}

#[test]
fn simple() {
    logging();
    // 8GiB
    const MEM_SIZE: usize = 8 * (1 << 30);
    const FRAMES: usize = MEM_SIZE / Frame::SIZE;

    let (alloc, request) = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();
    warn!("{alloc:?}");

    assert_eq!(alloc.tree_stats().free_frames, alloc.frames());

    warn!("start alloc...");
    let (small, _) = alloc.get(None, request(0, 0)).unwrap();

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        1,
        "{alloc:?}"
    );
    warn!("stress test...");

    // Stress test
    let mut frames = Vec::new();
    loop {
        match alloc.get(None, request(0, 0)) {
            Ok((frame, _)) => frames.push(frame),
            Err(Error::Memory) => break,
            Err(e) => panic!("{e:?}"),
        }
    }

    warn!("allocated {}", 1 + frames.len());
    warn!("check...");
    alloc.validate();

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        1 + frames.len()
    );
    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        alloc.frames()
    );
    frames.sort_unstable();

    // Check that the same frame was not allocated twice
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }

    warn!("realloc...");

    // Free some
    const FREE_NUM: usize = HUGE_FRAMES - 10;
    for frame in &frames[..FREE_NUM] {
        alloc.put(*frame, request(0, 0)).unwrap();
    }

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        1 + frames.len() - FREE_NUM,
        "{alloc:?}"
    );
    alloc.validate();

    // Realloc
    for frame in &mut frames[..FREE_NUM] {
        *frame = alloc.get(None, request(0, 0)).unwrap().0;
    }

    warn!("free...");

    alloc.put(small, request(0, 0)).unwrap();
    // Free all
    for frame in &frames {
        alloc.put(*frame, request(0, 0)).unwrap();
    }

    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate();
}

#[test]
fn alloc_at() {
    logging();
    const MEM_SIZE: usize = 1 << 30;
    let frames = MEM_SIZE / Frame::SIZE;

    let (alloc, request) = Allocator::create(1, frames, Init::FreeAll).unwrap();

    assert_eq!(alloc.tree_stats().free_frames, alloc.frames());

    alloc.get(Some(FrameId(1)), request(0, 0)).unwrap();
    alloc.get(Some(FrameId(2)), request(0, 0)).unwrap();
    alloc
        .get(Some(FrameId(HUGE_FRAMES)), request(HUGE_ORDER, 0))
        .unwrap();

    // Test normal allocation
    let (frame, _) = alloc.get(None, request(0, 0)).unwrap();
    assert!(frame != FrameId(1) && frame != FrameId(2) && frame != FrameId(HUGE_FRAMES));

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        3 + HUGE_FRAMES
    );
    alloc.validate();

    alloc
        .put(FrameId(HUGE_FRAMES), request(HUGE_ORDER, 0))
        .unwrap();
    alloc.put(FrameId(2), request(0, 0)).unwrap();
    alloc.put(FrameId(1), request(0, 0)).unwrap();
    alloc.put(frame, request(0, 0)).unwrap();

    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate();
}

#[test]
fn rand() {
    logging();
    // 8GiB
    const MEM_SIZE: usize = 4 << 30;
    const FRAMES: usize = MEM_SIZE / Frame::SIZE;

    let (alloc, request) = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

    warn!("start alloc...");
    const ALLOCS: usize = MEM_SIZE / Frame::SIZE / 2;
    let mut frames = Vec::with_capacity(ALLOCS);
    for _ in 0..ALLOCS {
        frames.push(alloc.get(None, request(0, 0)).unwrap().0);
    }
    warn!("allocated {}", frames.len());

    warn!("check...");
    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len()
    );
    alloc.validate();

    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }

    warn!("reallocate rand...");
    let mut rng = WyRand::new(100);
    rng.shuffle(&mut frames);

    for _ in 0..frames.len() {
        let i = rng.range(0..frames.len() as _) as usize;
        alloc.put(frames[i], request(0, 0)).unwrap();
        frames[i] = alloc.get(None, request(0, 0)).unwrap().0;
    }

    warn!("check...");
    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len()
    );
    alloc.validate();
    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }

    warn!("free...");
    rng.shuffle(&mut frames);
    for frame in &frames {
        alloc.put(*frame, request(0, 0)).unwrap();
    }
    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate()
}

#[test]
fn multirand() {
    const THREADS: usize = 4;
    const FRAMES: usize = (8 << 30) / Frame::SIZE;
    const ALLOCS: usize = ((FRAMES / THREADS) / 4) * 3;

    logging();

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

    let barrier = Barrier::new(THREADS);
    thread::parallel(0..THREADS, |t| {
        thread::pin(t);

        barrier.wait();
        warn!("start alloc...");
        let mut frames = Vec::with_capacity(ALLOCS);
        for _ in 0..ALLOCS {
            frames.push(alloc.get(None, request(0, t)).unwrap().0);
        }
        warn!("allocated {}", frames.len());

        warn!("check...");
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }

        barrier.wait();
        warn!("reallocate rand...");
        let mut rng = WyRand::new(t as _);
        rng.shuffle(&mut frames);

        for _ in 0..frames.len() {
            let i = rng.range(0..frames.len() as _) as usize;
            alloc.put(frames[i], request(0, t)).unwrap();
            frames[i] = alloc.get(None, request(0, t)).unwrap().0;
        }

        warn!("check...");
        // Check that the same frame was not allocated twice
        frames.sort_unstable();
        for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
            assert_ne!(a, b);
            assert!(a.0 < FRAMES && b.0 < FRAMES);
        }

        if barrier.wait().is_leader() {
            alloc.validate();
        }
        barrier.wait();

        warn!("free...");
        rng.shuffle(&mut frames);
        for frame in &frames {
            alloc.put(*frame, request(0, t)).unwrap();
        }
    });

    assert_eq!(alloc.stats().free_huge, FRAMES / HUGE_FRAMES);
    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate();
}

#[test]
fn parallel_alloc() {
    logging();

    const THREADS: usize = 4;
    const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 2 * THREADS);
    const FRAMES: usize = 2 * THREADS * HUGE_FRAMES * HUGE_FRAMES;

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

    // Stress test
    let mut frames = vec![FrameId(0); ALLOC_PER_THREAD * THREADS];
    let barrier = Barrier::new(THREADS);
    let timer = Instant::now();

    thread::parallel(
        frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
        |(t, frames)| {
            thread::pin(t);
            barrier.wait();

            for frame in frames {
                *frame = alloc.get(None, request(0, t)).unwrap().0;
            }
        },
    );
    warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len()
    );
    warn!("allocated frames: {}", frames.len());
    alloc.validate();

    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }
}

#[test]
fn alloc_all() {
    logging();

    const FRAMES: usize = 2 * HUGE_FRAMES * HUGE_FRAMES;

    let (alloc, request) = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

    // Stress test
    let mut frames = Vec::new();
    let timer = Instant::now();

    loop {
        match alloc.get(None, request(0, 0)) {
            Ok((frame, _)) => frames.push(frame),
            Err(Error::Memory) => break,
            Err(e) => panic!("{e:?}"),
        }
    }
    warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
    warn!("{alloc:?}");

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len()
    );
    assert_eq!(alloc.tree_stats().free_frames, 0);
    assert_eq!(alloc.stats().free_huge, 0);
    warn!("allocated frames: {}", frames.len());
    alloc.validate();

    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }
}

#[test]
fn parallel_alloc_all() {
    logging();

    const THREADS: usize = 4;
    const FRAMES: usize = 2 * THREADS * HUGE_FRAMES * HUGE_FRAMES;

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

    // Stress test
    let mut frames = vec![Vec::new(); THREADS];
    let barrier = Barrier::new(THREADS);
    let timer = Instant::now();

    thread::parallel(frames.iter_mut().enumerate(), |(t, frames)| {
        thread::pin(t);
        barrier.wait();

        loop {
            match alloc.get(None, request(0, t)) {
                Ok((frame, _)) => frames.push(frame),
                Err(Error::Memory) => break,
                Err(e) => panic!("{e:?}"),
            }
        }
    });
    warn!("Allocation finished in {}ms", timer.elapsed().as_millis());
    warn!("{alloc:?}");

    let mut frames = frames.into_iter().flatten().collect::<Vec<_>>();

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len()
    );
    warn!("allocated frames: {}/{}", frames.len(), alloc.frames());
    alloc.validate();

    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }
}

#[test]
fn less_mem() {
    logging();

    const THREADS: usize = 4;
    const FRAMES: usize = 4096;
    const ALLOC_PER_THREAD: usize = FRAMES / THREADS - THREADS;

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

    // Stress test
    let mut frames = vec![FrameId(0); ALLOC_PER_THREAD * THREADS];
    let barrier = Barrier::new(THREADS);
    let timer = Instant::now();

    thread::parallel(
        frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
        |(t, frames)| {
            thread::pin(t);
            barrier.wait();

            for (i, frame) in frames.iter_mut().enumerate() {
                if let Ok((f, _)) = alloc.get(None, request(0, t)) {
                    *frame = f;
                } else {
                    error!("OOM: {i}: {alloc:#?}");
                    panic!()
                }
            }
        },
    );
    warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len()
    );
    warn!("allocated frames: {}", frames.len());
    alloc.validate();

    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }

    thread::parallel(
        frames.chunks(ALLOC_PER_THREAD).enumerate(),
        |(t, frames)| {
            thread::pin(t);
            barrier.wait();

            for frame in frames {
                alloc.put(*frame, request(0, t)).unwrap();
            }
        },
    );

    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate();
}

#[test]
fn parallel_huge_alloc() {
    logging();

    const THREADS: usize = 4;
    const FRAMES: usize = (8 << 30) / Frame::SIZE; // 1GiB
    const ALLOC_PER_THREAD: usize = FRAMES / THREADS / HUGE_FRAMES;
    // additional space for the allocators metadata

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();

    // Stress test
    let mut frames = vec![FrameId(0); ALLOC_PER_THREAD * THREADS];
    let barrier = Barrier::new(THREADS);
    let timer = Instant::now();

    thread::parallel(
        frames.chunks_mut(ALLOC_PER_THREAD).enumerate(),
        |(t, frames)| {
            thread::pin(t);
            barrier.wait();

            for frame in frames {
                *frame = alloc.get(None, request(HUGE_ORDER, t)).unwrap().0;
            }
        },
    );
    warn!("Allocation finished in {}ms", timer.elapsed().as_millis());

    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        frames.len() * HUGE_FRAMES
    );
    warn!(
        "allocated frames: {}/{}",
        frames.len(),
        alloc.frames() / HUGE_FRAMES
    );
    alloc.validate();

    // Check that the same frame was not allocated twice
    frames.sort_unstable();
    for (a, b) in frames.windows(2).map(|p| (p[0], p[1])) {
        assert_ne!(a, b);
        assert!(a.0 < FRAMES && b.0 < FRAMES);
    }
}

#[test]
fn parallel_free() {
    logging();

    const THREADS: usize = 4;
    const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 2 * THREADS);
    const MEM_SIZE: usize = 2 * THREADS * HUGE_FRAMES * HUGE_FRAMES * Frame::SIZE;

    let area = MEM_SIZE / Frame::SIZE;

    let (alloc, request) = Allocator::create(THREADS, area, Init::FreeAll).unwrap();
    let barrier = Barrier::new(THREADS);

    // Stress test
    thread::parallel(0..THREADS, |t| {
        thread::pin(t);
        barrier.wait();

        let mut frames = vec![FrameId(0); ALLOC_PER_THREAD];

        for frame in &mut frames {
            *frame = alloc.get(None, request(0, t)).unwrap().0;
        }

        let mut rng = WyRand::new(t as _);
        rng.shuffle(&mut frames);

        for frame in frames {
            alloc.put(frame, request(0, t)).unwrap();
        }
    });

    warn!("check");
    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate();
}

#[test]
fn alloc_free() {
    logging();
    const THREADS: usize = 2;
    const ALLOC_PER_THREAD: usize = HUGE_FRAMES * (HUGE_FRAMES - 10) / 2;
    const FRAMES: usize = 4 * HUGE_FRAMES * HUGE_FRAMES;

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
    warn!("{alloc:?}");

    // Alloc on first thread
    thread::pin(0);
    let mut frames = vec![FrameId(0); ALLOC_PER_THREAD];
    for frame in &mut frames {
        *frame = alloc.get(None, request(0, 0)).unwrap().0;
    }
    alloc.validate();

    let barrier = Barrier::new(THREADS);
    std::thread::scope(|s| {
        s.spawn(|| {
            thread::pin(1);
            barrier.wait();
            // Free on another thread
            for frame in &frames {
                alloc.put(*frame, request(0, 1)).unwrap();
            }
        });

        let mut frames = vec![FrameId(0); ALLOC_PER_THREAD];

        barrier.wait();

        // Simultaneously alloc on first thread
        for frame in &mut frames {
            *frame = alloc.get(None, request(0, 0)).unwrap().0;
        }
    });

    warn!("check {alloc:?}");
    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        ALLOC_PER_THREAD
    );
    alloc.validate();
}

#[test]
fn recover() {
    #[cfg(feature = "llc")]
    type Allocator<'a> = NvmAlloc<'a, LLC>;
    #[cfg(not(feature = "llc"))]
    type Allocator<'a> = NvmAlloc<'a, LLFree<'a>>;

    logging();

    const FRAMES: usize = 8 * (1 << 30) / FRAME_SIZE;

    thread::pin(0);

    let expected_frames = 128 * (1 + (1 << HUGE_ORDER));

    let mut zone = mmap::Mapping::anon(0x1000_0000_0000, FRAMES, false, false).unwrap();
    let (tiering, request) = Tiering::simple(1);
    let m = Allocator::metadata_size(&tiering, FRAMES);
    let local = aligned_buf(m.local);
    let trees = aligned_buf(m.trees);

    {
        let alloc = Allocator::create(&mut zone, false, &tiering, local, trees).unwrap();

        let mut _allocated_frames = 0;
        for _ in 0..128 {
            alloc.get(None, request(0, 0)).unwrap();
            _allocated_frames = alloc.frames() - alloc.tree_stats().free_frames;
            alloc.get(None, request(HUGE_ORDER, 0)).unwrap();
            _allocated_frames = alloc.frames() - alloc.tree_stats().free_frames;
        }

        assert_eq!(
            alloc.frames() - alloc.tree_stats().free_frames,
            expected_frames
        );
        alloc.validate();

        // leak (crash)
        std::mem::forget(alloc);
    }

    let local = aligned_buf(m.local);
    let trees = aligned_buf(m.trees);
    let alloc = Allocator::create(&mut zone, true, &tiering, local, trees).unwrap();
    assert_eq!(
        alloc.frames() - alloc.tree_stats().free_frames,
        expected_frames
    );
    alloc.validate();
}

#[test]
fn different_orders() {
    const THREADS: usize = 4;
    const FRAMES: usize = (1 << MAX_ORDER) * (MAX_ORDER + 2) * THREADS; // 6 GiB for 16K

    logging();

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
    warn!("Created Allocator \n {:?}", alloc);
    let barrier = Barrier::new(THREADS);

    thread::parallel(0..THREADS, |t| {
        thread::pin(t);
        let mut rng = WyRand::new(42 + t as u64);
        let mut num_frames = 0;
        let mut frames = Vec::new();
        for order in 0..=MAX_ORDER {
            for _ in 0..1 << (MAX_ORDER - order) {
                frames.push((order, FrameId(0)));
                num_frames += 1 << order;
            }
        }
        rng.shuffle(&mut frames);

        warn!("allocate {num_frames} frames up to order <= {MAX_ORDER}");
        barrier.wait();

        // reallocate all
        let mut errors = 0;
        for (order, frame) in &mut frames {
            *frame = match alloc.get(None, request(*order, t)) {
                Ok((frame, _)) => frame,
                Err(e) => {
                    // Due to race conditions one might fail, but no more!
                    error!("{e:?} o={order} on core {t}");
                    errors += 1;
                    FrameId(0)
                }
            };
            assert!(frame.is_aligned(*order), "{frame:?} {:x}", 1 << *order);
        }
        assert!(
            errors <= THREADS,
            "Too many allocation failures: {errors} out of {}",
            frames.len()
        );

        rng.shuffle(&mut frames);

        // free all
        for (order, frame) in frames {
            if let Err(e) = alloc.put(frame, request(order, t)) {
                error!("{e:?} o={order}")
            }
        }
    });

    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);
    alloc.validate();
}

#[test]
fn init_reserved() {
    logging();

    const THREADS: usize = 2;
    const FRAMES: usize = 8 << 18;

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::AllocAll).unwrap();
    assert_eq!(alloc.frames(), FRAMES);
    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, FRAMES);

    for frame in (0..FRAMES).step_by(1 << HUGE_ORDER) {
        alloc.put(FrameId(frame), request(HUGE_ORDER, 0)).unwrap();
    }
    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, 0);

    thread::parallel(0..THREADS, |core| {
        thread::pin(core);
        for _ in (0..FRAMES / THREADS).step_by(1 << HUGE_ORDER) {
            alloc.get(None, request(HUGE_ORDER, core)).unwrap();
        }
    });
    assert_eq!(alloc.frames() - alloc.tree_stats().free_frames, FRAMES);
    alloc.validate();
}

#[test]
fn fragmentation_retry() {
    logging();

    const FRAMES: usize = TREE_FRAMES * 2;
    let (alloc, request) = Allocator::create(1, FRAMES, Init::FreeAll).unwrap();

    // Alloc a whole subtree
    let mut frames = Vec::with_capacity(TREE_FRAMES / 2);
    for i in 0..TREE_FRAMES {
        if i % 2 == 0 {
            frames.push(alloc.get(None, request(0, 0)).unwrap().0);
        } else {
            alloc.get(None, request(0, 0)).unwrap().0;
        }
    }
    // Free every second one -> fragmentation
    for frame in frames {
        alloc.put(frame, request(0, 0)).unwrap();
    }

    let huge = alloc.get(None, request(9, 0)).unwrap();
    warn!("huge = {:?}", huge);
    warn!("{alloc:?}");
    alloc.validate();
}

#[test]
fn drain() {
    logging();

    const FRAMES: usize = TREE_FRAMES * 8;
    let (alloc, request) = Allocator::create(2, FRAMES, Init::FreeAll).unwrap();
    // should not change anything
    alloc.drain();

    // allocate on second core => reserve a subtree
    alloc.get(None, request(0, 1)).unwrap();

    // completely the subtree of the first core
    for _ in 0..FRAMES - TREE_FRAMES {
        alloc.get(None, request(0, 0)).unwrap();
    }
    // next allocation should trigger drain+reservation (no subtree left)
    println!("{:?}", alloc.get(None, request(0, 0)));
    alloc.validate();
}

#[test]
fn stress() {
    const THREADS: usize = 4;
    const FRAMES: usize = (1 << 30) / Frame::SIZE;
    const ITER: usize = 100;

    logging();

    let (alloc, request) = Allocator::create(THREADS, FRAMES, Init::FreeAll).unwrap();
    alloc.validate();

    let rand = unsafe { libc::rand() as u64 };
    let barrier = Barrier::new(THREADS);

    let allocated = thread::parallel(0..THREADS, |t| {
        thread::pin(t);
        let mut rng = WyRand::new(rand + t as u64);
        let mut frames = Vec::with_capacity(FRAMES / THREADS);
        barrier.wait();

        for _ in 0..ITER {
            let target = rng.range(0..(FRAMES / THREADS) as _) as usize;
            while frames.len() != target {
                if frames.len() < target {
                    match alloc.get(None, request(0, t)) {
                        Ok((frame, _)) => frames.push(frame),
                        Err(Error::Memory) => break,
                        Err(e) => panic!("{e:?}"),
                    }
                } else {
                    alloc.put(frames.pop().unwrap(), request(0, t)).unwrap();
                }
            }
        }
        frames.len()
    });

    assert_eq!(
        allocated.into_iter().sum::<usize>(),
        alloc.frames() - alloc.tree_stats().free_frames
    );
    alloc.validate();
}
