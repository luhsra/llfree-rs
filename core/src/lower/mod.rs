use core::fmt;
use core::ops::Range;

use crate::frame::PFN;
use crate::upper::Init;
use crate::Result;

mod cache;
pub use cache::Cache;

/// Lower-level frame allocator.
///
/// This level implements the actual allocation/free operations.
/// Each allocation/free is limited to a chunk of [LowerAlloc::N] frames.
pub trait LowerAlloc: Default + fmt::Debug {
    /// Pages per chunk. Every alloc only searches in a chunk of this size.
    const N: usize;
    /// The maximal allowed order of this allocator
    const MAX_ORDER: usize;
    const HUGE_ORDER: usize;

    /// Create a new lower allocator.
    fn new(cores: usize, begin: PFN, len: usize, init: Init, free_all: bool) -> Self;

    fn frames(&self) -> usize;
    fn begin(&self) -> PFN;

    fn memory(&self) -> Range<PFN> {
        self.begin()..PFN(self.begin().0 + self.frames())
    }

    /// Recovers the data structures for the [LowerAlloc::N] sized chunk at `start`.
    /// `deep` indicates that the allocator has crashed and the
    /// recovery might have to be more extensive.
    /// Returns the number of recovered frames.
    fn recover(&self, start: usize, deep: bool) -> Result<usize>;

    /// Try allocating a new `frame` in the [LowerAlloc::N] sized chunk at `start`.
    fn get(&self, start: usize, order: usize) -> Result<usize>;
    /// Try freeing a `frame`.
    fn put(&self, frame: usize, order: usize) -> Result<()>;
    /// Returns if the frame is free. This might be racy!
    fn is_free(&self, frame: usize, order: usize) -> bool;

    /// Debug function, returning the number of allocated frames and performing internal checks.
    fn allocated_frames(&self) -> usize;
    /// Debug function returning number of free frames in each order 9 chunk
    fn for_each_huge_frame<F: FnMut(usize, usize)>(&self, f: F);

    fn size_per_gib() -> usize;
}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::sync::Arc;

    use crate::frame::Frame;
    use crate::lower::LowerAlloc;
    use crate::table::PT_LEN;
    use crate::thread;
    use crate::upper::Init;
    use crate::util::logging;

    type Lower = super::cache::Cache<32>;

    #[test]
    #[ignore]
    fn rand_realloc_first() {
        logging();

        const THREADS: usize = 6;
        let buffer = vec![Frame::new(); 2 * THREADS * PT_LEN * PT_LEN];

        for _ in 0..8 {
            let lower = Arc::new(Lower::new(
                THREADS,
                buffer.as_ptr().into(),
                buffer.len(),
                Init::Overwrite,
                true,
            ));
            assert_eq!(lower.allocated_frames(), 0);

            thread::parallel(0..THREADS, |t| {
                thread::pin(t);

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
        let mut frames = [0; PT_LEN];
        let buffer = vec![Frame::new(); 2 * THREADS * PT_LEN * PT_LEN];

        for _ in 0..8 {
            let lower = Arc::new(Lower::new(
                THREADS,
                buffer.as_ptr().into(),
                buffer.len(),
                Init::Overwrite,
                true,
            ));
            assert_eq!(lower.allocated_frames(), 0);

            for frame in &mut frames[..PT_LEN - 3] {
                *frame = lower.get(0, 0).unwrap();
            }

            thread::parallel(0..THREADS, |t| {
                thread::pin(t);

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
