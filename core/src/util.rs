//! General utility functions

use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut, Range};

/// Retries the condition n times and returns if it was successfull.
/// This pauses the CPU between retries if possible.
pub fn spin_wait(n: usize, mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..n {
        if cond() {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

/// Align v up to next `align`
#[inline(always)]
pub const fn align_up(v: usize, align: usize) -> usize {
    v.next_multiple_of(align)
}

/// Align v down to previous `align` (power of two!)
#[inline(always)]
pub const fn align_down(v: usize, align: usize) -> usize {
    (v / align) * align
}

/// Calculate the size of a slice of T, respecting any alignment constraints
///
/// Note: This might not be correct for all types, but it is for the ones we use.
pub const fn size_of_slice<T>(len: usize) -> usize {
    len * size_of::<T>().next_multiple_of(align_of::<T>())
}

/// Cache alignment for T
#[derive(Clone, Default, Hash, PartialEq, Eq)]
#[repr(align(64))]
pub struct Align<T = ()>(pub T);

const _: () = assert!(align_of::<Align>() == 64);
const _: () = assert!(align_of::<Align<usize>>() == 64);
const _: () = assert!(size_of::<Align<usize>>() == 64);

impl<T> Deref for Align<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T> DerefMut for Align<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
impl<T: fmt::Debug> fmt::Debug for Align<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[cfg(any(test, feature = "std"))]
pub fn logging() {
    use core::mem::transmute;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::boxed::Box;
    use std::io::Write;
    use std::thread::ThreadId;

    static MAX_LOC_WIDTH: AtomicUsize = AtomicUsize::new(0);

    struct Padding(usize);
    impl fmt::Display for Padding {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:width$}", "", width = self.0)
        }
    }

    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(move |buf, record| {
            const DIM: &str = "\x1b[90m";
            const RST: &str = "\x1b[0m";
            let color = match record.level() {
                log::Level::Error => "\x1b[91m",
                log::Level::Warn => "\x1b[93m",
                log::Level::Info => "",
                log::Level::Debug => DIM,
                log::Level::Trace => DIM,
            };

            let loc = format_args!(
                "{}:{:<4}",
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default()
            );
            let loc_len = record.file().map_or(0, str::len) + 1 + 4;
            let max = MAX_LOC_WIDTH.fetch_max(loc_len, Ordering::Relaxed);
            let padding = Padding(max.max(loc_len) - loc_len);

            let tid = unsafe { transmute::<ThreadId, u64>(std::thread::current().id()) };
            writeln!(
                buf,
                "{color}{:<5}{RST}{DIM} {tid:02?} {loc}{padding} >{RST}{color} {}{RST}",
                record.level(),
                record.args()
            )
        })
        .try_init();

    std::panic::set_hook(Box::new(panic_handler));
}

#[cfg(any(test, feature = "std"))]
fn panic_handler(info: &std::panic::PanicHookInfo) {
    use std::backtrace::Backtrace;
    log::error!("{info}\n{}", Backtrace::capture());
}

#[cfg(any(test, feature = "std"))]
/// Executed `f` in parallel for each element in `iter`.
pub fn parallel<I, T, F>(iter: I, f: F) -> std::vec::Vec<T>
where
    I: IntoIterator,
    I::Item: Send,
    T: Send,
    F: FnOnce(I::Item) -> T + Clone + Send,
{
    std::thread::scope(|scope| {
        let handles = iter
            .into_iter()
            .map(|t| {
                let f = f.clone();
                scope.spawn(move || f(t))
            })
            .collect::<std::vec::Vec<_>>();
        handles.into_iter().map(|t| t.join().unwrap()).collect()
    })
}

/// Simple bare bones random number generator based on wyhash.
///
/// - See <https://github.com/wangyi-fudan/wyhash>
#[cfg(any(test, feature = "std"))]
pub struct WyRand {
    pub seed: u64,
}

#[cfg(any(test, feature = "std"))]
impl WyRand {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
    pub fn generate(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0xa076_1d64_78bd_642f);
        let t: u128 = (self.seed as u128).wrapping_mul((self.seed ^ 0xe703_7ed1_a0b4_28db) as u128);
        (t.wrapping_shr(64) ^ t) as u64
    }
    pub fn range(&mut self, range: Range<u64>) -> u64 {
        let mut val = self.generate();
        if range.start < range.end {
            val %= range.end - range.start;
            val + range.start
        } else {
            0
        }
    }
    pub fn shuffle<T>(&mut self, target: &mut [T]) {
        if !target.is_empty() {
            for i in 0..target.len() - 1 {
                target.swap(i, self.range(i as u64..target.len() as u64) as usize);
            }
        }
    }
}

#[cfg(any(test, feature = "std"))]
pub fn aligned_buf(size: usize) -> &'static mut [u8] {
    use std::alloc::{Layout, alloc_zeroed};
    let ptr = unsafe { alloc_zeroed(Layout::from_size_align(size, align_of::<Align>()).unwrap()) };
    unsafe { std::slice::from_raw_parts_mut(ptr, size) }
}

#[cfg(test)]
mod test {
    use super::{WyRand, align_down, align_up};

    #[test]
    fn wy_rand() {
        let mut rng = WyRand::new(0);
        let mut buckets = [0usize; 512];
        for _ in 0..512 * buckets.len() {
            buckets[rng.range(0..buckets.len() as _) as usize] += 1;
        }
        let mut min = usize::MAX;
        let mut max = 0;
        let mut avg = 0.0;
        let mut std = 0.0;
        for v in buckets {
            min = min.min(v);
            max = max.max(v);
            avg += v as f64;
            std += (v * v) as f64;
        }
        avg /= buckets.len() as f64;
        std /= buckets.len() as f64;
        std = (std - (avg * avg)).sqrt();
        println!("avg={avg:.2}, std={std:.2}, min={min}, max={max}");
    }

    #[test]
    fn align() {
        assert_eq!(align_down(0, 64), 0);
        assert_eq!(align_down(1, 64), 0);
        assert_eq!(align_down(63, 64), 0);
        assert_eq!(align_down(64, 64), 64);
        assert_eq!(align_down(65, 64), 64);

        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }
}
