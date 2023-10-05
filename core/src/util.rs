//! General utility functions

use core::fmt;
use core::mem::align_of;
use core::ops::{Add, Deref, DerefMut, Div, Range};

/// Align v up to next `align` (power of two!)
#[inline(always)]
pub const fn align_up(v: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    (v + mask) & !mask
}

/// Align v up to previous `align` (power of two!)
#[inline(always)]
pub const fn align_down(v: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    v & !mask
}

/// Cache alignment for T
#[derive(Clone, Default, Hash, PartialEq, Eq)]
#[repr(align(64))]
pub struct Align<T = ()>(pub T);

const _: () = assert!(align_of::<Align>() == 64);
const _: () = assert!(align_of::<Align<usize>>() == 64);

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
impl<T> From<T> for Align<T> {
    fn from(t: T) -> Self {
        Align(t)
    }
}

#[cfg(feature = "std")]
pub fn logging() {
    use core::mem::transmute;
    use std::io::Write;
    use std::thread::ThreadId;

    use crate::thread::pinned;

    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(move |buf, record| {
            let color = match record.level() {
                log::Level::Error => "\x1b[91m",
                log::Level::Warn => "\x1b[93m",
                log::Level::Info => "\x1b[90m",
                log::Level::Debug => "\x1b[90m",
                log::Level::Trace => "\x1b[90m",
            };

            let pin = pinned();
            let pin = if pin > isize::MAX as usize {
                -1
            } else {
                pin as isize
            };

            writeln!(
                buf,
                "{}[{:5} {:02?}@{pin:02?} {}:{}] {}\x1b[0m",
                color,
                record.level(),
                unsafe { transmute::<ThreadId, u64>(std::thread::current().id()) },
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default(),
                record.args()
            )
        })
        .try_init();
}

/// Simple bare bones random number generator based on wyhash.
///
/// - See <https://github.com/wangyi-fudan/wyhash>
pub struct WyRand {
    pub seed: u64,
}

impl WyRand {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
    pub fn gen(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0xa076_1d64_78bd_642f);
        let t: u128 = (self.seed as u128).wrapping_mul((self.seed ^ 0xe703_7ed1_a0b4_28db) as u128);
        (t.wrapping_shr(64) ^ t) as u64
    }
    pub fn range(&mut self, range: Range<u64>) -> u64 {
        let mut val = self.gen();
        val %= range.end - range.start;
        val + range.start
    }
    pub fn shuffle<T>(&mut self, target: &mut [T]) {
        for i in 0..target.len() - 1 {
            target.swap(i, self.range(i as u64..target.len() as u64) as usize);
        }
    }
}

/// Retries the condition n times and returns if it was successfull.
/// This pauses the CPU between retries if possible.
#[inline(always)]
pub fn spin_wait(n: usize, mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..n {
        if cond() {
            return true;
        }
        core::hint::spin_loop()
    }
    false
}

pub fn avg_bounds<T>(iter: impl IntoIterator<Item = T>) -> Option<(T, T, T)>
where
    T: Ord + Add<T, Output = T> + Div<T, Output = T> + TryFrom<usize> + Copy,
{
    let mut iter = iter.into_iter();
    if let Some(first) = iter.next() {
        let mut min = first;
        let mut max = first;
        let mut mean = first;
        let mut count = 1;

        for x in iter {
            min = min.min(x);
            max = max.max(x);
            mean = mean + x;
            count += 1;
        }
        let count = match T::try_from(count) {
            Ok(c) => c,
            Err(_) => unreachable!("overflow"),
        };

        Some((min, mean / count, max))
    } else {
        None
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use super::{align_down, align_up, WyRand};

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
