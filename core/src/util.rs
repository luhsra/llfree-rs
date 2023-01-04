use core::mem::transmute;
use core::mem::{align_of, size_of};
use core::ops::{Add, Deref, DerefMut, Div, Range};
use core::{fmt, mem};

/// Correctly sized and aligned page.
#[derive(Clone)]
#[repr(align(0x1000))]
pub struct Page {
    _data: [u8; Self::SIZE],
}
const _: () = assert!(size_of::<Page>() == Page::SIZE);
const _: () = assert!(align_of::<Page>() == Page::SIZE);
impl Page {
    pub const SIZE: usize = 0x1000;
    pub const SIZE_BITS: usize = Self::SIZE.ilog2() as _;
    pub const fn new() -> Self {
        Self {
            _data: [0; Self::SIZE],
        }
    }
    pub fn cast<T>(&self) -> &T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { transmute(self) }
    }
    pub fn cast_mut<T>(&mut self) -> &mut T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { transmute(self) }
    }
}

/// Correctly sized and aligned cache line
#[derive(Clone)]
#[repr(align(64))]
pub struct CacheLine {
    _data: [u8; Self::SIZE],
}
const _: () = assert!(size_of::<CacheLine>() == CacheLine::SIZE);
const _: () = assert!(align_of::<CacheLine>() == CacheLine::SIZE);

impl CacheLine {
    pub const SIZE: usize = 64;
    pub const SIZE_BITS: usize = Self::SIZE.ilog2() as _;
    pub const fn new() -> Self {
        Self {
            _data: [0; Self::SIZE],
        }
    }
    pub fn cast<T>(&self) -> &T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { transmute(self) }
    }
    pub fn cast_mut<T>(&mut self) -> &mut T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { transmute(self) }
    }
}

#[inline(always)]
pub const fn align_up(v: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (v + align - 1) & !(align - 1)
}

#[inline(always)]
pub const fn align_down(v: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    v & !(align - 1)
}

/// Cache alignment for T
#[derive(Clone, Default, Hash, PartialEq, Eq)]
#[repr(align(64))]
pub struct CacheAlign<T>(pub T);

const _: () = assert!(align_of::<CacheAlign<usize>>() == CacheLine::SIZE);

impl<T> Deref for CacheAlign<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for CacheAlign<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for CacheAlign<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T> From<T> for CacheAlign<T> {
    fn from(t: T) -> Self {
        CacheAlign(t)
    }
}

/// Simple ring buffer implementation
#[derive(Debug)]
#[repr(align(64))]
pub struct RingBuffer<T, const N: usize> {
    buf: [T; N],
    idx: usize,
}
impl<T: Copy, const N: usize> RingBuffer<T, N> {
    pub const fn new(value: T) -> Self {
        debug_assert!(N > 0);
        Self {
            buf: [value; N],
            idx: 0,
        }
    }
    pub const fn len() -> usize {
        N
    }
    pub fn push(&mut self, value: T) {
        if N == 1 {
            self.buf[0] = value;
        } else {
            self.buf[self.idx] = value;
            self.idx = (self.idx + 1) % N;
        }
    }
}
impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    pub fn pop(&mut self) -> T {
        let value = mem::take(&mut self.buf[self.idx]);
        self.idx = (N + self.idx - 1) % N;
        value
    }
}
impl<T: Copy + Default, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new(T::default())
    }
}
impl<T: Copy + PartialEq, const N: usize> RingBuffer<T, N> {
    pub fn all_eq(&self, value: T) -> bool {
        self.buf.iter().all(|v| *v == value)
    }
}

#[cfg(feature = "std")]
fn core() -> usize {
    use crate::thread::PINNED;
    use core::sync::atomic::Ordering;
    PINNED.with(|p| p.load(Ordering::SeqCst))
}

#[cfg(feature = "std")]
pub fn logging() {
    use std::io::Write;
    use std::thread::ThreadId;

    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format(move |buf, record| {
            let color = match record.level() {
                log::Level::Error => "\x1b[91m",
                log::Level::Warn => "\x1b[93m",
                log::Level::Info => "\x1b[90m",
                log::Level::Debug => "\x1b[90m",
                log::Level::Trace => "\x1b[90m",
            };

            writeln!(
                buf,
                "{}[{:5} {:02?}@{:02?} {}:{}] {}\x1b[0m",
                color,
                record.level(),
                unsafe { transmute::<ThreadId, u64>(std::thread::current().id()) },
                core(),
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default(),
                record.args()
            )
        })
        .try_init();
}

/// Simple bare bones random number generator based on wyhash.
///
/// @see https://github.com/wangyi-fudan/wyhash
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
        for i in 0..target.len() {
            target.swap(i, self.range(0..target.len() as u64) as usize);
        }
    }
}

/// Retries the condition n times and returns if it was successfull.
/// This pauses the CPU between retries if possible.
#[inline(always)]
pub fn spin_wait<F: FnMut() -> bool>(n: usize, mut cond: F) -> bool {
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
