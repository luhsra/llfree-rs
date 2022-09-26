use core::mem::transmute;
use core::mem::{align_of, size_of};
use core::ops::Range;

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

pub const fn div_ceil(v: usize, d: usize) -> usize {
    (v + d - 1) / d
}

pub const fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

pub const fn align_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
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

/// Prevents the compiler from optimizing `dummy` away.
#[inline(always)]
pub fn black_box<T>(dummy: T) -> T {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!("", in("ax") &dummy, options(nostack));
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!("", in("x0") &dummy, options(nostack));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    compile_error!("Unsupported architecture!");
    dummy
}

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        /// Executes CLWB (cache-line write back) for the given address.
        ///
        /// # Safety
        /// Directly executes an asm instruction.
        #[inline(always)]
        pub unsafe fn _mm_clwb<T>(addr: *const T) {
            use core::arch::asm;
            asm!("clwb [rax]", in("rax") addr);
        }

        /// Executes RDTSC (read time-stamp counter) and returns the current cycle count.
        ///
        /// # Safety
        /// Directly executes an asm instruction.
        #[inline(always)]
        unsafe fn time_stamp_counter() -> u64 {
            use core::arch::asm;

            let mut lo: u32;
            let mut hi: u32;
            asm!("rdtsc", out("eax") lo, out("edx") hi);
            lo as u64 | (hi as u64) << 32
        }

        /// x86 cycle timer.
        #[derive(Debug, Clone, Copy)]
        #[repr(transparent)]
        pub struct Cycles(u64);

        impl Cycles {
            pub fn now() -> Self {
                Self(unsafe { time_stamp_counter() })
            }
            pub fn elapsed(self) -> u64 {
                unsafe { time_stamp_counter() }.wrapping_sub(self.0)
            }
        }
    } else if #[cfg(feature = "std")] {
        /// Fallback to default ns timer.
        #[derive(Debug, Clone, Copy)]
        pub struct Cycles(std::time::Instant);

        impl Cycles {
            pub fn now() -> Self {
                Self(std::time::Instant::now())
            }
            pub fn elapsed(self) -> u64 {
                self.0.elapsed().as_nanos() as _
            }
        }
    }
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

#[cfg(all(test, feature = "std"))]
mod test {
    use super::{Cycles, WyRand};

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn clwb() {
        use alloc::boxed::Box;

        let mut data = Box::new(43_u64);
        *data = 44;
        unsafe { super::_mm_clwb(data.as_ref() as *const _ as _) };
        assert!(*data == 44);
    }

    #[test]
    fn cycles() {
        let cycles = Cycles::now();
        println!("waiting...");
        println!("cycles {}", cycles.elapsed());
    }

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
}
