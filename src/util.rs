use core::alloc::Layout;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ops::{Index, Range};
use core::sync::atomic::{AtomicU64, Ordering};

use log::error;

use crate::entry::Entry3;

/// Correctly sized and aligned page.
#[derive(Clone)]
#[repr(align(0x1000))]
pub struct Page {
    _data: [u8; Page::SIZE],
}
const _: () = assert!(Layout::new::<Page>().size() == Page::SIZE);
const _: () = assert!(Layout::new::<Page>().align() == Page::SIZE);
impl Page {
    pub const SIZE_BITS: usize = 12; // 2^12 => 4KiB
    pub const SIZE: usize = 1 << Page::SIZE_BITS;
    pub const fn new() -> Self {
        Self {
            _data: [0; Page::SIZE],
        }
    }
    pub fn cast<T>(&self) -> &T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { std::mem::transmute(self) }
    }
    pub fn cast_mut<T>(&mut self) -> &mut T {
        debug_assert!(size_of::<T>() <= size_of::<Self>());
        unsafe { std::mem::transmute(self) }
    }
}

pub const fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

pub const fn align_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
}

/// Wrapper for 64bit atomic values.
#[repr(transparent)]
pub struct Atomic<T: From<u64> + Into<u64>>(pub AtomicU64, PhantomData<T>);

const _: () = assert!(std::mem::size_of::<Atomic<u64>>() == 8);

impl<T: From<u64> + Into<u64>> Atomic<T> {
    #[inline]
    pub fn new(v: T) -> Self {
        Self(AtomicU64::new(v.into()), PhantomData)
    }
    #[inline]
    pub fn compare_exchange(&self, current: T, new: T) -> Result<T, T> {
        match self.0.compare_exchange(
            current.into(),
            new.into(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    #[inline]
    pub fn update<F: FnMut(T) -> Option<T>>(&self, mut f: F) -> Result<T, T> {
        match self
            .0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                f(v.into()).map(T::into)
            }) {
            Ok(v) => Ok(v.into()),
            Err(v) => Err(v.into()),
        }
    }
    #[inline]
    pub fn load(&self) -> T {
        self.0.load(Ordering::SeqCst).into()
    }
    #[inline]
    pub fn store(&self, v: T) {
        self.0.store(v.into(), Ordering::SeqCst)
    }
    #[inline]
    pub fn swap(&self, v: T) -> T {
        self.0.swap(v.into(), Ordering::SeqCst).into()
    }
}

#[cfg(all(feature = "thread", any(test, feature = "logger")))]
fn core() -> usize {
    use crate::thread::PINNED;
    PINNED.with(|p| p.load(Ordering::SeqCst))
}
#[cfg(all(not(feature = "thread"), any(test, feature = "logger")))]
fn core() -> usize {
    0
}

#[cfg(any(test, feature = "logger"))]
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
                unsafe { std::mem::transmute::<ThreadId, u64>(std::thread::current().id()) },
                core(),
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default(),
                record.args()
            )
        })
        .try_init();
}

/// Executes CLWB (cache-line write back) for the given address.
///
/// # Safety
/// Directly executes an asm instruction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn _mm_clwb(addr: *const ()) {
    use std::arch::asm;

    asm!("clwb [rax]", in("rax") addr);
}

/// Executes RDTSC (read time-stamp counter) and returns the current cycle count.
///
/// # Safety
/// Directly executes an asm instruction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn time_stamp_counter() -> u64 {
    use std::arch::asm;

    let mut lo: u32;
    let mut hi: u32;
    asm!("rdtsc", out("eax") lo, out("edx") hi);
    lo as u64 | (hi as u64) << 32
}

/// x86 cycle timer.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Cycles(u64);

#[cfg(target_arch = "x86_64")]
impl Cycles {
    pub fn now() -> Self {
        Self(unsafe { time_stamp_counter() })
    }
    pub fn elapsed(self) -> u64 {
        unsafe { time_stamp_counter() }.wrapping_sub(self.0)
    }
}

/// Fallback to default ns timer.
#[cfg(not(target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy)]
pub struct Cycles(std::time::Instant);

#[cfg(not(target_arch = "x86_64"))]
impl Cycles {
    pub fn now() -> Self {
        Self(std::time::Instant::now())
    }
    pub fn elapsed(self) -> u64 {
        self.0.elapsed().as_nanos() as _
    }
}

/// Node of an atomic stack
pub trait ANode: Copy + From<u64> + Into<u64> {
    fn next(self) -> Option<usize>;
    fn with_next(self, next: Option<usize>) -> Self;
}

impl ANode for Entry3 {
    fn next(self) -> Option<usize> {
        match self.idx() {
            Entry3::IDX_MAX => None,
            v => Some(v),
        }
    }

    fn with_next(self, next: Option<usize>) -> Self {
        self.with_idx(next.unwrap_or(Entry3::IDX_MAX))
    }
}

/// Simple atomic stack with atomic entries.
pub struct AStack<T: ANode> {
    start: Atomic<T>,
}

unsafe impl<T: ANode> Send for AStack<T> {}
unsafe impl<T: ANode> Sync for AStack<T> {}

impl<T: ANode> AStack<T> {
    pub fn new() -> Self {
        Self {
            start: Atomic::new(T::from(0).with_next(None)),
        }
    }
    pub fn push<B>(&self, buf: &B, idx: usize)
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        let mut start = self.start.load();
        let elem = &buf[idx];
        loop {
            if elem.update(|v| Some(v.with_next(start.next()))).is_err() {
                panic!();
            }
            match self
                .start
                .compare_exchange(start, start.with_next(Some(idx)))
            {
                Ok(_) => return,
                Err(s) => start = s,
            }
        }
    }
    pub fn pop<B>(&self, buf: &B) -> Option<usize>
    where
        B: Index<usize, Output = Atomic<T>>,
    {
        let mut start = self.start.load();
        loop {
            let idx = start.next()?;
            let next = buf[idx].load().next();
            match self.start.compare_exchange(start, start.with_next(next)) {
                Ok(old) => {
                    let i = old.next()?;
                    let _ = buf[i].update(|v| Some(v.with_next(None)));
                    return Some(i);
                }
                Err(s) => start = s,
            }
        }
    }
}
#[allow(dead_code)]
pub struct AStackDbg<'a, T, B>(pub &'a AStack<T>, pub &'a B)
where
    T: ANode,
    B: Index<usize, Output = Atomic<T>>;

impl<'a, T, B> Debug for AStackDbg<'a, T, B>
where
    T: ANode,
    B: Index<usize, Output = Atomic<T>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_list();

        match self.0.start.load().next() {
            None => {}
            Some(i) => {
                let mut i = i as usize;
                let mut ended = false;
                for _ in 0..1000 {
                    dbg.entry(&i);
                    let elem = self.1[i].load();
                    if let Some(next) = elem.next() {
                        i = next;
                    } else {
                        ended = true;
                        break;
                    }
                }
                if !ended {
                    error!("Circular List!");
                }
            }
        }

        dbg.finish()
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
        self.seed = self.seed.wrapping_add(0xa0761d6478bd642f);
        let t: u128 = (self.seed as u128).wrapping_mul((self.seed ^ 0xe7037ed1a0b428db) as u128);
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

#[cfg(test)]
mod test {
    use std::marker::PhantomData;
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Barrier};

    use super::{ANode, AStack, AStackDbg, Cycles, WyRand};
    use crate::thread;
    use crate::{thread::parallel, util::Atomic};

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn clwb() {
        let mut data = Box::new(43_u64);
        *data = 44;
        unsafe { super::_mm_clwb(data.as_ref() as *const _ as _) };
    }

    #[test]
    fn cycles() {
        let cycles = Cycles::now();
        println!("waiting...");
        println!("cycles {}", cycles.elapsed());
    }

    const DATA_V: Atomic<u64> = Atomic(AtomicU64::new(0), PhantomData);
    static mut DATA: [Atomic<u64>; 16] = [DATA_V; 16];

    #[test]
    fn atomic_stack() {
        impl ANode for u64 {
            fn next(self) -> Option<usize> {
                (self != u64::MAX).then(|| self as _)
            }
            fn with_next(self, next: Option<usize>) -> Self {
                next.map(|v| v as u64).unwrap_or(u64::MAX)
            }
        }

        let stack = AStack::new();
        stack.push(unsafe { &DATA }, 0);
        stack.push(unsafe { &DATA }, 1);

        println!("{:?}", AStackDbg(&stack, unsafe { &DATA }));

        assert_eq!(stack.pop(unsafe { &DATA }), Some(1));
        assert_eq!(stack.pop(unsafe { &DATA }), Some(0));
        assert_eq!(stack.pop(unsafe { &DATA }), None);

        // Stress test
        let barrier = Arc::new(Barrier::new(4));
        let stack = Arc::new(stack);
        let copy = stack.clone();

        parallel(4, move |t| {
            for _ in 0..100 {
                thread::pin(t);

                barrier.wait();
                for i in 0..4 {
                    stack.push(unsafe { &DATA }, t * 4 + i);
                }
                for _ in 0..4 {
                    stack.pop(unsafe { &DATA }).unwrap();
                }
            }
        });
        assert_eq!(copy.pop(unsafe { &DATA }), None);
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
