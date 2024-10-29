#![feature(allocator_api)]

use core::{fmt, slice};
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::Barrier;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use llfree::frame::Frame;
use llfree::mmap::{self, MMap};
use llfree::util::{self, aligned_buf, WyRand};
use llfree::wrapper::NvmAlloc;
#[cfg(feature = "llc")]
use llfree::LLC;
use llfree::{thread, Alloc, Flags, LLFree, Result, MAX_ORDER};
use log::warn;

/// Number of allocations per block
const RAND_BLOCK_SIZE: usize = 8;

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    #[arg(value_enum)]
    bench: Benchmark,
    /// Names of the allocators to be benchmarked.
    allocs: Vec<String>,
    /// Tested number of threads / allocations / filling levels, depending on benchmark.
    #[arg(short, long, default_value = "1")]
    x: Vec<usize>,
    /// Max number of threads.
    #[arg(short, long, default_value = "6")]
    threads: usize,
    /// Where to store the benchmark results in csv format.
    #[arg(short, long, default_value = "bench/out/bench.csv")]
    outfile: String,
    /// DAX file to be used for the allocator.
    #[arg(long)]
    dax: Option<String>,
    /// Number of repetitions.
    #[arg(short, long, default_value_t = 1)]
    iterations: usize,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[arg(short = 's', long, default_value = "0")]
    order: Vec<usize>,
    /// Max amount of memory in GiB.
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
    /// Use every n-th cpu.
    #[arg(long, default_value_t = 1)]
    stride: usize,

    #[arg(long, default_value_t = 0)]
    offset: usize,
}

fn main() {
    let Args {
        bench,
        allocs,
        x,
        threads,
        outfile,
        dax,
        iterations,
        order,
        memory,
        stride,
        offset,
    } = Args::parse();

    util::logging();

    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    if offset > 0 {
        thread::OFFSET.store(offset, Ordering::Relaxed);
    }

    assert!(memory >= 1);

    let mut out = File::create(outfile).unwrap();
    writeln!(out, "alloc,x,order,iteration,memory,{}", Perf::header()).unwrap();

    warn!("Allocating orders {order:?}");

    let mut mapping = mapping(0x1000_0000_0000, (memory << 30) / Frame::SIZE, dax);

    for x in x {
        for o in order.iter().copied() {
            assert!(o <= MAX_ORDER);
            for name in &allocs {
                for i in 0..iterations {
                    let perf = bench.run(name, &mut mapping, o, threads, x);
                    writeln!(out, "{name},{x},{o},{i},{memory},{perf}").unwrap();
                }
            }
        }
    }
    warn!("Ok");
    drop(allocs); // drop first
}

#[allow(unused_variables)]
pub fn mapping(begin: usize, length: usize, dax: Option<String>) -> Box<[Frame], MMap> {
    #[cfg(target_os = "linux")]
    if let Some(file) = dax {
        warn!("MMap file {file} l={}G", (length * Frame::SIZE) >> 30);
        return mmap::file(begin, length, &file, true);
    }
    warn!("Mapping {length} frames ({} bytes).", length * Frame::SIZE);
    mmap::anon(begin, length, false, false)
}

/// Reduced, VTable-compatible alloc trait for dynamic dispatch
trait DynAlloc: fmt::Debug + Send + Sync {
    fn get(&self, core: usize, order: usize) -> Result<usize>;
    fn put(&self, core: usize, frame: usize, order: usize) -> Result<()>;

    fn frames(&self) -> usize;
    fn allocated_frames(&self) -> usize;
}

impl<'a, T: Alloc<'a>> DynAlloc for NvmAlloc<'a, T> {
    fn get(&self, core: usize, order: usize) -> Result<usize> {
        Alloc::get(self, core, Flags::o(order))
    }
    fn put(&self, core: usize, frame: usize, order: usize) -> Result<()> {
        Alloc::put(self, core, frame, Flags::o(order))
    }
    fn frames(&self) -> usize {
        Alloc::frames(self)
    }
    fn allocated_frames(&self) -> usize {
        Alloc::allocated_frames(self)
    }
}

fn alloc<'a>(name: &str, cores: usize, zone: &'a mut [Frame]) -> Box<dyn DynAlloc + 'a> {
    #[cfg(feature = "llc")]
    if LLC::name() == name {
        let m = NvmAlloc::<LLC>::metadata_size(cores, zone.len());
        let local = aligned_buf(m.local).leak();
        let trees = aligned_buf(m.trees).leak();
        return Box::new(NvmAlloc::<LLC>::create(cores, zone, false, local, trees).unwrap());
    }
    if LLFree::name() == name {
        let m = NvmAlloc::<LLFree>::metadata_size(cores, zone.len());
        let local = aligned_buf(m.local).leak();
        let trees = aligned_buf(m.trees).leak();
        return Box::new(NvmAlloc::<LLFree>::create(cores, zone, false, local, trees).unwrap());
    }
    panic!("Unknown allocator");
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Benchmark {
    /// Allocate half the memory at once and free it afterwards
    Bulk,
    /// Initially allocate half the memory and then repeatedly allocate and free the same page
    Repeat,
    /// Initially allocate half the memory and then repeatedly free an random page
    /// and replace it with a newly allocated one
    Rand,
    /// Like `rand`, but reallocating multiple pages in close proximity
    RandBlock,
    /// Compute times for different filling levels
    Filling,
}

impl Benchmark {
    fn run(
        self,
        name: &str,
        mapping: &mut [Frame],
        order: usize,
        threads: usize,
        x: usize,
    ) -> Perf {
        warn!(">>> bench {self:?} x={x} o={order} {name}\n");
        let mut alloc = alloc(name, threads, mapping);

        match self {
            Benchmark::Bulk => bulk(alloc.as_mut(), order, threads, x),
            Benchmark::Repeat => repeat(alloc.as_mut(), order, threads, x),
            Benchmark::Rand => rand(alloc.as_mut(), order, threads, x),
            Benchmark::RandBlock => rand_block(alloc.as_mut(), order, threads, x),
            Benchmark::Filling => filling(alloc.as_mut(), order, threads, x),
        }
    }
}

fn bulk(alloc: &mut dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();
    let allocs = alloc.frames() / max_threads / 2 / (1 << order);

    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        let mut get = 0;
        let mut put = 0;

        let total = Instant::now();

        let mut pages = Vec::with_capacity(allocs);

        for _ in 0..(1 << order) {
            thread::pin(t);

            barrier.wait();
            let timer = Instant::now();
            for _ in 0..allocs {
                pages.push(alloc.get(t, order).unwrap());
            }
            get += timer.elapsed().as_nanos() / allocs as u128;

            barrier.wait();
            let timer = Instant::now();
            while let Some(page) = pages.pop() {
                alloc.put(t, page, order).unwrap();
            }
            put += timer.elapsed().as_nanos() / allocs as u128;
        }
        get /= 1 << order;
        put /= 1 << order;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: total.elapsed().as_millis(),
            allocs: allocs * (1 << order),
        }
    }));
    assert_eq!(alloc.allocated_frames(), 0, "{alloc:?}");

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn repeat(alloc: &mut dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.frames() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        thread::pin(t);
        for _ in 0..allocs {
            alloc.get(t, order).unwrap();
        }

        barrier.wait();
        let timer = Instant::now();
        for _ in 0..(1 << order) {
            for _ in 0..allocs {
                let page = alloc.get(t, order).unwrap();
                let page = black_box(page);
                alloc.put(t, page, order).unwrap();
            }
        }

        let realloc = timer.elapsed().as_nanos() / (allocs * (1 << order)) as u128;
        Perf {
            get_min: realloc,
            get_avg: realloc,
            get_max: realloc,
            put_min: realloc,
            put_avg: realloc,
            put_max: realloc,
            init: 0,
            total: timer.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.allocated_frames(), allocs * threads * (1 << order));

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn rand(alloc: &mut dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.frames() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);

    let mut all_pages = vec![0; allocs * threads];
    let all_pages_ptr = all_pages.as_mut_ptr() as usize;

    let chunks = all_pages.chunks_mut(allocs).enumerate();
    let mut perf = Perf::avg(thread::parallel(chunks, |(t, pages)| {
        thread::pin(t);

        let timer = Instant::now();
        let mut get = 0;
        let mut put = 0;
        for _ in 0..(1 << order) {
            barrier.wait();
            let timer = Instant::now();
            for page in pages.iter_mut() {
                *page = alloc.get(t, order).unwrap();
            }
            get += timer.elapsed().as_nanos() / allocs as u128;

            barrier.wait();
            if t == 0 {
                assert_eq!(pages.len(), allocs);
                assert_eq!(alloc.allocated_frames(), allocs * threads * (1 << order));
                // Shuffle between all cores
                let pages = unsafe {
                    slice::from_raw_parts_mut(all_pages_ptr as *mut u64, allocs * threads)
                };
                WyRand::new(42).shuffle(pages);
            }
            barrier.wait();

            let timer = Instant::now();
            for page in pages.iter() {
                alloc.put(t, *page, order).unwrap();
            }
            put += timer.elapsed().as_nanos() / allocs as u128;
        }
        get /= 1 << order;
        put /= 1 << order;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: timer.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.allocated_frames(), 0);

    perf.init = init;
    perf.allocs = allocs;
    perf
}

/// reallocate multiple in close proximity at once
fn rand_block(alloc: &mut dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.frames() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);

    let mut all_pages = vec![0; allocs * threads];
    let all_pages_ptr = all_pages.as_mut_ptr() as usize;

    let chunks = all_pages.chunks_mut(allocs).enumerate();
    let mut perf = Perf::avg(thread::parallel(chunks, |(t, pages)| {
        thread::pin(t);

        for page in pages.iter_mut() {
            *page = alloc.get(t, order).unwrap();
        }

        barrier.wait();
        if t == 0 {
            let blocks = allocs / RAND_BLOCK_SIZE;
            assert!(blocks > 0);
            // Shuffle blocks between all cores
            let mut rng = WyRand::new(t as _);
            let pages =
                unsafe { slice::from_raw_parts_mut(all_pages_ptr as *mut u64, allocs * threads) };
            for _ in 0..blocks {
                let i = (rng.range(0..blocks as u64) as usize) * RAND_BLOCK_SIZE;
                let j = (rng.range(0..blocks as u64) as usize) * RAND_BLOCK_SIZE;
                if i != j {
                    for k in 0..RAND_BLOCK_SIZE {
                        pages.swap(i + k, j + k);
                    }
                }
            }
        }
        barrier.wait();

        let timer = Instant::now();
        for page in pages {
            alloc.put(t, *page, order).unwrap();
        }
        let put = timer.elapsed().as_nanos() / allocs as u128;

        Perf {
            get_min: 0,
            get_avg: 0,
            get_max: 0,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: timer.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.allocated_frames(), 0);

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn filling(alloc: &mut dyn DynAlloc, order: usize, threads: usize, level: usize) -> Perf {
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.frames() / threads / (1 << order);

    // Allocate to filling level
    let fill = (allocs * level) / 100;
    let allocs = allocs / 100; // allocate 1%
    warn!("fill={fill} allocs={allocs}");

    assert!((fill + allocs) * threads < alloc.frames());

    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        thread::pin(t);
        for _ in 0..fill {
            alloc.get(t, order).unwrap();
        }
        barrier.wait();

        let mut pages = Vec::with_capacity(allocs);
        barrier.wait();

        let mut get = 0;
        let mut put = 0;
        const N: usize = 100;
        for _ in 0..N {
            // Operate on filling level.
            let timer = Instant::now();
            for _ in 0..allocs {
                let Ok(page) = alloc.get(t, order) else {
                    break;
                };
                pages.push(page);
            }
            let num_alloc = pages.len();
            get += timer.elapsed().as_nanos() / num_alloc as u128;

            if num_alloc < allocs {
                warn!("Allocator completely full {num_alloc}");
            }

            let timer = Instant::now();
            while let Some(page) = pages.pop() {
                alloc.put(t, page, order).unwrap();
            }
            put += timer.elapsed().as_nanos() / num_alloc as u128;
        }
        get /= N as u128;
        put /= N as u128;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: 0,
            allocs,
        }
    }));
    assert_eq!(alloc.allocated_frames(), fill * threads * (1 << order));

    perf.init = init;
    perf.allocs = allocs;
    perf
}

#[derive(Debug)]
struct Perf {
    get_min: u128,
    get_avg: u128,
    get_max: u128,
    put_min: u128,
    put_avg: u128,
    put_max: u128,
    init: u128,
    total: u128,
    allocs: usize,
}

impl Default for Perf {
    fn default() -> Self {
        Self {
            get_min: u128::MAX,
            get_avg: 0,
            get_max: 0,
            put_min: u128::MAX,
            put_avg: 0,
            put_max: 0,
            init: 0,
            total: 0,
            allocs: 0,
        }
    }
}

impl Perf {
    fn avg(iter: impl IntoIterator<Item = Perf>) -> Perf {
        let mut res = Perf::default();
        let mut counter = 0;
        for p in iter {
            res.get_min = res.get_min.min(p.get_min);
            res.get_avg += p.get_avg;
            res.get_max = res.get_max.max(p.get_max);
            res.put_min = res.put_min.min(p.put_min);
            res.put_avg += p.put_avg;
            res.put_max = res.put_max.max(p.put_max);
            res.init += p.init;
            res.total += p.total;
            res.allocs += p.allocs;
            counter += 1;
        }
        assert!(counter > 0);
        res.get_avg /= counter;
        res.put_avg /= counter;
        res.init /= counter;
        res.total /= counter;
        res.allocs /= counter as usize;
        res
    }
    fn header() -> &'static str {
        "get_min,get_avg,get_max,put_min,put_avg,put_max,init,total,allocs"
    }
}

impl fmt::Display for Perf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Perf {
            get_min,
            get_avg,
            get_max,
            put_min,
            put_avg,
            put_max,
            init,
            total,
            allocs,
        } = self;
        write!(
            f,
            "{get_min},{get_avg},{get_max},{put_min},{put_avg},{put_max},{init},{total},{allocs}"
        )
    }
}
