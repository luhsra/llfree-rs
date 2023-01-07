use core::iter::FromIterator;
use core::{fmt, slice};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::Barrier;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use log::warn;

use nvalloc::lower::*;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::upper::*;
use nvalloc::util::{Page, WyRand};
use nvalloc::{thread, util};

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
    #[arg(short = 's', long, default_value_t = 0)]
    order: usize,
    /// Max amount of memory in GiB.
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
    /// Use every n-th cpu.
    #[arg(long, default_value_t = 1)]
    stride: usize,
    /// Write into every page before benchmarking.
    #[arg(long)]
    warmup: bool,
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
        warmup,
    } = Args::parse();

    util::logging();

    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    let ppt = (memory * PT_LEN * PT_LEN) / threads;
    assert!(ppt >= MIN_PAGES, "{ppt} > {MIN_PAGES}");

    let mut out = File::create(outfile).unwrap();
    writeln!(out, "alloc,x,iteration,memory,{}", Perf::header()).unwrap();

    warn!("Allocating order {order}");

    let mut mapping = mapping(0x1000_0000_0000, memory * PT_LEN * PT_LEN, dax).unwrap();

    if warmup {
        for page in &mut mapping[..] {
            *page.cast_mut::<usize>() = 1;
        }
    }

    let alloc_names: HashSet<String> = HashSet::from_iter(allocs.into_iter());

    type C32 = Cache<32>;
    let mut allocs: [Box<dyn Alloc>; 6] = [
        Box::<Array<0, C32>>::default(),
        Box::<Array<3, C32>>::default(),
        Box::<ArrayList<0, C32>>::default(),
        Box::<ArrayList<3, C32>>::default(),
        Box::<ListLocal>::default(),
        Box::<ListLocked>::default(),
    ];

    // Additional constraints (perf)
    let mut conditions = HashMap::<AllocName, &'static dyn Fn(usize, usize) -> bool>::new();
    conditions.insert(AllocName::new::<ListLocal>(), &|_, order| order == 0);
    conditions.insert(AllocName::new::<ListLocked>(), &|_, order| {
        order == 0 //&& (cores <= 16 || cores == 32)
    });

    for x in x {
        let t = bench.threads(threads, x);
        if t > threads {
            continue;
        }

        for alloc in &mut allocs {
            let name = alloc.name();
            if alloc_names.contains(&format!("{name}"))
                && conditions.get(&name).map(|f| f(t, order)).unwrap_or(true)
            {
                for i in 0..iterations {
                    let perf = bench.run(alloc.as_mut(), &mut mapping, order, threads, x);
                    writeln!(out, "{name},{x},{i},{memory},{perf}").unwrap();
                }
            }
        }
    }
    warn!("Ok");
    drop(allocs); // drop first
}

#[allow(unused_variables)]
fn mapping(
    begin: usize,
    length: usize,
    dax: Option<String>,
) -> core::result::Result<MMap<Page>, ()> {
    #[cfg(target_os = "linux")]
    if length > 0 {
        if let Some(file) = dax {
            warn!(
                "MMap file {file} l={}G ({:x})",
                (length * std::mem::size_of::<Page>()) >> 30,
                length * std::mem::size_of::<Page>()
            );
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(file)
                .unwrap();
            return MMap::dax(begin, length, f);
        }
    }
    MMap::anon(begin, length, false)
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
    fn threads(self, threads: usize, x: usize) -> usize {
        match self {
            Benchmark::Filling => threads,
            _ => x,
        }
    }

    fn run(
        self,
        alloc: &mut dyn Alloc,
        mapping: &mut [Page],
        order: usize,
        threads: usize,
        x: usize,
    ) -> Perf {
        warn!(">>> bench {self:?} x={x} o={order} {}\n", alloc.name());

        match self {
            Benchmark::Bulk => bulk(alloc, mapping, order, threads, x),
            Benchmark::Repeat => repeat(alloc, mapping, order, threads, x),
            Benchmark::Rand => rand(alloc, mapping, order, threads, x),
            Benchmark::RandBlock => rand_block(alloc, mapping, order, threads, x),
            Benchmark::Filling => filling(alloc, mapping, order, threads, x),
        }
    }
}

fn bulk(
    alloc: &mut dyn Alloc,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    alloc.init(threads, mapping, Init::Overwrite, true).unwrap();
    let init = timer.elapsed().as_millis();
    let allocs = alloc.pages() / max_threads / 2 / (1 << order);

    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        thread::pin(t);
        barrier.wait();
        let mut pages = Vec::with_capacity(allocs);
        let t1 = Instant::now();
        for _ in 0..allocs {
            pages.push(alloc.get(t, order).unwrap());
        }
        let get = t1.elapsed().as_nanos() / allocs as u128;
        pages.reverse();
        let pages = black_box(pages);

        barrier.wait();
        let t2 = Instant::now();
        for page in pages {
            alloc.put(t, page, order).unwrap();
        }
        let put = t2.elapsed().as_nanos() / allocs as u128;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: t1.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.dbg_allocated_pages(), 0, "{alloc:?}");

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn repeat(
    alloc: &mut dyn Alloc,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    alloc.init(threads, mapping, Init::Overwrite, true).unwrap();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.pages() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        thread::pin(t);
        for _ in 0..allocs {
            alloc.get(t, order).unwrap();
        }

        barrier.wait();
        let timer = Instant::now();
        for _ in 0..allocs {
            let page = alloc.get(t, order).unwrap();
            let page = black_box(page);
            alloc.put(t, page, order).unwrap();
        }

        let realloc = timer.elapsed().as_nanos() / allocs as u128;
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
    assert_eq!(alloc.dbg_allocated_pages(), allocs * threads * (1 << order));

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn rand(
    alloc: &mut dyn Alloc,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    alloc.init(threads, mapping, Init::Overwrite, true).unwrap();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.pages() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);

    let mut all_pages = vec![0u64; allocs * threads];
    let all_pages_ptr = all_pages.as_mut_ptr() as usize;

    let chunks = all_pages.chunks_mut(allocs).enumerate();
    let mut perf = Perf::avg(thread::parallel(chunks, |(t, pages)| {
        thread::pin(t);
        barrier.wait();

        let timer = Instant::now();
        for page in pages.iter_mut() {
            *page = alloc.get(t, order).unwrap();
        }
        let get = timer.elapsed().as_nanos() / allocs as u128;

        barrier.wait();
        if t == 0 {
            assert_eq!(pages.len(), allocs);
            assert_eq!(alloc.dbg_allocated_pages(), allocs * threads * (1 << order));
            // Shuffle between all cores
            let pages =
                unsafe { slice::from_raw_parts_mut(all_pages_ptr as *mut u64, allocs * threads) };
            WyRand::new(42).shuffle(pages);
        }
        barrier.wait();

        let timer = Instant::now();
        for page in pages {
            alloc.put(t, *page, order).unwrap();
        }
        let put = timer.elapsed().as_nanos() / allocs as u128;

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
    assert_eq!(alloc.dbg_allocated_pages(), 0);

    perf.init = init;
    perf.allocs = allocs;
    perf
}

/// reallocate multiple in close proximity at once
fn rand_block(
    alloc: &mut dyn Alloc,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    alloc.init(threads, mapping, Init::Overwrite, true).unwrap();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.pages() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);

    let mut all_pages = vec![0u64; allocs * threads];
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
    assert_eq!(alloc.dbg_allocated_pages(), 0);

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn filling(
    alloc: &mut dyn Alloc,
    mapping: &mut [Page],
    order: usize,
    threads: usize,
    x: usize,
) -> Perf {
    assert!(x <= 90);

    let timer = Instant::now();
    alloc.init(threads, mapping, Init::Overwrite, true).unwrap();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.pages() / threads / (1 << order);

    // Allocate to filling level
    let fill = (allocs * x) / 100;
    let allocs = allocs / 10;
    warn!("fill={fill} allocs={allocs}");

    assert!((fill + allocs) * threads < alloc.pages());

    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        thread::pin(t);
        for _ in 0..fill {
            alloc.get(t, order).unwrap();
        }
        barrier.wait();

        let mut pages = Vec::with_capacity(allocs);
        // Operate on filling level.
        let t1 = Instant::now();
        for _ in 0..allocs {
            let Ok(page) = alloc.get(t, order) else {
                break;
            };
            pages.push(page);
        }
        let num_alloc = pages.len();
        let get = t1.elapsed().as_nanos() / num_alloc as u128;

        if num_alloc < allocs {
            warn!("Allocator completely full {num_alloc}");
        }

        pages.reverse();
        let pages = black_box(pages);

        barrier.wait();
        let t2 = Instant::now();
        for page in pages {
            alloc.put(t, page, order).unwrap();
        }
        let put = t2.elapsed().as_nanos() / num_alloc as u128;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: t1.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.dbg_allocated_pages(), fill * threads * (1 << order));

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
