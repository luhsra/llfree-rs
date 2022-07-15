use core::iter::FromIterator;
use core::{fmt, slice};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use clap::{ArgEnum, Parser};
use log::warn;

use nvalloc::lower::*;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::upper::*;
use nvalloc::util::{black_box, Page, WyRand};
use nvalloc::{thread, util};

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(arg_enum)]
    bench: Benchmark,
    allocs: Vec<String>,
    /// Tested number of threads / allocations / filling levels, depending on benchmark.
    #[clap(short, long, default_value = "1")]
    x: Vec<usize>,
    /// Max number of threads
    #[clap(short, long, default_value = "6")]
    threads: usize,
    /// Where to store the benchmark results in csv format.
    #[clap(short, long, default_value = "bench/out/bench.csv")]
    outfile: String,
    /// DAX file to be used for the allocator.
    #[clap(long)]
    dax: Option<String>,
    #[clap(short, long, default_value_t = 1)]
    iterations: usize,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[clap(short = 's', long, default_value_t = 0)]
    order: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[clap(short, long, default_value_t = 16)]
    memory: usize,
    #[clap(long, default_value_t = 1)]
    stride: usize,
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

    // Warmup
    for page in &mut mapping[..] {
        *page.cast_mut::<usize>() = 1;
    }

    let alloc_names: HashSet<String> = HashSet::from_iter(allocs.into_iter());

    type C512 = CacheLower<512>;
    type C128 = CacheLower<128>;
    type C64 = CacheLower<64>;
    let allocs: [Arc<dyn Alloc>; 11] = [
        Arc::new(ArrayAlignedAlloc::<CacheAligned, C64>::default()),
        Arc::new(ArrayAlignedAlloc::<CacheAligned, C128>::default()),
        Arc::new(ArrayAlignedAlloc::<CacheAligned, C512>::default()),
        Arc::new(ArrayAtomicAlloc::<C64>::default()),
        Arc::new(ArrayAtomicAlloc::<C128>::default()),
        Arc::new(ArrayAtomicAlloc::<C512>::default()),
        Arc::new(TableAlloc::<C64>::default()),
        Arc::new(TableAlloc::<C128>::default()),
        Arc::new(TableAlloc::<C512>::default()),
        Arc::new(ListLocalAlloc::default()),
        Arc::new(ListLockedAlloc::default()),
    ];

    // Additional constraints
    let mut conditions = HashMap::<String, &'static dyn Fn(usize, usize) -> bool>::new();
    conditions.insert(name::<ListLocalAlloc>(), &|_, order| order == 0);
    conditions.insert(name::<ListLockedAlloc>(), &|cores, order| {
        order == 0 && (cores <= 16 || cores == 32)
    });

    for x in x {
        let t = bench.threads(threads, x);
        if t > threads {
            continue;
        }

        for alloc in &allocs {
            let name = alloc.name();
            if alloc_names.contains(&name)
                && conditions.get(&name).map(|f| f(t, order)).unwrap_or(true)
            {
                for i in 0..iterations {
                    let perf = bench.run(alloc.clone(), &mut mapping, order, threads, x);
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
    MMap::anon(begin, length)
}

#[derive(Debug, Clone, Copy, ArgEnum)]
enum Benchmark {
    /// Allocate half the memory at once and free it afterwards
    Bulk,
    /// Initially allocate half the memory and then repeatedly allocate and free the same page
    Repeat,
    /// Initially allocate half the memory and then repeatedly free an random page
    /// and replace it with a newly allocated one
    Rand,
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
        alloc: Arc<dyn Alloc>,
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
            Benchmark::Filling => filling(alloc, mapping, order, threads, x),
        }
    }
}

fn bulk(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, mapping, true)
        .unwrap();
    let init = timer.elapsed().as_millis();
    let allocs = alloc.pages() / max_threads / 2 / (1 << order);

    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
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
    assert_eq!(a.dbg_allocated_pages(), 0);

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn repeat(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, mapping, true)
        .unwrap();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.pages() / max_threads / 2 / (1 << order);
    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
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
    assert_eq!(a.dbg_allocated_pages(), allocs * threads * (1 << order));

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn rand(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    order: usize,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, mapping, true)
        .unwrap();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.pages() / max_threads / 2 / (1 << order);
    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();

    let mut all_pages = vec![0u64; allocs * threads];
    let all_pages_ptr = all_pages.as_mut_ptr() as usize;
    let pages_ptr = all_pages
        .chunks_mut(allocs)
        .map(|c| c.as_ptr() as usize)
        .collect::<Vec<_>>();

    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
        thread::pin(t);
        let pages = unsafe { slice::from_raw_parts_mut(pages_ptr[t] as *mut u64, allocs) };

        for page in pages.iter_mut() {
            *page = alloc.get(t, order).unwrap();
        }

        let mut rng = WyRand::new(t as _);

        barrier.wait();
        if t == 0 {
            // Shuffle between all cores
            let pages =
                unsafe { slice::from_raw_parts_mut(all_pages_ptr as *mut u64, allocs * threads) };
            rng.shuffle(pages);
        }
        barrier.wait();

        let timer = Instant::now();

        for _ in 0..allocs {
            let i = rng.range(0..allocs as u64) as usize;
            alloc.put(t, pages[i], order).unwrap();
            pages[i] = alloc.get(t, order).unwrap();
        }
        black_box(pages);

        let rand = timer.elapsed().as_nanos() / allocs as u128;
        Perf {
            get_min: rand,
            get_avg: rand,
            get_max: rand,
            put_min: rand,
            put_avg: rand,
            put_max: rand,
            init: 0,
            total: timer.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(a.dbg_allocated_pages(), allocs * threads * (1 << order));

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn filling(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    order: usize,
    threads: usize,
    x: usize,
) -> Perf {
    assert!(x <= 90);

    let timer = Instant::now();
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, mapping, true)
        .unwrap();
    let init = timer.elapsed().as_millis();
    let allocs = alloc.pages() / threads / (1 << order);

    // Allocate to filling level
    let fill = (allocs * x) / 100;
    let allocs = allocs / 10;
    warn!("fill={fill} allocs={allocs}");

    assert!((fill + allocs) * threads < alloc.pages());

    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
        thread::pin(t);
        for _ in 0..fill {
            alloc.get(t, order).unwrap();
        }
        barrier.wait();

        let mut pages = Vec::with_capacity(allocs);
        // Operate on filling level.
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
    assert_eq!(a.dbg_allocated_pages(), fill * threads * (1 << order));

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
