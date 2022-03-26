#![cfg(all(feature = "thread", feature = "logger"))]

use core::fmt;
use core::iter::FromIterator;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use clap::{ArgEnum, Parser};
use log::warn;

use nvalloc::alloc::{
    self, Alloc, ArrayAlignedAlloc, ArrayAtomicAlloc, ArrayLockedAlloc, ArrayUnalignedAlloc,
    ListLocalAlloc, ListLockedAlloc, Size, TableAlloc, MIN_PAGES,
};
use nvalloc::lower::{DynamicLower, FixedLower};
use nvalloc::mmap::MMap;
use nvalloc::table::Table;
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
    /// Size of the allocated pages: 0 = 4K, 1 = 2M, 2 = 1G
    #[clap(short, long, default_value_t = 0)]
    size: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[clap(short, long, default_value_t = 16)]
    memory: usize,
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
        size,
        memory,
    } = Args::parse();

    util::logging();

    let pages = (memory * Table::span(2)) / threads;
    assert!(pages >= MIN_PAGES);

    let mut out = File::create(outfile).unwrap();
    writeln!(out, "alloc,x,iteration,pages,{}", Perf::header()).unwrap();

    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        2 => Size::L2,
        _ => panic!("`size` has to be 0, 1 or 2"),
    };
    warn!("Allocating size {size:?}");

    let mut mapping = mapping(0x1000_0000_0000, memory * Table::span(2), dax).unwrap();

    // Warmup
    for page in &mut mapping[..] {
        *page.cast_mut::<usize>() = 1;
    }

    let alloc_names: HashSet<String> = HashSet::from_iter(allocs.into_iter());
    type F = FixedLower;
    type D = DynamicLower;
    let allocs: [Arc<dyn Alloc>; 12] = [
        Arc::new(ArrayAlignedAlloc::<F>::default()),
        Arc::new(ArrayUnalignedAlloc::<F>::default()),
        Arc::new(ArrayLockedAlloc::<F>::default()),
        Arc::new(ArrayAtomicAlloc::<F>::default()),
        Arc::new(TableAlloc::<F>::default()),
        Arc::new(ArrayAlignedAlloc::<D>::default()),
        Arc::new(ArrayUnalignedAlloc::<D>::default()),
        Arc::new(ArrayLockedAlloc::<D>::default()),
        Arc::new(ArrayAtomicAlloc::<D>::default()),
        Arc::new(TableAlloc::<D>::default()),
        Arc::new(ListLocalAlloc::default()),
        Arc::new(ListLockedAlloc::default()),
    ];

    let mut conditions = HashMap::<String, &'static dyn Fn(usize, Size) -> bool>::new();
    conditions.insert(alloc::name::<ListLocalAlloc>(), &|_, size| size == Size::L0);
    conditions.insert(alloc::name::<ListLockedAlloc>(), &|cores, size| {
        size == Size::L0 && (cores <= 16 || cores == 32)
    });

    for x in x {
        for alloc in &allocs {
            let name = alloc.name();
            if alloc_names.contains(&name)
                && conditions
                    .get(&name)
                    .map(|f| f(bench.threads(threads, x), size))
                    .unwrap_or(true)
            {
                for i in 0..iterations {
                    let perf = bench.run(
                        alloc.clone(),
                        &mut mapping[..pages * threads],
                        size,
                        threads,
                        x,
                    );
                    writeln!(out, "{name},{x},{i},{pages},{perf}").unwrap();
                }
            }
        }
    }
    warn!("Ok");
    drop(allocs); // drop first
}

fn mapping<'a>(begin: usize, length: usize, dax: Option<String>) -> Result<MMap<Page>, ()> {
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
        size: Size,
        threads: usize,
        x: usize,
    ) -> Perf {
        warn!("\n\n>>> bench {self:?} x={x} {size:?} {}\n", alloc.name());

        match self {
            Benchmark::Bulk => bulk(alloc, mapping, size, threads, x),
            Benchmark::Repeat => repeat(alloc, mapping, size, threads, x),
            Benchmark::Rand => rand(alloc, mapping, size, threads, x),
            Benchmark::Filling => filling(alloc, mapping, size, threads, x),
        }
    }
}

fn bulk(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    size: Size,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    let pages = mapping.len() / max_threads * threads;
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, &mut mapping[..pages], true)
        .unwrap();
    let init = timer.elapsed().as_millis();
    warn!("init time {init}ms");

    let allocs = alloc.pages() / threads / 2 / Table::span(size as _);
    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
        thread::pin(t);
        barrier.wait();
        let mut pages = Vec::with_capacity(allocs);
        let t1 = Instant::now();
        for _ in 0..allocs {
            pages.push(alloc.get(t, size).unwrap());
        }
        let get = t1.elapsed().as_nanos() / allocs as u128;
        pages.reverse();
        let pages = black_box(pages);

        barrier.wait();
        let t2 = Instant::now();
        for page in pages {
            alloc.put(t, page).unwrap();
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
        }
    }));
    assert_eq!(a.dbg_allocated_pages(), 0);

    perf.init = init;
    warn!("{perf:#?}");
    perf
}

fn repeat(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    size: Size,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    let pages = mapping.len() / max_threads * threads;
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, &mut mapping[..pages], true)
        .unwrap();
    let init = timer.elapsed().as_millis();
    warn!("init time {init}ms");

    let allocs = alloc.pages() / threads / 2 / Table::span(size as _);
    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
        thread::pin(t);
        for _ in 0..allocs {
            alloc.get(t, size).unwrap();
        }

        barrier.wait();
        let timer = Instant::now();
        for _ in 0..allocs {
            let page = alloc.get(t, size).unwrap();
            let page = black_box(page);
            alloc.put(t, page).unwrap();
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
        }
    }));
    assert_eq!(
        a.dbg_allocated_pages(),
        allocs * threads * Table::span(size as _)
    );

    perf.init = init;
    warn!("{perf:#?}");
    perf
}

fn rand(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    size: Size,
    max_threads: usize,
    threads: usize,
) -> Perf {
    let timer = Instant::now();
    let pages = mapping.len() / max_threads * threads;
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, &mut mapping[..pages], true)
        .unwrap();
    let init = timer.elapsed().as_millis();
    warn!("init time {init}ms");

    let allocs = alloc.pages() / threads / 2 / Table::span(size as _);
    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
        thread::pin(t);
        let mut pages = Vec::with_capacity(allocs);
        for _ in 0..allocs {
            pages.push(alloc.get(t, size).unwrap());
        }

        let mut rng = WyRand::new(t as _);

        barrier.wait();
        let timer = Instant::now();

        for _ in 0..allocs {
            let i = rng.range(0..pages.len() as _) as usize;
            alloc.put(t, pages[i]).unwrap();
            pages[i] = alloc.get(t, size).unwrap();
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
        }
    }));
    assert_eq!(
        a.dbg_allocated_pages(),
        allocs * threads * Table::span(size as _)
    );

    perf.init = init;
    warn!("{perf:#?}");
    perf
}

fn filling(
    alloc: Arc<dyn Alloc>,
    mapping: &mut [Page],
    size: Size,
    threads: usize,
    x: usize,
) -> Perf {
    assert!(x <= 90);

    let timer = Instant::now();
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, mapping, true)
        .unwrap();
    let init = timer.elapsed().as_millis();
    warn!("init time {init}ms");

    let allocs = alloc.pages() / threads / Table::span(size as _);

    // Allocate to filling level
    let fill = (allocs as f64 * (x as f64 / 100.0)) as usize;
    let allocs = allocs / 10;
    warn!("fill={fill} allocs={allocs}");

    assert!((fill + allocs) * threads < alloc.pages());

    let barrier = Arc::new(Barrier::new(threads));
    let a = alloc.clone();
    let mut perf = Perf::avg(thread::parallel(threads, move |t| {
        thread::pin(t);
        for _ in 0..fill {
            alloc.get(t, size).unwrap();
        }
        // Warm up
        let mut pages = Vec::with_capacity(allocs);
        for _ in 0..allocs {
            pages.push(alloc.get(t, size).unwrap());
        }
        pages.reverse();
        for &page in &pages {
            alloc.put(t, page).unwrap();
        }
        pages.clear();
        barrier.wait();

        // Operate on filling level.
        let t1 = Instant::now();
        for _ in 0..allocs {
            pages.push(alloc.get(t, size).unwrap());
        }
        let get = t1.elapsed().as_nanos() / allocs as u128;
        pages.reverse();
        let pages = black_box(pages);

        barrier.wait();
        let t2 = Instant::now();
        for page in pages {
            alloc.put(t, page).unwrap();
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
        }
    }));
    assert_eq!(
        a.dbg_allocated_pages(),
        fill * threads * Table::span(size as _)
    );

    perf.init = init;
    warn!("{perf:#?}");
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
        res.init /= counter as u128;
        res.total /= counter as u128;
        res
    }
    fn header() -> &'static str {
        "get_min,get_avg,get_max,put_min,put_avg,put_max,init,total"
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
        } = self;
        write!(
            f,
            "{get_min},{get_avg},{get_max},{put_min},{put_avg},{put_max},{init},{total}"
        )
    }
}
