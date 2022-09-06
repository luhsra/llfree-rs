use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use clap::Parser;
use log::warn;
use nvalloc::lower::Atom;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::upper::{Alloc, ArrayAtomic, MIN_PAGES};
use nvalloc::util::{logging, Cycles, Page};

type Allocator = ArrayAtomic<Atom<128>>;

/// Benchmarking an allocator in more detail.
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short, long, default_value_t = 1)]
    threads: usize,
    #[clap(short, long, default_value = "bench/out/detail.csv")]
    outfile: String,
    #[clap(long)]
    dax: Option<String>,
    #[clap(short, long, default_value_t = 1)]
    iterations: usize,
    #[clap(short, long, default_value_t = 0)]
    order: usize,
}

fn main() {
    let Args {
        threads,
        outfile,
        dax,
        iterations,
        order,
    } = Args::parse();

    logging();

    assert!(threads >= 1);
    assert!(threads <= std::thread::available_parallelism().unwrap().get());

    let mem_pages = 2 * threads * MIN_PAGES;
    let mut mapping = mapping(0x1000_0000_0000, mem_pages, dax).unwrap();

    let mut times = vec![Perf::default(); 2 * PT_LEN];

    // Warmup
    for page in &mut mapping[..] {
        *page.cast_mut::<usize>() = 1;
    }
    for _ in 0..iterations {
        let timer = Instant::now();
        let alloc = Arc::new({
            let mut a = Allocator::default();
            a.init(threads, &mut mapping, true).unwrap();
            a
        });
        warn!("init time {}ms", timer.elapsed().as_millis());
        // Allocate half the memory
        let allocs = mapping.len() / threads / 2 / (1 << order);
        assert!(allocs % PT_LEN == 0);
        warn!("\n\n>>> bench t={threads} o={order:?} {allocs}\n");

        let barrier = Arc::new(Barrier::new(threads));
        let a = alloc.clone();
        let t_times = thread::parallel(threads as _, move |t| {
            thread::pin(t);
            barrier.wait();
            let timer = Instant::now();
            let mut pages = Vec::new();
            let mut times = vec![Perf::default(); 2 * PT_LEN];

            let t_get = &mut times[0..PT_LEN];
            for i in 0..allocs {
                let timer = Cycles::now();
                pages.push(alloc.get(t, order).unwrap());
                let t = timer.elapsed() as f64;

                let p = &mut t_get[i % PT_LEN];
                p.avg += t;
                p.std += t * t;
                p.min = p.min.min(t);
                p.max = p.max.max(t);
            }

            barrier.wait();

            let t_put = &mut times[PT_LEN..];
            for (i, page) in pages.into_iter().enumerate() {
                let timer = Cycles::now();
                alloc.put(t, page, order).unwrap();
                let t = timer.elapsed() as f64;

                let p = &mut t_put[i % PT_LEN];
                p.avg += t;
                p.std += t * t;
                p.min = p.min.min(t);
                p.max = p.max.max(t);
            }

            let n = (allocs / PT_LEN) as f64;
            for t in &mut times {
                t.avg /= n;
                t.std /= n;
            }

            warn!("time {}ms", timer.elapsed().as_millis());
            times
        });

        assert_eq!(a.dbg_allocated_pages(), 0);
        drop(a);

        for t in t_times {
            for i in 0..2 * PT_LEN {
                times[i].avg += t[i].avg / threads as f64;
                times[i].std += t[i].std / threads as f64;
                times[i].min = times[i].min.min(t[i].min);
                times[i].max = times[i].max.max(t[i].max);
            }
        }
    }

    // https://en.wikipedia.org/wiki/Standard_deviation#Identities_and_mathematical_properties
    let n = iterations as f64;
    warn!("n = {n}");
    for t in &mut times {
        t.avg /= n;
        t.std = (t.std / n - t.avg * t.avg).sqrt();
    }

    let name = Allocator::default().name();
    let mut outfile = File::create(outfile).unwrap();
    writeln!(outfile, "alloc,threads,op,num,avg,std,min,max").unwrap();
    for (num, Perf { avg, std, min, max }) in times.into_iter().enumerate() {
        if num < 512 {
            writeln!(
                outfile,
                "{name},{threads},get,{num},{avg},{std},{min},{max}"
            )
            .unwrap();
        } else {
            let num = num - 512;
            writeln!(
                outfile,
                "{name},{threads},put,{num},{avg},{std},{min},{max}"
            )
            .unwrap();
        }
    }
}

#[derive(Clone, Copy)]
struct Perf {
    avg: f64,
    std: f64,
    min: f64,
    max: f64,
}

impl Default for Perf {
    fn default() -> Self {
        Self {
            avg: 0.0,
            std: 0.0,
            min: f64::INFINITY,
            max: 0.0,
        }
    }
}

#[allow(unused_variables)]
fn mapping(begin: usize, length: usize, dax: Option<String>) -> Result<MMap<Page>, ()> {
    #[cfg(target_os = "linux")]
    if length > 0 {
        if let Some(file) = dax {
            warn!(
                "MMap file {file} l={}G",
                (length * std::mem::size_of::<Page>()) >> 30
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
