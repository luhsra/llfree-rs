use std::any::type_name;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use clap::Parser;
use log::warn;
use nvalloc::alloc::table::TableAlloc;
use nvalloc::alloc::{stack::StackAlloc, Alloc, Size, MIN_PAGES};
use nvalloc::mmap::MMap;
use nvalloc::table::Table;
use nvalloc::thread;
use nvalloc::util::{logging, Cycles, Page};

/// Benchmarking an allocator in more detail.
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short, long, default_value_t = 1)]
    threads: usize,
    #[clap(short, long, default_value = "bench/detail.csv")]
    outfile: String,
    #[clap(long)]
    dax: Option<String>,
    #[clap(short, long, default_value_t = 1)]
    iterations: usize,
    #[clap(short, long, default_value_t = 0)]
    size: usize,
    #[clap(long, default_value_t = 1)]
    cpu_stride: usize,
}

type A = TableAlloc;

fn main() {
    let Args {
        threads,
        outfile,
        dax,
        iterations,
        size,
        cpu_stride,
    } = Args::parse();

    logging();

    assert!(threads >= 1);
    assert!(threads * cpu_stride <= num_cpus::get());

    unsafe { nvalloc::thread::CPU_STRIDE = cpu_stride };

    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        _ => panic!("`size` has to be 0 or 1"),
    };

    let mem_pages = 2 * threads * MIN_PAGES;
    let mut mapping = mapping(0x1000_0000_0000, mem_pages, dax).unwrap();

    warn!("\n\n>>> bench t={threads} {size:?} {}\n");
    // Allocate half the memory
    let allocs = mapping.len() / threads / 2 / Table::span(size as _);
    assert!(allocs % Table::LEN == 0);

    let mut times = vec![Perf::default(); 2 * Table::LEN];

    for _ in 0..iterations {
        let timer = Instant::now();
        A::init(threads, &mut mapping).unwrap();
        warn!("init time {}ms", timer.elapsed().as_millis());

        let barrier = Arc::new(Barrier::new(threads));

        let timer = Instant::now();
        let barrier = barrier.clone();
        let t_times = thread::parallel(threads as _, move |t| {
            thread::pin(t);
            barrier.wait();
            let mut pages = Vec::new();
            let mut times = vec![Perf::default(); 2 * Table::LEN];

            let t_get = &mut times[0..Table::LEN];
            for i in 0..allocs {
                let timer = Cycles::now();
                pages.push(A::instance().get(t, size).unwrap());
                let t = timer.elapsed() as f64;

                let p = &mut t_get[i % Table::LEN];
                p.avg += t;
                p.std += t * t;
                p.min = p.min.min(t);
                p.max = p.max.max(t);
            }

            barrier.wait();

            let t_put = &mut times[Table::LEN..];
            for (i, page) in pages.into_iter().enumerate() {
                let timer = Cycles::now();
                A::instance().put(t, page).unwrap();
                let t = timer.elapsed() as f64;

                let p = &mut t_put[i % Table::LEN];
                p.avg += t;
                p.std += t * t;
                p.min = p.min.min(t);
                p.max = p.max.max(t);
            }

            let n = (allocs / Table::LEN) as f64;
            for t in &mut times {
                t.avg /= n;
                t.std /= n;
            }

            let total = timer.elapsed().as_millis();
            warn!("time {total}ms");
            times
        });

        assert_eq!(A::instance().allocated_pages(), 0);

        A::destroy();

        for t in t_times {
            for i in 0..2 * Table::LEN {
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

    let alloc = type_name::<A>();
    let alloc = &alloc[alloc.rfind(':').map(|i| i + 1).unwrap_or_default()..];

    let mut outfile = File::create(outfile).unwrap();
    writeln!(outfile, "alloc,threads,op,num,avg,std,min,max").unwrap();
    for (num, Perf { avg, std, min, max }) in times.into_iter().enumerate() {
        if num < 512 {
            writeln!(
                outfile,
                "{alloc},{threads},get,{num},{avg},{std},{min},{max}"
            )
            .unwrap();
        } else {
            let num = num - 512;
            writeln!(
                outfile,
                "{alloc},{threads},put,{num},{avg},{std},{min},{max}"
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

fn mapping<'a>(begin: usize, length: usize, dax: Option<String>) -> Result<MMap<'a, Page>, ()> {
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
    MMap::anon(begin, length)
}
