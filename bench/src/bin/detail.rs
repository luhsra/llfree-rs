use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use clap::Parser;
use log::warn;
use nvalloc::lower::Atom;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::upper::{Alloc, ArrayAtomic};
use nvalloc::util::{div_ceil, logging, Page};

type Allocator = ArrayAtomic<3, Atom<128>>;

/// Benchmarking an allocator in more detail.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    #[arg(short, long, default_value_t = 1)]
    threads: usize,
    #[arg(short, long, default_value = "results/out/detail.csv")]
    outfile: PathBuf,
    #[arg(long)]
    dax: Option<String>,
    #[arg(short, long, default_value_t = 1)]
    iterations: usize,
    #[arg(short, long, default_value_t = 0)]
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

    let mut mapping = mapping(0x1000_0000_0000, PT_LEN * PT_LEN * threads, dax).unwrap();

    // Populate mapping
    warn!("populate {} pages", mapping.len());
    let chunk_size = div_ceil(mapping.len(), 4);
    thread::parallel(mapping.chunks_mut(chunk_size), |chunk| {
        for page in chunk {
            *page.cast_mut::<usize>() = 1;
        }
    });

    let mut times = vec![Perf::default(); 2 * PT_LEN];

    for _ in 0..iterations {
        let timer = Instant::now();

        let mut a = Allocator::default();
        a.init(threads, &mut mapping, true).unwrap();
        a.free_all().unwrap();
        let alloc = &a;

        warn!("init time {}ms", timer.elapsed().as_millis());

        // Allocate half the memory
        let allocs = (mapping.len() / threads) / (2 * (1 << order));
        assert!(allocs % PT_LEN == 0);
        warn!("\n\n>>> bench t={threads} o={order:?} {allocs}\n");

        let barrier = Arc::new(Barrier::new(threads));
        let t_times = thread::parallel(0..threads, move |t| {
            thread::pin(t);
            barrier.wait();

            let timer = Instant::now();
            let mut pages = Vec::with_capacity(allocs);
            let mut perf = vec![Perf::default(); 2 * PT_LEN];

            let t_get = &mut perf[0..PT_LEN];
            for i in 0..allocs {
                let timer = Instant::now();
                let page = alloc.get(t, order).unwrap();
                let t = timer.elapsed().as_nanos() as f64;

                t_get[i % PT_LEN].add_time(t);
                pages.push(page);
            }

            barrier.wait();

            let t_put = &mut perf[PT_LEN..];
            for (i, page) in pages.into_iter().enumerate() {
                let timer = Instant::now();
                alloc.put(t, page, order).unwrap();
                let t = timer.elapsed().as_nanos() as f64;

                t_put[i % PT_LEN].add_time(t);
            }

            let n = (allocs / PT_LEN) as f64;
            for t in &mut perf {
                t.finalize(n);
            }

            warn!("time {}ms", timer.elapsed().as_millis());
            perf
        });

        assert_eq!(alloc.dbg_allocated_pages(), 0);
        drop(a);

        for t in t_times {
            for i in 0..2 * PT_LEN {
                times[i].avg += t[i].avg / (threads * iterations) as f64;
                times[i].std += t[i].std / (threads * iterations) as f64;
                times[i].min = times[i].min.min(t[i].min);
                times[i].max = times[i].max.max(t[i].max);
            }
        }
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

impl Perf {
    fn add_time(&mut self, t: f64) {
        self.avg += t;
        self.std += t * t;
        self.min = self.min.min(t);
        self.max = self.max.max(t);
    }
    fn finalize(&mut self, n: f64) {
        self.avg /= n;
        // https://en.wikipedia.org/wiki/Standard_deviation#Identities_and_mathematical_properties
        self.std = (self.std / n - self.avg * self.avg).sqrt()
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
