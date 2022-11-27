#![feature(int_roundings)]

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Barrier;
use std::time::Instant;

use clap::Parser;
use log::warn;
use nvalloc::lower::Atom;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::upper::{Alloc, Array};
use nvalloc::util::{logging, Page};

type Allocator = Array<3, Atom<128>>;

/// Measuring the allocation times and collect them into time buckets for a time distribution analysis.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    #[arg(short, long, default_value_t = 1)]
    threads: usize,
    #[arg(short, long, default_value = "results/out/dist.csv")]
    outfile: PathBuf,
    #[arg(long)]
    dax: Option<String>,
    #[arg(short, long, default_value_t = 1)]
    iterations: usize,
    #[arg(short, long, default_value_t = 0)]
    order: usize,
    #[arg(short, long, default_value_t = 500)]
    buckets: usize,
    #[arg(short, long, default_value_t = 50)]
    start: usize,
    #[arg(short, long, default_value_t = 1050)]
    end: usize,
}

fn main() {
    let Args {
        threads,
        outfile,
        dax,
        iterations,
        order,
        buckets,
        start,
        end,
    } = Args::parse();

    logging();

    assert!(threads >= 1);
    assert!(threads <= std::thread::available_parallelism().unwrap().get());

    let mut mapping = mapping(0x1000_0000_0000, PT_LEN * PT_LEN * threads, dax).unwrap();

    // Populate mapping
    warn!("populate {} pages", mapping.len());
    let chunk_size = mapping.len().div_ceil(4);
    thread::parallel(mapping.chunks_mut(chunk_size), |chunk| {
        for page in chunk {
            *page.cast_mut::<usize>() = 1;
        }
    });

    assert!(start < end);
    let mut get_buckets = vec![0usize; buckets];
    let mut put_buckets = vec![0usize; buckets];
    let bucket_size = (end - start) / buckets;
    assert!(bucket_size > 0);

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

        let barrier = Barrier::new(threads);
        let t_buckets = thread::parallel(0..threads, |t| {
            thread::pin(t);
            barrier.wait();

            let timer = Instant::now();
            let mut pages = Vec::with_capacity(allocs);

            let mut get_buckets = vec![0usize; buckets];
            let mut put_buckets = vec![0usize; buckets];

            for _ in 0..allocs {
                let timer = Instant::now();
                let page = alloc.get(t, order).unwrap();
                let t = timer.elapsed().as_nanos() as usize;

                let n = ((t.saturating_sub(start)) / bucket_size).min(buckets - 1);
                get_buckets[n] += 1;
                pages.push(page);
            }

            barrier.wait();

            for page in pages {
                let timer = Instant::now();
                alloc.put(t, page, order).unwrap();
                let t = timer.elapsed().as_nanos() as usize;
                let n = ((t.saturating_sub(start)) / bucket_size).min(buckets - 1);

                put_buckets[n] += 1;
            }

            warn!("time {}ms", timer.elapsed().as_millis());
            (get_buckets, put_buckets)
        });

        assert_eq!(alloc.dbg_allocated_pages(), 0);
        drop(a);

        for (get_b, put_b) in t_buckets {
            for i in 0..buckets {
                get_buckets[i] += get_b[i];
                put_buckets[i] += put_b[i];
            }
        }
    }

    let mut outfile = File::create(outfile).unwrap();
    writeln!(outfile, "op,bucket,count").unwrap();
    for (i, count) in get_buckets.into_iter().enumerate() {
        writeln!(outfile, "get,{},{count}", i * bucket_size + start).unwrap();
    }
    for (i, count) in put_buckets.into_iter().enumerate() {
        writeln!(outfile, "put,{},{count}", i * bucket_size + start).unwrap();
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
