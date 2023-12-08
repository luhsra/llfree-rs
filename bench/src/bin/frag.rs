#![feature(int_roundings)]
#![feature(allocator_api)]
#![feature(new_uninit)]

use std::fs::File;
use std::io::{self, Write};
use std::sync::atomic::Ordering;
use std::sync::{Barrier, Mutex};

use clap::Parser;
use log::warn;

use llfree::frame::PT_LEN;
use llfree::util::{self, aligned_buf, WyRand};
use llfree::*;
use llfree::{thread, LLFree};

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// Max number of threads
    #[arg(short, long, default_value = "6")]
    threads: usize,
    /// Where to store the benchmark results in csv format.
    #[arg(short, long, default_value = "results/frag.csv")]
    outfile: String,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[arg(short = 's', long, default_value_t = 0)]
    order: usize,
    /// Number of iterations
    #[arg(short, long, default_value_t = 8)]
    iterations: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
    /// Percentage of free memory
    #[arg(short, long, default_value_t = 50)]
    free: usize,
    /// Using only every n-th CPU
    #[arg(long, default_value_t = 1)]
    stride: usize,
}

type Allocator<'a> = LLFree<'a>;

fn main() {
    let Args {
        threads,
        outfile,
        order,
        iterations,
        memory,
        free,
        stride,
    } = Args::parse();

    util::logging();

    assert!(free <= 90, "The maximum amount of free memory is 90%");
    assert!(order <= 6, "This benchmark is for small pages");

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    // Map memory for the allocator and initialize it
    let pages = memory * PT_LEN * PT_LEN;
    let ms = Allocator::metadata_size(threads, pages);
    let mut primary = aligned_buf(ms.primary);
    let mut secondary = aligned_buf(ms.secondary);
    let alloc =
        Allocator::new(threads, pages, Init::FreeAll, &mut primary, &mut secondary).unwrap();

    let out = Mutex::new({
        let mut out = File::create(outfile).unwrap();
        write!(out, "i,num_pages,allocs").unwrap();
        for i in 0..alloc.frames().div_ceil(PT_LEN) {
            write!(out, ",{i}").unwrap();
        }
        writeln!(out).unwrap();
        out
    });

    // Operate on half of the avaliable memory
    let allocs = (pages * free / 100) / (1 << order) / threads;
    warn!("allocs={allocs}");
    let barrier = Barrier::new(threads);

    let all_pages = {
        let mut v = Vec::with_capacity(threads);
        v.resize_with(threads, || Mutex::new(Vec::<usize>::new()));
        v
    };

    thread::parallel(0..threads, |t| {
        thread::pin(t);

        let mut rng = WyRand::new(t as u64 + 100);

        {
            let mut pages = all_pages[t].lock().unwrap();
            barrier.wait();

            while let Ok(page) = alloc.get(t, order) {
                pages.push(page);
            }
        };

        if barrier.wait().is_leader() {
            // shuffle and split even
            let mut all = Vec::new();
            for pages in all_pages.iter() {
                let pages = pages.lock().unwrap();
                all.extend_from_slice(&pages);
            }
            warn!("shuffle {}", all.len());
            assert!(all.len() > threads * allocs);
            assert_eq!(all.len(), alloc.frames());

            rng.shuffle(&mut all);
            for (chunk, pages) in all
                .chunks(all.len().div_ceil(threads))
                .zip(all_pages.iter())
            {
                let mut pages = pages.lock().unwrap();
                pages.clear();
                pages.extend_from_slice(chunk);
            }
        }
        barrier.wait();

        // Free half of it
        {
            let mut pages = all_pages[t].lock().unwrap();
            barrier.wait();

            for _ in 0..allocs {
                alloc.put(t, pages.pop().unwrap(), order).unwrap();
            }
        };

        if barrier.wait().is_leader() {
            warn!("stats 0");
            stats(&mut out.lock().unwrap(), &alloc, 0).unwrap();
        }
        barrier.wait();

        for i in 1..iterations {
            {
                let mut pages = all_pages[t].lock().unwrap();
                // realloc 10% of the remaining pages
                for _ in 0..allocs / 10 {
                    let i = rng.range(0..pages.len() as u64) as usize;
                    alloc.put(t, pages[i], order).unwrap();
                    pages[i] = alloc.get(t, order).unwrap();
                }
            };
            if barrier.wait().is_leader() {
                warn!("stats {i}");
                stats(&mut out.lock().unwrap(), &alloc, i).unwrap();
            }
            barrier.wait();
        }
    });

    warn!("{alloc:?}");
}

/// count and output stats
fn stats(out: &mut File, alloc: &Allocator, i: usize) -> io::Result<()> {
    let mut free_per_huge = Vec::with_capacity(alloc.frames() / 512);
    alloc.for_each_huge_frame(|_pfn, free| {
        free_per_huge.push(free as u16);
    });

    write!(out, "{i},{},{}", alloc.frames(), alloc.allocated_frames(),)?;
    for free in free_per_huge {
        write!(out, ",{free}")?;
    }
    writeln!(out)?;
    Ok(())
}
