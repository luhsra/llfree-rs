use std::fs::File;
use std::io::{self, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier, Mutex};

use clap::Parser;
use log::warn;

use nvalloc::lower::*;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::upper::*;
use nvalloc::util::{div_ceil, Page, WyRand};
use nvalloc::{thread, util};

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// Max number of threads
    #[arg(short, long, default_value = "6")]
    threads: usize,
    /// Where to store the benchmark results in csv format.
    #[arg(short, long, default_value = "results/bench.csv")]
    outfile: String,
    /// DAX file to be used for the allocator.
    #[arg(long)]
    dax: Option<String>,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[arg(short = 's', long, default_value_t = 0)]
    order: usize,
    /// Number of iterations
    #[arg(short, long, default_value_t = 8)]
    iterations: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
    #[arg(long, default_value_t = 1)]
    stride: usize,
}

type Allocator = Array<4, Atom<128>>;

fn main() {
    let Args {
        threads,
        outfile,
        dax,
        order,
        iterations,
        memory,
        stride,
    } = Args::parse();

    util::logging();

    let out = Arc::new(Mutex::new({
        let mut out = File::create(outfile).unwrap();
        write!(out, "i,num_pages,allocs").unwrap();
        for i in 0..PT_LEN + 1 {
            write!(out, ",{i}").unwrap();
        }
        writeln!(out).unwrap();
        out
    }));

    assert!(order <= 6, "This benchmark is for small pages");

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    let mut alloc = Allocator::default();
    let pages = memory * PT_LEN * PT_LEN;
    assert!(alloc.pages_needed(threads) <= (pages * 3 / 4));

    // Map memory for the allocator and initialize it
    let mut mapping = mapping(0x1000_0000_0000, pages, dax).unwrap();
    alloc.init(threads, &mut mapping, false).unwrap();
    alloc.free_all().unwrap();

    // Operate on half of the avaliable memory
    let allocs = (pages / 2) / (1 << order) / threads;
    warn!("allocs={allocs}");
    let alloc = Arc::new(alloc); // move into sharable reference counter.
    let barrier = Arc::new(Barrier::new(threads));

    let all_pages = Arc::new({
        let mut v = Vec::with_capacity(threads);
        v.resize_with(threads, || Mutex::new(Vec::<u64>::new()));
        v
    });
    let a = alloc.clone();
    thread::parallel(0..threads, move |t| {
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
            assert_eq!(all.len(), alloc.pages());

            rng.shuffle(&mut all);
            for (chunk, pages) in all
                .chunks(div_ceil(all.len(), threads))
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
            stats(&mut out.lock().unwrap(), alloc.as_ref(), 0).unwrap();
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
                stats(&mut out.lock().unwrap(), alloc.as_ref(), i).unwrap();
            }
            barrier.wait();
        }
    });

    warn!("{a:?}");
}

static mut FREE_PAGES: [usize; PT_LEN + 1] = [0; PT_LEN + 1];

fn count_pte2(free: usize) {
    unsafe { FREE_PAGES[free] += 1 };
}

/// count and output stats
fn stats(out: &mut File, alloc: &Allocator, i: usize) -> io::Result<()> {
    unsafe { FREE_PAGES.fill(0) };
    alloc.dbg_for_each_huge_page(count_pte2);

    write!(out, "{i},{},{}", alloc.pages(), alloc.dbg_allocated_pages(),)?;
    for b in unsafe { &FREE_PAGES } {
        write!(out, ",{b}")?;
    }
    writeln!(out)?;
    Ok(())
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
