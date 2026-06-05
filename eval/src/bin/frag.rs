use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::DerefMut;
use std::sync::atomic::Ordering;
use std::sync::{Barrier, Mutex};

use facet::Facet;
use figue::{self as args, FigueBuiltins};
use llfree::frame::Frame;
use llfree::util::WyRand;
use llfree::*;
use llfree_eval::thread;
use log::warn;

/// Benchmarking the allocators against each other.
#[derive(Facet, Debug)]
struct Args {
    /// Max number of threads
    #[facet(args::short, args::named, default = 6)]
    threads: usize,
    /// Where to store the benchmark results in txt format.
    #[facet(args::short, args::named, default = "results/frag.txt")]
    outfile: String,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[facet(args::short = 's', args::named, default = 0)]
    order: usize,
    /// Number of iterations
    #[facet(args::short, args::named, default = 8)]
    iterations: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[facet(args::short, args::named, default = 16)]
    memory: usize,
    /// Percentage of free memory
    #[facet(args::short, args::named, default = 50)]
    free: usize,
    /// Using only every n-th CPU
    #[facet(args::named, default = 1)]
    stride: usize,

    #[facet(flatten)]
    builtins: FigueBuiltins,
}

cfg_select! {
    feature = "llc" => {
        use llfree_eval::LLC;
        type Allocator = LLC;
    }
    _ => {
        type Allocator<'a> = LLFree<'a>;
    }
}

fn main() {
    let Args {
        threads,
        outfile,
        order,
        iterations,
        memory,
        free,
        stride,
        builtins: _,
    } = figue::from_std_args().unwrap();

    util::logging();

    assert!(free <= 90, "The maximum amount of free memory is 90%");
    assert!(order <= 6, "This benchmark is for small pages");

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    // Map memory for the allocator and initialize it
    let pages = (memory << 30) / Frame::SIZE;
    let (clustering, request) = Clustering::simple(threads);
    let ms = Allocator::metadata_size(&clustering, pages);
    let meta = MetaData::alloc(&ms);
    let alloc = Allocator::new(pages, Init::FreeAll, &clustering, meta).unwrap();

    let out = Mutex::new(BufWriter::new(File::create(outfile).unwrap()));

    // Operate on half of the avaliable memory
    let allocs = (pages * free / 100) / (1 << order) / threads;
    warn!("allocs={allocs}");
    let barrier = Barrier::new(threads);

    let all_pages = {
        let mut v = Vec::with_capacity(threads);
        v.resize_with(threads, || Mutex::new(Vec::<FrameId>::new()));
        v
    };

    let seed = unsafe { libc::rand() } as u64;

    thread::parallel(0..threads, |t| {
        thread::pin(t);

        let mut rng = WyRand::new(t as u64 + seed);

        {
            let mut pages = all_pages[t].lock().unwrap();
            barrier.wait();

            while let Ok((page, _)) = alloc.get(None, request(order, t)) {
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
                alloc.put(pages.pop().unwrap(), request(order, t)).unwrap();
            }
        };

        if barrier.wait().is_leader() {
            warn!("stats 0");
            let mut out = out.lock().unwrap();
            stats(out.deref_mut(), &alloc).unwrap();
        }
        barrier.wait();

        for i in 1..iterations {
            {
                let mut pages = all_pages[t].lock().unwrap();
                // realloc 10% of the remaining pages
                for _ in 0..allocs / 10 {
                    let i = rng.range(0..pages.len() as u64) as usize;
                    alloc.put(pages[i], request(order, t)).unwrap();
                    pages[i] = alloc.get(None, request(order, t)).unwrap().0;
                }
            };
            if barrier.wait().is_leader() {
                warn!("stats {i}");
                let mut out = out.lock().unwrap();
                stats(out.deref_mut(), &alloc).unwrap();
            }
            barrier.wait();
        }
    });

    warn!("{alloc:?}");
}

/// count and output stats
fn stats(out: &mut impl Write, alloc: &Allocator) -> io::Result<()> {
    for huge in 0..alloc.frames().div_ceil(1 << HUGE_ORDER) {
        let free = alloc
            .stats_at(HugeId(huge).as_frame(), HUGE_ORDER)
            .free_frames;
        let level = if free == 0 { 0 } else { free / 64 + 1 };
        assert!(level <= 9);
        write!(out, "{level}")?;
    }
    writeln!(out)?;
    Ok(())
}
