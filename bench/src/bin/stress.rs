use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::time::Instant;

use clap::Parser;
use llfree::frame::Frame;
use llfree::util::WyRand;
use llfree::*;
use log::warn;

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// Max number of threads
    #[arg(short, long, default_value = "8")]
    threads: usize,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[arg(short = 's', long, default_value_t = 0)]
    order: usize,
    /// Runtime in seconds
    #[arg(long, default_value_t = 20)]
    time: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[arg(short, long, default_value_t = 8)]
    memory: usize,
    /// Using only every n-th CPU
    #[arg(long, default_value_t = 2)]
    stride: usize,
    /// Monitor and output fragmentation
    #[arg(long)]
    frag: Option<PathBuf>,
}

#[cfg(feature = "llc")]
type Allocator = LLC;
#[cfg(not(feature = "llc"))]
type Allocator<'a> = LLFree<'a>;

fn main() {
    let Args {
        threads,
        order,
        time,
        memory,
        stride,
        frag,
    } = Args::parse();

    util::logging();

    assert!(order <= MAX_ORDER);

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    // Map memory for the allocator and initialize it
    let pages = (memory << 30) / Frame::SIZE;
    let ms = Allocator::metadata_size(threads, pages);
    let meta = MetaData::alloc(ms);
    let alloc = Allocator::new(threads, pages, Init::FreeAll, meta).unwrap();
    alloc.validate();

    // Operate on half of the avaliable memory
    let barrier = Barrier::new(threads + 1);

    let pages_per_thread = pages / threads;

    let start = Instant::now();
    let running = AtomicBool::new(true);

    warn!("start");
    let rand = unsafe { libc::rand() as u64 };

    let (allocated, score) = std::thread::scope(|s| {
        let monitor = s.spawn(|| {
            thread::pin(threads);
            let mut frag = frag.map(|path| BufWriter::new(File::create(path).unwrap()));

            barrier.wait();

            let mut frag_sec = 0;
            let mut score = 0.0;
            while running.load(Ordering::Relaxed) {
                let elapsed = start.elapsed();
                if elapsed.as_secs() > frag_sec {
                    if let Some(frag) = frag.as_mut() {
                        for i in 0..alloc.frames().div_ceil(1 << HUGE_ORDER) {
                            let free = alloc.free_at(i << HUGE_ORDER, HUGE_ORDER);
                            let level = if free == 0 { 0 } else { 1 + free / 64 };
                            write!(frag, "{level:?}").unwrap();
                        }
                        writeln!(frag).unwrap();
                    }

                    let huge = alloc.free_huge();
                    let optimal = alloc.free_frames() >> HUGE_ORDER;
                    let fraction = 100.0 * huge as f32 / optimal as f32;
                    warn!("free-huge {huge}/{optimal} = {fraction:.1}");
                    score += fraction;
                    frag_sec += 1;
                }

                if elapsed.as_secs() > time as u64 {
                    running.store(false, Ordering::Relaxed);
                    break;
                }
            }
            score / frag_sec as f32
        });

        let allocated = thread::parallel(0..threads, |t| {
            thread::pin(t);
            let mut rng = WyRand::new(t as u64 + rand);
            let mut pages = Vec::with_capacity(pages_per_thread);

            barrier.wait();

            while let Ok(page) = alloc.get(t, Flags::o(order)) {
                pages.push(page);
            }

            while running.load(Ordering::Relaxed) {
                // Random target filling level
                let target = rng.range(0..pages_per_thread as u64) as usize;

                rng.shuffle(&mut pages);
                while target != pages.len() {
                    if target < pages.len() {
                        let page = pages.pop().unwrap();
                        alloc.put(t, page, Flags::o(order)).unwrap();
                    } else {
                        match alloc.get(t, Flags::o(order)) {
                            Ok(page) => pages.push(page),
                            Err(Error::Memory) => break,
                            Err(e) => panic!("{e:?}"),
                        }
                    }
                }
            }

            warn!("thread {t}: {}", pages.len());
            pages.len()
        });

        let score = monitor.join().unwrap();
        (allocated, score)
    });

    assert_eq!(
        allocated.into_iter().sum::<usize>(),
        alloc.allocated_frames()
    );
    alloc.validate();
    warn!("{alloc:?}");
    warn!("score = {score:.1}")
}
