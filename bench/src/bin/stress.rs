#![feature(int_roundings)]
#![feature(allocator_api)]
#![feature(new_uninit)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::time::Instant;

use clap::Parser;
use llfree::frame::PT_LEN;
use llfree::util::{aligned_buf, WyRand};
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
    #[arg(long, default_value_t = 60)]
    time: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[arg(short, long, default_value_t = 8)]
    memory: usize,
    /// Using only every n-th CPU
    #[arg(long, default_value_t = 2)]
    stride: usize,
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
    } = Args::parse();

    util::logging();

    assert!(order <= Allocator::MAX_ORDER);

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

    // Operate on half of the avaliable memory
    let barrier = Barrier::new(threads);

    let pages_per_thread = pages / threads;

    let start = Instant::now();
    let running = AtomicBool::new(true);

    warn!("start");

    thread::parallel(0..threads, |t| {
        thread::pin(t);
        let mut rng = WyRand::new(t as u64 + 100);

        let mut pages = Vec::new();

        barrier.wait();

        while let Ok(page) = alloc.get(t, order) {
            pages.push(page);
        }

        while running.load(Ordering::Relaxed) {
            let target = rng.range(0..2 * pages_per_thread as u64) as usize;

            while target != pages.len() {
                if target < pages.len() {
                    let page = pages.pop().unwrap();
                    alloc.put(t, page, order).unwrap();
                } else {
                    match alloc.get(t, order) {
                        Ok(page) => pages.push(page),
                        Err(Error::Memory) => break,
                        Err(e) => panic!("{e:?}"),
                    }
                }
            }
            rng.shuffle(&mut pages);

            if t == 0 && start.elapsed().as_secs() > time as u64 {
                running.store(false, Ordering::Relaxed);
                break;
            }
        }
    });

    warn!("{alloc:?}");
}
