#![feature(int_roundings)]
#![feature(allocator_api)]
#![feature(new_uninit)]

use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

use clap::Parser;
use log::warn;

use llfree::frame::{Frame, PT_LEN};
use llfree::mmap::MMap;
use llfree::thread;
use llfree::util::{self, aligned_buf, WyRand};
use llfree::LLFree;
use llfree::wrapper::NvmAlloc;

/// Benchmarking the (crashed) recovery.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// DAX file to be used for the allocator.
    #[arg(long)]
    dax: String,
    /// Max amount of memory in GiB.
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
    /// Number of threads.
    #[arg(short, long)]
    threads: usize,
    /// Simulate a crash by killing the process after n seconds.
    #[arg(short, long)]
    crash: Option<u64>,
    /// Initialize and run the allocator
    #[arg(long)]
    init: bool,
    /// Where to store the benchmark results in csv format.
    #[arg(short, long, default_value = "")]
    outfile: String,
    /// Number of iterations
    #[arg(short, long, default_value_t = 1)]
    iter: usize,
}

type Allocator<'a> = NvmAlloc<'a, LLFree<'a>>;

fn main() {
    util::logging();

    let Args {
        dax,
        memory,
        threads,
        crash,
        init,
        outfile,
        iter,
    } = Args::parse();

    assert!(memory > 0 && threads > 0);

    let mut time_max = 0;
    let mut time_min = u128::MAX;
    let mut time_avg = 0;
    for _ in 0..iter {
        warn!("Iter: {iter}");
        if init {
            initialize(memory, &dax, threads, crash.is_some());
            return;
        } else {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.args([
                "--init",
                "--dax",
                &dax,
                &format!("-m{memory}"),
                &format!("-t{threads}"),
            ]);
            if crash.is_some() {
                command.arg("-c0");
            }
            let mut process = command.spawn().unwrap();
            if let Some(sec) = crash {
                std::thread::sleep(Duration::from_secs(sec));
                process.kill().unwrap();
            } else {
                let ret = process.wait().unwrap();
                assert!(ret.success());
            }

            let time = recover(threads, memory, &dax);
            time_max = time_max.max(time);
            time_min = time_min.min(time);
            time_avg += time;
        }
    }
    time_avg /= iter as u128;

    let mut out = File::create(outfile).unwrap();
    writeln!(out, "min,avg,max\n{time_min},{time_avg},{time_max}").unwrap();
}

fn initialize(memory: usize, dax: &str, threads: usize, crash: bool) {
    let mut mapping = mapping(0x1000_0000_0000, memory * PT_LEN * PT_LEN, dax);
    let volatile = aligned_buf(Allocator::metadata_size(threads, mapping.len())).leak();
    let alloc = Allocator::new(threads, &mut mapping, false, volatile).unwrap();
    warn!("Prepare alloc");

    thread::parallel(0..threads, |t| {
        let mut rng = WyRand::new(t as u64);

        let allocs = alloc.frames() / threads / 2;

        let mut pages = Vec::with_capacity(allocs);
        for _ in 0..allocs {
            pages.push(alloc.get(t, 0).unwrap());
        }

        if crash {
            loop {
                let i = rng.range(0..pages.len() as _) as usize;
                alloc.put(t, pages[i], 0).unwrap();
                black_box(pages[i]);
                pages[i] = alloc.get(t, 0).unwrap();
            }
        }
    });
    assert!(!crash);
    let num_allocated = alloc.allocated_frames();
    warn!("Allocated: {num_allocated}");
    assert!(0 < num_allocated && num_allocated < alloc.frames());
}

fn recover(threads: usize, memory: usize, dax: &str) -> u128 {
    let mut mapping = mapping(0x1000_0000_0000, memory * PT_LEN * PT_LEN, dax);
    let volatile = aligned_buf(Allocator::metadata_size(threads, mapping.len())).leak();

    warn!("Recover alloc");
    let timer = Instant::now();
    let alloc = Allocator::new(threads, &mut mapping, true, volatile).unwrap();
    let time = timer.elapsed().as_nanos();

    let num_alloc = alloc.allocated_frames();
    warn!("Recovered {num_alloc} allocations in {time} ns");
    let expected = alloc.frames() / 2;
    assert!(expected - threads <= num_alloc && num_alloc <= expected + threads);

    time
}

#[allow(unused_variables)]
pub fn mapping(begin: usize, length: usize, dax: &str) -> Box<[Frame], MMap> {
    #[cfg(target_os = "linux")]
    {
        warn!("MMap file {dax} l={}G", (length * Frame::SIZE) >> 30);
        llfree::mmap::file(begin, length, dax, true)
    }
    #[cfg(not(target_os = "linux"))]
    panic!("No NVRAM!")
}
