#![feature(int_roundings)]

use core::result::Result;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

use clap::Parser;
use log::warn;

use nvalloc::lower::Cache;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::upper::{Alloc, AllocExt, Array, Init};
use nvalloc::util::{Page, WyRand};
use nvalloc::{thread, util};

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

type Allocator = Array<3, Cache<32>>;

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

fn initialize(memory: usize, dax: &String, threads: usize, crash: bool) {
    let mut mapping = map(0x1000_0000_0000, memory * PT_LEN * PT_LEN, dax).unwrap();
    let alloc = Allocator::new(threads, &mut mapping, Init::Overwrite, false).unwrap();
    warn!("Prepare alloc");
    thread::parallel(
        mapping[..alloc.pages()]
            .chunks(alloc.pages().div_ceil(threads))
            .enumerate(),
        |(t, chunk)| {
            let mut rng = WyRand::new(t as u64);
            let mut pages = chunk
                .iter()
                .map(|p| p as *const _ as u64)
                .collect::<Vec<_>>();
            rng.shuffle(&mut pages);

            let frees = chunk.len() / 2;

            for page in &pages[..frees] {
                alloc.put(t, *page, 0).unwrap()
            }

            let alloc_pages = &mut pages[frees..];
            if crash {
                loop {
                    let i = rng.range(0..alloc_pages.len() as _) as usize;
                    alloc.put(t, alloc_pages[i], 0).unwrap();
                    black_box(alloc_pages[i]);
                    alloc_pages[i] = alloc.get(t, 0).unwrap();
                }
            }
        },
    );
    assert!(!crash);
    let num_allocated = alloc.dbg_allocated_pages();
    warn!("Allocated: {num_allocated}");
    assert!(0 < num_allocated && num_allocated < alloc.pages());
}

fn recover(threads: usize, memory: usize, dax: &str) -> u128 {
    let mut mapping = map(0x1000_0000_0000, memory * PT_LEN * PT_LEN, &dax).unwrap();

    warn!("Recover alloc");
    let timer = Instant::now();
    let alloc = Allocator::new(threads, &mut mapping, Init::Recover, false).unwrap();
    let time = timer.elapsed().as_nanos();

    let num_alloc = alloc.dbg_allocated_pages();
    warn!("Recovered {num_alloc} allocations in {time} ns");
    let expected = alloc.pages() / 2;
    assert!(expected - threads <= num_alloc && num_alloc <= expected + threads);

    time
}

#[allow(unused_variables)]
fn map(begin: usize, length: usize, dax: &str) -> Result<MMap<Page>, ()> {
    #[cfg(target_os = "linux")]
    {
        warn!(
            "MMap file {dax} l={}G ({:x})",
            (length * std::mem::size_of::<Page>()) >> 30,
            length * std::mem::size_of::<Page>()
        );
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dax)
            .unwrap();
        return MMap::dax(begin, length, f);
    }
    #[cfg(not(target_os = "linux"))]
    unimplemented!("No NVRAM!")
}
