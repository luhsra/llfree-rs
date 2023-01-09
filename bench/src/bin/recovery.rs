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
    /// Simulate a crash by changing counters and setting the active bit.
    #[arg(short, long)]
    crash: bool,
    /// Initialize and run the allocator
    #[arg(long)]
    init: bool,
    /// Where to store the benchmark results in csv format.
    #[arg(short, long, default_value = "")]
    outfile: String,
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
    } = Args::parse();

    assert!(memory > 0 && threads > 0);

    if init {
        initialize(memory, &dax, threads, crash);
    } else {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args([
            "--init",
            "--dax",
            &dax,
            &format!("-m{memory}"),
            &format!("-t{threads}"),
        ]);
        if crash {
            command.arg("--crash");
        }
        let mut process = command.spawn().unwrap();
        if crash {
            std::thread::sleep(Duration::from_secs(5));
            process.kill().unwrap();
        } else {
            let ret = process.wait().unwrap();
            assert!(ret.success());
        }

        recover(threads, memory, &dax, &outfile);
    }
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
            for page in pages.into_iter().take(frees) {
                alloc.put(t, page, 0).unwrap()
            }

            if crash {
                loop {
                    let page = alloc.get(t, 0).unwrap();
                    black_box(page);
                    alloc.put(t, page, 0).unwrap();
                }
            }
        },
    );
    assert!(!crash);
    let num_allocated = alloc.dbg_allocated_pages();
    warn!("Allocated: {num_allocated}");
    assert!(0 < num_allocated && num_allocated < alloc.pages());
}

fn recover(threads: usize, memory: usize, dax: &str, outfile: &str) {
    let mut mapping = map(0x1000_0000_0000, memory * PT_LEN * PT_LEN, &dax).unwrap();

    warn!("Recover alloc");
    let timer = Instant::now();
    let alloc = Allocator::new(threads, &mut mapping, Init::Recover, false).unwrap();
    let time = timer.elapsed().as_nanos();

    let num_alloc = alloc.dbg_allocated_pages();
    warn!("Recovered {num_alloc} allocations in {time} ns");
    let expected = alloc.pages() / 2;
    assert!(expected - threads <= num_alloc && num_alloc <= expected + threads);

    let mut out = File::create(outfile).unwrap();
    writeln!(out, "recovery\n{time}").unwrap();
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
