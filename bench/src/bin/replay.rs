#![feature(int_roundings)]
#![feature(allocator_api)]
#![feature(new_uninit)]
#![feature(let_chains)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use clap::Parser;
use llfree::*;
use log::warn;

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    trace: PathBuf,
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
    /// Monitor and output fragmentation
    #[arg(long)]
    frag: Option<PathBuf>,
}

#[cfg(feature = "llc")]
type Allocator = LLC;
#[cfg(not(feature = "llc"))]
type Allocator<'a> = LLFree<'a>;

fn main() {
    util::logging();

    let Args {
        trace,
        threads,
        order,
        time,
        memory,
        stride,
        frag,
    } = Args::parse();

    let trace = parse_trace(&trace);
    warn!(
        "trace: (l={}, a={}, m={}, f={})",
        trace.len(),
        trace
            .iter()
            .filter(|op| matches!(op, Operation::Get(_)))
            .count(),
        trace
            .iter()
            .filter(|op| matches!(op, Operation::GetMovable(_)))
            .count(),
        trace
            .iter()
            .filter(|op| matches!(op, Operation::Put(_, _)))
            .count()
    );
    let mut lifetime = Vec::new();
    for op in &trace {
        match op {
            Operation::Get(_) => {
                lifetime.push(trace.len() as u32);
            }
            Operation::GetMovable(_) => {
                lifetime.push(trace.len() as u32);
            }
            Operation::Put(_, idx) => {
                lifetime[*idx as usize] = lifetime.len() as u32 - *idx;
            }
        }
    }
    warn!(
        "lifetime: l={} 0={} >={} avg={:.3}",
        lifetime.len(),
        lifetime.iter().filter(|l| **l == 0).count(),
        lifetime
            .iter()
            .filter(|l| **l >= trace.len() as u32)
            .count(),
        lifetime
            .iter()
            .filter(|l| **l < trace.len() as u32)
            .map(|x| *x as usize)
            .sum::<usize>() as f64
            / lifetime.len() as f64
    );

    if let Some(frag) = frag {
        let mut frag = BufWriter::new(File::create(frag).unwrap());
        for l in lifetime {
            frag.write(&l.to_ne_bytes()).unwrap();
        }
    }

    // TODO: replay allocations
}

enum Operation {
    Get(u8),
    GetMovable(u8),
    Put(u8, u32),
}

fn parse_trace(trace: &PathBuf) -> Vec<Operation> {
    let mut trace = File::open(trace).unwrap();
    let mut buf = vec![0; 4096];
    let mut allocated = HashMap::new();

    let mut allocations = Vec::new();
    let mut alloc_idx = 0;
    loop {
        let len = trace.read(&mut buf[..]).unwrap();

        buf.resize(len, 0);
        for alloc in buf.chunks(4) {
            let op = u32::from_ne_bytes([alloc[0], alloc[1], alloc[2], alloc[3]]);

            let kind = op >> 30;
            let order = (op >> 24) & 0x3f;
            let frame = op & 0x00ff_ffff;

            match kind {
                0 => {
                    allocated.insert(frame, alloc_idx);
                    alloc_idx += 1;
                    allocations.push(Operation::Get(order as _))
                }
                1 => {
                    allocated.insert(frame, alloc_idx);
                    alloc_idx += 1;
                    allocations.push(Operation::GetMovable(order as _))
                }
                2 => {
                    if let Some(alloc_idx) = allocated.remove(&frame) {
                        allocations.push(Operation::Put(order as _, alloc_idx));
                    }
                }
                _ => panic!("unknown operation"),
            };
        }

        if len == 0 {
            break;
        }
    }
    allocations
}
