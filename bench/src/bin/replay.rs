use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::Parser;
use llfree::frame::Frame;
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

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    // TODO: replay allocations
    let frames = (memory << 30) / Frame::SIZE;
    let ms = Allocator::metadata_size(threads, frames);
    let meta = MetaData::alloc(ms);
    let alloc = Allocator::new(threads, frames, Init::FreeAll, meta).unwrap();
    alloc.validate();

    // Operate on half of the avaliable memory
    let barrier = Barrier::new(threads + 1);

    let start = Instant::now();
    let running = AtomicBool::new(true);

    warn!("start");

    let mut traces = vec![Vec::new(); threads];
    for (t, subtrace) in traces.iter_mut().enumerate() {
        subtrace.extend(trace.iter().skip(t).step_by(threads).copied());
    }

    let score = std::thread::scope(|s| {
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
                            let free = alloc.stats_at(i << HUGE_ORDER, HUGE_ORDER).free_frames;
                            let level = if free == 0 { 0 } else { 1 + free / 64 };
                            write!(frag, "{level:?}").unwrap();
                        }
                        writeln!(frag).unwrap();
                    }

                    let stats = alloc.stats();
                    let huge = stats.free_huge;
                    let optimal = stats.free_frames >> HUGE_ORDER;
                    let fraction = 100.0 * huge as f32 / optimal as f32;
                    warn!("free-huge {huge}/{optimal} = {fraction:.1}");
                    score += fraction;
                    frag_sec += 1;
                }
            }
            score / frag_sec as f32
        });

        thread::parallel(traces.into_iter().enumerate(), |(t, trace)| {
            thread::pin(t);
            let allocations = trace
                .iter()
                .filter(|op| matches!(op, Operation::Get(_) | Operation::GetMovable(_)))
                .count();
            let mut allocations = HashMap::with_capacity(allocations);

            barrier.wait();

            let mut alloc_idx = 0;
            for op in trace {
                match op {
                    Operation::Get(order) => {
                        let frame = alloc.get(t, None, Flags::o(order as _)).unwrap();
                        allocations.insert(alloc_idx, frame);
                        alloc_idx += 1;
                    }
                    Operation::GetMovable(order) => {
                        let frame = alloc
                            .get(order as _, None, Flags::o(order as _).with_movable(true))
                            .unwrap();
                        allocations.insert(alloc_idx, frame);
                        alloc_idx += 1;
                    }
                    Operation::Put(order, idx) => {
                        if let Some(frame) = allocations.remove(&(idx as usize)) {
                            let _ = alloc.put(t, frame, Flags::o(order as _));
                        }
                    }
                }
            }
            running.store(false, Ordering::Relaxed);
        });

        monitor.join().unwrap()
    });

    alloc.validate();
    warn!("{alloc:#?}");
    warn!("score = {score:.1}")
}

#[derive(Debug, Clone, Copy)]
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
