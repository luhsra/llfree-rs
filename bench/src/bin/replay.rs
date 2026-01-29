use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use bitfield_struct::bitfield;
use clap::Parser;
use llfree::frame::Frame;
use llfree::util::align_down;
use llfree::*;
use log::{info, warn};

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

    let size = std::fs::metadata(&trace)
        .expect("Failed accessing trace")
        .len() as usize;
    info!("Mapping {size} bytes from {trace:?}");
    let data = mmap::Mapping::<TracePage>::file(0, size / size_of::<TracePage>(), &trace, false)
        .expect("Failed opening trace");

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    let frames = (memory << 30) / Frame::SIZE;
    let meta = MetaData::alloc(Allocator::metadata_size(threads, frames));
    let alloc = Allocator::new(threads, frames, Init::FreeAll, meta).unwrap();
    alloc.validate();

    let barrier = Barrier::new(2);

    let start = Instant::now();
    let running = AtomicUsize::new(threads);

    warn!("start");

    let trace_pages: &[TracePage] = &data[..];

    let score = std::thread::scope(|s| {
        let monitor = s.spawn(|| {
            thread::pin(threads);
            let mut frag = frag.map(|path| BufWriter::new(File::create(path).unwrap()));

            barrier.wait();

            let mut frag_sec = 0;
            let mut score = 0.0;
            while running.load(Ordering::Relaxed) > 0 {
                let elapsed = start.elapsed();
                if elapsed.as_secs() > frag_sec {
                    if let Some(frag) = frag.as_mut() {
                        for i in 0..alloc.frames().div_ceil(1 << HUGE_ORDER) {
                            let free = alloc
                                .stats_at(FrameId(i << HUGE_ORDER), HUGE_ORDER)
                                .free_frames;
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

        s.spawn(|| {
            let mut allocated = HashMap::<u32, (FrameId, Flags)>::new();
            #[derive(Clone)]
            enum State {
                Init,
                Running { page: usize, index: usize },
                Done,
            }

            let mut curr_pages = vec![State::Init; threads];

            barrier.wait();

            loop {
                let mut next: Option<(usize, TraceEntry)> = None;
                for (t, page_idx) in curr_pages.iter_mut().enumerate() {
                    let entry = match page_idx {
                        State::Running { page, index }
                            if *index < trace_pages[*page].entries.len() =>
                        {
                            &trace_pages[*page].entries[*index]
                        }
                        State::Running { page, index } => {
                            if *page < trace_pages.len()
                                && let Some(next) = trace_pages[*page + 1..]
                                    .iter()
                                    .position(|p| p.cpuid as usize == t)
                            {
                                *index = 0;
                                *page += 1 + next;
                                &trace_pages[*page].entries[*index]
                            } else {
                                *page_idx = State::Done;
                                continue;
                            }
                        }
                        State::Init => {
                            if let Some(next) =
                                trace_pages.iter().position(|p| p.cpuid as usize == t)
                            {
                                *page_idx = State::Running {
                                    page: next,
                                    index: 0,
                                };
                                &trace_pages[next].entries[0]
                            } else {
                                *page_idx = State::Done;
                                continue;
                            }
                        }
                        State::Done => continue,
                    };
                    let entry = TraceEntry::from_bytes(*entry);
                    if next.is_none()
                        || entry.time() < next.as_ref().unwrap().1.time()
                        || (entry.alloc() && entry.time() <= next.as_ref().unwrap().1.time())
                    {
                        next = Some((t, entry));
                    }
                }
                if let Some((t, entry)) = next {
                    curr_pages[t] = match &curr_pages[t] {
                        State::Running { page, index } => State::Running {
                            page: *page,
                            index: *index + 1,
                        },
                        _ => unreachable!(),
                    };

                    if entry.alloc() {
                        let flags = flags_from_gfp(entry.flags(), entry.order() as usize);
                        let frame = alloc.get(t, None, flags).unwrap();
                        allocated.insert(entry.pfn(), (frame, flags));
                        continue;
                    }

                    // free
                    for order in entry.order() as usize..=MAX_ORDER {
                        let pfn = align_down(entry.pfn() as _, 1 << order);

                        if let Some((frame, flags)) = allocated.remove(&(pfn as u32)) {
                            // warn!("free pfn={pfn} order={order} flags_o={}", flags.order());
                            if !(flags.order() >= order) {
                                break;
                            }
                            alloc.put(entry.pfn() as _, frame, flags).unwrap();

                            for part in 0..(1 << (flags.order() - order)) {
                                if entry.pfn() as usize % (1 << (flags.order() - order)) != part {
                                    warn!("split free pfn={pfn}+{part} order={order}");
                                    allocated.insert(
                                        (pfn + part * (1 << flags.order())) as u32,
                                        (frame, flags.with_order(order)),
                                    );
                                }
                            }
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
        });

        monitor.join().unwrap()
    });

    alloc.validate();
    warn!("{alloc:#?}");
    warn!("score = {score:.1}")
}

#[bitfield(u128)]
struct TraceEntry {
    /// Time in microseconds -> max ~4500 hours
    #[bits(38)]
    time_us: u64,
    /// Page frame number -> max 32G with padding
    #[bits(24)]
    pfn: u32,
    /// Allocation (true) or free (false)
    alloc: bool,
    /// Allocation size order (0..=10)
    #[bits(4)]
    order: u8,
    /// GFP flags
    #[bits(29)]
    flags: u32,
    #[bits(32)]
    __: (),
}
impl TraceEntry {
    fn time(&self) -> f32 {
        self.time_us() as f32 / 1_000_000.0
    }
    fn from_bytes([a, b, c, d, e, f, g, h, i, j, k, l]: [u8; 12]) -> Self {
        Self::from_bits(u128::from_ne_bytes([
            a, b, c, d, e, f, g, h, i, j, k, l, 0, 0, 0, 0,
        ]))
    }
}

#[repr(C)]
struct TracePage {
    cpuid: u32,
    entries: [[u8; 12]; FRAME_SIZE / 12],
}
const _: () = assert!(size_of::<TracePage>() == FRAME_SIZE);

fn flags_from_gfp(gfp: u32, order: usize) -> Flags {
    const GFP_MOVABLE: u32 = 0x08;
    const GFP_ZERO: u32 = 0x100;
    const GFP_PAGE_CACHE: u32 = 0x10000000;
    Flags::o(order)
        .with_movable((gfp & GFP_MOVABLE) != 0)
        .with_zeroed((gfp & GFP_ZERO) != 0)
        .with_long_living((gfp & GFP_PAGE_CACHE) != 0)
}
