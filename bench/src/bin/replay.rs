use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use bitfield_struct::bitfield;
use clap::Parser;
use llfree::util::align_down;
use llfree::*;
use log::{error, info, warn};

/// Benchmarking the allocators against each other.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    trace: PathBuf,
    /// Using only every n-th CPU
    #[arg(long, default_value_t = 2)]
    stride: usize,
    /// Monitor and output fragmentation
    #[arg(long)]
    frag: Option<PathBuf>,
    /// Time interval is ms for monitoring fragmentation
    #[arg(long, default_value_t = 1000)]
    interval: u64,
}

#[cfg(feature = "llc")]
type Allocator = LLC;
#[cfg(not(feature = "llc"))]
type Allocator<'a> = LLFree<'a>;

fn main() {
    util::logging();

    let Args {
        trace,
        stride,
        frag,
        interval,
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

    let trace_pages: &[TracePage] = &data[..];
    let max_pfn = trace_pages
        .iter()
        .flat_map(|p| p.entries.iter())
        .map(|e| TraceEntry::from_bytes(*e).pfn())
        .max()
        .unwrap_or(0);
    let frames = (max_pfn as usize + 1).next_multiple_of(1 << HUGE_ORDER);

    let page_cache_allocs = trace_pages
        .iter()
        .flat_map(|p| p.entries.iter())
        .map(|e| TraceEntry::from_bytes(*e))
        .filter(|e| e.alloc() && e.flags() == GFP::PAGE_CACHE)
        .count();
    info!("page cache allocs = {page_cache_allocs}");

    let max_cpu = trace_pages.iter().map(|p| p.cpuid).max().unwrap_or(0);
    info!("max pfn = {max_pfn} max cpu = {max_cpu}");
    let threads = max_cpu as usize + 1;

    let kinds = [
        // Immovable
        KindDesc(Kind(0), threads as _),
        // Movable
        KindDesc(Kind(1), threads as _),
        // Page cache
        KindDesc(Kind(2), 1),
        // Huge
        KindDesc(Kind::HUGE, threads as _),
    ];

    let meta = MetaData::alloc(Allocator::metadata_size(&kinds, frames));
    let alloc = Allocator::new(&kinds, frames, Init::FreeAll, meta).unwrap();
    let llf = |order: usize, mut core: usize, gfp: u32| {
        core %= threads;
        let local = match (order, core, gfp) {
            (HUGE_ORDER..=MAX_ORDER, core, _) => core + 3 * threads + 1,
            (_, _, gfp) if gfp == GFP::PAGE_CACHE => 2 * threads,
            (_, core, gfp) if gfp == GFP::MOVABLE => core + threads,
            _ => core,
        };
        Flags::with(order, local)
    };
    alloc.validate();

    let barrier = Barrier::new(2);

    let running = AtomicUsize::new(threads);

    info!("start");

    let mut allocated = vec![Allocation::new(); frames];

    let score = std::thread::scope(|s| {
        let monitor = s.spawn(|| {
            thread::pin(threads);
            let mut frag = frag.map(|path| BufWriter::new(File::create(path).unwrap()));

            barrier.wait();

            let start = Instant::now();
            let mut count = 0;
            let mut score = 0.0;
            while running.load(Ordering::Relaxed) > 0 {
                if let Some(frag) = frag.as_mut() {
                    for i in 0..alloc.frames().div_ceil(1 << HUGE_ORDER) {
                        let free = alloc.stats_at(HugeId(i).as_frame(), HUGE_ORDER).free_frames;
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
                count += 1;

                let elapsed_ms = start.elapsed().as_millis();
                let next_ms = count as u128 * interval as u128;
                if next_ms > elapsed_ms {
                    sleep(Duration::from_millis((next_ms - elapsed_ms) as u64));
                }
            }
            score / count as f32
        });

        s.spawn(|| {
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
                        let flags = llf(entry.order() as usize, t, entry.flags());
                        let frame = alloc.get(None, flags).unwrap();
                        allocated[entry.pfn() as usize] =
                            Allocation::with(frame, entry.order() as _);
                        continue;
                    }

                    // free
                    for order in entry.order() as usize..=MAX_ORDER {
                        let pfn = align_down(entry.pfn() as _, 1 << order);

                        let entry = &mut allocated[pfn];
                        if entry.present() {
                            entry.set_present(false);
                            let frame = entry.frame();

                            if entry.order() < order {
                                break;
                            }
                            let flags = llf(order, t, 0);
                            if let Err(e) = alloc.put(frame, flags) {
                                error!(
                                    "failed to free pfn={pfn} order={order} flags_o={} error={e:?}",
                                    flags.order()
                                );
                            }

                            for part in 0..(1 << (flags.order() - order)) {
                                if frame.0 as usize % (1 << (flags.order() - order)) != part {
                                    info!("split free pfn={pfn}+{part} order={order}");
                                    allocated[pfn + part] = Allocation::with(
                                        FrameId((pfn + part * (1 << flags.order())) as _),
                                        order,
                                    );
                                }
                            }
                            break;
                        }
                    }
                } else {
                    running.store(0, Ordering::Relaxed);
                    break;
                }
            }
        });

        monitor.join().unwrap()
    });

    alloc.validate();
    let stats = alloc.stats();
    info!("free = {} / {}", stats.free_frames, alloc.frames());
    let huge = alloc.frames() >> HUGE_ORDER;
    info!("huge = {} / {huge}", stats.free_huge);
    warn!("score = {score:.1}")
}

#[bitfield(u32)]
struct Allocation {
    present: bool,
    /// Allocated frame number
    #[bits(27)]
    frame: FrameId,
    /// Allocation size order (0..=10)
    #[bits(4)]
    order: usize,
}
impl Allocation {
    fn with(frame: FrameId, order: usize) -> Self {
        Self::new()
            .with_present(true)
            .with_frame(frame)
            .with_order(order)
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms, non_camel_case_types, dead_code)]
enum GFP {
    DMA = 0x01,
    HIGHMEM = 0x02,
    DMA32 = 0x04,
    MOVABLE = 0x08,
    RECLAIMABLE = 0x10,
    HIGH = 0x20,
    IO = 0x40,
    FS = 0x80,
    ZERO = 0x100,
    ATOMIC = 0x200,
    DIRECT_RECLAIM = 0x400,
    KSWAPD_RECLAIM = 0x800,
    WRITE = 0x1000,
    NOWARN = 0x2000,
    RETRY_MAYFAIL = 0x4000,
    NOFAIL = 0x8000,
    NORETRY = 0x10000,
    MEMALLOC = 0x20000,
    COMP = 0x40000,
    NOMEMALLOC = 0x80000,
    HARDWALL = 0x100000,
    THISNODE = 0x200000,
    ACCOUNT = 0x400000,
    ZEROTAGS = 0x800000,
    PAGE_CACHE = 0x10000000,
}
impl PartialEq<u32> for GFP {
    fn eq(&self, other: &u32) -> bool {
        (*self as u32) & *other != 0
    }
}
impl PartialEq<GFP> for u32 {
    fn eq(&self, other: &GFP) -> bool {
        *self & (*other as u32) != 0
    }
}
