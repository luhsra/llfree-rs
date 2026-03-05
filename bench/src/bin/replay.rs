use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::slice;
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

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum Count {
    One,
    Core,
    Pid,
}

fn tiering(
    cores: usize,
    pids: usize,
) -> (
    Tiering<'static>,
    impl Fn(usize, u32, usize, usize) -> Request,
) {
    let tiers = vec![
        TierConfig::new(Tier(0), cores), // immovable frames
        TierConfig::new(Tier(1), pids),  // movable frames
        TierConfig::new(Tier(2), pids),  // page cache frames
        TierConfig::new(Tier(3), cores), // huge frames
    ];

    fn policy(requested: Tier, target: Tier, free: usize) -> Policy {
        if requested.0 > target.0 {
            return Policy::Steal;
        } else if requested.0 < target.0 {
            return Policy::Demote;
        }
        match free {
            f if f >= TREE_FRAMES / 2 => Policy::Match(1), // half free
            f if f >= TREE_FRAMES / 64 => Policy::Match(u8::MAX), // almost allocated
            _ => Policy::Match(2), // low free count -> causes frequent reservations
        }
    }

    fn request(
        order: usize,
        gfp: u32,
        core: usize,
        cores: usize,
        pid: usize,
        pids: usize,
    ) -> Request {
        if order >= HUGE_ORDER {
            Request::new(order, Tier(2), Some(core % cores + cores + pids + pids))
        } else if gfp == GFP::MOVABLE && gfp == GFP::PAGE_CACHE {
            Request::new(order, Tier(1), Some(pid % pids + cores + pids))
        } else if gfp == GFP::MOVABLE {
            Request::new(order, Tier(1), Some(pid % pids + cores))
        } else {
            Request::new(order, Tier(0), Some(core % cores))
        }
    }

    (
        Tiering {
            tiers: tiers.leak(),
            default: Tier(0),
            policy,
        },
        move |order, gfp, core, pid| request(order, gfp, core, cores, pid, pids),
    )
}

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
    let data = mmap::Mapping::<Page>::file(0, size / size_of::<Page>(), &trace, false)
        .expect("Failed opening trace");

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    let (header, trace_pages) = {
        let (header, trace_pages) = data.split_first().unwrap();
        let header = TraceHeader::from_page(header);

        let trace_pages: &[TracePage] =
            unsafe { slice::from_raw_parts(trace_pages.as_ptr().cast(), trace_pages.len()) };
        (header, trace_pages)
    };

    let frames = (header.max_pfn as usize + 1).next_multiple_of(1 << HUGE_ORDER);
    let cores = header.cores as usize;
    info!(
        "frames = {frames} threads = {cores} trace pages = {}",
        trace_pages.len()
    );

    let page_cache_allocs = trace_pages
        .iter()
        .flat_map(|p| p.entries.iter())
        .filter(|e| e.alloc() && e.flags() == GFP::PAGE_CACHE)
        .count();
    info!("page cache allocs = {page_cache_allocs}");

    let pids = 8;

    let (tiering, request) = tiering(cores, pids);
    let meta = MetaData::alloc(Allocator::metadata_size(&tiering, frames));
    let alloc = Allocator::new(frames, Init::FreeAll, &tiering, meta).unwrap();
    alloc.validate();

    let barrier = Barrier::new(2);

    let running = AtomicUsize::new(cores);

    info!("start");

    let mut allocated = vec![Allocation::new(); frames];

    let score = std::thread::scope(|s| {
        let monitor = s.spawn(|| {
            thread::pin(cores);
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

            let mut curr_pages = vec![State::Init; cores];

            barrier.wait();

            loop {
                let mut next: Option<(usize, TraceEntry)> = None;
                for (t, page_idx) in curr_pages.iter_mut().enumerate() {
                    let entry = match page_idx {
                        State::Running { page, index }
                            if *index < trace_pages[*page].entries.len() =>
                        {
                            trace_pages[*page].entries[*index]
                        }
                        State::Running { page, index } => {
                            if *page < trace_pages.len()
                                && let Some(next) = trace_pages[*page + 1..]
                                    .iter()
                                    .position(|p| p.cpuid as usize == t)
                            {
                                *index = 0;
                                *page += 1 + next;
                                trace_pages[*page].entries[*index]
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
                                trace_pages[next].entries[0]
                            } else {
                                *page_idx = State::Done;
                                continue;
                            }
                        }
                        State::Done => continue,
                    };
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
                        let flags = request(
                            entry.order() as usize,
                            entry.flags(),
                            t,
                            entry.pid() as usize,
                        );
                        let (_, frame) = alloc.get(None, flags).unwrap();
                        allocated[entry.pfn() as usize] =
                            Allocation::with(frame, entry.order() as _);
                        continue;
                    }

                    // free
                    for order in entry.order() as usize..=MAX_ORDER {
                        let pfn = align_down(entry.pfn() as _, 1 << order);

                        let allocation = &mut allocated[pfn];
                        if allocation.present() {
                            allocation.set_present(false);
                            let frame = allocation.frame();

                            if allocation.order() < order {
                                break;
                            }
                            let req = request(order, entry.flags(), t, entry.pid() as usize);
                            if let Err(e) = alloc.put(frame, req) {
                                error!(
                                    "failed to free pfn={pfn} order={order} flags_o={} error={e:?}",
                                    req.order
                                );
                            }

                            for part in 0..(1 << (req.order - order)) {
                                if frame.0 as usize % (1 << (req.order - order)) != part {
                                    info!("split free pfn={pfn}+{part} order={order}");
                                    allocated[pfn + part] = Allocation::with(
                                        FrameId((pfn + part * (1 << req.order)) as _),
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

#[repr(align(4096))]
struct Page([u8; FRAME_SIZE]);
const _: () = assert!(size_of::<Page>() == FRAME_SIZE);

#[repr(C)]
#[repr(align(4096))]
struct TraceHeader {
    pages: u32,
    cores: u32,
    max_pfn: u32,
}
impl TraceHeader {
    fn from_page(page: &Page) -> &Self {
        unsafe { &*page.0.as_ptr().cast() }
    }
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
    /// Current process ID
    pid: u32,
}
impl TraceEntry {
    fn time(&self) -> f32 {
        self.time_us() as f32 / 1_000_000.0
    }
}

#[repr(C, align(4096))]
struct TracePage {
    cpuid: u32,
    entries: [TraceEntry; (FRAME_SIZE - size_of::<u32>()) / size_of::<TraceEntry>()],
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
