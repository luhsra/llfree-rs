use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::atomic::Ordering;

use bitfield_struct::bitfield;
use facet::Facet;
use figue::{self as args, FigueBuiltins};
use llfree::util::align_down;
use llfree::*;
use llfree_eval::gfp::GFP;
use llfree_eval::tiering::TieringConfig;
use llfree_eval::{mmap, thread};
use log::{error, info, trace, warn};

/// Benchmarking the allocators against each other.
#[derive(Facet, Debug)]
struct Args {
    #[facet(args::positional)]
    trace: PathBuf,
    /// Using only every n-th CPU
    #[facet(args::named, default = 2)]
    stride: usize,
    /// Monitor and output fragmentation
    #[facet(args::named)]
    frag: Option<PathBuf>,
    /// Time interval is ms for monitoring fragmentation
    #[facet(args::named, default = 100000)]
    interval: usize,
    /// Optional path to a tiering configuration file (JSON)
    #[facet(args::named)]
    tiering: Option<PathBuf>,

    #[facet(flatten)]
    builtins: FigueBuiltins,
}

cfg_select! {
    feature = "llc" => {
        use llfree_eval::LLC;
        type Allocator = LLC;
    }
    _ => {
        type Allocator<'a> = LLFree<'a>;
    }
}

fn main() {
    util::logging();

    let Args {
        trace,
        stride,
        frag,
        interval,
        tiering,
        builtins: _,
    } = figue::from_std_args().unwrap();

    // `thread::pin` uses this to select every nth cpu
    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }

    let ParsedTrace {
        max_pfn,
        cores,
        events,
    } = ParsedTrace::parse(&trace);

    let page_cache_allocs = events
        .iter()
        .filter(|e| e.alloc && e.flags == GFP::PAGE_CACHE)
        .count();
    info!("page cache allocs = {page_cache_allocs}");

    #[allow(clippy::type_complexity)]
    let (tiering, request): (
        Tiering,
        Box<dyn Fn(usize, u32, usize, usize) -> Request + Send>,
    ) = if let Some(tiering) = tiering {
        let s = fs::read_to_string(tiering).unwrap();
        let tiering_config: TieringConfig = facet_json::from_str(&s).unwrap();
        info!("tiering = {tiering_config:?}");
        (
            tiering_config.tiering(cores),
            Box::new(move |order, gfp, core, pid| {
                tiering_config.request(order, core, cores, pid, gfp)
            }),
        )
    } else {
        let (tiering, request) = Tiering::movable(cores);
        (
            tiering,
            Box::new(move |order, gfp, core, _pid| request(order, core, GFP::MOVABLE == gfp)),
        )
    };
    let meta = MetaData::alloc(&Allocator::metadata_size(&tiering, max_pfn));
    let llfree = Allocator::new(max_pfn, Init::FreeAll, &tiering, meta).unwrap();
    llfree.validate();

    info!("start");

    let mut allocated = vec![Allocation::new(); max_pfn];

    let ops = events.len();
    let mut reallocs = 0;
    let mut free_unkown = 0;

    let score = {
        let mut frag = frag.map(|path| BufWriter::new(File::create(path).unwrap()));
        let mut score = 0.0;
        let mut count = 0;

        for (i, entry) in events.into_iter().enumerate() {
            if i % interval == 0 {
                score += measure(frag.as_mut(), &llfree);
                count += 1;
            }

            let pfn = entry.pfn as usize;
            let flags = request(
                entry.order as usize,
                entry.flags,
                entry.cpuid as usize,
                entry.pid as usize,
            );
            if entry.alloc {
                let (frame, _) = llfree.get(None, flags).unwrap();
                if allocated[pfn].present() {
                    trace!("Realloc pfn={pfn} order={}", flags.order);
                    reallocs += 1;
                }
                allocated[pfn] = Allocation::with(frame, entry.order as _);
                continue;
            }

            let mut found = None;
            for order in entry.order as usize..=TREE_ORDER {
                let pfn = align_down(pfn, 1 << order);
                if allocated[pfn].present() && allocated[pfn].order() >= order {
                    found = Some(pfn);
                    break;
                }
            }

            // free
            if let Some(a_pfn) = found {
                let allocation = &mut allocated[a_pfn];
                assert!(allocation.present());
                let frame = allocation.frame();
                let order = allocation.order();
                assert!(order >= entry.order as usize);

                // Mark as free and split if alloc was larger
                for part in 0..(1 << (order - entry.order as usize)) {
                    let part_pfn = a_pfn + part * (1 << entry.order as usize);
                    let part_frame = FrameId(frame.0 + part * (1 << entry.order as usize));

                    if pfn != part_pfn {
                        info!("Split pfn={part_pfn}");
                    }

                    allocated[part_pfn] = Allocation::new()
                        .with_present(pfn != part_pfn)
                        .with_frame(part_frame)
                        .with_order(entry.order as _);
                }

                if let Err(e) = llfree.put(frame, flags) {
                    error!("Free failed pfn={a_pfn} order={} error={e:?}", flags.order);
                }
            } else {
                trace!("Free unallocated pfn={pfn} order={}", flags.order);
                free_unkown += 1;
            }
        }

        score += measure(frag.as_mut(), &llfree);
        count += 1;

        score / count as f32
    };

    info!("{llfree:#?}");

    llfree.validate();
    let stats = llfree.stats();
    info!("free = {} / {}", stats.free_frames, llfree.frames());
    let huge = llfree.frames() >> HUGE_ORDER;
    info!("huge = {} / {huge}", stats.free_huge);

    info!(
        "reallocs = {} ({:.2}%)",
        reallocs,
        reallocs as f64 * 100.0 / ops as f64
    );
    info!(
        "free_unkown = {} ({:.2}%)",
        free_unkown,
        free_unkown as f64 * 100.0 / ops as f64
    );

    warn!("score = {score:.1}");

    println!(
        "{}",
        facet_json::to_string_pretty(&Output {
            free_frames: stats.free_frames,
            total_frames: llfree.frames(),
            free_huge: stats.free_huge,
            total_huge: huge,
            score,
        })
        .unwrap()
    );
}

#[derive(Facet, Debug, Default)]
struct Output {
    free_frames: usize,
    total_frames: usize,
    free_huge: usize,
    total_huge: usize,
    score: f32,
}

fn measure(frag: Option<&mut BufWriter<File>>, alloc: &Allocator) -> f32 {
    if let Some(frag) = frag {
        for i in 0..alloc.frames().div_ceil(1 << HUGE_ORDER) {
            let free = alloc.stats_at(HugeId(i).as_frame(), HUGE_ORDER).free_frames;
            let level = if free == 0 { 0 } else { 1 + free / 64 };
            write!(frag, "{level}").unwrap();
        }
        writeln!(frag).unwrap();
    }

    let stats = alloc.stats();
    let huge = stats.free_huge;
    let optimal = stats.free_frames >> HUGE_ORDER;
    let fraction = 100.0 * huge as f32 / optimal as f32;
    warn!("free-huge {huge}/{optimal} = {fraction:.1}");
    fraction
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
    entries: [TraceEntry; Self::ENTRIES],
}
impl TracePage {
    const ENTRIES: usize = (FRAME_SIZE - size_of::<u32>()) / size_of::<TraceEntry>();
}
const _: () = assert!(size_of::<TracePage>() == FRAME_SIZE);

struct ParsedTrace {
    max_pfn: usize,
    cores: usize,
    events: Vec<ParsedEntry>,
}

impl ParsedTrace {
    fn parse(path: &Path) -> ParsedTrace {
        let size = std::fs::metadata(path)
            .expect("Failed accessing trace")
            .len() as usize;
        info!("Mapping {size} bytes from {path:?}");
        let data = mmap::Mapping::<Page>::file(0, size / size_of::<Page>(), path, false)
            .expect("Failed opening trace");

        let (header, t_pages) = {
            let (header, t_pages) = data.split_first().unwrap();
            let header = TraceHeader::from_page(header);

            let t_pages: &[TracePage] =
                unsafe { slice::from_raw_parts(t_pages.as_ptr().cast(), t_pages.len()) };
            (header, t_pages)
        };

        let max_pfn = (header.max_pfn as usize + 1).next_multiple_of(1 << HUGE_ORDER);
        let cores = header.cores as usize;
        info!(
            "max_pfn = {max_pfn} threads = {cores} trace pages = {}",
            t_pages.len()
        );

        let events = events_from_trace(t_pages);
        drop(data);
        ParsedTrace {
            max_pfn,
            cores,
            events,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ParsedEntry {
    alloc: bool,
    pfn: u32,
    cpuid: u8,
    order: u8,
    flags: u32,
    time: f32,
    pid: u32,
}

/// Parses a trace buffer into a sorted list of events.
fn events_from_trace(buffer: &[TracePage]) -> Vec<ParsedEntry> {
    let mut events = Vec::with_capacity(buffer.len() * TracePage::ENTRIES);
    events.clear();
    for page in buffer {
        for entry in page.entries {
            if entry.pfn() == 0 {
                break;
            }
            events.push(ParsedEntry {
                alloc: entry.alloc(),
                pfn: entry.pfn(),
                cpuid: page.cpuid as _,
                order: entry.order(),
                flags: entry.flags(),
                time: entry.time(),
                pid: entry.pid(),
            });
        }
    }
    events.sort_by(|a, b| a.time.total_cmp(&b.time));
    events
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
