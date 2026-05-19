use core::{fmt, slice};
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use facet::Facet;
use figue::{self as args, FigueBuiltins};
use llfree::frame::Frame;
use llfree::util::{self, WyRand, aligned_buf};
use llfree::wrapper::NvmAlloc;
use llfree::{Alloc, FrameId, LLFree, Request, Result, TREE_ORDER, Tiering};
use llfree_eval::mmap::Mapping;
use llfree_eval::thread;
use llfree_eval::tiering::TieringConfig;
use log::warn;

/// Benchmarking the allocators against each other.
#[derive(Facet, Debug)]
struct Args {
    /// The benchmark to be executed.
    #[facet(args::positional)]
    bench: Benchmark,
    /// Names of the allocators to be benchmarked.
    #[facet(args::positional)]
    allocs: Vec<String>,
    /// Tested number of threads.
    #[facet(args::short, args::named, default = vec![1usize])]
    x: Vec<usize>,
    /// Max number of threads.
    #[facet(args::short, args::named, default = 6)]
    threads: usize,
    /// Where to store the benchmark results in csv format.
    #[facet(args::short, args::named, default = "bench/out/bench.csv")]
    outfile: String,
    /// DAX file to be used for the allocator.
    #[facet(args::named)]
    dax: Option<String>,
    /// Number of repetitions.
    #[facet(args::short, args::named, default = 1)]
    iterations: usize,
    /// Specifies how many pages should be allocated: #pages = 2^order
    #[facet(args::short = 's', args::named, default = vec![0usize])]
    order: Vec<usize>,
    /// Max amount of memory in GiB.
    #[facet(args::short, args::named, default = 16)]
    memory: usize,
    /// Use every n-th cpu.
    #[facet(args::named, default = 1)]
    stride: usize,
    /// Offset for the cpu selection.
    #[facet(args::named, default = 0)]
    offset: usize,
    /// Optional path to a tiering configuration file (JSON)
    #[facet(args::named)]
    tiering: Option<PathBuf>,

    #[facet(flatten)]
    builtins: FigueBuiltins,
}

static CORES: AtomicUsize = AtomicUsize::new(0);

fn main() {
    let Args {
        bench,
        allocs,
        x,
        threads,
        outfile,
        dax,
        iterations,
        order,
        memory,
        stride,
        offset,
        tiering,
        builtins: _,
    } = figue::from_std_args().unwrap();

    CORES.store(threads, Ordering::Relaxed);

    util::logging();

    if stride > 1 {
        thread::STRIDE.store(stride, Ordering::Relaxed);
    }
    if offset > 0 {
        thread::OFFSET.store(offset, Ordering::Relaxed);
    }

    assert!(memory >= 1);

    let mut out = File::create(outfile).unwrap();
    writeln!(out, "alloc,x,order,iteration,memory,{}", Perf::header()).unwrap();

    warn!("Allocating orders {order:?}");

    let mut mapping = mapping(0x1000_0000_0000, (memory << 30) / Frame::SIZE, dax);

    let (tiering, request): (Tiering, Box<dyn RequestFn>) = if let Some(tiering) = tiering {
        let tiering = std::fs::read_to_string(tiering).expect("Failed to read tiering config");
        let config =
            facet_json::from_str::<TieringConfig>(&tiering).expect("Failed to read tiering config");

        (
            config.tiering(threads),
            Box::new(move |order, core| config.request(order, core, threads, core, 0)),
        )
    } else {
        let (tiering, request) = Tiering::simple(threads);
        (tiering, Box::new(request))
    };

    for x in x {
        for o in order.iter().copied() {
            assert!(o <= TREE_ORDER);
            for name in &allocs {
                for i in 0..iterations {
                    warn!(">>> bench {bench:?} x={x} o={o} {name}\n");
                    let alloc = alloc(name, &mut mapping, &tiering, request.as_ref());
                    let perf = bench.run(&*alloc, o, threads, x);
                    writeln!(out, "{name},{x},{o},{i},{memory},{perf}").unwrap();
                }
            }
        }
    }
}

#[allow(unused_variables)]
pub fn mapping(begin: usize, length: usize, dax: Option<String>) -> Mapping<Frame> {
    #[cfg(target_os = "linux")]
    if let Some(file) = dax {
        warn!("MMap file {file} l={}G", (length * Frame::SIZE) >> 30);
        return Mapping::file(begin, length, &file, true).unwrap();
    }
    warn!("Mapping {length} frames ({} bytes).", length * Frame::SIZE);
    Mapping::anon(begin, length, false, false).unwrap()
}

trait RequestFn: Fn(usize, usize) -> Request + Send + Sync {}
impl<T: Fn(usize, usize) -> Request + Send + Sync> RequestFn for T {}

struct BenchAlloc<'a, T: Alloc<'a>, F: RequestFn> {
    alloc: T,
    request: F,
    __: core::marker::PhantomData<&'a ()>,
}
impl<'a, T: Alloc<'a>, F: RequestFn> fmt::Debug for BenchAlloc<'a, T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.alloc.fmt(f)
    }
}
impl<'a, T: Alloc<'a>, F: RequestFn> BenchAlloc<'a, T, F> {
    pub fn new(alloc: T, request: F) -> Self {
        Self {
            alloc,
            request,
            __: core::marker::PhantomData,
        }
    }
}

/// Reduced, VTable-compatible alloc trait for dynamic dispatch
trait DynAlloc: fmt::Debug + Send + Sync {
    fn get(&self, core: usize, order: usize) -> Result<FrameId>;
    fn put(&self, core: usize, frame: FrameId, order: usize) -> Result<()>;

    fn frames(&self) -> usize;
    fn allocated_frames(&self) -> usize;
}

impl<'a, T: Alloc<'a>, F: RequestFn> DynAlloc for BenchAlloc<'a, T, F> {
    fn get(&self, core: usize, order: usize) -> Result<FrameId> {
        self.alloc
            .get(None, (self.request)(order, core))
            .map(|(f, _)| f)
    }
    fn put(&self, core: usize, frame: FrameId, order: usize) -> Result<()> {
        self.alloc.put(frame, (self.request)(order, core))
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn allocated_frames(&self) -> usize {
        self.alloc.frames() - self.alloc.tree_stats().free_frames
    }
}

fn alloc<'a>(
    name: &str,
    zone: &'a mut [Frame],
    tiering: &Tiering,
    request: impl RequestFn + 'a,
) -> Box<dyn DynAlloc + 'a> {
    #[cfg(feature = "llc")]
    if llfree_eval::LLC::name() == name {
        let m = NvmAlloc::<llfree_eval::LLC>::metadata_size(tiering, zone.len());
        let local = aligned_buf(m.local);
        let trees = aligned_buf(m.trees);
        return Box::new(BenchAlloc::new(
            NvmAlloc::<llfree_eval::LLC>::create(zone, false, tiering, local, trees).unwrap(),
            request,
        ));
    }
    if LLFree::name() == name {
        let m = NvmAlloc::<LLFree>::metadata_size(tiering, zone.len());
        let local = aligned_buf(m.local);
        let trees = aligned_buf(m.trees);
        return Box::new(BenchAlloc::new(
            NvmAlloc::<LLFree>::create(zone, false, tiering, local, trees).unwrap(),
            request,
        ));
    }
    panic!("Unknown allocator");
}

#[derive(Debug, Clone, Copy, Facet)]
#[repr(u8)]
#[facet(rename_all = "snake_case")]
enum Benchmark {
    /// Allocate half the memory at once and free it afterwards
    Bulk,
    /// Initially allocate half the memory and then repeatedly allocate and free the same page
    Repeat,
    /// Initially allocate half the memory and then repeatedly free an random page
    /// and replace it with a newly allocated one
    Rand,
}

impl Benchmark {
    fn run(self, alloc: &dyn DynAlloc, order: usize, threads: usize, x: usize) -> Perf {
        match self {
            Benchmark::Bulk => bulk(alloc, order, threads, x),
            Benchmark::Repeat => repeat(alloc, order, threads, x),
            Benchmark::Rand => rand(alloc, order, threads, x),
        }
    }
}

fn bulk(alloc: &dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();
    let allocs = alloc.frames() / max_threads / 2 / (1 << order);

    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        let mut get = 0;
        let mut put = 0;

        let total = Instant::now();

        let mut pages = Vec::with_capacity(allocs);

        for _ in 0..(1 << order) {
            thread::pin(t);

            barrier.wait();
            let timer = Instant::now();
            for _ in 0..allocs {
                pages.push(alloc.get(t, order).unwrap());
            }
            get += timer.elapsed().as_nanos() / allocs as u128;

            barrier.wait();
            let timer = Instant::now();
            while let Some(page) = pages.pop() {
                alloc.put(t, page, order).unwrap();
            }
            put += timer.elapsed().as_nanos() / allocs as u128;
        }
        get /= 1 << order;
        put /= 1 << order;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: total.elapsed().as_millis(),
            allocs: allocs * (1 << order),
        }
    }));
    assert_eq!(alloc.allocated_frames(), 0, "{alloc:?}");

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn repeat(alloc: &dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.frames() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);
    let mut perf = Perf::avg(thread::parallel(0..threads, |t| {
        thread::pin(t);
        for _ in 0..allocs {
            alloc.get(t, order).unwrap();
        }

        barrier.wait();
        let timer = Instant::now();
        for _ in 0..(1 << order) {
            for _ in 0..allocs {
                let page = alloc.get(t, order).unwrap();
                let page = black_box(page);
                alloc.put(t, page, order).unwrap();
            }
        }

        let realloc = timer.elapsed().as_nanos() / (allocs * (1 << order)) as u128;
        Perf {
            get_min: realloc,
            get_avg: realloc,
            get_max: realloc,
            put_min: realloc,
            put_avg: realloc,
            put_max: realloc,
            init: 0,
            total: timer.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.allocated_frames(), allocs * threads * (1 << order));

    perf.init = init;
    perf.allocs = allocs;
    perf
}

fn rand(alloc: &dyn DynAlloc, order: usize, max_threads: usize, threads: usize) -> Perf {
    assert!(threads <= max_threads);
    let timer = Instant::now();
    let init = timer.elapsed().as_millis();

    let allocs = alloc.frames() / max_threads / 2 / (1 << order);
    let barrier = Barrier::new(threads);

    let mut all_pages = vec![FrameId(0); allocs * threads];
    let all_pages_ptr = all_pages.as_mut_ptr() as usize;

    let chunks = all_pages.chunks_mut(allocs).enumerate();
    let mut perf = Perf::avg(thread::parallel(chunks, |(t, pages)| {
        thread::pin(t);

        let timer = Instant::now();
        let mut get = 0;
        let mut put = 0;
        for _ in 0..(1 << order) {
            barrier.wait();
            let timer = Instant::now();
            for page in pages.iter_mut() {
                *page = alloc.get(t, order).unwrap();
            }
            get += timer.elapsed().as_nanos() / allocs as u128;

            barrier.wait();
            if t == 0 {
                assert_eq!(pages.len(), allocs);
                assert_eq!(alloc.allocated_frames(), allocs * threads * (1 << order));
                // Shuffle between all cores
                let pages = unsafe {
                    slice::from_raw_parts_mut(all_pages_ptr as *mut u64, allocs * threads)
                };
                WyRand::new(42).shuffle(pages);
            }
            barrier.wait();

            let timer = Instant::now();
            for page in pages.iter() {
                alloc.put(t, *page, order).unwrap();
            }
            put += timer.elapsed().as_nanos() / allocs as u128;
        }
        get /= 1 << order;
        put /= 1 << order;

        Perf {
            get_min: get,
            get_avg: get,
            get_max: get,
            put_min: put,
            put_avg: put,
            put_max: put,
            init: 0,
            total: timer.elapsed().as_millis(),
            allocs,
        }
    }));
    assert_eq!(alloc.allocated_frames(), 0);

    perf.init = init;
    perf.allocs = allocs;
    perf
}

#[derive(Debug)]
struct Perf {
    get_min: u128,
    get_avg: u128,
    get_max: u128,
    put_min: u128,
    put_avg: u128,
    put_max: u128,
    init: u128,
    total: u128,
    allocs: usize,
}

impl Default for Perf {
    fn default() -> Self {
        Self {
            get_min: u128::MAX,
            get_avg: 0,
            get_max: 0,
            put_min: u128::MAX,
            put_avg: 0,
            put_max: 0,
            init: 0,
            total: 0,
            allocs: 0,
        }
    }
}

impl Perf {
    fn avg(iter: impl IntoIterator<Item = Perf>) -> Perf {
        let mut res = Perf::default();
        let mut counter = 0;
        for p in iter {
            res.get_min = res.get_min.min(p.get_min);
            res.get_avg += p.get_avg;
            res.get_max = res.get_max.max(p.get_max);
            res.put_min = res.put_min.min(p.put_min);
            res.put_avg += p.put_avg;
            res.put_max = res.put_max.max(p.put_max);
            res.init += p.init;
            res.total += p.total;
            res.allocs += p.allocs;
            counter += 1;
        }
        assert!(counter > 0);
        res.get_avg /= counter;
        res.put_avg /= counter;
        res.init /= counter;
        res.total /= counter;
        res.allocs /= counter as usize;
        res
    }
    fn header() -> &'static str {
        "get_min,get_avg,get_max,put_min,put_avg,put_max,init,total,allocs"
    }
}

impl fmt::Display for Perf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Perf {
            get_min,
            get_avg,
            get_max,
            put_min,
            put_avg,
            put_max,
            init,
            total,
            allocs,
        } = self;
        write!(
            f,
            "{get_min},{get_avg},{get_max},{put_min},{put_avg},{put_max},{init},{total},{allocs}"
        )
    }
}
