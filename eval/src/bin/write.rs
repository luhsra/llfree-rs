use std::time::Instant;

use facet::Facet;
use figue::{self as args, FigueBuiltins};
use llfree::frame::Frame;
use llfree::util::{WyRand, logging};
use llfree_eval::mmap::{MAdvise, Mapping, madvise};
use llfree_eval::{avg_bounds, thread};

/// Benchmarking the page-fault performance of a mapped memory region.
#[derive(Facet, Debug)]
struct Args {
    /// Number of threads
    #[facet(args::short, args::named, default = 6)]
    threads: usize,
    /// Max amount of memory in GiB. Is by the max thread count
    #[facet(args::short, args::named, default = 16)]
    memory: usize,
    /// DAX file to be used for the allocator
    #[facet(args::named)]
    dax: Option<String>,
    /// Create a private mapping (incompatible with `--dax`)
    #[facet(args::named)]
    private: bool,
    /// Populate on mmap
    #[facet(args::named)]
    populate: bool,
    /// Use hugepages
    #[facet(args::named)]
    huge: bool,
    /// Allocate randomly
    #[facet(args::named)]
    rand: bool,

    #[facet(flatten)]
    builtins: FigueBuiltins,
}

/// Trust me, I really want to send this.
#[repr(transparent)]
struct DoSend<T>(T);
unsafe impl<T> Send for DoSend<T> {}

fn main() {
    let Args {
        threads,
        memory,
        dax,
        private,
        populate,
        #[allow(unused)]
        huge,
        rand,
        builtins: _,
    } = figue::from_std_args().unwrap();

    logging();

    assert!(threads > 0 && memory > 0);

    let t_map = Instant::now();
    let mut mapping = mapping(
        0x1000_0000_0000,
        (memory << 30) / Frame::SIZE,
        dax,
        private,
        populate,
    );
    let t_map = t_map.elapsed().as_millis();

    #[cfg(target_os = "linux")]
    madvise(
        &mut mapping,
        if huge {
            MAdvise::Hugepage
        } else {
            MAdvise::NoHugepage
        },
    );

    let chunk_size = mapping.len().div_ceil(threads);

    let mut pages = mapping.iter_mut().map(DoSend).collect::<Vec<_>>();
    if rand {
        WyRand::new(42).shuffle(&mut pages);
    }
    let times = thread::parallel(pages.chunks_mut(chunk_size), |chunk| {
        let timer = Instant::now();
        for page in chunk {
            *page.0.cast_mut::<usize>() = 1;
        }
        timer.elapsed().as_millis()
    });

    let (t_amin, t_aavg, t_amax) = avg_bounds(times).unwrap_or_default();

    // Measure freeing pages
    let times = thread::parallel(mapping.chunks_mut(chunk_size), |chunk| {
        let timer = Instant::now();
        madvise(chunk, MAdvise::DontNeed);
        timer.elapsed().as_millis()
    });

    let (t_fmin, t_favg, t_fmax) = avg_bounds(times).unwrap_or_default();

    let t_unmap = Instant::now();
    drop(mapping);
    let t_unmap = t_unmap.elapsed().as_millis();

    println!("map,amin,aavg,amax,fmin,favg,fmax,unmap");
    println!("{t_map},{t_amin},{t_aavg},{t_amax},{t_fmin},{t_favg},{t_fmax},{t_unmap}");
}

#[allow(unused_variables)]
pub fn mapping(
    begin: usize,
    length: usize,
    dax: Option<String>,
    private: bool,
    populate: bool,
) -> Mapping<Frame> {
    #[cfg(target_os = "linux")]
    if let Some(file) = dax {
        log::warn!("MMap file {file} l={}G", (length * Frame::SIZE) >> 30);
        return Mapping::file(begin, length, &file, true).unwrap();
    }
    Mapping::anon(begin, length, !private, populate).unwrap()
}
