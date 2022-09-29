use std::time::Instant;

use clap::Parser;
use log::warn;
use nvalloc::mmap::{madvise, MAdvise, MMap};
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::util::{div_ceil, logging, Page, avg_bounds};

/// Benchmarking the page-fault performance of a mapped memory region.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// Number of threads
    #[arg(short, long, default_value_t = 6)]
    threads: usize,
    /// Max amount of memory in GiB. Is by the max thread count
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
    /// DAX file to be used for the allocator
    #[arg(long)]
    dax: Option<String>,
    /// Create a private mapping (incompatible with `--dax`)
    #[arg(long)]
    private: bool,
    /// Populate on mmap
    #[arg(long)]
    populate: bool,
    /// Use hugepages
    #[arg(long)]
    huge: bool,
}

fn main() {
    let Args {
        threads,
        memory,
        dax,
        private,
        populate,
        huge,
    } = Args::parse();

    logging();

    assert!(threads > 0 && memory > 0);

    let t_map = Instant::now();
    let mut mapping = mapping(
        0x1000_0000_0000,
        memory * PT_LEN * PT_LEN,
        dax,
        private,
        populate,
    )
    .unwrap();
    let t_map = t_map.elapsed().as_millis();

    let adv = if huge {
        MAdvise::Hugepage
    } else {
        MAdvise::NoHugepage
    };
    madvise(&mut mapping, adv);

    let chunk_size = div_ceil(mapping.len(), threads);
    let times = thread::parallel(mapping.chunks_mut(chunk_size), |chunk| {
        let timer = Instant::now();
        for page in chunk {
            *page.cast_mut::<usize>() = 1;
        }
        timer.elapsed().as_millis()
    });

    let (t_min, t_avg, t_max) = avg_bounds(times).unwrap_or_default();

    let t_unmap = Instant::now();
    drop(mapping);
    let t_unmap = t_unmap.elapsed().as_millis();

    println!("map,min,avg,max,unmap");
    println!("{t_map},{t_min},{t_avg},{t_max},{t_unmap}");
}

#[allow(unused_variables)]
fn mapping(
    begin: usize,
    length: usize,
    dax: Option<String>,
    private: bool,
    populate: bool,
) -> core::result::Result<MMap<Page>, ()> {
    #[cfg(target_os = "linux")]
    if length > 0 {
        if let Some(file) = dax {
            warn!(
                "MMap file {file} l={}G ({:x})",
                (length * std::mem::size_of::<Page>()) >> 30,
                length * std::mem::size_of::<Page>()
            );
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(file)
                .unwrap();
            return MMap::dax(begin, length, f);
        }
    }
    if private {
        MMap::anon_private(begin, length, populate)
    } else {
        MMap::anon(begin, length, populate)
    }
}
