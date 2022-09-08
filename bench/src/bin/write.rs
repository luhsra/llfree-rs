use std::time::Instant;

use clap::Parser;
use log::warn;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::util::{div_ceil, logging, Page};

/// Crash testing an allocator.
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    /// Number of threads
    #[clap(short, long, default_value_t = 6)]
    threads: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[clap(short, long, default_value_t = 16)]
    memory: usize,
    /// DAX file to be used for the allocator.
    #[clap(long)]
    dax: Option<String>,
    #[clap(short, long)]
    populate: bool,
}

fn main() {
    let Args {
        threads,
        memory,
        dax,
        populate,
    } = Args::parse();

    logging();

    assert!(threads > 0 && memory > 0);

    let t_map = Instant::now();
    let mut mapping = mapping(0x1000_0000_0000, memory * PT_LEN * PT_LEN, dax, populate).unwrap();
    let t_map = t_map.elapsed().as_millis();

    let chunk_size = div_ceil(mapping.len(), threads);
    let times = thread::parallel(mapping.chunks_mut(chunk_size), |chunk| {
        let timer = Instant::now();
        for page in chunk {
            *page.cast_mut::<usize>() = 1;
        }
        timer.elapsed().as_millis()
    });

    let (t_min, t_max) = times
        .into_iter()
        .fold((u128::MAX, 0), |(min, max), x| (min.min(x), max.max(x)));

    let t_unmap = Instant::now();
    drop(mapping);
    let t_unmap = t_unmap.elapsed().as_millis();

    println!("map,min,max,unmap");
    println!("{t_map},{t_min},{t_max},{t_unmap}");
}

#[allow(unused_variables)]
fn mapping(
    begin: usize,
    length: usize,
    dax: Option<String>,
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
    MMap::anon(begin, length, populate)
}
