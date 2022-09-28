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
    #[clap(long)]
    private: bool,
    #[clap(long)]
    populate: bool,
    #[clap(long)]
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

    if huge {
        madvise(&mut mapping, MAdvice::Hugepage);
    }

    let chunk_size = div_ceil(mapping.len(), threads);
    let times = thread::parallel(mapping.chunks_mut(chunk_size), |chunk| {
        let timer = Instant::now();
        for page in chunk {
            *page.cast_mut::<usize>() = 1;
        }
        timer.elapsed().as_millis()
    });

    let (t_amin, t_amax) = times
        .into_iter()
        .fold((u128::MAX, 0), |(min, max), x| (min.min(x), max.max(x)));

    let t_unmap = Instant::now();
    drop(mapping);
    let t_unmap = t_unmap.elapsed().as_millis();

    println!("map,amin,amax,unmap");
    println!("{t_map},{t_amin},{t_amax},{t_unmap}");
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

#[allow(dead_code)]
#[repr(i32)]
enum MAdvice {
    Normal = libc::MADV_NORMAL,
    Random = libc::MADV_RANDOM,
    Sequential = libc::MADV_SEQUENTIAL,
    Willneed = libc::MADV_WILLNEED,
    DontNeed = libc::MADV_DONTNEED,
    Free = libc::MADV_FREE,
    Remove = libc::MADV_REMOVE,
    DontFork = libc::MADV_DONTFORK,
    DoFork = libc::MADV_DOFORK,
    Mergeable = libc::MADV_MERGEABLE,
    Unmergeable = libc::MADV_UNMERGEABLE,
    Hugepage = libc::MADV_HUGEPAGE,
    NoHugepage = libc::MADV_NOHUGEPAGE,
    DontDump = libc::MADV_DONTDUMP,
    DoDump = libc::MADV_DODUMP,
    HwPoison = libc::MADV_HWPOISON,
}

fn madvise(mem: &mut [Page], advice: MAdvice) {
    let ret = unsafe {
        libc::madvise(
            mem.as_mut_ptr() as *mut _,
            Page::SIZE * mem.len(),
            advice as _,
        )
    };
    if ret != 0 {
        unsafe { libc::perror(b"madvice\0" as *const u8 as *const _) };
        panic!("madvice {ret}");
    }
}
