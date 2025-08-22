use std::cmp::Ordering::*;
use std::sync::Barrier;
use std::time::Duration;

use clap::Parser;
use llfree::frame::Frame;
use llfree::mmap::Mapping;
use llfree::util::{self, WyRand, align_up, aligned_buf};
use llfree::wrapper::NvmAlloc;
use llfree::{Alloc, Flags, LLFree, thread};
use log::{error, warn};

/// Crash testing an allocator.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// Max number of threads
    #[arg(short, long, default_value = "6")]
    threads: usize,
    #[arg(long)]
    dax: Option<String>,
    #[arg(short, long, default_value_t = 0)]
    order: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[arg(short, long, default_value_t = 16)]
    memory: usize,
}

type Allocator<'a> = NvmAlloc<'a, LLFree<'a>>;

fn main() {
    let Args {
        threads,
        dax,
        order,
        memory,
    } = Args::parse();

    util::logging();

    let pages = (memory << 30) / Frame::SIZE;

    let allocs = pages / threads / 2 / (1 << order);
    let out_size = align_up(allocs + 2, Frame::SIZE) * threads;
    // Shared memory where the allocated pages are backupped
    // Layout: [ ( idx | realloc | pages... ) for each thread ]
    let mut out_mapping = Mapping::anon(0x1100_0000_0000, out_size, true, false).unwrap();
    let out_size = out_size / threads;
    // Initialize with zero
    for out in out_mapping.chunks_mut(out_size) {
        out[0] = 0; // idx = 0
        out[1] = 0; // realloc = 0
    }
    // Allocator mapping
    let mut mapping = mapping(0x1000_0000_0000, pages, dax);
    warn!("Alloc manages {pages} with {} allocs", allocs * threads);

    let pid = unsafe { libc::fork() };
    match pid.cmp(&0) {
        Equal => execute(allocs, threads, order, &mut mapping, &mut out_mapping),
        Greater => monitor(allocs, threads, order, pid, &mut mapping, &out_mapping),
        Less => {
            unsafe { libc::perror(c"fork failed".as_ptr()) };
            panic!();
        }
    }
}

/// Allocate and free memory indefinitely
fn execute(
    allocs: usize,
    threads: usize,
    order: usize,
    mapping: &mut [Frame],
    out_mapping: &mut [usize],
) {
    // Align to prevent false-sharing
    let out_size = align_up(allocs + 2, Frame::SIZE);

    let m = Allocator::metadata_size(threads, mapping.len());
    let local = aligned_buf(m.local);
    let trees = aligned_buf(m.trees);
    let alloc = Allocator::create(threads, mapping, false, local, trees).unwrap();
    warn!("initialized {}", alloc.frames());

    let barrier = Barrier::new(threads);
    thread::parallel(out_mapping.chunks_mut(out_size).enumerate(), |(t, out)| {
        thread::pin(t);

        // Currently modified index
        let (idx, data) = out.split_first_mut().unwrap();
        // If the benchmark already started reallocating pages
        let (realloc, data) = data.split_first_mut().unwrap();

        barrier.wait();
        warn!("alloc {allocs}");

        for (i, page) in data.iter_mut().enumerate() {
            *idx = i as _;
            *page = alloc.get(t, Flags::o(order)).unwrap();
        }

        warn!("repeat");
        *realloc = 1;

        let mut rng = WyRand::new(t as _);
        for _ in 0.. {
            let i = rng.range(0..allocs as u64) as usize;
            *idx = i as _;

            alloc.put(t, data[i], Flags::o(order)).unwrap();
            data[i] = alloc.get(t, Flags::o(order)).unwrap();
        }
    });
}

/// Wait for the allocator to allocate memory, kill it and check if recovery is successful.
fn monitor(
    allocs: usize,
    threads: usize,
    order: usize,
    child: i32,
    mapping: &mut [Frame],
    out_mapping: &[usize],
) {
    let out_size = align_up(allocs + 2, Frame::SIZE);

    // Wait for the allocator to finish initialization
    warn!("wait");
    let mut initializing = true;
    while initializing {
        std::thread::sleep(Duration::from_millis(1000));

        initializing = false;
        for data in out_mapping.chunks(out_size) {
            if data[1] != 1 {
                initializing = true;
                break;
            }
        }
        warn!("initializing {initializing}");

        // Check that the child is still running
        let mut status = unsafe { std::mem::zeroed() };
        let result = unsafe { libc::waitpid(child, &mut status, libc::WNOHANG) };
        if result == -1 {
            unsafe { libc::perror(b"waitpid failed\0" as *const _ as *mut _) };
            panic!();
        } else if result != 0 {
            error!("Child terminated");
            panic!();
        }
    }

    // Kill allocator (crash)
    warn!("kill child {child}");
    if unsafe { libc::kill(child, libc::SIGKILL) } != 0 {
        unsafe { libc::perror(b"kill failed\0" as *const _ as *mut _) };
        panic!();
    }
    let mut status = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_ne!(result, -1);
    assert_ne!(result, 0);

    warn!("check");

    // Recover allocator
    let m = Allocator::metadata_size(threads, mapping.len());
    let local = aligned_buf(m.local);
    let trees = aligned_buf(m.trees);
    let alloc = Allocator::create(threads, mapping, true, local, trees).unwrap();
    warn!("recovered {}", alloc.frames());

    let expected = allocs * threads - threads;
    let actual = alloc.frames() - alloc.fast_stats().free_frames;
    warn!("expected={expected} actual={actual}");
    assert!(expected <= actual && actual <= expected + threads);

    for (t, data) in out_mapping.chunks(out_size).enumerate() {
        let (idx, data) = data.split_first().unwrap();
        let (realloc, _) = data.split_first().unwrap();

        warn!("Out t{t} idx={idx} realloc={realloc}");
        assert_eq!(*realloc, 1);
    }

    // Try to free all allocated pages
    // Except those that were allocated/freed during the crash
    warn!("try free");
    for (t, data) in out_mapping.chunks(out_size).enumerate() {
        let (idx, data) = data.split_first().unwrap();
        let (_realloc, data) = data.split_first().unwrap();

        for (i, addr) in data[0..allocs].iter().enumerate() {
            if i != *idx {
                alloc.put(t, *addr, Flags::o(order)).unwrap();
            }
        }
    }
    // Check if the remaining pages, that were allocated/freed during the crash,
    // is less equal to the number of concurrent threads (upper bound).
    assert!(alloc.frames() - alloc.fast_stats().free_frames <= threads);
    warn!("Ok");
    drop(alloc); // Free alloc first
}

#[allow(unused_variables)]
pub fn mapping(begin: usize, length: usize, dax: Option<String>) -> Mapping<Frame> {
    #[cfg(target_os = "linux")]
    if let Some(file) = dax {
        warn!("MMap file {file} l={}G", (length * Frame::SIZE) >> 30);
        return Mapping::file(begin, length, &file, true).unwrap();
    }
    Mapping::anon(begin, length, true, false).unwrap()
}
