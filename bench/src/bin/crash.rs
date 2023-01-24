use std::sync::Barrier;
use std::time::Duration;

use clap::Parser;
use log::{error, warn};
use nvalloc::lower::*;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::upper::*;
use nvalloc::util::{self, align_up, WyRand};
use nvalloc::{pfn_range, Page, PFN};

/// Crash testing an allocator.
#[derive(Parser, Debug)]
#[command(about, version, author)]
struct Args {
    /// Names of the allocators to be tested.
    alloc: String,
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

fn main() {
    let Args {
        alloc,
        threads,
        dax,
        order,
        memory,
    } = Args::parse();

    util::logging();

    let pages = memory * PT_LEN * PT_LEN;
    assert!(pages >= MIN_PAGES * threads);

    type C = Cache<32>;
    let allocs: [Box<dyn Alloc>; 1] = [Box::<Array<4, C>>::default()];
    for a in allocs {
        if format!("{}", a.name()) == alloc {
            let allocs = pages / threads / 2 / (1 << order);
            let out_size = align_up(allocs + 2, Page::SIZE) * threads;
            // Shared memory where the allocated pages are backupped
            // Layout: [ ( idx | realloc | pages... ) for each thread ]
            let mut out_mapping =
                MMap::<usize>::anon(0x1100_0000_0000, out_size, true, true).unwrap();
            let out_size = out_size / threads;
            // Initialize with zero
            for out in out_mapping.chunks_mut(out_size) {
                out[0] = 0; // idx = 0
                out[1] = 0; // realloc = 0
            }
            // Allocator mapping
            let mapping = mapping(0x1000_0000_0000, pages, dax).unwrap();
            warn!("Alloc manages {pages} with {} allocs", allocs * threads);

            let pid = unsafe { libc::fork() };
            if pid < 0 {
                unsafe { libc::perror(b"fork failed\0" as *const _ as *mut _) };
                panic!();
            } else if pid == 0 {
                execute(a, allocs, threads, order, mapping, out_mapping);
            } else {
                monitor(a, allocs, threads, order, pid, mapping, out_mapping);
            }
            return;
        }
    }
    panic!("Unknown allocator: {alloc}");
}

/// Allocate and free memory indefinitely
fn execute(
    mut alloc: Box<dyn Alloc>,
    allocs: usize,
    threads: usize,
    order: usize,
    mut mapping: MMap<Page>,
    mut out_mapping: MMap<usize>,
) {
    // Align to prevent false-sharing
    let out_size = align_up(allocs + 2, Page::SIZE);

    // Warmup
    for page in &mut mapping[..] {
        *page.cast_mut::<usize>() = 1;
    }

    alloc
        .init(threads, pfn_range(&mapping), Init::Overwrite, true)
        .unwrap();
    warn!("initialized");

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
            *page = alloc.get(t, order).unwrap().0;
        }

        warn!("repeat");
        *realloc = 1;

        let mut rng = WyRand::new(t as _);
        for _ in 0.. {
            let i = rng.range(0..allocs as u64) as usize;
            *idx = i as _;

            alloc.put(t, PFN(data[i]), order).unwrap();
            data[i] = alloc.get(t, order).unwrap().0;
        }
    });
}

/// Wait for the allocator to allocate memory, kill it and check if recovery is successful.
fn monitor(
    mut alloc: Box<dyn Alloc>,
    allocs: usize,
    threads: usize,
    order: usize,
    child: i32,
    mapping: MMap<Page>,
    out_mapping: MMap<usize>,
) {
    let out_size = align_up(allocs + 2, Page::SIZE);

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
    alloc
        .init(threads, pfn_range(&mapping), Init::Recover, true)
        .unwrap();

    let expected = allocs * threads - threads;
    let actual = alloc.allocated_frames();
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
                alloc.put(t, PFN(*addr), order).unwrap();
            }
        }
    }
    // Check if the remaining pages, that were allocated/freed during the crash,
    // is less equal to the number of concurrent threads (upper bound).
    assert!(alloc.allocated_frames() <= threads);
    warn!("Ok");
    drop(alloc); // Free alloc first
}

#[allow(unused_variables)]
fn mapping(
    begin: usize,
    length: usize,
    dax: Option<String>,
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
    MMap::anon(begin, length, true, true)
}
