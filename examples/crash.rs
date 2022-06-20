#![cfg(all(feature = "thread", feature = "logger"))]

use std::slice;
use std::sync::Arc;
use std::sync::Barrier;
use std::time::Duration;

use clap::Parser;
use log::{error, warn};
use nvalloc::alloc::*;
use nvalloc::lower::DynamicLower;
use nvalloc::lower::FixedLower;
use nvalloc::mmap::MMap;
use nvalloc::table::PT_LEN;
use nvalloc::thread;
use nvalloc::util::{self, align_up, Page, WyRand};

/// Crash testing an allocator.
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    alloc: String,
    /// Max number of threads
    #[clap(short, long, default_value = "6")]
    threads: usize,
    #[clap(long)]
    dax: Option<String>,
    #[clap(short, long, default_value_t = 0)]
    size: usize,
    /// Max amount of memory in GiB. Is by the max thread count.
    #[clap(short, long, default_value_t = 16)]
    memory: usize,
}

fn main() {
    let Args {
        alloc,
        threads,
        dax,
        size,
        memory,
    } = Args::parse();

    util::logging();

    let pages = memory * PT_LEN * PT_LEN;
    assert!(pages >= MIN_PAGES * threads);

    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        _ => panic!("`size` has to be 0, 1 or 2"),
    };

    let allocs: [Arc<dyn Alloc>; 10] = [
        Arc::new(ArrayAlignedAlloc::<DynamicLower>::default()),
        Arc::new(ArrayUnalignedAlloc::<DynamicLower>::default()),
        Arc::new(ArrayLockedAlloc::<DynamicLower>::default()),
        Arc::new(ArrayAtomicAlloc::<DynamicLower>::default()),
        Arc::new(TableAlloc::<DynamicLower>::default()),
        Arc::new(ArrayAlignedAlloc::<FixedLower>::default()),
        Arc::new(ArrayUnalignedAlloc::<FixedLower>::default()),
        Arc::new(ArrayLockedAlloc::<FixedLower>::default()),
        Arc::new(ArrayAtomicAlloc::<FixedLower>::default()),
        Arc::new(TableAlloc::<FixedLower>::default()),
    ];
    for a in allocs {
        if a.name() == alloc {
            let allocs = pages / threads / 2 / size.span();
            let out_size = align_up(allocs + 2, Page::SIZE) * threads;
            // Shared memory where the allocated pages are backupped
            // Layout: [ ( idx | repeat | pages... ) for each thread ]
            let mut out_mapping = MMap::<u64>::anon(0x1100_0000_0000, out_size).unwrap();
            let out_size = out_size / threads;
            // Initialize with zero
            for t in 0..threads {
                out_mapping[t * out_size] = 0; // idx = 0
                out_mapping[t * out_size + 1] = 0; // repeat = 0
            }
            // Allocator mapping
            let mapping = mapping(0x1000_0000_0000, pages, dax).unwrap();

            let pid = unsafe { libc::fork() };
            if pid < 0 {
                unsafe { libc::perror(b"fork failed\0" as *const _ as *mut _) };
                panic!();
            } else if pid == 0 {
                execute(a, allocs, threads, size, mapping, out_mapping);
            } else {
                monitor(a, allocs, threads, pid, mapping, out_mapping);
            }
            return;
        }
    }
    panic!("Unknown allocator: {}", alloc);
}

/// Allocate and free memory indefinitely
fn execute(
    alloc: Arc<dyn Alloc>,
    allocs: usize,
    threads: usize,
    size: Size,
    mut mapping: MMap<Page>,
    mut out_mapping: MMap<u64>,
) {
    let out_size = align_up(allocs + 2, Page::SIZE);
    let out_begin = out_mapping.as_mut_ptr() as usize;

    // Warmup
    for page in &mut mapping[..] {
        *page.cast_mut::<usize>() = 1;
    }

    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, &mut mapping, true)
        .unwrap();
    warn!("initialized");

    let barrier = Arc::new(Barrier::new(threads));
    let out_mapping_len = out_mapping.len();
    thread::parallel(threads, move |t| {
        thread::pin(t);

        let out = unsafe {
            slice::from_raw_parts_mut((out_begin as *mut u64).add(t * out_size), out_size)
        };
        assert!(t * out_size <= out_mapping_len);

        let (idx, data) = out.split_first_mut().unwrap();
        let (repeat, data) = data.split_first_mut().unwrap();

        barrier.wait();
        warn!("alloc");

        for i in 0..allocs {
            *idx = i as _;
            data[i] = alloc.get(t, size).unwrap();
        }

        *repeat = 1;
        warn!("repeat {:?} -> {}", repeat as *mut u64, repeat);

        let mut rng = WyRand::new(t as _);
        for _ in 0.. {
            let i = rng.range(0..allocs as u64) as usize;
            *idx = i as _;

            alloc.put(t, data[i]).unwrap();
            data[i] = alloc.get(t, size).unwrap();
        }
    });
}

/// Wait for the allocator to allocate memory, kill it and check if recovery is successful.
fn monitor(
    alloc: Arc<dyn Alloc>,
    allocs: usize,
    threads: usize,
    child: i32,
    mut mapping: MMap<Page>,
    out_mapping: MMap<u64>,
) {
    let out_size = align_up(allocs + 2, Page::SIZE);

    // Wait for the allocator to finish initialization
    warn!("wait");

    let mut initializing = true;
    while initializing {
        std::thread::sleep(Duration::from_millis(1000));

        initializing = false;
        for t in 0..threads {
            let data = &out_mapping[t * out_size..(t + 1) * out_size];
            let (_idx, data) = data.split_at(1);
            let (repeat, _) = data.split_at(1);
            if repeat[0] != 1 {
                initializing = true;
                break;
            }
        }
        warn!("initializing {initializing}");

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
    unsafe { &mut *(Arc::as_ptr(&alloc) as *mut dyn Alloc) }
        .init(threads, &mut mapping, false)
        .unwrap();

    let expected = allocs * threads - threads;
    let actual = alloc.dbg_allocated_pages();
    warn!("expected={expected} actual={actual}");
    assert!(expected <= actual && actual <= expected + threads);

    for t in 0..threads {
        let data = &out_mapping[t * out_size..(t + 1) * out_size];
        let (idx, data) = data.split_first().unwrap();
        let (repeat, _) = data.split_first().unwrap();

        warn!("Out t{t} idx={} repeat={}", idx, repeat);
        assert_eq!(*repeat, 1);
    }

    // Try to free all allocated pages
    // Except those that were allocated/freed during the crash
    warn!("try free");
    for t in 0..threads {
        let data = &out_mapping[t * out_size..(t + 1) * out_size];
        let (idx, data) = data.split_first().unwrap();
        let (repeat, data) = data.split_first().unwrap();

        let max = if *repeat == 0 { *idx as usize } else { allocs };
        for (i, addr) in data[0..max].iter().enumerate() {
            if i != *idx as usize {
                alloc.put(t, *addr).unwrap();
            }
        }
    }
    // Check if the remaining pages, that were allocated/freed during the crash,
    // is less equal to the number of concurrent threads (upper bound).
    assert!(alloc.dbg_allocated_pages() <= threads);
    warn!("Ok");
}

#[allow(unused_variables)]
fn mapping<'a>(
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
    MMap::anon(begin, length)
}
