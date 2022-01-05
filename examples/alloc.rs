#![cfg(all(feature = "thread", feature = "logger"))]

use std::sync::{Arc, Barrier};
use std::time::Instant;

use log::warn;

use nvalloc_rs::alloc::{alloc, Allocator, MIN_PAGES};
use nvalloc_rs::mmap::MMap;
use nvalloc_rs::{thread, util, Error, Page, Size};

#[cfg(target_os = "linux")]
fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<'a, Page>, ()> {
    if let Ok(file) = std::env::var("NVM_FILE") {
        warn!(
            "MMap file {} l={}G",
            file,
            (length * std::mem::size_of::<Page>()) >> 30
        );
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file)
            .unwrap();
        MMap::dax(begin, length, f)
    } else {
        MMap::anon(begin, length)
    }
}
#[cfg(not(target_os = "linux"))]
fn mapping<'a>(begin: usize, length: usize) -> Result<MMap<'a, Page>, ()> {
    MMap::anon(begin, length)
}

fn main() {
    util::logging();

    const THREADS: usize = 6;
    const MEM_PAGES: usize = 2 * THREADS * MIN_PAGES;

    warn!("pages={MEM_PAGES}");
    let mut mapping = mapping(0x1000_0000_0000, MEM_PAGES).unwrap();

    Allocator::init(THREADS, &mut mapping).unwrap();

    let barrier = Arc::new(Barrier::new(THREADS));

    thread::parallel(THREADS as _, move |t| {
        thread::pin(t);
        barrier.wait();

        let mut pages = Vec::new();

        let mut min = u128::MAX;
        let mut max = 0;
        let mut sum = 0;

        loop {
            let timer = Instant::now();
            match alloc().get(t, Size::L0) {
                Ok(page) => pages.push(page),
                Err(Error::Memory) => break,
                Err(e) => panic!("{:?}", e),
            }
            let elapsed = timer.elapsed().as_nanos();
            min = min.min(elapsed);
            max = max.max(elapsed);
            sum += elapsed;
        }
        let len = pages.len() as u128;

        warn!("thread {t} allocated {} [{min}, {}, {max}]", len, sum / len);
        barrier.wait();

        let mut min = u128::MAX;
        let mut max = 0;
        let mut sum = 0;

        for page in pages {
            let timer = Instant::now();
            alloc().put(t, page).unwrap();
            let elapsed = timer.elapsed().as_nanos();
            min = min.min(elapsed);
            max = max.max(elapsed);
            sum += elapsed;
        }
        warn!("thread {t} freed {} [{min}, {}, {max}]", len, sum / len);
    });

    assert_eq!(alloc().allocated_pages(), 0);
    Allocator::uninit();
}
