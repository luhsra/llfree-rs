use std::env::args;
use std::fs::metadata;

use log::{error, info, warn};
use nvalloc::mmap::MMap;
use nvalloc::util::logging;

fn main() {
    logging();

    let mut args = args();
    args.next().unwrap();
    let filename = args.next().expect("usage: mmap <filename> [size]");

    let size = if let Some(size) = args.next() {
        let (multiplier, size) = match size.chars().last() {
            Some('K') => (1 << 10, &size[..size.len() - 1]),
            Some('M') => (1 << 20, &size[..size.len() - 1]),
            Some('G') => (1 << 30, &size[..size.len() - 1]),
            Some('T') => (1 << 40, &size[..size.len() - 1]),
            Some(_) => (1, &size[..]),
            _ => panic!("Invalid size"),
        };
        size.trim().parse::<usize>().expect("Invalid size") * multiplier
    } else {
        // Compute file size
        info!("compute file size");
        metadata(&filename).unwrap().len() as usize
    };

    info!("open file {filename}");

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&filename)
        .unwrap();

    info!("map file s={size}");

    let mut mapping = MMap::<u8>::file(0x1000_0000_0000_u64 as _, size, file).unwrap();
    info!(
        "mapping {:?} size={} len={}",
        mapping.as_ptr_range(),
        mapping.as_ptr_range().end as usize - mapping.as_ptr_range().start as usize,
        mapping.len()
    );

    info!("check read/write");

    warn!("previously written {}", mapping[0]);
    warn!("previously written {}", mapping[mapping.len() - 1]);

    mapping[0] = 42;
    let len = mapping.len();
    mapping[len - 1] = 33;

    if mapping[0] != 42 || mapping[mapping.len() - 1] != 33 {
        error!("Unexpected value!");
    }

    info!("unmap file");
}
