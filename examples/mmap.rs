use std::env::{args, Args};
use std::fs::metadata;
use std::io::{Seek, SeekFrom};
use std::slice;

use log::{error, info};
use nvalloc_rs::mmap;

fn main() {
    logging();

    let mut args = args();

    args.next().unwrap();
    let filename = args.next().expect("usage: mmap <filename> [size]");

    let size = if let Some(size) = args.next() {
        let mut multiplier = 1;
        let size = match size.chars().last() {
            Some('K') => {
                multiplier = 1 << 10;
                &size[..size.len() - 1]
            }
            Some('M') => {
                multiplier = 1 << 20;
                &size[..size.len() - 1]
            }
            Some('G') => {
                multiplier = 1 << 30;
                &size[..size.len() - 1]
            }
            Some('T') => {
                multiplier = 1 << 40;
                &size[..size.len() - 1]
            }
            Some(_) => panic!("Invalid size"),
            None => &size[..],
        };
        size.trim().parse::<usize>().expect("Invalid size") * multiplier
    } else {
        // Compute file size
        info!("compute file size");
        metadata(&filename).unwrap().len() as usize
    };

    info!("open file {}", filename);

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&filename)
        .unwrap();

    info!("map file s={}", size);

    let data = unsafe { slice::from_raw_parts_mut(0x1000_0000_0000_u64 as _, size) };
    mmap::c_mmap_fixed(data, file).unwrap();

    info!("check read/write");

    data[0] = 42;

    if data[0] != 42 {
        error!("Unexpected value!");
    }

    info!("unmap file");

    mmap::c_munmap(data).unwrap();
}

pub fn logging() {
    use std::{io::Write, thread::ThreadId};
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            writeln!(
                buf,
                "{}[{:5} {:2?} {}:{}] {}\x1b[0m",
                match record.level() {
                    log::Level::Error => "\x1b[91m",
                    log::Level::Warn => "\x1b[93m",
                    log::Level::Info => "\x1b[90m",
                    log::Level::Debug => "\x1b[90m",
                    log::Level::Trace => "\x1b[90m",
                },
                record.level(),
                unsafe { std::mem::transmute::<ThreadId, u64>(std::thread::current().id()) },
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default(),
                record.args()
            )
        })
        .init();
}
