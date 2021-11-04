pub const fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

pub const fn align_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
}

#[cfg(test)]
pub fn logging() {
    use std::{io::Write, thread::ThreadId};
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format(|buf, record| {
            writeln!(
                buf,
                "{}[{:5} {:2?}@{:<2?} {}:{}] {}\x1b[0m",
                match record.level() {
                    log::Level::Error => "\x1b[91m",
                    log::Level::Warn => "\x1b[93m",
                    log::Level::Info => "\x1b[90m",
                    log::Level::Debug => "\x1b[90m",
                    log::Level::Trace => "\x1b[90m",
                },
                record.level(),
                unsafe { std::mem::transmute::<ThreadId, u64>(std::thread::current().id()) },
                unsafe { libc::sched_getcpu() },
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default(),
                record.args()
            )
        })
        .init();
}
