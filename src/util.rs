pub const fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

pub const fn align_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
}

#[cfg(test)]
pub fn logging() {
    use std::io::Write;
    use std::panic::{self, Location};
    use std::sync::atomic::Ordering;
    use std::thread::ThreadId;

    use log::{Level, Record};

    use crate::cpu::PINNED;
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format(|buf, record| {
            let color = match record.level() {
                log::Level::Error => "\x1b[91m",
                log::Level::Warn => "\x1b[93m",
                log::Level::Info => "\x1b[90m",
                log::Level::Debug => "\x1b[90m",
                log::Level::Trace => "\x1b[90m",
            };

            writeln!(
                buf,
                "{}[{:5} {:02?}@{:02?} {}:{}] {}\x1b[0m",
                color,
                record.level(),
                unsafe { std::mem::transmute::<ThreadId, u64>(std::thread::current().id()) },
                PINNED.with(|p| p.load(Ordering::SeqCst)),
                record.file().unwrap_or_default(),
                record.line().unwrap_or_default(),
                record.args()
            )
        })
        .try_init();

    panic::set_hook(Box::new(|info| {
        if let Some(args) = info.message() {
            log::logger().log(
                &Record::builder()
                    .args(*args)
                    .level(Level::Error)
                    .file(info.location().map(Location::file))
                    .line(info.location().map(Location::line))
                    .build(),
            );
        } else if let Some(&payload) = info.payload().downcast_ref::<&'static str>() {
            log::logger().log(
                &Record::builder()
                    .args(format_args!("{}", payload))
                    .level(Level::Error)
                    .file(info.location().map(Location::file))
                    .line(info.location().map(Location::line))
                    .build(),
            );
        } else {
            log::logger().log(
                &Record::builder()
                    .args(format_args!("panic!"))
                    .level(Level::Error)
                    .file(info.location().map(Location::file))
                    .line(info.location().map(Location::line))
                    .build(),
            )
        }

        log::logger().flush();
    }));
}

#[cfg(test)]
pub fn parallel<F: FnOnce(usize) + Clone + Send + 'static>(n: usize, f: F) {
    let handles = (0..n)
        .into_iter()
        .map(|t| {
            let f = f.clone();
            std::thread::spawn(move || f(t))
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn _mm_clwb(addr: *const ()) {
    asm!("clwb [rax]", in("rax") addr);
}

#[cfg(test)]
mod test {
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn clwb() {
        let mut data = Box::new(43_u64);
        *data = 44;
        unsafe { super::_mm_clwb(data.as_ref() as *const _ as _) };
    }
}
