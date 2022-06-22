#![cfg(any(test, feature = "thread"))]
use core::sync::atomic::{AtomicUsize, Ordering};

pub static STRIDE: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    pub static PINNED: AtomicUsize = AtomicUsize::new(0);
}

/// Returns the number of virtual cores.
#[cfg(target_os = "linux")]
pub fn cores() -> usize {
    use core::mem::{size_of, zeroed};

    unsafe {
        let mut set: libc::cpu_set_t = zeroed();
        assert!(
            libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) == 0,
            "sched_getaffinity"
        );
        libc::CPU_COUNT(&set) as usize
    }
}

/// Returns the number of virtual cores.
#[cfg(target_os = "macos")]
pub fn cores() -> usize {
    // FIXME: Intel hyperthreading?
    let cores = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize };
    assert!(cores > 0, "sysconf");
    cores
}

/// Pins the current thread to the given virtual core
#[cfg(target_os = "linux")]
pub fn pin(core: usize) {
    use core::mem::{size_of, zeroed};

    let max = cores();
    assert!(core < max as usize, "not enough cores");

    let core = core * STRIDE.load(Ordering::Relaxed);
    let core = (core / max as usize) + (core % max as usize); // wrap around

    let mut set = unsafe { zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_SET(core, &mut set) };
    let ret = unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) };
    if ret != 0 {
        unsafe { libc::perror(b"sched_setaffinity\0" as *const _ as _) };
        panic!("sched_setaffinity failed");
    }

    PINNED.with(|p| {
        p.store(core, Ordering::SeqCst);
    });
}

/// Pins the current thread to the given virtual core
#[cfg(target_os = "macos")]
pub fn pin(core: usize) {
    #![allow(non_camel_case_types)]

    use std::mem::size_of;
    use std::os::raw::{c_int, c_uint};

    type kern_return_t = c_int;
    type thread_t = c_uint;
    type thread_policy_flavor_t = c_int;
    type mach_msg_type_number_t = c_int;

    #[repr(C)]
    struct thread_affinity_policy_data_t {
        affinity_tag: c_int,
    }

    type thread_policy_t = *mut thread_affinity_policy_data_t;

    const THREAD_AFFINITY_POLICY: thread_policy_flavor_t = 4;

    #[link(name = "System", kind = "framework")]
    extern "C" {
        fn thread_policy_set(
            thread: thread_t,
            flavor: thread_policy_flavor_t,
            policy_info: thread_policy_t,
            count: mach_msg_type_number_t,
        ) -> kern_return_t;
    }

    let max = cores();
    assert!(core < max as usize, "not enough cores");

    let core = core * STRIDE.load(Ordering::Relaxed);
    let core = (core / max as usize) + (core % max as usize); // wrap around

    let thread_affinity_policy_count: mach_msg_type_number_t =
        size_of::<thread_affinity_policy_data_t>() as mach_msg_type_number_t
            / size_of::<c_int>() as mach_msg_type_number_t;

    let mut info = thread_affinity_policy_data_t {
        affinity_tag: core as c_int,
    };

    unsafe {
        thread_policy_set(
            libc::pthread_self() as thread_t,
            THREAD_AFFINITY_POLICY,
            &mut info as thread_policy_t,
            thread_affinity_policy_count,
        );
    }

    PINNED.with(|p| {
        p.store(core, Ordering::SeqCst);
    });
}

/// Executed `f` on `n` parallel threads.
pub fn parallel<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send + 'static,
    F: FnOnce(usize) -> T + Clone + Send + 'static,
{
    let handles = (0..n)
        .into_iter()
        .map(|t| {
            let f = f.clone();
            std::thread::spawn(move || f(t))
        })
        .collect::<Vec<_>>();
    handles.into_iter().map(|t| t.join().unwrap()).collect()
}

#[cfg(test)]
mod test {
    use core::sync::atomic::Ordering;

    use crate::thread::STRIDE;

    #[test]
    fn pinning() {
        let cores = std::thread::available_parallelism().unwrap().get();

        println!("max cores: {cores}");

        super::pin(0);
        println!(
            "Pinned to {}",
            super::PINNED.with(|v| v.load(Ordering::SeqCst))
        );
        super::pin(cores - 1);
        println!(
            "Pinned to {}",
            super::PINNED.with(|v| v.load(Ordering::SeqCst))
        );
    }

    #[test]
    fn stride() {
        let old = STRIDE.swap(2, Ordering::Relaxed);

        let cores = std::thread::available_parallelism().unwrap().get();

        println!("max cores: {cores}");

        super::pin(0);
        println!(
            "Pinned to {}",
            super::PINNED.with(|v| v.load(Ordering::SeqCst))
        );
        super::pin(cores / 2);
        println!(
            "Pinned to {}",
            super::PINNED.with(|v| v.load(Ordering::SeqCst))
        );

        STRIDE.store(old, Ordering::Relaxed);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn cores() {
        println!("cores {}", super::cores());
        println!("nproc conf {}", unsafe {
            libc::sysconf(libc::_SC_NPROCESSORS_CONF)
        });
        println!("nproc onln {}", unsafe {
            libc::sysconf(libc::_SC_NPROCESSORS_ONLN)
        });
    }
}
