#![cfg(any(test, feature = "thread"))]
use std::sync::atomic::AtomicUsize;

thread_local! {
    pub static PINNED: AtomicUsize = AtomicUsize::new(0);
}

/// Useful if we only want to operate on a single numa node.
/// Pin then only selects every n'th cpu.
const NUMA_NODES: usize = 1;

#[cfg(target_os = "linux")]
pub fn pin(core: usize) {
    use std::sync::atomic::Ordering;

    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_SET(NUMA_NODES * core, &mut set) };
    let ret = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if ret != 0 {
        unsafe { libc::perror(b"sched_setaffinity\0" as *const _ as _) };
        panic!("sched_setaffinity failed");
    }

    PINNED.with(|p| {
        p.store(core, Ordering::SeqCst);
    });
}

#[cfg(target_os = "macos")]
pub fn pin(core: usize) {
    #![allow(non_camel_case_types)]

    use std::{mem, os::raw::c_int, sync::atomic::Ordering};

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
    let thread_affinity_policy_count: mach_msg_type_number_t =
        mem::size_of::<thread_affinity_policy_data_t>() as mach_msg_type_number_t
            / mem::size_of::<c_int>() as mach_msg_type_number_t;

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

pub fn parallel<T: Send + 'static, F: FnOnce(usize) -> T + Clone + Send + 'static>(
    n: usize,
    f: F,
) -> Vec<T> {
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
    #[test]
    fn pinning() {
        super::pin(0);
        println!("Pinned to 0");
        super::pin(num_cpus::get_physical() - 1);
        println!("Pinned to {}", num_cpus::get_physical() - 1);
    }
}
