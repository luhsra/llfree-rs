use std::os::raw::c_uint;
use std::panic;
use std::ptr::null_mut;

use libc::cpu_set_t;
use log::error;

mod raw {
    use std::ffi::c_void;
    use std::os::raw::{c_int, c_uint};

    extern "C" {
        pub(crate) fn getcpu(cpu: *mut c_uint, node: *mut c_uint, tcache: *mut c_void) -> c_int;
    }
}

struct Cpu {
    id: c_uint,
    node: c_uint,
}

fn getcpu() -> Cpu {
    let mut cpu = Cpu { id: 0, node: 0 };
    let ret = unsafe { raw::getcpu(&mut cpu.id as *mut _, &mut cpu.node as *mut _, null_mut()) };
    if ret != 0 {
        error!("getcpu failed");
        unsafe { libc::perror(b"getcpu\0" as *const _ as _) };
        panic!();
    }
    cpu
}

pub struct CoreIDs {
    set: libc::cpu_set_t,
    i: usize,
}

impl Iterator for CoreIDs {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        for i in self.i..libc::CPU_SETSIZE as usize {
            if unsafe { libc::CPU_ISSET(i, &self.set) } {
                self.i = i + 1;
                return Some(i);
            }
        }
        self.i = libc::CPU_SETSIZE as usize;
        None
    }
}

pub fn affinity() -> CoreIDs {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let ret = unsafe { libc::sched_getaffinity(0, std::mem::size_of::<cpu_set_t>(), &mut set) };
    if ret != 0 {
        error!("getcpu failed");
        unsafe { libc::perror(b"sched_getaffinity\0" as *const _ as _) };
        panic!();
    }
    CoreIDs { set, i: 0 }
}

pub fn pin(core: usize) {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_SET(core, &mut set) };
    let ret = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if ret != 0 {
        error!("getcpu failed");
        unsafe { libc::perror(b"sched_setaffinity\0" as *const _ as _) };
        panic!();
    }
}

#[cfg(test)]
mod test {
    use log::info;

    use crate::util::logging;

    #[test]
    fn cpunum() {
        let cpu = super::getcpu();
        println!("Running on core {}, node {}", cpu.id, cpu.node);
    }

    #[test]
    fn cores() {
        logging();
        info!("Size of CoreIDs: {}", std::mem::size_of::<super::CoreIDs>());

        info!("Retrieving affinity...");
        println!("Affinity: {:?}", super::affinity().collect::<Vec<_>>());

        info!("Switching core");
        super::pin(3);

        println!("Affinity: {:?}", super::affinity().collect::<Vec<_>>());
        let cpu = super::getcpu();
        println!("Running on core {}, node {}", cpu.id, cpu.node);
    }
}
