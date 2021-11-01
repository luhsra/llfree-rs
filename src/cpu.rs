use std::os::raw::c_uint;
use std::ptr::null_mut;

use crate::mmap::perror;

mod raw {
    use std::os::raw::{c_int, c_uint};

    extern "C" {
        pub(crate) fn getcpu(cpu: *mut c_uint, node: *mut c_uint, tcache: *mut ()) -> c_int;
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
        perror("getcpu");
    }
    cpu
}

#[cfg(test)]
mod test {
    use crate::cpu::getcpu;

    #[test]
    fn cpunum() {
        let cpu = getcpu();
        println!("Running on core {}, node {}", cpu.id, cpu.node);
    }
}
