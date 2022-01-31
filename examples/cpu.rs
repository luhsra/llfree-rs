#[cfg(target_os = "linux")]
fn getcpu() -> (usize, usize) {
    use std::ffi::c_void;
    use std::os::raw::{c_int, c_uint};
    use std::ptr::null_mut;

    extern "C" {
        fn getcpu(cpu: *mut c_uint, node: *mut c_uint, tcache: *mut c_void) -> c_int;
    }
    let mut cpu: c_uint = 0;
    let mut node: c_uint = 0;
    let ret = unsafe { getcpu(&mut cpu, &mut node, null_mut()) };
    if ret != 0 {
        panic!("getcpu error {}", ret);
    }
    (cpu as _, node as _)
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use nvalloc::thread;
    use raw_cpuid::CpuId;
    let cpuid = CpuId::new();
    if let Some(info) = cpuid.get_vendor_info() {
        println!("Vendor: {info}")
    }

    if let Some(info) = cpuid.get_extended_feature_info() {
        println!("flushopt: {}", info.has_clflushopt());
        println!("clwb:     {}", info.has_clwb());
        println!("avx2:     {}", info.has_avx2());
        println!("avx512f:  {}", info.has_avx512f());
    }

    unsafe { thread::CPU_STRIDE = 1 };

    #[cfg(target_os = "linux")]
    {
        println!("CPU cores");
        for i in 0..std::thread::available_parallelism().unwrap().get() {
            thread::pin(i);
            let (cpu, numa) = getcpu();
            println!("{i:>4} on C={cpu} N={numa}");
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    panic!("unsupported")
}
