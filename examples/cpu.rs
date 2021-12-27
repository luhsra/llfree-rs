#[cfg(feature = "x86")]
fn main() {
    use raw_cpuid::CpuId;
    let cpuid = CpuId::new();
    if let Some(info) = cpuid.get_vendor_info() {
        println!("Vendor: {}", info)
    }

    if let Some(info) = cpuid.get_extended_feature_info() {
        println!("flushopt: {}", info.has_clflushopt());
        println!("clwb:     {}", info.has_clwb());
        println!("avx2:     {}", info.has_avx2());
        println!("avx512f:  {}", info.has_avx512f());
    }
}

fn main() {
    panic!("unsupported")
}
