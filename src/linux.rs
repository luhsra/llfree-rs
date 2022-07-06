use core::alloc::GlobalAlloc;

extern "C" {
    // Linux provided alloc function
    fn nvalloc_linux_alloc(size: usize, align: usize) -> *mut u8;
    // Linux provided free function
    fn nvalloc_linux_free(ptr: *mut u8, size: usize, align: usize);
}

struct LinuxAlloc;
unsafe impl GlobalAlloc for LinuxAlloc {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        nvalloc_linux_alloc(layout.size(), layout.align())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        nvalloc_linux_free(ptr, layout.size(), layout.align());
    }
}

#[global_allocator]
static LINUX_ALLOC: LinuxAlloc = LinuxAlloc;
