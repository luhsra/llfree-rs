#![no_std]
#![feature(alloc_error_handler)]
#![feature(core_ffi_c)]

#[macro_use]
extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::{c_char, c_void};
use core::fmt;
use core::panic::PanicInfo;
use core::sync::atomic::{self, Ordering};

use alloc::boxed::Box;
use log::{error, warn, Level, Metadata, Record};
use log::{LevelFilter, SetLoggerError};

use nvalloc::lower::CacheLower;
use nvalloc::upper::{Alloc, ArrayAtomicAlloc};
use nvalloc::{Error, Size};

const MOD: &[u8] = b"nvalloc\0";

extern "C" {
    /// Linux provided alloc function
    fn nvalloc_linux_alloc(size: u64, align: u64) -> *mut u8;
    /// Linux provided free function
    fn nvalloc_linux_free(ptr: *mut u8, size: u64, align: u64);
    /// Linux provided printk function
    fn nvalloc_printk(format: *const u8, module_name: *const u8, args: *const c_void);
}

type Allocator = ArrayAtomicAlloc<CacheLower<64>>;
static mut ALLOC: Option<Box<Allocator>> = None;

/// Initialize the allocator for the given memory range.
/// If `overwrite` is nonzero no existing allocator state is recovered.
#[no_mangle]
pub extern "C" fn nvalloc_init(cores: u32, addr: *mut c_void, pages: u64, overwrite: u32) -> i64 {
    if let Err(_) = init_logging() {
        return -(Error::Corruption as i64);
    }

    warn!("Initializing inside rust");
    let memory = unsafe { core::slice::from_raw_parts_mut(addr.cast(), pages as _) };

    let mut alloc = Box::new(Allocator::default());
    match alloc.init(cores as _, memory, overwrite != 0) {
        Ok(_) => {
            warn!("{alloc:?}");
            unsafe { ALLOC = Some(alloc) };
            0
        }
        Err(e) => -(e as usize as i64),
    }
}

/// Shut down the allocator normally.
#[no_mangle]
pub extern "C" fn nvalloc_uninit() {
    unsafe { core::mem::drop(ALLOC.take()) };
}

/// Allocate a page of the given `size` on the given cpu `core`.
#[no_mangle]
pub extern "C" fn nvalloc_get(core: u32, size: u32) -> i64 {
    let size = match size {
        0 => Size::L0,
        1 => Size::L1,
        _ => return -(Error::Memory as usize as i64),
    };

    if let Some(alloc) = unsafe { ALLOC.as_ref() } {
        match alloc.get(core as _, size) {
            Ok(addr) => addr as i64,
            Err(e) => -(e as usize as i64),
        }
    } else {
        -(Error::Initialization as usize as i64)
    }
}

/// Frees the given page.
#[no_mangle]
pub extern "C" fn nvalloc_put(core: u32, addr: u64) -> i64 {
    if let Some(alloc) = unsafe { ALLOC.as_ref() } {
        match alloc.put(core as _, addr) {
            Ok(_) => 0,
            Err(e) => -(e as usize as i64),
        }
    } else {
        -(Error::Initialization as usize as i64)
    }
}

struct LinuxAlloc;
unsafe impl GlobalAlloc for LinuxAlloc {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        nvalloc_linux_alloc(layout.size() as _, layout.align() as _)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        nvalloc_linux_free(ptr, layout.size() as _, layout.align() as _);
    }
}

#[global_allocator]
static LINUX_ALLOC: LinuxAlloc = LinuxAlloc;

#[alloc_error_handler]
pub fn on_oom(layout: Layout) -> ! {
    error!("Unable to allocate {} bytes", layout.size());
    loop {
        atomic::compiler_fence(Ordering::SeqCst);
    }
}

#[inline(never)]
#[panic_handler]
pub fn panic(info: &PanicInfo) -> ! {
    error!("{info}");
    loop {
        atomic::compiler_fence(Ordering::SeqCst);
    }
}

/// Printing facilities.
///
/// C header: [`include/linux/printk.h`](../../../../include/linux/printk.h)
///
/// Reference: <https://www.kernel.org/doc/html/latest/core-api/printk-basics.html>
struct PrintKLogger;

impl log::Log for PrintKLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "max_level_debug")] {
                metadata.level() <= Level::Debug
            } else if #[cfg(feature = "max_level_info")] {
                metadata.level() <= Level::Info
            } else if #[cfg(feature = "max_level_error")] {
                metadata.level() <= Level::Error
            } else {
                metadata.level() <= Level::Warn
            }
        }
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            match record.metadata().level() {
                Level::Error => unsafe { call_printk(&format_strings::ERR, record.args()) },
                Level::Warn => unsafe { call_printk(&format_strings::WARNING, record.args()) },
                Level::Info => unsafe { call_printk(&format_strings::INFO, record.args()) },
                Level::Debug => unsafe { call_printk(&format_strings::DEBUG, record.args()) },
                Level::Trace => unsafe { call_printk(&format_strings::DEBUG, record.args()) },
            }
        }
    }

    fn flush(&self) {}
}

static LOGGER: PrintKLogger = PrintKLogger;

pub fn init_logging() -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Info))
}

// Linux bindings
mod bindings {
    pub const KERN_EMERG: &[u8; 3] = b"\x010\0";
    pub const KERN_ALERT: &[u8; 3] = b"\x011\0";
    pub const KERN_CRIT: &[u8; 3] = b"\x012\0";
    pub const KERN_ERR: &[u8; 3] = b"\x013\0";
    pub const KERN_WARNING: &[u8; 3] = b"\x014\0";
    pub const KERN_NOTICE: &[u8; 3] = b"\x015\0";
    pub const KERN_INFO: &[u8; 3] = b"\x016\0";
    pub const KERN_DEBUG: &[u8; 3] = b"\x017\0";
}

// Called from `vsprintf` with format specifier `%pA`.
#[no_mangle]
pub extern "C" fn rust_fmt_argument(
    buf: *mut c_char,
    end: *mut c_char,
    ptr: *const c_void,
) -> *mut c_char {
    use fmt::Write;
    // SAFETY: The C contract guarantees that `buf` is valid if it's less than `end`.
    let mut w = unsafe { RawFormatter::from_ptrs(buf.cast(), end.cast()) };
    let _ = w.write_fmt(unsafe { *(ptr as *const fmt::Arguments<'_>) });
    w.pos().cast()
}

/// Format strings.
///
/// Public but hidden since it should only be used from public macros.
#[doc(hidden)]
pub mod format_strings {
    use super::bindings;

    /// The length we copy from the `KERN_*` kernel prefixes.
    const LENGTH_PREFIX: usize = 2;

    /// The length of the fixed format strings.
    pub const LENGTH: usize = 10;

    /// Generates a fixed format string for the kernel's [`_printk`].
    ///
    /// The format string is always the same for a given level, i.e. for a
    /// given `prefix`, which are the kernel's `KERN_*` constants.
    ///
    /// [`_printk`]: ../../../../include/linux/printk.h
    const fn generate(is_cont: bool, prefix: &[u8; 3]) -> [u8; LENGTH] {
        // Ensure the `KERN_*` macros are what we expect.
        assert!(prefix[0] == b'\x01');
        if is_cont {
            assert!(prefix[1] == b'c');
        } else {
            assert!(prefix[1] >= b'0' && prefix[1] <= b'7');
        }
        assert!(prefix[2] == b'\x00');

        let suffix: &[u8; LENGTH - LENGTH_PREFIX] = if is_cont {
            b"%pA\0\0\0\0\0"
        } else {
            b"%s: %pA\0"
        };

        [
            prefix[0], prefix[1], suffix[0], suffix[1], suffix[2], suffix[3], suffix[4], suffix[5],
            suffix[6], suffix[7],
        ]
    }

    // Generate the format strings at compile-time.
    //
    // This avoids the compiler generating the contents on the fly in the stack.
    //
    // Furthermore, `static` instead of `const` is used to share the strings
    // for all the kernel.
    pub static EMERG: [u8; LENGTH] = generate(false, bindings::KERN_EMERG);
    pub static ALERT: [u8; LENGTH] = generate(false, bindings::KERN_ALERT);
    pub static CRIT: [u8; LENGTH] = generate(false, bindings::KERN_CRIT);
    pub static ERR: [u8; LENGTH] = generate(false, bindings::KERN_ERR);
    pub static WARNING: [u8; LENGTH] = generate(false, bindings::KERN_WARNING);
    pub static NOTICE: [u8; LENGTH] = generate(false, bindings::KERN_NOTICE);
    pub static INFO: [u8; LENGTH] = generate(false, bindings::KERN_INFO);
    pub static DEBUG: [u8; LENGTH] = generate(false, bindings::KERN_DEBUG);
}

/// Prints a message via the kernel's [`_printk`].
///
/// Public but hidden since it should only be used from public macros.
///
/// # Safety
///
/// The format string must be one of the ones in [`format_strings`], and
/// the module name must be null-terminated.
///
/// [`_printk`]: ../../../../include/linux/_printk.h
#[doc(hidden)]
pub unsafe fn call_printk(format_string: &[u8; format_strings::LENGTH], args: &fmt::Arguments<'_>) {
    // `_printk` does not seem to fail in any path.
    nvalloc_printk(
        format_string.as_ptr() as _,
        MOD.as_ptr(),
        args as *const _ as *const c_void,
    );
}

/// Allows formatting of [`fmt::Arguments`] into a raw buffer.
///
/// It does not fail if callers write past the end of the buffer so that they can calculate the
/// size required to fit everything.
///
/// # Invariants
///
/// The memory region between `pos` (inclusive) and `end` (exclusive) is valid for writes if `pos`
/// is less than `end`.
pub(crate) struct RawFormatter {
    // Use `usize` to use `saturating_*` functions.
    beg: usize,
    pos: usize,
    end: usize,
}

impl RawFormatter {
    /// Creates a new instance of [`RawFormatter`] with an empty buffer.
    fn new() -> Self {
        // INVARIANT: The buffer is empty, so the region that needs to be writable is empty.
        Self {
            beg: 0,
            pos: 0,
            end: 0,
        }
    }

    /// Creates a new instance of [`RawFormatter`] with the given buffer pointers.
    ///
    /// # Safety
    ///
    /// If `pos` is less than `end`, then the region between `pos` (inclusive) and `end`
    /// (exclusive) must be valid for writes for the lifetime of the returned [`RawFormatter`].
    pub(crate) unsafe fn from_ptrs(pos: *mut u8, end: *mut u8) -> Self {
        // INVARIANT: The safety requierments guarantee the type invariants.
        Self {
            beg: pos as _,
            pos: pos as _,
            end: end as _,
        }
    }

    /// Creates a new instance of [`RawFormatter`] with the given buffer.
    ///
    /// # Safety
    ///
    /// The memory region starting at `buf` and extending for `len` bytes must be valid for writes
    /// for the lifetime of the returned [`RawFormatter`].
    pub(crate) unsafe fn from_buffer(buf: *mut u8, len: usize) -> Self {
        let pos = buf as usize;
        // INVARIANT: We ensure that `end` is never less then `buf`, and the safety requirements
        // guarantees that the memory region is valid for writes.
        Self {
            pos,
            beg: pos,
            end: pos.saturating_add(len),
        }
    }

    /// Returns the current insert position.
    ///
    /// N.B. It may point to invalid memory.
    pub(crate) fn pos(&self) -> *mut u8 {
        self.pos as _
    }

    /// Return the number of bytes written to the formatter.
    pub(crate) fn bytes_written(&self) -> usize {
        self.pos - self.beg
    }
}

impl fmt::Write for RawFormatter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // `pos` value after writing `len` bytes. This does not have to be bounded by `end`, but we
        // don't want it to wrap around to 0.
        let pos_new = self.pos.saturating_add(s.len());

        // Amount that we can copy. `saturating_sub` ensures we get 0 if `pos` goes past `end`.
        let len_to_copy = core::cmp::min(pos_new, self.end).saturating_sub(self.pos);

        if len_to_copy > 0 {
            // SAFETY: If `len_to_copy` is non-zero, then we know `pos` has not gone past `end`
            // yet, so it is valid for write per the type invariants.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    s.as_bytes().as_ptr(),
                    self.pos as *mut u8,
                    len_to_copy,
                )
            };
        }

        self.pos = pos_new;
        Ok(())
    }
}
