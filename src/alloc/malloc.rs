use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info, warn};

use super::{Alloc, Error, Result, Size};
use crate::util::Page;

/// Wrapper for libc malloc.
#[derive(Default)]
pub struct MallocAlloc {
    local: Box<[Local]>,
}

#[repr(align(64))]
struct Local {
    counter: AtomicUsize,
}

unsafe impl Send for MallocAlloc {}
unsafe impl Sync for MallocAlloc {}

impl fmt::Debug for MallocAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {{", self.name())?;
        for (t, l) in self.local.iter().enumerate() {
            writeln!(f, "    L {t:>2} C={}", l.counter.load(Ordering::Relaxed))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Alloc for MallocAlloc {
    #[cold]
    fn init(&mut self, cores: usize, memory: &mut [Page], _overwrite: bool) -> Result<()> {
        info!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if !memory.is_empty() {
            warn!("This allocator uses its own memory range");
        }

        let mut local = Vec::with_capacity(cores);
        local.resize_with(cores, || Local {
            counter: AtomicUsize::new(0),
        });
        self.local = local.into();

        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("Invalid size {size:?}");
            return Err(Error::Memory);
        }

        let ptr = unsafe { libc::malloc(Page::SIZE) };
        if !ptr.is_null() {
            self.local[core].counter.fetch_add(1, Ordering::Relaxed);
            Ok(ptr as u64)
        } else {
            Err(Error::Memory)
        }
    }

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<Size> {
        unsafe { libc::free(addr as *mut _) };
        self.local[core].counter.fetch_sub(1, Ordering::Relaxed);
        Ok(Size::L0)
    }

    fn pages(&self) -> usize {
        0
    }

    #[cold]
    fn dbg_allocated_pages(&self) -> usize {
        self.local
            .iter()
            .map(|l| l.counter.load(Ordering::Relaxed))
            .sum()
    }

    fn span(&self, _size: Size) -> usize {
        1
    }
}
