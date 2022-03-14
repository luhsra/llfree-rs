use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

use log::{error, warn};

use super::{Alloc, Error, Result, Size};
use crate::table::Table;
use crate::util::Page;

/// Wrapper for libc malloc.
pub struct MallocAlloc {
    local: Vec<Local>,
}

#[repr(align(64))]
struct Local {
    counter: AtomicUsize,
}

impl MallocAlloc {
    pub fn new() -> Self {
        Self { local: Vec::new() }
    }
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
        warn!(
            "initializing c={cores} {:?} {}",
            memory.as_ptr_range(),
            memory.len()
        );
        if !memory.is_empty() {
            warn!("This allocator uses its own memory range");
        }

        self.local = Vec::with_capacity(cores);
        self.local.resize_with(cores, || Local {
            counter: AtomicUsize::new(0),
        });

        Ok(())
    }

    #[inline(never)]
    fn get(&self, core: usize, size: Size) -> Result<u64> {
        if size != Size::L0 {
            error!("Invalid size {size:?}");
            return Err(Error::Memory);
        }

        let ptr = unsafe { libc::malloc(Table::m_span(0)) };
        if !ptr.is_null() {
            self.local[core]
                .counter
                .fetch_add(Table::span(0), Ordering::Relaxed);
            Ok(ptr as u64)
        } else {
            Err(Error::Memory)
        }
    }

    #[inline(never)]
    fn put(&self, core: usize, addr: u64) -> Result<()> {
        unsafe { libc::free(addr as *mut _) };
        self.local[core]
            .counter
            .fetch_sub(Table::span(0), Ordering::Relaxed);
        Ok(())
    }

    fn pages(&self) -> usize {
        0
    }

    #[cold]
    fn allocated_pages(&self) -> usize {
        self.local
            .iter()
            .map(|l| l.counter.load(Ordering::Relaxed))
            .sum()
    }
}
