use facet::Facet;

/// Linux GFP flags for memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Facet)]
#[allow(clippy::upper_case_acronyms, non_camel_case_types, dead_code)]
#[repr(u32)]
pub enum GFP {
    DMA = 0x01,
    HIGHMEM = 0x02,
    DMA32 = 0x04,
    MOVABLE = 0x08,
    RECLAIMABLE = 0x10,
    HIGH = 0x20,
    IO = 0x40,
    FS = 0x80,
    ZERO = 0x100,
    ATOMIC = 0x200,
    DIRECT_RECLAIM = 0x400,
    KSWAPD_RECLAIM = 0x800,
    WRITE = 0x1000,
    NOWARN = 0x2000,
    RETRY_MAYFAIL = 0x4000,
    NOFAIL = 0x8000,
    NORETRY = 0x10000,
    MEMALLOC = 0x20000,
    COMP = 0x40000,
    NOMEMALLOC = 0x80000,
    HARDWALL = 0x100000,
    THISNODE = 0x200000,
    ACCOUNT = 0x400000,
    ZEROTAGS = 0x800000,
    PAGE_CACHE = 0x10000000,
}
impl PartialEq<u32> for GFP {
    fn eq(&self, other: &u32) -> bool {
        (*self as u32) & *other != 0
    }
}
impl PartialEq<GFP> for u32 {
    fn eq(&self, other: &GFP) -> bool {
        *self & (*other as u32) != 0
    }
}
