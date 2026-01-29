use core::marker::PhantomData;
use core::mem::size_of_val;
use core::sync::atomic::Ordering::*;
use core::sync::atomic::{AtomicBool, AtomicUsize};
use core::{fmt, slice};

use log::error;

use crate::frame::Frame;
use crate::{Alloc, Error, Flags, FrameId, Init, MAX_ORDER, MetaData, MetaSize, Result, Stats};

/// Zone allocator, managing a range of memory at a given page frame offset.
pub struct ZoneAlloc<'a, A: Alloc<'a>> {
    pub alloc: A,
    pub offset: usize,
    _p: PhantomData<&'a ()>,
}

impl<'a, A: Alloc<'a>> Alloc<'a> for ZoneAlloc<'a, A> {
    fn name() -> &'static str {
        A::name()
    }
    fn new(cores: usize, frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self> {
        Ok(Self {
            alloc: A::new(cores, frames, init, meta)?,
            offset: 0,
            _p: PhantomData,
        })
    }

    fn metadata_size(cores: usize, frames: usize) -> MetaSize {
        A::metadata_size(cores, frames)
    }
    unsafe fn metadata(&mut self) -> MetaData<'a> {
        unsafe { self.alloc.metadata() }
    }
    fn get(&self, core: usize, frame: Option<FrameId>, flags: Flags) -> Result<FrameId> {
        let frame = frame
            .map(|f| {
                f.0.checked_sub(self.offset)
                    .map(FrameId)
                    .ok_or(Error::Address)
            })
            .transpose()?;
        Ok(FrameId(self.alloc.get(core, frame, flags)?.0 + self.offset))
    }
    fn put(&self, core: usize, frame: FrameId, flags: Flags) -> Result<()> {
        let frame = FrameId(frame.0.checked_sub(self.offset).ok_or(Error::Address)?);
        self.alloc.put(core, frame, flags)
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn cores(&self) -> usize {
        self.alloc.cores()
    }
    fn fast_stats(&self) -> Stats {
        self.alloc.fast_stats()
    }
    fn fast_stats_at(&self, frame: FrameId, order: usize) -> Stats {
        let Some(frame) = frame.0.checked_sub(self.offset).map(FrameId) else {
            return Stats::default();
        };
        self.alloc.fast_stats_at(frame, order)
    }
    fn stats(&self) -> Stats {
        self.alloc.stats()
    }
    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        let Some(frame) = frame.0.checked_sub(self.offset).map(FrameId) else {
            return Stats::default();
        };
        self.alloc.stats_at(frame, order)
    }
    fn is_free(&self, frame: FrameId, order: usize) -> bool {
        let Some(frame) = frame.0.checked_sub(self.offset).map(FrameId) else {
            return false;
        };
        self.alloc.is_free(frame, order)
    }
    fn drain(&self, core: usize) -> Result<()> {
        self.alloc.drain(core)
    }
}

impl<'a, A: Alloc<'a>> ZoneAlloc<'a, A> {
    pub fn create(
        cores: usize,
        offset: usize,
        frames: usize,
        init: Init,
        meta: MetaData<'a>,
    ) -> Result<Self> {
        if !offset.is_multiple_of(1 << MAX_ORDER) {
            error!("zone alignment");
            return Err(Error::Initialization);
        }
        Ok(Self {
            alloc: A::new(cores, frames, init, meta)?,
            offset,
            _p: PhantomData,
        })
    }
}
impl<'a, A: Alloc<'a>> fmt::Debug for ZoneAlloc<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.alloc.fmt(f)
    }
}

/// Non-Volatile metadata that is used to recover the allocator at reboot
#[repr(align(0x1000))]
struct Meta {
    /// A magic number used to check if the persistent memory contains the allocator state
    magic: AtomicUsize,
    /// Number of frames managed by the persistent allocator
    frames: AtomicUsize,
    /// Flag that stores if the system has crashed or was shutdown correctly
    crashed: AtomicBool,
}
impl Meta {
    /// Magic marking the meta frame.
    const MAGIC: usize = 0x_dead_beef;
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Frame::SIZE);

/// Persistent memory allocator, that is able to recover its state from the memory it manages.
pub struct NvmAlloc<'a, A: Alloc<'a>> {
    pub alloc: ZoneAlloc<'a, A>,
    meta: &'a Meta,
}

impl<'a, A: Alloc<'a>> NvmAlloc<'a, A> {
    pub fn create(
        cores: usize,
        zone: &'a mut [Frame],
        recover: bool,
        local: &'a mut [u8],
        trees: &'a mut [u8],
    ) -> Result<Self> {
        let m = A::metadata_size(cores, zone.len());
        if size_of_val(zone) < m.lower + Frame::SIZE
            || !(zone.as_ptr() as usize).is_multiple_of(Frame::SIZE << MAX_ORDER)
        {
            error!("invalid memory region");
            return Err(Error::Initialization);
        }

        let (meta, zone) = zone.split_last_mut().ok_or(Error::Memory)?;
        let meta = meta.cast::<Meta>();

        let init = if recover {
            let frames = meta.frames.load(Acquire);
            let crashed = meta.crashed.swap(true, AcqRel);
            if meta.magic.load(Acquire) != Meta::MAGIC || frames != zone.len() {
                error!("no instance found");
                return Err(Error::Initialization);
            }
            Init::Recover(crashed)
        } else {
            meta.magic.store(Meta::MAGIC, Release);
            meta.frames.store(zone.len(), Release);
            meta.crashed.store(true, Release);
            Init::FreeAll
        };

        let (zone, p) = zone.split_at_mut(zone.len() - m.lower.div_ceil(Frame::SIZE));
        let lower = unsafe { slice::from_raw_parts_mut(p.as_mut_ptr().cast(), m.lower) };
        let metadata = MetaData {
            local,
            trees,
            lower,
        };

        let alloc = ZoneAlloc::create(
            cores,
            zone.as_ptr() as usize / Frame::SIZE,
            zone.len(),
            init,
            metadata,
        )?;
        Ok(Self { alloc, meta })
    }
}

impl<'a, A: Alloc<'a>> Alloc<'a> for NvmAlloc<'a, A> {
    fn name() -> &'static str {
        A::name()
    }
    fn new(_cores: usize, _frames: usize, _init: Init, _meta: MetaData) -> Result<Self> {
        unimplemented!()
    }
    fn metadata_size(cores: usize, frames: usize) -> MetaSize {
        A::metadata_size(cores, frames)
    }
    unsafe fn metadata(&mut self) -> MetaData<'a> {
        unsafe { self.alloc.metadata() }
    }
    fn get(&self, core: usize, frame: Option<FrameId>, flags: Flags) -> Result<FrameId> {
        self.alloc.get(core, frame, flags)
    }
    fn put(&self, core: usize, frame: FrameId, flags: Flags) -> Result<()> {
        self.alloc.put(core, frame, flags)
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn cores(&self) -> usize {
        self.alloc.cores()
    }
    fn fast_stats(&self) -> Stats {
        self.alloc.fast_stats()
    }
    fn fast_stats_at(&self, frame: FrameId, order: usize) -> Stats {
        self.alloc.fast_stats_at(frame, order)
    }
    fn stats(&self) -> Stats {
        self.alloc.stats()
    }
    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        self.alloc.stats_at(frame, order)
    }
    fn is_free(&self, frame: FrameId, order: usize) -> bool {
        self.alloc.is_free(frame, order)
    }
    fn drain(&self, core: usize) -> Result<()> {
        self.alloc.drain(core)
    }
}

impl<'a, A: Alloc<'a>> fmt::Debug for NvmAlloc<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.alloc.fmt(f)
    }
}

impl<'a, A: Alloc<'a>> Drop for NvmAlloc<'a, A> {
    fn drop(&mut self) {
        self.meta.crashed.store(false, Release);
    }
}
