use core::marker::PhantomData;
use core::mem::size_of_val;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::*;
use core::{fmt, slice};

use log::error;

use crate::frame::Frame;
use crate::{
    Alloc, Error, FrameId, Init, MAX_ORDER, MetaData, MetaSize, Request, Result, Stats, Tier,
    Tiering,
};

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
    fn new(frames: usize, init: Init, tiering: &Tiering, meta: MetaData<'a>) -> Result<Self> {
        Ok(Self {
            alloc: A::new(frames, init, tiering, meta)?,
            offset: 0,
            _p: PhantomData,
        })
    }

    fn metadata_size(tiering: &Tiering, frames: usize) -> MetaSize {
        A::metadata_size(tiering, frames)
    }
    unsafe fn metadata(&mut self) -> MetaData<'a> {
        unsafe { self.alloc.metadata() }
    }
    fn get(&self, frame: Option<FrameId>, flags: Request) -> Result<(FrameId, Tier)> {
        let frame = frame
            .map(|f| {
                f.0.checked_sub(self.offset)
                    .map(FrameId)
                    .ok_or(Error::Argument)
            })
            .transpose()?;
        let (frame_id, tier) = self.alloc.get(frame, flags)?;
        Ok((FrameId(frame_id.0 + self.offset), tier))
    }
    fn put(&self, frame: FrameId, flags: Request) -> Result<()> {
        let frame = FrameId(frame.0.checked_sub(self.offset).ok_or(Error::Argument)?);
        self.alloc.put(frame, flags)
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn tree_stats(&self) -> crate::TreeStats {
        self.alloc.tree_stats()
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
    fn drain(&self) {
        self.alloc.drain();
    }
}

impl<'a, A: Alloc<'a>> ZoneAlloc<'a, A> {
    pub fn create(
        offset: usize,
        frames: usize,
        init: Init,
        tiering: &Tiering,
        meta: MetaData<'a>,
    ) -> Result<Self> {
        if !offset.is_multiple_of(1 << MAX_ORDER) {
            error!("zone alignment");
            return Err(Error::Initialization);
        }
        Ok(Self {
            alloc: A::new(frames, init, tiering, meta)?,
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

/// Non-volatile metadata used to recover the allocator after reboot.
#[repr(align(0x1000))]
struct Meta {
    /// A magic number used to check if the persistent memory contains the allocator state
    magic: AtomicUsize,
    /// Number of frames managed by the persistent allocator
    frames: AtomicUsize,
}
impl Meta {
    /// Magic marking the meta frame.
    const MAGIC: usize = 0x_dead_beef;
}
const _: () = assert!(core::mem::size_of::<Meta>() <= Frame::SIZE);

/// Persistent memory allocator, that is able to recover its state from the memory it manages.
pub struct NvmAlloc<'a, A: Alloc<'a>> {
    pub alloc: ZoneAlloc<'a, A>,
}

impl<'a, A: Alloc<'a>> NvmAlloc<'a, A> {
    pub fn create(
        zone: &'a mut [Frame],
        recover: bool,
        tiering: &Tiering,
        local: &'a mut [u8],
        trees: &'a mut [u8],
    ) -> Result<Self> {
        let m = A::metadata_size(tiering, zone.len());
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
            if meta.magic.load(Acquire) != Meta::MAGIC || frames != zone.len() {
                error!("no instance found");
                return Err(Error::Initialization);
            }
            Init::Recover
        } else {
            meta.magic.store(Meta::MAGIC, Release);
            meta.frames.store(zone.len(), Release);
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
            zone.as_ptr() as usize / Frame::SIZE,
            zone.len(),
            init,
            tiering,
            metadata,
        )?;
        Ok(Self { alloc })
    }
}

impl<'a, A: Alloc<'a>> Alloc<'a> for NvmAlloc<'a, A> {
    fn name() -> &'static str {
        A::name()
    }
    fn new(_frames: usize, _init: Init, _tiering: &Tiering, _meta: MetaData) -> Result<Self> {
        unimplemented!()
    }
    fn metadata_size(tiering: &Tiering, frames: usize) -> MetaSize {
        A::metadata_size(tiering, frames)
    }
    unsafe fn metadata(&mut self) -> MetaData<'a> {
        unsafe { self.alloc.metadata() }
    }
    fn get(&self, frame: Option<FrameId>, flags: Request) -> Result<(FrameId, Tier)> {
        self.alloc.get(frame, flags)
    }
    fn put(&self, frame: FrameId, flags: Request) -> Result<()> {
        self.alloc.put(frame, flags)
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn tree_stats(&self) -> crate::TreeStats {
        self.alloc.tree_stats()
    }
    fn stats(&self) -> Stats {
        self.alloc.stats()
    }
    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        self.alloc.stats_at(frame, order)
    }
    fn drain(&self) {
        self.alloc.drain();
    }
}

impl<'a, A: Alloc<'a>> fmt::Debug for NvmAlloc<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.alloc.fmt(f)
    }
}
