use core::marker::PhantomData;
use core::mem::size_of_val;
use core::sync::atomic::Ordering::*;
use core::sync::atomic::{AtomicBool, AtomicUsize};
use core::{fmt, slice};

use log::error;

use crate::frame::Frame;
use crate::{Alloc, Error, Flags, Init, MetaData, MetaSize, Result, MAX_ORDER};

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
    fn new(frames: usize, init: Init, meta: MetaData<'a>) -> Result<Self> {
        Ok(Self {
            alloc: A::new(frames, init, meta)?,
            offset: 0,
            _p: PhantomData,
        })
    }

    fn metadata_size(frames: usize) -> MetaSize {
        A::metadata_size(frames)
    }
    fn metadata(&mut self) -> MetaData<'a> {
        self.alloc.metadata()
    }
    fn get(&self, flags: Flags) -> Result<usize> {
        Ok(self.alloc.get(flags)? + self.offset)
    }
    fn put(&self, frame: usize, flags: Flags) -> Result<()> {
        let frame = frame.checked_sub(self.offset).ok_or(Error::Address)?;
        self.alloc.put(frame, flags)
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn free_frames(&self) -> usize {
        self.alloc.free_frames()
    }
    fn free_huge(&self) -> usize {
        self.alloc.free_huge()
    }
    fn is_free(&self, frame: usize, order: usize) -> bool {
        let Some(frame) = frame.checked_sub(self.offset) else {
            return false;
        };
        self.alloc.is_free(frame, order)
    }
    fn free_at(&self, frame: usize, order: usize) -> usize {
        let Some(frame) = frame.checked_sub(self.offset) else {
            return 0;
        };
        self.alloc.free_at(frame, order)
    }
    fn drain(&self) -> Result<()> {
        self.alloc.drain()
    }
}

impl<'a, A: Alloc<'a>> ZoneAlloc<'a, A> {
    pub fn create(
        offset: usize,
        frames: usize,
        init: Init,
        meta: MetaData<'a>,
    ) -> Result<Self> {
        if offset % (1 << MAX_ORDER) != 0 {
            error!("zone alignment");
            return Err(Error::Initialization);
        }
        Ok(Self {
            alloc: A::new(frames, init, meta)?,
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
        zone: &'a mut [Frame],
        recover: bool,
        trees: &'a mut [u8],
    ) -> Result<Self> {
        let m = A::metadata_size(zone.len());
        if size_of_val(zone) < m.lower + Frame::SIZE
            || zone.as_ptr() as usize % (Frame::SIZE << MAX_ORDER) != 0
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
            trees,
            lower,
        };

        let alloc = ZoneAlloc::create(
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
    fn new(_frames: usize, _init: Init, _meta: MetaData) -> Result<Self> {
        unimplemented!()
    }
    fn metadata_size(frames: usize) -> MetaSize {
        A::metadata_size(frames)
    }
    fn metadata(&mut self) -> MetaData<'a> {
        self.alloc.metadata()
    }
    fn get(&self, flags: Flags) -> Result<usize> {
        self.alloc.get(flags)
    }
    fn put(&self, frame: usize, flags: Flags) -> Result<()> {
        self.alloc.put(frame, flags)
    }
    fn frames(&self) -> usize {
        self.alloc.frames()
    }
    fn free_frames(&self) -> usize {
        self.alloc.free_frames()
    }
    fn free_huge(&self) -> usize {
        self.alloc.free_huge()
    }
    fn is_free(&self, frame: usize, order: usize) -> bool {
        self.alloc.is_free(frame, order)
    }
    fn free_at(&self, frame: usize, order: usize) -> usize {
        self.alloc.free_at(frame, order)
    }
    fn drain(&self) -> Result<()> {
        self.alloc.drain()
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
