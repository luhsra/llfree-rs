use core::marker::PhantomData;
use core::mem::size_of_val;
use core::sync::atomic::Ordering::*;
use core::sync::atomic::{AtomicBool, AtomicUsize};
use core::{fmt, slice};

use log::error;

use crate::frame::Frame;
use crate::{Alloc, Error, Init, MetaSize, Result};

/// Zone allocator, managing a range of memory at a given page frame offset.
pub struct ZoneAlloc<'a, A: Alloc<'a>> {
    pub alloc: A,
    pub offset: usize,
    _p: PhantomData<&'a ()>,
}

impl<'a, A: Alloc<'a>> ZoneAlloc<'a, A> {
    pub fn metadata_size(cores: usize, frames: usize) -> MetaSize {
        A::metadata_size(cores, frames)
    }

    pub fn new(
        cores: usize,
        zone: &'a mut [Frame],
        free_all: bool,
        primary: &'a mut [u8],
        secondary: &'a mut [u8],
    ) -> Result<Self> {
        if zone.as_ptr() as usize % (Frame::SIZE << A::MAX_ORDER) != 0 {
            error!("zone alignment");
            return Err(Error::Initialization);
        }

        let init = if free_all {
            Init::FreeAll
        } else {
            Init::AllocAll
        };
        let alloc = A::new(cores, zone.len(), init, primary, secondary)?;
        Ok(Self {
            alloc,
            offset: zone.as_ptr() as usize / Frame::SIZE,
            _p: PhantomData,
        })
    }

    pub fn get(&self, core: usize, order: usize) -> Result<usize> {
        Ok(self.alloc.get(core, order)? + self.offset)
    }
    pub fn put(&self, core: usize, frame: usize, order: usize) -> Result<()> {
        self.alloc.put(
            core,
            frame.checked_sub(self.offset).ok_or(Error::Address)?,
            order,
        )
    }
    pub fn free_frames(&self) -> usize {
        self.alloc.free_frames()
    }
    pub fn frames(&self) -> usize {
        self.alloc.frames()
    }
    pub fn allocated_frames(&self) -> usize {
        self.alloc.allocated_frames()
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
    pub fn metadata_size(cores: usize, frames: usize) -> usize {
        A::metadata_size(cores, frames).secondary
    }

    pub fn new(
        cores: usize,
        zone: &'a mut [Frame],
        recover: bool,
        volatile: &'a mut [u8],
    ) -> Result<Self> {
        let m = A::metadata_size(cores, zone.len());
        if size_of_val(zone) < m.primary + Frame::SIZE
            || zone.as_ptr() as usize % (Frame::SIZE << A::MAX_ORDER) != 0
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

        let (zone, p) = zone.split_at_mut(zone.len() - m.primary.div_ceil(Frame::SIZE));
        let primary = unsafe { slice::from_raw_parts_mut(p.as_mut_ptr().cast(), m.primary) };

        let alloc = ZoneAlloc {
            alloc: A::new(cores, zone.len(), init, primary, volatile)?,
            offset: zone.as_ptr() as usize / Frame::SIZE,
            _p: PhantomData,
        };
        Ok(Self { alloc, meta })
    }

    pub fn get(&self, core: usize, order: usize) -> Result<usize> {
        self.alloc.get(core, order)
    }
    pub fn put(&self, core: usize, frame: usize, order: usize) -> Result<()> {
        self.alloc.put(core, frame, order)
    }
    pub fn free_frames(&self) -> usize {
        self.alloc.free_frames()
    }
    pub fn frames(&self) -> usize {
        self.alloc.frames()
    }
    pub fn allocated_frames(&self) -> usize {
        self.alloc.allocated_frames()
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
