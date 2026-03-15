use core::cell::UnsafeCell;
use core::ffi::{CStr, c_char, c_void};
use core::mem::align_of;
use core::sync::atomic::{AtomicPtr, Ordering};
use core::{fmt, slice};

use llfree::util::Align;
use llfree::{
    Alloc, FrameId, HUGE_ORDER, Init, MetaData, MetaSize, Policy, PolicyFn, Request, Result, Stats,
    TREE_FRAMES, TREE_HUGE, Tier, TierStats, Tiering, TreeStats,
};

/// Global storage for the Rust policy function pointer.
/// This is safe because PolicyFn is a plain fn pointer (no closure state).
static POLICY_FN: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// C-compatible policy trampoline that calls the stored Rust PolicyFn.
extern "C" fn policy_trampoline(
    requested: u8,
    target: u8,
    free: usize,
) -> bindings::llfree_policy_t {
    let ptr = POLICY_FN.load(Ordering::Relaxed);
    assert!(!ptr.is_null(), "policy function not set");
    let policy_fn: PolicyFn = unsafe { core::mem::transmute(ptr) };
    let result = policy_fn(Tier(requested), Tier(target), free);
    match result {
        Policy::Match(prio) => bindings::llfree_policy_t {
            type_: bindings::llfree_policy_type_t_LLFREE_POLICY_MATCH,
            priority: prio,
        },
        Policy::Steal => bindings::llfree_policy_t {
            type_: bindings::llfree_policy_type_t_LLFREE_POLICY_STEAL,
            priority: 0,
        },
        Policy::Demote => bindings::llfree_policy_t {
            type_: bindings::llfree_policy_type_t_LLFREE_POLICY_DEMOTE,
            priority: 0,
        },
        Policy::Invalid => bindings::llfree_policy_t {
            type_: bindings::llfree_policy_type_t_LLFREE_POLICY_INVALID,
            priority: 0,
        },
    }
}

const SIZE: usize = 2 * align_of::<Align>();
/// C implementation of LLFree
///
/// Note: This abstraction assumes that the state is movable and smaller than two cache lines!
pub struct LLC {
    raw: UnsafeCell<[u8; SIZE]>,
    ms: bindings::llfree_meta_size,
}

unsafe impl Send for LLC {}
unsafe impl Sync for LLC {}

impl<'a> Alloc<'a> for LLC {
    fn name() -> &'static str {
        "LLC"
    }

    fn metadata_size(tiering: &Tiering, frames: usize) -> MetaSize {
        let c_tiering = bindings::convert_tiering(tiering);
        let m = unsafe { bindings::llfree_metadata_size(&c_tiering, frames as _) };
        assert!(m.llfree as usize <= SIZE);
        MetaSize {
            local: m.local,
            trees: m.trees,
            lower: m.lower,
        }
    }

    unsafe fn metadata(&mut self) -> MetaData<'a> {
        unsafe {
            let m = bindings::llfree_metadata(self.raw.get().cast());
            let ms = &self.ms;
            MetaData {
                local: slice::from_raw_parts_mut(m.local, ms.local),
                trees: slice::from_raw_parts_mut(m.trees, ms.trees),
                lower: slice::from_raw_parts_mut(m.lower, ms.lower),
            }
        }
    }

    fn new(frames: usize, init: Init, tiering: &Tiering, meta: MetaData<'a>) -> Result<Self> {
        let raw = UnsafeCell::new([0u8; SIZE]);

        let init = match init {
            Init::FreeAll => 0,
            Init::AllocAll => 1,
            Init::Recover => 2,
            Init::None => 4,
        };

        let c_tiering = bindings::convert_tiering(tiering);

        let m = unsafe { bindings::llfree_metadata_size(&c_tiering, frames as _) };
        assert!(SIZE >= m.llfree);

        assert!(meta.local.len() >= m.local);
        assert!(meta.trees.len() >= m.trees);
        assert!(meta.lower.len() >= m.lower);
        let meta = bindings::llfree_meta {
            local: meta.local.as_mut_ptr(),
            trees: meta.trees.as_mut_ptr(),
            lower: meta.lower.as_mut_ptr(),
        };

        let ret =
            unsafe { bindings::llfree_init(raw.get().cast(), frames, init, meta, &c_tiering) };
        ret.ok().map(|_| LLC { raw, ms: m })
    }

    fn get(&self, frame: Option<FrameId>, request: Request) -> Result<(FrameId, Tier)> {
        let frame = match frame {
            Some(f) => bindings::ll_some(f.0 as _),
            None => bindings::ll_none(),
        };
        let ret = unsafe { bindings::llfree_get(self.raw.get().cast(), frame, request.into()) };
        let f = ret.ok()?;
        Ok((FrameId(f as _), Tier(ret.tier())))
    }

    fn put(&self, frame: FrameId, request: Request) -> Result<()> {
        let ret =
            unsafe { bindings::llfree_put(self.raw.get().cast(), frame.0 as _, request.into()) };
        ret.ok().map(|_| ())
    }

    fn is_free(&self, frame: FrameId, order: usize) -> bool {
        let stats =
            unsafe { bindings::llfree_stats_at(self.raw.get().cast(), frame.0 as _, order as _) };
        order == 0 && stats.free_frames == 1
            || order == HUGE_ORDER && stats.free_huge == 1
            || order == TREE_FRAMES.ilog2() as usize && stats.free_huge == TREE_HUGE
    }

    fn drain(&self) {
        unsafe {
            bindings::llfree_drain(self.raw.get().cast());
        }
    }

    fn frames(&self) -> usize {
        unsafe { bindings::llfree_frames(self.raw.get().cast()) as _ }
    }

    fn tree_stats(&self) -> TreeStats {
        let s = unsafe { bindings::llfree_tree_stats(self.raw.get().cast()) };
        let mut tiers = [const {
            TierStats {
                free_frames: 0,
                alloc_frames: 0,
            }
        }; Tier::LEN];
        for (i, t) in s.tiers.iter().enumerate() {
            tiers[i] = TierStats {
                free_frames: t.free_frames,
                alloc_frames: t.alloc_frames,
            };
        }
        TreeStats {
            free_frames: s.free_frames,
            free_trees: s.free_trees,
            tiers,
        }
    }

    fn stats(&self) -> Stats {
        unsafe { bindings::llfree_stats(self.raw.get().cast()).into() }
    }

    fn stats_at(&self, frame: FrameId, order: usize) -> Stats {
        unsafe { bindings::llfree_stats_at(self.raw.get().cast(), frame.0 as _, order as _).into() }
    }

    fn validate(&self) {
        unsafe { bindings::llfree_validate(self.raw.get().cast()) }
    }
}

impl fmt::Debug for LLC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // wrapper function that is called by the c implementation
        extern "C" fn writer(arg: *mut c_void, msg: *const c_char) {
            let f = unsafe { &mut *arg.cast::<fmt::Formatter<'_>>() };
            let c_str = unsafe { CStr::from_ptr(msg) };
            write!(f, "{}", c_str.to_str().unwrap()).unwrap();
        }

        unsafe {
            bindings::llfree_print_debug(
                self.raw.get().cast(),
                Some(writer),
                (f as *mut fmt::Formatter).cast(),
            )
        };

        Ok(())
    }
}

mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(unnecessary_transmutes)]
    #![allow(clippy::ptr_offset_with_cast)]
    #![allow(clippy::useless_transmute)]
    #![allow(clippy::unnecessary_cast)]
    #![allow(clippy::transmute_int_to_bool)]

    use core::sync::atomic::Ordering;

    use ::llfree::Error;

    include!(concat!(env!("OUT_DIR"), "/llc.rs"));

    impl From<super::Request> for llfree_request {
        fn from(req: super::Request) -> Self {
            llfree_request {
                order: req.order as _,
                tier: req.tier.0,
                local: match req.local {
                    Some(l) => l,
                    None => LLFREE_LOCAL_NONE as _,
                },
            }
        }
    }

    impl llfree_result_t {
        pub fn ok(self) -> super::Result<u64> {
            match self.error() {
                LLFREE_ERR_OK => Ok(self.frame()),
                LLFREE_ERR_MEMORY => Err(Error::Memory),
                LLFREE_ERR_RETRY => Err(Error::Retry),
                LLFREE_ERR_ADDRESS => Err(Error::Address),
                LLFREE_ERR_INIT => Err(Error::Initialization),
                _ => unreachable!("invalid return code"),
            }
        }
    }

    impl From<ll_stats> for super::Stats {
        fn from(val: ll_stats) -> Self {
            super::Stats {
                free_frames: val.free_frames,
                free_huge: val.free_huge,
                free_trees: 0,
            }
        }
    }

    pub fn convert_tiering(tiering: &super::Tiering) -> llfree_tiering {
        // Store the Rust policy fn in the global so the trampoline can call it.
        super::POLICY_FN.store(tiering.policy as *mut (), Ordering::Relaxed);

        let mut c = llfree_tiering {
            tiers: [llfree_tier_conf { tier: 0, count: 0 }; LLFREE_MAX_TIERS as _],
            num_tiers: tiering.tiers().len() as _,
            default_tier: tiering.default.0,
            policy: Some(super::policy_trampoline),
        };
        for (i, &(tier, count)) in tiering.tiers().iter().enumerate() {
            c.tiers[i] = llfree_tier_conf {
                tier: tier.0,
                count: count as _,
            };
        }
        c
    }

    pub fn ll_none() -> ll_optional {
        ll_optional {
            _bitfield_align_1: [0; 0],
            _bitfield_1: ll_optional::new_bitfield_1(false, 0),
        }
    }
    pub fn ll_some(value: usize) -> ll_optional {
        ll_optional {
            _bitfield_align_1: [0; 0],
            _bitfield_1: ll_optional::new_bitfield_1(true, value as _),
        }
    }
}

#[cfg(all(test, feature = "llc"))]
mod test {
    use super::LLC;
    use llfree::{Alloc, Init, MetaData, Tiering};

    #[test]
    fn test_debug() {
        let (tiering, _request) = Tiering::simple(1);
        let meta = MetaData::alloc(LLC::metadata_size(&tiering, 1024));
        let alloc = LLC::new(1024, Init::FreeAll, &tiering, meta).unwrap();
        println!("{alloc:?}");
    }
}
