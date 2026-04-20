use std::ops::{Add, Div};
use std::sync::OnceLock;

pub mod gfp;
pub mod mmap;
pub mod thread;

#[cfg(feature = "llc")]
mod llc;
#[cfg(feature = "llc")]
pub use llc::LLC;
#[cfg(feature = "llzig")]
mod llzig;
#[cfg(feature = "llzig")]
pub use llzig::LLZig;

use gfp::GFP;
use llfree::{Policy, Request, Tier, Tiering};

use facet::Facet;

pub fn avg_bounds<T>(iter: impl IntoIterator<Item = T>) -> Option<(T, T, T)>
where
    T: Ord + Add<T, Output = T> + Div<T, Output = T> + TryFrom<usize> + Copy,
{
    let mut iter = iter.into_iter();
    if let Some(first) = iter.next() {
        let mut min = first;
        let mut max = first;
        let mut mean = first;
        let mut count = 1;

        for x in iter {
            min = min.min(x);
            max = max.max(x);
            mean = mean + x;
            count += 1;
        }
        let Ok(count) = T::try_from(count) else {
            unreachable!("overflow")
        };

        Some((min, mean / count, max))
    } else {
        None
    }
}

#[derive(Clone, Debug, Facet)]
pub struct TieringConfig {
    tiers: Vec<TierConfig>,
    default: u8,
    pids: usize,
    perfect: (usize, usize), // (min, max) free
    good: (usize, usize),    // (min, max) free
}

impl TieringConfig {
    pub fn tiering(&self, cores: usize) -> Tiering {
        let tiers = self
            .tiers
            .iter()
            .map(|c| (Tier(c.id), c.count.to_count(cores, self.pids)))
            .collect::<Vec<_>>()
            .leak();

        static PERFECT: OnceLock<(usize, usize)> = OnceLock::new();
        static GOOD: OnceLock<(usize, usize)> = OnceLock::new();

        PERFECT.set(self.perfect).unwrap();
        GOOD.set(self.good).unwrap();

        fn policy(requested: Tier, target: Tier, free: usize) -> Policy {
            if requested.0 > target.0 {
                return Policy::Steal;
            } else if requested.0 < target.0 {
                return Policy::Demote;
            }
            if let Some(&(perfect_min, perfect_max)) = PERFECT.get()
                && free >= perfect_min
                && free <= perfect_max
            {
                return Policy::Match(u8::MAX);
            }
            if let Some(&(good_min, good_max)) = GOOD.get()
                && free >= good_min
                && free <= good_max
            {
                return Policy::Match(2);
            }
            Policy::Match(1)
        }

        Tiering::new(tiers, Tier(self.default), policy)
    }

    pub fn request(
        &self,
        order: usize,
        core: usize,
        cores: usize,
        pid: usize,
        gfp: u32,
    ) -> Request {
        for config in &self.tiers {
            if config.matches(order, gfp) {
                return Request::new(
                    order,
                    Tier(config.id),
                    config.count.to_local(core, cores, pid, self.pids),
                );
            }
        }
        // Default to first tier
        let config = &self.tiers[0];
        Request::new(
            order,
            Tier(config.id),
            config.count.to_local(core, cores, pid, self.pids),
        )
    }
}

#[derive(Clone, Copy, Debug, Facet)]
#[allow(dead_code)]
#[repr(u8)]
#[facet(rename_all = "snake_case")]
enum Count {
    Zero,
    One,
    Cores,
    CoresHalf,
    Pids,
}
impl Count {
    fn to_count(self, cores: usize, pids: usize) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Cores => cores,
            Self::CoresHalf => cores.div_ceil(2),
            Self::Pids => pids,
        }
    }
    fn to_local(self, core: usize, cores: usize, pid: usize, pids: usize) -> Option<usize> {
        match self {
            Self::Zero => None,
            Self::One => Some(1),
            Self::Cores => Some(core % cores),
            Self::CoresHalf => Some(core.div_ceil(2) % cores.div_ceil(2)),
            Self::Pids => Some(pid % pids),
        }
    }
}

#[derive(Clone, Debug, Facet)]
struct TierConfig {
    id: u8,
    count: Count,
    order: Option<(usize, usize)>, // (min, max)
    /// And-ed list of or-ed GFP flags.
    /// One of each inner list must match.
    #[facet(default)]
    gfp: Vec<Vec<GfpFlag>>,
}

impl TierConfig {
    fn matches(&self, order: usize, gfp: u32) -> bool {
        if let Some((min, max)) = self.order
            && !(min..max).contains(&order)
        {
            return false;
        }
        for group in &self.gfp {
            if !group.iter().any(|flag| flag.matches(gfp)) {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug, Facet)]
#[repr(u8)]
#[facet(rename_all = "snake_case")]
enum GfpFlag {
    On(GFP),
    Off(GFP),
}
impl GfpFlag {
    fn matches(&self, gfp: u32) -> bool {
        match self {
            Self::On(flag) => *flag == gfp,
            Self::Off(flag) => *flag != gfp,
        }
    }
}
