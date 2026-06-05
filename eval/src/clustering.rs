use std::sync::atomic::AtomicU64;

use bitfield_struct::bitfield;
use facet::Facet;
use llfree::atomic::{Atom, Atomic};
use llfree::{Cluster, Clustering, Policy, Request};

use crate::gfp::GFP;

#[derive(Clone, Debug, Facet)]
pub struct ClusteringConfig {
    clusters: Vec<ClusterConfig>,
    default: u8,
    perfect: (usize, usize), // (min, max) free, inclusive
    good: (usize, usize),    // (min, max) free, inclusive
}

impl ClusteringConfig {
    pub fn clustering(&self, cores: usize) -> Clustering {
        let clusters = self
            .clusters
            .iter()
            .map(|c| (Cluster(c.id), c.count.to_count(cores)))
            .collect::<Vec<_>>()
            .leak();

        #[bitfield(u64)]
        struct Range {
            #[bits(32)]
            min: usize,
            #[bits(32)]
            max: usize,
        }
        impl Atomic for Range {
            type I = AtomicU64;
        }
        impl Range {
            fn with((min, max): (usize, usize)) -> Self {
                Self::new().with_min(min).with_max(max)
            }
            fn contains(&self, value: usize) -> bool {
                self.min() <= value && value <= self.max()
            }
        }

        static PERFECT: Atom<Range> = Atom(AtomicU64::new(0));
        static GOOD: Atom<Range> = Atom(AtomicU64::new(0));

        PERFECT.store(Range::with(self.perfect));
        GOOD.store(Range::with(self.good));

        fn policy(requested: Cluster, target: Cluster, free: usize) -> Policy {
            if requested.0 > target.0 {
                return Policy::Steal;
            } else if requested.0 < target.0 {
                return Policy::Demote;
            }
            if PERFECT.load().contains(free) {
                return Policy::Match(u8::MAX);
            };
            if GOOD.load().contains(free) {
                return Policy::Match(2);
            }
            Policy::Match(1)
        }

        Clustering::new(clusters, Cluster(self.default), policy)
    }

    pub fn request(
        &self,
        order: usize,
        core: usize,
        cores: usize,
        pid: usize,
        gfp: u32,
    ) -> Request {
        for config in &self.clusters {
            if config.matches(order, gfp) {
                return Request::new(
                    order,
                    Cluster(config.id),
                    config.count.to_local(core, cores, pid),
                );
            }
        }
        // Default to first cluster
        let config = &self.clusters[0];
        Request::new(
            order,
            Cluster(config.id),
            config.count.to_local(core, cores, pid),
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
    fn to_count(self, cores: usize) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Cores => cores,
            Self::CoresHalf => cores.div_ceil(2),
            Self::Pids => cores,
        }
    }
    fn to_local(self, core: usize, cores: usize, pid: usize) -> Option<usize> {
        match self {
            Self::Zero => None,
            Self::One => Some(1),
            Self::Cores => Some(core % cores),
            Self::CoresHalf => Some(core.div_ceil(2) % cores.div_ceil(2)),
            Self::Pids => Some(pid % cores),
        }
    }
}

#[derive(Clone, Debug, Facet)]
struct ClusterConfig {
    id: u8,
    count: Count,
    order: Option<(usize, usize)>, // [min, max], inclusive
    /// And-ed list of or-ed GFP flags.
    /// One of each inner list must match.
    #[facet(default)]
    gfp: GfpMatch,
}

impl ClusterConfig {
    fn matches(&self, order: usize, gfp: u32) -> bool {
        if let Some((min, max)) = self.order
            && !(min..=max).contains(&order)
        {
            return false;
        }
        if !self.gfp.matches(gfp) {
            return false;
        }
        true
    }
}

#[derive(Facet, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
#[facet(rename_all = "snake_case")]
#[allow(unused)]
enum GfpMatch {
    On(GFP),
    Off(GFP),
    All(Vec<Self>),
    Any(Vec<Self>),
    Not(Box<Self>),
}
impl Default for GfpMatch {
    fn default() -> Self {
        Self::All(Vec::new()) // This evaluates to true.
    }
}
impl GfpMatch {
    fn matches(&self, gfp: u32) -> bool {
        match self {
            Self::On(flag) => *flag == gfp,
            Self::Off(flag) => *flag != gfp,
            Self::All(list) => list.iter().all(|m| m.matches(gfp)),
            Self::Any(list) => list.iter().any(|m| m.matches(gfp)),
            Self::Not(m) => !m.matches(gfp),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::clustering::GfpMatch;
    use crate::gfp::GFP;

    #[test]
    fn match_seralize() {
        let m = GfpMatch::On(GFP::DMA);
        let s = facet_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"on":"DMA"}"#);
        let m = GfpMatch::Not(GfpMatch::On(GFP::DMA).into());
        let s = facet_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"not":{"on":"DMA"}}"#);
        let m = GfpMatch::All(vec![GfpMatch::On(GFP::DMA), GfpMatch::On(GFP::HIGHMEM)]);
        let s = facet_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"all":[{"on":"DMA"},{"on":"HIGHMEM"}]}"#);
        let m = GfpMatch::Any(vec![GfpMatch::On(GFP::DMA), GfpMatch::On(GFP::HIGHMEM)]);
        let s = facet_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"any":[{"on":"DMA"},{"on":"HIGHMEM"}]}"#);
    }

    #[test]
    fn match_deserialize() {
        let s =
            r#"{"not":{"any":[{"off":"HIGHMEM"},{"on":"NOFAIL"},{"off":"FS"},{"on":"NORETRY" }]}}"#;
        let m: GfpMatch = facet_json::from_str(s).unwrap();
        assert_eq!(
            m,
            GfpMatch::Not(
                GfpMatch::Any(vec![
                    GfpMatch::Off(GFP::HIGHMEM),
                    GfpMatch::On(GFP::NOFAIL),
                    GfpMatch::Off(GFP::FS),
                    GfpMatch::On(GFP::NORETRY),
                ])
                .into()
            )
        );
    }

    #[test]
    fn gfp_match() {
        let m = GfpMatch::On(GFP::DMA);
        assert!(m.matches(GFP::DMA as _));
        assert!(!m.matches(GFP::HIGHMEM as _));
        let m = GfpMatch::Not(
            GfpMatch::Any(vec![
                GfpMatch::Off(GFP::HIGHMEM),
                GfpMatch::On(GFP::NOFAIL),
                GfpMatch::Off(GFP::FS),
                GfpMatch::On(GFP::NORETRY),
            ])
            .into(),
        );
        assert!(m.matches(GFP::HIGHMEM as u32 | GFP::FS as u32));
        assert!(!m.matches(GFP::FS as u32));
        assert!(!m.matches(GFP::HIGHMEM as u32 | GFP::FS as u32 | GFP::NOFAIL as u32));
        let complex = GfpMatch::All(vec![
            GfpMatch::On(GFP::MOVABLE),
            GfpMatch::Off(GFP::PAGE_CACHE),
            m,
        ]);
        assert!(complex.matches(GFP::MOVABLE as u32 | GFP::HIGHMEM as u32 | GFP::FS as u32));
        assert!(!complex.matches(
            GFP::MOVABLE as u32 | GFP::PAGE_CACHE as u32 | GFP::HIGHMEM as u32 | GFP::FS as u32
        ));
        assert!(!complex.matches(GFP::MOVABLE as u32 | GFP::HIGHMEM as u32));
    }
}
