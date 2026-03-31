use std::ops::{Add, Div};

pub mod mmap;
pub mod thread;

#[cfg(feature = "llc")]
mod llc;
#[cfg(feature = "llc")]
pub use llc::LLC;

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
