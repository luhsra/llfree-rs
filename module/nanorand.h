#include <linux/types.h>

/// Generates new random number modifying the seed for the next call.
///
/// @see
/// - https://github.com/wangyi-fudan/wyhash
/// - https://github.com/Absolucy/nanorand-rs
inline u64 nanorand_random(u64 *seed) {
    __uint128_t t;
    *seed = *seed + 0xa0761d6478bd642full;
    t = ((__uint128_t)*seed) * ((__uint128_t)(*seed ^ 0xe7037ed1a0b428dbull));
    return (u64)((t >> 64) ^ t);
}

/// Generates new random number in this range, modifying the seed for the next
/// call.
inline u64 nanorand_random_range(u64 *seed, u64 start, u64 end) {
    u64 val = nanorand_random(seed);
    val %= end - start;
    val += start;
    return val;
}
