#include "nanorand.h"

// @see
// - https://github.com/wangyi-fudan/wyhash
// - https://github.com/Absolucy/nanorand-rs

struct nanorand nanorand_new(u64 seed) {
    struct nanorand rng = {seed};
    return rng;
}

u64 nanorand_random(struct nanorand *rng) {
    __uint128_t t;
    rng->seed = rng->seed + 0xa0761d6478bd642full;
    t = ((__uint128_t)rng->seed) *
        ((__uint128_t)(rng->seed ^ 0xe7037ed1a0b428dbull));
    return (u64)((t >> 64) ^ t);
}

u64 nanorand_random_range(struct nanorand *rng, u64 start, u64 end) {
    u64 val = nanorand_random(rng);
    val %= end - start;
    val += start;
    return val;
}
