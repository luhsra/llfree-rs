#include <linux/types.h>

struct nanorand {
    u64 seed;
};

struct nanorand nanorand_new(u64 seed);
u64 nanorand_random(struct nanorand *rng);
u64 nanorand_random_range(struct nanorand *rng, u64 start, u64 end);
