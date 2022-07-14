#include <asm/page_types.h>

/// Error codes
enum nvalloc_error_t
{
    NVALLOC_ERROR_MEMORY = 1,
    NVALLOC_ERROR_CAS = 2,
    NVALLOC_ERROR_ADDRESS = 3,
    NVALLOC_ERROR_INIT = 4,
    NVALLOC_ERROR_CORRUPTION = 5,
};

/// Returns if the allocator was initialized
u32 nvalloc_initialized(void);
/// Allocates 2^order pages. Returns >=PAGE_SIZE on success an error code.
u8 *nvalloc_get(u32 core, u32 order);
/// Frees a previously allocated page. Returns 0 on success or an error code.
u64 nvalloc_put(u32 core, u8 *addr, u32 order);

inline bool nvalloc_err(u64 ret)
{
    return 0 < ret && ret < PAGE_SIZE;
}
