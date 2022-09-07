# Motivating NVAlloc

## Design Goals

- Lock & log free
- Reducing redundancy & state
  - No lru lists, pcp lists
  - Lower memory overhead
  - Cache-friendly
- Crash consistency
  - Can recover from power loss without loosing more than #cpu pages
- Reducing complexity
- Automatic defragmentation
  - No draining
  - No frequent compaction necessary
- Scalable
  - Multicore
  - Memory size


## Memory Overhead

### NVAlloc

- Atom
  - Static: 64B
  - Per Tree:
    - L1: 128^2 / 8 = 2KB
    - L2: 128 * 1 = 128B
  - Per GB: 34KB per GB
- ArrayList
  - Static: 128B
  - Per GB: 128B (L3: 16*8B)
  - Per CPU: 128B (aligned)
- 16C 64GB: 2.134MB
  - 64B + 128B = 196B
  - 34KB * 64 = 2176K
  - 128B * 64 = 8K
  - 128B * 16 = 2K

### Linux

- Static: 832B
  - zone->free_area: 792B
  - zone->lock: 4B (64B aligned)
- Per CPU:
  - zone->per_cpu_pagesets: 256B
- Per GB: 8MB
  - page->pcp_list: 16B
  - page->buddy_list: 16B
- 16C 64GB: 512.1MB
  - 832B
  - 256 * 16B = 4K
  - 64 * 8MB = 512MB
