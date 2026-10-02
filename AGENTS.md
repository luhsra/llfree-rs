# AGENTS.md

Operational guide for coding agents working in `llfree-rs`.

Repo scope:

- Rust allocator in `core/` (`-p llfree`)
- C allocator in `llc/`
- Evaluation/integration harness in `eval/` (`-p llfree-eval`)

## Architecture / Design

The Rust allocator is being redesigned for shared CXL memory without cache
coherence (see `CXL.md`). Allocation/free fast paths operate on host-local tree
clones; shared metadata is accessed under cross-host locking with explicit cache
flush/invalidation. Do not assume CPU atomics alone synchronize different hosts.
The C implementation and eval adapters have not yet been migrated to this design.

### Metadata and components

Metadata sizes (`MetaSize`) and buffer lengths are measured in 64-byte cache lines.
`MetaData` separates private DRAM from shared CXL storage:

| Location | Contents | Implementation |
| --- | --- | --- |
| Host DRAM (`local`) | Per-class reservation clones and batched free clones | `core/src/local.rs`: `Locals`, `TreeClone`, `Reservation`, `FreeClone` |
| Shared CXL (`remote`) | Bakery-lock slots, packed tree entries, then lower trees | `core/src/llfree.rs`: `CXLMetaSize`; `core/src/trees.rs`: `Trees` |
| Lower tree (shared or cloned) | Huge-entry counters/flags and allocation bitfields | `core/src/lower.rs`: `LowerTree`; `core/src/bitfield.rs` |

- `Alloc::new(hosts, host_id, frames, init, classing, meta)` identifies the host.
  Host-local buffers must be private; participating hosts share the remote buffer.
- `Classing` configures classes, reservation-slot counts, policy, and `free_clones`
  (normally one per CPU). Built-ins are `Classing::simple` and `Classing::movable`.
  Policies are `Match(priority)`, `Steal`, `Demote`, and `Invalid`.
- `Request.local` chooses an allocation reservation. `put(..., free_idx)` separately
  selects the free-clone slot; free batching is not an LRU replacement policy.

### Clone ownership and allocation/free flow

- **Reserve:** under the CXL lock, reserve an unreserved global tree and transfer
  its free lower state with `get_all_copy()`. Allocate from the resulting clone,
  not the emptied shared source. Install the post-allocation row and free counter.
  On failure, merge the clone back before restoring the global counter.
- **Allocate:** try the requested local reservation, synchronizing if needed;
  otherwise search shared trees for reservation/direct allocation. Exhaustion
  falls back to stealing from local reservations, then demoting local reservations.
  Direct global stealing allocates from shared lower state without cloning it.
- **Free:** prefer a matching allocation reservation, then the selected free clone.
  A new free clone starts in `Init::AllocAll` state and records returned frames.
  Switching that slot to another tree merges the displaced clone into its own
  original shared tree before incrementing that tree's global counter.
- **Synchronize:** a reserved tree can accumulate shared frees from other hosts.
  Hold the CXL lock, exclusively access the local slot, and revalidate its tree
  identity before transferring shared lower state into the clone. Publish added
  local credit only after the lower state is merged; counters alone are insufficient.
- **Replace/demote/drain:** merge every displaced clone into the tree identified by
  its stored row, not the newly allocated frame. `drain()` merges free clones and
  returns reservation capacity before unreserving. Never discard a live clone.

### Synchronization, cache access, and addressing

- `LLFree::cxl_lock` is a host-local `SpinMutex<CXLock>`. Acquire that mutex before
  the Bakery `CXLock`; the latter provides cross-host exclusion for shared trees
  and lower metadata. Each participant needs a unique host ID for the same lock.
- Local slots use atomic `LocalTree` state: a users counter permits concurrent
  lower operations; `switching` excludes them for replacement, synchronization,
  or detailed statistics. Retry switching slots before testing capacity/identity.
  Preserve these CAS/retry protocols; the allocator as a whole is not lock-free.
- `core/src/cxl.rs`: `Uncached` marks non-coherent storage. Under exclusion,
  `borrow()` invalidates before access and returns `Cached`, which flushes and
  invalidates on drop. Flush modified shared state before releasing the CXL lock.
- `core/src/cache.rs`: `Align`, `CacheLine`, and the unsafe `Aligned` trait describe
  valid, cache-aligned spans without unrelated data. Implementations must uphold
  pointer/span/lifetime invariants. Alignment does not imply zero-initializability.
  `flush_invalidate()` uses `clflushopt` plus a memory fence; `write_back()` alone
  is not synchronization. Avoid false sharing across independently owned CXL data.
- Public frame IDs and stored reservation rows are global. `LowerTree` accepts
  tree-relative frames/rows. Convert inputs at every lower boundary and restore
  the tree base on allocation results before returning or storing them.

### Statistics and invariants

- Shared counters describe shared lower free state; each local counter describes
  its own lower clone. Free capacity has one owner: shared state or one clone.
- `tree_stats()`, `stats()`, and `stats_at()` include shared state plus this host's
  reservation/free clones, including pending frees. Other hosts' private DRAM
  clones are invisible. Concurrent local operations can make snapshots approximate;
  `validate()` requires quiescent local operations.
- Aggregate free capacity per huge page/tree before counting fully free regions;
  adding each clone's `free_huge`/`free_trees` would miss split free capacity.
  Class allocated/free totals must account for local reservations, pending frees,
  and the actual capacity of a partial final tree. Ignore tree-chunk padding.
- `Online` restores global counters from shared lower state only, not clone-inclusive
  statistics; otherwise pending frees are counted twice when later merged.

### Initialization and current limitations

- One initializer uses `Init::FreeAll` or `Init::AllocAll`; other hosts join already
  initialized shared metadata with `Init::None`. Do not reset an active Bakery lock.
- `LowerTree` has recovery logic, but `LLFree::new(Init::Recover)` currently only
  joins existing shared state. It does not rebuild counters, recover private clones,
  or clear failed-host tickets. Crash recovery is not implemented end to end.
- Preserve existing allocator behavior when changing Rust/C interfaces, but do not
  assume the legacy C backend implements the CXL ownership/locking protocol.

## Build, Lint, Test Commands

### Rust workspace

- Use standard cargo commands, with `-p llfree` or `-p llfree-eval`
- Core tests (including doctests): `cargo test -p llfree`
- Alternate page size: `cargo test -p llfree --features 16K --lib`
- No-std check: `cargo check -p llfree --no-default-features`
- Eval adapters currently need migration to the new allocator API/metadata layout;
  workspace and eval builds may fail independently of core tests.
- Run integration tests with:
    - `cargo test -p llfree-eval --test integration [test_name]`

### Eval with C backend

- Init/update C submodule (if needed):
    - `git submodule update --init --checkout llc`
- Run one eval test against C impl:
    - `cargo test -p llfree-eval -F llc --test integration <test_name_substring>`

### C implementation (`llc/`)

- Build static C library:
    - `make -C llc`
- Build and run all C tests:
    - `make -C llc test`
- Run a single C test (substring filter):
    - `make -C llc test T=<name_substring>`
- Clean C artifacts:
    - `make -C llc clean`

### Lint/format

Rust:
- Format:
    - `cargo fmt --all`
- Lint (strict recommended):
    - `cargo clippy --workspace --all-targets -- -D warnings`

C:
- Formatting config is `llc/.clang-format`.
- If available, format changed C files before finalizing:
    - `clang-format -i llc/src/*.c llc/src/*.h llc/tests/*.c llc/include/*.h`

## Code Style and Conventions

### General

- Keep changes minimal and local.
- Preserve Rust/C semantic parity for allocator behavior.
- Avoid broad refactors unless explicitly requested.
- Prefer targeted tests first, broader suites second.

### Rust (`core/`, `eval/`)

- Respect standard Rust conventions.
- `core` is `#![no_std]`; do not introduce unguarded `std` usage.
- Toolchain is pinned by `rust-toolchain.toml`.
- Keep imports minimal and grouped (`core/std` first, crate-local next).
- Preserve newtype patterns (`FrameId`, `TreeId`, `HugeId`, `Class`).
- Avoid `unwrap()`/`expect()` outside tests or proven invariants.
- Concurrency/invariants:
    - Preserve local atomic retry loops and the CXL/host locking order.
    - Preserve clone ownership, lower-state/counter transfer, and cache flush ordering.
    - Preserve metadata-size/alignment checks.

### C (`llc/`)

- Build uses strict warnings in `llc/Makefile`; keep code warning-clean.
- Follow `llc/.clang-format` style (tabs, 80 cols, no include sorting).
- Naming:
    - Public API: `llfree_*`
    - Internal names: `snake_case`, typedefs with `_t`
    - Constants/macros: `UPPER_SNAKE_CASE`
- Optional/sentinel patterns:
    - Use `ll_optional_t` and sentinels such as `LLFREE_CLASS_NONE`.
    - Do not add extra presence booleans when optional/sentinel idioms exist.
- Error handling:
    - Return `llfree_result_t` at API boundaries.
    - Use `llfree_ok(...)` / `llfree_err(...)` helpers.
    - Keep existing error mapping conventions (e.g., no match -> `LLFREE_ERR_MEMORY`).
- Concurrency:
    - Preserve atomic compare-exchange loops and thread-safe state transitions.

### Testing conventions

- Rust:
    - Prefer deterministic, focused tests for modified behavior.
- C:
    - Register tests via `declare_test(...)`.
    - Use `check`, `check_m`, `check_equal` macros from `llc/tests/test.h`.
    - Run single tests via `make -C llc test T=<pattern>`.
- For class/tree changes:
    - Validate both operation results and stats (`llfree_tree_stats`, `trees_stats_at`).
