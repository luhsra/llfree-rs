# AGENTS.md

Operational guide for coding agents working in `llfree-rs`.

Repo scope:
- Rust allocator in `core/`
- C allocator in `llc/`
- Evaluation/integration harness in `eval/`

No Cursor rules were found (`.cursor/rules/`, `.cursorrules`).
No Copilot instructions file was found (`.github/copilot-instructions.md`).

## Architecture / Design (Short)

LLFree is a two-level allocator optimized for scalability and fragmentation control.

- **Lower level (`core/src/lower.rs`)**
  - Performs actual allocation/free at page and huge-page granularity.
  - Uses bitfields and huge-entry counters/flags.
  - Groups huge metadata into tree-sized chunks.
  - Supports `Init::{FreeAll, AllocAll, Recover, None}` modes.

- **Upper level (`core/src/llfree.rs`)**
  - Manages tree placement/locality and delegates concrete alloc/free to `Lower`.
  - Uses per-tier local reservations (`Locals`) to reduce contention.
  - Uses global tree metadata (`Trees`) updated via atomics/CAS.
  - Allocation fallback order: match -> demote -> tier downgrade -> steal.

- **Tiering/policy (`core/src/lib.rs`)**
  - `Tiering` defines tiers + local-slot counts + policy function.
  - `Policy` values: `Match(priority)`, `Steal`, `Demote`, `Invalid`.
  - Built-in examples: `Tiering::simple`, `Tiering::movable`.

- **Key invariants**
  - Tree counters stay consistent with lower-level state.
  - Reserved trees are not mutated by unsupported paths.
  - Tier stats (`TreeStats`) reflect tree transitions.
  - Keep lock-free retry/CAS semantics intact.

## 1) Build, Lint, Test Commands

### Nix Dev Shell & Direnv

- Direnv should auto-activate the Nix shell when you `cd` into the repo.
- Run `direnv reload` if you update `flake.nix` or the shell environment.

### Rust workspace

- Build all crates (debug):
  - `cargo build`
- Build release:
  - `cargo build -r`
- Build core crate only:
  - `cargo build -p llfree`
- Build eval crate only:
  - `cargo build -p llfree-eval`

### Rust tests

- Run all core tests:
  - `cargo test -p llfree`
- Run one Rust test (substring):
  - `cargo test -p llfree <test_name_substring>`
  - Example: `cargo test -p llfree change_tree`
- Run one Rust test with logs:
  - `cargo test -p llfree <test_name_substring> -- --nocapture`

### Eval integration tests

- Run integration suite:
  - `cargo test -p llfree-eval --test integration`
- Run one integration test:
  - `cargo test -p llfree-eval --test integration <test_name_substring>`

### Eval with C backend

- Init/update C submodule (if needed):
  - `git submodule update --init --checkout llc`
- Run eval tests against C impl:
  - `cargo test -p llfree-eval -F llc --test integration`
- Run one eval test against C impl:
  - `cargo test -p llfree-eval -F llc --test integration <test_name_substring>`

### C implementation (`llc/`)

- Build static C library:
  - `make -C llc`
- Build and run all C tests:
  - `make -C llc test`
- Run a single C test (substring filter):
  - `make -C llc test T=<name_substring>`
  - Example: `make -C llc test T=zeroed`
  - Example: `make -C llc test T=change_tree`
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

## 2) Code Style and Conventions

### General

- Keep changes minimal and local.
- Preserve Rust/C semantic parity for allocator behavior.
- Avoid broad refactors unless explicitly requested.
- Prefer targeted tests first, broader suites second.

### Rust (`core/`, `eval/`)

- `core` is `#![no_std]`; do not introduce unguarded `std` usage.
- Toolchain is pinned by `rust-toolchain.toml`.
- Imports:
  - Keep imports minimal and grouped (`core/std` first, crate-local next).
- Naming:
  - Types/traits/enums: `UpperCamelCase`
  - Functions/modules/fields: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- Types:
  - Preserve newtype patterns (`FrameId`, `TreeId`, `HugeId`, `Tier`).
- Error handling:
  - Use `Result<T, Error>` and explicit variant handling.
  - Prefer `?` and structured fallbacks over ad-hoc branching.
  - Avoid `unwrap()`/`expect()` outside tests or proven invariants.
- Concurrency/invariants:
  - Keep atomic retry loops and lock-free behavior unchanged.
  - Preserve metadata-size/alignment checks.

### C (`llc/`)

- Build uses strict warnings in `llc/Makefile`; keep code warning-clean.
- Follow `llc/.clang-format` style (tabs, 80 cols, no include sorting).
- Naming:
  - Public API: `llfree_*`
  - Internal names: `snake_case`, typedefs with `_t`
  - Constants/macros: `UPPER_SNAKE_CASE`
- Optional/sentinel patterns:
  - Use `ll_optional_t` and sentinels such as `LLFREE_TIER_NONE`.
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
- For tier/tree changes:
  - Validate both operation results and stats (`llfree_tree_stats`, `trees_stats_at`).

## Agent workflow (recommended)

1. Identify which layer is affected (lower, trees, locals, upper API, eval).
2. Implement minimally and preserve invariants.
3. Run the narrowest relevant single test first.
4. Run broader crate/suite tests if scope warrants.
5. Report commands executed and any known unrelated failures.
