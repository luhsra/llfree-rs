# LLFree: Lock- and Log-free Allocator

This repository contains prototypes of page allocators.
They are designed for multicore, hybrid systems with volatile and non-volatile memory.

The two main design goals are multicore scalability and crash consistency.

**Related Projects**
- Benchmarks: https://scm.sra.uni-hannover.de/research/nvalloc-bench
- Modified Linux: https://scm.sra.uni-hannover.de/research/nvalloc-linux
- Benchmark Module: https://scm.sra.uni-hannover.de/research/linux-alloc-bench

## Usage

To compile and test the allocator, you have to [install rust](https://www.rust-lang.org/learn/get-started).

The `nightly` version `1.62.0` (or newer) is required for inline assembly and custom compiler toolchains.

```sh
# Release build
cargo build -r

# Running unit-tests
cargo test -- --test-threads 1

# Running a benchmark (see down below for more infos on benchmarking)
cargo perf <benchmark> -- <args>
# For example print help for the `bench` benchmark
cargo perf bench -- -h
```

> Note: This project uses certian UNIX features directly (for memory mapping and thread pinning) and doesn't work on Windows without modifications.

## Project structure

The [core](core/) directory contains the main `nvalloc` crate and all the allocators.
The persistent lower allocators can be found in [lower](core/src/lower/).
Their interface is defined in [lower.rs](core/src/lower.rs).
These lower allocators manage the level-one and level-two page tables that are optionally persistent on non-volatile memory.
They allocate pages of 4K up to 4M in these subtrees.

- [`Cache`](core/src/lower/cache.rs): This allocator has a 512-bit-large bit field at the lowest level. It stores which 4K pages are allocated. The second level consists of tables with N 16-bit entries, one for each bit field. These entries contain a counter of free pages in the related bit field and a flag if the whole subtree is allocated as a 2M huge page.
The number of entries N in the second-level tables can be defined at compile-time.

The [upper](core/src/upper/) directory contains the different upper allocator variants.
The general interface is defined in [upper.rs](core/src/upper.rs), along with various unit and stress tests.
Most of the upper allocators depend on the lower allocator for the actual allocations and only manage the higher-level subtrees.
The upper allocators are completely volatile and have to be rebuilt on boot.
The different implementations are listed down below:

- [`Array`](core/src/upper/array.rs): It consists of a single array of tree entries, which is linearly searched for the appropriate subtrees.
- [`ListLocal`](core/src/upper/list_local.rs), [`ListLocked`](core/src/upper/list_locked.rs), and [`ListCAS`](core/src/upper/list_cas.rs): These reference implementations are used to evaluate the performance of allocators.

The allocator's data structures are defined in [core/src/entry.rs](core/src/entry.rs) and [core/src/table.rs](core/src/table.rs).

## Benchmarks

The benchmarks can be found in [bench/src/bin](bench/src/bin) and the benchmark evaluation and visualization in the [nvalloc-bench](https://scm.sra.uni-hannover.de/research/nvalloc-bench) repository.

These benchmarks can be executed with:

```bash
cargo perf bench -- bulk -x1 -x2 -x4 -t4 -m24 -o results/bench.csv ArrayAtomicA128
```

This runs the `bulk` benchmark for 1, 2, and 4 threads (`-t4` max 4 threads) on 24G DRAM and stores the result in `results/bench.csv`.

To execute the benchmark on NVM, use the `--dax` flag to specify a DAX file to be mapped.

> For more info on the cli arguments run `cargo perf bench -- -h`.
>
> The debug output can be suppressed with setting `RUST_LOG=error`.


## Integrating into the Linux Kernel

- See: https://scm.sra.uni-hannover.de/research/nvalloc-linux


## Profiling

- See: https://perf.wiki.kernel.org/index.php/Tutorial

General statistics:

```
perf stat -e <events> target/release/bench <args>
```

> Additional details with `-d -d -d`...
>
> Also `hotspot` is a great analysis tool for these statistics

Recording events:

```
perf record -g -F 999 target/release/bench <args>
```

After conversion `perf script -F +pid > test.perf`, this can be opened in firefox: https://profiler.firefox.com/

- See: https://profiler.firefox.com/docs/#/./guide-perf-profiling
