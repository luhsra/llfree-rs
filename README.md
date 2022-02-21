# Non-Volatile Memory Allocator

This repository contains prototypes of a page allocator for non-volatile memory.
They are designed for a hybrid system with volatile and non-volatile memory in the same address space.

The two main design goals are multicore scalability and crash consistency.

> The corresponding master's thesis can be found here: [MA_lar.wrenger](https://scm.sra.uni-hannover.de/theses/2021/MA_lar.wrenger).

## Usage

To compile and test the allocator, you have to [install rust](https://www.rust-lang.org/learn/get-started).

Currently, the nightly version `2022-02-08` (or newer) is used to have access to inline assembly.

## Project structure

The [src/alloc](src/alloc/) directory contains the different allocator variants.
The general interface is defined in [src/alloc.rs](src/alloc.rs), together with various unit tests and stress tests.

The persistent lower allocator can be found in [src/lower_alloc.rs](src/lower_alloc.rs).
It is responsible for managing the layer one and layer two-page tables that are persisted on the non-volatile memory.
Most of the upper allocators in [src/alloc](src/alloc/) use the lower allocator and focus only on managing the higher level 1G subtrees using volatile data structures.

The lower allocator is heavily tested for race conditions using synchronization points (`stop`) to control the execution order of parallel threads.
They are similar to barriers where, on every synchronization point, the next running CPU is chosen either by a previously defined order or in a pseudo-randomized manner.
This mechanism is implemented in [src/stop.rs](src/stop.rs).

The paging data structures are defined in [src/entry.rs](src/entry.rs) and [src/table.rs](src/table.rs).

## Benchmarks

The benchmarks can be found in [examples/bench.rs](examples/bench.rs) and the benchmark evaluation and visualization in [bench](bench/).

These benchmarks can be executed with:

```bash
cargo perf bench -- -t1 -t2 -t4 -m24 -b bulk -o bench/bulk.csv
```

This runs the `bulk` benchmark for 1, 2, and 4 threads on 24G DRAM and stores the result in `bench/out/bulk.csv`.

To execute the benchmark on NVM, use the `--dax` flag to specify a DAX file to be mmaped.

> The debug output can be suppressed with setting `RUST_LOG=error`.

## Profiling

@see https://perf.wiki.kernel.org/index.php/Tutorial

General statistics:

```
perf stat -e <events> target/release/deps/nvalloc_rs-02d02675d86de11a --nocapture --test-threads 1 parallel_free
```

> Additional details with `-d -d -d`...
>
> Also `hotspot` is a great analysis tool for these statistics

Recording events:

```
perf record -g -F 999 target/release/deps/nvalloc_rs-02d02675d86de11a --nocapture --test-threads 1 parallel_free
```

After conversion `perf script -F +pid > test.perf`, this can be opened in firefox: https://profiler.firefox.com/

@see https://profiler.firefox.com/docs/#/./guide-perf-profiling
