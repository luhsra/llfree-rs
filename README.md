# Non-Volatile Memory Allocator

This repository contains prototypes of a page allocator for non-volatile memory.
They are designed for a hybrid system with volatile and non-volatile memory in the same address space.

The two main design goals are multicore scalability and crash consistency.

> The corresponding master's thesis can be found here: [MA_lar.wrenger](https://scm.sra.uni-hannover.de/theses/2021/MA_lar.wrenger).

## Usage

To compile and test the allocator, you have to [install rust](https://www.rust-lang.org/learn/get-started).

The `nightly` version `1.59.0` (or newer) is required to have access to inline assembly and custom compiler toolchains.

## Project structure

The [core](core/) directory contains the main `nvalloc` crate and all the allocators.
The [upper](core/src/upper/) directory contains the different upper allocator variants.
The general interface is defined in [upper.rs](core/src/upper.rs), together with various unit tests and stress tests.

The persistent lower allocators can be found in [lower](core/src/lower/).
Again their interface is defined in [lower.rs](core/src/lower.rs).
These lower allocators are responsible for managing the level one and level two-page tables that are persisted on the non-volatile memory.
Most of the upper allocators in [upper](core/src/upper/) use the lower allocator and focus only on managing the higher level subtrees using volatile data structures.

The lower allocator is heavily tested for race conditions using synchronization points (`stop`) to control the execution order of parallel threads.
They are similar to barriers where, on every synchronization point, the next running CPU is chosen either by a previously defined order or in a pseudo-randomized manner.
This mechanism is implemented in [stop.rs](core/src/stop.rs).

The paging data structures are defined in [core/src/entry.rs](core/src/entry.rs) and [core/src/table.rs](core/src/table.rs).

## Benchmarks

The benchmarks can be found in [bench/src/bin/bench.rs](bench/src/bin/bench.rs) and the benchmark evaluation and visualization in [results](results/).

These benchmarks can be executed with:

```bash
cargo perf bench -- rand -s10 -t4 -i4 -m24 -x1 -x2 -x4 -o results/bench.csv ArrayAtomicC128
```

This runs the `bulk` benchmark for 1, 2, and 4 threads on 24G DRAM and stores the result in `results/bench.csv`.

To execute the benchmark on NVM, use the `--dax` flag to specify a DAX file to be mmaped.

> The debug output can be suppressed with setting `RUST_LOG=error`.

## Integrating into the Linux Kernel

The [module](module/) directory contains wrapper code and Makefiles for integrating the allocators into the Linux kernel.

First create a symlink in the linux tree to the `nvalloc-rs/module` directory:

```sh
ln -s </path/to/nvalloc-rs>/module </path/to/linux>/lib/nvalloc
```

Then add the `nvalloc` directory to the `</path/to/linux>/lib/Makefile`:

```diff
obj-y += nvalloc/
```

And finally compile the module and the linux kernel:

```sh
# in ./module
make LLVM=1 modules
# in </path/to/linux>
make LLVM=1
```

> Problems:
> - Please increase KSYM_NAME_LEN both in kernel and kallsyms.c
> - Non-allocatable sections: .llvmbc, .llvmcmd
> - Unknown sections: .text.__rust_probestack, .eh_frame

### Support for printk

The exported function `rust_fmt_argument` is a custom formatter for the rust print formatting.
It has to be called in `lib/vsprintf.c:pointer` for the `%pA` format argument:

```c
char *rust_fmt_argument(char *buf, char *end, void *ptr);

char *pointer(const char *fmt, char *buf, char *end, void *ptr,
	      struct printf_spec spec)
{
    switch(*fmt) {
// ...
    case 'A':
        return rust_fmt_argument(buf, end, ptr);
// ...
    }
}
```

Logging is automatically initialized with the allocator.

## Profiling

@see https://perf.wiki.kernel.org/index.php/Tutorial

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

@see https://profiler.firefox.com/docs/#/./guide-perf-profiling
