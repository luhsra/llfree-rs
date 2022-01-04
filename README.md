# Non-Volatile Memory Allocator

This is a prototype of a page allocator for a non-volatile memory.
It is designed for a hybrid system that has volatile and non-volatile memory in the same address space.

The two main design goals are multicore scalability and crash consistency.

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
