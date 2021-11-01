# Non-Volatile Memory Allocator

This is a prototype of a page allocator for a non-volatile memory.
It is designed for a hybrid system that has volatile and non-volatile memory in the same address space.

The two main design goals are multicore scalability and crash consistency.
