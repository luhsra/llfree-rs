---
name: lang-zig
description: Zig development standards — memory management, style, testing, and project structure. Load when working on Zig projects.
---

# Zig Standards

## Code Style

Follow the [Zig Style Guide](https://ziglang.org/documentation/master/#Style-Guide).

**Tooling (mandatory)**:
- `zig fmt` — built-in formatter (always run before commit)
- `zig build` — build system (no external build tools needed)

**Naming**:
- `camelCase` for functions and variables
- `PascalCase` for types
- `SCREAMING_SNAKE_CASE` for compile-time constants
- Prefix unused variables with `_`

---

## Memory Management

Zig uses explicit allocators — there is no hidden allocation.

```zig
// Good: accept allocator as parameter
pub fn createList(allocator: std.mem.Allocator) !std.ArrayList(u8) {
    return std.ArrayList(u8).init(allocator);
}

// Bad: using a global allocator or hiding allocation
var global_list = std.ArrayList(u8).init(std.heap.page_allocator);
```

- **Always pass allocators explicitly** — never use global allocators in library code
- Use `defer` and `errdefer` for cleanup
- Prefer `std.heap.GeneralPurposeAllocator` for debug builds (detects leaks and use-after-free)
- Use `ArenaAllocator` for batch allocations with a shared lifetime
- Document allocator ownership and lifetime expectations in public APIs

```zig
const gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer {
    const status = gpa.deinit();
    if (status == .leak) @panic("memory leak detected");
}
const allocator = gpa.allocator();
```

---

## Error Handling

Zig uses error unions — propagate errors explicitly.

```zig
// Good: return error with context
fn readConfig(path: []const u8) !Config {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.log.err("failed to open config {s}: {}", .{ path, err });
        return err;
    };
    defer file.close();
    // ... parse
}
```

- Return errors, never `@panic()` in library code (reserve for truly unrecoverable bugs)
- Use error sets to define expected failure modes
- Use `catch` for handling, `try` for propagation
- Use `errdefer` for cleanup on error paths
- Log or wrap errors with context before returning

---

## Comptime

Zig's `comptime` is a core feature — use it deliberately.

```zig
fn Vector(comptime T: type, comptime len: usize) type {
    return struct {
        data: [len]T,

        pub fn dot(self: @This(), other: @This()) T {
            var sum: T = 0;
            for (self.data, other.data) |a, b| {
                sum += a * b;
            }
            return sum;
        }
    };
}
```

- Document comptime parameters, invariants, and constraints
- Use `@compileError` for clear error messages when comptime constraints are violated
- Prefer comptime over runtime when the information is available at compile time
- Keep comptime logic simple and auditable

---

## Testing

**Framework**: built-in `zig test`.

```zig
const std = @import("std");
const testing = std.testing;

test "addition works" {
    const result = add(2, 3);
    try testing.expectEqual(@as(i32, 5), result);
}

test "handles overflow" {
    const result = addChecked(std.math.maxInt(i32), 1);
    try testing.expectError(error.Overflow, result);
}
```

- Use `test` blocks in the same file as the code they test
- Test error paths, not just happy paths
- Use `testing.allocator` for tests — it detects leaks automatically
- Use `testing.expectEqual`, `testing.expect`, `testing.expectError`
- Name tests descriptively: `test "parseConfig returns error for missing file"`
- Run with `zig build test` or `zig test src/file.zig`

---

## Build System

Zig uses `build.zig` — no external build tools required.

```zig
// build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    const tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
```

---

## Project Structure

```
├── src/
│   ├── main.zig          # Entry point
│   ├── lib.zig           # Library root (public API)
│   └── module/
│       ├── module.zig     # Module implementation
│       └── tests.zig      # Module tests (or inline)
├── build.zig
├── build.zig.zon          # Package manifest
└── README.md
```

- Keep `main.zig` minimal — parse args, wire dependencies, call into library
- Use `@import` for module organization
- Prefer inline tests (in the same file) unless test files get large

---

## Cross-Compilation

Zig excels at cross-compilation — leverage it.

```bash
zig build -Dtarget=x86_64-linux-gnu
zig build -Dtarget=aarch64-macos
zig build -Dtarget=x86_64-windows-gnu
```

- Use `standardTargetOptions` in build.zig to expose target selection
- Test on multiple targets in CI
- Be aware of platform-specific behavior (file paths, endianness, etc.)

---

## C Interop

Zig has first-class C interop — use it when wrapping C libraries.

```zig
const c = @cImport({
    @cInclude("sqlite3.h");
});
```

- Use `@cImport` for C headers
- Wrap C APIs in Zig-idiomatic interfaces (error unions, slices instead of pointer+length)
- Document any C library dependencies and link requirements in build.zig

---

## Security

- Load secrets from environment or secret manager
- Never commit secrets to VCS
- Validate and sanitize all inputs
- Use `std.crypto` for cryptographic operations — never roll your own
- Be explicit about buffer sizes to prevent overflows

---

## Observability

- Use `std.log` for structured logging with scoped loggers
- Include context (request IDs, operation names) in log messages
- Log levels: info (normal), warn (unexpected but tolerable), err (actionable)

```zig
const log = std.log.scoped(.my_module);
log.info("request processed", .{});
log.err("failed to connect: {}", .{err});
```

---

## Deliverables

- `README.md` with build/test instructions
- `build.zig` and `build.zig.zon` for build configuration
- Unit tests for public APIs
- Proper error handling throughout
- No memory leaks (verified with `testing.allocator` or GPA in debug)
