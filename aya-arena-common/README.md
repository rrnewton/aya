# aya-arena-common

Shared `#[repr(C)]` data structures for BPF arena memory, usable from both
eBPF programs and userspace.

## Overview

BPF arena maps (`BPF_MAP_TYPE_ARENA`) provide a shared memory region between
BPF programs and userspace. This crate provides three data structures designed
to live in that shared memory:

| Data Structure | Type | Best For |
|---------------|------|----------|
| **Linked List** | `ArenaNodeHeader` + polymorphic nodes | Heterogeneous collections, event logs |
| **Hash Map** | `ArenaHashMap` + `ArenaHashEntry` | Fast key→value lookup (task→cell, cgid→stats) |
| **B-Tree** | `ArenaBTreeMap` + `BTreeNode` | Ordered data (vtime queues, priority tracking) |

All types are `#[repr(C)]`, `no_std`, `Copy + Clone`, and use offset-based
`ArenaPtr<T>` pointers that work across the BPF/userspace address space boundary.

## Design Philosophy

1. **BPF verifier compatible**: All loops are bounded. No recursion. No dynamic
   allocation beyond the bump allocator. Maximum probe/depth limits are compile-time
   constants.

2. **Cross-boundary**: Same memory layout on both sides. `ArenaPtr<T>` stores
   byte offsets from the arena base — call `resolve(base)` to get a raw pointer.

3. **`no_std`**: Zero dependencies. Works in `#![no_std]` BPF programs and
   standard userspace alike.

4. **Unsafe but auditable**: All operations take raw pointers (necessary for BPF).
   Safety contracts are documented. The `unsafe` boundary is at the function level,
   not hidden inside abstractions.

## Data Structures

### Linked List

Tagged/polymorphic singly-linked list. Each node starts with an `ArenaNodeHeader`
containing a type tag, size, and next pointer.

```rust
use aya_arena_common::*;

// Define node types
let counter = CounterNode::new(42);
assert_eq!(counter.header.tag, TAG_COUNTER);

let label = LabelNode::new(b"hello");
assert_eq!(label.header.tag, TAG_LABEL);

// Traverse: check tag, cast to concrete type
// See test/arena-poc for full traversal example
```

Built-in node types: `CounterNode` (u64 value), `LabelNode` (32-byte string).
Add your own by placing `ArenaNodeHeader` as the first field.

### Hash Map

Open-addressing hash map with linear probing. Fixed capacity (power of 2).

```rust
use aya_arena_common::*;

let capacity = 64u32;
let entry_size = core::mem::size_of::<ArenaHashEntry>();
let header_size = core::mem::size_of::<ArenaHashMap>();
let mut buf = vec![0u8; header_size + entry_size * capacity as usize];
let base = buf.as_mut_ptr();
let map = base.cast::<ArenaHashMap>();
let entries = unsafe { base.add(header_size) }.cast::<ArenaHashEntry>();

unsafe {
    arena_hash_init(map, entries, capacity, base);
    arena_hash_insert(map, 1001, 0, base);  // task 1001 → cell 0
    arena_hash_insert(map, 1002, 1, base);  // task 1002 → cell 1

    let val = arena_hash_get(map, 1001, base);
    assert_eq!(*val, 0);

    arena_hash_delete(map, 1002, base);

    arena_hash_for_each(map, base, |key, value| {
        println!("task {key} → cell {value}");
    });
}
```

### B-Tree

Ordered map (ORDER=8) with top-down proactive splitting and lazy deletion.

```rust
use aya_arena_common::*;

let mut arena = vec![0u8; 64 * 1024];
let base = arena.as_mut_ptr();
let map = base.cast::<ArenaBTreeMap>();
unsafe { arena_btree_init(map) };

// Set up bump allocator after the header
let bump = unsafe { base.add(24).cast::<ArenaBumpState>() };
unsafe { *bump = ArenaBumpState::new(64 * 1024 - 48); (*bump).watermark = 48; }

unsafe {
    arena_btree_insert(map, bump, 30, 300, base);
    arena_btree_insert(map, bump, 10, 100, base);
    arena_btree_insert(map, bump, 20, 200, base);

    let val = arena_btree_get(map, 20, base);
    assert_eq!(*val, 200);

    // Iterate in sorted order: 10, 20, 30
    arena_btree_for_each(map, base, |key, value| {
        println!("{key} → {value}");
    });
}
```

## Performance

Benchmarked on AMD EPYC 9D64 (176 threads), release build:

| Operation | Linked List | Hash Map (50% load) | B-Tree (1000) |
|-----------|------------:|--------------------:|--------------:|
| Insert | 1.1 ns/op | 10 ns/op | 46 ns/op |
| Lookup | — | 9 ns/op | 33 ns/op |
| Delete | — | 9 ns/op | ~30 ns/op |
| Traverse | 1.7 ns/op | — | 6 ns/op |
| Bump alloc | 1.7 ns/op | — | — |

**vs std library** (same operations, same data):

| | Hash Map | B-Tree |
|-----------|------------------:|------------------:|
| Insert | 1.3-5x faster | 1.1x faster |
| Lookup hit | 1.2-1.8x faster | 1.2-1.5x faster |
| Lookup miss | 1.2x faster to 3.4x slower (load-dependent) | — |
| Ordered iter | N/A | 3.7x slower |

## Testing

```sh
cargo test --lib -p aya-arena-common
```

46 tests covering all three data structures:
- Layout verification for all types
- CRUD operations (insert, get, update, delete)
- Edge cases (capacity=1, key=0, key=u64::MAX, non-power-of-2 rejection)
- Collision handling, tombstone chains, probe chain integrity
- B-tree node splitting, internal node deletion, ordered iteration
- Stress tests (100-1000 entries with insert/delete/verify cycles)

## Benchmarks

```sh
cargo run --release -p arena-bench
```

See `ARENA_BENCHMARK_REPORT.md` for detailed analysis.

## Crate Structure

```
aya-arena-common/src/lib.rs
├── ArenaPtr<T>          — offset-based pointer (8 bytes)
├── ArenaNodeHeader      — tagged node header (16 bytes)
├── ArenaBumpState       — bump allocator state (16 bytes)
├── Linked List types
│   ├── CounterNode      — u64 counter (24 bytes)
│   ├── LabelNode        — 32-byte string (56 bytes)
│   └── ArenaListHead    — list head + count (16 bytes)
├── Hash Map
│   ├── ArenaHashEntry   — key/value/state slot (24 bytes)
│   ├── ArenaHashMap     — header (24 bytes)
│   └── Functions: init, insert, get, delete, for_each
└── B-Tree
    ├── BTreeNode        — ORDER=8 node (184 bytes)
    ├── ArenaBTreeMap    — header (24 bytes)
    └── Functions: init, insert, get, delete, for_each
```
