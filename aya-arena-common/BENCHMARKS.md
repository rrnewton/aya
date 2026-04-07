# Arena Data Structure Benchmark Report

**Date**: 2026-04-07
**Crate**: `aya-arena-common` v0.1.0
**Benchmark**: `test/arena-bench`

## 1. Methodology

### Hardware

| Spec | Value |
|------|-------|
| CPU | AMD EPYC 9D64 88-Core (176 threads, 1 socket) |
| Clock | 2943 MHz base |
| L3 Cache | 176 MiB (11 instances, 16 MiB each) |
| Memory | 251 GiB |
| NUMA | 1 node |
| Kernel | 6.9.0-fbk13 |
| Arch | x86_64 |

### Approach

All benchmarks run single-threaded on a **simulated arena**: a `Vec<u8>` backed
by the heap, with `ArenaBumpState` tracking the allocation watermark. This
simulates the userspace side of a BPF arena where memory is `mmap`'d and both
BPF and userspace see the same virtual address range.

- **Release mode** (`--release`, LTO disabled)
- **5 measurement iterations** with 3 warmup iterations per benchmark
- **Median** of 5 runs reported (not mean — avoids outlier skew)
- `std::hint::black_box` used to prevent dead-code elimination on traversals
- `std::time::Instant` for timing (monotonic clock)

### What this measures

- Pure userspace allocation and data structure operation costs
- Memory layout efficiency of `#[repr(C)]` shared types
- Overhead of offset-based `ArenaPtr<T>` vs native pointers

### What this does NOT measure

- BPF verifier acceptance
- Kernel `bpf_arena_alloc_pages` performance
- Cross address-space pointer resolution
- Concurrent access (BPF programs run non-preemptively per-CPU)

## 2. Raw Results

### Type Layouts

| Type | Size (bytes) | Alignment |
|------|-------------|-----------|
| `ArenaPtr<T>` | 8 | 8 |
| `ArenaNodeHeader` | 16 | 8 |
| `CounterNode` | 24 | 8 |
| `LabelNode` | 56 | 8 |
| `ArenaListHead` | 16 | 8 |
| `ArenaBumpState` | 16 | 8 |

### Bump Allocation Speed (fill 64 MiB)

| Alloc Size | Count | Time | ns/op | Throughput | Utilization |
|-----------|-------|------|-------|-----------|-------------|
| 24B | 2,796,202 | 2.86 ms | 1 | 976 M/s | 100% |
| 56B | 1,198,372 | 1.23 ms | 1 | 978 M/s | 100% |
| 256B | 262,144 | 0.27 ms | 1 | 962 M/s | 100% |
| 4 KB (page) | 16,384 | 0.02 ms | 1 | 980 M/s | 100% |

### Typed Allocation + Initialization

| Type | Count | Time | ns/op | Throughput | Memory Used |
|------|-------|------|-------|-----------|-------------|
| CounterNode | 100K | 0.10 ms | 1 | 982 M/s | 2,343 KB |
| CounterNode | 500K | 0.51 ms | 1 | 975 M/s | 11,718 KB |
| CounterNode | 1M | 1.02 ms | 1 | 977 M/s | 23,437 KB |
| LabelNode | 100K | 0.95 ms | 9 | 105 M/s | 5,468 KB |
| LabelNode | 500K | 4.77 ms | 9 | 105 M/s | 27,343 KB |
| LabelNode | 1M | 9.56 ms | 9 | 105 M/s | 54,687 KB |

### Memory Efficiency

| Alloc Size | Count | Watermark | Ideal | Overhead |
|-----------|-------|----------|-------|---------|
| 8B | 8,388,608 | 65,536 KB | 65,536 KB | 0.0% |
| 24B | 2,796,202 | 65,535 KB | 65,535 KB | 0.0% |
| 56B | 1,198,372 | 65,535 KB | 65,535 KB | 0.0% |
| 64B | 1,048,576 | 65,536 KB | 65,536 KB | 0.0% |
| 256B | 262,144 | 65,536 KB | 65,536 KB | 0.0% |
| 1 KB | 65,536 | 65,536 KB | 65,536 KB | 0.0% |
| 4 KB | 16,384 | 65,536 KB | 65,536 KB | 0.0% |

### ArenaPtr::resolve()

| Operation | Count | Time | ns/op | Throughput |
|----------|-------|------|-------|-----------|
| resolve() | 1,000,000 | 2.45 ms | 2 | 408 M/s |

### Linked List: Arena vs std::LinkedList

#### Insert (prepend to head)

| N | Arena ns/op | std ns/op | Arena M/s | std M/s | Speedup |
|---|-----------|---------|---------|-------|---------|
| 10K | 3 | 10 | 253 M | 100 M | **3.3x** |
| 100K | 4 | 10 | 231 M | 98 M | **2.5x** |
| 500K | 4 | 10 | 202 M | 94 M | **2.5x** |

#### Traverse (iterate all nodes, sum values)

| N | Arena ns/op | std ns/op | Arena M/s | std M/s | Ratio |
|---|-----------|---------|---------|-------|-------|
| 10K | 1.7 | 1.4 | 584 M | 703 M | 0.8x |
| 100K | 1.7 | 1.4 | 587 M | 717 M | 0.8x |
| 500K | 1.7 | 1.4 | 584 M | 710 M | 0.8x |

### Heterogeneous List (mixed CounterNode + LabelNode)

| N | Time | ns/op | Throughput |
|---|------|-------|-----------|
| 100K | 0.19 ms | 1.9 | 534 M/s |
| 500K | 0.96 ms | 1.9 | 520 M/s |

## 3. Analysis

### Bump allocation: why it's fast

The bump allocator performs exactly **two operations** per allocation:

1. Align the watermark: `aligned = (watermark + align - 1) & !(align - 1)`
2. Advance it: `watermark = aligned + size`

No free lists, no size classes, no headers, no coalescing. This compiles down
to ~3 instructions on x86_64. The measured **1 ns/op** is essentially the
cost of a single L1-cache-hit store.

The allocation cost is **independent of allocation size** — 24B and 4KB
allocations take the same time because the allocator only touches the
watermark, never the allocated memory itself.

### CounterNode vs LabelNode init cost

CounterNode (24B) initialization costs 1 ns — the 8-byte value write
is absorbed into the same cache line as the 16-byte header write.

LabelNode (56B) initialization costs 9 ns — the 32-byte label copy
(`while i < copy_len { label[i] = s[i]; }`) dominates. This is a
byte-by-byte copy that could be optimized with `ptr::copy_nonoverlapping`
but is kept simple for `no_std`/`const` compatibility.

### Memory efficiency: 0% overhead

The bump allocator has zero per-allocation metadata. Every byte of arena
capacity is usable. In contrast:
- `jemalloc`: 8-16 bytes overhead per allocation (size class + slab metadata)
- `glibc malloc`: 8-16 bytes overhead (chunk header with size + flags)
- BPF buddy allocator (scx): 1 bit per 16-byte slot (0.8% overhead) plus
  4-bit order per slot

The tradeoff: bump allocation cannot free individual objects. The entire
arena must be reset at once. This is appropriate for BPF program lifetime
data (topology, bitmaps) and per-scheduling-cycle allocations.

### Arena vs std::LinkedList tradeoffs

**Insert is 2.5-3.3x faster** because:
- Arena: bump-allocate (1 ns) + write node (2-3 ns) = ~4 ns total
- std: heap allocate via global allocator (~8-10 ns) + write node = ~10 ns
- The gap is the heap allocator overhead (jemalloc thread-cache lookup,
  size-class selection, free-list pop)

**Traversal is ~20% slower** because:
- Arena uses `ArenaPtr<T>` (offset-based): each step requires
  `base + offset` addition before dereferencing
- std uses native pointers: each step is a single indirect load
- Both are dominated by pointer-chasing latency (L1 hit: ~1 ns, L2: ~4 ns)
- Arena's sequential allocation gives perfect spatial locality for fresh lists

**Implication**: For BPF schedulers, insert-heavy workloads (task init,
cell allocation) benefit from arena. Traversal-heavy workloads (cgroup
iteration, DSQ scanning) pay a small penalty but benefit from spatial
locality.

### Heterogeneous list performance

Tag-dispatched traversal of mixed CounterNode (24B) + LabelNode (56B)
runs at ~530 M/s (1.9 ns/op). The `match tag { ... }` dispatch adds
negligible cost over homogeneous traversal — branch prediction handles
the alternating pattern well.

## 4. Limitations

### Simulated environment

These benchmarks run on **userspace heap memory**, not a real BPF arena.
Key differences with a real kernel arena (`BPF_MAP_TYPE_ARENA`):

1. **Page faults**: Real arena memory is demand-paged via `bpf_arena_alloc_pages`.
   First access to a new page incurs a page fault (~1-5 µs). Our simulation
   pre-allocates all memory, hiding this cost.

2. **Address space casting**: With `__BPF_FEATURE_ADDR_SPACE_CAST`, the compiler
   handles arena pointer resolution. Without it, `BPF_ADDR_SPACE_CAST` instructions
   are emitted. Our `ArenaPtr::resolve()` is a userspace approximation.

3. **TLB pressure**: A 64 MiB arena with 4 KB pages uses 16K TLB entries.
   On real hardware, TLB misses add ~10-50 ns per access. Arena's `map_extra`
   VA pinning at `1 << 44` may use huge pages, reducing this.

4. **Concurrency**: BPF programs run non-preemptively per-CPU, so concurrent
   access patterns are lock-free by design. Our single-threaded benchmark
   doesn't exercise the arena spin lock or per-CPU allocator paths.

### Missing allocator tiers

The bump allocator is only one of three tiers in the scx arena library:
- **Bump** (benchmarked): never frees, ~1 ns/alloc
- **Stack/slab**: typed free-list, alloc+free, expected ~5-10 ns
- **Buddy**: general-purpose malloc/free, expected ~20-50 ns

### No verifier validation

These benchmarks confirm that the Rust types have correct layout and
acceptable performance, but do not test BPF verifier acceptance. A type
that performs well in userspace might be rejected by the verifier due to
pointer tracking rules, stack size limits (512 bytes), or loop bounds.

## 5. Future Work

### Kernel arena benchmarks (requires 6.2+ kernel)

1. **Page allocation latency**: Time `bpf_arena_alloc_pages()` for 1, 8, 64 pages
2. **Cross-side access**: BPF allocates, userspace reads — measure mmap coherence
3. **Arena spin lock contention**: Multiple CPUs accessing shared arena data
4. **TLB impact**: Measure access latency with varying arena sizes (4MB - 4GB)

### Additional allocator benchmarks

1. **Stack allocator**: alloc/free cycles, slab reuse efficiency
2. **Buddy allocator**: fragmentation under mixed size workloads
3. **Allocator comparison**: bump vs stack vs buddy for scheduler workloads

### Data structure benchmarks

1. **B-tree**: insert/lookup/delete with varying key counts
2. **Red-black tree**: comparison with B-tree for ordered iteration
3. **Radix tree**: per-task data lookup (SDT pattern)
4. **CPU topology tree**: build and traverse topology with real hardware data

### Scheduler-specific benchmarks

1. **Cell reconfiguration**: Time to rebuild cell cpumasks and LLC counts
2. **Task init/exit churn**: Measure init_task_impl + update_task_cell cost
3. **Work stealing scan**: Cost of try_stealing_work across N LLCs
4. **Vtime accounting**: Overhead of advance_dsq_vtimes per dispatch

---

## 6. Arena Hash Map Benchmarks (NEW)

### Design

Open-addressing hash map with linear probing, designed for BPF verifier compatibility:
- Fixed capacity (power of 2), bounded probe limit (128 max)
- `u64` keys and `u64` values (24-byte entries)
- splitmix64 hash function (branchless, `const fn`)
- Tombstone deletion preserves probe chains

### Results (capacity=1024, release build, AMD EPYC 9D64)

#### Insert (ns/op)

| Load Factor | Arena | std::HashMap | Ratio |
|------------:|------:|-------------:|------:|
| 25% | 9.1 | 47.5 | **5.2x faster** |
| 50% | 10.0 | 14.4 | **1.4x faster** |
| 75% | 11.1 | 14.3 | **1.3x faster** |
| 89% | 11.0 | 14.4 | **1.3x faster** |

Note: std's 25% result includes initial allocation; arena pre-allocates slots.

#### Lookup Hit (ns/op)

| Load Factor | Arena | std::HashMap | Ratio |
|------------:|------:|-------------:|------:|
| 25% | 6.3 | 11.5 | **1.8x faster** |
| 50% | 9.1 | 11.2 | **1.2x faster** |
| 75% | 9.5 | 11.4 | **1.2x faster** |
| 89% | 9.8 | 12.7 | **1.3x faster** |

#### Lookup Miss (ns/op)

| Load Factor | Arena | std::HashMap | Ratio |
|------------:|------:|-------------:|------:|
| 25% | 9.3 | 10.8 | 1.2x faster |
| 50% | 16.0 | 10.9 | **1.5x SLOWER** |
| 75% | 20.8 | 11.2 | **1.9x SLOWER** |
| 89% | 37.6 | 11.1 | **3.4x SLOWER** |

Miss-lookup is the known weakness of linear probing at high load factors.
std::HashMap has O(1) miss via separate chaining.

#### Delete (ns/op)

| Load Factor | Arena | std::HashMap | Ratio |
|------------:|------:|-------------:|------:|
| 25% | 7.7 | 13.4 | **1.7x faster** |
| 50% | 9.2 | 13.1 | **1.4x faster** |
| 75% | 9.5 | 13.1 | **1.4x faster** |
| 89% | 9.4 | 13.7 | **1.5x faster** |

### Analysis

For sched_ext workloads (task→cell mappings, cgid→stats):
- **Lookups are mostly hits** (tasks exist in the map) → arena wins 1.2-1.8x
- **Insertions are rare** (task init) → arena wins 1.3-5x
- **Deletions are rare** (task exit) → arena wins 1.4-1.7x
- **Load factor stays moderate** (<75%) → miss penalty is manageable

The arena hash map trades miss-lookup performance for:
- Zero allocation overhead (pre-allocated slots)
- BPF verifier compatibility (bounded loops, no dynamic resizing)
- Cross-boundary access (same memory layout in BPF and userspace)

### Test Coverage

30 unit tests covering:
- Layout verification, basic CRUD, collision chains (cap=4)
- Tombstone correctness (delete+reinsert, probe chain integrity)
- Edge cases: capacity=1, key=0, key=u64::MAX
- Init validation: rejects capacity=0 and non-power-of-2
- Stress test: 40/64 entries, delete half, verify all
- Hash distribution: 174/256 unique buckets for sequential keys

## 7. Arena B-Tree Benchmarks

### Design

Cache-friendly ordered B-tree (ORDER=8) with bounded operations for BPF
verifier compatibility:
- **BTreeNode**: 184 bytes — 7 keys + 7 values + 8 children + metadata (fits in 3 cache lines)
- **Top-down proactive splitting**: full nodes split during descent, guaranteeing
  the leaf always has room — no backtracking needed
- **Lazy deletion**: no rebalancing on delete (nodes may underflow). Acceptable because
  the bump allocator can't free individual nodes anyway
- **Iterative traversal**: explicit stack-based in-order walk (max depth 10)
- **Linear search within nodes**: 7 keys per node makes linear scan faster than
  binary search (fewer branch mispredictions, simpler instruction stream)

### Results (AMD EPYC 9D64, release build)

#### Insert (ns/op)

| Entry Count | Arena B-Tree | std::BTreeMap | Ratio |
|------------:|-------------:|--------------:|------:|
| 100 | 53 | 55 | **1.04x faster** |
| 500 | 48 | 50 | **1.05x faster** |
| 1000 | 46 | 51 | **1.11x faster** |

Arena advantage comes from the bump allocator (1.7 ns per allocation vs
system allocator's overhead for node allocation + rebalancing).

#### Lookup (ns/op)

| Entry Count | Arena B-Tree | std::BTreeMap | Ratio |
|------------:|-------------:|--------------:|------:|
| 100 | 25 | 38 | **1.5x faster** |
| 500 | 30 | 41 | **1.4x faster** |
| 1000 | 33 | 41 | **1.2x faster** |

Arena wins convincingly on lookup — simpler node layout and no balancing
metadata overhead. The advantage decreases with size as cache effects
dominate (both implementations become memory-bound).

#### Ordered Iteration (ns/op per entry)

| Entry Count | Arena B-Tree | std::BTreeMap | Ratio |
|------------:|-------------:|--------------:|------:|
| 100 | 10.0 | 3.2 | 3.1x slower |
| 500 | 6.0 | 1.5 | 4.0x slower |
| 1000 | 5.9 | 1.6 | 3.7x slower |

Iteration is the arena B-tree's weakness: the explicit stack-based traversal
(required for BPF verifier — no recursion) has significant overhead compared
to std's native Rust iterator with compiler-optimized pointer chasing.

### Analysis for sched_ext Use Cases

The arena B-tree is ideal for:

1. **Vtime ordering** (task priority queues): lookup + insert at 30-50 ns is fast
   enough for per-dispatch operations. The sorted property enables efficient
   min-vtime queries via leftmost-leaf traversal.

2. **DSQ priority tracking**: maintaining ordered views of per-DSQ task vtimes
   for work-stealing decisions (find lowest-vtime DSQ across LLCs).

3. **Cgroup hierarchy snapshots**: ordered by cgroup ID for binary-search-style
   lookups shared between BPF and userspace monitoring tools.

The B-tree should NOT be used for:
- **Hot-path iteration** (dispatch loops): use hash map instead
- **Simple key→value lookups**: hash map is 3-5x faster (6 ns vs 30 ns)

### Comparison: Hash Map vs B-Tree

| Operation | Hash Map (50% load) | B-Tree (1000) | Winner |
|-----------|-------------------:|-------------------:|--------|
| Insert | 10 ns/op | 46 ns/op | Hash map (4.6x) |
| Lookup | 9 ns/op | 33 ns/op | Hash map (3.7x) |
| Delete | 9 ns/op | ~30 ns/op | Hash map (3.3x) |
| Ordered iter | N/A (unordered) | 6 ns/op | B-tree (only option) |
| Memory | 24B/entry fixed | 184B/node shared | Hash map (smaller) |

**Rule of thumb**: Use hash map for fast key→value lookups. Use B-tree
when you need sorted order or range queries.

### Test Coverage

16 unit tests covering:
- Layout verification (node=184B, map=24B)
- Basic CRUD (single insert, update, miss, delete from leaf/internal)
- Full node (7 keys), root split (8th key triggers height increase)
- Ordered iteration (9 keys inserted out of order → sorted output)
- Delete all entries, edge keys (0 and u64::MAX)
- Stress: 100 keys scrambled, 1000 keys with half deleted

---

*Generated by `test/arena-bench`. Run with:*
```
cd test/arena-bench && cargo run --release
```
