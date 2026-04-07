//! Arena data structure benchmarks.
//!
//! Compares arena hash map performance against std::collections::HashMap
//! at various load factors. All arena operations run in simulated arena
//! memory (process-local buffer), measuring the raw algorithmic cost.

use aya_arena_common::{
    ArenaBTreeMap, ArenaBumpState, ArenaHashEntry, ArenaHashMap, ArenaNodeHeader, ArenaPtr,
    ArenaSlabState, CounterNode, arena_btree_for_each, arena_btree_get, arena_btree_init,
    arena_btree_insert, arena_hash_delete, arena_hash_get, arena_hash_init, arena_hash_insert,
    arena_slab_alloc, arena_slab_free, arena_slab_init,
};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

// ── Timing helpers ────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    ops: u64,
    elapsed_ns: u64,
}

impl BenchResult {
    fn ns_per_op(&self) -> f64 {
        self.elapsed_ns as f64 / self.ops as f64
    }

    fn ops_per_sec(&self) -> f64 {
        self.ops as f64 / (self.elapsed_ns as f64 / 1e9)
    }

    fn print(&self) {
        println!(
            "  {:<40} {:>10} ops  {:>8.1} ns/op  {:>12.0} ops/sec",
            self.name,
            self.ops,
            self.ns_per_op(),
            self.ops_per_sec(),
        );
    }
}

/// Run a closure `iters` times and return elapsed nanoseconds.
fn bench<F: FnMut()>(iters: u64, mut f: F) -> u64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as u64
}

// ── Arena hash map setup ──────────────────────────────────────────────

struct ArenaMapFixture {
    buf: Vec<u8>,
    map: *mut ArenaHashMap,
    base: *mut u8,
    capacity: u32,
}

impl ArenaMapFixture {
    fn new(capacity: u32) -> Self {
        let header_size = size_of::<ArenaHashMap>();
        let entry_size = size_of::<ArenaHashEntry>();
        let total = header_size + entry_size * capacity as usize;
        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();
        let map = base.cast::<ArenaHashMap>();
        let entries = unsafe { base.add(header_size) }.cast::<ArenaHashEntry>();
        unsafe { arena_hash_init(map, entries, capacity, base) };
        Self {
            buf,
            map,
            base,
            capacity,
        }
    }

    fn reset(&mut self) {
        let header_size = size_of::<ArenaHashMap>();
        let entries = unsafe { self.base.add(header_size) }.cast::<ArenaHashEntry>();
        unsafe { arena_hash_init(self.map, entries, self.capacity, self.base) };
    }

    fn base(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }
}

// ── Hash map benchmarks ──────────────────────────────────────────────

fn bench_hash_insert(capacity: u32, fill_count: u32) -> BenchResult {
    let mut fix = ArenaMapFixture::new(capacity);
    let base = fix.base();
    let map = fix.map;
    let ops = fill_count as u64;

    let elapsed = bench(1, || {
        fix.reset();
        for i in 0..fill_count {
            unsafe { arena_hash_insert(map, (i as u64) * 7 + 1, i as u64, base) };
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("arena insert {fill_count}/{capacity} ({load_pct}% load)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_insert(capacity: u32, fill_count: u32) -> BenchResult {
    let ops = fill_count as u64;

    let elapsed = bench(1, || {
        let mut map = HashMap::with_capacity(capacity as usize);
        for i in 0..fill_count {
            map.insert((i as u64) * 7 + 1, i as u64);
        }
        std::hint::black_box(&map);
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("std insert {fill_count}/{capacity} ({load_pct}% load)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_hash_lookup_hit(capacity: u32, fill_count: u32) -> BenchResult {
    let mut fix = ArenaMapFixture::new(capacity);
    let base = fix.base();
    let map = fix.map;

    // Fill
    for i in 0..fill_count {
        unsafe { arena_hash_insert(map, (i as u64) * 7 + 1, i as u64, base) };
    }

    let ops = fill_count as u64;
    let elapsed = bench(1, || {
        for i in 0..fill_count {
            let v = unsafe { arena_hash_get(map, (i as u64) * 7 + 1, base) };
            std::hint::black_box(v);
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("arena lookup hit {fill_count}/{capacity} ({load_pct}%)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_lookup_hit(capacity: u32, fill_count: u32) -> BenchResult {
    let mut map = HashMap::with_capacity(capacity as usize);
    for i in 0..fill_count {
        map.insert((i as u64) * 7 + 1, i as u64);
    }

    let ops = fill_count as u64;
    let elapsed = bench(1, || {
        for i in 0..fill_count {
            let v = map.get(&((i as u64) * 7 + 1));
            std::hint::black_box(v);
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("std lookup hit {fill_count}/{capacity} ({load_pct}%)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_hash_lookup_miss(capacity: u32, fill_count: u32) -> BenchResult {
    let mut fix = ArenaMapFixture::new(capacity);
    let base = fix.base();
    let map = fix.map;

    // Fill with keys k*7+1
    for i in 0..fill_count {
        unsafe { arena_hash_insert(map, (i as u64) * 7 + 1, i as u64, base) };
    }

    // Lookup keys that DON'T exist (even numbers * 7 + 2)
    let ops = fill_count as u64;
    let elapsed = bench(1, || {
        for i in 0..fill_count {
            let v = unsafe { arena_hash_get(map, (i as u64) * 7 + 2, base) };
            std::hint::black_box(v);
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("arena lookup miss {fill_count}/{capacity} ({load_pct}%)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_lookup_miss(capacity: u32, fill_count: u32) -> BenchResult {
    let mut map = HashMap::with_capacity(capacity as usize);
    for i in 0..fill_count {
        map.insert((i as u64) * 7 + 1, i as u64);
    }

    let ops = fill_count as u64;
    let elapsed = bench(1, || {
        for i in 0..fill_count {
            let v = map.get(&((i as u64) * 7 + 2));
            std::hint::black_box(v);
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("std lookup miss {fill_count}/{capacity} ({load_pct}%)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_hash_delete(capacity: u32, fill_count: u32) -> BenchResult {
    let mut fix = ArenaMapFixture::new(capacity);
    let base = fix.base();
    let map = fix.map;

    for i in 0..fill_count {
        unsafe { arena_hash_insert(map, (i as u64) * 7 + 1, i as u64, base) };
    }

    let ops = fill_count as u64;
    let elapsed = bench(1, || {
        for i in 0..fill_count {
            unsafe { arena_hash_delete(map, (i as u64) * 7 + 1, base) };
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("arena delete {fill_count}/{capacity} ({load_pct}%)"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_delete(capacity: u32, fill_count: u32) -> BenchResult {
    let mut map = HashMap::with_capacity(capacity as usize);
    for i in 0..fill_count {
        map.insert((i as u64) * 7 + 1, i as u64);
    }

    let ops = fill_count as u64;
    let elapsed = bench(1, || {
        for i in 0..fill_count {
            map.remove(&((i as u64) * 7 + 1));
        }
    });

    let load_pct = fill_count * 100 / capacity;
    BenchResult {
        name: format!("std delete {fill_count}/{capacity} ({load_pct}%)"),
        ops,
        elapsed_ns: elapsed,
    }
}

// ── Linked list benchmark ─────────────────────────────────────────────

fn bench_linked_list_build(node_count: u32) -> BenchResult {
    let ops = node_count as u64;
    let buf_size = 8 + 16 + (node_count as usize) * size_of::<CounterNode>() + 1024;
    let mut buf = vec![0u8; buf_size];
    let base = buf.as_mut_ptr();

    let elapsed = bench(1, || {
        let mut bump = ArenaBumpState::new(buf_size as u64);
        let mut head: ArenaPtr<ArenaNodeHeader> = ArenaPtr::null();

        for i in 0..node_count {
            let offset = bump.alloc(size_of::<CounterNode>() as u64, 8).unwrap();
            let node = unsafe { base.add(offset as usize).cast::<CounterNode>() };
            unsafe {
                *node = CounterNode {
                    header: ArenaNodeHeader {
                        tag: 1,
                        size: size_of::<CounterNode>() as u32,
                        next: head,
                    },
                    value: i as u64,
                };
            }
            head = ArenaPtr::from_offset(offset);
        }
        std::hint::black_box(head);
    });

    BenchResult {
        name: format!("linked list build {node_count} nodes"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_linked_list_traverse(node_count: u32) -> BenchResult {
    let buf_size = (node_count as usize) * size_of::<CounterNode>() + 1024;
    let mut buf = vec![0u8; buf_size];
    let base = buf.as_mut_ptr();

    // Build list
    let mut bump = ArenaBumpState::new(buf_size as u64);
    let mut head: ArenaPtr<ArenaNodeHeader> = ArenaPtr::null();
    for i in 0..node_count {
        let offset = bump.alloc(size_of::<CounterNode>() as u64, 8).unwrap();
        let node = unsafe { base.add(offset as usize).cast::<CounterNode>() };
        unsafe {
            *node = CounterNode {
                header: ArenaNodeHeader {
                    tag: 1,
                    size: size_of::<CounterNode>() as u32,
                    next: head,
                },
                value: i as u64,
            };
        }
        head = ArenaPtr::from_offset(offset);
    }

    let ops = node_count as u64;
    let elapsed = bench(1, || {
        let mut current = head;
        let mut sum: u64 = 0;
        let mut steps = 0u32;
        while !current.is_null() && steps < node_count + 1 {
            let ptr = unsafe { current.resolve(base) };
            if ptr.is_null() {
                break;
            }
            let hdr = unsafe { &*ptr };
            let node = ptr.cast::<CounterNode>();
            sum += unsafe { (*node).value };
            current = hdr.next;
            steps += 1;
        }
        std::hint::black_box(sum);
    });

    BenchResult {
        name: format!("linked list traverse {node_count} nodes"),
        ops,
        elapsed_ns: elapsed,
    }
}

// ── Bump allocator benchmark ──────────────────────────────────────────

fn bench_bump_alloc(alloc_count: u32, alloc_size: u64) -> BenchResult {
    let ops = alloc_count as u64;
    let capacity = alloc_count as u64 * (alloc_size + 16); // generous

    let elapsed = bench(1, || {
        let mut bump = ArenaBumpState::new(capacity);
        for _ in 0..alloc_count {
            let _ = bump.alloc(alloc_size, 8);
        }
        std::hint::black_box(bump.watermark);
    });

    BenchResult {
        name: format!("bump alloc {alloc_count}x {alloc_size}B"),
        ops,
        elapsed_ns: elapsed,
    }
}

// ── B-tree benchmarks ─────────────────────────────────────────────────

struct BTreeFixture {
    buf: Vec<u8>,
    map: *mut ArenaBTreeMap,
    bump: *mut ArenaBumpState,
    base: *mut u8,
}

impl BTreeFixture {
    fn new() -> Self {
        let arena_size = 4 * 1024 * 1024; // 4 MiB
        let mut buf = vec![0u8; arena_size];
        let base = buf.as_mut_ptr();
        let map = base.cast::<ArenaBTreeMap>();
        unsafe { arena_btree_init(map) };

        let bump_offset = size_of::<ArenaBTreeMap>();
        let bump = unsafe { base.add(bump_offset).cast::<ArenaBumpState>() };
        let data_start = bump_offset + size_of::<ArenaBumpState>();
        unsafe {
            *bump = ArenaBumpState::new((arena_size - data_start) as u64);
            (*bump).watermark = data_start as u64;
        }
        Self {
            buf,
            map,
            bump,
            base,
        }
    }

    fn reset(&mut self) {
        let arena_size = self.buf.len();
        self.base = self.buf.as_mut_ptr();
        self.map = self.base.cast::<ArenaBTreeMap>();
        unsafe { arena_btree_init(self.map) };

        let bump_offset = size_of::<ArenaBTreeMap>();
        self.bump = unsafe { self.base.add(bump_offset).cast::<ArenaBumpState>() };
        let data_start = bump_offset + size_of::<ArenaBumpState>();
        unsafe {
            *self.bump = ArenaBumpState::new((arena_size - data_start) as u64);
            (*self.bump).watermark = data_start as u64;
        }
    }
}

fn bench_btree_insert(count: u32) -> BenchResult {
    let mut fix = BTreeFixture::new();
    let ops = count as u64;

    let elapsed = bench(1, || {
        fix.reset();
        for i in 0..count {
            unsafe {
                arena_btree_insert(fix.map, fix.bump, (i as u64) * 7 + 1, i as u64, fix.base);
            }
        }
    });

    BenchResult {
        name: format!("arena btree insert {count}"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_btree_insert(count: u32) -> BenchResult {
    let ops = count as u64;

    let elapsed = bench(1, || {
        let mut map = BTreeMap::new();
        for i in 0..count {
            map.insert((i as u64) * 7 + 1, i as u64);
        }
        std::hint::black_box(&map);
    });

    BenchResult {
        name: format!("std btree insert {count}"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_btree_lookup(count: u32) -> BenchResult {
    let fix = BTreeFixture::new();
    for i in 0..count {
        unsafe {
            arena_btree_insert(fix.map, fix.bump, (i as u64) * 7 + 1, i as u64, fix.base);
        }
    }

    let ops = count as u64;
    let elapsed = bench(1, || {
        for i in 0..count {
            let v = unsafe { arena_btree_get(fix.map, (i as u64) * 7 + 1, fix.base) };
            std::hint::black_box(v);
        }
    });

    BenchResult {
        name: format!("arena btree lookup {count}"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_btree_lookup(count: u32) -> BenchResult {
    let mut map = BTreeMap::new();
    for i in 0..count {
        map.insert((i as u64) * 7 + 1, i as u64);
    }

    let ops = count as u64;
    let elapsed = bench(1, || {
        for i in 0..count {
            let v = map.get(&((i as u64) * 7 + 1));
            std::hint::black_box(v);
        }
    });

    BenchResult {
        name: format!("std btree lookup {count}"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_btree_iterate(count: u32) -> BenchResult {
    let fix = BTreeFixture::new();
    for i in 0..count {
        unsafe {
            arena_btree_insert(fix.map, fix.bump, (i as u64) * 7 + 1, i as u64, fix.base);
        }
    }

    let ops = count as u64;
    let elapsed = bench(1, || {
        let mut sum: u64 = 0;
        unsafe {
            arena_btree_for_each(fix.map, fix.base, |_k, v| {
                sum += v;
            });
        }
        std::hint::black_box(sum);
    });

    BenchResult {
        name: format!("arena btree iterate {count}"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_std_btree_iterate(count: u32) -> BenchResult {
    let mut map = BTreeMap::new();
    for i in 0..count {
        map.insert((i as u64) * 7 + 1, i as u64);
    }

    let ops = count as u64;
    let elapsed = bench(1, || {
        let mut sum: u64 = 0;
        for &v in map.values() {
            sum += v;
        }
        std::hint::black_box(sum);
    });

    BenchResult {
        name: format!("std btree iterate {count}"),
        ops,
        elapsed_ns: elapsed,
    }
}

// ── Slab allocator benchmarks ─────────────────────────────────────────

fn bench_slab_alloc_free_cycle(count: u32, slot_size: u32) -> BenchResult {
    let header_size = size_of::<ArenaSlabState>();
    let arena_size = header_size + (count as usize) * (slot_size as usize) + 4096;
    let mut buf = vec![0u8; arena_size];
    let base = buf.as_mut_ptr();
    let slab = base.cast::<ArenaSlabState>();
    unsafe {
        arena_slab_init(slab, arena_size as u64, slot_size);
        (*slab).bump.watermark = header_size as u64;
    }

    // Pre-allocate all slots, then free them to fill the free list
    let mut slots: Vec<ArenaPtr<u8>> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        slots.push(unsafe { arena_slab_alloc(slab, base) });
    }
    for &s in &slots {
        unsafe { arena_slab_free(slab, s, base) };
    }

    // Benchmark: alloc + free cycle (all from free list)
    let ops = count as u64;
    let elapsed = bench(1, || {
        for _ in 0..count {
            let s = unsafe { arena_slab_alloc(slab, base) };
            unsafe { arena_slab_free(slab, s, base) };
        }
    });

    BenchResult {
        name: format!("slab alloc+free cycle {count}x {slot_size}B"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_slab_alloc_only(count: u32, slot_size: u32) -> BenchResult {
    let header_size = size_of::<ArenaSlabState>();
    let arena_size = header_size + (count as usize) * (slot_size as usize) + 4096;
    let mut buf = vec![0u8; arena_size];
    let base = buf.as_mut_ptr();
    let slab = base.cast::<ArenaSlabState>();
    unsafe {
        arena_slab_init(slab, arena_size as u64, slot_size);
        (*slab).bump.watermark = header_size as u64;
    }

    // Pre-alloc and free to fill the free list
    let mut slots: Vec<ArenaPtr<u8>> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        slots.push(unsafe { arena_slab_alloc(slab, base) });
    }
    for &s in &slots {
        unsafe { arena_slab_free(slab, s, base) };
    }

    // Benchmark: alloc-only from free list
    let ops = count as u64;
    let elapsed = bench(1, || {
        for _ in 0..count {
            let s = unsafe { arena_slab_alloc(slab, base) };
            std::hint::black_box(s);
        }
    });

    BenchResult {
        name: format!("slab alloc-only (free list) {count}x {slot_size}B"),
        ops,
        elapsed_ns: elapsed,
    }
}

fn bench_slab_bump_alloc(count: u32, slot_size: u32) -> BenchResult {
    let header_size = size_of::<ArenaSlabState>();
    let arena_size = header_size + (count as usize) * (slot_size as usize) + 4096;
    let mut buf = vec![0u8; arena_size];
    let base = buf.as_mut_ptr();
    let slab = base.cast::<ArenaSlabState>();

    let ops = count as u64;
    let elapsed = bench(1, || {
        // Re-init to reset bump watermark each time
        unsafe {
            arena_slab_init(slab, arena_size as u64, slot_size);
            (*slab).bump.watermark = header_size as u64;
        }
        for _ in 0..count {
            let s = unsafe { arena_slab_alloc(slab, base) };
            std::hint::black_box(s);
        }
    });

    BenchResult {
        name: format!("slab alloc-only (bump path) {count}x {slot_size}B"),
        ops,
        elapsed_ns: elapsed,
    }
}

// ── Main: run all benchmarks ─────────────────────────────────────────

fn main() {
    println!("Arena Data Structure Benchmarks");
    println!("==============================\n");

    // Use larger iteration counts for more stable timing
    let capacity: u32 = 1024;
    let load_factors: &[(u32, &str)] = &[
        (capacity / 4, "25%"),
        (capacity / 2, "50%"),
        (capacity * 3 / 4, "75%"),
        (capacity * 9 / 10, "90%"),
    ];

    // ── Hash Map: Insert ──────────────────────────────────────────
    println!("Hash Map Insert (capacity={capacity}):");
    for &(fill, _label) in load_factors {
        bench_hash_insert(capacity, fill).print();
        bench_std_insert(capacity, fill).print();
        println!();
    }

    // ── Hash Map: Lookup Hit ──────────────────────────────────────
    println!("Hash Map Lookup Hit (capacity={capacity}):");
    for &(fill, _label) in load_factors {
        bench_hash_lookup_hit(capacity, fill).print();
        bench_std_lookup_hit(capacity, fill).print();
        println!();
    }

    // ── Hash Map: Lookup Miss ─────────────────────────────────────
    println!("Hash Map Lookup Miss (capacity={capacity}):");
    for &(fill, _label) in load_factors {
        bench_hash_lookup_miss(capacity, fill).print();
        bench_std_lookup_miss(capacity, fill).print();
        println!();
    }

    // ── Hash Map: Delete ──────────────────────────────────────────
    println!("Hash Map Delete (capacity={capacity}):");
    for &(fill, _label) in load_factors {
        bench_hash_delete(capacity, fill).print();
        bench_std_delete(capacity, fill).print();
        println!();
    }

    // ── Linked List ───────────────────────────────────────────────
    println!("Linked List Operations:");
    for &count in &[100, 1000, 10000] {
        bench_linked_list_build(count).print();
        bench_linked_list_traverse(count).print();
        println!();
    }

    // ── Bump Allocator ────────────────────────────────────────────
    println!("Bump Allocator:");
    for &(count, size) in &[(1000, 64), (1000, 256), (10000, 64), (10000, 256)] {
        bench_bump_alloc(count, size).print();
    }
    println!();

    // ── B-tree vs std::BTreeMap ──────────────────────────────────
    println!("B-Tree Insert:");
    for &count in &[100, 500, 1000] {
        bench_btree_insert(count).print();
        bench_std_btree_insert(count).print();
        println!();
    }

    println!("B-Tree Lookup:");
    for &count in &[100, 500, 1000] {
        bench_btree_lookup(count).print();
        bench_std_btree_lookup(count).print();
        println!();
    }

    println!("B-Tree Ordered Iteration:");
    for &count in &[100, 500, 1000] {
        bench_btree_iterate(count).print();
        bench_std_btree_iterate(count).print();
        println!();
    }

    // ── Slab Allocator ───────────────────────────────────────────
    println!("Slab Allocator (alloc+free cycle):");
    for &(count, size) in &[(1000, 32), (1000, 64), (1000, 256), (10000, 64)] {
        bench_slab_alloc_free_cycle(count, size).print();
    }
    println!();

    println!("Slab Allocator (alloc-only comparison):");
    for &(count, size) in &[(10000, 64), (10000, 256)] {
        bench_slab_alloc_only(count, size).print();
        bench_slab_bump_alloc(count, size).print();
        bench_bump_alloc(count, size as u64).print();
        println!();
    }

    // ── Summary ───────────────────────────────────────────────────
    println!("Notes:");
    println!("  - Arena hash map uses open addressing with linear probing");
    println!("  - std::HashMap uses Robin Hood hashing with dynamic resizing");
    println!("  - Arena map is designed for BPF verifier compatibility (bounded probes)");
    println!("  - Arena map has zero allocation overhead (uses pre-allocated slots)");
    println!("  - Higher load factors increase probe chain length for arena map");
    println!("  - All measurements are single-run; results vary with system load");
}
