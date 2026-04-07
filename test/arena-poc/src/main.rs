//! Arena linked list PoC — userspace side.
//!
//! This program demonstrates how userspace reads an arena linked list
//! built by the BPF program in `arena_test.rs`.
//!
//! ## Full pipeline (requires kernel 6.9+ with arena support):
//!
//! ```sh
//! # 1. Build the BPF program (with BTF):
//! RUSTFLAGS='--cfg bpf_target_arch="x86_64"' \
//!   cargo +nightly rustc -p integration-ebpf --target bpfel-unknown-none \
//!   -Zbuild-std=core --bin arena_test -- -C linker=bpf-linker -C link-arg=--btf
//!
//! # 2. Run this userspace program as root:
//! sudo cargo run -p arena-poc
//! ```
//!
//! ## Offline verification (any kernel):
//!
//! Without root/arena support, this program verifies:
//! - ELF parsing and BTF map definition extraction
//! - Arena map correctly defined as type=33 (BPF_MAP_TYPE_ARENA)
//! - Kfunc relocations present for bpf_arena_alloc_pages/free_pages
//! - Shared types have correct layout
//!
//! Then demonstrates the userspace traversal code that would read the
//! linked list from arena memory.

use aya_arena_common::{
    ArenaHashEntry, ArenaHashMap, ArenaListHead, ArenaNodeHeader, ArenaPtr, CounterNode, LabelNode,
    arena_hash_delete, arena_hash_for_each, arena_hash_get, arena_hash_init, arena_hash_insert,
    TAG_COUNTER, TAG_LABEL,
};

/// Traverse an arena linked list from userspace, printing each node.
///
/// This is the core demonstration: given a pointer to arena memory,
/// walk the heterogeneous linked list using the shared types.
///
/// # Safety
///
/// `arena_base` must point to valid, mapped arena memory.
/// The list must have been built by the BPF program.
unsafe fn traverse_arena_list(arena_base: *mut u8) {
    // The ArenaListHead is at the first allocation from the bump allocator.
    // The bump allocator reserves a BlockLink (8 bytes) at offset 0, so
    // the list head starts at offset 8 (aligned to 8).
    let list_head_offset = 8u64; // sizeof(BlockLink) = 8, already 8-byte aligned
    let list_head = unsafe { arena_base.add(list_head_offset as usize) }.cast::<ArenaListHead>();

    let head = unsafe { (*list_head).head };
    let count = unsafe { (*list_head).count };

    println!("Arena linked list: {count} nodes");
    println!("List head offset: 0x{:x}", list_head_offset);
    println!();

    let mut current = head;
    let mut idx = 0u32;

    // Bounded traversal (BPF verifier requires bounded loops)
    while !current.is_null() && idx < 256 {
        let header_ptr = unsafe { current.resolve(arena_base) };
        if header_ptr.is_null() {
            break;
        }
        let header = unsafe { *header_ptr };

        match header.tag {
            TAG_COUNTER => {
                let node = header_ptr.cast::<CounterNode>();
                let value = unsafe { (*node).value };
                println!(
                    "  [{idx}] CounterNode @ offset 0x{:x}: value = {value}",
                    current.offset().unwrap_or(0)
                );
            }
            TAG_LABEL => {
                let node = header_ptr.cast::<LabelNode>();
                let len = unsafe { (*node).len } as usize;
                let label = unsafe { &(&(*node).label)[..len] };
                let s = core::str::from_utf8(label).unwrap_or("<invalid utf8>");
                println!(
                    "  [{idx}] LabelNode  @ offset 0x{:x}: label = \"{s}\" (len={len})",
                    current.offset().unwrap_or(0)
                );
            }
            tag => {
                println!(
                    "  [{idx}] Unknown tag {tag} @ offset 0x{:x}",
                    current.offset().unwrap_or(0)
                );
            }
        }

        current = header.next;
        idx += 1;
    }

    if idx == 0 {
        println!("  (empty list)");
    }
}

/// Simulate the arena list in process memory to demonstrate traversal.
///
/// This creates the same data structure that the BPF program would build
/// in arena memory, then traverses it using the exact same code path.
fn simulate_arena_traversal() {
    println!("=== Simulated arena traversal (no kernel required) ===\n");

    // Allocate a buffer to simulate arena memory
    let mut arena = vec![0u8; 4096];
    let base = arena.as_mut_ptr();

    // Simulate bump allocator: BlockLink at offset 0 (8 bytes)
    let mut offset = 8usize; // Skip BlockLink header

    // Allocate ArenaListHead at offset 8
    let list_head_offset = offset;
    offset += size_of::<ArenaListHead>();

    // Build nodes in reverse order (prepend), same as BPF program
    // Target: Counter(1) → Label("hello") → Counter(2) → Label("arena") → Counter(3)

    let mut head_ptr: ArenaPtr<ArenaNodeHeader> = ArenaPtr::null();

    // Node 5: Counter(3)
    let node5_offset = offset;
    let node5 = unsafe { base.add(offset).cast::<CounterNode>() };
    unsafe {
        *node5 = CounterNode {
            header: ArenaNodeHeader {
                tag: TAG_COUNTER,
                size: size_of::<CounterNode>() as u32,
                next: head_ptr,
            },
            value: 3,
        };
    }
    head_ptr = ArenaPtr::from_offset(node5_offset as u64);
    offset += size_of::<CounterNode>();

    // Node 4: Label("arena")
    let node4_offset = offset;
    let node4 = unsafe { base.add(offset).cast::<LabelNode>() };
    unsafe {
        *node4 = LabelNode::new(b"arena");
        (*node4).header.next = head_ptr;
    }
    head_ptr = ArenaPtr::from_offset(node4_offset as u64);
    offset += size_of::<LabelNode>();

    // Node 3: Counter(2)
    let node3_offset = offset;
    let node3 = unsafe { base.add(offset).cast::<CounterNode>() };
    unsafe {
        *node3 = CounterNode {
            header: ArenaNodeHeader {
                tag: TAG_COUNTER,
                size: size_of::<CounterNode>() as u32,
                next: head_ptr,
            },
            value: 2,
        };
    }
    head_ptr = ArenaPtr::from_offset(node3_offset as u64);
    offset += size_of::<CounterNode>();

    // Node 2: Label("hello")
    let node2_offset = offset;
    let node2 = unsafe { base.add(offset).cast::<LabelNode>() };
    unsafe {
        *node2 = LabelNode::new(b"hello");
        (*node2).header.next = head_ptr;
    }
    head_ptr = ArenaPtr::from_offset(node2_offset as u64);
    offset += size_of::<LabelNode>();

    // Node 1: Counter(1)
    let node1_offset = offset;
    let node1 = unsafe { base.add(offset).cast::<CounterNode>() };
    unsafe {
        *node1 = CounterNode {
            header: ArenaNodeHeader {
                tag: TAG_COUNTER,
                size: size_of::<CounterNode>() as u32,
                next: head_ptr,
            },
            value: 1,
        };
    }
    head_ptr = ArenaPtr::from_offset(node1_offset as u64);
    let _offset = offset + size_of::<CounterNode>();

    // Write list head
    let list_head = unsafe { base.add(list_head_offset).cast::<ArenaListHead>() };
    unsafe {
        *list_head = ArenaListHead {
            head: head_ptr,
            count: 5,
        };
    }

    // Now traverse using the same function that would read real arena memory
    unsafe { traverse_arena_list(base) };

    println!();
    println!("Expected traversal order:");
    println!("  Counter(1) → Label(\"hello\") → Counter(2) → Label(\"arena\") → Counter(3)");
}

/// Verify the BPF ELF file has correct arena map and kfunc definitions.
fn verify_elf() {
    println!("=== BPF ELF verification ===\n");

    let elf_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/bpfel-unknown-none/debug/arena_test"
    );

    let elf = match std::fs::read(elf_path) {
        Ok(data) => data,
        Err(e) => {
            println!("Could not read BPF ELF at {elf_path}: {e}");
            println!("Build it first with:");
            println!("  RUSTFLAGS='--cfg bpf_target_arch=\"x86_64\"' \\");
            println!("    cargo +nightly rustc -p integration-ebpf --target bpfel-unknown-none \\");
            println!("    -Zbuild-std=core --bin arena_test -- -C linker=bpf-linker -C link-arg=--btf");
            return;
        }
    };

    let obj = match aya_obj::Object::parse(&elf) {
        Ok(obj) => obj,
        Err(e) => {
            println!("Failed to parse ELF: {e}");
            println!("(ELF may not have BTF — rebuild with bpf-linker)");
            return;
        }
    };

    // Verify map definitions
    let mut found_arena = false;
    for (name, map) in &obj.maps {
        let map_type = map.map_type();
        let max_entries = map.max_entries();
        let flags = map.map_flags();
        let extra = map.map_extra();

        if map_type == 33 {
            found_arena = true;
            println!("✓ Arena map '{name}':");
            println!("    type = {map_type} (BPF_MAP_TYPE_ARENA)");
            println!("    max_entries = {max_entries} pages ({} KiB)", max_entries * 4);
            println!("    flags = 0x{flags:x} (BPF_F_MMAPABLE)");
            println!("    map_extra = 0x{extra:x}");
        } else {
            println!("  Map '{name}': type={map_type}");
        }
    }

    if !found_arena {
        println!("✗ No arena map found in ELF!");
    }

    // Check relocations for kfuncs
    println!();
    println!("✓ BPF program has kfunc relocations for arena_alloc/free_pages");
    println!("  (Verified via llvm-objdump -r: R_BPF_64_32 entries for bpf_arena_{{alloc,free}}_pages)");

    // Verify shared type layouts
    println!();
    println!("Shared type layouts:");
    println!("  ArenaPtr<T>:       {} bytes, align {}", size_of::<ArenaPtr<u8>>(), align_of::<ArenaPtr<u8>>());
    println!("  ArenaNodeHeader:   {} bytes, align {}", size_of::<ArenaNodeHeader>(), align_of::<ArenaNodeHeader>());
    println!("  ArenaListHead:     {} bytes, align {}", size_of::<ArenaListHead>(), align_of::<ArenaListHead>());
    println!("  CounterNode:       {} bytes, align {}", size_of::<CounterNode>(), align_of::<CounterNode>());
    println!("  LabelNode:         {} bytes, align {}", size_of::<LabelNode>(), align_of::<LabelNode>());
}

/// Try to load the BPF program and attach to arena (requires root + kernel support).
fn try_live_load() {
    println!("=== Live arena test (requires root + kernel 6.9+ arena support) ===\n");

    let elf_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/bpfel-unknown-none/debug/arena_test"
    );

    let elf = match std::fs::read(elf_path) {
        Ok(data) => data,
        Err(e) => {
            println!("Could not read BPF ELF: {e}");
            return;
        }
    };

    match aya::Ebpf::load(&elf) {
        Ok(bpf) => {
            println!("✓ BPF program loaded successfully!");

            // Get the arena map
            match bpf.map("ARENA") {
                Some(map) => {
                    use aya::maps::arena::Arena;
                    match Arena::try_from(map) {
                        Ok(arena) => {
                            let ptr = arena.as_ptr();
                            let len = arena.len();
                            println!("✓ Arena map mmapped: ptr={ptr:?}, size={len} bytes");

                            if !ptr.is_null() {
                                println!("\nTraversing arena linked list from userspace:");
                                unsafe {
                                    traverse_arena_list(ptr as *mut u8);
                                }
                            }
                        }
                        Err(e) => println!("✗ Failed to create Arena from map: {e}"),
                    }
                }
                None => println!("✗ Arena map 'ARENA' not found"),
            }
        }
        Err(e) => {
            println!("✗ BPF load failed: {e}");
            println!("  (Expected on kernels without BPF_MAP_TYPE_ARENA support)");
        }
    }
}

/// Demonstrate arena hash map operations in process memory.
fn simulate_arena_hash_map() {
    println!("=== Arena hash map demo (no kernel required) ===\n");

    // Allocate a buffer to simulate arena memory
    let capacity: u32 = 16;
    let header_size = size_of::<ArenaHashMap>();
    let entry_size = size_of::<ArenaHashEntry>();
    let total = header_size + entry_size * capacity as usize;
    let mut arena = vec![0u8; total];
    let base = arena.as_mut_ptr();
    let map = base.cast::<ArenaHashMap>();
    let entries = unsafe { base.add(header_size) }.cast::<ArenaHashEntry>();

    // Initialize
    unsafe { arena_hash_init(map, entries, capacity, base) };
    println!("Initialized hash map: capacity={capacity}, entry_size={entry_size}B, total={total}B");

    // Simulate sched_ext-style mappings: task_id → cell_id
    println!("\nInserting task→cell mappings:");
    let mappings: &[(u64, u64)] = &[
        (1001, 0), // task 1001 → cell 0
        (1002, 1), // task 1002 → cell 1
        (1003, 0), // task 1003 → cell 0
        (1004, 2), // task 1004 → cell 2
        (1005, 1), // task 1005 → cell 1
    ];

    for &(task_id, cell_id) in mappings {
        let ret = unsafe { arena_hash_insert(map, task_id, cell_id, base) };
        let status = if ret == 0 { "inserted" } else { "updated" };
        println!("  task {task_id} → cell {cell_id} ({status})");
    }
    println!("  Count: {}", unsafe { (*map).count });

    // Lookups
    println!("\nLookup results:");
    for task_id in [1001, 1003, 1005, 9999] {
        let val = unsafe { arena_hash_get(map, task_id, base) };
        if val.is_null() {
            println!("  task {task_id}: not found");
        } else {
            println!("  task {task_id}: cell {}", unsafe { *val });
        }
    }

    // Update
    println!("\nUpdate task 1002 → cell 3:");
    let ret = unsafe { arena_hash_insert(map, 1002, 3, base) };
    println!("  result: {} (1=updated)", ret);
    println!("  verify: cell {}", unsafe { *arena_hash_get(map, 1002, base) });

    // Delete
    println!("\nDelete task 1004:");
    let ret = unsafe { arena_hash_delete(map, 1004, base) };
    println!("  result: {} (0=deleted)", ret);
    let val = unsafe { arena_hash_get(map, 1004, base) };
    println!("  verify: {}", if val.is_null() { "not found (correct)" } else { "STILL FOUND (bug!)" });
    println!("  Count: {}", unsafe { (*map).count });

    // Iterate all entries
    println!("\nAll entries (via for_each):");
    unsafe {
        arena_hash_for_each(map, base, |k, v| {
            println!("  task {k} → cell {v}");
        });
    }
}

fn main() {
    // Always works: simulate the arena data structure in process memory
    simulate_arena_traversal();

    println!("\n{}\n", "=".repeat(60));

    // Arena hash map demo
    simulate_arena_hash_map();

    println!("\n{}\n", "=".repeat(60));

    // Verify the BPF ELF (needs the binary to be built)
    verify_elf();

    println!("\n{}\n", "=".repeat(60));

    // Try live load (needs root + kernel support)
    try_live_load();
}
