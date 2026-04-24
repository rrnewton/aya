use aya::{EbpfLoader, Ebpf, maps::Array, maps::arena::Arena, programs::SyscallProgram, util::KernelVersion};
use aya::VerifierLogLevel;
use aya_arena_common::{
    ArenaListHead, ArenaNodeHeader, ArenaPtr, CounterNode, LabelNode, TAG_COUNTER, TAG_LABEL,
    arena_hash_for_each, arena_hash_get, arena_btree_get, arena_btree_for_each,
    ArenaHashMap, ArenaBTreeMap,
};

fn skip_if_no_arena() -> bool {
    let kv = KernelVersion::current().unwrap();
    if kv < KernelVersion::new(6, 9, 0) {
        eprintln!("skipping arena test on kernel {kv:?}, arena requires 6.9+");
        return true;
    }
    false
}

/// Load a BPF_PROG_TYPE_SYSCALL program, invoke it via test_run,
/// and return the Ebpf instance for assertion.
fn load_and_trigger(bpf_bytes: &[u8], prog_name: &str) -> Ebpf {
    let mut bpf = EbpfLoader::new()
        .verifier_log_level(VerifierLogLevel::VERBOSE)
        .load(bpf_bytes).unwrap();

    let prog: &mut SyscallProgram = bpf
        .program_mut(prog_name)
        .unwrap()
        .try_into()
        .unwrap();
    prog.load().unwrap();
    prog.test_run().unwrap();

    bpf
}

#[test_log::test]
fn arena_hashmap() {
    if skip_if_no_arena() {
        return;
    }

    let bpf = load_and_trigger(crate::ARENA_HASHMAP_TEST, "arena_hashmap_test");
    let results = Array::<_, u64>::try_from(bpf.map("RESULTS").unwrap()).unwrap();

    assert_eq!(results.get(&0, 0).unwrap(), 5, "count after 5 inserts");
    assert_eq!(results.get(&1, 0).unwrap(), 10, "get(1001) should be 10");
    assert_eq!(results.get(&2, 0).unwrap(), 50, "get(1005) should be 50");
    assert_eq!(
        results.get(&3, 0).unwrap(),
        1,
        "get(9999) should return null (flag=1)"
    );
    assert_eq!(
        results.get(&4, 0).unwrap(),
        0,
        "delete(1003) should succeed (0)"
    );
    assert_eq!(
        results.get(&5, 0).unwrap(),
        4,
        "count after delete should be 4"
    );
    assert_eq!(
        results.get(&6, 0).unwrap(),
        1,
        "get(1003) should return null after delete (flag=1)"
    );
}

#[test_log::test]
fn arena_slab() {
    if skip_if_no_arena() {
        return;
    }

    let bpf = load_and_trigger(crate::ARENA_SLAB_TEST, "arena_slab_test");
    let results = Array::<_, u64>::try_from(bpf.map("RESULTS").unwrap()).unwrap();

    assert_eq!(
        results.get(&0, 0).unwrap(),
        4,
        "total_allocated after 4 allocs"
    );
    assert_eq!(
        results.get(&1, 0).unwrap(),
        2,
        "free_count after freeing 2 slots"
    );
    assert_eq!(
        results.get(&2, 0).unwrap(),
        4,
        "total_allocated after reuse (still 4, not 6)"
    );
    assert_eq!(
        results.get(&3, 0).unwrap(),
        0,
        "free_count after reuse (all slots reused)"
    );
    assert_eq!(
        results.get(&4, 0).unwrap(),
        1,
        "all slots should be non-null (flag=1)"
    );
}

#[test_log::test]
fn arena_btree() {
    if skip_if_no_arena() {
        return;
    }

    let bpf = load_and_trigger(crate::ARENA_BTREE_TEST, "arena_btree_test");
    let results = Array::<_, u64>::try_from(bpf.map("RESULTS").unwrap()).unwrap();

    assert_eq!(
        results.get(&0, 0).unwrap(),
        10,
        "count after 10 inserts"
    );
    assert_eq!(
        results.get(&1, 0).unwrap(),
        50,
        "get(5) should be 50"
    );
    assert_eq!(
        results.get(&2, 0).unwrap(),
        100,
        "get(10) should be 100"
    );
    assert_eq!(
        results.get(&3, 0).unwrap(),
        1,
        "get(999) should return null (flag=1)"
    );
    assert_eq!(
        results.get(&4, 0).unwrap(),
        0,
        "delete(3) should succeed (0)"
    );
    assert_eq!(
        results.get(&5, 0).unwrap(),
        9,
        "count after delete should be 9"
    );
    assert_eq!(
        results.get(&6, 0).unwrap(),
        1,
        "get(3) should return null after delete (flag=1)"
    );
}

// ── Cross-boundary tests: BPF writes, userspace reads arena mmap ──────

#[test_log::test]
fn arena_cross_linked_list() {
    if skip_if_no_arena() {
        return;
    }

    let bpf = load_and_trigger(crate::ARENA_CROSS_TEST, "arena_cross_test");
    let results = Array::<_, u64>::try_from(bpf.map("RESULTS").unwrap()).unwrap();

    let status = results.get(&4, 0).unwrap();
    assert_eq!(status, 0, "BPF cross-test failed with status {status}");

    let arena = Arena::try_from(bpf.map("ARENA").unwrap()).unwrap();
    let arena_base = arena.as_ptr() as *mut u8;
    assert!(!arena_base.is_null(), "arena mmap should not be null");

    let list_head_offset = results.get(&0, 0).unwrap() as usize;

    unsafe {
        let head = &*(arena_base.add(list_head_offset).cast::<ArenaListHead>());
        assert_eq!(head.count, 5, "linked list should have 5 nodes");

        // Traverse: Counter(1) → Label("hello") → Counter(2) → Label("arena") → Counter(3)
        let mut current = head.head;
        let mut idx = 0u32;

        // Node 0: Counter(1)
        assert!(!current.is_null(), "node 0 should exist");
        let hdr = &*current.resolve(arena_base);
        assert_eq!(hdr.tag, TAG_COUNTER);
        let cn = &*(current.resolve(arena_base) as *const CounterNode);
        assert_eq!(cn.value, 1, "first counter should be 1");
        current = hdr.next;
        idx += 1;

        // Node 1: Label("hello")
        assert!(!current.is_null(), "node 1 should exist");
        let hdr = &*current.resolve(arena_base);
        assert_eq!(hdr.tag, TAG_LABEL);
        let ln = &*(current.resolve(arena_base) as *const LabelNode);
        let label = &ln.label[..ln.len as usize];
        assert_eq!(label, b"hello", "first label should be 'hello'");
        current = hdr.next;
        idx += 1;

        // Node 2: Counter(2)
        assert!(!current.is_null(), "node 2 should exist");
        let hdr = &*current.resolve(arena_base);
        assert_eq!(hdr.tag, TAG_COUNTER);
        let cn = &*(current.resolve(arena_base) as *const CounterNode);
        assert_eq!(cn.value, 2, "second counter should be 2");
        current = hdr.next;
        idx += 1;

        // Node 3: Label("arena")
        assert!(!current.is_null(), "node 3 should exist");
        let hdr = &*current.resolve(arena_base);
        assert_eq!(hdr.tag, TAG_LABEL);
        let ln = &*(current.resolve(arena_base) as *const LabelNode);
        let label = &ln.label[..ln.len as usize];
        assert_eq!(label, b"arena", "second label should be 'arena'");
        current = hdr.next;
        idx += 1;

        // Node 4: Counter(3)
        assert!(!current.is_null(), "node 4 should exist");
        let hdr = &*current.resolve(arena_base);
        assert_eq!(hdr.tag, TAG_COUNTER);
        let cn = &*(current.resolve(arena_base) as *const CounterNode);
        assert_eq!(cn.value, 3, "third counter should be 3");
        current = hdr.next;
        idx += 1;

        // End of list
        assert!(current.is_null(), "list should end after 5 nodes");
        assert_eq!(idx, 5);
    }
}

#[test_log::test]
fn arena_cross_hashmap() {
    if skip_if_no_arena() {
        return;
    }

    let bpf = load_and_trigger(crate::ARENA_CROSS_TEST, "arena_cross_test");
    let results = Array::<_, u64>::try_from(bpf.map("RESULTS").unwrap()).unwrap();

    let status = results.get(&4, 0).unwrap();
    assert_eq!(status, 0, "BPF cross-test failed with status {status}");

    let arena = Arena::try_from(bpf.map("ARENA").unwrap()).unwrap();
    let arena_base = arena.as_ptr() as *mut u8;
    assert!(!arena_base.is_null());

    let hashmap_offset = results.get(&1, 0).unwrap() as usize;

    unsafe {
        let map_ptr = arena_base.add(hashmap_offset).cast::<ArenaHashMap>();

        // Verify count
        assert_eq!((*map_ptr).count, 5, "hash map should have 5 entries");

        // Verify each entry via arena_hash_get
        let expected: &[(u64, u64)] = &[
            (1001, 10),
            (1002, 20),
            (1003, 30),
            (1004, 40),
            (1005, 50),
        ];
        for &(key, expected_val) in expected {
            let val_ptr = arena_hash_get(map_ptr, key, arena_base);
            assert!(
                !val_ptr.is_null(),
                "hash map should contain key {key}"
            );
            assert_eq!(
                *val_ptr, expected_val,
                "hash map key {key} should have value {expected_val}"
            );
        }

        // Verify missing key
        let missing = arena_hash_get(map_ptr, 9999, arena_base);
        assert!(missing.is_null(), "key 9999 should not exist");

        // Verify iteration via arena_hash_for_each
        let mut entries = Vec::new();
        arena_hash_for_each(map_ptr, arena_base, |k, v| {
            entries.push((k, v));
        });
        entries.sort();
        assert_eq!(
            entries,
            vec![(1001, 10), (1002, 20), (1003, 30), (1004, 40), (1005, 50)],
            "arena_hash_for_each should yield all 5 entries"
        );
    }
}

#[test_log::test]
fn arena_cross_btree() {
    if skip_if_no_arena() {
        return;
    }

    let bpf = load_and_trigger(crate::ARENA_CROSS_TEST, "arena_cross_test");
    let results = Array::<_, u64>::try_from(bpf.map("RESULTS").unwrap()).unwrap();

    let status = results.get(&4, 0).unwrap();
    assert_eq!(status, 0, "BPF cross-test failed with status {status}");

    let arena = Arena::try_from(bpf.map("ARENA").unwrap()).unwrap();
    let arena_base = arena.as_ptr() as *mut u8;
    assert!(!arena_base.is_null());

    let btree_offset = results.get(&2, 0).unwrap() as usize;

    unsafe {
        let tree_ptr = arena_base.add(btree_offset).cast::<ArenaBTreeMap>();

        // Verify count
        assert_eq!((*tree_ptr).count, 10, "btree should have 10 entries");

        // Verify each entry via arena_btree_get
        for key in 1..=10u64 {
            let val_ptr = arena_btree_get(tree_ptr, key, arena_base);
            assert!(
                !val_ptr.is_null(),
                "btree should contain key {key}"
            );
            assert_eq!(
                *val_ptr,
                key * 10,
                "btree key {key} should have value {}",
                key * 10
            );
        }

        // Verify missing key
        let missing = arena_btree_get(tree_ptr, 999, arena_base);
        assert!(missing.is_null(), "key 999 should not exist");

        // Verify sorted iteration via arena_btree_for_each
        let mut keys = Vec::new();
        arena_btree_for_each(tree_ptr, arena_base, |k, v| {
            keys.push((k, v));
        });
        let expected: Vec<(u64, u64)> = (1..=10).map(|k| (k, k * 10)).collect();
        assert_eq!(
            keys, expected,
            "arena_btree_for_each should yield entries in sorted order"
        );
    }
}
