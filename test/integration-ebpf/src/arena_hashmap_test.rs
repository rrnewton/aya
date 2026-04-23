//! Arena hash map integration test — BPF side.
//!
//! Tests ArenaHashMap operations (insert, get, delete) using arena memory
//! with a bump allocator. Results are reported via an Array map.
//!
//! Requires kernel 6.9+ with BPF_MAP_TYPE_ARENA support.

#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

use aya_arena_common::{
    arena_hash_delete, arena_hash_get, arena_hash_init, arena_hash_insert, ArenaHashEntry,
    ArenaHashMap,
};
use aya_ebpf::{
    bindings::bpf_map_type::BPF_MAP_TYPE_ARENA,
    kfuncs::bump::{bump_alloc, bump_init, BumpAllocator},
    macros::{btf_map, lsm, map},
    maps::Array,
    programs::LsmContext,
};
use core::ffi::c_void;
#[cfg(not(test))]
extern crate ebpf_panic;

// ── Arena map definition ───────────────────────────────────────────────

const BPF_F_MMAPABLE: usize = 1024;
const ARENA_PAGES: usize = 256;

#[repr(C)]
struct ArenaMapDef {
    r#type: *const [i32; BPF_MAP_TYPE_ARENA as usize],
    key: *const u32,
    value: *const u32,
    max_entries: *const [i32; ARENA_PAGES],
    map_flags: *const [i32; BPF_F_MMAPABLE],
}

unsafe impl Sync for ArenaMapDef {}

impl ArenaMapDef {
    const fn new() -> Self {
        Self {
            r#type: core::ptr::null(),
            key: core::ptr::null(),
            value: core::ptr::null(),
            max_entries: core::ptr::null(),
            map_flags: core::ptr::null(),
        }
    }

    #[inline(always)]
    const fn as_ptr(&self) -> *mut c_void {
        core::ptr::from_ref(self).cast_mut().cast()
    }
}

#[btf_map]
static ARENA: ArenaMapDef = ArenaMapDef::new();

// ── Bump allocator state ───────────────────────────────────────────────

#[unsafe(no_mangle)]
static mut BUMP: BumpAllocator = BumpAllocator::new();

#[unsafe(no_mangle)]
static mut INITIALIZED: u64 = 0;

// ── Results map ────────────────────────────────────────────────────────

#[map]
static RESULTS: Array<u64> = Array::<u64>::with_max_entries(8, 0);

#[inline(always)]
fn result_set(index: u32, value: u64) {
    if let Some(ptr) = RESULTS.get_ptr_mut(index) {
        if let Some(dst) = unsafe { ptr.as_mut() } {
            *dst = value;
        }
    }
}

// ── Hash map test ──────────────────────────────────────────────────────

const BUMP_INIT_PAGES: u32 = 8;
const HASH_CAPACITY: u32 = 8;

#[inline(always)]
unsafe fn run_hashmap_test(arena_ptr: *mut c_void) -> i64 {
    // Initialize bump allocator
    let ret = unsafe { bump_init(&raw mut BUMP, arena_ptr, BUMP_INIT_PAGES) };
    if ret != 0 {
        return -1;
    }

    // Allocate hash map header
    let header_raw = unsafe {
        bump_alloc(
            &raw mut BUMP,
            arena_ptr,
            size_of::<ArenaHashMap>() as u64,
            align_of::<ArenaHashMap>() as u64,
        )
    };
    if header_raw.is_null() {
        return -2;
    }
    let header = header_raw.cast::<ArenaHashMap>();

    // Allocate entry slots
    let entries_raw = unsafe {
        bump_alloc(
            &raw mut BUMP,
            arena_ptr,
            (size_of::<ArenaHashEntry>() * HASH_CAPACITY as usize) as u64,
            align_of::<ArenaHashEntry>() as u64,
        )
    };
    if entries_raw.is_null() {
        return -3;
    }
    let entries = entries_raw.cast::<ArenaHashEntry>();

    let arena_base = arena_ptr as *mut u8;

    // Initialize hash map
    let ret = unsafe { arena_hash_init(header, entries, HASH_CAPACITY, arena_base) };
    if ret != 0 {
        return -4;
    }

    // Insert 5 entries
    unsafe {
        arena_hash_insert(header, 1001, 10, arena_base);
        arena_hash_insert(header, 1002, 20, arena_base);
        arena_hash_insert(header, 1003, 30, arena_base);
        arena_hash_insert(header, 1004, 40, arena_base);
        arena_hash_insert(header, 1005, 50, arena_base);
    }

    // RESULTS[0] = count (expect 5)
    result_set(0, unsafe { (*header).count } as u64);

    // RESULTS[1] = get(1001) value (expect 10)
    let val = unsafe { arena_hash_get(header, 1001, arena_base) };
    if !val.is_null() {
        result_set(1, unsafe { *val });
    }

    // RESULTS[2] = get(1005) value (expect 50)
    let val = unsafe { arena_hash_get(header, 1005, arena_base) };
    if !val.is_null() {
        result_set(2, unsafe { *val });
    }

    // RESULTS[3] = 1 if get(9999) returns null, else 0
    let val = unsafe { arena_hash_get(header, 9999, arena_base) };
    result_set(3, if val.is_null() { 1 } else { 0 });

    // RESULTS[4] = delete(1003) return code (expect 0)
    let ret = unsafe { arena_hash_delete(header, 1003, arena_base) };
    result_set(4, ret as u64);

    // RESULTS[5] = count after delete (expect 4)
    result_set(5, unsafe { (*header).count } as u64);

    // RESULTS[6] = 1 if get(1003) returns null after delete, else 0
    let val = unsafe { arena_hash_get(header, 1003, arena_base) };
    result_set(6, if val.is_null() { 1 } else { 0 });

    0
}

// ── BPF program entry point ────────────────────────────────────────────

#[lsm(hook = "socket_create", sleepable)]
fn arena_hashmap_test(_ctx: LsmContext) -> i32 {
    let initialized = unsafe { core::ptr::read_volatile(&raw const INITIALIZED) };
    if initialized != 0 {
        return 0;
    }

    let arena_ptr = ARENA.as_ptr();
    let _ret = unsafe { run_hashmap_test(arena_ptr) };

    // Always allow socket creation, even if test failed.
    // Results are communicated via the RESULTS array map.
    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
