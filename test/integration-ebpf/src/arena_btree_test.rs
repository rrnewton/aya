//! Arena B-tree map integration test — BPF side.
//!
//! Tests ArenaBTreeMap operations (insert, get, delete) using arena memory
//! with a bump allocator. Results are reported via an Array map.
//!
//! Requires kernel 6.9+ with BPF_MAP_TYPE_ARENA support.

#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

use aya_arena_common::{
    arena_btree_delete, arena_btree_get, arena_btree_init, arena_btree_insert, ArenaBTreeMap,
    ArenaBumpState,
};
use aya_ebpf::{
    bindings::bpf_map_type::BPF_MAP_TYPE_ARENA,
    kfuncs::bump::{bump_alloc, bump_init, BumpAllocator},
    macros::{btf_map, map},
    maps::Array,
    
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

#[unsafe(no_mangle)]
static mut ARENA_MAP_PTR: *mut c_void = core::ptr::null_mut();

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

// ── B-tree test ────────────────────────────────────────────────────────

const BUMP_INIT_PAGES: u32 = 16;
const BTREE_REGION_SIZE: u64 = 32 * 1024;

#[inline(always)]
unsafe fn run_btree_test(arena_ptr: *mut c_void) -> i64 {
    // Initialize bump allocator (16 pages for btree nodes)
    let ret = unsafe { bump_init(&raw mut BUMP, arena_ptr, BUMP_INIT_PAGES) };
    if ret != 0 {
        return -1;
    }

    // Allocate B-tree map header
    let tree_raw = unsafe {
        bump_alloc(
            &raw mut BUMP,
            arena_ptr,
            size_of::<ArenaBTreeMap>() as u64,
            align_of::<ArenaBTreeMap>() as u64,
        )
    };
    if tree_raw.is_null() {
        return -2;
    }
    let tree = tree_raw.cast::<ArenaBTreeMap>();

    // Allocate ArenaBumpState for btree node allocation
    let bump_state_raw = unsafe {
        bump_alloc(
            &raw mut BUMP,
            arena_ptr,
            size_of::<ArenaBumpState>() as u64,
            align_of::<ArenaBumpState>() as u64,
        )
    };
    if bump_state_raw.is_null() {
        return -3;
    }
    let bump_state = bump_state_raw.cast::<ArenaBumpState>();

    // Allocate a region for btree nodes
    let region_raw = unsafe {
        bump_alloc(
            &raw mut BUMP,
            arena_ptr,
            BTREE_REGION_SIZE,
            8,
        )
    };
    if region_raw.is_null() {
        return -4;
    }

    let arena_base = arena_ptr as *mut u8;
    let region = region_raw as *mut u8;

    // Initialize the ArenaBumpState for btree node allocation.
    // The bump state tracks offsets relative to arena_base, so watermark
    // starts at the region's offset and capacity is watermark + region size.
    let region_offset = region as u64 - arena_base as u64;
    core::ptr::write_volatile(
        bump_state,
        ArenaBumpState {
            watermark: region_offset,
            capacity: region_offset + BTREE_REGION_SIZE,
        },
    );

    // Initialize the btree
    unsafe { arena_btree_init(tree) };

    // Insert 10 entries: keys 1..=10, values key*10
    let mut i: u64 = 1;
    while i <= 10 {
        let ret = unsafe { arena_btree_insert(tree, bump_state, i, i * 10, arena_base) };
        if ret < 0 {
            return -5;
        }
        i += 1;
    }

    // RESULTS[0] = count (expect 10)
    result_set(0, unsafe { (*tree).count });

    // RESULTS[1] = get(5) value (expect 50)
    let val = unsafe { arena_btree_get(tree, 5, arena_base) };
    if !val.is_null() {
        result_set(1, unsafe { *val });
    }

    // RESULTS[2] = get(10) value (expect 100)
    let val = unsafe { arena_btree_get(tree, 10, arena_base) };
    if !val.is_null() {
        result_set(2, unsafe { *val });
    }

    // RESULTS[3] = 1 if get(999) returns null, else 0
    let val = unsafe { arena_btree_get(tree, 999, arena_base) };
    result_set(3, if val.is_null() { 1 } else { 0 });

    // RESULTS[4] = delete(3) return code (expect 0)
    let ret = unsafe { arena_btree_delete(tree, 3, arena_base) };
    result_set(4, ret as u64);

    // RESULTS[5] = count after delete (expect 9)
    result_set(5, unsafe { (*tree).count });

    // RESULTS[6] = 1 if get(3) returns null after delete, else 0
    let val = unsafe { arena_btree_get(tree, 3, arena_base) };
    result_set(6, if val.is_null() { 1 } else { 0 });

    0
}

// ── BPF program entry point ────────────────────────────────────────────

#[unsafe(no_mangle)]
#[unsafe(link_section = "syscall")]
fn arena_btree_test(_ctx: *mut c_void) -> i32 {
    let initialized = unsafe { core::ptr::read_volatile(&raw const INITIALIZED) };
    if initialized != 0 {
        return 0;
    }

    unsafe { core::ptr::write_volatile(&raw mut ARENA_MAP_PTR, ARENA.as_ptr()) };
    let arena_ptr = unsafe { core::ptr::read_volatile(&raw const ARENA_MAP_PTR) };
    let _ret = unsafe { run_btree_test(arena_ptr) };

    // Always allow socket creation, even if test failed.
    // Results are communicated via the RESULTS array map.
    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
