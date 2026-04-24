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
    kfuncs::{arena_alloc_pages, cast_kern, NUMA_NO_NODE},
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

// ── Globals ───────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
static mut INITIALIZED: u64 = 0;

/// Arena memory base pointer, stored via volatile after arena_alloc_pages.
#[unsafe(no_mangle)]
static mut ARENA_BASE: *mut u8 = core::ptr::null_mut();

/// Current allocation offset within arena memory.
#[unsafe(no_mangle)]
static mut ARENA_OFF: u64 = 0;

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

/// Simple bump allocator using volatile globals.
/// Returns a cast_kern'd arena pointer or null.
#[inline(always)]
unsafe fn arena_bump(size: u64, align: u64) -> *mut c_void {
    let base = cast_kern(core::ptr::read_volatile(&raw const ARENA_BASE));
    let off = core::ptr::read_volatile(&raw const ARENA_OFF);
    let aligned = (off + align - 1) & !(align - 1);
    let new_off = aligned + size;
    core::ptr::write_volatile(&raw mut ARENA_OFF, new_off);
    base.add(aligned as usize) as *mut c_void
}

// ── B-tree test ────────────────────────────────────────────────────────

const ALLOC_PAGES: u32 = 16;
const BTREE_REGION_SIZE: u64 = 32 * 1024;

/// Initialize arena: allocate pages, store base in volatile global.
/// This is a separate subprogram so the ARENA map_ptr doesn't leak
/// into the caller's register state.
#[inline(never)]
unsafe fn init_arena() -> i32 {
    let mem = arena_alloc_pages(ARENA.as_ptr(), core::ptr::null_mut(), ALLOC_PAGES, NUMA_NO_NODE, 0);
    if mem.is_null() {
        return -1;
    }
    core::ptr::write_volatile(&raw mut ARENA_BASE, mem as *mut u8);
    core::ptr::write_volatile(&raw mut ARENA_OFF, 0);
    0
}

#[inline(always)]
unsafe fn run_btree_test() -> i64 {
    // Allocate B-tree map header
    let tree_raw = arena_bump(size_of::<ArenaBTreeMap>() as u64, align_of::<ArenaBTreeMap>() as u64);
    if tree_raw.is_null() {
        return -2;
    }
    let tree = tree_raw.cast::<ArenaBTreeMap>();

    // Allocate ArenaBumpState for btree node allocation
    let bump_state_raw = arena_bump(
        size_of::<ArenaBumpState>() as u64,
        align_of::<ArenaBumpState>() as u64,
    );
    if bump_state_raw.is_null() {
        return -3;
    }
    let bump_state = bump_state_raw.cast::<ArenaBumpState>();

    // Allocate a region for btree nodes
    let region_raw = arena_bump(BTREE_REGION_SIZE, 8);
    if region_raw.is_null() {
        return -4;
    }

    let arena_base = cast_kern(core::ptr::read_volatile(&raw const ARENA_BASE));
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
    arena_btree_init(tree);

    // Insert 10 entries: keys 1..=10, values key*10
    let mut i: u64 = 1;
    while i <= 10 {
        let ret = arena_btree_insert(tree, bump_state, i, i * 10, arena_base);
        if ret < 0 {
            return -5;
        }
        i += 1;
    }

    // RESULTS[0] = count (expect 10)
    result_set(0, (*tree).count);

    // RESULTS[1] = get(5) value (expect 50)
    let val = arena_btree_get(tree, 5, arena_base);
    if !val.is_null() {
        result_set(1, *val);
    }

    // RESULTS[2] = get(10) value (expect 100)
    let val = arena_btree_get(tree, 10, arena_base);
    if !val.is_null() {
        result_set(2, *val);
    }

    // RESULTS[3] = 1 if get(999) returns null, else 0
    let val = arena_btree_get(tree, 999, arena_base);
    result_set(3, if val.is_null() { 1 } else { 0 });

    // RESULTS[4] = delete(3) return code (expect 0)
    let ret = arena_btree_delete(tree, 3, arena_base);
    result_set(4, ret as u64);

    // RESULTS[5] = count after delete (expect 9)
    result_set(5, (*tree).count);

    // RESULTS[6] = 1 if get(3) returns null after delete, else 0
    let val = arena_btree_get(tree, 3, arena_base);
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

    let ret = unsafe { init_arena() };
    if ret != 0 {
        return 0;
    }

    let _ret = unsafe { run_btree_test() };

    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
