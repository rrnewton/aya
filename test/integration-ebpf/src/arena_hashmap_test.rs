//! Arena hash map integration test — BPF side.
//!
//! Tests ArenaHashMap operations (insert, get, delete) using arena memory.
//! Results are reported via an Array map.
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

// ── Hash map test ──────────────────────────────────────────────────────

const ALLOC_PAGES: u32 = 8;
const HASH_CAPACITY: u32 = 8;

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
unsafe fn run_hashmap_test() -> i64 {
    // Allocate hash map header
    let header_raw = arena_bump(size_of::<ArenaHashMap>() as u64, align_of::<ArenaHashMap>() as u64);
    if header_raw.is_null() {
        return -2;
    }
    let header = header_raw.cast::<ArenaHashMap>();

    // Allocate entry slots
    let entries_raw = arena_bump(
        (size_of::<ArenaHashEntry>() * HASH_CAPACITY as usize) as u64,
        align_of::<ArenaHashEntry>() as u64,
    );
    if entries_raw.is_null() {
        return -3;
    }
    let entries = entries_raw.cast::<ArenaHashEntry>();

    // Use the arena base as the ArenaPtr base for hash map operations
    let arena_base = cast_kern(core::ptr::read_volatile(&raw const ARENA_BASE));

    // Initialize hash map
    let ret = arena_hash_init(header, entries, HASH_CAPACITY, arena_base);
    if ret != 0 {
        return -4;
    }

    // Insert 5 entries
    arena_hash_insert(header, 1001, 10, arena_base);
    arena_hash_insert(header, 1002, 20, arena_base);
    arena_hash_insert(header, 1003, 30, arena_base);
    arena_hash_insert(header, 1004, 40, arena_base);
    arena_hash_insert(header, 1005, 50, arena_base);

    // RESULTS[0] = count (expect 5)
    result_set(0, (*header).count as u64);

    // RESULTS[1] = get(1001) value (expect 10)
    let val = arena_hash_get(header, 1001, arena_base);
    if !val.is_null() {
        result_set(1, *val);
    }

    // RESULTS[2] = get(1005) value (expect 50)
    let val = arena_hash_get(header, 1005, arena_base);
    if !val.is_null() {
        result_set(2, *val);
    }

    // RESULTS[3] = 1 if get(9999) returns null, else 0
    let val = arena_hash_get(header, 9999, arena_base);
    result_set(3, if val.is_null() { 1 } else { 0 });

    // RESULTS[4] = delete(1003) return code (expect 0)
    let ret = arena_hash_delete(header, 1003, arena_base);
    result_set(4, ret as u64);

    // RESULTS[5] = count after delete (expect 4)
    result_set(5, (*header).count as u64);

    // RESULTS[6] = 1 if get(1003) returns null after delete, else 0
    let val = arena_hash_get(header, 1003, arena_base);
    result_set(6, if val.is_null() { 1 } else { 0 });

    0
}

// ── BPF program entry point ────────────────────────────────────────────

#[unsafe(no_mangle)]
#[unsafe(link_section = "syscall")]
fn arena_hashmap_test(_ctx: *mut c_void) -> i32 {
    let initialized = unsafe { core::ptr::read_volatile(&raw const INITIALIZED) };
    if initialized != 0 {
        return 0;
    }

    let ret = unsafe { init_arena() };
    if ret != 0 {
        return 0;
    }

    let _ret = unsafe { run_hashmap_test() };

    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
