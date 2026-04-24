//! Arena slab allocator integration test — BPF side.
//!
//! Tests ArenaSlabState operations (init, alloc, free, reuse) using arena
//! memory with a bump allocator. Results are reported via an Array map.
//!
//! Requires kernel 6.9+ with BPF_MAP_TYPE_ARENA support.

#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

use aya_arena_common::{arena_slab_alloc, arena_slab_free, arena_slab_init, arena_slab_stats, ArenaSlabState};
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

// ── Slab allocator test ────────────────────────────────────────────────

const ALLOC_PAGES: u32 = 8;
const SLAB_REGION_SIZE: u64 = 4096;
const SLOT_SIZE: u32 = 64;

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
unsafe fn run_slab_test() -> i64 {
    // Allocate slab state header
    let slab_raw = arena_bump(
        size_of::<ArenaSlabState>() as u64,
        align_of::<ArenaSlabState>() as u64,
    );
    if slab_raw.is_null() {
        return -2;
    }
    let slab = slab_raw.cast::<ArenaSlabState>();

    // Allocate a region for slab slots
    let region_raw = arena_bump(SLAB_REGION_SIZE, 8);
    if region_raw.is_null() {
        return -3;
    }

    // The slab's bump allocator works with offsets relative to arena_base.
    // We use the region start as the arena_base for slab operations, so the
    // slab's internal bump allocator covers exactly our allocated region.
    let region_base = region_raw as *mut u8;

    // Initialize slab: capacity = SLAB_REGION_SIZE, slot_size = 64
    let ret = arena_slab_init(slab, SLAB_REGION_SIZE, SLOT_SIZE);
    if ret != 0 {
        return -4;
    }

    // Allocate 4 slots
    let mut all_nonnull: u64 = 1;
    let slot0 = arena_slab_alloc(slab, region_base);
    if slot0.is_null() {
        all_nonnull = 0;
    }
    let slot1 = arena_slab_alloc(slab, region_base);
    if slot1.is_null() {
        all_nonnull = 0;
    }
    let slot2 = arena_slab_alloc(slab, region_base);
    if slot2.is_null() {
        all_nonnull = 0;
    }
    let slot3 = arena_slab_alloc(slab, region_base);
    if slot3.is_null() {
        all_nonnull = 0;
    }

    // RESULTS[0] = total_allocated after initial 4 allocs (expect 4)
    let stats = arena_slab_stats(slab);
    result_set(0, stats.total_allocated as u64);

    // Free 2 slots (slot1 and slot2)
    arena_slab_free(slab, slot1, region_base);
    arena_slab_free(slab, slot2, region_base);

    // RESULTS[1] = free_count after freeing 2 (expect 2)
    let stats = arena_slab_stats(slab);
    result_set(1, stats.free_count as u64);

    // Allocate 2 more (should reuse from free list)
    let slot4 = arena_slab_alloc(slab, region_base);
    if slot4.is_null() {
        all_nonnull = 0;
    }
    let slot5 = arena_slab_alloc(slab, region_base);
    if slot5.is_null() {
        all_nonnull = 0;
    }

    // RESULTS[2] = total_allocated after reuse allocs (expect 4, NOT 6)
    let stats = arena_slab_stats(slab);
    result_set(2, stats.total_allocated as u64);

    // RESULTS[3] = free_count after reuse allocs (expect 0)
    result_set(3, stats.free_count as u64);

    // RESULTS[4] = 1 if all allocated slots are non-null, else 0
    result_set(4, all_nonnull);

    0
}

// ── BPF program entry point ────────────────────────────────────────────

#[unsafe(no_mangle)]
#[unsafe(link_section = "syscall")]
fn arena_slab_test(_ctx: *mut c_void) -> i32 {
    let initialized = unsafe { core::ptr::read_volatile(&raw const INITIALIZED) };
    if initialized != 0 {
        return 0;
    }

    let ret = unsafe { init_arena() };
    if ret != 0 {
        return 0;
    }

    let _ret = unsafe { run_slab_test() };

    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
