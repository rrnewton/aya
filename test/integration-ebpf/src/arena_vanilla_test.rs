//! Vanilla Rust data structures in BPF arena memory.
//!
//! Proves that standard `alloc` crate types (Vec, BTreeMap) work
//! unmodified when backed by arena memory via ArenaGlobalAlloc.
//!
//! Results are reported via an Array map:
//!   [0] = Vec length after pushes
//!   [1] = Vec element at index 2
//!   [2] = Vec sum of all elements
//!   [3] = BTreeMap length after inserts
//!   [4] = BTreeMap get(42) value
//!   [5] = BTreeMap contains_key(999) (0 or 1)
//!   [6] = allocator bytes used
//!   [7] = 0xCAFE (success sentinel)

#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use aya_ebpf::{
    bindings::bpf_map_type::BPF_MAP_TYPE_ARENA,
    kfuncs::ArenaGlobalAlloc,
    macros::{btf_map, map},
    maps::Array,
};
use core::ffi::c_void;

#[cfg(not(test))]
extern crate ebpf_panic;

// ── Global arena allocator ───────────────────────────────────────────

#[global_allocator]
static ALLOC: ArenaGlobalAlloc = ArenaGlobalAlloc::new();

// ── Arena map definition ─────────────────────────────────────────────

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

// ── Results map ──────────────────────────────────────────────────────

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

// ── Test: vanilla Vec ────────────────────────────────────────────────

#[inline(never)]
unsafe fn test_vec() {
    let mut v: Vec<u64> = Vec::with_capacity(16);

    v.push(10);
    v.push(20);
    v.push(30);
    v.push(40);
    v.push(50);

    result_set(0, v.len() as u64);
    result_set(1, v[2]);

    let mut sum = 0u64;
    let mut i = 0usize;
    while i < v.len() {
        sum += v[i];
        i += 1;
    }
    result_set(2, sum);
}

// ── Test: vanilla BTreeMap ───────────────────────────────────────────

#[inline(never)]
unsafe fn test_btreemap() {
    let mut map: BTreeMap<u64, u64> = BTreeMap::new();

    map.insert(10, 100);
    map.insert(42, 420);
    map.insert(99, 990);
    map.insert(1, 10);
    map.insert(55, 550);

    result_set(3, map.len() as u64);

    if let Some(&val) = map.get(&42) {
        result_set(4, val);
    }

    result_set(5, if map.contains_key(&999) { 1 } else { 0 });
}

// ── BPF entry point ──────────────────────────────────────────────────

#[unsafe(no_mangle)]
#[unsafe(link_section = "syscall")]
fn arena_vanilla_test(_ctx: *mut c_void) -> i32 {
    let ret = unsafe { ALLOC.init(ARENA.as_ptr(), 64) };
    if ret != 0 {
        return ret;
    }

    unsafe { test_vec() };
    unsafe { test_btreemap() };

    result_set(6, ALLOC.used());
    result_set(7, 0xCAFE);

    0
}
