//! Arena cross-boundary integration test — BPF side.
//!
//! Builds three data structures in shared arena memory so userspace can
//! read them directly via the mmap'd region:
//!   1. Linked list: Counter(1) -> Label("hello") -> Counter(2) -> Label("arena") -> Counter(3)
//!   2. Hash map: 5 entries (1001->10, 1002->20, 1003->30, 1004->40, 1005->50)
//!   3. B-tree: 10 entries, keys 1..=10, values key*10
//!
//! RESULTS layout (offsets from arena base):
//!   [0] = ArenaListHead offset
//!   [1] = ArenaHashMap header offset
//!   [2] = ArenaBTreeMap header offset
//!   [3] = ArenaBumpState offset (for btree node resolution)
//!   [4] = status (0 = success, negative = error stage)
//!
//! Requires kernel 6.9+ with BPF_MAP_TYPE_ARENA support.

#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

use aya_arena_common::{
    ArenaBTreeMap, ArenaBumpState, ArenaHashEntry, ArenaHashMap, ArenaListHead, ArenaNodeHeader,
    ArenaPtr, CounterNode, LabelNode, LABEL_MAX_LEN, TAG_COUNTER, TAG_LABEL, arena_btree_init,
    arena_btree_insert, arena_hash_init, arena_hash_insert,
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

// ── Globals ──────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
static mut INITIALIZED: u64 = 0;

/// Arena memory base pointer, stored via volatile after arena_alloc_pages.
#[unsafe(no_mangle)]
static mut ARENA_BASE: *mut u8 = core::ptr::null_mut();

/// Current allocation offset within arena memory.
#[unsafe(no_mangle)]
static mut ARENA_OFF: u64 = 0;

// ── Results map ───────────────────────────────────────────────────────

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

// ── Constants ─────────────────────────────────────────────────────────

const ALLOC_PAGES: u32 = 32;
const HASH_CAPACITY: u32 = 8;
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

// ── Build linked list ─────────────────────────────────────────────────

#[inline(always)]
unsafe fn build_linked_list() -> i64 {
    let arena_base = cast_kern(core::ptr::read_volatile(&raw const ARENA_BASE));

    let head_raw = arena_bump(
        size_of::<ArenaListHead>() as u64,
        align_of::<ArenaListHead>() as u64,
    );
    if head_raw.is_null() {
        return -10;
    }
    let head = head_raw.cast::<ArenaListHead>();
    let head_offset = head_raw as u64 - arena_base as u64;
    result_set(0, head_offset);

    let mut list_head_ptr: ArenaPtr<ArenaNodeHeader> = ArenaPtr::null();
    let mut count: u64 = 0;

    macro_rules! push_counter {
        ($val:expr) => {{
            let raw = arena_bump(
                size_of::<CounterNode>() as u64,
                align_of::<CounterNode>() as u64,
            );
            if raw.is_null() {
                return -11;
            }
            let node = raw.cast::<CounterNode>();
            core::ptr::write_volatile(
                node,
                CounterNode {
                    header: ArenaNodeHeader {
                        tag: TAG_COUNTER,
                        size: size_of::<CounterNode>() as u32,
                        next: list_head_ptr,
                    },
                    value: $val,
                },
            );
            list_head_ptr = ArenaPtr::from_raw(node.cast(), arena_base as *mut u8);
            count += 1;
        }};
    }

    macro_rules! push_label {
        ($bytes:expr) => {{
            let raw = arena_bump(
                size_of::<LabelNode>() as u64,
                align_of::<LabelNode>() as u64,
            );
            if raw.is_null() {
                return -12;
            }
            let node = raw.cast::<LabelNode>();
            let s: &[u8] = $bytes;
            let mut label = [0u8; LABEL_MAX_LEN];
            let copy_len = if s.len() < LABEL_MAX_LEN {
                s.len()
            } else {
                LABEL_MAX_LEN - 1
            };
            let mut i = 0;
            while i < copy_len {
                label[i] = s[i];
                i += 1;
            }
            core::ptr::write_volatile(
                node,
                LabelNode {
                    header: ArenaNodeHeader {
                        tag: TAG_LABEL,
                        size: size_of::<LabelNode>() as u32,
                        next: list_head_ptr,
                    },
                    label,
                    len: copy_len as u32,
                    _pad: 0,
                },
            );
            list_head_ptr = ArenaPtr::from_raw(node.cast(), arena_base as *mut u8);
            count += 1;
        }};
    }

    // Build in reverse: Counter(3), Label("arena"), Counter(2), Label("hello"), Counter(1)
    push_counter!(3);
    push_label!(b"arena");
    push_counter!(2);
    push_label!(b"hello");
    push_counter!(1);

    core::ptr::write_volatile(
        head,
        ArenaListHead {
            head: list_head_ptr,
            count,
        },
    );

    0
}

// ── Build hash map ────────────────────────────────────────────────────

#[inline(always)]
unsafe fn build_hashmap() -> i64 {
    let arena_base = cast_kern(core::ptr::read_volatile(&raw const ARENA_BASE));

    let header_raw = arena_bump(
        size_of::<ArenaHashMap>() as u64,
        align_of::<ArenaHashMap>() as u64,
    );
    if header_raw.is_null() {
        return -20;
    }
    let header = header_raw.cast::<ArenaHashMap>();
    let header_offset = header_raw as u64 - arena_base as u64;
    result_set(1, header_offset);

    let entries_raw = arena_bump(
        (size_of::<ArenaHashEntry>() * HASH_CAPACITY as usize) as u64,
        align_of::<ArenaHashEntry>() as u64,
    );
    if entries_raw.is_null() {
        return -21;
    }
    let entries = entries_raw.cast::<ArenaHashEntry>();

    let ret = arena_hash_init(header, entries, HASH_CAPACITY, arena_base as *mut u8);
    if ret != 0 {
        return -22;
    }

    arena_hash_insert(header, 1001, 10, arena_base as *mut u8);
    arena_hash_insert(header, 1002, 20, arena_base as *mut u8);
    arena_hash_insert(header, 1003, 30, arena_base as *mut u8);
    arena_hash_insert(header, 1004, 40, arena_base as *mut u8);
    arena_hash_insert(header, 1005, 50, arena_base as *mut u8);

    0
}

// ── Build B-tree ──────────────────────────────────────────────────────

#[inline(always)]
unsafe fn build_btree() -> i64 {
    let arena_base = cast_kern(core::ptr::read_volatile(&raw const ARENA_BASE));

    let tree_raw = arena_bump(
        size_of::<ArenaBTreeMap>() as u64,
        align_of::<ArenaBTreeMap>() as u64,
    );
    if tree_raw.is_null() {
        return -30;
    }
    let tree = tree_raw.cast::<ArenaBTreeMap>();
    let tree_offset = tree_raw as u64 - arena_base as u64;
    result_set(2, tree_offset);

    let bump_state_raw = arena_bump(
        size_of::<ArenaBumpState>() as u64,
        align_of::<ArenaBumpState>() as u64,
    );
    if bump_state_raw.is_null() {
        return -31;
    }
    let bump_state = bump_state_raw.cast::<ArenaBumpState>();
    let bump_state_offset = bump_state_raw as u64 - arena_base as u64;
    result_set(3, bump_state_offset);

    let region_raw = arena_bump(BTREE_REGION_SIZE, 8);
    if region_raw.is_null() {
        return -32;
    }

    let region_offset = region_raw as u64 - arena_base as u64;
    core::ptr::write_volatile(
        bump_state,
        ArenaBumpState {
            watermark: region_offset,
            capacity: region_offset + BTREE_REGION_SIZE,
        },
    );

    arena_btree_init(tree);

    let mut i: u64 = 1;
    while i <= 10 {
        let ret = arena_btree_insert(tree, bump_state, i, i * 10, arena_base as *mut u8);
        if ret < 0 {
            return -33;
        }
        i += 1;
    }

    0
}

// ── Entry point ───────────────────────────────────────────────────────

#[inline(always)]
unsafe fn run_cross_test() -> i64 {
    let ret = build_linked_list();
    if ret != 0 {
        result_set(4, ret as u64);
        return ret;
    }

    let ret = build_hashmap();
    if ret != 0 {
        result_set(4, ret as u64);
        return ret;
    }

    let ret = build_btree();
    if ret != 0 {
        result_set(4, ret as u64);
        return ret;
    }

    result_set(4, 0);
    0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = "syscall")]
fn arena_cross_test(_ctx: *mut c_void) -> i32 {
    let initialized = unsafe { core::ptr::read_volatile(&raw const INITIALIZED) };
    if initialized != 0 {
        return 0;
    }

    let ret = unsafe { init_arena() };
    if ret != 0 {
        return 0;
    }

    let _ret = unsafe { run_cross_test() };

    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
