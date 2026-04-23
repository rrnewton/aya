//! Arena linked list PoC — BPF side.
//!
//! This BPF program demonstrates the full arena stack:
//!   arena map → bump allocator → shared types → linked list → cross-boundary access
//!
//! The LSM hook builds a heterogeneous linked list in arena memory using
//! CounterNode and LabelNode types. The list head is stored at offset 0 in the
//! arena so userspace can find it.
//!
//! Requires kernel 6.9+ with BPF_MAP_TYPE_ARENA support.

#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

use aya_arena_common::{
    ArenaListHead, ArenaNodeHeader, ArenaPtr, CounterNode, LabelNode, LABEL_MAX_LEN, TAG_COUNTER,
    TAG_LABEL,
};
use aya_ebpf::{
    bindings::bpf_map_type::BPF_MAP_TYPE_ARENA,
    kfuncs::bump::{bump_alloc, bump_init, BumpAllocator},
    macros::{btf_map, fentry},
    programs::FEntryContext,
};
use core::ffi::c_void;
#[cfg(not(test))]
extern crate ebpf_panic;

// ── Arena map definition ───────────────────────────────────────────────
//
// BTF map definition for BPF_MAP_TYPE_ARENA.
// Fields are encoded as `*const [i32; VALUE]` so the BTF array length
// carries the integer value, which aya-obj's BTF map parser extracts.
//
// Equivalent C:
//   struct { __uint(type, BPF_MAP_TYPE_ARENA); __uint(map_flags, BPF_F_MMAPABLE);
//            __uint(max_entries, 256); } arena SEC(".maps");

/// BPF_F_MMAPABLE flag value.
const BPF_F_MMAPABLE: usize = 1024;

/// Number of arena pages (256 pages = 1 MiB).
const ARENA_PAGES: usize = 256;

#[repr(C)]
struct ArenaMapDef {
    r#type: *const [i32; BPF_MAP_TYPE_ARENA as usize],
    // Arena maps have no key/value in the traditional sense, but BTF
    // map parsing expects these fields to exist (even if zero-sized).
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

/// Flag: has the list been built yet? (0 = no, 1 = yes)
#[unsafe(no_mangle)]
static mut INITIALIZED: u64 = 0;

// ── Arena list construction ────────────────────────────────────────────

/// Number of pages for the bump allocator's initial block.
const BUMP_INIT_PAGES: u32 = 8;

/// Build the linked list in arena memory.
///
/// Creates a heterogeneous list with CounterNode and LabelNode entries:
///   [Counter(1)] → [Label("hello")] → [Counter(2)] → [Label("arena")] → [Counter(3)]
///
/// The ArenaListHead is stored at the very start of the arena (offset 0)
/// so userspace can locate it by reading the first 16 bytes.
#[inline(always)]
unsafe fn build_list(arena_ptr: *mut c_void) -> i64 {
    // Initialize the bump allocator
    let ret = unsafe { bump_init(&raw mut BUMP, arena_ptr, BUMP_INIT_PAGES) };
    if ret != 0 {
        return -1;
    }

    // Allocate the list head at a known location.
    // We use the bump allocator's first allocation for this.
    let head_raw = unsafe {
        bump_alloc(
            &raw mut BUMP,
            arena_ptr,
            size_of::<ArenaListHead>() as u64,
            align_of::<ArenaListHead>() as u64,
        )
    };
    if head_raw.is_null() {
        return -2;
    }
    let head = head_raw.cast::<ArenaListHead>();
    let arena_base = arena_ptr as *mut u8;

    // We'll build the list in reverse order (prepend) for simplicity.
    // Final order: Counter(1) → Label("hello") → Counter(2) → Label("arena") → Counter(3)
    // Build order: Counter(3), Label("arena"), Counter(2), Label("hello"), Counter(1)

    let mut list_head_ptr: ArenaPtr<ArenaNodeHeader> = ArenaPtr::null();
    let mut count: u64 = 0;

    // Helper: allocate and link a counter node
    macro_rules! push_counter {
        ($val:expr) => {{
            let raw = bump_alloc(
                &raw mut BUMP,
                arena_ptr,
                size_of::<CounterNode>() as u64,
                align_of::<CounterNode>() as u64,
            );
            if raw.is_null() {
                return -3;
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
            list_head_ptr = ArenaPtr::from_raw(node.cast(), arena_base);
            count += 1;
        }};
    }

    // Helper: allocate and link a label node
    macro_rules! push_label {
        ($bytes:expr) => {{
            let raw = bump_alloc(
                &raw mut BUMP,
                arena_ptr,
                size_of::<LabelNode>() as u64,
                align_of::<LabelNode>() as u64,
            );
            if raw.is_null() {
                return -4;
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
            list_head_ptr = ArenaPtr::from_raw(node.cast(), arena_base);
            count += 1;
        }};
    }

    // Build the list (prepend order → reverse of desired traversal order)
    push_counter!(3);
    push_label!(b"arena");
    push_counter!(2);
    push_label!(b"hello");
    push_counter!(1);

    // Write the list head to the arena's header region
    core::ptr::write_volatile(
        head,
        ArenaListHead {
            head: list_head_ptr,
            count,
        },
    );

    0
}

// ── BPF program entry point ────────────────────────────────────────────

#[fentry(function = "hrtimer_nanosleep", sleepable)]
fn arena_bump_test(_ctx: FEntryContext) -> i32 {
    // Only build the list once
    let initialized = unsafe { core::ptr::read_volatile(&raw const INITIALIZED) };
    if initialized != 0 {
        return 0;
    }

    let arena_ptr = ARENA.as_ptr();
    let _ret = unsafe { build_list(arena_ptr) };

    // Always allow socket creation, even if test failed.
    unsafe {
        core::ptr::write_volatile(&raw mut INITIALIZED, 1);
    }

    0
}
