#![no_std]
#![no_main]
#![expect(unused_crate_dependencies, reason = "used in other bins")]

use aya_ebpf::{
    kfuncs::bump::{bump_alloc, bump_init, BumpAllocator},
    macros::socket_filter,
    programs::SkBuffContext,
};
use core::ffi::c_void;
#[cfg(not(test))]
extern crate ebpf_panic;

// Simulated arena map pointer — in a real program this comes from the map definition.
// For compilation testing only.
#[unsafe(no_mangle)]
static mut BUMP: BumpAllocator = BumpAllocator::new();

#[socket_filter]
fn arena_bump_test(_ctx: SkBuffContext) -> i64 {
    // This exercises the bump allocator code paths to verify they compile
    // to valid BPF bytecode. In a real program, arena_map_ptr would come
    // from a BPF_MAP_TYPE_ARENA map definition.
    let arena_map_ptr: *mut c_void = core::ptr::null_mut();

    let ret = unsafe { bump_init(&raw mut BUMP, arena_map_ptr, 8) };
    if ret != 0 {
        return -1;
    }

    let ptr = unsafe { bump_alloc(&raw mut BUMP, arena_map_ptr, 64, 8) };
    if ptr.is_null() {
        return -1;
    }

    0
}
