//! BPF arena global allocator.
//!
//! Implements [`GlobalAlloc`] on top of the BPF arena bump allocator,
//! enabling vanilla `alloc` crate data structures (Vec, BTreeMap, etc.)
//! to allocate directly from arena memory.
//!
//! ## Usage
//!
//! ```ignore
//! #![no_std]
//! #![no_main]
//! #![feature(allocator_api)]
//!
//! extern crate alloc;
//!
//! use aya_ebpf::kfuncs::global_alloc::ArenaGlobalAlloc;
//!
//! #[global_allocator]
//! static ALLOC: ArenaGlobalAlloc = ArenaGlobalAlloc::new();
//!
//! // In your BPF init function:
//! unsafe { ALLOC.init(arena_map_ptr, 64) }; // 64 pages = 256 KiB
//!
//! // Then use alloc types normally:
//! let mut v: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
//! v.push(42);
//! ```
//!
//! ## Limitations
//!
//! - **No deallocation**: the bump allocator never frees individual
//!   allocations. `dealloc` is a no-op. This is fine for BPF programs
//!   that build data structures during init and never drop them.
//! - **Not thread-safe**: single-CPU init paths only (SEC("syscall"),
//!   struct_ops init).
//! - **OOM aborts**: `handle_alloc_error` triggers a BPF program abort.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;
use core::ptr;

use super::{arena_alloc_pages, cast_kern, NUMA_NO_NODE};

const PAGE_SIZE: u64 = 4096;

/// Global allocator backed by BPF arena memory.
///
/// Uses a simple bump allocator: a contiguous region of arena pages
/// with a monotonically advancing offset. Individual deallocations
/// are no-ops.
pub struct ArenaGlobalAlloc {
    /// Base pointer to arena memory (set by `init`).
    base: core::cell::UnsafeCell<*mut u8>,
    /// Current allocation offset within the arena region.
    offset: core::cell::UnsafeCell<u64>,
    /// Total capacity in bytes.
    capacity: core::cell::UnsafeCell<u64>,
}

unsafe impl Sync for ArenaGlobalAlloc {}
unsafe impl Send for ArenaGlobalAlloc {}

impl ArenaGlobalAlloc {
    /// Create a new uninitialized arena allocator.
    ///
    /// Must call [`init`](Self::init) before any allocation.
    pub const fn new() -> Self {
        Self {
            base: core::cell::UnsafeCell::new(ptr::null_mut()),
            offset: core::cell::UnsafeCell::new(0),
            capacity: core::cell::UnsafeCell::new(0),
        }
    }

    /// Initialize the allocator with `page_count` pages from the arena.
    ///
    /// # Safety
    ///
    /// - `arena_map` must point to a valid `BPF_MAP_TYPE_ARENA` map.
    /// - Must be called exactly once, before any allocation.
    /// - Must be called from a single-CPU context (e.g. SEC("syscall")).
    #[inline(always)]
    pub unsafe fn init(&self, arena_map: *mut c_void, page_count: u32) -> i32 {
        let mem = arena_alloc_pages(arena_map, ptr::null_mut(), page_count, NUMA_NO_NODE, 0);
        if mem.is_null() {
            return -12; // ENOMEM
        }

        unsafe {
            let base = self.base.get();
            let offset = self.offset.get();
            let capacity = self.capacity.get();

            ptr::write_volatile(base, mem as *mut u8);
            ptr::write_volatile(offset, 0);
            ptr::write_volatile(capacity, u64::from(page_count) * PAGE_SIZE);
        }

        0
    }

    /// Returns the number of bytes currently allocated.
    #[inline(always)]
    pub fn used(&self) -> u64 {
        unsafe { ptr::read_volatile(self.offset.get()) }
    }

    /// Returns the total capacity in bytes.
    #[inline(always)]
    pub fn capacity(&self) -> u64 {
        unsafe { ptr::read_volatile(self.capacity.get()) }
    }
}

#[inline(always)]
fn round_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

unsafe impl GlobalAlloc for ArenaGlobalAlloc {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let base = cast_kern(ptr::read_volatile(self.base.get()));
            let off = ptr::read_volatile(self.offset.get());
            let cap = ptr::read_volatile(self.capacity.get());

            let align = layout.align() as u64;
            let size = layout.size() as u64;

            let aligned_off = round_up(off, align);
            let new_off = aligned_off + size;

            if new_off > cap {
                return ptr::null_mut();
            }

            ptr::write_volatile(self.offset.get(), new_off);
            base.add(aligned_off as usize)
        }
    }

    #[inline(always)]
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator: individual deallocation is a no-op.
    }
}
