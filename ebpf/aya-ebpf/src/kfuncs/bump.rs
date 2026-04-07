//! BPF arena bump allocator.
//!
//! Port of `scx/lib/alloc/bump.bpf.c` — a simple bump allocator backed by
//! `bpf_arena_alloc_pages`. Memory is allocated in large contiguous blocks
//! from the arena and parceled out via bump pointer. Individual allocations
//! cannot be freed; only the entire allocator can be destroyed.
//!
//! ## Usage
//!
//! ```ignore
//! use aya_ebpf::kfuncs::bump::{BumpAllocator, bump_init, bump_alloc};
//!
//! #[unsafe(no_mangle)]
//! static mut BUMP: BumpAllocator = BumpAllocator::new();
//!
//! // In init (SEC("syscall") or struct_ops init):
//! unsafe { bump_init(&raw mut BUMP, arena_map_ptr, 8) }; // 8 pages initial
//!
//! // In BPF program:
//! let ptr = unsafe { bump_alloc(&raw mut BUMP, arena_map_ptr, 64, 8) }; // 64 bytes, 8-byte aligned
//! ```

use core::ffi::c_void;
use core::ptr;

use super::{arena_alloc_pages, arena_free_pages, NUMA_NO_NODE};

/// Page size (4096 bytes on most architectures).
const PAGE_SIZE: u64 = 4096;

/// Maximum total arena memory (1 MiB by default, matching the C implementation).
const ARENA_MAX_MEMORY: u64 = 1 << 20;

/// Header embedded at the start of each allocated block to form a linked list.
/// This enables [`bump_destroy`] to free all blocks.
#[repr(C)]
struct BlockLink {
    next: *mut Self,
}

/// State for the BPF arena bump allocator.
///
/// This struct lives in BPF `.data` (a global variable). It tracks the current
/// allocation block and offset within it.
///
/// **Concurrency note**: The C version uses `bpf_spin_lock`. aya-ebpf does not
/// yet expose BPF spin locks, so concurrent allocations from multiple CPUs are
/// **not safe** with this implementation. For single-CPU init paths (e.g.,
/// `SEC("syscall")` probes or `struct_ops` `init`), this is fine.
#[repr(C)]
pub struct BumpAllocator {
    /// Size of each contiguous block in bytes.
    max_contig_bytes: u64,
    /// Pointer to the current memory block (arena address).
    memory: *mut c_void,
    /// Current offset within the block (next allocation starts here).
    off: u64,
    /// Maximum total memory usage allowed.
    lim_memusage: u64,
    /// Current total memory usage.
    cur_memusage: u64,
}

unsafe impl Sync for BumpAllocator {}
unsafe impl Send for BumpAllocator {}

impl Default for BumpAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl BumpAllocator {
    /// Create a new uninitialized bump allocator.
    ///
    /// Must call [`bump_init`] before use.
    pub const fn new() -> Self {
        Self {
            max_contig_bytes: 0,
            memory: ptr::null_mut(),
            off: 0,
            lim_memusage: ARENA_MAX_MEMORY,
            cur_memusage: 0,
        }
    }
}

/// Round `val` up to the nearest multiple of `align`.
///
/// `align` must be a power of two.
#[inline(always)]
const fn round_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

/// Initialize the bump allocator with `alloc_pages` pages from the arena.
///
/// # Arguments
///
/// * `bump` — pointer to the `BumpAllocator` global
/// * `arena_map` — pointer to the BPF arena map
/// * `alloc_pages` — number of pages for the initial block
///
/// # Returns
///
/// 0 on success, negative errno on failure.
///
/// # Safety
///
/// `bump` must point to a valid, writable `BumpAllocator`.
/// `arena_map` must point to a valid `BPF_MAP_TYPE_ARENA` map.
#[inline(always)]
pub unsafe fn bump_init(
    bump: *mut BumpAllocator,
    arena_map: *mut c_void,
    alloc_pages: u32,
) -> i32 {
    let max_bytes = u64::from(alloc_pages) * PAGE_SIZE;

    let memory = arena_alloc_pages(arena_map, ptr::null_mut(), alloc_pages, NUMA_NO_NODE, 0);
    if memory.is_null() {
        return -12; // ENOMEM
    }

    // Initialize the block link header at the start of the block.
    let link = memory.cast::<BlockLink>();
    unsafe {
        ptr::write_volatile(&raw mut (*link).next, ptr::null_mut());
    }

    let link_size = size_of::<BlockLink>() as u64;

    unsafe {
        ptr::write_volatile(&raw mut (*bump).max_contig_bytes, max_bytes);
        ptr::write_volatile(&raw mut (*bump).memory, memory);
        ptr::write_volatile(&raw mut (*bump).off, link_size);
        ptr::write_volatile(&raw mut (*bump).lim_memusage, ARENA_MAX_MEMORY);
        ptr::write_volatile(&raw mut (*bump).cur_memusage, max_bytes);
    }

    0
}

/// Allocate `bytes` with the given `alignment` from the bump allocator.
///
/// # Arguments
///
/// * `bump` — pointer to the `BumpAllocator` global
/// * `arena_map` — pointer to the BPF arena map
/// * `bytes` — number of bytes to allocate
/// * `alignment` — required alignment (must be a power of two)
///
/// # Returns
///
/// Pointer to the allocated memory, or null on failure.
///
/// # Safety
///
/// `bump` must point to a valid, initialized `BumpAllocator`.
/// `arena_map` must point to a valid `BPF_MAP_TYPE_ARENA` map.
#[inline(always)]
pub unsafe fn bump_alloc(
    bump: *mut BumpAllocator,
    arena_map: *mut c_void,
    bytes: u64,
    alignment: u64,
) -> *mut c_void {
    // Read current state with volatile to prevent reordering.
    let memory = unsafe { ptr::read_volatile(&raw const (*bump).memory) };
    let off = unsafe { ptr::read_volatile(&raw const (*bump).off) };
    let max_contig = unsafe { ptr::read_volatile(&raw const (*bump).max_contig_bytes) };
    let lim = unsafe { ptr::read_volatile(&raw const (*bump).lim_memusage) };
    let cur = unsafe { ptr::read_volatile(&raw const (*bump).cur_memusage) };

    let addr = (memory as u64) + off;
    let padding = round_up(addr, alignment) - addr;
    let alloc_bytes = bytes + padding;

    // Check if allocation fits in current block.
    if off + alloc_bytes <= max_contig {
        // Fast path: fits in current block.
        let result = (addr + padding) as *mut c_void;
        unsafe {
            ptr::write_volatile(&raw mut (*bump).off, off + alloc_bytes);
        }
        return result;
    }

    // Slow path: need a new block.
    if max_contig == 0 {
        return ptr::null_mut(); // Not initialized.
    }

    if cur + max_contig > lim {
        return ptr::null_mut(); // Memory limit exceeded.
    }

    #[expect(clippy::cast_possible_truncation, reason = "max_contig_bytes fits in u32 pages")]
    let alloc_pages = (max_contig / PAGE_SIZE) as u32;
    let new_memory = arena_alloc_pages(arena_map, ptr::null_mut(), alloc_pages, NUMA_NO_NODE, 0);
    if new_memory.is_null() {
        return ptr::null_mut();
    }

    // Link the new block to the old one (for destroy).
    let new_link = new_memory.cast::<BlockLink>();
    unsafe {
        ptr::write_volatile(&raw mut (*new_link).next, memory.cast::<BlockLink>());
    }

    let link_size = size_of::<BlockLink>() as u64;
    let new_addr = (new_memory as u64) + link_size;
    let new_padding = round_up(new_addr, alignment) - new_addr;
    let new_alloc_bytes = bytes + new_padding;

    if new_alloc_bytes + link_size > max_contig {
        // Allocation is too large even for a fresh block.
        arena_free_pages(arena_map, new_memory, alloc_pages);
        return ptr::null_mut();
    }

    let result = (new_addr + new_padding) as *mut c_void;

    unsafe {
        ptr::write_volatile(&raw mut (*bump).memory, new_memory);
        ptr::write_volatile(&raw mut (*bump).off, link_size + new_alloc_bytes);
        ptr::write_volatile(&raw mut (*bump).cur_memusage, cur + max_contig);
    }

    result
}

/// Destroy the bump allocator, freeing all arena pages.
///
/// After calling this, all pointers returned by [`bump_alloc`] are invalid.
///
/// Note: This function walks the block linked list. In BPF, loops must be
/// bounded — this limits to 256 blocks.
///
/// # Safety
///
/// `bump` must point to a valid `BumpAllocator`.
/// `arena_map` must point to a valid `BPF_MAP_TYPE_ARENA` map.
/// All pointers returned by previous [`bump_alloc`] calls become invalid.
#[inline(always)]
pub unsafe fn bump_destroy(bump: *mut BumpAllocator, arena_map: *mut c_void) {
    let max_contig = unsafe { ptr::read_volatile(&raw const (*bump).max_contig_bytes) };
    if max_contig == 0 {
        return;
    }

    #[expect(clippy::cast_possible_truncation, reason = "max_contig_bytes fits in u32 pages")]
    let alloc_pages = (max_contig / PAGE_SIZE) as u32;
    let mut current = unsafe { ptr::read_volatile(&raw const (*bump).memory) };

    // Walk the linked list and free each block.
    // BPF verifier needs bounded loops — limit to 256 iterations.
    let mut i = 0u32;
    while !current.is_null() && i < 256 {
        let link = current.cast::<BlockLink>();
        let next = unsafe { ptr::read_volatile(&raw const (*link).next) };
        arena_free_pages(arena_map, current, alloc_pages);
        current = next.cast();
        i += 1;
    }

    // Zero out the allocator state.
    unsafe {
        ptr::write_volatile(&raw mut (*bump).max_contig_bytes, 0);
        ptr::write_volatile(&raw mut (*bump).memory, ptr::null_mut());
        ptr::write_volatile(&raw mut (*bump).off, 0);
        ptr::write_volatile(&raw mut (*bump).cur_memusage, 0);
    }
}

/// Set the memory usage limit for the bump allocator.
///
/// # Returns
///
/// 0 on success, -EINVAL if the limit is invalid.
///
/// # Safety
///
/// `bump` must point to a valid `BumpAllocator`.
#[inline(always)]
pub unsafe fn bump_memlimit(bump: *mut BumpAllocator, limit: u64) -> i32 {
    if limit > ARENA_MAX_MEMORY || !limit.is_multiple_of(PAGE_SIZE) {
        return -22; // EINVAL
    }

    let cur = unsafe { ptr::read_volatile(&raw const (*bump).cur_memusage) };
    if limit < cur {
        return -22; // EINVAL
    }

    unsafe {
        ptr::write_volatile(&raw mut (*bump).lim_memusage, limit);
    }

    0
}
