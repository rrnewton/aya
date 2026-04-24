//! BPF arena kfunc wrappers.
//!
//! These wrap the kernel's `bpf_arena_alloc_pages` and `bpf_arena_free_pages`
//! kfuncs, which allocate and free pages within a BPF arena map.
//!
//! Arena maps (`BPF_MAP_TYPE_ARENA`) provide a shared memory region between
//! BPF programs and userspace. The kernel maps the arena at the same virtual
//! address on both sides (via `map_extra`), so raw pointers work directly
//! without translation.
//!
//! ## Address space casts
//!
//! The C BPF toolchain uses `__attribute__((address_space(1)))` (`__arena`)
//! to annotate arena pointers, and the compiler automatically inserts
//! `BPF_ADDR_SPACE_CAST` instructions. Rust's BPF backend does not support
//! address space annotations. For programs that set `BPF_F_NO_USER_CONV`
//! on the arena map, address space casts are not needed — the verifier
//! treats all arena pointers as kernel pointers.
//!
//! For programs that need address space casts, use the [`cast_kern`] and
//! [`cast_user`] inline asm helpers.

use core::ffi::c_void;

// ── NUMA constant ──────────────────────────────────────────────────────

/// NUMA node "any" — equivalent to the kernel's `NUMA_NO_NODE` (-1).
pub const NUMA_NO_NODE: i32 = -1;

// ── Extern kfunc declarations (sym targets for inline asm) ─────────────

unsafe extern "C" {
    fn bpf_arena_alloc_pages(
        p: *mut c_void,
        addr: *mut c_void,
        page_cnt: u32,
        node_id: i32,
        flags: u64,
    ) -> *mut c_void;

    fn bpf_arena_free_pages(p: *mut c_void, addr: *mut c_void, page_cnt: u32);
}

// ── Safe wrappers ──────────────────────────────────────────────────────

/// Allocate pages from a BPF arena map.
///
/// # Arguments
///
/// * `arena_map` — pointer to the arena map (obtained from the map definition)
/// * `addr` — hint address within the arena (NULL lets the kernel choose)
/// * `page_cnt` — number of pages to allocate
/// * `node_id` — NUMA node preference, or [`NUMA_NO_NODE`] for any
/// * `flags` — allocation flags (currently 0)
///
/// # Returns
///
/// Pointer to the allocated arena memory, or null on failure.
#[inline(always)]
pub fn arena_alloc_pages(
    arena_map: *mut c_void,
    addr: *mut c_void,
    page_cnt: u32,
    node_id: i32,
    flags: u64,
) -> *mut c_void {
    let ret: *mut c_void;
    unsafe {
        core::arch::asm!(
            "call {func}",
            func = sym bpf_arena_alloc_pages,
            inlateout("r1") arena_map => _,
            inlateout("r2") addr => _,
            inlateout("r3") u64::from(page_cnt) => _,
            inlateout("r4") i64::from(node_id) => _,
            inlateout("r5") flags => _,
            lateout("r0") ret,
        );
    }
    // The kfunc returns an arena pointer (address_space=1). Since Rust
    // doesn't support address_space annotations, we must explicitly cast
    // to kernel pointer (address_space=0) so the BPF verifier recognizes
    // the return value as a dereferenceable pointer.
    cast_kern(ret)
}

/// Free pages back to a BPF arena map.
///
/// # Arguments
///
/// * `arena_map` — pointer to the arena map
/// * `addr` — pointer to the arena memory to free (must have been returned
///   by [`arena_alloc_pages`])
/// * `page_cnt` — number of pages to free (must match the allocation)
#[inline(always)]
pub fn arena_free_pages(arena_map: *mut c_void, addr: *mut c_void, page_cnt: u32) {
    unsafe {
        core::arch::asm!(
            "call {func}",
            func = sym bpf_arena_free_pages,
            inlateout("r1") arena_map => _,
            inlateout("r2") addr => _,
            inlateout("r3") u64::from(page_cnt) => _,
            lateout("r0") _,
            lateout("r4") _,
            lateout("r5") _,
        );
    }
}

/// Cast a user-space (`address_space`=1) arena pointer to kernel-space (`address_space`=0).
///
/// On architectures/kernels that support `BPF_ADDR_SPACE_CAST`, this performs
/// the actual cast. When `BPF_F_NO_USER_CONV` is set on the arena map, this
/// is a no-op.
///
/// The BPF instruction encoding for `addr_space_cast` is:
///   `dst = (dst_as)((src_as)src)` with:
///   - `BPF_ALU64 | BPF_MOV | BPF_X` (opcode `0xbf`)
///   - off = 1 (`addr_space_cast` marker)
///   - imm = (`dst_as` << 16) | `src_as`
///
/// `cast_kern`: `dst_as`=0, `src_as`=1 → imm = 0x00000001
#[inline(always)]
pub fn cast_kern<T>(ptr: *mut T) -> *mut T {
    let result: *mut T;
    unsafe {
        // Use .ifc conditional assembly to emit the correct register nibble.
        // The BPF addr_space_cast instruction has dst_reg == src_reg.
        core::arch::asm!(
            ".byte 0xbf",
            ".ifc {0}, r0; .byte 0x00; .endif",
            ".ifc {0}, r1; .byte 0x11; .endif",
            ".ifc {0}, r2; .byte 0x22; .endif",
            ".ifc {0}, r3; .byte 0x33; .endif",
            ".ifc {0}, r4; .byte 0x44; .endif",
            ".ifc {0}, r5; .byte 0x55; .endif",
            ".ifc {0}, r6; .byte 0x66; .endif",
            ".ifc {0}, r7; .byte 0x77; .endif",
            ".ifc {0}, r8; .byte 0x88; .endif",
            ".ifc {0}, r9; .byte 0x99; .endif",
            ".short 1",             // off = 1 (addr_space_cast)
            ".int 1",               // imm = (0 << 16) | 1
            inlateout(reg) ptr => result,
        );
    }
    result
}

/// Cast a kernel-space (`address_space`=0) arena pointer to user-space (`address_space`=1).
///
/// See [`cast_kern`] for details. `cast_user` uses `dst_as`=1, `src_as`=0 → imm = 0x00010000.
#[inline(always)]
pub fn cast_user<T>(ptr: *mut T) -> *mut T {
    let result: *mut T;
    unsafe {
        core::arch::asm!(
            ".byte 0xbf",
            ".ifc {0}, r0; .byte 0x00; .endif",
            ".ifc {0}, r1; .byte 0x11; .endif",
            ".ifc {0}, r2; .byte 0x22; .endif",
            ".ifc {0}, r3; .byte 0x33; .endif",
            ".ifc {0}, r4; .byte 0x44; .endif",
            ".ifc {0}, r5; .byte 0x55; .endif",
            ".ifc {0}, r6; .byte 0x66; .endif",
            ".ifc {0}, r7; .byte 0x77; .endif",
            ".ifc {0}, r8; .byte 0x88; .endif",
            ".ifc {0}, r9; .byte 0x99; .endif",
            ".short 1",             // off = 1 (addr_space_cast)
            ".int 0x10000",         // imm = (1 << 16) | 0
            inlateout(reg) ptr => result,
        );
    }
    result
}
