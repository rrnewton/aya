//! A BPF arena map backed by `BPF_MAP_TYPE_ARENA`.
//!
//! Arena maps provide a shared memory region between BPF programs and userspace.
//! Both sides see the same virtual addresses when `map_extra` is used to pin the VA.
//! This enables pointer-based data structures (linked lists, trees, etc.) to be shared
//! between BPF and userspace without address translation.
//!
//! # Kernel requirements
//!
//! Requires kernel 6.9 or later.

use std::{
    borrow::Borrow,
    ffi::c_void,
    os::fd::AsFd as _,
    ptr,
};

use libc::{MAP_SHARED, PROT_READ, PROT_WRITE};

use crate::{
    maps::{MapData, MapError},
    util::MMap,
};

/// Default arena VA hint for `x86_64`.
pub const ARENA_VA_X86_64: u64 = 1 << 44;

/// Default arena VA hint for aarch64.
pub const ARENA_VA_AARCH64: u64 = 1 << 32;

/// Returns the default arena VA hint for the current architecture.
#[must_use]
pub const fn default_arena_va() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        ARENA_VA_X86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        ARENA_VA_AARCH64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // PORT_TODO(aya-XX): Arena VA hint for other architectures.
        // Need to determine safe VA ranges for riscv64, arm, etc.
        0
    }
}

/// A BPF arena map backed by `BPF_MAP_TYPE_ARENA`.
///
/// Arena maps allocate a contiguous virtual address region that is shared between BPF programs
/// and userspace. When `map_extra` is set to a non-zero VA hint, both the kernel (for BPF
/// programs) and userspace map the arena at the same virtual address, enabling pointer-based
/// data structures to work across the BPF/userspace boundary.
///
/// # Example
///
/// ```no_run
/// # use aya::maps::{MapData, arena::Arena};
/// # let bpf = aya::Ebpf::load(&[])?;
/// let arena = Arena::try_from(bpf.map("arena").unwrap())?;
/// let ptr = arena.as_ptr();
/// // ptr points to the shared arena memory, same VA as seen by BPF programs
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[doc(alias = "BPF_MAP_TYPE_ARENA")]
pub struct Arena<T> {
    pub(crate) inner: T,
    mmap: Option<MMap>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena")
            .field("inner", &self.inner)
            .field("mmap", &self.mmap.as_ref().map(MMap::ptr))
            .finish()
    }
}

impl<T: Borrow<MapData>> Arena<T> {
    pub(crate) fn new(map: T) -> Result<Self, MapError> {
        let data = map.borrow();

        // mmap the arena for shared userspace access
        let mmap = Self::mmap_arena(data)?;

        Ok(Self {
            inner: map,
            mmap: Some(mmap),
        })
    }

    fn mmap_arena(data: &MapData) -> Result<MMap, MapError> {
        let obj = data.obj();
        let max_entries = obj.max_entries();
        let map_extra = obj.map_extra();

        // max_entries is the number of pages
        let page_size = page_size();
        let mmap_size = max_entries as usize * page_size;

        if mmap_size == 0 {
            return Err(MapError::InvalidValueSize {
                size: 0,
                expected: 1,
            });
        }

        let fd = data.fd();

        if map_extra != 0 {
            // Fixed-address mapping: use MAP_SHARED | MAP_FIXED_NOREPLACE
            // so both BPF and userspace see the same virtual addresses.
            let addr = map_extra as *mut c_void;
            MMap::new_at(
                fd.as_fd(),
                addr,
                mmap_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | libc::MAP_FIXED_NOREPLACE,
                0,
            )
            .map_err(MapError::from)
        } else {
            // No VA hint: let the kernel choose the address.
            // Pointers won't match between BPF and userspace in this mode.
            MMap::new(
                fd.as_fd(),
                mmap_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                0,
            )
            .map_err(MapError::from)
        }
    }

    /// Returns a raw pointer to the start of the arena memory region.
    ///
    /// This pointer is valid for `max_entries * page_size` bytes and points to memory
    /// shared with BPF programs. When `map_extra` is set, this address matches the
    /// address seen by BPF programs.
    pub fn as_ptr(&self) -> *const c_void {
        self.mmap
            .as_ref()
            .map_or(ptr::null(), |m| m.ptr().as_ptr().cast_const())
    }

    /// Returns a mutable raw pointer to the start of the arena memory region.
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.mmap
            .as_ref()
            .map_or(ptr::null_mut(), |m| m.ptr().as_ptr())
    }

    /// Returns the size of the arena in bytes.
    pub fn len(&self) -> usize {
        let data = self.inner.borrow();
        let max_entries = data.obj().max_entries();
        max_entries as usize * page_size()
    }

    /// Returns `true` if the arena has zero size.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the `map_extra` VA hint configured for this arena.
    pub fn map_extra(&self) -> u64 {
        self.inner.borrow().obj().map_extra()
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) is always safe and returns a valid page size.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
