//! Shared `#[repr(C)]` types for BPF arena data structures.
//!
//! This crate provides types that are safe to use in BPF arena shared memory,
//! where both BPF programs and userspace access the same memory region at the
//! same virtual address (via `map_extra` VA pinning).
//!
//! All types are `#[repr(C)]` and `no_std` compatible.

#![no_std]

use core::marker::PhantomData;

/// An offset-based pointer into arena memory.
///
/// Unlike raw pointers, `ArenaPtr<T>` stores an offset from the arena base,
/// making it valid on both the BPF and userspace sides regardless of the
/// actual mapped address. When `map_extra` pins the VA (same address both
/// sides), the offset can be resolved by adding the arena base.
///
/// A null `ArenaPtr` is represented as offset `u64::MAX`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaPtr<T> {
    offset: u64,
    _marker: PhantomData<*mut T>,
}

impl<T> ArenaPtr<T> {
    /// Sentinel value representing a null arena pointer.
    const NULL_OFFSET: u64 = u64::MAX;

    /// Create a null arena pointer.
    pub const fn null() -> Self {
        Self {
            offset: Self::NULL_OFFSET,
            _marker: PhantomData,
        }
    }

    /// Create an arena pointer from a byte offset relative to the arena base.
    pub const fn from_offset(offset: u64) -> Self {
        Self {
            offset,
            _marker: PhantomData,
        }
    }

    /// Create an arena pointer from a raw pointer and base address.
    ///
    /// The pointer must be within the arena region starting at `base`.
    pub fn from_raw(ptr: *mut T, base: *mut u8) -> Self {
        if ptr.is_null() {
            return Self::null();
        }
        let offset = (ptr as u64).wrapping_sub(base as u64);
        Self::from_offset(offset)
    }

    /// Returns `true` if this is a null arena pointer.
    pub const fn is_null(&self) -> bool {
        self.offset == Self::NULL_OFFSET
    }

    /// Get the raw byte offset from the arena base.
    ///
    /// Returns `None` if null.
    pub const fn offset(&self) -> Option<u64> {
        if self.is_null() {
            None
        } else {
            Some(self.offset)
        }
    }

    /// Resolve this arena pointer to a raw pointer given the arena base address.
    ///
    /// Returns null if this is a null `ArenaPtr`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `base` points to the start of a valid arena mapping
    /// - The offset is within the arena bounds
    /// - The resulting pointer is properly aligned for `T`
    pub unsafe fn resolve(&self, base: *mut u8) -> *mut T {
        if self.is_null() {
            return core::ptr::null_mut();
        }
        base.add(self.offset as usize).cast()
    }
}

impl<T> Default for ArenaPtr<T> {
    fn default() -> Self {
        Self::null()
    }
}

// Safety: ArenaPtr is just an offset (u64), safe to send/share.
unsafe impl<T> Send for ArenaPtr<T> {}
unsafe impl<T> Sync for ArenaPtr<T> {}

/// Header for arena-allocated objects, enabling tagged/polymorphic collections.
///
/// Place this at the beginning of arena-allocated structs to identify their
/// type and track allocation metadata.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaNodeHeader {
    /// Type tag for identifying the concrete type in heterogeneous collections.
    pub tag: u32,
    /// Size of the allocation in bytes (including this header).
    pub size: u32,
    /// Link to the next node in a free list or collection (arena offset).
    pub next: ArenaPtr<ArenaNodeHeader>,
}

impl ArenaNodeHeader {
    /// Create a new header with the given type tag and total size.
    pub const fn new(tag: u32, size: u32) -> Self {
        Self {
            tag,
            size,
            next: ArenaPtr::null(),
        }
    }
}

// ── PoC node types for heterogeneous arena linked list ─────────────────

/// Type tag for [`CounterNode`].
pub const TAG_COUNTER: u32 = 1;

/// Type tag for [`LabelNode`].
pub const TAG_LABEL: u32 = 2;

/// Maximum label length in bytes (including nul terminator).
pub const LABEL_MAX_LEN: usize = 32;

/// A node containing a 64-bit counter value.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CounterNode {
    /// Node header (must be first field for polymorphic access).
    pub header: ArenaNodeHeader,
    /// The counter value.
    pub value: u64,
}

impl CounterNode {
    /// Create a new counter node with the given value.
    pub const fn new(value: u64) -> Self {
        Self {
            header: ArenaNodeHeader::new(TAG_COUNTER, size_of::<Self>() as u32),
            value,
        }
    }
}

/// A node containing a fixed-size label string.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LabelNode {
    /// Node header (must be first field for polymorphic access).
    pub header: ArenaNodeHeader,
    /// Label bytes (nul-terminated).
    pub label: [u8; LABEL_MAX_LEN],
    /// Actual length of the label (excluding nul terminator).
    pub len: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

impl LabelNode {
    /// Create a new label node from a byte slice.
    ///
    /// Truncates to [`LABEL_MAX_LEN`] - 1 bytes.
    pub fn new(s: &[u8]) -> Self {
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
        Self {
            header: ArenaNodeHeader::new(TAG_LABEL, size_of::<Self>() as u32),
            label,
            len: copy_len as u32,
            _pad: 0,
        }
    }
}

/// Shared state stored at the beginning of the arena, readable by both
/// BPF programs and userspace.
///
/// This is written by the BPF program and read by userspace to find the
/// linked list.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaListHead {
    /// Offset to the first node in the linked list (from arena base).
    pub head: ArenaPtr<ArenaNodeHeader>,
    /// Number of nodes in the list.
    pub count: u64,
}

/// A simple bump allocator state for arena memory.
///
/// Tracks the current allocation watermark within an arena region.
/// This is the simplest possible allocator — it never frees individual
/// allocations, only the entire arena.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaBumpState {
    /// Current allocation offset (next allocation starts here).
    pub watermark: u64,
    /// Total size of the arena region in bytes.
    pub capacity: u64,
}

impl ArenaBumpState {
    /// Create a new bump allocator state for an arena of the given capacity.
    pub const fn new(capacity: u64) -> Self {
        Self {
            watermark: 0,
            capacity,
        }
    }

    /// Try to allocate `size` bytes with the given alignment.
    ///
    /// Returns the offset of the allocation, or `None` if the arena is full.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        // Align up the watermark
        let aligned = (self.watermark + align - 1) & !(align - 1);
        let new_watermark = aligned + size;
        if new_watermark > self.capacity {
            return None;
        }
        self.watermark = new_watermark;
        Some(aligned)
    }

    /// Reset the allocator, freeing all allocations.
    pub fn reset(&mut self) {
        self.watermark = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn arena_ptr_layout() {
        assert_eq!(mem::size_of::<ArenaPtr<u32>>(), 8);
        assert_eq!(mem::align_of::<ArenaPtr<u32>>(), 8);
    }

    #[test]
    fn arena_ptr_null() {
        let p: ArenaPtr<u32> = ArenaPtr::null();
        assert!(p.is_null());
        assert!(p.offset().is_none());
    }

    #[test]
    fn arena_ptr_from_offset() {
        let p: ArenaPtr<u32> = ArenaPtr::from_offset(0x1000);
        assert!(!p.is_null());
        assert_eq!(p.offset(), Some(0x1000));
    }

    #[test]
    fn arena_node_header_layout() {
        assert_eq!(mem::size_of::<ArenaNodeHeader>(), 16);
        assert_eq!(mem::align_of::<ArenaNodeHeader>(), 8);
    }

    #[test]
    fn bump_allocator_basic() {
        let mut bump = ArenaBumpState::new(4096);
        let a = bump.alloc(64, 8).unwrap();
        assert_eq!(a, 0);
        let b = bump.alloc(128, 16).unwrap();
        assert_eq!(b, 64);
        assert_eq!(bump.watermark, 192);
    }

    #[test]
    fn bump_allocator_alignment() {
        let mut bump = ArenaBumpState::new(4096);
        bump.alloc(1, 1).unwrap(); // offset 0, watermark 1
        let aligned = bump.alloc(8, 8).unwrap();
        assert_eq!(aligned, 8); // aligned up from 1 to 8
    }

    #[test]
    fn bump_allocator_oom() {
        let mut bump = ArenaBumpState::new(64);
        assert!(bump.alloc(32, 8).is_some());
        assert!(bump.alloc(64, 8).is_none()); // exceeds capacity
    }

    #[test]
    fn counter_node_layout() {
        assert_eq!(mem::size_of::<CounterNode>(), 24); // 16 header + 8 value
        assert_eq!(mem::align_of::<CounterNode>(), 8);
    }

    #[test]
    fn label_node_layout() {
        assert_eq!(mem::size_of::<LabelNode>(), 56); // 16 header + 32 label + 4 len + 4 pad
        assert_eq!(mem::align_of::<LabelNode>(), 8);
    }

    #[test]
    fn arena_list_head_layout() {
        assert_eq!(mem::size_of::<ArenaListHead>(), 16); // 8 head + 8 count
        assert_eq!(mem::align_of::<ArenaListHead>(), 8);
    }

    #[test]
    fn counter_node_tag() {
        let node = CounterNode::new(42);
        assert_eq!(node.header.tag, TAG_COUNTER);
        assert_eq!(node.value, 42);
    }

    #[test]
    fn label_node_content() {
        let node = LabelNode::new(b"hello");
        assert_eq!(node.header.tag, TAG_LABEL);
        assert_eq!(node.len, 5);
        assert_eq!(&node.label[..5], b"hello");
        assert_eq!(node.label[5], 0); // nul terminated
    }
}
