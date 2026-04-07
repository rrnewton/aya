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

// ── Arena Hash Map ────────────────────────────────────────────────────
//
// Open-addressing hash map with linear probing, designed for shared
// BPF/userspace access via arena memory.
//
// Features:
// - Fixed capacity (power of 2) for fast modulo via bitmask
// - u64 keys and u64 values (covers task IDs, cgids, counters, etc.)
// - Bounded linear probing (BPF verifier compatible)
// - Tombstone deletion for correct probe chains
// - All types #[repr(C)] for cross-boundary compatibility

/// State of a hash map entry slot.
#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum HashEntryState {
    /// Slot is empty (never used or cleared after compaction).
    Empty = 0,
    /// Slot contains a valid key-value pair.
    Occupied = 1,
    /// Slot was deleted; probing must continue past it.
    Tombstone = 2,
}

/// A single entry in the arena hash map.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaHashEntry {
    /// The key.
    pub key: u64,
    /// The value.
    pub value: u64,
    /// Slot state (empty/occupied/tombstone).
    pub state: u32,
    /// Padding for 8-byte alignment.
    pub _pad: u32,
}

impl ArenaHashEntry {
    /// An empty entry (zero-initialized).
    pub const EMPTY: Self = Self {
        key: 0,
        value: 0,
        state: HashEntryState::Empty as u32,
        _pad: 0,
    };
}

/// Maximum number of probes before giving up.
/// Must be bounded for BPF verifier. 128 is generous for typical load factors.
pub const HASH_MAX_PROBES: u32 = 128;

/// Header for an arena-backed hash map.
///
/// The entry array is stored separately in arena memory (pointed to by
/// `entries`). This allows the hash map to be allocated from the bump
/// allocator like any other arena object.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaHashMap {
    /// Number of buckets (must be a power of 2).
    pub capacity: u32,
    /// Number of occupied entries (excludes tombstones).
    pub count: u32,
    /// Bitmask for fast modulo: `capacity - 1`.
    pub mask: u32,
    /// Padding for alignment.
    pub _pad: u32,
    /// Offset to the entry array in arena memory.
    pub entries: ArenaPtr<ArenaHashEntry>,
}

impl ArenaHashMap {
    /// FNV-1a-inspired hash for u64 keys.
    ///
    /// Fast, decent distribution, no branches — ideal for BPF.
    #[inline(always)]
    pub const fn hash_key(key: u64) -> u32 {
        // splitmix64 finalizer — excellent avalanche properties.
        let mut h = key;
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58476D1CE4E5B9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D049BB133111EB);
        h ^= h >> 31;
        h as u32
    }
}

/// Initialize an arena hash map in the given memory buffer.
///
/// `header_ptr` must point to space for an `ArenaHashMap`.
/// `entries_ptr` must point to space for `capacity` `ArenaHashEntry` slots.
/// `capacity` must be a power of 2 and > 0.
/// `arena_base` is the base address of the arena (for computing offsets).
///
/// Returns -1 if capacity is 0 or not a power of 2.
///
/// # Safety
///
/// All pointers must be valid and writable. `entries_ptr` must have room
/// for `capacity` entries.
pub unsafe fn arena_hash_init(
    header_ptr: *mut ArenaHashMap,
    entries_ptr: *mut ArenaHashEntry,
    capacity: u32,
    arena_base: *mut u8,
) -> i32 {
    // Validate capacity is a power of 2 and non-zero
    if capacity == 0 || (capacity & (capacity - 1)) != 0 {
        return -1;
    }

    // Zero-initialize all entries
    let mut i: u32 = 0;
    while i < capacity {
        let entry = entries_ptr.add(i as usize);
        core::ptr::write(entry, ArenaHashEntry::EMPTY);
        i += 1;
    }

    // Write header
    core::ptr::write(
        header_ptr,
        ArenaHashMap {
            capacity,
            count: 0,
            mask: capacity - 1,
            _pad: 0,
            entries: ArenaPtr::from_raw(entries_ptr, arena_base),
        },
    );

    0
}

/// Insert a key-value pair into the hash map.
///
/// If the key already exists, its value is updated.
///
/// Returns:
/// -  0 = inserted (new key)
/// -  1 = updated (existing key)
/// - -1 = map is full (no empty slot found within probe limit)
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaHashMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_hash_insert(
    map: *mut ArenaHashMap,
    key: u64,
    value: u64,
    arena_base: *mut u8,
) -> i32 {
    let header = &mut *map;
    let entries = header.entries.resolve(arena_base);
    if entries.is_null() {
        return -1;
    }

    let idx_start = ArenaHashMap::hash_key(key) & header.mask;
    let max_probes = if header.capacity < HASH_MAX_PROBES {
        header.capacity
    } else {
        HASH_MAX_PROBES
    };

    // Track first tombstone for reuse
    let mut first_tombstone: i32 = -1;
    let mut probe: u32 = 0;

    while probe < max_probes {
        let idx = (idx_start + probe) & header.mask;
        let entry = &mut *entries.add(idx as usize);

        match entry.state {
            s if s == HashEntryState::Empty as u32 => {
                // Empty slot — use tombstone if we found one, otherwise use this
                let target_idx = if first_tombstone >= 0 {
                    first_tombstone as u32
                } else {
                    idx
                };
                let target = &mut *entries.add(target_idx as usize);
                target.key = key;
                target.value = value;
                target.state = HashEntryState::Occupied as u32;
                header.count += 1;
                return 0; // Inserted
            }
            s if s == HashEntryState::Occupied as u32 => {
                if entry.key == key {
                    entry.value = value;
                    return 1; // Updated
                }
            }
            s if s == HashEntryState::Tombstone as u32 => {
                if first_tombstone < 0 {
                    first_tombstone = idx as i32;
                }
            }
            _ => {}
        }
        probe += 1;
    }

    // If we found a tombstone but no empty slot and no matching key:
    // We can only safely insert into the tombstone if we searched the
    // ENTIRE table (max_probes >= capacity), guaranteeing the key is
    // not present elsewhere. Otherwise, inserting would create a
    // duplicate if the key exists beyond our probe window.
    if first_tombstone >= 0 && max_probes >= header.capacity {
        let target = &mut *entries.add(first_tombstone as usize);
        target.key = key;
        target.value = value;
        target.state = HashEntryState::Occupied as u32;
        header.count += 1;
        return 0;
    }

    -1 // Full
}

/// Look up a key in the hash map.
///
/// Returns a pointer to the value if found, or null if not found.
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaHashMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_hash_get(
    map: *const ArenaHashMap,
    key: u64,
    arena_base: *mut u8,
) -> *const u64 {
    let header = &*map;
    let entries = header.entries.resolve(arena_base);
    if entries.is_null() {
        return core::ptr::null();
    }

    let idx_start = ArenaHashMap::hash_key(key) & header.mask;
    let max_probes = if header.capacity < HASH_MAX_PROBES {
        header.capacity
    } else {
        HASH_MAX_PROBES
    };

    let mut probe: u32 = 0;
    while probe < max_probes {
        let idx = (idx_start + probe) & header.mask;
        let entry = &*entries.add(idx as usize);

        match entry.state {
            s if s == HashEntryState::Empty as u32 => {
                return core::ptr::null(); // Key not in map
            }
            s if s == HashEntryState::Occupied as u32 => {
                if entry.key == key {
                    return &entry.value;
                }
            }
            // Tombstone: continue probing
            _ => {}
        }
        probe += 1;
    }

    core::ptr::null()
}

/// Delete a key from the hash map.
///
/// Returns:
/// -  0 = deleted
/// - -1 = key not found
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaHashMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_hash_delete(
    map: *mut ArenaHashMap,
    key: u64,
    arena_base: *mut u8,
) -> i32 {
    let header = &mut *map;
    let entries = header.entries.resolve(arena_base);
    if entries.is_null() {
        return -1;
    }

    let idx_start = ArenaHashMap::hash_key(key) & header.mask;
    let max_probes = if header.capacity < HASH_MAX_PROBES {
        header.capacity
    } else {
        HASH_MAX_PROBES
    };

    let mut probe: u32 = 0;
    while probe < max_probes {
        let idx = (idx_start + probe) & header.mask;
        let entry = &mut *entries.add(idx as usize);

        match entry.state {
            s if s == HashEntryState::Empty as u32 => {
                return -1; // Key not in map
            }
            s if s == HashEntryState::Occupied as u32 => {
                if entry.key == key {
                    entry.state = HashEntryState::Tombstone as u32;
                    header.count -= 1;
                    return 0;
                }
            }
            // Tombstone: continue probing
            _ => {}
        }
        probe += 1;
    }

    -1 // Not found
}

/// Iterate over all occupied entries in the hash map.
///
/// Calls `f(key, value)` for each occupied entry. The callback cannot
/// modify the map (no insert/delete during iteration).
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaHashMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_hash_for_each<F>(map: *const ArenaHashMap, arena_base: *mut u8, mut f: F)
where
    F: FnMut(u64, u64),
{
    let header = &*map;
    let entries = header.entries.resolve(arena_base);
    if entries.is_null() {
        return;
    }

    let mut i: u32 = 0;
    while i < header.capacity {
        let entry = &*entries.add(i as usize);
        if entry.state == HashEntryState::Occupied as u32 {
            f(entry.key, entry.value);
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;
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

    // ── Hash map tests ───────────────────────────────────────────────

    fn make_hash_map(capacity: u32) -> (Vec<u8>, *mut ArenaHashMap) {
        let header_size = mem::size_of::<ArenaHashMap>();
        let entry_size = mem::size_of::<ArenaHashEntry>();
        let total = header_size + entry_size * capacity as usize;
        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();
        let header_ptr = base.cast::<ArenaHashMap>();
        let entries_ptr = unsafe { base.add(header_size) }.cast::<ArenaHashEntry>();
        let ret = unsafe { arena_hash_init(header_ptr, entries_ptr, capacity, base) };
        assert_eq!(ret, 0, "arena_hash_init failed");
        (buf, header_ptr)
    }

    #[test]
    fn hash_entry_layout() {
        assert_eq!(mem::size_of::<ArenaHashEntry>(), 24);
        assert_eq!(mem::align_of::<ArenaHashEntry>(), 8);
    }

    #[test]
    fn hash_map_layout() {
        assert_eq!(mem::size_of::<ArenaHashMap>(), 24);
        assert_eq!(mem::align_of::<ArenaHashMap>(), 8);
    }

    #[test]
    fn hash_map_init() {
        let (buf, map) = make_hash_map(16);
        let base = buf.as_ptr() as *mut u8;
        let header = unsafe { &*map };
        assert_eq!(header.capacity, 16);
        assert_eq!(header.count, 0);
        assert_eq!(header.mask, 15);
        assert!(!header.entries.is_null());

        // All entries should be empty
        let entries = unsafe { header.entries.resolve(base) };
        for i in 0..16 {
            let e = unsafe { &*entries.add(i) };
            assert_eq!(e.state, HashEntryState::Empty as u32);
        }
    }

    #[test]
    fn hash_map_insert_and_get() {
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        // Insert
        let ret = unsafe { arena_hash_insert(map, 42, 100, base) };
        assert_eq!(ret, 0); // New insert

        // Lookup
        let val = unsafe { arena_hash_get(map, 42, base) };
        assert!(!val.is_null());
        assert_eq!(unsafe { *val }, 100);

        // Count
        assert_eq!(unsafe { (*map).count }, 1);
    }

    #[test]
    fn hash_map_update_existing() {
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        unsafe { arena_hash_insert(map, 42, 100, base) };
        let ret = unsafe { arena_hash_insert(map, 42, 200, base) };
        assert_eq!(ret, 1); // Updated

        let val = unsafe { arena_hash_get(map, 42, base) };
        assert_eq!(unsafe { *val }, 200);
        assert_eq!(unsafe { (*map).count }, 1); // Count unchanged
    }

    #[test]
    fn hash_map_miss() {
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        let val = unsafe { arena_hash_get(map, 999, base) };
        assert!(val.is_null());
    }

    #[test]
    fn hash_map_delete() {
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        unsafe { arena_hash_insert(map, 42, 100, base) };
        assert_eq!(unsafe { (*map).count }, 1);

        let ret = unsafe { arena_hash_delete(map, 42, base) };
        assert_eq!(ret, 0); // Deleted
        assert_eq!(unsafe { (*map).count }, 0);

        let val = unsafe { arena_hash_get(map, 42, base) };
        assert!(val.is_null()); // Gone
    }

    #[test]
    fn hash_map_delete_miss() {
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        let ret = unsafe { arena_hash_delete(map, 999, base) };
        assert_eq!(ret, -1); // Not found
    }

    #[test]
    fn hash_map_collision_chain() {
        // Use capacity 4 to force collisions
        let (mut buf, map) = make_hash_map(4);
        let base = buf.as_mut_ptr();

        // Insert 3 entries into a 4-slot table (high load factor)
        unsafe { arena_hash_insert(map, 10, 100, base) };
        unsafe { arena_hash_insert(map, 20, 200, base) };
        unsafe { arena_hash_insert(map, 30, 300, base) };
        assert_eq!(unsafe { (*map).count }, 3);

        // All lookups should succeed
        assert_eq!(unsafe { *arena_hash_get(map, 10, base) }, 100);
        assert_eq!(unsafe { *arena_hash_get(map, 20, base) }, 200);
        assert_eq!(unsafe { *arena_hash_get(map, 30, base) }, 300);
    }

    #[test]
    fn hash_map_delete_then_reinsert() {
        let (mut buf, map) = make_hash_map(8);
        let base = buf.as_mut_ptr();

        unsafe { arena_hash_insert(map, 1, 10, base) };
        unsafe { arena_hash_insert(map, 2, 20, base) };
        unsafe { arena_hash_delete(map, 1, base) };

        // Key 2 should still be found (tombstone doesn't break chain)
        assert_eq!(unsafe { *arena_hash_get(map, 2, base) }, 20);

        // Re-insert key 1 (should reuse tombstone)
        let ret = unsafe { arena_hash_insert(map, 1, 11, base) };
        assert_eq!(ret, 0); // New insert
        assert_eq!(unsafe { *arena_hash_get(map, 1, base) }, 11);
        assert_eq!(unsafe { (*map).count }, 2);
    }

    #[test]
    fn hash_map_for_each() {
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        unsafe {
            arena_hash_insert(map, 1, 10, base);
            arena_hash_insert(map, 2, 20, base);
            arena_hash_insert(map, 3, 30, base);
        }

        let mut sum_keys = 0u64;
        let mut sum_vals = 0u64;
        let mut n = 0u32;
        unsafe {
            arena_hash_for_each(map, base, |k, v| {
                sum_keys += k;
                sum_vals += v;
                n += 1;
            });
        }
        assert_eq!(n, 3);
        assert_eq!(sum_keys, 6); // 1+2+3
        assert_eq!(sum_vals, 60); // 10+20+30
    }

    #[test]
    fn hash_map_many_entries() {
        let (mut buf, map) = make_hash_map(64);
        let base = buf.as_mut_ptr();

        // Insert 40 entries into 64-slot table (~62% load factor)
        for i in 0..40u64 {
            let ret = unsafe { arena_hash_insert(map, i * 7 + 1, i * 100, base) };
            assert_eq!(ret, 0, "failed to insert key {}", i * 7 + 1);
        }
        assert_eq!(unsafe { (*map).count }, 40);

        // Verify all lookups
        for i in 0..40u64 {
            let val = unsafe { arena_hash_get(map, i * 7 + 1, base) };
            assert!(!val.is_null(), "key {} not found", i * 7 + 1);
            assert_eq!(unsafe { *val }, i * 100);
        }

        // Delete half
        for i in 0..20u64 {
            let ret = unsafe { arena_hash_delete(map, i * 7 + 1, base) };
            assert_eq!(ret, 0);
        }
        assert_eq!(unsafe { (*map).count }, 20);

        // Remaining half still found
        for i in 20..40u64 {
            let val = unsafe { arena_hash_get(map, i * 7 + 1, base) };
            assert!(!val.is_null(), "key {} not found after deletions", i * 7 + 1);
            assert_eq!(unsafe { *val }, i * 100);
        }

        // Deleted half not found
        for i in 0..20u64 {
            let val = unsafe { arena_hash_get(map, i * 7 + 1, base) };
            assert!(val.is_null());
        }
    }

    #[test]
    fn hash_map_full() {
        let (mut buf, map) = make_hash_map(4);
        let base = buf.as_mut_ptr();

        // Fill all 4 slots
        for i in 0..4u64 {
            let ret = unsafe { arena_hash_insert(map, i + 1, i * 10, base) };
            assert_eq!(ret, 0);
        }
        assert_eq!(unsafe { (*map).count }, 4);

        // 5th insert should fail (map full, no tombstones)
        let ret = unsafe { arena_hash_insert(map, 99, 990, base) };
        assert_eq!(ret, -1);
    }

    #[test]
    fn hash_key_distribution() {
        // Verify hash function doesn't produce degenerate patterns
        let mut seen = [false; 256];
        for i in 0..256u64 {
            let h = ArenaHashMap::hash_key(i) & 255;
            seen[h as usize] = true;
        }
        // At least 150 of 256 buckets should be hit (decent distribution
        // for sequential keys — perfect would be 256, birthday paradox
        // means ~256*(1-e^(-256/256)) ≈ 162 for random uniform).
        let hit_count = seen.iter().filter(|&&x| x).count();
        assert!(
            hit_count > 150,
            "poor hash distribution: only {hit_count}/256 buckets hit"
        );
    }

    #[test]
    fn hash_map_init_rejects_non_power_of_2() {
        let mut buf = vec![0u8; 1024];
        let base = buf.as_mut_ptr();
        let header = base.cast::<ArenaHashMap>();
        let entries = unsafe { base.add(size_of::<ArenaHashMap>()) }.cast::<ArenaHashEntry>();

        // capacity=0 should fail
        assert_eq!(unsafe { arena_hash_init(header, entries, 0, base) }, -1);
        // capacity=3 (not power of 2) should fail
        assert_eq!(unsafe { arena_hash_init(header, entries, 3, base) }, -1);
        // capacity=5 should fail
        assert_eq!(unsafe { arena_hash_init(header, entries, 5, base) }, -1);
        // capacity=1 should succeed (edge: power of 2)
        assert_eq!(unsafe { arena_hash_init(header, entries, 1, base) }, 0);
        // capacity=4 should succeed
        assert_eq!(unsafe { arena_hash_init(header, entries, 4, base) }, 0);
    }

    #[test]
    fn hash_map_capacity_1() {
        // Edge case: hash map with capacity 1
        let (mut buf, map) = make_hash_map(1);
        let base = buf.as_mut_ptr();

        // Insert one entry
        assert_eq!(unsafe { arena_hash_insert(map, 42, 100, base) }, 0);
        assert_eq!(unsafe { *arena_hash_get(map, 42, base) }, 100);

        // Second insert should fail (full)
        assert_eq!(unsafe { arena_hash_insert(map, 99, 200, base) }, -1);

        // Delete and reinsert
        assert_eq!(unsafe { arena_hash_delete(map, 42, base) }, 0);
        assert_eq!(unsafe { arena_hash_insert(map, 99, 200, base) }, 0);
        assert_eq!(unsafe { *arena_hash_get(map, 99, base) }, 200);
    }

    #[test]
    fn hash_map_key_zero() {
        // Edge case: key 0 should work (not confused with empty)
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        assert_eq!(unsafe { arena_hash_insert(map, 0, 999, base) }, 0);
        let val = unsafe { arena_hash_get(map, 0, base) };
        assert!(!val.is_null());
        assert_eq!(unsafe { *val }, 999);
    }

    #[test]
    fn hash_map_key_u64_max() {
        // Edge case: u64::MAX key
        let (mut buf, map) = make_hash_map(16);
        let base = buf.as_mut_ptr();

        assert_eq!(unsafe { arena_hash_insert(map, u64::MAX, 42, base) }, 0);
        assert_eq!(unsafe { *arena_hash_get(map, u64::MAX, base) }, 42);
    }
}
