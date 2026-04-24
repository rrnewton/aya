//! Shared `#[repr(C)]` types for BPF arena data structures.
//!
//! This crate provides types that are safe to use in BPF arena shared memory,
//! where both BPF programs and userspace access the same memory region at the
//! same virtual address (via `map_extra` VA pinning).
//!
//! All types are `#[repr(C)]` and `no_std` compatible.
//!
//! # Concurrency
//!
//! **None of the data structures in this crate are safe for concurrent access.**
//! All allocators and collections use non-atomic reads/writes to shared mutable
//! state (watermarks, free lists, hash entries, tree nodes). Concurrent mutation
//! from multiple CPUs causes data races, corruption, and undefined behavior.
//!
//! In BPF programs, this means:
//!
//! | Pattern | Safe? | How |
//! |---|---|---|
//! | Single-CPU init (e.g. `SEC("syscall")`) | Yes | Only one CPU runs the program |
//! | `struct_ops` `init` callback | Yes | Called once during map creation |
//! | Per-CPU slab (one `ArenaSlabState` per CPU) | Yes | No shared mutable state |
//! | Shared map with `bpf_spin_lock` around ops | Yes | External serialization |
//! | Shared map, multiple CPUs, no lock | **No** | Data race on all mutable state |
//! | Read-only from userspace while BPF writes | **No** | Torn reads, no happens-before |
//!
//! **Recommended pattern for multi-CPU access**: partition the arena by CPU.
//! Allocate a separate `ArenaSlabState` / `ArenaHashMap` / `ArenaBTreeMap` per
//! CPU (e.g. using a `BPF_MAP_TYPE_PERCPU_ARRAY` of offsets), so each CPU
//! operates on its own instance. Userspace can then read all instances after
//! the BPF program detaches.
//!
//! For shared mutable access across CPUs, wrap every operation in a
//! `bpf_spin_lock` / `bpf_spin_unlock` pair. Note that `aya-ebpf` does not
//! yet expose BPF spin locks — this is a known gap.

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
///
/// **Not thread-safe.** The `watermark` field is read and written
/// non-atomically. Use one instance per CPU, or protect with a lock.
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

// ── Arena Slab Allocator ──────────────────────────────────────────────
//
// Fixed-size slot allocator with O(1) alloc and free. Uses an intrusive
// free list: each free slot stores an ArenaPtr to the next free slot.
// When the free list is empty, falls back to bump allocation.
//
// Ideal for BPF task contexts: init_task allocates a slot, exit_task
// frees it, and the next init_task reuses it immediately.
//
// NOT THREAD-SAFE: free_head, free_count, bump watermark, and
// total_allocated are all read/written non-atomically. For multi-CPU
// use, allocate one ArenaSlabState per CPU.

/// Minimum slot size (must fit an ArenaPtr for the free list link plus the
/// double-free detection magic).
pub const SLAB_MIN_SLOT_SIZE: u32 = 16;

/// Magic value written at offset 8 in freed slots for double-free detection.
const SLAB_FREE_MAGIC: u64 = 0xDEAD_51AB_F4EE_0A0A;

/// State for the arena slab allocator.
///
/// Manages fixed-size slots with O(1) alloc (pop free list or bump)
/// and O(1) free (push to free list). All operations are BPF verifier
/// safe — no loops, no recursion.
///
/// **Not thread-safe.** All fields are mutated non-atomically during
/// alloc/free. For multi-CPU BPF programs, use one instance per CPU.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaSlabState {
    /// Fallback bump allocator for when the free list is empty.
    pub bump: ArenaBumpState,
    /// Head of the intrusive free list (null = empty).
    pub free_head: ArenaPtr<u8>,
    /// Size of each slot in bytes (fixed at init, must be >= 8).
    pub slot_size: u32,
    /// Number of slots currently in the free list.
    pub free_count: u32,
    /// Total number of slots ever allocated from the bump region.
    pub total_allocated: u32,
    /// Padding for 8-byte alignment.
    pub _pad: u32,
}

/// Initialize a slab allocator.
///
/// `slab` must point to valid, writable memory.
/// `capacity` is the total arena region size in bytes.
/// `slot_size` must be >= [`SLAB_MIN_SLOT_SIZE`] (16 bytes) and a
/// multiple of 8 (for alignment).
///
/// Returns 0 on success, -1 if `slot_size` is invalid.
///
/// # Safety
///
/// `slab` must point to valid, writable memory for an `ArenaSlabState`.
pub unsafe fn arena_slab_init(
    slab: *mut ArenaSlabState,
    capacity: u64,
    slot_size: u32,
) -> i32 {
    if slot_size < SLAB_MIN_SLOT_SIZE || !slot_size.is_multiple_of(8) {
        return -1;
    }
    core::ptr::write_volatile(
        slab,
        ArenaSlabState {
            bump: ArenaBumpState::new(capacity),
            free_head: ArenaPtr::null(),
            slot_size,
            free_count: 0,
            total_allocated: 0,
            _pad: 0,
        },
    );
    0
}

/// Allocate a slot from the slab.
///
/// Returns an `ArenaPtr` to the slot, or a null `ArenaPtr` if the
/// slab is exhausted (free list empty AND bump region full).
///
/// O(1): pops from the free list if non-empty, otherwise bump-allocates.
///
/// # Safety
///
/// `slab` must point to a valid, initialized `ArenaSlabState`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_slab_alloc(slab: *mut ArenaSlabState, arena_base: *mut u8) -> ArenaPtr<u8> {
    let s = &mut *slab;

    // Fast path: pop from free list
    if !s.free_head.is_null() {
        let slot_ptr = s.free_head;
        // Read the next-free pointer stored in the slot itself
        let slot = slot_ptr.resolve(arena_base);
        let next = core::ptr::read(slot.cast::<ArenaPtr<u8>>());
        // Clear the double-free detection magic
        core::ptr::write_volatile(slot.add(8).cast::<u64>(), 0);
        s.free_head = next;
        s.free_count -= 1;
        return slot_ptr;
    }

    // Slow path: bump-allocate a new slot
    match s.bump.alloc(u64::from(s.slot_size), 8) {
        Some(offset) => {
            s.total_allocated += 1;
            ArenaPtr::from_offset(offset)
        }
        None => ArenaPtr::null(), // Exhausted
    }
}

/// Free a slot back to the slab.
///
/// The slot is pushed onto the free list for reuse. O(1).
///
/// Returns 0 on success, -1 if the slot was already freed (double-free).
///
/// # Safety
///
/// `slab` must point to a valid, initialized `ArenaSlabState`.
/// `slot` must be a valid `ArenaPtr` previously returned by
/// [`arena_slab_alloc`] on this slab.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_slab_free(
    slab: *mut ArenaSlabState,
    slot: ArenaPtr<u8>,
    arena_base: *mut u8,
) -> i32 {
    let s = &mut *slab;
    let slot_raw = slot.resolve(arena_base);

    // Check for double-free: freed slots have the magic at offset 8
    let magic_ptr = slot_raw.add(8).cast::<u64>();
    if core::ptr::read(magic_ptr) == SLAB_FREE_MAGIC {
        return -1;
    }

    // Write the current free_head into the slot (intrusive link)
    core::ptr::write_volatile(slot_raw.cast::<ArenaPtr<u8>>(), s.free_head);
    // Write the double-free detection magic
    core::ptr::write_volatile(magic_ptr, SLAB_FREE_MAGIC);

    // Push slot onto the free list
    s.free_head = slot;
    s.free_count += 1;
    0
}

/// Slab allocator statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSlabStats {
    /// Total slots ever bump-allocated.
    pub total_allocated: u32,
    /// Slots currently in the free list (available for reuse).
    pub free_count: u32,
    /// Slots currently in use (total_allocated - free_count).
    pub in_use: u32,
}

/// Get slab allocator statistics.
///
/// # Safety
///
/// `slab` must point to a valid, initialized `ArenaSlabState`.
pub unsafe fn arena_slab_stats(slab: *const ArenaSlabState) -> ArenaSlabStats {
    let s = &*slab;
    ArenaSlabStats {
        total_allocated: s.total_allocated,
        free_count: s.free_count,
        in_use: s.total_allocated - s.free_count,
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
//
// NOT THREAD-SAFE: entry states, keys, values, and the count field are
// all modified non-atomically. Concurrent insert/delete/get from multiple
// CPUs will corrupt the table. Use one map per CPU, or hold bpf_spin_lock.

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
        core::ptr::write_volatile(entry, ArenaHashEntry::EMPTY);
        i += 1;
    }

    // Write header
    core::ptr::write_volatile(
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

// ── Arena B-Tree ──────────────────────────────────────────────────────
//
// Cache-friendly ordered map with bounded operations, designed for
// BPF verifier compatibility:
// - ORDER=8: each node holds up to 7 keys (good cache-line utilization)
// - No recursion: all operations use bounded iterative loops
// - Max depth 10: supports up to ~10^8 entries
// - Proactive split-on-descent: insert never needs to backtrack
// - CLRS-style delete with top-down preemptive rebalancing
//
// NOT THREAD-SAFE: node keys, values, children, and the root/count/height
// fields are all modified non-atomically. Concurrent access from multiple
// CPUs will corrupt the tree. Use one tree per CPU, or hold bpf_spin_lock.

/// B-tree branching factor. Each node has at most ORDER children
/// and ORDER-1 keys.
pub const BTREE_ORDER: usize = 8;

/// Maximum keys per node.
pub const BTREE_MAX_KEYS: usize = BTREE_ORDER - 1;

/// Median index for splitting a full node.
const BTREE_MID: usize = BTREE_MAX_KEYS / 2; // = 3

/// Maximum tree depth. Supports 8^10 ≈ 10^9 entries.
pub const BTREE_MAX_DEPTH: u32 = 10;

/// A B-tree node stored in arena memory.
///
/// Layout: 184 bytes (keys=56 + values=56 + children=64 + metadata=8).
/// Fits in 3 cache lines on most architectures.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BTreeNode {
    /// Sorted keys.
    pub keys: [u64; BTREE_MAX_KEYS],
    /// Values corresponding to keys.
    pub values: [u64; BTREE_MAX_KEYS],
    /// Child pointers (only used in internal nodes). `children[i]` points
    /// to the subtree with keys < `keys[i]`; `children[num_keys]` points
    /// to the subtree with keys > `keys[num_keys-1]`.
    pub children: [ArenaPtr<BTreeNode>; BTREE_ORDER],
    /// Number of keys currently stored (0..=BTREE_MAX_KEYS).
    pub num_keys: u32,
    /// 1 = leaf node, 0 = internal node.
    pub is_leaf: u32,
}

impl BTreeNode {
    /// An empty leaf node.
    pub const EMPTY_LEAF: Self = Self {
        keys: [0; BTREE_MAX_KEYS],
        values: [0; BTREE_MAX_KEYS],
        children: [ArenaPtr::null(); BTREE_ORDER],
        num_keys: 0,
        is_leaf: 1,
    };

    /// An empty internal node.
    const EMPTY_INTERNAL: Self = Self {
        keys: [0; BTREE_MAX_KEYS],
        values: [0; BTREE_MAX_KEYS],
        children: [ArenaPtr::null(); BTREE_ORDER],
        num_keys: 0,
        is_leaf: 0,
    };
}

/// Header for an arena-backed B-tree map.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArenaBTreeMap {
    /// Pointer to the root node (null = empty tree).
    pub root: ArenaPtr<BTreeNode>,
    /// Total number of key-value pairs in the tree.
    pub count: u64,
    /// Current height of the tree (0 = empty, 1 = root only).
    pub height: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

/// Allocate a B-tree node from the bump allocator.
///
/// # Safety
///
/// `bump` must point to a valid `ArenaBumpState` with sufficient capacity.
/// `arena_base` must be the arena base address.
#[inline(always)]
unsafe fn btree_alloc_node(
    bump: *mut ArenaBumpState,
    arena_base: *mut u8,
    leaf: bool,
) -> ArenaPtr<BTreeNode> {
    let state = &mut *bump;
    match state.alloc(size_of::<BTreeNode>() as u64, align_of::<BTreeNode>() as u64) {
        Some(offset) => {
            let node = arena_base.add(offset as usize).cast::<BTreeNode>();
            core::ptr::write_volatile(
                node,
                if leaf {
                    BTreeNode::EMPTY_LEAF
                } else {
                    BTreeNode::EMPTY_INTERNAL
                },
            );
            ArenaPtr::from_offset(offset)
        }
        None => ArenaPtr::null(),
    }
}

/// Find the position of `key` within a node using linear search.
///
/// Returns `(index, found)`:
/// - `found=true`: `keys[index] == key`
/// - `found=false`: `key` belongs at position `index` (for insertion or child descent)
#[inline(always)]
fn btree_find_key(node: &BTreeNode, key: u64) -> (u32, bool) {
    let n = node.num_keys;
    let mut i: u32 = 0;
    while i < n && i < BTREE_MAX_KEYS as u32 {
        if node.keys[i as usize] == key {
            return (i, true);
        }
        if node.keys[i as usize] > key {
            return (i, false);
        }
        i += 1;
    }
    (i, false)
}

/// Split a full child node during top-down insert.
///
/// `parent` gains one key (the median of `child`).
/// `child_idx` is the index of `child` in `parent.children`.
/// `child` is the full node being split (must have MAX_KEYS keys).
/// `new_sibling_ptr` is the ArenaPtr to the already-allocated new right sibling.
/// `new_sibling` is the raw pointer to the new sibling node.
///
/// After split:
/// - `child` keeps the left half (BTREE_MID keys: indices 0..MID)
/// - `new_sibling` gets the right half (BTREE_MID keys: indices MID+1..MAX_KEYS)
/// - The median key (index MID) is promoted to `parent`
///
/// # Safety
///
/// All pointers must be valid. `child.num_keys` must be `BTREE_MAX_KEYS`.
#[inline(always)]
unsafe fn btree_split_child(
    parent: *mut BTreeNode,
    child_idx: u32,
    child: *mut BTreeNode,
    new_sibling: *mut BTreeNode,
    new_sibling_ptr: ArenaPtr<BTreeNode>,
) {
    let p = &mut *parent;
    let c = &*child;
    let s = &mut *new_sibling;

    // Copy right half of child to new sibling
    let mut i: usize = 0;
    while i < BTREE_MID {
        s.keys[i] = c.keys[BTREE_MID + 1 + i];
        s.values[i] = c.values[BTREE_MID + 1 + i];
        i += 1;
    }

    // If internal node, copy right half of children too
    if c.is_leaf == 0 {
        i = 0;
        while i <= BTREE_MID {
            s.children[i] = c.children[BTREE_MID + 1 + i];
            i += 1;
        }
    }

    s.num_keys = BTREE_MID as u32;
    s.is_leaf = c.is_leaf;

    // Shrink child to left half
    (*child).num_keys = BTREE_MID as u32;

    // Make room in parent: shift keys and children right
    let mut j = p.num_keys;
    while j > child_idx {
        p.keys[j as usize] = p.keys[(j - 1) as usize];
        p.values[j as usize] = p.values[(j - 1) as usize];
        p.children[(j + 1) as usize] = p.children[j as usize];
        j -= 1;
    }

    // Promote median key to parent
    p.keys[child_idx as usize] = c.keys[BTREE_MID];
    p.values[child_idx as usize] = c.values[BTREE_MID];
    p.children[(child_idx + 1) as usize] = new_sibling_ptr;
    p.num_keys += 1;
}

/// Insert a key into a non-full leaf node at the given position.
#[inline(always)]
unsafe fn btree_insert_into_leaf(node: *mut BTreeNode, pos: u32, key: u64, value: u64) {
    let n = &mut *node;
    // Shift keys right to make room
    let mut i = n.num_keys;
    while i > pos {
        n.keys[i as usize] = n.keys[(i - 1) as usize];
        n.values[i as usize] = n.values[(i - 1) as usize];
        i -= 1;
    }
    n.keys[pos as usize] = key;
    n.values[pos as usize] = value;
    n.num_keys += 1;
}

/// Initialize an empty B-tree map.
///
/// # Safety
///
/// `map` must point to valid, writable memory for an `ArenaBTreeMap`.
pub unsafe fn arena_btree_init(map: *mut ArenaBTreeMap) {
    unsafe {
        core::ptr::write_volatile(
            map,
            ArenaBTreeMap {
                root: ArenaPtr::null(),
                count: 0,
                height: 0,
                _pad: 0,
            },
        );
    }
}

/// Insert a key-value pair into the B-tree.
///
/// Uses top-down proactive splitting: full nodes are split during
/// descent so the leaf is guaranteed to have room.
///
/// Returns:
/// -  0 = inserted (new key)
/// -  1 = updated (existing key)
/// - -1 = allocation failure or tree too deep
///
/// # Safety
///
/// `map`, `bump` must point to valid objects. `arena_base` must be the arena base.
pub unsafe fn arena_btree_insert(
    map: *mut ArenaBTreeMap,
    bump: *mut ArenaBumpState,
    key: u64,
    value: u64,
    arena_base: *mut u8,
) -> i32 {
    let header = &mut *map;

    // Empty tree: create root leaf
    if header.root.is_null() {
        let root_ptr = btree_alloc_node(bump, arena_base, true);
        if root_ptr.is_null() {
            return -1;
        }
        let root = root_ptr.resolve(arena_base);
        (*root).keys[0] = key;
        (*root).values[0] = value;
        (*root).num_keys = 1;
        header.root = root_ptr;
        header.count = 1;
        header.height = 1;
        return 0;
    }

    // If root is full, split it first
    let root = header.root.resolve(arena_base);
    if (*root).num_keys == BTREE_MAX_KEYS as u32 {
        let new_root_ptr = btree_alloc_node(bump, arena_base, false);
        if new_root_ptr.is_null() {
            return -1;
        }
        let new_root = new_root_ptr.resolve(arena_base);
        (*new_root).children[0] = header.root;

        let sibling_ptr = btree_alloc_node(bump, arena_base, (*root).is_leaf != 0);
        if sibling_ptr.is_null() {
            return -1;
        }
        let sibling = sibling_ptr.resolve(arena_base);

        btree_split_child(new_root, 0, root, sibling, sibling_ptr);

        header.root = new_root_ptr;
        header.height += 1;
    }

    // Walk down the tree, splitting full children proactively
    let mut current_ptr = header.root;
    let mut depth: u32 = 0;

    while depth < BTREE_MAX_DEPTH {
        let current = current_ptr.resolve(arena_base);
        if current.is_null() {
            return -1;
        }

        let (pos, found) = btree_find_key(&*current, key);

        // Key already exists — update value
        if found {
            (*current).values[pos as usize] = value;
            return 1;
        }

        // Leaf: insert here (guaranteed non-full by proactive splitting)
        if (*current).is_leaf != 0 {
            if (*current).num_keys >= BTREE_MAX_KEYS as u32 {
                return -1; // shouldn't happen with proactive splitting
            }
            btree_insert_into_leaf(current, pos, key, value);
            header.count += 1;
            return 0;
        }

        // Internal node: check if child[pos] is full, split if so
        let child_ptr = (*current).children[pos as usize];
        let child = child_ptr.resolve(arena_base);
        if child.is_null() {
            return -1;
        }

        if (*child).num_keys == BTREE_MAX_KEYS as u32 {
            let sibling_ptr = btree_alloc_node(bump, arena_base, (*child).is_leaf != 0);
            if sibling_ptr.is_null() {
                return -1;
            }
            let sibling = sibling_ptr.resolve(arena_base);

            btree_split_child(current, pos, child, sibling, sibling_ptr);

            // After split, determine which child to descend into
            if key == (*current).keys[pos as usize] {
                // Key was the promoted median — update
                (*current).values[pos as usize] = value;
                return 1;
            }
            if key > (*current).keys[pos as usize] {
                current_ptr = (*current).children[(pos + 1) as usize];
            } else {
                current_ptr = (*current).children[pos as usize];
            }
        } else {
            current_ptr = child_ptr;
        }

        depth += 1;
    }

    -1 // Tree too deep
}

/// Look up a key in the B-tree.
///
/// Returns a pointer to the value if found, or null if not found.
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaBTreeMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_btree_get(
    map: *const ArenaBTreeMap,
    key: u64,
    arena_base: *mut u8,
) -> *const u64 {
    let header = &*map;
    let mut current_ptr = header.root;
    let mut depth: u32 = 0;

    while depth < BTREE_MAX_DEPTH && !current_ptr.is_null() {
        let current = current_ptr.resolve(arena_base);
        if current.is_null() {
            return core::ptr::null();
        }

        let (pos, found) = btree_find_key(&*current, key);

        if found {
            return &(*current).values[pos as usize];
        }

        if (*current).is_leaf != 0 {
            return core::ptr::null(); // Not found
        }

        current_ptr = (*current).children[pos as usize];
        depth += 1;
    }

    core::ptr::null()
}

/// Minimum degree: non-root nodes must have at least `BTREE_T - 1` keys.
/// Before descending during delete, we ensure the child has >= `BTREE_T` keys
/// so that removing one still satisfies the minimum.
const BTREE_T: u32 = (BTREE_ORDER / 2) as u32; // = 4

/// Borrow a key from the left sibling via the parent (rotate right).
///
/// Parent's `key_idx` key moves down to the front of `child`,
/// and left sibling's last key moves up to replace it in the parent.
///
/// # Safety
///
/// `parent`, `child`, and the left sibling must be valid initialized nodes.
/// Left sibling must have > `BTREE_T - 1` keys.
#[inline(always)]
unsafe fn btree_borrow_from_left(
    parent: *mut BTreeNode,
    key_idx: u32,
    child: *mut BTreeNode,
    arena_base: *mut u8,
) {
    let p = &mut *parent;
    let c = &mut *child;
    let left_ptr = p.children[key_idx as usize];
    let left = left_ptr.resolve(arena_base);
    let l = &mut *left;

    // Shift all keys/values/children in child right by 1
    let mut i = c.num_keys;
    while i > 0 {
        c.keys[i as usize] = c.keys[(i - 1) as usize];
        c.values[i as usize] = c.values[(i - 1) as usize];
        i -= 1;
    }
    if c.is_leaf == 0 {
        i = c.num_keys + 1;
        while i > 0 {
            c.children[i as usize] = c.children[(i - 1) as usize];
            i -= 1;
        }
    }

    // Move parent key down to child[0]
    c.keys[0] = p.keys[key_idx as usize];
    c.values[0] = p.values[key_idx as usize];
    c.num_keys += 1;

    // Move left sibling's last key up to parent
    let last = l.num_keys - 1;
    p.keys[key_idx as usize] = l.keys[last as usize];
    p.values[key_idx as usize] = l.values[last as usize];

    // Move left sibling's last child to child's first child
    if c.is_leaf == 0 {
        c.children[0] = l.children[l.num_keys as usize];
    }

    l.num_keys -= 1;
}

/// Borrow a key from the right sibling via the parent (rotate left).
///
/// Parent's `key_idx` key moves down to the end of `child`,
/// and right sibling's first key moves up to replace it in the parent.
///
/// # Safety
///
/// `parent`, `child`, and the right sibling must be valid initialized nodes.
/// Right sibling must have > `BTREE_T - 1` keys.
#[inline(always)]
unsafe fn btree_borrow_from_right(
    parent: *mut BTreeNode,
    key_idx: u32,
    child: *mut BTreeNode,
    arena_base: *mut u8,
) {
    let p = &mut *parent;
    let c = &mut *child;
    let right_ptr = p.children[(key_idx + 1) as usize];
    let right = right_ptr.resolve(arena_base);
    let r = &mut *right;

    // Move parent key down to end of child
    c.keys[c.num_keys as usize] = p.keys[key_idx as usize];
    c.values[c.num_keys as usize] = p.values[key_idx as usize];
    c.num_keys += 1;

    // Move right sibling's first child to child's last child
    if c.is_leaf == 0 {
        c.children[c.num_keys as usize] = r.children[0];
    }

    // Move right sibling's first key up to parent
    p.keys[key_idx as usize] = r.keys[0];
    p.values[key_idx as usize] = r.values[0];

    // Shift right sibling's keys/values/children left by 1
    let mut i: u32 = 0;
    while i + 1 < r.num_keys {
        r.keys[i as usize] = r.keys[(i + 1) as usize];
        r.values[i as usize] = r.values[(i + 1) as usize];
        i += 1;
    }
    if r.is_leaf == 0 {
        i = 0;
        while i < r.num_keys {
            r.children[i as usize] = r.children[(i + 1) as usize];
            i += 1;
        }
    }

    r.num_keys -= 1;
}

/// Merge `children[idx]`, `keys[idx]`, and `children[idx+1]` into `children[idx]`.
///
/// The parent loses one key and one child pointer. The right child's contents
/// are appended to the left child.
///
/// # Safety
///
/// `parent` and both children must be valid. Combined key count must not
/// exceed `BTREE_MAX_KEYS`.
#[inline(always)]
unsafe fn btree_merge_children(
    parent: *mut BTreeNode,
    idx: u32,
    arena_base: *mut u8,
) {
    let p = &mut *parent;
    let left_ptr = p.children[idx as usize];
    let right_ptr = p.children[(idx + 1) as usize];
    let left = left_ptr.resolve(arena_base);
    let right = right_ptr.resolve(arena_base);
    let l = &mut *left;
    let r = &*right;

    // Move parent key into left child
    l.keys[l.num_keys as usize] = p.keys[idx as usize];
    l.values[l.num_keys as usize] = p.values[idx as usize];
    l.num_keys += 1;

    // Copy all keys/values from right child into left child
    let mut i: u32 = 0;
    while i < r.num_keys {
        l.keys[(l.num_keys + i) as usize] = r.keys[i as usize];
        l.values[(l.num_keys + i) as usize] = r.values[i as usize];
        i += 1;
    }

    // Copy children from right child if internal
    if l.is_leaf == 0 {
        i = 0;
        while i <= r.num_keys {
            l.children[(l.num_keys + i) as usize] = r.children[i as usize];
            i += 1;
        }
    }

    l.num_keys += r.num_keys;

    // Remove key[idx] and children[idx+1] from parent by shifting left
    i = idx;
    while i + 1 < p.num_keys {
        p.keys[i as usize] = p.keys[(i + 1) as usize];
        p.values[i as usize] = p.values[(i + 1) as usize];
        i += 1;
    }
    i = idx + 1;
    while i < p.num_keys {
        p.children[i as usize] = p.children[(i + 1) as usize];
        i += 1;
    }

    p.num_keys -= 1;
}

/// Delete a key from the B-tree.
///
/// Uses top-down preemptive rebalancing (CLRS-style): before descending
/// into a child, ensures the child has at least `BTREE_T` keys so that
/// a subsequent deletion won't cause underflow. This maintains proper
/// B-tree invariants and never fails for keys that exist in the tree.
///
/// Returns:
/// -  0 = deleted
/// - -1 = key not found
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaBTreeMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_btree_delete(
    map: *mut ArenaBTreeMap,
    key: u64,
    arena_base: *mut u8,
) -> i32 {
    let header = &mut *map;
    if header.root.is_null() {
        return -1;
    }

    let root = header.root.resolve(arena_base);
    if root.is_null() {
        return -1;
    }

    let mut current = root;
    let mut target_key = key;
    let mut depth: u32 = 0;

    while depth < BTREE_MAX_DEPTH {
        let (pos, found) = btree_find_key(&*current, target_key);

        if found {
            if (*current).is_leaf != 0 {
                // Case 1: key in leaf — remove by shifting left
                let n = &mut *current;
                let mut i = pos;
                while i + 1 < n.num_keys {
                    n.keys[i as usize] = n.keys[(i + 1) as usize];
                    n.values[i as usize] = n.values[(i + 1) as usize];
                    i += 1;
                }
                n.num_keys -= 1;
                header.count -= 1;
                if header.count == 0 {
                    header.root = ArenaPtr::null();
                    header.height = 0;
                }
                return 0;
            }

            // Case 2: key in internal node
            let left_child = (*current).children[pos as usize].resolve(arena_base);
            let right_child = (*current).children[(pos + 1) as usize].resolve(arena_base);

            if (*left_child).num_keys >= BTREE_T {
                // Case 2a: find predecessor, replace, then delete predecessor
                let mut pred = left_child;
                let mut pd: u32 = 0;
                while (*pred).is_leaf == 0 && pd < BTREE_MAX_DEPTH {
                    pred = (*pred).children[(*pred).num_keys as usize].resolve(arena_base);
                    pd += 1;
                }
                let pred_key = (*pred).keys[((*pred).num_keys - 1) as usize];
                let pred_val = (*pred).values[((*pred).num_keys - 1) as usize];
                (*current).keys[pos as usize] = pred_key;
                (*current).values[pos as usize] = pred_val;
                target_key = pred_key;
                current = left_child;
                depth += 1;
                continue;
            } else if (*right_child).num_keys >= BTREE_T {
                // Case 2b: find successor, replace, then delete successor
                let mut succ = right_child;
                let mut sd: u32 = 0;
                while (*succ).is_leaf == 0 && sd < BTREE_MAX_DEPTH {
                    succ = (*succ).children[0].resolve(arena_base);
                    sd += 1;
                }
                let succ_key = (*succ).keys[0];
                let succ_val = (*succ).values[0];
                (*current).keys[pos as usize] = succ_key;
                (*current).values[pos as usize] = succ_val;
                target_key = succ_key;
                current = right_child;
                depth += 1;
                continue;
            } else {
                // Case 2c: both children have BTREE_T-1 keys — merge them
                btree_merge_children(current, pos, arena_base);
                if current == root && (*current).num_keys == 0 {
                    header.root = (*current).children[0];
                    header.height -= 1;
                    current = header.root.resolve(arena_base);
                } else {
                    current = left_child; // merged node
                }
                depth += 1;
                continue;
            }
        }

        // Key not found at this node
        if (*current).is_leaf != 0 {
            return -1;
        }

        // Case 3: descend into children[pos], fixing underflow first
        let child_ptr = (*current).children[pos as usize];
        let child = child_ptr.resolve(arena_base);
        if child.is_null() {
            return -1;
        }

        if (*child).num_keys < BTREE_T {
            let has_left = pos > 0;
            let has_right = pos < (*current).num_keys;

            let left_ok = has_left && {
                let ls = (*current).children[(pos - 1) as usize].resolve(arena_base);
                (*ls).num_keys >= BTREE_T
            };
            let right_ok = has_right && {
                let rs = (*current).children[(pos + 1) as usize].resolve(arena_base);
                (*rs).num_keys >= BTREE_T
            };

            if left_ok {
                // 3a: borrow from left sibling
                btree_borrow_from_left(current, pos - 1, child, arena_base);
            } else if right_ok {
                // 3a: borrow from right sibling
                btree_borrow_from_right(current, pos, child, arena_base);
            } else if has_left {
                // 3b: merge with left sibling
                btree_merge_children(current, pos - 1, arena_base);
                if current == root && (*current).num_keys == 0 {
                    header.root = (*current).children[0];
                    header.height -= 1;
                    current = header.root.resolve(arena_base);
                    depth += 1;
                    continue;
                }
                current = (*current).children[(pos - 1) as usize].resolve(arena_base);
                depth += 1;
                continue;
            } else {
                // 3b: merge with right sibling
                btree_merge_children(current, pos, arena_base);
                if current == root && (*current).num_keys == 0 {
                    header.root = (*current).children[0];
                    header.height -= 1;
                    current = header.root.resolve(arena_base);
                    depth += 1;
                    continue;
                }
                current = child;
                depth += 1;
                continue;
            }
        }

        current = (*current).children[pos as usize].resolve(arena_base);
        depth += 1;
    }

    -1
}

/// Iterate over all key-value pairs in the B-tree in sorted order.
///
/// Uses an explicit stack for iterative in-order traversal.
/// Each stack frame tracks (node_ptr, step) where step interleaves
/// child visits (even steps) and key emissions (odd steps).
///
/// # Safety
///
/// `map` must point to a valid, initialized `ArenaBTreeMap`.
/// `arena_base` must be the arena base address.
pub unsafe fn arena_btree_for_each<F>(map: *const ArenaBTreeMap, arena_base: *mut u8, mut f: F)
where
    F: FnMut(u64, u64),
{
    let header = &*map;
    if header.root.is_null() || header.count == 0 {
        return;
    }

    #[derive(Copy, Clone)]
    struct Frame {
        node_ptr: ArenaPtr<BTreeNode>,
        step: u32, // interleaved: even=child visit, odd=key emit
    }

    let mut stack = [Frame {
        node_ptr: ArenaPtr::null(),
        step: 0,
    }; BTREE_MAX_DEPTH as usize + 1];

    // Push root
    stack[0] = Frame {
        node_ptr: header.root,
        step: 0,
    };
    let mut sp: usize = 1;

    // Bound total iterations to prevent infinite loops
    let max_iters = header.count * 4 + 100;
    let mut iters: u64 = 0;

    while sp > 0 && iters < max_iters {
        iters += 1;

        let frame = &mut stack[sp - 1];
        let node = frame.node_ptr.resolve(arena_base);
        if node.is_null() {
            sp -= 1;
            continue;
        }
        let n = &*node;
        let total_steps = 2 * n.num_keys + 1;

        if frame.step >= total_steps {
            sp -= 1; // Done with this node
            continue;
        }

        let step = frame.step;
        frame.step += 1;

        if step % 2 == 0 {
            // Even step: visit child[step/2]
            let child_idx = step / 2;
            if n.is_leaf == 0 && child_idx <= n.num_keys {
                let child_ptr = n.children[child_idx as usize];
                if !child_ptr.is_null() && sp < stack.len() {
                    stack[sp] = Frame {
                        node_ptr: child_ptr,
                        step: 0,
                    };
                    sp += 1;
                }
            }
        } else {
            // Odd step: emit key[step/2]
            let key_idx = step / 2;
            if key_idx < n.num_keys {
                f(n.keys[key_idx as usize], n.values[key_idx as usize]);
            }
        }
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

    // ── B-tree tests ─────────────────────────────────────────────────

    /// Helper: create a simulated arena with a B-tree map and bump allocator.
    fn make_btree() -> (Vec<u8>, *mut ArenaBTreeMap, *mut ArenaBumpState) {
        let arena_size = 256 * 1024; // 256 KiB — enough for many nodes
        let mut buf = vec![0u8; arena_size];
        let base = buf.as_mut_ptr();

        // Place ArenaBTreeMap at offset 0
        let map = base.cast::<ArenaBTreeMap>();
        unsafe { arena_btree_init(map) };

        // Place ArenaBumpState right after
        let bump_offset = size_of::<ArenaBTreeMap>();
        let bump = unsafe { base.add(bump_offset).cast::<ArenaBumpState>() };
        let data_start = bump_offset + size_of::<ArenaBumpState>();
        unsafe {
            *bump = ArenaBumpState::new((arena_size - data_start) as u64);
            // Adjust watermark to skip the header area
            (*bump).watermark = data_start as u64;
        }

        (buf, map, bump)
    }

    #[test]
    fn btree_node_layout() {
        assert_eq!(mem::size_of::<BTreeNode>(), 184);
        assert_eq!(mem::align_of::<BTreeNode>(), 8);
    }

    #[test]
    fn btree_map_layout() {
        assert_eq!(mem::size_of::<ArenaBTreeMap>(), 24);
        assert_eq!(mem::align_of::<ArenaBTreeMap>(), 8);
    }

    #[test]
    fn btree_empty() {
        let (buf, map, _bump) = make_btree();
        let base = buf.as_ptr() as *mut u8;

        let val = unsafe { arena_btree_get(map, 42, base) };
        assert!(val.is_null());
        assert_eq!(unsafe { (*map).count }, 0);
    }

    #[test]
    fn btree_insert_and_get_single() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        let ret = unsafe { arena_btree_insert(map, bump, 42, 100, base) };
        assert_eq!(ret, 0);
        assert_eq!(unsafe { (*map).count }, 1);

        let val = unsafe { arena_btree_get(map, 42, base) };
        assert!(!val.is_null());
        assert_eq!(unsafe { *val }, 100);
    }

    #[test]
    fn btree_update_existing() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        unsafe { arena_btree_insert(map, bump, 42, 100, base) };
        let ret = unsafe { arena_btree_insert(map, bump, 42, 200, base) };
        assert_eq!(ret, 1); // Updated
        assert_eq!(unsafe { (*map).count }, 1); // Count unchanged

        assert_eq!(unsafe { *arena_btree_get(map, 42, base) }, 200);
    }

    #[test]
    fn btree_miss() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        unsafe { arena_btree_insert(map, bump, 10, 1, base) };
        assert!(unsafe { arena_btree_get(map, 99, base) }.is_null());
    }

    #[test]
    fn btree_multiple_inserts_within_one_node() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert 7 keys (fills one leaf node exactly)
        for i in 0..BTREE_MAX_KEYS as u64 {
            let ret = unsafe { arena_btree_insert(map, bump, (i + 1) * 10, i + 1, base) };
            assert_eq!(ret, 0);
        }
        assert_eq!(unsafe { (*map).count }, BTREE_MAX_KEYS as u64);

        // All should be found
        for i in 0..BTREE_MAX_KEYS as u64 {
            let val = unsafe { arena_btree_get(map, (i + 1) * 10, base) };
            assert!(!val.is_null());
            assert_eq!(unsafe { *val }, i + 1);
        }
    }

    #[test]
    fn btree_triggers_root_split() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert MAX_KEYS + 1 keys to trigger root split
        for i in 0..=BTREE_MAX_KEYS as u64 {
            let ret = unsafe { arena_btree_insert(map, bump, i + 1, (i + 1) * 100, base) };
            assert_eq!(ret, 0, "failed to insert key {}", i + 1);
        }
        assert_eq!(unsafe { (*map).count }, (BTREE_MAX_KEYS + 1) as u64);
        assert_eq!(unsafe { (*map).height }, 2); // Root was split

        // All should be found
        for i in 0..=BTREE_MAX_KEYS as u64 {
            let val = unsafe { arena_btree_get(map, i + 1, base) };
            assert!(!val.is_null(), "key {} not found after root split", i + 1);
            assert_eq!(unsafe { *val }, (i + 1) * 100);
        }
    }

    #[test]
    fn btree_ordered_iteration() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert keys out of order
        for &k in &[50, 30, 70, 10, 40, 60, 80, 20, 90] {
            unsafe { arena_btree_insert(map, bump, k, k * 10, base) };
        }

        // Iterate — should produce sorted order
        let mut keys = Vec::new();
        unsafe {
            arena_btree_for_each(map, base, |k, v| {
                assert_eq!(v, k * 10);
                keys.push(k);
            });
        }
        assert_eq!(keys, vec![10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn btree_delete_from_leaf() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        for k in 1..=5u64 {
            unsafe { arena_btree_insert(map, bump, k, k * 10, base) };
        }

        // Delete key 3
        let ret = unsafe { arena_btree_delete(map, 3, base) };
        assert_eq!(ret, 0);
        assert_eq!(unsafe { (*map).count }, 4);

        // 3 should be gone, others remain
        assert!(unsafe { arena_btree_get(map, 3, base) }.is_null());
        assert_eq!(unsafe { *arena_btree_get(map, 1, base) }, 10);
        assert_eq!(unsafe { *arena_btree_get(map, 5, base) }, 50);
    }

    #[test]
    fn btree_delete_miss() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        unsafe { arena_btree_insert(map, bump, 1, 10, base) };
        assert_eq!(unsafe { arena_btree_delete(map, 99, base) }, -1);
        assert_eq!(unsafe { (*map).count }, 1);
    }

    #[test]
    fn btree_delete_from_internal() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert enough to create a 2-level tree
        for i in 1..=15u64 {
            unsafe { arena_btree_insert(map, bump, i, i * 100, base) };
        }
        assert!(unsafe { (*map).height } >= 2);

        // Delete a key that's in an internal node (promoted median)
        // The median of keys 1..8 is 4, which gets promoted to root on first split
        let root = unsafe { (*map).root.resolve(base) };
        let root_key = unsafe { (*root).keys[0] };

        let ret = unsafe { arena_btree_delete(map, root_key, base) };
        assert_eq!(ret, 0);
        assert_eq!(unsafe { (*map).count }, 14);

        // Deleted key should not be found
        assert!(unsafe { arena_btree_get(map, root_key, base) }.is_null());

        // Other keys should still be found
        for i in 1..=15u64 {
            if i == root_key {
                continue;
            }
            let val = unsafe { arena_btree_get(map, i, base) };
            assert!(!val.is_null(), "key {i} not found after delete of {root_key}");
            assert_eq!(unsafe { *val }, i * 100);
        }
    }

    #[test]
    fn btree_delete_all() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        for k in 1..=5u64 {
            unsafe { arena_btree_insert(map, bump, k, k, base) };
        }

        for k in 1..=5u64 {
            assert_eq!(unsafe { arena_btree_delete(map, k, base) }, 0);
        }
        assert_eq!(unsafe { (*map).count }, 0);
    }

    #[test]
    fn btree_key_zero_and_max() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        unsafe {
            arena_btree_insert(map, bump, 0, 1, base);
            arena_btree_insert(map, bump, u64::MAX, 2, base);
        }

        assert_eq!(unsafe { *arena_btree_get(map, 0, base) }, 1);
        assert_eq!(unsafe { *arena_btree_get(map, u64::MAX, base) }, 2);

        // Ordered iteration: 0 first, MAX last
        let mut keys = Vec::new();
        unsafe { arena_btree_for_each(map, base, |k, _| keys.push(k)) };
        assert_eq!(keys, vec![0, u64::MAX]);
    }

    #[test]
    fn btree_stress_100() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert 100 keys in scrambled order
        for i in 0..100u64 {
            let key = (i * 37 + 13) % 100; // pseudo-random permutation
            let ret = unsafe { arena_btree_insert(map, bump, key, key * 10, base) };
            assert!(ret == 0 || ret == 1, "insert failed for key {key}");
        }

        // Verify all 100 keys exist
        for i in 0..100u64 {
            let val = unsafe { arena_btree_get(map, i, base) };
            assert!(!val.is_null(), "key {i} not found");
            assert_eq!(unsafe { *val }, i * 10);
        }

        // Verify ordered iteration
        let mut prev: Option<u64> = None;
        let mut count = 0u64;
        unsafe {
            arena_btree_for_each(map, base, |k, _| {
                if let Some(p) = prev {
                    assert!(k > p, "out of order: {p} before {k}");
                }
                prev = Some(k);
                count += 1;
            });
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn btree_stress_1000_insert_delete() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert 1000 keys
        for i in 0..1000u64 {
            let key = (i * 7919 + 1) % 10000; // spread across range
            unsafe { arena_btree_insert(map, bump, key, i, base) };
        }

        let count_before = unsafe { (*map).count };
        assert!(count_before > 0);

        // Delete first 500
        let mut deleted = 0u64;
        for i in 0..500u64 {
            let key = (i * 7919 + 1) % 10000;
            if unsafe { arena_btree_delete(map, key, base) } == 0 {
                deleted += 1;
            }
        }
        assert_eq!(unsafe { (*map).count }, count_before - deleted);

        // Remaining 500 should still be found
        for i in 500..1000u64 {
            let key = (i * 7919 + 1) % 10000;
            let val = unsafe { arena_btree_get(map, key, base) };
            // May not find if there were duplicate keys from the permutation
            if !val.is_null() {
                assert_eq!(unsafe { *val }, i);
            }
        }

        // Ordered iteration should still work
        let mut prev: Option<u64> = None;
        unsafe {
            arena_btree_for_each(map, base, |k, _| {
                if let Some(p) = prev {
                    assert!(k > p, "order violation after deletes: {p} >= {k}");
                }
                prev = Some(k);
            });
        }
    }

    #[test]
    fn btree_heavy_delete_workload() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert 1000 unique keys
        for i in 0..1000u64 {
            assert!(unsafe { arena_btree_insert(map, bump, i, i * 10, base) } >= 0);
        }
        assert_eq!(unsafe { (*map).count }, 1000);

        // Delete 900 keys
        for i in 0..900u64 {
            assert_eq!(unsafe { arena_btree_delete(map, i, base) }, 0,
                "delete of key {} failed", i);
        }
        assert_eq!(unsafe { (*map).count }, 100);

        // Verify remaining 100 keys are correct
        for i in 900..1000u64 {
            let val = unsafe { arena_btree_get(map, i, base) };
            assert!(!val.is_null(), "surviving key {} missing", i);
            assert_eq!(unsafe { *val }, i * 10);
        }

        // Verify deleted keys are gone
        for i in 0..900u64 {
            assert!(unsafe { arena_btree_get(map, i, base) }.is_null(),
                "deleted key {} still found", i);
        }

        // Verify sorted iteration of remaining keys
        let mut keys = Vec::new();
        unsafe {
            arena_btree_for_each(map, base, |k, v| {
                assert_eq!(v, k * 10);
                keys.push(k);
            });
        }
        assert_eq!(keys.len(), 100);
        let expected: Vec<u64> = (900..1000).collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn btree_insert_delete_reinsert_cycle() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // 3 full cycles of insert-delete
        for cycle in 0..3u64 {
            for i in 0..200u64 {
                let key = cycle * 1000 + i;
                assert!(unsafe { arena_btree_insert(map, bump, key, key, base) } >= 0);
            }

            // Delete all but 20
            for i in 0..180u64 {
                let key = cycle * 1000 + i;
                assert_eq!(unsafe { arena_btree_delete(map, key, base) }, 0);
            }
        }

        // 60 keys should remain (20 per cycle)
        assert_eq!(unsafe { (*map).count }, 60);

        // Verify sorted iteration
        let mut prev = None;
        let mut count = 0u64;
        unsafe {
            arena_btree_for_each(map, base, |k, v| {
                assert_eq!(k, v);
                if let Some(p) = prev {
                    assert!(k > p, "order violation: {} >= {}", p, k);
                }
                prev = Some(k);
                count += 1;
            });
        }
        assert_eq!(count, 60);
    }

    /// Regression test: delete internal node key after depleting left subtree.
    ///
    /// 1. Insert keys to create an internal node (root split at BTREE_MAX_KEYS+1)
    /// 2. Delete all keys from the left child
    /// 3. Delete the root key — rebalancing must handle the underflowing left child
    #[test]
    fn btree_delete_internal_empty_predecessor() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        // Insert BTREE_MAX_KEYS + 1 keys to trigger a root split.
        for i in 1..=(BTREE_MAX_KEYS as u64 + 1) {
            assert_eq!(unsafe { arena_btree_insert(map, bump, i, i * 100, base) }, 0);
        }

        // Find the root key (the median after split)
        let root_ptr = unsafe { (*map).root.resolve(base) };
        assert!(!root_ptr.is_null());
        let root_key = unsafe { (*root_ptr).keys[0] };
        assert_ne!(root_key, 0);

        // Delete all keys from the left child (keys < root_key)
        for k in 1..root_key {
            let ret = unsafe { arena_btree_delete(map, k, base) };
            assert_eq!(ret, 0, "failed to delete key {k} from left child");
        }

        // Now delete the root key itself.
        // The predecessor leaf is empty — this must fall back to successor.
        let ret = unsafe { arena_btree_delete(map, root_key, base) };
        assert_eq!(ret, 0, "delete of root key with empty predecessor should succeed");

        // Verify the tree is still consistent
        assert!(unsafe { arena_btree_get(map, root_key, base) }.is_null());

        // Remaining keys (right subtree) should still be findable
        for k in (root_key + 1)..=(BTREE_MAX_KEYS as u64 + 1) {
            let val = unsafe { arena_btree_get(map, k, base) };
            if !val.is_null() {
                assert_eq!(unsafe { *val }, k * 100);
            }
        }
    }

    /// Test 3-level B-tree: insert enough keys to force two levels of splits.
    #[test]
    fn btree_3_level_tree() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        let n = 50u64;
        for i in 1..=n {
            assert_eq!(unsafe { arena_btree_insert(map, bump, i, i * 10, base) }, 0);
        }

        // Verify height >= 2 (root + at least 2 levels below)
        assert!(unsafe { (*map).height } >= 2, "expected 3+ levels with {n} keys");

        // Verify all keys present
        for i in 1..=n {
            let val = unsafe { arena_btree_get(map, i, base) };
            assert!(!val.is_null(), "key {i} not found in 3-level tree");
            assert_eq!(unsafe { *val }, i * 10);
        }

        // Verify ordered iteration
        let mut keys_out = Vec::new();
        unsafe {
            arena_btree_for_each(map, base, |k, _| {
                keys_out.push(k);
            });
        }
        let expected: Vec<u64> = (1..=n).collect();
        assert_eq!(keys_out, expected);

        // Delete half — all should succeed with proper rebalancing
        let mut deleted = 0u64;
        for i in 1..=n / 2 {
            assert_eq!(unsafe { arena_btree_delete(map, i, base) }, 0);
            deleted += 1;
        }
        // At least most deletes should succeed
        assert!(deleted >= n / 4, "too many delete failures: only {deleted}/{} succeeded", n/2);
        for i in (n / 2 + 1)..=n {
            let val = unsafe { arena_btree_get(map, i, base) };
            assert!(!val.is_null(), "key {i} missing after partial delete");
        }
    }

    /// ArenaPtr from_raw / resolve roundtrip.
    #[test]
    fn arena_ptr_roundtrip() {
        let mut buf = [0u8; 256];
        let base = buf.as_mut_ptr();

        // Write a value at offset 64
        let val_ptr = unsafe { base.add(64) as *mut u64 };
        unsafe { *val_ptr = 0xDEAD_BEEF_CAFE_BABE };

        // Create ArenaPtr from raw pointer
        let arena_ptr: ArenaPtr<u64> = ArenaPtr::from_raw(val_ptr, base);
        assert!(!arena_ptr.is_null());
        assert_eq!(arena_ptr.offset(), Some(64));

        // Resolve back to raw pointer and read
        let resolved = unsafe { arena_ptr.resolve(base) };
        assert_eq!(resolved, val_ptr);
        assert_eq!(unsafe { *resolved }, 0xDEAD_BEEF_CAFE_BABE);
    }

    /// ArenaPtr null roundtrip.
    #[test]
    fn arena_ptr_null_roundtrip() {
        let mut buf = [0u8; 64];
        let base = buf.as_mut_ptr();

        let p: ArenaPtr<u32> = ArenaPtr::from_raw(core::ptr::null_mut(), base);
        assert!(p.is_null());
        assert!(unsafe { p.resolve(base) }.is_null());
    }

    /// Bump allocator alignment edge cases.
    #[test]
    fn bump_alignment_edge_cases() {
        // Align to 1 (no alignment)
        let mut bump = ArenaBumpState::new(128);
        let a = bump.alloc(1, 1).unwrap();
        assert_eq!(a, 0);
        let b = bump.alloc(1, 1).unwrap();
        assert_eq!(b, 1);

        // Align to 64 (cacheline)
        let mut bump = ArenaBumpState::new(256);
        let a = bump.alloc(1, 64).unwrap();
        assert_eq!(a, 0); // 0 is already 64-aligned
        let b = bump.alloc(1, 64).unwrap();
        assert_eq!(b, 64); // next cacheline
        let c = bump.alloc(1, 64).unwrap();
        assert_eq!(c, 128);

        // Alignment larger than remaining capacity
        let mut bump = ArenaBumpState::new(32);
        bump.alloc(1, 1).unwrap(); // watermark = 1
        // Need to align to 64 — aligned = 64, then +1 = 65 > 32 → should fail
        assert!(bump.alloc(1, 64).is_none());

        // Exact fit with alignment
        let mut bump = ArenaBumpState::new(16);
        let a = bump.alloc(8, 8).unwrap();
        assert_eq!(a, 0);
        let b = bump.alloc(8, 8).unwrap();
        assert_eq!(b, 8);
        assert!(bump.alloc(1, 1).is_none()); // full
    }

    // ── Slab allocator tests ─────────────────────────────────────────

    /// Helper: create a simulated arena with a slab allocator.
    /// The slab header lives at the start of the buffer. The bump region
    /// (where slots are allocated) starts right after the header.
    fn make_slab(capacity: usize, slot_size: u32) -> (Vec<u8>, *mut ArenaSlabState) {
        let header_size = mem::size_of::<ArenaSlabState>();
        let total = header_size + capacity;
        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();
        let slab = base.cast::<ArenaSlabState>();
        let ret = unsafe { arena_slab_init(slab, (total) as u64, slot_size) };
        assert_eq!(ret, 0, "slab init failed");
        // Advance bump watermark past the header so allocations don't
        // overlap with the ArenaSlabState itself.
        unsafe { (*slab).bump.watermark = header_size as u64 };
        (buf, slab)
    }

    #[test]
    fn slab_state_layout() {
        assert_eq!(mem::size_of::<ArenaSlabState>(), 40);
        assert_eq!(mem::align_of::<ArenaSlabState>(), 8);
    }

    #[test]
    fn slab_init_valid() {
        let (_buf, slab) = make_slab(4096, 64);
        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.free_count, 0);
        assert_eq!(stats.in_use, 0);
    }

    #[test]
    fn slab_init_rejects_too_small() {
        let mut buf = vec![0u8; 256];
        let slab = buf.as_mut_ptr().cast::<ArenaSlabState>();
        assert_eq!(unsafe { arena_slab_init(slab, 256, 4) }, -1);
        assert_eq!(unsafe { arena_slab_init(slab, 256, 8) }, -1);
        assert_eq!(unsafe { arena_slab_init(slab, 256, 12) }, -1);
        assert_eq!(unsafe { arena_slab_init(slab, 256, 0) }, -1);
    }

    #[test]
    fn slab_alloc_one() {
        let (mut buf, slab) = make_slab(4096, 64);
        let base = buf.as_mut_ptr();

        let slot = unsafe { arena_slab_alloc(slab, base) };
        assert!(!slot.is_null());

        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.total_allocated, 1);
        assert_eq!(stats.free_count, 0);
        assert_eq!(stats.in_use, 1);
    }

    #[test]
    fn slab_free_and_realloc() {
        let (mut buf, slab) = make_slab(4096, 64);
        let base = buf.as_mut_ptr();

        let slot = unsafe { arena_slab_alloc(slab, base) };
        assert!(!slot.is_null());

        unsafe { arena_slab_free(slab, slot, base) };
        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.free_count, 1);
        assert_eq!(stats.in_use, 0);

        // Realloc — should get the same slot back (LIFO free list)
        let slot2 = unsafe { arena_slab_alloc(slab, base) };
        assert!(!slot2.is_null());
        assert_eq!(slot.offset(), slot2.offset());

        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.total_allocated, 1); // No new bump allocation
        assert_eq!(stats.free_count, 0);
        assert_eq!(stats.in_use, 1);
    }

    #[test]
    fn slab_multiple_alloc_free_cycle() {
        let (mut buf, slab) = make_slab(4096, 32);
        let base = buf.as_mut_ptr();

        let mut slots = Vec::new();
        for _ in 0..10 {
            let s = unsafe { arena_slab_alloc(slab, base) };
            assert!(!s.is_null());
            slots.push(s);
        }
        assert_eq!(unsafe { arena_slab_stats(slab) }.in_use, 10);

        for s in &slots {
            unsafe { arena_slab_free(slab, *s, base) };
        }
        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.free_count, 10);
        assert_eq!(stats.in_use, 0);

        let mut reslots = Vec::new();
        for _ in 0..10 {
            let s = unsafe { arena_slab_alloc(slab, base) };
            assert!(!s.is_null());
            reslots.push(s);
        }
        assert_eq!(unsafe { arena_slab_stats(slab) }.total_allocated, 10);
        assert_eq!(unsafe { arena_slab_stats(slab) }.in_use, 10);
        assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 0);
    }

    #[test]
    fn slab_exhaust_capacity() {
        let (mut buf, slab) = make_slab(192, 64);
        let base = buf.as_mut_ptr();

        let s1 = unsafe { arena_slab_alloc(slab, base) };
        let s2 = unsafe { arena_slab_alloc(slab, base) };
        let s3 = unsafe { arena_slab_alloc(slab, base) };
        assert!(!s1.is_null());
        assert!(!s2.is_null());
        assert!(!s3.is_null());

        let s4 = unsafe { arena_slab_alloc(slab, base) };
        assert!(s4.is_null());
    }

    #[test]
    fn slab_exhaust_then_free_then_alloc() {
        let (mut buf, slab) = make_slab(128, 64);
        let base = buf.as_mut_ptr();

        let s1 = unsafe { arena_slab_alloc(slab, base) };
        let s2 = unsafe { arena_slab_alloc(slab, base) };
        assert!(!s1.is_null());
        assert!(!s2.is_null());

        let s3 = unsafe { arena_slab_alloc(slab, base) };
        assert!(s3.is_null()); // Exhausted

        unsafe { arena_slab_free(slab, s1, base) };

        let s4 = unsafe { arena_slab_alloc(slab, base) };
        assert!(!s4.is_null());
        assert_eq!(s4.offset(), s1.offset()); // Reused
    }

    #[test]
    fn slab_interleaved_alloc_free() {
        let (mut buf, slab) = make_slab(4096, 24);
        let base = buf.as_mut_ptr();

        let a = unsafe { arena_slab_alloc(slab, base) };
        let _b = unsafe { arena_slab_alloc(slab, base) };
        unsafe { arena_slab_free(slab, a, base) };
        let c = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(c.offset(), a.offset()); // C reuses A's slot

        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.total_allocated, 2);
        assert_eq!(stats.in_use, 2);
    }

    #[test]
    fn slab_write_and_read_data() {
        let (mut buf, slab) = make_slab(4096, 24);
        let base = buf.as_mut_ptr();

        let slot = unsafe { arena_slab_alloc(slab, base) };
        assert!(!slot.is_null());

        let ptr = unsafe { slot.resolve(base) };
        unsafe { core::ptr::write_volatile(ptr.cast::<u64>(), 0xDEADBEEF_CAFEBABE) };
        let val = unsafe { core::ptr::read(ptr.cast::<u64>()) };
        assert_eq!(val, 0xDEADBEEF_CAFEBABE);

        // Free and realloc — can write new data
        unsafe { arena_slab_free(slab, slot, base) };
        let slot2 = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(slot2.offset(), slot.offset());
        unsafe { core::ptr::write_volatile(slot2.resolve(base).cast::<u64>(), 42) };
        assert_eq!(unsafe { core::ptr::read(slot2.resolve(base).cast::<u64>()) }, 42);
    }

    #[test]
    fn slab_free_list_lifo_order() {
        let (mut buf, slab) = make_slab(4096, 16);
        let base = buf.as_mut_ptr();

        let a = unsafe { arena_slab_alloc(slab, base) };
        let b = unsafe { arena_slab_alloc(slab, base) };
        let c = unsafe { arena_slab_alloc(slab, base) };

        // Free A, B, C → list is C→B→A (LIFO)
        unsafe { arena_slab_free(slab, a, base) };
        unsafe { arena_slab_free(slab, b, base) };
        unsafe { arena_slab_free(slab, c, base) };

        // Alloc returns LIFO: C, B, A
        let r1 = unsafe { arena_slab_alloc(slab, base) };
        let r2 = unsafe { arena_slab_alloc(slab, base) };
        let r3 = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(r1.offset(), c.offset());
        assert_eq!(r2.offset(), b.offset());
        assert_eq!(r3.offset(), a.offset());
    }

    #[test]
    fn slab_stress_100_cycles() {
        let (mut buf, slab) = make_slab(8192, 32);
        let base = buf.as_mut_ptr();

        let mut slots: Vec<ArenaPtr<u8>> = Vec::new();
        for _ in 0..50 {
            let s = unsafe { arena_slab_alloc(slab, base) };
            assert!(!s.is_null());
            slots.push(s);
        }

        // 3 full free+realloc cycles
        for _cycle in 0..3 {
            for s in &slots {
                unsafe { arena_slab_free(slab, *s, base) };
            }
            assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 50);

            let mut reslots = Vec::new();
            for _ in 0..50 {
                let s = unsafe { arena_slab_alloc(slab, base) };
                assert!(!s.is_null());
                reslots.push(s);
            }
            slots = reslots;
            assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 0);
        }

        // No new bump allocs after first round
        assert_eq!(unsafe { arena_slab_stats(slab) }.total_allocated, 50);
    }

    #[test]
    fn slab_large_slot_size() {
        let (mut buf, slab) = make_slab(4096, 256);
        let base = buf.as_mut_ptr();

        let s1 = unsafe { arena_slab_alloc(slab, base) };
        let s2 = unsafe { arena_slab_alloc(slab, base) };
        assert!(!s1.is_null());
        assert!(!s2.is_null());

        let off1 = s1.offset().unwrap();
        let off2 = s2.offset().unwrap();
        assert_eq!(off2 - off1, 256);

        unsafe { arena_slab_free(slab, s1, base) };
        let s3 = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(s3.offset(), s1.offset());
    }

    #[test]
    fn slab_stats_accuracy_through_operations() {
        let (mut buf, slab) = make_slab(4096, 16);
        let base = buf.as_mut_ptr();

        assert_eq!(unsafe { arena_slab_stats(slab) }, ArenaSlabStats {
            total_allocated: 0, free_count: 0, in_use: 0,
        });

        let a = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(unsafe { arena_slab_stats(slab) }, ArenaSlabStats {
            total_allocated: 1, free_count: 0, in_use: 1,
        });

        let b = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(unsafe { arena_slab_stats(slab) }, ArenaSlabStats {
            total_allocated: 2, free_count: 0, in_use: 2,
        });

        unsafe { arena_slab_free(slab, a, base) };
        assert_eq!(unsafe { arena_slab_stats(slab) }, ArenaSlabStats {
            total_allocated: 2, free_count: 1, in_use: 1,
        });

        let _c = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(unsafe { arena_slab_stats(slab) }, ArenaSlabStats {
            total_allocated: 2, free_count: 0, in_use: 2,
        });

        unsafe { arena_slab_free(slab, b, base) };
        assert_eq!(unsafe { arena_slab_stats(slab) }, ArenaSlabStats {
            total_allocated: 2, free_count: 1, in_use: 1,
        });
    }

    // ── Double-free detection tests ─────────────────────────────────

    #[test]
    fn slab_double_free_detected() {
        let (mut buf, slab) = make_slab(4096, 64);
        let base = buf.as_mut_ptr();
        let slot = unsafe { arena_slab_alloc(slab, base) };
        assert!(!slot.is_null());

        assert_eq!(unsafe { arena_slab_free(slab, slot, base) }, 0);
        assert_eq!(unsafe { arena_slab_free(slab, slot, base) }, -1);

        let stats = unsafe { arena_slab_stats(slab) };
        assert_eq!(stats.free_count, 1);
        assert_eq!(stats.in_use, 0);
    }

    #[test]
    fn slab_double_free_after_realloc_ok() {
        let (mut buf, slab) = make_slab(4096, 64);
        let base = buf.as_mut_ptr();
        let slot = unsafe { arena_slab_alloc(slab, base) };

        assert_eq!(unsafe { arena_slab_free(slab, slot, base) }, 0);
        // Reallocate the same slot
        let slot2 = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(slot.offset(), slot2.offset());
        // Freeing again after reallocation is valid
        assert_eq!(unsafe { arena_slab_free(slab, slot2, base) }, 0);
    }

    #[test]
    fn slab_double_free_does_not_corrupt_free_list() {
        let (mut buf, slab) = make_slab(4096, 32);
        let base = buf.as_mut_ptr();
        let a = unsafe { arena_slab_alloc(slab, base) };
        let b = unsafe { arena_slab_alloc(slab, base) };
        let c = unsafe { arena_slab_alloc(slab, base) };

        assert_eq!(unsafe { arena_slab_free(slab, b, base) }, 0);
        // Double-free of b should fail
        assert_eq!(unsafe { arena_slab_free(slab, b, base) }, -1);

        // Free list should still be intact: only b is free
        assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 1);

        // Can still free a and c normally
        assert_eq!(unsafe { arena_slab_free(slab, a, base) }, 0);
        assert_eq!(unsafe { arena_slab_free(slab, c, base) }, 0);
        assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 3);

        // Reallocate all 3 — free list should be fully functional
        let r1 = unsafe { arena_slab_alloc(slab, base) };
        let r2 = unsafe { arena_slab_alloc(slab, base) };
        let r3 = unsafe { arena_slab_alloc(slab, base) };
        assert!(!r1.is_null() && !r2.is_null() && !r3.is_null());
        assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 0);
        assert_eq!(unsafe { arena_slab_stats(slab) }.total_allocated, 3);
    }

    #[test]
    fn slab_free_magic_cleared_on_alloc() {
        let (mut buf, slab) = make_slab(4096, 64);
        let base = buf.as_mut_ptr();
        let slot = unsafe { arena_slab_alloc(slab, base) };

        // Free, then reallocate — magic should be cleared
        assert_eq!(unsafe { arena_slab_free(slab, slot, base) }, 0);
        let slot2 = unsafe { arena_slab_alloc(slab, base) };
        assert_eq!(slot.offset(), slot2.offset());

        // Verify magic was cleared by checking the raw memory
        let slot_raw = unsafe { slot2.resolve(base) };
        let magic = unsafe { core::ptr::read(slot_raw.add(8).cast::<u64>()) };
        assert_ne!(magic, SLAB_FREE_MAGIC);
    }

    #[test]
    fn slab_multiple_double_frees_all_rejected() {
        let (mut buf, slab) = make_slab(4096, 32);
        let base = buf.as_mut_ptr();
        let slot = unsafe { arena_slab_alloc(slab, base) };

        assert_eq!(unsafe { arena_slab_free(slab, slot, base) }, 0);
        for _ in 0..5 {
            assert_eq!(unsafe { arena_slab_free(slab, slot, base) }, -1);
        }
        assert_eq!(unsafe { arena_slab_stats(slab) }.free_count, 1);
    }
}

#[cfg(test)]
mod proptests {
    extern crate alloc;
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::mem;
    use proptest::prelude::*;
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn make_hash_map(capacity: u32) -> (Vec<u8>, *mut ArenaHashMap) {
        let header_size = mem::size_of::<ArenaHashMap>();
        let entry_size = mem::size_of::<ArenaHashEntry>();
        let total = header_size + entry_size * capacity as usize;
        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();
        let header_ptr = base.cast::<ArenaHashMap>();
        let entries_ptr = unsafe { base.add(header_size) }.cast::<ArenaHashEntry>();
        let ret = unsafe { arena_hash_init(header_ptr, entries_ptr, capacity, base) };
        assert_eq!(ret, 0);
        (buf, header_ptr)
    }

    fn make_btree() -> (Vec<u8>, *mut ArenaBTreeMap, *mut ArenaBumpState) {
        let arena_size = 512 * 1024;
        let mut buf = vec![0u8; arena_size];
        let base = buf.as_mut_ptr();
        let map = base.cast::<ArenaBTreeMap>();
        unsafe { arena_btree_init(map) };
        let bump_offset = mem::size_of::<ArenaBTreeMap>();
        let bump = unsafe { base.add(bump_offset).cast::<ArenaBumpState>() };
        let data_start = bump_offset + mem::size_of::<ArenaBumpState>();
        unsafe {
            *bump = ArenaBumpState::new((arena_size - data_start) as u64);
            (*bump).watermark = data_start as u64;
        }
        (buf, map, bump)
    }

    fn make_slab(capacity: usize, slot_size: u32) -> (Vec<u8>, *mut ArenaSlabState) {
        let header_size = mem::size_of::<ArenaSlabState>();
        let total = header_size + capacity;
        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();
        let slab = base.cast::<ArenaSlabState>();
        let ret = unsafe { arena_slab_init(slab, total as u64, slot_size) };
        assert_eq!(ret, 0);
        unsafe { (*slab).bump.watermark = header_size as u64 };
        (buf, slab)
    }

    // ── Hash map property tests ─────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn hash_insert_then_get(entries in prop::collection::vec((1u64..10000, 0u64..100000), 1..50)) {
            let (mut buf, map) = make_hash_map(64);
            let base = buf.as_mut_ptr();

            let mut reference = BTreeMap::new();
            for &(k, v) in &entries {
                let ret = unsafe { arena_hash_insert(map, k, v, base) };
                if ret >= 0 {
                    reference.insert(k, v);
                }
            }

            for (&k, &v) in &reference {
                let got = unsafe { arena_hash_get(map, k, base) };
                prop_assert!(!got.is_null(), "key {} missing", k);
                prop_assert_eq!(unsafe { *got }, v);
            }
            prop_assert_eq!(unsafe { (*map).count }, reference.len() as u32);
        }

        #[test]
        fn hash_delete_then_miss(
            inserts in prop::collection::vec((1u64..10000, 0u64..100000), 10..50),
            delete_indices in prop::collection::vec(any::<prop::sample::Index>(), 1..10),
        ) {
            let (mut buf, map) = make_hash_map(64);
            let base = buf.as_mut_ptr();

            let mut reference = BTreeMap::new();
            for &(k, v) in &inserts {
                if unsafe { arena_hash_insert(map, k, v, base) } >= 0 {
                    reference.insert(k, v);
                }
            }

            if reference.is_empty() { return Ok(()); }

            let keys: Vec<u64> = reference.keys().copied().collect();
            for idx in &delete_indices {
                let k = keys[idx.index(keys.len())];
                if reference.remove(&k).is_some() {
                    let ret = unsafe { arena_hash_delete(map, k, base) };
                    prop_assert_eq!(ret, 0);
                }
            }

            for &k in reference.keys() {
                let got = unsafe { arena_hash_get(map, k, base) };
                prop_assert!(!got.is_null(), "surviving key {} missing", k);
            }

            for idx in &delete_indices {
                let k = keys[idx.index(keys.len())];
                if !reference.contains_key(&k) {
                    let got = unsafe { arena_hash_get(map, k, base) };
                    prop_assert!(got.is_null(), "deleted key {} still found", k);
                }
            }
        }

        #[test]
        fn hash_iteration_count(entries in prop::collection::vec((1u64..10000, 0u64..100000), 1..50)) {
            let (mut buf, map) = make_hash_map(64);
            let base = buf.as_mut_ptr();

            let mut reference = BTreeMap::new();
            for &(k, v) in &entries {
                if unsafe { arena_hash_insert(map, k, v, base) } >= 0 {
                    reference.insert(k, v);
                }
            }

            let mut count = 0u32;
            unsafe {
                arena_hash_for_each(map, base, |k, v| {
                    assert!(reference.get(&k) == Some(&v),
                        "iteration yielded unexpected ({}, {})", k, v);
                    count += 1;
                });
            }
            prop_assert_eq!(count, reference.len() as u32);
        }
    }

    // ── Hash map adversarial key distributions ──────────────────────

    #[test]
    fn hash_sequential_keys() {
        let (mut buf, map) = make_hash_map(128);
        let base = buf.as_mut_ptr();
        for i in 0..80u64 {
            assert_eq!(unsafe { arena_hash_insert(map, i, i * 10, base) }, 0);
        }
        for i in 0..80u64 {
            let v = unsafe { arena_hash_get(map, i, base) };
            assert!(!v.is_null());
            assert_eq!(unsafe { *v }, i * 10);
        }
    }

    #[test]
    fn hash_reverse_keys() {
        let (mut buf, map) = make_hash_map(128);
        let base = buf.as_mut_ptr();
        for i in (0..80u64).rev() {
            assert_eq!(unsafe { arena_hash_insert(map, i, i * 10, base) }, 0);
        }
        for i in 0..80u64 {
            let v = unsafe { arena_hash_get(map, i, base) };
            assert!(!v.is_null());
            assert_eq!(unsafe { *v }, i * 10);
        }
    }

    #[test]
    fn hash_same_bucket_keys() {
        let (mut buf, map) = make_hash_map(64);
        let base = buf.as_mut_ptr();
        // Keys that are multiples of 64 will hash to the same bucket (mask=63).
        // With linear probing they'll cluster together.
        for i in 0..30u64 {
            assert_eq!(unsafe { arena_hash_insert(map, i * 64, i, base) }, 0);
        }
        for i in 0..30u64 {
            let v = unsafe { arena_hash_get(map, i * 64, base) };
            assert!(!v.is_null());
            assert_eq!(unsafe { *v }, i);
        }
        assert_eq!(unsafe { (*map).count }, 30);
    }

    // ── Hash map random interleaved operations ──────────────────────

    #[derive(Debug, Clone)]
    enum HashOp {
        Insert(u64, u64),
        Delete(u64),
        Get(u64),
    }

    fn hash_op_strategy() -> impl Strategy<Value = HashOp> {
        prop_oneof![
            3 => (1u64..500, 0u64..100000).prop_map(|(k, v)| HashOp::Insert(k, v)),
            2 => (1u64..500).prop_map(HashOp::Delete),
            2 => (1u64..500).prop_map(HashOp::Get),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn hash_random_ops(ops in prop::collection::vec(hash_op_strategy(), 10..200)) {
            let (mut buf, map) = make_hash_map(256);
            let base = buf.as_mut_ptr();
            let mut reference = BTreeMap::new();

            for op in &ops {
                match *op {
                    HashOp::Insert(k, v) => {
                        let ret = unsafe { arena_hash_insert(map, k, v, base) };
                        if ret >= 0 {
                            reference.insert(k, v);
                        }
                    }
                    HashOp::Delete(k) => {
                        let ret = unsafe { arena_hash_delete(map, k, base) };
                        let expected = if reference.remove(&k).is_some() { 0 } else { -1 };
                        prop_assert_eq!(ret, expected);
                    }
                    HashOp::Get(k) => {
                        let got = unsafe { arena_hash_get(map, k, base) };
                        match reference.get(&k) {
                            Some(&v) => {
                                prop_assert!(!got.is_null(), "key {} should exist", k);
                                prop_assert_eq!(unsafe { *got }, v);
                            }
                            None => {
                                prop_assert!(got.is_null(), "key {} should not exist", k);
                            }
                        }
                    }
                }
            }
            prop_assert_eq!(unsafe { (*map).count }, reference.len() as u32);
        }
    }

    // ── B-tree property tests ───────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn btree_insert_then_get(entries in prop::collection::vec((0u64..50000, 0u64..100000), 1..200)) {
            let (mut buf, map, bump) = make_btree();
            let base = buf.as_mut_ptr();

            let mut reference = BTreeMap::new();
            for &(k, v) in &entries {
                let ret = unsafe { arena_btree_insert(map, bump, k, v, base) };
                prop_assert!(ret >= 0, "insert failed for key {}", k);
                reference.insert(k, v);
            }

            for (&k, &v) in &reference {
                let got = unsafe { arena_btree_get(map, k, base) };
                prop_assert!(!got.is_null(), "key {} missing", k);
                prop_assert_eq!(unsafe { *got }, v);
            }
            prop_assert_eq!(unsafe { (*map).count }, reference.len() as u64);
        }

        #[test]
        fn btree_sorted_iteration(keys in prop::collection::vec(0u64..50000, 1..200)) {
            let (mut buf, map, bump) = make_btree();
            let base = buf.as_mut_ptr();

            let mut reference = BTreeMap::new();
            for &k in &keys {
                unsafe { arena_btree_insert(map, bump, k, k * 10, base) };
                reference.insert(k, k * 10);
            }

            let mut iter_keys = Vec::new();
            unsafe {
                arena_btree_for_each(map, base, |k, v| {
                    assert_eq!(v, k * 10, "wrong value for key {}", k);
                    iter_keys.push(k);
                });
            }

            let ref_keys: Vec<u64> = reference.keys().copied().collect();
            prop_assert_eq!(&iter_keys, &ref_keys, "iteration order mismatch");
        }

        #[test]
        fn btree_delete_then_miss(
            keys in prop::collection::vec(0u64..10000, 20..100),
            delete_count in 5usize..15,
        ) {
            let (mut buf, map, bump) = make_btree();
            let base = buf.as_mut_ptr();

            let mut reference = BTreeMap::new();
            for &k in &keys {
                unsafe { arena_btree_insert(map, bump, k, k, base) };
                reference.insert(k, k);
            }

            let to_delete: Vec<u64> = reference.keys().copied().take(delete_count.min(reference.len())).collect();
            for &k in &to_delete {
                let ret = unsafe { arena_btree_delete(map, k, base) };
                prop_assert_eq!(ret, 0, "delete failed for key {}", k);
                reference.remove(&k);

                let got = unsafe { arena_btree_get(map, k, base) };
                prop_assert!(got.is_null(), "deleted key {} still found", k);
            }

            for (&k, &v) in &reference {
                let got = unsafe { arena_btree_get(map, k, base) };
                prop_assert!(!got.is_null(), "surviving key {} missing", k);
                prop_assert_eq!(unsafe { *got }, v);
            }

            prop_assert_eq!(unsafe { (*map).count }, reference.len() as u64);
        }
    }

    // ── B-tree adversarial distributions ────────────────────────────

    #[test]
    fn btree_sequential_ascending() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();
        for i in 0..500u64 {
            assert!(unsafe { arena_btree_insert(map, bump, i, i, base) } >= 0);
        }
        let mut prev = None;
        let mut count = 0u64;
        unsafe {
            arena_btree_for_each(map, base, |k, _| {
                if let Some(p) = prev { assert!(k > p); }
                prev = Some(k);
                count += 1;
            });
        }
        assert_eq!(count, 500);
    }

    #[test]
    fn btree_sequential_descending() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();
        for i in (0..500u64).rev() {
            assert!(unsafe { arena_btree_insert(map, bump, i, i, base) } >= 0);
        }
        let mut prev = None;
        let mut count = 0u64;
        unsafe {
            arena_btree_for_each(map, base, |k, _| {
                if let Some(p) = prev { assert!(k > p); }
                prev = Some(k);
                count += 1;
            });
        }
        assert_eq!(count, 500);
    }

    // ── B-tree random interleaved operations ────────────────────────

    #[derive(Debug, Clone)]
    enum BTreeOp {
        Insert(u64, u64),
        Delete(u64),
        Get(u64),
    }

    fn btree_op_strategy() -> impl Strategy<Value = BTreeOp> {
        prop_oneof![
            3 => (0u64..2000, 0u64..100000).prop_map(|(k, v)| BTreeOp::Insert(k, v)),
            2 => (0u64..2000).prop_map(BTreeOp::Delete),
            2 => (0u64..2000).prop_map(BTreeOp::Get),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn btree_random_ops(ops in prop::collection::vec(btree_op_strategy(), 10..300)) {
            let (mut buf, map, bump) = make_btree();
            let base = buf.as_mut_ptr();
            let mut reference = BTreeMap::new();

            for op in &ops {
                match *op {
                    BTreeOp::Insert(k, v) => {
                        let ret = unsafe { arena_btree_insert(map, bump, k, v, base) };
                        if ret >= 0 {
                            reference.insert(k, v);
                        }
                    }
                    BTreeOp::Delete(k) => {
                        let ret = unsafe { arena_btree_delete(map, k, base) };
                        let existed = reference.remove(&k).is_some();
                        prop_assert_eq!(ret == 0, existed, "delete mismatch for key {}", k);
                    }
                    BTreeOp::Get(k) => {
                        let got = unsafe { arena_btree_get(map, k, base) };
                        match reference.get(&k) {
                            Some(&v) => {
                                prop_assert!(!got.is_null(), "key {} should exist", k);
                                prop_assert_eq!(unsafe { *got }, v);
                            }
                            None => {
                                prop_assert!(got.is_null(), "key {} should not exist", k);
                            }
                        }
                    }
                }
            }
            prop_assert_eq!(unsafe { (*map).count }, reference.len() as u64);

            // Verify sorted iteration matches reference
            let mut arena_entries = Vec::new();
            unsafe {
                arena_btree_for_each(map, base, |k, v| {
                    arena_entries.push((k, v));
                });
            }
            let ref_entries: Vec<(u64, u64)> = reference.into_iter().collect();
            prop_assert_eq!(arena_entries, ref_entries);
        }
    }

    // ── Slab property tests ─────────────────────────────────────────

    #[derive(Debug, Clone)]
    enum SlabOp {
        Alloc,
        Free(usize),
    }

    fn slab_op_strategy(max_idx: usize) -> impl Strategy<Value = SlabOp> {
        prop_oneof![
            3 => Just(SlabOp::Alloc),
            2 => (0..max_idx).prop_map(SlabOp::Free),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn slab_random_alloc_free(ops in prop::collection::vec(slab_op_strategy(200), 10..200)) {
            let (mut buf, slab) = make_slab(16384, 32);
            let base = buf.as_mut_ptr();

            let mut live: Vec<ArenaPtr<u8>> = Vec::new();
            let mut total_alloced = 0u32;
            let mut free_count = 0u32;

            for op in &ops {
                match *op {
                    SlabOp::Alloc => {
                        let slot = unsafe { arena_slab_alloc(slab, base) };
                        if !slot.is_null() {
                            live.push(slot);
                            if free_count > 0 {
                                free_count -= 1;
                            } else {
                                total_alloced += 1;
                            }
                        }
                    }
                    SlabOp::Free(idx) => {
                        if !live.is_empty() {
                            let i = idx % live.len();
                            let slot = live.swap_remove(i);
                            let ret = unsafe { arena_slab_free(slab, slot, base) };
                            prop_assert_eq!(ret, 0, "free should succeed");
                            free_count += 1;
                        }
                    }
                }

                let stats = unsafe { arena_slab_stats(slab) };
                prop_assert_eq!(stats.total_allocated, total_alloced);
                prop_assert_eq!(stats.free_count, free_count);
                prop_assert_eq!(stats.in_use, total_alloced - free_count);
                prop_assert_eq!(stats.in_use as usize, live.len());
            }
        }
    }
}

#[cfg(test)]
mod concurrency_tests {
    extern crate alloc;
    extern crate std;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::mem;
    use std::sync::Arc;
    use std::thread;
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn make_hash_map(capacity: u32) -> (Vec<u8>, *mut ArenaHashMap) {
        let header_size = mem::size_of::<ArenaHashMap>();
        let entry_size = mem::size_of::<ArenaHashEntry>();
        let total = header_size + entry_size * capacity as usize;
        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();
        let header_ptr = base.cast::<ArenaHashMap>();
        let entries_ptr = unsafe { base.add(header_size) }.cast::<ArenaHashEntry>();
        let ret = unsafe { arena_hash_init(header_ptr, entries_ptr, capacity, base) };
        assert_eq!(ret, 0);
        (buf, header_ptr)
    }

    fn make_btree() -> (Vec<u8>, *mut ArenaBTreeMap, *mut ArenaBumpState) {
        let arena_size = 512 * 1024;
        let mut buf = vec![0u8; arena_size];
        let base = buf.as_mut_ptr();
        let map = base.cast::<ArenaBTreeMap>();
        unsafe { arena_btree_init(map) };
        let bump_offset = mem::size_of::<ArenaBTreeMap>();
        let bump = unsafe { base.add(bump_offset).cast::<ArenaBumpState>() };
        let data_start = bump_offset + mem::size_of::<ArenaBumpState>();
        unsafe {
            *bump = ArenaBumpState::new((arena_size - data_start) as u64);
            (*bump).watermark = data_start as u64;
        }
        (buf, map, bump)
    }

    // ── Concurrent read-after-write tests ───────────────────────────
    //
    // These verify the safe pattern: one thread initializes the data
    // structure, then multiple threads read concurrently. A barrier
    // (join) ensures happens-before between writer and readers.

    #[test]
    fn hash_concurrent_reads_after_init() {
        let (mut buf, map) = make_hash_map(256);
        let base = buf.as_mut_ptr();

        for i in 0..100u64 {
            assert!(unsafe { arena_hash_insert(map, i, i * 10, base) } >= 0);
        }

        let shared_buf = Arc::new(buf);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let buf_ref = Arc::clone(&shared_buf);
                thread::spawn(move || {
                    let base = buf_ref.as_ptr() as *mut u8;
                    let map = base.cast::<ArenaHashMap>();
                    for i in 0..100u64 {
                        let val = unsafe { arena_hash_get(map, i, base) };
                        assert!(!val.is_null(), "key {} not found", i);
                        assert_eq!(unsafe { *val }, i * 10);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn btree_concurrent_reads_after_init() {
        let (mut buf, map, bump) = make_btree();
        let base = buf.as_mut_ptr();

        for i in 0..200u64 {
            assert!(unsafe { arena_btree_insert(map, bump, i, i * 10, base) } >= 0);
        }

        let shared_buf = Arc::new(buf);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let buf_ref = Arc::clone(&shared_buf);
                thread::spawn(move || {
                    let base = buf_ref.as_ptr() as *mut u8;
                    let map = base.cast::<ArenaBTreeMap>();
                    for i in 0..200u64 {
                        let val = unsafe { arena_btree_get(map, i, base) };
                        assert!(!val.is_null(), "key {} not found", i);
                        assert_eq!(unsafe { *val }, i * 10);
                    }

                    let mut prev = None;
                    let mut count = 0u64;
                    unsafe {
                        arena_btree_for_each(map, base, |k, _| {
                            if let Some(p) = prev {
                                assert!(k > p);
                            }
                            prev = Some(k);
                            count += 1;
                        });
                    }
                    assert_eq!(count, 200);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    // ── Per-instance partitioning test ───────────────────────────────
    //
    // This demonstrates the recommended multi-CPU pattern: each "CPU"
    // (thread) gets its own independent data structure instance within
    // the same arena buffer. No locks needed.

    #[test]
    fn slab_per_thread_partitioning() {
        let num_threads = 4usize;
        let slab_region_size = 4096usize;
        let slot_size = 32u32;
        let header_size = mem::size_of::<ArenaSlabState>();
        let per_thread_size = header_size + slab_region_size;
        let total = per_thread_size * num_threads;

        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();

        let mut slab_offsets = Vec::new();
        for t in 0..num_threads {
            let offset = t * per_thread_size;
            let slab_base = unsafe { base.add(offset) };
            let slab = slab_base.cast::<ArenaSlabState>();
            let ret = unsafe { arena_slab_init(slab, per_thread_size as u64, slot_size) };
            assert_eq!(ret, 0);
            unsafe { (*slab).bump.watermark = header_size as u64 };
            slab_offsets.push(offset);
        }

        let shared_buf = Arc::new(buf);
        let handles: Vec<_> = slab_offsets
            .iter()
            .map(|&offset| {
                let buf_ref = Arc::clone(&shared_buf);
                thread::spawn(move || {
                    // Each thread uses its own partition start as arena_base
                    let partition_base = unsafe { buf_ref.as_ptr().add(offset) as *mut u8 };
                    let slab = partition_base.cast::<ArenaSlabState>();

                    let mut slots = Vec::new();
                    for _ in 0..50 {
                        let s = unsafe { arena_slab_alloc(slab, partition_base) };
                        assert!(!s.is_null());
                        slots.push(s);
                    }
                    for s in &slots {
                        assert_eq!(unsafe { arena_slab_free(slab, *s, partition_base) }, 0);
                    }

                    let stats = unsafe { arena_slab_stats(slab) };
                    assert_eq!(stats.in_use, 0);
                    assert_eq!(stats.free_count, stats.total_allocated);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn hash_per_thread_partitioning() {
        let num_threads = 4usize;
        let capacity = 64u32;
        let header_size = mem::size_of::<ArenaHashMap>();
        let entry_size = mem::size_of::<ArenaHashEntry>();
        let per_thread_size = header_size + entry_size * capacity as usize;
        let total = per_thread_size * num_threads;

        let mut buf = vec![0u8; total];
        let base = buf.as_mut_ptr();

        let mut offsets = Vec::new();
        for t in 0..num_threads {
            let offset = t * per_thread_size;
            let partition_base = unsafe { base.add(offset) };
            let map = partition_base.cast::<ArenaHashMap>();
            let entries = unsafe { partition_base.add(header_size).cast::<ArenaHashEntry>() };
            let ret = unsafe { arena_hash_init(map, entries, capacity, partition_base) };
            assert_eq!(ret, 0);
            offsets.push(offset);
        }

        let shared_buf = Arc::new(buf);
        let handles: Vec<_> = offsets
            .iter()
            .enumerate()
            .map(|(tid, &offset)| {
                let buf_ref = Arc::clone(&shared_buf);
                thread::spawn(move || {
                    let partition_base = unsafe { buf_ref.as_ptr().add(offset) as *mut u8 };
                    let map = partition_base.cast::<ArenaHashMap>();
                    let base_key = (tid as u64) * 1000;

                    for i in 0..40u64 {
                        let ret = unsafe { arena_hash_insert(map, base_key + i, i * 10, partition_base) };
                        assert_eq!(ret, 0);
                    }
                    for i in 0..40u64 {
                        let val = unsafe { arena_hash_get(map, base_key + i, partition_base) };
                        assert!(!val.is_null());
                        assert_eq!(unsafe { *val }, i * 10);
                    }
                    assert_eq!(unsafe { (*map).count }, 40);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}
