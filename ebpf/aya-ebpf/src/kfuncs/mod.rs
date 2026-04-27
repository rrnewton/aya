//! Kfunc wrappers for BPF arena memory management.
//!
//! Kfuncs are kernel functions resolved by name at load time via BTF,
//! unlike BPF helpers which are resolved by number. The Rust BPF compiler
//! doesn't emit proper kfunc call instructions for `extern "C"` declarations,
//! so we use inline assembly with `call {func}` + `sym` operands.

pub mod arena;
pub mod bump;
pub mod global_alloc;

pub use arena::{arena_alloc_pages, arena_free_pages, cast_kern, cast_user, NUMA_NO_NODE};
pub use bump::{bump_alloc, bump_destroy, bump_init, bump_memlimit, BumpAllocator};
pub use global_alloc::ArenaGlobalAlloc;
