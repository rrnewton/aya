//! Kfunc wrappers for BPF arena memory management.
//!
//! Kfuncs are kernel functions resolved by name at load time via BTF,
//! unlike BPF helpers which are resolved by number. The Rust BPF compiler
//! doesn't emit proper kfunc call instructions for `extern "C"` declarations,
//! so we use inline assembly with `call {func}` + `sym` operands.

pub mod arena;

pub use arena::{arena_alloc_pages, arena_free_pages};
