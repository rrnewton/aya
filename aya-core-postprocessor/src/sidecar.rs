//! Sidecar TOML configuration for CO-RE relocations.
//!
//! The sidecar file describes which field accesses in the BPF program
//! should be made relocatable.  Each entry specifies the ELF section
//! containing the program, the instruction index of the field access,
//! the struct type name, and the dot-separated field path.

use serde::Deserialize;

/// Top-level sidecar configuration.
#[derive(Debug, Deserialize)]
pub struct SidecarConfig {
    /// List of field access relocations to add.
    pub relocation: Vec<RelocationEntry>,
}

/// A single CO-RE relocation to generate.
#[derive(Debug, Deserialize)]
pub struct RelocationEntry {
    /// ELF section name containing the BPF program
    /// (e.g. "tracepoint/sched/sched_switch").
    pub section: String,

    /// Instruction index within the section (0-based).
    /// This is the index of the LDX/STX/MOV instruction whose
    /// offset field encodes the struct field offset.
    pub insn_index: u32,

    /// Name of the struct type in BTF (e.g. "task_struct").
    pub struct_name: String,

    /// Dot-separated field path (e.g. "pid" or "scx.dsq_vtime").
    pub field_path: String,
}
