//! CO-RE post-processor library for BPF ELF objects.
//!
//! This crate provides the core logic for adding `bpf_core_relo` records
//! to BPF ELF objects.  It can be used:
//!
//!   - As a standalone CLI tool (`aya-core-postprocessor`)
//!   - As a library from `aya-build` to auto-process BPF binaries
//!
//! ## Usage from aya-build
//!
//! ```ignore
//! use aya_core_postprocessor::postprocess_core_relos;
//!
//! // After compiling a BPF binary:
//! let elf_path = out_dir.join("my-bpf-prog");
//! postprocess_core_relos(&elf_path)?;
//! ```

pub mod btf_ext_writer;
pub mod btf_parser;
pub mod elf_patcher;
pub mod insn_scanner;
pub mod marker_parser;
pub mod sidecar;

#[cfg(test)]
mod test_helpers;

use std::path::Path;

use anyhow::{Context, Result};

use btf_ext_writer::BtfExtWriter;
use btf_parser::BtfInfo;

/// Processes a BPF ELF file using sidecar TOML configuration.
///
/// Reads the sidecar file to determine which struct field accesses
/// need CO-RE relocations, then patches the ELF with the appropriate
/// `.BTF.ext` records.
pub fn process_elf(elf_data: &[u8], config: &sidecar::SidecarConfig) -> Result<Vec<u8>> {
    let btf = BtfInfo::parse_from_elf(elf_data).context("parsing BTF from ELF")?;

    let mut writer = BtfExtWriter::new(&btf);
    for (i, relo) in config.relocation.iter().enumerate() {
        writer
            .add_relocation(relo)
            .with_context(|| format!("processing relocation #{i}: {relo:?}"))?;
    }

    let (new_btf_data, new_btf_ext_data) = writer.finish(elf_data)?;
    elf_patcher::patch_elf_sections(elf_data, &new_btf_data, &new_btf_ext_data)
}

/// Auto-discovers and processes CO-RE relocations from `.aya.core_relo`
/// markers in a BPF ELF file.
///
/// This is the primary entry point for the aya-build integration.
/// It reads markers emitted by the `core_read!` proc macro, computes
/// field offsets from BTF, scans BPF instructions for matches, and
/// generates CO-RE relocation records.
///
/// Returns the modified ELF bytes, or the original bytes unchanged if
/// no markers are found.
pub fn process_elf_auto(elf_data: &[u8]) -> Result<Vec<u8>> {
    let markers = marker_parser::parse_markers_from_elf(elf_data)
        .context("parsing .aya.core_relo markers")?;
    let markers = marker_parser::deduplicate_markers(markers);

    if markers.is_empty() {
        return Ok(elf_data.to_vec());
    }

    let btf = BtfInfo::parse_from_elf(elf_data).context("parsing BTF from ELF")?;

    let mut relo_entries = Vec::new();

    for marker in &markers {
        let type_id = btf
            .find_struct_by_name(&marker.struct_name)
            .with_context(|| {
                format!("looking up struct '{}' in BTF", marker.struct_name)
            })?;

        let byte_offset = btf
            .compute_byte_offset(type_id, &marker.field_path)
            .with_context(|| {
                format!(
                    "computing byte offset for {}.{}",
                    marker.struct_name, marker.field_path
                )
            })?;

        let insn_matches = insn_scanner::find_insns_with_offset(elf_data, byte_offset)?;

        for m in &insn_matches {
            relo_entries.push(sidecar::RelocationEntry {
                section: m.section_name.clone(),
                insn_index: m.insn_index,
                struct_name: marker.struct_name.clone(),
                field_path: marker.field_path.clone(),
            });
        }
    }

    if relo_entries.is_empty() {
        return Ok(elf_data.to_vec());
    }

    let mut writer = BtfExtWriter::new(&btf);
    for (i, relo) in relo_entries.iter().enumerate() {
        writer
            .add_relocation(relo)
            .with_context(|| format!("processing auto-discovered relocation #{i}: {relo:?}"))?;
    }

    let (new_btf_data, new_btf_ext_data) = writer.finish(elf_data)?;
    elf_patcher::patch_elf_sections(elf_data, &new_btf_data, &new_btf_ext_data)
}

/// Convenience function: auto-discovers and processes CO-RE relocations
/// for a BPF ELF file on disk.
///
/// Reads the file, processes it, and writes the result back to the same
/// path (or does nothing if no markers are found).
///
/// This is the function that `aya-build` should call after compiling
/// each BPF binary.
pub fn postprocess_core_relos(elf_path: &Path) -> Result<()> {
    let elf_data = std::fs::read(elf_path)
        .with_context(|| format!("reading BPF ELF: {}", elf_path.display()))?;

    // Check if the ELF has any .aya.core_relo markers.
    let markers = marker_parser::parse_markers_from_elf(&elf_data)
        .context("checking for .aya.core_relo markers")?;

    if markers.is_empty() {
        // No markers -- nothing to do.
        return Ok(());
    }

    let result = process_elf_auto(&elf_data)?;

    std::fs::write(elf_path, &result)
        .with_context(|| format!("writing processed ELF: {}", elf_path.display()))?;

    Ok(())
}
