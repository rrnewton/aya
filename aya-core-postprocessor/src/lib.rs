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

use anyhow::{Context, Result, bail};

use btf_ext_writer::BtfExtWriter;
use btf_parser::BtfInfo;

/// Default path to the kernel's vmlinux BTF.
const DEFAULT_VMLINUX_BTF: &str = "/sys/kernel/btf/vmlinux";

/// A struct member access needed for CO-RE relocation.
#[derive(Clone)]
struct MemberAccess {
    name: String,
    bit_offset: u32,
    /// If this member is an intermediate struct, its name.
    child_struct_name: Option<String>,
}

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
/// markers in a BPF ELF file, using the kernel's vmlinux BTF for struct
/// resolution.
///
/// This is the primary entry point for the aya-build integration.
/// It reads markers emitted by the `core_read!`/`core_write!` macros,
/// resolves field offsets from vmlinux BTF, scans BPF instructions for
/// matches, adds the necessary struct definitions to the program's BTF,
/// and generates CO-RE relocation records.
///
/// Returns the modified ELF bytes, or the original bytes unchanged if
/// no markers are found.
pub fn process_elf_auto(elf_data: &[u8]) -> Result<Vec<u8>> {
    let vmlinux_path = std::env::var("SCX_VMLINUX_BTF")
        .unwrap_or_else(|_| DEFAULT_VMLINUX_BTF.to_string());

    process_elf_auto_with_vmlinux(elf_data, &vmlinux_path)
}

/// Like `process_elf_auto`, but with an explicit vmlinux BTF path.
pub fn process_elf_auto_with_vmlinux(
    elf_data: &[u8],
    vmlinux_path: &str,
) -> Result<Vec<u8>> {
    let markers = marker_parser::parse_markers_from_elf(elf_data)
        .context("parsing .aya.core_relo markers")?;
    let markers = marker_parser::deduplicate_markers(markers);

    if markers.is_empty() {
        return Ok(elf_data.to_vec());
    }

    // Load vmlinux BTF for struct resolution.
    let vmlinux_data = std::fs::read(vmlinux_path)
        .with_context(|| format!("reading vmlinux BTF: {vmlinux_path}"))?;
    let vmlinux_btf = BtfInfo::parse(&vmlinux_data)
        .context("parsing vmlinux BTF")?;

    // Load program BTF (we'll add struct definitions to it).
    let mut prog_btf = BtfInfo::parse_from_elf(elf_data)
        .context("parsing program BTF from ELF")?;

    // Phase 1: Resolve all markers using vmlinux BTF and scan for
    // matching instructions.
    let mut relo_entries = Vec::new();

    for marker in &markers {
        let vmlinux_type_id = vmlinux_btf
            .find_struct_by_name(&marker.struct_name)
            .with_context(|| {
                format!("looking up struct '{}' in vmlinux BTF", marker.struct_name)
            })?;

        let byte_offset = vmlinux_btf
            .compute_byte_offset(vmlinux_type_id, &marker.field_path)
            .with_context(|| {
                format!(
                    "computing byte offset for {}.{}",
                    marker.struct_name, marker.field_path
                )
            })?;

        let insn_matches = insn_scanner::find_insns_with_offset(elf_data, byte_offset)?;

        if insn_matches.is_empty() {
            eprintln!(
                "  warning: no instruction matches for {}.{} (offset={})",
                marker.struct_name, marker.field_path, byte_offset
            );
        }

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

    // Phase 2: Add struct definitions from vmlinux BTF into the
    // program's BTF.  We create minimal stub structs with just the
    // fields that are accessed.
    //
    // The CO-RE loader needs the struct in the program's BTF to:
    //   1. Look up the struct by name
    //   2. Walk the access string through the struct members
    //   3. Match members by name against vmlinux types
    //
    // Strategy: collect all unique (struct_name, field_name) pairs at
    // each level of nesting, then create one stub struct per unique
    // struct name with all its accessed members.

    // First, create a placeholder INT type for leaf member types.
    let placeholder_int_id = prog_btf.add_int("__u64", 8);

    // Collect all struct->member accesses needed.
    // Key: struct name (from vmlinux)
    // Value: set of (member_name, member_bit_offset, child_struct_name_or_none)
    let mut struct_members: std::collections::HashMap<String, Vec<MemberAccess>> =
        std::collections::HashMap::new();

    for marker in &markers {
        collect_struct_members(
            &vmlinux_btf,
            &marker.struct_name,
            &marker.field_path,
            &mut struct_members,
        )
        .with_context(|| {
            format!(
                "collecting members for {}.{}",
                marker.struct_name, marker.field_path
            )
        })?;
    }

    // Deduplicate members per struct.
    for members in struct_members.values_mut() {
        members.sort_by(|a, b| a.bit_offset.cmp(&b.bit_offset));
        members.dedup_by(|a, b| a.name == b.name);
    }

    // Create stub structs bottom-up (children before parents).
    // We do a topological sort: if struct A has a member pointing to
    // struct B, B must be created first.
    let mut struct_type_ids: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();

    // Simple iterative approach: keep creating structs whose children
    // are already created, until all are done.
    let struct_names: Vec<String> = struct_members.keys().cloned().collect();
    let mut remaining: std::collections::HashSet<String> =
        struct_names.iter().cloned().collect();

    for _ in 0..100 {
        // Safety bound to prevent infinite loops
        if remaining.is_empty() {
            break;
        }

        let mut made_progress = false;
        let to_process: Vec<String> = remaining.iter().cloned().collect();

        for struct_name in &to_process {
            let members = &struct_members[struct_name];

            // Check if all children are already created.
            let all_children_ready = members.iter().all(|m| {
                m.child_struct_name
                    .as_ref()
                    .is_none_or(|child| struct_type_ids.contains_key(child))
            });

            if !all_children_ready {
                continue;
            }

            // Get the struct size from vmlinux BTF.
            let vmlinux_type_id = vmlinux_btf.find_struct_by_name(struct_name)?;
            let resolved_id = vmlinux_btf.resolve_type(vmlinux_type_id)?;
            let size = match &vmlinux_btf.types[resolved_id as usize] {
                btf_parser::BtfType::Struct(c, _) | btf_parser::BtfType::Union(c, _) => {
                    c.size_or_type
                }
                _ => bail!("expected struct/union for '{struct_name}'"),
            };

            // Build the member list for add_struct.
            let btf_members: Vec<(&str, u32, u32)> = members
                .iter()
                .map(|m| {
                    let type_id = match &m.child_struct_name {
                        Some(child) => *struct_type_ids.get(child).unwrap(),
                        None => placeholder_int_id,
                    };
                    (m.name.as_str(), type_id, m.bit_offset)
                })
                .collect();

            let type_id = prog_btf.add_struct(struct_name, size, &btf_members);
            struct_type_ids.insert(struct_name.clone(), type_id);
            remaining.remove(struct_name);
            made_progress = true;
        }

        if !made_progress {
            bail!(
                "circular dependency or missing struct in import: {:?}",
                remaining
            );
        }
    }

    // Phase 3: Generate CO-RE relocation records using the newly-added
    // struct type IDs.
    let mut writer = BtfExtWriter::new(&prog_btf);
    for (i, relo) in relo_entries.iter().enumerate() {
        // Use the local type_id from the structs we just added.
        let local_type_id = struct_type_ids
            .get(&relo.struct_name)
            .copied()
            .with_context(|| {
                format!("struct '{}' not found in local type map", relo.struct_name)
            })?;

        let access_str = prog_btf
            .compute_access_string(local_type_id, &relo.field_path)?;

        writer
            .add_relocation_with_type_id(relo, local_type_id, &access_str)
            .with_context(|| format!("processing auto-discovered relocation #{i}: {relo:?}"))?;
    }

    let (new_btf_data, new_btf_ext_data) = writer.finish(elf_data)?;
    elf_patcher::patch_elf_sections(elf_data, &new_btf_data, &new_btf_ext_data)
}

/// Collects the struct members needed for a given field path.
///
/// For "task_struct" + "scx.dsq_vtime", this records:
///   - task_struct needs member "scx" (pointing to sched_ext_entity)
///   - sched_ext_entity needs member "dsq_vtime" (leaf)
fn collect_struct_members(
    vmlinux_btf: &BtfInfo,
    struct_name: &str,
    field_path: &str,
    struct_members: &mut std::collections::HashMap<String, Vec<MemberAccess>>,
) -> Result<()> {
    let field_parts: Vec<&str> = field_path.split('.').collect();
    let mut current_type_id = vmlinux_btf.find_struct_by_name(struct_name)?;

    for (depth, field_name) in field_parts.iter().enumerate() {
        let resolved_id = vmlinux_btf.resolve_type(current_type_id)?;
        let ty = vmlinux_btf
            .types
            .get(resolved_id as usize)
            .context("type_id out of range")?;

        let (common, members) = match ty {
            btf_parser::BtfType::Struct(c, m) | btf_parser::BtfType::Union(c, m) => (c, m),
            _ => bail!("expected struct/union at type_id {resolved_id}"),
        };

        let parent_name = vmlinux_btf.string_at(common.name_off)?.to_string();
        let struct_size = common.size_or_type;
        let is_last = depth == field_parts.len() - 1;

        // Find the member in vmlinux BTF.
        let mut found = false;
        for member in members {
            let member_name = vmlinux_btf.string_at(member.name_off)?;
            if member_name != *field_name {
                // Check anonymous members.
                if member.name_off == 0 {
                    if let Ok(inner_id) = vmlinux_btf.resolve_type(member.type_id) {
                        if let Some(inner_ty) = vmlinux_btf.types.get(inner_id as usize) {
                            if let btf_parser::BtfType::Struct(_, inner_members)
                            | btf_parser::BtfType::Union(_, inner_members) = inner_ty
                            {
                                for inner_member in inner_members {
                                    if let Ok(inner_name) =
                                        vmlinux_btf.string_at(inner_member.name_off)
                                    {
                                        if inner_name == *field_name {
                                            let base_off = if common.kind_flag() {
                                                member.offset & 0x00ff_ffff
                                            } else {
                                                member.offset
                                            };
                                            let inner_off = if common.kind_flag() {
                                                inner_member.offset & 0x00ff_ffff
                                            } else {
                                                inner_member.offset
                                            };
                                            let combined_offset = base_off + inner_off;

                                            // Determine child struct name
                                            let child_name = if !is_last {
                                                let child_resolved =
                                                    vmlinux_btf.resolve_type(inner_member.type_id)?;
                                                if let Some(child_ty) = vmlinux_btf
                                                    .types
                                                    .get(child_resolved as usize)
                                                {
                                                    match child_ty {
                                                        btf_parser::BtfType::Struct(c, _)
                                                        | btf_parser::BtfType::Union(c, _) => {
                                                            Some(
                                                                vmlinux_btf
                                                                    .string_at(c.name_off)?
                                                                    .to_string(),
                                                            )
                                                        }
                                                        _ => None,
                                                    }
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            };

                                            let entry = struct_members
                                                .entry(parent_name.clone())
                                                .or_default();
                                            entry.push(MemberAccess {
                                                name: field_name.to_string(),
                                                bit_offset: combined_offset,
                                                child_struct_name: child_name,
                                            });

                                            current_type_id = inner_member.type_id;
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if found {
                        break;
                    }
                }
                continue;
            }

            let bit_offset = if common.kind_flag() {
                member.offset & 0x00ff_ffff
            } else {
                member.offset
            };

            // Determine child struct name if this is an intermediate field.
            let child_name = if !is_last {
                let child_resolved = vmlinux_btf.resolve_type(member.type_id)?;
                if let Some(child_ty) = vmlinux_btf.types.get(child_resolved as usize) {
                    match child_ty {
                        btf_parser::BtfType::Struct(c, _)
                        | btf_parser::BtfType::Union(c, _) => {
                            Some(vmlinux_btf.string_at(c.name_off)?.to_string())
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let entry = struct_members
                .entry(parent_name.clone())
                .or_default();
            entry.push(MemberAccess {
                name: field_name.to_string(),
                bit_offset,
                child_struct_name: child_name,
            });

            current_type_id = member.type_id;
            found = true;
            break;
        }

        if !found {
            bail!(
                "field '{}' not found in vmlinux struct '{}'",
                field_name,
                parent_name
            );
        }
    }

    Ok(())
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
