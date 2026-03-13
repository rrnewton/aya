//! BPF instruction scanner for matching field access patterns.
//!
//! When using `.aya.core_relo` markers (instead of a sidecar TOML), we
//! know *which* struct fields are accessed but not *which instruction*
//! performs the access.  This module scans BPF instructions in program
//! sections to find instructions whose offset matches a known field
//! offset from BTF.
//!
//! ## Matching strategy
//!
//! The `core_read!` macro computes a field pointer using `offset_of!`
//! and then calls `bpf_probe_read_kernel`.  The compiler turns
//! `offset_of!` into an immediate ADD to the base pointer.  We look
//! for ALU64 ADD instructions with an immediate matching the expected
//! byte offset.
//!
//! This is a heuristic -- it may produce false positives if the same
//! constant appears in unrelated code.  For production use, the proc
//! macro should embed instruction-level labels.  For the prototype,
//! this heuristic works well enough because:
//!   1. Struct field offsets tend to be distinctive numbers
//!   2. We only scan program sections (not data sections)
//!   3. The marker tells us which struct/field to expect

use anyhow::{Context, Result};
use object::read::elf::ElfFile;
use object::{Endianness, Object, ObjectSection, elf};

/// Size of a single BPF instruction in bytes.
const BPF_INSN_SIZE: usize = 8;

/// BPF opcode classes
const BPF_ALU64: u8 = 0x07; // class bits for ALU64
const BPF_OP_ADD: u8 = 0x00;
const BPF_SRC_IMM: u8 = 0x00; // K source (immediate)

/// Full opcode for ALU64 ADD with immediate: 0x07
const BPF_ALU64_ADD_IMM: u8 = BPF_ALU64 | BPF_OP_ADD | BPF_SRC_IMM;

/// A match found by the instruction scanner.
#[derive(Debug, Clone)]
pub struct InsnMatch {
    /// ELF section name containing the instruction.
    pub section_name: String,
    /// Instruction index within the section (0-based).
    pub insn_index: u32,
}

/// Scans all program sections in a BPF ELF for ALU64 ADD instructions
/// whose immediate value matches the given byte offset.
///
/// Returns all matches across all program sections.
pub fn find_insns_with_offset(elf_data: &[u8], byte_offset: u64) -> Result<Vec<InsnMatch>> {
    let obj = ElfFile::<elf::FileHeader64<Endianness>>::parse(elf_data)
        .context("parsing ELF for instruction scan")?;

    let mut matches = Vec::new();

    for section in obj.sections() {
        let name = match section.name() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Skip non-program sections.
        if !is_program_section(name) {
            continue;
        }

        let data = match section.data() {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Scan instructions.
        let num_insns = data.len() / BPF_INSN_SIZE;
        for i in 0..num_insns {
            let insn_off = i * BPF_INSN_SIZE;
            let opcode = data[insn_off];

            // Check for ALU64 ADD with immediate (opcode 0x07).
            if opcode == BPF_ALU64_ADD_IMM {
                // The immediate is a signed 32-bit value at offset 4.
                let imm = i32::from_le_bytes(
                    data[insn_off + 4..insn_off + 8].try_into().unwrap(),
                );

                if imm >= 0 && imm as u64 == byte_offset {
                    matches.push(InsnMatch {
                        section_name: name.to_string(),
                        insn_index: i as u32,
                    });
                }
            }
        }
    }

    Ok(matches)
}

/// Returns the list of program section names in the ELF.
pub fn program_sections(elf_data: &[u8]) -> Result<Vec<String>> {
    let obj = ElfFile::<elf::FileHeader64<Endianness>>::parse(elf_data)
        .context("parsing ELF for section listing")?;

    let mut sections = Vec::new();
    for section in obj.sections() {
        if let Ok(name) = section.name() {
            if is_program_section(name) {
                sections.push(name.to_string());
            }
        }
    }
    Ok(sections)
}

/// Determines whether an ELF section name is a BPF program section.
///
/// BPF program sections typically have names like:
///   - "struct_ops/sched_ext_ops"
///   - "tp/sched/sched_switch"
///   - "tracepoint/sched/sched_switch"
///   - "kprobe/some_function"
///   - "xdp"
///   - ".text" (for subprograms)
///
/// We exclude metadata sections like ".BTF", ".maps", ".rodata", etc.
fn is_program_section(name: &str) -> bool {
    // Exclude well-known non-program sections.
    if name.is_empty()
        || name.starts_with(".BTF")
        || name == ".maps"
        || name == ".bss"
        || name.starts_with(".data")
        || name.starts_with(".rodata")
        || name == ".symtab"
        || name == ".strtab"
        || name == ".shstrtab"
        || name.starts_with(".rel")
        || name == ".aya.core_relo"
        || name == "license"
        || name == ".debug_info"
        || name.starts_with(".debug_")
    {
        return false;
    }

    // Accept everything else -- program sections have diverse naming
    // conventions and it's better to scan too many sections than miss one.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_program_section() {
        assert!(is_program_section(".text"));
        assert!(is_program_section("tp/test"));
        assert!(is_program_section("struct_ops/sched_ext_ops"));
        assert!(is_program_section("kprobe/sys_open"));
        assert!(is_program_section("xdp"));

        assert!(!is_program_section(".BTF"));
        assert!(!is_program_section(".BTF.ext"));
        assert!(!is_program_section(".maps"));
        assert!(!is_program_section(".rodata"));
        assert!(!is_program_section(".data"));
        assert!(!is_program_section(".bss"));
        assert!(!is_program_section(".rel.text"));
        assert!(!is_program_section(".aya.core_relo"));
        assert!(!is_program_section(""));
    }

    #[test]
    fn test_alu64_add_imm_opcode() {
        // BPF_ALU64 | BPF_OP_ADD | BPF_SRC_IMM = 0x07 | 0x00 | 0x00 = 0x07
        assert_eq!(BPF_ALU64_ADD_IMM, 0x07);
    }
}
