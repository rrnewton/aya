//! BPF instruction scanner for matching field access patterns.
//!
//! When using `.aya.core_relo` markers (instead of a sidecar TOML), we
//! know *which* struct fields are accessed but not *which instruction*
//! performs the access.  This module scans BPF instructions in program
//! sections to find instructions whose offset or immediate matches a
//! known field offset from BTF.
//!
//! ## Matching strategy
//!
//! The scanner matches three kinds of BPF instructions:
//!
//! 1. **ALU64 ADD IMM** (opcode `0x07`): The `core_read!` macro computes
//!    a field pointer using `offset_of!` and then calls
//!    `bpf_probe_read_kernel`. The compiler turns `offset_of!` into an
//!    immediate ADD to the base pointer.
//!
//! 2. **STX MEM** (opcodes `0x63`/`0x6b`/`0x73`/`0x7b`): The `core_write!`
//!    macro produces a `write_volatile` that compiles to a direct store
//!    instruction: `*(u64 *)(rX + off) = rY`. The field offset appears
//!    in the instruction's `off` field (16-bit signed, bytes 2-3).
//!
//! 3. **LDX MEM** (opcodes `0x61`/`0x69`/`0x71`/`0x79`): Direct loads
//!    from struct fields: `rY = *(u64 *)(rX + off)`. Same `off` field
//!    encoding as STX.
//!
//! The ALU match looks at the 32-bit immediate (bytes 4-7). The STX/LDX
//! match looks at the 16-bit offset (bytes 2-3).
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

// ── BPF opcode constants ──────────────────────────────────────────────

// ALU64 ADD with immediate: dst += imm
const BPF_ALU64_ADD_IMM: u8 = 0x07;

// STX MEM variants: *(size *)(dst + off) = src
const BPF_STX_MEM_B: u8 = 0x73;  // 8-bit store
const BPF_STX_MEM_H: u8 = 0x6b;  // 16-bit store
const BPF_STX_MEM_W: u8 = 0x63;  // 32-bit store
const BPF_STX_MEM_DW: u8 = 0x7b; // 64-bit store

// LDX MEM variants: dst = *(size *)(src + off)
const BPF_LDX_MEM_B: u8 = 0x71;  // 8-bit load
const BPF_LDX_MEM_H: u8 = 0x69;  // 16-bit load
const BPF_LDX_MEM_W: u8 = 0x61;  // 32-bit load
const BPF_LDX_MEM_DW: u8 = 0x79; // 64-bit load

// ST MEM variants: *(size *)(dst + off) = imm
const BPF_ST_MEM_B: u8 = 0x72;   // 8-bit store immediate
const BPF_ST_MEM_H: u8 = 0x6a;   // 16-bit store immediate
const BPF_ST_MEM_W: u8 = 0x62;   // 32-bit store immediate
const BPF_ST_MEM_DW: u8 = 0x7a;  // 64-bit store immediate

/// Returns true if the opcode is a STX MEM, LDX MEM, or ST MEM instruction
/// (i.e. an instruction where the field offset is in the `off` field).
fn is_mem_access_opcode(opcode: u8) -> bool {
    matches!(
        opcode,
        BPF_STX_MEM_B | BPF_STX_MEM_H | BPF_STX_MEM_W | BPF_STX_MEM_DW
        | BPF_LDX_MEM_B | BPF_LDX_MEM_H | BPF_LDX_MEM_W | BPF_LDX_MEM_DW
        | BPF_ST_MEM_B | BPF_ST_MEM_H | BPF_ST_MEM_W | BPF_ST_MEM_DW
    )
}

/// A match found by the instruction scanner.
#[derive(Debug, Clone)]
pub struct InsnMatch {
    /// ELF section name containing the instruction.
    pub section_name: String,
    /// Instruction index within the section (0-based).
    pub insn_index: u32,
}

/// Scans all program sections in a BPF ELF for instructions that
/// reference the given byte offset.
///
/// Matches:
/// - ALU64 ADD IMM instructions whose immediate equals `byte_offset`
///   (used by `core_read!` which adds the offset to a base pointer)
/// - STX/ST/LDX MEM instructions whose `off` field equals `byte_offset`
///   (used by `core_write!` and direct field accesses)
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
            // The immediate is a signed 32-bit value at bytes 4-7.
            if opcode == BPF_ALU64_ADD_IMM {
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

            // Check for STX/LDX/ST MEM instructions.
            // The field offset is a signed 16-bit value at bytes 2-3.
            if is_mem_access_opcode(opcode) {
                let off = i16::from_le_bytes(
                    data[insn_off + 2..insn_off + 4].try_into().unwrap(),
                );

                if off >= 0 && off as u64 == byte_offset {
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

    #[test]
    fn test_stx_mem_opcodes() {
        // BPF_STX = 0x03, BPF_MEM = 0x60
        // BPF_STX_MEM_W  = 0x03 | 0x60 | 0x00 (BPF_W) = 0x63
        // BPF_STX_MEM_DW = 0x03 | 0x60 | 0x18 (BPF_DW) = 0x7b
        assert_eq!(BPF_STX_MEM_W, 0x63);
        assert_eq!(BPF_STX_MEM_DW, 0x7b);
        assert_eq!(BPF_STX_MEM_H, 0x6b);
        assert_eq!(BPF_STX_MEM_B, 0x73);
    }

    #[test]
    fn test_ldx_mem_opcodes() {
        // BPF_LDX = 0x01, BPF_MEM = 0x60
        // BPF_LDX_MEM_W  = 0x01 | 0x60 | 0x00 (BPF_W) = 0x61
        // BPF_LDX_MEM_DW = 0x01 | 0x60 | 0x18 (BPF_DW) = 0x79
        assert_eq!(BPF_LDX_MEM_W, 0x61);
        assert_eq!(BPF_LDX_MEM_DW, 0x79);
        assert_eq!(BPF_LDX_MEM_H, 0x69);
        assert_eq!(BPF_LDX_MEM_B, 0x71);
    }

    #[test]
    fn test_st_mem_opcodes() {
        // BPF_ST = 0x02, BPF_MEM = 0x60
        // BPF_ST_MEM_W  = 0x02 | 0x60 | 0x00 (BPF_W) = 0x62
        // BPF_ST_MEM_DW = 0x02 | 0x60 | 0x18 (BPF_DW) = 0x7a
        assert_eq!(BPF_ST_MEM_W, 0x62);
        assert_eq!(BPF_ST_MEM_DW, 0x7a);
        assert_eq!(BPF_ST_MEM_H, 0x6a);
        assert_eq!(BPF_ST_MEM_B, 0x72);
    }

    #[test]
    fn test_is_mem_access_opcode() {
        // STX opcodes
        assert!(is_mem_access_opcode(BPF_STX_MEM_B));
        assert!(is_mem_access_opcode(BPF_STX_MEM_H));
        assert!(is_mem_access_opcode(BPF_STX_MEM_W));
        assert!(is_mem_access_opcode(BPF_STX_MEM_DW));
        // LDX opcodes
        assert!(is_mem_access_opcode(BPF_LDX_MEM_B));
        assert!(is_mem_access_opcode(BPF_LDX_MEM_H));
        assert!(is_mem_access_opcode(BPF_LDX_MEM_W));
        assert!(is_mem_access_opcode(BPF_LDX_MEM_DW));
        // ST opcodes
        assert!(is_mem_access_opcode(BPF_ST_MEM_B));
        assert!(is_mem_access_opcode(BPF_ST_MEM_H));
        assert!(is_mem_access_opcode(BPF_ST_MEM_W));
        assert!(is_mem_access_opcode(BPF_ST_MEM_DW));
        // Non-mem opcodes
        assert!(!is_mem_access_opcode(BPF_ALU64_ADD_IMM));
        assert!(!is_mem_access_opcode(0x85)); // BPF_CALL
        assert!(!is_mem_access_opcode(0x95)); // BPF_EXIT
        assert!(!is_mem_access_opcode(0x00)); // nop
    }
}
