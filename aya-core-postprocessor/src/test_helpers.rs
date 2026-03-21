//! Integration-level tests for the post-processor.
//!
//! These tests create minimal synthetic BPF ELF objects with .BTF sections
//! containing known struct types, then run the post-processor and verify
//! the output .BTF.ext section contains correct CO-RE relocation records.

use anyhow::Result;

/// Builds a minimal BPF ELF64 object file with a .BTF section.
///
/// The BTF contains:
///   - type 0: void
///   - type 1: struct "inner_struct" { u32 field_a; u64 field_b; }
///   - type 2: u32 (INT, 4 bytes)
///   - type 3: u64 (INT, 8 bytes)
///   - type 4: struct "outer_struct" { u32 x; struct inner_struct nested; }
///
/// The ELF has sections:
///   [0] "" (SHT_NULL)
///   [1] ".text" (SHT_PROGBITS) - fake BPF instructions
///   [2] ".BTF" (SHT_PROGBITS) - the BTF data
///   [3] "tp/test" (SHT_PROGBITS) - a fake program section
///   [4] ".shstrtab" (SHT_STRTAB) - section name string table
fn build_test_elf() -> Vec<u8> {
    // -- Build BTF data --
    let mut btf_strings = vec![0u8]; // offset 0 = empty string

    let add_str = |strings: &mut Vec<u8>, s: &str| -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(s.as_bytes());
        strings.push(0);
        off
    };

    let inner_struct_name_off = add_str(&mut btf_strings, "inner_struct");
    let field_a_name_off = add_str(&mut btf_strings, "field_a");
    let field_b_name_off = add_str(&mut btf_strings, "field_b");
    let u32_name_off = add_str(&mut btf_strings, "u32");
    let u64_name_off = add_str(&mut btf_strings, "u64");
    let outer_struct_name_off = add_str(&mut btf_strings, "outer_struct");
    let x_name_off = add_str(&mut btf_strings, "x");
    let nested_name_off = add_str(&mut btf_strings, "nested");

    // Build type data
    let mut type_data = Vec::new();

    // Helper to write a type header (12 bytes)
    let write_type_hdr = |buf: &mut Vec<u8>, name_off: u32, info: u32, size_or_type: u32| {
        buf.extend_from_slice(&name_off.to_le_bytes());
        buf.extend_from_slice(&info.to_le_bytes());
        buf.extend_from_slice(&size_or_type.to_le_bytes());
    };

    // Type 1: struct inner_struct { u32 field_a; u64 field_b; }
    // BTF_KIND_STRUCT = 4, vlen = 2, size = 16 (assuming 4-byte padding after field_a)
    // Actually for BPF/vmlinux, field_b at offset 8, total size = 16.
    let info = (4u32 << 24) | 2; // kind=STRUCT, vlen=2
    write_type_hdr(&mut type_data, inner_struct_name_off, info, 16);
    // member 0: field_a, type_id=2 (u32), offset=0 bits
    type_data.extend_from_slice(&field_a_name_off.to_le_bytes());
    type_data.extend_from_slice(&2u32.to_le_bytes()); // type_id
    type_data.extend_from_slice(&0u32.to_le_bytes()); // bit offset
    // member 1: field_b, type_id=3 (u64), offset=64 bits (8 bytes)
    type_data.extend_from_slice(&field_b_name_off.to_le_bytes());
    type_data.extend_from_slice(&3u32.to_le_bytes()); // type_id
    type_data.extend_from_slice(&64u32.to_le_bytes()); // bit offset

    // Type 2: INT u32 (4 bytes)
    let info = (1u32 << 24) | 0; // kind=INT, vlen=0
    write_type_hdr(&mut type_data, u32_name_off, info, 4);
    // INT extra data: encoding=0, offset=0, bits=32 -> (32 << 0) | (0 << 16) | 0
    type_data.extend_from_slice(&32u32.to_le_bytes());

    // Type 3: INT u64 (8 bytes)
    let info = (1u32 << 24) | 0;
    write_type_hdr(&mut type_data, u64_name_off, info, 8);
    type_data.extend_from_slice(&64u32.to_le_bytes());

    // Type 4: struct outer_struct { u32 x; struct inner_struct nested; }
    let info = (4u32 << 24) | 2; // kind=STRUCT, vlen=2
    // size: x=4 bytes at offset 0, nested=16 bytes at offset 8 (after padding) -> total 24
    // Actually, let's say x at offset 0, nested at offset 4 (packed) or 8 (aligned).
    // For simplicity: x at bit offset 0, nested at bit offset 32 (offset 4 bytes, packed).
    write_type_hdr(&mut type_data, outer_struct_name_off, info, 20);
    // member 0: x, type_id=2 (u32), offset=0
    type_data.extend_from_slice(&x_name_off.to_le_bytes());
    type_data.extend_from_slice(&2u32.to_le_bytes());
    type_data.extend_from_slice(&0u32.to_le_bytes());
    // member 1: nested, type_id=1 (inner_struct), offset=32 bits (4 bytes)
    type_data.extend_from_slice(&nested_name_off.to_le_bytes());
    type_data.extend_from_slice(&1u32.to_le_bytes());
    type_data.extend_from_slice(&32u32.to_le_bytes()); // 32 bits = 4 bytes

    // Build BTF header
    let hdr_len = 24u32;
    let type_off = 0u32;
    let type_len = type_data.len() as u32;
    let str_off = type_len;
    let str_len = btf_strings.len() as u32;

    let mut btf_section_data = Vec::new();
    btf_section_data.extend_from_slice(&0xEB9Fu16.to_le_bytes()); // magic
    btf_section_data.push(1); // version
    btf_section_data.push(0); // flags
    btf_section_data.extend_from_slice(&hdr_len.to_le_bytes());
    btf_section_data.extend_from_slice(&type_off.to_le_bytes());
    btf_section_data.extend_from_slice(&type_len.to_le_bytes());
    btf_section_data.extend_from_slice(&str_off.to_le_bytes());
    btf_section_data.extend_from_slice(&str_len.to_le_bytes());
    btf_section_data.extend_from_slice(&type_data);
    btf_section_data.extend_from_slice(&btf_strings);

    // -- Build ELF --
    // Section name string table
    let mut shstrtab = vec![0u8]; // offset 0 = ""
    let text_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".text\0");
    let btf_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".BTF\0");
    let tp_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b"tp/test\0");
    let shstrtab_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".shstrtab\0");

    // Fake BPF instructions for .text section (16 instructions = 128 bytes)
    let text_data = vec![0u8; 128];

    // Fake BPF instructions for tp/test section (16 instructions = 128 bytes)
    let tp_data = vec![0u8; 128];

    // Build the ELF file.
    // We have 5 sections:
    //   [0] SHT_NULL
    //   [1] .text
    //   [2] .BTF
    //   [3] tp/test
    //   [4] .shstrtab
    let num_sections = 5u16;
    let shstrtab_idx = 4u16;

    // Layout:
    //   0x00: ELF header (64 bytes)
    //   0x40: .text data (128 bytes)
    //   0xC0: .BTF data
    //   0xC0 + btf_len: tp/test data (128 bytes)
    //   then: .shstrtab data
    //   then: section header table (5 * 64 = 320 bytes)

    let text_offset = 64usize;
    let btf_offset = text_offset + text_data.len();
    let tp_offset = btf_offset + btf_section_data.len();
    // Align to 4
    let tp_offset = (tp_offset + 3) & !3;
    let shstrtab_offset = tp_offset + tp_data.len();
    let shdr_offset = shstrtab_offset + shstrtab.len();
    // Align to 8
    let shdr_offset = (shdr_offset + 7) & !7;

    let total_size = shdr_offset + num_sections as usize * 64;
    let mut elf = vec![0u8; total_size];

    // ELF header
    elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']); // e_ident magic
    elf[4] = 2; // ELFCLASS64
    elf[5] = 1; // ELFDATA2LSB
    elf[6] = 1; // EV_CURRENT
    // e_type = ET_REL (1)
    write_u16_le(&mut elf, 16, 1);
    // e_machine = EM_BPF (247)
    write_u16_le(&mut elf, 18, 247);
    // e_version = 1
    write_u32_le(&mut elf, 20, 1);
    // e_entry = 0
    // e_phoff = 0
    // e_shoff
    write_u64_le(&mut elf, 40, shdr_offset as u64);
    // e_flags = 0
    // e_ehsize = 64
    write_u16_le(&mut elf, 52, 64);
    // e_phentsize = 0
    // e_phnum = 0
    // e_shentsize = 64
    write_u16_le(&mut elf, 58, 64);
    // e_shnum
    write_u16_le(&mut elf, 60, num_sections);
    // e_shstrndx
    write_u16_le(&mut elf, 62, shstrtab_idx);

    // Copy section data
    elf[text_offset..text_offset + text_data.len()].copy_from_slice(&text_data);
    elf[btf_offset..btf_offset + btf_section_data.len()].copy_from_slice(&btf_section_data);
    elf[tp_offset..tp_offset + tp_data.len()].copy_from_slice(&tp_data);
    elf[shstrtab_offset..shstrtab_offset + shstrtab.len()].copy_from_slice(&shstrtab);

    // Section headers
    // [0] SHT_NULL - already zeroed

    // [1] .text
    let sh = shdr_offset + 64;
    write_u32_le(&mut elf, sh, text_name_off as u32); // sh_name
    write_u32_le(&mut elf, sh + 4, 1); // sh_type = SHT_PROGBITS
    write_u64_le(&mut elf, sh + 8, 0x6); // sh_flags = SHF_ALLOC|SHF_EXECINSTR
    write_u64_le(&mut elf, sh + 24, text_offset as u64); // sh_offset
    write_u64_le(&mut elf, sh + 32, text_data.len() as u64); // sh_size
    write_u64_le(&mut elf, sh + 48, 8); // sh_addralign

    // [2] .BTF
    let sh = shdr_offset + 128;
    write_u32_le(&mut elf, sh, btf_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 1); // SHT_PROGBITS
    write_u64_le(&mut elf, sh + 24, btf_offset as u64);
    write_u64_le(&mut elf, sh + 32, btf_section_data.len() as u64);
    write_u64_le(&mut elf, sh + 48, 4);

    // [3] tp/test
    let sh = shdr_offset + 192;
    write_u32_le(&mut elf, sh, tp_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 1); // SHT_PROGBITS
    write_u64_le(&mut elf, sh + 8, 0x6); // SHF_ALLOC|SHF_EXECINSTR
    write_u64_le(&mut elf, sh + 24, tp_offset as u64);
    write_u64_le(&mut elf, sh + 32, tp_data.len() as u64);
    write_u64_le(&mut elf, sh + 48, 8);

    // [4] .shstrtab
    let sh = shdr_offset + 256;
    write_u32_le(&mut elf, sh, shstrtab_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 3); // SHT_STRTAB
    write_u64_le(&mut elf, sh + 24, shstrtab_offset as u64);
    write_u64_le(&mut elf, sh + 32, shstrtab.len() as u64);
    write_u64_le(&mut elf, sh + 48, 1);

    elf
}

/// Helper: build a BPF instruction (8 bytes).
///
/// BPF instruction layout:
///   byte 0: opcode
///   byte 1: dst_reg(lo nibble) | src_reg(hi nibble)
///   bytes 2-3: offset (i16 LE)
///   bytes 4-7: immediate (i32 LE)
fn build_bpf_insn(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> [u8; 8] {
    let mut insn = [0u8; 8];
    insn[0] = opcode;
    insn[1] = (src << 4) | (dst & 0x0f);
    insn[2..4].copy_from_slice(&off.to_le_bytes());
    insn[4..8].copy_from_slice(&imm.to_le_bytes());
    insn
}

/// Builds a test ELF like `build_test_elf` but with specific BPF
/// instructions in the "tp/test" section that include an ALU64 ADD
/// with immediate=12 at instruction index 2.
///
/// Instruction layout in tp/test:
///   [0] mov r1, r6           (opcode=0xbf, nop-like)
///   [1] mov r2, 4            (opcode=0xb7, load immediate)
///   [2] add r1, 12           (opcode=0x07, ALU64 ADD IMM=12)
///   [3] call bpf_probe_read  (opcode=0x85)
///   [4..15] exit / padding   (opcode=0x95)
fn build_test_elf_with_insns() -> Vec<u8> {
    build_test_elf_inner(None)
}

/// Builds a test ELF with specific BPF instructions AND a
/// `.aya.core_relo` section containing a marker for
/// `outer_struct.nested.field_b`.
fn build_test_elf_with_markers_and_insns() -> Vec<u8> {
    // Build the marker data for outer_struct.nested.field_b
    let mut marker_data = Vec::new();
    marker_data.push(0xAC); // tag
    let name = b"outer_struct";
    marker_data.push(name.len() as u8);
    marker_data.extend_from_slice(name);
    let path = b"nested.field_b";
    marker_data.push(path.len() as u8);
    marker_data.extend_from_slice(path);

    build_test_elf_inner(Some(&marker_data))
}

/// Inner ELF builder that supports optional marker data and custom
/// BPF instructions.
///
/// Sections:
///   [0] "" (SHT_NULL)
///   [1] ".text" (SHT_PROGBITS)
///   [2] ".BTF" (SHT_PROGBITS)
///   [3] "tp/test" (SHT_PROGBITS) -- with specific instructions
///   [4] ".aya.core_relo" (SHT_PROGBITS) -- optional, if marker_data given
///   [N] ".shstrtab" (SHT_STRTAB)
fn build_test_elf_inner(marker_data: Option<&[u8]>) -> Vec<u8> {
    // -- Build BTF data (same as build_test_elf) --
    let mut btf_strings = vec![0u8]; // offset 0 = empty string

    let add_str = |strings: &mut Vec<u8>, s: &str| -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(s.as_bytes());
        strings.push(0);
        off
    };

    let inner_struct_name_off = add_str(&mut btf_strings, "inner_struct");
    let field_a_name_off = add_str(&mut btf_strings, "field_a");
    let field_b_name_off = add_str(&mut btf_strings, "field_b");
    let u32_name_off = add_str(&mut btf_strings, "u32");
    let u64_name_off = add_str(&mut btf_strings, "u64");
    let outer_struct_name_off = add_str(&mut btf_strings, "outer_struct");
    let x_name_off = add_str(&mut btf_strings, "x");
    let nested_name_off = add_str(&mut btf_strings, "nested");

    let mut type_data = Vec::new();

    let write_type_hdr = |buf: &mut Vec<u8>, name_off: u32, info: u32, size_or_type: u32| {
        buf.extend_from_slice(&name_off.to_le_bytes());
        buf.extend_from_slice(&info.to_le_bytes());
        buf.extend_from_slice(&size_or_type.to_le_bytes());
    };

    // Type 1: struct inner_struct { u32 field_a; u64 field_b; }
    let info = (4u32 << 24) | 2;
    write_type_hdr(&mut type_data, inner_struct_name_off, info, 16);
    type_data.extend_from_slice(&field_a_name_off.to_le_bytes());
    type_data.extend_from_slice(&2u32.to_le_bytes());
    type_data.extend_from_slice(&0u32.to_le_bytes());
    type_data.extend_from_slice(&field_b_name_off.to_le_bytes());
    type_data.extend_from_slice(&3u32.to_le_bytes());
    type_data.extend_from_slice(&64u32.to_le_bytes());

    // Type 2: INT u32 (4 bytes)
    let info = (1u32 << 24) | 0;
    write_type_hdr(&mut type_data, u32_name_off, info, 4);
    type_data.extend_from_slice(&32u32.to_le_bytes());

    // Type 3: INT u64 (8 bytes)
    let info = (1u32 << 24) | 0;
    write_type_hdr(&mut type_data, u64_name_off, info, 8);
    type_data.extend_from_slice(&64u32.to_le_bytes());

    // Type 4: struct outer_struct { u32 x; struct inner_struct nested; }
    let info = (4u32 << 24) | 2;
    write_type_hdr(&mut type_data, outer_struct_name_off, info, 20);
    type_data.extend_from_slice(&x_name_off.to_le_bytes());
    type_data.extend_from_slice(&2u32.to_le_bytes());
    type_data.extend_from_slice(&0u32.to_le_bytes());
    type_data.extend_from_slice(&nested_name_off.to_le_bytes());
    type_data.extend_from_slice(&1u32.to_le_bytes());
    type_data.extend_from_slice(&32u32.to_le_bytes());

    let hdr_len = 24u32;
    let type_off = 0u32;
    let type_len = type_data.len() as u32;
    let str_off = type_len;
    let str_len = btf_strings.len() as u32;

    let mut btf_section_data = Vec::new();
    btf_section_data.extend_from_slice(&0xEB9Fu16.to_le_bytes());
    btf_section_data.push(1);
    btf_section_data.push(0);
    btf_section_data.extend_from_slice(&hdr_len.to_le_bytes());
    btf_section_data.extend_from_slice(&type_off.to_le_bytes());
    btf_section_data.extend_from_slice(&type_len.to_le_bytes());
    btf_section_data.extend_from_slice(&str_off.to_le_bytes());
    btf_section_data.extend_from_slice(&str_len.to_le_bytes());
    btf_section_data.extend_from_slice(&type_data);
    btf_section_data.extend_from_slice(&btf_strings);

    // -- Build section name string table --
    let mut shstrtab = vec![0u8]; // offset 0 = ""
    let text_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".text\0");
    let btf_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".BTF\0");
    let tp_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b"tp/test\0");

    let marker_name_off = if marker_data.is_some() {
        let off = shstrtab.len();
        shstrtab.extend_from_slice(b".aya.core_relo\0");
        off
    } else {
        0
    };

    let shstrtab_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".shstrtab\0");

    // -- Build BPF instructions for tp/test --
    let mut tp_insns = Vec::new();
    // [0] mov r1, r6 (BPF_ALU64 | BPF_MOV | BPF_SRC_REG = 0xbf)
    tp_insns.extend_from_slice(&build_bpf_insn(0xbf, 1, 6, 0, 0));
    // [1] mov r2, 4 (BPF_ALU64 | BPF_MOV | BPF_SRC_IMM = 0xb7)
    tp_insns.extend_from_slice(&build_bpf_insn(0xb7, 2, 0, 0, 4));
    // [2] add r1, 12 (BPF_ALU64 | BPF_ADD | BPF_SRC_IMM = 0x07)
    tp_insns.extend_from_slice(&build_bpf_insn(0x07, 1, 0, 0, 12));
    // [3] call bpf_probe_read_kernel (BPF_JMP | BPF_CALL = 0x85, imm=113)
    tp_insns.extend_from_slice(&build_bpf_insn(0x85, 0, 0, 0, 113));
    // [4..15] exit instructions
    for _ in 4..16 {
        tp_insns.extend_from_slice(&build_bpf_insn(0x95, 0, 0, 0, 0));
    }

    let text_data = vec![0u8; 128]; // .text: 16 nop instructions

    // -- Compute layout --
    let has_marker = marker_data.is_some();
    let num_sections = if has_marker { 6u16 } else { 5u16 };
    let shstrtab_idx = num_sections - 1;

    let text_offset = 64usize;
    let btf_offset = text_offset + text_data.len();
    let tp_offset = (btf_offset + btf_section_data.len() + 3) & !3; // align to 4

    let marker_offset;
    let next_after_tp;
    if let Some(mdata) = marker_data {
        marker_offset = (tp_offset + tp_insns.len() + 3) & !3;
        next_after_tp = marker_offset + mdata.len();
    } else {
        marker_offset = 0;
        next_after_tp = tp_offset + tp_insns.len();
    }

    let shstrtab_offset = next_after_tp;
    let shdr_offset = (shstrtab_offset + shstrtab.len() + 7) & !7; // align to 8

    let total_size = shdr_offset + num_sections as usize * 64;
    let mut elf = vec![0u8; total_size];

    // ELF header
    elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    elf[4] = 2; // ELFCLASS64
    elf[5] = 1; // ELFDATA2LSB
    elf[6] = 1; // EV_CURRENT
    write_u16_le(&mut elf, 16, 1); // ET_REL
    write_u16_le(&mut elf, 18, 247); // EM_BPF
    write_u32_le(&mut elf, 20, 1); // e_version
    write_u64_le(&mut elf, 40, shdr_offset as u64);
    write_u16_le(&mut elf, 52, 64); // e_ehsize
    write_u16_le(&mut elf, 58, 64); // e_shentsize
    write_u16_le(&mut elf, 60, num_sections);
    write_u16_le(&mut elf, 62, shstrtab_idx);

    // Copy section data
    elf[text_offset..text_offset + text_data.len()].copy_from_slice(&text_data);
    elf[btf_offset..btf_offset + btf_section_data.len()].copy_from_slice(&btf_section_data);
    elf[tp_offset..tp_offset + tp_insns.len()].copy_from_slice(&tp_insns);

    if let Some(mdata) = marker_data {
        elf[marker_offset..marker_offset + mdata.len()].copy_from_slice(mdata);
    }

    elf[shstrtab_offset..shstrtab_offset + shstrtab.len()].copy_from_slice(&shstrtab);

    // Section headers
    let mut sh_idx = 0;

    // [0] SHT_NULL - already zeroed
    sh_idx += 1;

    // [1] .text
    let sh = shdr_offset + sh_idx * 64;
    write_u32_le(&mut elf, sh, text_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 1); // SHT_PROGBITS
    write_u64_le(&mut elf, sh + 8, 0x6); // SHF_ALLOC|SHF_EXECINSTR
    write_u64_le(&mut elf, sh + 24, text_offset as u64);
    write_u64_le(&mut elf, sh + 32, text_data.len() as u64);
    write_u64_le(&mut elf, sh + 48, 8);
    sh_idx += 1;

    // [2] .BTF
    let sh = shdr_offset + sh_idx * 64;
    write_u32_le(&mut elf, sh, btf_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 1); // SHT_PROGBITS
    write_u64_le(&mut elf, sh + 24, btf_offset as u64);
    write_u64_le(&mut elf, sh + 32, btf_section_data.len() as u64);
    write_u64_le(&mut elf, sh + 48, 4);
    sh_idx += 1;

    // [3] tp/test
    let sh = shdr_offset + sh_idx * 64;
    write_u32_le(&mut elf, sh, tp_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 1); // SHT_PROGBITS
    write_u64_le(&mut elf, sh + 8, 0x6); // SHF_ALLOC|SHF_EXECINSTR
    write_u64_le(&mut elf, sh + 24, tp_offset as u64);
    write_u64_le(&mut elf, sh + 32, tp_insns.len() as u64);
    write_u64_le(&mut elf, sh + 48, 8);
    sh_idx += 1;

    // [4] .aya.core_relo (optional)
    if let Some(mdata) = marker_data {
        let sh = shdr_offset + sh_idx * 64;
        write_u32_le(&mut elf, sh, marker_name_off as u32);
        write_u32_le(&mut elf, sh + 4, 1); // SHT_PROGBITS
        write_u64_le(&mut elf, sh + 24, marker_offset as u64);
        write_u64_le(&mut elf, sh + 32, mdata.len() as u64);
        write_u64_le(&mut elf, sh + 48, 1);
        sh_idx += 1;
    }

    // [N] .shstrtab
    let sh = shdr_offset + sh_idx * 64;
    write_u32_le(&mut elf, sh, shstrtab_name_off as u32);
    write_u32_le(&mut elf, sh + 4, 3); // SHT_STRTAB
    write_u64_le(&mut elf, sh + 24, shstrtab_offset as u64);
    write_u64_le(&mut elf, sh + 32, shstrtab.len() as u64);
    write_u64_le(&mut elf, sh + 48, 1);

    elf
}

fn write_u16_le(data: &mut [u8], offset: usize, val: u16) {
    data[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

fn write_u32_le(data: &mut [u8], offset: usize, val: u32) {
    data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

fn write_u64_le(data: &mut [u8], offset: usize, val: u64) {
    data[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::btf_ext_writer::BtfExtWriter;
    use crate::btf_parser::BtfInfo;
    use crate::sidecar::{RelocationEntry, SidecarConfig};

    #[test]
    fn test_build_test_elf_is_valid() {
        let elf = build_test_elf();
        // Check ELF magic
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        // Parse BTF from it
        let btf = BtfInfo::parse_from_elf(&elf).unwrap();
        assert_eq!(btf.header.magic, 0xEB9F);
        // We should have 5 types (void + 4 defined types)
        assert_eq!(btf.types.len(), 5);
    }

    #[test]
    fn test_btf_find_struct() {
        let elf = build_test_elf();
        let btf = BtfInfo::parse_from_elf(&elf).unwrap();

        let inner_id = btf.find_struct_by_name("inner_struct").unwrap();
        assert_eq!(inner_id, 1); // type_id 1

        let outer_id = btf.find_struct_by_name("outer_struct").unwrap();
        assert_eq!(outer_id, 4); // type_id 4

        assert!(btf.find_struct_by_name("nonexistent").is_err());
    }

    #[test]
    fn test_access_string_simple() {
        let elf = build_test_elf();
        let btf = BtfInfo::parse_from_elf(&elf).unwrap();

        // inner_struct.field_a -> member index 0 -> access string "0:0"
        let s = btf.compute_access_string(1, "field_a").unwrap();
        assert_eq!(s, "0:0");

        // inner_struct.field_b -> member index 1 -> access string "0:1"
        let s = btf.compute_access_string(1, "field_b").unwrap();
        assert_eq!(s, "0:1");
    }

    #[test]
    fn test_access_string_nested() {
        let elf = build_test_elf();
        let btf = BtfInfo::parse_from_elf(&elf).unwrap();

        // outer_struct.x -> "0:0"
        let s = btf.compute_access_string(4, "x").unwrap();
        assert_eq!(s, "0:0");

        // outer_struct.nested -> "0:1"
        let s = btf.compute_access_string(4, "nested").unwrap();
        assert_eq!(s, "0:1");

        // outer_struct.nested.field_b -> "0:1:1"
        let s = btf.compute_access_string(4, "nested.field_b").unwrap();
        assert_eq!(s, "0:1:1");

        // outer_struct.nested.field_a -> "0:1:0"
        let s = btf.compute_access_string(4, "nested.field_a").unwrap();
        assert_eq!(s, "0:1:0");
    }

    #[test]
    fn test_process_elf_adds_core_relo() {
        let elf = build_test_elf();

        let config = SidecarConfig {
            relocation: vec![
                RelocationEntry {
                    section: "tp/test".to_string(),
                    insn_index: 3,
                    struct_name: "outer_struct".to_string(),
                    field_path: "nested.field_b".to_string(),
                },
            ],
        };

        let result = crate::process_elf(&elf, &config).unwrap();

        // Verify the result is a valid ELF.
        assert_eq!(&result[0..4], &[0x7f, b'E', b'L', b'F']);

        // Parse the result and check for .BTF.ext section.
        let btf = BtfInfo::parse_from_elf(&result).unwrap();

        // Find the .BTF.ext section by scanning section headers.
        let e_shoff = read_u64_le(&result, 40) as usize;
        let e_shnum = read_u16_le(&result, 60) as usize;
        let e_shstrndx = read_u16_le(&result, 62) as usize;

        // Get shstrtab
        let shstrtab_shdr = e_shoff + e_shstrndx * 64;
        let shstrtab_off = read_u64_le(&result, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&result, shstrtab_shdr + 32) as usize;
        let shstrtab = &result[shstrtab_off..shstrtab_off + shstrtab_size];

        let section_name = |idx: usize| -> &str {
            let shdr = e_shoff + idx * 64;
            let name_off = read_u32_le(&result, shdr) as usize;
            let end = shstrtab[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab.len());
            std::str::from_utf8(&shstrtab[name_off..end]).unwrap_or("")
        };

        let mut btf_ext_idx = None;
        for i in 0..e_shnum {
            if section_name(i) == ".BTF.ext" {
                btf_ext_idx = Some(i);
                break;
            }
        }

        let btf_ext_idx = btf_ext_idx.expect(".BTF.ext section should exist in output");
        let btf_ext_shdr = e_shoff + btf_ext_idx * 64;
        let btf_ext_off = read_u64_le(&result, btf_ext_shdr + 24) as usize;
        let btf_ext_size = read_u64_le(&result, btf_ext_shdr + 32) as usize;
        let btf_ext_data = &result[btf_ext_off..btf_ext_off + btf_ext_size];

        // Parse the .BTF.ext header.
        assert!(btf_ext_data.len() >= 32, "BTF.ext too short: {} bytes", btf_ext_data.len());
        let magic = read_u16_le(btf_ext_data, 0);
        assert_eq!(magic, 0xEB9F);

        let hdr_len = read_u32_le(btf_ext_data, 4) as usize;
        let core_relo_off = read_u32_le(btf_ext_data, 24) as usize;
        let core_relo_len = read_u32_le(btf_ext_data, 28) as usize;

        assert!(core_relo_len > 0, "core_relo_len should be > 0");

        // Parse core_relo data.
        let relo_data = &btf_ext_data[hdr_len + core_relo_off..hdr_len + core_relo_off + core_relo_len];

        // First 4 bytes: rec_size
        let rec_size = read_u32_le(relo_data, 0);
        assert_eq!(rec_size, 16);

        // Then: sec_name_off (4) + num_info (4) + records
        let sec_name_off = read_u32_le(relo_data, 4);
        let num_info = read_u32_le(relo_data, 8);
        assert_eq!(num_info, 1);

        // Parse the relocation record.
        let insn_off = read_u32_le(relo_data, 12);
        let type_id = read_u32_le(relo_data, 16);
        let access_str_off = read_u32_le(relo_data, 20);
        let kind = read_u32_le(relo_data, 24);

        // insn_index=3, so insn_off = 3 * 8 = 24
        assert_eq!(insn_off, 24);
        // type_id should be 4 (outer_struct)
        assert_eq!(type_id, 4);
        // kind should be BPF_CORE_FIELD_BYTE_OFFSET = 0
        assert_eq!(kind, 0);

        // Verify the access string in BTF.
        let access_str = btf.string_at(access_str_off).unwrap();
        assert_eq!(access_str, "0:1:1"); // nested=member 1, field_b=member 1
    }

    #[test]
    fn test_process_elf_multiple_relos() {
        let elf = build_test_elf();

        let config = SidecarConfig {
            relocation: vec![
                RelocationEntry {
                    section: "tp/test".to_string(),
                    insn_index: 2,
                    struct_name: "outer_struct".to_string(),
                    field_path: "x".to_string(),
                },
                RelocationEntry {
                    section: "tp/test".to_string(),
                    insn_index: 5,
                    struct_name: "inner_struct".to_string(),
                    field_path: "field_b".to_string(),
                },
            ],
        };

        let result = crate::process_elf(&elf, &config).unwrap();

        // Just verify it's valid ELF and has .BTF.ext
        assert_eq!(&result[0..4], &[0x7f, b'E', b'L', b'F']);

        // Find .BTF.ext and check we have the right number of records
        let e_shoff = read_u64_le(&result, 40) as usize;
        let e_shnum = read_u16_le(&result, 60) as usize;
        let e_shstrndx = read_u16_le(&result, 62) as usize;

        let shstrtab_shdr = e_shoff + e_shstrndx * 64;
        let shstrtab_off = read_u64_le(&result, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&result, shstrtab_shdr + 32) as usize;
        let shstrtab = &result[shstrtab_off..shstrtab_off + shstrtab_size];

        for i in 0..e_shnum {
            let shdr = e_shoff + i * 64;
            let name_off = read_u32_le(&result, shdr) as usize;
            let end = shstrtab[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab.len());
            let name = std::str::from_utf8(&shstrtab[name_off..end]).unwrap_or("");

            if name == ".BTF.ext" {
                let ext_off = read_u64_le(&result, shdr + 24) as usize;
                let ext_size = read_u64_le(&result, shdr + 32) as usize;
                let ext_data = &result[ext_off..ext_off + ext_size];

                let hdr_len = read_u32_le(ext_data, 4) as usize;
                let core_relo_off = read_u32_le(ext_data, 24) as usize;
                let core_relo_len = read_u32_le(ext_data, 28) as usize;
                let relo_data = &ext_data[hdr_len + core_relo_off..hdr_len + core_relo_off + core_relo_len];

                // rec_size(4) + sec_name_off(4) + num_info(4) + 2 records(32) = 44
                // Both relos are in the same section so they should be grouped.
                let num_info = read_u32_le(relo_data, 8);
                assert_eq!(num_info, 2, "expected 2 relocations in the same section");

                // First record: insn_off = 2*8 = 16
                let insn_off_0 = read_u32_le(relo_data, 12);
                assert_eq!(insn_off_0, 16);

                // Second record: insn_off = 5*8 = 40
                let insn_off_1 = read_u32_le(relo_data, 28);
                assert_eq!(insn_off_1, 40);

                return;
            }
        }
        panic!(".BTF.ext not found in output");
    }

    // ---- Tests for new auto-discovery functionality ----

    #[test]
    fn test_compute_byte_offset_simple() {
        let elf = build_test_elf();
        let btf = BtfInfo::parse_from_elf(&elf).unwrap();

        // inner_struct.field_a at bit offset 0 -> byte offset 0
        let off = btf.compute_byte_offset(1, "field_a").unwrap();
        assert_eq!(off, 0);

        // inner_struct.field_b at bit offset 64 -> byte offset 8
        let off = btf.compute_byte_offset(1, "field_b").unwrap();
        assert_eq!(off, 8);
    }

    #[test]
    fn test_compute_byte_offset_nested() {
        let elf = build_test_elf();
        let btf = BtfInfo::parse_from_elf(&elf).unwrap();

        // outer_struct.x at bit offset 0 -> byte offset 0
        let off = btf.compute_byte_offset(4, "x").unwrap();
        assert_eq!(off, 0);

        // outer_struct.nested at bit offset 32 -> byte offset 4
        let off = btf.compute_byte_offset(4, "nested").unwrap();
        assert_eq!(off, 4);

        // outer_struct.nested.field_a: nested at 32 bits + field_a at 0 bits
        //   = 32 bits = 4 bytes
        let off = btf.compute_byte_offset(4, "nested.field_a").unwrap();
        assert_eq!(off, 4);

        // outer_struct.nested.field_b: nested at 32 bits + field_b at 64 bits
        //   = 96 bits = 12 bytes
        let off = btf.compute_byte_offset(4, "nested.field_b").unwrap();
        assert_eq!(off, 12);
    }

    #[test]
    fn test_insn_scanner_finds_add_immediate() {
        let elf = build_test_elf_with_insns();
        let matches = crate::insn_scanner::find_insns_with_offset(&elf, 12).unwrap();

        // The test ELF has an ALU64 ADD IMM with immediate=12 at
        // instruction index 2 in the "tp/test" section.
        assert!(!matches.is_empty(), "should find at least one match");
        let m = &matches[0];
        assert_eq!(m.section_name, "tp/test");
        assert_eq!(m.insn_index, 2);
    }

    #[test]
    fn test_insn_scanner_no_match() {
        let elf = build_test_elf_with_insns();
        let matches = crate::insn_scanner::find_insns_with_offset(&elf, 999).unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_marker_roundtrip() {
        // Build marker bytes and verify they parse correctly.
        let mut data = Vec::new();
        // Marker tag
        data.push(0xAC);
        // Struct name: "outer_struct"
        let name = b"outer_struct";
        data.push(name.len() as u8);
        data.extend_from_slice(name);
        // Field path: "nested.field_b"
        let path = b"nested.field_b";
        data.push(path.len() as u8);
        data.extend_from_slice(path);

        let markers = crate::marker_parser::parse_markers(&data).unwrap();
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].struct_name, "outer_struct");
        assert_eq!(markers[0].field_path, "nested.field_b");
    }

    #[test]
    fn test_auto_discovery_pipeline() {
        // Build an ELF with:
        //   - BTF containing inner_struct and outer_struct
        //   - A "tp/test" section with ALU64 ADD IMM=12 at insn index 2
        //   - A .aya.core_relo section with a marker for outer_struct.nested.field_b
        //
        // outer_struct.nested.field_b has byte offset 12 (nested at offset 4,
        // field_b within inner_struct at offset 8, total = 4+8 = 12).
        //
        // The auto-discovery pipeline should:
        //   1. Read the marker
        //   2. Compute byte offset = 12 (from vmlinux BTF)
        //   3. Find the ALU64 ADD IMM=12 instruction at insn index 2
        //   4. Generate a CO-RE relocation for it

        let elf = build_test_elf_with_markers_and_insns();

        // Extract the BTF section data to use as a fake vmlinux BTF file.
        // The test ELF contains real struct definitions that the postprocessor
        // needs for field resolution.
        let btf = crate::btf_parser::BtfInfo::parse_from_elf(&elf).unwrap();
        let vmlinux_btf_data = btf.to_bytes(&elf).unwrap();
        let tmp_dir = tempfile::tempdir().unwrap();
        let vmlinux_path = tmp_dir.path().join("vmlinux_btf");
        std::fs::write(&vmlinux_path, &vmlinux_btf_data).unwrap();

        let result = crate::process_elf_auto_with_vmlinux(
            &elf,
            vmlinux_path.to_str().unwrap(),
        )
        .unwrap();

        // Verify the result has a .BTF.ext section with a CO-RE relocation.
        assert_eq!(&result[0..4], &[0x7f, b'E', b'L', b'F']);

        let e_shoff = read_u64_le(&result, 40) as usize;
        let e_shnum = read_u16_le(&result, 60) as usize;
        let e_shstrndx = read_u16_le(&result, 62) as usize;

        let shstrtab_shdr = e_shoff + e_shstrndx * 64;
        let shstrtab_off = read_u64_le(&result, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&result, shstrtab_shdr + 32) as usize;
        let shstrtab_data = &result[shstrtab_off..shstrtab_off + shstrtab_size];

        let section_name = |idx: usize| -> &str {
            let shdr = e_shoff + idx * 64;
            let name_off = read_u32_le(&result, shdr) as usize;
            let end = shstrtab_data[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab_data.len());
            std::str::from_utf8(&shstrtab_data[name_off..end]).unwrap_or("")
        };

        let mut found_btf_ext = false;
        for i in 0..e_shnum {
            if section_name(i) == ".BTF.ext" {
                found_btf_ext = true;

                let shdr = e_shoff + i * 64;
                let ext_off = read_u64_le(&result, shdr + 24) as usize;
                let ext_size = read_u64_le(&result, shdr + 32) as usize;
                let ext_data = &result[ext_off..ext_off + ext_size];

                let hdr_len = read_u32_le(ext_data, 4) as usize;
                let core_relo_off = read_u32_le(ext_data, 24) as usize;
                let core_relo_len = read_u32_le(ext_data, 28) as usize;
                assert!(core_relo_len > 0, "should have core_relo records");

                let relo_data = &ext_data[hdr_len + core_relo_off..hdr_len + core_relo_off + core_relo_len];

                // Check rec_size
                let rec_size = read_u32_le(relo_data, 0);
                assert_eq!(rec_size, 16);

                // Check num_info
                let num_info = read_u32_le(relo_data, 8);
                assert_eq!(num_info, 1, "expected 1 relocation");

                // Check insn_off = 2 * 8 = 16
                let insn_off = read_u32_le(relo_data, 12);
                assert_eq!(insn_off, 16, "insn_off should be 16 (insn index 2)");

                // Check type_id points to outer_struct (the exact ID depends
                // on whether structs were imported from vmlinux or already
                // present in the program's BTF)
                let type_id = read_u32_le(relo_data, 16);
                let btf_check = BtfInfo::parse_from_elf(&result).unwrap();
                let type_name = match &btf_check.types[type_id as usize] {
                    crate::btf_parser::BtfType::Struct(c, _) => btf_check.string_at(c.name_off).unwrap(),
                    _ => panic!("type_id {} should be a struct", type_id),
                };
                assert_eq!(type_name, "outer_struct");

                // Check kind = 0 (BPF_CORE_FIELD_BYTE_OFFSET)
                let kind = read_u32_le(relo_data, 24);
                assert_eq!(kind, 0);

                // Verify access string
                let btf = BtfInfo::parse_from_elf(&result).unwrap();
                let access_str_off = read_u32_le(relo_data, 20);
                let access_str = btf.string_at(access_str_off).unwrap();
                assert_eq!(access_str, "0:0:0"); // nested=0, field_b=0 (stub structs have only accessed fields)

                break;
            }
        }

        assert!(found_btf_ext, ".BTF.ext section should exist in output");
    }
}
