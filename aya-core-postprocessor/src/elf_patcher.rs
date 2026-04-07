//! Direct ELF binary patching for replacing/adding section contents.
//!
//! BPF ELF objects are always ELF64, little-endian, relocatable (ET_REL).
//! This module rewrites the ELF by:
//!   1. Copying the ELF header
//!   2. Copying all sections, replacing .BTF and .BTF.ext data
//!   3. If no .BTF.ext section exists, appending a new one
//!   4. Rebuilding the section header table with updated offsets/sizes
//!
//! This approach avoids the `object::write::Object` API which doesn't
//! support BPF (EM_BPF = 247) well.

use anyhow::{Context, Result, bail};

/// ELF64 header size.
const ELF64_EHDR_SIZE: usize = 64;
/// ELF64 section header entry size.
const ELF64_SHDR_SIZE: usize = 64;

/// Patches an ELF file, replacing .BTF and .BTF.ext section contents.
///
/// Returns the modified ELF bytes.
pub fn patch_elf_sections(
    original: &[u8],
    new_btf: &[u8],
    new_btf_ext: &[u8],
) -> Result<Vec<u8>> {
    if original.len() < ELF64_EHDR_SIZE {
        bail!("ELF too short for header");
    }

    // Parse ELF header fields we need.
    let e_shoff = read_u64_le(original, 40) as usize;
    let e_shentsize = read_u16_le(original, 58) as usize;
    let e_shnum = read_u16_le(original, 60) as usize;
    let e_shstrndx = read_u16_le(original, 62) as usize;

    if e_shentsize != ELF64_SHDR_SIZE {
        bail!("unexpected section header entry size: {e_shentsize}");
    }
    if e_shoff + e_shnum * e_shentsize > original.len() {
        bail!("section header table extends beyond file");
    }

    // Parse section headers.
    let mut sections: Vec<SectionInfo> = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let shdr_off = e_shoff + i * ELF64_SHDR_SIZE;
        sections.push(SectionInfo {
            sh_name: read_u32_le(original, shdr_off),
            sh_type: read_u32_le(original, shdr_off + 4),
            sh_flags: read_u64_le(original, shdr_off + 8),
            sh_addr: read_u64_le(original, shdr_off + 16),
            sh_offset: read_u64_le(original, shdr_off + 24) as usize,
            sh_size: read_u64_le(original, shdr_off + 32) as usize,
            sh_link: read_u32_le(original, shdr_off + 40),
            sh_info: read_u32_le(original, shdr_off + 44),
            sh_addralign: read_u64_le(original, shdr_off + 48),
            sh_entsize: read_u64_le(original, shdr_off + 56),
        });
    }

    // Get the section name string table.
    let shstrtab = &sections[e_shstrndx];
    let shstrtab_data = &original[shstrtab.sh_offset..shstrtab.sh_offset + shstrtab.sh_size];

    // Find section indices by name.
    let section_name = |s: &SectionInfo| -> &str {
        let start = s.sh_name as usize;
        if start >= shstrtab_data.len() {
            return "";
        }
        let end = shstrtab_data[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| start + p)
            .unwrap_or(shstrtab_data.len());
        std::str::from_utf8(&shstrtab_data[start..end]).unwrap_or("")
    };

    let btf_idx = sections
        .iter()
        .position(|s| section_name(s) == ".BTF")
        .context("no .BTF section found")?;

    let btf_ext_idx = sections
        .iter()
        .position(|s| section_name(s) == ".BTF.ext");

    // Build the output ELF.
    //
    // Layout:
    //   ELF header (64 bytes)
    //   Section data (each section, aligned)
    //   Section header table (at the end)
    //
    // We need to handle the case where .BTF.ext doesn't exist yet.

    let mut output = Vec::new();

    // 1. Copy the ELF header (we'll patch e_shoff and e_shnum later).
    output.extend_from_slice(&original[..ELF64_EHDR_SIZE]);

    // 2. Emit section data, keeping track of new offsets.
    let needs_new_btf_ext = btf_ext_idx.is_none();
    let total_sections = if needs_new_btf_ext {
        e_shnum + 1
    } else {
        e_shnum
    };

    // We also need to extend the shstrtab if we're adding a new section.
    let mut new_shstrtab_data = shstrtab_data.to_vec();
    let btf_ext_name_off = if needs_new_btf_ext {
        let off = new_shstrtab_data.len();
        new_shstrtab_data.extend_from_slice(b".BTF.ext\0");
        off as u32
    } else {
        0
    };

    struct NewSection {
        original_index: usize,
        new_offset: usize,
        new_size: usize,
    }

    let mut new_offsets: Vec<NewSection> = Vec::with_capacity(total_sections);

    for (i, sec) in sections.iter().enumerate() {
        let name = section_name(sec);

        // SHT_NULL (index 0) has no data.
        if sec.sh_type == 0 {
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: 0,
                new_size: 0,
            });
            continue;
        }

        // SHT_NOBITS sections have no file data.
        if sec.sh_type == 8 {
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: 0,
                new_size: sec.sh_size,
            });
            continue;
        }

        // Align the output position.
        let align = sec.sh_addralign.max(1) as usize;
        let padding = (align - (output.len() % align)) % align;
        output.extend(std::iter::repeat(0u8).take(padding));

        let offset = output.len();

        // Determine the data to write.
        if i == btf_idx {
            output.extend_from_slice(new_btf);
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: offset,
                new_size: new_btf.len(),
            });
        } else if Some(i) == btf_ext_idx {
            output.extend_from_slice(new_btf_ext);
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: offset,
                new_size: new_btf_ext.len(),
            });
        } else if name == ".rel.BTF.ext" {
            // CRITICAL: When we replace .BTF.ext, the old ELF relocations
            // in .rel.BTF.ext become stale — they reference offsets in the
            // old .BTF.ext layout. Applying these to the new .BTF.ext would
            // corrupt type_ids and access strings, causing "field relocation
            // on a type that doesn't have fields" errors at load time.
            //
            // Zero out the relocation section to prevent corruption.
            // Our new .BTF.ext data doesn't need ELF relocations because
            // all offsets are resolved by the postprocessor.
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: offset,
                new_size: 0,
            });
        } else if name == ".rel.BTF" {
            // Same issue: .rel.BTF relocations reference the old .BTF
            // layout. Zero them out since our new BTF is self-contained.
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: offset,
                new_size: 0,
            });
        } else if i == e_shstrndx && needs_new_btf_ext {
            // Extended shstrtab with the new ".BTF.ext" name.
            output.extend_from_slice(&new_shstrtab_data);
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: offset,
                new_size: new_shstrtab_data.len(),
            });
        } else {
            // Copy original section data.
            let data = &original[sec.sh_offset..sec.sh_offset + sec.sh_size];
            output.extend_from_slice(data);
            new_offsets.push(NewSection {
                original_index: i,
                new_offset: offset,
                new_size: sec.sh_size,
            });
        }
    }

    // If we need a new .BTF.ext section, append it.
    if needs_new_btf_ext {
        let align = 4usize;
        let padding = (align - (output.len() % align)) % align;
        output.extend(std::iter::repeat(0u8).take(padding));

        let offset = output.len();
        output.extend_from_slice(new_btf_ext);
        new_offsets.push(NewSection {
            original_index: usize::MAX, // sentinel for new section
            new_offset: offset,
            new_size: new_btf_ext.len(),
        });
    }

    // 3. Align for section header table.
    let align = 8usize;
    let padding = (align - (output.len() % align)) % align;
    output.extend(std::iter::repeat(0u8).take(padding));

    let new_shoff = output.len();

    // 4. Write section headers.
    for (_i, new_sec) in new_offsets.iter().enumerate() {
        if new_sec.original_index == usize::MAX {
            // New .BTF.ext section header.
            let mut shdr = [0u8; ELF64_SHDR_SIZE];
            write_u32_le(&mut shdr, 0, btf_ext_name_off); // sh_name
            write_u32_le(&mut shdr, 4, 1); // sh_type = SHT_PROGBITS
            write_u64_le(&mut shdr, 8, 0); // sh_flags
            write_u64_le(&mut shdr, 16, 0); // sh_addr
            write_u64_le(&mut shdr, 24, new_sec.new_offset as u64); // sh_offset
            write_u64_le(&mut shdr, 32, new_sec.new_size as u64); // sh_size
            write_u32_le(&mut shdr, 40, 0); // sh_link
            write_u32_le(&mut shdr, 44, 0); // sh_info
            write_u64_le(&mut shdr, 48, 4); // sh_addralign
            write_u64_le(&mut shdr, 56, 0); // sh_entsize
            output.extend_from_slice(&shdr);
        } else {
            // Copy original section header with updated offset/size.
            let orig_idx = new_sec.original_index;
            let orig_shdr_off = e_shoff + orig_idx * ELF64_SHDR_SIZE;
            let mut shdr = [0u8; ELF64_SHDR_SIZE];
            shdr.copy_from_slice(&original[orig_shdr_off..orig_shdr_off + ELF64_SHDR_SIZE]);

            // Update offset and size.
            if sections[orig_idx].sh_type != 0 && sections[orig_idx].sh_type != 8 {
                write_u64_le(&mut shdr, 24, new_sec.new_offset as u64);
                write_u64_le(&mut shdr, 32, new_sec.new_size as u64);
            }
            output.extend_from_slice(&shdr);
        }
    }

    // 5. Patch ELF header: e_shoff, e_shnum.
    write_u64_le(&mut output, 40, new_shoff as u64);
    write_u16_le(&mut output, 60, total_sections as u16);

    Ok(output)
}

#[derive(Debug)]
#[allow(dead_code)]
struct SectionInfo {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: usize,
    sh_size: usize,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

// Little-endian read/write helpers.

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal BPF ELF with .BTF, .BTF.ext, .rel.BTF, .rel.BTF.ext,
    /// and a program section.
    fn build_test_elf_with_rel_sections() -> Vec<u8> {
        // Build section name string table
        let mut shstrtab = vec![0u8]; // index 0 = empty string
        let btf_name = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b".BTF\0");
        let btf_ext_name = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b".BTF.ext\0");
        let rel_btf_name = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b".rel.BTF\0");
        let rel_btf_ext_name = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b".rel.BTF.ext\0");
        let shstrtab_name = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b".shstrtab\0");
        let tp_test_name = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b"tp/test\0");

        // Original .BTF content (16 bytes of dummy data)
        let orig_btf = vec![0x9Fu8, 0xEB, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        // Original .BTF.ext content (32 bytes of dummy data)
        let orig_btf_ext = vec![0x9F, 0xEB, 1, 0, 32, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0];

        // .rel.BTF — stale relocation data (24 bytes = 3 entries of 8 bytes each)
        let rel_btf = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
                           0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                           0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11];

        // .rel.BTF.ext — stale relocation data
        let rel_btf_ext = vec![0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88,
                               0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00];

        // Program section (4 NOP instructions)
        let tp_test = vec![0x07u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

        // Build ELF: header + section data + section headers
        // Sections: [0]=NULL, [1]=.BTF, [2]=.BTF.ext, [3]=.rel.BTF,
        //           [4]=.rel.BTF.ext, [5]=.shstrtab, [6]=tp/test
        let num_sections = 7usize;
        let shstrtab_idx = 5usize;

        let mut elf = vec![0u8; ELF64_EHDR_SIZE];
        // ELF magic
        elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        elf[4] = 2; // ELFCLASS64
        elf[5] = 1; // ELFDATA2LSB
        elf[6] = 1; // EV_CURRENT
        write_u16_le(&mut elf, 16, 1); // e_type = ET_REL
        write_u16_le(&mut elf, 18, 247); // e_machine = EM_BPF
        write_u32_le(&mut elf, 20, 1); // e_version
        write_u16_le(&mut elf, 52, ELF64_EHDR_SIZE as u16); // e_ehsize
        write_u16_le(&mut elf, 58, ELF64_SHDR_SIZE as u16); // e_shentsize
        write_u16_le(&mut elf, 62, shstrtab_idx as u16); // e_shstrndx

        // Write section data (each aligned to 4 bytes)
        let sections_data: Vec<(&[u8], u32, u32)> = vec![
            // (data, sh_name, sh_type)
            (&orig_btf, btf_name, 1),           // .BTF (SHT_PROGBITS)
            (&orig_btf_ext, btf_ext_name, 1),   // .BTF.ext (SHT_PROGBITS)
            (&rel_btf, rel_btf_name, 9),        // .rel.BTF (SHT_REL)
            (&rel_btf_ext, rel_btf_ext_name, 9),// .rel.BTF.ext (SHT_REL)
            (&shstrtab, shstrtab_name, 3),       // .shstrtab (SHT_STRTAB)
            (&tp_test, tp_test_name, 1),         // tp/test (SHT_PROGBITS)
        ];

        let mut offsets = vec![(0usize, 0usize)]; // NULL section
        for (data, _, _) in &sections_data {
            let align = 4;
            let padding = (align - (elf.len() % align)) % align;
            elf.extend(std::iter::repeat(0u8).take(padding));
            let offset = elf.len();
            elf.extend_from_slice(data);
            offsets.push((offset, data.len()));
        }

        // Align for section headers
        let padding = (8 - (elf.len() % 8)) % 8;
        elf.extend(std::iter::repeat(0u8).take(padding));
        let shoff = elf.len();
        write_u64_le(&mut elf, 40, shoff as u64); // e_shoff
        write_u16_le(&mut elf, 60, num_sections as u16); // e_shnum

        // Write section headers
        // [0] NULL
        elf.extend_from_slice(&[0u8; ELF64_SHDR_SIZE]);

        for (i, (data, sh_name, sh_type)) in sections_data.iter().enumerate() {
            let (off, sz) = offsets[i + 1];
            let mut shdr = [0u8; ELF64_SHDR_SIZE];
            write_u32_le(&mut shdr, 0, *sh_name);
            write_u32_le(&mut shdr, 4, *sh_type);
            write_u64_le(&mut shdr, 24, off as u64);
            write_u64_le(&mut shdr, 32, sz as u64);
            write_u64_le(&mut shdr, 48, 4); // sh_addralign
            if *sh_type == 9 { // SHT_REL
                write_u64_le(&mut shdr, 56, 16); // sh_entsize
            }
            let _ = data; // suppress unused warning
            elf.extend_from_slice(&shdr);
        }

        elf
    }

    #[test]
    fn test_rel_btf_sections_are_zeroed() {
        let elf = build_test_elf_with_rel_sections();

        // Verify the original has non-zero .rel.BTF and .rel.BTF.ext
        let e_shoff = read_u64_le(&elf, 40) as usize;
        let e_shnum = read_u16_le(&elf, 60) as usize;
        let e_shstrndx = read_u16_le(&elf, 62) as usize;
        let shstrtab_shdr = e_shoff + e_shstrndx * ELF64_SHDR_SIZE;
        let shstrtab_off = read_u64_le(&elf, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&elf, shstrtab_shdr + 32) as usize;
        let shstrtab_data = &elf[shstrtab_off..shstrtab_off + shstrtab_size];

        let section_name = |idx: usize| -> String {
            let shdr = e_shoff + idx * ELF64_SHDR_SIZE;
            let name_off = read_u32_le(&elf, shdr) as usize;
            let end = shstrtab_data[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab_data.len());
            String::from_utf8_lossy(&shstrtab_data[name_off..end]).to_string()
        };

        // Find .rel.BTF section and verify it has non-zero content
        let mut found_rel_btf = false;
        for i in 0..e_shnum {
            if section_name(i) == ".rel.BTF" {
                let shdr = e_shoff + i * ELF64_SHDR_SIZE;
                let size = read_u64_le(&elf, shdr + 32) as usize;
                assert!(size > 0, "original .rel.BTF should have data");
                found_rel_btf = true;
            }
        }
        assert!(found_rel_btf, ".rel.BTF section should exist in test ELF");

        // Patch the ELF
        let new_btf = vec![0x9Fu8, 0xEB, 1, 0, 24, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 42, 0, 0, 0]; // slightly different
        let new_btf_ext = vec![0x9F, 0xEB, 1, 0, 32, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0];

        let patched = patch_elf_sections(&elf, &new_btf, &new_btf_ext).unwrap();

        // Verify .rel.BTF and .rel.BTF.ext sections are zeroed (size=0)
        let e_shoff = read_u64_le(&patched, 40) as usize;
        let e_shnum = read_u16_le(&patched, 60) as usize;
        let e_shstrndx = read_u16_le(&patched, 62) as usize;
        let shstrtab_shdr = e_shoff + e_shstrndx * ELF64_SHDR_SIZE;
        let shstrtab_off = read_u64_le(&patched, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&patched, shstrtab_shdr + 32) as usize;
        let shstrtab_data = &patched[shstrtab_off..shstrtab_off + shstrtab_size];

        let section_name = |idx: usize| -> String {
            let shdr = e_shoff + idx * ELF64_SHDR_SIZE;
            let name_off = read_u32_le(&patched, shdr) as usize;
            let end = shstrtab_data[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab_data.len());
            String::from_utf8_lossy(&shstrtab_data[name_off..end]).to_string()
        };

        for i in 0..e_shnum {
            let name = section_name(i);
            let shdr = e_shoff + i * ELF64_SHDR_SIZE;
            let size = read_u64_le(&patched, shdr + 32) as usize;

            if name == ".rel.BTF" || name == ".rel.BTF.ext" {
                assert_eq!(
                    size, 0,
                    "{name} section should have size=0 after patching (was stale)"
                );
            } else if name == ".BTF" {
                assert_eq!(size, new_btf.len(), ".BTF should have new content size");
            } else if name == ".BTF.ext" {
                assert_eq!(size, new_btf_ext.len(), ".BTF.ext should have new content size");
            }
        }
    }

    #[test]
    fn test_btf_content_is_replaced() {
        let elf = build_test_elf_with_rel_sections();
        let new_btf = vec![0xAAu8; 20]; // distinctive content
        let new_btf_ext = vec![0xBBu8; 36];

        let patched = patch_elf_sections(&elf, &new_btf, &new_btf_ext).unwrap();

        // Find .BTF section and verify it has new content
        let e_shoff = read_u64_le(&patched, 40) as usize;
        let e_shnum = read_u16_le(&patched, 60) as usize;
        let e_shstrndx = read_u16_le(&patched, 62) as usize;
        let shstrtab_shdr = e_shoff + e_shstrndx * ELF64_SHDR_SIZE;
        let shstrtab_off = read_u64_le(&patched, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&patched, shstrtab_shdr + 32) as usize;
        let shstrtab_data = &patched[shstrtab_off..shstrtab_off + shstrtab_size];

        let section_name = |idx: usize| -> String {
            let shdr = e_shoff + idx * ELF64_SHDR_SIZE;
            let name_off = read_u32_le(&patched, shdr) as usize;
            let end = shstrtab_data[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab_data.len());
            String::from_utf8_lossy(&shstrtab_data[name_off..end]).to_string()
        };

        for i in 0..e_shnum {
            let shdr = e_shoff + i * ELF64_SHDR_SIZE;
            let off = read_u64_le(&patched, shdr + 24) as usize;
            let size = read_u64_le(&patched, shdr + 32) as usize;

            if section_name(i) == ".BTF" {
                assert_eq!(size, 20);
                assert_eq!(&patched[off..off + size], &[0xAA; 20]);
            } else if section_name(i) == ".BTF.ext" {
                assert_eq!(size, 36);
                assert_eq!(&patched[off..off + size], &[0xBB; 36]);
            }
        }
    }

    #[test]
    fn test_program_section_preserved() {
        let elf = build_test_elf_with_rel_sections();
        let new_btf = vec![0u8; 16];
        let new_btf_ext = vec![0u8; 32];

        let patched = patch_elf_sections(&elf, &new_btf, &new_btf_ext).unwrap();

        // Verify tp/test program section is unchanged
        let e_shoff = read_u64_le(&patched, 40) as usize;
        let e_shnum = read_u16_le(&patched, 60) as usize;
        let e_shstrndx = read_u16_le(&patched, 62) as usize;
        let shstrtab_shdr = e_shoff + e_shstrndx * ELF64_SHDR_SIZE;
        let shstrtab_off = read_u64_le(&patched, shstrtab_shdr + 24) as usize;
        let shstrtab_size = read_u64_le(&patched, shstrtab_shdr + 32) as usize;
        let shstrtab_data = &patched[shstrtab_off..shstrtab_off + shstrtab_size];

        let section_name = |idx: usize| -> String {
            let shdr = e_shoff + idx * ELF64_SHDR_SIZE;
            let name_off = read_u32_le(&patched, shdr) as usize;
            let end = shstrtab_data[name_off..].iter().position(|&b| b == 0)
                .map(|p| name_off + p).unwrap_or(shstrtab_data.len());
            String::from_utf8_lossy(&shstrtab_data[name_off..end]).to_string()
        };

        let mut found = false;
        for i in 0..e_shnum {
            if section_name(i) == "tp/test" {
                let shdr = e_shoff + i * ELF64_SHDR_SIZE;
                let size = read_u64_le(&patched, shdr + 32) as usize;
                assert_eq!(size, 32, "tp/test section should be preserved");
                found = true;
            }
        }
        assert!(found, "tp/test section should exist");
    }
}
