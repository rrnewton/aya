//! Generates `.BTF.ext` section data with CO-RE relocation records.
//!
//! This module handles:
//!   1. Parsing the existing `.BTF.ext` section (if present)
//!   2. Adding new `bpf_core_relo` records for field accesses
//!   3. Serializing the complete `.BTF.ext` section

use anyhow::{Context, Result, bail};
use object::read::elf::ElfFile;
use object::{Endianness, Object, ObjectSection, elf};

use crate::btf_parser::BtfInfo;
use crate::sidecar::RelocationEntry;

/// Size of the `bpf_core_relo` struct.
const CORE_RELO_SIZE: u32 = 16;

/// Size of the `btf_ext_header`.
const BTF_EXT_HEADER_SIZE: u32 = 32;

/// BTF.ext magic number.
const BTF_MAGIC: u16 = 0xEB9F;

/// BPF_CORE_FIELD_BYTE_OFFSET relocation kind.
const BPF_CORE_FIELD_BYTE_OFFSET: u32 = 0;

/// Size of a BPF instruction.
const BPF_INSN_SIZE: u32 = 8;

/// A CO-RE relocation record ready to be serialized.
#[derive(Debug, Clone)]
struct CoreReloRecord {
    /// Byte offset of the instruction within its ELF section.
    insn_off: u32,
    /// BTF type ID of the struct being accessed.
    type_id: u32,
    /// Offset into the BTF string table for the access string.
    access_str_off: u32,
    /// Relocation kind (BPF_CORE_FIELD_BYTE_OFFSET = 0).
    kind: u32,
}

/// Groups CO-RE relocation records by ELF section.
#[derive(Debug)]
struct SectionRelos {
    /// Offset of the section name in the BTF string table.
    sec_name_off: u32,
    /// The relocation records for this section.
    records: Vec<CoreReloRecord>,
}

/// Builder for `.BTF.ext` section data.
pub struct BtfExtWriter<'a> {
    btf: &'a BtfInfo,
    /// Existing .BTF.ext raw data (if any).
    #[allow(dead_code)]
    existing_ext_data: Option<Vec<u8>>,
    /// New CO-RE relocation records grouped by section name.
    new_relos: Vec<(String, Vec<CoreReloRecord>)>,
    /// Strings added to BTF for access strings and section names.
    added_strings: Vec<(String, u32)>,
}

impl<'a> BtfExtWriter<'a> {
    /// Creates a new writer referencing the parsed BTF info.
    pub fn new(btf: &'a BtfInfo) -> Self {
        Self {
            btf,
            existing_ext_data: None,
            new_relos: Vec::new(),
            added_strings: Vec::new(),
        }
    }

    /// Adds a CO-RE relocation from a sidecar entry.
    ///
    /// This resolves the struct name and field path against the BTF
    /// to compute the type_id and access string, then creates a
    /// `bpf_core_relo` record.
    pub fn add_relocation(&mut self, entry: &RelocationEntry) -> Result<()> {
        // Find the struct type in BTF.
        let type_id = self
            .btf
            .find_struct_by_name(&entry.struct_name)
            .with_context(|| format!("looking up struct '{}'", entry.struct_name))?;

        // Compute the access string (e.g. "0:3:7").
        let access_str = self
            .btf
            .compute_access_string(type_id, &entry.field_path)
            .with_context(|| {
                format!(
                    "computing access string for {}.{}",
                    entry.struct_name, entry.field_path
                )
            })?;

        eprintln!(
            "  relo: section={} insn={} type={}(id={}) path={} -> access_str={}",
            entry.section, entry.insn_index, entry.struct_name, type_id, entry.field_path,
            access_str
        );

        let record = CoreReloRecord {
            insn_off: entry.insn_index * BPF_INSN_SIZE,
            type_id,
            // access_str_off will be filled in during finish()
            access_str_off: 0,
            kind: BPF_CORE_FIELD_BYTE_OFFSET,
        };

        // Group by section name.
        if let Some(group) = self.new_relos.iter_mut().find(|(s, _)| s == &entry.section) {
            group.1.push(record);
        } else {
            self.new_relos
                .push((entry.section.clone(), vec![record]));
        }

        // Remember the access string for later string table addition.
        self.added_strings
            .push((access_str, 0 /* placeholder */));

        Ok(())
    }

    /// Adds a CO-RE relocation with a pre-computed type_id and access string.
    ///
    /// This is used by the auto-discovery path where the type_id comes from
    /// a struct that was imported into the program's BTF from vmlinux, and
    /// the access string was computed from the vmlinux BTF.
    pub fn add_relocation_with_type_id(
        &mut self,
        entry: &RelocationEntry,
        type_id: u32,
        access_str: &str,
    ) -> Result<()> {
        eprintln!(
            "  relo: section={} insn={} type={}(id={}) path={} -> access_str={}",
            entry.section, entry.insn_index, entry.struct_name, type_id, entry.field_path,
            access_str
        );

        let record = CoreReloRecord {
            insn_off: entry.insn_index * BPF_INSN_SIZE,
            type_id,
            // access_str_off will be filled in during finish()
            access_str_off: 0,
            kind: BPF_CORE_FIELD_BYTE_OFFSET,
        };

        // Group by section name.
        if let Some(group) = self.new_relos.iter_mut().find(|(s, _)| s == &entry.section) {
            group.1.push(record);
        } else {
            self.new_relos
                .push((entry.section.clone(), vec![record]));
        }

        // Remember the access string for later string table addition.
        self.added_strings
            .push((access_str.to_string(), 0 /* placeholder */));

        Ok(())
    }

    /// Finalizes the BTF.ext data, producing the new .BTF and .BTF.ext
    /// section contents.
    ///
    /// Returns (new_btf_data, new_btf_ext_data).
    pub fn finish(mut self, elf_data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        // Load existing .BTF.ext data if present.
        let existing_ext = Self::load_existing_ext(elf_data);

        // Clone the BTF info so we can add strings to it.
        let mut btf = self.btf.clone();

        // Add all access strings to the BTF string table and record their offsets.
        let mut access_str_offsets = Vec::new();
        for (access_str, _) in &self.added_strings {
            let off = btf.add_string(access_str);
            access_str_offsets.push(off);
        }

        // Add section names to the BTF string table.
        let mut section_name_offsets = Vec::new();
        for (sec_name, _) in &self.new_relos {
            let off = btf.add_string(sec_name);
            section_name_offsets.push(off);
        }

        // Now fill in the access_str_off in the relocation records.
        let mut str_idx = 0;
        for (_, records) in &mut self.new_relos {
            for record in records.iter_mut() {
                record.access_str_off = access_str_offsets[str_idx];
                str_idx += 1;
            }
        }

        // Build SectionRelos from the grouped records.
        let section_relos: Vec<SectionRelos> = self
            .new_relos
            .iter()
            .enumerate()
            .map(|(i, (_, records))| SectionRelos {
                sec_name_off: section_name_offsets[i],
                records: records.clone(),
            })
            .collect();

        // Build the .BTF.ext binary data.
        let btf_ext_data = self.build_btf_ext(&existing_ext, &section_relos)?;

        // Build the updated .BTF binary data (with new strings).
        let btf_data = btf.to_bytes(elf_data)?;

        Ok((btf_data, btf_ext_data))
    }

    /// Loads existing .BTF.ext data from the ELF, if present.
    fn load_existing_ext(elf_data: &[u8]) -> Option<Vec<u8>> {
        let obj =
            ElfFile::<elf::FileHeader64<Endianness>>::parse(elf_data).ok()?;
        let section = obj.section_by_name(".BTF.ext")?;
        section.data().ok().map(|d| d.to_vec())
    }

    /// Builds the complete .BTF.ext section binary data.
    ///
    /// If there's existing .BTF.ext data, we preserve the func_info and
    /// line_info sections and append our core_relo records.
    fn build_btf_ext(
        &self,
        existing: &Option<Vec<u8>>,
        new_relos: &[SectionRelos],
    ) -> Result<Vec<u8>> {
        let (func_info_data, line_info_data, existing_core_relo_data) =
            if let Some(ext_data) = existing {
                self.parse_existing_ext(ext_data)?
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        // Build the core_relo section data.
        let core_relo_data = self.build_core_relo_section(
            &existing_core_relo_data,
            new_relos,
        );

        // Calculate offsets.
        // Layout after header:
        //   func_info_data
        //   line_info_data
        //   core_relo_data

        let func_info_off = 0u32;
        let func_info_len = func_info_data.len() as u32;
        let line_info_off = func_info_off + func_info_len;
        let line_info_len = line_info_data.len() as u32;
        let core_relo_off = line_info_off + line_info_len;
        let core_relo_len = core_relo_data.len() as u32;

        // Build the header.
        let mut buf = Vec::new();
        buf.extend_from_slice(&BTF_MAGIC.to_le_bytes());
        buf.push(1); // version
        buf.push(0); // flags
        buf.extend_from_slice(&BTF_EXT_HEADER_SIZE.to_le_bytes()); // hdr_len
        buf.extend_from_slice(&func_info_off.to_le_bytes());
        buf.extend_from_slice(&func_info_len.to_le_bytes());
        buf.extend_from_slice(&line_info_off.to_le_bytes());
        buf.extend_from_slice(&line_info_len.to_le_bytes());
        buf.extend_from_slice(&core_relo_off.to_le_bytes());
        buf.extend_from_slice(&core_relo_len.to_le_bytes());

        // Append the section data.
        buf.extend_from_slice(&func_info_data);
        buf.extend_from_slice(&line_info_data);
        buf.extend_from_slice(&core_relo_data);

        Ok(buf)
    }

    /// Parses existing .BTF.ext data to extract func_info, line_info, and
    /// core_relo raw section blobs.
    fn parse_existing_ext(
        &self,
        data: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        if data.len() < 8 {
            bail!("existing .BTF.ext data too short");
        }

        let magic = u16::from_le_bytes([data[0], data[1]]);
        if magic != BTF_MAGIC {
            bail!("invalid .BTF.ext magic: {:#06x}", magic);
        }

        let hdr_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if data.len() < hdr_len {
            bail!(".BTF.ext header extends beyond data");
        }

        // Parse the header fields (may be shorter than our full header struct).
        let read_u32 = |off: usize| -> u32 {
            if off + 4 <= hdr_len && off + 4 <= data.len() {
                u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
            } else {
                0
            }
        };

        let func_info_off = read_u32(8) as usize;
        let func_info_len = read_u32(12) as usize;
        let line_info_off = read_u32(16) as usize;
        let line_info_len = read_u32(20) as usize;
        let core_relo_off = read_u32(24) as usize;
        let core_relo_len = read_u32(28) as usize;

        let extract = |off: usize, len: usize| -> Vec<u8> {
            let start = hdr_len + off;
            let end = start + len;
            if end <= data.len() {
                data[start..end].to_vec()
            } else {
                Vec::new()
            }
        };

        Ok((
            extract(func_info_off, func_info_len),
            extract(line_info_off, line_info_len),
            extract(core_relo_off, core_relo_len),
        ))
    }

    /// Builds the core_relo section data, combining existing records with
    /// new ones.
    fn build_core_relo_section(
        &self,
        existing: &[u8],
        new_relos: &[SectionRelos],
    ) -> Vec<u8> {
        // If there are no relocations at all, return empty.
        if existing.is_empty() && new_relos.is_empty() {
            return Vec::new();
        }

        let mut buf = Vec::new();

        // Write rec_size (4 bytes) -- always sizeof(bpf_core_relo) = 16.
        buf.extend_from_slice(&CORE_RELO_SIZE.to_le_bytes());

        // Do NOT copy existing core_relo records. When the postprocessor
        // replaces the .BTF section with new type data, the type_ids in any
        // pre-existing core_relo records become stale — they reference types
        // from the OLD BTF layout that no longer exist. Only the
        // postprocessor's new relocations have valid type_ids pointing to
        // the newly-created stub structs.
        //
        // This was the root cause of the "field relocation on a type that
        // doesn't have fields" error: the Rust compiler generated CO-RE
        // relocations with type_ids from the original BTF, but after the
        // postprocessor replaced the BTF, those type_ids pointed to wrong
        // types (e.g., an Int instead of a Struct).
        let _ = existing;

        // Append new relocation record groups.
        for section in new_relos {
            // sec_name_off (4 bytes)
            buf.extend_from_slice(&section.sec_name_off.to_le_bytes());
            // num_info (4 bytes)
            let num_info = section.records.len() as u32;
            buf.extend_from_slice(&num_info.to_le_bytes());

            // Records
            for record in &section.records {
                buf.extend_from_slice(&record.insn_off.to_le_bytes());
                buf.extend_from_slice(&record.type_id.to_le_bytes());
                buf.extend_from_slice(&record.access_str_off.to_le_bytes());
                buf.extend_from_slice(&record.kind.to_le_bytes());
            }
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_relo_record_size() {
        // Verify our record size matches the kernel's bpf_core_relo.
        assert_eq!(CORE_RELO_SIZE, 16);
    }

    #[test]
    fn test_build_core_relo_section_empty_existing() {
        let writer = BtfExtWriter {
            btf: &BtfInfo {
                header: crate::btf_parser::BtfHeader {
                    magic: 0xEB9F,
                    version: 1,
                    flags: 0,
                    hdr_len: 24,
                    type_off: 0,
                    type_len: 0,
                    str_off: 0,
                    str_len: 1,
                },
                types: vec![],
                strings: vec![0],
                raw_header_len: 24,
                appended_type_data: Vec::new(),
            },
            existing_ext_data: None,
            new_relos: Vec::new(),
            added_strings: Vec::new(),
        };

        let relos = vec![SectionRelos {
            sec_name_off: 42,
            records: vec![CoreReloRecord {
                insn_off: 40,
                type_id: 7,
                access_str_off: 100,
                kind: 0,
            }],
        }];

        let data = writer.build_core_relo_section(&[], &relos);

        // Should be: rec_size(4) + sec_name_off(4) + num_info(4) + 1 record(16) = 28
        assert_eq!(data.len(), 28);

        // Check rec_size
        assert_eq!(
            u32::from_le_bytes(data[0..4].try_into().unwrap()),
            16
        );
        // Check sec_name_off
        assert_eq!(
            u32::from_le_bytes(data[4..8].try_into().unwrap()),
            42
        );
        // Check num_info
        assert_eq!(
            u32::from_le_bytes(data[8..12].try_into().unwrap()),
            1
        );
        // Check insn_off
        assert_eq!(
            u32::from_le_bytes(data[12..16].try_into().unwrap()),
            40
        );
        // Check type_id
        assert_eq!(
            u32::from_le_bytes(data[16..20].try_into().unwrap()),
            7
        );
        // Check access_str_off
        assert_eq!(
            u32::from_le_bytes(data[20..24].try_into().unwrap()),
            100
        );
        // Check kind
        assert_eq!(
            u32::from_le_bytes(data[24..28].try_into().unwrap()),
            0
        );
    }
}
