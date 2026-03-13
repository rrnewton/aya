//! Parser for `.aya.core_relo` ELF section markers.
//!
//! The `core_read!` proc macro emits marker records into the
//! `.aya.core_relo` section of the BPF ELF object.  Each record
//! encodes a struct name and field path.
//!
//! This module parses that section and produces a list of
//! `(struct_name, field_path)` pairs that the post-processor uses
//! to generate CO-RE relocation records.
//!
//! ## Marker format
//!
//! Each marker record is a variable-length byte sequence:
//!   - 1 byte:  tag (0xAC)
//!   - 1 byte:  struct name length (N)
//!   - N bytes: struct name (UTF-8, no NUL)
//!   - 1 byte:  field path length (M)
//!   - M bytes: field path (UTF-8, dot-separated, no NUL)
//!
//! Multiple records are concatenated in the section.  Because each
//! `core_read!` invocation emits a separate `#[link_section]` static,
//! the linker concatenates them into a single `.aya.core_relo` section.

use anyhow::{Context, Result, bail};
use object::read::elf::ElfFile;
use object::{Endianness, Object, ObjectSection, elf};

/// Tag byte that marks the start of each record.
const MARKER_TAG: u8 = 0xAC;

/// A parsed marker from the `.aya.core_relo` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreReloMarker {
    /// Name of the struct type (e.g., "task_struct").
    pub struct_name: String,
    /// Dot-separated field path (e.g., "scx.dsq_vtime").
    pub field_path: String,
}

/// Parses all marker records from an `.aya.core_relo` ELF section.
///
/// Returns `Ok(vec![])` if the section does not exist.
pub fn parse_markers_from_elf(elf_data: &[u8]) -> Result<Vec<CoreReloMarker>> {
    let obj = ElfFile::<elf::FileHeader64<Endianness>>::parse(elf_data)
        .context("parsing ELF for .aya.core_relo")?;

    let section = match obj.section_by_name(".aya.core_relo") {
        Some(s) => s,
        None => return Ok(Vec::new()),
    };

    let data = section.data().context("reading .aya.core_relo data")?;
    parse_markers(data)
}

/// Parses marker records from raw section data.
///
/// This is separate from `parse_markers_from_elf` for testability.
pub fn parse_markers(data: &[u8]) -> Result<Vec<CoreReloMarker>> {
    let mut markers = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        // Skip padding/zero bytes that the linker may insert between
        // the individual static arrays.
        if data[offset] == 0 {
            offset += 1;
            continue;
        }

        // Look for the tag byte.
        if data[offset] != MARKER_TAG {
            // Skip unknown bytes -- the linker may insert alignment
            // padding or other metadata.
            offset += 1;
            continue;
        }
        offset += 1; // consume tag

        // Read struct name length.
        if offset >= data.len() {
            bail!("truncated marker: missing struct name length at offset {}", offset);
        }
        let name_len = data[offset] as usize;
        offset += 1;

        // Read struct name.
        if offset + name_len > data.len() {
            bail!(
                "truncated marker: struct name extends beyond data (need {} bytes at offset {})",
                name_len,
                offset
            );
        }
        let struct_name = std::str::from_utf8(&data[offset..offset + name_len])
            .context("invalid UTF-8 in struct name")?
            .to_string();
        offset += name_len;

        // Read field path length.
        if offset >= data.len() {
            bail!("truncated marker: missing field path length at offset {}", offset);
        }
        let path_len = data[offset] as usize;
        offset += 1;

        // Read field path.
        if offset + path_len > data.len() {
            bail!(
                "truncated marker: field path extends beyond data (need {} bytes at offset {})",
                path_len,
                offset
            );
        }
        let field_path = std::str::from_utf8(&data[offset..offset + path_len])
            .context("invalid UTF-8 in field path")?
            .to_string();
        offset += path_len;

        markers.push(CoreReloMarker {
            struct_name,
            field_path,
        });
    }

    Ok(markers)
}

/// Deduplicates markers, returning unique (struct_name, field_path) pairs.
///
/// If the same `core_read!` call appears in multiple functions (e.g.,
/// via inlining or generic instantiation), the linker may produce
/// duplicate markers.  We only need one CO-RE relocation record per
/// unique (struct_name, field_path) combination per program section.
pub fn deduplicate_markers(markers: Vec<CoreReloMarker>) -> Vec<CoreReloMarker> {
    let mut seen = std::collections::HashSet::new();
    markers
        .into_iter()
        .filter(|m| seen.insert((m.struct_name.clone(), m.field_path.clone())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_marker() {
        // Build a marker: tag=0xAC, name_len=11, "task_struct", path_len=3, "pid"
        let mut data = vec![MARKER_TAG];
        let name = b"task_struct";
        data.push(name.len() as u8);
        data.extend_from_slice(name);
        let path = b"pid";
        data.push(path.len() as u8);
        data.extend_from_slice(path);

        let markers = parse_markers(&data).unwrap();
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].struct_name, "task_struct");
        assert_eq!(markers[0].field_path, "pid");
    }

    #[test]
    fn test_parse_multiple_markers() {
        let mut data = Vec::new();

        // Marker 1: task_struct.pid
        data.push(MARKER_TAG);
        let name = b"task_struct";
        data.push(name.len() as u8);
        data.extend_from_slice(name);
        let path = b"pid";
        data.push(path.len() as u8);
        data.extend_from_slice(path);

        // Marker 2: task_struct.scx.dsq_vtime
        data.push(MARKER_TAG);
        let name = b"task_struct";
        data.push(name.len() as u8);
        data.extend_from_slice(name);
        let path = b"scx.dsq_vtime";
        data.push(path.len() as u8);
        data.extend_from_slice(path);

        let markers = parse_markers(&data).unwrap();
        assert_eq!(markers.len(), 2);
        assert_eq!(markers[0].field_path, "pid");
        assert_eq!(markers[1].field_path, "scx.dsq_vtime");
    }

    #[test]
    fn test_parse_with_zero_padding() {
        let mut data = Vec::new();

        // Some zero padding before the marker.
        data.extend_from_slice(&[0, 0, 0, 0]);

        // Marker
        data.push(MARKER_TAG);
        data.push(3); // name_len
        data.extend_from_slice(b"foo");
        data.push(3); // path_len
        data.extend_from_slice(b"bar");

        // Trailing zeros
        data.extend_from_slice(&[0, 0]);

        let markers = parse_markers(&data).unwrap();
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].struct_name, "foo");
        assert_eq!(markers[0].field_path, "bar");
    }

    #[test]
    fn test_parse_empty() {
        let markers = parse_markers(&[]).unwrap();
        assert!(markers.is_empty());
    }

    #[test]
    fn test_deduplicate() {
        let markers = vec![
            CoreReloMarker {
                struct_name: "task_struct".into(),
                field_path: "pid".into(),
            },
            CoreReloMarker {
                struct_name: "task_struct".into(),
                field_path: "pid".into(),
            },
            CoreReloMarker {
                struct_name: "task_struct".into(),
                field_path: "tgid".into(),
            },
        ];
        let deduped = deduplicate_markers(markers);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].field_path, "pid");
        assert_eq!(deduped[1].field_path, "tgid");
    }
}
