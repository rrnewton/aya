//! Minimal BTF parser for the post-processor.
//!
//! Parses the `.BTF` section from a BPF ELF object to extract type
//! information and the string table.  This is sufficient to:
//!   - Look up struct types by name
//!   - Walk struct members to compute access strings
//!   - Add new strings to the string table

use anyhow::{Context, Result, bail};
use object::read::elf::ElfFile;
use object::{Endianness, Object, ObjectSection, elf};

/// Parsed BTF information from a `.BTF` section.
#[derive(Debug, Clone)]
pub struct BtfInfo {
    pub header: BtfHeader,
    pub types: Vec<BtfType>,
    pub strings: Vec<u8>,
    /// The raw header bytes (for reconstruction)
    pub raw_header_len: usize,
    /// Additional raw type data appended after the original types.
    /// These bytes are concatenated with the original type data during
    /// `to_bytes()`.
    pub appended_type_data: Vec<u8>,
}

/// BTF file header.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BtfHeader {
    pub magic: u16,
    pub version: u8,
    pub flags: u8,
    pub hdr_len: u32,
    pub type_off: u32,
    pub type_len: u32,
    pub str_off: u32,
    pub str_len: u32,
}

/// A parsed BTF type entry.
#[derive(Debug, Clone)]
pub enum BtfType {
    Void,
    Int(BtfTypeCommon),
    Ptr(BtfTypeCommon),
    Array(BtfTypeCommon, BtfArray),
    Struct(BtfTypeCommon, Vec<BtfMember>),
    Union(BtfTypeCommon, Vec<BtfMember>),
    Enum(BtfTypeCommon),
    Fwd(BtfTypeCommon),
    Typedef(BtfTypeCommon),
    Volatile(BtfTypeCommon),
    Const(BtfTypeCommon),
    Restrict(BtfTypeCommon),
    Func(BtfTypeCommon),
    FuncProto(BtfTypeCommon, Vec<BtfParam>),
    Var(BtfTypeCommon),
    DataSec(BtfTypeCommon),
    Float(BtfTypeCommon),
    DeclTag(BtfTypeCommon),
    TypeTag(BtfTypeCommon),
    Enum64(BtfTypeCommon),
    /// Unknown or unsupported kind
    Unknown(u32),
}

/// Common fields in a BTF type header (first 12 bytes).
#[derive(Debug, Clone, Copy)]
pub struct BtfTypeCommon {
    pub name_off: u32,
    pub info: u32,
    pub size_or_type: u32,
}

impl BtfTypeCommon {
    pub fn kind(&self) -> u32 {
        (self.info >> 24) & 0x1f
    }

    pub fn vlen(&self) -> u32 {
        self.info & 0xffff
    }

    pub fn kind_flag(&self) -> bool {
        (self.info >> 31) != 0
    }
}

/// BTF array descriptor.
#[derive(Debug, Clone, Copy)]
pub struct BtfArray {
    pub element_type: u32,
    pub index_type: u32,
    pub nelems: u32,
}

/// BTF struct/union member.
#[derive(Debug, Clone, Copy)]
pub struct BtfMember {
    pub name_off: u32,
    pub type_id: u32,
    pub offset: u32,
}

/// BTF function parameter.
#[derive(Debug, Clone, Copy)]
pub struct BtfParam {
    pub name_off: u32,
    pub type_id: u32,
}

// BTF type kind constants
const BTF_KIND_INT: u32 = 1;
const BTF_KIND_PTR: u32 = 2;
const BTF_KIND_ARRAY: u32 = 3;
const BTF_KIND_STRUCT: u32 = 4;
const BTF_KIND_UNION: u32 = 5;
const BTF_KIND_ENUM: u32 = 6;
const BTF_KIND_FWD: u32 = 7;
const BTF_KIND_TYPEDEF: u32 = 8;
const BTF_KIND_VOLATILE: u32 = 9;
const BTF_KIND_CONST: u32 = 10;
const BTF_KIND_RESTRICT: u32 = 11;
const BTF_KIND_FUNC: u32 = 12;
const BTF_KIND_FUNC_PROTO: u32 = 13;
const BTF_KIND_VAR: u32 = 14;
const BTF_KIND_DATASEC: u32 = 15;
const BTF_KIND_FLOAT: u32 = 16;
const BTF_KIND_DECL_TAG: u32 = 17;
const BTF_KIND_TYPE_TAG: u32 = 18;
const BTF_KIND_ENUM64: u32 = 19;

impl BtfInfo {
    /// Parses BTF from a BPF ELF file.
    pub fn parse_from_elf(elf_data: &[u8]) -> Result<Self> {
        let obj = ElfFile::<elf::FileHeader64<Endianness>>::parse(elf_data)
            .context("parsing ELF")?;

        let btf_section = obj.section_by_name(".BTF").context("no .BTF section")?;
        let btf_data = btf_section.data().context("reading .BTF data")?;

        Self::parse(btf_data)
    }

    /// Parses BTF from raw section data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            bail!("BTF data too short for header ({} bytes)", data.len());
        }

        let header = BtfHeader {
            magic: u16::from_le_bytes([data[0], data[1]]),
            version: data[2],
            flags: data[3],
            hdr_len: u32::from_le_bytes(data[4..8].try_into().unwrap()),
            type_off: u32::from_le_bytes(data[8..12].try_into().unwrap()),
            type_len: u32::from_le_bytes(data[12..16].try_into().unwrap()),
            str_off: u32::from_le_bytes(data[16..20].try_into().unwrap()),
            str_len: u32::from_le_bytes(data[20..24].try_into().unwrap()),
        };

        if header.magic != 0xEB9F {
            bail!(
                "invalid BTF magic: {:#06x} (expected 0xEB9F)",
                header.magic
            );
        }

        let hdr_len = header.hdr_len as usize;
        let type_start = hdr_len + header.type_off as usize;
        let type_end = type_start + header.type_len as usize;
        let str_start = hdr_len + header.str_off as usize;
        let str_end = str_start + header.str_len as usize;

        if type_end > data.len() || str_end > data.len() {
            bail!("BTF section data is truncated");
        }

        let type_data = &data[type_start..type_end];
        let strings = data[str_start..str_end].to_vec();

        let types = Self::parse_types(type_data)?;

        Ok(Self {
            header,
            types,
            strings,
            raw_header_len: hdr_len,
            appended_type_data: Vec::new(),
        })
    }

    fn parse_types(data: &[u8]) -> Result<Vec<BtfType>> {
        let mut types = vec![BtfType::Void]; // type_id 0 = void
        let mut offset = 0;

        while offset + 12 <= data.len() {
            let common = BtfTypeCommon {
                name_off: u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()),
                info: u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()),
                size_or_type: u32::from_le_bytes(
                    data[offset + 8..offset + 12].try_into().unwrap(),
                ),
            };
            offset += 12;

            let kind = common.kind();
            let vlen = common.vlen() as usize;

            let ty = match kind {
                BTF_KIND_INT => {
                    // INT has 4 extra bytes
                    offset += 4;
                    BtfType::Int(common)
                }
                BTF_KIND_PTR => BtfType::Ptr(common),
                BTF_KIND_ARRAY => {
                    if offset + 12 > data.len() {
                        bail!("truncated BTF array at offset {offset}");
                    }
                    let arr = BtfArray {
                        element_type: u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap(),
                        ),
                        index_type: u32::from_le_bytes(
                            data[offset + 4..offset + 8].try_into().unwrap(),
                        ),
                        nelems: u32::from_le_bytes(
                            data[offset + 8..offset + 12].try_into().unwrap(),
                        ),
                    };
                    offset += 12;
                    BtfType::Array(common, arr)
                }
                BTF_KIND_STRUCT | BTF_KIND_UNION => {
                    let member_size = 12; // name_off(4) + type(4) + offset(4)
                    let mut members = Vec::with_capacity(vlen);
                    for _ in 0..vlen {
                        if offset + member_size > data.len() {
                            bail!("truncated BTF struct/union members at offset {offset}");
                        }
                        members.push(BtfMember {
                            name_off: u32::from_le_bytes(
                                data[offset..offset + 4].try_into().unwrap(),
                            ),
                            type_id: u32::from_le_bytes(
                                data[offset + 4..offset + 8].try_into().unwrap(),
                            ),
                            offset: u32::from_le_bytes(
                                data[offset + 8..offset + 12].try_into().unwrap(),
                            ),
                        });
                        offset += member_size;
                    }
                    if kind == BTF_KIND_STRUCT {
                        BtfType::Struct(common, members)
                    } else {
                        BtfType::Union(common, members)
                    }
                }
                BTF_KIND_ENUM => {
                    // Each enum variant: name_off(4) + val(4) = 8 bytes
                    offset += vlen * 8;
                    BtfType::Enum(common)
                }
                BTF_KIND_FWD => BtfType::Fwd(common),
                BTF_KIND_TYPEDEF => BtfType::Typedef(common),
                BTF_KIND_VOLATILE => BtfType::Volatile(common),
                BTF_KIND_CONST => BtfType::Const(common),
                BTF_KIND_RESTRICT => BtfType::Restrict(common),
                BTF_KIND_FUNC => BtfType::Func(common),
                BTF_KIND_FUNC_PROTO => {
                    // Each param: name_off(4) + type(4) = 8 bytes
                    let mut params = Vec::with_capacity(vlen);
                    for _ in 0..vlen {
                        if offset + 8 > data.len() {
                            bail!("truncated BTF func_proto params at offset {offset}");
                        }
                        params.push(BtfParam {
                            name_off: u32::from_le_bytes(
                                data[offset..offset + 4].try_into().unwrap(),
                            ),
                            type_id: u32::from_le_bytes(
                                data[offset + 4..offset + 8].try_into().unwrap(),
                            ),
                        });
                        offset += 8;
                    }
                    BtfType::FuncProto(common, params)
                }
                BTF_KIND_VAR => {
                    // VAR has 4 extra bytes (linkage)
                    offset += 4;
                    BtfType::Var(common)
                }
                BTF_KIND_DATASEC => {
                    // Each entry: type(4) + offset(4) + size(4) = 12 bytes
                    offset += vlen * 12;
                    BtfType::DataSec(common)
                }
                BTF_KIND_FLOAT => BtfType::Float(common),
                BTF_KIND_DECL_TAG => {
                    // DECL_TAG has 4 extra bytes (component_idx)
                    offset += 4;
                    BtfType::DeclTag(common)
                }
                BTF_KIND_TYPE_TAG => BtfType::TypeTag(common),
                BTF_KIND_ENUM64 => {
                    // Each variant: name_off(4) + val_lo32(4) + val_hi32(4) = 12 bytes
                    offset += vlen * 12;
                    BtfType::Enum64(common)
                }
                _ => {
                    // Unknown kind -- we can't know its size, so stop here.
                    eprintln!("warning: unknown BTF kind {kind} at type #{}, stopping type parse", types.len());
                    break;
                }
            };

            types.push(ty);
        }

        Ok(types)
    }

    /// Looks up a string by offset in the BTF string table.
    pub fn string_at(&self, offset: u32) -> Result<&str> {
        let start = offset as usize;
        if start >= self.strings.len() {
            bail!("BTF string offset {offset} out of bounds");
        }
        let end = self.strings[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|pos| start + pos)
            .unwrap_or(self.strings.len());
        std::str::from_utf8(&self.strings[start..end])
            .with_context(|| format!("invalid UTF-8 in BTF string at offset {offset}"))
    }

    /// Finds a struct type by name, returning its type_id.
    pub fn find_struct_by_name(&self, name: &str) -> Result<u32> {
        for (type_id, ty) in self.types.iter().enumerate() {
            match ty {
                BtfType::Struct(common, _) | BtfType::Union(common, _) => {
                    if let Ok(type_name) = self.string_at(common.name_off) {
                        if type_name == name {
                            return Ok(type_id as u32);
                        }
                    }
                }
                _ => {}
            }
        }
        bail!("struct '{}' not found in BTF", name);
    }

    /// Resolves a type through typedefs, const, volatile, restrict modifiers.
    pub fn resolve_type(&self, mut type_id: u32) -> Result<u32> {
        for _ in 0..32 {
            let ty = self
                .types
                .get(type_id as usize)
                .context("type_id out of range")?;
            match ty {
                BtfType::Typedef(c) | BtfType::Volatile(c) | BtfType::Const(c)
                | BtfType::Restrict(c) | BtfType::TypeTag(c) => {
                    type_id = c.size_or_type;
                }
                _ => return Ok(type_id),
            }
        }
        bail!("max resolve depth reached for type_id {type_id}");
    }

    /// Given a struct type_id and a dot-separated field path like "scx.dsq_vtime",
    /// returns the colon-separated access string like "0:35:7".
    ///
    /// The first element is always "0" (index into the struct array).
    pub fn compute_access_string(&self, struct_type_id: u32, field_path: &str) -> Result<String> {
        let mut parts = vec![0usize]; // start with 0 for the base struct
        let mut current_type_id = struct_type_id;

        for field_name in field_path.split('.') {
            let resolved_id = self.resolve_type(current_type_id)?;
            let ty = self
                .types
                .get(resolved_id as usize)
                .context("type_id out of range during access string computation")?;

            let members = match ty {
                BtfType::Struct(_, members) | BtfType::Union(_, members) => members,
                _ => bail!(
                    "expected struct/union at type_id {resolved_id}, got {:?}",
                    std::mem::discriminant(ty)
                ),
            };

            let mut found = false;
            for (index, member) in members.iter().enumerate() {
                if let Ok(member_name) = self.string_at(member.name_off) {
                    if member_name == field_name {
                        parts.push(index);
                        current_type_id = member.type_id;
                        found = true;
                        break;
                    }
                }

                // Handle anonymous struct/union members (name_off == 0).
                // We need to recurse into them to find the field.
                if member.name_off == 0 {
                    if let Ok(inner_id) = self.resolve_type(member.type_id) {
                        if let Some(inner_ty) = self.types.get(inner_id as usize) {
                            if let BtfType::Struct(_, inner_members)
                            | BtfType::Union(_, inner_members) = inner_ty
                            {
                                for (inner_idx, inner_member) in inner_members.iter().enumerate() {
                                    if let Ok(inner_name) = self.string_at(inner_member.name_off) {
                                        if inner_name == field_name {
                                            parts.push(index);
                                            parts.push(inner_idx);
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
            }

            if !found {
                bail!(
                    "field '{}' not found in struct type_id {} (resolved {})",
                    field_name,
                    current_type_id,
                    resolved_id
                );
            }
        }

        Ok(parts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(":"))
    }

    /// Adds a string to the BTF string table and returns its offset.
    pub fn add_string(&mut self, s: &str) -> u32 {
        let offset = self.strings.len() as u32;
        self.strings.extend_from_slice(s.as_bytes());
        self.strings.push(0); // NUL terminator
        offset
    }

    /// Returns the next type_id that would be assigned to a newly added type.
    pub fn next_type_id(&self) -> u32 {
        self.types.len() as u32
    }

    /// Adds a struct type with the given name and members to the BTF.
    ///
    /// Each member is `(name, type_id, bit_offset)`.
    ///
    /// Returns the type_id of the newly added struct.
    pub fn add_struct(
        &mut self,
        name: &str,
        size: u32,
        members: &[(&str, u32, u32)],
    ) -> u32 {
        let type_id = self.next_type_id();

        let name_off = self.add_string(name);
        let vlen = members.len() as u32;
        // kind=4 (STRUCT), vlen=member count
        let info = (BTF_KIND_STRUCT << 24) | vlen;

        // Serialize the type header: name_off(4) + info(4) + size(4)
        self.appended_type_data.extend_from_slice(&name_off.to_le_bytes());
        self.appended_type_data.extend_from_slice(&info.to_le_bytes());
        self.appended_type_data.extend_from_slice(&size.to_le_bytes());

        // Serialize members: name_off(4) + type_id(4) + offset(4) each
        let mut parsed_members = Vec::with_capacity(members.len());
        for &(member_name, member_type_id, bit_offset) in members {
            let m_name_off = self.add_string(member_name);
            self.appended_type_data.extend_from_slice(&m_name_off.to_le_bytes());
            self.appended_type_data.extend_from_slice(&member_type_id.to_le_bytes());
            self.appended_type_data.extend_from_slice(&bit_offset.to_le_bytes());

            parsed_members.push(BtfMember {
                name_off: m_name_off,
                type_id: member_type_id,
                offset: bit_offset,
            });
        }

        // Also add to the in-memory type list for subsequent lookups.
        let common = BtfTypeCommon {
            name_off,
            info,
            size_or_type: size,
        };
        self.types.push(BtfType::Struct(common, parsed_members));

        type_id
    }

    /// Adds a BTF_KIND_INT type (used as a placeholder for leaf member types).
    ///
    /// Returns the type_id of the newly added int type.
    pub fn add_int(&mut self, name: &str, size: u32) -> u32 {
        let type_id = self.next_type_id();

        let name_off = self.add_string(name);
        // kind=1 (INT), vlen=0
        let info = BTF_KIND_INT << 24;

        // Type header: name_off(4) + info(4) + size(4)
        self.appended_type_data.extend_from_slice(&name_off.to_le_bytes());
        self.appended_type_data.extend_from_slice(&info.to_le_bytes());
        self.appended_type_data.extend_from_slice(&size.to_le_bytes());

        // INT has 4 extra bytes encoding bits and offset:
        //   (encoding << 24) | (offset << 16) | bits
        let int_data: u32 = size * 8; // bits = size*8, offset=0, encoding=0
        self.appended_type_data.extend_from_slice(&int_data.to_le_bytes());

        // Add to in-memory type list
        let common = BtfTypeCommon {
            name_off,
            info,
            size_or_type: size,
        };
        self.types.push(BtfType::Int(common));

        type_id
    }

    /// Given a struct type_id and a dot-separated field path like
    /// "scx.dsq_vtime", returns the byte offset of the field within the
    /// struct.
    ///
    /// This is used by the auto-discovery mode to match `offset_of!`
    /// constants in the compiled BPF instructions.
    ///
    /// Note: BTF member offsets are stored in bits for structs that use
    /// bitfields (kind_flag=1) and in bits for regular structs too.
    /// For regular structs, the bit offset is always a multiple of 8.
    pub fn compute_byte_offset(&self, struct_type_id: u32, field_path: &str) -> Result<u64> {
        let mut total_bit_offset: u64 = 0;
        let mut current_type_id = struct_type_id;

        for field_name in field_path.split('.') {
            let resolved_id = self.resolve_type(current_type_id)?;
            let ty = self
                .types
                .get(resolved_id as usize)
                .context("type_id out of range during byte offset computation")?;

            let (common, members) = match ty {
                BtfType::Struct(c, members) | BtfType::Union(c, members) => (c, members),
                _ => bail!(
                    "expected struct/union at type_id {resolved_id}, got {:?}",
                    std::mem::discriminant(ty)
                ),
            };

            let mut found = false;
            let kind_flag = common.kind_flag();

            for member in members {
                if let Ok(member_name) = self.string_at(member.name_off) {
                    if member_name == field_name {
                        if kind_flag {
                            // Bitfield encoding: offset = member.offset & 0xffffff
                            total_bit_offset += (member.offset & 0x00ff_ffff) as u64;
                        } else {
                            total_bit_offset += member.offset as u64;
                        }
                        current_type_id = member.type_id;
                        found = true;
                        break;
                    }
                }

                // Handle anonymous struct/union members.
                if member.name_off == 0 {
                    if let Ok(inner_id) = self.resolve_type(member.type_id) {
                        if let Some(inner_ty) = self.types.get(inner_id as usize) {
                            if let BtfType::Struct(_, inner_members)
                            | BtfType::Union(_, inner_members) = inner_ty
                            {
                                for inner_member in inner_members {
                                    if let Ok(inner_name) = self.string_at(inner_member.name_off) {
                                        if inner_name == field_name {
                                            let base = if kind_flag {
                                                (member.offset & 0x00ff_ffff) as u64
                                            } else {
                                                member.offset as u64
                                            };
                                            let inner = if kind_flag {
                                                (inner_member.offset & 0x00ff_ffff) as u64
                                            } else {
                                                inner_member.offset as u64
                                            };
                                            total_bit_offset += base + inner;
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
            }

            if !found {
                bail!(
                    "field '{}' not found in struct type_id {} (resolved {})",
                    field_name,
                    current_type_id,
                    resolved_id
                );
            }
        }

        // Convert bit offset to byte offset.
        if total_bit_offset % 8 != 0 {
            bail!(
                "field path results in non-byte-aligned offset: {} bits",
                total_bit_offset
            );
        }

        Ok(total_bit_offset / 8)
    }

    /// Serializes the BTF data (header + types + strings) back to bytes.
    ///
    /// This re-encodes the original type data from the raw ELF, appends
    /// any newly added types, and writes the (possibly extended) string table.
    pub fn to_bytes(&self, original_elf: &[u8]) -> Result<Vec<u8>> {
        let obj = ElfFile::<elf::FileHeader64<Endianness>>::parse(original_elf)
            .context("re-parsing ELF for BTF rebuild")?;
        let btf_section = obj.section_by_name(".BTF").context("no .BTF section")?;
        let btf_data = btf_section.data().context("reading .BTF data")?;

        let hdr_len = self.header.hdr_len as usize;
        let type_start = hdr_len + self.header.type_off as usize;
        let type_end = type_start + self.header.type_len as usize;

        // Copy the original type data verbatim
        let type_data = &btf_data[type_start..type_end];

        // Compute new type_len including appended types
        let new_type_len = self.header.type_len + self.appended_type_data.len() as u32;

        // Build new header with updated lengths
        let mut header = self.header;
        header.type_len = new_type_len;
        header.str_off = header.type_off + new_type_len;
        header.str_len = self.strings.len() as u32;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header.magic.to_le_bytes());
        buf.push(header.version);
        buf.push(header.flags);
        buf.extend_from_slice(&header.hdr_len.to_le_bytes());
        buf.extend_from_slice(&header.type_off.to_le_bytes());
        buf.extend_from_slice(&header.type_len.to_le_bytes());
        buf.extend_from_slice(&header.str_off.to_le_bytes());
        buf.extend_from_slice(&header.str_len.to_le_bytes());

        // Pad header if hdr_len > 24
        if hdr_len > 24 {
            buf.extend_from_slice(&btf_data[24..hdr_len]);
        }

        buf.extend_from_slice(type_data);
        buf.extend_from_slice(&self.appended_type_data);
        buf.extend_from_slice(&self.strings);

        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_add_and_lookup() {
        let mut btf = BtfInfo {
            header: BtfHeader {
                magic: 0xEB9F,
                version: 1,
                flags: 0,
                hdr_len: 24,
                type_off: 0,
                type_len: 0,
                str_off: 0,
                str_len: 1,
            },
            types: vec![BtfType::Void],
            strings: vec![0], // empty string at offset 0
            raw_header_len: 24,
            appended_type_data: Vec::new(),
        };

        let off = btf.add_string("0:3:7");
        assert_eq!(btf.string_at(off).unwrap(), "0:3:7");
    }
}
