//! CO-RE post-processor for BPF ELF objects.
//!
//! Takes a compiled BPF ELF file and a sidecar TOML file describing
//! field accesses, and adds `bpf_core_relo` records to the `.BTF.ext`
//! section so that aya's `relocate_btf()` can patch field offsets at
//! load time for different kernels.

mod btf_ext_writer;
mod btf_parser;
mod elf_patcher;
mod sidecar;

#[cfg(test)]
mod test_helpers;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use btf_ext_writer::BtfExtWriter;
use btf_parser::BtfInfo;
use sidecar::SidecarConfig;

#[derive(Parser)]
#[command(name = "aya-core-postprocessor")]
#[command(about = "Add CO-RE relocation records to a BPF ELF object")]
struct Cli {
    /// Path to the BPF ELF object file
    #[arg(short, long)]
    elf: PathBuf,

    /// Path to the sidecar TOML file describing relocations
    #[arg(short, long)]
    sidecar: PathBuf,

    /// Output path (defaults to overwriting the input)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let elf_data =
        std::fs::read(&cli.elf).with_context(|| format!("reading ELF: {:?}", cli.elf))?;

    let sidecar_text = std::fs::read_to_string(&cli.sidecar)
        .with_context(|| format!("reading sidecar: {:?}", cli.sidecar))?;
    let config: SidecarConfig =
        toml::from_str(&sidecar_text).context("parsing sidecar TOML")?;

    if config.relocation.is_empty() {
        eprintln!("No relocations specified in sidecar file, nothing to do.");
        return Ok(());
    }

    let output_path = cli.output.as_ref().unwrap_or(&cli.elf);
    let result = process_elf(&elf_data, &config)?;
    std::fs::write(output_path, &result)
        .with_context(|| format!("writing output: {:?}", output_path))?;

    eprintln!(
        "Wrote {} CO-RE relocations to {:?}",
        config.relocation.len(),
        output_path
    );
    Ok(())
}

/// Main processing pipeline.
fn process_elf(elf_data: &[u8], config: &SidecarConfig) -> Result<Vec<u8>> {
    // Step 1: Parse the existing BTF from the .BTF section.
    let btf = BtfInfo::parse_from_elf(elf_data).context("parsing BTF from ELF")?;

    // Step 2: Build CO-RE relocation records.
    let mut writer = BtfExtWriter::new(&btf);

    for (i, relo) in config.relocation.iter().enumerate() {
        writer
            .add_relocation(relo)
            .with_context(|| format!("processing relocation #{i}: {relo:?}"))?;
    }

    // Step 3: Generate the new .BTF and .BTF.ext section contents.
    let (new_btf_data, new_btf_ext_data) = writer.finish(elf_data)?;

    // Step 4: Patch the ELF with the new section contents.
    elf_patcher::patch_elf_sections(elf_data, &new_btf_data, &new_btf_ext_data)
}
