//! CO-RE post-processor CLI for BPF ELF objects.
//!
//! Supports two input modes:
//!
//! 1. **Sidecar TOML** (`--sidecar`): A hand-written file specifying
//!    exact section names, instruction indices, struct names, and field
//!    paths.
//!
//! 2. **Auto-discovery** (`--auto`): Reads `.aya.core_relo` markers
//!    emitted by the `core_read!` proc macro.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

use aya_core_postprocessor::sidecar::SidecarConfig;

#[derive(Parser)]
#[command(name = "aya-core-postprocessor")]
#[command(about = "Add CO-RE relocation records to a BPF ELF object")]
struct Cli {
    /// Path to the BPF ELF object file
    #[arg(short, long)]
    elf: PathBuf,

    /// Path to the sidecar TOML file describing relocations.
    /// Mutually exclusive with --auto.
    #[arg(short, long)]
    sidecar: Option<PathBuf>,

    /// Auto-discover relocations from .aya.core_relo markers.
    /// Mutually exclusive with --sidecar.
    #[arg(short, long)]
    auto: bool,

    /// Output path (defaults to overwriting the input)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.sidecar.is_some() && cli.auto {
        bail!("--sidecar and --auto are mutually exclusive");
    }
    if cli.sidecar.is_none() && !cli.auto {
        bail!("specify either --sidecar <FILE> or --auto");
    }

    let elf_data =
        std::fs::read(&cli.elf).with_context(|| format!("reading ELF: {:?}", cli.elf))?;

    let output_path = cli.output.as_ref().unwrap_or(&cli.elf);

    if let Some(sidecar_path) = &cli.sidecar {
        let sidecar_text = std::fs::read_to_string(sidecar_path)
            .with_context(|| format!("reading sidecar: {:?}", sidecar_path))?;
        let config: SidecarConfig =
            toml::from_str(&sidecar_text).context("parsing sidecar TOML")?;

        if config.relocation.is_empty() {
            eprintln!("No relocations specified in sidecar file, nothing to do.");
            return Ok(());
        }

        let result = aya_core_postprocessor::process_elf(&elf_data, &config)?;
        std::fs::write(output_path, &result)
            .with_context(|| format!("writing output: {:?}", output_path))?;

        eprintln!(
            "Wrote {} CO-RE relocations to {:?}",
            config.relocation.len(),
            output_path
        );
    } else {
        // Auto-discovery mode.
        let markers = aya_core_postprocessor::marker_parser::parse_markers_from_elf(&elf_data)
            .context("checking for .aya.core_relo markers")?;

        if markers.is_empty() {
            eprintln!("No .aya.core_relo markers found, nothing to do.");
            return Ok(());
        }

        eprintln!("Found {} markers, processing...", markers.len());
        let result = aya_core_postprocessor::process_elf_auto(&elf_data)?;
        std::fs::write(output_path, &result)
            .with_context(|| format!("writing output: {:?}", output_path))?;
        eprintln!("CO-RE relocations written to {:?}", output_path);
    }

    Ok(())
}
