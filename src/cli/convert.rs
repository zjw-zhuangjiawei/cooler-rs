//! `cooler-rs convert` — convert between Hi-C container formats.
//!
//! The *output* format is the subcommand, so the format flags and the
//! `required_if_eq` machinery they needed are gone. `.cool`/`.mcool` input is
//! auto-detected. Plain-text matrix formats are not conversions of this kind:
//! they live under `dump`/`load`.

use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct ConvertArgs {
    #[command(subcommand)]
    command: Target,
}

#[derive(Subcommand)]
enum Target {
    /// Convert a .cool/.mcool file to a multi-resolution .hic file
    Hic(HicArgs),
}

#[derive(Args)]
struct HicArgs {
    /// Input .cool or .mcool file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output .hic file
    #[arg(short = 'o', long, value_name = "FILE")]
    output: PathBuf,

    /// Genome identifier stored in the .hic header (.cool/.mcool carry none)
    #[arg(long, default_value = "unknown")]
    genome_id: String,

    /// Only convert these resolutions (bin sizes); by default every
    /// resolution of the input is converted
    #[arg(long, value_name = "BP", num_args = 1..)]
    resolutions: Vec<u32>,

    /// Names of bins columns to copy into the .hic footer as normalization
    /// vectors (e.g. "weight"); repeat the flag for several. Resolutions
    /// lacking a column get no vector for it
    #[arg(long, value_name = "COL", num_args = 1..)]
    weight: Vec<String>,

    /// Store the weight column under this name in the .hic (default: the
    /// column name); juicer looks up "KR"/"VC", cooler columns are "weight".
    /// Only valid with a single weight column
    #[arg(long, value_name = "NAME")]
    weight_name: Option<String>,
}

pub fn run(args: ConvertArgs) -> cooler_rs::Result<()> {
    match args.command {
        Target::Hic(a) => cooler_rs::convert::cooler_to_hic(
            &a.input,
            &a.output,
            &a.genome_id,
            &a.weight,
            a.weight_name.as_deref(),
            &a.resolutions,
        ),
    }
}
