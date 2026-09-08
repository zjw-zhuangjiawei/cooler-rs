//! `cooler-rs convert` — convert other matrix formats to/from cooler format.
//!
//! `--from`/`--to` select the input/output formats; each (from, to) pair is
//! dispatched to its own conversion function. Format-specific options live in
//! per-format option structs (flattened, with their own help headings) and use
//! clap's declarative `required_if_eq` so they become required only when the
//! matching format is selected.

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use cooler_rs::{Chrom, CoolerWriter, Error};

/// Input matrix format.
#[derive(Clone, Copy, ValueEnum)]
pub enum InputFormat {
    /// A .cool or .mcool file (auto-detected)
    Cooler,
    /// Dense N×N whitespace-separated text matrix (the original OnTAD .mat format)
    DenseTxt,
}

/// Output matrix format.
#[derive(Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Single-resolution .cool file (HDF5)
    Cool,
    /// Multi-resolution .hic (v8) file
    Hic,
}

#[derive(Args)]
pub struct ConvertArgs {
    /// Input matrix file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Input format
    #[arg(long, value_enum, value_name = "FORMAT", default_value = "cooler")]
    from: InputFormat,

    /// Output format
    #[arg(long, value_enum, value_name = "FORMAT", default_value = "cool")]
    to: OutputFormat,

    /// Output file
    #[arg(short = 'o', long, value_name = "FILE")]
    output: PathBuf,

    #[command(flatten)]
    dense_txt: DenseTxtInputOptions,

    #[command(flatten)]
    hic: HicOutputOptions,
}

/// Options specific to `--from dense-txt`.
#[derive(Args)]
struct DenseTxtInputOptions {
    /// Chromosome name
    #[arg(
        short = 'c',
        long,
        default_value = "chr1",
        help_heading = "dense-txt input options"
    )]
    chr: String,

    /// Chromosome length in base pairs
    #[arg(
        short = 'L',
        long,
        value_name = "BP",
        required_if_eq("from", "dense-txt"),
        help_heading = "dense-txt input options"
    )]
    chrlength: Option<i32>,

    /// Resolution (bin size) in base pairs
    #[arg(
        short = 'r',
        long,
        value_name = "BP",
        required_if_eq("from", "dense-txt"),
        help_heading = "dense-txt input options"
    )]
    resolution: Option<u32>,
}

/// Options specific to `--to hic`.
#[derive(Args)]
struct HicOutputOptions {
    /// Genome identifier stored in the .hic header (.cool/.mcool carry none)
    #[arg(long, default_value = "unknown", help_heading = "hic output options")]
    genome_id: String,

    /// Name of a bins column to copy into the .hic footer as normalization
    /// vectors (e.g. "weight"); resolutions lacking the column are skipped
    #[arg(long, value_name = "COL", help_heading = "hic output options")]
    weight: Option<String>,

    /// Store the weight column under this name in the .hic (default: the
    /// column name); juicer looks up "KR"/"VC", cooler columns are "weight"
    #[arg(long, value_name = "NAME", help_heading = "hic output options")]
    weight_name: Option<String>,
}

pub fn run(args: ConvertArgs) -> cooler_rs::Result<()> {
    match (args.from, args.to) {
        (InputFormat::DenseTxt, OutputFormat::Cool) => dense_txt_to_cool(&args),
        (InputFormat::Cooler, OutputFormat::Hic) => cooler_to_hic(&args),
        _ => Err(cooler_rs::Error::InvalidInput(
            "this (--from, --to) combination is not supported".into(),
        )),
    }
}

/// Convert a `.cool`/`.mcool` file to a multi-resolution `.hic` file.
fn cooler_to_hic(args: &ConvertArgs) -> cooler_rs::Result<()> {
    cooler_rs::convert::cooler_to_hic(
        &args.input,
        &args.output,
        &args.hic.genome_id,
        args.hic.weight.as_deref(),
        args.hic.weight_name.as_deref(),
    )
}

/// Convert a dense N×N text matrix to a single-chromosome `.cool` file.
fn dense_txt_to_cool(args: &ConvertArgs) -> cooler_rs::Result<()> {
    let text = std::fs::read_to_string(&args.input)
        .map_err(|e| Error::InvalidInput(format!("cannot read '{}': {e}", args.input.display())))?;

    let (n, pixels) = cooler_rs::convert::dense_txt_to_pixels(&text)?;
    if pixels.is_empty() {
        log::warn!("no non-zero pixels found in input");
    }

    // Required by clap when --from dense-txt (required_if_eq).
    let length = args.dense_txt.chrlength.expect("required by clap");
    let resolution = args.dense_txt.resolution.expect("required by clap");

    // The matrix dimension must match the number of bins, or the pixels
    // would reference non-existent bins (or leave bins with no data).
    let n_bins = (length as u64).div_ceil(resolution as u64);
    if n as u64 != n_bins {
        return Err(Error::InvalidInput(format!(
            "matrix is {n}×{n} but chromosome length {length} at resolution {resolution} gives {n_bins} bins; check --chrlength/--resolution"
        )));
    }

    let chrom = Chrom {
        name: args.dense_txt.chr.clone(),
        length,
    };

    let writer = CoolerWriter::create(&args.output, &[chrom], resolution)?;

    let n_pixels = pixels.len();
    writer.write_pixels(&pixels)?;

    log::info!(
        "Converted {n}×{n} matrix → {} pixels in '{}'",
        n_pixels,
        args.output.display()
    );

    Ok(())
}
