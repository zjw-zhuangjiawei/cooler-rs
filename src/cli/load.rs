//! `cooler-rs load` — build a `.cool` from an external text format.
//!
//! The input format is a dense N×N whitespace-separated text matrix (the
//! original OnTAD `.mat`), parsed by [`dense_txt_to_pixels`]. That format is
//! owned by the `dump` module, which also writes it back out
//! (`dump matrix`), so the two are inverses.
//!
//! A pairs file would be a second input format here. When it lands, that is
//! the moment to split this into one subcommand per input format.

use std::path::PathBuf;

use clap::Args;
use cooler_rs::{Chrom, CoolerWriter, Error};

use super::dump::dense_txt_to_pixels;

#[derive(Args)]
pub struct LoadArgs {
    /// Dense N×N whitespace-separated text matrix (the OnTAD .mat format)
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output .cool file
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: PathBuf,

    /// Chromosome name
    #[arg(short = 'c', long, default_value = "chr1")]
    pub chr: String,

    /// Chromosome length in base pairs
    #[arg(short = 'L', long, value_name = "BP")]
    pub chrlength: i32,

    /// Resolution (bin size) in base pairs
    #[arg(short = 'r', long, value_name = "BP")]
    pub resolution: u32,
}

pub fn run(args: LoadArgs) -> cooler_rs::Result<()> {
    let text = std::fs::read_to_string(&args.input)
        .map_err(|e| Error::InvalidInput(format!("cannot read '{}': {e}", args.input.display())))?;

    let (n, pixels) = dense_txt_to_pixels(&text)?;
    if pixels.is_empty() {
        log::warn!("no non-zero pixels found in input");
    }

    // The matrix dimension must match the number of bins, or the pixels
    // would reference non-existent bins (or leave bins with no data).
    let n_bins = (args.chrlength as u64).div_ceil(args.resolution as u64);
    if n as u64 != n_bins {
        return Err(Error::InvalidInput(format!(
            "matrix is {n}×{n} but chromosome length {} at resolution {} gives {n_bins} bins; \
             check --chrlength/--resolution",
            args.chrlength, args.resolution
        )));
    }

    let chrom = Chrom {
        name: args.chr.clone(),
        length: args.chrlength,
    };
    let writer = CoolerWriter::create(&args.output, &[chrom], args.resolution)?;
    let n_pixels = pixels.len();
    writer.write_pixels(&pixels)?;

    log::info!(
        "Loaded {n}×{n} matrix → {n_pixels} pixels in '{}'",
        args.output.display()
    );

    Ok(())
}
