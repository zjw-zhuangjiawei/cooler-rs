//! `mat2cool` — convert a dense N×N text matrix to a `.cool` file.
//!
//! Reads a tab/space-separated dense contact matrix (the original OnTAD
//! `.mat` format) and writes it as a single-chromosome `.cool` file.
//! Only upper-triangle non-zero entries are stored (symmetric-upper sparse
//! storage); zeros are implicit and omitted.

use std::path::PathBuf;

use clap::Parser;
use cooler_rs::{Chrom, CoolerWriter, Pixel};

#[derive(Parser)]
#[command(
    name = "mat2cool",
    about = "Convert a dense N×N text matrix to a .cool file",
    version
)]
struct Args {
    /// Input dense text matrix
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output .cool file
    #[arg(short = 'o', long, value_name = "FILE")]
    output: PathBuf,

    /// Chromosome name
    #[arg(short = 'c', long, default_value = "chr1")]
    chr: String,

    /// Chromosome length in base pairs
    #[arg(short = 'L', long, value_name = "BP")]
    chrlength: i32,

    /// Resolution (bin size) in base pairs
    #[arg(short = 'r', long, value_name = "BP")]
    resolution: u32,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let text = std::fs::read_to_string(&args.input).unwrap_or_else(|e| {
        log::error!("cannot read '{}': {e}", args.input.display());
        std::process::exit(1);
    });

    let mut pixels: Vec<Pixel> = Vec::new();
    let mut width: Option<usize> = None;

    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = 0;
        for (j, field) in line.split_whitespace().enumerate() {
            let v: f64 = field.parse().unwrap_or_else(|_| {
                log::error!(
                    "line {}, column {}: '{}' is not a number",
                    i + 1,
                    j + 1,
                    field
                );
                std::process::exit(1);
            });
            if j >= i && v > 0.0 {
                pixels.push(Pixel {
                    bin1_id: i as i64,
                    bin2_id: j as i64,
                    count: v,
                });
            }
            cols += 1;
        }
        match width {
            None => width = Some(cols),
            Some(w) if w != cols => {
                log::error!(
                    "input is not a square N×N matrix: line {} has {cols} columns, expected {w}",
                    i + 1
                );
                std::process::exit(1);
            }
            _ => {}
        }
    }

    let n = width.unwrap_or(0);
    if n == 0 {
        log::error!("input is empty");
        std::process::exit(1);
    }
    if pixels.is_empty() {
        log::warn!("no non-zero pixels found in input");
    }

    let chrom = Chrom {
        name: args.chr,
        length: args.chrlength,
    };

    let writer = CoolerWriter::create(&args.output, &[chrom], args.resolution)
        .unwrap_or_else(|e| {
            log::error!("cannot create '{}': {e}", args.output.display());
            std::process::exit(1);
        });

    let n_pixels = pixels.len();
    writer.write_pixels(&pixels).unwrap_or_else(|e| {
        log::error!("cannot write pixels: {e}");
        std::process::exit(1);
    });

    log::info!(
        "Converted {n}×{n} matrix → {} pixels in '{}'",
        n_pixels,
        args.output.display()
    );
}
