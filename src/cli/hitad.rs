//! `hitad`: TADLib's hierarchical domain caller (whole genome, one shared
//! HMM per dataset). Port of `scripts/hitad`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use cooler_rs::hitad::{Dataset, Genome, DEFAULT_MAXSIZE, DEFAULT_MIN_CHROM_SIZE};

/// Call hierarchical domains genome-wide (TADLib hitad port).
#[derive(Debug, Parser)]
pub struct HitadArgs {
    /// Metadata file describing Hi-C datasets: 'res:N' lines open a
    /// resolution block, following lines are 'label:path' rows (TADLib -d)
    #[arg(short = 'd', long)]
    pub datasets: PathBuf,
    /// Output file name (4-column: chrom, start, end, level)
    #[arg(short = 'O', long)]
    pub output: PathBuf,
    /// Name of the bins column used to construct the normalized matrix
    /// ('RAW' for the raw matrix)
    #[arg(short = 'W', long, default_value = "weight")]
    pub weight_col: String,
    /// Chromosomes to exclude (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = "chrY,chrM")]
    pub exclude: Vec<String>,
    /// Only chromosomes with at least this size (bp) are considered
    #[arg(long, default_value_t = DEFAULT_MIN_CHROM_SIZE)]
    pub minimum_chrom_size: u64,
    /// Maximum domain size (bp)
    #[arg(long, default_value_t = DEFAULT_MAXSIZE)]
    pub maxsize: u64,
}

/// Parse the hitad metadata file (TADLib `datasets_convert`): 'res:N' lines
/// open a resolution block, following non-empty lines are 'label:path' rows
/// (the path may contain ':'; only the first splits).
fn parse_datasets(path: &Path) -> cooler_rs::Result<Vec<Dataset>> {
    let text = fs::read_to_string(path)?;
    let mut datasets = Vec::new();
    let mut res: Option<u64> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(r) = line.strip_prefix("res:") {
            res = Some(r.trim().parse().map_err(|_| {
                cooler_rs::Error::InvalidInput(format!("bad resolution line: {line}"))
            })?);
        } else {
            let Some((label, uri)) = line.split_once(':') else {
                return Err(cooler_rs::Error::InvalidInput(format!(
                    "bad dataset row: {line}"
                )));
            };
            let res = res.ok_or_else(|| {
                cooler_rs::Error::InvalidInput(format!(
                    "dataset row before any 'res:' line: {line}"
                ))
            })?;
            datasets.push((res, label.trim().to_string(), uri.trim().to_string()));
        }
    }
    if datasets.is_empty() {
        return Err(cooler_rs::Error::InvalidInput(format!(
            "no datasets in {}",
            path.display()
        )));
    }
    Ok(datasets)
}

pub fn run(args: HitadArgs) -> cooler_rs::Result<()> {
    let t0 = Instant::now();
    log::info!("hitad (Rust port of TADLib)");
    let datasets = parse_datasets(&args.datasets)?;
    let exclude: Vec<&str> = args.exclude.iter().map(|s| s.as_str()).collect();
    let mut genome = Genome::new(
        &datasets,
        Some(args.weight_col.as_str()),
        &exclude,
        args.minimum_chrom_size,
        args.maxsize,
    )?;
    log::info!(
        " Loaded {} datasets ({}), {} chromosomes excluded",
        datasets.len(),
        datasets
            .iter()
            .map(|(res, rep, _)| format!("{rep}@{res}"))
            .collect::<Vec<_>>()
            .join(", "),
        genome.exclude.len()
    );

    genome.learning();
    log::info!(" HMM training done ({:.1?})", t0.elapsed());

    genome.call_domains();
    log::info!(
        " Called {} hierarchical domains ({:.1?})",
        genome.results.len(),
        t0.elapsed()
    );

    let mut out = fs::File::create(&args.output)?;
    for (chrom, start, end, level) in &genome.results {
        writeln!(out, "{chrom}\t{start}\t{end}\t{level}")?;
    }
    log::info!(" Output to {}", args.output.display());
    Ok(())
}
