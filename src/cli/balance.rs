//! `cooler-rs balance` — out-of-core matrix balancing (IC).
//!
//! Port of `cooler.cli.balance` (cooler-python). Writes the resulting weight
//! vector to the `bins` table of the input `.cool`/`.mcool` file, or prints
//! it to stdout with `--stdout`.

use std::path::PathBuf;

use clap::Args;
use cooler_rs::{
    balance_cooler, write_bins_column, AttrValue, BalanceParams, Cooler, Error, Mcool,
};

#[derive(Args)]
pub struct BalanceArgs {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Calculate weights against intra-chromosomal data only
    #[arg(long)]
    cis_only: bool,

    /// Calculate weights against inter-chromosomal data only
    #[arg(long)]
    trans_only: bool,

    /// Number of diagonals to ignore, including the main diagonal
    /// (0 ignores nothing, 1 ignores the main diagonal, 2 ignores (-1, 0, 1))
    #[arg(long, default_value_t = 2)]
    ignore_diags: usize,

    /// Distance from the diagonal in bp to ignore; the maximum of the
    /// corresponding number of diagonals and `--ignore-diags` is used
    #[arg(long, value_name = "BP")]
    ignore_dist: Option<u64>,

    /// MAD-max filter: drop bins whose log marginal sum is more than this
    /// many median absolute deviations below their chromosome's median
    #[arg(long, default_value_t = 5)]
    mad_max: usize,

    /// Drop bins with fewer than this many nonzero elements
    #[arg(long, default_value_t = 10)]
    min_nnz: usize,

    /// Drop bins with marginal sum below this value
    #[arg(long, default_value_t = 0)]
    min_count: usize,

    /// 3-column BED file of genomic regions to mask out
    #[arg(long, value_name = "FILE")]
    blacklist: Option<PathBuf>,

    /// Number of processes to split the work between
    /// (accepted for compatibility; currently runs single-threaded)
    #[arg(short = 'p', long, default_value_t = 8)]
    nproc: usize,

    /// Number of pixels handled per chunk
    #[arg(short = 'c', long, default_value_t = 10_000_000)]
    chunksize: usize,

    /// Variance threshold for convergence
    #[arg(long, default_value_t = 1e-5)]
    tol: f64,

    /// Maximum number of iterations if convergence is not achieved
    #[arg(long, default_value_t = 200)]
    max_iters: usize,

    /// Name of the column to write to
    #[arg(long, default_value = "weight")]
    name: String,

    /// Overwrite the target dataset if it already exists
    #[arg(short = 'f', long)]
    force: bool,

    /// Check whether the weight column already exists
    #[arg(long)]
    check: bool,

    /// Print the weight column to stdout instead of saving to the file
    #[arg(long)]
    stdout: bool,

    /// What to do when balancing does not converge
    #[arg(
        long,
        value_enum,
        default_value = "store_final",
        help = "'store_final': store the final result; 'store_nan': store NaN \
                weights; 'discard': store nothing and exit; 'error': abort \
                with non-zero exit status"
    )]
    convergence_policy: ConvergencePolicy,

    /// Resolution to balance (.mcool input)
    #[arg(long, value_name = "N")]
    res: Option<u64>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum ConvergencePolicy {
    #[value(name = "store_final")]
    StoreFinal,
    #[value(name = "store_nan")]
    StoreNan,
    #[value(name = "discard")]
    Discard,
    #[value(name = "error")]
    Error,
}

/// Resolve a .cool/.mcool path into a `Cooler` and the HDF5 group path that
/// holds its collection (used for writing the weight column back).
fn resolve_cooler(args: &BalanceArgs) -> cooler_rs::Result<(Cooler, String)> {
    let fin = args.input.display().to_string();
    if fin.ends_with(".cool") {
        Ok((Cooler::open_any(&fin)?, "/".into()))
    } else if fin.ends_with(".mcool") {
        let mcool = Mcool::open(&fin)?;
        let resolutions = mcool.resolutions()?;
        let res = match (args.res, resolutions.as_slice()) {
            (Some(r), _) => r,
            (None, [only]) => *only,
            (None, _) => {
                return Err(Error::InvalidInput(format!(
                    ".mcool contains multiple resolutions; select one with --res ({resolutions:?})"
                )));
            }
        };
        let cool = mcool.cooler(res)?;
        Ok((cool, format!("/resolutions/{res}")))
    } else {
        Err(Error::InvalidInput(
            "input must be a .cool or .mcool file".into(),
        ))
    }
}

/// Map BED regions to the indices of the bins they overlap.
fn blacklist_bins(clr: &Cooler, path: &std::path::Path) -> cooler_rs::Result<Vec<usize>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::InvalidInput(format!("cannot read blacklist '{path:?}': {e}")))?;
    let chroms = clr.chroms()?;
    let bins = clr.bins()?;
    let chrom_pos: std::collections::HashMap<&str, usize> = chroms
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();

    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        // Skip a header line, if present.
        if lineno == 0 && (cols[0].to_lowercase().contains("chrom") || cols[0].starts_with('#')) {
            continue;
        }
        let chrom = cols[0];
        let start: i32 = cols[1].parse().map_err(|_| {
            Error::InvalidInput(format!("bad blacklist start at line {}", lineno + 1))
        })?;
        let end: i32 = cols[2].parse().map_err(|_| {
            Error::InvalidInput(format!("bad blacklist end at line {}", lineno + 1))
        })?;
        let cid = *chrom_pos.get(chrom).ok_or_else(|| {
            Error::InvalidInput(format!("blacklist chromosome '{chrom}' not found in file"))
        })?;
        for (i, b) in bins.iter().enumerate() {
            if b.chrom_id as usize == cid && b.start < end && b.end > start {
                out.push(i);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub fn run(args: BalanceArgs) -> cooler_rs::Result<()> {
    let cool_path = args.input.display().to_string();

    if args.check {
        let cool = resolve_cooler(&args)?.0;
        if cool.bins_has_column(&args.name)? {
            log::info!("{cool_path}: balanced (column '{name}')", name = args.name);
            return Ok(());
        }
        return Err(Error::InvalidInput(format!(
            "{cool_path}: No '{}' column found.",
            args.name
        )));
    }

    if args.cis_only && args.trans_only {
        return Err(Error::InvalidInput(
            "Provide at most one of --cis-only and --trans-only flags".into(),
        ));
    }

    let (clr, group_path) = resolve_cooler(&args)?;

    if !args.stdout && clr.bins_has_column(&args.name)? && !args.force {
        return Err(Error::InvalidInput(format!(
            "'{}' column already exists. Use --force option to overwrite.",
            args.name
        )));
    }

    log::info!("Balancing \"{cool_path}\"");

    let mut ignore_diags = args.ignore_diags;
    if let Some(dist) = args.ignore_dist {
        let binsize = clr
            .bin_size()?
            .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))?;
        let diags = dist.div_ceil(binsize) as usize;
        ignore_diags = ignore_diags.max(diags);
    }

    let blacklist = match &args.blacklist {
        Some(p) => blacklist_bins(&clr, p)?,
        None => Vec::new(),
    };

    // ponytail: `--nproc` is accepted for CLI compatibility but the port runs
    // single-threaded; add rayon over chunks when measured to need it.
    let _ = args.nproc;

    let params = BalanceParams {
        cis_only: args.cis_only,
        trans_only: args.trans_only,
        ignore_diags,
        mad_max: args.mad_max,
        min_nnz: args.min_nnz,
        min_count: args.min_count,
        blacklist,
        rescale_marginals: true,
        x0: None,
        tol: args.tol,
        max_iters: args.max_iters,
        chunksize: args.chunksize,
    };

    let (mut bias, stats) = balance_cooler(&clr, &params)?;

    let all_converged = stats.converged.iter().all(|&c| c);
    if !all_converged {
        log::error!("Iteration limit reached without convergence");
        match args.convergence_policy {
            ConvergencePolicy::StoreFinal => {
                log::error!("Storing final result. Check log to assess convergence.");
            }
            ConvergencePolicy::StoreNan => {
                log::error!("Saving weights as NaN.");
                bias.fill(f64::NAN);
            }
            ConvergencePolicy::Discard => {
                log::error!("Discarding result and aborting.");
                return Ok(());
            }
            ConvergencePolicy::Error => {
                log::error!("Discarding result and aborting.");
                return Err(Error::InvalidInput(
                    "Iteration limit reached without convergence".into(),
                ));
            }
        }
    }

    if args.stdout {
        for v in &bias {
            if v.is_nan() {
                println!();
            } else {
                println!("{v}");
            }
        }
        return Ok(());
    }

    // Release the read handle first: HDF5 won't reopen the same file for
    // writing while a read-only handle is still open.
    drop(clr);
    let attrs: Vec<(&str, AttrValue)> = vec![
        ("tol", AttrValue::F64(stats.tol)),
        ("min_nnz", AttrValue::I64(stats.min_nnz as i64)),
        ("min_count", AttrValue::I64(stats.min_count as i64)),
        ("mad_max", AttrValue::I64(stats.mad_max as i64)),
        ("cis_only", AttrValue::I64(stats.cis_only as i64)),
        ("ignore_diags", AttrValue::I64(stats.ignore_diags as i64)),
        ("scale", scalar_or_array(&stats.scale)),
        ("converged", converged_attr(&stats.converged)),
        ("var", scalar_or_array(&stats.var)),
        ("divisive_weights", AttrValue::I64(0)),
    ];
    write_bins_column(&args.input, &group_path, &args.name, &bias, &attrs)?;
    log::info!(
        "Wrote {n} weights to '{input}'::{group} bins/{name}",
        n = bias.len(),
        input = cool_path,
        group = group_path,
        name = args.name
    );
    Ok(())
}

/// Python stores genome-wide stats as scalars and cis-only stats as
/// per-chromosome arrays.
fn scalar_or_array(v: &[f64]) -> AttrValue {
    match v {
        [x] => AttrValue::F64(*x),
        many => AttrValue::F64Array(many.to_vec()),
    }
}

fn converged_attr(v: &[bool]) -> AttrValue {
    match v {
        [x] => AttrValue::I64(*x as i64),
        many => AttrValue::F64Array(many.iter().map(|&b| b as u8 as f64).collect()),
    }
}
