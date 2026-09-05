//! `cooler-rs normalize` — contact matrix normalization.
//!
//! `--method ic` (default) is iterative correction, a port of
//! `cooler.cli.balance`; `--method raichu` is the Raichu sliding-window
//! optimizer, a port of RaichuNorm v1.1. Both write a per-bin bias column back
//! to the input `.cool`/`.mcool` file (or print it to stdout with `--stdout`).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use cooler_rs::{
    balance_cooler, raichu_normalize, write_bins_column, AttrValue, BalanceParams, Cooler, Error,
    Mcool, RaichuParams,
};

/// Normalization method.
#[derive(Clone, Copy, ValueEnum)]
pub enum Method {
    /// Iterative correction (port of `cooler balance`)
    Ic,
    /// Raichu sliding-window normalization (port of RaichuNorm)
    Raichu,
}

#[derive(Args)]
pub struct NormalizeArgs {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Normalization method
    #[arg(long, value_enum, value_name = "METHOD", default_value = "ic")]
    method: Method,

    /// Resolution to normalize (.mcool input)
    #[arg(long, value_name = "N")]
    res: Option<u64>,

    /// Name of the column to write to (default: 'weight' for ic,
    /// 'obj_weight' for raichu)
    #[arg(short = 'n', long, value_name = "NAME")]
    name: Option<String>,

    /// Overwrite the target dataset if it already exists
    #[arg(short = 'f', long)]
    force: bool,

    /// Number of processes to split the work between
    /// (accepted for compatibility; currently runs single-threaded)
    #[arg(short = 'p', long, default_value_t = 8)]
    nproc: usize,

    /// Number of diagonals to ignore, including the main diagonal
    /// (default: 2 for ic, 0 for raichu)
    #[arg(long, value_name = "N")]
    ignore_diags: Option<usize>,

    /// Drop bins with fewer than this many nonzero elements
    #[arg(long, default_value_t = 10)]
    min_nnz: usize,

    #[command(flatten)]
    ic: IcOptions,

    #[command(flatten)]
    raichu: RaichuOptions,
}

/// Options for `--method ic` (iterative correction).
#[derive(Args)]
struct IcOptions {
    /// Calculate weights against intra-chromosomal data only
    #[arg(long, help_heading = "IC options")]
    cis_only: bool,

    /// Calculate weights against inter-chromosomal data only
    #[arg(long, help_heading = "IC options")]
    trans_only: bool,

    /// Distance from the diagonal in bp to ignore; the maximum of the
    /// corresponding number of diagonals and `--ignore-diags` is used
    #[arg(long, value_name = "BP", help_heading = "IC options")]
    ignore_dist: Option<u64>,

    /// MAD-max filter: drop bins whose log marginal sum is more than this
    /// many median absolute deviations below their chromosome's median
    #[arg(long, default_value_t = 5, help_heading = "IC options")]
    mad_max: usize,

    /// Drop bins with marginal sum below this value
    #[arg(long, default_value_t = 0, help_heading = "IC options")]
    min_count: usize,

    /// 3-column BED file of genomic regions to mask out
    #[arg(long, value_name = "FILE", help_heading = "IC options")]
    blacklist: Option<PathBuf>,

    /// Number of pixels handled per chunk
    #[arg(
        short = 'c',
        long,
        default_value_t = 10_000_000,
        help_heading = "IC options"
    )]
    chunksize: usize,

    /// Variance threshold for convergence
    #[arg(long, default_value_t = 1e-5, help_heading = "IC options")]
    tol: f64,

    /// Maximum number of iterations if convergence is not achieved
    #[arg(long, default_value_t = 200, help_heading = "IC options")]
    max_iters: usize,

    /// Print the weight column to stdout instead of saving to the file
    #[arg(long, help_heading = "IC options")]
    stdout: bool,

    /// Check whether the weight column already exists
    #[arg(long, help_heading = "IC options")]
    check: bool,

    /// What to do when balancing does not converge
    #[arg(
        long,
        value_enum,
        default_value = "store_final",
        help_heading = "IC options",
        help = "'store_final': store the final result; 'store_nan': store NaN \
                weights; 'discard': store nothing and exit; 'error': abort \
                with non-zero exit status"
    )]
    convergence_policy: ConvergencePolicy,
}

/// Options for `--method raichu` (Raichu).
#[derive(Args)]
struct RaichuOptions {
    /// Chromosome names to include (matched exactly; empty = all)
    #[arg(
        short = 'C',
        long = "chroms",
        value_name = "NAME",
        help_heading = "Raichu options"
    )]
    chroms: Vec<String>,

    /// BED file of genomic regions to restrict calculation to
    #[arg(long, value_name = "FILE", help_heading = "Raichu options")]
    regions: Option<PathBuf>,

    /// Size of the sliding window, in bins
    #[arg(long, default_value_t = 200, help_heading = "Raichu options")]
    window_size: usize,

    /// Maximum genomic distance to consider, in bins
    #[arg(long, default_value_t = 200, help_heading = "Raichu options")]
    max_distance: usize,

    /// Maximum number of global search iterations (dual annealing)
    #[arg(long, default_value_t = 100, help_heading = "Raichu options")]
    maxiter: usize,

    /// Lower bound of the search space
    #[arg(long, default_value_t = 0.001, help_heading = "Raichu options")]
    lower_bound: f64,

    /// Upper bound of the search space
    #[arg(long, default_value_t = 1000.0, help_heading = "Raichu options")]
    upper_bound: f64,

    /// Number of threads (accepted for compatibility; runs single-threaded)
    #[arg(
        short = 't',
        long,
        default_value_t = 2,
        help_heading = "Raichu options"
    )]
    threads: usize,
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
fn resolve_cooler(input: &Path, res: Option<u64>) -> cooler_rs::Result<(Cooler, String)> {
    let fin = input.display().to_string();
    if fin.ends_with(".cool") {
        Ok((Cooler::open_any(&fin)?, "/".into()))
    } else if fin.ends_with(".mcool") {
        let mcool = Mcool::open(&fin)?;
        let resolutions = mcool.resolutions()?;
        let res = match (res, resolutions.as_slice()) {
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

/// Map BED regions to the indices of the bins they overlap (IC blacklist).
fn blacklist_bins(clr: &Cooler, path: &Path) -> cooler_rs::Result<Vec<usize>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::InvalidInput(format!("cannot read blacklist '{path:?}': {e}")))?;
    let chroms = clr.chroms()?;
    let bins = clr.bins()?;
    let chrom_pos: HashMap<&str, usize> = chroms
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

/// `load_BED` from Raichu: map each BED row to its local bin indices.
fn load_bed(path: &Path, bin_size: u64) -> cooler_rs::Result<HashMap<String, Vec<usize>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::InvalidInput(format!("cannot read regions '{path:?}': {e}")))?;
    let res = bin_size as i64;
    let mut d: HashMap<String, BTreeSet<usize>> = HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let chrom = cols[0].to_string();
        let s: i64 = cols[1]
            .parse()
            .map_err(|_| Error::InvalidInput("bad region start".into()))?;
        let e: i64 = cols[2]
            .parse()
            .map_err(|_| Error::InvalidInput("bad region end".into()))?;
        let lo = s / res;
        let hi = (e + res - 1) / res;
        let set = d.entry(chrom).or_default();
        for b in lo..hi {
            set.insert(b as usize);
        }
    }
    Ok(d.into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect())
}

pub fn run(args: NormalizeArgs) -> cooler_rs::Result<()> {
    let _ = args.nproc; // accepted for CLI compatibility; single-threaded
    match args.method {
        Method::Ic => run_ic(&args),
        Method::Raichu => run_raichu(&args),
    }
}

fn run_ic(args: &NormalizeArgs) -> cooler_rs::Result<()> {
    let cool_path = args.input.display().to_string();
    let name = args.name.clone().unwrap_or_else(|| "weight".into());
    let ignore_diags = args.ignore_diags.unwrap_or(2);

    if args.ic.check {
        let cool = resolve_cooler(&args.input, args.res)?.0;
        if cool.bins_has_column(&name)? {
            log::info!("{cool_path}: balanced (column '{name}')");
            return Ok(());
        }
        return Err(Error::InvalidInput(format!(
            "{cool_path}: No '{name}' column found."
        )));
    }

    if args.ic.cis_only && args.ic.trans_only {
        return Err(Error::InvalidInput(
            "Provide at most one of --cis-only and --trans-only flags".into(),
        ));
    }

    let (clr, group_path) = resolve_cooler(&args.input, args.res)?;

    if !args.ic.stdout && clr.bins_has_column(&name)? && !args.force {
        return Err(Error::InvalidInput(format!(
            "'{name}' column already exists. Use --force option to overwrite."
        )));
    }

    log::info!("Balancing \"{cool_path}\"");

    let mut ignore_diags = ignore_diags;
    if let Some(dist) = args.ic.ignore_dist {
        let binsize = clr
            .bin_size()?
            .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))?;
        let diags = dist.div_ceil(binsize) as usize;
        ignore_diags = ignore_diags.max(diags);
    }

    let blacklist = match &args.ic.blacklist {
        Some(p) => blacklist_bins(&clr, p)?,
        None => Vec::new(),
    };

    let params = BalanceParams {
        cis_only: args.ic.cis_only,
        trans_only: args.ic.trans_only,
        ignore_diags,
        mad_max: args.ic.mad_max,
        min_nnz: args.min_nnz,
        min_count: args.ic.min_count,
        blacklist,
        rescale_marginals: true,
        x0: None,
        tol: args.ic.tol,
        max_iters: args.ic.max_iters,
        chunksize: args.ic.chunksize,
    };

    let (mut bias, stats) = balance_cooler(&clr, &params)?;

    let all_converged = stats.converged.iter().all(|&c| c);
    if !all_converged {
        log::error!("Iteration limit reached without convergence");
        match args.ic.convergence_policy {
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

    if args.ic.stdout {
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
    write_bins_column(&args.input, &group_path, &name, &bias, &attrs)?;
    log::info!(
        "Wrote {n} weights to '{input}'::{group} bins/{name}",
        n = bias.len(),
        input = cool_path,
        group = group_path,
        name = name
    );
    Ok(())
}

fn run_raichu(args: &NormalizeArgs) -> cooler_rs::Result<()> {
    let cool_path = args.input.display().to_string();
    let name = args.name.clone().unwrap_or_else(|| "obj_weight".into());

    let (clr, group_path) = resolve_cooler(&args.input, args.res)?;

    if clr.bins_has_column(&name)? && !args.force {
        return Err(Error::InvalidInput(format!(
            "'{name}' column already exists. Use --force option to overwrite."
        )));
    }

    let bin_size = clr
        .bin_size()?
        .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))?;

    let included_bins = match &args.raichu.regions {
        Some(p) => Some(load_bed(p, bin_size)?),
        None => None,
    };

    let params = RaichuParams {
        window_size: args.raichu.window_size,
        max_distance: args.raichu.max_distance,
        ignore_diags: args.ignore_diags.unwrap_or(0),
        min_nnz: args.min_nnz,
        maxiter: args.raichu.maxiter,
        lower_bound: args.raichu.lower_bound,
        upper_bound: args.raichu.upper_bound,
        chroms: args.raichu.chroms.clone(),
        included_bins,
        ..RaichuParams::default()
    };

    log::info!("Normalizing \"{cool_path}\" with Raichu");
    let bias = raichu_normalize(&clr, &params)?;

    // Release the read handle before writing back to the same file.
    // `write_bins_column` unlinks any existing column first, so `--force`
    // needs no separate delete.
    drop(clr);
    write_bins_column(&args.input, &group_path, &name, &bias, &[])?;
    log::info!(
        "Wrote {n} weights to '{input}'::{group} bins/{name}",
        n = bias.len(),
        input = cool_path,
        group = group_path,
        name = name
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
