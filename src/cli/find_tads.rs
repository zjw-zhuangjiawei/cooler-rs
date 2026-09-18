//! `cooler-rs find-tads` — call TAD boundaries from a contact matrix with
//! HiCExplorer's `hicFindTADs` algorithm (Rust port).

use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};

use cooler_rs::findtads::{self, MultipleTesting, Params};
use cooler_rs::{Cooler, Error, Mcool, Result};

/// Multiple-testing correction for the boundary p-values.
#[derive(Clone, Copy, ValueEnum)]
pub enum Correction {
    /// Benjamini-Hochberg false discovery rate (q-value).
    Fdr,
    /// Bonferroni family-wise error rate (p-value).
    Bonferroni,
    /// Raw p-values, no correction.
    #[value(alias = "None")]
    None,
}

#[derive(Args)]
pub struct FindTadsArgs {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output prefix (default: the input file name)
    #[arg(short = 'o', long, value_name = "PREFIX")]
    out_prefix: Option<String>,

    /// Resolution to use (.mcool input)
    #[arg(long = "res", value_name = "N")]
    res: Option<u64>,

    /// Window length (bp) considered to each side of a bin, at minimum
    #[arg(long = "min-depth", value_name = "BP")]
    min_depth: Option<i64>,

    /// Window length (bp) considered to each side of a bin, at maximum
    #[arg(long = "max-depth", value_name = "BP")]
    max_depth: Option<i64>,

    /// First step (bp) between window lengths; later steps grow as
    /// `step * x**1.5`
    #[arg(long = "step", value_name = "BP")]
    step: Option<i64>,

    /// Minimum drop of a boundary below the mean score of the bins around it
    #[arg(long = "delta", value_name = "F", default_value_t = 0.01)]
    delta: f64,

    /// Minimum distance between boundaries (bp); defaults to four bins
    #[arg(long = "min-boundary-distance", value_name = "BP")]
    min_boundary_distance: Option<i64>,

    /// Multiple-testing correction
    #[arg(
        long = "correct-for-multiple-testing",
        value_enum,
        value_name = "METHOD",
        default_value = "fdr"
    )]
    correct_for_multiple_testing: Correction,

    /// p-value (Bonferroni) or q-value (FDR) threshold
    #[arg(
        long = "threshold-comparisons",
        value_name = "F",
        default_value_t = 0.01
    )]
    threshold_comparisons: f64,

    /// `bins` column holding the correction weights (e.g. `weight`)
    #[arg(long = "norm", value_name = "NAME")]
    norm: Option<String>,

    /// Chromosomes to analyse, in this order
    #[arg(long = "chromosomes", value_name = "NAME", num_args = 1..)]
    chromosomes: Option<Vec<String>>,

    /// Number of processors to use
    #[arg(
        short = 'p',
        long = "number-of-processors",
        value_name = "N",
        default_value_t = 1
    )]
    number_of_processors: usize,
}

/// Resolve the input into a `Cooler` at the requested resolution.
fn open_cooler(args: &FindTadsArgs, fin: &str) -> Result<Cooler> {
    if fin.ends_with(".cool") {
        Cooler::open_any(fin)
    } else if fin.ends_with(".mcool") {
        let mcool = Mcool::open(fin)?;
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
        mcool.cooler(res)
    } else {
        Err(Error::InvalidInput(
            "input must be a .cool or .mcool file".into(),
        ))
    }
}

pub fn run(args: FindTadsArgs) -> Result<()> {
    let fin = args.input.to_string_lossy().to_string();
    let cooler = open_cooler(&args, &fin)?;
    let prefix = args.out_prefix.clone().unwrap_or_else(|| fin.clone());

    let params = Params {
        min_depth: args.min_depth,
        max_depth: args.max_depth,
        step: args.step,
        delta: args.delta,
        min_boundary_distance: args.min_boundary_distance,
        correction: match args.correct_for_multiple_testing {
            Correction::Fdr => MultipleTesting::Fdr,
            Correction::Bonferroni => MultipleTesting::Bonferroni,
            Correction::None => MultipleTesting::None,
        },
        threshold_comparisons: args.threshold_comparisons,
        norm: args.norm.clone(),
        chromosomes: args.chromosomes.clone(),
        out_prefix: prefix,
    };

    log::info!(
        "hicFindTADs (Rust port): delta={}, correction={:?}, threshold={}, norm={:?}, chromosomes={:?}, processors={}",
        params.delta,
        params.correction,
        params.threshold_comparisons,
        params.norm,
        params.chromosomes,
        args.number_of_processors
    );
    let t0 = Instant::now();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.number_of_processors)
        .build()
        .map_err(|e| {
            Error::InvalidInput(format!(
                "cannot build thread pool with {} threads: {e}",
                args.number_of_processors
            ))
        })?;
    let outputs = pool.install(|| findtads::run_cooler(&cooler, &params))?;

    log::info!(
        "{} boundaries at delta {} and {} domains; wrote {} in {:.1}s",
        outputs.n_boundaries,
        params.delta,
        outputs.n_domains,
        outputs.tad_score.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
