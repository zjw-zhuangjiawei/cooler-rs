//! `cooler-rs call-tad` — call hierarchical TADs from a Hi-C contact matrix.
//!
//! The TAD calling algorithm is a subcommand. Every method shares
//! [`CommonArgs`] (input, output prefix, chromosome, resolution, threads,
//! log2, normalization) and adds its own option group; method-specific fields
//! are `Option<T>` so the defaults are resolved in `run()`.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Subcommand, ValueEnum};
use rand::Rng;

use cooler_rs::armatus;
use cooler_rs::arrowhead;
use cooler_rs::domaincaller::Chrom;
use cooler_rs::findtads::{self, MultipleTesting};
use cooler_rs::ontad::{self, Params};
use cooler_rs::{ChromMeta, Cooler, Error, File, Mcool};

/// Arguments every `call-tad` method shares.
#[derive(Args)]
struct CommonArgs {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output prefix (default: the input file name)
    #[arg(short = 'o', long, value_name = "PREFIX")]
    output: Option<String>,

    /// Chromosome to extract (methods that work on a single chromosome)
    #[arg(long = "chr", value_name = "NAME")]
    chr: Option<String>,

    /// Resolution to use (.mcool or .hic input)
    #[arg(long = "res", value_name = "N")]
    res: Option<u64>,

    /// Worker threads, for the per-window and per-chromosome parallelism
    #[arg(long, value_name = "N", default_value_t = 4)]
    threads: usize,

    /// Apply log2(x + 1) to the matrix
    #[arg(long = "log2")]
    log2: bool,

    /// Normalization to apply (a `bins` column for a `.cool`/`.mcool`, a
    /// normalization type for a `.hic`; `NONE` means raw counts)
    #[arg(long, value_name = "NAME")]
    norm: Option<String>,
}

impl CommonArgs {
    /// The input path as a string, which every runner keys its output names
    /// off.
    fn fin(&self) -> String {
        self.input.display().to_string()
    }

    /// The output prefix: `-o` when given, the input path otherwise.
    fn prefix(&self) -> String {
        self.output.clone().unwrap_or_else(|| self.fin())
    }
}

/// Which TAD caller to run.
#[derive(Subcommand)]
enum Method {
    /// OnTAD v1.4 (An et al., Genome Biology 2019; Rust port)
    Ontad(OntadArgs),

    /// DomainCaller (Dixon et al., Nature 2012; Rust port of TADLib)
    Domaincaller(DomaincallerArgs),

    /// Armatus 2.3 (Filippova et al., Algorithms Mol Biol 2014; Rust port)
    Armatus(ArmatusArgs),

    /// Arrowhead (Huntley & Durand, Cell Syst 2016; Rust port of juicer)
    Arrowhead(ArrowheadArgs),

    /// hicFindTADs (HiCExplorer): TAD-separation score and boundary caller
    Hicexplorer(HicexplorerArgs),
}

#[derive(Args)]
pub struct CallTadArgs {
    #[command(subcommand)]
    method: Method,
}

#[derive(Args)]
struct OntadArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    ontad: OntadOptions,
}

#[derive(Args)]
struct DomaincallerArgs {
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Args)]
struct ArmatusArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    armatus: ArmatusOptions,
}

#[derive(Args)]
struct ArrowheadArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    arrowhead: ArrowheadOptions,
}

#[derive(Args)]
struct HicexplorerArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    hicexplorer: HicexplorerOptions,
}

#[derive(Args)]
struct OntadOptions {
    /// Penalty for adding a TAD
    #[arg(long, value_name = "F")]
    penalty: Option<f64>,

    /// Maximum TAD size in bins
    #[arg(long, value_name = "N")]
    maxsz: Option<usize>,

    /// Minimum TAD size in bins
    #[arg(long, value_name = "N")]
    minsz: Option<usize>,

    /// Local-minimum window half-size
    #[arg(long, value_name = "N")]
    lsize: Option<usize>,

    /// Local-minimum threshold in stddevs
    #[arg(long, value_name = "F")]
    ldiff: Option<f64>,

    /// Shuffle each diagonal (null model)
    #[arg(long)]
    shuffle: bool,

    /// Also write a .bed file
    #[arg(long)]
    bedout: bool,
}

impl OntadOptions {
    /// Resolve the OnTAD defaults (kept out of clap so that explicitly-set
    /// options can be told apart from defaults when validating against
    /// `--method`).
    fn params(&self) -> Params {
        Params {
            maxsz: self.maxsz.unwrap_or(200).max(10),
            minsz: self.minsz.unwrap_or(3).max(1),
            penalty: self.penalty.unwrap_or(0.1),
            lsize: self.lsize.unwrap_or(5),
            ldiff: self.ldiff.unwrap_or(1.96),
        }
    }
}

/// Options specific to `--method armatus`.
#[derive(Args)]
struct ArmatusOptions {
    /// Highest gamma (resolution) to generate domains at
    #[arg(long, value_name = "G")]
    gamma: Option<f64>,

    /// Step size between resolutions
    #[arg(long, value_name = "S")]
    step: Option<f64>,

    /// Number of near-optimal solutions per resolution
    #[arg(long, value_name = "K")]
    top_k: Option<usize>,

    /// Minimum samples required to compute a per-size mean
    #[arg(long, value_name = "N")]
    min_mean_samples: Option<usize>,

    /// Only generate domains at the maximum gamma
    #[arg(long)]
    just_gamma_max: bool,

    /// Also write per-resolution domain files
    #[arg(long)]
    multiscale: bool,

    /// Apply natural log to positive counts (Armatus sparse/Rao semantics)
    #[arg(long)]
    log: bool,
}

impl ArmatusOptions {
    fn params(&self) -> armatus::Params {
        armatus::Params {
            gamma_max: self.gamma.unwrap_or(0.5),
            step_size: self.step.unwrap_or(0.05),
            top_k: self.top_k.unwrap_or(1),
            min_mean_samples: self.min_mean_samples.unwrap_or(100),
            just_gamma_max: self.just_gamma_max,
        }
    }
}

/// Options specific to `--method arrowhead`.
#[derive(Args)]
struct ArrowheadOptions {
    /// Sliding-window width along the diagonal, in bins
    #[arg(long, value_name = "N")]
    window: Option<usize>,

    /// High-confidence variance threshold
    #[arg(long, value_name = "F")]
    var_threshold: Option<f64>,

    /// High-confidence sign threshold
    #[arg(long, value_name = "F")]
    high_sign: Option<f64>,

    /// Low-confidence sign threshold sweep start (max)
    #[arg(long, value_name = "F")]
    max_low_sign: Option<f64>,

    /// Low-confidence sign threshold sweep end (min)
    #[arg(long, value_name = "F")]
    min_low_sign: Option<f64>,

    /// Low-confidence sign threshold sweep step
    #[arg(long, value_name = "F")]
    decrement_low_sign: Option<f64>,

    /// Minimum domain width, in bins
    #[arg(long, value_name = "N")]
    min_block_size: Option<usize>,

    /// Upstream/downstream gap for the directionality index
    #[arg(long, value_name = "N")]
    gap: Option<usize>,
}

impl ArrowheadOptions {
    fn params(&self) -> arrowhead::Params {
        arrowhead::Params {
            matrix_width: self.window.unwrap_or(2000),
            var_threshold: Some(self.var_threshold.unwrap_or(0.2)),
            high_sign_threshold: self.high_sign.unwrap_or(0.5),
            max_low_sign_threshold: self.max_low_sign.unwrap_or(0.4),
            min_low_sign_threshold: self.min_low_sign.unwrap_or(0.0),
            decrement_low_sign_threshold: self.decrement_low_sign.unwrap_or(0.1),
            min_block_size: self.min_block_size.unwrap_or(60),
            gap: self.gap.unwrap_or(7),
        }
    }
}

/// Multiple-testing correction for the hicFindTADs boundary p-values.
#[derive(Clone, Copy, ValueEnum)]
enum Correction {
    /// Benjamini-Hochberg false discovery rate (q-value).
    Fdr,
    /// Bonferroni family-wise error rate (p-value).
    Bonferroni,
    /// Raw p-values, no correction.
    #[value(alias = "None")]
    None,
}

/// Options specific to `--method hicexplorer`.
///
/// hicFindTADs runs genome-wide by default and takes a *list* of chromosomes,
/// so `--chromosomes` is the natural selector here; a single `--chr` is
/// accepted too. The `--threads` value becomes its processor count.
#[derive(Args)]
struct HicexplorerOptions {
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
    correction: Correction,

    /// p-value (Bonferroni) or q-value (FDR) threshold
    #[arg(
        long = "threshold-comparisons",
        value_name = "F",
        default_value_t = 0.01
    )]
    threshold_comparisons: f64,

    /// Chromosomes to analyse, in this order
    #[arg(
        long = "chromosomes",
        value_name = "NAME",
        num_args = 1..,
    )]
    chromosomes: Option<Vec<String>>,
}

pub fn run(args: CallTadArgs) -> cooler_rs::Result<()> {
    match args.method {
        Method::Ontad(a) => run_ontad(&a.common, &a.ontad),
        Method::Domaincaller(a) => run_domaincaller(&a.common),
        Method::Armatus(a) => run_armatus(&a.common, &a.armatus),
        Method::Arrowhead(a) => run_arrowhead(&a.common, &a.arrowhead),
        Method::Hicexplorer(a) => run_hicexplorer(&a.common, &a.hicexplorer),
    }
}

/// `--method hicexplorer`: HiCExplorer's `hicFindTADs`, genome-wide.
///
/// Unlike the other methods this one has no single-chromosome path, and it
/// writes a whole set of files rather than one TAD list, so it takes the
/// cooler straight from `open_cooler_file` instead of the per-chromosome
/// offsets the other runners need.
fn run_hicexplorer(common: &CommonArgs, hx: &HicexplorerOptions) -> cooler_rs::Result<()> {
    let fin = common.fin();
    let cooler = open_cooler_file(common, &fin)?;

    let params = findtads::Params {
        min_depth: hx.min_depth,
        max_depth: hx.max_depth,
        step: hx.step,
        delta: hx.delta,
        min_boundary_distance: hx.min_boundary_distance,
        correction: match hx.correction {
            Correction::Fdr => MultipleTesting::Fdr,
            Correction::Bonferroni => MultipleTesting::Bonferroni,
            Correction::None => MultipleTesting::None,
        },
        threshold_comparisons: hx.threshold_comparisons,
        norm: common.norm.clone(),
        chromosomes: hx
            .chromosomes
            .clone()
            .or_else(|| common.chr.clone().map(|c| vec![c])),
        out_prefix: common.prefix(),
    };

    log::info!(
        "hicFindTADs (Rust port): delta={}, correction={:?}, threshold={}, norm={:?}, chromosomes={:?}, threads={}",
        params.delta,
        params.correction,
        params.threshold_comparisons,
        params.norm,
        params.chromosomes,
        common.threads
    );
    let t0 = Instant::now();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(common.threads)
        .build()
        .map_err(|e| {
            Error::InvalidInput(format!(
                "cannot build thread pool with {} threads: {e}",
                common.threads
            ))
        })?;
    let outputs = pool.install(|| findtads::run_cooler(&cooler, &params))?;

    log::info!(
        "{} boundaries and {} domains; wrote {} in {:.1}s",
        outputs.n_boundaries,
        outputs.n_domains,
        outputs.tad_score.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Resolve the input into a `Cooler` at the requested resolution, without
/// selecting a chromosome.
fn open_cooler_file(args: &CommonArgs, fin: &str) -> cooler_rs::Result<Cooler> {
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
            "input must be a .cool or .mcool file \
             (use 'cooler-rs load' to build one from a dense text matrix)"
                .into(),
        ))
    }
}

/// Resolve the input into a `Cooler` and the selected chromosome's first/last
/// bin offsets (shared by the per-method runners).
fn open_cooler(
    common: &CommonArgs,
    fin: &str,
) -> cooler_rs::Result<(Cooler, usize, usize, ChromMeta)> {
    let cool = open_cooler_file(common, fin)?;

    let chroms = cool.chroms()?;
    let chrom_id = match common.chr.as_deref() {
        Some(name) => chroms.iter().position(|c| c.name == name).ok_or_else(|| {
            let available = chroms
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            Error::InvalidInput(format!(
                "chromosome '{name}' not found (available: {available})"
            ))
        })?,
        None if chroms.len() == 1 => 0,
        None => {
            let available = chroms
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::InvalidInput(format!(
                "file contains {} chromosomes; select one with -chr ({available})",
                chroms.len()
            )));
        }
    };
    let offsets = cool.chrom_offset()?;
    let first = offsets[chrom_id] as usize;
    let last = offsets[chrom_id + 1] as usize;
    let chrom = &chroms[chrom_id];
    let meta = ChromMeta {
        name: chrom.name.clone(),
        length: chrom.length as u64,
        resolution: cool
            .bin_size()?
            .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))?,
    };
    Ok((cool, first, last, meta))
}

fn run_domaincaller(common: &CommonArgs) -> cooler_rs::Result<()> {
    let fin = common.fin();
    log::info!("DomainCaller (Rust port of TADLib)");
    let t0 = Instant::now();

    let (cool, first, last, meta) = open_cooler(common, &fin)?;
    let res = meta.resolution as usize;
    let n = last - first;

    // Upper-triangle pixels of the selected chromosome (as TADLib feeds in);
    // drop cross-chromosome pixels whose bin2 falls beyond the chromosome.
    let mut entries = Vec::new();
    for p in cool.pixels_for_bins(first as i64, last as i64)? {
        let (b1, b2) = (p.bin1_id as usize, p.bin2_id as usize);
        if b2 < last {
            entries.push((b1 - first, b2 - first, p.count));
        }
    }
    log::info!(
        " Loaded {} pixels ({} bins, {} bp/res)",
        entries.len(),
        n,
        res
    );

    let mut chrom = Chrom::new(&meta.name, res as u64, n, &entries);
    chrom.call_domains();
    log::info!(
        " Called {} domains ({:.1?})",
        chrom.domains.len(),
        t0.elapsed()
    );

    let prefix = common.prefix();
    let fdom = format!("{prefix}.domains");
    let fdi = format!("{prefix}.DIs.bedGraph");

    let mut out = std::fs::File::create(&fdom)?;
    for d in &chrom.domains {
        writeln!(out, "{}\t{}\t{}", meta.name, d[0] as u64, d[1] as u64)?;
    }
    // DI track (bedGraph), as TADLib's genomeLev writes it.
    let mut out = std::fs::File::create(&fdi)?;
    for (i, &di) in chrom.dis.iter().enumerate() {
        let start = i * res;
        let end = ((i + 1) * res).min(meta.length as usize);
        writeln!(out, "{}\t{}\t{}\t{:.4}", meta.name, start, end, di)?;
    }
    log::info!(" Output to {fdom}, {fdi}");
    log::info!("Total run time: {:.1?}", t0.elapsed());
    Ok(())
}

fn run_armatus(common: &CommonArgs, opts: &ArmatusOptions) -> cooler_rs::Result<()> {
    let fin = common.fin();
    let params = opts.params();
    log::info!(
        "Armatus 2.3 (Rust port): gamma_max={}, step={}, top_k={}, min_mean_samples={}",
        params.gamma_max,
        params.step_size,
        params.top_k,
        params.min_mean_samples
    );
    let t0 = Instant::now();

    let prefix = common.prefix();

    if let Some(_chr) = common.chr.as_deref() {
        let (cool, first, last, meta) = open_cooler(common, &fin)?;
        let res = meta.resolution as usize;
        log::info!(" Loaded {meta} bins", meta = last - first);
        let (domains, ensemble) = armatus_for_chrom(&cool, first, last, opts.log, &params)?;
        let fout = format!("{prefix}.consensus.txt");
        write_domains_bed(&fout, &[(meta.name.as_str(), &domains)], res)?;
        if opts.multiscale {
            write_multiscale(
                &prefix,
                &[(meta.name.as_str(), ensemble)],
                res,
                params.top_k,
            )?;
        }
        log::info!(" Called {} domains ({:.1?})", domains.len(), t0.elapsed());
        log::info!("Output to {fout}");
        return Ok(());
    }

    // Whole-genome: iterate every chromosome in input order. The per-chrom
    // matrices are dense (n*n f64); skipping huge chroms is the caller's job
    // (--chr). We keep this serial — the matrix allocation per chromosome is
    // the bottleneck, and parallelism would just multiply peak memory.
    let cool = open_cooler_file(common, &fin)?;
    let chroms = cool.chroms()?;
    let offsets = cool.chrom_offset()?;
    let res =
        cool.bin_size()?
            .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))? as usize;
    let n = chroms.len();
    log::info!(" Whole-genome armatus: {n} chromosomes at {res} bp/res");

    let fout = format!("{prefix}.consensus.txt");
    let mut consensus = std::fs::File::create(&fout)?;
    let mut total = 0usize;
    let mut multi_buf: Vec<(&str, armatus::WeightedDomainEnsemble)> = Vec::new();
    for (i, chrom) in chroms.iter().enumerate() {
        let first = offsets[i] as usize;
        let last = offsets[i + 1] as usize;
        if first == last {
            log::info!(" [{}/{}] {} (empty, skipped)", i + 1, n, chrom.name);
            continue;
        }
        log::info!(" [{}/{}] {}: {} bins", i + 1, n, chrom.name, last - first);
        let (domains, ensemble) = armatus_for_chrom(&cool, first, last, opts.log, &params)?;
        total += domains.len();
        for d in &domains {
            writeln!(
                consensus,
                "{}\t{}\t{}",
                chrom.name,
                d.start * res,
                (d.end + 1) * res - 1
            )?;
        }
        if opts.multiscale {
            multi_buf.push((chrom.name.as_str(), ensemble));
        }
    }

    if opts.multiscale && !multi_buf.is_empty() {
        write_multiscale(&prefix, &multi_buf, res, params.top_k)?;
    }

    log::info!(
        " Called {total} domains across {n} chromosomes ({:.1?})",
        t0.elapsed()
    );
    log::info!("Output to {fout}");
    Ok(())
}

/// Build the dense symmetric matrix for `[first, last)`, optionally log the
/// positive counts, then run the multiscale sweep + consensus extraction.
fn armatus_for_chrom(
    cool: &Cooler,
    first: usize,
    last: usize,
    log_flag: bool,
    params: &armatus::Params,
) -> cooler_rs::Result<(Vec<armatus::Domain>, armatus::WeightedDomainEnsemble)> {
    let n = last - first;
    // Full symmetric dense matrix (Armatus sums over whole sub-matrices, so
    // no band is applied).
    let mut x = ndarray::Array2::<f64>::zeros((n, n));
    for p in cool.pixels_for_bins(first as i64, last as i64)? {
        let (b1, b2) = (p.bin1_id as usize, p.bin2_id as usize);
        if b2 < last {
            let v = p.count;
            x[[b1 - first, b2 - first]] = v;
            x[[b2 - first, b1 - first]] = v;
        }
    }
    if log_flag {
        for v in x.iter_mut() {
            if *v > 0.0 {
                *v = v.ln();
            }
        }
    }
    let ensemble = armatus::multiscale_domains(&x, params);
    let domains = armatus::consensus_domains(&ensemble);
    Ok((domains, ensemble))
}

fn write_domains_bed(
    path: &str,
    rows: &[(&str, &[armatus::Domain])],
    res: usize,
) -> cooler_rs::Result<()> {
    let mut out = std::fs::File::create(path)?;
    for (chrom, domains) in rows {
        for d in *domains {
            writeln!(
                out,
                "{}\t{}\t{}",
                chrom,
                d.start * res,
                (d.end + 1) * res - 1
            )?;
        }
    }
    Ok(())
}

/// Write per-(chrom, gamma, top-k) domain files. Filenames for single-chrom
/// runs match the original `{prefix}.gamma.{g}.{k}.txt`; multi-chrom runs
/// insert the chromosome between prefix and gamma to avoid clobbering across
/// chromosomes.
fn write_multiscale(
    prefix: &str,
    rows: &[(&str, armatus::WeightedDomainEnsemble)],
    res: usize,
    top_k: usize,
) -> cooler_rs::Result<()> {
    let multi = rows.len() > 1;
    for (chrom, ensemble) in rows {
        for (i, dset) in ensemble.domain_sets.iter().enumerate() {
            let gamma = ensemble.resolutions[i];
            let opt_idx = i % top_k;
            let path = if multi {
                format!("{prefix}.{chrom}.gamma.{gamma}.{opt_idx}.txt")
            } else {
                format!("{prefix}.gamma.{gamma}.{opt_idx}.txt")
            };
            write_domains_bed(&path, &[(chrom, dset)], res)?;
        }
    }
    Ok(())
}

fn run_ontad(common: &CommonArgs, opts: &OntadOptions) -> cooler_rs::Result<()> {
    let fin = common.fin();
    let params = opts.params();

    log::info!(
        "OnTAD v1.4 (Rust port): maxsz={}, minsz={}, penalty={:.3}, lsize={}, ldiff={}",
        params.maxsz,
        params.minsz,
        params.penalty,
        params.lsize,
        params.ldiff
    );

    let t0 = Instant::now();
    log::info!("Load {fin}:");

    let band = params.maxsz * 2;
    let cool = open_cooler_file(common, &fin)?;
    // Banded, mirrored dense matrix for the chromosome (see ontad module).
    let (mut x, file_meta) = ontad::matrix_from_cooler(&cool, common.chr.as_deref(), band)?;

    if common.log2 {
        for v in x.iter_mut() {
            *v = (*v + 1.0).log2();
        }
    }
    log::info!(" Done ({:.1?})", t0.elapsed());

    if opts.shuffle {
        log::info!("shuffling matrix");
        let l = x.nrows();
        let mut rng = rand::rng();
        for diag in 0..=params.maxsz.min(l.saturating_sub(1)) {
            for _ in 0..l * 10 {
                let i1 = rng.random_range(0..l - diag);
                let i2 = rng.random_range(0..l - diag);
                let tmp = x[[i1, i1 + diag]];
                x[[i1, i1 + diag]] = x[[i2, i2 + diag]];
                x[[i2, i2 + diag]] = tmp;
                let tmp = x[[i1 + diag, i1]];
                x[[i1 + diag, i1]] = x[[i2 + diag, i2]];
                x[[i2 + diag, i2]] = tmp;
            }
        }
    }

    let tad = ontad::call_tads(&mut x, &params);

    let prefix = common.prefix();
    let fout = format!("{prefix}.tad");
    ontad::write_tad(&fout, &tad)?;

    if opts.bedout {
        let foutbed = format!("{prefix}.bed");
        ontad::write_bed(&foutbed, &tad, &file_meta)?;
    }

    log::info!("Completed!");
    log::info!("Output to {fout}");
    log::info!("Total run time: {:.1?}", t0.elapsed());

    Ok(())
}

fn run_arrowhead(common: &CommonArgs, opts: &ArrowheadOptions) -> cooler_rs::Result<()> {
    let fin = common.fin();
    let params = opts.params();
    let norm = common.norm.as_deref();
    log::info!(
        "Arrowhead (Rust port of juicer): window={}, var={:?}, high_sign={}, min_block_size={}, norm={:?}, threads={}",
        params.matrix_width,
        params.var_threshold,
        params.high_sign_threshold,
        params.min_block_size,
        norm,
        common.threads
    );
    let t0 = Instant::now();

    let res = resolve_arrowhead_resolution(common, &fin)?;
    let f = File::open(&fin, res)?;
    let chroms: Option<Vec<String>> = common.chr.clone().map(|c| vec![c]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(common.threads)
        .build()
        .map_err(|e| {
            Error::InvalidInput(format!(
                "cannot build thread pool with {} threads: {e}",
                common.threads
            ))
        })?;
    let domains = pool.install(|| arrowhead::call_domains(&f, norm, &params, chroms.as_deref()))?;

    let prefix = common.prefix();
    let fout = format!("{prefix}.arrowhead.bedpe");
    let mut out = std::fs::File::create(&fout)?;
    for d in &domains {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            d.chrom,
            d.start,
            d.end,
            d.chrom,
            d.start,
            d.end,
            d.score,
            d.up_var,
            d.lo_var,
            d.up_sign,
            d.lo_sign
        )?;
    }

    log::info!("Called {} domains ({:.1?})", domains.len(), t0.elapsed());
    log::info!("Output to {fout}");
    Ok(())
}

fn resolve_arrowhead_resolution(common: &CommonArgs, fin: &str) -> cooler_rs::Result<u32> {
    if let Some(r) = common.res {
        return Ok(r as u32);
    }
    if fin.ends_with(".hic") {
        return Err(Error::InvalidInput(
            "arrowhead on .hic input needs --res (bp resolution)".into(),
        ));
    }
    if fin.ends_with(".mcool") {
        let mcool = Mcool::open(fin)?;
        return match mcool.resolutions()?.as_slice() {
            [only] => Ok(*only as u32),
            _ => Err(Error::InvalidInput(
                ".mcool contains multiple resolutions; select one with --res".into(),
            )),
        };
    }
    Ok(Cooler::open_any(fin)?
        .bin_size()?
        .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))? as u32)
}
