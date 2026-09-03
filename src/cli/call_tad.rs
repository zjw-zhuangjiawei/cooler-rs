//! `cooler-rs call-tad` — call hierarchical TADs from a Hi-C contact matrix.
//!
//! The `--method` flag selects the TAD calling algorithm. Each method has its
//! own option group (flattened into the help output under a per-method
//! heading); method-specific fields are `Option<T>` so that defaults are
//! resolved in `run()` and explicitly-set options can be validated against
//! the selected method.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};
use rand::Rng;

use cooler_rs::domaincaller::Chrom;
use cooler_rs::ontad::{self, Params};
use cooler_rs::{ChromMeta, Cooler, Error, Mcool};

/// TAD calling method.
#[derive(Clone, Copy, ValueEnum)]
pub enum TadMethod {
    /// OnTAD v1.4 (An et al., Genome Biology 2019; Rust port)
    Ontad,
    /// DomainCaller (Dixon et al., Nature 2012; Rust port of TADLib)
    Domaincaller,
}

#[derive(Args)]
pub struct CallTadArgs {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// TAD calling method
    #[arg(long, value_enum, value_name = "METHOD", default_value = "ontad")]
    method: TadMethod,

    /// Output prefix (default: input file name)
    #[arg(short = 'o', long, value_name = "PREFIX")]
    output: Option<String>,

    /// Chromosome to extract
    #[arg(long = "chr", value_name = "NAME")]
    chr: Option<String>,

    /// Resolution to use (.mcool input)
    #[arg(long = "res", value_name = "N")]
    res: Option<u64>,

    /// Apply log2(x + 1) to the matrix
    #[arg(long = "log2")]
    log2: bool,

    #[command(flatten)]
    ontad: OntadOptions,
}

/// Options specific to `--method ontad`.
#[derive(Args)]
struct OntadOptions {
    /// Penalty for adding a TAD
    #[arg(long, value_name = "F", help_heading = "OnTAD options")]
    penalty: Option<f64>,

    /// Maximum TAD size in bins
    #[arg(long, value_name = "N", help_heading = "OnTAD options")]
    maxsz: Option<usize>,

    /// Minimum TAD size in bins
    #[arg(long, value_name = "N", help_heading = "OnTAD options")]
    minsz: Option<usize>,

    /// Local-minimum window half-size
    #[arg(long, value_name = "N", help_heading = "OnTAD options")]
    lsize: Option<usize>,

    /// Local-minimum threshold in stddevs
    #[arg(long, value_name = "F", help_heading = "OnTAD options")]
    ldiff: Option<f64>,

    /// Shuffle each diagonal (null model)
    #[arg(long, help_heading = "OnTAD options")]
    shuffle: bool,

    /// Also write a .bed file
    #[arg(long, help_heading = "OnTAD options")]
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

pub fn run(args: CallTadArgs) -> cooler_rs::Result<()> {
    let fin = args.input.display().to_string();

    if fin.ends_with(".hic") {
        return Err(Error::InvalidInput(
            ".hic input is not supported; convert to .cool first".into(),
        ));
    }

    match args.method {
        TadMethod::Ontad => run_ontad(&args, &fin),
        TadMethod::Domaincaller => run_domaincaller(&args, &fin),
    }
}

/// Resolve the input into a `Cooler` at the requested resolution, without
/// selecting a chromosome.
fn open_cooler_file(args: &CallTadArgs, fin: &str) -> cooler_rs::Result<Cooler> {
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
             (use 'cooler-rs convert --from dense-txt' to convert dense text matrices)"
                .into(),
        ))
    }
}

/// Resolve the input into a `Cooler` and the selected chromosome's first/last
/// bin offsets (shared by the per-method runners).
fn open_cooler(
    args: &CallTadArgs,
    fin: &str,
) -> cooler_rs::Result<(Cooler, usize, usize, ChromMeta)> {
    let cool = open_cooler_file(args, fin)?;

    let chroms = cool.chroms()?;
    let chrom_id = match args.chr.as_deref() {
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

fn run_domaincaller(args: &CallTadArgs, fin: &str) -> cooler_rs::Result<()> {
    log::info!("DomainCaller (Rust port of TADLib)");
    let t0 = Instant::now();

    let (cool, first, last, meta) = open_cooler(args, fin)?;
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

    let prefix = args.output.as_deref().unwrap_or(fin);
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

fn run_ontad(args: &CallTadArgs, fin: &str) -> cooler_rs::Result<()> {
    let params = args.ontad.params();

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
    let cool = open_cooler_file(args, fin)?;
    // Banded, mirrored dense matrix for the chromosome (see ontad module).
    let (mut x, file_meta) = ontad::matrix_from_cooler(&cool, args.chr.as_deref(), band)?;

    if args.log2 {
        for v in x.iter_mut() {
            *v = (*v + 1.0).log2();
        }
    }
    log::info!(" Done ({:.1?})", t0.elapsed());

    if args.ontad.shuffle {
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

    let prefix = args.output.as_deref().unwrap_or(fin);
    let fout = format!("{prefix}.tad");
    ontad::write_tad(&fout, &tad)?;

    if args.ontad.bedout {
        let foutbed = format!("{prefix}.bed");
        ontad::write_bed(&foutbed, &tad, &file_meta)?;
    }

    log::info!("Completed!");
    log::info!("Output to {fout}");
    log::info!("Total run time: {:.1?}", t0.elapsed());

    Ok(())
}
