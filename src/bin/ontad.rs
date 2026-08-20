//! `ontad` — call hierarchical TADs from a Hi-C contact matrix.
//!
//! Rust port of OnTAD v1.4 (An et al., Genome Biology 2019). Input must be a
//! `.cool` or `.mcool` file. (Use the `mat2cool` tool to convert a dense
//! N×N text matrix to `.cool` first.)
//!
//! Usage:
//!   ontad <input.cool|input.mcool> [options]
//!
//! Options:
//!   -penalty <f>   penalty for adding a TAD (default 0.1)
//!   -maxsz <n>     maximum TAD size in bins (default 200)
//!   -minsz <n>     minimum TAD size in bins (default 3)
//!   -lsize <n>     local-minimum window half-size (default 5)
//!   -ldiff <f>     local-minimum threshold in stddevs (default 1.96)
//!   -log2          apply log2(x + 1) to the matrix
//!   -shuffle       shuffle each diagonal (null model)
//!   -o <prefix>    output prefix (default: input file name)
//!   -bedout <chr> <chrlength> <resolution>
//!                  also write a .bed file (overrides file metadata)
//!   -chr <name>    chromosome to extract (.cool/.mcool input)
//!   -res <n>       resolution to use (.mcool input)

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use cooler::ontad::{self, Params};
use cooler::{Cooler, ChromMeta, Mcool};

/// Known flags from the original OnTAD CLI (single-dash prefix).
/// These are rewritten to GNU-style `--flag` for clap compatibility.
const KNOWN_FLAGS: &[&str] = &[
    "penalty", "maxsz", "minsz", "lsize", "ldiff",
    "log2", "shuffle", "bedout", "chr", "res",
];

/// Rewrite single-dash long options (`-penalty`) to double-dash (`--penalty`)
/// so clap can parse them. Also handles `-o` and `-chr`/`-res` (Rust additions).
fn normalize_args() -> Vec<String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let arg = &raw[i];
        if arg.starts_with("--") {
            out.push(arg.clone());
            i += 1;
            continue;
        }
        if arg.starts_with('-') && !arg.starts_with("--") {
            let flag = &arg[1..]; // strip leading '-'
            // Check if this matches a known long flag name.
            if KNOWN_FLAGS.contains(&flag) {
                out.push(format!("--{flag}"));
                i += 1;
                continue;
            }
            // Handle remaining single-dash: pass through as-is (clap handles -o, -b, -h, -V).
        }
        out.push(arg.clone());
        i += 1;
    }
    out
}

#[derive(Parser)]
#[command(
    name = "ontad",
    about = "OnTAD v1.4: hierarchical TAD calling from Hi-C contact matrices (Rust port)",
    version
)]
struct Args {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Penalty for adding a TAD
    #[arg(long = "penalty", default_value = "0.1", value_name = "F")]
    penalty: f64,

    /// Maximum TAD size in bins
    #[arg(long = "maxsz", default_value = "200", value_name = "N")]
    maxsz: usize,

    /// Minimum TAD size in bins
    #[arg(long = "minsz", default_value = "3", value_name = "N")]
    minsz: usize,

    /// Local-minimum window half-size
    #[arg(long = "lsize", default_value = "5", value_name = "N")]
    lsize: usize,

    /// Local-minimum threshold in stddevs
    #[arg(long = "ldiff", default_value = "1.96", value_name = "F")]
    ldiff: f64,

    /// Apply log2(x + 1) to the matrix
    #[arg(long = "log2")]
    log2: bool,

    /// Shuffle each diagonal (null model)
    #[arg(long = "shuffle")]
    shuffle: bool,

    /// Output prefix (default: input file name)
    #[arg(short = 'o', long, value_name = "PREFIX")]
    output: Option<String>,

    /// Also write a .bed file: <CHR> <CHRLENGTH> <RESOLUTION>
    #[arg(
        short = 'b',
        long = "bedout",
        num_args = 3,
        value_names = ["CHR", "CHRLENGTH", "RESOLUTION"]
    )]
    bedout: Option<Vec<String>>,

    /// Chromosome to extract (.cool/.mcool input)
    #[arg(long = "chr", value_name = "NAME")]
    chr: Option<String>,

    /// Resolution to use (.mcool input)
    #[arg(long = "res", value_name = "N")]
    res: Option<u64>,
}

/// Tiny SplitMix64 PRNG for -shuffle (avoids a rand dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse_from(normalize_args());
    let fin = args.input.display().to_string();

    if fin.ends_with(".hic") {
        log::error!(".hic input is not supported in the Rust port; convert to .cool first");
        std::process::exit(1);
    }

    let params = Params {
        maxsz: args.maxsz.max(10),
        minsz: args.minsz.max(1),
        penalty: args.penalty,
        lsize: args.lsize,
        ldiff: args.ldiff,
    };

    log::info!(
        "OnTAD v1.4 (Rust port): maxsz={}, minsz={}, penalty={:.3}, lsize={}, ldiff={}",
        params.maxsz, params.minsz, params.penalty, params.lsize, params.ldiff
    );

    let t0 = Instant::now();
    log::info!("Load {fin}:");

    // Parse -bedout metadata (always 3 values when provided).
    let bed_meta: Option<ChromMeta> = args.bedout.as_ref().map(|v| {
        let length = v[1].parse::<u64>().unwrap_or_else(|_| {
            log::error!("invalid chrlength in -bedout: {}", v[1]);
            std::process::exit(1);
        });
        let resolution = v[2].parse::<u64>().unwrap_or_else(|_| {
            log::error!("invalid resolution in -bedout: {}", v[2]);
            std::process::exit(1);
        });
        ChromMeta {
            name: v[0].clone(),
            length,
            resolution,
        }
    });

    let band = params.maxsz * 2;
    let file_meta: ChromMeta;
    let mut x = if fin.ends_with(".cool") {
        let cool = Cooler::open_any(&fin).unwrap_or_else(|e| {
            log::error!("{e}");
            std::process::exit(1);
        });
        let (x, meta) = ontad::matrix_from_cooler(&cool, args.chr.as_deref(), band)
            .unwrap_or_else(|e| {
                log::error!("{e}");
                std::process::exit(1);
            });
        file_meta = meta;
        x
    } else if fin.ends_with(".mcool") {
        let mcool = Mcool::open(&fin).unwrap_or_else(|e| {
            log::error!("{e}");
            std::process::exit(1);
        });
        let resolutions = mcool.resolutions().unwrap_or_else(|e| {
            log::error!("{e}");
            std::process::exit(1);
        });
        let res = match (args.res, resolutions.as_slice()) {
            (Some(r), _) => r,
            (None, [only]) => *only,
            (None, _) => {
                log::error!(
                    ".mcool contains multiple resolutions; select one with --res ({resolutions:?})"
                );
                std::process::exit(1);
            }
        };
        let cool = mcool.cooler(res).unwrap_or_else(|e| {
            log::error!("{e}");
            std::process::exit(1);
        });
        let (x, meta) = ontad::matrix_from_cooler(&cool, args.chr.as_deref(), band)
            .unwrap_or_else(|e| {
                log::error!("{e}");
                std::process::exit(1);
            });
        file_meta = meta;
        x
    } else {
        log::error!(
            "input must be a .cool or .mcool file (use 'mat2cool' to convert dense text matrices)"
        );
        std::process::exit(1);
    };

    if args.log2 {
        for v in x.iter_mut() {
            *v = (*v + 1.0).log2();
        }
    }
    log::info!(" Done ({:.1?})", t0.elapsed());

    if args.shuffle {
        log::info!("shuffling matrix");
        let l = x.nrows();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
            .unwrap_or(0x1234_5678_9abc_def0);
        let mut rng = Rng(nanos);
        for diag in 0..=params.maxsz.min(l.saturating_sub(1)) {
            for _ in 0..l * 10 {
                let i1 = rng.below(l - diag);
                let i2 = rng.below(l - diag);
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

    let prefix = args.output.as_deref().unwrap_or(&fin);
    let fout = format!("{prefix}.tad");
    ontad::write_tad(&fout, &tad).unwrap_or_else(|e| {
        log::error!("{e}");
        std::process::exit(1);
    });

    if let Some(meta) = bed_meta.or(Some(file_meta)) {
        let foutbed = format!("{prefix}.bed");
        ontad::write_bed(&foutbed, &tad, &meta).unwrap_or_else(|e| {
            log::error!("{e}");
            std::process::exit(1);
        });
    }

    log::info!("Completed!");
    log::info!("Output to {fout}");
    log::info!("Total run time: {:.1?}", t0.elapsed());
}
