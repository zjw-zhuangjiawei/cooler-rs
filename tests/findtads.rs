//! Regression test: with default parameters (plus KR-normalization-style
//! weight handling, see below), the Rust find-tads port must reproduce
//! HiCExplorer's `hicFindTADs` output on the 4DNFIZ1ZVXC8 fixture.
//!
//! `tests/data/4DNFIZ1ZVXC8.mcool` (gitignored; regenerate with the other
//! fixture scripts if absent) carries the 5 kb layer the references were made
//! from; `scripts/generate_findtads_fixture.sh` reruns HiCExplorer 3.7.6 (the
//! version the port is aligned to, installed the way its docs recommend:
//! conda via micromamba) on it and writes the reference files
//! `tests/data/4DNFIZ1ZVXC8.5kb.findtads.<case>_*`.
//!
//! hicFindTADs applies the cooler `weight` column (multiplicative) by default
//! through hicmatrix's reader and masks NaN-weight bins as `nan_bins`, so the
//! port is run with `--norm weight`, which does the same.
//!
//! The port is byte-for-byte identical to the reference outputs, which took
//! replicating several numeric quirks of the original pipeline: duplicate
//! pixels are merged by the sparse-matrix construction before the weights are
//! applied, the `+= diag_mat_ones` / `data -= 1` dense-banding round trip is
//! not exact in floating point, and the pool statistics accumulate in the
//! row-major COO order (`np.bincount`). See `src/findtads/matrix.rs`.

use std::io::Read;
use std::path::{Path, PathBuf};

use cooler_rs::findtads::{self, MultipleTesting, Params};
use cooler_rs::Mcool;
use flate2::read::GzDecoder;

const MCOOL: &str = "tests/data/4DNFIZ1ZVXC8.mcool";
const RES: u64 = 5_000;

/// The five text outputs, in the order they are compared.
const OUTPUTS: [&str; 5] = [
    "tad_score.bm",
    "boundaries.bed",
    "boundaries.gff",
    "domains.bed",
    "score.bedgraph",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn assert_case(
    case: &str,
    correction: MultipleTesting,
    threshold: f64,
    chromosomes: Option<Vec<String>>,
) {
    let dir = tempfile::tempdir().expect("temp dir");
    let prefix = dir.path().join(case);
    let params = Params {
        min_depth: Some(60_000),
        max_depth: Some(180_000),
        step: Some(20_000),
        min_boundary_distance: Some(20_000),
        correction,
        threshold_comparisons: threshold,
        chromosomes,
        out_prefix: prefix.to_string_lossy().into_owned(),
        // hicFindTADs applies the cooler weight column by default
        // (hicmatrix's `correctionFactorTable = 'weight'`, multiplicative).
        norm: Some("weight".into()),
        ..Default::default()
    };

    let mcool = Mcool::open(MCOOL).unwrap_or_else(|e| {
        panic!("open {MCOOL} ({e}); the 5 kb references were regenerated from it (scripts/generate_findtads_fixture.sh)")
    });
    let cooler = mcool.cooler(RES).unwrap();
    let outputs = findtads::run_cooler(&cooler, &params).expect("run find-tads");

    // The z-score matrix is the one output this port writes as a cooler
    // rather than a `HiCMatrix` `.h5`; read it back so its bin table and
    // pixels are exercised too.
    let written =
        cooler_rs::Cooler::open_any(&outputs.zscore_matrix).expect("reopen z-score matrix");
    let chroms = cooler.chroms().expect("chroms");
    let selected: Vec<usize> = match &params.chromosomes {
        None => (0..chroms.len()).collect(),
        Some(names) => names
            .iter()
            .map(|name| chroms.iter().position(|c| &c.name == name).expect("chrom"))
            .collect(),
    };
    // Bins with a NaN weight are masked as `nan_bins`, so the z-score
    // matrix keeps only the weight-finite bins of the selected chromosomes.
    let weights = cooler
        .bins_column_f64("weight")
        .expect("weight column")
        .expect("bins/weight column");
    let expected = cooler
        .bins()
        .expect("bins")
        .iter()
        .zip(&weights)
        .filter(|(b, w)| selected.contains(&(b.chrom_id as usize)) && w.is_finite())
        .count();
    assert_eq!(
        written.bins().expect("bins").len(),
        expected,
        "{case}: z-score matrix has the wrong bin table"
    );
    assert!(written.n_pixels().expect("pixels") > 0, "{case}: empty");

    for suffix in OUTPUTS {
        let got = std::fs::read_to_string(dir.path().join(format!("{case}_{suffix}")))
            .unwrap_or_else(|e| panic!("{case}_{suffix}: {e}"));
        let want = reference(case, suffix);
        assert_eq!(got, want, "{case}_{suffix} differs from HiCExplorer");
    }
}

fn gunzip(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut s = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut s)
        .expect("fixture gunzip");
    s
}

/// Read one reference output.
///
/// The correction only enters after scoring, so the three full-genome cases
/// share one `tad_score.bm` / `score.bedgraph` pair (stored under `full_`),
/// while the chromosome-subset case has its own. The big files are gzipped
/// to stay under the repository's 500 KB fixture limit, and the shared
/// `tad_score.bm` is split in two parts at a line boundary.
fn reference(case: &str, suffix: &str) -> String {
    let read = |name: &str| {
        std::fs::read_to_string(root().join(name))
            .unwrap_or_else(|e| panic!("reference {name}: {e}"))
    };
    match suffix {
        "boundaries.bed" | "boundaries.gff" | "domains.bed" => {
            read(&format!("4DNFIZ1ZVXC8.5kb.findtads.{case}_{suffix}"))
        }
        "score.bedgraph" => {
            if case == "fdrchr" {
                read("4DNFIZ1ZVXC8.5kb.findtads.fdrchr_score.bedgraph")
            } else {
                gunzip(&root().join("4DNFIZ1ZVXC8.5kb.findtads.full_score.bedgraph.gz"))
            }
        }
        "tad_score.bm" => {
            if case == "fdrchr" {
                gunzip(&root().join("4DNFIZ1ZVXC8.5kb.findtads.fdrchr_tad_score.bm.gz"))
            } else {
                gunzip(&root().join("4DNFIZ1ZVXC8.5kb.findtads.full_tad_score.bm.part0.gz"))
                    + &gunzip(&root().join("4DNFIZ1ZVXC8.5kb.findtads.full_tad_score.bm.part1.gz"))
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn fdr_matches_hicexplorer() {
    assert_case("fdr", MultipleTesting::Fdr, 0.1, None);
}

#[test]
fn bonferroni_matches_hicexplorer() {
    assert_case("bonferroni", MultipleTesting::Bonferroni, 0.1, None);
}

#[test]
fn no_correction_matches_hicexplorer() {
    assert_case("none", MultipleTesting::None, 1.0, None);
}

#[test]
fn chromosome_subset_matches_hicexplorer() {
    assert_case(
        "fdrchr",
        MultipleTesting::Fdr,
        0.5,
        Some(vec!["chr2L".into(), "chr3R".into()]),
    );
}

/// The flag surface has to survive clap's own consistency check: two
/// arguments sharing an id panic at runtime rather than at build time, and
/// only a real parse catches it. `--norm` and `--step` are what would collide
/// if the methods shared one command instead of a subcommand each.
#[test]
fn call_tad_flags_parse() {
    let help = |args: &[&str]| {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
            .args(args)
            .output()
            .expect("run cooler-rs");
        assert!(out.status.success(), "{args:?} failed");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    assert!(help(&["call-tad", "--help"]).contains("hicexplorer"));
    let hx = help(&["call-tad", "hicexplorer", "--help"]);
    assert!(hx.contains("--step"), "{hx}");
    assert!(hx.contains("--norm"), "{hx}");
}

/// The window sizes are the step function the whole scoring stage is built
/// on; the values are the ones the `--step 20000 --minDepth 60000
/// --maxDepth 180000` invocation logs.
#[test]
fn window_sizes_grow_by_three_halves_power() {
    assert_eq!(
        findtads::incremental_step_size(60_000, 180_000, 20_000),
        vec![60_000, 80_000, 116_568, 163_923]
    );
}
