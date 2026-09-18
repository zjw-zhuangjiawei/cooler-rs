//! Identity test for the `hicFindTADs` port.
//!
//! Every output file must match HiCExplorer's, byte for byte, on the matrix
//! upstream tests itself with (`hicexplorer/test/test_data/small_test_matrix.h5`,
//! *Drosophila melanogaster* at 5 kb). The `.cool` under `tests/data/findtads`
//! was converted from that `.h5` without touching the values, so a mismatch
//! is a port bug rather than a fixture difference.
//!
//! The reference files under `tests/data/findtads/<case>/` were produced by
//! running `hicFindTADs.main` end to end on that matrix:
//!
//! ```text
//! --minDepth 60000 --maxDepth 180000 --step 20000 --minBoundaryDistance 20000
//!   fdr / bonferroni: --thresholdComparisons 0.1
//!   none:             --thresholdComparisons 1.0
//!   fdr_chromosomes:  --thresholdComparisons 0.5 --chromosomes chr2L chr3R
//! ```
//!
//! Two of the files upstream ships (`find_TADs/bonferroni`, `find_TADs/None`)
//! are not usable as references: those tests feed a pre-computed
//! `_tad_score.bm` back in, and `load_bedgraph_matrix` reads the six-decimal
//! text, so their scores — and nine of their boundaries — differ from what the
//! same parameters produce in one pass.

use std::path::{Path, PathBuf};

use cooler_rs::findtads::{self, MultipleTesting, Params};
use cooler_rs::Cooler;

/// The five text outputs, in the order they are compared.
const OUTPUTS: [&str; 5] = [
    "tad_score.bm",
    "boundaries.bed",
    "boundaries.gff",
    "domains.bed",
    "score.bedgraph",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/findtads")
}

/// Run one case and compare every output with its reference.
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
        ..Default::default()
    };

    let cooler = Cooler::open_any(root().join("small_test_matrix.cool")).expect("open fixture");
    let outputs = findtads::run_cooler(&cooler, &params).expect("run find-tads");

    // The z-score matrix is the one output this port writes as a cooler
    // rather than a `HiCMatrix` `.h5`; read it back so its bin table and
    // pixels are exercised too.
    let written = Cooler::open_any(&outputs.zscore_matrix).expect("reopen z-score matrix");
    let chroms = cooler.chroms().expect("chroms");
    let selected: Vec<usize> = match &params.chromosomes {
        None => (0..chroms.len()).collect(),
        Some(names) => names
            .iter()
            .map(|name| chroms.iter().position(|c| &c.name == name).expect("chrom"))
            .collect(),
    };
    let expected = cooler
        .bins()
        .expect("bins")
        .iter()
        .filter(|b| selected.contains(&(b.chrom_id as usize)))
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
        let want = std::fs::read_to_string(root().join(case).join(format!("{case}_{suffix}")))
            .unwrap_or_else(|e| panic!("reference {case}_{suffix}: {e}"));
        assert_eq!(got, want, "{case}_{suffix} differs from HiCExplorer");
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
        "fdr_chromosomes",
        MultipleTesting::Fdr,
        0.5,
        Some(vec!["chr2L".into(), "chr3R".into()]),
    );
}

/// The flag surface has to survive clap's own consistency check: two option
/// groups sharing an argument id panic at runtime rather than at build time,
/// and only a real parse catches it.
#[test]
fn call_tad_flags_parse() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
        .args(["call-tad", "--help"])
        .output()
        .expect("run cooler-rs");
    assert!(out.status.success(), "call-tad --help failed");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(help.contains("hicexplorer"), "missing method: {help}");
    assert!(help.contains("--window-step"), "missing flag: {help}");
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
