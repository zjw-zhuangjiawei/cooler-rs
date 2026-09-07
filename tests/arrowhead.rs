//! Arrowhead integration smoke tests. The `.hic`/`.mcool` fixtures are
//! gitignored; each test is skipped when its file is absent.

use std::path::Path;

use cooler_rs::{arrowhead, File, Mcool};

fn fixture(name: &str) -> Option<std::path::PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    p.exists().then_some(p)
}

fn assert_well_formed(domains: &[arrowhead::Domain]) {
    for d in domains {
        assert!(
            d.end > d.start,
            "domain {}:{} has end <= start",
            d.chrom,
            d.start
        );
        assert!(d.score.is_finite());
        assert!(d.up_var.is_finite() && d.lo_var.is_finite());
    }
}

/// A reduced window so the debug-mode smoke test stays fast; the full pipeline
/// (DI -> block score -> components -> two-pass merge -> binning) still runs.
fn small_params() -> arrowhead::Params {
    arrowhead::Params {
        matrix_width: 400,
        min_block_size: 20,
        ..Default::default()
    }
}

#[test]
fn finds_domains_on_hic_kr() {
    let Some(path) = fixture("4DNFIOTPSS3L.hic") else {
        eprintln!("skipping: 4DNFIOTPSS3L.hic not present");
        return;
    };
    let f = File::open(path.to_str().unwrap(), 5000).unwrap();
    let domains = arrowhead::call_chrom(&f, "2L", Some("KR"), &small_params()).unwrap();
    assert_well_formed(&domains);
    assert!(!domains.is_empty(), "expected domains on 2L with KR");
}

#[test]
fn runs_on_mcool_raw() {
    let Some(path) = fixture("4DNFIZ1ZVXC8.mcool") else {
        eprintln!("skipping: 4DNFIZ1ZVXC8.mcool not present");
        return;
    };
    let mcool = Mcool::open(path.to_str().unwrap()).unwrap();
    let res = *mcool.resolutions().unwrap().iter().max().unwrap(); // coarsest -> fewest windows
    let f = File::open(path.to_str().unwrap(), res as u32).unwrap();
    let chrom = f.chroms().unwrap()[0].name.clone();
    let domains = arrowhead::call_chrom(&f, &chrom, None, &small_params()).unwrap();
    assert_well_formed(&domains);
}
