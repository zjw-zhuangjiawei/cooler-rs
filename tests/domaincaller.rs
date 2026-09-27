//! End-to-end regression: the Rust `domaincaller` port must reproduce
//! TADLib's whole-genome CLI pipeline (the `domaincaller.Genome` flow behind
//! `scripts/domaincaller`) on the 4DNFIZ1ZVXC8 fixture.
//!
//! The pipeline is: load each non-excluded chromosome as a `Chrom` from the
//! balance-corrected upper-triangle matrix (Genome::new), train **one shared
//! 4-state GMM-HMM** on the non-zero DI segments of the whole genome
//! (Genome::learning), decode every chromosome with that model
//! (Genome::call_domains) and emit the CLI DI track (Genome::di_track).
//!
//! ## Expected files
//!
//! `data/4DNFIZ1ZVXC8.50kb.tadlib.dis` is the DI bedGraph
//! `chrom start end DI` (`%.4g` precision, whole genome, 2653 rows) and
//! `data/4DNFIZ1ZVXC8.50kb.tadlib.domains` is the merged domain list
//! `chrom start end` in base pairs (263 domains), noise > 0.5 dropped. Both
//! were produced by running TADLib 0.4.4 + pomegranate 0.10.0 (Docker image
//! `cooler-rs/tadlib-fixture:0.4.4`, see `scripts/tadlib-fixture/`) on
//! `data/4DNFIZ1ZVXC8.mcool` at 50 kb, excluding `chr4`/`chrY`/`chrM`
//! (TADLib's domaincaller uses a fixed 40-bin window and crashes on the
//! 27-bin chr4). The Rust port must match the domains exactly and the DI
//! track within the `%.4g` rounding on top of float slack.

use cooler_rs::domaincaller::Genome;
use cooler_rs::Mcool;

const MCOOL: &str = "tests/data/4DNFIZ1ZVXC8.mcool";
const RES: u64 = 50_000;
const BALANCE: Option<&str> = Some("weight");
const EXCLUDE: &[&str] = &["chr4", "chrY", "chrM"];

const EXPECTED_DIS: &str = include_str!("data/4DNFIZ1ZVXC8.50kb.tadlib.dis");
const EXPECTED_DOMAINS: &str = include_str!("data/4DNFIZ1ZVXC8.50kb.tadlib.domains");

fn run_genome() -> (Genome, Vec<(String, u64, u64)>) {
    let mcool = Mcool::open(MCOOL).unwrap();
    let cool = mcool.cooler(RES).unwrap();
    let mut g = Genome::new(&cool, RES, BALANCE, EXCLUDE).unwrap();
    g.learning();
    let domains = g.call_domains();
    (g, domains)
}

#[test]
fn whole_genome_domains_match_tadlib() {
    let (_g, got) = run_genome();
    let expected: Vec<(String, u64, u64)> = EXPECTED_DOMAINS
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (
                f[0].to_string(),
                f[1].parse().expect("bad expected domain start"),
                f[2].parse().expect("bad expected domain end"),
            )
        })
        .collect();
    assert_eq!(
        got.len(),
        expected.len(),
        "domain count: got {} != {}",
        got.len(),
        expected.len()
    );
    for (i, ((gc, gs, ge), (ec, es, ee))) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(gc, ec, "domain {i} chrom: Rust {gc} != TADLib {ec}");
        assert_eq!(gs, es, "domain {i} start: Rust {gs} != TADLib {es}");
        assert_eq!(ge, ee, "domain {i} end: Rust {ge} != TADLib {ee}");
    }
}

#[test]
fn di_track_matches_tadlib() {
    let (g, _) = run_genome();
    let got = g.di_track();
    let expected: Vec<(String, u64, u64, f64)> = EXPECTED_DIS
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (
                f[0].to_string(),
                f[1].parse().expect("bad expected start"),
                f[2].parse().expect("bad expected end"),
                f[3].parse().expect("bad expected DI"),
            )
        })
        .collect();
    assert_eq!(
        got.len(),
        expected.len(),
        "DI rows: got {} != {}",
        got.len(),
        expected.len()
    );
    for (i, ((gc, gs, ge, gv), (ec, es, ee, ev))) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(gc, ec, "DI row {i} chrom");
        assert_eq!(gs, es, "DI row {i} start");
        assert_eq!(ge, ee, "DI row {i} end");
        // TADLib writes `%.4g` (4 significant digits): the reference value
        // carries up to ~5e-4 relative rounding on top of float slack.
        let tol = 6e-4 * gv.abs().max(ev.abs()).max(1e-6);
        assert!(
            (gv - ev).abs() <= tol,
            "DI[{i}] = {gv} != TADLib {ev} (diff {})",
            (gv - ev).abs()
        );
    }
}
