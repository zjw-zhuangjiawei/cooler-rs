//! End-to-end regression: the Rust `hitad` port must reproduce TADLib's
//! whole-genome hierarchical domain caller (the `hitad.Genome` flow behind
//! `scripts/hitad`) on the 4DNFIZ1ZVXC8 fixture.
//!
//! The pipeline is: load each non-excluded chromosome as a `Chrom` from the
//! balance-corrected upper-triangle matrix (Genome::new), train **one
//! shared HMM** per dataset on the non-zero DI segments of the whole genome
//! (Genome::learning), run the hierarchical per-chromosome decode
//! (Genome::call_domains: oriIter + maxCore + fineDomain + preciseBound,
//! then the MultiReps identity path for a single replicate) and expose the
//! DI tracks (Genome::di_track).
//!
//! ## Expected files
//!
//! `data/4DNFIZ1ZVXC8.50kb.hitad.domains` is the merged domain list
//! `chrom start end level` (312 domains, levels 0..3) and
//! `data/4DNFIZ1ZVXC8.50kb.hitad.dis` is the DI bedGraph the CLI writes into
//! the `bins/DIs` column (2754 rows — all chromosomes, zeros for the
//! excluded ones), dumped as `%.4g`. Both were produced by TADLib 0.4.4 +
//! pomegranate 0.10.0 (Docker image `cooler-rs/tadlib-fixture:0.4.4`, see
//! `scripts/tadlib-fixture/`) on `data/4DNFIZ1ZVXC8.mcool` at 50 kb with the
//! CLI defaults (`--maxsize 4000000`, `--minimum-chrom-size 1000000`) and
//! `exclude = [chr4, chrY, chrM]`.
//!
//! chr4 (27 bins) must be excluded as well: TADLib 0.4.4 + pomegranate
//! 0.10 `viterbi` decodes in probability space and underflows with the HMM
//! trained on chr4's DI segments — every chromosome then fails and the
//! multiprocessing queue deadlocks (the Rust port decodes in log space and
//! would survive, but the reference has to match TADLib's own output).

use cooler_rs::hitad::{Dataset, Genome};

const MCOOL: &str = "tests/data/4DNFIZ1ZVXC8.mcool";
const RES: u64 = 50_000;
const EXCLUDE: &[&str] = &["chr4", "chrY", "chrM"];

const EXPECTED_DOMAINS: &str = include_str!("data/4DNFIZ1ZVXC8.50kb.hitad.domains");
const EXPECTED_DIS: &str = include_str!("data/4DNFIZ1ZVXC8.50kb.hitad.dis");

fn run_genome() -> Genome {
    let datasets: Vec<Dataset> = vec![(RES, "rep1".to_string(), MCOOL.to_string())];
    let mut g =
        Genome::new(&datasets, Some("weight"), EXCLUDE, 1_000_000, 4_000_000).expect("Genome::new");
    g.learning();
    g.call_domains();
    g
}

#[test]
fn hierarchical_domains_match_tadlib() {
    let g = run_genome();
    let expected: Vec<(String, usize, usize, usize)> = EXPECTED_DOMAINS
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (
                f[0].to_string(),
                f[1].parse().expect("bad expected domain start"),
                f[2].parse().expect("bad expected domain end"),
                f[3].parse().expect("bad expected domain level"),
            )
        })
        .collect();
    assert_eq!(
        g.results.len(),
        expected.len(),
        "domain count: got {} != {}",
        g.results.len(),
        expected.len()
    );
    for (i, (got, exp)) in g.results.iter().zip(expected.iter()).enumerate() {
        let (gc, gs, ge, gl) = got;
        let (ec, es, ee, el) = exp;
        assert_eq!(gc, ec, "domain {i} chrom: Rust {gc} != TADLib {ec}");
        assert_eq!(gs, es, "domain {i} start: Rust {gs} != TADLib {es}");
        assert_eq!(ge, ee, "domain {i} end: Rust {ge} != TADLib {ee}");
        assert_eq!(gl, el, "domain {i} level: Rust {gl} != TADLib {el}");
    }
}

#[test]
fn di_track_matches_tadlib() {
    let g = run_genome();
    let got = g.di_track(RES, "rep1");
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
    for (i, (got, exp)) in got.iter().zip(expected.iter()).enumerate() {
        let (gc, gs, ge, gv) = got;
        let (ec, es, ee, ev) = exp;
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
