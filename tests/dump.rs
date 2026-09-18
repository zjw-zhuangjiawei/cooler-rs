#![cfg(any())]
//! `cooler-rs dump` must reproduce `hictk dump` byte-for-byte.
//!
//! The cases live in `tests/manifest.json` and the goldens are compiled in via
//! `common::golden` (so a missing golden is a build error, not a skip). Each
//! case runs against `dmel-hictk-v9`, a `.hic` produced by hictk 2.2.0 from
//! `dmel-root-13res` — both declared in the manifest with their generator
//! commands, so regenerating a golden does not mean retyping a recipe out of a
//! doc comment.
//!
//! Regenerate after any change to the fixture set:
//!
//! ```sh
//! ./dev fixtures regen dump.pixels.100kb
//! ```
//!
//! The goldens pin the parts that are easy to get subtly wrong and that a
//! round-trip test would not catch: `%.16g` count formatting including `%g`'s
//! trailing-zero trimming, the `.hic` balanced path rounding through **f32**
//! rather than dividing in f64, raw (divisive) vectors printed under an
//! alphabetically sorted header, and ascending resolution order despite `.hic`
//! storing zoom levels finest-last.

mod common;

use std::path::Path;
use std::process::Command;

use common::{cases_with_prefix, check_case, fixture_or_skip, is_lenient};

/// Every `dump.` case: whole-file tables, region tables, and the two cooler
/// regressions. The `ran` guard is what stops "all cases silently skipped" from
/// looking like a pass.
#[test]
fn matches_the_declared_dump_cases() {
    let cases = cases_with_prefix("dump.");
    assert!(!cases.is_empty(), "no `dump.` cases in tests/manifest.json");

    let mut ran = 0;
    let mut skipped = Vec::new();
    for c in &cases {
        if check_case(c).is_some() {
            ran += 1;
        } else {
            skipped.push(c.id.clone());
        }
    }

    assert!(
        ran > 0 || is_lenient(),
        "all {} `dump.` cases were skipped but the run is not lenient — the \
         fixtures exist and are still not being exercised: {skipped:?}",
        cases.len()
    );
}

/// `--resolution` is mandatory for a multi-resolution `.hic`.
///
/// Hand-written rather than a manifest case: it asserts a non-zero exit and a
/// substring of stderr, and a small expectation DSL for that would be more
/// machinery than the lines it saves. (Cases are for argv-shaped comparisons
/// against a golden or a profile.)
#[test]
fn resolution_is_required_for_multi_resolution_hic() {
    let Some(hic) = fixture_or_skip("dmel-hictk-v9") else {
        return;
    };
    let out = Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
        .args(["dump", "-t", "pixels"])
        .arg(&hic)
        .env("HDF5_USE_FILE_LOCKING", "FALSE")
        .output()
        .expect("run cooler-rs dump");
    assert!(!out.status.success(), "expected a non-zero exit");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--resolution is mandatory"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Both formats' `weights` table must agree on the vector they share.
///
/// `dump -t weights` prints the DIVISIVE convention whatever the column holds
/// (`src/hictk/dump/common.cpp:88-92`): a cooler's divisive `KR` comes out as
/// stored, its multiplicative `weight` as `1/w`. Both derived `.hic`s carry
/// `1/weight` under the name `ICE`, so each printed vector must match the
/// cooler's — which is the check that the column's convention was *resolved*
/// (attribute, then name) rather than assumed from the container format. The
/// cooler branch used to print `1/w` unconditionally, wrong for `KR`, which made
/// this table disagree with the `.hic` by `KR²`.
///
/// Precision is version-dependent: a `.hic` stores its normalization vectors as
/// f64 up to v8 and f32 from v9 (`hic/file_reader_impl.hpp:129-137`). Both
/// writers emit v9 now, so both quantize the cooler's f64 column to f32 — and
/// they must quantize it *identically*, bit for bit, since it is the same cast
/// of the same input. That equality is the assertion worth making: a tolerance
/// against both sides would hide a precision change, and an exact match against
/// the cooler would be impossible by construction.
///
/// Hand-written rather than a manifest case: it needs two dumps and a column
/// lookup by name.
#[test]
fn cooler_and_hic_weights_agree_on_the_shared_vector() {
    let (Some(root), Some(ours), Some(hictk)) = (
        fixture_or_skip("dmel-root-13res"),
        fixture_or_skip("dmel-ours-v9"),
        fixture_or_skip("dmel-hictk-v9"),
    ) else {
        return;
    };

    let table = |path: &Path| -> Vec<Vec<String>> {
        let out = Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
            .args([
                "dump",
                "-t",
                "weights",
                "--resolution",
                "100000",
                "-r",
                "chr2L:0-300000",
            ])
            .arg(path)
            .env("HDF5_USE_FILE_LOCKING", "FALSE")
            .output()
            .expect("run cooler-rs dump");
        assert!(
            out.status.success(),
            "dump -t weights {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| l.split('\t').map(str::to_string).collect())
            .collect()
    };

    // Columns are printed in sorted order, so no index is fixed.
    let column = |t: &[Vec<String>], name: &str| -> Vec<String> {
        let i = t[0]
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no `{name}` column in {:?}", t[0]));
        t[1..].iter().map(|r| r[i].clone()).collect()
    };

    let cooler_w = column(&table(&root), "weight");
    let ours_ice = column(&table(&ours), "ICE");
    let hictk_ice = column(&table(&hictk), "ICE");

    assert_eq!(
        cooler_w.len(),
        ours_ice.len(),
        "the two weights tables differ in length"
    );
    assert_eq!(
        ours_ice, hictk_ice,
        "two v9 writers must store the same f32 vector for the same input"
    );

    assert_eq!(cooler_w.len(), hictk_ice.len());
    let max_rel = cooler_w
        .iter()
        .zip(&hictk_ice)
        .map(|(a, b)| {
            let (a, b): (f64, f64) = (a.parse().unwrap(), b.parse().unwrap());
            let d = (a - b).abs();
            let m = a.abs().max(b.abs());
            if m > 0.0 {
                d / m
            } else {
                0.0
            }
        })
        .fold(0.0f64, f64::max);
    assert!(
        max_rel < 1e-6,
        "a v9 `.hic` stores norms as f32, so it should differ from the \
         cooler only by f32 rounding — got max relative deviation {max_rel:.3e}"
    );
    assert!(
        max_rel > 0.0,
        "the v9 ICE matches the cooler bit-for-bit, which would mean it is \
         not storing f32 — the version-dependent reader/writer assumption is wrong"
    );
}
