//! `cooler-rs dump` must reproduce `hictk dump` byte-for-byte.
//!
//! # Expected fixtures (`tests/data/hictk.dump.*.txt`)
//!
//! Each fixture is the stdout of the corresponding `hictk dump` invocation on
//! `tests/data/4DNFIOTPSS3L.hic` (Drosophila, `.hic` v8), produced by hictk
//! 2.2.0 (`ghcr.io/paulsengroup/hictk:2.2.0`). They pin the parts that are
//! easy to get subtly wrong and that a round-trip test would not catch:
//!
//! - `%.16g` count formatting, including the trailing-zero trimming of `%g`;
//! - the `.hic` balanced-count path, which rounds through **f32**
//!   (`count /= (float)(w1 * w2)`) rather than dividing in f64;
//! - `weights` printing raw (divisive) vectors under an alphabetically sorted
//!   header;
//! - ascending resolution order, despite `.hic` zoom levels being stored
//!   finest-last.
//!
//! The `.hic` input is gitignored, so every test is skipped when it is absent.
//!
//! ## Regenerating
//!
//! ```sh
//! hictk() { podman run --rm -v "$PWD/tests/data:/d:ro" \
//!   ghcr.io/paulsengroup/hictk:2.2.0 dump "$@" /d/4DNFIOTPSS3L.hic; }
//! hictk -t chroms                  > tests/data/hictk.dump.chroms.txt
//! hictk -t resolutions             > tests/data/hictk.dump.resolutions.txt
//! hictk -t normalizations          > tests/data/hictk.dump.normalizations.txt
//! hictk -t bins    --resolution 10000 -r 2L:0-30000 > tests/data/hictk.dump.bins.10000.2L0-30000.txt
//! hictk -t pixels  --resolution 10000 -r 2L:0-30000 > tests/data/hictk.dump.pixels.10000.2L0-30000.txt
//! hictk -t pixels  --resolution 10000 -r 2L:0-30000 -b KR > tests/data/hictk.dump.pixels.10000.2L0-30000.KR.txt
//! hictk -t pixels  --resolution 10000 -r 2L:0-30000 --join > tests/data/hictk.dump.pixels.10000.2L0-30000.join.txt
//! hictk -t weights --resolution 10000 -r 2L:0-30000 > tests/data/hictk.dump.weights.10000.2L0-30000.txt
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

const HIC: &str = "tests/data/4DNFIOTPSS3L.hic";

fn hic() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(HIC);
    p.exists().then_some(p)
}

fn fixture(name: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(format!("hictk.dump.{name}.txt"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn dump(hic: &Path, args: &[String]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
        .arg("dump")
        .args(args)
        .arg(hic)
        .output()
        .expect("run cooler-rs dump");
    assert!(
        out.status.success(),
        "cooler-rs dump {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("dump output is UTF-8")
}

/// Table dumps that need no `--resolution` and no region.
#[test]
fn matches_hictk_for_whole_file_tables() {
    let Some(hic) = hic() else {
        eprintln!("skipping: {HIC} not present");
        return;
    };
    let cases: &[(&str, &[&str])] = &[
        ("chroms", &["-t", "chroms"]),
        ("resolutions", &["-t", "resolutions"]),
        ("normalizations", &["-t", "normalizations"]),
    ];
    for (name, args) in cases {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        assert_eq!(dump(&hic, &args), fixture(name), "table `{name}`");
    }
}

/// Region dumps at 10 kb on `2L:0-30000`, with and without normalization.
#[test]
fn matches_hictk_for_2l_region() {
    let Some(hic) = hic() else {
        eprintln!("skipping: {HIC} not present");
        return;
    };
    let with = |table: &str, extra: &[&str]| -> Vec<String> {
        let mut v = vec![
            "-t".to_string(),
            table.to_string(),
            "--resolution".to_string(),
            "10000".to_string(),
            "-r".to_string(),
            "2L:0-30000".to_string(),
        ];
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let cases: Vec<(&str, Vec<String>)> = vec![
        ("bins.10000.2L0-30000", with("bins", &[])),
        ("pixels.10000.2L0-30000", with("pixels", &[])),
        // The balanced path is the f32 one; a f64 divide differs in the last
        // printed digit.
        ("pixels.10000.2L0-30000.KR", with("pixels", &["-b", "KR"])),
        ("pixels.10000.2L0-30000.join", with("pixels", &["--join"])),
        ("weights.10000.2L0-30000", with("weights", &[])),
    ];
    for (name, args) in cases {
        assert_eq!(dump(&hic, &args), fixture(name), "table `{name}`");
    }
}

/// `--resolution` is mandatory for a multi-resolution `.hic`.
#[test]
fn resolution_is_required_for_multi_resolution_hic() {
    let Some(hic) = hic() else {
        eprintln!("skipping: {HIC} not present");
        return;
    };
    let out = Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
        .args(["dump", "-t", "pixels"])
        .arg(&hic)
        .output()
        .expect("run cooler-rs dump");
    assert!(!out.status.success(), "expected a non-zero exit");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--resolution is mandatory"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
