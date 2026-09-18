#![cfg(any())]
//! End-to-end tests for the `cooler-rs validate` subcommand: the library
//! checker itself is covered by `tests/update.rs`, these check the CLI wiring
//! (argument handling, per-resolution iteration, exit code).
//!
//! HDF5 file locking is disabled for the whole process: the spawned `cooler-rs`
//! child would otherwise inherit the fd of any open test file, keep its flock
//! alive, and make a sibling test's reopen fail with `EAGAIN`.

use std::path::{Path, PathBuf};

use cooler_rs::{Chrom, CoolerWriter, McoolWriter, Pixel};

fn chroms() -> Vec<Chrom> {
    vec![
        Chrom {
            name: "chr1".into(),
            length: 250_000,
        },
        Chrom {
            name: "chr2".into(),
            length: 100_000,
        },
    ]
}

/// 4 bins (chr1: 3, chr2: 1) at 100 kb, 5 stored pixels.
fn pixels() -> Vec<Pixel> {
    vec![
        Pixel {
            bin1_id: 0,
            bin2_id: 0,
            count: 4.0,
        },
        Pixel {
            bin1_id: 0,
            bin2_id: 1,
            count: 5.0,
        },
        Pixel {
            bin1_id: 1,
            bin2_id: 3,
            count: 2.5,
        },
        Pixel {
            bin1_id: 2,
            bin2_id: 2,
            count: 7.0,
        },
        Pixel {
            bin1_id: 3,
            bin2_id: 3,
            count: 1.0,
        },
    ]
}

fn make_cool(dir: &Path) -> PathBuf {
    let path = dir.join("t.cool");
    let writer = CoolerWriter::create(&path, &chroms(), 100_000).unwrap();
    writer.write_pixels(&pixels()).unwrap();
    path
}

/// Overwrite `pixels/bin1_id` in place (test corruption helper).
fn corrupt_bin1_ids(path: &Path, group: &str) {
    let file = hdf5_metno::File::open_rw(path).unwrap();
    file.group(group)
        .unwrap()
        .dataset("pixels/bin1_id")
        .unwrap()
        .write(&[5i64, 0, 1, 2, 3]) // bin 5 does not exist
        .unwrap();
}

/// Run `cooler-rs validate <path>`, returning (succeeded, output).
fn run_validate(path: &Path) -> (bool, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_cooler-rs"))
        .arg("validate")
        .arg(path)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[test]
fn exits_zero_on_a_clean_cool() {
    std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(dir.path());

    let (ok, out) = run_validate(&path);
    assert!(ok, "expected exit 0, got output: {out}");
}

#[test]
fn exits_nonzero_and_reports_a_corrupt_cool() {
    std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(dir.path());
    corrupt_bin1_ids(&path, "/");

    let (ok, out) = run_validate(&path);
    assert!(!ok, "expected a non-zero exit, got output: {out}");
    assert!(out.contains("out of range"), "output: {out}");
}

#[test]
fn checks_every_mcool_resolution() {
    std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.mcool");
    let writer = McoolWriter::create(&path).unwrap();
    for res in [100_000, 50_000] {
        writer
            .create_cooler(&chroms(), res)
            .unwrap()
            .write_pixels(&pixels())
            .unwrap();
    }
    drop(writer);

    let (ok, out) = run_validate(&path);
    assert!(ok, "expected exit 0, got output: {out}");

    // Corrupting only the 50 kb resolution must still be caught, and the
    // issue must be attributed to that resolution's group.
    corrupt_bin1_ids(&path, "/resolutions/50000");

    let (ok, out) = run_validate(&path);
    assert!(!ok, "expected a non-zero exit, got output: {out}");
    assert!(out.contains("/resolutions/50000"), "output: {out}");
}
