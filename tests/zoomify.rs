//! Identity / self-consistency tests for `zoomify`.
//!
//! # Expected fixture (`tests/data/syn.zoomify.cooler-python.json`)
//!
//! `matches_python_cooler_fixtures` compares the Rust `zoomify_cooler` output
//! pixel-for-pixel against Python cooler's `cooler.zoomify_cooler` on a small
//! two-chromosome matrix whose per-chromosome bin counts are *not* multiples of
//! the coarsening factor, so the boundary-rounding path is exercised. The
//! fixture stores one shared input matrix (`matrix`) and one expected pixel set
//! per resolution (`levels`, base resolution included). The test is skipped
//! (with a note) when the file is absent.
//!
//! ## Regenerating
//!
//! Requires a Python venv with `cooler` and `numpy`. Run the block below from
//! `tests/data/`; it writes `syn.zoomify.cooler-python.json`. The fixture pins
//! `cooler_version` so a cooler bump can be audited against a freshly
//! generated fixture.
//!
//! ```python
//! import json, os, tempfile
//! import pandas as pd
//! import cooler
//! from cooler.create import create
//!
//! OUT = "syn.zoomify.cooler-python.json"
//! BASE = 10
//! chroms = [("chr1", 25), ("chr2", 15)]
//! pixels = [
//!     (0,0,1.0),(0,1,2.0),(0,2,3.0),(1,1,4.0),(1,2,5.0),(2,2,6.0),
//!     (3,3,10.0),(3,4,20.0),(4,4,30.0),
//!     (0,3,100.0),(1,4,200.0),(2,3,300.0),(2,4,400.0),
//! ]
//! resolutions = [20, 40]
//!
//! rows = []
//! for cid, (name, length) in enumerate(chroms):
//!     start = 0
//!     while start < length:
//!         rows.append((cid, name, start, min(start + BASE, length)))
//!         start += BASE
//! bins = pd.DataFrame(rows, columns=["chrom_id", "chrom", "start", "end"])
//! px = pd.DataFrame(pixels, columns=["bin1_id", "bin2_id", "count"])
//!
//! levels = []
//! with tempfile.TemporaryDirectory() as tmp:
//!     cool = os.path.join(tmp, "in.cool")
//!     mcool = os.path.join(tmp, "out.mcool")
//!     create(cool, bins, px, symmetric_upper=True, assembly="test",
//!            dtypes={"count": "float64"})
//!     cooler.zoomify_cooler(cool, mcool, resolutions, chunksize=10_000_000)
//!     for r in sorted({BASE, *resolutions}):
//!         clr = cooler.Cooler(mcool + f"::resolutions/{r}")
//!         df = clr.pixels()[:]
//!         levels.append({"resolution": r,
//!                        "pixels": [[int(t.bin1_id), int(t.bin2_id), float(t.count)]
//!                                   for t in df.itertuples(index=False)]})
//!
//! doc = {"cooler_version": cooler.__version__,
//!        "matrix": {"chroms": [{"name": c, "length": n} for c, n in chroms],
//!                   "pixels": [[int(a), int(b), float(c)] for a, b, c in pixels]},
//!        "resolutions": resolutions,
//!        "levels": levels}
//! with open(OUT, "w") as f:
//!     json.dump(doc, f, indent=1)
//! ```

use std::path::Path;

use cooler_rs::{
    coarsen_pixels, nice_resolutions, pow2_resolutions, zoomify_cooler, Chrom, CoolerWriter, Mcool,
    Pixel, ZoomifyParams,
};

fn px(bin1_id: i64, bin2_id: i64, count: f64) -> Pixel {
    Pixel {
        bin1_id,
        bin2_id,
        count,
    }
}

#[test]
fn coarsen_sums_factor_blocks() {
    let pixels = vec![px(0, 0, 1.0), px(0, 1, 2.0), px(1, 1, 3.0), px(2, 3, 4.0)];
    let out = coarsen_pixels(pixels, 2);
    assert_eq!(out, vec![px(0, 0, 6.0), px(1, 1, 4.0)]);
}

#[test]
fn resolution_progressions() {
    assert_eq!(pow2_resolutions(1000, 8000), vec![1000, 2000, 4000, 8000]);
    assert_eq!(
        nice_resolutions(1000, 10_000),
        vec![1000, 2000, 5000, 10_000]
    );
    assert_eq!(
        nice_resolutions(1000, 100_000),
        vec![1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
    );
}

#[test]
fn zoomify_pools_per_chromosome() {
    let dir = tempfile::tempdir().unwrap();
    let cool_path = dir.path().join("in.cool");
    let mcool_path = dir.path().join("out.mcool");

    // chrA: 3 bins (0,1,2); chrB: 2 bins (3,4). Base resolution 10 bp.
    let chroms = vec![
        Chrom {
            name: "chrA".into(),
            length: 30,
        },
        Chrom {
            name: "chrB".into(),
            length: 20,
        },
    ];
    let pixels = vec![
        px(0, 0, 1.0),
        px(0, 1, 2.0),
        px(1, 1, 3.0),
        px(2, 2, 4.0),
        px(3, 3, 10.0),
        px(3, 4, 20.0),
        px(4, 4, 30.0),
        px(2, 3, 100.0), // trans chrA-chrB
    ];
    let writer = CoolerWriter::create(&cool_path, &chroms, 10).unwrap();
    writer.write_pixels(&pixels).unwrap();

    // Coarsen to 20 bp (factor 2), dropping the base resolution.
    zoomify_cooler(
        &cool_path,
        &mcool_path,
        &ZoomifyParams {
            resolutions: vec![20],
            copy_base_resolution: false,
            ..ZoomifyParams::default()
        },
    )
    .unwrap();

    let mcool = Mcool::open(&mcool_path).unwrap();
    assert_eq!(mcool.resolutions().unwrap(), vec![20]);

    let coarse = mcool.cooler(20).unwrap();
    let out = coarse.pixels().unwrap();
    assert_eq!(
        out,
        vec![
            px(0, 0, 6.0),
            px(1, 1, 4.0),
            px(1, 2, 100.0),
            px(2, 2, 60.0)
        ]
    );

    // Count is conserved under coarsening.
    let base_sum: f64 = pixels.iter().map(|p| p.count).sum();
    let coarse_sum: f64 = out.iter().map(|p| p.count).sum();
    assert_eq!(base_sum, coarse_sum);
}

#[test]
fn zoomify_copies_base_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let cool_path = dir.path().join("in.cool");
    let mcool_path = dir.path().join("out.mcool");

    let chroms = vec![Chrom {
        name: "chr1".into(),
        length: 40,
    }];
    let writer = CoolerWriter::create(&cool_path, &chroms, 10).unwrap();
    writer
        .write_pixels(&[px(0, 0, 1.0), px(1, 2, 2.0), px(2, 3, 3.0)])
        .unwrap();

    zoomify_cooler(
        &cool_path,
        &mcool_path,
        &ZoomifyParams {
            resolutions: vec![20],
            ..ZoomifyParams::default()
        },
    )
    .unwrap();

    let mcool = Mcool::open(&mcool_path).unwrap();
    assert_eq!(mcool.resolutions().unwrap(), vec![10, 20]);
}

/// Typed view of `tests/data/syn.zoomify.cooler-python.json` (see the
/// "Expected fixture" section in the module docs for how it is generated).
#[derive(serde::Deserialize)]
struct ExpectedFixture {
    #[allow(dead_code)]
    cooler_version: String,
    matrix: ExpectedMatrix,
    resolutions: Vec<i64>,
    levels: Vec<ExpectedLevel>,
}

#[derive(serde::Deserialize)]
struct ExpectedMatrix {
    chroms: Vec<ExpectedChrom>,
    pixels: Vec<(i64, i64, f64)>,
}

#[derive(serde::Deserialize)]
struct ExpectedChrom {
    name: String,
    length: i64,
}

#[derive(serde::Deserialize)]
struct ExpectedLevel {
    resolution: i64,
    pixels: Vec<(i64, i64, f64)>,
}

/// Identity check against expected fixtures generated by Python cooler.
/// Skipped (with a note) when the fixture is absent.
#[test]
fn matches_python_cooler_fixtures() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/syn.zoomify.cooler-python.json");
    if !fixture.exists() {
        eprintln!(
            "skipping identity test: {} not present (regenerate it with the recipe in \
             this file's module docs)",
            fixture.display()
        );
        return;
    }
    let text = std::fs::read_to_string(&fixture).unwrap();
    let doc: ExpectedFixture = serde_json::from_str(&text).unwrap();

    // The base resolution is the finest level in the fixture.
    let base = doc.levels.iter().map(|l| l.resolution).min().unwrap();
    let chroms: Vec<Chrom> = doc
        .matrix
        .chroms
        .iter()
        .map(|c| Chrom {
            name: c.name.clone(),
            length: c.length as i32,
        })
        .collect();
    let pixels: Vec<Pixel> = doc
        .matrix
        .pixels
        .iter()
        .map(|&(bin1_id, bin2_id, count)| Pixel {
            bin1_id,
            bin2_id,
            count,
        })
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let cool_path = dir.path().join("in.cool");
    let mcool_path = dir.path().join("out.mcool");
    let writer = CoolerWriter::create(&cool_path, &chroms, base as u32).unwrap();
    writer.write_pixels(&pixels).unwrap();

    let resolutions: Vec<u32> = doc.resolutions.iter().map(|&r| r as u32).collect();
    zoomify_cooler(
        &cool_path,
        &mcool_path,
        &ZoomifyParams {
            resolutions,
            ..ZoomifyParams::default()
        },
    )
    .unwrap();

    let mcool = Mcool::open(&mcool_path).unwrap();
    let expected_res: Vec<u64> = doc.levels.iter().map(|l| l.resolution as u64).collect();
    assert_eq!(mcool.resolutions().unwrap(), expected_res);

    for level in &doc.levels {
        let clr = mcool.cooler(level.resolution as u64).unwrap();
        let mut out = clr.pixels().unwrap();
        let mut expected: Vec<Pixel> = level
            .pixels
            .iter()
            .map(|&(bin1_id, bin2_id, count)| Pixel {
                bin1_id,
                bin2_id,
                count,
            })
            .collect();
        // cooler copies the base resolution verbatim (input order), while our
        // writer emits (bin1_id, bin2_id) order; compare sets, not order.
        out.sort_by_key(|p| (p.bin1_id, p.bin2_id));
        expected.sort_by_key(|p| (p.bin1_id, p.bin2_id));
        assert_eq!(
            out.len(),
            expected.len(),
            "resolution {}: pixel count",
            level.resolution
        );
        for (got, exp) in out.iter().zip(&expected) {
            assert_eq!(
                got.bin1_id, exp.bin1_id,
                "resolution {}: bin1_id",
                level.resolution
            );
            assert_eq!(
                got.bin2_id, exp.bin2_id,
                "resolution {}: bin2_id",
                level.resolution
            );
            assert!(
                (got.count - exp.count).abs() <= 1e-9 * exp.count.abs().max(1.0),
                "resolution {}: count {} vs {}",
                level.resolution,
                got.count,
                exp.count
            );
        }
    }
}
