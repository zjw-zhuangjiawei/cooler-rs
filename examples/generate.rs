//! Generate toy Hi-C contact matrices with TAD-like block structure,
//! written as both `.cool` and `.mcool`.
//!
//! Unlike a few stray pixels, the block-diagonal signal gives downstream
//! tools (e.g. `ontad`) something to find, so the sample files double as
//! input for the other examples/bins.
//!
//! Usage: `cargo run --example generate [output-prefix]`
//!
//! Writes `<prefix>.cool` (single resolution) and `<prefix>.mcool`
//! (three resolutions). Default prefix: `sample`.

use cooler_rs::{Chrom, CoolerWriter, McoolWriter, Pixel};

/// Toy interaction blocks for one chromosome: strong signal inside each
/// TAD, weak background elsewhere. Upper triangle only — cooler stores a
/// symmetric matrix once. `offset` is the chromosome's bin start in the
/// file-wide bin index.
fn toy_pixels(offset: i64, n_bins: i64, tads: i64) -> Vec<Pixel> {
    let per = (n_bins / tads).max(1);
    let mut px = Vec::new();
    for i in 0..n_bins {
        for j in i + 1..n_bins {
            let in_tad = i / per == j / per;
            px.push(Pixel {
                bin1_id: offset + i,
                bin2_id: offset + j,
                count: if in_tad { 50.0 } else { 2.0 },
            });
        }
    }
    px
}

/// Block pixels for every chromosome at a given resolution.
fn all_pixels(chroms: &[Chrom], res: u32) -> Vec<Pixel> {
    let mut px = Vec::new();
    let mut offset = 0i64;
    for ch in chroms {
        let n_bins = ch.length as i64 / res as i64;
        let tads = (n_bins / 5).max(1); // ~5 bins per TAD
        px.extend(toy_pixels(offset, n_bins, tads));
        offset += n_bins;
    }
    px
}

/// Small synthetic chromosomes so the whole example runs in milliseconds.
fn chroms() -> Vec<Chrom> {
    vec![
        Chrom {
            name: "chr1".into(),
            length: 2_000_000,
        },
        Chrom {
            name: "chr2".into(),
            length: 1_000_000,
        },
    ]
}

fn main() -> cooler_rs::Result<()> {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| "sample".into());
    let chroms = chroms();

    // Single-resolution .cool at 100 kb.
    let cool_path = format!("{prefix}.cool");
    let cool = CoolerWriter::create(&cool_path, &chroms, 100_000)?;
    let pixels = all_pixels(&chroms, 100_000);
    cool.write_pixels(&pixels)?;
    println!("wrote {cool_path} ({} pixels)", pixels.len());

    // Multi-resolution .mcool at 100 kb, 500 kb, and 1 Mb.
    let mcool_path = format!("{prefix}.mcool");
    let mw = McoolWriter::create(&mcool_path)?;
    for &res in &[100_000, 500_000, 1_000_000] {
        let c = mw.create_cooler(&chroms, res)?;
        let pixels = all_pixels(&chroms, res);
        c.write_pixels(&pixels)?;
        println!("wrote {mcool_path} @ {res} ({} pixels)", pixels.len());
    }
    Ok(())
}
