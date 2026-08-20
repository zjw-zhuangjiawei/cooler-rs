use std::path::Path;

use crate::error::Result;
use crate::types::ChromMeta;

use super::Tad;

/// Write the `.tad` file (1-based bin boundaries, level, mean, score).
pub fn write_tad<P: AsRef<Path>>(path: P, tad: &Tad) -> Result<()> {
    let mut out = String::new();
    for j in 0..tad.len() {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.3}\t{:.3}\n",
            tad.bound[j][0] + 1,
            tad.bound[j][1] + 1,
            tad.level[j],
            tad.mean[j],
            tad.score[j]
        ));
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Write a genome-browser `.bed` file with level-dependent colors
/// (C++ `outputBED`; skips the level-0 whole-chromosome entry).
pub fn write_bed<P: AsRef<Path>>(path: P, tad: &Tad, meta: &ChromMeta) -> Result<()> {
    const COLORS: [&str; 5] = [
        "56,108,176",
        "127,201,127",
        "190,174,212",
        "253,192,134",
        "255,0,0",
    ];
    let mut out = format!(
        "track name=\"OnTAD {}\" description=\"OnTAD {}\" visibility=2 itemRgb=\"On\"\n",
        meta.name, meta.name
    );
    for j in 1..tad.len() {
        let level = tad.level[j].min(5);
        let start = (tad.bound[j][0] as u64 + 1) * meta.resolution;
        let end = ((tad.bound[j][1] as u64 + 1) * meta.resolution).min(meta.length);
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t0\t.\t{}\t{}\t{}\n",
            meta.name,
            start,
            end,
            j,
            start,
            end,
            COLORS[level - 1]
        ));
    }
    std::fs::write(path, out)?;
    Ok(())
}
