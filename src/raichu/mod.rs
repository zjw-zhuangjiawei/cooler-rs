//! Raichu contact-normalization (port of RaichuNorm v1.1).
//!
//! Raichu fits a per-bin weight vector so that observed contacts match a
//! distance-dependent expected curve `Ed`: it computes `Ed` genome-wide, then
//! per chromosome initializes weights with a MAD-max filter, extracts
//! percentile-filtered O/E pixels, and optimizes the log-weights in
//! overlapping sliding windows via [`dual_annealing`]. The bias written to the
//! file is `10^(-w)`.
//!
//! Only intra-chromosomal contacts are used. The deterministic stages
//! (`expected_contacts`, `init_weights`, `extract_pixels`, `split_windows`,
//! `combine_weights`) are faithful ports of `raichu/util.py`; the optimizer is
//! approximate (see [`dual_annealing`]).

mod dual_annealing;
mod lbfgsb;

pub use dual_annealing::{dual_annealing, DualAnnealingResult};

use std::collections::{BTreeMap, HashMap};

use crate::balance::median;
use crate::cooler::Cooler;
use crate::error::{Error, Result};

/// Parameters for [`raichu_normalize`], mirroring the Raichu CLI defaults.
#[derive(Debug, Clone)]
pub struct RaichuParams {
    /// Sliding window size in bins.
    pub window_size: usize,
    /// Maximum genomic distance to consider, in bins.
    pub max_distance: usize,
    /// Number of diagonals to skip when extracting pixels (`start_diag`).
    pub ignore_diags: usize,
    /// Drop bins with fewer than this many valid pixels.
    pub min_nnz: usize,
    /// Maximum dual-annealing iterations per window.
    pub maxiter: usize,
    /// Lower bound of the search space (linear weight scale).
    pub lower_bound: f64,
    /// Upper bound of the search space (linear weight scale).
    pub upper_bound: f64,
    /// Chromosome names to include; empty = all.
    pub chroms: Vec<String>,
    /// Per-chromosome list of included bin indices (local), from a BED file.
    pub included_bins: Option<HashMap<String, Vec<usize>>>,
    /// `dynamic_window_size` in `calculate_expected` (Python hardcoded 10).
    pub dynamic_window_size: usize,
    /// `N` in `calculate_expected` (Python hardcoded 400).
    pub n_threshold: f64,
    /// Upper O/E percentile bound (Python hardcoded 99).
    pub top_per: f64,
    /// Lower O/E percentile bound (Python hardcoded 1).
    pub bottom_per: f64,
}

impl Default for RaichuParams {
    fn default() -> Self {
        RaichuParams {
            window_size: 200,
            max_distance: 200,
            ignore_diags: 0,
            min_nnz: 10,
            maxiter: 100,
            lower_bound: 0.001,
            upper_bound: 1000.0,
            chroms: Vec::new(),
            included_bins: None,
            dynamic_window_size: 10,
            n_threshold: 400.0,
            top_per: 99.0,
            bottom_per: 1.0,
        }
    }
}

/// Median absolute deviation (`cooler.balance.mad`): median(|x - median(x)|).
pub fn mad(x: &[f64]) -> f64 {
    let med = median(x);
    let dev: Vec<f64> = x.iter().map(|&v| (v - med).abs()).collect();
    median(&dev)
}

/// `np.percentile` with default linear interpolation. `q` in [0, 100].
pub fn percentile(a: &[f64], q: f64) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.total_cmp(y));
    let n = s.len();
    let idx = (n - 1) as f64 * q / 100.0;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        let frac = idx - lo as f64;
        s[lo] + (s[hi] - s[lo]) * frac
    }
}

/// Symmetric marginal sums from stored (symmetric-upper) local pixels.
/// Equivalent to `sum(axis=0)` of cooler's full-symmetric matrix.
pub fn marginal(n: usize, pixels: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut marg = vec![0.0; n];
    for &(i, j, v) in pixels {
        if i == j {
            marg[i] += v;
        } else {
            marg[i] += v;
            marg[j] += v;
        }
    }
    marg
}

/// `initialize_weights`: MAD-max filter on the marginal, then `sqrt(marg/scale)`.
///
/// Returns `(weights, valid_cols)`; `valid_cols[i] = weights[i] > 0`.
pub fn init_weights(marg: &[f64], included: Option<&[usize]>) -> (Vec<f64>, Vec<bool>) {
    let n = marg.len();
    let mut marg = match included {
        None => marg.to_vec(),
        Some(inc) => {
            let mut m = vec![0.0; n];
            for &i in inc {
                m[i] = marg[i];
            }
            m
        }
    };
    let log_pos: Vec<f64> = marg.iter().filter(|&&v| v > 0.0).map(|&v| v.ln()).collect();
    if log_pos.is_empty() {
        return (vec![0.0; n], vec![false; n]);
    }
    let med = median(&log_pos);
    let dev = mad(&log_pos);
    let cutoff = (med - 5.0 * dev).exp();
    for m in marg.iter_mut() {
        if *m < cutoff {
            *m = 0.0;
        }
    }
    let pos: Vec<f64> = marg.iter().filter(|&&v| v > 0.0).copied().collect();
    let scale = pos.iter().sum::<f64>() / pos.len() as f64;
    let weights: Vec<f64> = marg.iter().map(|&m| (m / scale).sqrt()).collect();
    let valid: Vec<bool> = weights.iter().map(|&w| w > 0.0).collect();
    (weights, valid)
}

/// `extract_valid_pixels`: keep valid pixels with O/E strictly between the
/// `bottom_per`/`top_per` percentiles. Returns local `(i, j)` coords and data.
#[allow(clippy::too_many_arguments)]
pub fn extract_pixels(
    pixels: &[(usize, usize, f64)],
    ed: &[f64],
    valid: &[bool],
    start_diag: usize,
    maxd: usize,
    top_per: f64,
    bottom_per: f64,
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let mut coords = Vec::new();
    let mut data = Vec::new();
    let mut oe_all = Vec::new();
    for &(i, j, v) in pixels {
        let d = j - i;
        if d < start_diag || d > maxd {
            continue;
        }
        if !valid[i] || !valid[j] || v <= 0.0 {
            continue;
        }
        let ed_d = ed[d];
        if ed_d.is_nan() {
            continue;
        }
        coords.push((i, j));
        data.push(v);
        oe_all.push(v / ed_d);
    }
    let top_v = percentile(&oe_all, top_per);
    let bottom_v = percentile(&oe_all, bottom_per);
    let mut out_coords = Vec::new();
    let mut out_data = Vec::new();
    for (k, &oe) in oe_all.iter().enumerate() {
        if oe > bottom_v && oe < top_v {
            out_coords.push(coords[k]);
            out_data.push(data[k]);
        }
    }
    (out_coords, out_data)
}

/// `calculate_expected`: genome-wide `Ed[d]` (expected contacts per valid bin
/// pair at distance `d`), with the dynamic-window search. Returns `(Ed, maxd)`;
/// `Ed` entries beyond `maxd` are `NaN`.
pub fn expected_contacts(
    chrom_pixels: &[Vec<(usize, usize, f64)>],
    chrom_n: &[usize],
    included: Option<&HashMap<usize, Vec<usize>>>,
    max_dis: usize,
    dyn_window: usize,
    n_threshold: f64,
) -> (Vec<f64>, usize) {
    let mut diag_sums = vec![0.0; max_dis + 1];
    let mut pixel_nums = vec![0.0; max_dis + 1];
    for (c, pixels) in chrom_pixels.iter().enumerate() {
        let n = chrom_n[c];
        let marg = marginal(n, pixels);
        let mut valid = vec![false; n];
        for i in 0..n {
            valid[i] = marg[i] > 0.0;
        }
        if let Some(inc) = included {
            if let Some(inc_c) = inc.get(&c) {
                let mut tmp = vec![false; n];
                for &i in inc_c {
                    tmp[i] = true;
                }
                for i in 0..n {
                    valid[i] = valid[i] && tmp[i];
                }
            }
        }
        let maxd = max_dis.min(n.saturating_sub(1));
        for &(i, j, v) in pixels {
            let d = j - i;
            if d <= maxd && valid[i] && valid[j] {
                diag_sums[d] += v;
            }
        }
        for d in 0..=maxd {
            for i in 0..(n - d) {
                if valid[i] && valid[i + d] {
                    pixel_nums[d] += 1.0;
                }
            }
        }
    }
    let mut ed = vec![f64::NAN; max_dis + 1];
    for (i, e) in ed.iter_mut().enumerate() {
        for w in 0..=dyn_window {
            let lo = i.saturating_sub(w);
            let hi = (i + w + 1).min(max_dis + 1);
            let n_count: f64 = diag_sums[lo..hi].iter().sum();
            let n_pixel: f64 = pixel_nums[lo..hi].iter().sum();
            if n_count > n_threshold {
                *e = n_count / n_pixel;
                break;
            }
        }
    }
    let maxd = (0..=max_dis).rev().find(|&i| !ed[i].is_nan()).unwrap_or(0);
    (ed, maxd)
}

/// `split_chromosome`: overlapping 90%-step windows over a chromosome (local
/// bin indices). `chrom_size` and `bin_size` in bp, `ws_bp = window_size*bin_size`.
pub fn split_windows(chrom_size: i64, bin_size: i64, ws_bp: i64) -> Vec<(usize, usize)> {
    let res = bin_size;
    let step = ws_bp / 10 * 9;
    let mut queue = Vec::new();
    let mut s = 0i64;
    while s < chrom_size {
        if chrom_size - s > ws_bp / 2 * 3 {
            let e = s + ws_bp;
            queue.push(((s / res) as usize, (e / res) as usize));
            s += step;
        } else {
            let end = if chrom_size % res == 0 {
                chrom_size / res
            } else {
                chrom_size / res + 1
            };
            queue.push(((s / res) as usize, end as usize));
            break;
        }
    }
    queue
}

/// `extract_valid_regions`: windowing over contiguous runs of included bins.
pub fn split_windows_regions(included: &[usize], bin_size: i64, ws_bp: i64) -> Vec<(usize, usize)> {
    let step = ws_bp / 10 * 9;
    let mut queue = Vec::new();
    let mut i = 0;
    while i < included.len() {
        let run_start = i;
        while i + 1 < included.len() && included[i + 1] == included[i] + 1 {
            i += 1;
        }
        let start_bp = included[run_start] as i64 * bin_size;
        let end_bp = included[i] as i64 * bin_size;
        let mut s = start_bp;
        while s < end_bp {
            if end_bp - s > ws_bp / 2 * 3 {
                let e = s + ws_bp;
                queue.push(((s / bin_size) as usize, (e / bin_size) as usize));
                s += step;
            } else {
                queue.push(((s / bin_size) as usize, (end_bp / bin_size) as usize));
                break;
            }
        }
        i += 1;
    }
    queue
}

/// `combine_weights`: average overlapping windows. The Python `is nan` branch
/// is dead code (NaN compares unequal even to itself), so we average.
pub fn combine_weights(windows: &BTreeMap<(usize, usize), Vec<f64>>, n: usize) -> Vec<f64> {
    let mut weights = vec![0.0; n];
    for (&(s, e), wv) in windows {
        for i in s..e {
            if weights[i] == 0.0 {
                weights[i] = wv[i - s];
            } else {
                weights[i] = (weights[i] + wv[i - s]) / 2.0;
            }
        }
    }
    weights
}

/// Compute the Raichu bias vector for a cooler collection.
///
/// Returns a per-bin vector of length `nbins`; `bias[i] = 10^(-w_i)`, `NaN`
/// for bins dropped by `min_nnz`, `0.0` for chromosomes excluded by
/// `RaichuParams::chroms`.
pub fn raichu_normalize(clr: &Cooler, p: &RaichuParams) -> Result<Vec<f64>> {
    let n_bins = clr.bins()?.len();
    let chroms = clr.chroms()?;
    let chrom_offset = clr.chrom_offset()?;
    let bin_size =
        clr.bin_size()?
            .ok_or_else(|| Error::Format("missing 'bin-size' attribute".into()))? as i64;

    let mut chrom_ids: Vec<usize> = Vec::new();
    for (cid, c) in chroms.iter().enumerate() {
        if p.chroms.is_empty() || p.chroms.iter().any(|n| n == &c.name) {
            chrom_ids.push(cid);
        }
    }

    let included_by_idx: Option<HashMap<usize, Vec<usize>>> = p.included_bins.as_ref().map(|m| {
        let mut out = HashMap::new();
        for (cid, c) in chroms.iter().enumerate() {
            if let Some(v) = m.get(&c.name) {
                out.insert(cid, v.clone());
            }
        }
        out
    });

    let mut chrom_pixels: Vec<Vec<(usize, usize, f64)>> = Vec::new();
    let mut chrom_n: Vec<usize> = Vec::new();
    for &cid in &chrom_ids {
        let lo = chrom_offset[cid];
        let hi = chrom_offset[cid + 1];
        let n = (hi - lo) as usize;
        chrom_n.push(n);
        let mut pixels = Vec::new();
        for px in clr.pixels_for_bins(lo, hi)? {
            if px.bin2_id >= lo && px.bin2_id < hi {
                pixels.push((
                    (px.bin1_id - lo) as usize,
                    (px.bin2_id - lo) as usize,
                    px.count,
                ));
            }
        }
        chrom_pixels.push(pixels);
    }

    let max_dis = p.max_distance;
    let (ed, maxd) = expected_contacts(
        &chrom_pixels,
        &chrom_n,
        included_by_idx.as_ref(),
        max_dis,
        p.dynamic_window_size,
        p.n_threshold,
    );

    let ws_bp = p.window_size as i64 * bin_size;
    let lb = p.lower_bound.log10();
    let ub = p.upper_bound.log10();

    let mut bias = vec![0.0; n_bins];

    for (k, &cid) in chrom_ids.iter().enumerate() {
        let lo = chrom_offset[cid];
        let hi = chrom_offset[cid + 1];
        let n = (hi - lo) as usize;
        let pixels = &chrom_pixels[k];
        let marg = marginal(n, pixels);
        let included_c: Option<&[usize]> = included_by_idx
            .as_ref()
            .and_then(|m| m.get(&cid).map(|v| v.as_slice()));
        let (mut ini_weights, valid_cols) = init_weights(&marg, included_c);
        for w in ini_weights.iter_mut() {
            if *w > 0.0 {
                *w = w.log10();
            }
        }

        let (coords, data) = extract_pixels(
            pixels,
            &ed,
            &valid_cols,
            p.ignore_diags,
            maxd,
            p.top_per,
            p.bottom_per,
        );

        let queue = match included_c {
            None => split_windows(chroms[cid].length as i64, bin_size, ws_bp),
            Some(inc) => split_windows_regions(inc, bin_size, ws_bp),
        };

        let mut collect: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        for &(s, e) in &queue {
            let rl = e - s;
            if included_c.is_none() {
                let n_valid = valid_cols[s..e].iter().filter(|&&v| v).count() as f64;
                if n_valid / (rl as f64) < 0.1 {
                    collect.insert((s, e), vec![f64::NAN; rl]);
                    continue;
                }
            }
            let mut coords_: Vec<(usize, usize)> = Vec::new();
            let mut data_: Vec<f64> = Vec::new();
            let mut earr: Vec<f64> = Vec::new();
            for (idx, &(a, b)) in coords.iter().enumerate() {
                if a >= s && b < e {
                    coords_.push((a - s, b - s));
                    data_.push(data[idx]);
                    earr.push(ed[b - a]);
                }
            }
            let mut f = |w: &[f64]| -> f64 {
                let mut obj = 0.0;
                for (idx, &(i, j)) in coords_.iter().enumerate() {
                    obj +=
                        (data_[idx] - earr[idx] * 10.0_f64.powf(w[i]) * 10.0_f64.powf(w[j])).abs();
                }
                obj
            };
            let bounds_low = vec![lb; rl];
            let bounds_up = vec![ub; rl];
            let res = dual_annealing(
                &mut f,
                &bounds_low,
                &bounds_up,
                &ini_weights[s..e],
                p.maxiter,
                42,
            );
            collect.insert((s, e), res.x);
        }

        let mut weights = combine_weights(&collect, n);

        let mut counts = vec![0usize; n];
        for &(a, b) in &coords {
            counts[a] += 1;
            counts[b] += 1;
        }
        for (i, w) in weights.iter_mut().enumerate() {
            if counts[i] < p.min_nnz {
                *w = f64::NAN;
            }
        }

        for i in 0..n {
            bias[(lo + i as i64) as usize] = 10.0_f64.powf(-weights[i]);
        }
    }

    Ok(bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_linear() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&a, 50.0) - 2.5).abs() < 1e-12);
        assert!((percentile(&a, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile(&a, 100.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn marginal_is_symmetric() {
        // 3 bins: pixels (0,1)=5, (0,0)=2, (1,2)=3
        let px = [(0, 1, 5.0), (0, 0, 2.0), (1, 2, 3.0)];
        let m = marginal(3, &px);
        assert_eq!(m, vec![7.0, 8.0, 3.0]); // bin0: 5+2, bin1: 5+3, bin2: 3
    }

    #[test]
    fn init_weights_marks_zero_marginal_invalid() {
        let marg = [0.0, 10.0, 20.0, 30.0];
        let (w, valid) = init_weights(&marg, None);
        assert!(!valid[0]);
        assert!(valid[1] && valid[2] && valid[3]);
        assert_eq!(w[0], 0.0);
    }

    #[test]
    fn combine_averages_overlap() {
        let mut windows = BTreeMap::new();
        windows.insert((0, 2), vec![1.0, 3.0]);
        windows.insert((1, 3), vec![5.0, 7.0]);
        let w = combine_weights(&windows, 3);
        // bin0: 1.0, bin1: (3+5)/2=4, bin2: 7.0
        assert_eq!(w, vec![1.0, 4.0, 7.0]);
    }
}
