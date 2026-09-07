//! Contact-matrix comparison — similarity between pairs of `.cool`/`.mcool`
//! files.
//!
//! Two families of metric:
//!
//! * [`CompareMetric::Scc`] — the HiCRep stratum-adjusted correlation
//!   coefficient (Genome Res. 2017;27(11):1939-1949), a port of
//!   `hicrep.hicrep.hicrepSCC` / `sccByDiag`: normalize, mean-filter, then a
//!   per-diagonal Pearson correlation weighted by the variance of the
//!   variance-stabilizing transform.
//! * [`CompareMetric::Pearson`] / [`CompareMetric::Spearman`] — plain
//!   whole-matrix correlation over the flattened upper triangle (common-zero
//!   positions excluded).
//!
//! Every metric is computed per chromosome and returned as a `(name, value)`
//! list; the CLI aggregates it into a pairwise heatmap.

use std::collections::HashMap;

use ndarray::Array2;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;

use crate::{Cooler, Error, Result};

/// Which similarity metric to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareMetric {
    /// HiCRep stratum-adjusted correlation coefficient.
    Scc,
    /// Whole-matrix Pearson correlation (upper triangle, common zeros excluded).
    Pearson,
    /// Whole-matrix Spearman rank correlation (upper triangle, common zeros excluded).
    Spearman,
}

/// Options controlling comparison. Shared by all metrics; `h` and `max_dist`
/// only affect [`CompareMetric::Scc`].
#[derive(Clone, Debug)]
pub struct CompareParams {
    /// Half-size of the mean filter used to smooth matrices before SCC
    /// (0 = no smoothing).
    pub h: usize,
    /// Maximum genomic distance in bp to include in SCC (-1 = whole chromosome).
    pub max_dist: i64,
    /// Downsample the larger matrix to the smaller's contact count instead of
    /// normalizing by total contacts.
    pub downsample: bool,
    /// Chromosome names to compare (empty = all).
    pub chroms: Vec<String>,
}

impl Default for CompareParams {
    fn default() -> Self {
        CompareParams {
            h: 1,
            max_dist: -1,
            downsample: false,
            chroms: Vec::new(),
        }
    }
}

/// Per-chromosome triplets in local coordinates (`0 <= i <= j < n`, upper
/// triangle), matching the cooler symmetric-upper storage.
struct ChromMat {
    n: usize,
    triplets: Vec<(usize, usize, f64)>,
}

/// Compute one metric between two coolers, per chromosome.
///
/// The two files must share bin size, chromosome names, and bin counts.
pub fn compare_pair(
    c1: &Cooler,
    c2: &Cooler,
    metric: CompareMetric,
    params: &CompareParams,
) -> Result<Vec<(String, f64)>> {
    let bin1 = c1
        .bin_size()?
        .ok_or_else(|| Error::Format("file 1: missing 'bin-size' attribute".into()))?;
    let bin2 = c2
        .bin_size()?
        .ok_or_else(|| Error::Format("file 2: missing 'bin-size' attribute".into()))?;
    if bin1 != bin2 {
        return Err(Error::InvalidInput(format!(
            "different bin sizes: {bin1} vs {bin2}"
        )));
    }
    let chroms1 = c1.chroms()?;
    let chroms2 = c2.chroms()?;
    if chroms1 != chroms2 {
        return Err(Error::InvalidInput("different chromosome names".into()));
    }

    let selected: Vec<usize> = if params.chroms.is_empty() {
        (0..chroms1.len()).collect()
    } else {
        let mut out = Vec::new();
        for name in &params.chroms {
            let cid = chroms1
                .iter()
                .position(|c| &c.name == name)
                .ok_or_else(|| {
                    Error::InvalidInput(format!("chromosome '{name}' not found in file"))
                })?;
            out.push(cid);
        }
        out
    };

    let n1 = total_sum(c1)?;
    let n2 = total_sum(c2)?;
    let off1 = c1.chrom_offset()?;
    let off2 = c2.chrom_offset()?;

    let mut out = Vec::with_capacity(selected.len());
    for &cid in &selected {
        let name = chroms1[cid].name.clone();
        let m1 = fetch_chrom(c1, off1[cid], off1[cid + 1])?;
        let m2 = fetch_chrom(c2, off2[cid], off2[cid + 1])?;
        debug_assert_eq!(m1.n, m2.n);
        let value = match metric {
            CompareMetric::Scc => scc_chrom(&m1, &m2, n1, n2, bin1 as i64, params)?,
            CompareMetric::Pearson => pearson_chrom(&m1, &m2),
            CompareMetric::Spearman => spearman_chrom(&m1, &m2),
        };
        out.push((name, value));
    }
    Ok(out)
}

/// Read one chromosome's pixels as local upper-triangle triplets.
fn fetch_chrom(clr: &Cooler, lo: i64, hi: i64) -> Result<ChromMat> {
    let n = (hi - lo) as usize;
    let mut triplets = Vec::new();
    for px in clr.pixels_for_bins(lo, hi)? {
        if px.bin2_id >= lo && px.bin2_id < hi && px.count.is_finite() {
            triplets.push((
                (px.bin1_id - lo) as usize,
                (px.bin2_id - lo) as usize,
                px.count,
            ));
        }
    }
    Ok(ChromMat { n, triplets })
}

/// Genome-wide sum of the `count` column, streamed in chunks.
fn total_sum(clr: &Cooler) -> Result<f64> {
    let nnz = clr.n_pixels()? as i64;
    let mut sum = 0.0;
    const CHUNK: i64 = 10_000_000;
    let mut lo = 0i64;
    while lo < nnz {
        let hi = (lo + CHUNK).min(nnz);
        for p in clr.pixels_range(lo, hi)? {
            sum += p.count;
        }
        lo = hi;
    }
    Ok(sum)
}

/// HiCRep SCC for a single chromosome.
fn scc_chrom(
    m1: &ChromMat,
    m2: &ChromMat,
    n1: f64,
    n2: f64,
    bin_size: i64,
    params: &CompareParams,
) -> Result<f64> {
    let n = m1.n;
    let d_max = if params.max_dist < 0 {
        n as i64
    } else {
        (params.max_dist / bin_size + 1).min(n as i64)
    };
    let n_diags = d_max as usize;
    if n_diags <= 1 {
        return Err(Error::InvalidInput(format!(
            "max distance {} bp is smaller than bin size {bin_size}",
            params.max_dist
        )));
    }

    // Trim main diagonal and diagonals >= n_diags, then normalize/downsample.
    let t1 = trim_diags(&m1.triplets, n_diags);
    let t2 = trim_diags(&m2.triplets, n_diags);

    let (a, b) = if params.downsample {
        let s1 = sum(&t1);
        let s2 = sum(&t2);
        if s1 > s2 {
            (resample(&t1, s2.round() as u64)?, t2)
        } else if s2 > s1 {
            (t1, resample(&t2, s1.round() as u64)?)
        } else {
            (t1, t2)
        }
    } else {
        // Scale by genome-wide total. Scale-invariant for SCC, kept for
        // fidelity with the reference.
        (scale(&t1, 1.0 / n1), scale(&t2, 1.0 / n2))
    };

    let scc = if params.h > 0 {
        let d1 = mean_filter(&a, n, params.h);
        let d2 = mean_filter(&b, n, params.h);
        scc_by_diag_dense(&d1, &d2, n_diags)
    } else {
        scc_by_diag_sparse(&a, &b, n_diags)
    };
    Ok(scc)
}

/// Keep only off-diagonal entries with `0 < (col - row) < n_diags`.
fn trim_diags(t: &[(usize, usize, f64)], n_diags: usize) -> Vec<(usize, usize, f64)> {
    t.iter()
        .filter(|&&(i, j, _)| {
            let d = j - i;
            d > 0 && d < n_diags
        })
        .copied()
        .collect()
}

fn sum(t: &[(usize, usize, f64)]) -> f64 {
    t.iter().map(|&(_, _, v)| v).sum()
}

fn scale(t: &[(usize, usize, f64)], f: f64) -> Vec<(usize, usize, f64)> {
    t.iter().map(|&(i, j, v)| (i, j, v * f)).collect()
}

/// Multinomial resample of the pixel counts to a target total (`resample` in
/// hicrep). Positions and their relative weights are preserved.
///
/// # ponytail: O(size) draw loop — billions of contacts make this slow; the
/// reference has the same cost. Replace with a Poisson/gamma approximation if
/// throughput matters.
fn resample(t: &[(usize, usize, f64)], size: u64) -> Result<Vec<(usize, usize, f64)>> {
    if size == 0 || t.is_empty() {
        return Ok(t.iter().map(|&(i, j, _)| (i, j, 0.0)).collect());
    }
    let weights: Vec<f64> = t.iter().map(|&(_, _, v)| v).collect();
    let dist = WeightedIndex::new(weights.iter().copied())
        .map_err(|e| Error::InvalidInput(format!("cannot resample (all-zero weights?): {e}")))?;
    let mut rng = rand::rng();
    let mut counts = vec![0u64; t.len()];
    for _ in 0..size {
        counts[dist.sample(&mut rng)] += 1;
    }
    Ok(t.iter()
        .zip(counts)
        .map(|(&(i, j, _), c)| (i, j, c as f64))
        .collect())
}

/// Box-mean filter with zero padding: each cell is the sum of its `(2h+1)²`
/// neighbourhood divided by the number of in-bounds neighbours. Matches
/// `hicrep.utils.meanFilterSparse`.
fn mean_filter(t: &[(usize, usize, f64)], n: usize, h: usize) -> Array2<f64> {
    let mut out = Array2::zeros((n, n));
    for &(i, j, v) in t {
        let rlo = i.saturating_sub(h);
        let rhi = (i + h + 1).min(n);
        let clo = j.saturating_sub(h);
        let chi = (j + h + 1).min(n);
        for r in rlo..rhi {
            for c in clo..chi {
                out[[r, c]] += v;
            }
        }
    }
    let hh = h as i64;
    let n_i64 = n as i64;
    for i in 0..n {
        let nrow = (i as i64).min(hh) + (n_i64 - 1 - i as i64).min(hh) + 1;
        for j in 0..n {
            let ncol = (j as i64).min(hh) + (n_i64 - 1 - j as i64).min(hh) + 1;
            out[[i, j]] /= (nrow * ncol) as f64;
        }
    }
    out
}

/// Running sums for a Pearson correlation over a set of `(x, y)` pairs.
#[derive(Default, Clone, Copy)]
struct Sums {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Sums {
    fn add(&mut self, x: f64, y: f64) {
        if x != 0.0 || y != 0.0 {
            self.n += 1.0;
            self.sx += x;
            self.sy += y;
            self.sxx += x * x;
            self.syy += y * y;
            self.sxy += x * y;
        }
    }

    /// Pearson r, or `None` when under-determined / degenerate.
    fn rho(&self) -> Option<f64> {
        if self.n < 2.0 {
            return None;
        }
        let cov = self.sxy - self.sx * self.sy / self.n;
        let varx = self.sxx - self.sx * self.sx / self.n;
        let vary = self.syy - self.sy * self.sy / self.n;
        let denom = (varx * vary).sqrt();
        if denom == 0.0 || !denom.is_finite() {
            None
        } else {
            Some((cov / denom).clamp(-1.0, 1.0))
        }
    }
}

/// Variance-stabilizing weight for a diagonal with `n` samples
/// (`(n + 1) / 12`), matching `hicrep.utils.varVstran`.
fn var_vstran_weight(n: f64) -> f64 {
    (n + 1.0) / 12.0
}

/// Diagonal-wise SCC over sparse matrices (`h == 0`).
fn scc_by_diag_sparse(a: &[(usize, usize, f64)], b: &[(usize, usize, f64)], n_diags: usize) -> f64 {
    let ba = bucket_by_diag(a, n_diags);
    let bb = bucket_by_diag(b, n_diags);
    weighted_scc(n_diags, |d| diagonal_sums(&ba[d], &bb[d]))
}

/// Diagonal-wise SCC over dense (already mean-filtered) matrices.
fn scc_by_diag_dense(a: &Array2<f64>, b: &Array2<f64>, n_diags: usize) -> f64 {
    let n = a.nrows();
    weighted_scc(n_diags, |d| {
        let mut s = Sums::default();
        if d < n {
            for k in 0..(n - d) {
                s.add(a[[k, k + d]], b[[k, k + d]]);
            }
        }
        s
    })
}

/// Shared tail of the SCC computation: weight each diagonal's Pearson r by
/// `var_vstran_weight(n)` and average.
fn weighted_scc<F>(n_diags: usize, mut diagonal: F) -> f64
where
    F: FnMut(usize) -> Sums,
{
    let mut num = 0.0;
    let mut den = 0.0;
    for d in 1..n_diags {
        let s = diagonal(d);
        if s.n < 2.0 {
            continue;
        }
        let ws = var_vstran_weight(s.n);
        let rho = s.rho().unwrap_or(0.0);
        num += rho * ws;
        den += ws;
    }
    if den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

/// Group triplets by diagonal offset `d = col - row`, each bucket sorted by
/// row index.
fn bucket_by_diag(t: &[(usize, usize, f64)], n_diags: usize) -> Vec<Vec<(usize, f64)>> {
    let mut buckets: Vec<Vec<(usize, f64)>> = (0..n_diags).map(|_| Vec::new()).collect();
    for &(i, j, v) in t {
        let d = j - i;
        if d > 0 && d < n_diags {
            buckets[d].push((i, v));
        }
    }
    for b in &mut buckets {
        b.sort_unstable_by_key(|&(k, _)| k);
    }
    buckets
}

/// Pearson sums over one diagonal, merging the two matrices' sparse values by
/// row index.
fn diagonal_sums(x: &[(usize, f64)], y: &[(usize, f64)]) -> Sums {
    let mut map: HashMap<usize, f64> = x.iter().copied().collect();
    let mut s = Sums::default();
    for &(k, v) in y {
        let vx = map.remove(&k).unwrap_or(0.0);
        s.add(vx, v);
    }
    for (_, vx) in map {
        s.add(vx, 0.0);
    }
    s
}

/// Whole-matrix Pearson over the flattened upper triangle (off-diagonal,
/// common zeros excluded).
fn pearson_chrom(m1: &ChromMat, m2: &ChromMat) -> f64 {
    let (x, y) = collect_offdiag(&m1.triplets, &m2.triplets);
    pearson(&x, &y)
}

/// Whole-matrix Spearman rank correlation over the flattened upper triangle.
fn spearman_chrom(m1: &ChromMat, m2: &ChromMat) -> f64 {
    let (x, y) = collect_offdiag(&m1.triplets, &m2.triplets);
    let rx = ranks(&x);
    let ry = ranks(&y);
    pearson(&rx, &ry)
}

/// Flatten off-diagonal upper-triangle positions into paired value vectors,
/// keeping positions nonzero in at least one matrix.
fn collect_offdiag(t1: &[(usize, usize, f64)], t2: &[(usize, usize, f64)]) -> (Vec<f64>, Vec<f64>) {
    let mut map: HashMap<(usize, usize), f64> = HashMap::new();
    for &(i, j, v) in t1 {
        if j > i {
            map.insert((i, j), v);
        }
    }
    let mut x = Vec::new();
    let mut y = Vec::new();
    for &(i, j, v) in t2 {
        if j <= i {
            continue;
        }
        match map.remove(&(i, j)) {
            Some(v1) => {
                x.push(v1);
                y.push(v);
            }
            None => {
                x.push(0.0);
                y.push(v);
            }
        }
    }
    for ((_, _), v1) in map {
        x.push(v1);
        y.push(0.0);
    }
    (x, y)
}

/// Pearson correlation of two equal-length slices.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mut s = Sums::default();
    for (&a, &b) in x.iter().zip(y) {
        s.add(a, b);
    }
    s.rho().unwrap_or(f64::NAN)
}

/// Average ranks (1-based) of `v`, ties share their average rank.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            out[k] = rank;
        }
        i = j + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_filter_single_pixel() {
        // n=2, h=1: a single pixel's window is the whole 2×2, every cell
        // gets 1.0 / (2*2) neighbours.
        let t = vec![(0usize, 0usize, 1.0)];
        let out = mean_filter(&t, 2, 1);
        for r in 0..2 {
            for c in 0..2 {
                assert!((out[[r, c]] - 0.25).abs() < 1e-12, "cell ({r},{c})");
            }
        }
    }

    #[test]
    fn ranks_average_ties() {
        assert_eq!(ranks(&[2.0, 2.0, 8.0]), vec![1.5, 1.5, 3.0]);
        assert_eq!(ranks(&[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn var_vstran_weight_formula() {
        assert!((var_vstran_weight(2.0) - 3.0 / 12.0).abs() < 1e-15);
        assert!((var_vstran_weight(3.0) - 4.0 / 12.0).abs() < 1e-15);
    }
}
