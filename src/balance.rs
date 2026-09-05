//! Iterative correction / matrix balancing of a sparse Hi-C contact map.
//!
//! Rust port of `cooler.balance.balance_cooler` (cooler-python): the
//! Sinkhorn–Knopp-style IC algorithm over chunked pixel streams, with the
//! same pre-filters (min-nnz, min-count, per-chromosome MAD-max, blacklist)
//! and the same genome-wide / cis-only / trans-only modes.

use crate::cooler::Cooler;
use crate::error::Result;

/// Parameters for [`balance_cooler`], mirroring the defaults of the Python
/// `cooler.balance.balance_cooler`.
#[derive(Debug, Clone)]
pub struct BalanceParams {
    /// Balance against intra-chromosomal data only (inter-chromosomal reads
    /// are ignored).
    pub cis_only: bool,
    /// Balance against inter-chromosomal data only (intra-chromosomal reads
    /// are ignored).
    pub trans_only: bool,
    /// Drop pixels on the first `ignore_diags` diagonals (including the main
    /// diagonal).
    pub ignore_diags: usize,
    /// Drop bins whose log marginal sum is more than `mad_max` median absolute
    /// deviations below the median log marginal sum of their chromosome.
    pub mad_max: usize,
    /// Drop bins with fewer than `min_nnz` nonzero elements.
    pub min_nnz: usize,
    /// Drop bins with marginal sum below `min_count`.
    pub min_count: usize,
    /// Indices of bins to mask out (e.g. from a blacklist BED).
    pub blacklist: Vec<usize>,
    /// Normalize the weights so balanced marginals sum to 1.0.
    pub rescale_marginals: bool,
    /// Initial weight vector. Defaults to ones.
    pub x0: Option<Vec<f64>>,
    /// Convergence threshold on the variance of the nonzero marginals.
    pub tol: f64,
    /// Iteration limit.
    pub max_iters: usize,
    /// Number of pixels per chunk read into memory at once.
    pub chunksize: usize,
}

impl Default for BalanceParams {
    fn default() -> Self {
        BalanceParams {
            cis_only: false,
            trans_only: false,
            ignore_diags: 2,
            mad_max: 5,
            min_nnz: 10,
            min_count: 0,
            blacklist: Vec::new(),
            rescale_marginals: true,
            x0: None,
            tol: 1e-5,
            max_iters: 200,
            chunksize: 10_000_000,
        }
    }
}

/// Summary of a balancing run, mirroring the Python `stats` dict.
///
/// For genome-wide and trans-only runs the `scale`/`var`/`converged` vectors
/// have length 1; for cis-only runs they hold one entry per chromosome.
#[derive(Debug, Clone)]
pub struct BalanceStats {
    /// Convergence threshold used.
    pub tol: f64,
    /// `min_nnz` used.
    pub min_nnz: usize,
    /// `min_count` used.
    pub min_count: usize,
    /// `mad_max` used.
    pub mad_max: usize,
    /// Whether the run was cis-only.
    pub cis_only: bool,
    /// `ignore_diags` used.
    pub ignore_diags: usize,
    /// Average magnitude of the balanced matrix's marginal sum at convergence.
    pub scale: Vec<f64>,
    /// Marginal-sum variance at convergence (per chromosome for cis-only).
    pub var: Vec<f64>,
    /// Whether the variance reached `tol` (per chromosome for cis-only).
    pub converged: Vec<bool>,
    /// Always false (no divisive weights in this algorithm).
    pub divisive_weights: bool,
}

/// Partition `[start, stop)` into equally sized subintervals, like Python's
/// `range`/`partition`.
fn partition(start: usize, stop: usize, step: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = start;
    while i < stop {
        out.push((i, (i + step).min(stop)));
        i += step;
    }
    out
}

/// Accumulate the marginal (row+col) sum of a slice of pixels into `marg`,
/// applying the requested filters and an optional bias outer product.
///
/// Mirrors the Python chunk pipeline: `_init` → `_binarize` → `_zero_trans` /
/// `_zero_cis` / `_zero_diags` → `_timesouterproduct` → `_marginalize`.
/// Pixel streams are read in stored order, so summation order matches Python.
#[allow(clippy::too_many_arguments)]
fn accumulate_marginal(
    clr: &Cooler,
    bin_chrom: &[i32],
    spans: &[(usize, usize)],
    binarize: bool,
    zero_trans: bool,
    zero_cis: bool,
    ignore_diags: usize,
    bias: Option<&[f64]>,
    marg: &mut [f64],
) -> Result<()> {
    for &(lo, hi) in spans {
        for p in clr.pixels_range(lo as i64, hi as i64)? {
            let b1 = p.bin1_id as usize;
            let b2 = p.bin2_id as usize;
            if ignore_diags > 0 && b1.abs_diff(b2) < ignore_diags {
                continue;
            }
            if zero_trans && bin_chrom[b1] != bin_chrom[b2] {
                continue;
            }
            if zero_cis && bin_chrom[b1] == bin_chrom[b2] {
                continue;
            }
            let mut v = if binarize && p.count != 0.0 {
                1.0
            } else {
                p.count
            };
            if let Some(bias) = bias {
                v *= bias[b1] * bias[b2];
            }
            marg[b1] += v;
            marg[b2] += v;
        }
    }
    Ok(())
}

/// Mean and biased variance of the nonzero entries of `marg`.
fn nz_mean_var(marg: &[f64]) -> Option<(f64, f64)> {
    let nz: Vec<f64> = marg.iter().copied().filter(|&m| m != 0.0).collect();
    if nz.is_empty() {
        return None;
    }
    let mean = nz.iter().sum::<f64>() / nz.len() as f64;
    let var = nz
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / nz.len() as f64;
    Some((mean, var))
}

// Parameter count mirrors the Python `_balance_*` signatures.
#[allow(clippy::too_many_arguments)]
fn converge(
    clr: &Cooler,
    bin_chrom: &[i32],
    spans: &[(usize, usize)],
    zero_trans: bool,
    zero_cis: bool,
    ignore_diags: usize,
    bias: &mut [f64],
    weights: Option<&[f64]>,
    tol: f64,
    max_iters: usize,
) -> Result<(f64, f64, bool)> {
    let n_bins = bias.len();
    let mut marg = vec![0.0; n_bins];
    let mut scale = 1.0;
    let mut var = 0.0;
    let mut converged = false;
    // Rebuild the outer-product vector once per iteration, since `bias`
    // mutates. For trans-only, `weights` is the constant `cweights`.
    let mut outer = vec![0.0; n_bins];
    for _ in 0..max_iters {
        marg.iter_mut().for_each(|m| *m = 0.0);
        if let Some(w) = weights {
            outer.copy_from_slice(w);
            for i in 0..n_bins {
                outer[i] *= bias[i];
            }
            accumulate_marginal(
                clr,
                bin_chrom,
                spans,
                false,
                zero_trans,
                zero_cis,
                ignore_diags,
                Some(&outer),
                &mut marg,
            )?;
        } else {
            accumulate_marginal(
                clr,
                bin_chrom,
                spans,
                false,
                zero_trans,
                zero_cis,
                ignore_diags,
                Some(bias),
                &mut marg,
            )?;
        }
        match nz_mean_var(&marg) {
            None => {
                bias.fill(f64::NAN);
                return Ok((f64::NAN, 0.0, false));
            }
            Some((mean, v)) => {
                for m in marg.iter_mut() {
                    *m /= mean;
                    if *m == 0.0 {
                        *m = 1.0;
                    }
                }
                for (b, m) in bias.iter_mut().zip(marg.iter()) {
                    *b /= m;
                }
                scale = mean;
                var = v;
                if v < tol {
                    converged = true;
                    break;
                }
            }
        }
    }
    Ok((scale, var, converged))
}

/// Genome-wide iterative correction (`_balance_genomewide`).
#[allow(clippy::too_many_arguments)]
fn balance_genomewide(
    clr: &Cooler,
    bin_chrom: &[i32],
    spans: &[(usize, usize)],
    zero_trans: bool,
    ignore_diags: usize,
    tol: f64,
    max_iters: usize,
    rescale_marginals: bool,
    bias: &mut [f64],
) -> Result<(f64, f64, bool)> {
    let (scale, var, converged) = converge(
        clr,
        bin_chrom,
        spans,
        zero_trans,
        false,
        ignore_diags,
        bias,
        None,
        tol,
        max_iters,
    )?;
    if !scale.is_nan() {
        for b in bias.iter_mut() {
            if *b == 0.0 {
                *b = f64::NAN;
            } else if rescale_marginals {
                *b /= scale.sqrt();
            }
        }
    }
    Ok((scale, var, converged))
}

/// Trans-only iterative correction (`_balance_transonly`).
#[allow(clippy::too_many_arguments)]
fn balance_transonly(
    clr: &Cooler,
    bin_chrom: &[i32],
    spans: &[(usize, usize)],
    chrom_offset: &[i64],
    ignore_diags: usize,
    tol: f64,
    max_iters: usize,
    rescale_marginals: bool,
    bias: &mut [f64],
) -> Result<(f64, f64, bool)> {
    let n_bins = bias.len();
    let mut cweights = vec![0.0; n_bins];
    for w in chrom_offset.windows(2) {
        let (lo, hi) = (w[0] as usize, w[1] as usize);
        let wgt = 1.0 / (1.0 - (hi - lo) as f64 / n_bins as f64);
        cweights[lo..hi].fill(wgt);
    }
    let (scale, var, converged) = converge(
        clr,
        bin_chrom,
        spans,
        false,
        true,
        ignore_diags,
        bias,
        Some(&cweights),
        tol,
        max_iters,
    )?;
    if !scale.is_nan() {
        for b in bias.iter_mut() {
            if *b == 0.0 {
                *b = f64::NAN;
            } else if rescale_marginals {
                *b /= scale.sqrt();
            }
        }
    }
    Ok((scale, var, converged))
}

/// Cis-only iterative correction, one chromosome at a time
/// (`_balance_cisonly`). Returns per-chromosome scale, variance and
/// convergence flags.
#[allow(clippy::too_many_arguments)]
fn balance_cisonly(
    clr: &Cooler,
    bin_chrom: &[i32],
    chrom_offset: &[i64],
    bin1_offset: &[i64],
    chunksize: usize,
    ignore_diags: usize,
    tol: f64,
    max_iters: usize,
    rescale_marginals: bool,
    bias: &mut [f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<bool>)> {
    let n_chroms = chrom_offset.len() - 1;
    let mut scales = vec![0.0; n_chroms];
    let mut variances = vec![0.0; n_chroms];
    let mut converged = vec![false; n_chroms];
    let n_bins = bias.len();

    for cid in 0..n_chroms {
        let (lo, hi) = (chrom_offset[cid] as usize, chrom_offset[cid + 1] as usize);
        let spans = partition(
            bin1_offset[lo] as usize,
            bin1_offset[hi] as usize,
            chunksize,
        );
        let mut marg = vec![0.0; n_bins];
        let mut scale = 1.0;
        let mut var = 0.0;
        for _ in 0..max_iters {
            marg.iter_mut().for_each(|m| *m = 0.0);
            accumulate_marginal(
                clr,
                bin_chrom,
                &spans,
                false,
                true, // cis_only implies zero_trans
                false,
                ignore_diags,
                Some(bias),
                &mut marg,
            )?;
            let marg_c = &mut marg[lo..hi];
            match nz_mean_var(marg_c) {
                None => {
                    bias[lo..hi].fill(f64::NAN);
                    scale = f64::NAN;
                    var = 0.0;
                    break;
                }
                Some((mean, v)) => {
                    for m in marg_c.iter_mut() {
                        *m /= mean;
                        if *m == 0.0 {
                            *m = 1.0;
                        }
                    }
                    for (b, m) in bias[lo..hi].iter_mut().zip(marg_c.iter()) {
                        *b /= m;
                    }
                    scale = mean;
                    var = v;
                    if v < tol {
                        break;
                    }
                }
            }
        }
        // Python records `var < tol` per chromosome, including the degenerate
        // all-zero case (var = 0.0, so reported converged).
        converged[cid] = var < tol;
        let b = &mut bias[lo..hi];
        for x in b.iter_mut() {
            if *x == 0.0 {
                *x = f64::NAN;
            } else if rescale_marginals && !scale.is_nan() {
                *x /= scale.sqrt();
            }
        }
        scales[cid] = scale;
        variances[cid] = var;
    }
    Ok((scales, variances, converged))
}

/// Iterative correction / matrix balancing of a sparse Hi-C contact map.
///
/// Returns the bin bias vector (`N[i, j] = O[i, j] * bias[i] * bias[j]`;
/// dropped bins are `NaN`) and a [`BalanceStats`] summary. See
/// [`BalanceParams`] for the tunables.
pub fn balance_cooler(clr: &Cooler, p: &BalanceParams) -> Result<(Vec<f64>, BalanceStats)> {
    let nnz = clr.n_pixels()? as usize;
    let spans = partition(0, nnz, p.chunksize);
    let n_bins = clr.bins()?.len();
    let bin_chrom = clr.bin_chrom()?;

    let mut bias = match &p.x0 {
        Some(x0) => {
            let mut v = x0.clone();
            for x in v.iter_mut() {
                if x.is_nan() {
                    *x = 0.0;
                }
            }
            v
        }
        None => vec![1.0; n_bins],
    };

    let zero_trans = p.cis_only;
    let mut marg = vec![0.0; n_bins];

    // Drop bins with too few nonzeros.
    if p.min_nnz > 0 {
        marg.iter_mut().for_each(|m| *m = 0.0);
        accumulate_marginal(
            clr,
            &bin_chrom,
            &spans,
            true, // binarize
            zero_trans,
            false,
            p.ignore_diags,
            None,
            &mut marg,
        )?;
        for (b, &m) in bias.iter_mut().zip(marg.iter()) {
            if m < p.min_nnz as f64 {
                *b = 0.0;
            }
        }
    }

    // Marginal sums used by min_count and MAD-max.
    marg.iter_mut().for_each(|m| *m = 0.0);
    accumulate_marginal(
        clr,
        &bin_chrom,
        &spans,
        false,
        zero_trans,
        false,
        p.ignore_diags,
        None,
        &mut marg,
    )?;

    // Drop bins with too few total counts.
    if p.min_count > 0 {
        for (b, &m) in bias.iter_mut().zip(marg.iter()) {
            if m < p.min_count as f64 {
                *b = 0.0;
            }
        }
    }

    // MAD-max filter: normalize marginals by the median of each chromosome,
    // then drop bins far below the median log marginal.
    if p.mad_max > 0 {
        let chrom_offset = clr.chrom_offset()?;
        let mut norm_marg = marg.clone();
        for w in chrom_offset.windows(2) {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let c_marg = &norm_marg[lo..hi];
            let pos: Vec<f64> = c_marg.iter().copied().filter(|&m| m > 0.0).collect();
            let med = median(&pos);
            for m in norm_marg[lo..hi].iter_mut() {
                *m /= med;
            }
        }
        let pos: Vec<f64> = norm_marg.iter().copied().filter(|&m| m > 0.0).collect();
        let log_pos: Vec<f64> = pos.iter().map(|m| m.ln()).collect();
        let med_log = median(&log_pos);
        let mad = log_pos
            .iter()
            .map(|l| (l - med_log).abs())
            .collect::<Vec<f64>>();
        let med_dev = median(&mad);
        let cutoff = (med_log - p.mad_max as f64 * med_dev).exp();
        for (b, &m) in bias.iter_mut().zip(marg.iter()) {
            if m < cutoff {
                *b = 0.0;
            }
        }
    }

    // Explicitly masked bins.
    for &b in &p.blacklist {
        bias[b] = 0.0;
    }

    let stats = if p.cis_only {
        let chrom_offset = clr.chrom_offset()?;
        let bin1_offset = clr.bin1_offset()?;
        let (scales, variances, converged) = balance_cisonly(
            clr,
            &bin_chrom,
            &chrom_offset,
            &bin1_offset,
            p.chunksize,
            p.ignore_diags,
            p.tol,
            p.max_iters,
            p.rescale_marginals,
            &mut bias,
        )?;
        BalanceStats {
            tol: p.tol,
            min_nnz: p.min_nnz,
            min_count: p.min_count,
            mad_max: p.mad_max,
            cis_only: true,
            ignore_diags: p.ignore_diags,
            scale: scales,
            var: variances,
            converged,
            divisive_weights: false,
        }
    } else {
        let (scale, var, converged) = if p.trans_only {
            let chrom_offset = clr.chrom_offset()?;
            balance_transonly(
                clr,
                &bin_chrom,
                &spans,
                &chrom_offset,
                p.ignore_diags,
                p.tol,
                p.max_iters,
                p.rescale_marginals,
                &mut bias,
            )?
        } else {
            balance_genomewide(
                clr,
                &bin_chrom,
                &spans,
                zero_trans,
                p.ignore_diags,
                p.tol,
                p.max_iters,
                p.rescale_marginals,
                &mut bias,
            )?
        };
        BalanceStats {
            tol: p.tol,
            min_nnz: p.min_nnz,
            min_count: p.min_count,
            mad_max: p.mad_max,
            cis_only: false,
            ignore_diags: p.ignore_diags,
            scale: vec![scale],
            var: vec![var],
            converged: vec![converged],
            divisive_weights: false,
        }
    };

    Ok((bias, stats))
}

/// Median of a slice. Mirrors `np.median` for odd lengths; the Python MAD-max
/// code relies on the even-length median (mean of the two middle values).
pub(crate) fn median(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let mut v = sorted.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
