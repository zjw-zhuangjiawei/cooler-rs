//! Boundary detection for the `hicFindTADs` port: local minima of the mean
//! TAD-separation score, the delta and p-value filters, and the multiple
//! testing correction.

use std::collections::HashMap;
use std::f64::consts::SQRT_2;

use super::matrix::diamond_at;
use super::Prepared;

/// One row of the TAD-separation score table: a bin and one score per window.
#[derive(Debug, Clone)]
pub struct ScoreRow {
    /// Chromosome name.
    pub chrom: String,
    /// Position of the bin within the chromosome, which is what the diamonds
    /// and the p-values are computed from. Rows whose diamonds held a `NaN`
    /// are missing from the table, so this is not the row index.
    pub bin: usize,
    /// Bin start.
    pub start: i64,
    /// Bin end.
    pub end: i64,
    /// TAD-separation score at each window, in the order the windows were
    /// generated.
    pub values: Vec<f64>,
}

/// The TAD-separation score table, kept in bin order. Bins whose diamonds
/// held a `NaN` are absent, exactly as in the original.
#[derive(Debug, Clone, Default)]
pub struct ScoreTable {
    /// One entry per scored bin.
    pub rows: Vec<ScoreRow>,
}

impl ScoreTable {
    /// Number of scored bins.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no bin produced a score.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The mean score per row, the signal the minima are found in.
    pub fn row_means(&self) -> Vec<f64> {
        self.rows
            .iter()
            .map(|r| r.values.iter().sum::<f64>() / r.values.len() as f64)
            .collect()
    }

    /// Chromosome names, one per row.
    pub fn chroms(&self) -> Vec<&str> {
        self.rows.iter().map(|r| r.chrom.as_str()).collect()
    }
}

/// A peak position and the value at it.
type Peak = (usize, f64);

/// `HicFindTads.peakdetect` (the Billauer peak detector), with `delta = 0`.
///
/// Returns the maximum and minimum candidates as `(index, value)`, in the
/// order they were found. State restarts at every chromosome change, and the
/// first hit is always dropped as a false positive.
pub fn peakdetect(y: &[f64], lookahead: usize, chrom: &[&str]) -> (Vec<Peak>, Vec<Peak>) {
    let mut max_peaks: Vec<Peak> = Vec::new();
    let mut min_peaks: Vec<Peak> = Vec::new();
    let mut dump: Vec<bool> = Vec::new();

    let n = y.len().saturating_sub(lookahead);
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_pos = 0usize;
    let mut min_pos = 0usize;
    let mut search_for: Option<bool> = None; // Some(true) = looking for a max
    let mut prev_chrom: Option<&str> = None;

    for index in 0..n {
        let value = y[index];
        debug_assert!(value.is_finite(), "infinity value at position {index}");

        if prev_chrom != Some(chrom[index]) {
            min_y = f64::INFINITY;
            max_y = f64::NEG_INFINITY;
            search_for = None;
        }
        prev_chrom = Some(chrom[index]);

        if value > max_y {
            max_y = value;
            max_pos = index;
        }
        if value < min_y {
            min_y = value;
            min_pos = index;
        }

        if value < max_y && max_y != f64::INFINITY && search_for != Some(false) {
            let window_max = y[index..index + lookahead]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if window_max < max_y {
                max_peaks.push((max_pos, max_y));
                dump.push(true);
                max_y = value;
                min_y = value;
                min_pos = index;
                search_for = Some(false);
                continue;
            }
        }

        if value > min_y && min_y != f64::NEG_INFINITY && search_for != Some(true) {
            let window_min = y[index..index + lookahead]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            if window_min > min_y {
                min_peaks.push((min_pos, min_y));
                dump.push(false);
                min_y = value;
                max_y = value;
                max_pos = index;
                search_for = Some(true);
            }
        }
    }

    match dump.first() {
        Some(true) => {
            max_peaks.remove(0);
        }
        Some(false) => {
            min_peaks.remove(0);
        }
        None => {}
    }

    (max_peaks, min_peaks)
}

/// `HicFindTads.delta_wrt_window`: how far below the surrounding bins each
/// minimum sits.
///
/// The neighbourhood is the ten bins either side of the minimum, minus the
/// three to its right that the original skips. Minima too close to a
/// chromosome edge come back as `NaN`; the ranges are built from
/// first-occurrence indices, so they partition the table rather than
/// following the chromosome names.
pub fn delta_wrt_window(
    min_idx: &[usize],
    matrix_avg: &[f64],
    chrom: &[&str],
    window_len: usize,
) -> HashMap<usize, f64> {
    let n = chrom.len();
    let mut starts: Vec<usize> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (i, name) in chrom.iter().enumerate() {
        if !seen.contains(name) {
            seen.push(name);
            starts.push(i);
        }
    }
    starts.push(n.saturating_sub(1));
    starts.sort_unstable();
    let ranges: Vec<(usize, usize)> = starts.windows(2).map(|w| (w[0], w[1])).collect();

    let mut out = HashMap::new();
    for &min in min_idx {
        let mut close_to_border = true;
        for &(start_range, end_range) in &ranges {
            if start_range < min
                && min < end_range
                && min >= window_len + start_range
                && min + window_len < end_range
            {
                close_to_border = false;
                continue;
            }
        }
        if close_to_border {
            out.insert(min, f64::NAN);
            continue;
        }
        let lo = min - window_len;
        let mut local: Vec<f64> = matrix_avg[lo..min + 3].to_vec();
        local.extend_from_slice(&matrix_avg[min + 4..min + window_len]);
        let mean = local.iter().sum::<f64>() / local.len() as f64;
        out.insert(min, mean - matrix_avg[min]);
    }
    out
}

/// `HicFindTads.find_consensus_minima`: the local minima of the mean score
/// and their deltas.
pub fn find_consensus_minima(
    table: &ScoreTable,
    lookahead: usize,
) -> (Vec<usize>, HashMap<usize, f64>) {
    let avg = table.row_means();
    let chrom = table.chroms();
    let (_, minima) = peakdetect(&avg, lookahead, &chrom);
    let min_idx: Vec<usize> = minima.iter().map(|&(pos, _)| pos).collect();
    let delta_to_mean = delta_wrt_window(&min_idx, &avg, &chrom, 10);
    (min_idx, delta_to_mean)
}

/// `scipy.stats.rankdata` with the default average method; `NaN` sorts last
/// and forms a group of its own, since `NaN != NaN`.
fn rankdata(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| match (values[a].is_nan(), values[b].is_nan()) {
        (true, true) => a.cmp(&b),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => values[a].partial_cmp(&values[b]).unwrap(),
    });

    let sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    // Group boundaries: the first element always starts a group, then every
    // step where the value changes. A `NaN` always starts one, since
    // `NaN != NaN`.
    let mut group_start: Vec<usize> = vec![0];
    for (k, pair) in sorted.windows(2).enumerate() {
        if !(pair[1] == pair[0]) {
            group_start.push(k + 1);
        }
    }

    let mut group_of = vec![0usize; sorted.len()];
    let mut group = 0usize;
    for (k, slot) in group_of.iter_mut().enumerate() {
        if group < group_start.len() && group_start[group] == k {
            group += 1;
        }
        *slot = group;
    }

    let mut bound = group_start.clone();
    bound.push(sorted.len());

    let mut ranks = vec![0.0; values.len()];
    for (rank, &original) in order.iter().enumerate() {
        let g = group_of[rank];
        ranks[original] = 0.5 * (bound[g] as f64 + bound[g - 1] as f64 + 1.0);
    }
    ranks
}

/// `scipy.stats.ranksums` — the two-sided Mann-Whitney p-value from the
/// normal approximation, with no tie correction.
pub fn ranksums(x: &[f64], y: &[f64]) -> f64 {
    let (n1, n2) = (x.len(), y.len());
    let mut all = Vec::with_capacity(n1 + n2);
    all.extend_from_slice(x);
    all.extend_from_slice(y);
    let ranked = rankdata(&all);

    let s: f64 = ranked[..n1].iter().sum();
    let expected = n1 as f64 * (n1 + n2 + 1) as f64 / 2.0;
    let spread = (n1 as f64 * n2 as f64 * (n1 + n2 + 1) as f64 / 12.0).sqrt();
    let z = (s - expected) / spread;
    libm::erfc(z.abs() / SQRT_2)
}

/// `HicFindTads.min_pvalue` and the multiple-testing correction that follows
/// it.
///
/// Each minimum is compared with the diamonds `min_depth` bp to its left and
/// right, and the smaller of the two Wilcoxon p-values wins. The FDR cutoff
/// is the largest p-value that clears Benjamini-Hochberg; `NaN` p-values are
/// replaced by 1 before the scan for FDR, but not for Bonferroni.
pub fn min_pvalues(
    table: &ScoreTable,
    prepared: &Prepared,
    min_idx: &[usize],
    window_len: i64,
    correction: super::MultipleTesting,
    threshold: f64,
) -> (Vec<usize>, HashMap<usize, f64>, f64) {
    let mut new_min_idx = Vec::with_capacity(min_idx.len());
    let mut pvalues: Vec<f64> = Vec::with_capacity(min_idx.len());

    for &idx in min_idx {
        let row = &table.rows[idx];
        let slot = prepared
            .chrom_bins
            .iter()
            .position(|b| b.name == row.chrom)
            .expect("score row outside the prepared chromosomes");
        let bins = &prepared.chrom_bins[slot];
        let band = &prepared.bands[slot];
        let cut = row.bin;
        new_min_idx.push(idx);

        let (left_idx, right_idx) = super::matrix::neighbour_bins(bins, cut, window_len);
        let left = diamond_at(band, bins, left_idx, window_len);
        let right = diamond_at(band, bins, right_idx, window_len);
        let boundary = diamond_at(band, bins, cut, window_len);

        // An empty diamond on either side leaves nothing to compare, and the
        // source's own `ValueError` path lands on the same answer.
        let pval = if boundary.is_empty() || left.is_empty() || right.is_empty() {
            f64::NAN
        } else {
            ranksums(&boundary, &left).min(ranksums(&boundary, &right))
        };
        pvalues.push(pval);
    }

    let mut fdr_cutoff = 0.0;
    let values: Vec<f64> = match correction {
        super::MultipleTesting::Fdr => {
            // Benjamini-Hochberg: the cutoff is the largest p-value that
            // clears its own rank, and a missing p-value counts as 1.
            let values: Vec<f64> = pvalues
                .iter()
                .map(|p| if p.is_nan() { 1.0 } else { *p })
                .collect();
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, p) in sorted.iter().enumerate() {
                if *p <= threshold * (i + 1) as f64 / sorted.len() as f64 && *p >= fdr_cutoff {
                    fdr_cutoff = *p;
                }
            }
            values
        }
        super::MultipleTesting::Bonferroni => {
            let scale = pvalues.len() as f64;
            pvalues
                .iter()
                .map(|p| {
                    if p.is_nan() {
                        f64::NAN
                    } else {
                        (p * scale).min(1.0)
                    }
                })
                .collect()
        }
        super::MultipleTesting::None => pvalues,
    };

    let result: HashMap<usize, f64> = new_min_idx.iter().copied().zip(values).collect();
    (new_min_idx, result, fdr_cutoff)
}

/// The cutoffs the boundary filter applies.
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// Which correction the p-values went through.
    pub correction: super::MultipleTesting,
    /// p-value (Bonferroni) or q-value (FDR) cutoff.
    pub threshold: f64,
    /// Benjamini-Hochberg cutoff, from [`min_pvalues`].
    pub fdr_cutoff: f64,
    /// Minimum delta for a minimum to count as a boundary.
    pub delta: f64,
}

/// `HicFindTads.save_domains_and_boundaries`' filter step.
///
/// Chromosome starts and ends are folded in as extra candidates; they never
/// carry a p-value, so they drop out again, but they do shift the order that
/// domains are assembled in.
pub fn filter_boundaries(
    table: &ScoreTable,
    minima: &[usize],
    delta_of_min: &HashMap<usize, f64>,
    pvalue_of_min: &HashMap<usize, f64>,
    thresholds: Thresholds,
) -> (Vec<usize>, Vec<usize>) {
    let chrom = table.chroms();
    let n = chrom.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut starts: Vec<usize> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (i, name) in chrom.iter().enumerate() {
        if !seen.contains(name) {
            seen.push(name);
            starts.push(i);
        }
    }

    // A single chromosome takes a separate branch; otherwise `chr_end_idx` is
    // the same array as `chr_start_idx`, so both end up holding the "last bin
    // before each chromosome start" indices.
    let (starts, ends) = if seen.len() == 1 {
        (vec![0usize], vec![n - 1])
    } else {
        let mut ends = starts.clone();
        for v in ends.iter_mut() {
            if *v == 0 {
                *v = n;
            }
            *v -= 1;
        }
        (ends.clone(), ends)
    };

    let mut candidates: Vec<usize> = Vec::with_capacity(starts.len() + ends.len() + minima.len());
    candidates.extend_from_slice(&starts);
    candidates.extend_from_slice(&ends);
    candidates.extend_from_slice(minima);
    candidates.sort_unstable();

    let mut filtered = Vec::new();
    for idx in candidates {
        let delta_value = delta_of_min.get(&idx).copied().unwrap_or(f64::NAN);
        let Some(&pvalue) = pvalue_of_min.get(&idx) else {
            continue;
        };
        // `NaN` deltas (minima at a chromosome edge) never pass.
        if delta_value < thresholds.delta || delta_value.is_nan() {
            continue;
        }
        let passes = match thresholds.correction {
            super::MultipleTesting::Fdr => pvalue <= thresholds.fdr_cutoff,
            super::MultipleTesting::Bonferroni | super::MultipleTesting::None => {
                pvalue <= thresholds.threshold
            }
        };
        if passes {
            filtered.push(idx);
        }
    }
    (filtered, ends)
}
