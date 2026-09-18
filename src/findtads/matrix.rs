//! The matrix side of the `hicFindTADs` port: bin preparation, the z-score
//! transform, and the TAD-separation score.
//!
//! Everything here mirrors `hicmatrix.HiCMatrix`. The one structural
//! difference is storage: `hicmatrix` keeps a CSR matrix that the z-score
//! step fills in densely along the diagonal band, so this port keeps the band
//! itself (`values[d][i]` is the cell `i, i + d`) and reads anything outside
//! it as the `0` a dense matrix would have.

/// The order `np.sum`, `np.mean` and `np.bincount` accumulate floats in.
///
/// numpy splits the array into blocks of eight, keeps eight running sums, and
/// combines them as a tree; arrays longer than 128 elements are halved first.
/// Reproducing that order matters because the TAD-separation scores and the
/// boundary p-values are compared against upstream output bit for bit, and
/// plain summation drifts in the last digits.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut res = 0.0;
        for value in values {
            res += *value;
        }
        return res;
    }
    if n <= BLOCK {
        let mut running = [0.0f64; 8];
        running.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (slot, value) in running.iter_mut().zip(&values[i..i + 8]) {
                *slot += *value;
            }
            i += 8;
        }
        let mut res = ((running[0] + running[1]) + (running[2] + running[3]))
            + ((running[4] + running[5]) + (running[6] + running[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// Bins of one chromosome, after masked bins are dropped and the gaps
/// redistributed. Coordinates are genomic, sorted and non-overlapping.
///
/// After [`enlarge_bins`] every chromosome starts at 0 and consecutive bins
/// touch, which the original relies on when it looks up a bin by position.
#[derive(Debug, Clone)]
pub struct ChromBins {
    /// Chromosome name.
    pub name: String,
    /// Start coordinate of each bin (0-based, inclusive).
    pub start: Vec<i64>,
    /// End coordinate of each bin (0-based, exclusive).
    pub end: Vec<i64>,
}

impl ChromBins {
    /// Number of bins.
    pub fn len(&self) -> usize {
        self.start.len()
    }

    /// Whether the chromosome has no bins left after masking.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }
}

/// `hicexplorer.utilities.enlarge_bins`: pin the first bin of every
/// chromosome to 0 and move the join between two bins that do not touch to
/// the midpoint of the gap.
///
/// Bins dropped by masking leave gaps, and both the position lookup and the
/// bedgraph output assume the remaining bins tile the chromosome.
pub fn enlarge_bins(bins: &mut [ChromBins]) {
    for chrom in bins.iter_mut() {
        for idx in 0..chrom.len().saturating_sub(1) {
            if idx == 0 {
                chrom.start[idx] = 0;
            }
            if chrom.end[idx] != chrom.start[idx + 1] {
                let middle = chrom.start[idx + 1] - (chrom.start[idx + 1] - chrom.end[idx]) / 2;
                chrom.end[idx] = middle;
                chrom.start[idx + 1] = middle;
            }
        }
    }
}

/// `HiCMatrix.getBinSize`: the median distance between consecutive bin starts
/// within chromosomes that have more than one bin, truncated to an integer.
///
/// The value is computed once, before masking, and cached by the original.
pub fn bin_size(bins: &[ChromBins]) -> i64 {
    let mut diffs: Vec<i64> = Vec::new();
    for chrom in bins {
        if chrom.len() < 2 {
            continue;
        }
        for pair in chrom.start.windows(2) {
            diffs.push(pair[1] - pair[0]);
        }
    }
    if diffs.is_empty() {
        return 0;
    }
    diffs.sort_unstable();
    let n = diffs.len();
    let median = if n % 2 == 1 {
        diffs[n / 2] as f64
    } else {
        (diffs[n / 2 - 1] as f64 + diffs[n / 2] as f64) / 2.0
    };
    median as i64
}

/// A symmetric matrix held as a band of diagonals.
///
/// `values[d][i]` is the cell at `(i, i + d)`; every diagonal is as long as
/// the chromosome allows, so cells outside the band read as `0`. This matches
/// the shape of the matrix `convert_to_zscore_matrix` leaves behind: it pads
/// the band with the zeros a dense matrix would have had.
pub struct Band {
    depth: usize,
    values: Vec<Vec<f64>>,
}

impl Band {
    /// An all-zero band of `depth` diagonals over `n` bins.
    pub fn new(n: usize, depth: usize) -> Self {
        let values = (0..depth).map(|d| vec![0.0; n.saturating_sub(d)]).collect();
        Band { depth, values }
    }

    /// Number of bins.
    pub fn size(&self) -> usize {
        self.values.first().map_or(0, |v| v.len())
    }

    /// Number of diagonals held (cells at bin distance `depth` and beyond are
    /// outside the band and read as 0).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Set the cell at bin distance `d` starting from bin `i`.
    pub fn set(&mut self, i: usize, d: usize, value: f64) {
        if d < self.depth && i + d < self.size() {
            self.values[d][i] = value;
        }
    }

    /// The cell at `(i, i + d)`, or 0 outside the band.
    pub fn get(&self, i: usize, d: usize) -> f64 {
        if d < self.depth && i + d < self.size() {
            self.values[d][i]
        } else {
            0.0
        }
    }

    /// Drop every diagonal at bin distance `limit` and beyond.
    pub fn truncate(&mut self, limit: usize) {
        self.depth = self.depth.min(limit);
        self.values.truncate(self.depth);
    }

    /// `HiCMatrix.convert_to_obs_exp_matrix(zscore=True, perchr=True)`.
    ///
    /// Cells are pooled by the original's distance key and each pool is
    /// standardized by its own mean and standard deviation. The original
    /// densifies the band first, with a "+1 then -1" sparse trick, so that the
    /// cells missing from the sparse input join the statistics as zeros;
    /// treating a missing cell as `0` gives exactly the same numbers, and it
    /// also produces the diagonal of `NaN` (mean 0, standard deviation 0) the
    /// original stores for every bin.
    ///
    /// The diagonal itself is zeroed first, as `diagflat(value=0)` does.
    ///
    /// `bins` must hold the **pre-`enlarge_bins`** coordinates: the distance
    /// key is derived from them, and the original computes the z-scores before
    /// it closes the gaps that masking left behind. `binsize` likewise comes
    /// from `bin_size`, which is resolved before masking.
    pub fn zscore(&mut self, bins: &ChromBins, binsize: i64) {
        let n = self.size();
        if self.depth == 0 || n == 0 || binsize <= 0 {
            return;
        }

        // TODO(reproduce-hicexplorer): this replicates a quirk of the original
        // and should be revisited upstream before it is treated as correct.
        //
        // `convert_to_obs_exp_matrix` does NOT pool cells by bin offset. Its
        // distance key comes from `getDistList`, which is the difference of bin
        // *start coordinates* -- a genomic distance in bp -- bucketed as
        //
        //     key = int((start[j] - start[i]) / binsize) + 1
        //
        // On a uniform grid that is the bin offset plus one, so the two agree.
        // They diverge wherever masking removed a bin: the gap is still in the
        // coordinates at this point, because the original runs
        // `convert_to_zscore_matrix` *before* `enlarge_bins`. Every pair that
        // spans a gap therefore gets an inflated key and is pooled with cells
        // that are genuinely further away. That is why the two implementations
        // disagree around masked bins -- and, since one gap shifts every pair
        // spanning it, over most of that chromosome besides.
        //
        // Consequence worth flagging upstream: because a cell's pool depends on
        // which bins happened to be dropped, the result is not a z-score in the
        // usual sense -- per pool it is not mean 0 / sd 1. Measured on the 40 kb
        // CNP0007920 leaf matrix the pools came out at mean ~ +0.10..+0.16 and
        // sd ~ 0.80..0.87, where the exact-offset grouping gives exactly 0
        // and 1.
        let key = |i: usize, d: usize| -> usize {
            let span = bins.start[i + d] - bins.start[i];
            (span / binsize) as usize + 1
        };

        // The key is not a function of `d` alone, so pool sizes are not known
        // up front; find the largest key first and index the pools by it.
        let mut max_key = 1usize;
        for d in 1..self.depth {
            let len = n - d;
            if len == 0 {
                break;
            }
            for i in 0..len {
                max_key = max_key.max(key(i, d));
            }
        }
        let n_keys = max_key + 1;

        let mut count = vec![0usize; n_keys];
        let mut sum = vec![0.0f64; n_keys];
        for d in 1..self.depth {
            let len = n - d;
            if len == 0 {
                break;
            }
            let column = &self.values[d];
            for (i, &v) in column.iter().enumerate() {
                let k = key(i, d);
                count[k] += 1;
                sum[k] += v;
            }
        }

        // `diagonal_length` is the original's pooling denominator: the number
        // of cells a pool would hold on a regular grid, `n - (key - 1)`, but
        // never fewer than the cells actually in it,
        //
        //     diagonal_length = max(n - (key - 1), count)
        //
        // It divides the sum for the mean, and it is the count the standard
        // deviation averages over -- a cell a pool is missing from
        // `diagonal_length` contributes as the zero it is.
        let mut mu = vec![f64::NAN; n_keys];
        let mut diagonal_length = vec![0usize; n_keys];
        for k in 1..n_keys {
            let on_grid = n.saturating_sub(k - 1);
            let dl = on_grid.max(count[k]);
            diagonal_length[k] = dl;
            if dl > 0 {
                mu[k] = sum[k] / dl as f64;
            }
        }

        // The deviations are summed with `np.sum`, which accumulates pairwise,
        // while the pool sums above come from `np.bincount`, which does not.
        // Keeping the two orders apart is what makes the last digits match --
        // see `pairwise_sum`.
        let mut deviations: Vec<Vec<f64>> = vec![Vec::new(); n_keys];
        for d in 1..self.depth {
            let len = n - d;
            if len == 0 {
                break;
            }
            let column = &self.values[d];
            for (i, &v) in column.iter().enumerate() {
                let k = key(i, d);
                let dev = v - mu[k];
                deviations[k].push(dev * dev);
            }
        }
        let sq: Vec<f64> = deviations.iter().map(|v| pairwise_sum(v)).collect();

        let mut std = vec![0.0f64; n_keys];
        for k in 1..n_keys {
            let dl = diagonal_length[k];
            if dl == 0 {
                continue;
            }
            let missing = (dl - count[k]) as f64;
            std[k] = ((sq[k] + missing * mu[k] * mu[k]) / dl as f64).sqrt();
        }

        // TODO(reproduce-hicexplorer): the original also forces the key `0`
        // pool to `NaN` (`if maxdepth and bin_dist_plus_one == 0`). Key 0 only
        // exists for inter-chromosomal cells, which `perchr=True` never puts in
        // a band, so it is unreachable here and no guard is kept for it.
        for d in 0..self.depth {
            let len = n - d;
            if len == 0 {
                break;
            }
            if d == 0 {
                // `diagflat(value=0)` zeroed the diagonal, so its pool holds
                // only zeros: mean 0, standard deviation 0, every cell `NaN`.
                self.values[d].iter_mut().for_each(|v| *v = f64::NAN);
                continue;
            }
            let column = &mut self.values[d];
            for (i, v) in column.iter_mut().enumerate() {
                let k = key(i, d);
                *v = if std[k] == 0.0 {
                    f64::NAN
                } else {
                    (*v - mu[k]) / std[k]
                };
            }
        }
    }
}

/// `get_idx_of_bins_at_given_distance`: the bins `window` bp to the left and
/// to the right of bin `cut`.
///
/// Both lookups can come up empty, and the original then slices with `None`,
/// which numpy turns into "to the edge of the matrix". That fallback is kept
/// so the diamonds match.
fn window_edges(bins: &ChromBins, cut: usize, window: i64) -> (usize, usize) {
    let n = bins.len();

    let left_start = 0.max(bins.start[cut] - window);
    let k = bins.start.partition_point(|&s| s <= left_start);
    let left = if k > 0 && bins.end[k - 1] > left_start {
        k - 1
    } else {
        0
    };

    let chrom_end = bins.end[n - 1];
    let right_end = chrom_end.min(bins.end[cut] + window) - 1;
    let right = if right_end < 0 {
        n
    } else {
        let k = bins.start.partition_point(|&s| s <= right_end);
        if k > 0 && bins.end[k - 1] > right_end {
            k - 1
        } else {
            n
        }
    };

    (left, right)
}

/// The bin-distance range of the `[left, cut) x [cut, right)` diamond, as
/// inclusive `d` bounds per row, clipped to the part of the band that is
/// still stored after truncation.
fn diamond_rows(
    band: &Band,
    left: usize,
    cut: usize,
    right: usize,
) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    let limit = band.depth();
    (left..cut).filter_map(move |i| {
        let lo = cut.max(i + 1);
        let hi = right.min(i + limit);
        (lo < hi).then_some((i, lo, hi))
    })
}

/// `get_cut_weight(..., return_mean=True)`: the mean of the dense
/// `[left, cut) x [cut, right)` block, with cells outside the stored band
/// counted as the zeros they are.
///
/// The mean is over the whole block, not just its stored cells, so it is
/// `None` only when the block is empty. The original returns 0 in that case
/// and when the block holds no stored cell at all; both come out as 0 here.
pub fn diamond_mean(band: &Band, left: usize, cut: usize, right: usize) -> f64 {
    let rows = cut.saturating_sub(left);
    let cols = right.saturating_sub(cut);
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let values = diamond_values(band, left, cut, right);
    pairwise_sum(&values) / values.len() as f64
}

/// `get_cut_weight(...)`: the dense `[left, cut) x [cut, right)` block,
/// row-major, with cells outside the stored band as zeros.
pub fn diamond_values(band: &Band, left: usize, cut: usize, right: usize) -> Vec<f64> {
    let rows = cut.saturating_sub(left);
    let cols = right.saturating_sub(cut);
    let mut out = vec![0.0; rows * cols];
    for (i, lo, hi) in diamond_rows(band, left, cut, right) {
        for j in lo..hi {
            out[(i - left) * cols + (j - cut)] = band.get(i, j - i);
        }
    }
    out
}

/// The TAD-separation score of one chromosome: for every bin, the mean
/// z-score of the diamond reaching `window` bp to each side, for every window
/// in `windows`.
///
/// A bin is skipped when any of its diamonds is `NaN`, which is how the
/// original drops bins whose band has a zero-variance diagonal.
pub fn tad_separation_scores(
    band: &Band,
    bins: &ChromBins,
    windows: &[i64],
) -> Vec<(usize, Vec<f64>)> {
    let mut rows = Vec::with_capacity(bins.len());
    for cut in 0..bins.len() {
        let mut values = Vec::with_capacity(windows.len());
        let mut ok = true;
        for &window in windows {
            let (left, right) = window_edges(bins, cut, window);
            let mean = diamond_mean(band, left, cut, right);
            if mean.is_nan() {
                ok = false;
                break;
            }
            values.push(mean);
        }
        if ok {
            rows.push((cut, values));
        }
    }
    rows
}

/// The diamond around `cut` at `window` bp, for the Wilcoxon comparison in
/// [`crate::findtads::call`].
pub fn diamond_at(band: &Band, bins: &ChromBins, cut: usize, window: i64) -> Vec<f64> {
    let (left, right) = window_edges(bins, cut, window);
    diamond_values(band, left, cut, right)
}

/// The bin `window` bp to the left of `cut` and to the right of it — the two
/// neighbours [`crate::findtads::call::min_pvalues`] compares against.
pub fn neighbour_bins(bins: &ChromBins, cut: usize, window: i64) -> (usize, usize) {
    window_edges(bins, cut, window)
}
