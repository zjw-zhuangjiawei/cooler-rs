//! The block-score matrix and its dynamic-programming prefixes, ported from
//! `MatrixTriangles` + `DynamicProgrammingUtils` (juicer arrowhead).
//!
//! Given the directionality-index-upstream matrix, `MatrixTriangles` builds
//! the upper/lower-triangle means, signs and variances (via column/row prefix
//! sums `right`/`upper`), combines them into a block score, thresholds it, and
//! reduces each connected component to its highest-scoring cell.

use ndarray::Array2;

use super::HighScore;

/// Column prefix sum toward the diagonal: `right[i][j] = sum_{k=i..=j} m[k][j]`.
fn right(m: &Array2<f64>, max_size: usize) -> Array2<f64> {
    let n = m.nrows().min(m.ncols());
    let mut out = Array2::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = m[[i, i]];
    }
    for j in 1..n {
        let end_point = (j as isize - 1 - max_size as isize).max(0) as usize;
        for i in (end_point..j).rev() {
            out[[i, j]] = m[[i, j]] + out[[i + 1, j]];
        }
    }
    out
}

/// Row prefix sum away from the diagonal: `upper[i][j] = sum_{k=i+1..=j} m[i][k]`.
fn upper(m: &Array2<f64>, max_size: usize) -> Array2<f64> {
    let n = m.nrows().min(m.ncols());
    let mut out = Array2::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = m[[i, i]];
    }
    for i in 0..n {
        let end_point = (i + 1 + max_size).min(n - 1);
        for j in i + 1..=end_point {
            out[[i, j]] = m[[i, j]] + out[[i, j - 1]];
        }
    }
    out
}

/// Divide a matrix by its largest element (`normalizeByMax` in juicer).
fn normalize_by_max(m: &Array2<f64>) -> Array2<f64> {
    let max = m.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    m.mapv(|v| v / max)
}

/// Replace every `0.0` with `1.0` (so element-wise division below is safe).
fn replace_zeros_with_ones(m: &Array2<f64>) -> Array2<f64> {
    m.mapv(|v| if v == 0.0 { 1.0 } else { v })
}

pub(crate) struct MatrixTriangles {
    up_sign: Array2<f64>,
    lo_sign: Array2<f64>,
    up_var: Array2<f64>,
    lo_var: Array2<f64>,
    block_score: Array2<f64>,
}

impl MatrixTriangles {
    /// Build the up/lo mean/sign/variance matrices and the block score.
    pub(crate) fn new(d_upstream: &Array2<f64>) -> Self {
        let n = d_upstream.nrows().min(d_upstream.ncols());
        let m = d_upstream.mapv(|v| if v.is_nan() { 0.0 } else { v });
        let squared = m.mapv(|v| v * v);
        let sign = m.mapv(|v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
        let ones = Array2::from_elem((n, n), 1.0);

        let r_sum = right(&m, n);
        let r_sign = right(&sign, n);
        let r_squared = right(&squared, n);
        let r_count = right(&ones, n);

        let u_sum = upper(&m, n);
        let u_sign = upper(&sign, n);
        let u_squared = upper(&squared, n);
        let u_count = upper(&ones, n);

        let mut up: Array2<f64> = Array2::zeros((n, n));
        let mut up_sign: Array2<f64> = Array2::zeros((n, n));
        let mut up_squared: Array2<f64> = Array2::zeros((n, n));
        let mut up_count: Array2<f64> = Array2::zeros((n, n));
        for i in 0..n {
            for j in i + 1..n {
                let bottom = (j - i).div_ceil(2);
                up[[i, j]] = up[[i, j - 1]] + r_sum[[i, j]] - r_sum[[i + bottom, j]];
                up_sign[[i, j]] = up_sign[[i, j - 1]] + r_sign[[i, j]] - r_sign[[i + bottom, j]];
                up_squared[[i, j]] =
                    up_squared[[i, j - 1]] + r_squared[[i, j]] - r_squared[[i + bottom, j]];
                up_count[[i, j]] =
                    up_count[[i, j - 1]] + r_count[[i, j]] - r_count[[i + bottom, j]];
            }
        }
        up = up / replace_zeros_with_ones(&up_count);
        up_sign = up_sign / replace_zeros_with_ones(&up_count);
        up_squared = up_squared / replace_zeros_with_ones(&up_count);

        let mut lo: Array2<f64> = Array2::zeros((n, n));
        let mut lo_sign: Array2<f64> = Array2::zeros((n, n));
        let mut lo_squared: Array2<f64> = Array2::zeros((n, n));
        let mut lo_count: Array2<f64> = Array2::zeros((n, n));
        for a in 0..n {
            for b in a + 1..n {
                let val = (b - a).div_ceil(2);
                let endpt = (2 * b - a).min(n - 1);
                lo_count[[a, b]] =
                    lo_count[[a, b - 1]] + u_count[[b, endpt]] - r_count[[a + val, b]];
                lo[[a, b]] = lo[[a, b - 1]] + u_sum[[b, endpt]] - r_sum[[a + val, b]];
                lo_sign[[a, b]] = lo_sign[[a, b - 1]] + u_sign[[b, endpt]] - r_sign[[a + val, b]];
                lo_squared[[a, b]] =
                    lo_squared[[a, b - 1]] + u_squared[[b, endpt]] - r_squared[[a + val, b]];
            }
        }
        lo = lo / replace_zeros_with_ones(&lo_count);
        lo_sign = lo_sign / replace_zeros_with_ones(&lo_count);
        lo_squared = lo_squared / replace_zeros_with_ones(&lo_count);

        let up_var = &up_squared - &(&up * &up);
        let lo_var = &lo_squared - &(&lo * &lo);

        let diff = normalize_by_max(&(&lo - &up));
        let diff_sign = normalize_by_max(&(&lo_sign - &up_sign));
        let diff_squared = normalize_by_max(&(&up_var + &lo_var));
        let block_score = &(&diff + &diff_sign) - &diff_squared;

        MatrixTriangles {
            up_sign,
            lo_sign,
            up_var,
            lo_var,
            block_score,
        }
    }

    /// Zero out cells that fail the sign/var thresholds.
    pub(crate) fn threshold_score_values(
        &mut self,
        var_threshold: Option<f64>,
        sign_threshold: f64,
    ) {
        for i in 0..self.block_score.nrows() {
            for j in 0..self.block_score.ncols() {
                if -self.up_sign[[i, j]] < sign_threshold || self.lo_sign[[i, j]] < sign_threshold {
                    self.block_score[[i, j]] = 0.0;
                }
            }
        }
        if let Some(t) = var_threshold {
            let sums = &self.up_var + &self.lo_var;
            for i in 0..self.block_score.nrows() {
                for j in 0..self.block_score.ncols() {
                    if sums[[i, j]] > t {
                        self.block_score[[i, j]] = 0.0;
                    }
                }
            }
        }
    }

    /// The connected components of the thresholded block score (`> 0`).
    pub(crate) fn extract_connected_components(&self) -> Vec<Vec<(usize, usize)>> {
        super::components::detection(&self.block_score, 0.0)
    }

    /// Reduce each component to its highest-scoring cell as a [`HighScore`].
    pub(crate) fn calculate_results(&self, components: &[Vec<(usize, usize)>]) -> Vec<HighScore> {
        components
            .iter()
            .map(|comp| {
                let (i, j) = comp
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        self.block_score[[a.0, a.1]].total_cmp(&self.block_score[[b.0, b.1]])
                    })
                    .unwrap();
                HighScore {
                    i: i as i64,
                    j: j as i64,
                    score: self.block_score[[i, j]],
                    u_var: self.up_var[[i, j]],
                    l_var: self.lo_var[[i, j]],
                    up_sign: -self.up_sign[[i, j]],
                    lo_sign: self.lo_sign[[i, j]],
                }
            })
            .collect()
    }
}
