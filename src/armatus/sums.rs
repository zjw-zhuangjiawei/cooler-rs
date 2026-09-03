//! Upper-triangular prefix sums and per-size means (port of
//! `ArmatusParams::computeSumMuSigma_`).

use ndarray::Array2;

/// Precomputed quantities for the Armatus dynamic program.
///
/// `sums(j, i)` is the sum of the upper-triangular sub-matrix `[j..=i] x [j..=i]`
/// (only `j <= i` is populated); `mu[d]` is the mean of `sums(j, i) / d^gamma`
/// over all domains of size `d`, or `f64::MAX` when fewer than
/// `min_mean_samples` such domains exist (which forces `q = -inf`, i.e. no
/// domain of that size, exactly as in the C++ `mu[i] = max()`).
pub struct Sums {
    /// Number of bins.
    pub n: usize,
    /// Resolution exponent `gamma`.
    pub gamma: f64,
    /// Row-major `n*n` upper-triangular prefix sums.
    pub sums: Vec<f64>,
    /// `mu[0..=n]`.
    pub mu: Vec<f64>,
}

impl Sums {
    pub fn compute(matrix: &Array2<f64>, gamma: f64, min_mean_samples: usize) -> Self {
        let n = matrix.nrows();
        let mut sums = vec![0.0; n * n];

        for i in 0..n {
            sums[i * n + i] = matrix[[i, i]];
        }

        let mut count = vec![0u64; n + 1];
        let mut mean = vec![0.0f64; n + 1];
        let mut column_sums = vec![0.0f64; n];

        for i in 1..n {
            column_sums[i] = matrix[[i, i]];
            for j in (0..i).rev() {
                column_sums[j] = column_sums[j + 1] + matrix[[j, i]];
                sums[j * n + i] = sums[j * n + (i - 1)] + column_sums[j];
                let d = i - j + 1;
                let s = sums[j * n + i] / (d as f64).powf(gamma);
                count[d] += 1;
                mean[d] += s;
            }
        }

        let mut mu = vec![0.0f64; n + 1];
        for d in 0..=n {
            mu[d] = if count[d] >= min_mean_samples as u64 {
                mean[d] / count[d] as f64
            } else {
                f64::MAX
            };
        }

        Sums { n, gamma, sums, mu }
    }

    /// `sums(j, i)` for `j <= i`.
    #[inline]
    pub fn at(&self, j: usize, i: usize) -> f64 {
        self.sums[j * self.n + i]
    }
}
