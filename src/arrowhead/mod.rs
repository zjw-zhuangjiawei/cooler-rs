//! Arrowhead contact-domain caller, ported from juicer
//! (`BlockBuster` + `BlockResults` + `MatrixTriangles`, Huntley & Durand 2016).
//!
//! Per chromosome: slide a diagonal window, apply the directionality-index
//! "arrowhead" transform, build a block-score matrix via triangle prefix sums,
//! threshold it, and reduce each connected component to its highest-scoring
//! cell; a two-pass threshold sweep then merges low/high-confidence blocks and
//! bins nearby domains by distance. Input is normalized on the fly through
//! [`crate::File::fetch`], so both `.cool` and `.hic` work with the same code.

mod components;
mod triangles;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use ndarray::Array2;

use crate::error::{Error, Result};
use crate::file::File;
use crate::region::Region;

/// Parameters controlling the sweep (defaults match juicer).
#[derive(Debug, Clone)]
pub struct Params {
    /// Sliding-window width along the diagonal, in bins.
    pub matrix_width: usize,
    /// High-confidence variance threshold (`None` in the low-confidence pass).
    pub var_threshold: Option<f64>,
    /// High-confidence sign threshold.
    pub high_sign_threshold: f64,
    /// Low-confidence sign threshold sweep start.
    pub max_low_sign_threshold: f64,
    /// Low-confidence sign threshold sweep end.
    pub min_low_sign_threshold: f64,
    /// Low-confidence sign threshold sweep step.
    pub decrement_low_sign_threshold: f64,
    /// Minimum domain width, in bins.
    pub min_block_size: usize,
    /// Upstream/downstream gap for the directionality index.
    pub gap: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            matrix_width: 2000,
            var_threshold: Some(0.2),
            high_sign_threshold: 0.5,
            max_low_sign_threshold: 0.4,
            min_low_sign_threshold: 0.0,
            decrement_low_sign_threshold: 0.1,
            min_block_size: 60,
            gap: 7,
        }
    }
}

/// A contact domain (a diagonal block) with its arrowhead score statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub score: f64,
    pub up_var: f64,
    pub lo_var: f64,
    pub up_sign: f64,
    pub lo_sign: f64,
}

/// A highest-scoring block cell. `i <= j` are bin indices (scaled to base
/// pairs after the sweep). Exact field equality + hash mirror juicer's
/// `HighScore.equals`/`hashCode` (used for the low/high set difference).
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighScore {
    pub i: i64,
    pub j: i64,
    pub score: f64,
    pub u_var: f64,
    pub l_var: f64,
    pub up_sign: f64,
    pub lo_sign: f64,
}

impl HighScore {
    fn offset_index(&mut self, offset: i64) {
        self.i += offset;
        self.j += offset;
    }

    fn scale_by_resolution(&mut self, resolution: i64) {
        self.i *= resolution;
        self.j *= resolution;
    }

    fn width(&self) -> i64 {
        (self.j - self.i).abs()
    }

    fn sort_value(&self) -> f64 {
        self.u_var + self.l_var
    }
}

impl PartialEq for HighScore {
    fn eq(&self, o: &Self) -> bool {
        self.i == o.i
            && self.j == o.j
            && self.score == o.score
            && self.u_var == o.u_var
            && self.l_var == o.l_var
            && self.up_sign == o.up_sign
            && self.lo_sign == o.lo_sign
    }
}

impl Eq for HighScore {}

impl Hash for HighScore {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.i.hash(state);
        self.j.hash(state);
        self.score.to_bits().hash(state);
        self.u_var.to_bits().hash(state);
        self.l_var.to_bits().hash(state);
        self.up_sign.to_bits().hash(state);
        self.lo_sign.to_bits().hash(state);
    }
}

/// Directionality-index upstream transform: for each row `i`, compare the
/// upstream `A` and downstream `B` contacts at symmetric distances, storing
/// `(A - B) / (A + B)` in the upper triangle.
fn directionality_index_upstream(observed: &Array2<f64>, gap: usize) -> Array2<f64> {
    let n = observed.nrows().min(observed.ncols());
    let mut d_up = Array2::zeros((n, n));
    for i in 0..n {
        let window = (n.saturating_sub(i + gap))
            .min(i.saturating_sub(gap))
            .min(n);
        if window >= gap {
            for j in i + gap..=i + window {
                let k = j - (i + gap);
                let a = observed[[i, i - gap - k]];
                let b = observed[[i, i + gap + k]];
                d_up[[i, j]] = (a - b) / (a + b);
            }
        }
    }
    d_up
}

/// Fetch a dense symmetric `n x n` window (`lo..hi` global bins) at `res`,
/// normalized by `norm` when given.
fn fetch_window_matrix(
    f: &File,
    chrom: &str,
    res: u64,
    lo: i64,
    hi: i64,
    chrom_len: u64,
    norm: Option<&str>,
) -> Result<Array2<f64>> {
    let region = Region::range(chrom, lo as u64 * res, (hi as u64 * res).min(chrom_len));
    let pixels = f.fetch(&region, norm)?;
    let n = (hi - lo) as usize;
    let mut m = Array2::zeros((n, n));
    for p in &pixels {
        if p.bin1_id >= lo && p.bin1_id < hi && p.bin2_id >= lo && p.bin2_id < hi {
            let i = (p.bin1_id - lo) as usize;
            let j = (p.bin2_id - lo) as usize;
            m[[i, j]] = p.count;
            m[[j, i]] = p.count;
        }
    }
    Ok(m)
}

/// Run arrowhead on one window and return its highest-scoring blocks
/// (local, un-offset indices).
fn block_results(
    observed: &Array2<f64>,
    var_threshold: Option<f64>,
    sign_threshold: f64,
    gap: usize,
) -> Vec<HighScore> {
    let d_up = directionality_index_upstream(observed, gap);
    let mut tri = triangles::MatrixTriangles::new(&d_up);
    tri.threshold_score_values(var_threshold, sign_threshold);
    let components = tri.extract_connected_components();
    tri.calculate_results(&components)
}

/// Slide across the chromosome diagonal, accumulating one pass's blocks.
#[allow(clippy::too_many_arguments)]
fn call_sub_blockbuster(
    f: &File,
    chrom: &str,
    res: u64,
    n_bins: usize,
    chrom_len: u64,
    norm: Option<&str>,
    var_threshold: Option<f64>,
    sign_threshold: f64,
    params: &Params,
) -> Result<Vec<HighScore>> {
    let increment = params.matrix_width / 2;
    let gap = params.gap;
    let mut results = Vec::new();

    let mut lim_start = 0usize;
    while lim_start < n_bins {
        let lim_end_incl = (lim_start + params.matrix_width).min(n_bins);
        // juicer backs the final window off so the tail is scanned at full
        // width, overlapping the previous window.
        let lo = if lim_end_incl == n_bins && n_bins > increment {
            n_bins.saturating_sub(params.matrix_width)
        } else {
            lim_start
        };
        let hi = (lim_end_incl + 1).min(n_bins);
        if hi <= lo {
            break;
        }
        let observed = fetch_window_matrix(f, chrom, res, lo as i64, hi as i64, chrom_len, norm)?;
        let mut window = block_results(&observed, var_threshold, sign_threshold, gap);
        for s in &mut window {
            // Offset by the true window start (juicer offsets by `limStart`,
            // which misplaces the final window's domains; see plan).
            s.offset_index(lo as i64);
        }
        results.extend(window);
        lim_start += increment;
    }
    Ok(results)
}

/// The low/high-confidence two-pass sweep + merge for one chromosome.
pub fn call_chrom(
    f: &File,
    chrom: &str,
    norm: Option<&str>,
    params: &Params,
) -> Result<Vec<Domain>> {
    let chroms = f.chroms()?;
    let c = chroms
        .iter()
        .find(|c| c.name == chrom)
        .ok_or_else(|| Error::InvalidInput(format!("chromosome '{chrom}' not found")))?;
    let res = f.resolution() as u64;
    let n_bins = (c.length as u64).div_ceil(res) as usize;

    // Low-confidence pass: relax the sign threshold until blocks appear.
    let mut sign_threshold = params.max_low_sign_threshold;
    let low = loop {
        let l = call_sub_blockbuster(
            f,
            chrom,
            res,
            n_bins,
            c.length as u64,
            norm,
            None,
            sign_threshold,
            params,
        )?;
        if !l.is_empty() {
            break l;
        }
        sign_threshold -= params.decrement_low_sign_threshold;
        if sign_threshold < params.min_low_sign_threshold - 1e-12 {
            break l;
        }
    };

    // High-confidence pass.
    let mut high = call_sub_blockbuster(
        f,
        chrom,
        res,
        n_bins,
        c.length as u64,
        norm,
        params.var_threshold,
        params.high_sign_threshold,
        params,
    )?;

    let unique = ordered_set_difference(&low, &high);
    let filtered = filter_blocks_by_size(unique, params.min_block_size);
    append_non_conflicting_blocks(&mut high, filtered);

    for s in &mut high {
        s.scale_by_resolution(res as i64);
    }

    let binned = bin_scores_by_distance(high, 5 * res as i64);
    let binned = bin_scores_by_distance(binned, 10 * res as i64);
    let mut sorted = binned;
    sorted.sort_by(|a, b| b.sort_value().total_cmp(&a.sort_value()));

    Ok(sorted
        .into_iter()
        .map(|s| Domain {
            chrom: chrom.to_string(),
            start: s.i as u64,
            end: s.j as u64,
            score: s.score,
            up_var: s.u_var,
            lo_var: s.l_var,
            up_sign: s.up_sign,
            lo_sign: s.lo_sign,
        })
        .collect())
}

/// Call domains on several chromosomes (all, when `chroms` is `None`).
pub fn call_domains(
    f: &File,
    norm: Option<&str>,
    params: &Params,
    chroms: Option<&[String]>,
) -> Result<Vec<Domain>> {
    let names: Vec<String> = match chroms {
        Some(cs) => cs.to_vec(),
        None => f.chroms()?.into_iter().map(|c| c.name).collect(),
    };
    let mut out = Vec::new();
    for name in names {
        out.extend(call_chrom(f, &name, norm, params)?);
    }
    Ok(out)
}

fn ordered_set_difference(a: &[HighScore], b: &[HighScore]) -> Vec<HighScore> {
    let set_b: HashSet<HighScore> = b.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut diff = Vec::new();
    for &s in a {
        if !set_b.contains(&s) && seen.insert(s) {
            diff.push(s);
        }
    }
    diff
}

fn filter_blocks_by_size(blocks: Vec<HighScore>, min_width: usize) -> Vec<HighScore> {
    blocks
        .into_iter()
        .filter(|s| s.width() > min_width as i64)
        .collect()
}

fn append_non_conflicting_blocks(main: &mut Vec<HighScore>, additions: Vec<HighScore>) {
    let mut edges: HashSet<i64> = HashSet::new();
    for s in main.iter() {
        edges.insert(s.i);
        edges.insert(s.j);
    }
    for s in additions {
        let conflict = (s.i..=s.j).any(|k| edges.contains(&k));
        if !conflict {
            edges.insert(s.i);
            edges.insert(s.j);
            main.push(s);
        }
    }
}

/// A running bin of nearby blocks (juicer `BinnedScore`).
struct BinnedScore {
    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
    scores: Vec<f64>,
    u_vars: Vec<f64>,
    l_vars: Vec<f64>,
    up_signs: Vec<f64>,
    lo_signs: Vec<f64>,
}

impl BinnedScore {
    fn new(s: HighScore) -> Self {
        let mut b = BinnedScore {
            min_x: s.i,
            max_x: s.i,
            min_y: s.j,
            max_y: s.j,
            scores: Vec::new(),
            u_vars: Vec::new(),
            l_vars: Vec::new(),
            up_signs: Vec::new(),
            lo_signs: Vec::new(),
        };
        b.push(s);
        b
    }

    fn push(&mut self, s: HighScore) {
        self.scores.push(s.score);
        self.u_vars.push(s.u_var);
        self.l_vars.push(s.l_var);
        self.up_signs.push(s.up_sign);
        self.lo_signs.push(s.lo_sign);
    }

    fn is_near(&self, s: &HighScore, dist: i64) -> bool {
        ((self.min_x - s.i).abs() < dist || (self.max_x - s.i).abs() < dist)
            && ((self.min_y - s.j).abs() < dist || (self.max_y - s.j).abs() < dist)
    }

    fn add_score(&mut self, s: HighScore) {
        if s.i < self.min_x {
            self.min_x = s.i;
        } else if s.i > self.max_x {
            self.max_x = s.i;
        }
        if s.j < self.min_y {
            self.min_y = s.j;
        } else if s.j > self.max_y {
            self.max_y = s.j;
        }
        self.push(s);
    }

    fn convert(self) -> HighScore {
        HighScore {
            i: self.max_x,
            j: self.max_y,
            score: mean(&self.scores),
            u_var: mean(&self.u_vars),
            l_var: mean(&self.l_vars),
            up_sign: mean(&self.up_signs),
            lo_sign: mean(&self.lo_signs),
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn bin_scores_by_distance(scores: Vec<HighScore>, dist: i64) -> Vec<HighScore> {
    let mut bins: Vec<BinnedScore> = Vec::new();
    for s in scores {
        let mut binned = false;
        for b in &mut bins {
            if b.is_near(&s, dist) {
                b.add_score(s);
                binned = true;
                break;
            }
        }
        if !binned {
            bins.push(BinnedScore::new(s));
        }
    }
    bins.into_iter().map(BinnedScore::convert).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn dense(n: usize, fill: &[(usize, usize, f64)]) -> Array2<f64> {
        let mut m = Array2::zeros((n, n));
        for &(i, j, v) in fill {
            m[[i, j]] = v;
            m[[j, i]] = v;
        }
        m
    }

    #[test]
    fn di_upstream_matches_hand_computation() {
        // n=8, gap=2, i=4: only j=6 fires, a=obs[4,2], b=obs[4,6].
        let m = dense(8, &[(4, 2, 10.0), (4, 6, 30.0)]);
        let d = directionality_index_upstream(&m, 2);
        assert!((d[[4, 6]] - (-0.5)).abs() < 1e-12);
        assert_eq!(d[[6, 4]], 0.0); // lower triangle untouched
    }

    #[test]
    fn connected_components_8_connectivity() {
        let m = array![[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let comps = components::detection(&m, 0.0);
        assert_eq!(comps.len(), 2);
        let mut sizes: Vec<usize> = comps.iter().map(|c| c.len()).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, [1, 3]);
    }
}
