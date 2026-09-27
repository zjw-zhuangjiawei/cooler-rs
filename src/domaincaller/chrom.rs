//! Chromosome-level TAD calling: the `domaincaller` pipeline from
//! TADLib (`tadlib/domaincaller/chromLev.py` + the HMM setup in
//! `tadlib/hitad/genomeLev.py`). The pipeline is:
//! 1. estimate an adaptive window size per bin (`minWindows`),
//! 2. compute a directionality index per bin (`calDI`),
//! 3. split the chromosome into gap-free regions (`splitChrom`),
//! 4. train a 4-state Gaussian-mixture HMM on the DI values (`oriHMMParams`),
//! 5. decode each region with Viterbi and read boundaries off the state
//!    sequence (`_getBounds`/`pipe`),
//! 6. iterate: refine window sizes from the latest bottom domains and repeat
//!    (`oriIter`, up to 5 rounds, exiting early when the mismatch ratio from
//!    the domain aligner drops below 0.05).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::f64::consts::PI;
use std::rc::Rc;

use ndarray::{s, Array2};
use sprs::{CsMat, TriMat};

use super::aligner::{Domain, DomainAligner, DomainSet};
use crate::stats::{GeneralMixtureModel, HiddenMarkovModel, NormalDistribution, END, START};

/// Chromosome window for DI estimation, in base pairs (TADLib default).
const DEFAULT_WINDOW: u64 = 2_000_000;
/// Minimum domain size in bins (TADLib).
const MINSIZE: usize = 5;
/// Max domain size in bp (TADLib hitad `maxapart`, CLI `--maxsize`).
const MAX_APART: u64 = 4_000_000;

/// The shifted `rawMatrix` of TADLib: a `3n × 3n` sparse matrix with entries
/// at `(row + n, col + n)`. Kept in both CSR and CSC so the per-bin DI slices
/// read either orientation in O(nnz of the row/column); sprs has no dense
/// window-slice primitive (`slice_outer` only slices whole outer dims), so
/// the scipy `toarray().ravel()` semantics live here.
struct RawMatrix {
    csr: CsMat<f64>,
    csc: CsMat<f64>,
}

impl RawMatrix {
    /// Build from (row, col, value) entries; duplicate positions are summed
    /// (scipy's COO → CSR semantics).
    fn from_entries(n: usize, entries: &[(usize, usize, f64)]) -> Self {
        let n = n.max(1);
        let mut tri = TriMat::new((n, n));
        for &(r, c, v) in entries {
            tri.add_triplet(r, c, v);
        }
        let csr = tri.to_csr();
        let csc = csr.to_csc();
        RawMatrix { csr, csc }
    }

    /// Row `i`, columns `[start, end)`, zero-filled to length `end - start`
    /// (scipy `csr[i, start:end].toarray().ravel()` semantics).
    fn row_slice(&self, i: usize, start: usize, end: usize) -> Vec<f64> {
        let mut out = vec![0.0; end.saturating_sub(start)];
        if let Some(row) = self.csr.outer_view(i) {
            for (c, &v) in row.iter() {
                if c >= start && c < end {
                    out[c - start] = v;
                }
            }
        }
        out
    }

    /// Column `i`, rows `[start, end)`, zero-filled to length `end - start`,
    /// in ascending row order. Reads the CSC copy — sprs `transpose_view`
    /// only relabels CSR storage and would return row data for a column
    /// access.
    fn col_slice(&self, i: usize, start: usize, end: usize) -> Vec<f64> {
        let mut out = vec![0.0; end.saturating_sub(start)];
        if let Some(col) = self.csc.outer_view(i) {
            for (r, &v) in col.iter() {
                if r >= start && r < end {
                    out[r - start] = v;
                }
            }
        }
        out
    }

    /// Dense submatrix `[a, b) × [a, b)`.
    /// Point lookup (missing entries are 0), as scipy's dense `raw[x, y]`.
    fn at(&self, i: usize, j: usize) -> f64 {
        self.csr.get(i, j).copied().unwrap_or(0.0)
    }

    fn submatrix(&self, a: usize, b: usize) -> Array2<f64> {
        let n = self.csr.rows();
        let a = a.min(n);
        let b = b.min(n);
        // slice_outer_rbr keeps inner (column) indices global, so fill with c - a.
        let rows = self.csr.view().slice_outer_rbr(a..b);
        let mut mat = Array2::zeros((b - a, b - a));
        for (i, row) in rows.outer_iterator().enumerate() {
            for (c, &v) in row.iter() {
                if c >= a && c < b {
                    mat[[i, c - a]] = v;
                }
            }
        }
        mat
    }
}

/// Per-bin windows (int32 in TADLib) and DI values.
pub struct Chrom {
    pub chrom: String,
    pub res: u64,
    pub chrom_len: usize,
    rm: usize,
    dw: usize,
    raw: RawMatrix,
    windows: Vec<i32>,
    /// Directionality index per bin, after `call_domains`.
    pub dis: Vec<f64>,
    region_dis: BTreeMap<(usize, usize), Vec<f64>>,
    gapbins: HashSet<usize>,
    hmm: Option<Rc<HiddenMarkovModel>>,
    min_domains: BTreeMap<(usize, usize), Vec<[f64; 4]>>,
    /// Final domains: `[start_bp, end_bp, noise, level]`.
    pub domains: Vec<[f64; 4]>,
    /// Max domain size in bp (TADLib hitad `maxapart`, CLI `--maxsize`).
    maxapart: u64,
    /// `ref_noise` result for degenerate domains: domaincaller returns 1
    /// (the CLI noise filter drops them), hitad returns 0.
    refnoise_degenerate: f64,
    /// TADs from the bottom domains (TADLib hitad `maxCore`).
    max_domains: BTreeMap<(usize, usize), Vec<[f64; 4]>>,
    /// Hierarchical domains per gap-free region: `[start_bp, end_bp, noise,
    /// level]`, after `call_hier_domain`.
    pub(crate) hier_domains: BTreeMap<(usize, usize), Vec<[f64; 4]>>,
}

impl Chrom {
    /// `res` is the bin size in base pairs; `n` the number of bins.
    /// `entries` are (row, col, count) pixel coordinates (0-based, any
    /// triangle; the shifted raw matrix mirrors TADLib).
    pub fn new(chrom: &str, res: u64, n: usize, entries: &[(usize, usize, f64)]) -> Self {
        let dw = (DEFAULT_WINDOW / res.max(1)) as usize;
        let shifted: Vec<(usize, usize, f64)> = entries
            .iter()
            .map(|&(r, c, v)| (r + n, c + n, if v.is_nan() { 0.0 } else { v }))
            .collect();
        let raw = RawMatrix::from_entries(3 * n, &shifted);
        Chrom {
            chrom: chrom.to_string(),
            res,
            chrom_len: n,
            rm: 1,
            dw,
            raw,
            windows: Vec::new(),
            dis: Vec::new(),
            region_dis: BTreeMap::new(),
            gapbins: HashSet::new(),
            hmm: None,
            min_domains: BTreeMap::new(),
            domains: Vec::new(),
            maxapart: MAX_APART,
            refnoise_degenerate: 1.0,
            max_domains: BTreeMap::new(),
            hier_domains: BTreeMap::new(),
        }
    }

    /// Detect peaks in a 1-D series (TADLib `detectPeaks`).
    fn detect_peaks(&self, trends: &[f64], mph: f64, mpd: usize) -> Vec<usize> {
        if trends.len() < 2 {
            return Vec::new();
        }
        let dx: Vec<f64> = trends.windows(2).map(|w| w[1] - w[0]).collect();
        // ind = where ((dx,0) < 0) & ((0,dx) > 0)
        let mut ind = Vec::new();
        for i in 0..trends.len() {
            let a = if i < dx.len() { dx[i] } else { 0.0 };
            let b = if i > 0 { dx[i - 1] } else { 0.0 };
            if a < 0.0 && b > 0.0 {
                ind.push(i);
            }
        }
        // sp = where(trends == 1); use the last such position
        if let Some(last_one) = trends.iter().rposition(|&v| v == 1.0) {
            ind.insert(0, last_one);
        }
        if dx.last().map(|&d| d > 0.0).unwrap_or(false) {
            ind.push(trends.len() - 1);
        }
        if !ind.is_empty() {
            ind.retain(|&i| trends[i] > mph);
        }
        if !ind.is_empty() && mpd > 1 {
            // sort by descending trend value, then greedily drop close peaks
            let mut order: Vec<usize> = (0..ind.len()).collect();
            order.sort_by(|&x, &y| trends[ind[y]].partial_cmp(&trends[ind[x]]).unwrap());
            let mut idel = vec![false; ind.len()];
            for &oi in &order {
                if !idel[oi] {
                    for j in 0..ind.len() {
                        let in_range =
                            ind[j] >= ind[oi].saturating_sub(mpd) && ind[j] <= ind[oi] + mpd;
                        idel[j] |= in_range && trends[ind[oi]] > trends[ind[j]];
                    }
                    idel[oi] = false;
                }
            }
            let mut kept: Vec<usize> = (0..ind.len())
                .filter(|&j| !idel[j])
                .map(|j| ind[j])
                .collect();
            kept.sort_unstable();
            ind = kept;
        }
        ind
    }

    /// Chi-square randomness test on a 0/1 sequence (TADLib `randomCheck`).
    /// p-value of chisquare(df=3) via the closed form Q(3/2, x) = erfc(sqrt x)
    /// + 2 sqrt(x/pi) e^{-x}.
    fn random_check(&self, seq: &[u8], pthre: f64) -> bool {
        let mut counts = [0usize; 4];
        for w in seq.windows(2) {
            let idx = match (w[0], w[1]) {
                (b'0', b'0') => 0,
                (b'0', b'1') => 1,
                (b'1', b'0') => 2,
                (b'1', b'1') => 3,
                _ => continue,
            };
            counts[idx] += 1;
        }
        let mean = counts.iter().sum::<usize>() as f64 / 4.0;
        let stat = counts
            .iter()
            .map(|&c| (c as f64 - mean).powi(2) / mean)
            .sum::<f64>();
        // pval = Q(1.5, stat/2)
        let x = stat / 2.0;
        let pval = libm::erfc(x.sqrt()) + 2.0 * (x / PI).sqrt() * (-x).exp();
        pval <= pthre
    }

    /// Estimate the best window size for a single bin (TADLib `oriWindow`).
    fn ori_window(&self, p: &[f64]) -> usize {
        let noise: Vec<bool> = p.iter().map(|&x| x == 0.0).collect();
        let check_len = noise.len().min(20);
        let noiselevel =
            noise[..check_len].iter().filter(|&&n| n).count() as f64 / check_len as f64;
        if noiselevel > 0.6 {
            return 0;
        }
        // Boolean bias indicators (TADLib `m = [P < 0, P > 0]`).
        let m_neg: Vec<bool> = p.iter().map(|&v| v < 0.0).collect();
        let m_pos: Vec<bool> = p.iter().map(|&v| v > 0.0).collect();
        // Cumulative fractions of each indicator.
        let trends = |m: &[bool]| -> Vec<f64> {
            let mut c = 0usize;
            m.iter()
                .enumerate()
                .map(|(k, &b)| {
                    if b {
                        c += 1;
                    }
                    c as f64 / (k + 1) as f64
                })
                .collect()
        };
        let trends_neg = trends(&m_neg);
        let trends_pos = trends(&m_pos);
        let inds = [
            self.detect_peaks(&trends_neg, 0.5, 5),
            self.detect_peaks(&trends_pos, 0.5, 5),
        ];
        let mut pool: BTreeMap<usize, usize> = BTreeMap::new();
        for (i, idxs) in inds.iter().enumerate() {
            for &p_idx in idxs {
                pool.insert(p_idx, i);
            }
        }
        for (&p_idx, &i) in &pool {
            let m = if i == 0 { &m_neg } else { &m_pos };
            // seq = ''.join(str(int(x)) for x in m[:(p_idx+1)])
            let seq: Vec<u8> = m[..p_idx + 1]
                .iter()
                .map(|&b| if b { b'1' } else { b'0' })
                .collect();
            let tmp = (p_idx + 1) + self.rm + 1;
            if tmp >= MINSIZE && self.random_check(&seq, 0.05) {
                return tmp;
            }
        }
        self.dw
    }

    /// Window size per bin over `[start, end)` (TADLib `minWindows`).
    fn min_windows(&mut self, start: usize, end: usize, maxw: usize) {
        let s = start + self.chrom_len;
        let e = end + self.chrom_len;
        self.windows = vec![0i32; e - s];
        for (k, i) in (s..e).enumerate() {
            let mut down = self.raw.row_slice(i, i, i + maxw);
            let mut up = self.raw.col_slice(i, (i + 1).saturating_sub(maxw), i + 1);
            up.reverse();
            let band = (self.rm + 1).min(down.len());
            for v in down[..band].iter_mut() {
                *v = 0.0;
            }
            let band = (self.rm + 1).min(up.len());
            for v in up[..band].iter_mut() {
                *v = 0.0;
            }
            let diff: Vec<f64> = up.iter().zip(down.iter()).map(|(u, d)| u - d).collect();
            let ws = self.ori_window(&diff[self.rm + 1..]);
            self.windows[k] = ws as i32;
        }
    }

    /// DI value from upstream/downstream interaction sums (TADLib `_binbias`).
    fn binbias(&self, up: &[f64], down: &[f64]) -> f64 {
        let mut up = up.to_vec();
        let mut down = down.to_vec();
        let zeromask: Vec<bool> = up
            .iter()
            .zip(down.iter())
            .map(|(u, d)| *u != 0.0 && *d != 0.0)
            .collect();
        if zeromask.iter().filter(|&&z| z).count() >= 5 {
            let nz: Vec<usize> = (0..up.len()).filter(|&i| zeromask[i]).collect();
            up = nz.iter().map(|&i| up[i]).collect();
            down = nz.iter().map(|&i| down[i]).collect();
        }
        if up.len() <= 1 {
            return 0.0;
        }
        let upmean = up.iter().sum::<f64>() / up.len() as f64;
        let downmean = down.iter().sum::<f64>() / down.len() as f64;
        let sd1 = up.iter().map(|&x| (x - upmean).powi(2)).sum::<f64>()
            / (up.len() * (up.len() - 1)) as f64;
        let sd2 = down.iter().map(|&x| (x - downmean).powi(2)).sum::<f64>()
            / (down.len() * (down.len() - 1)) as f64;
        let sd_pool = (sd1 + sd2).sqrt();
        if sd_pool != 0.0 {
            (upmean - downmean) / sd_pool
        } else {
            0.0
        }
    }

    /// Directionality index per bin (TADLib `calDI`), with outlier trimming.
    fn cal_di(&mut self, windows: &[i32], start: usize) {
        let s = start + self.chrom_len;
        self.dis = vec![0.0; windows.len()];
        for (k, i) in (s..s + windows.len()).enumerate() {
            let mut w = windows[k] as usize;
            if w == 0 {
                w = self.dw;
            }
            let mut down = self.raw.row_slice(i, i, i + w);
            let mut up = self.raw.col_slice(i, (i + 1).saturating_sub(w), i + 1);
            up.reverse();
            down = down[self.rm + 1..].to_vec();
            up = up[self.rm + 1..].to_vec();
            let tmp = self.binbias(&up, &down);
            if tmp != 0.0 {
                self.dis[k] = tmp;
            } else if w < self.dw {
                w = self.dw;
                let mut down2 = self.raw.row_slice(i, i, i + w);
                let mut up2 = self.raw.col_slice(i, (i + 1).saturating_sub(w), i + 1);
                up2.reverse();
                down2 = down2[self.rm + 1..].to_vec();
                up2 = up2[self.rm + 1..].to_vec();
                self.dis[k] = self.binbias(&up2, &down2);
            }
        }
        // trim outliers at 0.1% / 99.9% percentiles of the negative/positive parts
        let neg: Vec<f64> = self.dis.iter().filter(|&&d| d < 0.0).copied().collect();
        let pos: Vec<f64> = self.dis.iter().filter(|&&d| d > 0.0).copied().collect();
        let lthre = percentile(&neg, 0.1);
        let hthre = percentile(&pos, 99.9);
        for d in self.dis.iter_mut() {
            if *d < lthre {
                *d = lthre;
            }
            if *d > hthre {
                *d = hthre;
            }
        }
    }

    /// Split the DI array into gap-free regions (TADLib `splitChrom`).
    fn split_chrom(&mut self, dis: &[f64]) {
        let maxgaplen = (100_000 / self.res).max(5) as usize;
        let minregion = maxgaplen * 2;
        let valid_pos: Vec<usize> = dis
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, _)| i)
            .collect();
        let mut regions: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        if valid_pos.len() > 1 {
            let gapsizes: Vec<usize> = valid_pos.windows(2).map(|w| w[1] - w[0]).collect();
            let ends_idx: Vec<usize> = gapsizes
                .iter()
                .enumerate()
                .filter(|&(_, &g)| g > maxgaplen + 1)
                .map(|(i, _)| i)
                .collect();
            let starts_idx: Vec<usize> = ends_idx.iter().map(|&e| e + 1).collect();
            for i in 0..starts_idx.len().saturating_sub(1) {
                let start = valid_pos[starts_idx[i]];
                let end = valid_pos[ends_idx[i + 1]] + 1;
                if end - start > minregion {
                    regions.insert((start, end), dis[start..end].to_vec());
                }
            }
            if !starts_idx.is_empty() {
                let start = valid_pos[*starts_idx.last().unwrap()];
                // TADLib: `end = valid_pos[-1] + 1` (the value, cut the tail
                // region at the chromosome end beyond the last valid bin)
                let end = *valid_pos.last().unwrap() + 1;
                if end - start > minregion {
                    regions.insert((start, end), dis[start..end].to_vec());
                }
                let start = valid_pos[0];
                let end = valid_pos[ends_idx[0]] + 1;
                if end - start > minregion {
                    regions.insert((start, end), dis[start..end].to_vec());
                }
            } else {
                // No gap large enough: the whole span is one region. Note
                // TADLib's `end` here is `valid_pos[-1]` (exclusive), unlike
                // the gap cases which use `+ 1`.
                let start = valid_pos[0];
                let end = valid_pos[valid_pos.len() - 1];
                if end - start > minregion {
                    regions.insert((start, end), dis[start..end].to_vec());
                }
            }
        }
        let mut gapmask = vec![true; dis.len()];
        for &(s, e) in regions.keys() {
            for g in &mut gapmask[s..e] {
                *g = false;
            }
        }
        self.region_dis = regions;
        self.gapbins = gapmask
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g)
            .map(|(i, _)| i * self.res as usize)
            .collect();
    }

    /// The 4-state Gaussian-mixture HMM from TADLib (`oriHMMParams`).
    pub fn ori_hmm_params() -> HiddenMarkovModel {
        let numdists = 3usize;
        let step = 7.5 / (numdists - 1) as f64; // 3.75, used as the Gaussian *std*
        let mut means: Vec<Vec<f64>> = vec![Vec::new(); 4];
        for i in 0..numdists {
            let v = i as f64 * step;
            means[3].push(v + 2.5);
            means[2].push(v);
            means[1].push(-v);
            means[0].push(-v - 2.5);
        }
        let mut m = HiddenMarkovModel::new("domaincaller");
        let mut states = [0usize; 4];
        for (i, ms) in means.iter().enumerate() {
            let components: Vec<NormalDistribution> = ms
                .iter()
                .map(|&mu| NormalDistribution::new(mu, step))
                .collect();
            let gmm = GeneralMixtureModel::new(components, None);
            let eid = m.add_emission(Box::new(gmm));
            states[i] = m.add_state(i.to_string(), eid, 1.0);
        }
        m.add_transition(START, states[0], 1.0);
        m.add_transition(states[0], states[1], 1.0);
        m.add_transition(states[1], states[1], 0.5);
        m.add_transition(states[1], states[2], 0.5);
        m.add_transition(states[2], states[2], 0.5);
        m.add_transition(states[2], states[3], 0.5);
        m.add_transition(states[3], states[0], 1.0);
        m.add_transition(states[3], END, 1.0);
        m.bake();
        m
    }

    /// Collect training sequences (non-zero DI segments, length > 20).
    /// Must be called after [`Chrom::compute_di`] (TADLib `Genome.train_data`).
    pub fn train_data(&self) -> Vec<Vec<f64>> {
        self.region_dis
            .values()
            .map(|seg| seg.iter().filter(|&&v| v != 0.0).copied().collect())
            .filter(|seg: &Vec<f64>| seg.len() > 20)
            .collect()
    }

    /// Train the 4-state HMM on DI segments (TADLib `Genome.learning`:
    /// Baum-Welch from `oriHMMParams`). Genome-wide callers train once on the
    /// concatenated segments of all chromosomes and share the model.
    pub fn train_hmm(seqs: &[Vec<f64>]) -> HiddenMarkovModel {
        let mut hmm = Self::ori_hmm_params();
        hmm.fit(seqs, 1e-5, 10_000, 0, false);
        hmm
    }

    /// Attach a pre-trained HMM before decoding (TADLib `Genome.learning`
    /// writes the shared model back into every cached `Chrom`). Passing
    /// `Rc::clone` of one model lets every chromosome share it.
    pub fn set_hmm(&mut self, hmm: Rc<HiddenMarkovModel>) {
        self.hmm = Some(hmm);
    }

    /// hitad `Chrom` limits (called before any compute): `maxapart` (max
    /// domain size in bp), the small-chromosome window guard
    /// `_dw = min(defaultwindow // res, chromLen // 2)` (the domaincaller
    /// variant has no guard and fails on chr4-sized chromosomes) and hitad's
    /// `refNoise`, which returns 0 instead of 1 for degenerate domains.
    pub fn set_maxapart(&mut self, maxapart: u64) {
        self.maxapart = maxapart;
        self.dw = ((DEFAULT_WINDOW / self.res.max(1)) as usize).min(self.chrom_len / 2);
        self.refnoise_degenerate = 0.0;
    }

    /// Decode a DI segment into a state sequence (TADLib `viterbi`). If the
    /// trained model makes the segment impossible (viterbi logp = -inf, which
    /// pomegranate hits on degenerate data and turns into a crash), an empty
    /// path is returned and the region simply yields no domains.
    fn viterbi(&self, seq: &[f64]) -> Vec<usize> {
        let (_, path) = self.hmm.as_ref().unwrap().viterbi(seq);
        // strip the silent start/end states ([1:-1])
        if path.len() < 2 {
            return Vec::new();
        }
        path[1..path.len() - 1].to_vec()
    }

    /// Boundary positions from a state sequence (TADLib `_getBounds`).
    fn get_bounds(&self, path: &[usize], junctions: &[&str]) -> Vec<usize> {
        let pathseq: String = path.iter().map(|&s| s.to_string()).collect();
        let mut pieces = vec![pathseq];
        for &junc in junctions {
            let mut gen = Vec::new();
            for seq in &pieces {
                let tmp: Vec<&str> = seq.split(junc).collect();
                if tmp.len() == 1 {
                    gen.push(tmp[0].to_string());
                } else {
                    gen.push(format!("{}{}", tmp[0], &junc[..1]));
                    for s in &tmp[1..tmp.len() - 1] {
                        gen.push(format!("{}{}{}", &junc[1..], s, &junc[..1]));
                    }
                    gen.push(format!("{}{}", &junc[1..], tmp[tmp.len() - 1]));
                }
            }
            pieces = gen;
        }
        let mut bounds = vec![0usize];
        let mut acc = 0usize;
        for p in &pieces {
            acc += p.len();
            bounds.push(acc);
        }
        bounds
    }

    /// Transform a DI segment into domains (TADLib `pipe`).
    fn pipe(&self, seq: &[f64], start: usize) -> Vec<[f64; 4]> {
        let bounds = self.get_bounds(&self.viterbi(seq), &["30"]);
        bounds
            .windows(2)
            .map(|w| {
                [
                    w[0] as f64 + start as f64,
                    w[1] as f64 + start as f64,
                    0.0,
                    0.0,
                ]
            })
            .collect()
    }

    /// Bottom domain list for each region (TADLib `minCore`).
    fn min_core(
        &self,
        region_dis: &BTreeMap<(usize, usize), Vec<f64>>,
    ) -> BTreeMap<(usize, usize), Vec<[f64; 4]>> {
        let mut tmp = BTreeMap::new();
        for (&(rs, re), seq) in region_dis {
            let domains = self.pipe(seq, rs);
            let cr = (rs * self.res as usize, re * self.res as usize);
            let mut out = Vec::new();
            for mut d in domains {
                d[0] *= self.res as f64;
                d[1] *= self.res as f64;
                d[2] = self.ref_noise(&d);
                out.push(d);
            }
            tmp.insert(cr, out);
        }
        self.ori_filter(tmp)
    }

    /// Keep domains at least `MINSIZE * res` long (TADLib `_orifilter`).
    fn ori_filter(
        &self,
        ori: BTreeMap<(usize, usize), Vec<[f64; 4]>>,
    ) -> BTreeMap<(usize, usize), Vec<[f64; 4]>> {
        let mut filtered = BTreeMap::new();
        for (region, list) in ori {
            let kept: Vec<[f64; 4]> = list
                .into_iter()
                .filter(|d| d[1] - d[0] >= (MINSIZE * self.res as usize) as f64)
                .collect();
            if !kept.is_empty() {
                filtered.insert(region, kept);
            }
        }
        filtered
    }

    /// Submatrix symmetrized like TADLib `getSelfMatrix` (start/end in bp).
    fn get_self_matrix(&self, start: usize, end: usize) -> Array2<f64> {
        let si = start / self.res as usize + self.chrom_len;
        let ei = end / self.res as usize + self.chrom_len;
        let mut m = self.raw.submatrix(si, ei);
        let (nr, nc) = (m.nrows(), m.ncols());
        for r in 0..nr {
            for c in 0..nc {
                if m[[r, c]] != 0.0 {
                    m[[c, r]] = m[[r, c]];
                }
            }
        }
        m
    }

    /// Noise level of a domain: zero-entry ratio off the diagonal.
    fn ref_noise(&self, domain: &[f64; 4]) -> f64 {
        if domain[1] - domain[0] < (self.res * MINSIZE as u64) as f64 {
            return self.refnoise_degenerate;
        }
        let m = self.get_self_matrix(domain[0] as usize, domain[1] as usize);
        let n = m.nrows();
        // total = n^2 - sum(arange(n, n-rm-1, -1)) * 2 + n
        let band_sum: f64 = ((self.rm + 1) * (n + n - self.rm)) as f64 / 2.0;
        let total = (n * n) as f64 - band_sum * 2.0 + n as f64;
        if total < 5.0 {
            return self.refnoise_degenerate;
        }
        let mut sig = 0usize;
        for r in 0..n {
            for c in 0..n {
                if m[[r, c]] != 0.0 && (c as isize - r as isize).unsigned_abs() > self.rm {
                    sig += 1;
                }
            }
        }
        1.0 - sig as f64 / total
    }

    /// Mismatch ratio between two bottom-domain lists (TADLib `iterCore`),
    /// via the domain aligner's `conserved` count.
    fn iter_core(
        &self,
        min_domains: &BTreeMap<(usize, usize), Vec<[f64; 4]>>,
        tmp_domains: &BTreeMap<(usize, usize), Vec<[f64; 4]>>,
    ) -> f64 {
        let reflist: Vec<Domain> = min_domains
            .values()
            .flatten()
            .map(|d| (self.chrom.clone(), d[0] as usize, d[1] as usize, 0usize))
            .collect();
        if reflist.is_empty() {
            return 1.0;
        }
        let alignlist: Vec<Domain> = tmp_domains
            .values()
            .flatten()
            .map(|d| (self.chrom.clone(), d[0] as usize, d[1] as usize, 0usize))
            .collect();
        let ref_set = DomainSet::new("ref", &reflist, self.res as usize);
        let align_set = DomainSet::new("align", &alignlist, self.res as usize);
        let n_ref = ref_set.domains.len();
        let mut worker = DomainAligner::new(vec![ref_set, align_set]);
        worker.align("ref", "align");
        let count = worker.conserved("ref", "align").len();
        1.0 - count as f64 / n_ref as f64
    }

    /// Iteratively refine windows and bottom domains (TADLib `oriIter`),
    /// breaking early when the mismatch ratio drops below 0.05.
    fn ori_iter(&mut self) {
        let mut min_domains: BTreeMap<(usize, usize), Vec<[f64; 4]>> = BTreeMap::new();
        for _ in 0..5 {
            let tmp_domains = self.min_core(&self.region_dis);
            let tol = self.iter_core(&min_domains, &tmp_domains);
            min_domains = tmp_domains;
            for list in min_domains.values() {
                for d in list {
                    let ds = d[0] as usize / self.res as usize;
                    let de = d[1] as usize / self.res as usize;
                    let len = de - ds;
                    // pyramid window: max of 1..len and len..1
                    for (k, (a, b)) in (1..=len).zip((1..=len).rev()).enumerate() {
                        if ds + k < self.windows.len() {
                            self.windows[ds + k] = a.max(b) as i32;
                        }
                    }
                }
            }
            let windows = self.windows.clone();
            self.cal_di(&windows, 0);
            let dis = self.dis.clone();
            self.split_chrom(&dis);
            if tol < 0.05 {
                break;
            }
        }
        self.min_domains = min_domains;
    }

    /// Adaptive windows, DI and region split for the whole chromosome
    /// (TADLib `minWindows` + `calDI` + `splitChrom`).
    pub fn compute_di(&mut self) {
        self.min_windows(0, self.chrom_len, self.dw);
        let windows = self.windows.clone();
        self.cal_di(&windows, 0);
        let dis = self.dis.clone();
        self.split_chrom(&dis);
    }

    fn finish(&mut self) {
        self.domains = self.min_domains.values().flatten().cloned().collect();
    }

    /// Decode with the attached HMM (TADLib `Chrom.callDomains` after
    /// `Genome.learning`): minWindows/calDI/splitChrom, then iterative
    /// refinement. Requires [`Chrom::set_hmm`] beforehand.
    pub fn call_with_hmm(&mut self) {
        self.compute_di();
        self.ori_iter();
        self.finish();
    }

    /// Run the full self-contained pipeline on one chromosome: adaptive
    /// windows, DI, split, HMM training and domain calling. For whole-genome
    /// runs with a shared HMM, use [`Chrom::set_hmm`] + [`Chrom::call_with_hmm`]
    /// instead (mirrors TADLib's `Genome` flow).
    pub fn call_domains(&mut self) {
        self.compute_di();
        let seqs = self.train_data();
        let hmm = Rc::new(Self::train_hmm(&seqs));
        self.hmm = Some(hmm);
        self.call_with_hmm();
    }

    // ------------------------------------------------------------------
    // hitad hierarchy layer (TADLib `tadlib/hitad/chromLev.py`), stacked on
    // the shared DI/oriIter base above. Requires `set_hmm` (a genome-wide
    // trained model) and `set_maxapart` beforehand.
    // ------------------------------------------------------------------

    /// Insulation score at bin `idx` (TADLib hitad `calIS`, default `w = 5`):
    /// mean of the `w x w` block upstream of the diagonal point.
    fn cal_is(&self, idx: usize, w: usize) -> f64 {
        let i = idx + self.chrom_len;
        let mut sum = 0.0;
        for k in i - w..i {
            sum += self.raw.row_slice(k, i + 1, i + w + 1).iter().sum::<f64>();
        }
        sum / (w * w) as f64
    }

    /// TAD score of a `[start, end)` bp interval (TADLib hitad `stablescore` +
    /// `idxmatch`): mean contrast between the intra-domain interactions and
    /// the distance-matched upstream/downstream inter-domain interactions.
    /// The `weights` branch is commented out in TADLib, so the weight is a
    /// constant 0.5.
    fn stable_score(&self, domain: (usize, usize)) -> f64 {
        let res = self.res as usize;
        let ds = domain.0 / res + self.chrom_len;
        let de = domain.1 / res + self.chrom_len;
        // idxmatch skips domains shorter than 2 bins
        if de - ds < 2 {
            return 0.0;
        }
        let mut acc = 0.0;
        let mut n = 0usize;
        for x in ds..de {
            for y in ds..de {
                let cur = self.raw.at(x, y);
                // the pair grid covers the full square, so y - x can be
                // negative (Python's numpy handles it)
                if (y as isize - x as isize) > self.rm as isize && cur != 0.0 {
                    // upstream pair (2x - y, x); in-bounds because the
                    // matrix is shifted by chrom_len
                    let up = self.raw.at(2 * x - y, x);
                    // Umask: 2x - ori <= y (ori = ds)
                    if up != 0.0 && 2 * x - ds <= y {
                        acc += (cur - up) / (cur + up);
                        n += 1;
                    }
                    // downstream pair (y, 2y - x)
                    let down = self.raw.at(y, 2 * y - x);
                    // Dmask: y - x > rm, 2y - x >= end, all != 0 (the
                    // 2y - ori >= 2y - x term is always true since x >= ds)
                    if down != 0.0 && 2 * y - x >= de {
                        acc += (cur - down) / (cur + down);
                        n += 1;
                    }
                }
            }
        }
        if n > 3 {
            acc * 0.5 / n as f64
        } else {
            0.0
        }
    }

    /// Pre-compute TAD scores for all runs of consecutive bottom domains
    /// (TADLib hitad `scoreCache`); domains with noise > 0.2 stay alone.
    /// Keys are `(start, end)` bp pairs; `cache` is shared across regions
    /// within one `max_core` call, as upstream does.
    fn score_cache(&self, domainlist: &[[f64; 4]], cache: &mut BTreeMap<(usize, usize), f64>) {
        for i in 0..domainlist.len() {
            let key = (domainlist[i][0] as usize, domainlist[i][1] as usize);
            cache.entry(key).or_insert_with(|| self.stable_score(key));
            if domainlist[i][2] > 0.2 {
                continue;
            }
            for j in i + 1..domainlist.len() {
                if domainlist[j][1] - domainlist[i][0] > self.maxapart as f64 {
                    break;
                }
                if domainlist[j][2] > 0.2 {
                    continue;
                }
                let key = (domainlist[i][0] as usize, domainlist[j][1] as usize);
                cache.entry(key).or_insert_with(|| self.stable_score(key));
            }
        }
    }

    /// Dynamic programming for the best separation of a bottom-domain list
    /// given cached TAD scores (TADLib hitad `maxscorepath`). Returns the
    /// merged `(start-index, end-index)` pairs of the winning path. Entries
    /// keep Python dict insertion order for tie-breaking parity.
    fn max_score_path(
        &self,
        domainlist: &[(usize, usize)],
        scores: &BTreeMap<(usize, usize), f64>,
    ) -> BTreeSet<(usize, usize)> {
        let n = domainlist.len();
        let res = self.res as usize;
        // paths[i]: insertion-ordered (key, (score, parent)) entries
        type PathEntry = ((usize, usize), (f64, (isize, isize)));
        let mut paths: Vec<Vec<PathEntry>> = Vec::with_capacity(n);
        for i in 0..n {
            let lens = (domainlist[i].1 / res - domainlist[i].0 / res) as f64;
            let mut cur: Vec<PathEntry> = Vec::new();
            let mut maxp: (isize, isize) = (-1, -1);
            let mut maxs = f64::NEG_INFINITY;
            if i == 0 {
                // pre = {(-1, -1): (0, sentinel)}: no carried entries and
                // maxs = 0 with the (-1, -1) parent
                maxs = 0.0;
            } else {
                for &(pp, (score, _)) in &paths[i - 1] {
                    if (i as isize) <= pp.1 as isize {
                        let unit = scores
                            .get(&(domainlist[pp.0].0, domainlist[pp.1].1))
                            .expect("maxscorepath: missing carried score");
                        cur.push((pp, (score + lens * unit, (pp.0 as isize, pp.1 as isize))));
                    } else if score > maxs {
                        maxs = score;
                        maxp = (pp.0 as isize, pp.1 as isize);
                    }
                }
            }
            for j in i..n {
                let key = (domainlist[i].0, domainlist[j].1);
                if (key.1 - key.0) as u64 > self.maxapart && j > i {
                    break;
                }
                let Some(&unit) = scores.get(&key) else {
                    continue;
                };
                cur.push(((i, j), (maxs + lens * unit, maxp)));
            }
            paths.push(cur);
        }
        // Backtracking
        let mut lasts = f64::NEG_INFINITY;
        let mut lastp: Option<(usize, usize)> = None;
        for &(pp, (score, _)) in &paths[n - 1] {
            if score > lasts {
                lasts = score;
                lastp = Some(pp);
            }
        }
        let Some(mut lastp) = lastp else {
            // Python returns None here (and the caller then fails); keep an
            // empty result instead
            return BTreeSet::new();
        };
        let mut bestpath = vec![lastp];
        for i in (0..n).rev() {
            let Some(&(_key, (_score, parent))) = paths[i].iter().find(|(k, _)| *k == lastp) else {
                panic!("maxscorepath: lost the backtrack path at index {i}");
            };
            if parent == (-1, -1) {
                break;
            }
            lastp = (parent.0 as usize, parent.1 as usize);
            bestpath.insert(0, lastp);
        }
        bestpath.into_iter().collect()
    }

    /// Identify TADs from the bottom domains (TADLib hitad `maxCore`): per
    /// gap-free region, cached TAD scores + `maxscorepath`, then the
    /// `MINSIZE * res` filter. Sets `max_domains`.
    ///
    /// `cache` mirrors TADLib's `def maxCore(self, cache={})` **mutable
    /// default argument**: one dict shared by every `maxCore` call in the
    /// same process, so a whole-genome run reuses earlier chromosomes'
    /// scores whenever two chromosomes have domains at the same
    /// `(start, end)` bp pair. That cross-chromosome reuse is TADLib's real
    /// CLI behavior and shapes its reference output, so the shared cache
    /// must be threaded through in chromosome order.
    fn max_core(&mut self, cache: &mut BTreeMap<(usize, usize), f64>) {
        let mut max_domains: BTreeMap<(usize, usize), Vec<[f64; 4]>> = BTreeMap::new();
        for (&region, rdomains) in &self.min_domains {
            if rdomains.is_empty() {
                continue;
            }
            if rdomains.len() < 2 {
                max_domains.insert(region, rdomains.clone());
            } else {
                self.score_cache(rdomains, cache);
                let pairs: Vec<(usize, usize)> = rdomains
                    .iter()
                    .map(|d| (d[0] as usize, d[1] as usize))
                    .collect();
                let premerge = self.max_score_path(&pairs, cache);
                let mut newdomain: Vec<[f64; 4]> = Vec::new();
                for b in premerge {
                    let tl = &rdomains[b.0..=b.1];
                    newdomain.push([tl[0][0], tl[tl.len() - 1][1], 0.0, 0.0]);
                }
                newdomain.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if !newdomain.is_empty() {
                    max_domains.insert(region, newdomain);
                }
            }
        }
        self.max_domains = self.ori_filter(max_domains);
    }

    /// Arrowhead-transformed matrix of `[start, end)` bp (TADLib hitad
    /// `toArrowhead`), symmetrized first like the Python `raw[ty, tx] =
    /// raw[tx, ty]` pass over nonzero pairs.
    fn to_arrowhead(&self, start: usize, end: usize) -> Array2<f64> {
        let res = self.res as usize;
        let startidx = start / res + self.chrom_len;
        let endidx = end / res + self.chrom_len;
        let maxd = endidx - startidx - 1;
        let forward = startidx - maxd;
        let mut raw = self.raw.submatrix(forward, endidx);
        let n = raw.nrows();
        for i in 0..n {
            for j in 0..n {
                if raw[[i, j]] != 0.0 {
                    raw[[j, i]] = raw[[i, j]];
                }
            }
        }
        let size = maxd + 1;
        let mut a = Array2::zeros((size, size));
        for x in 0..size {
            for y in x..size {
                let rawx = x + maxd;
                let rawy = y + maxd;
                let minus_y = rawx - (y - x);
                let m1 = raw[[minus_y, rawx]];
                let m2 = raw[[rawx, rawy]];
                let mut den = m1 + m2;
                if den == 0.0 {
                    den = 1.0;
                }
                let elem = (m1 - m2) / den;
                a[[x, y]] = elem;
                a[[y, x]] = elem;
            }
        }
        a
    }

    /// Resolve the domain hierarchy inside each TAD (TADLib hitad
    /// `fineDomain`). Sets `hier_domains`.
    fn fine_domain(&mut self) {
        let mut hier: BTreeMap<(usize, usize), Vec<[f64; 4]>> = BTreeMap::new();
        for (&region, rdomains) in &self.max_domains {
            if rdomains.is_empty() {
                continue;
            }
            let reflist = &self.min_domains[&region];
            let mut hdomains: Vec<[f64; 4]> = Vec::new();
            for rdomain in rdomains {
                let top = (rdomain[0] as usize, rdomain[1] as usize);
                let mut subdomains: BTreeMap<(usize, usize), usize> = BTreeMap::new();
                self.sub_domains(top, reflist, 0, None, &mut subdomains);
                if subdomains.len() == 1 {
                    let noise = self.ref_noise(&[top.0 as f64, top.1 as f64, 0.0, 0.0]);
                    hdomains.push([top.0 as f64, top.1 as f64, noise, 0.0]);
                } else {
                    for (&(s, e), &lv) in &subdomains {
                        let noise = self.ref_noise(&[s as f64, e as f64, 0.0, 0.0]);
                        hdomains.push([s as f64, e as f64, noise, lv as f64]);
                    }
                }
            }
            hdomains.sort_by(|a, b| a.partial_cmp(b).unwrap());
            hier.insert(region, hdomains);
        }
        self.hier_domains = hier;
    }

    /// Refine every hierarchical domain boundary by minimal insulation
    /// score (TADLib hitad `preciseBound`): each boundary may shift by one
    /// bin. Mutates `hier_domains` in place.
    fn precise_bound(&mut self) {
        let res = self.res as usize;
        let chrom_len = self.chrom_len;
        let mut hier = std::mem::take(&mut self.hier_domains);
        for list in hier.values_mut() {
            for d in list.iter_mut() {
                let si = d[0] as usize / res;
                let mut ini = f64::INFINITY;
                for i in si.saturating_sub(1)..(si + 2).min(chrom_len) {
                    let is = self.cal_is(i, 5);
                    if is < ini {
                        d[0] = (i * res) as f64;
                        ini = is;
                    }
                }
                let ei = d[1] as usize / res;
                let mut ini = f64::INFINITY;
                for i in ei.saturating_sub(1)..(ei + 2).min(chrom_len) {
                    let is = self.cal_is(i, 5);
                    if is < ini {
                        d[1] = (i * res) as f64;
                        ini = is;
                    }
                }
            }
        }
        self.hier_domains = hier;
    }

    /// Full hitad pipeline (TADLib hitad `Chrom.callDomain`): windows, DI,
    /// iterative bottom-domain refinement, TAD separation (`maxCore`),
    /// hierarchy resolution (`fineDomain`) and boundary refinement
    /// (`preciseBound`). Requires `set_hmm` + `set_maxapart` beforehand.
    /// Uses a private score cache, i.e. a fresh single-chromosome process
    /// (TADLib `maxCore`'s default); whole-genome runs must use
    /// [`Chrom::call_hier_domain_with_cache`] to share the cache across
    /// chromosomes like the CLI's single worker process does.
    pub fn call_hier_domain(&mut self) {
        let mut cache = BTreeMap::new();
        self.call_hier_domain_with_cache(&mut cache);
    }

    /// [`Chrom::call_hier_domain`] with the shared `maxCore` score cache
    /// (TADLib's mutable-default-argument cache; see [`Chrom::max_core`]).
    pub fn call_hier_domain_with_cache(&mut self, cache: &mut BTreeMap<(usize, usize), f64>) {
        self.compute_di();
        self.ori_iter();
        self.max_core(cache);
        self.fine_domain();
        self.precise_bound();
    }

    /// Recursion identifying the inner hierarchy of a TAD (TADLib hitad
    /// `subDomains`). `am` is the arrowhead matrix of `domain` (computed
    /// here when `None`); the `W` weight matrix is the constant 0.5 upstream
    /// (its computation is commented out in TADLib) and is folded into
    /// `sub_local_update`. `subdomains` collects `(start, end)` bp -> level.
    fn sub_domains(
        &self,
        domain: (usize, usize),
        reflist: &[[f64; 4]],
        clv: usize,
        am: Option<&Array2<f64>>,
        subdomains: &mut BTreeMap<(usize, usize), usize>,
    ) {
        subdomains.insert(domain, clv);
        let res = self.res as usize;
        let am = match am {
            Some(m) => m.to_owned(),
            None => self.to_arrowhead(domain.0, domain.1),
        };
        let mut sublist: Vec<(usize, usize)> = reflist
            .iter()
            .map(|d| (d[0] as usize, d[1] as usize))
            .filter(|&(s, e)| s >= domain.0 && e <= domain.1)
            .collect();
        sublist.sort();
        if sublist.len() <= 1 {
            return;
        }
        let d0 = domain.0 / res;
        let mut biases: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for i in 0..sublist.len() {
            sub_local_update(&am, self.rm, res, d0, sublist[i], &mut biases);
            for j in i + 1..sublist.len() {
                let key = (sublist[i].0, sublist[j].1);
                sub_local_update(&am, self.rm, res, d0, key, &mut biases);
            }
        }
        // the outer domain itself scores 0
        biases.insert(domain, 0.0);
        let prebests: Vec<(usize, usize)> =
            self.max_score_path(&sublist, &biases).into_iter().collect();
        let mut dlist = vec![sublist[0].0];
        for b in prebests.iter().take(prebests.len().saturating_sub(1)) {
            dlist.push(sublist[b.1].1);
        }
        dlist.push(sublist[sublist.len() - 1].1);
        if dlist.len() == 2 {
            return;
        }
        for w in dlist.windows(2) {
            let tmpdomain = (w[0], w[1]);
            let tmpori = tmpdomain.0 / res - d0;
            let tmpend = tmpdomain.1 / res - d0;
            let tam = am.slice(s![tmpori..tmpend, tmpori..tmpend]).to_owned();
            self.sub_domains(tmpdomain, reflist, clv + 1, Some(&tam), subdomains);
        }
    }
}

/// `localUpdate` inside TADLib hitad `subDomains`: arrowhead contrast for
/// one `(start, end)` bp key. `d0` is the outer domain's start bin (the local
/// origin of `am`); the W weight matrix is a constant 0.5 upstream (its
/// computation is commented out in TADLib) and is folded in.
fn sub_local_update(
    am: &Array2<f64>,
    rm: usize,
    res: usize,
    d0: usize,
    key: (usize, usize),
    biases: &mut BTreeMap<(usize, usize), f64>,
) {
    let o = key.0 / res - d0;
    let e = key.1 / res - d0;
    let n = am.nrows();
    let mut sum = 0.0;
    let mut count = 0usize;
    for x in 0..n {
        for y in 0..n {
            let a = am[[x, y]];
            // common: y - x > rm and aM not +/- 1
            if (y as isize - x as isize) <= rm as isize {
                continue;
            }
            if a == 1.0 || a == -1.0 {
                continue;
            }
            // mask1: y >= e, x < e, 2x - o >= y
            if y >= e && x < e && 2 * x as isize - o as isize >= y as isize {
                sum += a;
                count += 1;
            }
            // mask2: y < e, x >= o, 2x >= y, 2x - o <= y
            if y < e
                && x >= o
                && 2 * x as isize >= y as isize
                && 2 * x as isize - o as isize <= y as isize
            {
                sum -= a;
                count += 1;
            }
        }
    }
    if count > 3 {
        biases.insert(key, sum * 0.5 / count as f64);
    } else {
        biases.insert(key, 0.0);
    }
}

/// numpy `percentile` with the default 'linear' method.
fn percentile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut a = data.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let pos = (a.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return a[lo];
    }
    let frac = pos - lo as f64;
    a[lo] * (1.0 - frac) + a[hi] * frac
}
