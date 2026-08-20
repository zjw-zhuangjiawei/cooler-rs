use ndarray::{Array2, ArrayView2, ArrayViewMut2};

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::types::ChromMeta;

use super::{Params, Tad};

/// Extract a single-chromosome matrix from a cooler collection, keeping
/// only the band `|bin2 - bin1| <= band` (like [`load_matrix`]).
///
/// `chr` selects the chromosome by name; it may be omitted when the file
/// contains exactly one chromosome.
pub fn matrix_from_cooler(
    cool: &Cooler,
    chr: Option<&str>,
    band: usize,
) -> Result<(Array2<f64>, ChromMeta)> {
    let chroms = cool.chroms()?;
    let chrom_id = match chr {
        Some(name) => chroms
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| {
                let available = chroms
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                Error::InvalidInput(format!(
                    "chromosome '{name}' not found (available: {available})"
                ))
            })?,
        None if chroms.len() == 1 => 0,
        None => {
            let available = chroms
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::InvalidInput(format!(
                "file contains {} chromosomes; select one with -chr ({available})",
                chroms.len()
            )));
        }
    };

    let offsets = cool.chrom_offset()?;
    let first = offsets[chrom_id] as usize;
    let last = offsets[chrom_id + 1] as usize;
    let l = last - first;
    let mut x = Array2::zeros((l, l));

    for p in cool.pixels()? {
        // Pixels are stored symmetric-upper (bin1_id <= bin2_id).
        let (b1, b2) = (p.bin1_id as usize, p.bin2_id as usize);
        if b1 >= first && b2 < last && b2 - b1 <= band {
            x[[b1 - first, b2 - first]] = p.count;
            x[[b2 - first, b1 - first]] = p.count;
        }
    }

    let chrom = &chroms[chrom_id];
    let meta = ChromMeta {
        name: chrom.name.clone(),
        length: chrom.length as u64,
        resolution: cool.bin_size()?.ok_or_else(|| {
            Error::Format("missing 'bin-size' attribute".into())
        })?,
    };
    Ok((x, meta))
}

// ---------------------------------------------------------------------------
// Step 2: corner score and candidate boundaries
// ---------------------------------------------------------------------------

/// 2D prefix sums: `sx[i][j]` = sum of `x[0..=i][0..=j]`.
pub fn cumsum(x: ArrayView2<'_, f64>) -> Array2<f64> {
    let l = x.nrows();
    let mut sx = x.to_owned();
    for i in 0..l {
        for j in 1..l {
            sx[[i, j]] += sx[[i, j - 1]];
        }
        if i > 0 {
            for j in 0..l {
                sx[[i, j]] += sx[[i - 1, j]];
            }
        }
    }
    sx
}

/// Mean-contact score for each candidate corner, `score[i][j]` for a corner
/// at row `i` reaching `j + 1` bins up/right (C++ `getScore`).
pub fn get_score(sx: ArrayView2<'_, f64>, maxsz: usize) -> Array2<f64> {
    let l = sx.nrows();
    let mut score = Array2::zeros((l, maxsz));
    for i in 1..l {
        for j in 0..maxsz {
            let col = (l - 1).min(i + j + 1);
            let mut s = sx[[i - 1, col]] - sx[[i - 1, i]];
            if i >= j + 2 {
                s += -sx[[i - j - 2, col]] + sx[[i - j - 2, i]];
            }
            let k = (l - i).min(j + 1) * i.min(j + 1);
            score[[i, j]] = s / k as f64;
        }
    }
    score
}

/// Mark local minima of the score matrix (C++ `calMins`, including its
/// `min(l, ...)` bound on the window check).
pub fn cal_mins(score: ArrayView2<'_, f64>, lsize: usize, ldiff: f64) -> Array2<bool> {
    let l = score.nrows();
    let n = score.ncols();
    let mut lm = Array2::from_elem((l, n), false);

    // Rows with non-zero variance across columns.
    let mut map: Vec<usize> = Vec::new();
    for i in 0..l {
        let mut m = 0.0;
        let mut s = 0.0;
        for j in 0..n {
            let v = score[[i, j]];
            m += v;
            s += v * v;
        }
        if s > m * m / n as f64 {
            map.push(i);
        }
    }
    if map.len() < 2 {
        return lm;
    }

    for i in 0..n {
        let mut m = 0.0;
        let mut s = 0.0;
        for w in map.windows(2) {
            let d = score[[w[1], i]] - score[[w[0], i]];
            m += d;
            s += d * d;
        }
        m /= (map.len() - 1) as f64;
        s = (s / (map.len() - 1) as f64 - m * m).sqrt();
        let cut = s * ldiff;
        for j in 0..map.len() {
            let mins = score[[map[j], i]];
            let mut maxs = mins;
            let mut k = j.saturating_sub(lsize);
            while k < map.len().min(j + lsize + 1) {
                if mins >= score[[map[k], i]] && k != j {
                    break;
                }
                maxs = maxs.max(score[[map[k], i]]);
                k += 1;
            }
            if k >= l.min(j + lsize + 1) && maxs - mins > cut {
                lm[[map[j], i]] = true;
            }
        }
        lm[[0, i]] = true;
        lm[[l - 1, i]] = true;
    }
    lm
}

/// From local minima, mark allowed boundary pairs (C++ `setPair`): pairs of
/// minima up to 5 apart, each paired with the first/last row.
pub fn set_pair(lm: ArrayView2<'_, bool>) -> Array2<bool> {
    let l = lm.nrows();
    let n = lm.ncols();
    let mut sel = Array2::from_elem((l, l), false);
    for i in 0..n {
        let map: Vec<usize> = (0..l).filter(|&j| lm[[j, i]]).collect();
        for j in 0..map.len() {
            for k in 1..=5 {
                if j + k >= map.len() {
                    break;
                }
                sel[[0, map[j]]] = true;
                sel[[0, map[j + k]]] = true;
                sel[[map[j], l - 1]] = true;
                sel[[map[j + k], l - 1]] = true;
                sel[[map[j], map[j + k]]] = true;
            }
        }
    }
    sel
}

// ---------------------------------------------------------------------------
// Step 3: remove distance effect
// ---------------------------------------------------------------------------

/// Standardize each diagonal of the (symmetric) matrix in place
/// (C++ `HiCnorm`, called there with `band = maxsz * 2`).
pub fn hicnorm(x: &mut ArrayViewMut2<'_, f64>, band: usize) {
    let l = x.nrows();
    if l < 2 {
        return;
    }
    for i in 0..(l - 1).min(band) {
        let mut tm = 0.0;
        let mut ts = 0.0;
        for j in 0..l - i {
            let v = x[[j, j + i]];
            x[[j + i, j]] = v;
            tm += v;
            ts += v * v;
        }
        tm /= (l - i) as f64;
        ts = (1e-6_f64).max(ts / (l - i) as f64 - tm * tm).sqrt();
        for j in 0..l - i {
            x[[j, j + i]] = (x[[j, j + i]] - tm) / ts + 1.0;
            if i > 0 {
                x[[j + i, j]] = (x[[j + i, j]] - tm) / ts + 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4: dynamic programming
// ---------------------------------------------------------------------------

/// Memoization tables shared by the recursive DP (the C++ globals `S`, `M`, `B`).
struct DpTables {
    /// Best score for interval `[st, ed)` (index `[st][ed - 1]`), None = not computed.
    s: Array2<Option<f64>>,
    /// Mean contact of interval `[st, ed)`.
    m: Array2<f64>,
    /// Split points per interval: `(st * l + ed, boundary)`.
    b: Vec<(usize, usize)>,
}

impl DpTables {
    fn new(l: usize) -> Self {
        DpTables {
            s: Array2::from_elem((l, l), None),
            m: Array2::zeros((l, l)),
            b: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dpcall(
    x: ArrayView2<'_, f64>,
    sx: ArrayView2<'_, f64>,
    st: usize,
    ed: usize,
    minsz: usize,
    maxsz: usize,
    penalty: f64,
    sel: ArrayView2<'_, bool>,
    tables: &mut DpTables,
) -> (f64, f64) {
    if let Some(s) = tables.s[[st, ed - 1]] {
        return (s, tables.m[[st, ed - 1]]);
    }

    let l = ed - st;
    let mut ssum = sx[[ed - 1, ed - 1]];
    if st > 0 {
        ssum = ssum - sx[[st - 1, ed - 1]] - sx[[ed - 1, st - 1]] + sx[[st - 1, st - 1]];
    }
    let sn = (l * l) as f64;

    let mut trace: Vec<usize> = Vec::new();
    let (mut score, mean);
    if l <= minsz.max(1) {
        score = 0.0;
        mean = ssum / sn;
    } else {
        let mut rtscore: Vec<Option<f64>> = vec![None; l];
        rtscore[0] = Some(0.0);
        let mut ttrace = vec![-1_i64; l];
        for i in 1..l {
            if i < l - 1 {
                let has_pair = (i + 1..l).any(|j| sel[[st + i, st + j]]);
                if !has_pair {
                    continue;
                }
            }
            let mut tscore: Vec<Option<f64>> = vec![None; i];
            let mut flag = false;
            let mut k = 0_usize;
            for j in (0..i).rev() {
                if !sel[[st + j, st + i]] {
                    continue;
                }
                if flag && i - j > maxsz {
                    break;
                }
                flag = true;
                let ts = if j == 0 && i == l - 1 {
                    0.0
                } else {
                    dpcall(x, sx, st + j, st + i + 1, minsz, maxsz, penalty, sel, tables).0
                };
                tscore[j] = Some(ts);
                if j > 0 {
                    tscore[j] = match (tscore[j], rtscore[j]) {
                        (Some(ts_j), Some(rs_j)) => Some(ts_j + rs_j),
                        _ => None,
                    };
                }
                if tscore[k] <= tscore[j] {
                    k = j;
                }
            }
            rtscore[i] = tscore[k];
            ttrace[i] = k as i64;
        }

        // Traceback: subtract the score of accepted sub-TADs.
        let mut tsum = 0.0;
        let mut tn = 0.0;
        let mut j = l - 1;
        let mut flag = false;
        while j > 0 {
            let k = ttrace[j];
            if k < 0 {
                break;
            }
            let k = k as usize;
            if k > 0 {
                trace.push(k);
            }
            if rtscore[j].unwrap_or(0.0) - rtscore[k].unwrap_or(0.0) > 0.0 {
                tsum += sx[[st + j, st + j]];
                if st + k > 0 {
                    tsum = tsum - sx[[st + k - 1, st + j]] - sx[[st + j, st + k - 1]]
                        + sx[[st + k - 1, st + k - 1]];
                }
                tn += ((j - k + 1) * (j - k + 1)) as f64;
                if flag {
                    tsum -= x[[st + j, st + j]];
                    tn -= 1.0;
                }
                flag = true;
            } else {
                flag = false;
            }
            j = k;
        }
        mean = (ssum - tsum) / (sn - tn + 1e-5);
        score = rtscore[l - 1].unwrap_or(0.0);
    }

    // Boundary contrast: mean inside vs. mean of flanking regions.
    let mut bsuml = 0.0;
    let mut bsumr = 0.0;
    let mut bnl = 1e-5;
    let mut bnr = 1e-5;
    if st > 0 {
        let ta = st.saturating_sub(l - 1);
        bsuml += sx[[st - 1, ed - 1]];
        if ta > 0 {
            bsuml = bsuml - sx[[ta - 1, ed - 1]] - sx[[st - 1, st]] + sx[[ta - 1, st]];
        }
        bnl += ((st - ta) * (l - 1)) as f64;
    }
    if ed < sx.nrows() {
        let ta = (sx.nrows() - 1).min(ed + l - 2);
        bsumr += sx[[ed - 2, ta]];
        if st > 0 {
            bsumr = bsumr - sx[[st - 1, ta]] - sx[[ed - 2, ed - 1]] + sx[[st - 1, ed - 1]];
        }
        bnr += ((l - 1) * (ta + 1 - ed)) as f64;
    }
    let delta = mean - (bsuml / bnl).max(bsumr / bnr);
    score += delta;
    score -= penalty;
    score = score.max(0.0);
    tables.s[[st, ed - 1]] = Some(score);
    tables.m[[st, ed - 1]] = mean;
    for &t in &trace {
        tables.b.push((st * sx.nrows() + ed, st + t));
    }
    (score, mean)
}

#[allow(clippy::too_many_arguments)]
fn get_bound(
    st: usize,
    ed: usize,
    level: usize,
    x: ArrayView2<'_, f64>,
    sx: ArrayView2<'_, f64>,
    tables: &DpTables,
    tad: &mut Tad,
) {
    let l = sx.nrows();
    let mut _s = sx[[ed - 1, ed - 1]];
    if st > 0 {
        _s = _s - sx[[st - 1, ed - 1]] - sx[[ed - 1, st - 1]] + sx[[st - 1, st - 1]];
    }
    let mut _n = ((ed - st) * (ed - st)) as f64;
    let key = st * l + ed;

    let mut loc: Vec<usize> = tables
        .b
        .iter()
        .filter(|(k, _)| *k == key)
        .map(|&(_, v)| v)
        .collect();
    loc.sort_unstable();
    if !loc.is_empty() {
        loc.insert(0, st);
        loc.push(ed - 1);
        let mut flag = false;
        for i in (0..loc.len() - 1).rev() {
            get_bound(loc[i], loc[i + 1] + 1, level + 1, x, sx, tables, tad);
            if tables.s[[loc[i], loc[i + 1]]].unwrap_or(0.0) > 0.0 {
                let mut ts = sx[[loc[i + 1], loc[i + 1]]];
                if loc[i] > 0 {
                    ts = ts - sx[[loc[i] - 1, loc[i + 1]]] - sx[[loc[i + 1], loc[i] - 1]]
                        + sx[[loc[i] - 1, loc[i] - 1]];
                }
                _s -= ts;
                let tn = ((loc[i + 1] - loc[i] + 1) * (loc[i + 1] - loc[i] + 1)) as f64;
                _n -= tn;
                if flag {
                    _s += x[[loc[i + 1], loc[i + 1]]];
                    _n += 1.0;
                }
                flag = true;
            } else {
                flag = false;
            }
        }
    }

    if tables.s[[st, ed - 1]].unwrap_or(0.0) > 0.0 {
        tad.bound.push([st, ed - 1]);
        tad.level.push(level);
        tad.score.push(tables.s[[st, ed - 1]].unwrap_or(0.0));
        tad.mean.push(tables.m[[st, ed - 1]]);
    }
}

/// Run the full OnTAD pipeline on a loaded matrix and return the TAD calls.
///
/// The matrix is modified in place by [`hicnorm`].
pub fn call_tads(x: &mut Array2<f64>, params: &Params) -> Tad {
    let l = x.nrows();
    log::info!("Cumsum data:");

    let sx = cumsum(x.view());
    log::info!("Calculate score:");
    let score = get_score(sx.view(), params.maxsz);
    log::info!("Find local min:");
    let lm = cal_mins(score.view(), params.lsize, params.ldiff);
    let sel = set_pair(lm.view());
    log::info!("Normalize data:");
    hicnorm(&mut x.view_mut(), params.maxsz * 2);

    log::info!("Cumsum data:");

    let sx = cumsum(x.view());
    log::info!("Call TADs:");
    let mut tables = DpTables::new(l);
    dpcall(
        x.view(),
        sx.view(),
        0,
        l,
        params.minsz,
        params.maxsz,
        params.penalty,
        sel.view(),
        &mut tables,
    );
    let mut tad = Tad::default();
    get_bound(0, l, 0, x.view(), sx.view(), &tables, &mut tad);
    // get_bound emits innermost-first at the top level; reverse like the C++.
    tad.bound.reverse();
    tad.level.reverse();
    tad.score.reverse();
    tad.mean.reverse();
    tad
}
