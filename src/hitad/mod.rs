//! Whole-genome hierarchical domain calling: the TADLib `hitad` pipeline
//! (`tadlib/hitad/genomeLev.py` `Genome` + the hierarchy layer of
//! `tadlib/hitad/chromLev.py`), as run by `scripts/hitad`:
//!
//! 1. [`Genome::new`] loads every non-excluded chromosome of every dataset
//!    (resolution x replicate) as a [`Chrom`] from the balance-corrected
//!    upper-triangle matrix; chromosomes below `min_chrom_size` are excluded
//!    automatically (the CLI `--minimum-chrom-size`);
//! 2. [`Genome::learning`] trains one shared HMM per dataset (Baum-Welch on
//!    the non-zero DI segments of all its chromosomes);
//! 3. [`Genome::call_domains`] runs the hierarchical
//!    [`Chrom::call_hier_domain`] per chromosome, merges replicates through
//!    the `MultiReps` layer and keeps the minimum-resolution results;
//! 4. [`Genome::di_track`] exposes the per-chromosome DI tracks.
//!
//! Not ported: the `bins/DIs` column write-back (the DIs stay in memory,
//! `di_track` reads them out), the pickle caches (everything is in memory)
//! and the `plot` layer. Multi-replicate consensus (`repAligner` and the
//! cross-replicate `MultiReps` correction) is also not ported;
//! single-replicate datasets take the identity path of
//! `MultiReps.callDomain`, which is the same code TADLib runs for one
//! replicate.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::mcool::Mcool;

use crate::domaincaller::aligner::{hier_format, Domain, DomainSet, Node, Region};
use crate::domaincaller::chrom::Chrom;

/// One dataset row of the hitad metadata file: (resolution, replicate
/// label, path to the `.cool`/`.mcool`).
pub type Dataset = (u64, String, String);

/// Default max domain size in bp (`scripts/hitad --maxsize`).
pub const DEFAULT_MAXSIZE: u64 = 4_000_000;
/// Default minimum chromosome size in bp (`--minimum-chrom-size`).
pub const DEFAULT_MIN_CHROM_SIZE: u64 = 1_000_000;

/// Whole-genome hierarchical domain caller with one shared HMM per dataset.
pub struct Genome {
    /// Datasets (resolution, replicate label, path) in metadata order.
    pub datasets: Vec<Dataset>,
    /// Explicitly excluded chromosomes plus the `min_chrom_size` exclusions.
    pub exclude: BTreeSet<String>,
    /// Max domain size in bp (TADLib `maxapart`).
    pub maxsize: u64,
    /// Chromosomes in cooler file order (from the first dataset).
    chrom_order: Vec<String>,
    /// chrom -> resolution -> replicate -> `Chrom`.
    data: BTreeMap<String, BTreeMap<u64, BTreeMap<String, Chrom>>>,
    /// (resolution, replicate) -> [(chrom, size)] in cooler order, for
    /// `di_track` (which emits zeros for excluded chromosomes, as the CLI
    /// writes into `bins/DIs`).
    layout: BTreeMap<(u64, String), Vec<(String, u64)>>,
    /// Final consistent domains (chrom, start, end, level) from the minimum
    /// resolution, chromosomes in file order.
    pub results: Vec<Domain>,
}

impl Genome {
    /// `balance` names a `bins` column (e.g. `"weight"`); `"RAW"` in any
    /// case selects raw counts (the CLI `-W RAW`). `exclude` is the explicit
    /// exclusion list; chromosomes with fewer than `min_chrom_size` covered
    /// bp are added to it automatically. `maxsize` is the max domain size.
    pub fn new(
        datasets: &[Dataset],
        balance: Option<&str>,
        exclude: &[&str],
        min_chrom_size: u64,
        maxsize: u64,
    ) -> Result<Self> {
        if datasets.is_empty() {
            return Err(Error::InvalidInput("no datasets given".into()));
        }
        let balance = match balance {
            Some(b) if b.eq_ignore_ascii_case("raw") => None,
            other => other,
        };
        let mut exclude_set: BTreeSet<String> = exclude.iter().map(|s| s.to_string()).collect();

        // Pass 1: chromosome layouts and the min_chrom_size exclusions
        // (union over all datasets, as the CLI builds its exclusion list).
        let mut chrom_order: Vec<String> = Vec::new();
        type Layout = Vec<(String, u64, usize)>;
        let mut layouts: BTreeMap<(u64, String), Layout> = BTreeMap::new();
        for &(res, ref rep, ref path) in datasets {
            let cool = open_dataset(res, path)?;
            let chroms = cool.chroms()?;
            let mut layout = Vec::with_capacity(chroms.len());
            for ch in &chroms {
                // bins covering ch.length at res
                let n = ((ch.length as u64).div_ceil(res)) as usize;
                if (n as u64) * res < min_chrom_size {
                    exclude_set.insert(ch.name.clone());
                }
                layout.push((ch.name.clone(), ch.length as u64, n));
                if !chrom_order.contains(&ch.name) {
                    chrom_order.push(ch.name.clone());
                }
            }
            layouts.insert((res, rep.clone()), layout);
        }

        // Pass 2: load each included chromosome as a `Chrom` (the balanced
        // upper-triangle entries, exactly as TADLib's
        // `triu(matrix(balance=..., sparse=True).fetch(chrom))`).
        let mut data: BTreeMap<String, BTreeMap<u64, BTreeMap<String, Chrom>>> = BTreeMap::new();
        for &(res, ref rep, ref path) in datasets {
            let cool = open_dataset(res, path)?;
            let weight = match balance {
                Some(name) => cool.bins_column_f64(name)?.ok_or_else(|| {
                    Error::InvalidInput(format!("no 'bins/{name}' column for balance"))
                })?,
                None => Vec::new(),
            };
            let offsets = cool.chrom_offset()?;
            let layout = &layouts[&(res, rep.clone())];
            for (ci, (name, _size, n)) in layout.iter().enumerate() {
                if exclude_set.contains(name) {
                    continue;
                }
                let first = offsets[ci] as i64;
                let last = first + *n as i64;
                let mut entries = Vec::new();
                for p in cool.pixels_for_bins(first, last)? {
                    let (b1, b2) = (p.bin1_id as usize, p.bin2_id as usize);
                    // drop pixels beyond the chromosome and the lower
                    // triangle, as the CLI does
                    if b2 >= last as usize || b1 > b2 {
                        continue;
                    }
                    let mut v = p.count;
                    if !weight.is_empty() {
                        v *= weight[b1] * weight[b2];
                    }
                    entries.push((b1 - first as usize, b2 - first as usize, v));
                }
                let mut chrom = Chrom::new(name, res, *n, &entries);
                chrom.set_maxapart(maxsize);
                data.entry(name.clone())
                    .or_default()
                    .entry(res)
                    .or_default()
                    .insert(rep.clone(), chrom);
            }
        }

        let layout = layouts
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.into_iter().map(|(name, size, _)| (name, size)).collect(),
                )
            })
            .collect();

        Ok(Genome {
            datasets: datasets.to_vec(),
            exclude: exclude_set,
            maxsize,
            chrom_order,
            data,
            layout,
            results: Vec::new(),
        })
    }

    /// TADLib hitad `Genome.learning`: per dataset (resolution x
    /// replicate), one Baum-Welch HMM trained on the concatenated non-zero
    /// DI segments of all its chromosomes, then attached to each of them.
    pub fn learning(&mut self) {
        for &(res, ref rep, _) in &self.datasets {
            let mut seqs = Vec::new();
            for chrom in &self.chrom_order {
                if self.exclude.contains(chrom) {
                    continue;
                }
                let c = self
                    .data
                    .get_mut(chrom)
                    .and_then(|m| m.get_mut(&res))
                    .and_then(|m| m.get_mut(rep))
                    .expect("learning: missing chrom dataset");
                c.compute_di();
                seqs.extend(c.train_data());
            }
            let hmm = Rc::new(Chrom::train_hmm(&seqs));
            for chrom in &self.chrom_order {
                if self.exclude.contains(chrom) {
                    continue;
                }
                self.data
                    .get_mut(chrom)
                    .and_then(|m| m.get_mut(&res))
                    .and_then(|m| m.get_mut(rep))
                    .expect("learning: missing chrom dataset")
                    .set_hmm(Rc::clone(&hmm));
            }
        }
    }

    /// TADLib hitad `Genome.callHierDomain`: hierarchical domain calling on
    /// every chromosome, replicate consensus per (chrom, resolution), and
    /// the minimum-resolution results merged in chromosome file order.
    /// Requires [`Genome::learning`] first. Sets `results`.
    pub fn call_domains(&mut self) {
        // One shared maxCore score cache across every (chrom, res, rep) in
        // processing order: TADLib's `def maxCore(self, cache={})` mutable
        // default argument, shared by all chromosomes of the CLI's single
        // worker process (see `Chrom::max_core`). Cross-chromosome key
        // collisions reuse the earlier chromosome's score — that is
        // TADLib's real behavior.
        let mut score_cache = BTreeMap::new();
        for chrom in self.chrom_order.clone() {
            if let Some(by_res) = self.data.get_mut(&chrom) {
                for by_rep in by_res.values_mut() {
                    for c in by_rep.values_mut() {
                        c.call_hier_domain_with_cache(&mut score_cache);
                    }
                }
            }
        }
        let mut results = Vec::new();
        for chrom in &self.chrom_order {
            let Some(by_res) = self.data.get(chrom) else {
                continue;
            };
            let mut min: Option<(u64, Vec<Domain>)> = None;
            for (&res, reps) in by_res {
                let merged = multi_reps(chrom, res, reps);
                if min.as_ref().is_none_or(|(r, _)| res < *r) {
                    min = Some((res, merged));
                }
            }
            if let Some((_, merged)) = min {
                results.extend(merged);
            }
        }
        self.results = results;
    }

    /// DI track rows (chrom, start, end, DI) for one dataset in cooler
    /// chromosome order, zeros for excluded chromosomes — mirrors what the
    /// hitad CLI writes into the `bins/DIs` column of the cool (that
    /// write-back is not ported; the tracks are exposed here instead).
    /// Requires [`Genome::call_domains`] first.
    pub fn di_track(&self, res: u64, rep: &str) -> Vec<(String, u64, u64, f64)> {
        let empty = Vec::new();
        let layout = self.layout.get(&(res, rep.to_string())).unwrap_or(&empty);
        let mut out = Vec::new();
        for (name, size) in layout {
            let chrom = self
                .data
                .get(name)
                .and_then(|m| m.get(&res))
                .and_then(|m| m.get(rep));
            if let Some(chrom) = chrom {
                for (i, &di) in chrom.dis.iter().enumerate() {
                    let start = i as u64 * res;
                    let end = ((i as u64 + 1) * res).min(*size);
                    out.push((name.clone(), start, end, di));
                }
            } else {
                // excluded chromosome: zeros, as the CLI writes
                let n = (size.div_ceil(res)) as usize;
                for i in 0..n {
                    let start = i as u64 * res;
                    let end = ((i as u64 + 1) * res).min(*size);
                    out.push((name.clone(), start, end, 0.0));
                }
            }
        }
        out
    }
}

/// TADLib hitad `MultiReps.callDomain`, single-replicate path only:
/// `getDomainList` (domains >= 3*res) -> `DomainSet` tree ->
/// `_correctDomainTree` (child bounds snap to parent bounds within 2*res)
/// -> `hierFormat` (levels rebuilt from containment).
// ponytail: single-rep identity path; port repAligner when a multi-replicate
// fixture exists to validate it against
fn multi_reps(chrom: &str, res: u64, reps: &BTreeMap<String, Chrom>) -> Vec<Domain> {
    if reps.len() > 1 {
        unimplemented!("multi-replicate hitad consensus (MultiReps/repAligner) is not ported");
    }
    let (label, rep) = reps.iter().next().unwrap();
    // getDomainList: [chrom, start, end, level], dropping domains < 3*res
    let mut domainlist: Vec<Domain> = Vec::new();
    for region in rep.hier_domains.values() {
        for d in region {
            if d[1] - d[0] >= 3.0 * res as f64 {
                domainlist.push((
                    chrom.to_string(),
                    d[0] as usize,
                    d[1] as usize,
                    d[3] as usize,
                ));
            }
        }
    }
    let set = DomainSet::new(label, &domainlist, res as usize);
    let mut pool: BTreeSet<Region> = BTreeSet::new();
    for (d, node) in &set.domains {
        correct_domain_tree(d, node, &mut pool, res as usize, None, None, None);
    }
    let regions: Vec<Region> = pool.into_iter().collect();
    hier_format(&regions)
}

/// TADLib hitad `MultiReps._correctDomainTree`: collect every tree domain,
/// snapping a child's start (end) to the parent's start (end) when within
/// 2*res. `cur` carries the ORIGINAL parent bounds (Python compares against
/// those) while `ref_s`/`ref_e` carry the parent's snapped bounds.
fn correct_domain_tree(
    d: &Domain,
    node: &Node,
    pool: &mut BTreeSet<Region>,
    res: usize,
    cur: Option<&Domain>,
    ref_s: Option<usize>,
    ref_e: Option<usize>,
) {
    let (chrom, mut start, mut end, _) = (d.0.clone(), d.1, d.2, d.3);
    if let (Some(cur), Some(rs), Some(re)) = (cur, ref_s, ref_e) {
        // start - cur.start <= 2*res, cur.end - end <= 2*res
        if start <= cur.1 + 2 * res {
            start = rs;
        }
        if cur.2 <= end + 2 * res {
            end = re;
        }
    }
    pool.insert((chrom, start, end));
    for (child, cnode) in &node.children {
        correct_domain_tree(child, cnode, pool, res, Some(d), Some(start), Some(end));
    }
}

/// Open the `.cool`/`.mcool` of one dataset row at `res` (a `.mcool` needs
/// the dataset's resolution to select the cooler).
fn open_dataset(res: u64, path: &str) -> Result<Cooler> {
    if path.ends_with(".mcool") {
        let mcool = Mcool::open(path)?;
        mcool.cooler(res)
    } else if path.ends_with(".cool") {
        Cooler::open_any(path)
    } else {
        Err(Error::InvalidInput(format!(
            "dataset path must be a .cool or .mcool file: {path}"
        )))
    }
}
