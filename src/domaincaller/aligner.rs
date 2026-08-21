//! Domain-based hierarchical alignment, ported from TADLib
//! `tadlib/hitad/aligner.py` (`BoundSet`, `DomainSet`, `SingleBound`,
//! `SingleDomain`, `DomainAligner`) plus `graph.py`'s connected components.
//!
//! `domaincaller` uses only the two-way `align` + `conserved`, which compute
//! the mismatch ratio that terminates `oriIter` early.
// The loops keep TADLib's Python index arithmetic one-to-one (including the
// in-loop `sidx` bound mutation and `&x` comparisons).
#![allow(clippy::needless_range_loop, clippy::mut_range_bound, clippy::op_ref)]

use std::collections::{BTreeMap, BTreeSet};

/// Nested-domain lookup for `hier_format`.
type Nested = BTreeMap<(usize, usize), Vec<(usize, usize)>>;
/// Aligned-region cache: region-pair -> container.
type Cache = BTreeMap<(Region, Region), Container>;

/// (chrom, start, end)
pub type Region = (String, usize, usize);
/// (chrom, start, end, level)
pub type Domain = (String, usize, usize, usize);
/// A bound (chrom, position).
pub type Bound = (String, usize);
/// (label, chrom, start, end, level) graph vertex.
pub type Vertex = (String, String, usize, usize, usize);

/// Dict-like node of the hierarchical domain tree.
#[derive(Debug, Default, Clone)]
pub struct Node {
    pub children: BTreeMap<Domain, Box<Node>>,
}

/// A container for one aligned region pair (dict-like in TADLib).
#[derive(Debug, Default, Clone)]
pub struct Container {
    pub info: (Vec<Domain>, Vec<Domain>),
    levels: BTreeMap<usize, BTreeMap<(Region, Region), Container>>,
}

impl Container {
    fn new(info: (Vec<Domain>, Vec<Domain>)) -> Self {
        Container {
            info,
            levels: BTreeMap::new(),
        }
    }
    fn levels(&self) -> impl Iterator<Item = &usize> {
        self.levels.keys()
    }
}

/// Bound set (TADLib `BoundSet`).
#[derive(Clone)]
pub struct BoundSet {
    pub label: String,
    pub bounds: Vec<Bound>,
}

impl BoundSet {
    fn new(label: &str, domainlist: &[Domain], res: usize) -> Self {
        let mut boundclass: BTreeMap<Bound, usize> = BTreeMap::new();
        for (chrom, start, end, level) in domainlist {
            if end - start < 5 * res {
                continue;
            }
            for pos in [start, end] {
                let b = (chrom.clone(), *pos);
                let e = boundclass.entry(b).or_insert(*level);
                if *level < *e {
                    *e = *level;
                }
            }
        }
        let bounds: Vec<Bound> = boundclass.keys().cloned().collect();
        BoundSet {
            label: label.to_string(),
            bounds,
        }
    }
}

/// Assign nesting levels to a list of (chrom, start, end) domains.
fn hier_format(domainlist: &[Region]) -> Vec<Domain> {
    let mut bychroms: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    let mut hierlabel: BTreeMap<&str, BTreeMap<(usize, usize), usize>> = BTreeMap::new();
    for (chrom, start, end) in domainlist {
        bychroms
            .entry(chrom.as_str())
            .or_default()
            .push((*start, *end));
        hierlabel
            .entry(chrom.as_str())
            .or_default()
            .insert((*start, *end), 0);
    }
    for v in bychroms.values_mut() {
        v.sort();
    }
    let mut nested: BTreeMap<&str, Nested> = BTreeMap::new();
    for (&c, domains) in &bychroms {
        let mut sidx = 0usize;
        let mut n = BTreeMap::new();
        for &q in domains {
            let mut pool = Vec::new();
            let mut label = true;
            for i in sidx..domains.len() {
                let d = domains[i];
                if d.0 >= q.0 && d.1 <= q.1 {
                    pool.push(d);
                }
                if d.0 >= q.0 && label {
                    sidx = i;
                    label = false;
                }
                if d.0 > q.1 {
                    break;
                }
            }
            for &p in &pool {
                if p != q {
                    hierlabel
                        .get_mut(&c)
                        .unwrap()
                        .entry(p)
                        .and_modify(|l| *l += 1);
                }
            }
            n.insert(q, pool);
        }
        nested.insert(c, n);
    }
    let mut out = Vec::new();
    for (&c, labels) in &hierlabel {
        for (&(start, end), &level) in labels {
            out.push((c.to_string(), start, end, level));
        }
    }
    out.sort();
    out
}

/// A domain set with hierarchy (TADLib `DomainSet`).
#[derive(Clone)]
pub struct DomainSet {
    pub label: String,
    pub res: usize,
    boundset: BoundSet,
    pub bychroms: BTreeMap<String, Vec<[usize; 3]>>,
    pub levs: BTreeSet<usize>,
    pretree: BTreeMap<Domain, Vec<Domain>>,
    subpool: BTreeMap<Domain, Vec<Domain>>,
    lidx: BTreeMap<Bound, usize>,
    ridx: BTreeMap<Bound, usize>,
    /// Domain tree: top-level domain -> nested node tree.
    pub domains: BTreeMap<Domain, Box<Node>>,
    pub bottoms: BTreeMap<Region, Domain>,
}

impl DomainSet {
    pub fn new(label: &str, domainlist: &[Domain], res: usize) -> Self {
        let boundset = BoundSet::new(label, domainlist, res);
        let mut bychroms: BTreeMap<String, Vec<[usize; 3]>> = BTreeMap::new();
        let mut levs = BTreeSet::new();
        for (chrom, start, end, level) in domainlist {
            if end - start < 5 * res {
                continue;
            }
            bychroms
                .entry(chrom.clone())
                .or_default()
                .push([*start, *end, *level]);
            levs.insert(*level);
        }
        let (pretree, subpool, lidx, ridx) = nested_domains(&bychroms);
        let mut domains: BTreeMap<Domain, Box<Node>> = BTreeMap::new();
        for (d, hit) in &pretree {
            if d.3 > 0 {
                continue;
            }
            let mut node = Box::new(Node::default());
            gen_domain_tree(&mut node, &pretree, hit);
            domains.insert(d.clone(), node);
        }
        DomainSet {
            label: label.to_string(),
            res,
            boundset,
            bychroms,
            levs,
            pretree,
            subpool,
            lidx,
            ridx,
            domains,
            bottoms: BTreeMap::new(),
        }
    }

    /// TADLib `SingleDomain.align`: best overlap match in another set.
    fn single_domain_align(
        &self,
        chrom: &str,
        interval: (usize, usize),
        qy: &DomainSet,
    ) -> Option<(Region, f64)> {
        let qb = &qy.boundset.bounds;
        if qb.is_empty() {
            return None;
        }
        let res = qy.res;
        let left = single_bound_align(qb, chrom, interval.0, 5 * res)?;
        let right = single_bound_align(qb, chrom, interval.1, 5 * res)?;
        let lidx = left[0];
        let ridx = *right.last().unwrap();
        let mut candis: Vec<(f64, Vec<usize>)> = Vec::new();
        for i in lidx..ridx {
            for j in i + 1..ridx + 1 {
                let tmp: Vec<usize> = (i..j + 1).map(|t| qb[t].1).collect();
                let score = overlap(interval, (tmp[0], *tmp.last().unwrap()));
                candis.push((score, tmp));
            }
        }
        if candis.is_empty() {
            return None;
        }
        // max(candis): first maximal (score, tmp) tuple.
        let mut best = &candis[0];
        for c in candis.iter().skip(1) {
            if py_max(&c.0, &c.1, &best.0, &best.1) {
                best = c;
            }
        }
        if best.0 == 0.0 {
            return None;
        }
        Some((
            (chrom.to_string(), best.1[0], *best.1.last().unwrap()),
            best.0,
        ))
    }

    /// Domain list within a region (TADLib `getregion`).
    fn get_region(&self, chrom: &str, start: usize, end: usize, lev: Option<usize>) -> Vec<Domain> {
        let sidx = self.lidx[&(chrom.to_string(), start)];
        let eidx = self.ridx[&(chrom.to_string(), end)] + 1;
        let candis = &self.bychroms[chrom];
        let mut cache: BTreeMap<Domain, BTreeMap<usize, Domain>> = BTreeMap::new();
        for d in candis.iter().take(eidx).skip(sidx) {
            if d[0] < start {
                continue;
            }
            if d[1] > end {
                continue;
            }
            let tmp: Domain = (chrom.to_string(), d[0], d[1], d[2]);
            if let Some(subs) = self.subpool.get(&tmp) {
                for sub in subs {
                    if self.pretree[sub].is_empty() {
                        let lv = tmp.3;
                        cache
                            .entry(sub.clone())
                            .or_default()
                            .insert(lv, tmp.clone());
                    }
                }
            }
        }
        let mut pool: BTreeSet<Domain> = BTreeSet::new();
        let mut rdomains = Vec::new();
        for bylevel in cache.values() {
            let maxl = *bylevel.keys().max().unwrap();
            for (&l, d) in bylevel {
                if let Some(lv) = lev {
                    if l != lv && (maxl >= lv || l < maxl) {
                        continue;
                    }
                }
                if pool.insert(d.clone()) {
                    rdomains.push(d.clone());
                }
            }
        }
        rdomains.sort();
        rdomains
    }

    /// Link bottom domains to outer domains (TADLib `getBottoms`).
    fn get_bottoms(&mut self) {
        self.bottoms = BTreeMap::new();
        for (d, subs) in &self.subpool {
            if d.3 == 0 {
                for sub in subs {
                    if self.pretree[sub].is_empty() {
                        self.bottoms
                            .insert((sub.0.clone(), sub.1, sub.2), (d.0.clone(), d.1, d.2, d.3));
                    }
                }
            }
        }
    }
}

/// TADLib `SingleBound.align` on an explicit bound list.
fn single_bound_align(qb: &[Bound], chrom: &str, pos: usize, tol: usize) -> Option<Vec<usize>> {
    if qb.is_empty() {
        return None;
    }
    let tidx = qb.binary_search_by(|probe| {
        if probe.0.as_str() < chrom {
            std::cmp::Ordering::Less
        } else if probe.0.as_str() > chrom {
            std::cmp::Ordering::Greater
        } else {
            probe.1.cmp(&pos)
        }
    });
    let tidx = match tidx {
        Ok(i) => i + 1,
        Err(i) => i,
    };
    // Python's `qb[tidx - 1]` wraps to the last element when tidx == 0.
    let prev_idx = if tidx > 0 { tidx - 1 } else { qb.len() - 1 };
    let mut lidx: isize = -1;
    let mut ridx: isize = -1;
    if (chrom.to_string(), pos) == qb[prev_idx] {
        lidx = tidx as isize - 1;
        ridx = lidx;
    } else if tidx < qb.len() {
        if chrom != &qb[prev_idx].0 {
            if chrom == &qb[tidx].0 {
                lidx = tidx as isize;
                ridx = lidx;
            }
        } else if chrom == &qb[tidx].0 {
            lidx = tidx as isize - 1;
            ridx = tidx as isize;
        } else {
            lidx = tidx as isize - 1;
            ridx = lidx;
        }
    } else if chrom == &qb[prev_idx].0 {
        lidx = tidx as isize - 1;
        ridx = lidx;
    }
    if tidx == 0 && chrom == &qb[tidx].0 {
        lidx = tidx as isize;
        ridx = lidx;
    }
    if lidx == -1 {
        return None;
    }
    let mut nindices: BTreeSet<usize> = BTreeSet::new();
    nindices.insert(lidx as usize);
    nindices.insert(ridx as usize);
    let mut t = lidx - 1;
    while t >= 0 {
        if qb[t as usize].0 != chrom {
            break;
        }
        nindices.insert(t as usize);
        if pos.abs_diff(qb[t as usize].1) >= tol {
            break;
        }
        t -= 1;
    }
    let mut t = ridx + 1;
    while (t as usize) < qb.len() {
        if qb[t as usize].0 != chrom {
            break;
        }
        nindices.insert(t as usize);
        if pos.abs_diff(qb[t as usize].1) >= tol {
            break;
        }
        t += 1;
    }
    Some(nindices.into_iter().collect())
}

/// Overlap ratio of two intervals (TADLib `SingleDomain.overlap`).
fn overlap(ta: (usize, usize), qa: (usize, usize)) -> f64 {
    if ta.1 <= qa.0 || qa.1 <= ta.0 {
        return 0.0;
    }
    let mut mi = [ta.0, ta.1, qa.0, qa.1];
    mi.sort_unstable();
    (mi[2] - mi[1]) as f64 / (mi[3] - mi[0]) as f64
}

/// Python `max` on (f64, Vec<usize>): `c > best` lexicographically.
fn py_max(c_score: &f64, c_tmp: &[usize], b_score: &f64, b_tmp: &[usize]) -> bool {
    if c_score != b_score {
        return c_score > b_score;
    }
    c_tmp > b_tmp
}

/// TADLib `NestedDomains`.
type NestedDomains = (
    BTreeMap<Domain, Vec<Domain>>,
    BTreeMap<Domain, Vec<Domain>>,
    BTreeMap<Bound, usize>,
    BTreeMap<Bound, usize>,
);
fn nested_domains(bychroms: &BTreeMap<String, Vec<[usize; 3]>>) -> NestedDomains {
    let mut tmpdict: BTreeMap<Domain, Vec<Domain>> = BTreeMap::new();
    let mut subpool: BTreeMap<Domain, Vec<Domain>> = BTreeMap::new();
    let mut lidx: BTreeMap<Bound, usize> = BTreeMap::new();
    let mut ridx: BTreeMap<Bound, usize> = BTreeMap::new();
    for (c, domains) in bychroms {
        let mut sidx = 0usize;
        let mut ds = domains.clone();
        ds.sort();
        for q in &ds {
            let mut pres = Vec::new();
            let mut pool = Vec::new();
            let mut label = true;
            let key: Domain = (c.clone(), q[0], q[1], q[2]);
            let lk: Bound = (c.clone(), q[0]);
            let rk: Bound = (c.clone(), q[1]);
            lidx.entry(lk.clone())
                .and_modify(|v| *v = (*v).min(sidx))
                .or_insert(sidx.min(ds.len().saturating_sub(1)));
            lidx.entry(rk.clone())
                .and_modify(|v| *v = (*v).min(sidx))
                .or_insert(sidx.min(ds.len().saturating_sub(1)));
            let mut last_i = 0usize;
            for i in sidx..ds.len() {
                last_i = i;
                let d = ds[i];
                if d[0] >= q[0] && d[1] <= q[1] {
                    if d[2] == q[2] + 1 {
                        pres.push((c.clone(), d[0], d[1], d[2]));
                    }
                    pool.push((c.clone(), d[0], d[1], d[2]));
                }
                if d[0] >= q[0] && label {
                    sidx = i;
                    label = false;
                }
                if d[0] > q[1] {
                    break;
                }
            }
            ridx.entry(lk)
                .and_modify(|v| *v = (*v).max(last_i))
                .or_insert(last_i);
            ridx.entry(rk)
                .and_modify(|v| *v = (*v).max(last_i))
                .or_insert(last_i);
            tmpdict.insert(key.clone(), pres);
            subpool.insert(key, pool);
        }
    }
    (tmpdict, subpool, lidx, ridx)
}

fn gen_domain_tree(node: &mut Box<Node>, pretree: &BTreeMap<Domain, Vec<Domain>>, cur: &[Domain]) {
    for d in cur {
        let mut child = Box::new(Node::default());
        if let Some(hit) = pretree.get(d) {
            gen_domain_tree(&mut child, pretree, hit);
        }
        node.children.insert(d.clone(), child);
    }
}

/// TADLib `_getTree`: collect domains contained in [start, end].
fn get_tree(
    start: usize,
    end: usize,
    map: &BTreeMap<Domain, Box<Node>>,
    pool: &mut BTreeMap<Region, usize>,
) {
    for (d, child) in map {
        if d.1 >= end {
            continue;
        }
        if d.2 <= start {
            continue;
        }
        if d.1 >= start && d.2 <= end {
            pool.insert((d.0.clone(), d.1, d.2), d.3);
            continue;
        }
        get_tree(start, end, &child.children, pool);
    }
}

/// A bipartite alignment graph (TADLib `Graph`).
struct Graph {
    adj: BTreeMap<Vertex, BTreeMap<Vertex, f64>>,
}

impl Graph {
    fn new(vs: &BTreeSet<Vertex>, es: &BTreeMap<(Vertex, Vertex), f64>) -> Self {
        let mut adj: BTreeMap<Vertex, BTreeMap<Vertex, f64>> = BTreeMap::new();
        for v in vs {
            adj.entry(v.clone()).or_default();
        }
        for ((v, w), s) in es {
            adj.entry(v.clone()).or_default().insert(w.clone(), *s);
            adj.entry(w.clone()).or_default().insert(v.clone(), *s);
        }
        Graph { adj }
    }

    fn connected_components(&self) -> Vec<BTreeSet<Vertex>> {
        let mut pool: BTreeSet<Vertex> = BTreeSet::new();
        let mut components = Vec::new();
        for v in self.adj.keys() {
            if !pool.contains(v) {
                let mut cache = BTreeSet::new();
                self.dfs(v, &mut cache);
                pool.extend(cache.iter().cloned());
                components.push(cache);
            }
        }
        components
    }

    fn dfs(&self, v: &Vertex, cache: &mut BTreeSet<Vertex>) {
        cache.insert(v.clone());
        if let Some(neighbors) = self.adj.get(v) {
            for w in neighbors.keys() {
                if !cache.contains(w) {
                    self.dfs(w, cache);
                }
            }
        }
    }
}

/// The domain aligner (TADLib `DomainAligner`).
pub struct DomainAligner {
    domain_sets: BTreeMap<String, DomainSet>,
    results: BTreeMap<String, BTreeMap<String, Cache>>,
}

impl DomainAligner {
    pub fn new(sets: Vec<DomainSet>) -> Self {
        let mut domain_sets = BTreeMap::new();
        for s in sets {
            domain_sets.insert(s.label.clone(), s);
        }
        DomainAligner {
            domain_sets,
            results: BTreeMap::new(),
        }
    }

    /// TADLib `_oneway`.
    fn one_way(
        &self,
        tg: &DomainSet,
        qy: &DomainSet,
        t_ref: Option<&DomainSet>,
        q_ref: Option<&DomainSet>,
        vs: &mut BTreeSet<Vertex>,
        es: &mut BTreeMap<(Vertex, Vertex), f64>,
    ) {
        for td in tg.domains.keys() {
            let tk: Vertex = if let Some(t) = t_ref {
                let b = &t.bottoms[&(td.0.clone(), td.1, td.2)];
                (tg.label.clone(), b.0.clone(), b.1, b.2, b.3)
            } else {
                (tg.label.clone(), td.0.clone(), td.1, td.2, td.3)
            };
            vs.insert(tk.clone());
            let Some((pse, _)) = tg.single_domain_align(&td.0, (td.1, td.2), qy) else {
                continue;
            };
            let qds = qy.get_region(&pse.0, pse.1, pse.2, None);
            if qds.is_empty() {
                continue;
            }
            let qks: Vec<Vertex> = if let Some(q) = q_ref {
                qds.iter()
                    .map(|d| {
                        let b = &q.bottoms[&(d.0.clone(), d.1, d.2)];
                        (qy.label.clone(), b.0.clone(), b.1, b.2, b.3)
                    })
                    .collect()
            } else {
                qds.iter()
                    .map(|d| (qy.label.clone(), d.0.clone(), d.1, d.2, d.3))
                    .collect()
            };
            for k in qks {
                vs.insert(k.clone());
                let op = overlap((td.1, td.2), (k.2, k.3));
                es.insert((tk.clone(), k), op);
            }
        }
    }

    /// TADLib `_aligncore`.
    fn align_core(
        &self,
        tg: &DomainSet,
        qy: &DomainSet,
        t_ref: Option<&DomainSet>,
        q_ref: Option<&DomainSet>,
    ) -> BTreeMap<(Region, Region), (Vec<Domain>, Vec<Domain>)> {
        let mut vs: BTreeSet<Vertex> = BTreeSet::new();
        let mut pes: BTreeMap<(Vertex, Vertex), f64> = BTreeMap::new();
        if !qy.domains.is_empty() {
            self.one_way(tg, qy, t_ref, q_ref, &mut vs, &mut pes);
            self.one_way(qy, tg, q_ref, t_ref, &mut vs, &mut pes);
        } else {
            for d in tg.domains.keys() {
                let v: Vertex = if let Some(t) = t_ref {
                    let b = &t.bottoms[&(d.0.clone(), d.1, d.2)];
                    (tg.label.clone(), b.0.clone(), b.1, b.2, b.3)
                } else {
                    (tg.label.clone(), d.0.clone(), d.1, d.2, d.3)
                };
                vs.insert(v);
            }
        }
        let es: BTreeMap<(Vertex, Vertex), f64> = pes
            .iter()
            .filter(|((a, b), _)| pes.contains_key(&(b.clone(), a.clone())))
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        let g = Graph::new(&vs, &es);
        let mut pairs = BTreeMap::new();
        for c in g.connected_components() {
            let mut med: [Vec<Domain>; 2] = [Vec::new(), Vec::new()];
            for d in &c {
                let idx = if d.0 == tg.label { 0 } else { 1 };
                med[idx].push((d.1.clone(), d.2, d.3, d.4));
            }
            med[0].sort();
            med[1].sort();
            if med[0].is_empty() || med[1].is_empty() {
                continue;
            }
            let (tk, qk, merged) = to_be_robust(&med[0], &med[1], tg, qy, t_ref, q_ref);
            pairs.insert((tk, qk), merged);
        }
        pairs
    }

    /// TADLib `_localhits`.
    fn local_hits(&self, tg: &DomainSet, qy: &DomainSet) -> Vec<Domain> {
        let mut table: BTreeMap<Region, usize> = BTreeMap::new();
        for td in tg.domains.keys() {
            if let Some((pse, _)) = tg.single_domain_align(&td.0, (td.1, td.2), qy) {
                get_tree(pse.1, pse.2, &qy.domains, &mut table);
            }
        }
        if table.is_empty() {
            return Vec::new();
        }
        let rmin = table.keys().map(|d| d.1).min().unwrap();
        let rmax = table.keys().map(|d| d.2).max().unwrap();
        let reorg: Vec<Region> = table.keys().cloned().collect();
        let nqy = DomainSet::new(&qy.label, &hier_format(&reorg), qy.res);
        let mut pool: BTreeMap<Region, usize> = BTreeMap::new();
        get_tree(rmin, rmax, &nqy.domains, &mut pool);
        let mut ql = Vec::new();
        for (d, &lv) in &pool {
            ql.push((d.0.clone(), d.1, d.2, lv));
        }
        ql
    }

    /// TADLib `_align`.
    fn align_one_way(&self, tg: &DomainSet, qy: &DomainSet) -> Cache {
        let mut tg = tg.clone();
        let mut qy = qy.clone();
        tg.get_bottoms();
        qy.get_bottoms();
        let ttl = hier_format(&tg.bottoms.keys().cloned().collect::<Vec<_>>());
        let tql = hier_format(&qy.bottoms.keys().cloned().collect::<Vec<_>>());
        let ttg = DomainSet::new(&tg.label, &ttl, tg.res);
        let tqy = DomainSet::new(&qy.label, &tql, qy.res);
        let pairs = self.align_core(&ttg, &tqy, Some(&tg), Some(&qy));
        let mut cache: Cache = BTreeMap::new();
        for (k, merged) in &pairs {
            let tl = tg.get_region(&k.0 .0, k.0 .1, k.0 .2, None);
            let ftg = DomainSet::new(&tg.label, &tl, tg.res);
            let ql = qy.get_region(&k.1 .0, k.1 .1, k.1 .2, None);
            let fqy = DomainSet::new(&qy.label, &ql, qy.res);
            let mut cont = Container::new(merged.clone());
            let ori = if merged.0.len() == 1 { 1 } else { 0 };
            let max_lev = *ftg.levs.iter().max().unwrap_or(&0);
            for tv in ori..=max_lev {
                let mut level = BTreeMap::new();
                let tl = ftg.get_region(&k.0 .0, k.0 .1, k.0 .2, Some(tv));
                let ntg = DomainSet::new(&ftg.label, &tl, ftg.res);
                let ql = self.local_hits(&ntg, &fqy);
                let nqy = DomainSet::new(&fqy.label, &ql, fqy.res);
                let npairs = self.align_core(&ntg, &nqy, None, None);
                for (p, merged) in npairs {
                    level.insert(p, Container::new(merged));
                }
                cont.levels.insert(tv, level);
            }
            cache.insert(k.clone(), cont);
        }
        cache
    }

    /// TADLib `align`: two-way alignment, storing cross-corrected results.
    pub fn align(&mut self, tn: &str, qn: &str) {
        let tg = self.domain_sets[tn].clone();
        let qy = self.domain_sets[qn].clone();
        let tcache = self.align_one_way(&tg, &qy);
        let qcache = self.align_one_way(&qy, &tg);
        let tcore = cross_correct(&tcache, &qcache);
        let qcore = cross_correct(&qcache, &tcache);
        self.results
            .insert(tn.to_string(), BTreeMap::from([(qn.to_string(), tcore)]));
        self.results
            .insert(qn.to_string(), BTreeMap::from([(tn.to_string(), qcore)]));
    }

    /// TADLib `conserved`: conserved TAD pairs.
    pub fn conserved(&self, tn: &str, qn: &str) -> BTreeSet<(Region, Region)> {
        let pool = &self.results[tn][qn];
        let mut pairs: BTreeSet<(Region, Region)> = BTreeSet::new();
        for (k, c) in pool {
            if c.info.0.len() == 1 && c.info.1.len() == 1 {
                pairs.insert(k.clone());
            }
        }
        let changed = self.inner_changed(tn, qn);
        pairs.retain(|k| !changed.contains(k));
        pairs
    }

    fn inner_changed(&self, tn: &str, qn: &str) -> BTreeSet<(Region, Region)> {
        let mut pairs = self.lowlevel_changed(tn, qn);
        let r_pairs = self.lowlevel_changed(qn, tn);
        for k in r_pairs {
            pairs.insert((k.1, k.0));
        }
        pairs
    }

    fn lowlevel_changed(&self, tn: &str, qn: &str) -> BTreeSet<(Region, Region)> {
        let pool = &self.results[tn][qn];
        let dset = &self.domain_sets[tn];
        let mut pairs = BTreeSet::new();
        for (k, c) in pool {
            if c.info.0.len() > 1 || c.info.1.len() > 1 {
                continue;
            }
            let tl0 = &c.info.0[0];
            let alls = dset.get_region(&tl0.0, tl0.1, tl0.2, None);
            let mut labels: BTreeMap<Domain, usize> = BTreeMap::new();
            for d in &alls {
                if d.3 > 0 {
                    labels.insert(d.clone(), 0);
                }
            }
            if labels.is_empty() {
                continue;
            }
            for lv in c.levels() {
                for sub in c.levels[lv].values() {
                    if sub.info.0.len() == 1
                        && sub.info.1.len() == 1
                        && sub.info.0[0].3 == sub.info.1[0].3
                    {
                        labels.insert(sub.info.0[0].clone(), 1);
                    }
                }
            }
            if labels.values().any(|&v| v == 0) {
                pairs.insert(k.clone());
            }
        }
        pairs
    }
}

/// TADLib `_crosscorrect`.
fn cross_correct(
    cache: &BTreeMap<(Region, Region), Container>,
    ref_cache: &BTreeMap<(Region, Region), Container>,
) -> BTreeMap<(Region, Region), Container> {
    let mut tcore: BTreeMap<(Region, Region), Container> = BTreeMap::new();
    for (k, target) in cache {
        let sk = (k.1.clone(), k.0.clone());
        let Some(hit) = ref_cache.get(&sk) else {
            continue;
        };
        let mut cont = Container::new(target.info.clone());
        for lv in target.levels() {
            for (p, sub) in &target.levels[lv] {
                if search(hit, p) {
                    let mlv = sub.info.0.iter().map(|d| d.3).min().unwrap();
                    cont.levels
                        .entry(mlv)
                        .or_default()
                        .insert(p.clone(), sub.clone());
                }
            }
        }
        tcore.insert(k.clone(), cont);
    }
    tcore
}

/// TADLib `_search`.
fn search(hit: &Container, k: &(Region, Region)) -> bool {
    for lv in hit.levels() {
        for h in hit.levels[lv].keys() {
            if (h.1.clone(), h.0.clone()) == *k {
                return true;
            }
        }
    }
    false
}

/// TADLib `_toberobust`.
fn to_be_robust(
    tl: &[Domain],
    ql: &[Domain],
    tg: &DomainSet,
    qy: &DomainSet,
    t_ref: Option<&DomainSet>,
    q_ref: Option<&DomainSet>,
) -> (Region, Region, (Vec<Domain>, Vec<Domain>)) {
    let tl_merged = if let Some(t) = t_ref {
        t.get_region(&tl[0].0, tl[0].1, tl[tl.len() - 1].2, Some(0))
    } else {
        tg.get_region(&tl[0].0, tl[0].1, tl[tl.len() - 1].2, None)
    };
    let ql_merged = if let Some(q) = q_ref {
        q.get_region(&ql[0].0, ql[0].1, ql[ql.len() - 1].2, Some(0))
    } else {
        qy.get_region(&ql[0].0, ql[0].1, ql[ql.len() - 1].2, None)
    };
    let mut tl = tl_merged;
    let mut ql = ql_merged;
    tl.sort();
    ql.sort();
    let tk = (tl[0].0.clone(), tl[0].1, tl[tl.len() - 1].2);
    let qk = (ql[0].0.clone(), ql[0].1, ql[ql.len() - 1].2);
    (tk, qk, (tl, ql))
}
