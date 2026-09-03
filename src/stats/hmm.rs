//! Hidden Markov Model with arbitrary emission distributions, trained by
//! multi-sequence Baum-Welch.
//!
//! Port of pomegranate 0.10.0 `hmm.pyx`, restricted to what `domaincaller`
//! needs: dense models, univariate Gaussian/GMM emissions, and silent
//! `start`/`end` states. The DP tables, the CSR transition layout and the
//! numerics (log space, `pair_lse`) mirror the Cython (see `tests/stats.rs`).

use std::collections::HashMap;

use ndarray::Array2;

use crate::stats::normal::{log_pos, pair_lse, Emission, NEG_INF};

/// Builder node id of the implicit silent start state.
pub const START: usize = 0;
/// Builder node id of the implicit silent end state.
pub const END: usize = 1;

/// A state in the model: an emission distribution plus a name.
///
/// `distribution` indexes the HMM's emission registry; `None` marks a
/// silent state. Mirrors pomegranate's `State`.
#[derive(Debug, Clone)]
pub struct State {
    pub name: String,
    pub distribution: Option<usize>,
    pub weight: f64,
}

#[derive(Debug, Clone)]
struct Edge {
    from: usize,
    to: usize,
    log_prob: f64,
    pseudocount: f64,
}

/// A Hidden Markov Model over `Emission` distributions.
///
/// Build the topology with `add_state`/`add_transition`, then `bake()` to
/// finalize before any inference or training call. `bake` normalizes each
/// node's out-transitions to sum to 1, sorts emitting states by name and
/// orders silent states topologically. Corresponds to pomegranate's
/// `bake(merge="All")` without orphan removal or silent-state merging.
pub struct HiddenMarkovModel {
    // build phase
    nodes: Vec<State>,
    edges: Vec<Edge>,
    emissions: Vec<Box<dyn Emission>>,
    // baked phase
    states: Vec<State>,
    /// Baked emitting state index -> emission registry index.
    distributions: Vec<usize>,
    /// Log emission weights (ln of each emitting state's `weight`).
    state_weights: Vec<f64>,
    start_index: usize,
    end_index: usize,
    silent_start: usize,
    finite: bool,
    in_edge_count: Vec<usize>,
    in_transitions: Vec<usize>,
    in_log_probs: Vec<f64>,
    in_pseudocounts: Vec<f64>,
    out_edge_count: Vec<usize>,
    out_transitions: Vec<usize>,
    out_log_probs: Vec<f64>,
    out_pseudocounts: Vec<f64>,
    expected_transitions: Vec<f64>,
    n_edges: usize,
    n_summarized: usize,
}

impl Default for HiddenMarkovModel {
    fn default() -> Self {
        Self::new("model")
    }
}

impl HiddenMarkovModel {
    pub fn new(name: &str) -> Self {
        let start = State {
            name: format!("{name}-start"),
            distribution: None,
            weight: 1.0,
        };
        let end = State {
            name: format!("{name}-end"),
            distribution: None,
            weight: 1.0,
        };
        HiddenMarkovModel {
            nodes: vec![start, end],
            edges: Vec::new(),
            emissions: Vec::new(),
            states: Vec::new(),
            distributions: Vec::new(),
            state_weights: Vec::new(),
            start_index: 0,
            end_index: 0,
            silent_start: 0,
            finite: true,
            in_edge_count: Vec::new(),
            in_transitions: Vec::new(),
            in_log_probs: Vec::new(),
            in_pseudocounts: Vec::new(),
            out_edge_count: Vec::new(),
            out_transitions: Vec::new(),
            out_log_probs: Vec::new(),
            out_pseudocounts: Vec::new(),
            expected_transitions: Vec::new(),
            n_edges: 0,
            n_summarized: 0,
        }
    }

    /// Register an emission distribution; returns its handle for `add_state`.
    pub fn add_emission(&mut self, emission: Box<dyn Emission>) -> usize {
        self.emissions.push(emission);
        self.emissions.len() - 1
    }

    /// Add an emitting state; returns its builder node id.
    pub fn add_state(&mut self, name: impl Into<String>, emission: usize, weight: f64) -> usize {
        self.nodes.push(State {
            name: name.into(),
            distribution: Some(emission),
            weight,
        });
        self.nodes.len() - 1
    }

    /// Add a silent (non-emitting) state; returns its builder node id.
    pub fn add_silent_state(&mut self, name: impl Into<String>) -> usize {
        self.nodes.push(State {
            name: name.into(),
            distribution: None,
            weight: 1.0,
        });
        self.nodes.len() - 1
    }

    /// Add a transition with a (non-log) probability. Probabilities will be
    /// normalized per source node by `bake`.
    pub fn add_transition(&mut self, from: usize, to: usize, probability: f64) {
        self.edges.push(Edge {
            from,
            to,
            log_prob: log_pos(probability),
            pseudocount: probability,
        });
    }

    /// Finalize the model topology (pomegranate `bake`, `merge="All"`).
    pub fn bake(&mut self) {
        let start_id = START;
        let end_id = END;
        let n_nodes = self.nodes.len();

        // Normalize each node's out-transitions to sum to 1.
        let mut out_edges: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, e) in self.edges.iter().enumerate() {
            out_edges.entry(e.from).or_default().push(i);
        }
        for (&node, edges) in &out_edges {
            if node == end_id {
                continue;
            }
            let lin: f64 = edges.iter().map(|&i| self.edges[i].log_prob.exp()).sum();
            if (lin - 1.0).abs() > 1e-8 {
                let lse = lin.ln();
                for &i in edges {
                    self.edges[i].log_prob -= lse;
                }
            }
        }

        // Split into emitting and silent states; emitting sorted by name,
        // silent sorted by name then topologically ordered.
        let mut normal: Vec<usize> = Vec::new();
        let mut silent: Vec<usize> = Vec::new();
        for id in 0..n_nodes {
            if self.nodes[id].distribution.is_some() {
                normal.push(id);
            } else {
                silent.push(id);
            }
        }
        normal.sort_by_key(|&id| self.nodes[id].name.clone());
        silent.sort_by_key(|&id| self.nodes[id].name.clone());
        let silent_sorted = self.topo_sort_silent(&silent);

        self.silent_start = normal.len();
        self.states = normal
            .iter()
            .chain(silent_sorted.iter())
            .map(|&id| self.nodes[id].clone())
            .collect();
        let n = self.states.len();

        let mut indices = vec![0usize; n_nodes];
        for (bi, &id) in normal.iter().chain(silent_sorted.iter()).enumerate() {
            indices[id] = bi;
        }
        self.start_index = indices[start_id];
        self.end_index = indices[end_id];

        // CSR transition layout: `in_edge_count[k]..in_edge_count[k+1]` is the
        // slice of `in_transitions` holding the source state of each edge into
        // state k (edge order = insertion order within the node's slot).
        let m_edges = self.edges.len();
        let mut in_edge_count = vec![0usize; n + 1];
        let mut out_edge_count = vec![0usize; n + 1];
        for e in &self.edges {
            in_edge_count[indices[e.to] + 1] += 1;
            out_edge_count[indices[e.from] + 1] += 1;
        }
        self.finite = in_edge_count[indices[end_id] + 1] != 0;
        for i in 1..=n {
            in_edge_count[i] += in_edge_count[i - 1];
            out_edge_count[i] += out_edge_count[i - 1];
        }
        let mut in_transitions = vec![usize::MAX; m_edges];
        let mut in_log_probs = vec![0.0; m_edges];
        let mut in_pseudocounts = vec![0.0; m_edges];
        let mut out_transitions = vec![usize::MAX; m_edges];
        let mut out_log_probs = vec![0.0; m_edges];
        let mut out_pseudocounts = vec![0.0; m_edges];
        for e in &self.edges {
            let bi = indices[e.to];
            let mut s = in_edge_count[bi];
            while in_transitions[s] != usize::MAX {
                s += 1;
            }
            in_transitions[s] = indices[e.from];
            in_log_probs[s] = e.log_prob;
            in_pseudocounts[s] = e.pseudocount;

            let ai = indices[e.from];
            let mut s = out_edge_count[ai];
            while out_transitions[s] != usize::MAX {
                s += 1;
            }
            out_transitions[s] = indices[e.to];
            out_log_probs[s] = e.log_prob;
            out_pseudocounts[s] = e.pseudocount;
        }

        self.in_edge_count = in_edge_count;
        self.in_transitions = in_transitions;
        self.in_log_probs = in_log_probs;
        self.in_pseudocounts = in_pseudocounts;
        self.out_edge_count = out_edge_count;
        self.out_transitions = out_transitions;
        self.out_log_probs = out_log_probs;
        self.out_pseudocounts = out_pseudocounts;
        self.n_edges = m_edges;
        self.expected_transitions = vec![0.0; m_edges];

        self.distributions = (0..self.silent_start)
            .map(|k| self.states[k].distribution.unwrap())
            .collect();
        self.state_weights = (0..self.silent_start)
            .map(|k| log_pos(self.states[k].weight))
            .collect();
    }

    /// Topological sort of the silent-state subgraph, replicating networkx
    /// 1.11 `topological_sort(G, nbunch=sorted_by_name)`.
    fn topo_sort_silent(&self, silent: &[usize]) -> Vec<usize> {
        let silent_set: std::collections::HashSet<usize> = silent.iter().copied().collect();
        let mut succ: HashMap<usize, Vec<usize>> = HashMap::new();
        for e in &self.edges {
            if silent_set.contains(&e.from) && silent_set.contains(&e.to) {
                succ.entry(e.from).or_default().push(e.to);
            }
        }
        let mut seen = std::collections::HashSet::new();
        let mut explored = std::collections::HashSet::new();
        let mut order = Vec::new();
        for &v in silent {
            if explored.contains(&v) {
                continue;
            }
            let mut fringe = vec![v];
            while let Some(&w) = fringe.last() {
                if explored.contains(&w) {
                    fringe.pop();
                    continue;
                }
                seen.insert(w);
                let new_nodes: Vec<usize> = succ
                    .get(&w)
                    .into_iter()
                    .flat_map(|s| s.iter())
                    .copied()
                    .filter(|n| !explored.contains(n) && !seen.contains(n))
                    .collect();
                if !new_nodes.is_empty() {
                    fringe.extend(new_nodes);
                } else {
                    explored.insert(w);
                    order.push(w);
                    fringe.pop();
                }
            }
        }
        order.reverse();
        order
    }

    /// Number of baked states (including silent `start`/`end`).
    pub fn n_states(&self) -> usize {
        self.states.len()
    }

    /// Baked state name by index (used to read `viterbi` paths).
    pub fn state_name(&self, idx: usize) -> &str {
        &self.states[idx].name
    }

    /// Emission log-probabilities `e[l*n + i]` for each emitting state `l`.
    fn emissions(&self, xs: &[f64]) -> Vec<f64> {
        let n = xs.len();
        let p = self.silent_start;
        let mut e = vec![0.0; n * p];
        for l in 0..p {
            let ei = self.distributions[l];
            for i in 0..n {
                e[l * n + i] = self.emissions[ei].log_probability(xs[i]) + self.state_weights[l];
            }
        }
        e
    }

    /// Forward algorithm. Returns the `(n+1) × m` log-probability table.
    pub fn forward(&self, xs: &[f64]) -> Array2<f64> {
        let e = self.emissions(xs);
        let f = self.forward_em(xs, &e);
        let (n, m) = (xs.len(), self.states.len());
        Array2::from_shape_vec((n + 1, m), f).unwrap()
    }

    fn forward_em(&self, xs: &[f64], e: &[f64]) -> Vec<f64> {
        let n = xs.len();
        let m = self.states.len();
        let p = self.silent_start;
        let mut f = vec![NEG_INF; m * (n + 1)];
        f[self.start_index] = 0.0;

        // Transitions between silent states before the first symbol.
        for l in p..m {
            if l == self.start_index {
                continue;
            }
            let mut acc = NEG_INF;
            for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                let ki = self.in_transitions[k];
                if ki < p || ki >= l {
                    continue;
                }
                acc = pair_lse(acc, f[ki] + self.in_log_probs[k]);
            }
            f[l] = acc;
        }

        for i in 0..n {
            for l in 0..p {
                let mut acc = NEG_INF;
                for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                    let ki = self.in_transitions[k];
                    acc = pair_lse(acc, f[i * m + ki] + self.in_log_probs[k]);
                }
                f[(i + 1) * m + l] = acc + e[i + l * n];
            }
            for l in p..m {
                let mut acc = NEG_INF;
                for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                    let ki = self.in_transitions[k];
                    if ki >= p {
                        continue;
                    }
                    acc = pair_lse(acc, f[(i + 1) * m + ki] + self.in_log_probs[k]);
                }
                f[(i + 1) * m + l] = acc;
            }
            for l in p..m {
                let mut acc = NEG_INF;
                for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                    let ki = self.in_transitions[k];
                    if ki < p || ki >= l {
                        continue;
                    }
                    acc = pair_lse(acc, f[(i + 1) * m + ki] + self.in_log_probs[k]);
                }
                f[(i + 1) * m + l] = pair_lse(f[(i + 1) * m + l], acc);
            }
        }
        f
    }

    /// Backward algorithm. Returns the `(n+1) × m` log-probability table.
    pub fn backward(&self, xs: &[f64]) -> Array2<f64> {
        let e = self.emissions(xs);
        let b = self.backward_em(xs, &e);
        let (n, m) = (xs.len(), self.states.len());
        Array2::from_shape_vec((n + 1, m), b).unwrap()
    }

    fn backward_em(&self, xs: &[f64], e: &[f64]) -> Vec<f64> {
        let n = xs.len();
        let m = self.states.len();
        let p = self.silent_start;
        let mut b = vec![NEG_INF; m * (n + 1)];

        // Base case: end the sequence in the end state.
        if self.finite {
            b[n * m + self.end_index] = 0.0;
            // Silent states at t = n.
            for kr in 0..(m - p) {
                let k = m - kr - 1;
                if k == self.end_index {
                    continue;
                }
                let mut acc = NEG_INF;
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li < k + 1 {
                        continue;
                    }
                    acc = pair_lse(acc, b[n * m + li] + self.out_log_probs[l]);
                }
                b[n * m + k] = acc;
            }
            for k in 0..p {
                let mut acc = NEG_INF;
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li < p {
                        continue;
                    }
                    acc = pair_lse(acc, b[n * m + li] + self.out_log_probs[l]);
                }
                b[n * m + k] = acc;
            }
        } else {
            for i in 0..p {
                b[n * m + i] = 0.0;
            }
        }

        // Recurrence, walking time backwards.
        for ir in 0..n {
            let i = n - ir - 1;
            // Silent states depend on subsequent non-silent states.
            for kr in 0..(m - p) {
                let k = m - kr - 1;
                let mut acc = NEG_INF;
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li >= p {
                        continue;
                    }
                    acc = pair_lse(
                        acc,
                        b[(i + 1) * m + li] + self.out_log_probs[l] + e[i + li * n],
                    );
                }
                b[i * m + k] = acc;
            }
            // Silent states depend on other current-step silent states.
            for kr in 0..(m - p) {
                let k = m - kr - 1;
                let mut acc = NEG_INF;
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li < k + 1 {
                        continue;
                    }
                    acc = pair_lse(acc, b[i * m + li] + self.out_log_probs[l]);
                }
                b[i * m + k] = pair_lse(acc, b[i * m + k]);
            }
            // Emitting states.
            for k in 0..p {
                let mut acc = NEG_INF;
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li >= p {
                        continue;
                    }
                    acc = pair_lse(
                        acc,
                        b[(i + 1) * m + li] + self.out_log_probs[l] + e[i + li * n],
                    );
                }
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li < p {
                        continue;
                    }
                    acc = pair_lse(acc, b[i * m + li] + self.out_log_probs[l]);
                }
                b[i * m + k] = acc;
            }
        }
        b
    }

    /// Log probability of a sequence under the model.
    pub fn log_probability(&self, xs: &[f64]) -> f64 {
        let e = self.emissions(xs);
        let f = self.forward_em(xs, &e);
        let (n, m) = (xs.len(), self.states.len());
        if self.finite {
            f[n * m + self.end_index]
        } else {
            (0..self.silent_start).fold(NEG_INF, |a, i| pair_lse(a, f[n * m + i]))
        }
    }

    /// Log posterior probability of each emitting state at each time step.
    pub fn predict_log_proba(&self, xs: &[f64]) -> Array2<f64> {
        let n = xs.len();
        let m = self.states.len();
        let p = self.silent_start;
        let e = self.emissions(xs);
        let f = self.forward_em(xs, &e);
        let b = self.backward_em(xs, &e);
        let logp_seq = if self.finite {
            f[n * m + self.end_index]
        } else {
            (0..p).fold(NEG_INF, |a, i| pair_lse(a, f[n * m + i]))
        };
        let mut r = Array2::zeros((n, p));
        for k in 0..p {
            for i in 0..n {
                r[[i, k]] = f[(i + 1) * m + k] + b[(i + 1) * m + k] - logp_seq;
            }
        }
        r
    }

    /// Posterior probability of each emitting state at each time step.
    pub fn predict_proba(&self, xs: &[f64]) -> Array2<f64> {
        self.predict_log_proba(xs).mapv(|x| x.exp())
    }

    /// Maximum-a-posteriori state per time step (`predict`, algorithm='map').
    pub fn predict(&self, xs: &[f64]) -> Vec<usize> {
        let r = self.predict_log_proba(xs);
        (0..r.nrows())
            .map(|i| {
                let row = r.row(i);
                let mut best = 0;
                let mut bestv = row[0];
                for (k, &v) in row.iter().enumerate().skip(1) {
                    if v > bestv {
                        bestv = v;
                        best = k;
                    }
                }
                best
            })
            .collect()
    }

    /// Viterbi algorithm. Returns `(logp, path)` where `path` includes the
    /// silent `start` and `end` states (pomegranate returns these too).
    pub fn viterbi(&self, xs: &[f64]) -> (f64, Vec<usize>) {
        let n = xs.len();
        let m = self.states.len();
        let p = self.silent_start;
        let e = self.emissions(xs);
        let mut v = vec![NEG_INF; m * (n + 1)];
        let mut tracebackx = vec![0usize; m * (n + 1)];
        let mut tracebacky = vec![0usize; m * (n + 1)];
        v[self.start_index] = 0.0;

        for l in p..m {
            if l == self.start_index {
                continue;
            }
            for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                let ki = self.in_transitions[k];
                if ki < p || ki >= l {
                    continue;
                }
                let slp = v[ki] + self.in_log_probs[k];
                if slp > v[l] {
                    v[l] = slp;
                    tracebackx[l] = 0;
                    tracebacky[l] = ki;
                }
            }
        }

        for i in 0..n {
            for l in 0..p {
                v[(i + 1) * m + l] = NEG_INF;
                for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                    let ki = self.in_transitions[k];
                    let slp = v[i * m + ki] + self.in_log_probs[k] + e[i + l * n];
                    if slp > v[(i + 1) * m + l] {
                        v[(i + 1) * m + l] = slp;
                        tracebackx[(i + 1) * m + l] = i;
                        tracebacky[(i + 1) * m + l] = ki;
                    }
                }
            }
            for l in p..m {
                v[(i + 1) * m + l] = NEG_INF;
                for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                    let ki = self.in_transitions[k];
                    if ki >= p {
                        continue;
                    }
                    let slp = v[(i + 1) * m + ki] + self.in_log_probs[k];
                    if slp > v[(i + 1) * m + l] {
                        v[(i + 1) * m + l] = slp;
                        tracebackx[(i + 1) * m + l] = i + 1;
                        tracebacky[(i + 1) * m + l] = ki;
                    }
                }
            }
            for l in p..m {
                for k in self.in_edge_count[l]..self.in_edge_count[l + 1] {
                    let ki = self.in_transitions[k];
                    if ki < p || ki >= l {
                        continue;
                    }
                    let slp = v[(i + 1) * m + ki] + self.in_log_probs[k];
                    if slp > v[(i + 1) * m + l] {
                        v[(i + 1) * m + l] = slp;
                        tracebackx[(i + 1) * m + l] = i + 1;
                        tracebacky[(i + 1) * m + l] = ki;
                    }
                }
            }
        }

        let (logp, end_index) = if self.finite {
            (v[n * m + self.end_index], self.end_index)
        } else {
            let mut ei = 0;
            let mut lp = NEG_INF;
            for i in 0..m {
                if v[n * m + i] > lp {
                    lp = v[n * m + i];
                    ei = i;
                }
            }
            (lp, ei)
        };
        if logp == NEG_INF {
            return (logp, Vec::new());
        }

        let mut path = vec![0usize; n + m];
        let mut length = 0usize;
        let mut px = n;
        let mut py = end_index;
        while px != 0 || py != self.start_index {
            path[length] = py;
            length += 1;
            let npx = tracebackx[px * m + py];
            py = tracebacky[px * m + py];
            px = npx;
        }
        path[length] = py;
        // pomegranate reverses `path[0..=length]` with `range((length+1)//2)`.
        for i in 0..length.div_ceil(2) {
            path.swap(i, length - i);
        }
        (logp, path[..=length].to_vec())
    }

    /// E step: accumulate expected transition counts and per-state emission
    /// summaries for one sequence. Returns the sequence's log probability.
    // The CSR loops mirror pomegranate's Cython pointer loops one-to-one.
    #[allow(clippy::needless_range_loop)]
    pub fn summarize(&mut self, xs: &[f64], sequence_weight: f64) -> f64 {
        let n = xs.len();
        let m = self.states.len();
        let p = self.silent_start;
        let e = self.emissions(xs);
        let f = self.forward_em(xs, &e);
        let b = self.backward_em(xs, &e);
        let logp_seq = if self.finite {
            f[n * m + self.end_index]
        } else {
            (0..p).fold(NEG_INF, |a, i| pair_lse(a, f[n * m + i]))
        };

        if logp_seq != NEG_INF {
            let mut local = vec![0.0; self.n_edges];
            for k in 0..m {
                // Emitting targets.
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li >= p {
                        continue;
                    }
                    let mut acc = NEG_INF;
                    for i in 0..n {
                        acc = pair_lse(
                            acc,
                            f[i * m + k]
                                + self.out_log_probs[l]
                                + e[li * n + i]
                                + b[(i + 1) * m + li],
                        );
                    }
                    local[l] += (acc - logp_seq).exp();
                }
                // Silent targets.
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    if li < p {
                        continue;
                    }
                    let mut acc = NEG_INF;
                    for i in 0..=n {
                        acc = pair_lse(acc, f[i * m + k] + self.out_log_probs[l] + b[i * m + li]);
                    }
                    local[l] += (acc - logp_seq).exp();
                }
                if k < p {
                    let mut weights = vec![0.0; n];
                    for i in 0..n {
                        weights[i] = (f[(i + 1) * m + k] + b[(i + 1) * m + k] - logp_seq).exp()
                            * sequence_weight;
                    }
                    let ei = self.distributions[k];
                    self.emissions[ei].summarize(xs, &weights);
                }
            }
            for l in 0..self.n_edges {
                self.expected_transitions[l] += local[l] * sequence_weight;
            }
        }
        self.n_summarized += 1;
        logp_seq
    }

    /// M step: re-estimate transition probabilities and emission parameters
    /// from the accumulated summaries (pomegranate `from_summaries`).
    pub fn from_summaries(
        &mut self,
        transition_pseudocount: f64,
        emission_pseudocount: f64,
        use_pseudocount: bool,
        edge_inertia: f64,
        distribution_inertia: f64,
    ) {
        if self.n_summarized == 0 {
            return;
        }
        let m = self.states.len();
        let mut expected = vec![0.0; m * m];
        for k in 0..m {
            for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                expected[k * m + self.out_transitions[l]] = self.expected_transitions[l];
            }
        }
        let use_pc = use_pseudocount as usize as f64;
        let mut norm = vec![0.0; m];
        for k in 0..m {
            for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                norm[k] += expected[k * m + self.out_transitions[l]]
                    + transition_pseudocount
                    + self.out_pseudocounts[l] * use_pc;
            }
        }
        for k in 0..m {
            if norm[k] > 0.0 {
                for l in self.out_edge_count[k]..self.out_edge_count[k + 1] {
                    let li = self.out_transitions[l];
                    let prob = (expected[k * m + li]
                        + transition_pseudocount
                        + self.out_pseudocounts[l] * use_pc)
                        / norm[k];
                    self.out_log_probs[l] = (self.out_log_probs[l].exp() * edge_inertia
                        + prob * (1.0 - edge_inertia))
                        .ln();
                }
            }
            // In-transitions are re-estimated for every state k, including the
            // silent end state whose `norm[k]` is 0 (it has no out-edges); the
            // guard is on the *source* state's norm, not k's. pomegranate's
            // `_from_summaries` keeps this loop outside `if norm[k] > 0`.
            for l in self.in_edge_count[k]..self.in_edge_count[k + 1] {
                let li = self.in_transitions[l];
                if norm[li] > 0.0 {
                    let prob = (expected[li * m + k]
                        + transition_pseudocount
                        + self.in_pseudocounts[l] * use_pc)
                        / norm[li];
                    self.in_log_probs[l] = (self.in_log_probs[l].exp() * edge_inertia
                        + prob * (1.0 - edge_inertia))
                        .ln();
                }
            }
        }
        for k in 0..self.silent_start {
            let ei = self.distributions[k];
            self.emissions[ei].from_summaries(distribution_inertia, emission_pseudocount);
        }
        self.expected_transitions.iter_mut().for_each(|x| *x = 0.0);
        self.n_summarized = 0;
    }

    /// Baum-Welch training over multiple sequences (pomegranate `fit`).
    /// Returns the total improvement in log probability.
    pub fn fit(
        &mut self,
        sequences: &[Vec<f64>],
        stop_threshold: f64,
        max_iterations: usize,
        min_iterations: usize,
        use_pseudocount: bool,
    ) -> f64 {
        let mut iteration = 0usize;
        let mut improvement = f64::INFINITY;
        let mut total_improvement = 0.0;
        let mut last = 0.0;
        while improvement > stop_threshold || iteration < min_iterations + 1 {
            self.from_summaries(0.0, 0.0, use_pseudocount, 0.0, 0.0);
            let logp_sum: f64 = sequences.iter().map(|s| self.summarize(s, 1.0)).sum();
            if iteration > 0 {
                improvement = logp_sum - last;
                total_improvement += improvement;
            }
            iteration += 1;
            last = logp_sum;
            if iteration > max_iterations {
                break;
            }
        }
        total_improvement
    }
}
