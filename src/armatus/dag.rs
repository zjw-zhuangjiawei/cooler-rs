//! Armatus dynamic program (port of `ArmatusDAG`):
//!
//! ```text
//! OPT(l)  = max{ max_{k<l} OPTD(k-1),  // l ends in a non-domain
//!                OPTD(l) }             // l ends in a domain
//! OPTD(l) = max_{k<l} OPT(k-1) + q(k,l)
//! q(k,l)  = { s(k,l) - mu[d(k,l)]   if > 0
//!           { -inf                  otherwise
//! ```

use std::collections::BinaryHeap;

use super::sums::Sums;
use super::{Domain, WeightedDomainEnsemble};

/// One near-optimal sub-solution: score, the back-pointer `k`, and the rank of
/// the sub-solution at `k-1` it was derived from. `BinaryHeap` is a max-heap,
/// so ordering by score ascending puts the highest-scoring candidate on top
/// (matching the C++ `binomial_heap`).
#[derive(Clone, Copy, Debug)]
pub struct SubProblem {
    pub score: f64,
    pub back_pointer: usize,
    pub back_optimal_index: usize,
}

impl PartialEq for SubProblem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
            && self.back_pointer == other.back_pointer
            && self.back_optimal_index == other.back_optimal_index
    }
}
impl Eq for SubProblem {}
impl PartialOrd for SubProblem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SubProblem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&other.score)
    }
}

pub struct ArmatusDag<'a> {
    sums: &'a Sums,
    k: usize,
    opt: Vec<Vec<SubProblem>>,
    optd: Vec<Vec<SubProblem>>,
}

impl<'a> ArmatusDag<'a> {
    pub fn new(sums: &'a Sums, k: usize) -> Self {
        let n = sums.n;
        let row = vec![
            SubProblem {
                score: f64::NEG_INFINITY,
                back_pointer: 1,
                back_optimal_index: 0
            };
            k
        ];
        ArmatusDag {
            sums,
            k,
            opt: vec![row.clone(); n + 1],
            optd: vec![row; n + 1],
        }
    }

    fn d(&self, k: usize, l: usize) -> usize {
        l - k + 1
    }

    /// `s(k,l) = sums(k-1, l-1) / d(k,l)^gamma`.
    fn s(&self, k: usize, l: usize) -> f64 {
        let d = self.d(k, l);
        self.sums.at(k - 1, l - 1) / (d as f64).powf(self.sums.gamma)
    }

    /// `q(k,l)`, thresholded to `-inf` when not positive.
    fn q(&self, k: usize, l: usize) -> f64 {
        let d = self.d(k, l);
        let score = self.s(k, l) - self.sums.mu[d];
        if score > 0.0 {
            score
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Optimal score of the best full solution.
    pub fn optimal_score(&self) -> f64 {
        self.opt[self.sums.n][0].score
    }

    /// Best (index-0) solution for every `l`; also initialises the base cases
    /// `OPT(0)=OPT(1)=OPTD(0)=OPTD(1)=0` that `compute_top_k` relies on.
    pub fn build(&mut self) {
        let n = self.sums.n;
        let k = self.k;
        let neg_inf = f64::NEG_INFINITY;
        let zero = SubProblem {
            score: 0.0,
            back_pointer: 1,
            back_optimal_index: 0,
        };
        let neg = SubProblem {
            score: neg_inf,
            back_pointer: 1,
            back_optimal_index: 0,
        };

        self.opt[0][0] = zero;
        self.opt[1][0] = zero;
        self.optd[0][0] = zero;
        self.optd[1][0] = zero;
        for i in 1..k {
            self.opt[0][i] = neg;
            self.opt[1][i] = neg;
            self.optd[0][i] = neg;
            self.optd[1][i] = neg;
        }

        for l in 2..=n {
            let mut score_domain = neg_inf;
            let mut score_non_domain = neg_inf;
            let mut bp_domain = 1usize;
            let mut bp_non_domain = 1usize;

            for kk in 1..l {
                if self.optd[kk - 1][0].score > score_non_domain {
                    score_non_domain = self.optd[kk - 1][0].score;
                    bp_non_domain = kk;
                }
            }
            for kk in 1..l {
                let candidate = self.opt[kk - 1][0].score + self.q(kk, l);
                if candidate > score_domain {
                    score_domain = candidate;
                    bp_domain = kk;
                }
            }

            self.optd[l][0] = SubProblem {
                score: score_domain,
                back_pointer: bp_domain,
                back_optimal_index: 0,
            };

            self.opt[l][0] = if score_non_domain > score_domain {
                SubProblem {
                    score: score_non_domain,
                    back_pointer: bp_non_domain,
                    back_optimal_index: 0,
                }
            } else {
                self.optd[l][0]
            };
        }
    }

    /// After popping the top candidate, push its next-ranked sibling
    /// (`back_optimal_index + 1`) so the heap yields the K best in order.
    fn push_next(&self, heap: &mut BinaryHeap<SubProblem>, sub: SubProblem, is_domain: bool, l: usize) {
        let next = sub.back_optimal_index + 1;
        let kk = sub.back_pointer;
        if next < self.k {
            let mut cand = if is_domain {
                let mut c = self.opt[kk - 1][next];
                c.score += self.q(kk, l);
                c
            } else {
                self.optd[kk - 1][next]
            };
            cand.back_pointer = kk;
            cand.back_optimal_index = next;
            heap.push(cand);
        }
    }

    /// Fill `OPT[l][*]` / `OPTD[l][*]` with the top-K solutions for every `l`.
    /// Requires `build()` to have initialised the base cases.
    pub fn compute_top_k(&mut self) {
        let n = self.sums.n;
        let k = self.k;

        for l in 2..=n {
            let mut heap_non_domain: BinaryHeap<SubProblem> = BinaryHeap::new();
            let mut heap_domain: BinaryHeap<SubProblem> = BinaryHeap::new();

            for kk in 1..l {
                let mut nd = self.optd[kk - 1][0];
                nd.back_pointer = kk;
                heap_non_domain.push(nd);

                let mut dm = self.opt[kk - 1][0];
                dm.score += self.q(kk, l);
                dm.back_pointer = kk;
                heap_domain.push(dm);
            }

            let mut i = 0;
            let mut j = 0;
            while i < k {
                let top_domain = heap_domain.pop().unwrap();
                self.optd[l][i] = top_domain;
                self.push_next(&mut heap_domain, top_domain, true, l);

                let non_domain = *heap_non_domain.peek().unwrap();

                if non_domain.score > self.optd[l][j].score {
                    self.opt[l][i] = non_domain;
                    let popped = heap_non_domain.pop().unwrap();
                    self.push_next(&mut heap_non_domain, popped, false, l);
                } else {
                    self.opt[l][i] = self.optd[l][j];
                    j += 1;
                }
                i += 1;
            }
        }
    }

    /// Backtrack the `i`-th best solution into a set of `[k-1, l-1]` domains.
    pub fn extract_domains(&self, i: usize) -> Vec<Domain> {
        let mut dset = Vec::new();
        let mut l = self.sums.n;
        loop {
            let kk = self.opt[l][i].back_pointer;
            if self.q(kk, l) > 0.0 {
                dset.push(Domain {
                    start: kk - 1,
                    end: l - 1,
                });
            }
            l = kk - 1;
            if l <= 1 {
                break;
            }
        }
        dset
    }

    /// All top-K domain sets with weights `score[i] / score[0]`.
    pub fn extract_top_k(&self) -> WeightedDomainEnsemble {
        let base = self.opt[self.sums.n][0].score;
        let mut domain_sets = Vec::with_capacity(self.k);
        let mut weights = Vec::with_capacity(self.k);
        for i in 0..self.k {
            domain_sets.push(self.extract_domains(i));
            weights.push(self.opt[self.sums.n][i].score / base);
        }
        WeightedDomainEnsemble {
            domain_sets,
            weights,
            resolutions: Vec::new(),
        }
    }
}
