//! End-to-end regression: the Rust `domaincaller` port must reproduce
//! TADLib's `tadlib/domaincaller/chromLev.py` output on
//! `mES_rep2.40Kb.raw.chr19.mat` (mouse ES cells, chr19, 40 kb; the same
//! matrix the OnTAD regression uses).
//!
//! ## Golden files
//!
//! `data/mES_rep2.40Kb.raw.chr19.tadlib.dis` is the per-bin Directionality
//! Index array and `data/mES_rep2.40Kb.raw.chr19.tadlib.domains` is the final
//! domain list `[start, end, noise, level]` in base pairs (97 domains). Both
//! were produced by running TADLib (with pomegranate 0.10.0)
//! `domaincaller.Chrom.callDomains()` on the matrix and are bit-exact
//! reference values: the Rust port must match them to `1e-6` (DI) and exactly
//! (domains).
//!
//! ## Regenerating the golden files
//!
//! Requires a Python 3.11 venv with numpy 1.26, scipy, networkx 1.11,
//! matplotlib, and pomegranate 0.10.0. pomegranate 0.10.0 no longer builds
//! on modern toolchains via `pip install .`; the extracted
//! `pomegranate-0.10.0/` tree ships pre-generated Cython `.c` files, so
//! `python setup.py build_ext --inplace` (with *no* Cython installed) compiles
//! them directly. networkx 1.11 needs `fractions.gcd` patched to `math.gcd`
//! on py3.11.
//!
//! ```python
//! import sys, math, fractions, types
//! fractions.gcd = math.gcd
//! sys.path.insert(0, '/path/to/TADLib')
//! sys.path.insert(0, '/path/to/pomegranate-0.10.0')   # in-place built .so
//!
//! # chromLev.py imports tadlib.calfea.analyze (sklearn/scipy.stats.itemfreq)
//! # but never calls it on this path; stub it out so the import succeeds.
//! stub = types.ModuleType('tadlib.calfea.analyze')
//! stub.Core = stub.manipulation = None
//! sys.modules['tadlib.calfea.analyze'] = stub
//!
//! import numpy as np
//! from scipy.sparse import csr_matrix
//! from tadlib.domaincaller.chromLev import Chrom
//! from pomegranate import NormalDistribution, HiddenMarkovModel, GeneralMixtureModel, State
//!
//! M = np.loadtxt('mES_rep2.40Kb.raw.chr19.mat')
//! smat = csr_matrix(np.triu(M))          # upper triangle, as genomeLev feeds in
//! chrom = Chrom('chr19', 40000, smat)
//! chrom.minWindows(0, chrom.chromLen, chrom._dw)
//! chrom.calDI(chrom.windows, 0)
//! chrom.splitChrom(chrom.DIs)
//!
//! # The 4-state GMM-HMM, built exactly as tadlib/hitad/genomeLev.oriHMMParams.
//! hmm = HiddenMarkovModel()
//! nd, var = 3, 7.5 / (3 - 1)             # Gaussian *std* = 3.75
//! means = [[], [], [], []]
//! for i in range(nd):
//!     means[3].append(i * 7.5 / (nd - 1) + 2.5)
//!     means[2].append(i * 7.5 / (nd - 1))
//!     means[1].append(-i * 7.5 / (nd - 1))
//!     means[0].append(-i * 7.5 / (nd - 1) - 2.5)
//! states = [State(GeneralMixtureModel([NormalDistribution(m, var) for m in ms]), name=str(i))
//!           for i, ms in enumerate(means)]
//! hmm.add_states(*states)
//! hmm.add_transition(states[0], states[1], 1)
//! hmm.add_transition(states[1], states[1], .5)
//! hmm.add_transition(states[1], states[2], .5)
//! hmm.add_transition(states[2], states[2], .5)
//! hmm.add_transition(states[2], states[3], .5)
//! hmm.add_transition(states[3], states[0], 1)
//! hmm.add_transition(hmm.start, states[0], 1)
//! hmm.add_transition(states[3], hmm.end, 1)
//! hmm.bake()
//!
//! # Train on the non-zero DI segments (length > 20), as genomeLev.learning.
//! seqs = [seg[seg != 0] for seg in chrom.regionDIs.values() if (seg[seg != 0]).size > 20]
//! hmm.fit(seqs, algorithm='baum-welch', max_iterations=10000,
//!         stop_threshold=1e-5, n_jobs=1, verbose=False)
//! chrom.hmm = hmm
//! chrom.callDomains()
//!
//! np.savetxt('mES_rep2.40Kb.raw.chr19.tadlib.dis', chrom.DIs, fmt='%.17g')
//! with open('mES_rep2.40Kb.raw.chr19.tadlib.domains', 'w') as f:
//!     for d in chrom.domains:
//!         f.write(f'{d[0]}\t{d[1]}\t{d[2]}\t{d[3]}\n')
//! ```
//!
//! Note: `chrom.callDomains()` runs `oriIter` with early exit via
//! `DomainAligner` (mismatch ratio < 0.05). On this matrix the loop runs all
//! five rounds, but the golden captures the exact TADLib output either way.

use cooler_rs::domaincaller::Chrom;

const MATRIX: &str = include_str!("data/mES_rep2.40Kb.raw.chr19.mat");
const GOLDEN_DIS: &str = include_str!("data/mES_rep2.40Kb.raw.chr19.tadlib.dis");
const GOLDEN_DOMAINS: &str = include_str!("data/mES_rep2.40Kb.raw.chr19.tadlib.domains");

// Helpers

/// Parse a dense N×N whitespace-separated text matrix.
fn parse_dense_matrix(text: &str) -> ndarray::Array2<f64> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let first = lines.next().expect("empty matrix");
    let n = first.split_whitespace().count();
    let mut x = ndarray::Array2::zeros((n, n));
    for (j, field) in first.split_whitespace().enumerate() {
        x[[0, j]] = field.parse().expect("non-numeric matrix entry");
    }
    for (i, line) in lines.enumerate() {
        for (j, field) in line.split_whitespace().enumerate() {
            x[[i + 1, j]] = field.parse().expect("non-numeric matrix entry");
        }
    }
    x
}

/// Run the Rust domaincaller on the test matrix.
fn run_domaincaller() -> Chrom {
    let m = parse_dense_matrix(MATRIX);
    let n = m.nrows();
    // upper triangle, as TADLib's genomeLev feeds in (triu(...).tocsr())
    let mut entries = Vec::new();
    for i in 0..n {
        for j in i..n {
            let v = m[[i, j]];
            if v != 0.0 {
                entries.push((i, j, v));
            }
        }
    }
    let mut chrom = Chrom::new("chr19", 40_000, n, &entries);
    chrom.call_domains();
    chrom
}

#[test]
fn di_track_matches_tadlib() {
    let chrom = run_domaincaller();
    let golden: Vec<f64> = GOLDEN_DIS
        .lines()
        .map(|l| l.parse().expect("bad golden dis"))
        .collect();
    assert_eq!(chrom.dis.len(), golden.len(), "DI length");
    // The DI pipeline must be essentially exact; allow tiny float slack.
    for (i, (&got, &exp)) in chrom.dis.iter().zip(golden.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "DI[{i}] = {got} != TADLib {exp} (diff {})",
            (got - exp).abs()
        );
    }
}

#[test]
fn domains_match_tadlib() {
    let chrom = run_domaincaller();
    let golden: Vec<(f64, f64)> = GOLDEN_DOMAINS
        .lines()
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            (f[0], f[1])
        })
        .collect();
    let got: Vec<(f64, f64)> = chrom.domains.iter().map(|d| (d[0], d[1])).collect();
    assert_eq!(
        got.len(),
        golden.len(),
        "domain count: got {} != {}",
        got.len(),
        golden.len()
    );
    for (i, ((gs, ge), (ss, se))) in golden.iter().zip(got.iter()).enumerate() {
        assert_eq!(ss, gs, "domain {i} start: Rust {ss} != TADLib {gs}");
        assert_eq!(se, ge, "domain {i} end: Rust {se} != TADLib {ge}");
    }
}
