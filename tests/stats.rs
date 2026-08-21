//! Regression tests for `cooler_rs::stats` — the pomegranate 0.10.0 port.
//!
//! The hard-coded numeric oracles come verbatim from
//! `pomegranate/tests/test_hmm.py` and `tests/test_gmm.py` (univariate-Gaussian
//! dense HMM, univariate GMM, discrete HMM) — exactly the model types
//! `domaincaller` uses. Plus the `test_hmm_bw_fit` Baum-Welch oracle
//! (total_improvement 83.1132, verified against real pomegranate 0.10.0) and
//! the key-order-invariance check that makes that oracle reproducible despite
//! pomegranate's `set()`-ordered keymap.

use ndarray::Array2;

use cooler_rs::stats::{
    DiscreteDistribution, GeneralMixtureModel, HiddenMarkovModel, NormalDistribution, END, START,
};

// Model builders

/// `dense_model(d1..d4)` from `test_hmm.py` with
/// `NormalDistribution(5,1),(1,1),(13,2),(16,.5)`.
fn gaussian_dense_model() -> HiddenMarkovModel {
    let mut m = HiddenMarkovModel::new("model");
    let d1 = m.add_emission(Box::new(NormalDistribution::new(5.0, 1.0)));
    let d2 = m.add_emission(Box::new(NormalDistribution::new(1.0, 1.0)));
    let d3 = m.add_emission(Box::new(NormalDistribution::new(13.0, 2.0)));
    let d4 = m.add_emission(Box::new(NormalDistribution::new(16.0, 0.5)));
    let s1 = m.add_state("s1", d1, 1.0);
    let s2 = m.add_state("s2", d2, 1.0);
    let s3 = m.add_state("s3", d3, 1.0);
    let s4 = m.add_state("s4", d4, 1.0);

    m.add_transition(START, s1, 0.1);
    m.add_transition(START, s2, 0.3);
    m.add_transition(START, s3, 0.2);
    m.add_transition(START, s4, 0.4);
    m.add_transition(s1, s1, 0.5);
    m.add_transition(s1, s2, 0.1);
    m.add_transition(s1, s3, 0.1);
    m.add_transition(s1, s4, 0.2);
    m.add_transition(s2, s1, 0.2);
    m.add_transition(s2, s2, 0.1);
    m.add_transition(s2, s3, 0.4);
    m.add_transition(s2, s4, 0.2);
    m.add_transition(s3, s1, 0.1);
    m.add_transition(s3, s2, 0.1);
    m.add_transition(s3, s3, 0.3);
    m.add_transition(s3, s4, 0.4);
    m.add_transition(s4, s1, 0.2);
    m.add_transition(s4, s2, 0.2);
    m.add_transition(s4, s3, 0.1);
    m.add_transition(s4, s4, 0.4);
    m.add_transition(s1, END, 0.1);
    m.add_transition(s2, END, 0.1);
    m.add_transition(s3, END, 0.1);
    m.add_transition(s4, END, 0.1);
    m.bake();
    m
}

/// numpy `assert_array_almost_equal` default: `|a-b| < 1.5e-6`.
fn assert_close(actual: f64, expected: f64) {
    if expected == f64::NEG_INFINITY {
        assert_eq!(actual, f64::NEG_INFINITY, "expected -inf");
        return;
    }
    assert!(
        (actual - expected).abs() < 1.5e-6,
        "actual {actual} != expected {expected}"
    );
}

fn assert_table_close(actual: &Array2<f64>, expected: &[&[f64]]) {
    let (nr, nc) = actual.dim();
    assert_eq!(nr, expected.len(), "row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(nc, row.len(), "col count at row {i}");
        for (j, &exp) in row.iter().enumerate() {
            assert_close(actual[[i, j]], exp);
        }
    }
}

// HMM inference oracles (univariate Gaussian dense)
// `test_hmm_univariate_gaussian_dense_forward`
#[test]
fn hmm_gaussian_dense_forward() {
    let model = gaussian_dense_model();
    let f = model.forward(&[3.0, 5.0, 8.0, 19.0, 13.0]);
    let expected = [
        &[
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            f64::NEG_INFINITY,
        ][..],
        &[
            -5.221523626198319,
            -4.122911337530209,
            -15.72152362619832,
            -339.14208208451845,
            f64::NEG_INFINITY,
            -6.137807473983832,
        ][..],
        &[
            -6.045149476287824,
            -15.056746007188107,
            -14.571238720950003,
            -247.67044476202696,
            f64::NEG_INFINITY,
            -8.347414404915495,
        ][..],
        &[
            -12.157146753448874,
            -33.766352938119766,
            -13.083738234853534,
            -135.8798604314077,
            f64::NEG_INFINITY,
            -14.126191869688759,
        ][..],
        &[
            -111.69303081629936,
            -177.04513040289305,
            -19.78896563212587,
            -31.409154369913637,
            f64::NEG_INFINITY,
            -22.091541742268795,
        ][..],
        &[
            -55.01047129270264,
            -95.01047129270263,
            -22.605021155923353,
            -38.93103873379323,
            f64::NEG_INFINITY,
            -24.90760616769035,
        ][..],
    ];
    assert_table_close(&f, &expected);
}

// `test_hmm_univariate_gaussian_dense_backward`
#[test]
fn hmm_gaussian_dense_backward() {
    let model = gaussian_dense_model();
    let b = model.backward(&[3.0, 5.0, 8.0, 19.0, 13.0]);
    let expected = [
        &[
            -24.010022764471987,
            -24.820878919065986,
            -25.359784144328874,
            -24.666641886986977,
            -24.907606167690343,
            f64::NEG_INFINITY,
        ][..],
        &[
            -20.47495458748052,
            -21.390489786220005,
            -22.08295469697486,
            -21.390527588974052,
            -22.081667004696868,
            f64::NEG_INFINITY,
        ][..],
        &[
            -18.86308443640411,
            -18.00715991329825,
            -18.32109321121691,
            -19.183852463904262,
            -18.700307092256082,
            f64::NEG_INFINITY,
        ][..],
        &[
            -13.533310680731095,
            -12.147019061523556,
            -12.434699610689767,
            -13.533307024859651,
            -12.840163500171148,
            f64::NEG_INFINITY,
        ][..],
        &[
            -6.217255777912353,
            -4.830961508172448,
            -5.1186435298576365,
            -6.217255656072613,
            -5.524108597352521,
            f64::NEG_INFINITY,
        ][..],
        &[
            -2.3025850929940455,
            -2.3025850929940455,
            -2.3025850929940455,
            -2.3025850929940455,
            f64::NEG_INFINITY,
            0.0,
        ][..],
    ];
    assert_table_close(&b, &expected);
}

// `test_hmm_univariate_gaussian_dense_predict_log_proba`
#[test]
fn hmm_gaussian_dense_predict_log_proba() {
    let model = gaussian_dense_model();
    let r = model.predict_log_proba(&[3.0, 5.0, 8.0, 19.0, 13.0]);
    let expected = [
        &[-0.78887205, -0.60579496, -12.89687216, -335.62500351][..],
        &[-0.00062775, -8.15629975, -7.98472576, -241.94669106][..],
        &[-0.78285127, -21.00576583, -0.61083168, -124.50556129][..],
        &[-93.00268043, -156.96848574, -2.99e-06, -12.71880386][..],
        &[-32.40545022, -72.40545022, -8e-08, -16.32601766][..],
    ];
    assert_table_close(&r, &expected);
}

// `test_hmm_univariate_gaussian_dense_predict_proba`
#[test]
fn hmm_gaussian_dense_predict_proba() {
    let model = gaussian_dense_model();
    let p = model.predict_proba(&[3.0, 5.0, 8.0, 19.0, 13.0]);
    let expected = [
        &[0.454357, 0.54564049, 2.51e-06, 0.0][..],
        &[0.99937245, 0.00028692, 0.00034063, 0.0][..],
        &[0.45710084, 0.0, 0.54289916, 0.0][..],
        &[0.0, 0.0, 0.99999701, 2.99e-06][..],
        &[0.0, 0.0, 0.99999992, 8e-08][..],
    ];
    assert_table_close(&p, &expected);
}

// `test_hmm_univariate_gaussian_dense_predict` (algorithm='map')
#[test]
fn hmm_gaussian_dense_predict_map() {
    let model = gaussian_dense_model();
    assert_eq!(
        model.predict(&[3.0, 5.0, 8.0, 19.0, 13.0]),
        vec![1, 0, 2, 2, 2]
    );
}

// `test_hmm_univariate_gaussian_dense_predict_viterbi`
#[test]
fn hmm_gaussian_dense_predict_viterbi() {
    let model = gaussian_dense_model();
    let (_logp, path) = model.viterbi(&[3.0, 5.0, 8.0, 19.0, 13.0]);
    assert_eq!(path, vec![4, 1, 0, 2, 2, 2, 5]);
}

// GMM oracles
// `test_gmm_univariate_gaussian_log_probability` (all 8 datasets)
#[test]
fn gmm_univariate_gaussian_log_probability() {
    let components = (0..3)
        .map(|i| NormalDistribution::new(i as f64 * 3.0, 1.0))
        .collect();
    let gmm = GeneralMixtureModel::new(components, None);

    let datasets = [
        (
            [1.1, 2.7, 3.0, 4.8, 6.2],
            [
                -2.35925975,
                -2.03120691,
                -1.99557605,
                -2.39638244,
                -2.03147258,
            ],
        ),
        (
            [1.8, 2.1, 3.1, 5.2, 6.5],
            [
                -2.39618117,
                -2.26893273,
                -1.9995911,
                -2.22202965,
                -2.14007514,
            ],
        ),
        (
            [0.9, 2.2, 3.2, 5.0, 5.8],
            [
                -2.26957032,
                -2.22113386,
                -2.01155305,
                -2.31613252,
                -2.01751101,
            ],
        ),
        (
            [1.0, 2.1, 3.5, 4.3, 5.2],
            [
                -2.31613252,
                -2.26893273,
                -2.09160506,
                -2.42491769,
                -2.22202965,
            ],
        ),
        (
            [1.2, 2.9, 3.1, 4.2, 5.5],
            [
                -2.39638244,
                -1.9995911,
                -1.9995911,
                -2.39618117,
                -2.09396318,
            ],
        ),
        (
            [1.8, 1.9, 3.0, 4.9, 5.7],
            [
                -2.39618117,
                -2.35895351,
                -1.99557605,
                -2.35925975,
                -2.03559364,
            ],
        ),
        (
            [1.2, 3.1, 2.9, 4.2, 5.9],
            [
                -2.39638244,
                -1.9995911,
                -1.9995911,
                -2.39618117,
                -2.00766654,
            ],
        ),
        (
            [1.0, 2.9, 3.9, 4.1, 6.0],
            [
                -2.31613252,
                -1.9995911,
                -2.26893273,
                -2.35895351,
                -2.00650306,
            ],
        ),
    ];
    for (xs, expected) in datasets {
        for (x, &exp) in xs.iter().zip(expected.iter()) {
            assert_close(gmm.log_probability(*x), exp);
        }
    }
}

// Discrete HMM

/// The discrete dense model from `setup_univariate_discrete_dense`.
fn discrete_dense_model() -> HiddenMarkovModel {
    let mut m = HiddenMarkovModel::new("model");
    let d1 = m.add_emission(Box::new(DiscreteDistribution::new(&[
        0.90, 0.02, 0.03, 0.05,
    ])));
    let d2 = m.add_emission(Box::new(DiscreteDistribution::new(&[
        0.02, 0.90, 0.03, 0.05,
    ])));
    let d3 = m.add_emission(Box::new(DiscreteDistribution::new(&[
        0.03, 0.02, 0.90, 0.05,
    ])));
    let d4 = m.add_emission(Box::new(DiscreteDistribution::new(&[
        0.05, 0.02, 0.03, 0.90,
    ])));
    let s1 = m.add_state("s1", d1, 1.0);
    let s2 = m.add_state("s2", d2, 1.0);
    let s3 = m.add_state("s3", d3, 1.0);
    let s4 = m.add_state("s4", d4, 1.0);
    m.add_transition(START, s1, 0.1);
    m.add_transition(START, s2, 0.3);
    m.add_transition(START, s3, 0.2);
    m.add_transition(START, s4, 0.4);
    m.add_transition(s1, s1, 0.5);
    m.add_transition(s1, s2, 0.1);
    m.add_transition(s1, s3, 0.1);
    m.add_transition(s1, s4, 0.2);
    m.add_transition(s2, s1, 0.2);
    m.add_transition(s2, s2, 0.1);
    m.add_transition(s2, s3, 0.4);
    m.add_transition(s2, s4, 0.2);
    m.add_transition(s3, s1, 0.1);
    m.add_transition(s3, s2, 0.1);
    m.add_transition(s3, s3, 0.3);
    m.add_transition(s3, s4, 0.4);
    m.add_transition(s4, s1, 0.2);
    m.add_transition(s4, s2, 0.2);
    m.add_transition(s4, s3, 0.1);
    m.add_transition(s4, s4, 0.4);
    m.add_transition(s1, END, 0.1);
    m.add_transition(s2, END, 0.1);
    m.add_transition(s3, END, 0.1);
    m.add_transition(s4, END, 0.1);
    m.bake();
    m
}

// `test_hmm_univariate_discrete_dense_forward`: dense 4-state model (no
// internal silent states) with discrete emissions. The model is invariant
// under symbol relabeling, so the hard-coded oracle is reproduced for any
// key order; this isolates the discrete-emission + forward machinery from
// the silent-state DP used by the sparse model.
#[test]
fn hmm_discrete_dense_forward() {
    let model = discrete_dense_model();
    // A,B,C,D -> indices 0,1,2,3
    let f = model.forward(&[0.0, 1.0, 3.0, 3.0, 2.0]);
    let expected = [
        &[
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            f64::NEG_INFINITY,
        ][..],
        &[
            -2.40794561,
            -5.11599581,
            -5.11599581,
            -3.91202301,
            f64::NEG_INFINITY,
            -4.40631933,
        ][..],
        &[
            -6.89188193,
            -4.35987383,
            -8.09848286,
            -7.43200392,
            f64::NEG_INFINITY,
            -6.52303724,
        ][..],
        &[
            -8.73634472,
            -9.47926612,
            -8.22377759,
            -5.87605232,
            f64::NEG_INFINITY,
            -8.01305991,
        ][..],
        &[
            -10.28388158,
            -10.39501067,
            -10.80077009,
            -6.76858029,
            f64::NEG_INFINITY,
            -8.99969597,
        ][..],
        &[
            -11.780373,
            -11.84820305,
            -9.00308599,
            -11.14654124,
            f64::NEG_INFINITY,
            -11.09251037,
        ][..],
    ];
    assert_table_close(&f, &expected);
}

// `test_hmm_univariate_discrete_dense_predict_viterbi`
#[test]
fn hmm_discrete_dense_predict_viterbi() {
    let model = discrete_dense_model();
    let (_logp, path) = model.viterbi(&[0.0, 1.0, 3.0, 3.0, 2.0]);
    assert_eq!(path, vec![4, 0, 1, 3, 3, 2, 5]);
}

/// All permutations of `[0,1,2,3]` (Heap's algorithm).
fn permutations(items: &mut [usize], k: usize, out: &mut Vec<Vec<usize>>) {
    if k == items.len() {
        out.push(items.to_vec());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permutations(items, k + 1, out);
        items.swap(k, i);
    }
}

/// The 11-state profile HMM from `test_hmm.py` `sparse_model(d1, d2, d3, i_d)`
/// with the `setup()` discrete distributions. `key_order` is a permutation of
/// `[A,C,G,T]` (base indices) fixing the model's symbol→index mapping.
fn sparse_model(key_order: &[usize]) -> HiddenMarkovModel {
    // Emission probabilities indexed by base alphabet [A,C,G,T].
    let i_d = [0.25, 0.25, 0.25, 0.25];
    let d1 = [0.95, 0.01, 0.01, 0.02];
    let d2 = [0.003, 0.99, 0.003, 0.004];
    let d3 = [0.01, 0.01, 0.01, 0.97];
    let reorder = |t: &[f64; 4]| -> Vec<f64> { key_order.iter().map(|&i| t[i]).collect() };

    let mut m = HiddenMarkovModel::new("Global Alignment");
    let i_d_e = m.add_emission(Box::new(DiscreteDistribution::new(&reorder(&i_d))));
    let d1_e = m.add_emission(Box::new(DiscreteDistribution::new(&reorder(&d1))));
    let d2_e = m.add_emission(Box::new(DiscreteDistribution::new(&reorder(&d2))));
    let d3_e = m.add_emission(Box::new(DiscreteDistribution::new(&reorder(&d3))));
    let i0 = m.add_state("I0", i_d_e, 1.0);
    let i1 = m.add_state("I1", i_d_e, 1.0);
    let i2 = m.add_state("I2", i_d_e, 1.0);
    let i3 = m.add_state("I3", i_d_e, 1.0);
    let m1 = m.add_state("M1", d1_e, 1.0);
    let m2 = m.add_state("M2", d2_e, 1.0);
    let m3 = m.add_state("M3", d3_e, 1.0);
    let d1s = m.add_silent_state("D1");
    let d2s = m.add_silent_state("D2");
    let d3s = m.add_silent_state("D3");

    m.add_transition(START, m1, 0.9);
    m.add_transition(START, i0, 0.1);
    m.add_transition(m1, m2, 0.9);
    m.add_transition(m1, i1, 0.05);
    m.add_transition(m1, d2s, 0.05);
    m.add_transition(m2, m3, 0.9);
    m.add_transition(m2, i2, 0.05);
    m.add_transition(m2, d3s, 0.05);
    m.add_transition(m3, END, 0.9);
    m.add_transition(m3, i3, 0.1);
    m.add_transition(i0, i0, 0.70);
    m.add_transition(i0, d1s, 0.15);
    m.add_transition(i0, m1, 0.15);
    m.add_transition(i1, i1, 0.70);
    m.add_transition(i1, d2s, 0.15);
    m.add_transition(i1, m2, 0.15);
    m.add_transition(i2, i2, 0.70);
    m.add_transition(i2, d3s, 0.15);
    m.add_transition(i2, m3, 0.15);
    m.add_transition(i3, i3, 0.85);
    m.add_transition(i3, END, 0.15);
    m.add_transition(d1s, d2s, 0.15);
    m.add_transition(d1s, i1, 0.15);
    m.add_transition(d1s, m2, 0.70);
    m.add_transition(d2s, d3s, 0.15);
    m.add_transition(d2s, i2, 0.15);
    m.add_transition(d2s, m3, 0.70);
    m.add_transition(d3s, i3, 0.30);
    m.add_transition(d3s, END, 0.70);
    m.bake();
    m
}

// Baum-Welch training
// `test_hmm_bw_fit`: Baum-Welch with `use_pseudocount=True`, 5 iterations,
// must give `total_improvement == 83.1132` (verified against pomegranate
// 0.10.0 running on py3.11: 83.11321266465426). The discrete HMM is invariant
// under symbol relabeling — permuting the alphabet and the sequence encoding
// together yields an isomorphic model — so the canonical key order `A,C,G,T`
// suffices and the result is independent of pomegranate's `set()` ordering.
#[test]
fn hmm_bw_fit_discrete_reproduces_pomegranate_oracle() {
    let seqs_str = [
        "ACT", "ACT", "ACC", "ACTC", "ACT", "ACT", "CCT", "CCC", "AAT", "CT", "AT", "CT", "CT",
        "CT", "CT", "CT", "CT", "ACT", "ACT", "CT", "ACT", "CT", "CT", "CT", "CT",
    ];
    let mut model = sparse_model(&[0, 1, 2, 3]); // A,C,G,T
    let seqs: Vec<Vec<f64>> = seqs_str
        .iter()
        .map(|s| s.chars().map(|c| "ACGT".find(c).unwrap() as f64).collect())
        .collect();
    let improvement = model.fit(&seqs, 1e-9, 5, 0, true);
    assert!(
        (improvement - 83.1132).abs() < 5e-5,
        "total_improvement {improvement} != pomegranate's 83.1132"
    );
}

// The sparse profile HMM is invariant under symbol relabeling: every one of
// the 24 key orders gives the same total_improvement (83.1132). This is why
// the pomegranate oracle is reproducible despite its `set()`-order keymap.
#[test]
fn hmm_bw_fit_discrete_is_key_order_invariant() {
    let seqs_str = [
        "ACT", "ACT", "ACC", "ACTC", "ACT", "ACT", "CCT", "CCC", "AAT", "CT", "AT", "CT", "CT",
        "CT", "CT", "CT", "CT", "ACT", "ACT", "CT", "ACT", "CT", "CT", "CT", "CT",
    ];
    let base = ['A', 'C', 'G', 'T'];
    let mut items = vec![0, 1, 2, 3];
    let mut perms = Vec::new();
    permutations(&mut items, 0, &mut perms);
    let mut improvements = Vec::new();
    for perm in &perms {
        let mut model = sparse_model(perm);
        let seqs: Vec<Vec<f64>> = seqs_str
            .iter()
            .map(|s| {
                s.chars()
                    .map(|c| perm.iter().position(|&i| base[i] == c).unwrap() as f64)
                    .collect()
            })
            .collect();
        improvements.push(model.fit(&seqs, 1e-9, 5, 0, true));
    }
    let first = improvements[0];
    for imp in &improvements {
        assert!(
            (imp - first).abs() < 1e-9,
            "not order-invariant: {improvements:?}"
        );
    }
}

/// EM must monotonically improve the fit on separable 1-D data.
#[test]
fn gmm_univariate_gaussian_fit() {
    // Two well-separated clusters around 0 and 6.
    let mut xs = Vec::new();
    for &(mu, n) in &[(0.0, 30), (6.0, 30)] {
        for k in 0..n {
            xs.push(mu + (k % 5) as f64 - 2.0);
        }
    }
    let components = vec![
        NormalDistribution::new(1.0, 1.0),
        NormalDistribution::new(5.0, 1.0),
    ];
    let mut gmm = GeneralMixtureModel::new(components, None);
    let before: f64 = xs.iter().map(|&x| gmm.log_probability(x)).sum();
    let improvement = gmm.fit(&xs, 1e-9, 100);
    let after: f64 = xs.iter().map(|&x| gmm.log_probability(x)).sum();
    assert!(improvement > 0.0, "EM total improvement must be positive");
    assert!(after > before, "log probability must increase");
}

/// Baum-Welch must increase the log probability of the training sequences.
#[test]
fn hmm_bw_fit_improves() {
    let mut model = gaussian_dense_model();
    let seqs: Vec<Vec<f64>> = vec![
        vec![3.0, 5.0, 8.0, 19.0, 13.0],
        vec![4.0, 6.0, 9.0, 17.0, 12.0],
        vec![2.0, 4.0, 7.0, 20.0, 14.0],
        vec![5.0, 3.0, 6.0, 18.0, 15.0],
    ];
    let before: f64 = seqs.iter().map(|s| model.log_probability(s)).sum();
    let improvement = model.fit(&seqs, 1e-9, 50, 0, false);
    let after: f64 = seqs.iter().map(|s| model.log_probability(s)).sum();
    assert!(
        improvement > 0.0,
        "Baum-Welch total improvement must be positive"
    );
    assert!(after > before, "log probability must increase");
}
