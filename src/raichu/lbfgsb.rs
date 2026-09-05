//! Unbounded L-BFGS local minimizer for [`super::dual_annealing`].
//!
//! Raichu calls scipy's `dual_annealing` with `minimizer_kwargs =
//! {'method': 'L-BFGS-B'}` and no `jac` and no `bounds` (see
//! `LocalSearchWrapper.__init__` in scipy `_dual_annealing.py`): a non-empty
//! `minimizer_kwargs` skips the bounds/options defaults, so the local search
//! is an *unbounded* L-BFGS-B run at scipy's default settings (`maxiter`
//! 15000, `gtol` 1e-5, `ftol` 2.22e-9) with a forward finite-difference
//! gradient (scipy's default when `jac` is absent). Bounds are only checked
//! by the caller after the fact, and out-of-bounds results are rejected.
//!
//! This is a standard Nocedal two-loop L-BFGS with an Armijo backtracking
//! line search — same algorithm family as scipy's Fortran L-BFGS-B, so it
//! converges to the same optimum; it is not bit-for-bit identical.

/// Forward finite-difference gradient, matching scipy's default `2-point`
/// `approx_derivative`: `h_i = sign(x_i) * sqrt(eps)` (and `+sqrt(eps)` at 0).
fn grad(f: &mut dyn FnMut(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = f64::EPSILON.sqrt();
    let fx = f(x);
    let mut g = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        let hi = if x[i] > 0.0 {
            h
        } else if x[i] < 0.0 {
            -h
        } else {
            h
        };
        xp[i] += hi;
        g[i] = (f(&xp) - fx) / hi;
    }
    g
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Nocedal two-loop recursion: `d = -H_k g`.
fn two_loop(g: &[f64], s_hist: &[Vec<f64>], y_hist: &[Vec<f64>], rho_hist: &[f64]) -> Vec<f64> {
    let k = s_hist.len();
    let mut q = g.to_vec();
    let mut alpha = vec![0.0; k];
    for i in (0..k).rev() {
        alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
        for j in 0..q.len() {
            q[j] -= alpha[i] * y_hist[i][j];
        }
    }
    let gamma = if k > 0 {
        let (s, y) = (&s_hist[k - 1], &y_hist[k - 1]);
        let sy = dot(s, y);
        if sy > 0.0 {
            sy / dot(y, y)
        } else {
            1.0
        }
    } else {
        1.0
    };
    let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();
    for i in 0..k {
        let beta = rho_hist[i] * dot(&y_hist[i], &r);
        for j in 0..r.len() {
            r[j] += s_hist[i][j] * (alpha[i] - beta);
        }
    }
    r.iter().map(|&ri| -ri).collect()
}

/// Armijo backtracking line search. Returns `(step, grad_new, f_new)`.
fn line_search(
    f: &mut dyn FnMut(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],
    f0: f64,
    g0: &[f64],
) -> (f64, Vec<f64>, f64) {
    let c1 = 1e-4;
    let slope = dot(g0, d);
    let mut step = 1.0;
    while step > 1e-16 {
        let xn: Vec<f64> = x.iter().zip(d).map(|(&xi, &di)| xi + step * di).collect();
        let fnv = f(&xn);
        if fnv <= f0 + c1 * step * slope {
            let gn = grad(f, &xn);
            return (step, gn, fnv);
        }
        step *= 0.5;
    }
    (0.0, g0.to_vec(), f0)
}

/// Limited-memory BFGS minimization of an unconstrained smooth function.
///
/// Returns `(x, f(x))` at the last accepted point.
pub fn lbfgsb(
    f: &mut dyn FnMut(&[f64]) -> f64,
    x0: &[f64],
    maxiter: usize,
    gtol: f64,
    ftol: f64,
) -> (Vec<f64>, f64) {
    let m = 10; // scipy `maxcor`
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(f, &x);
    let mut fval = f(&x);
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for _ in 0..maxiter {
        let ginf = g.iter().fold(0.0_f64, |a, &gi| a.max(gi.abs()));
        if ginf <= gtol * (1.0_f64).max(fval.abs()) {
            break;
        }
        let d = two_loop(&g, &s_hist, &y_hist, &rho_hist);
        let (step, g_new, f_new) = line_search(f, &x, &d, fval, &g);
        if step == 0.0 {
            break;
        }
        let s: Vec<f64> = d.iter().map(|&di| di * step).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(gn, go)| gn - go).collect();
        let sy = dot(&s, &y);
        if sy > 0.0 {
            let rho = 1.0 / sy;
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(rho);
        }
        for i in 0..n {
            x[i] += d[i] * step;
        }
        g = g_new;
        // ftol: stop on a negligible relative function decrease.
        let f_prev = fval;
        fval = f_new;
        if (f_prev - fval).abs() <= ftol * (1.0_f64).max(f_prev.abs()).max(fval.abs()) {
            break;
        }
    }
    (x, fval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_recovers_minimum() {
        // f(x) = (x0-3)^2 + 2*(x1+1)^2 -> min at (3, -1).
        let mut f = |x: &[f64]| (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
        let (x, fv) = lbfgsb(&mut f, &[0.0, 0.0], 15000, 1e-5, 2.22e-9);
        assert!((x[0] - 3.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!((x[1] + 1.0).abs() < 1e-3, "x1 = {}", x[1]);
        assert!(fv < 1e-6, "f = {fv}");
    }

    #[test]
    fn grad_matches_analytic() {
        // f(x) = x0^2 + x1^2 -> grad = (2x0, 2x1).
        let mut f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let g = grad(&mut f, &[1.0, -2.0]);
        assert!((g[0] - 2.0).abs() < 1e-5, "g0 = {}", g[0]);
        assert!((g[1] + 4.0).abs() < 1e-5, "g1 = {}", g[1]);
    }
}
