"""Unit tests for the IPOPT least-squares (min ||g||^2) cruise formulation.

`PiecewiseLeastSquaresNLP` poses the terminal-state miss as the OBJECTIVE
(m=0, bounds-only) instead of a hard equality constraint, so a cold solve
descends ||g||^2 monotonically (no restoration thrashing). These tests use a tiny
ANALYTIC linear defect (no propagation, no SPICE) so they run in the fast loop.
"""
import numpy as np
import pytest

from reflectors.cruise_piecewise import smoothness_gradient, smoothness_objective
from reflectors.cruise_solve import (
    PiecewiseLeastSquaresNLP,
    _parallel_central_fd_jacobian,
    solve_piecewise_ipopt,
)


def _serial_map(func, pts):
    """Order-preserving serial stand-in for CloudpickleMap (the analytic defect
    is trivially picklable; no need to fork a pool in a unit test)."""
    return [func(p) for p in pts]


def _toy(N=3, seed=0):
    """A linear defect g(x) = A x - b with a known interior minimiser x_true
    (||g||^2 = 0 there). n_vars = 2N+1 (the trailing var is the 'duration')."""
    n_vars = 2 * N + 1
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, n_vars))
    x_true = 0.1 * rng.standard_normal(n_vars)
    b = A @ x_true

    def defect(x):
        return A @ np.asarray(x, dtype=float) - b

    return defect, A, b, x_true, N, n_vars


def _nlp(defect, N, n_vars, **es):
    return PiecewiseLeastSquaresNLP(defect, n_vars, N, _serial_map, **es)


def test_fd_jacobian_recovers_linear_map():
    """For g = A x - b, the parallel central-FD Jacobian recovers A exactly."""
    defect, A, b, x_true, N, n_vars = _toy()
    J = _parallel_central_fd_jacobian(defect, np.zeros(n_vars), n_vars, 1e-3, _serial_map)
    assert np.allclose(J, A, atol=1e-7)


def test_objective_is_squared_norm_plus_smoothness():
    defect, A, b, x_true, N, n_vars = _toy()
    x = 0.3 * np.arange(n_vars, dtype=float)
    g = defect(x)
    # pure ||g||^2
    nlp0 = _nlp(defect, N, n_vars, w_smooth=0.0)
    assert nlp0.objective(x) == pytest.approx(float(g @ g))
    # with smoothness
    nlp1 = _nlp(defect, N, n_vars, w_smooth=0.5)
    assert nlp1.objective(x) == pytest.approx(
        float(g @ g) + smoothness_objective(x, N, w_smooth=0.5)
    )


@pytest.mark.parametrize("w_smooth", [0.0, 0.7])
def test_gradient_matches_finite_difference_of_objective(w_smooth):
    """gradient(x) = 2 J^T g (+ smoothness grad) must match a scalar FD of the
    objective."""
    defect, A, b, x_true, N, n_vars = _toy(seed=2)
    nlp = _nlp(defect, N, n_vars, w_smooth=w_smooth)
    x = 0.2 * np.arange(n_vars, dtype=float) - 0.5
    grad = nlp.gradient(x)
    # analytic check: 2 A^T g (+ smoothness grad)
    g = defect(x)
    analytic = 2.0 * (A.T @ g)
    if w_smooth:
        analytic = analytic + smoothness_gradient(x, N, w_smooth=w_smooth)
    assert np.allclose(grad, analytic, atol=1e-7)
    # FD-of-objective check
    h = 1e-6
    fd = np.empty(n_vars)
    for i in range(n_vars):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fd[i] = (nlp.objective(xp) - nlp.objective(xm)) / (2 * h)
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-6)


def _call(nlp, it, obj):
    # intermediate(self, alg_mod, it, obj, inf_pr, inf_du, mu, dn, rg, a_du, a_pr, ls)
    return nlp.intermediate(0, it, float(obj), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def test_early_stop_keys_on_objective():
    defect, A, b, x_true, N, n_vars = _toy()
    # default off
    nlp_off = _nlp(defect, N, n_vars)
    assert nlp_off.early_stop_patience is None
    for it in range(40):
        assert _call(nlp_off, it, 0.3) is None
    # flat objective plateau -> abort after warm-up + patience
    nlp = _nlp(defect, N, n_vars, early_stop_patience=5, early_stop_rtol=0.02,
               early_stop_min_iters=10)
    res = [_call(nlp, it, 0.3) for it in range(11)]
    assert all(r is None for r in res[:10])
    assert res[10] is False
    # descending objective never aborts
    nlp2 = _nlp(defect, N, n_vars, early_stop_patience=5, early_stop_rtol=0.02,
                early_stop_min_iters=10)
    for it in range(30):
        assert _call(nlp2, it, 0.3 * (0.5 ** it)) is None


@pytest.mark.parametrize("scaling", [None, "none"])
def test_m0_least_squares_solve_reaches_zero_residual(scaling):
    """A tiny m=0 least-squares solve converges to the known minimiser, confirming
    cyipopt accepts m=0 and the formulation='least_squares' wiring runs end-to-end."""
    pytest.importorskip("cyipopt")
    defect, A, b, x_true, N, n_vars = _toy(seed=5)
    lb = [-1.0] * n_vars
    ub = [1.0] * n_vars
    x0 = np.zeros(n_vars)
    xb, info = solve_piecewise_ipopt(
        defect, n_vars, N, lb=lb, ub=ub, x0=x0, cp_map=_serial_map,
        w_smooth=0.0, max_iter=200, tol=1e-10, print_level=0,
        formulation="least_squares", nlp_scaling_method=scaling,
    )
    assert info["best_defect_norm"] < 1e-4
    assert float(np.linalg.norm(defect(xb))) < 1e-4
