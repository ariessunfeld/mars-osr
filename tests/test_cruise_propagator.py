"""Fast tests for the deterministic fixed-step RK4 cruise propagator
(``reflectors.cruise_propagator``).

Synthetic Sun-central circular problem (no real endpoint files needed). Covers:
  - two-body conservation (coast),
  - SRP positive signal (the command matters),
  - SMOOTHNESS of g(x) in both an angle and the duration D (no isolated spikes -- the
    grid is command-independent by construction),
  - EQUIVALENCE cross-check: fourth-order self-convergence plus comparison with the
    escape-based propagator at a fine common step (same ODE -> agree to ~machine eps),
  - defect 6-vector == terminal-miss norms; picklability (fixed + var-time, callable
    target).
"""

from __future__ import annotations

import math

import cloudpickle
import numpy as np
import pytest

from reflectors.central_body import sun_central_body
from reflectors.cruise_command import heliocentric_orbit_normal
from reflectors.cruise_piecewise import propagate_cruise_piecewise
from reflectors.cruise_propagator import (
    clean_cruise_terminal_miss,
    ideal_sail_achar_kmps2,
    make_clean_cruise_defect,
    make_clean_cruise_vartime_defect,
    propagate_cruise_clean,
)
from reflectors.dynamics import sun_gm_km3_per_s2
from reflectors.ephemeris import body_state
from reflectors.kernels import load_kernels
from reflectors.sail_designs import make_canonical_sail

load_kernels()

AU_KM = 1.495978707e8


def _problem():
    mu = sun_gm_km3_per_s2()
    z0 = np.array([AU_KM, 0.0, 0.0, 0.0, math.sqrt(mu / AU_KM), 0.0])
    ref = heliocentric_orbit_normal(z0)  # = +z for this in-ecliptic circular state
    sail = make_canonical_sail(0.018)
    cb = sun_central_body()
    return mu, z0, ref, sail, cb


def _spike_ratio(samples: np.ndarray) -> float:
    """max|d2| / median|d2| over the dominant column of an (M, k) sweep. A smooth,
    slowly-curving function gives O(1); an isolated grid-jump spike gives >> 1."""
    rng = samples.max(axis=0) - samples.min(axis=0)
    col = samples[:, int(np.argmax(rng))]
    d2 = np.abs(col[2:] - 2.0 * col[1:-1] + col[:-2])
    med = float(np.median(d2))
    return float(np.max(d2) / med) if med > 0 else float("inf")


# ---------------------------------------------------------------------------
# Physics: conservation + SRP signal
# ---------------------------------------------------------------------------


def test_clean_two_body_conservation_on_coast():
    """Anti-sunward sail (phi=pi -> n.s_hat<0 -> one-sided McInnes returns zero) leaves a
    pure two-body coast; specific energy and |h| are conserved to RK4 round-off."""
    mu, z0, ref, sail, cb = _problem()
    N = 8
    phis = np.full(N, math.pi)
    thetas = np.zeros(N)
    T_s = 100.0 * 86400.0
    zT = propagate_cruise_clean(phis, thetas, z0, 0.0, T_s, ref, sail, cb, (),
                                max_step_s=3600.0)
    eps0 = 0.5 * np.dot(z0[3:], z0[3:]) - mu / np.linalg.norm(z0[:3])
    epsT = 0.5 * np.dot(zT[3:], zT[3:]) - mu / np.linalg.norm(zT[:3])
    h0 = np.linalg.norm(np.cross(z0[:3], z0[3:]))
    hT = np.linalg.norm(np.cross(zT[:3], zT[3:]))
    assert abs(epsT - eps0) / abs(eps0) < 1e-10
    assert abs(hT - h0) / h0 < 1e-10


def test_clean_srp_positive_signal():
    """Two different commands (phi=+0.3 vs -0.3) drive the SRP and diverge -> the force is
    wired and the steering angle matters (a no-SRP coast cannot reach this)."""
    mu, z0, ref, sail, cb = _problem()
    N = 8
    thetas = np.zeros(N)
    T_s = 30.0 * 86400.0
    zA = propagate_cruise_clean(np.full(N, 0.3), thetas, z0, 0.0, T_s, ref, sail, cb, (),
                                max_step_s=3600.0)
    zB = propagate_cruise_clean(np.full(N, -0.3), thetas, z0, 0.0, T_s, ref, sail, cb, (),
                                max_step_s=3600.0)
    assert np.linalg.norm(zA[:3] - zB[:3]) > 1e3


# ---------------------------------------------------------------------------
# Smoothness invariant: g(x) has no isolated spike versus an angle or duration
# ---------------------------------------------------------------------------


def test_clean_defect_smooth_in_angle():
    """g(phi_0) over a fine sweep is smooth -- the fixed-step grid is command-independent,
    so the terminal map carries no step-count discontinuity in the angles."""
    mu, z0, ref, sail, cb = _problem()
    N = 16
    vsc = math.sqrt(mu / AU_KM)
    T_s = 200.0 * 86400.0
    z_tgt = propagate_cruise_clean(np.full(N, 0.1), np.zeros(N), z0, 0.0, T_s, ref, sail,
                                   cb, (), max_step_s=72000.0)
    defect = make_clean_cruise_defect(z0, z_tgt, 0.0, T_s, ref, sail, cb, (), N=N,
                                      r_scale_km=AU_KM, v_scale_kmps=vsc, max_step_s=72000.0)
    phis0 = np.linspace(-0.5, 0.5, 201)
    g = np.empty((phis0.size, 6))
    for k, p0 in enumerate(phis0):
        x = np.zeros(2 * N)
        x[0] = p0
        g[k] = defect(x)
    assert _spike_ratio(g) < 4.0  # reference case is approximately 1.66


def test_clean_terminal_state_smooth_in_duration():
    """z_T(D) over a fine sweep is smooth -- the uniform-sub-step-per-segment scheme makes
    the step count constant and h continuous in D (the escape path's terminal-remainder
    step-count jump is the variable-D non-smoothness this propagator removes)."""
    mu, z0, ref, sail, cb = _problem()
    N = 16
    phis = np.full(N, 0.2)
    thetas = np.full(N, 0.05)
    Ds = np.linspace(595.0, 605.0, 201)
    zT = np.empty((Ds.size, 3))
    for k, D in enumerate(Ds):
        zT[k] = propagate_cruise_clean(phis, thetas, z0, 0.0, D * 86400.0, ref, sail, cb,
                                       (), max_step_s=72000.0)[:3]
    assert _spike_ratio(zT) < 4.0  # reference case is approximately 1.25


# ---------------------------------------------------------------------------
# Equivalence cross-check (independent-implementation validation)
# ---------------------------------------------------------------------------


def test_clean_self_convergence_is_fourth_order():
    """Halving the step shrinks the terminal error ~16x -> RK4 is 4th-order accurate here
    (the integrator converges; a coding error in the step would break the order)."""
    mu, z0, ref, sail, cb = _problem()
    N = 16
    phis = np.full(N, 0.2)
    thetas = np.full(N, 0.05)
    T_s = 60.0 * 86400.0
    h = 21600.0
    z1 = propagate_cruise_clean(phis, thetas, z0, 0.0, T_s, ref, sail, cb, (), max_step_s=h)
    z2 = propagate_cruise_clean(phis, thetas, z0, 0.0, T_s, ref, sail, cb, (), max_step_s=h / 2)
    z4 = propagate_cruise_clean(phis, thetas, z0, 0.0, T_s, ref, sail, cb, (), max_step_s=h / 4)
    d12 = np.linalg.norm(z1[:3] - z2[:3])
    d24 = np.linalg.norm(z2[:3] - z4[:3])
    assert 8.0 < d12 / d24 < 32.0


def test_clean_matches_escape_based_propagator_at_fine_step():
    """Fixed-step and escape-based propagators integrate the same ODE.

    The physics terms are reused directly, so at a fine common step the
    propagators agree to approximately machine precision. An incorrect sign or
    missing term would produce kilometre-scale divergence over 60 days.
    """
    mu, z0, ref, sail, cb = _problem()
    N = 16
    phis = np.full(N, 0.2)
    thetas = np.full(N, 0.05)
    T_s = 60.0 * 86400.0
    hs = 5400.0
    z_clean = propagate_cruise_clean(phis, thetas, z0, 0.0, T_s, ref, sail, cb, (),
                                     max_step_s=hs)
    run = propagate_cruise_piecewise(phis, thetas, z0, 0.0, T_s, ref, sail, cb, (),
                                     max_step_s=hs, steps_per_orbit=2000)
    z_esc = np.asarray(run.orbit_state_km_kmps[-1], float)
    assert np.linalg.norm(z_clean[:3] - z_esc[:3]) < 1e-2
    assert np.linalg.norm(z_clean[3:6] - z_esc[3:6]) < 1e-6


# ---------------------------------------------------------------------------
# Defect contract + picklability
# ---------------------------------------------------------------------------


def test_clean_defect_vector_matches_terminal_miss_norms():
    mu, z0, ref, sail, cb = _problem()
    N = 8
    vsc = math.sqrt(mu / AU_KM)
    T_s = 120.0 * 86400.0
    phis = np.full(N, 0.2)
    thetas = np.full(N, 0.05)
    # target = some offset state so the miss is nonzero
    z_tgt = z0 + np.array([1.0e6, -2.0e6, 5.0e5, 0.1, -0.05, 0.02])
    defect = make_clean_cruise_defect(z0, z_tgt, 0.0, T_s, ref, sail, cb, (), N=N,
                                      r_scale_km=AU_KM, v_scale_kmps=vsc, max_step_s=3600.0)
    x = np.concatenate([phis, thetas])
    g = defect(x)
    r_miss, v_miss, _ = clean_cruise_terminal_miss(phis, thetas, z0, z_tgt, 0.0, T_s, ref,
                                                   sail, cb, (), max_step_s=3600.0)
    assert np.linalg.norm(g[:3]) * AU_KM == pytest.approx(r_miss, rel=1e-9)
    assert np.linalg.norm(g[3:]) * vsc == pytest.approx(v_miss, rel=1e-9)


def test_clean_defect_is_cloudpicklable():
    mu, z0, ref, sail, cb = _problem()
    N = 8
    vsc = math.sqrt(mu / AU_KM)
    z_tgt = z0 + np.array([1.0e6, 0.0, 0.0, 0.0, 0.1, 0.0])
    defect = make_clean_cruise_defect(z0, z_tgt, 0.0, 120.0 * 86400.0, ref, sail, cb, (),
                                      N=N, r_scale_km=AU_KM, v_scale_kmps=vsc, max_step_s=3600.0)
    x = np.concatenate([np.full(N, 0.1), np.zeros(N)])
    g0 = defect(x)
    g1 = cloudpickle.loads(cloudpickle.dumps(defect))(x)
    assert np.array_equal(g0, g1)


def test_clean_vartime_defect_pulls_moving_target_and_pickles():
    """The var-time defect (x=[phis,thetas,D]) pulls the target at dep_et+D*86400 via the
    injected callable; D<=0 returns the finite penalty; the closure round-trips through
    cloudpickle (the parallel-FD Jacobian requirement)."""
    mu, z0, ref, sail, cb = _problem()
    N = 8
    vsc = math.sqrt(mu / AU_KM)
    from reflectors.ephemeris import utc_to_et
    dep_et = utc_to_et("2011-10-06T00:00:00")

    def target_fn(et):
        return body_state("MARS BARYCENTER", et, observer="SUN")[0]

    defect = make_clean_cruise_vartime_defect(
        z0, dep_et, ref, sail, cb, (), N=N, r_scale_km=AU_KM, v_scale_kmps=vsc,
        max_step_s=3600.0, target_state_fn=target_fn,
    )
    x = np.concatenate([np.full(N, 0.1), np.zeros(N), [300.0]])
    g = defect(x)
    assert g.shape == (6,) and np.all(np.isfinite(g))
    # D<=0 -> finite penalty, not a crash
    xbad = np.concatenate([np.full(N, 0.1), np.zeros(N), [-5.0]])
    assert np.array_equal(defect(xbad), np.full(6, 1.0e3))
    # cloudpickle round-trip (body_state closure)
    g1 = cloudpickle.loads(cloudpickle.dumps(defect))(x)
    assert np.array_equal(g, g1)


def test_clean_vartime_defect_ideal_sail_matches_direct_and_differs_from_mcinnes():
    """The var-time defect threads ``ideal_achar_kmps2`` to the propagator: built with the
    IDEAL cos^2 sail it equals a direct ``propagate_cruise_clean(ideal)`` defect BYTE-FOR-BYTE
    (plumbing correctness), and DIFFERS from the McInnes default at the same ``x`` (positive
    signal that the ideal force is applied rather than omitted)."""
    from reflectors.ephemeris import utc_to_et

    mu, z0, ref, sail, cb = _problem()
    N = 8
    a_ideal = ideal_sail_achar_kmps2(sail)
    assert a_ideal > 0.0  # canonical sigma=18 sail -> ~5.04e-7 km/s^2
    r_scale, v_scale = AU_KM, math.sqrt(mu / AU_KM)
    max_step_s = 3600.0
    dep_et = utc_to_et("2011-10-06T00:00:00")

    def target_fn(et):
        return body_state("MARS BARYCENTER", et, observer="SUN")[0]

    x = np.concatenate([np.full(N, 0.2), np.full(N, 0.05), [300.0]])
    phis, thetas, D = x[:N], x[N : 2 * N], float(x[2 * N])
    T_s = D * 86400.0
    z_tgt = np.asarray(target_fn(dep_et + T_s), dtype=float)

    # Direct ideal-sail propagation + identical scaling -> the expected scaled defect.
    z_T_ideal = propagate_cruise_clean(
        phis, thetas, z0, dep_et, T_s, ref, sail, cb, (),
        max_step_s=max_step_s, ideal_achar_kmps2=a_ideal,
    )
    g_expected = np.empty(6)
    g_expected[:3] = (z_T_ideal[:3] - z_tgt[:3]) / r_scale
    g_expected[3:] = (z_T_ideal[3:6] - z_tgt[3:6]) / v_scale

    defect_ideal = make_clean_cruise_vartime_defect(
        z0, dep_et, ref, sail, cb, (), N=N, r_scale_km=r_scale, v_scale_kmps=v_scale,
        max_step_s=max_step_s, target_state_fn=target_fn, ideal_achar_kmps2=a_ideal,
    )
    assert np.array_equal(defect_ideal(x), g_expected)  # byte-identical plumbing

    # McInnes default (ideal_achar None) must differ -> the ideal force is genuinely applied.
    defect_mcinnes = make_clean_cruise_vartime_defect(
        z0, dep_et, ref, sail, cb, (), N=N, r_scale_km=r_scale, v_scale_kmps=v_scale,
        max_step_s=max_step_s, target_state_fn=target_fn,
    )
    assert np.linalg.norm(defect_ideal(x) - defect_mcinnes(x)) > 1e-6
