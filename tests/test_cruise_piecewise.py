"""Tests for the piecewise-constant RTN tilt-angle cruise command + cost
(reflectors.cruise_piecewise).

Covers steerer geometry, angle recovery, segment boundaries, the scaled defect
vector, and serialization on a synthetic Sun-centred problem without external
endpoint files.
"""

from __future__ import annotations

import inspect
import math

import cloudpickle
import numpy as np
import pytest

from reflectors.attitude_control import AttitudeLimits
from reflectors.central_body import sun_central_body
from reflectors.cruise_command import heliocentric_orbit_normal
from reflectors.cruise_piecewise import (
    assert_piecewise_slew_feasible,
    make_piecewise_cruise_defect,
    make_piecewise_rtn_steerer,
    pack_angles,
    piecewise_cruise_terminal_miss,
    propagate_cruise_piecewise,
    recover_phi_theta,
    rtn_sail_normal,
    segment_index,
    smoothness_gradient,
    smoothness_objective,
    unpack_angles,
)
from reflectors.dynamics import sun_gm_km3_per_s2
from reflectors.sail_designs import make_canonical_sail
from reflectors.third_body import earth_third_body, mars_third_body

AU_KM = 1.495978707e8


def _problem():
    mu = sun_gm_km3_per_s2()
    z0 = np.array([AU_KM, 0.0, 0.0, 0.0, math.sqrt(mu / AU_KM), 0.0])
    ref = heliocentric_orbit_normal(z0)  # = +z for this in-ecliptic circular state
    sail = make_canonical_sail(0.018)
    cb = sun_central_body()
    tb = (earth_third_body(), mars_third_body())
    return mu, z0, ref, sail, cb, tb


def _eval_steerer(phis, thetas, ref_normal, s_hat, et, et0=0.0, T_s=1.0):
    fn = make_piecewise_rtn_steerer(phis, thetas, et0, T_s, ref_normal)
    sail = make_canonical_sail(0.018)
    return np.asarray(fn(None, None, s_hat, 1361.0, sail, None, et=et))


# ---------------------------------------------------------------------------
# Packing + segment indexing
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip():
    phis = np.array([0.1, -0.2, 0.3, -0.4])
    thetas = np.array([0.05, 0.06, -0.07, 0.08])
    x = pack_angles(phis, thetas)
    assert x.shape == (8,)
    p2, t2 = unpack_angles(x, 4)
    assert np.array_equal(p2, phis) and np.array_equal(t2, thetas)


def test_pack_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pack_angles(np.zeros(3), np.zeros(4))


def test_segment_index_boundaries():
    N = 4
    # tau in [i/N, (i+1)/N) -> segment i.
    assert segment_index(0.0, N) == 0
    assert segment_index(0.24, N) == 0
    assert segment_index(0.25, N) == 1
    assert segment_index(0.5, N) == 2
    assert segment_index(0.99, N) == 3
    # Required upper clamp: tau == 1.0 (final RHS) -> last segment, not N.
    assert segment_index(1.0, N) == N - 1
    # Out-of-range clamps.
    assert segment_index(-0.1, N) == 0
    assert segment_index(1.5, N) == N - 1


# ---------------------------------------------------------------------------
# n_des geometry
# ---------------------------------------------------------------------------


def test_n_des_is_unit_for_random_inputs():
    rng = np.random.default_rng(0)
    ref = np.array([0.1, -0.2, 1.0])
    N = 4
    for _ in range(20):
        phis = rng.uniform(-0.6, 0.6, size=N)
        thetas = rng.uniform(-0.6, 0.6, size=N)
        s = rng.normal(size=3)
        s = s / np.linalg.norm(s)
        tau = float(rng.uniform(0, 1))
        n = _eval_steerer(phis, thetas, ref, s, tau)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)


def test_rtn_sail_normal_matches_steerer():
    """The single-source helper rtn_sail_normal reproduces the steerer's n_des
    bit-for-bit using the same arithmetic. The steerer delegates its geometry
    to the helper, which ``cruise_propagator`` also uses."""
    rng = np.random.default_rng(1)
    ref = np.array([0.1, -0.2, 1.0])
    ref_hat = ref / np.linalg.norm(ref)
    for _ in range(50):
        phi = float(rng.uniform(-0.785, 0.785))   # +/-45 deg box
        theta = float(rng.uniform(-0.785, 0.785))
        s = rng.normal(size=3)
        s = s / np.linalg.norm(s)
        # The steerer re-normalises s_hat internally as s / float(norm(s)); feed the
        # helper that exact s_u so the comparison isolates the geometry (bit-exact).
        s_u = s / float(np.linalg.norm(s))
        # Single-segment steerer evaluated at tau=0.5 -> segment 0 -> (phi, theta).
        n_steer = _eval_steerer(np.array([phi]), np.array([theta]), ref, s, 0.5)
        n_helper = rtn_sail_normal(s_u, ref_hat, phi, theta)
        assert np.array_equal(n_steer, n_helper)


def test_rtn_sail_normal_degenerate_fallback():
    """s_hat ~parallel to k_hat triggers the deterministic perpendicular fallback;
    the result stays a finite unit vector (never reached on a real heliocentric arc,
    pinned as an independent safeguard)."""
    k_hat = np.array([0.0, 0.0, 1.0])
    s = np.array([0.0, 0.0, 1.0])  # exactly parallel -> k x s == 0
    n = rtn_sail_normal(s, k_hat, 0.2, 0.1)
    assert np.all(np.isfinite(n))
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)


def test_recover_phi_theta_round_trips_in_plane():
    """recover_phi_theta inverts the construction exactly when r_hat is
    perpendicular to ref_normal (the departure / in-plane geometry)."""
    rng = np.random.default_rng(1)
    ref = np.array([0.0, 0.0, 1.0])
    # r_hat = -s_hat = +x, perpendicular to ref (+z).
    s = np.array([-1.0, 0.0, 0.0])
    N = 5
    phis = rng.uniform(-0.6, 0.6, size=N)
    thetas = rng.uniform(-0.6, 0.6, size=N)
    for i in range(N):
        et = (i + 0.5) / N  # tau lands squarely in segment i (et0=0, T_s=1)
        n = _eval_steerer(phis, thetas, ref, s, et)
        phi_r, theta_r = recover_phi_theta(n, s, ref)
        assert phi_r == pytest.approx(phis[i], abs=1e-12)
        assert theta_r == pytest.approx(thetas[i], abs=1e-12)


def test_piecewise_constant_within_segment():
    ref = np.array([0.0, 0.0, 1.0])
    s = np.array([-1.0, 0.0, 0.0])
    phis = np.array([0.1, 0.4, -0.2, 0.3])
    thetas = np.array([0.0, 0.1, -0.1, 0.2])
    # Two taus inside segment 1 -> identical n_des (same s_hat).
    n_a = _eval_steerer(phis, thetas, ref, s, 0.26)
    n_b = _eval_steerer(phis, thetas, ref, s, 0.49)
    assert np.allclose(n_a, n_b, atol=1e-14)
    # A tau in segment 2 differs.
    n_c = _eval_steerer(phis, thetas, ref, s, 0.6)
    assert not np.allclose(n_a, n_c, atol=1e-6)


def test_steering_fn_exposes_et_keyword():
    fn = make_piecewise_rtn_steerer(np.zeros(3), np.zeros(3), 0.0, 1.0, np.array([0, 0, 1.0]))
    assert "et" in inspect.signature(fn).parameters


def test_k_hat_is_ref_normal_not_ecliptic_z():
    """The out-of-plane axis is ref_normal, not a hard-coded J2000-z. Build with a
    ref_normal tilted off +z; with theta != 0 the OOP component of n_des must lie
    along ref_normal (n_des . ref_hat == sin theta), NOT along +z."""
    ref = np.array([0.3, -0.4, 1.0])
    ref_hat = ref / np.linalg.norm(ref)
    # r_hat perpendicular to ref: project +x off ref_hat.
    a = np.array([1.0, 0.0, 0.0])
    r_hat = a - np.dot(a, ref_hat) * ref_hat
    r_hat /= np.linalg.norm(r_hat)
    s = -r_hat
    theta = 0.3
    n = _eval_steerer(np.array([0.0]), np.array([theta]), ref, s, 0.5)
    assert float(np.dot(n, ref_hat)) == pytest.approx(math.sin(theta), abs=1e-12)
    # If k_hat were wrongly +z, n.z would equal sin(theta); it must not.
    assert float(n[2]) != pytest.approx(math.sin(theta), abs=1e-6)


def test_steerer_rejects_bad_inputs():
    with pytest.raises(ValueError):
        make_piecewise_rtn_steerer(np.zeros(3), np.zeros(2), 0.0, 1.0, np.array([0, 0, 1.0]))
    with pytest.raises(ValueError):
        make_piecewise_rtn_steerer(np.zeros(3), np.zeros(3), 0.0, 0.0, np.array([0, 0, 1.0]))
    with pytest.raises(ValueError):
        make_piecewise_rtn_steerer(np.zeros(3), np.zeros(3), 0.0, 1.0, np.zeros(3))


# ---------------------------------------------------------------------------
# Propagation + defect (mirror test_cruise_cost)
# ---------------------------------------------------------------------------


def test_propagate_piecewise_runs_to_t_final():
    mu, z0, ref, sail, cb, tb = _problem()
    T_s = 30.0 * 86400.0
    N = 4
    phis = np.full(N, 0.2)
    thetas = np.zeros(N)
    run = propagate_cruise_piecewise(phis, thetas, z0, 0.0, T_s, ref, sail, cb, tb)
    assert run.termination_reason == "t_final"
    assert np.all(np.isfinite(run.orbit_state_km_kmps))
    assert np.linalg.norm(run.orbit_state_km_kmps[-1, :3] - z0[:3]) > 1e6


def test_kinematic_command_drives_srp():
    """Positive signal: two DIFFERENT piecewise commands produce different
    terminal states -- only possible if the steered attitude reaches the SRP
    force (without SRP, attitude is irrelevant and both would be the identical
    two-body arc)."""
    mu, z0, ref, sail, cb, tb = _problem()
    T_s = 30.0 * 86400.0
    N = 4
    rA, vA, zA = piecewise_cruise_terminal_miss(
        np.full(N, 0.2), np.zeros(N), z0, z0, 0.0, T_s, ref, sail, cb, tb
    )
    rB, vB, zB = piecewise_cruise_terminal_miss(
        np.full(N, -0.2), np.zeros(N), z0, z0, 0.0, T_s, ref, sail, cb, tb
    )
    assert np.linalg.norm(zA[:3] - zB[:3]) > 1.0e3  # the command matters via SRP


def test_piecewise_defect_vector_matches_terminal_miss_norms():
    mu, z0, ref, sail, cb, tb = _problem()
    T_s = 30.0 * 86400.0
    N = 4
    x = pack_angles(np.full(N, 0.2), np.full(N, 0.05))
    z_tgt = z0 + np.array([3.0e6, -2.0e6, 1.0e6, 0.1, -0.2, 0.05])
    r_scale, v_scale = AU_KM, 29.78
    defect = make_piecewise_cruise_defect(
        z0, z_tgt, 0.0, T_s, ref, sail, cb, tb,
        N=N, r_scale_km=r_scale, v_scale_kmps=v_scale,
    )
    g = defect(x)
    assert g.shape == (6,)
    phis, thetas = unpack_angles(x, N)
    r_miss, v_miss, _ = piecewise_cruise_terminal_miss(
        phis, thetas, z0, z_tgt, 0.0, T_s, ref, sail, cb, tb
    )
    assert np.linalg.norm(g[:3]) * r_scale == pytest.approx(r_miss, rel=1e-9)
    assert np.linalg.norm(g[3:]) * v_scale == pytest.approx(v_miss, rel=1e-9)


def test_piecewise_defect_is_cloudpicklable():
    mu, z0, ref, sail, cb, tb = _problem()
    N = 4
    defect = make_piecewise_cruise_defect(
        z0, z0.copy(), 0.0, 20.0 * 86400.0, ref, sail, cb, tb,
        N=N, r_scale_km=AU_KM, v_scale_kmps=29.78,
    )
    x = pack_angles(np.full(N, 0.1), np.zeros(N))
    g0 = defect(x)
    defect2 = cloudpickle.loads(cloudpickle.dumps(defect))
    assert np.array_equal(defect2(x), g0)


# ---------------------------------------------------------------------------
# Smoothness regulariser
# ---------------------------------------------------------------------------


def test_smoothness_objective_zero_for_constant_command():
    N = 6
    x = pack_angles(np.full(N, 0.3), np.full(N, -0.1))
    assert smoothness_objective(x, N) == 0.0


def test_smoothness_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    N = 5
    x = np.concatenate([rng.uniform(-0.6, 0.6, size=2 * N), [540.0]])  # +trailing D
    g = smoothness_gradient(x, N)
    assert g[-1] == 0.0  # the trailing D is not regularised
    h = 1e-6
    for i in range(2 * N):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fd = (smoothness_objective(xp, N) - smoothness_objective(xm, N)) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-12)


# ---------------------------------------------------------------------------
# Slew feasibility (post-hoc)
# ---------------------------------------------------------------------------


def test_piecewise_slew_trivially_feasible_over_transit():
    N = 16
    rng = np.random.default_rng(4)
    phis = rng.uniform(-math.radians(35), math.radians(35), size=N)
    thetas = rng.uniform(-math.radians(35), math.radians(35), size=N)
    T_s = 540.0 * 86400.0
    limits = AttitudeLimits()
    worst, budget = assert_piecewise_slew_feasible(phis, thetas, T_s, limits, N)
    assert worst < budget
    assert budget > 1e3  # enormous margin over a multi-month segment


def test_piecewise_slew_raises_when_overcranked():
    # A large per-node jump over a tiny segment is infeasible.
    phis = np.array([0.0, 1.0, 0.0])
    thetas = np.zeros(3)
    limits = AttitudeLimits()
    with pytest.raises(ValueError):
        assert_piecewise_slew_feasible(phis, thetas, T_s=1.0, limits=limits, N=3)
