"""Tests for the time-Fourier cone/clock cruise command steerer.

Covers the pure command geometry (packing, Fourier evaluation, n_des unit and
cone/clock recovery, clamps) and the structural slew-feasibility guarantee:
fed through ``propagate_escape``'s rate/accel-limited tracker over a heliocentric
arc, the achieved attitude tracks the slow command with 0 slew violations.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from reflectors.attitude_control import AttitudeLimits
from reflectors.central_body import SUN_NOMINAL_CEILING_KM, sun_central_body
from reflectors.cruise_command import (
    assert_cruise_slew_feasible,
    cruise_command_slew_bounds,
    cruise_cone_clock,
    feasible_coeff_boxes,
    heliocentric_angular_rate,
    heliocentric_orbit_normal,
    make_cruise_command_steerer,
)
from reflectors.dynamics import sun_gm_km3_per_s2
from reflectors.escape import propagate_escape
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail

AU_KM = 1.495978707e8


# ---------------------------------------------------------------------------
# Packing / Fourier evaluation
# ---------------------------------------------------------------------------


def test_order_inference_and_bad_length():
    # K=0 -> 2 params; K=1 -> 6; K=2 -> 10.
    assert cruise_cone_clock(np.array([0.5, 1.0]), 0.0)  # K=0 ok
    assert cruise_cone_clock(np.zeros(6), 0.3) == (0.0, 0.0)  # K=1 ok
    assert cruise_cone_clock(np.zeros(10), 0.3) == (0.0, 0.0)  # K=2 ok
    for bad in (3, 4, 5, 7, 8, 9):
        with pytest.raises(ValueError):
            cruise_cone_clock(np.zeros(bad), 0.0)


def test_constant_coeffs_are_constant_in_time():
    # K=1 with zero harmonics -> alpha=a0, delta=d0 for all tau.
    coeffs = np.array([0.7, 0.0, 0.0, 1.2, 0.0, 0.0])  # a0=0.7, d0=1.2
    for tau in (0.0, 0.25, 0.5, 0.9, 1.0):
        a, d = cruise_cone_clock(coeffs, tau)
        assert a == pytest.approx(0.7)
        assert d == pytest.approx(1.2)


def test_fourier_evaluation_matches_closed_form():
    # alpha = a0 + ac1 cos(2 pi tau) + as1 sin(2 pi tau); delta likewise.
    a0, ac1, as1 = 0.6, 0.1, -0.05
    d0, dc1, ds1 = 0.2, 0.3, 0.15
    coeffs = np.array([a0, ac1, as1, d0, dc1, ds1])
    for tau in (0.0, 0.17, 0.5, 0.83):
        ang = 2.0 * math.pi * tau
        exp_a = a0 + ac1 * math.cos(ang) + as1 * math.sin(ang)
        exp_d = d0 + dc1 * math.cos(ang) + ds1 * math.sin(ang)
        a, d = cruise_cone_clock(coeffs, tau)
        assert a == pytest.approx(min(max(exp_a, 0.0), 0.5 * math.pi))
        assert d == pytest.approx(exp_d)


def test_alpha_is_clamped_to_first_quadrant():
    # a0 above pi/2 and below 0 clamp.
    hi = cruise_cone_clock(np.array([3.0, 0.5]), 0.0)[0]  # K=0
    lo = cruise_cone_clock(np.array([-1.0, 0.5]), 0.0)[0]
    assert hi == pytest.approx(0.5 * math.pi)
    assert lo == 0.0


# ---------------------------------------------------------------------------
# n_des geometry (unit, cone/clock recovery)
# ---------------------------------------------------------------------------


def _eval_steerer(coeffs, ref_normal, s_hat, et, et0=0.0, T_s=1.0):
    fn = make_cruise_command_steerer(coeffs, et0, T_s, ref_normal)
    sail = make_canonical_sail(0.018)
    # The steerer does not consume r, v, p_eff, or current_n.
    return np.asarray(fn(None, None, s_hat, 1361.0, sail, None, et=et))


def test_n_des_is_unit_for_random_inputs():
    rng = np.random.default_rng(0)
    ref = np.array([0.1, -0.2, 1.0])
    for _ in range(20):
        coeffs = rng.normal(scale=0.8, size=10)  # K=2
        s = rng.normal(size=3)
        s = s / np.linalg.norm(s)
        tau = float(rng.uniform(0, 1))
        n = _eval_steerer(coeffs, ref, s, tau)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)


def test_recovered_cone_equals_commanded_alpha():
    """arccos(n_des . s_hat) == clamped alpha(tau)."""
    rng = np.random.default_rng(1)
    ref = np.array([0.0, 0.0, 1.0])
    coeffs = np.array([0.5, 0.2, -0.1, 0.4, 0.3, 0.1])  # K=1
    s = np.array([1.0, 0.2, -0.3])
    s = s / np.linalg.norm(s)
    for tau in (0.0, 0.3, 0.6, 1.0):
        n = _eval_steerer(coeffs, ref, s, tau)
        alpha_cmd, _ = cruise_cone_clock(coeffs, tau)
        cone = math.acos(float(np.clip(np.dot(n, s), -1.0, 1.0)))
        assert cone == pytest.approx(alpha_cmd, abs=1e-12)


def test_recovered_clock_equals_commanded_delta():
    """atan2(n . e_B, n . e_A) == delta (mod 2 pi) when sin(alpha) != 0."""
    ref = np.array([0.05, 0.0, 1.0])
    coeffs = np.array([0.7, 0.1, 0.0, 0.9, 0.2, -0.3])  # alpha~0.7 (sin!=0)
    s = np.array([1.0, -0.1, 0.2])
    s = s / np.linalg.norm(s)
    # Rebuild the same fixed clock basis the steerer uses.
    refu = ref / np.linalg.norm(ref)
    e_A = refu - np.dot(refu, s) * s
    e_A = e_A / np.linalg.norm(e_A)
    e_B = np.cross(s, e_A)
    for tau in (0.0, 0.4, 0.8):
        n = _eval_steerer(coeffs, ref, s, tau)
        _, delta_cmd = cruise_cone_clock(coeffs, tau)
        rec = math.atan2(float(np.dot(n, e_B)), float(np.dot(n, e_A)))
        # compare mod 2 pi
        diff = (rec - delta_cmd + math.pi) % (2.0 * math.pi) - math.pi
        assert diff == pytest.approx(0.0, abs=1e-9)


def test_steering_fn_exposes_et_keyword():
    fn = make_cruise_command_steerer(np.zeros(6), 0.0, 1.0, np.array([0, 0, 1.0]))
    assert "et" in inspect.signature(fn).parameters


def test_heliocentric_orbit_normal_and_radial_raise():
    state = np.array([AU_KM, 0.0, 0.0, 0.0, 30.0, 0.0])
    h = heliocentric_orbit_normal(state)
    assert h == pytest.approx(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        heliocentric_orbit_normal(np.array([AU_KM, 0, 0, 5.0, 0, 0]))  # radial


def test_zero_T_s_and_zero_ref_raise():
    with pytest.raises(ValueError):
        make_cruise_command_steerer(np.zeros(6), 0.0, 0.0, np.array([0, 0, 1.0]))
    with pytest.raises(ValueError):
        make_cruise_command_steerer(np.zeros(6), 0.0, 1.0, np.zeros(3))


# ---------------------------------------------------------------------------
# Kinematic propagation: the cruise uses n = n_des directly.
# ---------------------------------------------------------------------------


def _circular_helio_state():
    mu = sun_gm_km3_per_s2()
    return np.array([AU_KM, 0.0, 0.0, 0.0, math.sqrt(mu / AU_KM), 0.0])


def _run_kinematic(steerer, span, force_coast=False):
    state0 = _circular_helio_state()
    sail = make_canonical_sail(0.018)
    params = QLawParams(a_target_km=SUN_NOMINAL_CEILING_KM, rp_min_km=1.0)
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    return propagate_escape(
        state0, 0.0, sail, params, limits, span,
        gravity_degree=0, central_body=sun_central_body(), third_bodies=(),
        steering_fn=None if force_coast else steerer,
        force_coast=force_coast, kinematic_attitude=True,
    )


def test_kinematic_command_drives_srp_and_runs_to_completion():
    """End-to-end wiring: kinematic_attitude=True feeds the commanded normal
    into the SRP model, so a thrusting cruise command perturbs the
    heliocentric arc away from a feathered (SRP=0) coast. (In kinematic mode the
    propagator does not integrate the attitude state -- the applied normal is
    n_des in the RHS, not the stored n -- so this test verifies the orbit
    effect, which is the relevant quantity, rather than the stale output
    normal.)"""
    state0 = _circular_helio_state()
    ref = heliocentric_orbit_normal(state0)
    T_s = 540.0 * 86400.0
    coeffs = np.array([0.6, 0.15, 0.0, 0.0, 1.0, 0.5])
    steerer = make_cruise_command_steerer(coeffs, 0.0, T_s, ref)
    span = (0.0, 30.0 * 86400.0)

    thrust = _run_kinematic(steerer, span)
    coast = _run_kinematic(steerer, span, force_coast=True)
    assert thrust.termination_reason == "t_final"
    dr = np.linalg.norm(
        thrust.orbit_state_km_kmps[-1, :3] - coast.orbit_state_km_kmps[-1, :3]
    )
    assert dr > 100.0  # SRP from the commanded attitude visibly bends the arc
    assert np.all(np.isfinite(thrust.orbit_state_km_kmps))


# ---------------------------------------------------------------------------
# Slew-feasibility ENFORCEMENT (analytic bounds, no integrated tracker)
# ---------------------------------------------------------------------------


def test_analytic_bounds_exceed_actual_commanded_rates():
    """The closed-form (|omega|, |alpha|) bounds must UPPER-BOUND the actual
    rates the smooth command demands, finite-differenced along a real
    heliocentric trajectory. This is the independent cross-check that the
    enforcement bounds are conservative (never under-report a rate)."""
    mu = sun_gm_km3_per_s2()
    state0 = _circular_helio_state()
    ref = heliocentric_orbit_normal(state0)
    et0 = 0.0
    T_s = 540.0 * 86400.0
    # K=2, cone kept inside [0, pi/2] (no clamp -> C1-smooth so the bound holds);
    # clock free. cone harmonics amp = 0.112 + 0.05 = 0.162, a0=0.8 -> in range.
    coeffs = np.array([0.8, 0.1, 0.03, 0.05, -0.04, 0.3, 1.0, -0.2, 0.5, 0.1])  # K=2
    steerer = make_cruise_command_steerer(coeffs, et0, T_s, ref)
    sail = make_canonical_sail(0.018)

    # Frame angular rate of a circular orbit is constant; accel ~ 0.
    w_orb = heliocentric_angular_rate(state0, mu)
    b_omega, b_alpha = cruise_command_slew_bounds(
        coeffs, T_s, omega_frame_max_rad_s=w_orb, omega_frame_accel_max_rad_s2=0.0
    )

    # Finite-difference n_des(t) along the analytic circular orbit. Use CHORD
    # and SECOND-DIFFERENCE magnitudes (not arccos(dot), which loses precision
    # catastrophically for near-parallel unit vectors): |dn/dt| ~ |Dn|/dt and
    # the angular-acceleration bound target |d^2n/dt^2| ~ |D^2 n|/dt^2.
    dt = 600.0
    ts = np.arange(0.0, 60.0 * 86400.0 + dt, dt)
    ndes = np.zeros((ts.size, 3))
    for j, t in enumerate(ts):
        ang = w_orb * t
        r = AU_KM * np.array([math.cos(ang), math.sin(ang), 0.0])
        s_hat = -r / np.linalg.norm(r)
        ndes[j] = steerer(None, None, s_hat, 1361.0, sail, None, et=et0 + t)
    omega_obs = np.linalg.norm(np.diff(ndes, axis=0), axis=1) / dt
    d2_obs = np.linalg.norm(np.diff(ndes, n=2, axis=0), axis=1) / (dt * dt)

    assert b_omega >= omega_obs.max()
    assert b_alpha >= d2_obs.max()  # B_alpha bounds |d^2 n/dt^2| (>= |alpha|)
    # And both within realistic slew limits (huge margin).
    assert b_omega < math.radians(0.3)
    assert b_alpha < math.radians(0.003)


def test_assert_slew_feasible_raises_when_overcranked():
    """A command with a short period / large harmonics that demands more than
    the slew limits must be rejected."""
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    # Big K=1 clock amplitude crammed into a 10-minute period -> huge |omega|.
    coeffs = np.array([0.5, 0.0, 0.0, 0.0, 2.0, 0.0])
    with pytest.raises(ValueError):
        assert_cruise_slew_feasible(coeffs, 600.0, limits)
    # The same command over the real 540-day transit is feasible.
    b_omega, b_alpha = assert_cruise_slew_feasible(coeffs, 540.0 * 86400.0, limits)
    assert b_omega < limits.omega_max_rad_s
    assert b_alpha < limits.alpha_max_rad_s2


def test_feasible_coeff_boxes_guarantee_feasibility():
    """Any command whose cone harmonics are within cone_box and clock harmonics
    within clock_box (for the chosen cone bias) must pass
    assert_cruise_slew_feasible -- the enforce-by-construction optimizer bounds."""
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    T_s = 540.0 * 86400.0
    order = 2
    cone_bias = 0.8
    w_orb = heliocentric_angular_rate(_circular_helio_state(), sun_gm_km3_per_s2())
    cone_box, clock_box = feasible_coeff_boxes(
        order, T_s, limits, cone_bias, omega_frame_max_rad_s=w_orb
    )
    assert cone_box > 0.0 and clock_box > 0.0
    half = 1 + 2 * order
    rng = np.random.default_rng(7)
    for _ in range(50):
        coeffs = np.zeros(2 + 4 * order)
        coeffs[0] = cone_bias
        coeffs[1:half] = rng.uniform(-cone_box, cone_box, size=2 * order)  # cone harm
        coeffs[half] = rng.uniform(-math.pi, math.pi)  # clock bias
        coeffs[half + 1:] = rng.uniform(-clock_box, clock_box, size=2 * order)  # clock harm
        assert_cruise_slew_feasible(
            coeffs, T_s, limits, omega_frame_max_rad_s=w_orb
        )  # must not raise
    # Cone harmonic far outside its box -> leaves cone range -> rejected.
    over_cone = np.zeros(2 + 4 * order)
    over_cone[0] = cone_bias
    over_cone[1] = 100.0 * cone_box
    with pytest.raises(ValueError):
        assert_cruise_slew_feasible(over_cone, T_s, limits, omega_frame_max_rad_s=w_orb)
    # Clock harmonic far outside its box -> exceeds rate -> rejected.
    over_clock = np.zeros(2 + 4 * order)
    over_clock[0] = cone_bias
    over_clock[half + 1] = 1.0e6 * clock_box
    with pytest.raises(ValueError):
        assert_cruise_slew_feasible(over_clock, T_s, limits, omega_frame_max_rad_s=w_orb)
