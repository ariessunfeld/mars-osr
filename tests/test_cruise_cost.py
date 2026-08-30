"""Tests for the interplanetary cruise single-shooting cost (reflectors.cruise_cost).

A synthetic Sun-centred problem (no dependency on the gitignored endpoint files):
propagate a heliocentric arc under a Fourier command and check the terminal-miss
machinery and the picklable cost closure (required for the CloudpickleMap DE solve).
"""

from __future__ import annotations

import math

import cloudpickle
import numpy as np
import pytest

from reflectors.central_body import sun_central_body
from reflectors.cruise_command import heliocentric_orbit_normal
from reflectors.cruise_cost import (
    cruise_terminal_miss,
    make_cruise_terminal_miss_cost,
    propagate_cruise,
)
from reflectors.dynamics import sun_gm_km3_per_s2
from reflectors.sail_designs import make_canonical_sail
from reflectors.third_body import earth_third_body, mars_third_body

AU_KM = 1.495978707e8


def _problem():
    mu = sun_gm_km3_per_s2()
    z0 = np.array([AU_KM, 0.0, 0.0, 0.0, math.sqrt(mu / AU_KM), 0.0])
    ref = heliocentric_orbit_normal(z0)
    sail = make_canonical_sail(0.018)
    cb = sun_central_body()
    tb = (earth_third_body(), mars_third_body())
    return mu, z0, ref, sail, cb, tb


def test_propagate_cruise_runs_to_t_final():
    mu, z0, ref, sail, cb, tb = _problem()
    T_s = 30.0 * 86400.0
    coeffs = np.array([0.6, 0.1, 0.0, 0.0, 1.0, 0.0])
    run = propagate_cruise(coeffs, z0, 0.0, T_s, ref, sail, cb, tb)
    assert run.termination_reason == "t_final"
    assert np.all(np.isfinite(run.orbit_state_km_kmps))
    # The arc advanced roughly a heliocentric month (it should have moved).
    assert np.linalg.norm(run.orbit_state_km_kmps[-1, :3] - z0[:3]) > 1e6


def test_terminal_miss_and_cost_consistency():
    mu, z0, ref, sail, cb, tb = _problem()
    T_s = 30.0 * 86400.0
    # Target = where a feathered (no-SRP-ish) arc lands, so the miss is moderate.
    coeffs = np.array([0.6, 0.0, 0.0, 1.0, 0.0, 0.0])  # cone bias only-ish
    z_tgt = z0.copy()
    r_miss, v_miss, zT = cruise_terminal_miss(
        coeffs, z0, z_tgt, 0.0, T_s, ref, sail, cb, tb
    )
    assert r_miss > 0.0 and v_miss > 0.0 and np.all(np.isfinite(zT))

    r_scale, v_scale, w_v = AU_KM, 29.78, 1.0
    cost = make_cruise_terminal_miss_cost(
        z0, z_tgt, 0.0, T_s, ref, sail, cb, tb,
        r_scale_km=r_scale, v_scale_kmps=v_scale, w_v=w_v,
    )
    c = cost(coeffs)
    assert c == (r_miss / r_scale + w_v * v_miss / v_scale)


def test_osculating_period_guards_overflow_near_parabolic():
    """Regression for the cruise overflow: a near-parabolic bound orbit (E->0^-)
    has a huge semi-major axis whose a**3 would overflow double precision; the
    step-sizing period helper must return None (use the fallback step) instead of
    raising OverflowError, which crashed DE workers on such candidates."""
    from reflectors.escape import _osculating_period_s

    mu = sun_gm_km3_per_s2()
    r = np.array([AU_KM, 0.0, 0.0])
    v_esc = math.sqrt(2.0 * mu / AU_KM)
    v = np.array([0.0, v_esc * (1.0 - 1e-12), 0.0])  # just bound -> a >> 1e12 km
    p = _osculating_period_s(np.concatenate([r, v]), mu)
    assert p is None
    # A normal bound orbit still returns a sensible finite period.
    v_circ = math.sqrt(mu / AU_KM)
    p2 = _osculating_period_s(np.array([AU_KM, 0, 0, 0, v_circ, 0.0]), mu)
    assert p2 is not None and math.isfinite(p2) and p2 > 0.0


def test_cruise_defect_vector_matches_terminal_miss_norms():
    """make_cruise_defect returns the signed, non-dimensional 6-vector residual;
    its position/velocity sub-norms must equal the scalar misses from
    cruise_terminal_miss (the same propagation), scaled. This is the IPOPT
    equality-constraint vector."""
    from reflectors.cruise_cost import cruise_terminal_miss, make_cruise_defect

    mu, z0, ref, sail, cb, tb = _problem()
    T_s = 30.0 * 86400.0
    coeffs = np.array([0.6, 0.1, 0.0, 0.0, 1.0, 0.0])
    z_tgt = z0 + np.array([3.0e6, -2.0e6, 1.0e6, 0.1, -0.2, 0.05])
    r_scale, v_scale = AU_KM, 29.78
    defect = make_cruise_defect(
        z0, z_tgt, 0.0, T_s, ref, sail, cb, tb,
        r_scale_km=r_scale, v_scale_kmps=v_scale,
    )
    g = defect(coeffs)
    assert g.shape == (6,)
    r_miss, v_miss, _ = cruise_terminal_miss(
        coeffs, z0, z_tgt, 0.0, T_s, ref, sail, cb, tb
    )
    assert np.linalg.norm(g[:3]) * r_scale == pytest.approx(r_miss, rel=1e-9)
    assert np.linalg.norm(g[3:]) * v_scale == pytest.approx(v_miss, rel=1e-9)


def test_cost_closure_is_cloudpicklable():
    """The DE workers receive the cost via cloudpickle; it must round-trip and
    still evaluate (the closure captures only picklable data)."""
    mu, z0, ref, sail, cb, tb = _problem()
    cost = make_cruise_terminal_miss_cost(
        z0, z0.copy(), 0.0, 20.0 * 86400.0, ref, sail, cb, tb,
        r_scale_km=AU_KM, v_scale_kmps=29.78,
    )
    coeffs = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    c0 = cost(coeffs)
    cost2 = cloudpickle.loads(cloudpickle.dumps(cost))
    assert cost2(coeffs) == c0
