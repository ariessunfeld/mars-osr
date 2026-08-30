"""Fast tests for reflectors.dynamics (two-body Mars propagator).

Each test pins a representative value or a geometric/physical identity. Conserved
quantities are checked against the Kepler invariants (energy, angular
momentum) of the two-body problem; geometry is checked against closed-form
formulas (vis-viva, Kepler period, hyperbolic escape).

All orbits use the physical Mars mu from SPICE, so any kernel change that
perturbs mu_Mars at the 1e-5 level will move these test residuals by a
correspondingly small amount.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflectors.dynamics import (
    DEFAULT_OPTIONS,
    PropagationOptions,
    PropagationResult,
    mars_gm_km3_per_s2,
    propagate,
    two_body_acceleration,
)
from reflectors.surface import mars_equatorial_radius_km


def _circular_state(altitude_km: float, mu: float) -> tuple[np.ndarray, float]:
    r = mars_equatorial_radius_km() + altitude_km
    v = float(np.sqrt(mu / r))
    state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    T = 2.0 * np.pi * float(np.sqrt(r ** 3 / mu))
    return state0, T


def _eccentric_state(periapsis_alt_km: float, e: float, mu: float) -> tuple[np.ndarray, float, float]:
    """Eccentric orbit with periapsis on +x, v along +y at periapsis.

    Returns (state0, T, a).
    """
    r_p = mars_equatorial_radius_km() + periapsis_alt_km
    a = r_p / (1.0 - e)
    v_p = float(np.sqrt(mu * (2.0 / r_p - 1.0 / a)))
    state0 = np.array([r_p, 0.0, 0.0, 0.0, v_p, 0.0])
    T = 2.0 * np.pi * float(np.sqrt(a ** 3 / mu))
    return state0, T, a


# ---------------------------------------------------------------------------
# Basic sanity: mu, acceleration shape and magnitude
# ---------------------------------------------------------------------------


def test_mars_gm_matches_published_value():
    """mu_Mars from SPICE (gm_de440.tpc) matches the published 4.2828e4 value."""
    mu = mars_gm_km3_per_s2()
    # Konopliv 2016/2020 Mars Geophysical Parameters: 4.2828372e4 km^3/s^2
    # within +-0.001 km^3/s^2 on the reported number.
    assert mu == pytest.approx(4.282837e4, rel=1e-5)
    assert 42828.0 < mu < 42829.0


def test_two_body_acceleration_direction_and_magnitude():
    """a = -mu r / |r|^3: direction is -r_hat, magnitude is mu / |r|^2."""
    mu = mars_gm_km3_per_s2()
    # Random positions away from the origin; magnitude check.
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.uniform(-1.0, 1.0, size=3) * 1e4  # km, up to ~Mars-orbit scales
        if np.linalg.norm(r) < 1e-6:
            continue
        a = two_body_acceleration(r, mu)
        # Magnitude equals mu/|r|^2.
        assert np.linalg.norm(a) == pytest.approx(mu / np.dot(r, r), rel=1e-12)
        # Direction antiparallel to r.
        r_hat = r / np.linalg.norm(r)
        a_hat = a / np.linalg.norm(a)
        assert np.allclose(a_hat, -r_hat, atol=1e-12)


def test_two_body_acceleration_zero_radius_raises():
    with pytest.raises(ValueError):
        two_body_acceleration(np.zeros(3), mars_gm_km3_per_s2())


# ---------------------------------------------------------------------------
# Closure: circular and eccentric orbits return to their initial state
# ---------------------------------------------------------------------------


def test_circular_400km_orbit_closes_after_one_period():
    """Circular orbit at 400 km returns to start within sub-um at default tol."""
    mu = mars_gm_km3_per_s2()
    state0, T = _circular_state(400.0, mu)
    result = propagate(state0, (0.0, T))
    dr = np.linalg.norm(result.state_km_kmps[-1, :3] - state0[:3])
    dv = np.linalg.norm(result.state_km_kmps[-1, 3:] - state0[3:])
    # 400 km circular orbit, default tolerances: expect sub-um position error.
    assert dr < 1e-6, f"position residual {dr:.3e} km"
    assert dv < 1e-9, f"velocity residual {dv:.3e} km/s"


def test_eccentric_orbit_closes_after_one_period():
    """e=0.5 orbit closes after one period at default tolerances."""
    mu = mars_gm_km3_per_s2()
    state0, T, _ = _eccentric_state(400.0, 0.5, mu)
    result = propagate(state0, (0.0, T))
    dr = np.linalg.norm(result.state_km_kmps[-1, :3] - state0[:3])
    dv = np.linalg.norm(result.state_km_kmps[-1, 3:] - state0[3:])
    # The eccentric case requires additional integration margin near periapsis.
    assert dr < 1e-4, f"position residual {dr:.3e} km"
    assert dv < 1e-7, f"velocity residual {dv:.3e} km/s"


def test_vis_viva_periapsis_apoapsis_speed_ratio():
    """For an elliptic orbit, v_periapsis / v_apoapsis = (1+e)/(1-e)."""
    mu = mars_gm_km3_per_s2()
    e = 0.5
    state0, T, a = _eccentric_state(400.0, e, mu)
    # t_eval: sample densely and find the minimum-r (apoapsis) sample.
    t_eval = np.linspace(0.0, T, 2001)
    result = propagate(state0, (0.0, T), t_eval_s=t_eval)
    r_norm = np.linalg.norm(result.positions(), axis=1)
    v_norm = np.linalg.norm(result.velocities(), axis=1)
    i_peri = int(np.argmin(r_norm))
    i_apo = int(np.argmax(r_norm))
    ratio = v_norm[i_peri] / v_norm[i_apo]
    expected = (1.0 + e) / (1.0 - e)
    # 2001 samples over one period -> up to ~5 s timing error, <~0.1% speed error.
    assert ratio == pytest.approx(expected, rel=2e-3), f"got {ratio}, expected {expected}"


# ---------------------------------------------------------------------------
# Conservation laws (pure two-body)
# ---------------------------------------------------------------------------


def test_energy_conservation_over_one_period():
    """Specific mechanical energy is conserved to high precision."""
    mu = mars_gm_km3_per_s2()
    state0, T, _ = _eccentric_state(400.0, 0.3, mu)
    result = propagate(
        state0, (0.0, T),
        t_eval_s=np.linspace(0.0, T, 201),
        options=PropagationOptions.high_accuracy(),
    )
    E = result.specific_energy()
    rel = float(np.max(np.abs((E - E[0]) / E[0])))
    assert rel < 1e-11, f"relative energy drift {rel:.3e}"


def test_angular_momentum_conservation_over_one_period():
    """Specific angular-momentum vector is conserved in magnitude and direction."""
    mu = mars_gm_km3_per_s2()
    state0, T, _ = _eccentric_state(400.0, 0.3, mu)
    result = propagate(
        state0, (0.0, T),
        t_eval_s=np.linspace(0.0, T, 201),
        options=PropagationOptions.high_accuracy(),
    )
    h = result.specific_angular_momentum()
    h0 = h[0]
    h0_norm = float(np.linalg.norm(h0))
    # Magnitude
    mag_rel = float(np.max(np.abs(np.linalg.norm(h, axis=1) - h0_norm) / h0_norm))
    assert mag_rel < 1e-12, f"|h| drift {mag_rel:.3e}"
    # Direction (cosine of angle with h0 should stay ~1).
    h_unit = h / np.linalg.norm(h, axis=1, keepdims=True)
    cos_theta = h_unit @ (h0 / h0_norm)
    assert np.all(cos_theta > 1.0 - 1e-12), f"min cos(h, h0) = {cos_theta.min():.12f}"


# ---------------------------------------------------------------------------
# Integrator-behaviour tests (time reversal, t_eval, determinism)
# ---------------------------------------------------------------------------


def test_time_reversal_returns_to_initial_state():
    """Integrate forward one period, then backward one period; recover start."""
    mu = mars_gm_km3_per_s2()
    state0, T, _ = _eccentric_state(400.0, 0.3, mu)
    fwd = propagate(state0, (0.0, T), options=PropagationOptions.high_accuracy())
    end_state = fwd.state_km_kmps[-1]
    bwd = propagate(end_state, (T, 0.0), options=PropagationOptions.high_accuracy())
    back_state = bwd.state_km_kmps[-1]
    dr = float(np.linalg.norm(back_state[:3] - state0[:3]))
    dv = float(np.linalg.norm(back_state[3:] - state0[3:]))
    # Round trip of one eccentric period at high-accuracy preset.
    assert dr < 1e-6, f"time-reversal position drift {dr:.3e} km"
    assert dv < 1e-9, f"time-reversal velocity drift {dv:.3e} km/s"


def test_t_eval_times_are_honored_exactly():
    """Times returned match the requested t_eval exactly."""
    mu = mars_gm_km3_per_s2()
    state0, T = _circular_state(400.0, mu)
    t_eval = np.array([0.0, T / 4, T / 2, 3 * T / 4, T])
    result = propagate(state0, (0.0, T), t_eval_s=t_eval)
    assert result.t_s.shape == t_eval.shape
    assert np.allclose(result.t_s, t_eval, atol=1e-12)


def test_propagation_is_deterministic():
    """Two identical runs produce byte-identical output arrays."""
    mu = mars_gm_km3_per_s2()
    state0, T = _circular_state(400.0, mu)
    r1 = propagate(state0, (0.0, T))
    r2 = propagate(state0, (0.0, T))
    # Byte-identical trajectories across back-to-back calls (same scipy version,
    # same kwargs, same initial state).
    assert np.array_equal(r1.t_s, r2.t_s)
    assert np.array_equal(r1.state_km_kmps, r2.state_km_kmps)
    assert r1.n_rhs_calls == r2.n_rhs_calls


def test_propagation_options_defaults_and_presets():
    """Default options are DOP853 @ 1e-12 / 1e-9 and presets are sensible."""
    assert DEFAULT_OPTIONS.method == "DOP853"
    assert DEFAULT_OPTIONS.rtol == 1e-12
    assert DEFAULT_OPTIONS.atol == 1e-9
    fast = PropagationOptions.fast()
    high = PropagationOptions.high_accuracy()
    assert fast.rtol > DEFAULT_OPTIONS.rtol
    assert high.rtol < DEFAULT_OPTIONS.rtol
    # Frozen dataclass -- attempting mutation raises.
    with pytest.raises(Exception):
        DEFAULT_OPTIONS.method = "RK45"  # type: ignore[misc]


def test_result_helpers_return_correct_shapes():
    mu = mars_gm_km3_per_s2()
    state0, T = _circular_state(400.0, mu)
    t_eval = np.linspace(0.0, T, 11)
    result = propagate(state0, (0.0, T), t_eval_s=t_eval)
    assert isinstance(result, PropagationResult)
    assert result.positions().shape == (11, 3)
    assert result.velocities().shape == (11, 3)
    assert result.specific_energy().shape == (11,)
    assert result.specific_angular_momentum().shape == (11, 3)


# ---------------------------------------------------------------------------
# Hyperbolic trajectory (e > 1): positive energy, monotone r.v
# ---------------------------------------------------------------------------


def test_hyperbolic_trajectory_energy_positive_and_conserved():
    """An escape trajectory has specific energy > 0 and conserves it."""
    mu = mars_gm_km3_per_s2()
    # Periapsis at 3800 km, speed 150% of escape -> hyperbolic.
    r_p = 3800.0
    v_esc = np.sqrt(2.0 * mu / r_p)
    state0 = np.array([r_p, 0.0, 0.0, 0.0, 1.5 * v_esc, 0.0])
    # Integrate long enough to leave Mars unambiguously.
    tf = 6 * 3600.0  # 6 h
    result = propagate(state0, (0.0, tf), options=PropagationOptions.high_accuracy())
    E = result.specific_energy()
    # Specific energy is positive (hyperbolic).
    assert np.all(E > 0.0)
    # Conserved over the trajectory.
    rel = float(np.max(np.abs((E - E[0]) / E[0])))
    assert rel < 1e-10
    # r.v is monotone increasing on an outbound leg past periapsis.
    r = result.positions()
    v = result.velocities()
    rv = np.einsum("ij,ij->i", r, v)
    # Skip the initial 0 at periapsis; after that it must increase monotonically.
    diffs = np.diff(rv[1:])
    assert np.all(diffs > 0.0), f"r.v not monotone; min diff {diffs.min():.3e}"


def test_synthetic_mu_override_works():
    """Callers can override mu for synthetic-body tests (e.g. unit-mu)."""
    mu_syn = 1.0  # km^3/s^2 -- tiny, just checks plumbing
    r0 = 10.0
    v_circ = float(np.sqrt(mu_syn / r0))
    state0 = np.array([r0, 0.0, 0.0, 0.0, v_circ, 0.0])
    T = 2.0 * np.pi * float(np.sqrt(r0 ** 3 / mu_syn))
    result = propagate(state0, (0.0, T), mu_km3_s2=mu_syn)
    assert result.mu_km3_s2 == mu_syn
    dr = np.linalg.norm(result.state_km_kmps[-1, :3] - state0[:3])
    assert dr < 1e-8
