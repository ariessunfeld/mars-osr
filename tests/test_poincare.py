"""Fast tests for ``reflectors.poincare``.

Detection helpers operate on a dense ``(N, 6)`` Cartesian state history
independently of the propagator. These tests construct the state history from
Kepler's equation and check detector arithmetic at machine precision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.dynamics import mars_gm_km3_per_s2
from reflectors.elements import (
    elements_in_mme2000,
    state_from_classical_mme2000,
)
from reflectors.poincare import (
    PoincareCrossing,
    find_ascending_node_crossings,
    find_periapsis_crossings,
)
from reflectors.surface import mars_equatorial_radius_km
from reflectors.sun_sync import repeat_ground_track_altitude


def _solve_kepler_E(M_rad: float, e: float, tol: float = 1e-13) -> float:
    """Solve Kepler's equation M = E - e sin E by Newton iteration."""
    E = M_rad if e < 0.8 else math.pi
    for _ in range(60):
        f = E - e * math.sin(E) - M_rad
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            return E
    return E


def _kepler_state_at(
    *,
    a_km: float,
    e: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    M0_rad: float,
    t_s: float,
    mu_km3_s2: float,
    epoch_et: float,
) -> np.ndarray:
    """Analytic Kepler state at time ``t_s`` past ``epoch_et``."""
    n = math.sqrt(mu_km3_s2 / a_km ** 3)
    M = M0_rad + n * t_s
    E = _solve_kepler_E(M, e)
    cos_E = math.cos(E)
    sin_E = math.sin(E)
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )
    return state_from_classical_mme2000(
        a_km=a_km, e=e, inclination_rad=inc_rad,
        raan_rad=raan_rad, argp_rad=argp_rad, nu_rad=nu,
        mu_km3_s2=mu_km3_s2, epoch_et=epoch_et + t_s,
    )


def _kepler_state_history(
    *,
    a_km: float,
    e: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    M0_rad: float,
    duration_s: float,
    sample_dt_s: float,
    mu_km3_s2: float,
    epoch_et: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    t_s = np.arange(0.0, duration_s + 1e-6, sample_dt_s)
    state_arr = np.zeros((len(t_s), 6), dtype=float)
    for k, t in enumerate(t_s):
        state_arr[k] = _kepler_state_at(
            a_km=a_km, e=e, inc_rad=inc_rad,
            raan_rad=raan_rad, argp_rad=argp_rad,
            M0_rad=M0_rad, t_s=float(t),
            mu_km3_s2=mu_km3_s2, epoch_et=epoch_et,
        )
    return t_s, state_arr


# ---------------------------------------------------------------------------
# Group 1: Ascending-node detection
# ---------------------------------------------------------------------------


class TestFindAscendingNodeCrossings:
    def test_K12_kepler_orbit_yields_K_crossings(self):
        """At the K=12 sun-sync repeating ground-track altitude, an
        unperturbed Kepler orbit produces exactly 12 ascending-node
        crossings per Mars solar sol."""
        a_km = repeat_ground_track_altitude(12)[0]
        mu = mars_gm_km3_per_s2()
        t_s, state_arr = _kepler_state_history(
            a_km=a_km, e=0.001,
            inc_rad=math.radians(93.224),
            raan_rad=math.radians(45.0),
            argp_rad=math.radians(0.0),
            M0_rad=math.radians(0.0),
            duration_s=88775.0, sample_dt_s=5.0,
            mu_km3_s2=mu, epoch_et=0.0,
        )
        crossings = find_ascending_node_crossings(t_s, state_arr, mu)
        assert len(crossings) == 12, [c.t_rel_s for c in crossings]

    def test_kepler_orbit_e_constant_across_crossings(self):
        """``e`` at every asc-node crossing of a pure Kepler orbit is
        constant — no perturbation, no secular drift, sampled at
        matched orbital phase."""
        a_km = 3901.19
        mu = mars_gm_km3_per_s2()
        t_s, state_arr = _kepler_state_history(
            a_km=a_km, e=0.005,
            inc_rad=math.radians(93.0),
            raan_rad=math.radians(20.0),
            argp_rad=math.radians(15.0),
            M0_rad=math.radians(0.0),
            duration_s=88775.0, sample_dt_s=5.0,
            mu_km3_s2=mu, epoch_et=0.0,
        )
        crossings = find_ascending_node_crossings(t_s, state_arr, mu)
        assert len(crossings) >= 2
        e_values = np.array([c.e for c in crossings])
        # Linear-interpolation truncation error at 5-sec sampling
        # (sample-dt over ~7300-sec period, ~7e-4 of period), combined
        # with spice.oscltx finite precision near e=0, leaves a residual
        # of a few × 1e-6 on osculating e at the crossing. That's small
        # enough to confirm "no secular drift along a Kepler orbit"; a
        # tighter pin would chase round-off rather than physics.
        assert np.max(np.abs(e_values - e_values[0])) < 1e-5

    def test_zero_crossings_when_z_always_negative(self):
        """Construct a fake state history with z<0 throughout; detector
        should return 0 crossings (no neg-to-pos transition)."""
        # 100 samples; z is always -1; all other columns arbitrary.
        n_samples = 100
        state_arr = np.tile(
            np.array([1000.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
            (n_samples, 1),
        )
        t_s = np.arange(n_samples, dtype=float) * 5.0
        mu = mars_gm_km3_per_s2()
        crossings = find_ascending_node_crossings(t_s, state_arr, mu)
        assert crossings == []

    def test_linear_interp_recovers_t_to_subsample_accuracy(self):
        """At 5-sec sampling, the linear-interpolation crossing time
        should match the analytic asc-node time to better than 5e-3 s
        (i.e. linear interp is accurate to ~1e-6 of the orbital
        period for the reference geometry).
        """
        a_km = repeat_ground_track_altitude(12)[0]
        mu = mars_gm_km3_per_s2()
        period_s = 2.0 * math.pi * math.sqrt(a_km ** 3 / mu)
        # Geometry: argp=0 ⇒ periapsis at ascending node ⇒ asc-node is
        # at u = argp + nu = 0 (or 2π). Start at M0 = 270° (circular,
        # so nu = M0 = 270°) ⇒ u_start = 270°. Asc-node is at u = 360°,
        # i.e. +90° away ⇒ first crossing at t = period / 4.
        t_s, state_arr = _kepler_state_history(
            a_km=a_km, e=0.0,
            inc_rad=math.radians(93.224),
            raan_rad=0.0,
            argp_rad=0.0,
            M0_rad=math.radians(270.0),
            duration_s=period_s * 1.1, sample_dt_s=5.0,
            mu_km3_s2=mu, epoch_et=0.0,
        )
        crossings = find_ascending_node_crossings(t_s, state_arr, mu)
        assert len(crossings) >= 1
        t_expected = period_s / 4.0
        assert abs(crossings[0].t_rel_s - t_expected) < 5e-3

    def test_crossings_indexed_in_order(self):
        a_km = repeat_ground_track_altitude(12)[0]
        mu = mars_gm_km3_per_s2()
        t_s, state_arr = _kepler_state_history(
            a_km=a_km, e=0.001,
            inc_rad=math.radians(93.224),
            raan_rad=math.radians(45.0),
            argp_rad=math.radians(0.0),
            M0_rad=math.radians(0.0),
            duration_s=88775.0, sample_dt_s=5.0,
            mu_km3_s2=mu, epoch_et=0.0,
        )
        crossings = find_ascending_node_crossings(t_s, state_arr, mu)
        for k, c in enumerate(crossings):
            assert c.crossing_index == k
        ts = [c.t_rel_s for c in crossings]
        assert ts == sorted(ts)

    def test_altitude_field_matches_r_minus_R_eq(self):
        """Pin the altitude convention: spherical altitude = ``r - R_eq_Mars``."""
        a_km = 3901.19
        mu = mars_gm_km3_per_s2()
        t_s, state_arr = _kepler_state_history(
            a_km=a_km, e=0.0,
            inc_rad=math.radians(93.224),
            raan_rad=0.0, argp_rad=0.0, M0_rad=0.0,
            duration_s=15000.0, sample_dt_s=5.0,
            mu_km3_s2=mu, epoch_et=0.0,
        )
        crossings = find_ascending_node_crossings(t_s, state_arr, mu)
        assert len(crossings) >= 1
        c = crossings[0]
        assert c.altitude_km == pytest.approx(
            c.r_km - mars_equatorial_radius_km(), abs=1e-12,
        )

    def test_invalid_state_array_shape_raises(self):
        t_s = np.arange(10, dtype=float)
        bad_state = np.zeros((10, 5))
        mu = mars_gm_km3_per_s2()
        with pytest.raises(ValueError, match=r"state_arr must be shape"):
            find_ascending_node_crossings(t_s, bad_state, mu)

    def test_length_mismatch_raises(self):
        t_s = np.arange(10, dtype=float)
        bad_state = np.zeros((9, 6))
        mu = mars_gm_km3_per_s2()
        with pytest.raises(ValueError, match=r"length mismatch"):
            find_ascending_node_crossings(t_s, bad_state, mu)


# ---------------------------------------------------------------------------
# Group 2: Periapsis detection (smoke-level)
# ---------------------------------------------------------------------------


class TestFindPeriapsisCrossings:
    def test_eccentric_kepler_one_periapsis_per_orbit(self):
        """An eccentric Kepler orbit has exactly one periapsis crossing
        per orbital period; pin the count over 5 periods."""
        a_km = 3901.19
        e = 0.05
        mu = mars_gm_km3_per_s2()
        period_s = 2.0 * math.pi * math.sqrt(a_km ** 3 / mu)
        # Start at apoapsis (nu=180°) so the first periapsis crossing
        # is at exactly t = period/2.
        t_s, state_arr = _kepler_state_history(
            a_km=a_km, e=e,
            inc_rad=math.radians(93.0),
            raan_rad=math.radians(20.0),
            argp_rad=math.radians(0.0),
            M0_rad=math.radians(180.0),
            duration_s=5.0 * period_s, sample_dt_s=5.0,
            mu_km3_s2=mu, epoch_et=0.0,
        )
        crossings = find_periapsis_crossings(t_s, state_arr, mu)
        assert len(crossings) == 5, [c.t_rel_s for c in crossings]
        # First crossing within 5 s of period/2 (sample cadence).
        assert abs(crossings[0].t_rel_s - period_s / 2.0) < 5.0
