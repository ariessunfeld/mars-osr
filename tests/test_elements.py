"""Fast tests for reflectors.elements (classical orbital elements).

The tests anchor against:
  - closed-form sanity (circular -> e=0, equatorial-in-frame -> i=0),
  - round-trip stability (elements round-tripped through a propagator
    should return the same a, e, i, and other secular elements should
    drift only by the pure-two-body mean-anomaly advance),
  - MME2000 rotation being a proper rotation (orthogonal, det=+1),
  - Mars pole direction being consistent with the IAU 2015 model in the
    PCK at the J2000 epoch.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflectors.dynamics import mars_gm_km3_per_s2, propagate
from reflectors.elements import (
    ClassicalElements,
    classical_elements,
    elements_in_mme2000,
    mean_motion_rad_per_s,
    mme2000_rotation_from_j2000,
    secular_argp_rate_J2_rad_per_s,
    semi_latus_rectum_km,
)
from reflectors.surface import mars_equatorial_radius_km


def _circular_state_j2000(altitude_km: float, inc_deg: float, mu: float) -> np.ndarray:
    """Circular orbit at given altitude with inclination ``inc_deg`` relative
    to the J2000 z-axis. Periapsis on +x, ascending-node at +x as well.
    """
    r = mars_equatorial_radius_km() + altitude_km
    v = float(np.sqrt(mu / r))
    inc = np.radians(inc_deg)
    return np.array([r, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])


# ---------------------------------------------------------------------------
# Sanity: circular / equatorial -> expected element values
# ---------------------------------------------------------------------------


def test_circular_orbit_reports_zero_eccentricity():
    mu = mars_gm_km3_per_s2()
    state = _circular_state_j2000(400.0, 60.0, mu)
    el = classical_elements(state, mu)
    assert el.e == pytest.approx(0.0, abs=1e-12)
    # Circular: a == |r|.
    assert el.a_km == pytest.approx(np.linalg.norm(state[:3]), rel=1e-12)


def test_equatorial_orbit_in_frame_reports_zero_inclination():
    """With v purely in the xy-plane, inclination in this frame is 0."""
    mu = mars_gm_km3_per_s2()
    state = _circular_state_j2000(400.0, 0.0, mu)
    el = classical_elements(state, mu)
    assert el.inclination_rad == pytest.approx(0.0, abs=1e-12)


def test_inclination_matches_input_60_deg():
    mu = mars_gm_km3_per_s2()
    state = _circular_state_j2000(400.0, 60.0, mu)
    el = classical_elements(state, mu)
    assert np.degrees(el.inclination_rad) == pytest.approx(60.0, abs=1e-9)


def test_period_matches_kepler_third_law():
    mu = mars_gm_km3_per_s2()
    state = _circular_state_j2000(400.0, 60.0, mu)
    el = classical_elements(state, mu)
    r = np.linalg.norm(state[:3])
    T_expected = 2 * np.pi * np.sqrt(r ** 3 / mu)
    assert el.period_s == pytest.approx(T_expected, rel=1e-12)


def test_mean_motion_and_semi_latus_rectum_helpers():
    assert mean_motion_rad_per_s(10.0, 1.0) == pytest.approx(np.sqrt(1.0 / 1000.0), rel=1e-12)
    # p = a(1 - e^2)
    assert semi_latus_rectum_km(1000.0, 0.5) == pytest.approx(750.0, rel=1e-12)
    # Degenerate cases
    assert np.isnan(mean_motion_rad_per_s(-1.0, 1.0))


# ---------------------------------------------------------------------------
# Round-trip through the propagator
# ---------------------------------------------------------------------------


def test_elements_are_invariant_under_two_body_propagation():
    """a, e, i, RAAN, argp are constants of the motion in pure two-body dynamics."""
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state_j2000(400.0, 45.0, mu)
    # Make it elliptic to exercise argp (circular has argp degenerate).
    state0[3] += 0.5  # small perturbation to break the circular degeneracy
    el0 = classical_elements(state0, mu)
    T = el0.period_s
    result = propagate(state0, (0.0, 0.5 * T))
    elf = classical_elements(result.state_km_kmps[-1], mu)
    # Conserved elements under two-body
    assert elf.a_km == pytest.approx(el0.a_km, rel=1e-10)
    assert elf.e == pytest.approx(el0.e, abs=1e-10)
    assert elf.inclination_rad == pytest.approx(el0.inclination_rad, abs=1e-12)
    assert elf.raan_rad == pytest.approx(el0.raan_rad, abs=1e-10)
    assert elf.argp_rad == pytest.approx(el0.argp_rad, abs=1e-9)
    # True anomaly is NOT conserved -- it must advance (strictly) over half a period.
    def wrap(x):  # to [0, 2pi)
        return x % (2 * np.pi)
    d_nu = (wrap(elf.nu_rad) - wrap(el0.nu_rad)) % (2 * np.pi)
    assert 0.1 < d_nu < 2 * np.pi - 0.1


def test_state_to_elements_roundtrip_via_propagation_closure():
    """One full period: elements equal AND state equal (to tolerance)."""
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state_j2000(400.0, 30.0, mu)
    state0[0] += 1200.0  # displace so orbit becomes elliptic (e ~ 0.3)
    el0 = classical_elements(state0, mu)
    T = el0.period_s
    result = propagate(state0, (0.0, T))
    elf = classical_elements(result.state_km_kmps[-1], mu)
    assert elf.a_km == pytest.approx(el0.a_km, rel=1e-10)
    assert elf.e == pytest.approx(el0.e, abs=1e-10)
    assert elf.inclination_rad == pytest.approx(el0.inclination_rad, abs=1e-12)


def test_hyperbolic_orbit_reports_negative_a_and_infinite_period():
    mu = mars_gm_km3_per_s2()
    r_p = 3800.0
    v_esc = np.sqrt(2 * mu / r_p)
    state = np.array([r_p, 0.0, 0.0, 0.0, 1.5 * v_esc, 0.0])
    el = classical_elements(state, mu)
    assert el.e > 1.0
    assert el.a_km < 0.0  # SPICE convention for hyperbolic
    assert np.isinf(el.period_s)
    # Mean motion is NaN for hyperbolic
    assert np.isnan(el.mean_motion_rad_s)
    # periapsis still well-defined via rp = a(1-e)
    assert el.periapsis_km == pytest.approx(r_p, rel=1e-9)


# ---------------------------------------------------------------------------
# MME2000 rotation
# ---------------------------------------------------------------------------


def test_mme2000_rotation_is_proper_orthogonal():
    R = mme2000_rotation_from_j2000(0.0)
    assert R.shape == (3, 3)
    # R R^T == I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
    # det(R) = +1 (proper rotation)
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-12)


def test_mme2000_z_axis_matches_iau_mars_z_axis():
    """MME-of-date z-axis must equal the IAU_MARS body-fixed z-axis in J2000.

    The zonal potential is axisymmetric about the IAU_MARS z-axis (Mars's
    spin pole WITH all IAU 2015 nutation terms), so for analytical J_2 drift
    tests to match, the MME2000 reporting frame must reference exactly that
    axis -- NOT the polynomial-only POLE_RA / POLE_DEC approximation, which
    is off by ~1.5 deg at 2026-era epochs.
    """
    import spiceypy as spice
    # Pick a nontrivial epoch so nutation phases are visibly nonzero.
    et = spice.str2et("2026-06-01T00:00:00")
    R_j2000_to_mme = mme2000_rotation_from_j2000(et)
    R_j2000_to_bf = np.asarray(spice.pxform("J2000", "IAU_MARS", et))
    z_mme_in_j2000 = R_j2000_to_mme[2, :]
    z_bf_in_j2000 = R_j2000_to_bf[2, :]
    assert np.allclose(z_mme_in_j2000, z_bf_in_j2000, atol=1e-14)
    # Also verify at the J2000 reference epoch itself.
    R2 = mme2000_rotation_from_j2000(0.0)
    R2_bf = np.asarray(spice.pxform("J2000", "IAU_MARS", 0.0))
    assert np.allclose(R2[2, :], R2_bf[2, :], atol=1e-14)


def test_mme2000_inclination_differs_from_j2000_for_equatorial_j2000_orbit():
    """A J2000-equatorial orbit (i_J2000 = 0) is NOT Mars-equatorial.

    The residual inclination equals the angle between the J2000 z-axis and
    Mars's spin pole. For 2026-era epochs the IAU 2015 pole (inclusive of
    nutation) sits at roughly 37 deg from J2000 z.
    """
    import spiceypy as spice
    mu = mars_gm_km3_per_s2()
    et = spice.str2et("2026-06-01T00:00:00")
    state = _circular_state_j2000(400.0, 0.0, mu)
    # In J2000 this has i = 0.
    el_j2000 = classical_elements(state, mu)
    assert el_j2000.inclination_rad == pytest.approx(0.0, abs=1e-12)
    # In MME-of-date the inclination equals the J2000-z-to-Mars-pole angle.
    el_mme = elements_in_mme2000(state, mu, epoch_et=et)
    i_mme_deg = float(np.degrees(el_mme.inclination_rad))
    # Also compute the angle directly from spice.pxform for the expected value.
    pole = np.asarray(spice.pxform("J2000", "IAU_MARS", et))[2, :]
    expected_deg = float(np.degrees(np.arccos(np.clip(pole[2], -1.0, 1.0))))
    assert i_mme_deg == pytest.approx(expected_deg, abs=1e-10)
    # Cross-check: in the 2020s this is in the 36-38 deg band.
    assert 36.0 < i_mme_deg < 38.0


def test_classical_elements_shape_validation():
    with pytest.raises(ValueError):
        classical_elements(np.zeros(5), 1.0)


# ---------------------------------------------------------------------------
# Brouwer first-order secular argument-of-periapsis rate under J_2
# ---------------------------------------------------------------------------


class TestSecularArgpRateJ2:
    """Pin Brouwer 1959 Eq. (40): dω/dt = (3/4) J_2 n (R/a)^2 (5cos²i − 1)/(1−e²)²."""

    def test_canonical_k12_orbit_value(self):
        """Direct evaluation at the K=12 sun-sync orbit
        (a=3901.19 km, e=0, i=93.224°).

        At J_2(Mars)=1.96e-3, R/a=0.871, n=8.49e-4 rad/s, cos i=-0.0563
        (5cos²i−1=-0.984): dω/dt ≈ -9.31e-7 rad/s ≈ -4.73 deg/sol.
        argp regresses (5cos²i < 1 ⇒ negative). The Brouwer-formula evaluation
        lands at −4.73 deg/sol.
        """
        from reflectors.mars_constants import SECONDS_PER_SOLAR_SOL_S

        a_km = 3901.19
        rate_rad_per_s = secular_argp_rate_J2_rad_per_s(
            a_km=a_km, e=0.0, inc_rad=np.radians(93.224),
        )
        rate_deg_per_sol = (
            np.degrees(rate_rad_per_s) * SECONDS_PER_SOLAR_SOL_S
        )
        # argp regresses at sun-sync ~93°.
        assert rate_rad_per_s < 0.0
        # Numeric pin: -4.73 ± 0.1 deg/sol per direct hand calculation
        # (J_2=1.96e-3, R=3396 km, μ=42828 km³/s²).
        assert -4.85 < rate_deg_per_sol < -4.6, rate_deg_per_sol

    def test_sign_flip_at_critical_inclination(self):
        """5 cos²i = 1 ⇒ i ≈ 63.4349° (prograde). Rate above this is
        positive, below it is negative, exactly zero at the critical
        inclination itself.
        """
        a_km = 7000.0
        i_crit_rad = np.arccos(np.sqrt(1.0 / 5.0))  # exactly 5cos²i=1

        rate_at = secular_argp_rate_J2_rad_per_s(a_km, 0.0, i_crit_rad)
        rate_below = secular_argp_rate_J2_rad_per_s(
            a_km, 0.0, i_crit_rad - np.radians(5.0),
        )
        rate_above = secular_argp_rate_J2_rad_per_s(
            a_km, 0.0, i_crit_rad + np.radians(5.0),
        )
        assert rate_at == pytest.approx(0.0, abs=1e-15)
        assert rate_below > 0.0
        assert rate_above < 0.0

    def test_linear_in_J2_with_overrides(self):
        """Doubling J_2 doubles the rate (formula is exactly linear in J_2)."""
        a_km = 7000.0
        i_rad = np.radians(45.0)
        rate_1 = secular_argp_rate_J2_rad_per_s(
            a_km=a_km, e=0.01, inc_rad=i_rad,
            mu_km3_s2=42828.0, ref_radius_km=3396.0, J2=1.0e-3,
        )
        rate_2 = secular_argp_rate_J2_rad_per_s(
            a_km=a_km, e=0.01, inc_rad=i_rad,
            mu_km3_s2=42828.0, ref_radius_km=3396.0, J2=2.0e-3,
        )
        assert rate_2 == pytest.approx(2.0 * rate_1, rel=1e-15)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="a_km"):
            secular_argp_rate_J2_rad_per_s(0.0, 0.0, np.radians(45.0))
        with pytest.raises(ValueError, match="e must satisfy"):
            secular_argp_rate_J2_rad_per_s(7000.0, 1.5, np.radians(45.0))
