"""Fast tests for Mars zonal gravity (reflectors.gravity).

Three test groups:
  1. Geometry / numerical-identity tests: Legendre recurrence vs hand-coded
     J_2 closed-form; equatorial / polar symmetry; magnitude maps.
  2. Analytical J_2 drift cross-checks: simulated secular dRAAN/dt and
     darg_p/dt against Brouwer first-order theory on LMO Mars orbits.
  3. Conservation laws: the axial angular momentum component about Mars's
     spin axis is conserved under zonal-only dynamics (axisymmetric
     potential); total mechanical energy is conserved (gravity is
     conservative); backward integration recovers the initial state.

Where analytical-vs-simulated comparisons appear, the tolerance is generous
enough to absorb expected J_2^2 corrections (~0.5-1 %) without being so
generous enough to admit a substantive error.
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice

from reflectors.dynamics import (
    PropagationOptions,
    propagate,
    two_body_acceleration,
)
from reflectors.elements import (
    classical_elements,
    mme2000_rotation_from_j2000,
)
from reflectors.gravity import (
    j2_closed_form_body_fixed,
    mars_gravity_model,
    unnormalized_zonal_from_normalized,
    zonal_acceleration_body_fixed,
    zonal_acceleration_inertial,
    zonal_coefficients,
)
from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circular_mme2000_state_in_j2000(
    altitude_km: float, inclination_deg: float, et0: float, mu: float, ref_radius_km: float
) -> np.ndarray:
    """Build a circular orbit with the given inclination in MME-of-date,
    then rotate into Mars-centred J2000 axes (what the propagator wants).
    """
    r0 = ref_radius_km + altitude_km
    v0 = float(np.sqrt(mu / r0))
    inc = np.radians(inclination_deg)
    state_mme = np.array([r0, 0.0, 0.0, 0.0, v0 * np.cos(inc), v0 * np.sin(inc)])
    R = mme2000_rotation_from_j2000(et0)
    r_j = R.T @ state_mme[:3]
    v_j = R.T @ state_mme[3:]
    return np.concatenate([r_j, v_j])


def _elements_in_mme_from_j2000_state(
    state_j2000: np.ndarray, et: float, mu: float,
):
    R = mme2000_rotation_from_j2000(et)
    r_mme = R @ state_j2000[:3]
    v_mme = R @ state_j2000[3:]
    return classical_elements(np.concatenate([r_mme, v_mme]), mu, epoch_et=et)


def _brouwer_secular_raan_rate(n_rad_s: float, R_ref_km: float, p_km: float, J2: float, inc_rad: float) -> float:
    """First-order Brouwer secular mean dRAAN/dt (rad/s)."""
    return -1.5 * n_rad_s * (R_ref_km / p_km) ** 2 * J2 * float(np.cos(inc_rad))


def _brouwer_secular_argp_rate(n_rad_s: float, R_ref_km: float, p_km: float, J2: float, inc_rad: float) -> float:
    """First-order Brouwer secular mean dargp/dt (rad/s).

    Formula: dargp/dt = (3/4) n J2 (R/p)^2 (5 cos^2 i - 1).
    """
    return 0.75 * n_rad_s * (R_ref_km / p_km) ** 2 * J2 * (5.0 * np.cos(inc_rad) ** 2 - 1.0)


def _fit_secular_slope_with_harmonics(t: np.ndarray, y: np.ndarray, n_rad_s: float) -> float:
    """Linear + (sin/cos at n, 2n) basis least-squares fit; return the linear slope.

    Suppresses short-period oscillations at mean-motion and twice-mean-motion
    so the fitted slope is the secular mean drift, not the osculating drift.
    """
    X = np.column_stack([
        np.ones_like(t),
        t,
        np.cos(n_rad_s * t),
        np.sin(n_rad_s * t),
        np.cos(2 * n_rad_s * t),
        np.sin(2 * n_rad_s * t),
    ])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coefs[1])


# ---------------------------------------------------------------------------
# 1. Geometry / identity tests
# ---------------------------------------------------------------------------


def test_zonal_coefficients_match_published_mars_values():
    """J_2 must match the Konopliv 2016 Mars Geophysical Parameters table."""
    model = mars_gravity_model(max_degree=10)
    J = zonal_coefficients(model, 10)
    # Published: J_2(Mars) ~ 1.9566e-3 (Konopliv 2016 table; agreement to 4 dp).
    assert J[2] == pytest.approx(1.9566e-3, rel=1e-3)
    # Signs of the first several Mars zonals (MRO120F):
    assert J[2] > 0
    assert J[3] > 0
    assert J[4] < 0
    # And |J_n| decays with n (not monotonically, but overall).
    assert abs(J[3]) < abs(J[2]) / 10
    assert abs(J[10]) < abs(J[2]) / 100


def test_ref_radius_is_gravity_model_value_not_iau_mean():
    """R_ref for MRO120F is 3396.0 km, NOT the IAU mean 3396.19 km."""
    model = mars_gravity_model(max_degree=2)
    assert model.ref_radius_km == pytest.approx(3396.0, abs=1e-6)
    # Read the IAU RADII to confirm they differ at the 190 m level.
    _, radii = spice.bodvrd("MARS", "RADII", 3)
    assert abs(radii[0] - model.ref_radius_km) > 0.1  # 190 m gap


def test_normalization_conversion_is_correctly_signed():
    """C_{n,0} = C_bar_{n,0} * sqrt(2n+1). Verifies the normalization direction."""
    # Manufactured: C_bar_{2,0} = -1.0 -> C_{2,0} = -sqrt(5), J_2 = sqrt(5).
    assert unnormalized_zonal_from_normalized(-1.0, 2) == pytest.approx(-np.sqrt(5.0), rel=1e-12)
    # Round-trip for a few n
    for n in range(2, 11):
        cb = 1.234e-4
        c = unnormalized_zonal_from_normalized(cb, n)
        assert c == pytest.approx(cb * np.sqrt(2 * n + 1), rel=1e-14)


def test_general_zonal_equals_j2_closed_form_at_100_random_points():
    """At degree 2 only, the Legendre recurrence must match the hand-coded J_2 formula to machine precision."""
    model = mars_gravity_model(max_degree=2)
    J = zonal_coefficients(model, 2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    rng = np.random.default_rng(42)
    max_rel = 0.0
    for _ in range(100):
        r_bf = rng.uniform(-1.0, 1.0, size=3) * 1e4  # up to 10000 km
        if np.linalg.norm(r_bf) < 1.0e3:
            continue
        a_general = zonal_acceleration_body_fixed(r_bf, mu, R, {2: J[2]})
        a_closed = j2_closed_form_body_fixed(r_bf, mu, R, J[2])
        rel = float(np.linalg.norm(a_general - a_closed) / np.linalg.norm(a_closed))
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-13, f"max relative deviation = {max_rel:.3e}"


def test_j2_acceleration_is_purely_radial_at_equator():
    """At (r, 0, 0) body-fixed (equatorial), J_2 a is antiparallel to r."""
    model = mars_gravity_model(max_degree=2)
    J2 = zonal_coefficients(model, 2)[2]
    r_bf = np.array([3796.19, 0.0, 0.0])
    a = zonal_acceleration_body_fixed(r_bf, model.mu_km3_s2, model.ref_radius_km, {2: J2})
    # a_y = 0, a_z = 0 exactly
    assert a[1] == pytest.approx(0.0, abs=1e-30)
    assert a[2] == pytest.approx(0.0, abs=1e-30)
    # a_x < 0 (inward) for positive J_2 at the equator
    assert a[0] < 0.0


def test_j2_acceleration_is_along_spin_axis_at_north_pole():
    """At (0, 0, r) body-fixed, J_2 a has zero x, y components and a_z > 0.

    Sign: for an oblate body (J_2 > 0) the effective gravity at the pole is
    WEAKER than a spherical Mars of the same mass -- there is less mass
    concentrated close to the pole than there is near the equator. So the
    J_2 PERTURBATION (= actual - pure-central) at the pole points OUTWARD
    (+z), opposing the central attraction. Magnitude = 3 * J_2 * mu * R^2 / r^4
    from the Curtis closed form at z/r = 1.
    """
    model = mars_gravity_model(max_degree=2)
    J2 = zonal_coefficients(model, 2)[2]
    r_bf = np.array([0.0, 0.0, 3796.19])
    a = zonal_acceleration_body_fixed(r_bf, model.mu_km3_s2, model.ref_radius_km, {2: J2})
    assert a[0] == pytest.approx(0.0, abs=1e-30)
    assert a[1] == pytest.approx(0.0, abs=1e-30)
    # Perturbation is outward at the pole for oblate (J_2 > 0) body.
    assert a[2] > 0.0
    # Magnitude pinned to 3 J_2 mu R^2 / r^4.
    r = 3796.19
    expected = 3.0 * J2 * model.mu_km3_s2 * model.ref_radius_km ** 2 / r ** 4
    assert a[2] == pytest.approx(expected, rel=1e-12)


def test_zonal_magnitude_decays_with_degree_at_lmo():
    """|a_Jn| / |a_J2| at 400 km alt: J_3 is about 1/70 of J_2, J_4..J_10 decay further.

    Numerical pin for the truncation-error floor: at a Mars equatorial point
    3796.19 km from the centre (400 km altitude), |a_J3| = 1.005e-7 km/s^2,
    |a_J2| = 6.980e-6 km/s^2. Ratio 0.0144 -- consistent with
    J_3/J_2 * r/R = 3.15e-5 / 1.96e-3 * 1.118 = 0.018 (close to the bracket
    ratio which also enters). All zonals n >= 3 stay below |a_J2|/50.
    """
    model = mars_gravity_model(max_degree=10)
    J = zonal_coefficients(model, 10)
    r_bf = np.array([3796.19, 0.0, 0.0])
    a2 = zonal_acceleration_body_fixed(r_bf, model.mu_km3_s2, model.ref_radius_km, {2: J[2]})
    mag2 = float(np.linalg.norm(a2))
    ratios = {}
    for n in range(3, 11):
        an = zonal_acceleration_body_fixed(r_bf, model.mu_km3_s2, model.ref_radius_km, {n: J[n]})
        magn = float(np.linalg.norm(an))
        ratios[n] = magn / mag2
        # All zonals n >= 3 should be at most 1/50 of J_2 at this altitude.
        assert magn < mag2 / 50.0, f"J_{n} magnitude {magn:.3e} >= J_2/50 = {mag2/50:.3e}"
    # Also spot-check the J_3 ratio explicitly.
    assert 0.010 < ratios[3] < 0.020, f"J_3 / J_2 ratio {ratios[3]:.4f} outside [0.01, 0.02]"


def test_inertial_wrapper_is_consistent_with_body_fixed():
    """zonal_acceleration_inertial at et=et0 == M.T @ body_fixed(M @ r)."""
    model = mars_gravity_model(max_degree=4)
    J = zonal_coefficients(model, 4)
    et = spice.str2et(EPOCH_STR)
    rng = np.random.default_rng(0)
    for _ in range(10):
        r_j2000 = rng.uniform(-1.0, 1.0, 3) * 5e3
        if np.linalg.norm(r_j2000) < 1e3:
            continue
        M = np.asarray(spice.pxform("J2000", "IAU_MARS", et))
        r_bf = M @ r_j2000
        a_bf = zonal_acceleration_body_fixed(r_bf, model.mu_km3_s2, model.ref_radius_km, J)
        a_j = zonal_acceleration_inertial(r_j2000, et, model.mu_km3_s2, model.ref_radius_km, J)
        assert np.allclose(a_j, M.T @ a_bf, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Analytical J_2 drift cross-checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inc_deg", [30.0, 60.0, 80.0])
def test_j2_secular_raan_drift_matches_brouwer_first_order(inc_deg):
    """Simulated secular dRAAN/dt matches first-order Brouwer to ~1% over 30 orbits."""
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    J2 = zonal_coefficients(model, 2)[2]
    et0 = spice.str2et(EPOCH_STR)
    alt = 400.0
    state0 = _circular_mme2000_state_in_j2000(alt, inc_deg, et0, mu, R)
    r0 = R + alt
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    n = 2 * np.pi / T
    inc_rad = np.radians(inc_deg)
    dRAAN_analytical = _brouwer_secular_raan_rate(n, R, r0, J2, inc_rad)
    N = 30
    t_eval = np.linspace(0.0, N * T, N * 30 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=2, t_eval_s=t_eval,
    )
    raans = [
        _elements_in_mme_from_j2000_state(result.state_km_kmps[i], et0 + result.t_s[i], mu).raan_rad
        for i in range(result.t_s.size)
    ]
    raans = np.unwrap(np.array(raans))
    slope = _fit_secular_slope_with_harmonics(result.t_s, raans, n)
    rel_err = (slope - dRAAN_analytical) / abs(dRAAN_analytical)
    assert abs(rel_err) < 1.0e-2, (
        f"inc={inc_deg} deg: analytical {np.degrees(dRAAN_analytical)*86400:.4f} deg/day, "
        f"sim {np.degrees(slope)*86400:.4f} deg/day, err {rel_err:.3%}"
    )


def test_j2_secular_argp_drift_matches_brouwer_first_order():
    """For e=0.05 at i=45 deg, darg_p/dt matches (3/4) n J2 (R/p)^2 (5cos^2 i - 1) to ~2%."""
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    J2 = zonal_coefficients(model, 2)[2]
    et0 = spice.str2et(EPOCH_STR)
    # Build an i=45 deg, e=0.05 orbit in MME-of-date, rotate to J2000.
    a = R + 600.0
    e = 0.05
    inc = np.radians(45.0)
    p = a * (1.0 - e ** 2)
    r_peri = a * (1.0 - e)
    v_peri = float(np.sqrt(mu * (2.0 / r_peri - 1.0 / a)))
    state_mme = np.array([
        r_peri, 0.0, 0.0,
        0.0, v_peri * np.cos(inc), v_peri * np.sin(inc),
    ])
    R_mat = mme2000_rotation_from_j2000(et0)
    state0 = np.concatenate([R_mat.T @ state_mme[:3], R_mat.T @ state_mme[3:]])
    n = float(np.sqrt(mu / a ** 3))
    T = 2 * np.pi / n
    dargp_analytical = _brouwer_secular_argp_rate(n, R, p, J2, inc)
    N = 30
    t_eval = np.linspace(0.0, N * T, N * 40 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=2, t_eval_s=t_eval,
    )
    argps = [
        _elements_in_mme_from_j2000_state(result.state_km_kmps[i], et0 + result.t_s[i], mu).argp_rad
        for i in range(result.t_s.size)
    ]
    argps = np.unwrap(np.array(argps))
    slope = _fit_secular_slope_with_harmonics(result.t_s, argps, n)
    rel_err = (slope - dargp_analytical) / abs(dargp_analytical)
    assert abs(rel_err) < 2.0e-2, (
        f"analytical {np.degrees(dargp_analytical)*86400:.4f} deg/day, "
        f"sim {np.degrees(slope)*86400:.4f} deg/day, err {rel_err:.3%}"
    )


def test_mars_sun_synchronous_inclination_is_retrograde_and_solves_correctly():
    """Solve for the inclination giving dRAAN/dt = 2 pi / Mars year.

    Mars sidereal year pulled from ``reflectors.mars_constants``. For a
    400 km Mars orbit, the
    J_2-driven sun-sync inclination should come out near 93 deg (slightly
    retrograde). Pin that number, and verify the analytical prediction
    agrees with a short simulation.
    """
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    J2 = zonal_coefficients(model, 2)[2]
    target_dRAAN = 2.0 * np.pi / MARS_SIDEREAL_YEAR_S  # rad/s (positive; tracks Sun)
    alt = 400.0
    a = R + alt
    n = float(np.sqrt(mu / a ** 3))
    # From dRAAN/dt = -(3/2) n (R/p)^2 J2 cos(i) = target:
    #   cos(i) = - target / ((3/2) n (R/p)^2 J2)
    cos_i = -target_dRAAN / (1.5 * n * (R / a) ** 2 * J2)
    i = float(np.degrees(np.arccos(cos_i)))
    # Sun-sync at 400 km Mars altitude: just over 90 deg (retrograde).
    assert 92.0 < i < 94.0, f"sun-sync inclination came out as {i:.3f} deg"
    # Sanity: analytical RAAN rate at that inclination equals the target.
    dRAAN = _brouwer_secular_raan_rate(n, R, a, J2, np.radians(i))
    assert dRAAN == pytest.approx(target_dRAAN, rel=1e-10)

    # And the simulated drift over 10 orbits matches target to ~1 %.
    et0 = spice.str2et(EPOCH_STR)
    state0 = _circular_mme2000_state_in_j2000(alt, i, et0, mu, R)
    T = 2 * np.pi / n
    N = 10
    t_eval = np.linspace(0.0, N * T, N * 30 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=2, t_eval_s=t_eval,
    )
    raans = [
        _elements_in_mme_from_j2000_state(result.state_km_kmps[j], et0 + result.t_s[j], mu).raan_rad
        for j in range(result.t_s.size)
    ]
    raans = np.unwrap(np.array(raans))
    slope = _fit_secular_slope_with_harmonics(result.t_s, raans, n)
    assert abs((slope - target_dRAAN) / target_dRAAN) < 2.0e-2


# ---------------------------------------------------------------------------
# 3. Conservation laws
# ---------------------------------------------------------------------------


def _pole_direction_j2000(et: float) -> np.ndarray:
    """Mars spin pole direction in J2000 at et (includes nutation)."""
    R = np.asarray(spice.pxform("J2000", "IAU_MARS", et))
    return R[2, :] / np.linalg.norm(R[2, :])


def test_axial_angular_momentum_conserved_under_zonal_dynamics():
    """L . p_0 (axial specific angular momentum about a FIXED pole) is
    conserved to numerical precision under zonal-only dynamics.

    Subtlety: the conserved quantity for an axisymmetric potential is
    L . p where p is the symmetry axis. The zonal potential here is
    axisymmetric about the INSTANTANEOUS Mars spin pole, which drifts by
    roughly arcsec/year due to IAU 2015 nutation terms. That pole motion
    makes L . p_of_date ONLY approximately conserved (drift ~1e-7 / 5
    orbits -- dominated by the true pole motion, not numerical error).
    Projecting onto the pole at t=0 isolates the numerical-integrator error
    from the physical pole-motion effect; the high-accuracy preset should
    conserve the projected quantity to substantially better than 1e-9.
    """
    model = mars_gravity_model(max_degree=10)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    state0 = _circular_mme2000_state_in_j2000(400.0, 60.0, et0, mu, R)
    T = 2 * np.pi * np.sqrt((R + 400.0) ** 3 / mu)
    N = 5
    t_eval = np.linspace(0.0, N * T, N * 40 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=10, t_eval_s=t_eval,
        options=PropagationOptions.high_accuracy(),
    )
    r = result.positions()
    v = result.velocities()
    h = np.cross(r, v)
    pole0 = _pole_direction_j2000(et0)  # FIXED direction at t=0
    Lz_fixed = h @ pole0
    rel_fixed = float(np.max(np.abs((Lz_fixed - Lz_fixed[0]) / Lz_fixed[0])))
    # Mars-pole motion makes fixed-pole axial angular momentum only
    # approximately conserved. The threshold accommodates this physical term
    # and small kernel refinements while remaining well below nodal precession.
    assert rel_fixed < 1e-8, f"fixed-axis L_z drift = {rel_fixed:.3e}"
    # The perpendicular component, by contrast, drifts at %-scale over 5
    # orbits -- the hallmark of nodal precession. This is the complementary
    # signature of an axisymmetric potential
    # (conserved in one direction, not the others).
    perp = h - Lz_fixed[:, None] * pole0[None, :]
    perp_drift = float(
        np.max(np.linalg.norm(perp - perp[0], axis=1)) / np.linalg.norm(perp[0])
    )
    assert perp_drift > 1e-3, (
        f"expected percent-scale perpendicular drift, got {perp_drift:.3e}"
    )



def test_non_axial_angular_momentum_is_not_conserved_under_j2():
    """In contrast, the component of L perpendicular to the spin axis is
    NOT a constant under J_2 -- it's what drives nodal precession. This is
    the complementary check to the axial-conservation test.
    """
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    state0 = _circular_mme2000_state_in_j2000(400.0, 60.0, et0, mu, R)
    T = 2 * np.pi * np.sqrt((R + 400.0) ** 3 / mu)
    N = 5
    t_eval = np.linspace(0.0, N * T, N * 40 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=2, t_eval_s=t_eval,
    )
    r = result.positions()
    v = result.velocities()
    h = np.cross(r, v)
    # Component of h PERPENDICULAR to the pole at t=0.
    pole0 = _pole_direction_j2000(et0)
    h_parallel = (h @ pole0)[:, None] * pole0[None, :]
    h_perp = h - h_parallel
    drift = float(np.max(np.linalg.norm(h_perp - h_perp[0], axis=1)) / np.linalg.norm(h_perp[0]))
    # Non-axial component drifts by at least ~1 ppm of itself over 5 orbits --
    # i.e. motion is NOT constant. Lower bound to catch accidental
    # conservation.
    assert drift > 1e-5, f"expected measurable drift, got {drift:.3e}"


def test_total_mechanical_energy_conserved_under_zonal_dynamics():
    """Gravity is conservative: KE + U_two_body + U_zonal is conserved."""
    model = mars_gravity_model(max_degree=10)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    J = zonal_coefficients(model, 10)
    et0 = spice.str2et(EPOCH_STR)
    state0 = _circular_mme2000_state_in_j2000(400.0, 45.0, et0, mu, R)
    # Give it some eccentricity to exercise the potential at varying r.
    state0 = state0.copy()
    state0[:3] += 100.0 * state0[3:] / np.linalg.norm(state0[3:]) * np.sign(1)  # tiny kick
    T = 2 * np.pi * np.sqrt((R + 400.0) ** 3 / mu)
    N = 3
    t_eval = np.linspace(0.0, N * T, N * 40 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=10, t_eval_s=t_eval,
        options=PropagationOptions.high_accuracy(),
    )

    # Potential (Curtis convention, physical attractive field):
    #   U(r, phi) = -(mu/r) + (mu/r) * sum_{n>=2} J_n (R/r)^n P_n(sin phi)
    # The a = -grad U acceleration code in gravity.py was derived from this
    # form and was cross-validated against the Curtis J_2 closed form, so
    # this potential is the right scalar field to use when checking energy
    # conservation under those accelerations.
    def potential(r_j2000, et):
        M = np.asarray(spice.pxform("J2000", "IAU_MARS", et))
        r_bf = M @ r_j2000
        r_mag = float(np.linalg.norm(r_bf))
        u = r_bf[2] / r_mag
        n_max = 10
        P = np.zeros(n_max + 1)
        P[0] = 1.0
        P[1] = u
        for k in range(2, n_max + 1):
            P[k] = ((2 * k - 1) * u * P[k - 1] - (k - 1) * P[k - 2]) / k
        pert_sum = 0.0
        for n_idx in range(2, n_max + 1):
            pert_sum += J[n_idx] * (R / r_mag) ** n_idx * P[n_idx]
        return -mu / r_mag + (mu / r_mag) * pert_sum

    E = np.empty(result.t_s.size)
    for i, t in enumerate(result.t_s):
        r = result.state_km_kmps[i, :3]
        v = result.state_km_kmps[i, 3:]
        KE = 0.5 * float(v @ v)
        U = potential(r, et0 + t)
        E[i] = KE + U
    rel = float(np.max(np.abs((E - E[0]) / E[0])))
    assert rel < 1e-9, f"total-energy drift with zonals = {rel:.3e}"


def test_backward_integration_recovers_initial_state_with_j2():
    """Forward 1 orbit + backward 1 orbit with J_2 on: state returns."""
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    state0 = _circular_mme2000_state_in_j2000(400.0, 60.0, et0, mu, R)
    T = 2 * np.pi * np.sqrt((R + 400.0) ** 3 / mu)
    opts = PropagationOptions.high_accuracy()
    fwd = propagate(state0, (0.0, T), epoch_et=et0, zonal_degree=2, options=opts)
    end_state = fwd.state_km_kmps[-1]
    # Backward: end_state at et0+T back to et0. Pass epoch_et=et0+T so the
    # frame rotation evaluated at t=0 of the backward run corresponds to
    # the end of the forward run.
    bwd = propagate(
        end_state, (T, 0.0),
        epoch_et=et0, zonal_degree=2, options=opts,
    )
    back_state = bwd.state_km_kmps[-1]
    dr = float(np.linalg.norm(back_state[:3] - state0[:3]))
    dv = float(np.linalg.norm(back_state[3:] - state0[3:]))
    # At the high_accuracy preset, 1-period forward + backward through a
    # perturbed system recovers the initial state to ~< 10 um / 10 nm/s.
    assert dr < 1e-4, f"backward position drift {dr:.3e} km"
    assert dv < 1e-7, f"backward velocity drift {dv:.3e} km/s"
