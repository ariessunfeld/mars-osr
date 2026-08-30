"""Fast tests for Gauss's variational equations (``reflectors.gauss``).

Independent cross-checks (each physics term is validated against a check
that does not share an implementation with it):

  1. ``osculating_elements`` vs ``elements.classical_elements`` (the SPICE
     ``spice.oscltx`` path) -- the two element extractors must agree.
  2. ``gauss_variational_rates`` ``da/dt`` vs the independent energy identity
     ``da/dt = (2 a^2 / mu)(v . f)``.
  3. ``gauss_variational_rates`` vs a central finite-difference of
     ``classical_elements`` along a short two-body + constant-thrust
     propagation -- the end-to-end check.
  4. ``semimajor_axis_rate_max`` / ``eccentricity_rate_max`` vs a brute-force
     maximisation of ``gauss_variational_rates`` over thrust direction and
     true anomaly.
  5. Near-circular regularisation: finite, correctly-signed ``de/dt`` at e ~ 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from reflectors.dynamics import mars_gm_km3_per_s2
from reflectors.elements import ClassicalElements, classical_elements
from reflectors.gauss import (
    eccentricity_rate_max,
    gauss_variational_rates,
    osculating_elements,
    rtn_basis,
    semimajor_axis_rate_max,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_states() -> list[tuple[np.ndarray, np.ndarray, str]]:
    """A spread of bound Cartesian states (r km, v km/s) for cross-checks."""
    mu = mars_gm_km3_per_s2()
    states = []
    # Circular-ish equatorial.
    r = np.array([4000.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(mu / 4000.0), 0.0])
    states.append((r, v, "circular_equatorial"))
    # Eccentric, inclined, generic orientation.
    r = np.array([5000.0, 1500.0, 2200.0])
    v = np.array([-0.4, 2.6, 1.1])
    states.append((r, v, "eccentric_inclined"))
    # Moderately eccentric, retrograde-ish.
    r = np.array([3900.0, -800.0, 600.0])
    v = np.array([0.9, 2.9, -0.7])
    states.append((r, v, "eccentric_2"))
    return states


def _two_body_plus_thrust_rhs(mu: float, f_inertial: np.ndarray):
    """RHS closure dy/dt = [v, -mu r/|r|^3 + f] for an inertially-fixed f."""

    def rhs(t, y):
        r = y[:3]
        v = y[3:]
        a = -mu * r / np.linalg.norm(r) ** 3 + f_inertial
        return np.concatenate([v, a])

    return rhs


# ---------------------------------------------------------------------------
# Cross-check 1 -- osculating_elements vs spice.oscltx
# ---------------------------------------------------------------------------


def test_osculating_elements_matches_spice_oscltx():
    """The numpy element extractor agrees with ``classical_elements``
    (``spice.oscltx``) on a, e, i and the angles for generic orbits."""
    mu = mars_gm_km3_per_s2()
    for r, v, name in _sample_states():
        state = np.concatenate([r, v])
        spice_el = classical_elements(state, mu, epoch_et=0.0)
        numpy_el = osculating_elements(r, v, mu)
        assert numpy_el.a_km == pytest.approx(spice_el.a_km, rel=1e-9), name
        assert numpy_el.e == pytest.approx(spice_el.e, rel=1e-9, abs=1e-12), name
        assert numpy_el.inclination_rad == pytest.approx(
            spice_el.inclination_rad, rel=1e-9, abs=1e-12
        ), name
        # Angles: compare via sin/cos to be wrap-safe.
        for computed_angle, reference_angle, label in (
            (numpy_el.raan_rad, spice_el.raan_rad, "raan"),
            (numpy_el.argp_rad, spice_el.argp_rad, "argp"),
            (numpy_el.nu_rad, spice_el.nu_rad, "nu"),
        ):
            assert math.cos(computed_angle) == pytest.approx(
                math.cos(reference_angle), abs=1e-8
            ), f"{name}/{label}"
            assert math.sin(computed_angle) == pytest.approx(
                math.sin(reference_angle), abs=1e-8
            ), f"{name}/{label}"


# ---------------------------------------------------------------------------
# Cross-check 2 -- da/dt vs the energy identity
# ---------------------------------------------------------------------------


def test_da_dt_matches_energy_identity():
    """``da/dt`` from Gauss equals ``(2 a^2 / mu)(v . f)`` -- the specific-
    orbital-energy identity, an independent derivation."""
    mu = mars_gm_km3_per_s2()
    rng = np.random.default_rng(20260519)
    for r, v, name in _sample_states():
        el = osculating_elements(r, v, mu)
        r_hat, theta_hat, h_hat = rtn_basis(r, v)
        for _ in range(5):
            f_inertial = rng.normal(size=3) * 1.0e-6  # km/s^2
            f_r = float(np.dot(f_inertial, r_hat))
            f_t = float(np.dot(f_inertial, theta_hat))
            f_h = float(np.dot(f_inertial, h_hat))
            rates = gauss_variational_rates(el, f_r, f_t, f_h)
            da_energy = (2.0 * el.a_km ** 2 / mu) * float(np.dot(v, f_inertial))
            assert rates.da_dt_km_s == pytest.approx(
                da_energy, rel=1e-10, abs=1e-18
            ), name


# ---------------------------------------------------------------------------
# Cross-check 3 -- Gauss rates vs finite-difference of classical_elements
# ---------------------------------------------------------------------------


def test_gauss_rates_match_finite_difference():
    """Element rates from Gauss match a central finite-difference of
    ``classical_elements`` along a two-body + constant-thrust propagation."""
    mu = mars_gm_km3_per_s2()
    dt = 2.0  # s; central difference half-step
    for r, v, name in _sample_states():
        el0 = osculating_elements(r, v, mu)
        r_hat, theta_hat, h_hat = rtn_basis(r, v)
        # A thrust strong enough to dominate FD round-off, small enough that
        # the rate is ~constant over +/- dt.
        f_inertial = np.array([3.0e-6, -2.0e-6, 1.5e-6])  # km/s^2
        f_r = float(np.dot(f_inertial, r_hat))
        f_t = float(np.dot(f_inertial, theta_hat))
        f_h = float(np.dot(f_inertial, h_hat))
        rates = gauss_variational_rates(el0, f_r, f_t, f_h)

        rhs = _two_body_plus_thrust_rhs(mu, f_inertial)
        y0 = np.concatenate([r, v])
        sol_fwd = solve_ivp(
            rhs, (0.0, dt), y0, method="DOP853", rtol=1e-13, atol=1e-14
        )
        sol_bwd = solve_ivp(
            rhs, (0.0, -dt), y0, method="DOP853", rtol=1e-13, atol=1e-14
        )
        el_fwd = classical_elements(sol_fwd.y[:, -1], mu, 0.0)
        el_bwd = classical_elements(sol_bwd.y[:, -1], mu, 0.0)

        # da/dt is well-defined for every orbit.
        da_fd = (el_fwd.a_km - el_bwd.a_km) / (2.0 * dt)
        assert rates.da_dt_km_s == pytest.approx(da_fd, rel=1e-5), name

        # Scalar e and i have a corner at 0 (|e_vec|, |i| fold), so a
        # finite-difference of the SCALAR element across e~0 / i~0 is
        # ill-defined -- skip de/di/dnu there (the escape launch orbit sits
        # in exactly that regularised regime; see the module docstring).
        if el0.e > 1.0e-3:
            de_fd = (el_fwd.e - el_bwd.e) / (2.0 * dt)
            assert rates.de_dt_per_s == pytest.approx(
                de_fd, rel=1e-5, abs=1e-13
            ), name
            dnu = el_fwd.nu_rad - el_bwd.nu_rad
            dnu = (dnu + math.pi) % (2.0 * math.pi) - math.pi
            dnu_fd = dnu / (2.0 * dt)
            assert rates.dnu_dt_rad_s == pytest.approx(dnu_fd, rel=1e-4), name
        if el0.inclination_rad > 1.0e-3:
            di_fd = (
                el_fwd.inclination_rad - el_bwd.inclination_rad
            ) / (2.0 * dt)
            assert rates.di_dt_rad_s == pytest.approx(
                di_fd, rel=1e-5, abs=1e-13
            ), name


# ---------------------------------------------------------------------------
# Cross-check 4 -- a_dot_xx / e_dot_xx vs brute-force maximisation
# ---------------------------------------------------------------------------


def _brute_force_rate_max(a, e, mu, f_mag, which):
    """Max |da/dt| (which='a') or |de/dt| (which='e') over thrust direction
    and true anomaly, evaluated directly through gauss_variational_rates."""
    best = 0.0
    # Fibonacci sphere of thrust directions.
    n_dir = 400
    ga = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []
    for k in range(n_dir):
        z = 1.0 - 2.0 * (k + 0.5) / n_dir
        rad = math.sqrt(max(0.0, 1.0 - z * z))
        phi = ga * k
        dirs.append((rad * math.cos(phi), rad * math.sin(phi), z))
    for j in range(360):
        nu = 2.0 * math.pi * j / 360.0
        el = ClassicalElements(a, e, 0.5, 0.0, 0.0, nu, 1.0, mu)
        for (dr, dt_, dh) in dirs:
            rates = gauss_variational_rates(
                el, f_mag * dr, f_mag * dt_, f_mag * dh
            )
            val = abs(rates.da_dt_km_s) if which == "a" else abs(rates.de_dt_per_s)
            if val > best:
                best = val
    return best


def test_semimajor_axis_rate_max_matches_brute_force():
    mu = mars_gm_km3_per_s2()
    f_mag = 1.0e-6
    for a, e in ((4000.0, 0.0), (5000.0, 0.3), (6000.0, 0.6)):
        closed = semimajor_axis_rate_max(a, e, f_mag, mu)
        brute = _brute_force_rate_max(a, e, mu, f_mag, "a")
        assert closed == pytest.approx(brute, rel=2e-3), f"a={a}, e={e}"


def test_eccentricity_rate_max_matches_brute_force():
    mu = mars_gm_km3_per_s2()
    f_mag = 1.0e-6
    for a, e in ((4000.0, 0.05), (5000.0, 0.3), (6000.0, 0.6)):
        closed = eccentricity_rate_max(a, e, f_mag, mu)
        brute = _brute_force_rate_max(a, e, mu, f_mag, "e")
        assert closed == pytest.approx(brute, rel=2e-3), f"a={a}, e={e}"


# ---------------------------------------------------------------------------
# Cross-check 5 -- near-circular regularisation
# ---------------------------------------------------------------------------


def test_near_circular_de_dt_is_finite_and_transverse_thrust_grows_e():
    """At e ~ 0 the regularised nu=0 gives a finite de/dt, and a transverse
    thrust grows eccentricity (de/dt > 0)."""
    mu = mars_gm_km3_per_s2()
    r = np.array([4000.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(mu / 4000.0), 0.0])
    el = osculating_elements(r, v, mu)
    assert el.e < 1e-9  # genuinely circular
    rates = gauss_variational_rates(el, 0.0, 1.0e-6, 0.0)
    assert math.isfinite(rates.de_dt_per_s)
    assert math.isfinite(rates.da_dt_km_s)
    assert rates.de_dt_per_s > 0.0
    assert rates.da_dt_km_s > 0.0  # transverse thrust also raises a


def test_transverse_thrust_raises_semimajor_axis():
    """Sign convention: a +theta_hat (along-velocity) thrust gives da/dt > 0
    at every sampled orbit -- the orbit-raising sense the escape law uses."""
    mu = mars_gm_km3_per_s2()
    for r, v, name in _sample_states():
        el = osculating_elements(r, v, mu)
        rates = gauss_variational_rates(el, 0.0, 1.0e-6, 0.0)
        assert rates.da_dt_km_s > 0.0, name
