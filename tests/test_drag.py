"""Tests for ``reflectors.drag`` (flat-plate atmospheric drag).

Drag is NON-conservative (velocity-dependent), so the finite-difference-of-
potential cross-check used for gravity/third-body does NOT apply. The
independent validation is the King-Hele (1964, Ch.4 §18) circular-orbit
contraction ``da/dt = -(C_D A/m) rho sqrt(mu a)``, which is derivable from the
Gauss energy equation ``da/dt = (2 a^2 / mu) (a_drag . v)`` -- a different path
from the Cartesian drag core, so their agreement is a genuine check.

Groups: limits/identities, magnitude anchor, King-Hele (fast single-state +
slow multi-orbit), co-rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.atmosphere import ExponentialAtmosphere
from reflectors.drag import (
    atmosphere_relative_velocity,
    drag_acceleration,
)
from reflectors.dynamics import body_gm_km3_per_s2

R_EQ_EARTH = 6378.137
C_D = 2.2
AREA = 1000.0
MASS = 18.0  # sigma=18 g/m^2 sail


# ---------------------------------------------------------------------------
# Group 1 -- limits & identities (fast)
# ---------------------------------------------------------------------------


def test_zero_density_gives_zero_drag():
    a = drag_acceleration(np.array([7.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                          0.0, C_D, AREA, MASS)
    assert np.array_equal(a, np.zeros(3))


def test_zero_relative_speed_gives_zero_drag():
    a = drag_acceleration(np.zeros(3), np.array([1.0, 0.0, 0.0]),
                          1e-11, C_D, AREA, MASS)
    assert np.array_equal(a, np.zeros(3))


def test_edge_on_sail_gives_zero_drag():
    # n perpendicular to v_rel -> projected area 0 -> no drag (the attitude
    # coupling; analogue of SRP's edge-on -> zero).
    a = drag_acceleration(np.array([7.5, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                          1e-11, C_D, AREA, MASS)
    assert np.array_equal(a, np.zeros(3))


def test_drag_is_anti_parallel_to_relative_velocity_no_lift():
    # For any non-edge-on orientation the v1 model is pure anti-velocity: a x v = 0
    # and a . v < 0. (Pins the "no lift/normal component" decision -- adding one
    # would break this and the test would flag it.)
    v_rel = np.array([6.0, 3.0, -2.0])
    n = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    a = drag_acceleration(v_rel, n, 1e-11, C_D, AREA, MASS)
    assert np.linalg.norm(np.cross(a, v_rel)) < 1e-18
    assert float(np.dot(a, v_rel)) < 0.0


def test_broadside_magnitude_matches_formula():
    # |a| = 1/2 C_D (A/m) rho |v|^2, broadside (A_proj = A), in km/s^2.
    v = np.array([7.5, 0.0, 0.0])  # km/s
    rho = 1e-11
    a = drag_acceleration(v, np.array([1.0, 0.0, 0.0]), rho, C_D, AREA, MASS)
    v_mps = 7.5e3
    expected_mps2 = 0.5 * C_D * (AREA / MASS) * rho * v_mps ** 2
    assert np.linalg.norm(a) == pytest.approx(expected_mps2 * 1e-3, rel=1e-12)


def test_projected_area_cosine_scaling():
    # At 60 deg incidence the projected area is |cos60| = 1/2 of broadside.
    v = np.array([7.5, 0.0, 0.0])
    n60 = np.array([math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0])
    a_broad = drag_acceleration(v, np.array([1.0, 0.0, 0.0]), 1e-11, C_D, AREA, MASS)
    a_60 = drag_acceleration(v, n60, 1e-11, C_D, AREA, MASS)
    assert np.linalg.norm(a_60) == pytest.approx(0.5 * np.linalg.norm(a_broad), rel=1e-9)


# ---------------------------------------------------------------------------
# Group 2 -- King-Hele independent cross-check
# ---------------------------------------------------------------------------


def test_kinghele_circular_decay_single_state():
    """King-Hele (1964) Ch.4 §18: for a circular orbit, da/dt = -B rho sqrt(mu a),
    B = C_D A/m. Cross-check the drag core via the Gauss energy relation
    da/dt = (2 a^2 / mu)(a_drag . v) at one circular state (broadside sail,
    non-rotating atmosphere). The two expressions are independent derivations."""
    mu = body_gm_km3_per_s2(399)  # km^3/s^2
    a_km = R_EQ_EARTH + 500.0
    v_circ = math.sqrt(mu / a_km)
    r = np.array([a_km, 0.0, 0.0])
    v = np.array([0.0, v_circ, 0.0])  # circular, perpendicular to r
    n = v / np.linalg.norm(v)  # broadside (A_proj = A)
    rho = 1e-12

    a_drag = drag_acceleration(v, n, rho, C_D, AREA, MASS)  # km/s^2, co-rot off
    # Gauss energy relation (independent of the Cartesian core's internals):
    da_dt_gauss = (2.0 * a_km ** 2 / mu) * float(np.dot(a_drag, v))
    # King-Hele closed form (A_proj = A broadside):
    B = C_D * (AREA / MASS) * 1e-6  # m^2/kg -> km^2/kg so B*rho*sqrt(mu a) in km/s
    # rho is kg/m^3; B*rho has units km^2/kg * kg/m^3 = km^2/m^3. Keep SI instead:
    # do the whole comparison in km/s with explicit unit handling below.
    # da/dt [km/s] = -(C_D A/m)[m^2/kg] * rho[kg/m^3] * sqrt(mu a)[km^1.5/s/...]
    # Simplest: compare da_dt_gauss to the SI-consistent King-Hele value.
    B_si = C_D * (AREA / MASS)  # m^2/kg
    da_dt_kh_mps = -B_si * rho * math.sqrt((mu * 1e9) * (a_km * 1e3))  # m/s
    da_dt_kh_km = da_dt_kh_mps * 1e-3  # km/s
    assert da_dt_gauss == pytest.approx(da_dt_kh_km, rel=1e-9)
    assert da_dt_gauss < 0.0  # orbit contracts


@pytest.mark.slow
def test_kinghele_circular_decay_multi_orbit():
    """Propagate a circular orbit under drag-only (broadside, exponential
    atmosphere, non-rotating) and confirm (a) it DECAYS -- the positive signal a
    no-drag control cannot produce -- and (b) the measured <da/dt> matches the
    King-Hele closed form within a few percent."""
    mu = body_gm_km3_per_s2(399)
    a0 = R_EQ_EARTH + 600.0
    atm = ExponentialAtmosphere(rho0_kg_m3=1e-13, scale_height_km=60.0, h0_ref_km=600.0)

    def rhs(state):
        r, v = state[:3], state[3:]
        rmag = float(np.linalg.norm(r))
        alt = rmag - R_EQ_EARTH
        rho = atm.density_kg_m3(alt)
        n = v / float(np.linalg.norm(v))  # broadside
        a = -mu * r / rmag ** 3 + drag_acceleration(v, n, rho, C_D, AREA, MASS)
        return np.concatenate([v, a])

    v0 = math.sqrt(mu / a0)
    y = np.array([a0, 0.0, 0.0, 0.0, v0, 0.0])
    T = 2.0 * math.pi * math.sqrt(a0 ** 3 / mu)
    h = T / 400.0
    n_steps = int(3 * T / h)
    a_series = []
    for _ in range(n_steps):
        k1 = rhs(y); k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2); k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rmag = float(np.linalg.norm(y[:3]))
        vmag2 = float(np.dot(y[3:], y[3:]))
        a_series.append(1.0 / (2.0 / rmag - vmag2 / mu))  # vis-viva
    a_series = np.array(a_series)

    # (a) Positive signal: the orbit decays.
    assert a_series[-1] < a0 - 1.0  # a fell by > 1 km

    # (b) Measured <da/dt> vs King-Hele at the mean altitude.
    da_dt_measured = (a_series[-1] - a0) / (n_steps * h)  # km/s
    a_mean = 0.5 * (a0 + a_series[-1])
    rho_mean = atm.density_kg_m3(a_mean - R_EQ_EARTH)
    B_si = C_D * (AREA / MASS)
    da_dt_kh = -B_si * rho_mean * math.sqrt((mu * 1e9) * (a_mean * 1e3)) * 1e-3  # km/s
    assert da_dt_measured == pytest.approx(da_dt_kh, rel=0.03)


# ---------------------------------------------------------------------------
# Group 3 -- co-rotation (M&G Eq. 3.98)
# ---------------------------------------------------------------------------


def test_corotation_reduces_relative_speed_prograde_equatorial():
    from reflectors.ephemeris import utc_to_et

    et = utc_to_et("2028-01-01T00:00:00")
    r = np.array([7000.0, 0.0, 0.0])  # equatorial
    v = np.array([0.0, 7.5, 0.0])     # prograde
    v_rel = atmosphere_relative_velocity(r, v, et, "IAU_EARTH")
    # Co-rotation subtracts omega x r (~+y here) -> relative speed drops by ~omega*r.
    assert np.linalg.norm(v_rel) < np.linalg.norm(v)
    omega = 7.292115e-5  # rad/s
    assert np.linalg.norm(v) - np.linalg.norm(v_rel) == pytest.approx(
        omega * 7000.0, rel=0.05
    )


# ---------------------------------------------------------------------------
# Group 4 -- propagator plumbing (the escape RHS drag hook)
# ---------------------------------------------------------------------------


def test_drag_hook_removes_energy_through_propagate_escape():
    """End-to-end: with drag_force_fn wired in, a low Earth orbit loses energy
    (a falls); without it, a stays within its J2 oscillation. force_coast (no
    SRP) + no third bodies isolates gravity+drag, so the only difference is the
    drag hook -- the positive signal that drag is summed into the RHS."""
    from reflectors.atmosphere import HarrisPriester
    from reflectors.attitude_control import AttitudeLimits
    from reflectors.central_body import earth_central_body
    from reflectors.drag import make_drag_force_fn
    from reflectors.ephemeris import utc_to_et
    from reflectors.escape import propagate_escape
    from reflectors.qlaw import QLawParams
    from reflectors.sail_designs import make_canonical_sail

    earth = earth_central_body()
    et = utc_to_et("2028-01-01T00:00:00")
    mu = earth.mu_km3_s2
    a0 = earth.equatorial_radius_km + 500.0
    state0 = np.array([a0, 0.0, 0.0, 0.0, math.sqrt(mu / a0), 0.0])
    sail = make_canonical_sail(0.018)
    limits = AttitudeLimits()
    shell = QLawParams(a_target_km=earth.hill_radius_km,
                       rp_min_km=earth.equatorial_radius_km + 100.0)
    span = 0.2 * 86400.0  # ~3 revs at 500 km
    common = dict(gravity_degree=2, central_body=earth, third_bodies=(),
                  force_coast=True, max_step_s=20.0)

    res_nodrag = propagate_escape(state0, et, sail, shell, limits, (0.0, span), **common)
    drag_fn = make_drag_force_fn(sail, HarrisPriester(), central_body=earth)
    res_drag = propagate_escape(state0, et, sail, shell, limits, (0.0, span),
                                drag_force_fn=drag_fn, **common)

    def a_end(res):
        r = res.orbit_state_km_kmps[-1, :3]
        v = res.orbit_state_km_kmps[-1, 3:]
        return 1.0 / (2.0 / np.linalg.norm(r) - np.dot(v, v) / mu)

    assert res_drag.metadata["drag"] is True
    assert res_nodrag.metadata["drag"] is False
    # Drag removes energy: a falls well beyond the J2 osculating oscillation.
    assert a_end(res_drag) < a_end(res_nodrag) - 10.0
