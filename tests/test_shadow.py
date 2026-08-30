"""Fast tests for the Mars-umbra shadow model.

Three groups:

  1. Physics anchors -- ``R_Sun`` from the PCK matches the IAU 2015
     nominal value; the predicate ``in_mars_umbra`` and the scalar
     ``shadow_factor`` agree and have the expected types.

  2. Geometric limits -- sails on the sub-solar / anti-solar / tangential
     sides of Mars land in the expected shadow state across a range of
     altitudes; the analytical umbra-cone length is reproduced; crossing
     the cone apex flips the predicate as the angular-radii inequality
     predicts.

  3. Orbital integration -- the fraction of a circular in-plane orbit
     that lies inside the umbra matches the analytical
     ``arcsin(R_Mars/R_sat) / pi`` relation (the finite sail-Sun
     distance and the small ``sigma_Sun`` are both negligible at LMO).
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice

from reflectors.ephemeris import utc_to_et
from reflectors.shadow import (
    in_mars_umbra,
    shadow_factor,
    sun_radius_km,
    umbra_cone_length_km,
)
from reflectors.surface import mars_equatorial_radius_km


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Module-scoped SPICE-derived test anchors
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def epoch_et() -> float:
    return utc_to_et(EPOCH_STR)


@pytest.fixture(scope="module")
def sun_hat_j2000(epoch_et) -> np.ndarray:
    """Unit vector from Mars centre toward the Sun in J2000 at the test epoch."""
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    r_sun = np.asarray(state[:3], dtype=float)
    return r_sun / np.linalg.norm(r_sun)


@pytest.fixture(scope="module")
def mars_sun_distance_km(epoch_et) -> float:
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    return float(np.linalg.norm(state[:3]))


def _unit_perp_to(v: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to ``v``. Used for tangential-geometry tests."""
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(tmp @ v) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    perp = tmp - (tmp @ v) * v
    return perp / np.linalg.norm(perp)


# ---------------------------------------------------------------------------
# Group 1: physics anchors
# ---------------------------------------------------------------------------


def test_sun_radius_matches_iau_2015_nominal():
    """``sun_radius_km`` reads 6.957e5 km from BODY10_RADII.

    Matches the IAU 2015 Resolution B3 nominal solar radius
    ``R_Sun_N = 6.957e5 km`` (Mamajek et al. 2015 / Prsa et al. 2016,
    AJ 152:41). Pinned against the literature value so a PCK
    drift triggers an explicit test failure.
    """
    IAU_2015_NOMINAL_KM = 6.957e5
    assert abs(sun_radius_km() - IAU_2015_NOMINAL_KM) < 1.0


def test_shadow_factor_consistent_with_in_mars_umbra_and_returns_float(
    epoch_et, sun_hat_j2000
):
    """``shadow_factor`` is 0.0 iff ``in_mars_umbra`` is True, else 1.0."""
    R_sat = mars_equatorial_radius_km() + 400.0

    r_lit = R_sat * sun_hat_j2000
    assert in_mars_umbra(r_lit, epoch_et) is False
    sf_lit = shadow_factor(r_lit, epoch_et)
    assert sf_lit == 1.0
    assert isinstance(sf_lit, float)

    r_dark = -R_sat * sun_hat_j2000
    assert in_mars_umbra(r_dark, epoch_et) is True
    sf_dark = shadow_factor(r_dark, epoch_et)
    assert sf_dark == 0.0
    assert isinstance(sf_dark, float)


# ---------------------------------------------------------------------------
# Group 2: geometric limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alt_km", [100.0, 200.0, 400.0, 1000.0, 5000.0])
def test_sub_solar_sail_is_never_in_umbra(epoch_et, sun_hat_j2000, alt_km):
    """A sail on the sub-solar side of Mars at any altitude is not shadowed.

    ``D`` (angular separation of Mars and Sun centres as seen from the
    sail) is ~pi here because Mars is on the opposite side of the sail
    from the Sun. ``D >> sigma_Mars + sigma_Sun`` at all altitudes.
    """
    R_sat = mars_equatorial_radius_km() + alt_km
    r_sat = R_sat * sun_hat_j2000
    assert not in_mars_umbra(r_sat, epoch_et)


@pytest.mark.parametrize("alt_km", [100.0, 200.0, 400.0, 1000.0])
def test_anti_solar_sail_at_lmo_is_in_umbra(epoch_et, sun_hat_j2000, alt_km):
    """A sail on the anti-solar axis at low Mars altitude is in umbra.

    The umbra cone is ~1.12e6 km long at 1.524 AU; any LMO sail on the
    anti-solar axis is very deep in the full-occultation regime.
    """
    R_sat = mars_equatorial_radius_km() + alt_km
    r_sat = -R_sat * sun_hat_j2000
    assert in_mars_umbra(r_sat, epoch_et)


def test_tangential_sail_is_not_in_umbra(epoch_et, sun_hat_j2000):
    """A sail perpendicular to the Mars-Sun line at LMO is not in umbra.

    At 400 km altitude ``sigma_Mars ~ 1.11 rad (63.5 deg)``, and
    ``D ~ pi/2`` for the tangential position, so ``D > sigma_Mars`` and
    the total-eclipse condition fails.
    """
    R_sat = mars_equatorial_radius_km() + 400.0
    perp = _unit_perp_to(sun_hat_j2000)
    r_sat = R_sat * perp
    assert not in_mars_umbra(r_sat, epoch_et)


def test_umbra_cone_length_matches_analytical_relation(
    epoch_et, mars_sun_distance_km
):
    """``L_umbra = R_Mars * d_{Mars,Sun} / (R_Sun - R_Mars)`` to machine precision."""
    R_M = mars_equatorial_radius_km()
    R_S = sun_radius_km()
    expected = R_M * mars_sun_distance_km / (R_S - R_M)
    actual = umbra_cone_length_km(epoch_et)
    assert abs(actual - expected) / expected < 1e-14
    # Order-of-magnitude sanity: Mars at ~1.5 AU gives L ~ 1.12e6 km.
    assert 1.0e6 < actual < 1.3e6


def test_just_inside_umbra_apex_is_in_umbra(epoch_et, sun_hat_j2000):
    """A sail on the anti-Sun axis at 0.95 L_umbra is inside the umbra.

    At this distance ``sigma_Mars > sigma_Sun`` (Mars disc still larger
    than the Sun disc as seen from the sail) and ``D == 0`` on-axis, so
    the total-eclipse condition ``D + sigma_Sun <= sigma_Mars`` holds.
    """
    L = umbra_cone_length_km(epoch_et)
    r_sat = -0.95 * L * sun_hat_j2000
    assert in_mars_umbra(r_sat, epoch_et)


def test_just_past_umbra_apex_is_not_in_umbra(epoch_et, sun_hat_j2000):
    """A sail on the anti-Sun axis at 1.05 L_umbra is outside the umbra.

    Past the apex the Sun's angular disc exceeds Mars's, so from the
    sail's perspective the Mars shadow is now annular-penumbral -- a
    sun-ring is visible around Mars and the TOTAL-eclipse condition
    fails. This is the geometric signature of the umbra tip.
    """
    L = umbra_cone_length_km(epoch_et)
    r_sat = -1.05 * L * sun_hat_j2000
    assert not in_mars_umbra(r_sat, epoch_et)


def test_sail_inside_mars_reference_sphere_raises(epoch_et):
    """Sail position inside R_Mars,eq is a propagation pathology -> ValueError."""
    R_M = mars_equatorial_radius_km()
    r_sat = np.array([0.5 * R_M, 0.0, 0.0])
    with pytest.raises(ValueError, match="inside the Mars reference sphere"):
        in_mars_umbra(r_sat, epoch_et)


# ---------------------------------------------------------------------------
# Group 3: orbital integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alt_km", [200.0, 400.0, 1000.0])
def test_shadow_fraction_of_in_plane_circular_orbit(
    epoch_et, sun_hat_j2000, alt_km
):
    """Fraction of a circular in-plane orbit inside the umbra.

    For a circular orbit in the plane containing the Mars-Sun direction,
    the sail's angular position ``theta`` (measured from the sub-solar
    direction) enters the umbra when ``D(theta) <= sigma_Mars``, where
    ``D(theta) ~ pi - theta`` for a distant Sun. The umbra half-angle
    in ``theta`` is thus ``sigma_Mars = arcsin(R_Mars / R_sat)``, and the
    fraction of the orbit inside the umbra is ``sigma_Mars / pi``
    (neglecting the small ``sigma_Sun`` correction, which is
    ``<= 1e-4`` rad at LMO).

    A dense angular sweep must match the analytical fraction to within 0.5%.
    """
    R_sat = mars_equatorial_radius_km() + alt_km
    perp = _unit_perp_to(sun_hat_j2000)

    n_points = 4096
    thetas = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    positions = R_sat * (
        np.cos(thetas)[:, None] * sun_hat_j2000[None, :]
        + np.sin(thetas)[:, None] * perp[None, :]
    )
    in_umbra_count = sum(
        1 for pos in positions if in_mars_umbra(pos, epoch_et)
    )
    numerical_fraction = in_umbra_count / n_points
    analytical = np.arcsin(mars_equatorial_radius_km() / R_sat) / np.pi
    assert abs(numerical_fraction - analytical) < 5e-3


def test_in_plane_orbit_has_exactly_two_umbra_transitions(
    epoch_et, sun_hat_j2000
):
    """Sweeping around an in-plane circular LMO orbit crosses umbra twice.

    The umbra is a single connected region along the anti-Sun axis, so
    a full revolution has exactly one entry and one exit transition.
    """
    R_sat = mars_equatorial_radius_km() + 400.0
    perp = _unit_perp_to(sun_hat_j2000)
    n_points = 4096
    thetas = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    samples = []
    for th in thetas:
        pos = R_sat * (np.cos(th) * sun_hat_j2000 + np.sin(th) * perp)
        samples.append(in_mars_umbra(pos, epoch_et))
    # Wrap-around transition count = number of state flips over the loop.
    flips = sum(
        1 for i in range(n_points) if samples[i] != samples[(i + 1) % n_points]
    )
    assert flips == 2
