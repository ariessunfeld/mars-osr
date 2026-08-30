"""Fast tests for solar radiation pressure on a flat, non-perfect sail.

Organised in six groups:

  1. Dataclass contracts -- SailOptical / SolarSail field validation and
     the canonical factory returns (ideal, JPL square sail).
  2. Attitude callables -- fixed_j2000 and sun_pointing return
     unit vectors with the expected behaviour.
  3. One-sidedness and shadow gating -- cos alpha <= 0 or inside umbra
     gives exactly zero force, parameterized over several sail materials.
  4. Closed-form limits of the McInnes (1999) Eq. 2.57 optical force
     model -- ideal mirror, pure absorber, pure Lambertian, symmetric
     thermal. Assertions are in dimensionless ratios so they are
     independent of the chosen area/mass.
  5. Linearity / scaling -- force scales linearly with area and inversely
     with mass; solar flux falls off as 1/r^2 from the SPICE-sourced
     Sun-sail distance.
  6. Propagator plumbing -- ``propagate(solar_sail=..., sail_normal=...)``
     stacks additively with gravity and third bodies, emits the expected
     metadata, rejects the usual partial-arg mistakes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import spiceypy as spice

from reflectors.dynamics import (
    PropagationOptions,
    mars_gm_km3_per_s2,
    propagate,
)
from reflectors.ephemeris import utc_to_et
from reflectors.shadow import umbra_cone_length_km
from reflectors.solar_constants import solar_flux_at
from reflectors.attitude import fixed_j2000, sun_pointing
from reflectors.sail_designs import make_canonical_sail
from reflectors.srp import (
    SailOptical,
    SolarSail,
    SphericalParticle,
    TumbleAveragedSail,
    mcinnes_srp_acceleration,
    spherical_particle_acceleration,
    srp_acceleration,
    tumble_averaged_acceleration,
)
from reflectors.surface import mars_equatorial_radius_km


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Shared fixtures: a lit sail geometry, plus the SPICE anchors the physics
# relations reference.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def epoch_et() -> float:
    return utc_to_et(EPOCH_STR)


@pytest.fixture(scope="module")
def sun_state_km(epoch_et):
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    return np.asarray(state[:3], dtype=float)


@pytest.fixture(scope="module")
def sun_hat_j2000(sun_state_km):
    return sun_state_km / np.linalg.norm(sun_state_km)


@pytest.fixture(scope="module")
def sub_solar_lmo_position(sun_hat_j2000):
    """400 km altitude sail, sub-solar side: guaranteed lit."""
    R_sat = mars_equatorial_radius_km() + 400.0
    return R_sat * sun_hat_j2000


@pytest.fixture(scope="module")
def anti_solar_lmo_position(sun_hat_j2000):
    """400 km altitude sail, anti-solar side: in umbra."""
    R_sat = mars_equatorial_radius_km() + 400.0
    return -R_sat * sun_hat_j2000


def _unit_perp_to(v: np.ndarray) -> np.ndarray:
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(tmp @ v) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    perp = tmp - (tmp @ v) * v
    return perp / np.linalg.norm(perp)


def _solar_pressure_at_sail_pa(r_sat: np.ndarray, r_sun: np.ndarray) -> float:
    """P at the exact sail-to-Sun distance -- what the SRP code uses."""
    r_helio_km = float(np.linalg.norm(r_sun - r_sat))
    return solar_flux_at(r_helio_km)


def _test_sail(
    optical: SailOptical,
    area_m2: float = 1000.0,
    mass_kg: float = 50.0,
) -> SolarSail:
    """Ordinary sail wrapper for physics-relationship assertions.

    The default (1000 m^2 / 50 kg → sigma = 0.05) is a lightweight but
    not extreme representative sail. Tests that depend on a particular
    area/mass parameterize explicitly; the rest are dimensionless
    assertions that work for any ``_test_sail(...)``.

    Routed through ``make_canonical_sail`` so construction is centralized, but
    the (area, mass) signature is
    preserved because several scaling tests vary one or both
    independently.
    """
    return make_canonical_sail(
        mass_kg / area_m2, area_m2=area_m2, optical=optical,
    )


# ---------------------------------------------------------------------------
# Group 1 -- dataclass contracts
# ---------------------------------------------------------------------------


def test_sail_optical_ideal_factory_matches_mcinnes_row_1():
    """McInnes Table 2.1 'Ideal sail' row: rho=s=1, eps=0, B=2/3."""
    o = SailOptical.ideal()
    assert o.rho == 1.0
    assert o.s == 1.0
    assert o.eps_front == 0.0
    assert o.eps_back == 0.0
    assert o.B_front == pytest.approx(2.0 / 3.0)
    assert o.B_back == pytest.approx(2.0 / 3.0)


def test_sail_optical_square_sail_jpl_matches_mcinnes_table_2_1():
    """McInnes Table 2.1 'Square sail' row, cited to Wright (1992) App. A-B."""
    o = SailOptical.square_sail_jpl()
    assert o.rho == 0.88
    assert o.s == 0.94
    assert o.eps_front == 0.05
    assert o.eps_back == 0.55
    assert o.B_front == 0.79
    assert o.B_back == 0.55


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("rho", -0.1),
        ("rho", 1.1),
        ("s", -0.5),
        ("s", 2.0),
        ("eps_front", 1.2),
        ("eps_back", -0.01),
        ("B_front", 1.1),
        ("B_back", -1.0),
    ],
)
def test_sail_optical_rejects_out_of_range(field, bad_value):
    kwargs = dict(rho=0.5, s=0.5, eps_front=0.5, eps_back=0.5, B_front=0.5, B_back=0.5)
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=field):
        SailOptical(**kwargs)


@pytest.mark.parametrize("bad_area", [0.0, -1.0])
def test_solar_sail_rejects_nonpositive_area(bad_area):
    with pytest.raises(ValueError, match="area_m2"):
        SolarSail(area_m2=bad_area, mass_kg=10.0, optical=SailOptical.ideal())


@pytest.mark.parametrize("bad_mass", [0.0, -5.0])
def test_solar_sail_rejects_nonpositive_mass(bad_mass):
    with pytest.raises(ValueError, match="mass_kg"):
        SolarSail(area_m2=100.0, mass_kg=bad_mass, optical=SailOptical.ideal())


def test_solar_sail_loading_matches_mass_over_area():
    s = SolarSail(area_m2=800.0, mass_kg=40.0, optical=SailOptical.ideal())
    assert s.loading_kg_per_m2 == pytest.approx(40.0 / 800.0)


# ---------------------------------------------------------------------------
# Group 2 -- attitude callables
# ---------------------------------------------------------------------------


def test_fixed_j2000_returns_unit_vector_insensitive_to_args():
    n_raw = np.array([2.0, 0.0, 0.0])
    att = fixed_j2000(n_raw)
    for r, et in [(np.zeros(3), 0.0), (np.array([1e3, 2e3, 3e3]), 1e9)]:
        out = att(r, et)
        assert out == pytest.approx(np.array([1.0, 0.0, 0.0]))
        assert np.linalg.norm(out) == pytest.approx(1.0)


def test_fixed_j2000_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        fixed_j2000(np.zeros(3))


def test_sun_pointing_tracks_sail_to_sun_direction(
    epoch_et, sub_solar_lmo_position, sun_state_km
):
    att = sun_pointing()
    n = att(sub_solar_lmo_position, epoch_et)
    expected = (sun_state_km - sub_solar_lmo_position) / np.linalg.norm(
        sun_state_km - sub_solar_lmo_position
    )
    assert n == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert np.linalg.norm(n) == pytest.approx(1.0, rel=1e-14)


# ---------------------------------------------------------------------------
# Group 3 -- one-sidedness and shadow gating
# ---------------------------------------------------------------------------

_MATERIALS = [
    pytest.param(SailOptical.ideal(), id="ideal"),
    pytest.param(SailOptical.square_sail_jpl(), id="jpl_square"),
    pytest.param(
        SailOptical(
            rho=0.0, s=0.0,
            eps_front=0.5, eps_back=0.5,
            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
        ),
        id="absorber",
    ),
]


@pytest.mark.parametrize("optical", _MATERIALS)
def test_edge_on_sail_produces_zero_force(
    epoch_et, sub_solar_lmo_position, sun_hat_j2000, optical
):
    """``cos alpha = 0`` -> force = 0 (one-sided sail)."""
    perp = _unit_perp_to(sun_hat_j2000)
    a = srp_acceleration(
        sub_solar_lmo_position,
        epoch_et,
        _test_sail(optical),
        fixed_j2000(perp),
    )
    assert np.linalg.norm(a) == 0.0


@pytest.mark.parametrize("optical", _MATERIALS)
def test_back_lit_sail_produces_zero_force(
    epoch_et, sub_solar_lmo_position, sun_hat_j2000, optical
):
    """``cos alpha < 0`` -> force = 0 (one-sided sail)."""
    a = srp_acceleration(
        sub_solar_lmo_position,
        epoch_et,
        _test_sail(optical),
        fixed_j2000(-sun_hat_j2000),
    )
    assert np.linalg.norm(a) == 0.0


@pytest.mark.parametrize("optical", _MATERIALS)
def test_inside_umbra_produces_zero_force_regardless_of_attitude(
    epoch_et, anti_solar_lmo_position, sun_hat_j2000, optical
):
    """Shadow gate wins over everything: no photons, no thrust."""
    sail = _test_sail(optical)
    # Even a sun-pointing attitude (would otherwise give peak thrust) is 0.
    a = srp_acceleration(anti_solar_lmo_position, epoch_et, sail, sun_pointing())
    assert np.linalg.norm(a) == 0.0


# ---------------------------------------------------------------------------
# Group 4 -- closed-form limits
# ---------------------------------------------------------------------------


def test_ideal_mirror_face_on_gives_2PA_over_m_along_minus_s_hat(
    epoch_et, sub_solar_lmo_position, sun_state_km
):
    """McInnes Eq. 2.57 at rho=1, s=1, eps=0, alpha=0:
    a = -2 (P A / m) s_hat.  Pinned to <1e-12 relative."""
    sail = _test_sail(SailOptical.ideal())
    s_hat = (sun_state_km - sub_solar_lmo_position)
    s_hat = s_hat / np.linalg.norm(s_hat)
    att = fixed_j2000(s_hat)
    a = srp_acceleration(sub_solar_lmo_position, epoch_et, sail, att)
    P = _solar_pressure_at_sail_pa(sub_solar_lmo_position, sun_state_km)
    expected = -2.0 * (P * sail.area_m2 / sail.mass_kg) * 1e-3 * s_hat
    assert a == pytest.approx(expected, rel=1e-12, abs=1e-20)


def test_pure_absorber_face_on_gives_PA_over_m_along_minus_s_hat(
    epoch_et, sub_solar_lmo_position, sun_state_km
):
    """rho=0 at alpha=0:  a = -(P A / m) s_hat (pure photon drag)."""
    optical = SailOptical(
        rho=0.0, s=0.0,
        eps_front=0.5, eps_back=0.5,
        B_front=2.0 / 3.0, B_back=2.0 / 3.0,
    )
    sail = _test_sail(optical)
    s_hat = (sun_state_km - sub_solar_lmo_position)
    s_hat = s_hat / np.linalg.norm(s_hat)
    att = fixed_j2000(s_hat)
    a = srp_acceleration(sub_solar_lmo_position, epoch_et, sail, att)
    P = _solar_pressure_at_sail_pa(sub_solar_lmo_position, sun_state_km)
    expected = -(P * sail.area_m2 / sail.mass_kg) * 1e-3 * s_hat
    assert a == pytest.approx(expected, rel=1e-12, abs=1e-20)


def test_perfect_lambertian_face_on_gives_1_plus_B_f_times_PA_over_m(
    epoch_et, sub_solar_lmo_position, sun_state_km
):
    """rho=1, s=0 (fully diffuse reflection), eps=0:
    a = -(1 + B_f) (P A / m) s_hat.  For Lambertian B_f = 2/3: 5/3 P A /m."""
    B_f = 2.0 / 3.0
    optical = SailOptical(
        rho=1.0, s=0.0,
        eps_front=0.0, eps_back=0.0,
        B_front=B_f, B_back=B_f,
    )
    sail = _test_sail(optical)
    s_hat = (sun_state_km - sub_solar_lmo_position)
    s_hat = s_hat / np.linalg.norm(s_hat)
    a = srp_acceleration(
        sub_solar_lmo_position, epoch_et, sail, fixed_j2000(s_hat)
    )
    P = _solar_pressure_at_sail_pa(sub_solar_lmo_position, sun_state_km)
    expected = -(1.0 + B_f) * (P * sail.area_m2 / sail.mass_kg) * 1e-3 * s_hat
    assert a == pytest.approx(expected, rel=1e-12, abs=1e-20)


def test_thermal_term_vanishes_when_front_and_back_balanced():
    """McInnes Eq. 2.56: eps_f = eps_b AND B_f = B_b -> thermal contribution 0.

    The residual force at alpha=0 is then fully determined by rho and s and
    is independent of the (matched) emissivity/Lambertian values. Pinned
    here by comparing two sails that differ ONLY in emissivity magnitude
    (both symmetric): the resulting accelerations are identical.
    """
    et = utc_to_et(EPOCH_STR)
    R_sat = mars_equatorial_radius_km() + 400.0
    state, _ = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")
    r_sun = np.asarray(state[:3])
    s_hat = r_sun / np.linalg.norm(r_sun)
    r_sat = R_sat * s_hat

    def _make_sail(eps: float) -> SolarSail:
        return SolarSail(
            area_m2=1000.0,
            mass_kg=50.0,
            optical=SailOptical(
                rho=0.5, s=0.3, eps_front=eps, eps_back=eps,
                B_front=0.5, B_back=0.5,
            ),
        )

    a1 = srp_acceleration(r_sat, et, _make_sail(0.1), fixed_j2000(s_hat))
    a2 = srp_acceleration(r_sat, et, _make_sail(0.9), fixed_j2000(s_hat))
    # Both should match to machine precision; the thermal term
    # cancels whenever eps_f * B_f = eps_b * B_b.
    assert a1 == pytest.approx(a2, rel=1e-13)


def test_thermal_term_vanishes_when_both_emissivities_zero():
    """eps_f + eps_b = 0 exercises the module's explicit 0/0 guard.

    With alpha=0, n_hat = s_hat, force is purely along -s_hat. The
    closed-form coefficient from the module docstring is

        (1 - rho*s) [absorption/tangential residual]
        + 2*rho*s*cos(alpha=0) [specular reflection]
        + B_f*(1-s)*rho       [diffuse reflection, Lambertian]
        + (1-rho)*0           [thermal, zero here by eps construction]
        = 1 + rho*s + B_f*(1-s)*rho.
    """
    et = utc_to_et(EPOCH_STR)
    R_sat = mars_equatorial_radius_km() + 400.0
    state, _ = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")
    s_hat = np.asarray(state[:3]) / np.linalg.norm(state[:3])
    r_sat = R_sat * s_hat
    optical = SailOptical(
        rho=0.5, s=0.3, eps_front=0.0, eps_back=0.0,
        B_front=0.5, B_back=0.5,
    )
    sail = SolarSail(area_m2=1000.0, mass_kg=50.0, optical=optical)
    a = srp_acceleration(r_sat, et, sail, fixed_j2000(s_hat))
    P = solar_flux_at(float(np.linalg.norm(state[:3] - r_sat)))
    PA_over_m_km = P * sail.area_m2 / sail.mass_kg * 1e-3
    expected_coeff = (
        1.0 + optical.rho * optical.s
        + optical.B_front * (1.0 - optical.s) * optical.rho
    )
    expected = -expected_coeff * PA_over_m_km * s_hat
    assert a == pytest.approx(expected, rel=1e-12)


_OFF_AXIS_ALPHA_DEGREES = (10.0, 30.0, 45.0, 60.0, 85.0)


@pytest.mark.parametrize("alpha_deg", _OFF_AXIS_ALPHA_DEGREES)
def test_pure_absorber_magnitude_scales_as_cos_alpha(
    epoch_et, sub_solar_lmo_position, sun_state_km, alpha_deg
):
    """Pure absorber: |a| = (P A / m) cos(alpha), independent of n_hat direction."""
    optical = SailOptical(
        rho=0.0, s=0.0,
        eps_front=0.5, eps_back=0.5,
        B_front=2.0 / 3.0, B_back=2.0 / 3.0,
    )
    sail = _test_sail(optical)
    s_hat = (sun_state_km - sub_solar_lmo_position)
    s_hat = s_hat / np.linalg.norm(s_hat)
    perp = _unit_perp_to(s_hat)
    alpha = math.radians(alpha_deg)
    n_hat = math.cos(alpha) * s_hat + math.sin(alpha) * perp
    a = srp_acceleration(
        sub_solar_lmo_position, epoch_et, sail, fixed_j2000(n_hat)
    )
    P = _solar_pressure_at_sail_pa(sub_solar_lmo_position, sun_state_km)
    expected_magnitude = (P * sail.area_m2 / sail.mass_kg) * math.cos(alpha) * 1e-3
    assert np.linalg.norm(a) == pytest.approx(expected_magnitude, rel=1e-12)
    # Force direction for a pure absorber is always along -s_hat.
    assert a / np.linalg.norm(a) == pytest.approx(-s_hat, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("alpha_deg", _OFF_AXIS_ALPHA_DEGREES)
def test_ideal_specular_mirror_acceleration_scales_as_cos_squared_alpha(
    epoch_et, sub_solar_lmo_position, sun_state_km, alpha_deg
):
    """Ideal mirror: ``a = -2 (P A / m) cos^2(alpha) n_hat``.

    This vector-level form of McInnes Eqs. 2.20 and 2.57 checks both the
    cosine-squared magnitude law and the sail-normal force direction.
    """
    sail = _test_sail(SailOptical.ideal())
    s_hat = sun_state_km - sub_solar_lmo_position
    s_hat = s_hat / np.linalg.norm(s_hat)
    perp = _unit_perp_to(s_hat)
    alpha = math.radians(alpha_deg)
    n_hat = math.cos(alpha) * s_hat + math.sin(alpha) * perp
    a = srp_acceleration(
        sub_solar_lmo_position, epoch_et, sail, fixed_j2000(n_hat)
    )
    P = _solar_pressure_at_sail_pa(sub_solar_lmo_position, sun_state_km)
    expected = (
        -2.0
        * (P * sail.area_m2 / sail.mass_kg)
        * math.cos(alpha) ** 2
        * 1e-3
        * n_hat
    )
    assert a == pytest.approx(expected, rel=1e-12, abs=1e-20)


def test_jpl_square_sail_face_on_ratio_to_ideal_matches_mcinnes_figure_2_9(
    epoch_et, sub_solar_lmo_position, sun_state_km
):
    """JPL square sail gives ~91% of the ideal-sail force at alpha=0.

    The analytical ratio is (1 + rho s) / 2 for pure mirror/JPL
    at face-on with no Lambertian diffuse and no thermal (in the lossless
    limit). With rho=0.88, s=0.94: (1 + 0.8272) / 2 = 0.9136. Thermal and
    diffuse terms shift this by a few percent -- ~0.908 computed. McInnes
    Fig. 2.9 shows the corresponding curves at face-on bunched near 0.9.
    """
    s_hat = (sun_state_km - sub_solar_lmo_position)
    s_hat = s_hat / np.linalg.norm(s_hat)
    att = fixed_j2000(s_hat)

    a_ideal = srp_acceleration(
        sub_solar_lmo_position, epoch_et, _test_sail(SailOptical.ideal()), att
    )
    a_jpl = srp_acceleration(
        sub_solar_lmo_position, epoch_et,
        _test_sail(SailOptical.square_sail_jpl()),
        att,
    )
    ratio = np.linalg.norm(a_jpl) / np.linalg.norm(a_ideal)
    assert 0.88 < ratio < 0.93


# ---------------------------------------------------------------------------
# Group 5 -- linearity / scaling
# ---------------------------------------------------------------------------


def test_force_scales_linearly_with_area(
    epoch_et, sub_solar_lmo_position, sun_hat_j2000
):
    """For fixed mass, ``|a|`` doubles when area doubles."""
    base = SolarSail(
        area_m2=500.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl()
    )
    double = SolarSail(
        area_m2=1000.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl()
    )
    att = fixed_j2000(sun_hat_j2000)
    a1 = srp_acceleration(sub_solar_lmo_position, epoch_et, base, att)
    a2 = srp_acceleration(sub_solar_lmo_position, epoch_et, double, att)
    assert np.linalg.norm(a2) / np.linalg.norm(a1) == pytest.approx(2.0, rel=1e-12)


def test_force_scales_inversely_with_mass(
    epoch_et, sub_solar_lmo_position, sun_hat_j2000
):
    """For fixed area, ``|a|`` halves when mass doubles."""
    base = SolarSail(
        area_m2=500.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl()
    )
    heavy = SolarSail(
        area_m2=500.0, mass_kg=100.0, optical=SailOptical.square_sail_jpl()
    )
    att = fixed_j2000(sun_hat_j2000)
    a1 = srp_acceleration(sub_solar_lmo_position, epoch_et, base, att)
    a2 = srp_acceleration(sub_solar_lmo_position, epoch_et, heavy, att)
    assert np.linalg.norm(a2) / np.linalg.norm(a1) == pytest.approx(0.5, rel=1e-12)


def test_force_scales_as_inverse_square_of_sail_sun_distance():
    """Flux falls off as 1/d^2 from the exact SPICE-sourced sail-Sun distance.

    Two ideal-mirror sails face-on on the sub-solar side at very
    different sub-solar altitudes so the sail-Sun distances differ by
    ~1%. Names track the orbital-altitude convention (``lmo`` = 400 km;
    ``high`` = 2e6 km sub-solar offset). Since ``high`` is closer to the
    Sun, the flux there is HIGHER, so ``|a_lmo| / |a_high| < 1`` and
    the expected ratio is ``(d_high / d_lmo)^2``.
    """
    et = utc_to_et(EPOCH_STR)
    state, _ = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")
    r_sun = np.asarray(state[:3])
    s_hat = r_sun / np.linalg.norm(r_sun)

    R_sat_lmo = mars_equatorial_radius_km() + 400.0
    R_sat_high = mars_equatorial_radius_km() + 400.0 + 2.0e6  # ~1% of d_MS
    r_lmo = R_sat_lmo * s_hat
    r_high = R_sat_high * s_hat
    sail = SolarSail(
        area_m2=1000.0, mass_kg=50.0, optical=SailOptical.ideal()
    )
    att = fixed_j2000(s_hat)
    a_lmo = srp_acceleration(r_lmo, et, sail, att)
    a_high = srp_acceleration(r_high, et, sail, att)
    d_lmo = float(np.linalg.norm(r_sun - r_lmo))
    d_high = float(np.linalg.norm(r_sun - r_high))
    expected_ratio = (d_high / d_lmo) ** 2
    actual_ratio = np.linalg.norm(a_lmo) / np.linalg.norm(a_high)
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-10)


# ---------------------------------------------------------------------------
# Group 6 -- propagator plumbing
# ---------------------------------------------------------------------------


def _circular_state(R_sat_km: float, mu_km3_s2: float, axis: str = "x"):
    """Circular 2-body initial state along a chosen orbit axis."""
    v_circ = math.sqrt(mu_km3_s2 / R_sat_km)
    if axis == "x":
        return np.array([R_sat_km, 0.0, 0.0, 0.0, v_circ, 0.0])
    raise ValueError(axis)


def test_propagate_with_solar_sail_perturbs_two_body_at_expected_magnitude(
    epoch_et,
):
    """SRP over ~one LMO orbit displaces the trajectory at the predicted scale.

    Order-of-magnitude: the SRP acceleration at Mars (~1.42 AU in June 2026)
    is ``P A / m ~ (2.3e-6 Pa) * A/m``. With A=1000 m^2, m=50 kg and a
    sun-pointing ideal sail (so the full 2x reflector factor applies), the
    scalar acceleration |a| ~ 9.1e-8 km/s^2. Over one ~6700 s LMO orbit the
    Hill-limit position displacement is |a|/n^2 ~ |a| * T^2 / (4 pi^2) --
    at most a few hundred metres. The test pins the delta order of
    magnitude to [5, 5000] m.
    """
    R_sat = mars_equatorial_radius_km() + 400.0
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state(R_sat, mu)
    T = 2.0 * math.pi * math.sqrt(R_sat ** 3 / mu)
    sail = SolarSail(
        area_m2=1000.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl()
    )
    opts = PropagationOptions.fast()
    t_eval = np.array([T])

    res_no = propagate(
        state0, (0.0, T), epoch_et=epoch_et, options=opts, t_eval_s=t_eval
    )
    res_srp = propagate(
        state0, (0.0, T), epoch_et=epoch_et,
        solar_sail=sail, sail_normal=sun_pointing(),
        options=opts, t_eval_s=t_eval,
    )
    delta_km = np.linalg.norm(
        res_srp.state_km_kmps[-1, :3] - res_no.state_km_kmps[-1, :3]
    )
    delta_m = delta_km * 1000.0
    assert 5.0 < delta_m < 5000.0


def test_propagate_solar_sail_metadata_is_populated(epoch_et):
    R_sat = mars_equatorial_radius_km() + 400.0
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state(R_sat, mu)
    T = 2.0 * math.pi * math.sqrt(R_sat ** 3 / mu)
    sail = SolarSail(
        area_m2=800.0, mass_kg=40.0, optical=SailOptical.square_sail_jpl()
    )
    res = propagate(
        state0, (0.0, 0.1 * T), epoch_et=epoch_et,
        solar_sail=sail, sail_normal=sun_pointing(),
        options=PropagationOptions.fast(),
    )
    meta = res.metadata["solar_sail"]
    assert meta["area_m2"] == 800.0
    assert meta["mass_kg"] == 40.0
    assert meta["loading_kg_per_m2"] == pytest.approx(0.05)
    opt = meta["optical"]
    assert opt["rho"] == 0.88
    assert opt["s"] == 0.94


def test_propagate_rejects_solar_sail_without_sail_normal(epoch_et):
    R_sat = mars_equatorial_radius_km() + 400.0
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state(R_sat, mu)
    sail = SolarSail(
        area_m2=100.0, mass_kg=10.0, optical=SailOptical.ideal()
    )
    with pytest.raises(ValueError, match="supplied together"):
        propagate(state0, (0.0, 100.0), epoch_et=epoch_et, solar_sail=sail)


def test_propagate_rejects_sail_normal_without_solar_sail(epoch_et):
    R_sat = mars_equatorial_radius_km() + 400.0
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state(R_sat, mu)
    with pytest.raises(ValueError, match="supplied together"):
        propagate(
            state0, (0.0, 100.0),
            epoch_et=epoch_et, sail_normal=sun_pointing(),
        )


def test_propagate_solar_sail_stacks_additively_with_harmonics_and_third_bodies(
    epoch_et,
):
    """SRP layered on top of full harmonics + third bodies is distinct from
    the same run without SRP, and the delta is of the same order as SRP
    alone vs. two-body.
    """
    from reflectors.third_body import sun_third_body
    R_sat = mars_equatorial_radius_km() + 400.0
    mu = mars_gm_km3_per_s2()
    state0 = _circular_state(R_sat, mu)
    T = 2.0 * math.pi * math.sqrt(R_sat ** 3 / mu)
    sail = SolarSail(
        area_m2=1000.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl()
    )
    opts = PropagationOptions.fast()
    t_eval = np.array([T])

    res_grav = propagate(
        state0, (0.0, T), epoch_et=epoch_et,
        gravity_degree=4, third_bodies=[sun_third_body()],
        options=opts, t_eval_s=t_eval,
    )
    res_all = propagate(
        state0, (0.0, T), epoch_et=epoch_et,
        gravity_degree=4, third_bodies=[sun_third_body()],
        solar_sail=sail, sail_normal=sun_pointing(),
        options=opts, t_eval_s=t_eval,
    )
    delta_km = np.linalg.norm(
        res_all.state_km_kmps[-1, :3] - res_grav.state_km_kmps[-1, :3]
    )
    delta_m = delta_km * 1000.0
    # Same order as SRP alone vs two-body (5..5000 m).
    assert 5.0 < delta_m < 5000.0
    # SRP metadata alongside the other perturbations.
    assert "solar_sail" in res_all.metadata
    assert "third_bodies" in res_all.metadata
    assert res_all.metadata["gravity_degree"] == 4


# ---------------------------------------------------------------------------
# Group 7 -- spherical-grain SRP path (Hamilton & Krivov 1996; Burns 1979)
#
# The flat-sail McInnes path stays unchanged; this group exercises the
# parallel ``SphericalParticle`` / ``spherical_particle_acceleration`` /
# ``propagate(spherical_particle=...)`` API. The primary correctness
# pin is ``test_spherical_matches_flat_sail_in_absorbing_limit``: at α=0
# with the absorbing-Lambertian preset (rho=0, s=0, eps_f=eps_b, B_f=B_b),
# the McInnes formula reduces to Q_pr=1 along the anti-Sun line, and the
# two SRP paths must agree to machine precision.
# ---------------------------------------------------------------------------


def test_spherical_particle_property_relations_match_closed_form():
    """A/m, mass, cross-section pinned to closed forms at 1e-12 relative."""
    r = 1.5e-4
    rho = 3000.0
    p = SphericalParticle(radius_m=r, density_kg_per_m3=rho)
    expected_A = math.pi * r * r
    expected_m = (4.0 / 3.0) * math.pi * r ** 3 * rho
    expected_AoM = 3.0 / (4.0 * r * rho)
    assert math.isclose(p.cross_section_m2, expected_A, rel_tol=1e-12)
    assert math.isclose(p.mass_kg, expected_m, rel_tol=1e-12)
    assert math.isclose(p.area_to_mass_m2_per_kg, expected_AoM, rel_tol=1e-12)


def test_spherical_particle_default_qpr_is_unity():
    """H&K96 page 508 Phobos/Deimos ejecta convention: Q_pr = 1.0."""
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    assert p.Q_pr == 1.0


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("radius_m", 0.0),
        ("radius_m", -1.5e-4),
        ("density_kg_per_m3", 0.0),
        ("density_kg_per_m3", -3000.0),
        ("Q_pr", 0.0),
        ("Q_pr", -0.5),
    ],
)
def test_spherical_particle_rejects_nonpositive_inputs(field, bad_value):
    kwargs = dict(radius_m=1.5e-4, density_kg_per_m3=3000.0, Q_pr=1.0)
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        SphericalParticle(**kwargs)


def test_spherical_acceleration_points_along_anti_sun_line(
    sub_solar_lmo_position, epoch_et, sun_state_km,
):
    """Force is parallel to (r_sat - r_sun); zero transverse component."""
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    a = spherical_particle_acceleration(sub_solar_lmo_position, epoch_et, p)
    sun_to_sat = sub_solar_lmo_position - sun_state_km
    sun_to_sat_hat = sun_to_sat / np.linalg.norm(sun_to_sat)
    a_mag = float(np.linalg.norm(a))
    assert a_mag > 0.0
    # Parallel: cosine to anti-Sun direction is unity.
    cos = float(np.dot(a, sun_to_sat_hat) / a_mag)
    assert math.isclose(cos, 1.0, rel_tol=1e-12, abs_tol=1e-14)
    # Transverse component is zero to round-off.
    transverse = a - (a @ sun_to_sat_hat) * sun_to_sat_hat
    assert float(np.linalg.norm(transverse)) < 1e-18


def test_spherical_magnitude_matches_hk96_eq3(
    sub_solar_lmo_position, epoch_et, sun_state_km,
):
    """|a_SRP| matches H&K96 Eq. (3): (3/4) * Q_pr * F_solar / (c rho_g r_g).

    Equivalently, |a_SRP| = Q_pr * P(r_helio) * pi r^2 / m. The two
    closed forms agree algebraically; the test pins both. F_solar is
    the solar irradiance at the SPICE-fed sail-Sun distance, NOT the
    1-AU value -- correct H&K96 reading is "F_solar at the heliocentric
    distance of the planet" but this implementation evaluates at the
    sail-Sun distance step-by-step (consistent with the parallel-ray
    approximation; differs only at the ~1e-5 level for an LMO sail).
    """
    from reflectors.solar_constants import (
        SPEED_OF_LIGHT_M_PER_S, solar_irradiance_W_per_m2_at,
    )
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    a = spherical_particle_acceleration(sub_solar_lmo_position, epoch_et, p)
    r_helio_km = float(np.linalg.norm(sun_state_km - sub_solar_lmo_position))
    F_solar = solar_irradiance_W_per_m2_at(r_helio_km)
    expected_mag_mps2 = (
        0.75 * p.Q_pr * F_solar
        / (SPEED_OF_LIGHT_M_PER_S * p.density_kg_per_m3 * p.radius_m)
    )
    expected_mag_kmps2 = expected_mag_mps2 * 1.0e-3
    assert math.isclose(
        float(np.linalg.norm(a)), expected_mag_kmps2, rel_tol=1e-10,
    )


def test_spherical_inside_umbra_returns_zero(
    anti_solar_lmo_position, epoch_et,
):
    """Anti-solar LMO position is in Mars umbra: zero force returned."""
    # Sanity: the umbra reaches well past LMO at Mars-Sun mean distance.
    assert umbra_cone_length_km(epoch_et) > 1e6
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    a = spherical_particle_acceleration(anti_solar_lmo_position, epoch_et, p)
    assert np.array_equal(a, np.zeros(3))


def test_spherical_apply_shadow_false_keeps_force_in_umbra(
    anti_solar_lmo_position, epoch_et, sun_state_km,
):
    """apply_shadow=False disables the umbra gate (H&K96 shadow-neglect).

    At the same anti-solar (in-umbra) position where the default path
    returns zero (test_spherical_inside_umbra_returns_zero), the
    no-shadow path returns the full anti-Sun force. Pins both the
    default (bit-identical zero) and the no-shadow option in one place.
    """
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    # Default: shadowed -> zero (regression guard for the default).
    a_shadowed = spherical_particle_acceleration(
        anti_solar_lmo_position, epoch_et, p,
    )
    assert np.array_equal(a_shadowed, np.zeros(3))
    # No-shadow: full anti-Sun force.
    a_noshadow = spherical_particle_acceleration(
        anti_solar_lmo_position, epoch_et, p, apply_shadow=False,
    )
    assert float(np.linalg.norm(a_noshadow)) > 0.0
    sun_to_sat = anti_solar_lmo_position - sun_state_km
    sun_to_sat_hat = sun_to_sat / np.linalg.norm(sun_to_sat)
    cos = float(np.dot(a_noshadow, sun_to_sat_hat) / np.linalg.norm(a_noshadow))
    assert math.isclose(cos, 1.0, rel_tol=1e-12, abs_tol=1e-14)


def test_spherical_matches_flat_sail_in_absorbing_limit(
    sub_solar_lmo_position, epoch_et,
):
    """Absorbing-sphere equivalence pin.

    For a SolarSail with the absorbing-Lambertian preset
    (rho=0, s=0, eps_f=eps_b, B_f=B_b=2/3) and sun_pointing attitude,
    the McInnes (1999) Eq. 2.57 formula collapses at alpha = 0 to:
        a = -(P A / m) * s_hat
    -- with no contribution from any of (rho, s, eps, B) terms.
    The H&K96 spherical formula with Q_pr = 1.0 evaluates to the same
    expression for A = pi r^2, m = (4/3) pi r^3 rho. The two paths
    must therefore produce bit-identical accelerations to machine
    precision.

    This is the most direct correctness pin: if either side is altered
    in a way that breaks the absorbing-sphere equivalence (e.g., a
    factor-of-2 normalization slip, a sign convention flip), the test
    raises an explicit test failure.
    """
    radius_m = 1.5e-4
    density = 3000.0
    optical_absorbing = SailOptical(
        rho=0.0, s=0.0,
        eps_front=0.9, eps_back=0.9,        # symmetric -> thermal cancels
        B_front=2.0 / 3.0, B_back=2.0 / 3.0,
    )
    sail = SolarSail(
        area_m2=math.pi * radius_m ** 2,
        mass_kg=(4.0 / 3.0) * math.pi * radius_m ** 3 * density,
        optical=optical_absorbing,
    )
    a_flat = srp_acceleration(
        sub_solar_lmo_position, epoch_et, sail, sun_pointing(),
    )
    p = SphericalParticle(
        radius_m=radius_m, density_kg_per_m3=density, Q_pr=1.0,
    )
    a_sphere = spherical_particle_acceleration(
        sub_solar_lmo_position, epoch_et, p,
    )
    assert np.allclose(a_flat, a_sphere, rtol=1e-12, atol=0.0)


def test_propagate_spherical_particle_produces_nonzero_e(epoch_et):
    """Positive signal: SRP wired through the propagator.

    Five-orbit run at LMO with a 300 um sphere should pump e from 0
    to a measurable value (>1e-6). A no-SRP control run with the
    same gravity stays at e ~ machine zero by construction. This positive
    signal verifies that spherical-particle SRP contributes to the dynamics.
    """
    # Circular LMO state: 400 km altitude, equatorial.
    R_sat = mars_equatorial_radius_km() + 400.0
    state0 = np.array([R_sat, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    mu = mars_gm_km3_per_s2()
    v_circ = math.sqrt(mu / R_sat)
    state0[4] = v_circ
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    T_orb = 2.0 * math.pi * math.sqrt(R_sat ** 3 / mu)
    opts = PropagationOptions.fast()
    res = propagate(
        state0, (0.0, 5.0 * T_orb), epoch_et=epoch_et,
        zonal_degree=2, spherical_particle=p,
        options=opts,
    )
    # Final-state e from the Cartesian state.
    r_f = res.state_km_kmps[-1, :3]
    v_f = res.state_km_kmps[-1, 3:]
    h = np.cross(r_f, v_f)
    e_vec = np.cross(v_f, h) / mu - r_f / float(np.linalg.norm(r_f))
    e_final = float(np.linalg.norm(e_vec))
    assert e_final > 1e-6
    assert "spherical_particle" in res.metadata
    assert res.metadata["spherical_particle"]["Q_pr"] == 1.0


def test_propagate_rejects_solar_sail_and_spherical_particle_together(epoch_et):
    """Mutual-exclusion validation."""
    R_sat = mars_equatorial_radius_km() + 400.0
    state0 = np.array([R_sat, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    mu = mars_gm_km3_per_s2()
    state0[4] = math.sqrt(mu / R_sat)
    sail = _test_sail(SailOptical.ideal())
    p = SphericalParticle(radius_m=1.5e-4, density_kg_per_m3=3000.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        propagate(
            state0, (0.0, 100.0), epoch_et=epoch_et,
            solar_sail=sail, sail_normal=sun_pointing(),
            spherical_particle=p,
        )


# ---------------------------------------------------------------------------
# Group 7 -- two-sided sail (uncommanded / tumbling attitude)
#
# The one-sided model returns exactly zero when back-lit, which is McInnes's
# own assumption (§2.6.1: (M-2.48) takes tau = 0 "on the reflecting side" and
# the Fig. 2.8 p.49 thermal balance lights the FRONT face only). That is right
# for a commanded sail that keeps its front to the Sun, and invalid for a
# tumbling one, which is back-lit ~half the time. ``two_sided`` defaults to
# False so all of the above stays intact.
# ---------------------------------------------------------------------------


_TWO_SIDED_P_PA = 1.9673e-6      # any positive pressure; cancels in the ratios
_TWO_SIDED_AREA = 10000.0
_TWO_SIDED_MASS = 180.0          # sigma = 18 g/m^2


def _two_sided_pair(**kwargs):
    """(one_sided_sail, two_sided_sail) sharing one film and one bus."""
    one = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                    optical=SailOptical(**kwargs))
    two = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                    optical=SailOptical(**kwargs, two_sided=True))
    return one, two


def test_two_sided_defaults_to_false_on_every_factory():
    """Every factory preserves the one-sided default."""
    assert SailOptical.ideal().two_sided is False
    assert SailOptical.square_sail_jpl().two_sided is False
    assert SailOptical.heliogyro_jpl().two_sided is False
    assert SailOptical(rho=0.5, s=0.5, eps_front=0.5, eps_back=0.5,
                       B_front=0.5, B_back=0.5).two_sided is False


def test_with_faces_swapped_swaps_thermal_pairs_and_is_an_involution():
    opt = SailOptical.square_sail_jpl()
    sw = opt.with_faces_swapped()
    assert (sw.eps_front, sw.B_front) == (opt.eps_back, opt.B_back)
    assert (sw.eps_back, sw.B_back) == (opt.eps_front, opt.B_front)
    # rho / s are film properties the six-parameter model does not split.
    assert (sw.rho, sw.s) == (opt.rho, opt.s)
    assert sw.with_faces_swapped() == opt


@pytest.mark.parametrize("optical", _MATERIALS)
def test_two_sided_front_lit_is_bit_identical_to_one_sided(optical):
    """Turning the flag on must change NOTHING on the front-lit half."""
    kwargs = {k: getattr(optical, k) for k in
              ("rho", "s", "eps_front", "eps_back", "B_front", "B_back")}
    one, two = _two_sided_pair(**kwargs)
    s_hat = np.array([1.0, 0.0, 0.0])
    for alpha_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0):
        a = math.radians(alpha_deg)
        n_hat = np.array([math.cos(a), math.sin(a), 0.0])
        a_one = mcinnes_srp_acceleration(n_hat, s_hat, _TWO_SIDED_P_PA, one)
        a_two = mcinnes_srp_acceleration(n_hat, s_hat, _TWO_SIDED_P_PA, two)
        assert np.array_equal(np.asarray(a_one), np.asarray(a_two))


@pytest.mark.parametrize("optical", _MATERIALS)
def test_two_sided_edge_on_is_still_exactly_zero(optical):
    """cos alpha == 0 has no lit face either way -- no force from nowhere."""
    kwargs = {k: getattr(optical, k) for k in
              ("rho", "s", "eps_front", "eps_back", "B_front", "B_back")}
    _, two = _two_sided_pair(**kwargs)
    a = mcinnes_srp_acceleration(np.array([0.0, 1.0, 0.0]),
                                 np.array([1.0, 0.0, 0.0]),
                                 _TWO_SIDED_P_PA, two)
    assert np.linalg.norm(np.asarray(a)) == 0.0


def test_two_sided_back_lit_equals_one_sided_with_normal_and_faces_flipped():
    """The extension rule, stated as an identity.

    A back-lit two-sided sail at ``n_hat`` must equal a one-sided sail
    evaluated at the illuminated-face normal ``-n_hat`` with the front/back
    thermal pairs swapped. This is the whole content of the extension, so it is
    pinned exactly rather than approximately.
    """
    kwargs = dict(rho=0.88, s=0.94, eps_front=0.05, eps_back=0.55,
                  B_front=0.79, B_back=0.55)     # deliberately ASYMMETRIC
    _, two = _two_sided_pair(**kwargs)
    swapped_one = SolarSail(
        area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
        optical=SailOptical(**kwargs).with_faces_swapped())
    s_hat = np.array([1.0, 0.0, 0.0])
    for alpha_deg in (91.0, 105.0, 135.0, 160.0, 180.0):
        a = math.radians(alpha_deg)
        n_hat = np.array([math.cos(a), math.sin(a), 0.0])
        got = mcinnes_srp_acceleration(n_hat, s_hat, _TWO_SIDED_P_PA, two)
        want = mcinnes_srp_acceleration(-n_hat, s_hat, _TWO_SIDED_P_PA,
                                        swapped_one)
        assert np.array_equal(np.asarray(got), np.asarray(want))


def test_two_sided_symmetric_film_is_invariant_under_normal_flip():
    """With eps_f == eps_b and B_f == B_b the two faces are indistinguishable,
    so the force must not care which way the normal is labelled."""
    _, two = _two_sided_pair(rho=0.88, s=0.94, eps_front=0.3, eps_back=0.3,
                             B_front=0.6, B_back=0.6)
    s_hat = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(20260729)
    for _ in range(50):
        n_hat = rng.normal(size=3)
        n_hat /= np.linalg.norm(n_hat)
        a_plus = np.asarray(mcinnes_srp_acceleration(
            n_hat, s_hat, _TWO_SIDED_P_PA, two))
        a_minus = np.asarray(mcinnes_srp_acceleration(
            -n_hat, s_hat, _TWO_SIDED_P_PA, two))
        assert a_plus == pytest.approx(a_minus, rel=1e-15, abs=1e-30)


def _monte_carlo_average_coefficient(sail, n_samples, seed):
    """<a> . (-s_hat) / (P A/m) by brute-force averaging over random normals.

    The INDEPENDENT numerical leg of the tumble-average cross-check: it calls
    the (M-2.57) force directly and never touches the closed form. Uniform
    orientation is generated by normalising Gaussian vectors (the standard
    isotropic construction), which is a different route to "uniform on the
    sphere" than the ``mu`` uniform-on-[0,1] argument the closed form uses.
    """
    rng = np.random.default_rng(seed)
    s_hat = np.array([1.0, 0.0, 0.0])
    n_hats = rng.normal(size=(n_samples, 3))
    n_hats /= np.linalg.norm(n_hats, axis=1, keepdims=True)
    a = np.asarray(mcinnes_srp_acceleration(n_hats, s_hat, _TWO_SIDED_P_PA, sail))
    a_mean = a.mean(axis=0)
    scale = _TWO_SIDED_P_PA * sail.area_m2 / sail.mass_kg * 1.0e-3
    return a_mean, float(np.dot(a_mean, -s_hat) / scale), scale


def test_tumble_average_coefficient_closed_form_values():
    """Closed-form leg: pinned analytic values, not a re-derivation in code."""
    ideal = TumbleAveragedSail(SolarSail(
        area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
        optical=SailOptical(rho=1.0, s=1.0, eps_front=0.0, eps_back=0.0,
                            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
                            two_sided=True)))
    # Ideal film: k = C_s/2 + rho*s/2 + 0 = 0 + 1/2 = exactly 1/2, i.e. one
    # QUARTER of the 2 P A/m sun-facing peak.
    assert ideal.average_coefficient == pytest.approx(0.5, rel=1e-15)

    jpl_kw = dict(rho=0.88, s=0.94, eps_front=0.05, eps_back=0.55,
                  B_front=0.79, B_back=0.55)
    jpl = TumbleAveragedSail(SolarSail(
        area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
        optical=SailOptical(**jpl_kw, two_sided=True)))
    # k = 0.1728/2 + 0.8272/2 + 0.06*0.88*(0.79+0.55)/6
    expected = 0.1728 / 2 + 0.8272 / 2 + 0.06 * 0.88 * (0.79 + 0.55) / 6
    assert jpl.average_coefficient == pytest.approx(expected, rel=1e-12)
    assert jpl.average_coefficient == pytest.approx(0.511792, abs=1e-6)


@pytest.mark.parametrize("optical_kwargs", [
    pytest.param(dict(rho=1.0, s=1.0, eps_front=0.0, eps_back=0.0,
                      B_front=2.0 / 3.0, B_back=2.0 / 3.0), id="ideal"),
    pytest.param(dict(rho=0.88, s=0.94, eps_front=0.05, eps_back=0.55,
                      B_front=0.79, B_back=0.55), id="jpl_square"),
    pytest.param(dict(rho=0.6, s=0.0, eps_front=0.2, eps_back=0.8,
                      B_front=2.0 / 3.0, B_back=0.4), id="fully_diffuse"),
    pytest.param(dict(rho=0.0, s=0.0, eps_front=0.5, eps_back=0.5,
                      B_front=2.0 / 3.0, B_back=2.0 / 3.0), id="absorber"),
])
def test_tumble_average_matches_monte_carlo_over_random_orientations(
    optical_kwargs
):
    """Monte-Carlo leg: the closed form must reproduce a brute-force average
    of the actual force over uniformly random orientations.

    Tolerance is set by Monte-Carlo error, not by the physics: with 200k
    samples the standard error on the mean is ~1/sqrt(N) ~ 0.2%, so 1% is a
    ~5-sigma bound. Seeded, so it cannot flake.
    """
    sail = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                     optical=SailOptical(**optical_kwargs, two_sided=True))
    tumbling = TumbleAveragedSail(sail)
    a_mean, k_mc, scale = _monte_carlo_average_coefficient(
        sail, n_samples=200_000, seed=20260729)
    assert k_mc == pytest.approx(tumbling.average_coefficient, rel=1e-2)
    # And the average must be ANTI-SUNWARD: transverse components cancel.
    s_hat = np.array([1.0, 0.0, 0.0])
    transverse = a_mean - np.dot(a_mean, s_hat) * s_hat
    assert float(np.linalg.norm(transverse)) / scale < 1e-2


def test_one_sided_tumble_average_is_exactly_half_the_two_sided_one():
    """A one-sided film has half the average force of a symmetric two-sided film.

    A one-sided film sees zero force on the back-lit half of the sphere, so its
    orientation average is half the two-sided one (for an otherwise identical
    film with symmetric thermal properties, where the two faces contribute
    equally).
    """
    kw = dict(rho=0.88, s=0.94, eps_front=0.3, eps_back=0.3,
              B_front=0.6, B_back=0.6)     # symmetric film -> exact factor 2
    one = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                    optical=SailOptical(**kw))
    two = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                    optical=SailOptical(**kw, two_sided=True))
    _, k_one, _ = _monte_carlo_average_coefficient(one, 200_000, seed=11)
    _, k_two, _ = _monte_carlo_average_coefficient(two, 200_000, seed=11)
    assert k_two == pytest.approx(2.0 * k_one, rel=2e-3)


def test_tumble_averaged_sail_rejects_a_one_sided_film():
    """Constructing the averaged model from a one-sided film is the exact
    mistake that halves the force, so it is refused."""
    sail = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                     optical=SailOptical.square_sail_jpl())
    with pytest.raises(ValueError, match="two_sided"):
        TumbleAveragedSail(sail)


def test_tumble_averaged_acceleration_is_antisunward_and_scales_correctly(
    epoch_et, sub_solar_lmo_position, sun_state_km
):
    """Vector-level check of the propagator-facing function, against the
    SPICE-sourced Sun direction and solar pressure."""
    sail = SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                     optical=SailOptical(**dict(
                         rho=0.88, s=0.94, eps_front=0.05, eps_back=0.55,
                         B_front=0.79, B_back=0.55), two_sided=True))
    tumbling = TumbleAveragedSail(sail)
    a = tumble_averaged_acceleration(sub_solar_lmo_position, epoch_et, tumbling)
    s_hat = sun_state_km - sub_solar_lmo_position
    s_hat = s_hat / np.linalg.norm(s_hat)
    P = _solar_pressure_at_sail_pa(sub_solar_lmo_position, sun_state_km)
    expected = -(tumbling.average_coefficient
                 * P * sail.area_m2 / sail.mass_kg * 1e-3) * s_hat
    assert a == pytest.approx(expected, rel=1e-12, abs=1e-20)


def test_propagate_rejects_more_than_one_srp_model(epoch_et):
    """tumble_averaged_sail is mutually exclusive with the other two paths."""
    R_sat = mars_equatorial_radius_km() + 400.0
    state0 = np.array([R_sat, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    mu = mars_gm_km3_per_s2()
    state0[4] = math.sqrt(mu / R_sat)
    sail = _test_sail(SailOptical.ideal())
    tumbling = TumbleAveragedSail(SolarSail(
        area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
        optical=SailOptical(rho=1.0, s=1.0, eps_front=0.0, eps_back=0.0,
                            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
                            two_sided=True)))
    with pytest.raises(ValueError, match="mutually exclusive"):
        propagate(state0, (0.0, 100.0), epoch_et=epoch_et,
                  solar_sail=sail, sail_normal=sun_pointing(),
                  tumble_averaged_sail=tumbling)
    with pytest.raises(ValueError, match="mutually exclusive"):
        propagate(state0, (0.0, 100.0), epoch_et=epoch_et,
                  spherical_particle=SphericalParticle(
                      radius_m=1.5e-4, density_kg_per_m3=3000.0),
                  tumble_averaged_sail=tumbling)


def test_propagate_tumble_averaged_emits_metadata_and_perturbs(epoch_et):
    """Propagator plumbing: the averaged force reaches the integrator."""
    R_sat = mars_equatorial_radius_km() + 500.0
    mu = mars_gm_km3_per_s2()
    state0 = np.array([R_sat, 0.0, 0.0, 0.0, math.sqrt(mu / R_sat), 0.0])
    tumbling = TumbleAveragedSail(SolarSail(
        area_m2=10000.0, mass_kg=180.0,
        optical=SailOptical(**dict(rho=0.88, s=0.94, eps_front=0.05,
                                   eps_back=0.55, B_front=0.79, B_back=0.55),
                            two_sided=True)))
    res_off = propagate(state0, (0.0, 6000.0), epoch_et=epoch_et,
                        options=PropagationOptions.fast())
    res_on = propagate(state0, (0.0, 6000.0), epoch_et=epoch_et,
                       tumble_averaged_sail=tumbling,
                       options=PropagationOptions.fast())
    assert "tumble_averaged_sail" in res_on.metadata
    assert res_on.metadata["tumble_averaged_sail"]["optical"]["two_sided"] is True
    assert res_on.metadata["tumble_averaged_sail"][
        "average_coefficient"] == pytest.approx(0.511792, abs=1e-6)
    # Positive signal: the trajectory actually moved relative to no-SRP.
    drift_km = float(np.linalg.norm(
        res_on.state_km_kmps[-1, :3] - res_off.state_km_kmps[-1, :3]))
    assert drift_km > 1e-3


def _resolved_tumble_coefficients(att, sail, window_s, n_samples=40_001):
    """Time-average an actual rotation: (k_parallel, k_transverse).

    Companion to ``_monte_carlo_average_coefficient``, but averaging over TIME
    under a real smooth rotation instead of over uniformly random orientations.
    The difference between the two is the point of the tests below.
    """
    s_hat = np.array([1.0, 0.0, 0.0])
    r_dummy = np.array([3904.0, 0.0, 0.0])   # tumble() is position-independent
    ets = np.linspace(0.0, window_s, n_samples)
    n_hats = np.empty((n_samples, 3))
    for i, dt in enumerate(ets):
        n_hats[i] = att(r_dummy, dt)
    a = np.asarray(mcinnes_srp_acceleration(n_hats, s_hat, _TWO_SIDED_P_PA,
                                            sail))
    a_mean = a.mean(axis=0)
    scale = _TWO_SIDED_P_PA * sail.area_m2 / sail.mass_kg * 1.0e-3
    k_par = float(np.dot(a_mean, -s_hat)) / scale
    transverse = a_mean - float(np.dot(a_mean, s_hat)) * s_hat
    return k_par, float(np.linalg.norm(transverse)) / scale


def _jpl_two_sided_sail():
    return SolarSail(area_m2=_TWO_SIDED_AREA, mass_kg=_TWO_SIDED_MASS,
                     optical=SailOptical(
                         rho=0.88, s=0.94, eps_front=0.05, eps_back=0.55,
                         B_front=0.79, B_back=0.55, two_sided=True))


def test_spin_about_the_sun_line_is_permanently_edge_on_and_feels_no_srp():
    """Exact geometric result, and a striking one: a sail spinning about the
    Sun-line with its normal in the perpendicular plane never presents any
    projected area, so its SRP is identically zero -- not merely averaged away.

    This is the zero-tilt limiting case.
    """
    from reflectors.attitude import tumble
    sail = _jpl_two_sided_sail()
    att = tumble(2.0 * math.pi / 60.0, 0.0,
                 n_hat_0=[0.0, 0.0, 1.0], spin_axis=[1.0, 0.0, 0.0])
    k_par, k_perp = _resolved_tumble_coefficients(att, sail, 7405.0)
    assert k_par == pytest.approx(0.0, abs=1e-12)
    assert k_perp == pytest.approx(0.0, abs=1e-12)


def test_single_axis_spin_perpendicular_to_the_sun_far_exceeds_the_uniform_average():
    """The other extreme of the same sweep: axis perpendicular to the Sun-line,
    90 deg cone, gives 1.62x the uniform-average anti-sunward coefficient.

    Together with the test above this brackets a pure spin at [0x, 1.62x] of
    the uniform value, so a single-axis tumble is not described by the
    uniform-orientation average.
    """
    from reflectors.attitude import tumble
    sail = _jpl_two_sided_sail()
    k_uniform = TumbleAveragedSail(sail).average_coefficient
    att = tumble(2.0 * math.pi / 60.0, 0.0,
                 n_hat_0=[1.0, 0.0, 0.0], spin_axis=[0.0, 0.0, 1.0])
    k_par, k_perp = _resolved_tumble_coefficients(att, sail, 7405.0)
    assert k_par == pytest.approx(0.829269, rel=2e-3)
    assert k_par / k_uniform == pytest.approx(1.6203, rel=2e-3)
    assert k_perp / k_uniform < 0.01     # transverse cancels in THIS geometry


def test_sphere_covering_tumble_does_not_converge_to_the_uniform_average():
    """A sphere-covering tumble need not approach the uniform average.

    A spin plus an incommensurate precession sweeps a 2-D region of the sphere,
    but its invariant measure is still not the uniform one: seed 0 converges to
    0.987 x k_uniform with a 0.012 x k_uniform residual TRANSVERSE component,
    and stays there out to 64 orbits of averaging. The test guards against
    equating the averaged and resolved models.
    """
    from reflectors.attitude import tumble
    sail = _jpl_two_sided_sail()
    k_uniform = TumbleAveragedSail(sail).average_coefficient
    w_spin = 2.0 * math.pi / 60.0
    att = tumble(w_spin, 0.0, seed=0,
                 precession_rate_rad_per_s=w_spin / math.pi)
    k_par, k_perp = _resolved_tumble_coefficients(att, sail, 7405.0)
    assert k_par / k_uniform == pytest.approx(0.9859, rel=5e-3)
    # Distinguishable from the uniform limit, i.e. NOT within round-off of 1.
    assert abs(k_par / k_uniform - 1.0) > 5e-3
    # And a persistent transverse component the uniform model says is zero.
    assert k_perp / k_uniform == pytest.approx(0.0138, rel=0.05)


def test_resolved_tumbles_retain_a_transverse_srp_component():
    """The physically important half of the negative result: the uniform
    average is purely radial (which cannot pump eccentricity secularly), while
    real tumbles keep a transverse force that can. Pinned at a geometry where
    it is large -- spin axis 60 deg to the Sun-line, ~0.53x the radial term.
    """
    from reflectors.attitude import tumble
    sail = _jpl_two_sided_sail()
    k_uniform = TumbleAveragedSail(sail).average_coefficient
    a = math.radians(60.0)
    axis = np.array([math.cos(a), math.sin(a), 0.0])
    n0 = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    att = tumble(2.0 * math.pi / 60.0, 0.0, n_hat_0=n0, spin_axis=axis)
    k_par, k_perp = _resolved_tumble_coefficients(att, sail, 7405.0)
    assert k_perp / k_uniform == pytest.approx(0.5290, rel=5e-3)
    assert k_perp > 0.4 * k_par          # not a small correction


def test_two_sided_never_pushes_toward_the_sun_for_a_reflective_film():
    """Sanity direction check over the full sphere: the anti-sunward component
    is non-negative everywhere for a mostly-reflective sail (a lit face cannot
    suck the sail sunward)."""
    _, two = _two_sided_pair(rho=0.88, s=0.94, eps_front=0.05, eps_back=0.55,
                             B_front=0.79, B_back=0.55)
    s_hat = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(7)
    for _ in range(500):
        n_hat = rng.normal(size=3)
        n_hat /= np.linalg.norm(n_hat)
        a = np.asarray(mcinnes_srp_acceleration(
            n_hat, s_hat, _TWO_SIDED_P_PA, two))
        assert float(np.dot(a, s_hat)) <= 1e-30
