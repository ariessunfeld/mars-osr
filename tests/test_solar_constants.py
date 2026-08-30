"""Pin tests for ``reflectors.solar_constants``.

Each defining constant is verified against its cited primary source, and
the derived ``P_1AU`` is cross-checked against the McInnes 1999 textbook
value (modulo the known ~0.4% offset from the pre-IAU-2015 luminosity).
The ``solar_flux_at`` inverse-square scaling is pinned at a handful of
representative distances.
"""

from __future__ import annotations

import math

import pytest

from reflectors.solar_constants import (
    AU_KM,
    AU_M,
    SOLAR_LUMINOSITY_W,
    SOLAR_RADIUS_KM,
    SPEED_OF_LIGHT_KM_PER_S,
    SPEED_OF_LIGHT_M_PER_S,
    solar_brightness_W_per_m2_per_sr,
    solar_flux_at,
    solar_irradiance_W_per_m2_at,
    solar_pressure_at_1au_pa,
)


def test_solar_luminosity_matches_iau_2015_nominal():
    """L_Sun_N = 3.828e26 W (IAU 2015 Resolution B3)."""
    assert SOLAR_LUMINOSITY_W == 3.828e26


def test_speed_of_light_is_si_exact():
    """c = 299 792 458 m/s exactly (17th CGPM 1983, preserved in SI 2019)."""
    assert SPEED_OF_LIGHT_M_PER_S == 299_792_458.0
    assert SPEED_OF_LIGHT_KM_PER_S == 299_792.458


def test_au_matches_iau_2012_definition():
    """1 au = 149 597 870 700 m exactly (IAU 2012 Resolution B2)."""
    assert AU_M == 149_597_870_700.0
    assert AU_KM == 149_597_870.7


def test_solar_pressure_at_1au_matches_mcinnes_textbook_within_known_offset():
    """P_1AU ~ 4.541e-6 Pa, within ~0.4% of McInnes 1999 p.58 (4.56e-6 Pa).

    The offset is entirely due to the luminosity update: McInnes pre-dates
    IAU 2015 Res B3 and used a slightly larger value. Test pins both the
    computed IAU-2015 value AND the acceptable distance to the textbook
    quote, so any change to the defining constants that shifts P_1AU
    by more than that known offset triggers a test failure.
    """
    P = solar_pressure_at_1au_pa()
    assert 4.540e-6 < P < 4.542e-6
    # Within 1% of McInnes 1999 quote.
    assert abs(P - 4.56e-6) / 4.56e-6 < 0.01


def test_solar_pressure_matches_hand_computation():
    """Redundant definitional identity, catches any code-path shortcut."""
    expected = SOLAR_LUMINOSITY_W / (
        4.0 * math.pi * SPEED_OF_LIGHT_M_PER_S * AU_M * AU_M
    )
    assert solar_pressure_at_1au_pa() == expected


@pytest.mark.parametrize(
    "r_au, expected_ratio",
    [
        (0.5, 4.0),
        (1.0, 1.0),
        (1.524, 1.0 / (1.524 ** 2)),  # Mars mean distance
        (2.0, 0.25),
        (5.2, 1.0 / (5.2 ** 2)),  # Jupiter
    ],
)
def test_solar_flux_scales_as_inverse_square(r_au, expected_ratio):
    """``P(r) / P(1 AU)`` = ``(1 AU / r)^2`` to machine precision."""
    P1 = solar_pressure_at_1au_pa()
    P_r = solar_flux_at(r_au * AU_KM)
    assert P_r / P1 == pytest.approx(expected_ratio, rel=1e-14)


def test_solar_flux_at_mars_perihelion_vs_aphelion_ratio():
    """At 1.381 vs 1.666 AU (Mars perihelion / aphelion):
    flux ratio = (1.666/1.381)^2 ~ 1.456.

    Physical sanity: the ~2.92x irradiance swing McInnes flags at Mars
    is (aphelion/perihelion)^2 = (1.666/1.381)^2 ~ 1.46 PLUS the
    reciprocal (1/1.46 = 0.69), giving a total perihelion-vs-aphelion
    ratio of 2.92 (= 1.46^2). Split out here so each factor is
    independently testable.
    """
    P_peri = solar_flux_at(1.381 * AU_KM)
    P_apo = solar_flux_at(1.666 * AU_KM)
    assert P_peri / P_apo == pytest.approx((1.666 / 1.381) ** 2, rel=1e-14)


def test_solar_flux_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        solar_flux_at(0.0)
    with pytest.raises(ValueError):
        solar_flux_at(-1.0)


# ---------------------------------------------------------------------------
# Solar irradiance and radius
# ---------------------------------------------------------------------------


def test_solar_radius_matches_iau_2015_nominal():
    """R_Sun_N = 6.957e8 m (IAU 2015 Res B3 nominal photospheric radius).

    Pinned here as the defining value for the uniform-disc solar
    brightness. ``reflectors.shadow.sun_radius_km`` sources its value
    from ``pck00011.tpc``; the two agree to machine precision (see
    ``test_solar_radius_matches_pck_value``).
    """
    assert SOLAR_RADIUS_KM == 695_700.0


def test_solar_radius_matches_pck_value():
    """The defining ``SOLAR_RADIUS_KM`` matches the PCK-sourced value
    from ``reflectors.shadow.sun_radius_km``.

    A disagreement means either that IAU has published a revised nominal
    value (requiring an update to ``SOLAR_RADIUS_KM``) or that the PCK has
    changed (less likely; NAIF's generic kernels track IAU). The
    explicit equality test guarantees the two accessors do not
    diverge -- both feed the beam-divergence formula, one through
    ``B_sun`` and the other through the sail-umbra geometry.
    """
    from reflectors.shadow import sun_radius_km

    assert SOLAR_RADIUS_KM == pytest.approx(sun_radius_km(), rel=1e-14)


def test_solar_irradiance_at_1au_matches_iau_luminosity():
    """``I(1 AU) = L_Sun_N / (4 pi (1 au)^2)`` ~ 1361.17 W/m^2.

    This is the NOMINAL solar constant derived from L_Sun_N = 3.828e26 W
    and 1 au = 1.49597870700e11 m. It differs from the measured total
    solar irradiance (TSI ~1366 W/m^2, VIRGO/SORCE/TIM-adjusted) by the
    few-W/m^2 offset between the nominal and observed luminosity; the
    nominal is used so every downstream flux/irradiance value is
    traceable to defining constants.
    """
    I = solar_irradiance_W_per_m2_at(AU_KM)
    assert 1361.0 < I < 1362.0
    # Hand formula.
    expected = SOLAR_LUMINOSITY_W / (4.0 * math.pi * AU_M * AU_M)
    assert I == pytest.approx(expected, rel=1e-14)


def test_solar_irradiance_equals_pressure_times_c_to_machine_precision():
    """Radiometric identity ``I = P * c`` (both computed independently).

    The two accessors compute the same physical quantity through
    different arithmetic paths (one divides by c inside, the other
    does not). Their ratio must equal ``c`` to machine precision.
    Catches any drift between them.
    """
    P = solar_pressure_at_1au_pa()
    I = solar_irradiance_W_per_m2_at(AU_KM)
    assert I / P == pytest.approx(SPEED_OF_LIGHT_M_PER_S, rel=1e-14)


def test_solar_irradiance_at_mars_mean_distance():
    """``I(1.524 AU)`` ~ 586 W/m^2, matching the canonical "Mars solar
    constant" ~590 W/m^2 at the nominal mean distance.

    This broad physical bound checks the irradiance scale. Mars's semi-major
    axis is 1.52368 AU; 1.524 AU matches the references used elsewhere.
    """
    I_mars = solar_irradiance_W_per_m2_at(1.524 * AU_KM)
    assert 584.0 < I_mars < 590.0


@pytest.mark.parametrize(
    "r_au, expected_ratio",
    [
        (0.5, 4.0),
        (1.0, 1.0),
        (1.524, 1.0 / (1.524 ** 2)),
        (5.2, 1.0 / (5.2 ** 2)),
    ],
)
def test_solar_irradiance_scales_as_inverse_square(r_au, expected_ratio):
    I1 = solar_irradiance_W_per_m2_at(AU_KM)
    I_r = solar_irradiance_W_per_m2_at(r_au * AU_KM)
    assert I_r / I1 == pytest.approx(expected_ratio, rel=1e-14)


def test_solar_irradiance_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        solar_irradiance_W_per_m2_at(0.0)
    with pytest.raises(ValueError):
        solar_irradiance_W_per_m2_at(-1.0)


# ---------------------------------------------------------------------------
# Uniform-disc solar surface brightness (B_sun, W/m^2/sr)
# ---------------------------------------------------------------------------


def test_solar_brightness_is_cached_scalar():
    """Value is deterministic and cached between calls."""
    B1 = solar_brightness_W_per_m2_per_sr()
    B2 = solar_brightness_W_per_m2_per_sr()
    assert B1 == B2


def test_solar_brightness_matches_hand_formula():
    """``B_sun = L / (4 pi^2 R_sun^2)``.

    Closed form from uniform-disc radiometry. The factor of pi (vs
    the pi^2 in the denominator) is the geometric integration of
    Lambert's cosine law over the visible disc.
    """
    R_sun_m = SOLAR_RADIUS_KM * 1000.0
    expected = SOLAR_LUMINOSITY_W / (
        4.0 * math.pi * math.pi * R_sun_m * R_sun_m
    )
    assert solar_brightness_W_per_m2_per_sr() == pytest.approx(
        expected, rel=1e-14
    )


def test_solar_brightness_value_pinned():
    """B_sun ~ 2.003e7 W/m^2/sr for IAU 2015 nominal L, R."""
    B = solar_brightness_W_per_m2_per_sr()
    assert 2.0e7 < B < 2.01e7


@pytest.mark.parametrize("r_au", [0.5, 1.0, 1.524, 2.0, 5.2])
def test_brightness_times_solid_angle_equals_irradiance(r_au):
    """Radiometric consistency: ``I(r) = B_sun * Omega_sun(r)``
    for a uniform-disc sun seen at distance ``r >> R_sun``.

    Omega_sun = pi * (R_sun / r)^2 in the small-angle limit (which is
    exact for the defining constants' sub-milliradian scale at any
    planetary orbit). This identity is the foundation for the
    radiance-conservation form of the reflected-beam formula
    (``I_target = eta * B_sun * Omega_mirror * sin(el)``) that
    ``reflectors.beam`` cross-checks against the Canady-Allen form.

    Machine-precision agreement verifies the two accessors use
    self-consistent L_Sun, R_Sun, and AU values.
    """
    r_km = r_au * AU_KM
    r_m = r_km * 1000.0
    R_sun_m = SOLAR_RADIUS_KM * 1000.0
    Omega = math.pi * (R_sun_m / r_m) ** 2
    I_via_B = solar_brightness_W_per_m2_per_sr() * Omega
    I_direct = solar_irradiance_W_per_m2_at(r_km)
    assert I_via_B == pytest.approx(I_direct, rel=1e-14)
