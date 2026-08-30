"""Solar physical constants and the solar-radiation-pressure formula.

Target physics: compute the solar radiation pressure ``P(r) = I(r) / c``
at arbitrary heliocentric distance ``r``, from the nominal solar
luminosity and fundamental constants. Used by ``reflectors.srp`` to
evaluate the pressure each RHS step from the live SPICE sail-Sun
distance (so the 1.381-1.666 AU Mars-year swing, a 2.92x swing in
irradiance, is captured automatically without any seasonal hardcoding).

Also exposes the corresponding solar IRRADIANCE ``I(r) = P(r) * c`` in
W/m^2 and the uniform-disc solar SURFACE BRIGHTNESS
``B_sun = L_sun / (4 pi^2 R_sun^2)`` in W/m^2/sr, which are the natural
inputs to the reflected-beam irradiance formula in
``reflectors.beam`` (Canady & Allen 1982 Eq. 9; Çelik & McInnes 2022
Eq. 16; Viale et al. 2023 Eq. 12). Brightness is a conserved quantity
in geometric optics (Born & Wolf, *Principles of Optics*, §4.8) and
reduces the reflected-beam formula to ``I = eta * chi * B_sun *
Omega_mirror * sin(elevation)`` -- with the sail-Sun distance
cancelling exactly, a useful cross-check on the distance-dependent
Canady-Allen form.

All constants are CITED. The module's purpose is to be a single source of
truth for solar / photometric numerical values that would otherwise be
duplicated in scripts and tests. Each value is cross-checked against
its literature source in ``tests/test_solar_constants.py`` so changes to
the defining values trigger an explicit test failure.

References.

  - IAU 2015 Resolution B3 on recommended nominal conversion constants
    for selected solar and planetary quantities. Defines
    ``L_Sun_N = 3.828e26 W`` as the nominal solar luminosity; see
    Mamajek et al. 2015 (arXiv:1510.07674) and the companion paper
    Prsa et al. 2016, "Nominal values for selected solar and planetary
    quantities", Astronomical Journal 152:41,
    https://doi.org/10.3847/0004-6256/152/2/41.

  - CODATA 2018 / SI 2019 redefinition: the speed of light in vacuum
    is EXACTLY ``c = 299 792 458 m/s`` by definition of the metre
    (17th CGPM, 1983; preserved in the 2019 SI redefinition).

  - IAU 2012 Resolution B2 on the astronomical unit. Defines
    ``1 au = 149 597 870 700 m`` exactly (re-casting the AU as a
    defined conversion constant, decoupled from the Gaussian
    gravitational constant). See Capitaine, Klioner & McCarthy 2012,
    "Recommendations of IAU Working Group on Nominal Units for Stellar
    and Planetary Astronomy" and Prsa et al. 2016 above.

Derived: the canonical 1 AU solar radiation pressure

    P_1AU = L_Sun_N / (4 pi c (1 au)^2)
          ~ 4.541e-6 N/m^2

is pinned by a test, and differs from the McInnes 1999 textbook value
4.56e-6 N/m^2 (p.58) by ~0.4% because the book pre-dates the IAU 2015
nominal luminosity; the IAU nominal is used so that all downstream
solar-sail acceleration scales are traceable to the 2015 resolution.
"""

from __future__ import annotations

import math
from functools import lru_cache


# ---------------------------------------------------------------------------
# Defining constants (cited)
# ---------------------------------------------------------------------------


# IAU 2015 Resolution B3, nominal solar luminosity L_Sun_N.
SOLAR_LUMINOSITY_W: float = 3.828e26

# CODATA / 17th CGPM 1983: speed of light in vacuum, defined exactly.
SPEED_OF_LIGHT_M_PER_S: float = 299_792_458.0
SPEED_OF_LIGHT_KM_PER_S: float = SPEED_OF_LIGHT_M_PER_S / 1000.0

# IAU 2012 Resolution B2, astronomical unit as a defined length.
AU_M: float = 149_597_870_700.0
AU_KM: float = AU_M / 1000.0


# ---------------------------------------------------------------------------
# Derived: solar radiation pressure
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def solar_pressure_at_1au_pa() -> float:
    """Canonical 1 AU solar radiation pressure in pascals (N/m^2).

    ``P_1AU = L_Sun_N / (4 pi c (1 au)^2)``. Computed from the defining
    constants above rather than hard-coded; ``~ 4.541e-6 Pa``. Pinned
    by ``tests/test_solar_constants.py`` against the textbook value
    (McInnes 1999 p.58 reports 4.56e-6, which predates IAU 2015 nominal
    luminosity and differs by ~0.4%).
    """
    return SOLAR_LUMINOSITY_W / (
        4.0 * math.pi * SPEED_OF_LIGHT_M_PER_S * AU_M * AU_M
    )


def solar_flux_at(r_km: float) -> float:
    """Solar radiation pressure at heliocentric distance ``r_km``, pascals.

    ``P(r) = L_Sun_N / (4 pi c r^2)``, with ``r`` the distance from the
    Sun centre to the point of evaluation. Returns N/m^2 (= pascals).

    The full formula is used rather than ``P_1AU (1 au / r)^2`` to keep
    propagation of floating-point error transparent: a single division
    and a single square are cheaper than forming the ratio and squaring
    it, and there is no risk of losing precision when ``r``
    approaches 1 au.

    Parameters
    ----------
    r_km
        Distance from the Sun centre in km. Must be strictly positive.
    """
    if r_km <= 0.0:
        raise ValueError(f"solar_flux_at: r_km must be > 0, got {r_km!r}")
    r_m = r_km * 1000.0
    return SOLAR_LUMINOSITY_W / (4.0 * math.pi * SPEED_OF_LIGHT_M_PER_S * r_m * r_m)


def solar_irradiance_W_per_m2_at(r_km: float) -> float:
    """Solar IRRADIANCE at heliocentric distance ``r_km``, W/m^2.

    ``I(r) = L_Sun_N / (4 pi r^2)``. Identical in content to
    ``solar_flux_at(r_km) * c`` (the radiometric conversion ``P = I/c``)
    but computed directly from ``L_Sun_N`` rather than through the
    pressure accessor, so each function owns its own roundoff budget.
    Returns W/m^2.

    Used by ``reflectors.beam.delivered_surface_irradiance_W_per_m2``:
    the Canady-Allen 1982 Eq. 9 form has ``I_0`` (solar constant) as an
    explicit factor, and for a sail at Mars heliocentric distance
    ``|r_sail - r_sun|`` the correct value of ``I_0`` is this
    function's output -- NOT the 1366.1 W/m^2 "1 AU solar constant"
    quoted in terrestrial-solar references.

    Parameters
    ----------
    r_km
        Distance from the Sun centre in km. Must be strictly positive.
    """
    if r_km <= 0.0:
        raise ValueError(
            f"solar_irradiance_W_per_m2_at: r_km must be > 0, got {r_km!r}"
        )
    r_m = r_km * 1000.0
    return SOLAR_LUMINOSITY_W / (4.0 * math.pi * r_m * r_m)


# ---------------------------------------------------------------------------
# Uniform-disc solar surface brightness
# ---------------------------------------------------------------------------


# Nominal photospheric radius of the Sun, km. IAU 2015 Resolution B3
# (Prsa et al. 2016 AJ 152:41) defines the NOMINAL solar radius
# R_Sun_N = 6.957e8 m exactly; used here as the defining value. The
# value in ``pck00011.tpc`` (reachable via ``reflectors.shadow.
# sun_radius_km``) matches to machine precision; ``shadow`` uses the
# PCK-sourced value so the Sun as a body is consistent with every other
# SPICE-backed geometry, while this module uses the defining IAU value
# so the brightness constant is purely a function of defining constants.
# The two values agree exactly (pinned by a test).
SOLAR_RADIUS_KM: float = 695_700.0


@lru_cache(maxsize=1)
def solar_brightness_W_per_m2_per_sr() -> float:
    """Uniform-disc solar surface brightness, W/m^2/sr.

    ``B_sun = L_Sun_N / (4 pi^2 R_sun^2)``, the radiance of the sun
    modelled as a uniformly-emitting disc. Derivation: a uniform disc
    of radius ``R`` at distance ``d`` emits total flux at ``d`` of
    ``Phi(d) = pi B (R/d)^2`` (small-angle limit, seen from a point on
    the disc-axis at distance ``d >> R``); equating this to the
    inverse-square law ``Phi(d) = L / (4 pi d^2)`` gives
    ``B = L / (4 pi^2 R^2)``, independent of ``d`` as required for a
    Lambertian-limit surface brightness.

    Pinned value: ~ 2.00e7 W/m^2/sr. Independently confirmed:
    direct solar irradiance at Mars mean distance (1.524 AU) equals
    ``B_sun * Omega_sun(1.524 AU) = B_sun * pi * (R_sun / r_sun)^2``,
    matching ``solar_irradiance_W_per_m2_at(1.524 * AU_KM)`` to
    1e-14 relative (test pin).

    Primary reference: Born & Wolf, *Principles of Optics*, §4.8
    "Radiometry and photometry"; radiance conservation applies to the
    ideal-mirror reflected-beam geometry in ``reflectors.beam`` (the
    reflected sun image has brightness ``eta * B_sun`` where ``eta`` is
    the specular reflectance fraction).
    """
    R_sun_m = SOLAR_RADIUS_KM * 1000.0
    return SOLAR_LUMINOSITY_W / (
        4.0 * math.pi * math.pi * R_sun_m * R_sun_m
    )
