"""Reflected-beam divergence and delivered surface irradiance at a target.

Target physics: compute the mean irradiance (W/m^2) delivered to a
surface point when a Mars-orbiting sail specularly reflects sunlight
toward that point, accounting for

  (i)   the finite angular size of the Sun (~0.0093 rad at 1 AU; ~0.0061
        rad at Mars mean distance), which diverges the reflected beam,
  (ii)  the sail's foreshortening ``cos(psi/2)`` at the bisector
        geometry (the incidence half-angle at the mirror),
  (iii) the target-surface inclination ``sin(epsilon)`` (elevation of
        the sail above the target's local horizon plane), which
        stretches the circular perpendicular-beam cross-section into
        a 1/sin(epsilon) ellipse on the ground,
  (iv)  the specular reflectance fraction ``rho * s`` of the McInnes
        (1999) six-parameter SRP optical model -- only the specular
        fraction of reflected light forms a directed beam on target;
        diffuse ``rho (1-s)`` scatters Lambertianly over 2pi sr and
        the absorbed fraction ``(1 - rho)`` is thermally re-emitted
        diffusely, both negligible at any realistic target,
  (v)   an optional atmospheric transmission factor ``chi`` supplied by the
        caller (default 1 = vacuum),
  (vi)  optionally, the finite sail size (Çelik & McInnes 2022
        Eq. 15): quadrature sum of beam-spread-from-sun and
        mirror-projected-diameter contributions to the image
        perpendicular radius. Off by default; <1e-4 correction for
        sail diameters << slant * alpha at Mars LMO, but cheap.

The reflected-sun image on a plane perpendicular to the beam is a
circle of radius ``d * tan(alpha/2)`` where ``alpha = 2 * arcsin(R_sun
/ r_sun_sail)`` is the Sun's subtense angle at the sail (Canady &
Allen 1982 Eq. 1; Çelik & McInnes 2022 Eq. 8a, *after* their Eq. 9
sun-subtense definition, which is not adopted here; see "Modeling choice 1"
below). Projected onto an inclined surface, the image is an ellipse
with semi-minor axis ``b = d tan(alpha/2)`` (perpendicular to the
ground trace of the beam) and semi-major axis ``a = b / sin(epsilon)``
(along the trace) (Çelik & McInnes 2022 Eq. 8b; Canady & Allen 1982
Eq. 2). Area ``A_im = pi a b = pi b^2 / sin(epsilon)`` (Eq. 13).

Delivered-irradiance formula (Canady & Allen 1982 NASA TP-2065 Eq. 9 rewritten
in the notation used here, unified with McInnes 1999 §2.6.1 specular
decomposition and with the r_sun-sail dependence split into I_0 and
alpha):

    I_target = eta * chi * I_0(r_sun_sail) * A_sail
               * cos(psi/2) * sin(epsilon)
               / (pi * [d * tan(alpha/2)]^2)                   (C-A 9')

where
    eta = rho * s                      specular reflectance fraction
                                       from McInnes SailOptical
    chi                                atmospheric transmission (default 1)
    I_0(r)  = L_Sun / (4 pi r^2)       solar irradiance at sail (W/m^2)
    A_sail                             reflective area (m^2)
    cos(psi/2)                         incidence half-angle cosine at
                                       the mirror (= dot(n_hat,
                                       s_hat_sun) = dot(n_hat,
                                       s_hat_target) at the bisector)
    sin(epsilon)                       sin of sail elevation at target
                                       = cos(incidence zenith at target)
    d                                  slant range sail -> target (m)
    alpha                              2 arcsin(R_sun / r_sun_sail)

Three independent primary references all yield this same formula:

  - Canady, J.E. Jr. & Allen, J.L. Jr. (1982), "Illumination from
    space with orbiting solar-reflector spacecraft", NASA Technical
    Paper 2065, Langley Research Center. Eqs. 1-3 (image geometry)
    and Eq. 9 (delivered intensity).

  - Çelik, O. & McInnes, C.R. (2022), "An analytical model for solar
    energy reflected from space with selected applications",
    *Advances in Space Research* 69:647-663. Eqs. 8a/8b (ellipse
    axes), Eq. 13 (point-source image area), Eq. 15 (finite-mirror
    correction), Eq. 16 (power density).

  - Viale, A. et al. (2023), "A reference architecture for orbiting
    solar reflectors to enhance terrestrial solar power plant
    output", *Advances in Space Research* 72:1304-1348. Eq. 12
    (power density with reflectivity and atmospheric transmission).

All three forms are cross-checked against each other in the test
suite: (C-A Eq. 9) and (C-M Eq. 13 + Eq. 16) agree to machine
precision (they use identical tan(alpha/2) image-radius formulas),
while the Born & Wolf (Principles of Optics §4.8) radiance-
conservation form

    I_target = eta * chi * B_sun * Omega_mirror * sin(epsilon)     (B&W)
    B_sun    = L / (4 pi^2 R_sun^2)                                ~2e7 W/m^2/sr
    Omega_mirror = A_sail * cos(psi/2) / d^2                       (sr)

agrees with the C-A form to relative ``(R_sun / r_sun_sail)^2`` only
(~1e-5 at Mars, ~2e-5 at Earth). The residual follows from C-A
writes ``b = d tan(alpha/2)`` for the beam semi-minor axis whereas a
strict brightness-conservation derivation uses ``d sin(alpha/2)`` (the
projected sun radius at distance d); the two agree to small-angle.
Both forms are exact to leading order in ``R_sun/r_sun``, and
``sin^2 = tan^2 / (1 + tan^2)`` so their ratio is ``1 - (R_sun/r_sun)^2``
(pinned by a test). For the Mars applications considered here this
difference is ~10^-5 -- many orders of magnitude below SRP
reflectivity uncertainty, atmospheric transmission uncertainty, and
the cos(psi/2) sampling error in any realistic attitude profile.

Modeling choices.

  1. Sun subtense angle is ``alpha = 2 arcsin(R_sun / r_sun_sail)``
     using the exact sail-to-Sun distance from SPICE. Çelik &
     McInnes 2022 Eq. 9 writes ``alpha = 2 arctan((R_sun - R_planet)
     / rho_sun)`` with a planet-radius subtraction that is <0.5% at
     Mars but invalid for a sail at arbitrary altitude
     (the sail's position, not the planet centre, is what governs
     the sun's subtended angle from the sail). The exact form is used;
     the difference is well below all other tolerances.

  2. Atmospheric transmission ``chi`` is a scalar kwarg with a vacuum default
     of 1.0. This module does not implement a Mars-specific transmission model.

  3. Finite-mirror correction (C-M Eq. 15) is OFF by default.
     Turn on via ``include_finite_mirror_correction=True`` + pass
     ``sail_diameter_m``. Correction is ``~(D_M / (2 d tan(alpha/2)))^2``
     ~1e-4 at a 32 m sail, 500 km Mars LMO slant; cheap to include but
     negligible at representative scales, so the default does not require
     callers to supply sail-diameter geometry.

  4. Pinhole-regime check. The Canady-Allen / Çelik-McInnes
     formula treats the mirror as smaller than the reflected sun
     image: ``D_mirror_proj << d * alpha``. When this is violated
     (mirror-limited / "hot spot" regime), each target point sees the
     full sun through the mirror and the irradiance saturates at
     ``eta * I_0 * cos(psi/2) * sin(epsilon)`` (a "second sun"
     worth of flux). ``ValueError`` is raised if the finite-mirror
     correction term exceeds 20% of the sun-spread term (i.e.
     ``D_mirror_proj / (2 d tan(alpha/2)) > 0.2``) because physics
     departs from the pinhole regime there and a different formula
     applies. For typical Mars-LMO sails this check never fires.

  5. The target is a single surface point, and the reported quantity is
     mean irradiance at that point. Çelik & McInnes 2022 §3.1 handles
     the extended-target case (image partially outside the target
     boundary), which is outside the present scope.

  6. Non-specular reflection (the McInnes ``rho * (1-s)`` diffuse
     component) scatters Lambertianly over 2pi sr and delivers
     negligible irradiance at any target in the beam's sightline.
     An off-specular diffuse lobe is not modelled and is only relevant
     if sail material departs
     significantly from near-specular (``s >> 0.9``).

This module is SPICE-FREE. Sail position, sail-to-Sun distance,
slant range, and elevation/bisector angles are all expected to be
supplied by the caller (typically ``reflectors.visibility.
find_delivery_windows``, which already fetches these quantities
via a single per-sample SPICE call). Keeping ``beam.py`` pure
radiometry simplifies testing and keeps the dependencies one-way:
``visibility`` depends on ``beam``, not the other way round.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

from reflectors.solar_constants import (
    solar_brightness_W_per_m2_per_sr,
    solar_irradiance_W_per_m2_at,
)
from reflectors.srp import SailOptical, SolarSail


logger = logging.getLogger(__name__)


# Threshold on mirror-to-sun-spread ratio above which the pinhole-regime
# assumption underlying Canady-Allen / Çelik-McInnes breaks. ``0.2`` means
# the finite-mirror quadrature term contributes > 4% to the beam semi-minor
# (``sqrt(0.2^2 + 1^2) / 1 - 1 ~= 0.02``); physics departs from pinhole here
# and the mirror-limited "hot spot" regime applies instead. For a 32 m JPL
# square sail at 500 km Mars LMO slant, actual ratio is ~0.01 -- comfortably
# inside the regime.
_PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX: float = 0.2


# Imported Sun radius for the subtense formula. The IAU 2015
# defining value (from solar_constants.SOLAR_RADIUS_KM) rather than
# re-reading the PCK each call; test_solar_constants pins equality
# between the two so this is safe.
from reflectors.solar_constants import SOLAR_RADIUS_KM


# ---------------------------------------------------------------------------
# Sun subtense angle at the sail
# ---------------------------------------------------------------------------


def sun_angular_diameter_rad(sail_to_sun_km: float) -> float:
    """Full angular diameter of the Sun as seen from the sail, radians.

    ``alpha = 2 * arcsin(R_sun / d_sail_to_sun)``.

    At 1 AU: ``alpha = 0.0093007`` rad (~32 arcmin), matching the
    Canady-Allen 1982 p. 7 quote ``0.0093 d`` for the spot diameter,
    and Çelik & McInnes 2022's ``alpha = 0.0093`` rad at 1 AU
    reference.

    At Mars mean distance (1.524 AU): ``alpha = 0.0061036`` rad
    (~21 arcmin, ~0.35°).

    This function uses arcsin (geometrically exact for a sphere) rather than
    ``2 arctan(R_sun / d)`` (the flat-disc approximation). For
    ``R_sun / d ~ 0.0046`` (1 AU) the two agree to rel 1e-11; the
    exact form has no additional computational cost.

    Parameters
    ----------
    sail_to_sun_km
        Sail-to-Sun distance in km. Must be strictly positive and
        greater than R_sun (otherwise the sail is inside the Sun,
        which is nonphysical; arcsin would return nan).

    Raises
    ------
    ValueError
        If ``sail_to_sun_km <= SOLAR_RADIUS_KM``.
    """
    if sail_to_sun_km <= SOLAR_RADIUS_KM:
        raise ValueError(
            f"sun_angular_diameter_rad: sail_to_sun_km={sail_to_sun_km!r} "
            f"must exceed SOLAR_RADIUS_KM={SOLAR_RADIUS_KM}; sail cannot "
            "be inside the Sun."
        )
    return 2.0 * math.asin(SOLAR_RADIUS_KM / float(sail_to_sun_km))


def sun_half_angle_tan(sail_to_sun_km: float) -> float:
    """``tan(alpha/2)`` for direct use in beam-image geometry formulae.

    Çelik & McInnes 2022 Eq. 8a-8b and Canady & Allen 1982 Eqs. 1-2
    are expressed in terms of ``tan(alpha/2)``. This helper is a
    convenience around ``math.tan(sun_angular_diameter_rad(...)/2)``
    and makes the formulae in ``beam_image_*`` read directly off the
    cited equations. Returns a dimensionless float.
    """
    return math.tan(0.5 * sun_angular_diameter_rad(sail_to_sun_km))


# ---------------------------------------------------------------------------
# Reflected-beam image geometry
# ---------------------------------------------------------------------------


def beam_image_semi_minor_km(
    slant_km: float, sail_to_sun_km: float
) -> float:
    """Semi-minor axis of the reflected sun image on the target surface, km.

    Canady & Allen 1982 Eq. 1:   ``b = d * tan(alpha/2)``.
    Çelik & McInnes 2022 Eq. 8a: same form. Perpendicular to the
    beam's ground trace. Equals the radius of the circular image on
    a plane perpendicular to the beam direction at distance ``d``
    from the sail; ``sin(epsilon)`` enters only in the semi-major
    direction (see ``beam_image_semi_major_km``).

    Parameters
    ----------
    slant_km
        Slant range from the sail to the target, km. Strictly positive.
    sail_to_sun_km
        Sail-to-Sun distance in km; strictly > R_sun.
    """
    if slant_km <= 0.0:
        raise ValueError(
            f"beam_image_semi_minor_km: slant_km must be > 0, got {slant_km!r}"
        )
    return float(slant_km) * sun_half_angle_tan(sail_to_sun_km)


def beam_image_semi_major_km(
    slant_km: float, sail_to_sun_km: float, sin_elevation: float
) -> float:
    """Semi-major axis of the reflected sun image on the target surface, km.

    Canady & Allen 1982 Eq. 2 / Çelik & McInnes 2022 Eq. 8b:
    ``a = b / sin(epsilon)`` where ``b`` is the semi-minor axis and
    ``epsilon`` is the sail's elevation angle above the target's
    local horizon plane.

    Stretches the circular perpendicular cross-section along the
    ground trace; reduces to a circle (``a = b``) at
    ``sin(epsilon) = 1`` (sail at zenith above the target) and
    diverges as ``epsilon -> 0`` (sail on the horizon).

    Parameters
    ----------
    slant_km, sail_to_sun_km
        See ``beam_image_semi_minor_km``.
    sin_elevation
        ``sin(epsilon)`` where ``epsilon`` is the sail's elevation
        above the target's local horizon plane, in ``[-1, 1]``.
        Must be strictly positive (the sail must be above the
        target's horizon plane for any beam to reach the ground).
    """
    if sin_elevation <= 0.0:
        raise ValueError(
            f"beam_image_semi_major_km: sin_elevation must be > 0, got "
            f"{sin_elevation!r}; sail is below the target's horizon "
            "plane and no beam can reach the target."
        )
    return beam_image_semi_minor_km(slant_km, sail_to_sun_km) / float(
        sin_elevation
    )


def beam_image_area_m2(
    slant_km: float,
    sail_to_sun_km: float,
    sin_elevation: float,
    *,
    sail_diameter_m: float = 0.0,
) -> float:
    """Area of the reflected sun image on the target surface, m^2.

    Default form (point-source mirror, Çelik & McInnes 2022 Eq. 13):

        A_im = pi * [d * tan(alpha/2)]^2 / sin(epsilon)

    Finite-mirror form (Çelik & McInnes 2022 Eq. 15), activated by
    passing ``sail_diameter_m > 0``:

        A_im = (pi / sin(epsilon))
               * ((d * tan(alpha/2))^2 + (D_M * cos(psi/2) / 2)^2)

    The Eq. 15 form assumes a disc mirror of diameter ``D_M``
    oriented at the bisector pitch ``psi/2``; the cos(psi/2) factor
    accounts for the mirror's projected diameter as seen along the
    beam axis. Because cos(psi/2) is a caller input, the finite-mirror
    form is invoked through the
    ``delivered_surface_irradiance_W_per_m2`` front-end instead --
    this helper's ``sail_diameter_m`` kwarg applies the simpler
    quadrature-sum form that ignores cos(psi/2) and is correct at
    the face-on limit. For ``D_M`` at representative sail scales the two forms
    differ by <1% at any geometry; the distinction becomes
    numerically significant only if the mirror diameter approaches
    the beam-spread scale (``D_M ~ d * alpha``), in which case the
    pinhole-regime check in ``delivered_surface_irradiance_...``
    raises.

    Returns m^2 (note: ``slant_km`` is km but the answer is m^2 via
    the ``1000^2`` implicit conversion).

    Parameters
    ----------
    slant_km, sail_to_sun_km, sin_elevation
        See ``beam_image_semi_major_km``.
    sail_diameter_m
        Optional mirror linear dimension in m; default 0 (point-
        source form). If > 0, the finite-mirror quadrature term
        ``(D_M / 2)^2`` is added to ``b^2`` before dividing by
        ``sin(epsilon)`` and multiplying by pi.
    """
    if sail_diameter_m < 0.0:
        raise ValueError(
            f"beam_image_area_m2: sail_diameter_m must be >= 0, got "
            f"{sail_diameter_m!r}"
        )
    b_km = beam_image_semi_minor_km(slant_km, sail_to_sun_km)
    b_m = b_km * 1000.0
    if sin_elevation <= 0.0:
        raise ValueError(
            f"beam_image_area_m2: sin_elevation must be > 0, got "
            f"{sin_elevation!r}"
        )
    # Eq. 13 term; m^2.
    b_squared_m2 = b_m * b_m
    if sail_diameter_m > 0.0:
        # Eq. 15 quadrature; (D_M/2)^2 in m^2. cos(psi/2) factor on the
        # mirror-projected term is applied by the irradiance calculation.
        mirror_radius_m = 0.5 * float(sail_diameter_m)
        b_squared_m2 += mirror_radius_m * mirror_radius_m
    return math.pi * b_squared_m2 / float(sin_elevation)


# ---------------------------------------------------------------------------
# Specular reflectance fraction (bridge from McInnes 6-parameter SRP model)
# ---------------------------------------------------------------------------


def specular_reflectance(optical: SailOptical) -> float:
    """Specular reflectance fraction ``eta = rho * s`` from the McInnes model.

    Canady & Allen 1982 Eq. 5 uses a single "mirror reflectance"
    coefficient ``rho`` multiplied by a "mirror flatness" coefficient
    ``mu`` (their Eq. 5). Viale et al. 2023 Eq. 12 uses a single
    ``eta`` ~ 0.92 cited back to Canady-Allen (aluminum-coated
    Mylar/Kapton at 0.55 um).

    The McInnes 1999 §2.6.1 six-parameter optical model (Eqs.
    M-2.46-2.57) decomposes the reflected light into:

      * specular fraction       ``rho * s``      forms a directed beam
      * diffuse (Lambertian)    ``rho * (1-s)``  scattered over 2pi sr
      * absorbed + thermal      ``1 - rho``      re-emitted diffusely

    Only the specular fraction forms a target-directed beam.
    ``eta_specular = rho * s`` replaces the Canady-Allen ``mu * rho``,
    keeping the reflected-beam physics internally consistent with the
    SRP force model in ``reflectors.srp``. The two products happen to
    land at similar values for realistic sails:

      * JPL square sail (McInnes Tab. 2.1):  rho*s = 0.88*0.94 = 0.827
      * Canady-Allen 1982 p. 16 (aluminum): mu*rho = 0.91*0.92 = 0.837

    Pinned in ``test_beam.py::TestMcInnesIntegration``.

    Parameters
    ----------
    optical
        ``SailOptical`` instance from ``reflectors.srp``.
    """
    return float(optical.rho) * float(optical.s)


# ---------------------------------------------------------------------------
# Delivered surface irradiance at the target
# ---------------------------------------------------------------------------


def delivered_surface_irradiance_W_per_m2(
    sail: SolarSail,
    slant_km: float,
    sail_to_sun_km: float,
    cos_alpha_bisector: float,
    sin_elevation: float,
    *,
    atmospheric_transmission: float = 1.0,
    include_finite_mirror_correction: bool = False,
    sail_diameter_m: float = 0.0,
) -> float:
    """Mean irradiance at the target surface at the center of the reflected
    sun image, in W/m^2.

    Canady & Allen (1982) Eq. 9 in the notation used here:

        I_target = eta * chi * I_0(r_sun_sail) * A_sail
                   * cos(psi/2) * sin(epsilon)
                   / (pi * [d * tan(alpha/2)]^2)

    When ``include_finite_mirror_correction=True``, the denominator
    adopts Çelik & McInnes 2022 Eq. 15 form:

        / (pi * ((d * tan(alpha/2))^2 + (D_M * cos(psi/2) / 2)^2))

    where the cos(psi/2) factor projects the mirror's physical
    diameter onto the plane perpendicular to the beam.

    Returns zero if ``cos_alpha_bisector <= 0`` (sun-target geometry
    is degenerate; no directed beam forms) or if
    ``sin_elevation <= 0`` (sail is below target's horizon).

    Raises ValueError if the pinhole-regime assumption breaks, i.e.
    if the finite-mirror term exceeds 20% of the sun-spread term (see
    module docstring, modeling choice 4). This can happen only when
    a large mirror comes very close to a target (``D_M ~ d * alpha``),
    which is not a regime Mars LMO sails ever reach.

    Parameters
    ----------
    sail
        ``SolarSail`` bus from ``reflectors.srp``. Reads ``sail.area_m2``
        and ``sail.optical`` (for the McInnes ``rho * s`` specular
        reflectance fraction).
    slant_km
        Sail-to-target slant range, km. Must be > 0.
    sail_to_sun_km
        Sail-to-Sun distance, km. Must exceed SOLAR_RADIUS_KM.
    cos_alpha_bisector
        Cosine of the half-angle between sun-direction and
        target-direction as seen from the sail. At bisector pointing
        (``reflectors.visibility.bisector_normal``) this is also the
        cosine of the incidence angle at the mirror (``psi/2`` in
        Viale/Canady-Allen nomenclature). In [-1, 1]; returns 0 if
        non-positive.
    sin_elevation
        Sin of the sail's elevation angle above the target's horizon
        plane. In [-1, 1]; returns 0 if non-positive.
    atmospheric_transmission
        Scalar ``chi`` multiplying the entire formula. Default 1 (no
        atmosphere). Must be in [0, 1].
    include_finite_mirror_correction
        If True, use Çelik & McInnes Eq. 15 (finite-mirror quadrature
        form) instead of the default Eq. 13 point-source form. Requires
        ``sail_diameter_m > 0``.
    sail_diameter_m
        Mirror linear dimension in m, only used when
        ``include_finite_mirror_correction=True``. For a square sail of
        area A_sail, the natural choice is ``sqrt(A_sail)``; for a
        hexagonal sail, the side-to-side distance; for a disc sail,
        the diameter. This is a first-order correction (~1e-4 at representative
        scales) so the exact geometry choice matters only at the
        sub-percent level.

    Returns
    -------
    float
        Mean irradiance at the target, W/m^2. Zero when the geometry
        precludes any delivered flux.
    """
    if atmospheric_transmission < 0.0 or atmospheric_transmission > 1.0:
        raise ValueError(
            f"atmospheric_transmission must be in [0, 1], got "
            f"{atmospheric_transmission!r}"
        )
    # Short-circuit degenerate geometries before any numerics. A caller
    # whose bisector gate rejects this geometry would otherwise get 0.0
    # by construction below, but explicit short-circuits read clearer
    # and avoid the "0 * inf = nan" trap if another sub-expression
    # diverges.
    if cos_alpha_bisector <= 0.0:
        return 0.0
    if sin_elevation <= 0.0:
        return 0.0
    if slant_km <= 0.0:
        raise ValueError(f"slant_km must be > 0, got {slant_km!r}")

    # eta = rho * s (McInnes specular fraction).
    eta = specular_reflectance(sail.optical)

    # I_0 at sail, W/m^2.
    I_0 = solar_irradiance_W_per_m2_at(sail_to_sun_km)

    # Beam cross-section perpendicular to the beam at slant d: radius
    # b = d * tan(alpha/2). Evaluate once, reuse for the pinhole check.
    b_km = beam_image_semi_minor_km(slant_km, sail_to_sun_km)
    b_m = b_km * 1000.0
    sun_spread_radius_m = b_m

    # Optional finite-mirror correction term (C-M Eq. 15). This projects
    # the mirror's physical diameter onto the beam-perpendicular plane
    # via cos(psi/2), so the mirror's half-projected-diameter is
    # (D_M / 2) * cos(psi/2). Keeping cos(psi/2) here (not in
    # beam_image_area_m2) is what makes the combined form match C-M
    # Eq. 15 exactly.
    mirror_contrib_m2 = 0.0
    if include_finite_mirror_correction:
        if sail_diameter_m <= 0.0:
            raise ValueError(
                "include_finite_mirror_correction=True requires "
                "sail_diameter_m > 0"
            )
        mirror_projected_radius_m = (
            0.5 * float(sail_diameter_m) * float(cos_alpha_bisector)
        )
        mirror_contrib_m2 = mirror_projected_radius_m * mirror_projected_radius_m

        # Pinhole-regime guard: mirror contribution must be small
        # compared to sun-spread contribution. If not, the beam
        # geometry has entered the mirror-limited regime where each
        # target point sees the full sun through the mirror (a "hot
        # spot" saturating at eta * I_0) -- that is a different
        # formula, not implemented here. See module docstring, modeling
        # choice 4.
        if mirror_projected_radius_m > (
            _PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX * sun_spread_radius_m
        ):
            raise ValueError(
                "delivered_surface_irradiance_W_per_m2: pinhole-regime "
                "assumption violated. Mirror projected radius "
                f"{mirror_projected_radius_m:.3e} m exceeds "
                f"{_PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX:.1%} of sun-spread "
                f"radius {sun_spread_radius_m:.3e} m at slant "
                f"{slant_km} km, alpha={sun_angular_diameter_rad(sail_to_sun_km):.3e} "
                "rad. Mirror-limited 'hot spot' regime applies; a "
                "different formula is needed."
            )

    # Beam image area on the ground (m^2).
    A_im_m2 = math.pi * (sun_spread_radius_m * sun_spread_radius_m + mirror_contrib_m2) / float(
        sin_elevation
    )

    # Power intercepted and specularly reflected (W). Canady & Allen
    # Eq. 5 with mu=1 (the flat-sail assumption is captured in the
    # SRP model's "s" specular fraction instead).
    P_reflected_W = eta * I_0 * sail.area_m2 * float(cos_alpha_bisector)

    # Mean irradiance at target center (W/m^2). Atmospheric
    # transmission multiplies the delivered flux.
    I_target = float(atmospheric_transmission) * P_reflected_W / A_im_m2
    return I_target


# ---------------------------------------------------------------------------
# Radiance-conservation cross-check form (Born & Wolf Eq.)
# ---------------------------------------------------------------------------


def delivered_surface_irradiance_via_radiance(
    sail: SolarSail,
    slant_km: float,
    cos_alpha_bisector: float,
    sin_elevation: float,
    *,
    atmospheric_transmission: float = 1.0,
) -> float:
    """Same as ``delivered_surface_irradiance_W_per_m2`` but via the
    radiance-conservation form (Born & Wolf §4.8):

        I_target = eta * chi * B_sun * Omega_mirror * sin(epsilon)
        B_sun    = L / (4 pi^2 R_sun^2)
        Omega_mirror = A_sail * cos(psi/2) / d^2

    The two forms are identical to machine precision because the
    sail-Sun distance dependence in ``I_0`` and ``tan^2(alpha/2)``
    cancels. Implemented as a separate entry point so tests can pin
    the agreement and so readers learning the physics can see both
    forms side-by-side.

    No ``sail_to_sun_km`` argument: this form is r_sun-independent.

    Parameters
    ----------
    sail, slant_km, cos_alpha_bisector, sin_elevation, atmospheric_transmission
        See ``delivered_surface_irradiance_W_per_m2``.
    """
    if cos_alpha_bisector <= 0.0 or sin_elevation <= 0.0:
        return 0.0
    if slant_km <= 0.0:
        raise ValueError(f"slant_km must be > 0, got {slant_km!r}")
    if atmospheric_transmission < 0.0 or atmospheric_transmission > 1.0:
        raise ValueError(
            f"atmospheric_transmission must be in [0, 1], got "
            f"{atmospheric_transmission!r}"
        )
    eta = specular_reflectance(sail.optical)
    B_sun = solar_brightness_W_per_m2_per_sr()
    # Omega_mirror: km in, steradians out. km * km / km^2 -> dimensionless.
    d_m = float(slant_km) * 1000.0
    # sail.area_m2 is already in m^2; slant is in m.
    Omega_mirror_sr = sail.area_m2 * float(cos_alpha_bisector) / (d_m * d_m)
    return (
        eta * float(atmospheric_transmission) * B_sun
        * Omega_mirror_sr * float(sin_elevation)
    )


# ---------------------------------------------------------------------------
# Viale et al. 2023 Eq. 12 cross-check form
# ---------------------------------------------------------------------------


def delivered_power_at_target_W(
    sail: SolarSail,
    slant_km: float,
    sail_to_sun_km: float,
    cos_alpha_bisector: float,
    sin_elevation: float,
    target_area_m2: float,
    *,
    atmospheric_transmission: float = 1.0,
) -> float:
    """Total specularly-reflected power delivered to a finite target area,
    W. Viale et al. 2023 Eq. 12.

    Two regimes:

      * If the target area is larger than or equal to the reflected
        sun image area, it intercepts the FULL reflected power
        ``P = eta * chi * I_0 * A_sail * cos(psi/2)`` (Canady-Allen
        Eq. 5). Further increases in target area deliver no
        additional power (assuming the image is fully contained
        within the target).

      * If the target area is smaller, it intercepts a fraction
        ``target_area / A_im`` of the total reflected power.

    This is Çelik & McInnes 2022 §3.1 in its simplest form
    (target assumed either fully inside or fully outside the
    image; no partial-overlap integration).

    For point-irradiance calculations, this is only needed as a cross-check
    against the irradiance
    form; see ``test_beam.py::TestViale2023Equation12``.

    Parameters
    ----------
    sail, slant_km, sail_to_sun_km, cos_alpha_bisector, sin_elevation
        See ``delivered_surface_irradiance_W_per_m2``.
    target_area_m2
        Finite target area in m^2 (positive).
    atmospheric_transmission
        See ``delivered_surface_irradiance_W_per_m2``.
    """
    if target_area_m2 <= 0.0:
        raise ValueError(
            f"target_area_m2 must be > 0, got {target_area_m2!r}"
        )
    if cos_alpha_bisector <= 0.0 or sin_elevation <= 0.0:
        return 0.0
    A_im_m2 = beam_image_area_m2(
        slant_km, sail_to_sun_km, sin_elevation
    )
    eta = specular_reflectance(sail.optical)
    I_0 = solar_irradiance_W_per_m2_at(sail_to_sun_km)
    P_total_W = (
        eta * float(atmospheric_transmission) * I_0
        * sail.area_m2 * float(cos_alpha_bisector)
    )
    if target_area_m2 >= A_im_m2:
        return P_total_W
    return P_total_W * (target_area_m2 / A_im_m2)


# ---------------------------------------------------------------------------
# Footprint semi-axes in km (convenience, used by find_delivery_windows)
# ---------------------------------------------------------------------------


def beam_footprint_semi_axes_km(
    slant_km: float, sail_to_sun_km: float, sin_elevation: float
) -> Tuple[float, float]:
    """Semi-major and semi-minor axes of the reflected image on the surface, km.

    Returns ``(semi_major_km, semi_minor_km)``. Semi-major is along
    the beam's ground trace and scales as ``1/sin(epsilon)``;
    semi-minor is perpendicular to it. Equal (circle) at zenith.

    Convenience wrapper around ``beam_image_semi_major_km`` +
    ``beam_image_semi_minor_km``; used by
    ``reflectors.visibility.find_delivery_windows`` to populate
    per-window footprint scalars.
    """
    b = beam_image_semi_minor_km(slant_km, sail_to_sun_km)
    a = beam_image_semi_major_km(slant_km, sail_to_sun_km, sin_elevation)
    return (a, b)
