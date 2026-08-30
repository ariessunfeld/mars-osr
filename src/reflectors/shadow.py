"""Binary geometric umbra test for a Mars-centered sail.

Target physics: decide whether a sail in Mars orbit is inside Mars's umbra
-- the region where the Mars disc fully occults the Sun's disc as seen
from the sail. Used by the SRP module (``reflectors.srp``) to gate the
solar-radiation-pressure force to zero during a total eclipse.

Binary on/off by design: a sail is either in the umbra (shadow factor 0)
or outside it (shadow factor 1). The partial-eclipse (penumbra) transition
is deliberately not modelled here. Penumbra effects smooth the SRP amplitude
over a ~1 s to few-s
transit at each terminator crossing and average to a small secular
correction; they matter for long-term sun-sync maintenance but not for
first-pass dynamics.

Reference (primary): Montenbruck, O. and Gill, E. (2000), *Satellite
Orbits: Models, Methods, and Applications*, Springer, §3.4 ("Solar
Radiation Pressure"), pp.77-82, where the shadow geometry is expressed
via the apparent angular diameters of the occulting body and the Sun as
seen from the satellite.

Formulation.

    Let ``s`` be the sail position vector in Mars-centered J2000, and
    ``s_sun`` the Sun position in the same frame (fetched via SPICE).
    Define

        d_M     = |s|                       sail-to-Mars distance
        d_S     = |s_sun - s|               sail-to-Sun distance
        sigma_M = asin(R_Mars / d_M)        Mars angular radius from sail
        sigma_S = asin(R_Sun / d_S)         Sun angular radius from sail
        cos D   = ((-s) . (s_sun - s))      numerator of angular sep.
                  / (d_M * d_S)             D in [0, pi]

    The Mars disc fully covers the Sun disc (umbra) iff

        D + sigma_S  <=  sigma_M.                                   (*)

    Geometric cross-check: on the anti-Sun axis at distance
    ``L_umbra = R_Mars * d_{Mars,Sun} / (R_Sun - R_Mars)`` behind Mars,
    ``sigma_M == sigma_S`` and ``D == 0``, so (*) holds with equality --
    the umbra cone tapers to a point there. For Mars at 1.524 AU,
    ``L_umbra ~ 1.12e6 km``; any low Mars orbit is therefore very deep
    inside the full-occultation regime whenever it crosses the anti-Sun
    side.

Body-size convention. Mars is treated as a sphere of *equatorial* radius
``R_{Mars,eq} = 3396.19 km`` (from the loaded PCK via
``surface.mars_equatorial_radius_km``). Using the equatorial radius
slightly over-bounds the umbra at high-latitude shadow crossings, since
the polar radius is 3376.20 km (0.59% smaller). The systematic bias in
"fraction of orbit in umbra" is therefore at most ~0.6% and is in the
conservative direction (over-estimating shadow time). An oblate-silhouette
upgrade belongs to the penumbra item in the checklist.

The Sun is modelled as a sphere of radius ``R_Sun = 6.957e5 km`` from
``BODY10_RADII`` in ``pck00011.tpc``. This matches the IAU 2015
Resolution B3 nominal solar radius
(Mamajek et al. 2015 / Prsa et al. 2016, AJ 152:41) to machine precision;
``tests/test_shadow.py`` pins the value against the IAU constant so a
PCK change triggers an explicit test failure.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import spiceypy as spice

from reflectors.ephemeris import sun_state_j2000
from reflectors.surface import mars_equatorial_radius_km


logger = logging.getLogger(__name__)


SUN_NAIF_ID = 10
MARS_NAIF_ID = 499


# ---------------------------------------------------------------------------
# Solar radius
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def sun_radius_km() -> float:
    """Solar radius in km, read from ``BODY10_RADII`` in the loaded PCK.

    For ``pck00011.tpc`` this returns 6.957e5 km (all three triaxial
    components equal; the Sun is modeled as a sphere). The value matches
    the IAU 2015 Resolution B3 nominal solar radius
    ``R_Sun_N = 6.957e5 km`` to machine precision. See Prsa et al. 2016,
    "Nominal values for selected solar and planetary quantities",
    Astronomical Journal 152:41, https://doi.org/10.3847/0004-6256/152/2/41.

    Averaging the three triaxial components (rather than picking one)
    keeps this helper agnostic to a PCK that reports
    distinct equatorial / polar solar radii.
    """
    n, radii = spice.bodvcd(SUN_NAIF_ID, "RADII", 3)
    assert n == 3
    r_eq_a, r_eq_b, r_pol = float(radii[0]), float(radii[1]), float(radii[2])
    return (r_eq_a + r_eq_b + r_pol) / 3.0


# ---------------------------------------------------------------------------
# Sun position helper
# ---------------------------------------------------------------------------


def _sun_position_j2000_km(et: float, observer_naif_id: int) -> np.ndarray:
    """SPICE-fed Sun position relative to an observer, J2000 axes, km."""
    return np.asarray(sun_state_j2000(et, observer_naif_id)[:3], dtype=float)


# ---------------------------------------------------------------------------
# Umbra test
# ---------------------------------------------------------------------------


def in_mars_umbra(
    r_sat_j2000_km: np.ndarray,
    et: float,
    observer_naif_id: int = MARS_NAIF_ID,
    *,
    sun_position_j2000_km: Optional[np.ndarray] = None,
    central_radius_km: Optional[float] = None,
) -> bool:
    """Return True iff the sail is inside the central body's umbra at ``et``.

    The angular-disc total-eclipse geometry is body-generic; only the occulting
    disc radius differs. ``central_radius_km`` defaults to ``None`` -> Mars
    equatorial radius. Pass the Earth equatorial radius for Earth escape. Despite
    the Mars-specific function name, the math is body-agnostic given the radius.

    Implements the angular-disc total-eclipse condition
    ``D + sigma_Sun <= sigma_Mars`` where ``D`` is the angular
    separation of Mars and Sun centres as seen from the sail, and
    ``sigma_{Mars,Sun}`` are the corresponding angular radii computed
    from the equatorial Mars radius and the SPICE-pool solar radius.

    Parameters
    ----------
    r_sat_j2000_km
        Sail position in observer-centered J2000 axes, km, shape (3,).
    et
        SPICE ephemeris time (TDB seconds past J2000). Used to fetch the
        Sun's position relative to the observer if
        ``sun_position_j2000_km`` is not supplied.
    observer_naif_id
        Central body the sail position is referenced to. Default NAIF
        499 (Mars planet centre), matching the propagator convention.
    sun_position_j2000_km
        Optional: pre-fetched Sun position relative to the observer in
        J2000 axes, km, shape (3,). Allows callers (e.g. ``srp.py``)
        that already have the Sun state in hand to avoid a redundant
        ``spkezr`` call in the RHS hot loop. When ``None`` (default) the
        position is fetched via SPICE.

    Raises
    ------
    ValueError
        If the sail position lies inside the Mars equatorial reference
        sphere (``d_M <= R_{Mars,eq}``) -- that is a propagation
        pathology (sail below the destruction boundary or numerical
        blow-up), not an umbra question.
    """
    # The occulting body is the one at the frame origin (this function models
    # its disc occulting the Sun, taking sail->occulter = -s). If that body IS
    # the Sun, the test is degenerate -- the occulting disc coincides with the
    # illuminating disc (D=0, sigma_occulter == sigma_Sun), so the umbra
    # condition would hold with spurious equality and report a permanent
    # eclipse. A body cannot eclipse the light source it is: return "sunlit".
    # (A heliocentric sail occulted by a PLANET is a different geometry -- the
    # occulter is not at the origin -- and would need a separate test; it is
    # physically negligible during the interplanetary cruise.) Mars/Earth
    # callers (observer 499/399) are unaffected.
    if observer_naif_id == SUN_NAIF_ID:
        return False

    s = np.asarray(r_sat_j2000_km, dtype=float)
    d_M = float(np.linalg.norm(s))
    R_M = (
        mars_equatorial_radius_km()
        if central_radius_km is None
        else float(central_radius_km)
    )
    if d_M <= R_M:
        raise ValueError(
            f"sail inside the Mars reference sphere: |r_sat| = {d_M:.3f} km "
            f"<= R_Mars_eq = {R_M:.3f} km; umbra test is undefined there"
        )
    if sun_position_j2000_km is None:
        s_sun = _sun_position_j2000_km(et, observer_naif_id)
    else:
        s_sun = np.asarray(sun_position_j2000_km, dtype=float)
    sat_to_sun = s_sun - s
    d_S = float(np.linalg.norm(sat_to_sun))
    R_S = sun_radius_km()

    sigma_M = float(np.arcsin(R_M / d_M))
    sigma_S = float(np.arcsin(R_S / d_S))

    # Angular separation D between "sail -> Mars centre" and "sail -> Sun".
    # Mars centre is at the origin of this frame, so sail -> Mars = -s.
    u_M = -s / d_M
    u_S = sat_to_sun / d_S
    cos_D = float(np.clip(np.dot(u_M, u_S), -1.0, 1.0))
    D = float(np.arccos(cos_D))

    return (D + sigma_S) <= sigma_M


def shadow_factor(
    r_sat_j2000_km: np.ndarray,
    et: float,
    observer_naif_id: int = MARS_NAIF_ID,
    *,
    sun_position_j2000_km: Optional[np.ndarray] = None,
    central_radius_km: Optional[float] = None,
) -> float:
    """Scalar solar-irradiance multiplier due to the central body's shadow.

    Returns 0.0 inside the umbra, 1.0 outside. Binary by design; a
    penumbra-capable implementation can return intermediate values in [0, 1]
    without changing call sites in ``reflectors.srp``. Accepts an optional
    ``sun_position_j2000_km`` to share the SPICE fetch with a caller that
    already has the Sun state in hand, and an optional ``central_radius_km``
    (default ``None`` -> Mars radius) to set the occulting-disc radius for a
    non-Mars central body.
    """
    return (
        0.0
        if in_mars_umbra(
            r_sat_j2000_km,
            et,
            observer_naif_id,
            sun_position_j2000_km=sun_position_j2000_km,
            central_radius_km=central_radius_km,
        )
        else 1.0
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def umbra_cone_length_km(
    et: float,
    observer_naif_id: int = MARS_NAIF_ID,
    *,
    central_radius_km: Optional[float] = None,
) -> float:
    """Analytical length of Mars's umbra cone behind Mars, km.

    ``L = R_Mars * d_{Mars,Sun} / (R_Sun - R_Mars)``.

    At distance ``L`` along the anti-Sun axis from the Mars centre,
    ``sigma_Mars == sigma_Sun`` exactly, so the umbra tapers to a point.
    A sail placed ON the anti-Sun axis just inside that distance is deep
    inside the umbra; just beyond, the Sun angular disc exceeds the Mars
    disc and the sail is in penumbra (or fully sunlit at larger
    separations from the axis). Diagnostic helper -- the propagator
    does not use this value directly.

    For Mars at 1.524 AU, ``L ~ 1.12e6 km`` -- far beyond any LMO
    altitude, so shadow crossings in low Mars orbit are always deep
    inside the full-occultation regime and the binary on/off model is
    faithful away from the ~few-second terminator transits.
    """
    s_sun_from_mars = _sun_position_j2000_km(et, observer_naif_id)
    d_MS = float(np.linalg.norm(s_sun_from_mars))
    R_M = (
        mars_equatorial_radius_km()
        if central_radius_km is None
        else float(central_radius_km)
    )
    R_S = sun_radius_km()
    return R_M * d_MS / (R_S - R_M)
