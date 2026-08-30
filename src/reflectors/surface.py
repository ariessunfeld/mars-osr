"""Mars surface-point tracking in the IAU_MARS body-fixed frame.

The IAU_MARS frame co-rotates with Mars: its z-axis lies along the IAU-defined
Mars spin pole, its x-axis passes through the Airy-0-anchored prime meridian.
Points on the Mars surface therefore have constant coordinates in IAU_MARS and
time-varying coordinates in any inertial frame (e.g. J2000).

This module exposes:
    - Mars triaxial radii and flattening from the loaded text PCK
      (pck00011.tpc as of 2026-04-20).
    - Conversion of (lat, lon, alt) to a body-fixed position vector.
    - Transformation of that point to an inertial 6-vector via SPICE's
      state-transformation matrix.
    - A helper for Mars's instantaneous rotation rate.

All callers are expected to have loaded the kernel set (see
``reflectors.kernels.load_kernels``).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import spiceypy as spice

from reflectors.ephemeris import EpochLike, frame_rotation, utc_to_et


logger = logging.getLogger(__name__)

BODY_FIXED_FRAME = "IAU_MARS"
MARS_NAIF_ID = 499


@lru_cache(maxsize=1)
def mars_radii() -> tuple[float, float, float]:
    """Triaxial Mars radii ``(a, b, c)`` in km from the loaded PCK.

    For Mars, ``a == b`` (the IAU model treats Mars as a biaxial spheroid).
    Cached because the values are fixed constants of the kernel pool.
    """
    n, radii = spice.bodvrd("MARS", "RADII", 3)
    assert n == 3
    a, b, c = float(radii[0]), float(radii[1]), float(radii[2])
    logger.debug("Mars radii from PCK: a=%.6f b=%.6f c=%.6f km", a, b, c)
    return a, b, c


def mars_equatorial_radius_km() -> float:
    a, _b, _c = mars_radii()
    return a


def mars_polar_radius_km() -> float:
    _a, _b, c = mars_radii()
    return c


def mars_flattening() -> float:
    """Oblateness ``f = (a - c) / a`` of the Mars reference spheroid."""
    a, _b, c = mars_radii()
    return (a - c) / a


@lru_cache(maxsize=None)
def body_radii(naif_id: int) -> tuple[float, float, float]:
    """Triaxial radii ``(a, b, c)`` in km for ANY body, from the loaded PCK.

    Generic-by-NAIF-id companion to the Mars-specific :func:`mars_radii`.
    Cached because the values are fixed kernel-pool constants. The integer id
    is resolved to its SPICE name via
    ``spice.bodc2n`` so ``spice.bodvrd`` can fetch ``BODY<id>_RADII`` (e.g.
    399 -> "EARTH" -> ``BODY399_RADII`` = 6378.137 / 6356.752 km in
    pck00011.tpc).
    """
    name = spice.bodc2n(naif_id)
    n, radii = spice.bodvrd(name, "RADII", 3)
    assert n == 3
    a, b, c = float(radii[0]), float(radii[1]), float(radii[2])
    logger.debug("body %d (%s) radii: a=%.6f b=%.6f c=%.6f km",
                 naif_id, name, a, b, c)
    return a, b, c


def body_equatorial_radius_km(naif_id: int) -> float:
    """Equatorial radius (km) for any body, from the loaded PCK."""
    a, _b, _c = body_radii(naif_id)
    return a


def earth_equatorial_radius_km() -> float:
    """Earth equatorial radius (km) from the PCK (WGS84-consistent, 6378.137)."""
    return body_equatorial_radius_km(399)


def surface_point_body_fixed(
    lat_deg: float,
    lon_deg: float,
    alt_km: float = 0.0,
    *,
    planetographic: bool = True,
) -> np.ndarray:
    """Position of a surface point in IAU_MARS (km), shape (3,).

    ``planetographic=True`` (default) interprets ``lat_deg`` as geodetic
    latitude on the Mars reference spheroid (the convention used by MOLA,
    HiRISE, and mission planning). ``planetographic=False`` treats
    ``lat_deg`` as planetocentric (angle between the position vector and
    the equatorial plane); ``alt_km`` in that mode is radial distance above
    the sphere of equatorial radius ``a``.

    Longitude is EAST-positive in both cases, matching the IAU_MARS
    body-fixed frame definition (x-axis through the IAU prime meridian,
    rotation direction via the W(t) polynomial). Note: SPICE's ``pgrrec``
    defaults to WEST-positive planetographic longitude for Mars because of
    the older cartographic convention. The spheroid transformation is
    computed directly rather than through ``pgrrec`` so that
    ``lon_deg`` truly means "degrees east".
    """
    a, _b, c = mars_radii()
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)

    if planetographic:
        # Biaxial oblate spheroid (a == b, polar radius c). Standard geodetic
        # -> rectangular transformation with east-positive longitude.
        e2 = 1.0 - (c / a) ** 2
        N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
        x = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1.0 - e2) + alt_km) * np.sin(lat_rad)
        return np.array([x, y, z], dtype=float)

    # Planetocentric: spherical coordinates, east-positive longitude.
    r_from_center = a + alt_km
    r = spice.latrec(r_from_center, lon_rad, lat_rad)
    return np.asarray(r, dtype=float)


def surface_point_position(
    lat_deg: float,
    lon_deg: float,
    epoch: EpochLike,
    alt_km: float = 0.0,
    *,
    frame: str = "J2000",
    planetographic: bool = True,
) -> np.ndarray:
    """Inertial position (km) of the named surface point at ``epoch``.

    Returns a shape-(3,) ndarray. Uses ``spice.pxform`` (rotation only -- no
    Mars-centre translation), so the result is the point's position relative
    to Mars's centre, expressed in the target inertial frame.
    """
    r_bf = surface_point_body_fixed(lat_deg, lon_deg, alt_km, planetographic=planetographic)
    et = utc_to_et(epoch)
    M = frame_rotation(BODY_FIXED_FRAME, frame, et)
    return M @ r_bf


def surface_point_state(
    lat_deg: float,
    lon_deg: float,
    epoch: EpochLike,
    alt_km: float = 0.0,
    *,
    frame: str = "J2000",
    planetographic: bool = True,
) -> np.ndarray:
    """Inertial state (6-vec, km and km/s) of the surface point at ``epoch``.

    The velocity comes entirely from the body-fixed-to-inertial rotation
    (omega x r), since the point is stationary in IAU_MARS. The position is
    relative to Mars centre; to get a heliocentric or SSB-centered state, add
    the Mars state at the same epoch.
    """
    r_bf = surface_point_body_fixed(lat_deg, lon_deg, alt_km, planetographic=planetographic)
    bf_state = np.concatenate([r_bf, np.zeros(3)])
    et = utc_to_et(epoch)
    T6 = spice.sxform(BODY_FIXED_FRAME, frame, et)
    return np.asarray(T6, dtype=float) @ bf_state


def mars_rotation_rate_rad_per_s(epoch: EpochLike) -> float:
    """Instantaneous Mars rotation rate (rad/s) from the loaded PCK.

    Computed as the magnitude of the angular-velocity vector implied by the
    IAU_MARS -> J2000 state transformation. Specifically, if T6 is the 6x6
    sxform matrix, the lower-left 3x3 block equals dR/dt, and omega_hat is
    recovered from R^T (dR/dt). Taking the norm gives the scalar spin rate.
    This is invariant to frame choice for the target.
    """
    et = utc_to_et(epoch)
    T6 = np.asarray(spice.sxform(BODY_FIXED_FRAME, "J2000", et), dtype=float)
    R = T6[:3, :3]
    Rdot = T6[3:, :3]
    skew = R.T @ Rdot  # omega represented in the body-fixed frame (skew-symmetric)
    omega_vec = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return float(np.linalg.norm(omega_vec))
