"""Pins and live cross-checks for ``reflectors.earth_constants``.

Mirrors ``tests/test_mars_constants.py``: literature-not-in-kernel values are
asserted at their pinned value (with the source documented in the module), and
the values SPICE provides live (Earth GM, radius, spin rate) are cross-checked
against the literature so a kernel-pool drift fails the tests.
"""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from reflectors.dynamics import body_gm_km3_per_s2
from reflectors.earth_constants import (
    EARTH_ATMOSPHERE_ROTATION_RATE_RAD_S,
    EARTH_HELIOCENTRIC_SEMIMAJOR_AXIS_KM,
    EARTH_HILL_RADIUS_KM,
    EARTH_J2,
    MOON_MEAN_ORBIT_RADIUS_KM,
)
from reflectors.ephemeris import utc_to_et
from reflectors.surface import body_equatorial_radius_km, earth_equatorial_radius_km

EPOCH = "2028-01-01T00:00:00"


def test_earth_j2_pinned_value():
    # EGM2008 tide-free J2 (positive: oblate). Pure literature pin (J2 is not in
    # the PCK). The 6-figure value is unambiguous across tide conventions.
    assert EARTH_J2 == 1.0826267e-3
    assert EARTH_J2 > 0.0  # oblate (equatorial bulge)


def test_earth_gm_live_matches_literature():
    # Live BODY399_GM is the DE440 ephemeris value 398600.435507 km^3/s^2
    # (shipped by gm_de440.tpc). A tight pin makes a kernel-pool change an
    # explicit test failure; this is the value every Earth propagation uses.
    mu = body_gm_km3_per_s2(399)
    assert mu == np.float64(mu)  # resolves (no exception)
    assert abs(mu - 398600.435507) < 1.0e-5  # DE440 kernel value, ~2.5e-11 rel
    # Cross-convention sanity bound (NOT the in-use number): the IERS/WGS84
    # conventional GM is 398600.4418 km^3/s^2, which differs from the DE440
    # value at ~1.6e-8 relative (~6.3e-3 km^3/s^2).
    assert abs(mu - 398600.4418) < 1.0e-2


def test_earth_equatorial_radius_live_matches_wgs84():
    a = earth_equatorial_radius_km()
    assert abs(a - 6378.137) < 1.0e-2
    # the generic-by-NAIF reader agrees with the Earth convenience wrapper
    assert earth_equatorial_radius_km() == body_equatorial_radius_km(399)


def test_earth_hill_radius_recomputed_from_kernel_pool():
    # r_Hill = a_Earth * (mu_E / (3 mu_Sun))^(1/3), with mu from the live pool
    # and a_Earth the pinned mean semi-major axis. Verifies the pinned constant
    # is the correct function of the kernel GMs (Murray & Dermott Eq. 3.82).
    mu_e = body_gm_km3_per_s2(399)
    mu_s = body_gm_km3_per_s2(10)
    hill = EARTH_HELIOCENTRIC_SEMIMAJOR_AXIS_KM * (mu_e / (3.0 * mu_s)) ** (1.0 / 3.0)
    assert hill == np.float64(hill)
    assert abs(hill - EARTH_HILL_RADIUS_KM) / EARTH_HILL_RADIUS_KM < 1.0e-3


def test_moon_is_inside_the_earth_hill_sphere():
    # The defining difference from Mars: the Moon orbits well inside the escape
    # ceiling. Pinned mean distance + a live cross-check at the epoch.
    assert MOON_MEAN_ORBIT_RADIUS_KM == 384400.0
    assert MOON_MEAN_ORBIT_RADIUS_KM < 0.30 * EARTH_HILL_RADIUS_KM
    et = utc_to_et(EPOCH)
    moon_state, _ = spice.spkezr("301", et, "J2000", "NONE", "399")
    d_moon = float(np.linalg.norm(moon_state[:3]))
    assert d_moon < EARTH_HILL_RADIUS_KM  # Moon inside Hill at the epoch


def test_earth_atmosphere_rotation_rate_matches_sxform():
    # Recover the spin rate from the loaded PCK via the IAU_EARTH->J2000 state
    # transformation (same construction surface.mars_rotation_rate_rad_per_s uses
    # for Mars) and pin the literature constant against it.
    et = utc_to_et(EPOCH)
    T6 = np.asarray(spice.sxform("IAU_EARTH", "J2000", et), dtype=float)
    R = T6[:3, :3]
    Rdot = T6[3:, :3]
    skew = R.T @ Rdot
    omega = float(np.linalg.norm(np.array([skew[2, 1], skew[0, 2], skew[1, 0]])))
    assert abs(omega - EARTH_ATMOSPHERE_ROTATION_RATE_RAD_S) < 1.0e-9
