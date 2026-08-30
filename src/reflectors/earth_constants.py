"""Pinned Earth constants that are NOT carried in the SPICE kernel pool.

Single source of truth for Earth-related numerical values cited from the
literature. Mirrors ``mars_constants.py`` in discipline:

  - Values SPICE itself provides are read from the live kernel pool, never pinned here:
      * Earth GM           -> ``dynamics.body_gm_km3_per_s2(399)`` (BODY399_GM)
      * Earth radii        -> ``surface.earth_equatorial_radius_km()`` (BODY399_RADII)
      * Earth spin / pole  -> ``spice.sxform("IAU_EARTH", ...)``
    The ``tests/test_earth_constants.py`` suite cross-checks those live values
    against the literature so a kernel-pool drift fails the tests.
  - Only literature-not-in-kernel values (J2, the heliocentric semi-major axis,
    the derived Hill radius, the lunar distance, the atmospheric rotation rate)
    are pinned below, each with its source.

Tests confirm Earth GM, Moon SPK coverage, and the IAU_EARTH frame at the
escape epoch.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Earth oblateness coefficient J2
# ---------------------------------------------------------------------------
#
# The dynamic form factor J2 = -C_{2,0} (unnormalized), the dominant zonal
# harmonic of Earth's gravity field, driving the secular nodal regression and
# apsidal precession that matter at the low-altitude start of the escape spiral
# (the J2 perturbation scales as (R_eq/r)^2, so it fades as the sail climbs).
#
# Value: J2 = 1.0826267e-3.
#
# Primary source: EGM2008 (Pavlis, Holmes, Kenyon & Factor 2012, "The
# development and evaluation of the Earth Gravitational Model 2008 (EGM2008)",
# J. Geophys. Res. 117, B04406), tide-free convention; reproduced in
# Montenbruck & Gill (2000), *Satellite Orbits*, Earth gravity-field table,
# and Vallado (2013), App. D.
#
# Convention note: WGS84's defining value (1.08262999e-3) and the zero-tide
# value (1.0826359e-3) differ from the tide-free value at the ~1e-4 relative
# level. That spread is far below any threshold that matters for an escape
# spiral whose J2 contribution is itself a small, fast-fading perturbation; the
# tide-free EGM2008 value is pinned for definiteness.
#
# This is a pure literature pin (J2 is NOT in pck00011, which carries only
# radii + pole orientation). ``tests/test_earth_constants.py`` asserts it and
# documents the source; it cannot be cross-checked against the kernel pool.
EARTH_J2: float = 1.0826267e-3


# ---------------------------------------------------------------------------
# Earth heliocentric orbit semi-major axis
# ---------------------------------------------------------------------------
#
# Earth's mean orbital semi-major axis about the Sun, used to size the Hill
# sphere (below) and to place the escape spiral's initial circular orbit in
# Earth's heliocentric plane.
#
# Value: a_Earth = 1.000001018 AU = 1.495979e8 km
#        (1 AU = 1.495978707e8 km, IAU 2012; the single source for the AU is
#         ``reflectors.solar_constants.AU_KM``).
#
# Primary source: JPL Solar System Dynamics / NASA Earth Fact Sheet -- Earth
# orbital semi-major axis (mean element, J2000). Mirrors how
# ``mars_constants.MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM`` is pinned. The DE440
# osculating value swings ~+-1.7% over Earth's e=0.0167 orbit; this is the mean.
EARTH_HELIOCENTRIC_SEMIMAJOR_AXIS_KM: float = 1.495979e8


# ---------------------------------------------------------------------------
# Earth Hill sphere radius
# ---------------------------------------------------------------------------
#
# Classical Hill (Roche) radius for a body on a circular heliocentric orbit:
#
#     r_Hill = a_Earth * ( mu_Earth / (3 * mu_Sun) ) ** (1/3)
#
# (Murray & Dermott 1999, *Solar System Dynamics*, Eq. 3.82, eccentricity
# factor omitted -- same convention as MARS_HILL_RADIUS_KM.)
#
# Inputs:
#     a_Earth = 1.495979e8 km       (EARTH_HELIOCENTRIC_SEMIMAJOR_AXIS_KM)
#     mu_Earth = 3.98600e5 km^3/s^2 (BODY399_GM, gm_de440.tpc)
#     mu_Sun  = 1.32712e11 km^3/s^2 (BODY10_GM,  gm_de440.tpc)
# -> r_Hill ~ 1.4966e6 km.
#
# Used as the OUTER termination radius for the SRP escape spiral (the handoff
# boundary to a heliocentric solver). Unlike Mars, the Moon orbits
# at ~384,400 km = ~0.26 r_Hill, DEEP inside this sphere, so the outbound
# spiral threads cislunar space and the lunar third body is first-order. The
# Hill-exit criterion is therefore paired with a forward-validation propagation
# (no recapture) for the Earth-escape workflow, not taken as sufficient alone.
#
# Verified live in ``tests/test_earth_constants.py`` against a kernel-pool
# recomputation (mu_Earth + mu_Sun from the loaded GM kernels).
EARTH_HILL_RADIUS_KM: float = 1.4966e6


# ---------------------------------------------------------------------------
# Mean Earth-Moon distance
# ---------------------------------------------------------------------------
#
# Mean lunar orbital semi-major axis, used only to document the Moon's position
# relative to the Hill sphere (the cislunar-threading caveat above); the actual
# lunar position in the dynamics is always the live DE440 ephemeris.
#
# Value: 384,400 km. Source: standard lunar mean distance (e.g. NASA Moon Fact
# Sheet; Murray & Dermott 1999). The instantaneous DE440 distance ranges
# ~363,000-406,000 km over a lunar month.
MOON_MEAN_ORBIT_RADIUS_KM: float = 384400.0


# ---------------------------------------------------------------------------
# Earth atmosphere rotation rate (for drag co-rotation)
# ---------------------------------------------------------------------------
#
# Earth's sidereal angular rotation rate, used to form the co-rotating
# atmosphere velocity v_atm = omega x r in the drag model (Montenbruck & Gill
# 2000, Eq. 3.98, p.85; the atmosphere is assumed to co-rotate rigidly with the
# solid Earth to within <5% in drag force, King-Hele 1987).
#
# Value: 7.292115e-5 rad/s.
#
# Primary source: IERS Conventions (2010) nominal mean angular velocity of the
# Earth; reproduced in Montenbruck & Gill (2000) Eq. 3.98 (omega = 7.292e-5).
#
# Cross-check: ``tests/test_earth_constants.py`` recovers the spin rate from the
# loaded PCK via ``spice.sxform("IAU_EARTH","J2000",et)`` (the same construction
# ``surface.mars_rotation_rate_rad_per_s`` uses for Mars) and pins this value
# against it. The drag
# model itself uses the full sxform-derived angular-velocity VECTOR (true pole
# direction), not this scalar; the scalar is pinned for documentation + the
# kernel cross-check.
EARTH_ATMOSPHERE_ROTATION_RATE_RAD_S: float = 7.292115e-5


# ---------------------------------------------------------------------------
# Earth sidereal year (for the sun-synchronous nodal-precession target)
# ---------------------------------------------------------------------------
#
# Period over which a sun-synchronous orbit's node must precess one full turn
# (2 pi) to track the mean Sun -- the Earth analogue of MARS_SIDEREAL_YEAR_S.
# Used by sun_sync.sun_sync_inclination_rad(..., target_period_s=...) to get the
# ~98.6 deg Earth-LEO sun-sync inclination. (Sidereal vs tropical year differ by
# ~0.004%, shifting the inclination by <0.001 deg -- negligible; sidereal is used
# to match the Mars convention.)
#
# Value: 365.256363 d x 86400 s/d = 3.15581498e7 s.
# Primary source: IAU sidereal year (e.g. Allen's Astrophysical Quantities, 4th
# ed., Cox 2000, Table 1.1; Seidelmann 1992, Explanatory Supplement).
EARTH_SIDEREAL_YEAR_S: float = 365.256363 * 86400.0
