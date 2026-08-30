"""Pinned Mars constants that are NOT carried in the SPICE kernel pool.

Values that SPICE itself provides (mu_Mars, Mars radii, Mars rotation rate,
Mars pole direction) are always read live from the kernel pool via the
surface / dynamics modules -- do NOT add them here. This module is the single
source of truth for Mars-related numerical values cited from the
literature but that would otherwise be duplicated in multiple tests and
scripts.

Each value is documented with its source and cross-checked against a live
ephemeris in ``tests/test_mars_constants.py`` so literature drift
(or kernel-pool drift) gets caught by the test suite rather than baked in.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Mars sidereal orbital period
# ---------------------------------------------------------------------------
#
# The time Mars takes to return to the same direction in a heliocentric
# inertial frame -- the "year" as used in the sun-synchronous RAAN-drift
# target dRAAN/dt = 2*pi / MARS_SIDEREAL_YEAR_S.
#
# Value: 686.9710 Earth days.
#
# Primary source: Allison, M. & McEwen, M. (2000), "A post-Pathfinder
# evaluation of areocentric solar coordinates with improved timing recipes
# for Mars seasonal/diurnal climate studies", Planet. Space Sci. 48:215-235,
# which gives Mars's sidereal year as 686.9710 Earth days (their Table 2,
# anchored to a 1980-2000 interval and consistent with modern JPL planetary
# ephemerides at the 1e-4 day level).
#
# Cross-checks:
#   - JPL Mars Fact Sheet quotes 686.98 days (rounded).
#   - Osculating Keplerian period from DE440 at J2000 is within ~0.2 days
#     of this value (the spread reflects osculating-vs-mean over an
#     eccentric orbit).
#
# The pinned value is verified against DE440 in
# ``tests/test_mars_constants.py::test_mars_sidereal_year_matches_de440``
# to within a generous 0.5 day tolerance.
MARS_SIDEREAL_YEAR_DAYS: float = 686.9710
MARS_SIDEREAL_YEAR_S: float = MARS_SIDEREAL_YEAR_DAYS * 86400.0


# ---------------------------------------------------------------------------
# Mars solar day ("sol")
# ---------------------------------------------------------------------------
#
# The "sol": Mars's mean solar day, i.e. the time between successive solar
# meridian transits at a Mars-fixed surface point. Distinct from the
# sidereal day (one full Mars rotation in inertial frame, ~88642.66 s).
# The synodic relationship is
#
#     1 / T_solar = 1 / T_sidereal_day  -  1 / T_sidereal_year
#
# (prograde Mars rotation, prograde Mars heliocentric motion).
#
# Equivalently, for a sun-synchronous orbit at Mars: the orbit-plane RAAN
# precesses at exactly 2 pi / T_sidereal_year by construction, so the
# orbit-plane rotation rate relative to the Mars-fixed body frame equals
# (omega_Mars_inertial - 2 pi / T_sidereal_year) = 2 pi / T_solar. THE
# GROUND-TRACK SYNODIC PERIOD OF A SUN-SYNC ORBIT EQUALS THE MARS SOLAR
# DAY EXACTLY. This is the constant used by the repeating-ground-
# track helper ``sun_sync.repeat_ground_track_altitude``.
#
# Value: 88775.244 s (24 h 39 min 35.244 s).
#
# Primary source: Allison, M. & McEwen, M. (2000), "A post-Pathfinder
# evaluation of areocentric solar coordinates ...", Planet. Space Sci.
# 48:215-235, Table 2: Mars mean solar day = 88775.244147 s. Rounded to
# the value below; sub-second digits are below the ms-scale variability
# from atmospheric / tidal effects outside this model.
#
# Cross-check: the synodic identity
#     T_solar = 1 / (1 / T_sd - 1 / T_year_sidereal)
# is verified live in
# ``tests/test_mars_constants.py::test_mars_solar_day_matches_synodic_identity``
# using the Mars rotation rate from the loaded PCK kernel and
# ``MARS_SIDEREAL_YEAR_S``. Tolerance is generous (1.0 s) so routine
# kernel refinements do not flag the test, but tight enough to surface
# a real drift.
SECONDS_PER_SOLAR_SOL_S: float = 88775.244


# ---------------------------------------------------------------------------
# Konopliv 2020 / MRO120F gravitational-parameter quartet
# ---------------------------------------------------------------------------
#
# Self-consistent decomposition of the Mars-system GM as published in the
# PDS label that ships alongside the MRO120F SHADR table:
#
#   data product:  jgmro_120f_sha (PDS Geosciences, mrors_1xxx delivery)
#   label file:    jgmro_120f_sha.lbl, lines 41-45
#   URL (label):   https://pds-geosciences.wustl.edu/mro/mro-m-rss-5-sdp-v1/
#                  mrors_1xxx/data/shadr/jgmro_120f_sha.lbl
#
#       The Mars System GM (Mars+Phobos+Deimos) =
#                            42828.3756640  +/- 0.0002      km^3/s^2
#           The GM of Mars   = 42828.3748574                 km^3/s^2
#           The GM of Phobos = (7.10 +/- 0.05) x 10^-4       km^3/s^2
#           The GM of Deimos = (9.68 +/- 1.30) x 10^-5       km^3/s^2
#
# Primary citation:
#   Konopliv, A. S., Park, R. S., Rivoldini, A., Baland, R.-M.,
#   Le Maistre, S., Van Hoolst, T., Yseboodt, M., and Dehant, V. (2020).
#   "Detection of the Chandler wobble of Mars from orbiting spacecraft."
#   Geophys. Res. Lett. 47, e2020GL090568. doi:10.1029/2020GL090568.
#
# Selection relative to BODY499/401/402_GM from gm_de440.tpc:
#
#   The MRO120F harmonic potential is U = (GM_system / r) * [1 + Sigma h_nm]
#   with the leading factor GM_system equal to the lumped (Mars + Phobos +
#   Deimos) point-mass GM. When Phobos and Deimos are also modelled as
#   separate Eq. 3.37 third-body perturbers in the same propagation, their
#   masses are double-counted: once at Mars centre via GM_system, and again
#   at their actual orbital positions via the third-body direct term. The
#   third-body indirect term subtracts the moon's pull on Mars at its real
#   position, but does NOT cancel the lumped-at-Mars-centre piece.
#
#   The decoupled calculation uses the Konopliv 2020 self-consistent quartet:
#     - subtract PHOBOS_GM_KONOPLIV_2020 + DEIMOS_GM_KONOPLIV_2020 from the
#       central mu when those moons are separate third bodies, leaving
#       MARS_PLANET_GM_KONOPLIV_2020 as the central two-body GM;
#     - feed PHOBOS_GM_KONOPLIV_2020 / DEIMOS_GM_KONOPLIV_2020 into the
#       third-body Eq. 3.37 evaluations.
#
#   Pairing Konopliv-fit moon GMs with MRO120F harmonics is more
#   self-consistent than mixing in DE440 BODY40[12]_GM, since the moon GMs
#   and the harmonic field were jointly estimated from the same MRO /
#   Odyssey / MGS tracking data set (Konopliv 2020 Section 3, "Mars and
#   Phobos and Deimos GM ... estimated together"). The Konopliv values
#   differ from DE440's by ~0.2% (Phobos), ~0.5% (Deimos), within the
#   Konopliv-2020 1-sigma error bars.
#
# Identity asserted in tests/test_mars_constants.py:
#     MARS_PLANET_GM_KONOPLIV_2020_KM3_S2
#   + PHOBOS_GM_KONOPLIV_2020_KM3_S2
#   + DEIMOS_GM_KONOPLIV_2020_KM3_S2
#   == mars_gravity_model().mu_km3_s2  (to 1e-6 km^3/s^2)
#
MARS_PLANET_GM_KONOPLIV_2020_KM3_S2: float = 42828.3748574
PHOBOS_GM_KONOPLIV_2020_KM3_S2: float = 7.10e-4
DEIMOS_GM_KONOPLIV_2020_KM3_S2: float = 9.68e-5


# ---------------------------------------------------------------------------
# Mars heliocentric orbit semi-major axis
# ---------------------------------------------------------------------------
#
# Mars's mean orbital semi-major axis about the Sun. Needed to (a) place the
# escape spiral's initial circular orbit in Mars's heliocentric orbital plane
# and (b) size the Mars Hill sphere (below).
#
# Value: a_Mars = 1.52371034 AU = 2.279439e8 km
#        (1 AU = 1.495978707e8 km, IAU 2012 definition;
#         1.52371034 * 1.495978707e8 = 2.2794387e8 km).
#
# Primary source: JPL Solar System Dynamics / NASA Mars Fact Sheet -- Mars
# orbital semi-major axis (mean element, J2000 epoch). The km value is pinned
# directly here so this module stays a dependency-free leaf; the AU figure is
# recorded as provenance, not as a competing definition of the AU (the single
# source for the AU is ``reflectors.solar_constants.AU_KM``).
#
# Cross-check: the DE440 osculating Keplerian semi-major axis of MARS
# BARYCENTER about the SUN at J2000 agrees within the osculating-vs-mean
# spread over Mars's e = 0.0934 orbit; verified live in
# ``tests/test_mars_constants.py``.
MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM: float = 2.279439e8


# ---------------------------------------------------------------------------
# Mars Hill sphere radius
# ---------------------------------------------------------------------------
#
# The classical Hill (Roche) radius for a body on a circular heliocentric
# orbit:
#
#     r_Hill = a_Mars * ( mu_Mars / (3 * mu_Sun) ) ** (1/3)
#
# (Murray & Dermott 1999, *Solar System Dynamics*, Cambridge Univ. Press,
# Eq. 3.82, with the eccentricity factor (1 - e) omitted -- see note below.)
#
# Inputs:
#     a_Mars  = 2.279439e8  km            (MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM)
#     mu_Mars = 4.282837e4  km^3/s^2      (BODY499_GM, gm_de440.tpc)
#     mu_Sun  = 1.32712440e11 km^3/s^2    (BODY10_GM,  gm_de440.tpc)
# -> r_Hill ~ 1.0841e6 km.
#
# Used as the OUTER termination radius for the SRP escape spiral: once the
# sail crosses this radius, Mars's gravity is no longer dominant and the
# state is handed off to a separate interplanetary solver.
#
# Note on the eccentricity factor: the instantaneous Hill radius scales with
# the Mars-Sun distance, so it swings ~+/-9.3% (Mars e = 0.0934) over the
# Mars year. MARS_HILL_RADIUS_KM pins the SEMI-MAJOR-AXIS value (the orbit
# mean); the escape-termination boundary is representative, not epoch-exact.
#
# Verified live in ``tests/test_mars_constants.py`` against a kernel-pool
# recomputation (mu_Mars + mu_Sun from the loaded GM kernels).
MARS_HILL_RADIUS_KM: float = 1.0841e6
