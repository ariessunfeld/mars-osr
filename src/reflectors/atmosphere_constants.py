"""Single-source atmospheric-density constants for the Earth-escape drag model.

Primary reference: Montenbruck, O. & Gill, E. (2000),
*Satellite Orbits: Models, Methods, and Applications*, Springer, §3.5
"Atmospheric Drag". Every value below is transcribed from that text with its
equation, table, or page number so the provenance remains auditable.

The dominant uncertainty in LEO drag is the density model itself: M&G p.91 note
the Harris-Priester table differs from Jacchia-1971 by ~40% at mean solar
activity (up to 60% at maximum), and that "the accuracy of empirical drag models
has not significantly improved during the past two decades" (p.86). The
constants here are therefore the leading drag-error source -- documented, not
hidden.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Units conversion: Harris-Priester Table 3.8 is tabulated in g/km^3.
# ---------------------------------------------------------------------------
#   1 g / km^3 = (1e-3 kg) / (1e9 m^3) = 1e-12 kg/m^3.
# This is the single most error-prone step in the whole drag model; it is pinned
# explicitly and asserted in tests/test_atmosphere.py.
G_PER_KM3_TO_KG_PER_M3: float = 1.0e-12


# ---------------------------------------------------------------------------
# Drag coefficient C_D
# ---------------------------------------------------------------------------
#
# Free-molecular-flow drag coefficient. M&G p.84-85: in the free-molecular
# regime (Knudsen number K = lambda/l >= 10, which holds above ~150 km, their
# Fig. 3.9), C_D ~ 2.0 for a spherical body and 2.0-2.3 for non-spherical convex
# spacecraft. 2.2 is the standard satellite value. This is the second-largest
# drag uncertainty after the density; it is a named, citable default that the
# drag API exposes as a parameter.
DEFAULT_DRAG_COEFFICIENT: float = 2.2


# ---------------------------------------------------------------------------
# Harris-Priester diurnal-bulge parameters (M&G Eqs. 3.103-3.105, p.90)
# ---------------------------------------------------------------------------
#
# The day-night density variation is a cos^n(Psi/2) interpolation between the
# tabulated antapex (minimum, rho_m) and apex (maximum, rho_M) densities, where
# Psi is the angle from the satellite to the diurnal bulge apex. The apex lags
# the sub-solar point by ~30 deg in longitude (the atmosphere's thermal lag;
# max density ~2 h after local noon, M&G p.87-90).
HARRIS_PRIESTER_BULGE_LAG_DEG: float = 30.0  # lambda_l, M&G Eq. 3.105
# Exponent n in cos^n(Psi/2): 2 for low-inclination orbits, 6 for polar
# (M&G p.90). The reference Earth escape is polar, so 6 is the default;
# exposed as a parameter.
HARRIS_PRIESTER_BULGE_EXPONENT_LOW_INCLINATION: float = 2.0
HARRIS_PRIESTER_BULGE_EXPONENT_POLAR: float = 6.0


# ---------------------------------------------------------------------------
# Harris-Priester density table (M&G Table 3.8, p.91; Long et al. 1989)
# ---------------------------------------------------------------------------
#
# Mean-solar-activity minimum (antapex) and maximum (apex) atmospheric density
# vs geodetic height, 100-1000 km. Units: g/km^3 (convert with the factor
# above). Rows are (h_km, rho_min_g_km3, rho_max_g_km3), exactly as printed.
#
# Interpolation between rows is exponential (M&G Eq. 3.101-3.102); see
# reflectors.atmosphere.HarrisPriester. The table is for MEAN solar activity;
# minimum-activity phases want lower coefficients (M&G p.90 caveat).
HARRIS_PRIESTER_TABLE_G_PER_KM3: tuple[tuple[float, float, float], ...] = (
    (100.0, 497400.0, 497400.0),
    (120.0, 24900.0, 24900.0),
    (130.0, 8377.0, 8710.0),
    (140.0, 3899.0, 4059.0),
    (150.0, 2122.0, 2215.0),
    (160.0, 1263.0, 1344.0),
    (170.0, 800.8, 875.8),
    (180.0, 528.3, 601.0),
    (190.0, 361.7, 429.7),
    (200.0, 255.7, 316.2),
    (210.0, 183.9, 239.6),
    (220.0, 134.1, 185.3),
    (230.0, 99.49, 145.5),
    (240.0, 74.88, 115.7),
    (250.0, 57.09, 93.08),
    (260.0, 44.03, 75.55),
    (270.0, 34.30, 61.82),
    (280.0, 26.97, 50.95),
    (290.0, 21.39, 42.26),
    (300.0, 17.08, 35.26),
    (320.0, 10.99, 25.11),
    (340.0, 7.214, 18.19),
    (360.0, 4.824, 13.37),
    (380.0, 3.274, 9.955),
    (400.0, 2.249, 7.492),
    (420.0, 1.558, 5.684),
    (440.0, 1.091, 4.355),
    (460.0, 0.7701, 3.362),
    (480.0, 0.5474, 2.612),
    (500.0, 0.3916, 2.042),
    (520.0, 0.2819, 1.605),
    (540.0, 0.2042, 1.267),
    (560.0, 0.1488, 1.005),
    (580.0, 0.1092, 0.7997),
    (600.0, 0.08070, 0.6390),
    (620.0, 0.06012, 0.5123),
    (640.0, 0.04519, 0.4121),
    (660.0, 0.03430, 0.3325),
    (680.0, 0.02632, 0.2691),
    (700.0, 0.02043, 0.2185),
    (720.0, 0.01607, 0.1779),
    (740.0, 0.01281, 0.1452),
    (760.0, 0.01036, 0.1190),
    (780.0, 0.008496, 0.09776),
    (800.0, 0.007069, 0.08059),
    (840.0, 0.004680, 0.05741),
    (880.0, 0.003200, 0.04210),
    (920.0, 0.002210, 0.03130),
    (960.0, 0.001560, 0.02360),
    (1000.0, 0.001150, 0.01810),
)

# Top tabulated altitude (M&G Table 3.8 ceiling).
HARRIS_PRIESTER_TOP_ALTITUDE_KM: float = 1000.0

# ---------------------------------------------------------------------------
# >1000 km exospheric tail scale height (NRLMSIS 2.1 calibrated)
# ---------------------------------------------------------------------------
#
# The escape spiral spends most of its life above the 1000 km table ceiling
# (out to the Hill sphere ~1.496e6 km), so the tail is required. Naively
# Extrapolating the H-P top-interval [960,1000] km scale heights
# (H ~ 131-151 km) above ~1500 km underestimates density because the exosphere
# becomes dominated by light species (He/H) with larger scale heights.
#
# Instead, above HARRIS_PRIESTER_TOP_ALTITUDE_KM the model continues each density
# column from its 1000 km value with this single exospheric scale height, fit to
# NRLMSIS 2.1 (pymsis version 2.1, mean solar activity F10.7=F10.7a=150, Ap=4)
# over 1000-2500 km: NRLMSIS total density drops by ~exp(-1500/H) there, giving
# H ~ 387 km (noon) / 405 km (midnight); 400 km is the pinned representative
# value. With it, H-P tracks NRLMSIS to within the ~constant factor-4 model
# offset (the irreducible empirical-model disagreement, M&G p.91) all the way to
# 2500 km, unlike a continuation of the top-interval scale height. The comparison
# against pymsis is pinned in tests/test_atmosphere.py. (Drag above ~1500 km is
# <1% of SRP regardless, so this tail is a small calibrated correction.)
EXOSPHERIC_SCALE_HEIGHT_KM: float = 400.0
