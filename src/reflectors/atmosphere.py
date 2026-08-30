"""Earth upper-atmosphere density models for the SRP-sail drag force.

Two interchangeable density models behind a common duck-typed interface
``density_kg_m3(alt_km, r_hat_j2000, sun_hat_j2000) -> kg/m^3``:

  - ``HarrisPriester`` -- the default model. Exponential interpolation of the
    Montenbruck & Gill (2000) Table 3.8 coefficients (mean solar activity, Long
    et al. 1989), 100-1000 km, with the cos^n(Psi/2) diurnal density bulge
    (M&G Eqs. 3.101-3.105, p.90). Above 1000 km it extrapolates with the
    top-interval exospheric scale heights -- checked against an NRLMSIS 2.1
    oracle. The escape spiral spends most of its life above 1000 km, out to the
    Hill sphere ~1.496e6 km, so the tail is required; see
    ``tests/test_atmosphere.py``.

  - ``ExponentialAtmosphere`` -- a single-exponential model (M&G Eq. 3.99). Its
    purpose is to be the INDEPENDENT oracle for the King-Hele drag-decay
    cross-check (King-Hele's closed form assumes an exponential atmosphere), and
    a deliberately-simple second implementation cross-tested against H-P in
    their overlap region, following the same independent-implementations pattern
    as the zonal-vs-Cunningham gravity paths.

Both are SPICE-free in the hot loop: the caller (``reflectors.drag``) supplies
the precomputed satellite position direction and Sun direction so the diurnal
bulge (which needs the Sun) adds no SPICE call here.

Primary reference: Montenbruck & Gill (2000), *Satellite Orbits*.
Constants: ``reflectors.atmosphere_constants``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from reflectors.atmosphere_constants import (
    EXOSPHERIC_SCALE_HEIGHT_KM,
    G_PER_KM3_TO_KG_PER_M3,
    HARRIS_PRIESTER_BULGE_EXPONENT_POLAR,
    HARRIS_PRIESTER_BULGE_LAG_DEG,
    HARRIS_PRIESTER_TABLE_G_PER_KM3,
)


def _bulge_apex_direction(sun_hat_j2000: np.ndarray, lag_deg: float) -> np.ndarray:
    """Unit vector to the diurnal-bulge apex (M&G Eq. 3.105).

    Eq. 3.105 builds e_b from the Sun's right ascension + declination with a
    longitude lag ``lambda_l`` (~30 deg): the apex sits ``lag`` east of the
    sub-solar point at the same declination. That is exactly the Sun direction
    rotated by ``+lag`` about the celestial pole (the J2000 +z axis, since RA/dec
    are J2000-equatorial). Rotating the supplied Sun unit vector about +z by the
    lag reproduces Eq. 3.105 without re-deriving alpha_sun/delta_sun.
    """
    s = np.asarray(sun_hat_j2000, dtype=float)
    c, sn = math.cos(math.radians(lag_deg)), math.sin(math.radians(lag_deg))
    # R_z(+lag) applied to s.
    return np.array([c * s[0] - sn * s[1], sn * s[0] + c * s[1], s[2]])


@dataclass(frozen=True)
class HarrisPriester:
    """Harris-Priester density model (M&G §3.5.2, Table 3.8, Eqs. 3.101-3.105).

    Parameters
    ----------
    bulge_exponent
        ``n`` in ``cos^n(Psi/2)`` (M&G Eq. 3.103): 2 for low-inclination, 6 for
        polar orbits. The default is appropriate for the polar escape geometry.
    bulge_lag_deg
        Apex longitude lag east of the sub-solar point (M&G Eq. 3.105). Default
        30 deg.
    """

    bulge_exponent: float = HARRIS_PRIESTER_BULGE_EXPONENT_POLAR
    bulge_lag_deg: float = HARRIS_PRIESTER_BULGE_LAG_DEG
    label: str = "harris_priester"
    # Precomputed table columns in SI (kg/m^3). Built in __post_init__.
    _h_km: np.ndarray = field(default=None, repr=False, compare=False)
    _rho_min: np.ndarray = field(default=None, repr=False, compare=False)
    _rho_max: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        h = np.array([row[0] for row in HARRIS_PRIESTER_TABLE_G_PER_KM3], dtype=float)
        rmin = np.array(
            [row[1] for row in HARRIS_PRIESTER_TABLE_G_PER_KM3], dtype=float
        ) * G_PER_KM3_TO_KG_PER_M3
        rmax = np.array(
            [row[2] for row in HARRIS_PRIESTER_TABLE_G_PER_KM3], dtype=float
        ) * G_PER_KM3_TO_KG_PER_M3
        # frozen dataclass: set via object.__setattr__.
        object.__setattr__(self, "_h_km", h)
        object.__setattr__(self, "_rho_min", rmin)
        object.__setattr__(self, "_rho_max", rmax)

    def _interp_segment(self, alt_km: float, col: np.ndarray) -> float:
        """Exponential interpolation/extrapolation of one density column.

        In-table (100-1000 km): exponential interpolation between the bracketing
        rows (M&G Eq. 3.101-3.102), ``rho(h) = rho(h_i) exp((h_i - h)/H)`` with
        ``H = (h_i - h_{i+1}) / ln(rho(h_{i+1})/rho(h_i))``.

        Above 1000 km: continue from the 1000 km value with the NRLMSIS 2.1
        calibrated exospheric scale height ``EXOSPHERIC_SCALE_HEIGHT_KM`` (~400
        km) -- NOT the heavy-species top-interval scale height, which would fall
        10-100x too fast (the exosphere is light-species dominated). Verified
        against pymsis; see atmosphere_constants.

        Below 100 km: extrapolate the bottom interval (not encountered above the
        altitude floor, but keeps the model exception-free and monotonic).
        """
        h = self._h_km
        n = len(h)
        if alt_km >= h[-1]:  # exospheric tail (>1000 km), NRLMSIS-calibrated H
            return float(col[-1] * math.exp((h[-1] - alt_km) / EXOSPHERIC_SCALE_HEIGHT_KM))
        # Largest i with h[i] <= alt; clamp so [i, i+1] is always a valid interval.
        i = int(np.searchsorted(h, alt_km, side="right")) - 1
        i = max(0, min(i, n - 2))
        hi, hip1 = h[i], h[i + 1]
        ri, rip1 = col[i], col[i + 1]
        H = (hi - hip1) / math.log(rip1 / ri)  # >0: hi<hip1, rip1<ri
        return float(ri * math.exp((hi - alt_km) / H))

    def density_kg_m3(
        self,
        alt_km: float,
        r_hat_j2000: Optional[np.ndarray] = None,
        sun_hat_j2000: Optional[np.ndarray] = None,
    ) -> float:
        """Atmospheric density (kg/m^3) at geodetic altitude ``alt_km``.

        With ``r_hat_j2000`` and ``sun_hat_j2000`` supplied, the diurnal bulge
        (M&G Eq. 3.103-3.105) interpolates between the antapex (min) and apex
        (max) densities. Without them, returns the antapex (minimum) density --
        the conservative-for-lifetime lower bound; the drag wrapper always
        supplies both, so the bulge is active during propagation.
        """
        rho_min = self._interp_segment(alt_km, self._rho_min)
        rho_max = self._interp_segment(alt_km, self._rho_max)
        if r_hat_j2000 is None or sun_hat_j2000 is None:
            return rho_min
        e_b = _bulge_apex_direction(sun_hat_j2000, self.bulge_lag_deg)
        cos_psi = float(np.clip(np.dot(np.asarray(r_hat_j2000, dtype=float), e_b),
                                -1.0, 1.0))
        # cos^n(Psi/2) = (1/2 + e_r.e_b/2)^(n/2)  (M&G Eq. 3.104).
        bulge = (0.5 + 0.5 * cos_psi) ** (self.bulge_exponent / 2.0)
        return rho_min + (rho_max - rho_min) * bulge


@dataclass(frozen=True)
class ExponentialAtmosphere:
    """Single-exponential density model (M&G Eq. 3.99): ``rho = rho0 exp(-(h-h0)/H)``.

    The King-Hele drag-decay cross-check assumes an exponential atmosphere, so
    this is the oracle for that validation (``tests/test_drag.py``); it is also a
    simple second implementation cross-tested against Harris-Priester. No diurnal
    bulge -- the position and Sun arguments are accepted and ignored so density
    models share a common call signature.
    """

    rho0_kg_m3: float
    scale_height_km: float
    h0_ref_km: float = 0.0
    label: str = "exponential"

    def density_kg_m3(
        self,
        alt_km: float,
        r_hat_j2000: Optional[np.ndarray] = None,
        sun_hat_j2000: Optional[np.ndarray] = None,
    ) -> float:
        return float(
            self.rho0_kg_m3
            * math.exp(-(alt_km - self.h0_ref_km) / self.scale_height_km)
        )
