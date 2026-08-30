"""Mission-design helpers for sun-synchronous Mars orbits.

Forward transformation ``(a, LTAN, M0, epoch) -> (r, v)_J2000``, for use
by sweeps and trajectory optimisers.

Functions
---------
``sun_sync_inclination_rad(a_km, ...)``
    First-order Brouwer retrograde inclination at which the J_2-driven
    RAAN drift equals ``2 pi / MARS_SIDEREAL_YEAR_S``. For Mars this is
    ~92.92 deg at 400 km altitude and trends toward 90 deg as altitude
    decreases; above roughly 6600 km altitude, no solution exists.

``raan_mme2000_from_ltan(ltan_h, epoch_et)``
    LTAN (local mean solar time of the ascending node, hours) ->
    RAAN in MME2000 at epoch.  Convention: 12 h = sub-solar (noon),
    18 h = evening terminator (dawn-dusk orbit), 6 h = dawn, 0/24 h
    = anti-sub-solar (midnight).

``initial_state_j2000(a_km, ltan_h, M0_rad, epoch_et, ...)``
    Composes the two into a circular sun-sync state, e=0, argp=0,
    ``nu = M_0``.  Returned in Mars-centred J2000 (the frame
    ``reflectors.dynamics.propagate`` consumes).

Conventions
-----------
All inclinations are relative to the Mars equator (MME2000); RAAN
likewise lives in MME2000.  The state is then rotated to Mars-centred
J2000 before being returned, matching the frame ``propagate`` uses.
The MME2000 rotation is evaluated at the same epoch the caller asks
about; over a multi-year mission the pole precesses by ~arcsec and
the distinction between "MME-of-epoch-A" and "MME-of-epoch-B" only
matters at the arcsec level.

References
----------
Brouwer, D. (1959). "Solution of the problem of artificial satellite
theory without drag." AJ 64: 378-396.  First-order secular node drift
under a zonal potential.

Vallado, D.A. (2013). *Fundamentals of Astrodynamics and Applications*,
4th ed., Microcosm Press.  §2.6 classical-elements <-> Cartesian.
See also ``reflectors.elements.state_from_classical_mme2000`` for the
perifocal -> MME2000 -> J2000 chain used here.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import spiceypy as spice

from reflectors.elements import (
    mme2000_rotation_from_j2000,
    state_from_classical_mme2000,
)
from reflectors.ephemeris import sun_state_j2000
from reflectors.mars_constants import (
    MARS_SIDEREAL_YEAR_S,
    SECONDS_PER_SOLAR_SOL_S,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gravity-model anchors (lazy so importing sun_sync.py doesn't force the
# MRO120F load path if the caller provides explicit mu/R/J_2).
# ---------------------------------------------------------------------------


def _default_gravity_anchors() -> tuple[float, float, float]:
    """(mu, R_ref, J_2) from MRO120F at degree 2.  Cached by ``gravity``."""
    from reflectors.gravity import mars_gravity_model, zonal_coefficients

    model = mars_gravity_model(max_degree=2)
    J2 = zonal_coefficients(model, 2)[2]
    return model.mu_km3_s2, model.ref_radius_km, J2


# ---------------------------------------------------------------------------
# Brouwer sun-sync inclination
# ---------------------------------------------------------------------------


def sun_sync_inclination_rad(
    a_km: float,
    *,
    mu_km3_s2: Optional[float] = None,
    ref_radius_km: Optional[float] = None,
    J2: Optional[float] = None,
    target_period_s: Optional[float] = None,
) -> float:
    """Brouwer first-order retrograde sun-sync inclination, radians.

    Solves ``dRAAN/dt = -(3/2) n J_2 (R/p)^2 cos(i) = 2 pi / T_year`` for
    a circular orbit (``p = a``).  Raises ``ValueError`` when no solution
    exists (``|cos(i)| > 1``; occurs above roughly 6600 km altitude at
    Mars).

    The returned inclination is in [pi/2, pi] (retrograde).  At Mars,
    ``J_2 > 0`` and ``dRAAN/dt > 0`` target imply ``cos(i) < 0``.

    ``target_period_s`` is the orbital period over which the node must
    precess one full turn to stay sun-synchronous; defaults to
    ``MARS_SIDEREAL_YEAR_S``. Pass the Earth sidereal year (with Earth ``mu_km3_s2`` /
    ``ref_radius_km`` / ``J2``) to get the ~98.6 deg Earth-LEO sun-sync
    inclination.
    """
    if a_km <= 0.0:
        raise ValueError(f"a_km must be > 0, got {a_km}")
    if mu_km3_s2 is None or ref_radius_km is None or J2 is None:
        default_mu, default_R, default_J2 = _default_gravity_anchors()
        if mu_km3_s2 is None:
            mu_km3_s2 = default_mu
        if ref_radius_km is None:
            ref_radius_km = default_R
        if J2 is None:
            J2 = default_J2
    if target_period_s is None:
        target_period_s = MARS_SIDEREAL_YEAR_S
    n = float(np.sqrt(mu_km3_s2 / a_km ** 3))
    target_rate = 2.0 * np.pi / target_period_s
    denom = 1.5 * n * (ref_radius_km / a_km) ** 2 * J2
    cos_i = -target_rate / denom
    if abs(cos_i) > 1.0:
        raise ValueError(
            f"no sun-sync solution at a_km={a_km}: cos(i)={cos_i:.4f} "
            "(altitude too high for J_2 drift to match the Mars year)"
        )
    return float(np.arccos(cos_i))


# ---------------------------------------------------------------------------
# Repeating ground track on a sun-sync orbit
# ---------------------------------------------------------------------------


def repeat_ground_track_altitude(
    K_orbits_per_solar_sol: int,
    *,
    mu_km3_s2: Optional[float] = None,
    ref_radius_km: Optional[float] = None,
    J2: Optional[float] = None,
    solar_sol_s: float = SECONDS_PER_SOLAR_SOL_S,
) -> tuple[float, float, float]:
    """Semi-major axis for a repeating-ground-track sun-sync orbit at Mars.

    Solves for the circular semi-major axis ``a`` such that exactly
    ``K`` orbital revolutions complete in one Mars solar day, using
    the pinned ``SECONDS_PER_SOLAR_SOL_S``.

    The sun-sync identity at Mars is

        d(Omega)/dt_sun_sync = 2 pi / T_sidereal_year

    so the orbit-plane rotation rate relative to the Mars-fixed body
    frame equals

        omega_eff = omega_Mars_inertial - d(Omega)/dt
                  = 2 pi / T_sidereal_day - 2 pi / T_sidereal_year
                  = 2 pi / T_solar_sol

    by the synodic identity (cf. ``mars_constants.SECONDS_PER_SOLAR_SOL_S``
    docstring). Therefore for K orbits to fill exactly one Mars-fixed
    rotation of the orbit plane,

        K * T_orb = T_solar_sol  =>  T_orb = T_solar_sol / K

    and Kepler's third law gives ``a = (mu T_orb^2 / (4 pi^2))^(1/3)``.
    The returned ``a`` is then verified against ``sun_sync_inclination_rad``;
    a ``ValueError`` propagates if no sun-sync solution exists at this
    altitude (above ~6600 km altitude on Mars).

    Parameters
    ----------
    K_orbits_per_solar_sol
        Integer K >= 1: number of orbital revolutions per Mars solar
        day for ground-track closure.
    mu_km3_s2, ref_radius_km, J2
        Gravity anchors for the Kepler period and the sun-sync
        inclination.  Default to MRO120F values when omitted, matching
        ``sun_sync_inclination_rad``.
    solar_sol_s
        Synodic period; default ``SECONDS_PER_SOLAR_SOL_S``. Exposed
        for testing the synodic-identity reduction.

    Returns
    -------
    a_km
        Circular semi-major axis (km) for the K-orbit-per-sol repeat.
    i_sun_sync_rad
        Brouwer first-order sun-sync inclination at this ``a``, radians.
    T_orb_s
        Orbital period at this ``a`` (= solar_sol_s / K), seconds.

    Raises
    ------
    ValueError
        If ``K_orbits_per_solar_sol < 1`` or non-integer; if no sun-sync
        solution exists at the computed altitude (propagated from
        ``sun_sync_inclination_rad``).

    References
    ----------
    Vallado, D.A. (2013). *Fundamentals of Astrodynamics and
    Applications*, 4th ed., Microcosm Press, Sec. 11.4 (repeating
    ground tracks).  The Mars sun-sync simplification is the synodic
    identity ``T_solar = (1/T_sd - 1/T_year)^{-1}`` applied to the
    Brouwer J_2 condition; cf. ``mars_constants`` docstring.
    """
    if not isinstance(K_orbits_per_solar_sol, (int, np.integer)):
        raise ValueError(
            f"K_orbits_per_solar_sol must be an integer, got "
            f"{type(K_orbits_per_solar_sol).__name__}"
        )
    K = int(K_orbits_per_solar_sol)
    if K < 1:
        raise ValueError(
            f"K_orbits_per_solar_sol must be >= 1, got {K}"
        )
    if solar_sol_s <= 0.0:
        raise ValueError(
            f"solar_sol_s must be > 0, got {solar_sol_s}"
        )
    if mu_km3_s2 is None or ref_radius_km is None or J2 is None:
        default_mu, default_R, default_J2 = _default_gravity_anchors()
        if mu_km3_s2 is None:
            mu_km3_s2 = default_mu
        if ref_radius_km is None:
            ref_radius_km = default_R
        if J2 is None:
            J2 = default_J2

    T_orb = float(solar_sol_s) / float(K)
    a_km = float(
        (float(mu_km3_s2) * T_orb * T_orb / (4.0 * np.pi * np.pi)) ** (1.0 / 3.0)
    )
    i_ss = sun_sync_inclination_rad(
        a_km,
        mu_km3_s2=float(mu_km3_s2),
        ref_radius_km=float(ref_radius_km),
        J2=float(J2),
    )
    return a_km, i_ss, T_orb


# ---------------------------------------------------------------------------
# LTAN -> RAAN (MME2000)
# ---------------------------------------------------------------------------


def raan_mme2000_from_ltan(ltan_h: float, epoch_et: float) -> float:
    """RAAN (rad, MME2000) corresponding to the given LTAN at ``epoch_et``.

    LTAN convention (local mean solar time of the ascending node):
        12 h  -- sub-solar point at AN (noon); orbit plane contains the Mars-Sun line.
        18 h  -- evening terminator at AN; orbit plane perpendicular to Mars-Sun (dawn-dusk).
         6 h  -- morning terminator at AN (dawn-dusk, opposite sense).
         0/24 h -- anti-sub-solar at AN (midnight).

    The ascending node is ``(LTAN - 12) * 15 deg`` east of the sub-solar
    meridian.  In an inertial MME2000 frame, the sub-solar meridian's
    right ascension equals ``atan2(r_sun_y_MME, r_sun_x_MME)``, so
        RAAN = RA_sun_MME + (LTAN - 12) * 15 deg   (mod 2 pi).

    Note on "mean" vs "true" solar time: this calculation uses the Sun's
    instantaneous MME-frame direction, which corresponds to *true* local
    solar time.
    The mean-vs-true discrepancy at Mars (equation of time) is a few
    minutes, i.e. <1 deg of LTAN -- below the LTAN grid spacing
    (typically 1 h / 15 deg) in any practical sweep.  For strict
    mean-solar-time LTAN, add the equation-of-time correction at the
    epoch.
    """
    state_sun = sun_state_j2000(float(epoch_et), "MARS")
    r_sun_j2000 = np.asarray(state_sun[:3], dtype=float)
    R = mme2000_rotation_from_j2000(float(epoch_et))
    r_sun_mme = R @ r_sun_j2000
    ra_sun_mme = float(np.arctan2(r_sun_mme[1], r_sun_mme[0]))
    raan = ra_sun_mme + np.radians((float(ltan_h) - 12.0) * 15.0)
    return float(raan % (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Composed initial state
# ---------------------------------------------------------------------------


def initial_state_j2000(
    a_km: float,
    ltan_h: float,
    M0_rad: float,
    epoch_et: float,
    *,
    mu_km3_s2: Optional[float] = None,
    ref_radius_km: Optional[float] = None,
    J2: Optional[float] = None,
) -> np.ndarray:
    """Circular sun-synchronous initial state in Mars-centred J2000.

    Parameters
    ----------
    a_km
        Circular semi-major axis (km).  Must admit a sun-sync solution
        (see ``sun_sync_inclination_rad``).
    ltan_h
        Local mean solar time of the ascending node (hours); sets RAAN.
    M0_rad
        Initial mean anomaly (rad).  For e=0, equals the initial true
        anomaly, i.e. the angular position from the ascending node at
        ``epoch_et``.  Full [0, 2 pi) is meaningful.
    epoch_et
        Absolute SPICE ET at which the state is defined.

    ``mu``, ``R_ref``, ``J_2`` default to MRO120F values when omitted.
    ``argp`` is pinned to 0 (degenerate for e=0).

    Returns
    -------
    np.ndarray, shape (6,)
        ``[r_x, r_y, r_z, v_x, v_y, v_z]`` in km and km/s, Mars-centred
        J2000 axes.
    """
    if mu_km3_s2 is None or ref_radius_km is None or J2 is None:
        default_mu, default_R, default_J2 = _default_gravity_anchors()
        if mu_km3_s2 is None:
            mu_km3_s2 = default_mu
        if ref_radius_km is None:
            ref_radius_km = default_R
        if J2 is None:
            J2 = default_J2

    i_rad = sun_sync_inclination_rad(
        a_km, mu_km3_s2=mu_km3_s2, ref_radius_km=ref_radius_km, J2=J2,
    )
    raan_rad = raan_mme2000_from_ltan(ltan_h, epoch_et)

    return state_from_classical_mme2000(
        a_km=a_km,
        e=0.0,
        inclination_rad=i_rad,
        raan_rad=raan_rad,
        argp_rad=0.0,
        nu_rad=float(M0_rad),
        mu_km3_s2=mu_km3_s2,
        epoch_et=epoch_et,
    )
