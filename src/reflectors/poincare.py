"""Poincaré-section diagnostics for orbital trajectories.

Detect canonical reference events along a propagated state history and
report osculating elements at each crossing. Two reference events are
exposed:

  - **Ascending node**: ``z`` (J2000 z-component of position) transitions
    from negative to positive (satellite crosses the equator going
    north). For prograde orbits with non-zero inclination this gives
    exactly one crossing per orbit, regardless of eccentricity. At the
    K-orbit-per-Mars-solar-sol repeating ground track this yields exactly
    K crossings per sol — the preferred Poincaré section for closure
    diagnostics on near-circular orbits.

  - **Periapsis**: ``r·v`` transitions from negative (descending) to
    positive (ascending) — i.e. true anomaly ν crosses 0. Cleanly
    one-per-orbit only for orbits with eccentricity well above the
    perturbation-driven radial-noise floor; for very low-e orbits (e
    below ~1e-3 in the reference geometry) perturbation harmonics produce
    multiple sign changes per orbit.

Detection uses linear interpolation across the sample pair where the
sign change is observed. State at the crossing is also linearly
interpolated; osculating elements at the crossing are computed via
``elements.elements_in_mme2000`` so inclination / RAAN / argp are
referenced to Mars's equator (not J2000) — matching the convention used
throughout the rest of the codebase.

These helpers provide the closure-diagnostics pass over a per-sol
propagation. This module is the single home for Poincaré
detection going forward; the cost machinery in ``optimize.py``
imports ``find_ascending_node_crossings`` to populate the
``Δe_at_same_u`` term used by the closure-targeted cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import math

import numpy as np

from reflectors.elements import elements_in_mme2000
from reflectors.surface import mars_equatorial_radius_km

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoincareCrossing:
    """One Poincaré-section crossing.

    Times are returned relative to the propagation's t=0 origin. To
    place crossings on an absolute SPICE ET timeline the caller adds
    its own ``epoch_et`` offset; this module takes no opinion on what
    ``epoch_et`` the caller's propagation started from.

    ``crossing_index`` is the 0-based index of the crossing within the
    full crossings list returned by the detector — useful for matching
    crossings across propagations (e.g. Δe at the same crossing index
    in two different sols).
    """

    crossing_index: int
    t_rel_s: float
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    nu_deg: float
    r_km: float
    altitude_km: float


def _scan_zero_crossings_neg_to_pos(
    f_arr: np.ndarray,
) -> list[tuple[int, float]]:
    """Find indices ``k`` where ``f[k] < 0`` and ``f[k+1] >= 0``.

    Returns a list of ``(k, frac)`` tuples where ``frac`` is the
    linear-interpolation fraction in ``[0, 1]`` for the zero-crossing
    within the sample pair ``(k, k+1)``. Skips degenerate pairs where
    ``f[k+1] - f[k]`` rounds to zero (cannot occur for a real sign
    change).
    """
    crossings: list[tuple[int, float]] = []
    n = len(f_arr)
    for k in range(n - 1):
        if f_arr[k] < 0.0 and f_arr[k + 1] >= 0.0:
            denom = f_arr[k + 1] - f_arr[k]
            if abs(denom) < 1e-30:
                continue
            frac = float(-f_arr[k] / denom)
            crossings.append((k, frac))
    return crossings


def _build_crossing_record(
    crossing_index: int,
    k: int,
    frac: float,
    t_s: np.ndarray,
    state_arr: np.ndarray,
    mu_km3_s2: float,
    epoch_et_offset: float,
) -> PoincareCrossing:
    """Linearly interpolate the crossing time + state, compute
    osculating elements at the interpolated state, and package the
    record.

    Elements are computed in MME2000 (Mars's equator), matching the
    rest of the codebase's convention for sun-sync orbital diagnostics.
    Altitude is reported as ``r - R_eq_Mars`` (spherical, not geodetic
    — geodetic conversion lives in ``surface.py``/``spice.recgeo`` if
    callers need it).
    """
    t_cross_rel = float(t_s[k]) + frac * (float(t_s[k + 1]) - float(t_s[k]))
    state_cross = (1.0 - frac) * state_arr[k] + frac * state_arr[k + 1]
    elts = elements_in_mme2000(
        state_cross,
        mu_km3_s2,
        epoch_et=epoch_et_offset + t_cross_rel,
    )
    r_cross = float(np.linalg.norm(state_cross[:3]))
    return PoincareCrossing(
        crossing_index=crossing_index,
        t_rel_s=t_cross_rel,
        a_km=float(elts.a_km),
        e=float(elts.e),
        inc_deg=math.degrees(float(elts.inclination_rad)),
        raan_deg=math.degrees(float(elts.raan_rad)),
        argp_deg=math.degrees(float(elts.argp_rad)),
        nu_deg=math.degrees(float(elts.nu_rad)),
        r_km=r_cross,
        altitude_km=r_cross - mars_equatorial_radius_km(),
    )


def find_ascending_node_crossings(
    t_s: np.ndarray,
    state_arr: np.ndarray,
    mu_km3_s2: float,
    *,
    epoch_et_offset: float = 0.0,
) -> list[PoincareCrossing]:
    """Detect ascending-node crossings on a dense propagation history.

    A crossing is the satellite's z-coordinate transitioning from
    ``< 0`` to ``≥ 0`` between consecutive samples — i.e. the
    satellite crossing the equator going north. For prograde orbits
    with non-zero inclination this is exactly one crossing per orbit
    regardless of eccentricity, making it a direct Poincaré
    section for near-circular sun-sync orbits.

    Parameters
    ----------
    t_s : (N,) array_like
        Propagation sample times relative to propagation t=0, seconds.
    state_arr : (N, 6) array_like
        Cartesian states ``[r_x, r_y, r_z, v_x, v_y, v_z]`` (km, km/s)
        in Mars-centred J2000.
    mu_km3_s2 : float
        Central-body GM (km³/s²) for osculating-element computation.
    epoch_et_offset : float, optional
        SPICE ET corresponding to ``t_s[0] = 0``. Applied to crossing
        ``t_rel_s`` when computing osculating elements so the
        MME2000 rotation pole is evaluated at the right epoch). The
        returned ``t_rel_s`` is propagation-relative, NOT absolute ET.
        Default 0.0.

    Returns
    -------
    list[PoincareCrossing]
        One record per crossing, in time order, with ``crossing_index``
        starting at 0.

    Notes
    -----
    Detection is one-sided (negative to non-negative); a satellite
    crossing the equator going SOUTH (``z`` going positive to negative)
    is filtered out. This guarantees exactly K crossings per sol at
    K-orbit-per-sol repeating ground tracks, regardless of eccentricity
    or short-period z oscillation amplitude.
    """
    t_s = np.asarray(t_s, dtype=float)
    state_arr = np.asarray(state_arr, dtype=float)
    if state_arr.ndim != 2 or state_arr.shape[1] != 6:
        raise ValueError(
            f"state_arr must be shape (N, 6), got {state_arr.shape}"
        )
    if state_arr.shape[0] != t_s.shape[0]:
        raise ValueError(
            f"t_s and state_arr length mismatch: "
            f"{t_s.shape[0]} vs {state_arr.shape[0]}"
        )

    z_arr = state_arr[:, 2]
    out: list[PoincareCrossing] = []
    for cnt, (k, frac) in enumerate(_scan_zero_crossings_neg_to_pos(z_arr)):
        out.append(_build_crossing_record(
            cnt, k, frac, t_s, state_arr, mu_km3_s2, epoch_et_offset,
        ))
    return out


def find_periapsis_crossings(
    t_s: np.ndarray,
    state_arr: np.ndarray,
    mu_km3_s2: float,
    *,
    epoch_et_offset: float = 0.0,
) -> list[PoincareCrossing]:
    """Detect periapsis crossings on a dense propagation history.

    A crossing is the dot product ``r·v`` transitioning from negative
    (descending) to positive (ascending) — i.e. true anomaly ν
    crossing 0. For orbits with eccentricity well above the
    perturbation-driven radial-noise floor, this is one crossing per
    orbit; for very low-e orbits (e ~ 1e-3 or smaller in the reference
    geometry) higher-frequency perturbation harmonics can produce
    multiple sign changes per orbit, which is physically informative
    but makes the diagnostic harder to interpret as a Poincaré
    section. Prefer ``find_ascending_node_crossings`` for closure
    diagnostics on near-circular orbits.

    Parameters and returns: see ``find_ascending_node_crossings``.
    """
    t_s = np.asarray(t_s, dtype=float)
    state_arr = np.asarray(state_arr, dtype=float)
    if state_arr.ndim != 2 or state_arr.shape[1] != 6:
        raise ValueError(
            f"state_arr must be shape (N, 6), got {state_arr.shape}"
        )
    if state_arr.shape[0] != t_s.shape[0]:
        raise ValueError(
            f"t_s and state_arr length mismatch: "
            f"{t_s.shape[0]} vs {state_arr.shape[0]}"
        )

    r_arr = state_arr[:, :3]
    v_arr = state_arr[:, 3:6]
    rdotv = np.einsum("ij,ij->i", r_arr, v_arr)

    out: list[PoincareCrossing] = []
    for cnt, (k, frac) in enumerate(_scan_zero_crossings_neg_to_pos(rdotv)):
        out.append(_build_crossing_record(
            cnt, k, frac, t_s, state_arr, mu_km3_s2, epoch_et_offset,
        ))
    return out
