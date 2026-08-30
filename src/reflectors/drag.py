"""Atmospheric drag on a flat solar sail (Earth LEO-start escape).

Target physics: the drag acceleration on a flat sail of area ``A``, mass ``m``,
and normal ``n_hat`` moving at relative velocity ``v_rel`` through an atmosphere
of density ``rho``. For a LEO-start escape the sail's huge area-to-mass ratio
(~55 m^2/kg at sigma=18) makes drag first-order, NOT a small correction.

Primary reference: Montenbruck & Gill (2000), *Satellite
Orbits*, §3.5 "Atmospheric Drag". Drag acceleration, Eq. (3.97) p.84:

    r-ddot = -(1/2) C_D (A/m) rho v_r^2 e_v ,   e_v = v_r / |v_r|

(the 1/2 is the aerodynamic-pressure convention, M&G p.84). The relative
velocity accounts for the co-rotating atmosphere, Eq. (3.98) p.85:

    v_r = v - omega_earth x r ,

with the rigid co-rotation assumption good to <5% in drag force (King-Hele
1987, cited by M&G p.85). Winds above 220 km are outside this model.

FLAT-PLATE COUPLING. M&G's ``A`` is the cross-sectional area into the flow. For
a sail whose normal ``n_hat`` is an integrated attitude state (not Earth-pointing
with constant area, M&G p.85), the silhouette into the flow is the PROJECTED
area ``A_proj = A * |n_hat . e_v|``: full area broadside to the flow, zero
edge-on. Only the anti-velocity (drag) component is retained; M&G p.83 notes the
lift / binormal forces "can safely be neglected in most cases". A full
free-molecular flat-plate model with a normal/lift term (Sentman 1961; Cook
1965) is not included. That term vanishes at 0 and 90 deg incidence and does
not alter the King-Hele secular ``<da/dt>`` comparison used for validation.

C_D = 2.2 (free-molecular convex body, M&G p.84-85); see
``reflectors.atmosphere_constants.DEFAULT_DRAG_COEFFICIENT``.

Validation (drag is NON-conservative -> no potential, so the finite-difference-
of-potential cross-check used for gravity/third-body does NOT apply): the
independent check is the King-Hele (1964) Ch.4 §18 circular-orbit contraction
``da/dt = -(C_D A/m) rho sqrt(mu a)`` (equivalently ``Delta a_per_rev =
-2 pi (C_D A/m) rho a^2``), derivable from the Gauss energy equation -- see
``tests/test_drag.py``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import spiceypy as spice

from reflectors.atmosphere_constants import DEFAULT_DRAG_COEFFICIENT

SUN_NAIF_ID = 10


# ---------------------------------------------------------------------------
# Pure core (SPICE-free, hot loop)
# ---------------------------------------------------------------------------


def drag_acceleration(
    v_rel_km_s: np.ndarray,
    n_hat: np.ndarray,
    rho_kg_m3: float,
    C_d: float,
    area_m2: float,
    mass_kg: float,
) -> np.ndarray:
    """Flat-plate drag acceleration (km/s^2), M&G Eq. (3.97) with A -> A_proj.

        a = -(1/2) C_D (A_proj / m) rho |v_rel| v_rel ,
        A_proj = area * |n_hat . e_v| ,  e_v = v_rel / |v_rel| .

    Parameters
    ----------
    v_rel_km_s
        Atmosphere-relative velocity ``v - omega x r`` (km/s, J2000), shape (3,).
    n_hat
        Unit sail normal (J2000). The projected area depends on it: broadside to
        the flow (n_hat || e_v) -> full area; edge-on (n_hat . e_v = 0) -> zero.
    rho_kg_m3
        Atmospheric mass density at the sail (kg/m^3).
    C_d, area_m2, mass_kg
        Drag coefficient, sail area (m^2), total mass (kg).

    Returns
    -------
    ndarray, shape (3,)
        Acceleration in km/s^2, anti-parallel to ``v_rel``. Zero vector if
        ``rho`` is 0, the relative speed is 0, or the sail is edge-on.

    Units: computed in SI (m, kg, s) and converted to km/s^2 by 1e-3 at the end.
    ``A_proj/m`` [m^2/kg] * rho [kg/m^3] * v^2 [m^2/s^2] = m/s^2.
    """
    v_rel = np.asarray(v_rel_km_s, dtype=float) * 1.0e3  # km/s -> m/s
    vmag = float(np.linalg.norm(v_rel))
    if rho_kg_m3 <= 0.0 or vmag == 0.0:
        return np.zeros(3)
    e_v = v_rel / vmag
    n = np.asarray(n_hat, dtype=float)
    n = n / float(np.linalg.norm(n))
    a_proj = area_m2 * abs(float(np.dot(n, e_v)))  # m^2 (silhouette into the flow)
    if a_proj == 0.0:  # edge-on: no drag in the projected-area model
        return np.zeros(3)
    a_mps2 = -0.5 * C_d * (a_proj / mass_kg) * rho_kg_m3 * vmag * v_rel  # m/s^2
    return a_mps2 * 1.0e-3  # -> km/s^2


# ---------------------------------------------------------------------------
# Co-rotating atmosphere velocity
# ---------------------------------------------------------------------------


def atmosphere_relative_velocity(
    r_j2000_km: np.ndarray,
    v_j2000_km_s: np.ndarray,
    et: float,
    body_frame: str,
) -> np.ndarray:
    """Atmosphere-relative velocity ``v - omega x r`` (M&G Eq. 3.98).

    The body's angular-velocity vector in J2000 is recovered from the
    state-transformation matrix: with ``R = pxform(body_frame -> J2000)`` and
    ``Rdot`` the lower-left block of ``sxform``, ``[omega x] = Rdot R^T`` in
    J2000 (the true spin-pole direction, not an assumed +z). This is the vector
    companion of ``surface.mars_rotation_rate_rad_per_s`` (which takes the norm).
    """
    T6 = np.asarray(spice.sxform(body_frame, "J2000", et), dtype=float)
    R = T6[:3, :3]
    Rdot = T6[3:, :3]
    W = Rdot @ R.T  # skew(omega) in J2000
    omega = np.array([W[2, 1], W[0, 2], W[1, 0]])
    return np.asarray(v_j2000_km_s, dtype=float) - np.cross(
        omega, np.asarray(r_j2000_km, dtype=float)
    )


# ---------------------------------------------------------------------------
# SPICE-fed state wrapper
# ---------------------------------------------------------------------------


def drag_acceleration_from_state(
    r_j2000_km: np.ndarray,
    v_j2000_km_s: np.ndarray,
    et: float,
    n_hat: np.ndarray,
    sail,
    density_model,
    *,
    central_body,
    C_d: float = DEFAULT_DRAG_COEFFICIENT,
) -> np.ndarray:
    """Drag acceleration at a state (km/s^2): density + co-rotation + core.

    Computes geodetic altitude ``|r| - R_eq``, the satellite + Sun unit vectors
    (for the Harris-Priester diurnal bulge), the co-rotating relative velocity,
    then the flat-plate core. ``sail`` provides ``area_m2`` / ``mass_kg``;
    ``central_body`` provides the equatorial radius + body-fixed frame + NAIF id.
    """
    r = np.asarray(r_j2000_km, dtype=float)
    r_mag = float(np.linalg.norm(r))
    alt_km = r_mag - central_body.equatorial_radius_km
    r_hat = r / r_mag

    sun_state, _ = spice.spkezr(
        str(SUN_NAIF_ID), et, "J2000", "NONE", str(central_body.naif_id)
    )
    sun_vec = np.asarray(sun_state[:3], dtype=float)
    sun_hat = sun_vec / float(np.linalg.norm(sun_vec))

    rho = density_model.density_kg_m3(alt_km, r_hat, sun_hat)
    v_rel = atmosphere_relative_velocity(r, v_j2000_km_s, et, central_body.body_frame)
    return drag_acceleration(v_rel, n_hat, rho, C_d, sail.area_m2, sail.mass_kg)


# ---------------------------------------------------------------------------
# Adapter for the escape RHS drag hook
# ---------------------------------------------------------------------------


def make_drag_force_fn(
    sail,
    density_model,
    *,
    central_body=None,
    C_d: float = DEFAULT_DRAG_COEFFICIENT,
) -> Callable[..., np.ndarray]:
    """Build the ``drag_force_fn(r, v, n, et) -> a_km_s2`` closure for
    ``escape.propagate_escape(drag_force_fn=...)``.

    ``central_body`` defaults to Earth (the LEO-start drag regime). The closure
    captures the sail, density model, central body, and C_d; the escape RHS
    passes it the actual integrated sail normal ``n`` each step.
    """
    if central_body is None:
        from reflectors.central_body import earth_central_body

        central_body = earth_central_body()

    def drag_force_fn(r, v, n, et):
        return drag_acceleration_from_state(
            r, v, et, n, sail, density_model,
            central_body=central_body, C_d=C_d,
        )

    return drag_force_fn
