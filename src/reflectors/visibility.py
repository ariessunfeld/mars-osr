"""Visibility gates and delivery-quality scalars for sail->target reflection.

Target physics: decide whether and how well a Mars-orbiting solar sail
can specularly reflect sunlight onto a fixed surface target at a given
epoch, and -- over a propagated trajectory -- enumerate the "delivery
windows" during which all four necessary conditions hold simultaneously.
This module supplies the reflection-delivery foundation: the specular pointing
law, the four-condition visibility gate, and the window finder with
per-window quality scalars. Consumers build higher-level analyses on top.

The geometry uses the ``ephemeris``, ``surface``, ``shadow``, and
``attitude`` modules.
Finite sun angular size and the finite-sail footprint are implemented by the
beam module used here. A caller may supply scalar or sample-aligned direct
atmospheric transmission; the default is unity (vacuum). Target-surface
albedo, off-specular reflection lobes, and MOLA-corrected local horizons
remain outside this module's geometry boundary.

The module codifies the four-condition gate, three-phase attitude history, and
coupling between orbit design and attitude feasibility.

Reference (pointing law): the specular half-angle-bisector identity
``n_hat* = unit(s_hat_sun + s_hat_target)`` is a standard optics-
textbook result (e.g., Born & Wolf, *Principles of Optics*, §1.5
"Reflection and refraction at a plane surface"): a mirror oriented
along the bisector of the incoming and outgoing unit vectors produces
specular reflection between them.

Conventions
-----------

All vectors are in Mars-centred J2000 axes, km. All angles passed to
helpers are in radians unless the function name says ``_deg``.
Longitudes are east-positive (matching ``reflectors.surface`` and the
IAU_MARS body-fixed frame; NOT the Mars west-positive planetographic
convention SPICE's ``pgrrec`` would use by default).

Outward-normal formula
----------------------

For a biaxial oblate spheroid, the outward normal at planetographic
latitude ``phi`` and east longitude ``lambda`` is exactly

    n_hat_body_fixed = (cos phi cos lambda,
                        cos phi sin lambda,
                        sin phi)

by the defining property of planetographic latitude (the angle between
the surface normal and the equatorial plane). For the planetocentric-
sphere mode used by ``surface.surface_point_body_fixed(
planetographic=False)`` the outward normal coincides with the radial
direction, which has the same expression. So one formula covers both
conventions -- this module uses it directly rather than finite-
differencing the position. Derivable as the gradient of the spheroid
constraint ``(x/a)^2 + (y/a)^2 + (z/c)^2 = 1`` evaluated at the
corresponding rectangular coordinates; after normalisation the ``a``
and ``c`` factors cancel.

Horizon / elevation test
------------------------

A sail at position ``r_sat`` lies above the target's local horizon
plane (the plane through ``r_target`` perpendicular to the outward
normal) iff ``dot(r_sat - r_target, n_outward) > 0``. The elevation
angle above the horizon is
``arcsin(dot((r_sat - r_target)/|r_sat - r_target|, n_outward))``,
valued in ``[-pi/2, +pi/2]``.

These are pure-geometry tests. For a sphere of radius
``|r_target|``, "above the target's horizon plane" is equivalent to
"line of sight from sail to target clears the sphere" because the
plane is tangent to the sphere at ``r_target`` and the sphere's
interior is strictly in the opposite half-space. For Mars's biaxial
spheroid (flattening ~0.59%) the approximation slightly under-bounds
the occultation at high latitudes; the residual is bounded above by
the flattening and is systematically in the conservative direction
(a sail just "above the horizon" by a few metres might still be
geometrically visible over the planet's limb, not vice versa). This
is consistent with the spherical silhouette in ``reflectors.shadow``;
neither module includes an oblate-limb correction.

Geometry primitives
-------------------

    target_outward_normal_j2000(lat_deg, lon_deg, et)
    sail_above_target_horizon(r_sat, r_target, n_outward)
    target_elevation_angle_rad(r_sat, r_target, n_outward)
    slant_range_km(r_sat, r_target)

Bisector pointing law
---------------------

For a mirror-sail to specularly reflect sunlight onto a ground target,
the sail normal must lie along the bisector of the incoming sun
direction and the outgoing target direction (both measured FROM the
sail):

    s_hat_sun    = unit(r_sun    - r_sat)          # FROM sail TO sun
    s_hat_target = unit(r_target - r_sat)          # FROM sail TO target
    n_hat_star   = unit(s_hat_sun + s_hat_target)

Under this normal, reflecting ``s_hat_target`` about the mirror yields
``s_hat_sun`` (and vice versa) -- the defining property of specular
reflection.

The angle ``alpha`` between ``n_hat_star`` and ``s_hat_sun`` equals the
half-angle between ``s_hat_sun`` and ``s_hat_target`` (bisector
identity). So ``cos alpha`` measures "how face-on the sail is to the
sun" at this pointing: ``cos alpha = 1`` when sun and target are
coincident from the sail (sail directly below the sun and directly
above the target), dropping to ``cos alpha = 0`` as sun and target
become antiparallel from the sail. The case ``cos alpha -> 0`` is
GEOMETRICALLY DEGENERATE: no mirror normal can reflect a light ray back
toward its origin. Callers gate on ``cos alpha > threshold`` before
using ``n_hat_star``.

    bisector_normal(r_sat, r_target, r_sun)
        -> (n_hat_star, cos_alpha), with (zeros, 0.0) on degenerate.

    bisector_pointing(target_lat_deg, target_lon_deg)
        -> AttitudeCallable (r_sat_km, et) -> n_hat_star_j2000,
        raising on degenerate.

The signature of ``bisector_pointing`` matches
``reflectors.attitude.AttitudeCallable`` so it composes directly with
``srp.srp_acceleration`` and ``attitude.angular_acceleration`` (the
latter is how peak slew-rate demand is computed per delivery window).

Higher-level helpers (four-gate evaluator, window finder, and
per-window quality scalars) are implemented below.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline

from reflectors.attitude import AttitudeCallable, angular_acceleration
from reflectors.ephemeris import (
    ephemeris_evaluation_context,
    frame_rotation,
    sun_state_j2000,
)
from reflectors.beam import (
    beam_footprint_semi_axes_km,
    delivered_surface_irradiance_W_per_m2,
)
from reflectors.shadow import in_mars_umbra
from reflectors.srp import SolarSail
from reflectors.surface import BODY_FIXED_FRAME, surface_point_position


logger = logging.getLogger(__name__)


MARS_NAIF_ID = 499
SUN_NAIF_ID = 10


# Threshold on |s_hat_sun + s_hat_target| below which the bisector is
# considered degenerate. |sum| = 2 cos(half_angle), so |sum| < 2e-3
# means half_angle > pi/2 - 1e-3, i.e. sun and target are within ~0.06°
# of antiparallel from the sail -- a degenerate geometry where a
# single mirror cannot reflect the sun back toward the target. Exposed
# as a module-level constant so tests can reference the same value.
_BISECTOR_DEGENERATE_SUM_TOL = 2.0e-3

# Minimum samples in a window for finite-difference-based slew-demand
# evaluation. CubicSpline requires >= 4 distinct x-values; the central
# difference needs a (-dt, 0, +dt) stencil, so the window must also be
# longer than 2 * dt. Both conditions enforced at the call site.
_SLEW_MIN_SAMPLES = 4
# Default FD half-step for alpha(t). Small enough to resolve the
# bisector variation over a fraction of the fastest orbital timescale
# (few-second scale) while large enough to stay above interpolant
# round-off.
_SLEW_DEFAULT_DT_S = 0.5
# Default subgrid density across a window for locating the peak.
# 50 points across a ~1500 s overhead LMO pass is 30 s spacing --
# enough to resolve a peak that evolves on the pass-duration
# timescale to ~10-20% accuracy. Parameterised but not
# exposed on find_delivery_windows -- callers wanting finer can
# re-invoke directly on the returned DeliveryWindow.
_SLEW_DEFAULT_SUBGRID_POINTS = 50


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def _target_outward_normal_body_fixed(
    lat_deg: float, lon_deg: float
) -> np.ndarray:
    """Unit outward normal at the target, in IAU_MARS body-fixed axes.

    Closed form from the planetographic-latitude definition; see module
    docstring for the derivation. Covers both ``planetographic=True``
    (biaxial spheroid) and ``planetographic=False`` (sphere) conventions
    in ``surface.surface_point_body_fixed``.
    """
    lat = np.radians(float(lat_deg))
    lon = np.radians(float(lon_deg))
    cl = np.cos(lat)
    return np.array([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], dtype=float)


def target_outward_normal_j2000(
    lat_deg: float, lon_deg: float, et: float
) -> np.ndarray:
    """Unit outward normal at the target at epoch ``et``, in J2000 axes.

    The body-fixed normal is the closed-form
    ``(cos phi cos lambda, cos phi sin lambda, sin phi)`` (see module
    docstring for the derivation); this function rotates it into J2000
    via ``spice.pxform``. Same vector across both planetographic (biaxial
    spheroid) and planetocentric (sphere) ``surface_point_body_fixed``
    conventions, so no such parameter here.

    Parameters
    ----------
    lat_deg
        East-positive latitude in degrees (planetographic by project
        convention).
    lon_deg
        East-positive longitude in degrees.
    et
        SPICE TDB seconds past J2000.

    Returns
    -------
    ndarray, shape (3,)
        Unit outward normal in J2000 axes.
    """
    n_bf = _target_outward_normal_body_fixed(lat_deg, lon_deg)
    M = frame_rotation(BODY_FIXED_FRAME, "J2000", float(et))
    return M @ n_bf


def sail_above_target_horizon(
    r_sat_j2000_km: np.ndarray,
    r_target_j2000_km: np.ndarray,
    n_target_outward_j2000: np.ndarray,
) -> bool:
    """Return True iff the sail is above the target's local horizon plane.

    Geometric line-of-sight test: ``dot(r_sat - r_target, n_outward) > 0``.
    See module docstring for the equivalence to "line of sight clears
    Mars" under the spherical approximation (exact) and the biaxial-
    spheroid correction (≲ 0.6%, conservative direction).

    Parameters
    ----------
    r_sat_j2000_km
        Sail position in Mars-centred J2000 axes, km, shape (3,).
    r_target_j2000_km
        Target position in the same frame, shape (3,).
    n_target_outward_j2000
        Unit outward normal at the target in J2000 axes, shape (3,).
        Not re-normalised here; callers are expected to pass a
        normalised vector (e.g., from ``target_outward_normal_j2000``).
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)
    r_target = np.asarray(r_target_j2000_km, dtype=float)
    n = np.asarray(n_target_outward_j2000, dtype=float)
    return float(np.dot(r_sat - r_target, n)) > 0.0


def target_elevation_angle_rad(
    r_sat_j2000_km: np.ndarray,
    r_target_j2000_km: np.ndarray,
    n_target_outward_j2000: np.ndarray,
) -> float:
    """Elevation angle of the sail above the target's horizon plane (radians).

    Computed as ``arcsin(dot(u_sail_from_target, n_outward))`` where
    ``u_sail_from_target`` is the unit vector from the target to the
    sail. Positive above the horizon, zero on the horizon, negative
    below. Values in ``[-pi/2, +pi/2]``; peaks at +pi/2 when the sail
    is directly along the outward normal (sail "at zenith" relative to
    the target's local vertical).

    Parameters
    ----------
    r_sat_j2000_km, r_target_j2000_km, n_target_outward_j2000
        See ``sail_above_target_horizon``.

    Raises
    ------
    ValueError
        If ``r_sat`` and ``r_target`` coincide (the direction is
        undefined).
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)
    r_target = np.asarray(r_target_j2000_km, dtype=float)
    n = np.asarray(n_target_outward_j2000, dtype=float)
    delta = r_sat - r_target
    d = float(np.linalg.norm(delta))
    if d == 0.0:
        raise ValueError(
            "target_elevation_angle_rad: sail and target positions coincide; "
            "elevation is undefined"
        )
    # Clamp to [-1, 1] to guard against round-off stepping the argument
    # outside arcsin's domain when the sail is exactly at zenith.
    sin_el = float(np.clip(np.dot(delta, n) / d, -1.0, 1.0))
    return float(np.arcsin(sin_el))


def slant_range_km(
    r_sat_j2000_km: np.ndarray, r_target_j2000_km: np.ndarray
) -> float:
    """Euclidean sail-to-target distance in km."""
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)
    r_target = np.asarray(r_target_j2000_km, dtype=float)
    return float(np.linalg.norm(r_sat - r_target))


# ---------------------------------------------------------------------------
# Bisector pointing law
# ---------------------------------------------------------------------------


def bisector_normal(
    r_sat_j2000_km: np.ndarray,
    r_target_j2000_km: np.ndarray,
    r_sun_j2000_km: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Specular-reflection sail normal and its cosine-alpha at the bisector.

    Computes ``n_hat_star = unit(s_hat_sun + s_hat_target)`` where
    both unit vectors point FROM THE SAIL toward the respective body
    (sun, target). See module docstring for the derivation and the
    ``cos alpha = cos(half-angle)`` identity.

    Parameters
    ----------
    r_sat_j2000_km
        Sail position in Mars-centred J2000 axes, km, shape (3,). Must
        not coincide with ``r_target_j2000_km`` or ``r_sun_j2000_km``.
    r_target_j2000_km
        Target position in the same frame, shape (3,).
    r_sun_j2000_km
        Sun position in the same frame, shape (3,).

    Returns
    -------
    tuple of (ndarray shape (3,), float)
        ``(n_hat_star, cos_alpha)``.
        ``n_hat_star`` is a unit vector in J2000 axes when the geometry
        is well-conditioned; when the geometry is degenerate (sun and
        target are antiparallel from the sail, within
        ``_BISECTOR_DEGENERATE_SUM_TOL``), returns ``(zeros(3), 0.0)``
        to signal the degeneracy. ``cos_alpha`` is always in ``[0, 1]``
        by construction (the bisector of two unit vectors cannot make
        an obtuse angle with either).

    Raises
    ------
    ValueError
        If ``r_sat`` coincides with ``r_target`` or ``r_sun`` (unit
        vectors would be undefined).
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)
    r_target = np.asarray(r_target_j2000_km, dtype=float)
    r_sun = np.asarray(r_sun_j2000_km, dtype=float)

    d_target = r_target - r_sat
    norm_target = float(np.linalg.norm(d_target))
    if norm_target == 0.0:
        raise ValueError(
            "bisector_normal: sail and target positions coincide; "
            "s_hat_target is undefined"
        )
    s_hat_target = d_target / norm_target

    d_sun = r_sun - r_sat
    norm_sun = float(np.linalg.norm(d_sun))
    if norm_sun == 0.0:
        raise ValueError(
            "bisector_normal: sail and sun positions coincide; "
            "s_hat_sun is undefined"
        )
    s_hat_sun = d_sun / norm_sun

    sum_vec = s_hat_sun + s_hat_target
    sum_norm = float(np.linalg.norm(sum_vec))

    if sum_norm < _BISECTOR_DEGENERATE_SUM_TOL:
        return np.zeros(3, dtype=float), 0.0

    n_hat_star = sum_vec / sum_norm
    cos_alpha = float(np.dot(n_hat_star, s_hat_sun))
    # cos_alpha is cos(half-angle) and lies in (0, 1] for all
    # non-degenerate geometries; clip for FP robustness.
    cos_alpha = max(0.0, min(1.0, cos_alpha))
    return n_hat_star, cos_alpha


def bisector_pointing(
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    alt_km: float = 0.0,
    planetographic: bool = True,
    observer_naif_id: int = MARS_NAIF_ID,
) -> AttitudeCallable:
    """Attitude callable for specular reflection onto a fixed surface target.

    Returned closure has signature ``(r_sat_j2000_km, et) ->
    n_hat_star_j2000`` and can be passed directly to
    ``reflectors.srp.srp_acceleration`` (as the ``n_hat_func`` kwarg)
    or to ``reflectors.attitude.angular_acceleration`` (as the
    ``profile`` argument). Both Sun and target positions are fetched
    fresh via SPICE on every evaluation so the profile inherits exact
    time-dependence of both.

    Parameters
    ----------
    target_lat_deg, target_lon_deg
        East-positive latitude / longitude of the target, degrees.
    alt_km
        Altitude above the reference spheroid for the target, km.
        Default 0 (on the spheroid surface); MOLA corrections are not modeled.
    planetographic
        Passed through to ``surface.surface_point_position``. Default
        True (biaxial spheroid); False selects the spherical
        approximation. See ``reflectors.surface`` docstring for the
        distinction.
    observer_naif_id
        Central body NAIF id for the Sun/target SPICE queries. Default
        499 (Mars planet centre), matching the propagator convention.

    Returns
    -------
    AttitudeCallable
        Closure raising ``ValueError`` at epochs where the geometry is
        degenerate (sun and target antiparallel from sail). Callers are
        expected to gate on ``bisector_feasible`` from ``delivery_gates``
        before evaluating.
    """
    lat = float(target_lat_deg)
    lon = float(target_lon_deg)
    alt = float(alt_km)
    planeto = bool(planetographic)
    obs_id = int(observer_naif_id)

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        r_sat = np.asarray(r_sat_km, dtype=float)
        r_target = surface_point_position(
            lat, lon, float(et), alt_km=alt, planetographic=planeto
        )
        state = sun_state_j2000(float(et), obs_id)
        r_sun = np.asarray(state[:3], dtype=float)
        n_hat_star, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        if cos_alpha == 0.0:
            raise ValueError(
                "bisector_pointing: geometry degenerate at "
                f"et={et} (sun and target antiparallel from sail). "
                "Gate on bisector_feasible before evaluating."
            )
        return n_hat_star

    return _n_hat


# ---------------------------------------------------------------------------
# Four-gate evaluator
# ---------------------------------------------------------------------------


class DeliveryGates(NamedTuple):
    """Four-condition gate for specular reflection onto a surface target.

    The four gates are computed but not
    combined inside this tuple; callers AND the subset they care
    about (see ``find_delivery_windows`` for the per-gate AND / IGNORE
    switch pattern).

    Fields
    ------
    sail_sunlit
        True iff the sail is NOT inside Mars's umbra. Shadow-gated
        by ``reflectors.shadow.in_mars_umbra``.
    sail_above_target_horizon
        True iff the sail's elevation angle above the target's local
        horizon plane is at least ``target_elevation_min_rad`` (default
        0 -> true horizon). "Above horizon" is the line-of-sight
        clearance condition; a positive threshold additionally
        prunes grazing / low-elevation passes where atmospheric
        extinction can be large.
    bisector_feasible
        True iff the specular half-angle bisector
        ``n_hat* = unit(s_hat_sun + s_hat_target)`` is well-defined
        AND ``cos_alpha = cos(half-angle)`` is at least
        ``bisector_cos_alpha_min`` (default 0.1). Excludes geometries
        where sun and target are near-antiparallel from the sail.
    target_in_sunlight
        True iff the target surface point is illuminated (sun above
        the target's local horizon plane:
        ``dot(s_sun_from_target, n_target_outward) > 0``). Not enabled
        for the default "total delivered energy" objective, but the
        capability is preserved for "night-only
        illumination" or "day-only augmentation" missions.
    """

    sail_sunlit: bool
    sail_above_target_horizon: bool
    bisector_feasible: bool
    target_in_sunlight: bool


class DeliveryGeometry(NamedTuple):
    """One-epoch delivery gates plus the geometry used to derive them.

    Returning these values prevents the window finder from repeating the Sun,
    surface-frame, elevation, slant-range, and bisector calculations
    immediately after evaluating the gates.
    """

    gates: DeliveryGates
    sun_position_j2000_km: np.ndarray
    target_position_j2000_km: np.ndarray
    target_outward_normal_j2000: np.ndarray
    slant_range_km: float
    target_elevation_rad: float
    cos_alpha: float


def target_in_sunlight_at(
    lat_deg: float,
    lon_deg: float,
    et: float,
    *,
    alt_km: float = 0.0,
    planetographic: bool = True,
    observer_naif_id: int = MARS_NAIF_ID,
) -> bool:
    """Return True iff the target surface point is illuminated at ``et``.

    Semantically: the Sun is above the target's local horizon plane at
    ``et``. Computed as ``dot(s_sun_from_target, n_target_outward) > 0``
    where ``s_sun_from_target`` is the unit vector from target to Sun.

    Simpler than a cone test because surface points cannot be inside
    the body's own umbra -- they are on the body. Nighttime at the
    target is equivalent to the Sun having set below the local
    horizon, which is captured exactly by this predicate (Mars's
    oblate-silhouette limb correction for terminator-crossing geometry
    is ≤ 0.6%, same budget as the sail-umbra approximation in
    ``reflectors.shadow``).

    Parameters
    ----------
    lat_deg, lon_deg
        East-positive latitude / longitude of the target, degrees.
    et
        SPICE TDB seconds past J2000.
    alt_km, planetographic
        Passed through to ``surface.surface_point_position``.
    observer_naif_id
        Central body NAIF id for the ``spkezr`` Sun query.
    """
    r_target = surface_point_position(
        float(lat_deg), float(lon_deg), float(et),
        alt_km=float(alt_km), planetographic=bool(planetographic),
    )
    state = sun_state_j2000(float(et), int(observer_naif_id))
    r_sun = np.asarray(state[:3], dtype=float)
    n_target_outward = target_outward_normal_j2000(
        float(lat_deg), float(lon_deg), float(et)
    )
    return float(np.dot(r_sun - r_target, n_target_outward)) > 0.0


def _delivery_geometry_uncached(
    r_sat_j2000_km: np.ndarray,
    et: float,
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    alt_km: float = 0.0,
    planetographic: bool = True,
    bisector_cos_alpha_min: float = 0.1,
    target_elevation_min_rad: float = 0.0,
    observer_naif_id: int = MARS_NAIF_ID,
) -> DeliveryGeometry:
    """Evaluate and return all delivery geometry at a single epoch.

    Shares a single ``spkezr`` call for the Sun across the sail-umbra,
    bisector, and target-in-sunlight gates (three SPICE queries
    collapsed to one). The outward normal at the target and the
    target's J2000 position are each computed once here rather than
    re-derived inside each helper.

    Parameters
    ----------
    r_sat_j2000_km
        Sail position in Mars-centred J2000 axes, km, shape (3,).
    et
        SPICE TDB seconds past J2000.
    target_lat_deg, target_lon_deg
        East-positive latitude / longitude of the target, degrees.
    alt_km, planetographic
        Target-position options; see ``surface.surface_point_position``.
    bisector_cos_alpha_min
        Minimum ``cos_alpha = cos(half-angle)`` for the bisector to be
        considered feasible. Default 0.1 (half-angle ≤ ~84.3°, i.e.
        the full sun-sail-target angle ψ must be ≤ ~168.5° as seen
        from the sail). The 0.1 default rejects near-antiparallel geometries
        that produce physically unrealizable peak-alpha spikes. Callers may
        explicitly pass 0.01 for a more permissive threshold.
    target_elevation_min_rad
        Minimum elevation angle of the sail above the target's
        horizon plane (radians). Default 0 = strict horizon; positive
        values implement "grazing-pass" pruning.
    observer_naif_id
        Central body NAIF id; default 499 (Mars planet centre).

    Returns
    -------
    DeliveryGeometry
        Gate booleans and the already-computed diagnostic geometry.
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)
    et_f = float(et)

    # One Sun fetch; shared across gates.
    state = sun_state_j2000(et_f, int(observer_naif_id))
    r_sun = np.asarray(state[:3], dtype=float)

    # Target position and outward normal.
    r_target = surface_point_position(
        float(target_lat_deg), float(target_lon_deg), et_f,
        alt_km=float(alt_km), planetographic=bool(planetographic),
    )
    n_target_outward = target_outward_normal_j2000(
        float(target_lat_deg), float(target_lon_deg), et_f
    )

    # Gate 1: sail sunlit (outside Mars umbra). Share pre-fetched sun.
    sail_sunlit = not in_mars_umbra(
        r_sat, et_f, int(observer_naif_id),
        sun_position_j2000_km=r_sun,
    )

    # Gate 2: sail above target's elevation threshold.
    elev = target_elevation_angle_rad(r_sat, r_target, n_target_outward)
    sail_above_horizon = elev >= float(target_elevation_min_rad)

    # Gate 3: bisector feasibility.
    _n_hat_star, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
    bisector_feasible = cos_alpha >= float(bisector_cos_alpha_min)

    # Gate 4: target sunlit (sun above target's local horizon plane).
    target_sunlit = float(np.dot(r_sun - r_target, n_target_outward)) > 0.0

    gates = DeliveryGates(
        sail_sunlit=bool(sail_sunlit),
        sail_above_target_horizon=bool(sail_above_horizon),
        bisector_feasible=bool(bisector_feasible),
        target_in_sunlight=bool(target_sunlit),
    )
    return DeliveryGeometry(
        gates=gates,
        sun_position_j2000_km=r_sun,
        target_position_j2000_km=r_target,
        target_outward_normal_j2000=n_target_outward,
        slant_range_km=slant_range_km(r_sat, r_target),
        target_elevation_rad=float(elev),
        cos_alpha=float(cos_alpha),
    )


def delivery_geometry(
    r_sat_j2000_km: np.ndarray,
    et: float,
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    alt_km: float = 0.0,
    planetographic: bool = True,
    bisector_cos_alpha_min: float = 0.1,
    target_elevation_min_rad: float = 0.0,
    observer_naif_id: int = MARS_NAIF_ID,
) -> DeliveryGeometry:
    """Return one-epoch gates and geometry with exact SPICE sharing.

    A nested-safe evaluation context collapses the target position and normal's
    identical frame requests. When called inside the propagator RHS it reuses
    that broader context, including its already-fetched Sun state.
    """
    with ephemeris_evaluation_context():
        return _delivery_geometry_uncached(
            r_sat_j2000_km,
            et,
            target_lat_deg,
            target_lon_deg,
            alt_km=alt_km,
            planetographic=planetographic,
            bisector_cos_alpha_min=bisector_cos_alpha_min,
            target_elevation_min_rad=target_elevation_min_rad,
            observer_naif_id=observer_naif_id,
        )


def delivery_gates(
    r_sat_j2000_km: np.ndarray,
    et: float,
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    alt_km: float = 0.0,
    planetographic: bool = True,
    bisector_cos_alpha_min: float = 0.1,
    target_elevation_min_rad: float = 0.0,
    observer_naif_id: int = MARS_NAIF_ID,
) -> DeliveryGates:
    """Evaluate the four delivery gates at a single epoch.

    Return the gate-only projection of :func:`delivery_geometry`. Use the full
    result when slant range, elevation, or ``cos_alpha`` is also required.
    """
    return delivery_geometry(
        r_sat_j2000_km,
        et,
        target_lat_deg,
        target_lon_deg,
        alt_km=alt_km,
        planetographic=planetographic,
        bisector_cos_alpha_min=bisector_cos_alpha_min,
        target_elevation_min_rad=target_elevation_min_rad,
        observer_naif_id=observer_naif_id,
    ).gates


# ---------------------------------------------------------------------------
# Window finder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryWindow:
    """A single interval during which the selected delivery gates are open.

    A "window" is a maximal run of consecutive propagator samples
    across which the caller-selected combined gate is continuously True
    (see ``find_delivery_windows`` for the per-gate AND / IGNORE
    switch semantics). The window's endpoints are the first and last
    sample at which the gate is True; the interior fields are
    integrated or minimised / maximised across those samples via the
    trapezoid rule (for integrals) or ``np.min``/``np.max`` (for
    extremes).

    Fields
    ------
    t_start_s, t_end_s
        Window endpoints in seconds relative to
        ``PropagationResult.epoch_et``. By convention these are the
        first and last propagator-sample times at which the gate is
        True; resolution is the underlying sample spacing.
    et_start, et_end
        Same, converted to absolute SPICE ET
        (``epoch_et + t_start_s``). ``None`` if the source
        ``PropagationResult.epoch_et`` is ``None``.
    duration_s
        ``t_end_s - t_start_s``.
    min_slant_range_km
        Minimum sail-to-target slant range observed in the window, km.
    max_elevation_deg
        Maximum elevation angle of the sail above the target's horizon
        plane, degrees.
    peak_alpha_demand_rad_s2
        Peak bisector-profile angular acceleration demand across the
        window, rad/s^2. ``None`` when no attitude profile is available.
    integral_cos_alpha_s
        Time integral of ``cos_alpha = cos(half-angle)`` across the
        window, units s (dimensionally: dimensionless * seconds).
        A "delivered-photon proxy" that ignores finite-sun beam
        spread, atmospheric extinction, and finite-sail geometry.
        For absolute-photometry analyses, use ``fluence_J_per_m2`` when a sail
        is supplied.
    peak_irradiance_W_per_m2
        Maximum mean-center irradiance at the target during the
        window, W/m^2. Populated only when ``find_delivery_windows``
        is called with a ``sail`` argument (``None`` otherwise).
        Mean-center of the reflected solar image on the surface;
        the image has uniform brightness across its interior to
        leading order (Canady & Allen 1982, Çelik & McInnes 2022).
    mean_irradiance_W_per_m2
        Time-averaged irradiance across the window, W/m^2.
        = ``fluence_J_per_m2 / duration_s``. Populated only when
        ``sail`` supplied.
    fluence_J_per_m2
        Integrated surface irradiance over the window: ``integral I(t)
        dt``, units J/m^2. Trapezoid rule across propagator samples.
        This is the absolute-photometry quantity used to compare delivery,
        whereas ``integral_cos_alpha_s`` omits beam divergence. Populated only
        when ``sail`` is supplied.
    peak_footprint_semi_major_km, peak_footprint_semi_minor_km
        Semi-major and semi-minor axes of the reflected solar-disc
        image on the target surface, in km, evaluated at the
        peak-irradiance sample of the window. Populated only when
        ``sail`` supplied. At zenith the ellipse reduces to a circle
        (semi-major == semi-minor); for oblique passes the semi-major
        scales as ``1/sin(elevation)``.
    n_samples
        Number of propagator samples inside the window, inclusive of
        both endpoints. Useful for judging whether the window's
        integrated scalars are based on enough samples to be
        meaningful.
    target_idx
        Index of the surface target this window serves, within the
        caller's target list. ``find_delivery_windows`` always emits
        0 (it evaluates one target per call);
        ``find_delivery_windows_multi`` re-stamps the index per
        target. Default 0 selects the primary target.
    target_lat_deg, target_lon_deg
        The surface target (planetographic latitude / east longitude,
        degrees) this window was evaluated against. Stamped by
        ``find_delivery_windows`` at construction; ``None`` only for
        windows constructed directly by callers using
        ``DeliveryWindow`` directly. Downstream schedule builders
        fall back to their own target arguments when these are
        ``None``.
    """

    t_start_s: float
    t_end_s: float
    et_start: Optional[float]
    et_end: Optional[float]
    duration_s: float
    min_slant_range_km: float
    max_elevation_deg: float
    peak_alpha_demand_rad_s2: Optional[float]
    integral_cos_alpha_s: float
    n_samples: int
    # Irradiance scalars (populated when find_delivery_windows is
    # called with a sail; otherwise None).
    peak_irradiance_W_per_m2: Optional[float] = None
    mean_irradiance_W_per_m2: Optional[float] = None
    fluence_J_per_m2: Optional[float] = None
    peak_footprint_semi_major_km: Optional[float] = None
    peak_footprint_semi_minor_km: Optional[float] = None
    # Identifies which target this window serves. The default is the primary
    # target in a single-target analysis.
    target_idx: int = 0
    target_lat_deg: Optional[float] = None
    target_lon_deg: Optional[float] = None


class DeliverySamples(NamedTuple):
    """Per-sample diagnostic arrays from ``find_delivery_windows``.

    Returned (alongside the window list) only when
    ``find_delivery_windows(..., return_samples=True)``. Every array has
    length ``n_samples`` (one entry per propagator sample), parallel to the
    input ``result.t_s``. These are exactly the internal arrays the window
    finder already builds per sample; exposing them lets callers plot the
    delivered-irradiance / power TIME SERIES without re-implementing the
    per-sample gate + Canady-Allen loop elsewhere (single-source).

    The irradiance is the IDEAL-BISECTOR-TRACKING delivered ground
    irradiance (``beam.delivered_surface_irradiance_W_per_m2``): the flux
    the target would receive if the sail tracked the sun-target bisector at
    that instant, independent of the propagation attitude. To recover the
    delivered power over a *kept* window ``w`` (i.e. one that survived the
    ``alpha_max`` / ``min_window_fluence`` filters), mask via
    ``(t_s >= w.t_start_s) & (t_s <= w.t_end_s)``.

    Fields
    ------
    t_s
        Sample times, seconds relative to ``result.epoch_et`` (== input
        ``result.t_s``).
    gate_open
        Combined-gate boolean per sample, PRE window-level
        ``alpha_max`` / ``min_window_fluence`` filtering. True where all
        required gates hold; a True sample may be removed with its window by
        those filters, so mask to kept windows for delivered
        power rather than using ``gate_open`` directly.
    slant_km, elev_deg, cos_alpha
        Sail->target slant range (km), elevation above the target horizon
        plane (deg), and cosine of the sun-sail-target bisector half-angle.
    irradiance_W_per_m2
        Atmosphere-attenuated direct ground irradiance (W/m^2) under bisector
        tracking; ``None`` when called without a sail.
    vacuum_irradiance_W_per_m2
        The corresponding chi=1 irradiance before atmospheric extinction.
        ``None`` when called without a sail.
    atmospheric_transmission
        Applied direct transmission per sample. ``None`` when called without
        a sail. This diagnostic makes vacuum/attenuated cache comparisons
        explicit and prevents an attenuated series from being mislabeled.
    sail_to_sun_km
        Sail->Sun distance (km); ``None`` when called without a sail.
    """

    t_s: np.ndarray
    gate_open: np.ndarray
    slant_km: np.ndarray
    elev_deg: np.ndarray
    cos_alpha: np.ndarray
    irradiance_W_per_m2: Optional[np.ndarray]
    vacuum_irradiance_W_per_m2: Optional[np.ndarray]
    atmospheric_transmission: Optional[np.ndarray]
    sail_to_sun_km: Optional[np.ndarray]


class WindowContinuationError(RuntimeError):
    """A local window search could not validate the seeded topology.

    This is deliberately distinct from an ordinary input ``ValueError``:
    callers may respond by running the full-window finder, while malformed
    inputs should still fail immediately.
    """


@dataclass(frozen=True)
class ContinuedDeliveryWindows:
    """Validated result of a guarded local search around seeded windows."""

    windows: Tuple[DeliveryWindow, ...]
    search_intervals_by_target_s: Tuple[Tuple[int, float, float], ...]
    n_sample_target_evaluations: int
    n_samples_per_target: int
    max_boundary_shift_s: float


def trajectory_interpolant(result) -> Optional[CubicSpline]:
    """Cubic-spline interpolant of ``result.state_km_kmps[:, :3]`` vs absolute ET.

    Returns a ``scipy.interpolate.CubicSpline`` such that
    ``cs(et)`` yields the sail position vector (J2000, km) at
    absolute SPICE TDB seconds ``et``, or ``None`` if the result
    does not have a known ``epoch_et`` or has fewer than
    ``_SLEW_MIN_SAMPLES = 4`` samples (the cubic spline's minimum).

    Shared by ``find_delivery_windows`` (peak-alpha-across-window
    diagnostic) and ``attitude_schedule.refine_delivery_schedule``
    (fixed-point iteration, which feeds the interpolant into the next
    schedule-build step as ``r_sat_predicted_fn``).
    """
    t_arr = np.asarray(result.t_s, dtype=float)
    state_arr = np.asarray(result.state_km_kmps, dtype=float)
    epoch_et = result.epoch_et
    if epoch_et is None or t_arr.shape[0] < _SLEW_MIN_SAMPLES:
        return None
    return CubicSpline(t_arr + epoch_et, state_arr[:, :3], axis=0)


def _peak_alpha_across_window(
    trajectory_interpolant: CubicSpline,
    et_start: float,
    et_end: float,
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    alt_km: float = 0.0,
    planetographic: bool = True,
    observer_naif_id: int = MARS_NAIF_ID,
    dt_s: float = _SLEW_DEFAULT_DT_S,
    n_subgrid: int = _SLEW_DEFAULT_SUBGRID_POINTS,
) -> Optional[float]:
    """Peak magnitude of the bisector-pointing profile's angular
    acceleration across ``[et_start, et_end]``, rad/s^2.

    Strategy: use the caller-provided CubicSpline of the sail
    trajectory (fit once over the full propagation in
    ``find_delivery_windows``) as the ``r_sat_fn`` argument to
    ``attitude.angular_acceleration``. Sample ``n_subgrid`` points
    uniformly across the window, INSET by ``dt_s`` on each side so the
    central-difference stencil ``[et - dt, et, et + dt]`` never
    queries outside the interpolant. Evaluate the bisector-pointing
    profile at each; take the max Euclidean norm of alpha.

    Returns ``None`` if the window is too short for FD (needs
    ``t_end - t_start > 2 * dt_s``).
    """
    if et_end - et_start <= 2.0 * dt_s:
        return None
    profile = bisector_pointing(
        float(target_lat_deg), float(target_lon_deg),
        alt_km=float(alt_km), planetographic=bool(planetographic),
        observer_naif_id=int(observer_naif_id),
    )

    def r_sat_fn(et: float) -> np.ndarray:
        return np.asarray(trajectory_interpolant(float(et)), dtype=float)

    subgrid = np.linspace(et_start + dt_s, et_end - dt_s, int(n_subgrid))
    peak = 0.0
    for et_pt in subgrid:
        alpha = angular_acceleration(
            profile, r_sat_fn, float(et_pt), dt=float(dt_s)
        )
        mag = float(np.linalg.norm(alpha))
        if mag > peak:
            peak = mag
    return peak


def _combined_gate(
    g: DeliveryGates,
    *,
    require_sail_sunlit: Optional[bool],
    require_sail_above_horizon: Optional[bool],
    require_bisector_feasible: Optional[bool],
    require_target_sunlit: Optional[bool],
) -> bool:
    """Combine per-gate AND / IGNORE switches into a single decision.

    Each ``require_*`` switch can be:
        True  -- the corresponding gate must be open (True)
        False -- the corresponding gate must be closed (False)
        None  -- the gate is ignored (do not filter on it)

    Returns True iff every non-None switch matches the corresponding
    field of ``g``.
    """
    if require_sail_sunlit is not None and g.sail_sunlit != require_sail_sunlit:
        return False
    if (
        require_sail_above_horizon is not None
        and g.sail_above_target_horizon != require_sail_above_horizon
    ):
        return False
    if (
        require_bisector_feasible is not None
        and g.bisector_feasible != require_bisector_feasible
    ):
        return False
    if (
        require_target_sunlit is not None
        and g.target_in_sunlight != require_target_sunlit
    ):
        return False
    return True


def _bisect_boolean_transition(
    predicate: Callable[[float], bool],
    t_left_s: float,
    t_right_s: float,
    *,
    left_value: bool,
    right_value: bool,
    tolerance_s: float,
) -> float:
    """Locate one Boolean transition inside a sampled time bracket.

    This is ordinary interval bisection applied to a predicate rather than a
    signed scalar.  It is appropriate for the combined delivery gate because
    the sampled finder has already established opposite Boolean values at the
    adjacent bracket endpoints.  The caller is responsible for choosing a
    cadence fine enough that the bracket does not hide multiple transitions.

    The returned midpoint is within ``tolerance_s / 2`` of the final bracket
    and therefore within ``tolerance_s`` of the transition.  Both transition
    directions (closed-to-open and open-to-closed) are supported.
    """
    left_s = float(t_left_s)
    right_s = float(t_right_s)
    tol_s = float(tolerance_s)
    if not (math.isfinite(left_s) and math.isfinite(right_s)):
        raise ValueError("Boolean-transition bracket endpoints must be finite")
    if right_s <= left_s:
        raise ValueError(
            "Boolean-transition bracket must satisfy t_right_s > t_left_s"
        )
    if not math.isfinite(tol_s) or tol_s <= 0.0:
        raise ValueError(
            "Boolean-transition tolerance_s must be positive and finite"
        )
    left_bool = bool(left_value)
    right_bool = bool(right_value)
    if left_bool == right_bool:
        raise ValueError("Boolean-transition bracket endpoint values must differ")
    if bool(predicate(left_s)) != left_bool:
        raise ValueError("predicate(t_left_s) does not match left_value")
    if bool(predicate(right_s)) != right_bool:
        raise ValueError("predicate(t_right_s) does not match right_value")

    while right_s - left_s > tol_s:
        midpoint_s = 0.5 * (left_s + right_s)
        midpoint_value = bool(predicate(midpoint_s))
        if midpoint_value == left_bool:
            left_s = midpoint_s
        else:
            right_s = midpoint_s
    return 0.5 * (left_s + right_s)


def find_delivery_windows(
    result,  # PropagationResult; duck-typed to avoid the import.
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    target_elevation_min_deg: float = 10.0,
    bisector_cos_alpha_min: float = 0.1,
    alt_km: float = 0.0,
    planetographic: bool = True,
    observer_naif_id: int = MARS_NAIF_ID,
    require_sail_sunlit: Optional[bool] = True,
    require_sail_above_horizon: Optional[bool] = True,
    require_bisector_feasible: Optional[bool] = True,
    require_target_sunlit: Optional[bool] = None,
    sail: Optional[SolarSail] = None,
    atmospheric_transmission: Union[float, Sequence[float], np.ndarray] = 1.0,
    alpha_max_rad_s2: Optional[float] = None,
    min_window_fluence_J_per_m2: Optional[float] = None,
    boundary_refinement_tol_s: Optional[float] = None,
    return_samples: bool = False,
    search_intervals_s: Optional[Sequence[Tuple[float, float]]] = None,
) -> Union[List[DeliveryWindow], Tuple[List[DeliveryWindow], DeliverySamples]]:
    """Walk a propagation result and enumerate delivery windows.

    For each sample in the ``PropagationResult``, evaluate
    ``delivery_gates`` at the corresponding (sail position, epoch);
    combine the four booleans using the per-gate require switches;
    group consecutive True samples into windows; compute integrated
    quality scalars per window. By default, window endpoints are the first and
    last True-sample times. When
    ``boundary_refinement_tol_s`` is supplied, each sampled closed/open
    transition is bracketed by its adjacent samples and the exact combined
    gate is bisected on the cubic-spline trajectory to the requested time
    tolerance.

    Parameters
    ----------
    result
        A ``reflectors.dynamics.PropagationResult``. Must carry
        ``t_s`` (shape (N,)), ``state_km_kmps`` (shape (N, 6)), and
        ``epoch_et`` (absolute TDB seconds at ``t_s[0]``; may be
        ``None``, in which case ``et_start`` / ``et_end`` will be
        ``None`` on each returned window).
    target_lat_deg, target_lon_deg
        East-positive latitude / longitude of the target, degrees.
    target_elevation_min_deg
        Minimum elevation angle above the target's horizon plane
        (degrees) for the sail_above_target_horizon gate to count as
        open. Default 10 deg (prunes grazing passes; physically
        relevant for excluding grazing passes). Converted internally to radians
        and passed to ``delivery_gates``.
    bisector_cos_alpha_min
        Minimum cos(half-angle) for bisector_feasible. Default 0.1
        (half-angle ≤ ~84.3°, ψ ≤ ~168.5°); see ``delivery_gates`` for the
        rationale.
    alt_km, planetographic
        Target-position options (see ``surface.surface_point_position``).
    observer_naif_id
        Central body NAIF id.
    require_sail_sunlit, require_sail_above_horizon,
    require_bisector_feasible, require_target_sunlit
        Per-gate switches: True requires open, False requires closed,
        None ignores. Defaults match the "total energy delivered"
        objective: gate on sail sunlit + sail
        above horizon + bisector feasible; ignore target sunlight.
        For a "night-only illumination" objective,
        ``require_target_sunlit=False``; for "day-only augmentation",
        ``require_target_sunlit=True``.
    sail
        Optional ``reflectors.srp.SolarSail`` bus. When supplied, each
        returned ``DeliveryWindow`` populates the five absolute-photometry
        fields (``peak_irradiance_W_per_m2``, ``mean_irradiance_W_per_m2``,
        ``fluence_J_per_m2``, ``peak_footprint_semi_major_km``,
        ``peak_footprint_semi_minor_km``) using the reflected-beam
        formula in ``reflectors.beam`` (Canady-Allen 1982 Eq. 9;
        Çelik-McInnes 2022 Eq. 13+16; McInnes 1999 §2.6.1 specular
        decomposition via ``sail.optical.rho * sail.optical.s``).
        Default ``None`` -> those fields stay ``None`` and the
        integrated ``cos alpha`` proxy remains the only delivery scalar.
    atmospheric_transmission
        Direct-beam transmission ``chi`` multiplying delivered irradiance
        (see ``reflectors.beam``). May be a scalar or a one-dimensional array
        exactly aligned with ``result.t_s``. Every value must be finite and in
        ``[0, 1]``. The vector path supports elevation/time-dependent Mars
        extinction while keeping the atmospheric calculation outside this
        geometry module. Peak, mean, fluence, and the minimum-fluence filter
        all use the attenuated series. Default 1.0 is vacuum/no atmosphere.
    alpha_max_rad_s2
        Optional attitude-feasibility cap on each window's
        ``peak_alpha_demand_rad_s2``. When set, any window whose peak
        exceeds the cap is dropped from the returned list (post-hoc
        window filter that applies the kinematic ``alpha_max`` path constraint
        as a window-level feasibility gate).
        Windows with ``peak_alpha_demand_rad_s2 is None`` (uncomputable:
        missing epoch, sparse trajectory, or window shorter than the FD
        stencil) are conservatively dropped because unverifiable windows
        cannot be classified as feasible. Default ``None`` disables the filter.

        The cap is a kinematic upper bound on the bisector profile's angular
        acceleration; the
        sail's actuator must deliver this ``alpha`` if it is to track
        the specular bisector during the window. Recommended values are
        within an order of magnitude of the Viale et al. 2023 Earth-LMO
        worst-case tracking pass (~4.86e-5 rad/s^2 = ~2.79e-3 deg/s^2);
        a sensible default for Mars conceptual feasibility is
        ``math.radians(0.003)`` = 5.24e-5 rad/s^2.
    min_window_fluence_J_per_m2
        Optional window-value filter. When set, windows whose
        ``fluence_J_per_m2``
        falls below the threshold are dropped from the returned list,
        so downstream schedulers never spend slew budget on them and
        kept-only accounting never credits them. Requires ``sail``
        (fluence is only computed when a sail is supplied); raises
        ``ValueError`` otherwise. Default ``None`` disables the filter
        and preserves unfiltered behavior.
    boundary_refinement_tol_s
        Optional positive time tolerance for sub-sample gate-boundary
        refinement, seconds. The sampled pass topology is retained, but each
        adjacent false/true or true/false bracket is bisected using the same
        complete delivery gate evaluated on the cubic-spline trajectory.
        This removes first/last-True-sample snapping from downstream slew
        schedules. It requires a known ``result.epoch_et`` and at least four
        trajectory samples. ``None`` uses sampled endpoints.
    return_samples
        When True, also return the per-sample diagnostic arrays as a
        :class:`DeliverySamples` (the internal gate / slant / elevation /
        cos_alpha / delivered-irradiance arrays the finder already builds),
        so callers can plot the delivered-power TIME SERIES without
        duplicating the per-sample loop. Default False uses the single-value
        return type; the window list and
        its scalars are identical whether or not samples are requested.
    search_intervals_s
        Optional guarded intervals, in seconds relative to ``result.epoch_et``,
        within which to evaluate the delivery geometry. Samples outside the
        union of these intervals are treated as gate-closed and diagnostic
        arrays contain NaN there. This is an acceleration hook for
        :func:`continue_delivery_windows_multi`; ordinary callers should leave
        it as ``None`` so the complete propagation is searched. Supplying
        intervals asserts only where to look, not that the resulting topology
        is complete; the continuation wrapper performs the required guard and
        count validation.

    Returns
    -------
    list of DeliveryWindow
        Ordered by ``t_start_s``. Empty list if the combined gate is
        never True over the propagation.
    tuple(list of DeliveryWindow, DeliverySamples)
        Only when ``return_samples=True``: the same window list plus the
        per-sample arrays (length ``n_samples``).
    """
    t_arr = np.asarray(result.t_s, dtype=float)
    state_arr = np.asarray(result.state_km_kmps, dtype=float)
    epoch_et = result.epoch_et  # may be None

    boundary_tol_s: Optional[float]
    if boundary_refinement_tol_s is None:
        boundary_tol_s = None
    else:
        boundary_tol_s = float(boundary_refinement_tol_s)
        if not math.isfinite(boundary_tol_s) or boundary_tol_s <= 0.0:
            raise ValueError(
                "boundary_refinement_tol_s must be a positive finite float"
            )

    if t_arr.shape[0] != state_arr.shape[0]:
        raise ValueError(
            f"find_delivery_windows: t_s length {t_arr.shape[0]} does not "
            f"match state_km_kmps length {state_arr.shape[0]}"
        )

    if search_intervals_s is None:
        evaluation_mask = np.ones(t_arr.shape, dtype=bool)
    else:
        evaluation_mask = np.zeros(t_arr.shape, dtype=bool)
        for interval in search_intervals_s:
            if len(interval) != 2:
                raise ValueError(
                    "each search_intervals_s entry must be a (start_s, end_s) pair"
                )
            start_s, end_s = float(interval[0]), float(interval[1])
            if not (math.isfinite(start_s) and math.isfinite(end_s)):
                raise ValueError("search_intervals_s endpoints must be finite")
            if end_s < start_s:
                raise ValueError(
                    "search_intervals_s endpoints must satisfy end_s >= start_s"
                )
            evaluation_mask |= (t_arr >= start_s) & (t_arr <= end_s)

    transmission_scalar = 1.0
    transmission_by_sample: Optional[np.ndarray] = None
    if sail is not None:
        transmission_input = np.asarray(atmospheric_transmission, dtype=float)
        if transmission_input.ndim == 0:
            transmission_scalar = float(transmission_input)
            if not math.isfinite(transmission_scalar) or not (
                0.0 <= transmission_scalar <= 1.0
            ):
                raise ValueError(
                    "atmospheric_transmission scalar must be finite and in [0, 1]"
                )
        elif transmission_input.ndim == 1 and transmission_input.shape == t_arr.shape:
            if not np.all(np.isfinite(transmission_input)):
                raise ValueError(
                    "atmospheric_transmission array must contain only finite values"
                )
            if np.any((transmission_input < 0.0) | (transmission_input > 1.0)):
                raise ValueError(
                    "atmospheric_transmission array values must lie in [0, 1]"
                )
            transmission_by_sample = transmission_input
        else:
            raise ValueError(
                "atmospheric_transmission must be a scalar or a 1-D array "
                "matching result.t_s"
            )

    elev_min_rad = float(np.radians(target_elevation_min_deg))

    n_samples = t_arr.shape[0]
    if n_samples == 0:
        if return_samples:
            _empty = np.empty(0, dtype=float)
            return [], DeliverySamples(
                t_s=_empty,
                gate_open=np.empty(0, dtype=bool),
                slant_km=_empty,
                elev_deg=_empty,
                cos_alpha=_empty,
                irradiance_W_per_m2=(_empty if sail is not None else None),
                vacuum_irradiance_W_per_m2=(
                    _empty if sail is not None else None
                ),
                atmospheric_transmission=(_empty if sail is not None else None),
                sail_to_sun_km=(_empty if sail is not None else None),
            )
        return []

    # One-shot cubic spline of the sail trajectory for peak-alpha evaluation
    # (via trajectory_interpolant helper,
    # shared with attitude_schedule.refine_delivery_schedule).
    trajectory_cs = trajectory_interpolant(result)
    if boundary_tol_s is not None and trajectory_cs is None:
        raise ValueError(
            "boundary_refinement_tol_s requires a known result.epoch_et "
            "and at least four trajectory samples"
        )

    def _geometry_at_relative_time(t_s: float) -> DeliveryGeometry:
        if trajectory_cs is None or epoch_et is None:
            raise RuntimeError(
                "sub-sample delivery geometry requires a trajectory interpolant"
            )
        et = float(epoch_et) + float(t_s)
        return delivery_geometry(
            np.asarray(trajectory_cs(et), dtype=float),
            et,
            float(target_lat_deg),
            float(target_lon_deg),
            alt_km=float(alt_km),
            planetographic=bool(planetographic),
            bisector_cos_alpha_min=float(bisector_cos_alpha_min),
            target_elevation_min_rad=elev_min_rad,
            observer_naif_id=int(observer_naif_id),
        )

    def _combined_gate_at_relative_time(t_s: float) -> bool:
        return _combined_gate(
            _geometry_at_relative_time(t_s).gates,
            require_sail_sunlit=require_sail_sunlit,
            require_sail_above_horizon=require_sail_above_horizon,
            require_bisector_feasible=require_bisector_feasible,
            require_target_sunlit=require_target_sunlit,
        )

    # Per-sample evaluation. Build the combined-gate boolean array and
    # parallel arrays for the scalars integrated / extremised
    # per window.
    gate_open = np.zeros(n_samples, dtype=bool)
    partial_search = search_intervals_s is not None
    array_fill = np.nan if partial_search else None
    slant_km = (
        np.full(n_samples, array_fill, dtype=float)
        if partial_search else np.empty(n_samples, dtype=float)
    )
    elev_deg = (
        np.full(n_samples, array_fill, dtype=float)
        if partial_search else np.empty(n_samples, dtype=float)
    )
    cos_alpha_arr = (
        np.full(n_samples, array_fill, dtype=float)
        if partial_search else np.empty(n_samples, dtype=float)
    )
    # Irradiance arrays are allocated only when a sail is supplied.
    if sail is not None:
        if partial_search:
            irradiance_W_per_m2 = np.full(n_samples, np.nan, dtype=float)
            vacuum_irradiance_W_per_m2 = np.full(n_samples, np.nan, dtype=float)
            atmospheric_transmission_arr = np.full(n_samples, np.nan, dtype=float)
            sail_to_sun_km_arr = np.full(n_samples, np.nan, dtype=float)
        else:
            irradiance_W_per_m2 = np.empty(n_samples, dtype=float)
            vacuum_irradiance_W_per_m2 = np.empty(n_samples, dtype=float)
            atmospheric_transmission_arr = np.empty(n_samples, dtype=float)
            sail_to_sun_km_arr = np.empty(n_samples, dtype=float)
    else:
        irradiance_W_per_m2 = None
        vacuum_irradiance_W_per_m2 = None
        atmospheric_transmission_arr = None
        sail_to_sun_km_arr = None

    for i in np.flatnonzero(evaluation_mask):
        et_i = (epoch_et + t_arr[i]) if epoch_et is not None else float(t_arr[i])
        r_sat_i = state_arr[i, :3]
        geometry_i = delivery_geometry(
            r_sat_i, et_i,
            float(target_lat_deg), float(target_lon_deg),
            alt_km=float(alt_km),
            planetographic=bool(planetographic),
            bisector_cos_alpha_min=float(bisector_cos_alpha_min),
            target_elevation_min_rad=elev_min_rad,
            observer_naif_id=int(observer_naif_id),
        )
        g_i = geometry_i.gates
        gate_open[i] = _combined_gate(
            g_i,
            require_sail_sunlit=require_sail_sunlit,
            require_sail_above_horizon=require_sail_above_horizon,
            require_bisector_feasible=require_bisector_feasible,
            require_target_sunlit=require_target_sunlit,
        )
        # Reuse the exact values from the gate calculation.
        slant_km[i] = geometry_i.slant_range_km
        if slant_km[i] == 0.0:
            elev_deg[i] = 0.0
        else:
            elev_deg[i] = math.degrees(geometry_i.target_elevation_rad)
        r_sun_i = geometry_i.sun_position_j2000_km
        cos_alpha_i = geometry_i.cos_alpha
        cos_alpha_arr[i] = cos_alpha_i

        # Per-sample delivered surface irradiance (Canady-Allen 1982
        # Eq. 9 via reflectors.beam). Only computed when sail was
        # supplied. No additional SPICE calls are needed; r_sun, r_sat,
        # slant, cos_alpha, and elevation are already available.
        if irradiance_W_per_m2 is not None:
            sail_to_sun_i_km = float(
                np.linalg.norm(r_sun_i - np.asarray(r_sat_i, dtype=float))
            )
            sail_to_sun_km_arr[i] = sail_to_sun_i_km
            sin_el_i = math.sin(math.radians(elev_deg[i]))
            vacuum_irradiance = delivered_surface_irradiance_W_per_m2(
                sail,
                slant_km[i],
                sail_to_sun_i_km,
                cos_alpha_i,
                sin_el_i,
                atmospheric_transmission=1.0,
            )
            chi_i = (
                transmission_scalar
                if transmission_by_sample is None
                else float(transmission_by_sample[i])
            )
            vacuum_irradiance_W_per_m2[i] = vacuum_irradiance
            atmospheric_transmission_arr[i] = chi_i
            irradiance_W_per_m2[i] = vacuum_irradiance * chi_i

    # Find maximal runs of gate_open = True.
    windows: List[DeliveryWindow] = []
    i = 0
    while i < n_samples:
        if not gate_open[i]:
            i += 1
            continue
        j = i
        while j + 1 < n_samples and gate_open[j + 1]:
            j += 1
        # Window covers sampled open points [i, j] inclusive. Optionally move
        # each endpoint into the adjacent closed/open transition bracket.
        t_start = float(t_arr[i])
        t_end = float(t_arr[j])
        start_geometry: Optional[DeliveryGeometry] = None
        end_geometry: Optional[DeliveryGeometry] = None
        if (
            boundary_tol_s is not None
            and i > 0
            and bool(evaluation_mask[i - 1])
            and not bool(gate_open[i - 1])
        ):
            t_start = _bisect_boolean_transition(
                _combined_gate_at_relative_time,
                float(t_arr[i - 1]),
                float(t_arr[i]),
                left_value=False,
                right_value=True,
                tolerance_s=boundary_tol_s,
            )
            start_geometry = _geometry_at_relative_time(t_start)
        if (
            boundary_tol_s is not None
            and j + 1 < n_samples
            and bool(evaluation_mask[j + 1])
            and not bool(gate_open[j + 1])
        ):
            t_end = _bisect_boolean_transition(
                _combined_gate_at_relative_time,
                float(t_arr[j]),
                float(t_arr[j + 1]),
                left_value=True,
                right_value=False,
                tolerance_s=boundary_tol_s,
            )
            end_geometry = _geometry_at_relative_time(t_end)
        if epoch_et is not None:
            et_start_w: Optional[float] = float(epoch_et + t_start)
            et_end_w: Optional[float] = float(epoch_et + t_end)
        else:
            et_start_w = None
            et_end_w = None
        # Integrated / extremised scalars over the sampled open points plus
        # any refined boundary points. Boundary values are continuous
        # geometry evaluations; they are not added to ``n_samples``, whose
        # documented meaning remains the number of propagator samples.
        t_values = [float(value) for value in t_arr[i : j + 1]]
        slant_values = [float(value) for value in slant_km[i : j + 1]]
        elevation_values = [float(value) for value in elev_deg[i : j + 1]]
        cos_alpha_values = [float(value) for value in cos_alpha_arr[i : j + 1]]

        # The Sun-distance component depends on the boundary sail position;
        # evaluate it inline rather than through the sample arrays.
        boundary_sun_distances: dict[str, float] = {}
        if start_geometry is not None:
            start_et = float(epoch_et) + t_start
            start_position = np.asarray(trajectory_cs(start_et), dtype=float)
            t_values.insert(0, t_start)
            slant_values.insert(0, float(start_geometry.slant_range_km))
            elevation_values.insert(
                0, math.degrees(float(start_geometry.target_elevation_rad))
            )
            cos_alpha_values.insert(0, float(start_geometry.cos_alpha))
            boundary_sun_distances["start"] = float(
                np.linalg.norm(start_geometry.sun_position_j2000_km - start_position)
            )
        if end_geometry is not None:
            end_et = float(epoch_et) + t_end
            end_position = np.asarray(trajectory_cs(end_et), dtype=float)
            t_values.append(t_end)
            slant_values.append(float(end_geometry.slant_range_km))
            elevation_values.append(
                math.degrees(float(end_geometry.target_elevation_rad))
            )
            cos_alpha_values.append(float(end_geometry.cos_alpha))
            boundary_sun_distances["end"] = float(
                np.linalg.norm(end_geometry.sun_position_j2000_km - end_position)
            )

        t_slice = np.asarray(t_values, dtype=float)
        slant_slice = np.asarray(slant_values, dtype=float)
        elevation_slice = np.asarray(elevation_values, dtype=float)
        cos_alpha_slice = np.asarray(cos_alpha_values, dtype=float)
        min_slant = float(np.min(slant_slice))
        max_elev = float(np.max(elevation_slice))
        # Trapezoid integration; handles the single-sample case (returns 0).
        if t_slice.size > 1:
            integral_cos_alpha = float(
                np.trapezoid(cos_alpha_slice, t_slice)
            )
        else:
            integral_cos_alpha = 0.0
        # Peak slew-rate demand for the bisector-pointing profile,
        # evaluated against the cubic-spline interpolant of the whole
        # trajectory. None when no interpolant is available
        # (epoch_et unknown or too few samples) or when the window is
        # shorter than the finite-difference stencil can handle.
        if trajectory_cs is not None and et_start_w is not None \
                and et_end_w is not None and (j - i + 1) >= _SLEW_MIN_SAMPLES:
            peak_alpha = _peak_alpha_across_window(
                trajectory_cs,
                et_start_w,
                et_end_w,
                float(target_lat_deg),
                float(target_lon_deg),
                alt_km=float(alt_km),
                planetographic=bool(planetographic),
                observer_naif_id=int(observer_naif_id),
            )
        else:
            peak_alpha = None

        # Per-window irradiance scalars (Canady-Allen 1982 Eq. 9 via
        # reflectors.beam) and footprint ellipse axes at the
        # peak-irradiance sample. Populated only when a sail was
        # supplied to find_delivery_windows; None otherwise.
        if irradiance_W_per_m2 is not None:
            irradiance_values = [
                float(value) for value in irradiance_W_per_m2[i : j + 1]
            ]
            sail_to_sun_values = [
                float(value) for value in sail_to_sun_km_arr[i : j + 1]
            ]

            def _boundary_irradiance(
                geometry: DeliveryGeometry,
                t_s: float,
                sun_distance_km: float,
            ) -> float:
                elevation_deg = math.degrees(
                    float(geometry.target_elevation_rad)
                )
                vacuum = delivered_surface_irradiance_W_per_m2(
                    sail,
                    float(geometry.slant_range_km),
                    sun_distance_km,
                    float(geometry.cos_alpha),
                    math.sin(math.radians(elevation_deg)),
                    atmospheric_transmission=1.0,
                )
                transmission = (
                    transmission_scalar
                    if transmission_by_sample is None
                    else float(np.interp(t_s, t_arr, transmission_by_sample))
                )
                return float(vacuum * transmission)

            if start_geometry is not None:
                start_sun_distance = boundary_sun_distances["start"]
                irradiance_values.insert(
                    0,
                    _boundary_irradiance(
                        start_geometry, t_start, start_sun_distance
                    ),
                )
                sail_to_sun_values.insert(0, start_sun_distance)
            if end_geometry is not None:
                end_sun_distance = boundary_sun_distances["end"]
                irradiance_values.append(
                    _boundary_irradiance(end_geometry, t_end, end_sun_distance)
                )
                sail_to_sun_values.append(end_sun_distance)

            window_irr = np.asarray(irradiance_values, dtype=float)
            sail_to_sun_slice = np.asarray(sail_to_sun_values, dtype=float)
            peak_i_local = int(np.argmax(window_irr))
            peak_irradiance = float(window_irr[peak_i_local])
            # Trapezoid fluence (J/m^2) across the window.
            if t_slice.size > 1:
                fluence = float(np.trapezoid(window_irr, t_slice))
            else:
                fluence = 0.0
            duration = t_end - t_start
            if duration > 0.0:
                mean_irradiance: Optional[float] = fluence / duration
            else:
                mean_irradiance = None
            # Footprint semi-axes at the peak-irradiance sample.
            # sin(elevation) must be > 0 for the beam to reach the
            # ground; if the peak happens at elev = 0 (shouldn't occur
            # inside an open window with elevation gate on, but guard
            # anyway), skip the footprint computation.
            sin_el_peak = math.sin(math.radians(elevation_slice[peak_i_local]))
            if sin_el_peak > 0.0 and cos_alpha_slice[peak_i_local] > 0.0:
                a_km_peak, b_km_peak = beam_footprint_semi_axes_km(
                    slant_slice[peak_i_local],
                    sail_to_sun_slice[peak_i_local],
                    sin_el_peak,
                )
                peak_semi_major_km: Optional[float] = a_km_peak
                peak_semi_minor_km: Optional[float] = b_km_peak
            else:
                peak_semi_major_km = None
                peak_semi_minor_km = None
        else:
            peak_irradiance = None
            mean_irradiance = None
            fluence = None
            peak_semi_major_km = None
            peak_semi_minor_km = None

        windows.append(
            DeliveryWindow(
                t_start_s=t_start,
                t_end_s=t_end,
                et_start=et_start_w,
                et_end=et_end_w,
                duration_s=t_end - t_start,
                min_slant_range_km=min_slant,
                max_elevation_deg=max_elev,
                peak_alpha_demand_rad_s2=peak_alpha,
                integral_cos_alpha_s=integral_cos_alpha,
                n_samples=j - i + 1,
                peak_irradiance_W_per_m2=peak_irradiance,
                mean_irradiance_W_per_m2=mean_irradiance,
                fluence_J_per_m2=fluence,
                peak_footprint_semi_major_km=peak_semi_major_km,
                peak_footprint_semi_minor_km=peak_semi_minor_km,
                target_lat_deg=float(target_lat_deg),
                target_lon_deg=float(target_lon_deg),
            )
        )
        i = j + 1

    n_windows_pre_filter = len(windows)
    if alpha_max_rad_s2 is not None:
        if not math.isfinite(alpha_max_rad_s2) or alpha_max_rad_s2 < 0.0:
            raise ValueError(
                f"alpha_max_rad_s2 must be a non-negative finite float; "
                f"got {alpha_max_rad_s2!r}"
            )
        windows = [
            w for w in windows
            if w.peak_alpha_demand_rad_s2 is not None
            and w.peak_alpha_demand_rad_s2 <= alpha_max_rad_s2
        ]

    if min_window_fluence_J_per_m2 is not None:
        if sail is None:
            raise ValueError(
                "min_window_fluence_J_per_m2 requires a sail: fluence "
                "is only computed when a sail is supplied to "
                "find_delivery_windows."
            )
        if (not math.isfinite(min_window_fluence_J_per_m2)
                or min_window_fluence_J_per_m2 < 0.0):
            raise ValueError(
                f"min_window_fluence_J_per_m2 must be a non-negative "
                f"finite float; got {min_window_fluence_J_per_m2!r}"
            )
        n_before_fluence_filter = len(windows)
        windows = [
            w for w in windows
            if w.fluence_J_per_m2 is not None
            and w.fluence_J_per_m2 >= min_window_fluence_J_per_m2
        ]
        if len(windows) < n_before_fluence_filter:
            logger.info(
                "find_delivery_windows: min_window_fluence filter "
                "(%.3f J/m^2) dropped %d window(s).",
                min_window_fluence_J_per_m2,
                n_before_fluence_filter - len(windows),
            )

    logger.info(
        "find_delivery_windows: %d/%d evaluated sample(s), %d window(s) at "
        "(lat=%.3f, lon=%.3f deg), elev_min=%.2f deg, require=[sunlit=%s, "
        "horiz=%s, bisector=%s, target_sun=%s], alpha_max_rad_s2=%s "
        "boundary_refinement_tol_s=%s (pre-filter %d, post-filter %d)",
        int(np.count_nonzero(evaluation_mask)), n_samples, len(windows),
        target_lat_deg, target_lon_deg,
        target_elevation_min_deg,
        require_sail_sunlit, require_sail_above_horizon,
        require_bisector_feasible, require_target_sunlit,
        alpha_max_rad_s2, boundary_tol_s, n_windows_pre_filter, len(windows),
    )
    if return_samples:
        return windows, DeliverySamples(
            t_s=t_arr,
            gate_open=gate_open,
            slant_km=slant_km,
            elev_deg=elev_deg,
            cos_alpha=cos_alpha_arr,
            irradiance_W_per_m2=irradiance_W_per_m2,
            vacuum_irradiance_W_per_m2=vacuum_irradiance_W_per_m2,
            atmospheric_transmission=atmospheric_transmission_arr,
            sail_to_sun_km=sail_to_sun_km_arr,
        )
    return windows


def find_delivery_windows_multi(
    result,
    targets: Sequence[Tuple[float, float]],
    **kwargs,
) -> List[DeliveryWindow]:
    """Find delivery windows for SEVERAL surface targets on one trajectory.

    Thin multi-target wrapper around ``find_delivery_windows``: runs
    the single-target finder once per ``(lat_deg, lon_deg)`` pair in
    ``targets``, stamps each window's ``target_idx`` with its position
    in the target list, merges the per-target lists, and returns them
    sorted by ``(t_start_s, target_idx)`` (the index tie-break makes
    the ordering deterministic if two targets ever opened windows at
    the same sample, which is geometrically impossible for targets
    farther apart than twice the visibility-cone radius).

    With a single target, the output is exactly the single-target finder's
    output (the same objects, without re-stamping).

    Parameters
    ----------
    result
        ``PropagationResult`` (same contract as
        ``find_delivery_windows``).
    targets
        Sequence of ``(target_lat_deg, target_lon_deg)`` pairs,
        planetographic latitude / east longitude in degrees. Must be
        non-empty.
    **kwargs
        Forwarded verbatim to every ``find_delivery_windows`` call
        (gates, elevation minimum, sail, alpha_max post-filter, ...).
    """
    if len(targets) == 0:
        raise ValueError("targets must be a non-empty sequence of "
                         "(lat_deg, lon_deg) pairs")
    windows: List[DeliveryWindow] = []
    for idx, (lat_deg, lon_deg) in enumerate(targets):
        found = find_delivery_windows(
            result, float(lat_deg), float(lon_deg), **kwargs,
        )
        if idx > 0:
            found = [replace(w, target_idx=idx) for w in found]
        windows.extend(found)
    windows.sort(key=lambda w: (w.t_start_s, w.target_idx))
    return windows


def _merge_search_intervals(
    intervals_s: Sequence[Tuple[float, float]],
) -> list[Tuple[float, float]]:
    """Return sorted, overlapping/touching intervals as disjoint unions."""
    if not intervals_s:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals_s)
    merged: list[Tuple[float, float]] = [ordered[0]]
    for start_s, end_s in ordered[1:]:
        old_start_s, old_end_s = merged[-1]
        if start_s <= old_end_s:
            merged[-1] = (old_start_s, max(old_end_s, end_s))
        else:
            merged.append((start_s, end_s))
    return merged


def continue_delivery_windows_multi(
    result,
    targets: Sequence[Tuple[float, float]],
    seed_windows: Sequence[DeliveryWindow],
    *,
    search_margin_s: float = 900.0,
    max_boundary_shift_s: Optional[float] = None,
    **kwargs,
) -> ContinuedDeliveryWindows:
    """Update a known window topology using guarded local gate searches.

    This routine is intended for consecutive repeats of a repeat-ground-track
    orbit. It evaluates the *same* :func:`find_delivery_windows` physics and
    window metrics as a global search, but only in time bands around the prior
    sol's windows. A band must be gate-closed at both sampled edges, the number
    of post-filter windows must be unchanged for every target, and each
    boundary must remain within ``max_boundary_shift_s`` of its seed. Failure
    raises :class:`WindowContinuationError`; callers can then run the global
    finder and record the topology change.

    The method necessarily assumes that a wholly new, disconnected window
    cannot appear outside all guarded bands. That assumption is appropriate
    only when the orbit design supplies the repeated topology (as in a
    repeat-ground-track, sun-synchronous chain); it is not a general-purpose
    replacement for :func:`find_delivery_windows_multi`.

    ``max_boundary_shift_s`` defaults to half the search margin. This leaves a
    second half-margin as an explicit guard against a window approaching the
    edge of the locally evaluated band.
    """
    if len(targets) == 0:
        raise ValueError("targets must be a non-empty sequence")
    margin_s = float(search_margin_s)
    if not math.isfinite(margin_s) or margin_s <= 0.0:
        raise ValueError(
            f"search_margin_s must be a positive finite float, got {search_margin_s!r}"
        )
    allowed_shift_s = (
        0.5 * margin_s
        if max_boundary_shift_s is None
        else float(max_boundary_shift_s)
    )
    if (
        not math.isfinite(allowed_shift_s)
        or allowed_shift_s < 0.0
        or allowed_shift_s >= margin_s
    ):
        raise ValueError(
            "max_boundary_shift_s must be finite and satisfy "
            f"0 <= max_boundary_shift_s < search_margin_s; got "
            f"{max_boundary_shift_s!r} and {search_margin_s!r}"
        )
    if "return_samples" in kwargs or "search_intervals_s" in kwargs:
        raise ValueError(
            "continue_delivery_windows_multi owns return_samples and "
            "search_intervals_s; do not supply them in kwargs"
        )

    t_arr = np.asarray(result.t_s, dtype=float)
    if t_arr.ndim != 1 or t_arr.size == 0:
        raise WindowContinuationError(
            "cannot continue windows on an empty or non-1-D time grid"
        )
    if np.any(~np.isfinite(t_arr)) or np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("result.t_s must be finite and strictly increasing")
    t_min_s = float(t_arr[0])
    t_max_s = float(t_arr[-1])

    n_targets = len(targets)
    seeds_by_target: list[list[DeliveryWindow]] = [
        [] for _ in range(n_targets)
    ]
    for window in seed_windows:
        idx = int(window.target_idx)
        if idx < 0 or idx >= n_targets:
            raise WindowContinuationError(
                f"seed window target_idx={idx} is outside [0, {n_targets - 1}]"
            )
        seeds_by_target[idx].append(window)

    all_found: list[DeliveryWindow] = []
    interval_records: list[Tuple[int, float, float]] = []
    n_sample_target_evaluations = 0
    max_observed_shift_s = 0.0

    for target_idx, ((lat_deg, lon_deg), target_seeds) in enumerate(
        zip(targets, seeds_by_target)
    ):
        target_seeds.sort(key=lambda w: w.t_start_s)
        if not target_seeds:
            raise WindowContinuationError(
                f"target {target_idx} has no seed windows; a local search "
                "cannot validate the absence of a newly appearing window"
            )

        intervals = _merge_search_intervals([
            (
                max(t_min_s, float(w.t_start_s) - margin_s),
                min(t_max_s, float(w.t_end_s) + margin_s),
            )
            for w in target_seeds
        ])
        interval_records.extend(
            (target_idx, start_s, end_s) for start_s, end_s in intervals
        )
        evaluation_mask = np.zeros(t_arr.shape, dtype=bool)
        for start_s, end_s in intervals:
            evaluation_mask |= (t_arr >= start_s) & (t_arr <= end_s)
        n_sample_target_evaluations += int(np.count_nonzero(evaluation_mask))

        found_raw, samples = find_delivery_windows(
            result,
            float(lat_deg),
            float(lon_deg),
            return_samples=True,
            search_intervals_s=intervals,
            **kwargs,
        )
        found = (
            found_raw
            if target_idx == 0
            else [replace(w, target_idx=target_idx) for w in found_raw]
        )

        # Every evaluated band must contain a closed sample on both sides.
        # Otherwise a returned run may have been clipped by the local search.
        for start_s, end_s in intervals:
            idx = np.flatnonzero((t_arr >= start_s) & (t_arr <= end_s))
            if idx.size < 2:
                raise WindowContinuationError(
                    f"target {target_idx} search band [{start_s}, {end_s}] s "
                    "contains fewer than two propagation samples"
                )
            if bool(samples.gate_open[idx[0]]) or bool(samples.gate_open[idx[-1]]):
                raise WindowContinuationError(
                    f"target {target_idx} has an open gate at a local-search "
                    f"edge in [{start_s}, {end_s}] s; the window may be clipped"
                )

        if len(found) != len(target_seeds):
            raise WindowContinuationError(
                f"target {target_idx} window count changed from "
                f"{len(target_seeds)} to {len(found)} inside guarded bands"
            )

        for old, new in zip(target_seeds, found):
            shift_s = max(
                abs(float(new.t_start_s) - float(old.t_start_s)),
                abs(float(new.t_end_s) - float(old.t_end_s)),
            )
            max_observed_shift_s = max(max_observed_shift_s, shift_s)
            if shift_s > allowed_shift_s:
                raise WindowContinuationError(
                    f"target {target_idx} boundary shifted {shift_s:.3f} s, "
                    f"exceeding the {allowed_shift_s:.3f} s continuation limit"
                )
        all_found.extend(found)

    all_found.sort(key=lambda w: (w.t_start_s, w.target_idx))
    return ContinuedDeliveryWindows(
        windows=tuple(all_found),
        search_intervals_by_target_s=tuple(interval_records),
        n_sample_target_evaluations=n_sample_target_evaluations,
        n_samples_per_target=int(t_arr.size),
        max_boundary_shift_s=max_observed_shift_s,
    )
