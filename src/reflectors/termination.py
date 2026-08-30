"""Terminal-event hooks for the Mars-centred propagator.

Target physics: hard safety boundary that terminates a ``propagate``
call cleanly when the sail altitude drops below a configured floor.
Below the floor the force model is no longer valid. The event is a
single-scalar zero-crossing with
SciPy's native ``solve_ivp`` event API; termination is rendered in
the ``PropagationResult`` as a first-class status rather than an unreported
truncation.

Default floor is 300 km above the Mars equatorial radius -- a
conservative choice that stops propagation well above the thermopause
(~120 km). The floor is parameterised per call via an ``AltitudeFloor`` frozen
dataclass.

Altitude convention: ``altitude = |r| - R_reference``, with
``R_reference`` defaulting to Mars's equatorial radius (3396.19 km
from the PCK), matching the codebase convention used throughout the
orbit-state helpers in ``tests/test_dynamics.py``. A custom
``reference_radius_km`` is exposed so callers can use the polar or mean radius.

Reference (primary): SciPy SourceForge, ``solve_ivp`` ``events``
parameter -- each event function returns a scalar that the IVP solver
monitors for zero crossings, with ``terminal`` and ``direction``
attributes controlling whether a crossing stops the integration and
which sign of crossing qualifies. Setting ``direction=-1`` ensures that
only downward crossings (altitude decreasing through the floor) fire the
event; the initial-state check in ``dynamics.propagate`` catches any
trajectory that starts below the floor and rejects it before the
integrator begins.

Out of scope: Martian atmospheric density, flat-plate drag, atmospheric winds,
and seasonal variability. Oblate-ellipsoid altitude
``altitude = |r| - R_local(lat)`` differs from the equatorial-radius
convention by at most 20 km at the poles; immaterial against a 300 km
floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# Type alias for SciPy ``solve_ivp`` event callables. The callable must
# additionally carry ``terminal`` and ``direction`` attributes; these are set
# on the returned closure before handing it to ``solve_ivp``.
ScipyEvent = Callable[[float, np.ndarray], float]


@dataclass(frozen=True)
class AltitudeFloor:
    """Hard altitude floor for sail propagation.

    A sail that crosses ``altitude_km`` above ``reference_radius_km``
    downward terminates the ``propagate`` call with
    ``termination_reason == label``. The comparison is on the
    Euclidean magnitude of the inertial position vector, so the floor
    is a sphere of radius ``reference_radius_km + altitude_km``
    centred on the integration origin (Mars J2000 centre, for the
    current propagator).

    Parameters
    ----------
    altitude_km
        Minimum allowed altitude above ``reference_radius_km``, km.
        Must be strictly positive -- a floor of zero or negative
        places the boundary at or inside the reference body and is
        not physically meaningful.
    reference_radius_km
        Reference body radius (km) for the altitude definition. In
        the default (``None``) path, the equatorial Mars radius from
        the loaded PCK is used via ``AltitudeFloor.at_km(...)`` or by
        the caller supplying it explicitly.
    label
        String stored in ``PropagationResult.termination_reason`` when
        the event fires. Defaults to ``"atmosphere_intersected"`` to
        identify the atmospheric-safety boundary; a custom label
        lets a caller distinguish between, e.g., a primary 300 km
        boundary and a secondary 200 km "alarm" boundary.

    Examples
    --------
    >>> from reflectors.termination import AltitudeFloor
    >>> AltitudeFloor.at_km(300.0)
    AltitudeFloor(altitude_km=300.0, reference_radius_km=3396.19,
                  label='atmosphere_intersected')
    """

    altitude_km: float
    reference_radius_km: float
    label: str = "atmosphere_intersected"

    def __post_init__(self) -> None:
        if self.altitude_km <= 0.0:
            raise ValueError(
                "AltitudeFloor.altitude_km must be > 0, got "
                f"{self.altitude_km}"
            )
        if self.reference_radius_km <= 0.0:
            raise ValueError(
                "AltitudeFloor.reference_radius_km must be > 0, got "
                f"{self.reference_radius_km}"
            )
        if not self.label:
            raise ValueError("AltitudeFloor.label must be a non-empty string")

    @property
    def floor_radius_km(self) -> float:
        """Geocentric radius (km) at which the floor lies.

        Convenience property: ``floor_radius_km = reference_radius_km
        + altitude_km``. This is the radius at which the event
        function changes sign.
        """
        return self.reference_radius_km + self.altitude_km

    @classmethod
    def at_km(
        cls,
        altitude_km: float,
        *,
        label: str = "atmosphere_intersected",
    ) -> "AltitudeFloor":
        """Build an ``AltitudeFloor`` using the Mars equatorial radius.

        Convenience factory that pulls ``R_eq`` from the loaded PCK
        via ``reflectors.surface.mars_equatorial_radius_km``. Lifts
        the R_eq lookup out of call sites so downstream code stays
        compact.

        Parameters
        ----------
        altitude_km
            Minimum allowed altitude above R_eq, km.
        label
            Optional override for the termination label.
        """
        # Imported locally to avoid a module-level dependency on
        # reflectors.surface (which in turn has a heavier SPICE surface).
        from reflectors.surface import mars_equatorial_radius_km

        return cls(
            altitude_km=altitude_km,
            reference_radius_km=mars_equatorial_radius_km(),
            label=label,
        )


def make_altitude_floor_event(floor: AltitudeFloor) -> ScipyEvent:
    """Build a SciPy ``solve_ivp``-compatible terminal event function.

    The returned callable has signature ``g(t, y) -> float`` where
    ``y`` is the 6-vector propagator state ``[r_x, r_y, r_z, v_x, v_y,
    v_z]``. It computes

        g(t, y) = |r| - (R_reference + altitude)

    which is **positive above the floor** and **negative below**.
    SciPy's event root-finder detects zero crossings; the attached
    ``terminal = True`` attribute makes such a crossing stop the
    integrator, and ``direction = -1`` restricts detection to
    DOWNWARD crossings only. A sail that grazes the floor while
    climbing (e.g. after lifting from a sub-floor initial state --
    which ``propagate`` rejects anyway) does not trigger termination.

    Parameters
    ----------
    floor
        The ``AltitudeFloor`` configuration.

    Returns
    -------
    callable
        SciPy-compatible event function with ``terminal=True`` and
        ``direction=-1`` attached.
    """
    floor_radius = floor.floor_radius_km

    def event(t: float, y: np.ndarray) -> float:
        r = y[:3]
        return float(np.linalg.norm(r)) - floor_radius

    # SciPy inspects these attributes on the event callable itself.
    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1.0  # type: ignore[attr-defined]
    return event


@dataclass(frozen=True)
class RadiusCeiling:
    """Hard OUTER radius boundary for sail propagation.

    The mirror image of :class:`AltitudeFloor`: a sail that crosses
    ``radius_km`` (Mars-centred inertial radius) *upward* terminates the
    ``propagate`` call with ``termination_reason == label``. Used as the
    outer boundary of the SRP escape spiral -- the Mars Hill sphere --
    after which Mars's gravity is no longer dominant and the state is
    handed off to a separate interplanetary solver.

    Unlike ``AltitudeFloor`` the boundary is specified as an ABSOLUTE
    Mars-centred radius (not an altitude above a reference body), because
    the Hill sphere is naturally an absolute distance and there is no
    "altitude above the surface" intuition at ~10^6 km.

    Parameters
    ----------
    radius_km
        Mars-centred inertial radius (km) at which the event fires. Must
        be strictly positive.
    label
        String stored in ``PropagationResult.termination_reason`` when
        the event fires. Defaults to ``"hill_sphere_exit"``.

    Examples
    --------
    >>> from reflectors.termination import RadiusCeiling
    >>> RadiusCeiling.hill_sphere()
    RadiusCeiling(radius_km=1084100.0, label='hill_sphere_exit')
    """

    radius_km: float
    label: str = "hill_sphere_exit"

    def __post_init__(self) -> None:
        if self.radius_km <= 0.0:
            raise ValueError(
                f"RadiusCeiling.radius_km must be > 0, got {self.radius_km}"
            )
        if not self.label:
            raise ValueError("RadiusCeiling.label must be a non-empty string")

    @classmethod
    def hill_sphere(cls, *, label: str = "hill_sphere_exit") -> "RadiusCeiling":
        """Build a ``RadiusCeiling`` at the Mars Hill radius.

        Convenience factory pinning ``radius_km`` to
        ``reflectors.mars_constants.MARS_HILL_RADIUS_KM`` (~1.084e6 km;
        see that module for the derivation and verification test).

        Parameters
        ----------
        label
            Optional override for the termination label.
        """
        # Local import keeps termination.py a light leaf module.
        from reflectors.mars_constants import MARS_HILL_RADIUS_KM

        return cls(radius_km=MARS_HILL_RADIUS_KM, label=label)


def make_radius_ceiling_event(ceiling: RadiusCeiling) -> ScipyEvent:
    """Build a SciPy ``solve_ivp``-compatible terminal event function.

    The returned callable has signature ``g(t, y) -> float`` where ``y``
    is a propagator state whose first three components are the inertial
    position ``[r_x, r_y, r_z]`` (true of both the 6-D point-mass state
    and the 12-D augmented escape state ``[r, v, n, omega]``). It computes

        g(t, y) = |r| - radius_ceiling

    which is **negative below the ceiling** and **positive above**.
    ``terminal = True`` stops the integrator on a crossing, and
    ``direction = +1`` restricts detection to UPWARD crossings (radius
    increasing through the boundary) -- a sail momentarily above the
    ceiling and falling back does not trigger termination.

    Parameters
    ----------
    ceiling
        The ``RadiusCeiling`` configuration.

    Returns
    -------
    callable
        SciPy-compatible event function with ``terminal=True`` and
        ``direction=+1`` attached.
    """
    radius = ceiling.radius_km

    def event(t: float, y: np.ndarray) -> float:
        r = y[:3]
        return float(np.linalg.norm(r)) - radius

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = +1.0  # type: ignore[attr-defined]
    return event


def make_energy_gated_radius_ceiling_events(
    ceiling: RadiusCeiling,
    mu_km3_s2: float,
    *,
    outer_kill_factor: float = 2.0,
) -> list:
    """Energy-gated Hill-exit termination: escape = first time E>=0 AND r>=Hill.

    The plain :func:`make_radius_ceiling_event` fires on ``|r| >= radius`` ALONE,
    so a BOUND orbit (specific energy ``E < 0``) whose apoapsis merely grazes the
    Hill radius would be wrongly reported as an escape. The genuine escape
    condition is BOTH ``|r| >= R_hill`` AND ``E = v^2/2 - mu/|r| >= 0`` (the
    planet-relative two-body energy non-negative -> unbound from the central
    body). Requiring both conditions prevents a bound orbit that merely touches
    the Hill sphere from being classified as escaped.

    Returns a list of ``(g, direction, label, accept)`` event tuples for the
    fixed-step escape integrator
    (:func:`reflectors.escape._integrate_escape_rk4`), whose
    ``accept(y_cross) -> bool`` predicate gates termination at a located
    crossing. Two SMOOTH detectors share ``ceiling.label`` so EITHER ordering of
    the two conditions terminates:

    - **radius up-cross** (``g = |r| - R_hill``), accepted only if ``E >= 0`` --
      the usual case (energy goes positive well before the Hill crossing);
    - **energy up-cross** (``g = E``), accepted only if ``|r| >= R_hill`` -- the
      "climb-then-energize" case (a bound graze that keeps rising under SRP and
      crosses ``E = 0`` while already beyond the Hill radius), which the radius
      detector alone would miss.

    A third **outer kill-radius** detector at ``outer_kill_factor * R_hill``
    (label ``"outer_kill_radius"``, always accepted) bounds a weakly bound
    graze that keeps climbing without ever reaching ``E >= 0``: it terminates as
    a clear NON-escape (label != ``ceiling.label``) rather than running to
    ``t_final`` and burning the whole integration span. A genuine escape always
    fires a Hill-exit detector first (between ``R_hill`` and
    ``outer_kill_factor*R_hill``), so the outer kill only triggers on real
    non-escapes.

    Using two smooth single-condition detectors (rather than one
    ``min(|r|-R, c*E)`` event) keeps every in-step bisection on a
    monotonic-near-crossing scalar and avoids a problem-dependent scale factor
    between the radius (km) and energy (km^2/s^2) units.

    Parameters
    ----------
    ceiling
        The Hill-sphere :class:`RadiusCeiling` (radius + label).
    mu_km3_s2
        Central-body gravitational parameter (km^3/s^2) for the specific energy.
    outer_kill_factor
        Outer kill-radius as a multiple of ``ceiling.radius_km``. Default 2.0.

    Returns
    -------
    list of (callable, int, str, callable | None)
        ``(g, direction, label, accept)`` tuples for the escape integrator.
    """
    if mu_km3_s2 <= 0.0:
        raise ValueError(f"mu_km3_s2 must be > 0, got {mu_km3_s2}")
    if outer_kill_factor <= 1.0:
        raise ValueError(
            f"outer_kill_factor must be > 1, got {outer_kill_factor}"
        )
    radius = ceiling.radius_km

    def _radius(y: np.ndarray) -> float:
        return float(np.linalg.norm(y[:3]))

    def _energy(y: np.ndarray) -> float:
        r_mag = float(np.linalg.norm(y[:3]))
        v2 = float(np.dot(y[3:6], y[3:6]))
        return 0.5 * v2 - mu_km3_s2 / r_mag

    def g_radius(t: float, y: np.ndarray) -> float:
        return _radius(y) - radius

    def accept_radius(y: np.ndarray) -> bool:
        return _energy(y) >= 0.0

    def g_energy(t: float, y: np.ndarray) -> float:
        return _energy(y)

    def accept_energy(y: np.ndarray) -> bool:
        return _radius(y) >= radius

    def g_outer(t: float, y: np.ndarray) -> float:
        return _radius(y) - outer_kill_factor * radius

    return [
        (g_radius, +1, ceiling.label, accept_radius),
        (g_energy, +1, ceiling.label, accept_energy),
        (g_outer, +1, "outer_kill_radius", None),
    ]
