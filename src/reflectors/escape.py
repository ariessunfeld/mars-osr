"""Coupled orbit + attitude propagation of the SRP Mars-escape spiral.

This module combines the escape-model components into one integrated
propagation. The sail spirals out of a Mars orbit under solar-radiation-
pressure thrust, steered by the Q-law (:mod:`reflectors.qlaw`); its attitude
is an integrated dynamical state slewed toward the Q-law's desired normal by
the rate/acceleration-limited controller (:mod:`reflectors.attitude_control`).
The run terminates when the sail crosses the Mars Hill radius -- the exit
state is the handoff boundary condition for a separate interplanetary solver.

State vector (12-D, integrated in Mars-centred J2000):

    y = [ r(3), v(3), n(3), omega(3) ]   km, km/s, unit, rad/s

    dr/dt     = v
    dv/dt     = a_gravity(r, t) + a_third_body(r, t) + a_SRP(r, n, t)
    dn/dt     = omega x n
    domega/dt = alpha_command(n, omega, n*(r, v), limits)

The SRP acceleration uses the **actual** integrated sail normal ``n`` -- never
the instantaneously-desired ``n*`` -- so the angular-velocity / angular-
acceleration limits are honoured at every instant by construction (the
controller can only command ``|alpha| <= alpha_max``, and the sail force
follows whatever attitude the sail has physically slewed to).

Coupled propagation. The Q-law is a feedback law: ``n*`` depends on the
osculating orbit, which evolves continuously over the thousands of revolutions
of an escape spiral. It cannot be pre-built as a kinematic attitude profile,
so the attitude must be integrated alongside the orbit.

Termination. Two terminal events (``reflectors.termination``): the Hill-sphere
``RadiusCeiling`` (the escape objective -- ``termination_reason ==
'hill_sphere_exit'``) and the ``AltitudeFloor`` safety boundary (a failed
escape that spirals back into the atmosphere).

References: see :mod:`reflectors.qlaw`, :mod:`reflectors.gauss`,
:mod:`reflectors.attitude_control`, and :mod:`reflectors.srp` for the
constituent physics and their primary citations.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import spiceypy as spice

from reflectors.attitude_control import (
    AttitudeLimits,
    GovernorParams,
    attitude_derivatives,
    governor_omega_ref,
)
from reflectors.central_body import CentralBody, mars_central_body
from reflectors.dynamics import mars_gm_km3_per_s2, two_body_acceleration
from reflectors.gauss import osculating_elements
from reflectors.qlaw import QLawParams, steer
from reflectors.shadow import shadow_factor
from reflectors.solar_constants import solar_flux_at
from reflectors.srp import SolarSail, mcinnes_srp_acceleration
from reflectors.termination import (
    AltitudeFloor,
    RadiusCeiling,
    make_altitude_floor_event,
    make_energy_gated_radius_ceiling_events,
    make_radius_ceiling_event,
)

logger = logging.getLogger(__name__)

SUN_NAIF_ID = 10
MARS_NAIF_ID = 499

__all__ = [
    "EscapeResult",
    "initial_circular_state",
    "propagate_escape",
]


# Perpendicular-projection floor for the feathered-normal construction.
_PERP_TOL = 1.0e-9


@dataclass(frozen=True)
class EscapeResult:
    """Output of :func:`propagate_escape`.

    Attributes
    ----------
    t_s
        Sample times (s) relative to ``epoch_et``, shape (N,).
    orbit_state_km_kmps
        Orbit states ``[r, v]`` (km, km/s, J2000), shape (N, 6).
    attitude_state
        Attitude states ``[n, omega]`` (unit normal, rad/s, J2000), shape
        (N, 6). The sail normal is renormalised on output.
    termination_reason
        ``'hill_sphere_exit'`` (escape achieved), the altitude-floor label
        (failed -- spiralled back in), or ``'t_final'`` (integration ran to
        the end of ``t_span`` without either event).
    termination_t_s, termination_et
        Time / epoch of the terminal event (``None`` if ``'t_final'``).
    termination_orbit_state_km_kmps, termination_attitude_state
        Orbit / attitude state at termination (``None`` if ``'t_final'``).
        For ``'hill_sphere_exit'`` the orbit state is the handoff boundary
        condition for the interplanetary solver.
    n_rhs_calls
        Number of RHS evaluations (a cost metric).
    solver_message
        SciPy ``solve_ivp`` status message.
    epoch_et
        Absolute SPICE ET of ``t_s = 0``.
    metadata
        Run configuration (sail, Q-law params, gravity degree, ...).
    """

    t_s: np.ndarray
    orbit_state_km_kmps: np.ndarray
    attitude_state: np.ndarray
    termination_reason: str
    termination_t_s: Optional[float]
    termination_et: Optional[float]
    termination_orbit_state_km_kmps: Optional[np.ndarray]
    termination_attitude_state: Optional[np.ndarray]
    n_rhs_calls: int
    solver_message: str
    epoch_et: float
    metadata: dict = field(default_factory=dict)
    # Reference-governor output. When the governor is enabled,
    # ``n_ref`` is the rate-limited reference normal the tracker tracks
    # (see :class:`reflectors.attitude_control.GovernorParams`); it
    # mediates between the steering's possibly-discontinuous ``n_des`` and
    # the bounded-slew tracker. ``None`` when no governor was used.
    reference_normals: Optional[np.ndarray] = None

    @property
    def positions_km(self) -> np.ndarray:
        """Sample positions, shape (N, 3)."""
        return self.orbit_state_km_kmps[:, :3]

    @property
    def sail_normals(self) -> np.ndarray:
        """Sample sail normals, shape (N, 3)."""
        return self.attitude_state[:, :3]

    @property
    def angular_velocities_rad_s(self) -> np.ndarray:
        """Sample angular velocities, shape (N, 3)."""
        return self.attitude_state[:, 3:]

    @property
    def escaped(self) -> bool:
        """True iff the run terminated by crossing the Hill sphere."""
        return self.termination_reason == "hill_sphere_exit"


def initial_circular_state(
    altitude_km: float,
    epoch_et: float,
    *,
    mu_km3_s2: Optional[float] = None,
) -> np.ndarray:
    """Circular sail orbit in Mars's heliocentric orbital plane.

    Builds a circular orbit of the given altitude whose plane is Mars's own
    orbital plane about the Sun. In that plane the Sun stays in-plane for the
    whole mission, so the SRP force has no wasted out-of-plane component --
    a direct geometry for the in-plane (a, e) escape Q-law. Mars obliquity
    is deliberately ignored in this idealised geometry.

    Construction: the orbit-plane normal is Mars's heliocentric orbital
    angular-momentum direction ``h_hat = unit(r_Mars x v_Mars)``; the sail
    starts on the Mars-Sun line (sub-solar side) with a prograde circular
    velocity ``sqrt(mu/a) * unit(h_hat x r_hat)``.

    Parameters
    ----------
    altitude_km
        Orbit altitude above the Mars equatorial radius (km).
    epoch_et
        SPICE ET at which the state is defined (sets the Sun / Mars geometry).
    mu_km3_s2
        Central gravitational parameter for the circular velocity. Defaults
        to the Mars-planet GM (``reflectors.dynamics.mars_gm_km3_per_s2``).

    Returns
    -------
    ndarray, shape (6,)
        ``[r, v]`` in km and km/s, Mars-centred J2000.
    """
    from reflectors.surface import mars_equatorial_radius_km

    if altitude_km <= 0.0:
        raise ValueError(f"altitude_km must be > 0, got {altitude_km}")
    if mu_km3_s2 is None:
        mu_km3_s2 = mars_gm_km3_per_s2()

    a_km = mars_equatorial_radius_km() + altitude_km

    # Mars heliocentric orbital plane normal.
    mars_helio, _ = spice.spkezr("MARS", epoch_et, "J2000", "NONE", "SUN")
    r_mars = np.asarray(mars_helio[:3], dtype=float)
    v_mars = np.asarray(mars_helio[3:], dtype=float)
    h_mars = np.cross(r_mars, v_mars)
    h_hat = h_mars / float(np.linalg.norm(h_mars))

    # Sun direction from Mars (lies in the orbital plane by construction).
    sun_state, _ = spice.spkezr(
        str(SUN_NAIF_ID), epoch_et, "J2000", "NONE", str(MARS_NAIF_ID)
    )
    sun_dir = np.asarray(sun_state[:3], dtype=float)
    # Project onto the orbital plane and normalise -> starting radial direction.
    r_hat = sun_dir - float(np.dot(sun_dir, h_hat)) * h_hat
    r_hat = r_hat / float(np.linalg.norm(r_hat))

    v_hat = np.cross(h_hat, r_hat)  # prograde in-plane direction
    v_circ = math.sqrt(mu_km3_s2 / a_km)
    return np.concatenate([a_km * r_hat, v_circ * v_hat])


def time_reversed_capture_initial_state(
    desired_arrival_state_km_kmps: np.ndarray,
) -> np.ndarray:
    """Backward-run initial state for a capture that must ARRIVE in a given orbit.

    The Mars capture is designed by running the escape (energy-maximising)
    procedure with the ephemeris clocked backward (``ephemeris_time_direction =
    -1``): the integrator still steps FORWARD, so the sail moves along ``+v`` and
    spirals OUTWARD while the Sun and central-body orientation are sampled
    backward in time. The physical capture is the TIME-REVERSE of that
    spiral-out, and time reversal negates velocity. Hence a backward run started
    at ``(r, v)`` produces a forward capture that ARRIVES at ``(r, -v)`` -- the
    same orbit *plane*, but the OPPOSITE circulation (inclination ``i -> 180-i``).

    To make the forward capture arrive at the *desired* orbit ``(r, +v)`` -- e.g.
    the retrograde sun-synchronous LMO, NOT its prograde supplement, which J2
    precesses in the opposite direction and is not sun-synchronous -- the
    backward run must
    therefore START at ``(r, -v)``. This function performs that negation.

    Omitting it delivers the time-reversed supplement orbit: a K12
    sun-synchronous target at ``i = 93.22 deg`` would instead arrive at
    ``i = 86.78 deg`` (prograde, not sun-synchronous).
    The construction is valid because the capture omits atmospheric drag, so the
    forces (gravity, third body, SRP at a given normal) are velocity-independent
    and the time-reverse is an exact solution of the forward dynamics.

    Parameters
    ----------
    desired_arrival_state_km_kmps
        The orbit state ``[r, v]`` (km, km/s, central-body J2000) the forward
        capture should arrive in.

    Returns
    -------
    ndarray, shape (6,)
        ``[r, -v]`` -- the backward-run initial state.
    """
    s = np.asarray(desired_arrival_state_km_kmps, dtype=float)
    if s.shape != (6,):
        raise ValueError(
            f"expected a 6-vector [r(3), v(3)], got shape {s.shape}"
        )
    return np.concatenate([s[:3], -s[3:]])


def _feathered_normal(n_hat: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
    """Edge-on (``n . s_hat = 0``) sail normal nearest ``n_hat``.

    Used for the ``force_coast`` diagnostic mode. Mirrors the feather
    construction in :mod:`reflectors.qlaw`.
    """
    n_perp = n_hat - float(np.dot(n_hat, s_hat)) * s_hat
    norm = float(np.linalg.norm(n_perp))
    if norm > _PERP_TOL:
        return n_perp / norm
    trial = np.array([1.0, 0.0, 0.0])
    if abs(s_hat[0]) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    perp = trial - float(np.dot(trial, s_hat)) * s_hat
    return perp / float(np.linalg.norm(perp))


# Semi-major axis (km) above which the orbit is treated as effectively
# non-elliptic for step sizing: a NEAR-parabolic bound orbit (E -> 0^-) has
# a = -mu/(2E) -> infinity, and ``a**3`` would overflow double precision
# (~1e103 km). Any a this large is physically unbound in the modeled regimes
# (>> the Sun-central nominal ceiling 1e12 km), so None selects the integrator's
# fallback step) rather than overflow. Real bound orbits (Mars/Earth Hill ~1e6
# km, heliocentric transfers ~1e8 km) are far below this and unaffected.
_MAX_ELLIPTIC_SEMIMAJOR_AXIS_KM = 1.0e12


def _osculating_period_s(state6: np.ndarray, mu_km3_s2: float) -> Optional[float]:
    """Osculating orbital period (s), or ``None`` for a non-elliptic orbit.

    Returns ``None`` for energy >= 0 (hyperbolic/parabolic) AND for a bound but
    near-parabolic orbit whose semi-major axis exceeds
    ``_MAX_ELLIPTIC_SEMIMAJOR_AXIS_KM`` -- the latter guards ``a**3`` against
    double-precision overflow when the energy is a hair below zero.
    """
    r = state6[:3]
    v = state6[3:6]
    r_mag = float(np.linalg.norm(r))
    energy = 0.5 * float(np.dot(v, v)) - mu_km3_s2 / r_mag
    if energy >= 0.0:
        return None
    a = -mu_km3_s2 / (2.0 * energy)
    if a > _MAX_ELLIPTIC_SEMIMAJOR_AXIS_KM:
        return None
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu_km3_s2)


def _rk4_step(rhs, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step of size ``h``."""
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _project_omega_to_max(y: np.ndarray, omega_max_rad_s: Optional[float]) -> None:
    """Clip the angular-velocity slot ``y[9:12]`` to ``omega_max_rad_s``.

    In-place projection (``y`` is mutated). No-op when ``omega_max_rad_s`` is
    ``None``. This independent safeguard complements the hard-brake branch in
    :func:`reflectors.attitude_control.alpha_command`: even if a single RK4
    step pushes ``|omega|`` past ``omega_max`` (e.g. when the brake activates
    mid-step), the committed state respects the bound.

    The projection preserves the direction of ``omega``; the post-step
    attitude is consistent with the slew tracker's intent (brake along the
    same axis the controller was already commanding).
    """
    if omega_max_rad_s is None:
        return
    omega = y[9:12]
    omega_mag = float(np.linalg.norm(omega))
    if omega_mag > omega_max_rad_s:
        y[9:12] = omega * (omega_max_rad_s / omega_mag)


def _integrate_escape_rk4(
    rhs, y0, t_span, steps_per_orbit, mu_central, events,
    progress_callback=None, progress_interval=2000,
    omega_max_rad_s: Optional[float] = None,
    max_step_true_anomaly_rad: Optional[float] = None,
    max_step_s: Optional[float] = None,
):
    """Fixed-step RK4 integration of the escape state with terminal events.

    The Q-law is a feedback law: ``qlaw.steer`` runs a numerical search on
    every RHS evaluation, so the RHS is *not* a smooth function of state --
    it has grid-resolution kinks. Adaptive step control can collapse near
    those kinks. Fixed-step RK4 avoids that response and is the standard
    choice for Q-law propagation (pyqlaw also defaults to RK4).

    The step is sized to the osculating orbital period,
    ``h = period / steps_per_orbit``, so the per-orbit resolution stays
    constant as the orbit expands. ``events`` is a list of
    ``(g, direction, label, accept)``: a sign change of ``g(0.0, y)`` across a
    step in the event's ``direction`` (-1 downward, +1 upward) locates a crossing
    by bisection within the step; the run terminates there unless an optional
    ``accept(y_cross) -> bool`` predicate (``None`` -> always accept) rejects it,
    in which case integration continues. Multiple events may register a crossing
    on one step; they are tried in order and the first ACCEPTED one terminates.

    ``progress_callback``, if given, is called ``progress_callback(t, y,
    step_index)`` every ``progress_interval`` accepted steps -- for live
    monitoring of a long escape run.

    ``omega_max_rad_s``, when supplied, enables a strict post-step projection
    of the attitude angular velocity (state slot ``y[9:12]``) to that
    magnitude -- guarantees ``|omega| <= omega_max_rad_s`` on every committed
    state. See :func:`_project_omega_to_max`.

    ``max_step_true_anomaly_rad``, when supplied, additionally caps the step so
    the spacecraft advances at most this much TRUE ANOMALY per step:
    ``h <= max_step_true_anomaly_rad / nu_dot`` with ``nu_dot = |r x v| / |r|^2``
    the instantaneous true-anomaly rate. Because ``nu_dot`` peaks at periapsis
    (small ``|r|``), this clusters time-steps through the fast, dynamically
    stiff periapsis passage and stays loose near apoapsis -- a Sundman-style
    regularisation. ``steps_per_orbit`` alone is insufficient near
    near periapsis: as ``a`` grows the period grows but the periapsis passage
    stays short, so a fixed steps/orbit under-resolves it. Default ``None``
    disables the true-anomaly cap.

    ``max_step_s``, when supplied, is an absolute per-step ceiling (s). A fixed
    ``steps_per_orbit`` gives ever-larger absolute steps as ``a`` grows, so the
    coarse-step attitude-ODE error returns at high ``a``; this caps it so the
    attitude tracker stays converged all the way to escape. Default ``None``
    disables the absolute cap.

    Returns ``(t_array, y_array, termination_reason, t_cross, y_cross)``;
    the last two are ``None`` when the integration runs to ``t_span[1]``.
    """
    t0, tf = t_span
    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    ts = [t]
    ys = [y.copy()]
    n_accepted = 0
    # Fallback step when the orbit is non-elliptic (period undefined) -- it
    # is escaping and near the Hill boundary by then.
    fallback_h = (tf - t0) / (20.0 * steps_per_orbit)

    g_prev = [g(0.0, y) for (g, _d, _l, _a) in events]

    while t < tf:
        period = _osculating_period_s(y, mu_central)
        h = period / steps_per_orbit if period is not None else fallback_h
        if max_step_true_anomaly_rad is not None:
            # Cap the true-anomaly advance per step (Sundman-style; resolves
            # the fast, stiff periapsis passage independent of a).
            r_vec = y[:3]
            v_vec = y[3:6]
            r_mag = float(np.linalg.norm(r_vec))
            h_ang = float(np.linalg.norm(np.cross(r_vec, v_vec)))
            if h_ang > 0.0 and r_mag > 0.0:
                nu_dot = h_ang / (r_mag * r_mag)  # rad/s
                h = min(h, max_step_true_anomaly_rad / nu_dot)
        if max_step_s is not None:
            # Absolute step ceiling -- bounds the (stiff-ish) attitude-tracker
            # ODE integration error at ALL semi-major axes. A fixed
            # steps_per_orbit gives ever-larger absolute steps as a grows, so
            # attitude-tracking integration error returns at high a; this caps it.
            h = min(h, max_step_s)
        h = min(h, tf - t)
        if h <= 0.0:
            break

        y_new = _rk4_step(rhs, t, y, h)
        _project_omega_to_max(y_new, omega_max_rad_s)
        t_new = t + h
        g_new = [g(0.0, y_new) for (g, _d, _l, _a) in events]

        # All events whose g changed sign across the step in the event's
        # direction. Usually 0 or 1; the energy-gated Hill exit can register
        # TWO (its radius + energy detectors) on one step, so try each in turn
        # and terminate on the first whose accept-predicate passes.
        fired_indices = [
            i
            for i, (_g, direction, _label, _accept) in enumerate(events)
            if (direction < 0 and g_prev[i] > 0.0 and g_new[i] <= 0.0)
            or (direction > 0 and g_prev[i] < 0.0 and g_new[i] >= 0.0)
        ]
        for fired in fired_indices:
            # Bisect within [t, t + h] for this event's crossing.
            g, direction, label, accept = events[fired]
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                g_mid = g(0.0, _rk4_step(rhs, t, y, mid * h))
                before_crossing = (
                    (direction < 0 and g_mid > 0.0)
                    or (direction > 0 and g_mid < 0.0)
                )
                if before_crossing:
                    lo = mid
                else:
                    hi = mid
                if (hi - lo) * h < 1.0e-3:
                    break
            theta = 0.5 * (lo + hi)
            y_cross = _rk4_step(rhs, t, y, theta * h)
            _project_omega_to_max(y_cross, omega_max_rad_s)
            # An optional accept-predicate gates termination at the located
            # crossing (energy-gated Hill exit: the radius detector counts only
            # when E>=0, the energy detector only when |r|>=Hill). accept is
            # None for unconditional floor/ceiling events -> always accept, so
            # the first fired event terminates. A
            # rejected crossing does NOT terminate; keep integrating.
            if accept is not None and not accept(y_cross):
                continue
            t_cross = t + theta * h
            ts.append(t_cross)
            ys.append(y_cross.copy())
            return np.array(ts), np.array(ys), label, t_cross, y_cross

        t, y, g_prev = t_new, y_new, g_new
        ts.append(t)
        ys.append(y.copy())
        n_accepted += 1
        if progress_callback is not None and n_accepted % progress_interval == 0:
            progress_callback(t, y, n_accepted)

    return np.array(ts), np.array(ys), "t_final", None, None


def propagate_escape(
    initial_orbit_state_km_kmps: np.ndarray,
    epoch_et: float,
    sail: SolarSail,
    qlaw_params: QLawParams,
    attitude_limits: AttitudeLimits,
    t_span_s: tuple[float, float],
    *,
    initial_n_hat: Optional[np.ndarray] = None,
    initial_omega_rad_s: Optional[np.ndarray] = None,
    gravity_degree: int = 2,
    include_sun_third_body: bool = True,
    radius_ceiling: Optional[RadiusCeiling] = None,
    altitude_floor: Optional[AltitudeFloor] = None,
    steps_per_orbit: int = 200,
    max_step_true_anomaly_deg: Optional[float] = None,
    max_step_s: Optional[float] = None,
    force_coast: bool = False,
    steering_fn: Optional[Callable[..., np.ndarray]] = None,
    kinematic_attitude: bool = False,
    governor_params: Optional[GovernorParams] = None,
    anticipatory_lead_s: float = 0.0,
    anticipatory_activate_deg: float = 20.0,
    central_body: Optional[CentralBody] = None,
    third_bodies: Optional[Sequence] = None,
    drag_force_fn: Optional[Callable[..., np.ndarray]] = None,
    ephemeris_time_direction: int = +1,
    energy_gated: bool = False,
    progress_callback=None,
) -> EscapeResult:
    """Propagate the coupled orbit + attitude SRP escape spiral.

    Parameters
    ----------
    initial_orbit_state_km_kmps
        Initial orbit state ``[r, v]`` (km, km/s, Mars-centred J2000),
        shape (6,). See :func:`initial_circular_state`.
    epoch_et
        Absolute SPICE ET of ``t_span_s[0]``.
    sail
        ``reflectors.srp.SolarSail`` bus.
    qlaw_params
        ``reflectors.qlaw.QLawParams`` -- the escape Q-law configuration.
    attitude_limits
        ``reflectors.attitude_control.AttitudeLimits`` -- the slew limits.
    t_span_s
        ``(t0, tf)`` in seconds relative to ``epoch_et``.
    initial_n_hat
        Initial sail normal (J2000). Default: the Q-law's desired normal at
        the initial state, so the sail starts already optimally oriented.
    initial_omega_rad_s
        Initial angular velocity (rad/s). Default: zero (sail at rest).
    gravity_degree
        Mars spherical-harmonic degree (and order) for the gravity
        perturbation. ``0`` = point-mass Mars only. Default 2 (J2 -- the
        dominant harmonic, important near the low-altitude periapsis).
    include_sun_third_body
        Include the Sun third-body perturbation (Montenbruck & Gill). Default
        True -- essential out at the Hill sphere where solar tides dominate.
    radius_ceiling
        Outer terminal boundary. Default: ``RadiusCeiling.hill_sphere()``.
    altitude_floor
        Inner terminal boundary. Default: ``AltitudeFloor.at_km(300.0)``.
    steps_per_orbit
        Fixed-step RK4 resolution: the step is the osculating orbital period
        divided by this. Default 200. The escape RHS is a feedback law (a
        numerical search runs every call) and is not smooth, so an adaptive
        integrator is unusable here -- see :func:`_integrate_escape_rk4`.
    max_step_true_anomaly_deg
        Optional per-step cap on the true-anomaly advance (deg). When set, the
        step is additionally bounded so the spacecraft sweeps at most this much
        true anomaly per RK4 step, which clusters steps through the fast,
        stiff periapsis passage at any semi-major axis (``steps_per_orbit``
        alone under-resolves periapsis as ``a`` grows -- the period grows but
        the periapsis passage stays short). Default ``None`` disables the cap. See
        :func:`_integrate_escape_rk4`.
    max_step_s
        Optional absolute per-step ceiling (s). A fixed ``steps_per_orbit``
        gives ever-larger absolute steps as ``a`` grows, so the coarse-step
        attitude-tracker integration error returns at high ``a``; this caps the
        step so the tracker stays converged all the way to escape. Default
        ``None``.
    force_coast
        Diagnostic mode: if True the Q-law / ``steering_fn`` is bypassed and
        the sail is always commanded to feather (edge-on, ~zero SRP). Used
        as the control-off baseline -- the orbit then evolves under gravity
        + Sun only.
    steering_fn
        Optional alternative steering callable. When supplied, replaces the
        Q-law :func:`reflectors.qlaw.steer` for computing the desired sail
        normal ``n*`` at each RHS evaluation. Signature:
        ``steering_fn(r, v, s_hat, p_eff, sail, current_n_hat) -> n_star``
        (J2000 unit vector). Used by :mod:`reflectors.escape_dedot` to
        plug in the direct dE/dt-maximising steering. When ``None``
        (default), the Q-law is used and ``qlaw_params`` configures it.
    kinematic_attitude
        Bypass the integrated rate/accel-limited attitude tracker: the SRP
        force at each RHS evaluation uses the steering's desired normal
        ``n*`` DIRECTLY (so the sail effectively snaps to ``n*`` every
        step, no slew limits). The attitude state ``(n, omega)`` is still
        carried for output bookkeeping but ``omega`` stays zero and ``n``
        is reset to ``n*`` at each evaluation. This provides a no-slew-limit
        reference case. Default False
        (use the integrated attitude tracker for the full mission model).
    governor_params
        Optional :class:`reflectors.attitude_control.GovernorParams`. When
        supplied, enables the reference governor: an extra state
        slot ``n_ref`` (the rate-limited reference normal) is integrated
        alongside the tracker, and ``alpha_command`` targets ``n_ref``
        instead of the steering's possibly-discontinuous ``n_des``. The
        integrator state grows from 12-D to 15-D. Required for
        realistic-slew escape with full-physics, blended-steering at sigma=18.
        Mutually exclusive with ``kinematic_attitude=True``.
    anticipatory_lead_s
        Feed-forward command lead (s). When > 0 (and not kinematic), the
        tracker's target is re-evaluated at the state ``anticipatory_lead_s``
        ahead (2-body Kepler propagation; Sun + shadow advanced by the same
        dt). If the lead command differs from the present command by more
        than ``anticipatory_activate_deg``, the sail begins slewing toward
        the look-ahead orientation immediately, so it arrives in the required
        (e.g.
        feather) orientation by the time the transition occurs, rather than
        spending the slew mid-transition applying force in the opposing
        direction. Smooth phases (lead ~= present command) are unaffected.
        Default 0.0 (disabled; reactive tracking).
    anticipatory_activate_deg
        Lead activation threshold (deg). The pre-slew engages only when the
        present and lead commands differ by more than this (a genuine
        imminent reorientation), so smooth arcs incur no lead. Default 20.
    central_body
        :class:`reflectors.central_body.CentralBody` selecting the central
        body (NAIF id, body-fixed frame, GM, equatorial radius, Hill radius,
        gravity-model factory). Default ``None`` reconstructs Mars via
        :func:`reflectors.central_body.mars_central_body`. Pass
        ``earth_central_body()`` for Earth escape.
    third_bodies
        Explicit sequence of :class:`reflectors.third_body.ThirdBody` perturbers.
        Default ``None`` uses Sun-only perturbations, gated by
        ``include_sun_third_body``. For Earth pass
        ``(sun_third_body(), moon_third_body())`` -- the Moon is a first-order
        cislunar perturber (inside the Hill sphere), unlike at Mars.
    drag_force_fn
        Optional atmospheric-drag callable
        ``drag_force_fn(r_km, v_kmps, n_hat, et) -> a_km_s2`` (J2000), summed
        into the acceleration after SRP; ``n_hat`` is the actual integrated sail
        normal (the drag projected area depends on attitude). Default ``None``
        -> no drag (a true no-op for every Mars caller). Built by the drag
        module (:mod:`reflectors.drag`) for the LEO-start Earth escape.
    ephemeris_time_direction
        Sign of the map from integration time ``t`` to the SPICE epoch the
        ephemeris is queried at: ``et(t) = epoch_et + ephemeris_time_direction * t``.
        Default ``+1`` selects forward time.
        ``-1`` clocks the Sun/third-body ephemeris BACKWARD while the sail state
        still integrates forward (steps stay positive). Used to generate a
        CAPTURE node: the escape law under a backward-walking Sun spirals out to
        a far Hill-sphere state whose time-reverse is a real, forward-Sun Mars
        capture. The far endpoint is OUTBOUND, so as a
        capture handoff the delivered velocity is ``-v`` (inbound). The dedot
        steering law and the integrator are unchanged -- only the epoch the Sun
        is read at moves backward.
    energy_gated
        When ``True``, the Hill-exit termination requires BOTH ``|r| >= Hill``
        AND a non-negative planet-relative two-body energy ``E = v^2/2 - mu/|r|``
        -- the genuine escape condition. The default ``False`` uses the
        radius-only ``RadiusCeiling`` (which would label a bound orbit grazing the
        Hill radius as an escape). With the gate on, a bound graze does not
        terminate; integration continues until ``E >= 0`` at ``|r| >= Hill`` (or
        an outer kill-radius at 2*Hill fires as a clear NON-escape). When
        ``E >= 0`` is reached before the Hill crossing, the gate does not alter
        the termination state; it distinguishes the bound-graze case. See
        :func:`reflectors.termination.make_energy_gated_radius_ceiling_events`.
        Default ``False``.
    progress_callback
        Optional ``callback(t_s, y, step_index)`` invoked every ~2000 accepted
        integration steps -- for live monitoring of a long escape run.

    Returns
    -------
    EscapeResult
    """
    if ephemeris_time_direction not in (+1, -1):
        raise ValueError(
            "ephemeris_time_direction must be +1 (forward) or -1 (backward), "
            f"got {ephemeris_time_direction}"
        )
    state0 = np.asarray(initial_orbit_state_km_kmps, dtype=float)
    if state0.shape != (6,):
        raise ValueError(f"initial orbit state must be (6,), got {state0.shape}")
    if gravity_degree < 0:
        raise ValueError(f"gravity_degree must be >= 0, got {gravity_degree}")
    if steps_per_orbit < 1:
        raise ValueError(
            f"steps_per_orbit must be >= 1, got {steps_per_orbit}"
        )
    if governor_params is not None and kinematic_attitude:
        raise ValueError(
            "governor_params and kinematic_attitude are mutually exclusive: "
            "the governor rate-limits the tracker's reference normal, but "
            "kinematic_attitude bypasses the tracker entirely."
        )
    if governor_params is not None:
        if governor_params.omega_ref_max_rad_s >= attitude_limits.omega_max_rad_s:
            raise ValueError(
                "governor omega_ref_max_rad_s "
                f"({governor_params.omega_ref_max_rad_s:.4e}) must be < tracker "
                f"omega_max_rad_s ({attitude_limits.omega_max_rad_s:.4e}) so "
                "the tracker has headroom to follow n_ref."
            )

    # Central body: default None selects Mars. Earth uses earth_central_body().
    if central_body is None:
        central_body = mars_central_body()

    r0 = state0[:3]
    v0 = state0[3:]

    # Forecast-aware steering callables may need the absolute epoch. Detect once
    # whether steering_fn accepts an ``et`` keyword and pass it only if so.
    _steering_wants_et = (
        steering_fn is not None
        and "et" in inspect.signature(steering_fn).parameters
    )

    def _call_steering(r_, v_, s_hat_, p_eff_, sail_, cur_, et_):
        if _steering_wants_et:
            return steering_fn(r_, v_, s_hat_, p_eff_, sail_, cur_, et=et_)
        return steering_fn(r_, v_, s_hat_, p_eff_, sail_, cur_)

    # --- Gravity model + central mu --------------------------------------
    model = None
    if gravity_degree > 0:
        model = central_body.gravity_model_factory(max_degree=max(gravity_degree, 2))
        mu_central = model.mu_km3_s2
    else:
        mu_central = central_body.mu_km3_s2

    # --- Third bodies -----------------------------------------------------
    # Default (third_bodies=None) selects Sun-only perturbations for Mars,
    # gated by include_sun_third_body. An explicit list (e.g. Sun + Moon for
    # Earth) overrides it.
    if third_bodies is None:
        from reflectors.third_body import sun_third_body

        active_third_bodies: tuple = (
            (sun_third_body(),) if include_sun_third_body else ()
        )
    else:
        active_third_bodies = tuple(third_bodies)

    # --- Terminal events --------------------------------------------------
    # Built directly from central_body (not the Mars factories) so the same
    # propagator retargets to Earth; for Mars these reproduce
    # RadiusCeiling.hill_sphere() / AltitudeFloor.at_km(300.0) exactly (same
    # radii, same default labels).
    if radius_ceiling is None:
        radius_ceiling = RadiusCeiling(radius_km=central_body.hill_radius_km)
    if altitude_floor is None:
        altitude_floor = AltitudeFloor(
            altitude_km=300.0,
            reference_radius_km=central_body.equatorial_radius_km,
        )

    r0_mag = float(np.linalg.norm(r0))
    alt0 = r0_mag - altitude_floor.reference_radius_km
    if alt0 < altitude_floor.altitude_km:
        raise ValueError(
            f"initial state is below altitude_floor: alt0 = {alt0:.3f} km"
        )
    if r0_mag > radius_ceiling.radius_km:
        raise ValueError(
            f"initial state is above radius_ceiling: r0 = {r0_mag:.3f} km"
        )

    # --- Sun-geometry helper (one Sun fetch per epoch) -------------------
    def _sun_geometry(r_sat: np.ndarray, et: float):
        """Return (s_hat, P_effective_pa) at the state.

        ``P_effective`` is the solar radiation pressure gated by the binary
        central-body umbra -- zero in eclipse, so the steering naturally
        feathers there. Observer + umbra-disc radius come from ``central_body``.
        """
        sun_state, _ = spice.spkezr(
            str(SUN_NAIF_ID), et, "J2000", "NONE", str(central_body.naif_id)
        )
        r_sun = np.asarray(sun_state[:3], dtype=float)
        sat_to_sun = r_sun - r_sat
        r_helio = float(np.linalg.norm(sat_to_sun))
        s_hat = sat_to_sun / r_helio
        # The umbra is bypassed when the central body cannot occult the Sun it
        # illuminates (SUN-as-central interplanetary cruise): with the Sun at
        # the frame origin the shadow geometry is degenerate (always-eclipse),
        # and a 1-AU sail is never occulted by the modelled bodies. Mars/Earth
        # keep occults_sun=True.
        if central_body.occults_sun:
            shadow = shadow_factor(
                r_sat, et, central_body.naif_id, sun_position_j2000_km=r_sun,
                central_radius_km=central_body.equatorial_radius_km,
            )
        else:
            shadow = 1.0
        p_eff = solar_flux_at(r_helio) * shadow
        return s_hat, p_eff

    # --- Initial attitude -------------------------------------------------
    if initial_n_hat is None:
        s_hat0, p_eff0 = _sun_geometry(r0, epoch_et)
        if steering_fn is None:
            el0 = osculating_elements(r0, v0, mu_central)
            steering0 = steer(
                el0, r0, v0, s_hat0, p_eff0, sail,
                current_n_hat=s_hat0, params=qlaw_params,
            )
            n0 = steering0.n_star_j2000
        else:
            n0 = np.asarray(
                _call_steering(r0, v0, s_hat0, p_eff0, sail, s_hat0, epoch_et),
                dtype=float,
            )
            n0 = n0 / float(np.linalg.norm(n0))
    else:
        n0 = np.asarray(initial_n_hat, dtype=float)
        n0 = n0 / float(np.linalg.norm(n0))
    omega0 = (
        np.zeros(3) if initial_omega_rad_s is None
        else np.asarray(initial_omega_rad_s, dtype=float)
    )
    # Initial state: 12-D without governor, 15-D with. n_ref(0) = n(0)
    # so the governor starts already at its target; no startup transient.
    if governor_params is None:
        y0 = np.concatenate([r0, v0, n0, omega0])
    else:
        y0 = np.concatenate([r0, v0, n0, omega0, n0])

    # --- RHS --------------------------------------------------------------
    # Body-generic gravity-perturbation closure, built once. Dispatch on the
    # model type: the Earth zonal model -> singular-free zonal recurrence in
    # IAU_EARTH; the Mars MRO120F model -> the Cunningham full-harmonic path in
    # IAU_MARS.
    gravity_accel_inertial = None
    if model is not None:
        from reflectors.earth_gravity import EarthGravityModel

        if isinstance(model, EarthGravityModel):
            from reflectors.gravity import zonal_acceleration_inertial

            _grav_frame = central_body.body_frame
            _grav_J = model.J_by_degree
            _grav_mu = model.mu_km3_s2
            _grav_rref = model.ref_radius_km

            def gravity_accel_inertial(r_vec, et_):
                return zonal_acceleration_inertial(
                    r_vec, et_, _grav_mu, _grav_rref, _grav_J,
                    body_frame=_grav_frame,
                )
        else:  # MarsGravityModel (Cunningham full-harmonic, IAU_MARS)
            from reflectors.gravity import mars_gravity_acceleration_inertial

            def gravity_accel_inertial(r_vec, et_):
                return mars_gravity_acceleration_inertial(
                    r_vec, et_, model, gravity_degree, gravity_degree,
                    include_central=False,
                )
    if active_third_bodies:
        from reflectors.third_body import third_body_acceleration_from_spice
    from reflectors.qlaw import evaluate_orbit_effectivity

    # Petropoulos effectivity-coast cache. The orbit-effectivity envelope
    # (qdot_min, qdot_max) is a slow function of orbit shape and Sun
    # direction; refresh every effectivity_refresh_steps RK4 substages.
    # The list-as-mutable-closure-state pattern keeps the rhs() closure
    # itself stateless.
    effectivity_active = (
        not force_coast
        and (
            qlaw_params.eta_a_threshold > 0.0
            or qlaw_params.eta_r_threshold > 0.0
        )
    )
    eff_cache: list = [None, None, 10**9]  # qdot_min, qdot_max, calls_since_refresh
    state_size = 15 if governor_params is not None else 12

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:6]
        n_raw = y[6:9]
        omega = y[9:12]
        n = n_raw / float(np.linalg.norm(n_raw))
        # n_ref is the governor's slew-rate-limited reference normal. Only
        # carried in 15-D mode (governor_params not None). Read it now so
        # the RHS below can branch on it without re-checking state_size.
        if governor_params is not None:
            n_ref_raw = y[12:15]
            n_ref = n_ref_raw / float(np.linalg.norm(n_ref_raw))
        et = epoch_et + ephemeris_time_direction * t

        # Orbit accelerations.
        a = two_body_acceleration(r, mu_central)
        if gravity_accel_inertial is not None:
            a = a + gravity_accel_inertial(r, et)
        if active_third_bodies:
            a = a + third_body_acceleration_from_spice(
                r, et, active_third_bodies,
                observer_naif_id=central_body.naif_id,
            )

        # Sun geometry shared between the SRP force and the steering.
        s_hat, p_eff = _sun_geometry(r, et)

        # Desired sail normal: alternate steering_fn, Q-law, or feather.
        if force_coast:
            n_des = _feathered_normal(n, s_hat)
        elif steering_fn is not None:
            n_des = np.asarray(
                _call_steering(r, v, s_hat, p_eff, sail, n, et), dtype=float,
            )
            n_des = n_des / float(np.linalg.norm(n_des))
        else:
            elements = osculating_elements(r, v, mu_central)
            envelope = None
            if effectivity_active and p_eff > 0.0:
                eff_cache[2] += 1
                if eff_cache[2] >= qlaw_params.effectivity_refresh_steps:
                    env = evaluate_orbit_effectivity(
                        elements, s_hat, p_eff, sail, qlaw_params,
                    )
                    if env is not None:
                        eff_cache[0], eff_cache[1] = env
                    else:
                        eff_cache[0] = eff_cache[1] = None
                    eff_cache[2] = 0
                if eff_cache[0] is not None:
                    envelope = (eff_cache[0], eff_cache[1])
            steering = steer(
                elements, r, v, s_hat, p_eff, sail,
                current_n_hat=n, params=qlaw_params,
                effectivity_envelope=envelope,
            )
            n_des = steering.n_star_j2000

        # SRP: kinematic mode uses n_des (the sail instantly tracks the
        # commanded normal); integrated mode uses the actual attitude n.
        n_force = n_des if kinematic_attitude else n
        a = a + mcinnes_srp_acceleration(n_force, s_hat, p_eff, sail)

        # Atmospheric drag (Earth LEO-start escape). Default None -> no-op, so
        # every Mars caller is unaffected. The hook is passed the actual
        # integrated sail normal n -- the drag projected area A*|n.v_hat| depends
        # on attitude, so drag must see n (it is a state, not a function of
        # (r, et)). Signature: drag_force_fn(r_km, v_kmps, n_hat, et) -> a_km_s2.
        if drag_force_fn is not None:
            a = a + np.asarray(drag_force_fn(r, v, n, et), dtype=float)

        # Anticipatory feed-forward lead. The bounded-slew tracker reacts to
        # the present command, so at a feather<->thrust or shadow transition
        # the sail begins slewing only AT the transition and spends minutes
        # mid-slew applying force in the opposing direction during the arc
        # where orientation matters. To pre-position, re-evaluate the steering
        # law at the state ``anticipatory_lead_s`` ahead (2-body Kepler
        # propagation + Sun/shadow advanced by the same dt); if a LARGE
        # reorientation is imminent within that window, start slewing toward
        # the look-ahead command immediately. In smooth phases that command is
        # nearly the present one, so the target is unchanged (no over-lead /
        # no smooth-phase efficiency loss). Disabled (lead==0) and in
        # kinematic mode (no slew to lead). This starts large slews early
        # enough to prioritise arriving in the correct feather orientation.
        n_track = n_des
        if (
            anticipatory_lead_s > 0.0
            and not kinematic_attitude
            and not force_coast
            and steering_fn is not None
        ):
            fut = np.asarray(
                spice.prop2b(
                    mu_central,
                    np.concatenate([r, v]),
                    anticipatory_lead_s,
                ),
                dtype=float,
            )
            r_f, v_f = fut[:3], fut[3:6]
            et_f = et + ephemeris_time_direction * anticipatory_lead_s
            s_hat_f, p_eff_f = _sun_geometry(r_f, et_f)
            n_des_f = np.asarray(
                _call_steering(r_f, v_f, s_hat_f, p_eff_f, sail, n, et_f),
                dtype=float,
            )
            n_des_f = n_des_f / float(np.linalg.norm(n_des_f))
            cos_delta = max(-1.0, min(1.0, float(np.dot(n_des, n_des_f))))
            if math.degrees(math.acos(cos_delta)) > anticipatory_activate_deg:
                n_track = n_des_f

        # Attitude dynamics. Three branches:
        #   - kinematic: zero (the sail snaps to n_des each step).
        #   - integrated tracker without governor: tracker targets n_track
        #     (= n_des unless the anticipatory lead pre-positions for an
        #     imminent transition).
        #   - integrated tracker with governor: tracker targets n_ref, and
        #     n_ref slews toward n_track at rate omega_ref (saturated-prop
        #     slerp; see :func:`reflectors.attitude_control.governor_omega_ref`).
        if kinematic_attitude:
            n_dot = np.zeros(3)
            omega_dot = np.zeros(3)
            n_ref_dot = None
        elif governor_params is None:
            n_dot, omega_dot = attitude_derivatives(
                n, omega, n_track, attitude_limits
            )
            n_ref_dot = None
        else:
            n_dot, omega_dot = attitude_derivatives(
                n, omega, n_ref, attitude_limits
            )
            omega_ref = governor_omega_ref(n_ref, n_track, governor_params)
            n_ref_dot = np.cross(omega_ref, n_ref)

        out = np.empty(state_size)
        out[:3] = v
        out[3:6] = a
        out[6:9] = n_dot
        out[9:12] = omega_dot
        if state_size == 15:
            out[12:15] = n_ref_dot
        return out

    # Terminal events as (g, direction, label, accept): floor fires on a
    # downward crossing, ceiling on an upward crossing. ``accept`` (None for the
    # plain events) is an optional predicate gating termination at the located
    # crossing -- used by the energy-gated Hill exit.
    events = [
        (
            make_altitude_floor_event(altitude_floor),
            -1,
            altitude_floor.label,
            None,
        ),
    ]
    if energy_gated:
        # Escape = first time E>=0 AND |r|>=Hill (NOT a bound graze of the Hill
        # radius). Two smooth gated detectors (radius up-cross accepted when
        # E>=0; energy up-cross accepted when |r|>=Hill) + an outer kill-radius.
        events.extend(
            make_energy_gated_radius_ceiling_events(radius_ceiling, mu_central)
        )
    else:
        events.append(
            (
                make_radius_ceiling_event(radius_ceiling),
                +1,
                radius_ceiling.label,
                None,
            )
        )

    logger.info(
        "propagate_escape: t_span=%s, body=%s, gravity_degree=%d, "
        "third_bodies=%s, steps_per_orbit=%d, force_coast=%s",
        t_span_s, central_body.label, gravity_degree,
        [getattr(b, "label", str(b)) for b in active_third_bodies],
        steps_per_orbit, force_coast,
    )
    # Strict |omega| <= omega_max post-step projection. Disabled when
    # kinematic_attitude=True (the attitude sub-dynamics are zero by
    # construction, so omega never grows from its zero initial value).
    omega_cap = None if kinematic_attitude else attitude_limits.omega_max_rad_s
    max_nu_rad = (
        math.radians(max_step_true_anomaly_deg)
        if max_step_true_anomaly_deg is not None else None
    )
    t_arr, y_arr, termination_reason, t_cross, y_cross = _integrate_escape_rk4(
        rhs, y0, t_span_s, steps_per_orbit, mu_central, events,
        progress_callback=progress_callback,
        omega_max_rad_s=omega_cap,
        max_step_true_anomaly_rad=max_nu_rad,
        max_step_s=max_step_s,
    )
    # RK4 evaluates the RHS 4x per step (the one-time event bisection adds a
    # small fixed overhead, not counted here).
    n_rhs_calls = 4 * max(len(t_arr) - 1, 0)

    # Renormalise the integrated sail normal (and n_ref if governor is on).
    y_out = np.asarray(y_arr, dtype=float)  # (N, 12) or (N, 15)
    n_norms = np.linalg.norm(y_out[:, 6:9], axis=1, keepdims=True)
    y_out[:, 6:9] = y_out[:, 6:9] / n_norms
    reference_normals_out: Optional[np.ndarray] = None
    if governor_params is not None:
        ref_norms = np.linalg.norm(y_out[:, 12:15], axis=1, keepdims=True)
        y_out[:, 12:15] = y_out[:, 12:15] / ref_norms
        reference_normals_out = y_out[:, 12:15].copy()

    termination_t_s: Optional[float] = None
    termination_et: Optional[float] = None
    termination_orbit: Optional[np.ndarray] = None
    termination_attitude: Optional[np.ndarray] = None
    if t_cross is not None:
        y_cross = np.asarray(y_cross, dtype=float).copy()
        y_cross[6:9] = y_cross[6:9] / float(np.linalg.norm(y_cross[6:9]))
        if governor_params is not None:
            y_cross[12:15] = y_cross[12:15] / float(
                np.linalg.norm(y_cross[12:15])
            )
        termination_t_s = float(t_cross)
        termination_et = epoch_et + ephemeris_time_direction * float(t_cross)
        termination_orbit = y_cross[:6].copy()
        termination_attitude = y_cross[6:12].copy()
        logger.info(
            "propagate_escape: terminal event %r at t=%.1f s (|r|=%.1f km)",
            termination_reason, termination_t_s,
            float(np.linalg.norm(y_cross[:3])),
        )

    metadata = {
        "sail": {
            "area_m2": sail.area_m2,
            "mass_kg": sail.mass_kg,
            "loading_kg_per_m2": sail.loading_kg_per_m2,
        },
        "qlaw_params": {
            "a_target_km": qlaw_params.a_target_km,
            "rp_min_km": qlaw_params.rp_min_km,
            "w_apo": qlaw_params.w_apo,
            "w_a": qlaw_params.w_a,
            "w_e": qlaw_params.w_e,
            "e_ref": qlaw_params.e_ref,
            "w_p": qlaw_params.w_p,
            "k_petro": qlaw_params.k_petro,
            "eta_a_threshold": qlaw_params.eta_a_threshold,
            "eta_r_threshold": qlaw_params.eta_r_threshold,
            "effectivity_n_samples": qlaw_params.effectivity_n_samples,
            "effectivity_refresh_steps": qlaw_params.effectivity_refresh_steps,
        },
        "attitude_limits": {
            "alpha_max_rad_s2": attitude_limits.alpha_max_rad_s2,
            "omega_max_rad_s": attitude_limits.omega_max_rad_s,
        },
        "governor_params": (
            None if governor_params is None
            else {
                "omega_ref_max_rad_s": governor_params.omega_ref_max_rad_s,
                "theta_settle_rad": governor_params.theta_settle_rad,
            }
        ),
        "gravity_degree": gravity_degree,
        "include_sun_third_body": bool(active_third_bodies),
        "mu_central_km3_s2": mu_central,
        "central_body": {
            "naif_id": central_body.naif_id,
            "label": central_body.label,
            "body_frame": central_body.body_frame,
            "equatorial_radius_km": central_body.equatorial_radius_km,
            "hill_radius_km": central_body.hill_radius_km,
            "occults_sun": central_body.occults_sun,
        },
        "third_bodies": [getattr(b, "label", str(b)) for b in active_third_bodies],
        "gravity_source": (model.source if model is not None else "point_mass"),
        "drag": bool(drag_force_fn is not None),
        "energy_gated": energy_gated,
        "radius_ceiling_km": radius_ceiling.radius_km,
        "altitude_floor_km": altitude_floor.altitude_km,
        "steps_per_orbit": steps_per_orbit,
        "max_step_true_anomaly_deg": max_step_true_anomaly_deg,
        "max_step_s": max_step_s,
        "ephemeris_time_direction": ephemeris_time_direction,
        "force_coast": force_coast,
        "steering": (
            "force_coast" if force_coast
            else ("steering_fn" if steering_fn is not None else "qlaw")
        ),
    }

    return EscapeResult(
        t_s=t_arr,
        orbit_state_km_kmps=y_out[:, :6],
        attitude_state=y_out[:, 6:12],
        termination_reason=termination_reason,
        termination_t_s=termination_t_s,
        termination_et=termination_et,
        termination_orbit_state_km_kmps=termination_orbit,
        termination_attitude_state=termination_attitude,
        reference_normals=reference_normals_out,
        n_rhs_calls=n_rhs_calls,
        solver_message=f"fixed-step RK4, {len(t_arr)} steps",
        epoch_et=epoch_et,
        metadata=metadata,
    )
