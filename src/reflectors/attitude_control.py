"""Rate- and acceleration-limited sail-attitude tracking controller.

The Q-law escape steering (:mod:`reflectors.qlaw`) is a *feedback* law: the
desired sail normal ``n*`` is recomputed from the osculating orbit at every
integration step. The sail cannot snap to ``n*`` instantaneously -- it must
slew there subject to hard limits on angular velocity and angular acceleration.
This module supplies the controller that does so.

It is a **time-optimal, accel/rate-limited geodesic tracker**. Given the
current sail normal ``n``, its angular velocity ``omega``, and the desired
normal ``n*``, :func:`alpha_command` returns the angular acceleration
``alpha`` to command, with the guarantees:

  - ``|alpha| <= alpha_max``  -- by construction (the command is a vector of
    magnitude at most ``alpha_max``);
  - ``|omega| <= omega_max``  -- the controller steers ``omega`` toward a
    target rate capped at ``omega_max`` and decelerates to avoid overshoot. A
    separate hard-brake branch fires when
    ``|omega|`` reaches ``omega_max``: ``alpha = -alpha_max * omega / |omega|``
    (full deceleration along the rotation axis). The hard-brake branch, paired
    with a post-step ``|omega|`` projection in
    :mod:`reflectors.escape`, ``|omega|`` is bounded by ``omega_max`` to
    integration precision regardless of step size.
  - ``n -> n*`` without overshoot -- the target rate follows the braking
    profile ``omega_des = sqrt(2 alpha_max theta_e)`` (capped), the largest
    rate from which a constant ``-alpha_max`` deceleration brings the slew
    exactly to rest at the target.

The same controller handles both regimes the escape mission needs uniformly:
small continuous tracking of a slowly-drifting Q-law ``n*`` (``theta_e`` small,
``omega_des`` small) and large thrust<->feather reorientations (``theta_e``
large -- a full accel-limited bang/ramp/bang slew). It is a closed-loop
feedback realisation of the same accel-limited reorientation that the
open-loop :func:`reflectors.attitude.smooth_slew` quintic performs; unlike the
quintic it can track a *moving* target, which a feedback Q-law requires.

Scope. This is a kinematic attitude model: no inertia tensor or torque model.
The angular acceleration ``alpha`` is the directly commanded control,
hard-bounded by ``alpha_max``. Escape propagation uses an integrated attitude
state ``(n, omega)`` whose evolution

    dn/dt     = omega x n
    domega/dt = alpha_command(n, omega, n*)

is integrated alongside the orbit in :mod:`reflectors.escape`. The kinematic
identities ``omega = n x dn/dt`` and ``alpha = n x d2n/dt2`` (S^2, with
``omega . n = 0``; roll about the sail normal is unobservable for a flat sail)
are the same ones :mod:`reflectors.attitude` uses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AttitudeLimits",
    "GovernorParams",
    "alpha_command",
    "attitude_derivatives",
    "governor_omega_ref",
]


# Cross-product magnitude below which the geodesic rotation axis from n to n*
# is treated as undefined (n parallel or antiparallel to n*). Dimensionless --
# |n x n*| = sin(theta_e) for unit vectors.
_AXIS_TOL = 1.0e-12

# Early-braking margin. The controller switches from accelerating to braking
# when the progress rate reaches (1 - _BRAKE_MARGIN_FRAC) times the stop-rate
# sqrt(2 alpha_max theta_e). The margin guarantees the slew can always be
# brought to rest within the remaining error: the trajectory rides just inside
# the time-optimal bang-bang switching curve, converging with a tiny undershoot
# rather than overshooting. The cost is a slew ~_BRAKE_MARGIN_FRAC slower than
# the strict time-optimal minimum -- still far faster than the C2 quintic
# (reflectors.attitude.smooth_slew). Without this margin a proportional law
# under-brakes at the accelerate->brake transition and the stop-rate curve
# then falls faster than alpha_max can track, producing an overshoot.
_BRAKE_MARGIN_FRAC = 0.05


@dataclass(frozen=True)
class AttitudeLimits:
    """Hard kinematic limits and tuning for the attitude tracker.

    Attributes
    ----------
    alpha_max_rad_s2
        Peak angular acceleration (rad/s^2). The commanded ``alpha`` never
        exceeds this. Default 0.003 -- the canonical value used throughout
        the attitude-feasibility analyses.
    omega_max_rad_s
        Peak angular velocity (rad/s). The controller's commanded progress
        rate is capped here. Default 1.0 -- far above the natural peak of any
        sail slew at ``alpha_max = 0.003`` (a 90 deg slew peaks near
        0.1 rad/s), so by default the slew is acceleration-governed, consistent
        with the open-loop ``smooth_slew`` limit. Set it to
        a spacecraft rate limit to make it bind.
    omega_smooth_rad_s
        Angular-rate width of the ``tanh`` blend between the accelerate
        (``+alpha_max``) and brake (``-alpha_max``) commands. Small enough
        that the controller brakes at essentially full authority once on the
        switching curve, large enough that the command is smooth (no
        bang-bang chatter -- keeps the integrated-attitude ODE well-behaved).
        Default 1e-3 rad/s.
    """

    alpha_max_rad_s2: float = 0.003
    omega_max_rad_s: float = 1.0
    omega_smooth_rad_s: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.alpha_max_rad_s2 <= 0.0:
            raise ValueError(
                f"alpha_max_rad_s2 must be > 0, got {self.alpha_max_rad_s2}"
            )
        if self.omega_max_rad_s <= 0.0:
            raise ValueError(
                f"omega_max_rad_s must be > 0, got {self.omega_max_rad_s}"
            )
        if self.omega_smooth_rad_s <= 0.0:
            raise ValueError(
                f"omega_smooth_rad_s must be > 0, got "
                f"{self.omega_smooth_rad_s}"
            )


@dataclass(frozen=True)
class GovernorParams:
    """Rate-limited reference-normal governor.

    The integrated tracker (:func:`alpha_command`) can follow a reference
    that changes slowly enough for its slew bounds, but the escape steering
    (``blended_steer`` and similar) can produce a desired normal ``n_des``
    that flips discontinuously across mode boundaries (feather threshold,
    eclipse penumbra, energy-vs-safety mode switch). When ``n_des`` jumps
    faster than ``omega_max_rad_s`` can deliver, the tracker accumulates a
    growing pointing error.

    The governor inserts an intermediate reference ``n_ref`` between
    ``n_des`` and the tracker. ``n_ref`` slews toward ``n_des`` along the
    great circle at a rate bounded by ``omega_ref_max_rad_s``. The tracker
    targets ``n_ref`` instead of ``n_des``; because ``|dn_ref/dt| <=
    omega_ref_max_rad_s`` is strictly below the tracker's ``omega_max``,
    the tracker has headroom to follow ``n_ref`` without saturating.

    Rate law (saturated-proportional slerp in continuous form):

        theta_e_ref       = acos(n_ref . n_des)
        e_hat_ref         = unit(n_ref x n_des)        # rotation axis
        |omega_ref|       = omega_ref_max * min(1, theta_e_ref / theta_settle)
        omega_ref         = |omega_ref| * e_hat_ref

    So when ``n_ref`` is far from ``n_des``, the governor slews at the rate
    cap; within ``theta_settle`` of the target it ramps proportionally so
    ``n_ref`` converges smoothly rather than chattering at saturation.

    The continuous-form rate law is the ODE limit of per-guidance-interval
    spherical linear interpolation (slerp):
    independent of any discrete guidance cadence, numerically smooth, and
    makes the strict-rate guarantee a property of the dynamics rather than
    a property of the integration timestep.

    Attributes
    ----------
    omega_ref_max_rad_s
        Peak rate at which ``n_ref`` slews toward ``n_des``. Must be
        strictly less than the tracker's ``omega_max_rad_s`` -- typically
        0.7-0.9 of it (20-30% headroom so the tracker keeps up). No
        default: the proper value is set by the caller from the paired
        :class:`AttitudeLimits`.
    theta_settle_rad
        Small-angle softening width. Inside this band the slew rate ramps
        proportionally; outside it saturates at ``omega_ref_max_rad_s``.
        Default 0.01 rad ~= 0.57 deg -- below this, n_ref is "close
        enough" to n_des that the governor backs off the rate cap. Smaller
        values give snappier convergence but stiffer dynamics; larger
        values give smoother but laggier convergence.
    """

    omega_ref_max_rad_s: float
    theta_settle_rad: float = 0.01

    def __post_init__(self) -> None:
        if self.omega_ref_max_rad_s <= 0.0:
            raise ValueError(
                f"omega_ref_max_rad_s must be > 0, got "
                f"{self.omega_ref_max_rad_s}"
            )
        if self.theta_settle_rad <= 0.0:
            raise ValueError(
                f"theta_settle_rad must be > 0, got {self.theta_settle_rad}"
            )


def governor_omega_ref(
    n_ref_hat: np.ndarray,
    n_des_hat: np.ndarray,
    params: GovernorParams,
) -> np.ndarray:
    """Angular velocity at which ``n_ref`` slews toward ``n_des``.

    Implements the saturated-proportional slerp described in
    :class:`GovernorParams`. The returned ``omega_ref`` has magnitude
    bounded by ``params.omega_ref_max_rad_s`` and points along the
    great-circle axis from ``n_ref`` to ``n_des`` (``e_hat_ref = unit(n_ref
    x n_des)``). At the aligned / antipodal degeneracies the rate is zero
    (already at target) or the axis is irrelevant (any 180 deg slew works
    -- a perpendicular axis is chosen).

    Parameters
    ----------
    n_ref_hat
        Current reference normal (J2000, unit vector), shape (3,).
        Re-normalised internally.
    n_des_hat
        Desired normal from the guidance (J2000, unit), shape (3,).
        Re-normalised internally.
    params
        :class:`GovernorParams`.

    Returns
    -------
    ndarray, shape (3,)
        Angular velocity vector of ``n_ref`` (rad/s, J2000), strictly
        bounded by ``params.omega_ref_max_rad_s``.
    """
    n_ref = np.asarray(n_ref_hat, dtype=float)
    n_ref = n_ref / float(np.linalg.norm(n_ref))
    n_des = np.asarray(n_des_hat, dtype=float)
    n_des = n_des / float(np.linalg.norm(n_des))

    cos_te = max(-1.0, min(1.0, float(np.dot(n_ref, n_des))))
    theta_e = math.acos(cos_te)

    if theta_e <= 0.0:
        # Already at the target -- omega_ref is exactly zero.
        return np.zeros(3)

    cross = np.cross(n_ref, n_des)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm > _AXIS_TOL:
        e_hat = cross / cross_norm
    else:
        # Antipodal (theta_e ~ pi): any perpendicular axis is a valid
        # 180 deg slew; pick a deterministic one.
        e_hat = _perpendicular_unit(n_ref)

    rate_factor = min(1.0, theta_e / params.theta_settle_rad)
    omega_mag = params.omega_ref_max_rad_s * rate_factor
    return omega_mag * e_hat


def _perpendicular_unit(n_hat: np.ndarray) -> np.ndarray:
    """An arbitrary unit vector orthogonal to ``n_hat``.

    Used as the slew axis only in the degenerate cases where the geodesic
    axis is undefined: ``n`` aligned with ``n*`` (then the target rate is
    zero and the axis is irrelevant) or antipodal (any perpendicular axis
    is an equally valid 180 deg slew direction).
    """
    # Cross with whichever world axis is least aligned with n_hat.
    trial = np.array([1.0, 0.0, 0.0])
    if abs(n_hat[0]) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    perp = np.cross(n_hat, trial)
    return perp / float(np.linalg.norm(perp))


def alpha_command(
    n_hat: np.ndarray,
    omega_rad_s: np.ndarray,
    n_star_hat: np.ndarray,
    limits: AttitudeLimits,
) -> np.ndarray:
    """Angular acceleration to slew ``n`` toward ``n*`` within the limits.

    Time-optimal accel/rate-limited geodesic tracker -- see the module
    docstring for the guarantees. The command is built as follows:

    1. Geodesic error: angle ``theta_e`` between ``n`` and ``n*`` and the
       rotation axis ``e_hat = unit(n x n*)`` (an arbitrary perpendicular
       axis in the aligned / antipodal degeneracies). ``omega_e = omega .
       e_hat`` is the rate of progress toward the target.
    2. Switching curve. ``omega_stop = sqrt(2 alpha_max theta_e)`` is the
       largest progress rate from which a constant ``-alpha_max`` brings the
       slew to rest exactly at ``theta_e = 0``; ``omega_ref = min(omega_max,
       omega_stop)`` also respects the rate cap. The command along ``e_hat``
       is bang-bang -- accelerate (``+alpha_max``) below the switch, brake
       (``-alpha_max``) above it -- smoothed by a ``tanh`` of width
       ``omega_smooth``:

           alpha_e = -alpha_max * tanh( (omega_e - omega_switch) / omega_smooth )

       with ``omega_switch = (1 - _BRAKE_MARGIN_FRAC) * omega_ref``. The
       early-brake margin keeps the trajectory just inside the switching
       curve -- it converges with a tiny undershoot, never an overshoot --
       and, crucially, the command retains full ``alpha_max`` magnitude on
       the curve (a proportional law would command ~zero there and
       under-brake).
    3. A light ``tanh`` damping term nulls any cross-track ``omega``
       component (the nominal geodesic slew keeps ``omega`` along
       the fixed great-circle axis).

    The result is clamped to ``|alpha| <= alpha_max`` and projected onto the
    plane perpendicular to ``n`` so the integrated ``omega`` stays orthogonal
    to the sail normal (the flat-sail ``omega . n = 0`` convention; roll
    about ``n`` is unobservable).

    Parameters
    ----------
    n_hat
        Current sail-normal unit vector (J2000), shape (3,). Re-normalised
        internally.
    omega_rad_s
        Current angular velocity (rad/s, J2000), shape (3,).
    n_star_hat
        Desired sail-normal unit vector (J2000), shape (3,). Re-normalised
        internally.
    limits
        ``AttitudeLimits``.

    Returns
    -------
    ndarray, shape (3,)
        Commanded angular acceleration (rad/s^2, J2000), ``|alpha| <=
        alpha_max``.
    """
    n = np.asarray(n_hat, dtype=float)
    n = n / float(np.linalg.norm(n))
    ns = np.asarray(n_star_hat, dtype=float)
    ns = ns / float(np.linalg.norm(ns))
    omega = np.asarray(omega_rad_s, dtype=float)

    alpha_max = limits.alpha_max_rad_s2

    cos_te = max(-1.0, min(1.0, float(np.dot(n, ns))))
    theta_e = math.acos(cos_te)

    cross = np.cross(n, ns)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm > _AXIS_TOL:
        e_hat = cross / cross_norm
    else:
        # Aligned (theta_e ~ 0; omega_stop ~ 0, axis irrelevant) or antipodal
        # (theta_e ~ pi; any perpendicular axis is a valid 180 deg slew).
        e_hat = _perpendicular_unit(n)

    omega_smooth = limits.omega_smooth_rad_s

    # Progress rate toward the target along the geodesic axis.
    omega_e = float(np.dot(omega, e_hat))

    # Switching curve: largest progress rate from which -alpha_max still stops
    # the slew within the remaining error, also capped at omega_max.
    omega_stop = math.sqrt(2.0 * alpha_max * theta_e)
    omega_ref = min(limits.omega_max_rad_s, omega_stop)
    omega_switch = (1.0 - _BRAKE_MARGIN_FRAC) * omega_ref

    # Bang-bang accelerate/brake along e_hat, tanh-smoothed. Full alpha_max
    # magnitude is retained on the switching curve (no proportional fade).
    alpha_e = -alpha_max * math.tanh((omega_e - omega_switch) / omega_smooth)

    # Cross-track safeguard: null any omega not along e_hat.
    omega_perp = omega - omega_e * e_hat
    perp_norm = float(np.linalg.norm(omega_perp))
    if perp_norm > 0.0:
        alpha_perp = (
            -alpha_max * math.tanh(perp_norm / omega_smooth)
            * (omega_perp / perp_norm)
        )
    else:
        alpha_perp = np.zeros(3)

    alpha = alpha_e * e_hat + alpha_perp

    # Strict omega_max enforcement (hard-brake branch). The interior switching
    # curve relies on a 5% early-brake margin + tanh-smoothed bang-bang and is
    # adequate when an adaptive integrator can shrink its step near the
    # switch. Under fixed-step RK4 (~36 s/step in the late
    # escape phase), |omega| can creep past omega_max without this guard.
    # Override with full deceleration along
    # -omega/|omega| whenever |omega| is at or above the limit, so the bound
    # holds at integration precision regardless of step size. The post-step
    # |omega| projection in :func:`reflectors.escape._integrate_escape_rk4`
    # catches any residual intra-step overshoot.
    omega_max = limits.omega_max_rad_s
    omega_mag = float(np.linalg.norm(omega))
    if omega_mag >= omega_max:
        alpha = -alpha_max * (omega / omega_mag)

    # Clamp the total command to the acceleration limit.
    alpha_mag = float(np.linalg.norm(alpha))
    if alpha_mag > alpha_max:
        alpha = alpha * (alpha_max / alpha_mag)

    # Keep the command in the tangent plane so omega stays perpendicular to n.
    alpha = alpha - float(np.dot(alpha, n)) * n
    return alpha


def attitude_derivatives(
    n_hat: np.ndarray,
    omega_rad_s: np.ndarray,
    n_star_hat: np.ndarray,
    limits: AttitudeLimits,
) -> tuple[np.ndarray, np.ndarray]:
    """Time derivatives of the integrated attitude state ``(n, omega)``.

    Packages the attitude sub-dynamics for the coupled escape propagation
    (and for stand-alone slew tests):

        dn/dt     = omega x n
        domega/dt = alpha_command(n, omega, n*, limits)

    Returns ``(n_dot, omega_dot)``. ``n`` is re-normalised on read inside
    :func:`alpha_command`; ``n_dot = omega x n`` keeps ``|n|`` constant to
    integration tolerance, and callers may renormalise ``n`` between steps.
    """
    n = np.asarray(n_hat, dtype=float)
    omega = np.asarray(omega_rad_s, dtype=float)
    n_dot = np.cross(omega, n)
    omega_dot = alpha_command(n, omega, n_star_hat, limits)
    return n_dot, omega_dot
