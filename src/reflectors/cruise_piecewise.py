"""Piecewise-constant RTN tilt-angle sail command + terminal-miss cost for the
interplanetary Earth-Hill -> Mars-Hill SRP-sail cruise.

This complements :mod:`reflectors.cruise_command` (time-Fourier cone/clock)
and :mod:`reflectors.cruise_cost` with a different attitude
parameterisation for the single-shooting cruise solve:

  * The McInnes cone/clock polar coordinate is singular at small cone because
    clock is undefined when the sail is near face-on to the Sun.
  * A piecewise-constant command in two small RTN tilt angles off the radial /
    Sun-line avoids that singularity and gives IPOPT a well-conditioned terminal
    Jacobian.

Parameterisation (per time segment ``i`` of ``N`` equal segments in mission time):

    s_hat                        Sun-line (sail -> Sun; Sun-central: s_hat = -r_hat).
                                 The PRIMARY axis: the sail normal is a small tilt
                                 OFF the Sun-line, so it stays in the sunward
                                 hemisphere (n . s_hat > 0) and the one-sided
                                 McInnes sail is lit -> nonzero thrust.
    k_hat = unit(ref_normal)     FIXED out-of-plane axis = heliocentric orbit
                                 normal at departure; all calculations use
                                 J2000, and at departure r0 . (r0 x v0) = 0 so
                                 s_hat is in-plane there).
    t_hat = unit(k_hat x s_hat)  in-plane transverse
    n_planar = cos(phi_i) s_hat + sin(phi_i) t_hat      phi = IN-PLANE pitch off s_hat
    n_des    = cos(theta_i) n_planar + sin(theta_i) k_hat   theta = OUT-OF-PLANE tilt

``phi`` and ``theta`` are each bounded to a small box (typically +/-35 deg);
the cone angle from the Sun-line is ``acos(cos phi cos theta) in [0, ~49 deg]`` at the
box corners, spanning the ~35 deg optimal-transverse-thrust regime.

The McInnes SRP model in :mod:`reflectors.srp` is one-sided: force is zero when
``n . s_hat <= 0``. The command therefore tilts from the sunward ``s_hat``
rather than an anti-sunward radial direction.

The steerer returns only the unit sail normal ``n_des``. The downstream SRP
model applies the cone-angle efficiency and optical coefficients, so this
module must not apply an additional force factor. The command is normalized
here and again at the propagation boundary.

The cruise runs the command KINEMATICALLY (``kinematic_attitude=True``); slew
feasibility is enforced separately and trivially (a piecewise-constant command's
only slew demand is the per-node reorientation, which over a multi-month segment
sits ~6 orders of magnitude under the limits -- see
:func:`assert_piecewise_slew_feasible`).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import numpy as np

from reflectors.attitude_control import AttitudeLimits
from reflectors.central_body import CentralBody
from reflectors.cruise_cost import _ensure_worker_kernels  # fork-safe kernel guard
from reflectors.escape import EscapeResult, propagate_escape
from reflectors.qlaw import QLawParams

# Below this a cross product / norm is treated as degenerate (mirrors
# cruise_command._EA_DEGENERATE_TOL).
_DEGENERATE_TOL = 1.0e-12


# ---------------------------------------------------------------------------
# Decision-vector packing + segment indexing
# ---------------------------------------------------------------------------


def pack_angles(phis: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Pack ``[phi_0..phi_{N-1}, theta_0..theta_{N-1}]`` (block layout, radians)."""
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if phis.shape != thetas.shape or phis.ndim != 1:
        raise ValueError(
            f"phis and thetas must be 1-D of equal length; got {phis.shape}, {thetas.shape}"
        )
    return np.concatenate([phis, thetas])


def unpack_angles(x: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_angles`: return ``(phis, thetas)`` from a length-2N vector."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 * N:
        raise ValueError(f"x has {x.size} entries; need >= 2N = {2 * N}")
    return x[:N].copy(), x[N : 2 * N].copy()


def segment_index(tau: float, N: int) -> int:
    """Map normalised mission time ``tau in [0, 1]`` to piecewise-constant segment 0..N-1.

    ``i = floor(tau * N)`` clamped to ``[0, N-1]``. At the final RHS
    evaluation, ``tau = 1`` and ``floor(tau * N) = N``; the upper clamp maps
    that endpoint to the final segment.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    i = int(math.floor(float(tau) * N))
    if i < 0:
        return 0
    if i >= N:
        return N - 1
    return i


# ---------------------------------------------------------------------------
# Steerer
# ---------------------------------------------------------------------------


def rtn_sail_normal(
    s_hat: np.ndarray, k_hat: np.ndarray, phi: float, theta: float
) -> np.ndarray:
    """Unit sail normal from the RTN tilt construction -- the SINGLE SOURCE of the
    cruise steering geometry, shared by the steerer (``make_piecewise_rtn_steerer``)
    and the deterministic fixed-step propagator (``reflectors.cruise_propagator``).

    Parameters
    ----------
    s_hat
        Unit Sun-line (sail -> Sun). The PRIMARY axis: ``n_des`` is a small tilt off
        it, staying sunward (``n . s_hat > 0``) so the one-sided McInnes sail is lit.
    k_hat
        Unit fixed out-of-plane axis (the heliocentric orbit normal at departure).
    phi
        In-plane pitch off ``s_hat`` (radians).
    theta
        Out-of-plane tilt toward ``k_hat`` (radians).

    Both ``s_hat`` and ``k_hat`` are assumed UNIT (callers normalise once). Returns a
    unit ``n_des`` (J2000)::

        t_hat   = unit(k_hat x s_hat)                       in-plane transverse
        n_planar = cos(phi) s_hat + sin(phi) t_hat
        n_des    = unit(cos(theta) n_planar + sin(theta) k_hat)

    The ``s_hat ~parallel k_hat`` degenerate case (never reached on a real
    heliocentric arc) takes a deterministic perpendicular fallback.
    """
    s_u = np.asarray(s_hat, dtype=float)
    k_u = np.asarray(k_hat, dtype=float)
    t_hat = np.cross(k_u, s_u)
    tmag = float(np.linalg.norm(t_hat))
    if tmag < _DEGENERATE_TOL:
        # s_hat ~parallel to k_hat -- never for a real heliocentric arc;
        # deterministic perpendicular fallback (safeguard only).
        trial = np.array([1.0, 0.0, 0.0]) if abs(s_u[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
        t_hat = np.cross(trial, s_u)
        tmag = float(np.linalg.norm(t_hat))
    t_hat = t_hat / tmag
    n_planar = math.cos(phi) * s_u + math.sin(phi) * t_hat
    n_des = math.cos(theta) * n_planar + math.sin(theta) * k_u
    return n_des / float(np.linalg.norm(n_des))


def make_piecewise_rtn_steerer(
    phis: np.ndarray,
    thetas: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
):
    """Build a piecewise-constant RTN ``steering_fn`` for ``propagate_escape``.

    Parameters
    ----------
    phis, thetas
        Per-segment in-plane pitch and out-of-plane tilt (radians), length ``N``.
    et0
        Mission start ET (``tau = 0``).
    T_s
        Transit duration (s); ``tau = (et - et0) / T_s`` spans ``[0, 1]``.
    ref_normal
        FIXED out-of-plane axis ``k_hat`` (J2000), the heliocentric orbit normal
        at departure (``cruise_command.heliocentric_orbit_normal(z0)``). Need not
        be pre-normalised.

    Returns
    -------
    callable
        ``steering_fn(r, v, s_hat, p_eff, sail, current_n, *, et) -> n_des``
        (J2000 unit). Captures only float arrays + scalars -> picklable for the
        parallel finite-difference Jacobian.
    """
    phis = np.asarray(phis, dtype=float).copy()
    thetas = np.asarray(thetas, dtype=float).copy()
    if phis.shape != thetas.shape or phis.ndim != 1 or phis.size < 1:
        raise ValueError(
            f"phis and thetas must be 1-D of equal length >= 1; got {phis.shape}, {thetas.shape}"
        )
    N = int(phis.size)
    if not (T_s > 0.0):
        raise ValueError(f"T_s must be > 0, got {T_s}")
    k = np.asarray(ref_normal, dtype=float)
    kmag = float(np.linalg.norm(k))
    if kmag < _DEGENERATE_TOL:
        raise ValueError("ref_normal is the zero vector")
    k_hat = k / kmag

    def steering_fn(r, v, s_hat, p_eff, sail, current_n, *, et):
        s_u = np.asarray(s_hat, dtype=float)
        s_u = s_u / float(np.linalg.norm(s_u))  # Sun-line (toward Sun)
        tau = (et - et0) / T_s
        i = segment_index(tau, N)
        # rtn_sail_normal is the single source of the RTN tilt geometry (shared
        # with reflectors.cruise_propagator); s_u + k_hat are already unit here.
        return rtn_sail_normal(s_u, k_hat, float(phis[i]), float(thetas[i]))

    return steering_fn


def recover_phi_theta(
    n_des: np.ndarray, s_hat: np.ndarray, ref_normal: np.ndarray
) -> tuple[float, float]:
    """Invert the RTN construction: recover ``(phi, theta)`` from an inertial unit ``n_des``.

    EXACT when ``s_hat`` is perpendicular to ``ref_normal`` (the departure /
    in-plane geometry, where ``r0 . (r0 x v0) = 0``); used by the tests. Off that
    plane the renormalisation makes the inverse approximate -- but the FORWARD
    construction (and therefore the SRP physics) is exact everywhere; recovery is a
    test helper only.
    """
    n = np.asarray(n_des, dtype=float)
    n = n / float(np.linalg.norm(n))
    s_u = np.asarray(s_hat, dtype=float)
    s_u = s_u / float(np.linalg.norm(s_u))
    k = np.asarray(ref_normal, dtype=float)
    k_hat = k / float(np.linalg.norm(k))
    t_hat = np.cross(k_hat, s_u)
    t_hat = t_hat / float(np.linalg.norm(t_hat))
    theta = math.asin(max(-1.0, min(1.0, float(np.dot(n, k_hat)))))
    n_planar = n - math.sin(theta) * k_hat  # = cos(theta) * (cos phi s_hat + sin phi t_hat)
    phi = math.atan2(float(np.dot(n_planar, t_hat)), float(np.dot(n_planar, s_u)))
    return phi, theta


# ---------------------------------------------------------------------------
# Propagation + terminal miss + defect (mirror reflectors.cruise_cost)
# ---------------------------------------------------------------------------


def propagate_cruise_piecewise(
    phis: np.ndarray,
    thetas: np.ndarray,
    z0_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
    attitude_limits: Optional[AttitudeLimits] = None,
) -> EscapeResult:
    """Propagate the Sun-centred cruise under the piecewise-RTN command.

    Byte-for-byte ``cruise_cost.propagate_cruise`` with the steerer swapped:
    point-mass solar gravity (``gravity_degree=0``), the supplied third bodies,
    no atmosphere, KINEMATIC attitude, terminating at ``t_final = T_s``.
    """
    steerer = make_piecewise_rtn_steerer(phis, thetas, et0, T_s, ref_normal)
    limits = attitude_limits if attitude_limits is not None else AttitudeLimits()
    # QLawParams is required positionally but inert when a steering_fn + kinematic
    # attitude are supplied; give it valid, unused values (matches propagate_cruise).
    params = QLawParams(a_target_km=central_body.hill_radius_km, rp_min_km=1.0)
    return propagate_escape(
        z0_km_kmps,
        et0,
        sail,
        params,
        limits,
        (0.0, T_s),
        gravity_degree=0,
        central_body=central_body,
        third_bodies=tuple(third_bodies),
        steering_fn=steerer,
        kinematic_attitude=True,
        steps_per_orbit=steps_per_orbit,
        max_step_s=max_step_s,
    )


def piecewise_cruise_terminal_miss(
    phis: np.ndarray,
    thetas: np.ndarray,
    z0_km_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
) -> tuple[float, float, np.ndarray]:
    """Return ``(r_miss_km, v_miss_kmps, z_T)`` at the transit time (no verdict)."""
    run = propagate_cruise_piecewise(
        phis, thetas, z0_km_kmps, et0, T_s, ref_normal, sail, central_body,
        third_bodies, max_step_s=max_step_s, steps_per_orbit=steps_per_orbit,
    )
    z_T = np.asarray(run.orbit_state_km_kmps[-1], dtype=float)
    r_miss = float(np.linalg.norm(z_T[:3] - z_target_km_kmps[:3]))
    v_miss = float(np.linalg.norm(z_T[3:6] - z_target_km_kmps[3:6]))
    return r_miss, v_miss, z_T


def make_piecewise_cruise_defect(
    z0_km_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    N: int,
    r_scale_km: float,
    v_scale_kmps: float,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
):
    """Build the SCALED 6-vector terminal defect ``g(x)`` for the IPOPT equality
    formulation, over ``x = [phi_0..phi_{N-1}, theta_0..theta_{N-1}]`` (radians).

    Uses the scaled-defect convention of
    :func:`reflectors.cruise_cost.make_cruise_defect`. The closure is picklable
    (captures arrays and floats only) and fork-safe
    (``_ensure_worker_kernels`` per worker) for the parallel finite-difference
    Jacobian.
    """
    z0 = np.asarray(z0_km_kmps, dtype=float)
    z_tgt = np.asarray(z_target_km_kmps, dtype=float)
    ref = np.asarray(ref_normal, dtype=float)
    tb = tuple(third_bodies)
    creator_pid = os.getpid()

    def defect(x: np.ndarray) -> np.ndarray:
        _ensure_worker_kernels(creator_pid)
        x = np.asarray(x, dtype=float)
        phis, thetas = x[:N], x[N : 2 * N]
        run = propagate_cruise_piecewise(
            phis, thetas, z0, et0, T_s, ref, sail, central_body, tb,
            max_step_s=max_step_s, steps_per_orbit=steps_per_orbit,
        )
        z_T = np.asarray(run.orbit_state_km_kmps[-1], dtype=float)
        g = np.empty(6)
        g[:3] = (z_T[:3] - z_tgt[:3]) / r_scale_km
        g[3:] = (z_T[3:6] - z_tgt[3:6]) / v_scale_kmps
        if not np.all(np.isfinite(g)):
            return np.full(6, 1.0e3)
        return g

    return defect


# ---------------------------------------------------------------------------
# Node-difference smoothness regulariser for the IPOPT objective:
# w_smooth * sum((d phi)^2 + (d theta)^2) / scale^2
# ---------------------------------------------------------------------------

# The default weight is 1e-4 with a 5 deg angular scale. The decision vector is
# in radians, so the scale is radians(5).
DEFAULT_SMOOTH_WEIGHT = 1.0e-4
DEFAULT_SMOOTH_SCALE_RAD = math.radians(5.0)


def smoothness_objective(
    x: np.ndarray,
    N: int,
    *,
    w_smooth: float = DEFAULT_SMOOTH_WEIGHT,
    scale_rad: float = DEFAULT_SMOOTH_SCALE_RAD,
) -> float:
    """``w_smooth * (sum (d phi)^2 + sum (d theta)^2) / scale_rad^2`` over the angle
    blocks of ``x = [phi_0..phi_{N-1}, theta_0..theta_{N-1}, (optional extras)]``.

    Penalises node-to-node jumps -> smooth control. Any trailing decision variables
    (e.g. a variable duration ``D``) are NOT regularised. Returns 0 for ``N < 2``.
    """
    x = np.asarray(x, dtype=float)
    if N < 2:
        return 0.0
    phis, thetas = x[:N], x[N : 2 * N]
    return float(
        w_smooth * (np.sum(np.diff(phis) ** 2) + np.sum(np.diff(thetas) ** 2)) / scale_rad ** 2
    )


def smoothness_gradient(
    x: np.ndarray,
    N: int,
    *,
    w_smooth: float = DEFAULT_SMOOTH_WEIGHT,
    scale_rad: float = DEFAULT_SMOOTH_SCALE_RAD,
) -> np.ndarray:
    """Analytic gradient of :func:`smoothness_objective` (the discrete block
    Laplacian). Length ``len(x)``; trailing non-angle variables get 0.
    """
    x = np.asarray(x, dtype=float)
    g = np.zeros(x.size)
    if N < 2:
        return g
    inv = 2.0 * w_smooth / scale_rad ** 2
    for start in (0, N):
        b = x[start : start + N]
        gb = np.empty(N)
        gb[1:-1] = inv * (2.0 * b[1:-1] - b[:-2] - b[2:])
        gb[0] = inv * (b[0] - b[1])
        gb[-1] = inv * (b[-1] - b[-2])
        g[start : start + N] = gb
    return g


# ---------------------------------------------------------------------------
# Slew feasibility (post-hoc; the analytic Fourier bounds do not apply to a step
# command)
# ---------------------------------------------------------------------------


def assert_piecewise_slew_feasible(
    phis: np.ndarray,
    thetas: np.ndarray,
    T_s: float,
    limits: AttitudeLimits,
    N: Optional[int] = None,
) -> tuple[float, float]:
    """Per-node slew feasibility for the piecewise-constant kinematic command.

    The command is constant within a segment; the only slew demand is the
    reorientation at each node boundary, which must complete within one segment:

        worst_node = max_i hypot(|phi_{i+1}-phi_i|, |theta_{i+1}-theta_i|)  (rad)
        budget     = omega_max * (T_s / N)                                  (rad)

    ``hypot(dphi, dtheta)`` conservatively upper-bounds the geodesic reorientation
    of ``n_des`` across a boundary (``n_des`` is a 1-Lipschitz-or-better function of
    ``(phi, theta)``). Over a multi-month segment ``budget`` is enormous (omega_max
    ~5e-3 rad/s * ~3e6 s/segment ~ 1e4 rad), so any +/-35 deg (0.61 rad) bounded
    profile passes by ~6 orders -- a reporting gate, not a binding constraint
    (contrast the Fourier path, where slew shaped the coefficient boxes). Returns
    ``(worst_node_rad, budget_rad)``.
    """
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if N is None:
        N = int(phis.size)
    if not (T_s > 0.0):
        raise ValueError(f"T_s must be > 0, got {T_s}")
    if N < 2:
        return 0.0, float("inf")
    dpsi = np.hypot(np.abs(np.diff(phis)), np.abs(np.diff(thetas)))
    dt_seg = T_s / N
    budget = float(limits.omega_max_rad_s) * dt_seg
    worst = float(dpsi.max())
    if worst > budget:
        raise ValueError(
            f"per-node slew {worst:.3e} rad exceeds the segment budget {budget:.3e} rad "
            f"(omega_max {limits.omega_max_rad_s:.3e} rad/s * {dt_seg:.3e} s)"
        )
    return worst, budget
