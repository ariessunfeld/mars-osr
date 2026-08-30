"""Single-shooting terminal-miss cost for the interplanetary cruise.

The Earth-Hill -> Mars-Hill SRP-sail cruise is a
fixed-time, fixed-endpoint low-thrust transfer solved by DIRECT SINGLE SHOOTING:
the decision vector ``x`` is the Fourier cone/clock command (see
``reflectors.cruise_command``), and the objective is the non-dimensional terminal
MISS between the propagated state at the transit time ``T`` and the Mars capture
node. This module builds (a) the Sun-centred cruise propagation and (b) the
picklable cost closure for the DE -> FD-polish solve (``reflectors.parallel``).

The attitude is applied KINEMATICALLY (``kinematic_attitude=True``): the smooth,
slow command needs no bang-bang slew tracker, and
slew-feasibility is enforced separately and exactly by
``cruise_command.assert_cruise_slew_feasible`` / ``feasible_coeff_boxes`` (the
optimizer's box bounds), not by the integrated plant.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from reflectors.attitude_control import AttitudeLimits
from reflectors.central_body import CentralBody
from reflectors.cruise_command import make_cruise_command_steerer
from reflectors.escape import EscapeResult, propagate_escape
from reflectors.qlaw import QLawParams


# SPICE fork-safety: a forked DE worker inherits the parent's furnished kernel
# pool but SHARES its DAF file descriptors, so concurrent spkezr reads across
# workers corrupt one another (SPICE(DAFFRNOTFOUND/DAFNOSUCHHANDLE)). Each
# worker must furnish its OWN private handles once (kclear + furnsh). This set
# tracks which PIDs have done so; the parent (cost creator) is exempt -- it
# already has valid handles.
_KERNELS_LOADED_PIDS: set[int] = set()


def _ensure_worker_kernels(creator_pid: int) -> None:
    pid = os.getpid()
    if pid == creator_pid or pid in _KERNELS_LOADED_PIDS:
        return
    from reflectors.kernels import load_kernels

    load_kernels()  # kclear + furnsh -> private DAF handles for this process
    _KERNELS_LOADED_PIDS.add(pid)


def propagate_cruise(
    coeffs: np.ndarray,
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
    """Propagate the Sun-centred cruise under the time-Fourier cone/clock command.

    Point-mass solar gravity (``gravity_degree=0``), the supplied third bodies
    (Earth/Moon/Mars), no atmosphere, KINEMATIC attitude. The default Sun-central
    terminal events (a ~1e12 km ceiling and an R_sun floor) never fire over a
    heliocentric transfer, so the run terminates at ``t_final = T_s``.
    """
    steerer = make_cruise_command_steerer(coeffs, et0, T_s, ref_normal)
    limits = attitude_limits if attitude_limits is not None else AttitudeLimits()
    # QLawParams is required positionally but is unused when a steering_fn is
    # supplied and the attitude is kinematic; give it inert, valid values.
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


def cruise_terminal_miss(
    coeffs: np.ndarray,
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
    """Return ``(r_miss_km, v_miss_kmps, z_T)`` at the transit time.

    ``z_T`` is the full terminal state (6,). Dimensional miss components -- the
    raw quantities to report (no verdict); the non-dimensional scalar cost is
    :func:`make_cruise_terminal_miss_cost`.
    """
    run = propagate_cruise(
        coeffs, z0_km_kmps, et0, T_s, ref_normal, sail, central_body,
        third_bodies, max_step_s=max_step_s, steps_per_orbit=steps_per_orbit,
    )
    z_T = np.asarray(run.orbit_state_km_kmps[-1], dtype=float)
    r_miss = float(np.linalg.norm(z_T[:3] - z_target_km_kmps[:3]))
    v_miss = float(np.linalg.norm(z_T[3:6] - z_target_km_kmps[3:6]))
    return r_miss, v_miss, z_T


def make_cruise_terminal_miss_cost(
    z0_km_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    r_scale_km: float,
    v_scale_kmps: float,
    w_v: float = 1.0,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
):
    """Build the non-dimensional terminal-miss cost ``f(x) -> float``.

    ``cost = ||r(T) - r_tgt|| / r_scale  +  w_v * ||v(T) - v_tgt|| / v_scale``.

    Non-dimensionalised by ``r_scale_km`` (e.g. 1 AU) and ``v_scale_kmps`` (e.g.
    a heliocentric orbital speed) so the position and velocity residuals are
    comparable; ``w_v`` reweights velocity vs position. The returned closure
    captures only picklable data (arrays + frozen ``CentralBody`` /
    ``ThirdBody`` / sail), so it works with ``reflectors.parallel.CloudpickleMap``
    for the DE / parallel-FD solve.
    """
    z0 = np.asarray(z0_km_kmps, dtype=float)
    z_tgt = np.asarray(z_target_km_kmps, dtype=float)
    ref = np.asarray(ref_normal, dtype=float)
    tb = tuple(third_bodies)
    creator_pid = os.getpid()

    def cost(x: np.ndarray) -> float:
        _ensure_worker_kernels(creator_pid)
        r_miss, v_miss, _ = cruise_terminal_miss(
            np.asarray(x, dtype=float), z0, z_tgt, et0, T_s, ref, sail,
            central_body, tb, max_step_s=max_step_s, steps_per_orbit=steps_per_orbit,
        )
        if not (np.isfinite(r_miss) and np.isfinite(v_miss)):
            return 1.0e6  # infeasible candidate -> large finite penalty for DE
        return r_miss / r_scale_km + w_v * v_miss / v_scale_kmps

    return cost


def make_cruise_defect(
    z0_km_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    r_scale_km: float,
    v_scale_kmps: float,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
):
    """Build the SCALED 6-vector terminal defect ``g(x) = [(r(T)-r_tgt)/r_scale,
    (v(T)-v_tgt)/v_scale]`` for the IPOPT equality-constraint formulation.

    Unlike :func:`make_cruise_terminal_miss_cost` (a scalar weighted norm), this
    returns the full signed, non-dimensional residual vector so a gradient-based
    NLP solver (cyipopt) can drive each component to zero -- far better
    conditioned than minimizing a weighted sum on the ~1-rev single-shooting map.
    Picklable (per-worker kernel ensure) for the
    parallel finite-difference Jacobian."""
    z0 = np.asarray(z0_km_kmps, dtype=float)
    z_tgt = np.asarray(z_target_km_kmps, dtype=float)
    ref = np.asarray(ref_normal, dtype=float)
    tb = tuple(third_bodies)
    creator_pid = os.getpid()

    def defect(x: np.ndarray) -> np.ndarray:
        _ensure_worker_kernels(creator_pid)
        run = propagate_cruise(
            np.asarray(x, dtype=float), z0, et0, T_s, ref, sail, central_body,
            tb, max_step_s=max_step_s, steps_per_orbit=steps_per_orbit,
        )
        z_T = np.asarray(run.orbit_state_km_kmps[-1], dtype=float)
        g = np.empty(6)
        g[:3] = (z_T[:3] - z_tgt[:3]) / r_scale_km
        g[3:] = (z_T[3:6] - z_tgt[3:6]) / v_scale_kmps
        if not np.all(np.isfinite(g)):
            return np.full(6, 1.0e3)
        return g

    return defect


def make_free_departure_defect(
    r0_km: np.ndarray,
    v_earth_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    order: int,
    r_scale_km: float,
    v_scale_kmps: float,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
):
    """Scaled 6-vector terminal defect over (cone/clock coeffs + departure v_inf)
    for the IPOPT free-departure solve. x = ``[coeffs (2+4K), v_inf_x,y,z]``;
    ``z0 = [r0, v_earth + v_inf]``; the clock reference is the departure orbit
    normal of each candidate. Picklable for the parallel FD Jacobian."""
    from reflectors.cruise_command import heliocentric_orbit_normal

    r0 = np.asarray(r0_km, dtype=float)
    v_e = np.asarray(v_earth_kmps, dtype=float)
    z_tgt = np.asarray(z_target_km_kmps, dtype=float)
    tb = tuple(third_bodies)
    n_coeffs = 2 + 4 * order
    creator_pid = os.getpid()

    def defect(x: np.ndarray) -> np.ndarray:
        _ensure_worker_kernels(creator_pid)
        x = np.asarray(x, dtype=float)
        coeffs = x[:n_coeffs]
        v_inf = x[n_coeffs:n_coeffs + 3]
        z0 = np.concatenate([r0, v_e + v_inf])
        ref = heliocentric_orbit_normal(z0)
        run = propagate_cruise(
            coeffs, z0, et0, T_s, ref, sail, central_body, tb,
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


def make_free_departure_cost(
    r0_km: np.ndarray,
    v_earth_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence,
    *,
    order: int,
    r_scale_km: float,
    v_scale_kmps: float,
    w_v: float = 1.0,
    max_step_s: Optional[float] = None,
    steps_per_orbit: int = 200,
):
    """Cost over cone/clock coefficients and departure excess velocity.

    The Earth-escape exit delivers a fixed (retrograde/out-of-plane) v_inf; this
    cost FREES the Earth-relative excess velocity (last 3 elements of x) so the
    optimiser can choose an ideal injection from the same departure position.
    This sensitivity formulation separates departure-injection constraints from
    sail-control authority. The ``x`` layout is:
    ``[a0,...,clock harmonics (2+4K), v_inf_x, v_inf_y, v_inf_z]``. The
    heliocentric departure is ``z0 = [r0, v_earth + v_inf]`` and the clock
    reference normal is recomputed from each candidate's departure orbit plane.
    """
    from reflectors.cruise_command import heliocentric_orbit_normal

    r0 = np.asarray(r0_km, dtype=float)
    v_e = np.asarray(v_earth_kmps, dtype=float)
    z_tgt = np.asarray(z_target_km_kmps, dtype=float)
    tb = tuple(third_bodies)
    n_coeffs = 2 + 4 * order
    creator_pid = os.getpid()

    def cost(x: np.ndarray) -> float:
        _ensure_worker_kernels(creator_pid)
        x = np.asarray(x, dtype=float)
        coeffs = x[:n_coeffs]
        v_inf = x[n_coeffs:n_coeffs + 3]
        z0 = np.concatenate([r0, v_e + v_inf])
        ref = heliocentric_orbit_normal(z0)
        r_miss, v_miss, _ = cruise_terminal_miss(
            coeffs, z0, z_tgt, et0, T_s, ref, sail, central_body, tb,
            max_step_s=max_step_s, steps_per_orbit=steps_per_orbit,
        )
        if not (np.isfinite(r_miss) and np.isfinite(v_miss)):
            return 1.0e6
        return r_miss / r_scale_km + w_v * v_miss / v_scale_kmps

    return cost
