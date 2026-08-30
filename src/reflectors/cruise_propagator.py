"""Deterministic fixed-step RK4 propagation for an Earth-to-Mars SRP-sail cruise.

This is the optimization propagator for the piecewise-RTN single-shooting solve.
The planetary-escape integrator uses a trajectory-dependent step grid, event
detection, and an attitude ODE, so its terminal-state map is not sufficiently
smooth for a finite-difference nonlinear-programming Jacobian.

The propagator therefore uses fixed-step RK4: the integration grid is a
deterministic function of ``(N, T_s, max_step_s)`` and is independent of the
command angles, so ``g(angles)`` is smooth by construction.
The duration ``D`` column is also smooth: each of the ``N`` equal mission-time
segments uses a uniform sub-step ``h = seg_span / ceil(seg_span/max_step_s)``.
Within a finite-difference bracket the step count is constant and ``h`` varies
continuously with ``D``.

Physics terms are reused from the escape path so the two propagators are
physically identical (an independent cross-check, ``test_cruise_propagator``):
:func:`reflectors.dynamics.two_body_acceleration` (point-mass solar gravity),
:func:`reflectors.srp.mcinnes_srp_acceleration` (the one-sided McInnes optical sail,
consistent with the escape/capture legs), :func:`reflectors.solar_constants.solar_flux_at`, and
:func:`reflectors.third_body.third_body_acceleration_from_spice`. The RTN steering
geometry comes from the single-source helper
:func:`reflectors.cruise_piecewise.rtn_sail_normal`.

Sun-central fast path: with ``central_body`` the Sun (``naif_id==10``,
``occults_sun=False``) the Sun is the frame origin, so ``s_hat = -r/|r|`` exactly and
``p_eff = solar_flux_at(|r|)`` with no shadow (verified against
``escape.py::_sun_geometry``). The two-body + SRP RHS therefore makes no SPICE calls.
SPICE is used only when ``third_bodies`` is
non-empty (Earth/Moon/Mars perturbation homotopy). This propagator supports only
Sun-central SRP geometry; a planetary-central configuration raises
``NotImplementedError``.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional, Sequence

import numpy as np

from reflectors.central_body import CentralBody
from reflectors.cruise_cost import _ensure_worker_kernels  # fork-safe kernel guard
from reflectors.cruise_piecewise import rtn_sail_normal  # single-source RTN geometry
from reflectors.dynamics import two_body_acceleration
from reflectors.ephemeris import SUN_NAIF_ID
from reflectors.solar_constants import solar_flux_at
from reflectors.srp import mcinnes_srp_acceleration
from reflectors.third_body import third_body_acceleration_from_spice

# Below this a cross product / norm is treated as degenerate (mirrors cruise_piecewise).
_DEGENERATE_TOL = 1.0e-12

# IAU 2012 astronomical unit (km), used only by the optional ideal-sail force
# model ``a_char*(AU/r)^2*cos_cone^2*n`` for optimizer conditioning comparisons.
# The default McInnes path is unaffected.
_AU_KM = 1.495978707e8


def ideal_sail_achar_kmps2(sail) -> float:
    """Characteristic acceleration (km/s^2) of an IDEAL perfectly-reflecting flat sail at 1 AU.

    ``a_c = 2 * P_0 * A / m`` with ``P_0`` the solar radiation pressure at 1 AU
    (``solar_flux_at(_AU_KM)`` ~ 4.54e-6 Pa; cf McInnes 1999, *Solar Sailing*, p.58 value
    4.56e-6 Pa) and the factor 2 the perfect-reflector momentum transfer (incident +
    specularly-reflected photon flux), per the characteristic-acceleration definition
    (McInnes 1999 sec 2.6, Eq. 2.20 / lightness number). ``A`` = ``sail.area_m2`` [m^2],
    ``m`` = ``sail.mass_kg`` [kg]; ``2 P A / m`` is m/s^2, ``* 1e-3`` -> km/s^2.

    This is consumed only by the optional ideal ``cos^2`` force in
    :func:`_make_cruise_rhs` (``ideal_achar_kmps2``), an optimizer-conditioning
    model; the physical default is the McInnes optical sail. For the canonical
    sigma=18 g/m^2 sail this returns approximately 5.045e-7 km/s^2
    (0.5045 mm/s^2).
    """
    return 2.0 * solar_flux_at(_AU_KM) * sail.area_m2 / sail.mass_kg * 1.0e-3


# ---------------------------------------------------------------------------
# Fixed-step RK4 core
# ---------------------------------------------------------------------------


def _rk4_step(rhs, t: float, y: np.ndarray, h: float, phi: float, theta: float) -> np.ndarray:
    """One classic 4-stage RK4 step of ``rhs(t, y, phi, theta) -> dy``.

    ``phi`` and ``theta`` are the constant segment angles.
    """
    k1 = rhs(t, y, phi, theta)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1, phi, theta)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2, phi, theta)
    k4 = rhs(t + h, y + h * k3, phi, theta)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _make_cruise_rhs(
    mu: float,
    k_hat: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: tuple,
    et0: float,
    ephemeris_time_direction: int,
    ideal_achar_kmps2: Optional[float] = None,
):
    """Build the cruise RHS closure ``rhs(t, y, phi, theta) -> dy`` (km, km/s).

    ``a = two_body + SRP(rtn_sail_normal(...)) + [third bodies]``. The Sun-line and SRP
    pressure are the Sun-central fast path; third bodies (if any) are clocked at
    ``et = et0 + ephemeris_time_direction * t``.

    SRP model: default is the one-sided McInnes optical sail (the physically-correct,
    escape/capture-consistent force). If ``ideal_achar_kmps2`` is given, the IDEAL
    cos^2 sail is used instead: ``a_srp = -a_char*(AU/r)^2*(n.s_hat)^2*n_des`` (force
    anti-sunward; ``-n_des`` because ``n_des`` is sunward). This optional model
    supports optimizer-conditioning comparisons; the McInnes path is the default.
    """
    sun_central = (central_body.naif_id == SUN_NAIF_ID) and (not central_body.occults_sun)
    obs = central_body.naif_id
    tb = tuple(third_bodies)

    def rhs(t: float, y: np.ndarray, phi: float, theta: float) -> np.ndarray:
        r = y[:3]
        v = y[3:6]
        a = two_body_acceleration(r, mu)
        rn = float(np.linalg.norm(r))
        if sun_central:
            # Sun at the frame origin -> sail->Sun line is -r_hat; no umbra at ~1 AU.
            s_hat = -r / rn
            p_eff = solar_flux_at(rn)
        else:  # pragma: no cover - cruise is always Sun-central
            raise NotImplementedError(
                "cruise_propagator implements only the Sun-central SRP geometry "
                "(naif_id==10, occults_sun=False)."
            )
        n_des = rtn_sail_normal(s_hat, k_hat, phi, theta)
        if ideal_achar_kmps2 is None:
            a = a + mcinnes_srp_acceleration(n_des, s_hat, p_eff, sail)
        else:
            cos_cone = float(np.dot(n_des, s_hat))  # >= 0 (sail kept lit by the cone box)
            a = a - ideal_achar_kmps2 * (_AU_KM / rn) ** 2 * cos_cone ** 2 * n_des
        if tb:
            et = et0 + ephemeris_time_direction * t
            a = a + third_body_acceleration_from_spice(r, et, tb, observer_naif_id=obs)
        out = np.empty(6)
        out[:3] = v
        out[3:] = a
        return out

    return rhs


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def propagate_cruise_clean(
    phis: np.ndarray,
    thetas: np.ndarray,
    z0_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence = (),
    *,
    max_step_s: float,
    ephemeris_time_direction: int = +1,
    return_trajectory: bool = False,
    ideal_achar_kmps2: Optional[float] = None,
):
    """Propagate the Sun-centred cruise under the piecewise-constant RTN command with a
    deterministic fixed-step RK4 (smooth ``g(x)`` by construction).

    ``ideal_achar_kmps2`` (optional): use the IDEAL cos^2 sail force with this
    characteristic acceleration (km/s^2) instead of the default McInnes optical model --
    for optimizer cold-start conditioning comparisons.

    Parameters
    ----------
    phis, thetas
        Per-segment in-plane pitch / out-of-plane tilt (radians), length ``N``.
    z0_km_kmps
        Initial heliocentric state (6,) [km, km/s].
    et0, T_s
        Mission start ET and transit duration (s). ``tau = (et-et0)/T_s`` spans [0, 1]
        over ``N`` equal segments; segment ``i`` uses ``(phis[i], thetas[i])``.
    ref_normal
        Fixed out-of-plane axis (heliocentric orbit normal at departure); need not be
        pre-normalised.
    sail, central_body, third_bodies
        The McInnes sail bus, the Sun central body, and the optional perturbers.
    max_step_s
        Absolute integration step ceiling (s); each segment is walked with a UNIFORM
        sub-step ``seg_span / ceil(seg_span/max_step_s) <= max_step_s``.
    ephemeris_time_direction
        +1 forward (default); -1 clocks the third-body ephemeris backward.
    return_trajectory
        If True, return ``(t_arr, y_arr)`` (mission-time samples + states); else the
        terminal state (6,).

    Returns
    -------
    np.ndarray
        Terminal state (6,), or ``(t_arr, y_arr)`` if ``return_trajectory``.
    """
    phis = np.asarray(phis, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if phis.ndim != 1 or thetas.shape != phis.shape or phis.size < 1:
        raise ValueError(
            f"phis and thetas must be 1-D of equal length >= 1; got {phis.shape}, {thetas.shape}"
        )
    N = int(phis.size)
    if not (T_s > 0.0):
        raise ValueError(f"T_s must be > 0, got {T_s}")
    if not (max_step_s > 0.0):
        raise ValueError(f"max_step_s must be > 0, got {max_step_s}")
    k = np.asarray(ref_normal, dtype=float)
    kmag = float(np.linalg.norm(k))
    if kmag < _DEGENERATE_TOL:
        raise ValueError("ref_normal is the zero vector")
    k_hat = k / kmag

    rhs = _make_cruise_rhs(
        central_body.mu_km3_s2, k_hat, sail, central_body, tuple(third_bodies),
        et0, ephemeris_time_direction, ideal_achar_kmps2=ideal_achar_kmps2,
    )

    seg_span = T_s / N
    y = np.asarray(z0_km_kmps, dtype=float).copy()
    if y.shape != (6,):
        raise ValueError(f"z0_km_kmps must be shape (6,), got {y.shape}")
    t = 0.0

    if return_trajectory:
        ts = [0.0]
        ys = [y.copy()]
    for i in range(N):
        phi = float(phis[i])
        theta = float(thetas[i])
        t_end = (i + 1) * seg_span
        span = t_end - t
        n_sub = max(1, int(math.ceil(span / max_step_s)))
        h = span / n_sub  # uniform sub-step <= max_step_s; continuous in T_s
        for _ in range(n_sub):
            y = _rk4_step(rhs, t, y, h, phi, theta)
            t += h
            if return_trajectory:
                ts.append(t)
                ys.append(y.copy())
        t = t_end  # pin against accumulated float drift at the segment boundary

    if return_trajectory:
        return np.asarray(ts), np.vstack(ys)
    return y


# ---------------------------------------------------------------------------
# Terminal miss + scaled defect (mirror reflectors.cruise_piecewise)
# ---------------------------------------------------------------------------


def clean_cruise_terminal_miss(
    phis: np.ndarray,
    thetas: np.ndarray,
    z0_km_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence = (),
    *,
    max_step_s: float,
    ephemeris_time_direction: int = +1,
    ideal_achar_kmps2: Optional[float] = None,
) -> tuple[float, float, np.ndarray]:
    """Return ``(r_miss_km, v_miss_kmps, z_T)`` at the transit time (no verdict).

    ``ideal_achar_kmps2`` (optional) selects the IDEAL cos^2 sail force (default None =
    McInnes), so the reported miss is computed under the same sail the caller optimised.
    """
    z_T = propagate_cruise_clean(
        phis, thetas, z0_km_kmps, et0, T_s, ref_normal, sail, central_body,
        third_bodies, max_step_s=max_step_s,
        ephemeris_time_direction=ephemeris_time_direction,
        ideal_achar_kmps2=ideal_achar_kmps2,
    )
    z_T = np.asarray(z_T, dtype=float)
    z_tgt = np.asarray(z_target_km_kmps, dtype=float)
    r_miss = float(np.linalg.norm(z_T[:3] - z_tgt[:3]))
    v_miss = float(np.linalg.norm(z_T[3:6] - z_tgt[3:6]))
    return r_miss, v_miss, z_T


def make_clean_cruise_defect(
    z0_km_kmps: np.ndarray,
    z_target_km_kmps: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence = (),
    *,
    N: int,
    r_scale_km: float,
    v_scale_kmps: float,
    max_step_s: float,
    ephemeris_time_direction: int = +1,
    ideal_achar_kmps2: Optional[float] = None,
):
    """Build the SCALED 6-vector terminal defect ``g(x)`` (FIXED endpoints / FIXED time)
    over ``x = [phi_0..phi_{N-1}, theta_0..theta_{N-1}]`` (radians) for the IPOPT
    equality formulation, using the deterministic fixed-step propagator.

    ``ideal_achar_kmps2`` (optional) selects the IDEAL cos^2 sail force (default None =
    McInnes), threaded to the propagator for cold-start optimiser conditioning.

    Picklable (captures arrays/floats only) and fork-safe (``_ensure_worker_kernels``
    per worker) for the parallel finite-difference Jacobian.
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
        z_T = propagate_cruise_clean(
            phis, thetas, z0, et0, T_s, ref, sail, central_body, tb,
            max_step_s=max_step_s, ephemeris_time_direction=ephemeris_time_direction,
            ideal_achar_kmps2=ideal_achar_kmps2,
        )
        g = np.empty(6)
        g[:3] = (z_T[:3] - z_tgt[:3]) / r_scale_km
        g[3:] = (z_T[3:6] - z_tgt[3:6]) / v_scale_kmps
        if not np.all(np.isfinite(g)):
            return np.full(6, 1.0e3)
        return g

    return defect


def make_clean_cruise_vartime_defect(
    z0_km_kmps: np.ndarray,
    dep_et: float,
    ref_normal: np.ndarray,
    sail,
    central_body: CentralBody,
    third_bodies: Sequence = (),
    *,
    N: int,
    r_scale_km: float,
    v_scale_kmps: float,
    max_step_s: float,
    target_state_fn: Callable[[float], np.ndarray],
    ephemeris_time_direction: int = +1,
    ideal_achar_kmps2: Optional[float] = None,
):
    """Build the variable-time defect over ``x = [phis, thetas, D_days]``.

    Each call recomputes ``T_s = D*86400``, propagates from ``dep_et`` for ``T_s``, and
    pulls the MOVING target at the arrival epoch via ``target_state_fn(dep_et + T_s)``
    -- so the parallel-FD Jacobian over ``[angles, D]`` captures BOTH the trajectory's
    D-sensitivity AND the target's motion with D. The target seam is a callable
    (``lambda et: body_state("MARS BARYCENTER", et, observer="SUN")[0]`` for a Mars-state
    rendezvous, or ``PhaseInterp.boundary_state`` for the capture node), keeping this
    module dependency-free of cruise_phasing/ephemeris.

    ``ideal_achar_kmps2`` (optional) selects the IDEAL cos^2 sail force (default None =
    McInnes), threaded to the propagator for cold-start optimiser conditioning.

    Picklable + fork-safe. ``D <= 0`` returns a finite penalty (``full(6, 1e3)``).
    """
    z0 = np.asarray(z0_km_kmps, dtype=float)
    ref = np.asarray(ref_normal, dtype=float)
    tb = tuple(third_bodies)
    creator_pid = os.getpid()

    def defect(x: np.ndarray) -> np.ndarray:
        _ensure_worker_kernels(creator_pid)
        x = np.asarray(x, dtype=float)
        phis, thetas, D = x[:N], x[N : 2 * N], float(x[2 * N])
        if not (D > 0.0):
            return np.full(6, 1.0e3)
        T_s = D * 86400.0
        z_tgt = np.asarray(target_state_fn(dep_et + T_s), dtype=float)
        z_T = propagate_cruise_clean(
            phis, thetas, z0, dep_et, T_s, ref, sail, central_body, tb,
            max_step_s=max_step_s, ephemeris_time_direction=ephemeris_time_direction,
            ideal_achar_kmps2=ideal_achar_kmps2,
        )
        g = np.empty(6)
        g[:3] = (z_T[:3] - z_tgt[:3]) / r_scale_km
        g[3:] = (z_T[3:6] - z_tgt[3:6]) / v_scale_kmps
        if not np.all(np.isfinite(g)):
            return np.full(6, 1.0e3)
        return g

    return defect
