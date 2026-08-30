"""scripts/illumination_window_feasibility.py — analytical α_max feasibility per window.

Lightweight CLI that answers: given altitude (or K orbits per Mars-solar-sol)
+ initial mean anomaly + an α_max slew budget + target + epoch, propagate one
(or N) Mars-solar-sols with J_2 + Sun third body (NO SRP, NO sail dynamics —
geometry-only), find all delivery windows, and tag each FEAS / INFEAS based
on whether the window's peak |omega_dot_bisector| is at or below alpha_max.

The α_max feasibility is purely geometric: the bisector pointing direction
n_hat_bisector(t) is determined by Sun, spacecraft, and target positions, so
its angular acceleration profile depends only on orbital geometry, NOT on sail
mass loading (sigma) or cruise law. A window with peak |omega_dot| > alpha_max
cannot be tracked within that slew budget, independent of the cruise law.

What this script DOES NOT model (intentional — geometric ceiling only):
  - The slew from cruise-end attitude into the bisector at window start.
    A FEAS-stamped window may still be unreachable if a particular cruise
    law lands far from n_hat_bisector(t_start) and the slew at alpha_max
    consumes the window. A propagation that includes the cruise law is
    required to assess this transition.
  - Multi-sol secular SRP drift (set --n-sols=1 default; longer horizons
    accumulate ~km-scale drifts at sigma=18 g/m^2 in alpha=0).
  - Atmospheric scattering, target BRDF, sail thermal limits.

Fluence numbers in the per-window table are computed at a CANONICAL sail
(sigma = --sigma-kg-per-m2 default 0.018 kg/m^2; canonical specular reflectance
via reflectors.sail_designs.make_canonical_sail) for reference. They scale
linearly with sail area in the pinhole regime.
The α_max FEAS / INFEAS verdict is sigma-independent by construction.

Sun-sync inclination is derived from altitude via
``reflectors.sun_sync.sun_sync_inclination_rad``. With --K (default 12), the
altitude is itself derived via ``repeat_ground_track_altitude`` for an exact
repeat-ground-track sun-sync orbit (~509 km at K=12). With --altitude-km the
caller pins altitude directly; sun-sync inclination is still solved at that
altitude (general altitudes do not give an integer K orbits-per-sol repeat).

CLI examples
------------

  Default (perihelion 2028, K=12, M0=0, alpha_max=0.003 deg/s^2,
  target 40N/200E, 1 sol):

      python scripts/illumination_window_feasibility.py

  Aphelion 2029 sweep over a couple M0 anchors (use a shell loop or the
  sister utility scripts/longitudinal_phasing_sweep.py for grid sweeps):

      python scripts/illumination_window_feasibility.py \\
          --epoch-utc 2029-01-20T00:21:07.201 \\
          --M0-deg 100 \\
          --alpha-max-deg-s2 0.001

  Pin altitude directly (off-K, sun-sync inclination only):

      python scripts/illumination_window_feasibility.py \\
          --altitude-km 600 --M0-deg 0

References (cited modules)
--------------------------

reflectors.sun_sync.repeat_ground_track_altitude  — Mars synodic identity:
    K orbits per solar sol -> a; cf. Vallado 2013 Sec. 11.4.
reflectors.visibility.find_delivery_windows       — gates + per-window
    peak_alpha_demand_rad_s2 via FD on bisector_pointing direction sampled
    along trajectory_interpolant (cubic spline, sub-sample dt).
reflectors.beam (Canady-Allen 1982 Eq. 9, Celik-McInnes 2022 pinhole
    regime) — fluence_J_per_m2 photometry per window.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from scipy.optimize import brentq, minimize

import spiceypy as spice

from reflectors.attitude import sun_pointing
from reflectors.dynamics import PropagationOptions, propagate
from reflectors.elements import state_from_classical_mme2000
from reflectors.ephemeris import AU_KM, body_state, utc_to_et
from reflectors.kernels import load_kernels
from reflectors.mars_constants import (
    MARS_SIDEREAL_YEAR_S,
    SECONDS_PER_SOLAR_SOL_S,
)
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import (
    initial_state_j2000,
    raan_mme2000_from_ltan,
    repeat_ground_track_altitude,
    sun_sync_inclination_rad,
)
from reflectors.termination import AltitudeFloor
from reflectors.third_body import sun_third_body
from reflectors.visibility import find_delivery_windows


logger = logging.getLogger(__name__)

# Defaults use the perihelion-2028 design point;
# refined a* derived from K=12; LTAN 18 h; target 40N/200E; sigma=18 g/m^2;
# alpha_max = 0.003 deg/s^2).
DEFAULT_K_ORBITS_PER_SOL = 12
DEFAULT_EPOCH_UTC = "2028-02-11T12:42:00"     # Mars perihelion 2028
DEFAULT_LTAN_H = 18.0                          # dawn-dusk sun-sync
DEFAULT_M0_DEG = 0.0
DEFAULT_TARGET_LAT_DEG = 40.0
DEFAULT_TARGET_LON_DEG = 200.0
DEFAULT_ALPHA_MAX_DEG_PER_S2 = 0.003           # 5.236e-5 rad/s^2
DEFAULT_SIGMA_KG_PER_M2 = 0.018
DEFAULT_ATMOSPHERIC_TRANSMISSION = 1.0
DEFAULT_ALTITUDE_FLOOR_KM = 300.0
DEFAULT_ELEV_MIN_DEG = 10.0
DEFAULT_BISECTOR_COS_ALPHA_MIN = 0.1
DEFAULT_T_EVAL_CADENCE_S = 60.0
DEFAULT_N_SOLS = 1


@dataclass(frozen=True)
class WindowRow:
    """One row of the per-window output table."""
    sol_index: int                 # 1-based: which sol the window lives in
    window_index_in_sol: int       # 1-based within that sol
    t_start_s: float               # rel epoch_et
    t_end_s: float                 # rel epoch_et
    duration_s: float
    peak_alpha_demand_rad_s2: Optional[float]
    peak_irradiance_W_per_m2: Optional[float]
    fluence_J_per_m2: Optional[float]
    min_slant_range_km: float
    max_elevation_deg: float
    feasible: Optional[bool]       # peak <= alpha_max; None when peak is None


def _self_consistent_repeat_ground_track_altitude(
    K: int,
    *,
    mu_km3_s2: float,
    ref_radius_km: float,
    J2: float,
    solar_sol_s: float = SECONDS_PER_SOLAR_SOL_S,
    rtol: float = 1e-12,
) -> Tuple[float, float, float]:
    """Self-consistent (a_km, i_rad, T_nodal_s) for K orbits per Mars-solar-sol.

    Solves jointly:
      (a) sun-sync condition  d(RAAN)/dt = 2 pi / T_year         (Brouwer J_2)
      (b) repeat-ground-track K * T_nodal = T_solar_sol

    where ``T_nodal`` is the J_2-corrected draconic (nodal) period — the
    time between consecutive ascending-node crossings — rather than the
    Keplerian period. For e = 0 the secular-rate sum (Brouwer / Vallado
    2013, 4th ed Sec. 9.7.1) gives the argument-of-latitude rate

        u_dot = n * [ 1 + (3/2) J_2 (R/a)^2 (3 - 4 sin^2 i) ]

    and ``T_nodal = 2 pi / u_dot``.

    The closed-form sister :func:`reflectors.sun_sync.repeat_ground_track_altitude`
    instead solves ``K * T_kepler = T_solar_sol`` (Kepler third law alone),
    which omits the J_2 nodal-period correction. The correction is sub-
    percent at Mars sun-sync altitudes — at K=12 it shifts the altitude
    by ~5-6 km — but the analytical-feasibility script wants exact
    agreement between the orbit it constructs and the orbit the
    propagator integrates with J_2 active.

    The result is **purely analytical (J_2 only)** and intentionally
    distinct from the higher-fidelity ``A_REFINED_KM = 3903.924477`` value,
    obtained by minimising ``||delta_r_iau||`` under degree-6 gravity +
    Phobos + Deimos + Sun + SRP — physics that this analytical script
    does NOT model.

    Parameters / Returns
    --------------------
    See :func:`reflectors.sun_sync.repeat_ground_track_altitude` (same
    contract; ``T_orb_s`` here is ``T_nodal_s``).

    References
    ----------
    Vallado, D.A. (2013), *Fundamentals of Astrodynamics and
    Applications*, 4th ed., Microcosm Press:
      Sec. 9.7.1  -- secular rates for omega, M (e=0 simplifications)
      Sec. 11.4   -- repeating ground tracks; nodal period vs Keplerian
    Brouwer, D. (1959), Solution of the problem of artificial satellite
    theory without drag, *Astron. J.*, 64, 378-396.
    """
    if K < 1 or not isinstance(K, (int, np.integer)):
        raise ValueError(f"K must be integer >= 1; got {K!r}")

    def residual(a_km: float) -> float:
        """K * T_nodal(a, i_ss(a)) - T_solar_sol; root at self-consistent a."""
        i_rad = sun_sync_inclination_rad(
            a_km,
            mu_km3_s2=mu_km3_s2,
            ref_radius_km=ref_radius_km,
            J2=J2,
        )
        sin2_i = math.sin(i_rad) ** 2
        n = math.sqrt(mu_km3_s2 / a_km ** 3)
        u_dot_factor = (
            1.0
            + 1.5 * J2 * (ref_radius_km / a_km) ** 2 * (3.0 - 4.0 * sin2_i)
        )
        if u_dot_factor <= 0.0:
            # This branch is unreachable across the supported sun-sync
            # altitude range
            # (sin^2 i ~ 1 at Mars sun-sync, J_2(R/a)^2 ~ 1e-3 → factor ≈ 0.998).
            raise ValueError(
                f"u_dot non-positive at a_km={a_km}: factor={u_dot_factor}"
            )
        T_nodal = 2.0 * math.pi / (n * u_dot_factor)
        return float(K) * T_nodal - float(solar_sol_s)

    # Bracket: Kepler-only a +/- 1% — the J_2 correction is well under that.
    a_kepler, _, _ = repeat_ground_track_altitude(
        int(K),
        mu_km3_s2=mu_km3_s2,
        ref_radius_km=ref_radius_km,
        J2=J2,
        solar_sol_s=solar_sol_s,
    )
    a_lo = a_kepler * 0.99
    a_hi = a_kepler * 1.01
    f_lo = residual(a_lo)
    f_hi = residual(a_hi)
    if f_lo * f_hi > 0.0:
        # If the initial interval does not bracket a root, widen it once.
        a_lo = a_kepler * 0.95
        a_hi = a_kepler * 1.05
        f_lo = residual(a_lo)
        f_hi = residual(a_hi)
        if f_lo * f_hi > 0.0:
            raise RuntimeError(
                f"could not bracket self-consistent a at K={K}; "
                f"residual({a_lo})={f_lo}, residual({a_hi})={f_hi}"
            )

    a_sc = float(brentq(residual, a_lo, a_hi, rtol=rtol))
    i_ss = sun_sync_inclination_rad(
        a_sc, mu_km3_s2=mu_km3_s2, ref_radius_km=ref_radius_km, J2=J2,
    )
    sin2_i = math.sin(i_ss) ** 2
    n = math.sqrt(mu_km3_s2 / a_sc ** 3)
    u_dot_factor = (
        1.0 + 1.5 * J2 * (ref_radius_km / a_sc) ** 2 * (3.0 - 4.0 * sin2_i)
    )
    T_nodal_s = 2.0 * math.pi / (n * u_dot_factor)
    return a_sc, float(i_ss), float(T_nodal_s)


def _self_consistent_numerical_repeat_ground_track_altitude(
    K: int,
    *,
    mu_km3_s2: float,
    ref_radius_km: float,
    J2: float,
    epoch_et: float,
    ltan_h: float,
    M0_deg: float,
    altitude_floor_km: float,
    t_eval_cadence_s: float,
    include_sun_third_body: bool,
    a_box_km: float = 30.0,
    i_box_deg: float = 2.0,
    nm_xatol: float = 1e-3,        # 1 m on a, 0.001 deg on i
    nm_fatol: float = 1e-9,
    nm_maxiter: int = 80,
) -> Tuple[float, float, float]:
    """Numerical self-consistent (a_km, i_rad, T_solar_s) for K orbits per sol.

    Solves jointly under FULL numerical propagation (NOT analytical
    first-order approximation):

      (a) sun-sync condition: orbit's body-fixed sub-satellite ground
          track returns to its starting point after exactly one Mars
          mean solar day.
      (b) repeat-ground-track: K orbital revolutions complete in that
          interval.

    Both conditions are enforced jointly by 2D Nelder-Mead minimization
    of body-fixed closure cost

        cost(a, i) = (||Δr_iau||/1000 km)^2 + (||Δv_iau||/0.0008 km/s)^2

    over (a, i), where Δr_iau, Δv_iau are the IAU_MARS-frame end-minus-
    start state differences after one Mars-solar-sol propagation at α=0
    sun-pointing attitude.

    The propagator config matches the script's main propagation
    (zonal_degree=2 + optional Sun third body, NO SRP), so the
    constructed orbit is internally consistent with what the script
    then propagates downstream — closure residual at the returned
    (a*, i*) is sub-meter under the same physics.

    Distinct from
    :func:`_self_consistent_repeat_ground_track_altitude` (the Brouwer
    analytical first-order J_2 solver), which is approximate at the
    ~7.6 km level versus the numerical result at K=12.
    The numerical solver here is preferred for higher accuracy; the
    analytical one is available via
    ``--solver analytical-j2`` for speed (1 ms vs ~10 s).

    Accuracy notes
    --------------

    The 8.5 km analytical-to-numerical gap is dominated by the ~7.6 km
    Brouwer J_2 approximation at K=12; tesserals contribute -0.24 km, while
    third bodies and SRP are imperceptible. The synodic identity includes the
    RAAN-precession correction in ``T_solar_mean``; the equation of time does
    not affect the repeat-ground-track target.

    Returns
    -------
    a_km, i_rad, T_solar_s
        Same contract as the analytical sister functions. ``T_solar_s``
        is just ``SECONDS_PER_SOLAR_SOL_S`` (exact closure target);
        actual orbital period under propagation is implicit.
    """
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError(f"K must be integer >= 1; got {K!r}")

    # J_2-analytical baseline — fast, near the true minimum, used as
    # warm-start AND as the bounds anchor.
    a_J2, i_J2_rad, _ = _self_consistent_repeat_ground_track_altitude(
        K, mu_km3_s2=mu_km3_s2, ref_radius_km=ref_radius_km, J2=J2,
    )
    i_J2_deg = math.degrees(i_J2_rad)
    bounds = [
        (a_J2 - a_box_km, a_J2 + a_box_km),
        (i_J2_deg - i_box_deg, i_J2_deg + i_box_deg),
    ]

    duration_s = float(SECONDS_PER_SOLAR_SOL_S)
    t_eval = np.arange(0.0, duration_s + 0.1, t_eval_cadence_s)
    if t_eval[-1] < duration_s:
        # Append the exact T_solar endpoint so the closure metric
        # measures at the true sub-satellite repeat instant. Without
        # this, the last sample is up to ``t_eval_cadence_s`` seconds
        # short of T_solar, which biases the optimum by tens of meters
        # to ~1 km depending on cadence.
        t_eval = np.append(t_eval, duration_s)

    third_bodies_factory = (
        (lambda: [sun_third_body()]) if include_sun_third_body else (lambda: [])
    )
    altitude_floor_obj = AltitudeFloor.at_km(
        float(altitude_floor_km), label="altitude_floor"
    )

    def _eval_closure(a_km: float, i_deg: float) -> Tuple[float, float]:
        """One propagation; returns (||Δr_iau||_km, ||Δv_iau||_kmps)."""
        raan_rad = raan_mme2000_from_ltan(ltan_h, epoch_et)
        state0 = state_from_classical_mme2000(
            a_km=a_km,
            e=0.0,
            inclination_rad=math.radians(i_deg),
            raan_rad=raan_rad,
            argp_rad=0.0,
            nu_rad=math.radians(M0_deg),
            mu_km3_s2=mu_km3_s2,
            epoch_et=epoch_et,
        )
        result = propagate(
            state0_km_kmps=state0,
            t_span_s=(0.0, duration_s),
            epoch_et=epoch_et,
            zonal_degree=2,
            gravity_degree=0,
            third_bodies=third_bodies_factory(),
            solar_sail=None,
            sail_normal=None,
            altitude_floor=altitude_floor_obj,
            options=PropagationOptions.fast(),
            t_eval_s=t_eval,
        )
        et_start = epoch_et + float(result.t_s[0])
        et_end = epoch_et + float(result.t_s[-1])
        T6_start = np.asarray(spice.sxform("J2000", "IAU_MARS", et_start),
                              dtype=float)
        T6_end = np.asarray(spice.sxform("J2000", "IAU_MARS", et_end),
                            dtype=float)
        s_iau_start = T6_start @ np.asarray(
            result.state_km_kmps[0], dtype=float
        )
        s_iau_end = T6_end @ np.asarray(
            result.state_km_kmps[-1], dtype=float
        )
        dr = s_iau_end[:3] - s_iau_start[:3]
        dv = s_iau_end[3:] - s_iau_start[3:]
        return (
            float(np.linalg.norm(dr)),
            float(np.linalg.norm(dv)),
        )

    SCALE_DR_KM = 1000.0
    SCALE_DV_KMPS = 0.8

    def cost_fn(x: np.ndarray) -> float:
        a, i = float(x[0]), float(x[1])
        try:
            dr_km, dv_kmps = _eval_closure(a, i)
        except Exception:
            return 1e12
        return ((dr_km / SCALE_DR_KM) ** 2
                + (dv_kmps / SCALE_DV_KMPS) ** 2)

    x0 = np.array([a_J2, i_J2_deg])
    initial_simplex = np.array([
        [a_J2,        i_J2_deg],
        [a_J2 + 0.5,  i_J2_deg],
        [a_J2,        i_J2_deg + 0.05],
    ])
    nm = minimize(
        cost_fn, x0, method="Nelder-Mead",
        bounds=bounds,
        options={
            "initial_simplex": initial_simplex,
            "maxiter": nm_maxiter,
            "xatol": nm_xatol,
            "fatol": nm_fatol,
            "disp": False,
        },
    )
    a_star = float(nm.x[0])
    i_star_deg = float(nm.x[1])
    return a_star, math.radians(i_star_deg), duration_s


def _resolve_altitude(
    *,
    K_orbits_per_sol: Optional[int],
    altitude_km_override: Optional[float],
    solver_method: str,
    epoch_et: float,
    ltan_h: float,
    M0_deg: float,
    altitude_floor_km: float,
    t_eval_cadence_s: float,
    include_sun_third_body: bool,
) -> Tuple[float, float, Optional[int], Optional[float], float]:
    """Resolve (a_km, R_mars_km, K_used, T_orb_s, i_rad).

    Returns the semi-major axis (km), the MRO120F reference radius used
    for altitude annotation, K (or ``None`` when an explicit altitude
    override disabled the K-derived path), the orbital period
    (``T_nodal_s`` for analytical-J_2, ``T_kepler_s`` for kepler,
    ``T_solar_s`` for numerical, or ``None`` for override), and the
    inclination for the selected configuration in radians.

    Solver methods:
      "numerical"    — 2D NM on (a, i), full propagation, closure cost.
                       Most accurate. ~10 s wall.
      "analytical-j2" — Brouwer first-order J_2 self-consistent analytical
                       solve. ~1 ms. Approximate at ~7.6 km level.
      "kepler"       — Kepler third law alone, ignores J_2 nodal-period
                       correction. ~5 km off the J_2 self-consistent
                       answer at K=12.
    """
    # Pull the same MRO120F (mu, R_ref, J_2) the sun-sync routines use
    # so the altitude annotation matches the orbit construction
    # self-consistently.
    from reflectors.gravity import mars_gravity_model, zonal_coefficients
    model = mars_gravity_model(max_degree=2)
    mu = float(model.mu_km3_s2)
    R_mars = float(model.ref_radius_km)
    J2 = float(zonal_coefficients(model, 2)[2])

    if altitude_km_override is not None:
        if altitude_km_override <= 0.0:
            raise ValueError(
                f"--altitude-km must be > 0; got {altitude_km_override}"
            )
        a_km = float(altitude_km_override) + R_mars
        # Explicit-altitude mode evaluates the analytical J_2 sun-synchronous
        # inclination rather than a K-targeted closure optimum.
        i_rad = sun_sync_inclination_rad(a_km)
        return a_km, R_mars, None, None, i_rad

    if K_orbits_per_sol is None:
        K_orbits_per_sol = DEFAULT_K_ORBITS_PER_SOL
    K = int(K_orbits_per_sol)

    if solver_method == "kepler":
        a_km, i_rad, T_orb_s = repeat_ground_track_altitude(K)
    elif solver_method == "analytical-j2":
        a_km, i_rad, T_orb_s = _self_consistent_repeat_ground_track_altitude(
            K, mu_km3_s2=mu, ref_radius_km=R_mars, J2=J2,
        )
    elif solver_method == "numerical":
        a_km, i_rad, T_orb_s = (
            _self_consistent_numerical_repeat_ground_track_altitude(
                K,
                mu_km3_s2=mu,
                ref_radius_km=R_mars,
                J2=J2,
                epoch_et=epoch_et,
                ltan_h=ltan_h,
                M0_deg=M0_deg,
                altitude_floor_km=altitude_floor_km,
                t_eval_cadence_s=t_eval_cadence_s,
                include_sun_third_body=include_sun_third_body,
            )
        )
    else:
        raise ValueError(
            f"unknown solver_method {solver_method!r}; expected one of: "
            "'numerical', 'analytical-j2', 'kepler'"
        )
    return float(a_km), R_mars, K, float(T_orb_s), float(i_rad)


def _propagate_one_sol(
    *,
    state0: np.ndarray,
    sol_index: int,             # 0-based offset in sols past epoch_et
    epoch_et: float,
    altitude_floor_km: float,
    t_eval_cadence_s: float,
    include_sun_third_body: bool,
):
    """Propagate one Mars-solar-sol with J2 + (optional) Sun third body.

    NO SRP, NO sail. Returns the PropagationResult for the sol.
    """
    duration_s = float(SECONDS_PER_SOLAR_SOL_S)
    t_eval = np.arange(0.0, duration_s + 0.1, t_eval_cadence_s)
    if t_eval[-1] < duration_s:
        t_eval = np.append(t_eval, duration_s)

    third_bodies = [sun_third_body()] if include_sun_third_body else []
    epoch_et_sol = float(epoch_et) + sol_index * duration_s

    return propagate(
        state0_km_kmps=state0,
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et_sol,
        zonal_degree=2,
        gravity_degree=0,
        third_bodies=third_bodies,
        solar_sail=None,         # NO SRP — geometric ceiling
        sail_normal=None,
        altitude_floor=AltitudeFloor.at_km(
            float(altitude_floor_km), label="altitude_floor"
        ),
        options=PropagationOptions.fast(),
        t_eval_s=t_eval,
    )


def evaluate(
    *,
    a_km: float,
    ltan_h: float,
    M0_deg: float,
    epoch_utc: str,
    target_lat_deg: float,
    target_lon_deg: float,
    alpha_max_rad_s2: float,
    sigma_kg_per_m2: float,
    atmospheric_transmission: float,
    elev_min_deg: float,
    bisector_cos_alpha_min: float,
    altitude_floor_km: float,
    t_eval_cadence_s: float,
    n_sols: int,
    include_sun_third_body: bool,
) -> Tuple[List[WindowRow], dict]:
    """Run the analytical feasibility evaluation across n_sols Mars-solar-sols.

    Returns (rows, summary_dict).
    """
    epoch_et = utc_to_et(epoch_utc)
    state0 = initial_state_j2000(
        a_km=a_km,
        ltan_h=ltan_h,
        M0_rad=math.radians(M0_deg),
        epoch_et=epoch_et,
    )

    # Photometry sail (canonical reflectance + area; sigma scales mass only,
    # which is irrelevant here because SRP is off in the propagation).
    sail = make_canonical_sail(float(sigma_kg_per_m2))

    rows: List[WindowRow] = []
    state = state0.copy()
    total_pre_filter = 0
    total_feasible = 0
    total_fluence_feasible = 0.0
    total_fluence_all = 0.0

    for sol_idx in range(int(n_sols)):
        result = _propagate_one_sol(
            state0=state,
            sol_index=sol_idx,
            epoch_et=epoch_et,
            altitude_floor_km=altitude_floor_km,
            t_eval_cadence_s=t_eval_cadence_s,
            include_sun_third_body=include_sun_third_body,
        )

        # All geometric windows + their per-window peak_alpha_demand_rad_s2
        # (no alpha_max filter, allowing inspection of the demand profile).
        windows = find_delivery_windows(
            result,
            target_lat_deg, target_lon_deg,
            target_elevation_min_deg=elev_min_deg,
            bisector_cos_alpha_min=bisector_cos_alpha_min,
            require_sail_sunlit=True,
            require_sail_above_horizon=True,
            require_bisector_feasible=True,
            sail=sail,
            atmospheric_transmission=atmospheric_transmission,
            alpha_max_rad_s2=None,
        )

        total_pre_filter += len(windows)
        for j, w in enumerate(windows, start=1):
            peak = w.peak_alpha_demand_rad_s2
            if peak is None:
                feas = None
            else:
                feas = bool(peak <= alpha_max_rad_s2)
                if feas:
                    total_feasible += 1

            flu = w.fluence_J_per_m2 or 0.0
            total_fluence_all += flu
            if feas is True:
                total_fluence_feasible += flu

            rows.append(WindowRow(
                sol_index=sol_idx + 1,
                window_index_in_sol=j,
                t_start_s=float(w.t_start_s),
                t_end_s=float(w.t_end_s),
                duration_s=float(w.duration_s),
                peak_alpha_demand_rad_s2=(
                    None if peak is None else float(peak)
                ),
                peak_irradiance_W_per_m2=(
                    None if w.peak_irradiance_W_per_m2 is None
                    else float(w.peak_irradiance_W_per_m2)
                ),
                fluence_J_per_m2=(
                    None if w.fluence_J_per_m2 is None
                    else float(w.fluence_J_per_m2)
                ),
                min_slant_range_km=float(w.min_slant_range_km),
                max_elevation_deg=float(w.max_elevation_deg),
                feasible=feas,
            ))

        # Continue from end-of-sol state for n_sols > 1.
        state = result.state_km_kmps[-1].copy()

    # Sun-Mars distance + sub-solar latitude at epoch (context).
    state_sun, _ = body_state(
        "SUN", epoch_et, frame="IAU_MARS", abcorr="NONE", observer="MARS",
    )
    rsx, rsy, rsz = (float(state_sun[0]), float(state_sun[1]),
                     float(state_sun[2]))
    r_mag = math.sqrt(rsx * rsx + rsy * rsy + rsz * rsz)
    sub_solar_lat_deg = math.degrees(math.asin(rsz / r_mag))
    state_mars, _ = body_state(
        "MARS", epoch_et, frame="J2000", abcorr="NONE", observer="SUN",
    )
    r_mars_sun_au = (
        float(np.linalg.norm(state_mars[:3])) / AU_KM
    )

    summary = dict(
        n_sols=int(n_sols),
        n_windows_geometric=int(total_pre_filter),
        n_windows_feasible=int(total_feasible),
        n_windows_infeasible=int(total_pre_filter - total_feasible),
        total_fluence_all_J_per_m2=float(total_fluence_all),
        total_fluence_feasible_J_per_m2=float(total_fluence_feasible),
        epoch_et=float(epoch_et),
        epoch_utc=str(epoch_utc),
        r_mars_sun_au=float(r_mars_sun_au),
        sub_solar_lat_iau_mars_deg=float(sub_solar_lat_deg),
    )
    return rows, summary


def _print_table(
    rows: List[WindowRow],
    *,
    alpha_max_rad_s2: float,
    out_stream=sys.stdout,
) -> None:
    """Print the per-window table (fixed-width, no analysis lines)."""
    header = (
        "sol  win  t_start_s    t_end_s    dur_s   "
        "peak_a (rad/s^2)  peak_I (W/m^2)  fluence (J/m^2)  "
        "minR (km)  maxEL (deg)  verdict"
    )
    print(header, file=out_stream)
    print("-" * len(header), file=out_stream)
    for r in rows:
        peak_str = (
            f"{r.peak_alpha_demand_rad_s2:.3e}"
            if r.peak_alpha_demand_rad_s2 is not None else "    n/a    "
        )
        irr_str = (
            f"{r.peak_irradiance_W_per_m2:.3e}"
            if r.peak_irradiance_W_per_m2 is not None else "    n/a    "
        )
        flu_str = (
            f"{r.fluence_J_per_m2:.3e}"
            if r.fluence_J_per_m2 is not None else "    n/a    "
        )
        if r.feasible is None:
            verdict = "n/a"
        elif r.feasible:
            verdict = "FEAS"
        else:
            verdict = "INFEAS"
        print(
            f"{r.sol_index:>3d}  {r.window_index_in_sol:>3d}  "
            f"{r.t_start_s:>9.1f}  {r.t_end_s:>9.1f}  "
            f"{r.duration_s:>6.1f}  "
            f"{peak_str:>15s}  {irr_str:>14s}  {flu_str:>14s}  "
            f"{r.min_slant_range_km:>9.2f}  {r.max_elevation_deg:>10.2f}  "
            f"{verdict}",
            file=out_stream,
        )


def _write_csv(rows: List[WindowRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sol_index", "window_index_in_sol",
        "t_start_s", "t_end_s", "duration_s",
        "peak_alpha_demand_rad_s2",
        "peak_irradiance_W_per_m2", "fluence_J_per_m2",
        "min_slant_range_km", "max_elevation_deg",
        "feasible",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "sol_index": r.sol_index,
                "window_index_in_sol": r.window_index_in_sol,
                "t_start_s": r.t_start_s,
                "t_end_s": r.t_end_s,
                "duration_s": r.duration_s,
                "peak_alpha_demand_rad_s2": (
                    "" if r.peak_alpha_demand_rad_s2 is None
                    else r.peak_alpha_demand_rad_s2
                ),
                "peak_irradiance_W_per_m2": (
                    "" if r.peak_irradiance_W_per_m2 is None
                    else r.peak_irradiance_W_per_m2
                ),
                "fluence_J_per_m2": (
                    "" if r.fluence_J_per_m2 is None
                    else r.fluence_J_per_m2
                ),
                "min_slant_range_km": r.min_slant_range_km,
                "max_elevation_deg": r.max_elevation_deg,
                "feasible": (
                    "" if r.feasible is None
                    else ("True" if r.feasible else "False")
                ),
            })


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-window analytical alpha_max feasibility at a given "
            "altitude / longitudinal phasing. Geometric ceiling — no "
            "SRP, no sail dynamics, no cruise law. See module docstring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Orbit specification (one of K or altitude_km).
    p.add_argument(
        "--K", dest="K_orbits_per_sol", type=int,
        default=DEFAULT_K_ORBITS_PER_SOL,
        help="Integer orbits per Mars-solar-sol (sun-sync repeat).",
    )
    p.add_argument(
        "--altitude-km", dest="altitude_km", type=float, default=None,
        help=(
            "Override --K with an explicit altitude (km). Disables "
            "the K-derived repeat-ground-track guarantee; sun-sync "
            "inclination is still solved at this altitude."
        ),
    )
    p.add_argument(
        "--ltan-h", dest="ltan_h", type=float, default=DEFAULT_LTAN_H,
        help="Local mean solar time of the ascending node (hours).",
    )
    # Initial phasing + epoch.
    p.add_argument(
        "--M0-deg", dest="M0_deg", type=float, default=DEFAULT_M0_DEG,
        help="Initial mean anomaly (degrees).",
    )
    p.add_argument(
        "--epoch-utc", dest="epoch_utc", type=str,
        default=DEFAULT_EPOCH_UTC,
        help="Epoch UTC ISO string.",
    )
    # Target.
    p.add_argument(
        "--target-lat", dest="target_lat_deg", type=float,
        default=DEFAULT_TARGET_LAT_DEG,
        help="Target planetographic latitude (deg).",
    )
    p.add_argument(
        "--target-lon", dest="target_lon_deg", type=float,
        default=DEFAULT_TARGET_LON_DEG,
        help="Target planetographic east longitude (deg).",
    )
    # Slew budget.
    p.add_argument(
        "--alpha-max-deg-s2", dest="alpha_max_deg_s2", type=float,
        default=DEFAULT_ALPHA_MAX_DEG_PER_S2,
        help=(
            "Slew angular acceleration budget (deg/s^2). The peak "
            "|omega_dot_bisector| is compared against this per window."
        ),
    )
    # Photometry / gates.
    p.add_argument(
        "--sigma-kg-per-m2", dest="sigma_kg_per_m2", type=float,
        default=DEFAULT_SIGMA_KG_PER_M2,
        help=(
            "Canonical sail mass loading (kg/m^2). Affects fluence "
            "numbers only (alpha_max FEAS verdict is sigma-independent)."
        ),
    )
    p.add_argument(
        "--elev-min-deg", dest="elev_min_deg", type=float,
        default=DEFAULT_ELEV_MIN_DEG,
        help="Target-horizon elevation gate (deg).",
    )
    p.add_argument(
        "--bisector-cos-alpha-min", dest="bisector_cos_alpha_min",
        type=float, default=DEFAULT_BISECTOR_COS_ALPHA_MIN,
        help="Bisector half-angle gate as cos(alpha_max).",
    )
    p.add_argument(
        "--altitude-floor-km", dest="altitude_floor_km", type=float,
        default=DEFAULT_ALTITUDE_FLOOR_KM,
        help="Propagation altitude-floor termination (km).",
    )
    p.add_argument(
        "--t-eval-cadence-s", dest="t_eval_cadence_s", type=float,
        default=DEFAULT_T_EVAL_CADENCE_S,
        help=(
            "Propagation t_eval sample cadence (s). Drives the "
            "trajectory_interpolant resolution that "
            "find_delivery_windows uses for peak_alpha_demand. "
            "60 s resolves the reference cadence; coarser sampling "
            "may misfilter narrow windows."
        ),
    )
    # Horizon.
    p.add_argument(
        "--n-sols", dest="n_sols", type=int, default=DEFAULT_N_SOLS,
        help=(
            "Number of consecutive Mars-solar-sols to evaluate "
            "(default 1; longer accumulates SRP-free secular drift)."
        ),
    )
    # Toggles.
    p.add_argument(
        "--no-sun-third-body", dest="include_sun_third_body",
        action="store_false", default=True,
        help=(
            "Disable Sun third-body perturbation (default on for "
            "1e-10 km/s^2 tide; window geometry shifts are "
            "sub-meter over a single sol)."
        ),
    )
    p.add_argument(
        "--solver", dest="solver_method", type=str,
        choices=["numerical", "analytical-j2", "kepler"],
        default="numerical",
        help=(
            "Method for resolving K -> (a, i). "
            "'numerical' (default) does 2D Nelder-Mead on (a, i) under "
            "full propagation matching the script's main physics path "
            "(J_2 + optional Sun, no SRP), minimising body-fixed "
            "closure after one Mars-solar-sol; ~10 s wall, accurate "
            "to ~1 m / 0.001 deg. "
            "'analytical-j2' uses the Brouwer first-order J_2 "
            "self-consistent formula; ~1 ms wall, but approximate at "
            "~7.6 km. "
            "'kepler' uses Kepler third law alone (the "
            "reflectors.sun_sync.repeat_ground_track_altitude); ~5 km "
            "off the J_2 self-consistent at K=12."
        ),
    )
    # Output.
    p.add_argument(
        "--csv-out", dest="csv_out", type=Path, default=None,
        help=(
            "Optional path to write the per-window table as CSV. "
            "If omitted, only stdout is printed."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    log_level = logging.WARNING - 10 * min(int(args.verbose), 2)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_kernels()

    epoch_et = utc_to_et(str(args.epoch_utc))
    a_km, R_mars, K_used, T_orb_s, i_ss_rad = _resolve_altitude(
        K_orbits_per_sol=args.K_orbits_per_sol,
        altitude_km_override=args.altitude_km,
        solver_method=str(args.solver_method),
        epoch_et=epoch_et,
        ltan_h=float(args.ltan_h),
        M0_deg=float(args.M0_deg),
        altitude_floor_km=float(args.altitude_floor_km),
        t_eval_cadence_s=float(args.t_eval_cadence_s),
        include_sun_third_body=bool(args.include_sun_third_body),
    )
    altitude_km = a_km - R_mars
    alpha_max_rad_s2 = math.radians(float(args.alpha_max_deg_s2))
    if K_used is None:
        a_i_path = "explicit --altitude-km (i_ss from analytical J_2)"
    elif args.solver_method == "kepler":
        a_i_path = "Kepler-only (sun_sync.repeat_ground_track_altitude)"
    elif args.solver_method == "analytical-j2":
        a_i_path = "self-consistent J_2-nodal-period analytical solve"
    else:
        a_i_path = (
            "numerical 2D NM on (a, i) under full propagation [default]"
        )

    # Banner (stdout — same stream the table goes to so a single log capture
    # has the full context).
    print("=" * 78)
    print("illumination_window_feasibility — analytical alpha_max per window")
    print("=" * 78)
    if K_used is not None:
        print(f"  K orbits/solar-sol  = {K_used}")
    else:
        print(f"  K orbits/solar-sol  = (n/a — explicit --altitude-km)")
    print(f"  (a, i) source       = {a_i_path}")
    print(
        f"  a_km                = {a_km:.4f}  "
        f"(altitude = {altitude_km:.2f} km)"
    )
    print(f"  i_sun_sync_deg      = {math.degrees(i_ss_rad):.5f}")
    if T_orb_s is not None:
        kind = {
            "kepler": "T_kepler",
            "analytical-j2": "T_nodal_J2",
            "numerical": "T_solar (closure target)",
        }.get(str(args.solver_method), "T_orb")
        print(f"  {kind:<24s} = {T_orb_s:.4f} s")
    print(f"  ltan_h              = {args.ltan_h}")
    print(f"  M0_deg              = {args.M0_deg}")
    print(f"  epoch_utc           = {args.epoch_utc}")
    print(
        f"  target              = ({args.target_lat_deg} N, "
        f"{args.target_lon_deg} E)"
    )
    print(
        f"  alpha_max           = {args.alpha_max_deg_s2} deg/s^2  "
        f"= {alpha_max_rad_s2:.6e} rad/s^2"
    )
    print(f"  sigma (kg/m^2)      = {args.sigma_kg_per_m2}")
    print(
        "  optical path        = vacuum "
        f"(transmission={DEFAULT_ATMOSPHERIC_TRANSMISSION})"
    )
    print(f"  elev_min_deg        = {args.elev_min_deg}")
    print(f"  bisector cos_a_min  = {args.bisector_cos_alpha_min}")
    print(f"  altitude_floor_km   = {args.altitude_floor_km}")
    print(f"  t_eval_cadence_s    = {args.t_eval_cadence_s}")
    print(f"  n_sols              = {args.n_sols}")
    print(
        f"  Sun third body      = "
        f"{'on' if args.include_sun_third_body else 'off'}"
    )
    print("-" * 78)

    t0 = time.perf_counter()
    rows, summary = evaluate(
        a_km=a_km,
        ltan_h=float(args.ltan_h),
        M0_deg=float(args.M0_deg),
        epoch_utc=str(args.epoch_utc),
        target_lat_deg=float(args.target_lat_deg),
        target_lon_deg=float(args.target_lon_deg),
        alpha_max_rad_s2=alpha_max_rad_s2,
        sigma_kg_per_m2=float(args.sigma_kg_per_m2),
        atmospheric_transmission=DEFAULT_ATMOSPHERIC_TRANSMISSION,
        elev_min_deg=float(args.elev_min_deg),
        bisector_cos_alpha_min=float(args.bisector_cos_alpha_min),
        altitude_floor_km=float(args.altitude_floor_km),
        t_eval_cadence_s=float(args.t_eval_cadence_s),
        n_sols=int(args.n_sols),
        include_sun_third_body=bool(args.include_sun_third_body),
    )
    wall_s = time.perf_counter() - t0

    print(
        f"  r_mars_sun_au       = {summary['r_mars_sun_au']:.6f}"
    )
    print(
        f"  sub_solar_lat_deg   = "
        f"{summary['sub_solar_lat_iau_mars_deg']:.3f}"
    )
    print(f"  evaluation wall_s   = {wall_s:.2f}")
    print("-" * 78)
    _print_table(rows, alpha_max_rad_s2=alpha_max_rad_s2)
    print("-" * 78)
    print(
        f"summary: n_geometric_windows = "
        f"{summary['n_windows_geometric']}, "
        f"n_feasible = {summary['n_windows_feasible']}, "
        f"n_infeasible = {summary['n_windows_infeasible']}"
    )
    print(
        f"summary: total_fluence_all = "
        f"{summary['total_fluence_all_J_per_m2']:.4f} J/m^2, "
        f"total_fluence_feasible = "
        f"{summary['total_fluence_feasible_J_per_m2']:.4f} J/m^2"
    )

    if args.csv_out is not None:
        _write_csv(rows, args.csv_out)
        print(f"per-window CSV written: {args.csv_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
