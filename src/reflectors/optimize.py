"""Cruise-attitude optimizer at a fixed Mars-orbit grid point.

The module supports constant and harmonic-in-u cruise families through the
parameterization-agnostic ``optimize_cruise_general`` and
``evaluate_cruise_general`` core. A ``CruiseFactory`` callable isolates family
construction from the optimizer. Dynamic-endpoint slews, acceleration-driven
slew sizing, parametric cruise attitudes, and fixed-point window-boundary
refinement are included. Harmonic families expose the cos(u) and sin(u) Fourier
modes that drive ``<de/dt>`` per McInnes 1999 ch. 4 Eqs. 4.14a-f and 4.15a-c.

Cost / constraint structure:

    minimize   |Δa_per_sol_km(β, φ_u)|
    subject to fluence(β, φ_u) ≥ 0.9 · fluence_baseline

Implemented as a quadratic exterior penalty:

    cost(x) = |Δa_per_sol|
            + λ · max(0, fluence_floor − fluence)²

Bisector-feasibility is already enforced inside
``visibility.find_delivery_windows`` via ``bisector_cos_alpha_min=0.1``
+ the optional α_max post-filter, so windows that fail those gates
simply drop out of the fluence sum and the constraint penalty handles
the consequence; no separate bisector term is needed.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import spiceypy as spice
from scipy.optimize import (
    differential_evolution as scipy_differential_evolution,
    minimize as scipy_minimize,
)

from reflectors.attitude import AttitudeCallable, piecewise
from reflectors.attitude_schedule import (
    CruiseSlewMetadata,
    DeliveryScheduleSeed,
    RefinedSchedule,
    cruise_to_cruise_slew,
    refine_delivery_schedule,
)
from reflectors.cruise import (
    sun_offset_from_state,
    sun_offset_harmonic_from_state,
    sun_offset_harmonic_full_from_state,
    sun_offset_harmonic_full_mode2_from_state,
    sun_offset_harmonic_full_mode3_from_state,
)
from reflectors.elements import (
    elements_in_mme2000,
    secular_argp_rate_J2_rad_per_s,
)
from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S
from reflectors.poincare import find_ascending_node_crossings
from reflectors.srp import SolarSail
from reflectors.sun_sync import initial_state_j2000


logger = logging.getLogger(__name__)


# Sidereal sol used to normalize Δa across horizons. Same value the rest
# of the codebase uses (1 sidereal day ≈ 88775 s on Mars). Pinned here
# so the cost function's "per sol" unit is well-defined even when a
# caller picks a non-sol duration.
SECONDS_PER_SIDEREAL_SOL = 88775.0


# Brouwer first-order secular sun-sync RAAN drift rate.
# Ω̇_sunsync = 2π / MARS_SIDEREAL_YEAR_S (same constant sun_sync.py:112
# uses for the inclination solver). The Δ(RAAN − sunsync) term in the
# multi-element cost measures the miss from this design target.
SUN_SYNC_RAAN_RATE_RAD_PER_S = 2.0 * math.pi / MARS_SIDEREAL_YEAR_S


@dataclass(frozen=True)
class OrbitConfig:
    """Everything that does not change per evaluation.

    Captures the fixed orbit, sail bus, target, mission constraints, and
    propagator / window-finder kwargs once at construction. The
    composed initial state vector is computed eagerly via
    ``sun_sync.initial_state_j2000`` and stored on the instance so the
    cost function can avoid recomputing it on every call.
    """

    a_km: float
    ltan_h: float
    M0_rad: float
    epoch_et: float
    duration_s: float
    sail: SolarSail
    target_lat_deg: float
    target_lon_deg: float
    alpha_max_rad_s2: float
    slew_duration_s: float = 300.0
    omega_max_rad_s: Optional[float] = None
    propagate_kwargs: dict = field(default_factory=dict)
    find_windows_kwargs: dict = field(default_factory=dict)
    max_iterations: int = 5
    convergence_tol_et_s: float = 1.0
    damping: float = 0.7
    window_continuation_seed: Optional[DeliveryScheduleSeed] = None
    window_continuation_search_margin_s: float = 900.0
    window_continuation_max_boundary_shift_s: Optional[float] = None
    window_continuation_failure: str = "full_scan"
    initial_state_override_km_kmps: Optional[np.ndarray] = None
    # Additional surface targets beyond the primary
    # (target_lat_deg, target_lon_deg). Order defines target_idx 1, 2, ...;
    # the default empty tuple selects a single-target analysis.
    extra_targets: Tuple[Tuple[float, float], ...] = ()
    initial_state_km_kmps: np.ndarray = field(init=False)
    mu_km3_s2: float = field(init=False)

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        if self.slew_duration_s <= 0.0:
            raise ValueError(
                f"slew_duration_s must be > 0, got {self.slew_duration_s}"
            )
        if self.alpha_max_rad_s2 <= 0.0:
            raise ValueError(
                f"alpha_max_rad_s2 must be > 0, got {self.alpha_max_rad_s2}"
            )
        if self.omega_max_rad_s is not None and (
            not math.isfinite(self.omega_max_rad_s)
            or self.omega_max_rad_s <= 0.0
        ):
            raise ValueError(
                f"omega_max_rad_s must be positive and finite, got "
                f"{self.omega_max_rad_s}"
            )
        if self.window_continuation_failure not in {"full_scan", "raise"}:
            raise ValueError(
                "window_continuation_failure must be 'full_scan' or 'raise'"
            )
        margin_s = float(self.window_continuation_search_margin_s)
        if not math.isfinite(margin_s) or margin_s <= 0.0:
            raise ValueError(
                "window_continuation_search_margin_s must be positive and finite"
            )
        if self.window_continuation_max_boundary_shift_s is not None:
            shift_s = float(self.window_continuation_max_boundary_shift_s)
            if (
                not math.isfinite(shift_s)
                or shift_s < 0.0
                or shift_s >= margin_s
            ):
                raise ValueError(
                    "window_continuation_max_boundary_shift_s must be finite "
                    "and satisfy 0 <= shift < search margin"
                )
        # Canonicalize extra_targets to a tuple of (float, float) pairs
        # so the frozen config hashes/pickles uniformly regardless of
        # the caller's sequence types.
        canon_extra = []
        for entry in self.extra_targets:
            lat, lon = entry
            lat_f, lon_f = float(lat), float(lon)
            if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
                raise ValueError(
                    f"extra_targets entries must be finite "
                    f"(lat_deg, lon_deg) pairs, got {entry!r}"
                )
            canon_extra.append((lat_f, lon_f))
        object.__setattr__(self, "extra_targets", tuple(canon_extra))
        # Frozen dataclass: object.__setattr__ to populate init=False fields.
        # When the caller passes initial_state_override_km_kmps, use it
        # directly (e.g. periodic re-optimization where each sol's
        # initial state is the previous sol's propagated end-state, not
        # the sun-sync analytic state). Otherwise compute the sun-sync
        # state from (a, ltan, M0, epoch).
        if self.initial_state_override_km_kmps is not None:
            override = np.asarray(self.initial_state_override_km_kmps, dtype=float)
            if override.shape != (6,):
                raise ValueError(
                    f"initial_state_override_km_kmps must have shape (6,), "
                    f"got {override.shape}"
                )
            state0 = override
        else:
            state0 = initial_state_j2000(
                a_km=float(self.a_km),
                ltan_h=float(self.ltan_h),
                M0_rad=float(self.M0_rad),
                epoch_et=float(self.epoch_et),
            )
        object.__setattr__(self, "initial_state_km_kmps", state0)
        # Same μ source the gravity model uses (MRO120F system GM); kept
        # here so element extraction is self-consistent with the
        # propagator's central-body μ choice.
        from reflectors.sun_sync import _default_gravity_anchors

        mu, _R, _J2 = _default_gravity_anchors()
        object.__setattr__(self, "mu_km3_s2", float(mu))


@dataclass(frozen=True)
class EvaluationResult:
    """Per-evaluation snapshot. Produced per ``evaluate_cruise(_general)`` call.

    ``beta_rad`` and ``phi_u_rad`` carry the DC cone-angle and clock-
    angle respectively in BOTH parameterizations (constant cruise:
    beta_rad = β, phi_u_rad = φ_u; harmonic cruise: beta_rad = α_0,
    phi_u_rad = φ_u). They preserve the constant-family reporting
    contract. The full parameterization vector lives in
    ``decision_vector_rad``.

    Multi-element cost fields:

      delta_i_deg               signed Δi over the propagation, MME2000
                                frame, first → last sample, deg
      delta_raan_deg            signed ΔΩ, MME2000 frame, first → last
                                sample, wrapped to (−180°, +180°], deg
      raan_sun_sync_target_deg  Brouwer first-order secular sun-sync
                                target ΔΩ over the propagation duration:
                                = (2π / MARS_SIDEREAL_YEAR_S) ·
                                  duration_actual_s, deg
      e_end_osculating          Osculating e at the last sparse sample.
                                Distinct from ``e_max`` (peak over the
                                trajectory).
      cost_breakdown            For multi-element costs, a tuple of
                                ("element_name", contribution_value)
                                pairs recording each term's value
                                BEFORE summation. Empty tuple under
                                single-element costs.

    Closure-targeted cost fields:

      delta_e_same_u            ``e[ascending_node[-1]] − e[ascending_node[0]]``,
                                i.e. the change in osculating eccentricity
                                across one propagation, sampled at matched
                                orbital phase via the ascending-node Poincaré
                                section. Same-u sampling cancels the J_2
                                short-period oscillation (Brouwer 1959 ``δe_sp``)
                                so only the secular Δ⟨e⟩ part survives.
      delta_argp_deg            Signed Δω, MME2000 frame, first → last
                                sample, wrapped to (−180°, +180°], deg.
      argp_J2_target_deg        Brouwer first-order secular argp drift
                                target over the propagation duration:
                                ω̇_J2(elts_0) · duration_actual_s, deg.
                                Per Brouwer 1959 Eq. (40) at the
                                osculating elements at sol-start
                                (O(J_2²) error vs mean elements is ~4e-6,
                                negligible vs the rate magnitude).

    Mars-fixed Poincaré-map closure fields:

      delta_r_iau_mars_km       3-vector ``r_iau_mars(sol_end) −
                                r_iau_mars(sol_start)`` after transforming
                                the J2000 endpoints through
                                ``spice.sxform("J2000", "IAU_MARS", et)``.
                                Captures structural Poincaré-map fixed-
                                point closure not captured by the
                                osculating-element residuals.
      delta_v_iau_mars_kmps     3-vector velocity counterpart, same
                                transform. ``sxform`` (not ``pxform``)
                                is required so the Coriolis term in
                                the velocity row of the 6×6 state
                                transform is included; at orbital
                                speeds (~3 km/s) this term is ~m/s
                                and dominates over ``pxform``-only
                                rotation residuals.
    """

    beta_rad: float
    phi_u_rad: float
    total_fluence_J_per_m2: float
    delta_a_km: float
    delta_a_per_sol_km: float
    e_max: float
    inc_range_deg: float
    n_windows_kept: int
    n_windows_dropped: int
    n_unstable_windows: int
    converged: bool
    n_iterations: int
    wall_s: float
    decision_vector_rad: Tuple[float, ...] = ()
    # Multi-element cost over orbital-element drifts.
    delta_i_deg: float = 0.0
    delta_raan_deg: float = 0.0
    raan_sun_sync_target_deg: float = 0.0
    e_end_osculating: float = 0.0
    cost_breakdown: Tuple[Tuple[str, float], ...] = ()
    # Closure-targeted cost.
    delta_e_same_u: float = 0.0
    delta_argp_deg: float = 0.0
    argp_J2_target_deg: float = 0.0
    # Mars-fixed (r, v) Poincaré-map closure.
    delta_r_iau_mars_km: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    delta_v_iau_mars_kmps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # One entry per target,
    # primary target first (index order = config.extra_targets order).
    # Fluence is credited over KEPT (schedule-served) windows only;
    # empty tuples represent single-target results.
    fluence_by_target_J_per_m2: Tuple[float, ...] = ()
    n_windows_by_target: Tuple[int, ...] = ()
    window_initialization_mode: str = "global_cruise_scan"
    window_search_modes: Tuple[str, ...] = ()
    n_window_continuation_fallbacks: int = 0
    n_propagations: int = 0


@dataclass(frozen=True)
class OptimizationRun:
    """Output of ``optimize_cruise``.

    ``history`` is the full per-evaluation trace, in call order, so callers
    can write a CSV or inspect constraint-active behaviour.
    ``best_eval`` is the lowest-cost evaluation across history. It may
    differ from the optimizer's reported optimum when termination occurs
    on a nonminimal simplex vertex.
    """

    scipy_result: object  # scipy.optimize.OptimizeResult
    history: Tuple[EvaluationResult, ...]
    best_eval: EvaluationResult
    baseline_eval: EvaluationResult
    config: OrbitConfig
    fluence_floor_J_per_m2: float
    penalty_lambda: float
    cost_metric: str
    wall_total_s: float
    # Multi-element cost configuration. Empty mappings under single-element costs.
    multi_element_weights: Mapping[str, float] = field(default_factory=dict)
    multi_element_scales: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cruise-family factory abstraction
#
# A CruiseFactory builds a parameterization-specific AttitudeCallable from a
# decision vector x and an OrbitConfig. The factory is the only seam between
# the optimizer core and a particular cruise family, so adding new families is
# purely additive.
# ---------------------------------------------------------------------------


CruiseFactory = Callable[[np.ndarray, "OrbitConfig"], AttitudeCallable]


def _constant_cruise_factory(
    x: np.ndarray, config: "OrbitConfig"
) -> AttitudeCallable:
    """Constant cruise: x = [β_rad, φ_u_rad]."""
    return sun_offset_from_state(
        beta_rad=float(x[0]),
        phi_u_rad=float(x[1]),
        initial_state_km_kmps=config.initial_state_km_kmps,
    )


def _harmonic_cruise_factory(
    x: np.ndarray, config: "OrbitConfig"
) -> AttitudeCallable:
    """Harmonic-in-u cruise: x = [α_0_rad, α_c_rad, α_s_rad, φ_u_rad]."""
    return sun_offset_harmonic_from_state(
        alpha_0_rad=float(x[0]),
        alpha_c_rad=float(x[1]),
        alpha_s_rad=float(x[2]),
        phi_u_rad=float(x[3]),
        initial_state_km_kmps=config.initial_state_km_kmps,
    )


def _harmonic_full_cruise_factory(
    x: np.ndarray, config: "OrbitConfig"
) -> AttitudeCallable:
    """Harmonic-(α, δ) cruise: x = [α_0, α_c, α_s, δ_0, δ_c, δ_s] rad.

    Decision-vector ordering groups by angle: cone-angle (DC, cos, sin)
    then clock-angle (DC, cos, sin). Setting x[4] = x[5] = 0 reduces
    bit-for-bit through the pipeline to ``_harmonic_cruise_factory(x[:4])``;
    setting x[1] = x[2] = x[4] = x[5] = 0 further reduces to
    ``_constant_cruise_factory([x[0], x[3]])``.
    """
    return sun_offset_harmonic_full_from_state(
        alpha_0_rad=float(x[0]),
        alpha_c_rad=float(x[1]),
        alpha_s_rad=float(x[2]),
        delta_0_rad=float(x[3]),
        delta_c_rad=float(x[4]),
        delta_s_rad=float(x[5]),
        initial_state_km_kmps=config.initial_state_km_kmps,
    )


def _harmonic_full_mode2_cruise_factory(
    x: np.ndarray, config: "OrbitConfig"
) -> AttitudeCallable:
    """Mode-2 harmonic-(α, δ) cruise: 10-D x = [
        α_0, α_c1, α_s1, α_c2, α_s2,
        δ_0, δ_c1, δ_s1, δ_c2, δ_s2
    ] rad.

    Decision-vector ordering groups by angle (cone-angle DC, mode-1
    cosine/sine, mode-2 cosine/sine, then clock-angle in the same
    pattern). Setting x[3:5] = x[8:10] = 0 reduces bit-for-bit through
    the cruise constructor to ``_harmonic_full_cruise_factory(x[[0,1,2,5,6,7]])``;
    setting all higher-mode amplitudes to zero further reduces to
    constant cruise.
    """
    return sun_offset_harmonic_full_mode2_from_state(
        alpha_0_rad=float(x[0]),
        alpha_c1_rad=float(x[1]),
        alpha_s1_rad=float(x[2]),
        alpha_c2_rad=float(x[3]),
        alpha_s2_rad=float(x[4]),
        delta_0_rad=float(x[5]),
        delta_c1_rad=float(x[6]),
        delta_s1_rad=float(x[7]),
        delta_c2_rad=float(x[8]),
        delta_s2_rad=float(x[9]),
        initial_state_km_kmps=config.initial_state_km_kmps,
    )


def _harmonic_full_mode3_cruise_factory(
    x: np.ndarray, config: "OrbitConfig"
) -> AttitudeCallable:
    """Mode-3 harmonic-(α, δ) cruise: 14-D x = [
        α_0, α_c1, α_s1, α_c2, α_s2, α_c3, α_s3,
        δ_0, δ_c1, δ_s1, δ_c2, δ_s2, δ_c3, δ_s3
    ] rad.

    Decision-vector ordering groups by angle (cone-angle DC then
    mode-1/2/3 cosine/sine, then clock-angle in the same pattern).
    Setting x[5:7] = x[12:14] = 0 reduces bit-for-bit through the
    cruise constructor to ``_harmonic_full_mode2_cruise_factory(
    x[[0,1,2,3,4,7,8,9,10,11]])``; setting all higher-mode amplitudes
    to zero further reduces to mode-1, then to constant cruise.
    """
    return sun_offset_harmonic_full_mode3_from_state(
        alpha_0_rad=float(x[0]),
        alpha_c1_rad=float(x[1]),
        alpha_s1_rad=float(x[2]),
        alpha_c2_rad=float(x[3]),
        alpha_s2_rad=float(x[4]),
        alpha_c3_rad=float(x[5]),
        alpha_s3_rad=float(x[6]),
        delta_0_rad=float(x[7]),
        delta_c1_rad=float(x[8]),
        delta_s1_rad=float(x[9]),
        delta_c2_rad=float(x[10]),
        delta_s2_rad=float(x[11]),
        delta_c3_rad=float(x[12]),
        delta_s3_rad=float(x[13]),
        initial_state_km_kmps=config.initial_state_km_kmps,
    )


def make_handoff_cruise_factory(
    base_factory: CruiseFactory,
    cruise_old: AttitudeCallable,
    *,
    central_body_gm_km3_s2: float,
    slew_floor_s: float = 60.0,
) -> CruiseFactory:
    """Wrap a base ``CruiseFactory`` with an initial sol-boundary slew.

    Returned factory composes a piecewise schedule per call:

      [config.epoch_et, config.epoch_et + T_slew]: smooth_slew_hermite
        from cruise_old(state_b, t_b) to base_factory(x, config) evaluated
        at the Kepler-predicted state at t_b + T_slew. Endpoints' n_hat
        and omega both match by construction.
      [config.epoch_et + T_slew, config.epoch_et + config.duration_s]:
        the new cruise law from base_factory(x, config).

    The slew is sized via ``cruise_to_cruise_slew`` (which uses
    the configured alpha and optional omega limits). State
    continuity is automatic: the propagator hands the actual state
    along, the slew callable accepts it, and ``cruise_new`` accepts the
    handed-off state at the slew end.

    Used inside ``optimize_cruise_general`` for per-sol re-optimization:
    every objective-function evaluation rebuilds the slew from the
    candidate cruise_new (option (ii) — optimizer sees what actually
    runs).

    Parameters
    ----------
    base_factory
        The underlying cruise factory (e.g. ``_harmonic_full_cruise_factory``)
        producing ``cruise_new(x, config) -> AttitudeCallable``.
    cruise_old
        Cruise callable from the preceding sol, providing the slew's starting
        attitude and rate.
    central_body_gm_km3_s2
        Forwarded to ``cruise_to_cruise_slew`` for the two-body Kepler
        predictor. Use the same μ the propagator uses (typically
        ``mars_gm_km3_per_s2()``).
    slew_floor_s
        Forwarded to ``cruise_to_cruise_slew``.

    Returns
    -------
    CruiseFactory
        ``(x, config) -> AttitudeCallable`` giving the slew + cruise
        piecewise composite.
    """

    def factory(x: np.ndarray, config: "OrbitConfig") -> AttitudeCallable:
        cruise_new = base_factory(x, config)
        slew, slew_meta = cruise_to_cruise_slew(
            cruise_old, cruise_new,
            state_b_km_kmps=config.initial_state_km_kmps,
            epoch_et_b=float(config.epoch_et),
            central_body_gm_km3_s2=float(central_body_gm_km3_s2),
            alpha_max_rad_s2=float(config.alpha_max_rad_s2),
            omega_max_rad_s=(
                None
                if config.omega_max_rad_s is None
                else float(config.omega_max_rad_s)
            ),
            slew_floor_s=float(slew_floor_s),
        )
        t_end_slew = slew_meta.t_end_et
        t_end_sol = float(config.epoch_et) + float(config.duration_s)
        if t_end_slew >= t_end_sol:
            raise ValueError(
                f"make_handoff_cruise_factory: slew end {t_end_slew} >= "
                f"sol end {t_end_sol}; slew larger than the cost-evaluation "
                f"window. Increase config.duration_s or shrink slew_floor_s."
            )
        return piecewise([
            (float(config.epoch_et), t_end_slew, slew),
            (t_end_slew, t_end_sol, cruise_new),
        ])

    return factory


def _kept_window_fluence(
    refined: "RefinedSchedule",  # noqa: F821 (duck-typed)
    n_targets: int,
) -> Tuple[float, Tuple[float, ...], Tuple[int, ...]]:
    """Total and per-target fluence over KEPT (schedule-served) windows.

    ``refined.final_windows`` is exactly the
    time-sorted list the final ``build_delivery_schedule`` call
    consumed, and ``refined.metadata.dropped_window_reasons`` indexes
    into that ordering, so positions not in the dropped set are the
    windows the schedule actually bisector-tracks. Windows the
    schedule DROPS (slew-buffer conflicts, boundary clipping) deliver
    nothing — the sail cruises through them — so crediting their
    fluence would reward undelivered light. Single-target calculations
    drops ~0 windows, in which case this is numerically identical to
    the previous sum-over-all-windows.

    Returns ``(total_fluence, fluence_by_target, n_windows_by_target)``
    where the per-target tuples have length ``n_targets`` and are
    indexed by ``DeliveryWindow.target_idx``.
    """
    dropped_positions = {
        int(i) for i, _reason in refined.metadata.dropped_window_reasons
    }
    dropped_fluence = sum(
        float(w.fluence_J_per_m2)
        for i, w in enumerate(refined.final_windows)
        if i in dropped_positions and w.fluence_J_per_m2 is not None
    )
    if dropped_fluence > 0.0:
        logger.info(
            "_kept_window_fluence: %d schedule-dropped window(s) "
            "carried %.4f J/m^2 of (uncredited) fluence.",
            len(dropped_positions), dropped_fluence,
        )

    total_fluence = 0.0
    fluence_by_target = [0.0] * int(n_targets)
    n_windows_by_target = [0] * int(n_targets)
    for i, w in enumerate(refined.final_windows):
        if i in dropped_positions:
            continue
        t_idx = int(w.target_idx)
        if not 0 <= t_idx < n_targets:
            raise RuntimeError(
                f"_kept_window_fluence: window target_idx {t_idx} out "
                f"of range for {n_targets} configured target(s)."
            )
        n_windows_by_target[t_idx] += 1
        if w.fluence_J_per_m2 is not None:
            total_fluence += float(w.fluence_J_per_m2)
            fluence_by_target[t_idx] += float(w.fluence_J_per_m2)
    return total_fluence, tuple(fluence_by_target), tuple(n_windows_by_target)


def evaluate_cruise_general(
    x: np.ndarray,
    config: OrbitConfig,
    cruise_factory: CruiseFactory,
) -> EvaluationResult:
    """Parameterization-agnostic evaluation core.

    Builds the cruise callable via ``cruise_factory(x, config)``, calls
    ``attitude_schedule.refine_delivery_schedule`` with the config's
    propagator + window-finder kwargs, and reduces the result to the
    scalars the cost function needs (Δa, fluence, e_max, window counts).

    The ``EvaluationResult.beta_rad`` field is set to ``x[0]`` (the DC
    cone-angle component, which equals β for constant cruise and α_0
    for harmonic cruise); ``phi_u_rad`` is set to ``x[-1]`` (the last
    decision-vector component, conventionally the constant clock
    angle in this codebase's parameterizations). The full vector is
    recorded in ``decision_vector_rad``.

    Δa is computed in MME2000 (Mars-mean-equator) via
    ``elements.elements_in_mme2000`` at the first and last propagation
    samples (osculating; sub-100 m osc-vs-mean offset cancels at
    endpoints over 1-sol horizons).
    """
    t0 = time.perf_counter()
    x_arr = np.asarray(x, dtype=float).ravel()

    cruise_profile = cruise_factory(x_arr, config)

    refined: RefinedSchedule = refine_delivery_schedule(
        config.initial_state_km_kmps,
        t_span_s=(0.0, float(config.duration_s)),
        epoch_et=float(config.epoch_et),
        cruise_profile=cruise_profile,
        target_lat_deg=float(config.target_lat_deg),
        target_lon_deg=float(config.target_lon_deg),
        sail=config.sail,
        extra_targets=tuple(config.extra_targets),
        slew_duration_s=float(config.slew_duration_s),
        alpha_max_rad_s2=float(config.alpha_max_rad_s2),
        omega_max_rad_s=(
            None
            if config.omega_max_rad_s is None
            else float(config.omega_max_rad_s)
        ),
        max_iterations=int(config.max_iterations),
        convergence_tol_et_s=float(config.convergence_tol_et_s),
        damping=float(config.damping),
        propagate_kwargs=dict(config.propagate_kwargs),
        find_windows_kwargs=dict(config.find_windows_kwargs),
        continuation_seed=config.window_continuation_seed,
        continuation_search_margin_s=float(
            config.window_continuation_search_margin_s
        ),
        continuation_max_boundary_shift_s=(
            None
            if config.window_continuation_max_boundary_shift_s is None
            else float(config.window_continuation_max_boundary_shift_s)
        ),
        continuation_failure=str(config.window_continuation_failure),
    )

    # Fluence: credit kept, schedule-served windows only; see
    # _kept_window_fluence.
    n_targets = 1 + len(config.extra_targets)
    total_fluence, fluence_by_target, n_windows_by_target = (
        _kept_window_fluence(refined, n_targets)
    )

    # Δa via osculating elements at the first and last samples.
    result = refined.final_result
    elts_0 = elements_in_mme2000(
        result.state_km_kmps[0],
        config.mu_km3_s2,
        epoch_et=float(config.epoch_et),
    )
    elts_f = elements_in_mme2000(
        result.state_km_kmps[-1],
        config.mu_km3_s2,
        epoch_et=float(config.epoch_et) + float(result.t_s[-1]),
    )
    delta_a_km = float(elts_f.a_km - elts_0.a_km)
    duration_actual_s = float(result.t_s[-1] - result.t_s[0])
    if duration_actual_s <= 0.0:
        raise RuntimeError(
            f"evaluate_cruise_general: propagation produced non-positive "
            f"duration ({duration_actual_s} s); cannot compute Δa_per_sol."
        )
    delta_a_per_sol_km = delta_a_km * SECONDS_PER_SIDEREAL_SOL / duration_actual_s

    # Signed first-to-last drifts of inclination and RAAN, plus
    # the Brouwer first-order secular sun-sync RAAN target. ΔΩ wraps to
    # (-180°, +180°] so multi-sol horizons that approach a full revolution
    # don't produce a spurious 359°-vs-1° miss.
    delta_i_deg = math.degrees(
        float(elts_f.inclination_rad - elts_0.inclination_rad)
    )
    delta_raan_raw_deg = math.degrees(
        float(elts_f.raan_rad - elts_0.raan_rad)
    )
    delta_raan_deg = ((delta_raan_raw_deg + 180.0) % 360.0) - 180.0
    raan_sun_sync_target_deg = math.degrees(
        SUN_SYNC_RAAN_RATE_RAD_PER_S * duration_actual_s
    )

    # Closure-cost fields. Δargp uses the same wrap convention as
    # ΔΩ (so multi-sol horizons that approach a full revolution don't
    # produce spurious 359° vs 1° miss). The Brouwer secular target uses
    # the osculating elements at sol-start as input; mean-vs-osculating
    # discrepancy is O(J_2²) ~4e-6, negligible vs the rate itself.
    delta_argp_raw_deg = math.degrees(
        float(elts_f.argp_rad - elts_0.argp_rad)
    )
    delta_argp_deg = ((delta_argp_raw_deg + 180.0) % 360.0) - 180.0
    argp_rate_J2_rad_per_s = secular_argp_rate_J2_rad_per_s(
        a_km=float(elts_0.a_km),
        e=float(elts_0.e),
        inc_rad=float(elts_0.inclination_rad),
    )
    argp_J2_target_deg = math.degrees(
        argp_rate_J2_rad_per_s * duration_actual_s
    )

    # Δe at matched orbital phase: detect ascending-node crossings on
    # the dense state history and take ``e[last] − e[first]``. Same-u
    # sampling cancels the J_2 short-period oscillation (Brouwer 1959
    # δe_sp), so only the secular Δ⟨e⟩ component survives — the right
    # target for the closure-cost ``delta_e_same_u`` term.
    asc_crossings = find_ascending_node_crossings(
        result.t_s,
        result.state_km_kmps,
        config.mu_km3_s2,
        epoch_et_offset=float(config.epoch_et),
    )
    if len(asc_crossings) >= 2:
        delta_e_same_u = float(
            asc_crossings[-1].e - asc_crossings[0].e
        )
    else:
        delta_e_same_u = 0.0
        logger.warning(
            "evaluate_cruise_general: only %d ascending-node crossings "
            "detected; delta_e_same_u set to 0.0 as fallback.",
            len(asc_crossings),
        )

    # Eccentricity / inclination envelope from a sparse element scan.
    n_sparse = min(16, len(result.t_s))
    idx = np.linspace(0, len(result.t_s) - 1, n_sparse, dtype=int)
    e_vals = []
    inc_vals_deg = []
    for k in idx:
        elts_k = elements_in_mme2000(
            result.state_km_kmps[k],
            config.mu_km3_s2,
            epoch_et=float(config.epoch_et) + float(result.t_s[k]),
        )
        e_vals.append(float(elts_k.e))
        inc_vals_deg.append(math.degrees(float(elts_k.inclination_rad)))
    e_max = float(max(e_vals))
    e_end_osculating = float(e_vals[-1])
    inc_range_deg = float(max(inc_vals_deg) - min(inc_vals_deg))

    n_kept = int(refined.metadata.n_windows_kept)
    n_dropped = int(refined.metadata.n_windows_dropped)
    n_unstable = int(len(refined.unstable_windows))

    # Mars-fixed (IAU_MARS) state-vector closure deltas.
    # spice.sxform is the 6×6 state transform; the velocity rows include
    # the rate-of-rotation cross-term that pxform omits. Required for
    # velocity comparisons at orbital speed (~3 km/s × ω_mars × r ≈
    # 0.7 m/s scale). Precedent: src/reflectors/surface.py:151.
    et_start = float(config.epoch_et) + float(result.t_s[0])
    et_end = float(config.epoch_et) + float(result.t_s[-1])
    T6_start = np.asarray(
        spice.sxform("J2000", "IAU_MARS", et_start), dtype=float,
    )
    T6_end = np.asarray(
        spice.sxform("J2000", "IAU_MARS", et_end), dtype=float,
    )
    state_start_iau = T6_start @ np.asarray(
        result.state_km_kmps[0], dtype=float,
    )
    state_end_iau = T6_end @ np.asarray(
        result.state_km_kmps[-1], dtype=float,
    )
    delta_r_iau_mars_km = tuple(
        float(v) for v in (state_end_iau[:3] - state_start_iau[:3])
    )
    delta_v_iau_mars_kmps = tuple(
        float(v) for v in (state_end_iau[3:] - state_start_iau[3:])
    )

    return EvaluationResult(
        beta_rad=float(x_arr[0]),
        phi_u_rad=float(x_arr[-1]),
        total_fluence_J_per_m2=total_fluence,
        delta_a_km=delta_a_km,
        delta_a_per_sol_km=delta_a_per_sol_km,
        e_max=e_max,
        inc_range_deg=inc_range_deg,
        n_windows_kept=n_kept,
        n_windows_dropped=n_dropped,
        n_unstable_windows=n_unstable,
        converged=bool(refined.converged),
        n_iterations=int(refined.n_iterations),
        wall_s=time.perf_counter() - t0,
        decision_vector_rad=tuple(float(v) for v in x_arr),
        delta_i_deg=delta_i_deg,
        delta_raan_deg=delta_raan_deg,
        raan_sun_sync_target_deg=raan_sun_sync_target_deg,
        e_end_osculating=e_end_osculating,
        delta_e_same_u=delta_e_same_u,
        delta_argp_deg=delta_argp_deg,
        argp_J2_target_deg=argp_J2_target_deg,
        delta_r_iau_mars_km=delta_r_iau_mars_km,
        delta_v_iau_mars_kmps=delta_v_iau_mars_kmps,
        fluence_by_target_J_per_m2=tuple(fluence_by_target),
        n_windows_by_target=tuple(n_windows_by_target),
        window_initialization_mode=str(refined.initialization_mode),
        window_search_modes=tuple(refined.window_search_modes),
        n_window_continuation_fallbacks=len(
            refined.continuation_fallback_reasons
        ),
        n_propagations=int(refined.n_propagations),
    )


def evaluate_cruise(
    beta_rad: float,
    phi_u_rad: float,
    config: OrbitConfig,
) -> EvaluationResult:
    """Constant-cruise evaluation. Thin wrapper around
    ``evaluate_cruise_general`` with the constant cruise factory.

    Builds ``cruise.sun_offset_from_state(β, φ_u, ...)`` from the
    config's frozen initial state and reduces a 1-sol propagation to
    (Δa, fluence, e_max, window counts).
    """
    return evaluate_cruise_general(
        np.array([float(beta_rad), float(phi_u_rad)], dtype=float),
        config,
        _constant_cruise_factory,
    )


COST_METRIC_DELTA_A = "delta_a"
COST_METRIC_E_MAX = "e_max"
COST_METRIC_MULTI_ELEMENT_WEIGHTED = "multi_element_weighted"
COST_METRIC_CLOSURE_WEIGHTED = "closure_weighted"
COST_METRIC_RV_CLOSURE_WEIGHTED = "rv_closure_weighted"
_VALID_COST_METRICS = frozenset({
    COST_METRIC_DELTA_A,
    COST_METRIC_E_MAX,
    COST_METRIC_MULTI_ELEMENT_WEIGHTED,
    COST_METRIC_CLOSURE_WEIGHTED,
    COST_METRIC_RV_CLOSURE_WEIGHTED,
})


# Optimizer algorithm selector for ``optimize_cruise_general``. Nelder-Mead is
# the default; differential evolution provides global basin exploration.
ALGORITHM_NELDER_MEAD = "nelder-mead"
ALGORITHM_DIFFERENTIAL_EVOLUTION = "differential_evolution"
_VALID_ALGORITHMS = frozenset({
    ALGORITHM_NELDER_MEAD,
    ALGORITHM_DIFFERENTIAL_EVOLUTION,
})


def warm_start_population_for_de(
    x_prior: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    popsize: int,
    *,
    jitter_frac: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Build an ``(M, N)`` initial population for warm-starting
    ``scipy.optimize.differential_evolution``.

    ``scipy.optimize.differential_evolution`` accepts an array-valued
    ``init`` keyword argument; when provided as ``(M, N)`` ndarray the
    solver uses that as its initial population (M = number of members,
    N = problem dimension), and the ``popsize`` kwarg is ignored.
    See SciPy v1.x docs: "If init is an array, then the solver behaves
    as if it has been initialized using the given array (which must be
    of shape (M, len(x)))."

    The returned population has:
        - row 0 = ``x_prior`` clamped per-axis to ``bounds`` (so the
          prior optimum is exactly preserved as a population member;
          DE's ``polish=True`` then refines from this point if it is
          the best)
        - rows 1..(M-1) = ``x_prior`` plus per-axis Gaussian jitter with
          standard deviation ``jitter_frac * (hi - lo)``, clamped to
          per-axis bounds.

    M = ``popsize * N`` to mirror the cold-start scipy DE convention
    (``popsize=15`` default → 15 N members) so warm-start population
    diversity is comparable to cold-start.

    Parameters
    ----------
    x_prior
        Prior optimum / warm-start centre, shape ``(N,)``. May be
        partly outside ``bounds``; values outside are clamped.
    bounds
        Per-axis ``(lo, hi)`` tuples. Length defines ``N``.
    popsize
        Population multiplier. Total population size ``M = popsize * N``.
        Must be ``>= 1``. (SciPy DE requires the active population to be
        at least 5 for differential mutation; with N >= 5 and popsize=1
        this helper produces a population at the floor. Defaults
        elsewhere use popsize >= 5.)
    jitter_frac
        Gaussian jitter standard deviation as a fraction of each axis's
        ``(hi - lo)`` range. ``0.05`` → 5 % of axis range, σ. ``0.0``
        produces a constant population (all rows = clipped ``x_prior``);
        useful for tests but not recommended for live runs (DE differential
        mutation needs population diversity). Must be ``>= 0``.
    seed
        Optional PRNG seed for reproducible jitter.

    Returns
    -------
    np.ndarray of shape ``(M, N)``, dtype float, with row 0 the clipped
    prior and rows 1..(M-1) jittered.

    Raises
    ------
    ValueError
        If ``x_prior`` is not a 1-D array, if ``len(x_prior) != len(bounds)``,
        if ``popsize < 1``, if ``jitter_frac < 0``, or if any
        ``bounds[i]`` has ``lo >= hi``.
    """
    x_prior_arr = np.asarray(x_prior, dtype=float).ravel()
    if x_prior_arr.ndim != 1 or x_prior_arr.size == 0:
        raise ValueError(
            f"x_prior must be a non-empty 1-D array, got shape "
            f"{np.asarray(x_prior).shape}"
        )
    bounds_list = [tuple(b) for b in bounds]
    if len(bounds_list) != x_prior_arr.size:
        raise ValueError(
            f"bounds length must equal x_prior length, got "
            f"len(bounds)={len(bounds_list)}, len(x_prior)={x_prior_arr.size}"
        )
    for i, (lo, hi) in enumerate(bounds_list):
        if lo >= hi:
            raise ValueError(
                f"bounds[{i}] must satisfy lo < hi, got {(lo, hi)}"
            )
    if popsize < 1:
        raise ValueError(f"popsize must be >= 1, got {popsize}")
    if jitter_frac < 0.0:
        raise ValueError(f"jitter_frac must be >= 0, got {jitter_frac}")

    n_dim = x_prior_arr.size
    n_pop = popsize * n_dim
    los = np.array([b[0] for b in bounds_list], dtype=float)
    his = np.array([b[1] for b in bounds_list], dtype=float)
    ranges = his - los

    population = np.empty((n_pop, n_dim), dtype=float)
    population[0] = np.clip(x_prior_arr, los, his)
    if n_pop > 1:
        rng = np.random.default_rng(seed)
        sigma = jitter_frac * ranges
        # Draw (n_pop - 1, n_dim) standard normals and scale per-axis.
        jitter = rng.standard_normal(size=(n_pop - 1, n_dim)) * sigma
        candidates = x_prior_arr[np.newaxis, :] + jitter
        population[1:] = np.clip(candidates, los, his)
    return population


# Multi-element cost: default per-element scales are pinned to the baseline
# natural-drift orders of magnitude. Sized so that the
# residual contribution of any single element is O(1) at the natural
# drift, allowing the optimiser to find values that beat baseline AND
# permitting weights all = 1.0 by default to give a fair O(1) summation.
DEFAULT_SCALE_DELTA_A_KM_PER_SOL = 0.5         # natural ≈ 0.58 km/sol
DEFAULT_SCALE_E_MAX = 0.005                    # baseline e_max ≈ 0.003
DEFAULT_SCALE_DELTA_I_DEG = 0.1                # natural Δi ≈ 0.10 deg/sol
DEFAULT_SCALE_DELTA_RAAN_VS_SUNSYNC_DEG = 0.01 # natural miss ~0.005-0.01°/sol

DEFAULT_MULTI_ELEMENT_SCALES: Mapping[str, float] = {
    "delta_a": DEFAULT_SCALE_DELTA_A_KM_PER_SOL,
    "e_max": DEFAULT_SCALE_E_MAX,
    "delta_i": DEFAULT_SCALE_DELTA_I_DEG,
    "delta_raan_vs_sunsync": DEFAULT_SCALE_DELTA_RAAN_VS_SUNSYNC_DEG,
}

DEFAULT_MULTI_ELEMENT_WEIGHTS: Mapping[str, float] = {
    "delta_a": 1.0,
    "e_max": 1.0,
    "delta_i": 1.0,
    "delta_raan_vs_sunsync": 1.0,
}

_MULTI_ELEMENT_NAMES = frozenset(DEFAULT_MULTI_ELEMENT_SCALES.keys())


# Closure-targeted cost uses ``delta_e_same_u`` instead of ``e_max`` and
# ``delta_e_same_u`` (intra-propagation Δe sampled at matched orbital
# phase via the ascending-node Poincaré section, cancelling the J_2
# short-period floor) and adds ``delta_argp_vs_J2`` (signed Δω miss
# vs the Brouwer secular J_2 target, in deg). Δa, Δi, ΔΩ-sunsync
# carry over identically from the multi-element cost.
#
# Scales pinned by a K=12, sigma=0.018, alpha=0, one-sol baseline:
# drifts of |delta_e_same_u| ~ 2.5e-3 and |delta_argp_miss| ~ 5.9°.
# Default scales sit at ~3× natural so the term contributes ~0.1 at
# baseline and the optimizer has headroom to push terms below 1.0.
DEFAULT_SCALE_DELTA_E_SAME_U = 0.008          # natural |Δe_u| ≈ 2.5e-3 at K=12 σ=0.018
DEFAULT_SCALE_DELTA_ARGP_VS_J2_DEG = 18.0     # natural |Δω miss| ≈ 5.9°/sol

DEFAULT_CLOSURE_SCALES: Mapping[str, float] = {
    "delta_a": DEFAULT_SCALE_DELTA_A_KM_PER_SOL,
    "delta_e_same_u": DEFAULT_SCALE_DELTA_E_SAME_U,
    "delta_i": DEFAULT_SCALE_DELTA_I_DEG,
    "delta_raan_vs_sunsync": DEFAULT_SCALE_DELTA_RAAN_VS_SUNSYNC_DEG,
    "delta_argp_vs_J2": DEFAULT_SCALE_DELTA_ARGP_VS_J2_DEG,
}

DEFAULT_CLOSURE_WEIGHTS: Mapping[str, float] = {
    "delta_a": 1.0,
    "delta_e_same_u": 1.0,
    "delta_i": 1.0,
    "delta_raan_vs_sunsync": 1.0,
    "delta_argp_vs_J2": 1.0,
}

_CLOSURE_NAMES = frozenset(DEFAULT_CLOSURE_SCALES.keys())


# Mars-fixed (r, v) Poincaré-map closure cost. Osculating-element residuals
# cover only 4 of 6 d.o.f.; Δω weight=0
# because osculating argp is degenerate at near-circular orbits, ΔM
# missing entirely) with two scalar 3-vector norm residuals computed in
# the IAU_MARS body-fixed frame:
#
#   |Δr_iau_mars|  =  ||r_iau_mars(sol_end) − r_iau_mars(sol_start)||  km
#   |Δv_iau_mars|  =  ||v_iau_mars(sol_end) − v_iau_mars(sol_start)||  km/s
#
# Six scalar residuals (3 r + 3 v) are collapsed into two norms while retaining
# the full Poincaré-map fixed-point condition.
DEFAULT_SCALE_DELTA_R_IAU_MARS_KM = 1000.0
DEFAULT_SCALE_DELTA_V_IAU_MARS_KMPS = 0.8

# The e_max scale makes eccentricity a secondary tiebreaker: with unit weight,
# e_max=0.01 contributes 1e-6, while the term remains below 0.01 for every
# physical eccentricity e<1.
DEFAULT_SCALE_E_MAX_RV_CLOSURE = 10.0

# The fluence reward promotes fluence to a primary cost objective via a
# NEGATIVE quadratic term:
#   fluence_reward = -w_f × (fluence_J_per_m2 / scale_f)²
# Cost decreases as fluence increases. The 28 J/m² scale is representative of
# the target-delivery magnitude used by this objective.
#
# The fluence_penalty is a conditional shortfall guard. When the
# reward is enabled, it applies at all fluence levels.
DEFAULT_SCALE_FLUENCE_J_PER_M2 = 28.0

DEFAULT_RV_CLOSURE_SCALES: Mapping[str, float] = {
    "delta_r_iau_mars": DEFAULT_SCALE_DELTA_R_IAU_MARS_KM,
    "delta_v_iau_mars": DEFAULT_SCALE_DELTA_V_IAU_MARS_KMPS,
    "e_max": DEFAULT_SCALE_E_MAX_RV_CLOSURE,
    "fluence": DEFAULT_SCALE_FLUENCE_J_PER_M2,
    # Scale for the optional in-phase per-sol nodal-rate error term.
    "delta_raan_vs_sunsync": 0.01,  # deg/sol
}

DEFAULT_RV_CLOSURE_WEIGHTS: Mapping[str, float] = {
    "delta_r_iau_mars": 1.0,
    "delta_v_iau_mars": 1.0,
    "e_max": 1.0,
    # Optional reward and control terms are disabled by default.
    "fluence": 0.0,
    "delta_raan_vs_sunsync": 0.0,
}

_RV_CLOSURE_NAMES = frozenset(DEFAULT_RV_CLOSURE_SCALES.keys())


# PI cumulative-Δr_iau / Δv_iau feedback in rv_closure_weighted mirrors the
# K_i pattern for closure_weighted.
#
# Telescoping: Σ_{k=1..K} per-sol Δr_iau_k = r_iau(end K) − r_iau(start 1),
# because adjacent sol boundaries share the same ET so sxform terms
# cancel in the body-fixed sum. So cumulative body-fixed drift is the
# simple vector sum of per-sol Δr_iau 3-vectors — no cross-sol frame
# transforms needed. Same for Δv_iau.
#
# The cumulative scales represent 5 km of body-fixed displacement and 5 m/s
# of velocity mismatch. The squared-norm penalty preserves vector direction in
# its gradient and drives the current-sol residual against accumulated drift.
#
# References: PI controller form per Åström & Murray, "Feedback Systems"
# (Princeton 2008) §10.3; squared-norm aggregate-quantity penalty per
# Boyd & Vandenberghe, "Convex Optimization" (Cambridge 2004) §4.6.
#
# Cumulative scales remain separate from DEFAULT_RV_CLOSURE_SCALES because
# they describe cumulative body-fixed displacement rather than per-sol
# residuals.
DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM = 5.0
DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS = 0.005


def _valid_element_names_for(cost_metric: str) -> frozenset:
    """Return the recognised element-name vocabulary for a cost metric."""
    if cost_metric == COST_METRIC_MULTI_ELEMENT_WEIGHTED:
        return _MULTI_ELEMENT_NAMES
    if cost_metric == COST_METRIC_CLOSURE_WEIGHTED:
        return _CLOSURE_NAMES
    if cost_metric == COST_METRIC_RV_CLOSURE_WEIGHTED:
        return _RV_CLOSURE_NAMES
    return frozenset()


def _validate_multi_element_overrides(
    multi_element_weights: Optional[Mapping[str, float]],
    multi_element_scales: Optional[Mapping[str, float]],
    cost_metric: str,
) -> None:
    """Validate that weights/scales overrides are used only with a
    weighted-sum cost (``multi_element_weighted``, ``closure_weighted``,
    or ``rv_closure_weighted``). Raises ``ValueError`` otherwise. Also
    validates per-element keys + sign of values when overrides are
    supplied; the recognised vocabulary is dispatched per cost metric
    via ``_valid_element_names_for``.
    """
    valid_names = _valid_element_names_for(cost_metric)
    if not valid_names:
        if multi_element_weights is not None:
            raise ValueError(
                "multi_element_weights only valid with cost_metric in "
                f"{{{COST_METRIC_MULTI_ELEMENT_WEIGHTED!r}, "
                f"{COST_METRIC_CLOSURE_WEIGHTED!r}, "
                f"{COST_METRIC_RV_CLOSURE_WEIGHTED!r}}}, "
                f"got cost_metric={cost_metric!r}"
            )
        if multi_element_scales is not None:
            raise ValueError(
                "multi_element_scales only valid with cost_metric in "
                f"{{{COST_METRIC_MULTI_ELEMENT_WEIGHTED!r}, "
                f"{COST_METRIC_CLOSURE_WEIGHTED!r}, "
                f"{COST_METRIC_RV_CLOSURE_WEIGHTED!r}}}, "
                f"got cost_metric={cost_metric!r}"
            )
        return
    if multi_element_weights is not None:
        for k, v in multi_element_weights.items():
            if k not in valid_names:
                raise ValueError(
                    f"multi_element_weights[{k!r}] not a recognised element "
                    f"under cost_metric={cost_metric!r}; "
                    f"must be one of {sorted(valid_names)}"
                )
            if v < 0.0:
                raise ValueError(
                    f"multi_element_weights[{k!r}] must be >= 0, got {v}"
                )
    if multi_element_scales is not None:
        for k, v in multi_element_scales.items():
            if k not in valid_names:
                raise ValueError(
                    f"multi_element_scales[{k!r}] not a recognised element "
                    f"under cost_metric={cost_metric!r}; "
                    f"must be one of {sorted(valid_names)}"
                )
            if v <= 0.0:
                raise ValueError(
                    f"multi_element_scales[{k!r}] must be > 0, got {v}"
                )


def cruise_cost_with_breakdown(
    eval_result: EvaluationResult,
    fluence_floor_J_per_m2: float,
    penalty_lambda: float,
    cost_metric: str = COST_METRIC_DELTA_A,
    *,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
    cumulative_delta_e_same_u: float = 0.0,
    cumulative_integral_weight: float = 0.0,
    # Optional PI feedback on cumulative body-fixed Δr_iau / Δv_iau.
    cumulative_delta_r_iau_km: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_delta_v_iau_kmps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_integral_weight_r: float = 0.0,
    cumulative_integral_weight_v: float = 0.0,
    scale_cum_delta_r_iau_mars_km: float = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM,
    scale_cum_delta_v_iau_mars_kmps: float = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS,
    # Optional per-target fluence floors. When given,
    # the shortfall penalty becomes λ · Σ_t max(0, floor_t − F_t)²
    # over eval_result.fluence_by_target_J_per_m2; a config/eval length
    # mismatch raises explicitly. When None (default), the
    # scalar floor applies to total fluence. The fluence reward term is
    # unaffected (total fluence)
    # in both modes.
    fluence_floor_by_target_J_per_m2: Optional[Sequence[float]] = None,
) -> Tuple[float, Tuple[Tuple[str, float], ...]]:
    """Cost + per-element breakdown.

    Returns ``(cost, breakdown)`` where ``cost`` is the scalar to
    minimise and ``breakdown`` is a tuple of ``(element_name, value)``
    pairs recording the contribution of each term BEFORE summation.
    The transparent logging path writes the breakdown to
    INFO and to ``EvaluationResult.cost_breakdown`` so the per-eval
    history CSV captures it.

    Cost forms:

      cost_metric = "delta_a" (default):
          cost = |Δa_per_sol_km| + λ · max(0, floor − fluence)²
          breakdown = ()

      cost_metric = "e_max":
          cost = e_max + λ · max(0, floor − fluence)²
          breakdown = ()

      cost_metric = "multi_element_weighted":
          residual = Σ_k w_k · (Δ_k / scale_k)²
                k ∈ {delta_a, e_max, delta_i, delta_raan_vs_sunsync}
                Δ_delta_a              = delta_a_per_sol_km
                Δ_e_max                = e_max
                Δ_delta_i              = delta_i_deg
                Δ_delta_raan_vs_sunsync = (delta_raan_deg
                                          − raan_sun_sync_target_deg)
          cost = residual + λ · max(0, floor − fluence)²
          breakdown = (
              ("delta_a", w_a · (...)²),
              ("e_max", w_e · (...)²),
              ("delta_i", w_i · (...)²),
              ("delta_raan_vs_sunsync", w_Ω · (...)²),
              ("fluence_penalty", penalty_value),
          )

      cost_metric = "closure_weighted":
          residual = Σ_k w_k · (Δ_k / scale_k)²
                k ∈ {delta_a, delta_e_same_u, delta_i,
                     delta_raan_vs_sunsync, delta_argp_vs_J2}
                Δ_delta_a              = delta_a_per_sol_km
                Δ_delta_e_same_u       = delta_e_same_u
                                          (Brouwer δe_sp cancels via
                                           same-u sampling — secular Δ⟨e⟩)
                Δ_delta_i              = delta_i_deg
                Δ_delta_raan_vs_sunsync = (delta_raan_deg
                                          − raan_sun_sync_target_deg)
                Δ_delta_argp_vs_J2     = (delta_argp_deg
                                          − argp_J2_target_deg)
          breakdown = (
              ("delta_a", ...), ("delta_e_same_u", ...),
              ("delta_i", ...), ("delta_raan_vs_sunsync", ...),
              ("delta_argp_vs_J2", ...), ("fluence_penalty", ...),
          )

      cost_metric = "rv_closure_weighted":
          residual = w_r · (||Δr_iau_mars|| / scale_r)²
                   + w_v · (||Δv_iau_mars|| / scale_v)²
                   + w_e · (e_max / scale_e_max)²
                Δr_iau_mars = r_iau_mars(sol_end) − r_iau_mars(sol_start)
                Δv_iau_mars = v_iau_mars(sol_end) − v_iau_mars(sol_start)
                              (sxform-transformed so velocity Coriolis
                               cross-term is included)
                e_max       = within-sol peak osculating eccentricity
          (r, v) closure in IAU_MARS implies all six classical orbital
          elements + body-fixed orbit-plane orientation close at sol
          boundary; e_max is the remaining within-sol concern (orbit can
          swing eccentric mid-sol then return to closure). Default
          scale_e_max=10.0 sets e_max as a soft tiebreaker that is
          strictly subordinate to closure for all physical eccentricities
          (e<1).
          breakdown = (
              ("delta_r_iau_mars", ...), ("delta_v_iau_mars", ...),
              ("e_max", ...),
              ("cum_delta_r_iau_mars", ...), ("cum_delta_v_iau_mars", ...),
              ("fluence_reward", ...), ("fluence_penalty", ...),
          )

          The fluence reward adds a negative quadratic term
          −w_f · (fluence_J_per_m2 / scale_f)² to
          residual. Promotes fluence delivery to a primary cost objective.
          Default weight=0.0 leaves the term inactive; set the "fluence"
          weight via ``multi_element_weights`` to opt in. Default scale=28.0
          sets the characteristic fluence magnitude. Coexists with the
          fluence_penalty
          shortfall guard (which stays at zero in normal operation).

    Default scales are defined by the ``DEFAULT_SCALE_*`` module constants.
    Override per-element via ``multi_element_weights`` /
    ``multi_element_scales`` mappings (partial overrides are merged
    with defaults). The same override-mapping arguments are reused for
    closure-weighted; the recognised element
    vocabulary is dispatched per cost metric.

    PI cumulative Δe feedback (closure_weighted only):

      cumulative_delta_e_same_u   Running sum of prior sols'
                                  delta_e_same_u. Fixed for a given
                                  sol's optimization; the calling
                                  workflow tracks this across sols.
      cumulative_integral_weight  Gain K_i >= 0. When > 0, adds:
                                  K_i · ((cum + Δe_current) / scale_de)²
                                  Biases optimizer toward zeroing the
                                  running sum. Default 0.0 = no integral term.

    PI cumulative Δr_iau / Δv_iau feedback (rv_closure_weighted only):

      cumulative_delta_r_iau_km   Running vector sum (3-tuple) of prior
                                  sols' delta_r_iau_mars_km. Per the
                                  telescoping property (adjacent sol
                                  boundaries share the same ET; sxform
                                  terms cancel in the body-fixed sum),
                                  this IS r_iau(end of sol K-1) − r_iau
                                  (start of sol 1) at the START of sol K.
                                  No cross-sol frame transformation
                                  needed — direct vector sum in body-
                                  fixed coords.
      cumulative_delta_v_iau_kmps Same for velocity (km/s).
      cumulative_integral_weight_r  Gain K_r >= 0. When > 0, adds:
                                  K_r · (||cum_r + Δr_current|| / scale_cum_r)²
                                  The norm preserves vector-direction
                                  info through the gradient (∇‖a+x‖² =
                                  2(a+x)), biasing this-sol Δr to
                                  oppose the prior cumulative. Default
                                  0.0 = no integral term, preserving the
                                  default breakdown.
      cumulative_integral_weight_v  Symmetric for v.
      scale_cum_delta_r_iau_mars_km   Calibration scale for the cum-r
                                  term. Default 5.0 km.
      scale_cum_delta_v_iau_mars_kmps Default 0.005 km/s = 5 m/s.
    """
    if penalty_lambda < 0.0:
        raise ValueError(
            f"penalty_lambda must be >= 0, got {penalty_lambda}"
        )
    if not math.isfinite(fluence_floor_J_per_m2):
        raise ValueError(
            f"fluence_floor_J_per_m2 must be finite, got {fluence_floor_J_per_m2}"
        )
    if cost_metric not in _VALID_COST_METRICS:
        raise ValueError(
            f"cost_metric must be one of {sorted(_VALID_COST_METRICS)}, "
            f"got {cost_metric!r}"
        )
    _validate_multi_element_overrides(
        multi_element_weights, multi_element_scales, cost_metric,
    )

    if fluence_floor_by_target_J_per_m2 is None:
        shortfall = max(
            0.0,
            fluence_floor_J_per_m2 - eval_result.total_fluence_J_per_m2,
        )
        penalty = penalty_lambda * shortfall * shortfall
    else:
        floors = [float(f) for f in fluence_floor_by_target_J_per_m2]
        if any(not math.isfinite(f) for f in floors):
            raise ValueError(
                f"fluence_floor_by_target_J_per_m2 entries must be "
                f"finite, got {floors}"
            )
        by_target = eval_result.fluence_by_target_J_per_m2
        if len(by_target) != len(floors):
            raise ValueError(
                f"fluence_floor_by_target_J_per_m2 has {len(floors)} "
                f"floor(s) but eval_result carries "
                f"{len(by_target)} per-target fluence value(s); "
                f"config/eval target lists are inconsistent."
            )
        penalty = 0.0
        for floor_t, fluence_t in zip(floors, by_target):
            shortfall_t = max(0.0, floor_t - float(fluence_t))
            penalty += penalty_lambda * shortfall_t * shortfall_t

    if cost_metric == COST_METRIC_DELTA_A:
        return abs(eval_result.delta_a_per_sol_km) + penalty, ()
    if cost_metric == COST_METRIC_E_MAX:
        return float(eval_result.e_max) + penalty, ()

    if cost_metric == COST_METRIC_MULTI_ELEMENT_WEIGHTED:
        weights = dict(DEFAULT_MULTI_ELEMENT_WEIGHTS)
        if multi_element_weights:
            weights.update(multi_element_weights)
        scales = dict(DEFAULT_MULTI_ELEMENT_SCALES)
        if multi_element_scales:
            scales.update(multi_element_scales)

        raan_miss_deg = (
            eval_result.delta_raan_deg
            - eval_result.raan_sun_sync_target_deg
        )

        delta_a_term = (
            weights["delta_a"]
            * (eval_result.delta_a_per_sol_km / scales["delta_a"]) ** 2
        )
        e_max_term = (
            weights["e_max"]
            * (eval_result.e_max / scales["e_max"]) ** 2
        )
        delta_i_term = (
            weights["delta_i"]
            * (eval_result.delta_i_deg / scales["delta_i"]) ** 2
        )
        delta_raan_term = (
            weights["delta_raan_vs_sunsync"]
            * (raan_miss_deg / scales["delta_raan_vs_sunsync"]) ** 2
        )

        residual = (
            delta_a_term + e_max_term + delta_i_term + delta_raan_term
        )
        cost = residual + penalty

        breakdown: Tuple[Tuple[str, float], ...] = (
            ("delta_a", float(delta_a_term)),
            ("e_max", float(e_max_term)),
            ("delta_i", float(delta_i_term)),
            ("delta_raan_vs_sunsync", float(delta_raan_term)),
            ("fluence_penalty", float(penalty)),
        )
        return cost, breakdown

    if cost_metric == COST_METRIC_CLOSURE_WEIGHTED:
        weights = dict(DEFAULT_CLOSURE_WEIGHTS)
        if multi_element_weights:
            weights.update(multi_element_weights)
        scales = dict(DEFAULT_CLOSURE_SCALES)
        if multi_element_scales:
            scales.update(multi_element_scales)

        raan_miss_deg = (
            eval_result.delta_raan_deg
            - eval_result.raan_sun_sync_target_deg
        )
        argp_miss_deg = (
            eval_result.delta_argp_deg
            - eval_result.argp_J2_target_deg
        )

        delta_a_term = (
            weights["delta_a"]
            * (eval_result.delta_a_per_sol_km / scales["delta_a"]) ** 2
        )
        delta_e_same_u_term = (
            weights["delta_e_same_u"]
            * (eval_result.delta_e_same_u / scales["delta_e_same_u"]) ** 2
        )
        delta_i_term = (
            weights["delta_i"]
            * (eval_result.delta_i_deg / scales["delta_i"]) ** 2
        )
        delta_raan_term = (
            weights["delta_raan_vs_sunsync"]
            * (raan_miss_deg / scales["delta_raan_vs_sunsync"]) ** 2
        )
        delta_argp_term = (
            weights["delta_argp_vs_J2"]
            * (argp_miss_deg / scales["delta_argp_vs_J2"]) ** 2
        )

        # PI integral term: penalizes (cum_prior + Δe_current) / scale_de.
        cum_delta_e_term = 0.0
        if cumulative_integral_weight > 0.0:
            cum_total = (
                cumulative_delta_e_same_u + eval_result.delta_e_same_u
            )
            cum_delta_e_term = (
                cumulative_integral_weight
                * (cum_total / scales["delta_e_same_u"]) ** 2
            )

        residual = (
            delta_a_term
            + delta_e_same_u_term
            + delta_i_term
            + delta_raan_term
            + delta_argp_term
            + cum_delta_e_term
        )
        cost = residual + penalty

        breakdown = (
            ("delta_a", float(delta_a_term)),
            ("delta_e_same_u", float(delta_e_same_u_term)),
            ("delta_i", float(delta_i_term)),
            ("delta_raan_vs_sunsync", float(delta_raan_term)),
            ("delta_argp_vs_J2", float(delta_argp_term)),
            ("cum_delta_e", float(cum_delta_e_term)),
            ("fluence_penalty", float(penalty)),
        )
        return cost, breakdown

    # COST_METRIC_RV_CLOSURE_WEIGHTED.
    # 3-term cost: ||Δr_iau|| + ||Δv_iau|| + e_max, all weight 1.0 by default.
    # Closure of (r, v) in IAU_MARS implies closure of all six classical
    # orbital elements + body-fixed orbit-plane orientation; e_max is the only
    # remaining within-sol
    # physical concern, added as a soft tiebreaker at scale = 10.0 so it
    # is strictly subordinate to closure for all real eccentricities.
    weights = dict(DEFAULT_RV_CLOSURE_WEIGHTS)
    if multi_element_weights:
        weights.update(multi_element_weights)
    scales = dict(DEFAULT_RV_CLOSURE_SCALES)
    if multi_element_scales:
        scales.update(multi_element_scales)

    delta_r_norm_km = float(np.linalg.norm(
        np.asarray(eval_result.delta_r_iau_mars_km, dtype=float)
    ))
    delta_v_norm_kmps = float(np.linalg.norm(
        np.asarray(eval_result.delta_v_iau_mars_kmps, dtype=float)
    ))

    delta_r_term = (
        weights["delta_r_iau_mars"]
        * (delta_r_norm_km / scales["delta_r_iau_mars"]) ** 2
    )
    delta_v_term = (
        weights["delta_v_iau_mars"]
        * (delta_v_norm_kmps / scales["delta_v_iau_mars"]) ** 2
    )
    e_max_term = (
        weights["e_max"]
        * (eval_result.e_max / scales["e_max"]) ** 2
    )

    # PI integral terms penalize cumulative body-fixed (Δr_iau, Δv_iau) summed
    # across sols. By the telescoping property, adjacent sol
    # boundaries share the same ET, so sxform terms in the per-sol Δ
    # cancel pairwise across sols, and Σ per-sol Δr_iau_k = r_iau(end K)
    # − r_iau(start 1). Therefore cum_prior + Δr_current IS the body-
    # fixed displacement at the END of THIS sol relative to sol 1's
    # start. Zero weights make both terms vanish.
    cum_delta_r_term = 0.0
    if cumulative_integral_weight_r > 0.0:
        cum_total_r = (
            np.asarray(cumulative_delta_r_iau_km, dtype=float)
            + np.asarray(eval_result.delta_r_iau_mars_km, dtype=float)
        )
        cum_total_r_norm_km = float(np.linalg.norm(cum_total_r))
        cum_delta_r_term = (
            cumulative_integral_weight_r
            * (cum_total_r_norm_km / scale_cum_delta_r_iau_mars_km) ** 2
        )

    cum_delta_v_term = 0.0
    if cumulative_integral_weight_v > 0.0:
        cum_total_v = (
            np.asarray(cumulative_delta_v_iau_kmps, dtype=float)
            + np.asarray(eval_result.delta_v_iau_mars_kmps, dtype=float)
        )
        cum_total_v_norm_kmps = float(np.linalg.norm(cum_total_v))
        cum_delta_v_term = (
            cumulative_integral_weight_v
            * (cum_total_v_norm_kmps / scale_cum_delta_v_iau_mars_kmps) ** 2
        )

    # Negative-quadratic fluence reward. A zero weight leaves it inactive.
    fluence_reward_term = -(
        weights["fluence"]
        * (eval_result.total_fluence_J_per_m2 / scales["fluence"]) ** 2
    )

    # Optional in-phase nodal-rate control penalizes the per-sol difference
    # between actual ΔΩ and the Brouwer secular sun-synchronous rate. This
    # proportional term addresses periodic forcing, while cumulative integral
    # terms address secular drift. A zero weight leaves it inactive.
    node_rate_term = 0.0
    if weights.get("delta_raan_vs_sunsync", 0.0) > 0.0:
        raan_rate_miss_deg = (
            eval_result.delta_raan_deg
            - eval_result.raan_sun_sync_target_deg
        )
        node_rate_term = (
            weights["delta_raan_vs_sunsync"]
            * (raan_rate_miss_deg / scales["delta_raan_vs_sunsync"]) ** 2
        )

    residual = (
        delta_r_term + delta_v_term + e_max_term
        + cum_delta_r_term + cum_delta_v_term
        + fluence_reward_term + node_rate_term
    )
    cost = residual + penalty

    breakdown = (
        ("delta_r_iau_mars", float(delta_r_term)),
        ("delta_v_iau_mars", float(delta_v_term)),
        ("e_max", float(e_max_term)),
        ("cum_delta_r_iau_mars", float(cum_delta_r_term)),
        ("cum_delta_v_iau_mars", float(cum_delta_v_term)),
        ("fluence_reward", float(fluence_reward_term)),
        ("fluence_penalty", float(penalty)),
        ("delta_raan_vs_sunsync", float(node_rate_term)),
    )
    return cost, breakdown


def cruise_cost_from_eval(
    eval_result: EvaluationResult,
    fluence_floor_J_per_m2: float,
    penalty_lambda: float,
    cost_metric: str = COST_METRIC_DELTA_A,
    *,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
    cumulative_delta_e_same_u: float = 0.0,
    cumulative_integral_weight: float = 0.0,
    cumulative_delta_r_iau_km: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_delta_v_iau_kmps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_integral_weight_r: float = 0.0,
    cumulative_integral_weight_v: float = 0.0,
    scale_cum_delta_r_iau_mars_km: float = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM,
    scale_cum_delta_v_iau_mars_kmps: float = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS,
    fluence_floor_by_target_J_per_m2: Optional[Sequence[float]] = None,
) -> float:
    """Scalar cost from an existing ``EvaluationResult`` (no breakdown).

    Thin wrapper around :func:`cruise_cost_with_breakdown` that drops
    the breakdown tuple. Callers wanting the per-element residuals call
    :func:`cruise_cost_with_breakdown` directly.
    """
    cost, _ = cruise_cost_with_breakdown(
        eval_result,
        fluence_floor_J_per_m2,
        penalty_lambda,
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
        cumulative_delta_e_same_u=cumulative_delta_e_same_u,
        cumulative_integral_weight=cumulative_integral_weight,
        cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
        cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
        cumulative_integral_weight_r=cumulative_integral_weight_r,
        cumulative_integral_weight_v=cumulative_integral_weight_v,
        scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
        scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
        fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
    )
    return cost


def _format_breakdown(breakdown: Tuple[Tuple[str, float], ...]) -> str:
    """Render a cost_breakdown tuple as a compact log/print string.
    ``"Δa=1.50  e_max=0.36  Δi=0.50  ΔΩ-sunsync=0.40  penalty=0.00"``.
    """
    name_map = {
        "delta_a": "Δa",
        "e_max": "e_max",
        "delta_i": "Δi",
        "delta_raan_vs_sunsync": "ΔΩ-sunsync",
        "delta_e_same_u": "Δe_u",
        "delta_argp_vs_J2": "Δω-J2",
        "cum_delta_e": "cum_Δe",
        "delta_r_iau_mars": "Δr",
        "delta_v_iau_mars": "Δv",
        "cum_delta_r_iau_mars": "cum_Δr",
        "cum_delta_v_iau_mars": "cum_Δv",
        "fluence_reward": "flu_R",
        "fluence_penalty": "penalty",
    }
    return "  ".join(
        f"{name_map.get(k, k)}={v:.4f}" for k, v in breakdown
    )


def cruise_cost(
    x: np.ndarray,
    config: OrbitConfig,
    fluence_floor_J_per_m2: float,
    penalty_lambda: float,
    history: Optional[list] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    *,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
) -> float:
    """Optimizer-facing scalar cost. ``x = [β_rad, φ_u_rad]``.

    When ``history`` is supplied, every evaluation appends an
    ``EvaluationResult`` so the caller can record the full trace
    (Nelder-Mead's iteration callback only fires once per simplex
    update, while the full trace requires every function evaluation). Under
    ``cost_metric="multi_element_weighted"`` the per-element breakdown
    is also recorded on each ``EvaluationResult.cost_breakdown`` and
    logged at INFO for transparency.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.shape != (2,):
        raise ValueError(
            f"cruise_cost expects x shape (2,), got {x_arr.shape}"
        )
    eval_result = evaluate_cruise(float(x_arr[0]), float(x_arr[1]), config)
    cost, breakdown = cruise_cost_with_breakdown(
        eval_result, fluence_floor_J_per_m2, penalty_lambda,
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
    )
    if breakdown:
        eval_result = replace(eval_result, cost_breakdown=breakdown)
    if history is not None:
        history.append(eval_result)
    if breakdown:
        logger.info(
            "cruise_cost[%s] eval %d: β=%.4f rad (%.2f deg), "
            "φ_u=%.4f rad (%.2f deg) → cost=%.6f  (%s)  wall=%.2f s",
            cost_metric,
            len(history) if history is not None else -1,
            x_arr[0], math.degrees(x_arr[0]),
            x_arr[1], math.degrees(x_arr[1]),
            cost,
            _format_breakdown(breakdown),
            eval_result.wall_s,
        )
    else:
        logger.info(
            "cruise_cost[%s] eval %d: β=%.4f rad (%.2f deg), "
            "φ_u=%.4f rad (%.2f deg) → Δa/sol=%+0.4f km, e_max=%.5f, "
            "fluence=%.4f J/m^2, cost=%.6f, wall=%.2f s",
            cost_metric,
            len(history) if history is not None else -1,
            x_arr[0], math.degrees(x_arr[0]),
            x_arr[1], math.degrees(x_arr[1]),
            eval_result.delta_a_per_sol_km,
            eval_result.e_max,
            eval_result.total_fluence_J_per_m2,
            cost,
            eval_result.wall_s,
        )
    return cost


def optimize_cruise_general(
    config: OrbitConfig,
    *,
    x0: np.ndarray,
    cruise_factory: CruiseFactory,
    bounds: Sequence[Tuple[float, float]],
    fluence_floor_J_per_m2: float,
    penalty_lambda: float = 1.0,
    nelder_mead_options: Optional[dict] = None,
    baseline_eval: Optional[EvaluationResult] = None,
    baseline_x: Optional[np.ndarray] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
    algorithm: str = ALGORITHM_NELDER_MEAD,
    de_options: Optional[dict] = None,
    cumulative_delta_e_same_u: float = 0.0,
    cumulative_integral_weight: float = 0.0,
    cumulative_delta_r_iau_km: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_delta_v_iau_kmps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_integral_weight_r: float = 0.0,
    cumulative_integral_weight_v: float = 0.0,
    scale_cum_delta_r_iau_mars_km: float = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM,
    scale_cum_delta_v_iau_mars_kmps: float = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS,
    fluence_floor_by_target_J_per_m2: Optional[Sequence[float]] = None,
) -> OptimizationRun:
    """Parameterization-agnostic Nelder-Mead optimizer core.

    Drives ``scipy.optimize.minimize(method="Nelder-Mead", bounds=...)``
    over a decision vector ``x`` of arbitrary dimension. The
    ``cruise_factory`` callable maps ``x`` to an
    :class:`AttitudeCallable` per evaluation; ``bounds`` is a sequence
    of ``(lo, hi)`` tuples matching ``x``'s dimension. Records every
    objective evaluation (not just simplex updates) into
    ``OptimizationRun.history`` via a closure-list pattern.

    ``optimize_cruise`` and ``optimize_cruise_harmonic``
    are thin wrappers around this function with their family-specific
    bounds and ``cruise_factory`` defaults.

    Parameters
    ----------
    config
        Frozen ``OrbitConfig`` describing the fixed orbit, sail,
        target, and propagator / window-finder settings.
    x0
        Initial simplex centre, shape ``(n,)``. Must lie inside the
        per-component bounds. ``n`` matches the ``cruise_factory``'s
        expected dimensionality.
    cruise_factory
        Callable ``(x, config) -> AttitudeCallable``. The optimizer
        does not introspect ``x``'s structure beyond
        ``EvaluationResult.beta_rad = x[0]`` and
        ``EvaluationResult.phi_u_rad = x[-1]``.
    bounds
        Sequence of ``(lo, hi)`` tuples, length must equal ``len(x0)``.
    fluence_floor_J_per_m2
        Constraint floor for the quadratic exterior penalty.
    penalty_lambda
        Constraint-penalty coefficient.
    nelder_mead_options
        Forwarded to ``minimize(options=...)``. Sensible defaults:
        ``xatol=radians(0.1), fatol=0.001, maxiter=100, maxfev=100,
        adaptive=True``.
    baseline_eval
        Optional pre-computed baseline evaluation (e.g. at ``x = 0``).
        When omitted, ``optimize_cruise_general`` evaluates the
        baseline at ``baseline_x`` (if supplied) or at
        ``np.zeros_like(x0)`` (constant-cruise reduces to sun-pointing).
        Not counted against ``maxfev``.
    baseline_x
        Override the default baseline x of zero vector. Useful for
        warm-start scenarios where the baseline is the previous
        sol's optimum.
    algorithm
        Optimizer algorithm: one of ``ALGORITHM_NELDER_MEAD``
        (default) or ``ALGORITHM_DIFFERENTIAL_EVOLUTION``. NM is
        local but warm-startable from ``x0``; DE is global, ignores
        ``x0`` (uses ``bounds`` to seed a population), and runs
        ``polish=True`` by default for a final NM polish on the
        best vertex.
    de_options
        Forwarded to ``differential_evolution(...)`` as
        keyword arguments when ``algorithm == "differential_evolution"``.
        Sensible defaults (cite ``scipy.optimize.differential_evolution``
        v1.x docs):
        ``init="sobol"`` (better space-filling at low dim than
        ``"latinhypercube"``), ``popsize=15``, ``mutation=(0.5, 1.0)``,
        ``recombination=0.7``, ``tol=0.01``, ``maxiter=100``,
        ``polish=True``, ``seed=None``. Caller-supplied keys override
        the per-key defaults; non-defaulted scipy DE kwargs (workers,
        constraints, etc.) flow through untouched.

    Raises
    ------
    ValueError
        If ``x0``'s shape does not match ``len(bounds)``, if any
        bound has ``lo >= hi``, if any ``x0[i]`` is outside
        ``bounds[i]``, if ``penalty_lambda`` is negative, if
        ``cost_metric`` is invalid, or if ``algorithm`` is invalid.
    """
    x0_arr = np.asarray(x0, dtype=float).ravel()
    if x0_arr.ndim != 1 or x0_arr.size == 0:
        raise ValueError(
            f"x0 must be a non-empty 1-D array, got shape {x0_arr.shape}"
        )
    bounds_list = [tuple(b) for b in bounds]
    if len(bounds_list) != x0_arr.size:
        raise ValueError(
            f"bounds length must equal x0 length, got "
            f"len(bounds)={len(bounds_list)}, len(x0)={x0_arr.size}"
        )
    for i, (lo, hi) in enumerate(bounds_list):
        if lo >= hi:
            raise ValueError(
                f"bounds[{i}] must satisfy lo < hi, got {(lo, hi)}"
            )
        if not (lo <= x0_arr[i] <= hi):
            raise ValueError(
                f"x0[{i}]={x0_arr[i]} outside bounds[{i}]={(lo, hi)}"
            )
    if penalty_lambda < 0.0:
        raise ValueError(
            f"penalty_lambda must be >= 0, got {penalty_lambda}"
        )
    if cost_metric not in _VALID_COST_METRICS:
        raise ValueError(
            f"cost_metric must be one of {sorted(_VALID_COST_METRICS)}, "
            f"got {cost_metric!r}"
        )
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {sorted(_VALID_ALGORITHMS)}, "
            f"got {algorithm!r}"
        )
    _validate_multi_element_overrides(
        multi_element_weights, multi_element_scales, cost_metric,
    )

    t_total_0 = time.perf_counter()

    if baseline_eval is None:
        if baseline_x is None:
            base_x = np.zeros_like(x0_arr)
        else:
            base_x = np.asarray(baseline_x, dtype=float).ravel()
            if base_x.size != x0_arr.size:
                raise ValueError(
                    f"baseline_x size must equal x0 size, got "
                    f"{base_x.size} vs {x0_arr.size}"
                )
        logger.info(
            "optimize_cruise_general: computing baseline at x=%s", base_x
        )
        baseline_eval = evaluate_cruise_general(base_x, config, cruise_factory)
        logger.info(
            "baseline: Δa/sol=%+.4f km, fluence=%.4f J/m^2, e_max=%.5f",
            baseline_eval.delta_a_per_sol_km,
            baseline_eval.total_fluence_J_per_m2,
            baseline_eval.e_max,
        )

    history: list = []

    options = {
        "xatol": math.radians(0.1),
        "fatol": 0.001,
        "maxiter": 100,
        "maxfev": 100,
        "adaptive": True,
        "disp": False,
    }
    if nelder_mead_options:
        options.update(nelder_mead_options)

    floor = float(fluence_floor_J_per_m2)
    lam = float(penalty_lambda)

    # Penalty cost returned for infeasible simplex vertices (e.g. harmonic
    # cruise where α_0 ± α_amp leaves [0, π/2]). Finite + much larger than
    # any feasible cost (cost arithmetic gives O(1) for Δa, O(0.01) for
    # e_max, plus O(1e3) constraint penalty), so Nelder-Mead naturally
    # rejects the vertex.
    INFEASIBLE_COST = 1.0e9

    def cost_closure(x_inner: np.ndarray) -> float:
        try:
            eval_result = evaluate_cruise_general(
                x_inner, config, cruise_factory,
            )
        except ValueError as exc:
            x_str = ",".join(f"{math.degrees(float(v)):+.2f}" for v in x_inner)
            logger.warning(
                "optimize_cruise_general[%s] infeasible vertex x=[%s] deg: "
                "%s — returning penalty cost %.0e",
                cost_metric, x_str, exc, INFEASIBLE_COST,
            )
            return INFEASIBLE_COST
        cost, breakdown = cruise_cost_with_breakdown(
            eval_result, floor, lam, cost_metric=cost_metric,
            multi_element_weights=multi_element_weights,
            multi_element_scales=multi_element_scales,
            cumulative_delta_e_same_u=cumulative_delta_e_same_u,
            cumulative_integral_weight=cumulative_integral_weight,
            cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
            cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
            cumulative_integral_weight_r=cumulative_integral_weight_r,
            cumulative_integral_weight_v=cumulative_integral_weight_v,
            scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
            scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
            fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
        )
        if breakdown:
            eval_result = replace(eval_result, cost_breakdown=breakdown)
        history.append(eval_result)
        x_str = ",".join(f"{math.degrees(float(v)):+.2f}" for v in x_inner)
        if breakdown:
            logger.info(
                "optimize_cruise_general[%s] eval %d: x=[%s] deg → "
                "cost=%.6f  (%s)  wall=%.2f s",
                cost_metric, len(history), x_str,
                cost,
                _format_breakdown(breakdown),
                eval_result.wall_s,
            )
        else:
            logger.info(
                "optimize_cruise_general[%s] eval %d: x=[%s] deg → "
                "Δa/sol=%+0.4f km, e_max=%.5f, fluence=%.4f J/m^2, "
                "cost=%.6f, wall=%.2f s",
                cost_metric, len(history), x_str,
                eval_result.delta_a_per_sol_km,
                eval_result.e_max,
                eval_result.total_fluence_J_per_m2,
                cost,
                eval_result.wall_s,
            )
        return cost

    if algorithm == ALGORITHM_NELDER_MEAD:
        scipy_result = scipy_minimize(
            fun=cost_closure,
            x0=x0_arr,
            method="Nelder-Mead",
            bounds=bounds_list,
            options=options,
        )
    else:
        # Differential-evolution dispatch.
        # Per-key defaults (citing scipy.optimize.differential_evolution
        # v1.x docs); caller overrides via ``de_options`` win.
        de_kwargs = {
            "init": "sobol",
            "popsize": 15,
            "mutation": (0.5, 1.0),
            "recombination": 0.7,
            "tol": 0.01,
            "maxiter": 100,
            "polish": True,
            "seed": None,
        }
        if de_options:
            de_kwargs.update(de_options)
        # ``x0`` is intentionally not passed: DE seeds its population
        # from ``bounds`` via ``init``. NM-style warm-start is not part
        # of the DE algorithm; ``polish=True`` runs a final L-BFGS-B
        # refinement on the best vertex, approximating a local
        # Nelder-Mead-style refinement at the global basin.
        scipy_result = scipy_differential_evolution(
            func=cost_closure,
            bounds=bounds_list,
            **de_kwargs,
        )

    if not history:
        # Two routes can produce an empty history:
        #   (a) scipy genuinely returned without evaluating the cost at
        #       all because of an internal SciPy issue or malformed cost.
        #   (b) scipy DE was invoked with workers != 1 (a custom map
        #       callable like CloudpickleMap), so every population +
        #       polish-step evaluation happened in worker processes.
        #       Each worker's history list is its own copy-on-write
        #       fork; the parent's history is empty even though the
        #       optimization did real work.
        # Distinguish them by checking whether scipy_result.x carries a
        # finite candidate optimum. If yes, this is route (b): run
        # cost_closure once in the parent at scipy_result.x to
        # repopulate history with the optimum's eval, so the
        # downstream best_eval extraction has an evaluation to inspect.
        # Per src/reflectors/parallel.py docstring, the per-population
        # eval history under workers > 1 is a documented limitation;
        # this recovery preserves the OPTIMUM in history without
        # re-running the population.
        result_x = getattr(scipy_result, "x", None)
        if result_x is not None and np.all(np.isfinite(np.asarray(result_x, dtype=float))):
            logger.info(
                "optimize_cruise_general: parent history empty "
                "post-scipy (workers > 1 route); rerunning cost_closure "
                "at scipy_result.x to populate optimum eval"
            )
            cost_closure(np.asarray(result_x, dtype=float))
        if not history:
            raise RuntimeError(
                "optimize_cruise_general: scipy minimize completed without "
                "any objective evaluations. Internal SciPy issue or "
                "malformed cost."
            )
    best_idx = int(
        np.argmin([
            cruise_cost_from_eval(
                ev, floor, lam, cost_metric=cost_metric,
                multi_element_weights=multi_element_weights,
                multi_element_scales=multi_element_scales,
                cumulative_delta_e_same_u=cumulative_delta_e_same_u,
                cumulative_integral_weight=cumulative_integral_weight,
                cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
                cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
                cumulative_integral_weight_r=cumulative_integral_weight_r,
                cumulative_integral_weight_v=cumulative_integral_weight_v,
                scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
                scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
                fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
            )
            for ev in history
        ])
    )
    best_eval = history[best_idx]

    # Record the resolved weights/scales (defaults merged with any
    # caller overrides) so OptimizationRun is self-describing. All
    # weighted-sum metrics need this; single-element metrics
    # (delta_a, e_max) leave it empty.
    if cost_metric == COST_METRIC_MULTI_ELEMENT_WEIGHTED:
        resolved_weights = dict(DEFAULT_MULTI_ELEMENT_WEIGHTS)
        resolved_scales = dict(DEFAULT_MULTI_ELEMENT_SCALES)
    elif cost_metric == COST_METRIC_CLOSURE_WEIGHTED:
        resolved_weights = dict(DEFAULT_CLOSURE_WEIGHTS)
        resolved_scales = dict(DEFAULT_CLOSURE_SCALES)
    elif cost_metric == COST_METRIC_RV_CLOSURE_WEIGHTED:
        resolved_weights = dict(DEFAULT_RV_CLOSURE_WEIGHTS)
        resolved_scales = dict(DEFAULT_RV_CLOSURE_SCALES)
    else:
        resolved_weights = {}
        resolved_scales = {}
    if (
        cost_metric in {
            COST_METRIC_MULTI_ELEMENT_WEIGHTED,
            COST_METRIC_CLOSURE_WEIGHTED,
            COST_METRIC_RV_CLOSURE_WEIGHTED,
        }
        and multi_element_weights
    ):
        resolved_weights.update(multi_element_weights)
    if (
        cost_metric in {
            COST_METRIC_MULTI_ELEMENT_WEIGHTED,
            COST_METRIC_CLOSURE_WEIGHTED,
            COST_METRIC_RV_CLOSURE_WEIGHTED,
        }
        and multi_element_scales
    ):
        resolved_scales.update(multi_element_scales)

    return OptimizationRun(
        scipy_result=scipy_result,
        history=tuple(history),
        best_eval=best_eval,
        baseline_eval=baseline_eval,
        config=config,
        fluence_floor_J_per_m2=floor,
        penalty_lambda=lam,
        cost_metric=str(cost_metric),
        wall_total_s=time.perf_counter() - t_total_0,
        multi_element_weights=resolved_weights,
        multi_element_scales=resolved_scales,
    )


def optimize_cruise_polish_only(
    config: OrbitConfig,
    *,
    x0: np.ndarray,
    cruise_factory: CruiseFactory,
    bounds: Sequence[Tuple[float, float]],
    fluence_floor_J_per_m2: float,
    penalty_lambda: float = 1.0,
    baseline_eval: Optional[EvaluationResult] = None,
    baseline_x: Optional[np.ndarray] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
    cumulative_delta_e_same_u: float = 0.0,
    cumulative_integral_weight: float = 0.0,
    cumulative_delta_r_iau_km: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_delta_v_iau_kmps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_integral_weight_r: float = 0.0,
    cumulative_integral_weight_v: float = 0.0,
    scale_cum_delta_r_iau_mars_km: float = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM,
    scale_cum_delta_v_iau_mars_kmps: float = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS,
    fluence_floor_by_target_J_per_m2: Optional[Sequence[float]] = None,
    lbfgsb_options: Optional[dict] = None,
    fd_step_h: float = 1e-4,
    jac_workers: Optional["object"] = None,
) -> OptimizationRun:
    """L-BFGS-B local optimization with a parallel finite-difference Jacobian.

    The parent process runs ``scipy.optimize.minimize(method='L-BFGS-B')``;
    objective evaluations are recorded in the history, while Jacobian
    evaluations route through
    :func:`reflectors.parallel.parallel_fd_jacobian` with the
    optional ``jac_workers`` (a
    :class:`reflectors.parallel.CloudpickleMap`) for parallel dispatch
    of the 2N+1 cost evaluations per iteration.

    Returns an :class:`OptimizationRun` with the same result fields as
    :func:`optimize_cruise_general`. Worker-side Jacobian samples are not added
    to the objective history.

    Parameters
    ----------
    config, x0, cruise_factory, bounds, fluence_floor_J_per_m2,
    penalty_lambda, baseline_eval, baseline_x, cost_metric,
    multi_element_weights, multi_element_scales,
    cumulative_delta_e_same_u, cumulative_integral_weight,
    cumulative_delta_r_iau_km, cumulative_delta_v_iau_kmps,
    cumulative_integral_weight_r, cumulative_integral_weight_v,
    scale_cum_delta_r_iau_mars_km, scale_cum_delta_v_iau_mars_kmps
        See :func:`optimize_cruise_general`. Cost form, scales, and
        cumulative-state plumbing are identical. ``cost_metric`` may be
        any value in ``_VALID_COST_METRICS``; the polish-only path is
        cost-form agnostic.
    lbfgsb_options
        Forwarded to ``scipy.optimize.minimize(options=...)``. Sensible
        defaults: ``ftol=1e-7``, ``gtol=1e-5``, ``maxiter=50``,
        ``maxfun=200`` (caps total fun calls including jac-internal
        gradient evaluations).
    fd_step_h
        FD step size passed through to
        :func:`reflectors.parallel.parallel_fd_jacobian`. Default
        ``1e-4`` rad, pinned by regression tests.
    jac_workers
        Optional :class:`reflectors.parallel.CloudpickleMap` instance
        for parallel FD jacobian dispatch. ``None`` → serial FD in
        parent (slow but correct; useful for tests / debugging).

    Raises
    ------
    ValueError
        Same input-validation contract as
        :func:`optimize_cruise_general`.
    """
    # Lazy import to avoid an optimize.py ↔ parallel.py dependency cycle
    # at module-load time (parallel.py imports nothing from optimize).
    from reflectors.parallel import parallel_fd_jacobian as _parallel_fd_jacobian

    x0_arr = np.asarray(x0, dtype=float).ravel()
    if x0_arr.ndim != 1 or x0_arr.size == 0:
        raise ValueError(
            f"x0 must be a non-empty 1-D array, got shape {x0_arr.shape}"
        )
    bounds_list = [tuple(b) for b in bounds]
    if len(bounds_list) != x0_arr.size:
        raise ValueError(
            f"bounds length must equal x0 length, got "
            f"len(bounds)={len(bounds_list)}, len(x0)={x0_arr.size}"
        )
    for i, (lo, hi) in enumerate(bounds_list):
        if lo >= hi:
            raise ValueError(
                f"bounds[{i}] must satisfy lo < hi, got {(lo, hi)}"
            )
        if not (lo <= x0_arr[i] <= hi):
            raise ValueError(
                f"x0[{i}]={x0_arr[i]} outside bounds[{i}]={(lo, hi)}"
            )
    if penalty_lambda < 0.0:
        raise ValueError(
            f"penalty_lambda must be >= 0, got {penalty_lambda}"
        )
    if cost_metric not in _VALID_COST_METRICS:
        raise ValueError(
            f"cost_metric must be one of {sorted(_VALID_COST_METRICS)}, "
            f"got {cost_metric!r}"
        )
    _validate_multi_element_overrides(
        multi_element_weights, multi_element_scales, cost_metric,
    )

    t_total_0 = time.perf_counter()

    if baseline_eval is None:
        if baseline_x is None:
            base_x = np.zeros_like(x0_arr)
        else:
            base_x = np.asarray(baseline_x, dtype=float).ravel()
            if base_x.size != x0_arr.size:
                raise ValueError(
                    f"baseline_x size must equal x0 size, got "
                    f"{base_x.size} vs {x0_arr.size}"
                )
        logger.info(
            "optimize_cruise_polish_only: computing baseline at x=%s", base_x
        )
        baseline_eval = evaluate_cruise_general(base_x, config, cruise_factory)
        logger.info(
            "baseline: Δa/sol=%+.4f km, fluence=%.4f J/m^2, e_max=%.5f",
            baseline_eval.delta_a_per_sol_km,
            baseline_eval.total_fluence_J_per_m2,
            baseline_eval.e_max,
        )

    history: list = []

    options = {
        "ftol": 1e-7,
        "gtol": 1e-5,
        "maxiter": 50,
        "maxfun": 200,
    }
    if lbfgsb_options:
        options.update(lbfgsb_options)

    floor = float(fluence_floor_J_per_m2)
    lam = float(penalty_lambda)

    INFEASIBLE_COST = 1.0e9

    def cost_closure(x_inner: np.ndarray) -> float:
        try:
            eval_result = evaluate_cruise_general(
                x_inner, config, cruise_factory,
            )
        except ValueError as exc:
            x_str = ",".join(f"{math.degrees(float(v)):+.2f}" for v in x_inner)
            logger.warning(
                "optimize_cruise_polish_only[%s] infeasible vertex x=[%s] "
                "deg: %s — returning penalty cost %.0e",
                cost_metric, x_str, exc, INFEASIBLE_COST,
            )
            return INFEASIBLE_COST
        cost, breakdown = cruise_cost_with_breakdown(
            eval_result, floor, lam, cost_metric=cost_metric,
            multi_element_weights=multi_element_weights,
            multi_element_scales=multi_element_scales,
            cumulative_delta_e_same_u=cumulative_delta_e_same_u,
            cumulative_integral_weight=cumulative_integral_weight,
            cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
            cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
            cumulative_integral_weight_r=cumulative_integral_weight_r,
            cumulative_integral_weight_v=cumulative_integral_weight_v,
            scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
            scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
            fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
        )
        if breakdown:
            eval_result = replace(eval_result, cost_breakdown=breakdown)
        history.append(eval_result)
        x_str = ",".join(f"{math.degrees(float(v)):+.2f}" for v in x_inner)
        if breakdown:
            logger.info(
                "optimize_cruise_polish_only[%s] eval %d: x=[%s] deg → "
                "cost=%.6f  (%s)  wall=%.2f s",
                cost_metric, len(history), x_str,
                cost,
                _format_breakdown(breakdown),
                eval_result.wall_s,
            )
        else:
            logger.info(
                "optimize_cruise_polish_only[%s] eval %d: x=[%s] deg → "
                "Δa/sol=%+0.4f km, e_max=%.5f, fluence=%.4f J/m^2, "
                "cost=%.6f, wall=%.2f s",
                cost_metric, len(history), x_str,
                eval_result.delta_a_per_sol_km,
                eval_result.e_max,
                eval_result.total_fluence_J_per_m2,
                cost,
                eval_result.wall_s,
            )
        return cost

    # Avoid relying on SciPy's internal fun(x)/jac(x) call order.
    # Recomputing f(x) costs one extra evaluation per gradient call only
    # when an axis is clipped by a bound.
    def jac_callable(x_inner: np.ndarray) -> np.ndarray:
        return _parallel_fd_jacobian(
            cost_closure, x_inner,
            h=fd_step_h,
            workers=jac_workers,
            bounds=bounds_list,
        )

    scipy_result = scipy_minimize(
        fun=cost_closure,
        x0=x0_arr,
        method="L-BFGS-B",
        jac=jac_callable,
        bounds=bounds_list,
        options=options,
    )

    if not history:
        raise RuntimeError(
            "optimize_cruise_polish_only: scipy minimize completed without "
            "any objective evaluations. Internal SciPy issue or malformed "
            "cost."
        )
    best_idx = int(
        np.argmin([
            cruise_cost_from_eval(
                ev, floor, lam, cost_metric=cost_metric,
                multi_element_weights=multi_element_weights,
                multi_element_scales=multi_element_scales,
                cumulative_delta_e_same_u=cumulative_delta_e_same_u,
                cumulative_integral_weight=cumulative_integral_weight,
                cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
                cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
                cumulative_integral_weight_r=cumulative_integral_weight_r,
                cumulative_integral_weight_v=cumulative_integral_weight_v,
                scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
                scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
                fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
            )
            for ev in history
        ])
    )
    best_eval = history[best_idx]

    # Resolve weights/scales for OptimizationRun provenance, mirroring
    # optimize_cruise_general's logic.
    if cost_metric == COST_METRIC_MULTI_ELEMENT_WEIGHTED:
        resolved_weights = dict(DEFAULT_MULTI_ELEMENT_WEIGHTS)
        resolved_scales = dict(DEFAULT_MULTI_ELEMENT_SCALES)
    elif cost_metric == COST_METRIC_CLOSURE_WEIGHTED:
        resolved_weights = dict(DEFAULT_CLOSURE_WEIGHTS)
        resolved_scales = dict(DEFAULT_CLOSURE_SCALES)
    elif cost_metric == COST_METRIC_RV_CLOSURE_WEIGHTED:
        resolved_weights = dict(DEFAULT_RV_CLOSURE_WEIGHTS)
        resolved_scales = dict(DEFAULT_RV_CLOSURE_SCALES)
    else:
        resolved_weights = {}
        resolved_scales = {}
    if (
        cost_metric in {
            COST_METRIC_MULTI_ELEMENT_WEIGHTED,
            COST_METRIC_CLOSURE_WEIGHTED,
            COST_METRIC_RV_CLOSURE_WEIGHTED,
        }
        and multi_element_weights
    ):
        resolved_weights.update(multi_element_weights)
    if (
        cost_metric in {
            COST_METRIC_MULTI_ELEMENT_WEIGHTED,
            COST_METRIC_CLOSURE_WEIGHTED,
            COST_METRIC_RV_CLOSURE_WEIGHTED,
        }
        and multi_element_scales
    ):
        resolved_scales.update(multi_element_scales)

    return OptimizationRun(
        scipy_result=scipy_result,
        history=tuple(history),
        best_eval=best_eval,
        baseline_eval=baseline_eval,
        config=config,
        fluence_floor_J_per_m2=floor,
        penalty_lambda=lam,
        cost_metric=str(cost_metric),
        wall_total_s=time.perf_counter() - t_total_0,
        multi_element_weights=resolved_weights,
        multi_element_scales=resolved_scales,
    )


def optimize_cruise_de_with_parallel_polish(
    config: OrbitConfig,
    *,
    x0: np.ndarray,
    cruise_factory: CruiseFactory,
    bounds: Sequence[Tuple[float, float]],
    fluence_floor_J_per_m2: float,
    penalty_lambda: float = 1.0,
    baseline_eval: Optional[EvaluationResult] = None,
    baseline_x: Optional[np.ndarray] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
    cumulative_delta_e_same_u: float = 0.0,
    cumulative_integral_weight: float = 0.0,
    cumulative_delta_r_iau_km: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_delta_v_iau_kmps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cumulative_integral_weight_r: float = 0.0,
    cumulative_integral_weight_v: float = 0.0,
    scale_cum_delta_r_iau_mars_km: float = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM,
    scale_cum_delta_v_iau_mars_kmps: float = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS,
    fluence_floor_by_target_J_per_m2: Optional[Sequence[float]] = None,
    de_options: Optional[dict] = None,
    fd_step_h: float = 1e-4,
    lbfgsb_options: Optional[dict] = None,
) -> OptimizationRun:
    """Hybrid DE basin-finding plus parallel-FD-Jacobian L-BFGS-B polish.

    Runs scipy DE with its internal serial polish disabled, then applies the
    parallel-FD-jacobian polish from :func:`optimize_cruise_polish_only`.
    Population mutation provides global exploration before local refinement.

    Parallel finite-difference polish reuses the same ``CloudpickleMap`` as the
    differential-evolution phase.

    Internal flow:

      1. Validate inputs (delegated to ``optimize_cruise_general`` and
         ``optimize_cruise_polish_only``); reject ``de_options['polish']
         = True`` with a clear error; the wrapper enforces ``polish=False``.
      2. Force ``polish=False`` into a local ``de_options`` copy.
      3. Call ``optimize_cruise_general(..., algorithm=DE, de_options=
         local_de_options)`` — scipy DE runs without internal polish.
      4. Call ``optimize_cruise_polish_only(..., x0=de_run.best_eval.
         decision_vector_rad, jac_workers=cp_map, baseline_eval=
         de_run.baseline_eval)`` — parallel-FD polish on the DE
         optimum, reusing the workers pool from ``de_options['workers']``
         when it's a :class:`reflectors.parallel.CloudpickleMap`.
      5. Return a single combined ``OptimizationRun``: history is the
         concatenation of DE + polish histories (DE history sparse under
         workers>1 by design), ``best_eval`` is polish's best (polish
         starts from DE's optimum, monotone in cost, so always at-or-
         below DE), ``scipy_result`` is the polish-phase result (the
         final iterate stats), ``wall_total_s`` is the sum.

    Lifecycle: the caller owns the ``CloudpickleMap`` if any (passed in
    ``de_options['workers']``). The wrapper does NOT close the pool —
    that responsibility stays with the caller (for example,
    ``finally: cp_map.close()``). The pool is reused across DE and
    polish phases; ``CloudpickleMap`` rebuilds the underlying
    ``multiprocessing.Pool`` internally when the closure identity
    changes, so the DE and polish cost closures use the same map object but
    distinct pool instances.

    Parameters
    ----------
    config, x0, cruise_factory, bounds, fluence_floor_J_per_m2,
    penalty_lambda, baseline_eval, baseline_x, cost_metric,
    multi_element_weights, multi_element_scales,
    cumulative_*  (Δe / Δr / Δv vectors and PI gains and scales)
        See :func:`optimize_cruise_general`. Cost form, scales, and
        cumulative-state plumbing flow identically through DE phase
        and polish phase, so both operate on the same cost surface.
    de_options
        Forwarded to scipy DE (via ``optimize_cruise_general``). The
        wrapper FORCES ``polish=False`` regardless of caller value;
        ``polish=True`` raises a clear ``ValueError``. Other keys
        flow through untouched (popsize, maxiter, seed, workers,
        init, tol, mutation, recombination, updating).
    fd_step_h
        Pass-through to :func:`optimize_cruise_polish_only`'s
        ``fd_step_h``; default ``1e-4`` rad.
    lbfgsb_options
        Pass-through to :func:`optimize_cruise_polish_only`'s
        ``lbfgsb_options``; default ``None`` → scipy L-BFGS-B
        defaults from ``optimize_cruise_polish_only``.

    Raises
    ------
    ValueError
        If ``de_options['polish']`` is ``True``. All other input
        validation flows through ``optimize_cruise_general`` and
        ``optimize_cruise_polish_only``.

    References
    ----------
    Storn & Price 1997, Differential Evolution; Byrd, Lu, Nocedal, Zhu 1995,
    L-BFGS-B (active-set bound-constrained quasi-Newton).
    """
    # Wrapper enforces polish=False — its whole purpose is to replace
    # scipy's serial L-BFGS-B polish with the parallel-FD-jac polish.
    # Reject polish=True explicitly to prevent accidental double polishing.
    de_options_in = dict(de_options or {})
    if de_options_in.get("polish", False):
        raise ValueError(
            "optimize_cruise_de_with_parallel_polish: "
            "de_options['polish']=True is rejected — this wrapper "
            "REPLACES scipy's serial L-BFGS-B polish with parallel-FD "
            "L-BFGS-B polish via optimize_cruise_polish_only. Drop the "
            "'polish' key from de_options or set polish=False explicitly."
        )
    de_options_in["polish"] = False

    # Differential evolution without SciPy's serial polish.
    de_run = optimize_cruise_general(
        config,
        x0=x0,
        cruise_factory=cruise_factory,
        bounds=bounds,
        fluence_floor_J_per_m2=fluence_floor_J_per_m2,
        penalty_lambda=penalty_lambda,
        baseline_eval=baseline_eval,
        baseline_x=baseline_x,
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
        cumulative_delta_e_same_u=cumulative_delta_e_same_u,
        cumulative_integral_weight=cumulative_integral_weight,
        cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
        cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
        cumulative_integral_weight_r=cumulative_integral_weight_r,
        cumulative_integral_weight_v=cumulative_integral_weight_v,
        scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
        scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
        fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
        algorithm=ALGORITHM_DIFFERENTIAL_EVOLUTION,
        de_options=de_options_in,
    )

    # Reuse the workers pool from de_options for the polish phase. scipy
    # DE accepts ``workers=int`` (uses stdlib Pool internally) OR a
    # caller-supplied map callable like CloudpickleMap. Only the latter
    # is reusable as parallel_fd_jacobian's ``workers``; an int means
    # "serial polish jacobian" (workers=None).
    workers_obj = de_options_in.get("workers", 1)
    if isinstance(workers_obj, int):
        jac_workers = None
    else:
        jac_workers = workers_obj

    # Parallel-FD L-BFGS-B polish from the DE optimum. Read
    # the dv from best_eval.decision_vector_rad (populated by the
    # the main evaluate_cruise_general path). Fall back to scipy_result.x
    # when decision_vector_rad is empty — this catches the workers>1
    # history-recovery path's edge case and keeps synthetic-cost monkeypatch
    # tests representative (helpers may not set
    # decision_vector_rad).
    dv_tuple = de_run.best_eval.decision_vector_rad
    if dv_tuple:
        polish_x0 = np.asarray(dv_tuple, dtype=float)
    else:
        polish_x0 = np.asarray(de_run.scipy_result.x, dtype=float)
    polish_run = optimize_cruise_polish_only(
        config,
        x0=polish_x0,
        cruise_factory=cruise_factory,
        bounds=bounds,
        fluence_floor_J_per_m2=fluence_floor_J_per_m2,
        penalty_lambda=penalty_lambda,
        baseline_eval=de_run.baseline_eval,  # already computed; skip rerun
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
        cumulative_delta_e_same_u=cumulative_delta_e_same_u,
        cumulative_integral_weight=cumulative_integral_weight,
        cumulative_delta_r_iau_km=cumulative_delta_r_iau_km,
        cumulative_delta_v_iau_kmps=cumulative_delta_v_iau_kmps,
        cumulative_integral_weight_r=cumulative_integral_weight_r,
        cumulative_integral_weight_v=cumulative_integral_weight_v,
        scale_cum_delta_r_iau_mars_km=scale_cum_delta_r_iau_mars_km,
        scale_cum_delta_v_iau_mars_kmps=scale_cum_delta_v_iau_mars_kmps,
        fluence_floor_by_target_J_per_m2=fluence_floor_by_target_J_per_m2,
        lbfgsb_options=lbfgsb_options,
        fd_step_h=fd_step_h,
        jac_workers=jac_workers,
    )

    # Combined OptimizationRun. History concatenates DE + polish.
    # best_eval is polish's: polish starts from DE's optimum and is
    # monotone in cost, so polish.best <= DE.best.
    combined_history = de_run.history + polish_run.history
    return OptimizationRun(
        scipy_result=polish_run.scipy_result,
        history=combined_history,
        best_eval=polish_run.best_eval,
        baseline_eval=de_run.baseline_eval,
        config=config,
        fluence_floor_J_per_m2=float(fluence_floor_J_per_m2),
        penalty_lambda=float(penalty_lambda),
        cost_metric=str(cost_metric),
        wall_total_s=de_run.wall_total_s + polish_run.wall_total_s,
        multi_element_weights=polish_run.multi_element_weights,
        multi_element_scales=polish_run.multi_element_scales,
    )


def optimize_cruise(
    config: OrbitConfig,
    *,
    x0: np.ndarray,
    fluence_floor_J_per_m2: float,
    penalty_lambda: float = 1.0,
    beta_bounds: Tuple[float, float] = (0.0, math.radians(30.0)),
    phi_u_bounds: Tuple[float, float] = (0.0, 2.0 * math.pi),
    nelder_mead_options: Optional[dict] = None,
    baseline_eval: Optional[EvaluationResult] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
) -> OptimizationRun:
    """Constant-cruise optimizer over (β, φ_u). Thin wrapper
    around :func:`optimize_cruise_general` with the constant cruise
    factory and 2-D bounds.

    Error messages use the parameter-specific ``beta_bounds`` and
    ``phi_u_bounds`` names. See :func:`optimize_cruise_general` for the full
    parameter contract.
    """
    x0_arr = np.asarray(x0, dtype=float).ravel()
    if x0_arr.shape != (2,):
        raise ValueError(f"x0 must have shape (2,), got {x0_arr.shape}")
    if penalty_lambda < 0.0:
        raise ValueError(
            f"penalty_lambda must be >= 0, got {penalty_lambda}"
        )
    if beta_bounds[0] >= beta_bounds[1]:
        raise ValueError(
            f"beta_bounds must satisfy lo < hi, got {beta_bounds}"
        )
    if phi_u_bounds[0] >= phi_u_bounds[1]:
        raise ValueError(
            f"phi_u_bounds must satisfy lo < hi, got {phi_u_bounds}"
        )
    if not (beta_bounds[0] <= x0_arr[0] <= beta_bounds[1]):
        raise ValueError(
            f"x0[0]={x0_arr[0]} outside beta_bounds={beta_bounds}"
        )
    if not (phi_u_bounds[0] <= x0_arr[1] <= phi_u_bounds[1]):
        raise ValueError(
            f"x0[1]={x0_arr[1]} outside phi_u_bounds={phi_u_bounds}"
        )
    if cost_metric not in _VALID_COST_METRICS:
        raise ValueError(
            f"cost_metric must be one of {sorted(_VALID_COST_METRICS)}, "
            f"got {cost_metric!r}"
        )
    _validate_multi_element_overrides(
        multi_element_weights, multi_element_scales, cost_metric,
    )

    return optimize_cruise_general(
        config,
        x0=x0_arr,
        cruise_factory=_constant_cruise_factory,
        bounds=[tuple(beta_bounds), tuple(phi_u_bounds)],
        fluence_floor_J_per_m2=fluence_floor_J_per_m2,
        penalty_lambda=penalty_lambda,
        nelder_mead_options=nelder_mead_options,
        baseline_eval=baseline_eval,
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
    )


def optimize_cruise_harmonic(
    config: OrbitConfig,
    *,
    x0: np.ndarray,
    fluence_floor_J_per_m2: float,
    penalty_lambda: float = 1.0,
    alpha_0_bounds: Tuple[float, float] = (0.0, math.radians(60.0)),
    alpha_c_bounds: Tuple[float, float] = (
        -math.radians(30.0), math.radians(30.0),
    ),
    alpha_s_bounds: Tuple[float, float] = (
        -math.radians(30.0), math.radians(30.0),
    ),
    phi_u_bounds: Tuple[float, float] = (0.0, 2.0 * math.pi),
    nelder_mead_options: Optional[dict] = None,
    baseline_eval: Optional[EvaluationResult] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
) -> OptimizationRun:
    """Harmonic-in-u cruise optimizer over (α_0, α_c, α_s, φ_u).

    Default bounds form a conservative box that keeps α(u) ∈ [0, π/2]
    over the full u sweep when α_c, α_s are independently maximized
    (α_0 ≤ 60°, |α_c| ≤ 30°, |α_s| ≤ 30°: max α(u) = 60° + 42.4° =
    102° at the box corners, but the cone-angle saturator inside
    ``cruise.sun_offset_harmonic`` raises ``ValueError`` for any
    simplex vertex that violates α ∈ [0, π/2], which Nelder-Mead
    treats as an infeasible vertex).

    Thin wrapper around :func:`optimize_cruise_general` with the
    harmonic cruise factory and 4-D bounds. See
    :func:`optimize_cruise_general` for the full parameter contract.
    """
    return optimize_cruise_general(
        config,
        x0=np.asarray(x0, dtype=float).ravel(),
        cruise_factory=_harmonic_cruise_factory,
        bounds=[
            tuple(alpha_0_bounds),
            tuple(alpha_c_bounds),
            tuple(alpha_s_bounds),
            tuple(phi_u_bounds),
        ],
        fluence_floor_J_per_m2=fluence_floor_J_per_m2,
        penalty_lambda=penalty_lambda,
        nelder_mead_options=nelder_mead_options,
        baseline_eval=baseline_eval,
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
    )


def optimize_cruise_harmonic_full(
    config: OrbitConfig,
    *,
    x0: np.ndarray,
    fluence_floor_J_per_m2: float,
    penalty_lambda: float = 1.0,
    alpha_0_bounds: Tuple[float, float] = (0.0, math.radians(60.0)),
    alpha_c_bounds: Tuple[float, float] = (
        -math.radians(30.0), math.radians(30.0),
    ),
    alpha_s_bounds: Tuple[float, float] = (
        -math.radians(30.0), math.radians(30.0),
    ),
    delta_0_bounds: Tuple[float, float] = (0.0, 2.0 * math.pi),
    delta_c_bounds: Tuple[float, float] = (
        -math.radians(60.0), math.radians(60.0),
    ),
    delta_s_bounds: Tuple[float, float] = (
        -math.radians(60.0), math.radians(60.0),
    ),
    nelder_mead_options: Optional[dict] = None,
    baseline_eval: Optional[EvaluationResult] = None,
    cost_metric: str = COST_METRIC_DELTA_A,
    multi_element_weights: Optional[Mapping[str, float]] = None,
    multi_element_scales: Optional[Mapping[str, float]] = None,
) -> OptimizationRun:
    """Harmonic-(α, δ) cruise optimizer over (α_0, α_c, α_s,
    δ_0, δ_c, δ_s).

    Decision vector ``x = [α_0, α_c, α_s, δ_0, δ_c, δ_s]`` (radians).
    Setting ``x[4] = x[5] = 0`` reduces bit-for-bit through the
    pipeline to ``optimize_cruise_harmonic``; setting
    ``x[1] = x[2] = x[4] = x[5] = 0`` reduces further to
    ``optimize_cruise``. Pinned by the slow regression test
    ``test_harmonic_full_zero_delta_amplitudes_matches_harmonic_alpha_through_pipeline``.

    Default α bounds form a conservative box that is wider than the
    Cauchy-Schwarz-tight α-cone constraint (α_0 ≤ 60°, |α_c|
    ≤ 30°, |α_s| ≤ 30°: max α(u) = 60° + 42.4° = 102° at the corner).
    Default δ bounds are ±60° on δ_c, δ_s, wider than the soft
    Cauchy-Schwarz amplitude bound (default π/2 = 90°) inside
    ``cruise.sun_offset_harmonic_full``. The cone-angle saturator and
    the δ-amplitude saturator both raise ``ValueError`` for vertices
    that violate the constraints, which the cost closure
    (optimize.py:625-632) converts to ``INFEASIBLE_COST = 1e9`` so
    Nelder-Mead naturally rejects them.

    Thin wrapper around :func:`optimize_cruise_general` with the
    harmonic-(α,δ) cruise factory and 6-D bounds. See
    :func:`optimize_cruise_general` for the full parameter contract.
    """
    return optimize_cruise_general(
        config,
        x0=np.asarray(x0, dtype=float).ravel(),
        cruise_factory=_harmonic_full_cruise_factory,
        bounds=[
            tuple(alpha_0_bounds),
            tuple(alpha_c_bounds),
            tuple(alpha_s_bounds),
            tuple(delta_0_bounds),
            tuple(delta_c_bounds),
            tuple(delta_s_bounds),
        ],
        fluence_floor_J_per_m2=fluence_floor_J_per_m2,
        penalty_lambda=penalty_lambda,
        nelder_mead_options=nelder_mead_options,
        baseline_eval=baseline_eval,
        cost_metric=cost_metric,
        multi_element_weights=multi_element_weights,
        multi_element_scales=multi_element_scales,
    )
