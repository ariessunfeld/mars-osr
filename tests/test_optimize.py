"""Cost, configuration, optimizer, and propagation tests for
``reflectors.optimize``.
"""

from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from types import SimpleNamespace

import numpy as np
import pytest

from reflectors.attitude_schedule import ScheduleMetadata
from reflectors.dynamics import PropagationOptions
from reflectors.visibility import DeliveryWindow
from reflectors.ephemeris import utc_to_et
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import initial_state_j2000
from reflectors.optimize import (
    ALGORITHM_DIFFERENTIAL_EVOLUTION,
    ALGORITHM_NELDER_MEAD,
    COST_METRIC_CLOSURE_WEIGHTED,
    COST_METRIC_DELTA_A,
    COST_METRIC_E_MAX,
    COST_METRIC_MULTI_ELEMENT_WEIGHTED,
    COST_METRIC_RV_CLOSURE_WEIGHTED,
    CruiseFactory,
    DEFAULT_CLOSURE_SCALES,
    DEFAULT_CLOSURE_WEIGHTS,
    DEFAULT_MULTI_ELEMENT_SCALES,
    DEFAULT_MULTI_ELEMENT_WEIGHTS,
    DEFAULT_RV_CLOSURE_SCALES,
    DEFAULT_RV_CLOSURE_WEIGHTS,
    DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM,
    DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS,
    EvaluationResult,
    OrbitConfig,
    SUN_SYNC_RAAN_RATE_RAD_PER_S,
    _constant_cruise_factory,
    _harmonic_cruise_factory,
    _kept_window_fluence,
    _harmonic_full_cruise_factory,
    _harmonic_full_mode2_cruise_factory,
    cruise_cost_from_eval,
    cruise_cost_with_breakdown,
    evaluate_cruise,
    evaluate_cruise_general,
    make_handoff_cruise_factory,
    optimize_cruise,
    optimize_cruise_general,
    optimize_cruise_harmonic,
    optimize_cruise_harmonic_full,
    optimize_cruise_de_with_parallel_polish,
    optimize_cruise_polish_only,
    warm_start_population_for_de,
)
from reflectors.termination import AltitudeFloor
from reflectors.third_body import (
    deimos_third_body, phobos_third_body, sun_third_body,
)


# ---------------------------------------------------------------------------
# Canonical rank-1 configuration shared across slow tests and workflows
# ---------------------------------------------------------------------------

EPOCH_UTC = "2028-02-11T12:42:00"  # Mars perihelion 2028
A_KM = 501.0 + 3396.19
LTAN_H = 18.0
M0_DEG = 0.0
DURATION_S = 88775.0  # one sidereal sol
TARGET_LAT_DEG = 40.0
TARGET_LON_DEG = 200.0
ALPHA_MAX_RAD_S2 = math.radians(0.003)


def _reference_config(duration_s: float = DURATION_S) -> OrbitConfig:
    """Build the canonical ``OrbitConfig`` used by tests and workflows.

    Pinned at sigma = 0.05 kg/m^2 to preserve the reference inline
    construction that the slow regressions assert against.
    """
    epoch_et = utc_to_et(EPOCH_UTC)
    sail = make_canonical_sail(0.05)
    propagate_kwargs = dict(
        gravity_degree=6,
        gravity_order=6,
        third_bodies=[
            sun_third_body(), phobos_third_body(), deimos_third_body(),
        ],
        altitude_floor=AltitudeFloor.at_km(300.0, label="altitude_floor"),
        options=PropagationOptions.fast(),
        t_eval_s=np.arange(0.0, duration_s + 0.1, 5.0),
    )
    find_windows_kwargs = dict(alpha_max_rad_s2=ALPHA_MAX_RAD_S2)
    return OrbitConfig(
        a_km=A_KM,
        ltan_h=LTAN_H,
        M0_rad=math.radians(M0_DEG),
        epoch_et=epoch_et,
        duration_s=duration_s,
        sail=sail,
        target_lat_deg=TARGET_LAT_DEG,
        target_lon_deg=TARGET_LON_DEG,
        alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
        slew_duration_s=300.0,
        propagate_kwargs=propagate_kwargs,
        find_windows_kwargs=find_windows_kwargs,
    )


def _synth_eval(
    *,
    delta_a_per_sol_km: float,
    fluence: float,
    e_max: float = 0.0,
    beta_rad: float = 0.0,
    phi_u_rad: float = 0.0,
    delta_i_deg: float = 0.0,
    delta_raan_deg: float = 0.0,
    raan_sun_sync_target_deg: float = 0.0,
    e_end_osculating: float = 0.0,
    delta_e_same_u: float = 0.0,
    delta_argp_deg: float = 0.0,
    argp_J2_target_deg: float = 0.0,
    delta_r_iau_mars_km: tuple = (0.0, 0.0, 0.0),
    delta_v_iau_mars_kmps: tuple = (0.0, 0.0, 0.0),
    fluence_by_target: tuple = (),
    n_windows_by_target: tuple = (),
) -> EvaluationResult:
    """Build a synthetic EvaluationResult for cost-arithmetic tests."""
    return EvaluationResult(
        beta_rad=beta_rad,
        phi_u_rad=phi_u_rad,
        total_fluence_J_per_m2=float(fluence),
        delta_a_km=float(delta_a_per_sol_km),
        delta_a_per_sol_km=float(delta_a_per_sol_km),
        e_max=float(e_max),
        inc_range_deg=0.0,
        n_windows_kept=1,
        n_windows_dropped=0,
        n_unstable_windows=0,
        converged=True,
        n_iterations=1,
        wall_s=0.0,
        delta_i_deg=float(delta_i_deg),
        delta_raan_deg=float(delta_raan_deg),
        raan_sun_sync_target_deg=float(raan_sun_sync_target_deg),
        e_end_osculating=float(e_end_osculating),
        delta_e_same_u=float(delta_e_same_u),
        delta_argp_deg=float(delta_argp_deg),
        argp_J2_target_deg=float(argp_J2_target_deg),
        delta_r_iau_mars_km=tuple(float(v) for v in delta_r_iau_mars_km),
        delta_v_iau_mars_kmps=tuple(float(v) for v in delta_v_iau_mars_kmps),
        fluence_by_target_J_per_m2=tuple(
            float(v) for v in fluence_by_target
        ),
        n_windows_by_target=tuple(int(v) for v in n_windows_by_target),
    )


# ---------------------------------------------------------------------------
# Group 1: Config construction (fast)
# ---------------------------------------------------------------------------


class TestOrbitConfigConstruction:
    def test_initial_state_matches_direct_call(self):
        """OrbitConfig.initial_state_km_kmps == initial_state_j2000(...)."""
        cfg = _reference_config()
        direct = initial_state_j2000(
            a_km=A_KM, ltan_h=LTAN_H,
            M0_rad=math.radians(M0_DEG), epoch_et=cfg.epoch_et,
        )
        np.testing.assert_array_equal(cfg.initial_state_km_kmps, direct)

    def test_mu_populated_from_default_anchors(self):
        cfg = _reference_config()
        assert cfg.mu_km3_s2 > 42000.0  # mu_Mars ~ 42828
        assert cfg.mu_km3_s2 < 43000.0

    def test_rejects_nonpositive_duration(self):
        with pytest.raises(ValueError, match="duration_s"):
            OrbitConfig(
                a_km=A_KM, ltan_h=LTAN_H, M0_rad=0.0,
                epoch_et=utc_to_et(EPOCH_UTC),
                duration_s=0.0,
                sail=make_canonical_sail(0.05),
                target_lat_deg=40.0, target_lon_deg=200.0,
                alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
            )

    def test_rejects_nonpositive_alpha_max(self):
        with pytest.raises(ValueError, match="alpha_max_rad_s2"):
            OrbitConfig(
                a_km=A_KM, ltan_h=LTAN_H, M0_rad=0.0,
                epoch_et=utc_to_et(EPOCH_UTC),
                duration_s=DURATION_S,
                sail=make_canonical_sail(0.05),
                target_lat_deg=40.0, target_lon_deg=200.0,
                alpha_max_rad_s2=0.0,
            )


# ---------------------------------------------------------------------------
# Group 1b: EvaluationResult element-drift fields (fast)
# ---------------------------------------------------------------------------


class TestEvaluationResultElementDriftFields:
    """Element-drift fields and their defaults.

    These are fast tests — no propagation. They pin the dataclass shape.
    """

    def test_constructs_with_default_signature(self):
        """Callers may omit the element-drift fields; defaults must apply."""
        ev = EvaluationResult(
            beta_rad=0.0,
            phi_u_rad=0.0,
            total_fluence_J_per_m2=10.0,
            delta_a_km=-0.5,
            delta_a_per_sol_km=-0.5,
            e_max=0.003,
            inc_range_deg=0.1,
            n_windows_kept=2,
            n_windows_dropped=0,
            n_unstable_windows=0,
            converged=True,
            n_iterations=1,
            wall_s=2.5,
        )
        assert ev.delta_i_deg == 0.0
        assert ev.delta_raan_deg == 0.0
        assert ev.raan_sun_sync_target_deg == 0.0
        assert ev.e_end_osculating == 0.0
        assert ev.cost_breakdown == ()

    def test_accepts_all_new_fields(self):
        ev = EvaluationResult(
            beta_rad=0.0, phi_u_rad=0.0,
            total_fluence_J_per_m2=10.0,
            delta_a_km=-0.5, delta_a_per_sol_km=-0.5,
            e_max=0.003, inc_range_deg=0.1,
            n_windows_kept=2, n_windows_dropped=0, n_unstable_windows=0,
            converged=True, n_iterations=1, wall_s=2.5,
            delta_i_deg=0.10,
            delta_raan_deg=0.50,
            raan_sun_sync_target_deg=0.499,
            e_end_osculating=0.0028,
            cost_breakdown=(("delta_a", 1.0), ("e_max", 0.36)),
        )
        assert ev.delta_i_deg == 0.10
        assert ev.delta_raan_deg == 0.50
        assert ev.raan_sun_sync_target_deg == pytest.approx(0.499)
        assert ev.e_end_osculating == 0.0028
        assert ev.cost_breakdown == (("delta_a", 1.0), ("e_max", 0.36))

    def test_cost_breakdown_is_tuple_of_tuples(self):
        """Tuple[Tuple[str, float], ...] keeps the dataclass hashable
        (frozen=True) and round-trips through dataclasses.asdict for the
        history CSV writer."""
        from dataclasses import asdict

        ev = EvaluationResult(
            beta_rad=0.0, phi_u_rad=0.0,
            total_fluence_J_per_m2=10.0,
            delta_a_km=0.0, delta_a_per_sol_km=0.0,
            e_max=0.0, inc_range_deg=0.0,
            n_windows_kept=0, n_windows_dropped=0, n_unstable_windows=0,
            converged=True, n_iterations=0, wall_s=0.0,
            cost_breakdown=(("delta_a", 1.5), ("e_max", 0.3)),
        )
        d = asdict(ev)
        # asdict converts inner tuples to inner tuples (lists if there
        # were any nested dataclasses, but these values are plain tuples).
        assert d["cost_breakdown"] == (("delta_a", 1.5), ("e_max", 0.3))

    def test_sun_sync_raan_rate_constant_value(self):
        """Pin the secular sun-sync drift rate constant against the
        Brouwer first-order target Ω̇ = 2π / T_year. T_year_Mars ≈
        59.35e6 s → rate ≈ 1.058e-7 rad/s ≈ 0.524°/sol over a sidereal
        sol = 88775 s.
        """
        # Sanity: rate is in the expected order of magnitude.
        assert SUN_SYNC_RAAN_RATE_RAD_PER_S == pytest.approx(
            1.058e-7, rel=2e-3
        )
        # Per-sol drift in degrees; the baseline is approximately 0.5 deg.
        per_sol_deg = math.degrees(
            SUN_SYNC_RAAN_RATE_RAD_PER_S * 88775.0
        )
        assert per_sol_deg == pytest.approx(0.539, abs=5e-3)


# ---------------------------------------------------------------------------
# Group 2: Cost arithmetic (fast)
# ---------------------------------------------------------------------------


class TestCruiseCostArithmetic:
    def test_unconstrained_equals_abs_delta_a(self):
        ev = _synth_eval(delta_a_per_sol_km=-0.42, fluence=30.0)
        cost = cruise_cost_from_eval(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=10.0,
        )
        assert cost == pytest.approx(0.42, rel=0.0, abs=1e-15)

    def test_constraint_violation_adds_quadratic_penalty(self):
        ev = _synth_eval(delta_a_per_sol_km=0.10, fluence=20.0)
        # Shortfall = 25 - 20 = 5; penalty = 2.0 * 5^2 = 50; cost = 0.10 + 50.
        cost = cruise_cost_from_eval(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=2.0,
        )
        assert cost == pytest.approx(50.10, rel=0.0, abs=1e-12)

    def test_zero_fluence_pegs_full_penalty(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=0.0)
        # Shortfall = 25 - 0 = 25; penalty = 1.0 * 625; cost = 0 + 625.
        cost = cruise_cost_from_eval(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
        )
        assert cost == pytest.approx(625.0, rel=0.0, abs=1e-12)

    def test_negative_lambda_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="penalty_lambda"):
            cruise_cost_from_eval(
                ev, fluence_floor_J_per_m2=25.0, penalty_lambda=-0.1,
            )

    def test_nonfinite_floor_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="fluence_floor"):
            cruise_cost_from_eval(
                ev, fluence_floor_J_per_m2=float("nan"), penalty_lambda=1.0,
            )

    def test_e_max_metric_returns_e_max(self):
        ev = _synth_eval(delta_a_per_sol_km=10.0, fluence=30.0, e_max=0.0042)
        cost = cruise_cost_from_eval(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_E_MAX,
        )
        # No penalty (fluence above floor); cost is just e_max.
        # Δa is ignored under e_max metric.
        assert cost == pytest.approx(0.0042, rel=0.0, abs=1e-15)

    def test_e_max_metric_adds_fluence_penalty(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=20.0, e_max=0.005)
        # Shortfall = 25 - 20 = 5; penalty = 1.0 * 25; cost = 0.005 + 25.
        cost = cruise_cost_from_eval(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_E_MAX,
        )
        assert cost == pytest.approx(25.005, rel=0.0, abs=1e-12)

    def test_invalid_cost_metric_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="cost_metric"):
            cruise_cost_from_eval(
                ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
                cost_metric="not_a_metric",
            )

    def test_single_element_metrics_return_empty_breakdown(self):
        """Single-element cost metrics leave ``cost_breakdown`` empty."""
        ev = _synth_eval(delta_a_per_sol_km=-0.5, fluence=30.0, e_max=0.005)
        _, b1 = cruise_cost_with_breakdown(
            ev, 25.0, 1.0, cost_metric=COST_METRIC_DELTA_A,
        )
        _, b2 = cruise_cost_with_breakdown(
            ev, 25.0, 1.0, cost_metric=COST_METRIC_E_MAX,
        )
        assert b1 == ()
        assert b2 == ()


# ---------------------------------------------------------------------------
# Group 2b: Multi-element weighted cost arithmetic (fast)
# ---------------------------------------------------------------------------


class TestMultiElementCostArithmetic:
    """Pin the weighted-sum arithmetic without invoking propagation.

    Defaults pinned in optimize.py:
      scale_a = 0.5 km/sol      scale_e = 0.005
      scale_i = 0.1 deg          scale_Ω = 0.01 deg
      all weights = 1.0
    """

    def test_zero_drift_zero_cost(self):
        """All four element drifts zero AND fluence above floor → cost = 0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0, e_max=0.0,
            delta_i_deg=0.0, delta_raan_deg=0.0,
            raan_sun_sync_target_deg=0.0,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        assert cost == pytest.approx(0.0, abs=1e-15)
        # Breakdown must still be populated even when all values are zero.
        assert len(breakdown) == 5
        for k, v in breakdown:
            assert v == pytest.approx(0.0, abs=1e-15), f"{k}={v}"

    def test_pure_delta_a_drift_at_natural_scale(self):
        """Delta-a = -0.58 km/sol at baseline drift, all other terms zero.
        cost = (0.58/0.5)² = 1.3456.
        """
        ev = _synth_eval(
            delta_a_per_sol_km=-0.58, fluence=30.0,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        expected_delta_a_term = (0.58 / 0.5) ** 2
        assert cost == pytest.approx(expected_delta_a_term, abs=1e-12)
        # Breakdown elements: only delta_a is non-zero.
        d = dict(breakdown)
        assert d["delta_a"] == pytest.approx(expected_delta_a_term, abs=1e-12)
        assert d["e_max"] == 0.0
        assert d["delta_i"] == 0.0
        assert d["delta_raan_vs_sunsync"] == 0.0
        assert d["fluence_penalty"] == 0.0

    def test_breakdown_sums_to_residual(self):
        """Σ(non-penalty breakdown values) + penalty == cost.
        Identity: weighted-sum cost is exactly the sum of its parts.
        """
        ev = _synth_eval(
            delta_a_per_sol_km=-0.5, fluence=20.0, e_max=0.01,
            delta_i_deg=0.05,
            delta_raan_deg=0.45,
            raan_sun_sync_target_deg=0.539,  # natural target
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=2.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        total_from_breakdown = sum(v for _, v in breakdown)
        assert cost == pytest.approx(total_from_breakdown, abs=1e-12)

    def test_raan_target_subtraction(self):
        """When ΔRAAN observed equals the sun-sync target, the
        Δ(RAAN−sunsync) term contributes ZERO regardless of magnitude.
        Exercises the (delta_raan_deg − raan_sun_sync_target_deg)
        subtraction.
        """
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0, e_max=0.0,
            delta_i_deg=0.0,
            delta_raan_deg=0.539,
            raan_sun_sync_target_deg=0.539,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_raan_vs_sunsync"] == pytest.approx(0.0, abs=1e-15)
        assert cost == pytest.approx(0.0, abs=1e-15)

    def test_raan_miss_off_target_contributes(self):
        """ΔRAAN miss = 0.01° (one scale unit) → term = 1.0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_raan_deg=0.549,           # = target + 0.01
            raan_sun_sync_target_deg=0.539,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_raan_vs_sunsync"] == pytest.approx(1.0, abs=1e-12)
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_fluence_shortfall_adds_quadratic_penalty(self):
        """Penalty arithmetic identical to single-element costs."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=20.0,  # shortfall = 5
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=2.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["fluence_penalty"] == pytest.approx(50.0, abs=1e-12)  # = 2 * 5²
        assert cost == pytest.approx(50.0, abs=1e-12)

    def test_weighted_sum_NOT_chebyshev(self):
        """Verify weighted-sum (a+b+c), not Chebyshev
        (max). Two equal-magnitude term residuals → cost = 2x, not 1x."""
        # delta_a / scale_a = 1, delta_i / scale_i = 1; others zero.
        ev = _synth_eval(
            delta_a_per_sol_km=0.5,
            fluence=30.0,
            delta_i_deg=0.1,
        )
        cost, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        )
        # Sum: 1 + 1 = 2. Chebyshev would give 1.
        assert cost == pytest.approx(2.0, abs=1e-12)

    def test_custom_weights_override_defaults(self):
        """Per-element weight override scales the corresponding term."""
        ev = _synth_eval(delta_a_per_sol_km=0.5, fluence=30.0)
        cost, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
            multi_element_weights={"delta_a": 5.0},
        )
        # delta_a term: 5 * (0.5/0.5)² = 5.
        assert cost == pytest.approx(5.0, abs=1e-12)

    def test_custom_scales_override_defaults(self):
        """Per-element scale override changes the normalisation."""
        ev = _synth_eval(delta_a_per_sol_km=1.0, fluence=30.0)
        cost, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
            multi_element_scales={"delta_a": 1.0},
        )
        # delta_a term with new scale: 1 * (1.0/1.0)² = 1.
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_default_scales_pin_natural_drift_orders(self):
        """Each element contributes O(1) at its baseline drift scale."""
        # Pinned defaults (must match optimize.py).
        assert DEFAULT_MULTI_ELEMENT_SCALES["delta_a"] == 0.5
        assert DEFAULT_MULTI_ELEMENT_SCALES["e_max"] == 0.005
        assert DEFAULT_MULTI_ELEMENT_SCALES["delta_i"] == 0.1
        assert DEFAULT_MULTI_ELEMENT_SCALES["delta_raan_vs_sunsync"] == 0.01
        # Default weights all 1.0.
        for name in DEFAULT_MULTI_ELEMENT_WEIGHTS:
            assert DEFAULT_MULTI_ELEMENT_WEIGHTS[name] == 1.0


# ---------------------------------------------------------------------------
# Group 2c: Multi-element cost input validation (fast)
# ---------------------------------------------------------------------------


class TestMultiElementCostInputValidation:
    def test_negative_weight_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="weights"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
                multi_element_weights={"delta_a": -1.0},
            )

    def test_nonpositive_scale_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="scales"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
                multi_element_scales={"delta_a": 0.0},
            )

    def test_unknown_element_in_weights_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="not a recognised element"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
                multi_element_weights={"bogus_element": 1.0},
            )

    def test_unknown_element_in_scales_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="not a recognised element"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
                multi_element_scales={"bogus_element": 0.5},
            )

    def test_weights_for_single_element_metric_raises(self):
        """Weights are only meaningful with multi-element cost."""
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(
            ValueError, match="multi_element_weights only valid"
        ):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_DELTA_A,
                multi_element_weights={"delta_a": 1.0},
            )

    def test_scales_for_single_element_metric_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(
            ValueError, match="multi_element_scales only valid"
        ):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_E_MAX,
                multi_element_scales={"e_max": 0.001},
            )


# ---------------------------------------------------------------------------
# Group 2d: Closure-cost arithmetic (fast)
# ---------------------------------------------------------------------------


class TestClosureCostArithmetic:
    """Pin closure_weighted cost arithmetic without propagation.

    Five terms: {delta_a, delta_e_same_u, delta_i,
    delta_raan_vs_sunsync, delta_argp_vs_J2}. Same weighted-sum form as the
    multi-element cost; the e_max term is replaced by
    ``delta_e_same_u`` (unfightable J_2 short-period floor cancels via
    same-u sampling) and ``delta_argp_vs_J2`` is added (signed Δω miss
    vs Brouwer secular target).

    Defaults pinned in optimize.py at K=12, sigma=0.018:
      scale_a = 0.5 km/sol            scale_Δe_u = 0.008
      scale_i = 0.1 deg                scale_Ω-ss = 0.01 deg
      scale_Δω-J2 = 18.0 deg           all weights = 1.0
    """

    def test_zero_drift_zero_cost(self):
        """All five element drifts zero AND fluence above floor → cost=0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_i_deg=0.0, delta_raan_deg=0.0,
            raan_sun_sync_target_deg=0.0,
            delta_e_same_u=0.0,
            delta_argp_deg=0.0, argp_J2_target_deg=0.0,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        assert cost == pytest.approx(0.0, abs=1e-15)
        # 5 element terms + 1 fluence_penalty term = 6 entries.
        assert len(breakdown) == 7
        for k, v in breakdown:
            assert v == pytest.approx(0.0, abs=1e-15), f"{k}={v}"

    def test_pure_delta_e_same_u_at_natural_scale(self):
        """Δe_same_u = 0.008 (one scale unit), all other zero ⇒ cost=1."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.008,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_e_same_u"] == pytest.approx(1.0, abs=1e-12)
        assert d["delta_a"] == 0.0
        assert d["delta_i"] == 0.0
        assert d["delta_raan_vs_sunsync"] == 0.0
        assert d["delta_argp_vs_J2"] == 0.0
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_argp_target_subtraction_zeros_when_observed_equals_target(self):
        """When Δargp observed equals the J_2 secular target, the
        Δargp-J2 term contributes ZERO regardless of magnitude. Pins
        the ``delta_argp_deg − argp_J2_target_deg`` subtraction."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_argp_deg=-4.73,
            argp_J2_target_deg=-4.73,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_argp_vs_J2"] == pytest.approx(0.0, abs=1e-15)
        assert cost == pytest.approx(0.0, abs=1e-15)

    def test_argp_miss_off_target_contributes(self):
        """Δargp miss = 18° (one scale unit) → term = 1.0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_argp_deg=13.27,           # = target + 18.0
            argp_J2_target_deg=-4.73,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_argp_vs_J2"] == pytest.approx(1.0, abs=1e-12)
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_breakdown_sums_to_residual(self):
        """Σ(non-penalty breakdown) + penalty == cost. Identity for
        weighted-sum costs.
        """
        ev = _synth_eval(
            delta_a_per_sol_km=-0.5, fluence=20.0,
            delta_i_deg=0.05,
            delta_raan_deg=0.45,
            raan_sun_sync_target_deg=0.539,
            delta_e_same_u=0.0008,
            delta_argp_deg=-4.5,
            argp_J2_target_deg=-4.73,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=2.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        total_from_breakdown = sum(v for _, v in breakdown)
        assert cost == pytest.approx(total_from_breakdown, abs=1e-12)

    def test_default_scales_pinned(self):
        """Pinned defaults (must match optimize.py)."""
        assert DEFAULT_CLOSURE_SCALES["delta_a"] == 0.5
        assert DEFAULT_CLOSURE_SCALES["delta_e_same_u"] == 0.008
        assert DEFAULT_CLOSURE_SCALES["delta_i"] == 0.1
        assert DEFAULT_CLOSURE_SCALES["delta_raan_vs_sunsync"] == 0.01
        assert DEFAULT_CLOSURE_SCALES["delta_argp_vs_J2"] == 18.0
        for name in DEFAULT_CLOSURE_WEIGHTS:
            assert DEFAULT_CLOSURE_WEIGHTS[name] == 1.0

    def test_custom_scales_for_closure_terms(self):
        """Per-element scale override changes the normalisation; works
        for the new closure-only term names too."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.001,
        )
        cost, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            multi_element_scales={"delta_e_same_u": 0.001},
        )
        # Term: 1 * (0.001 / 0.001)² = 1.
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_tightened_raan_scale_propagates_through_closure_branch(self):
        """Tightening ``delta_raan_vs_sunsync``
        scale via ``multi_element_scales`` must propagate through the
        closure-cost branch and change the term value.

        At |Delta-Omega-ss| ≈ 0.00428 deg/sol:
          scale=0.01  -> term = (0.00428/0.01)^2  = 0.183
          scale=0.003 -> term = (0.00428/0.003)^2 = 2.036
        """
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_raan_deg=0.00428,
            raan_sun_sync_target_deg=0.0,
        )
        # Default scale 0.01: term approximately 0.183.
        cost_default, breakdown_default = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        d_default = dict(breakdown_default)
        assert d_default["delta_raan_vs_sunsync"] == pytest.approx(
            (0.00428 / 0.01) ** 2, abs=1e-12,
        )
        # Tightened scale 0.003: term approximately 2.036.
        cost_tight, breakdown_tight = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            multi_element_scales={"delta_raan_vs_sunsync": 0.003},
        )
        d_tight = dict(breakdown_tight)
        assert d_tight["delta_raan_vs_sunsync"] == pytest.approx(
            (0.00428 / 0.003) ** 2, abs=1e-12,
        )
        # Tightening must increase the term.
        assert d_tight["delta_raan_vs_sunsync"] > d_default["delta_raan_vs_sunsync"]
        # Other closure terms unchanged.
        for term_name in ("delta_a", "delta_e_same_u", "delta_i", "delta_argp_vs_J2"):
            assert d_tight[term_name] == d_default[term_name], (
                f"{term_name} changed unexpectedly under RAAN scale override"
            )

    def test_e_max_not_in_closure_vocabulary(self):
        """``e_max`` is rejected because closure cost uses delta_e_same_u."""
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="not a recognised element"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
                multi_element_scales={"e_max": 0.005},
            )

    def test_delta_e_same_u_not_in_multi_element_vocabulary(self):
        """Symmetric: ``delta_e_same_u`` is rejected under multi_element."""
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="not a recognised element"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
                multi_element_scales={"delta_e_same_u": 0.0005},
            )


class TestClosureWeightedCumulativeDeltaE:
    """Pin PI cumulative Δe integral-feedback term in closure_weighted.

    Form: K_i · ((cum_prior + Δe_current) / scale_de)²
    where cum_prior is the running sum of prior sols' delta_e_same_u,
    Δe_current is the current evaluation's delta_e_same_u, scale_de
    is the same DEFAULT_SCALE_DELTA_E_SAME_U = 0.008, and K_i is
    cumulative_integral_weight.
    """

    def test_zero_weight_no_effect(self):
        """K_i=0 with nonzero cum → cost identical to no-cum call."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.004,
        )
        cost_base, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        cost_with_cum, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=0.05,
            cumulative_integral_weight=0.0,
        )
        assert cost_with_cum == cost_base

    def test_positive_cum_biases_toward_negative_de(self):
        """K_i=1.0, cum=+0.008 → cost at Δe=-0.004 lower than at Δe=0."""
        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.0,
        )
        ev_negative = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=-0.004,
        )
        cost_zero, _ = cruise_cost_with_breakdown(
            ev_zero, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=0.008,
            cumulative_integral_weight=1.0,
        )
        cost_neg, _ = cruise_cost_with_breakdown(
            ev_negative, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=0.008,
            cumulative_integral_weight=1.0,
        )
        assert cost_neg < cost_zero

    def test_negative_cum_biases_toward_positive_de(self):
        """Symmetric: K_i=1.0, cum=-0.008 → cost at Δe=+0.004 lower."""
        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.0,
        )
        ev_positive = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=+0.004,
        )
        cost_zero, _ = cruise_cost_with_breakdown(
            ev_zero, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=-0.008,
            cumulative_integral_weight=1.0,
        )
        cost_pos, _ = cruise_cost_with_breakdown(
            ev_positive, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=-0.008,
            cumulative_integral_weight=1.0,
        )
        assert cost_pos < cost_zero

    def test_cum_zero_adds_to_per_sol_weight(self):
        """K_i=1.0, cum=0, Δe=0.008 → cum term = 1.0 (same as per-sol).
        Total Δe-related cost = per-sol (1.0) + cum (1.0) = 2.0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.008,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=0.0,
            cumulative_integral_weight=1.0,
        )
        d = dict(breakdown)
        assert d["delta_e_same_u"] == pytest.approx(1.0, abs=1e-12)
        assert d["cum_delta_e"] == pytest.approx(1.0, abs=1e-12)
        assert cost == pytest.approx(2.0, abs=1e-12)

    def test_breakdown_has_cum_entry(self):
        """Breakdown always has 7 entries under closure_weighted."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.002,
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=0.01,
            cumulative_integral_weight=2.0,
        )
        assert len(breakdown) == 7
        d = dict(breakdown)
        assert "cum_delta_e" in d
        # cum_total = 0.01 + 0.002 = 0.012; term = 2.0 * (0.012/0.008)² = 4.5
        expected = 2.0 * (0.012 / 0.008) ** 2
        assert d["cum_delta_e"] == pytest.approx(expected, abs=1e-12)

    def test_analytical_minimum(self):
        """K_i=1, w_e=1, cum=+0.016. Optimal Δe = -cum/(1+1) = -0.008.
        Cost at Δe=-0.008 should be less than at Δe=0."""
        cum = 0.016
        de_opt = -cum / 2.0  # = -0.008
        ev_opt = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=de_opt,
        )
        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_e_same_u=0.0,
        )
        kwargs = dict(
            fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_e_same_u=cum,
            cumulative_integral_weight=1.0,
        )
        cost_opt, _ = cruise_cost_with_breakdown(ev_opt, **kwargs)
        cost_zero, _ = cruise_cost_with_breakdown(ev_zero, **kwargs)
        assert cost_opt < cost_zero
        # At the Δe optimum (-0.008):
        #   per-sol term: 1.0 * (-0.008 / 0.008)² = 1.0
        #   cum term: 1.0 * ((0.016 - 0.008) / 0.008)² = 1.0
        #   total Δe-related = 2.0
        # At Δe=0:
        #   per-sol term: 0.0
        #   cum term: 1.0 * (0.016 / 0.008)² = 4.0
        #   total Δe-related = 4.0
        assert cost_opt == pytest.approx(2.0, abs=1e-12)
        assert cost_zero == pytest.approx(4.0, abs=1e-12)


class TestRVClosureCostArithmetic:
    """Pin rv_closure_weighted cost arithmetic without propagation.

    Three-term cost over (||Δr_iau||, ||Δv_iau||, e_max):
      - ``delta_r_iau_mars``: ||Δr_iau_mars|| in km
      - ``delta_v_iau_mars``: ||Δv_iau_mars|| in km/s
      - ``e_max``: within-sol peak osculating eccentricity, with
        scale_e_max=10.0 makes e_max subordinate to position and velocity
        closure.

    Both r/v deltas are r_iau(sol_end) − r_iau(sol_start) (resp. v_iau)
    after spice.sxform("J2000", "IAU_MARS", et) at each endpoint.

    The default scale for e_max=10.0 is set so the
    term reaches 1.0 only at unphysical e=10, keeping it strictly
    subordinate to closure for all real eccentricities (e<1).
    """

    def test_zero_drift_zero_cost(self):
        """Both norms zero AND fluence above floor → cost=0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        assert cost == pytest.approx(0.0, abs=1e-15)
        # 3 element terms (delta_r, delta_v, e_max) + 2 cum (cum_Δr, cum_Δv)
        # + 1 fluence_reward (default weight 0.0 -> term 0)
        # + 1 fluence_penalty + 1 delta_raan_vs_sunsync
        # (default weight 0.0 -> term 0) = 8.
        assert len(breakdown) == 8
        for k, v in breakdown:
            assert v == pytest.approx(0.0, abs=1e-15), f"{k}={v}"

    def test_pure_delta_r_at_one_scale_unit(self):
        """||Δr|| = scale_r → cost = 1.0 (the term reaches unity)."""
        scale_r = DEFAULT_RV_CLOSURE_SCALES["delta_r_iau_mars"]
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(scale_r, 0.0, 0.0),
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_r_iau_mars"] == pytest.approx(1.0, abs=1e-12)
        assert d["delta_v_iau_mars"] == 0.0
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_pure_delta_v_at_one_scale_unit(self):
        """||Δv|| = scale_v → cost = 1.0."""
        scale_v = DEFAULT_RV_CLOSURE_SCALES["delta_v_iau_mars"]
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_v_iau_mars_kmps=(0.0, scale_v, 0.0),
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_v_iau_mars"] == pytest.approx(1.0, abs=1e-12)
        assert d["delta_r_iau_mars"] == 0.0
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_norm_uses_3vector_magnitude(self):
        """The cost uses the EUCLIDEAN NORM of the 3-vector, not any
        single axis. ||(3,4,0)|| = 5 → term = (5/scale_r)^2."""
        scale_r = DEFAULT_RV_CLOSURE_SCALES["delta_r_iau_mars"]
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(3.0 * scale_r, 4.0 * scale_r, 0.0),
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_r_iau_mars"] == pytest.approx(25.0, abs=1e-12)

    def test_breakdown_sums_to_residual(self):
        """Σ(non-penalty breakdown) + penalty == cost."""
        scale_r = DEFAULT_RV_CLOSURE_SCALES["delta_r_iau_mars"]
        scale_v = DEFAULT_RV_CLOSURE_SCALES["delta_v_iau_mars"]
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=20.0,
            delta_r_iau_mars_km=(0.5 * scale_r, 0.3 * scale_r, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.2 * scale_v, 0.4 * scale_v),
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=2.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        total_from_breakdown = sum(v for _, v in breakdown)
        assert cost == pytest.approx(total_from_breakdown, abs=1e-12)

    def test_node_rate_default_off_bit_exact(self):
        """delta_raan_vs_sunsync defaults to weight 0.0, so the in-phase
        node-rate term is EXACTLY 0.0 even when the per-sol RAAN rate error
        is nonzero — the rv_closure cost stays bit-exact for callers that
        never opt in."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
            delta_raan_deg=0.05, raan_sun_sync_target_deg=0.02,  # 0.03 deg/sol rate error
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_raan_vs_sunsync"] == 0.0  # default weight 0 → term off
        assert cost == pytest.approx(0.0, abs=1e-15)

    def test_node_rate_activates_via_multi_element_weight(self):
        """Opting in via multi_element_weights adds exactly
        w·(rate_err/scale)² to the cost; at rate_err == scale the term = w."""
        scale = 0.01
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_raan_deg=0.02 + scale,  # rate error = exactly one scale unit
            raan_sun_sync_target_deg=0.02,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"delta_raan_vs_sunsync": 3.0},
            multi_element_scales={"delta_raan_vs_sunsync": scale},
        )
        d = dict(breakdown)
        assert d["delta_raan_vs_sunsync"] == pytest.approx(3.0, abs=1e-12)
        assert cost == pytest.approx(3.0, abs=1e-12)  # only the node-rate term active

    # -- e_max term tests --

    def test_e_max_term_at_one_scale_unit(self):
        """e_max = scale_e_max → e_max term = 1.0 (the term reaches
        unity). With default scale_e_max = 10.0, this requires the
        physically-impossible e_max = 10."""
        scale_e = DEFAULT_RV_CLOSURE_SCALES["e_max"]
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0, e_max=scale_e,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["e_max"] == pytest.approx(1.0, abs=1e-12)
        assert d["delta_r_iau_mars"] == 0.0
        assert d["delta_v_iau_mars"] == 0.0
        assert cost == pytest.approx(1.0, abs=1e-12)

    def test_e_max_term_at_calibration_target(self):
        """At e_max = 0.01 with default scale_e_max = 10.0 and weight = 1.0,
        e_max term = (0.01/10)² = 1e-6 — matching the converged-closure
        cost contribution of ||Δr|| = 1 km (which gives (1/1000)² = 1e-6).
        This is what makes e_max a strict tiebreaker that only registers
        once closure is essentially satisfied."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0, e_max=0.01,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["e_max"] == pytest.approx(1e-6, rel=1e-12)
        assert cost == pytest.approx(1e-6, rel=1e-12)

    def test_e_max_dominates_when_closure_converged(self):
        """When closure is fully satisfied (Δr=Δv=0) but e_max is
        substantial (0.5, far above operational ceiling ~0.05), the
        e_max term becomes the only nonzero gradient: cost = (0.5/10)² =
        0.0025. This is the "rv first THEN e_max" handoff: closure
        gradient is exhausted, e_max becomes the primary objective."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0, e_max=0.5,
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d = dict(breakdown)
        assert d["delta_r_iau_mars"] == 0.0
        assert d["delta_v_iau_mars"] == 0.0
        assert d["e_max"] == pytest.approx(0.0025, abs=1e-12)
        assert cost == pytest.approx(0.0025, abs=1e-12)

    def test_breakdown_length_seven(self):
        """Structural pin on the breakdown: exactly 7
        entries with the expected key order. Cum entries always
        present (value 0.0 when K_r=K_v=0); fluence_reward always
        present (value 0.0 when w_f=0).
        Mirrors the K_i pattern's cum_delta_e key always-present."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(1.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.001, 0.0),
            e_max=0.01,
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        assert len(breakdown) == 8
        names = [k for k, _ in breakdown]
        assert names == [
            "delta_r_iau_mars",
            "delta_v_iau_mars",
            "e_max",
            "cum_delta_r_iau_mars",
            "cum_delta_v_iau_mars",
            "fluence_reward",
            "fluence_penalty",
            "delta_raan_vs_sunsync",
        ]

    def test_e_max_scale_override_via_multi_element_scales(self):
        """multi_element_scales={"e_max": ...} overrides the default
        scale of 10.0, using the same override path as delta_r/delta_v."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0, e_max=0.005,
        )
        # Override scale to 0.005 → ratio = 1.0 → term = 1.0.
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_scales={"e_max": 0.005},
        )
        d = dict(breakdown)
        assert d["e_max"] == pytest.approx(1.0, abs=1e-12)
        # Default behavior at the same e_max (no override) gives
        # (0.005/10)² = 2.5e-7 — confirming default vs. override differ.
        cost_default, breakdown_default = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        d_default = dict(breakdown_default)
        assert d_default["e_max"] == pytest.approx(2.5e-7, rel=1e-12)

    # -- Default-pin and validation tests --

    def test_default_scales_pinned(self):
        """Defaults: r,v scales are 1000 km and 0.8 km/s. The e_max scale
        at 10.0 makes eccentricity a subordinate tiebreaker.
        Fluence uses a characteristic scale of 28.0. All non-fluence weights
        default to 1.0; fluence weight defaults to 0.0."""
        assert DEFAULT_RV_CLOSURE_SCALES["delta_r_iau_mars"] == 1000.0
        assert DEFAULT_RV_CLOSURE_SCALES["delta_v_iau_mars"] == 0.8
        assert DEFAULT_RV_CLOSURE_SCALES["e_max"] == 10.0
        assert DEFAULT_RV_CLOSURE_SCALES["fluence"] == 28.0
        for name in ("delta_r_iau_mars", "delta_v_iau_mars", "e_max"):
            assert DEFAULT_RV_CLOSURE_WEIGHTS[name] == 1.0
        assert DEFAULT_RV_CLOSURE_WEIGHTS["fluence"] == 0.0

    def test_custom_scale_override_propagates(self):
        """Per-element scale override: tightening ``delta_r_iau_mars``
        scale increases the term value at fixed ||Δr||.
        ||Δr|| = 1 km, scale=0.5 km → term = 4.0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(1.0, 0.0, 0.0),
        )
        cost, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_scales={"delta_r_iau_mars": 0.5},
        )
        d = dict(breakdown)
        assert d["delta_r_iau_mars"] == pytest.approx(4.0, abs=1e-12)

    def test_delta_e_same_u_not_in_rv_closure_vocabulary(self):
        """``delta_e_same_u`` is not an
        rv-closure element name (closure of (r, v) already implies
        Δe=0 at sol boundaries; including delta_e_same_u would
        double-count). Validation should reject it."""
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="not a recognised element"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
                multi_element_scales={"delta_e_same_u": 0.008},
            )

    def test_delta_r_iau_mars_not_in_closure_vocabulary(self):
        """Symmetric: ``delta_r_iau_mars`` rejected under closure_weighted."""
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        with pytest.raises(ValueError, match="not a recognised element"):
            cruise_cost_with_breakdown(
                ev, 25.0, 1.0,
                cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
                multi_element_scales={"delta_r_iau_mars": 1.0},
            )


class TestRVClosureWeightedCumulativeDeltaRV:
    """PI integral term on cumulative body-fixed Δr_iau / Δv_iau in
    rv_closure_weighted.
    Mirrors the K_i pattern in TestClosureWeightedCumulativeDeltaE.

    The PI term penalizes ‖cum_prior_3vec + Δ_current_3vec‖² scaled by
    scale_cum_*. Even though the cost is a scalar (Euclidean norm), the
    optimizer's gradient w.r.t. the per-sol residual is a vector that
    pulls Δ_current toward −cum_prior, biasing this-sol residual to
    cancel cumulative drift. Default K_r = K_v = 0.0 preserves the baseline
    breakdown.
    """

    def test_zero_weight_no_effect_r(self):
        """K_r=0, large cum_r — cost identical to no-cum call."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.5, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0005, 0.0),
            e_max=0.005,
        )
        cost_no_cum, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        cost_zero_K, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(100.0, 50.0, -75.0),
            cumulative_integral_weight_r=0.0,
        )
        assert cost_zero_K == pytest.approx(cost_no_cum, abs=1e-15)

    def test_zero_weight_no_effect_v(self):
        """K_v=0, large cum_v — cost identical to no-cum call."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.5, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0005, 0.0),
            e_max=0.005,
        )
        cost_no_cum, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        cost_zero_K, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_v_iau_kmps=(0.1, -0.05, 0.075),
            cumulative_integral_weight_v=0.0,
        )
        assert cost_zero_K == pytest.approx(cost_no_cum, abs=1e-15)

    def test_zero_weight_both_no_effect(self):
        """K_r = K_v = 0 with both cum vectors nonzero — bit-exact."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.3, 0.4, 0.0),
            delta_v_iau_mars_kmps=(0.001, 0.0, 0.0),
        )
        cost_a, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        cost_b, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(10.0, 20.0, -5.0),
            cumulative_delta_v_iau_kmps=(0.05, -0.02, 0.01),
            # K_r and K_v both default 0.0
        )
        assert cost_a == pytest.approx(cost_b, abs=1e-15)

    def test_positive_cum_r_biases_toward_negative(self):
        """K_r=1, cum_r=(scale, 0, 0). A this-sol Δr that opposes
        cum_prior gives lower cost than Δr=0."""
        scale = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM
        cum_r = (scale, 0.0, 0.0)

        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        ev_oppose = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(-scale / 2, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        cost_zero, _ = cruise_cost_with_breakdown(
            ev_zero, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=cum_r,
            cumulative_integral_weight_r=1.0,
        )
        cost_oppose, _ = cruise_cost_with_breakdown(
            ev_oppose, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=cum_r,
            cumulative_integral_weight_r=1.0,
        )
        # cost_zero: cum_total=(scale,0,0)→term=1.0; cost_oppose:
        # cum_total=(scale/2,0,0)→term=0.25 + per-sol term ≪ 1.
        assert cost_oppose < cost_zero

    def test_positive_cum_v_biases_toward_negative(self):
        """K_v=1, cum_v=(scale, 0, 0). Symmetric to the r case."""
        scale = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS
        cum_v = (scale, 0.0, 0.0)

        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        ev_oppose = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(-scale / 2, 0.0, 0.0),
        )
        cost_zero, _ = cruise_cost_with_breakdown(
            ev_zero, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_v_iau_kmps=cum_v,
            cumulative_integral_weight_v=1.0,
        )
        cost_oppose, _ = cruise_cost_with_breakdown(
            ev_oppose, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_v_iau_kmps=cum_v,
            cumulative_integral_weight_v=1.0,
        )
        assert cost_oppose < cost_zero

    def test_cum_r_uses_3vector_norm(self):
        """cum_r = (3,4,0)·scale/5 → ‖cum_r‖ = scale → cum_r term =
        K_r·1² = 1.0. Verifies the Euclidean-norm path (not single-axis
        magnitude)."""
        scale = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM
        cum_r = (3.0 * scale / 5.0, 4.0 * scale / 5.0, 0.0)
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=cum_r,
            cumulative_integral_weight_r=1.0,
        )
        d = dict(breakdown)
        assert d["cum_delta_r_iau_mars"] == pytest.approx(1.0, abs=1e-12)

    def test_cum_zero_adds_to_per_sol_for_r(self):
        """K_r=1, cum_r=(0,0,0), Δr_current=(scale_cum_r, 0, 0).
        Per-sol Δr term = (scale_cum_r/scale_r)² = (5/1000)² = 2.5e-5.
        Cum term = (scale_cum_r/scale_cum_r)² = 1.0.
        The cumulative term is much larger than the per-sol term."""
        scale_cum = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM
        scale_r = DEFAULT_RV_CLOSURE_SCALES["delta_r_iau_mars"]
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(scale_cum, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(0.0, 0.0, 0.0),
            cumulative_integral_weight_r=1.0,
        )
        d = dict(breakdown)
        assert d["cum_delta_r_iau_mars"] == pytest.approx(1.0, abs=1e-12)
        assert d["delta_r_iau_mars"] == pytest.approx(
            (scale_cum / scale_r) ** 2, abs=1e-15
        )
        # Domination ratio ~ (scale_r / scale_cum)² = (1000/5)² = 4e4.
        assert d["cum_delta_r_iau_mars"] > 1e4 * d["delta_r_iau_mars"]

    def test_breakdown_length_seven_with_cum_active(self):
        """With K_r and K_v both > 0, breakdown still has exactly 7
        entries with the same key order; fluence_reward is always present."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.5, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0005, 0.0),
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(1.0, 1.0, 1.0),
            cumulative_delta_v_iau_kmps=(0.001, 0.001, 0.001),
            cumulative_integral_weight_r=1.0,
            cumulative_integral_weight_v=1.0,
        )
        assert len(breakdown) == 8
        names = [k for k, _ in breakdown]
        assert names == [
            "delta_r_iau_mars",
            "delta_v_iau_mars",
            "e_max",
            "cum_delta_r_iau_mars",
            "cum_delta_v_iau_mars",
            "fluence_reward",
            "fluence_penalty",
            "delta_raan_vs_sunsync",
        ]

    def test_analytical_minimum_with_cum_r(self):
        """At Δr_current = -cum_prior, cum term zeros out. Optimal cost
        equals per-sol term alone. Verifies that the norm preserves vector
        direction.
        """
        scale_cum = DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM
        cum_prior = (2.0 * scale_cum, 0.0, 0.0)  # 10 km in +x

        # Case A: Δr_current = -cum_prior → cum cancels.
        ev_cancel = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(-2.0 * scale_cum, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        cost_cancel, bd_cancel = cruise_cost_with_breakdown(
            ev_cancel, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=cum_prior,
            cumulative_integral_weight_r=1.0,
        )
        d_cancel = dict(bd_cancel)
        assert d_cancel["cum_delta_r_iau_mars"] == pytest.approx(0.0, abs=1e-15)

        # Case B: Δr_current = 0 → cum stays at 2·scale.
        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        cost_zero, bd_zero = cruise_cost_with_breakdown(
            ev_zero, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=cum_prior,
            cumulative_integral_weight_r=1.0,
        )
        d_zero = dict(bd_zero)
        # Cum term = (2·scale_cum/scale_cum)² = 4.
        assert d_zero["cum_delta_r_iau_mars"] == pytest.approx(4.0, abs=1e-12)
        # Cancel-case cost is dominated by per-sol Δr term
        # ((2·scale_cum)/scale_r)² = (10/1000)² = 1e-4.
        # Zero-case cost = 4.0 (cum term). 4-orders-of-magnitude gap.
        assert cost_cancel < cost_zero
        assert cost_zero / cost_cancel > 1e3

    def test_analytical_minimum_with_cum_v(self):
        """Symmetric to the r case for v."""
        scale_cum = DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS
        cum_prior = (2.0 * scale_cum, 0.0, 0.0)

        ev_cancel = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(-2.0 * scale_cum, 0.0, 0.0),
        )
        cost_cancel, bd_cancel = cruise_cost_with_breakdown(
            ev_cancel, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_v_iau_kmps=cum_prior,
            cumulative_integral_weight_v=1.0,
        )
        d_cancel = dict(bd_cancel)
        assert d_cancel["cum_delta_v_iau_mars"] == pytest.approx(0.0, abs=1e-15)

        ev_zero = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        cost_zero, bd_zero = cruise_cost_with_breakdown(
            ev_zero, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_v_iau_kmps=cum_prior,
            cumulative_integral_weight_v=1.0,
        )
        d_zero = dict(bd_zero)
        assert d_zero["cum_delta_v_iau_mars"] == pytest.approx(4.0, abs=1e-12)
        assert cost_cancel < cost_zero

    def test_default_scales_pinned_cum(self):
        """Pin module-level cumulative-scale defaults."""
        assert DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM == 5.0
        assert DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS == 0.005

    def test_cum_terms_are_rv_closure_only_r(self):
        """cum_r kwargs should be ignored under closure_weighted
        (those keys aren't part of the closure-weighted vocabulary)."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.05, fluence=30.0, delta_e_same_u=1e-3,
            delta_i_deg=0.01, delta_raan_deg=0.001,
            raan_sun_sync_target_deg=0.0,
        )
        cost_a, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        cost_b, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(10.0, 0.0, 0.0),
            cumulative_integral_weight_r=1.0,  # ← would matter under rv_closure
        )
        # closure_weighted ignores the cum_r kwargs entirely.
        assert cost_a == pytest.approx(cost_b, abs=1e-15)

    def test_cum_terms_are_rv_closure_only_v(self):
        """Symmetric: cum_v kwargs ignored under closure_weighted."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.05, fluence=30.0, delta_e_same_u=1e-3,
            delta_i_deg=0.01, delta_raan_deg=0.001,
            raan_sun_sync_target_deg=0.0,
        )
        cost_a, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
        )
        cost_b, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_CLOSURE_WEIGHTED,
            cumulative_delta_v_iau_kmps=(0.01, 0.0, 0.0),
            cumulative_integral_weight_v=1.0,
        )
        assert cost_a == pytest.approx(cost_b, abs=1e-15)

    def test_independence_K_r_K_v(self):
        """K_r=1, K_v=0 with both cum vectors nonzero — only r branch
        contributes to the cum term values."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(DEFAULT_SCALE_CUM_DELTA_R_IAU_MARS_KM, 0, 0),
            cumulative_delta_v_iau_kmps=(DEFAULT_SCALE_CUM_DELTA_V_IAU_MARS_KMPS, 0, 0),
            cumulative_integral_weight_r=1.0,
            cumulative_integral_weight_v=0.0,
        )
        d = dict(breakdown)
        assert d["cum_delta_r_iau_mars"] == pytest.approx(1.0, abs=1e-12)
        assert d["cum_delta_v_iau_mars"] == pytest.approx(0.0, abs=1e-15)

    def test_cum_scale_override_via_kwargs(self):
        """scale_cum_delta_r_iau_mars_km / scale_cum_delta_v_iau_mars_kmps
        overrides set the expected normalization: at scale=10 km, a
        cum of 5 km gives term = 0.25, while at scale=5 km (default), the
        same 5-km cum gives term = 1.0."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        _, bd_default = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(5.0, 0.0, 0.0),
            cumulative_integral_weight_r=1.0,
        )
        _, bd_override = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(5.0, 0.0, 0.0),
            cumulative_integral_weight_r=1.0,
            scale_cum_delta_r_iau_mars_km=10.0,  # override
        )
        d_default = dict(bd_default)
        d_override = dict(bd_override)
        assert d_default["cum_delta_r_iau_mars"] == pytest.approx(1.0, abs=1e-12)
        assert d_override["cum_delta_r_iau_mars"] == pytest.approx(0.25, abs=1e-12)


class TestRVClosureWeightedFluenceReward:
    """Fluence reward in rv_closure_weighted.

    cost_metric=rv_closure_weighted gains a 7th breakdown entry: a NEGATIVE
    quadratic term −w_f · (fluence_J_per_m2 / scale_f)². Promotes fluence
    delivery to a primary cost objective so the optimizer actively seeks
    high-fluence solutions rather than accepting the fluence produced by
    closure-only optimization.

    Default weight=0.0 leaves the term inactive. Enable it via
    multi_element_weights={"fluence": 1.0}.

    Coexists with the existing fluence_penalty (shortfall guard); the
    penalty stays zero in normal operation, while the reward contributes at
    all fluence levels when its weight is positive.
    """

    def test_zero_weight_no_effect(self):
        """w_f=0 leaves the reward term at zero."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=27.2,
            delta_r_iau_mars_km=(0.5, 0.5, 0.5),
            delta_v_iau_mars_kmps=(0.0001, 0.0001, 0.0001),
            e_max=0.005,
        )
        cost_default, bd_default = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
        )
        # Explicitly set the weight to 0 — should match default
        cost_explicit, bd_explicit = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 0.0},
        )
        assert cost_default == pytest.approx(cost_explicit, abs=1e-15)
        d_default = dict(bd_default)
        assert d_default["fluence_reward"] == 0.0

    def test_higher_fluence_lower_cost(self):
        """At fixed Δr=Δv=e_max=0, positive weight favors higher fluence."""
        ev_low = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=12.7,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
            e_max=0.0,
        )
        ev_high = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=27.2,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
            e_max=0.0,
        )
        kwargs = dict(
            fluence_floor_J_per_m2=11.46,  # below both fluence values
            penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
        )
        cost_low, _ = cruise_cost_with_breakdown(ev_low, **kwargs)
        cost_high, _ = cruise_cost_with_breakdown(ev_high, **kwargs)
        assert cost_high < cost_low
        # Magnitude check: at scale=28, w=1, the gap is
        # (27.2/28)² − (12.7/28)² = 0.944 − 0.206 = 0.738 (in absolute
        # value; cost is lower by 0.738 going low → high since term is
        # NEGATIVE).
        assert (cost_low - cost_high) == pytest.approx(0.738, abs=1e-3)

    def test_fluence_reward_sign_negative(self):
        """fluence_reward term value is strictly NEGATIVE (or zero) — never
        positive. Confirms the sign convention: cost decreases as fluence
        increases."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=20.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
            e_max=0.0,
        )
        _, bd = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=25.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
        )
        d = dict(bd)
        assert d["fluence_reward"] < 0.0
        # At fluence=20, scale=28, w=1: -(20/28)² = -0.510.
        assert d["fluence_reward"] == pytest.approx(-(20.0 / 28.0) ** 2, abs=1e-12)

    def test_fluence_reward_quadratic(self):
        """Reward magnitude scales as fluence², not linearly. Pin via two
        fluence values at fixed scale=10, w=1: fluence=10 → reward=−1.0;
        fluence=20 → reward=−4.0 (4× larger, not 2×)."""
        kwargs = dict(
            fluence_floor_J_per_m2=5.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
            multi_element_scales={"fluence": 10.0},
        )
        ev10 = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=10.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        ev20 = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=20.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        _, bd10 = cruise_cost_with_breakdown(ev10, **kwargs)
        _, bd20 = cruise_cost_with_breakdown(ev20, **kwargs)
        d10 = dict(bd10)
        d20 = dict(bd20)
        assert d10["fluence_reward"] == pytest.approx(-1.0, abs=1e-12)
        assert d20["fluence_reward"] == pytest.approx(-4.0, abs=1e-12)

    def test_default_scale_pinned(self):
        """Default characteristic scale is 28.0 J/m²."""
        from reflectors.optimize import DEFAULT_SCALE_FLUENCE_J_PER_M2
        assert DEFAULT_SCALE_FLUENCE_J_PER_M2 == 28.0
        assert DEFAULT_RV_CLOSURE_SCALES["fluence"] == 28.0

    def test_default_weight_zero(self):
        """Default weight is 0.0; callers explicitly opt in."""
        assert DEFAULT_RV_CLOSURE_WEIGHTS["fluence"] == 0.0

    def test_coexists_with_fluence_penalty(self):
        """Both reward AND penalty active simultaneously. With fluence
        BELOW the floor AND w_f>0, BOTH terms contribute: penalty
        (positive shortfall guard) and reward (negative seeking)."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=10.0,  # below floor 11.46
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
            e_max=0.0,
        )
        cost, bd = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=11.46, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
        )
        d = dict(bd)
        assert d["fluence_penalty"] > 0.0   # shortfall: penalty active
        assert d["fluence_reward"] < 0.0    # reward: reward active (still wants more)
        # Total cost = penalty + reward (delta_r/v/e_max/cum all 0)
        # penalty = (11.46 - 10)² = 2.1316
        # reward = -(10/28)² = -0.1276
        # cost = 2.1316 - 0.1276 = 2.0040
        assert cost == pytest.approx(2.0040, abs=1e-3)

    def test_fluence_scale_override(self):
        """multi_element_scales={"fluence": ...} overrides default 28.0.
        Override scale = 10 with fluence=10 gives reward = −1.0 vs. default
        scale 28 same fluence giving reward = -(10/28)² ≈ −0.128."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=10.0,
            delta_r_iau_mars_km=(0.0, 0.0, 0.0),
            delta_v_iau_mars_kmps=(0.0, 0.0, 0.0),
        )
        _, bd_default = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=5.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
        )
        _, bd_override = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=5.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
            multi_element_scales={"fluence": 10.0},
        )
        d_default = dict(bd_default)
        d_override = dict(bd_override)
        assert d_default["fluence_reward"] == pytest.approx(-(10.0 / 28.0) ** 2, abs=1e-12)
        assert d_override["fluence_reward"] == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Group 3: Optimizer wrapper input validation (fast)
# ---------------------------------------------------------------------------


class TestOptimizeCruiseInputValidation:
    def test_x0_wrong_shape_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="x0 must have shape"):
            optimize_cruise(
                cfg, x0=np.array([0.1, 0.2, 0.3]),
                fluence_floor_J_per_m2=10.0,
            )

    def test_negative_lambda_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="penalty_lambda"):
            optimize_cruise(
                cfg, x0=np.array([0.0, 0.0]),
                fluence_floor_J_per_m2=10.0,
                penalty_lambda=-1.0,
            )

    def test_x0_outside_beta_bounds_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="beta_bounds"):
            optimize_cruise(
                cfg, x0=np.array([math.radians(45.0), 0.0]),
                fluence_floor_J_per_m2=10.0,
                beta_bounds=(0.0, math.radians(30.0)),
            )

    def test_degenerate_bounds_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="beta_bounds"):
            optimize_cruise(
                cfg, x0=np.array([0.0, 0.0]),
                fluence_floor_J_per_m2=10.0,
                beta_bounds=(0.5, 0.5),
            )

    def test_invalid_cost_metric_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="cost_metric"):
            optimize_cruise(
                cfg, x0=np.array([0.0, 0.0]),
                fluence_floor_J_per_m2=10.0,
                cost_metric="bogus",
            )


# ---------------------------------------------------------------------------
# Group 4: initial_state_override (fast)
# ---------------------------------------------------------------------------


class TestOrbitConfigInitialStateOverride:
    def test_override_replaces_sun_sync_state(self):
        """When initial_state_override_km_kmps is supplied, OrbitConfig
        skips initial_state_j2000 and uses the override directly.
        Used for periodic re-optimization where each sol's initial state
        is the previous sol's propagated end-state."""
        custom = np.array(
            [3897.19, 100.0, 50.0, 0.5, 1.5, 3.0], dtype=float,
        )
        cfg = OrbitConfig(
            a_km=A_KM, ltan_h=LTAN_H, M0_rad=math.radians(M0_DEG),
            epoch_et=utc_to_et(EPOCH_UTC),
            duration_s=DURATION_S,
            sail=make_canonical_sail(0.05),
            target_lat_deg=40.0, target_lon_deg=200.0,
            alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
            initial_state_override_km_kmps=custom,
        )
        np.testing.assert_array_equal(cfg.initial_state_km_kmps, custom)

    def test_override_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            OrbitConfig(
                a_km=A_KM, ltan_h=LTAN_H, M0_rad=0.0,
                epoch_et=utc_to_et(EPOCH_UTC),
                duration_s=DURATION_S,
                sail=make_canonical_sail(0.05),
                target_lat_deg=40.0, target_lon_deg=200.0,
                alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
                initial_state_override_km_kmps=np.zeros(7),
            )


# ---------------------------------------------------------------------------
# Group 5: Cruise-factory abstraction (fast)
# ---------------------------------------------------------------------------


class TestCruiseFactory:
    """Smoke + identity tests for the parameterization-agnostic factory.

    Avoids invoking ``evaluate_cruise_general`` (which propagates) — pins
    only the factory contract: it returns a callable that produces unit
    vectors of shape (3,) from (r_sat_km, et).
    """

    def test_constant_factory_returns_attitude_callable(self):
        cfg = _reference_config()
        x = np.array([math.radians(5.0), math.radians(45.0)])
        attitude = _constant_cruise_factory(x, cfg)
        # Sample at the initial state.
        r_sat = cfg.initial_state_km_kmps[:3]
        n_hat = attitude(r_sat, cfg.epoch_et)
        assert n_hat.shape == (3,)
        assert np.isclose(np.linalg.norm(n_hat), 1.0, atol=1e-12)

    def test_harmonic_factory_returns_attitude_callable(self):
        cfg = _reference_config()
        x = np.array([
            math.radians(5.0), math.radians(2.0),
            math.radians(2.0), math.radians(45.0),
        ])
        attitude = _harmonic_cruise_factory(x, cfg)
        r_sat = cfg.initial_state_km_kmps[:3]
        n_hat = attitude(r_sat, cfg.epoch_et)
        assert n_hat.shape == (3,)
        assert np.isclose(np.linalg.norm(n_hat), 1.0, atol=1e-12)

    def test_harmonic_factory_zero_amplitudes_matches_constant(self):
        """At α_c = α_s = 0, harmonic factory returns same n_hat as
        constant factory at the same DC cone-angle / clock-angle."""
        cfg = _reference_config()
        beta = math.radians(7.5)
        phi_u = math.radians(60.0)
        x_const = np.array([beta, phi_u])
        x_harm = np.array([beta, 0.0, 0.0, phi_u])
        a_const = _constant_cruise_factory(x_const, cfg)
        a_harm = _harmonic_cruise_factory(x_harm, cfg)
        r_sat = cfg.initial_state_km_kmps[:3]
        n_const = a_const(r_sat, cfg.epoch_et)
        n_harm = a_harm(r_sat, cfg.epoch_et)
        np.testing.assert_array_equal(n_const, n_harm)

    def test_harmonic_factory_raises_on_cone_angle_lower_violation(self):
        """When α_0 - sqrt(α_c² + α_s²) < 0, the cone angle would dip
        below 0 over the u sweep — sun_offset_harmonic's construction-
        time validator raises. Pin this so the optimizer's
        infeasibility guard can rely on the ValueError signal.
        """
        cfg = _reference_config()
        x_bad = np.array([math.radians(5.0), math.radians(30.0),
                           0.0, math.pi])  # α_0 - α_amp = -25°
        with pytest.raises(ValueError, match=">= 0"):
            _harmonic_cruise_factory(x_bad, cfg)

    def test_harmonic_factory_raises_on_cone_angle_upper_violation(self):
        """When α_0 + sqrt(α_c² + α_s²) > π/2, the cone angle would
        exceed π/2 over the u sweep — sun_offset_harmonic's validator
        raises."""
        cfg = _reference_config()
        x_bad = np.array([math.radians(70.0), math.radians(25.0),
                           math.radians(25.0), math.pi])
        with pytest.raises(ValueError, match="<= pi/2"):
            _harmonic_cruise_factory(x_bad, cfg)

    # ---- harmonic-(alpha, delta) factory tests ----

    def test_harmonic_full_factory_returns_attitude_callable(self):
        """6-D x → unit-vector AttitudeCallable."""
        cfg = _reference_config()
        x = np.array([
            math.radians(5.0), math.radians(2.0), math.radians(1.0),
            math.radians(45.0), math.radians(10.0), math.radians(5.0),
        ])
        attitude = _harmonic_full_cruise_factory(x, cfg)
        r_sat = cfg.initial_state_km_kmps[:3]
        n_hat = attitude(r_sat, cfg.epoch_et)
        assert n_hat.shape == (3,)
        assert np.isclose(np.linalg.norm(n_hat), 1.0, atol=1e-12)

    def test_harmonic_full_factory_zero_delta_amps_matches_harmonic_factory(self):
        """At δ_c = δ_s = 0, harmonic-full factory returns same n_hat as
        harmonic factory at the same (α_0, α_c, α_s, δ_0=phi_u)."""
        cfg = _reference_config()
        alpha_0 = math.radians(7.5)
        alpha_c = math.radians(2.0)
        alpha_s = math.radians(1.5)
        delta_0 = math.radians(60.0)
        x_harm = np.array([alpha_0, alpha_c, alpha_s, delta_0])
        x_full = np.array([alpha_0, alpha_c, alpha_s, delta_0, 0.0, 0.0])
        a_harm = _harmonic_cruise_factory(x_harm, cfg)
        a_full = _harmonic_full_cruise_factory(x_full, cfg)
        r_sat = cfg.initial_state_km_kmps[:3]
        n_harm = a_harm(r_sat, cfg.epoch_et)
        n_full = a_full(r_sat, cfg.epoch_et)
        np.testing.assert_array_equal(n_harm, n_full)

    def test_harmonic_full_factory_zero_all_amps_matches_constant_factory(self):
        """α_c = α_s = δ_c = δ_s = 0 reduces to constant cruise at
        (β = α_0, φ_u = δ_0)."""
        cfg = _reference_config()
        beta = math.radians(7.5)
        phi_u = math.radians(60.0)
        x_const = np.array([beta, phi_u])
        x_full = np.array([beta, 0.0, 0.0, phi_u, 0.0, 0.0])
        a_const = _constant_cruise_factory(x_const, cfg)
        a_full = _harmonic_full_cruise_factory(x_full, cfg)
        r_sat = cfg.initial_state_km_kmps[:3]
        n_const = a_const(r_sat, cfg.epoch_et)
        n_full = a_full(r_sat, cfg.epoch_et)
        np.testing.assert_array_equal(n_const, n_full)

    def test_harmonic_full_factory_raises_on_cone_angle_violation(self):
        """The cone-angle constraint binds in the full harmonic family."""
        cfg = _reference_config()
        x_bad = np.array([
            math.radians(5.0), math.radians(30.0), 0.0,
            0.0, 0.0, 0.0,
        ])  # α_0 - α_amp = -25°
        with pytest.raises(ValueError, match=">= 0"):
            _harmonic_full_cruise_factory(x_bad, cfg)

    def test_harmonic_full_factory_raises_on_delta_amp_violation(self):
        """δ_amp > π/2 (default) raises through the factory."""
        cfg = _reference_config()
        # δ_c = δ_s = π/2 each → amp ≈ 2.22 > π/2.
        x_bad = np.array([
            math.radians(5.0), 0.0, 0.0,
            math.radians(60.0), math.pi / 2.0, math.pi / 2.0,
        ])
        with pytest.raises(ValueError, match="delta_amp_max_rad"):
            _harmonic_full_cruise_factory(x_bad, cfg)

    # ---- Mode-2 factory ----

    def test_mode2_factory_returns_attitude_callable(self):
        cfg = _reference_config()
        x = np.array([
            math.radians(10.0),
            math.radians(2.0), math.radians(1.5),
            math.radians(0.5), math.radians(0.3),
            math.radians(180.0),
            math.radians(15.0), math.radians(-10.0),
            math.radians(5.0), math.radians(3.0),
        ])
        n_hat = _harmonic_full_mode2_cruise_factory(x, cfg)
        assert callable(n_hat)
        n = n_hat(cfg.initial_state_km_kmps[:3], cfg.epoch_et)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1.0e-12)

    def test_mode2_factory_zero_mode2_amps_matches_mode1_factory(self):
        """With α_c2=α_s2=δ_c2=δ_s2=0, the mode-2 factory must produce
        bit-equal n_hat to ``_harmonic_full_cruise_factory`` evaluated
        on (α_0, α_c1, α_s1, δ_0, δ_c1, δ_s1)."""
        cfg = _reference_config()
        x_mode1 = np.array([
            math.radians(10.0), math.radians(2.0), math.radians(1.5),
            math.radians(180.0), math.radians(15.0), math.radians(-10.0),
        ])
        x_mode2 = np.array([
            x_mode1[0], x_mode1[1], x_mode1[2], 0.0, 0.0,
            x_mode1[3], x_mode1[4], x_mode1[5], 0.0, 0.0,
        ])
        n_mode1 = _harmonic_full_cruise_factory(x_mode1, cfg)
        n_mode2 = _harmonic_full_mode2_cruise_factory(x_mode2, cfg)
        for dt in [0.0, 60.0, 3600.0]:
            et = cfg.epoch_et + dt
            v1 = n_mode1(cfg.initial_state_km_kmps[:3], et)
            v2 = n_mode2(cfg.initial_state_km_kmps[:3], et)
            np.testing.assert_array_equal(v1, v2)

    def test_mode2_factory_raises_on_cone_angle_violation(self):
        """Conservative cone-angle bound is enforced through the factory."""
        cfg = _reference_config()
        x_bad = np.array([
            math.radians(30.0),
            math.radians(35.0), 0.0, math.radians(35.0), 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ])  # 30 + 35 + 35 = 100° > 90°
        with pytest.raises(ValueError, match="alpha_0 \\+"):
            _harmonic_full_mode2_cruise_factory(x_bad, cfg)


# ---------------------------------------------------------------------------
# Group 6: optimize_cruise_general / optimize_cruise_harmonic input validation
# ---------------------------------------------------------------------------


class TestOptimizeCruiseGeneralInputValidation:
    def test_x0_empty_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="non-empty"):
            optimize_cruise_general(
                cfg, x0=np.array([]),
                cruise_factory=_constant_cruise_factory,
                bounds=[],
                fluence_floor_J_per_m2=10.0,
            )

    def test_bounds_length_mismatch_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="bounds length"):
            optimize_cruise_general(
                cfg, x0=np.array([0.0, 0.0, 0.0]),
                cruise_factory=_harmonic_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 1.0)],  # 2 bounds vs 3-D x0
                fluence_floor_J_per_m2=10.0,
            )

    def test_x0_outside_bounds_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match=r"outside bounds\[0\]"):
            optimize_cruise_general(
                cfg, x0=np.array([2.0, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
            )

    def test_degenerate_bound_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match=r"bounds\[0\] must satisfy"):
            optimize_cruise_general(
                cfg, x0=np.array([0.5, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.5, 0.5), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
            )

    def test_algorithm_invalid_raises(self):
        """``algorithm`` validation rejects unknown selectors."""
        cfg = _reference_config()
        with pytest.raises(ValueError, match="algorithm must be one of"):
            optimize_cruise_general(
                cfg, x0=np.array([0.05, math.pi]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
                algorithm="genetic-algorithm",   # not supported
            )

    def test_algorithm_constants_exported(self):
        """Both algorithm selectors are importable and present in
        the validator's frozen vocabulary."""
        from reflectors.optimize import _VALID_ALGORITHMS  # noqa: PLC0415
        assert ALGORITHM_NELDER_MEAD == "nelder-mead"
        assert ALGORITHM_DIFFERENTIAL_EVOLUTION == "differential_evolution"
        assert ALGORITHM_NELDER_MEAD in _VALID_ALGORITHMS
        assert ALGORITHM_DIFFERENTIAL_EVOLUTION in _VALID_ALGORITHMS


class TestWarmStartPopulationForDE:
    """``warm_start_population_for_de`` helper for DE
    warm-start. Returns ``(M, N)`` array suitable as the ``init=`` kwarg
    of ``scipy.optimize.differential_evolution``. Row 0 is the prior;
    rows 1..(M-1) are Gaussian-jittered around the prior, all clamped
    to bounds.
    """

    def _bounds_2d(self):
        return [(0.0, 1.0), (-2.0, 2.0)]

    def test_shape_is_popsize_times_n_by_n(self):
        bounds = self._bounds_2d()
        x_prior = np.array([0.5, 0.0])
        pop = warm_start_population_for_de(
            x_prior, bounds, popsize=15, jitter_frac=0.05, seed=0,
        )
        # M = popsize * N = 15 * 2 = 30; N = 2.
        assert pop.shape == (30, 2)
        assert pop.dtype == np.float64

    def test_row_zero_is_x_prior_clamped(self):
        bounds = self._bounds_2d()
        x_prior = np.array([0.5, 0.0])
        pop = warm_start_population_for_de(
            x_prior, bounds, popsize=5, jitter_frac=0.1, seed=1,
        )
        # In-bounds prior preserved exactly in row 0.
        np.testing.assert_array_equal(pop[0], x_prior)

    def test_row_zero_clamped_when_x_prior_outside_bounds(self):
        bounds = self._bounds_2d()
        # x_prior[0]=2.0 outside [0,1]; x_prior[1]=-3.0 outside [-2, 2].
        x_prior = np.array([2.0, -3.0])
        pop = warm_start_population_for_de(
            x_prior, bounds, popsize=3, jitter_frac=0.0, seed=2,
        )
        # Clamped per-axis: [1.0, -2.0].
        np.testing.assert_array_equal(pop[0], np.array([1.0, -2.0]))

    def test_all_rows_inside_bounds(self):
        bounds = self._bounds_2d()
        x_prior = np.array([0.5, 0.0])
        # Large jitter to stress the per-axis clamp.
        pop = warm_start_population_for_de(
            x_prior, bounds, popsize=20, jitter_frac=1.5, seed=42,
        )
        assert (pop[:, 0] >= 0.0).all()
        assert (pop[:, 0] <= 1.0).all()
        assert (pop[:, 1] >= -2.0).all()
        assert (pop[:, 1] <= 2.0).all()

    def test_jitter_frac_zero_gives_constant_population(self):
        bounds = self._bounds_2d()
        x_prior = np.array([0.7, 1.5])
        pop = warm_start_population_for_de(
            x_prior, bounds, popsize=10, jitter_frac=0.0, seed=3,
        )
        # All rows = clipped x_prior = x_prior (already in bounds).
        for i in range(pop.shape[0]):
            np.testing.assert_array_equal(pop[i], x_prior)

    def test_seed_reproducible(self):
        bounds = self._bounds_2d()
        x_prior = np.array([0.5, 0.0])
        pop_a = warm_start_population_for_de(
            x_prior, bounds, popsize=15, jitter_frac=0.05, seed=12345,
        )
        pop_b = warm_start_population_for_de(
            x_prior, bounds, popsize=15, jitter_frac=0.05, seed=12345,
        )
        np.testing.assert_array_equal(pop_a, pop_b)

    def test_seed_different_gives_different_jitter(self):
        bounds = self._bounds_2d()
        x_prior = np.array([0.5, 0.0])
        pop_a = warm_start_population_for_de(
            x_prior, bounds, popsize=15, jitter_frac=0.05, seed=1,
        )
        pop_b = warm_start_population_for_de(
            x_prior, bounds, popsize=15, jitter_frac=0.05, seed=2,
        )
        # Row 0 deterministic (= x_prior) for both.
        np.testing.assert_array_equal(pop_a[0], pop_b[0])
        # Rows 1..M-1 differ.
        assert not np.array_equal(pop_a[1:], pop_b[1:])

    def test_x_prior_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            warm_start_population_for_de(
                np.array([]), bounds=[], popsize=5,
            )

    def test_bounds_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="bounds length"):
            warm_start_population_for_de(
                np.array([0.5, 0.0, 0.0]),
                bounds=[(0.0, 1.0), (-1.0, 1.0)],  # 2 vs 3
                popsize=5,
            )

    def test_degenerate_bound_raises(self):
        with pytest.raises(ValueError, match=r"bounds\[0\] must satisfy"):
            warm_start_population_for_de(
                np.array([0.5, 0.0]),
                bounds=[(0.5, 0.5), (0.0, 1.0)],
                popsize=5,
            )

    def test_popsize_below_one_raises(self):
        with pytest.raises(ValueError, match="popsize"):
            warm_start_population_for_de(
                np.array([0.5, 0.0]),
                bounds=[(0.0, 1.0), (-1.0, 1.0)],
                popsize=0,
            )

    def test_negative_jitter_frac_raises(self):
        with pytest.raises(ValueError, match="jitter_frac"):
            warm_start_population_for_de(
                np.array([0.5, 0.0]),
                bounds=[(0.0, 1.0), (-1.0, 1.0)],
                popsize=5,
                jitter_frac=-0.01,
            )


class TestOptimizeCruiseHarmonicInputValidation:
    def test_x0_wrong_dimension_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="bounds length"):
            optimize_cruise_harmonic(
                cfg, x0=np.array([0.05, 0.0]),  # 2-D, but harmonic is 4-D
                fluence_floor_J_per_m2=10.0,
            )

    def test_invalid_cost_metric_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="cost_metric"):
            optimize_cruise_harmonic(
                cfg, x0=np.array([0.05, 0.0, 0.0, math.pi]),
                fluence_floor_J_per_m2=10.0,
                cost_metric="bogus",
            )

    def test_x0_outside_alpha_0_bounds_raises(self):
        cfg = _reference_config()
        # alpha_0 default upper bound is radians(60) = 1.047; provide alpha_0=1.5.
        with pytest.raises(ValueError, match="outside bounds"):
            optimize_cruise_harmonic(
                cfg, x0=np.array([1.5, 0.0, 0.0, math.pi]),
                fluence_floor_J_per_m2=10.0,
            )


class TestOptimizeCruiseHarmonicFullInputValidation:
    """Six-dimensional wrapper input validation. Construction-only; does
    NOT invoke evaluate_cruise_general (which propagates), so each test
    completes in milliseconds.
    """

    def test_x0_wrong_dimension_raises(self):
        """The full harmonic family expects 6-D x0."""
        cfg = _reference_config()
        with pytest.raises(ValueError, match="bounds length"):
            optimize_cruise_harmonic_full(
                cfg, x0=np.array([0.05, 0.0, 0.0, math.pi]),  # 4-D
                fluence_floor_J_per_m2=10.0,
            )

    def test_invalid_cost_metric_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="cost_metric"):
            optimize_cruise_harmonic_full(
                cfg,
                x0=np.array([0.05, 0.0, 0.0, math.pi, 0.0, 0.0]),
                fluence_floor_J_per_m2=10.0,
                cost_metric="bogus",
            )

    def test_x0_outside_alpha_0_bounds_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="outside bounds"):
            optimize_cruise_harmonic_full(
                cfg,
                x0=np.array([1.5, 0.0, 0.0, math.pi, 0.0, 0.0]),
                fluence_floor_J_per_m2=10.0,
            )

    def test_x0_outside_delta_c_bounds_raises(self):
        cfg = _reference_config()
        # delta_c default bounds: ±60° = ±1.047. Provide 1.5.
        with pytest.raises(ValueError, match="outside bounds"):
            optimize_cruise_harmonic_full(
                cfg,
                x0=np.array([0.05, 0.0, 0.0, math.pi, 1.5, 0.0]),
                fluence_floor_J_per_m2=10.0,
            )

    def test_negative_penalty_lambda_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="penalty_lambda"):
            optimize_cruise_harmonic_full(
                cfg,
                x0=np.array([0.05, 0.0, 0.0, math.pi, 0.0, 0.0]),
                fluence_floor_J_per_m2=10.0,
                penalty_lambda=-1.0,
            )


def test_full_evaluate_phi_u_rad_is_x_minus_one():
    """Under decision-vector convention
    [α_0, α_c, α_s, δ_0, δ_c, δ_s], EvaluationResult.phi_u_rad equals
    x[-1] = delta_s, not delta_0. The canonical record is decision_vector_rad.
    CSV consumers must read decision_vector_rad[3] for the
    DC clock angle, not phi_u_rad.

    Construction-only test — does NOT propagate, so it's fast.
    """
    cfg = _reference_config()
    # Build a synthetic eval at an arbitrary 6-D vector, bypassing the
    # propagator by building EvaluationResult directly.
    alpha_0 = math.radians(5.0)
    alpha_c = math.radians(2.0)
    alpha_s = math.radians(1.5)
    delta_0 = math.radians(120.0)
    delta_c = math.radians(15.0)
    delta_s = math.radians(20.0)
    x = (alpha_0, alpha_c, alpha_s, delta_0, delta_c, delta_s)
    ev = EvaluationResult(
        beta_rad=alpha_0,                # = x[0]
        phi_u_rad=delta_s,               # scalar summary of x[-1]
        total_fluence_J_per_m2=0.0,
        delta_a_km=0.0,
        delta_a_per_sol_km=0.0,
        e_max=0.0,
        inc_range_deg=0.0,
        n_windows_kept=0,
        n_windows_dropped=0,
        n_unstable_windows=0,
        converged=True,
        n_iterations=0,
        wall_s=0.0,
        decision_vector_rad=x,
    )
    # Pin: x[-1] = delta_s, not delta_0.
    assert ev.phi_u_rad == delta_s
    assert ev.phi_u_rad != delta_0
    # Canonical record: decision_vector_rad has the full 6-tuple.
    assert ev.decision_vector_rad == x
    assert ev.decision_vector_rad[3] == delta_0  # canonical δ_0 lookup


# ---------------------------------------------------------------------------
# Group 4: Slow physics validation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_optimize_cruise_multi_element_records_breakdown_in_history():
    """Under cost_metric='multi_element_weighted', every
    EvaluationResult in history.cost_breakdown is populated with the 5
    expected element-name keys, AND the OptimizationRun records the
    resolved weights/scales used.

    Uses a TINY maxfev=3 so the test is fast (~6-9 s wall) but still
    exercises the end-to-end factory + cost + history pipeline.
    """
    cfg = _reference_config()
    # Reuse one baseline evaluation across the optimization.
    baseline = evaluate_cruise(0.0, 0.0, cfg)
    run = optimize_cruise(
        cfg,
        x0=np.array([math.radians(5.0), math.radians(90.0)]),
        fluence_floor_J_per_m2=0.9 * baseline.total_fluence_J_per_m2,
        penalty_lambda=1.0,
        cost_metric=COST_METRIC_MULTI_ELEMENT_WEIGHTED,
        baseline_eval=baseline,
        nelder_mead_options={"maxfev": 3, "maxiter": 3},
    )
    assert len(run.history) >= 1
    expected_keys = {
        "delta_a", "e_max", "delta_i", "delta_raan_vs_sunsync",
        "fluence_penalty",
    }
    for ev in run.history:
        assert ev.cost_breakdown != ()
        keys = {k for k, _ in ev.cost_breakdown}
        assert keys == expected_keys, (
            f"breakdown missing keys: expected {expected_keys}, got {keys}"
        )
    # OptimizationRun records the resolved weights/scales (default = 1.0
    # per element, default scales).
    assert run.multi_element_weights == dict(DEFAULT_MULTI_ELEMENT_WEIGHTS)
    assert run.multi_element_scales == dict(DEFAULT_MULTI_ELEMENT_SCALES)


@pytest.mark.slow
def test_evaluate_cruise_populates_element_fields():
    """At the canonical baseline (beta=0, phi_u=0),
    the EvaluationResult fields delta_i_deg, delta_raan_deg,
    raan_sun_sync_target_deg, e_end_osculating must be finite, of the
    expected sign / order, and self-consistent.

    Magnitudes pinned from the baseline: natural Δi ≈ +0.10°/sol,
    ΔRAAN ≈ +0.53°/sol (≈ sun-sync
    target), e_max ≈ 0.003 at sol 1. The orders of magnitude are pinned
    and signs, NOT tight values, to leave room for the propagator's
    natural noise across rebuilds.
    """
    cfg = _reference_config()
    ev = evaluate_cruise(0.0, 0.0, cfg)

    # Finite and non-NaN.
    assert math.isfinite(ev.delta_i_deg)
    assert math.isfinite(ev.delta_raan_deg)
    assert math.isfinite(ev.raan_sun_sync_target_deg)
    assert math.isfinite(ev.e_end_osculating)

    # Sun-sync target is the design value; positive over a 1-sol sidereal
    # horizon, ~0.539° per sol.
    assert ev.raan_sun_sync_target_deg == pytest.approx(0.539, abs=5e-3)

    # ΔRAAN observed should be close to the sun-sync target (the orbit
    # was constructed for sun-sync at this i, a). Within ~0.05° per sol.
    miss = ev.delta_raan_deg - ev.raan_sun_sync_target_deg
    assert abs(miss) < 0.05, (
        f"ΔRAAN miss from sun-sync target = {miss:.4f}° per sol "
        f"(observed Δ={ev.delta_raan_deg:.4f}°, target="
        f"{ev.raan_sun_sync_target_deg:.4f}°)"
    )

    # |Δi| should be small per sol (sun-sync ideally = 0; J3 + third-body
    # drive a small drift on the order of 0.1°/sol).
    assert abs(ev.delta_i_deg) < 1.0

    # e_end is bounded by e_max (peak over trajectory ≥ end value).
    assert 0.0 <= ev.e_end_osculating <= ev.e_max + 1e-12

    # cost_breakdown empty for single-element costs (default).
    assert ev.cost_breakdown == ()


@pytest.mark.slow
def test_evaluate_cruise_populates_closure_fields():
    """At the canonical baseline (beta=0, phi_u=0),
    the EvaluationResult fields delta_e_same_u, delta_argp_deg,
    argp_J2_target_deg must be finite and of the expected sign / order.

    The canonical config is at sigma=0.05, a=3897 km (LMO 501 km altitude),
    not the K=12 geometry; both should satisfy the qualitative checks below.
    """
    cfg = _reference_config()
    ev = evaluate_cruise(0.0, 0.0, cfg)

    assert math.isfinite(ev.delta_e_same_u)
    assert math.isfinite(ev.delta_argp_deg)
    assert math.isfinite(ev.argp_J2_target_deg)

    # The baseline orbit is sun-synchronous (i approximately 92.92 deg at LMO),
    # well above the critical inclination ⇒ argp regresses ⇒ target < 0.
    assert ev.argp_J2_target_deg < 0.0
    # |target| should be a few deg/sol order at this inclination.
    assert 1.0 < abs(ev.argp_J2_target_deg) < 10.0

    # |Δe_same_u| should be small per sol (sail authority bounded
    # secular Delta-<e> at sigma=0.05 by approximately 0.003/sol).
    assert abs(ev.delta_e_same_u) < 0.01


@pytest.mark.slow
def test_evaluate_cruise_beta_zero_reproduces_sun_pointing():
    """β=0 ⟹ cruise.sun_offset reduces to sun_pointing bit-for-bit
    (cruise.py:50). evaluate_cruise(0, 0, config) should therefore
    produce the same propagated trajectory + windows as a direct
    refine_delivery_schedule call with cruise_profile=sun_pointing().

    This test pins ``evaluate_cruise`` against the direct propagation
    pipeline.
    """
    from reflectors.attitude import sun_pointing
    from reflectors.attitude_schedule import refine_delivery_schedule
    from reflectors.elements import elements_in_mme2000
    from reflectors.sun_sync import _default_gravity_anchors

    cfg = _reference_config()
    ev = evaluate_cruise(0.0, 0.0, cfg)

    # Direct sun-pointing reference.
    ref = refine_delivery_schedule(
        cfg.initial_state_km_kmps,
        t_span_s=(0.0, cfg.duration_s),
        epoch_et=cfg.epoch_et,
        cruise_profile=sun_pointing(),
        target_lat_deg=cfg.target_lat_deg,
        target_lon_deg=cfg.target_lon_deg,
        sail=cfg.sail,
        slew_duration_s=cfg.slew_duration_s,
        alpha_max_rad_s2=cfg.alpha_max_rad_s2,
        max_iterations=cfg.max_iterations,
        convergence_tol_et_s=cfg.convergence_tol_et_s,
        damping=cfg.damping,
        propagate_kwargs=dict(cfg.propagate_kwargs),
        find_windows_kwargs=dict(cfg.find_windows_kwargs),
    )
    ref_fluence = sum(
        float(w.fluence_J_per_m2) for w in ref.final_windows
        if w.fluence_J_per_m2 is not None
    )
    mu, _R, _J2 = _default_gravity_anchors()
    ref_a0 = elements_in_mme2000(
        ref.final_result.state_km_kmps[0], mu, epoch_et=cfg.epoch_et,
    ).a_km
    ref_af = elements_in_mme2000(
        ref.final_result.state_km_kmps[-1], mu,
        epoch_et=cfg.epoch_et + float(ref.final_result.t_s[-1]),
    ).a_km
    ref_delta_a = ref_af - ref_a0

    # β=0 short-circuits the cruise.sun_offset interior to return s_hat
    # directly (cruise.py:252), so the resulting trajectory must match
    # the sun_pointing reference at machine precision.
    assert ev.total_fluence_J_per_m2 == pytest.approx(
        ref_fluence, rel=0.0, abs=1e-9
    )
    assert ev.delta_a_km == pytest.approx(
        ref_delta_a, rel=0.0, abs=1e-6
    )
    assert ev.n_windows_kept == ref.metadata.n_windows_kept
    # This canonical grid point must produce nonzero fluence.
    assert ev.total_fluence_J_per_m2 > 1.0


@pytest.mark.slow
def test_harmonic_zero_amplitudes_matches_constant_cruise_through_pipeline():
    """At alpha_c = alpha_s = 0, the harmonic
    family produces an EvaluationResult bit-for-bit identical to the
    constant family at (β = α_0, φ_u = φ_u_rad). Tests the full factory
    + evaluate_cruise_general pipeline, not just the cruise closure.
    """
    cfg = _reference_config()
    beta = math.radians(5.0)
    phi_u = math.radians(120.0)

    ev_const = evaluate_cruise(beta, phi_u, cfg)

    x_harm = np.array([beta, 0.0, 0.0, phi_u])
    ev_harm = evaluate_cruise_general(x_harm, cfg, _harmonic_cruise_factory)

    # Bit-for-bit: same trajectory → same physics outputs.
    assert ev_harm.total_fluence_J_per_m2 == ev_const.total_fluence_J_per_m2
    assert ev_harm.delta_a_km == ev_const.delta_a_km
    assert ev_harm.delta_a_per_sol_km == ev_const.delta_a_per_sol_km
    assert ev_harm.e_max == ev_const.e_max
    assert ev_harm.inc_range_deg == ev_const.inc_range_deg
    assert ev_harm.n_windows_kept == ev_const.n_windows_kept
    assert ev_harm.n_windows_dropped == ev_const.n_windows_dropped
    assert ev_harm.converged == ev_const.converged

    # decision_vector_rad records the full parameterization.
    assert ev_harm.decision_vector_rad == (beta, 0.0, 0.0, phi_u)
    assert ev_const.decision_vector_rad == (beta, phi_u)

    # Scalar summary fields populate consistently.
    assert ev_harm.beta_rad == beta  # = alpha_0 for harmonic
    assert ev_harm.phi_u_rad == phi_u  # = x[-1] = phi_u
    assert ev_const.beta_rad == beta
    assert ev_const.phi_u_rad == phi_u


@pytest.mark.slow
def test_harmonic_full_zero_delta_amplitudes_matches_harmonic_alpha_through_pipeline():
    """At delta_c = delta_s = 0, the
    harmonic-(alpha, delta) family produces an EvaluationResult bit-for-bit
    identical to the harmonic-alpha family at the same
    (α_0, α_c, α_s, δ_0=phi_u). Tests the full factory +
    evaluate_cruise_general + propagator pipeline.
    """
    cfg = _reference_config()
    alpha_0 = math.radians(5.0)
    alpha_c = math.radians(2.0)
    alpha_s = math.radians(1.5)
    delta_0 = math.radians(120.0)

    x_harm = np.array([alpha_0, alpha_c, alpha_s, delta_0])
    ev_harm = evaluate_cruise_general(x_harm, cfg, _harmonic_cruise_factory)

    x_full = np.array([alpha_0, alpha_c, alpha_s, delta_0, 0.0, 0.0])
    ev_full = evaluate_cruise_general(x_full, cfg, _harmonic_full_cruise_factory)

    # Bit-for-bit: same trajectory → same physics outputs.
    assert ev_full.total_fluence_J_per_m2 == ev_harm.total_fluence_J_per_m2
    assert ev_full.delta_a_km == ev_harm.delta_a_km
    assert ev_full.delta_a_per_sol_km == ev_harm.delta_a_per_sol_km
    assert ev_full.e_max == ev_harm.e_max
    assert ev_full.inc_range_deg == ev_harm.inc_range_deg
    assert ev_full.n_windows_kept == ev_harm.n_windows_kept
    assert ev_full.n_windows_dropped == ev_harm.n_windows_dropped
    assert ev_full.converged == ev_harm.converged

    # decision_vector_rad records the full parameterization.
    assert ev_full.decision_vector_rad == (
        alpha_0, alpha_c, alpha_s, delta_0, 0.0, 0.0,
    )
    assert ev_harm.decision_vector_rad == (alpha_0, alpha_c, alpha_s, delta_0)

    # beta_rad equals alpha_0 for both families.
    assert ev_full.beta_rad == alpha_0
    assert ev_harm.beta_rad == alpha_0
    # phi_u_rad is delta_0 for harmonic and delta_s for harmonic-full.
    assert ev_harm.phi_u_rad == delta_0
    assert ev_full.phi_u_rad == 0.0  # = x[-1] = δ_s = 0


# ---------------------------------------------------------------------------
# Group 8: make_handoff_cruise_factory
# ---------------------------------------------------------------------------


class TestMakeHandoffCruiseFactory:
    def _handoff_config(self):
        """One-sol config at the 505 km ground-track-repeat altitude with
        sigma=0.018."""
        from reflectors.sun_sync import repeat_ground_track_altitude

        a, _, _ = repeat_ground_track_altitude(12)
        epoch_et = utc_to_et(EPOCH_UTC)
        sail = make_canonical_sail(0.018)
        propagate_kwargs = dict(
            gravity_degree=6, gravity_order=6,
            third_bodies=[
                sun_third_body(), phobos_third_body(), deimos_third_body(),
            ],
            altitude_floor=AltitudeFloor.at_km(300.0, label="altitude_floor"),
            options=PropagationOptions.fast(),
            t_eval_s=np.arange(0.0, DURATION_S + 0.1, 5.0),
        )
        find_windows_kwargs = dict(alpha_max_rad_s2=ALPHA_MAX_RAD_S2)
        return OrbitConfig(
            a_km=a, ltan_h=LTAN_H,
            M0_rad=math.radians(M0_DEG), epoch_et=epoch_et,
            duration_s=DURATION_S, sail=sail,
            target_lat_deg=TARGET_LAT_DEG, target_lon_deg=TARGET_LON_DEG,
            alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
            slew_duration_s=300.0,
            propagate_kwargs=propagate_kwargs,
            find_windows_kwargs=find_windows_kwargs,
        )

    def _build_cruise_old(self, config):
        from reflectors.cruise import sun_offset_harmonic_full_from_state

        return sun_offset_harmonic_full_from_state(
            alpha_0_rad=math.radians(15.0),
            alpha_c_rad=math.radians(-1.5),
            alpha_s_rad=math.radians(10.5),
            delta_0_rad=math.radians(284.0),
            delta_c_rad=math.radians(38.0),
            delta_s_rad=math.radians(-41.0),
            initial_state_km_kmps=config.initial_state_km_kmps,
        )

    def test_factory_returns_attitude_callable(self):
        from reflectors.dynamics import mars_gm_km3_per_s2

        cfg = self._handoff_config()
        cruise_old = self._build_cruise_old(cfg)
        wrapped = make_handoff_cruise_factory(
            _harmonic_full_cruise_factory, cruise_old,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
        )
        x = np.array([
            math.radians(15.5), math.radians(-1.0), math.radians(10.0),
            math.radians(285.0), math.radians(38.5), math.radians(-40.5),
        ])
        profile = wrapped(x, cfg)
        assert callable(profile)
        # Query during the slew: should produce a unit-norm vector.
        n = profile(cfg.initial_state_km_kmps[:3], float(cfg.epoch_et))
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-12)

    def test_factory_routes_to_slew_at_boundary_and_cruise_after(self):
        """At t = epoch_et, the wrapped factory's output equals
        cruise_old's evaluation (slew start). At t > t_b + T_slew (deep
        in the cruise segment), it equals cruise_new at that t."""
        from reflectors.dynamics import mars_gm_km3_per_s2

        cfg = self._handoff_config()
        cruise_old = self._build_cruise_old(cfg)
        wrapped = make_handoff_cruise_factory(
            _harmonic_full_cruise_factory, cruise_old,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            slew_floor_s=60.0,
        )
        x = np.array([
            math.radians(15.5), math.radians(-1.0), math.radians(10.0),
            math.radians(285.0), math.radians(38.5), math.radians(-40.5),
        ])
        profile = wrapped(x, cfg)
        # At t_start_et: should match cruise_old(state_b, t_b).
        r0 = cfg.initial_state_km_kmps[:3]
        n_at_start = profile(r0, float(cfg.epoch_et))
        n_old = cruise_old(r0, float(cfg.epoch_et))
        n_old = n_old / float(np.linalg.norm(n_old))
        np.testing.assert_allclose(n_at_start, n_old, atol=1e-12)

        # Deep in cruise segment (well past T_slew = 66 s default):
        # call at t = epoch + 600 s with the orbital position at that time.
        from reflectors.cruise import sun_offset_harmonic_full_from_state

        cruise_new = sun_offset_harmonic_full_from_state(
            alpha_0_rad=float(x[0]), alpha_c_rad=float(x[1]),
            alpha_s_rad=float(x[2]), delta_0_rad=float(x[3]),
            delta_c_rad=float(x[4]), delta_s_rad=float(x[5]),
            initial_state_km_kmps=cfg.initial_state_km_kmps,
        )
        # Shift r slightly to simulate orbital advance.
        r_late = cfg.initial_state_km_kmps[:3] + np.array([10.0, 0.0, 0.0])
        et_late = float(cfg.epoch_et) + 600.0
        n_at_late = profile(r_late, et_late)
        n_new_direct = cruise_new(r_late, et_late)
        n_new_direct = n_new_direct / float(np.linalg.norm(n_new_direct))
        np.testing.assert_allclose(n_at_late, n_new_direct, atol=1e-12)

    def test_factory_raises_when_slew_exceeds_sol(self):
        """If duration_s is shorter than the slew (artificial), helper
        raises ValueError rather than producing a degenerate piecewise."""
        from reflectors.dynamics import mars_gm_km3_per_s2

        # 30-second config; slew floor of 60 s won't fit.
        cfg = self._handoff_config()
        cfg = OrbitConfig(
            a_km=cfg.a_km, ltan_h=cfg.ltan_h, M0_rad=cfg.M0_rad,
            epoch_et=cfg.epoch_et, duration_s=30.0, sail=cfg.sail,
            target_lat_deg=cfg.target_lat_deg,
            target_lon_deg=cfg.target_lon_deg,
            alpha_max_rad_s2=cfg.alpha_max_rad_s2,
            slew_duration_s=cfg.slew_duration_s,
            propagate_kwargs=cfg.propagate_kwargs,
            find_windows_kwargs=cfg.find_windows_kwargs,
        )
        cruise_old = self._build_cruise_old(cfg)
        wrapped = make_handoff_cruise_factory(
            _harmonic_full_cruise_factory, cruise_old,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            slew_floor_s=60.0,
        )
        x = np.array([
            math.radians(15.5), math.radians(-1.0), math.radians(10.0),
            math.radians(285.0), math.radians(38.5), math.radians(-40.5),
        ])
        with pytest.raises(ValueError, match="slew end"):
            wrapped(x, cfg)


# ---------------------------------------------------------------------------
# optimize_cruise_polish_only tests
# ---------------------------------------------------------------------------
#
# Companion to optimize_cruise_general; runs scipy.optimize.minimize(
# method='L-BFGS-B', jac=parallel_fd_jacobian) for sols >= 2 in the
# polish-only warm chain pattern.
#
# These tests stay narrow on purpose: input validation paths +
# plumbing tests that exercise the optimizer loop with a monkeypatched
# evaluate_cruise_general to bypass the propagator. These are unit tests of the
# optimizer rather than convergence tests under the full propagator.


def _make_synth_eval_for_polish(x_target: np.ndarray):
    """Build a closure that returns synthetic EvaluationResults whose
    cost is sum((x - x_target)^2) under cost_metric=delta_a.

    Used to monkeypatch evaluate_cruise_general so the polish-only
    optimizer loop can be exercised without touching the propagator.
    Sets delta_a_per_sol_km = ||x - x_target||^2 (the optimization
    target under cost_metric=delta_a is |delta_a_per_sol|).

    Returns callable matching evaluate_cruise_general's signature
    ``(x, config, cruise_factory) -> EvaluationResult``.
    """
    target = np.asarray(x_target, dtype=float)

    def _synth(x, config, cruise_factory):
        x_arr = np.asarray(x, dtype=float).ravel()
        diff = x_arr - target
        cost_proxy = float(np.sum(diff * diff))
        return _synth_eval(
            delta_a_per_sol_km=cost_proxy,
            fluence=100.0,  # comfortably above any floor → penalty=0
            e_max=0.0,
            beta_rad=float(x_arr[0]) if x_arr.size else 0.0,
            phi_u_rad=float(x_arr[-1]) if x_arr.size else 0.0,
        )

    return _synth


class TestOptimizeCruisePolishOnlyInputValidation:
    """Input-validation contract mirrors
    optimize_cruise_general's; same code paths reused)."""

    def test_x0_empty_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="non-empty"):
            optimize_cruise_polish_only(
                cfg, x0=np.array([]),
                cruise_factory=_constant_cruise_factory,
                bounds=[],
                fluence_floor_J_per_m2=10.0,
            )

    def test_bounds_length_mismatch_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="bounds length"):
            optimize_cruise_polish_only(
                cfg, x0=np.array([0.0, 0.0, 0.0]),
                cruise_factory=_harmonic_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 1.0)],  # 2 bounds vs 3-D
                fluence_floor_J_per_m2=10.0,
            )

    def test_x0_outside_bounds_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match=r"outside bounds\[0\]"):
            optimize_cruise_polish_only(
                cfg, x0=np.array([2.0, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
            )

    def test_degenerate_bound_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match=r"bounds\[0\] must satisfy"):
            optimize_cruise_polish_only(
                cfg, x0=np.array([0.5, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.5, 0.5), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
            )

    def test_negative_penalty_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="penalty_lambda must be >= 0"):
            optimize_cruise_polish_only(
                cfg, x0=np.array([0.5, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
                penalty_lambda=-1.0,
            )

    def test_invalid_cost_metric_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="cost_metric must be one of"):
            optimize_cruise_polish_only(
                cfg, x0=np.array([0.5, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
                cost_metric="not-a-real-metric",
            )


class TestOptimizeCruisePolishOnlyConvergence:
    """With a synthetic quadratic cost and no propagator,
    L-BFGS-B + parallel FD jac should converge to the target.

    Tests run fast (<1 sec each) by monkeypatching
    evaluate_cruise_general to a quadratic synth — skips propagation.
    """

    def test_converges_to_target_2d(self, monkeypatch):
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        run = optimize_cruise_polish_only(
            cfg,
            x0=np.array([0.05, math.pi - 0.5]),  # off-target
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            fd_step_h=1e-4,
            jac_workers=None,  # serial path; parallel exercised in test_parallel.py
        )
        x_opt = np.asarray(run.scipy_result.x, dtype=float)
        np.testing.assert_allclose(x_opt, target, atol=1e-3)

    def test_history_is_populated(self, monkeypatch):
        """Polish-only path runs fun() in parent; history captures
        every iterate (no empty-history fallback needed)."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        run = optimize_cruise_polish_only(
            cfg,
            x0=np.array([0.05, math.pi - 0.5]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            jac_workers=None,
        )
        # L-BFGS-B at 2-D with parallel-FD jac runs fun() at each
        # iterate AND at each parallel jac-batch entry (since
        # jac_workers=None, the FD evals also go through cost_closure).
        # Evaluation history can therefore exceed N_iter; require a nonempty
        # trace.
        assert len(run.history) > 0
        # best_eval should be the lowest-cost entry in history.
        costs = [
            cruise_cost_from_eval(
                ev, run.fluence_floor_J_per_m2, run.penalty_lambda,
                cost_metric=run.cost_metric,
            )
            for ev in run.history
        ]
        assert run.best_eval is run.history[int(np.argmin(costs))]

    def test_returns_optimization_run_contract(self, monkeypatch):
        """OptimizationRun fields populated; matches
        optimize_cruise_general's contract."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        run = optimize_cruise_polish_only(
            cfg,
            x0=np.array([0.05, math.pi - 0.5]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            jac_workers=None,
        )
        # Same fields populated as optimize_cruise_general's return.
        assert run.scipy_result is not None
        assert isinstance(run.history, tuple)
        assert run.best_eval is not None
        assert run.baseline_eval is not None
        assert run.config is cfg
        assert run.fluence_floor_J_per_m2 == 10.0
        assert run.penalty_lambda == 1.0
        assert run.cost_metric == COST_METRIC_DELTA_A
        assert run.wall_total_s > 0.0

    def test_lbfgsb_options_threaded(self, monkeypatch):
        """Caller-supplied lbfgsb_options pass through to scipy."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        # maxiter=1 → optimizer runs at most one iteration; should
        # see scipy_result.nit <= 1 (or close).
        run = optimize_cruise_polish_only(
            cfg,
            x0=np.array([0.05, math.pi - 0.5]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            lbfgsb_options={"maxiter": 1, "ftol": 1e-3},
            jac_workers=None,
        )
        # The limited run may report zero or one iteration; convergence is
        # not required.
        assert hasattr(run.scipy_result, "nit")
        assert int(run.scipy_result.nit) <= 1

    def test_cumulative_kwargs_threaded(self, monkeypatch):
        """Cum_Δr / Δv kwargs flow to cost_closure → reach the
        rv_closure_weighted cum terms in the breakdown."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        # cost_metric=rv_closure_weighted exposes cum_Δr/Δv breakdown
        # entries even when their values are zero.
        run = optimize_cruise_polish_only(
            cfg,
            x0=np.array([0.05, math.pi - 0.5]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(1.0, 2.0, -3.0),
            cumulative_delta_v_iau_kmps=(0.001, -0.002, 0.003),
            cumulative_integral_weight_r=1.0,
            cumulative_integral_weight_v=1.0,
            scale_cum_delta_r_iau_mars_km=5.0,
            scale_cum_delta_v_iau_mars_kmps=0.005,
            lbfgsb_options={"maxiter": 1},
            jac_workers=None,
        )
        # Best-eval breakdown must show cum_delta_r_iau_mars (key from
        # the rv_closure path); value should be > 0 since cum_prior is
        # non-zero and synthetic Δr_iau on the eval is also (0,0,0) by
        # default → ||cum_total|| = ||cum_prior|| > 0.
        names = {name for name, _ in run.best_eval.cost_breakdown}
        assert "cum_delta_r_iau_mars" in names
        assert "cum_delta_v_iau_mars" in names
        for name, val in run.best_eval.cost_breakdown:
            if name == "cum_delta_r_iau_mars":
                assert val > 0.0
            if name == "cum_delta_v_iau_mars":
                assert val > 0.0


# ---------------------------------------------------------------------------
# optimize_cruise_de_with_parallel_polish (hybrid DE + parallel polish)
# ---------------------------------------------------------------------------


class TestOptimizeCruiseDeWithParallelPolish:
    """Wrapper that replaces scipy DE's serial L-BFGS-B
    polish with the parallel-FD-jac polish from
    ``optimize_cruise_polish_only``. Tests use the same monkeypatched
    synthetic-cost pattern as TestOptimizeCruisePolishOnlyConvergence to
    keep the propagator out of the loop.

    DE is run with tiny popsize/maxiter so each test stays under 1 sec
    even with the synthetic cost.
    """

    def test_polish_true_in_de_options_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="polish"):
            optimize_cruise_de_with_parallel_polish(
                cfg,
                x0=np.array([0.5, 0.0]),
                cruise_factory=_constant_cruise_factory,
                bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
                fluence_floor_J_per_m2=10.0,
                cost_metric=COST_METRIC_DELTA_A,
                de_options={"polish": True, "popsize": 3, "maxiter": 2,
                            "seed": 0},
            )

    def test_polish_false_in_de_options_accepted(self, monkeypatch):
        """Caller may pass polish=False explicitly. The wrapper still
        forces it internally; this just confirms no spurious rejection."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        run = optimize_cruise_de_with_parallel_polish(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            de_options={"polish": False, "popsize": 3, "maxiter": 2,
                        "seed": 0},
            lbfgsb_options={"maxiter": 5},
        )
        assert run.best_eval is not None

    def test_polish_below_de_only(self, monkeypatch):
        """End-to-end: polish phase always at-or-below DE-only's best
        cost (polish starts from DE's optimum, monotone descent)."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        # Reference: DE-only (no polish) using optimize_cruise_general.
        de_only = optimize_cruise_general(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            algorithm=ALGORITHM_DIFFERENTIAL_EVOLUTION,
            de_options={"polish": False, "popsize": 3, "maxiter": 2,
                        "seed": 0},
        )
        de_only_cost = cruise_cost_from_eval(
            de_only.best_eval, de_only.fluence_floor_J_per_m2,
            de_only.penalty_lambda, cost_metric=de_only.cost_metric,
        )
        # DE+parallel-polish wrapper at the same DE seed.
        hybrid = optimize_cruise_de_with_parallel_polish(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            de_options={"popsize": 3, "maxiter": 2, "seed": 0},
            lbfgsb_options={"maxiter": 20},
        )
        hybrid_cost = cruise_cost_from_eval(
            hybrid.best_eval, hybrid.fluence_floor_J_per_m2,
            hybrid.penalty_lambda, cost_metric=hybrid.cost_metric,
        )
        # Polish refines: hybrid <= DE-only on the synthetic quadratic.
        assert hybrid_cost <= de_only_cost + 1e-9

    def test_returns_optimization_run_contract(self, monkeypatch):
        """Combined OptimizationRun has all fields populated; matches
        optimize_cruise_general's contract."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        run = optimize_cruise_de_with_parallel_polish(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            de_options={"popsize": 3, "maxiter": 2, "seed": 0},
            lbfgsb_options={"maxiter": 5},
        )
        assert run.scipy_result is not None
        assert isinstance(run.history, tuple)
        assert run.best_eval is not None
        assert run.baseline_eval is not None
        assert run.config is cfg
        assert run.fluence_floor_J_per_m2 == 10.0
        assert run.penalty_lambda == 1.0
        assert run.cost_metric == COST_METRIC_DELTA_A
        assert run.wall_total_s > 0.0

    def test_history_concatenates_de_and_polish(self, monkeypatch):
        """Combined history >= polish history alone (DE history may be
        sparse under workers>1; under workers=1 it has at least the
        recovery-eval entry from optimize_cruise_general)."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        # Reference polish-only history length at the same DE optimum
        # would be hard to factor cleanly; instead, just verify that
        # combined.history is non-empty and includes at least one
        # polish-phase entry (best_eval is in there).
        run = optimize_cruise_de_with_parallel_polish(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            de_options={"popsize": 3, "maxiter": 2, "seed": 0},
            lbfgsb_options={"maxiter": 5},
        )
        assert len(run.history) > 0
        assert run.best_eval in run.history

    def test_cumulative_state_threaded_to_polish(self, monkeypatch):
        """cum_Δr/Δv kwargs reach polish phase: best_eval breakdown
        contains cum_delta_r_iau_mars / cum_delta_v_iau_mars terms with
        non-zero value when cum_prior is non-zero."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general",
            _make_synth_eval_for_polish(target),
        )
        run = optimize_cruise_de_with_parallel_polish(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            cumulative_delta_r_iau_km=(1.0, 2.0, -3.0),
            cumulative_delta_v_iau_kmps=(0.001, -0.002, 0.003),
            cumulative_integral_weight_r=1.0,
            cumulative_integral_weight_v=1.0,
            scale_cum_delta_r_iau_mars_km=5.0,
            scale_cum_delta_v_iau_mars_kmps=0.005,
            de_options={"popsize": 3, "maxiter": 2, "seed": 0},
            lbfgsb_options={"maxiter": 1},
        )
        names = {name for name, _ in run.best_eval.cost_breakdown}
        assert "cum_delta_r_iau_mars" in names
        assert "cum_delta_v_iau_mars" in names
        for name, val in run.best_eval.cost_breakdown:
            if name == "cum_delta_r_iau_mars":
                assert val > 0.0
            if name == "cum_delta_v_iau_mars":
                assert val > 0.0

    def test_baseline_computed_once(self, monkeypatch):
        """Baseline eval is computed at most once (DE phase computes it,
        polish phase reuses via baseline_eval=de_run.baseline_eval).
        Counts evaluate_cruise_general calls at x=0 (the default
        baseline_x)."""
        cfg = _reference_config()
        target = np.array([0.3, math.pi])
        baseline_call_count = {"n": 0}

        synth = _make_synth_eval_for_polish(target)

        def counting_synth(x, config, cruise_factory):
            x_arr = np.asarray(x, dtype=float).ravel()
            if np.allclose(x_arr, np.zeros_like(x_arr), atol=0.0):
                baseline_call_count["n"] += 1
            return synth(x, config, cruise_factory)

        monkeypatch.setattr(
            "reflectors.optimize.evaluate_cruise_general", counting_synth,
        )
        optimize_cruise_de_with_parallel_polish(
            cfg,
            x0=np.array([0.5, math.pi]),
            cruise_factory=_constant_cruise_factory,
            bounds=[(0.0, 1.0), (0.0, 2.0 * math.pi)],
            fluence_floor_J_per_m2=10.0,
            cost_metric=COST_METRIC_DELTA_A,
            de_options={"popsize": 3, "maxiter": 2, "seed": 0},
            lbfgsb_options={"maxiter": 1},
        )
        # DE phase computes baseline once at x=0; polish phase reuses
        # de_run.baseline_eval, so baseline call count should be 1.
        # (Could be 2 if polish recomputed; reusing matters for the
        # 60s-cadence run where each baseline costs ~3 sec.)
        assert baseline_call_count["n"] == 1


# ---------------------------------------------------------------------------
# Multi-target support: per-target floors, kept-only
# fluence, OrbitConfig.extra_targets, EvaluationResult fields.
# ---------------------------------------------------------------------------


def _mk_dw(
    t_start_s: float,
    t_end_s: float,
    *,
    fluence: float | None,
    target_idx: int = 0,
) -> DeliveryWindow:
    """Minimal DeliveryWindow for _kept_window_fluence arithmetic."""
    return DeliveryWindow(
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        et_start=t_start_s,
        et_end=t_end_s,
        duration_s=t_end_s - t_start_s,
        min_slant_range_km=500.0,
        max_elevation_deg=45.0,
        peak_alpha_demand_rad_s2=1.0e-5,
        integral_cos_alpha_s=10.0,
        n_samples=10,
        fluence_J_per_m2=fluence,
        target_idx=target_idx,
    )


def _mk_refined(windows, dropped_reasons=()):
    """Duck-typed RefinedSchedule carrying only the fields
    _kept_window_fluence reads."""
    meta = ScheduleMetadata(
        n_windows_kept=len(windows) - len(dropped_reasons),
        n_windows_dropped=len(dropped_reasons),
        dropped_window_reasons=tuple(dropped_reasons),
        segment_boundaries_et=(),
    )
    return SimpleNamespace(final_windows=tuple(windows), metadata=meta)


class TestKeptWindowFluence:
    def test_no_drops_sums_all(self):
        refined = _mk_refined([
            _mk_dw(0.0, 10.0, fluence=12.0, target_idx=0),
            _mk_dw(20.0, 30.0, fluence=8.0, target_idx=1),
            _mk_dw(40.0, 50.0, fluence=5.0, target_idx=0),
        ])
        total, by_target, n_by_target = _kept_window_fluence(refined, 2)
        assert total == pytest.approx(25.0)
        assert by_target == pytest.approx((17.0, 8.0))
        assert n_by_target == (2, 1)

    def test_dropped_window_fluence_excluded(self):
        """A schedule-dropped window's fluence is not credited because the
        sail cruises through it."""
        refined = _mk_refined(
            [
                _mk_dw(0.0, 10.0, fluence=12.0, target_idx=0),
                _mk_dw(20.0, 30.0, fluence=8.0, target_idx=1),
            ],
            dropped_reasons=[
                (1, "slew_in overlaps previous window's slew_out"),
            ],
        )
        total, by_target, n_by_target = _kept_window_fluence(refined, 2)
        assert total == pytest.approx(12.0)
        assert by_target == pytest.approx((12.0, 0.0))
        assert n_by_target == (1, 0)

    def test_none_fluence_counts_window_not_fluence(self):
        refined = _mk_refined([
            _mk_dw(0.0, 10.0, fluence=None, target_idx=0),
        ])
        total, by_target, n_by_target = _kept_window_fluence(refined, 1)
        assert total == 0.0
        assert by_target == (0.0,)
        assert n_by_target == (1,)

    def test_out_of_range_target_idx_raises(self):
        refined = _mk_refined([
            _mk_dw(0.0, 10.0, fluence=1.0, target_idx=1),
        ])
        with pytest.raises(RuntimeError, match="out of range"):
            _kept_window_fluence(refined, 1)


class TestPerTargetFluenceFloor:
    def test_none_floor_matches_scalar_path(self):
        """fluence_floor_by_target=None reproduces the scalar-floor
        cost exactly."""
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=10.0)
        cost_default, bd_default = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=15.0, penalty_lambda=2.0,
        )
        cost_none, bd_none = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=15.0, penalty_lambda=2.0,
            fluence_floor_by_target_J_per_m2=None,
        )
        assert cost_none == cost_default == pytest.approx(2.0 * 25.0)
        assert bd_none == bd_default

    def test_per_target_shortfalls_sum(self):
        """penalty = λ · Σ_t max(0, floor_t − F_t)²; scalar floor is
        ignored when by-target floors are given."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(20.0, 10.0),
        )
        cost, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=1.0e6, penalty_lambda=1.0,
            fluence_floor_by_target_J_per_m2=[25.0, 12.0],
        )
        # shortfalls: (5)² + (2)² = 29; scalar floor (1e6) unused.
        assert cost == pytest.approx(29.0)

    def test_no_penalty_when_both_above_floor(self):
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(20.0, 10.0),
        )
        cost, _ = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=1.0e6, penalty_lambda=1.0,
            fluence_floor_by_target_J_per_m2=[18.0, 9.0],
        )
        assert cost == 0.0

    def test_length_mismatch_raises(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=30.0)
        # _synth_eval default fluence_by_target=() vs 2 floors.
        with pytest.raises(ValueError, match="inconsistent"):
            cruise_cost_with_breakdown(
                ev, fluence_floor_J_per_m2=0.0, penalty_lambda=1.0,
                fluence_floor_by_target_J_per_m2=[25.0, 12.0],
            )

    def test_non_finite_floor_raises(self):
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(20.0, 10.0),
        )
        with pytest.raises(ValueError, match="finite"):
            cruise_cost_with_breakdown(
                ev, fluence_floor_J_per_m2=0.0, penalty_lambda=1.0,
                fluence_floor_by_target_J_per_m2=[25.0, float("nan")],
            )

    def test_breakdown_stays_seven_entries_with_summed_penalty(self):
        """rv_closure_weighted breakdown keeps the 7-entry contract;
        fluence_penalty is the SUM of per-target shortfall terms."""
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(20.0, 10.0),
        )
        _, breakdown = cruise_cost_with_breakdown(
            ev, fluence_floor_J_per_m2=0.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            fluence_floor_by_target_J_per_m2=[25.0, 12.0],
        )
        assert len(breakdown) == 8
        names = [k for k, _ in breakdown]
        assert names == [
            "delta_r_iau_mars",
            "delta_v_iau_mars",
            "e_max",
            "cum_delta_r_iau_mars",
            "cum_delta_v_iau_mars",
            "fluence_reward",
            "fluence_penalty",
            "delta_raan_vs_sunsync",
        ]
        penalty = dict(breakdown)["fluence_penalty"]
        assert penalty == pytest.approx(29.0)

    def test_reward_uses_total_fluence_not_split(self):
        """Two evals with the same TOTAL fluence but different
        per-target splits cost the same when no floor binds because the reward
        reads total fluence only."""
        kwargs = dict(
            fluence_floor_J_per_m2=0.0, penalty_lambda=1.0,
            cost_metric=COST_METRIC_RV_CLOSURE_WEIGHTED,
            multi_element_weights={"fluence": 1.0},
            fluence_floor_by_target_J_per_m2=[0.0, 0.0],
        )
        ev_a = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(20.0, 10.0),
        )
        ev_b = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(10.0, 20.0),
        )
        cost_a, _ = cruise_cost_with_breakdown(ev_a, **kwargs)
        cost_b, _ = cruise_cost_with_breakdown(ev_b, **kwargs)
        assert cost_a == cost_b
        assert cost_a < 0.0  # reward active


class TestOrbitConfigExtraTargets:
    def test_default_empty(self):
        cfg = _reference_config()
        assert cfg.extra_targets == ()

    def test_canonicalized_to_float_tuples(self):
        cfg = _reference_config()
        cfg2 = dc_replace(cfg, extra_targets=[[40, 315]])
        assert cfg2.extra_targets == ((40.0, 315.0),)
        assert isinstance(cfg2.extra_targets, tuple)
        assert all(
            isinstance(v, float)
            for pair in cfg2.extra_targets for v in pair
        )

    def test_non_finite_entry_raises(self):
        cfg = _reference_config()
        with pytest.raises(ValueError, match="finite"):
            dc_replace(cfg, extra_targets=[(40.0, float("inf"))])


class TestEvaluationResultMultiTargetDefaults:
    def test_defaults_empty_tuples(self):
        ev = _synth_eval(delta_a_per_sol_km=0.0, fluence=10.0)
        assert ev.fluence_by_target_J_per_m2 == ()
        assert ev.n_windows_by_target == ()

    def test_fields_populated_when_given(self):
        ev = _synth_eval(
            delta_a_per_sol_km=0.0, fluence=30.0,
            fluence_by_target=(20.0, 10.0),
            n_windows_by_target=(2, 1),
        )
        assert ev.fluence_by_target_J_per_m2 == (20.0, 10.0)
        assert ev.n_windows_by_target == (2, 1)
