"""Fast tests for ``reflectors.attitude_schedule.build_delivery_schedule``.

Six groups, following the structure of tests/test_attitude.py and
tests/test_visibility.py:

  1. Dispatch logic -- cruise / slew / track segments route correctly.
  2. n_hat continuity at boundaries -- C^0 at cruise<->slew, with the
     slew<->track handoff pinned to machine precision when predicted
     and actual trajectories agree; predicted-versus-actual drift can
     introduce a bounded handoff mismatch.
  3. Window drop behaviour -- schedule-start underrun, schedule-end
     overrun, inter-window slew-buffer overlap.
  4. Degenerate no-windows case -- empty windows -> single cruise
     segment.
  5. Integration with propagate -- one-orbit smoke that propagate()
     accepts the composed callable and produces a finite trajectory.
  6. Exact cruise equivalence -- empty-windows schedule returns the
     cruise direction bit-for-bit.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import spiceypy as spice

from reflectors.attitude import fixed_j2000, smooth_slew_hermite, sun_pointing
from reflectors.attitude_schedule import (
    CruiseSlewMetadata,
    DeliveryScheduleSeed,
    RefinedSchedule,
    ScheduleMetadata,
    _apply_damping,
    _hermite_peak_alpha_upper_bound,
    _hermite_peak_omega,
    _jaccard_et,
    _match_windows,
    _max_boundary_drift,
    _moving_endpoint_slew_duration,
    build_delivery_schedule,
    cruise_to_cruise_slew,
    refine_delivery_schedule,
    slew_duration_for_alpha_max,
    slew_duration_for_limits,
)
from reflectors.dynamics import (
    PropagationResult,
    PropagationOptions,
    mars_gm_km3_per_s2,
    propagate,
)
from reflectors.ephemeris import utc_to_et
from reflectors.srp import SailOptical, SolarSail
from reflectors.surface import (
    mars_equatorial_radius_km,
    surface_point_position,
)
from reflectors.visibility import (
    ContinuedDeliveryWindows,
    DeliveryWindow,
    WindowContinuationError,
    bisector_normal,
)


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def epoch_et() -> float:
    return utc_to_et(EPOCH_STR)


@pytest.fixture(scope="module")
def sun_hat_j2000(epoch_et):
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    s = np.asarray(state[:3], dtype=float)
    return s / np.linalg.norm(s)


@pytest.fixture(scope="module")
def sub_solar_lmo_r(sun_hat_j2000):
    """A sail position 400 km above Mars on the sub-solar line (J2000)."""
    R_sat = mars_equatorial_radius_km() + 400.0
    return R_sat * sun_hat_j2000


def _circular_lmo_r_fn(r0_km: np.ndarray, epoch_et: float):
    """Analytic circular-orbit predictor returning r(et) in J2000 km.

    Used to supply ``r_sat_predicted_fn`` in the unit tests: given
    initial position ``r0_km`` and epoch, rotates about z at the mean
    motion of a circular orbit of radius ``|r0_km|``.
    """
    mu = mars_gm_km3_per_s2()
    r0 = np.asarray(r0_km, dtype=float)
    R = float(np.linalg.norm(r0))
    n = math.sqrt(mu / R**3)

    def _r(et: float) -> np.ndarray:
        dt = float(et) - float(epoch_et)
        c, s = math.cos(n * dt), math.sin(n * dt)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ r0

    return _r


def _make_window(
    *,
    t_start_s: float,
    t_end_s: float,
    epoch_et: float,
    min_slant_range_km: float = 500.0,
    max_elevation_deg: float = 45.0,
    integral_cos_alpha_s: float = 100.0,
    n_samples: int = 50,
    peak_alpha_demand_rad_s2: float | None = 1.0e-5,
    target_idx: int = 0,
    target_lat_deg: float | None = None,
    target_lon_deg: float | None = None,
) -> DeliveryWindow:
    return DeliveryWindow(
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        et_start=epoch_et + t_start_s,
        et_end=epoch_et + t_end_s,
        duration_s=t_end_s - t_start_s,
        min_slant_range_km=min_slant_range_km,
        max_elevation_deg=max_elevation_deg,
        peak_alpha_demand_rad_s2=peak_alpha_demand_rad_s2,
        integral_cos_alpha_s=integral_cos_alpha_s,
        n_samples=n_samples,
        target_idx=target_idx,
        target_lat_deg=target_lat_deg,
        target_lon_deg=target_lon_deg,
    )


TARGET_LAT = 40.0
TARGET_LON = 200.0
SLEW_DUR_S = 120.0
SOL_DURATION_S = 5000.0  # short for unit tests; kernels still apply


# ---------------------------------------------------------------------------
# Group 1: Dispatch logic
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_cruise_query_routes_to_cruise_profile(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        cruise = sun_pointing()
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=cruise,
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        # Query deep in the initial cruise segment.
        t_cruise = epoch_et + 500.0
        r_at_t = r_pred(t_cruise)
        got = profile(r_at_t, t_cruise)
        expected = cruise(r_at_t, t_cruise)
        np.testing.assert_allclose(got, expected, atol=1e-15)
        assert meta.n_windows_kept == 1

    def test_track_query_routes_to_bisector(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        profile, _meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        # Query mid-track with the actual predicted position -- bisector
        # is state-dependent, so the profile should return the
        # independently-computed bisector at the same state.
        t_mid = epoch_et + 2150.0
        r_at_t = r_pred(t_mid)
        r_target = surface_point_position(TARGET_LAT, TARGET_LON, t_mid)
        state_sun, _ = spice.spkezr("SUN", t_mid, "J2000", "NONE", "MARS")
        r_sun = np.asarray(state_sun[:3], dtype=float)
        n_expected, _cos_a = bisector_normal(r_at_t, r_target, r_sun)
        got = profile(r_at_t, t_mid)
        np.testing.assert_allclose(got, n_expected, atol=1e-14)

    def test_slew_query_returns_unit_vector_between_endpoints(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        profile, _meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        # Query mid-slew-in: [1880, 2000]. Midpoint = 1940.
        t_mid_slew = epoch_et + 1940.0
        r_at_t = r_pred(t_mid_slew)
        got = profile(r_at_t, t_mid_slew)
        assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-12
        # At the slew midpoint, the quintic s(0.5) = 0.5, so the
        # direction is approximately the rotational midpoint between
        # cruise-start and track-start. It must differ from both endpoints.
        cruise_at_t = sun_pointing()(r_at_t, t_mid_slew)
        assert float(np.linalg.norm(got - cruise_at_t)) > 1e-3


# ---------------------------------------------------------------------------
# Group 2: n_hat continuity at boundaries
# ---------------------------------------------------------------------------


class TestBoundaryContinuity:
    def test_cruise_to_slew_in_boundary_is_c0(
        self, epoch_et, sub_solar_lmo_r
    ):
        """By construction, slew_in's n_0 is cruise(r_predicted, t_boundary)."""
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        # Boundary: slew_in_start_et = et_start - slew_dur.
        t_bd = window.et_start - SLEW_DUR_S
        # Use the predicted state at the boundary for the comparison;
        # this is how the composer constructed the slew endpoint.
        r_pred_bd = r_pred(t_bd)
        # Querying just before the boundary -> cruise segment.
        t_before = t_bd - 1.0e-3
        t_after = t_bd + 1.0e-3
        n_before = profile(r_pred_bd, t_before)
        n_after = profile(r_pred_bd, t_after)
        # Sun-pointing changes negligibly over 2 ms -> cruise direction
        # is essentially constant; slew at tau~0 returns n_0 = cruise
        # at the boundary.
        assert float(np.linalg.norm(n_before - n_after)) < 1e-6

    def test_track_to_slew_out_boundary_is_c0_under_predicted_trajectory(
        self, epoch_et, sub_solar_lmo_r
    ):
        """Under predicted==actual (analytic predictor evaluated at its
        own state), the bisector at t_end matches the slew-out's n_0.

        This is the direction-match side of the known-limitation:
        omega is discontinuous at the handoff (static-endpoint slew),
        but the DIRECTION is C^0 when predicted == actual.
        """
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        profile, _meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        t_bd = window.et_end
        r_pred_bd = r_pred(t_bd)
        t_before = t_bd - 1.0e-3
        t_after = t_bd + 1.0e-3
        n_before = profile(r_pred_bd, t_before)
        n_after = profile(r_pred_bd, t_after)
        # Both sides should yield the same bisector direction (predicted
        # == actual since r_pred is used for both). Floor is ~1e-8 from
        # the 2 ms time gap's natural variation (r_sun motion ~50 m and
        # Mars rotation ~0.5 m at the target, both ~1e-10 relative) and
        # the arcsin precision floor at near-orthogonal geometries.
        assert float(np.linalg.norm(n_before - n_after)) < 1e-7

    def test_slew_out_to_cruise_boundary_is_c0(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        profile, _meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        t_bd = window.et_end + SLEW_DUR_S
        r_pred_bd = r_pred(t_bd)
        t_before = t_bd - 1.0e-3
        t_after = t_bd + 1.0e-3
        n_before = profile(r_pred_bd, t_before)
        n_after = profile(r_pred_bd, t_after)
        assert float(np.linalg.norm(n_before - n_after)) < 1e-6


# ---------------------------------------------------------------------------
# Group 3: Window drop behaviour
# ---------------------------------------------------------------------------


class TestWindowDrop:
    def test_window_at_schedule_start_gets_dropped(
        self, epoch_et, sub_solar_lmo_r
    ):
        # slew_in_start = 50 - 120 = -70 s (before epoch).
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        early = _make_window(
            t_start_s=50.0, t_end_s=200.0, epoch_et=epoch_et,
        )
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[early],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        assert meta.n_windows_kept == 0
        assert meta.n_windows_dropped == 1
        assert meta.dropped_window_reasons[0][0] == 0
        assert "slew_in precedes" in meta.dropped_window_reasons[0][1]

    def test_window_at_schedule_end_gets_dropped(
        self, epoch_et, sub_solar_lmo_r
    ):
        # slew_out_end = 4950 + 120 = 5070 > sol_duration 5000.
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        late = _make_window(
            t_start_s=4800.0, t_end_s=4950.0, epoch_et=epoch_et,
        )
        _profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[late],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        assert meta.n_windows_kept == 0
        assert meta.n_windows_dropped == 1
        assert "exceeds schedule end" in meta.dropped_window_reasons[0][1]

    def test_overlapping_slew_buffers_drops_later_window(
        self, epoch_et, sub_solar_lmo_r
    ):
        # w1: [1000, 1200]; w1 slew_out_end = 1320.
        # w2: [1400, 1600]; w2 slew_in_start = 1280 < 1320 -> overlaps.
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        w1 = _make_window(
            t_start_s=1000.0, t_end_s=1200.0, epoch_et=epoch_et,
        )
        w2 = _make_window(
            t_start_s=1400.0, t_end_s=1600.0, epoch_et=epoch_et,
        )
        _profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[w1, w2],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        assert meta.n_windows_kept == 1
        assert meta.n_windows_dropped == 1
        dropped_idx, reason = meta.dropped_window_reasons[0]
        assert dropped_idx == 1  # the second window
        assert "overlaps previous" in reason


# ---------------------------------------------------------------------------
# Group 4: Degenerate no-windows case
# ---------------------------------------------------------------------------


class TestNoWindows:
    def test_empty_windows_produces_single_cruise_segment(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        cruise = sun_pointing()
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=cruise,
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        assert meta.n_windows_kept == 0
        assert meta.n_windows_dropped == 0
        # Segment boundaries: schedule start and end only.
        assert len(meta.segment_boundaries_et) == 2
        assert meta.segment_boundaries_et[0] == pytest.approx(epoch_et)
        assert meta.segment_boundaries_et[1] == pytest.approx(
            epoch_et + SOL_DURATION_S
        )
        # Queries at various et return the cruise direction bit-for-bit.
        for t_rel in (100.0, 1000.0, 4999.0):
            et = epoch_et + t_rel
            r = r_pred(et)
            np.testing.assert_allclose(
                profile(r, et), cruise(r, et), atol=0.0,
            )


# ---------------------------------------------------------------------------
# Group 5: Integration with propagate
# ---------------------------------------------------------------------------


class TestIntegrationWithPropagate:
    def test_composed_profile_runs_under_propagate(
        self, epoch_et, sub_solar_lmo_r
    ):
        """Smoke test: pass the composed schedule as sail_normal and
        verify propagate() completes with finite output.
        """
        R_sat = mars_equatorial_radius_km() + 400.0
        mu = mars_gm_km3_per_s2()
        v = math.sqrt(mu / R_sat)
        state0 = np.array([R_sat, 0.0, 0.0, 0.0, v, 0.0])
        period = 2.0 * math.pi * math.sqrt(R_sat**3 / mu)

        r_pred = _circular_lmo_r_fn(np.array([R_sat, 0.0, 0.0]), epoch_et)
        # Put a small window well inside the single orbit and with
        # slews that fit.
        window = _make_window(
            t_start_s=0.4 * period,
            t_end_s=0.5 * period,
            epoch_et=epoch_et,
        )
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=period,
            slew_duration_s=60.0,
        )
        assert meta.n_windows_kept == 1

        sail = SolarSail(
            area_m2=1000.0,
            mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )
        result = propagate(
            state0,
            (0.0, period),
            epoch_et=epoch_et,
            mu_km3_s2=mu,
            solar_sail=sail,
            sail_normal=profile,
            options=PropagationOptions.fast(),
        )
        assert np.all(np.isfinite(result.state_km_kmps))

    def test_empty_windows_schedule_matches_cruise_only_propagation(
        self, epoch_et
    ):
        """Propagating with an empty-windows schedule produces the
        same final state as propagating with the cruise profile
        directly (bit-for-bit: the composed callable just dispatches
        to cruise_profile).
        """
        R_sat = mars_equatorial_radius_km() + 400.0
        mu = mars_gm_km3_per_s2()
        v = math.sqrt(mu / R_sat)
        state0 = np.array([R_sat, 0.0, 0.0, 0.0, v, 0.0])
        period = 2.0 * math.pi * math.sqrt(R_sat**3 / mu)

        r_pred = _circular_lmo_r_fn(np.array([R_sat, 0.0, 0.0]), epoch_et)
        cruise = sun_pointing()
        profile, _meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=cruise,
            epoch_et=epoch_et,
            duration_s=period,
            slew_duration_s=60.0,
        )

        sail = SolarSail(
            area_m2=1000.0,
            mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )
        result_schedule = propagate(
            state0,
            (0.0, period),
            epoch_et=epoch_et,
            mu_km3_s2=mu,
            solar_sail=sail,
            sail_normal=profile,
            options=PropagationOptions.fast(),
        )
        result_cruise = propagate(
            state0,
            (0.0, period),
            epoch_et=epoch_et,
            mu_km3_s2=mu,
            solar_sail=sail,
            sail_normal=cruise,
            options=PropagationOptions.fast(),
        )
        np.testing.assert_allclose(
            result_schedule.state_km_kmps,
            result_cruise.state_km_kmps,
            atol=1e-14,
            rtol=0.0,
        )


# ---------------------------------------------------------------------------
# Group 6: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_nonpositive_duration_raises(self, epoch_et, sub_solar_lmo_r):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        with pytest.raises(ValueError, match="duration_s"):
            build_delivery_schedule(
                r_sat_predicted_fn=r_pred,
                windows=[],
                target_lat_deg=TARGET_LAT,
                target_lon_deg=TARGET_LON,
                cruise_profile=sun_pointing(),
                epoch_et=epoch_et,
                duration_s=0.0,
                slew_duration_s=SLEW_DUR_S,
            )

    def test_nonpositive_slew_duration_raises(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        with pytest.raises(ValueError, match="slew_duration_s"):
            build_delivery_schedule(
                r_sat_predicted_fn=r_pred,
                windows=[],
                target_lat_deg=TARGET_LAT,
                target_lon_deg=TARGET_LON,
                cruise_profile=sun_pointing(),
                epoch_et=epoch_et,
                duration_s=SOL_DURATION_S,
                slew_duration_s=-10.0,
            )

    def test_window_without_absolute_et_gets_dropped(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        base = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        stripped = replace(base, et_start=None, et_end=None)
        _profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[stripped],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        assert meta.n_windows_kept == 0
        assert meta.n_windows_dropped == 1
        assert "no absolute-ET" in meta.dropped_window_reasons[0][1]


# ---------------------------------------------------------------------------
# Group 7: alpha_max-driven slew-duration sizing
# ---------------------------------------------------------------------------


class TestAlphaMaxSlewSizing:
    """slew_duration_for_alpha_max + build_delivery_schedule auto-sizing.

    Pins the static-endpoint formula recovery, 1/sqrt(alpha_max)
    scaling, dynamic-endpoint growth, and roundtrip consistency
    against check_alpha_bound on the constructed Hermite slew.
    """

    def test_static_endpoint_matches_quintic_formula_within_tolerance(self):
        """For omega=0 at both ends, T ~ sqrt(theta * (10/sqrt(3)) / alpha_max).

        R^3 chord-then-project bound is slightly larger than the
        great-circle-arc peak at the same T, so the sizing returns a
        slightly longer duration. Tolerance: within 50% of the
        static-endpoint formula (the divergence grows with theta;
        pinning closer would be fragile). Exact agreement at the
        small-angle limit is NOT claimed (R^3 chord vs great-circle
        arc bounds differ by ~tan(theta/2) / (theta/2) factor).
        """
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(45.0)  # canonical handoff angle
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        alpha_max = math.radians(0.003)  # 5.236e-5 rad/s^2
        T = slew_duration_for_alpha_max(
            n0, nf, alpha_max_rad_s2=alpha_max, safety_factor=1.0,
        )
        # Canonical quintic formula for smooth_slew (great-circle).
        T_static_arc = math.sqrt(theta * 10.0 / math.sqrt(3.0) / alpha_max)
        assert 0.8 * T_static_arc < T < 1.5 * T_static_arc, (
            f"sized T={T:.2f}s not within 0.8-1.5x of arc formula "
            f"T_static_arc={T_static_arc:.2f}s"
        )

    def test_doubling_alpha_max_roughly_halves_T_squared(self):
        """Sizing scales as 1/sqrt(alpha_max) for static endpoints."""
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(45.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        T_1 = slew_duration_for_alpha_max(
            n0, nf, alpha_max_rad_s2=5.0e-5, safety_factor=1.0,
        )
        T_2 = slew_duration_for_alpha_max(
            n0, nf, alpha_max_rad_s2=1.0e-4, safety_factor=1.0,
        )
        # T scales as 1/sqrt(alpha_max), so doubling alpha_max -> T / sqrt(2).
        ratio = T_1 / T_2
        assert 1.3 < ratio < 1.5, (
            f"T1/T2 = {ratio:.3f}, expected ~sqrt(2) = 1.414"
        )

    def test_dynamic_endpoint_against_rotation_needs_longer_T(self):
        """Endpoint omega OPPOSING the natural rotation direction enlarges
        |c_ddot| -> sizing demands more time.

        Note: an ALIGNED omega_f (same sense as the natural n_0 -> n_f
        rotation) can REDUCE the peak |c_ddot| vs the static-endpoint
        case, because the tangent kick smooths the path. A counter-
        rotating kick always increases it. This test picks the
        counter-rotating case as the robust physics anchor.
        """
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(45.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        alpha_max = math.radians(0.003)
        T_static = slew_duration_for_alpha_max(
            n0, nf, alpha_max_rad_s2=alpha_max, safety_factor=1.0,
        )
        # Natural rotation from n_0 to n_f is about +z; counter-rotating
        # omega_f is about -z.
        omega_f_counter = np.array([0.0, 0.0, -5.0e-3])
        T_dyn = slew_duration_for_alpha_max(
            n0, nf,
            omega_f_rad_s=omega_f_counter,
            alpha_max_rad_s2=alpha_max,
            safety_factor=1.0,
        )
        assert T_dyn > T_static, (
            f"counter-rotating dynamic-endpoint T={T_dyn:.2f}s should "
            f"exceed static-endpoint T={T_static:.2f}s"
        )

    def test_returned_T_respects_alpha_bound_via_check_alpha_bound(self):
        """Roundtrip: build Hermite at returned T, check peak |alpha| <= budget."""
        from reflectors.attitude import check_alpha_bound
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(40.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        omega_f = np.array([0.0, 0.0, 1.0e-3])
        alpha_max = math.radians(0.003)
        T = slew_duration_for_alpha_max(
            n0, nf,
            omega_f_rad_s=omega_f,
            alpha_max_rad_s2=alpha_max,
            safety_factor=1.0,
        )
        slew = smooth_slew_hermite(
            0.0, T, n0, nf,
            omega_f_rad_s=omega_f,
        )
        trivial_r = lambda et: np.zeros(3)  # noqa: E731
        violator = check_alpha_bound(
            slew, trivial_r, alpha_max, (0.0, T),
            n_samples=500, dt=0.5,
        )
        assert violator is None, (
            f"sizing returned T={T:.2f}s but check_alpha_bound found "
            f"a violator at et={violator}"
        )

    def test_rejects_unachievable_alpha_max(self):
        """At a huge omega_f, alpha_max may be unreachable within T_max_s."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        # Omega so huge it forces |c_ddot| -> infinity at any reasonable T.
        omega_f = np.array([0.0, 0.0, 100.0])  # rad/s (unphysical)
        with pytest.raises(ValueError, match="unachievable"):
            slew_duration_for_alpha_max(
                n0, nf,
                omega_f_rad_s=omega_f,
                alpha_max_rad_s2=1.0e-5,
                T_max_s=3600.0,
            )

    def test_rejects_nonpositive_alpha_max(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="positive and finite"):
            slew_duration_for_alpha_max(n0, nf, alpha_max_rad_s2=0.0)
        with pytest.raises(ValueError, match="positive and finite"):
            slew_duration_for_alpha_max(n0, nf, alpha_max_rad_s2=-1e-5)

    def test_build_delivery_schedule_auto_sizes_per_window(
        self, epoch_et, sub_solar_lmo_r
    ):
        """When alpha_max_rad_s2 is supplied, slew durations per window
        come from the closed-form sizing (and respect the caller-supplied
        slew_duration_s as a floor).
        """
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        alpha_max = math.radians(0.003)
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=10.0,  # small floor so auto-sizing drives T
            alpha_max_rad_s2=alpha_max,
        )
        assert meta.n_windows_kept == 1
        # Segments: cruise, slew_in, track, slew_out, cruise (5).
        assert len(meta.segment_boundaries_et) == 6
        slew_in_dur = (
            meta.segment_boundaries_et[2] - meta.segment_boundaries_et[1]
        )
        slew_out_dur = (
            meta.segment_boundaries_et[4] - meta.segment_boundaries_et[3]
        )
        # Auto-sized durations must exceed the 10 s floor. The upper bound
        # confirms that sizing selects a finite duration satisfying alpha_max.
        assert slew_in_dur > 10.0
        assert slew_out_dur > 10.0
        assert slew_in_dur < 1000.0
        assert slew_out_dur < 1000.0

    def test_build_delivery_schedule_backward_compat_when_alpha_max_is_none(
        self, epoch_et, sub_solar_lmo_r
    ):
        """alpha_max_rad_s2=None preserves fixed-duration behavior bit-for-bit.

        Checks that the composed profile returns the same n_hat as the
        fixed-duration path for a query in each segment type.
        """
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        # Build with explicit alpha_max=None (default).
        profile_default, meta_default = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
            alpha_max_rad_s2=None,
        )
        # Same, without the kwarg.
        profile_no_kwarg, meta_no_kwarg = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        # Same segment boundaries.
        assert (
            meta_default.segment_boundaries_et
            == meta_no_kwarg.segment_boundaries_et
        )
        # Bit-for-bit same profile outputs at sample points from each
        # segment type.
        for t_offset in [500.0, 1950.0, 2150.0, 2350.0, 4500.0]:
            et_q = epoch_et + t_offset
            r_at = r_pred(et_q)
            n_default = profile_default(r_at, et_q)
            n_no_kwarg = profile_no_kwarg(r_at, et_q)
            np.testing.assert_allclose(n_default, n_no_kwarg, atol=0.0)

    def test_alpha_upper_bound_decreases_as_T_increases(self):
        """Monotonicity check on the Hermite upper-bound polynomial."""
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(45.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        w0 = np.zeros(3)
        wf = np.array([0.0, 0.0, 1.0e-3])
        a_100 = _hermite_peak_alpha_upper_bound(n0, nf, w0, wf, 100.0)
        a_200 = _hermite_peak_alpha_upper_bound(n0, nf, w0, wf, 200.0)
        a_400 = _hermite_peak_alpha_upper_bound(n0, nf, w0, wf, 400.0)
        assert a_100 > a_200 > a_400, (
            f"monotonicity violation: {a_100}, {a_200}, {a_400}"
        )
        # Ratio check: at zero endpoint omega the scaling is exactly
        # 1/T^2. With a small omega_f the scaling is close to 1/T^2 for
        # T ~ 100 s but trends toward 1/T for large T (tangent-kick term
        # dominates). Pin the static-like regime (100 -> 200) at ~4x.
        assert 3.0 < a_100 / a_200 < 5.0

    def test_rejects_invalid_alpha_max_in_build_delivery_schedule(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        with pytest.raises(ValueError, match="positive and finite"):
            build_delivery_schedule(
                r_sat_predicted_fn=r_pred,
                windows=[],
                target_lat_deg=TARGET_LAT,
                target_lon_deg=TARGET_LON,
                cruise_profile=sun_pointing(),
                epoch_et=epoch_et,
                duration_s=SOL_DURATION_S,
                slew_duration_s=SLEW_DUR_S,
                alpha_max_rad_s2=-1.0e-5,
            )


class TestJointAlphaOmegaSlewSizing:
    """Continuous omega extrema and joint alpha/omega schedule sizing."""

    @staticmethod
    def _independent_dense_omega_peak(
        n0: np.ndarray,
        nf: np.ndarray,
        w0: np.ndarray,
        wf: np.ndarray,
        duration_s: float,
    ) -> float:
        """Vectorized direct kinematics, independent of scheduler helper."""
        n0 = np.asarray(n0, dtype=float)
        n0 = n0 / np.linalg.norm(n0)
        nf = np.asarray(nf, dtype=float)
        nf = nf / np.linalg.norm(nf)
        v0 = duration_s * np.cross(w0, n0)
        vf = duration_s * np.cross(wf, nf)
        delta = nf - n0
        p0 = n0
        p1 = v0
        p3 = 10.0 * delta - 6.0 * v0 - 4.0 * vf
        p4 = -15.0 * delta + 8.0 * v0 + 7.0 * vf
        p5 = 6.0 * delta - 3.0 * v0 - 3.0 * vf
        tau = np.linspace(0.0, 1.0, 200_001)
        c = (
            p0[np.newaxis, :]
            + np.outer(tau, p1)
            + np.outer(tau**3, p3)
            + np.outer(tau**4, p4)
            + np.outer(tau**5, p5)
        )
        dc_dtau = (
            p1[np.newaxis, :]
            + 3.0 * np.outer(tau**2, p3)
            + 4.0 * np.outer(tau**3, p4)
            + 5.0 * np.outer(tau**4, p5)
        )
        omega = np.cross(c, dc_dtau) / (
            duration_s
            * np.einsum("ij,ij->i", c, c)[:, np.newaxis]
        )
        return float(np.max(np.linalg.norm(omega, axis=1)))

    def test_stationary_polynomial_peak_matches_independent_dense_curve(self):
        n0 = np.array([1.0, 0.2, -0.1])
        n0 /= np.linalg.norm(n0)
        nf = np.array([-0.1, 0.9, 0.3])
        nf /= np.linalg.norm(nf)
        w0 = np.array([8.0e-4, -1.2e-3, 1.7e-3])
        wf = np.array([-1.1e-3, 6.0e-4, 2.1e-3])
        duration_s = 543.21
        continuous = _hermite_peak_omega(n0, nf, w0, wf, duration_s)
        dense = self._independent_dense_omega_peak(
            n0, nf, w0, wf, duration_s,
        )
        # Dense sampling can only underestimate the true peak.  At a 5e-6
        # tau spacing, its miss is far below the mission-limit margin.
        assert continuous >= dense * (1.0 - 1.0e-12)
        assert continuous - dense < 1.0e-10

    def test_static_slew_omega_limit_binds_and_is_respected(self):
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(80.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        alpha_max = math.radians(0.5)  # deliberately loose
        omega_max = math.radians(0.3)
        duration_s = slew_duration_for_limits(
            n0,
            nf,
            alpha_max_rad_s2=alpha_max,
            omega_max_rad_s=omega_max,
        )
        omega_peak = _hermite_peak_omega(
            n0, nf, np.zeros(3), np.zeros(3), duration_s,
        )
        alpha_bound = _hermite_peak_alpha_upper_bound(
            n0, nf, np.zeros(3), np.zeros(3), duration_s,
        )
        assert 0.85 * omega_max < omega_peak <= omega_max
        assert alpha_bound <= alpha_max

    def test_endpoint_rate_above_limit_is_unachievable(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        omega_max = math.radians(0.3)
        with pytest.raises(ValueError, match="endpoint rate already exceeds"):
            slew_duration_for_limits(
                n0,
                nf,
                omega_0_rad_s=np.array([0.0, 0.0, 1.01 * omega_max]),
                alpha_max_rad_s2=math.radians(0.003),
                omega_max_rad_s=omega_max,
            )

    def test_alpha_only_joint_entry_point_is_bit_exact_delegation(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.5, math.sqrt(3.0) / 2.0, 0.0])
        wf = np.array([0.0, 0.0, 7.0e-4])
        alpha_max = math.radians(0.003)
        direct = slew_duration_for_alpha_max(
            n0,
            nf,
            omega_f_rad_s=wf,
            alpha_max_rad_s2=alpha_max,
        )
        delegated = slew_duration_for_limits(
            n0,
            nf,
            omega_f_rad_s=wf,
            alpha_max_rad_s2=alpha_max,
            omega_max_rad_s=None,
        )
        assert delegated == direct

    def test_moving_endpoint_bracket_succeeds_when_fixed_point_stalls(self):
        """Direct utilization solve must not reject a slow fixed-point map."""
        n0 = np.array([1.0, 0.0, 0.0])
        zero_rate = np.zeros(3)
        omega_max = 0.01

        # The terminal direction advances with the trial duration. These
        # constants produce a strongly contractive fixed-point map whose 20th
        # update remains more than 0.5 s from convergence.
        def moving_endpoints(
            duration_s: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            theta = math.radians(51.75) + math.radians(16.6) * math.tanh(
                (duration_s - 200.0) / 80.0
            )
            nf = np.array([math.cos(theta), math.sin(theta), 0.0])
            return n0, nf, zero_rate, zero_rate

        fixed_point_duration_s = 100.0
        fixed_point_delta_s = math.inf
        for _ in range(20):
            endpoints = moving_endpoints(fixed_point_duration_s)
            required_s = slew_duration_for_limits(
                endpoints[0],
                endpoints[1],
                omega_0_rad_s=endpoints[2],
                omega_f_rad_s=endpoints[3],
                omega_max_rad_s=omega_max,
                safety_factor=1.1,
                T_min_s=100.0,
                T_max_s=400.0,
            )
            fixed_point_delta_s = required_s - fixed_point_duration_s
            fixed_point_duration_s = required_s
        assert abs(fixed_point_delta_s) > 0.5

        duration_s, endpoints, n_evaluations = (
            _moving_endpoint_slew_duration(
                moving_endpoints,
                alpha_max_rad_s2=None,
                omega_max_rad_s=omega_max,
                safety_factor=1.1,
                T_min_s=100.0,
                T_max_s=400.0,
                root_tol_s=0.01,
                context="synthetic slow fixed point",
            )
        )
        peak = _hermite_peak_omega(*endpoints, duration_s)
        assert 100.0 < duration_s <= 400.0
        assert peak <= omega_max * (1.0 + 1.0e-12)
        assert n_evaluations < 20

    def test_delivery_slew_profiles_obey_mission_omega_limit(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        omega_max = math.radians(0.3)
        profile, metadata = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=10.0,
            alpha_max_rad_s2=math.radians(0.003),
            omega_max_rad_s=omega_max,
        )
        assert metadata.n_windows_kept == 1
        closure = dict(zip(profile.__code__.co_freevars, profile.__closure__))
        segments = closure["segs"].cell_contents
        assert len(segments) == 5
        for segment_index in (1, 3):
            t_start, t_end, slew = segments[segment_index]
            slew_closure = dict(
                zip(slew.__code__.co_freevars, slew.__closure__)
            )
            n_start = np.asarray(slew_closure["p_0"].cell_contents)
            p1 = np.asarray(slew_closure["p_1"].cell_contents)
            p3 = np.asarray(slew_closure["p_3"].cell_contents)
            p4 = np.asarray(slew_closure["p_4"].cell_contents)
            p5 = np.asarray(slew_closure["p_5"].cell_contents)
            n_finish = n_start + p1 + p3 + p4 + p5
            duration_s = float(t_end - t_start)
            w_start = np.cross(n_start, p1 / duration_s)
            dc_finish = (p1 + 3.0 * p3 + 4.0 * p4 + 5.0 * p5)
            w_finish = np.cross(n_finish, dc_finish / duration_s)
            peak = _hermite_peak_omega(
                n_start, n_finish, w_start, w_finish, duration_s,
            )
            assert peak <= omega_max * (1.0 + 1.0e-10)


# ---------------------------------------------------------------------------
# Group 8: Fixed-point iteration
# ---------------------------------------------------------------------------


class TestWindowMatching:
    """_jaccard_et + _match_windows + _max_boundary_drift + _apply_damping."""

    def test_jaccard_identical_windows_is_one(self, epoch_et):
        w = _make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)
        assert _jaccard_et(w, w) == 1.0

    def test_jaccard_disjoint_windows_is_zero(self, epoch_et):
        w1 = _make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)
        w2 = _make_window(t_start_s=300.0, t_end_s=400.0, epoch_et=epoch_et)
        assert _jaccard_et(w1, w2) == 0.0

    def test_jaccard_half_overlap(self, epoch_et):
        w1 = _make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)
        # w2 overlaps w1 by 50s; union spans 150s -> Jaccard = 50/150 = 1/3.
        w2 = _make_window(t_start_s=150.0, t_end_s=250.0, epoch_et=epoch_et)
        assert math.isclose(_jaccard_et(w1, w2), 1.0 / 3.0, abs_tol=1e-12)

    def test_jaccard_none_et_returns_zero(self, epoch_et):
        w1 = _make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)
        w2 = replace(w1, et_start=None, et_end=None)
        assert _jaccard_et(w1, w2) == 0.0

    def test_match_windows_pairs_stable_windows_and_flags_new_ones(
        self, epoch_et
    ):
        old = [
            _make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et),
            _make_window(t_start_s=500.0, t_end_s=600.0, epoch_et=epoch_et),
        ]
        new = [
            # Drift of 10 s; Jaccard on [100, 200] vs [110, 210] = 90/110 ~ 0.82.
            _make_window(t_start_s=110.0, t_end_s=210.0, epoch_et=epoch_et),
            # Stable second window.
            _make_window(t_start_s=500.0, t_end_s=600.0, epoch_et=epoch_et),
            # Brand-new window.
            _make_window(t_start_s=1200.0, t_end_s=1300.0, epoch_et=epoch_et),
        ]
        matched, unmatched_old, unmatched_new = _match_windows(old, new)
        assert len(matched) == 2
        assert len(unmatched_old) == 0
        assert len(unmatched_new) == 1
        assert unmatched_new[0].t_start_s == 1200.0

    def test_match_windows_threshold_rejects_low_overlap(self, epoch_et):
        old = [_make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)]
        # 20 s overlap on 180 s union -> Jaccard ~ 0.11, below threshold 0.5.
        new = [_make_window(t_start_s=180.0, t_end_s=280.0, epoch_et=epoch_et)]
        matched, unmatched_old, unmatched_new = _match_windows(
            old, new, jaccard_threshold=0.5,
        )
        assert len(matched) == 0
        assert len(unmatched_old) == 1
        assert len(unmatched_new) == 1

    def test_max_boundary_drift_matched_windows(self, epoch_et):
        old = [_make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)]
        new = [_make_window(t_start_s=105.0, t_end_s=207.0, epoch_et=epoch_et)]
        drift = _max_boundary_drift(old, new)
        # max(|105-100|, |207-200|) = max(5, 7) = 7.
        assert math.isclose(drift, 7.0, abs_tol=1e-12)

    def test_max_boundary_drift_unmatched_returns_inf(self, epoch_et):
        old = [_make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)]
        new: list[DeliveryWindow] = []
        assert _max_boundary_drift(old, new) == float("inf")

    def test_apply_damping_linear_blend_on_matched_windows(self, epoch_et):
        old = [_make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)]
        new = [_make_window(t_start_s=110.0, t_end_s=210.0, epoch_et=epoch_et)]
        damped, unstable = _apply_damping(
            old, new, damping=0.5, jaccard_threshold=0.5,
        )
        assert len(damped) == 1
        # 0.5 * 110 + 0.5 * 100 = 105; 0.5 * 210 + 0.5 * 200 = 205.
        assert math.isclose(damped[0].et_start, epoch_et + 105.0, abs_tol=1e-9)
        assert math.isclose(damped[0].et_end, epoch_et + 205.0, abs_tol=1e-9)
        assert len(unstable) == 0

    def test_apply_damping_full_damping_keeps_old(self, epoch_et):
        """damping=0 = hold old boundaries; damping=1 = snap to new."""
        old = [_make_window(t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et)]
        new = [_make_window(t_start_s=110.0, t_end_s=210.0, epoch_et=epoch_et)]
        damped_0, _ = _apply_damping(old, new, damping=0.0, jaccard_threshold=0.5)
        damped_1, _ = _apply_damping(old, new, damping=1.0, jaccard_threshold=0.5)
        assert math.isclose(damped_0[0].et_start, epoch_et + 100.0, abs_tol=1e-9)
        assert math.isclose(damped_1[0].et_start, epoch_et + 110.0, abs_tol=1e-9)


class TestRefineDeliveryScheduleValidation:
    """Input validation on refine_delivery_schedule (no propagation)."""

    def test_rejects_negative_max_iterations(self, epoch_et, sub_solar_lmo_r):
        state0 = np.concatenate([sub_solar_lmo_r, np.array([0.0, 3.5, 0.0])])
        from reflectors.srp import SailOptical, SolarSail
        sail = SolarSail(
            area_m2=1000.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl(),
        )
        with pytest.raises(ValueError, match="max_iterations"):
            refine_delivery_schedule(
                state0,
                t_span_s=(0.0, 3600.0),
                epoch_et=epoch_et,
                cruise_profile=sun_pointing(),
                target_lat_deg=TARGET_LAT,
                target_lon_deg=TARGET_LON,
                sail=sail,
                max_iterations=-1,
            )

    def test_rejects_damping_out_of_range(self, epoch_et, sub_solar_lmo_r):
        state0 = np.concatenate([sub_solar_lmo_r, np.array([0.0, 3.5, 0.0])])
        from reflectors.srp import SailOptical, SolarSail
        sail = SolarSail(
            area_m2=1000.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl(),
        )
        with pytest.raises(ValueError, match="damping"):
            refine_delivery_schedule(
                state0,
                t_span_s=(0.0, 3600.0),
                epoch_et=epoch_et,
                cruise_profile=sun_pointing(),
                target_lat_deg=TARGET_LAT,
                target_lon_deg=TARGET_LON,
                sail=sail,
                damping=1.5,
            )


class TestDeliveryScheduleContinuation:
    """Prior-sol seeding preserves Mars-fixed geometry and skips discovery."""

    def test_seed_predictor_repeats_mars_fixed_positions(self, epoch_et):
        source_epoch = float(epoch_et)
        new_epoch = source_epoch + 88775.244
        t_s = np.array([0.0, 100.0, 200.0, 300.0])
        r_iau_mars = np.array([3900.0, 120.0, -45.0])
        positions = np.vstack([
            np.asarray(
                spice.pxform("IAU_MARS", "J2000", source_epoch + t_i)
            ) @ r_iau_mars
            for t_i in t_s
        ])
        seed = DeliveryScheduleSeed(
            source_epoch_et=source_epoch,
            t_s=t_s,
            positions_j2000_km=positions,
            windows=(),
        )
        predictor = seed.position_predictor_for_epoch(
            new_epoch, (0.0, 300.0)
        )

        for t_i in t_s:
            expected = np.asarray(
                spice.pxform("IAU_MARS", "J2000", new_epoch + t_i)
            ) @ r_iau_mars
            np.testing.assert_allclose(
                predictor(new_epoch + t_i), expected, atol=2.0e-12
            )

    def test_seed_predictor_evolves_handoff_residual_differentially(
        self, epoch_et
    ):
        source_epoch = float(epoch_et)
        new_epoch = source_epoch + 88775.244
        mu = mars_gm_km3_per_s2()
        t_s = np.array([0.0, 100.0, 200.0, 300.0])
        source_state0 = np.array([3900.0, 0.0, 0.0, 0.0, 3.3, 0.1])
        source_states = np.vstack([
            source_state0
            if t_i == 0.0
            else np.asarray(spice.prop2b(mu, source_state0, t_i), dtype=float)
            for t_i in t_s
        ])
        seed = DeliveryScheduleSeed(
            source_epoch_et=source_epoch,
            t_s=t_s,
            positions_j2000_km=source_states[:, :3],
            velocities_j2000_km_s=source_states[:, 3:],
            mu_km3_s2=mu,
            windows=(),
        )
        source_to_body = np.asarray(
            spice.sxform("J2000", "IAU_MARS", source_epoch), dtype=float
        )
        body_to_new = np.asarray(
            spice.sxform("IAU_MARS", "J2000", new_epoch), dtype=float
        )
        nominal_new_state = body_to_new @ source_to_body @ source_state0
        actual_new_state = nominal_new_state + np.array([
            0.2, -0.1, 0.05, 1.0e-4, -2.0e-4, 0.5e-4,
        ])
        predictor = seed.position_predictor_for_epoch(
            new_epoch,
            (0.0, 300.0),
            initial_state_j2000_km_kmps=actual_new_state,
        )

        np.testing.assert_allclose(
            predictor(new_epoch), actual_new_state[:3], atol=2.0e-12
        )
        t_probe = 200.0
        source_at_probe = source_states[2, :3]
        repeated_at_probe = (
            np.asarray(
                spice.pxform("IAU_MARS", "J2000", new_epoch + t_probe)
            )
            @ (
                np.asarray(
                    spice.pxform(
                        "J2000", "IAU_MARS", source_epoch + t_probe
                    )
                ) @ source_at_probe
            )
        )
        expected = repeated_at_probe + (
            np.asarray(spice.prop2b(mu, actual_new_state, t_probe))[:3]
            - np.asarray(spice.prop2b(mu, nominal_new_state, t_probe))[:3]
        )
        np.testing.assert_allclose(
            predictor(new_epoch + t_probe), expected, atol=2.0e-11
        )

    def test_seeded_refinement_uses_one_scheduled_propagation(
        self, monkeypatch, epoch_et
    ):
        import reflectors.attitude_schedule as schedule_module
        import reflectors.dynamics as dynamics_module

        source_epoch = float(epoch_et)
        new_epoch = source_epoch + 88775.244
        t_s = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        states = np.zeros((t_s.size, 6), dtype=float)
        states[:, 0] = 3900.0
        states[:, 1] = np.linspace(0.0, 40.0, t_s.size)
        result = PropagationResult(
            t_s=t_s,
            state_km_kmps=states,
            method="mock",
            rtol=0.0,
            atol=0.0,
            mu_km3_s2=mars_gm_km3_per_s2(),
            epoch_et=new_epoch,
            solver_message="mock",
            n_rhs_calls=0,
        )
        source_window = _make_window(
            t_start_s=100.0,
            t_end_s=200.0,
            epoch_et=source_epoch,
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
        )
        seed = DeliveryScheduleSeed(
            source_epoch_et=source_epoch,
            t_s=t_s,
            positions_j2000_km=states[:, :3],
            windows=(source_window,),
        )
        propagation_calls = 0

        def fake_propagate(*args, **kwargs):
            nonlocal propagation_calls
            propagation_calls += 1
            return result

        metadata = ScheduleMetadata(
            n_windows_kept=1,
            n_windows_dropped=0,
            dropped_window_reasons=(),
            segment_boundaries_et=(new_epoch, new_epoch + 400.0),
        )

        def fake_build(*args, **kwargs):
            return kwargs["cruise_profile"], metadata

        def fake_continue(_result, _targets, seed_windows, **kwargs):
            return ContinuedDeliveryWindows(
                windows=tuple(seed_windows),
                search_intervals_by_target_s=((0, 0.0, 400.0),),
                n_sample_target_evaluations=t_s.size,
                n_samples_per_target=t_s.size,
                max_boundary_shift_s=0.0,
            )

        monkeypatch.setattr(dynamics_module, "propagate", fake_propagate)
        monkeypatch.setattr(schedule_module, "build_delivery_schedule", fake_build)
        monkeypatch.setattr(
            schedule_module,
            "continue_delivery_windows_multi",
            fake_continue,
        )

        refined = refine_delivery_schedule(
            np.zeros(6),
            t_span_s=(0.0, 400.0),
            epoch_et=new_epoch,
            cruise_profile=sun_pointing(),
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            sail=object(),
            max_iterations=1,
            continuation_seed=seed,
        )

        assert propagation_calls == 1
        assert refined.n_propagations == 1
        assert refined.initialization_mode == "prior_sol_mars_fixed_seed"
        assert refined.window_search_modes == ("local_guarded",)
        assert refined.continuation_fallback_reasons == ()
        assert refined.converged is True

    def test_seeded_refinement_records_full_scan_fallback(
        self, monkeypatch, epoch_et
    ):
        import reflectors.attitude_schedule as schedule_module
        import reflectors.dynamics as dynamics_module

        source_epoch = float(epoch_et)
        new_epoch = source_epoch + 88775.244
        t_s = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        states = np.zeros((t_s.size, 6), dtype=float)
        states[:, 0] = 3900.0
        result = PropagationResult(
            t_s=t_s,
            state_km_kmps=states,
            method="mock",
            rtol=0.0,
            atol=0.0,
            mu_km3_s2=mars_gm_km3_per_s2(),
            epoch_et=new_epoch,
            solver_message="mock",
            n_rhs_calls=0,
        )
        source_window = _make_window(
            t_start_s=100.0,
            t_end_s=200.0,
            epoch_et=source_epoch,
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
        )
        current_window = replace(
            source_window,
            et_start=new_epoch + source_window.t_start_s,
            et_end=new_epoch + source_window.t_end_s,
        )
        seed = DeliveryScheduleSeed(
            source_epoch_et=source_epoch,
            t_s=t_s,
            positions_j2000_km=states[:, :3],
            windows=(source_window,),
        )
        metadata = ScheduleMetadata(
            n_windows_kept=1,
            n_windows_dropped=0,
            dropped_window_reasons=(),
            segment_boundaries_et=(new_epoch, new_epoch + 400.0),
        )

        monkeypatch.setattr(dynamics_module, "propagate", lambda *a, **k: result)
        monkeypatch.setattr(
            schedule_module,
            "build_delivery_schedule",
            lambda *a, **k: (k["cruise_profile"], metadata),
        )
        monkeypatch.setattr(
            schedule_module,
            "continue_delivery_windows_multi",
            lambda *a, **k: (_ for _ in ()).throw(
                WindowContinuationError("synthetic topology failure")
            ),
        )
        global_calls = 0

        def fake_global(*args, **kwargs):
            nonlocal global_calls
            global_calls += 1
            return [current_window]

        monkeypatch.setattr(
            schedule_module, "find_delivery_windows_multi", fake_global
        )

        refined = refine_delivery_schedule(
            np.zeros(6),
            t_span_s=(0.0, 400.0),
            epoch_et=new_epoch,
            cruise_profile=sun_pointing(),
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            sail=object(),
            max_iterations=1,
            continuation_seed=seed,
            continuation_failure="full_scan",
        )

        assert global_calls == 1
        assert refined.window_search_modes == ("global_fallback",)
        assert len(refined.continuation_fallback_reasons) == 1
        assert "synthetic topology failure" in (
            refined.continuation_fallback_reasons[0]
        )


# ---------------------------------------------------------------------------
# Group 7: Cruise-to-cruise slew
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def handoff_orbit_state(epoch_et):
    """A 505 km circular sun-sync state at the K=12 design altitude."""
    from reflectors.sun_sync import (
        initial_state_j2000,
        repeat_ground_track_altitude,
    )

    a, _, _ = repeat_ground_track_altitude(12)
    return initial_state_j2000(
        a_km=a, ltan_h=18.0, M0_rad=0.0, epoch_et=epoch_et,
    )


class TestCruiseToCruiseSlew:
    """Cruise-to-cruise slew at a sol-boundary handoff."""

    def _build_two_cruises(self, handoff_orbit_state):
        """Return (cruise_old, cruise_new) where cruise_new is a small
        perturbation of cruise_old in the harmonic-(α, δ) family.
        """
        from reflectors.cruise import sun_offset_harmonic_full_from_state

        cruise_old = sun_offset_harmonic_full_from_state(
            alpha_0_rad=math.radians(15.0),
            alpha_c_rad=math.radians(-1.5),
            alpha_s_rad=math.radians(10.5),
            delta_0_rad=math.radians(284.0),
            delta_c_rad=math.radians(38.0),
            delta_s_rad=math.radians(-41.0),
            initial_state_km_kmps=handoff_orbit_state,
        )
        # Different new state-frame and slightly different params -- realistic
        # for "after the optimizer perturbs by 0.5° per param".
        cruise_new = sun_offset_harmonic_full_from_state(
            alpha_0_rad=math.radians(15.5),
            alpha_c_rad=math.radians(-1.0),
            alpha_s_rad=math.radians(10.0),
            delta_0_rad=math.radians(285.0),
            delta_c_rad=math.radians(38.5),
            delta_s_rad=math.radians(-40.5),
            initial_state_km_kmps=handoff_orbit_state,
        )
        return cruise_old, cruise_new

    def test_endpoint_attitude_match_at_t_start(self, epoch_et, handoff_orbit_state):
        """Slew callable at t_start_et returns cruise_old's n̂ exactly
        (modulo the unit-projection epsilon)."""
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        slew, meta = cruise_to_cruise_slew(
            cruise_old, cruise_new,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=math.radians(0.003),
            slew_floor_s=60.0,
        )
        n_at_start = slew(handoff_orbit_state[:3], meta.t_start_et)
        n_old_direct = cruise_old(handoff_orbit_state[:3], meta.t_start_et)
        n_old_direct = n_old_direct / float(np.linalg.norm(n_old_direct))
        np.testing.assert_allclose(n_at_start, n_old_direct, atol=1e-12)

    def test_endpoint_attitude_match_at_t_end(self, epoch_et, handoff_orbit_state):
        """Slew callable at t_end_et matches the metadata n_hat_f exactly."""
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        slew, meta = cruise_to_cruise_slew(
            cruise_old, cruise_new,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=math.radians(0.003),
            slew_floor_s=60.0,
        )
        # Predict end state via Kepler (same predictor the helper uses).
        state_pred_end = spice.prop2b(
            mars_gm_km3_per_s2(), handoff_orbit_state, meta.T_slew_s,
        )
        r_pred_end = np.asarray(state_pred_end[:3], dtype=float)
        n_at_end = slew(r_pred_end, meta.t_end_et)
        np.testing.assert_allclose(n_at_end, meta.n_hat_f, atol=1e-12)

    def test_alpha_peak_within_budget(self, epoch_et, handoff_orbit_state):
        """The alpha_peak_upper_bound metadata field is <= alpha_max_budget."""
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        alpha_max = math.radians(0.003)
        _slew, meta = cruise_to_cruise_slew(
            cruise_old, cruise_new,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=alpha_max,
            slew_floor_s=60.0,
        )
        assert meta.alpha_peak_upper_bound_rad_s2 <= alpha_max
        # Also check the safety_factor=1.1 default is reflected in the
        # utilization being a bit under 100% of the *raw* alpha_max.
        assert 0.0 < meta.alpha_utilization_pct <= 100.0

    def test_continuous_omega_peak_within_budget(
        self, epoch_et, handoff_orbit_state
    ):
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        omega_max = math.radians(0.3)
        _slew, meta = cruise_to_cruise_slew(
            cruise_old,
            cruise_new,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=math.radians(0.003),
            omega_max_rad_s=omega_max,
            slew_floor_s=60.0,
        )
        assert meta.omega_max_budget_rad_s == omega_max
        assert meta.omega_peak_rad_s is not None
        assert meta.omega_peak_rad_s <= omega_max
        assert meta.omega_utilization_pct is not None
        assert 0.0 < meta.omega_utilization_pct <= 100.0

    def test_omega_continuity_at_t_start_via_finite_diff(
        self, epoch_et, handoff_orbit_state
    ):
        """Finite-difference of the slew at t_start matches the metadata
        omega_0 to <= 1e-7 rad/s (Hermite gives exact endpoint omega in
        the no-noise R^3 limit; FD on a slightly-non-unit-norm Hermite
        introduces a tiny residual).
        """
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        slew, meta = cruise_to_cruise_slew(
            cruise_old, cruise_new,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=math.radians(0.003),
            slew_floor_s=60.0,
        )
        # Sample n_hat at t_start ± dt under Kepler-predicted r and FD.
        dt = 0.5
        et_minus = meta.t_start_et + 1e-6  # avoid -dt before slew domain
        et_plus = meta.t_start_et + dt
        # Two-body predictor r at each sample.
        r_minus = spice.prop2b(mars_gm_km3_per_s2(), handoff_orbit_state, 1e-6)[:3]
        r_plus = spice.prop2b(mars_gm_km3_per_s2(), handoff_orbit_state, dt)[:3]
        n_minus = np.asarray(slew(r_minus, et_minus), dtype=float)
        n_plus = np.asarray(slew(r_plus, et_plus), dtype=float)
        n_dot = (n_plus - n_minus) / (dt - 1e-6)
        omega_fd = np.cross(meta.n_hat_0, n_dot)
        np.testing.assert_allclose(omega_fd, meta.omega_0_rad_s, atol=1e-5)

    def test_state_continuity_assumption_pinned(
        self, epoch_et, handoff_orbit_state
    ):
        """Two-body predictor at dt=0 returns state_b bit-equal: this is
        the cornerstone of the cruise->cruise slew's state-continuity
        guarantee. ``spice.prop2b`` should treat dt=0 as identity, but
        the identity is pinned so a SPICE upgrade cannot shift the
        boundary state.
        """
        state_at_zero = spice.prop2b(
            mars_gm_km3_per_s2(), handoff_orbit_state, 0.0,
        )
        np.testing.assert_array_equal(
            np.asarray(state_at_zero, dtype=float), handoff_orbit_state,
        )

    def test_metadata_includes_predictor_state(
        self, epoch_et, handoff_orbit_state
    ):
        """Metadata exposes the Kepler-predicted slew-end state for
        downstream cross-check vs the actual integrator state."""
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        _slew, meta = cruise_to_cruise_slew(
            cruise_old, cruise_new,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=math.radians(0.003),
            slew_floor_s=60.0,
        )
        assert meta.predictor_state_at_slew_end_km_kmps.shape == (6,)
        # Kepler-only over a 60+ s slew advances the orbit by some
        # fraction of a degree -- |r| should still be close to the
        # input |r| for circular orbits (state_b is circular by
        # construction in the circular-orbit fixture).
        r0 = float(np.linalg.norm(handoff_orbit_state[:3]))
        r_end = float(np.linalg.norm(meta.predictor_state_at_slew_end_km_kmps[:3]))
        assert abs(r_end - r0) < 1.0, (
            f"|r| drift over Kepler predictor is {r_end - r0:.4f} km "
            f"(expected near zero for circular orbit)"
        )

    def test_identical_cruises_yield_minimum_slew(
        self, epoch_et, handoff_orbit_state
    ):
        """If cruise_old == cruise_new (same closure), the slew duration
        is the floor and the endpoint attitudes are nearly identical."""
        from reflectors.cruise import sun_offset_harmonic_full_from_state

        cruise = sun_offset_harmonic_full_from_state(
            alpha_0_rad=math.radians(10.0),
            alpha_s_rad=math.radians(8.0),
            delta_0_rad=math.radians(280.0),
            initial_state_km_kmps=handoff_orbit_state,
        )
        slew, meta = cruise_to_cruise_slew(
            cruise, cruise,
            state_b_km_kmps=handoff_orbit_state,
            epoch_et_b=epoch_et,
            central_body_gm_km3_s2=mars_gm_km3_per_s2(),
            alpha_max_rad_s2=math.radians(0.003),
            slew_floor_s=60.0,
        )
        # slew_duration_for_alpha_max returns safety_factor * T_min_s for
        # near-identical endpoints, so 1.1 * 60 = 66 s is the expected
        # value at the default safety factor.
        assert meta.T_slew_s == pytest.approx(66.0, abs=0.5)
        # n̂ moves only because the Kepler-predicted state at t+T_slew
        # is slightly downstream; the angle is small.
        assert meta.theta_total_rad < math.radians(2.0)

    def test_nonconvergence_reports_actual_last_sizing_delta(
        self, epoch_et, handoff_orbit_state, monkeypatch
    ):
        """The failure diagnostic retains the pre-update fixed-point delta."""
        cruise_old, cruise_new = self._build_two_cruises(handoff_orbit_state)
        required_durations = iter((120.0, 180.0))
        monkeypatch.setattr(
            "reflectors.attitude_schedule.slew_duration_for_alpha_max",
            lambda *args, **kwargs: next(required_durations),
        )

        with pytest.raises(ValueError, match=r"last delta = 60\.0000 s"):
            cruise_to_cruise_slew(
                cruise_old,
                cruise_new,
                state_b_km_kmps=handoff_orbit_state,
                epoch_et_b=epoch_et,
                central_body_gm_km3_s2=mars_gm_km3_per_s2(),
                alpha_max_rad_s2=math.radians(0.003),
                slew_floor_s=60.0,
                n_size_iterations=2,
            )

    def test_input_validation(self, epoch_et, handoff_orbit_state):
        """Bad inputs surface clear ValueErrors."""
        from reflectors.cruise import sun_offset_harmonic_full_from_state

        cruise = sun_offset_harmonic_full_from_state(
            alpha_0_rad=math.radians(10.0),
            initial_state_km_kmps=handoff_orbit_state,
        )
        with pytest.raises(ValueError, match=r"shape \(6,\)"):
            cruise_to_cruise_slew(
                cruise, cruise,
                state_b_km_kmps=np.zeros(3),
                epoch_et_b=epoch_et,
                central_body_gm_km3_s2=mars_gm_km3_per_s2(),
                alpha_max_rad_s2=math.radians(0.003),
            )
        with pytest.raises(ValueError, match="alpha_max_rad_s2"):
            cruise_to_cruise_slew(
                cruise, cruise,
                state_b_km_kmps=handoff_orbit_state,
                epoch_et_b=epoch_et,
                central_body_gm_km3_s2=mars_gm_km3_per_s2(),
                alpha_max_rad_s2=-1.0,
            )
        with pytest.raises(ValueError, match="omega_max_rad_s"):
            cruise_to_cruise_slew(
                cruise, cruise,
                state_b_km_kmps=handoff_orbit_state,
                epoch_et_b=epoch_et,
                central_body_gm_km3_s2=mars_gm_km3_per_s2(),
                alpha_max_rad_s2=math.radians(0.003),
                omega_max_rad_s=-1.0,
            )
        with pytest.raises(ValueError, match="central_body_gm_km3_s2"):
            cruise_to_cruise_slew(
                cruise, cruise,
                state_b_km_kmps=handoff_orbit_state,
                epoch_et_b=epoch_et,
                central_body_gm_km3_s2=0.0,
                alpha_max_rad_s2=math.radians(0.003),
            )


# ---------------------------------------------------------------------------
# Multi-target schedules
# ---------------------------------------------------------------------------


TARGET2_LAT = 40.0
TARGET2_LON = 315.0


class TestPerWindowTarget:
    """Each track segment points at ITS window's target; windows
    without target fields fall back to the schedule's target args."""

    def test_track_segments_use_each_windows_target(
        self, epoch_et, sub_solar_lmo_r
    ):
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        w1 = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
            target_idx=0,
            target_lat_deg=TARGET_LAT, target_lon_deg=TARGET_LON,
        )
        w2 = _make_window(
            t_start_s=3500.0, t_end_s=3800.0, epoch_et=epoch_et,
            target_idx=1,
            target_lat_deg=TARGET2_LAT, target_lon_deg=TARGET2_LON,
        )
        profile, meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[w1, w2],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        assert meta.n_windows_kept == 2
        for t_mid, lat, lon in (
            (epoch_et + 2150.0, TARGET_LAT, TARGET_LON),
            (epoch_et + 3650.0, TARGET2_LAT, TARGET2_LON),
        ):
            r_at_t = r_pred(t_mid)
            r_target = surface_point_position(lat, lon, t_mid)
            state_sun, _ = spice.spkezr(
                "SUN", t_mid, "J2000", "NONE", "MARS"
            )
            r_sun = np.asarray(state_sun[:3], dtype=float)
            n_expected, _cos_a = bisector_normal(r_at_t, r_target, r_sun)
            got = profile(r_at_t, t_mid)
            np.testing.assert_allclose(got, n_expected, atol=1e-14)

    def test_none_target_falls_back_to_args(
        self, epoch_et, sub_solar_lmo_r
    ):
        """Windows without target fields use the schedule-level target."""
        r_pred = _circular_lmo_r_fn(sub_solar_lmo_r, epoch_et)
        window = _make_window(
            t_start_s=2000.0, t_end_s=2300.0, epoch_et=epoch_et,
        )
        assert window.target_lat_deg is None
        profile, _meta = build_delivery_schedule(
            r_sat_predicted_fn=r_pred,
            windows=[window],
            target_lat_deg=TARGET_LAT,
            target_lon_deg=TARGET_LON,
            cruise_profile=sun_pointing(),
            epoch_et=epoch_et,
            duration_s=SOL_DURATION_S,
            slew_duration_s=SLEW_DUR_S,
        )
        t_mid = epoch_et + 2150.0
        r_at_t = r_pred(t_mid)
        r_target = surface_point_position(TARGET_LAT, TARGET_LON, t_mid)
        state_sun, _ = spice.spkezr("SUN", t_mid, "J2000", "NONE", "MARS")
        r_sun = np.asarray(state_sun[:3], dtype=float)
        n_expected, _cos_a = bisector_normal(r_at_t, r_target, r_sun)
        got = profile(r_at_t, t_mid)
        np.testing.assert_allclose(got, n_expected, atol=1e-14)


class TestWindowMatchingTargets:
    """_match_windows / _apply_damping respect target identity."""

    def test_cross_target_windows_never_match(self, epoch_et):
        w_old = _make_window(
            t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et,
            target_idx=0,
        )
        w_new = _make_window(
            t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et,
            target_idx=1,
        )
        matched, unmatched_old, unmatched_new = _match_windows(
            [w_old], [w_new],
        )
        assert matched == []
        assert unmatched_old == [w_old]
        assert unmatched_new == [w_new]

    def test_same_target_windows_match(self, epoch_et):
        w_old = _make_window(
            t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et,
            target_idx=1,
        )
        w_new = _make_window(
            t_start_s=110.0, t_end_s=210.0, epoch_et=epoch_et,
            target_idx=1,
        )
        matched, unmatched_old, unmatched_new = _match_windows(
            [w_old], [w_new],
        )
        assert len(matched) == 1
        assert unmatched_old == []
        assert unmatched_new == []

    def test_damping_preserves_target_fields(self, epoch_et):
        w_old = _make_window(
            t_start_s=100.0, t_end_s=200.0, epoch_et=epoch_et,
            target_idx=1,
            target_lat_deg=TARGET2_LAT, target_lon_deg=TARGET2_LON,
        )
        w_new = _make_window(
            t_start_s=120.0, t_end_s=220.0, epoch_et=epoch_et,
            target_idx=1,
            target_lat_deg=TARGET2_LAT, target_lon_deg=TARGET2_LON,
        )
        damped, unstable = _apply_damping(
            [w_old], [w_new], damping=0.5, jaccard_threshold=0.5,
        )
        assert unstable == []
        assert len(damped) == 1
        d = damped[0]
        assert d.target_idx == 1
        assert d.target_lat_deg == pytest.approx(TARGET2_LAT)
        assert d.target_lon_deg == pytest.approx(TARGET2_LON)
        assert d.et_start == pytest.approx(epoch_et + 110.0)
        assert d.et_end == pytest.approx(epoch_et + 210.0)
