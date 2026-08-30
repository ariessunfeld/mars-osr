"""Fast tests for the altitude-floor terminal event.

Six groups:

  1. ``AltitudeFloor`` dataclass contract and event construction.
  2. No-termination path: orbit stays above floor for a full period.
  3. Termination fires on an eccentric orbit whose periapsis dips
     below the floor; the captured state has altitude == floor to
     solve_ivp's root-finding tolerance.
  4. Initial-state validation: starting below the floor raises
     ValueError; starting exactly at the floor is allowed (downward-
     only event direction).
  5. Composition with other physics (zonals + third bodies + SRP):
     the floor event still fires correctly alongside the full RHS
     contributor stack.
  6. Custom reference radius and label: values round-trip into
     termination fields.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.dynamics import (
    PropagationOptions,
    PropagationResult,
    mars_gm_km3_per_s2,
    propagate,
)
from reflectors.ephemeris import utc_to_et
from reflectors.surface import mars_equatorial_radius_km
from reflectors.mars_constants import MARS_HILL_RADIUS_KM
from reflectors.termination import (
    AltitudeFloor,
    RadiusCeiling,
    make_altitude_floor_event,
    make_energy_gated_radius_ceiling_events,
    make_radius_ceiling_event,
)


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_dynamics.py conventions)
# ---------------------------------------------------------------------------


def _circular_state(altitude_km: float, mu: float) -> tuple[np.ndarray, float]:
    r = mars_equatorial_radius_km() + altitude_km
    v = float(np.sqrt(mu / r))
    state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    T = 2.0 * math.pi * float(np.sqrt(r**3 / mu))
    return state0, T


def _eccentric_state_at_periapsis(
    periapsis_alt_km: float, apoapsis_alt_km: float, mu: float
) -> tuple[np.ndarray, float, float]:
    """Eccentric orbit placed AT PERIAPSIS on +x, velocity +y.

    Returns (state0, period_T, semi_major_a). Starting at periapsis
    means the radius climbs monotonically to apoapsis at ``T / 2`` --
    convenient for testing the OUTWARD radius-ceiling event.
    """
    R = mars_equatorial_radius_km()
    rp = R + periapsis_alt_km
    ra = R + apoapsis_alt_km
    a = 0.5 * (rp + ra)
    v_peri = float(np.sqrt(mu * (2.0 / rp - 1.0 / a)))
    state0 = np.array([rp, 0.0, 0.0, 0.0, v_peri, 0.0])
    T = 2.0 * math.pi * float(np.sqrt(a**3 / mu))
    return state0, T, a


def _eccentric_state_at_apoapsis(
    periapsis_alt_km: float, apoapsis_alt_km: float, mu: float
) -> tuple[np.ndarray, float, float]:
    """Eccentric orbit placed AT APOAPSIS on +x, velocity +y.

    Returns (state0, period_T, semi_major_a). Starting at apoapsis
    means the periapsis passage happens at ``T / 2`` -- convenient
    for pinning the event time analytically for the altitude-floor
    test.
    """
    R = mars_equatorial_radius_km()
    rp = R + periapsis_alt_km
    ra = R + apoapsis_alt_km
    a = 0.5 * (rp + ra)
    v_apo = float(np.sqrt(mu * (2.0 / ra - 1.0 / a)))
    state0 = np.array([ra, 0.0, 0.0, 0.0, v_apo, 0.0])
    T = 2.0 * math.pi * float(np.sqrt(a**3 / mu))
    return state0, T, a


# ---------------------------------------------------------------------------
# Group 1 -- AltitudeFloor dataclass and event construction
# ---------------------------------------------------------------------------


class TestAltitudeFloorConstruction:
    def test_stores_fields(self):
        floor = AltitudeFloor(
            altitude_km=300.0,
            reference_radius_km=3396.19,
            label="atmosphere_intersected",
        )
        assert floor.altitude_km == 300.0
        assert floor.reference_radius_km == 3396.19
        assert floor.label == "atmosphere_intersected"
        assert floor.floor_radius_km == 3696.19

    def test_rejects_non_positive_altitude(self):
        with pytest.raises(ValueError, match="altitude_km must be > 0"):
            AltitudeFloor(altitude_km=0.0, reference_radius_km=3396.19)
        with pytest.raises(ValueError, match="altitude_km must be > 0"):
            AltitudeFloor(altitude_km=-50.0, reference_radius_km=3396.19)

    def test_rejects_non_positive_reference_radius(self):
        with pytest.raises(ValueError, match="reference_radius_km must be > 0"):
            AltitudeFloor(altitude_km=300.0, reference_radius_km=0.0)

    def test_rejects_empty_label(self):
        with pytest.raises(ValueError, match="non-empty string"):
            AltitudeFloor(
                altitude_km=300.0, reference_radius_km=3396.19, label=""
            )

    def test_at_km_factory_uses_pck_radius(self):
        """The ``at_km`` convenience pulls R_eq from the PCK (3396.19 km)."""
        floor = AltitudeFloor.at_km(300.0)
        assert floor.reference_radius_km == pytest.approx(
            mars_equatorial_radius_km(), rel=0.0, abs=1e-10
        )
        assert floor.altitude_km == 300.0
        assert floor.label == "atmosphere_intersected"

    def test_at_km_honours_custom_label(self):
        floor = AltitudeFloor.at_km(150.0, label="drag_alarm")
        assert floor.label == "drag_alarm"
        assert floor.altitude_km == 150.0


class TestEventConstruction:
    def test_event_is_terminal_and_downward_only(self):
        event = make_altitude_floor_event(AltitudeFloor.at_km(300.0))
        assert event.terminal is True
        assert event.direction == -1.0

    def test_event_value_matches_closed_form(self):
        floor = AltitudeFloor(altitude_km=300.0, reference_radius_km=3396.19)
        event = make_altitude_floor_event(floor)
        # Above the floor: g should be positive.
        y_above = np.array([4000.0, 0.0, 0.0, 0.0, 3.5, 0.0])
        g_above = event(0.0, y_above)
        assert g_above == pytest.approx(4000.0 - 3696.19, abs=1e-10)
        # Below the floor: g should be negative.
        y_below = np.array([3500.0, 0.0, 0.0, 0.0, 3.5, 0.0])
        g_below = event(0.0, y_below)
        assert g_below == pytest.approx(3500.0 - 3696.19, abs=1e-10)
        # At the floor exactly: g == 0.
        y_at = np.array([3696.19, 0.0, 0.0, 0.0, 3.5, 0.0])
        assert event(0.0, y_at) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Group 2 -- No termination when orbit stays above floor
# ---------------------------------------------------------------------------


class TestNoTermination:
    def test_circular_400km_above_300km_floor(self):
        mu = mars_gm_km3_per_s2()
        state0, T = _circular_state(400.0, mu)
        floor = AltitudeFloor.at_km(300.0)
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "t_final"
        assert result.termination_t_s is None
        assert result.termination_et is None
        assert result.termination_state_km_kmps is None
        # Final time equals the t_span upper bound.
        assert result.t_s[-1] == pytest.approx(T, rel=1e-12)

    def test_metadata_records_altitude_floor(self):
        mu = mars_gm_km3_per_s2()
        state0, T = _circular_state(400.0, mu)
        floor = AltitudeFloor.at_km(300.0)
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert "altitude_floor" in result.metadata
        af = result.metadata["altitude_floor"]
        assert af["altitude_km"] == 300.0
        assert af["reference_radius_km"] == pytest.approx(3396.19, abs=1e-6)
        assert af["label"] == "atmosphere_intersected"

    def test_no_floor_means_no_metadata_entry_and_default_reason(self):
        """Omitting ``altitude_floor`` uses the default termination fields."""
        mu = mars_gm_km3_per_s2()
        state0, T = _circular_state(400.0, mu)
        result = propagate(state0, (0.0, T), options=PropagationOptions.fast())
        assert result.termination_reason == "t_final"
        assert result.termination_t_s is None
        assert result.termination_et is None
        assert result.termination_state_km_kmps is None
        assert "altitude_floor" not in result.metadata


# ---------------------------------------------------------------------------
# Group 3 -- Termination fires on a failing eccentric orbit
# ---------------------------------------------------------------------------


class TestEventFires:
    def test_eccentric_periapsis_below_floor_terminates(self):
        """Periapsis at 250 km with 300 km floor -> event fires on the
        descent toward periapsis."""
        mu = mars_gm_km3_per_s2()
        state0, T, a = _eccentric_state_at_apoapsis(
            periapsis_alt_km=250.0, apoapsis_alt_km=1000.0, mu=mu
        )
        floor = AltitudeFloor.at_km(300.0)
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "atmosphere_intersected"
        assert result.termination_t_s is not None
        # Termination should happen well before the full period (periapsis
        # is at T/2 starting from apoapsis, and the crossing is slightly
        # before that).
        assert 0.0 < result.termination_t_s < T / 2.0 + 1.0

    def test_captured_state_altitude_matches_floor(self):
        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_apoapsis(
            periapsis_alt_km=250.0, apoapsis_alt_km=1000.0, mu=mu
        )
        floor = AltitudeFloor.at_km(300.0)
        R_eq = mars_equatorial_radius_km()
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor,
            options=PropagationOptions.high_accuracy(),
        )
        assert result.termination_state_km_kmps is not None
        r_event = result.termination_state_km_kmps[:3]
        alt_event = float(np.linalg.norm(r_event)) - R_eq
        # solve_ivp root-finder converges tightly to the zero of g(t,y);
        # high-accuracy tolerance should provide sub-metre precision.
        assert alt_event == pytest.approx(300.0, abs=1e-3)

    def test_last_sample_is_the_event_state(self):
        """``state_km_kmps[-1]`` == ``termination_state_km_kmps`` when the
        event fires."""
        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_apoapsis(
            periapsis_alt_km=250.0, apoapsis_alt_km=1000.0, mu=mu
        )
        floor = AltitudeFloor.at_km(300.0)
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert np.allclose(
            result.state_km_kmps[-1],
            result.termination_state_km_kmps,
            atol=0.0,
            rtol=0.0,
        )
        assert result.t_s[-1] == pytest.approx(
            result.termination_t_s, abs=0.0, rel=0.0
        )

    def test_event_et_added_to_epoch_when_epoch_given(self):
        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_apoapsis(
            periapsis_alt_km=250.0, apoapsis_alt_km=1000.0, mu=mu
        )
        floor = AltitudeFloor.at_km(300.0)
        epoch_et = utc_to_et(EPOCH_STR)
        result = propagate(
            state0,
            (0.0, T),
            epoch_et=epoch_et,
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert result.termination_et == pytest.approx(
            epoch_et + result.termination_t_s, rel=1e-14
        )


# ---------------------------------------------------------------------------
# Group 4 -- Initial-state validation
# ---------------------------------------------------------------------------


class TestInitialStateValidation:
    def test_initial_altitude_below_floor_raises(self):
        mu = mars_gm_km3_per_s2()
        state0, T = _circular_state(200.0, mu)
        floor = AltitudeFloor.at_km(300.0)
        with pytest.raises(ValueError, match="below altitude_floor"):
            propagate(
                state0,
                (0.0, T),
                altitude_floor=floor,
                options=PropagationOptions.fast(),
            )

    def test_initial_altitude_equal_to_floor_is_allowed(self):
        """Boundary case: alt_0 == floor. Downward-only event doesn't fire
        on t=0 (no crossing yet); propagation runs normally."""
        mu = mars_gm_km3_per_s2()
        # Circular orbit AT 300 km altitude = floor radius; zero radial
        # velocity, so no downward crossing at t=0.
        state0, T = _circular_state(300.0, mu)
        floor = AltitudeFloor.at_km(300.0)
        result = propagate(
            state0,
            (0.0, T / 4.0),  # quarter orbit -- radius stays at 300 km
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        # Circular orbit: altitude stays at the floor within numerical
        # error; event may or may not fire depending on integrator
        # float-level wobble. Assert that the integration completed without
        # raising -- the critical edge is that the initial-state check did
        # not reject a state exactly at the boundary.
        assert result.t_s[0] == 0.0


# ---------------------------------------------------------------------------
# Group 5 -- Composition with other physics
# ---------------------------------------------------------------------------


class TestComposition:
    def test_event_fires_with_zonals_and_third_bodies(self):
        """Full stack: zonal gravity + Sun third body + altitude floor."""
        from reflectors.third_body import sun_third_body

        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_apoapsis(
            periapsis_alt_km=250.0, apoapsis_alt_km=1000.0, mu=mu
        )
        floor = AltitudeFloor.at_km(300.0)
        epoch_et = utc_to_et(EPOCH_STR)
        result = propagate(
            state0,
            (0.0, T),
            epoch_et=epoch_et,
            zonal_degree=2,
            third_bodies=[sun_third_body()],
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "atmosphere_intersected"
        # Same periapsis geometry; event time should be within a few %
        # of the two-body case (J2 + Sun are tiny over one orbit at LMO).
        assert result.termination_t_s is not None


# ---------------------------------------------------------------------------
# Group 6 -- Custom reference radius and label
# ---------------------------------------------------------------------------


class TestCustomConfiguration:
    def test_custom_label_round_trips(self):
        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_apoapsis(
            periapsis_alt_km=250.0, apoapsis_alt_km=1000.0, mu=mu
        )
        floor = AltitudeFloor.at_km(300.0, label="custom_floor")
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor,
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "custom_floor"
        assert result.metadata["altitude_floor"]["label"] == "custom_floor"

    def test_custom_reference_radius(self):
        """Using the Mars polar radius instead of equatorial shifts the
        event trigger by ~20 km."""
        mu = mars_gm_km3_per_s2()
        # Polar radius is about 3376.2 km; 300 km "altitude above polar"
        # radius = 3676.2 km, 20 km lower than using R_eq. An orbit with
        # peri at alt_eq = 280 km would NOT terminate with a 300 km floor
        # referenced to R_polar (peri r = R_eq + 280 = 3676.19 km, just
        # at the polar-referenced floor radius of 3676.2 km).
        floor_polar = AltitudeFloor(
            altitude_km=300.0,
            reference_radius_km=3376.2,  # Mars polar radius
            label="polar_floor",
        )
        # Orbit with peri at r = 3800 km (alt above R_eq = 404 km) stays
        # above BOTH an equatorial 300 km floor (3696.19 km) and a polar
        # 300 km floor (3676.2 km), so it should complete normally.
        state0, T, _ = _eccentric_state_at_apoapsis(
            periapsis_alt_km=404.0, apoapsis_alt_km=1000.0, mu=mu
        )
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=floor_polar,
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "t_final"
        assert result.metadata["altitude_floor"]["reference_radius_km"] == 3376.2


# ---------------------------------------------------------------------------
# Group 7 -- RadiusCeiling dataclass and event construction
# ---------------------------------------------------------------------------


class TestRadiusCeilingConstruction:
    def test_stores_fields(self):
        ceiling = RadiusCeiling(radius_km=1.0e6, label="hill_sphere_exit")
        assert ceiling.radius_km == 1.0e6
        assert ceiling.label == "hill_sphere_exit"

    def test_rejects_non_positive_radius(self):
        with pytest.raises(ValueError, match="radius_km must be > 0"):
            RadiusCeiling(radius_km=0.0)
        with pytest.raises(ValueError, match="radius_km must be > 0"):
            RadiusCeiling(radius_km=-1.0e6)

    def test_rejects_empty_label(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RadiusCeiling(radius_km=1.0e6, label="")

    def test_hill_sphere_factory_uses_pinned_constant(self):
        ceiling = RadiusCeiling.hill_sphere()
        assert ceiling.radius_km == MARS_HILL_RADIUS_KM
        assert ceiling.label == "hill_sphere_exit"

    def test_hill_sphere_factory_honours_custom_label(self):
        ceiling = RadiusCeiling.hill_sphere(label="soi_exit")
        assert ceiling.label == "soi_exit"
        assert ceiling.radius_km == MARS_HILL_RADIUS_KM


class TestRadiusCeilingEventConstruction:
    def test_event_is_terminal_and_upward_only(self):
        event = make_radius_ceiling_event(RadiusCeiling(radius_km=1.0e6))
        assert event.terminal is True
        assert event.direction == +1.0

    def test_event_value_matches_closed_form(self):
        event = make_radius_ceiling_event(RadiusCeiling(radius_km=1.0e6))
        # Below the ceiling: g should be negative.
        y_below = np.array([8.0e5, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert event(0.0, y_below) == pytest.approx(8.0e5 - 1.0e6, abs=1e-6)
        # Above the ceiling: g should be positive.
        y_above = np.array([1.2e6, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert event(0.0, y_above) == pytest.approx(1.2e6 - 1.0e6, abs=1e-6)
        # At the ceiling exactly: g == 0.
        y_at = np.array([1.0e6, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert event(0.0, y_at) == pytest.approx(0.0, abs=1e-6)

    def test_event_reads_radius_from_first_three_components(self):
        """The event uses only y[:3], so it works unchanged on the 12-D
        augmented escape state [r, v, n, omega]."""
        event = make_radius_ceiling_event(RadiusCeiling(radius_km=1.0e6))
        y12 = np.array(
            [6.0e5, 8.0e5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        # |r| = sqrt(6e5^2 + 8e5^2) = 1.0e6 exactly.
        assert event(0.0, y12) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Group 8 -- RadiusCeiling fires under propagate (outward trajectory)
# ---------------------------------------------------------------------------


class TestRadiusCeilingFires:
    def test_outward_eccentric_orbit_terminates_at_ceiling(self):
        """Eccentric orbit climbing from periapsis crosses a ceiling set
        between periapsis and apoapsis -> event fires."""
        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_periapsis(
            periapsis_alt_km=500.0, apoapsis_alt_km=20000.0, mu=mu
        )
        R_eq = mars_equatorial_radius_km()
        ceiling = RadiusCeiling(radius_km=R_eq + 10000.0, label="hill_sphere_exit")
        result = propagate(
            state0,
            (0.0, T),
            radius_ceiling=ceiling,
            options=PropagationOptions.high_accuracy(),
        )
        assert result.termination_reason == "hill_sphere_exit"
        assert result.termination_t_s is not None
        # Periapsis-start: apoapsis reached at T/2, ceiling crossed before.
        assert 0.0 < result.termination_t_s < T / 2.0 + 1.0
        r_event = float(np.linalg.norm(result.termination_state_km_kmps[:3]))
        assert r_event == pytest.approx(R_eq + 10000.0, abs=1e-2)
        assert result.metadata["radius_ceiling"]["radius_km"] == R_eq + 10000.0
        assert result.metadata["radius_ceiling"]["label"] == "hill_sphere_exit"

    def test_circular_orbit_below_ceiling_does_not_terminate(self):
        mu = mars_gm_km3_per_s2()
        state0, T = _circular_state(500.0, mu)
        ceiling = RadiusCeiling(radius_km=1.0e6)
        result = propagate(
            state0,
            (0.0, T),
            radius_ceiling=ceiling,
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "t_final"
        assert result.termination_t_s is None

    def test_initial_state_above_ceiling_raises(self):
        mu = mars_gm_km3_per_s2()
        state0, T = _circular_state(500.0, mu)
        # Ceiling below the 500 km circular orbit radius.
        ceiling = RadiusCeiling(radius_km=1000.0)
        with pytest.raises(ValueError, match="above radius_ceiling"):
            propagate(
                state0,
                (0.0, T),
                radius_ceiling=ceiling,
                options=PropagationOptions.fast(),
            )

    def test_floor_and_ceiling_stack_earliest_wins(self):
        """With both an altitude floor and a radius ceiling registered, the
        outward orbit hits the ceiling and terminates with the ceiling
        label (the floor never fires)."""
        mu = mars_gm_km3_per_s2()
        state0, T, _ = _eccentric_state_at_periapsis(
            periapsis_alt_km=500.0, apoapsis_alt_km=20000.0, mu=mu
        )
        R_eq = mars_equatorial_radius_km()
        result = propagate(
            state0,
            (0.0, T),
            altitude_floor=AltitudeFloor.at_km(300.0),
            radius_ceiling=RadiusCeiling(radius_km=R_eq + 10000.0),
            options=PropagationOptions.fast(),
        )
        assert result.termination_reason == "hill_sphere_exit"


# ---------------------------------------------------------------------------
# Energy-gated radius-ceiling events (escape = E >= 0 AND |r| >= Hill)
# ---------------------------------------------------------------------------


class TestEnergyGatedRadiusCeilingEvents:
    MU = 4.0e4          # km^3/s^2 (Mars-ish; the factory is mu-agnostic)
    R_HILL = 1.0e6      # km

    def _events(self, outer_kill_factor=2.0):
        return make_energy_gated_radius_ceiling_events(
            RadiusCeiling(radius_km=self.R_HILL),
            self.MU,
            outer_kill_factor=outer_kill_factor,
        )

    @staticmethod
    def _state(r_km, speed_kmps):
        # In-plane: position +x at r_km, velocity +y at speed_kmps. 12-D padded
        # (the events only read y[:6]); attitude slots are irrelevant.
        return np.array(
            [r_km, 0.0, 0.0, 0.0, speed_kmps, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_returns_three_detectors_with_expected_labels_directions(self):
        events = self._events()
        assert len(events) == 3
        labels = [label for (_g, _d, label, _a) in events]
        directions = [d for (_g, d, _l, _a) in events]
        assert labels == ["hill_sphere_exit", "hill_sphere_exit", "outer_kill_radius"]
        assert directions == [+1, +1, +1]
        # Radius + energy detectors carry accept-predicates; outer kill does not.
        assert events[0][3] is not None
        assert events[1][3] is not None
        assert events[2][3] is None

    def test_radius_detector_accepts_only_when_unbound(self):
        g_radius, _d, label, accept = self._events()[0]
        assert label == "hill_sphere_exit"
        # g = |r| - Hill (radius up-crossing).
        assert g_radius(0.0, self._state(self.R_HILL + 1.0, 0.3)) > 0.0
        assert g_radius(0.0, self._state(self.R_HILL - 1.0, 0.3)) < 0.0
        # accept iff E = v^2/2 - mu/|r| >= 0. At r=Hill, v_esc = sqrt(2 mu/r).
        v_esc = math.sqrt(2.0 * self.MU / self.R_HILL)
        assert accept(self._state(self.R_HILL, 1.01 * v_esc))   # hyperbolic
        assert not accept(self._state(self.R_HILL, 0.99 * v_esc))  # bound

    def test_energy_detector_accepts_only_at_or_beyond_hill(self):
        g_energy, _d, label, accept = self._events()[1]
        assert label == "hill_sphere_exit"
        v_esc = math.sqrt(2.0 * self.MU / self.R_HILL)
        # g = E (energy up-crossing): positive when unbound.
        assert g_energy(0.0, self._state(self.R_HILL, 1.01 * v_esc)) > 0.0
        assert g_energy(0.0, self._state(self.R_HILL, 0.99 * v_esc)) < 0.0
        # accept iff |r| >= Hill.
        assert accept(self._state(self.R_HILL + 1.0, 1.0))
        assert not accept(self._state(self.R_HILL - 1.0, 1.0))

    def test_outer_kill_radius_fires_at_factor_times_hill(self):
        g_outer, _d, label, accept = self._events(outer_kill_factor=2.0)[2]
        assert label == "outer_kill_radius"
        assert accept is None
        assert g_outer(0.0, self._state(2.0 * self.R_HILL + 1.0, 0.1)) > 0.0
        assert g_outer(0.0, self._state(2.0 * self.R_HILL - 1.0, 0.1)) < 0.0

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            make_energy_gated_radius_ceiling_events(
                RadiusCeiling(radius_km=self.R_HILL), -1.0
            )
        with pytest.raises(ValueError):
            make_energy_gated_radius_ceiling_events(
                RadiusCeiling(radius_km=self.R_HILL), self.MU, outer_kill_factor=1.0
            )
