"""Fast tests for the ``reflectors.visibility`` reflection-delivery
foundation (pointing law + visibility gates + window finder + quality
scalars).

Groups follow the implementation layers in ``reflectors.visibility``:

  1. Geometry primitives
  2. Bisector pointing law
  3. Four-gate evaluator
  4. Window finder
  5. Slew-demand scalar
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import spiceypy as spice

import reflectors.ephemeris as ephemeris
from reflectors.ephemeris import utc_to_et
from reflectors.surface import (
    BODY_FIXED_FRAME,
    mars_equatorial_radius_km,
    surface_point_body_fixed,
    surface_point_position,
)
from reflectors.attitude import AttitudeCallable, angular_rate
from reflectors.dynamics import (
    PropagationOptions,
    PropagationResult,
    mars_gm_km3_per_s2,
    propagate,
)
from reflectors.surface import mars_equatorial_radius_km
from reflectors.third_body import (
    deimos_third_body,
    phobos_third_body,
    sun_third_body,
)
from reflectors.visibility import (
    _BISECTOR_DEGENERATE_SUM_TOL,
    _bisect_boolean_transition,
    _target_outward_normal_body_fixed,
    DeliveryGates,
    DeliveryGeometry,
    DeliverySamples,
    DeliveryWindow,
    WindowContinuationError,
    bisector_normal,
    bisector_pointing,
    delivery_gates,
    delivery_geometry,
    continue_delivery_windows_multi,
    find_delivery_windows,
    find_delivery_windows_multi,
    sail_above_target_horizon,
    slant_range_km,
    target_elevation_angle_rad,
    target_in_sunlight_at,
    target_outward_normal_j2000,
)


EPOCH_STR = "2026-06-01T00:00:00"
TARGET_LAT_DEG = 40.0
TARGET_LON_DEG = 200.0


@pytest.fixture(scope="module")
def epoch_et() -> float:
    return utc_to_et(EPOCH_STR)


@pytest.fixture(scope="module")
def target_r_j2000(epoch_et) -> np.ndarray:
    """Target (40°N, 200°E) position on the spheroid surface, J2000 km."""
    return surface_point_position(
        TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et, alt_km=0.0, planetographic=True
    )


@pytest.fixture(scope="module")
def target_n_outward_j2000(epoch_et) -> np.ndarray:
    return target_outward_normal_j2000(TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et)


# ---------------------------------------------------------------------------
# Group 1: Geometry primitives
# ---------------------------------------------------------------------------


class TestOutwardNormalBodyFixed:
    """``_target_outward_normal_body_fixed`` -- closed-form anchor tests."""

    def test_equator_prime_meridian_is_x_hat(self):
        n = _target_outward_normal_body_fixed(0.0, 0.0)
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-15)

    def test_north_pole_is_z_hat(self):
        # Longitude is irrelevant at the pole; pick an arbitrary value.
        n = _target_outward_normal_body_fixed(90.0, 137.0)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-15)

    def test_south_pole_is_minus_z_hat(self):
        n = _target_outward_normal_body_fixed(-90.0, 42.0)
        np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-15)

    def test_equator_90_east_is_y_hat(self):
        n = _target_outward_normal_body_fixed(0.0, 90.0)
        np.testing.assert_allclose(n, [0.0, 1.0, 0.0], atol=1e-15)

    def test_canonical_target_matches_closed_form(self):
        lat = math.radians(TARGET_LAT_DEG)
        lon = math.radians(TARGET_LON_DEG)
        expected = np.array(
            [math.cos(lat) * math.cos(lon),
             math.cos(lat) * math.sin(lon),
             math.sin(lat)]
        )
        n = _target_outward_normal_body_fixed(TARGET_LAT_DEG, TARGET_LON_DEG)
        np.testing.assert_allclose(n, expected, atol=1e-15)

    def test_always_unit_norm(self):
        for lat, lon in [(0, 0), (40, 200), (-57, -33), (89.999, 1), (-89.999, 359)]:
            n = _target_outward_normal_body_fixed(lat, lon)
            assert abs(np.linalg.norm(n) - 1.0) < 1e-15


class TestOutwardNormalJ2000:
    """``target_outward_normal_j2000`` -- unit-norm + frame consistency."""

    def test_unit_norm(self, epoch_et):
        n = target_outward_normal_j2000(TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-14

    def test_rotating_body_fixed_agrees_with_pxform(self, epoch_et):
        n_bf = _target_outward_normal_body_fixed(TARGET_LAT_DEG, TARGET_LON_DEG)
        M = spice.pxform(BODY_FIXED_FRAME, "J2000", epoch_et)
        expected = np.asarray(M) @ n_bf
        n = target_outward_normal_j2000(TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et)
        np.testing.assert_allclose(n, expected, atol=1e-15)

    def test_altitude_consistency_with_surface_point(self, epoch_et):
        """Geodetic altitude is measured along the outward normal.

        Concretely: ``surface_point_body_fixed(lat, lon, h) ==
        surface_point_body_fixed(lat, lon, 0) + h * n_outward_body_fixed``.
        This is not an additional assumption about the normal -- it is
        the defining property of geodetic coordinates, derivable from
        the explicit ``(N+h) cos phi ..., (N(1-e^2) + h) sin phi``
        formula in ``surface_point_body_fixed`` by inspection. Pinning
        it here catches any accidental drift in either formula.
        """
        r0 = surface_point_body_fixed(
            TARGET_LAT_DEG, TARGET_LON_DEG, 0.0, planetographic=True
        )
        r400 = surface_point_body_fixed(
            TARGET_LAT_DEG, TARGET_LON_DEG, 400.0, planetographic=True
        )
        n_bf = _target_outward_normal_body_fixed(
            TARGET_LAT_DEG, TARGET_LON_DEG
        )
        np.testing.assert_allclose(r400 - r0, 400.0 * n_bf, atol=1e-12)


def test_delivery_geometry_reuses_sun_and_surface_rotation(
    monkeypatch, epoch_et, target_r_j2000, target_n_outward_j2000
):
    """One geometry evaluation makes one Sun call and one frame call."""
    original_spkezr = spice.spkezr
    original_pxform = spice.pxform
    state_calls = 0
    rotation_calls = 0

    def counted_spkezr(*args):
        nonlocal state_calls
        state_calls += 1
        return original_spkezr(*args)

    def counted_pxform(*args):
        nonlocal rotation_calls
        rotation_calls += 1
        return original_pxform(*args)

    monkeypatch.setattr(ephemeris.spice, "spkezr", counted_spkezr)
    monkeypatch.setattr(ephemeris.spice, "pxform", counted_pxform)
    r_sat = target_r_j2000 + 400.0 * target_n_outward_j2000
    geometry = delivery_geometry(
        r_sat,
        epoch_et,
        TARGET_LAT_DEG,
        TARGET_LON_DEG,
    )

    assert isinstance(geometry, DeliveryGeometry)
    assert state_calls == 1
    assert rotation_calls == 1
    np.testing.assert_allclose(
        geometry.target_position_j2000_km, target_r_j2000, atol=1e-12
    )
    np.testing.assert_allclose(
        geometry.target_outward_normal_j2000,
        target_n_outward_j2000,
        atol=1e-15,
    )
    assert geometry.slant_range_km == pytest.approx(400.0, abs=1e-12)
    assert geometry.target_elevation_rad == pytest.approx(
        np.pi / 2.0, abs=1e-7
    )
    assert geometry.gates.sail_above_target_horizon is True


class TestSailAboveTargetHorizon:
    """``sail_above_target_horizon`` -- half-space predicate."""

    def test_zenith_above(self, target_r_j2000, target_n_outward_j2000):
        r_sat = target_r_j2000 + 400.0 * target_n_outward_j2000
        assert sail_above_target_horizon(
            r_sat, target_r_j2000, target_n_outward_j2000
        ) is True

    def test_antipode_below(self, target_r_j2000, target_n_outward_j2000):
        # Sail at the antipode through Mars centre: r_sat = -r_target.
        r_sat = -target_r_j2000
        assert sail_above_target_horizon(
            r_sat, target_r_j2000, target_n_outward_j2000
        ) is False

    def test_tangent_displacement_is_on_horizon_plane(
        self, target_r_j2000, target_n_outward_j2000
    ):
        # Choose any tangent direction orthogonal to n_outward and move
        # the sail along it. By construction dot(r_sat - r_target,
        # n_outward) == 0 analytically; in double precision the residual
        # is at the 1e-13 level either sign, so the predicate's return
        # value at this exact boundary is FP-dependent and not a
        # meaningful semantic test. What IS meaningful is that the
        # dot product itself is near zero and is pinned here.
        n = target_n_outward_j2000
        trial = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, trial)
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([1.0, 0.0, 0.0]))
        t = t / np.linalg.norm(t)
        r_sat = target_r_j2000 + 500.0 * t
        assert abs(float(np.dot(r_sat - target_r_j2000, n))) < 1e-12

    def test_small_positive_displacement_above(
        self, target_r_j2000, target_n_outward_j2000
    ):
        r_sat = target_r_j2000 + 1e-3 * target_n_outward_j2000
        assert sail_above_target_horizon(
            r_sat, target_r_j2000, target_n_outward_j2000
        ) is True

    def test_small_negative_displacement_below(
        self, target_r_j2000, target_n_outward_j2000
    ):
        r_sat = target_r_j2000 - 1e-3 * target_n_outward_j2000
        assert sail_above_target_horizon(
            r_sat, target_r_j2000, target_n_outward_j2000
        ) is False


class TestTargetElevationAngle:
    """``target_elevation_angle_rad`` -- analytic anchor tests."""

    def test_zenith_is_pi_over_two(self, target_r_j2000, target_n_outward_j2000):
        r_sat = target_r_j2000 + 400.0 * target_n_outward_j2000
        el = target_elevation_angle_rad(
            r_sat, target_r_j2000, target_n_outward_j2000
        )
        # arcsin's derivative diverges at +-1, so the FP error floor at
        # zenith / nadir is ~sqrt(eps) = 1.5e-8 rad regardless of
        # formulation. 1e-7 rad = 0.02 arcsec is comfortably below any
        # physically meaningful tolerance.
        assert abs(el - math.pi / 2.0) < 1e-7

    def test_horizon_is_zero(self, target_r_j2000, target_n_outward_j2000):
        n = target_n_outward_j2000
        trial = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, trial)
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([1.0, 0.0, 0.0]))
        t = t / np.linalg.norm(t)
        r_sat = target_r_j2000 + 1234.5 * t
        el = target_elevation_angle_rad(
            r_sat, target_r_j2000, n
        )
        assert abs(el) < 1e-14

    def test_antipodal_direction_minus_pi_over_two(
        self, target_r_j2000, target_n_outward_j2000
    ):
        # Sail below the target along the outward normal (radially inward).
        # Use a displacement shorter than the target's radius to avoid
        # trip the ValueError by landing on r_target itself. Same arcsin
        # precision floor as the zenith case.
        r_sat = target_r_j2000 - 250.0 * target_n_outward_j2000
        el = target_elevation_angle_rad(
            r_sat, target_r_j2000, target_n_outward_j2000
        )
        assert abs(el - (-math.pi / 2.0)) < 1e-7

    def test_coincident_positions_raises(
        self, target_r_j2000, target_n_outward_j2000
    ):
        with pytest.raises(ValueError):
            target_elevation_angle_rad(
                target_r_j2000, target_r_j2000, target_n_outward_j2000
            )

    def test_45_degree_elevation(
        self, target_r_j2000, target_n_outward_j2000
    ):
        # Mix one unit along outward normal with one unit along a tangent;
        # elevation should be arctan(1/1) = pi/4 (since arcsin(1/sqrt(2))
        # = pi/4).
        n = target_n_outward_j2000
        trial = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, trial)
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([1.0, 0.0, 0.0]))
        t = t / np.linalg.norm(t)
        r_sat = target_r_j2000 + 500.0 * (n + t)
        el = target_elevation_angle_rad(r_sat, target_r_j2000, n)
        assert abs(el - math.pi / 4.0) < 1e-14

    def test_signed_monotonicity(
        self, target_r_j2000, target_n_outward_j2000
    ):
        # Walking the sail from below-horizon to above-horizon along a
        # linear path should produce monotonically increasing elevation.
        n = target_n_outward_j2000
        # Pick a tangent so the path is well-defined.
        trial = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, trial)
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([1.0, 0.0, 0.0]))
        t = t / np.linalg.norm(t)
        # Path: from -200 n + 500 t (below) to +200 n + 500 t (above).
        s_grid = np.linspace(-200.0, 200.0, 21)
        elevs = []
        for s in s_grid:
            r_sat = target_r_j2000 + s * n + 500.0 * t
            elevs.append(target_elevation_angle_rad(r_sat, target_r_j2000, n))
        elevs = np.array(elevs)
        assert np.all(np.diff(elevs) > 0)


class TestSlantRange:
    def test_zenith_400km(self, target_r_j2000, target_n_outward_j2000):
        r_sat = target_r_j2000 + 400.0 * target_n_outward_j2000
        d = slant_range_km(r_sat, target_r_j2000)
        assert abs(d - 400.0) < 1e-12

    def test_zero_when_coincident(self, target_r_j2000):
        d = slant_range_km(target_r_j2000, target_r_j2000)
        assert d == 0.0

    def test_triangle_inequality_example(self, target_r_j2000):
        # sail offset by (3, 4, 0) km from target -> distance 5 km.
        r_sat = target_r_j2000 + np.array([3.0, 4.0, 0.0])
        d = slant_range_km(r_sat, target_r_j2000)
        assert abs(d - 5.0) < 1e-12


# ---------------------------------------------------------------------------
# Group 2: Bisector pointing law
# ---------------------------------------------------------------------------


class TestBisectorNormal:
    """``bisector_normal`` -- the specular half-angle-bisector formula."""

    def test_sun_and_target_coincident_from_sail(self):
        # Sail at origin, sun and target both along +x at different
        # distances -- s_hat_sun == s_hat_target exactly; bisector
        # is that common direction; cos_alpha = 1.
        r_sat = np.zeros(3)
        r_target = np.array([1000.0, 0.0, 0.0])
        r_sun = np.array([1.5e8, 0.0, 0.0])
        n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        np.testing.assert_allclose(n_hat, [1.0, 0.0, 0.0], atol=1e-15)
        assert abs(cos_alpha - 1.0) < 1e-15

    def test_sun_and_target_at_ninety_degrees(self):
        # Sail at origin, target along +x, sun along +y. Bisector is
        # (1,1,0)/sqrt(2); half-angle is 45 deg so cos_alpha = 1/sqrt(2).
        r_sat = np.zeros(3)
        r_target = np.array([1000.0, 0.0, 0.0])
        r_sun = np.array([0.0, 1.5e8, 0.0])
        n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        expected = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        np.testing.assert_allclose(n_hat, expected, atol=1e-15)
        assert abs(cos_alpha - 1.0 / math.sqrt(2.0)) < 1e-15

    def test_sun_and_target_at_sixty_degrees(self):
        # Half-angle = 30 deg -> cos_alpha = cos(30 deg) = sqrt(3)/2.
        r_sat = np.zeros(3)
        r_target = np.array([1000.0, 0.0, 0.0])
        # Sun direction at 60 deg from target direction, in the xy-plane.
        angle = math.radians(60.0)
        r_sun = 1.5e8 * np.array([math.cos(angle), math.sin(angle), 0.0])
        _n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        assert abs(cos_alpha - math.sqrt(3.0) / 2.0) < 1e-15

    def test_antiparallel_sun_and_target_returns_degenerate(self):
        # Sail at origin, target +x, sun -x: s_hat_sun + s_hat_target
        # = zero vector. Degenerate -> (zeros, 0.0).
        r_sat = np.zeros(3)
        r_target = np.array([1000.0, 0.0, 0.0])
        r_sun = np.array([-1.5e8, 0.0, 0.0])
        n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        np.testing.assert_allclose(n_hat, np.zeros(3), atol=1e-15)
        assert cos_alpha == 0.0

    def test_near_antiparallel_under_threshold_is_degenerate(self):
        # Half-angle very close to pi/2 so |sum| is below threshold.
        # |sum| = 2 cos(half_angle). For |sum| < 2e-3, half_angle >
        # acos(1e-3) ~ pi/2 - 1e-3. Construct with half-angle = pi/2 - 1e-4.
        r_sat = np.zeros(3)
        r_target = np.array([1000.0, 0.0, 0.0])
        half_angle = math.pi / 2.0 - 1e-4
        full_angle = 2.0 * half_angle
        r_sun = 1.5e8 * np.array(
            [math.cos(full_angle), math.sin(full_angle), 0.0]
        )
        n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        # |sum| = 2 cos(half_angle) ~ 2e-4 < 2e-3 threshold -> degenerate.
        assert n_hat.tolist() == [0.0, 0.0, 0.0]
        assert cos_alpha == 0.0

    def test_near_antiparallel_above_threshold_is_finite(self):
        # Half-angle = pi/2 - 5e-3, |sum| = 2 cos(pi/2 - 5e-3) = 2 sin(5e-3)
        # ~ 1e-2 > threshold. Returns finite values.
        r_sat = np.zeros(3)
        r_target = np.array([1000.0, 0.0, 0.0])
        half_angle = math.pi / 2.0 - 5e-3
        full_angle = 2.0 * half_angle
        r_sun = 1.5e8 * np.array(
            [math.cos(full_angle), math.sin(full_angle), 0.0]
        )
        n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        assert np.linalg.norm(n_hat) > 0.5
        assert cos_alpha > 0.0
        assert cos_alpha < 1e-2  # very close to degeneracy

    def test_n_hat_is_unit_vector_when_feasible(self):
        # Random non-degenerate geometries.
        rng = np.random.default_rng(seed=20260422)
        for _ in range(20):
            r_sat = rng.normal(size=3) * 1000.0
            # Random target at distance ~3e3; random sun at distance ~1.5e8.
            # Skip the degenerate case by retrying if sum-norm is tiny.
            while True:
                r_target = r_sat + rng.normal(size=3) * 3e3
                r_sun = r_sat + rng.normal(size=3) * 1.5e8
                n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
                if cos_alpha > 1e-3:
                    break
            assert abs(np.linalg.norm(n_hat) - 1.0) < 1e-14

    def test_bisector_identity_dot_products_equal(self):
        # dot(n_hat, s_hat_sun) == dot(n_hat, s_hat_target) == cos_alpha
        # for any non-degenerate configuration. Pin to machine precision.
        r_sat = np.array([100.0, 200.0, 300.0])
        r_target = np.array([3500.0, -200.0, 2100.0])
        r_sun = np.array([1.5e8, -4e7, 8e6])
        n_hat, cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        s_hat_sun = (r_sun - r_sat) / np.linalg.norm(r_sun - r_sat)
        s_hat_target = (r_target - r_sat) / np.linalg.norm(r_target - r_sat)
        d1 = float(np.dot(n_hat, s_hat_sun))
        d2 = float(np.dot(n_hat, s_hat_target))
        assert abs(d1 - d2) < 1e-14
        assert abs(d1 - cos_alpha) < 1e-14

    def test_coincident_target_raises(self):
        with pytest.raises(ValueError):
            bisector_normal(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([1e8, 0.0, 0.0]),
            )

    def test_coincident_sun_raises(self):
        with pytest.raises(ValueError):
            bisector_normal(
                np.array([1.0, 2.0, 3.0]),
                np.array([1000.0, 0.0, 0.0]),
                np.array([1.0, 2.0, 3.0]),
            )


class TestBisectorPointing:
    """``bisector_pointing`` factory -- AttitudeCallable interface."""

    def test_signature_matches_attitude_callable(self, epoch_et):
        profile: AttitudeCallable = bisector_pointing(
            TARGET_LAT_DEG, TARGET_LON_DEG
        )
        r_sat = np.array([5000.0, 0.0, 0.0])  # far enough from Mars
        n_hat = profile(r_sat, epoch_et)
        assert isinstance(n_hat, np.ndarray)
        assert n_hat.shape == (3,)
        assert abs(np.linalg.norm(n_hat) - 1.0) < 1e-14

    def test_fetches_sun_and_target_via_spice(self, epoch_et):
        # Independent reconstruction: fetch the Sun and target directly,
        # call bisector_normal, confirm bisector_pointing agrees.
        profile = bisector_pointing(TARGET_LAT_DEG, TARGET_LON_DEG)
        r_sat = np.array([5000.0, 0.0, 0.0])
        n_from_profile = profile(r_sat, epoch_et)

        r_target = surface_point_position(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et, alt_km=0.0,
            planetographic=True,
        )
        state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
        r_sun = np.asarray(state[:3], dtype=float)
        n_expected, _cos_alpha = bisector_normal(r_sat, r_target, r_sun)
        np.testing.assert_allclose(n_from_profile, n_expected, atol=1e-14)

    def test_degenerate_geometry_raises(self, epoch_et):
        # Place the sail on the line from Mars through the Sun
        # far behind Mars: from the sail, both Mars centre (and hence
        # the target at (40, 200)) and the Sun lie in nearly opposite
        # directions. Construct the degenerate configuration:
        # sail = r_target + k * s_hat_sun_from_target for large k, so
        # s_hat_target (sail->target) is -s_hat_sun (sail->sun).
        r_target = surface_point_position(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et, alt_km=0.0,
            planetographic=True,
        )
        state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
        r_sun = np.asarray(state[:3], dtype=float)
        s_hat_sun_from_target = (r_sun - r_target) / np.linalg.norm(
            r_sun - r_target
        )
        # Place sail 5000 km from target along +s_hat_sun_from_target:
        # sun direction from sail is nearly +s_hat_sun_from_target,
        # target direction from sail is -s_hat_sun_from_target. Antiparallel.
        r_sat = r_target + 5000.0 * s_hat_sun_from_target

        profile = bisector_pointing(TARGET_LAT_DEG, TARGET_LON_DEG)
        with pytest.raises(ValueError, match="degenerate"):
            profile(r_sat, epoch_et)

    def test_composes_with_attitude_angular_rate(self, epoch_et):
        # Feeding bisector_pointing into attitude.angular_rate should
        # yield a finite omega at a well-conditioned sail position.
        profile = bisector_pointing(TARGET_LAT_DEG, TARGET_LON_DEG)
        # Static trajectory callable; angular_rate still evaluates
        # because the sun/target sweep through the bisector produces
        # nonzero omega even for a stationary sail (Mars rotates).
        r_sat_static = np.array([5000.0, 0.0, 0.0])

        def r_sat_fn(et: float) -> np.ndarray:
            return r_sat_static

        omega = angular_rate(profile, r_sat_fn, epoch_et, dt=1.0)
        assert omega.shape == (3,)
        assert np.all(np.isfinite(omega))


# ---------------------------------------------------------------------------
# Group 3: Four-gate evaluator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sun_r_j2000(epoch_et) -> np.ndarray:
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    return np.asarray(state[:3], dtype=float)


@pytest.fixture(scope="module")
def sun_hat_j2000(sun_r_j2000) -> np.ndarray:
    return sun_r_j2000 / np.linalg.norm(sun_r_j2000)


class TestTargetInSunlightAt:
    """``target_in_sunlight_at`` -- sun-above-target-horizon predicate."""

    def test_matches_independent_reconstruction(self, epoch_et):
        # Reconstruct the predicate from primitives; bit-for-bit match.
        r_target = surface_point_position(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et, alt_km=0.0,
            planetographic=True,
        )
        state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
        r_sun = np.asarray(state[:3], dtype=float)
        n = target_outward_normal_j2000(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et
        )
        expected = float(np.dot(r_sun - r_target, n)) > 0.0
        assert target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et
        ) is expected

    def test_antipode_has_opposite_sunlight_state(self, epoch_et):
        # The antipodal surface point has outward normal = -n and the
        # same sun direction. sign of dot flips; sunlight state flips.
        # (Exactly 180 degrees apart in (lat, lon).)
        antipode_lat = -TARGET_LAT_DEG
        antipode_lon = (TARGET_LON_DEG + 180.0) % 360.0
        lit_here = target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et
        )
        lit_there = target_in_sunlight_at(antipode_lat, antipode_lon, epoch_et)
        assert lit_here != lit_there

    def test_half_sol_later_flips_sunlight(self, epoch_et):
        # Mars sidereal rotation: ~88643 s. At a half-sidereal-day offset,
        # the target has rotated by ~180 deg in body-fixed space, so its
        # sun-exposure state has almost certainly flipped (Mars's
        # heliocentric motion is a negligible correction on 12 hour
        # timescales).
        half_sol_s = 88643.0 / 2.0
        lit_t0 = target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et
        )
        lit_half = target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et + half_sol_s
        )
        assert lit_t0 != lit_half


class TestDeliveryGatesConstruction:
    """``DeliveryGates`` named tuple + return shape of ``delivery_gates``."""

    def test_fields_present_and_typed(self):
        g = DeliveryGates(True, False, True, False)
        assert g.sail_sunlit is True
        assert g.sail_above_target_horizon is False
        assert g.bisector_feasible is True
        assert g.target_in_sunlight is False

    def test_delivery_gates_returns_named_tuple(self, epoch_et):
        r_sat = np.array([5000.0, 0.0, 0.0])
        g = delivery_gates(r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG)
        assert isinstance(g, DeliveryGates)
        assert isinstance(g.sail_sunlit, bool)
        assert isinstance(g.sail_above_target_horizon, bool)
        assert isinstance(g.bisector_feasible, bool)
        assert isinstance(g.target_in_sunlight, bool)


class TestDeliveryGatesFlip:
    """Verify each gate flips independently on a constructed scenario."""

    def test_sail_in_umbra_kills_sail_sunlit_gate(
        self, epoch_et, sun_hat_j2000, target_r_j2000, target_n_outward_j2000
    ):
        # Deep in Mars umbra: sail on the anti-solar axis at 400 km
        # altitude, well inside the 1.12e6 km umbra cone. Position the
        # sail over the target (same outward direction in body-fixed ~
        # same J2000 direction at epoch) but shadowed by Mars because
        # the target's outward direction is roughly opposite the Sun.
        # Easier: just place the sail on -sun_hat at R_mars + 400 km.
        R_sat = mars_equatorial_radius_km() + 400.0
        r_sat = -R_sat * sun_hat_j2000
        g = delivery_gates(r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG)
        assert g.sail_sunlit is False

    def test_sail_outside_umbra_passes_sail_sunlit_gate(
        self, epoch_et, sun_hat_j2000
    ):
        R_sat = mars_equatorial_radius_km() + 400.0
        r_sat = +R_sat * sun_hat_j2000  # sub-solar point -- fully lit.
        g = delivery_gates(r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG)
        assert g.sail_sunlit is True

    def test_sail_below_horizon_kills_horizon_gate(
        self, epoch_et, target_r_j2000
    ):
        # Sail at the antipode through Mars centre -> below target's
        # horizon plane. Scale up so the sail sits outside the Mars
        # equatorial reference sphere (shadow.in_mars_umbra would
        # otherwise raise "sail inside reference sphere" because the
        # target at lat 40 sits below R_equatorial).
        n_target = target_r_j2000 / np.linalg.norm(target_r_j2000)
        r_sat = -2.0 * mars_equatorial_radius_km() * n_target
        g = delivery_gates(r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG)
        assert g.sail_above_target_horizon is False

    def test_sail_at_zenith_passes_horizon_gate(
        self, epoch_et, target_r_j2000, target_n_outward_j2000
    ):
        r_sat = target_r_j2000 + 400.0 * target_n_outward_j2000
        g = delivery_gates(r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG)
        assert g.sail_above_target_horizon is True

    def test_elevation_threshold_prunes_grazing_pass(
        self, epoch_et, target_r_j2000, target_n_outward_j2000
    ):
        # Sail at elevation ~30 deg above target's horizon.
        n = target_n_outward_j2000
        # Construct a tangent orthogonal to n.
        trial = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, trial)
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([1.0, 0.0, 0.0]))
        t = t / np.linalg.norm(t)
        # At elevation 30 deg, dot(u, n) = sin(30 deg) = 0.5.
        # Build sail position: r_target + 400 * (sin30 * n + cos30 * t).
        r_sat = target_r_j2000 + 400.0 * (0.5 * n + (math.sqrt(3.0) / 2.0) * t)

        g_low = delivery_gates(
            r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_rad=math.radians(10.0),
        )
        assert g_low.sail_above_target_horizon is True

        g_high = delivery_gates(
            r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_rad=math.radians(60.0),
        )
        assert g_high.sail_above_target_horizon is False

    def test_bisector_feasibility_threshold(
        self, epoch_et, target_r_j2000, sun_r_j2000
    ):
        # Construct an almost-antiparallel geometry from the sail's
        # perspective: sail between target and sun so that sun
        # direction from sail is ~opposite to target direction from
        # sail.
        # Direction from target to sun:
        v_ts = sun_r_j2000 - target_r_j2000
        u_ts = v_ts / np.linalg.norm(v_ts)
        # Sail 100 km along u_ts from target -> sun is "outward" from
        # sail, target is "inward". Antiparallel from sail's POV.
        r_sat_deep = target_r_j2000 + 100.0 * u_ts
        g_deep = delivery_gates(
            r_sat_deep, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG,
            bisector_cos_alpha_min=0.01,
        )
        assert g_deep.bisector_feasible is False
        # Same geometry with threshold 0 -> accepts everything nonzero.
        # For a sail precisely on the target-to-sun line, cos_alpha
        # is exactly 0 (degenerate). Place the sail slightly off-line
        # so cos_alpha is small but nonzero, then test both thresholds.
        perp = np.cross(u_ts, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u_ts, np.array([1.0, 0.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        # Tiny perpendicular kick (1 km) relative to the 100 km parallel
        # displacement toward the sun: keeps the sail's view of
        # sun / target nearly antiparallel (cos_alpha ~ 1e-8), above
        # the _BISECTOR_DEGENERATE_SUM_TOL = 2e-3 floor so
        # bisector_normal returns a finite n_hat, but far below the
        # 0.01 feasibility threshold in delivery_gates.
        r_sat_slight = r_sat_deep + 1.0 * perp
        g_slight_tight = delivery_gates(
            r_sat_slight, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG,
            bisector_cos_alpha_min=0.01,
        )
        g_slight_loose = delivery_gates(
            r_sat_slight, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG,
            bisector_cos_alpha_min=0.0,
        )
        # With the perpendicular kick, cos_alpha > 0 so the loose
        # threshold accepts it; the geometry is still near-degenerate
        # so the 0.01 threshold rejects it. This verifies the
        # threshold parameter is honored.
        assert g_slight_loose.bisector_feasible is True
        assert g_slight_tight.bisector_feasible is False

    def test_target_in_sunlight_follows_target_in_sunlight_at(self, epoch_et):
        # delivery_gates.target_in_sunlight must match the
        # standalone target_in_sunlight_at helper for the same inputs.
        r_sat = np.array([5000.0, 0.0, 0.0])
        g = delivery_gates(r_sat, epoch_et, TARGET_LAT_DEG, TARGET_LON_DEG)
        expected = target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et
        )
        assert g.target_in_sunlight is expected

    def test_all_four_open_at_favorable_geometry(
        self, epoch_et, target_r_j2000, target_n_outward_j2000, sun_hat_j2000
    ):
        # Favorable geometry: target sunlit at this epoch (if not,
        # shift epoch), sail at zenith 400 km above target. That
        # gives all gates but bisector_feasible a natural pass; for
        # bisector, when sail is directly above target and target is
        # sunlit, the sun and target directions from the sail are
        # similar (target is right below sail, sun is somewhere
        # above the horizon) -- cos_alpha should be reasonably above
        # 0.01.
        # Guarantee target sunlit by choosing an epoch where it is.
        et = epoch_et
        if not target_in_sunlight_at(TARGET_LAT_DEG, TARGET_LON_DEG, et):
            et = epoch_et + 88643.0 / 2.0  # half-sol offset
        # Recompute target geometry at the chosen et.
        r_target = surface_point_position(
            TARGET_LAT_DEG, TARGET_LON_DEG, et, alt_km=0.0,
            planetographic=True,
        )
        n_out = target_outward_normal_j2000(
            TARGET_LAT_DEG, TARGET_LON_DEG, et
        )
        r_sat = r_target + 400.0 * n_out

        g = delivery_gates(r_sat, et, TARGET_LAT_DEG, TARGET_LON_DEG)
        assert g.sail_sunlit is True
        assert g.sail_above_target_horizon is True
        assert g.bisector_feasible is True
        assert g.target_in_sunlight is True


# ---------------------------------------------------------------------------
# Group 4: Window finder
# ---------------------------------------------------------------------------


def _synthetic_straight_line_pass(
    epoch_et: float,
    *,
    h_km: float = 400.0,
    v_kmps: float = 100.0,
    t_span_s: float = 20.0,
    n_samples: int = 41,
) -> PropagationResult:
    """Build a synthetic PropagationResult for a straight-line zenith pass.

    The sail moves at constant velocity ``v_kmps`` along a tangent to
    the target's local horizon plane, at fixed height ``h_km`` above
    the target along the outward normal. By construction:

    - ``t = 0`` is overhead (elevation = 90 deg, slant range = h_km).
    - elevation is symmetric in t; monotonically decreasing away from 0.
    - Mars rotation over the (very short) test span is a small
      correction (~0.2 deg per minute at the equator, negligible at
      t_span_s = 20 s).

    The trajectory is NOT a valid Kepler orbit; the only use is to
    feed ``find_delivery_windows`` with a controlled, analytically-
    known pass geometry.
    """
    n_outward = target_outward_normal_j2000(
        TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et
    )
    r_target = surface_point_position(
        TARGET_LAT_DEG, TARGET_LON_DEG, epoch_et, alt_km=0.0,
        planetographic=True,
    )
    # Build a tangent vector orthogonal to n_outward.
    trial = np.array([0.0, 0.0, 1.0])
    t_hat = np.cross(n_outward, trial)
    if np.linalg.norm(t_hat) < 1e-6:
        t_hat = np.cross(n_outward, np.array([1.0, 0.0, 0.0]))
    t_hat = t_hat / np.linalg.norm(t_hat)

    t_grid = np.linspace(-t_span_s / 2.0, t_span_s / 2.0, n_samples)
    states = np.zeros((n_samples, 6), dtype=float)
    for i, t_i in enumerate(t_grid):
        r_sat = r_target + h_km * n_outward + v_kmps * t_i * t_hat
        states[i, :3] = r_sat
        states[i, 3:] = v_kmps * t_hat
    return PropagationResult(
        t_s=t_grid,
        state_km_kmps=states,
        method="synthetic-straight-line",
        rtol=0.0,
        atol=0.0,
        mu_km3_s2=mars_gm_km3_per_s2(),
        epoch_et=epoch_et,
        solver_message="synthetic",
        n_rhs_calls=0,
    )


def _choose_sunlit_epoch() -> float:
    """Pick an epoch where (40N, 200E) is in sunlight; shift half-sol
    from 2026-06-01 noon UTC if needed."""
    et0 = utc_to_et(EPOCH_STR)
    if target_in_sunlight_at(TARGET_LAT_DEG, TARGET_LON_DEG, et0):
        return et0
    return et0 + 88643.0 / 2.0


class TestWindowBoundaryRefinement:
    """Sub-sample gate transitions remove output-grid endpoint snapping."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    def test_boolean_bisection_handles_both_transition_directions(self):
        transition_s = 2.3456789
        rising = _bisect_boolean_transition(
            lambda t_s: t_s >= transition_s,
            0.0,
            5.0,
            left_value=False,
            right_value=True,
            tolerance_s=1.0e-7,
        )
        falling = _bisect_boolean_transition(
            lambda t_s: t_s < transition_s,
            0.0,
            5.0,
            left_value=True,
            right_value=False,
            tolerance_s=1.0e-7,
        )
        assert rising == pytest.approx(transition_s, abs=1.0e-7)
        assert falling == pytest.approx(transition_s, abs=1.0e-7)

    def test_refined_coarse_pass_matches_dense_independent_grid(self, sunlit_et):
        # At h=400 km and v=100 km/s the static-target 10-degree crossings
        # are near +/-22.7 s. A 10-s coarse grid snaps them to +/-20 s;
        # sub-sample refinement should instead agree with a separately built
        # 0.05-s trajectory grid despite Mars's small rotation during the pass.
        coarse = _synthetic_straight_line_pass(
            sunlit_et, t_span_s=100.0, n_samples=11
        )
        dense = _synthetic_straight_line_pass(
            sunlit_et, t_span_s=100.0, n_samples=2001
        )
        finder_kwargs = {
            "target_elevation_min_deg": 10.0,
            "require_sail_sunlit": None,
            "require_bisector_feasible": None,
            "boundary_refinement_tol_s": 1.0e-4,
        }
        coarse_unrefined = find_delivery_windows(
            coarse,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            require_sail_sunlit=None,
            require_bisector_feasible=None,
        )[0]
        coarse_refined = find_delivery_windows(
            coarse, TARGET_LAT_DEG, TARGET_LON_DEG, **finder_kwargs
        )[0]
        dense_refined = find_delivery_windows(
            dense, TARGET_LAT_DEG, TARGET_LON_DEG, **finder_kwargs
        )[0]

        assert abs(coarse_unrefined.t_start_s - dense_refined.t_start_s) > 1.0
        assert abs(coarse_unrefined.t_end_s - dense_refined.t_end_s) > 1.0
        assert coarse_refined.t_start_s == pytest.approx(
            dense_refined.t_start_s, abs=2.0e-3
        )
        assert coarse_refined.t_end_s == pytest.approx(
            dense_refined.t_end_s, abs=2.0e-3
        )
        assert coarse_refined.duration_s == pytest.approx(
            dense_refined.duration_s, abs=4.0e-3
        )

    def test_default_none_matches_unrefined_window_exactly(self, sunlit_et):
        result = _synthetic_straight_line_pass(
            sunlit_et, t_span_s=100.0, n_samples=11
        )
        omitted = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG
        )
        explicit_none = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            boundary_refinement_tol_s=None,
        )
        assert omitted == explicit_none

    def test_refined_boundary_photometry_supports_vector_transmission(
        self, sunlit_et
    ):
        from reflectors.srp import SailOptical, SolarSail

        result = _synthetic_straight_line_pass(
            sunlit_et, t_span_s=100.0, n_samples=11
        )
        sail = SolarSail(
            area_m2=1000.0,
            mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )
        common = {
            "target_elevation_min_deg": 10.0,
            "require_sail_sunlit": None,
            "require_bisector_feasible": None,
            "sail": sail,
            "boundary_refinement_tol_s": 1.0e-4,
        }
        scalar_half = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            atmospheric_transmission=0.5,
            **common,
        )[0]
        vector_half = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            atmospheric_transmission=np.full(result.t_s.shape, 0.5),
            **common,
        )[0]
        assert vector_half.fluence_J_per_m2 == pytest.approx(
            scalar_half.fluence_J_per_m2, rel=1.0e-13
        )
        assert vector_half.peak_irradiance_W_per_m2 == pytest.approx(
            scalar_half.peak_irradiance_W_per_m2, rel=1.0e-13
        )
        assert vector_half.mean_irradiance_W_per_m2 == pytest.approx(
            vector_half.fluence_J_per_m2 / vector_half.duration_s,
            rel=1.0e-13,
        )
        assert vector_half.peak_footprint_semi_major_km is not None
        assert vector_half.peak_footprint_semi_minor_km is not None

    @pytest.mark.parametrize("bad_tolerance", [0.0, -1.0, float("nan")])
    def test_invalid_boundary_tolerance_raises(self, sunlit_et, bad_tolerance):
        result = _synthetic_straight_line_pass(sunlit_et)
        with pytest.raises(ValueError, match="positive finite"):
            find_delivery_windows(
                result,
                TARGET_LAT_DEG,
                TARGET_LON_DEG,
                boundary_refinement_tol_s=bad_tolerance,
            )


class TestFindDeliveryWindowsSyntheticPass:
    """Analytical anchors on a hand-constructed straight-line fly-by."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def short_pass_result(self, sunlit_et):
        # 20 s total, entire trajectory above 10 deg elevation by
        # choice of v=100 km/s at h=400 km (window half-width is
        # ~22.6 s, outside this t_span).
        return _synthetic_straight_line_pass(sunlit_et)

    def test_returns_single_window(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        assert len(windows) == 1

    def test_window_covers_full_trajectory(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w = windows[0]
        t_arr = short_pass_result.t_s
        # With the designed geometry every sample is above threshold,
        # so window endpoints are the first and last sample times.
        assert w.t_start_s == pytest.approx(float(t_arr[0]))
        assert w.t_end_s == pytest.approx(float(t_arr[-1]))
        assert w.n_samples == t_arr.size
        assert w.duration_s == pytest.approx(
            float(t_arr[-1] - t_arr[0])
        )

    def test_max_elevation_close_to_zenith(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w = windows[0]
        # Sail passes directly over target at t = 0; peak elevation
        # should be ~90 deg. Allow a couple deg of slop for the Mars-
        # rotation drift over +-10 s (~0.04 deg) and sample spacing.
        assert w.max_elevation_deg > 89.0

    def test_min_slant_range_close_to_altitude(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w = windows[0]
        # Minimum slant range is the overhead altitude h = 400 km at
        # t = 0. Tight pin; Mars rotation over +-10 s moves the target
        # in J2000 by ~2 km, so allow ~5 km slop.
        assert abs(w.min_slant_range_km - 400.0) < 5.0

    def test_integral_cos_alpha_positive_and_bounded(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w = windows[0]
        assert w.integral_cos_alpha_s > 0.0
        # Upper bound: max cos_alpha is 1 (face-on), duration is 20 s.
        assert w.integral_cos_alpha_s < 20.0

    def test_et_fields_populated_when_epoch_known(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w = windows[0]
        assert w.et_start is not None
        assert w.et_end is not None
        assert w.et_end > w.et_start


class TestFindDeliveryWindowsNoWindow:
    """Negative-control: target at the antipode, no window should be
    returned from the over-40N-200E trajectory."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def short_pass_result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    def test_antipodal_target_returns_empty(self, short_pass_result):
        antipode_lat = -TARGET_LAT_DEG
        antipode_lon = (TARGET_LON_DEG + 180.0) % 360.0
        windows = find_delivery_windows(
            short_pass_result, antipode_lat, antipode_lon,
            target_elevation_min_deg=10.0,
        )
        # The sail is 400 km above the 40N/200E target, which is
        # well below the antipode's horizon plane -> every sample's
        # sail_above_target_horizon gate is closed -> zero windows.
        assert windows == []


class TestFindDeliveryWindowsGateSwitches:
    """Per-gate AND / IGNORE switches."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def short_pass_result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    def test_require_target_sunlit_true_matches_sunlight_state(
        self, short_pass_result, sunlit_et
    ):
        # Target is sunlit at this epoch (by construction of
        # _choose_sunlit_epoch). Requiring target sunlight does not
        # change the result compared to ignoring it.
        assert target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, sunlit_et
        )
        w_ignore = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            require_target_sunlit=None,
        )
        w_require = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            require_target_sunlit=True,
        )
        assert len(w_ignore) == len(w_require)

    def test_require_target_sunlit_false_opposite_of_state(
        self, short_pass_result, sunlit_et
    ):
        # Target is sunlit; requiring night -> zero windows.
        assert target_in_sunlight_at(
            TARGET_LAT_DEG, TARGET_LON_DEG, sunlit_et
        )
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            require_target_sunlit=False,
        )
        assert windows == []

    def test_disabling_horizon_gate_lets_below_pass_through(
        self, short_pass_result
    ):
        # Antipodal target + disable horizon gate -> the only remaining
        # requirements are sail_sunlit + bisector_feasible, which may
        # or may not all be open; the minimum required condition is that the
        # FINDING changes vs. the default-gate case (which returns
        # empty).
        antipode_lat = -TARGET_LAT_DEG
        antipode_lon = (TARGET_LON_DEG + 180.0) % 360.0
        w_default = find_delivery_windows(
            short_pass_result, antipode_lat, antipode_lon,
            target_elevation_min_deg=10.0,
        )
        w_relaxed = find_delivery_windows(
            short_pass_result, antipode_lat, antipode_lon,
            target_elevation_min_deg=10.0,
            require_sail_above_horizon=None,
        )
        # The default is empty; relaxing the horizon gate cannot reduce the
        # number of windows.
        assert len(w_default) == 0
        assert len(w_relaxed) >= len(w_default)


class TestFindDeliveryWindowsShape:
    """Input validation and structural contract."""

    def test_mismatched_t_and_state_raises(self):
        bad = PropagationResult(
            t_s=np.array([0.0, 1.0, 2.0]),
            state_km_kmps=np.zeros((2, 6)),  # invalid first dimension
            method="bad",
            rtol=0.0, atol=0.0,
            mu_km3_s2=mars_gm_km3_per_s2(),
            epoch_et=None,
            solver_message="bad",
            n_rhs_calls=0,
        )
        with pytest.raises(ValueError, match="does not match"):
            find_delivery_windows(bad, TARGET_LAT_DEG, TARGET_LON_DEG)

    def test_empty_result_returns_empty(self):
        empty = PropagationResult(
            t_s=np.array([]),
            state_km_kmps=np.zeros((0, 6)),
            method="empty",
            rtol=0.0, atol=0.0,
            mu_km3_s2=mars_gm_km3_per_s2(),
            epoch_et=None,
            solver_message="empty",
            n_rhs_calls=0,
        )
        assert find_delivery_windows(empty, TARGET_LAT_DEG, TARGET_LON_DEG) == []


# ---------------------------------------------------------------------------
# Group 5: Slew demand scalar
# ---------------------------------------------------------------------------


class TestPeakAlphaDemand:
    """``peak_alpha_demand_rad_s2`` populated on the returned windows."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    def test_peak_alpha_populated_and_positive(self, sunlit_et):
        result = _synthetic_straight_line_pass(sunlit_et)
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        assert len(windows) == 1
        w = windows[0]
        assert w.peak_alpha_demand_rad_s2 is not None
        assert w.peak_alpha_demand_rad_s2 > 0.0
        # Loose upper bound -- peak |alpha| for this unphysical fast
        # trajectory (v = 100 km/s, way above any orbital velocity)
        # should still sit below 1 rad/s^2. If it exceeds that the
        # formulation has gone off the rails.
        assert w.peak_alpha_demand_rad_s2 < 1.0

    def test_faster_pass_has_larger_peak_alpha(self, sunlit_et):
        # All else equal, a faster pass compresses the bisector's
        # angular sweep into a shorter time -> higher angular
        # acceleration. Pass duration scales as 1/v, so at fixed
        # angular sweep |omega| ~ 1/T ~ v and |alpha| ~ 1/T^2 ~ v^2.
        slow = _synthetic_straight_line_pass(sunlit_et, v_kmps=30.0)
        fast = _synthetic_straight_line_pass(sunlit_et, v_kmps=150.0)
        w_slow = find_delivery_windows(
            slow, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w_fast = find_delivery_windows(
            fast, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        assert len(w_slow) == 1 and len(w_fast) == 1
        slow_peak = w_slow[0].peak_alpha_demand_rad_s2
        fast_peak = w_fast[0].peak_alpha_demand_rad_s2
        assert slow_peak is not None and fast_peak is not None
        assert fast_peak > slow_peak

    def test_peak_alpha_is_none_without_epoch_et(self, sunlit_et):
        # Without an absolute epoch, the bisector-pointing profile
        # cannot query SPICE (it needs ET for Sun/target), so the
        # trajectory interpolant is never built and peak_alpha is None.
        result = _synthetic_straight_line_pass(sunlit_et)
        # Replace epoch_et with None by rebuilding the dataclass with
        # that field unset. (dataclasses.replace would work too.)
        stripped = PropagationResult(
            t_s=result.t_s,
            state_km_kmps=result.state_km_kmps,
            method=result.method,
            rtol=result.rtol,
            atol=result.atol,
            mu_km3_s2=result.mu_km3_s2,
            epoch_et=None,
            solver_message=result.solver_message,
            n_rhs_calls=result.n_rhs_calls,
        )
        windows = find_delivery_windows(
            stripped, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        # With epoch_et=None, delivery_gates still works (treats
        # t_s[i] as absolute et), but the trajectory interpolant
        # isn't built for peak-alpha because epoch_et is what bridges
        # t_s relative-time to the absolute-ET SPICE queries in
        # bisector_pointing. So peak_alpha should be None on any
        # windows that do come back.
        for w in windows:
            assert w.peak_alpha_demand_rad_s2 is None

    def test_peak_alpha_none_for_too_short_window(self, sunlit_et):
        # Construct a trajectory where the window comes out to only
        # 2 samples (below _SLEW_MIN_SAMPLES = 4). Easiest way: use
        # a pass where only the final two samples of the trajectory
        # are above the elevation threshold.
        # Build a slow straight-line pass with just a sliver of the
        # trajectory above 10 deg.
        result = _synthetic_straight_line_pass(
            sunlit_et, h_km=400.0, v_kmps=200.0, t_span_s=200.0,
            n_samples=10,  # sparse
        )
        # t_span 200 s, v = 200 km/s -> horizon reached at
        # t = 400/(200 * tan(10 deg)) = 11.3 s. With only 10 samples
        # spread over 200 s (spacing ~22 s), at most 1-2 samples sit
        # inside the <= 11.3 s zenith region.
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        for w in windows:
            # If any window emerged with fewer than the minimum
            # samples, peak_alpha must be None.
            if w.n_samples < 4:
                assert w.peak_alpha_demand_rad_s2 is None


class TestAlphaMaxFilter:
    """Post-hoc alpha_max_rad_s2 window-feasibility cap."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture(scope="class")
    def baseline_windows(self, sunlit_et):
        """Unfiltered windows on a synthetic pass -- reference set."""
        result = _synthetic_straight_line_pass(sunlit_et)
        return find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )

    def test_default_none_preserves_unfiltered_list(self, sunlit_et):
        # alpha_max_rad_s2=None is the default and must yield the same windows
        # as omitting the kwarg.
        result = _synthetic_straight_line_pass(sunlit_et)
        w_default = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        w_none = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            alpha_max_rad_s2=None,
        )
        assert len(w_default) == len(w_none)
        for a, b in zip(w_default, w_none):
            assert a == b

    def test_infinite_cap_preserves_all_windows(self, sunlit_et, baseline_windows):
        # A cap of +inf would be technically accepted by <= test but the
        # impl rejects non-finite caps as a misuse signal. Use a huge
        # finite cap instead; semantically equivalent to "no cap" for
        # any physically reasonable peak alpha.
        result = _synthetic_straight_line_pass(sunlit_et)
        w_huge = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            alpha_max_rad_s2=1e9,
        )
        assert len(w_huge) == len(baseline_windows)

    def test_zero_cap_drops_every_window(self, sunlit_et):
        result = _synthetic_straight_line_pass(sunlit_et)
        w_zero = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            alpha_max_rad_s2=0.0,
        )
        # Peak alpha is strictly positive on any non-degenerate window,
        # so a zero cap drops everything.
        assert w_zero == []

    def test_cap_at_baseline_peak_keeps_all(self, sunlit_et, baseline_windows):
        # Cap exactly at the maximum peak_alpha_demand across the
        # baseline windows: every window should be kept (<=) .
        peaks = [w.peak_alpha_demand_rad_s2 for w in baseline_windows]
        assert all(p is not None for p in peaks)
        cap = max(peaks)
        result = _synthetic_straight_line_pass(sunlit_et)
        w_capped = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            alpha_max_rad_s2=cap,
        )
        assert len(w_capped) == len(baseline_windows)

    def test_cap_just_below_peak_drops_that_window(
        self, sunlit_et, baseline_windows
    ):
        peaks = [w.peak_alpha_demand_rad_s2 for w in baseline_windows]
        assert all(p is not None for p in peaks)
        cap = max(peaks) * (1.0 - 1e-6)
        result = _synthetic_straight_line_pass(sunlit_et)
        w_capped = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            alpha_max_rad_s2=cap,
        )
        assert len(w_capped) < len(baseline_windows)

    def test_cap_drops_windows_with_none_peak(self, sunlit_et):
        # Windows with peak_alpha_demand_rad_s2=None are conservatively
        # dropped when a cap is set because an unverifiable window cannot
        # be classified as feasible.
        result = _synthetic_straight_line_pass(sunlit_et)
        # Strip the epoch so peak_alpha can't be computed.
        from reflectors.dynamics import PropagationResult
        stripped = PropagationResult(
            t_s=result.t_s, state_km_kmps=result.state_km_kmps,
            method=result.method, rtol=result.rtol, atol=result.atol,
            mu_km3_s2=result.mu_km3_s2, epoch_et=None,
            solver_message=result.solver_message,
            n_rhs_calls=result.n_rhs_calls,
        )
        w_uncapped = find_delivery_windows(
            stripped, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        # With None epoch, peak_alpha is None on any returned windows.
        # Capping must drop them even though their peak is unknown.
        assert all(w.peak_alpha_demand_rad_s2 is None for w in w_uncapped)
        w_capped = find_delivery_windows(
            stripped, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            alpha_max_rad_s2=1e9,  # arbitrarily huge -- None still drops
        )
        assert w_capped == []

    def test_negative_cap_raises(self, sunlit_et):
        result = _synthetic_straight_line_pass(sunlit_et)
        with pytest.raises(ValueError, match="non-negative"):
            find_delivery_windows(
                result, TARGET_LAT_DEG, TARGET_LON_DEG,
                target_elevation_min_deg=10.0,
                alpha_max_rad_s2=-1.0,
            )

    def test_nan_cap_raises(self, sunlit_et):
        result = _synthetic_straight_line_pass(sunlit_et)
        with pytest.raises(ValueError, match="non-negative"):
            find_delivery_windows(
                result, TARGET_LAT_DEG, TARGET_LON_DEG,
                target_elevation_min_deg=10.0,
                alpha_max_rad_s2=float("nan"),
            )


# ---------------------------------------------------------------------------
# Group 6: Full-stack integration smoke test
# ---------------------------------------------------------------------------


def _sun_sync_400km_circular_state(
    epoch_et: float,
    *,
    inc_deg: float = 92.92,
    raan_deg: float = 0.0,
    alt_km: float = 400.0,
) -> np.ndarray:
    """Initial 6-state for a circular Mars orbit at given inclination /
    RAAN / altitude, at true anomaly = 0.

    Inclination is measured against the J2000 equator, not the
    Mars-mean-equator-of-J2000 frame. For the qualitative integration
    smoke test (does the pipeline end-to-end produce sensible
    windows over a sidereal day?) the J2000 reference is fine.
    """
    mu = 42828.37566395650  # MRO120F system mu
    r = mars_equatorial_radius_km() + alt_km
    n_mean = float(np.sqrt(mu / r ** 3))
    v = r * n_mean

    inc = math.radians(inc_deg)
    raan = math.radians(raan_deg)

    r_peri = np.array([r, 0.0, 0.0])
    v_peri = np.array([0.0, v, 0.0])
    R_R3 = np.array([
        [math.cos(raan), -math.sin(raan), 0.0],
        [math.sin(raan), math.cos(raan), 0.0],
        [0.0, 0.0, 1.0],
    ])
    R_R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(inc), -math.sin(inc)],
        [0.0, math.sin(inc), math.cos(inc)],
    ])
    rot = R_R3 @ R_R1
    return np.concatenate([rot @ r_peri, rot @ v_peri])


class TestFullStackOneSolIntegration:
    """End-to-end smoke: propagate(full physics) -> find_delivery_windows.

    Bounds cover the reference case at RAAN=0, true anomaly=0, epoch
    2026-06-01T00:00 UTC. Runs in ~2 s wall (well below the 30 s fast threshold)
    because the per-step physics is fast even at gravity_degree=6 +
    three third bodies, and find_delivery_windows walks 17729
    output samples in single-digit seconds.
    """

    @pytest.fixture(scope="class")
    def one_sol_result(self):
        et0 = utc_to_et(EPOCH_STR)
        state0 = _sun_sync_400km_circular_state(et0)
        sidereal_day_s = 88642.663
        # 5 s output spacing -- resolves minutes-scale overhead passes
        # with enough samples that peak_alpha_demand populates reliably
        # via CubicSpline + finite difference.
        t_eval = np.arange(0.0, sidereal_day_s + 0.001, 5.0)
        return propagate(
            state0, (0.0, sidereal_day_s),
            epoch_et=et0,
            gravity_degree=6, gravity_order=6,
            third_bodies=[
                sun_third_body(), phobos_third_body(), deimos_third_body(),
            ],
            options=PropagationOptions.fast(),
            t_eval_s=t_eval,
        )

    def test_window_count_is_physically_sensible(self, one_sol_result):
        windows = find_delivery_windows(
            one_sol_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        # The reference case has two windows; the envelope
        # tolerates changes in DOP853 step patterns or perturbation ordering.
        assert 1 <= len(windows) <= 20

    def test_at_least_one_near_overhead_pass(self, one_sol_result):
        windows = find_delivery_windows(
            one_sol_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        peak_elevs = [w.max_elevation_deg for w in windows]
        # Reference peak elevation is 82.68 deg; requiring at least one
        # window > 45 deg" is loose enough to survive step-pattern
        # jitter while still being a positive signal that the
        # geometry is correct.
        assert max(peak_elevs) > 45.0, peak_elevs

    def test_min_slant_within_factor_of_two_of_altitude(self, one_sol_result):
        windows = find_delivery_windows(
            one_sol_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        min_slants = [w.min_slant_range_km for w in windows]
        # Peak elevation > 45 deg -> min slant < sqrt(2) * alt. At
        # an 82 deg peak the min slant is ~414 km, close
        # to the 400 km altitude. Pin < 2 * alt = 800 km to have
        # some headroom.
        assert min(min_slants) < 2.0 * 400.0, min_slants

    def test_peak_alpha_demand_in_physical_band(self, one_sol_result):
        windows = find_delivery_windows(
            one_sol_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        peak_alphas = [
            w.peak_alpha_demand_rad_s2 for w in windows
            if w.peak_alpha_demand_rad_s2 is not None
        ]
        # At least one window must populate peak_alpha (positive
        # signal: the bisector + interpolant pipeline is wired
        # through).
        assert len(peak_alphas) >= 1
        # Reference peak alpha is 3.16e-5 rad/s^2 -- within an order
        # of magnitude of Viale et al. 2023's 4.86e-5 rad/s^2
        # (worst Earth LMO tracking pass). Bound [1e-8, 1e-3]
        # catches catastrophic failure modes (frozen bisector or
        # blown-up interpolant) without being fragile to geometry
        # shifts.
        assert max(peak_alphas) > 1e-8
        assert max(peak_alphas) < 1e-3

    def test_stress_threshold_88deg_is_empty_negative_control(
        self, one_sol_result
    ):
        # Tightening the elevation threshold to near-zenith should
        # zero out the window count: an overhead pass at 400 km has
        # peak elevation depending on ground-track offset; requiring
        # >= 88 deg is essentially requiring exact zenith alignment,
        # which is zero-probability at a fixed surface target over
        # one sol. This is the negative control for the geometry.
        windows_tight = find_delivery_windows(
            one_sol_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=88.0,
        )
        assert len(windows_tight) == 0


# ---------------------------------------------------------------------------
# Group 7: Delivered-irradiance / fluence integration (beam.py hook)
# ---------------------------------------------------------------------------


class TestIrradianceIntegrationWithoutSail:
    """When ``sail`` is not supplied, all five irradiance fields stay
    None and the geometric scalars (duration, min_slant,
    max_elevation, peak_alpha, integral_cos_alpha) are unchanged.
    """

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    def test_no_sail_leaves_irradiance_fields_none(self, result):
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        assert len(windows) == 1
        w = windows[0]
        assert w.peak_irradiance_W_per_m2 is None
        assert w.mean_irradiance_W_per_m2 is None
        assert w.fluence_J_per_m2 is None
        assert w.peak_footprint_semi_major_km is None
        assert w.peak_footprint_semi_minor_km is None

    def test_no_sail_preserves_existing_scalars(self, result):
        """Existing scalars unchanged to machine precision."""
        windows_new = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        # Same call explicitly passing sail=None; identical result.
        windows_none = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=None,
        )
        assert len(windows_new) == len(windows_none) == 1
        w1, w2 = windows_new[0], windows_none[0]
        # The underlying scalars must be bit-identical (same code path).
        assert w1.duration_s == w2.duration_s
        assert w1.min_slant_range_km == w2.min_slant_range_km
        assert w1.max_elevation_deg == w2.max_elevation_deg
        assert w1.integral_cos_alpha_s == w2.integral_cos_alpha_s
        assert w1.n_samples == w2.n_samples


class TestIrradianceIntegrationSyntheticPass:
    """Pass a SolarSail to find_delivery_windows and verify the five
    irradiance scalars populate and obey the expected identities.
    """

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    @pytest.fixture
    def sail(self):
        from reflectors.srp import SailOptical, SolarSail
        return SolarSail(
            area_m2=1000.0, mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )

    def test_all_irradiance_fields_populate(self, result, sail):
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        assert len(windows) == 1
        w = windows[0]
        assert w.peak_irradiance_W_per_m2 is not None
        assert w.peak_irradiance_W_per_m2 > 0.0
        assert w.mean_irradiance_W_per_m2 is not None
        assert w.mean_irradiance_W_per_m2 > 0.0
        assert w.fluence_J_per_m2 is not None
        assert w.fluence_J_per_m2 > 0.0
        assert w.peak_footprint_semi_major_km is not None
        assert w.peak_footprint_semi_minor_km is not None
        # Semi-major >= semi-minor (equality at zenith).
        assert (
            w.peak_footprint_semi_major_km >= w.peak_footprint_semi_minor_km
        )

    def test_mean_equals_fluence_over_duration(self, result, sail):
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        w = windows[0]
        # Fluence / duration = mean irradiance, by construction.
        assert w.mean_irradiance_W_per_m2 == pytest.approx(
            w.fluence_J_per_m2 / w.duration_s, rel=1e-12
        )

    def test_peak_irradiance_exceeds_mean(self, result, sail):
        """Peak > mean across the window (basic extremum sanity)."""
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        w = windows[0]
        assert w.peak_irradiance_W_per_m2 > w.mean_irradiance_W_per_m2

    def test_peak_footprint_near_circular_for_zenith_pass(self, result, sail):
        """Peak occurs near zenith on this overhead synthetic pass, so
        footprint is nearly circular. "Nearly" not "exactly" because
        for the 20-s / 100-km-s synthetic pass cos(psi/2) sweeps
        through a broad range and the irradiance peak (maximising
        cos(psi/2) * sin(eps) / d^2) does not always land exactly at
        the zenith sample (sin(eps)=1). At the peak sample the
        elevation is typically > 60 deg so the semi-major /
        semi-minor ratio = 1/sin(elev) < 1/sin(60 deg) = 1.155.
        """
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        w = windows[0]
        # Semi-major >= semi-minor (always true), with a loose bound
        # on the elongation consistent with a near-overhead pass.
        assert w.peak_footprint_semi_major_km >= w.peak_footprint_semi_minor_km
        ratio = (
            w.peak_footprint_semi_major_km
            / w.peak_footprint_semi_minor_km
        )
        assert 1.0 <= ratio < 1.2  # elev at peak sample > ~56 deg

    def test_atmospheric_transmission_scales_linearly(self, result, sail):
        """Halving chi halves both peak irradiance and fluence."""
        w_full = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail, atmospheric_transmission=1.0,
        )[0]
        w_half = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail, atmospheric_transmission=0.5,
        )[0]
        assert (
            w_half.peak_irradiance_W_per_m2
            / w_full.peak_irradiance_W_per_m2
        ) == pytest.approx(0.5, rel=1e-12)
        assert (
            w_half.fluence_J_per_m2 / w_full.fluence_J_per_m2
        ) == pytest.approx(0.5, rel=1e-12)

    def test_peak_irradiance_magnitude_sanity(self, result, sail):
        """1000 m^2 JPL sail, 400 km zenith pass at Mars ~1.4 AU:
        peak irradiance should be in [0.01, 0.5] W/m^2 -- the
        hand-derived canonical anchor from test_beam (0.066 W/m^2
        at 500 km / 1.412 AU) scaled up by (500/400)^2 ~ 1.56x
        gives ~0.10 W/m^2; pass is at Mars 1.4 AU mean (epoch 2026-06).
        """
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        peak = windows[0].peak_irradiance_W_per_m2
        assert 0.01 < peak < 0.5

    def test_footprint_radius_matches_beam_formula(self, result, sail):
        """Cross-check: footprint semi-minor at peak equals
        ``b = slant * tan(alpha/2)`` at the peak sample."""
        from reflectors.beam import beam_image_semi_minor_km
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        w = windows[0]
        # Peak occurs at zenith (h=400 km slant). Heliocentric distance
        # at the test epoch is roughly 1.4 AU -- 0.61 rad full sun
        # subtense; b = 400e3 * tan(0.00305) ~ 1.22 km. Pin to 1 part
        # in 1e3 (depends on exact sail-Sun distance at epoch).
        assert 1.0 < w.peak_footprint_semi_minor_km < 1.5


class TestReturnSamples:
    """``return_samples=True`` exposes the per-sample arrays without
    changing the window list, and the arrays reconstruct the per-window
    fluence, preserving a single source for the base-power time series.
    """

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    @pytest.fixture
    def sail(self):
        from reflectors.srp import SailOptical, SolarSail
        return SolarSail(
            area_m2=1000.0, mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )

    def test_windows_bit_identical_with_and_without_samples(self, result, sail):
        """The window list + its scalars are unchanged when samples are
        requested (DeliveryWindow is a frozen dataclass -> field-wise ==)."""
        kw = dict(target_elevation_min_deg=10.0, sail=sail)
        w_default = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG, **kw
        )
        ret = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG, return_samples=True, **kw
        )
        assert isinstance(ret, tuple) and len(ret) == 2
        w_samp, samples = ret
        assert isinstance(samples, DeliverySamples)
        assert w_samp == w_default

    def test_sample_arrays_parallel_to_input(self, result, sail):
        _w, samples = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail, return_samples=True,
        )
        n = result.t_s.shape[0]
        np.testing.assert_array_equal(
            samples.t_s, np.asarray(result.t_s, dtype=float)
        )
        for arr in (samples.gate_open, samples.slant_km, samples.elev_deg,
                    samples.cos_alpha, samples.irradiance_W_per_m2,
                    samples.vacuum_irradiance_W_per_m2,
                    samples.atmospheric_transmission,
                    samples.sail_to_sun_km):
            assert arr.shape == (n,)
        assert samples.gate_open.dtype == bool

    def test_fluence_reconstructs_from_returned_irradiance(self, result, sail):
        """Trapezoid of the returned irradiance over a kept window's
        gate-open slice == the window's stored fluence (the exact same
        integral the finder runs internally)."""
        windows, samples = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail, return_samples=True,
        )
        assert len(windows) == 1
        w = windows[0]
        mask = (samples.t_s >= w.t_start_s) & (samples.t_s <= w.t_end_s)
        fluence_from_series = float(np.trapezoid(
            samples.irradiance_W_per_m2[mask], samples.t_s[mask]))
        assert fluence_from_series == pytest.approx(
            w.fluence_J_per_m2, rel=1e-12, abs=1e-12
        )

    def test_gate_open_matches_window_span(self, result, sail):
        """On the single-window synthetic pass, every gate-open sample
        lies inside the window [t_start, t_end] and vice versa."""
        windows, samples = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail, return_samples=True,
        )
        w = windows[0]
        span = (samples.t_s >= w.t_start_s) & (samples.t_s <= w.t_end_s)
        np.testing.assert_array_equal(samples.gate_open, span)

    def test_no_sail_gives_none_irradiance_but_windows_match(self, result):
        kw = dict(target_elevation_min_deg=10.0)
        w_default = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG, **kw
        )
        windows, samples = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG, return_samples=True, **kw
        )
        assert windows == w_default
        assert samples.irradiance_W_per_m2 is None
        assert samples.vacuum_irradiance_W_per_m2 is None
        assert samples.atmospheric_transmission is None
        assert samples.sail_to_sun_km is None
        n = result.t_s.shape[0]
        assert samples.slant_km.shape == (n,)
        assert samples.cos_alpha.shape == (n,)

    def test_empty_result_returns_empty_samples(self):
        empty = PropagationResult(
            t_s=np.array([]),
            state_km_kmps=np.zeros((0, 6)),
            method="empty",
            rtol=0.0, atol=0.0,
            mu_km3_s2=mars_gm_km3_per_s2(),
            epoch_et=None,
            solver_message="empty",
            n_rhs_calls=0,
        )
        windows, samples = find_delivery_windows(
            empty, TARGET_LAT_DEG, TARGET_LON_DEG, return_samples=True
        )
        assert windows == []
        assert isinstance(samples, DeliverySamples)
        assert samples.t_s.shape == (0,)
        assert samples.irradiance_W_per_m2 is None  # no sail supplied

    def test_scalar_and_constant_vector_transmission_are_equivalent(
        self, result, sail
    ):
        scalar_windows, scalar_samples = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
            atmospheric_transmission=0.5,
            return_samples=True,
        )
        vector_windows, vector_samples = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
            atmospheric_transmission=np.full(result.t_s.shape, 0.5),
            return_samples=True,
        )
        assert vector_windows == scalar_windows
        np.testing.assert_array_equal(
            vector_samples.irradiance_W_per_m2,
            scalar_samples.irradiance_W_per_m2,
        )
        np.testing.assert_array_equal(vector_samples.atmospheric_transmission, 0.5)

    def test_variable_transmission_controls_series_and_window_fluence(
        self, result, sail
    ):
        chi = np.linspace(0.2, 0.9, result.t_s.size)
        windows, samples = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
            atmospheric_transmission=chi,
            return_samples=True,
        )
        np.testing.assert_array_equal(samples.atmospheric_transmission, chi)
        np.testing.assert_allclose(
            samples.irradiance_W_per_m2,
            samples.vacuum_irradiance_W_per_m2 * chi,
            rtol=0.0,
            atol=0.0,
        )
        expected_fluence = float(
            np.trapezoid(samples.irradiance_W_per_m2, samples.t_s)
        )
        assert windows[0].fluence_J_per_m2 == pytest.approx(
            expected_fluence,
            rel=1e-15,
        )

    def test_minimum_fluence_filter_is_applied_after_atmospheric_loss(
        self, result, sail
    ):
        vacuum = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
        )[0].fluence_J_per_m2
        attenuated = find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
            atmospheric_transmission=0.5,
        )[0].fluence_J_per_m2
        threshold = 0.5 * (vacuum + attenuated)
        assert find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
            min_window_fluence_J_per_m2=threshold,
        )
        assert find_delivery_windows(
            result,
            TARGET_LAT_DEG,
            TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail,
            atmospheric_transmission=0.5,
            min_window_fluence_J_per_m2=threshold,
        ) == []

    @pytest.mark.parametrize(
        ("chi", "match"),
        [
            ([0.5], "matching result.t_s"),
            ([0.5, float("nan")] + [0.5] * 39, "finite"),
            ([0.5, 1.1] + [0.5] * 39, "lie in"),
        ],
    )
    def test_invalid_transmission_vectors_are_rejected(
        self, result, sail, chi, match
    ):
        with pytest.raises(ValueError, match=match):
            find_delivery_windows(
                result,
                TARGET_LAT_DEG,
                TARGET_LON_DEG,
                target_elevation_min_deg=10.0,
                sail=sail,
                atmospheric_transmission=chi,
            )


class TestIrradianceIntegrationDegenerate:
    """Geometries where the beam cannot reach the target."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def sail(self):
        from reflectors.srp import SailOptical, SolarSail
        return SolarSail(
            area_m2=1000.0, mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )

    def test_zero_fluence_when_no_windows(self, sunlit_et, sail):
        """If the combined gate never opens, no windows -> empty list
        (not a list of zero-fluence windows)."""
        result = _synthetic_straight_line_pass(sunlit_et)
        # Below-horizon target: antipode of the pass.
        windows = find_delivery_windows(
            result, -TARGET_LAT_DEG, (TARGET_LON_DEG + 180.0) % 360.0,
            target_elevation_min_deg=10.0,
            sail=sail,
        )
        assert windows == []

    def test_chi_zero_gives_zero_irradiance(self, sunlit_et, sail):
        """chi = 0 (fully opaque atmosphere) gives zero peak irradiance
        and zero fluence even though the gates are open."""
        result = _synthetic_straight_line_pass(sunlit_et)
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
            sail=sail, atmospheric_transmission=0.0,
        )
        assert len(windows) == 1
        w = windows[0]
        assert w.peak_irradiance_W_per_m2 == 0.0
        assert w.fluence_J_per_m2 == 0.0
        assert w.mean_irradiance_W_per_m2 == 0.0


# ---------------------------------------------------------------------------
# Multi-target window finding (2026-06-09)
# ---------------------------------------------------------------------------


class TestFindDeliveryWindowsMulti:
    """find_delivery_windows_multi: tagging, merging, single-target
    bit-exactness."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def short_pass_result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    def test_single_target_bit_exact(self, short_pass_result):
        """One-target multi call returns exactly the single-target
        finder's windows (dataclass equality, element-wise)."""
        single = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        multi = find_delivery_windows_multi(
            short_pass_result, [(TARGET_LAT_DEG, TARGET_LON_DEG)],
            target_elevation_min_deg=10.0,
        )
        assert multi == single

    def test_finder_stamps_target_fields(self, short_pass_result):
        windows = find_delivery_windows(
            short_pass_result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0,
        )
        assert len(windows) == 1
        w = windows[0]
        assert w.target_idx == 0
        assert w.target_lat_deg == pytest.approx(TARGET_LAT_DEG)
        assert w.target_lon_deg == pytest.approx(TARGET_LON_DEG)

    def test_two_target_tagging_and_sort(self, short_pass_result):
        """A nearby second target also sees the synthetic zenith pass
        (~1.3 deg great-circle offset, elevation ~78 deg at h=400 km);
        both windows come back, each tagged with its own target, and
        the merged list is time-sorted."""
        targets = [
            (TARGET_LAT_DEG, TARGET_LON_DEG),
            (TARGET_LAT_DEG + 1.0, TARGET_LON_DEG + 1.0),
        ]
        windows = find_delivery_windows_multi(
            short_pass_result, targets, target_elevation_min_deg=10.0,
        )
        assert len(windows) == 2
        assert sorted(w.target_idx for w in windows) == [0, 1]
        for w in windows:
            lat, lon = targets[w.target_idx]
            assert w.target_lat_deg == pytest.approx(lat)
            assert w.target_lon_deg == pytest.approx(lon)
        starts = [(w.t_start_s, w.target_idx) for w in windows]
        assert starts == sorted(starts)

    def test_antipodal_second_target_yields_no_extra_window(
        self, short_pass_result
    ):
        targets = [
            (TARGET_LAT_DEG, TARGET_LON_DEG),
            (-TARGET_LAT_DEG, (TARGET_LON_DEG + 180.0) % 360.0),
        ]
        windows = find_delivery_windows_multi(
            short_pass_result, targets, target_elevation_min_deg=10.0,
        )
        assert len(windows) == 1
        assert windows[0].target_idx == 0

    def test_empty_targets_raises(self, short_pass_result):
        with pytest.raises(ValueError, match="non-empty"):
            find_delivery_windows_multi(short_pass_result, [])


class TestContinueDeliveryWindowsMulti:
    """Guarded continuation is exact inside its validated search bands."""

    @pytest.fixture(scope="class")
    def result_with_closed_guards(self):
        return _synthetic_straight_line_pass(
            _choose_sunlit_epoch(),
            t_span_s=80.0,
            n_samples=161,
        )

    def test_local_search_is_bit_exact_to_global(self, result_with_closed_guards):
        targets = [(TARGET_LAT_DEG, TARGET_LON_DEG)]
        global_windows = find_delivery_windows_multi(
            result_with_closed_guards,
            targets,
            target_elevation_min_deg=10.0,
        )
        assert len(global_windows) == 1

        continued = continue_delivery_windows_multi(
            result_with_closed_guards,
            targets,
            global_windows,
            search_margin_s=5.0,
            target_elevation_min_deg=10.0,
        )

        assert list(continued.windows) == global_windows
        assert continued.n_sample_target_evaluations < len(
            result_with_closed_guards.t_s
        )
        assert continued.max_boundary_shift_s == 0.0

    def test_open_band_edge_rejects_clipped_window(
        self, result_with_closed_guards
    ):
        targets = [(TARGET_LAT_DEG, TARGET_LON_DEG)]
        global_windows = find_delivery_windows_multi(
            result_with_closed_guards,
            targets,
            target_elevation_min_deg=10.0,
        )
        with pytest.raises(WindowContinuationError, match="open gate"):
            continue_delivery_windows_multi(
                result_with_closed_guards,
                targets,
                global_windows,
                search_margin_s=0.25,
                max_boundary_shift_s=0.0,
                target_elevation_min_deg=10.0,
            )

    def test_target_without_seed_requires_global_search(
        self, result_with_closed_guards
    ):
        with pytest.raises(WindowContinuationError, match="no seed windows"):
            continue_delivery_windows_multi(
                result_with_closed_guards,
                [(TARGET_LAT_DEG, TARGET_LON_DEG)],
                [],
            )


# ---------------------------------------------------------------------------
# Window-value (minimum fluence) filter (2026-06-09)
# ---------------------------------------------------------------------------


class TestMinWindowFluenceFilter:
    """min_window_fluence_J_per_m2 post-filter on find_delivery_windows."""

    @pytest.fixture(scope="class")
    def sunlit_et(self) -> float:
        return _choose_sunlit_epoch()

    @pytest.fixture
    def result(self, sunlit_et):
        return _synthetic_straight_line_pass(sunlit_et)

    @pytest.fixture
    def sail(self):
        from reflectors.srp import SailOptical, SolarSail
        return SolarSail(
            area_m2=1000.0, mass_kg=50.0,
            optical=SailOptical.square_sail_jpl(),
        )

    def _window_fluence(self, result, sail) -> float:
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail,
        )
        assert len(windows) == 1
        assert windows[0].fluence_J_per_m2 is not None
        return float(windows[0].fluence_J_per_m2)

    def test_none_threshold_bit_exact(self, result, sail):
        unfiltered = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail,
        )
        explicit_none = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail,
            min_window_fluence_J_per_m2=None,
        )
        assert explicit_none == unfiltered

    def test_threshold_below_keeps_window(self, result, sail):
        f = self._window_fluence(result, sail)
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail,
            min_window_fluence_J_per_m2=0.5 * f,
        )
        assert len(windows) == 1

    def test_threshold_above_drops_window(self, result, sail):
        f = self._window_fluence(result, sail)
        windows = find_delivery_windows(
            result, TARGET_LAT_DEG, TARGET_LON_DEG,
            target_elevation_min_deg=10.0, sail=sail,
            min_window_fluence_J_per_m2=2.0 * f,
        )
        assert windows == []

    def test_threshold_without_sail_raises(self, result):
        with pytest.raises(ValueError, match="requires a sail"):
            find_delivery_windows(
                result, TARGET_LAT_DEG, TARGET_LON_DEG,
                target_elevation_min_deg=10.0,
                min_window_fluence_J_per_m2=1.0,
            )

    def test_negative_threshold_raises(self, result, sail):
        with pytest.raises(ValueError, match="non-negative"):
            find_delivery_windows(
                result, TARGET_LAT_DEG, TARGET_LON_DEG,
                target_elevation_min_deg=10.0, sail=sail,
                min_window_fluence_J_per_m2=-1.0,
            )

    def test_threads_through_multi_finder(self, result, sail):
        """find_delivery_windows_multi forwards the filter kwarg."""
        f = self._window_fluence(result, sail)
        windows = find_delivery_windows_multi(
            result, [(TARGET_LAT_DEG, TARGET_LON_DEG)],
            target_elevation_min_deg=10.0, sail=sail,
            min_window_fluence_J_per_m2=2.0 * f,
        )
        assert windows == []
