"""Fast tests for the kinematic attitude profile layer.

Five groups:

  1. Primitives -- fixed_j2000, sun_pointing, smooth_slew endpoint and
     unit-norm behaviour, plus argument validation.
  2. Quintic analytic anchors -- peak |omega|, peak |alpha|, endpoint
     zero-rate / zero-acceleration pinned against closed-form
     expressions derived from s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5.
  3. Piecewise composition -- interior agreement, continuity at
     boundaries, all validation rejections.
  4. Diagnostics -- angular_rate, angular_acceleration,
     check_alpha_bound, alpha_profile.
  5. Public API compatibility -- attitude callable aliases and a one-orbit
     propagate() roundtrip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import spiceypy as spice

from reflectors.attitude import (
    AttitudeCallable,
    alpha_profile,
    angular_acceleration,
    angular_rate,
    check_alpha_bound,
    fixed_j2000,
    mars_pole_j2000,
    orbit_frame_fixed,
    piecewise,
    smooth_slew,
    smooth_slew_hermite,
    sun_pointing,
    tumble,
)
from reflectors.dynamics import (
    PropagationOptions,
    mars_gm_km3_per_s2,
    propagate,
)
from reflectors.ephemeris import utc_to_et
from reflectors.srp import SailOptical, SolarSail
from reflectors.surface import mars_equatorial_radius_km


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def epoch_et() -> float:
    return utc_to_et(EPOCH_STR)


@pytest.fixture(scope="module")
def sun_state_km(epoch_et):
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    return np.asarray(state[:3], dtype=float)


@pytest.fixture(scope="module")
def sun_hat_j2000(sun_state_km):
    return sun_state_km / np.linalg.norm(sun_state_km)


@pytest.fixture(scope="module")
def sub_solar_lmo_position(sun_hat_j2000):
    R_sat = mars_equatorial_radius_km() + 400.0
    return R_sat * sun_hat_j2000


def _circular_lmo_trajectory_fn(r0_km: np.ndarray, mu_km3_s2: float):
    """Analytic circular-orbit trajectory callable r(et) for diagnostic tests.

    Given a position vector ``r0_km`` at et=0, synthesise an ideal
    circular orbit in the plane perpendicular to a fixed orbit-normal
    derived from ``r0 x z_hat``. Returns ``lambda et: r(et)``. Used
    only in diagnostic tests that need a non-trivial trajectory for
    ``sun_pointing``; the numerical value of the orbit is not under
    test -- only that ``sun_pointing`` receives a sensibly evolving
    ``r_sat``.
    """
    r0 = np.asarray(r0_km, dtype=float)
    r = float(np.linalg.norm(r0))
    # Arbitrary orbit-normal: take z_hat unless r0 is near-aligned with it.
    z = np.array([0.0, 0.0, 1.0])
    if abs(r0 @ z) / r > 0.9:
        z = np.array([0.0, 1.0, 0.0])
    orbit_normal = np.cross(r0, z)
    orbit_normal = orbit_normal / np.linalg.norm(orbit_normal)
    # In-plane unit vectors.
    u1 = r0 / r
    u2 = np.cross(orbit_normal, u1)
    n_mean = math.sqrt(mu_km3_s2 / (r**3))

    def traj(et: float) -> np.ndarray:
        theta = n_mean * et
        return r * (math.cos(theta) * u1 + math.sin(theta) * u2)

    return traj


# ---------------------------------------------------------------------------
# Group 1: Primitives
# ---------------------------------------------------------------------------


class TestFixedJ2000:
    def test_returns_unit_vector(self):
        n = fixed_j2000(np.array([3.0, 4.0, 0.0]))(np.zeros(3), 0.0)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-15)
        assert np.allclose(n, np.array([0.6, 0.8, 0.0]), atol=1e-15)

    def test_independent_of_state_and_epoch(self):
        f = fixed_j2000(np.array([1.0, 0.0, 0.0]))
        n1 = f(np.zeros(3), 0.0)
        n2 = f(np.array([1e5, -2e5, 3e5]), 7.25e9)
        assert np.allclose(n1, n2, atol=0.0)

    def test_rejects_zero_vector(self):
        with pytest.raises(ValueError, match="zero vector"):
            fixed_j2000(np.zeros(3))


class TestSunPointing:
    def test_returns_unit_vector_toward_sun(self, epoch_et, sun_state_km):
        f = sun_pointing()
        r_sat = np.array([3796.0, 0.0, 0.0])  # arbitrary LMO position
        n = f(r_sat, epoch_et)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-15)
        expected = sun_state_km - r_sat
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(n, expected, atol=1e-15)

    def test_direction_evolves_with_epoch(self, epoch_et):
        f = sun_pointing()
        r_sat = np.array([3796.0, 0.0, 0.0])
        n_t0 = f(r_sat, epoch_et)
        n_t1 = f(r_sat, epoch_et + 3600.0)  # 1-hour offset
        # Non-zero change: Sun drifts by ~0.5 deg/day, so 1 hour is ~0.02 deg
        # and the vectors should not be identical.
        assert not np.allclose(n_t0, n_t1, atol=1e-10)


class TestSmoothSlewConstruction:
    def test_endpoint_matches_exactly_at_t0(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        slew = smooth_slew(100.0, 200.0, n0, nf)
        n = slew(np.zeros(3), 100.0)
        assert np.allclose(n, n0, atol=1e-14)

    def test_endpoint_matches_exactly_at_tf(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        slew = smooth_slew(100.0, 200.0, n0, nf)
        n = slew(np.zeros(3), 200.0)
        assert np.allclose(n, nf, atol=1e-14)

    def test_interior_output_is_unit_norm(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.3, 0.8, 0.5])  # non-axis-aligned
        slew = smooth_slew(0.0, 100.0, n0, nf)
        for et in np.linspace(0.0, 100.0, 33):
            n = slew(np.zeros(3), float(et))
            assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-14)

    def test_midpoint_matches_half_angle_analytic(self):
        """At tau=0.5, s(tau)=0.5 so theta = theta_total/2 = pi/4."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        slew = smooth_slew(0.0, 100.0, n0, nf)
        mid = slew(np.zeros(3), 50.0)
        expected = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0])
        assert np.allclose(mid, expected, atol=1e-14)

    def test_rejects_inverted_time(self):
        with pytest.raises(ValueError, match="tf_et must exceed"):
            smooth_slew(10.0, 5.0, np.array([1, 0, 0]), np.array([0, 1, 0]))

    def test_rejects_zero_duration(self):
        with pytest.raises(ValueError, match="tf_et must exceed"):
            smooth_slew(10.0, 10.0, np.array([1, 0, 0]), np.array([0, 1, 0]))

    def test_rejects_zero_start_vector(self):
        with pytest.raises(ValueError, match="zero vector"):
            smooth_slew(0.0, 10.0, np.zeros(3), np.array([1, 0, 0]))

    def test_rejects_zero_end_vector(self):
        with pytest.raises(ValueError, match="zero vector"):
            smooth_slew(0.0, 10.0, np.array([1, 0, 0]), np.zeros(3))

    def test_rejects_antipodal_endpoints(self):
        with pytest.raises(ValueError, match="antipodal"):
            smooth_slew(0.0, 10.0, np.array([1, 0, 0]), np.array([-1, 0, 0]))

    def test_degenerate_identical_endpoints_returns_fixed(self):
        """Near-parallel endpoints should degenerate cleanly."""
        n = np.array([1.0, 0.0, 0.0])
        slew = smooth_slew(0.0, 100.0, n, n.copy())
        for et in [0.0, 25.0, 50.0, 75.0, 100.0]:
            out = slew(np.zeros(3), float(et))
            assert np.allclose(out, n, atol=1e-14)

    def test_rejects_query_before_t0(self):
        slew = smooth_slew(0.0, 100.0, np.array([1, 0, 0]), np.array([0, 1, 0]))
        with pytest.raises(ValueError, match="outside domain"):
            slew(np.zeros(3), -1.0)

    def test_rejects_query_after_tf(self):
        slew = smooth_slew(0.0, 100.0, np.array([1, 0, 0]), np.array([0, 1, 0]))
        with pytest.raises(ValueError, match="outside domain"):
            slew(np.zeros(3), 101.0)


class TestSmoothSlewHermite:
    """Group 1b: dynamic-endpoint Hermite slew primitive.

    Pins exact endpoint matching on both n_hat and omega, unit-norm
    interior, the deliberate non-bit-for-bit difference vs smooth_slew
    at zero endpoint omega, and all construction-time rejections
    (antipodal endpoints, zero vectors, projection guard).
    """

    def test_endpoint_nhat_matches_exactly_at_t0(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        slew = smooth_slew_hermite(100.0, 200.0, n0, nf)
        n = slew(np.zeros(3), 100.0)
        assert np.allclose(n, n0, atol=1e-14)

    def test_endpoint_nhat_matches_exactly_at_tf(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        slew = smooth_slew_hermite(100.0, 200.0, n0, nf)
        n = slew(np.zeros(3), 200.0)
        assert np.allclose(n, nf, atol=1e-14)

    def test_interior_is_unit_norm(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        slew = smooth_slew_hermite(0.0, 100.0, n0, nf)
        trivial_r = lambda et: np.zeros(3)  # noqa: E731
        # 100 interior samples (skip exact endpoints to avoid FP at boundary).
        for tau in np.linspace(0.01, 0.99, 100):
            et = 100.0 * tau
            n = slew(trivial_r(et), et)
            assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-14)

    def test_endpoint_omega_matches_at_t0(self):
        """omega(t0) must equal the commanded omega_0 (bisector handoff case)."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        # omega_0 perpendicular to n0 (the only physically meaningful
        # component under omega . n_hat = 0).
        omega_0 = np.array([0.0, 0.0, 2.0e-3])  # rad/s
        slew = smooth_slew_hermite(
            0.0, 100.0, n0, nf,
            omega_0_rad_s=omega_0, omega_f_rad_s=np.zeros(3),
        )
        trivial_r = lambda et: np.zeros(3)  # noqa: E731
        # Central-diff near t0 + dt to avoid the strict domain floor.
        omega_measured = angular_rate(slew, trivial_r, 1.0e-2, dt=1.0e-3)
        assert np.allclose(omega_measured, omega_0, atol=1.0e-6)

    def test_endpoint_omega_matches_at_tf(self):
        """omega(tf) must equal the commanded omega_f (track -> cruise)."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        omega_f = np.array([0.0, 0.0, 2.0e-3])  # rad/s, perpendicular to nf
        slew = smooth_slew_hermite(
            0.0, 100.0, n0, nf,
            omega_0_rad_s=np.zeros(3), omega_f_rad_s=omega_f,
        )
        trivial_r = lambda et: np.zeros(3)  # noqa: E731
        omega_measured = angular_rate(slew, trivial_r, 100.0 - 1.0e-2, dt=1.0e-3)
        assert np.allclose(omega_measured, omega_f, atol=1.0e-6)

    def test_static_endpoints_differ_from_smooth_slew_at_tau_0p25(self):
        """At omega=0, Hermite traverses straight chord, smooth_slew great circle.

        They coincide at endpoints and at tau=0.5 by symmetry, but at
        tau=0.25 the Hermite lies strictly INSIDE the great-circle arc
        (after unit projection, still on S^2 but at a different angle).
        This test pins the deliberate distinction so the two primitives
        are never conflated.
        """
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        T = 100.0
        hermite = smooth_slew_hermite(0.0, T, n0, nf)
        arc = smooth_slew(0.0, T, n0, nf)
        et_mid_quarter = 0.25 * T
        n_hermite = hermite(np.zeros(3), et_mid_quarter)
        n_arc = arc(np.zeros(3), et_mid_quarter)
        # Non-negligible angular difference (more than FP noise).
        dot_val = float(np.dot(n_hermite, n_arc))
        assert dot_val < 1.0 - 1.0e-6, (
            f"Hermite and smooth_slew agree to dot={dot_val} at tau=0.25; "
            "they should not (the two primitives are distinct by design)."
        )
        # But BOTH are unit vectors on S^2 and in the span{n0, nf} plane.
        orbit_plane_normal = np.cross(n0, nf)
        orbit_plane_normal = orbit_plane_normal / np.linalg.norm(orbit_plane_normal)
        assert abs(float(np.dot(n_hermite, orbit_plane_normal))) < 1.0e-14
        assert abs(float(np.dot(n_arc, orbit_plane_normal))) < 1.0e-14

    def test_static_endpoints_meet_at_tau_half(self):
        """At omega=0, Hermite and smooth_slew coincide at tau=0.5 by symmetry."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        T = 100.0
        hermite = smooth_slew_hermite(0.0, T, n0, nf)
        arc = smooth_slew(0.0, T, n0, nf)
        n_h = hermite(np.zeros(3), 0.5 * T)
        n_a = arc(np.zeros(3), 0.5 * T)
        assert np.allclose(n_h, n_a, atol=1.0e-14)

    def test_rejects_tf_le_t0(self):
        with pytest.raises(ValueError, match="tf_et must exceed t0_et"):
            smooth_slew_hermite(
                100.0, 100.0,
                np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            )

    def test_rejects_zero_vector_endpoint(self):
        with pytest.raises(ValueError, match="zero vector"):
            smooth_slew_hermite(
                0.0, 10.0, np.array([1.0, 0.0, 0.0]), np.zeros(3),
            )

    def test_rejects_antipodal_endpoints(self):
        with pytest.raises(ValueError, match="antipodal"):
            smooth_slew_hermite(
                0.0, 10.0,
                np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
            )

    def test_rejects_query_outside_domain(self):
        slew = smooth_slew_hermite(
            0.0, 100.0,
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        )
        with pytest.raises(ValueError, match="outside domain"):
            slew(np.zeros(3), -1.0)
        with pytest.raises(ValueError, match="outside domain"):
            slew(np.zeros(3), 101.0)

    def test_projection_guard_fires_at_large_theta(self):
        """Large rotation angle (near pi) triggers the projection guard.

        For static endpoints the R^3 chord midpoint has |c| = cos(theta/2);
        at theta = 160 deg this is cos(80 deg) ~ 0.174, well below
        the 0.3 guard threshold.
        """
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(160.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        with pytest.raises(ValueError, match="projection ill-conditioned"):
            smooth_slew_hermite(0.0, 100.0, n0, nf)

    def test_projection_guard_passes_at_theta_just_under_threshold(self):
        """theta such that cos(theta/2) > 0.3 must NOT trigger the guard.

        cos(theta/2) = 0.3 at theta ~ 145.1 deg. At theta = 140 deg
        (cos(70 deg) = 0.342) the guard should pass with margin.
        """
        n0 = np.array([1.0, 0.0, 0.0])
        theta = math.radians(140.0)
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        # Should not raise.
        _ = smooth_slew_hermite(0.0, 100.0, n0, nf)

    def test_realistic_bisector_handoff_well_conditioned(self):
        """501 km LMO bisector omega ~ 2e-3 rad/s, slew 240 s, theta ~ 45 deg.

        This representative pointing case must not trigger the projection
        guard; min|c| should remain comfortably above 0.3. The
        delivered-vs-requested omega_f residual is also pinned near machine
        precision.
        """
        n0 = np.array([1.0, 0.0, 0.0])  # sun-pointing-ish
        theta = math.radians(45.0)  # typical bisector offset from sun
        nf = np.array([math.cos(theta), math.sin(theta), 0.0])
        T = 240.0
        omega_f = np.array([0.0, 0.0, 2.0e-3])  # bisector peak angular rate
        slew = smooth_slew_hermite(
            0.0, T, n0, nf,
            omega_f_rad_s=omega_f,
        )
        trivial_r = lambda et: np.zeros(3)  # noqa: E731
        # omega at tf must equal omega_f.
        omega_measured = angular_rate(slew, trivial_r, T - 1.0e-2, dt=1.0e-3)
        assert np.allclose(omega_measured, omega_f, atol=1.0e-7)


# ---------------------------------------------------------------------------
# Group 2: Quintic analytic anchors
# ---------------------------------------------------------------------------


class TestQuinticPeaks:
    """Pin the closed-form peaks of the rest-to-rest quintic slew."""

    @pytest.fixture
    def slew_90deg_100s(self):
        """90-degree slew over T=100 s about the z-axis."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        return smooth_slew(0.0, 100.0, n0, nf), 100.0, math.pi / 2

    def test_omega_zero_at_t0(self, slew_90deg_100s):
        slew, T, _ = slew_90deg_100s
        # Cannot query exactly at t0 because angular_rate needs t0 - dt; use
        # a small offset instead. Quintic's zero-rate at t0 means omega is
        # O(dt^2) there.
        dt = 0.01
        w = angular_rate(
            slew, lambda et: np.zeros(3), dt, dt=dt
        )
        # omega magnitude near t0 should be tiny (O(dt^2) * analytic peak).
        assert np.linalg.norm(w) < 1e-5

    def test_omega_zero_at_tf(self, slew_90deg_100s):
        slew, T, _ = slew_90deg_100s
        dt = 0.01
        w = angular_rate(
            slew, lambda et: np.zeros(3), T - dt, dt=dt
        )
        assert np.linalg.norm(w) < 1e-5

    def test_alpha_zero_at_t0(self, slew_90deg_100s):
        slew, T, _ = slew_90deg_100s
        dt = 0.01
        a = angular_acceleration(
            slew, lambda et: np.zeros(3), dt, dt=dt
        )
        assert np.linalg.norm(a) < 1e-5

    def test_alpha_zero_at_tf(self, slew_90deg_100s):
        slew, T, _ = slew_90deg_100s
        dt = 0.01
        a = angular_acceleration(
            slew, lambda et: np.zeros(3), T - dt, dt=dt
        )
        assert np.linalg.norm(a) < 1e-5

    def test_peak_omega_matches_analytic(self, slew_90deg_100s):
        """At tau=0.5, |omega| = theta_total * (15/8) / T."""
        slew, T, theta_total = slew_90deg_100s
        # Peak of s'(tau) = 30 tau^2 (1-tau)^2 is at tau=1/2.
        et_peak = T / 2.0
        w = angular_rate(slew, lambda et: np.zeros(3), et_peak, dt=0.01)
        w_mag = float(np.linalg.norm(w))
        analytic = theta_total * (15.0 / 8.0) / T
        rel_err = abs(w_mag - analytic) / analytic
        assert rel_err < 1e-4, (
            f"|omega|_peak numerical {w_mag:.6e} vs analytic "
            f"{analytic:.6e}, rel err {rel_err:.3e}"
        )

    def test_peak_alpha_matches_analytic(self, slew_90deg_100s):
        """At tau = (3 - sqrt(3))/6, |alpha| = theta_total * (10/sqrt(3)) / T^2."""
        slew, T, theta_total = slew_90deg_100s
        tau_peak = (3.0 - math.sqrt(3.0)) / 6.0
        et_peak = tau_peak * T
        a = angular_acceleration(slew, lambda et: np.zeros(3), et_peak, dt=0.01)
        a_mag = float(np.linalg.norm(a))
        analytic = theta_total * (10.0 / math.sqrt(3.0)) / (T * T)
        rel_err = abs(a_mag - analytic) / analytic
        assert rel_err < 1e-4, (
            f"|alpha|_peak numerical {a_mag:.6e} vs analytic "
            f"{analytic:.6e}, rel err {rel_err:.3e}"
        )

    def test_omega_direction_colinear_with_rotation_axis(self, slew_90deg_100s):
        """omega is along the slew rotation axis e_hat = z_hat (for x -> y)."""
        slew, T, _ = slew_90deg_100s
        e_hat = np.array([0.0, 0.0, 1.0])
        w = angular_rate(slew, lambda et: np.zeros(3), T / 2.0, dt=0.01)
        # Normalise and take abs dot product; colinear means |dot| ~ 1.
        w_unit = w / np.linalg.norm(w)
        assert abs(float(w_unit @ e_hat)) > 1.0 - 1e-6

    def test_alpha_direction_colinear_with_rotation_axis(self, slew_90deg_100s):
        slew, T, _ = slew_90deg_100s
        e_hat = np.array([0.0, 0.0, 1.0])
        tau_peak = (3.0 - math.sqrt(3.0)) / 6.0
        a = angular_acceleration(
            slew, lambda et: np.zeros(3), tau_peak * T, dt=0.01
        )
        a_unit = a / np.linalg.norm(a)
        assert abs(float(a_unit @ e_hat)) > 1.0 - 1e-6

    def test_peak_alpha_scales_as_theta_over_T_squared(self):
        """Doubling theta doubles alpha; doubling T quarters alpha."""
        e_hat = np.array([0.0, 0.0, 1.0])
        tau_peak = (3.0 - math.sqrt(3.0)) / 6.0

        def peak_alpha_for(theta: float, T: float) -> float:
            # Build slew from (1, 0, 0) to Rodrigues(z_hat, theta).(1, 0, 0).
            n0 = np.array([1.0, 0.0, 0.0])
            nf = np.array([math.cos(theta), math.sin(theta), 0.0])
            slew = smooth_slew(0.0, T, n0, nf)
            a = angular_acceleration(
                slew, lambda et: np.zeros(3), tau_peak * T, dt=0.01
            )
            return float(np.linalg.norm(a))

        a_ref = peak_alpha_for(math.pi / 4, 100.0)
        # Doubling theta: expect 2x.
        a_2theta = peak_alpha_for(math.pi / 2, 100.0)
        assert abs(a_2theta / a_ref - 2.0) < 1e-3
        # Doubling T: expect 1/4x.
        a_2T = peak_alpha_for(math.pi / 4, 200.0)
        assert abs(a_2T / a_ref - 0.25) < 1e-3


# ---------------------------------------------------------------------------
# Group 3: Piecewise composition
# ---------------------------------------------------------------------------


class TestPiecewise:
    def test_three_segments_agree_with_constituents(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        s_idle = fixed_j2000(n0)
        s_slew = smooth_slew(10.0, 50.0, n0, nf)
        s_track = fixed_j2000(nf)
        pw = piecewise(
            [
                (0.0, 10.0, s_idle),
                (10.0, 50.0, s_slew),
                (50.0, 60.0, s_track),
            ]
        )
        # Idle segment.
        assert np.allclose(pw(np.zeros(3), 5.0), n0, atol=1e-14)
        # Slew interior: quintic at et=30, tau=0.5 => theta = pi/4.
        assert np.allclose(
            pw(np.zeros(3), 30.0),
            np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0]),
            atol=1e-14,
        )
        # Tracking segment.
        assert np.allclose(pw(np.zeros(3), 55.0), nf, atol=1e-14)

    def test_n_hat_continuous_at_internal_boundaries(self):
        """At a boundary shared between compatible segments, n_hat is continuous."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        s_idle = fixed_j2000(n0)
        s_slew = smooth_slew(10.0, 50.0, n0, nf)
        pw = piecewise([(0.0, 10.0, s_idle), (10.0, 50.0, s_slew)])
        n_before = pw(np.zeros(3), 10.0 - 1e-9)
        n_at = pw(np.zeros(3), 10.0)
        n_after = pw(np.zeros(3), 10.0 + 1e-9)
        # All should equal n0 (idle just before, slew starts at n0).
        assert np.allclose(n_before, n0, atol=1e-14)
        assert np.allclose(n_at, n0, atol=1e-14)
        assert np.allclose(n_after, n0, atol=1e-10)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            piecewise([])

    def test_rejects_gap(self):
        s = fixed_j2000(np.array([1, 0, 0]))
        with pytest.raises(ValueError, match="gap or overlap"):
            piecewise([(0.0, 10.0, s), (15.0, 20.0, s)])

    def test_rejects_overlap(self):
        s = fixed_j2000(np.array([1, 0, 0]))
        with pytest.raises(ValueError, match="gap or overlap"):
            piecewise([(0.0, 10.0, s), (5.0, 15.0, s)])

    def test_rejects_out_of_order(self):
        s = fixed_j2000(np.array([1, 0, 0]))
        # (10, 20) before (0, 10) -> the second tuple's start precedes the
        # first tuple's end, detected as gap/overlap.
        with pytest.raises(ValueError, match="gap or overlap"):
            piecewise([(10.0, 20.0, s), (0.0, 10.0, s)])

    def test_rejects_non_positive_duration(self):
        s = fixed_j2000(np.array([1, 0, 0]))
        with pytest.raises(ValueError, match="non-positive duration"):
            piecewise([(10.0, 10.0, s)])

    def test_rejects_query_outside_range(self):
        s = fixed_j2000(np.array([1, 0, 0]))
        pw = piecewise([(0.0, 10.0, s)])
        with pytest.raises(ValueError, match="outside covered range"):
            pw(np.zeros(3), -0.5)
        with pytest.raises(ValueError, match="outside covered range"):
            pw(np.zeros(3), 10.5)


# ---------------------------------------------------------------------------
# Group 4: Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_angular_rate_of_fixed_profile_is_zero(self):
        f = fixed_j2000(np.array([1, 0, 0]))
        w = angular_rate(f, lambda et: np.zeros(3), 42.0, dt=1.0)
        assert np.allclose(w, np.zeros(3), atol=1e-14)

    def test_angular_acceleration_of_fixed_profile_is_zero(self):
        f = fixed_j2000(np.array([0, 1, 0]))
        a = angular_acceleration(f, lambda et: np.zeros(3), 42.0, dt=1.0)
        assert np.allclose(a, np.zeros(3), atol=1e-14)

    def test_sun_pointing_omega_order_of_magnitude(self, epoch_et):
        """At an LMO, sun_pointing is inertially-near-fixed; omega is tiny.

        The Sun is effectively at infinity relative to any orbital
        excursion of the sail, so sun_pointing's ``n_hat`` rotates at:

          - the sail's orbital-velocity component perpendicular to the
            Mars-Sun line divided by the sail-Sun distance
            (v / r_MS ~ 3.5 km/s / 2.1e8 km ~ 2e-8 rad/s at 400 km LMO);
          - plus Mars's heliocentric angular velocity
            (2 pi / Mars year ~ 1e-7 rad/s).

        Total is O(1e-7) rad/s -- four orders of magnitude below the
        sail's orbital mean motion. Pin a loose order-of-magnitude
        bracket [1e-8, 1e-5] rad/s to catch any gross regression while
        tolerating geometry-dependent variation.
        """
        R_sat = mars_equatorial_radius_km() + 400.0
        r0 = np.array([R_sat, 0.0, 0.0])
        mu = mars_gm_km3_per_s2()
        traj = _circular_lmo_trajectory_fn(r0, mu)
        f = sun_pointing()
        w = angular_rate(f, lambda et: traj(et), epoch_et, dt=1.0)
        w_mag = float(np.linalg.norm(w))
        assert 1.0e-8 < w_mag < 1.0e-5, (
            f"sun_pointing |omega|={w_mag:.3e}, expected O(1e-7) rad/s "
            "at 400 km LMO with Sun effectively at infinity"
        )

    def test_check_alpha_bound_returns_none_above_peak(self):
        """With alpha_max slightly above the analytic peak, no violation."""
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        T = 100.0
        slew = smooth_slew(0.0, T, n0, nf)
        alpha_peak_analytic = (math.pi / 2) * (10.0 / math.sqrt(3.0)) / (T * T)
        res = check_alpha_bound(
            slew,
            lambda et: np.zeros(3),
            alpha_peak_analytic * 1.01,
            (0.0, T),
            n_samples=500,
            dt=0.1,
        )
        assert res is None

    def test_check_alpha_bound_returns_finite_below_peak(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        T = 100.0
        slew = smooth_slew(0.0, T, n0, nf)
        alpha_peak_analytic = (math.pi / 2) * (10.0 / math.sqrt(3.0)) / (T * T)
        res = check_alpha_bound(
            slew,
            lambda et: np.zeros(3),
            alpha_peak_analytic * 0.5,  # well below the peak
            (0.0, T),
            n_samples=500,
            dt=0.1,
        )
        assert res is not None
        # First violator should lie in (0, T/2) -- the rising half of the
        # |alpha|(t) curve crosses 0.5 * peak around tau ~ 0.08 and stays
        # above until tau ~ 0.35, so et should land in roughly [8, 35].
        assert 0.0 < res < T

    def test_check_alpha_bound_rejects_negative_bound(self):
        f = fixed_j2000(np.array([1, 0, 0]))
        with pytest.raises(ValueError, match="alpha_max must be"):
            check_alpha_bound(
                f,
                lambda et: np.zeros(3),
                -1.0,
                (0.0, 10.0),
                n_samples=10,
                dt=0.1,
            )

    def test_alpha_profile_shapes_and_bound(self):
        n0 = np.array([1.0, 0.0, 0.0])
        nf = np.array([0.0, 1.0, 0.0])
        T = 100.0
        slew = smooth_slew(0.0, T, n0, nf)
        alpha_peak_analytic = (math.pi / 2) * (10.0 / math.sqrt(3.0)) / (T * T)
        et_arr, alpha_mag = alpha_profile(
            slew, lambda et: np.zeros(3), (0.0, T), n_samples=1000, dt=0.1
        )
        assert et_arr.shape == (1000,)
        assert alpha_mag.shape == (1000,)
        # Numerical peak should not meaningfully exceed analytic peak.
        assert alpha_mag.max() < alpha_peak_analytic * 1.001
        # And should be above ~99% of analytic peak (central-diff with
        # dt=0.1 on a quintic is essentially exact).
        assert alpha_mag.max() > alpha_peak_analytic * 0.999

    def test_alpha_profile_rejects_too_narrow_window(self):
        f = fixed_j2000(np.array([1, 0, 0]))
        with pytest.raises(ValueError, match="too narrow"):
            alpha_profile(
                f, lambda et: np.zeros(3), (0.0, 1.0), n_samples=10, dt=1.0
            )


# ---------------------------------------------------------------------------
# Group 5: Public API contract
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_attitude_callable_type_alias_importable_from_both_modules(self):
        from reflectors.attitude import AttitudeCallable as AC_attitude
        from reflectors.srp import AttitudeCallable as AC_srp
        assert AC_attitude is AC_srp

    def test_propagate_with_attitude_sun_pointing(self, epoch_et):
        """One-orbit smoke that propagate() accepts an attitude profile from
        the attitude module and produces a finite trajectory."""
        R_sat = mars_equatorial_radius_km() + 400.0
        mu = mars_gm_km3_per_s2()
        v = math.sqrt(mu / R_sat)
        state0 = np.array([R_sat, 0.0, 0.0, 0.0, v, 0.0])
        period = 2 * math.pi * math.sqrt(R_sat**3 / mu)
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
            sail_normal=sun_pointing(),
            options=PropagationOptions.fast(),
        )
        # Trajectory should be finite and return close to start after one orbit.
        # |r(T) - r(0)| is dominated by SRP perturbation; this assertion
        # checks that the trajectory remains finite and bounded.
        assert np.all(np.isfinite(result.state_km_kmps))
        r_end = result.state_km_kmps[-1, :3]
        r_start = result.state_km_kmps[0, :3]
        delta = float(np.linalg.norm(r_end - r_start))
        # One-orbit closure: nominal ~0 km (two-body closes exactly).  With
        # full physics + SRP the residual is O(km) at most at LMO; pin < 100 km.
        assert delta < 100.0, f"Orbit closure residual {delta:.3f} km too large"


# ---------------------------------------------------------------------------
# Group 6: Uncommanded attitude -- orbit_frame_fixed and tumble
#
# These describe an uncontrolled sail. Both must be smooth, exact, and
# reproducible: a
# discontinuous or stochastic n_hat(t) would wreck DOP853's step control.
# ---------------------------------------------------------------------------


_LMO_A_KM = 3904.0        # ~K=12 sun-sync design point (508 km altitude)


@pytest.fixture(scope="module")
def lmo_h_hat(epoch_et):
    """A REALISTIC sun-sync orbit normal: ~92.9 deg from the Mars spin pole.

    Built relative to the actual pole rather than hard-coded in J2000 axes.
    Mars sun-sync orbits are near-polar retrograde (i ~ 92.9 deg at the K=12
    shell), so the orbit normal is very nearly PERPENDICULAR to the pole. That
    matters for more than realism: a normal close to the pole makes
    ``h x pole`` small and the pole-perpendicular projection used below
    ill-conditioned.
    """
    pole = mars_pole_j2000(epoch_et)
    # Any direction perpendicular to the pole, then tilt by (i - 90) deg.
    seed_vec = np.array([1.0, 0.0, 0.0])
    perp = seed_vec - float(np.dot(seed_vec, pole)) * pole
    perp /= np.linalg.norm(perp)
    incl = math.radians(92.9)
    h = math.cos(incl) * pole + math.sin(incl) * perp
    return h / np.linalg.norm(h)


class TestMarsPoleJ2000:
    def test_is_a_unit_vector(self, epoch_et):
        p = mars_pole_j2000(epoch_et)
        assert float(np.linalg.norm(p)) == pytest.approx(1.0, rel=1e-14)

    def test_matches_the_mme2000_third_row(self, epoch_et):
        """Thin accessor -- must not diverge from its source of truth."""
        from reflectors.elements import mme2000_rotation_from_j2000
        assert np.array_equal(mars_pole_j2000(epoch_et),
                              mme2000_rotation_from_j2000(epoch_et)[2])


class TestOrbitFrameFixed:
    def test_radial_choice_returns_exactly_r_hat(self, epoch_et, lmo_h_hat):
        att = orbit_frame_fixed([1.0, 0.0, 0.0], lmo_h_hat, epoch_et)
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        assert att(r, epoch_et) == pytest.approx(np.array([1.0, 0.0, 0.0]),
                                                 rel=1e-14, abs=1e-15)

    @pytest.mark.parametrize("c", [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                                   (1, 1, 0), (0.3, -0.5, 0.81)])
    def test_returns_unit_vector_everywhere_on_an_orbit(self, epoch_et,
                                                        lmo_h_hat, c):
        att = orbit_frame_fixed(np.array(c, dtype=float), lmo_h_hat, epoch_et)
        traj = _circular_lmo_trajectory_fn(
            np.array([_LMO_A_KM, 0.0, 0.0]), mars_gm_km3_per_s2())
        for dt in np.linspace(0.0, 7405.0, 25):
            n = att(traj(dt), epoch_et + dt)
            assert float(np.linalg.norm(n)) == pytest.approx(1.0, rel=1e-14)

    def test_rtn_basis_is_orthonormal(self, epoch_et, lmo_h_hat):
        """R, T, N built by the closure must be a right-handed orthonormal
        triad -- checked by asking for each axis in turn."""
        r = np.array([0.0, _LMO_A_KM * 0.6, _LMO_A_KM * 0.8])
        axes = [orbit_frame_fixed(c, lmo_h_hat, epoch_et)(r, epoch_et)
                for c in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
        M = np.vstack(axes)
        assert M @ M.T == pytest.approx(np.eye(3), rel=1e-13, abs=1e-13)
        assert float(np.linalg.det(M)) == pytest.approx(1.0, rel=1e-13)

    def test_orbit_normal_choice_is_perpendicular_to_position(self, epoch_et,
                                                              lmo_h_hat):
        att = orbit_frame_fixed([0.0, 0.0, 1.0], lmo_h_hat, epoch_et)
        for r in (np.array([_LMO_A_KM, 0.0, 0.0]),
                  np.array([0.0, _LMO_A_KM, 0.0]),
                  np.array([1.0, 2.0, 3.0]) / math.sqrt(14) * _LMO_A_KM):
            n = att(r, epoch_et)
            assert float(np.dot(n, r / np.linalg.norm(r))) == pytest.approx(
                0.0, abs=1e-14)

    def test_normal_precesses_about_the_mars_pole_at_the_sun_sync_rate(
        self, epoch_et, lmo_h_hat
    ):
        """After advancing h_hat analytically for a quarter of
        a Mars year the orbit plane must have rotated 90 deg about the pole.

        Evaluated at ``r`` ALONG THE POLE, which makes the check exact. The
        closure returns the orbit normal projected perpendicular to r_hat and
        renormalised; with ``r_hat = pole`` that projection is exactly the
        pole-perpendicular component of ``h_hat``, and a rotation of ``h_hat``
        about the pole rotates that component by precisely the same angle. At a
        general ``r`` the projection mixes in the (changing) radial component
        and the identity only holds approximately -- which is a property of the
        RTN triad, not of the precession.
        """
        from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S
        att = orbit_frame_fixed([0.0, 0.0, 1.0], lmo_h_hat, epoch_et)
        pole = mars_pole_j2000(epoch_et)
        r = _LMO_A_KM * pole
        n0 = att(r, epoch_et)
        for frac, want_cos in ((0.25, 0.0), (0.5, -1.0), (1.0, 1.0)):
            n1 = att(r, epoch_et + frac * MARS_SIDEREAL_YEAR_S)
            # Both vectors lie in the pole-perpendicular plane by construction.
            assert float(np.dot(n0, pole)) == pytest.approx(0.0, abs=1e-14)
            assert float(np.dot(n1, pole)) == pytest.approx(0.0, abs=1e-14)
            assert float(np.dot(n0, n1)) == pytest.approx(want_cos, abs=1e-9)

    def test_zero_node_rate_freezes_the_plane(self, epoch_et, lmo_h_hat):
        from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S
        att = orbit_frame_fixed([0.0, 0.0, 1.0], lmo_h_hat, epoch_et,
                                node_rate_rad_per_s=0.0)
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        assert att(r, epoch_et) == pytest.approx(
            att(r, epoch_et + MARS_SIDEREAL_YEAR_S / 4.0), rel=1e-14)

    def test_rejects_zero_vectors(self, epoch_et, lmo_h_hat):
        with pytest.raises(ValueError, match="n_hat_rtn"):
            orbit_frame_fixed([0.0, 0.0, 0.0], lmo_h_hat, epoch_et)
        with pytest.raises(ValueError, match="h_hat_0_j2000"):
            orbit_frame_fixed([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], epoch_et)

    def test_rejects_degenerate_triad(self, epoch_et):
        """h parallel to r has no transverse direction."""
        att = orbit_frame_fixed([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], epoch_et,
                                node_rate_rad_per_s=0.0)
        with pytest.raises(ValueError, match="degenerate"):
            att(np.array([_LMO_A_KM, 0.0, 0.0]), epoch_et)


class TestTumble:
    def test_returns_unit_vector_over_many_revolutions(self, epoch_et):
        rate = 2.0 * math.pi / 7405.0
        att = tumble(rate, epoch_et, seed=3)
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        for dt in np.linspace(0.0, 100 * 7405.0, 200):
            n = att(r, epoch_et + dt)
            assert float(np.linalg.norm(n)) == pytest.approx(1.0, rel=1e-13)

    def test_phase_zero_at_reference_epoch(self, epoch_et):
        n0 = np.array([0.0, 0.0, 1.0])
        att = tumble(1e-3, epoch_et, n_hat_0=n0, spin_axis=[1.0, 0.0, 0.0])
        assert att(np.array([_LMO_A_KM, 0.0, 0.0]), epoch_et) == pytest.approx(
            n0, rel=1e-15, abs=1e-16)

    def test_single_axis_spin_is_periodic(self, epoch_et):
        """One full revolution returns the normal exactly -- confirms the rate
        means what it says."""
        period = 7405.0
        att = tumble(2.0 * math.pi / period, epoch_et,
                     n_hat_0=[0.0, 0.0, 1.0], spin_axis=[1.0, 0.0, 0.0])
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        assert att(r, epoch_et + period) == pytest.approx(
            att(r, epoch_et), rel=1e-12, abs=1e-13)

    def test_rate_matches_the_commanded_spin_rate(self, epoch_et):
        """Central-difference the profile against the EXACT chord length.

        For a rigid rotation at rate ``w`` about ``axis``, ``n`` traces a circle
        of radius ``sin(gamma)``, so the chord spanning ``+/- h`` is exactly
        ``2 sin(gamma) sin(w h)`` and

            |n(t+h) - n(t-h)| / (2h) = w sin(gamma) * sin(w h)/(w h).

        Comparing against that closed form rather than the ``w sin(gamma)``
        small-angle limit removes truncation error entirely, leaving only
        round-off.

        ``h = 1.0 s``, not something tiny: ET here is ~8.3e8, where the
        double-precision spacing is ~1.2e-7 s, so an ``h`` of 1e-3 carries a
        ~1e-4 relative error in the step itself. Using ``h = 1.0 s`` keeps
        time-representation error below the asserted precision.
        """
        rate = 2.0 * math.pi / 7405.0
        n0 = np.array([0.0, 0.0, 1.0])
        axis = np.array([1.0, 0.0, 0.0])
        att = tumble(rate, epoch_et, n_hat_0=n0, spin_axis=axis)
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        h = 1.0
        for dt in (0.0, 900.0, 1850.0):
            t_lo, t_hi = epoch_et + dt - h, epoch_et + dt + h
            a = att(r, t_lo)
            b = att(r, t_hi)
            # Use the epochs the floats actually hold, not the nominal 2h.
            speed = float(np.linalg.norm(b - a)) / (t_hi - t_lo)
            n_mid = att(r, epoch_et + dt)
            sin_gamma = float(np.linalg.norm(np.cross(axis, n_mid)))
            exact = rate * sin_gamma * math.sin(rate * h) / (rate * h)
            assert speed == pytest.approx(exact, rel=1e-7)

    def test_single_axis_spin_stays_on_a_cone_about_the_spin_axis(self,
                                                                 epoch_et):
        """The caveat that motivates the sphere-covering option: a pure spin
        does NOT sample the sphere -- it traces one circle."""
        axis = np.array([0.3, -0.5, 0.81])
        axis = axis / np.linalg.norm(axis)
        att = tumble(1e-3, epoch_et, n_hat_0=[0.0, 0.0, 1.0], spin_axis=axis)
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        cos0 = float(np.dot(att(r, epoch_et), axis))
        for dt in np.linspace(0.0, 50000.0, 60):
            assert float(np.dot(att(r, epoch_et + dt), axis)) == pytest.approx(
                cos0, abs=1e-12)

    def test_precession_breaks_the_cone(self, epoch_et):
        """With a second incommensurate rotation the normal leaves the single
        cone -- the sphere-covering mode used for the rapid-tumble case."""
        axis = np.array([0.0, 0.0, 1.0])
        att = tumble(1e-3, epoch_et, n_hat_0=[1.0, 0.0, 0.0], spin_axis=axis,
                     precession_rate_rad_per_s=1e-3 / math.pi,
                     precession_axis=[1.0, 0.0, 0.0])
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        cosines = [float(np.dot(att(r, epoch_et + dt), axis))
                   for dt in np.linspace(0.0, 40000.0, 200)]
        assert max(cosines) - min(cosines) > 0.5

    def test_is_reproducible_from_the_seed(self, epoch_et):
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        a = tumble(1e-3, epoch_et, seed=1234)
        b = tumble(1e-3, epoch_et, seed=1234)
        c = tumble(1e-3, epoch_et, seed=4321)
        for dt in (0.0, 500.0, 5000.0):
            assert np.array_equal(a(r, epoch_et + dt), b(r, epoch_et + dt))
        assert not np.allclose(a(r, epoch_et + 500.0), c(r, epoch_et + 500.0))

    def test_requires_a_seed_when_geometry_is_left_to_chance(self, epoch_et):
        with pytest.raises(ValueError, match="seed is required"):
            tumble(1e-3, epoch_et)
        with pytest.raises(ValueError, match="seed is required"):
            tumble(1e-3, epoch_et, n_hat_0=[0.0, 0.0, 1.0])
        # Fully specified single-axis spin needs no seed.
        tumble(1e-3, epoch_et, n_hat_0=[0.0, 0.0, 1.0],
               spin_axis=[1.0, 0.0, 0.0])

    def test_rejects_spin_axis_parallel_to_the_normal(self, epoch_et):
        with pytest.raises(ValueError, match="parallel"):
            tumble(1e-3, epoch_et, n_hat_0=[0.0, 0.0, 1.0],
                   spin_axis=[0.0, 0.0, 1.0])

    def test_is_smooth_enough_for_an_integrator(self, epoch_et):
        """Second difference stays bounded -- i.e. the profile has a continuous
        second derivative, which is what the adaptive stepper relies on."""
        rate = 2.0 * math.pi / 600.0
        att = tumble(rate, epoch_et, seed=9)
        r = np.array([_LMO_A_KM, 0.0, 0.0])
        h = 0.5
        worst = 0.0
        for dt in np.linspace(0.0, 3000.0, 200):
            a = att(r, epoch_et + dt - h)
            b = att(r, epoch_et + dt)
            c2 = att(r, epoch_et + dt + h)
            worst = max(worst, float(np.linalg.norm(a - 2 * b + c2)) / h ** 2)
        # |d2n/dt2| <= rate^2 for a rigid rotation; allow a small FD margin.
        assert worst < 1.5 * rate ** 2
