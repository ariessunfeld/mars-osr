"""Fast tests for the parametric cruise-attitude family.

Covers:
  1. beta=0 identity -- sun_offset reduces to sun_pointing bit-for-bit.
  2. Geometry -- angle between n_hat and s_hat equals beta; offset
     direction perpendicular to s_hat; phi_u phase rotates offset in
     the perp-to-Sun plane.
  3. Orbit-plane basis construction from initial state.
  4. Sweep-integration smoke -- run_grid_point accepts cruise_attitude.
  5. Input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import spiceypy as spice

from reflectors.attitude import sun_pointing
from reflectors.cruise import (
    DELTA_AMP_MAX_DEFAULT_RAD,
    _orbit_plane_basis_j2000,
    _orbit_plane_normal_j2000,
    sun_offset,
    sun_offset_from_state,
    sun_offset_harmonic,
    sun_offset_harmonic_from_state,
    sun_offset_harmonic_full,
    sun_offset_harmonic_full_from_state,
    sun_offset_harmonic_full_mode2,
    sun_offset_harmonic_full_mode2_from_state,
    sun_offset_harmonic_full_mode3,
    sun_offset_harmonic_full_mode3_from_state,
)
from reflectors.ephemeris import utc_to_et
from reflectors.surface import mars_equatorial_radius_km


EPOCH_STR = "2026-06-01T00:00:00"


@pytest.fixture(scope="module")
def epoch_et() -> float:
    return utc_to_et(EPOCH_STR)


@pytest.fixture(scope="module")
def sub_solar_lmo_state(epoch_et):
    """Canonical 501 km sun-pointing-line sail state (J2000).

    Position: 501 km + R_Mars_equator along the sail->Sun line.
    Velocity: v_circ tangent in the orbit plane with z_hat ~ orbit pole.
    """
    state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    s = np.asarray(state[:3], dtype=float)
    s_hat = s / np.linalg.norm(s)
    R_sat = mars_equatorial_radius_km() + 501.0
    r0 = R_sat * s_hat
    # Velocity: pick direction perpendicular to r0 and mostly in the xy-plane
    # so the orbit normal has a big z-component.
    z_hat = np.array([0.0, 0.0, 1.0])
    v_dir = np.cross(z_hat, r0)
    v_dir = v_dir / np.linalg.norm(v_dir)
    mu_mars = 42828.37362069909  # km^3/s^2
    v_mag = math.sqrt(mu_mars / R_sat)
    v0 = v_mag * v_dir
    return np.concatenate([r0, v0])


# ---------------------------------------------------------------------------
# Group 1: beta=0 identity with sun_pointing
# ---------------------------------------------------------------------------


class TestBetaZeroReducesToSunPointing:
    def test_beta_zero_matches_sun_pointing_bit_for_bit(
        self, epoch_et, sub_solar_lmo_state
    ):
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        cruise = sun_offset(
            0.0, 0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0, 86400.0]:
            et = epoch_et + dt
            n_off = cruise(r_sat, et)
            n_sp = sp(r_sat, et)
            np.testing.assert_allclose(n_off, n_sp, atol=1.0e-15)

    def test_beta_zero_via_from_state_helper(self, epoch_et, sub_solar_lmo_state):
        cruise = sun_offset_from_state(
            0.0, 0.0, initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        n_off = cruise(sub_solar_lmo_state[:3], epoch_et)
        n_sp = sp(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_allclose(n_off, n_sp, atol=1.0e-15)


# ---------------------------------------------------------------------------
# Group 2: Geometric properties
# ---------------------------------------------------------------------------


class TestOffsetGeometry:
    def test_angle_between_n_hat_and_sun_equals_beta(
        self, epoch_et, sub_solar_lmo_state
    ):
        cruise = sun_offset_from_state(
            math.radians(10.0), 0.0, initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        r_sat = sub_solar_lmo_state[:3]
        n_off = cruise(r_sat, epoch_et)
        s_hat = sp(r_sat, epoch_et)
        cos_angle = float(np.dot(n_off, s_hat))
        # Numeric tolerance: cos(10 deg) = 0.9848077530...
        assert math.isclose(
            cos_angle, math.cos(math.radians(10.0)), abs_tol=1.0e-12
        )

    def test_angle_between_n_hat_and_sun_equals_beta_45deg(
        self, epoch_et, sub_solar_lmo_state
    ):
        cruise = sun_offset_from_state(
            math.radians(45.0), 0.0, initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        r_sat = sub_solar_lmo_state[:3]
        n_off = cruise(r_sat, epoch_et)
        s_hat = sp(r_sat, epoch_et)
        cos_angle = float(np.dot(n_off, s_hat))
        assert math.isclose(
            cos_angle, math.cos(math.radians(45.0)), abs_tol=1.0e-12
        )

    def test_n_hat_is_unit_norm(self, epoch_et, sub_solar_lmo_state):
        cruise = sun_offset_from_state(
            math.radians(15.0), math.radians(30.0),
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        for dt in [0.0, 1234.5, 45678.9]:
            et = epoch_et + dt
            n_hat = cruise(sub_solar_lmo_state[:3], et)
            assert np.isclose(np.linalg.norm(n_hat), 1.0, atol=1.0e-14)

    def test_phi_u_90deg_rotates_offset_direction(
        self, epoch_et, sub_solar_lmo_state
    ):
        """At two values of phi_u differing by 90 deg, the offset-from-Sun
        components should be 90 deg apart in the sun-perpendicular plane.
        """
        beta = math.radians(15.0)
        cruise_0 = sun_offset_from_state(
            beta, 0.0, initial_state_km_kmps=sub_solar_lmo_state,
        )
        cruise_90 = sun_offset_from_state(
            beta, math.radians(90.0), initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        r_sat = sub_solar_lmo_state[:3]
        n_0 = cruise_0(r_sat, epoch_et)
        n_90 = cruise_90(r_sat, epoch_et)
        s_hat = sp(r_sat, epoch_et)
        # Offset vectors: n_hat - cos(beta) * s_hat, then normalise.
        # Their dot product should be cos(90 deg) = 0.
        offset_0 = n_0 - math.cos(beta) * s_hat
        offset_90 = n_90 - math.cos(beta) * s_hat
        offset_0 = offset_0 / np.linalg.norm(offset_0)
        offset_90 = offset_90 / np.linalg.norm(offset_90)
        cos_sep = float(np.dot(offset_0, offset_90))
        assert abs(cos_sep) < 1.0e-12, (
            f"phi_u=0 and phi_u=90 offsets should be perpendicular, "
            f"got cos(sep) = {cos_sep}"
        )


# ---------------------------------------------------------------------------
# Group 3: Orbit-plane basis construction
# ---------------------------------------------------------------------------


class TestOrbitPlaneBasis:
    def test_orbit_normal_is_unit_and_perpendicular_to_r_and_v(
        self, sub_solar_lmo_state
    ):
        n_orb = _orbit_plane_normal_j2000(sub_solar_lmo_state)
        assert np.isclose(np.linalg.norm(n_orb), 1.0, atol=1.0e-14)
        r = sub_solar_lmo_state[:3]
        v = sub_solar_lmo_state[3:6]
        assert abs(float(np.dot(n_orb, r))) < 1.0e-12
        assert abs(float(np.dot(n_orb, v))) < 1.0e-12

    def test_orbit_plane_basis_is_orthonormal(self, sub_solar_lmo_state):
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        for v in (e_ref, e_ortho, n_orb):
            assert np.isclose(np.linalg.norm(v), 1.0, atol=1.0e-14)
        # Pairwise perpendicularity.
        assert abs(float(np.dot(e_ref, e_ortho))) < 1.0e-14
        assert abs(float(np.dot(e_ref, n_orb))) < 1.0e-14
        assert abs(float(np.dot(e_ortho, n_orb))) < 1.0e-14

    def test_collinear_r_v_raises(self):
        # r and v parallel -> h = 0.
        state = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="collinear"):
            _orbit_plane_normal_j2000(state)


# ---------------------------------------------------------------------------
# Group 4: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_beta_out_of_range_raises(self, sub_solar_lmo_state):
        _e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="beta_rad"):
            sun_offset(
                math.radians(91.0), 0.0,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=sub_solar_lmo_state[:3] / np.linalg.norm(sub_solar_lmo_state[:3]),
            )
        with pytest.raises(ValueError, match="beta_rad"):
            sun_offset(
                -0.1, 0.0,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=sub_solar_lmo_state[:3] / np.linalg.norm(sub_solar_lmo_state[:3]),
            )

    def test_zero_orbit_normal_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            sun_offset(
                math.radians(10.0), 0.0,
                orbit_plane_normal_j2000=np.zeros(3),
                orbit_ref_direction_j2000=np.array([1.0, 0.0, 0.0]),
            )

    def test_parallel_basis_inputs_raise(self):
        with pytest.raises(ValueError, match="parallel"):
            sun_offset(
                math.radians(10.0), 0.0,
                orbit_plane_normal_j2000=np.array([1.0, 0.0, 0.0]),
                orbit_ref_direction_j2000=np.array([1.0, 0.0, 0.0]),
            )


# ---------------------------------------------------------------------------
# Group 5: Harmonic-in-u cruise family
# ---------------------------------------------------------------------------


class TestSunOffsetHarmonic:
    """Tests for ``sun_offset_harmonic`` (Fourier-mode-1 cone angle).

    Mirrors ``TestBetaZeroReducesToSunPointing`` and ``TestOffsetGeometry``
    structure. The reduction-identity assertions pin the contract that
    ``alpha_c = alpha_s = 0`` collapses to ``sun_offset`` bit-for-bit.
    """

    def test_zero_harmonics_matches_sun_offset_bit_for_bit(
        self, epoch_et, sub_solar_lmo_state
    ):
        """alpha_c = alpha_s = 0 reduces to sun_offset(alpha_0, phi_u)."""
        alpha_0 = math.radians(7.5)
        phi_u = math.radians(45.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        constant = sun_offset(
            alpha_0, phi_u,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        harmonic = sun_offset_harmonic(
            alpha_0,
            alpha_c_rad=0.0, alpha_s_rad=0.0, phi_u_rad=phi_u,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0, 86400.0]:
            et = epoch_et + dt
            n_const = constant(r_sat, et)
            n_harm = harmonic(r_sat, et)
            np.testing.assert_array_equal(n_const, n_harm)

    def test_zero_alpha_0_zero_harmonics_matches_sun_pointing(
        self, epoch_et, sub_solar_lmo_state
    ):
        """alpha_0 = alpha_c = alpha_s = 0 reduces to sun_pointing exactly."""
        harmonic = sun_offset_harmonic_from_state(
            0.0,
            alpha_c_rad=0.0, alpha_s_rad=0.0, phi_u_rad=0.0,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        n_h = harmonic(sub_solar_lmo_state[:3], epoch_et)
        n_sp = sp(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_allclose(n_h, n_sp, atol=1.0e-15)

    def test_zero_harmonics_via_from_state(
        self, epoch_et, sub_solar_lmo_state
    ):
        """from_state wrapper preserves the reduction identity."""
        alpha_0 = math.radians(10.0)
        constant = sun_offset_from_state(
            alpha_0, 0.0, initial_state_km_kmps=sub_solar_lmo_state,
        )
        harmonic = sun_offset_harmonic_from_state(
            alpha_0,
            alpha_c_rad=0.0, alpha_s_rad=0.0, phi_u_rad=0.0,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        n_c = constant(sub_solar_lmo_state[:3], epoch_et)
        n_h = harmonic(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_array_equal(n_c, n_h)

    def test_cone_angle_at_u_zero(self, epoch_et, sub_solar_lmo_state):
        """At u = 0 (initial position, by construction), alpha(u) = alpha_0
        + alpha_c. Verified by extracting cos(alpha) = n_hat . s_hat at the
        initial sample epoch."""
        alpha_0 = math.radians(8.0)
        alpha_c = math.radians(3.0)
        harmonic = sun_offset_harmonic_from_state(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=0.0, phi_u_rad=0.0,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        r_sat = sub_solar_lmo_state[:3]
        # u = 0 at the initial position by construction of e_ref (the
        # initial position direction). At u = 0, cos(u) = 1, sin(u) = 0,
        # so alpha(0) = alpha_0 + alpha_c.
        n_h = harmonic(r_sat, epoch_et)
        s_hat = sp(r_sat, epoch_et)
        cos_angle = float(np.dot(n_h, s_hat))
        assert math.isclose(
            cos_angle, math.cos(alpha_0 + alpha_c), abs_tol=1.0e-12
        )

    def test_cone_angle_at_u_pi(self, sub_solar_lmo_state):
        """At u = pi (opposite the initial position), alpha(u) = alpha_0
        - alpha_c. Tested by sampling the closure at a position that lies
        antipodal in the orbit plane."""
        alpha_0 = math.radians(8.0)
        alpha_c = math.radians(3.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        harmonic = sun_offset_harmonic(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=0.0, phi_u_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        # Antipodal point: r = -r_0, so u = atan2(0, -1) = pi.
        r_anti = -np.asarray(sub_solar_lmo_state[:3])
        et = utc_to_et(EPOCH_STR)
        n_h = harmonic(r_anti, et)
        s_hat = sp(r_anti, et)
        cos_angle = float(np.dot(n_h, s_hat))
        assert math.isclose(
            cos_angle, math.cos(alpha_0 - alpha_c), abs_tol=1.0e-12
        )

    def test_cone_angle_at_u_pi_over_2(self, sub_solar_lmo_state):
        """At u = pi/2 (quarter orbit from initial), alpha(u) = alpha_0
        + alpha_s."""
        alpha_0 = math.radians(8.0)
        alpha_s = math.radians(4.0)
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        harmonic = sun_offset_harmonic(
            alpha_0,
            alpha_c_rad=0.0, alpha_s_rad=alpha_s, phi_u_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        # u = pi/2: r along e_ortho.
        R_sat = float(np.linalg.norm(sub_solar_lmo_state[:3]))
        r_quarter = R_sat * e_ortho
        et = utc_to_et(EPOCH_STR)
        n_h = harmonic(r_quarter, et)
        s_hat = sp(r_quarter, et)
        cos_angle = float(np.dot(n_h, s_hat))
        assert math.isclose(
            cos_angle, math.cos(alpha_0 + alpha_s), abs_tol=1.0e-12
        )

    def test_alpha_u_traces_harmonic_over_orbit(
        self, sub_solar_lmo_state
    ):
        """Sweep r through one orbit period in the orbit plane and verify
        cos(alpha(u)) traced from n_hat . s_hat equals
        cos(alpha_0 + alpha_c cos u + alpha_s sin u) at every u."""
        alpha_0 = math.radians(10.0)
        alpha_c = math.radians(3.0)
        alpha_s = math.radians(2.0)
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        harmonic = sun_offset_harmonic(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s, phi_u_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        et = utc_to_et(EPOCH_STR)
        R_sat = float(np.linalg.norm(sub_solar_lmo_state[:3]))
        for u in np.linspace(-math.pi, math.pi, 17, endpoint=False):
            r = R_sat * (math.cos(u) * e_ref + math.sin(u) * e_ortho)
            n_h = harmonic(r, et)
            s_hat = sp(r, et)
            cos_angle = float(np.dot(n_h, s_hat))
            expected = math.cos(
                alpha_0 + alpha_c * math.cos(u) + alpha_s * math.sin(u)
            )
            assert math.isclose(cos_angle, expected, abs_tol=1.0e-12), (
                f"u={u}: got cos(alpha)={cos_angle}, expected {expected}"
            )

    def test_unit_norm_preservation(self, epoch_et, sub_solar_lmo_state):
        """|n_hat| = 1 across multiple epochs and harmonic configs."""
        configs = [
            (math.radians(5.0), 0.0, 0.0),
            (math.radians(15.0), math.radians(5.0), 0.0),
            (math.radians(15.0), 0.0, math.radians(5.0)),
            (math.radians(20.0), math.radians(7.0), math.radians(7.0)),
        ]
        for alpha_0, alpha_c, alpha_s in configs:
            harmonic = sun_offset_harmonic_from_state(
                alpha_0,
                alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
                phi_u_rad=math.radians(30.0),
                initial_state_km_kmps=sub_solar_lmo_state,
            )
            for dt in [0.0, 1234.5, 45678.9]:
                et = epoch_et + dt
                n_hat = harmonic(sub_solar_lmo_state[:3], et)
                assert np.isclose(
                    np.linalg.norm(n_hat), 1.0, atol=1.0e-14
                ), f"|n_hat| != 1 for ({alpha_0}, {alpha_c}, {alpha_s}) at dt={dt}"

    def test_alpha_0_out_of_range_raises(self, sub_solar_lmo_state):
        """alpha_0 outside [0, pi/2] raises."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="alpha_0_rad"):
            sun_offset_harmonic(
                math.radians(91.0),
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )
        with pytest.raises(ValueError, match="alpha_0_rad"):
            sun_offset_harmonic(
                -0.01,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_saturated_cone_angle_upper_bound_raises(
        self, sub_solar_lmo_state
    ):
        """alpha_0 + sqrt(alpha_c^2 + alpha_s^2) > pi/2 raises.

        Example: alpha_0 = 50 deg, alpha_c = 25 deg, alpha_s = 25 deg
        gives amp = 35.36 deg, sum = 85.36 deg > 90 deg.
        """
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="<= pi/2"):
            sun_offset_harmonic(
                math.radians(70.0),
                alpha_c_rad=math.radians(25.0),
                alpha_s_rad=math.radians(25.0),
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_saturated_cone_angle_lower_bound_raises(
        self, sub_solar_lmo_state
    ):
        """alpha_0 - sqrt(alpha_c^2 + alpha_s^2) < 0 raises."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match=">= 0"):
            sun_offset_harmonic(
                math.radians(5.0),
                alpha_c_rad=math.radians(10.0),
                alpha_s_rad=0.0,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_zero_orbit_normal_raises(self):
        """Same validation contract as sun_offset."""
        with pytest.raises(ValueError, match="zero vector"):
            sun_offset_harmonic(
                math.radians(10.0),
                orbit_plane_normal_j2000=np.zeros(3),
                orbit_ref_direction_j2000=np.array([1.0, 0.0, 0.0]),
            )

    def test_alpha_amp_matches_analytic_dn_dt_magnitude(
        self, sub_solar_lmo_state
    ):
        """Analytical |dn_hat/dt| for harmonic alpha(u) at small alpha_amp
        should be ~ alpha_amp * (du/dt) when the offset_dir term dominates.

        Verified by central-difference of n_hat over the orbit plane at
        the canonical 501 km LMO mean motion n ~ sqrt(mu/a^3).

        The exact formula derived from differentiating Eq. 4.7 with
        alpha = alpha(u(t)) and circular u(t) = u_0 + n*t is non-trivial
        because n_hat involves both s_hat and offset_dir; this test
        instead pins the order-of-magnitude leading behaviour.
        """
        alpha_0 = math.radians(5.0)
        alpha_c = math.radians(2.0)
        alpha_s = math.radians(2.0)
        alpha_amp = math.hypot(alpha_c, alpha_s)
        # Mars 501 km LMO: a ~ 3897 km, mu ~ 42828.4 km^3/s^2.
        a_km = 3897.19
        mu_km3_s2 = 42828.37362069909
        n_orbit = math.sqrt(mu_km3_s2 / a_km**3)  # ~ 1.13e-3 rad/s
        # Sample r_sat sweeping u, central difference n_hat in time
        # by stepping r along the orbit.
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        harmonic = sun_offset_harmonic(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s, phi_u_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        et = utc_to_et(EPOCH_STR)
        R_sat = float(np.linalg.norm(sub_solar_lmo_state[:3]))
        u0 = 0.5  # rad, away from singular points
        du = 1.0e-4  # small u increment
        # dt corresponding to du at circular mean motion n_orbit.
        dt = du / n_orbit
        r_minus = R_sat * (
            math.cos(u0 - du) * e_ref + math.sin(u0 - du) * e_ortho
        )
        r_plus = R_sat * (
            math.cos(u0 + du) * e_ref + math.sin(u0 + du) * e_ortho
        )
        n_minus = harmonic(r_minus, et - dt)
        n_plus = harmonic(r_plus, et + dt)
        dn_dt = (n_plus - n_minus) / (2.0 * dt)
        dn_dt_mag = float(np.linalg.norm(dn_dt))
        # Order-of-magnitude bound: |dn_hat/dt| should be of order
        # max(alpha_amp, 1) * n_orbit. With alpha_amp ~ 0.05 rad and
        # n_orbit ~ 1.13e-3, dn_dt_mag is dominated by the offset_dir
        # rotation in u (which is ~ 1 * n_orbit = 1.13e-3, since
        # offset_dir rotates with u at rate n_orbit and sin(alpha) ~ 0.1).
        # Pin upper bound: dn_dt_mag < 5 * n_orbit.
        assert dn_dt_mag < 5.0 * n_orbit, (
            f"|dn_hat/dt| = {dn_dt_mag:.3e} rad/s, "
            f"expected ~ O(alpha_amp + sin(alpha)) * n_orbit "
            f"~ {n_orbit:.3e} rad/s"
        )
        # And lower bound: must be at least alpha_amp * n_orbit / 2 to
        # confirm the harmonic IS being evaluated.
        assert dn_dt_mag > 0.5 * alpha_amp * n_orbit, (
            f"|dn_hat/dt| = {dn_dt_mag:.3e} rad/s, "
            f"expected at least {0.5 * alpha_amp * n_orbit:.3e}"
        )


# ---------------------------------------------------------------------------
# Group 6: Harmonic-(alpha, delta) cruise family
# ---------------------------------------------------------------------------


class TestSunOffsetHarmonicFull:
    """Tests for ``sun_offset_harmonic_full`` (Fourier-mode-1 in BOTH
    cone angle alpha AND clock angle delta).

    Reduction-identity tests pin two contracts:
      - delta_c = delta_s = 0 collapses to ``sun_offset_harmonic`` with
        phi_u = delta_0.
      - All amplitudes zero collapses to ``sun_offset(alpha_0, delta_0)``
        (constant-family reduction).

    Geometry tests verify alpha(u) and delta(u) Fourier components at
    u = 0, pi/2, pi, 3pi/2 and trace them across one orbit period.
    """

    # ---- Reduction identities ----

    def test_zero_delta_amplitudes_matches_harmonic_alpha_bit_for_bit(
        self, epoch_et, sub_solar_lmo_state
    ):
        """delta_c = delta_s = 0 reduces to sun_offset_harmonic(alpha_0,
        alpha_c, alpha_s, phi_u=delta_0) bit-for-bit."""
        alpha_0 = math.radians(7.5)
        alpha_c = math.radians(3.0)
        alpha_s = math.radians(2.0)
        delta_0 = math.radians(45.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        harmonic = sun_offset_harmonic(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s, phi_u_rad=delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        full = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
            delta_0_rad=delta_0, delta_c_rad=0.0, delta_s_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0, 86400.0]:
            et = epoch_et + dt
            n_h = harmonic(r_sat, et)
            n_f = full(r_sat, et)
            np.testing.assert_array_equal(n_h, n_f)

    def test_zero_alpha_amplitudes_zero_delta_amplitudes_matches_constant_offset(
        self, epoch_et, sub_solar_lmo_state
    ):
        """alpha_c = alpha_s = delta_c = delta_s = 0 reduces to
        sun_offset(alpha_0, delta_0) bit-for-bit."""
        alpha_0 = math.radians(10.0)
        delta_0 = math.radians(60.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        constant = sun_offset(
            alpha_0, delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        full = sun_offset_harmonic_full(
            alpha_0,
            delta_0_rad=delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0]:
            et = epoch_et + dt
            n_c = constant(r_sat, et)
            n_f = full(r_sat, et)
            np.testing.assert_array_equal(n_c, n_f)

    def test_zero_everything_matches_sun_pointing(
        self, epoch_et, sub_solar_lmo_state
    ):
        """All six amplitudes = 0 with alpha_0 = delta_0 = 0 reduces
        bit-for-bit to sun_pointing."""
        full = sun_offset_harmonic_full_from_state(
            0.0,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        sp = sun_pointing()
        n_f = full(sub_solar_lmo_state[:3], epoch_et)
        n_sp = sp(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_allclose(n_f, n_sp, atol=1.0e-15)

    def test_zero_delta_amplitudes_via_from_state_helper(
        self, epoch_et, sub_solar_lmo_state
    ):
        """from_state wrapper preserves the delta_c = delta_s = 0 reduction."""
        alpha_0 = math.radians(8.0)
        alpha_c = math.radians(2.0)
        delta_0 = math.radians(30.0)
        harmonic = sun_offset_harmonic_from_state(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=0.0, phi_u_rad=delta_0,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        full = sun_offset_harmonic_full_from_state(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=0.0,
            delta_0_rad=delta_0,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        n_h = harmonic(sub_solar_lmo_state[:3], epoch_et)
        n_f = full(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_array_equal(n_h, n_f)

    # ---- alpha and delta recovery at specific u ----

    @staticmethod
    def _recover_alpha_delta(
        n_hat: np.ndarray, s_hat: np.ndarray,
        e_A: np.ndarray, e_B: np.ndarray,
        u: float,
    ) -> tuple[float, float]:
        """Recover (alpha(u), delta(u)) from n_hat by projection.

        alpha(u): cos(alpha) = n_hat . s_hat -> alpha = acos(n_hat.s_hat)
        delta(u): n_hat - cos(alpha) * s_hat = sin(alpha) * offset_dir,
                 where offset_dir = cos(u + delta) * e_A + sin(u + delta) * e_B.
        Returns (alpha, delta) where delta is normalised to [-pi, pi).
        """
        cos_alpha = float(np.dot(n_hat, s_hat))
        # Clamp for floating-point safety
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)
        sin_alpha = math.sin(alpha)
        if sin_alpha == 0.0:
            return alpha, 0.0
        offset = (n_hat - cos_alpha * s_hat) / sin_alpha
        x = float(np.dot(offset, e_A))
        y = float(np.dot(offset, e_B))
        u_plus_delta = math.atan2(y, x)
        delta = u_plus_delta - u
        # Normalise to [-pi, pi)
        while delta >= math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return alpha, delta

    def test_recovery_at_u_zero(self, sub_solar_lmo_state):
        """At u = 0 (initial position), alpha(0) = alpha_0 + alpha_c
        and delta(0) = delta_0 + delta_c."""
        alpha_0 = math.radians(8.0)
        alpha_c = math.radians(3.0)
        alpha_s = math.radians(1.0)
        delta_0 = math.radians(60.0)
        delta_c = math.radians(15.0)
        delta_s = math.radians(5.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        full = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
            delta_0_rad=delta_0, delta_c_rad=delta_c, delta_s_rad=delta_s,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        r_sat = sub_solar_lmo_state[:3]
        et = utc_to_et(EPOCH_STR)
        n_hat = full(r_sat, et)
        s_hat = sp(r_sat, et)
        # Reconstruct e_A, e_B for projection (matches _per_call_geometry).
        n_perp = n_orb - float(np.dot(n_orb, s_hat)) * s_hat
        e_A = n_perp / float(np.linalg.norm(n_perp))
        e_B = np.cross(s_hat, e_A)
        alpha_recovered, delta_recovered = self._recover_alpha_delta(
            n_hat, s_hat, e_A, e_B, u=0.0,
        )
        assert math.isclose(
            alpha_recovered, alpha_0 + alpha_c, abs_tol=1.0e-12
        ), f"alpha(0): got {alpha_recovered}, expected {alpha_0 + alpha_c}"
        assert math.isclose(
            delta_recovered, delta_0 + delta_c, abs_tol=1.0e-12
        ), f"delta(0): got {delta_recovered}, expected {delta_0 + delta_c}"

    def test_recovery_at_u_pi_over_2(self, sub_solar_lmo_state):
        """At u = pi/2 (quarter orbit), alpha(pi/2) = alpha_0 + alpha_s
        and delta(pi/2) = delta_0 + delta_s."""
        alpha_0 = math.radians(10.0)
        alpha_c = math.radians(2.0)
        alpha_s = math.radians(4.0)
        delta_0 = math.radians(30.0)
        delta_c = math.radians(8.0)
        delta_s = math.radians(20.0)
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        full = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
            delta_0_rad=delta_0, delta_c_rad=delta_c, delta_s_rad=delta_s,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        et = utc_to_et(EPOCH_STR)
        # u = pi/2: r along e_ortho.
        R_sat = float(np.linalg.norm(sub_solar_lmo_state[:3]))
        r_quarter = R_sat * e_ortho
        n_hat = full(r_quarter, et)
        s_hat = sp(r_quarter, et)
        n_perp = n_orb - float(np.dot(n_orb, s_hat)) * s_hat
        e_A = n_perp / float(np.linalg.norm(n_perp))
        e_B = np.cross(s_hat, e_A)
        alpha_recovered, delta_recovered = self._recover_alpha_delta(
            n_hat, s_hat, e_A, e_B, u=math.pi / 2.0,
        )
        assert math.isclose(
            alpha_recovered, alpha_0 + alpha_s, abs_tol=1.0e-12
        )
        assert math.isclose(
            delta_recovered, delta_0 + delta_s, abs_tol=1.0e-12
        )

    def test_recovery_at_u_pi(self, sub_solar_lmo_state):
        """At u = pi (antipodal), alpha(pi) = alpha_0 - alpha_c and
        delta(pi) = delta_0 - delta_c."""
        alpha_0 = math.radians(10.0)
        alpha_c = math.radians(3.0)
        alpha_s = math.radians(1.0)
        delta_0 = math.radians(45.0)
        delta_c = math.radians(15.0)
        delta_s = math.radians(5.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        full = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
            delta_0_rad=delta_0, delta_c_rad=delta_c, delta_s_rad=delta_s,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        et = utc_to_et(EPOCH_STR)
        # Antipodal: r = -r_0, u = pi.
        r_anti = -np.asarray(sub_solar_lmo_state[:3])
        n_hat = full(r_anti, et)
        s_hat = sp(r_anti, et)
        n_perp = n_orb - float(np.dot(n_orb, s_hat)) * s_hat
        e_A = n_perp / float(np.linalg.norm(n_perp))
        e_B = np.cross(s_hat, e_A)
        alpha_recovered, delta_recovered = self._recover_alpha_delta(
            n_hat, s_hat, e_A, e_B, u=math.pi,
        )
        assert math.isclose(
            alpha_recovered, alpha_0 - alpha_c, abs_tol=1.0e-12
        )
        assert math.isclose(
            delta_recovered, delta_0 - delta_c, abs_tol=1.0e-12
        )

    def test_alpha_delta_trace_harmonic_over_orbit(
        self, sub_solar_lmo_state
    ):
        """Sweep r through one orbit period in the orbit plane and verify
        the recovered (alpha(u), delta(u)) follow the prescribed
        Fourier-mode-1 modulation at every u sample."""
        alpha_0 = math.radians(10.0)
        alpha_c = math.radians(3.0)
        alpha_s = math.radians(2.0)
        delta_0 = math.radians(60.0)
        delta_c = math.radians(15.0)
        delta_s = math.radians(10.0)
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        full = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
            delta_0_rad=delta_0, delta_c_rad=delta_c, delta_s_rad=delta_s,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        sp = sun_pointing()
        et = utc_to_et(EPOCH_STR)
        R_sat = float(np.linalg.norm(sub_solar_lmo_state[:3]))
        for u in np.linspace(-math.pi, math.pi, 17, endpoint=False):
            r = R_sat * (math.cos(u) * e_ref + math.sin(u) * e_ortho)
            n_hat = full(r, et)
            s_hat = sp(r, et)
            n_perp = n_orb - float(np.dot(n_orb, s_hat)) * s_hat
            e_A = n_perp / float(np.linalg.norm(n_perp))
            e_B = np.cross(s_hat, e_A)
            alpha_rec, delta_rec = self._recover_alpha_delta(
                n_hat, s_hat, e_A, e_B, u=float(u),
            )
            alpha_expected = (
                alpha_0 + alpha_c * math.cos(u) + alpha_s * math.sin(u)
            )
            delta_expected_raw = (
                delta_0 + delta_c * math.cos(u) + delta_s * math.sin(u)
            )
            # Normalise expected to [-pi, pi) for fair comparison.
            delta_expected = delta_expected_raw
            while delta_expected >= math.pi:
                delta_expected -= 2.0 * math.pi
            while delta_expected < -math.pi:
                delta_expected += 2.0 * math.pi
            assert math.isclose(alpha_rec, alpha_expected, abs_tol=1.0e-12), (
                f"u={u}: alpha got {alpha_rec}, expected {alpha_expected}"
            )
            assert math.isclose(delta_rec, delta_expected, abs_tol=1.0e-12), (
                f"u={u}: delta got {delta_rec}, expected {delta_expected}"
            )

    # ---- Validation contracts ----

    def test_alpha_amp_violation_raises(self, sub_solar_lmo_state):
        """Same Cauchy-Schwarz alpha bound as the cone-only family."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="alpha_0 - sqrt"):
            sun_offset_harmonic_full(
                math.radians(5.0),
                alpha_c_rad=math.radians(10.0),
                alpha_s_rad=0.0,
                delta_0_rad=0.0,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_alpha_amp_upper_bound_violation_raises(self, sub_solar_lmo_state):
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="alpha_0 \\+ sqrt"):
            sun_offset_harmonic_full(
                math.radians(70.0),
                alpha_c_rad=math.radians(25.0),
                alpha_s_rad=math.radians(25.0),
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_alpha_0_out_of_range_raises(self, sub_solar_lmo_state):
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="alpha_0_rad"):
            sun_offset_harmonic_full(
                math.radians(91.0),
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_delta_amp_default_violation_raises(self, sub_solar_lmo_state):
        """delta_amp > pi/2 by default raises."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        # delta_c = delta_s = pi/2 each -> amp = pi/sqrt(2) ~ 2.22 > pi/2.
        with pytest.raises(ValueError, match="delta_amp_max_rad"):
            sun_offset_harmonic_full(
                math.radians(10.0),
                delta_c_rad=math.pi / 2.0,
                delta_s_rad=math.pi / 2.0,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_delta_amp_custom_max_passes(self, sub_solar_lmo_state):
        """Caller can widen the soft bound via delta_amp_max_rad kwarg."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        # Same config as above but raise the cap.
        full = sun_offset_harmonic_full(
            math.radians(10.0),
            delta_c_rad=math.pi / 2.0,
            delta_s_rad=math.pi / 2.0,
            delta_amp_max_rad=2.0 * math.pi,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        # Smoke test: the closure is callable without crashing.
        n_hat = full(np.array([3897.19, 0.0, 0.0]), 0.0)
        assert np.isclose(np.linalg.norm(n_hat), 1.0, atol=1.0e-14)

    def test_negative_delta_amp_max_raises(self, sub_solar_lmo_state):
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="delta_amp_max_rad must be >= 0"):
            sun_offset_harmonic_full(
                math.radians(5.0),
                delta_amp_max_rad=-0.1,
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_zero_orbit_normal_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            sun_offset_harmonic_full(
                math.radians(10.0),
                orbit_plane_normal_j2000=np.zeros(3),
                orbit_ref_direction_j2000=np.array([1.0, 0.0, 0.0]),
            )

    def test_default_delta_amp_max_is_pi_over_2(self):
        """Pin the canonical default value."""
        assert math.isclose(DELTA_AMP_MAX_DEFAULT_RAD, math.pi / 2.0)

    # ---- Unit-norm preservation ----

    def test_unit_norm_preservation(self, epoch_et, sub_solar_lmo_state):
        """|n_hat| = 1 across multiple epochs and (alpha_*, delta_*) configs."""
        configs = [
            # (alpha_0, alpha_c, alpha_s, delta_0, delta_c, delta_s)
            (math.radians(5.0), 0.0, 0.0, math.radians(45.0), 0.0, 0.0),
            (math.radians(10.0), math.radians(3.0), 0.0,
             math.radians(60.0), math.radians(15.0), 0.0),
            (math.radians(10.0), 0.0, math.radians(3.0),
             math.radians(60.0), 0.0, math.radians(15.0)),
            (math.radians(15.0), math.radians(5.0), math.radians(3.0),
             math.radians(120.0), math.radians(15.0), math.radians(20.0)),
        ]
        for alpha_0, alpha_c, alpha_s, delta_0, delta_c, delta_s in configs:
            full = sun_offset_harmonic_full_from_state(
                alpha_0,
                alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
                delta_0_rad=delta_0, delta_c_rad=delta_c, delta_s_rad=delta_s,
                initial_state_km_kmps=sub_solar_lmo_state,
            )
            for dt in [0.0, 1234.5, 45678.9]:
                et = epoch_et + dt
                n_hat = full(sub_solar_lmo_state[:3], et)
                assert np.isclose(
                    np.linalg.norm(n_hat), 1.0, atol=1.0e-14
                ), f"|n_hat| != 1 for {(alpha_0, alpha_c, alpha_s, delta_0, delta_c, delta_s)} at dt={dt}"

    # ---- |dn_hat/dt| order-of-magnitude with delta_amp active ----

    def test_dn_dt_magnitude_with_harmonic_delta(self, sub_solar_lmo_state):
        """With delta_amp active, |dn_hat/dt| picks up an additional
        d(delta)/du * n_orbit contribution to the offset_dir rotation rate.
        Pin order-of-magnitude (not exact)."""
        alpha_0 = math.radians(5.0)
        alpha_c = math.radians(2.0)
        alpha_s = math.radians(2.0)
        delta_0 = math.radians(45.0)
        delta_c = math.radians(20.0)
        delta_s = math.radians(20.0)
        delta_amp = math.hypot(delta_c, delta_s)
        # Mars 501 km LMO: a ~ 3897 km, mu ~ 42828.4 km^3/s^2.
        a_km = 3897.19
        mu_km3_s2 = 42828.37362069909
        n_orbit = math.sqrt(mu_km3_s2 / a_km**3)
        e_ref, e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        full = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
            delta_0_rad=delta_0, delta_c_rad=delta_c, delta_s_rad=delta_s,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        et = utc_to_et(EPOCH_STR)
        R_sat = float(np.linalg.norm(sub_solar_lmo_state[:3]))
        u0 = 0.5  # rad
        du = 1.0e-4
        dt = du / n_orbit
        r_minus = R_sat * (
            math.cos(u0 - du) * e_ref + math.sin(u0 - du) * e_ortho
        )
        r_plus = R_sat * (
            math.cos(u0 + du) * e_ref + math.sin(u0 + du) * e_ortho
        )
        n_minus = full(r_minus, et - dt)
        n_plus = full(r_plus, et + dt)
        dn_dt_mag = float(np.linalg.norm((n_plus - n_minus) / (2.0 * dt)))
        # Upper bound: with delta_amp ~ 0.49 rad, the offset_dir rotates
        # at rate (1 + dδ/du) * n_orbit <= (1 + delta_amp) * n_orbit
        # in the sin(alpha) * offset_dir term. Including the cos(alpha)*ds_hat/dt
        # term (small at LMO scales), conservative bound: 5 * n_orbit.
        assert dn_dt_mag < 5.0 * n_orbit, (
            f"|dn_hat/dt| = {dn_dt_mag:.3e} rad/s, expected < {5.0 * n_orbit:.3e}"
        )
        # Lower bound: at least sin(alpha_0) * n_orbit / 2 (the offset_dir
        # is being rotated at rate ~ n_orbit). With sin(alpha_0) ~ 0.087,
        # this is 0.087/2 * 1.13e-3 ~ 5e-5 rad/s.
        assert dn_dt_mag > 0.5 * math.sin(alpha_0) * n_orbit, (
            f"|dn_hat/dt| = {dn_dt_mag:.3e} rad/s, "
            f"expected at least {0.5 * math.sin(alpha_0) * n_orbit:.3e}"
        )


# ---------------------------------------------------------------------------
# Mode-2 harmonic family
# ---------------------------------------------------------------------------


class TestSunOffsetHarmonicFullMode2:
    """Tests for ``sun_offset_harmonic_full_mode2`` (mode-1 + mode-2 in
    BOTH alpha and delta).

    Reduction-identity tests pin: with all mode-2 amplitudes zero,
    ``sun_offset_harmonic_full_mode2`` collapses to
    ``sun_offset_harmonic_full(alpha_0, alpha_c=alpha_c1,
    alpha_s=alpha_s1, delta_0, delta_c=delta_c1, delta_s=delta_s1)``
    bit-for-bit. Validation tests pin the conservative cone-angle
    bound and delta-amplitude triangle inequality.
    """

    def test_zero_mode2_matches_mode1_bit_for_bit(
        self, epoch_et, sub_solar_lmo_state,
    ):
        alpha_0 = math.radians(10.0)
        a_c1 = math.radians(3.0)
        a_s1 = math.radians(2.0)
        delta_0 = math.radians(45.0)
        d_c1 = math.radians(8.0)
        d_s1 = math.radians(-5.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        mode1 = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=a_c1, alpha_s_rad=a_s1,
            delta_0_rad=delta_0, delta_c_rad=d_c1, delta_s_rad=d_s1,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        mode2 = sun_offset_harmonic_full_mode2(
            alpha_0,
            alpha_c1_rad=a_c1, alpha_s1_rad=a_s1,
            alpha_c2_rad=0.0, alpha_s2_rad=0.0,
            delta_0_rad=delta_0,
            delta_c1_rad=d_c1, delta_s1_rad=d_s1,
            delta_c2_rad=0.0, delta_s2_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0, 86400.0]:
            et = epoch_et + dt
            n_1 = mode1(r_sat, et)
            n_2 = mode2(r_sat, et)
            np.testing.assert_array_equal(n_1, n_2)

    def test_zero_all_amplitudes_matches_constant_offset(
        self, epoch_et, sub_solar_lmo_state,
    ):
        alpha_0 = math.radians(15.0)
        delta_0 = math.radians(60.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        constant = sun_offset(
            alpha_0, delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        mode2 = sun_offset_harmonic_full_mode2(
            alpha_0, delta_0_rad=delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0]:
            et = epoch_et + dt
            n_c = constant(r_sat, et)
            n_2 = mode2(r_sat, et)
            np.testing.assert_array_equal(n_c, n_2)

    def test_from_state_wrapper_matches_direct_call(
        self, epoch_et, sub_solar_lmo_state,
    ):
        alpha_0 = math.radians(8.0)
        a_c1 = math.radians(1.5)
        a_s2 = math.radians(0.5)
        delta_0 = math.radians(30.0)
        d_c2 = math.radians(7.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        direct = sun_offset_harmonic_full_mode2(
            alpha_0,
            alpha_c1_rad=a_c1, alpha_s2_rad=a_s2,
            delta_0_rad=delta_0, delta_c2_rad=d_c2,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        wrapper = sun_offset_harmonic_full_mode2_from_state(
            alpha_0,
            alpha_c1_rad=a_c1, alpha_s2_rad=a_s2,
            delta_0_rad=delta_0, delta_c2_rad=d_c2,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0]:
            et = epoch_et + dt
            np.testing.assert_array_equal(direct(r_sat, et), wrapper(r_sat, et))

    def test_cone_angle_conservative_bound_violation_raises(
        self, sub_solar_lmo_state,
    ):
        """Conservative cone-angle bound: alpha_0 + amp1 + amp2 must <= pi/2.

        Set alpha_0 = 30°, amp1 = 35° (just under), amp2 = 35° -> sum = 100°
        > 90°. The conservative bound triggers even though a tighter
        analytic peak may be lower; the conservative bound still applies.
        """
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="alpha_0 \\+"):
            sun_offset_harmonic_full_mode2(
                math.radians(30.0),
                alpha_c1_rad=math.radians(35.0),
                alpha_c2_rad=math.radians(35.0),
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_delta_amp_conservative_bound_violation_raises(
        self, sub_solar_lmo_state,
    ):
        """Conservative delta-amp bound: |amp1| + |amp2| <= delta_amp_max."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="delta_c1"):
            sun_offset_harmonic_full_mode2(
                math.radians(10.0),
                delta_c1_rad=math.radians(80.0),
                delta_c2_rad=math.radians(80.0),  # sum amps = 160° > 90° default cap
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_mode2_alpha_oscillates_at_double_frequency_in_u(
        self, epoch_et, sub_solar_lmo_state,
    ):
        """With pure mode-2 alpha (alpha_c2 only, mode-1 alpha = 0 and
        delta = 0), n_hat at u and u+pi should be IDENTICAL (mode-2 is
        pi-periodic in u), whereas mode-1 alpha would flip the sign of
        the alpha-deviation between u and u+pi."""
        alpha_0 = math.radians(20.0)
        a_c2 = math.radians(5.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        mode2 = sun_offset_harmonic_full_mode2(
            alpha_0,
            alpha_c2_rad=a_c2,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        # Sample alpha(u) at u = 0 (mode2 cos2u = 1) and u = pi (cos2u = 1)
        # by varying the satellite's argument-of-latitude. For e=0, u
        # advances linearly with mean anomaly. Easiest: place satellite
        # at +e_ref direction (u=0) and -e_ref direction (u=pi).
        n_at_u0 = mode2(sub_solar_lmo_state[:3], epoch_et)
        # u = pi: r along -e_ref. Note |r| same, but pose changes.
        r_pi = -np.asarray(sub_solar_lmo_state[:3], dtype=float)
        n_at_upi = mode2(r_pi, epoch_et)
        # alpha at u=0 and u=pi should be the same (cos(2*0) = cos(2*pi) = 1).
        # Therefore the cone-angle component cos(alpha)*s_hat is identical
        # if s_hat is identical -- but s_hat depends on r_sat. So instead
        # of full vector identity, project onto s_hat to extract cos(alpha).
        # cos(alpha) = n_hat . s_hat at each sample.
        import spiceypy as spice
        SUN = 10
        MARS = 499
        state_sun, _ = spice.spkezr(str(SUN), epoch_et, "J2000", "NONE", str(MARS))
        r_sun = np.asarray(state_sun[:3], dtype=float)
        s0 = (r_sun - sub_solar_lmo_state[:3])
        s0 = s0 / np.linalg.norm(s0)
        s_pi = (r_sun - r_pi)
        s_pi = s_pi / np.linalg.norm(s_pi)
        cos_alpha_0 = float(np.dot(n_at_u0, s0))
        cos_alpha_pi = float(np.dot(n_at_upi, s_pi))
        np.testing.assert_allclose(cos_alpha_0, cos_alpha_pi, atol=1.0e-12)

    def test_mode2_constructs_valid_unit_normal(
        self, epoch_et, sub_solar_lmo_state,
    ):
        """All-modes-active sample produces unit n_hat."""
        mode2 = sun_offset_harmonic_full_mode2_from_state(
            math.radians(15.0),
            alpha_c1_rad=math.radians(2.0),
            alpha_s1_rad=math.radians(1.5),
            alpha_c2_rad=math.radians(0.5),
            alpha_s2_rad=math.radians(0.3),
            delta_0_rad=math.radians(180.0),
            delta_c1_rad=math.radians(15.0),
            delta_s1_rad=math.radians(-10.0),
            delta_c2_rad=math.radians(5.0),
            delta_s2_rad=math.radians(3.0),
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        n = mode2(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1.0e-12)


# ---------------------------------------------------------------------------
# Mode-3 harmonic family
# ---------------------------------------------------------------------------


class TestSunOffsetHarmonicFullMode3:
    """Tests for ``sun_offset_harmonic_full_mode3`` (mode-1 + mode-2 +
    mode-3 in BOTH alpha and delta).

    Reduction-identity tests pin: with all mode-3 amplitudes zero,
    ``sun_offset_harmonic_full_mode3`` collapses to
    ``sun_offset_harmonic_full_mode2`` bit-for-bit; further setting all
    mode-2 amplitudes to zero collapses to ``sun_offset_harmonic_full``
    (mode-1). Validation tests pin the conservative cone-angle bound
    and delta-amplitude triangle inequality across all three modes.
    """

    def test_zero_mode3_matches_mode2_bit_for_bit(
        self, epoch_et, sub_solar_lmo_state,
    ):
        alpha_0 = math.radians(10.0)
        a_c1, a_s1 = math.radians(3.0), math.radians(2.0)
        a_c2, a_s2 = math.radians(1.5), math.radians(-1.0)
        delta_0 = math.radians(45.0)
        d_c1, d_s1 = math.radians(8.0), math.radians(-5.0)
        d_c2, d_s2 = math.radians(4.0), math.radians(2.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        mode2 = sun_offset_harmonic_full_mode2(
            alpha_0,
            alpha_c1_rad=a_c1, alpha_s1_rad=a_s1,
            alpha_c2_rad=a_c2, alpha_s2_rad=a_s2,
            delta_0_rad=delta_0,
            delta_c1_rad=d_c1, delta_s1_rad=d_s1,
            delta_c2_rad=d_c2, delta_s2_rad=d_s2,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        mode3 = sun_offset_harmonic_full_mode3(
            alpha_0,
            alpha_c1_rad=a_c1, alpha_s1_rad=a_s1,
            alpha_c2_rad=a_c2, alpha_s2_rad=a_s2,
            alpha_c3_rad=0.0, alpha_s3_rad=0.0,
            delta_0_rad=delta_0,
            delta_c1_rad=d_c1, delta_s1_rad=d_s1,
            delta_c2_rad=d_c2, delta_s2_rad=d_s2,
            delta_c3_rad=0.0, delta_s3_rad=0.0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0, 86400.0]:
            et = epoch_et + dt
            np.testing.assert_array_equal(mode2(r_sat, et), mode3(r_sat, et))

    def test_zero_higher_modes_matches_mode1_bit_for_bit(
        self, epoch_et, sub_solar_lmo_state,
    ):
        """Setting mode-2 AND mode-3 amplitudes to zero should reduce
        bit-for-bit through to mode-1 (sun_offset_harmonic_full)."""
        alpha_0 = math.radians(8.0)
        a_c1, a_s1 = math.radians(2.0), math.radians(1.5)
        delta_0 = math.radians(60.0)
        d_c1, d_s1 = math.radians(6.0), math.radians(-3.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        mode1 = sun_offset_harmonic_full(
            alpha_0,
            alpha_c_rad=a_c1, alpha_s_rad=a_s1,
            delta_0_rad=delta_0, delta_c_rad=d_c1, delta_s_rad=d_s1,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        mode3 = sun_offset_harmonic_full_mode3(
            alpha_0,
            alpha_c1_rad=a_c1, alpha_s1_rad=a_s1,
            delta_0_rad=delta_0,
            delta_c1_rad=d_c1, delta_s1_rad=d_s1,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0]:
            et = epoch_et + dt
            np.testing.assert_array_equal(mode1(r_sat, et), mode3(r_sat, et))

    def test_zero_all_amplitudes_matches_constant_offset(
        self, epoch_et, sub_solar_lmo_state,
    ):
        alpha_0 = math.radians(15.0)
        delta_0 = math.radians(60.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        constant = sun_offset(
            alpha_0, delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        mode3 = sun_offset_harmonic_full_mode3(
            alpha_0, delta_0_rad=delta_0,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0]:
            et = epoch_et + dt
            np.testing.assert_array_equal(constant(r_sat, et), mode3(r_sat, et))

    def test_from_state_wrapper_matches_direct_call(
        self, epoch_et, sub_solar_lmo_state,
    ):
        alpha_0 = math.radians(8.0)
        a_c3 = math.radians(0.5)
        a_s2 = math.radians(0.3)
        delta_0 = math.radians(30.0)
        d_c3 = math.radians(2.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        direct = sun_offset_harmonic_full_mode3(
            alpha_0,
            alpha_s2_rad=a_s2, alpha_c3_rad=a_c3,
            delta_0_rad=delta_0, delta_c3_rad=d_c3,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        wrapper = sun_offset_harmonic_full_mode3_from_state(
            alpha_0,
            alpha_s2_rad=a_s2, alpha_c3_rad=a_c3,
            delta_0_rad=delta_0, delta_c3_rad=d_c3,
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        r_sat = sub_solar_lmo_state[:3]
        for dt in [0.0, 60.0, 3600.0]:
            et = epoch_et + dt
            np.testing.assert_array_equal(direct(r_sat, et), wrapper(r_sat, et))

    def test_cone_angle_conservative_bound_violation_raises(
        self, sub_solar_lmo_state,
    ):
        """alpha_0 + amp1 + amp2 + amp3 must <= pi/2.

        Set alpha_0 = 30°, amp1 = 25°, amp2 = 25°, amp3 = 15° -> sum = 95° > 90°.
        """
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="alpha_0 \\+"):
            sun_offset_harmonic_full_mode3(
                math.radians(30.0),
                alpha_c1_rad=math.radians(25.0),
                alpha_c2_rad=math.radians(25.0),
                alpha_c3_rad=math.radians(15.0),
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_delta_amp_conservative_bound_violation_raises(
        self, sub_solar_lmo_state,
    ):
        """|amp1| + |amp2| + |amp3| (delta) <= delta_amp_max."""
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        with pytest.raises(ValueError, match="delta_c1"):
            sun_offset_harmonic_full_mode3(
                math.radians(10.0),
                delta_c1_rad=math.radians(40.0),
                delta_c2_rad=math.radians(40.0),
                delta_c3_rad=math.radians(40.0),  # 120° > 90° default cap
                orbit_plane_normal_j2000=n_orb,
                orbit_ref_direction_j2000=e_ref,
            )

    def test_mode3_alpha_oscillates_at_triple_frequency_in_u(
        self, epoch_et, sub_solar_lmo_state,
    ):
        """With pure mode-3 alpha (alpha_c3 only, all other amplitudes
        zero, delta = 0), alpha at u and u + 2*pi/3 should be IDENTICAL
        (mode-3 is 2*pi/3-periodic in u). Verify by sampling cos(alpha)
        at the satellite's argument-of-latitude u=0 and at u=2*pi/3 (rotate
        r_sat by 120° in the orbit plane)."""
        alpha_0 = math.radians(15.0)
        a_c3 = math.radians(2.0)
        e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(sub_solar_lmo_state)
        mode3 = sun_offset_harmonic_full_mode3(
            alpha_0,
            alpha_c3_rad=a_c3,
            orbit_plane_normal_j2000=n_orb,
            orbit_ref_direction_j2000=e_ref,
        )
        # Sample at u=0 (along +e_ref) and u=2*pi/3 (rotate r_sat by 120°
        # about n_orb). Mode-3 cos(3u) is the same at u=0 and u=2*pi/3
        # (cos(0) = cos(2*pi) = 1).
        r_u0 = np.asarray(sub_solar_lmo_state[:3], dtype=float)
        # r_sat is along +e_ref at u=0 by the basis convention; |r| sets
        # altitude. To get u=2*pi/3, rotate r_u0 by 2*pi/3 about n_orb
        # using Rodrigues' formula.
        n = np.asarray(n_orb, dtype=float)
        n = n / float(np.linalg.norm(n))
        theta = 2.0 * math.pi / 3.0
        K = np.array([
            [0.0, -n[2], n[1]],
            [n[2], 0.0, -n[0]],
            [-n[1], n[0], 0.0],
        ])
        R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * K @ K
        r_u120 = R @ r_u0

        n_at_u0 = mode3(r_u0, epoch_et)
        n_at_u120 = mode3(r_u120, epoch_et)
        # Project both onto sun-pointing direction at the respective r_sat
        # to extract cos(alpha). At a fixed epoch, s_hat depends on r_sat,
        # so compute s_hat per sample.
        import spiceypy as spice
        SUN = 10
        MARS = 499
        state_sun, _ = spice.spkezr(str(SUN), epoch_et, "J2000", "NONE", str(MARS))
        r_sun = np.asarray(state_sun[:3], dtype=float)
        s_u0 = (r_sun - r_u0) / np.linalg.norm(r_sun - r_u0)
        s_u120 = (r_sun - r_u120) / np.linalg.norm(r_sun - r_u120)
        cos_alpha_u0 = float(np.dot(n_at_u0, s_u0))
        cos_alpha_u120 = float(np.dot(n_at_u120, s_u120))
        np.testing.assert_allclose(cos_alpha_u0, cos_alpha_u120, atol=1.0e-12)

    def test_mode3_constructs_valid_unit_normal(
        self, epoch_et, sub_solar_lmo_state,
    ):
        """All-modes-active (1, 2, 3) sample produces unit n_hat."""
        mode3 = sun_offset_harmonic_full_mode3_from_state(
            math.radians(15.0),
            alpha_c1_rad=math.radians(2.0), alpha_s1_rad=math.radians(1.5),
            alpha_c2_rad=math.radians(0.5), alpha_s2_rad=math.radians(0.3),
            alpha_c3_rad=math.radians(0.2), alpha_s3_rad=math.radians(0.1),
            delta_0_rad=math.radians(180.0),
            delta_c1_rad=math.radians(15.0), delta_s1_rad=math.radians(-10.0),
            delta_c2_rad=math.radians(5.0), delta_s2_rad=math.radians(3.0),
            delta_c3_rad=math.radians(2.0), delta_s3_rad=math.radians(-1.0),
            initial_state_km_kmps=sub_solar_lmo_state,
        )
        n = mode3(sub_solar_lmo_state[:3], epoch_et)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1.0e-12)
