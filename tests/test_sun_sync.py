"""Tests for ``reflectors.sun_sync``: Brouwer inclination, LTAN->RAAN,
and the composed forward transformation (a, LTAN, M0, epoch) -> state.

Anchors:
  sun-sync inclination at 400 km altitude ~ 92.92 deg from the Brouwer
  derivation.
  LTAN = 18h orbit has its angular-momentum vector perpendicular to the
  Mars-Sun line (dawn-dusk geometry).
  state_from_classical_mme2000 -> elements_in_mme2000 round-trip recovers
  the inputs at machine precision for well-conditioned cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.elements import (
    elements_in_mme2000,
    mme2000_rotation_from_j2000,
    state_from_classical_mme2000,
)
from reflectors.ephemeris import utc_to_et
from reflectors.surface import mars_equatorial_radius_km
from reflectors.mars_constants import (
    MARS_SIDEREAL_YEAR_S,
    SECONDS_PER_SOLAR_SOL_S,
)
from reflectors.sun_sync import (
    initial_state_j2000,
    raan_mme2000_from_ltan,
    repeat_ground_track_altitude,
    sun_sync_inclination_rad,
)

import spiceypy as spice


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# Brouwer sun-sync inclination
# ---------------------------------------------------------------------------


class TestSunSyncInclination:
    def test_at_400km_altitude_matches_pinned_9292_deg(self):
        a = mars_equatorial_radius_km() + 400.0
        i_deg = math.degrees(sun_sync_inclination_rad(a))
        # Pinned in tests/test_gravity_zonals.py at 92.92 deg; tolerance
        # 0.05 deg covers the R_ref distinction
        # (3396.0 km gravity model vs 3396.19 km PCK equatorial).
        assert abs(i_deg - 92.92) < 0.05, i_deg

    def test_is_retrograde_across_lmo_altitudes(self):
        R = mars_equatorial_radius_km()
        for alt in (200.0, 400.0, 800.0, 1500.0, 3000.0):
            i_deg = math.degrees(sun_sync_inclination_rad(R + alt))
            assert 90.0 < i_deg < 180.0, (alt, i_deg)

    def test_inclination_monotone_increases_with_altitude(self):
        R = mars_equatorial_radius_km()
        incs = [sun_sync_inclination_rad(R + alt) for alt in (200.0, 400.0, 800.0, 1500.0, 3000.0)]
        assert all(incs[k] < incs[k + 1] for k in range(len(incs) - 1))

    def test_raises_above_altitude_ceiling(self):
        R = mars_equatorial_radius_km()
        # At ~6600+ km altitude the required |cos(i)| > 1 (J_2 drift
        # can't match the Mars-year rate even at i=90 deg).
        with pytest.raises(ValueError, match="no sun-sync solution"):
            sun_sync_inclination_rad(R + 10000.0)

    def test_rejects_non_positive_a(self):
        with pytest.raises(ValueError):
            sun_sync_inclination_rad(-1.0)

    def test_accepts_explicit_overrides(self):
        # When caller passes all three anchors, no SPICE / gravity-model
        # load path is exercised.  Numerical agreement with the default-
        # path result at matching anchors pins the explicit-override path.
        from reflectors.gravity import mars_gravity_model, zonal_coefficients

        model = mars_gravity_model(max_degree=2)
        J2 = zonal_coefficients(model, 2)[2]
        a = model.ref_radius_km + 400.0
        i_default = sun_sync_inclination_rad(a)
        i_explicit = sun_sync_inclination_rad(
            a, mu_km3_s2=model.mu_km3_s2, ref_radius_km=model.ref_radius_km, J2=J2,
        )
        assert i_default == pytest.approx(i_explicit, rel=1e-14)

    def test_target_period_default_is_mars_year_bit_identical(self):
        # target_period_s defaults to the Mars sidereal year.
        a = mars_equatorial_radius_km() + 400.0
        i_no_kwarg = sun_sync_inclination_rad(a)
        i_explicit_mars = sun_sync_inclination_rad(
            a, target_period_s=MARS_SIDEREAL_YEAR_S)
        assert i_no_kwarg == i_explicit_mars  # exact, no tolerance

    def test_earth_sun_sync_inclination_at_800km(self):
        # With Earth anchors + the Earth sidereal year, the helper returns the
        # textbook ~98.6 deg Earth-LEO sun-sync inclination at 800 km.
        from reflectors.earth_constants import EARTH_J2, EARTH_SIDEREAL_YEAR_S
        from reflectors.surface import earth_equatorial_radius_km
        from reflectors.dynamics import body_gm_km3_per_s2

        a = earth_equatorial_radius_km() + 800.0
        i_deg = math.degrees(sun_sync_inclination_rad(
            a, mu_km3_s2=body_gm_km3_per_s2(399),
            ref_radius_km=earth_equatorial_radius_km(),
            J2=EARTH_J2, target_period_s=EARTH_SIDEREAL_YEAR_S))
        assert abs(i_deg - 98.6) < 0.1, i_deg
        assert 90.0 < i_deg < 100.0, i_deg  # retrograde


# ---------------------------------------------------------------------------
# LTAN -> RAAN
# ---------------------------------------------------------------------------


class TestLtanToRaan:
    def test_ltan_6h_and_18h_differ_by_pi(self):
        et = utc_to_et(EPOCH_STR)
        raan_dawn = raan_mme2000_from_ltan(6.0, et)
        raan_dusk = raan_mme2000_from_ltan(18.0, et)
        delta = (raan_dusk - raan_dawn) % (2.0 * math.pi)
        assert delta == pytest.approx(math.pi, abs=1e-12)

    def test_ltan_18h_gives_orbit_plane_perpendicular_to_mars_sun_line(self):
        """LTAN=18h (dawn-dusk) orbits have orbit-normal PARALLEL to the
        Mars->Sun direction (i.e. the orbit plane is perpendicular to
        that line).  Pin the angular-momentum direction against the
        Sun direction from Mars.
        """
        et = utc_to_et(EPOCH_STR)
        a = mars_equatorial_radius_km() + 400.0
        state_j2000 = initial_state_j2000(a_km=a, ltan_h=18.0, M0_rad=0.0, epoch_et=et)
        r, v = state_j2000[:3], state_j2000[3:]
        h = np.cross(r, v)
        h_hat = h / np.linalg.norm(h)

        state_sun, _lt = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")
        s_hat = np.asarray(state_sun[:3], dtype=float)
        s_hat /= np.linalg.norm(s_hat)

        # Sun-sync at Mars is retrograde (~92.92 deg), so h_hat tilts
        # slightly out of the Sun direction.  The IN-PLANE component
        # (projection onto the Mars-equator plane) should be aligned
        # with the Sun's equatorial projection.
        R = mme2000_rotation_from_j2000(et)
        h_mme = R @ h_hat
        s_mme = R @ s_hat
        # Project to Mars equator (drop z-component) and compare.
        h_eq = h_mme[:2] / np.linalg.norm(h_mme[:2])
        s_eq = s_mme[:2] / np.linalg.norm(s_mme[:2])
        # LTAN=18h with this convention puts the orbit ANGULAR MOMENTUM
        # aligned with the Mars->Sun direction (dawn-dusk).  Dot product
        # near +1.
        assert float(np.dot(h_eq, s_eq)) > 0.9999, (h_eq, s_eq)

    def test_ltan_12h_orbit_plane_contains_mars_sun_line(self):
        """LTAN=12h (noon-midnight) orbits have orbit-plane CONTAINING
        the Mars-Sun line; equivalently, the orbit normal is
        PERPENDICULAR to the Mars-Sun direction in the Mars-equator
        plane.
        """
        et = utc_to_et(EPOCH_STR)
        a = mars_equatorial_radius_km() + 400.0
        state_j2000 = initial_state_j2000(a_km=a, ltan_h=12.0, M0_rad=0.0, epoch_et=et)
        r, v = state_j2000[:3], state_j2000[3:]
        h = np.cross(r, v)

        R = mme2000_rotation_from_j2000(et)
        h_mme = R @ h
        state_sun, _lt = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")
        s_mme = R @ np.asarray(state_sun[:3], dtype=float)

        # Orbit normal's equatorial projection perpendicular to Sun's
        # equatorial projection.
        h_eq = h_mme[:2] / np.linalg.norm(h_mme[:2])
        s_eq = s_mme[:2] / np.linalg.norm(s_mme[:2])
        assert abs(float(np.dot(h_eq, s_eq))) < 1e-3

    def test_result_in_zero_to_two_pi(self):
        et = utc_to_et(EPOCH_STR)
        for ltan in (0.0, 6.0, 12.0, 18.0, 23.99):
            raan = raan_mme2000_from_ltan(ltan, et)
            assert 0.0 <= raan < 2.0 * math.pi


# ---------------------------------------------------------------------------
# Round-trip via classical elements
# ---------------------------------------------------------------------------


class TestStateClassicalRoundTrip:
    def test_circular_state_recovers_a_i_raan(self):
        """state_from_classical_mme2000 followed by elements_in_mme2000
        recovers (a, i, raan) to machine precision for a circular orbit.
        argp / nu are degenerate at e=0 and are not checked individually.
        """
        et = utc_to_et(EPOCH_STR)
        from reflectors.gravity import mars_gravity_model

        mu = mars_gravity_model(max_degree=2).mu_km3_s2
        a_in = mars_equatorial_radius_km() + 400.0
        inc_in = math.radians(92.92)
        raan_in = math.radians(45.0)
        state = state_from_classical_mme2000(
            a_km=a_in, e=0.0,
            inclination_rad=inc_in, raan_rad=raan_in,
            argp_rad=0.0, nu_rad=math.radians(30.0),
            mu_km3_s2=mu, epoch_et=et,
        )
        elems = elements_in_mme2000(state, mu, epoch_et=et)
        assert elems.a_km == pytest.approx(a_in, rel=1e-12)
        assert elems.inclination_rad == pytest.approx(inc_in, abs=1e-12)
        assert elems.raan_rad == pytest.approx(raan_in, abs=1e-12)
        assert elems.e < 1e-12

    def test_elliptic_state_recovers_full_six(self):
        et = utc_to_et(EPOCH_STR)
        from reflectors.gravity import mars_gravity_model

        mu = mars_gravity_model(max_degree=2).mu_km3_s2
        a_in = mars_equatorial_radius_km() + 1000.0
        e_in = 0.05
        inc_in = math.radians(80.0)
        raan_in = math.radians(123.4)
        argp_in = math.radians(77.7)
        nu_in = math.radians(200.0)
        state = state_from_classical_mme2000(
            a_km=a_in, e=e_in,
            inclination_rad=inc_in, raan_rad=raan_in,
            argp_rad=argp_in, nu_rad=nu_in,
            mu_km3_s2=mu, epoch_et=et,
        )
        elems = elements_in_mme2000(state, mu, epoch_et=et)
        assert elems.a_km == pytest.approx(a_in, rel=1e-12)
        assert elems.e == pytest.approx(e_in, abs=1e-12)
        assert elems.inclination_rad == pytest.approx(inc_in, abs=1e-12)
        assert elems.raan_rad == pytest.approx(raan_in, abs=1e-12)
        assert elems.argp_rad == pytest.approx(argp_in, abs=1e-12)
        assert elems.nu_rad == pytest.approx(nu_in, abs=1e-12)


# ---------------------------------------------------------------------------
# Composed sanity: sweep grid point produces an expected-looking state
# ---------------------------------------------------------------------------


class TestComposedInitialState:
    def test_state_satisfies_circular_energy(self):
        et = utc_to_et(EPOCH_STR)
        from reflectors.gravity import mars_gravity_model

        mu = mars_gravity_model(max_degree=2).mu_km3_s2
        a = mars_equatorial_radius_km() + 500.0
        state = initial_state_j2000(a_km=a, ltan_h=18.0, M0_rad=1.23, epoch_et=et)
        r = float(np.linalg.norm(state[:3]))
        v = float(np.linalg.norm(state[3:]))
        # Circular energy -mu/(2a); also v = sqrt(mu/a) at e=0.
        assert r == pytest.approx(a, rel=1e-12)
        assert v == pytest.approx(math.sqrt(mu / a), rel=1e-12)

    def test_M0_positions_on_ascending_side_for_0_to_pi(self):
        """For e=0 argp=0, the orbit passes through its ascending node
        at nu=0; nu in (0, pi) is the "northbound" half (for i<90) or
        "equivalent" half (for i>90).  Pin that the Mars-equator
        z-component of r is zero at M0=0 and non-zero for M0 in (0, pi).
        """
        et = utc_to_et(EPOCH_STR)
        a = mars_equatorial_radius_km() + 500.0
        R = mme2000_rotation_from_j2000(et)

        state_an = initial_state_j2000(a_km=a, ltan_h=18.0, M0_rad=0.0, epoch_et=et)
        z_an = float((R @ state_an[:3])[2])
        assert abs(z_an) < 1e-7  # at ascending node, on the equator

        state_north = initial_state_j2000(a_km=a, ltan_h=18.0, M0_rad=math.pi / 2, epoch_et=et)
        z_north = float((R @ state_north[:3])[2])
        assert abs(z_north) > 100.0  # ~500 km above, well off the equator


# ---------------------------------------------------------------------------
# Repeating ground track on a sun-sync orbit
# ---------------------------------------------------------------------------


class TestRepeatGroundTrackAltitude:
    def test_returns_a_i_T_orb_consistent_with_keplers_third_law(self):
        """T_orb at the returned a satisfies Kepler's third law and equals
        SECONDS_PER_SOLAR_SOL_S / K to machine precision (closed form).
        """
        K = 12
        a, i_ss, T_orb = repeat_ground_track_altitude(K)
        # T_orb identity (closed-form synodic relation)
        assert abs(T_orb - SECONDS_PER_SOLAR_SOL_S / K) < 1e-9
        # Kepler's third law a^3 = mu T^2 / (4 pi^2)
        from reflectors.gravity import mars_gravity_model

        mu = mars_gravity_model(max_degree=2).mu_km3_s2
        T_kepler = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
        assert abs(T_kepler - T_orb) < 1e-6
        # Sun-sync feasibility: returned i_ss is in (pi/2, pi).
        assert math.pi / 2.0 < i_ss < math.pi

    def test_K_12_lands_around_509_km_altitude(self):
        """At K=12, the returned a is ~509 km above the Mars equatorial
        radius, matching the K=12 design target.
        """
        from reflectors.surface import mars_equatorial_radius_km

        a, _, _ = repeat_ground_track_altitude(12)
        altitude_km = a - mars_equatorial_radius_km()
        # Tight window, the closed form is deterministic.
        assert 505.0 < altitude_km < 515.0, altitude_km

    def test_synodic_identity_reduces_to_pure_sidereal_components(self):
        """The default solar_sol_s is exactly the synodic of sidereal day
        and sidereal year. Construct an alternative
        ``solar_sol_s = 1 / (1/T_sd - 1/T_year)`` from the live PCK and
        verify the helper agrees within 1 m altitude with the pinned
        SECONDS_PER_SOLAR_SOL_S branch.

        Cross-checks the sun-sync-RAAN-cancels-sidereal-vs-solar
        derivation in the docstring against the live ephemeris.
        """
        import spiceypy as spice
        from reflectors.kernels import load_kernels

        load_kernels()
        _, pm = spice.bodvcd(499, "PM", 3)
        rot_rate_deg_per_day = float(pm[1])
        sidereal_day_s = (360.0 / rot_rate_deg_per_day) * 86400.0
        synodic_s = 1.0 / (1.0 / sidereal_day_s - 1.0 / MARS_SIDEREAL_YEAR_S)
        a_pinned, _, _ = repeat_ground_track_altitude(12)
        a_live, _, _ = repeat_ground_track_altitude(12, solar_sol_s=synodic_s)
        # 1 m tolerance: the pinned constant is rounded; live is exact.
        assert abs(a_pinned - a_live) < 1.0e-3, (
            f"a_pinned={a_pinned:.6f} km vs a_live={a_live:.6f} km "
            f"differ by {(a_pinned - a_live) * 1000:.2f} m"
        )

    def test_K_must_be_positive_integer(self):
        with pytest.raises(ValueError, match=">= 1"):
            repeat_ground_track_altitude(0)
        with pytest.raises(ValueError, match=">= 1"):
            repeat_ground_track_altitude(-3)
        with pytest.raises(ValueError, match="integer"):
            repeat_ground_track_altitude(12.0)  # type: ignore[arg-type]

    def test_K_too_small_raises_no_sun_sync_solution(self):
        """K=1 gives a_orb at ~20000 km altitude, well above Mars's sun-sync
        feasibility ceiling (~6600 km altitude). The helper should propagate
        the underlying ValueError.
        """
        with pytest.raises(ValueError, match="no sun-sync solution"):
            repeat_ground_track_altitude(1)

    def test_K_11_and_K_13_bracket_K_12_altitude(self):
        """Monotonic: more orbits per sol -> smaller a (lower altitude)."""
        from reflectors.surface import mars_equatorial_radius_km

        a_11, _, _ = repeat_ground_track_altitude(11)
        a_12, _, _ = repeat_ground_track_altitude(12)
        a_13, _, _ = repeat_ground_track_altitude(13)
        assert a_11 > a_12 > a_13
        R_eq = mars_equatorial_radius_km()
        # K=13 lands below 200 km altitude; not used at Mars but the
        # helper should still return a valid sun-sync a (Brouwer
        # solution still exists at lower altitudes).
        assert a_13 - R_eq > 0.0
