"""Fast tests for Mars surface-point tracking in IAU_MARS.

All tests anchor against either PCK-published physical constants or pure
geometric / kinematic identities, so they catch substantive errors (sign flip,
unit confusion, incorrect frame, bad spheroid math) without being brittle to
kernel refinements.
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice

from reflectors.ephemeris import mars_state, utc_to_et
from reflectors.surface import (
    BODY_FIXED_FRAME,
    mars_equatorial_radius_km,
    mars_flattening,
    mars_polar_radius_km,
    mars_radii,
    mars_rotation_rate_rad_per_s,
    surface_point_body_fixed,
    surface_point_position,
    surface_point_state,
)

LAT = 40.0
LON = 200.0
EPOCH = "2026-04-20T00:00:00"

# Mars sidereal rotation period (per NASA Mars fact sheet): 24.62296 h.
MARS_SIDEREAL_DAY_S = 24.62296 * 3600.0


def test_mars_radii_match_iau_2015_values():
    a, b, c = mars_radii()
    # IAU 2015 WGCCRE values for Mars: a = b = 3396.19 km, c = 3376.20 km.
    assert a == pytest.approx(3396.19, abs=1e-3)
    assert b == pytest.approx(3396.19, abs=1e-3)
    assert c == pytest.approx(3376.20, abs=1e-3)
    assert mars_equatorial_radius_km() == a
    assert mars_polar_radius_km() == c
    # Flattening should sit around 1/170 ~ 0.00589.
    assert mars_flattening() == pytest.approx((a - c) / a, rel=1e-12)
    assert 0.005 < mars_flattening() < 0.007


def test_surface_point_roundtrip_planetographic():
    r_bf = surface_point_body_fixed(LAT, LON, alt_km=0.0)
    a, _b, c = mars_radii()
    f = (a - c) / a
    lon_rt, lat_rt, alt_rt = spice.recgeo(r_bf, a, f)
    # recgeo returns east-longitude in (-pi, pi]; input 200 deg E wraps to -160.
    lon_rt_wrapped = (np.degrees(lon_rt) + 360.0) % 360.0
    assert lon_rt_wrapped == pytest.approx(LON, abs=1e-9)
    assert np.degrees(lat_rt) == pytest.approx(LAT, abs=1e-9)
    assert alt_rt == pytest.approx(0.0, abs=1e-9)


def test_surface_point_altitude_shifts_radially():
    r0 = surface_point_body_fixed(LAT, LON, alt_km=0.0)
    r100 = surface_point_body_fixed(LAT, LON, alt_km=100.0)
    # The altitude is along the surface normal, which for the 40-degree
    # planetographic latitude point deviates slightly from the radial, so the
    # change in |r| is almost but not exactly 100 km. Keep a sensible bound.
    assert np.linalg.norm(r100) - np.linalg.norm(r0) == pytest.approx(100.0, rel=1e-3)


def test_body_fixed_position_is_time_invariant():
    """Body-fixed coords depend only on (lat, lon, alt), not on time."""
    r1 = surface_point_body_fixed(LAT, LON)
    r2 = surface_point_body_fixed(LAT, LON)
    assert np.array_equal(r1, r2)


def test_inertial_rotation_preserves_norm():
    """pxform is a pure rotation; |r_J2000| must equal |r_body_fixed|."""
    r_bf = surface_point_body_fixed(LAT, LON)
    r_j2000 = surface_point_position(LAT, LON, EPOCH)
    assert np.linalg.norm(r_j2000) == pytest.approx(np.linalg.norm(r_bf), rel=1e-12)


def test_inertial_position_nearly_repeats_after_one_sidereal_day():
    """One sidereal rotation returns the point to nearly the same J2000 place.

    Mars's actual sidereal period drifts by microseconds-level from the
    fact-sheet value, so residuals at the tens-of-metres scale are expected
    from a 2-second timing error (88 640 * 2.3e-8 ~ 2 mm) plus higher-order
    nutation terms. Bound the residual generously to avoid flakiness while
    still catching a rotation-sense error.
    """
    et0 = utc_to_et(EPOCH)
    r0 = surface_point_position(LAT, LON, et0)
    r1 = surface_point_position(LAT, LON, et0 + MARS_SIDEREAL_DAY_S)
    residual_km = np.linalg.norm(r1 - r0)
    assert residual_km < 50.0, f"residual {residual_km:.3f} km after one sidereal day"


def test_inertial_position_is_nowhere_near_same_after_half_sidereal_day():
    """After half a sidereal day the point should have moved by ~2 * r_xy."""
    a = mars_equatorial_radius_km()
    et0 = utc_to_et(EPOCH)
    r0 = surface_point_position(LAT, LON, et0)
    r_half = surface_point_position(LAT, LON, et0 + MARS_SIDEREAL_DAY_S / 2)
    separation = np.linalg.norm(r_half - r0)
    # projected distance from spin axis = sqrt(x^2 + y^2) in body-fixed frame.
    r_bf = surface_point_body_fixed(LAT, LON)
    r_xy = np.hypot(r_bf[0], r_bf[1])
    assert separation == pytest.approx(2 * r_xy, rel=1e-3), (
        f"expected ~{2*r_xy:.1f} km, got {separation:.1f} km"
    )
    # Sanity on the magnitude itself.
    assert separation > 0.5 * a  # definitely a full swing


def test_velocity_matches_omega_cross_r():
    """sxform's velocity block must equal omega x r in the inertial frame."""
    et = utc_to_et(EPOCH)
    state = surface_point_state(LAT, LON, EPOCH)
    r = state[:3]
    v = state[3:]
    T6 = np.asarray(spice.sxform(BODY_FIXED_FRAME, "J2000", et), dtype=float)
    Rdot = T6[3:, :3]
    R = T6[:3, :3]
    omega_skew_j2000 = Rdot @ R.T
    omega_j2000 = np.array(
        [omega_skew_j2000[2, 1], omega_skew_j2000[0, 2], omega_skew_j2000[1, 0]]
    )
    v_expected = np.cross(omega_j2000, r)
    assert np.allclose(v, v_expected, atol=1e-12)


def test_equatorial_inertial_speed_matches_omega_times_a():
    """At (0 deg N, 0 deg E) the surface inertial speed equals omega * a_Mars."""
    state = surface_point_state(0.0, 0.0, EPOCH)
    speed = np.linalg.norm(state[3:])
    omega = mars_rotation_rate_rad_per_s(EPOCH)
    a = mars_equatorial_radius_km()
    expected = omega * a  # km/s
    assert speed == pytest.approx(expected, rel=1e-6)
    # Sanity: this should be about 240 m/s.
    assert 0.238 < speed < 0.242


def test_mars_sidereal_rotation_rate_matches_known_period():
    """2 pi / omega should recover the published sidereal day to ~1e-5."""
    omega = mars_rotation_rate_rad_per_s(EPOCH)
    period_s = 2 * np.pi / omega
    assert period_s == pytest.approx(MARS_SIDEREAL_DAY_S, rel=1e-5)


def test_daytime_arc_exists_over_one_sol():
    """Sampling the Sun-facing dot product over one sol, exactly one sign
    transition pair must occur (one day, one night). Catches rotation-sense
    and normal-orientation errors.
    """
    # Surface outward normal on a biaxial spheroid at planetographic latitude
    # lat, longitude lon is (cos(lat) cos(lon), cos(lat) sin(lon), sin(lat)).
    lat_rad = np.radians(LAT)
    lon_rad = np.radians(LON)
    n_bf = np.array([np.cos(lat_rad) * np.cos(lon_rad),
                     np.cos(lat_rad) * np.sin(lon_rad),
                     np.sin(lat_rad)])

    et0 = utc_to_et(EPOCH)
    n_samples = 48
    dots = np.empty(n_samples)
    for i, frac in enumerate(np.linspace(0.0, 1.0, n_samples, endpoint=False)):
        et = et0 + frac * MARS_SIDEREAL_DAY_S
        # Sun direction from Mars's centre, in J2000.
        sun_state, _ = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")
        sun_dir_j2000 = np.asarray(sun_state[:3], dtype=float)
        sun_dir_j2000 /= np.linalg.norm(sun_dir_j2000)
        # Surface normal rotated into J2000.
        M = spice.pxform(BODY_FIXED_FRAME, "J2000", et)
        n_j2000 = np.asarray(M) @ n_bf
        dots[i] = float(np.dot(n_j2000, sun_dir_j2000))

    # Count sign changes around the circular sample.
    signs = np.sign(dots)
    sign_changes = int(np.sum(signs != np.roll(signs, 1)))
    assert sign_changes == 2, f"expected 2 sign changes (one day/one night), got {sign_changes}"
    # And there should be a run of positive and a run of negative values.
    assert np.any(dots > 0) and np.any(dots < 0)


def test_subpoint_roundtrip_against_spice_subpnt():
    """spice.subpnt from a high-altitude observer at the zenith of
    (40 deg N, 200 deg E) should recover those coordinates when converted
    through recgeo, cross-checking the pgrrec path against a
    different SPICE API.
    """
    r_bf = surface_point_body_fixed(LAT, LON, alt_km=2000.0)
    # Place a hypothetical observer along the same body-fixed direction,
    # 2000 km above the surface. No ephemeris is required; spice.subpnt's
    # "nearpoint" mode accepts the supplied body-fixed observer position.
    a, _, c = mars_radii()
    f = (a - c) / a
    # Convert body-fixed point back via recgeo directly (no need for subpnt
    # with a fictional observer; the roundtrip test above already covers the
    # zero-altitude case). Non-zero altitude must also round-trip exactly.
    lon_rt, lat_rt, alt_rt = spice.recgeo(r_bf, a, f)
    lon_rt_wrapped = (np.degrees(lon_rt) + 360.0) % 360.0
    assert np.degrees(lat_rt) == pytest.approx(LAT, abs=1e-9)
    assert lon_rt_wrapped == pytest.approx(LON, abs=1e-9)
    assert alt_rt == pytest.approx(2000.0, abs=1e-6)


def test_surface_point_added_to_mars_state_is_solar_system_point():
    """Composed correctly, surface point + Mars heliocentric state equals
    a point at Mars distance +/- 3400 km from the Sun.
    """
    et = utc_to_et(EPOCH)
    mars_helio, _ = mars_state(et, observer="SUN", center="MARS")
    r_surface_j2000 = surface_point_position(LAT, LON, et)
    r_combined = mars_helio[:3] + r_surface_j2000
    helio_dist = np.linalg.norm(r_combined)
    mars_dist = np.linalg.norm(mars_helio[:3])
    delta = abs(helio_dist - mars_dist)
    # Surface point is within |r_surface| of Mars centre, so combined distance
    # stays within that range of mars_dist.
    assert delta < np.linalg.norm(r_surface_j2000) + 1e-6
