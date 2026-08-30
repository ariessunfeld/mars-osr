"""Fast sanity / physical-plausibility tests for the ephemeris module.

These are deliberately anchored to physics rather than pre-fetched Horizons
numbers, so they are robust to kernel updates and small numerical drift while
still catching any gross errors (incorrect frame, flipped observer, bad UTC
conversion, missing leapseconds, etc).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import spiceypy as spice

import reflectors.ephemeris as ephemeris
from reflectors.ephemeris import (
    AU_KM,
    EphemerisCacheStats,
    body_state,
    corotating_heliocentric_triad,
    decompose_into_triad,
    ephemeris_evaluation_context,
    frame_rotation,
    mars_state,
    spice_state_at_et,
    sun_mars_distance_au,
    sun_mars_distance_km,
    sun_state,
    utc_to_et,
)

# Heliocentric gravitational parameter (GM_sun) in km^3/s^2 (DE440 value).
MU_SUN = 1.32712440041279419e11

# Mars orbital envelope. Perihelion 1.3814 AU, aphelion 1.6660 AU (NASA fact
# sheet). Widen by ~1% to be safe against ephemeris drift at non-epoch times.
MARS_PERIHELION_AU = 1.38
MARS_APHELION_AU = 1.67


def test_utc_to_et_at_j2000_equals_tt_offset():
    """At the J2000 UTC epoch, ET ~ TT offset from UTC = 64.184 s.

    Specifically: ET = TDB, TT - TAI = 32.184 s (constant), TAI - UTC = 32 s on
    2000-01-01 (valid range), so ET - UTC ~= 64.184 s. TDB - TT is periodic and
    stays sub-ms at this date.
    """
    et = utc_to_et("2000-01-01T12:00:00")
    assert et == pytest.approx(64.184, abs=5e-3)


def test_utc_to_et_accepts_datetime_naive_and_aware():
    naive = datetime(2026, 4, 20, 0, 0, 0)
    aware = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
    from_string = utc_to_et("2026-04-20T00:00:00")
    assert utc_to_et(naive) == pytest.approx(from_string, abs=1e-9)
    assert utc_to_et(aware) == pytest.approx(from_string, abs=1e-9)


def test_utc_to_et_passthrough_float():
    assert utc_to_et(12345.6789) == 12345.6789


def test_sun_mars_distance_bounds_over_a_year():
    """Over a dense monthly grid the Sun-Mars distance must stay in bounds."""
    epochs = [f"2026-{m:02d}-15T00:00:00" for m in range(1, 13)]
    distances_au = np.array([sun_mars_distance_au(e) for e in epochs])
    assert np.all(distances_au > MARS_PERIHELION_AU), distances_au
    assert np.all(distances_au < MARS_APHELION_AU), distances_au


def test_sun_mars_distance_matches_body_state():
    """The scalar helper agrees with the vector helper."""
    epoch = "2026-04-20T00:00:00"
    d_km = sun_mars_distance_km(epoch)
    state, _ = mars_state(epoch, observer="SUN")
    assert d_km == pytest.approx(np.linalg.norm(state[:3]), rel=1e-12)


def test_sun_ssb_wobble_bounded():
    """Sun relative to SSB stays within ~1.5 solar radii (< 2e6 km)."""
    sun, _ = sun_state("2026-04-20T00:00:00")
    r = np.linalg.norm(sun[:3])
    assert r < 2.0e6, f"|Sun - SSB| = {r:.0f} km, too large"


def test_earth_sun_distance_is_near_one_au():
    """A basic observer/target sanity check: Earth-Sun ~ 1 AU year-round."""
    epochs = [f"2026-{m:02d}-15T00:00:00" for m in range(1, 13)]
    dists_au = np.array(
        [
            np.linalg.norm(body_state("EARTH", e, observer="SUN")[0][:3]) / AU_KM
            for e in epochs
        ]
    )
    assert np.all(dists_au > 0.98) and np.all(dists_au < 1.02), dists_au


def test_mars_osculating_semimajor_axis_near_known_value():
    """Compute Mars' osculating semi-major axis at a reference epoch from the
    vis-viva equation and check it lies near the textbook 1.524 AU.

    vis-viva: 1/a = 2/r - v^2 / mu_sun
    """
    state, _ = mars_state("2026-04-20T00:00:00", observer="SUN")
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    inv_a = 2.0 / r - v * v / MU_SUN
    a_km = 1.0 / inv_a
    a_au = a_km / AU_KM
    # Mars semi-major axis is 1.523679 AU; osculating value stays within ~0.5%.
    assert a_au == pytest.approx(1.52368, rel=5e-3), f"a = {a_au:.6f} AU"


def test_mars_j2000_frame_is_close_to_ecliptic():
    """Mars orbital plane tilts only ~1.85 deg off the ecliptic. In the J2000
    equatorial frame the plane is tilted by (obliquity + inclination) = ~23.4
    + 1.85 deg. This test checks the orbit normal in J2000 matches that.
    """
    state, _ = mars_state("2026-04-20T00:00:00", observer="SUN")
    r = state[:3]
    v = state[3:]
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    z_hat = np.array([0.0, 0.0, 1.0])
    angle_deg = np.degrees(np.arccos(np.dot(h_hat, z_hat)))
    # Ecliptic inclined ~23.44 deg to J2000 equator; Mars orbit ~1.85 deg to
    # ecliptic. Expect between ~21.5 and ~25.3 deg.
    assert 20.0 < angle_deg < 26.0, f"Mars orbit normal to J2000 z = {angle_deg:.3f} deg"


def test_mars_planet_matches_jpl_horizons_anchor():
    """Cross-check against JPL Horizons for Mars planet center (NAIF 499)
    heliocentric state in J2000 at 2026-04-20T00:00:00 TDB.

    Values fetched 2026-04-20 via
        https://ssd.jpl.nasa.gov/api/horizons.api
        ?COMMAND=499 &CENTER=500@10 &REF_PLANE=FRAME &REF_SYSTEM=J2000
        &TIME_TYPE=TDB &OUT_UNITS=KM-S &VEC_TABLE=2
    Horizons sources reported: Mars (499) -> mar099; Sun (10) -> DE441.
    Test kernels: de440.bsp + mar099.bsp. Expected agreement is at the
    meter / mm-per-second level (DE440 vs DE441 drift in the 2020s).
    """
    horizons_pos_km = np.array(
        [2.050765183203375e08, -2.498251400478343e07, -1.699006198239387e07]
    )
    horizons_vel_kms = np.array(
        [4.399908298196852e00, 2.372715500460080e01, 1.076442996051492e01]
    )
    et = spice.str2et("2026-04-20 00:00:00 TDB")
    state, _ = spice.spkezr("499", et, "J2000", "NONE", "SUN")
    pos = np.asarray(state[:3])
    vel = np.asarray(state[3:])

    # 1 m tolerance on each position component; 1 mm/s on each velocity
    # component. This is well below relevant solar-sail dynamics scales but
    # tight enough to catch frame/observer errors or kernel drift.
    assert np.allclose(pos, horizons_pos_km, atol=1e-3), pos - horizons_pos_km
    assert np.allclose(vel, horizons_vel_kms, atol=1e-6), vel - horizons_vel_kms


def test_spkezr_light_time_roughly_matches_distance_over_c():
    """Light-time returned by SPICE equals range / c (to the km level)."""
    state, lt = body_state("MARS BARYCENTER", "2026-04-20T00:00:00", observer="EARTH")
    r = np.linalg.norm(state[:3])
    expected_lt = r / spice.clight()
    assert lt == pytest.approx(expected_lt, rel=1e-12)


def test_exact_ephemeris_context_shares_state_and_returns_independent_copies(
    monkeypatch,
):
    """Only an identical five-part SPICE key is shared inside one context."""
    et = spice.str2et("2026-06-01T00:00:00")
    original_spkezr = spice.spkezr
    reference_state, reference_lt = original_spkezr(
        "10", et, "J2000", "NONE", "499"
    )
    calls = []

    def counted_spkezr(*args):
        calls.append(args)
        return original_spkezr(*args)

    monkeypatch.setattr(ephemeris.spice, "spkezr", counted_spkezr)
    stats = EphemerisCacheStats()
    with ephemeris_evaluation_context(stats):
        first, first_lt = spice_state_at_et(10, et, observer=499)
        first[0] = np.nan
        second, second_lt = spice_state_at_et(10, et, observer=499)
        different_et, _ = spice_state_at_et(10, et + 1.0, observer=499)

    assert len(calls) == 2
    assert np.array_equal(second, np.asarray(reference_state))
    assert first_lt == reference_lt
    assert second_lt == reference_lt
    assert np.all(np.isfinite(different_et))
    assert stats.as_dict() == {
        "enabled": True,
        "contexts": 1,
        "state_requests": 3,
        "state_cache_hits": 1,
        "state_spice_calls": 2,
        "rotation_requests": 0,
        "rotation_cache_hits": 0,
        "rotation_inverse_hits": 0,
        "rotation_spice_calls": 0,
    }

    # Context exit discards values: the same request reaches SPICE again.
    outside, _ = spice_state_at_et(10, et, observer=499)
    assert len(calls) == 3
    assert np.array_equal(outside, np.asarray(reference_state))


def test_exact_ephemeris_context_nests_and_reuses_inverse_rotations(
    monkeypatch,
):
    """Nested helpers reuse a cache; reverse pxform is the exact transpose."""
    et = spice.str2et("2026-06-01T00:00:00")
    original_pxform = spice.pxform
    direct_reference = np.asarray(
        original_pxform("J2000", "IAU_MARS", et), dtype=float
    )
    reverse_reference = np.asarray(
        original_pxform("IAU_MARS", "J2000", et), dtype=float
    )
    calls = []

    def counted_pxform(*args):
        calls.append(args)
        return original_pxform(*args)

    monkeypatch.setattr(ephemeris.spice, "pxform", counted_pxform)
    stats = EphemerisCacheStats()
    with ephemeris_evaluation_context(stats):
        direct = frame_rotation("J2000", "IAU_MARS", et)
        with ephemeris_evaluation_context():
            direct_again = frame_rotation("J2000", "IAU_MARS", et)
        reverse = frame_rotation("IAU_MARS", "J2000", et)

    assert len(calls) == 1
    assert np.array_equal(direct, direct_reference)
    assert np.array_equal(direct_again, direct_reference)
    assert np.array_equal(reverse, reverse_reference)
    assert stats.contexts == 1
    assert stats.rotation_requests == 3
    assert stats.rotation_cache_hits == 2
    assert stats.rotation_inverse_hits == 1
    assert stats.rotation_spice_calls == 1


# ---------------------------------------------------------------------------
# Co-rotating heliocentric triad (escape-phasing hand-off frame)
# ---------------------------------------------------------------------------


class TestCorotatingHeliocentricTriad:
    """The orthonormal {p_hat (prograde), e_hat (in-plane), n_hat (orbit normal)}
    frame used to express escape/capture hand-off states (v_inf, Hill-exit
    position) as prograde / in-plane / out-of-plane components."""

    def test_triad_is_orthonormal_and_right_handed(self):
        # Mars heliocentric state (a real, near-circular eccentric orbit).
        state, _ = body_state("MARS BARYCENTER", "2028-01-01T00:00:00",
                              observer="SUN")
        p_hat, e_hat, n_hat = corotating_heliocentric_triad(state)
        for u in (p_hat, e_hat, n_hat):
            assert float(np.linalg.norm(u)) == pytest.approx(1.0, abs=1e-12)
        assert float(np.dot(p_hat, e_hat)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.dot(p_hat, n_hat)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.dot(e_hat, n_hat)) == pytest.approx(0.0, abs=1e-12)
        # Right-handed: p x e == n.
        np.testing.assert_allclose(np.cross(p_hat, e_hat), n_hat, atol=1e-12)
        # p_hat is the prograde direction.
        v = state[3:6]
        np.testing.assert_allclose(p_hat, v / np.linalg.norm(v), atol=1e-12)

    def test_triad_stays_orthonormal_for_an_eccentric_state(self):
        """The decisive correctness check: for a state where v is NOT
        perpendicular to r (true off the apsides), the naive {v_hat, r_hat, oop}
        frame is non-orthonormal, but {p_hat, e_hat, n_hat} is orthonormal by
        construction."""
        state = np.array([1.5e8, 0.0, 0.0, 5.0, 25.0, 0.0])  # v has radial part
        p_hat, e_hat, n_hat = corotating_heliocentric_triad(state)
        r_hat = state[:3] / np.linalg.norm(state[:3])
        # r and v are NOT orthogonal here (so {v,r,oop} would be skew)...
        assert abs(float(np.dot(r_hat, p_hat))) > 0.1
        # ...yet the returned triad is exactly orthonormal.
        assert float(np.dot(p_hat, e_hat)) == pytest.approx(0.0, abs=1e-14)
        assert float(np.dot(p_hat, n_hat)) == pytest.approx(0.0, abs=1e-14)
        assert float(np.linalg.norm(e_hat)) == pytest.approx(1.0, abs=1e-14)

    def test_decompose_reconstructs_vector_exactly(self):
        state = np.array([1.5e8, 0.0, 0.0, 5.0, 25.0, 0.0])
        p_hat, e_hat, n_hat = corotating_heliocentric_triad(state)
        vec = np.array([-3.1, 4.2, 9.7])
        pro, inp, oop = decompose_into_triad(vec, state)
        recon = pro * p_hat + inp * e_hat + oop * n_hat
        np.testing.assert_allclose(recon, vec, atol=1e-10)

    def test_decompose_of_planet_velocity_is_pure_prograde(self):
        state, _ = body_state("MARS BARYCENTER", "2028-01-01T00:00:00",
                              observer="SUN")
        v = state[3:6]
        pro, inp, oop = decompose_into_triad(v, state)
        assert pro == pytest.approx(float(np.linalg.norm(v)), rel=1e-12)
        assert inp == pytest.approx(0.0, abs=1e-6)
        assert oop == pytest.approx(0.0, abs=1e-6)

    def test_invalid_shapes_and_degenerate_states_raise(self):
        good = np.array([1.5e8, 0.0, 0.0, 5.0, 25.0, 0.0])
        with pytest.raises(ValueError):
            corotating_heliocentric_triad(np.zeros(3))          # not (6,)
        with pytest.raises(ValueError):
            corotating_heliocentric_triad(np.array(
                [1.5e8, 0.0, 0.0, 0.0, 0.0, 0.0]))              # zero velocity
        with pytest.raises(ValueError):
            corotating_heliocentric_triad(np.array(
                [1.5e8, 0.0, 0.0, 5.0, 0.0, 0.0]))              # r x v == 0
        with pytest.raises(ValueError):
            decompose_into_triad(np.zeros(2), good)             # vec not (3,)
