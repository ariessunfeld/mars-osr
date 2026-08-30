"""Infrastructure for the interplanetary Sun-centred cruise.

The Earth-Hill -> Mars-Hill SRP-sail cruise reuses
``escape.propagate_escape`` with the Sun as the
central body. This module pins the required interfaces:

  - ``central_body.sun_central_body()`` field correctness (NAIF 10, live solar
    GM, solar radius, ``occults_sun=False``, raising gravity stub).
  - ``third_body.earth_third_body()`` / ``mars_third_body()`` specs.
  - The ``occults_sun`` field defaults ``True`` (Mars/Earth byte-identical) and
    gates the umbra: with the Sun as central body the shadow geometry is
    degenerate (a spurious permanent eclipse), so the cruise needs the bypass.
  - A Sun-centred SRP-acts positive signal: with the umbra bypassed the sail's
    SRP genuinely perturbs a heliocentric two-body arc (it would not if the
    degenerate shadow zeroed the flux).
"""

from __future__ import annotations

import numpy as np
import pytest

from reflectors.attitude_control import AttitudeLimits
from reflectors.central_body import (
    SUN_NOMINAL_CEILING_KM,
    earth_central_body,
    mars_central_body,
    sun_central_body,
)
from reflectors.dynamics import body_gm_km3_per_s2, sun_gm_km3_per_s2
from reflectors.escape import propagate_escape
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail
from reflectors.shadow import shadow_factor, sun_radius_km
from reflectors.third_body import earth_third_body, mars_third_body


# ---------------------------------------------------------------------------
# sun_central_body factory
# ---------------------------------------------------------------------------


def test_sun_central_body_fields():
    """Sun central body reads live solar GM/radius and bypasses the umbra."""
    sun = sun_central_body()
    assert sun.naif_id == 10
    assert sun.mu_km3_s2 == sun_gm_km3_per_s2()
    assert sun.mu_km3_s2 == body_gm_km3_per_s2(10)
    assert sun.equatorial_radius_km == sun_radius_km()
    assert sun.hill_radius_km == SUN_NOMINAL_CEILING_KM
    # The nominal ceiling must exceed every heliocentric distance the cruise
    # visits (Mars aphelion ~2.49e8 km) by a wide margin.
    assert sun.hill_radius_km > 1.0e9
    assert sun.occults_sun is False
    assert sun.label == "SUN"


def test_sun_gravity_model_factory_raises():
    """The cruise runs gravity_degree=0; building a solar gravity model is a
    modelling error and must raise explicitly (no solar zonal field is wired)."""
    sun = sun_central_body()
    with pytest.raises(NotImplementedError):
        sun.gravity_model_factory(max_degree=2)


# ---------------------------------------------------------------------------
# third-body factories
# ---------------------------------------------------------------------------


def test_earth_third_body_spec():
    eb = earth_third_body()
    assert eb.naif_id == 399
    assert eb.mu_km3_s2 == body_gm_km3_per_s2(399)
    assert eb.label == "EARTH"


def test_mars_third_body_spec():
    mb = mars_third_body()
    assert mb.naif_id == 499
    assert mb.mu_km3_s2 == body_gm_km3_per_s2(499)
    assert mb.label == "MARS"


# ---------------------------------------------------------------------------
# occults_sun default + the degenerate-shadow finding it works around
# ---------------------------------------------------------------------------


def test_occults_sun_defaults_true_for_orbited_planets():
    """Mars and Earth default to casting an umbra on the sail."""
    assert mars_central_body().occults_sun is True
    assert earth_central_body().occults_sun is True


def test_no_self_eclipse_with_sun_as_central_body():
    """A body must not eclipse itself when it is the light source.

    The geometry is degenerate: the occulting disc (R_sun)
    coincides with the illuminating disc, so D=0 and sigma_occulter==sigma_Sun,
    making the total-eclipse condition ``D + sigma_S <= sigma_M`` hold with
    spurious equality. ``shadow_factor`` therefore returns 1.0 (sunlit), and the
    propagator also skips the test for the Sun via ``occults_sun=False``."""
    r_sat = np.array([sun_radius_km() * 215.0, 0.0, 0.0])  # ~1 AU on the x-axis
    # Both with a pre-fetched Sun-at-origin position and with a SPICE fetch:
    factor_prefetched = shadow_factor(
        r_sat, 0.0, 10,
        sun_position_j2000_km=np.zeros(3),
        central_radius_km=sun_radius_km(),
    )
    factor_fetched = shadow_factor(r_sat, 0.0, 10, central_radius_km=sun_radius_km())
    assert factor_prefetched == 1.0  # sunlit -- no self-eclipse
    assert factor_fetched == 1.0


def test_shadow_guard_leaves_planet_callers_unchanged():
    """The Sun-self-eclipse guard keys on observer==Sun, so Mars/Earth umbra
    tests (observers 499/399) are untouched: a sail on the exact anti-Sun axis
    deep behind the planet is still reported in umbra."""
    import spiceypy as spice
    from reflectors.surface import mars_equatorial_radius_km

    et = 0.0
    sun_wrt_mars = np.asarray(
        spice.spkezr("10", et, "J2000", "NONE", "499")[0][:3], dtype=float
    )
    # A point just behind Mars on the anti-Sun axis, well inside the umbra cone.
    anti = -sun_wrt_mars / np.linalg.norm(sun_wrt_mars)
    r_sat = anti * (mars_equatorial_radius_km() + 500.0)
    assert shadow_factor(r_sat, et, 499) == 0.0  # still eclipsed by Mars


# ---------------------------------------------------------------------------
# Sun-centred SRP positive signal (the bypass actually delivers flux)
# ---------------------------------------------------------------------------


def _heliocentric_circular_state(radius_km):
    """A Sun-centred circular orbit state in the x-y plane."""
    mu = sun_gm_km3_per_s2()
    v_circ = np.sqrt(mu / radius_km)
    return np.array([radius_km, 0.0, 0.0, 0.0, v_circ, 0.0])


def test_sun_central_srp_perturbs_heliocentric_arc():
    """Positive signal: with the umbra bypassed, a sunward-pointing sail's SRP
    perturbs a heliocentric two-body arc away from the feathered (coast) arc.

    If the degenerate shadow zeroed the solar flux, both the
    'thrusting' and the feathered run would reduce to identical pure two-body
    motion and the displacement would vanish -- so a non-zero displacement
    proves the flux is delivered."""
    au_km = 1.495978707e8
    state0 = _heliocentric_circular_state(au_km)
    epoch_et = 0.0
    sail = make_canonical_sail(0.018)  # sigma = 18 g/m^2
    params = QLawParams(a_target_km=SUN_NOMINAL_CEILING_KM, rp_min_km=1.0)
    limits = AttitudeLimits()
    span_s = (0.0, 43200.0)  # half a day -- a tiny heliocentric arc
    sun = sun_central_body()

    # Sunward sail normal (n = s_hat) -> maximal SRP; vs feathered coast.
    def sunward(r, v, s_hat, p_eff, sail_, current_n):
        return s_hat

    thrust = propagate_escape(
        state0, epoch_et, sail, params, limits, span_s,
        gravity_degree=0, central_body=sun, third_bodies=(),
        steering_fn=sunward,
    )
    coast = propagate_escape(
        state0, epoch_et, sail, params, limits, span_s,
        gravity_degree=0, central_body=sun, third_bodies=(),
        force_coast=True,
    )
    dr = np.linalg.norm(
        thrust.orbit_state_km_kmps[-1, :3] - coast.orbit_state_km_kmps[-1, :3]
    )
    # SRP characteristic accel ~5e-7 km/s^2 over 4.3e4 s -> ~hundreds of km.
    assert dr > 100.0
    # Sanity: both runs ran the full span and stayed finite.
    assert np.all(np.isfinite(thrust.orbit_state_km_kmps))
    assert thrust.metadata["central_body"]["occults_sun"] is False


def test_sun_central_two_body_conserves_energy_and_ang_mom():
    """Sun-central point-mass gravity wiring: a feathered (SRP=0) heliocentric
    coast conserves specific energy and angular momentum to RK4 truncation.

    Uses kinematic_attitude + force_coast so the sail normal is held exactly
    edge-on (n.s=0) every step -> SRP is identically zero -> the run is pure
    two-body about the Sun, isolating the gravity + integration wiring."""
    mu = sun_gm_km3_per_s2()
    radius = 1.495978707e8  # 1 AU
    state0 = np.array([radius, 0.0, 0.0, 0.0, np.sqrt(mu / radius), 0.0])
    sail = make_canonical_sail(0.018)
    params = QLawParams(a_target_km=SUN_NOMINAL_CEILING_KM, rp_min_km=1.0)
    limits = AttitudeLimits()
    span = (0.0, 30.0 * 86400.0)  # 30 days, a short heliocentric arc
    run = propagate_escape(
        state0, 0.0, sail, params, limits, span,
        gravity_degree=0, central_body=sun_central_body(), third_bodies=(),
        force_coast=True, kinematic_attitude=True,
    )
    s0 = run.orbit_state_km_kmps[0]
    s1 = run.orbit_state_km_kmps[-1]

    def energy(s):
        return 0.5 * float(np.dot(s[3:6], s[3:6])) - mu / float(np.linalg.norm(s[:3]))

    eps0, eps1 = energy(s0), energy(s1)
    h0 = np.cross(s0[:3], s0[3:6])
    h1 = np.cross(s1[:3], s1[3:6])
    assert abs(eps1 - eps0) / abs(eps0) < 1e-7
    assert np.linalg.norm(h1 - h0) / np.linalg.norm(h0) < 1e-7
