"""Fast tests for the Q-law escape steering (``reflectors.qlaw``).

The escape law is a rate-normalised radii-based blend: raise apoapsis
``r_a = a(1+e)`` toward Hill, barrier-defend periapsis ``r_p = a(1-e)`` from
the atmosphere, with an optional explicit eccentricity term. See the
``reflectors.qlaw`` module docstring.

Independent cross-checks:

  1. The 1-D sail-pitch search inside ``steer`` finds the same ``dQ/dt``
     optimum as a brute-force search over a dense hemisphere of sail normals.
  2. With ``w_e = 0`` and the barrier off (healthy
     periapsis) the chosen orientation maximises ``rdot_a`` (apoapsis-rate),
     and the chosen ``rdot_a`` is strictly bigger than the maximum ``adot``
     achievable -- proving the law uses the ``a*edot`` contribution that an
     adot-only steering would miss.
  3. Turning the eccentricity term on (``w_e = 1`` vs
     ``w_e = 0``) shifts the steering by an O(1)-of-rate-maximum amount: the
     confirming that the eccentricity term actively changes the command.
  4. The periapsis barrier engages (penalty ``> 1``, protective steering)
     when periapsis drops below ``r_p_min``, and ``G_e`` flips sign as
     ``rho_p`` ramps (from "pump e for apoapsis" to "circularise for safety").
  5. The coast/feather branch fires when no achievable orientation raises the
     escape reward (Sun ahead of the motion; eclipse; escaped state).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.dynamics import mars_gm_km3_per_s2
from reflectors.gauss import (
    eccentricity_rate_max,
    gauss_variational_rates,
    osculating_elements,
    rtn_basis,
)
from reflectors.qlaw import (
    QLawParams,
    QLawSteering,
    escape_reward_gradients,
    evaluate_orbit_effectivity,
    periapsis_penalty,
    steer,
)
from reflectors.sail_designs import make_canonical_sail
from reflectors.srp import SailOptical, SolarSail, mcinnes_srp_acceleration

R_EQ = 3396.19
P_SRP = 2.0e-6  # representative Pa; steering direction is P-independent


def _mu() -> float:
    """Mars GM -- fetched lazily (kernels load via the autouse conftest
    fixture, after module collection)."""
    return mars_gm_km3_per_s2()


def _params(**kw) -> QLawParams:
    base = dict(a_target_km=1.0e6, rp_min_km=R_EQ + 300.0)
    base.update(kw)
    return QLawParams(**base)


def _state() -> tuple[np.ndarray, np.ndarray]:
    """A generic bound, eccentric, inclined state (km, km/s)."""
    r = np.array([4200.0, 1000.0, 1500.0])
    v = np.array([-0.35, 2.85, 0.65])
    return r, v


def _state_at_nu(
    a_km: float, e: float, nu_deg: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian state of an equatorial orbit (a, e) at true anomaly nu.

    Built in the perifocal frame. nu away from 0 / 180 deg keeps f_r in play,
    so semi-major axis and eccentricity are genuinely steerable against each
    other (at periapsis de/dt depends only on f_theta -- a degenerate test
    geometry where the a- and e-weighted laws coincide)."""
    nu = math.radians(nu_deg)
    p = a_km * (1.0 - e * e)
    h = math.sqrt(mu * p)
    r = p / (1.0 + e * math.cos(nu))
    r_pf = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pf = (mu / h) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    return r_pf, v_pf


def _hemisphere(s_hat: np.ndarray, n_pts: int) -> list[np.ndarray]:
    """Fibonacci-sphere unit vectors restricted to the lit hemisphere."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for k in range(n_pts):
        z = 1.0 - 2.0 * (k + 0.5) / n_pts
        rad = math.sqrt(max(0.0, 1.0 - z * z))
        phi = ga * k
        n = np.array([rad * math.cos(phi), rad * math.sin(phi), z])
        if float(np.dot(n, s_hat)) > 1.0e-6:
            out.append(n)
    return out


def _rtn_rates(n, elements, r, v, s_hat, sail):
    """Gauss element rates produced by sail normal ``n``."""
    a_srp = mcinnes_srp_acceleration(n, s_hat, P_SRP, sail)
    r_hat, theta_hat, h_hat = rtn_basis(r, v)
    return gauss_variational_rates(
        elements,
        float(np.dot(a_srp, r_hat)),
        float(np.dot(a_srp, theta_hat)),
        float(np.dot(a_srp, h_hat)),
    )


def _dqdt_for_normal(n, elements, r, v, s_hat, sail, params):
    """dQ/dt produced by sail normal ``n`` -- the same quantity ``steer``
    minimises, recomputed from the public pieces for the cross-check."""
    rates = _rtn_rates(n, elements, r, v, s_hat, sail)
    f_char = float(np.linalg.norm(
        mcinnes_srp_acceleration(s_hat, s_hat, P_SRP, sail)
    ))
    g_a, g_e = escape_reward_gradients(elements, f_char, params)
    return g_a * rates.da_dt_km_s + g_e * rates.de_dt_per_s


# ---------------------------------------------------------------------------
# Periapsis penalty + parameter validation
# ---------------------------------------------------------------------------


def test_periapsis_penalty_unity_at_rp_min_and_monotone():
    params = _params()
    assert periapsis_penalty(params.rp_min_km, params) == pytest.approx(1.0)
    # Decreasing in r_p above the floor, growing below it.
    assert periapsis_penalty(2.0 * params.rp_min_km, params) < 1.0
    assert periapsis_penalty(0.5 * params.rp_min_km, params) > 1.0


def test_qlaw_params_validation():
    with pytest.raises(ValueError, match="a_target_km"):
        QLawParams(a_target_km=-1.0, rp_min_km=3696.0)
    with pytest.raises(ValueError, match="rp_min_km"):
        QLawParams(a_target_km=1e6, rp_min_km=0.0)
    with pytest.raises(ValueError, match="e_ref"):
        QLawParams(a_target_km=1e6, rp_min_km=3696.0, e_ref=0.0)
    with pytest.raises(ValueError, match="w_apo, w_a, w_e"):
        QLawParams(a_target_km=1e6, rp_min_km=3696.0, w_e=-1.0)


# ---------------------------------------------------------------------------
# Cross-check 1 -- 1-D search vs brute-force hemisphere search
# ---------------------------------------------------------------------------


def test_steer_optimum_matches_brute_force_hemisphere():
    """The 1-D sail-pitch search finds the same dQ/dt minimum as a dense
    brute-force search over all lit sail normals."""
    params = _params()
    sail = make_canonical_sail(0.018)
    r, v = _state()
    el = osculating_elements(r, v, _mu())
    r_hat, theta_hat, h_hat = rtn_basis(r, v)
    # Sun behind the motion (a thrust-favourable geometry).
    s_hat = -theta_hat
    current_n = np.array(s_hat)  # arbitrary starting attitude

    result = steer(el, r, v, s_hat, P_SRP, sail=sail,
                   current_n_hat=current_n, params=params)
    assert result.thrust is True

    brute_min = min(
        _dqdt_for_normal(n, el, r, v, s_hat, sail, params)
        for n in _hemisphere(s_hat, 6000)
    )
    # steer's golden-section refinement is at least as good as the grid.
    assert result.dQ_dt <= brute_min + 1e-12
    # ... and not wildly better -- same optimum, within the grid resolution.
    assert result.dQ_dt >= brute_min * 1.01
    # n_star reproduces the reported dQ/dt.
    dqdt_at_nstar = _dqdt_for_normal(
        result.n_star_j2000, el, r, v, s_hat, sail, params
    )
    assert dqdt_at_nstar == pytest.approx(result.dQ_dt, rel=1e-6, abs=1e-18)
    assert float(np.linalg.norm(result.n_star_j2000)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cross-check 2 -- w_e = 0, w_p = 0 maximises apoapsis rate (rdot_a)
# ---------------------------------------------------------------------------


def test_w_e_zero_w_p_zero_maximises_rdot_a():
    """With the eccentricity and periapsis terms off the law steers to
    maximise ``rdot_a = (1+e) adot + a edot`` -- the apoapsis-rate the
    escape-termination radius actually depends on -- NOT ``da/dt`` alone.
    Positive signal that the apoapsis-pumping framing is wired through; an
    ``adot``-only steering would give a strictly smaller ``rdot_a`` at the
    chosen orientation."""
    params = _params(w_apo=1.0, w_e=0.0, w_p=0.0)
    sail = make_canonical_sail(0.018)
    # nu = 60 deg so f_r and f_theta both matter; periapsis or apoapsis would
    # be degenerate (the apoapsis-rate and a-rate optima coincide there).
    r, v = _state_at_nu(4200.0, 0.08, 60.0, _mu())
    el = osculating_elements(r, v, _mu())
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat

    result = steer(el, r, v, s_hat, P_SRP, sail=sail,
                   current_n_hat=np.array(s_hat), params=params)
    assert result.thrust is True

    def rdot_a(n):
        rates = _rtn_rates(n, el, r, v, s_hat, sail)
        return (1.0 + el.e) * rates.da_dt_km_s + el.a_km * rates.de_dt_per_s

    rdot_a_chosen = rdot_a(result.n_star_j2000)
    samples = list(_hemisphere(s_hat, 6000))
    rdot_a_brute = max(rdot_a(n) for n in samples)
    assert rdot_a_chosen > 0.0
    assert rdot_a_chosen >= 0.999 * rdot_a_brute  # within grid res of the max

    # The apoapsis-maximising orientation gives a strictly
    # bigger rdot_a than the adot-maximising orientation -- proving the
    # ``a*edot`` contribution is being exploited. The margin depends on
    # geometry (at peri/apoapsis the two optima coincide; the divergence is
    # set by orbit phase and McInnes-cone availability of radial thrust).
    n_da_max = max(
        samples, key=lambda n: _rtn_rates(n, el, r, v, s_hat, sail).da_dt_km_s
    )
    rdot_a_at_da_optimum = rdot_a(n_da_max)
    assert rdot_a_chosen > rdot_a_at_da_optimum * 1.005


# ---------------------------------------------------------------------------
# Cross-check 3 -- the eccentricity term co-steers
# ---------------------------------------------------------------------------


def test_eccentricity_term_genuinely_co_steers():
    """Turning the e-term on (w_e = 1) vs off (w_e = 0) shifts the chosen
    orientation by an amount that is an O(1) fraction of the eccentricity
    rate maximum. This guards the rate-normalisation behavior."""
    sail = make_canonical_sail(0.018)
    # nu = 90 deg -- not at peri/apoapsis, so f_r genuinely co-controls de/dt
    # and the a- and e-weighted laws can pick distinct orientations.
    r, v = _state_at_nu(4200.0, 0.08, 90.0, _mu())
    el = osculating_elements(r, v, _mu())
    assert el.e > 0.05  # genuine eccentricity to control
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat

    no_e = _params(w_apo=1.0, w_e=0.0, w_p=0.0)
    with_e = _params(w_apo=1.0, w_e=1.0, w_p=0.0)
    res_no_e = steer(el, r, v, s_hat, P_SRP, sail=sail,
                     current_n_hat=np.array(s_hat), params=no_e)
    res_with_e = steer(el, r, v, s_hat, P_SRP, sail=sail,
                       current_n_hat=np.array(s_hat), params=with_e)
    assert res_no_e.thrust and res_with_e.thrust

    de_no_e = _rtn_rates(res_no_e.n_star_j2000, el, r, v, s_hat, sail).de_dt_per_s
    de_with_e = _rtn_rates(
        res_with_e.n_star_j2000, el, r, v, s_hat, sail
    ).de_dt_per_s
    f_char = float(np.linalg.norm(
        mcinnes_srp_acceleration(s_hat, s_hat, P_SRP, sail)
    ))
    edot_xx = eccentricity_rate_max(el.a_km, el.e, f_char, el.mu_km3_s2)

    # The e-term steers toward lower de/dt ...
    assert de_with_e < de_no_e
    # ... by at least 1% of the eccentricity-rate maximum. A contribution at
    # only ~1e-7 * edot_xx would not satisfy this threshold.
    assert (de_no_e - de_with_e) >= 0.01 * edot_xx


# ---------------------------------------------------------------------------
# Cross-check 4 -- periapsis barrier engages when periapsis is low
# ---------------------------------------------------------------------------


def test_periapsis_barrier_engages_below_rp_min():
    """Below r_p_min the penalty exceeds unity and a w_p-weighted law steers
    more protectively (higher periapsis rate) than one with the barrier off."""
    sail = make_canonical_sail(0.018)
    params = _params()
    # Periapsis well below r_p_min (= R_eq + 300 = 3696.19 km), at nu = 90 deg
    # so r_p is genuinely steerable (at peri/apo, transverse thrust leaves r_p
    # unchanged regardless of weighting).
    r, v = _state_at_nu(4000.0, 0.12, 90.0, _mu())
    el = osculating_elements(r, v, _mu())
    rp = el.a_km * (1.0 - el.e)
    assert rp < params.rp_min_km

    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat
    res = steer(el, r, v, s_hat, P_SRP, sail=sail,
                current_n_hat=np.array(s_hat), params=params)
    # The diagnostic penalty is engaged (> 1) for a stressed periapsis.
    assert res.q_value > 1.0

    # A barrier-on law raises periapsis faster than a barrier-off one.
    barrier_off = _params(w_apo=1.0, w_e=0.0, w_p=0.0)
    barrier_on = _params(w_apo=1.0, w_e=0.0, w_p=50.0)
    res_off = steer(el, r, v, s_hat, P_SRP, sail=sail,
                    current_n_hat=np.array(s_hat), params=barrier_off)
    res_on = steer(el, r, v, s_hat, P_SRP, sail=sail,
                   current_n_hat=np.array(s_hat), params=barrier_on)

    def rp_dot(res_):
        rates = _rtn_rates(res_.n_star_j2000, el, r, v, s_hat, sail)
        return (1.0 - el.e) * rates.da_dt_km_s - el.a_km * rates.de_dt_per_s

    assert rp_dot(res_on) > rp_dot(res_off)


# ---------------------------------------------------------------------------
# Sanity anchors + coast / feather branches
# ---------------------------------------------------------------------------


def test_ideal_sail_force_is_purely_normal():
    """Sanity anchor: an ideal sail (rho=s=1, eps=0) at any pitch produces a
    force with no transverse (sunward-tangent) component -- C_s = 1 - rho s = 0."""
    sail = SolarSail(area_m2=1000.0, mass_kg=18.0, optical=SailOptical.ideal())
    s_hat = np.array([1.0, 0.0, 0.0])
    n = np.array([math.cos(0.4), math.sin(0.4), 0.0])
    a_srp = mcinnes_srp_acceleration(n, s_hat, P_SRP, sail)
    # Force is along -n_hat for an ideal sail; cross product with n is zero.
    assert float(np.linalg.norm(np.cross(a_srp, n))) == pytest.approx(0.0, abs=1e-22)


def test_coasts_when_sun_is_ahead_in_direction_of_motion():
    """When the Sun lies in the +theta_hat direction (ahead, along the
    velocity) the law wants prograde thrust but SRP can only push
    anti-sunward -> no orientation raises the reward -> coast (feather)."""
    params = _params()
    sail = make_canonical_sail(0.018)
    r, v = _state()
    el = osculating_elements(r, v, _mu())
    r_hat, theta_hat, h_hat = rtn_basis(r, v)
    s_hat = np.array(theta_hat)  # Sun ahead, along the motion

    current_n = np.array([1.0, 0.0, 0.0])
    result = steer(el, r, v, s_hat, P_SRP, sail=sail, current_n_hat=current_n,
                   params=params)
    assert result.thrust is False
    assert result.dQ_dt == 0.0
    # Feathered: edge-on to the Sun -> zero projected area.
    assert abs(float(np.dot(result.n_star_j2000, s_hat))) == pytest.approx(
        0.0, abs=1e-9
    )
    assert float(np.linalg.norm(result.n_star_j2000)) == pytest.approx(1.0)


def test_shadow_pressure_zero_coasts():
    """Zero SRP pressure (eclipse) -> no force achievable -> feather."""
    params = _params()
    sail = make_canonical_sail(0.018)
    r, v = _state()
    el = osculating_elements(r, v, _mu())
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat
    result = steer(el, r, v, s_hat, 0.0, sail=sail,
                   current_n_hat=np.array([1.0, 0.0, 0.0]), params=params)
    assert result.thrust is False
    assert result.f_char_km_s2 == 0.0


def test_escaped_orbit_feathers():
    """A hyperbolic state (a < 0) is past the escape objective -> feather."""
    params = _params()
    sail = make_canonical_sail(0.018)
    # Hyperbolic: speed well above local escape speed.
    r = np.array([5000.0, 0.0, 0.0])
    v = np.array([0.0, 6.0, 0.0])
    el = osculating_elements(r, v, _mu())
    assert el.a_km < 0.0
    _, theta_hat, _ = rtn_basis(r, v)
    result = steer(el, r, v, -theta_hat, P_SRP, sail=sail,
                   current_n_hat=np.array([1.0, 0.0, 0.0]), params=params)
    assert result.thrust is False


# ---------------------------------------------------------------------------
# Cross-check 5 -- Petropoulos effectivity-coast
# ---------------------------------------------------------------------------


def test_evaluate_orbit_effectivity_returns_envelope():
    """The envelope is non-degenerate on an eccentric state: a thrust-effective
    moment exists somewhere around the orbit (qdot_min < 0) and the spread
    across orbit phases is meaningful (qdot_min < qdot_max)."""
    params = _params()
    sail = make_canonical_sail(0.018)
    r, v = _state_at_nu(4200.0, 0.08, 60.0, _mu())
    el = osculating_elements(r, v, _mu())
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat
    envelope = evaluate_orbit_effectivity(
        el, s_hat, P_SRP, sail, params, n_samples=36
    )
    assert envelope is not None
    qdot_min, qdot_max = envelope
    assert qdot_min < 0.0
    assert qdot_min <= qdot_max
    # The spread is real -- not all phases of the orbit are equally effective.
    assert qdot_min < 0.95 * qdot_max  # both negative; qdot_min more negative


def test_effectivity_threshold_zero_preserves_default_behavior():
    """With both thresholds = 0, supplying an envelope does NOT change the
    decision -- the default 'thrust whenever dQ/dt < 0' behaviour is preserved.
    Regression guard: callers that set effectivity_envelope without setting
    thresholds are unaffected."""
    params = _params()  # eta_a_threshold and eta_r_threshold default 0
    sail = make_canonical_sail(0.018)
    r, v = _state()
    el = osculating_elements(r, v, _mu())
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat
    current_n = np.array(s_hat)

    r_no_env = steer(el, r, v, s_hat, P_SRP, sail=sail,
                     current_n_hat=current_n, params=params)
    # Synthesise an envelope that WOULD coast if thresholds were positive.
    r_env = steer(el, r, v, s_hat, P_SRP, sail=sail,
                  current_n_hat=current_n, params=params,
                  effectivity_envelope=(-1.0e-6, -1.0e-12))
    assert r_no_env.thrust == r_env.thrust
    assert r_no_env.dQ_dt == pytest.approx(r_env.dQ_dt)
    np.testing.assert_allclose(r_no_env.n_star_j2000, r_env.n_star_j2000)


def test_effectivity_coasts_off_effectivity_arc():
    """Positive signal: with eta_a_threshold=0.5 and an envelope where the
    current achievable dQ/dt is far less negative than qdot_min (i.e. the state is
    on a low-effectivity arc), the law coasts. With the same envelope and
    threshold=0, the law thrusts -- isolating the gate to the threshold."""
    params_gated = _params(eta_a_threshold=0.5)
    params_default = _params()
    sail = make_canonical_sail(0.018)
    r, v = _state()
    el = osculating_elements(r, v, _mu())
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat
    current_n = np.array(s_hat)

    # Get the actual achievable dQ/dt at this state.
    r_default = steer(el, r, v, s_hat, P_SRP, sail=sail,
                      current_n_hat=current_n, params=params_default)
    assert r_default.thrust is True
    qdot_current = r_default.dQ_dt
    # Synthesise an envelope where qdot_min is 10x more negative than current:
    # eta_a = qdot_current / qdot_min = 0.1 << 0.5 threshold -> coast.
    envelope = (10.0 * qdot_current, 0.1 * qdot_current)
    r_gated = steer(el, r, v, s_hat, P_SRP, sail=sail,
                    current_n_hat=current_n, params=params_gated,
                    effectivity_envelope=envelope)
    assert r_gated.thrust is False
    assert r_gated.dQ_dt == 0.0
    # And feathered: n_star perpendicular to s_hat.
    assert abs(float(np.dot(r_gated.n_star_j2000, s_hat))) == pytest.approx(
        0.0, abs=1e-9
    )


def test_effectivity_thrusts_at_best_moment():
    """When the current state IS the orbit's best moment (qdot_current ~
    qdot_min, eta_a ~ 1), the threshold-0.5 gate passes through and the law
    thrusts."""
    params = _params(eta_a_threshold=0.5)
    sail = make_canonical_sail(0.018)
    r, v = _state()
    el = osculating_elements(r, v, _mu())
    _, theta_hat, _ = rtn_basis(r, v)
    s_hat = -theta_hat
    current_n = np.array(s_hat)

    r_no_env = steer(el, r, v, s_hat, P_SRP, sail=sail,
                     current_n_hat=current_n, params=_params())
    qdot_current = r_no_env.dQ_dt
    # Envelope: qdot_min = qdot_current (the state is at the best moment;
    # eta_a = 1.0).
    envelope = (qdot_current, 0.5 * qdot_current)
    r_gated = steer(el, r, v, s_hat, P_SRP, sail=sail,
                    current_n_hat=current_n, params=params,
                    effectivity_envelope=envelope)
    assert r_gated.thrust is True
    assert r_gated.dQ_dt == pytest.approx(qdot_current, rel=1e-9)
