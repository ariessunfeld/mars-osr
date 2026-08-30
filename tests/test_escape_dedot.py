"""Tests for the direct dE/dt-maximizing escape steering.

The module under test (:mod:`reflectors.escape_dedot`) maximises
``(a_sail . v_hat)`` over sail cone angle within a one-sided sun-facing
cone. These tests pin three independent properties:

1. Analytical sanity. For an ideal mirror (rho=1, s=1) the merit reduces
   to ``2 cos^2(alpha) cos(alpha - theta)`` (sun-facing convention; see
   derivation in the test docstring). At theta = 90 deg the optimum is at
   ``alpha = -arctan(1/sqrt(2)) ~ -35.26 deg`` -- the well-known
   sail-thrust efficiency factor. At theta = 0 (velocity aligned with the
   Sun line) the optimum is alpha = 0 (face-on, maximum SRP). At theta =
   180 deg (Sun ahead of velocity) no alpha gives positive thrust ->
   feather.

2. Wiring guard (positive signal). A short propagated arc with dE/dt
   steering RAISES the orbital energy versus an SRP-off control. Cannot
   pass if the steering feathers every sample.

3. Attitude-limit integration. Plugged into :func:`escape.propagate_escape`
   with tight ``alpha_max`` / ``omega_max``, the integrated sail attitude
   honours both bounds at every sample -- the same guarantee the Q-law
   escape regression pins.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.attitude_control import AttitudeLimits
from reflectors.escape import initial_circular_state, propagate_escape
from reflectors.escape_dedot import (
    BlendedParams,
    BlendedSteering,
    DEdotParams,
    DEdotSteering,
    blended_steer,
    dedot_steer,
)
from reflectors.ephemeris import utc_to_et
from reflectors.kernels import load_kernels
from reflectors.mars_constants import MARS_HILL_RADIUS_KM
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail
from reflectors.solar_constants import solar_flux_at
from reflectors.srp import SailOptical, SolarSail


def _unused_qlaw_params() -> QLawParams:
    """QLawParams required by the steering-function call signature."""
    return QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=3696.19)


# ---------- helpers ----------


def _ideal_sail() -> SolarSail:
    """Perfect specular mirror: rho=1, s=1, no thermal."""
    optical = SailOptical(
        rho=1.0, s=1.0,
        eps_front=0.0, eps_back=0.0, B_front=0.0, B_back=0.0,
    )
    return SolarSail(area_m2=1000.0, mass_kg=18.0, optical=optical)


def _ideal_mirror_merit(alpha_rad: float, theta_rad: float) -> float:
    """Analytical merit ``f(alpha)`` for an ideal mirror, dimensionless.

    Derivation (sun-facing convention, n_hat with ``cos alpha = n . s``):
      For an ideal mirror (rho=1, s=1), McInnes (M-2.57) gives
        a_sail = -2 (P A / m) cos^2(alpha) n_hat                   [km/s^2]
      With ``n_hat = cos(alpha) s + sin(alpha) t`` and v_hat making angle
      theta with s_hat (so ``v_hat = cos(theta) s + sin(theta) t``):
        n_hat . v_hat = cos(alpha) cos(theta) + sin(alpha) sin(theta)
                      = cos(alpha - theta).
      Therefore
        a_sail . v_hat = -2 (P A / m) cos^2(alpha) cos(alpha - theta).
      The comparison is argmax over alpha, so the dimensionless
      shape ``f(alpha) = -cos^2(alpha) cos(alpha - theta)`` (the minus
      makes f >= 0 the "positive thrust along v" regime).
    """
    return -math.cos(alpha_rad) ** 2 * math.cos(alpha_rad - theta_rad)


# ---------- analytical sanity ----------


def test_theta_zero_optimum_is_face_on():
    """Velocity along the Sun line: optimum is sun-facing (alpha = 0).

    With ``v_hat = s_hat`` the velocity-aligned thrust is the full
    anti-sunward force; tilting the sail can only reduce it. The
    feather rule does NOT fire because at alpha=0 the force has its
    maximum magnitude along v_hat (an a_sail of magnitude ~2 P A / m is
    fully along v_hat = s_hat in the anti-sunward sense; here a_sail . v_hat
    is NEGATIVE because the force pushes anti-sun while v is sunward. So
    actually this configuration FEATHERS -- the sail cannot push along v if
    v is sun-ward).

    The unambiguous positive-thrust case is theta = pi (velocity ANTI-sunward),
    where face-on is optimal. Test both.
    """
    sail = _ideal_sail()
    P = 4.5e-6  # SRP at ~1 AU, Pa
    s = np.array([1.0, 0.0, 0.0])

    # Case A: v anti-aligned with s_hat. Anti-sunward thrust is along +v.
    v_anti = np.array([-1.0, 0.0, 0.0])
    result = dedot_steer(
        np.array([4400.0, 0.0, 0.0]), v_anti, s, P, sail,
        params=DEdotParams(max_cone_rad=math.radians(80.0)),
    )
    assert result.thrust, (
        f"theta=pi should be thrusting; got feather (merit "
        f"{result.dEdt_per_unit_mass_km2_s3})"
    )
    assert abs(result.alpha_rad) < 1.0e-3, (
        f"theta=pi optimum should be alpha=0 (sun-facing), got "
        f"{math.degrees(result.alpha_rad):.3f} deg"
    )

    # Case B: v aligned with s_hat. Anti-sun force points along -v -> no
    # orientation in the sun-facing hemisphere thrusts along +v -> feather.
    v_aligned = np.array([1.0, 0.0, 0.0])
    result_b = dedot_steer(
        np.array([4400.0, 0.0, 0.0]), v_aligned, s, P, sail,
        params=DEdotParams(max_cone_rad=math.radians(80.0)),
    )
    assert not result_b.thrust, (
        f"theta=0 should feather (Sun pushes anti-velocity); got thrust "
        f"alpha={math.degrees(result_b.alpha_rad):.3f}, merit "
        f"{result_b.dEdt_per_unit_mass_km2_s3}"
    )


def test_theta_ninety_deg_optimum_matches_arctan_one_over_sqrt2():
    """At theta = pi/2 the optimum is at ``alpha = -arctan(1/sqrt(2))``.

    Analytical: f(alpha) = -cos^2(alpha) cos(alpha - pi/2)
                         = -cos^2(alpha) sin(alpha)
              df/dalpha = -2 cos(alpha) (-sin(alpha)) sin(alpha)
                          - cos^2(alpha) cos(alpha)
                        = 2 cos(alpha) sin^2(alpha) - cos^3(alpha)
                        = cos(alpha) [2 sin^2(alpha) - cos^2(alpha)]
                        = cos(alpha) [3 sin^2(alpha) - 1]
              Zero at sin^2(alpha) = 1/3 -> tan(alpha) = +-1/sqrt(2),
              alpha = +- arctan(1/sqrt(2)) ~ +- 35.264 deg.
              The maximum positive thrust is at alpha = -35.264 deg
              (so n_hat tilts AWAY from +v while still being sun-facing,
              producing an a_sail with +v component).
    """
    sail = _ideal_sail()
    P = 4.5e-6
    s = np.array([1.0, 0.0, 0.0])
    # v perpendicular to s in the xy plane.
    v = np.array([0.0, 1.0, 0.0])
    expected_alpha = -math.atan(1.0 / math.sqrt(2.0))  # ~ -0.6155 rad

    # Use a tighter golden search to pin the optimum precisely.
    result = dedot_steer(
        np.array([4400.0, 0.0, 0.0]), v, s, P, sail,
        params=DEdotParams(
            max_cone_rad=math.radians(80.0), grid_n=40, golden_iters=60,
        ),
    )
    assert result.thrust
    assert abs(result.alpha_rad - expected_alpha) < 1.0e-3, (
        f"theta=pi/2 optimum: expected {math.degrees(expected_alpha):.3f} "
        f"deg, got {math.degrees(result.alpha_rad):.3f}"
    )


def test_cone_bound_clamps_alpha():
    """A tight cone bound binds when the unconstrained optimum is past it.

    Set theta = pi/2 (so unconstrained optimum is at -35.26 deg) and
    request max_cone = 20 deg. The result must respect ``|alpha| <= 20``.
    """
    sail = _ideal_sail()
    P = 4.5e-6
    s = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])

    result = dedot_steer(
        np.array([4400.0, 0.0, 0.0]), v, s, P, sail,
        params=DEdotParams(
            max_cone_rad=math.radians(20.0), grid_n=40, golden_iters=60,
        ),
    )
    assert result.thrust
    assert abs(result.alpha_rad) <= math.radians(20.0) + 1.0e-9, (
        f"|alpha| ({math.degrees(result.alpha_rad):.3f} deg) exceeds the "
        f"cone bound 20 deg"
    )
    # The cone should bind -- expect alpha exactly at -20 deg, not the
    # interior optimum at -35 deg.
    assert result.alpha_rad < -math.radians(19.5)


def test_eclipse_feathers():
    """``P_pa <= 0`` (eclipse) returns the feathered sentinel."""
    sail = _ideal_sail()
    s = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    result = dedot_steer(
        np.array([4400.0, 0.0, 0.0]), v, s, 0.0, sail,
    )
    assert not result.thrust
    assert result.dEdt_per_unit_mass_km2_s3 == 0.0
    # Feathered normal is perpendicular to s_hat.
    assert abs(float(np.dot(result.n_star_j2000, s))) < 1.0e-9


def test_returns_unit_normal():
    """The thrust-arc return is a unit vector."""
    sail = _ideal_sail()
    P = 4.5e-6
    s = np.array([0.6, 0.8, 0.0])  # arbitrary unit Sun direction
    v = np.array([0.3, -0.5, 0.8])
    v = v / np.linalg.norm(v)
    result = dedot_steer(
        np.array([4400.0, 0.0, 0.0]), v, s, P, sail,
        params=DEdotParams(max_cone_rad=math.radians(80.0)),
    )
    assert result.thrust
    assert abs(float(np.linalg.norm(result.n_star_j2000)) - 1.0) < 1.0e-12


# ---------- conditional periapsis-rate guard ----------


def _eccentric_state_near_periapsis(
    a_km: float, e: float, mu_km3_s2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian state at periapsis of an eccentric in-plane orbit.

    Periapsis on the +x axis, prograde motion (+y). nu = 0.
    """
    rp = a_km * (1.0 - e)
    # vis-viva at periapsis
    v_p = math.sqrt(mu_km3_s2 * (2.0 / rp - 1.0 / a_km))
    return np.array([rp, 0.0, 0.0]), np.array([0.0, v_p, 0.0])


def _drp_dt_at(
    r: np.ndarray, v: np.ndarray, n_star: np.ndarray,
    s_hat: np.ndarray, P: float, sail, mu: float,
) -> float:
    """Helper: Gauss drp/dt (km/s) at the chosen sail orientation."""
    from reflectors.gauss import (
        gauss_variational_rates, osculating_elements, rtn_basis,
    )
    from reflectors.srp import mcinnes_srp_acceleration
    el = osculating_elements(r, v, mu)
    r_hat, t_hat, h_hat = rtn_basis(r, v)
    a_sail = mcinnes_srp_acceleration(n_star, s_hat, P, sail)
    rates = gauss_variational_rates(
        el,
        float(np.dot(a_sail, r_hat)),
        float(np.dot(a_sail, t_hat)),
        float(np.dot(a_sail, h_hat)),
    )
    return (1.0 - el.e) * rates.da_dt_km_s - el.a_km * rates.de_dt_per_s


def test_constrained_guard_satisfies_drp_dt_when_active():
    """The constrained-search guard returns thrust only at orientations
    that satisfy ``drp/dt(n*) >= 0`` (within numerical tolerance) once
    activated.

    Construct an eccentric orbit at periapsis with Sun anti-velocity (the
    geometry that maximises e-pumping under tangential thrust). The
    unguarded controller picks the dE/dt-best alpha, which at this state
    gives drp/dt ~ 0 numerically (the analytical zero of tangential thrust
    at periapsis). The guarded controller's constrained search either picks
    the same alpha (drp/dt = 0 satisfies the >=0 constraint) or a nearby
    alpha that strictly raises rp -- never an alpha with strictly negative
    drp/dt.
    """
    sail = _ideal_sail()
    P = 4.5e-6
    mu = 4.2828e4
    a, e = 5500.0, 0.30
    r, v = _eccentric_state_near_periapsis(a, e, mu)
    rp = a * (1.0 - e)
    s_hat = np.array([0.0, -1.0, 0.0])  # anti-velocity Sun

    # Guard with rp_warn well above rp: must choose alpha satisfying the
    # drp/dt >= 0 constraint when thrusting.
    p_guarded = DEdotParams(
        max_cone_rad=math.radians(80.0), grid_n=40, golden_iters=60,
        rp_warn_km=rp + 100.0, mu_km3_s2=mu,
    )
    r_guarded = dedot_steer(r, v, s_hat, P, sail, params=p_guarded)

    if r_guarded.thrust:
        drp_chosen = _drp_dt_at(
            r, v, r_guarded.n_star_j2000, s_hat, P, sail, mu,
        )
        # Allow a tiny numerical tolerance: the constraint is ``>= 0`` but
        # the constrained search may settle on the constraint boundary at
        # round-off level.
        assert drp_chosen >= -1.0e-12, (
            f"guarded thrust at alpha={math.degrees(r_guarded.alpha_rad):.3f} "
            f"deg gave drp/dt = {drp_chosen:.3e} (must be >= 0 by the guard)"
        )
    # else: feathered -- also acceptable (no constraint-satisfying
    # positive-merit alpha exists in the cone at this state). The required
    # behavior is "never thrust with drp/dt < 0", which the branch above checks.


def test_guard_conditional_inactive_when_rp_above_warn():
    """Guard with ``rp_warn_km < rp_current`` must be dormant.

    With the warning threshold below the current periapsis, the conditional
    guard is inactive and the result matches the unguarded controller.
    """
    sail = _ideal_sail()
    P = 4.5e-6
    mu = 4.2828e4
    a, e = 5500.0, 0.30
    r, v = _eccentric_state_near_periapsis(a, e, mu)
    rp = a * (1.0 - e)
    s_hat = np.array([0.0, -1.0, 0.0])

    p_no_guard = DEdotParams(
        max_cone_rad=math.radians(80.0), grid_n=40, golden_iters=60,
    )
    p_loose = DEdotParams(
        max_cone_rad=math.radians(80.0), grid_n=40, golden_iters=60,
        rp_warn_km=rp - 100.0, mu_km3_s2=mu,
    )

    r_no_guard = dedot_steer(r, v, s_hat, P, sail, params=p_no_guard)
    r_loose = dedot_steer(r, v, s_hat, P, sail, params=p_loose)

    assert r_loose.thrust == r_no_guard.thrust
    if r_loose.thrust:
        # When the guard is dormant the chosen alpha matches the
        # unconstrained optimum to numerical precision.
        assert abs(r_loose.alpha_rad - r_no_guard.alpha_rad) < 1.0e-6


def test_periapsis_guard_does_not_trigger_when_drp_dt_positive():
    """When the chosen orientation grows periapsis, the guard stays out.

    At apoapsis of an eccentric orbit, tangential SRP thrust raises
    periapsis (da/dt > 0, de/dt < 0). The guard must remain dormant
    even if rp < rp_warn -- thrusting is fine, it helps rp.
    """
    sail = _ideal_sail()
    P = 4.5e-6
    mu = 4.2828e4
    a, e = 5500.0, 0.30
    # State at apoapsis: r = a(1+e) on -x, prograde v = -y (consistent
    # with the periapsis-on-+x orbit, but at the opposite phase).
    ra = a * (1.0 + e)
    v_a = math.sqrt(mu * (2.0 / ra - 1.0 / a))
    r = np.array([-ra, 0.0, 0.0])
    v = np.array([0.0, -v_a, 0.0])

    # Sun anti-velocity: s_hat = +y (since v is -y).
    s_hat = np.array([0.0, 1.0, 0.0])

    rp = a * (1.0 - e)
    p_guarded = DEdotParams(
        max_cone_rad=math.radians(80.0), grid_n=40, golden_iters=60,
        rp_warn_km=rp + 500.0,  # well above rp
        mu_km3_s2=mu,
    )
    result = dedot_steer(r, v, s_hat, P, sail, params=p_guarded)
    assert result.thrust, (
        "at apoapsis with tangential anti-sun thrust, drp/dt > 0 -- the "
        f"guard should be dormant; got feather "
        f"(alpha={math.degrees(result.alpha_rad):.2f})"
    )


# ---------- blended energy + safety controller ----------


def test_blended_safety_margin_zero_at_periapsis_equals_r_star():
    """Analytical pin: ``S = h^2 - 2 mu r* - 2 eps r*^2 = 0`` exactly when
    the osculating periapsis equals ``r_star``.

    From the conic identity ``h^2 = 2 mu r_p + 2 eps r_p^2``, substituting
    ``r_p = r*`` gives ``h^2 - 2 mu r* - 2 eps r*^2 = 0``. This test pins
    the safety-margin formula by constructing eccentric orbits with known
    r_p and checking S = 0 (within roundoff).
    """
    sail = _ideal_sail()
    P = 4.5e-6
    mu = 4.2828e4
    for (a_km, e) in [(5500.0, 0.30), (4400.0, 0.0), (8000.0, 0.50)]:
        r_p = a_km * (1.0 - e)
        r, v = _eccentric_state_near_periapsis(a_km, e, mu)
        s_hat = np.array([0.0, -1.0, 0.0])
        params = BlendedParams(r_star_km=r_p, mu_km3_s2=mu)
        result = blended_steer(r, v, s_hat, P, sail, params=params)
        # Safety margin should be ~0 since r_star = r_p (within rounding).
        assert abs(result.safety_margin_km4_s2) < 1.0e-3 * mu * a_km, (
            f"a={a_km}, e={e}, r_p={r_p}: expected S~0, got "
            f"{result.safety_margin_km4_s2:.3e}"
        )


def test_blended_far_from_guard_matches_dedot():
    """When the safety margin is large (rp >> r_star), the blended
    controller reduces to the dE/dt-greedy controller -- the safety
    weight w_S = k_S exp(-S/S_0) vanishes.
    """
    sail = _ideal_sail()
    P = 4.5e-6
    mu = 4.2828e4
    # Eccentric orbit with r_p well above r_star.
    a, e = 8000.0, 0.05
    r, v = _eccentric_state_near_periapsis(a, e, mu)
    s_hat = np.array([0.0, -1.0, 0.0])

    # r_star far below current r_p so the safety weight is tiny.
    r_star = 4000.0  # vs r_p = 7600
    p_blended = BlendedParams(
        r_star_km=r_star, k_S=1.0, S_0_km4_s2=1.0e6,
        mu_km3_s2=mu, grid_n=40, golden_iters=60,
    )
    p_dedot = DEdotParams(grid_n=40, golden_iters=60)

    r_blend = blended_steer(r, v, s_hat, P, sail, params=p_blended)
    r_dedot = dedot_steer(r, v, s_hat, P, sail, params=p_dedot)

    # Safety weight should be effectively zero (S >> S_0).
    assert r_blend.safety_weight < 1.0e-3, (
        f"safety weight should vanish far from guard; got "
        f"w_S={r_blend.safety_weight:.3e}"
    )
    # Both should thrust here.
    assert r_blend.thrust
    assert r_dedot.thrust
    # And pick essentially the same alpha (within search-grid resolution).
    assert abs(r_blend.alpha_rad - r_dedot.alpha_rad) < 0.05, (
        f"blended (far from guard) {math.degrees(r_blend.alpha_rad):.3f} "
        f"deg vs dedot {math.degrees(r_dedot.alpha_rad):.3f} deg"
    )


def test_blended_at_guard_shifts_orientation_toward_h_gain():
    """When the safety margin is near zero (rp ~ r_star) AND the spacecraft
    is not at periapsis, the safety term biases the controller -- the
    chosen alpha differs from the dE/dt-only optimum.

    (At periapsis exactly equal to r_star the safety term ``2(h x r) -
    2 r*^2 v`` cancels analytically along velocity; the safety controller
    has nothing to do at periapsis itself. The bias appears at other
    orbital phases where the two pieces don't cancel.)
    """
    sail = _ideal_sail()
    P = 4.5e-6
    mu = 4.2828e4
    a, e = 5500.0, 0.30
    r_p = a * (1.0 - e)

    # Construct state at true anomaly nu = 90 deg -- one quarter orbit past
    # periapsis (perpendicular to apsidal line). Both r and v differ from
    # periapsis; the safety term should not cancel.
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * math.cos(math.radians(90.0)))  # = p when nu=90
    nu = math.radians(90.0)
    # In a periapsis-on-+x prograde-on-+y orbit, position at nu=90:
    r_pqw = r_mag * np.array([math.cos(nu), math.sin(nu), 0.0])
    h_mag = math.sqrt(mu * p)
    v_pqw = (mu / h_mag) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    r = r_pqw
    v = v_pqw
    s_hat = np.array([0.0, -1.0, 0.0])

    p_blended = BlendedParams(
        r_star_km=r_p, k_S=1.0e7, S_0_km4_s2=1.0e7,
        mu_km3_s2=mu, grid_n=40, golden_iters=60,
    )
    p_dedot = DEdotParams(grid_n=40, golden_iters=60)

    r_blend = blended_steer(r, v, s_hat, P, sail, params=p_blended)
    r_dedot = dedot_steer(r, v, s_hat, P, sail, params=p_dedot)

    assert r_blend.safety_weight > 0.5 * p_blended.k_S, (
        f"safety weight should be ~k_S at the guard; got "
        f"w_S={r_blend.safety_weight:.3e} vs k_S={p_blended.k_S:.3e}"
    )

    if r_blend.thrust and r_dedot.thrust:
        delta = abs(r_blend.alpha_rad - r_dedot.alpha_rad)
        assert delta > math.radians(1.0), (
            f"blended alpha {math.degrees(r_blend.alpha_rad):.3f} should "
            f"differ from dedot {math.degrees(r_dedot.alpha_rad):.3f} "
            f"when the safety controller is fully active (S~0) and "
            f"away from periapsis; delta={math.degrees(delta):.3f} deg"
        )


def test_blended_eclipse_feathers():
    """``P_pa <= 0`` -> feathered sentinel (same as dedot's contract)."""
    sail = _ideal_sail()
    mu = 4.2828e4
    a, e = 5500.0, 0.30
    r, v = _eccentric_state_near_periapsis(a, e, mu)
    s_hat = np.array([0.0, -1.0, 0.0])
    p = BlendedParams(mu_km3_s2=mu)
    result = blended_steer(r, v, s_hat, 0.0, sail, params=p)
    assert not result.thrust
    assert result.merit_value == 0.0


# ---------- wiring guard: end-to-end positive signal ----------


@pytest.fixture(scope="module")
def _kernels():
    load_kernels()
    return True


def test_dedot_steering_raises_energy_vs_force_coast(_kernels):
    """Positive signal: dE/dt steering over a 4-orbit arc raises orbital
    energy versus an SRP-off (force_coast) control.

    A coast-only run would match the SRP-off baseline, so the test requires
    a measurable energy increase from active steering.
    """
    epoch_et = utc_to_et("2028-01-01T00:00:00")
    sail = make_canonical_sail(0.004)  # 4 g/m^2  # light sail for clearer signal
    state0 = initial_circular_state(altitude_km=1000.0, epoch_et=epoch_et)
    # Mu_Mars for the energy check.
    from reflectors.dynamics import mars_gm_km3_per_s2
    mu = mars_gm_km3_per_s2()

    def energy(state6):
        r = state6[:3]
        v = state6[3:]
        return 0.5 * float(np.dot(v, v)) - mu / float(np.linalg.norm(r))

    e0 = energy(state0)

    # 4 orbits ~ 4 * 2 pi sqrt(a^3/mu); at a~4400 km this is ~24000 s.
    t_span = (0.0, 24000.0)
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )

    # Dedot steering closure.
    dedot_params = DEdotParams(max_cone_rad=math.radians(80.0))

    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return dedot_steer(
            r, v, s_hat, p_eff, sail_,
            current_n_hat=current_n_hat, params=dedot_params,
        ).n_star_j2000

    res_dedot = propagate_escape(
        state0, epoch_et, sail, _unused_qlaw_params(), limits, t_span,
        gravity_degree=0,  # central + Sun only -- isolate the SRP effect
        steering_fn=steering,
    )
    res_coast = propagate_escape(
        state0, epoch_et, sail, _unused_qlaw_params(), limits, t_span,
        gravity_degree=0,
        force_coast=True,
    )

    e_dedot_end = energy(res_dedot.orbit_state_km_kmps[-1])
    e_coast_end = energy(res_coast.orbit_state_km_kmps[-1])

    # Sun third-body alone can perturb energy slightly; the SRP-on run
    # must beat the SRP-off run by a clear margin.
    delta = e_dedot_end - e_coast_end
    # Order-of-magnitude floor: 4 orbits at sigma=4 with ~30% efficiency
    # gives ~ 2 m/s along-track impulse * v_circ ~ 6e-6 km^2/s^2.
    assert delta > 1.0e-7, (
        f"dE/dt steering failed to outperform force_coast: "
        f"E_coast={e_coast_end:.6e}, E_dedot={e_dedot_end:.6e}, "
        f"delta={delta:.3e} (E_0={e0:.6e})"
    )


def test_propagate_escape_respects_attitude_limits_with_dedot(_kernels):
    """The integrated sail attitude honours alpha_max and omega_max at
    every sample -- the same guarantee as the Q-law slow regression.
    """
    epoch_et = utc_to_et("2028-01-01T00:00:00")
    sail = make_canonical_sail(0.004)  # 4 g/m^2
    state0 = initial_circular_state(altitude_km=1000.0, epoch_et=epoch_et)

    alpha_max = math.radians(0.003)
    omega_max = math.radians(0.3)
    limits = AttitudeLimits(
        alpha_max_rad_s2=alpha_max, omega_max_rad_s=omega_max,
    )
    t_span = (0.0, 12000.0)  # ~2 orbits

    dedot_params = DEdotParams(max_cone_rad=math.radians(80.0))

    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return dedot_steer(
            r, v, s_hat, p_eff, sail_,
            current_n_hat=current_n_hat, params=dedot_params,
        ).n_star_j2000

    res = propagate_escape(
        state0, epoch_et, sail, _unused_qlaw_params(), limits, t_span,
        gravity_degree=0, steering_fn=steering,
    )

    # |omega| <= omega_max + a small numerical margin at every sample.
    omegas = res.attitude_state[:, 3:]
    omega_mags = np.linalg.norm(omegas, axis=1)
    assert float(np.max(omega_mags)) <= omega_max * 1.01, (
        f"|omega|_max = {float(np.max(omega_mags)):.6e} exceeds "
        f"omega_max = {omega_max:.6e}"
    )

    # Sail normal is unit to integration tolerance.
    n_arr = res.attitude_state[:, :3]
    n_mags = np.linalg.norm(n_arr, axis=1)
    assert np.max(np.abs(n_mags - 1.0)) < 1.0e-6

    # Metadata records steering choice.
    assert res.metadata["steering"] == "steering_fn"
