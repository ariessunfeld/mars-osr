"""Fast smoke tests for the coupled escape propagation (``reflectors.escape``).

These are short (few-revolution) propagations -- the full escape regression
lives in ``tests/test_slow_regression.py``. They check that the coupled
12-D propagation is wired correctly: the integrated sail normal stays a unit
vector, the attitude limits are respected, and -- the positive control signal
-- the Q-law-steered SRP thrust actually raises the orbit, which a zero-force
wiring defect could not do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.attitude_control import (
    AttitudeLimits,
    GovernorParams,
    alpha_command,
)
from reflectors.central_body import mars_central_body
from reflectors.elements import classical_elements
from reflectors.ephemeris import utc_to_et
from reflectors.escape import initial_circular_state, propagate_escape
from reflectors.termination import RadiusCeiling
from reflectors.escape_dedot import BlendedParams, blended_steer
from reflectors.mars_constants import MARS_HILL_RADIUS_KM, SECONDS_PER_SOLAR_SOL_S
from reflectors.third_body import sun_third_body
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail
from reflectors.dynamics import mars_gm_km3_per_s2

R_EQ = 3396.19
EPOCH = "2028-01-01T00:00:00"


def _setup(altitude_km=500.0):
    epoch_et = utc_to_et(EPOCH)
    state0 = initial_circular_state(altitude_km, epoch_et)
    sail = make_canonical_sail(0.018)
    params = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=R_EQ + 300.0)
    limits = AttitudeLimits()
    return epoch_et, state0, sail, params, limits


# ---------------------------------------------------------------------------
# initial_circular_state
# ---------------------------------------------------------------------------


def test_initial_circular_state_is_circular_at_the_right_altitude():
    epoch_et = utc_to_et(EPOCH)
    state0 = initial_circular_state(500.0, epoch_et)
    r = state0[:3]
    v = state0[3:]
    assert float(np.linalg.norm(r)) == pytest.approx(R_EQ + 500.0, abs=1e-6)
    # Circular: r perpendicular to v.
    assert abs(float(np.dot(r, v))) / (
        np.linalg.norm(r) * np.linalg.norm(v)
    ) < 1e-12
    # Osculating eccentricity is ~0.
    from reflectors.dynamics import mars_gm_km3_per_s2
    el = classical_elements(state0, mars_gm_km3_per_s2(), epoch_et)
    assert el.e < 1e-6


# ---------------------------------------------------------------------------
# Central-body generalization: Mars stays BYTE-IDENTICAL
# ---------------------------------------------------------------------------


def test_central_body_default_none_is_byte_identical_to_explicit_mars():
    """The body-generalization (central_body / third_bodies kwargs) must leave
    the Mars path bit-for-bit unchanged. Run the same short propagation twice --
    once with the default call (central_body=None, third_bodies=None) and once
    with the explicit Mars objects selected by those defaults. The trajectories
    must be exactly equal (np.array_equal, not approximate)."""
    epoch_et, state0, sail, params, limits = _setup()
    span = 0.2 * SECONDS_PER_SOLAR_SOL_S
    bare = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
    )
    explicit = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
        central_body=mars_central_body(),
        third_bodies=(sun_third_body(),),
    )
    assert np.array_equal(bare.t_s, explicit.t_s)
    assert np.array_equal(bare.orbit_state_km_kmps, explicit.orbit_state_km_kmps)
    assert np.array_equal(bare.attitude_state, explicit.attitude_state)
    assert bare.termination_reason == explicit.termination_reason
    # The central-body and third-body metadata retain their public schema.
    assert explicit.metadata["central_body"]["naif_id"] == 499
    assert explicit.metadata["central_body"]["body_frame"] == "IAU_MARS"
    assert explicit.metadata["third_bodies"] == ["SUN"]
    assert explicit.metadata["drag"] is False


def test_drag_force_fn_none_is_a_noop():
    """The drag hook defaults to None and must be a true no-op: a run with
    drag_force_fn=None is byte-identical to one without the kwarg at all."""
    epoch_et, state0, sail, params, limits = _setup()
    span = 0.15 * SECONDS_PER_SOLAR_SOL_S
    a = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
    )
    b = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
        drag_force_fn=None,
    )
    assert np.array_equal(a.orbit_state_km_kmps, b.orbit_state_km_kmps)


# ---------------------------------------------------------------------------
# Coupled propagation smoke
# ---------------------------------------------------------------------------


def test_escape_smoke_raises_orbit_and_keeps_state_valid():
    """A few-revolution escape propagation: the sail normal stays unit, the
    attitude limits hold, and the orbit's semi-major axis increases -- the
    positive signal that Q-law-steered SRP thrust is wired through."""
    epoch_et, state0, sail, params, limits = _setup()
    span = 0.4 * SECONDS_PER_SOLAR_SOL_S  # ~4-5 revolutions
    result = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
    )
    assert result.termination_reason == "t_final"
    assert len(result.t_s) > 5

    # Sail normal stays a unit vector.
    n_norms = np.linalg.norm(result.sail_normals, axis=1)
    assert np.allclose(n_norms, 1.0, atol=1e-6)

    # Angular velocity within the limit.
    omega_mag = np.linalg.norm(result.angular_velocities_rad_s, axis=1)
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-6))

    # Commanded |alpha| within the limit at every sample.
    for i in range(len(result.t_s)):
        a_cmd = alpha_command(
            result.sail_normals[i],
            result.angular_velocities_rad_s[i],
            result.sail_normals[i],  # n_star not stored; |alpha| <= alpha_max
            limits,                  # holds for any n_star by construction
        )
        assert float(np.linalg.norm(a_cmd)) <= limits.alpha_max_rad_s2 * (1 + 1e-9)

    # Positive signal: the orbit was raised.
    mu = result.metadata["mu_central_km3_s2"]
    a_start = classical_elements(result.orbit_state_km_kmps[0], mu, epoch_et).a_km
    a_end = classical_elements(result.orbit_state_km_kmps[-1], mu, epoch_et).a_km
    assert a_end > a_start

    # No NaNs anywhere.
    assert np.all(np.isfinite(result.orbit_state_km_kmps))
    assert np.all(np.isfinite(result.attitude_state))


def test_force_coast_isolates_the_srp_driven_orbit_raising():
    """Control-off baseline. ``force_coast`` feathers the sail -> ~zero SRP
    thrust; the orbit then evolves under gravity + the Sun third body only.
    The Sun third body still exchanges energy with the Mars-centred orbit, so
    the coast run is not exactly energy-conserving -- but the SRP-attributable
    energy gain (the steered run minus that background) is unambiguously positive
    and dominates the background. Confirms the smoke test's orbit-raising is
    Q-law-steered SRP thrust, not a propagation artefact."""
    epoch_et, state0, sail, params, limits = _setup()
    span = 0.4 * SECONDS_PER_SOLAR_SOL_S

    coast = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span),
        gravity_degree=2, force_coast=True,
    )
    steered = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span),
        gravity_degree=2, force_coast=False,
    )
    mu = coast.metadata["mu_central_km3_s2"]

    def _energy(state):
        r = float(np.linalg.norm(state[:3]))
        v = float(np.linalg.norm(state[3:]))
        return 0.5 * v * v - mu / r

    de_coast = (
        _energy(coast.orbit_state_km_kmps[-1])
        - _energy(coast.orbit_state_km_kmps[0])
    )
    de_steered = (
        _energy(steered.orbit_state_km_kmps[-1])
        - _energy(steered.orbit_state_km_kmps[0])
    )
    # The steered run gains energy, and the SRP-attributable part (steered
    # minus the control-off background) is positive and exceeds the
    # background third-body exchange.
    assert de_steered > 0.0
    srp_part = de_steered - de_coast
    assert srp_part > 0.0
    assert srp_part > abs(de_coast)


# ---------------------------------------------------------------------------
# Periapsis true-anomaly step cap (Sundman-style periapsis resolution)
# ---------------------------------------------------------------------------


def test_periapsis_true_anomaly_step_cap_resolves_periapsis():
    """The true-anomaly step cap clusters RK4 steps through periapsis, cutting
    the one-orbit closure error of a coarse-step high-e two-body propagation.

    Pure Kepler (force_coast -> ~zero SRP, gravity_degree=0 -> point mass,
    kinematic_attitude -> frozen sail): the orbit must return to its start
    after exactly one period. A coarse fixed-steps/orbit under-resolves the
    fast periapsis passage; the cap (max true-anomaly advance per step) should
    materially reduce the closure error and add steps near periapsis.
    """
    epoch_et = utc_to_et(EPOCH)
    mu = mars_gm_km3_per_s2()
    a, e = 10000.0, 0.5
    rp = a * (1.0 - e)
    vp = math.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    state0 = np.array([rp, 0.0, 0.0, 0.0, vp, 0.0])
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    sail = make_canonical_sail(0.018)
    qlaw_shell = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=R_EQ + 300.0)
    limits = AttitudeLimits()
    common = dict(
        gravity_degree=0, include_sun_third_body=False,
        force_coast=True, kinematic_attitude=True, steps_per_orbit=40,
    )
    uncapped = propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits, (0.0, period), **common
    )
    capped = propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits, (0.0, period),
        max_step_true_anomaly_deg=1.0, **common
    )

    def closure_km(res):
        return float(np.linalg.norm(res.orbit_state_km_kmps[-1, :3] - state0[:3]))

    err_uncapped = closure_km(uncapped)
    err_capped = closure_km(capped)
    # The cap materially reduces the one-orbit closure error (periapsis now
    # resolved) -- a coarse 40-step/orbit run badly under-resolves e=0.5.
    assert err_capped < 0.5 * err_uncapped
    # The cap bites: many more steps (clustered through periapsis).
    assert capped.t_s.size > 2 * uncapped.t_s.size
    # Metadata records the knob; default None disables the cap.
    assert capped.metadata["max_step_true_anomaly_deg"] == 1.0
    assert uncapped.metadata["max_step_true_anomaly_deg"] is None


def test_absolute_step_cap_bounds_step_size():
    """``max_step_s`` caps the absolute RK4 step at every committed sample --
    holds attitude-tracker resolution as the orbit (and its period) grows.
    """
    epoch_et = utc_to_et(EPOCH)
    state0 = initial_circular_state(1000.0, epoch_et)
    sail = make_canonical_sail(0.018)
    qlaw_shell = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=R_EQ + 300.0)
    limits = AttitudeLimits()
    span = 0.2 * SECONDS_PER_SOLAR_SOL_S
    common = dict(
        gravity_degree=0, include_sun_third_body=False,
        force_coast=True, kinematic_attitude=True, steps_per_orbit=200,
    )
    # steps_per_orbit=200 at ~1000 km altitude gives a step well above 30 s;
    # the cap must bring every step at or below 30 s.
    uncapped = propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits, (0.0, span), **common
    )
    capped = propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits, (0.0, span),
        max_step_s=30.0, **common
    )
    assert float(np.max(np.diff(uncapped.t_s))) > 30.0  # cap actually binds
    assert float(np.max(np.diff(capped.t_s))) <= 30.0 + 1e-6
    assert capped.metadata["max_step_s"] == 30.0
    assert uncapped.metadata["max_step_s"] is None


# ---------------------------------------------------------------------------
# Reference governor -- end-to-end propagate_escape
# ---------------------------------------------------------------------------


def test_governor_kinematic_attitude_mutual_exclusion():
    """``governor_params`` and ``kinematic_attitude`` are mutually
    exclusive -- the governor wraps the tracker, kinematic_attitude
    bypasses the tracker."""
    epoch_et, state0, sail, params, limits = _setup()
    governor = GovernorParams(
        omega_ref_max_rad_s=0.8 * limits.omega_max_rad_s,
        theta_settle_rad=0.01,
    )
    span = 100.0  # short span; the error is raised before integration
    with pytest.raises(ValueError, match="mutually exclusive"):
        propagate_escape(
            state0, epoch_et, sail, params, limits, (0.0, span),
            gravity_degree=2,
            governor_params=governor,
            kinematic_attitude=True,
        )


def test_governor_omega_ref_max_must_be_below_tracker():
    """``GovernorParams.omega_ref_max_rad_s`` must be strictly < the
    tracker's ``omega_max_rad_s`` so the tracker has slew headroom."""
    epoch_et, state0, sail, params, limits = _setup()
    over_cap = GovernorParams(
        omega_ref_max_rad_s=limits.omega_max_rad_s,  # equal -- no headroom
        theta_settle_rad=0.01,
    )
    span = 100.0
    with pytest.raises(ValueError, match="headroom"):
        propagate_escape(
            state0, epoch_et, sail, params, limits, (0.0, span),
            gravity_degree=2,
            governor_params=over_cap,
        )


def test_governor_enabled_propagate_escape_strict_bounds_and_unit_normals():
    """End-to-end smoke with the governor enabled (realistic slew, blended
    steering). Verifies:
      - 15-D state propagates cleanly,
      - ``reference_normals`` is populated and unit,
      - sail normal stays unit,
      - strict |omega| <= omega_max,
      - tracking error |angle(n, n_ref)| is bounded.

    Positive signal: the orbit is raised by the SRP thrust.
    """
    epoch_et = utc_to_et(EPOCH)
    state0 = initial_circular_state(1000.0, epoch_et)
    sail = make_canonical_sail(0.018)
    qlaw_shell = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=R_EQ + 300.0)
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )
    governor = GovernorParams(
        omega_ref_max_rad_s=0.8 * limits.omega_max_rad_s,
        theta_settle_rad=0.01,
    )
    blended_params = BlendedParams(
        r_star_km=3996.19, w_E=1.0, k_S=1.0, S_0_km4_s2=1.0e7,
        max_cone_rad=math.radians(80.0), mu_km3_s2=mars_gm_km3_per_s2(),
    )

    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return blended_steer(
            r, v, s_hat, p_eff, sail_,
            current_n_hat=current_n_hat, params=blended_params,
        ).n_star_j2000

    span = 2.0 * SECONDS_PER_SOLAR_SOL_S  # 2 sols -- a couple revolutions
    result = propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits, (0.0, span),
        gravity_degree=2,
        steering_fn=steering,
        governor_params=governor,
    )
    assert result.termination_reason == "t_final"

    # Governor populated reference_normals.
    assert result.reference_normals is not None
    assert result.reference_normals.shape == (len(result.t_s), 3)

    # |n_ref| = 1 throughout.
    n_ref_norms = np.linalg.norm(result.reference_normals, axis=1)
    assert np.allclose(n_ref_norms, 1.0, atol=1e-6)

    # |n| = 1 throughout (unchanged from the non-governor smoke).
    n_norms = np.linalg.norm(result.sail_normals, axis=1)
    assert np.allclose(n_norms, 1.0, atol=1e-6)

    # Strict |omega| <= omega_max with the governor enabled.
    omega_mag = np.linalg.norm(result.angular_velocities_rad_s, axis=1)
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-12))

    # Tracking error |angle(n, n_ref)| stays bounded. The 30 deg limit checks
    # that the integrated tracker follows the slowly slewing reference.
    cos_track = np.einsum('ij,ij->i', result.sail_normals, result.reference_normals)
    cos_track = np.clip(cos_track, -1.0, 1.0)
    track_deg = np.degrees(np.arccos(cos_track))
    assert track_deg.max() < 30.0, (
        f"tracker lagging n_ref by {track_deg.max():.2f} deg "
        "(governor should keep this small)"
    )

    # Positive signal: orbit raised.
    mu = result.metadata["mu_central_km3_s2"]
    a_start = classical_elements(result.orbit_state_km_kmps[0], mu, epoch_et).a_km
    a_end = classical_elements(result.orbit_state_km_kmps[-1], mu, epoch_et).a_km
    assert a_end > a_start

    # Metadata records the governor params (provenance).
    assert result.metadata["governor_params"] is not None
    assert result.metadata["governor_params"]["omega_ref_max_rad_s"] == pytest.approx(
        governor.omega_ref_max_rad_s
    )


# ---------------------------------------------------------------------------
# Energy-gated escape definition: escape = first time E >= 0 AND |r| >= Hill.
# The plain RadiusCeiling fires on |r| >= Hill alone, so a bound orbit grazing
# the Hill radius is a false escape.
# ---------------------------------------------------------------------------


def _twobody_periapsis_state(periapsis_km, apoapsis_km, mu):
    """Planet-centred state at the PERIAPSIS of a bound 2-body orbit, in the xy
    plane moving +y (so |r| climbs from periapsis toward apoapsis)."""
    a = 0.5 * (periapsis_km + apoapsis_km)
    v_peri = math.sqrt(mu * (2.0 / periapsis_km - 1.0 / a))
    return np.array([periapsis_km, 0.0, 0.0, 0.0, v_peri, 0.0])


def _coast_2body(state0, epoch_et, sail, params, limits, span_s, ceiling_km,
                 *, energy_gated):
    """Deterministic pure 2-body coast harness for the gate tests: feathered
    sail (SRP=0), point-mass gravity, no third bodies, custom Hill ceiling."""
    return propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span_s),
        gravity_degree=0, force_coast=True, third_bodies=(),
        radius_ceiling=RadiusCeiling(radius_km=ceiling_km),
        energy_gated=energy_gated,
    )


def test_energy_gate_byte_identical_when_nothing_fires():
    """Adding the energy-gate must not perturb the dynamics: a short bound run
    (in which no terminal event fires) is byte-identical with the gate on vs
    off -- the gated detectors are evaluated but never accept."""
    epoch_et, state0, sail, params, limits = _setup()
    span = 0.2 * SECONDS_PER_SOLAR_SOL_S
    off = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
    )
    on = propagate_escape(
        state0, epoch_et, sail, params, limits, (0.0, span), gravity_degree=2,
        energy_gated=True,
    )
    assert off.termination_reason == on.termination_reason == "t_final"
    assert np.array_equal(off.t_s, on.t_s)
    assert np.array_equal(off.orbit_state_km_kmps, on.orbit_state_km_kmps)
    assert np.array_equal(off.attitude_state, on.attitude_state)


def test_energy_gate_byte_identical_on_genuine_hyperbolic_crossing():
    """When the orbit crosses the Hill ceiling with E >= 0 (a genuine escape),
    the gate accepts at the same crossing the radius-only event finds -> the run
    is byte-identical and still labelled hill_sphere_exit."""
    epoch_et, _, sail, params, limits = _setup()
    mu = mars_gm_km3_per_s2()
    r0 = 7000.0
    v_esc = math.sqrt(2.0 * mu / r0)
    # Outward (radial +x) AND hyperbolic: |v|^2 = 1.17 v_esc^2 -> E = +0.17 mu/r0.
    state0 = np.array([r0, 0.0, 0.0, 0.6 * v_esc, 0.9 * v_esc, 0.0])
    assert 0.5 * float(np.dot(state0[3:], state0[3:])) - mu / r0 > 0.0
    off = _coast_2body(state0, epoch_et, sail, params, limits, 5000.0, 9000.0,
                       energy_gated=False)
    on = _coast_2body(state0, epoch_et, sail, params, limits, 5000.0, 9000.0,
                      energy_gated=True)
    assert off.termination_reason == "hill_sphere_exit"
    assert on.termination_reason == "hill_sphere_exit"
    assert on.escaped
    assert np.array_equal(off.t_s, on.t_s)
    assert np.array_equal(off.orbit_state_km_kmps, on.orbit_state_km_kmps)


def test_energy_gate_rejects_bound_graze_that_radius_only_calls_escape():
    """THE positive signal: a BOUND orbit (E<0) whose apoapsis grazes the Hill
    radius is a FALSE escape under the radius-only event and is correctly NOT an
    escape under the gate. Pins that the gate is wired through to the integrator
    (an envelope-only check could not distinguish these)."""
    epoch_et, _, sail, params, limits = _setup()
    mu = mars_gm_km3_per_s2()
    # Periapsis 7000, apoapsis 8500 (bound); ceiling 8000 lies between them.
    state0 = _twobody_periapsis_state(7000.0, 8500.0, mu)
    assert 0.5 * float(np.dot(state0[3:], state0[3:])) - mu / 7000.0 < 0.0
    span = 22000.0  # > one orbital period; apoapsis reached well within span
    off = _coast_2body(state0, epoch_et, sail, params, limits, span, 8000.0,
                       energy_gated=False)
    on = _coast_2body(state0, epoch_et, sail, params, limits, span, 8000.0,
                      energy_gated=True)
    # Radius-only wrongly reports the bound graze as an escape...
    assert off.termination_reason == "hill_sphere_exit"
    assert off.escaped
    # ...the gate rejects it (E<0 at the crossing) -> NOT an escape.
    assert on.termination_reason != "hill_sphere_exit"
    assert not on.escaped


def test_energy_gate_outer_kill_radius_bounds_a_climbing_bound_graze():
    """A bound orbit that climbs past 2*Hill without ever reaching E>=0
    terminates 'outer_kill_radius' (a clear non-escape), not hill_sphere_exit
    and not a wasted full-span run."""
    epoch_et, _, sail, params, limits = _setup()
    mu = mars_gm_km3_per_s2()
    # Periapsis 7000, apoapsis 20000 (bound); ceiling 8000 -> outer kill 16000.
    state0 = _twobody_periapsis_state(7000.0, 20000.0, mu)
    assert 0.5 * float(np.dot(state0[3:], state0[3:])) - mu / 7000.0 < 0.0
    on = _coast_2body(state0, epoch_et, sail, params, limits, 25000.0, 8000.0,
                      energy_gated=True)
    assert on.termination_reason == "outer_kill_radius"
    assert not on.escaped
