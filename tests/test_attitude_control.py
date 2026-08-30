"""Fast tests for the attitude tracking controller (``reflectors.attitude_control``).

Cross-checks:

  1. A rest-to-rest reorientation respects ``|alpha| <= alpha_max`` and
     ``|omega| <= omega_max`` at every step and converges without overshoot.
  2. The slew time is sandwiched between the time-optimal bang-bang minimum
     (it cannot beat it) and the open-loop quintic ``smooth_slew`` time (a
     time-optimal accel-limited slew is faster than the C2 quintic), providing
     an independent comparison with the ``smooth_slew`` primitive.
  3. Tracking a slowly-moving target keeps the pointing error bounded.
  4. A binding ``omega_max`` is respected.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from reflectors.attitude_control import (
    AttitudeLimits,
    GovernorParams,
    alpha_command,
    attitude_derivatives,
    governor_omega_ref,
)


def _integrate_slew(n0, omega0, n_star_fn, limits, t_max, n_eval=1500):
    """Integrate the attitude state [n, omega] under the tracker.

    ``n_star_fn(t) -> n*`` allows a static or moving target. Returns
    ``(t, n, omega)`` arrays sampled uniformly on [0, t_max]."""

    def rhs(t, y):
        n = y[:3]
        omega = y[3:]
        n_dot, omega_dot = attitude_derivatives(n, omega, n_star_fn(t), limits)
        return np.concatenate([n_dot, omega_dot])

    y0 = np.concatenate([np.asarray(n0, float), np.asarray(omega0, float)])
    t_eval = np.linspace(0.0, t_max, n_eval)
    # Modest tolerances: the tests check bounds and ~1e-3 convergence, not
    # high-precision trajectories, and tight tolerances make DOP853 chase the
    # controller's ~1e-9 settling oscillation with tiny steps.
    sol = solve_ivp(
        rhs, (0.0, t_max), y0, method="DOP853",
        rtol=1e-8, atol=1e-10, t_eval=t_eval, max_step=2.0,
    )
    assert sol.success, sol.message
    return sol.t, sol.y[:3].T, sol.y[3:].T


def _theta_e(n, n_star):
    return math.acos(max(-1.0, min(1.0, float(np.dot(
        n / np.linalg.norm(n), n_star / np.linalg.norm(n_star)
    )))))


def _integrate_slew_rk4(n0, omega0, n_star_fn, limits, t_max, h_step):
    """Fixed-step RK4 integration of the attitude state under the tracker --
    the fixed-step integration mode (vs the adaptive ``_integrate_slew``).

    Mirrors :func:`reflectors.escape._integrate_escape_rk4`'s step structure
    over just the 6-D attitude slice ``[n, omega]``: classical RK4 with a
    fixed step ``h_step``, plus the post-step ``|omega|`` projection that
    enforces the strict bound (the same projection
    :func:`reflectors.escape._project_omega_to_max` applies to the 12-D
    escape state's omega slot).
    """

    def rhs(t, y):
        # Renormalise n before evaluation so the RHS sees
        # the same input attitude_derivatives expects. Without this, ~50k
        # RK4 steps drift |n| visibly at this precision (RHS is
        # ``dn/dt = omega x n`` which conserves |n| only at integration
        # tolerance).
        n = y[:3] / float(np.linalg.norm(y[:3]))
        omega = y[3:]
        n_dot, omega_dot = attitude_derivatives(n, omega, n_star_fn(t), limits)
        return np.concatenate([n_dot, omega_dot])

    y = np.concatenate([np.asarray(n0, float), np.asarray(omega0, float)])
    ts = [0.0]
    ys = [y.copy()]
    t = 0.0
    while t < t_max:
        h = min(h_step, t_max - t)
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Renormalise n after the step (the propagation path renormalises
        # on output via :func:`reflectors.escape.propagate_escape`'s
        # post-integration sweep; doing it here keeps the test's n a unit
        # vector for the bound checks).
        y[:3] = y[:3] / float(np.linalg.norm(y[:3]))
        # Apply the same post-step strict omega projection as the propagator.
        omega_mag = float(np.linalg.norm(y[3:]))
        if omega_mag > limits.omega_max_rad_s:
            y[3:] = y[3:] * (limits.omega_max_rad_s / omega_mag)
        t += h
        ts.append(t)
        ys.append(y.copy())
    arr = np.asarray(ys)
    return np.asarray(ts), arr[:, :3], arr[:, 3:]


# ---------------------------------------------------------------------------
# alpha_command basics
# ---------------------------------------------------------------------------


def test_zero_command_at_target_with_zero_rate():
    limits = AttitudeLimits()
    n = np.array([0.0, 0.0, 1.0])
    alpha = alpha_command(n, np.zeros(3), n, limits)
    assert float(np.linalg.norm(alpha)) == pytest.approx(0.0, abs=1e-15)


def test_command_never_exceeds_alpha_max():
    limits = AttitudeLimits(alpha_max_rad_s2=0.003)
    rng = np.random.default_rng(20260519)
    for _ in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        ns = rng.normal(size=3)
        ns /= np.linalg.norm(ns)
        omega = rng.normal(size=3) * 0.05
        alpha = alpha_command(n, omega, ns, limits)
        assert float(np.linalg.norm(alpha)) <= limits.alpha_max_rad_s2 * (1.0 + 1e-9)


def test_attitude_limits_validation():
    with pytest.raises(ValueError, match="alpha_max"):
        AttitudeLimits(alpha_max_rad_s2=0.0)
    with pytest.raises(ValueError, match="omega_max"):
        AttitudeLimits(omega_max_rad_s=-1.0)
    with pytest.raises(ValueError, match="omega_smooth"):
        AttitudeLimits(omega_smooth_rad_s=0.0)


# ---------------------------------------------------------------------------
# Cross-check 1 -- rest-to-rest slew respects the limits, no overshoot
# ---------------------------------------------------------------------------


def test_rest_to_rest_slew_respects_limits_and_converges():
    limits = AttitudeLimits(alpha_max_rad_s2=0.003)
    n0 = np.array([1.0, 0.0, 0.0])
    n_star = np.array([0.0, 1.0, 0.0])  # 90 deg slew
    t, n, omega = _integrate_slew(n0, np.zeros(3), lambda _t: n_star, limits, 120.0)

    # |omega| <= omega_max at every sample.
    omega_mag = np.linalg.norm(omega, axis=1)
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-6))

    # |alpha| <= alpha_max at every sample (recompute the command there).
    for ni, wi in zip(n, omega):
        a = alpha_command(ni, wi, n_star, limits)
        assert float(np.linalg.norm(a)) <= limits.alpha_max_rad_s2 * (1.0 + 1e-9)

    # Converged to the target.
    theta = np.array([_theta_e(ni, n_star) for ni in n])
    assert theta[-1] < 1.0e-3

    # No overshoot: the pointing error never grows (within numerical wiggle).
    assert np.all(np.diff(theta) < 1.0e-3)

    # n stays a unit vector through the integration.
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Cross-check 2 -- slew time vs the bang-bang minimum and the quintic anchor
# ---------------------------------------------------------------------------


def test_slew_time_between_bang_bang_minimum_and_quintic():
    """A 90 deg slew completes no faster than the time-optimal bang-bang
    minimum 2*sqrt(theta/alpha_max), and faster than the open-loop C2
    quintic smooth_slew sized to the same alpha_max."""
    limits = AttitudeLimits(alpha_max_rad_s2=0.003)
    theta = 0.5 * math.pi
    a_max = limits.alpha_max_rad_s2

    t, n, omega = _integrate_slew(
        np.array([1.0, 0.0, 0.0]), np.zeros(3),
        lambda _t: np.array([0.0, 1.0, 0.0]), limits, 120.0,
    )
    theta_e = np.array([_theta_e(ni, np.array([0.0, 1.0, 0.0])) for ni in n])
    omega_mag = np.linalg.norm(omega, axis=1)
    # "Slew complete" = at the target AND at rest (theta_e small reached
    # while still rotating happens a couple of seconds before braking is
    # finished -- that earlier instant is not the slew time).
    done = np.where((theta_e < 1.0e-3) & (omega_mag < 1.0e-3))[0]
    assert len(done) > 0, "slew did not complete"
    t_slew = t[done[0]]

    # Time-optimal bang-bang minimum (triangular accel profile).
    t_bang_bang = 2.0 * math.sqrt(theta / a_max)
    # Quintic smooth_slew time at the same alpha_max:
    #   |alpha|_max = theta * (10/sqrt 3) / T^2  ->  T = sqrt(theta*(10/sqrt3)/a).
    t_quintic = math.sqrt(theta * (10.0 / math.sqrt(3.0)) / a_max)

    # Cannot beat the time-optimal minimum; faster than the C2 quintic.
    assert t_slew >= 0.98 * t_bang_bang
    assert t_slew < t_quintic


# ---------------------------------------------------------------------------
# Cross-check 3 -- tracking a slowly-moving target
# ---------------------------------------------------------------------------


def test_tracks_slowly_moving_target_with_bounded_error():
    """A target rotating slowly in the x-y plane is tracked with a small,
    bounded pointing error after the initial acquisition transient."""
    limits = AttitudeLimits(alpha_max_rad_s2=0.003)
    rate = 1.0e-3  # rad/s -- slow vs the controller's slew authority

    def n_star_fn(t):
        return np.array([math.cos(rate * t), math.sin(rate * t), 0.0])

    n0 = n_star_fn(0.0)
    t, n, omega = _integrate_slew(n0, np.zeros(3), n_star_fn, limits, 600.0)
    theta_e = np.array([_theta_e(ni, n_star_fn(ti)) for ti, ni in zip(t, n)])

    # After the acquisition transient the lag error stays small and bounded.
    settled = theta_e[t > 120.0]
    assert np.max(settled) < 5.0e-2


# ---------------------------------------------------------------------------
# Cross-check 4 -- a binding omega_max is respected
# ---------------------------------------------------------------------------


def test_binding_omega_max_is_respected():
    """With omega_max set well below the natural slew peak, |omega| stays
    capped -- the slew runs through a rate-limited (trapezoidal) phase."""
    limits = AttitudeLimits(alpha_max_rad_s2=0.003, omega_max_rad_s=0.02)
    n_star = np.array([0.0, 1.0, 0.0])
    t, n, omega = _integrate_slew(
        np.array([1.0, 0.0, 0.0]), np.zeros(3), lambda _t: n_star,
        limits, 150.0,
    )
    omega_mag = np.linalg.norm(omega, axis=1)
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-3))
    # The cap actually bound -- the slew rate reached it.
    assert np.max(omega_mag) > 0.9 * limits.omega_max_rad_s
    # Still converges.
    assert _theta_e(n[-1], n_star) < 1.0e-3


# ---------------------------------------------------------------------------
# Cross-check 5 -- strict omega_max under the escape fixed-step RK4 stride
# ---------------------------------------------------------------------------


def test_alpha_command_hard_brakes_when_omega_at_or_above_omega_max():
    """At the boundary the command becomes full deceleration along the
    rotation axis. Closes the regime the interior switching curve covers
    weakly (~26% authority right at the limit).
    """
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )
    n = np.array([1.0, 0.0, 0.0])
    n_star = np.array([0.0, 1.0, 0.0])  # 90 deg slew, target far away

    # Just above the limit: full brake along -omega/|omega|.
    omega_dir = np.array([0.0, 0.0, 1.0])  # perpendicular to n; rotates n toward n_star
    omega_above = omega_dir * (limits.omega_max_rad_s * 1.05)
    alpha = alpha_command(n, omega_above, n_star, limits)
    expected = -limits.alpha_max_rad_s2 * omega_dir
    assert float(np.linalg.norm(alpha - expected)) < 1.0e-12
    # The magnitude saturates at alpha_max.
    assert float(np.linalg.norm(alpha)) == pytest.approx(
        limits.alpha_max_rad_s2, rel=1e-12,
    )

    # Right at the limit: the hard-brake branch fires (>= condition).
    omega_at = omega_dir * limits.omega_max_rad_s
    alpha = alpha_command(n, omega_at, n_star, limits)
    assert float(np.linalg.norm(alpha)) == pytest.approx(
        limits.alpha_max_rad_s2, rel=1e-12,
    )

    # Well below the limit: the interior switching-curve law (full
    # acceleration toward the target -- theta_e is large, omega_e ~ 0).
    omega_below = omega_dir * (0.1 * limits.omega_max_rad_s)
    alpha = alpha_command(n, omega_below, n_star, limits)
    # The interior law commands +alpha_max along the geodesic axis,
    # which here is +z = omega_dir; check the sign.
    assert float(np.dot(alpha, omega_dir)) > 0.5 * limits.alpha_max_rad_s2


def test_omega_strictly_below_omega_max_under_rk4_escape_stride():
    """Strict |omega| <= omega_max under the escape fixed-step RK4
    integrator at the late-phase escape stride (~36 s/step). The setup is a
    deliberately-fast rotating target the
    controller cannot track -- which keeps the tracker rate-saturated over
    the whole run. With the hard-brake branch + post-step projection, the
    bound holds strictly with no margin. The interior switching curve
    saturates near 0.95*omega_max (its 5% margin); reach that to confirm
    the regime is rate-saturated.
    """
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )
    # Target rotates faster than the controller's rate limit -> chronic
    # rate saturation. Mimics the worst-case n_des-vs-tracker mismatch the
    # late-phase escape exhibits.
    rate = 2.0 * limits.omega_max_rad_s

    def n_star_fn(t):
        return np.array([math.cos(rate * t), math.sin(rate * t), 0.0])

    n0 = n_star_fn(0.0)
    # 20-sol equivalent at ~36 s/step = ~50k steps -- enough for any
    # creep mechanism to manifest. (Each call to the test is ~1 s wall.)
    t_max = 20.0 * 88775.0  # 20 sols
    h_step = 36.0  # late-phase escape stride
    t, n, omega = _integrate_slew_rk4(
        n0, np.zeros(3), n_star_fn, limits, t_max, h_step,
    )
    omega_mag = np.linalg.norm(omega, axis=1)
    # Strict bound. Tolerance is float-precision (the post-step projection
    # clips to omega_max exactly; the only residual is a few epsilon from
    # the division-by-norm rescaling).
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-12))
    # The interior controller saturates at ~0.95*omega_max (the early-brake
    # margin); confirms the test is in the rate-saturated regime.
    assert np.max(omega_mag) > 0.9 * limits.omega_max_rad_s
    # The sail normal remains bounded around unit length; renormalization occurs
    # only on the propagation path's output.
    n_norms = np.linalg.norm(n, axis=1)
    assert np.all(n_norms > 0.9) and np.all(n_norms < 1.1)


def test_omega_strict_under_rk4_with_discontinuous_target_jumps():
    """The rate bound holds across discontinuous desired-normal changes.

    A discontinuity can rotate the geodesic axis before the interior switching
    curve brakes the preceding component. This exercises the hard-brake branch
    and post-step projection.
    """
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )
    # Four target orientations with approximately 90-degree transitions.
    targets = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    flip_period_s = 200.0

    def n_star_fn(t):
        return targets[int(t // flip_period_s) % 4]

    n0 = np.array([1.0, 0.0, 0.0])
    t_max = 5.0 * 88775.0  # 5 sols, hundreds of flips
    h_step = 36.0
    t, n, omega = _integrate_slew_rk4(
        n0, np.zeros(3), n_star_fn, limits, t_max, h_step,
    )
    omega_mag = np.linalg.norm(omega, axis=1)
    # Strict bound under discontinuous jumps that repeatedly reset the
    # geodesic-axis brake. Tolerance is float precision.
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-12))
    # The cap is exercised -- omega reaches at least 90% of omega_max
    # somewhere in the run (the controller bang-bangs through each jump).
    assert np.max(omega_mag) > 0.9 * limits.omega_max_rad_s


# ---------------------------------------------------------------------------
# Cross-check 6 -- reference governor
# ---------------------------------------------------------------------------


def test_governor_params_validation():
    with pytest.raises(ValueError, match="omega_ref_max"):
        GovernorParams(omega_ref_max_rad_s=0.0)
    with pytest.raises(ValueError, match="theta_settle"):
        GovernorParams(omega_ref_max_rad_s=0.1, theta_settle_rad=-0.01)


def test_governor_omega_ref_magnitude_strictly_bounded():
    """Direct unit test: |omega_ref| <= omega_ref_max for any input.

    The rate-saturated slerp must respect the rate cap at the rate-law level,
    independent of integration -- this is the foundation of the governor's
    guarantee that the tracker has slew headroom.
    """
    params = GovernorParams(
        omega_ref_max_rad_s=math.radians(0.24), theta_settle_rad=0.01,
    )
    rng = np.random.default_rng(20260520)
    # Far-from-target: rate magnitude == omega_ref_max exactly.
    n_ref = np.array([1.0, 0.0, 0.0])
    n_des = np.array([0.0, 1.0, 0.0])  # 90 deg away
    omega_ref = governor_omega_ref(n_ref, n_des, params)
    assert float(np.linalg.norm(omega_ref)) == pytest.approx(
        params.omega_ref_max_rad_s, rel=1e-12,
    )

    # Random orientations: always <= cap.
    for _ in range(200):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        b = rng.normal(size=3); b /= np.linalg.norm(b)
        w = governor_omega_ref(a, b, params)
        assert float(np.linalg.norm(w)) <= params.omega_ref_max_rad_s * (1.0 + 1e-12)


def test_governor_proportional_inside_theta_settle():
    """Inside the small-angle softening band the rate ramps proportionally
    so n_ref converges smoothly to n_des rather than chattering."""
    params = GovernorParams(
        omega_ref_max_rad_s=0.01, theta_settle_rad=0.1,
    )
    # 0.05 rad off-target = 50% of theta_settle -> expect 50% of cap.
    theta = 0.05
    n_ref = np.array([1.0, 0.0, 0.0])
    n_des = np.array([math.cos(theta), math.sin(theta), 0.0])
    omega_ref = governor_omega_ref(n_ref, n_des, params)
    expected_mag = 0.5 * params.omega_ref_max_rad_s
    assert float(np.linalg.norm(omega_ref)) == pytest.approx(expected_mag, rel=1e-6)


def test_governor_n_ref_converges_to_discontinuous_n_des_smoothly():
    """The integration test: when n_des jumps 90 deg, n_ref slews toward it
    at the rate cap without overshoot, converging within ~theta_settle.
    """
    params = GovernorParams(
        omega_ref_max_rad_s=math.radians(0.24), theta_settle_rad=0.01,
    )
    n_des = np.array([0.0, 1.0, 0.0])
    n_ref = np.array([1.0, 0.0, 0.0])

    # Integrate n_ref dynamics with a simple Euler step at 1 s (way below
    # the natural slew time of ~6 min for 90 deg at 0.24 deg/s).
    dt = 1.0
    history_theta = []
    history_rate = []
    for _ in range(500):  # 500 s
        omega_ref = governor_omega_ref(n_ref, n_des, params)
        n_ref = n_ref + np.cross(omega_ref, n_ref) * dt
        n_ref = n_ref / float(np.linalg.norm(n_ref))
        theta = math.acos(max(-1.0, min(1.0, float(np.dot(n_ref, n_des)))))
        history_theta.append(theta)
        history_rate.append(float(np.linalg.norm(omega_ref)))

    history_theta = np.asarray(history_theta)
    history_rate = np.asarray(history_rate)
    # Monotone decrease (no overshoot).
    assert np.all(np.diff(history_theta) <= 1e-6)
    # Rate never exceeds the cap.
    assert np.all(history_rate <= params.omega_ref_max_rad_s * (1.0 + 1e-12))
    # Converged.
    assert history_theta[-1] < 1e-3
