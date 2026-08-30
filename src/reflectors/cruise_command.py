"""Time-Fourier cone/clock command steerer for the interplanetary cruise.

The interplanetary Earth-Hill -> Mars-Hill SRP-sail cruise is a fixed-time,
fixed-endpoint low-thrust transfer solved by direct single shooting: the
decision vector is a smooth, global, time-parameterised sail attitude command,
and the existing rate/accel-limited attitude tracker
(``reflectors.escape.propagate_escape``) makes the achieved trajectory
slew-feasible BY CONSTRUCTION.

This module supplies that command as a ``steering_fn`` plugged into
``propagate_escape``. The attitude is specified in the McInnes cone/clock
parameterisation (same basis convention as ``reflectors.cruise``), but with two
deliberate differences from the orbit-locked Mars cruise law:

  * the cone ``alpha(tau)`` and clock ``delta(tau)`` are functions of NORMALISED
    MISSION TIME ``tau = (et - et0) / T`` in ``[0, 1]`` (global Fourier series),
    NOT of the orbit phase ``u`` -- there is no repeating orbit over a one-way
    heliocentric transfer; and
  * the clock reference direction uses a FIXED reference normal (the heliocentric
    orbit normal at departure), not the live orbit normal -- this keeps the
    clock reference free of state feedback, which spreads the terminal
    sensitivity smoothly for single shooting.

Geometry (McInnes 1999 Eq 4.7 convention, matching ``cruise._per_call_geometry``):

    s_hat                : sail -> Sun unit vector (supplied live by the
                           propagator's _sun_geometry; for a Sun-centred run
                           s_hat = -r_hat).
    e_A = unit(ref - (ref . s_hat) s_hat)   clock zero-direction: the FIXED
                           reference normal projected into the plane perp to
                           s_hat (orbit-normal projection, cruise.py:260-276).
    e_B = s_hat x e_A     completes the right-handed clock basis.
    n_des = cos(alpha) s_hat + sin(alpha) (cos(delta) e_A + sin(delta) e_B)

``alpha`` (cone, angle of n from the Sun-line) is clamped to ``[0, pi/2]`` so the
sunward face always catches the light (``n . s_hat = cos(alpha) >= 0``); at
``alpha = pi/2`` the sail is edge-on (zero SRP). ``delta`` (clock, azimuth about
the Sun-line from e_A) is unconstrained (it enters only through cos/sin).

``n_des`` is unit by construction (s_hat, e_A, e_B orthonormal:
``|n_des|^2 = cos^2 a + sin^2 a (cos^2 d + sin^2 d) = 1``).
"""

from __future__ import annotations

import math

import numpy as np

# Below this the fixed reference normal is nearly parallel to the Sun line and
# its projection perp to s_hat is ill-defined; fall back to a deterministic
# perpendicular basis (mirrors cruise._per_call_geometry's pole fallback).
_EA_DEGENERATE_TOL = 1.0e-12


def heliocentric_orbit_normal(state6: np.ndarray) -> np.ndarray:
    """Unit orbit normal ``h_hat = unit(r x v)`` of a Cartesian state.

    Used to build the FIXED clock reference normal from the departure state.
    Raises if the state is (near-)radial (``|r x v| ~ 0``), which has no
    defined orbit plane.
    """
    state6 = np.asarray(state6, dtype=float)
    h = np.cross(state6[:3], state6[3:6])
    h_mag = float(np.linalg.norm(h))
    if h_mag < _EA_DEGENERATE_TOL:
        raise ValueError("state has no defined orbit plane (|r x v| ~ 0)")
    return h / h_mag


def heliocentric_angular_rate(state6: np.ndarray, mu_km3_s2: float) -> float:
    """Instantaneous angular rate of the radius vector ``|r x v| / |r|^2`` (rad/s).

    The clock reference frame (s_hat, e_A, e_B) is carried by the Sun-line
    s_hat = -r_hat, so the frame's angular speed equals this heliocentric
    angular rate. ``mu_km3_s2`` is accepted for interface symmetry (not used;
    the rate is purely kinematic) -- kept so callers can pass the same mu they
    propagate with.
    """
    state6 = np.asarray(state6, dtype=float)
    r = state6[:3]
    v = state6[3:6]
    r_mag = float(np.linalg.norm(r))
    if r_mag == 0.0:
        raise ValueError("radius is zero; angular rate undefined")
    return float(np.linalg.norm(np.cross(r, v))) / (r_mag * r_mag)


def _amp_sums(block: np.ndarray, order: int) -> tuple[float, float]:
    """Return ``(S, Q) = (sum_k k*amp_k, sum_k k^2*amp_k)`` for a coeff block.

    ``amp_k = sqrt(cc_k^2 + cs_k^2)`` is the amplitude of harmonic ``k``; ``S``
    bounds ``max|d/dtau|`` and ``Q`` bounds ``max|d^2/dtau^2|`` of the series
    (each harmonic contributes at most ``2 pi k amp_k`` to the first derivative
    and ``(2 pi k)^2 amp_k`` to the second, via |sin|,|cos| <= 1).
    """
    S = 0.0
    Q = 0.0
    for k in range(1, order + 1):
        amp = math.hypot(float(block[k]), float(block[order + k]))
        S += k * amp
        Q += k * k * amp
    return S, Q


def cruise_command_slew_bounds(
    coeffs: np.ndarray,
    T_s: float,
    omega_frame_max_rad_s: float = 0.0,
    omega_frame_accel_max_rad_s2: float = 0.0,
) -> tuple[float, float]:
    """Conservative analytic upper bounds on the commanded slew ``(|omega|, |alpha|)``.

    The commanded normal is ``n(t) = cos a s + sin a (cos d e_A + sin d e_B)``
    with ``a,d`` Fourier series in ``tau = (et-et0)/T``. Writing ``Sa,Sd`` and
    ``Qa,Qd`` for the harmonic-amplitude sums (``_amp_sums``):

      |dn/dt| <= |da/dt| + |dd/dt| + Omega_frame
              <= (2 pi / T)(Sa + Sd) + Omega_frame  =: B_omega        (1)

    using ``|dn/da|=1``, ``|dn/dd|=sin a <= 1`` and that the (s,e_A,e_B) frame
    rotates at the heliocentric angular rate ``Omega_frame``. The angular
    acceleration the slew must supply obeys ``|alpha| <= |d^2 n/dt^2|`` (for the
    omega.n=0 convention, ``|d^2n/dt^2|^2 = |alpha|^2 + |omega|^4``), and

      |d^2n/dt^2| <= (2 pi / T)^2 (Qa + Qd) + B_omega^2 + Omega_frame_dot =: B_alpha  (2)

    where the ``B_omega^2`` term bounds every quadratic-in-rate (path-curvature /
    centripetal) contribution and ``Omega_frame_dot`` is the frame angular
    acceleration. Both bounds are deliberately conservative (triangle
    inequalities); a test pins that they exceed the finite-differenced actual
    rates along a real trajectory.

    Parameters
    ----------
    coeffs, T_s
        As in :func:`make_cruise_command_steerer`.
    omega_frame_max_rad_s
        Max heliocentric angular rate over the arc (the clock frame's angular
        speed); from :func:`heliocentric_angular_rate`. Default 0 (frozen frame).
    omega_frame_accel_max_rad_s2
        Max heliocentric angular acceleration over the arc. Default 0.

    Returns
    -------
    (B_omega, B_alpha)
        Guaranteed upper bounds (rad/s, rad/s^2).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    order = _validate_order(coeffs.size)
    if not (T_s > 0.0):
        raise ValueError(f"T_s must be > 0, got {T_s}")
    half = 1 + 2 * order
    Sa, Qa = _amp_sums(coeffs[:half], order)
    Sd, Qd = _amp_sums(coeffs[half:], order)
    w = 2.0 * math.pi / T_s
    b_omega = w * (Sa + Sd) + float(omega_frame_max_rad_s)
    b_alpha = (w * w) * (Qa + Qd) + b_omega * b_omega + float(omega_frame_accel_max_rad_s2)
    return b_omega, b_alpha


def assert_cruise_slew_feasible(
    coeffs: np.ndarray,
    T_s: float,
    limits,
    omega_frame_max_rad_s: float = 0.0,
    omega_frame_accel_max_rad_s2: float = 0.0,
) -> tuple[float, float]:
    """Raise ``ValueError`` unless the command's guaranteed slew bounds fit the limits.

    Checks (a) the cone never saturates -- ``alpha(tau)`` stays inside
    ``[0, pi/2]`` for all tau, so the ``[0, pi/2]`` CLAMP never engages (a clamp
    would put a CORNER in ``n_des`` -> an instantaneous slew reversal ->
    unbounded angular acceleration, which the smooth rate bound does not model);
    and (b) ``B_omega <= limits.omega_max_rad_s`` and ``B_alpha <=
    limits.alpha_max_rad_s2`` (the conservative bounds from
    :func:`cruise_command_slew_bounds`). Returns the ``(B_omega, B_alpha)`` pair
    on success. The cruise propagation runs the command KINEMATICALLY (the
    smooth slow command needs no bang-bang tracker), so
    this is the slew-feasibility guarantee, replacing the integrated tracker.

    The cone-range gate ensures the command is C^1-smooth; together with the
    rate bound this makes "feasible" == smooth-in-range AND rate-within-limits.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    order = _validate_order(coeffs.size)
    half = 1 + 2 * order
    a0 = float(coeffs[0])
    cone_amp = 0.0  # worst-case |sum of cone harmonics|
    for k in range(1, order + 1):
        cone_amp += math.hypot(float(coeffs[k]), float(coeffs[order + k]))
    if a0 - cone_amp < 0.0 or a0 + cone_amp > 0.5 * math.pi:
        raise ValueError(
            f"cone alpha(tau) can leave [0, pi/2] (a0={a0:.4f} +/- amp={cone_amp:.4f}) "
            "-> the clamp would kink n_des; tighten the cone coefficients"
        )

    b_omega, b_alpha = cruise_command_slew_bounds(
        coeffs, T_s, omega_frame_max_rad_s, omega_frame_accel_max_rad_s2
    )
    if b_omega > limits.omega_max_rad_s:
        raise ValueError(
            f"commanded |omega| bound {b_omega:.3e} rad/s exceeds omega_max "
            f"{limits.omega_max_rad_s:.3e} rad/s"
        )
    if b_alpha > limits.alpha_max_rad_s2:
        raise ValueError(
            f"commanded |alpha| bound {b_alpha:.3e} rad/s^2 exceeds alpha_max "
            f"{limits.alpha_max_rad_s2:.3e} rad/s^2"
        )
    return b_omega, b_alpha


def feasible_coeff_boxes(
    order: int,
    T_s: float,
    limits,
    cone_bias_rad: float,
    omega_frame_max_rad_s: float = 0.0,
    omega_frame_accel_max_rad_s2: float = 0.0,
) -> tuple[float, float]:
    """Per-block harmonic box half-widths guaranteeing slew feasibility.

    Returns ``(cone_harmonic_max, clock_harmonic_max)`` such that for the given
    cone bias ``alpha0 = cone_bias_rad`` (with arbitrary clock bias), ANY command
    whose cone harmonics lie in ``[-cone_harmonic_max, cone_harmonic_max]`` and
    whose clock harmonics lie in ``[-clock_harmonic_max, clock_harmonic_max]``
    passes :func:`assert_cruise_slew_feasible`. This enforces feasibility via
    the Fourier representation: the
    optimizer's box bounds are set to these, so NO command it explores can
    leave the cone range OR exceed the slew limits, by construction.

    Two DIFFERENT binding constraints, so two boxes:

    * CONE harmonics are limited by the cone RANGE: the cone must stay in
      ``[0, pi/2]`` (else the clamp kinks ``n_des``). Worst cone excursion
      ``sum_k amp_k <= sqrt(2) * cone_harmonic_max * K`` must not exceed the
      margin ``m = min(alpha0, pi/2 - alpha0)`` -> ``cone_harmonic_max =
      m / (sqrt(2) K)``. (The cone rate contribution is negligible over a
      multi-month transit, so range is the binding cone constraint.)
    * CLOCK harmonics have NO range constraint (delta is an azimuth); they are
      limited by the slew RATE. The omega budget left for the clock is
      ``omega_max - Omega_frame - (cone rate)``; the alpha budget similarly.
      Solving the (conservative) bound for the clock box gives the rate limit.

    Feasibility scales with the period ``T_s`` (the clock box grows ~linearly in
    ``T_s`` for omega and ~quadratically for alpha). Returns ``inf`` for the
    clock box at ``order == 0`` (no harmonics).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    a0 = float(cone_bias_rad)
    margin = min(a0, 0.5 * math.pi - a0)
    if margin < 0.0:
        raise ValueError(f"cone_bias_rad {a0} is outside [0, pi/2]")
    if order == 0:
        return 0.0, float("inf")  # constant command: no harmonics, zero rate

    root2 = math.sqrt(2.0)
    sum_k = order * (order + 1) / 2.0
    sum_k2 = order * (order + 1) * (2 * order + 1) / 6.0

    # Cone box: range-limited.
    cone_box = margin / (root2 * order)

    # Cone rate contribution at the cone box (for the clock rate budget).
    w = 2.0 * math.pi / T_s
    cone_Sa = root2 * cone_box * sum_k          # worst sum_k k*amp (cone)
    cone_Qa = root2 * cone_box * sum_k2
    # Clock box from the omega bound: w(Sa+Sd) + Wf <= omega_max.
    head_omega = limits.omega_max_rad_s - float(omega_frame_max_rad_s) - w * cone_Sa
    # Clock box from the alpha bound: w^2(Qa+Qd) + B_omega^2 + Wfdot <= alpha_max,
    # using the worst B_omega = omega_max (decouples the two).
    head_alpha = (
        limits.alpha_max_rad_s2
        - limits.omega_max_rad_s ** 2
        - float(omega_frame_accel_max_rad_s2)
        - (w * w) * cone_Qa
    )
    if head_omega <= 0.0 or head_alpha <= 0.0:
        return cone_box, 0.0
    clock_box_omega = head_omega / (w * root2 * sum_k)
    clock_box_alpha = head_alpha / (w * w * root2 * sum_k2)
    return cone_box, min(clock_box_omega, clock_box_alpha)


def _validate_order(n_coeffs: int) -> int:
    """Infer the Fourier order K from the packed-coefficient length.

    Layout (see ``cruise_cone_clock``): alpha block (1 + 2K) then delta block
    (1 + 2K) -> total 2 + 4K. Returns K; raises if the length is inconsistent.
    """
    if (n_coeffs - 2) % 4 != 0 or n_coeffs < 2:
        raise ValueError(
            f"coeffs length {n_coeffs} is not 2 + 4K for an integer K>=0 "
            "(alpha: 1+2K, delta: 1+2K)"
        )
    return (n_coeffs - 2) // 4


def _eval_fourier(block: np.ndarray, tau: float, order: int) -> float:
    """Evaluate ``c0 + sum_k cc_k cos(2 pi k tau) + cs_k sin(2 pi k tau)``.

    ``block`` is ``[c0, cc_1..cc_K, cs_1..cs_K]`` (length 1 + 2K).
    """
    value = float(block[0])
    for k in range(1, order + 1):
        ang = 2.0 * math.pi * k * tau
        value += float(block[k]) * math.cos(ang)
        value += float(block[order + k]) * math.sin(ang)
    return value


def cruise_cone_clock(coeffs: np.ndarray, tau: float) -> tuple[float, float]:
    """Commanded ``(alpha, delta)`` (cone, clock; rad) at normalised time tau.

    ``alpha`` is clamped to ``[0, pi/2]``; ``delta`` is returned raw (azimuth).
    Exposed for tests / diagnostics so the recovered cone can be checked against
    the command without reconstructing the basis.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    order = _validate_order(coeffs.size)
    half = 1 + 2 * order
    alpha = _eval_fourier(coeffs[:half], tau, order)
    delta = _eval_fourier(coeffs[half:], tau, order)
    alpha = min(max(alpha, 0.0), 0.5 * math.pi)
    return alpha, delta


def make_cruise_command_steerer(
    coeffs: np.ndarray,
    et0: float,
    T_s: float,
    ref_normal: np.ndarray,
):
    """Build a time-Fourier cone/clock ``steering_fn`` for ``propagate_escape``.

    Parameters
    ----------
    coeffs
        Packed Fourier coefficients, length ``2 + 4K`` for order ``K``:
        ``[a0, ac_1..ac_K, as_1..as_K, d0, dc_1..dc_K, ds_1..ds_K]`` (cone
        block then clock block). ``K=1`` -> 6 params, ``K=2`` -> 10.
    et0
        Mission start ET (tau = 0).
    T_s
        Transit duration (s); ``tau = (et - et0) / T_s`` spans ``[0, 1]``.
    ref_normal
        FIXED clock reference normal (J2000 unit), the heliocentric orbit
        normal at departure (``heliocentric_orbit_normal(z0)``). Need not be
        pre-normalised; it is normalised here.

    Returns
    -------
    callable
        ``steering_fn(r, v, s_hat, p_eff, sail, current_n, *, et) -> n_des``
        (J2000 unit). The keyword-only ``et`` lets ``propagate_escape`` detect
        and supply the absolute epoch.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    order = _validate_order(coeffs.size)
    half = 1 + 2 * order
    alpha_block = coeffs[:half].copy()
    delta_block = coeffs[half:].copy()

    ref = np.asarray(ref_normal, dtype=float)
    ref_mag = float(np.linalg.norm(ref))
    if ref_mag < _EA_DEGENERATE_TOL:
        raise ValueError("ref_normal is the zero vector")
    ref = ref / ref_mag

    if not (T_s > 0.0):
        raise ValueError(f"T_s must be > 0, got {T_s}")

    def _clock_basis(s_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e_A = ref - float(np.dot(ref, s_hat)) * s_hat
        mag = float(np.linalg.norm(e_A))
        if mag < _EA_DEGENERATE_TOL:
            # ref ~parallel to the Sun line: pick a deterministic perpendicular.
            trial = np.array([1.0, 0.0, 0.0])
            if abs(s_hat[0]) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            e_A = trial - float(np.dot(trial, s_hat)) * s_hat
            mag = float(np.linalg.norm(e_A))
        e_A = e_A / mag
        e_B = np.cross(s_hat, e_A)
        return e_A, e_B

    def steering_fn(r, v, s_hat, p_eff, sail, current_n, *, et):
        s_hat = np.asarray(s_hat, dtype=float)
        tau = (et - et0) / T_s
        alpha = _eval_fourier(alpha_block, tau, order)
        alpha = min(max(alpha, 0.0), 0.5 * math.pi)
        delta = _eval_fourier(delta_block, tau, order)
        e_A, e_B = _clock_basis(s_hat)
        offset = math.cos(delta) * e_A + math.sin(delta) * e_B
        n_des = math.cos(alpha) * s_hat + math.sin(alpha) * offset
        return n_des

    return steering_fn
