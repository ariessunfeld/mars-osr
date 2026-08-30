"""Piecewise attitude schedules for sail-delivery missions.

Given a predicted trajectory and delivery windows, this module composes

    cruise -> [slew_in, track, slew_out, cruise] * N

and iterates the trajectory/window pair to a self-consistent schedule.  Cruise
and bisector segments remain state-dependent.  Each intervening projected
quintic-Hermite slew matches the endpoint direction and angular rate of those
moving laws.  Optional angular-rate and angular-acceleration limits determine
the slew duration; windows that cannot accommodate their slews are reported
and dropped rather than clipped.

The main entry points are :func:`build_delivery_schedule`, which builds one
schedule from a predicted trajectory, and :func:`refine_delivery_schedule`,
which performs the damped trajectory/window fixed-point iteration.  The
returned attitude callable can be passed directly to
``dynamics.propagate(sail_normal=...)``.

Fixed-duration and alpha-only configurations are also supported. Both limits
can be enforced simultaneously.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import spiceypy as spice
from numpy.polynomial import polynomial as poly
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from reflectors.attitude import (
    AttitudeCallable,
    piecewise,
    smooth_slew,
    smooth_slew_hermite,
)
from reflectors.ephemeris import frame_rotation, sun_state_j2000
from reflectors.surface import surface_point_position
from reflectors.visibility import (
    DeliveryWindow,
    WindowContinuationError,
    bisector_normal,
    bisector_pointing,
    continue_delivery_windows_multi,
    find_delivery_windows_multi,
    trajectory_interpolant,
)


logger = logging.getLogger(__name__)


MARS_NAIF_ID = 499
SUN_NAIF_ID = 10


@dataclass(frozen=True)
class ScheduleMetadata:
    """Diagnostics from ``build_delivery_schedule``.

    Fields
    ------
    n_windows_kept
        Count of input windows that survived the drop filter and have
        track + slew segments in the returned schedule.
    n_windows_dropped
        Count of input windows rejected by the drop filter.
    dropped_window_reasons
        Tuple of ``(original_window_index, reason_string)`` for each
        dropped window, in input order.
    segment_boundaries_et
        Absolute ET of every segment boundary, including the schedule
        start and end. Length = (n_segments + 1). Useful for tests and
        for a caller who wants to sample just inside each segment.
    """

    n_windows_kept: int
    n_windows_dropped: int
    dropped_window_reasons: Tuple[Tuple[int, str], ...]
    segment_boundaries_et: Tuple[float, ...]


# ---------------------------------------------------------------------------
# Closed-form slew-duration sizing
# ---------------------------------------------------------------------------

# Number of tau-grid samples used to compute the Hermite peak-alpha upper
# bound. 400 is ~2x the projection-guard density so the extrema of the
# R^3 c''(tau) polynomial (degree 3) are resolved cleanly.
_HERMITE_ALPHA_BOUND_SAMPLES = 400


def _hermite_peak_alpha_upper_bound(
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    omega_0: np.ndarray,
    omega_f: np.ndarray,
    T: float,
) -> float:
    """Upper bound on |alpha(t)| over the Hermite slew duration T.

    Derivation. For ``n_hat = c / |c|``,

        n_ddot = c_ddot / |c| - 2 c_dot (c.c_dot) / |c|^3
                 - c (c.c_ddot + |c_dot|^2) / |c|^3
                 + 3 c (c.c_dot)^2 / |c|^5

    so

        alpha = n_hat x n_ddot
              = (c x c_ddot) / |c|^2 - 2 (c x c_dot)(c.c_dot) / |c|^4

    (the last two terms drop because c x c = 0). The triangle inequality
    plus |a x b| <= |a||b| gives

        |alpha| <= |c_ddot| / |c| + 2 |c_dot|^2 / |c|^2

    evaluated at each tau. ``|c|`` is bounded below by the projection
    guard (min|c| >= _HERMITE_MIN_MAG >= 0.3 for any constructable
    Hermite), so this bound is finite and typically conservative by
    a factor of ~1-2x vs the true ``|alpha|``.

    Parameters
    ----------
    n_hat_0, n_hat_f
        Pre-normalised endpoint sail-normal unit vectors, shape (3,).
    omega_0, omega_f
        Endpoint angular velocity vectors, shape (3,), rad/s in J2000.
        Pass zeros for static endpoints.
    T
        Candidate slew duration, seconds. Must be > 0.

    Returns
    -------
    float
        An upper bound on ``max_t |alpha(t)|`` over ``t in [t0, t0+T]``,
        in rad/s^2.
    """
    if T <= 0.0:
        raise ValueError(f"_hermite_peak_alpha_upper_bound: T must be > 0, got {T}")

    # Endpoint R^3 tangents (same formula smooth_slew_hermite uses).
    v_0 = T * np.cross(omega_0, n_hat_0)
    v_f = T * np.cross(omega_f, n_hat_f)
    delta = n_hat_f - n_hat_0

    # Hermite coefficients (p_2 = 0 identically).
    p_0 = n_hat_0
    p_1 = v_0
    p_3 = 10.0 * delta - 6.0 * v_0 - 4.0 * v_f
    p_4 = -15.0 * delta + 8.0 * v_0 + 7.0 * v_f
    p_5 = 6.0 * delta - 3.0 * v_0 - 3.0 * v_f

    tau = np.linspace(0.0, 1.0, _HERMITE_ALPHA_BOUND_SAMPLES)
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    # c(tau), c'(tau) = dc/dtau, c''(tau) = d^2 c / dtau^2.
    c = (
        p_0[np.newaxis, :]
        + np.outer(tau, p_1)
        + np.outer(tau3, p_3)
        + np.outer(tau4, p_4)
        + np.outer(tau5, p_5)
    )
    c_dot_tau = (
        np.outer(np.ones_like(tau), p_1)
        + 3.0 * np.outer(tau2, p_3)
        + 4.0 * np.outer(tau3, p_4)
        + 5.0 * np.outer(tau4, p_5)
    )
    c_ddot_tau = (
        6.0 * np.outer(tau, p_3)
        + 12.0 * np.outer(tau2, p_4)
        + 20.0 * np.outer(tau3, p_5)
    )

    c_mag = np.linalg.norm(c, axis=1)
    # Convert d/dtau to d/dt: c_dot(t) = c_dot(tau) / T, c_ddot(t) = c_ddot(tau) / T^2.
    c_dot_mag = np.linalg.norm(c_dot_tau, axis=1) / T
    c_ddot_mag = np.linalg.norm(c_ddot_tau, axis=1) / (T * T)

    # Guard against numerical underflow in |c| if the caller passes a
    # T that the projection guard would reject at construction (e.g.
    # during Brent's iteration probing). Clip |c| at a tiny positive
    # floor to avoid infinities; the returned alpha_upper will then be
    # huge and Brent's will bracket away from it.
    c_mag_safe = np.maximum(c_mag, 1e-12)

    alpha_upper_per_tau = c_ddot_mag / c_mag_safe + 2.0 * c_dot_mag * c_dot_mag / (
        c_mag_safe * c_mag_safe
    )
    return float(np.max(alpha_upper_per_tau))


def _hermite_polynomial_coefficients(
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    omega_0: np.ndarray,
    omega_f: np.ndarray,
    T: float,
) -> np.ndarray:
    """Return the projected-Hermite R^3 polynomial coefficients.

    Rows are Cartesian components and columns are ascending powers of
    ``tau``.  This is the same polynomial constructed by
    :func:`reflectors.attitude.smooth_slew_hermite`; keeping the compact
    coefficient form here lets schedule-time constraint checks operate on the
    continuous curve without introspecting the returned closure.
    """
    v_0 = float(T) * np.cross(omega_0, n_hat_0)
    v_f = float(T) * np.cross(omega_f, n_hat_f)
    delta = n_hat_f - n_hat_0
    coefficients = np.zeros((3, 6), dtype=float)
    coefficients[:, 0] = n_hat_0
    coefficients[:, 1] = v_0
    coefficients[:, 3] = 10.0 * delta - 6.0 * v_0 - 4.0 * v_f
    coefficients[:, 4] = -15.0 * delta + 8.0 * v_0 + 7.0 * v_f
    coefficients[:, 5] = 6.0 * delta - 3.0 * v_0 - 3.0 * v_f
    return coefficients


def _hermite_omega_magnitude_at_tau(
    coefficients: np.ndarray,
    T: float,
    tau: float,
) -> float:
    """Evaluate exact ``|omega|`` of a projected Hermite at one ``tau``."""
    c = np.array(
        [poly.polyval(float(tau), axis) for axis in coefficients],
        dtype=float,
    )
    dc_dtau = np.array(
        [
            poly.polyval(float(tau), poly.polyder(axis))
            for axis in coefficients
        ],
        dtype=float,
    )
    c_norm_sq = float(np.dot(c, c))
    if c_norm_sq <= 1.0e-24:
        return math.inf
    # n=c/|c| gives omega=n x n_dot=(c x dc/dt)/|c|^2 exactly.
    return float(
        np.linalg.norm(np.cross(c, dc_dtau)) / (float(T) * c_norm_sq)
    )


def _hermite_peak_omega(
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    omega_0: np.ndarray,
    omega_f: np.ndarray,
    T: float,
) -> float:
    """Continuous global peak ``|omega|`` of a projected Hermite slew.

    For ``n=c/|c|`` and dimensionless time ``tau``,

    ``|omega|^2 = |c x c'|^2 / (T^2 |c|^4)``.

    Both numerator and denominator are polynomials.  Interior extrema are
    therefore roots of ``N' D - N D'``. Every real root in
    ``[0, 1]`` plus both endpoints is evaluated, avoiding any dependence on
    propagation or audit timestep. Roots are obtained in float64, while the
    duration safety factor supplies margin against root-conditioning and endpoint-prediction
    error.
    """
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"_hermite_peak_omega: T must be > 0, got {T}")
    coefficients = _hermite_polynomial_coefficients(
        np.asarray(n_hat_0, dtype=float),
        np.asarray(n_hat_f, dtype=float),
        np.asarray(omega_0, dtype=float),
        np.asarray(omega_f, dtype=float),
        float(T),
    )
    derivatives = [poly.polyder(axis) for axis in coefficients]
    cross_polynomials = [
        poly.polysub(
            poly.polymul(
                coefficients[(axis + 1) % 3],
                derivatives[(axis + 2) % 3],
            ),
            poly.polymul(
                coefficients[(axis + 2) % 3],
                derivatives[(axis + 1) % 3],
            ),
        )
        for axis in range(3)
    ]
    numerator = np.zeros(1, dtype=float)
    for axis in cross_polynomials:
        numerator = poly.polyadd(numerator, poly.polymul(axis, axis))
    c_norm_sq = np.zeros(1, dtype=float)
    for axis in coefficients:
        c_norm_sq = poly.polyadd(c_norm_sq, poly.polymul(axis, axis))
    denominator = poly.polymul(c_norm_sq, c_norm_sq)
    stationarity = poly.polysub(
        poly.polymul(poly.polyder(numerator), denominator),
        poly.polymul(numerator, poly.polyder(denominator)),
    )

    candidates = [0.0, 1.0]
    coefficient_scale = float(np.max(np.abs(stationarity)))
    if coefficient_scale > 0.0 and math.isfinite(coefficient_scale):
        # Drop cancellation-scale leading coefficients before constructing the
        # companion matrix.  Exact quintic identities commonly cancel the
        # formal highest-order terms.
        while (
            stationarity.size > 1
            and abs(float(stationarity[-1]))
            <= 1.0e-13 * coefficient_scale
        ):
            stationarity = stationarity[:-1]
        roots = poly.polyroots(stationarity)
        for root in roots:
            real = float(root.real)
            imag = abs(float(root.imag))
            if imag <= 1.0e-7 * max(1.0, abs(real)):
                if -1.0e-10 <= real <= 1.0 + 1.0e-10:
                    candidates.append(min(1.0, max(0.0, real)))

    return max(
        _hermite_omega_magnitude_at_tau(coefficients, float(T), tau)
        for tau in candidates
    )


def slew_duration_for_alpha_max(
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    *,
    omega_0_rad_s: Optional[np.ndarray] = None,
    omega_f_rad_s: Optional[np.ndarray] = None,
    alpha_max_rad_s2: float,
    safety_factor: float = 1.1,
    T_min_s: float = 1.0,
    T_max_s: float = 3600.0,
) -> float:
    """Closed-form slew duration such that peak |alpha| <= alpha_max.

    Sizes a ``smooth_slew_hermite`` so its peak angular acceleration
    bound sits at or below ``alpha_max_rad_s2``. For static endpoints
    (omega=0) this yields the standard quintic scaling
    ``T = sqrt(theta_total * (10/sqrt(3)) / alpha_max)`` modulo a small
    correction from the R^3-chord vs great-circle-arc distinction. For
    dynamic endpoints (omega != 0) the sizing iterates via Brent's on
    the analytic upper-bound polynomial.

    Parameters
    ----------
    n_hat_0, n_hat_f
        Endpoint sail-normal vectors (re-normalised internally).
    omega_0_rad_s, omega_f_rad_s
        Endpoint angular velocities (3-vectors in J2000), or ``None``
        (= zero, static endpoint).
    alpha_max_rad_s2
        Path-constraint budget on |alpha|, rad/s^2. Must be > 0.
    safety_factor
        Multiplicative margin applied to the computed T. Default 1.1
        (10% margin). Exists because the upper bound is conservative
        but the caller may want extra head room for integrator noise.
    T_min_s, T_max_s
        Clamp range on the returned duration.

    Returns
    -------
    float
        Minimum slew duration (seconds), clamped to ``[T_min_s, T_max_s]``
        and multiplied by ``safety_factor``.

    Raises
    ------
    ValueError
        If ``alpha_max_rad_s2`` is not strictly positive, if the
        handoff cannot achieve ``alpha_max`` within ``T_max_s``, or
        if the endpoints are degenerate (zero vectors, antipodal).

    Notes
    -----
    Uses ``_hermite_peak_alpha_upper_bound`` + ``scipy.optimize.brentq``.
    Typical cost: ~10-20 evaluations of the 400-sample polynomial
    sweep. Intended for schedule-build time, not RHS hot loop.
    """
    if not math.isfinite(alpha_max_rad_s2) or alpha_max_rad_s2 <= 0.0:
        raise ValueError(
            f"alpha_max_rad_s2 must be positive and finite, got {alpha_max_rad_s2}"
        )
    if T_min_s <= 0.0 or T_max_s <= T_min_s:
        raise ValueError(
            f"T range must be 0 < T_min_s < T_max_s, got "
            f"T_min_s={T_min_s}, T_max_s={T_max_s}"
        )

    n0 = np.asarray(n_hat_0, dtype=float)
    nf = np.asarray(n_hat_f, dtype=float)
    norm0 = float(np.linalg.norm(n0))
    normf = float(np.linalg.norm(nf))
    if norm0 == 0.0 or normf == 0.0:
        raise ValueError("slew_duration_for_alpha_max: endpoint n_hat is zero")
    n0 = n0 / norm0
    nf = nf / normf

    cos_total = max(-1.0, min(1.0, float(np.dot(n0, nf))))
    theta_total = math.acos(cos_total)

    w0 = np.asarray(
        omega_0_rad_s if omega_0_rad_s is not None else np.zeros(3), dtype=float
    )
    wf = np.asarray(
        omega_f_rad_s if omega_f_rad_s is not None else np.zeros(3), dtype=float
    )

    # Initial estimate from the smooth_slew static-endpoint formula. The
    # R^3 Hermite differs from the great-circle parameterisation but the
    # theta-driven 1/T^2 scaling is still the leading term.
    if theta_total < 1e-12:
        # Near-identical endpoints: trivial slew, just clamp T_min.
        return max(T_min_s, min(T_max_s, safety_factor * T_min_s))

    T_static = math.sqrt(theta_total * 10.0 / math.sqrt(3.0) / alpha_max_rad_s2)

    def f(T: float) -> float:
        return (
            _hermite_peak_alpha_upper_bound(n0, nf, w0, wf, T)
            - alpha_max_rad_s2
        )

    # Search for a bracket [T_lo, T_hi] with f(T_lo) > 0, f(T_hi) < 0.
    T_lo = max(T_min_s, 0.25 * T_static)
    T_hi = min(T_max_s, 4.0 * T_static)
    # Expand T_hi until the bound is satisfied (if possible).
    for _ in range(20):
        if f(T_hi) < 0.0:
            break
        if T_hi >= T_max_s:
            # Infeasible within T_max_s: raise explicitly.
            raise ValueError(
                f"slew_duration_for_alpha_max: alpha_max={alpha_max_rad_s2:.3e} "
                f"unachievable within T_max_s={T_max_s}s for this handoff "
                f"(theta={math.degrees(theta_total):.2f} deg, "
                f"|omega_0|={float(np.linalg.norm(w0)):.3e}, "
                f"|omega_f|={float(np.linalg.norm(wf)):.3e} rad/s)."
            )
        T_hi = min(T_max_s, 2.0 * T_hi)
    # Contract T_lo until the bound is violated (too-short slew).
    for _ in range(30):
        if f(T_lo) > 0.0:
            break
        if T_lo <= T_min_s:
            # Bound already satisfied at T_min_s: return T_min.
            return max(T_min_s, min(T_max_s, safety_factor * T_min_s))
        T_lo = max(T_min_s, 0.5 * T_lo)

    T_solution = brentq(f, T_lo, T_hi, xtol=0.05, rtol=1e-4)
    return max(T_min_s, min(T_max_s, safety_factor * T_solution))


def slew_duration_for_limits(
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    *,
    omega_0_rad_s: Optional[np.ndarray] = None,
    omega_f_rad_s: Optional[np.ndarray] = None,
    alpha_max_rad_s2: Optional[float] = None,
    omega_max_rad_s: Optional[float] = None,
    safety_factor: float = 1.1,
    T_min_s: float = 1.0,
    T_max_s: float = 3600.0,
) -> float:
    """Size a projected-Hermite slew against alpha and omega limits.

    At least one limit must be supplied.  When only ``alpha_max_rad_s2`` is
    supplied, this delegates to :func:`slew_duration_for_alpha_max`. With an
    omega limit, the residual
    is the larger of the alpha utilization and the *continuous* omega
    utilization from :func:`_hermite_peak_omega`.

    A nonzero endpoint angular rate is independent of duration.  If either
    endpoint already exceeds ``omega_max_rad_s``, the handoff is infeasible
    and this function raises instead of concealing the violation by extending
    the slew.
    """
    if alpha_max_rad_s2 is None and omega_max_rad_s is None:
        raise ValueError(
            "slew_duration_for_limits requires alpha_max_rad_s2 and/or "
            "omega_max_rad_s"
        )
    if alpha_max_rad_s2 is not None and (
        not math.isfinite(alpha_max_rad_s2) or alpha_max_rad_s2 <= 0.0
    ):
        raise ValueError(
            "alpha_max_rad_s2 must be positive and finite, got "
            f"{alpha_max_rad_s2}"
        )
    if omega_max_rad_s is not None and (
        not math.isfinite(omega_max_rad_s) or omega_max_rad_s <= 0.0
    ):
        raise ValueError(
            f"omega_max_rad_s must be positive and finite, got {omega_max_rad_s}"
        )
    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError(
            f"safety_factor must be finite and >= 1, got {safety_factor}"
        )
    if T_min_s <= 0.0 or T_max_s <= T_min_s:
        raise ValueError(
            f"T range must be 0 < T_min_s < T_max_s, got "
            f"T_min_s={T_min_s}, T_max_s={T_max_s}"
        )
    if omega_max_rad_s is None:
        assert alpha_max_rad_s2 is not None
        return slew_duration_for_alpha_max(
            n_hat_0,
            n_hat_f,
            omega_0_rad_s=omega_0_rad_s,
            omega_f_rad_s=omega_f_rad_s,
            alpha_max_rad_s2=alpha_max_rad_s2,
            safety_factor=safety_factor,
            T_min_s=T_min_s,
            T_max_s=T_max_s,
        )

    n0 = np.asarray(n_hat_0, dtype=float)
    nf = np.asarray(n_hat_f, dtype=float)
    if n0.shape != (3,) or nf.shape != (3,):
        raise ValueError(
            "slew_duration_for_limits endpoint n_hat vectors must have "
            f"shape (3,), got {n0.shape} and {nf.shape}"
        )
    norm0 = float(np.linalg.norm(n0))
    normf = float(np.linalg.norm(nf))
    if norm0 == 0.0 or normf == 0.0:
        raise ValueError("slew_duration_for_limits: endpoint n_hat is zero")
    n0 = n0 / norm0
    nf = nf / normf
    w0 = np.asarray(
        omega_0_rad_s if omega_0_rad_s is not None else np.zeros(3),
        dtype=float,
    )
    wf = np.asarray(
        omega_f_rad_s if omega_f_rad_s is not None else np.zeros(3),
        dtype=float,
    )
    if w0.shape != (3,) or wf.shape != (3,):
        raise ValueError(
            "slew_duration_for_limits endpoint omega vectors must have "
            f"shape (3,), got {w0.shape} and {wf.shape}"
        )
    if np.any(~np.isfinite(w0)) or np.any(~np.isfinite(wf)):
        raise ValueError("slew_duration_for_limits endpoint omega must be finite")

    # Only the component perpendicular to n changes n.  It is also exactly
    # the Hermite path's endpoint angular-rate magnitude.
    omega_endpoint_0 = float(np.linalg.norm(w0 - np.dot(w0, n0) * n0))
    omega_endpoint_f = float(np.linalg.norm(wf - np.dot(wf, nf) * nf))
    endpoint_peak = max(omega_endpoint_0, omega_endpoint_f)
    if endpoint_peak > omega_max_rad_s * (1.0 + 1.0e-12):
        raise ValueError(
            "slew_duration_for_limits: omega_max_rad_s is unachievable "
            "because an endpoint rate already exceeds it "
            f"(endpoint peak={endpoint_peak:.6e}, "
            f"omega_max={omega_max_rad_s:.6e} rad/s)"
        )

    cos_total = max(-1.0, min(1.0, float(np.dot(n0, nf))))
    theta_total = math.acos(cos_total)
    T_estimates = [float(T_min_s)]
    if alpha_max_rad_s2 is not None and theta_total > 0.0:
        T_estimates.append(
            math.sqrt(
                theta_total * 10.0 / math.sqrt(3.0) / alpha_max_rad_s2
            )
        )
    if theta_total > 0.0:
        # Exact rest-to-rest great-circle quintic peak: theta*(15/8)/T.
        T_estimates.append(theta_total * 15.0 / (8.0 * omega_max_rad_s))
    T_estimate = min(float(T_max_s), max(T_estimates))

    def utilization(T: float) -> float:
        return _hermite_constraint_utilization(
            n0,
            nf,
            w0,
            wf,
            T,
            alpha_max_rad_s2=alpha_max_rad_s2,
            omega_max_rad_s=omega_max_rad_s,
        )

    def residual(T: float) -> float:
        return utilization(T) - 1.0

    residual_at_min = residual(float(T_min_s))
    if residual_at_min <= 0.0:
        T_solution = float(T_min_s)
    else:
        T_lo = float(T_min_s)
        T_hi = max(float(T_min_s), T_estimate)
        if T_hi == T_lo:
            T_hi = min(float(T_max_s), 2.0 * T_lo)
        for _ in range(30):
            if residual(T_hi) <= 0.0:
                break
            if T_hi >= T_max_s:
                raise ValueError(
                    "slew_duration_for_limits: joint alpha/omega limits "
                    f"unachievable within T_max_s={T_max_s}s for this "
                    f"handoff (theta={math.degrees(theta_total):.2f} deg, "
                    f"|omega_0|={float(np.linalg.norm(w0)):.3e}, "
                    f"|omega_f|={float(np.linalg.norm(wf)):.3e} rad/s)"
                )
            T_hi = min(float(T_max_s), 2.0 * T_hi)
        T_solution = brentq(
            residual,
            T_lo,
            T_hi,
            xtol=0.01,
            rtol=1.0e-6,
        )

    T_returned = min(float(T_max_s), safety_factor * T_solution)
    # Dynamic endpoint tangents scale with T, so utilization is not assumed
    # globally monotone.  Verify the actual returned duration and, if the
    # safety-factor move crossed a non-monotone pocket, expand until a feasible
    # point is found; otherwise raise at T_max.
    for _ in range(20):
        if utilization(T_returned) <= 1.0 + 1.0e-12:
            return max(float(T_min_s), T_returned)
        if T_returned >= T_max_s:
            break
        T_returned = min(float(T_max_s), 1.25 * T_returned)
    raise ValueError(
        "slew_duration_for_limits: safety-margined duration does not satisfy "
        f"the requested limits within T_max_s={T_max_s}s"
    )


def _hermite_constraint_utilization(
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    omega_0: np.ndarray,
    omega_f: np.ndarray,
    T_s: float,
    *,
    alpha_max_rad_s2: Optional[float],
    omega_max_rad_s: Optional[float],
) -> float:
    """Return the largest continuous-path Hermite constraint utilization."""
    terms: list[float] = []
    if omega_max_rad_s is not None:
        terms.append(
            _hermite_peak_omega(
                n_hat_0, n_hat_f, omega_0, omega_f, T_s,
            )
            / omega_max_rad_s
        )
    if alpha_max_rad_s2 is not None:
        terms.append(
            _hermite_peak_alpha_upper_bound(
                n_hat_0, n_hat_f, omega_0, omega_f, T_s,
            )
            / alpha_max_rad_s2
        )
    if not terms:
        raise ValueError(
            "_hermite_constraint_utilization requires an alpha and/or "
            "omega limit"
        )
    return max(terms)


def _moving_endpoint_slew_duration(
    endpoint_fn: Callable[
        [float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    *,
    alpha_max_rad_s2: Optional[float],
    omega_max_rad_s: Optional[float],
    safety_factor: float = 1.1,
    T_min_s: float = 1.0,
    T_max_s: float = 3600.0,
    root_tol_s: float = 0.05,
    context: str = "moving-endpoint slew",
) -> Tuple[
    float,
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    int,
]:
    """Size a slew while evaluating its moving endpoint at each duration.

    A raw fixed-point update, ``T <- required_duration(endpoint(T))``, can
    converge arbitrarily slowly despite the existence of a feasible duration.
    This routine instead solves the physical scalar inequality directly:
    the continuous Hermite alpha/omega utilization formed from
    ``endpoint_fn(T)`` must not exceed one.  It brackets the first sampled
    feasible crossing, applies the duration safety factor, and re-evaluates
    the *final* moving-endpoint polynomial before returning.

    The common path uses geometrically expanding samples.  If no crossing is
    found, a 30-s fallback grid guards against a feasible non-monotone pocket
    before the routine reports that it could not bracket a solution.
    """
    if alpha_max_rad_s2 is None and omega_max_rad_s is None:
        raise ValueError(f"{context}: at least one slew limit is required")
    if alpha_max_rad_s2 is not None and (
        not math.isfinite(alpha_max_rad_s2) or alpha_max_rad_s2 <= 0.0
    ):
        raise ValueError(
            f"{context}: alpha_max_rad_s2 must be positive and finite"
        )
    if omega_max_rad_s is not None and (
        not math.isfinite(omega_max_rad_s) or omega_max_rad_s <= 0.0
    ):
        raise ValueError(
            f"{context}: omega_max_rad_s must be positive and finite"
        )
    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError(f"{context}: safety_factor must be finite and >= 1")
    if (
        not math.isfinite(T_min_s)
        or not math.isfinite(T_max_s)
        or T_min_s <= 0.0
        or T_max_s < T_min_s
    ):
        raise ValueError(
            f"{context}: require 0 < T_min_s <= T_max_s, got "
            f"{T_min_s}, {T_max_s}"
        )
    if not math.isfinite(root_tol_s) or root_tol_s <= 0.0:
        raise ValueError(f"{context}: root_tol_s must be positive and finite")

    cache: dict[
        float,
        Tuple[
            float,
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
    ] = {}

    def evaluate(
        T_s: float,
    ) -> Tuple[
        float,
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        key = float(T_s)
        if key in cache:
            return cache[key]
        raw_endpoints = endpoint_fn(key)
        if len(raw_endpoints) != 4:
            raise ValueError(
                f"{context}: endpoint_fn must return (n0, nf, omega0, omegaf)"
            )
        n0 = np.asarray(raw_endpoints[0], dtype=float)
        nf = np.asarray(raw_endpoints[1], dtype=float)
        w0 = np.asarray(raw_endpoints[2], dtype=float)
        wf = np.asarray(raw_endpoints[3], dtype=float)
        if n0.shape != (3,) or nf.shape != (3,):
            raise ValueError(f"{context}: endpoint normals must have shape (3,)")
        if w0.shape != (3,) or wf.shape != (3,):
            raise ValueError(f"{context}: endpoint rates must have shape (3,)")
        if (
            np.any(~np.isfinite(n0))
            or np.any(~np.isfinite(nf))
            or np.any(~np.isfinite(w0))
            or np.any(~np.isfinite(wf))
        ):
            raise ValueError(f"{context}: endpoint data must be finite")
        norm0 = float(np.linalg.norm(n0))
        normf = float(np.linalg.norm(nf))
        if norm0 == 0.0 or normf == 0.0:
            raise ValueError(f"{context}: endpoint normal is zero")
        endpoints = (n0 / norm0, nf / normf, w0, wf)
        utilization = _hermite_constraint_utilization(
            *endpoints,
            key,
            alpha_max_rad_s2=alpha_max_rad_s2,
            omega_max_rad_s=omega_max_rad_s,
        )
        cache[key] = (utilization - 1.0, endpoints)
        return cache[key]

    def residual(T_s: float) -> float:
        return evaluate(T_s)[0]

    def bracket_feasible_crossing(
        start_s: float,
    ) -> Optional[Tuple[float, float]]:
        """Find a positive-to-nonpositive utilization crossing."""
        T_lo = float(start_s)
        r_lo = residual(T_lo)
        if r_lo <= 0.0:
            return T_lo, T_lo

        # Fast common path: endpoint motion is slow on the slew timescale,
        # while duration relief is strong.  Expansion is capped by T_max.
        while T_lo < T_max_s:
            T_hi = min(float(T_max_s), max(T_lo + 15.0, 1.2 * T_lo))
            r_hi = residual(T_hi)
            if r_hi <= 0.0:
                return T_lo, T_hi
            if T_hi == T_max_s:
                break
            T_lo, r_lo = T_hi, r_hi

        # A moving endpoint makes global monotonicity impossible to assume.
        # Search the whole remaining domain on a modest absolute-time grid
        # before rejecting a candidate merely because the geometric samples
        # stepped over a feasible pocket.
        span_s = float(T_max_s - start_s)
        n_intervals = max(1, int(math.ceil(span_s / 30.0)))
        T_previous = float(start_s)
        r_previous = residual(T_previous)
        for T_sample in np.linspace(start_s, T_max_s, n_intervals + 1)[1:]:
            T_current = float(T_sample)
            r_current = residual(T_current)
            if r_current <= 0.0:
                return T_previous, T_current
            T_previous, r_previous = T_current, r_current
        return None

    T_base = float(T_min_s)
    for _ in range(4):
        bracket = bracket_feasible_crossing(T_base)
        if bracket is None:
            sampled_min_T, sampled_min = min(
                cache.items(), key=lambda item: item[1][0],
            )
            raise ValueError(
                f"{context}: could not bracket a duration satisfying the "
                f"moving-endpoint slew limits within [{T_base:.3f}, "
                f"{T_max_s:.3f}] s; lowest sampled utilization was "
                f"{sampled_min[0] + 1.0:.6f} at T={sampled_min_T:.3f} s"
            )
        T_lo, T_hi = bracket
        if T_lo == T_hi:
            T_root = T_lo
        else:
            T_root = float(
                brentq(
                    residual,
                    T_lo,
                    T_hi,
                    xtol=root_tol_s,
                    rtol=1.0e-10,
                )
            )
        T_candidate = min(float(T_max_s), safety_factor * T_root)
        candidate_residual, candidate_endpoints = evaluate(T_candidate)
        if candidate_residual <= 1.0e-12:
            return T_candidate, candidate_endpoints, len(cache)
        if T_candidate >= T_max_s:
            break
        # The safety-factor move crossed back into a non-monotone infeasible
        # region.  Seek the next feasible crossing rather than accepting an
        # unchecked polynomial.
        T_base = min(
            float(T_max_s),
            T_candidate + max(root_tol_s, 1.0e-6),
        )

    final_residual, _ = evaluate(float(T_max_s))
    raise ValueError(
        f"{context}: safety-margined moving-endpoint duration remains "
        f"infeasible at T_max_s={T_max_s:.3f} s "
        f"(utilization={final_residual + 1.0:.6f})"
    )


def _bisector_direction_at(
    r_sat_km: np.ndarray,
    et: float,
    target_lat_deg: float,
    target_lon_deg: float,
    observer_naif_id: int,
) -> np.ndarray:
    """Bisector unit vector at (r_sat, et), evaluated directly.

    Raises ``ValueError`` at geometrically degenerate geometries
    (sun and target antiparallel from the sail), matching
    ``visibility.bisector_normal`` semantics but surfacing the error
    with a context tag useful during schedule construction.
    """
    r_target = surface_point_position(
        float(target_lat_deg), float(target_lon_deg), float(et),
    )
    state_sun = sun_state_j2000(float(et), int(observer_naif_id))
    r_sun = np.asarray(state_sun[:3], dtype=float)
    n_hat, cos_alpha = bisector_normal(
        np.asarray(r_sat_km, dtype=float), r_target, r_sun
    )
    if cos_alpha == 0.0:
        raise ValueError(
            "bisector is degenerate at this geometry "
            f"(et={et:.3f}): sun and target near-antiparallel from sail."
        )
    return n_hat


def _measure_omega_via_predicted_trajectory(
    profile: AttitudeCallable,
    r_sat_predicted_fn: Callable[[float], np.ndarray],
    et: float,
    dt: float = 0.5,
) -> np.ndarray:
    """Central-difference omega(et) for a state-dependent profile.

    Uses the predicted trajectory callable to feed ``profile`` with
    consistent r_sat at each of the three stencil points. Matches the
    evaluation pattern of ``attitude.angular_rate`` but with the
    predicted (not actual) trajectory -- used during schedule build
    to sample endpoint omegas of cruise / bisector before the
    integrator has run.
    """
    r_minus = r_sat_predicted_fn(et - dt)
    r_now = r_sat_predicted_fn(et)
    r_plus = r_sat_predicted_fn(et + dt)
    n_minus = np.asarray(profile(r_minus, et - dt), dtype=float)
    n_now = np.asarray(profile(r_now, et), dtype=float)
    n_plus = np.asarray(profile(r_plus, et + dt), dtype=float)
    n_dot = (n_plus - n_minus) / (2.0 * dt)
    return np.cross(n_now, n_dot)


# ---------------------------------------------------------------------------
# Cruise-to-cruise slew
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CruiseSlewMetadata:
    """Diagnostics from ``cruise_to_cruise_slew``.

    Each field is logged at every reoptimization boundary,
    so a state-continuity or alpha-budget regression surfaces in the
    boundary CSV before the integration runs out of bounds.

    Fields
    ------
    t_start_et, t_end_et
        Slew start / end as absolute SPICE TDB seconds.
    T_slew_s
        Slew duration (seconds), = t_end_et - t_start_et, after sizing
        iterations and the slew-floor clamp.
    n_hat_0, n_hat_f
        Endpoint sail normals (unit-norm) in J2000.
    omega_0_rad_s, omega_f_rad_s
        Endpoint angular-velocity vectors (rad/s, J2000).
    theta_total_rad
        Total slew angle (acos(n_hat_0 . n_hat_f)).
    alpha_max_budget_rad_s2
        The alpha_max cap passed in.
    alpha_peak_upper_bound_rad_s2
        ``_hermite_peak_alpha_upper_bound`` evaluated at the final
        T_slew and chosen endpoints. Should be <= alpha_max_budget by
        construction (with the safety_factor margin).
    alpha_utilization_pct
        100 * alpha_peak_upper_bound / alpha_max_budget. Useful diagnostic;
        if this is consistently near 100%, the safety_factor is being
        consumed by integrator noise, and a higher safety_factor is
        warranted.
    sizing_iterations
        Number of endpoint/constraint evaluations used by omega-aware
        moving-endpoint sizing, or fixed-point iterations on the alpha-only
        path.
    predictor_state_at_slew_end_km_kmps
        Two-body Kepler-predicted state at t_end_et. Stored for cross-
        check against the actual integrator output; the difference is
        the predictor error.
    """

    t_start_et: float
    t_end_et: float
    T_slew_s: float
    n_hat_0: np.ndarray
    n_hat_f: np.ndarray
    omega_0_rad_s: np.ndarray
    omega_f_rad_s: np.ndarray
    theta_total_rad: float
    alpha_max_budget_rad_s2: float
    alpha_peak_upper_bound_rad_s2: float
    alpha_utilization_pct: float
    sizing_iterations: int
    predictor_state_at_slew_end_km_kmps: np.ndarray
    omega_max_budget_rad_s: Optional[float] = None
    omega_peak_rad_s: Optional[float] = None
    omega_utilization_pct: Optional[float] = None


def cruise_to_cruise_slew(
    cruise_old: AttitudeCallable,
    cruise_new: AttitudeCallable,
    state_b_km_kmps: np.ndarray,
    epoch_et_b: float,
    *,
    central_body_gm_km3_s2: float,
    alpha_max_rad_s2: float,
    omega_max_rad_s: Optional[float] = None,
    slew_floor_s: float = 60.0,
    safety_factor: float = 1.1,
    fd_dt_s: float = 0.5,
    n_size_iterations: int = 3,
    sizing_tol_s: float = 0.5,
    T_max_s: float = 3600.0,
    predicted_trajectory_fn: Optional[Callable[[float], np.ndarray]] = None,
) -> Tuple[AttitudeCallable, CruiseSlewMetadata]:
    """Compose a Hermite slew bridging cruise_old at t_b -> cruise_new at t_b + T_slew.

    Used at sol boundaries during periodic re-optimization.
    The sail's physical state (r, v) is continuous across the boundary
    by construction (handed off by the integrator); this helper bridges
    the *attitude* and *attitude rate* with a quintic-Hermite slew sized
    to respect the angular-acceleration budget and, when supplied, the
    angular-rate budget. After the slew, the sail follows ``cruise_new``.

    Predictor. The slew end-state ``state(t_b + T_slew)`` is needed to
    sample ``n_hat_f`` and ``omega_f`` under ``cruise_new`` before the
    actual integrator has run. This uses ``spice.prop2b`` (two-body Kepler)
    from ``state_b`` under ``central_body_gm_km3_s2``. Over a 5-min slew
    at Mars LMO the unmodelled perturbations (J_2, SRP, third-bodies)
    contribute O(meters) in r and O(mm/s) in v, giving n_hat-direction
    errors of O(1e-7 rad) and omega errors of O(1e-9 rad/s) -- both
    well below the radians(0.003) rad/s^2 alpha budget.

    Moving-endpoint sizing. The endpoint ``(n_hat_f, omega_f)`` depends on
    the predicted state at ``t_b + T_slew``. With an omega limit, the solver
    evaluates that endpoint as a function of duration and brackets a root of
    the final Hermite's continuous alpha/omega utilization directly. This
    avoids relying on convergence of the raw fixed-point map. The alpha-only
    path uses fixed-point iteration.

    Endpoint angular-velocity continuity. ``omega_0`` is sampled by
    central-difference of ``cruise_old`` around (predictor(t), t) at
    t = t_b; ``omega_f`` by central-difference of ``cruise_new`` around
    t = t_b + T_slew. Same primitive
    (``_measure_omega_via_predicted_trajectory``) and same FD horizon
    used inside ``build_delivery_schedule`` for cruise -> bisector
    transitions. Hermite then matches BOTH endpoint omegas exactly at
    the slew start/end (in addition to the endpoint n_hats), so a
    propagation that hits the slew at exactly t_b under cruise_old
    omega and exits at exactly t_b + T_slew under cruise_new omega has
    zero discontinuity in attitude or attitude-rate at either boundary.

    The known limitation is the slew->cruise_new handoff: the slew is
    constructed against the PREDICTED state at slew end, but the
    integrator advances under the actual dynamics during the slew (J_2,
    SRP, third-bodies all act). The mismatch between predicted and
    actual at slew end -- of order meters in r, mm/s in v -- translates
    to an attitude/omega mismatch at the slew->cruise_new handoff
    bounded by the perturbation magnitudes; verified < 1e-3 rad in
    n_hat and < 1e-5 rad/s in omega (small versus alpha_max * t_slew
    = 0.9 rad of swept angle / 1.5e-3 rad/s of omega change).

    Parameters
    ----------
    cruise_old, cruise_new
        ``AttitudeCallable``s: ``(r_sat_km, et) -> n_hat_j2000``. Both
        must accept the same J2000 state convention; both will be
        evaluated near the boundary.
    state_b_km_kmps
        Sail state at the boundary epoch, shape (6,), Mars-centred J2000.
    epoch_et_b
        Boundary epoch (SPICE TDB).
    central_body_gm_km3_s2
        Central-body GM for the two-body Kepler predictor (Mars: ~4.28e4).
    alpha_max_rad_s2
        Angular-acceleration budget; the sized slew satisfies
        ``alpha_peak_upper_bound <= alpha_max`` (with safety_factor
        margin).
    omega_max_rad_s
        Optional angular-rate budget.  When supplied, sizing evaluates the
        continuous Hermite rate peak from its polynomial stationary points and
        enforces both alpha and omega. ``None`` selects alpha-only sizing.
    slew_floor_s
        Hard floor on T_slew (seconds). The sized duration is clamped
        to this floor; useful when the alpha-max-derived T is shorter
        than is operationally desirable (sub-minute slews put thermal
        and spacecraft-attitude noise above the alpha-budget motion).
    safety_factor
        Multiplicative margin on the alpha-budget sizing (forwarded to
        ``slew_duration_for_alpha_max``).
    fd_dt_s
        Finite-difference horizon for omega measurement (seconds).
        Smaller is more accurate for slow-varying cruise omega; matches
        the ``_measure_omega_via_predicted_trajectory`` default.
    n_size_iterations
        Maximum fixed-point sizing passes on the alpha-only path.
        Omega-aware sizing instead uses a bracketed moving-endpoint solve.
    sizing_tol_s
        Alpha-only fixed-point convergence tolerance, or omega-aware scalar
        root tolerance, in seconds.
    T_max_s
        Hard upper bound on the sized T_slew. Forwarded to
        ``slew_duration_for_alpha_max``.
    predicted_trajectory_fn
        Optional callable ``et -> r_km_j2000`` (3-vector) overriding the
        default two-body Kepler predictor for sampling cruise endpoint
        n_hat / omega. When ``None`` (default), uses ``spice.prop2b``
        from ``state_b`` under ``central_body_gm_km3_s2`` -- accurate to
        meters in r over a 5-min slew but accumulates 3-7 km error under
        full Mars dynamics (J_2, SRP, third bodies). When the caller has
        already propagated the trajectory through the slew window and
        can supply r(et) (typically a CubicSpline of the propagator's
        result.state_km_kmps[:, :3] vs result.t_s + epoch_et), passing
        it here drives the slew->cruise handoff omega match from ~1e-4
        rad/s (Kepler-predicted) to spline-interp accuracy ~1e-8 rad/s
        (BELOW the bisector floor of 1.2e-7). Operationally this mirrors
        the post-loop rebuild in ``refine_delivery_schedule`` but for
        cruise-to-cruise sol boundaries; the metadata's
        ``predictor_state_at_slew_end_km_kmps`` field is still populated
        from Kepler for cross-reference (the actual handoff state is
        observable directly from the integrator's result).

    Returns
    -------
    slew_callable
        ``AttitudeCallable`` valid for ``et in [epoch_et_b, epoch_et_b
        + T_slew_s]``. Constructed from ``smooth_slew_hermite`` with
        the sized endpoints.
    metadata
        ``CruiseSlewMetadata`` with the sizing trace and endpoint
        diagnostics suitable for per-boundary logging.

    Raises
    ------
    ValueError
        If ``state_b_km_kmps`` is not a 6-vector; if
        ``alpha_max_rad_s2`` is not positive; if no feasible moving-endpoint
        duration can be bracketed within ``T_max_s``; if alpha-only
        sizing fails to converge within ``n_size_iterations``; or any
        underlying primitive raises (degenerate endpoints, antipodal n_hats,
        ...).

    Notes
    -----
    Construction occurs only when the schedule is rebuilt, never in the
    propagation RHS. Endpoint/constraint evaluations are cached within the
    omega-aware scalar solve, which is suitable for the optimizer's inner
    loop.
    """
    state_b = np.asarray(state_b_km_kmps, dtype=float)
    if state_b.shape != (6,):
        raise ValueError(
            f"state_b_km_kmps must be shape (6,), got {state_b.shape}"
        )
    if not math.isfinite(alpha_max_rad_s2) or alpha_max_rad_s2 <= 0.0:
        raise ValueError(
            f"alpha_max_rad_s2 must be positive and finite, got {alpha_max_rad_s2}"
        )
    if omega_max_rad_s is not None and (
        not math.isfinite(omega_max_rad_s) or omega_max_rad_s <= 0.0
    ):
        raise ValueError(
            f"omega_max_rad_s must be positive and finite, got {omega_max_rad_s}"
        )
    if not math.isfinite(central_body_gm_km3_s2) or central_body_gm_km3_s2 <= 0.0:
        raise ValueError(
            f"central_body_gm_km3_s2 must be positive and finite, "
            f"got {central_body_gm_km3_s2}"
        )
    if slew_floor_s <= 0.0:
        raise ValueError(f"slew_floor_s must be > 0, got {slew_floor_s}")
    if fd_dt_s <= 0.0:
        raise ValueError(f"fd_dt_s must be > 0, got {fd_dt_s}")

    # Two-body Kepler predictor for state(t_b + dt) under central-body GM.
    # spice.prop2b accepts dt = 0 (returns the input state) and negative
    # dt (so it also serves as the backward-difference predictor for omega_0).
    # Always defined: even when an actual-trajectory override is supplied
    # for the n_hat / omega sampling, the Kepler state is preserved in
    # the metadata as the documented predictor reference.
    def kepler_predictor(et: float) -> np.ndarray:
        dt = float(et - epoch_et_b)
        if dt == 0.0:
            return state_b.copy()
        return np.asarray(
            spice.prop2b(float(central_body_gm_km3_s2), state_b, dt),
            dtype=float,
        )

    def kepler_predictor_r(et: float) -> np.ndarray:
        return kepler_predictor(et)[:3]

    # When supplied, predicted_trajectory_fn provides all n_hat and omega
    # sampling; otherwise use the Kepler predictor.
    if predicted_trajectory_fn is not None:
        predictor_r = predicted_trajectory_fn
    else:
        predictor_r = kepler_predictor_r

    # Endpoint omega + n_hat measurements.
    n_hat_0 = np.asarray(cruise_old(state_b[:3], epoch_et_b), dtype=float)
    n_hat_0 = n_hat_0 / float(np.linalg.norm(n_hat_0))
    omega_0 = _measure_omega_via_predicted_trajectory(
        cruise_old, predictor_r, epoch_et_b, dt=fd_dt_s,
    )

    if omega_max_rad_s is None:
        # Alpha-only sizing uses the fixed-point path.
        T_slew = float(slew_floor_s)
        n_hat_f = None
        omega_f = None
        state_pred_end = None
        sizing_iters = 0
        last_sizing_delta_s = math.nan
        for sizing_iters in range(1, n_size_iterations + 1):
            et_end = epoch_et_b + T_slew
            state_pred_end = kepler_predictor(et_end)
            r_pred_end = predictor_r(et_end)
            n_hat_f = np.asarray(cruise_new(r_pred_end, et_end), dtype=float)
            n_hat_f = n_hat_f / float(np.linalg.norm(n_hat_f))
            omega_f = _measure_omega_via_predicted_trajectory(
                cruise_new, predictor_r, et_end, dt=fd_dt_s,
            )
            T_required = slew_duration_for_alpha_max(
                n_hat_0, n_hat_f,
                omega_0_rad_s=omega_0, omega_f_rad_s=omega_f,
                alpha_max_rad_s2=alpha_max_rad_s2,
                safety_factor=safety_factor,
                T_min_s=slew_floor_s,
                T_max_s=T_max_s,
            )
            T_required = max(T_required, slew_floor_s)
            last_sizing_delta_s = T_required - T_slew
            if abs(last_sizing_delta_s) < sizing_tol_s:
                T_slew = T_required
                break
            T_slew = T_required
        else:
            # Loop completed without breaking: did not converge.
            raise ValueError(
                f"cruise_to_cruise_slew sizing did not converge in "
                f"{n_size_iterations} iterations: last delta "
                f"= {last_sizing_delta_s:.4f} s"
            )
    else:
        def moving_endpoints(
            T_candidate_s: float,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            et_end = epoch_et_b + T_candidate_s
            r_pred_end = predictor_r(et_end)
            n_hat_f_candidate = np.asarray(
                cruise_new(r_pred_end, et_end), dtype=float,
            )
            n_hat_f_candidate /= float(np.linalg.norm(n_hat_f_candidate))
            omega_f_candidate = _measure_omega_via_predicted_trajectory(
                cruise_new, predictor_r, et_end, dt=fd_dt_s,
            )
            return n_hat_0, n_hat_f_candidate, omega_0, omega_f_candidate

        T_slew, final_endpoints, sizing_iters = (
            _moving_endpoint_slew_duration(
                moving_endpoints,
                alpha_max_rad_s2=alpha_max_rad_s2,
                omega_max_rad_s=omega_max_rad_s,
                safety_factor=safety_factor,
                T_min_s=slew_floor_s,
                T_max_s=T_max_s,
                root_tol_s=sizing_tol_s,
                context="cruise_to_cruise_slew",
            )
        )
        _, n_hat_f, _, omega_f = final_endpoints
        state_pred_end = kepler_predictor(epoch_et_b + T_slew)

    # Final endpoint sample at the converged T_slew.
    et_end_final = epoch_et_b + T_slew
    state_pred_end = kepler_predictor(et_end_final)
    r_pred_end = predictor_r(et_end_final)
    n_hat_f = np.asarray(cruise_new(r_pred_end, et_end_final), dtype=float)
    n_hat_f = n_hat_f / float(np.linalg.norm(n_hat_f))
    omega_f = _measure_omega_via_predicted_trajectory(
        cruise_new, predictor_r, et_end_final, dt=fd_dt_s,
    )

    slew_callable = smooth_slew_hermite(
        epoch_et_b, et_end_final,
        n_hat_0, n_hat_f,
        omega_0_rad_s=omega_0, omega_f_rad_s=omega_f,
    )

    cos_total = max(-1.0, min(1.0, float(np.dot(n_hat_0, n_hat_f))))
    theta_total = math.acos(cos_total)
    alpha_peak_bound = _hermite_peak_alpha_upper_bound(
        n_hat_0, n_hat_f, omega_0, omega_f, T_slew,
    )
    omega_peak = _hermite_peak_omega(
        n_hat_0, n_hat_f, omega_0, omega_f, T_slew,
    )
    if alpha_peak_bound > alpha_max_rad_s2 * (1.0 + 1.0e-12):
        raise RuntimeError(
            "cruise_to_cruise_slew constructed an alpha-infeasible slew: "
            f"peak bound={alpha_peak_bound:.6e}, "
            f"limit={alpha_max_rad_s2:.6e} rad/s^2"
        )
    if (
        omega_max_rad_s is not None
        and omega_peak > omega_max_rad_s * (1.0 + 1.0e-12)
    ):
        raise RuntimeError(
            "cruise_to_cruise_slew constructed an omega-infeasible slew: "
            f"peak={omega_peak:.6e}, limit={omega_max_rad_s:.6e} rad/s"
        )

    metadata = CruiseSlewMetadata(
        t_start_et=float(epoch_et_b),
        t_end_et=float(et_end_final),
        T_slew_s=float(T_slew),
        n_hat_0=n_hat_0,
        n_hat_f=n_hat_f,
        omega_0_rad_s=omega_0,
        omega_f_rad_s=omega_f,
        theta_total_rad=float(theta_total),
        alpha_max_budget_rad_s2=float(alpha_max_rad_s2),
        alpha_peak_upper_bound_rad_s2=float(alpha_peak_bound),
        alpha_utilization_pct=float(100.0 * alpha_peak_bound / alpha_max_rad_s2),
        sizing_iterations=int(sizing_iters),
        predictor_state_at_slew_end_km_kmps=state_pred_end.copy(),
        omega_max_budget_rad_s=(
            None if omega_max_rad_s is None else float(omega_max_rad_s)
        ),
        omega_peak_rad_s=float(omega_peak),
        omega_utilization_pct=(
            None
            if omega_max_rad_s is None
            else float(100.0 * omega_peak / omega_max_rad_s)
        ),
    )
    return slew_callable, metadata


def build_delivery_schedule(
    r_sat_predicted_fn: Callable[[float], np.ndarray],
    windows: Sequence[DeliveryWindow],
    target_lat_deg: float,
    target_lon_deg: float,
    *,
    cruise_profile: AttitudeCallable,
    epoch_et: float,
    duration_s: float,
    slew_duration_s: float,
    alpha_max_rad_s2: Optional[float] = None,
    omega_max_rad_s: Optional[float] = None,
    observer_naif_id: int = MARS_NAIF_ID,
) -> Tuple[AttitudeCallable, ScheduleMetadata]:
    """Compose a piecewise delivery schedule over one mission segment.

    Returns a pair ``(profile, metadata)`` where ``profile`` is an
    ``AttitudeCallable`` suitable for ``dynamics.propagate(...,
    sail_normal=profile, ...)``, and ``metadata`` is a
    ``ScheduleMetadata`` describing which input windows were kept,
    which were dropped (and why), and the segment boundary epochs of
    the composed schedule.

    Parameters
    ----------
    r_sat_predicted_fn
        Callable ``et -> r_sat_km (shape (3,))`` giving the sail
        position under a PREDICTED trajectory. Evaluated at slew and
        window boundaries only (not inside cruise or tracking
        segments), so the caller can supply e.g. a
        ``scipy.interpolate.CubicSpline`` built from a prior
        sun-pointing ``PropagationResult``, or any other predictor.
        Must be valid across ``[epoch_et, epoch_et + duration_s]``.
    windows
        Iterable of ``DeliveryWindow`` objects with populated
        absolute-ET fields (``et_start``, ``et_end`` must not be
        ``None``). Input ordering is preserved in the drop-reason
        reporting; processing order is by ``t_start_s`` ascending.
    target_lat_deg, target_lon_deg
        FALLBACK surface target for the tracking segments, used for
        any window whose own ``target_lat_deg`` / ``target_lon_deg``
        fields are ``None`` (for direct construction). Windows
        stamped by the finder carry their own target, and each track
        segment / slew endpoint computation uses ITS window's target
        in multi-target schedules.
    cruise_profile
        ``AttitudeCallable`` used between windows (and at the head /
        tail of the schedule). Typically ``attitude.sun_pointing()``
        unless an orbit-maintenance cruise profile is supplied.
    epoch_et
        Absolute SPICE TDB ET at the start of the schedule. Must
        match the ``epoch_et`` of the predicted trajectory and the
        ``DeliveryWindow.et_*`` fields.
    duration_s
        Total schedule duration (seconds). Schedule end is
        ``epoch_et + duration_s``.
    slew_duration_s
        Slew duration (seconds). Interpretation depends on
        ``alpha_max_rad_s2`` / ``omega_max_rad_s``:

        * both limits ``None`` (default): fixed slew duration
          applied to both slew-in and slew-out of every kept window,
          using the static-endpoint ``smooth_slew`` primitive.
        * either limit given: per-window slew durations are auto-sized and
          ``smooth_slew_hermite`` is used (dynamic endpoints matching
          cruise and bisector omega at the handoff). ``slew_duration_s``
          acts as a floor: if the auto-sized duration falls below it,
          the floor is used instead.
    alpha_max_rad_s2
        Optional angular-acceleration budget in rad/s^2. When supplied,
        each kept window's slew-in and slew-out durations are auto-
        sized to respect this budget, and the Hermite primitive is used so
        endpoint omega matches cruise / bisector exactly (no velocity
        discontinuity at handoff). With no omega limit,
        ``slew_duration_for_alpha_max`` performs alpha-only sizing.
    omega_max_rad_s
        Optional angular-rate budget in rad/s.  When supplied, slew duration
        sizing jointly enforces this continuous-path constraint and the alpha
        constraint (if supplied). The moving cruise-side endpoint is evaluated
        inside a bracketed duration solve so the final Hermite—not a
        nominal-floor estimate—is checked. ``None`` selects alpha-only sizing.
    observer_naif_id
        Central body NAIF id for SPICE lookups; default 499 (Mars).

    Raises
    ------
    ValueError
        For non-positive ``duration_s`` or ``slew_duration_s``, or if
        a kept window produces a bisector-degenerate geometry at its
        window endpoints (surfaces from ``_bisector_direction_at``).

    Notes
    -----
    The returned profile is state-dependent end to end. The composer evaluates
    ``r_sat_predicted_fn`` at slew/window boundaries to pin each Hermite
    endpoint and its rate; cruise and tracking segments consult the
    propagator's actual state at every evaluation.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if slew_duration_s <= 0.0:
        raise ValueError(
            f"slew_duration_s must be > 0, got {slew_duration_s}"
        )
    if alpha_max_rad_s2 is not None and (
        not math.isfinite(alpha_max_rad_s2) or alpha_max_rad_s2 <= 0.0
    ):
        raise ValueError(
            f"alpha_max_rad_s2 must be positive and finite, "
            f"got {alpha_max_rad_s2}"
        )
    if omega_max_rad_s is not None and (
        not math.isfinite(omega_max_rad_s) or omega_max_rad_s <= 0.0
    ):
        raise ValueError(
            f"omega_max_rad_s must be positive and finite, "
            f"got {omega_max_rad_s}"
        )

    schedule_end_et = epoch_et + duration_s
    auto_size = alpha_max_rad_s2 is not None or omega_max_rad_s is not None

    # Per-target bisector-pointing closures are built lazily and keyed on the
    # resolved (lat, lon).
    # Windows stamped with their own target (multi-target finder) get
    # that target's profile; windows whose target fields are None fall
    # back to the function's target arguments.
    bisector_profiles: dict[Tuple[float, float], AttitudeCallable] = {}

    def _resolve_window_target(w: DeliveryWindow) -> Tuple[float, float]:
        if w.target_lat_deg is None or w.target_lon_deg is None:
            return float(target_lat_deg), float(target_lon_deg)
        return float(w.target_lat_deg), float(w.target_lon_deg)

    def _bisector_profile_for(
        key: Tuple[float, float],
    ) -> AttitudeCallable:
        if key not in bisector_profiles:
            bisector_profiles[key] = bisector_pointing(
                key[0], key[1], observer_naif_id=int(observer_naif_id),
            )
        return bisector_profiles[key]

    # Sort windows by t_start_s while preserving the ORIGINAL index so
    # dropped-window reasons reference the caller's ordering.
    indexed_windows = list(enumerate(windows))
    indexed_windows.sort(key=lambda iw: iw[1].t_start_s)

    # -----------------------------------------------------------------
    # Pass 1: filter windows by slew-buffer feasibility. When
    # auto-sizing, measure endpoint omegas and compute per-window
    # slew durations here so the feasibility check uses actual slew
    # durations, not just the floor.
    # -----------------------------------------------------------------
    # kept record: (orig_idx, window, slew_in_start_et, slew_out_end_et,
    #               T_in, T_out, omega_0_in, omega_f_in, omega_0_out,
    #               omega_f_out)
    # The four omegas are set only in auto-size mode; static-mode paths
    # carry None and the Pass 2 builder uses smooth_slew in that case.
    kept: list[
        Tuple[
            int, DeliveryWindow, float, float, float, float,
            Optional[np.ndarray], Optional[np.ndarray],
            Optional[np.ndarray], Optional[np.ndarray],
        ]
    ] = []
    dropped: list[Tuple[int, str]] = []
    prev_slew_out_end_et = epoch_et

    for orig_idx, w in indexed_windows:
        if w.et_start is None or w.et_end is None:
            dropped.append((orig_idx, "window has no absolute-ET fields"))
            continue

        w_lat_deg, w_lon_deg = _resolve_window_target(w)
        bisector_profile = _bisector_profile_for((w_lat_deg, w_lon_deg))

        omega_0_in = omega_f_in = omega_0_out = omega_f_out = None

        if not auto_size:
            T_in = T_out = slew_duration_s
        elif omega_max_rad_s is None:
            # Measure endpoint omegas against the predicted trajectory.
            # Cruise omega is slowly varying (sun-pointing: ~1e-7 rad/s,
            # orbit-phase-locked cruise will be ~1e-3 but only drifts
            # minutes-scale), so sampling at the FLOOR candidate slew-in
            # start is a close-enough one-shot approximation. Bisector
            # omega is measured at the exact window boundaries.
            t_candidate_in_start = w.et_start - slew_duration_s
            t_candidate_out_end = w.et_end + slew_duration_s
            try:
                omega_0_in = _measure_omega_via_predicted_trajectory(
                    cruise_profile, r_sat_predicted_fn, t_candidate_in_start,
                )
                omega_f_in = _measure_omega_via_predicted_trajectory(
                    bisector_profile, r_sat_predicted_fn, w.et_start,
                )
                omega_0_out = _measure_omega_via_predicted_trajectory(
                    bisector_profile, r_sat_predicted_fn, w.et_end,
                )
                omega_f_out = _measure_omega_via_predicted_trajectory(
                    cruise_profile, r_sat_predicted_fn, t_candidate_out_end,
                )
            except (ValueError, RuntimeError) as exc:
                dropped.append(
                    (orig_idx, f"endpoint omega sampling failed: {exc}")
                )
                continue

            # Endpoint normals for sizing (used only here; Pass 2
            # re-evaluates them at the finalised boundary times).
            r_pred_cruise_end = r_sat_predicted_fn(t_candidate_in_start)
            n_cruise_end_est = cruise_profile(
                r_pred_cruise_end, t_candidate_in_start,
            )
            r_pred_track_start = r_sat_predicted_fn(w.et_start)
            try:
                n_bis_track_start_est = _bisector_direction_at(
                    r_pred_track_start, w.et_start,
                    w_lat_deg, w_lon_deg, observer_naif_id,
                )
            except ValueError as exc:
                dropped.append((orig_idx, f"bisector degenerate at et_start: {exc}"))
                continue
            r_pred_track_end = r_sat_predicted_fn(w.et_end)
            try:
                n_bis_track_end_est = _bisector_direction_at(
                    r_pred_track_end, w.et_end,
                    w_lat_deg, w_lon_deg, observer_naif_id,
                )
            except ValueError as exc:
                dropped.append((orig_idx, f"bisector degenerate at et_end: {exc}"))
                continue
            r_pred_cruise_resume = r_sat_predicted_fn(t_candidate_out_end)
            n_cruise_resume_est = cruise_profile(
                r_pred_cruise_resume, t_candidate_out_end,
            )

            try:
                T_in = slew_duration_for_alpha_max(
                    n_cruise_end_est, n_bis_track_start_est,
                    omega_0_rad_s=omega_0_in,
                    omega_f_rad_s=omega_f_in,
                    alpha_max_rad_s2=alpha_max_rad_s2,
                )
                T_out = slew_duration_for_alpha_max(
                    n_bis_track_end_est, n_cruise_resume_est,
                    omega_0_rad_s=omega_0_out,
                    omega_f_rad_s=omega_f_out,
                    alpha_max_rad_s2=alpha_max_rad_s2,
                )
            except ValueError as exc:
                dropped.append((orig_idx, f"slew auto-size failed: {exc}"))
                continue

            # Apply the caller-supplied floor.
            T_in = max(T_in, slew_duration_s)
            T_out = max(T_out, slew_duration_s)
        else:
            # Omega-aware path: the bisector side of each slew is fixed at the
            # window boundary, but the cruise side moves when T changes.  The
            # The alpha-only path samples that moving endpoint once at
            # the nominal floor. Solve the final moving-endpoint constraint
            # directly here; raw duration fixed-point iteration can converge
            # too slowly and falsely reject physically feasible windows.
            r_pred_track_start = r_sat_predicted_fn(w.et_start)
            r_pred_track_end = r_sat_predicted_fn(w.et_end)
            try:
                n_bis_track_start_est = _bisector_direction_at(
                    r_pred_track_start, w.et_start,
                    w_lat_deg, w_lon_deg, observer_naif_id,
                )
                n_bis_track_end_est = _bisector_direction_at(
                    r_pred_track_end, w.et_end,
                    w_lat_deg, w_lon_deg, observer_naif_id,
                )
                omega_f_in = _measure_omega_via_predicted_trajectory(
                    bisector_profile, r_sat_predicted_fn, w.et_start,
                )
                omega_0_out = _measure_omega_via_predicted_trajectory(
                    bisector_profile, r_sat_predicted_fn, w.et_end,
                )
            except (ValueError, RuntimeError) as exc:
                dropped.append(
                    (orig_idx, f"fixed bisector endpoint sampling failed: {exc}")
                )
                continue

            def _size_against_moving_cruise(
                *,
                role: str,
                n_bisector: np.ndarray,
                omega_bisector: np.ndarray,
            ) -> Tuple[float, np.ndarray]:
                def moving_endpoints(
                    T_candidate: float,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    if role == "slew_in":
                        et_cruise = float(w.et_start) - T_candidate
                    else:
                        et_cruise = float(w.et_end) + T_candidate
                    r_cruise = r_sat_predicted_fn(et_cruise)
                    n_cruise = np.asarray(
                        cruise_profile(r_cruise, et_cruise), dtype=float,
                    )
                    omega_cruise = _measure_omega_via_predicted_trajectory(
                        cruise_profile, r_sat_predicted_fn, et_cruise,
                    )
                    if role == "slew_in":
                        n0, nf = n_cruise, n_bisector
                        w0, wf = omega_cruise, omega_bisector
                    else:
                        n0, nf = n_bisector, n_cruise
                        w0, wf = omega_bisector, omega_cruise
                    return n0, nf, w0, wf

                if role == "slew_in":
                    available_gap_s = float(w.et_start - prev_slew_out_end_et)
                else:
                    available_gap_s = float(schedule_end_et - w.et_end)
                T_max_role_s = min(3600.0, available_gap_s)
                T_sized, final_endpoints, _ = _moving_endpoint_slew_duration(
                    moving_endpoints,
                    alpha_max_rad_s2=alpha_max_rad_s2,
                    omega_max_rad_s=omega_max_rad_s,
                    T_min_s=float(slew_duration_s),
                    T_max_s=T_max_role_s,
                    root_tol_s=0.05,
                    context=f"window {orig_idx} {role}",
                )
                if role == "slew_in":
                    omega_final_cruise = final_endpoints[2]
                else:
                    omega_final_cruise = final_endpoints[3]
                return T_sized, omega_final_cruise

            try:
                T_in, omega_0_in = _size_against_moving_cruise(
                    role="slew_in",
                    n_bisector=n_bis_track_start_est,
                    omega_bisector=omega_f_in,
                )
                T_out, omega_f_out = _size_against_moving_cruise(
                    role="slew_out",
                    n_bisector=n_bis_track_end_est,
                    omega_bisector=omega_0_out,
                )
            except (ValueError, RuntimeError) as exc:
                dropped.append((orig_idx, f"slew auto-size failed: {exc}"))
                continue

        slew_in_start_et = w.et_start - T_in
        slew_out_end_et = w.et_end + T_out
        if slew_in_start_et < epoch_et:
            dropped.append((orig_idx, "slew_in precedes schedule epoch"))
            continue
        if slew_out_end_et > schedule_end_et:
            dropped.append((orig_idx, "slew_out exceeds schedule end"))
            continue
        if slew_in_start_et < prev_slew_out_end_et:
            dropped.append(
                (orig_idx, "slew_in overlaps previous window's slew_out")
            )
            continue
        kept.append((
            orig_idx, w, slew_in_start_et, slew_out_end_et,
            T_in, T_out,
            omega_0_in, omega_f_in, omega_0_out, omega_f_out,
        ))
        prev_slew_out_end_et = slew_out_end_et

    for orig_idx, reason in dropped:
        logger.warning(
            "attitude_schedule: dropping window %d (%s)", orig_idx, reason
        )

    # -----------------------------------------------------------------
    # Pass 2: build segments.
    # -----------------------------------------------------------------
    segments: list[Tuple[float, float, AttitudeCallable]] = []
    boundaries: list[float] = [epoch_et]
    last_end_et = epoch_et

    for (
        orig_idx, w, slew_in_start_et, slew_out_end_et,
        T_in, T_out, omega_0_in, omega_f_in, omega_0_out, omega_f_out,
    ) in kept:
        w_lat_deg, w_lon_deg = _resolve_window_target(w)
        bisector_profile = _bisector_profile_for((w_lat_deg, w_lon_deg))

        # Cruise segment: [last_end_et, slew_in_start_et]. Skip if
        # zero-duration (exact contiguity).
        if slew_in_start_et > last_end_et:
            segments.append((last_end_et, slew_in_start_et, cruise_profile))
            boundaries.append(slew_in_start_et)

        # --- slew_in ---
        r_pred_cruise_end = r_sat_predicted_fn(slew_in_start_et)
        n_cruise_end = cruise_profile(r_pred_cruise_end, slew_in_start_et)
        r_pred_track_start = r_sat_predicted_fn(w.et_start)
        n_bis_track_start = _bisector_direction_at(
            r_pred_track_start, w.et_start,
            w_lat_deg, w_lon_deg, observer_naif_id,
        )
        if auto_size:
            # Pass 1 sampled cruise ω at the FLOOR candidate
            # (w.et_start - slew_duration_s) for slew sizing. The actual
            # slew_in_start_et = w.et_start - T_in may be earlier when
            # auto-sized T_in > slew_duration_s. Resample cruise ω at the
            # actual slew start so the Hermite endpoint ω matches what
            # the cruise law's ω will be at the integrator handoff.
            # Without this, cruise→slew_in has a residual ω-step of order
            # |dω_cruise/dt| × (T_in - slew_duration_s), which can reach
            # ~3e-4 rad/s with T_in=527 s and a 300 s floor.
            omega_0_in_actual = _measure_omega_via_predicted_trajectory(
                cruise_profile, r_sat_predicted_fn, slew_in_start_et,
            )
            slew_in = smooth_slew_hermite(
                slew_in_start_et, w.et_start,
                n_cruise_end, n_bis_track_start,
                omega_0_rad_s=omega_0_in_actual,
                omega_f_rad_s=omega_f_in,
            )
            if omega_max_rad_s is not None:
                omega_peak_in = _hermite_peak_omega(
                    n_cruise_end / float(np.linalg.norm(n_cruise_end)),
                    n_bis_track_start,
                    omega_0_in_actual,
                    omega_f_in,
                    T_in,
                )
                if omega_peak_in > omega_max_rad_s * (1.0 + 1.0e-12):
                    raise RuntimeError(
                        "build_delivery_schedule constructed an "
                        f"omega-infeasible slew_in for window {orig_idx}: "
                        f"peak={omega_peak_in:.6e}, "
                        f"limit={omega_max_rad_s:.6e} rad/s"
                    )
                if alpha_max_rad_s2 is not None:
                    alpha_peak_in = _hermite_peak_alpha_upper_bound(
                        n_cruise_end / float(np.linalg.norm(n_cruise_end)),
                        n_bis_track_start,
                        omega_0_in_actual,
                        omega_f_in,
                        T_in,
                    )
                    if alpha_peak_in > alpha_max_rad_s2 * (1.0 + 1.0e-12):
                        raise RuntimeError(
                            "build_delivery_schedule constructed an "
                            f"alpha-infeasible slew_in for window {orig_idx}: "
                            f"peak bound={alpha_peak_in:.6e}, "
                            f"limit={alpha_max_rad_s2:.6e} rad/s^2"
                        )
        else:
            slew_in = smooth_slew(
                slew_in_start_et, w.et_start, n_cruise_end, n_bis_track_start,
            )
        segments.append((slew_in_start_et, w.et_start, slew_in))
        boundaries.append(w.et_start)

        # --- track ---
        segments.append((w.et_start, w.et_end, bisector_profile))
        boundaries.append(w.et_end)

        # --- slew_out ---
        r_pred_track_end = r_sat_predicted_fn(w.et_end)
        n_bis_track_end = _bisector_direction_at(
            r_pred_track_end, w.et_end,
            w_lat_deg, w_lon_deg, observer_naif_id,
        )
        r_pred_cruise_resume = r_sat_predicted_fn(slew_out_end_et)
        n_cruise_resume = cruise_profile(
            r_pred_cruise_resume, slew_out_end_et
        )
        if auto_size:
            # Symmetric resample for slew_out's exit ω (cruise side).
            # See slew_in's resample comment above for the mechanism.
            omega_f_out_actual = _measure_omega_via_predicted_trajectory(
                cruise_profile, r_sat_predicted_fn, slew_out_end_et,
            )
            slew_out = smooth_slew_hermite(
                w.et_end, slew_out_end_et,
                n_bis_track_end, n_cruise_resume,
                omega_0_rad_s=omega_0_out,
                omega_f_rad_s=omega_f_out_actual,
            )
            if omega_max_rad_s is not None:
                omega_peak_out = _hermite_peak_omega(
                    n_bis_track_end,
                    n_cruise_resume / float(np.linalg.norm(n_cruise_resume)),
                    omega_0_out,
                    omega_f_out_actual,
                    T_out,
                )
                if omega_peak_out > omega_max_rad_s * (1.0 + 1.0e-12):
                    raise RuntimeError(
                        "build_delivery_schedule constructed an "
                        f"omega-infeasible slew_out for window {orig_idx}: "
                        f"peak={omega_peak_out:.6e}, "
                        f"limit={omega_max_rad_s:.6e} rad/s"
                    )
                if alpha_max_rad_s2 is not None:
                    alpha_peak_out = _hermite_peak_alpha_upper_bound(
                        n_bis_track_end,
                        n_cruise_resume / float(np.linalg.norm(n_cruise_resume)),
                        omega_0_out,
                        omega_f_out_actual,
                        T_out,
                    )
                    if alpha_peak_out > alpha_max_rad_s2 * (1.0 + 1.0e-12):
                        raise RuntimeError(
                            "build_delivery_schedule constructed an "
                            f"alpha-infeasible slew_out for window {orig_idx}: "
                            f"peak bound={alpha_peak_out:.6e}, "
                            f"limit={alpha_max_rad_s2:.6e} rad/s^2"
                        )
        else:
            slew_out = smooth_slew(
                w.et_end, slew_out_end_et, n_bis_track_end, n_cruise_resume,
            )
        segments.append((w.et_end, slew_out_end_et, slew_out))
        boundaries.append(slew_out_end_et)

        last_end_et = slew_out_end_et

    # Final cruise segment out to schedule end.
    if schedule_end_et > last_end_et:
        segments.append((last_end_et, schedule_end_et, cruise_profile))
        boundaries.append(schedule_end_et)

    # Degenerate case: no windows kept, no cruise segments added yet.
    # ``last_end_et == epoch_et`` and the "final cruise" block above
    # would have appended the single covering segment; only if the
    # schedule is empty (duration_s = 0) would segments be empty, but
    # zero duration is rejected above.
    if not segments:
        raise RuntimeError(
            "attitude_schedule: empty segment list despite non-zero "
            "duration -- internal schedule consistency error."
        )

    composed = piecewise(segments)
    metadata = ScheduleMetadata(
        n_windows_kept=len(kept),
        n_windows_dropped=len(dropped),
        dropped_window_reasons=tuple(dropped),
        segment_boundaries_et=tuple(boundaries),
    )
    return composed, metadata


# ---------------------------------------------------------------------------
# Fixed-point iteration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinedSchedule:
    """Output of ``refine_delivery_schedule``.

    Fields
    ------
    final_result
        The ``PropagationResult`` from the last iteration's
        propagation under the converged schedule.
    final_windows
        Tuple of ``DeliveryWindow`` objects that were STABLE across
        the final two iterations (matched via Jaccard > 0.5 on
        absolute ET overlap). These are the windows the schedule
        was built around at convergence.
    unstable_windows
        Tuple of ``DeliveryWindow`` objects that appeared or
        disappeared between the final two iterations. Reported
        separately so callers can decide whether to treat them as
        optimistic windows or discard. Empty at convergence.
    final_profile
        The ``AttitudeCallable`` returned by the final
        ``build_delivery_schedule`` call. Suitable for further
        propagate() use.
    metadata
        The ``ScheduleMetadata`` from the final schedule build.
    n_iterations
        Number of propagate + build cycles actually executed
        (<= ``max_iterations``; 0 if the caller set max_iterations=0
        and the loop short-circuited).
    converged
        True if the window-boundary drift fell below the configured
        tolerance AND the stable / unstable split was unambiguous on the
        last iteration. False if the budget was exhausted.
    boundary_drift_history_et_s
        Tuple of ``max|et_start_new - et_start_old|`` (across matched
        windows) per iteration. Length = n_iterations. Useful for
        diagnostics and for deciding whether a non-converged result
        is still usable.
    """

    final_result: "PropagationResult"  # noqa: F821 (duck-typed)
    final_windows: Tuple[DeliveryWindow, ...]
    unstable_windows: Tuple[DeliveryWindow, ...]
    final_profile: AttitudeCallable
    metadata: ScheduleMetadata
    n_iterations: int
    converged: bool
    boundary_drift_history_et_s: Tuple[float, ...]
    initialization_mode: str = "global_cruise_scan"
    window_search_modes: Tuple[str, ...] = ()
    continuation_fallback_reasons: Tuple[str, ...] = ()
    n_propagations: int = 0


@dataclass(frozen=True)
class DeliveryScheduleSeed:
    """Compact prior-sol state needed to continue a repeated schedule.

    The stored trajectory remains in its source-sol J2000 frame. For a new
    epoch, :meth:`position_predictor_for_epoch` maps each requested position
    through ``J2000 -> IAU_MARS`` at the source epoch and back through
    ``IAU_MARS -> J2000`` at the corresponding new-sol epoch. Thus the seed
    carries forward the *Mars-fixed* repeat-ground-track geometry rather than
    treating the preceding sol's inertial vectors as repeatable.
    """

    source_epoch_et: float
    t_s: np.ndarray
    positions_j2000_km: np.ndarray
    windows: Tuple[DeliveryWindow, ...]
    velocities_j2000_km_s: Optional[np.ndarray] = None
    mu_km3_s2: Optional[float] = None

    def __post_init__(self) -> None:
        source_epoch_et = float(self.source_epoch_et)
        if not math.isfinite(source_epoch_et):
            raise ValueError("source_epoch_et must be finite")
        t_s = np.asarray(self.t_s, dtype=float).copy()
        positions = np.asarray(self.positions_j2000_km, dtype=float).copy()
        velocities = (
            None
            if self.velocities_j2000_km_s is None
            else np.asarray(self.velocities_j2000_km_s, dtype=float).copy()
        )
        if t_s.ndim != 1 or t_s.size < 4:
            raise ValueError("t_s must be one-dimensional with at least 4 samples")
        if positions.shape != (t_s.size, 3):
            raise ValueError(
                "positions_j2000_km must have shape (len(t_s), 3), got "
                f"{positions.shape} for {t_s.size} times"
            )
        if velocities is not None and velocities.shape != (t_s.size, 3):
            raise ValueError(
                "velocities_j2000_km_s must have shape (len(t_s), 3), got "
                f"{velocities.shape} for {t_s.size} times"
            )
        mu = None if self.mu_km3_s2 is None else float(self.mu_km3_s2)
        if mu is not None and (not math.isfinite(mu) or mu <= 0.0):
            raise ValueError("mu_km3_s2 must be positive and finite")
        if (
            np.any(~np.isfinite(t_s))
            or np.any(~np.isfinite(positions))
            or (velocities is not None and np.any(~np.isfinite(velocities)))
            or np.any(np.diff(t_s) <= 0.0)
        ):
            raise ValueError(
                "seed times/positions must be finite and times strictly increasing"
            )
        t_s.setflags(write=False)
        positions.setflags(write=False)
        if velocities is not None:
            velocities.setflags(write=False)
        object.__setattr__(self, "source_epoch_et", source_epoch_et)
        object.__setattr__(self, "t_s", t_s)
        object.__setattr__(self, "positions_j2000_km", positions)
        object.__setattr__(self, "velocities_j2000_km_s", velocities)
        object.__setattr__(self, "mu_km3_s2", mu)
        object.__setattr__(self, "windows", tuple(self.windows))

    @classmethod
    def from_refined(cls, refined: RefinedSchedule) -> "DeliveryScheduleSeed":
        """Capture the final sampled trajectory and windows of one sol."""
        result = refined.final_result
        if result.epoch_et is None:
            raise ValueError(
                "cannot seed continuation from a result without epoch_et"
            )
        return cls(
            source_epoch_et=float(result.epoch_et),
            t_s=np.asarray(result.t_s, dtype=float),
            positions_j2000_km=np.asarray(result.state_km_kmps, dtype=float)[:, :3],
            windows=tuple(refined.final_windows),
            velocities_j2000_km_s=np.asarray(
                result.state_km_kmps, dtype=float
            )[:, 3:],
            mu_km3_s2=float(result.mu_km3_s2),
        )

    def shifted_windows(
        self,
        epoch_et: float,
        t_span_s: Tuple[float, float],
    ) -> Tuple[DeliveryWindow, ...]:
        """Return seed windows with absolute ET fields shifted to a new sol."""
        t_start_s, t_end_s = map(float, t_span_s)
        shifted: list[DeliveryWindow] = []
        for window in self.windows:
            if (
                float(window.t_start_s) < t_start_s
                or float(window.t_end_s) > t_end_s
            ):
                raise WindowContinuationError(
                    "seed window lies outside the new propagation span: "
                    f"[{window.t_start_s}, {window.t_end_s}] vs "
                    f"[{t_start_s}, {t_end_s}] s"
                )
            shifted.append(
                replace(
                    window,
                    et_start=float(epoch_et) + float(window.t_start_s),
                    et_end=float(epoch_et) + float(window.t_end_s),
                )
            )
        return tuple(shifted)

    def position_predictor_for_epoch(
        self,
        epoch_et: float,
        t_span_s: Tuple[float, float],
        initial_state_j2000_km_kmps: Optional[np.ndarray] = None,
    ) -> Callable[[float], np.ndarray]:
        """Map the source Mars-fixed trajectory into a new sol's J2000 axes.

        When source velocities, the gravity parameter, and the actual new-sol
        state are available, the mapped repeat receives the difference between
        two Kepler propagations: one from the actual new state and one from the
        source start state mapped into the new epoch. This aligns position and
        velocity at the handoff and evolves their small closure residual
        dynamically. Position-only seeds use a constant Mars-fixed
        translation instead.
        """
        requested_start_s, requested_end_s = map(float, t_span_s)
        tol_s = 1.0e-6
        if (
            requested_start_s < float(self.t_s[0]) - tol_s
            or requested_end_s > float(self.t_s[-1]) + tol_s
        ):
            raise WindowContinuationError(
                "seed trajectory does not span the requested propagation: "
                f"seed [{self.t_s[0]}, {self.t_s[-1]}] s, requested "
                f"[{requested_start_s}, {requested_end_s}] s"
            )

        source_cs = CubicSpline(
            self.t_s,
            self.positions_j2000_km,
            axis=0,
            extrapolate=False,
        )
        new_epoch_et = float(epoch_et)
        mars_fixed_alignment_km = np.zeros(3, dtype=float)
        differential_state_actual = None
        differential_state_nominal = None
        if initial_state_j2000_km_kmps is not None:
            current_initial_state = np.asarray(
                initial_state_j2000_km_kmps, dtype=float
            )
            if current_initial_state.shape != (6,) or np.any(
                ~np.isfinite(current_initial_state)
            ):
                raise ValueError(
                    "initial_state_j2000_km_kmps must be a finite 6-vector"
                )
            source_initial_et = self.source_epoch_et + requested_start_s
            source_initial_j2000 = np.asarray(
                source_cs(requested_start_s), dtype=float
            )
            source_initial_iau_mars = (
                frame_rotation(
                    "J2000", "IAU_MARS", source_initial_et
                ) @ source_initial_j2000
            )
            current_initial_iau_mars = (
                frame_rotation(
                    "J2000", "IAU_MARS", new_epoch_et + requested_start_s
                ) @ current_initial_state[:3]
            )
            if self.velocities_j2000_km_s is None or self.mu_km3_s2 is None:
                mars_fixed_alignment_km = (
                    current_initial_iau_mars - source_initial_iau_mars
                )
            else:
                source_velocity_cs = CubicSpline(
                    self.t_s,
                    self.velocities_j2000_km_s,
                    axis=0,
                    extrapolate=False,
                )
                source_initial_state = np.concatenate([
                    source_initial_j2000,
                    np.asarray(
                        source_velocity_cs(requested_start_s), dtype=float
                    ),
                ])
                source_to_body_fixed = np.asarray(
                    spice.sxform(
                        "J2000", "IAU_MARS", source_initial_et
                    ),
                    dtype=float,
                )
                body_fixed_to_new = np.asarray(
                    spice.sxform(
                        "IAU_MARS",
                        "J2000",
                        new_epoch_et + requested_start_s,
                    ),
                    dtype=float,
                )
                differential_state_nominal = (
                    body_fixed_to_new @ source_to_body_fixed
                    @ source_initial_state
                )
                differential_state_actual = current_initial_state.copy()

        def predictor(et: float) -> np.ndarray:
            relative_t_s = float(et) - new_epoch_et
            if (
                relative_t_s < float(self.t_s[0]) - tol_s
                or relative_t_s > float(self.t_s[-1]) + tol_s
            ):
                raise WindowContinuationError(
                    f"seed predictor queried outside its span at t={relative_t_s} s"
                )
            source_et = self.source_epoch_et + relative_t_s
            r_source_j2000 = np.asarray(source_cs(relative_t_s), dtype=float)
            r_iau_mars = (
                frame_rotation("J2000", "IAU_MARS", source_et)
                @ r_source_j2000
            )
            r_iau_mars = r_iau_mars + mars_fixed_alignment_km
            repeated_j2000 = (
                frame_rotation("IAU_MARS", "J2000", float(et))
                @ r_iau_mars
            )
            if (
                differential_state_actual is None
                or differential_state_nominal is None
            ):
                return repeated_j2000
            dt_s = relative_t_s - requested_start_s
            if dt_s == 0.0:
                differential_position_km = (
                    differential_state_actual[:3]
                    - differential_state_nominal[:3]
                )
            else:
                actual_two_body = np.asarray(
                    spice.prop2b(
                        float(self.mu_km3_s2),
                        differential_state_actual,
                        dt_s,
                    ),
                    dtype=float,
                )
                nominal_two_body = np.asarray(
                    spice.prop2b(
                        float(self.mu_km3_s2),
                        differential_state_nominal,
                        dt_s,
                    ),
                    dtype=float,
                )
                differential_position_km = (
                    actual_two_body[:3] - nominal_two_body[:3]
                )
            return repeated_j2000 + differential_position_km

        return predictor


def _jaccard_et(
    w_a: DeliveryWindow, w_b: DeliveryWindow
) -> float:
    """Jaccard overlap on absolute ET intervals ``[et_start, et_end]``.

    Returns 0 if either window lacks absolute ET fields, or if the
    intervals are disjoint.
    """
    if (
        w_a.et_start is None or w_a.et_end is None
        or w_b.et_start is None or w_b.et_end is None
    ):
        return 0.0
    s_a, e_a = w_a.et_start, w_a.et_end
    s_b, e_b = w_b.et_start, w_b.et_end
    inter_start = max(s_a, s_b)
    inter_end = min(e_a, e_b)
    inter = max(0.0, inter_end - inter_start)
    union_start = min(s_a, s_b)
    union_end = max(e_a, e_b)
    union = max(0.0, union_end - union_start)
    if union == 0.0:
        return 0.0
    return inter / union


def _match_windows(
    old_windows: Sequence[DeliveryWindow],
    new_windows: Sequence[DeliveryWindow],
    *,
    jaccard_threshold: float = 0.5,
) -> Tuple[
    list[Tuple[DeliveryWindow, DeliveryWindow]],  # matched pairs (old, new)
    list[DeliveryWindow],                          # unmatched old (disappeared)
    list[DeliveryWindow],                          # unmatched new (appeared)
]:
    """Greedy Jaccard-based matching of two window lists on absolute ET.

    For each old window, find the new window with highest Jaccard
    overlap. If that overlap exceeds ``jaccard_threshold`` (default
    0.5), pair them; otherwise mark the old as "disappeared". Any
    new windows not paired to an old are "appeared".

    Greedy (first-come-first-served) matching is appropriate because the
    two window lists are time-ordered and changes are local; under these
    conditions the optimal and greedy matchings almost always coincide.
    """
    matched_pairs: list[Tuple[DeliveryWindow, DeliveryWindow]] = []
    unmatched_old: list[DeliveryWindow] = []
    claimed_new: set[int] = set()

    for w_old in old_windows:
        best_idx = -1
        best_j = 0.0
        for j, w_new in enumerate(new_windows):
            if j in claimed_new:
                continue
            # Multi-target: never match windows of different targets,
            # regardless of ET overlap. Single-target lists all carry
            # target_idx == 0, so this guard is inert there.
            if w_old.target_idx != w_new.target_idx:
                continue
            jac = _jaccard_et(w_old, w_new)
            if jac > best_j:
                best_j = jac
                best_idx = j
        if best_idx >= 0 and best_j >= jaccard_threshold:
            matched_pairs.append((w_old, new_windows[best_idx]))
            claimed_new.add(best_idx)
        else:
            unmatched_old.append(w_old)

    unmatched_new = [
        w for j, w in enumerate(new_windows) if j not in claimed_new
    ]
    return matched_pairs, unmatched_old, unmatched_new


def _apply_damping(
    old_windows: Sequence[DeliveryWindow],
    new_windows: Sequence[DeliveryWindow],
    damping: float,
    jaccard_threshold: float,
) -> Tuple[list[DeliveryWindow], list[DeliveryWindow]]:
    """Damp matched-window absolute ET endpoints between iterations.

    For each matched pair (w_old, w_new), the damped window takes:
        et_start_damped = damping * et_start_new + (1 - damping) * et_start_old
        et_end_damped   = damping * et_end_new   + (1 - damping) * et_end_old
    and preserves all other fields from w_new (duration, scalars).

    Unmatched NEW windows are passed through unmodified (no history
    to damp against). Unmatched OLD windows are DROPPED (they
    disappeared this iteration).

    Returns (damped_windows, unstable_windows) where unstable_windows
    collects the old-disappeared and new-appeared sets for reporting.
    """
    matched_pairs, unmatched_old, unmatched_new = _match_windows(
        old_windows, new_windows, jaccard_threshold=jaccard_threshold,
    )

    damped: list[DeliveryWindow] = []
    unstable: list[DeliveryWindow] = list(unmatched_old) + list(unmatched_new)
    a = float(damping)

    for w_old, w_new in matched_pairs:
        et_start_new = w_new.et_start
        et_end_new = w_new.et_end
        et_start_old = w_old.et_start
        et_end_old = w_old.et_end
        # Confirmed non-None by _jaccard_et; re-check for mypy-style safety.
        if (
            et_start_new is None or et_end_new is None
            or et_start_old is None or et_end_old is None
        ):
            damped.append(w_new)
            continue
        et_start_damped = a * et_start_new + (1.0 - a) * et_start_old
        et_end_damped = a * et_end_new + (1.0 - a) * et_end_old
        # DeliveryWindow is a frozen dataclass; build a replacement.
        from dataclasses import replace
        damped.append(replace(
            w_new,
            et_start=et_start_damped,
            et_end=et_end_damped,
        ))

    # Add the new-appeared set last (no damping history).
    damped.extend(unmatched_new)
    # Sort by t_start_s to maintain monotone ordering.
    damped.sort(key=lambda w: w.t_start_s)

    return damped, unstable


def _max_boundary_drift(
    old_windows: Sequence[DeliveryWindow],
    new_windows: Sequence[DeliveryWindow],
    *,
    jaccard_threshold: float = 0.5,
) -> float:
    """Max |et_start_new - et_start_old| + |et_end_new - et_end_old|
    across matched windows. Returns inf if any window was unmatched
    (i.e. count changed), signaling non-convergence on that iteration.
    """
    matched_pairs, unmatched_old, unmatched_new = _match_windows(
        old_windows, new_windows, jaccard_threshold=jaccard_threshold,
    )
    if unmatched_old or unmatched_new:
        return float("inf")
    max_drift = 0.0
    for w_old, w_new in matched_pairs:
        if (
            w_old.et_start is None or w_new.et_start is None
            or w_old.et_end is None or w_new.et_end is None
        ):
            return float("inf")
        ds = abs(w_new.et_start - w_old.et_start)
        de = abs(w_new.et_end - w_old.et_end)
        if max(ds, de) > max_drift:
            max_drift = max(ds, de)
    return max_drift


def refine_delivery_schedule(
    initial_state_km_kmps: np.ndarray,
    t_span_s: Tuple[float, float],
    *,
    epoch_et: float,
    cruise_profile: AttitudeCallable,
    target_lat_deg: float,
    target_lon_deg: float,
    sail: "SolarSail",  # noqa: F821 (duck-typed)
    extra_targets: Sequence[Tuple[float, float]] = (),
    slew_duration_s: float = 300.0,
    alpha_max_rad_s2: Optional[float] = None,
    omega_max_rad_s: Optional[float] = None,
    max_iterations: int = 5,
    convergence_tol_et_s: float = 1.0,
    damping: float = 0.7,
    jaccard_threshold: float = 0.5,
    propagate_kwargs: Optional[dict] = None,
    find_windows_kwargs: Optional[dict] = None,
    observer_naif_id: int = MARS_NAIF_ID,
    continuation_seed: Optional[DeliveryScheduleSeed] = None,
    continuation_search_margin_s: float = 900.0,
    continuation_max_boundary_shift_s: Optional[float] = None,
    continuation_failure: str = "full_scan",
) -> RefinedSchedule:
    """Fixed-point iteration: propagate, find windows, build schedule, repeat.

    Resolves the coupling between the scheduled attitude profile and
    the window boundaries it implies: the schedule is built from
    windows found under a predicted trajectory, but the resulting
    SRP acceleration shifts the actual trajectory, which shifts the
    windows. One-shot ``build_delivery_schedule`` runs on stale
    windows; this routine iterates until the boundaries stabilise.

    First-sol initialization propagates under the cruise profile and performs
    one global window search. When ``continuation_seed`` is supplied for a
    noninitial repeated sol, its prior trajectory is mapped through IAU_MARS into
    the new epoch and its shifted windows build the initial schedule directly;
    the redundant cruise-only propagation and global discovery scan are then
    skipped.

    Algorithm per refinement iteration:
      1. Propagate initial_state under the current attitude profile.
      2. Build a cubic-spline interpolant of the new trajectory.
      3. Find delivery windows on the new trajectory. Seeded continuations
         evaluate the exact gates only in guarded bands around the preceding
         boundaries; validation failure either raises or falls back to a
         complete scan according to ``continuation_failure``.
      4. Match new windows to previous-iteration windows by Jaccard
         overlap on absolute ET; damp the et endpoints of matched
         windows by (1 - damping) * old + damping * new. Unmatched
         windows are tracked as unstable.
      5. If ``max|et_start_drift|`` across matched windows falls
         below ``convergence_tol_et_s`` AND the unstable set is
         empty, declare convergence.
      6. Otherwise, build a new schedule from the damped windows and
         loop.

    Parameters
    ----------
    initial_state_km_kmps
        Initial sail state (6-vector, position + velocity in J2000).
    t_span_s
        Propagation time span, relative to epoch_et.
    epoch_et
        Absolute SPICE TDB of the propagation start.
    cruise_profile
        Cruise-attitude callable used in the very first iteration
        (before any schedule is built) AND between windows in all
        refinement iterations.
    target_lat_deg, target_lon_deg
        PRIMARY surface target (target_idx 0) for bisector tracking
        and window evaluation.
    extra_targets
        Additional ``(lat_deg, lon_deg)`` surface targets
        (target_idx 1, 2, ... in order). Windows are found for ALL
        targets each iteration via
        ``visibility.find_delivery_windows_multi`` and each window's
        tracking segments point at its own target. Default ``()`` matches the
        single-target calculation exactly.
    sail
        ``SolarSail`` passed to both ``propagate`` and
        ``find_delivery_windows`` (for SRP + fluence scoring).
    slew_duration_s
        Slew duration floor (also the baseline if alpha_max is None).
    alpha_max_rad_s2
        Optional auto-sizing budget. See ``build_delivery_schedule``.
    omega_max_rad_s
        Optional continuous angular-rate budget. See
        ``build_delivery_schedule``.
    max_iterations
        Hard cap on iterations. With ``0``, the schedule is built once from
        the cruise-only trajectory without re-propagating.
    convergence_tol_et_s
        Max |et_start| drift across matched windows (seconds) that
        counts as converged. 1.0 s is tight enough to stabilise
        fluence scoring below 0.1% for representative windows.
    damping
        Linear-blend coefficient in [0, 1] applied to matched
        windows. 1.0 = no damping (can oscillate); 0.0 = no update and
        therefore cannot converge. 0.7 is the default.
    jaccard_threshold
        Minimum Jaccard overlap (fraction) on absolute ET to match a
        new window to an old one. 0.5 is conservative; windows that
        drift >= half their span are treated as newly-appeared.
    propagate_kwargs
        Extra kwargs forwarded to ``dynamics.propagate`` (e.g.
        ``gravity_degree``, ``third_bodies``, ``altitude_floor``).
    find_windows_kwargs
        Extra kwargs forwarded to ``find_delivery_windows`` (e.g.
        ``target_elevation_min_deg``, ``atmospheric_transmission``,
        ``alpha_max_rad_s2`` as a POST-FILTER on window feasibility
        -- distinct from the same-named kwarg here, which controls
        slew-duration auto-sizing).
    continuation_seed
        Prior repeated sol's sampled trajectory and final windows. ``None``
        selects first-sol global discovery. A supplied seed is used only when
        ``max_iterations > 0``; with zero iterations, the one-shot shortcut is
        used.
    continuation_search_margin_s
        Half-width added outside every seeded window for exact local gate
        evaluation. Both sampled band edges must be gate-closed.
    continuation_max_boundary_shift_s
        Maximum accepted endpoint motion from the seed; default is half the
        search margin, leaving the remaining half as a guard band.
    continuation_failure
        ``"full_scan"`` (default) logs the failed continuation invariant and
        runs the global finder on that propagated trajectory. ``"raise"``
        makes any continuation failure fatal.

    Returns
    -------
    RefinedSchedule

    Raises
    ------
    ValueError
        For invalid max_iterations, damping, tolerances.
    """
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    if not 0.0 <= damping <= 1.0:
        raise ValueError(f"damping must be in [0, 1], got {damping}")
    if convergence_tol_et_s <= 0.0:
        raise ValueError(
            f"convergence_tol_et_s must be > 0, got {convergence_tol_et_s}"
        )
    if not 0.0 < jaccard_threshold < 1.0:
        raise ValueError(
            f"jaccard_threshold must be in (0, 1), got {jaccard_threshold}"
        )
    if continuation_failure not in {"full_scan", "raise"}:
        raise ValueError(
            "continuation_failure must be 'full_scan' or 'raise', got "
            f"{continuation_failure!r}"
        )
    if (
        not math.isfinite(float(continuation_search_margin_s))
        or float(continuation_search_margin_s) <= 0.0
    ):
        raise ValueError(
            "continuation_search_margin_s must be a positive finite float"
        )
    if continuation_max_boundary_shift_s is not None:
        max_shift_s = float(continuation_max_boundary_shift_s)
        if (
            not math.isfinite(max_shift_s)
            or max_shift_s < 0.0
            or max_shift_s >= float(continuation_search_margin_s)
        ):
            raise ValueError(
                "continuation_max_boundary_shift_s must be finite and satisfy "
                "0 <= shift < continuation_search_margin_s"
            )

    # Local imports to avoid circular module dependencies.
    from reflectors.dynamics import propagate as _propagate

    propagate_kwargs = dict(propagate_kwargs) if propagate_kwargs else {}
    find_windows_kwargs = (
        dict(find_windows_kwargs) if find_windows_kwargs else {}
    )
    if "return_samples" in find_windows_kwargs:
        raise ValueError(
            "refine_delivery_schedule requires find_windows_kwargs to return "
            "windows only; remove return_samples"
        )
    if "search_intervals_s" in find_windows_kwargs:
        raise ValueError(
            "refine_delivery_schedule owns search_intervals_s during "
            "continuation; remove it from find_windows_kwargs"
        )
    duration_s = float(t_span_s[1] - t_span_s[0])

    # Full target list: primary first (target_idx 0), extras in order.
    all_targets: list[Tuple[float, float]] = [
        (float(target_lat_deg), float(target_lon_deg)),
        *((float(lat), float(lon)) for lat, lon in extra_targets),
    ]

    # First-sol/global path: propagate under cruise-only and discover the
    # topology. Repeated-sol path: map the prior Mars-fixed trajectory into
    # this sol and build the scheduled profile before the first propagation.
    current_profile: AttitudeCallable = cruise_profile
    windows_current: Sequence[DeliveryWindow] = ()
    drift_history: list[float] = []
    unstable_windows: list[DeliveryWindow] = []
    window_search_modes: list[str] = []
    continuation_fallback_reasons: list[str] = []
    n_propagations = 0
    result = None

    use_seeded_local_search = continuation_seed is not None and max_iterations > 0
    initialization_mode = "global_cruise_scan"
    if use_seeded_local_search:
        assert continuation_seed is not None
        try:
            windows_current = list(
                continuation_seed.shifted_windows(epoch_et, t_span_s)
            )
            r_predicted_seed = continuation_seed.position_predictor_for_epoch(
                epoch_et,
                t_span_s,
                initial_state_j2000_km_kmps=np.asarray(
                    initial_state_km_kmps, dtype=float
                ),
            )
            current_profile, current_metadata = build_delivery_schedule(
                r_predicted_seed,
                windows_current,
                float(target_lat_deg), float(target_lon_deg),
                cruise_profile=cruise_profile,
                epoch_et=epoch_et,
                duration_s=duration_s,
                slew_duration_s=slew_duration_s,
                alpha_max_rad_s2=alpha_max_rad_s2,
                omega_max_rad_s=omega_max_rad_s,
                observer_naif_id=observer_naif_id,
            )
            initialization_mode = "prior_sol_mars_fixed_seed"
        except WindowContinuationError as exc:
            if continuation_failure == "raise":
                raise
            reason = f"seed initialization failed: {exc}"
            continuation_fallback_reasons.append(reason)
            logger.warning(
                "refine_delivery_schedule: %s; falling back to global "
                "cruise-only discovery.",
                reason,
            )
            use_seeded_local_search = False
            initialization_mode = "global_cruise_scan_after_seed_rejection"

    if not use_seeded_local_search:
        result = _propagate(
            initial_state_km_kmps,
            t_span_s=t_span_s,
            epoch_et=epoch_et,
            solar_sail=sail,
            sail_normal=current_profile,
            **propagate_kwargs,
        )
        n_propagations += 1
        windows_current = list(find_delivery_windows_multi(
            result,
            all_targets,
            sail=sail,
            **find_windows_kwargs,
        ))
        window_search_modes.append("global_initial")

        cs = trajectory_interpolant(result)
        if cs is None:
            raise RuntimeError(
                "refine_delivery_schedule: trajectory_interpolant returned None "
                "(result has no epoch_et or < 4 samples). Cannot refine."
            )

        def r_predicted_fn(et: float) -> np.ndarray:
            return np.asarray(cs(float(et)), dtype=float)

        current_profile, current_metadata = build_delivery_schedule(
            r_predicted_fn,
            windows_current,
            float(target_lat_deg), float(target_lon_deg),
            cruise_profile=cruise_profile,
            epoch_et=epoch_et,
            duration_s=duration_s,
            slew_duration_s=slew_duration_s,
            alpha_max_rad_s2=alpha_max_rad_s2,
            omega_max_rad_s=omega_max_rad_s,
            observer_naif_id=observer_naif_id,
        )

    # If max_iterations == 0, short-circuit: return the one-shot
    # result (propagated under cruise-only, schedule built from
    # cruise-only windows) when the caller wants no refinement.
    converged = False
    n_iterations = 0
    if max_iterations == 0:
        assert result is not None
        return RefinedSchedule(
            final_result=result,
            final_windows=tuple(windows_current),
            unstable_windows=(),
            final_profile=current_profile,
            metadata=current_metadata,
            n_iterations=0,
            converged=False,
            boundary_drift_history_et_s=(),
            initialization_mode=initialization_mode,
            window_search_modes=tuple(window_search_modes),
            continuation_fallback_reasons=tuple(
                continuation_fallback_reasons
            ),
            n_propagations=n_propagations,
        )

    # Iterate: propagate under schedule, find new windows, damp, rebuild.
    for iteration in range(1, max_iterations + 1):
        result_new = _propagate(
            initial_state_km_kmps,
            t_span_s=t_span_s,
            epoch_et=epoch_et,
            solar_sail=sail,
            sail_normal=current_profile,
            **propagate_kwargs,
        )
        n_propagations += 1
        if use_seeded_local_search:
            try:
                continued = continue_delivery_windows_multi(
                    result_new,
                    all_targets,
                    windows_current,
                    search_margin_s=float(continuation_search_margin_s),
                    max_boundary_shift_s=continuation_max_boundary_shift_s,
                    sail=sail,
                    **find_windows_kwargs,
                )
                windows_new = list(continued.windows)
                window_search_modes.append("local_guarded")
            except WindowContinuationError as exc:
                if continuation_failure == "raise":
                    raise
                reason = f"iteration {iteration}: {exc}"
                continuation_fallback_reasons.append(reason)
                logger.warning(
                    "refine_delivery_schedule: local continuation failed (%s); "
                    "running full-window search on the same trajectory.",
                    exc,
                )
                windows_new = list(find_delivery_windows_multi(
                    result_new,
                    all_targets,
                    sail=sail,
                    **find_windows_kwargs,
                ))
                window_search_modes.append("global_fallback")
        else:
            windows_new = list(find_delivery_windows_multi(
                result_new,
                all_targets,
                sail=sail,
                **find_windows_kwargs,
            ))
            window_search_modes.append("global_iteration")
        drift = _max_boundary_drift(
            windows_current, windows_new,
            jaccard_threshold=jaccard_threshold,
        )
        drift_history.append(drift)
        n_iterations = iteration

        if drift < convergence_tol_et_s:
            converged = True
            windows_current = windows_new
            result = result_new
            unstable_windows = []
            break

        # Not converged: apply damping and rebuild.
        windows_damped, unstable_windows = _apply_damping(
            windows_current, windows_new,
            damping=damping, jaccard_threshold=jaccard_threshold,
        )
        windows_current = windows_damped
        result = result_new

        cs = trajectory_interpolant(result)
        if cs is None:
            raise RuntimeError(
                f"refine_delivery_schedule: iteration {iteration} "
                f"produced a result without a valid trajectory interpolant."
            )

        def r_predicted_fn_iter(et: float) -> np.ndarray:
            return np.asarray(cs(float(et)), dtype=float)

        current_profile, current_metadata = build_delivery_schedule(
            r_predicted_fn_iter,
            windows_current,
            float(target_lat_deg), float(target_lon_deg),
            cruise_profile=cruise_profile,
            epoch_et=epoch_et,
            duration_s=duration_s,
            slew_duration_s=slew_duration_s,
            alpha_max_rad_s2=alpha_max_rad_s2,
            omega_max_rad_s=omega_max_rad_s,
            observer_naif_id=observer_naif_id,
        )

    # Align the returned profile with the final propagated trajectory. Profiles
    # built inside the loop use the preceding trajectory estimate, so a final
    # rebuild removes the one-iteration lag in Hermite endpoint rates.
    cs_final = trajectory_interpolant(result)
    if cs_final is not None:
        def r_predicted_fn_final(et: float) -> np.ndarray:
            return np.asarray(cs_final(float(et)), dtype=float)
        current_profile, current_metadata = build_delivery_schedule(
            r_predicted_fn_final,
            windows_current,
            float(target_lat_deg), float(target_lon_deg),
            cruise_profile=cruise_profile,
            epoch_et=epoch_et,
            duration_s=duration_s,
            slew_duration_s=slew_duration_s,
            alpha_max_rad_s2=alpha_max_rad_s2,
            omega_max_rad_s=omega_max_rad_s,
            observer_naif_id=observer_naif_id,
        )

    return RefinedSchedule(
        # ``result`` is assigned by the cruise discovery propagation or by the
        # first scheduled propagation in every max_iterations > 0 seed path.
        final_result=result,
        final_windows=tuple(windows_current),
        unstable_windows=tuple(unstable_windows),
        final_profile=current_profile,
        metadata=current_metadata,
        n_iterations=n_iterations,
        converged=converged,
        boundary_drift_history_et_s=tuple(drift_history),
        initialization_mode=initialization_mode,
        window_search_modes=tuple(window_search_modes),
        continuation_fallback_reasons=tuple(
            continuation_fallback_reasons
        ),
        n_propagations=n_propagations,
    )
