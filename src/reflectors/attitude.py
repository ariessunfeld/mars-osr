"""Kinematic attitude profile layer for a Mars solar sail.

This module represents the commanded sail-normal orientation
``n_hat(t)`` as a pure function of time, with primitive factories and a
piecewise composition operator, and provides diagnostic kinematics
(``omega``, ``alpha``) plus a peak-angular-acceleration bound check. This
module provides the structural pieces needed to design multi-orbit attitude
programs for the Mars reflector scenario.

The sail attitude subsystem is kinematic, not rigid-body-dynamic. Under the
model's scoping assumptions
(3-axis control with unbounded torque authority up to ``alpha_max``,
CoP = CoM, homogeneous symmetric mass distribution), Euler's equation
``I omega_dot + omega x I omega = tau`` becomes an identity: the
actuator implicitly produces whatever ``tau`` realizes a commanded
``alpha(t)``, so the inertia tensor drops out of everything observable
in the simulation. The angular acceleration constraint
``|alpha(t)| <= alpha_max`` is enforced as a path-level diagnostic on
the profile itself, independent of integrator step placement.

Reference (primary): Viale, A., Celik, O., Oderinwale, T., Sulbhewar, L.,
McInnes, C.R. (2023), "A reference architecture for orbiting solar
reflectors to enhance terrestrial solar power plant output", *Advances
in Space Research* 72, 1304-1348. The formulation here is a strict subset
of Viale 2023 §4.2.2: Eqs. (37) and (40) derive ``omega`` and
``omega_dot`` by direct differentiation of the commanded TRF orientation,
matching the central-difference evaluation
of the attitude callable. Their §4.3 (CMG actuator sizing) is
deliberately NOT reproduced -- that is where rigid-body Euler, the
inertia tensor, and gravity-gradient torques enter. This implementation
stops at the kinematic layer.

Profile signature.

    AttitudeCallable = Callable[[ndarray(3,), float], ndarray(3,)]
    # (r_sat_j2000_km, et) -> unit_vec_j2000

Matches what ``reflectors.srp.srp_acceleration`` already consumes. No
state is carried; a profile is a pure function of (sail position, TDB
epoch).

Quintic slew formulation. ``smooth_slew`` realises a rest-to-rest
reorientation between two static sail normals ``n_0`` and ``n_f`` over
an interval ``[t_0, t_f]``. The rotation is a single-axis great-circle
arc about

    e_hat = (n_0 x n_f) / |n_0 x n_f|,    theta_total = arccos(n_0 . n_f)

with the scalar angle parameterised by a degree-five polynomial

    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,    tau = (t - t_0) / T

that satisfies six boundary conditions by construction: ``s(0) = 0,
s(1) = 1, s'(0) = s'(1) = 0, s''(0) = s''(1) = 0``. The actual sail
normal is recovered by Rodrigues rotation:

    n_hat(t) = n_0 cos(theta(t)) + (e_hat x n_0) sin(theta(t))

(the third Rodrigues term vanishes because ``e_hat . n_0 = 0`` by
construction). The peak angular rate and angular acceleration on the
arc are analytic,

    |omega|_max = theta_total * (15/8) / T
    |alpha|_max = theta_total * (10/sqrt(3)) / T^2

reached at ``tau = 1/2`` and ``tau = (3 -/+ sqrt(3))/6`` respectively.
These are pinned by the test suite as closed-form anchors.

S^2 kinematic identities. For a unit vector ``n_hat(t)`` parameterising
the sail normal,

    omega(t) = n_hat(t) x dn_hat/dt                                (K1)
    alpha(t) = n_hat(t) x d^2 n_hat/dt^2                           (K2)

both valid under the ``omega . n_hat = 0`` convention (roll about the
sail normal is unobservable for the CoP = CoM flat-sail scope of this
project). (K1) follows from ``dn/dt = omega x n`` and crossing with
``n``. (K2) follows from differentiating (K1) and using
``dn/dt x dn/dt = 0``. The diagnostics ``angular_rate`` and
``angular_acceleration`` implement (K1) and (K2) by central-difference
evaluation of the attitude callable on a caller-selected grid, with step
size ``dt`` decoupled from any integrator step placement -- the profile
is a signal, not a state.

Out of scope for this kinematic model:
Rigid-body rotational dynamics, inertia tensor, gravity-gradient torque,
SRP torque from CoP/CoM offset, attitude actuator modelling, residual-
momentum desaturation.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import spiceypy as spice

from reflectors.ephemeris import sun_state_j2000


logger = logging.getLogger(__name__)


SUN_NAIF_ID = 10
MARS_NAIF_ID = 499


# Signature: (r_sat_j2000_km, et_tdb_s) -> unit-vector in J2000 axes, shape (3,).
# Arbitrary closures can be built and passed in; the helpers below cover the
# standard static, sun-tracking, and slew cases needed by the SRP and
# reflection-delivery layers.
AttitudeCallable = Callable[[np.ndarray, float], np.ndarray]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def fixed_j2000(n_hat_j2000: np.ndarray) -> AttitudeCallable:
    """Attitude callable for a sail whose normal is fixed in J2000 axes.

    Parameters
    ----------
    n_hat_j2000
        Any non-zero vector, shape (3,). Stored as a unit vector inside
        the closure; the callable returns it verbatim regardless of
        state or epoch.

    Raises
    ------
    ValueError
        If ``n_hat_j2000`` is the zero vector.
    """
    v = np.asarray(n_hat_j2000, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("fixed_j2000: n_hat_j2000 is the zero vector")
    unit = v / norm

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        return unit

    return _n_hat


def sun_pointing(observer_naif_id: int = MARS_NAIF_ID) -> AttitudeCallable:
    """Attitude callable that tracks the Sun: ``n_hat(r, et) = s_hat(r, et)``.

    Sun-pointing attitude maximises radial (anti-sunward) thrust and is
    the canonical "idle" attitude between delivery windows. The callable fetches
    the Sun position fresh via ``spkezr`` on every evaluation so the
    propagator inherits exact time-dependence of the Sun direction.

    Parameters
    ----------
    observer_naif_id
        Central body NAIF id for the SPICE query. Default 499 (Mars
        planet centre), matching ``reflectors.srp``.
    """

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        state = sun_state_j2000(et, observer_naif_id)
        sat_to_sun = np.asarray(state[:3], dtype=float) - np.asarray(
            r_sat_km, dtype=float
        )
        d = float(np.linalg.norm(sat_to_sun))
        return sat_to_sun / d

    return _n_hat


# ---------------------------------------------------------------------------
# Uncommanded attitude when no pointing command is active.
#
# The primitives above are all COMMANDED -- they realise a pointing intent. The
# two below describe a sail that is not being controlled. Both are exact,
# smooth, deterministic functions of epoch: no randomness at evaluation time,
# because a discontinuous
# or stochastic right-hand side destroys the adaptive step-size control in
# DOP853 (the answer stops converging and stops being reproducible). Any
# "randomness" is drawn once, from a seed, when the closure is built.
# ---------------------------------------------------------------------------


def _rodrigues(v: np.ndarray, axis_hat: np.ndarray, theta: float) -> np.ndarray:
    """Rotate ``v`` about the unit vector ``axis_hat`` by ``theta`` radians.

    Full Rodrigues formula (all three terms, unlike the reduced form in
    :func:`smooth_slew` which exploits ``axis . v == 0``):

        v_rot = v cos(theta) + (axis x v) sin(theta)
                + axis (axis . v) (1 - cos(theta))

    Reference: any rigid-body kinematics text; e.g. Markley & Crassidis (2014),
    *Fundamentals of Spacecraft Attitude Determination and Control*, Eq. (2.60)
    p. 33 (the "Euler axis/angle" rotation of a vector).
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return (
        v * c
        + np.cross(axis_hat, v) * s
        + axis_hat * float(np.dot(axis_hat, v)) * (1.0 - c)
    )


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError(f"{what} is the zero vector")
    return v / norm


def mars_pole_j2000(et: float) -> np.ndarray:
    """Mars spin-pole direction as a J2000 unit vector at epoch ``et``.

    Thin accessor over ``elements.mme2000_rotation_from_j2000``, whose third
    row IS the pole in J2000 (extracted from the full ``IAU_MARS`` transform,
    so all IAU-2015 nutation harmonics are included -- see that function's
    docstring explaining the ~1.5 deg error in the polynomial-only pole).
    """
    from reflectors.elements import mme2000_rotation_from_j2000

    return np.asarray(mme2000_rotation_from_j2000(et)[2], dtype=float)


def orbit_frame_fixed(
    n_hat_rtn: np.ndarray,
    h_hat_0_j2000: np.ndarray,
    epoch_et_ref: float,
    *,
    node_rate_rad_per_s: Optional[float] = None,
) -> AttitudeCallable:
    """Sail normal FIXED in the orbit (RTN) frame, so it turns 1x per orbit.

    The sail holds a constant orientation relative to its own orbit -- e.g.
    always nadir-pointing, always transverse, always orbit-normal. In INERTIAL
    space that is a rotation once per orbital period, which is the plausible
    behaviour of a sail that is stabilised but not steered.

    Frame. ``(R, T, N)`` with ``R = r_hat`` (instantaneous radial),
    ``N = h_hat(et)`` (orbit normal) and ``T = N x R`` (transverse). This is
    the standard RTN triad, exact at any eccentricity; ``T`` coincides with the
    along-track direction only for a circular orbit.

    Orbit-normal input. An
    ``AttitudeCallable`` receives ``(r_sat_km, et)`` -- position only, no
    velocity, so ``h = r x v`` is not available at evaluation time. The
    normal is therefore pinned from the initial state by the caller and
    advanced analytically: for a sun-synchronous orbit the node precesses at
    exactly ``2 pi / MARS_SIDEREAL_YEAR_S`` about the Mars pole (that IS the
    defining sun-sync identity, see ``sun_sync.sun_sync_inclination_rad``), so

        h_hat(et) = Rodrigues(pole, node_rate * (et - epoch_et_ref)) @ h_hat_0

    The pole is evaluated once at ``epoch_et_ref`` and held fixed in the
    closure: it is itself precessing, but only by ~arcsec over a multi-year
    mission (noted in ``sun_sync``), which is negligible against the ~deg-per-
    day node motion, and re-deriving it would cost a ``pxform`` per RHS call.

    Parameters
    ----------
    n_hat_rtn
        Desired sail normal as components ``(c_R, c_T, c_N)`` in the RTN
        basis. Normalised internally. E.g. ``(1, 0, 0)`` = nadir/zenith-
        pointing, ``(0, 1, 0)`` = transverse, ``(0, 0, 1)`` = orbit-normal.
    h_hat_0_j2000
        Orbit normal at ``epoch_et_ref``, J2000 axes; typically
        ``np.cross(r0, v0)`` from the initial state. Normalised internally.
    epoch_et_ref
        Epoch at which ``h_hat_0_j2000`` is valid.
    node_rate_rad_per_s
        Nodal precession rate about the Mars pole. ``None`` (default) uses the
        sun-synchronous rate ``2 pi / MARS_SIDEREAL_YEAR_S``, which is correct
        for every orbit in this study; pass ``0.0`` to freeze the plane, or an
        explicit J2 rate for a non-sun-sync orbit.

    Raises
    ------
    ValueError
        If either vector is zero, or if ``h_hat_0`` is parallel to the radial
        direction at evaluation time (degenerate triad).
    """
    from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S

    c = _unit(n_hat_rtn, "orbit_frame_fixed: n_hat_rtn")
    h0 = _unit(h_hat_0_j2000, "orbit_frame_fixed: h_hat_0_j2000")
    rate = (2.0 * math.pi / MARS_SIDEREAL_YEAR_S
            if node_rate_rad_per_s is None else float(node_rate_rad_per_s))
    pole = _unit(mars_pole_j2000(epoch_et_ref), "orbit_frame_fixed: Mars pole")

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        h_hat = (h0 if rate == 0.0
                 else _rodrigues(h0, pole, rate * (et - epoch_et_ref)))
        r_hat = _unit(r_sat_km, "orbit_frame_fixed: r_sat_km")
        t_vec = np.cross(h_hat, r_hat)
        t_norm = float(np.linalg.norm(t_vec))
        if t_norm < 1e-12:
            raise ValueError(
                "orbit_frame_fixed: orbit normal is parallel to the radial "
                f"direction at et={et}; RTN triad is degenerate"
            )
        t_hat = t_vec / t_norm
        # h_hat is only approximately perpendicular to r_hat once the orbit has
        # evolved, so rebuild N from the actual R,T pair to keep the triad
        # orthonormal to machine precision.
        n_axis = np.cross(r_hat, t_hat)
        out = c[0] * r_hat + c[1] * t_hat + c[2] * n_axis
        return out / float(np.linalg.norm(out))

    return _n_hat


def tumble(
    spin_rate_rad_per_s: float,
    epoch_et_ref: float,
    *,
    n_hat_0: Optional[np.ndarray] = None,
    spin_axis: Optional[np.ndarray] = None,
    precession_rate_rad_per_s: float = 0.0,
    precession_axis: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> AttitudeCallable:
    """Uncommanded tumble: smooth rotation with no reference to Sun or orbit.

    ``n_hat(et) = R(p_hat, w_p dt) @ R(e_hat, w_s dt) @ n_hat_0``, with
    ``dt = et - epoch_et_ref``. Exact, smooth and infinitely differentiable in
    ``et``, so the propagator's error control still works; deterministic given
    the seed, so a run is reproducible.

    SINGLE-AXIS vs SPHERE-COVERING -- read before using this for the
    "rapid tumbling averages SRP" case. With ``precession_rate_rad_per_s = 0``
    this is a pure spin about one fixed axis, and ``n_hat`` traces a CIRCLE on
    the unit sphere at constant angle from ``spin_axis``. Its time-average is
    therefore NOT the uniform-over-the-sphere average that
    ``srp.TumbleAveragedSail`` implements -- it depends on the spin-axis
    geometry relative to the Sun-line. Adding an incommensurate
    ``precession_rate_rad_per_s`` about a second axis sweeps that circle around
    and covers a 2-D region of the sphere, which is much closer to (but still
    not exactly) the uniform measure. This is distinct from a uniform
    orientation average.

    Note that a real torque-free rigid body does not sample orientations
    uniformly either: for an axisymmetric body the symmetry axis precesses on a
    cone about the (conserved) angular-momentum vector, so it stays at fixed
    angle from it. "Uniform orientation" is an idealisation of tumbling, not a
    consequence of it.

    Parameters
    ----------
    spin_rate_rad_per_s
        Angular rate of the primary rotation. For approximately one revolution
        per orbit, pass ``2 pi / T_orbit``.
    epoch_et_ref
        Epoch at which the phase is zero (``n_hat(epoch_et_ref) = n_hat_0``).
    n_hat_0, spin_axis, precession_axis
        Optional explicit geometry, J2000 axes; each normalised internally.
        Any left as ``None`` is drawn from the seeded RNG as an isotropic
        random direction -- "uninformed", i.e. deliberately unrelated to the
        Sun direction or the orbit plane.
    precession_rate_rad_per_s
        Rate of the secondary rotation. ``0.0`` (default) gives a pure
        single-axis spin.
    seed
        Seed for any geometry not given explicitly. Required if anything is
        left to chance, so that a run can be reproduced exactly.

    Raises
    ------
    ValueError
        If a vector is left to the RNG but no ``seed`` was supplied, or if a
        supplied vector is zero, or if ``spin_axis`` is parallel to
        ``n_hat_0`` (the spin would then be a no-op).
    """
    needs_rng = (n_hat_0 is None or spin_axis is None
                 or (precession_rate_rad_per_s != 0.0
                     and precession_axis is None))
    if needs_rng and seed is None:
        raise ValueError(
            "tumble: seed is required when n_hat_0 / spin_axis / "
            "precession_axis are left to the RNG, so the run is reproducible"
        )
    rng = np.random.default_rng(seed)

    def _draw(name: str) -> np.ndarray:
        # Isotropic direction: normalised Gaussian (the standard construction).
        return _unit(rng.normal(size=3), f"tumble: drawn {name}")

    n0 = (_draw("n_hat_0") if n_hat_0 is None
          else _unit(n_hat_0, "tumble: n_hat_0"))
    e_hat = (_draw("spin_axis") if spin_axis is None
             else _unit(spin_axis, "tumble: spin_axis"))
    if abs(float(np.dot(e_hat, n0))) > 1.0 - 1e-12:
        raise ValueError(
            "tumble: spin_axis is parallel to n_hat_0, so the spin would not "
            "move the sail normal at all"
        )
    w_s = float(spin_rate_rad_per_s)
    w_p = float(precession_rate_rad_per_s)
    p_hat = None
    if w_p != 0.0:
        p_hat = (_draw("precession_axis") if precession_axis is None
                 else _unit(precession_axis, "tumble: precession_axis"))

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        dt = et - epoch_et_ref
        out = _rodrigues(n0, e_hat, w_s * dt)
        if p_hat is not None:
            out = _rodrigues(out, p_hat, w_p * dt)
        # Rodrigues is norm-preserving analytically; renormalise so long dt
        # cannot let round-off accumulate into a non-unit normal.
        return out / float(np.linalg.norm(out))

    return _n_hat


# Peak-magnitude coefficients of the rest-to-rest quintic ``s(tau) =
# 10 tau^3 - 15 tau^4 + 6 tau^5``. Derivation:
#   s'(tau)  = 30 tau^2 (1 - tau)^2            -> max at tau = 1/2, = 15/8.
#   s''(tau) = 60 tau (1 - tau) (1 - 2 tau)    -> |max| at tau = (3 -/+ sqrt(3))/6,
#                                                 = 10/sqrt(3).
_QUINTIC_OMEGA_PEAK_COEFF = 15.0 / 8.0
_QUINTIC_ALPHA_PEAK_COEFF = 10.0 / math.sqrt(3.0)

# Sin(theta) floor for detecting near-antipodal endpoints (below which the
# cross-product rotation axis is not well-defined in double precision). At
# 1e-10, the boundary between "accepted near-antipodal" and "rejected" sits
# at an angle of pi - 2e-10 rad, comfortably deep into the region where
# numerical conditioning has already collapsed.
_ANTIPODAL_SIN_TOL = 1.0e-10


# R^3 Hermite quintic projection-stability floor. For c(t) traversing a
# degree-5 polynomial from n_0 to n_f with endpoint tangents
# v_0 = T*(omega_0 x n_0), v_f = T*(omega_f x n_f), min|c(t)| along the
# path can contract well below 1 when either the total rotation angle
# approaches pi (chord length approaches 2) or the endpoint tangent kicks
# push the curve past the sphere's surface. Below |c|~0.3, the projection
# f(c) = c/|c| becomes ill-conditioned: |d^2 n_hat / dt^2| carries large
# centripetal terms that swamp the tangential alpha the commanded profile
# was meant to deliver. Callers past this threshold should decompose into
# sub-slews with intermediate static endpoints. The threshold covers the
# representative delivery geometries exercised by the tests.
_HERMITE_MIN_MAG = 0.3

# Number of uniform tau samples used to validate the projection guard at
# construction time. 200 points gives sub-percent detection resolution on
# the min|c| minimum without measurable construction cost.
_HERMITE_GUARD_SAMPLES = 200

# Dot-product floor for near-antipodal endpoints in the Hermite primitive.
# Tighter than _ANTIPODAL_SIN_TOL because the R^3 chord passes near the
# origin at antipodes, so rejection occurs before the projection
# becomes ill-conditioned. Corresponds to cos(theta) < -1 + 1e-8, i.e.
# theta > pi - ~1.4e-4 rad (179.99 deg).
_HERMITE_ANTIPODAL_COS_FLOOR = -1.0 + 1.0e-8


def smooth_slew(
    t0_et: float,
    tf_et: float,
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
) -> AttitudeCallable:
    """Quintic-profile rest-to-rest single-axis slew between two sail normals.

    Realises a single-axis great-circle rotation from ``n_hat_0`` to
    ``n_hat_f`` over ``[t0_et, tf_et]`` with zero angular rate and zero
    angular acceleration at both endpoints (6 boundary conditions met
    by a degree-5 polynomial in scaled time). See module docstring for
    the formulation and analytic peaks.

    Parameters
    ----------
    t0_et, tf_et
        SPICE TDB seconds, with ``tf_et > t0_et``.
    n_hat_0, n_hat_f
        Sail normals at the start and end of the slew. Re-normalised
        internally; sign-sensitive.

    Raises
    ------
    ValueError
        If ``tf_et <= t0_et``; if either input is the zero vector; or
        if ``n_hat_0`` and ``n_hat_f`` are (within numerical tolerance)
        antipodal, in which case the rotation axis is ambiguous and
        the caller should decompose into two sequential slews.

    Notes
    -----
    A query at ``et`` strictly outside ``[t0_et, tf_et]`` raises; use
    ``piecewise`` to compose with adjacent idle / slew / tracking
    segments.

    Near-identical endpoints (angle below ``arcsin(1e-10)`` rad)
    degenerate to a fixed-attitude callable at ``n_hat_0``; this is
    not an error because a zero-amplitude slew is legal.
    """
    if tf_et <= t0_et:
        raise ValueError(
            f"smooth_slew: tf_et must exceed t0_et, got t0={t0_et}, tf={tf_et}"
        )
    n0 = np.asarray(n_hat_0, dtype=float)
    nf = np.asarray(n_hat_f, dtype=float)
    norm0 = float(np.linalg.norm(n0))
    normf = float(np.linalg.norm(nf))
    if norm0 == 0.0 or normf == 0.0:
        raise ValueError("smooth_slew: endpoint n_hat is the zero vector")
    n0 = n0 / norm0
    nf = nf / normf

    # Rotation axis: e_hat = (n0 x nf) / |n0 x nf|.
    cross = np.cross(n0, nf)
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < _ANTIPODAL_SIN_TOL:
        cos_total = float(np.dot(n0, nf))
        if cos_total > 0.0:
            # Near-parallel endpoints -> zero slew, return the fixed
            # attitude at n0 throughout the interval. Still domain-
            # restricted to [t0_et, tf_et] for composition consistency.
            def _n_hat_degenerate(r_sat_km: np.ndarray, et: float) -> np.ndarray:
                if et < t0_et or et > tf_et:
                    raise ValueError(
                        "smooth_slew: et outside domain "
                        f"[{t0_et}, {tf_et}] (degenerate slew): et={et}"
                    )
                return n0
            return _n_hat_degenerate
        raise ValueError(
            "smooth_slew: endpoints are (near-)antipodal; rotation axis "
            "is ambiguous. Decompose into two sequential slews with an "
            "intermediate normal."
        )

    # Clamp cos for safety at the extremes of arccos (n0 . nf must be in
    # [-1, 1], but round-off can push it slightly outside).
    cos_total = max(-1.0, min(1.0, float(np.dot(n0, nf))))
    theta_total = math.acos(cos_total)

    e_hat = cross / cross_norm
    # In-plane perpendicular to n0, lying in the span{n0, nf} plane.
    # |m_hat| = |e_hat x n0| = 1 since e_hat is a unit vector perpendicular
    # to n0.
    m_hat = np.cross(e_hat, n0)

    period = tf_et - t0_et

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        if et < t0_et or et > tf_et:
            raise ValueError(
                f"smooth_slew: et outside domain [{t0_et}, {tf_et}]: et={et}"
            )
        tau = (et - t0_et) / period
        # s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 via Horner-ish nesting.
        s_tau = tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)
        theta = theta_total * s_tau
        return math.cos(theta) * n0 + math.sin(theta) * m_hat

    return _n_hat


def smooth_slew_hermite(
    t0_et: float,
    tf_et: float,
    n_hat_0: np.ndarray,
    n_hat_f: np.ndarray,
    *,
    omega_0_rad_s: Optional[np.ndarray] = None,
    omega_f_rad_s: Optional[np.ndarray] = None,
) -> AttitudeCallable:
    """Dynamic-endpoint slew: R^3 quintic Hermite with unit projection.

    Extends ``smooth_slew`` to nonzero endpoint angular velocities. Used
    when the slew must take the baton from a non-static cruise (e.g. a
    bisector-tracking segment with ``omega`` up to ~2e-3 rad/s at Mars
    LMO) without a velocity discontinuity at the handoff.

    Formulation.
    ``c(t) in R^3`` is a degree-5 polynomial in normalised time
    ``tau = (t - t0) / T`` with 6 boundary conditions:

        c(0)   = n_0,          c(T)   = n_f
        c'(0)  = omega_0 x n_0, c'(T)  = omega_f x n_f
        c''(0) = 0,            c''(T) = 0

    where primes denote d/dt. Explicit coefficients (derivation in the
    module docstring section "Hermite quintic derivation"):

        p_0 = n_0
        p_1 = T * (omega_0 x n_0)
        p_2 = 0
        p_3 = 10 * (n_f - n_0) - 6 * v_0 - 4 * v_f
        p_4 = -15 * (n_f - n_0) + 8 * v_0 + 7 * v_f
        p_5 = 6 * (n_f - n_0) - 3 * v_0 - 3 * v_f

    with v_0 = T * (omega_0 x n_0), v_f = T * (omega_f x n_f). The
    sail normal is recovered as n_hat(t) = c(t) / |c(t)|.

    Exact endpoint matching. At ``t = t0``, ``|c| = 1`` (since
    ``n_hat_0`` was pre-normalised), and ``c . c' = n_0 . (omega_0 x n_0)
    = 0`` by the triple-product identity. Differentiating the projection
    ``n_hat = c / |c|`` gives
    ``dn_hat/dt = c' / |c| - c (c . c') / |c|^3``, which collapses at
    ``t = t0`` to ``dn_hat/dt = omega_0 x n_0`` -- exactly the desired
    S^2 tangent satisfying the kinematic constraint omega_0 . n_0 = 0.
    Same argument at ``t = tf``.

    Projection stability guard. For large rotation angles (theta_total
    near pi) or large endpoint tangent kicks (|omega| * T near 1), c(tau)
    can approach the origin, at which point the projection
    n_hat = c/|c| becomes ill-conditioned. A 200-sample sweep of tau in
    [0, 1] validates min|c(tau)| >= ``_HERMITE_MIN_MAG`` (= 0.3) at
    construction time; below this, ``ValueError`` with a decomposition
    hint is raised. Near-antipodal endpoints (cos(theta) below
    ``_HERMITE_ANTIPODAL_COS_FLOOR``) raise earlier with a cleaner
    message since the chord trajectory literally passes through the
    origin.

    Not bit-for-bit equivalent to ``smooth_slew`` when
    ``omega_0 = omega_f = 0``. Both primitives match n_hat at both
    endpoints and at tau = 0.5 (by symmetry of the rest-to-rest
    quintic), but between those points the Hermite traverses a STRAIGHT
    CHORD of R^3 length ``2 sin(theta_total / 2)`` (reparameterised by
    the quintic ``s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5``), while
    ``smooth_slew`` traverses a GREAT CIRCLE ARC. The peak |omega| and
    |alpha| differ (though both bounded); callers asking for analytic
    peaks should use ``smooth_slew`` when a static-endpoint slew is
    sufficient.

    Parameters
    ----------
    t0_et, tf_et
        SPICE TDB seconds, with ``tf_et > t0_et``.
    n_hat_0, n_hat_f
        Sail normals at the start and end of the slew. Re-normalised
        internally; sign-sensitive.
    omega_0_rad_s, omega_f_rad_s
        Angular velocities (rad/s, 3-vector in J2000) at the start and
        end of the slew. ``None`` = zero (static endpoint). Only the
        in-plane component (orthogonal to n_hat at that endpoint) drives
        the kinematics; any tiny component along n_hat is absorbed
        by the cross product.

    Raises
    ------
    ValueError
        If ``tf_et <= t0_et``; either n_hat is the zero vector; endpoints
        are near-antipodal; or the projection guard fires
        (``min|c(tau)| < _HERMITE_MIN_MAG``) because the chord dipped
        too close to the origin for a well-conditioned unit projection.

    Notes
    -----
    Construction cost is O(_HERMITE_GUARD_SAMPLES); query cost is O(1)
    (polynomial evaluation + single norm). Intended for use inside
    ``build_delivery_schedule`` where the schedule has O(10) slews per
    sol, not inside the RHS hot loop.
    """
    if tf_et <= t0_et:
        raise ValueError(
            f"smooth_slew_hermite: tf_et must exceed t0_et, "
            f"got t0={t0_et}, tf={tf_et}"
        )
    n0 = np.asarray(n_hat_0, dtype=float)
    nf = np.asarray(n_hat_f, dtype=float)
    norm0 = float(np.linalg.norm(n0))
    normf = float(np.linalg.norm(nf))
    if norm0 == 0.0 or normf == 0.0:
        raise ValueError(
            "smooth_slew_hermite: endpoint n_hat is the zero vector"
        )
    n0 = n0 / norm0
    nf = nf / normf

    cos_total = float(np.dot(n0, nf))
    if cos_total < _HERMITE_ANTIPODAL_COS_FLOOR:
        raise ValueError(
            "smooth_slew_hermite: endpoints are (near-)antipodal "
            f"(cos(theta) = {cos_total:.6f}); the R^3 chord passes "
            "through the origin, making the unit projection ill-defined. "
            "Decompose into two sequential slews with an intermediate "
            "normal."
        )

    T = float(tf_et - t0_et)

    # Endpoint R^3 tangents.
    if omega_0_rad_s is None:
        v_0 = np.zeros(3)
    else:
        omega_0 = np.asarray(omega_0_rad_s, dtype=float)
        v_0 = T * np.cross(omega_0, n0)
    if omega_f_rad_s is None:
        v_f = np.zeros(3)
    else:
        omega_f = np.asarray(omega_f_rad_s, dtype=float)
        v_f = T * np.cross(omega_f, nf)

    # Hermite quintic coefficients.
    delta = nf - n0
    p_0 = n0
    p_1 = v_0
    p_3 = 10.0 * delta - 6.0 * v_0 - 4.0 * v_f
    p_4 = -15.0 * delta + 8.0 * v_0 + 7.0 * v_f
    p_5 = 6.0 * delta - 3.0 * v_0 - 3.0 * v_f

    # Projection stability guard.
    tau_grid = np.linspace(0.0, 1.0, _HERMITE_GUARD_SAMPLES)
    tau_grid_3 = tau_grid ** 3
    tau_grid_4 = tau_grid_3 * tau_grid
    tau_grid_5 = tau_grid_4 * tau_grid
    c_grid = (
        p_0[np.newaxis, :]
        + np.outer(tau_grid, p_1)
        + np.outer(tau_grid_3, p_3)
        + np.outer(tau_grid_4, p_4)
        + np.outer(tau_grid_5, p_5)
    )
    c_mag = np.linalg.norm(c_grid, axis=1)
    min_c_mag = float(np.min(c_mag))
    if min_c_mag < _HERMITE_MIN_MAG:
        raise ValueError(
            f"smooth_slew_hermite: projection ill-conditioned "
            f"(min|c(tau)| = {min_c_mag:.4f} < {_HERMITE_MIN_MAG}). "
            f"Decompose into sub-slews with intermediate static "
            f"endpoints, or increase slew duration to reduce the "
            f"endpoint tangent kick |omega| * T."
        )

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        if et < t0_et or et > tf_et:
            raise ValueError(
                f"smooth_slew_hermite: et outside domain "
                f"[{t0_et}, {tf_et}]: et={et}"
            )
        tau = (et - t0_et) / T
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau
        c = p_0 + p_1 * tau + p_3 * tau3 + p_4 * tau4 + p_5 * tau5
        mag = float(np.linalg.norm(c))
        return c / mag

    return _n_hat


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


# Numerical tolerance for interval contiguity in piecewise(). Absolute TDB
# seconds; 1e-6 s is well below any physical timescale of interest (fastest
# n_hat variation in a Mars LMO bisector is ~milliseconds, making one microsecond
# is 1000x smaller). Hardcoded because the field already carries SPICE ET
# precision (~microseconds at best).
_PIECEWISE_CONTIG_TOL = 1.0e-6


def piecewise(
    segments: Sequence[Tuple[float, float, AttitudeCallable]],
) -> AttitudeCallable:
    """Top-level attitude profile as a time-ordered sequence of sub-profiles.

    Each segment ``(t_i, t_{i+1}, profile_i)`` covers the half-open
    interval ``[t_i, t_{i+1})``, except the last which is closed
    ``[t_{N-1}, t_N]``, so every et in the overall covered range belongs
    to exactly one segment. Segments must be contiguous (no gaps, no
    overlaps) and listed in order of increasing ``t_i``.

    Parameters
    ----------
    segments
        Sequence of ``(t0_et, tf_et, profile)`` tuples.

    Raises
    ------
    ValueError
        If ``segments`` is empty, a segment has non-positive duration,
        consecutive segments do not join within
        ``_PIECEWISE_CONTIG_TOL`` seconds, or a query et is outside the
        overall covered range.
    """
    if len(segments) == 0:
        raise ValueError("piecewise: segments list is empty")

    segs = list(segments)
    for i, (t0, tf, _prof) in enumerate(segs):
        if tf <= t0:
            raise ValueError(
                f"piecewise: segment {i} has non-positive duration "
                f"[{t0}, {tf}]"
            )
    for i in range(len(segs) - 1):
        t_end_i = segs[i][1]
        t_start_next = segs[i + 1][0]
        if abs(t_end_i - t_start_next) > _PIECEWISE_CONTIG_TOL:
            raise ValueError(
                f"piecewise: gap or overlap between segment {i} "
                f"(ends at {t_end_i}) and segment {i + 1} "
                f"(starts at {t_start_next})"
            )

    t_range_min = segs[0][0]
    t_range_max = segs[-1][1]
    # Boundaries array has length N+1 for N segments.
    boundaries = np.array(
        [s[0] for s in segs] + [segs[-1][1]], dtype=float
    )

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        if et < t_range_min or et > t_range_max:
            raise ValueError(
                f"piecewise: et={et} outside covered range "
                f"[{t_range_min}, {t_range_max}]"
            )
        # et belongs to segment i iff boundaries[i] <= et < boundaries[i+1]
        # for i < N-1, or boundaries[N-1] <= et <= boundaries[N] for i=N-1.
        idx = int(np.searchsorted(boundaries, et, side="right")) - 1
        # Clamp to the last segment for et == t_range_max.
        if idx >= len(segs):
            idx = len(segs) - 1
        if idx < 0:
            idx = 0
        _, _, prof = segs[idx]
        return prof(r_sat_km, et)

    return _n_hat


# ---------------------------------------------------------------------------
# Kinematic diagnostics
# ---------------------------------------------------------------------------

# One-argument trajectory callable: r_sat_fn(et) -> ndarray(3,) in J2000 km.
# Diagnostic-only type alias; the module does not export it publicly to keep
# the API surface narrow.
_TrajectoryCallable = Callable[[float], np.ndarray]


def _three_point_normals(
    profile: AttitudeCallable,
    r_sat_fn: _TrajectoryCallable,
    et: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the profile at et-dt, et, et+dt. Internal helper."""
    r_minus = np.asarray(r_sat_fn(et - dt), dtype=float)
    r_now = np.asarray(r_sat_fn(et), dtype=float)
    r_plus = np.asarray(r_sat_fn(et + dt), dtype=float)
    n_minus = np.asarray(profile(r_minus, et - dt), dtype=float)
    n_now = np.asarray(profile(r_now, et), dtype=float)
    n_plus = np.asarray(profile(r_plus, et + dt), dtype=float)
    return n_minus, n_now, n_plus


def angular_rate(
    profile: AttitudeCallable,
    r_sat_fn: _TrajectoryCallable,
    et: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Angular velocity omega(et) (rad/s, 3-vector in J2000) via Eq. (K1).

    Central difference over ``[et - dt, et + dt]``. The profile is
    evaluated at three epochs (two for the rate, one for the normal
    at ``et``); the trajectory callable is evaluated at the same three
    epochs so geometry-dependent primitives such as ``sun_pointing`` and
    bisector tracking see the correct sail positions.

    Parameters
    ----------
    profile
        Attitude profile callable.
    r_sat_fn
        Sail-trajectory callable ``et -> r_sat_j2000_km``. For profiles
        that do not use ``r_sat`` (e.g. ``fixed_j2000``) this can be
        a trivial ``lambda et: np.zeros(3)``.
    et
        TDB seconds at which to evaluate omega.
    dt
        Central-difference half-step, seconds. Default 1.0. Keep
        ``dt`` smaller than the fastest timescale in the profile; for
        Mars-LMO signals (orbital period ~2 h), ``dt`` up to ~10 s
        still resolves omega to several significant figures.
    """
    n_minus, n_now, n_plus = _three_point_normals(profile, r_sat_fn, et, dt)
    n_dot = (n_plus - n_minus) / (2.0 * dt)
    return np.cross(n_now, n_dot)


def angular_acceleration(
    profile: AttitudeCallable,
    r_sat_fn: _TrajectoryCallable,
    et: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Angular acceleration alpha(et) (rad/s^2, 3-vector in J2000) via Eq. (K2).

    Second central difference over ``[et - dt, et + dt]``. Identical
    profile/trajectory evaluation count to ``angular_rate``; callers
    that need both omega and alpha at the same epoch can share the
    three-point evaluation via ``_three_point_normals`` (internal).

    Parameters
    ----------
    profile, r_sat_fn, et, dt
        As for ``angular_rate``.
    """
    n_minus, n_now, n_plus = _three_point_normals(profile, r_sat_fn, et, dt)
    n_ddot = (n_plus - 2.0 * n_now + n_minus) / (dt * dt)
    return np.cross(n_now, n_ddot)


def alpha_profile(
    profile: AttitudeCallable,
    r_sat_fn: _TrajectoryCallable,
    et_range: Tuple[float, float],
    n_samples: int = 1000,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Magnitude of alpha(t) sampled uniformly across ``et_range``.

    The sweep is **inset by ``dt``** from each end of ``et_range`` so
    the central-difference stencil never queries the profile outside
    its declared domain; callers wanting the exact endpoints should
    widen ``et_range`` by ``dt`` accordingly.

    Parameters
    ----------
    profile, r_sat_fn, dt
        As for ``angular_acceleration``.
    et_range
        ``(t_start, t_end)`` SPICE TDB seconds; must satisfy
        ``t_end - t_start > 2 * dt`` so the inset interval is non-empty.
    n_samples
        Number of sample points on the inset interval. Default 1000.

    Returns
    -------
    tuple of ndarray
        ``(et_array, alpha_magnitudes)`` each of shape ``(n_samples,)``.
        ``alpha_magnitudes`` is the Euclidean norm of alpha at each et,
        in rad/s^2.
    """
    t_start, t_end = et_range
    if t_end <= t_start:
        raise ValueError(
            f"alpha_profile: et_range must have t_end > t_start, got {et_range}"
        )
    if n_samples < 2:
        raise ValueError(
            f"alpha_profile: n_samples must be >= 2, got {n_samples}"
        )
    if (t_end - t_start) <= 2.0 * dt:
        raise ValueError(
            f"alpha_profile: et_range width {t_end - t_start} s is too "
            f"narrow for the chosen dt={dt} s (need width > 2*dt)."
        )
    et_array = np.linspace(t_start + dt, t_end - dt, n_samples)
    alpha_mag = np.empty(n_samples, dtype=float)
    for i, et in enumerate(et_array):
        a = angular_acceleration(profile, r_sat_fn, float(et), dt=dt)
        alpha_mag[i] = float(np.linalg.norm(a))
    return et_array, alpha_mag


def check_alpha_bound(
    profile: AttitudeCallable,
    r_sat_fn: _TrajectoryCallable,
    alpha_max: float,
    et_range: Tuple[float, float],
    n_samples: int = 1000,
    dt: float = 1.0,
) -> Optional[float]:
    """Sweep ``et_range`` and return the first et at which ``|alpha| > alpha_max``.

    Returns ``None`` if the bound holds across every sample. The sweep
    is inset by ``dt`` on each end (see ``alpha_profile``). Use a
    ``n_samples`` density comparable to the fastest feature in the
    profile; for a quintic ``smooth_slew`` the peak is at a known
    fractional position of the slew, so ~100 samples across the slew
    duration comfortably localise it.

    Parameters
    ----------
    profile, r_sat_fn, et_range, n_samples, dt
        As for ``alpha_profile``.
    alpha_max
        Non-negative bound in rad/s^2.

    Returns
    -------
    float or None
        First et (TDB seconds) in the swept grid where
        ``|alpha(et)| > alpha_max``, or ``None`` if the bound holds at
        every sample. Resolution is
        ``(et_range[1] - et_range[0] - 2*dt) / (n_samples - 1)``.
    """
    if alpha_max < 0.0:
        raise ValueError(
            f"check_alpha_bound: alpha_max must be >= 0, got {alpha_max}"
        )
    et_array, alpha_mag = alpha_profile(
        profile, r_sat_fn, et_range, n_samples=n_samples, dt=dt
    )
    violators = np.where(alpha_mag > alpha_max)[0]
    if len(violators) == 0:
        return None
    return float(et_array[violators[0]])
