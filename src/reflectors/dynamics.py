"""Translational dynamics of a point mass near Mars.

Design: Cartesian state ``y = [r_x, r_y, r_z, v_x, v_y, v_z]`` with position
in kilometres and velocity in kilometres per second, expressed in
Mars-centred J2000 axes:

    - Origin: Mars body centre (NAIF ID 499). Obtained from ``mar099.bsp``
      plus DE440.
    - Axes: inherited from the J2000 inertial frame (Earth mean equator and
      equinox of J2000.0). Axes do NOT rotate.
    - Units: km, km/s, s throughout. SPICE native units.

This frame is what SPICE returns from ``spkezr(target, et, "J2000", ...,
"MARS")``. It is sometimes called "Mars-Centred Inertial" (MCI) in the
literature; the "inertial" specifier here refers to the J2000 orientation
(true inertial -- no precession / rotation), NOT to the Mars-mean-equator-of-
J2000 (MME2000) frame used in some mission-design documents. Conversion to
MME2000 for reporting orbital elements relative to Mars's equator lives in
``reflectors.elements``.

This module exposes:
    - ``PropagationOptions``: frozen dataclass carrying integrator choice and
      tolerances, with named presets (``fast``, ``default``, ``high_accuracy``).
    - ``mars_gm_km3_per_s2``: cached accessor for BODY499_GM.
    - ``two_body_acceleration``: pure central gravity, factored so J2 / SRP /
      additional force layers can add into the acceleration sum.
    - ``PropagationResult``: dataclass returned by the propagator.
    - ``propagate``: scipy.integrate.solve_ivp wrapper supporting two-body
      dynamics and optional zonal, third-body, and SRP layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Integrator configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationOptions:
    """Integrator choice and tolerances for ``propagate``.

    Defaults target high-accuracy numerical work: DOP853 (Dormand-Prince 8th
    order, adaptive step), ``rtol`` at 1e-12, ``atol`` at 1e-9 km (equivalently
    km/s via the scalar-atol broadcast scipy performs). These are deliberately
    tight -- for a 400 km Mars orbit that is sub-micrometre spatial resolution
    on a 3800 km radius vector. Relax via ``PropagationOptions.fast()`` or by
    instantiating with explicit values for speed-sensitive use cases.

    The dataclass is frozen so a single ``DEFAULT_OPTIONS`` instance can be
    shared across call sites without aliasing risk.
    """

    method: str = "DOP853"
    rtol: float = 1e-12
    atol: float = 1e-9
    max_step_s: Optional[float] = None

    @classmethod
    def fast(cls) -> "PropagationOptions":
        """Looser tolerances for quick-look propagations (still DOP853)."""
        return cls(method="DOP853", rtol=1e-9, atol=1e-6)

    @classmethod
    def high_accuracy(cls) -> "PropagationOptions":
        """Tighter than default -- use for conservation-law verification."""
        return cls(method="DOP853", rtol=1e-13, atol=1e-12)

    @classmethod
    def rk45(cls) -> "PropagationOptions":
        """Drop-in RK45 preset for integrator cross-checks."""
        return cls(method="RK45", rtol=1e-10, atol=1e-10)


DEFAULT_OPTIONS = PropagationOptions()


# ---------------------------------------------------------------------------
# Gravitational parameters
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def body_gm_km3_per_s2(naif_id: int) -> float:
    """Return BODY{id}_GM in km^3/s^2 from the loaded kernel pool.

    Cached. Raises ``spiceypy`` errors if the requested BODY{id}_GM is not
    present in the loaded kernel pool. The canonical source is
    ``gm_de440.tpc`` which ships GM values for all DE440 planets, the Sun,
    Phobos/Deimos, and Jupiter barycenter (among others).
    """
    n, vals = spice.bodvcd(naif_id, "GM", 1)
    assert n == 1, f"expected 1 GM value for {naif_id}, got {n}"
    gm = float(vals[0])
    logger.debug("GM(%d) = %.9e km^3/s^2", naif_id, gm)
    return gm


def mars_gm_km3_per_s2() -> float:
    """Mars gravitational parameter mu_Mars = G * M_Mars (km^3/s^2).

    Sourced from ``gm_de440.tpc``. At the time of writing the loaded value
    is 4.28283736...e4 km^3/s^2, matching the Konopliv 2016 "Mars
    Geophysical Parameters" value to the ~1e-5 relative level.
    """
    return body_gm_km3_per_s2(499)


def sun_gm_km3_per_s2() -> float:
    """Sun gravitational parameter mu_Sun = G * M_Sun (km^3/s^2).

    Sourced from ``gm_de440.tpc``. Used by third-body perturbations.
    """
    return body_gm_km3_per_s2(10)


# ---------------------------------------------------------------------------
# Accelerations
# ---------------------------------------------------------------------------


def two_body_acceleration(r_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Central-body Newtonian acceleration ``-mu r / |r|^3``.

    Parameters
    ----------
    r_km
        Position of the test particle relative to the central body, shape (3,),
        km. Passing a zero vector raises (undefined acceleration).
    mu_km3_s2
        Gravitational parameter of the central body, km^3/s^2.

    Returns
    -------
    ndarray, shape (3,)
        Acceleration in km/s^2.
    """
    r = np.asarray(r_km, dtype=float)
    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise ValueError("two-body acceleration undefined at r = 0")
    return -mu_km3_s2 * r / (r_norm ** 3)


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------


@dataclass
class PropagationResult:
    """Result of a single ``propagate`` call.

    ``state_km_kmps`` is shape (N, 6) with column layout
    [r_x, r_y, r_z, v_x, v_y, v_z], km and km/s, Mars-centred J2000.

    Termination fields (``termination_reason``, ``termination_et``,
    ``termination_t_s``, ``termination_state_km_kmps``) report how and
    when integration stopped. In the default case -- integration ran
    to ``t_span_s[1]`` normally -- ``termination_reason`` is
    ``"t_final"`` and the other three fields are ``None``. When a
    SciPy terminal event fires (e.g. an ``AltitudeFloor``), the
    ``termination_reason`` is set to the event's ``label``, and the
    crossing epoch + 6-state are captured. ``state_km_kmps[-1]``
    already matches ``termination_state_km_kmps`` in that case --
    SciPy inserts the event point as the last sample.
    """

    t_s: np.ndarray
    state_km_kmps: np.ndarray
    method: str
    rtol: float
    atol: float
    mu_km3_s2: float
    epoch_et: Optional[float]
    solver_message: str
    n_rhs_calls: int
    metadata: dict = field(default_factory=dict)
    termination_reason: str = "t_final"
    termination_et: Optional[float] = None
    termination_t_s: Optional[float] = None
    termination_state_km_kmps: Optional[np.ndarray] = None

    def positions(self) -> np.ndarray:
        return self.state_km_kmps[:, :3]

    def velocities(self) -> np.ndarray:
        return self.state_km_kmps[:, 3:]

    def specific_energy(self) -> np.ndarray:
        """Specific mechanical energy eps = v^2/2 - mu/r per sample (km^2/s^2).

        Only meaningful for pure two-body (no non-conservative forces and no
        zonal potential). For perturbed runs, compute energy with the correct
        potential elsewhere.
        """
        r = np.linalg.norm(self.positions(), axis=1)
        v2 = np.sum(self.velocities() ** 2, axis=1)
        return 0.5 * v2 - self.mu_km3_s2 / r

    def specific_angular_momentum(self) -> np.ndarray:
        """Specific angular-momentum vector h = r x v per sample, shape (N, 3)."""
        return np.cross(self.positions(), self.velocities())


def _make_rhs(mu: float, contributors: list, *, share_ephemeris: bool = True):
    """Build a closure ``rhs(t, y)`` for ``solve_ivp``.

    The RHS evaluates dy/dt = [v, a(r, t)], where the acceleration is
    the sum of the central two-body term and any number of additional
    "contributor" closures, each with signature ``(r, t_offset) ->
    ndarray(3,)``. ``t_offset`` is the integrator time relative to
    the propagator epoch (so contributors that depend on absolute
    SPICE ET should add ``epoch_et`` themselves).

    Contributors compose ADDITIVELY -- the order of summation does not
    affect physics but is fixed (insertion order) for bit-reproducibility.
    Empty list reduces exactly to pure two-body.
    """
    # All current non-central contributors are ephemeris/frame dependent. One
    # fresh exact-key cache per RHS evaluation lets gravity, third bodies, SRP,
    # and attitude share the same ET's SPICE answers without persisting values
    # across integrator times or propagations.
    from reflectors.ephemeris import (
        EphemerisCacheStats,
        ephemeris_evaluation_context,
    )

    ephemeris_stats = EphemerisCacheStats()

    def evaluate(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        a = two_body_acceleration(r, mu)
        for contrib in contributors:
            a = a + contrib(r, t)
        out = np.empty(6)
        out[:3] = v
        out[3:] = a
        return out

    if contributors and share_ephemeris:
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            with ephemeris_evaluation_context(ephemeris_stats):
                return evaluate(t, y)
    else:
        # Preserve the pure-two-body hot path without context-manager overhead.
        rhs = evaluate

    rhs.ephemeris_cache_stats = ephemeris_stats

    return rhs


# Every ephemeris-dependent contributor evaluates SPICE at
# ``epoch_et + time_dir * t``. ``time_dir = +1`` is the default forward clock;
# ``time_dir = -1`` clocks the ephemeris backward for
# reverse-time propagation (capture-node / time-reversal). The integration still
# advances forward in ``t``; only the ephemeris query is reflected. Mirrors the
# ``ephemeris_time_direction`` knob in ``reflectors.escape``. The central
# two-body term carries no ephemeris dependence, so it is unaffected.
def _make_zonal_contributor(mu: float, ref_radius_km: float, J_by_degree: dict,
                            epoch_et: float, time_dir: int = +1):
    """Return ``(r, t_offset) -> a_zonal_inertial(r, epoch_et + time_dir*t, ...)``.

    The lazy import of ``reflectors.gravity`` keeps the dependency optional.
    """
    from reflectors.gravity import zonal_acceleration_inertial
    return lambda r, t: zonal_acceleration_inertial(
        r, epoch_et + time_dir * t, mu, ref_radius_km, J_by_degree
    )


def _make_harmonic_contributor(
    model, gravity_degree: int, gravity_order: int, epoch_et: float,
    gravity_backend: str,
    time_dir: int = +1,
):
    """Return ``(r, t_offset) -> a_harmonic_inertial(...)`` (perturbation only).

    The Cunningham implementation returns the perturbation acceleration
    when ``include_central=False``; the central term comes from
    ``two_body_acceleration`` inside the unified RHS so the split
    matches the zonal path bit-for-bit.
    """
    from reflectors.gravity import mars_gravity_acceleration_inertial
    return lambda r, t: mars_gravity_acceleration_inertial(
        r, epoch_et + time_dir * t, model, gravity_degree, gravity_order,
        include_central=False,
        backend=gravity_backend,
    )


def _make_third_body_contributor(third_bodies, epoch_et: float, time_dir: int = +1):
    """Return ``(r, t_offset) -> sum of third-body accelerations at epoch_et + time_dir*t``.

    Lazy import to keep ``reflectors.third_body`` optional.
    """
    from reflectors.third_body import third_body_acceleration_from_spice
    bodies = tuple(third_bodies)  # freeze ordering for reproducibility
    return lambda r, t: third_body_acceleration_from_spice(
        r, epoch_et + time_dir * t, bodies
    )


def _make_srp_contributor(solar_sail, sail_normal, epoch_et: float, time_dir: int = +1):
    """Return ``(r, t_offset) -> a_SRP(r, epoch_et + time_dir*t)``.

    Lazy import to keep ``reflectors.srp`` optional (same pattern as the
    other perturbation contributors).
    """
    from reflectors.srp import make_srp_contributor
    return make_srp_contributor(
        solar_sail, sail_normal, epoch_et, ephemeris_time_direction=time_dir
    )


def _make_spherical_srp_contributor(spherical_particle, epoch_et: float,
                                    time_dir: int = +1):
    """Return ``(r, t_offset) -> a_SRP_sphere(r, epoch_et + time_dir*t)``.

    Spherical-grain SRP path (Hamilton & Krivov 1996). Lazy import
    matches the flat-sail helper above.
    """
    from reflectors.srp import make_spherical_srp_contributor
    return make_spherical_srp_contributor(
        spherical_particle, epoch_et, ephemeris_time_direction=time_dir
    )


def _make_tumble_averaged_contributor(tumble_averaged_sail, epoch_et: float,
                                      time_dir: int = +1):
    """Return ``(r, t_offset) -> <a_SRP>(r, epoch_et + time_dir*t)``.

    Orientation-averaged flat-sail SRP path for an uncommanded (tumbling)
    attitude. Lazy import matches the two helpers above.
    """
    from reflectors.srp import make_tumble_averaged_contributor
    return make_tumble_averaged_contributor(
        tumble_averaged_sail, epoch_et, ephemeris_time_direction=time_dir
    )


def propagate(
    state0_km_kmps: np.ndarray,
    t_span_s: tuple[float, float],
    *,
    mu_km3_s2: Optional[float] = None,
    epoch_et: Optional[float] = None,
    zonal_degree: int = 0,
    gravity_degree: int = 0,
    gravity_order: Optional[int] = None,
    gravity_backend: str = "numba",
    share_ephemeris: bool = True,
    third_bodies=None,
    solar_sail=None,
    sail_normal=None,
    spherical_particle=None,
    tumble_averaged_sail=None,
    altitude_floor=None,
    radius_ceiling=None,
    options: PropagationOptions = DEFAULT_OPTIONS,
    t_eval_s: Optional[np.ndarray] = None,
    ephemeris_time_direction: int = +1,
) -> PropagationResult:
    """Propagate a point-mass state under Mars-centred dynamics.

    Acceleration model (additive composition):

      a(r, t)  =  -mu r/|r|^3                            (central two-body)
                +  a_central_perturbation                (one of: zonal, harmonic)
                +  sum over k of  a_3b(r; body_k, t)     (third bodies)

    The two central-perturbation paths (zonals via scalar Legendre,
    full harmonics via Cunningham-1970 complex-V) are mutually
    exclusive because they are alternative computations of the same
    physical effect. ``zonal_degree=0, gravity_degree=0`` gives pure
    two-body. They are validated to agree to machine precision when
    ``gravity_order=0`` vs ``zonal_degree=gravity_degree``
    (tests/test_gravity_harmonics.py); both kept because the zonal
    path is an independent cross-check of the Cunningham code and is
    modestly cheaper for zonal-only runs.

    The third-body term is independently switchable and stacks with
    either central-perturbation mode (or pure two-body).

    Parameters
    ----------
    state0_km_kmps
        Initial Cartesian state, shape (6,): [r, v] km and km/s in Mars-centred
        J2000.
    t_span_s
        (t0, tf) in seconds relative to ``epoch_et``. For pure two-body the
        absolute epoch is irrelevant; for any perturbation that depends on
        absolute time (gravity harmonics need IAU_MARS rotation; third-body
        needs SPICE positions) it pins the time origin.
    mu_km3_s2
        Override for the central-body gravitational parameter. ``None`` has
        these behaviours:
          - pure two-body (no central perturbation): pull mu_Mars from SPICE
            (BODY499_GM).
          - zonal or gravity on: use the gravity-model mu that the
            C_bar / S_bar coefficients were fit against (MRO120F system GM),
            for self-consistency with the coefficients.
        Passing an explicit value overrides both paths.
    epoch_et
        Absolute SPICE ET corresponding to ``t_span_s[0]``. Required when
        any perturbation other than pure two-body is on (gravity harmonics
        need it for the IAU_MARS rotation; third-body needs it for spkezr).
        Optional and merely recorded for pure two-body.
    zonal_degree
        If > 0 and ``gravity_degree == 0``: include Mars zonal harmonics
        J_2 .. J_{zonal_degree} from MRO120F via the scalar Legendre path.
    gravity_degree
        If > 0: include full spherical harmonics (zonals + tesserals +
        sectorals) through this degree via the Cunningham path. Must be
        ``<= model.max_degree`` and cannot be combined with ``zonal_degree > 0``.
    gravity_order
        Maximum order for the Cunningham path; defaults to ``gravity_degree``
        (full triangle). ``gravity_order = 0`` reduces to the zonal slice.
    gravity_backend
        Cunningham implementation. ``"numba"`` (default) is the compiled,
        exact-operation-order compiled path with ``fastmath`` disabled;
        ``"python"`` selects the retained reference recurrence. Ignored when
        ``gravity_degree == 0``.
    share_ephemeris
        If True (default), identical SPICE state and frame requests are shared
        only within one RHS evaluation at the exact same ET. False is the
        uncached regression oracle. No interpolation or time rounding is used.
    third_bodies
        Optional iterable of ``reflectors.third_body.ThirdBody`` specs.
        Each contributes a Montenbruck Eq. 3.37 perturbation evaluated at
        the absolute SPICE ET ``epoch_et + t``. Stacks additively with
        the central-perturbation choice.
    solar_sail, sail_normal
        Optional ``reflectors.srp.SolarSail`` + attitude callable
        ``(r, et) -> unit_vec`` pair. Both must be supplied together
        (or both omitted). Enables solar-radiation-pressure thrust via
        the McInnes (1999) §2.6.1 six-parameter optical force model,
        shadow-gated by ``reflectors.shadow`` (binary Mars umbra).
        Stacks additively with zonals / harmonics / third bodies.
        Mutually exclusive with ``spherical_particle``.
    spherical_particle
        Optional ``reflectors.srp.SphericalParticle`` for the spherical-
        grain SRP path (Hamilton & Krivov 1996, *Icarus* 123:503-523,
        §2.1 + Eq. (3); Burns et al. 1979). Force is
        ``Q_pr * P(r_helio) * pi r_g^2 / m_g`` along the anti-Sun line,
        orientation-independent (no attitude callable needed). Same
        binary umbra gate as the flat-sail path. Mutually exclusive
        with ``solar_sail``; supplying both raises ``ValueError``.
    tumble_averaged_sail
        Optional ``reflectors.srp.TumbleAveragedSail`` for a flat sail
        tumbling fast enough that its SRP force time-averages. The
        average over uniform orientations is purely anti-sunward,
        ``<a> = -k P (A/m) s_hat``, so like ``spherical_particle`` it
        needs no attitude callable. It supports orientation-averaged survival
        calculations; see that dataclass for the averaging assumptions. Same
        umbra gate. Mutually
        exclusive with both ``solar_sail`` and ``spherical_particle``.
    altitude_floor
        Optional ``reflectors.termination.AltitudeFloor`` configuring
        a hard altitude-floor terminal event. When the sail altitude
        crosses the floor downward, ``solve_ivp``'s event root-finder
        stops integration cleanly and the returned ``PropagationResult``
        has ``termination_reason = floor.label`` plus the crossing
        epoch + state populated. The initial state is validated up-
        front; a trajectory starting below the floor raises
        ``ValueError`` rather than starting propagation.
    radius_ceiling
        Optional ``reflectors.termination.RadiusCeiling`` configuring a
        hard OUTER radius terminal event (mirror of ``altitude_floor``).
        When the inertial radius crosses the ceiling upward,
        ``solve_ivp``'s event root-finder stops integration cleanly and
        ``termination_reason`` is set to ``ceiling.label`` with the
        crossing epoch + state populated. Used for the Mars Hill-sphere
        boundary. A trajectory starting above the ceiling raises
        ``ValueError``. Stacks with ``altitude_floor`` -- whichever
        event fires first terminates the run.
    options
        ``PropagationOptions`` instance controlling method and tolerances.
    t_eval_s
        Optional array of output times (seconds, same reference as ``t_span_s``).
        If None, scipy picks adaptive output points.

    Returns
    -------
    PropagationResult
    """
    state0 = np.asarray(state0_km_kmps, dtype=float)
    if state0.shape != (6,):
        raise ValueError(f"state0 must be shape (6,), got {state0.shape}")
    if zonal_degree < 0:
        raise ValueError(f"zonal_degree must be >= 0, got {zonal_degree}")
    if gravity_degree < 0:
        raise ValueError(f"gravity_degree must be >= 0, got {gravity_degree}")
    if zonal_degree > 0 and gravity_degree > 0:
        raise ValueError(
            "zonal_degree and gravity_degree are mutually exclusive; pick one "
            "perturbation path"
        )

    if ephemeris_time_direction not in (+1, -1):
        raise ValueError(
            "ephemeris_time_direction must be +1 (forward) or -1 (backward), "
            f"got {ephemeris_time_direction}"
        )

    third_bodies_list = list(third_bodies) if third_bodies else []

    # Resolve gravity model and mu. Done here so the test / RHS hot path
    # never touches SPICE pool variables.
    ref_radius_km: Optional[float] = None
    J_by_degree: Optional[dict] = None
    model = None
    resolved_gravity_order: Optional[int] = None
    metadata: dict = {}
    if zonal_degree > 0:
        if epoch_et is None:
            raise ValueError("zonal_degree > 0 requires epoch_et (to rotate J2000 <-> IAU_MARS)")
        from reflectors.gravity import mars_gravity_model, zonal_coefficients
        model = mars_gravity_model(max_degree=max(zonal_degree, 2))
        ref_radius_km = model.ref_radius_km
        J_by_degree = zonal_coefficients(model, zonal_degree)
        if mu_km3_s2 is None:
            # Default to the gravity-model mu when zonals are on (consistency
            # with the coefficients). Callers can still override explicitly.
            mu_km3_s2 = model.mu_km3_s2
        metadata = {
            "path": "zonal_scalar_legendre",
            "zonal_degree": zonal_degree,
            "gravity_model": model.source,
            "ref_radius_km": ref_radius_km,
            "J_by_degree": J_by_degree,
        }
    elif gravity_degree > 0:
        if epoch_et is None:
            raise ValueError(
                "gravity_degree > 0 requires epoch_et (to rotate J2000 <-> IAU_MARS)"
            )
        from reflectors.gravity import (
            cunningham_backend_metadata,
            mars_gravity_model,
            warm_cunningham_backend,
        )
        if gravity_order is None:
            resolved_gravity_order = gravity_degree
        else:
            resolved_gravity_order = gravity_order
        if not (0 <= resolved_gravity_order <= gravity_degree):
            raise ValueError(
                f"gravity_order={resolved_gravity_order} must satisfy "
                f"0 <= gravity_order <= gravity_degree={gravity_degree}"
            )
        model = mars_gravity_model(max_degree=max(gravity_degree, 2))
        ref_radius_km = model.ref_radius_km
        backend_metadata = cunningham_backend_metadata(gravity_backend)
        # Compile before solve_ivp enters its RHS loop.  Calling this once in
        # the parent process also lets forked optimizer workers inherit the
        # ready dispatcher; cached machine code covers every degree/order.
        warm_cunningham_backend(
            model,
            gravity_degree,
            resolved_gravity_order,
            backend=gravity_backend,
        )
        if mu_km3_s2 is None:
            mu_km3_s2 = model.mu_km3_s2
        metadata = {
            "path": "cunningham_full_harmonics",
            "gravity_degree": gravity_degree,
            "gravity_order": resolved_gravity_order,
            "gravity_model": model.source,
            "ref_radius_km": ref_radius_km,
            **backend_metadata,
        }
    else:
        if mu_km3_s2 is None:
            mu_km3_s2 = mars_gm_km3_per_s2()

    if third_bodies_list:
        if epoch_et is None:
            raise ValueError(
                "third_bodies require epoch_et (used to fetch each body's "
                "position via spkezr at epoch_et + t)"
            )
        metadata["third_bodies"] = [
            {"label": b.label, "naif_id": b.naif_id, "mu_km3_s2": b.mu_km3_s2}
            for b in third_bodies_list
        ]

    # Decouple Phobos / Deimos from the lumped Mars-system central mu when
    # they are also passed as separate third-body perturbers. The MRO120F
    # SHADR header mu = 42828.3756640 km^3/s^2 is the Mars-system total
    # (Mars + Phobos + Deimos), per the PDS label jgmro_120f_sha.lbl
    # lines 41-45. Leaving that lump in place while also adding the moons
    # via Eq. 3.37 double-counts their masses: once at Mars centre via
    # the lumped mu, and again at their actual orbital positions via the
    # third-body direct term. The third-body indirect term subtracts the
    # moon's pull on Mars at its real position, but that does not cancel
    # the lumped-at-centre piece.
    #
    # Subtract each moon's mu_km3_s2 from the central mu so the central
    # two-body term becomes Mars-planet-alone (Konopliv-fit when the
    # default factories are used). Pure two-body propagations (no gravity
    # model) skip this -- they already use BODY499_GM for the central
    # term, which is Mars-planet-alone, so no decoupling is needed.
    #
    central_mu_decouple_naif_ids: list[int] = []
    central_mu_subtracted_km3_s2: float = 0.0
    if (zonal_degree > 0 or gravity_degree > 0) and third_bodies_list:
        for body in third_bodies_list:
            if body.naif_id in (401, 402):
                mu_km3_s2 = mu_km3_s2 - body.mu_km3_s2
                central_mu_decouple_naif_ids.append(body.naif_id)
                central_mu_subtracted_km3_s2 += body.mu_km3_s2
        if central_mu_decouple_naif_ids:
            metadata["central_mu_decouple"] = {
                "naif_ids": central_mu_decouple_naif_ids,
                "mu_subtracted_km3_s2": central_mu_subtracted_km3_s2,
                "mu_central_after_decouple_km3_s2": mu_km3_s2,
            }

    if (solar_sail is None) != (sail_normal is None):
        raise ValueError(
            "solar_sail and sail_normal must be supplied together (or both "
            "omitted); got solar_sail=%s, sail_normal=%s"
            % (solar_sail, sail_normal)
        )
    # The three SRP models are alternative descriptions of the same physical
    # effect, so exactly one may be active. The validation message uses the
    # general term "mutually exclusive" for all supported SRP models.
    _srp_models_on = [
        name for name, obj in (
            ("solar_sail", solar_sail),
            ("spherical_particle", spherical_particle),
            ("tumble_averaged_sail", tumble_averaged_sail),
        ) if obj is not None
    ]
    if len(_srp_models_on) > 1:
        raise ValueError(
            "the SRP models are mutually exclusive; supply exactly one, got "
            + " and ".join(_srp_models_on)
        )
    if solar_sail is not None:
        if epoch_et is None:
            raise ValueError(
                "solar_sail requires epoch_et (SRP needs absolute Sun "
                "position via spkezr at epoch_et + t)"
            )
        metadata["solar_sail"] = {
            "area_m2": solar_sail.area_m2,
            "mass_kg": solar_sail.mass_kg,
            "loading_kg_per_m2": solar_sail.loading_kg_per_m2,
            "optical": {
                "rho": solar_sail.optical.rho,
                "s": solar_sail.optical.s,
                "eps_front": solar_sail.optical.eps_front,
                "eps_back": solar_sail.optical.eps_back,
                "B_front": solar_sail.optical.B_front,
                "B_back": solar_sail.optical.B_back,
            },
        }
    if spherical_particle is not None:
        if epoch_et is None:
            raise ValueError(
                "spherical_particle requires epoch_et (SRP needs absolute "
                "Sun position via spkezr at epoch_et + t)"
            )
        metadata["spherical_particle"] = {
            "radius_m": spherical_particle.radius_m,
            "density_kg_per_m3": spherical_particle.density_kg_per_m3,
            "Q_pr": spherical_particle.Q_pr,
            "mass_kg": spherical_particle.mass_kg,
            "cross_section_m2": spherical_particle.cross_section_m2,
            "area_to_mass_m2_per_kg": spherical_particle.area_to_mass_m2_per_kg,
        }
    if tumble_averaged_sail is not None:
        if epoch_et is None:
            raise ValueError(
                "tumble_averaged_sail requires epoch_et (SRP needs absolute "
                "Sun position via spkezr at epoch_et + t)"
            )
        _tas = tumble_averaged_sail.sail
        metadata["tumble_averaged_sail"] = {
            "area_m2": _tas.area_m2,
            "mass_kg": _tas.mass_kg,
            "loading_kg_per_m2": _tas.loading_kg_per_m2,
            "average_coefficient": tumble_averaged_sail.average_coefficient,
            "optical": {
                "rho": _tas.optical.rho,
                "s": _tas.optical.s,
                "eps_front": _tas.optical.eps_front,
                "eps_back": _tas.optical.eps_back,
                "B_front": _tas.optical.B_front,
                "B_back": _tas.optical.B_back,
                "two_sided": _tas.optical.two_sided,
            },
        }

    ivp_kwargs = dict(
        method=options.method,
        rtol=options.rtol,
        atol=options.atol,
        dense_output=False,
        vectorized=False,
    )
    if options.max_step_s is not None:
        ivp_kwargs["max_step"] = options.max_step_s
    if t_eval_s is not None:
        ivp_kwargs["t_eval"] = np.asarray(t_eval_s, dtype=float)

    # Terminal-event hooks: altitude floor (inner) and radius ceiling
    # (outer). Both are collected into one ``events`` list passed to
    # solve_ivp; ``event_labels`` runs parallel to it so the post-
    # integration unpacking knows which event fired. Initial-state
    # validation is performed here (not inside termination.py) because
    # it depends on the propagator's starting state. The floor event
    # uses direction=-1 and the ceiling event direction=+1 so a state
    # exactly at a boundary moving the "safe" way does not spuriously
    # fire on step 0.
    events: list = []
    event_labels: list = []
    if altitude_floor is not None:
        from reflectors.termination import make_altitude_floor_event
        r0_mag = float(np.linalg.norm(state0[:3]))
        alt_0 = r0_mag - altitude_floor.reference_radius_km
        if alt_0 < altitude_floor.altitude_km:
            raise ValueError(
                f"initial state is below altitude_floor: alt_0 = "
                f"{alt_0:.3f} km < floor {altitude_floor.altitude_km:.3f} km"
            )
        events.append(make_altitude_floor_event(altitude_floor))
        event_labels.append(altitude_floor.label)
        metadata["altitude_floor"] = {
            "altitude_km": altitude_floor.altitude_km,
            "reference_radius_km": altitude_floor.reference_radius_km,
            "label": altitude_floor.label,
        }
    if radius_ceiling is not None:
        from reflectors.termination import make_radius_ceiling_event
        r0_mag = float(np.linalg.norm(state0[:3]))
        if r0_mag > radius_ceiling.radius_km:
            raise ValueError(
                f"initial state is above radius_ceiling: r0 = "
                f"{r0_mag:.3f} km > ceiling {radius_ceiling.radius_km:.3f} km"
            )
        events.append(make_radius_ceiling_event(radius_ceiling))
        event_labels.append(radius_ceiling.label)
        metadata["radius_ceiling"] = {
            "radius_km": radius_ceiling.radius_km,
            "label": radius_ceiling.label,
        }
    if events:
        ivp_kwargs["events"] = events

    metadata["ephemeris_time_direction"] = ephemeris_time_direction

    contributors: list = []
    if zonal_degree > 0:
        contributors.append(
            _make_zonal_contributor(mu_km3_s2, ref_radius_km, J_by_degree,
                                    epoch_et, ephemeris_time_direction)
        )
    elif gravity_degree > 0:
        contributors.append(
            _make_harmonic_contributor(
                model, gravity_degree, resolved_gravity_order, epoch_et,
                gravity_backend,
                ephemeris_time_direction,
            )
        )
    if third_bodies_list:
        contributors.append(
            _make_third_body_contributor(third_bodies_list, epoch_et,
                                         ephemeris_time_direction)
        )
    if solar_sail is not None:
        contributors.append(
            _make_srp_contributor(solar_sail, sail_normal, epoch_et,
                                  ephemeris_time_direction)
        )
    if spherical_particle is not None:
        contributors.append(
            _make_spherical_srp_contributor(spherical_particle, epoch_et,
                                            ephemeris_time_direction)
        )
    if tumble_averaged_sail is not None:
        contributors.append(
            _make_tumble_averaged_contributor(tumble_averaged_sail, epoch_et,
                                              ephemeris_time_direction)
        )
    rhs = _make_rhs(
        mu_km3_s2, contributors, share_ephemeris=bool(share_ephemeris)
    )

    logger.debug(
        "propagate: t_span=%s, mu=%.6e, zonal_degree=%d, gravity_degree=%d, "
        "gravity_order=%s, gravity_backend=%s, third_bodies=%s, solar_sail=%s, "
        "spherical_particle=%s, tumble_averaged_sail=%s, "
        "method=%s, rtol=%.1e, atol=%.1e",
        t_span_s, mu_km3_s2, zonal_degree, gravity_degree,
        resolved_gravity_order,
        gravity_backend if gravity_degree > 0 else None,
        [b.label for b in third_bodies_list] or None,
        "on" if solar_sail is not None else "off",
        "on" if spherical_particle is not None else "off",
        "on" if tumble_averaged_sail is not None else "off",
        options.method, options.rtol, options.atol,
    )
    sol = solve_ivp(rhs, t_span_s, state0, **ivp_kwargs)
    if not sol.success:
        raise RuntimeError(f"propagator failed: {sol.message}")

    if contributors and share_ephemeris:
        metadata["ephemeris_cache"] = rhs.ephemeris_cache_stats.as_dict()
    else:
        metadata["ephemeris_cache"] = {"enabled": False}

    # Default-path termination fields; overwritten below if a terminal
    # event fired. With several events registered, the earliest-firing
    # one wins (solve_ivp stops at the first terminal crossing anyway;
    # the min-over-events guard is an independent safeguard).
    termination_reason: str = "t_final"
    termination_t_s: Optional[float] = None
    termination_et: Optional[float] = None
    termination_state: Optional[np.ndarray] = None
    if events:
        t_events_list = sol.t_events or []
        y_events_list = sol.y_events or []
        best_idx: Optional[int] = None
        best_t: Optional[float] = None
        for idx in range(len(events)):
            if idx < len(t_events_list) and len(t_events_list[idx]) > 0:
                t_first = float(t_events_list[idx][0])
                if best_t is None or t_first < best_t:
                    best_t = t_first
                    best_idx = idx
        if best_idx is not None:
            t_event = best_t
            y_event = np.asarray(y_events_list[best_idx][0], dtype=float)
            termination_reason = event_labels[best_idx]
            termination_t_s = t_event
            termination_state = y_event
            if epoch_et is not None:
                termination_et = epoch_et + t_event
            logger.info(
                "propagate: terminal event %r fired at t=%.3f s "
                "(|r|=%.3f km)",
                termination_reason,
                t_event,
                float(np.linalg.norm(y_event[:3])),
            )

    logger.info(
        "propagate: %d samples, %d RHS calls, method=%s, zonal_degree=%d, "
        "gravity_degree=%d, gravity_order=%s, gravity_backend=%s, "
        "third_bodies=%s, solar_sail=%s, "
        "spherical_particle=%s, tumble_averaged_sail=%s, termination=%s",
        sol.t.size, sol.nfev, options.method, zonal_degree,
        gravity_degree, resolved_gravity_order,
        gravity_backend if gravity_degree > 0 else None,
        [b.label for b in third_bodies_list] or None,
        "on" if solar_sail is not None else "off",
        "on" if spherical_particle is not None else "off",
        "on" if tumble_averaged_sail is not None else "off",
        termination_reason,
    )

    return PropagationResult(
        t_s=np.asarray(sol.t, dtype=float),
        state_km_kmps=np.asarray(sol.y.T, dtype=float),
        method=options.method,
        rtol=options.rtol,
        atol=options.atol,
        mu_km3_s2=mu_km3_s2,
        epoch_et=epoch_et,
        solver_message=str(sol.message),
        n_rhs_calls=int(sol.nfev),
        metadata=metadata,
        termination_reason=termination_reason,
        termination_et=termination_et,
        termination_t_s=termination_t_s,
        termination_state_km_kmps=termination_state,
    )
