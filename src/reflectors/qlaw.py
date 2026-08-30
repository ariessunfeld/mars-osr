"""Q-law steering for the SRP Mars-escape spiral.

This module implements a Lyapunov / Q-law feedback steering law (Petropoulos)
specialised to the **escape** problem and adapted to a *solar sail*: the thrust
is not freely directable but constrained to the McInnes flat-sail acceleration
set, so the "pick the thrust direction" step is a search over the achievable
sail normal rather than the closed-form free-thrust optimum.

The escape law steers on the two orbit *radii* that bound the trajectory:

  * **apoapsis** ``r_a = a (1 + e)`` -- raise it so the orbit reaches the Hill
    sphere (the actual escape termination is ``|r| > r_Hill``, i.e. apoapsis
    crosses Hill); and
  * **periapsis** ``r_p = a (1 - e)`` -- defend it as a barrier so the orbit
    never enters the atmosphere.

Eccentricity is not steered for its own sake -- it is just an internal state
variable; what the mission cares about are the two radii. An optional explicit
``e`` term is available (off by default); when ``r_p`` is healthy the law
deliberately allows ``e`` to grow (the ``a edot`` contribution to ``rdot_a``
is the same order as ``(1+e) adot``, so a max-``rdot_a`` thrust often pumps
``e`` -- helpful for escape), and the barrier flips the sign of the ``e``
coupling automatically as ``r_p -> r_p_min`` to circularise out of trouble.

Rate-normalised blend
---------------------
The textbook Q-law is a sum of squared element "proximity quotients"
``Q = (1 + W_p P) Sum_oe W_oe S_oe (d_oe / oedot_xx)^2`` with ``d_oe = oe -
oe_T`` the element error and ``oedot_xx`` its locally-optimal rate maximum.
That form assumes the elements are transferred over *commensurable* times --
true for an orbit transfer, **false for an escape**. An escape raises ``a``
from ~3900 km to ~10^6 km (the ``a`` "time-to-go" is the whole mission) while
``e`` is merely regulated within ``[0, ~0.1]`` (the ``e`` "time-to-go" is a few
revolutions). With ``a_T`` = the Hill radius the ``a`` error ``d_a`` is ~3e5x
the ``e`` error, and *squaring* the quotient turns that ~4.5-order time-to-go
gap into a ~9-order term-magnitude gap: the ``e`` term becomes ~1e-7 of the
``a`` term and contributes negligibly to the steering gradient.

The implemented law instead uses a **rate-normalised blend** in which each rate's
contribution to ``dQ/dt`` is O(1) *by construction*. The law picks the
achievable thrust that maximises the escape reward

    R = W_apo (radot / radot_xx)              raise r_a  (-> escape)
      + W_p   rho_p(r_p) (rpdot / rpdot_xx)   raise r_p  (periapsis barrier)
      - W_e   rho_e(e)   (edot  / edot_xx)    reduce e   (optional, off by default)

with ``rdot_a = (1+e) adot + a edot`` and ``rdot_p = (1-e) adot - a edot``
(from ``r_a = a(1+e)``, ``r_p = a(1-e)``). Each rate is divided by its *local*
maximum so every ratio lies in ``[-1, 1]`` and the ``W`` weights are a
genuine O(1) trade. ``rho_p(r_p)`` is the Petropoulos periapsis penalty,
dormant when periapsis is healthy and exploding as ``r_p -> r_p_min`` -- the
hard barrier.

The ``rdot_a`` steering quantity. The mission's termination
condition is ``|r| > r_Hill`` (apoapsis reaches Hill), not ``a -> infinity``;
``r_a`` is the relevant boundary. And ``rdot_a`` decomposes as ``(1+e) adot +
a edot``: the ``a edot`` contribution at full transverse thrust is the *same
order* as the ``(1+e) adot`` contribution (e.g. at ``a~4000 km``,
``adot_xx*1.05 ~ 228 km/sol`` vs ``a*edot_xx ~ 216 km/sol``). An ``adot``-only
law captures only half the achievable escape rate -- it treats the ``edot``
that naturally comes with one-sided SRP thrust as collateral damage instead of
exploiting it. Steering on ``rdot_a`` exploits both. When ``r_p`` becomes
stressed, ``rho_p`` ramps and the barrier's ``-a*rho_p/rpdot_xx`` contribution
to the e-gradient overrides the apoapsis-pumping ``+a/rdot_a_xx`` term -- the
law automatically flips from "pump e for escape" to "circularise for safety"
without an explicit e-target.

Steering as a feedback law
--------------------------
``adot`` and ``edot`` are linear in the RTN thrust-acceleration components
(Gauss's variational equations; the orbit-normal component ``f_h`` affects
neither ``a`` nor ``e`` nor ``r_p``). The reward ``R`` is therefore linear in
the thrust acceleration, ``R = c . a_thrust``, and minimising ``dQ/dt :=
-R = g . a_thrust`` over the achievable McInnes accelerations picks the
steering. When no achievable orientation makes ``dQ/dt < 0`` the law coasts --
it commands the sail edge-on to the Sun (zero projected area): "feather."

References (primary)
--------------------
Petropoulos, A.E. (2014), *Low-Thrust Trajectories: An overview of the Q-law
and other analytic techniques*. The
Q-law / Lyapunov feedback structure, the feedback law ``dQ/dt = Sum
(dQ/d_oe) oedot``, the locally-optimal element-rate maxima ``oedot_xx``, and
the coast / effectivity idea.

Petropoulos, A.E. (2005), *Refinements to the Q-law for low-thrust orbit
transfers*, AAS 05-162 -- the periapsis penalty ``P``. The implemented
functional form follows the open-source reference implementation **pyqlaw**
(Yuri Shimane, https://github.com/Yuricst/pyqlaw):
``p_rp = exp(k_petro (1 - rp/rpmin))``.

McInnes, C.R. (1999), *Solar Sailing* -- the flat-sail optical force model, via
``reflectors.srp.mcinnes_srp_acceleration``.

Solar-sail adaptation
---------------------
The optimal sail normal is found by a 1-D search. For a fixed thrust-frame
gradient ``g`` (with ``dQ/dt = g . a_thrust``), the McInnes force ``a_SRP(n)``
minimising ``g . a_SRP`` has its sail normal in the plane spanned by the
sail-to-Sun direction ``s_hat`` and ``g`` (McInnes coplanarity: for a fixed
cone angle the in-plane normal extremises the force component along any target
direction). The search is therefore over a single sail-pitch angle within that
plane -- cheap enough for the propagator RHS hot loop, and verified against a
brute-force 2-D normal search in ``tests/test_qlaw.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from reflectors.elements import ClassicalElements
from reflectors.gauss import (
    eccentricity_rate_max,
    gauss_variational_rates,
    rtn_basis,
    semimajor_axis_rate_max,
)
from reflectors.srp import SolarSail, mcinnes_srp_acceleration

logger = logging.getLogger(__name__)

__all__ = [
    "QLawParams",
    "QLawSteering",
    "periapsis_penalty",
    "escape_reward_gradients",
    "evaluate_orbit_effectivity",
    "steer",
]


# Sail pitch search bound: n . s_hat = cos(delta) > 0 strictly inside +/-pi/2.
_DELTA_MAX_RAD = 0.5 * math.pi - 0.01

# A direction is treated as parallel to s_hat when its perpendicular component
# falls below this (dimensionless, components of unit vectors).
_PERP_TOL = 1.0e-9

# Eccentricity at or above which the law feathers. Beyond this point the rate
# maxima, which contain 1/(1-e) and p=a(1-e^2), lose conditioning; the
# altitude-floor event handles trajectories with insufficient periapsis.
_E_MAX_STEER = 0.95


@dataclass(frozen=True)
class QLawParams:
    """Parameters of the rate-normalised radii-based escape Q-law.

    The law raises apoapsis ``r_a = a(1+e)`` toward the Hill sphere with a
    barrier protecting periapsis ``r_p = a(1-e)`` from the atmosphere; see
    the module docstring for the reward ``R`` the steering maximises.

    Attributes
    ----------
    a_target_km
        Semi-major axis (km) at/above which the law stops steering and
        feathers -- the powered phase is over and the coast to the Hill sphere
        is unpowered. In practice the run terminates at the Hill-sphere
        ``RadiusCeiling`` event (apoapsis crossing) well before ``a`` reaches
        this value; it is a guard, not a steering target. Default: the Mars
        Hill radius.
    rp_min_km
        Periapsis-barrier reference radius ``r_p_min`` (km): the penalty
        ``rho_p`` is unity at ``r_p = r_p_min`` and explodes below it.
    w_apo
        Weight of the apoapsis-rate reward term ``W_apo (rdot_a / rdot_a_xx)``.
        Default 1.0.
    w_a
        Weight of an optional pure semi-major-axis-rate reward
        ``W_a (adot / adot_xx)`` -- "maximise orbital-energy gain." Default
        ``0.0``. Set ``w_apo=0, w_a=1`` to recover the naive "maximise adot
        + feather when impossible" steering (diagnostic / comparison; the
        apoapsis-rate framing is the default because it exploits the
        ``a*edot`` contribution that an adot-only law leaves on the table).
    w_e
        Weight of the optional explicit eccentricity-reduction term. **The
        mission does not care about ``e`` for its own sake** -- only about the
        radii, which the apoapsis and periapsis terms already address. Default
        ``w_e = 0``; with ``w_apo`` driving apoapsis and ``rho_p`` automatically
        flipping the e-coupling when periapsis is stressed, no explicit
        circularisation term is needed. The ``w_e > 0`` mode is retained as a
        tunable; its non-degeneracy is regression-tested
        (``tests/test_qlaw.py::test_eccentricity_term_genuinely_co_steers``).
    e_ref
        Eccentricity-urgency reference (used only when ``w_e > 0``): the
        ``e`` effort scales as ``rho_e = e / e_ref``. Default 0.05.
    w_p
        Weight of the periapsis-barrier reward term. Default 1.0; larger values
        strengthen the barrier against periapsis collapse.
    k_petro
        Periapsis-penalty sharpness ``k`` in ``P = exp(k (1 - r_p/r_p_min))``.
        Larger ``k`` -> a steeper, more localised barrier (closer to a hard
        wall at ``r_p_min``). Default 10.0 -- steeper than the Petropoulos
        orbit-transfer default (1.0): here the penalty is a safety barrier,
        not a transfer-shaping term.
    eta_a_threshold, eta_r_threshold
        Petropoulos effectivity-coast thresholds. The law coasts when the
        current state's achievable ``dQ/dt`` is *less effective* than a
        threshold fraction of the best ``dQ/dt`` achievable across all phases
        of the current osculating orbit. Specifically the law coasts when

            eta_a := |qdot_current / qdot_min|       < eta_a_threshold, or
            eta_r := (qdot_current - qdot_max)
                     / (qdot_min - qdot_max)         < eta_r_threshold

        with ``qdot_min`` / ``qdot_max`` the most / least negative achievable
        ``dQ/dt`` over the orbit (from :func:`evaluate_orbit_effectivity`).
        This is Petropoulos's coast technique (pyqlaw ``_qlaw.py``): it
        prevents the law from thrusting on arcs where the achievable ``dQ/dt``
        is only marginally negative -- arcs which under a one-sided SRP
        thrust pump ``e`` as a side effect. Defaults
        ``0.0`` for both -> coasting disabled, the law thrusts whenever any
        orientation gives ``dQ/dt < 0`` (the default behaviour). Typical
        literature values are 0.5-0.9; tune per mission.
    effectivity_n_samples
        Number of true-anomaly samples used to estimate ``qdot_min`` /
        ``qdot_max`` over the orbit. Default 36 (every 10 degrees).
    effectivity_refresh_steps
        Recompute the effectivity envelope every this many RK4 steps. The
        envelope is a slow function of orbit shape (and a slower function
        of Sun direction), so refreshing every half-orbit at 200 steps/orbit
        is conservative and inexpensive. Default 100.
    """

    a_target_km: float
    rp_min_km: float
    w_apo: float = 1.0
    w_a: float = 0.0
    w_e: float = 0.0
    e_ref: float = 0.05
    w_p: float = 1.0
    k_petro: float = 10.0
    # Petropoulos effectivity-coast thresholds (off by default; both 0 ->
    # the law thrusts whenever any achievable orientation gives dQ/dt < 0,
    # which causes thrust on marginally-helpful arcs that pump e as a side
    # effect.
    eta_a_threshold: float = 0.0
    eta_r_threshold: float = 0.0
    effectivity_n_samples: int = 36
    effectivity_refresh_steps: int = 100

    def __post_init__(self) -> None:
        if self.a_target_km <= 0.0:
            raise ValueError(f"a_target_km must be > 0, got {self.a_target_km}")
        if self.rp_min_km <= 0.0:
            raise ValueError(f"rp_min_km must be > 0, got {self.rp_min_km}")
        if self.w_apo < 0.0 or self.w_a < 0.0 or self.w_e < 0.0:
            raise ValueError("w_apo, w_a, w_e must be >= 0")
        if self.e_ref <= 0.0:
            raise ValueError(f"e_ref must be > 0, got {self.e_ref}")
        if self.w_p < 0.0:
            raise ValueError(f"w_p must be >= 0, got {self.w_p}")
        if self.k_petro < 0.0:
            raise ValueError(f"k_petro must be >= 0, got {self.k_petro}")
        if not (0.0 <= self.eta_a_threshold <= 1.0):
            raise ValueError(
                f"eta_a_threshold must satisfy 0 <= eta_a <= 1, got "
                f"{self.eta_a_threshold}"
            )
        if not (0.0 <= self.eta_r_threshold <= 1.0):
            raise ValueError(
                f"eta_r_threshold must satisfy 0 <= eta_r <= 1, got "
                f"{self.eta_r_threshold}"
            )
        if self.effectivity_n_samples < 4:
            raise ValueError(
                f"effectivity_n_samples must be >= 4, got "
                f"{self.effectivity_n_samples}"
            )
        if self.effectivity_refresh_steps < 1:
            raise ValueError(
                f"effectivity_refresh_steps must be >= 1, got "
                f"{self.effectivity_refresh_steps}"
            )


@dataclass(frozen=True)
class QLawSteering:
    """Result of one Q-law steering evaluation.

    Attributes
    ----------
    n_star_j2000
        Desired sail-normal unit vector (J2000). When ``thrust`` is True this
        is the reward-maximising orientation; when False it is the feathered
        (edge-on) orientation. The attitude controller slews the *actual*
        sail normal toward this target subject to the rate/accel limits.
    thrust
        True if an achievable sail orientation strictly increases the escape
        reward (``dQ/dt = -R < 0``); False if the law coasts (feathers).
    dQ_dt
        The most negative achievable ``dQ/dt = -R`` (dimensionless: ``R`` is a
        sum of rate ratios). Zero when coasting.
    q_value
        Periapsis penalty ``P = exp(k(1 - r_p/r_p_min))`` at the current state
        (diagnostic: ``P >> 1`` flags a stressed periapsis). Zero for an
        escaped (``a <= 0``) state.
    f_char_km_s2
        Characteristic SRP acceleration -- the face-on (``n = s_hat``)
        magnitude at the current heliocentric distance -- the thrust scale
        that normalises the element-rate maxima (diagnostic).
    """

    n_star_j2000: np.ndarray
    thrust: bool
    dQ_dt: float
    q_value: float
    f_char_km_s2: float


def periapsis_penalty(rp_km: float, params: QLawParams) -> float:
    """Petropoulos periapsis penalty ``P = exp(k (1 - r_p / r_p_min))``.

    Unity at ``r_p = r_p_min``, decaying toward zero well above it and
    growing without bound below it. Cited form (pyqlaw ``_symbolic.py``:
    ``p_rp = sym.exp(k_petro*(1.0 - rp/rpmin))``).
    """
    return math.exp(params.k_petro * (1.0 - rp_km / params.rp_min_km))


def escape_reward_gradients(
    elements: ClassicalElements,
    f_char_km_s2: float,
    params: QLawParams,
) -> tuple[float, float]:
    """Effective steering gradients ``(G_a, G_e)`` of the escape reward.

    Returns the coefficients such that ``dQ/dt = G_a * da/dt + G_e * de/dt``
    equals ``-R`` (negative of the rate-normalised escape reward). Minimising
    ``dQ/dt`` over the achievable thrust therefore maximises ``R``.

    Derivation. The reward (module docstring) is

        R = w_apo (rdot_a/rdot_a_xx)  +  w_p rho_p (rdot_p/rdot_p_xx)
            - w_e rho_e (edot/edot_xx),

    with the radii rates ``rdot_a = (1+e) adot + a edot`` (from
    ``r_a = a(1+e)``) and ``rdot_p = (1-e) adot - a edot`` (from
    ``r_p = a(1-e)``). Collecting ``adot`` and ``edot`` coefficients and using
    ``dQ/dt = -R``:

        G_a = -( w_apo (1+e)/rdot_a_xx + w_p rho_p (1-e)/rdot_p_xx )
        G_e = -( w_apo a    /rdot_a_xx - w_p rho_p a    /rdot_p_xx
                 - w_e rho_e/edot_xx )

    ``G_a`` is always negative (every term raises ``a`` or ``r_p`` or ``r_a``).
    ``G_e`` flips sign automatically with periapsis stress: when
    ``r_p`` is healthy (``rho_p ~ 0``) the apoapsis term ``-w_apo a/rdot_a_xx``
    dominates -> ``G_e < 0`` -> the steering rewards ``edot > 0`` (pump ``e``
    for faster apoapsis growth); as ``r_p -> r_p_min`` the barrier term
    ``+w_p rho_p a/rdot_p_xx`` overtakes it -> ``G_e > 0`` -> the steering
    fights ``edot > 0``. The transition is governed by the single shape
    ``rho_p(r_p)`` with no e-target.

    Normalisers (all evaluated at the *current* ``(a, e)``; ``f_char`` is the
    thrust-magnitude scale):

    - ``adot_xx``, ``edot_xx`` -- the locally-optimal element-rate maxima
      (:mod:`reflectors.gauss`).
    - ``rdot_a_xx = (1+e) adot_xx + a edot_xx`` -- triangle-inequality bound
      on ``|rdot_a|``; tight at periapsis (transverse thrust hits both maxes
      simultaneously) and loose elsewhere -- a valid normaliser either way.
    - ``rdot_p_xx = (1-e) adot_xx + a edot_xx`` -- analogous bound on
      ``|rdot_p|``.
    - ``rho_e = e / e_ref`` -- eccentricity urgency (unused when ``w_e = 0``).
    - ``rho_p`` folded in via :func:`periapsis_penalty`.

    Unlike the textbook squared proximity quotient, these normalisers appear
    only as instantaneous scale factors of a *linear* reward -- they are never
    differentiated with respect to the elements, so using the true
    ``(a, e)``-dependent rate maxima introduces no spurious ``e``-pumping
    gradient, unlike the squared-quotient formulation.

    Requires ``a > 0`` and ``0 <= e < 1`` (:func:`steer` feathers escaped /
    near-parabolic states before reaching here) and ``f_char > 0``.
    """
    a = elements.a_km
    e = elements.e
    mu = elements.mu_km3_s2

    adot_xx = semimajor_axis_rate_max(a, e, f_char_km_s2, mu)
    edot_xx = eccentricity_rate_max(a, e, f_char_km_s2, mu)
    rdot_a_xx = (1.0 + e) * adot_xx + a * edot_xx
    rdot_p_xx = (1.0 - e) * adot_xx + a * edot_xx

    rp = a * (1.0 - e)
    rho_p = params.w_p * periapsis_penalty(rp, params)
    rho_e = e / params.e_ref

    g_a = -(
        params.w_apo * (1.0 + e) / rdot_a_xx
        + params.w_a / adot_xx
        + rho_p * (1.0 - e) / rdot_p_xx
    )
    g_e = -(
        params.w_apo * a / rdot_a_xx
        - rho_p * a / rdot_p_xx
        - params.w_e * rho_e / edot_xx
    )
    return g_a, g_e


def _state_from_elements_inertial(
    a_km: float,
    e: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    nu_rad: float,
    mu_km3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian ``(r, v)`` from classical elements in the *same* inertial
    frame the elements are referenced to.

    Standard textbook perifocal-to-inertial rotation
    ``R3(-RAAN) R1(-i) R3(-argp)`` (Vallado 2013 Sec. 2.5). Since the rotation
    is between two inertial frames sharing only the frame's reference
    elements, the routine is frame-agnostic -- if the elements are in J2000,
    the returned state is in J2000; if MME2000, MME2000. The escape law uses
    J2000 elements from :func:`reflectors.gauss.osculating_elements`, so the
    returned state is J2000.

    Used by :func:`evaluate_orbit_effectivity` to sample states around the
    current osculating orbit by varying ``nu_rad``. Does not call SPICE.
    """
    p = a_km * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu_rad))
    sn, cn = math.sin(nu_rad), math.cos(nu_rad)
    r_pf = np.array([r * cn, r * sn, 0.0])
    h = math.sqrt(mu_km3_s2 * p)
    v_pf = (mu_km3_s2 / h) * np.array([-sn, e + cn, 0.0])

    sR, cR = math.sin(raan_rad), math.cos(raan_rad)
    si, ci = math.sin(inc_rad), math.cos(inc_rad)
    sw, cw = math.sin(argp_rad), math.cos(argp_rad)
    # R3(-RAAN) R1(-i) R3(-argp); columns are the perifocal axes in the
    # inertial frame.
    rot = np.array([
        [cR * cw - sR * ci * sw, -cR * sw - sR * ci * cw,  sR * si],
        [sR * cw + cR * ci * sw, -sR * sw + cR * ci * cw, -cR * si],
        [si * sw,                 si * cw,                 ci],
    ])
    return rot @ r_pf, rot @ v_pf


def evaluate_orbit_effectivity(
    elements: ClassicalElements,
    s_hat: np.ndarray,
    srp_pressure_pa: float,
    sail: SolarSail,
    params: QLawParams,
    n_samples: Optional[int] = None,
) -> Optional[tuple[float, float]]:
    """Estimate ``(qdot_min, qdot_max)`` across the orbit at the current state.

    Samples ``n_samples`` true anomalies uniformly in ``[0, 2 pi)``, builds the
    Cartesian state at each from the current orbit's ``(a, e, i, RAAN, argp,
    mu)``, calls :func:`steer` (with ``effectivity_envelope=None`` to disable
    recursion) at each, and returns ``(min, max)`` of the achieved ``dQ/dt``
    across the samples that successfully thrust.

    The Sun direction and SRP pressure are held fixed at the inputs across all
    samples -- a low-Mars orbit sweeps the Sun by only ~0.04 deg per period,
    so the approximation is negligible.

    Returns ``None`` if no sample's pitch search produces ``dQ/dt < 0`` (no
    thrust achievable anywhere around the orbit -- e.g. the law is fully
    feathered at this geometry).

    Used by :func:`steer` (via the effectivity-coast gate) to decide whether
    the *current* phase of the orbit is one of the most effective moments to
    thrust. Petropoulos's coast technique (pyqlaw ``_qlaw.py``); see the
    ``QLawParams`` ``eta_a_threshold`` / ``eta_r_threshold`` docstring.
    """
    if n_samples is None:
        n_samples = params.effectivity_n_samples

    qdots: list[float] = []
    s_hat_arr = np.asarray(s_hat, dtype=float)
    for k in range(int(n_samples)):
        nu_k = 2.0 * math.pi * k / n_samples
        r_k, v_k = _state_from_elements_inertial(
            elements.a_km,
            elements.e,
            elements.inclination_rad,
            elements.raan_rad,
            elements.argp_rad,
            nu_k,
            elements.mu_km3_s2,
        )
        # Rebuild a ClassicalElements at this nu so steer sees the right nu
        # (the rate-max calls don't depend on nu, but the Gauss rates do).
        synth = ClassicalElements(
            a_km=elements.a_km,
            e=elements.e,
            inclination_rad=elements.inclination_rad,
            raan_rad=elements.raan_rad,
            argp_rad=elements.argp_rad,
            nu_rad=nu_k,
            period_s=elements.period_s,
            mu_km3_s2=elements.mu_km3_s2,
        )
        res = steer(
            synth, r_k, v_k, s_hat_arr, srp_pressure_pa, sail,
            current_n_hat=s_hat_arr, params=params,
            effectivity_envelope=None,
        )
        if res.thrust:
            qdots.append(res.dQ_dt)

    if not qdots:
        return None
    return min(qdots), max(qdots)


def _minimize_scalar(func, lo: float, hi: float) -> tuple[float, float]:
    """Minimum of a smooth scalar ``func`` on ``[lo, hi]``.

    Coarse grid scan to bracket the minimum, then golden-section refinement.
    ``func`` must accept a numpy array (the grid scan is one vectorised call)
    and scalars (the refinement). Returns ``(x_min, f_min)``. Used for the
    1-D sail-pitch search; the 1e-4 rad refinement floor is far finer than
    the pitch precision that matters for the force.
    """
    n_grid = 25
    xs = np.linspace(lo, hi, n_grid)
    fs = np.asarray(func(xs), dtype=float)
    i = int(np.argmin(fs))
    # Bracket around the best grid node.
    x_lo = float(xs[max(0, i - 1)])
    x_hi = float(xs[min(n_grid - 1, i + 1)])
    # Golden-section search on the bracket.
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = x_hi - inv_phi * (x_hi - x_lo)
    d = x_lo + inv_phi * (x_hi - x_lo)
    fc = float(func(c))
    fd = float(func(d))
    for _ in range(30):
        if fc < fd:
            x_hi, d, fd = d, c, fc
            c = x_hi - inv_phi * (x_hi - x_lo)
            fc = float(func(c))
        else:
            x_lo, c, fc = c, d, fd
            d = x_lo + inv_phi * (x_hi - x_lo)
            fd = float(func(d))
        if abs(x_hi - x_lo) < 1.0e-4:
            break
    x_min = 0.5 * (x_lo + x_hi)
    return x_min, float(func(x_min))


def _feather_normal(
    current_n_hat: np.ndarray,
    s_hat: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Edge-on (``n . s_hat = 0``) sail normal nearest the current attitude.

    Feathering = zero projected area to the Sun. Among the circle of edge-on
    orientations the one nearest ``current_n_hat`` minimises the slew the
    attitude controller must perform. If the current normal is (near-)
    parallel to ``s_hat`` the perpendicular projection is degenerate and
    ``fallback`` (projected perpendicular to ``s_hat``) is used instead.
    """
    n_perp = current_n_hat - float(np.dot(current_n_hat, s_hat)) * s_hat
    norm = float(np.linalg.norm(n_perp))
    if norm > _PERP_TOL:
        return n_perp / norm
    fb_perp = fallback - float(np.dot(fallback, s_hat)) * s_hat
    fb_norm = float(np.linalg.norm(fb_perp))
    if fb_norm > _PERP_TOL:
        return fb_perp / fb_norm
    # Both degenerate: build any vector perpendicular to s_hat.
    trial = np.array([1.0, 0.0, 0.0])
    if abs(s_hat[0]) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    perp = trial - float(np.dot(trial, s_hat)) * s_hat
    return perp / float(np.linalg.norm(perp))


def steer(
    elements: ClassicalElements,
    r_vec_km: np.ndarray,
    v_vec_kmps: np.ndarray,
    s_hat: np.ndarray,
    srp_pressure_pa: float,
    sail: SolarSail,
    current_n_hat: np.ndarray,
    params: QLawParams,
    *,
    effectivity_envelope: Optional[tuple[float, float]] = None,
) -> QLawSteering:
    """Evaluate the escape Q-law: desired sail normal + thrust/coast decision.

    Parameters
    ----------
    elements
        Osculating elements at the current state
        (``reflectors.gauss.osculating_elements``).
    r_vec_km, v_vec_kmps
        Current Cartesian position / velocity (J2000), shape (3,).
    s_hat
        Sail-to-Sun unit vector (J2000).
    srp_pressure_pa
        Solar radiation pressure at the sail (Pa). Pass ``0.0`` when the sail
        is in shadow -- the law then coasts (no force is achievable).
    sail
        ``reflectors.srp.SolarSail`` bus.
    current_n_hat
        Current actual sail-normal unit vector (J2000) -- used only to choose
        the nearest feathered orientation when coasting.
    params
        ``QLawParams``.
    effectivity_envelope
        Optional ``(qdot_min, qdot_max)`` -- the most / least negative
        achievable ``dQ/dt`` across the current orbit, from
        :func:`evaluate_orbit_effectivity`. When supplied AND
        ``params.eta_a_threshold`` or ``params.eta_r_threshold`` is positive,
        the law applies the Petropoulos effectivity coast: even if the
        current state's best ``dQ/dt`` is negative, the law feathers when
        the *effectivity* (``qdot_current / qdot_min`` or its relative
        variant) is below the threshold -- preventing thrust on
        marginally-helpful arcs that pump ``e`` as a side effect. ``None``
        (default) disables the gate and is also the recursion
        guard used by :func:`evaluate_orbit_effectivity`).

    Returns
    -------
    QLawSteering
        ``n_star_j2000`` (desired normal), ``thrust`` flag, and diagnostics.
    """
    a = elements.a_km
    e = elements.e

    r_hat, theta_hat, h_hat = rtn_basis(r_vec_km, v_vec_kmps)

    # Terminal-state guard. There is nothing useful to steer once the orbit
    # is (near-)parabolic or hyperbolic (a <= 0; the coast to the Hill sphere
    # is unpowered), once a reaches a_target, or once e is so high the orbit
    # has effectively de-orbited (the altitude-floor event then terminates).
    escaped = (a <= 0.0) or (a >= params.a_target_km) or (e >= _E_MAX_STEER)

    # Characteristic SRP acceleration: face-on (n = s_hat) force magnitude.
    f_char = 0.0
    if srp_pressure_pa > 0.0:
        a_face = mcinnes_srp_acceleration(s_hat, s_hat, srp_pressure_pa, sail)
        f_char = float(np.linalg.norm(a_face))

    # Diagnostic periapsis penalty (undefined for an escaped state).
    penalty = (
        periapsis_penalty(a * (1.0 - e), params) if a > 0.0 else 0.0
    )

    if escaped or f_char <= 0.0:
        # No useful thrust in a terminal state or shadow: feather.
        n_star = _feather_normal(np.asarray(current_n_hat, float), s_hat, h_hat)
        return QLawSteering(
            n_star_j2000=n_star,
            thrust=False,
            dQ_dt=0.0,
            q_value=penalty,
            f_char_km_s2=f_char,
        )

    # Effective gradients of the rate-normalised escape reward, and the linear
    # map thrust-acceleration -> dQ/dt. da/dt, de/dt are linear in the RTN
    # thrust components; unit-thrust calls give the sensitivity coefficients
    # (f_h affects neither a nor e).
    g_a, g_e = escape_reward_gradients(elements, f_char, params)
    rates_r = gauss_variational_rates(elements, 1.0, 0.0, 0.0)
    rates_t = gauss_variational_rates(elements, 0.0, 1.0, 0.0)
    g_r = g_a * rates_r.da_dt_km_s + g_e * rates_r.de_dt_per_s
    g_t = g_a * rates_t.da_dt_km_s + g_e * rates_t.de_dt_per_s
    # Gradient in J2000: dQ/dt = g_j2000 . a_thrust.
    g_j2000 = g_r * r_hat + g_t * theta_hat
    g_mag = float(np.linalg.norm(g_j2000))

    if g_mag == 0.0:
        # Flat reward: nothing to gain. Feather.
        n_star = _feather_normal(np.asarray(current_n_hat, float), s_hat, h_hat)
        return QLawSteering(n_star, False, 0.0, penalty, f_char)

    # Desired thrust direction: most negative dQ/dt -> a_thrust along -g.
    d_hat = -g_j2000 / g_mag
    d_par = float(np.dot(d_hat, s_hat))
    d_perp_vec = d_hat - d_par * s_hat
    d_perp = float(np.linalg.norm(d_perp_vec))

    if d_perp > _PERP_TOL:
        e_hat = d_perp_vec / d_perp

        def objective(delta):
            # delta: scalar or 1-D array. n(delta) = cos(d) s_hat + sin(d) e_hat
            # vectorises over the leading axis, so the grid scan is one
            # batched SRP-force evaluation. Returns dQ/dt = g . a_SRP.
            d = np.asarray(delta, dtype=float)
            n = (
                np.cos(d)[..., np.newaxis] * s_hat
                + np.sin(d)[..., np.newaxis] * e_hat
            )
            a_srp = mcinnes_srp_acceleration(n, s_hat, srp_pressure_pa, sail)
            return np.sum(a_srp * g_j2000, axis=-1)

        delta_opt, dQ_dt_best = _minimize_scalar(
            objective, -_DELTA_MAX_RAD, _DELTA_MAX_RAD
        )
        n_opt = math.cos(delta_opt) * s_hat + math.sin(delta_opt) * e_hat
    else:
        # Desired direction is (anti-)parallel to s_hat: the only useful
        # normal is face-on (n = s_hat); the force is then purely anti-sunward.
        n_opt = np.asarray(s_hat, dtype=float)
        a_srp = mcinnes_srp_acceleration(n_opt, s_hat, srp_pressure_pa, sail)
        dQ_dt_best = float(np.dot(g_j2000, a_srp))

    if dQ_dt_best < 0.0:
        # An achievable orientation strictly increases the escape reward.
        # Apply the Petropoulos effectivity-coast gate, if armed:
        if effectivity_envelope is not None and (
            params.eta_a_threshold > 0.0 or params.eta_r_threshold > 0.0
        ):
            qdot_min, qdot_max = effectivity_envelope
            if qdot_min is not None and qdot_min < 0.0:
                # eta_a in (0, 1] for qdot_min <= dQ_dt_best <= 0; closer to 1
                # means closer to the orbit's best moment to thrust.
                eta_a = dQ_dt_best / qdot_min
                if qdot_max is not None and (qdot_min - qdot_max) != 0.0:
                    eta_r = (dQ_dt_best - qdot_max) / (qdot_min - qdot_max)
                else:
                    eta_r = 1.0
                if (
                    eta_a < params.eta_a_threshold
                    or eta_r < params.eta_r_threshold
                ):
                    # Off-effectivity arc: coast (feather) to avoid pumping
                    # e as a side effect of a marginally-helpful thrust.
                    n_star = _feather_normal(
                        np.asarray(current_n_hat, float), s_hat, h_hat
                    )
                    return QLawSteering(n_star, False, 0.0, penalty, f_char)
        # Effective enough (or no envelope): thrust.
        n_star = n_opt / float(np.linalg.norm(n_opt))
        return QLawSteering(n_star, True, dQ_dt_best, penalty, f_char)

    # No achievable orientation increases the reward: coast (feather).
    n_star = _feather_normal(np.asarray(current_n_hat, float), s_hat, h_hat)
    return QLawSteering(n_star, False, 0.0, penalty, f_char)
