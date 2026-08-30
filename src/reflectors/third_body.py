"""Third-body gravitational perturbation on a Mars-centered sail.

Target physics: acceleration of a test particle in a Mars-centered, non-
rotating frame due to the gravity of an external body (Sun, Phobos,
Deimos, Jupiter barycenter, ...). The perturbing body is itself
accelerating the Mars center, so the relative acceleration of the sail
is the difference between the direct pull of the third body on the sail
and the pull of the third body on Mars (the "indirect" term).

Reference (primary): Montenbruck & Gill (2000), *Satellite Orbits:
Models, Methods, and Applications*, §3.3.1 ("Perturbing Acceleration"),
Eq. (3.37), p. 69. Full citation:

    Montenbruck, O. and Gill, E. (2000). *Satellite Orbits: Models,
    Methods, and Applications*. Springer, Berlin. Chapter 3.

Eq. (3.37) in Mars-centered, non-rotating coordinates (translated from
Montenbruck's Earth-centered convention by a straight symbol swap):

    a_3b(r) = mu_3 * [ (r_3 - r) / |r_3 - r|^3  -  r_3 / |r_3|^3 ]

where:
    - r     : sail position relative to Mars center, in any inertial
              frame (J2000 throughout this package), km.
    - r_3   : third-body position relative to Mars center, same frame, km.
    - mu_3  : gravitational parameter of the third body, km^3/s^2.

Sign convention: both terms carry a leading ``+mu_3``; the minus sign
between the two terms is the split between direct (first) and indirect
(second). At r = 0 (sail coincident with Mars center) the two terms
cancel exactly -- in a Mars-centered non-rotating frame, free-falling
Mars sees no Sun gravity. Machine-precision pinned in the test suite.

Direct-form precision regime
----------------------------

Montenbruck presents only the direct form (his Eq. 3.37). The standard
motivation for switching to the Battin/Encke stable form is catastrophic
cancellation when |r| << |r_3|. Quantitative precision budget for
Mars/Sun at low Mars orbit:

    individual term ~ mu_Sun / |r_Sun|^2 ~ 1.327e11 / (2.28e8)^2
                                        ~ 2.55e-6 km/s^2
    tidal residual  ~ 2 mu_Sun r_sat / |r_Sun|^3
                    ~ 2 * 1.327e11 * 3796 / (2.28e8)^3
                    ~ 8.5e-11 km/s^2
    residual / term ~ 3.3e-5  (~4.5 decimal digits lost to cancellation)
    double-precision noise in each term ~ 2.55e-6 * 1e-16 ~ 2.55e-22 km/s^2
    noise / residual ratio ~ 3e-12

The direct form therefore retains ~11 digits of relative precision on the
tidal residual for the Mars/Sun problem, comfortably above the DOP853 default
rtol of 1e-12. A precision-headroom regression pins the applicable numerical
regime; more cancellation-prone geometries require the Battin stable form
(Battin 1999, Eq. 8.60). See
``tests/test_third_body.py::test_sun_third_body_precision_headroom_...``.

Conservative-force validation: the third-body perturbation potential
in Mars-centered non-rotating coordinates is

    U_3b(r) = -mu_3 * [ 1/|r - r_3|  -  1/|r_3|  -  r . r_3 / |r_3|^3 ]

(Montenbruck Eq. 3.38 on p. 69, with the constant term ``-1/|r_3|``
included so U_3b(0) = 0, and the linear term ``- r . r_3 / |r_3|^3``
representing the indirect-acceleration potential). A finite-difference
of this potential matches the analytic acceleration to the central-
difference truncation floor (~1e-6 relative). See the conservative-
force test in the suite.

Magnitude orientation at Mars LMO (400 km altitude, Mars-Sun distance
1.524 AU):
    |a_Sun|      ~ 8e-11 km/s^2  (tidal; dominant third-body term)
    |a_Phobos|   ~ 1e-13 km/s^2 when sail is far from Phobos orbit
    |a_Deimos|   ~ 1e-14 km/s^2
    |a_Jupiter|  ~ 4e-15 km/s^2 (framework correctness check; five+
                                 orders below every other retained term)

Compare against Mars J_2 at the same altitude, |a_J2| ~ 7e-6 km/s^2.
Third-body terms are ~5 orders smaller than the dominant harmonic but
not negligible on multi-orbit timescales -- they drive long-period
drifts of the orbit plane that matter for sun-synchronous maintenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from reflectors.dynamics import body_gm_km3_per_s2
from reflectors.ephemeris import (
    SUN_NAIF_ID,
    spice_state_at_et,
    sun_state_j2000,
)
from reflectors.mars_constants import (
    DEIMOS_GM_KONOPLIV_2020_KM3_S2,
    PHOBOS_GM_KONOPLIV_2020_KM3_S2,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Body spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThirdBody:
    """Specification of a single third (perturbing) body.

    Immutable so instances can be safely shared across propagations and
    used as dict keys. The factory helpers below pull ``mu_km3_s2`` from
    the SPICE kernel pool at construction time; callers who want to
    override the GM (e.g. sensitivity studies) can instantiate the
    dataclass directly.

    Attributes
    ----------
    naif_id
        NAIF integer body ID. Used as the target argument to
        ``spice.spkezr`` to fetch the body's position.
    mu_km3_s2
        Gravitational parameter of the body, km^3/s^2.
    label
        Human-readable name. Purely informational (logs, metadata).
    """

    naif_id: int
    mu_km3_s2: float
    label: str


def sun_third_body() -> ThirdBody:
    """Sun perturber spec (NAIF 10, BODY10_GM from ``gm_de440.tpc``)."""
    return ThirdBody(naif_id=10, mu_km3_s2=body_gm_km3_per_s2(10), label="SUN")


def moon_third_body() -> ThirdBody:
    """Moon perturber spec (NAIF 301, BODY301_GM from ``gm_de440.tpc``).

    The dominant non-central perturber for an EARTH-centred escape: the Moon
    orbits at ~384,400 km, well INSIDE Earth's Hill sphere (~1.496e6 km, the
    escape ceiling), so an outbound spiral threads cislunar space and the
    lunar tide is first-order (unlike Mars, whose moons Phobos/Deimos are
    ~1e-13 km/s^2 and negligible). GM from the kernel pool (gm_de440.tpc
    carries BODY301_GM ~ 4902.800 km^3/s^2). Position fetched from DE440 via
    ``third_body_acceleration_from_spice`` with ``observer_naif_id=399``.
    """
    return ThirdBody(naif_id=301, mu_km3_s2=body_gm_km3_per_s2(301), label="MOON")


def earth_third_body() -> ThirdBody:
    """Earth perturber spec (NAIF 399, BODY399_GM from ``gm_de440.tpc``).

    For the SUN-centred interplanetary cruise: Earth is a third body the
    heliocentric sail flies past near departure (the cruise starts at Earth's
    Hill-sphere boundary). GM from the kernel pool; position fetched from DE440
    via ``third_body_acceleration_from_spice`` with ``observer_naif_id=10``
    (Sun).
    """
    return ThirdBody(naif_id=399, mu_km3_s2=body_gm_km3_per_s2(399), label="EARTH")


def mars_third_body() -> ThirdBody:
    """Mars perturber spec (NAIF 499, BODY499_GM from ``gm_de440.tpc``).

    For the SUN-centred interplanetary cruise: Mars is a third body the
    heliocentric sail approaches near arrival (the cruise ends at Mars's
    Hill-sphere capture node). GM from the kernel pool; position fetched from
    DE440 via ``third_body_acceleration_from_spice`` with ``observer_naif_id=10``
    (Sun).
    """
    return ThirdBody(naif_id=499, mu_km3_s2=body_gm_km3_per_s2(499), label="MARS")


def phobos_third_body() -> ThirdBody:
    """Phobos perturber spec (NAIF 401, GM from Konopliv 2020 / MRO120F).

    Uses ``PHOBOS_GM_KONOPLIV_2020_KM3_S2 = 7.10e-4`` km^3/s^2 from the
    PDS label jgmro_120f_sha.lbl (Konopliv et al. 2020), NOT the DE440
    kernel-pool value ``BODY401_GM ~ 7.0875e-4`` km^3/s^2 from
    gm_de440.tpc. The two agree within Konopliv's published 1-sigma
    error bar (5e-6 km^3/s^2) but differ by ~0.18%.

    This factory is intended for use as a third-body perturber alongside
    MRO120F gravity. Konopliv 2020 jointly fit the
    Mars-system spherical-harmonic field, Mars-planet GM, Phobos GM, and
    Deimos GM from a single 18-year MRO/Odyssey/MGS tracking data set.
    Pairing the Konopliv-fitted Phobos GM with the Konopliv-fitted
    harmonics keeps the dynamics on a single self-consistent solution.
    The propagator's central-mu decoupling logic in
    ``reflectors.dynamics.propagate`` subtracts this body's
    ``mu_km3_s2`` from the lumped MRO120F header mu, so using the
    Konopliv value here makes the central-mu correction land exactly
    on Mars-planet-alone (Konopliv).

    Position is still fetched from DE440 / mar099 via SPICE; the
    moon-position ephemeris and the moon-mass third-body GM are
    technically from different fits. The mass mismatch enters the
    sail's acceleration directly (~0.2% on a ~3.5e-12 km/s^2 perturber
    => 7e-15 km/s^2 sub-noise effect); the position mismatch shows up
    as a smaller systematic via the Konopliv solution's slightly
    different Phobos orbit. Both are below any meaningful science
    threshold here.
    """
    return ThirdBody(
        naif_id=401,
        mu_km3_s2=PHOBOS_GM_KONOPLIV_2020_KM3_S2,
        label="PHOBOS",
    )


def deimos_third_body() -> ThirdBody:
    """Deimos perturber spec (NAIF 402, GM from Konopliv 2020 / MRO120F).

    Uses ``DEIMOS_GM_KONOPLIV_2020_KM3_S2 = 9.68e-5`` km^3/s^2 from the
    PDS label jgmro_120f_sha.lbl (Konopliv et al. 2020), NOT the DE440
    kernel-pool value ``BODY402_GM ~ 9.6156e-5`` km^3/s^2. Differ by
    ~0.54%, well within Konopliv's 1.30e-5 1-sigma bar.

    Same self-consistency rationale as ``phobos_third_body``: Konopliv
    2020 jointly fit Mars / Phobos / Deimos GMs alongside the MRO120F
    harmonic field, so this is the matched mass for the matched gravity
    field. The propagator decouples this from the lumped central mu via
    ``reflectors.dynamics.propagate``.
    """
    return ThirdBody(
        naif_id=402,
        mu_km3_s2=DEIMOS_GM_KONOPLIV_2020_KM3_S2,
        label="DEIMOS",
    )


def jupiter_third_body() -> ThirdBody:
    """Jupiter perturber spec.

    Uses NAIF 5 (Jupiter barycenter) rather than 599 (Jupiter planet
    centre) for two reasons:
      1. DE440 + ``mar099.bsp`` (the default ephemerides) carry
         barycenter 5 but NOT planet 599 -- spkezr on 599 raises
         SPICE(SPKINSUFFDATA). Using 5 avoids pulling an additional
         Jupiter-specific SPK for a body whose perturbation is already
         five orders of magnitude below the Sun tide at Mars.
      2. The Jupiter-system satellites (Io, Europa, Ganymede, Callisto)
         tug the planet-to-barycenter offset by ~1e5 km; at the Mars
         perspective (|r| ~ 8.6e8 km to Jupiter) this is ~1e-4 relative,
         negligible against the already-tiny Jupiter tide. Using the
         barycenter position with the barycenter GM (BODY5_GM, which
         includes all four Galilean moons) is the self-consistent
         effective-point-mass formulation.
    """
    return ThirdBody(naif_id=5, mu_km3_s2=body_gm_km3_per_s2(5), label="JUPITER_BARYCENTER")


# ---------------------------------------------------------------------------
# Core acceleration (no SPICE)
# ---------------------------------------------------------------------------


def third_body_acceleration(
    r_sat_km: np.ndarray,
    r_third_km: np.ndarray,
    mu_third_km3_s2: float,
) -> np.ndarray:
    """Perturbing acceleration of a Mars-centered particle by one third body.

    Implements Montenbruck & Gill (2000) Eq. (3.37):

        a_3b = mu_3 * [ (r_3 - r) / |r_3 - r|^3  -  r_3 / |r_3|^3 ]

    The two terms are the direct attraction of the third body on the
    particle and the indirect acceleration of the Mars center itself
    (subtracted because the result is in a Mars-centered non-rotating
    frame). At r = 0 the terms cancel identically.

    Parameters
    ----------
    r_sat_km, r_third_km
        Particle and third-body positions relative to the central body
        (Mars), in the same inertial frame, km. Both shape (3,).
    mu_third_km3_s2
        Gravitational parameter of the third body, km^3/s^2.

    Returns
    -------
    ndarray, shape (3,)
        Acceleration in the same frame, km/s^2.

    Notes
    -----
    This is the pure, SPICE-free hot-loop routine. The convenience
    wrapper ``third_body_acceleration_from_spice`` fetches the third
    body's position via ``spice.spkezr`` and sums contributions over
    multiple bodies.
    """
    r_sat = np.asarray(r_sat_km, dtype=float)
    r_third = np.asarray(r_third_km, dtype=float)
    d = r_third - r_sat
    d_norm = float(np.linalg.norm(d))
    r3_norm = float(np.linalg.norm(r_third))
    if r3_norm == 0.0:
        raise ValueError("third-body position at Mars center is undefined")
    if d_norm == 0.0:
        raise ValueError("sail coincident with third body: acceleration singular")
    return mu_third_km3_s2 * (d / d_norm ** 3 - r_third / r3_norm ** 3)


# ---------------------------------------------------------------------------
# SPICE-fed wrapper
# ---------------------------------------------------------------------------


def third_body_acceleration_from_spice(
    r_sat_j2000_km: np.ndarray,
    et: float,
    bodies: Sequence[ThirdBody],
    observer_naif_id: int = 499,
) -> np.ndarray:
    """Summed third-body acceleration at a specified epoch.

    For each body, fetches its J2000 position relative to the observer
    (default: NAIF 499, Mars planet centre -- NOT 4, Mars barycenter;
    the distinction is sub-metre) and adds the contribution from
    ``third_body_acceleration``.

    Parameters
    ----------
    r_sat_j2000_km
        Sail position in Mars-centered J2000, km, shape (3,).
    et
        SPICE ephemeris time (TDB seconds past J2000).
    bodies
        Iterable of ``ThirdBody`` specs. Order is irrelevant.
    observer_naif_id
        NAIF ID of the central body the positions are referenced to.
        Default 499 (Mars planet centre).

    Returns
    -------
    ndarray, shape (3,)
        Total third-body acceleration in J2000, km/s^2.
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)
    a_total = np.zeros(3, dtype=float)
    for body in bodies:
        if body.naif_id == SUN_NAIF_ID:
            # Route the Sun through the injectable provider so ablation sun-models
            # (no-obliquity / circular) apply to the third-body term too. Other
            # bodies (Phobos, Deimos, ...) keep the direct ephemeris.
            r_third = np.asarray(sun_state_j2000(et, observer_naif_id)[:3], dtype=float)
        else:
            state, _ = spice_state_at_et(
                body.naif_id, et, "J2000", "NONE", observer_naif_id
            )
            r_third = np.asarray(state[:3], dtype=float)
        a_total += third_body_acceleration(r_sat, r_third, body.mu_km3_s2)
    return a_total


# ---------------------------------------------------------------------------
# Perturbation potential (for finite-difference validation)
# ---------------------------------------------------------------------------


def third_body_perturbation_potential(
    r_sat_km: np.ndarray,
    r_third_km: np.ndarray,
    mu_third_km3_s2: float,
) -> float:
    """Perturbing potential U_3b(r) such that a_3b = -grad U_3b.

    Mars-centered non-rotating frame. Starting from Montenbruck & Gill
    (2000) Eq. (3.38) and normalizing so U_3b(0) = 0:

        U_3b(r) = -mu_3 * [ 1/|r - r_3|  -  1/|r_3|  -  r . r_3 / |r_3|^3 ]  (*)

    The three terms are: direct potential of the third body; constant
    term that sets U_3b(0) = 0; linear term that cancels the indirect-
    acceleration contribution so the first non-trivial behaviour is
    the tidal quadrupole.

    Gradient check. Expanding (*):
        U_3b(r)    = -mu_3 / |r - r_3| + mu_3 / |r_3| + mu_3 (r . r_3) / |r_3|^3
        grad U_3b  =  mu_3 (r - r_3) / |r - r_3|^3     + mu_3 r_3 / |r_3|^3
        -grad U_3b =  mu_3 (r_3 - r) / |r - r_3|^3     - mu_3 r_3 / |r_3|^3
                   =  Eq. (3.37)    ✓

    Numerical stability
    -------------------

    Expression (*) above is mathematically correct but NUMERICALLY
    UNSTABLE when |r| << |r_3|. For Mars/Sun at LMO the first two
    terms are each ~5e-9 km^-1 while the physical tidal residual
    is ~2e-16 km^-1 -- nine digits of cancellation in double precision
    wipe out the tidal signal.

    The stable form used below factors the cancellation analytically.
    Define q = (|r|^2 - 2 r.r_3) / |r_3|^2. Then |r - r_3|^2 = |r_3|^2
    (1+q) and 1/|r - r_3| = (1/|r_3|) (1+q)^{-1/2}. Let
    A = sqrt(1+q). The identity

        (1+q)^{-1/2} - 1 + q/2  =  q^2 (A+2) / [2 (A+1) (1 + A + q)]

    (derived by writing (1+q)^{-1/2} - 1 = -q / (A(1+A)), rationalising,
    and adding q/2) provides a closed form for the residual AFTER the
    O(q) cancellation has been done symbolically. Substituting into
    (*) and using q/(2|r_3|) = |r|^2/(2|r_3|^3) - r.r_3/|r_3|^3, the
    r.r_3/|r_3|^3 terms cancel analytically, leaving

        U_3b = mu_3 |r|^2 / (2 |r_3|^3)   -  mu_3 f3(q) / |r_3|         (**)

        where  f3(q) = q^2 (A+2) / [2 (A+1) (1 + A + q)],
               A     = sqrt(1+q),
               q     = (|r|^2 - 2 r.r_3) / |r_3|^2.

    Both terms in (**) are O((r/r_3)^2); no catastrophic cancellation.
    Matches (*) to within machine precision at any r/r_3 < 1.

    Validity: f3 remains finite and analytic for q > -1 (equivalently
    |r - r_3| > 0, i.e. sail not at the third body). At q = -1 the
    physical 1/|r - r_3| singularity reappears through the (1+A+q)
    factor in the denominator.

    This function is NOT used in the propagator hot loop -- it exists
    only to provide a second, independent expression of the physics
    that the acceleration code can be validated against (finite-
    difference of the potential vs analytic acceleration).
    """
    r_sat = np.asarray(r_sat_km, dtype=float)
    r_third = np.asarray(r_third_km, dtype=float)
    r_sat_sq = float(np.dot(r_sat, r_sat))
    r_dot_r3 = float(np.dot(r_sat, r_third))
    r3_sq = float(np.dot(r_third, r_third))
    if r3_sq == 0.0:
        raise ValueError("third-body position at Mars center is undefined")
    r3_norm = np.sqrt(r3_sq)
    q = (r_sat_sq - 2.0 * r_dot_r3) / r3_sq
    if q <= -1.0:
        # |r - r_3|^2 <= 0: sail at or past the third-body position.
        raise ValueError(
            f"sail at/past third-body position (q={q:.3e}); potential singular"
        )
    A = np.sqrt(1.0 + q)
    f3 = (q * q) * (A + 2.0) / (2.0 * (A + 1.0) * (1.0 + A + q))
    return mu_third_km3_s2 * r_sat_sq / (2.0 * r3_norm ** 3) - mu_third_km3_s2 * f3 / r3_norm
