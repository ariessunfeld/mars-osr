"""Solar radiation pressure: flat-sail (McInnes) and spherical-grain (H&K96).

Two parallel SRP paths live in this module:

  Flat sail (McInnes 1999 §2.6.1).  Translational acceleration of a
  Mars-centered solar sail under the six-parameter optical force model,
  shadow-gated by the binary Mars umbra test in ``reflectors.shadow``.
  Sail is treated as rigid and flat (no billow, no wrinkling); attitude
  supplied as a caller-provided callable ``n_hat(r_sat_km, et) ->
  unit_vec``. Functions: ``srp_acceleration``, ``make_srp_contributor``;
  data: ``SailOptical``, ``SolarSail``.

  Spherical grain (Hamilton & Krivov 1996, *Icarus* 123:503-523, §2.1
  + Eq. (3); foundational treatment in Burns, Lamy & Soter 1979,
  *Icarus* 40:1-48). SRP on a uniform spherical dust / rock particle:
  force is ``Q_pr * P(r_helio) * pi r_g^2 / m_g`` along the anti-Sun
  line, orientation-independent (no attitude callable). Same binary
  umbra gate as the flat-sail path. Functions:
  ``spherical_particle_acceleration``, ``make_spherical_srp_contributor``;
  data: ``SphericalParticle``. Used for circumplanetary dust and small
  rock fragments where the flat-sail abstraction (with its attitude
  dependence and one-sidedness) is physically invalid even if the
  ``alpha = 0`` limit of the McInnes formula happens to give the same
  number for an absorbing sphere. The two paths are mutually exclusive
  at the propagator level (callers supply exactly one of ``solar_sail`` or
  ``spherical_particle``).

Reference (primary): McInnes, C.R. (1999), *Solar Sailing: Technology,
Dynamics and Mission Applications*, Springer-Praxis, Chapter 2 §2.6.1
("Optical force model"), Eqs. (2.46)-(2.57). Equation numbers in
``(M-x.y)`` annotations below refer to that edition.

Sign-convention note. McInnes defines his sail normal ``n`` to point
AWAY from the Sun at zero pitch angle (his "dark-side" normal; see
Fig. 2.7). The equations in §2.6.1 are therefore natural in that
convention. The API exposed by this module uses the more intuitive
SUN-FACING convention: ``n_hat`` points toward the Sun when the sail
is face-on to the light (``cos alpha = n_hat . s_hat > 0`` in illumi-
nated states). The translation between the two is a single internal
sign flip; the resulting closed-form acceleration is:

    a_SRP = -(P A / m) * max(0, n_hat . s_hat)
            * [ (1 - rho * s) * s_hat  +  C_n(alpha) * n_hat ]
            * shadow_factor(r_sat, et)                                 (*)

    C_n(alpha) = 2 * rho * s * cos(alpha)
                 + B_f * (1 - s) * rho
                 + (1 - rho) * (eps_f * B_f - eps_b * B_b)
                                      / (eps_f + eps_b)

where ``s_hat`` is the sail-to-Sun unit vector, ``P = L_Sun /
(4 pi c r_helio^2)`` the solar radiation pressure at the sail,
``A`` the sail area, ``m`` the sail+payload mass, and
``{rho, s, eps_f, eps_b, B_f, B_b}`` are the optical coefficients of
the sail film.

Derivation that (*) is equivalent to McInnes's normal + transverse
decomposition of Eqs. (M-2.57a)-(M-2.57b):

    In McInnes's convention ``n_McInnes = -n_hat`` and
    ``t_McInnes = (cos alpha * n_hat - s_hat) / sin alpha``
    (from (M-2.49a) with ``u = -s_hat``). Substituting into
    ``f = f_n n_McInnes + f_t t_McInnes`` and collecting the
    ``n_hat`` and ``s_hat`` coefficients yields (*) term-for-term.
    Concretely the ``n_hat`` coefficient combines ``-f_n / (PA)`` with
    ``+f_t cos alpha / (sin alpha * PA) * sin alpha = f_t cos alpha /
    (PA)`` and the ``s_hat`` coefficient is ``-f_t / (PA)``.

Sanity anchors baked into the test suite:
  - Ideal perfect mirror (rho=1, s=1, eps=0) at alpha=0:
        a_SRP = -2 (P A / m) s_hat, i.e. 2 P A / m directed AWAY
        from the Sun. This is the textbook "photon bounce delivers 2p"
        momentum exchange.
  - Pure absorber (rho=0) at alpha=0: a_SRP = -(P A / m) s_hat.
  - Edge-on (cos alpha = 0) or back-lit (cos alpha < 0): a_SRP = 0.
  - Inside Mars umbra: a_SRP = 0 (shadow gate).

Thermal term. McInnes Eq. (M-2.56) gives the force from thermal
re-radiation of absorbed energy:

    f_e = P A (1 - rho) * (eps_f B_f - eps_b B_b) / (eps_f + eps_b)
                                                   * cos alpha * n

This vanishes when ``eps_f + eps_b = 0`` (no absorbing sail can
re-radiate if it has zero emissivity on both faces). When the
aluminium front / chromium back design (``eps_f << eps_b``) is used
-- the default here -- the term is NEGATIVE, producing a small
sunward drag that slightly reduces net thrust; when the sail is
symmetric (``eps_f == eps_b``) the term is zero by construction.
``reflectors.srp`` handles the ``eps_f + eps_b = 0`` edge case
explicitly rather than relying on 0/0 propagation.

Default sail material. ``SailOptical.square_sail_jpl()`` returns the
McInnes Table 2.1 "Square sail" coefficients derived from the JPL
comet-Halley-rendezvous solar-sail design (aluminised Kapton front,
chromium back):

    rho = 0.88,  s = 0.94,  eps_f = 0.05,  eps_b = 0.55,
    B_f = 0.79,  B_b = 0.55

cited in McInnes 1999 to Wright, J.L. (1992), *Space Sailing*, Gordon &
Breach, Appendices A-B. Swap in a different material by instantiating
``SailOptical`` directly or adding a new ``classmethod`` factory.

Model scope. Sail billow, wrinkles, localised reflectivity loss from UV
degradation, sail-temperature dependence of ``rho``, and the parametric
JPL-Halley sail-shape force model (McInnes §2.6.2) are not modelled. Finite
solar-disc effects (McInnes §2.5) are also excluded; they deviate from the
inverse-square law by <5% only inside ~10 solar radii, orders of magnitude
closer than the Mars trajectories considered here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import spiceypy as spice

from reflectors.attitude import AttitudeCallable
from reflectors.ephemeris import sun_state_j2000
from reflectors.shadow import shadow_factor
from reflectors.solar_constants import solar_flux_at


logger = logging.getLogger(__name__)


SUN_NAIF_ID = 10
MARS_NAIF_ID = 499

# ``AttitudeCallable`` is re-exported here; attitude primitive factories
# (``fixed_j2000``, ``sun_pointing``, ``smooth_slew``, ``piecewise``) live in
# ``reflectors.attitude``.
__all__ = [
    "AttitudeCallable",
    "SailOptical",
    "SolarSail",
    "mcinnes_srp_acceleration",
    "srp_acceleration",
    "make_srp_contributor",
    "SphericalParticle",
    "spherical_particle_acceleration",
    "make_spherical_srp_contributor",
]


# ---------------------------------------------------------------------------
# Optical coefficients
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SailOptical:
    """Six-parameter optical coefficients for the McInnes §2.6.1 force model.

    Attributes
    ----------
    rho
        Total reflectance ``rho_tilde`` in (M-2.47). Fraction of incident
        photons reflected (specular + diffuse combined). The remainder
        ``1 - rho`` is absorbed and then thermally re-radiated. In [0, 1].
    s
        Specular fraction of the reflected component (``s`` in (M-2.51)).
        ``s = 1`` is a perfect mirror; ``s = 0`` reflects diffusely. In
        [0, 1].
    eps_front, eps_back
        Thermal emissivities of the sun-facing and anti-sun-facing surfaces
        (``eps_f``, ``eps_b`` in (M-2.56)). In [0, 1]. The asymmetry
        ``eps_f - eps_b`` (weighted by the Lambertian factors below)
        drives the small residual thermal thrust.
    B_front, B_back
        Non-Lambertian coefficients of the front and back surfaces (``B_f``,
        ``B_b`` in (M-2.52), (M-2.56)). Exactly ``2/3`` for a perfectly
        Lambertian emitter; ``0`` for a perfectly specular emitter. In
        [0, 1] in the modeled regime.
    two_sided
        When ``False`` (the default) the sail is one-sided: a back-lit sail
        (``n_hat . s_hat < 0``) produces exactly zero force, which is
        McInnes's own assumption (see below) and the default behaviour of this
        module. When ``True`` the back face is treated as illuminated in
        its own right -- required for UNCOMMANDED attitude (tumbling), where
        the sail is back-lit roughly half the time and the one-sided model
        would omit half the SRP impulse. See
        :func:`mcinnes_srp_acceleration` for the extension rule and its
        limits. Defaults to ``False``, preserving the one-sided McInnes model.

    All coefficients are frozen so optical dataclasses can be shared across
    propagations and used as dict keys in sensitivity studies.
    """

    rho: float
    s: float
    eps_front: float
    eps_back: float
    B_front: float
    B_back: float
    two_sided: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("rho", self.rho),
            ("s", self.s),
            ("eps_front", self.eps_front),
            ("eps_back", self.eps_back),
            ("B_front", self.B_front),
            ("B_back", self.B_back),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"SailOptical.{name} must be in [0, 1], got {value}"
                )

    def with_faces_swapped(self) -> "SailOptical":
        """Same film viewed from the other side: swap the front/back pairs.

        Exchanges ``(eps_front, B_front) <-> (eps_back, B_back)`` and leaves
        ``rho``, ``s`` alone. Used by the two-sided force path when the BACK
        face is the illuminated one: McInnes's (M-2.57a) is written for the
        lit face, so re-labelling which face is lit is exactly this swap.

        Consequences, both exact:
          - the (M-2.52) diffuse-reflection term's ``B_f`` becomes ``B_b``;
          - the (M-2.56) thermal asymmetry ``eps_f B_f - eps_b B_b`` becomes
            its own negative, flipping the small thermal thrust.

        ``rho`` and ``s`` are unchanged because the six-parameter model
        carries no back-side reflectance -- see the caveat in
        :func:`mcinnes_srp_acceleration`.
        """
        return SailOptical(
            rho=self.rho, s=self.s,
            eps_front=self.eps_back, eps_back=self.eps_front,
            B_front=self.B_back, B_back=self.B_front,
            two_sided=self.two_sided,
        )

    @classmethod
    def ideal(cls) -> "SailOptical":
        """Perfect specular reflector, zero absorption, Lambertian thermal.

        Returns the (trivial) "Ideal sail" row of McInnes Table 2.1:
        ``rho=1, s=1, eps_f=eps_b=0, B_f=B_b=2/3``. All reflected,
        fully specular, no thermal contribution. Used as a sanity limit
        in tests (``a = 2 P A / m`` at alpha=0).
        """
        return cls(
            rho=1.0, s=1.0,
            eps_front=0.0, eps_back=0.0,
            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
        )

    @classmethod
    def square_sail_jpl(cls) -> "SailOptical":
        """McInnes Table 2.1 'Square sail' coefficients.

        JPL comet-Halley rendezvous study design: aluminised Kapton
        front (rho = 0.88, eps_f = 0.05) and chromium back
        (eps_b = 0.55). Cited in McInnes 1999 §2.6.1 to Wright, J.L.
        (1992), *Space Sailing*, Gordon & Breach, Appendices A-B.
        """
        return cls(
            rho=0.88, s=0.94,
            eps_front=0.05, eps_back=0.55,
            B_front=0.79, B_back=0.55,
        )

    @classmethod
    def heliogyro_jpl(cls) -> "SailOptical":
        """McInnes Table 2.1 'Heliogyro' coefficients.

        Same film as the JPL square sail per Table 2.1; the distinct factory
        permits sensitivity studies to assign independent optical treatments.
        """
        return cls.square_sail_jpl()


# ---------------------------------------------------------------------------
# Sail bus (area, mass, optical)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolarSail:
    """Physical bus of a flat, rigid solar sail.

    Attributes
    ----------
    area_m2
        Effective reflective area in m^2 (strictly positive).
    mass_kg
        Total sail + payload mass in kg (strictly positive).
    optical
        ``SailOptical`` instance carrying the six film-optical coefficients.

    The attitude profile (which way the sail normal points at time ``t``)
    is NOT owned by this dataclass; it is supplied independently as a
    callable to ``srp_acceleration`` and ``propagate``. That decoupling
    lets the same physical sail fly different attitude programs (fixed,
    sun-pointing, reflection-bisector) without multiplying dataclass
    variants.
    """

    area_m2: float
    mass_kg: float
    optical: SailOptical

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0:
            raise ValueError(f"SolarSail.area_m2 must be > 0, got {self.area_m2}")
        if self.mass_kg <= 0.0:
            raise ValueError(f"SolarSail.mass_kg must be > 0, got {self.mass_kg}")

    @property
    def loading_kg_per_m2(self) -> float:
        """Sail loading ``sigma = m / A`` in kg/m^2 (McInnes Eq. M-3.1)."""
        return self.mass_kg / self.area_m2


# ---------------------------------------------------------------------------
# Core acceleration
# ---------------------------------------------------------------------------


def mcinnes_srp_acceleration(
    n_hat: np.ndarray,
    s_hat: np.ndarray,
    P_pa: float,
    sail: SolarSail,
) -> np.ndarray:
    """McInnes flat-sail SRP acceleration core -- pure geometry, no SPICE.

    The closed-form heart of the six-parameter optical force model (the
    formula (*) in the module docstring). Given a pre-fetched sail-to-Sun
    unit vector ``s_hat``, the solar radiation pressure ``P_pa`` at the sail
    (Pa), a unit sail normal ``n_hat``, and the sail bus, return the SRP
    acceleration in the same axes as ``s_hat`` / ``n_hat`` (km/s^2).

    ``n_hat`` may be a single unit vector, shape ``(3,)``, or a batch of
    them, shape ``(K, 3)``; the return matches (``(3,)`` or ``(K, 3)``).
    The batch form lets the Q-law escape steering evaluate the force at many
    trial sail normals in one vectorised call. Both ``n_hat`` and ``s_hat``
    must already be unit vectors -- this core does no re-normalisation, no
    Sun ephemeris lookup, and no shadow gate. Callers needing those use
    :func:`srp_acceleration` (which delegates its closed-form arithmetic
    here).

    One-sided (``sail.optical.two_sided is False``, the default) returns the
    zero vector for an edge-on or back-lit sail (``n_hat . s_hat <= 0``).

    A two-sided sail (``sail.optical.two_sided is True``) instead treats the back
    face as illuminated in its own right. This is needed for uncommanded attitude
    (tumbling), where the sail is back-lit ~half the time and the one-sided
    model would drop half the SRP impulse -- exactly the regime the
    orientation-average is supposed to describe.

    Extension rule. Let ``n_ill = sign(n_hat . s_hat) * n_hat`` be the
    outward normal of the illuminated face, so ``n_ill . s_hat = |cos alpha|
    > 0``. Evaluate the same (M-2.57a)/(M-2.57b) force with ``n_ill`` in place
    of ``n_hat``, using ``optical.with_faces_swapped()`` when the back face is
    the lit one. Edge-on (``cos alpha == 0``) is still exactly zero.

    This is an extension beyond the cited source rather than a transcription
    of it. McInnes §2.6.1 is written for a front-lit sail only:
    (M-2.48) assumes ``tau = 0`` "on the reflecting side", and the Fig. 2.8
    (p.49) thermal balance has the incident power ``(1 - rho) W cos alpha``
    arriving on the front surface alone. The model therefore carries one
    ``rho`` and one ``s`` (properties of the reflecting side) plus separate
    front/back thermal pairs. Consequently the back face here reuses the
    front's ``rho`` and ``s``: an optically-symmetric-reflectance idealisation.
    A real sail (aluminised Kapton front, chromium back -- see
    :meth:`SailOptical.square_sail_jpl`) has a markedly less reflective back,
    which would need ``rho_back`` / ``s_back`` parameters that the
    six-parameter model does not define.
    """
    n = np.asarray(n_hat, dtype=float)
    s = np.asarray(s_hat, dtype=float)
    cos_alpha = np.sum(n * s, axis=-1)  # scalar or (K,)

    if not sail.optical.two_sided:
        return _mcinnes_lit_face_acceleration(n, s, cos_alpha, P_pa, sail,
                                              sail.optical)

    # Two-sided: split on which face is lit and evaluate each with that
    # face's optical roles. Both branches use the illuminated-face normal
    # n_ill, whose dot with s_hat is |cos alpha| >= 0, so the shared core
    # sees a front-lit problem in both cases.
    n_ill = np.where(cos_alpha[..., np.newaxis] >= 0.0, n, -n)
    abs_cos = np.abs(cos_alpha)
    a_front = _mcinnes_lit_face_acceleration(
        n_ill, s, abs_cos, P_pa, sail, sail.optical)
    a_back = _mcinnes_lit_face_acceleration(
        n_ill, s, abs_cos, P_pa, sail, sail.optical.with_faces_swapped())
    return np.where(cos_alpha[..., np.newaxis] >= 0.0, a_front, a_back)


def _mcinnes_lit_face_acceleration(
    n: np.ndarray,
    s: np.ndarray,
    cos_alpha: np.ndarray,
    P_pa: float,
    sail: "SolarSail",
    opt: "SailOptical",
) -> np.ndarray:
    """(M-2.57) force for a FRONT-LIT face, given its optical coefficients.

    Factored out of :func:`mcinnes_srp_acceleration` so the one-sided path and
    each branch of the two-sided path share one arithmetic implementation
    (there is only one place the formula lives). ``opt`` is passed explicitly
    rather than read off ``sail`` so the two-sided path can hand in
    ``sail.optical.with_faces_swapped()`` for a back-lit face.

    Still returns zero wherever ``cos_alpha <= 0`` -- for the one-sided caller
    that is the back-lit gate; for the two-sided caller ``cos_alpha`` is
    already ``|n . s|`` so the gate only bites exactly edge-on.
    """
    # Thermal re-emission coefficient (McInnes (M-2.56)). Handle the
    # eps_f + eps_b = 0 edge case explicitly.
    eps_sum = opt.eps_front + opt.eps_back
    if eps_sum == 0.0:
        thermal_term = 0.0
    else:
        thermal_term = (
            opt.eps_front * opt.B_front - opt.eps_back * opt.B_back
        ) / eps_sum

    # Normal-direction combined coefficient C_n(alpha); dimensionless.
    C_n = (
        2.0 * opt.rho * opt.s * cos_alpha
        + opt.B_front * (1.0 - opt.s) * opt.rho
        + (1.0 - opt.rho) * thermal_term
    )
    # Tangential (sunward) coefficient; dimensionless.
    C_s = 1.0 - opt.rho * opt.s

    # ``P_pa * area / mass`` is in m/s^2; convert to km/s^2.
    pa_over_m_km = P_pa * sail.area_m2 / sail.mass_kg * 1.0e-3
    a = (
        -pa_over_m_km
        * cos_alpha[..., np.newaxis]
        * (C_s * s + C_n[..., np.newaxis] * n)
    )
    # Edge-on or back-lit (cos_alpha <= 0): zero force.
    return np.where(cos_alpha[..., np.newaxis] > 0.0, a, 0.0)


def srp_acceleration(
    r_sat_j2000_km: np.ndarray,
    et: float,
    sail: SolarSail,
    n_hat_func: AttitudeCallable,
    *,
    observer_naif_id: int = MARS_NAIF_ID,
) -> np.ndarray:
    """Solar-radiation-pressure acceleration on the sail at epoch ``et``.

    Implements McInnes (1999) Eq. (M-2.57a/b) rewritten in the sunward-
    normal convention (see module docstring, derivation). The sail is
    treated as flat, rigid, and one-sided: force is zeroed when
    ``cos alpha = n_hat . s_hat <= 0`` (edge-on or back-lit) and when the
    sail is inside the Mars umbra (binary shadow gate).

    Parameters
    ----------
    r_sat_j2000_km
        Sail position in observer-centered J2000 axes, km, shape (3,).
    et
        SPICE ephemeris time, TDB seconds past J2000.
    sail
        ``SolarSail`` instance carrying (area_m2, mass_kg, optical).
    n_hat_func
        Attitude callable ``(r_sat_km, et) -> unit_vector_j2000``. Need
        NOT return a unit vector; re-normalised inside this routine.
    observer_naif_id
        Central body for ``spkezr``. Default 499 (Mars planet centre).

    Returns
    -------
    ndarray, shape (3,)
        SRP acceleration in J2000 axes, km/s^2.
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)

    # Sun position in observer-centered J2000; shared with the shadow test.
    state = sun_state_j2000(et, observer_naif_id)
    r_sun = np.asarray(state[:3], dtype=float)

    # Binary umbra gate (shares the pre-fetched Sun position).
    if shadow_factor(r_sat, et, observer_naif_id, sun_position_j2000_km=r_sun) == 0.0:
        return np.zeros(3, dtype=float)

    sat_to_sun = r_sun - r_sat
    r_helio_km = float(np.linalg.norm(sat_to_sun))
    s_hat = sat_to_sun / r_helio_km

    # Attitude, re-normalised as a round-off safeguard.
    n_raw = np.asarray(n_hat_func(r_sat, et), dtype=float)
    n_norm = float(np.linalg.norm(n_raw))
    if n_norm == 0.0:
        raise ValueError("n_hat_func returned the zero vector")
    n_hat = n_raw / n_norm

    # Solar radiation pressure at the sail (Pa = N/m^2).
    P_pa = solar_flux_at(r_helio_km)

    # Closed-form flat-sail optical force. The arithmetic lives in
    # ``mcinnes_srp_acceleration`` so the Q-law escape steering can reuse
    # the exact same force model without a redundant Sun lookup.
    return mcinnes_srp_acceleration(n_hat, s_hat, P_pa, sail)


# ---------------------------------------------------------------------------
# Propagator glue (lazy-imported from dynamics.py)
# ---------------------------------------------------------------------------


def make_srp_contributor(
    sail: SolarSail,
    n_hat_func: AttitudeCallable,
    epoch_et: float,
    *,
    observer_naif_id: int = MARS_NAIF_ID,
    ephemeris_time_direction: int = +1,
):
    """Build a ``(r, t_offset) -> a_SRP`` closure for ``dynamics._make_rhs``.

    Matches the contributor signature used by zonal / harmonic / third-body
    perturbations in ``reflectors.dynamics``. ``propagate`` owns the
    construction of the closure so callers of ``propagate`` need not
    touch this helper directly; it is exposed so downstream modules can
    compose bespoke RHS assemblies if needed.

    ``ephemeris_time_direction`` (+1 forward, -1 backward) sets the sign of the
    ephemeris clock: the Sun position AND the attitude callable are evaluated at
    ``epoch_et + ephemeris_time_direction * t_offset``. The default +1 clocks
    forward; -1 clocks the Sun backward for reverse-time
    (capture-node / time-reversal) runs. Mirrors escape.py's identically-named
    knob.
    """

    def contributor(r: np.ndarray, t_offset: float) -> np.ndarray:
        return srp_acceleration(
            r,
            epoch_et + ephemeris_time_direction * t_offset,
            sail,
            n_hat_func,
            observer_naif_id=observer_naif_id,
        )

    return contributor


# ---------------------------------------------------------------------------
# Spherical-grain path (Hamilton & Krivov 1996 / Burns et al. 1979)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SphericalParticle:
    """Uniform spherical dust / rock grain for SRP propagation.

    Used by ``spherical_particle_acceleration`` and the propagator
    kwarg ``propagate(spherical_particle=...)``. Conceptually distinct
    from ``SolarSail`` -- a sphere has no preferred orientation, no
    attitude callable, and presents the same projected disc
    ``pi r_g^2`` to the Sun at every step. Force magnitude per
    Hamilton & Krivov (1996) Eq. (3):

        |a_SRP| = Q_pr * P(r_helio) * pi r_g^2 / m_g
                = (3/4) * Q_pr * F_solar / (c * rho_g * r_g)

    along the anti-Sun line.

    Attributes
    ----------
    radius_m
        Grain radius ``r_g`` in metres. Strictly positive.
    density_kg_per_m3
        Material density ``rho_g`` in kg/m^3. Strictly positive.
        Typical values: ordinary chondrites 3.4-3.7e3 kg/m^3
        (Britt & Consolmagno 2003 *Meteoritics & Planetary Science*
        38(8):1161 Table 1); basalt analogue 2.9-3.0e3 kg/m^3;
        carbonaceous chondrites 2.2-2.7e3 kg/m^3. H&K96 use
        ``rho_g = 2.0 g/cm^3`` for Phobos/Deimos ejecta (p. 508).
    Q_pr
        Dimensionless radiation-pressure efficiency factor. Default
        ``1.0`` matches H&K96's value for Mars ejecta dust (page 508
        explicitly: ``Q_pr = 1.0`` for ``rho_g = 2.0 g/cm^3`` Phobos
        and Deimos ejecta examples). For ``r_g >> lambda_solar``
        (geometric-optics regime, size parameter
        ``x = 2 pi r_g / lambda > ~ 50``), ``Q_pr -> 1`` for absorbing
        grains with weak corrections from diffuse vs back-scatter
        asymmetry; see Bohren & Huffman 1983, *Absorption and
        Scattering of Light by Small Particles*, Ch. 7. Strictly
        positive in the absorbing-and-scattering regime; bounded
        above by 2.0 (perfect-mirror limit).

    The dataclass is frozen so a single particle instance can be
    shared across propagations and used as a dict key for sweeps.
    """

    radius_m: float
    density_kg_per_m3: float
    Q_pr: float = 1.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError(
                f"SphericalParticle.radius_m must be > 0, got {self.radius_m}"
            )
        if self.density_kg_per_m3 <= 0.0:
            raise ValueError(
                f"SphericalParticle.density_kg_per_m3 must be > 0, got "
                f"{self.density_kg_per_m3}"
            )
        if self.Q_pr <= 0.0:
            raise ValueError(
                f"SphericalParticle.Q_pr must be > 0, got {self.Q_pr}"
            )

    @property
    def cross_section_m2(self) -> float:
        """Projected disc area ``pi r_g^2`` (m^2). Photon-collection area."""
        return math.pi * self.radius_m * self.radius_m

    @property
    def mass_kg(self) -> float:
        """Sphere mass ``(4/3) pi r_g^3 rho_g`` in kg."""
        return (4.0 / 3.0) * math.pi * self.radius_m ** 3 * self.density_kg_per_m3

    @property
    def area_to_mass_m2_per_kg(self) -> float:
        """Area-to-mass ``A/m = 3 / (4 r_g rho_g)`` in m^2/kg.

        Closed form via ``A/m = (pi r^2) / ((4/3) pi r^3 rho) =
        3 / (4 r rho)``. Computed analytically (no division between the
        two derived properties) so a 1e-12-relative property test stays
        stable under roundoff.
        """
        return 3.0 / (4.0 * self.radius_m * self.density_kg_per_m3)


def spherical_particle_acceleration(
    r_sat_j2000_km: np.ndarray,
    et: float,
    particle: SphericalParticle,
    *,
    observer_naif_id: int = MARS_NAIF_ID,
    apply_shadow: bool = True,
) -> np.ndarray:
    """SRP acceleration on a uniform spherical grain at the given epoch.

    Implements Hamilton & Krivov (1996), *Icarus* 123:503-523, §2.1
    + Eq. (3): force magnitude

        |a_SRP| = Q_pr * P(r_helio) * pi r_g^2 / m_g

    along the anti-Sun line (from Sun towards grain). Orientation-
    independent: a uniform sphere presents the same projected disc
    in every direction, so the McInnes flat-sail attitude apparatus
    does not apply. The parallel-ray approximation (H&K96 §2.1: "the
    strength and direction of radiation pressure are assumed not to
    vary over the dust grain's orbit") is implicit -- ``P(r_helio)`` is
    evaluated at the satellite-Sun distance each step rather than
    averaging.

    Shadow-gated by ``reflectors.shadow.shadow_factor`` (binary Mars
    umbra). Per H&K96 §2.1, planetary albedo and partial-shadow
    contributions are "at least an order of magnitude weaker than
    those produced by direct solar illumination" and are excluded;
    when the grain is in full umbra, the function returns zero.

    Parameters
    ----------
    r_sat_j2000_km
        Grain position in observer-centred J2000 axes, km, shape (3,).
    et
        SPICE ephemeris time, TDB seconds past J2000.
    particle
        ``SphericalParticle`` instance carrying ``radius_m``,
        ``density_kg_per_m3``, and ``Q_pr``.
    observer_naif_id
        Central body for ``spkezr``. Default 499 (Mars planet centre).
    apply_shadow
        When ``True`` (default), the binary Mars-umbra gate zeroes the
        force inside total eclipse.
        Set ``False`` to reproduce the H&K96 convention (p. 504), which
        neglects planetary shadow and assumes continuous direct solar
        illumination.

    Returns
    -------
    ndarray, shape (3,)
        SRP acceleration in J2000 axes, km/s^2, along the anti-Sun
        unit vector ``(r_sat - r_sun) / |r_sat - r_sun|``.
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)

    # Sun position in observer-centred J2000; shared with the shadow test.
    state = sun_state_j2000(et, observer_naif_id)
    r_sun = np.asarray(state[:3], dtype=float)

    # Binary umbra gate (shares the pre-fetched Sun position). Skipped
    # entirely when apply_shadow is False (H&K96 shadow-neglect convention).
    if apply_shadow and shadow_factor(
        r_sat, et, observer_naif_id, sun_position_j2000_km=r_sun
    ) == 0.0:
        return np.zeros(3, dtype=float)

    sat_to_sun = r_sun - r_sat
    r_helio_km = float(np.linalg.norm(sat_to_sun))
    s_hat = sat_to_sun / r_helio_km  # sat-to-Sun unit vector

    # Solar radiation pressure at the grain (Pa = N/m^2).
    P_pa = solar_flux_at(r_helio_km)

    # H&K96 Eq. (3) magnitude. ``P A / m`` is in m/s^2; multiply by
    # Q_pr and convert to km/s^2.
    a_mag_mps2 = (
        particle.Q_pr * P_pa * particle.cross_section_m2 / particle.mass_kg
    )
    a_mag_kmps2 = a_mag_mps2 * 1.0e-3

    # Force is along the anti-Sun line, i.e. along -s_hat (from Sun to
    # grain). Multiplying ``-s_hat`` by the positive magnitude
    # ``a_mag_kmps2`` gives the acceleration vector in J2000 km/s^2.
    return -a_mag_kmps2 * s_hat


@dataclass(frozen=True)
class TumbleAveragedSail:
    """A flat sail tumbling fast enough that its SRP force time-averages.

    Wraps the same physical ``SolarSail`` (area, mass, film) and replaces the
    attitude-dependent (M-2.57) force with its exact average over uniformly
    distributed orientations, which is purely anti-sunward by symmetry. Because
    the result is orientation-independent, no attitude callable is needed --
    conceptually the same situation as ``SphericalParticle``, and wired into
    ``propagate`` the same way (``propagate(tumble_averaged_sail=...)``).

    The averaged model avoids two problems with explicit rotation:

    1. A genuinely *random* ``n_hat(t)`` makes the ODE right-hand side
       discontinuous, which destroys the adaptive step-size control in DOP853:
       the answer stops converging and stops being reproducible. A physically
       meaningful random-tumble model must be smooth in time.
    2. Even smooth, a fast tumble (say one revolution per minute) over a
       lifetime of weeks costs 1e5-1e6 integrator steps to resolve rotation
       that -- by construction -- averages out.

    The average is a modelling limit. ``tests/test_srp.py`` checks it against a
    Monte Carlo average of :func:`mcinnes_srp_acceleration` over random normals
    and the time average of a resolved fast tumble.

    Assumptions:
      - Orientation is uniform on the sphere. For a torque-free rigid body the
        true orientation distribution depends on the inertia tensor and angular
        momentum; uniform is the natural idealisation of "tumbling randomly".
      - The sail is two-sided. A tumbling sail is back-lit half the time, so
        the one-sided model would drop half the impulse; ``sail.optical`` is
        therefore required to have ``two_sided=True``, and the constructor
        rejects a one-sided film to prevent under-thrusting.

    Attributes
    ----------
    sail
        The underlying ``SolarSail``. Its ``optical.two_sided`` must be True.
    """

    sail: SolarSail

    def __post_init__(self) -> None:
        if not self.sail.optical.two_sided:
            raise ValueError(
                "TumbleAveragedSail requires optical.two_sided=True: a "
                "tumbling sail is back-lit about half the time, and the "
                "one-sided model would omit half the SRP impulse "
                "(see srp.mcinnes_srp_acceleration)"
            )

    @property
    def average_coefficient(self) -> float:
        """Dimensionless ``k`` with ``<a> = -k (P A / m) s_hat``.

        Derivation. Write the two-sided force with
        ``mu = |n_hat . s_hat|`` and ``n_ill`` the lit-face normal, so
        ``a = -(P A/m) mu [C_s s_hat + C_n(mu, face) n_ill]`` with
        ``C_s = 1 - rho s`` and ``C_n = 2 rho s mu + B_face (1-s) rho
        + (1-rho) T_face``.

        For orientation uniform on the sphere, ``mu`` is uniform on [0, 1]
        (because ``cos theta`` is uniform on [-1, 1]). At fixed ``mu`` the lit
        normal is uniform on the cone of half-angle ``arccos(mu)`` about
        ``s_hat``, so its average is ``mu s_hat`` -- the transverse part
        cancels by azimuthal symmetry. Hence

            <a> = -(P A/m) [ C_s <mu> + <mu^2 C_n> ] s_hat

        and with ``<mu> = 1/2``, ``<mu^2> = 1/3``, ``<mu^3> = 1/4``, the two
        faces equiprobable (so ``B_face`` averages to ``(B_f + B_b)/2`` and the
        thermal asymmetry ``T_face`` averages to exactly zero, being ``+T``
        front-lit and ``-T`` back-lit):

            k = C_s/2 + rho s/2 + (1 - s) rho (B_f + B_b) / 6

        Sanity values: ideal film (rho = s = 1) gives exactly ``k = 1/2``, i.e.
        one quarter of the ``2 P A/m`` sun-facing peak; the McInnes Table 2.1
        JPL square sail gives ``k = 0.511792...``. The realistic film lands
        within 2.4% of the ideal one precisely because the thermal term -- the
        main front/back asymmetry -- cancels in the average.
        """
        opt = self.sail.optical
        C_s = 1.0 - opt.rho * opt.s
        return (
            0.5 * C_s
            + 0.5 * opt.rho * opt.s
            + (1.0 - opt.s) * opt.rho * (opt.B_front + opt.B_back) / 6.0
        )


def tumble_averaged_acceleration(
    r_sat_j2000_km: np.ndarray,
    et: float,
    tumbling: TumbleAveragedSail,
    *,
    observer_naif_id: int = MARS_NAIF_ID,
    apply_shadow: bool = True,
) -> np.ndarray:
    """Orientation-averaged SRP acceleration on a rapidly tumbling flat sail.

    ``<a> = -k P(r_helio) (A / m) s_hat`` with ``k =
    tumbling.average_coefficient`` (see that property for the derivation) and
    ``s_hat`` the sail-to-Sun unit vector. Same binary Mars-umbra gate as the
    flat-sail and spherical-grain paths.

    Returns
    -------
    ndarray, shape (3,)
        Acceleration in J2000 axes, km/s^2, along the anti-Sun direction.
    """
    r_sat = np.asarray(r_sat_j2000_km, dtype=float)

    state = sun_state_j2000(et, observer_naif_id)
    r_sun = np.asarray(state[:3], dtype=float)

    if apply_shadow and shadow_factor(
        r_sat, et, observer_naif_id, sun_position_j2000_km=r_sun
    ) == 0.0:
        return np.zeros(3, dtype=float)

    sat_to_sun = r_sun - r_sat
    r_helio_km = float(np.linalg.norm(sat_to_sun))
    s_hat = sat_to_sun / r_helio_km

    P_pa = solar_flux_at(r_helio_km)
    sail = tumbling.sail
    a_mag_kmps2 = (
        tumbling.average_coefficient
        * P_pa
        * sail.area_m2
        / sail.mass_kg
        * 1.0e-3
    )
    return -a_mag_kmps2 * s_hat


def make_tumble_averaged_contributor(
    tumbling: TumbleAveragedSail,
    epoch_et: float,
    *,
    observer_naif_id: int = MARS_NAIF_ID,
    apply_shadow: bool = True,
    ephemeris_time_direction: int = +1,
):
    """Build a ``(r, t_offset) -> <a_SRP>`` closure for the propagator.

    Parallels ``make_srp_contributor`` / ``make_spherical_srp_contributor``.
    ``propagate`` constructs this internally from its
    ``tumble_averaged_sail`` kwarg.
    """

    def contributor(r: np.ndarray, t_offset: float) -> np.ndarray:
        return tumble_averaged_acceleration(
            r,
            epoch_et + ephemeris_time_direction * t_offset,
            tumbling,
            observer_naif_id=observer_naif_id,
            apply_shadow=apply_shadow,
        )

    return contributor


def make_spherical_srp_contributor(
    particle: SphericalParticle,
    epoch_et: float,
    *,
    observer_naif_id: int = MARS_NAIF_ID,
    apply_shadow: bool = True,
    ephemeris_time_direction: int = +1,
):
    """Build a ``(r, t_offset) -> a_SRP_sphere`` closure for the propagator.

    Parallels ``make_srp_contributor`` for the spherical-grain path.
    ``propagate`` constructs this internally when called with a
    ``spherical_particle`` kwarg; exposed so downstream modules can
    assemble RHS contributor lists by hand if needed. ``apply_shadow``
    (default ``True``) threads through to
    ``spherical_particle_acceleration`` -- set ``False`` for the H&K96
    shadow-neglect convention.

    ``ephemeris_time_direction`` (+1 forward, -1 backward) sets the sign of the
    ephemeris clock (Sun position evaluated at
    ``epoch_et + ephemeris_time_direction * t_offset``). The default is +1.
    """

    def contributor(r: np.ndarray, t_offset: float) -> np.ndarray:
        return spherical_particle_acceleration(
            r,
            epoch_et + ephemeris_time_direction * t_offset,
            particle,
            observer_naif_id=observer_naif_id,
            apply_shadow=apply_shadow,
        )

    return contributor
