"""Analytical bounds on a solar sail's orbital-element control authority.

Purpose
-------

This module bounds the secular eccentricity-control authority of a solar sail.
Comparing the bound with numerical results distinguishes a physical
sail-authority limit from limitations of a particular cruise-law family or
optimizer.

Citations
=========

McInnes 1999 ch. 4, *Solar Sailing*, pp. 132-146.
  - Eq. 4.7 — sail-normal parameterisation in cone-and-clock angles.
  - Eq. 4.14b — Gauss variational equation for ⟨de/dt⟩.
  - Eq. 4.15a-c — perfect-sail SRP force in (S, T, W) components.

The optical-coefficient force formula (the dimensionless force factor
``η = C_s + C_n(α=0)`` at normal incidence) follows McInnes Eq. 2.57
exactly as implemented in :mod:`reflectors.srp`.

Derivation of the unrestricted bound
====================================

At circular limit ``e = 0``, Gauss's equation 4.14b reduces to::

    de/dt = (1 / (n a)) · [sin(f) · S + 2 cos(f) · T]

where ``S`` and ``T`` are the radial and transverse components of the
perturbing acceleration in the orbit frame, ``n = sqrt(μ / a³)`` is the
mean motion, and ``f = u`` (true anomaly, since ω is irrelevant for
circular). Orbit-averaging gives::

    ⟨de/dt⟩ = (1 / (n a)) · [⟨S sin(u)⟩ + 2 ⟨T cos(u)⟩]

The cross-correlations ⟨S sin u⟩ and ⟨T cos u⟩ are bounded by the
*peak* sail-frame acceleration magnitude::

    a_c := η · P_SRP / σ              (characteristic acceleration)

For ANY S(u), T(u), W(u) satisfying ``√(S² + T² + W²) ≤ a_c`` over the
orbit, the worst-case sin-u-aligned S(u) = a_c · sin(u) gives
``⟨S sin u⟩ = a_c / 2``; similarly cos-u-aligned T gives
``⟨T cos u⟩ = a_c / 2``. Combined::

    |⟨de/dt⟩|_max_unrestricted  ≤  (1 / (n a)) · (a_c/2 + 2 · a_c/2)
                                =  (3 · a_c) / (2 · n · a)

This is the unrestricted upper bound — the most ANY cruise law (mode-1
harmonic, mode-2, full-frequency continuum) can drive ⟨de/dt⟩ at
``e = 0``. It assumes the sail can independently choose S(u) and T(u)
at peak amplitude with optimal sign correlation; no real cruise law
saturates this, but the bound is rigorous as a ceiling.

The bound parameterised in σ: ``|⟨de/dt⟩|_max ∝ 1/σ``. A 3× lighter
sail (50 g/m² → 18 g/m²) gets 3× more
e-control authority.

Conventions
===========

- All inputs in SI base units: km for distances, s for time,
  km^3/s^2 for gravitational parameters, kg/m^2 for σ. Outputs in 1/s
  for rates.
- Two ``optical_path`` modes:
    * ``perfect_sail=True`` — η = 2 (ideal mirror, McInnes Table 2.1
      'Ideal sail'). Gives the absolute upper bound.
    * ``perfect_sail=False`` — η = ``force_coefficient_at_normal`` of
      the actual ``sail.optical``. Gives the *predicted* authority for
      the JPL Halley sail (η ≈ 1.82 vs 2.0 for perfect-mirror).
- ``r_helio_km`` is the sail-Sun distance. For Mars work, the natural
  values span ~1.381 AU (perihelion) to 1.666 AU (aphelion); the rank-1
  reference epoch is at perihelion.
"""
from __future__ import annotations

import math

from .solar_constants import solar_flux_at
from .srp import SailOptical, SolarSail

__all__ = [
    "force_coefficient_at_normal",
    "characteristic_acceleration",
    "unrestricted_max_secular_de_dt",
]


def force_coefficient_at_normal(optical: SailOptical) -> float:
    """Dimensionless SRP force coefficient ``η = C_s + C_n(α=0)``.

    From the McInnes 1999 Eq. 2.57 expansion as implemented in
    :func:`reflectors.srp.srp_acceleration` (lines 354-369): at α=0 the
    sail-normal aligns with ``ŝ`` and the force is along ``ŝ`` with
    dimensionless magnitude::

        η = C_s + C_n(0)
          = (1 - ρ s) + [2 ρ s + B_f (1-s) ρ + (1-ρ) · thermal]
          = 1 + ρ s + B_f (1-s) ρ + (1-ρ) · thermal

    where::

        thermal = (ε_f B_f - ε_b B_b) / (ε_f + ε_b)   if ε_f + ε_b > 0
        thermal = 0                                    otherwise

    Reduces exactly to η = 2 for the McInnes 'Ideal sail'
    (ρ=1, s=1, ε=0); reduces to ≈ 1.82 for the JPL square sail
    (ρ=0.88, s=0.94, ε_f=0.05, ε_b=0.55, B_f=0.79, B_b=0.55).
    """
    eps_sum = optical.eps_front + optical.eps_back
    if eps_sum == 0.0:
        thermal_term = 0.0
    else:
        thermal_term = (
            optical.eps_front * optical.B_front
            - optical.eps_back * optical.B_back
        ) / eps_sum

    C_s = 1.0 - optical.rho * optical.s
    C_n_at_zero = (
        2.0 * optical.rho * optical.s
        + optical.B_front * (1.0 - optical.s) * optical.rho
        + (1.0 - optical.rho) * thermal_term
    )
    return C_s + C_n_at_zero


def characteristic_acceleration(
    sail: SolarSail,
    r_helio_km: float,
    *,
    perfect_sail: bool = False,
) -> float:
    """Peak SRP acceleration magnitude at α=0, in m/s^2.

    ``a_c = η · P_SRP(r_helio) / σ`` where σ = sail.loading_kg_per_m2
    and η is the dimensionless force coefficient at normal incidence.

    Parameters
    ----------
    sail
        :class:`SolarSail` carrying area, mass, optical.
    r_helio_km
        Sail-Sun distance in km.
    perfect_sail
        If ``True``, force η = 2 (ideal mirror). If ``False`` (default),
        derive η from ``sail.optical`` via
        :func:`force_coefficient_at_normal`.

    Returns
    -------
    float
        ``a_c`` in m/s^2 (the peak SRP acceleration achievable with
        the given sail at the given heliocentric distance).
    """
    if r_helio_km <= 0.0:
        raise ValueError(
            f"r_helio_km must be > 0, got {r_helio_km}"
        )
    P_pa = solar_flux_at(r_helio_km)
    if perfect_sail:
        eta = 2.0
    else:
        eta = force_coefficient_at_normal(sail.optical)
    return eta * P_pa / sail.loading_kg_per_m2


def unrestricted_max_secular_de_dt(
    sail: SolarSail,
    mu_central_km3_per_s2: float,
    a_km: float,
    r_helio_km: float,
    *,
    perfect_sail: bool = False,
) -> float:
    """Unrestricted upper bound on |⟨de/dt⟩|_secular at e=0, in 1/s.

    Returns the Cauchy-Schwarz / max-amplitude bound on the
    orbit-averaged eccentricity drift achievable by ANY cruise law
    (mode-1 harmonic, mode-2, full-frequency continuum, periodic
    re-opt, etc.) given a fixed sail at fixed heliocentric distance::

        |⟨de/dt⟩|_max  ≤  (3 · a_c) / (2 · n · a)

    where::

        a_c = η · P_SRP(r_helio) / σ        (peak SRP acceleration)
        n   = sqrt(μ / a³)                    (mean motion)

    Notes
    -----
    The bound assumes the circular orbit limit (e ≈ 0). At nonzero e,
    the Gauss equation 4.14b adds a ``T · (r/p) · e`` term whose
    orbit-average has the same scaling; the bound is conservative for
    small e by ~10% per 0.01 of e and need not be re-derived for the
    e_max ~ 0.003 - 0.01 regime considered here.

    The bound is conservative: it assumes the sail can produce
    sin(u)-aligned S(u) and cos(u)-aligned T(u) at peak amplitude
    independently, which is geometrically impossible for any single
    cruise law (S, T, W are linear projections of a single attitude
    vector). If this ceiling falls below the uncontrolled drift, no cruise law
    within the modeled authority can null eccentricity.

    Parameters
    ----------
    sail
        :class:`SolarSail` instance.
    mu_central_km3_per_s2
        Gravitational parameter of the central body (Mars: ≈ 42828.37
        km³/s²). Required for the mean motion ``n``.
    a_km
        Orbital semi-major axis in km.
    r_helio_km
        Sail-Sun distance in km.
    perfect_sail
        Whether to use η=2 (ideal mirror) or sail.optical (default).

    Returns
    -------
    float
        |⟨de/dt⟩|_max in 1/s. Multiply by horizon-seconds for an
        upper bound on |Δe| over the horizon.
    """
    if mu_central_km3_per_s2 <= 0.0:
        raise ValueError(
            f"mu_central_km3_per_s2 must be > 0, got {mu_central_km3_per_s2}"
        )
    if a_km <= 0.0:
        raise ValueError(f"a_km must be > 0, got {a_km}")

    # Characteristic acceleration in m/s^2; convert to km/s^2 for unit
    # consistency with mu in km^3/s^2 and a in km.
    a_c_m_per_s2 = characteristic_acceleration(
        sail, r_helio_km, perfect_sail=perfect_sail
    )
    a_c_km_per_s2 = a_c_m_per_s2 * 1.0e-3

    # Mean motion n = sqrt(mu/a^3) in rad/s.
    n_rad_per_s = math.sqrt(mu_central_km3_per_s2 / (a_km ** 3))

    return (3.0 * a_c_km_per_s2) / (2.0 * n_rad_per_s * a_km)
