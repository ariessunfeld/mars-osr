"""Canonical sail-design factory.

Single chokepoint for constructing :class:`~reflectors.srp.SolarSail`
instances parameterised by sail loading sigma = mass / area (kg/m^2).

Rationale
=========

Centralizing sail construction lets scripts and tests vary areal density
without duplicating area, mass, and optical parameters. ``loading_kg_per_m2``
is required positional
(no default) on purpose: changing the operating point implicitly would
invalidate optimisation results across the codebase.
"""
from __future__ import annotations

from typing import Optional

from .srp import SailOptical, SolarSail

__all__ = ["make_canonical_sail"]


def make_canonical_sail(
    loading_kg_per_m2: float,
    *,
    area_m2: float = 1000.0,
    optical: Optional[SailOptical] = None,
) -> SolarSail:
    """Build a :class:`SolarSail` with the given areal density.

    Parameters
    ----------
    loading_kg_per_m2
        Sail loading sigma = mass / area in kg/m^2. Required positional.
        Must be strictly positive.
    area_m2
        Effective reflective area in m^2. Defaults to the canonical
        1000 m^2 (JPL Halley-rendezvous baseline).
    optical
        Optical coefficient set. Defaults to
        :meth:`SailOptical.square_sail_jpl` (McInnes Table 2.1 'Square
        sail': rho=0.88, s=0.94, eps_f=0.05, eps_b=0.55, B_f=0.79,
        B_b=0.55).

    Returns
    -------
    SolarSail
        Frozen dataclass with ``mass_kg = loading_kg_per_m2 * area_m2``.

    Raises
    ------
    ValueError
        If ``loading_kg_per_m2 <= 0`` or ``area_m2 <= 0`` (the latter
        propagating from ``SolarSail.__post_init__``).
    """
    if loading_kg_per_m2 <= 0.0:
        raise ValueError(
            f"loading_kg_per_m2 must be > 0, got {loading_kg_per_m2}"
        )
    if optical is None:
        optical = SailOptical.square_sail_jpl()
    return SolarSail(
        area_m2=area_m2,
        mass_kg=loading_kg_per_m2 * area_m2,
        optical=optical,
    )
