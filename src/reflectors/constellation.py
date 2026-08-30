"""Walker-style geometric sizing for the manuscript constellation.

This module implements the constellation-sizing calculation used by the
companion manuscript. A shell contains circular,
equal-altitude rings distributed uniformly over a centered LTAN interval.
For ``P`` rings and ``S`` reflectors per ring, the initial argument of
latitude is

``u[p,s] = u0 + 2*pi*s/S + 2*pi*F*p/(P*S)``.

The integer search maximizes ``P*S`` subject to the manuscript's plane-density,
same-ring chord, and equal-altitude inter-ring closest-approach constraints.
The inter-ring distance is a purely geometric circular-orbit result; it is not
a covariance-based conjunction assessment.

Two orbit-model profiles are intentionally distinct:

``manuscript-v1``
    Frozen constants used for the manuscript constellation results. This is
    the default reproducibility profile.

``mro120f``
    Live PCK equatorial radius, MRO120F degree-2 gravity
    anchors, and Mars-year constant.  Hard packing thresholds make a few shell
    optima differ from the frozen manuscript calculation.

References
----------
Walker, J. G. (1984), "Satellite Constellations," *Journal of the British
Interplanetary Society* 37, 559.  The ``F`` convention here is the usual
Walker phase increment of ``360 F / (P S)`` between equivalent satellites in
adjacent rings.

Brouwer, D. (1959), "Solution of the problem of artificial satellite theory
without drag," *Astronomical Journal* 64, 378--397.  The first-order J2
sun-synchronous inclination follows the same secular node-rate condition as
``reflectors.sun_sync.sun_sync_inclination_rad``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


logger = logging.getLogger(__name__)

JULIAN_YEAR_S = 365.25 * 86400.0
MANUSCRIPT_V1_PROFILE_NAME = "manuscript-v1"
MRO120F_PROFILE_NAME = "mro120f"


@dataclass(frozen=True)
class OrbitModel:
    """Constants used for altitude conversion and first-order J2 inclination."""

    name: str
    equatorial_radius_km: float
    mu_km3_s2: float
    j2_reference_radius_km: float
    j2: float
    sidereal_year_s: float
    provenance: str

    def __post_init__(self) -> None:
        positive = {
            "equatorial_radius_km": self.equatorial_radius_km,
            "mu_km3_s2": self.mu_km3_s2,
            "j2_reference_radius_km": self.j2_reference_radius_km,
            "j2": self.j2,
            "sidereal_year_s": self.sidereal_year_s,
        }
        for field_name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and > 0; got {value!r}")
        if not self.name:
            raise ValueError("orbit-model name must be non-empty")

    def sun_sync_inclination_rad(self, semimajor_axis_km: float) -> float:
        """Return the retrograde first-order J2 sun-synchronous inclination.

        For a circular orbit, Brouwer's secular node rate is

        ``Omega_dot = -(3/2) J2 sqrt(mu/a^3) (R/a)^2 cos(i)``.

        It is set equal to ``2*pi/sidereal_year_s``.  All quantities use km
        and seconds, so the solved inclination is dimensionless (radians).
        """
        if not math.isfinite(semimajor_axis_km) or semimajor_axis_km <= 0.0:
            raise ValueError(
                "semimajor_axis_km must be finite and > 0; "
                f"got {semimajor_axis_km!r}"
            )
        mean_motion_rad_s = math.sqrt(self.mu_km3_s2 / semimajor_axis_km**3)
        target_node_rate_rad_s = 2.0 * math.pi / self.sidereal_year_s
        coefficient = (
            1.5
            * self.j2
            * mean_motion_rad_s
            * (self.j2_reference_radius_km / semimajor_axis_km) ** 2
        )
        cos_inclination = -target_node_rate_rad_s / coefficient
        if abs(cos_inclination) > 1.0:
            raise ValueError(
                "no first-order sun-synchronous solution at "
                f"a={semimajor_axis_km:.6f} km for profile {self.name!r}: "
                f"cos(i)={cos_inclination:.6g}"
            )
        return math.acos(cos_inclination)


MANUSCRIPT_V1_ORBIT_MODEL = OrbitModel(
    name=MANUSCRIPT_V1_PROFILE_NAME,
    equatorial_radius_km=3396.19,
    mu_km3_s2=42828.375816,
    j2_reference_radius_km=3396.19,
    j2=1.96045e-3,
    sidereal_year_s=1.8808476 * JULIAN_YEAR_S,
    provenance=(
        "Frozen constants used for the manuscript constellation results."
    ),
)


def mro120f_orbit_model() -> OrbitModel:
    """Build the live PCK/MRO120F orbit-model profile.

    The caller must furnish the SPICE kernels before calling this function,
    because the altitude reference is the live PCK equatorial radius.
    """
    from reflectors.gravity import mars_gravity_model, zonal_coefficients
    from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S
    from reflectors.surface import mars_equatorial_radius_km

    gravity_model = mars_gravity_model(max_degree=2)
    return OrbitModel(
        name=MRO120F_PROFILE_NAME,
        equatorial_radius_km=mars_equatorial_radius_km(),
        mu_km3_s2=float(gravity_model.mu_km3_s2),
        j2_reference_radius_km=float(gravity_model.ref_radius_km),
        j2=float(zonal_coefficients(gravity_model, 2)[2]),
        sidereal_year_s=MARS_SIDEREAL_YEAR_S,
        provenance=(
            "Live PCK BODY499_RADII altitude reference, MRO120F degree-2 "
            "mu/reference-radius/J2, and reflectors.mars_constants Mars year."
        ),
    )


@dataclass(frozen=True)
class FamilySpec:
    """One repeat-ground-track family and its whole-year eclipse-free LTAN band."""

    k_orbits_per_sol: int
    base_altitude_km: float
    eclipse_free_ltan_start_h: float
    eclipse_free_ltan_end_h: float

    def __post_init__(self) -> None:
        if isinstance(self.k_orbits_per_sol, bool) or self.k_orbits_per_sol < 1:
            raise ValueError("k_orbits_per_sol must be an integer >= 1")
        if int(self.k_orbits_per_sol) != self.k_orbits_per_sol:
            raise ValueError("k_orbits_per_sol must be an integer >= 1")
        if not math.isfinite(self.base_altitude_km) or self.base_altitude_km < 0.0:
            raise ValueError("base_altitude_km must be finite and >= 0")
        lo = self.eclipse_free_ltan_start_h
        hi = self.eclipse_free_ltan_end_h
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi <= 24.0):
            raise ValueError(
                "eclipse-free LTAN endpoints must satisfy 0 <= start < end <= 24 h"
            )

    def centered_ltan_band_h(self, center_h: float = 18.0) -> tuple[float, float]:
        """Widest subinterval centered on ``center_h`` inside the raw band."""
        return centered_ltan_subband_h(
            self.eclipse_free_ltan_start_h,
            self.eclipse_free_ltan_end_h,
            center_h,
        )


def centered_ltan_subband_h(
    eclipse_free_ltan_start_h: float,
    eclipse_free_ltan_end_h: float,
    center_h: float = 18.0,
) -> tuple[float, float]:
    """Widest interval centered on ``center_h`` inside an eclipse-free band."""
    lo = eclipse_free_ltan_start_h
    hi = eclipse_free_ltan_end_h
    if not (
        math.isfinite(lo)
        and math.isfinite(hi)
        and math.isfinite(center_h)
        and 0.0 <= lo < hi <= 24.0
    ):
        raise ValueError(
            "LTAN endpoints and center must be finite, with "
            "0 <= start < end <= 24 h"
        )
    if not lo <= center_h <= hi:
        raise ValueError(
            f"LTAN center {center_h} h is outside eclipse-free band [{lo}, {hi}] h"
        )
    half_width_h = min(center_h - lo, hi - center_h)
    return center_h - half_width_h, center_h + half_width_h


# Reference whole-year eclipse-free bands used by the manuscript-v1 profile.
DEFAULT_FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec(12, 508.0, 17.381, 18.376),
    FamilySpec(11, 741.0, 16.742, 19.146),
    FamilySpec(10, 1012.0, 16.359, 19.597),
    FamilySpec(9, 1332.0, 16.014, 19.941),
)


@dataclass(frozen=True)
class ExplicitShellSpec:
    """One explicitly located shell with its own computed eclipse-free band.

    ``reference_family_k`` labels the nearest reference repeat-ground-track
    anchor. Away from that anchor it is a label, not a claim that the shell
    completes exactly ``K`` revolutions per sol.
    """

    reference_family_k: int
    shell_index: int
    altitude_km: float
    eclipse_free_ltan_start_h: float
    eclipse_free_ltan_end_h: float
    band_method: str
    lower_binding_true_anomaly_deg: float | None = None
    upper_binding_true_anomaly_deg: float | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "reference_family_k": self.reference_family_k,
            "shell_index": self.shell_index,
        }
        for field_name, value in integer_fields.items():
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{field_name} must be an integer >= 0")
        if self.reference_family_k < 1:
            raise ValueError("reference_family_k must be >= 1")
        if not math.isfinite(self.altitude_km) or self.altitude_km < 0.0:
            raise ValueError("altitude_km must be finite and >= 0")
        centered_ltan_subband_h(
            self.eclipse_free_ltan_start_h,
            self.eclipse_free_ltan_end_h,
        )
        if not self.band_method:
            raise ValueError("band_method must be non-empty")
        for field_name, value in (
            ("lower_binding_true_anomaly_deg", self.lower_binding_true_anomaly_deg),
            ("upper_binding_true_anomaly_deg", self.upper_binding_true_anomaly_deg),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite or None")


@dataclass(frozen=True)
class WalkerSizingConstraints:
    """Geometric packing assumptions from constellation-sizing v1."""

    shell_count: int = 21
    shell_spacing_km: float = 5.0
    centered_ltan_h: float = 18.0
    min_raan_spacing_deg: float = 5.0
    min_same_ring_spacing_km: float = 300.0
    min_inter_ring_spacing_km: float = 50.0

    def __post_init__(self) -> None:
        if isinstance(self.shell_count, bool) or int(self.shell_count) != self.shell_count:
            raise ValueError("shell_count must be an integer >= 1")
        if self.shell_count < 1:
            raise ValueError("shell_count must be an integer >= 1")
        positive = {
            "shell_spacing_km": self.shell_spacing_km,
            "min_raan_spacing_deg": self.min_raan_spacing_deg,
            "min_same_ring_spacing_km": self.min_same_ring_spacing_km,
            "min_inter_ring_spacing_km": self.min_inter_ring_spacing_km,
        }
        for field_name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and > 0; got {value!r}")
        if not math.isfinite(self.centered_ltan_h) or not 0.0 <= self.centered_ltan_h <= 24.0:
            raise ValueError("centered_ltan_h must be finite and in [0, 24]")


def family_stack_altitudes_km(
    family_specs: Sequence[FamilySpec] = DEFAULT_FAMILY_SPECS,
    constraints: WalkerSizingConstraints = WalkerSizingConstraints(),
) -> tuple[float, ...]:
    """Return every altitude center in the fixed-count stacks."""
    altitudes = [
        family.base_altitude_km + shell_index * constraints.shell_spacing_km
        for family in family_specs
        for shell_index in range(constraints.shell_count)
    ]
    if len(set(altitudes)) != len(altitudes):
        raise ValueError("family stacks contain duplicate altitude centers")
    return tuple(sorted(altitudes))


def maximally_fill_altitude_gaps_km(
    preserved_altitudes_km: Sequence[float],
    minimum_spacing_km: float,
) -> tuple[float, ...]:
    """Preserve anchor centers and greedily fill every gap at minimum spacing.

    The result is the maximum-cardinality ordered set that contains all input
    centers, introduces no center below the minimum or above the maximum input,
    and keeps adjacent centers at least ``minimum_spacing_km`` apart.  New
    centers advance upward from the lower preserved center.  Any indivisible
    remainder is therefore left immediately below the next preserved center.
    """
    if not math.isfinite(minimum_spacing_km) or minimum_spacing_km <= 0.0:
        raise ValueError("minimum_spacing_km must be finite and > 0")
    preserved = tuple(sorted(float(value) for value in preserved_altitudes_km))
    if not preserved:
        raise ValueError("preserved_altitudes_km must not be empty")
    if any(not math.isfinite(value) for value in preserved):
        raise ValueError("preserved_altitudes_km must contain only finite values")
    tolerance_km = 1.0e-10 * max(1.0, max(abs(value) for value in preserved))
    for lower, upper in zip(preserved, preserved[1:]):
        if upper - lower < minimum_spacing_km - tolerance_km:
            raise ValueError(
                "preserved altitude centers violate the requested minimum spacing: "
                f"{lower} and {upper} km"
            )

    filled = [preserved[0]]
    for preserved_upper in preserved[1:]:
        candidate = filled[-1] + minimum_spacing_km
        while preserved_upper - candidate >= minimum_spacing_km - tolerance_km:
            filled.append(candidate)
            candidate += minimum_spacing_km
        filled.append(preserved_upper)
    return tuple(filled)


def nearest_reference_family_k(
    altitude_km: float,
    family_specs: Sequence[FamilySpec] = DEFAULT_FAMILY_SPECS,
) -> int:
    """Label an arbitrary altitude by its nearest reference K anchor."""
    specs = tuple(family_specs)
    if not specs:
        raise ValueError("family_specs must not be empty")
    if not math.isfinite(altitude_km):
        raise ValueError("altitude_km must be finite")
    nearest = min(
        enumerate(specs),
        key=lambda item: (abs(altitude_km - item[1].base_altitude_km), item[0]),
    )[1]
    return nearest.k_orbits_per_sol


@dataclass(frozen=True)
class ClosestApproach:
    """Limiting equal-altitude inter-ring satellite pair for one candidate."""

    distance_km: float
    ring_index_separation: int
    relative_satellite_index: int
    raan_separation_deg: float
    phase_difference_deg: float


@dataclass(frozen=True)
class WalkerCandidate:
    """One feasible ``(P, S, F)`` candidate in a circular altitude shell."""

    rings: int
    satellites_per_ring: int
    phasing: int
    same_ring_spacing_km: float
    closest_approach: ClosestApproach | None

    @property
    def satellite_count(self) -> int:
        return self.rings * self.satellites_per_ring


@dataclass(frozen=True)
class ShellSolution:
    """Selected Walker candidate and geometry for one altitude shell."""

    family_k: int
    shell_index: int
    altitude_km: float
    semimajor_axis_km: float
    inclination_deg: float
    raw_ltan_start_h: float
    raw_ltan_end_h: float
    centered_ltan_start_h: float
    centered_ltan_end_h: float
    raan_span_deg: float
    maximum_rings: int
    rings: int
    satellites_per_ring: int
    phasing: int
    satellite_count: int
    same_ring_spacing_km: float
    minimum_inter_ring_spacing_km: float | None
    limiting_ring_index_separation: int | None
    limiting_relative_satellite_index: int | None
    limiting_raan_separation_deg: float | None
    limiting_phase_difference_deg: float | None
    band_method: str = "fixed-family-band"
    lower_binding_true_anomaly_deg: float | None = None
    upper_binding_true_anomaly_deg: float | None = None


@dataclass(frozen=True)
class WalkerConstellation:
    """A complete collection of independently sized altitude shells."""

    orbit_model: OrbitModel
    constraints: WalkerSizingConstraints
    family_specs: tuple[FamilySpec, ...]
    shells: tuple[ShellSolution, ...]
    shell_layout_name: str = "fixed-count-family-stacks"
    explicit_shell_specs: tuple[ExplicitShellSpec, ...] = ()

    @property
    def satellite_count(self) -> int:
        return sum(shell.satellite_count for shell in self.shells)

    @property
    def ring_count(self) -> int:
        return sum(shell.rings for shell in self.shells)

    def family_shells(self, family_k: int) -> tuple[ShellSolution, ...]:
        return tuple(shell for shell in self.shells if shell.family_k == family_k)


def same_ring_chord_spacing_km(
    semimajor_axis_km: float,
    satellites_per_ring: int,
) -> float:
    """Chord between adjacent satellites uniformly spaced on one circular ring."""
    if not math.isfinite(semimajor_axis_km) or semimajor_axis_km <= 0.0:
        raise ValueError("semimajor_axis_km must be finite and > 0")
    if (
        isinstance(satellites_per_ring, bool)
        or int(satellites_per_ring) != satellites_per_ring
        or satellites_per_ring < 2
    ):
        raise ValueError("satellites_per_ring must be an integer >= 2")
    return 2.0 * semimajor_axis_km * math.sin(math.pi / satellites_per_ring)


def maximum_satellites_per_ring(
    semimajor_axis_km: float,
    minimum_chord_km: float,
) -> int:
    """Largest integer ``S >= 2`` satisfying the same-ring chord threshold."""
    if not math.isfinite(semimajor_axis_km) or semimajor_axis_km <= 0.0:
        raise ValueError("semimajor_axis_km must be finite and > 0")
    if not math.isfinite(minimum_chord_km) or minimum_chord_km <= 0.0:
        raise ValueError("minimum_chord_km must be finite and > 0")
    ratio = minimum_chord_km / (2.0 * semimajor_axis_km)
    if ratio > 1.0:
        raise ValueError(
            f"minimum chord {minimum_chord_km} km exceeds circular diameter "
            f"{2.0 * semimajor_axis_km} km"
        )
    estimate = max(2, math.floor(math.pi / math.asin(ratio)))

    # Resolve a possible floating-point floor exactly against the defining
    # inequality rather than padding the hard physical threshold.
    while same_ring_chord_spacing_km(semimajor_axis_km, estimate) < minimum_chord_km:
        estimate -= 1
    while same_ring_chord_spacing_km(semimajor_axis_km, estimate + 1) >= minimum_chord_km:
        estimate += 1
    return estimate


def _wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def equal_altitude_minimum_distance_km(
    semimajor_axis_km: float,
    inclination_rad: float,
    raan_separation_rad: float,
    phase_difference_rad: float | np.ndarray,
) -> float | np.ndarray:
    """Minimum Euclidean separation of two equal-altitude circular satellites.

    The satellites have common ``a`` and ``i``, RAAN separation ``D``, and
    fixed argument-of-latitude offset ``delta``.  Minimizing their Cartesian
    distance over their common orbital phase gives

    ``d = 2 a sqrt(r) |sin(wrap(delta - delta0)/2)|``,

    where

    ``q = [cos(D) + sin(i)^2 + cos(i)^2 cos(D)] / 2``,
    ``r = hypot(q, cos(i) sin(D))``, and
    ``delta0 = atan2(-cos(i) sin(D), q)``.

    This is the closed-form evaluation of the Euclidean one-orbit minimum.
    It is independently checked against direct Cartesian
    minimization in ``tests/test_constellation.py``.
    """
    if not math.isfinite(semimajor_axis_km) or semimajor_axis_km <= 0.0:
        raise ValueError("semimajor_axis_km must be finite and > 0")
    if not math.isfinite(inclination_rad) or not 0.0 <= inclination_rad <= math.pi:
        raise ValueError("inclination_rad must be finite and in [0, pi]")
    if not math.isfinite(raan_separation_rad):
        raise ValueError("raan_separation_rad must be finite")
    phase = np.asarray(phase_difference_rad, dtype=float)
    if not np.all(np.isfinite(phase)):
        raise ValueError("phase_difference_rad must contain only finite values")

    cos_i = math.cos(inclination_rad)
    sin_i = math.sin(inclination_rad)
    cos_D = math.cos(raan_separation_rad)
    sin_D = math.sin(raan_separation_rad)
    q = 0.5 * (cos_D + sin_i**2 + cos_i**2 * cos_D)
    r = math.hypot(q, cos_i * sin_D)
    delta0 = math.atan2(-cos_i * sin_D, q)
    wrapped_offset = _wrap_to_pi(phase - delta0)
    distance = (
        2.0
        * semimajor_axis_km
        * math.sqrt(r)
        * np.abs(np.sin(0.5 * wrapped_offset))
    )
    if distance.ndim == 0:
        return float(distance)
    return distance


def candidate_closest_approach(
    semimajor_axis_km: float,
    inclination_rad: float,
    raan_span_rad: float,
    rings: int,
    satellites_per_ring: int,
    phasing: int,
) -> ClosestApproach | None:
    """Minimum across every distinct ring separation and satellite offset."""
    integer_values = {
        "rings": rings,
        "satellites_per_ring": satellites_per_ring,
        "phasing": phasing,
    }
    for field_name, value in integer_values.items():
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{field_name} must be an integer")
    if rings < 1:
        raise ValueError("rings must be >= 1")
    if satellites_per_ring < 2:
        raise ValueError("satellites_per_ring must be >= 2")
    if not 0 <= phasing < rings:
        raise ValueError("phasing must satisfy 0 <= F < P")
    if not math.isfinite(raan_span_rad) or raan_span_rad < 0.0:
        raise ValueError("raan_span_rad must be finite and >= 0")
    if rings == 1:
        return None

    relative_satellite_indices = np.arange(satellites_per_ring, dtype=float)
    best: ClosestApproach | None = None
    for ring_separation in range(1, rings):
        raan_separation_rad = ring_separation * raan_span_rad / (rings - 1)
        phase_differences_rad = (
            2.0 * math.pi * relative_satellite_indices / satellites_per_ring
            + 2.0
            * math.pi
            * phasing
            * ring_separation
            / (rings * satellites_per_ring)
        )
        distances_km = np.asarray(
            equal_altitude_minimum_distance_km(
                semimajor_axis_km,
                inclination_rad,
                raan_separation_rad,
                phase_differences_rad,
            )
        )
        relative_index = int(np.argmin(distances_km))
        distance_km = float(distances_km[relative_index])
        candidate = ClosestApproach(
            distance_km=distance_km,
            ring_index_separation=ring_separation,
            relative_satellite_index=relative_index,
            raan_separation_deg=math.degrees(raan_separation_rad),
            phase_difference_deg=math.degrees(phase_differences_rad[relative_index]) % 360.0,
        )
        if best is None or candidate.distance_km < best.distance_km:
            best = candidate
    assert best is not None
    return best


def maximum_ring_count(raan_span_deg: float, minimum_raan_spacing_deg: float) -> int:
    """Manuscript Eq. (plane count): ``1 + floor(span/minimum spacing)``."""
    if not math.isfinite(raan_span_deg) or raan_span_deg < 0.0:
        raise ValueError("raan_span_deg must be finite and >= 0")
    if not math.isfinite(minimum_raan_spacing_deg) or minimum_raan_spacing_deg <= 0.0:
        raise ValueError("minimum_raan_spacing_deg must be finite and > 0")
    return 1 + math.floor(raan_span_deg / minimum_raan_spacing_deg)


def _candidate_clearance(candidate: WalkerCandidate) -> float:
    if candidate.closest_approach is None:
        return math.inf
    return candidate.closest_approach.distance_km


def _candidate_score(candidate: WalkerCandidate) -> tuple[float, ...]:
    # First two fields are the manuscript rules.  Smaller P and F make exact
    # geometric equivalences deterministic without changing the stated result.
    return (
        float(candidate.satellite_count),
        _candidate_clearance(candidate),
        float(-candidate.rings),
        float(-candidate.phasing),
    )


def size_walker_shell(
    semimajor_axis_km: float,
    inclination_rad: float,
    raan_span_rad: float,
    constraints: WalkerSizingConstraints = WalkerSizingConstraints(),
) -> tuple[int, WalkerCandidate]:
    """Search integer ``(P,S,F)`` and return ``(P_max, optimum)``.

    For each ``P``, ``S`` is visited from the same-ring maximum downward.  Once
    any ``F`` is feasible at a given ``S``, smaller ``S`` cannot improve ``P*S``
    for that ``P`` and is not evaluated.  All ``P`` values remain in the search.
    """
    raan_span_deg = math.degrees(raan_span_rad)
    p_max = maximum_ring_count(raan_span_deg, constraints.min_raan_spacing_deg)
    s_max = maximum_satellites_per_ring(
        semimajor_axis_km,
        constraints.min_same_ring_spacing_km,
    )
    best: WalkerCandidate | None = None

    for rings in range(1, p_max + 1):
        best_for_ring_count: WalkerCandidate | None = None
        for satellites_per_ring in range(s_max, 1, -1):
            same_ring_spacing_km = same_ring_chord_spacing_km(
                semimajor_axis_km,
                satellites_per_ring,
            )
            for phasing in range(rings):
                closest = candidate_closest_approach(
                    semimajor_axis_km,
                    inclination_rad,
                    raan_span_rad,
                    rings,
                    satellites_per_ring,
                    phasing,
                )
                if (
                    closest is not None
                    and closest.distance_km < constraints.min_inter_ring_spacing_km
                ):
                    continue
                candidate = WalkerCandidate(
                    rings=rings,
                    satellites_per_ring=satellites_per_ring,
                    phasing=phasing,
                    same_ring_spacing_km=same_ring_spacing_km,
                    closest_approach=closest,
                )
                if (
                    best_for_ring_count is None
                    or _candidate_score(candidate) > _candidate_score(best_for_ring_count)
                ):
                    best_for_ring_count = candidate
            if best_for_ring_count is not None:
                break

        if best_for_ring_count is not None and (
            best is None or _candidate_score(best_for_ring_count) > _candidate_score(best)
        ):
            best = best_for_ring_count

    if best is None:
        raise RuntimeError(
            "no feasible Walker candidate found at "
            f"a={semimajor_axis_km:.6f} km, i={math.degrees(inclination_rad):.6f} deg"
        )
    return p_max, best


def _size_shell_solution(
    *,
    family_k: int,
    shell_index: int,
    altitude_km: float,
    raw_ltan_start_h: float,
    raw_ltan_end_h: float,
    constraints: WalkerSizingConstraints,
    orbit_model: OrbitModel,
    band_method: str,
    lower_binding_true_anomaly_deg: float | None = None,
    upper_binding_true_anomaly_deg: float | None = None,
) -> ShellSolution:
    centered_start_h, centered_end_h = centered_ltan_subband_h(
        raw_ltan_start_h,
        raw_ltan_end_h,
        constraints.centered_ltan_h,
    )
    raan_span_deg = 15.0 * (centered_end_h - centered_start_h)
    raan_span_rad = math.radians(raan_span_deg)
    semimajor_axis_km = orbit_model.equatorial_radius_km + altitude_km
    inclination_rad = orbit_model.sun_sync_inclination_rad(semimajor_axis_km)
    p_max, candidate = size_walker_shell(
        semimajor_axis_km,
        inclination_rad,
        raan_span_rad,
        constraints,
    )
    closest = candidate.closest_approach
    shell = ShellSolution(
        family_k=family_k,
        shell_index=shell_index,
        altitude_km=altitude_km,
        semimajor_axis_km=semimajor_axis_km,
        inclination_deg=math.degrees(inclination_rad),
        raw_ltan_start_h=raw_ltan_start_h,
        raw_ltan_end_h=raw_ltan_end_h,
        centered_ltan_start_h=centered_start_h,
        centered_ltan_end_h=centered_end_h,
        raan_span_deg=raan_span_deg,
        maximum_rings=p_max,
        rings=candidate.rings,
        satellites_per_ring=candidate.satellites_per_ring,
        phasing=candidate.phasing,
        satellite_count=candidate.satellite_count,
        same_ring_spacing_km=candidate.same_ring_spacing_km,
        minimum_inter_ring_spacing_km=(
            None if closest is None else closest.distance_km
        ),
        limiting_ring_index_separation=(
            None if closest is None else closest.ring_index_separation
        ),
        limiting_relative_satellite_index=(
            None if closest is None else closest.relative_satellite_index
        ),
        limiting_raan_separation_deg=(
            None if closest is None else closest.raan_separation_deg
        ),
        limiting_phase_difference_deg=(
            None if closest is None else closest.phase_difference_deg
        ),
        band_method=band_method,
        lower_binding_true_anomaly_deg=lower_binding_true_anomaly_deg,
        upper_binding_true_anomaly_deg=upper_binding_true_anomaly_deg,
    )
    logger.info(
        "K%d shell %02d altitude %.1f km: P=%d S=%d F=%d, "
        "same-ring=%.6f km, inter-ring=%s km",
        shell.family_k,
        shell.shell_index,
        shell.altitude_km,
        shell.rings,
        shell.satellites_per_ring,
        shell.phasing,
        shell.same_ring_spacing_km,
        (
            "n/a"
            if shell.minimum_inter_ring_spacing_km is None
            else f"{shell.minimum_inter_ring_spacing_km:.6f}"
        ),
    )
    return shell


def generate_walker_constellation(
    family_specs: Sequence[FamilySpec] = DEFAULT_FAMILY_SPECS,
    constraints: WalkerSizingConstraints = WalkerSizingConstraints(),
    orbit_model: OrbitModel = MANUSCRIPT_V1_ORBIT_MODEL,
) -> WalkerConstellation:
    """Size every shell independently and return a complete constellation."""
    specs = tuple(family_specs)
    if not specs:
        raise ValueError("family_specs must contain at least one family")
    family_ids = [spec.k_orbits_per_sol for spec in specs]
    if len(set(family_ids)) != len(family_ids):
        raise ValueError(f"family_specs contains duplicate K values: {family_ids}")

    shells: list[ShellSolution] = []
    for family in specs:
        for shell_index in range(constraints.shell_count):
            altitude_km = (
                family.base_altitude_km + shell_index * constraints.shell_spacing_km
            )
            shells.append(
                _size_shell_solution(
                family_k=family.k_orbits_per_sol,
                shell_index=shell_index,
                altitude_km=altitude_km,
                raw_ltan_start_h=family.eclipse_free_ltan_start_h,
                raw_ltan_end_h=family.eclipse_free_ltan_end_h,
                constraints=constraints,
                orbit_model=orbit_model,
                band_method="fixed-family-band",
                )
            )

    constellation = WalkerConstellation(
        orbit_model=orbit_model,
        constraints=constraints,
        family_specs=specs,
        shells=tuple(shells),
    )
    logger.info(
        "Generated %d shells, %d rings, %d satellites with profile %s",
        len(constellation.shells),
        constellation.ring_count,
        constellation.satellite_count,
        orbit_model.name,
    )
    return constellation


def generate_walker_constellation_from_shell_specs(
    shell_specs: Sequence[ExplicitShellSpec],
    *,
    family_specs: Sequence[FamilySpec] = DEFAULT_FAMILY_SPECS,
    constraints: WalkerSizingConstraints = WalkerSizingConstraints(),
    orbit_model: OrbitModel = MANUSCRIPT_V1_ORBIT_MODEL,
    shell_layout_name: str = "explicit-shell-grid",
) -> WalkerConstellation:
    """Size an arbitrary altitude grid whose eclipse band is known per shell."""
    explicit_specs = tuple(sorted(shell_specs, key=lambda spec: spec.altitude_km))
    if not explicit_specs:
        raise ValueError("shell_specs must contain at least one shell")
    if not shell_layout_name:
        raise ValueError("shell_layout_name must be non-empty")
    identifiers = [
        (spec.reference_family_k, spec.shell_index) for spec in explicit_specs
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"shell_specs contains duplicate identifiers: {identifiers}")
    altitudes = [spec.altitude_km for spec in explicit_specs]
    if len(set(altitudes)) != len(altitudes):
        raise ValueError("shell_specs contains duplicate altitude centers")

    reference_specs = tuple(family_specs)
    reference_family_ids = {spec.k_orbits_per_sol for spec in reference_specs}
    unknown = sorted(
        {spec.reference_family_k for spec in explicit_specs} - reference_family_ids
    )
    if unknown:
        raise ValueError(f"shell_specs references unknown family K values: {unknown}")

    shells = tuple(
        _size_shell_solution(
            family_k=spec.reference_family_k,
            shell_index=spec.shell_index,
            altitude_km=spec.altitude_km,
            raw_ltan_start_h=spec.eclipse_free_ltan_start_h,
            raw_ltan_end_h=spec.eclipse_free_ltan_end_h,
            constraints=constraints,
            orbit_model=orbit_model,
            band_method=spec.band_method,
            lower_binding_true_anomaly_deg=spec.lower_binding_true_anomaly_deg,
            upper_binding_true_anomaly_deg=spec.upper_binding_true_anomaly_deg,
        )
        for spec in explicit_specs
    )
    constellation = WalkerConstellation(
        orbit_model=orbit_model,
        constraints=constraints,
        family_specs=reference_specs,
        shells=shells,
        shell_layout_name=shell_layout_name,
        explicit_shell_specs=explicit_specs,
    )
    logger.info(
        "Generated %d explicit shells, %d rings, %d satellites with profile %s",
        len(constellation.shells),
        constellation.ring_count,
        constellation.satellite_count,
        orbit_model.name,
    )
    return constellation


def _family_summary(constellation: WalkerConstellation, family_k: int) -> dict[str, object]:
    shells = constellation.family_shells(family_k)
    return {
        "family_k": family_k,
        "shell_count": len(shells),
        "ring_count": sum(shell.rings for shell in shells),
        "satellite_count": sum(shell.satellite_count for shell in shells),
        "altitude_range_km": [
            min(shell.altitude_km for shell in shells),
            max(shell.altitude_km for shell in shells),
        ],
        "rings_per_shell_range": [
            min(shell.rings for shell in shells),
            max(shell.rings for shell in shells),
        ],
        "satellites_per_ring_range": [
            min(shell.satellites_per_ring for shell in shells),
            max(shell.satellites_per_ring for shell in shells),
        ],
        "phasing_values": sorted({shell.phasing for shell in shells}),
    }


def constellation_manifest(constellation: WalkerConstellation) -> dict[str, object]:
    """Return a JSON-serializable, deterministic manifest."""
    populated_families = {shell.family_k for shell in constellation.shells}
    family_order = [
        spec.k_orbits_per_sol
        for spec in constellation.family_specs
        if spec.k_orbits_per_sol in populated_families
    ]
    return {
        "schema_version": 2,
        "model": "equal-altitude circular Walker-style geometric packing",
        "design_specification": "frozen Mars OSR v1 constellation design",
        "orbit_model": asdict(constellation.orbit_model),
        "constraints": asdict(constellation.constraints),
        "family_specs": [asdict(spec) for spec in constellation.family_specs],
        "shell_layout": {
            "name": constellation.shell_layout_name,
            "explicit_shell_specs": [
                asdict(spec) for spec in constellation.explicit_shell_specs
            ],
        },
        "summary": {
            "shell_count": len(constellation.shells),
            "ring_count": constellation.ring_count,
            "satellite_count": constellation.satellite_count,
            "families": [
                _family_summary(constellation, family_k) for family_k in family_order
            ],
        },
        "shells": [asdict(shell) for shell in constellation.shells],
        "limitations": [
            "Circular equal-altitude geometry only.",
            "No covariance, navigation uncertainty, orbit-keeping bands, "
            "failures, or maneuver margins.",
            "No persistent relative phasing constraint between different altitude shells.",
            "LTAN is recorded, but epoch-specific inertial RAAN is intentionally not invented.",
        ],
    }


def _ltan_grid_h(shell: ShellSolution) -> np.ndarray:
    if shell.rings == 1:
        return np.array(
            [0.5 * (shell.centered_ltan_start_h + shell.centered_ltan_end_h)]
        )
    return np.linspace(
        shell.centered_ltan_start_h,
        shell.centered_ltan_end_h,
        shell.rings,
    )


def iter_ring_records(constellation: WalkerConstellation) -> Iterator[dict[str, object]]:
    """Yield one deterministic tabular record per ring."""
    global_ring_index = 0
    for shell in constellation.shells:
        ltan_grid_h = _ltan_grid_h(shell)
        for ring_index, ltan_h in enumerate(ltan_grid_h):
            phase_offset_deg = (
                360.0
                * shell.phasing
                * ring_index
                / (shell.rings * shell.satellites_per_ring)
            )
            yield {
                "ring_id": f"K{shell.family_k}-H{shell.shell_index:02d}-P{ring_index:02d}",
                "global_ring_index": global_ring_index,
                "family_k": shell.family_k,
                "shell_index": shell.shell_index,
                "ring_index": ring_index,
                "altitude_km": shell.altitude_km,
                "semimajor_axis_km": shell.semimajor_axis_km,
                "inclination_deg": shell.inclination_deg,
                "ltan_h": float(ltan_h),
                "raan_offset_from_first_ring_deg": 15.0 * (float(ltan_h) - ltan_grid_h[0]),
                "satellites_per_ring": shell.satellites_per_ring,
                "walker_phasing": shell.phasing,
                "equivalent_satellite_phase_offset_deg": phase_offset_deg,
            }
            global_ring_index += 1


def iter_satellite_records(
    constellation: WalkerConstellation,
    *,
    initial_argument_of_latitude_offset_deg: float = 0.0,
) -> Iterator[dict[str, object]]:
    """Yield one circular-orbit element record per reflector.

    For circular orbits, v1 sets initial mean anomaly equal to argument of
    latitude.  Absolute inertial RAAN requires an epoch and is therefore left
    to ``reflectors.sun_sync.raan_mme2000_from_ltan`` at propagation time.
    """
    if not math.isfinite(initial_argument_of_latitude_offset_deg):
        raise ValueError("initial_argument_of_latitude_offset_deg must be finite")
    global_satellite_index = 0
    for shell in constellation.shells:
        ltan_grid_h = _ltan_grid_h(shell)
        for ring_index, ltan_h in enumerate(ltan_grid_h):
            raan_offset_deg = 15.0 * (float(ltan_h) - ltan_grid_h[0])
            equivalent_phase_offset_deg = (
                360.0
                * shell.phasing
                * ring_index
                / (shell.rings * shell.satellites_per_ring)
            )
            for slot_index in range(shell.satellites_per_ring):
                argument_of_latitude_deg = (
                    initial_argument_of_latitude_offset_deg
                    + 360.0 * slot_index / shell.satellites_per_ring
                    + equivalent_phase_offset_deg
                ) % 360.0
                yield {
                    "satellite_id": (
                        f"K{shell.family_k}-H{shell.shell_index:02d}-"
                        f"P{ring_index:02d}-N{slot_index:03d}"
                    ),
                    "global_satellite_index": global_satellite_index,
                    "family_k": shell.family_k,
                    "shell_index": shell.shell_index,
                    "ring_index": ring_index,
                    "slot_index": slot_index,
                    "altitude_km": shell.altitude_km,
                    "semimajor_axis_km": shell.semimajor_axis_km,
                    "eccentricity": 0.0,
                    "inclination_deg": shell.inclination_deg,
                    "ltan_h": float(ltan_h),
                    "raan_offset_from_first_ring_deg": raan_offset_deg,
                    "argument_of_periapsis_deg": 0.0,
                    "initial_argument_of_latitude_deg": argument_of_latitude_deg,
                    "initial_mean_anomaly_deg": argument_of_latitude_deg,
                    "walker_phasing": shell.phasing,
                }
                global_satellite_index += 1


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        raise ValueError(f"cannot write empty CSV product {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def write_constellation_products(
    constellation: WalkerConstellation,
    output_directory: str | Path,
    *,
    include_satellites: bool = True,
    initial_argument_of_latitude_offset_deg: float = 0.0,
    overwrite: bool = False,
    manifest_metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Write manifest, shell, ring, and optionally satellite products."""
    if not math.isfinite(initial_argument_of_latitude_offset_deg):
        raise ValueError("initial_argument_of_latitude_offset_deg must be finite")
    output_dir = Path(output_directory)
    products = {
        "manifest": output_dir / "manifest.json",
        "shells": output_dir / "shells.csv",
        "rings": output_dir / "rings.csv",
    }
    if include_satellites:
        products["satellites"] = output_dir / "satellites.csv"

    existing = [path for path in products.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing constellation products: {joined}; "
            "pass overwrite=True (CLI: --overwrite) to replace them"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = constellation_manifest(constellation)
    manifest["products"] = {
        key: path.name for key, path in products.items()
    }
    manifest["initial_argument_of_latitude_offset_deg"] = (
        initial_argument_of_latitude_offset_deg
    )
    if manifest_metadata is not None:
        manifest["generation"] = manifest_metadata
    products["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        products["shells"],
        (asdict(shell) for shell in constellation.shells),
    )
    _write_csv(products["rings"], iter_ring_records(constellation))
    if include_satellites:
        _write_csv(
            products["satellites"],
            iter_satellite_records(
                constellation,
                initial_argument_of_latitude_offset_deg=(
                    initial_argument_of_latitude_offset_deg
                ),
            ),
        )
    logger.info("Wrote constellation products under %s", output_dir)
    return products
