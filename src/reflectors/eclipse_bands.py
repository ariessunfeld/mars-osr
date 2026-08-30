"""Whole-Mars-year eclipse-free LTAN bands for circular Mars orbits.

This module implements the numerical eclipse-band calculation used by the
public tools:

1. sample seasons uniformly in *time* over one Mars sidereal year;
2. at each season, initialize a circular orbit at each trial effective LTAN;
3. propagate 1.05 orbital periods with degree-2 zonal gravity plus the Sun as a
   third body, with no sail force;
4. call an LTAN eclipse-free when none of the 20-s output samples lies in the
   binary Mars umbra;
5. bracket and bisect the two seasonal band edges; and
6. shift every seasonal interval by the Mars equation of center before taking
   their intersection at the reference (perihelion) epoch.

The final shift represents a station-kept node that tracks the *mean* Sun while
the true Sun advances non-uniformly around Mars's eccentric heliocentric orbit.
Consequently these are station-kept bands.  They are not valid for an
uncontrolled orbit whose node drifts away from the mean-Sun condition.

The shadow predicate is the apparent-disc total-eclipse construction of
Montenbruck & Gill (2000), *Satellite Orbits*, Sec. 3.4, pp. 77--82, implemented
in :mod:`reflectors.shadow`.  The first-order sun-synchronous inclination used
by callers follows Brouwer (1959), *Astronomical Journal* 64, 378--397.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from reflectors.dynamics import (
    PropagationOptions,
    propagate,
    sun_gm_km3_per_s2,
)
from reflectors.elements import (
    mme2000_rotation_from_j2000,
    state_from_classical_mme2000,
)
from reflectors.ephemeris import body_state, sun_state_j2000
from reflectors.gravity import mars_gravity_model
from reflectors.mars_constants import MARS_SIDEREAL_YEAR_S
from reflectors.shadow import in_mars_umbra
from reflectors.sun_sync import raan_mme2000_from_ltan
from reflectors.termination import AltitudeFloor
from reflectors.third_body import sun_third_body


REFERENCE_EPOCH_UTC = "2028-02-11T12:42:00"
ECLIPSE_BAND_METHOD_NAME = "j2-sun-sampled-umbra-v1"


@dataclass(frozen=True)
class EclipseBandSearchConfig:
    """Numerical settings for the versioned eclipse-band search."""

    season_count: int = 60
    ltan_scan_start_h: float = 13.0
    ltan_scan_end_h: float = 23.0
    ltan_scan_step_h: float = 0.5
    edge_tolerance_h: float = 0.01
    maximum_bisection_iterations: int = 26
    orbit_duration_factor: float = 1.05
    output_cadence_s: float = 20.0
    reference_ltan_h: float = 18.0
    altitude_floor_km: float = 300.0
    propagation_method: str = "DOP853"
    propagation_rtol: float = 1.0e-9
    propagation_atol: float = 1.0e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.season_count, bool)
            or int(self.season_count) != self.season_count
            or self.season_count < 4
        ):
            raise ValueError("season_count must be an integer >= 4")
        if not (
            math.isfinite(self.ltan_scan_start_h)
            and math.isfinite(self.ltan_scan_end_h)
            and 0.0 <= self.ltan_scan_start_h < self.ltan_scan_end_h <= 24.0
        ):
            raise ValueError(
                "LTAN scan endpoints must satisfy 0 <= start < end <= 24 h"
            )
        positive = {
            "ltan_scan_step_h": self.ltan_scan_step_h,
            "edge_tolerance_h": self.edge_tolerance_h,
            "orbit_duration_factor": self.orbit_duration_factor,
            "output_cadence_s": self.output_cadence_s,
            "altitude_floor_km": self.altitude_floor_km,
            "propagation_rtol": self.propagation_rtol,
            "propagation_atol": self.propagation_atol,
        }
        for field_name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and > 0; got {value!r}")
        if (
            isinstance(self.maximum_bisection_iterations, bool)
            or int(self.maximum_bisection_iterations)
            != self.maximum_bisection_iterations
            or self.maximum_bisection_iterations < 1
        ):
            raise ValueError("maximum_bisection_iterations must be an integer >= 1")
        if not math.isfinite(self.reference_ltan_h) or not (
            0.0 <= self.reference_ltan_h <= 24.0
        ):
            raise ValueError("reference_ltan_h must be finite and in [0, 24]")
        if not self.propagation_method:
            raise ValueError("propagation_method must be non-empty")

    def propagation_options(self) -> PropagationOptions:
        return PropagationOptions(
            method=self.propagation_method,
            rtol=self.propagation_rtol,
            atol=self.propagation_atol,
        )

    def manifest_record(self) -> dict[str, object]:
        record = asdict(self)
        record["method_name"] = ECLIPSE_BAND_METHOD_NAME
        record["force_model"] = "central + MRO120F J2 + Sun third body; no SRP"
        record["shadow_model"] = "binary spherical-Mars umbra; sampled at output cadence"
        record["season_grid"] = "uniform in ephemeris time over one Mars sidereal year"
        record["stationkeeping_assumption"] = (
            "node is held fixed relative to the mean Sun; equation-of-center shift "
            "accounts for the true Sun"
        )
        return record


@dataclass(frozen=True)
class EclipseSeason:
    """One season shared by every altitude in a band campaign."""

    season_index: int
    epoch_et: float
    true_anomaly_deg: float
    equation_of_center_offset_h: float


@dataclass(frozen=True)
class SeasonalEclipseBand:
    """Effective- and reference-epoch LTAN intervals at one season."""

    season_index: int
    epoch_et: float
    true_anomaly_deg: float
    equation_of_center_offset_h: float
    effective_ltan_start_h: float
    effective_ltan_end_h: float
    reference_ltan_start_h: float
    reference_ltan_end_h: float


@dataclass(frozen=True)
class EclipseBandResult:
    """Whole-year interval and the seasonal rows from which it was intersected."""

    semimajor_axis_km: float
    inclination_deg: float
    orbital_period_s: float
    ltan_start_h: float
    ltan_end_h: float
    lower_binding_season_index: int
    upper_binding_season_index: int
    lower_binding_true_anomaly_deg: float
    upper_binding_true_anomaly_deg: float
    seasons: tuple[SeasonalEclipseBand, ...]

    @property
    def width_h(self) -> float:
        return self.ltan_end_h - self.ltan_start_h


def keplerian_period_s(semimajor_axis_km: float, mu_km3_s2: float) -> float:
    """Circular two-body period, seconds."""
    if not math.isfinite(semimajor_axis_km) or semimajor_axis_km <= 0.0:
        raise ValueError("semimajor_axis_km must be finite and > 0")
    if not math.isfinite(mu_km3_s2) or mu_km3_s2 <= 0.0:
        raise ValueError("mu_km3_s2 must be finite and > 0")
    return 2.0 * math.pi * math.sqrt(semimajor_axis_km**3 / mu_km3_s2)


def sun_right_ascension_mme2000_rad(epoch_et: float) -> float:
    """Instantaneous Mars-to-Sun right ascension in MME2000."""
    sun_position_j2000_km = np.asarray(
        sun_state_j2000(epoch_et, "MARS")[:3],
        dtype=float,
    )
    sun_position_mme2000_km = (
        mme2000_rotation_from_j2000(epoch_et) @ sun_position_j2000_km
    )
    return float(
        math.atan2(sun_position_mme2000_km[1], sun_position_mme2000_km[0])
    )


def equation_of_center_ltan_offset_h(
    epoch_et: float,
    *,
    perihelion_epoch_et: float,
    reference_raan_rad: float,
    reference_ltan_h: float = 18.0,
) -> float:
    """Effective-true-Sun LTAN minus the perihelion mean-Sun setpoint.

    The station-kept node advances uniformly by ``2*pi/T_Mars-year``.  Its
    angular difference from the instantaneous true Sun is converted to hours;
    the wrapped reference value is then subtracted.
    """
    mean_sun_raan_rad = reference_raan_rad + (
        2.0
        * math.pi
        * (epoch_et - perihelion_epoch_et)
        / MARS_SIDEREAL_YEAR_S
    )
    effective_ltan_h = (
        12.0
        + math.degrees(
            mean_sun_raan_rad - sun_right_ascension_mme2000_rad(epoch_et)
        )
        / 15.0
    ) % 24.0
    return effective_ltan_h - reference_ltan_h


def mars_true_anomaly_deg(epoch_et: float) -> float:
    """Osculating Mars heliocentric true anomaly from the SPICE state."""
    state, _ = body_state(
        "MARS",
        epoch_et,
        frame="J2000",
        abcorr="NONE",
        observer="SUN",
    )
    position_km = np.asarray(state[:3], dtype=float)
    velocity_km_s = np.asarray(state[3:], dtype=float)
    angular_momentum = np.cross(position_km, velocity_km_s)
    eccentricity_vector = (
        np.cross(velocity_km_s, angular_momentum) / sun_gm_km3_per_s2()
        - position_km / np.linalg.norm(position_km)
    )
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    cosine_true_anomaly = float(
        np.clip(
            np.dot(eccentricity_vector, position_km)
            / (eccentricity * np.linalg.norm(position_km)),
            -1.0,
            1.0,
        )
    )
    true_anomaly_rad = math.acos(cosine_true_anomaly)
    if np.dot(position_km, velocity_km_s) < 0.0:
        true_anomaly_rad = 2.0 * math.pi - true_anomaly_rad
    return math.degrees(true_anomaly_rad)


def prepare_eclipse_seasons(
    perihelion_epoch_et: float,
    config: EclipseBandSearchConfig = EclipseBandSearchConfig(),
) -> tuple[EclipseSeason, ...]:
    """Build the shared uniform-time season grid and equation-of-center shifts."""
    reference_raan_rad = raan_mme2000_from_ltan(
        config.reference_ltan_h,
        perihelion_epoch_et,
    )
    epochs_et = perihelion_epoch_et + np.linspace(
        0.0,
        MARS_SIDEREAL_YEAR_S,
        config.season_count,
        endpoint=False,
    )
    return tuple(
        EclipseSeason(
            season_index=index,
            epoch_et=float(epoch_et),
            true_anomaly_deg=mars_true_anomaly_deg(float(epoch_et)),
            equation_of_center_offset_h=equation_of_center_ltan_offset_h(
                float(epoch_et),
                perihelion_epoch_et=perihelion_epoch_et,
                reference_raan_rad=reference_raan_rad,
                reference_ltan_h=config.reference_ltan_h,
            ),
        )
        for index, epoch_et in enumerate(epochs_et)
    )


def sampled_orbit_is_eclipse_free(
    semimajor_axis_km: float,
    inclination_deg: float,
    effective_ltan_h: float,
    epoch_et: float,
    orbital_period_s: float,
    *,
    config: EclipseBandSearchConfig = EclipseBandSearchConfig(),
    mu_km3_s2: float | None = None,
) -> bool:
    """Return whether the sampled one-orbit method sees no umbra."""
    if mu_km3_s2 is None:
        mu_km3_s2 = float(mars_gravity_model(max_degree=2).mu_km3_s2)
    if not math.isfinite(orbital_period_s) or orbital_period_s <= 0.0:
        raise ValueError("orbital_period_s must be finite and > 0")
    raan_rad = raan_mme2000_from_ltan(effective_ltan_h, epoch_et)
    initial_state = state_from_classical_mme2000(
        a_km=semimajor_axis_km,
        e=0.0,
        inclination_rad=math.radians(inclination_deg),
        raan_rad=raan_rad,
        argp_rad=0.0,
        nu_rad=0.0,
        mu_km3_s2=mu_km3_s2,
        epoch_et=epoch_et,
    )
    duration_s = config.orbit_duration_factor * orbital_period_s
    output_times_s = np.arange(0.0, duration_s, config.output_cadence_s)
    propagation = propagate(
        state0_km_kmps=initial_state,
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et,
        zonal_degree=2,
        gravity_degree=0,
        third_bodies=[sun_third_body()],
        solar_sail=None,
        sail_normal=None,
        altitude_floor=AltitudeFloor.at_km(config.altitude_floor_km),
        options=config.propagation_options(),
        t_eval_s=output_times_s,
    )
    return not any(
        in_mars_umbra(
            np.asarray(state[:3], dtype=float),
            epoch_et + float(time_s),
        )
        for time_s, state in zip(propagation.t_s, propagation.state_km_kmps)
    )


def seasonal_eclipse_free_ltan_band(
    semimajor_axis_km: float,
    inclination_deg: float,
    epoch_et: float,
    orbital_period_s: float,
    *,
    config: EclipseBandSearchConfig = EclipseBandSearchConfig(),
    mu_km3_s2: float | None = None,
) -> tuple[float, float]:
    """Find one season's sampled eclipse-free effective-LTAN interval."""
    scan = np.arange(
        config.ltan_scan_start_h,
        config.ltan_scan_end_h + 0.5 * config.ltan_scan_step_h,
        config.ltan_scan_step_h,
    )

    def is_free(ltan_h: float) -> bool:
        return sampled_orbit_is_eclipse_free(
            semimajor_axis_km,
            inclination_deg,
            ltan_h,
            epoch_et,
            orbital_period_s,
            config=config,
            mu_km3_s2=mu_km3_s2,
        )

    free = np.array([is_free(float(ltan_h)) for ltan_h in scan], dtype=bool)
    free_indices = np.flatnonzero(free)
    if free_indices.size == 0:
        raise RuntimeError(
            "no eclipse-free LTAN found inside scan interval "
            f"[{config.ltan_scan_start_h}, {config.ltan_scan_end_h}] h"
        )
    expected_contiguous = np.arange(free_indices[0], free_indices[-1] + 1)
    if not np.array_equal(free_indices, expected_contiguous):
        raise RuntimeError(
            "eclipse-free LTAN samples are not one contiguous interval; "
            f"indices={free_indices.tolist()}"
        )

    def bisect_edge(good_h: float, bad_h: float) -> float:
        for _ in range(config.maximum_bisection_iterations):
            if abs(bad_h - good_h) < config.edge_tolerance_h:
                break
            midpoint_h = 0.5 * (good_h + bad_h)
            if is_free(midpoint_h):
                good_h = midpoint_h
            else:
                bad_h = midpoint_h
        return good_h

    lower_index = int(free_indices[0])
    upper_index = int(free_indices[-1])
    lower_h = (
        float(scan[0])
        if lower_index == 0
        else bisect_edge(float(scan[lower_index]), float(scan[lower_index - 1]))
    )
    upper_h = (
        float(scan[-1])
        if upper_index == len(scan) - 1
        else bisect_edge(float(scan[upper_index]), float(scan[upper_index + 1]))
    )
    return lower_h, upper_h


def intersect_seasonal_eclipse_bands(
    seasonal_bands: Sequence[SeasonalEclipseBand],
) -> tuple[SeasonalEclipseBand, SeasonalEclipseBand]:
    """Return the rows that supply the lower and upper intersection edges."""
    rows = tuple(seasonal_bands)
    if not rows:
        raise ValueError("seasonal_bands must not be empty")
    lower_binding = max(rows, key=lambda row: row.reference_ltan_start_h)
    upper_binding = min(rows, key=lambda row: row.reference_ltan_end_h)
    if lower_binding.reference_ltan_start_h >= upper_binding.reference_ltan_end_h:
        raise RuntimeError(
            "seasonal LTAN intervals have an empty whole-year intersection: "
            f"lower={lower_binding.reference_ltan_start_h:.6f} h, "
            f"upper={upper_binding.reference_ltan_end_h:.6f} h"
        )
    return lower_binding, upper_binding


def whole_year_eclipse_free_ltan_band(
    semimajor_axis_km: float,
    inclination_deg: float,
    perihelion_epoch_et: float,
    *,
    orbital_period_s: float | None = None,
    seasons: Sequence[EclipseSeason] | None = None,
    config: EclipseBandSearchConfig = EclipseBandSearchConfig(),
    mu_km3_s2: float | None = None,
) -> EclipseBandResult:
    """Compute the station-kept whole-year LTAN intersection for one shell."""
    if mu_km3_s2 is None:
        mu_km3_s2 = float(mars_gravity_model(max_degree=2).mu_km3_s2)
    if orbital_period_s is None:
        orbital_period_s = keplerian_period_s(semimajor_axis_km, mu_km3_s2)
    season_rows = (
        prepare_eclipse_seasons(perihelion_epoch_et, config)
        if seasons is None
        else tuple(seasons)
    )
    if len(season_rows) != config.season_count:
        raise ValueError(
            f"expected {config.season_count} seasons, got {len(season_rows)}"
        )

    seasonal_bands: list[SeasonalEclipseBand] = []
    for season in season_rows:
        effective_start_h, effective_end_h = seasonal_eclipse_free_ltan_band(
            semimajor_axis_km,
            inclination_deg,
            season.epoch_et,
            orbital_period_s,
            config=config,
            mu_km3_s2=mu_km3_s2,
        )
        offset_h = season.equation_of_center_offset_h
        seasonal_bands.append(
            SeasonalEclipseBand(
                season_index=season.season_index,
                epoch_et=season.epoch_et,
                true_anomaly_deg=season.true_anomaly_deg,
                equation_of_center_offset_h=offset_h,
                effective_ltan_start_h=effective_start_h,
                effective_ltan_end_h=effective_end_h,
                reference_ltan_start_h=effective_start_h - offset_h,
                reference_ltan_end_h=effective_end_h - offset_h,
            )
        )

    lower_binding, upper_binding = intersect_seasonal_eclipse_bands(seasonal_bands)
    return EclipseBandResult(
        semimajor_axis_km=semimajor_axis_km,
        inclination_deg=inclination_deg,
        orbital_period_s=orbital_period_s,
        ltan_start_h=lower_binding.reference_ltan_start_h,
        ltan_end_h=upper_binding.reference_ltan_end_h,
        lower_binding_season_index=lower_binding.season_index,
        upper_binding_season_index=upper_binding.season_index,
        lower_binding_true_anomaly_deg=lower_binding.true_anomaly_deg,
        upper_binding_true_anomaly_deg=upper_binding.true_anomaly_deg,
        seasons=tuple(seasonal_bands),
    )
