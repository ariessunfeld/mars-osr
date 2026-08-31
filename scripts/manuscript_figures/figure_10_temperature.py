#!/usr/bin/env python3
"""Create the 600-layer annual and fixed-season temperature composite.

Panel A shows sol minimum, arithmetic mean, and maximum temperature versus
solar longitude. Panels B and C show local-time profiles at L_s=105 and
L_s=270 degrees, respectively.

The no-reflector and with-reflector thermal-model integrations are dashed and
solid continuous curves.  The independent reference GCM is shown as
unconnected x markers.  Its distinct five-model-day-cadence,
24-local-time-bin schema is ingested separately from the simulations'
96-samples-per-sol schema.  Each GCM marker comes from a representative
diurnal profile formed by averaging corresponding local-time bins over a
centered seasonal L_s window.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from netCDF4 import Dataset, Variable

from temperature_model_output import (
    DailyTemperatureStatistics,
    STATISTIC_COLORS,
    ModelOutput,
    load_model_output,
)


LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT: Final = (
    REPOSITORY_ROOT
    / "figures"
    / "manuscript"
    / "generated"
    / "figure_10_surface_temperature.png"
)
DEFAULT_MODEL_INPUT_DIRECTORY: Final = (
    REPOSITORY_ROOT
    / "simulation_outputs"
    / "manuscript_climate"
)
DEFAULT_GCM_INPUT: Final = (
    DEFAULT_MODEL_INPUT_DIRECTORY.parent / "GCM_diurnal_T_zavg_40N.nc"
)
DEFAULT_SAILS_INPUT: Final = (
    DEFAULT_MODEL_INPUT_DIRECTORY / "run_1p44_sail_comimart_600Lz.nc"
)
DEFAULT_NO_REFLECTOR_INPUT: Final = (
    DEFAULT_MODEL_INPUT_DIRECTORY / "run_no_sail_comimart_600Lz.nc"
)
REFERENCE_GCM_LOCAL_TIME_BINS: Final = 24
DEFAULT_GCM_WINDOW_WIDTH_DEG: Final = 10.0
TEMPERATURE_LOWER_LIMIT_K: Final = 155.0
TEMPERATURE_LOWEST_LABELED_TICK_K: Final = 160.0
TEMPERATURE_UPPER_LIMIT_K: Final = 310.0
TEMPERATURE_MINOR_TICK_INTERVAL_K: Final = 2.0
DIURNAL_TEMPERATURE_MINOR_TICK_INTERVAL_K: Final = 5.0
DIURNAL_LOCAL_TIME_MINOR_TICK_INTERVAL_H: Final = 2.0
GCM_TO_SIMULATION_LS_TOLERANCE_DEG: Final = 0.05
DIURNAL_PANEL_SOLAR_LONGITUDES_DEG: Final = (105.0, 270.0)
WATER_FREEZING_TEMPERATURE_K: Final = 273.0
PANEL_LABEL_UPWARD_SHIFT_K: Final = 4.0
DIURNAL_SEASON_LABEL_UPWARD_SHIFT_K: Final = 2.0
PANEL_A_STATISTIC_LEGEND_RIGHT_EDGE_DEG: Final = 270.0
PANEL_A_SIMULATION_LEGEND_LEFT_SHIFT_DEG: Final = 10.0
PANEL_A_SIMULATION_LEGEND_UPWARD_SHIFT_K: Final = 10.0

StatisticName = Literal["minimum", "mean", "maximum"]


@dataclass(frozen=True)
class SeasonalPanelSelection:
    """The nearest continuous-simulation sol for a requested season."""

    target_solar_longitude_deg: float
    sol_index: int
    simulation_solar_longitude_deg: float
    simulation_offset_deg: float


@dataclass(frozen=True)
class ReferenceGCMOutput:
    """Validated seasonal statistics and local-time profiles from the GCM."""

    path: Path
    sha256: str
    statistics: DailyTemperatureStatistics
    samples_per_sol: int
    temperature_blocks_K: np.ndarray
    local_time_blocks_h: np.ndarray
    latitude_deg: float
    longitude_deg: float
    time_days: np.ndarray
    profile_interval_days: float
    description: str


@dataclass(frozen=True)
class ReferenceGCMSeasonalComposite:
    """Mean diurnal GCM profiles within centered seasonal L_s windows."""

    statistics: DailyTemperatureStatistics
    samples_per_sol: int
    temperature_blocks_K: np.ndarray
    local_time_blocks_h: np.ndarray
    window_width_deg: float
    window_counts: np.ndarray


TemperatureOutput = ModelOutput | ReferenceGCMOutput | ReferenceGCMSeasonalComposite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcm-input",
        type=Path,
        default=DEFAULT_GCM_INPUT,
        help="Independent reference-GCM NetCDF.",
    )
    parser.add_argument(
        "--sails-input",
        type=Path,
        default=DEFAULT_SAILS_INPUT,
        help="With-reflectors NetCDF supplied by the climate model.",
    )
    parser.add_argument(
        "--no-sails-input",
        type=Path,
        default=DEFAULT_NO_REFLECTOR_INPUT,
        help="No-reflector thermal-model integration (dashed curves).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gcm-ls-step-deg",
        type=float,
        default=10.0,
        help="Panel-A GCM sampling interval (default: 10 degrees L_s).",
    )
    parser.add_argument(
        "--gcm-window-width-deg",
        type=float,
        default=DEFAULT_GCM_WINDOW_WIDTH_DEG,
        help=(
            "Width of each centered seasonal GCM composite in degrees L_s "
            "(default: 10 degrees)."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_unmasked(variable: Variable) -> np.ndarray:
    values = np.ma.asarray(variable[:])
    if np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{variable.name} contains masked values")
    result = np.asarray(np.ma.getdata(values), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{variable.name} contains non-finite values")
    return result


def _read_scalar(dataset: Dataset, name: str) -> float:
    values = _read_unmasked(dataset.variables[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must contain exactly one value, found {values.size}")
    return float(values[0])


def load_reference_gcm_output(path: Path) -> ReferenceGCMOutput:
    """Load the independent five-model-day-cadence, 24-bin reference GCM."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with Dataset(path, mode="r") as dataset:
        required_variables = {
            "areo",
            "lat",
            "lon",
            "time",
            "time_of_day_24",
            "ts",
        }
        missing = sorted(required_variables.difference(dataset.variables))
        if missing:
            raise ValueError(f"{path} is missing required GCM variables: {missing}")
        expected_dimensions = {
            "areo": ("time", "time_of_day_24", "scalar_axis"),
            "lat": ("lat",),
            "lon": ("lon",),
            "time": ("time",),
            "time_of_day_24": ("time_of_day_24",),
            "ts": ("time", "time_of_day_24", "lat", "lon"),
        }
        for name, dimensions in expected_dimensions.items():
            if dataset.variables[name].dimensions != dimensions:
                raise ValueError(
                    f"GCM {name} dimensions must be {dimensions}, found "
                    f"{dataset.variables[name].dimensions}"
                )
        temperature_units = str(getattr(dataset.variables["ts"], "units", ""))
        if temperature_units.strip().lower() != "k":
            raise ValueError(f"GCM ts units must be K, found {temperature_units!r}")
        solar_longitude_units = str(
            getattr(dataset.variables["areo"], "units", "")
        )
        if solar_longitude_units.strip().lower() not in {"deg", "degree", "degrees"}:
            raise ValueError(
                "GCM areo units must be degrees, found "
                f"{solar_longitude_units!r}"
            )
        local_time_units = str(
            getattr(dataset.variables["time_of_day_24"], "units", "")
        )
        if not local_time_units.strip().lower().startswith("hours since"):
            raise ValueError(
                "GCM time_of_day_24 units must be hours since an epoch, found "
                f"{local_time_units!r}"
            )
        time_units = str(getattr(dataset.variables["time"], "units", ""))
        if not time_units.strip().lower().startswith("days since"):
            raise ValueError(
                f"GCM time units must be days since an epoch, found {time_units!r}"
            )

        time_days = _read_unmasked(dataset.variables["time"]).reshape(-1)
        local_time_one_profile_h = _read_unmasked(
            dataset.variables["time_of_day_24"]
        ).reshape(-1)
        solar_longitude_blocks_deg = _read_unmasked(
            dataset.variables["areo"]
        ).reshape(time_days.size, local_time_one_profile_h.size)
        temperature_blocks_K = _read_unmasked(dataset.variables["ts"]).reshape(
            time_days.size, local_time_one_profile_h.size
        )
        latitude_deg = _read_scalar(dataset, "lat")
        longitude_deg = _read_scalar(dataset, "lon")
        description = str(getattr(dataset, "description", ""))

    if local_time_one_profile_h.size != REFERENCE_GCM_LOCAL_TIME_BINS:
        raise ValueError(
            "GCM must contain exactly 24 local-time bins, found "
            f"{local_time_one_profile_h.size}"
        )
    if time_days.size < 2:
        raise ValueError("GCM must contain at least two seasonal profiles")
    if not np.all(np.diff(time_days) > 0.0):
        raise ValueError("GCM time coordinate is not strictly increasing")
    time_steps_days = np.diff(time_days)
    if not np.allclose(
        time_steps_days,
        time_steps_days[0],
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("GCM seasonal profiles are not uniformly spaced in time")
    expected_local_time_step_h = 24.0 / REFERENCE_GCM_LOCAL_TIME_BINS
    cyclic_local_time_steps_h = np.diff(
        np.concatenate(
            (local_time_one_profile_h, local_time_one_profile_h[:1] + 24.0)
        )
    )
    if not np.allclose(
        cyclic_local_time_steps_h,
        expected_local_time_step_h,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("GCM local-time bins do not uniformly cover 24 hours")
    if local_time_one_profile_h[0] < 0.0 or local_time_one_profile_h[-1] >= 24.0:
        raise ValueError("GCM local-time bins must lie in [0, 24)")

    solar_longitude_unwrapped_deg = solar_longitude_blocks_deg.mean(axis=1)
    maximum_profile_ls_span_deg = float(
        np.max(np.ptp(solar_longitude_blocks_deg, axis=1))
    )
    if maximum_profile_ls_span_deg > 5.0:
        raise ValueError(
            "GCM areocentric longitude changes by more than 5 degrees within "
            f"one diurnal profile: {maximum_profile_ls_span_deg}"
        )
    if not np.all(np.diff(solar_longitude_unwrapped_deg) > 0.0):
        raise ValueError("GCM mean areocentric longitude is not strictly increasing")
    solar_longitude_deg = np.mod(solar_longitude_unwrapped_deg, 360.0)
    if not np.all(np.diff(solar_longitude_deg) > 0.0):
        raise ValueError(
            "GCM areocentric longitude does not reduce to one monotonic Mars year"
        )
    if solar_longitude_deg[0] > 5.0 or solar_longitude_deg[-1] < 355.0:
        raise ValueError(
            "GCM seasonal L_s grid does not cover a complete Mars year: "
            f"{solar_longitude_deg[0]}..{solar_longitude_deg[-1]}"
        )

    local_time_blocks_h = np.broadcast_to(
        local_time_one_profile_h,
        temperature_blocks_K.shape,
    ).copy()
    statistics = DailyTemperatureStatistics(
        solar_longitude_deg=solar_longitude_deg,
        minimum_K=temperature_blocks_K.min(axis=1),
        mean_K=temperature_blocks_K.mean(axis=1),
        maximum_K=temperature_blocks_K.max(axis=1),
    )
    result = ReferenceGCMOutput(
        path=path.resolve(),
        sha256=_sha256(path),
        statistics=statistics,
        samples_per_sol=REFERENCE_GCM_LOCAL_TIME_BINS,
        temperature_blocks_K=temperature_blocks_K,
        local_time_blocks_h=local_time_blocks_h,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        time_days=time_days,
        profile_interval_days=float(time_steps_days[0]),
        description=description,
    )
    LOGGER.info(
        "Loaded reference GCM %s: %d seasonal profiles x %d local-time bins "
        "at %.6f-day cadence (time %.6f..%.6f days); stored coordinate "
        "%.6f deg N, %.6f deg E; LT %.6f..%.6f h; L_s %.6f..%.6f deg; "
        "max within-profile delta L_s %.6f deg; Tsfc %.6f..%.6f K; "
        "description=%r; sha256=%s",
        path,
        result.statistics.n_sols,
        result.samples_per_sol,
        result.profile_interval_days,
        float(result.time_days[0]),
        float(result.time_days[-1]),
        result.latitude_deg,
        result.longitude_deg,
        float(result.local_time_blocks_h[0, 0]),
        float(result.local_time_blocks_h[0, -1]),
        result.statistics.solar_longitude_deg[0],
        result.statistics.solar_longitude_deg[-1],
        maximum_profile_ls_span_deg,
        float(np.min(result.temperature_blocks_K)),
        float(np.max(result.temperature_blocks_K)),
        result.description,
        result.sha256,
    )
    return result


def build_reference_gcm_seasonal_composite(
    gcm: ReferenceGCMOutput,
    target_solar_longitude_deg: np.ndarray,
    window_width_deg: float,
) -> ReferenceGCMSeasonalComposite:
    """Average each GCM local-time bin within centered seasonal windows."""

    targets_deg = np.asarray(target_solar_longitude_deg, dtype=np.float64)
    if targets_deg.ndim != 1 or targets_deg.size == 0:
        raise ValueError("GCM seasonal-composite targets must be a nonempty vector")
    if not np.all(np.isfinite(targets_deg)):
        raise ValueError("GCM seasonal-composite targets must be finite")
    if not math.isfinite(window_width_deg) or not 0.0 < window_width_deg <= 360.0:
        raise ValueError("--gcm-window-width-deg must lie in (0, 360]")

    half_width_deg = 0.5 * window_width_deg
    composite_profiles_K: list[np.ndarray] = []
    window_counts: list[int] = []
    for target_deg in targets_deg:
        signed_offsets_deg = (
            (gcm.statistics.solar_longitude_deg - target_deg + 180.0) % 360.0
        ) - 180.0
        in_window = (signed_offsets_deg >= -half_width_deg) & (
            signed_offsets_deg < half_width_deg
        )
        count = int(np.count_nonzero(in_window))
        if count == 0:
            raise ValueError(
                f"GCM seasonal window centered at L_s={target_deg} degrees is empty"
            )
        composite_profiles_K.append(gcm.temperature_blocks_K[in_window].mean(axis=0))
        window_counts.append(count)

    temperature_blocks_K = np.stack(composite_profiles_K)
    local_time_blocks_h = np.broadcast_to(
        gcm.local_time_blocks_h[0],
        temperature_blocks_K.shape,
    ).copy()
    statistics = DailyTemperatureStatistics(
        solar_longitude_deg=targets_deg.copy(),
        minimum_K=temperature_blocks_K.min(axis=1),
        mean_K=temperature_blocks_K.mean(axis=1),
        maximum_K=temperature_blocks_K.max(axis=1),
    )
    return ReferenceGCMSeasonalComposite(
        statistics=statistics,
        samples_per_sol=gcm.samples_per_sol,
        temperature_blocks_K=temperature_blocks_K,
        local_time_blocks_h=local_time_blocks_h,
        window_width_deg=window_width_deg,
        window_counts=np.asarray(window_counts, dtype=int),
    )


def validate_simulation_alignment(
    anchor: ModelOutput,
    other: ModelOutput,
    anchor_label: str,
    other_label: str,
) -> None:
    """Require matching physics parameters and time coordinates."""

    if anchor.model_parameters != other.model_parameters:
        raise ValueError(f"{anchor_label} and {other_label} model parameters differ")
    if anchor.samples_per_sol != other.samples_per_sol:
        raise ValueError(f"{anchor_label} and {other_label} samples-per-sol differ")
    if anchor.time_step_value != other.time_step_value:
        raise ValueError(f"{anchor_label} and {other_label} time steps differ")
    if not np.array_equal(
        anchor.statistics.solar_longitude_deg,
        other.statistics.solar_longitude_deg,
    ):
        raise ValueError(
            f"{anchor_label} and {other_label} daily solar-longitude grids differ"
        )
    if not np.array_equal(
        anchor.time_solar_longitude_deg,
        other.time_solar_longitude_deg,
    ):
        raise ValueError(
            f"{anchor_label} and {other_label} time-resolved solar-longitude grids differ"
        )
    if not np.array_equal(anchor.local_time_h, other.local_time_h):
        raise ValueError(f"{anchor_label} and {other_label} local-time grids differ")
    if anchor.description != other.description:
        LOGGER.warning(
            "%s and %s global descriptions differ (%r vs %r), but their explicit "
            "model parameters and coordinates match",
            anchor_label,
            other_label,
            anchor.description,
            other.description,
        )


def statistic_values(
    model: TemperatureOutput,
    statistic_name: StatisticName,
) -> np.ndarray:
    if statistic_name == "minimum":
        return model.statistics.minimum_K
    if statistic_name == "mean":
        return model.statistics.mean_K
    return model.statistics.maximum_K


def ascending_local_time_sol(
    model: TemperatureOutput,
    sol_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one canonical sol sorted onto its ascending cyclic LT coordinate."""

    local_time_h = model.local_time_blocks_h[sol_index]
    temperatures_K = model.temperature_blocks_K[sol_index]
    order = np.argsort(local_time_h, kind="stable")
    ordered_time_h = local_time_h[order]
    ordered_temperatures_K = temperatures_K[order]
    expected_step_h = 24.0 / model.samples_per_sol
    cyclic_time_steps_h = np.diff(
        np.concatenate((ordered_time_h, ordered_time_h[:1] + 24.0))
    )
    if not np.allclose(
        cyclic_time_steps_h,
        expected_step_h,
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise ValueError(
            "canonical sol does not contain the expected uniform 0--24 h local-time grid"
        )
    return ordered_time_h, ordered_temperatures_K


def nearest_model_indices(
    source_solar_longitude_deg: np.ndarray,
    target_solar_longitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each target L_s to the nearest source-model sol on a circular grid."""

    indices: list[int] = []
    offsets_deg: list[float] = []
    for target_deg in target_solar_longitude_deg:
        signed_offsets_deg = (
            (source_solar_longitude_deg - target_deg + 180.0) % 360.0
        ) - 180.0
        index = int(np.argmin(np.abs(signed_offsets_deg)))
        indices.append(index)
        offsets_deg.append(float(signed_offsets_deg[index]))
    return np.asarray(indices, dtype=int), np.asarray(offsets_deg, dtype=np.float64)


def select_seasonal_panels(
    simulation: ModelOutput,
    target_solar_longitudes_deg: Sequence[float],
) -> tuple[SeasonalPanelSelection, ...]:
    """Select the nearest simulation sol for each requested seasonal panel."""

    targets_deg = np.asarray(target_solar_longitudes_deg, dtype=np.float64)
    indices, offsets_deg = nearest_model_indices(
        simulation.statistics.solar_longitude_deg,
        targets_deg,
    )
    seasonal_steps_deg = np.diff(
        np.concatenate(
            (
                simulation.statistics.solar_longitude_deg,
                simulation.statistics.solar_longitude_deg[:1] + 360.0,
            )
        )
    )
    if np.any(seasonal_steps_deg <= 0.0):
        raise ValueError("simulation solar-longitude grid is not circularly increasing")
    maximum_nearest_offset_deg = 0.5 * float(np.max(seasonal_steps_deg))
    if np.any(np.abs(offsets_deg) > maximum_nearest_offset_deg + 1.0e-12):
        raise ValueError(
            "nearest seasonal-panel mapping exceeds half the largest simulation "
            f"L_s step ({maximum_nearest_offset_deg:.9f} degrees)"
        )
    return tuple(
        SeasonalPanelSelection(
            target_solar_longitude_deg=float(target_deg),
            sol_index=int(sol_index),
            simulation_solar_longitude_deg=float(
                simulation.statistics.solar_longitude_deg[sol_index]
            ),
            simulation_offset_deg=float(offset_deg),
        )
        for target_deg, sol_index, offset_deg in zip(
            targets_deg,
            indices,
            offsets_deg,
            strict=True,
        )
    )


def gcm_simulation_diurnal_alignment(
    gcm: ReferenceGCMOutput,
    simulation: ModelOutput,
    mapped_simulation_indices: np.ndarray,
) -> tuple[float, float]:
    """Compare explicit-LT GCM anomalies with nearest-season simulation profiles."""

    gcm_profiles_K: list[np.ndarray] = []
    simulation_profiles_K: list[np.ndarray] = []
    for gcm_profile_index, simulation_sol_index in enumerate(
        mapped_simulation_indices
    ):
        gcm_time_h, gcm_temperature_K = ascending_local_time_sol(
            gcm, gcm_profile_index
        )
        simulation_time_h, simulation_temperature_K = ascending_local_time_sol(
            simulation, int(simulation_sol_index)
        )
        simulation_indices = np.rint(
            gcm_time_h * simulation.samples_per_sol / 24.0
        ).astype(int) % simulation.samples_per_sol
        if not np.allclose(
            simulation_time_h[simulation_indices],
            gcm_time_h,
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("GCM hourly bins are absent from the simulation LT grid")
        gcm_profiles_K.append(gcm_temperature_K)
        simulation_profiles_K.append(simulation_temperature_K[simulation_indices])

    gcm_matrix_K = np.stack(gcm_profiles_K)
    simulation_matrix_K = np.stack(simulation_profiles_K)
    gcm_anomaly_K = gcm_matrix_K - gcm_matrix_K.mean(axis=1, keepdims=True)
    simulation_anomaly_K = simulation_matrix_K - simulation_matrix_K.mean(
        axis=1, keepdims=True
    )
    correlation = float(
        np.corrcoef(gcm_anomaly_K.reshape(-1), simulation_anomaly_K.reshape(-1))[0, 1]
    )
    anomaly_rmse_K = float(
        np.sqrt(np.mean((gcm_anomaly_K - simulation_anomaly_K) ** 2))
    )
    return correlation, anomaly_rmse_K


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5.5,
        width=0.9,
        labelsize=10.5,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.7,
    )
    axis.grid(False, which="both")


def add_water_freezing_reference(axis: plt.Axes) -> None:
    """Draw the requested unlabeled 273 K visual reference."""

    axis.axhline(
        WATER_FREEZING_TEMPERATURE_K,
        color="black",
        linewidth=1.1,
        linestyle=":",
        label="_nolegend_",
        zorder=1.5,
    )


def panel_label(axis: plt.Axes, text: str) -> None:
    temperature_span_K = float(np.diff(axis.get_ylim())[0])
    axis.text(
        0.015,
        0.965 + PANEL_LABEL_UPWARD_SHIFT_K / temperature_span_K,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )


def _plot_annual_panel(
    axis: plt.Axes,
    gcm: ReferenceGCMSeasonalComposite,
    sails: ModelOutput,
    no_sails: ModelOutput,
) -> None:
    for statistic_name in ("minimum", "mean", "maximum"):
        color = STATISTIC_COLORS[statistic_name]
        axis.fill_between(
            sails.statistics.solar_longitude_deg,
            statistic_values(no_sails, statistic_name),
            statistic_values(sails, statistic_name),
            color=color,
            alpha=0.35,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            sails.statistics.solar_longitude_deg,
            statistic_values(sails, statistic_name),
            color=color,
            linewidth=1.8,
            linestyle="-",
            zorder=2,
        )
        axis.plot(
            no_sails.statistics.solar_longitude_deg,
            statistic_values(no_sails, statistic_name),
            color=color,
            linewidth=1.45,
            linestyle="--",
            dashes=(5.0, 3.0),
            zorder=3,
        )
        axis.plot(
            gcm.statistics.solar_longitude_deg,
            statistic_values(gcm, statistic_name),
            color=color,
            marker="x",
            markersize=5.0,
            markeredgewidth=1.1,
            linestyle="None",
            zorder=4,
        )

    axis.set_xlim(0.0, 360.0)
    axis.set_ylim(TEMPERATURE_LOWER_LIMIT_K, TEMPERATURE_UPPER_LIMIT_K)
    axis.set_xticks(np.arange(0.0, 360.0 + 15.0, 15.0))
    axis.xaxis.set_minor_locator(MultipleLocator(5.0))
    axis.set_yticks(
        np.arange(
            TEMPERATURE_LOWEST_LABELED_TICK_K,
            TEMPERATURE_UPPER_LIMIT_K + 10.0,
            10.0,
        )
    )
    axis.yaxis.set_minor_locator(MultipleLocator(TEMPERATURE_MINOR_TICK_INTERVAL_K))
    axis.set_xlabel(r"Solar longitude, $L_s$ (degrees)")
    axis.set_ylabel("Surface temperature (K)")
    add_water_freezing_reference(axis)
    style_axis(axis)
    panel_label(axis, "A")

    statistic_handles = [
        Line2D([0], [0], color=STATISTIC_COLORS[name], linewidth=2.0, label=label)
        for name, label in (
            ("minimum", "Sol minimum"),
            ("mean", "Sol mean"),
            ("maximum", "Sol maximum"),
        )
    ]
    simulation_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="With reflectors"),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.45,
            linestyle="--",
            dashes=(5.0, 3.0),
            label="No reflectors",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            markersize=5.0,
            markeredgewidth=1.1,
            linestyle="None",
            label="Reference GCM",
        ),
    ]

    solar_longitude_span_deg = float(np.diff(axis.get_xlim())[0])
    temperature_span_K = float(np.diff(axis.get_ylim())[0])
    statistic_legend = axis.legend(
        handles=statistic_handles[::-1],
        loc="upper right",
        bbox_to_anchor=(
            PANEL_A_STATISTIC_LEGEND_RIGHT_EDGE_DEG
            / solar_longitude_span_deg,
            1.0 + PANEL_A_SIMULATION_LEGEND_UPWARD_SHIFT_K / temperature_span_K,
        ),
        frameon=False,
        handlelength=2.7,
        borderaxespad=0.35,
    )
    axis.add_artist(statistic_legend)
    simulation_legend = axis.legend(
        handles=simulation_handles,
        title="Simulation (all panels)",
        loc="upper right",
        bbox_to_anchor=(
            1.0
            - PANEL_A_SIMULATION_LEGEND_LEFT_SHIFT_DEG
            / solar_longitude_span_deg,
            1.0 + PANEL_A_SIMULATION_LEGEND_UPWARD_SHIFT_K / temperature_span_K,
        ),
        frameon=False,
        handlelength=2.7,
        borderaxespad=0.35,
    )
    simulation_legend.get_title().set_fontstyle("italic")


def _plot_diurnal_panel(
    axis: plt.Axes,
    selection: SeasonalPanelSelection,
    gcm_sol_index: int,
    panel_text: str,
    gcm: ReferenceGCMSeasonalComposite,
    sails: ModelOutput,
    no_sails: ModelOutput,
) -> None:
    sails_time_h, sails_temperature_K = ascending_local_time_sol(
        sails, selection.sol_index
    )
    no_sails_time_h, no_sails_temperature_K = ascending_local_time_sol(
        no_sails, selection.sol_index
    )
    if not np.array_equal(sails_time_h, no_sails_time_h):
        raise ValueError("continuous simulations differ on selected local-time grid")
    gcm_time_h, gcm_temperature_K = ascending_local_time_sol(gcm, gcm_sol_index)

    axis.plot(
        sails_time_h,
        sails_temperature_K,
        color="black",
        linewidth=1.9,
        linestyle="-",
        zorder=3,
    )
    axis.plot(
        no_sails_time_h,
        no_sails_temperature_K,
        color="black",
        linewidth=1.55,
        linestyle="--",
        dashes=(5.0, 3.0),
        zorder=2,
    )
    axis.plot(
        gcm_time_h,
        gcm_temperature_K,
        color="black",
        marker="x",
        markersize=4.2,
        markeredgewidth=0.9,
        linestyle="None",
        zorder=4,
    )

    axis.set_xlim(0.0, 24.0)
    axis.set_ylim(TEMPERATURE_LOWER_LIMIT_K, TEMPERATURE_UPPER_LIMIT_K)
    axis.xaxis.set_major_locator(MultipleLocator(4.0))
    axis.xaxis.set_minor_locator(
        MultipleLocator(DIURNAL_LOCAL_TIME_MINOR_TICK_INTERVAL_H)
    )
    axis.yaxis.set_major_locator(MultipleLocator(10.0))
    axis.yaxis.set_minor_locator(
        MultipleLocator(DIURNAL_TEMPERATURE_MINOR_TICK_INTERVAL_K)
    )
    add_water_freezing_reference(axis)
    style_axis(axis)
    panel_label(axis, panel_text)
    axis.text(
        0.09,
        0.965
        + DIURNAL_SEASON_LABEL_UPWARD_SHIFT_K / float(np.diff(axis.get_ylim())[0]),
        rf"$L_s={selection.target_solar_longitude_deg:g}^\circ$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
    )


def _validate_temperature_limits(
    models: Sequence[tuple[str, TemperatureOutput]],
) -> None:
    for label, model in models:
        minimum_K = float(np.min(model.temperature_blocks_K))
        maximum_K = float(np.max(model.temperature_blocks_K))
        if minimum_K < TEMPERATURE_LOWER_LIMIT_K or maximum_K > TEMPERATURE_UPPER_LIMIT_K:
            raise ValueError(
                f"{label} temperature range {minimum_K:.6f}..{maximum_K:.6f} K "
                f"falls outside fixed figure limits {TEMPERATURE_LOWER_LIMIT_K:.0f}.."
                f"{TEMPERATURE_UPPER_LIMIT_K:.0f} K"
            )


def make_figure(
    annual_gcm: ReferenceGCMSeasonalComposite,
    seasonal_panel_gcm: ReferenceGCMSeasonalComposite,
    sails: ModelOutput,
    no_sails: ModelOutput,
    panel_b_selection: SeasonalPanelSelection,
    panel_c_selection: SeasonalPanelSelection,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "legend.fontsize": 9.2,
            "legend.title_fontsize": 9.2,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.5, 6.3), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 0.75))
    annual_axis = figure.add_subplot(grid[0, :])
    left_axis = figure.add_subplot(grid[1, 0])
    right_axis = figure.add_subplot(
        grid[1, 1], sharex=left_axis, sharey=left_axis
    )
    figure.set_constrained_layout_pads(
        w_pad=0.035,
        h_pad=0.035,
        wspace=0.08,
        hspace=0.08,
    )

    _plot_annual_panel(
        annual_axis,
        annual_gcm,
        sails,
        no_sails,
    )
    _plot_diurnal_panel(
        left_axis,
        panel_b_selection,
        0,
        "B",
        seasonal_panel_gcm,
        sails,
        no_sails,
    )
    _plot_diurnal_panel(
        right_axis,
        panel_c_selection,
        1,
        "C",
        seasonal_panel_gcm,
        sails,
        no_sails,
    )
    left_axis.set_ylabel("Surface temperature (K)")
    right_axis.tick_params(axis="y", which="both", labelleft=True)
    figure.supxlabel("Local time (h)", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    no_sails = load_model_output(args.no_sails_input)
    sails = load_model_output(args.sails_input)
    validate_simulation_alignment(
        no_sails,
        sails,
        "No reflectors",
        "With reflectors",
    )
    gcm = load_reference_gcm_output(args.gcm_input)

    gcm_simulation_indices, gcm_simulation_offsets_deg = nearest_model_indices(
        no_sails.statistics.solar_longitude_deg,
        gcm.statistics.solar_longitude_deg,
    )
    if np.unique(gcm_simulation_indices).size != gcm.statistics.n_sols:
        raise ValueError("GCM-to-simulation L_s mapping is not one-to-one")
    maximum_grid_offset_deg = float(np.max(np.abs(gcm_simulation_offsets_deg)))
    if maximum_grid_offset_deg > GCM_TO_SIMULATION_LS_TOLERANCE_DEG:
        raise ValueError(
            "simulation and GCM seasonal L_s grids differ by more than "
            f"{GCM_TO_SIMULATION_LS_TOLERANCE_DEG} degrees: "
            f"{maximum_grid_offset_deg}"
        )
    gcm_phase_correlation, gcm_phase_anomaly_rmse_K = (
        gcm_simulation_diurnal_alignment(
            gcm,
            no_sails,
            gcm_simulation_indices,
        )
    )

    if not 0.0 < args.gcm_ls_step_deg <= 180.0:
        raise ValueError("--gcm-ls-step-deg must lie in (0, 180]")
    annual_gcm_targets_deg = np.arange(0.0, 360.0, args.gcm_ls_step_deg)
    annual_gcm = build_reference_gcm_seasonal_composite(
        gcm,
        annual_gcm_targets_deg,
        args.gcm_window_width_deg,
    )
    panel_b_selection, panel_c_selection = select_seasonal_panels(
        no_sails,
        DIURNAL_PANEL_SOLAR_LONGITUDES_DEG,
    )
    seasonal_panel_gcm = build_reference_gcm_seasonal_composite(
        gcm,
        np.asarray(DIURNAL_PANEL_SOLAR_LONGITUDES_DEG),
        args.gcm_window_width_deg,
    )
    _validate_temperature_limits(
        (
            ("Reference GCM annual composites", annual_gcm),
            ("Reference GCM fixed-season composites", seasonal_panel_gcm),
            ("No reflectors", no_sails),
            ("With reflectors", sails),
        )
    )
    LOGGER.info(
        "Figure mode: with-reflectors solid, no-reflectors dashed, true GCM "
        "unconnected x markers; %d Panel-A GCM composites at %.6g-degree "
        "spacing use centered %.6g-degree L_s windows (%d..%d profiles/window, "
        "mean %.3f); GCM-to-simulation seasonal-grid maximum |L_s offset| "
        "%.6f degrees",
        annual_gcm.statistics.n_sols,
        args.gcm_ls_step_deg,
        annual_gcm.window_width_deg,
        int(np.min(annual_gcm.window_counts)),
        int(np.max(annual_gcm.window_counts)),
        float(np.mean(annual_gcm.window_counts)),
        maximum_grid_offset_deg,
    )
    LOGGER.info(
        "Reference-GCM explicit-local-time diagnostic against nearest-season "
        "no-reflector simulation profiles: diurnal-anomaly correlation=%.9f, "
        "RMSE=%.9f K",
        gcm_phase_correlation,
        gcm_phase_anomaly_rmse_K,
    )
    for label, selection, gcm_panel_index in (
        ("Panel B", panel_b_selection, 0),
        ("Panel C", panel_c_selection, 1),
    ):
        LOGGER.info(
            "%s fixed season: target L_s=%.9f deg; nearest canonical simulation "
            "sol index=%d (sol number %d), L_s=%.9f deg, signed offset=%+.9f "
            "deg; reference-GCM window width=%.6f deg, n_profiles=%d",
            label,
            selection.target_solar_longitude_deg,
            selection.sol_index,
            selection.sol_index + 1,
            selection.simulation_solar_longitude_deg,
            selection.simulation_offset_deg,
            seasonal_panel_gcm.window_width_deg,
            int(seasonal_panel_gcm.window_counts[gcm_panel_index]),
        )

    make_figure(
        annual_gcm,
        seasonal_panel_gcm,
        sails,
        no_sails,
        panel_b_selection,
        panel_c_selection,
        args.output,
    )
    LOGGER.info("Wrote temperature composite to %s", args.output.resolve())


if __name__ == "__main__":
    main()
