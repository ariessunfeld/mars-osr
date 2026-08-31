#!/usr/bin/env python3
"""Plot sol-wise surface-temperature statistics from climate-model outputs.

The no-reflector GCM reference is sampled at the sol nearest each 10-degree
solar-longitude target and drawn as dashed curves.  The reflector case is drawn
for every available sol as solid curves.  Color identifies the daily minimum,
arithmetic mean, and maximum in both cases.

The handed-off NetCDF files incorrectly label their solar-longitude arrays as
radians even though their values are in degrees.  The loader detects and logs
that inconsistency, then verifies the inferred degree-valued sol means against
the files' stored daily arrays before plotting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from netCDF4 import Dataset, Variable


LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIRECTORY: Final = (
    REPOSITORY_ROOT / "simulation_outputs" / "manuscript_climate"
)
DEFAULT_REFERENCE_INPUT: Final = (
    DEFAULT_DATA_DIRECTORY / "run_reference_no_sail_GCM_IRD.nc"
)
DEFAULT_REFLECTOR_INPUT: Final = (
    DEFAULT_DATA_DIRECTORY / "run_reference_sail_analytical_IRD_cos60.nc"
)
DEFAULT_OUTPUT: Final = (
    REPOSITORY_ROOT
    / "figures"
    / "temperature"
    / "fig_surface_temperature_reference_vs_reflectors.png"
)
DEFAULT_SUMMARY_CSV: Final = (
    DEFAULT_DATA_DIRECTORY / "surface_temperature_daily_statistics.csv"
)

MINIMUM_COLOR: Final = "#1f77b4"
MEAN_COLOR: Final = "#000000"
MAXIMUM_COLOR: Final = "#d62728"
STATISTIC_COLORS: Final = {
    "minimum": MINIMUM_COLOR,
    "mean": MEAN_COLOR,
    "maximum": MAXIMUM_COLOR,
}

EXPECTED_LOCAL_TIME_STEP_H: Final = 0.25
MEAN_TEMPERATURE_TOLERANCE_K: Final = 2.0e-5
MEAN_SOLAR_LONGITUDE_TOLERANCE_DEG: Final = 3.0e-5
MODEL_PARAMETER_NAMES: Final = (
    "dt",
    "nSOL",
    "tstep_per_sol",
    "b_atm",
    "P",
    "f_IR",
    "wind",
    "albd",
    "Rsun",
    "cp_regolith",
    "rho_regolith",
    "k_regolith",
    "Lx",
    "Fgeo",
)


@dataclass(frozen=True)
class DailyTemperatureStatistics:
    """One surface-temperature statistic for every canonical output sol."""

    solar_longitude_deg: np.ndarray
    minimum_K: np.ndarray
    mean_K: np.ndarray
    maximum_K: np.ndarray

    @property
    def n_sols(self) -> int:
        return int(self.solar_longitude_deg.size)


@dataclass(frozen=True)
class ModelOutput:
    """Validated data and provenance from one handed-off NetCDF file."""

    path: Path
    description: str
    sha256: str
    statistics: DailyTemperatureStatistics
    model_parameters: tuple[tuple[str, float], ...]
    samples_per_sol: int
    time_step_value: float
    extra_complete_time_blocks: int
    declared_ls_units: str
    inferred_ls_units: str
    max_stored_temperature_mean_residual_K: float
    max_stored_solar_longitude_mean_residual_deg: float
    time_solar_longitude_deg: np.ndarray
    local_time_h: np.ndarray
    temperature_blocks_K: np.ndarray
    local_time_blocks_h: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-input", type=Path, default=DEFAULT_REFERENCE_INPUT)
    parser.add_argument("--reflector-input", type=Path, default=DEFAULT_REFLECTOR_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--reference-ls-step-deg",
        type=float,
        default=10.0,
        help="Spacing of the no-reflector GCM reference markers (default: 10 degrees).",
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
    if np.ma.is_masked(values) and np.any(np.ma.getmaskarray(values)):
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


def _integer_scalar(dataset: Dataset, name: str) -> int:
    value = _read_scalar(dataset, name)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must be integer-valued, found {value}")
    return rounded


def _solar_longitude_degrees(variable: Variable) -> tuple[np.ndarray, str]:
    values = _read_unmasked(variable).reshape(-1)
    declared_units = str(getattr(variable, "units", "")).strip().lower()
    maximum = float(np.max(values))
    minimum = float(np.min(values))

    if declared_units.startswith("degree"):
        converted = values
        inferred_units = "degrees"
    elif declared_units.startswith("rad") and maximum <= 2.0 * np.pi + 0.1:
        converted = np.degrees(values)
        inferred_units = "radians"
    elif (
        declared_units.startswith("rad")
        and minimum >= -1.0e-6
        and 300.0 <= maximum <= 360.1
    ):
        converted = values
        inferred_units = "degrees (NetCDF attribute mislabeled as radians)"
        LOGGER.warning(
            "%s declares units=%r but spans %.6f..%.6f; treating numeric values "
            "as degrees",
            variable.name,
            declared_units,
            minimum,
            maximum,
        )
    else:
        raise ValueError(
            f"Cannot infer solar-longitude units for {variable.name}: "
            f"declared={declared_units!r}, range={minimum:.9g}..{maximum:.9g}"
        )

    return converted, inferred_units


def _validate_local_time(local_time_h: np.ndarray, samples_per_sol: int) -> None:
    if local_time_h.size < samples_per_sol + 1:
        raise ValueError("LT_t is too short to validate one complete sol")
    phase_rad = local_time_h * (2.0 * np.pi / 24.0)
    unwrapped_h = np.unwrap(phase_rad) * 24.0 / (2.0 * np.pi)
    increments_h = np.diff(unwrapped_h)
    if not np.allclose(
        increments_h,
        EXPECTED_LOCAL_TIME_STEP_H,
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise ValueError(
            "LT_t is not a uniform 0.25-hour grid: "
            f"range={increments_h.min():.9g}..{increments_h.max():.9g} h"
        )
    first_sol = local_time_h[:samples_per_sol]
    if np.unique(first_sol).size != samples_per_sol:
        raise ValueError("the first sol does not contain 96 unique local-time bins")


def load_model_output(path: Path) -> ModelOutput:
    if not path.is_file():
        raise FileNotFoundError(path)

    with Dataset(path, mode="r") as dataset:
        required_variables = {
            *MODEL_PARAMETER_NAMES,
            "Ls_t",
            "LT_t",
            "Tsfc_t",
            "Ls_sol_s",
            "Tsfc_sol_s",
        }
        missing = sorted(required_variables.difference(dataset.variables))
        if missing:
            raise ValueError(f"{path} is missing required variables: {missing}")

        declared_sols = _integer_scalar(dataset, "nSOL")
        samples_per_sol = _integer_scalar(dataset, "tstep_per_sol")
        time_step_value = _read_scalar(dataset, "dt")
        if samples_per_sol != 96:
            raise ValueError(
                f"expected 96 samples per sol, found {samples_per_sol} in {path}"
            )
        if not math.isclose(time_step_value, 900.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"expected numeric dt=900, found {time_step_value} in {path}")

        temperature_time_K = _read_unmasked(dataset.variables["Tsfc_t"]).reshape(-1)
        local_time_h = _read_unmasked(dataset.variables["LT_t"]).reshape(-1)
        ls_time_deg, ls_time_inference = _solar_longitude_degrees(
            dataset.variables["Ls_t"]
        )
        ls_stored_deg, ls_stored_inference = _solar_longitude_degrees(
            dataset.variables["Ls_sol_s"]
        )
        temperature_stored_K = _read_unmasked(
            dataset.variables["Tsfc_sol_s"]
        ).reshape(-1)

        coordinate_lengths = {
            temperature_time_K.size,
            local_time_h.size,
            ls_time_deg.size,
        }
        if len(coordinate_lengths) != 1:
            raise ValueError(f"time-coordinate lengths differ in {path}")
        if temperature_stored_K.size != declared_sols:
            raise ValueError(
                f"Tsfc_sol_s has {temperature_stored_K.size} values, expected {declared_sols}"
            )
        if ls_stored_deg.size != declared_sols:
            raise ValueError(
                f"Ls_sol_s has {ls_stored_deg.size} values, expected {declared_sols}"
            )
        if ls_time_inference != ls_stored_inference:
            raise ValueError(
                "Ls_t and Ls_sol_s require different unit interpretations: "
                f"{ls_time_inference!r} vs {ls_stored_inference!r}"
            )

        _validate_local_time(local_time_h, samples_per_sol)
        intervals = temperature_time_K.size - 1
        if intervals % samples_per_sol != 0:
            raise ValueError(
                f"{intervals} time intervals do not divide into {samples_per_sol}-sample sols"
            )
        complete_blocks = intervals // samples_per_sol
        extra_complete_blocks = complete_blocks - declared_sols
        if extra_complete_blocks != 1:
            raise ValueError(
                "expected the handed-off one-extra-block layout "
                f"({declared_sols} stored sols plus one), found {complete_blocks} blocks"
            )

        canonical_samples = declared_sols * samples_per_sol
        temperature_blocks_K = temperature_time_K[:canonical_samples].reshape(
            declared_sols, samples_per_sol
        )
        local_time_blocks_h = local_time_h[:canonical_samples].reshape(
            declared_sols, samples_per_sol
        )
        ls_blocks_deg = ls_time_deg[:canonical_samples].reshape(
            declared_sols, samples_per_sol
        )
        computed_temperature_mean_K = temperature_blocks_K.mean(axis=1)
        computed_ls_mean_deg = ls_blocks_deg.mean(axis=1)

        temperature_mean_residual_K = computed_temperature_mean_K - temperature_stored_K
        ls_mean_residual_deg = computed_ls_mean_deg - ls_stored_deg
        max_temperature_residual_K = float(
            np.max(np.abs(temperature_mean_residual_K))
        )
        max_ls_residual_deg = float(np.max(np.abs(ls_mean_residual_deg)))
        if max_temperature_residual_K > MEAN_TEMPERATURE_TOLERANCE_K:
            raise ValueError(
                "96-bin mean does not reproduce Tsfc_sol_s: "
                f"max residual={max_temperature_residual_K:.9g} K"
            )
        if max_ls_residual_deg > MEAN_SOLAR_LONGITUDE_TOLERANCE_DEG:
            raise ValueError(
                "96-bin mean does not reproduce Ls_sol_s: "
                f"max residual={max_ls_residual_deg:.9g} degrees"
            )
        if not np.all(np.diff(ls_stored_deg) > 0.0):
            raise ValueError("canonical daily solar longitudes are not strictly increasing")
        if ls_stored_deg[0] < 0.0 or ls_stored_deg[-1] >= 360.0:
            raise ValueError(
                f"canonical solar-longitude range is invalid: "
                f"{ls_stored_deg[0]}..{ls_stored_deg[-1]}"
            )

        declared_ls_units = str(getattr(dataset.variables["Ls_t"], "units", ""))
        description = str(getattr(dataset, "description", ""))
        model_parameters = tuple(
            (name, _read_scalar(dataset, name)) for name in MODEL_PARAMETER_NAMES
        )

    statistics = DailyTemperatureStatistics(
        solar_longitude_deg=ls_stored_deg,
        minimum_K=temperature_blocks_K.min(axis=1),
        mean_K=computed_temperature_mean_K,
        maximum_K=temperature_blocks_K.max(axis=1),
    )
    result = ModelOutput(
        path=path.resolve(),
        description=description,
        sha256=_sha256(path),
        statistics=statistics,
        model_parameters=model_parameters,
        samples_per_sol=samples_per_sol,
        time_step_value=time_step_value,
        extra_complete_time_blocks=extra_complete_blocks,
        declared_ls_units=declared_ls_units,
        inferred_ls_units=ls_time_inference,
        max_stored_temperature_mean_residual_K=max_temperature_residual_K,
        max_stored_solar_longitude_mean_residual_deg=max_ls_residual_deg,
        time_solar_longitude_deg=ls_time_deg,
        local_time_h=local_time_h,
        temperature_blocks_K=temperature_blocks_K,
        local_time_blocks_h=local_time_blocks_h,
    )
    LOGGER.info(
        "Loaded %s: %d canonical sols x %d bins; Ls %.6f..%.6f deg; "
        "Tsfc %.6f..%.6f K; sha256=%s",
        path,
        result.statistics.n_sols,
        result.samples_per_sol,
        result.statistics.solar_longitude_deg[0],
        result.statistics.solar_longitude_deg[-1],
        min(
            float(np.min(result.statistics.minimum_K)),
            float(np.min(result.statistics.mean_K)),
        ),
        max(
            float(np.max(result.statistics.maximum_K)),
            float(np.max(result.statistics.mean_K)),
        ),
        result.sha256,
    )
    return result


def validate_pair(reference: ModelOutput, reflectors: ModelOutput) -> None:
    if reference.description != reflectors.description:
        raise ValueError("reference and reflector descriptions differ")
    if reference.model_parameters != reflectors.model_parameters:
        raise ValueError("reference and reflector model parameters differ")
    if reference.samples_per_sol != reflectors.samples_per_sol:
        raise ValueError("reference and reflector samples-per-sol differ")
    if reference.time_step_value != reflectors.time_step_value:
        raise ValueError("reference and reflector time steps differ")
    if not np.array_equal(
        reference.statistics.solar_longitude_deg,
        reflectors.statistics.solar_longitude_deg,
    ):
        raise ValueError("reference and reflector daily solar-longitude grids differ")
    if not np.array_equal(
        reference.time_solar_longitude_deg,
        reflectors.time_solar_longitude_deg,
    ):
        raise ValueError(
            "reference and reflector time-resolved solar-longitude grids differ"
        )
    if not np.array_equal(reference.local_time_h, reflectors.local_time_h):
        raise ValueError("reference and reflector local-time grids differ")


def nearest_reference_indices(
    solar_longitude_deg: np.ndarray, spacing_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < spacing_deg <= 180.0:
        raise ValueError("--reference-ls-step-deg must be in (0, 180]")
    target_longitudes_deg = np.arange(0.0, 360.0, spacing_deg)
    indices: list[int] = []
    offsets_deg: list[float] = []
    for target_deg in target_longitudes_deg:
        signed_offsets_deg = (
            (solar_longitude_deg - target_deg + 180.0) % 360.0
        ) - 180.0
        index = int(np.argmin(np.abs(signed_offsets_deg)))
        indices.append(index)
        offsets_deg.append(float(signed_offsets_deg[index]))
    index_array = np.asarray(indices, dtype=int)
    if np.unique(index_array).size != index_array.size:
        raise ValueError("reference marker spacing selected the same sol more than once")
    return (
        target_longitudes_deg,
        index_array,
        np.asarray(offsets_deg, dtype=np.float64),
    )


def write_summary_csv(
    output_path: Path,
    reference: ModelOutput,
    reflectors: ModelOutput,
    reference_target_longitudes_deg: np.ndarray,
    reference_indices: np.ndarray,
) -> None:
    target_by_index = {
        int(index): float(target)
        for target, index in zip(
            reference_target_longitudes_deg, reference_indices, strict=True
        )
    }
    fieldnames = (
        "scenario",
        "source_sha256",
        "sol_index",
        "solar_longitude_deg",
        "surface_temperature_minimum_K",
        "surface_temperature_mean_K",
        "surface_temperature_maximum_K",
        "selected_as_reference_marker",
        "reference_target_solar_longitude_deg",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, model in (
            ("reference_gcm", reference),
            ("reflectors", reflectors),
        ):
            stats = model.statistics
            for sol_index in range(stats.n_sols):
                target = (
                    target_by_index.get(sol_index)
                    if scenario == "reference_gcm"
                    else None
                )
                writer.writerow(
                    {
                        "scenario": scenario,
                        "source_sha256": model.sha256,
                        "sol_index": sol_index,
                        "solar_longitude_deg": f"{stats.solar_longitude_deg[sol_index]:.9f}",
                        "surface_temperature_minimum_K": f"{stats.minimum_K[sol_index]:.9f}",
                        "surface_temperature_mean_K": f"{stats.mean_K[sol_index]:.9f}",
                        "surface_temperature_maximum_K": f"{stats.maximum_K[sol_index]:.9f}",
                        "selected_as_reference_marker": target is not None,
                        "reference_target_solar_longitude_deg": (
                            "" if target is None else f"{target:.9f}"
                        ),
                    }
                )


def _style_axis(axis: plt.Axes) -> None:
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
        labelsize=11,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.7,
    )
    axis.grid(which="major", color="#d9d9d9", linewidth=0.55)
    axis.set_axisbelow(True)


def make_figure(
    reference: ModelOutput,
    reflectors: ModelOutput,
    reference_indices: np.ndarray,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "mathtext.fontset": "stix",
            "axes.labelsize": 13,
            "legend.fontsize": 10.5,
            "legend.title_fontsize": 10.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    figure.set_constrained_layout_pads(w_pad=0.035, h_pad=0.035)

    reference_stats = reference.statistics
    reflector_stats = reflectors.statistics
    series = (
        ("minimum", reflector_stats.minimum_K, reference_stats.minimum_K),
        ("mean", reflector_stats.mean_K, reference_stats.mean_K),
        ("maximum", reflector_stats.maximum_K, reference_stats.maximum_K),
    )
    for statistic_name, reflector_values_K, reference_values_K in series:
        color = STATISTIC_COLORS[statistic_name]
        axis.plot(
            reflector_stats.solar_longitude_deg,
            reflector_values_K,
            color=color,
            linewidth=1.8,
            linestyle="-",
            zorder=2,
        )
        axis.plot(
            reference_stats.solar_longitude_deg[reference_indices],
            reference_values_K[reference_indices],
            color=color,
            linewidth=1.45,
            linestyle="--",
            dashes=(5.0, 3.0),
            zorder=3,
        )

    axis.set_xlim(0.0, 360.0)
    axis.set_ylim(160.0, 290.0)
    axis.set_xticks(np.arange(0.0, 360.0 + 15.0, 15.0))
    axis.xaxis.set_minor_locator(MultipleLocator(5.0))
    axis.set_yticks(np.arange(160.0, 290.0 + 10.0, 10.0))
    axis.yaxis.set_minor_locator(MultipleLocator(1.0))
    axis.set_xlabel(r"Solar longitude, $L_s$ (degrees)")
    axis.set_ylabel("Surface temperature (K)")
    _style_axis(axis)

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
            label=r"Reference GCM (10$^\circ$ $L_s$ spacing)",
        ),
    ]
    statistic_legend = axis.legend(
        handles=statistic_handles,
        title="Daily statistic",
        loc="upper left",
        frameon=False,
        handlelength=2.7,
        borderaxespad=0.45,
    )
    axis.add_artist(statistic_legend)
    axis.legend(
        handles=simulation_handles,
        title="Simulation",
        loc="upper right",
        frameon=False,
        handlelength=2.7,
        borderaxespad=0.45,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    reference = load_model_output(args.reference_input)
    reflectors = load_model_output(args.reflector_input)
    validate_pair(reference, reflectors)
    targets_deg, reference_indices, reference_offsets_deg = nearest_reference_indices(
        reference.statistics.solar_longitude_deg,
        args.reference_ls_step_deg,
    )
    LOGGER.info(
        "Selected %d reference sols for %.6g-degree targets; maximum |Ls offset| "
        "is %.6f degrees",
        reference_indices.size,
        args.reference_ls_step_deg,
        float(np.max(np.abs(reference_offsets_deg))),
    )
    mean_temperature_delta_K = (
        reflectors.statistics.mean_K - reference.statistics.mean_K
    )
    LOGGER.info(
        "Reflector-minus-reference sol-mean temperature: range %.6f..%.6f K; "
        "annual arithmetic mean %.6f K",
        float(np.min(mean_temperature_delta_K)),
        float(np.max(mean_temperature_delta_K)),
        float(np.mean(mean_temperature_delta_K)),
    )
    write_summary_csv(
        args.summary_csv,
        reference,
        reflectors,
        targets_deg,
        reference_indices,
    )
    make_figure(reference, reflectors, reference_indices, args.output)
    LOGGER.info("Wrote derived daily statistics to %s", args.summary_csv.resolve())
    LOGGER.info("Wrote figure to %s", args.output.resolve())


if __name__ == "__main__":
    main()
