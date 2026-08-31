#!/usr/bin/env python3
"""Create the 600-layer annual/diurnal TOA and surface-flux composite.

All five displayed curves come from paired COMIMART NetCDF outputs supplied by
the climate-model coauthor.
Natural direct surface irradiance is obtained sample-by-sample as
``Qsun_sfc_total_t - Qsun_sfc_diff_t``.

Panels B/C use the nearest COMIMART sols to fixed L_s targets. Every displayed
field remains at its native NetCDF time index against ``LT_t``; the script does
not ingest, phase-align, or normalize an external CSV profile. The handed-off
sail NetCDF already represents the 1.44x-area (14,400 m2 per reflector) case.
An optional additional uniform reflector-area scale is applied identically to
the reflector TOA and surface curves; its default is 1.0, and natural sunlight
is never scaled.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from netCDF4 import Dataset, Variable


LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE_INPUT_DIRECTORY: Final = (
    REPOSITORY_ROOT
    / "simulation_outputs"
    / "manuscript_climate"
)
DEFAULT_NO_SAIL_SURFACE_INPUT: Final = (
    DEFAULT_SURFACE_INPUT_DIRECTORY / "run_no_sail_comimart_600Lz.nc"
)
DEFAULT_SAIL_SURFACE_INPUT: Final = (
    DEFAULT_SURFACE_INPUT_DIRECTORY / "run_1p44_sail_comimart_600Lz.nc"
)
DEFAULT_OUTPUT: Final = (
    REPOSITORY_ROOT
    / "figures"
    / "manuscript"
    / "generated"
    / "figure_09_insolation.png"
)

# The supplied COMIMART sail output embeds the 1.44x reflector case. This
# constant describes optional, explicit post-processing scaling; the default
# figure plots the NetCDF amplitudes unchanged.
BASELINE_REFLECTOR_AREA_M2: Final = 14_400.0
BASELINE_REFLECTOR_COUNT: Final = 93_881

SURFACE_FLUX_VARIABLES: Final = (
    "Qsun_toa_t",
    "Qsun_sfc_total_t",
    "Qsun_sfc_diff_t",
    "Qreflect_toa_t",
    "Qreflect_sfc_t",
)
EXPECTED_FLUX_LONG_NAMES: Final = {
    "Qsun_toa_t": "vis solar irradiance at the TOA",
    "Qsun_sfc_total_t": "total solar irradiance at the surface",
    "Qsun_sfc_diff_t": "diffuse component of the solar irradiance at the surface",
    "Qreflect_toa_t": "vis reflector irradiance at the TOA",
    "Qreflect_sfc_t": "direct reflector irradiance at the surface",
}
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

SAMPLES_PER_SOL: Final = 96
EXPECTED_TIME_STEP_S: Final = 900.0
EXPECTED_LOCAL_TIME_STEP_H: Final = 0.25
SOLAR_LONGITUDE_MEAN_TOLERANCE_DEG: Final = 3.0e-5
DIURNAL_PANEL_SOLAR_LONGITUDES_DEG: Final = (105.0, 270.0)

NATURAL_COLOR: Final = "#1f77b4"
REFLECTED_COLOR: Final = "#ff7f0e"
TOA_LINESTYLE: Final = "--"
TOA_DASHES: Final = (5.0, 3.0)
SURFACE_LINESTYLE: Final = "-"
DIRECT_ONLY_LINESTYLE: Final = ":"
REFLECTOR_TOA_LEGEND_LABEL: Final = "Reflectors only, top of atmosphere"
REFLECTOR_SURFACE_LEGEND_LABEL: Final = "Reflectors only, surface (direct only)"


@dataclass(frozen=True)
class SurfaceFluxOutput:
    path: Path
    sha256: str
    description: str
    model_parameters: tuple[tuple[str, float], ...]
    samples_per_sol: int
    time_step_s: float
    extra_complete_time_blocks: int
    solar_longitude_blocks_deg: np.ndarray
    local_time_blocks_h: np.ndarray
    flux_blocks_W_m2: dict[str, np.ndarray]

    @property
    def n_sols(self) -> int:
        return int(self.solar_longitude_blocks_deg.shape[0])

    @property
    def solar_longitude_sol_mean_deg(self) -> np.ndarray:
        return np.mean(self.solar_longitude_blocks_deg, axis=1)


@dataclass(frozen=True)
class SurfaceFluxPair:
    no_sail: SurfaceFluxOutput
    sail: SurfaceFluxOutput
    natural_toa_blocks_W_m2: np.ndarray
    natural_surface_direct_blocks_W_m2: np.ndarray
    natural_surface_total_blocks_W_m2: np.ndarray
    reflected_surface_blocks_W_m2: np.ndarray
    reflected_toa_blocks_W_m2: np.ndarray

    @property
    def solar_longitude_blocks_deg(self) -> np.ndarray:
        return self.no_sail.solar_longitude_blocks_deg

    @property
    def local_time_blocks_h(self) -> np.ndarray:
        return self.no_sail.local_time_blocks_h

    @property
    def solar_longitude_sol_mean_deg(self) -> np.ndarray:
        return self.no_sail.solar_longitude_sol_mean_deg


@dataclass(frozen=True)
class DiurnalSurfaceSelection:
    surface_sol_index: int
    surface_solar_longitude_deg: float
    seasonal_offset_deg: float
    local_time_h: np.ndarray
    natural_toa_W_m2: np.ndarray
    natural_surface_direct_W_m2: np.ndarray
    natural_surface_total_W_m2: np.ndarray
    reflected_toa_W_m2: np.ndarray
    reflected_surface_W_m2: np.ndarray


def positive_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-sail-surface-input",
        type=Path,
        default=DEFAULT_NO_SAIL_SURFACE_INPUT,
        help="COMIMART no-sail NetCDF carrying natural TOA and surface flux.",
    )
    parser.add_argument(
        "--sail-surface-input",
        type=Path,
        default=DEFAULT_SAIL_SURFACE_INPUT,
        help="COMIMART sail NetCDF carrying reflector TOA and surface flux.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bin-width",
        type=float,
        default=2.0,
        help="Solar-longitude bin width for Panel A (default: 2 degrees).",
    )
    parser.add_argument(
        "--reflector-area-scale",
        type=positive_finite_float,
        default=1.0,
        help=(
            "Additional post-process reflector-area multiplier applied to both "
            "reflector TOA and surface irradiance in the embedded 1.44x input "
            "case, with accepted delivery windows fixed (default: 1.0)."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_difference_degrees(values_deg: np.ndarray, reference_deg: float) -> np.ndarray:
    return (values_deg - reference_deg + 180.0) % 360.0 - 180.0


def scale_reflector_irradiance(
    values_W_m2: Sequence[float] | np.ndarray,
    reflector_area_scale: float,
) -> np.ndarray:
    """Scale validated reflector irradiance at fixed rays and accepted windows."""
    if not math.isfinite(reflector_area_scale) or reflector_area_scale <= 0.0:
        raise ValueError("reflector_area_scale must be finite and greater than zero")
    values = np.asarray(values_W_m2, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("reflector irradiance must be finite and nonnegative")
    return values * reflector_area_scale


def _read_unmasked(variable: Variable) -> np.ndarray:
    values = np.ma.asarray(variable[:])
    if np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{variable.name} contains masked values")
    result = np.asarray(np.ma.getdata(values), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{variable.name} contains non-finite values")
    return result


def _read_scalar(dataset: Dataset, name: str) -> float:
    values = _read_unmasked(dataset.variables[name])
    if values.size != 1:
        raise ValueError(f"{name} must contain one value, found {values.size}")
    return float(values[0])


def _read_integer_scalar(dataset: Dataset, name: str) -> int:
    value = _read_scalar(dataset, name)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must be integer-valued, found {value}")
    return rounded


def _normalized_metadata_text(value: object) -> str:
    return " ".join(str(value).split())


def load_surface_flux_output(path: Path) -> SurfaceFluxOutput:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Dataset(path, mode="r") as dataset:
        required_variables = {
            *MODEL_PARAMETER_NAMES,
            "Ls_t",
            "LT_t",
            "Ls_sol_s",
            *SURFACE_FLUX_VARIABLES,
        }
        missing = sorted(required_variables.difference(dataset.variables))
        if missing:
            raise ValueError(f"{path} is missing variables {missing}")
        declared_sols = _read_integer_scalar(dataset, "nSOL")
        samples_per_sol = _read_integer_scalar(dataset, "tstep_per_sol")
        time_step_s = _read_scalar(dataset, "dt")
        if samples_per_sol != SAMPLES_PER_SOL:
            raise ValueError(
                f"{path} has {samples_per_sol} samples/sol, expected {SAMPLES_PER_SOL}"
            )
        if not math.isclose(
            time_step_s,
            EXPECTED_TIME_STEP_S,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"{path} has dt={time_step_s}, expected {EXPECTED_TIME_STEP_S} s"
            )

        solar_longitude_time_deg = _read_unmasked(dataset.variables["Ls_t"])
        local_time_h = _read_unmasked(dataset.variables["LT_t"])
        solar_longitude_stored_deg = _read_unmasked(
            dataset.variables["Ls_sol_s"]
        )
        flux_time_W_m2: dict[str, np.ndarray] = {}
        for name in SURFACE_FLUX_VARIABLES:
            variable = dataset.variables[name]
            units = str(getattr(variable, "units", "")).strip()
            long_name = _normalized_metadata_text(getattr(variable, "long_name", ""))
            if variable.dimensions != ("time",):
                raise ValueError(
                    f"{path}:{name} dimensions are {variable.dimensions}, expected ('time',)"
                )
            if units != "W/m2":
                raise ValueError(f"{path}:{name} units are {units!r}, expected 'W/m2'")
            if long_name != EXPECTED_FLUX_LONG_NAMES[name]:
                raise ValueError(
                    f"{path}:{name} long_name is {long_name!r}, expected "
                    f"{EXPECTED_FLUX_LONG_NAMES[name]!r}"
                )
            flux_time_W_m2[name] = _read_unmasked(variable)

        coordinate_lengths = {
            solar_longitude_time_deg.size,
            local_time_h.size,
            *(values.size for values in flux_time_W_m2.values()),
        }
        if len(coordinate_lengths) != 1:
            raise ValueError(f"time-dependent lengths differ in {path}")
        if solar_longitude_stored_deg.size != declared_sols:
            raise ValueError(
                f"{path}: Ls_sol_s has {solar_longitude_stored_deg.size} values, "
                f"expected {declared_sols}"
            )
        intervals = solar_longitude_time_deg.size - 1
        if intervals % samples_per_sol != 0:
            raise ValueError(f"{path}: time intervals do not divide into complete sols")
        complete_blocks = intervals // samples_per_sol
        extra_complete_time_blocks = complete_blocks - declared_sols
        if extra_complete_time_blocks != 1:
            raise ValueError(
                f"{path}: expected one extra complete time block, found "
                f"{extra_complete_time_blocks}"
            )

        canonical_sample_count = declared_sols * samples_per_sol
        solar_longitude_blocks_deg = solar_longitude_time_deg[
            :canonical_sample_count
        ].reshape(declared_sols, samples_per_sol)
        local_time_blocks_h = local_time_h[:canonical_sample_count].reshape(
            declared_sols, samples_per_sol
        )
        flux_blocks_W_m2 = {
            name: values[:canonical_sample_count].reshape(
                declared_sols,
                samples_per_sol,
            )
            for name, values in flux_time_W_m2.items()
        }
        computed_sol_longitude_deg = np.mean(solar_longitude_blocks_deg, axis=1)
        maximum_stored_ls_residual_deg = float(
            np.max(np.abs(computed_sol_longitude_deg - solar_longitude_stored_deg))
        )
        if maximum_stored_ls_residual_deg > SOLAR_LONGITUDE_MEAN_TOLERANCE_DEG:
            raise ValueError(
                f"{path}: 96-bin L_s means disagree with Ls_sol_s by "
                f"{maximum_stored_ls_residual_deg:.9g} degrees"
            )
        if not np.all(np.diff(computed_sol_longitude_deg) > 0.0):
            raise ValueError(f"{path}: canonical sol-mean L_s is not increasing")

        expected_sorted_local_time_h = np.arange(samples_per_sol) * (
            24.0 / samples_per_sol
        )
        if not np.allclose(
            np.sort(local_time_blocks_h, axis=1),
            expected_sorted_local_time_h,
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError(f"{path}: local-time blocks are not the expected 0.25-h grid")
        phase_rad = local_time_h * (2.0 * np.pi / 24.0)
        local_time_increments_h = np.diff(np.unwrap(phase_rad)) * 24.0 / (
            2.0 * np.pi
        )
        if not np.allclose(
            local_time_increments_h,
            EXPECTED_LOCAL_TIME_STEP_H,
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError(f"{path}: LT_t is not uniformly increasing cyclically")

        for name, values in flux_blocks_W_m2.items():
            if np.any(values < 0.0):
                raise ValueError(f"{path}:{name} contains negative values")

        model_parameters = tuple(
            (name, _read_scalar(dataset, name)) for name in MODEL_PARAMETER_NAMES
        )
        description = str(getattr(dataset, "description", ""))

    result = SurfaceFluxOutput(
        path=path.resolve(),
        sha256=sha256(path),
        description=description,
        model_parameters=model_parameters,
        samples_per_sol=samples_per_sol,
        time_step_s=time_step_s,
        extra_complete_time_blocks=extra_complete_time_blocks,
        solar_longitude_blocks_deg=solar_longitude_blocks_deg,
        local_time_blocks_h=local_time_blocks_h,
        flux_blocks_W_m2=flux_blocks_W_m2,
    )
    LOGGER.info(
        "Loaded COMIMART flux output %s: %d canonical sols x %d samples, "
        "L_s %.6f..%.6f deg, description=%r, sha256=%s",
        path,
        result.n_sols,
        result.samples_per_sol,
        result.solar_longitude_sol_mean_deg[0],
        result.solar_longitude_sol_mean_deg[-1],
        result.description,
        result.sha256,
    )
    return result


def load_surface_flux_pair(
    no_sail_path: Path,
    sail_path: Path,
) -> SurfaceFluxPair:
    no_sail = load_surface_flux_output(no_sail_path)
    sail = load_surface_flux_output(sail_path)
    if no_sail.model_parameters != sail.model_parameters:
        raise ValueError("no-sail and sail COMIMART model parameters differ")
    if no_sail.description != sail.description:
        raise ValueError("no-sail and sail COMIMART descriptions differ")
    if not np.array_equal(
        no_sail.solar_longitude_blocks_deg,
        sail.solar_longitude_blocks_deg,
    ):
        raise ValueError("no-sail and sail COMIMART L_s grids differ")
    if not np.array_equal(no_sail.local_time_blocks_h, sail.local_time_blocks_h):
        raise ValueError("no-sail and sail COMIMART local-time grids differ")

    for name in ("Qsun_toa_t", "Qsun_sfc_total_t", "Qsun_sfc_diff_t"):
        if not np.array_equal(
            no_sail.flux_blocks_W_m2[name],
            sail.flux_blocks_W_m2[name],
        ):
            raise ValueError(f"natural field {name} differs between paired files")
    for name in ("Qreflect_toa_t", "Qreflect_sfc_t"):
        if np.count_nonzero(no_sail.flux_blocks_W_m2[name]) != 0:
            raise ValueError(f"no-sail field {name} is not identically zero")

    natural_toa = no_sail.flux_blocks_W_m2["Qsun_toa_t"]
    natural_surface_total = no_sail.flux_blocks_W_m2["Qsun_sfc_total_t"]
    natural_diffuse = no_sail.flux_blocks_W_m2["Qsun_sfc_diff_t"]
    natural_surface_direct = natural_surface_total - natural_diffuse
    reflected_toa = sail.flux_blocks_W_m2["Qreflect_toa_t"]
    reflected_surface = sail.flux_blocks_W_m2["Qreflect_sfc_t"]
    if np.any(natural_diffuse > natural_surface_total + 1.0e-6):
        raise ValueError("COMIMART natural diffuse flux exceeds total surface flux")
    if np.any(natural_surface_direct < 0.0):
        raise ValueError("derived COMIMART natural direct surface flux is negative")
    reconstruction_error_W_m2 = float(
        np.max(
            np.abs(
                natural_surface_direct + natural_diffuse - natural_surface_total
            )
        )
    )
    reconstruction_tolerance_W_m2 = (
        4.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(natural_surface_total)))
    )
    if reconstruction_error_W_m2 > reconstruction_tolerance_W_m2:
        raise ValueError(
            "derived direct plus diffuse does not reconstruct COMIMART natural "
            f"surface total: {reconstruction_error_W_m2:.9g} W/m2"
        )
    if np.any(natural_surface_total > natural_toa + 1.0e-5):
        raise ValueError("COMIMART natural surface flux exceeds paired TOA flux")
    if np.any(reflected_surface > reflected_toa + 1.0e-5):
        raise ValueError("COMIMART reflector surface flux exceeds paired TOA flux")
    if np.any((reflected_toa == 0.0) & (reflected_surface > 0.0)):
        raise ValueError("reflector surface flux is positive where paired TOA is zero")

    LOGGER.info(
        "Paired surface components validated: natural mean %.6f top of atmosphere "
        "-> %.6f direct + %.6f diffuse = %.6f total surface W/m2 "
        "(maximum reconstruction residual %.3g W/m2); reflected mean %.6f top "
        "of atmosphere -> %.6f surface W/m2",
        float(np.mean(natural_toa)),
        float(np.mean(natural_surface_direct)),
        float(np.mean(natural_diffuse)),
        float(np.mean(natural_surface_total)),
        reconstruction_error_W_m2,
        float(np.mean(reflected_toa)),
        float(np.mean(reflected_surface)),
    )
    return SurfaceFluxPair(
        no_sail=no_sail,
        sail=sail,
        natural_toa_blocks_W_m2=natural_toa,
        natural_surface_direct_blocks_W_m2=natural_surface_direct,
        natural_surface_total_blocks_W_m2=natural_surface_total,
        reflected_surface_blocks_W_m2=reflected_surface,
        reflected_toa_blocks_W_m2=reflected_toa,
    )


def _bin_sol_means(
    solar_longitudes_deg: np.ndarray,
    component_values_W_m2: Sequence[np.ndarray],
    bin_width_deg: float,
) -> tuple[list[float], list[list[float]]]:
    if not 0.0 < bin_width_deg <= 360.0:
        raise ValueError("--bin-width must be greater than 0 and at most 360")
    if not component_values_W_m2:
        raise ValueError("at least one annual component is required")
    if any(
        len(values_W_m2) != len(solar_longitudes_deg)
        for values_W_m2 in component_values_W_m2
    ):
        raise ValueError("annual component and longitude lengths differ")
    component_bins: list[dict[int, list[float]]] = [
        defaultdict(list) for _ in component_values_W_m2
    ]
    for sol_index, longitude_deg in enumerate(solar_longitudes_deg):
        index = int((longitude_deg % 360.0) // bin_width_deg)
        for bins, values_W_m2 in zip(
            component_bins,
            component_values_W_m2,
            strict=True,
        ):
            bins[index].append(float(values_W_m2[sol_index]))
    bin_indices = [set(bins) for bins in component_bins]
    if any(indices != bin_indices[0] for indices in bin_indices[1:]):
        raise ValueError("annual component bins differ")
    indices = sorted(bin_indices[0])
    centers_deg = [(index + 0.5) * bin_width_deg for index in indices]
    binned_components = [
        [float(np.mean(bins[index])) for index in indices]
        for bins in component_bins
    ]
    return centers_deg, binned_components


def netcdf_annual_series(
    surface: SurfaceFluxPair,
    bin_width_deg: float,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Bin all five plotted NetCDF components on their shared seasonal grid."""
    centers_deg, components = _bin_sol_means(
        surface.solar_longitude_sol_mean_deg,
        (
            np.mean(surface.natural_toa_blocks_W_m2, axis=1),
            np.mean(surface.natural_surface_direct_blocks_W_m2, axis=1),
            np.mean(surface.natural_surface_total_blocks_W_m2, axis=1),
            np.mean(surface.reflected_toa_blocks_W_m2, axis=1),
            np.mean(surface.reflected_surface_blocks_W_m2, axis=1),
        ),
        bin_width_deg,
    )
    return (
        centers_deg,
        components[0],
        components[1],
        components[2],
        components[3],
        components[4],
    )


def select_netcdf_profiles(
    surface: SurfaceFluxPair,
    target_solar_longitude_deg: float,
) -> DiurnalSurfaceSelection:
    """Select one native NetCDF sol without shifting its local-time profile."""

    if not math.isfinite(target_solar_longitude_deg):
        raise ValueError("diurnal target L_s must be finite")
    surface_sol_longitudes_deg = surface.solar_longitude_sol_mean_deg
    seasonal_offsets_deg = circular_difference_degrees(
        surface_sol_longitudes_deg,
        target_solar_longitude_deg,
    )
    surface_sol_index = int(np.argmin(np.abs(seasonal_offsets_deg)))

    circular_steps_deg = np.diff(
        np.concatenate(
            (
                surface_sol_longitudes_deg,
                surface_sol_longitudes_deg[:1] + 360.0,
            )
        )
    )
    if np.any(circular_steps_deg <= 0.0):
        raise ValueError("COMIMART sol-mean L_s grid is not circularly increasing")
    maximum_nearest_offset_deg = 0.5 * float(np.max(circular_steps_deg))
    selected_offset_deg = float(seasonal_offsets_deg[surface_sol_index])
    if abs(selected_offset_deg) > maximum_nearest_offset_deg + 1.0e-12:
        raise ValueError(
            "nearest COMIMART season exceeds half the largest grid step: "
            f"{selected_offset_deg} vs {maximum_nearest_offset_deg} degrees"
        )

    surface_order = np.argsort(surface.local_time_blocks_h[surface_sol_index])
    surface_reflector_toa_W_m2 = surface.reflected_toa_blocks_W_m2[
        surface_sol_index,
        surface_order,
    ]

    local_time_h = surface.local_time_blocks_h[
        surface_sol_index,
        surface_order,
    ]
    natural_toa_W_m2 = surface.natural_toa_blocks_W_m2[
        surface_sol_index,
        surface_order,
    ]
    natural_surface_direct_W_m2 = surface.natural_surface_direct_blocks_W_m2[
        surface_sol_index,
        surface_order,
    ]
    natural_surface_total_W_m2 = surface.natural_surface_total_blocks_W_m2[
        surface_sol_index,
        surface_order,
    ]
    reflected_surface_W_m2 = surface.reflected_surface_blocks_W_m2[
        surface_sol_index,
        surface_order,
    ]
    if np.any(natural_surface_total_W_m2 > natural_toa_W_m2 + 1.0e-5):
        raise ValueError("natural surface curve exceeds paired TOA curve")
    if np.any(reflected_surface_W_m2 > surface_reflector_toa_W_m2 + 1.0e-5):
        raise ValueError("reflector surface curve exceeds paired TOA curve")

    return DiurnalSurfaceSelection(
        surface_sol_index=surface_sol_index,
        surface_solar_longitude_deg=float(
            surface_sol_longitudes_deg[surface_sol_index]
        ),
        seasonal_offset_deg=selected_offset_deg,
        local_time_h=local_time_h,
        natural_toa_W_m2=natural_toa_W_m2,
        natural_surface_direct_W_m2=natural_surface_direct_W_m2,
        natural_surface_total_W_m2=natural_surface_total_W_m2,
        reflected_toa_W_m2=surface_reflector_toa_W_m2,
        reflected_surface_W_m2=reflected_surface_W_m2,
    )


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


def panel_label(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.015,
        0.965,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )


def plot_five_components(
    axis: plt.Axes,
    toa_x_values: Sequence[float],
    natural_toa_W_m2: Sequence[float],
    reflected_toa_W_m2: Sequence[float],
    surface_x_values: Sequence[float],
    natural_surface_direct_W_m2: Sequence[float],
    natural_surface_total_W_m2: Sequence[float],
    reflected_surface_W_m2: Sequence[float],
) -> None:
    axis.plot(
        toa_x_values,
        natural_toa_W_m2,
        color=NATURAL_COLOR,
        linewidth=1.8,
        linestyle=TOA_LINESTYLE,
        dashes=TOA_DASHES,
        zorder=3,
    )
    axis.plot(
        surface_x_values,
        natural_surface_direct_W_m2,
        color=NATURAL_COLOR,
        linewidth=1.6,
        linestyle=DIRECT_ONLY_LINESTYLE,
        dash_capstyle="round",
        zorder=4,
    )
    axis.plot(
        surface_x_values,
        natural_surface_total_W_m2,
        color=NATURAL_COLOR,
        linewidth=1.8,
        linestyle=SURFACE_LINESTYLE,
        zorder=5,
    )
    axis.plot(
        toa_x_values,
        reflected_toa_W_m2,
        color=REFLECTED_COLOR,
        linewidth=1.8,
        linestyle=TOA_LINESTYLE,
        dashes=TOA_DASHES,
        zorder=2,
    )
    axis.plot(
        surface_x_values,
        reflected_surface_W_m2,
        color=REFLECTED_COLOR,
        linewidth=1.6,
        linestyle=SURFACE_LINESTYLE,
        zorder=6,
    )


def make_figure(
    surface: SurfaceFluxPair,
    output_path: Path,
    bin_width_deg: float,
    reflector_area_scale: float,
) -> tuple[DiurnalSurfaceSelection, DiurnalSurfaceSelection]:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.5, 6.3), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(0.78, 1.12))
    annual_axis = figure.add_subplot(grid[0, :])
    left_axis = figure.add_subplot(grid[1, 0])
    right_axis = figure.add_subplot(
        grid[1, 1],
        sharex=left_axis,
        sharey=left_axis,
    )
    figure.set_constrained_layout_pads(
        w_pad=0.035,
        h_pad=0.035,
        wspace=0.08,
        hspace=0.08,
    )

    (
        annual_longitude_deg,
        annual_natural_toa_W_m2,
        annual_natural_surface_direct_W_m2,
        annual_natural_surface_total_W_m2,
        annual_reflected_toa_W_m2,
        annual_reflected_surface_W_m2,
    ) = netcdf_annual_series(surface, bin_width_deg)
    annual_reflected_toa_W_m2 = scale_reflector_irradiance(
        annual_reflected_toa_W_m2,
        reflector_area_scale,
    )
    annual_reflected_surface_W_m2 = scale_reflector_irradiance(
        annual_reflected_surface_W_m2,
        reflector_area_scale,
    )
    plot_five_components(
        annual_axis,
        annual_longitude_deg,
        annual_natural_toa_W_m2,
        annual_reflected_toa_W_m2,
        annual_longitude_deg,
        annual_natural_surface_direct_W_m2,
        annual_natural_surface_total_W_m2,
        annual_reflected_surface_W_m2,
    )
    annual_maximum_W_m2 = max(
        float(np.max(annual_natural_toa_W_m2)),
        float(np.max(annual_natural_surface_direct_W_m2)),
        float(np.max(annual_natural_surface_total_W_m2)),
        float(np.max(annual_reflected_toa_W_m2)),
        float(np.max(annual_reflected_surface_W_m2)),
    )
    annual_y_limit_W_m2 = max(
        200.0,
        math.ceil(1.02 * annual_maximum_W_m2 / 20.0) * 20.0,
    )
    annual_axis.set_xlim(0.0, 360.0)
    annual_axis.set_ylim(0.0, annual_y_limit_W_m2)
    annual_axis.xaxis.set_major_locator(MultipleLocator(60.0))
    annual_axis.xaxis.set_minor_locator(MultipleLocator(30.0))
    annual_axis.yaxis.set_major_locator(MultipleLocator(20.0))
    annual_axis.yaxis.set_minor_locator(MultipleLocator(10.0))
    annual_axis.set_xlabel(r"Solar longitude, $L_s$ (degrees)")
    annual_axis.set_ylabel(r"Sol-mean insolation (W m$^{-2}$)")
    style_axis(annual_axis)
    panel_label(annual_axis, "A")

    legend_handles = (
        Line2D(
            [0],
            [0],
            color=NATURAL_COLOR,
            linewidth=1.8,
            linestyle=TOA_LINESTYLE,
            dashes=TOA_DASHES,
            label="Natural sunlight only, top of atmosphere",
        ),
        Line2D(
            [0],
            [0],
            color=NATURAL_COLOR,
            linewidth=1.8,
            linestyle=SURFACE_LINESTYLE,
            label="Natural sunlight only, surface (direct + diffuse)",
        ),
        Line2D(
            [0],
            [0],
            color=NATURAL_COLOR,
            linewidth=1.6,
            linestyle=DIRECT_ONLY_LINESTYLE,
            dash_capstyle="round",
            label="Natural sunlight only, surface (direct only)",
        ),
        Line2D(
            [0],
            [0],
            color=REFLECTED_COLOR,
            linewidth=1.8,
            linestyle=TOA_LINESTYLE,
            dashes=TOA_DASHES,
            label=REFLECTOR_TOA_LEGEND_LABEL,
        ),
        Line2D(
            [0],
            [0],
            color=REFLECTED_COLOR,
            linewidth=1.6,
            linestyle=SURFACE_LINESTYLE,
            label=REFLECTOR_SURFACE_LEGEND_LABEL,
        ),
    )
    legend = annual_axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        title="Legend applies to all panels",
        handlelength=3.2,
        borderaxespad=0.5,
        ncol=2,
    )
    legend.get_title().set_fontstyle("italic")
    for legend_text, legend_handle in zip(
        legend.get_texts(),
        legend_handles,
        strict=True,
    ):
        if legend_handle.get_linestyle() == SURFACE_LINESTYLE:
            legend_text.set_fontweight("bold")

    selected = (
        (left_axis, DIURNAL_PANEL_SOLAR_LONGITUDES_DEG[0], "B"),
        (right_axis, DIURNAL_PANEL_SOLAR_LONGITUDES_DEG[1], "C"),
    )
    surface_selections: list[DiurnalSurfaceSelection] = []
    daily_maximum_W_m2 = 0.0
    panel_data: list[
        tuple[
            plt.Axes,
            str,
            float,
            np.ndarray,
            np.ndarray,
            DiurnalSurfaceSelection,
        ]
    ] = []
    for axis, target_solar_longitude_deg, label in selected:
        surface_selection = select_netcdf_profiles(
            surface,
            target_solar_longitude_deg,
        )
        surface_selections.append(surface_selection)
        reflected_toa_W_m2 = scale_reflector_irradiance(
            surface_selection.reflected_toa_W_m2,
            reflector_area_scale,
        )
        reflected_surface_W_m2 = scale_reflector_irradiance(
            surface_selection.reflected_surface_W_m2,
            reflector_area_scale,
        )
        daily_maximum_W_m2 = max(
            daily_maximum_W_m2,
            float(np.max(surface_selection.natural_toa_W_m2)),
            float(np.max(reflected_toa_W_m2)),
            float(np.max(surface_selection.natural_surface_direct_W_m2)),
            float(np.max(surface_selection.natural_surface_total_W_m2)),
            float(np.max(reflected_surface_W_m2)),
        )
        panel_data.append(
            (
                axis,
                label,
                target_solar_longitude_deg,
                reflected_toa_W_m2,
                reflected_surface_W_m2,
                surface_selection,
            )
        )
    daily_y_limit_W_m2 = math.ceil(
        1.02 * daily_maximum_W_m2 / 50.0
    ) * 50.0

    for (
        axis,
        label,
        target_solar_longitude_deg,
        reflected_toa_W_m2,
        reflected_surface_W_m2,
        surface_selection,
    ) in panel_data:
        plot_five_components(
            axis,
            surface_selection.local_time_h,
            surface_selection.natural_toa_W_m2,
            reflected_toa_W_m2,
            surface_selection.local_time_h,
            surface_selection.natural_surface_direct_W_m2,
            surface_selection.natural_surface_total_W_m2,
            reflected_surface_W_m2,
        )
        axis.set_xlim(0.0, 24.0)
        axis.set_ylim(0.0, daily_y_limit_W_m2)
        axis.xaxis.set_major_locator(MultipleLocator(4.0))
        axis.xaxis.set_minor_locator(MultipleLocator(2.0))
        axis.yaxis.set_major_locator(MultipleLocator(100.0))
        axis.yaxis.set_minor_locator(MultipleLocator(50.0))
        style_axis(axis)
        panel_label(axis, label)
        axis.text(
            0.09,
            0.965,
            rf"$L_s={target_solar_longitude_deg:g}^\circ$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
        )

    left_axis.set_ylabel(r"Insolation (W m$^{-2}$)")
    right_axis.tick_params(axis="y", which="both", labelleft=False)
    figure.supxlabel("Local time (hours)", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)
    return surface_selections[0], surface_selections[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    surface = load_surface_flux_pair(
        args.no_sail_surface_input,
        args.sail_surface_input,
    )
    panel_b_selection, panel_c_selection = make_figure(
        surface,
        args.output,
        args.bin_width,
        args.reflector_area_scale,
    )

    baseline_total_area_km2 = (
        BASELINE_REFLECTOR_COUNT * BASELINE_REFLECTOR_AREA_M2 / 1.0e6
    )
    scaled_reflector_area_m2 = (
        BASELINE_REFLECTOR_AREA_M2 * args.reflector_area_scale
    )
    scaled_total_area_km2 = baseline_total_area_km2 * args.reflector_area_scale
    natural_toa_mean_W_m2 = float(np.mean(surface.natural_toa_blocks_W_m2))
    netcdf_reflector_toa_mean_W_m2 = float(
        np.mean(surface.reflected_toa_blocks_W_m2)
    )
    surface_reflector_mean_W_m2 = float(
        np.mean(surface.reflected_surface_blocks_W_m2)
    )
    LOGGER.info(
        "Reflector-area scenario: %.9gx baseline; %.6f -> %.6f m2 per "
        "reflector; %.6f -> %.6f km2 across %d reflectors. Raw/scaled year "
        "means: TOA %.9f -> %.9f W/m2; surface %.9f -> %.9f W/m2",
        args.reflector_area_scale,
        BASELINE_REFLECTOR_AREA_M2,
        scaled_reflector_area_m2,
        baseline_total_area_km2,
        scaled_total_area_km2,
        BASELINE_REFLECTOR_COUNT,
        netcdf_reflector_toa_mean_W_m2,
        netcdf_reflector_toa_mean_W_m2 * args.reflector_area_scale,
        surface_reflector_mean_W_m2,
        surface_reflector_mean_W_m2 * args.reflector_area_scale,
    )
    LOGGER.info(
        "All plotted coordinates and ordinate series come from the paired "
        "NetCDF fields; natural TOA year mean=%.9f W/m2. Fixed diurnal seasons "
        "are selected directly on the NetCDF grid. No CSV, cyclic roll, phase "
        "shift, interpolation, or normalization is used.",
        natural_toa_mean_W_m2,
    )

    for panel, target_solar_longitude_deg, selection in (
        ("B", DIURNAL_PANEL_SOLAR_LONGITUDES_DEG[0], panel_b_selection),
        ("C", DIURNAL_PANEL_SOLAR_LONGITUDES_DEG[1], panel_c_selection),
    ):
        LOGGER.info(
            "Panel %s: target L_s=%.9f deg; nearest COMIMART sol number=%d, "
            "L_s=%.9f deg, signed offset=%+.9f deg; native NetCDF means natural "
            "TOA/direct surface/total "
            "surface=%.9f/%.9f/%.9f W/m2; reflected TOA raw/scaled="
            "%.9f/%.9f W/m2; reflected surface raw/scaled=%.9f/%.9f W/m2",
            panel,
            target_solar_longitude_deg,
            selection.surface_sol_index + 1,
            selection.surface_solar_longitude_deg,
            selection.seasonal_offset_deg,
            float(np.mean(selection.natural_toa_W_m2)),
            float(np.mean(selection.natural_surface_direct_W_m2)),
            float(np.mean(selection.natural_surface_total_W_m2)),
            float(np.mean(selection.reflected_toa_W_m2)),
            float(np.mean(selection.reflected_toa_W_m2))
            * args.reflector_area_scale,
            float(np.mean(selection.reflected_surface_W_m2)),
            float(np.mean(selection.reflected_surface_W_m2))
            * args.reflector_area_scale,
        )
    LOGGER.info("Wrote surface-flux insolation composite to %s", args.output.resolve())


if __name__ == "__main__":
    main()
