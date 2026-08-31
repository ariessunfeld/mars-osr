"""Production K12 508-km shell: one, two, and three LTAN rings.

It reads the exact 508-km shell from the continuous-5km Walker products and
compares representative subsets
of its three saved rings:

* one ring: the central 18 h ring (81 reflectors);
* two rings: the lower-LTAN and central rings (162 reflectors); and
* three rings: the complete production shell (243 reflectors).

The panels form a nested sequence: central ring, then lower + central, then all
three rings.  Every curve uses the saved production mean anomalies, including
the Walker F=1 inter-ring phase offsets.  The direct run evaluates all 243
physical reflectors; it does not use the G=18 phase quadrature employed by the
annual 93,881-reflector production campaign.

The orbit/delivery model follows the vacuum-production geometry path:
central + MRO120F J2 + Sun third body, no SRP in the reference orbit, circular
states built from each saved LTAN, ideal-bisector reflected-light geometry,
elevation >= 10 deg, sail sunlit, feasible bisector and slew acceleration,
and a 1 J/m^2 vacuum window threshold.  Reflectors are 10,000 m^2 at
sigma=0.018 kg/m^2.  There are no atmospheric losses.  A 30 s cadence is used
for the figure (twice the temporal resolution of the 60 s annual products).

Inputs (hash-pinned below):
``simulation_outputs/walker_constellation_continuous_5km/{manifest.json,
shells.csv,rings.csv,satellites.csv}``.

Outputs:
``simulation_outputs/20260818_figH3_K12_production_ltan_rings_exact_cache.npz``
``simulation_outputs/20260818_figH3_K12_production_ltan_rings_summary.csv``
``simulation_outputs/20260818_figH3_K12_production_ltan_rings_timeseries.csv``
``simulation_outputs/manuscript_figures/figure_08b_ltan_rings.png``
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import spiceypy as spice

from reflectors.dynamics import PropagationOptions, propagate
from reflectors.elements import state_from_classical_mme2000
from reflectors.ephemeris import utc_to_et
from reflectors.gravity import mars_gravity_model
from reflectors.kernels import load_kernels
from reflectors.mars_constants import SECONDS_PER_SOLAR_SOL_S
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import raan_mme2000_from_ltan
from reflectors.termination import AltitudeFloor
from reflectors.third_body import sun_third_body
from reflectors.visibility import find_delivery_windows

logger = logging.getLogger(__name__)

# --- Authoritative generated constellation -----------------------------------
CONSTELLATION_DIR = Path("simulation_outputs/walker_constellation_continuous_5km")
PRODUCT_PATHS = {
    "manifest": CONSTELLATION_DIR / "manifest.json",
    "shells": CONSTELLATION_DIR / "shells.csv",
    "rings": CONSTELLATION_DIR / "rings.csv",
    "satellites": CONSTELLATION_DIR / "satellites.csv",
}
EXPECTED_PRODUCT_SHA256 = {
    "manifest": "4dd021516b752dcbc5908c614e80d3ad1befd91e2c37a4ded286ee2f360f1090",
    "shells": "9050d64a5508ee5f170efcbb3f0d82f11dae7c97bcb99cb15c6c0f2bea275644",
    "rings": "b004e61078c076bba2840924633360d12cc728d34af90bca4dfcad01dd86fee7",
    "satellites": "2676d1e78da94f7baa5a1e7b219ca9c5356a26baa054e4cca252b9e92d3ed616",
}
EXPECTED_CONSTELLATION_COUNTS = (185, 1388, 93881)
FAMILY_K = 12
SHELL_INDEX = 0
ALTITUDE_KM = 508.0
EXPECTED_RING_COUNT = 3
EXPECTED_SATELLITES_PER_RING = 81
EXPECTED_WALKER_PHASING = 1

# Nested configurations assembled from exact production rings: central ring,
# lower + central rings, then the complete three-ring production shell.
CONFIGURATIONS = (
    ("1 ring", (1,)),
    ("2 rings", (0, 1)),
    ("3 rings", (0, 1, 2)),
)

# --- Base / collector / sail -------------------------------------------------
TARGET_LAT_DEG = 40.0
TARGET_LON_DEG = 200.0
# Exact target height inherited by the accepted annual vacuum G18 production
TARGET_ALTITUDE_KM = 0.4407089901803089
SIGMA_KG_PER_M2 = 0.018000
SAIL_AREA_M2 = 10000.0

# --- Epoch / propagation / delivery gates -----------------------------------
PERIHELION_UTC = "2028-02-11T12:42:00"
N_SOLS = 1
CADENCE_S = 30.0
ELEV_MIN_DEG = 10.0
BISECTOR_COS_ALPHA_MIN = 0.1
ALPHA_MAX_RAD_S2 = math.radians(0.003)
MIN_WINDOW_FLUENCE_J_PER_M2 = 1.0
ATMOSPHERIC_TRANSMISSION = 1.0
ALTITUDE_FLOOR_KM = 300.0
MODEL_VERSION = "production-k12-508km-exact-rings-v1"

_GRAVITY_DEGREE_2 = mars_gravity_model(max_degree=2)
MU_MARS_KM3_S2 = float(_GRAVITY_DEGREE_2.mu_km3_s2)

# Preserve every local-time sample while compressing the three inactive spans.
LST_SEGMENTS = (
    (0.0, 4.0, 0.45),
    (4.0, 9.0, 5.0),
    (9.0, 15.0, 0.60),
    (15.0, 20.0, 5.0),
    (20.0, 24.0, 0.45),
)
LST_TICKS_H = (0.0, 4.0, 9.0, 15.0, 20.0, 24.0)
LST_MINOR_TICKS_H = (5.0, 6.0, 7.0, 8.0, 16.0, 17.0, 18.0, 19.0)
LST_BREAK_CENTRES_H = (2.0, 12.0, 22.0)
Y_LINTHRESH_W_PER_M2 = 0.1
Y_TICKS_W_PER_M2 = (0.0, 0.1, 1.0)

OUT_DIR = Path("simulation_outputs")
FIG_DIR = Path("simulation_outputs/manuscript_figures")
OUT_PREFIX = "20260818_figH3_K12_production_ltan_rings"
CACHE_PATH = OUT_DIR / f"{OUT_PREFIX}_exact_cache.npz"
SUMMARY_PATH = OUT_DIR / f"{OUT_PREFIX}_summary.csv"
TIMESERIES_PATH = OUT_DIR / f"{OUT_PREFIX}_timeseries.csv"
FIGURE_PATH = FIG_DIR / "figure_08b_ltan_rings.png"

# --- Styling -----------------------------------------------------------------
AXIS_FS = 22
TICK_FS = 18
LABEL_FS = 16
TITLE_FS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty constellation product: {path}")
    return rows


def load_production_shell() -> dict[str, object]:
    """Load and independently cross-check K12-H00 from the generated products."""
    obtained_hashes = {name: sha256(path) for name, path in PRODUCT_PATHS.items()}
    if obtained_hashes != EXPECTED_PRODUCT_SHA256:
        raise RuntimeError(
            "continuous-5km product hashes changed; audit the new production "
            f"constellation before plotting: {obtained_hashes}"
        )

    manifest = json.loads(PRODUCT_PATHS["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("production constellation manifest must use schema version 2")
    if manifest.get("shell_layout", {}).get("name") != "continuous-5km":
        raise ValueError("expected the continuous-5km production layout")
    summary = manifest.get("summary", {})
    counts = (
        int(summary.get("shell_count", -1)),
        int(summary.get("ring_count", -1)),
        int(summary.get("satellite_count", -1)),
    )
    if counts != EXPECTED_CONSTELLATION_COUNTS:
        raise ValueError(f"unexpected production constellation counts: {counts}")

    shell_rows = [
        row
        for row in read_csv_rows(PRODUCT_PATHS["shells"])
        if int(row["family_k"]) == FAMILY_K
        and int(row["shell_index"]) == SHELL_INDEX
    ]
    if len(shell_rows) != 1:
        raise ValueError(f"expected one K{FAMILY_K}-H{SHELL_INDEX:02d} shell row")
    shell_row = shell_rows[0]
    if not math.isclose(float(shell_row["altitude_km"]), ALTITUDE_KM, abs_tol=1e-12):
        raise ValueError("K12-H00 is not the expected 508-km shell")
    declared_ring_count = int(shell_row["rings"])
    declared_satellites_per_ring = int(shell_row["satellites_per_ring"])
    declared_phasing = int(shell_row["phasing"])
    declared_total = int(shell_row["satellite_count"])
    expected_shell_tuple = (
        EXPECTED_RING_COUNT,
        EXPECTED_SATELLITES_PER_RING,
        EXPECTED_WALKER_PHASING,
        EXPECTED_RING_COUNT * EXPECTED_SATELLITES_PER_RING,
    )
    if (
        declared_ring_count,
        declared_satellites_per_ring,
        declared_phasing,
        declared_total,
    ) != expected_shell_tuple:
        raise ValueError("508-km shell sizing no longer matches the audited production design")

    ring_rows = [
        row
        for row in read_csv_rows(PRODUCT_PATHS["rings"])
        if int(row["family_k"]) == FAMILY_K
        and int(row["shell_index"]) == SHELL_INDEX
    ]
    ring_rows.sort(key=lambda row: int(row["ring_index"]))
    if [int(row["ring_index"]) for row in ring_rows] != list(
        range(EXPECTED_RING_COUNT)
    ):
        raise ValueError("508-km ring indices are not contiguous")

    satellite_rows = [
        row
        for row in read_csv_rows(PRODUCT_PATHS["satellites"])
        if int(row["family_k"]) == FAMILY_K
        and int(row["shell_index"]) == SHELL_INDEX
    ]
    if len(satellite_rows) != declared_total:
        raise ValueError("satellites.csv does not contain the declared 508-km inventory")

    rings: list[dict[str, object]] = []
    for ring_index, ring_row in enumerate(ring_rows):
        ring_satellites = [
            row for row in satellite_rows if int(row["ring_index"]) == ring_index
        ]
        ring_satellites.sort(key=lambda row: int(row["slot_index"]))
        if [int(row["slot_index"]) for row in ring_satellites] != list(
            range(declared_satellites_per_ring)
        ):
            raise ValueError(f"ring {ring_index} satellite slots are not contiguous")

        ltan_h = float(ring_row["ltan_h"])
        phase_offset_deg = float(ring_row["equivalent_satellite_phase_offset_deg"])
        expected_offset_deg = (
            360.0
            * declared_phasing
            * ring_index
            / (declared_ring_count * declared_satellites_per_ring)
        )
        if not math.isclose(phase_offset_deg, expected_offset_deg, abs_tol=1e-12):
            raise ValueError(f"ring {ring_index} Walker phase offset is inconsistent")
        mean_anomaly_deg = np.array(
            [float(row["initial_mean_anomaly_deg"]) for row in ring_satellites]
        )
        expected_anomaly_deg = (
            phase_offset_deg
            + np.arange(declared_satellites_per_ring)
            * 360.0
            / declared_satellites_per_ring
        ) % 360.0
        np.testing.assert_allclose(mean_anomaly_deg, expected_anomaly_deg, atol=1e-12)
        rings.append(
            {
                "ring_index": ring_index,
                "ring_id": ring_row["ring_id"],
                "ltan_h": ltan_h,
                "phase_offset_deg": phase_offset_deg,
                "mean_anomaly_deg": mean_anomaly_deg,
            }
        )

    return {
        "semimajor_axis_km": float(shell_row["semimajor_axis_km"]),
        "inclination_deg": float(shell_row["inclination_deg"]),
        "ltan_lower_h": float(shell_row["centered_ltan_start_h"]),
        "ltan_upper_h": float(shell_row["centered_ltan_end_h"]),
        "satellites_per_ring": declared_satellites_per_ring,
        "walker_phasing": declared_phasing,
        "rings": tuple(rings),
        "product_hashes": obtained_hashes,
        "constellation_counts": counts,
    }


# --- Local apparent solar time at the base -----------------------------------
def subsolar_lon_deg(et: float) -> float:
    position, _ = spice.spkpos("SUN", et, "IAU_MARS", "NONE", "MARS")
    return math.degrees(math.atan2(float(position[1]), float(position[0])))


def hour_angle_signed_deg(et: float) -> float:
    return (TARGET_LON_DEG - subsolar_lon_deg(et) + 180.0) % 360.0 - 180.0


def local_solar_time_h(et: float) -> float:
    return (hour_angle_signed_deg(et) / 15.0 + 12.0) % 24.0


def compressed_lstime_coordinate(lst_h):
    values = np.asarray(lst_h, dtype=float)
    if np.any((values < 0.0) | (values > 24.0)):
        raise ValueError("local-solar hours must lie in [0, 24]")
    mapped = np.zeros_like(values)
    for start_h, stop_h, display_width in LST_SEGMENTS:
        fraction_h = np.clip(values - start_h, 0.0, stop_h - start_h)
        mapped += display_width * fraction_h / (stop_h - start_h)
    return mapped


def add_x_break_marks(ax) -> None:
    break_x = compressed_lstime_coordinate(LST_BREAK_CENTRES_H)
    transform = ax.get_xaxis_transform()
    for centre in break_x:
        for offset in (-0.045, 0.045):
            ax.plot(
                [centre + offset - 0.035, centre + offset + 0.035],
                [-0.018, 0.018],
                color="black",
                lw=1.0,
                clip_on=False,
                transform=transform,
            )


def build_state(
    semimajor_axis_km: float,
    inclination_deg: float,
    ltan_h: float,
    mean_anomaly_deg: float,
    epoch_et: float,
) -> np.ndarray:
    return state_from_classical_mme2000(
        a_km=semimajor_axis_km,
        e=0.0,
        inclination_rad=math.radians(inclination_deg),
        raan_rad=raan_mme2000_from_ltan(ltan_h, epoch_et),
        argp_rad=0.0,
        nu_rad=math.radians(mean_anomaly_deg),
        mu_km3_s2=MU_MARS_KM3_S2,
        epoch_et=epoch_et,
    )


def reflector_series(
    semimajor_axis_km: float,
    inclination_deg: float,
    ltan_h: float,
    mean_anomaly_deg: float,
    epoch_et: float,
    duration_s: float,
    t_eval_s: np.ndarray,
    sail,
) -> np.ndarray:
    """Vacuum horizontal irradiance from one saved production reflector."""
    result = propagate(
        state0_km_kmps=build_state(
            semimajor_axis_km,
            inclination_deg,
            ltan_h,
            mean_anomaly_deg,
            epoch_et,
        ),
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et,
        zonal_degree=2,
        gravity_degree=0,
        third_bodies=[sun_third_body()],
        solar_sail=None,
        sail_normal=None,
        altitude_floor=AltitudeFloor.at_km(
            ALTITUDE_FLOOR_KM,
            label="altitude_floor",
        ),
        options=PropagationOptions.fast(),
        t_eval_s=t_eval_s,
    )
    windows, samples = find_delivery_windows(
        result,
        TARGET_LAT_DEG,
        TARGET_LON_DEG,
        target_elevation_min_deg=ELEV_MIN_DEG,
        bisector_cos_alpha_min=BISECTOR_COS_ALPHA_MIN,
        alt_km=TARGET_ALTITUDE_KM,
        planetographic=True,
        require_sail_sunlit=True,
        require_sail_above_horizon=True,
        require_bisector_feasible=True,
        sail=sail,
        atmospheric_transmission=ATMOSPHERIC_TRANSMISSION,
        alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
        min_window_fluence_J_per_m2=MIN_WINDOW_FLUENCE_J_PER_M2,
        return_samples=True,
    )
    kept = np.zeros_like(samples.t_s, dtype=bool)
    for window in windows:
        kept |= (samples.t_s >= window.t_start_s) & (
            samples.t_s <= window.t_end_s
        )
    return np.where(kept, samples.vacuum_irradiance_W_per_m2, 0.0)


def propagate_exact_rings(
    shell: dict[str, object],
    epoch_et: float,
    duration_s: float,
    t_eval_s: np.ndarray,
) -> np.ndarray:
    """Return one exact summed irradiance history per saved production ring."""
    rings = shell["rings"]
    satellites_per_ring = int(shell["satellites_per_ring"])
    expected_shape = (len(rings), t_eval_s.size)
    product_hashes_json = json.dumps(shell["product_hashes"], sort_keys=True)
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as cached:
            valid = (
                str(cached["model_version"].item()) == MODEL_VERSION
                and str(cached["product_hashes_json"].item()) == product_hashes_json
                and cached["ring_irradiance_W_m2"].shape == expected_shape
                and np.array_equal(cached["t_eval_s"], t_eval_s)
                and np.allclose(
                    cached["ring_ltan_h"],
                    np.array([ring["ltan_h"] for ring in rings]),
                )
                and int(cached["satellites_per_ring"].item())
                == satellites_per_ring
            )
            if valid:
                print(f"  loaded exact-ring cache {CACHE_PATH}")
                return np.array(cached["ring_irradiance_W_m2"])

    sail = make_canonical_sail(SIGMA_KG_PER_M2, area_m2=SAIL_AREA_M2)
    ring_irradiance = np.zeros(expected_shape, dtype=float)
    wall_start = time.perf_counter()
    completed = 0
    total = len(rings) * satellites_per_ring
    for ring in rings:
        ring_index = int(ring["ring_index"])
        for mean_anomaly_deg in ring["mean_anomaly_deg"]:
            ring_irradiance[ring_index] += reflector_series(
                float(shell["semimajor_axis_km"]),
                float(shell["inclination_deg"]),
                float(ring["ltan_h"]),
                float(mean_anomaly_deg),
                epoch_et,
                duration_s,
                t_eval_s,
                sail,
            )
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(
                    f"  propagated {completed}/{total} exact reflectors "
                    f"[{time.perf_counter() - wall_start:.1f} s]"
                )

    np.savez_compressed(
        CACHE_PATH,
        model_version=np.array(MODEL_VERSION),
        product_hashes_json=np.array(product_hashes_json),
        t_eval_s=t_eval_s,
        ring_irradiance_W_m2=ring_irradiance,
        ring_ltan_h=np.array([ring["ltan_h"] for ring in rings]),
        ring_phase_offset_deg=np.array(
            [ring["phase_offset_deg"] for ring in rings]
        ),
        semimajor_axis_km=float(shell["semimajor_axis_km"]),
        inclination_deg=float(shell["inclination_deg"]),
        satellites_per_ring=satellites_per_ring,
        sail_area_m2=SAIL_AREA_M2,
        cadence_s=CADENCE_S,
        target_altitude_km=TARGET_ALTITUDE_KM,
    )
    print(f"  wrote exact-ring cache {CACHE_PATH}")
    return ring_irradiance


def assemble_configurations(
    shell: dict[str, object],
    ring_irradiance: np.ndarray,
    t_eval_s: np.ndarray,
) -> list[dict[str, object]]:
    configurations: list[dict[str, object]] = []
    duration_s = float(t_eval_s[-1] - t_eval_s[0])
    for label, ring_indices in CONFIGURATIONS:
        series = ring_irradiance[np.array(ring_indices)].sum(axis=0)
        configurations.append(
            {
                "label": label,
                "ring_indices": ring_indices,
                "ltan_h": tuple(
                    float(shell["rings"][index]["ltan_h"])
                    for index in ring_indices
                ),
                "reflector_count": len(ring_indices)
                * int(shell["satellites_per_ring"]),
                "series": series,
                "peak_W_m2": float(series.max()),
                "mean_W_m2": float(np.trapezoid(series, t_eval_s)) / duration_s,
                "fluence_J_m2": float(np.trapezoid(series, t_eval_s)),
                "duty_fraction": float(np.count_nonzero(series > 0.0))
                / float(series.size),
            }
        )
    return configurations


def write_data_products(
    shell: dict[str, object],
    configurations: list[dict[str, object]],
    ring_irradiance: np.ndarray,
    t_eval_s: np.ndarray,
    epoch_et: float,
) -> np.ndarray:
    lst_h = np.array([local_solar_time_h(epoch_et + float(t)) for t in t_eval_s])
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "configuration",
                "ring_indices",
                "ltan_h",
                "reflector_count",
                "peak_W_m2",
                "mean_W_m2",
                "fluence_J_m2_per_sol",
                "duty_fraction",
            ]
        )
        for configuration in configurations:
            writer.writerow(
                [
                    configuration["label"],
                    ";".join(str(value) for value in configuration["ring_indices"]),
                    ";".join(f"{value:.12f}" for value in configuration["ltan_h"]),
                    configuration["reflector_count"],
                    f"{configuration['peak_W_m2']:.12g}",
                    f"{configuration['mean_W_m2']:.12g}",
                    f"{configuration['fluence_J_m2']:.12g}",
                    f"{configuration['duty_fraction']:.12g}",
                ]
            )

    with TIMESERIES_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "t_s",
                "local_apparent_solar_time_h",
                "ring0_W_m2",
                "ring1_W_m2",
                "ring2_W_m2",
                "one_ring_W_m2",
                "two_rings_W_m2",
                "three_rings_W_m2",
            ]
        )
        for sample_index, t_s in enumerate(t_eval_s):
            writer.writerow(
                [
                    f"{t_s:.1f}",
                    f"{lst_h[sample_index]:.12g}",
                    *[
                        f"{ring_irradiance[ring_index, sample_index]:.12g}"
                        for ring_index in range(EXPECTED_RING_COUNT)
                    ],
                    *[
                        f"{configuration['series'][sample_index]:.12g}"
                        for configuration in configurations
                    ],
                ]
            )
    print(f"  wrote {SUMMARY_PATH}")
    print(f"  wrote {TIMESERIES_PATH}")
    return lst_h


def plot_figure(
    shell: dict[str, object],
    configurations: list[dict[str, object]],
    lst_h: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
    order = np.argsort(lst_h)
    x_h = lst_h[order]
    x_plot = compressed_lstime_coordinate(x_h)
    positive_values = np.concatenate(
        [
            configuration["series"][configuration["series"] > 0.0]
            for configuration in configurations
        ]
    )
    if positive_values.size == 0:
        raise ValueError("cannot plot: every production-ring irradiance sample is zero")
    ymax = float(positive_values.max()) * 1.08
    y_minor_ticks = np.concatenate(
        (
            np.arange(0.2, min(1.0, ymax), 0.1),
            np.arange(2.0, ymax, 1.0),
        )
    )
    tick_positions = compressed_lstime_coordinate(LST_TICKS_H)
    minor_tick_positions = compressed_lstime_coordinate(LST_MINOR_TICKS_H)
    x_max = float(compressed_lstime_coordinate(24.0))

    fig, axes = plt.subplots(
        len(configurations),
        1,
        figsize=(12, 7.8),
        sharex=True,
        sharey=True,
    )
    for panel_index, (ax, configuration) in enumerate(
        zip(axes, configurations, strict=True),
        start=1,
    ):
        y = configuration["series"][order]
        ax.plot(x_plot, np.where(y > 0.0, y, np.nan), color="black", lw=1.1)
        ax.set_yscale("symlog", linthresh=Y_LINTHRESH_W_PER_M2, linscale=1.0, base=10)
        ax.set_ylim(0.0, ymax)
        ax.set_yticks(Y_TICKS_W_PER_M2, ["0", "0.1", "1"])
        ax.set_yticks(y_minor_ticks, minor=True)
        ax.set_xlim(0.0, x_max)
        ax.set_xticks(tick_positions, [f"{hour:.0f}" for hour in LST_TICKS_H])
        ax.set_xticks(minor_tick_positions, minor=True)
        ax.set_ylabel(r"$\mathrm{W/m^{2}}$", fontsize=AXIS_FS)
        ax.tick_params(labelsize=TICK_FS)
        ltan_text = ", ".join(f"{value:.2f}" for value in configuration["ltan_h"])
        duty_label = (
            rf"$T_{{\mathrm{{illum.}}}}="
            rf"{100.0 * configuration['duty_fraction']:.0f}\%$"
        )
        ax.text(
            0.985,
            0.93,
            f"{configuration['label']} ({configuration['reflector_count']} reflectors)\n"
            f"LTAN {ltan_text} h\n"
            f"{duty_label}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=LABEL_FS,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        ax.text(
            0.02,
            0.92,
            f"B{panel_index}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=30,
            fontweight="bold",
        )
        ax.grid(which="major", axis="both", alpha=0.22, linewidth=0.8)
        ax.grid(which="minor", axis="x", alpha=0.18, linewidth=0.6)
        ax.grid(which="minor", axis="y", alpha=0.12, linewidth=0.5)
        add_x_break_marks(ax)

    axes[-1].set_xlabel("Local apparent solar time at base (h)", fontsize=AXIS_FS)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(
        f"  shared symlog range 0--{ymax:.3f} W/m^2; "
        f"positive samples {positive_values.min():.6g}--{positive_values.max():.6g}"
    )
    print(f"  wrote {FIGURE_PATH}")


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_kernels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    shell = load_production_shell()
    epoch_et = utc_to_et(PERIHELION_UTC)
    duration_s = float(N_SOLS) * SECONDS_PER_SOLAR_SOL_S
    t_eval_s = np.arange(0.0, duration_s, CADENCE_S)
    if t_eval_s[-1] < duration_s:
        t_eval_s = np.append(t_eval_s, duration_s)

    print("=" * 88)
    print("FIG H3 COMPANION — exact production K12 508-km LTAN rings")
    print("=" * 88)
    print(
        f"  production inventory={shell['constellation_counts'][0]} shells / "
        f"{shell['constellation_counts'][1]:,} rings / "
        f"{shell['constellation_counts'][2]:,} reflectors"
    )
    print(
        f"  shell K12-H00: a={shell['semimajor_axis_km']:.9f} km, "
        f"alt={ALTITUDE_KM:.1f} km, i={shell['inclination_deg']:.12f} deg"
    )
    print(
        f"  {len(shell['rings'])} rings x {shell['satellites_per_ring']} reflectors; "
        f"Walker F={shell['walker_phasing']}; "
        f"centered LTAN interval=[{shell['ltan_lower_h']:.12f}, "
        f"{shell['ltan_upper_h']:.12f}] h"
    )
    for ring in shell["rings"]:
        print(
            f"    {ring['ring_id']}: LTAN={ring['ltan_h']:.12f} h, "
            f"phase offset={ring['phase_offset_deg']:.12f} deg"
        )
    print(
        f"  exact physical phases; {CADENCE_S:.0f} s; one sol; 10,000 m^2; "
        "vacuum; central+J2+Sun reference geometry"
    )
    print("-" * 88)

    ring_irradiance = propagate_exact_rings(shell, epoch_et, duration_s, t_eval_s)
    configurations = assemble_configurations(shell, ring_irradiance, t_eval_s)
    lst_h = write_data_products(
        shell,
        configurations,
        ring_irradiance,
        t_eval_s,
        epoch_et,
    )

    print("\nconfiguration        LTAN_h                         N    peak_W/m2   mean_W/m2  duty_%")
    for configuration in configurations:
        ltan_text = ",".join(f"{value:.6f}" for value in configuration["ltan_h"])
        print(
            f"{configuration['label']:<20s} {ltan_text:<30s} "
            f"{configuration['reflector_count']:>3d} "
            f"{configuration['peak_W_m2']:>11.6f} "
            f"{configuration['mean_W_m2']:>11.6f} "
            f"{100.0 * configuration['duty_fraction']:>7.3f}"
        )
    plot_figure(shell, configurations, lst_h)
    print(f"\ntotal wall = {time.perf_counter() - wall_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
