"""K12 single-ring fluence versus sail count for six perihelion LTANs.

The calculation uses the closure-refined orbit, a 10,000 m^2 reflector, the
one-sol delivery model, production gates, and full force model. It evaluates
every integer ring size N=1..120 at each selected LTAN.

For a ring of N sails, sail j starts at exactly

    M0_j = 360 deg * j / N,  j = 0, ..., N-1.

The union of all required phases is represented by reduced ``Fraction`` keys,
not rounded floating-point angles.  Each distinct single-sail phase is
propagated once per LTAN.  Total ring fluence is then the exact sum of the
independently gated single-sail window fluences.  This is valid because the
Canady--Allen delivered irradiance model is linear in simultaneous optical
contributions; overlap changes instantaneous irradiance but not its integral.

The full union is large, so one Midway array task computes one LTAN.  Results
are written atomically as per-phase and per-N CSVs.  ``--plot-only`` combines
the six completed per-N CSVs and renders the tracked figure.

Physics inherited from figH3
----------------------------
* K12 (a*, i*) = (3903.924477 km, 93.420985 deg), approximately 508 km altitude.
* Epoch 2028-02-11T12:42:00 UTC (Mars perihelion), one Mars solar sol.
* sigma=0.018 kg/m^2, reflector area=10,000 m^2.
* Mars gravity 6x6, Sun/Phobos/Deimos third bodies, and SRP.
* 30 s output cadence; elevation >=10 deg; bisector feasibility;
  sail sunlit; alpha_max=0.003 deg/s^2; minimum useful-window fluence
  1 J/m^2; atmospheric transmission chi=1 (vacuum result).

The outer requested LTANs (17 and 19 h) are outside K12's *whole-year*
eclipse-free band, but the experiment is explicitly fixed at perihelion.
Eclipse loss at that epoch remains active through ``require_sail_sunlit``.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import logging
import math
import os
from pathlib import Path
import time

import numpy as np

from reflectors.attitude import sun_pointing
from reflectors.dynamics import PropagationOptions, propagate
from reflectors.elements import state_from_classical_mme2000
from reflectors.ephemeris import utc_to_et
from reflectors.gravity import mars_gravity_model
from reflectors.kernels import load_kernels
from reflectors.mars_constants import SECONDS_PER_SOLAR_SOL_S
from reflectors.parallel import CloudpickleMap, configure_multiprocessing_for_spice
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import raan_mme2000_from_ltan
from reflectors.termination import AltitudeFloor
from reflectors.third_body import (
    deimos_third_body,
    phobos_third_body,
    sun_third_body,
)
from reflectors.visibility import find_delivery_windows


logger = logging.getLogger(__name__)

TARGET_LAT_DEG = 40.0
TARGET_LON_DEG = 200.0
SIGMA_KG_PER_M2 = 0.018
SAIL_AREA_M2 = 10_000.0

K = 12
A_REFINED_KM = 3903.924477
I_REFINED_DEG = 93.420985
PERIHELION_UTC = "2028-02-11T12:42:00"
LTANS_H = (17.0, 17.5, 17.9, 18.1, 18.5, 19.0)

N_MAX_DEFAULT = 120
CADENCE_S = 30.0
ELEV_MIN_DEG = 10.0
BISECTOR_COS_ALPHA_MIN = 0.1
ALPHA_MAX_RAD_S2 = math.radians(0.003)
MIN_WINDOW_FLUENCE_J_PER_M2 = 1.0
ATMOSPHERIC_TRANSMISSION = 1.0
ALT_FLOOR_KM = 300.0

_GRAV2 = mars_gravity_model(max_degree=2)
_MU_KM3_S2 = float(_GRAV2.mu_km3_s2)
R_MARS_KM = float(_GRAV2.ref_radius_km)

OUT_DIR = Path("simulation_outputs")
FIG_DIR = Path("figures/base_power")
OUT_PREFIX = "20260811_K12_ring_fluence_N1_120_LTAN"
FIG_OUT = FIG_DIR / "figH4_K12_ring_fluence_vs_N_LTAN.png"
FIGH3_CACHE = OUT_DIR / "20260627_figH3_K12_10000m2_irr_cache.npz"

_WORKER_PID: int | None = None
_WORKER_SAIL = None


def phase_set(n: int) -> tuple[Fraction, ...]:
    """Exactly evenly spaced orbital phases in turns for an N-sail ring."""
    if n < 1:
        raise ValueError(f"ring size N must be >=1, got {n}")
    return tuple(Fraction(j, n) for j in range(n))


def phase_union(n_max: int) -> tuple[Fraction, ...]:
    """Sorted reduced-rational union needed by all rings N=1..n_max."""
    phases: set[Fraction] = set()
    for n in range(1, n_max + 1):
        phases.update(phase_set(n))
    return tuple(sorted(phases))


def _phase_csv_path(ltan_h: float, n_max: int) -> Path:
    tag = f"{ltan_h:.1f}".replace(".", "p")
    return OUT_DIR / f"{OUT_PREFIX}{tag}_Nmax{n_max}_per_phase.csv"


def _vsn_csv_path(ltan_h: float, n_max: int) -> Path:
    tag = f"{ltan_h:.1f}".replace(".", "p")
    return OUT_DIR / f"{OUT_PREFIX}{tag}_Nmax{n_max}_vsN.csv"


def _combined_csv_path(n_max: int) -> Path:
    return OUT_DIR / f"{OUT_PREFIX}_Nmax{n_max}_all_LTANs.csv"


def _atomic_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _ensure_worker_state() -> None:
    global _WORKER_PID, _WORKER_SAIL
    pid = os.getpid()
    if _WORKER_PID != pid:
        load_kernels()
        _WORKER_SAIL = make_canonical_sail(
            SIGMA_KG_PER_M2, area_m2=SAIL_AREA_M2
        )
        _WORKER_PID = pid


def _t_eval() -> np.ndarray:
    duration_s = float(SECONDS_PER_SOLAR_SOL_S)
    values = np.arange(0.0, duration_s, CADENCE_S)
    if values[-1] < duration_s:
        values = np.append(values, duration_s)
    return values


def phase_worker(task: tuple[float, int, int, float]) -> tuple[int, int, float, int, float]:
    """One exact (LTAN, reduced phase) case; safe for CloudpickleMap."""
    _ensure_worker_state()
    ltan_h, numerator, denominator, epoch_et = task
    phase = Fraction(numerator, denominator)
    m0_deg = 360.0 * float(phase)
    raan_rad = raan_mme2000_from_ltan(ltan_h, epoch_et)
    state0 = state_from_classical_mme2000(
        a_km=A_REFINED_KM,
        e=0.0,
        inclination_rad=math.radians(I_REFINED_DEG),
        raan_rad=raan_rad,
        argp_rad=0.0,
        nu_rad=math.radians(m0_deg),
        mu_km3_s2=_MU_KM3_S2,
        epoch_et=epoch_et,
    )
    duration_s = float(SECONDS_PER_SOLAR_SOL_S)
    result = propagate(
        state0_km_kmps=state0,
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et,
        gravity_degree=6,
        gravity_order=6,
        third_bodies=[sun_third_body(), phobos_third_body(), deimos_third_body()],
        solar_sail=_WORKER_SAIL,
        sail_normal=sun_pointing(),
        altitude_floor=AltitudeFloor.at_km(
            ALT_FLOOR_KM, label="altitude_floor"
        ),
        options=PropagationOptions.fast(),
        t_eval_s=_t_eval(),
    )
    windows = find_delivery_windows(
        result,
        TARGET_LAT_DEG,
        TARGET_LON_DEG,
        target_elevation_min_deg=ELEV_MIN_DEG,
        bisector_cos_alpha_min=BISECTOR_COS_ALPHA_MIN,
        require_sail_sunlit=True,
        require_sail_above_horizon=True,
        require_bisector_feasible=True,
        sail=_WORKER_SAIL,
        atmospheric_transmission=ATMOSPHERIC_TRANSMISSION,
        alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
        min_window_fluence_J_per_m2=MIN_WINDOW_FLUENCE_J_PER_M2,
    )
    fluence = float(sum((window.fluence_J_per_m2 or 0.0) for window in windows))
    return numerator, denominator, m0_deg, len(windows), fluence


def _read_phase_cache(path: Path, expected: tuple[Fraction, ...]) -> dict[Fraction, tuple[int, float]]:
    rows: dict[Fraction, tuple[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            phase = Fraction(int(row["phase_numerator"]), int(row["phase_denominator"]))
            if phase in rows:
                raise RuntimeError(f"duplicate phase {phase} in {path}")
            rows[phase] = (
                int(row["n_windows"]),
                float(row["fluence_J_per_m2_per_sol"]),
            )
    if tuple(sorted(rows)) != expected:
        raise RuntimeError(
            f"phase cache {path} does not match the exact N-union: "
            f"found {len(rows)}, expected {len(expected)} phases"
        )
    return rows


def _write_phase_cache(
    path: Path,
    ltan_h: float,
    results: list[tuple[int, int, float, int, float]],
) -> None:
    rows: list[list[object]] = []
    for numerator, denominator, m0_deg, n_windows, fluence in results:
        rows.append([
            f"{ltan_h:.6f}",
            numerator,
            denominator,
            f"{m0_deg:.12f}",
            n_windows,
            f"{fluence:.12f}",
        ])
    _atomic_csv(
        path,
        [
            "ltan_h",
            "phase_numerator",
            "phase_denominator",
            "m0_deg",
            "n_windows",
            "fluence_J_per_m2_per_sol",
        ],
        rows,
    )


def assemble_vsn(
    ltan_h: float,
    n_max: int,
    phase_data: dict[Fraction, tuple[int, float]],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    previous_total = math.nan
    for n in range(1, n_max + 1):
        phases = phase_set(n)
        windows = int(sum(phase_data[phase][0] for phase in phases))
        total_fluence = float(sum(phase_data[phase][1] for phase in phases))
        delta = total_fluence - previous_total if n > 1 else math.nan
        rows.append({
            "ltan_h": float(ltan_h),
            "N": float(n),
            "total_windows_per_sol": float(windows),
            "total_fluence_J_per_m2_per_sol": total_fluence,
            "fluence_per_sail_J_per_m2_per_sol": total_fluence / n,
            "difference_from_previous_N_J_per_m2_per_sol": delta,
        })
        previous_total = total_fluence
    return rows


def _write_vsn(path: Path, rows: list[dict[str, float]]) -> None:
    output_rows: list[list[object]] = []
    for row in rows:
        delta = row["difference_from_previous_N_J_per_m2_per_sol"]
        output_rows.append([
            f"{row['ltan_h']:.6f}",
            int(row["N"]),
            int(row["total_windows_per_sol"]),
            f"{row['total_fluence_J_per_m2_per_sol']:.12f}",
            f"{row['fluence_per_sail_J_per_m2_per_sol']:.12f}",
            "" if math.isnan(delta) else f"{delta:.12f}",
        ])
    _atomic_csv(
        path,
        [
            "ltan_h",
            "N",
            "total_windows_per_sol",
            "total_fluence_J_per_m2_per_sol",
            "fluence_per_sail_J_per_m2_per_sol",
            "difference_from_previous_N_J_per_m2_per_sol",
        ],
        output_rows,
    )


def compute_ltan(ltan_h: float, n_max: int, workers: int) -> None:
    phases = phase_union(n_max)
    phase_path = _phase_csv_path(ltan_h, n_max)
    vsn_path = _vsn_csv_path(ltan_h, n_max)
    print(f"LTAN {ltan_h:.1f} h: N=1..{n_max}; {len(phases)} exact phases")
    if phase_path.exists():
        phase_data = _read_phase_cache(phase_path, phases)
        print(f"loaded complete phase cache {phase_path}")
    else:
        epoch_et = utc_to_et(PERIHELION_UTC)
        tasks = [
            (ltan_h, phase.numerator, phase.denominator, epoch_et)
            for phase in phases
        ]
        start = time.perf_counter()
        with CloudpickleMap(n_workers=workers) as mapper:
            results = mapper(phase_worker, tasks)
        if len(results) != len(phases):
            raise RuntimeError(
                f"worker result count {len(results)} != phase count {len(phases)}"
            )
        result_phases = tuple(
            Fraction(row[0], row[1]) for row in results
        )
        if result_phases != phases:
            raise RuntimeError("parallel worker results changed exact phase order")
        _write_phase_cache(phase_path, ltan_h, results)
        phase_data = _read_phase_cache(phase_path, phases)
        print(
            f"wrote {phase_path} after {time.perf_counter() - start:.1f} s"
        )

    vsn_rows = assemble_vsn(ltan_h, n_max, phase_data)
    _write_vsn(vsn_path, vsn_rows)
    print(f"wrote {vsn_path}")
    for n in dict.fromkeys((1, 2, 3, 4, 12, 36, 120, n_max)):
        if n <= n_max:
            row = vsn_rows[n - 1]
            print(
                f"  N={n:3d}: fluence="
                f"{row['total_fluence_J_per_m2_per_sol']:.6f} J/m^2/sol; "
                f"per_sail={row['fluence_per_sail_J_per_m2_per_sol']:.6f}; "
                f"windows={int(row['total_windows_per_sol'])}"
            )


def _read_all_vsn(n_max: int) -> dict[float, list[dict[str, float]]]:
    result: dict[float, list[dict[str, float]]] = {}
    for ltan_h in LTANS_H:
        path = _vsn_csv_path(ltan_h, n_max)
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; all six LTAN tasks must finish before plotting"
            )
        rows: list[dict[str, float]] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "ltan_h": float(row["ltan_h"]),
                    "N": float(row["N"]),
                    "total_windows_per_sol": float(row["total_windows_per_sol"]),
                    "total_fluence_J_per_m2_per_sol": float(
                        row["total_fluence_J_per_m2_per_sol"]
                    ),
                    "fluence_per_sail_J_per_m2_per_sol": float(
                        row["fluence_per_sail_J_per_m2_per_sol"]
                    ),
                    "difference_from_previous_N_J_per_m2_per_sol": (
                        math.nan
                        if not row["difference_from_previous_N_J_per_m2_per_sol"]
                        else float(
                            row[
                                "difference_from_previous_N_J_per_m2_per_sol"
                            ]
                        )
                    ),
                })
        if [int(row["N"]) for row in rows] != list(range(1, n_max + 1)):
            raise RuntimeError(f"{path} does not contain N=1..{n_max} exactly")
        result[ltan_h] = rows
    return result


def plot_results(n_max: int) -> None:
    per_ltan = _read_all_vsn(n_max)
    combined_rows: list[list[object]] = []
    for ltan_h in LTANS_H:
        for row in per_ltan[ltan_h]:
            delta = row["difference_from_previous_N_J_per_m2_per_sol"]
            combined_rows.append([
                f"{ltan_h:.6f}",
                int(row["N"]),
                int(row["total_windows_per_sol"]),
                f"{row['total_fluence_J_per_m2_per_sol']:.12f}",
                f"{row['fluence_per_sail_J_per_m2_per_sol']:.12f}",
                "" if math.isnan(delta) else f"{delta:.12f}",
            ])
    combined_path = _combined_csv_path(n_max)
    _atomic_csv(
        combined_path,
        [
            "ltan_h",
            "N",
            "total_windows_per_sol",
            "total_fluence_J_per_m2_per_sol",
            "fluence_per_sail_J_per_m2_per_sol",
            "difference_from_previous_N_J_per_m2_per_sol",
        ],
        combined_rows,
    )
    print(f"wrote {combined_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    })
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(LTANS_H)))
    fig, ax = plt.subplots(figsize=(12.5, 7.8), layout="constrained")
    inset = (
        ax.inset_axes([0.035, 0.51, 0.45, 0.45]) if n_max > 40 else None
    )
    for color, ltan_h in zip(colors, LTANS_H):
        rows = per_ltan[ltan_h]
        n = np.array([int(row["N"]) for row in rows])
        fluence_kj = np.array(
            [row["total_fluence_J_per_m2_per_sol"] for row in rows]
        ) / 1000.0
        label = f"LTAN = {ltan_h:g} h"
        ax.plot(n, fluence_kj, color=color, lw=2.0, label=label)
        if inset is not None:
            early = n <= 12
            fluence_per_sail = np.array(
                [row["fluence_per_sail_J_per_m2_per_sol"] for row in rows]
            )
            inset.plot(
                n[early],
                fluence_per_sail[early],
                color=color,
                lw=1.4,
                marker="o",
                markersize=2.5,
            )

    ax.set_xlim(1, n_max)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Number of sails in ring, N", fontsize=18)
    ax.set_ylabel(
        r"Delivered fluence over one sol (kJ m$^{-2}$)", fontsize=18
    )
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=12, frameon=False, ncol=2)

    if inset is not None:
        inset.set_xlim(1, 12)
        inset.set_title("Early phase bias and convergence (N ≤ 12)", fontsize=12)
        inset.set_xlabel("Sails", fontsize=10)
        inset.set_ylabel(r"J m$^{-2}$ sail$^{-1}$", fontsize=10)
        inset.tick_params(labelsize=9)
        inset.grid(alpha=0.2)

    altitude_km = A_REFINED_KM - R_MARS_KM
    fig.suptitle(
        f"K={K} ring ({altitude_km:.0f} km): fluence versus evenly spaced sails "
        "at Mars perihelion\n"
        r"10,000 m$^2$ sails; $M_{0,j}=360^\circ j/N$; 40$^\circ$N, "
        r"200$^\circ$E target; no atmospheric losses",
        fontsize=17,
    )
    fig.savefig(FIG_OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {FIG_OUT}")

    print("RAW ENDPOINTS (J/m^2/sol):")
    for ltan_h in LTANS_H:
        rows = per_ltan[ltan_h]
        n_all = np.array([int(row["N"]) for row in rows], dtype=int)
        fluence_all = np.array(
            [row["total_fluence_J_per_m2_per_sol"] for row in rows],
            dtype=float,
        )
        for n in dict.fromkeys((1, 2, 3, 12, 120, n_max)):
            if n <= n_max:
                row = rows[n - 1]
                print(
                    f"  LTAN={ltan_h:4.1f} N={n:3d} "
                    f"total={row['total_fluence_J_per_m2_per_sol']:.6f} "
                    f"per_sail={row['fluence_per_sail_J_per_m2_per_sol']:.6f}"
                )
        asymptotic_per_sail = fluence_all[-1] / n_all[-1]
        relative_to_terminal_line = np.abs(
            fluence_all - n_all * asymptotic_per_sail
        ) / (n_all * asymptotic_per_sail)
        linear_from = None
        for index in range(n_all.size):
            if np.all(relative_to_terminal_line[index:] <= 0.01):
                linear_from = int(n_all[index])
                break
        decreasing_steps = n_all[1:][np.diff(fluence_all) < 0.0]
        print(
            f"  LTAN={ltan_h:4.1f} diagnostics: terminal slope="
            f"{asymptotic_per_sail:.6f} J/m^2/sail/sol; "
            f"within 1% of terminal line thereafter from N={linear_from}; "
            f"negative total-fluence steps={decreasing_steps.size}"
        )


def self_test() -> None:
    assert phase_set(1) == (Fraction(0, 1),)
    assert phase_set(2) == (Fraction(0, 1), Fraction(1, 2))
    assert phase_set(3) == (
        Fraction(0, 1),
        Fraction(1, 3),
        Fraction(2, 3),
    )
    assert len(phase_union(120)) == 4_386
    for n_max in (1, 2, 12, 36):
        union = phase_union(n_max)
        lookup = {phase: (2, 7.5) for phase in union}
        rows = assemble_vsn(18.0, n_max, lookup)
        for n, row in enumerate(rows, start=1):
            assert row["total_windows_per_sol"] == 2 * n
            assert row["total_fluence_J_per_m2_per_sol"] == 7.5 * n
    print("PASS: exact phase sets, reduced-rational union, and additive assembly")


def validate_against_figh3(workers: int) -> None:
    """Cross-check scalar window fluence against the established figH3 cache.

    The two paths account for the same kept irradiance differently: this script
    sums each DeliveryWindow's stored fluence, whereas figH3 stores the kept
    30 s irradiance series.  Agreement of their one-sol integrals exercises the
    copied propagation/gating configuration and the additive ring assembly.
    """
    if not FIGH3_CACHE.exists():
        raise FileNotFoundError(f"established figH3 cache missing: {FIGH3_CACHE}")
    epoch_et = utc_to_et(PERIHELION_UTC)
    phases = phase_set(4)
    tasks = [
        (18.0, phase.numerator, phase.denominator, epoch_et)
        for phase in phases
    ]
    with CloudpickleMap(n_workers=workers) as mapper:
        results = mapper(phase_worker, tasks)

    cache = np.load(FIGH3_CACHE)
    cached_phases = np.asarray(cache["phases"], dtype=float)
    cached_t = np.asarray(cache["t_eval"], dtype=float)
    cached_irr = np.asarray(cache["irr"], dtype=float)
    show = cached_t <= float(SECONDS_PER_SOLAR_SOL_S)

    scalar_total = 0.0
    cached_total = 0.0
    for result in results:
        _num, _den, m0_deg, _n_windows, scalar_fluence = result
        matches = np.flatnonzero(np.isclose(cached_phases, m0_deg, atol=1e-12))
        if matches.size != 1:
            raise RuntimeError(
                f"figH3 cache has {matches.size} matches for M0={m0_deg} deg"
            )
        cached_fluence = float(
            np.trapezoid(cached_irr[int(matches[0]), show], cached_t[show])
        )
        rel = abs(scalar_fluence - cached_fluence) / max(
            abs(scalar_fluence), 1e-12
        )
        print(
            f"M0={m0_deg:6.1f} deg: window_sum={scalar_fluence:.9f}, "
            f"figH3_series_integral={cached_fluence:.9f}, rel={rel:.3e}"
        )
        if rel > 0.03:
            raise AssertionError(
                f"M0={m0_deg} figH3 fluence disagreement {rel:.3%} > 3%"
            )
        scalar_total += scalar_fluence
        cached_total += cached_fluence

    total_rel = abs(scalar_total - cached_total) / max(abs(scalar_total), 1e-12)
    if total_rel > 0.03:
        raise AssertionError(f"N=4 additive fluence disagreement {total_rel:.3%}")
    print(
        f"PASS: N=4 scalar sum={scalar_total:.9f}, figH3 integrated sum="
        f"{cached_total:.9f}, rel={total_rel:.3e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltan-index", type=int, choices=range(len(LTANS_H)))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=N_MAX_DEFAULT)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-existing-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_max < 1:
        raise ValueError(f"--n-max must be >=1, got {args.n_max}")
    if args.workers < 1:
        raise ValueError(f"--workers must be >=1, got {args.workers}")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        self_test()
        return 0
    if args.plot_only:
        plot_results(args.n_max)
        return 0
    if args.validate_existing_cache:
        configure_multiprocessing_for_spice()
        load_kernels()
        validate_against_figh3(args.workers)
        return 0
    if args.ltan_index is None:
        raise ValueError("--ltan-index is required for computation")

    configure_multiprocessing_for_spice()
    load_kernels()
    ltan_h = LTANS_H[args.ltan_index]
    print("=" * 80)
    print("K12 RING FLUENCE VS N AND LTAN")
    print("=" * 80)
    print(f"LTAN index={args.ltan_index}; LTAN={ltan_h:.1f} h")
    print(f"N=1..{args.n_max}; exact phase union={len(phase_union(args.n_max))}")
    print(f"workers={args.workers}; epoch={PERIHELION_UTC}; cadence={CADENCE_S} s")
    print(
        f"a*={A_REFINED_KM:.6f} km; i*={I_REFINED_DEG:.6f} deg; "
        f"area={SAIL_AREA_M2:.0f} m^2; sigma={SIGMA_KG_PER_M2}; chi=1"
    )
    compute_ltan(ltan_h, args.n_max, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
