"""K11/K10/K9 single-ring fluence versus sail count and LTAN.

This extends the K12 producer to the closure-refined K11, K10, and K9 shell
design points and selected LTAN lists. All orbit,
photometry, delivery-gate, cadence, force-model, phase-union, cache, and
vacuum-transmission choices remain those of the K12 calculation.

For every shell/LTAN case, each integer ring size N=1..120 is assembled from
the exact uniformly spaced phases M0_j = 360 deg j/N.  The reduced-rational
union contains 4,386 distinct phases per case.  One Midway array element
computes one of the 18 shell/LTAN cases.

After all cases finish, ``--plot-only`` writes two figures per shell:

* total delivered fluence over one sol for N=1..120; and
* the ring-to-ring fluence increment F_N-F_(N-1) for N=1..12, exposing
  phase-grid bias and convergence.

Every N uses a newly uniform phase grid.  The finite difference therefore
includes re-spacing of the existing sails; it is not the isolated output of a
new sail appended while the earlier phases remain fixed.

"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


BASE_PRODUCER_NAME = "figure_s4_generate_k12.py"
N_MAX_DEFAULT = 120
EARLY_N_MAX = 12


@dataclass(frozen=True)
class ShellConfig:
    k: int
    a_km: float
    inclination_deg: float
    ltans_h: tuple[float, ...]

    @property
    def output_prefix(self) -> str:
        return f"20260811_K{self.k}_ring_fluence_N1_120_LTAN"

    @property
    def total_figure(self) -> Path:
        return Path(
            f"figures/base_power/figH4_K{self.k}_ring_fluence_vs_N_LTAN.png"
        )

    @property
    def early_figure(self) -> Path:
        return Path(
            "figures/base_power/"
            f"figH4_K{self.k}_ring_fluence_marginal_N1_12_LTAN.png"
        )


SHELLS = {
    11: ShellConfig(
        k=11,
        a_km=4136.808474,
        inclination_deg=94.204104,
        ltans_h=(16.8, 17.0, 17.5, 18.0, 18.5, 19.1),
    ),
    10: ShellConfig(
        k=10,
        a_km=4407.918053,
        inclination_deg=95.277366,
        ltans_h=(16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5),
    ),
    9: ShellConfig(
        k=9,
        a_km=4728.393153,
        inclination_deg=96.799160,
        ltans_h=(16.0, 17.0, 18.0, 19.0, 20.0),
    ),
}

CASES = tuple(
    (config, ltan_index, ltan_h)
    for config in SHELLS.values()
    for ltan_index, ltan_h in enumerate(config.ltans_h)
)


def _load_base_producer():
    path = Path(__file__).resolve().with_name(BASE_PRODUCER_NAME)
    module_name = "_k12_ring_fluence_base"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import base producer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, path, None


def _configure_base(base, config: ShellConfig) -> None:
    base.K = config.k
    base.A_REFINED_KM = config.a_km
    base.I_REFINED_DEG = config.inclination_deg
    base.LTANS_H = config.ltans_h
    base.OUT_PREFIX = config.output_prefix
    base._WORKER_PID = None
    base._WORKER_SAIL = None


def _write_combined_csv(base, config: ShellConfig, n_max: int, per_ltan) -> Path:
    rows: list[list[object]] = []
    for ltan_h in config.ltans_h:
        for row in per_ltan[ltan_h]:
            delta = row["difference_from_previous_N_J_per_m2_per_sol"]
            rows.append([
                config.k,
                f"{ltan_h:.6f}",
                int(row["N"]),
                int(row["total_windows_per_sol"]),
                f"{row['total_fluence_J_per_m2_per_sol']:.12f}",
                f"{row['fluence_per_sail_J_per_m2_per_sol']:.12f}",
                "" if math.isnan(delta) else f"{delta:.12f}",
            ])
    path = base._combined_csv_path(n_max)
    base._atomic_csv(
        path,
        [
            "K",
            "ltan_h",
            "N",
            "total_windows_per_sol",
            "total_fluence_J_per_m2_per_sol",
            "fluence_per_sail_J_per_m2_per_sol",
            "difference_from_previous_N_J_per_m2_per_sol",
        ],
        rows,
    )
    return path


def _diagnostics(config: ShellConfig, per_ltan) -> None:
    print(f"K={config.k} DIAGNOSTICS (J/m^2/sol):")
    for ltan_h in config.ltans_h:
        rows = per_ltan[ltan_h]
        n = np.array([int(row["N"]) for row in rows], dtype=int)
        fluence = np.array(
            [row["total_fluence_J_per_m2_per_sol"] for row in rows]
        )
        terminal_slope = fluence[-1] / n[-1]
        relative = np.abs(fluence - n * terminal_slope) / (
            n * terminal_slope
        )
        linear_from = None
        for index in range(n.size):
            if np.all(relative[index:] <= 0.01):
                linear_from = int(n[index])
                break
        negative_steps = int(np.count_nonzero(np.diff(fluence) < 0.0))
        print(
            f"  LTAN={ltan_h:4.1f}: N1={fluence[0]:.6f}; "
            f"N{n[-1]}={fluence[-1]:.6f}; terminal_per_sail="
            f"{terminal_slope:.6f}; linear_within_1pct_from_N="
            f"{linear_from}; negative_steps={negative_steps}"
        )


def _plot_shell(base, config: ShellConfig, n_max: int) -> None:
    _configure_base(base, config)
    per_ltan = base._read_all_vsn(n_max)
    combined_path = _write_combined_csv(base, config, n_max, per_ltan)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    })
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(config.ltans_h)))
    altitude_km = config.a_km - base.R_MARS_KM

    fig, ax = plt.subplots(figsize=(12.5, 7.8), layout="constrained")
    for color, ltan_h in zip(colors, config.ltans_h):
        rows = per_ltan[ltan_h]
        n = np.array([int(row["N"]) for row in rows])
        fluence_kj = np.array(
            [row["total_fluence_J_per_m2_per_sol"] for row in rows]
        ) / 1000.0
        ax.plot(
            n,
            fluence_kj,
            color=color,
            lw=2.0,
            label=f"LTAN = {ltan_h:g} h",
        )
    ax.set_xlim(1, n_max)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Number of sails in ring, N", fontsize=18)
    ax.set_ylabel(
        r"Delivered fluence over one sol (kJ m$^{-2}$)",
        fontsize=18,
    )
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=12, frameon=False, ncol=2)
    fig.suptitle(
        f"K={config.k} ring ({altitude_km:.0f} km): fluence versus evenly "
        "spaced sails at Mars perihelion\n"
        r"10,000 m$^2$ sails; $M_{0,j}=360^\circ j/N$; 40$^\circ$N, "
        r"200$^\circ$E target; no atmospheric losses",
        fontsize=17,
    )
    config.total_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.total_figure, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 7.8), layout="constrained")
    for color, ltan_h in zip(colors, config.ltans_h):
        rows = per_ltan[ltan_h]
        n = np.array([int(row["N"]) for row in rows])
        total = np.array(
            [row["total_fluence_J_per_m2_per_sol"] for row in rows]
        )
        increment = np.diff(total, prepend=0.0)
        early = n <= EARLY_N_MAX
        ax.plot(
            n[early],
            increment[early],
            color=color,
            lw=2.0,
            marker="o",
            markersize=5.0,
            label=f"LTAN = {ltan_h:g} h",
        )
    ax.set_xlim(1, EARLY_N_MAX)
    ax.set_xticks(np.arange(1, EARLY_N_MAX + 1))
    ax.set_xlabel("Number of sails in ring, N", fontsize=18)
    ax.set_ylabel(
        r"Ring-to-ring fluence increment $F_N-F_{N-1}$ "
        r"(J m$^{-2}$)",
        fontsize=18,
    )
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=12, frameon=False, ncol=2)
    fig.suptitle(
        f"K={config.k} ring ({altitude_km:.0f} km): early ring-to-ring "
        "fluence increments at Mars perihelion\n"
        r"10,000 m$^2$ sails; $M_{0,j}=360^\circ j/N$; 40$^\circ$N, "
        r"200$^\circ$E target; all phases re-spaced at each N; vacuum delivery",
        fontsize=17,
    )
    fig.savefig(config.early_figure, dpi=200)
    plt.close(fig)

    print(f"wrote {combined_path}")
    print(f"wrote {config.total_figure}")
    print(f"wrote {config.early_figure}")
    _diagnostics(config, per_ltan)


def _self_test(base) -> None:
    base.self_test()
    assert len(CASES) == 18
    assert len(base.phase_union(N_MAX_DEFAULT)) == 4_386
    assert tuple(SHELLS) == (11, 10, 9)
    assert SHELLS[11].ltans_h == (16.8, 17.0, 17.5, 18.0, 18.5, 19.1)
    assert SHELLS[10].ltans_h == (
        16.5,
        17.0,
        17.5,
        18.0,
        18.5,
        19.0,
        19.5,
    )
    assert SHELLS[9].ltans_h == (16.0, 17.0, 18.0, 19.0, 20.0)
    print("PASS: 18-case shell/LTAN map and two-figure configuration")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=N_MAX_DEFAULT)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--shell-k", type=int, choices=tuple(SHELLS))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    if args.n_max < 1:
        raise ValueError("--n-max must be >=1")
    base, base_path, base_sha = _load_base_producer()

    if args.self_test:
        _self_test(base)
        return 0
    if args.plot_only:
        configs = (
            (SHELLS[args.shell_k],)
            if args.shell_k is not None
            else tuple(SHELLS.values())
        )
        for config in configs:
            _plot_shell(base, config, args.n_max)
        return 0
    if args.case_index is None:
        raise ValueError("--case-index is required for computation")

    config, ltan_index, ltan_h = CASES[args.case_index]
    _configure_base(base, config)
    base.configure_multiprocessing_for_spice()
    base.load_kernels()
    print("=" * 80)
    print("K11/K10/K9 RING FLUENCE VS N AND LTAN")
    print("=" * 80)
    print(f"base_producer={base_path}; sha256={base_sha}")
    print(
        f"case_index={args.case_index}; K={config.k}; "
        f"LTAN index={ltan_index}; LTAN={ltan_h:.1f} h"
    )
    print(
        f"N=1..{args.n_max}; exact phase union="
        f"{len(base.phase_union(args.n_max))}"
    )
    print(
        f"workers={args.workers}; epoch={base.PERIHELION_UTC}; "
        f"cadence={base.CADENCE_S} s"
    )
    print(
        f"a*={config.a_km:.6f} km; i*={config.inclination_deg:.6f} deg; "
        f"area={base.SAIL_AREA_M2:.0f} m^2; "
        f"sigma={base.SIGMA_KG_PER_M2}; chi=1"
    )
    base.compute_ltan(ltan_h, args.n_max, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
