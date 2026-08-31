"""Render coordinated multishell total- and marginal-fluence figures.

Figure 1 overlays every shell/LTAN series for N=1..120.  A fixed continuous
viridis scale maps LTAN from 16 to 20 h, while line style maps orbital shell.
Figure 2 uses the identical encodings in four shell panels and shows the
ring-to-ring increment F_N-F_(N-1) through N=12.

Without ``--require-complete``, the script discovers available per-LTAN CSVs
and writes conspicuously labelled PARTIAL diagnostics under
``simulation_outputs``.  With ``--require-complete``, it refuses to render
unless all 24 shell/LTAN series are present, then writes the two final figures
under ``figures/manuscript/generated``.

Every N uses a newly uniform phase grid.  The finite difference therefore
includes re-spacing of the existing sails and is not the isolated output of a
single appended sail.
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


SHELL_PRODUCER_NAME = "figure_s4_generate_other_shells.py"
N_MAX_DEFAULT = 120
MARGINAL_N_MAX_DEFAULT = 12
LTAN_COLOR_MIN_H = 16.0
LTAN_COLOR_MAX_H = 20.0
MARGINAL_AXIS_LABEL_FONTSIZE = 30
MARGINAL_TICK_LABEL_FONTSIZE = 26
MARGINAL_LEGEND_FONTSIZE = 28
MARGINAL_COLORBAR_LABEL_FONTSIZE = 30
MARGINAL_COLORBAR_TICK_FONTSIZE = 28
MARGINAL_TITLE_FONTSIZE = 24
MARGINAL_FIGSIZE = (13.0, 7.8)
MARGINAL_Y_LABEL = r"Ring-to-ring fluence increment (J m$^{-2}$)"
MARGINAL_SHOW_GRID = False
MARGINAL_ALL_SIDE_TICKS = True
MARGINAL_SHOW_TITLE = False
MARGINAL_LEGEND_TITLE = None
MARGINAL_LEGEND_NCOL = 2
MARGINAL_LEGEND_LABELSPACING = 0.3
MARGINAL_LEGEND_BORDERPAD = 0.2
MARGINAL_LEGEND_BORDERAXESPAD = 0.2

DATA_DIR = Path("simulation_outputs")
FINAL_DIR = Path("figures/manuscript/generated")
PARTIAL_TOTAL = DATA_DIR / "20260811_multishell_ring_fluence_vs_N_LTAN_PARTIAL.png"
PARTIAL_MARGINAL = (
    DATA_DIR
    / "20260811_multishell_ring_fluence_marginal_N1_12_LTAN_PARTIAL.png"
)
FINAL_TOTAL = FINAL_DIR / "figH5_multishell_ring_fluence_vs_N_LTAN.png"
FINAL_MARGINAL = FINAL_DIR / "figure_s4_ring_fluence.png"


@dataclass(frozen=True)
class PlotConfig:
    k: int
    a_km: float
    ltans_h: tuple[float, ...]
    output_prefix: str
    linestyle: object


LINESTYLES = {
    12: "-",
    11: (0, (7, 3)),
    10: (0, (7, 2, 1.5, 2)),
    9: (0, (1.5, 2)),
}


def _load_shell_producer():
    path = Path(__file__).resolve().with_name(SHELL_PRODUCER_NAME)
    module_name = "_multishell_ring_fluence_producer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import shell producer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _configs(shell_producer, base) -> tuple[PlotConfig, ...]:
    configs = [
        PlotConfig(
            k=12,
            a_km=base.A_REFINED_KM,
            ltans_h=tuple(base.LTANS_H),
            output_prefix=base.OUT_PREFIX,
            linestyle=LINESTYLES[12],
        )
    ]
    for k in (11, 10, 9):
        source = shell_producer.SHELLS[k]
        configs.append(PlotConfig(
            k=k,
            a_km=source.a_km,
            ltans_h=source.ltans_h,
            output_prefix=source.output_prefix,
            linestyle=LINESTYLES[k],
        ))
    return tuple(configs)


def _vsn_path(config: PlotConfig, ltan_h: float, n_max: int) -> Path:
    tag = f"{ltan_h:.1f}".replace(".", "p")
    return DATA_DIR / (
        f"{config.output_prefix}{tag}_Nmax{n_max}_vsN.csv"
    )


def _read_vsn(path: Path, n_max: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "N": float(row["N"]),
                "total_fluence_J_per_m2_per_sol": float(
                    row["total_fluence_J_per_m2_per_sol"]
                ),
                "difference_from_previous_N_J_per_m2_per_sol": (
                    math.nan
                    if not row["difference_from_previous_N_J_per_m2_per_sol"]
                    else float(
                        row["difference_from_previous_N_J_per_m2_per_sol"]
                    )
                ),
            })
    if [int(row["N"]) for row in rows] != list(range(1, n_max + 1)):
        raise RuntimeError(f"{path} does not contain N=1..{n_max} exactly")
    return rows


def _load_available(configs: tuple[PlotConfig, ...], n_max: int):
    data: dict[int, dict[float, list[dict[str, float]]]] = {}
    missing: list[tuple[int, float, Path]] = []
    for config in configs:
        data[config.k] = {}
        for ltan_h in config.ltans_h:
            path = _vsn_path(config, ltan_h, n_max)
            if path.exists():
                data[config.k][ltan_h] = _read_vsn(path, n_max)
            else:
                missing.append((config.k, ltan_h, path))
    return data, missing


def _increment(rows: list[dict[str, float]]) -> np.ndarray:
    total = np.array(
        [row["total_fluence_J_per_m2_per_sol"] for row in rows],
        dtype=float,
    )
    stored = np.array(
        [row["difference_from_previous_N_J_per_m2_per_sol"] for row in rows],
        dtype=float,
    )
    delta = np.diff(total, prepend=0.0)
    if not math.isnan(stored[0]):
        raise RuntimeError("expected blank stored finite difference at N=1")
    if not np.allclose(stored[1:], delta[1:], rtol=0.0, atol=1e-9):
        raise RuntimeError("stored finite differences disagree with totals")
    return delta


def _context(require_complete: bool, available: int, requested: int) -> str:
    if require_complete:
        return ""
    return f"PARTIAL ({available}/{requested} shell/LTAN cases): "


def render(
    n_max: int,
    marginal_n_max: int,
    require_complete: bool,
) -> tuple[Path, Path]:
    shell_producer = _load_shell_producer()
    base, _, _ = shell_producer._load_base_producer()
    configs = _configs(shell_producer, base)
    data, missing = _load_available(configs, n_max)
    requested = sum(len(config.ltans_h) for config in configs)
    available = requested - len(missing)
    if require_complete and missing:
        details = ", ".join(
            f"K{k}/LTAN{ltan:g}" for k, ltan, _ in missing
        )
        raise FileNotFoundError(
            f"final render requires all {requested} series; missing {details}"
        )
    if available == 0:
        raise FileNotFoundError("no completed shell/LTAN CSVs are available")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    })
    cmap = plt.cm.viridis
    norm = Normalize(vmin=LTAN_COLOR_MIN_H, vmax=LTAN_COLOR_MAX_H)
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    total_path = FINAL_TOTAL if require_complete else PARTIAL_TOTAL
    marginal_path = FINAL_MARGINAL if require_complete else PARTIAL_MARGINAL
    total_path.parent.mkdir(parents=True, exist_ok=True)
    marginal_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = _context(require_complete, available, requested)

    fig, ax = plt.subplots(figsize=(13.0, 7.8), layout="constrained")
    for config in configs:
        for ltan_h, rows in data[config.k].items():
            n = np.array([int(row["N"]) for row in rows])
            total = np.array(
                [row["total_fluence_J_per_m2_per_sol"] for row in rows]
            )
            ax.plot(
                n,
                total / 1000.0,
                color=cmap(norm(ltan_h)),
                linestyle=config.linestyle,
                lw=2.2,
                alpha=0.9,
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
    shell_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            lw=2.5,
            linestyle=config.linestyle,
            label=(
                f"K={config.k} "
                f"({config.a_km - base.R_MARS_KM:.0f} km)"
            ),
        )
        for config in configs
    ]
    ax.legend(
        handles=shell_handles,
        title="Orbital shell (altitude)",
        loc="lower right",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
        ncol=2,
    )
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.015)
    colorbar.set_label("Local time of ascending node (h)", fontsize=14)
    colorbar.ax.tick_params(labelsize=11)
    fig.suptitle(
        f"{prefix}fluence from evenly spaced rings across four Mars "
        "orbital shells\n"
        r"Mars perihelion; 10,000 m$^2$ sails; $M_{0,j}=360^\circ j/N$; "
        r"40$^\circ$N, 200$^\circ$E target; no atmospheric losses",
        fontsize=17,
    )
    fig.savefig(total_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=MARGINAL_FIGSIZE, layout="constrained")
    for config in configs:
        available_shell = data[config.k]
        for ltan_h, rows in available_shell.items():
            n = np.array([int(row["N"]) for row in rows])
            early = n <= marginal_n_max
            ax.plot(
                n[early],
                _increment(rows)[early],
                color=cmap(norm(ltan_h)),
                linestyle=config.linestyle,
                lw=2.0,
                alpha=0.9,
            )
    ax.set_xlim(1, marginal_n_max)
    ax.set_xticks(np.arange(1, marginal_n_max + 1, 2))
    ax.set_xlabel(
        "Number of sails in ring, N",
        fontsize=MARGINAL_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        MARGINAL_Y_LABEL,
        fontsize=MARGINAL_AXIS_LABEL_FONTSIZE,
    )
    if MARGINAL_ALL_SIDE_TICKS:
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=7,
            width=1.1,
            labelsize=MARGINAL_TICK_LABEL_FONTSIZE,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=4,
            width=0.9,
        )
    else:
        ax.tick_params(labelsize=MARGINAL_TICK_LABEL_FONTSIZE)
    if MARGINAL_SHOW_GRID:
        ax.grid(alpha=0.22)
    marginal_shell_handles = []
    for config in configs:
        available_shell = data[config.k]
        coverage = (
            ""
            if require_complete
            else f"; {len(available_shell)}/{len(config.ltans_h)} LTANs"
        )
        marginal_shell_handles.append(
            Line2D(
                [0],
                [0],
                color="0.2",
                lw=2.5,
                linestyle=config.linestyle,
                label=(
                    rf"$K_{{{config.k}}}$ "
                    f"({config.a_km - base.R_MARS_KM:.0f} km{coverage})"
                ),
            )
        )
    ax.legend(
        handles=marginal_shell_handles,
        title=MARGINAL_LEGEND_TITLE,
        loc="upper right",
        fontsize=MARGINAL_LEGEND_FONTSIZE,
        title_fontsize=MARGINAL_LEGEND_FONTSIZE,
        frameon=False,
        ncol=MARGINAL_LEGEND_NCOL,
        labelspacing=MARGINAL_LEGEND_LABELSPACING,
        borderpad=MARGINAL_LEGEND_BORDERPAD,
        borderaxespad=MARGINAL_LEGEND_BORDERAXESPAD,
    )
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.015)
    colorbar.set_label(
        "Local time of ascending node (h)",
        fontsize=MARGINAL_COLORBAR_LABEL_FONTSIZE,
    )
    colorbar.ax.tick_params(labelsize=MARGINAL_COLORBAR_TICK_FONTSIZE)
    if MARGINAL_SHOW_TITLE:
        fig.suptitle(
            f"{prefix}marginal fluence across four Mars orbital shells\n"
            r"$F_0=0$; all reflector phases re-spaced at each N; Mars "
            r"perihelion; no atmospheric losses",
            fontsize=MARGINAL_TITLE_FONTSIZE,
        )
    fig.savefig(marginal_path, dpi=200)
    plt.close(fig)

    print(
        f"coverage={available}/{requested}; "
        f"missing={[f'K{k}/LTAN{ltan:g}' for k, ltan, _ in missing]}"
    )
    for config in configs:
        shell_data = data[config.k]
        if not shell_data:
            print(f"K={config.k}: no complete LTAN cases")
            continue
        terminal = np.array([
            rows[-1]["total_fluence_J_per_m2_per_sol"]
            for rows in shell_data.values()
        ])
        marginal_values = np.concatenate([
            _increment(rows)[1:marginal_n_max]
            for rows in shell_data.values()
        ])
        print(
            f"K={config.k}: LTANs={list(shell_data)}; "
            f"N{n_max}={terminal.min():.6f}..{terminal.max():.6f}; "
            f"DeltaF N2-{marginal_n_max}="
            f"{marginal_values.min():.6f}..{marginal_values.max():.6f}"
        )
    print(f"wrote {total_path}")
    print(f"wrote {marginal_path}")
    return total_path, marginal_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-max", type=int, default=N_MAX_DEFAULT)
    parser.add_argument(
        "--marginal-n-max",
        type=int,
        default=MARGINAL_N_MAX_DEFAULT,
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.n_max < 1:
        raise ValueError("--n-max must be >=1")
    if not 2 <= args.marginal_n_max <= args.n_max:
        raise ValueError("--marginal-n-max must be in [2, n-max]")
    render(args.n_max, args.marginal_n_max, args.require_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
