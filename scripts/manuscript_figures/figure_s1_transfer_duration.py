#!/usr/bin/env python
"""Plot Earth-escape and circulation-corrected Mars-capture times.

The Mars inputs are the uniform-blended, corrected-circulation six-season
campaign:

* the completed cells from the sigma={15,20,25,30,35,40,45,50}, K9/K12 grid at
  six arrival seasons spaced by exactly 60 degrees in Mars solar longitude;
  and
* the five exact K12 phase-0 pilot cells at sigma=15--35 that the production
  launcher deliberately reused rather than recomputed.

Earth values are reconstructed from canonical-epoch raw summary files rather
than from the earlier aggregate CSV.  Points are means over successful runs;
whiskers are +/- one sample standard deviation and require at least two
successful epochs.  An incomplete Mars grid is allowed: unavailable cells do
not enter the statistic.  All plotted durations are ordinary 86,400-second
Earth days; the Mars summary parser selects the converted day value rather than
the preceding value in Martian sols.

"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "simulation_outputs" / "midway_results"
EARTH_DIR = RESULTS / "earth_escape"
MARS_GRID_DIR = RESULTS / "marscap_sigma6_equalLs_out"
MARS_PILOT_DIR = RESULTS / "marscap_k12_sigma_pilot_out"
OUT = REPO / "figures" / "manuscript" / "generated" / "figure_s1_transfer_duration.png"

EARTH_ANCHOR = dt.date(2031, 1, 21)
N_EARTH_EPOCHS = 52
PLOT_SIGMAS = (15, 20, 25, 30, 35, 40, 45, 50)
MIN_SIGMA = min(PLOT_SIGMAS)
MAX_SIGMA = max(PLOT_SIGMAS)
MARS_SIGMAS = PLOT_SIGMAS
N_MARS_PHASES = 6
SHELL_ALTITUDE_KM = {"K9": 1332.39, "K12": 507.92}

EARTH_CELL_RE = re.compile(r"_mw_exitphase_e(\d+)_s(\d+)_summary\.txt$")
MARS_GRID_CELL_RE = re.compile(
    r"_sigma6Ls_mw_marscap_p(\d+)_s(\d+)_(K9|K12)_summary\.txt$"
)
MARS_PILOT_CELL_RE = re.compile(
    r"_k12sigpilot_mw_marscap_e(\d+)_s(\d+)_(K12)_summary\.txt$"
)

TIMES_NEW_ROMAN = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
if TIMES_NEW_ROMAN.exists():
    font_manager.fontManager.addfont(TIMES_NEW_ROMAN)
    PLOT_FONT = "Times New Roman"
else:
    PLOT_FONT = "serif"
plt.rcParams.update(
    {
        "font.family": PLOT_FONT,
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
    }
)


def _grab(pattern: str, text: str, *, default=None):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else default


def _parse_summary(
    path: Path, *, campaign: str, cell_re: re.Pattern[str] | None = None
) -> dict:
    """Read the fields needed for the time plot from one raw summary."""
    text = path.read_text()
    pattern = EARTH_CELL_RE if campaign == "earth" else cell_re
    if pattern is None:
        raise ValueError("Mars summary parsing requires an explicit filename pattern")
    match = pattern.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized {campaign} summary filename: {path.name}")

    event_days = _grab(
        r"^event time\s*:\s*(?:[\d.]+\s+sols\s*=\s*)?([\d.]+)\s+days",
        text,
    )
    escaped = _grab(r"^escaped \(Hill\)\s*:\s*(True|False)", text)
    if escaped is None:
        raise ValueError(f"Missing Hill outcome in {path}")
    if escaped == "True" and event_days is None:
        raise ValueError(f"Successful run is missing its event time in {path}")

    record = {
        "epoch_idx": int(match.group(1)),
        "sigma": int(match.group(2)),
        "successful": escaped == "True",
        "time_days": float(event_days) if event_days is not None else np.nan,
        "epoch": _grab(r"^epoch\s*:\s*(\S+)", text),
        "source": path.name,
    }
    if campaign == "mars":
        record["shell"] = match.group(3)
    return record


def load_earth() -> pd.DataFrame:
    """Load one newest canonical raw summary per Earth (epoch, sigma) cell."""
    records = []
    for path in sorted(EARTH_DIR.glob("*mw_exitphase*_summary.txt")):
        match = EARTH_CELL_RE.search(path.name)
        if match is None:
            continue
        sigma = int(match.group(2))
        epoch_idx = int(match.group(1))
        if (
            sigma not in PLOT_SIGMAS
            or not 0 <= epoch_idx < N_EARTH_EPOCHS
        ):
            continue
        record = _parse_summary(path, campaign="earth")
        expected = EARTH_ANCHOR + dt.timedelta(days=7 * epoch_idx)
        record["canonical_epoch"] = record["epoch"][:10] == expected.isoformat()
        records.append(record)

    frame = pd.DataFrame(records)
    canonical = frame[frame.canonical_epoch].copy()
    # Filenames begin with a sortable run timestamp.  The final record is the
    # newest deterministic rerun when a canonical cell was repeated.
    canonical = canonical.sort_values("source").drop_duplicates(
        ["epoch_idx", "sigma"], keep="last"
    )
    if canonical.empty:
        raise RuntimeError("No canonical Earth-escape summaries were found")
    if canonical.duplicated(["epoch_idx", "sigma"]).any():
        raise AssertionError("Earth campaign cell deduplication failed")
    canonical["series"] = "Earth"
    return canonical


def load_corrected_mars() -> tuple[pd.DataFrame, set[tuple[str, int, int]]]:
    """Load the available six-season grid plus five exact pilot anchors."""
    grid_paths = sorted(MARS_GRID_DIR.glob("*sigma6Ls*_summary.txt"))
    if not grid_paths:
        raise RuntimeError(f"No equal-Ls Mars summaries found in {MARS_GRID_DIR}")
    grid = pd.DataFrame(
        _parse_summary(path, campaign="mars", cell_re=MARS_GRID_CELL_RE)
        for path in grid_paths
    )

    pilot_paths = sorted(MARS_PILOT_DIR.glob("*k12sigpilot*_summary.txt"))
    pilot = pd.DataFrame(
        _parse_summary(path, campaign="mars", cell_re=MARS_PILOT_CELL_RE)
        for path in pilot_paths
    )
    reusable_sigmas = set(MARS_SIGMAS[:5])
    pilot = pilot[
        (pilot.shell == "K12")
        & (pilot.epoch_idx == 0)
        & pilot.sigma.isin(reusable_sigmas)
        & pilot.successful
    ].copy()
    expected_reuse = {("K12", 0, sigma) for sigma in reusable_sigmas}
    actual_reuse = set(zip(pilot.shell, pilot.epoch_idx, pilot.sigma))
    if actual_reuse != expected_reuse or len(pilot) != len(expected_reuse):
        raise AssertionError(
            "Exact K12 phase-0 pilot reuse mismatch: "
            f"missing={sorted(expected_reuse - actual_reuse)}, "
            f"extra={sorted(actual_reuse - expected_reuse)}"
        )

    grid["provenance"] = "20260817 equal-Ls production"
    pilot["provenance"] = "20260815 exact K12 phase-0 pilot reuse"
    combined = pd.concat([grid, pilot], ignore_index=True)
    key_columns = ["shell", "epoch_idx", "sigma"]
    if combined.duplicated(key_columns).any():
        duplicates = combined[combined.duplicated(key_columns, keep=False)]
        raise AssertionError(
            "Equal-Ls Mars campaign contains duplicate cells:\n"
            + duplicates[key_columns + ["source"]].to_string(index=False)
        )

    expected = {
        (shell, phase, sigma)
        for shell in SHELL_ALTITUDE_KM
        for sigma in MARS_SIGMAS
        for phase in range(N_MARS_PHASES)
    }
    actual = set(zip(combined.shell, combined.epoch_idx, combined.sigma))
    extra = actual - expected
    if extra:
        raise AssertionError(f"Unexpected cells in equal-Ls Mars inputs: {sorted(extra)}")
    missing = expected - actual
    combined["series"] = combined.shell
    return combined, missing


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """Return attempted/success counts and successful-run mean/sample SD."""
    rows = []
    for (series, sigma), group in records.groupby(["series", "sigma"]):
        successful = group[group.successful]
        n_success = len(successful)
        rows.append(
            {
                "series": series,
                "sigma": int(sigma),
                "n_attempted": len(group),
                "n_success": n_success,
                "mean_days": successful.time_days.mean() if n_success else np.nan,
                "sd_days": successful.time_days.std(ddof=1) if n_success >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["series", "sigma"])


def plot_time(stats: pd.DataFrame) -> None:
    styles = {
        "Earth": {
            "label": "Earth escape from 800 km altitude",
            "color": "#1f4e79",
            "marker": "o",
        },
        "K12": {
            "label": f"Mars capture to $K_{{12}}$ ({SHELL_ALTITUDE_KM['K12']:,.0f} km altitude)",
            "color": "#2b8c5a",
            "marker": "s",
        },
        "K9": {
            "label": f"Mars capture to $K_9$ ({SHELL_ALTITUDE_KM['K9']:,.0f} km altitude)",
            "color": "#d06b2c",
            "marker": "^",
        },
    }

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    for series in ("Earth", "K12", "K9"):
        values = stats[(stats.series == series) & stats.mean_days.notna()]
        style = styles[series]
        ax.plot(
            values.sigma,
            values.mean_days,
            color=style["color"],
            marker=style["marker"],
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            label=style["label"],
            zorder=3,
        )
        with_whiskers = values[values.sd_days.notna()]
        ax.errorbar(
            with_whiskers.sigma,
            with_whiskers.mean_days,
            yerr=with_whiskers.sd_days,
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.8,
            capsize=5,
            capthick=1.8,
            zorder=2,
        )

    ax.set_xlabel("Sail areal density, σ (g/m²)", labelpad=8)
    ax.set_ylabel("Escape or capture duration (Earth days)", labelpad=8)
    ax.set_xlim(MIN_SIGMA - 2, MAX_SIGMA + 2)
    ax.set_xticks(PLOT_SIGMAS)
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=7,
        width=1.1,
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
    ax.legend(
        loc="upper left",
        frameon=False,
        labelspacing=0.7,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    earth = load_earth()
    mars, missing_mars = load_corrected_mars()
    stats = summarize(pd.concat([earth, mars], ignore_index=True))
    plot_time(stats)

    print("Data provenance:")
    print(f"  Earth raw canonical summaries: {len(earth)} cells from {EARTH_DIR}")
    print(
        f"  Mars equal-Ls production summaries: "
        f"{sum(mars.provenance == '20260817 equal-Ls production')} cells from "
        f"{MARS_GRID_DIR}"
    )
    print("  Mars exact K12 phase-0 pilot reuse: 5 cells at sigma=15--35")
    if missing_mars:
        print("  Missing Mars cells (not plotted):")
        for shell, phase, sigma in sorted(missing_mars):
            print(f"    {shell} sigma={sigma} phase={phase}")
    else:
        print("  Missing Mars cells: none")
    print(f"  Plot font: {font_manager.findfont(PLOT_FONT)}")
    print("\nPlotted statistics (sample SD is blank where n_success < 2):")
    print(stats.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
