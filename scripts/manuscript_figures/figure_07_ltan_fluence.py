"""Render single-reflector delivered fluence versus initial LTAN.

  - Times New Roman throughout.
  - Four subplots, one per repeat-ground-track family (K9, K10, K11, K12).
  - x-axis = LTAN over [12, 24) h for all four (shared).
  - Same y-axis scaling for all four (shared), so panel heights are comparable.
  - Each panel annotated with vertical dashed lines marking the eclipse-free
    LTAN band for that family (from the manuscript orbit table; see BANDS).
  - Top-left label is just the K family ("$K_9$" ...), not altitude.
  - x label spells out Local Time of the Ascending Node (LTAN).
  - y label states the one-sol integration interval and uses J/m^2 notation.
  - Curve SMOOTHED (Savitzky-Golay) to suppress the discrete window-count
    "nicks"/jumps in the raw sweep, then rendered shape-preserving (PCHIP).

This reads the archived fine-sweep CSVs. Faint raw markers remain visible under
the smoothed line.

CAVEAT (data provenance): the cached CSVs were generated at repeat-ground-track
altitudes/inclinations that are CLOSE TO but not EXACTLY the manuscript-table
values (e.g. K9 CSV alt 1330.0 km / i 96.30 deg vs table 1332 km / 96.80 deg).
The eclipse-free bands drawn here use the manuscript-table values. The plotted
curve inputs and this small orbital-element discrepancy are retained as part of
the manuscript result record.

Inputs (must exist):
  simulation_outputs/20260611_ltan_energy_K9.csv                  (grid=='fine')
  simulation_outputs/20260611_ltan_energy_sweep_fine_12_24_pchip.csv  (K10/11/12)
Output:
  figures/manuscript/generated/figure_07_ltan_fluence.png
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

INPUT_DIR = Path("simulation_outputs")
K9_CSV = INPUT_DIR / "20260611_ltan_energy_K9.csv"
FINE_CSV = INPUT_DIR / "20260611_ltan_energy_sweep_fine_12_24_pchip.csv"
OUT_PNG = Path("figures/manuscript/generated/figure_07_ltan_fluence.png")

# Panel order (reading order, top-left -> bottom-right): K12 first.
K_ORDER = [12, 11, 10, 9]

# These correspond to the corrected altitudes and inclinations, while the
# cached fluence curves use slightly different elements (see module caveat).
#   Family   orbits/sol  alt(km)  inc(deg)  L_eclipse-free(h)
BANDS = {
    9:  (16.01, 19.94),
    10: (16.36, 19.60),
    11: (16.74, 19.15),
    12: (17.38, 18.38),
}

LTAN_LO_H, LTAN_HI_H = 12.0, 24.0

# Savitzky-Golay smoothing of the raw fine sweep (step 0.02 h -> ~50 pts/h).
# window 41 pts ~= 0.8 h smoothing scale, polyorder 3: kills the sub-0.2 h
# window-count nicks while preserving the broad ~3-4 h delivery lobe + its peak.
SMOOTH_WINDOW = 41
SMOOTH_POLY = 3

LINE_COLOR = "black"     # black curve
BAND_COLOR = "black"     # black dashed band edges
RAW_COLOR = "0.55"       # faint gray raw markers


def _read_k9_fine():
    """K9 fine grid (ltan, fluence) from the standalone K9 CSV."""
    L, F = [], []
    with K9_CSV.open() as fh:
        for r in csv.DictReader(fh):
            if r["grid"] != "fine":
                continue
            L.append(float(r["ltan_h"]))
            F.append(float(r["fluence_J_per_m2"]))
    return _sorted_arrays(L, F)


def _read_fine_k(k):
    """K10/11/12 fine grid (ltan, fluence) for one K from the fine-sweep CSV."""
    L, F = [], []
    with FINE_CSV.open() as fh:
        for r in csv.DictReader(fh):
            if int(r["K"]) != k:
                continue
            L.append(float(r["ltan_h"]))
            F.append(float(r["fluence_J_per_m2"]))
    return _sorted_arrays(L, F)


def _sorted_arrays(L, F):
    L = np.asarray(L, float)
    F = np.asarray(F, float)
    order = np.argsort(L)
    return L[order], F[order]


def _smooth(L, F):
    """Savitzky-Golay smooth (clipped >= 0), returned on a dense PCHIP grid."""
    win = min(SMOOTH_WINDOW, len(F) - (1 - len(F) % 2))  # odd, <= len
    if win >= SMOOTH_POLY + 2 and win % 2 == 1:
        Fs = savgol_filter(F, win, SMOOTH_POLY)
    else:
        Fs = F.copy()
    Fs = np.clip(Fs, 0.0, None)
    pch = PchipInterpolator(L, Fs)
    dense = np.linspace(L.min(), L.max(), 3000)
    return dense, np.clip(pch(dense), 0.0, None)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # serif math glyphs to match TNR
        "axes.titlesize": 13,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    # ---- load ---------------------------------------------------------------
    raw = {9: _read_k9_fine()}
    for k in (10, 11, 12):
        raw[k] = _read_fine_k(k)
    for k in K_ORDER:
        L, F = raw[k]
        print(f"K={k}: {len(L)} pts, raw peak {F.max():.3f} J/m^2 "
              f"at LTAN {L[int(np.argmax(F))]:.2f} h")

    smooth = {k: _smooth(*raw[k]) for k in K_ORDER}

    # shared y-limit from raw peaks (smoothed peak <= raw peak)
    y_max = max(raw[k][1].max() for k in K_ORDER)
    y_top = y_max * 1.08

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), K_ORDER):
        L, F = raw[k]
        dense, Fs = smooth[k]
        lo, hi = BANDS[k]

        # eclipse-free band: dashed edges only
        for x in (lo, hi):
            ax.axvline(x, color=BAND_COLOR, ls="--", lw=1.3, alpha=0.85)

        # Faint raw markers retain visibility of the unsmoothed samples.
        ax.plot(L, F, ".", color=RAW_COLOR, ms=1.6, alpha=0.18, zorder=2)
        ax.plot(dense, Fs, "-", color=LINE_COLOR, lw=2.2, zorder=3)

        ax.text(0.035, 0.93, rf"$K_{{{k}}}$", transform=ax.transAxes,
                fontsize=20, va="top", ha="left")

        ax.set_xlim(LTAN_LO_H, LTAN_HI_H)
        ax.set_ylim(0.0, y_top)
        ax.set_xticks(range(12, 25, 2))
        ax.grid(alpha=0.22, lw=0.6)
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout(rect=(0.035, 0.075, 1.0, 1.0))
    fig.supxlabel("Local Time of the Ascending Node (LTAN, h)", fontsize=18,
                  y=0.025)
    fig.supylabel(r"Fluence over one sol (J/m$^2$)", fontsize=18,
                  x=0.015, y=0.5)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
