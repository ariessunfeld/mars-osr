"""Render the representative one-sol cone and clock-angle time series.

Illumination windows are shaded so that the corresponding attitude excursions
remain visible. The manuscript output is title-free.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "simulation_outputs"
DATESTAMP = "20260602"
OUTDIR = REPO / "figures" / "manuscript" / "generated"
BASE_PREFIX = "20260529_slice17_cas24_K_r_4_3year_sig18gm2_abl"
STAGE = "de_chain"

# Fixed run config (shared by all versions) for orientation in the titles.
SIGMA_G_M2 = 18.0
K_ORBITS = 12
TARGET = "40N, 200E"

# label -> (one-line description of what is ablated, optional cone-cap deg)
VERSIONS = {
    "baseline":     ("no features ablated (illumination + real Sun + mode-2 + uncapped cone)", None),
    "no_illum":     ("illumination removed (0 delivery windows; fluence reward/penalty + PI off)", None),
    "no_obliquity": ("Mars obliquity removed (Sun in equatorial plane, sub-solar lat=0; J2/sun-sync kept)", None),
    "circular":     ("Mars eccentricity removed (circular heliocentric orbit, Sun distance = a_Mars)", None),
    "mode1":        ("attitude law reduced to mode-1 (mode-2 harmonics pinned ~0)", None),
    "cone70":       ("max cone angle capped at 70 deg", 70),
    "cone60":       ("max cone angle capped at 60 deg", 60),
    "cone50":       ("max cone angle capped at 50 deg", 50),
    "cone40":       ("max cone angle capped at 40 deg", 40),
    "cone70_noillum": ("max cone 70 deg AND illumination removed (0 windows)", 70),
    "cone60_noillum": ("max cone 60 deg AND illumination removed (0 windows)", 60),
    "cone50_noillum": ("max cone 50 deg AND illumination removed (0 windows)", 50),
    "cone40_noillum": ("max cone 40 deg AND illumination removed (0 windows)", 40),
}

CONE_COLOR = "#0b5394"
CLOCK_COLOR = "#cc4125"
WINDOW_SHADE = "#cc3b3b"
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 15
CAP_ANNOTATION_FONT_SIZE = 11
BASELINE_AXIS_LABEL_FONT_SIZE = 28
BASELINE_TICK_LABEL_FONT_SIZE = 24
BASELINE_LEGEND_FONT_SIZE = 28
BASELINE_FIGSIZE = (16.0, 5.0)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def mcinnes_cone_clock(n, s, h, sign=1):
    """McInnes 1999 Eq 4.7 cone + clock; clock wrapped to [0, 360)."""
    s, n = _unit(s), _unit(n)
    cone = np.degrees(np.arccos(np.clip(np.sum(n * s, axis=-1), -1.0, 1.0)))
    e_A = _unit(h - np.sum(h * s, axis=-1, keepdims=True) * s)
    e_B = sign * _unit(np.cross(e_A, s))
    clock = np.degrees(np.arctan2(np.sum(n * e_B, axis=-1),
                                  np.sum(n * e_A, axis=-1))) % 360.0
    return cone, clock


def load_traj(label):
    p = SIM / f"{BASE_PREFIX}_{label}_{STAGE}_sol1_trajectory.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, newline="")))
    g = lambda k: np.array([float(r[k]) for r in rows])
    n = np.column_stack([g("n_x"), g("n_y"), g("n_z")])
    s = np.column_stack([g("s_hat_x"), g("s_hat_y"), g("s_hat_z")])
    h = np.column_stack([g("h_hat_x"), g("h_hat_y"), g("h_hat_z")])
    cone, clock = mcinnes_cone_clock(n, s, h)
    roles = [r["schedule_role"] for r in rows]
    return dict(t_min=g("t_s") / 60.0, cone=cone, clock=clock, roles=roles)


def load_persol(label):
    p = SIM / f"{BASE_PREFIX}_{label}_{STAGE}_per_sol.csv"
    if not p.exists():
        return None
    row = next((r for r in csv.DictReader(open(p, newline="")) if int(r["sol_idx"]) == 1), None)
    if row is None:
        return None

    def gf(k):
        try:
            return float(row[k])
        except (KeyError, ValueError):
            return float("nan")

    dr = np.linalg.norm([gf(f"delta_r_iau_mars_km_{k}") for k in "xyz"])
    dv = np.linalg.norm([gf(f"delta_v_iau_mars_kmps_{k}") for k in "xyz"]) * 1e3
    return dict(a0=gf("dv_0_deg"), dr_km=float(dr), dv_mps=float(dv),
                e_max=gf("e_max"), fluence=gf("fluence_J_per_m2"))


def _shade_windows(ax, t, roles):
    """Shade contiguous bisector (delivery) intervals."""
    i = 0
    n = len(roles)
    while i < n:
        if roles[i] == "bisector":
            j = i
            while j + 1 < n and roles[j + 1] == "bisector":
                j += 1
            ax.axvspan(t[i], t[j], color=WINDOW_SHADE, alpha=0.12, lw=0)
            i = j + 1
        else:
            i += 1


def make_version_figure(label):
    tr = load_traj(label)
    ps = load_persol(label)
    if tr is None or ps is None:
        print(f"  {label}: missing output -- skipped")
        return False
    desc, cap = VERSIONS[label]
    n_bis = sum(1 for r in tr["roles"] if r == "bisector")
    mean_cone = float(np.mean(tr["cone"]))
    is_baseline = label == "baseline"
    axis_label_font_size = (
        BASELINE_AXIS_LABEL_FONT_SIZE if is_baseline else AXIS_LABEL_FONT_SIZE
    )
    tick_label_font_size = (
        BASELINE_TICK_LABEL_FONT_SIZE if is_baseline else TICK_LABEL_FONT_SIZE
    )
    legend_font_size = (
        BASELINE_LEGEND_FONT_SIZE if is_baseline else LEGEND_FONT_SIZE
    )

    figsize = BASELINE_FIGSIZE if is_baseline else (14.5, 5.2)
    fig, (axc, axk) = plt.subplots(1, 2, figsize=figsize)
    for ax in (axc, axk):
        _shade_windows(ax, tr["t_min"], tr["roles"])
        ax.set_xlabel(
            "time from sol-0 start (min)",
            fontsize=axis_label_font_size,
        )
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=7,
            width=1.1,
            labelsize=tick_label_font_size,
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

    axc.plot(
        tr["t_min"],
        tr["cone"],
        color=CONE_COLOR,
        lw=1.7,
        label=r"cone angle $\alpha(t)$",
    )
    axc.set_ylabel(
        r"cone angle $\alpha$ ($^\circ$)",
        fontsize=axis_label_font_size,
    )
    axc.set_ylim(0, 90)
    axc.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=legend_font_size,
        frameon=False,
        borderaxespad=0.0,
    )
    if cap is not None:
        axc.axhline(cap, color="#444444", ls="--", lw=1.0)
        axc.text(
            tr["t_min"][-1],
            cap + 1.0,
            f"cap {cap}°",
            ha="right",
            va="bottom",
            fontsize=CAP_ANNOTATION_FONT_SIZE,
            color="#444444",
        )

    axk.plot(
        tr["t_min"],
        tr["clock"],
        color=CLOCK_COLOR,
        lw=1.7,
        label=r"clock angle $\delta(t)$",
    )
    axk.set_ylabel(
        r"clock angle $\delta$ ($^\circ$)",
        fontsize=axis_label_font_size,
    )
    axk.set_ylim(0, 360)
    axk.set_yticks([0, 90, 180, 270, 360])
    axk.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=legend_font_size,
        frameon=False,
        borderaxespad=0.0,
    )

    line1 = f"Ablation — {label}: {desc}"
    line2 = (
        f"sol-0  |  $\\sigma$={SIGMA_G_M2:.0f} g/m$^2$  |  "
        f"K={K_ORBITS} orbits/sol  |  target {TARGET}  |  "
        f"$\\alpha_0$={ps['a0']:.1f}°  "
        f"$\\langle\\alpha\\rangle$={mean_cone:.1f}°  |  "
        f"$\\|\\Delta r\\|$={ps['dr_km']:.2f} km  "
        f"$\\|\\Delta v\\|$={ps['dv_mps']:.2f} m/s  "
        f"$e_{{max}}$={ps['e_max']:.1e}  |  "
        f"fluence={ps['fluence']:.1f} J/m$^2$ "
        f"({n_bis} bisector samples)"
    )
    if is_baseline:
        fig.tight_layout()
    else:
        fig.suptitle(line1 + "\n" + line2, fontsize=12.5)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "figure_s2_attitude_angles.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")
    return True


def main():
    if not make_version_figure("baseline"):
        raise FileNotFoundError("baseline trajectory or per-sol input is missing")


if __name__ == "__main__":
    main()
