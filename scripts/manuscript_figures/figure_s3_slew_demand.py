"""Render peak illuminating-arc slew demand and fluence versus LTAN and M0.

For each repeat-ground-track design K=9,10,11,12, the calculation samples a
common 0.05 h LTAN grid and the full initial-mean-anomaly range. Cells outside
each family's eclipse-free band remain undefined. The first two columns show
the peak demanded angular velocity and acceleration during bisector delivery;
the third shows sol-integrated fluence for a 10,000 m^2 reflector. The archived
sweep stores a 1,000 m^2 fluence, so this last quantity is scaled by exactly ten
and checked against the stored scaled column before plotting.

The trajectory model uses J2 and the Sun third body at Mars perihelion without
SRP. Bisector-pointing derivatives are evaluated on a cubic spline with a 0.5 s
central difference over 50 interior samples per delivery window. Initial mean
anomaly is injected through ``nu_rad=M0`` for the circular initial orbit.

The angular-acceleration filter is disabled so that unconstrained peak demand
is retained; the 0.003 deg/s^2 control limit is overlaid as a feasibility
boundary. The 1.0 J/m^2 useful-window gate remains active.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from reflectors.attitude import _three_point_normals
from reflectors.dynamics import PropagationOptions, propagate
from reflectors.visibility import bisector_pointing
from reflectors.elements import state_from_classical_mme2000
from reflectors.ephemeris import utc_to_et
from reflectors.gravity import mars_gravity_model
from reflectors.kernels import load_kernels
from reflectors.mars_constants import SECONDS_PER_SOLAR_SOL_S
from reflectors.parallel import CloudpickleMap, configure_multiprocessing_for_spice
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import raan_mme2000_from_ltan
from reflectors.termination import AltitudeFloor
from reflectors.third_body import sun_third_body
from reflectors.visibility import find_delivery_windows

logger = logging.getLogger(__name__)

# --- canonical setup ---------------------------------------------------------
EPOCH_UTC = "2028-02-11T12:42:00"            # Mars perihelion 2028
TARGET_LAT_DEG, TARGET_LON_DEG = 40.0, 200.0
SIGMA = 0.018                                # canonical sail loading (kg/m^2)
ALPHA_MAX_RAD_S2 = math.radians(0.003)       # control limit (overlaid, NOT a filter here)
MIN_WINDOW_FLUENCE_J_PER_M2 = 1.0            # delivery-window definition gate
ELEV_MIN_DEG = 10.0
BISECTOR_COS_ALPHA_MIN = 0.1
CADENCE_S = 60.0
DT_S = 0.5                                    # FD half-step (== visibility._SLEW_DEFAULT_DT_S)
N_SUB = 50                                    # subgrid pts (== _SLEW_DEFAULT_SUBGRID_POINTS)
_MU = float(mars_gravity_model(max_degree=2).mu_km3_s2)
_R = float(mars_gravity_model(max_degree=2).ref_radius_km)
BANDS_CSV = Path("simulation_outputs/20260612_eclipsefree_bands.csv")
ARCHIVED_RESULTS_CSV = Path(
    "simulation_outputs/20260615-150016_slew_demand_vs_ltan_phasing.csv"
)
OUT_DIR = Path("simulation_outputs")
FIG_DIR = Path("figures/manuscript/generated")
KS = (9, 10, 11, 12)
LTAN_STEP_H = 0.05                           # common-grid LTAN step (base-power dense step)
# 3rd column: sol-integrated delivered fluence from a larger reflector. The sweep's
# canonical sail is 1000 m^2 (make_canonical_sail(0.018), confirmed area_m2=1000);
# 10000 m^2 reflector delivers REFLECTOR_AREA_M2 / SAIL_AREA_M2 = 10x the recorded
# 1000 m^2 fluence. Window definition stays at 1000 m^2 / min_fluence 1.0.
SAIL_AREA_M2 = 1000.0
REFLECTOR_AREA_M2 = 10000.0
AREA_SCALE = REFLECTOR_AREA_M2 / SAIL_AREA_M2  # = 10.0
STAMP = "20260820-153413"
BASE_FONT_SIZE = 20
TITLE_FONT_SIZE = 28
AXIS_LABEL_FONT_SIZE = 32
TICK_FONT_SIZE = 27
COLORBAR_TICK_FONT_SIZE = 26
CONTOUR_LABEL_FONT_SIZE = 20
TITLE_PAD_DEFAULT = 9


def sol_fluence_J_per_m2_10000m2(fluence_J_per_m2_1000m2: float) -> float:
    """Scale one-sol fluence from the swept 1000 m^2 sail to 10000 m^2.

    The beam model is linear in reflective area at fixed geometry
    (``beam.delivered_surface_irradiance_W_per_m2`` and
    ``tests/test_beam.py::TestDeliveredIrradianceScaling``). The delivery-window
    definition remains the original 1000 m^2, 1 J/m^2-gated definition so the
    slew-demand panels are unchanged from the durable 20260615 sweep.
    """
    return AREA_SCALE * float(fluence_J_per_m2_1000m2)


def _load_results_csv(path: Path):
    """Load a durable sweep and verify its archived 10000 m^2 scaling column."""
    results = []
    archived_abs_errors = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "K", "altitude_km", "ltan_h", "m0_deg", "n_windows",
            "total_illum_dur_s", "total_fluence_J_per_m2",
            "peak_omega_rad_s", "peak_alpha_rad_s2",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        archived_col = f"sol_avg_irradiance_Wm2_{int(REFLECTOR_AREA_M2)}m2"
        for row in reader:
            fluence = float(row["total_fluence_J_per_m2"])
            results.append((
                int(row["K"]), float(row["altitude_km"]),
                float(row["ltan_h"]), float(row["m0_deg"]),
                int(row["n_windows"]), float(row["total_illum_dur_s"]),
                fluence, float(row["peak_omega_rad_s"]),
                float(row["peak_alpha_rad_s2"]),
            ))
            if archived_col in row and row[archived_col]:
                expected = (
                    sol_fluence_J_per_m2_10000m2(fluence)
                    / float(SECONDS_PER_SOLAR_SOL_S)
                )
                archived_abs_errors.append(abs(float(row[archived_col]) - expected))
    if not results:
        raise ValueError(f"{path} contains no sweep rows")
    if archived_abs_errors:
        # Raw fluence was rounded to 0.001 J/m^2 in the archived CSV before reload;
        tolerance_W_per_m2 = (
            AREA_SCALE * 0.0005 / float(SECONDS_PER_SOLAR_SOL_S) + 5.0e-10
        )
        worst = max(archived_abs_errors)
        if worst > tolerance_W_per_m2:
            raise ValueError(
                f"archived 10000 m^2 scaling check failed: max abs error "
                f"{worst:.3e} W/m^2 > {tolerance_W_per_m2:.3e} W/m^2"
            )
        print(
            f"archived area-scaling audit passed: 10000 m^2 column = "
            f"10 x raw 1000 m^2 fluence / sol duration; max abs error "
            f"{worst:.3e} W/m^2",
            flush=True,
        )
    return results


def _t_eval():
    dur = float(SECONDS_PER_SOLAR_SOL_S)
    t = np.arange(0.0, dur, CADENCE_S)
    if t[-1] < dur:
        t = np.append(t, dur)
    return t, dur


def _peak_slew_in_window(traj_fn, et_a, et_b, profile, dt_s=DT_S, n_sub=N_SUB):
    """(peak ||omega||, peak ||alpha||) of the bisector profile over [et_a, et_b].

    Shares the 3-point normal stencil between omega and alpha (identical math to
    attitude.angular_rate / angular_acceleration). Returns (0,0) if too short."""
    if et_b - et_a <= 2.0 * dt_s:
        return 0.0, 0.0
    grid = np.linspace(et_a + dt_s, et_b - dt_s, int(n_sub))
    peak_o = peak_a = 0.0
    for et in grid:
        n_m, n_0, n_p = _three_point_normals(profile, traj_fn, float(et), dt_s)
        n_dot = (n_p - n_m) / (2.0 * dt_s)
        n_ddot = (n_p - 2.0 * n_0 + n_m) / (dt_s * dt_s)
        o = float(np.linalg.norm(np.cross(n_0, n_dot)))
        a = float(np.linalg.norm(np.cross(n_0, n_ddot)))
        if o > peak_o:
            peak_o = o
        if a > peak_a:
            peak_a = a
    return peak_o, peak_a


_KPID = None


def _ensure_kernels():
    global _KPID
    if _KPID != os.getpid():
        load_kernels()
        _KPID = os.getpid()


def case_worker(args):
    """One (K, LTAN, M0) case -> peak illuminating-arc omega & alpha. Picklable."""
    _ensure_kernels()
    K, a, i_deg, alt, ltan, m0, epoch_et = args
    t_eval, dur = _t_eval()
    sail = make_canonical_sail(SIGMA)
    raan = raan_mme2000_from_ltan(ltan, epoch_et)
    s0 = state_from_classical_mme2000(
        a_km=a, e=0.0, inclination_rad=math.radians(i_deg),
        raan_rad=raan, argp_rad=0.0, nu_rad=math.radians(m0),
        mu_km3_s2=_MU, epoch_et=epoch_et)
    res = propagate(state0_km_kmps=s0, t_span_s=(0.0, dur), epoch_et=epoch_et,
                    zonal_degree=2, gravity_degree=0,
                    third_bodies=[sun_third_body()], solar_sail=None,
                    sail_normal=None, altitude_floor=AltitudeFloor.at_km(300.0),
                    options=PropagationOptions.fast(), t_eval_s=t_eval)
    windows = find_delivery_windows(
        res, TARGET_LAT_DEG, TARGET_LON_DEG, target_elevation_min_deg=ELEV_MIN_DEG,
        bisector_cos_alpha_min=BISECTOR_COS_ALPHA_MIN, require_sail_sunlit=True,
        require_sail_above_horizon=True, require_bisector_feasible=True, sail=sail,
        atmospheric_transmission=1.0, alpha_max_rad_s2=None,
        min_window_fluence_J_per_m2=MIN_WINDOW_FLUENCE_J_PER_M2)
    if not windows:
        return (K, alt, ltan, m0, 0, 0.0, 0.0, 0.0, 0.0)

    et_grid = epoch_et + np.asarray(res.t_s, dtype=float)
    pos = np.asarray(res.state_km_kmps, dtype=float)[:, :3]
    traj = CubicSpline(et_grid, pos, axis=0)

    def traj_fn(et):
        return np.asarray(traj(float(et)), dtype=float)

    profile = bisector_pointing(TARGET_LAT_DEG, TARGET_LON_DEG)
    peak_o = peak_a = tot_dur = tot_flu = 0.0
    for w in windows:
        tot_dur += float(w.t_end_s - w.t_start_s)
        tot_flu += float(w.fluence_J_per_m2 or 0.0)
        o, a = _peak_slew_in_window(traj_fn, w.et_start, w.et_end, profile)
        peak_o = max(peak_o, o)
        peak_a = max(peak_a, a)
    return (K, alt, ltan, m0, len(windows), tot_dur, tot_flu, peak_o, peak_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid (K=10 only, coarse LTAN x 4 M0) for wiring/timing")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ltan-step", type=float, default=LTAN_STEP_H,
                    help="common-grid LTAN step (h)")
    ap.add_argument("--m0-step", type=float, default=5.0, help="M0 step (deg)")
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "reuse a durable sweep CSV instead of propagating; the canonical "
            f"full-sweep input is {ARCHIVED_RESULTS_CSV}"
        ),
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    bands = {int(r["K"]): r for r in csv.DictReader(BANDS_CSV.open())}
    if args.smoke and args.input_csv is not None:
        ap.error("--smoke and --input-csv are mutually exclusive")
    ks = (10,) if args.smoke else KS

    # COMMON LTAN grid spanning the WIDEST band (the K with the largest [lo,hi]).
    # Step = args.ltan_step (default 0.05 h). Grid is clamped to <= the widest hi so
    # we never exceed any band's upper edge. Each K then computes only the common-grid
    # LTANs inside its OWN [lo_K, hi_K] band (eclipse-free region); the rest are NaN.
    widest_lo = min(float(bands[K]["ltan_lo_h"]) for K in KS)
    widest_hi = max(float(bands[K]["ltan_hi_h"]) for K in KS)
    ltan_step = args.ltan_step
    n_common = int(math.floor((widest_hi - widest_lo) / ltan_step + 1e-9)) + 1
    ltan_common = widest_lo + ltan_step * np.arange(n_common)  # all <= widest_hi
    if args.smoke:
        ltan_common = ltan_common[::8]  # coarse for wiring
    m0_grid = (np.arange(0.0, 360.0, 90.0) if args.smoke
               else np.arange(0.0, 360.0, args.m0_step))

    tasks = []
    meta = {}
    per_k_ltans = {}
    for K in ks:
        r = bands[K]
        a, i_deg = float(r["a_km"]), float(r["i_deg"])
        lo, hi = float(r["ltan_lo_h"]), float(r["ltan_hi_h"])
        alt = a - _R
        meta[K] = dict(a=a, i=i_deg, lo=lo, hi=hi, alt=alt)
        # only the COMMON-grid LTANs inside this K's own band are eclipse-free
        in_band = ltan_common[(ltan_common >= lo - 1e-9) & (ltan_common <= hi + 1e-9)]
        per_k_ltans[K] = in_band
        for L in in_band:
            for m0 in m0_grid:
                tasks.append((K, a, i_deg, alt, float(L), float(m0), None))
    print(f"common LTAN grid: [{widest_lo:.4f},{widest_hi:.4f}] h step {ltan_step} h "
          f"-> {len(ltan_common)} pts", flush=True)
    for K in ks:
        print(f"  K{K} band [{meta[K]['lo']:.3f},{meta[K]['hi']:.3f}] -> "
              f"{len(per_k_ltans[K])} in-band LTANs", flush=True)
    if args.input_csv is None:
        print(f"sweep: {len(tasks)} cases ({len(m0_grid)} M0 each), "
              f"{args.workers} workers", flush=True)
        configure_multiprocessing_for_spice()
        load_kernels()
        epoch_et = utc_to_et(EPOCH_UTC)
        # Rebuild tasks now that SPICE is loaded and the epoch is available.
        tasks = [task[:-1] + (epoch_et,) for task in tasks]
        with CloudpickleMap(n_workers=args.workers) as cp:
            results = cp(case_worker, tasks)
        print(f"sweep done [{time.perf_counter()-t0:.0f}s]", flush=True)
    else:
        results = _load_results_csv(args.input_csv)
        input_ks = tuple(sorted({row[0] for row in results}))
        if input_ks != KS:
            raise ValueError(f"expected K families {KS}, found {input_ks} in {args.input_csv}")
        expected_cases = sum(len(per_k_ltans[K]) for K in ks) * len(m0_grid)
        if len(results) != expected_cases:
            raise ValueError(
                f"expected {expected_cases} rows for the configured LTAN/M0 grid, "
                f"found {len(results)} in {args.input_csv}"
            )
        print(f"reused {len(results)} cases from {args.input_csv}", flush=True)

    # --- save CSV -------------------------------------------------------------
    csv_path = OUT_DIR / f"{STAMP}_slew_demand_vs_ltan_phasing_panel_scales.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "altitude_km", "ltan_h", "m0_deg", "n_windows",
                    "total_illum_dur_s", "total_fluence_J_per_m2",
                    "peak_omega_rad_s", "peak_omega_deg_s",
                    "peak_alpha_rad_s2", "peak_alpha_deg_s2",
                    f"sol_fluence_Jm2_{int(REFLECTOR_AREA_M2)}m2"])
        for (K, alt, ltan, m0, nw, dur, flu, po, pa) in results:
            w.writerow([K, f"{alt:.2f}", f"{ltan:.4f}", f"{m0:.2f}", nw,
                        f"{dur:.1f}", f"{flu:.3f}",
                        f"{po:.6e}", f"{math.degrees(po):.6e}",
                        f"{pa:.6e}", f"{math.degrees(pa):.6e}",
                        f"{sol_fluence_J_per_m2_10000m2(flu):.6e}"])
    print(f"CSV -> {csv_path}", flush=True)

    # --- raw summary table (no interpretation) --------------------------------
    arr = {K: [] for K in ks}
    for row in results:
        arr[row[0]].append(row)
    print("=" * 92)
    print("RAW SUMMARY (peak over the LTAN x M0 grid; omega in deg/s, alpha in deg/s^2)")
    print(f"alpha control limit = {math.degrees(ALPHA_MAX_RAD_S2):.4f} deg/s^2")
    print("=" * 92)
    hdr = (f"{'K':>3}{'alt_km':>9}{'band_h':>16}{'n_case':>7}{'maxOmega_deg_s':>16}"
           f"{'maxAlpha_deg_s2':>17}{'%alpha_lim':>11}"
           f"{'Jm2_min':>12}{'Jm2_max':>12}")
    print(hdr)
    for K in ks:
        rows = arr[K]
        m = meta[K]
        mo = max(r[7] for r in rows)
        ma = max(r[8] for r in rows)
        flu10k = [sol_fluence_J_per_m2_10000m2(r[6]) for r in rows]
        band_str = f"[{m['lo']:.2f},{m['hi']:.2f}]"
        print(f"{K:>3}{m['alt']:>9.1f}{band_str:>16}"
              f"{len(rows):>7}{math.degrees(mo):>16.4e}{math.degrees(ma):>17.4e}"
              f"{100*ma/ALPHA_MAX_RAD_S2:>10.1f}%"
              f"{min(flu10k):>12.4e}{max(flu10k):>12.4e}")
    all_flu10k = [
        sol_fluence_J_per_m2_10000m2(r[6]) for K in ks for r in arr[K]
    ]
    print(f"sol-integrated delivered fluence from {int(REFLECTOR_AREA_M2)} m^2 reflector "
          f"(= AREA_SCALE x swept 1000 m^2 fluence, AREA_SCALE={AREA_SCALE:g}):")
    print(f"  GLOBAL J/m^2 range = [{min(all_flu10k):.6e}, {max(all_flu10k):.6e}]  "
          f"(all finite: {all(math.isfinite(x) for x in all_flu10k)}, "
          f"all >=0: {all(x >= 0.0 for x in all_flu10k)})")
    # Reference values at M0=0 provide an independent check of the sweep.
    print("-" * 92)
    print("REFERENCE CHECK (M0=0; K10 maxAlpha~3.56e-4, K11~4.54e-4 deg/s^2):")
    for K in ks:
        m0z = [r for r in arr[K] if abs(r[3]) < 1e-9]
        if m0z:
            mo = max(r[7] for r in m0z); ma = max(r[8] for r in m0z)
            print(f"  K{K} M0=0: maxOmega {math.degrees(mo):.4e} deg/s, "
                  f"maxAlpha {math.degrees(ma):.4e} deg/s^2 (over band)")

    # --- figure: per-K heatmaps on a COMMON LTAN grid; 3 columns -----------------
    # Columns: peak |omega| (deg/s), peak |alpha| (deg/s^2), and sol-integrated
    # fluence (J/m^2, REFLECTOR_AREA_M2 reflector). Rows K12 (top, low alt) ->
    # K9 (bottom). Every row shares the SAME x-axis = full common LTAN range; cells
    # outside a K's eclipse-free band are NaN and rendered white. Every PANEL has
    # its own finite-data min/max color scale and adjacent colorbar.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": BASE_FONT_SIZE,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    krows = sorted(ks, reverse=True)            # K12 top -> K9 bottom
    m0_axis = m0_grid
    nL = len(ltan_common)
    lt_idx = {round(float(L), 6): j for j, L in enumerate(ltan_common)}
    m0_idx = {round(float(M), 6): k for k, M in enumerate(m0_axis)}

    # build per-K matrices on the FULL common grid (NaN outside each band)
    OW = {K: np.full((len(m0_axis), nL), np.nan) for K in ks}
    AL = {K: np.full((len(m0_axis), nL), np.nan) for K in ks}
    FL = {K: np.full((len(m0_axis), nL), np.nan) for K in ks}
    for K in ks:
        for (kk, alt, ltan, m0, nw, dur, flu, po, pa) in arr[K]:
            j = lt_idx[round(float(ltan), 6)]
            k = m0_idx[round(float(m0), 6)]
            OW[K][k, j] = math.degrees(po)
            AL[K][k, j] = math.degrees(pa)
            FL[K][k, j] = sol_fluence_J_per_m2_10000m2(flu)

    def panel_lims(Z):
        vals = Z[np.isfinite(Z)]
        if vals.size == 0:
            raise ValueError("cannot color-scale a panel containing no finite data")
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo == hi:
            pad = max(abs(lo) * 0.01, np.finfo(float).eps)
            return lo - pad, hi + pad
        return lo, hi

    def two_sig_mantissa(value):
        """Format a scaled colorbar tick with exactly two significant figures."""
        if value == 0.0:
            return "0.0"
        decimals = max(0, 1 - math.floor(math.log10(abs(value))))
        return f"{value:.{decimals}f}"

    # extent uses the full common LTAN range so EVERY row's x-axis is identical
    # (half-cell padding so imshow pixel centers land on the grid LTAN values)
    half = 0.5 * ltan_step
    x_lo, x_hi = ltan_common[0] - half, ltan_common[-1] + half
    ext = [x_lo, x_hi, m0_axis[0], m0_axis[-1]]
    alpha_lim_deg = math.degrees(ALPHA_MAX_RAD_S2)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")                       # NaN (out-of-band) -> white

    nrows = len(krows)
    fig, axes = plt.subplots(nrows, 3, figsize=(24, 16.2), squeeze=False,
                             constrained_layout=True)
    col_specs = (
        (r"$\omega_{\max}\,(^\circ/\mathrm{s})$", OW, None),
        (r"$\alpha_{\max}\,(^\circ/\mathrm{s}^{2})$", AL, alpha_lim_deg),
        (r"sol fluence $(\mathrm{J}/\mathrm{m}^{2})$", FL, None),
    )
    print("FIGURE color scales (independent finite-data min/max for each panel):")
    for ridx, K in enumerate(krows):
        for cidx, (label, D, lim) in enumerate(col_specs):
            Z = D[K]
            ax = axes[ridx][cidx]
            vlo, vhi = panel_lims(Z)
            print(
                f"  K{K} col{cidx + 1}: [{vlo:.6e}, {vhi:.6e}]",
                flush=True,
            )
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=ext,
                           cmap=cmap, vmin=vlo, vmax=vhi)
            # CHANGE 6: alpha control-limit contour where the data reaches it
            if lim is not None and np.isfinite(np.nanmax(Z)) and np.nanmax(Z) >= lim:
                cs = ax.contour(ltan_common,
                                np.asarray(m0_axis, dtype=float), Z,
                                levels=[lim], colors="red", linewidths=1.8)
                ax.clabel(
                    cs, fmt=f"limit {lim:.3g}", fontsize=CONTOUR_LABEL_FONT_SIZE
                )
            ax.set_xlim(x_lo, x_hi)              # CHANGE 2: identical x-limits per row
            ax.set_yticks([0, 90, 180, 270, 360])   # CHANGE 5: fixed M0 ticks
            altitude_label = (
                f"{meta[K]['alt']:.0f} km"
                if cidx == 2
                else f"{meta[K]['alt']:.0f} km altitude"
            )
            ax.set_title(
                rf"$K_{{{K}}}$ ({altitude_label}), {label}",
                fontsize=TITLE_FONT_SIZE,
                pad=TITLE_PAD_DEFAULT,
            )
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=True,
                labelsize=TICK_FONT_SIZE,
                length=6,
                width=1.1,
            )
            ax.tick_params(axis="x", which="major", pad=8)
            ax.tick_params(axis="y", which="major", pad=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.018)
            exponent = math.floor(math.log10(max(abs(vlo), abs(vhi))))
            scale = 10.0 ** exponent
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _position, scale=scale:
                    two_sig_mantissa(value / scale)
                )
            )
            cbar.ax.set_title(
                rf"$\times 10^{{{exponent}}}$",
                fontsize=COLORBAR_TICK_FONT_SIZE,
                pad=7,
            )
            cbar.ax.tick_params(
                labelsize=COLORBAR_TICK_FONT_SIZE, length=4, width=1.0
            )
    fig.supxlabel(
        "Local Time of the Ascending Node (LTAN, h)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    fig.supylabel(
        r"Initial phasing $M_0$ ($^\circ$)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    fig_path = FIG_DIR / "figure_s3_slew_demand.png"
    fig.savefig(fig_path, dpi=180)
    print(f"FIG -> {fig_path}", flush=True)
    print(f"[total {time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
