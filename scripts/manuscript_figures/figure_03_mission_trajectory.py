"""Render the K12 Earth-to-Mars trajectory used as manuscript Figure 3.

All panels use Matplotlib's perceptually uniform ``viridis`` colormap and a
log(1 + elapsed days) normalization. The input solution must be closed,
full-physics, and joined to the capture segment on its <=28-day outer leg.
Independent re-propagation must satisfy the 10 km and 1 m/s closure bounds.

Panel A is the real geocentric Earth-escape trajectory. Panel B re-propagates the
stored cruise control with the same Earth, Moon, and Mars point-mass perturbations
used by the solver. Panel C begins at the *actual* interpolated cruise-to-capture
handoff, not at the earlier Hill-node tip, and follows only the physically flown
remainder of the real backward-generated Mars-capture trajectory to LMO.

Usage: python scripts/manuscript_figures/figure_03_mission_trajectory.py \
       --earth-csv <real escape per_step> --tendril-json <closed outer-leg solution> \
       --capture-csv <matching real capture per_step>
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import FuncNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text as _MplText

# (3) Times New Roman everywhere; stix gives Times-compatible math glyphs for the few mathtext
# bits (sigma, ->, superscripts) and the sci-notation x10^n offset text.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 15

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import spiceypy as spice
from reflectors.kernels import load_kernels
from reflectors.cruise_command import heliocentric_orbit_normal
from reflectors.cruise_propagator import propagate_cruise_clean
from reflectors.central_body import sun_central_body
from reflectors.sail_designs import make_canonical_sail
from reflectors.surface import earth_equatorial_radius_km, mars_equatorial_radius_km
from reflectors.third_body import earth_third_body, mars_third_body, moon_third_body

VIRIDIS = plt.get_cmap("viridis")
OUTER_LEG_MAX_D = 28.0
REPLAY_R_MISS_MAX_KM = 10.0
REPLAY_V_MISS_MAX_KMPS = 1.0e-3
REQUIRED_CAPTURE_SHELL = "K12"

# (11) shared-epoch markers: one shape per handoff epoch, identical across panels. Distinct
# SHAPES (star vs square) keep the two readable under red/green color blindness.
T1_COLOR = "red"        # sail @ t1  (A end == B start: Earth-Hill exit / transfer start)
T2_COLOR = "limegreen"  # sail @ t2  (B end == C node : Mars-Hill arrival / capture start)
T1_KW = dict(marker="*", s=360, edgecolor="black", linewidth=0.9)    # red star (panel A endpoint)
T2_KW = dict(marker="s", s=150, edgecolor="black", linewidth=0.9)    # green square (panel C node)
T1_KW_B = dict(marker="*", s=150, edgecolor="black", linewidth=0.8)
T2_KW_B = dict(marker="s", s=55, edgecolor="black", linewidth=0.8)   # smaller green square in B so Mars shows behind it
DOT_S = 170

# (7) line weights -- escape/capture spirals are dense, so thinnest; transfer arc heavier
LW_A = 0.7
LW_B = 1.3
LW_C = 0.7
LW_ORBIT = 1.2
CUTOUT_LW = 0.12        # hairline within both cutouts (crisp only at high dpi)
CUTOUT_WIN_DIAM = 3.0   # cutout window width in BODY DIAMETERS (half-window = this * R_body)


class _LegendText:
    """Sentinel legend handle that renders TEXT (a $t_n$ symbol) in the handle/symbol column, so a
    definition row reads '<symbol>   <definition>' aligned with the marker rows above it."""
    def __init__(self, text):
        self.text = text


class _LegendTextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        t = _MplText(width / 2.0, height / 2.0, orig_handle.text, fontsize=fontsize, ha="center", va="center")
        t.set_transform(trans)
        return [t]

ROT_A_DEG = 180.0       # in-plane rotation of the Panel-A view for layout. A proper rotation that
                        # gives the y=-x-reflected LOOK (Earth -> bottom-left, departure -> top-right)


def load_per_step(csv_path):
    et, r, v = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            et.append(float(row["et"]))
            r.append([float(row["r_x_km"]), float(row["r_y_km"]), float(row["r_z_km"])])
            v.append([float(row["v_x_kmps"]), float(row["v_y_kmps"]), float(row["v_z_kmps"])])
    return np.array(et), np.array(r), np.array(v)


def best_plane_2d(r):
    """Project 3D positions onto their SVD best-fit plane. The body center (r=0) maps to (0,0)
    in the returned 2D coords, so a to-scale planet dot can be drawn at the origin."""
    c = r - r.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    return np.column_stack([r @ e1, r @ e2]), (e1, e2)


def orbit_plane_2d(r, v):
    """Project onto the INITIAL orbit plane (in-plane basis from the first state: a1 = r0_hat,
    a2 = (r0 x v0) x r0_hat). Faithful for the near-body phase where the plane is ~fixed (the
    escape orbit-normal stays within ~3 deg of h0 until the final hyperbolic departure). The SVD
    best-fit plane of the full 3D escape is not faithful near the body: near-Earth samples can
    falsely project inside Earth. This projection preserves the true periapsis geometry."""
    h0 = np.cross(r[0], v[0]); h0 = h0 / np.linalg.norm(h0)
    a1 = r[0] / np.linalg.norm(r[0]); a2 = np.cross(h0, a1)
    return np.column_stack([r @ a1, r @ a2]), (a1, a2)


def cruise_trajectory_from_json(tendril_json):
    """Reconstruct the cruise with the exact perturbation set recorded by the solver."""
    d = json.load(open(tendril_json))
    z0 = np.asarray(d["z0_helio"], float)
    x = np.asarray(d["x"], float); N = int(d["N"])
    phis, thetas, D = x[:N], x[N:2 * N], float(x[2 * N])
    dep_et = float(d["dep_et"]); T_s = D * 86400.0
    ref = heliocentric_orbit_normal(z0)
    sail = make_canonical_sail(0.018); cb = sun_central_body()
    third_bodies = (
        (earth_third_body(), moon_third_body(), mars_third_body())
        if d.get("full_physics", False)
        else ()
    )
    t_arr, y_arr = propagate_cruise_clean(phis, thetas, z0, dep_et, T_s, ref, sail, cb, third_bodies,
                                          max_step_s=7200.0, return_trajectory=True)
    return dep_et, T_s, np.asarray(t_arr), y_arr, z0, d


def planet_orbit(naif, et_center, span_s, n=400):
    ets = np.linspace(et_center - span_s / 2, et_center + span_s / 2, n)
    return np.array([spice.spkezr(naif, e, "J2000", "NONE", "10")[0][:3] for e in ets])


def elapsed_time_norm(days):
    vmax = float(np.max(days))
    if vmax <= 0.0:
        raise ValueError("elapsed-time range must extend beyond day 0")
    return FuncNorm((np.log1p, np.expm1), vmin=0.0, vmax=vmax, clip=True)


def colored_2d(ax, xy, c, cmap=VIRIDIS, lw=1.5, norm=None):
    """Time-colored polyline with a quantitative log(1+days) normalization."""
    pts = xy.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    if norm is None:
        norm = elapsed_time_norm(c)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(0.5 * (c[:-1] + c[1:]))
    ax.add_collection(lc); return lc


def densify(xy, c, factor=6):
    """Cubic-spline upsample a strided trajectory (parameter = cumulative chord length) so the
    cutout renders smooth periapsis arcs rather than chord-cut polylines. This interpolates the
    REAL samples of a smooth real trajectory -- the per-step CSV is strided ~Nx from a much finer
    integration, so the inward chord-cut near fast periapsis is a storage artifact, not physics.
    Color (c, monotone in time) is linearly interpolated so the coloring stays 1:1 with the panel."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(d) > 0])      # drop zero-length duplicate samples
    d, xy, c = d[keep], xy[keep], c[keep]
    u = np.linspace(d[0], d[-1], (len(d) - 1) * factor + 1)
    xy_f = np.column_stack([CubicSpline(d, xy[:, 0])(u), CubicSpline(d, xy[:, 1])(u)])
    return xy_f, np.interp(u, d, c)


def rotate2d(xy, deg):
    """Rigid in-plane rotation of projected 2D points (proper, det=+1 -> preserves chirality)."""
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return xy @ np.array([[c, s], [-s, c]])


def _hcbar(fig, ax, lc, label):
    """One horizontal colorbar with log-spaced elapsed-day colors and readable day ticks."""
    cb = fig.colorbar(lc, ax=ax, orientation="horizontal", location="bottom",
                      fraction=0.05, pad=0.02, aspect=32)
    vmax = float(lc.norm.vmax)
    candidates = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000], dtype=float)
    ticks = candidates[candidates <= vmax]
    cb.set_ticks(ticks)
    cb.set_label(f"{label} (log scale)", fontsize=17)
    cb.ax.tick_params(labelsize=15)
    return cb


def add_zoom_cutout(fig, ax, xy, days, norm, r_body, body_color):
    """Top-left zoom cutout sharing the panel's exact projection.

    Same colormap + same clim (full-range, set in
    colored_2d) -> coloring is 1:1 with the panel. The cutout uses a cubic-spline-DENSIFIED copy + a
    hairline so the dense spiral stays crisp at high dpi. The small indicator box around the body in
    the MAIN panel is drawn WHITE (pops against the dense near-body lines); the larger cutout frame +
    connectors stay black."""
    win = CUTOUT_WIN_DIAM * r_body
    axin = ax.inset_axes([0.04, 0.62, 0.334, 0.334])
    axin.set_box_aspect(1.0)
    xy_d, day_d = densify(xy, days, factor=8)
    colored_2d(axin, xy_d, day_d, VIRIDIS, lw=CUTOUT_LW, norm=norm)
    axin.add_patch(Circle((0.0, 0.0), r_body, facecolor=body_color, edgecolor="none", zorder=6))
    axin.set_xlim(-win, win); axin.set_ylim(-win, win); axin.set_aspect("equal")
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_linewidth(1.0); s.set_edgecolor("black")
    ind = ax.indicate_inset_zoom(axin, edgecolor="black", linewidth=1.0)
    rect = ind.rectangle if hasattr(ind, "rectangle") else ind[0]
    rect.set_edgecolor("white"); rect.set_linewidth(1.4); rect.set_zorder(7)
    return axin


def place_offsets(fig, specs, unit=""):
    """Re-place each sci-notation x10^n offset at an explicit axes-coord position, with an optional
    unit suffix (e.g. 'km' -> 'x10^6 km'). ``specs`` = [(ax, x_spec, y_spec), ...] where each spec is
    (x, y, ha, va) in axes coords, or None to leave matplotlib's default. Offset strings are computed
    at draw time, so draw once first, then hide the auto offsets and re-place the specified ones."""
    fig.canvas.draw()
    suf = (" " + unit) if unit else ""
    for ax, xspec, yspec in specs:
        for axis, spec in ((ax.xaxis, xspec), (ax.yaxis, yspec)):
            if spec is None:
                continue
            off = axis.get_offset_text(); s = off.get_text()
            off.set_visible(False)
            if s:
                ax.text(spec[0], spec[1], s + suf, transform=ax.transAxes, ha=spec[2], va=spec[3], fontsize=14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--earth-csv", required=True, help="REAL escape per_step.csv (geocentric)")
    ap.add_argument("--tendril-json", required=True, help="closed tendril solution JSON")
    ap.add_argument("--capture-csv", required=True, help="REAL capture per_step.csv (Mars-centred)")
    ap.add_argument("--cutouts", action=argparse.BooleanOptionalAction, default=True,
                    help="draw the near-body zoom cutouts in panels A and C (use --no-cutouts to disable)")
    ap.add_argument("--dpi", type=int, default=1000, help="output raster DPI for the 2D figure")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "figures" / "manuscript" / "generated" / "figure_03_mission_trajectory.png",
        help="output PNG",
    )
    args = ap.parse_args()
    load_kernels()

    R_EARTH = earth_equatorial_radius_km()   # 6378.137 km, single-source (BODY399_RADII)
    R_MARS = mars_equatorial_radius_km()     # 3396.19 km, single-source (BODY499_RADII)

    e_et, e_r, e_v = load_per_step(args.earth_csv)
    e_days = (e_et - e_et[0]) / 86400.0
    dep_et, T_s, cruise_t, cruise_y, z0, sol = cruise_trajectory_from_json(args.tendril_json)
    cruise_days = (cruise_t - cruise_t[0]) / 86400.0
    c_et, c_r, _c_v = load_per_step(args.capture_csv)
    order = np.argsort(c_et)                              # forward time: node -> LMO
    c_et, c_r = c_et[order], c_r[order]

    arr_et = dep_et + T_s
    g_norm = float(sol.get("g_norm", float("nan")))
    t1 = spice.et2utc(dep_et, "ISOC", 0)[:10]; t2 = spice.et2utc(arr_et, "ISOC", 0)[:10]

    # Input-consistency and trajectory-closure gates.
    # it was Sun-only and arrived 295 d / 154 revs into the capture spiral.
    expected_escape = Path(sol.get("escape_summary", "")).name.replace("_summary.txt", "_per_step.csv")
    expected_capture = Path(sol.get("capture_csv", "")).name
    if Path(args.earth_csv).name != expected_escape:
        raise ValueError(f"Earth CSV does not match solution provenance: expected {expected_escape}")
    if Path(args.capture_csv).name != expected_capture:
        raise ValueError(f"capture CSV does not match solution provenance: expected {expected_capture}")
    if f"_{REQUIRED_CAPTURE_SHELL}_" not in f"_{expected_capture}_":
        raise ValueError(
            f"this companion requires a {REQUIRED_CAPTURE_SHELL} capture, got {expected_capture}"
        )
    if not sol.get("closed", False):
        raise ValueError("solution JSON is not marked closed")
    if not sol.get("full_physics", False):
        raise ValueError("solution is not full-physics (Earth + Moon + Mars cruise gravity)")
    offset_d = float(sol.get("tendril_offset_d", np.inf))
    window_d = float(sol.get("window_d", np.inf))
    if offset_d < 0.0 or offset_d > min(window_d, OUTER_LEG_MAX_D):
        raise ValueError(f"handoff is not on the <= {OUTER_LEG_MAX_D:g} d outer leg: {offset_d:.3f} d")
    if not np.isclose(arr_et, float(sol["handoff_et"]), atol=0.1, rtol=0.0):
        raise ValueError("reconstructed arrival epoch differs from solution handoff_et")
    target = np.asarray(sol["X2_helio"], float)
    replay_r_miss = float(np.linalg.norm(cruise_y[-1, :3] - target[:3]))
    replay_v_miss = float(np.linalg.norm(cruise_y[-1, 3:] - target[3:]))
    if replay_r_miss > REPLAY_R_MISS_MAX_KM or replay_v_miss > REPLAY_V_MISS_MAX_KMPS:
        raise ValueError(
            "full-physics replay exceeds the closure bound: "
            f"r={replay_r_miss:.6f} km, v={replay_v_miss:.9f} km/s"
        )
    if not (c_et[0] <= arr_et <= c_et[-1]):
        raise ValueError("solution handoff epoch lies outside the capture trajectory")

    # Panel C begins at the exact physical handoff. The saved capture samples are dense but
    # generally do not land exactly on arr_et, so interpolate the real trajectory at arr_et,
    # prepend that state, and discard the unflown Hill-tip -> handoff prefix.
    handoff_r = np.asarray(CubicSpline(c_et, c_r, axis=0)(arr_et), float)
    after_handoff = c_et > arr_et
    c_et = np.concatenate([[arr_et], c_et[after_handoff]])
    c_r = np.vstack([handoff_r, c_r[after_handoff]])
    c_days = (c_et - arr_et) / 86400.0

    earth_orb = planet_orbit("399", dep_et, 365.25 * 86400.0)
    mars_orb = planet_orbit("4", arr_et, 687.0 * 86400.0)
    earth_at_t0 = spice.spkezr("399", e_et[0], "J2000", "NONE", "10")[0][:3]   # LEO start (escape start)
    earth_at_t1 = spice.spkezr("399", dep_et, "J2000", "NONE", "10")[0][:3]    # Earth-Hill exit (cruise start)
    mars_at_t2 = spice.spkezr("4", arr_et, "J2000", "NONE", "10")[0][:3]       # cruise arrival (capture handoff)
    mars_at_t3 = spice.spkezr("4", c_et[-1], "J2000", "NONE", "10")[0][:3]     # capture end (LMO insertion)

    def render_2d():
        fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 7.0), constrained_layout=True)

        # --- Panel A: Earth escape (Earth-centered J2000), km ---
        # INITIAL-ORBIT-PLANE projection (not SVD): faithful near-Earth so the spiral clears the
        # planet by its true ~0.12 R_earth gap instead of false projection crossings (20260617-2045).
        xyA, _ = orbit_plane_2d(e_r, e_v)
        xyA = rotate2d(xyA, ROT_A_DEG)                                                             # 180 deg in-plane rotation
        xyA = xyA[:, ::-1]
        lcA = colored_2d(axA, xyA, e_days, VIRIDIS, lw=LW_A)                                       # log-normalized viridis
        axA.add_patch(Circle((0.0, 0.0), R_EARTH, facecolor="blue", edgecolor="none", zorder=5))  # (6) to-scale Earth
        axA.scatter(*xyA[-1], c=T1_COLOR, zorder=6, **T1_KW)                                       # (11) sail @ t1 (red star; labelled in legend)
        axA.set_title("A. Earth escape (Earth-centered)", fontsize=19)
        # legend MOVED here from B (1 column) into A's large upper-left whitespace (occupancy
        # 20260618-09xx: the escape knot fills bottom-left, upper-left is empty). Proxy markers
        # mirror B's epoch dots; the one legend applies to all three panels.
        marker_handles = [
            Line2D([0], [0], marker="o", ls="none", mfc="gold", mec="orange", ms=14, label="Sun"),
            Line2D([0], [0], marker="o", ls="none", mfc="none", mec="blue", mew=1.8, ms=13, label="Earth @ $t_0$"),
            Line2D([0], [0], marker="o", ls="none", mfc="blue", mec="blue", ms=13, label="Earth @ $t_1$"),
            Line2D([0], [0], marker="o", ls="none", mfc="red", mec="red", ms=13, label="Mars @ $t_2$"),
            Line2D([0], [0], marker="o", ls="none", mfc="none", mec="red", mew=1.8, ms=13, label="Mars @ $t_3$"),
            Line2D([0], [0], marker="*", ls="none", mfc=T1_COLOR, mec="black", mew=0.8, ms=18, label="Sail @ $t_1$"),
            Line2D([0], [0], marker="s", ls="none", mfc=T2_COLOR, mec="black", mew=0.8, ms=12, label="Sail @ $t_2$"),
        ]
        # time-definition rows: $t_n$ symbol in the handle (left) column, definition in the label (right) column
        def_handles = [_LegendText("$t_0$"), _LegendText("$t_1$"), _LegendText("$t_2$")]
        def_labels = ["escape start", "transfer start", "capture start"]
        axA.legend(marker_handles + def_handles,
                   [h.get_label() for h in marker_handles] + def_labels,
                   loc="upper left", ncol=1, framealpha=0.4, fontsize=16,
                   title="Legend applies to all panels.",
                   title_fontproperties={"style": "italic", "size": 12},
                   handler_map={_LegendText: _LegendTextHandler()})
        if args.cutouts:
            add_zoom_cutout(fig, axA, xyA, e_days, lcA.norm, R_EARTH, "blue")                       # near-Earth zoom cutout (3 Earth-diameters, top-left)
        _hcbar(fig, axA, lcA, "days since $t_0$")

        # --- Panel B: interplanetary transfer (heliocentric), km ---
        axB.plot(earth_orb[:, 0], earth_orb[:, 1], color="darkblue", lw=LW_ORBIT, ls="--", alpha=0.85)  # (10)
        axB.plot(mars_orb[:, 0], mars_orb[:, 1], color="red", lw=LW_ORBIT, ls="--", alpha=0.85)         # (10)
        lcB = colored_2d(axB, cruise_y[:, :2], cruise_days, VIRIDIS, lw=LW_B)                            # log-normalized viridis
        axB.scatter(0.0, 0.0, c="gold", s=190, marker="o", edgecolor="orange", zorder=6)               # Sun (in panel-A legend)
        axB.scatter(earth_at_t0[0], earth_at_t0[1], facecolors="none", edgecolors="blue", linewidths=1.8,
                    s=DOT_S, zorder=6)                                                                  # Earth @ t0 (open circle)
        axB.scatter(earth_at_t1[0], earth_at_t1[1], c="blue", s=DOT_S, zorder=6)                        # Earth @ t1 (filled)
        axB.scatter(mars_at_t2[0], mars_at_t2[1], c="red", s=DOT_S, zorder=6)                           # Mars @ t2 (filled)
        axB.scatter(mars_at_t3[0], mars_at_t3[1], facecolors="none", edgecolors="red", linewidths=1.8,
                    s=DOT_S, zorder=6)                                                                  # Mars @ t3 (open circle)
        axB.scatter(cruise_y[0, 0], cruise_y[0, 1], c=T1_COLOR, zorder=8, **T1_KW_B)                    # (11) sail @ t1 (smaller red star)
        axB.scatter(cruise_y[-1, 0], cruise_y[-1, 1], c=T2_COLOR, zorder=8, **T2_KW_B)                  # (11) sail @ t2 (smaller green square, Mars shows behind)
        axB.set_title("B. Transfer (Sun-centered)", fontsize=19)                                        # (9)
        # legend moved to panel A; the t0..t3 dots here are explained there
        _hcbar(fig, axB, lcB, "days since $t_1$")

        # --- Panel C: Mars capture (Mars-centered J2000), km ---
        # SVD plane is already faithful near Mars here (0.0% false-inside-Mars crossings, vs 43.8%
        xyC, _ = best_plane_2d(c_r)
        xyC = rotate2d(xyC, 90.0)
        lcC = colored_2d(axC, xyC, c_days, VIRIDIS, lw=LW_C)
        axC.add_patch(Circle((0.0, 0.0), R_MARS, facecolor="red", edgecolor="none", zorder=5))   # (6) to-scale Mars
        # c_r/xyC now START at the exact interpolated handoff; the earlier Hill-tip prefix
        # has been removed, so the green square and colorbar day 0 are both physically t2.
        axC.scatter(*xyC[0], c=T2_COLOR, zorder=6, **T2_KW)                                      # sail @ t2 = TRUE handoff
        # (11) t3 / LMO marker removed
        axC.set_title("C. Mars capture (Mars-centered)", fontsize=19)
        if args.cutouts:
            add_zoom_cutout(fig, axC, xyC, c_days, lcC.norm, R_MARS, "red")                        # near-Mars zoom cutout (3 Mars-diameters, top-left)
        _hcbar(fig, axC, lcC, "days since $t_2$")

        # (1) identical box + aspect for all three; (4) limits adapt per-panel (datalim).
        for ax in (axA, axB, axC):
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="datalim")
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both", useMathText=True)  # (2) sci offset
            ax.tick_params(labelsize=16)
            ax.xaxis.get_offset_text().set_fontsize(14)
            ax.yaxis.get_offset_text().set_fontsize(14)

        axC.yaxis.tick_right()

        # all six x10^n offsets get a "km" suffix (axis labels removed). x stays INSIDE (bottom-right);
        place_offsets(fig, [
            (axA, (0.985, 0.015, "right", "bottom"), (0.0, 1.012, "left", "bottom")),
            (axB, (0.985, 0.015, "right", "bottom"), (0.0, 1.012, "left", "bottom")),
            (axC, (0.985, 0.015, "right", "bottom"), (1.0, 1.012, "right", "bottom")),
        ], unit="km")

        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi); print(f"wrote {out} @ {args.dpi} dpi, cutouts={args.cutouts}")

    render_2d()
    print(
        f"cruise: dep {t1} -> arr {t2}, D*={T_s/86400:.3f} d, "
        f"offset={offset_d:.3f} d, {len(cruise_y)} steps, |g|={g_norm:.2e}, "
        f"replay miss={replay_r_miss:.6f} km / {replay_v_miss*1e3:.6f} m/s"
    )


if __name__ == "__main__":
    main()
