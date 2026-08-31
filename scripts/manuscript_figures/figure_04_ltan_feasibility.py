"""Render eclipse-free LTAN ranges in altitude-polar coordinates.

  - ANGULAR axis = LTAN (hours), gridded; noon (12 h) at top, dusk (18 h) left,
    midnight (0 h) bottom, dawn (6 h) right.  screen angle = (h - 6) * 15 deg.
  - RADIAL axis = orbital altitude (km); one labelled circle per K shell.
  - Each shell's whole-year eclipse-free band is a thick arc at its altitude,
    spanning its LTAN (ascending-node) range -- arc length maps onto the hour grid.
  - Each shell's DESCENDING node (LTAN + 12 h) is a fainter mirror arc on the dawn
    (right) side -- the point-reflection of its LTAN band through Mars' centre --
    with a short radial arrow pointing at it and a "descending node" group label.
  - Small textured Mars at the centre (icon, not to scale).
  - A few yellow arrows at the top = incident sunlight (Sun direction).
  - Altitude labels curve along their bands, fanned across the lower arc.
  - Bands are read from the fine-season CSV used by the manuscript.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

REPO_ROOT = Path(__file__).resolve().parents[2]
BANDS_CSV = REPO_ROOT / "simulation_outputs" / "20260612_eclipsefree_bands.csv"
MARS_TEXTURE_PATH = REPO_ROOT / "data" / "manuscript_figures" / "mars_texture.jpg"
OUT_PNG = (
    REPO_ROOT / "figures" / "manuscript" / "generated" / "figure_04_ltan_feasibility.png"
)

# ---- layout knobs ----------------------------------------------------------
FIG_W_IN, FIG_H_IN = 9.0, 9.4
SAVE_DPI = 320
R_MARS_KM = 3396.19        # IAU_MARS equatorial radius -> Mars drawn to scale
RMAX_FACTOR = 1.12         # outer radial limit = (max orbital radius) x this
ARC_LW = 5.0
DN_ARC_LW = 3.4            # descending-node (mirror) arc line width (thinner than AN)
DN_ARC_ALPHA = 0.5         # descending-node arcs drawn fainter than the AN bands
DN_ARROW_LEN_KM = 90.0     # radial length of the tiny "almost-removed" DN-arc nudge arrow
DN_ARROW_LW = 1.0          # thin DN arrow line width
ALT_LABEL_OFFSET_H = 0.7   # altitude labels sit +/- this many hours either side of LTAN=0
ALT_LABEL_FONTSIZE = 18.0  # curved altitude labels on the bands
NODE_LABEL_FONTSIZE = 21.0 # "ascending/descending node" labels (straight, vertical, side edges)
EDGE_TICK_KM = 70.0        # half-length of the radial band-edge ticks
ANG_GRID_STEP_H = 2.0      # angular gridline spacing (hours)
PYVISTA_SUPERSAMPLE = 2.0
_PV_BASE_PX = 900

_CMAP = plt.get_cmap("plasma")
_CMAP_SAMPLES = {12: 0.08, 11: 0.34, 10: 0.55, 9: 0.72}
TXT = "#1a1a1a"
GRID = "#c7ccd6"


def read_bands(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({"K": int(row["K"]), "a_km": float(row["a_km"]),
                         "alt_km": float(row["altitude_km"]),
                         "lo": float(row["ltan_lo_h"]), "hi": float(row["ltan_hi_h"]),
                         "width": float(row["width_h"])})
    return rows


def hour_to_theta(h: float) -> float:
    """LTAN hour -> polar theta (rad), paired with theta_zero_location('N') +
    theta_direction(1): noon(12 h) top, dusk(18 h) left, midnight(0 h) bottom,
    dawn(6 h) right."""
    return math.radians((h - 12.0) * 15.0)


def curved_text(ax, fig, hour_center: float, r: float, s: str, color: str,
                fontsize: float):
    """Render `s` CURVED along the radius-r circle, centred on hour_center, each
    glyph individually rotated to the local tangent so the word follows the arc.
    Reads left->right (increasing theta -> increasing screen angle in this polar
    setup).  Needs a prior fig.canvas.draw() (renderer + transData realised)."""
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    widths = []
    for ch in s:
        if ch == " ":
            widths.append(fontsize * 0.42 * px_per_pt)
        else:
            t = ax.text(0, 0, ch, fontsize=fontsize)
            widths.append(float(t.get_window_extent(renderer=renderer).width))
            t.remove()
    c = np.asarray(ax.transData.transform((0.0, 0.0)))
    e = np.asarray(ax.transData.transform((0.0, r)))
    R_px = float(np.hypot(*(e - c)))                 # circle display radius (px)
    dthetas = [w / R_px for w in widths]             # theta == display angle in polar
    theta = hour_to_theta(hour_center) - sum(dthetas) / 2.0
    for ch, dth in zip(s, dthetas):
        th_c = theta + dth / 2.0
        p0 = np.asarray(ax.transData.transform((th_c, r)))
        p1 = np.asarray(ax.transData.transform((th_c + 0.01, r)))
        ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        ax.text(th_c, r, ch, rotation=ang, rotation_mode="anchor", ha="center",
                va="center", color=color, fontsize=fontsize, zorder=9,
                transform=ax.transData)
        theta += dth


def render_mars_poleon_rgba(px: int) -> np.ndarray:
    """Small textured Mars, north pole toward viewer, transparent background."""
    import pyvista as pv
    R, n, m = 1.0, 420, 210
    phi = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0.0, np.pi, m)
    PHI, TH = np.meshgrid(phi, th, indexing="ij")
    pts = np.column_stack([(R * np.sin(TH) * np.cos(PHI)).ravel(),
                           (R * np.sin(TH) * np.sin(PHI)).ravel(),
                           (R * np.cos(TH)).ravel()])
    u = (PHI / (2 * np.pi) + 0.5).ravel()
    v = (1.0 - TH / np.pi).ravel()
    ii, jj = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    a = (ii * m + jj).ravel(); b = ((ii + 1) * m + jj).ravel()
    c = ((ii + 1) * m + jj + 1).ravel(); d = (ii * m + jj + 1).ravel()
    faces = np.column_stack([np.full(a.size, 4, np.int64), a, b, c, d]).ravel()
    mesh = pv.PolyData(pts, faces=faces)
    mesh.active_texture_coordinates = np.column_stack([u, v])
    mesh.compute_normals(inplace=True, auto_orient_normals=True)
    pl = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
    tex = pv.read_texture(MARS_TEXTURE_PATH) if Path(MARS_TEXTURE_PATH).exists() else None
    if tex is not None:
        pl.add_mesh(mesh, texture=tex, smooth_shading=True, ambient=0.72, diffuse=0.42)
    else:
        pl.add_mesh(mesh, color="#b05a3c", smooth_shading=True, ambient=0.72, diffuse=0.42)
    light = pv.Light(position=(0.0, 2.0, 2.5), focal_point=(0.0, 0.0, 0.0), color="white")
    light.positional = False
    light.intensity = 0.5
    pl.add_light(light)
    pl.enable_parallel_projection()
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.position = (0.0, 0.0, 6.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = 1.0
    img = pl.screenshot(return_img=True, transparent_background=True)
    pl.close()
    return img


def main() -> None:
    bands = read_bands(BANDS_CSV)
    bands.sort(key=lambda d: d["alt_km"])
    print("eclipse-free bands (fine-season, from CSV):")
    for d in bands:
        print(f"  K={d['K']:2d}  alt {d['alt_km']:7.1f} km  "
              f"L=[{d['lo']:.3f},{d['hi']:.3f}] h  width {d['width']:.3f} h")

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    fig.patch.set_facecolor("white")
    AX_RECT = [0.07, 0.075, 0.86, 0.80]    # headroom for arrows/title above, note below
    ax = fig.add_axes(AX_RECT, projection="polar")
    ax.set_facecolor("white")
    ax.set_theta_zero_location("N")        # LTAN 12 h (noon) at the top
    ax.set_theta_direction(1)              # CCW: dusk 18 h left, dawn 6 h right

    # RADIAL coordinate = distance from Mars centre (km) so Mars and the orbit
    # shells are mutually TO SCALE: r = R_Mars + altitude = orbital radius a.
    rmax = max(d["a_km"] for d in bands) * RMAX_FACTOR
    ax.set_ylim(0, rmax)

    # angular grid = LTAN hours (the readable scale); mod 360 -> full clock
    hrs = np.arange(0, 24, ANG_GRID_STEP_H).astype(int)
    ax.set_thetagrids([((h - 12) * 15) % 360 for h in hrs],
                      labels=[f"{h}" for h in hrs], fontsize=15)
    # radial grid = one circle per shell at its orbital radius; the altitude
    # labels are drawn separately (curved, on the bands), so blank these
    ax.set_rgrids([d["a_km"] for d in bands], labels=[""] * len(bands))
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # ---- band arcs at each shell's orbital radius, spanning the LTAN range ------
    for d in bands:
        col = _CMAP(_CMAP_SAMPLES[d["K"]])
        th = np.linspace(hour_to_theta(d["lo"]), hour_to_theta(d["hi"]), 80)
        ax.plot(th, np.full_like(th, d["a_km"]), color=col, lw=ARC_LW,
                solid_capstyle="round", zorder=6)
        for h in (d["lo"], d["hi"]):
            t = hour_to_theta(h)
            ax.plot([t, t], [d["a_km"] - EDGE_TICK_KM, d["a_km"] + EDGE_TICK_KM],
                    color=col, lw=1.8, zorder=6, solid_capstyle="butt")

    # ---- descending-node (mirror) arcs at LTAN + 12 h, on the dawn (right) side -
    # A sun-synchronous orbit crosses the equator southbound 12 h of local time from
    # its ascending node, so each shell's descending-node band is the point-reflection
    # of its LTAN band through Mars' centre (hour_to_theta(h+12) = hour_to_theta(h)+pi).
    # Drawn fainter in the same shell colour, each with a short radial arrow pointing
    # at it (the "descending node" group label is added, curved, further below).
    for d in bands:
        col = _CMAP(_CMAP_SAMPLES[d["K"]])
        th = np.linspace(hour_to_theta(d["lo"] + 12.0), hour_to_theta(d["hi"] + 12.0), 80)
        ax.plot(th, np.full_like(th, d["a_km"]), color=col, lw=DN_ARC_LW,
                alpha=DN_ARC_ALPHA, solid_capstyle="round", zorder=5)
        for h in (d["lo"] + 12.0, d["hi"] + 12.0):
            t = hour_to_theta(h)
            ax.plot([t, t], [d["a_km"] - EDGE_TICK_KM, d["a_km"] + EDGE_TICK_KM],
                    color=col, lw=1.4, alpha=DN_ARC_ALPHA, zorder=5,
                    solid_capstyle="butt")
        th_c = hour_to_theta(0.5 * (d["lo"] + d["hi"]) + 12.0)
        ax.annotate("", xy=(th_c, d["a_km"] + EDGE_TICK_KM),
                    xytext=(th_c, d["a_km"] + EDGE_TICK_KM + DN_ARROW_LEN_KM),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=DN_ARROW_LW,
                                    alpha=DN_ARC_ALPHA, mutation_scale=8.0,
                                    shrinkA=0.0, shrinkB=0.0), zorder=7)

    # ---- incident sunlight: parallel VERTICAL arrows across the top ------------
    for xf in (0.32, 0.41, 0.50, 0.59, 0.68):
        ax.annotate("", xy=(xf, 0.905), xytext=(xf, 0.963),
                    xycoords="figure fraction", textcoords="figure fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#e6a91f", lw=3.4),
                    zorder=8)

    # ---- Mars, drawn TO SCALE (radius R_Mars in the radial coordinate) ---------
    mars = render_mars_poleon_rgba(int(_PV_BASE_PX * PYVISTA_SUPERSAMPLE))
    fig.canvas.draw()                      # realise transforms before measuring
    c_disp = ax.transData.transform((0.0, 0.0))
    e_disp = ax.transData.transform((math.pi / 2.0, R_MARS_KM))
    r_px = float(np.hypot(e_disp[0] - c_disp[0], e_disp[1] - c_disp[1]))
    cx_f, cy_f = fig.transFigure.inverted().transform(c_disp)
    w_f, h_f = 2 * r_px / fig.bbox.width, 2 * r_px / fig.bbox.height
    mars_ax = fig.add_axes([cx_f - w_f / 2, cy_f - h_f / 2, w_f, h_f], zorder=11)
    mars_ax.imshow(mars, interpolation="bilinear")
    mars_ax.patch.set_alpha(0.0)
    mars_ax.axis("off")
    mars_ax.text(0.5, 0.5, "N. Pole", transform=mars_ax.transAxes, ha="center",
                 va="center", color="black", fontsize=15, fontweight="bold",
                 zorder=13, path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    # ---- altitude labels: CURVED on each band, straddling the LTAN=0 (midnight,
    # straight-down) line, ALTERNATING left/right of it band-to-band so successive
    # labels don't stack (inner K12 left, K11 right, K10 left, K9 right). ----------
    label_hours = [(-ALT_LABEL_OFFSET_H if i % 2 == 0 else ALT_LABEL_OFFSET_H)
                   for i in range(len(bands))]
    for d, hlab in zip(bands, label_hours):
        col = _CMAP(_CMAP_SAMPLES[d["K"]])
        curved_text(ax, fig, hlab, d["a_km"], f"{d['alt_km']:.0f} km", col,
                    ALT_LABEL_FONTSIZE)

    # ---- node group labels: STRAIGHT vertical text on the two side edges ---------
    # NOT curved: curved_text flips each glyph to the local tangent, and on the side
    # edges (6 h / 18 h) the tangent is vertical, so glyphs straddling the horizontal
    # midline get opposite flips -- the word "reflected" across the midline. Single
    # upright vertical labels read cleanly: ascending node on the dusk (left, 18 h)
    # side where the LTAN bands live, descending node on the dawn (right, 6 h) side.
    ax.text(hour_to_theta(18.0), rmax * 0.985, "ascending node", rotation=90,
            rotation_mode="anchor", ha="center", va="center", color="#6b7280",
            fontsize=NODE_LABEL_FONTSIZE, zorder=9, clip_on=False)
    ax.text(hour_to_theta(6.0), rmax * 0.985, "descending node", rotation=90,
            rotation_mode="anchor", ha="center", va="center", color="#6b7280",
            fontsize=NODE_LABEL_FONTSIZE, zorder=9, clip_on=False)

    # ---- minimal labels --------------------------------------------------------
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_PNG,
        dpi=SAVE_DPI,
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {OUT_PNG}  ({FIG_W_IN}x{FIG_H_IN} in @ {SAVE_DPI} dpi)")


if __name__ == "__main__":
    main()
