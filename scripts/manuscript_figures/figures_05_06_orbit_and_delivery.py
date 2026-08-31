"""Render the station-keeping and delivery attitude figures.

Each figure pairs cone and clock angles versus time with a three-dimensional
Mars scene at the corresponding labeled states. Orientation coordinates follow
McInnes (1999), Eq. 4.7, p. 116:
  cone  = angle(sail normal n, Sun-line s)
  clock = azimuth of n about s, measured FROM the orbit normal's projection
          into the plane normal to s, increasing toward (e_A x s).

The clock is instantaneous and orbit-normal-referenced. The reflector roll is
illustrative because the trajectory determines the sail normal but not roll:
the sail's local +u1 edge is the orbit-normal direction projected into the sail
plane.  The asymmetric "dog-ear" corner marks +u1+u2 so roll is visible.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import PchipInterpolator

from reflectors.kernels import load_kernels, kernels_available

# Times New Roman everywhere (serif math for the alpha/delta glyphs).
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

# ===========================================================================
# PARAMETERS
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_OUT = REPO_ROOT / "simulation_outputs"          # INPUT: trajectory CSV lives here
FIG_DIR = REPO_ROOT / "figures" / "manuscript" / "generated"
CSV_STEM = "20260507_slice17_halfmarsyear_rvclos_pi_fluence_reward_60s_de_pjac_polish_de_chain"
SOL = 1  # both figures read sol-1 trajectory

# --- clock convention -------------------------------------------------------
CLOCK_SIGN = +1          # +1 = McInnes sense (e_B = e_A x s); -1 = archived (e_B = s x e_A)
UNWRAP_CLOCK = False     # False -> proper azimuth wrapped to [0,360) (jumps visibly at the wrap);
                         # True -> np.unwrap within the segment (smooth but can exceed [0,360))
CLOCK_YTICKS = [0, 90, 180, 270, 360]  # clock axis ticks when wrapped to [0,360)
XLIM_PAD_MIN = 2.0       # Fig B left-panel x-limits = [first sample - pad, last sample + pad]
FIGA_LEFT_MARGIN_MIN = 3.0  # Fig A: small gap left of t=0 (leftmost tick stays 0); right edge = full
                            # orbit-1 data end, so the curve runs to orbital closure.
# Plotted curves are PCHIP-interpolated onto a fine time grid, then wrapped, so the
# clock's 360->0 wrap is a clean (near-vertical) jump rather than a slanted line
# spanning one coarse sample interval.  Markers stay on the real data points.
FINE_GRID_N = 4000       # samples on the fine plotting grid (per segment)

# --- Figure A: typical cruise orbit ----------------------------------------
FIGA_ORBIT_INDEX = 1     # 1-based orbit number within the sol (pick a pure-cruise orbit)
FIGA_N_SNAPSHOTS = 8     # equally spaced in time across the orbit
FIGA_OUT = FIG_DIR / "figure_05_stationkeeping_orbit.png"
# Whitespace around the Mars+orbit render cut. The render content fills
# 0.759 of its axes width and 0.830 of its height (outermost ink at
# 0.863*view_half_span), with the square render letterboxed into a 6.92x6.19 in axes.
FIGA_NUMBER_ABOVE_FRAME = True  # place the 1..8 labels just above the frame (blended
                                # transform) instead of at NUMBER_Y_DEG_A inside it
# Canvas retuned to open headroom for those labels WITHOUT changing the orientation
# panel's height: (0.9287-0.1115)*7.58 = 6.19 in, exactly matching Fig A's previous
# above the panel = (1-0.9287)*7.58 = 0.54 in for the numbers.
# The bottom margin is 0.845 in (was 0.72): the enlarged AXIS_LABEL/TICK fonts made
# the xlabel+ticks stack ~0.74 in, which exceeded the smaller margin and let matplotlib
# silently crop the x-axis label at the canvas edge. A 0.845 in margin restores
# approximately 0.107 in of clearance.
FIGA_FIG_HEIGHT = 7.58
FIGA_FIG_TOP = 0.9287
FIGA_FIG_BOTTOM = 0.1115
FIGA_RENDER_ZOOM = 1.15  # 1/0.863 would exactly touch the frame; 1.15 puts the outermost
                         # ink at ~0.955 of the frame half-height (~4.5% clearance)
# Narrow the right column so the render axes (6.29 in) is only just wider than its
# 6.19 in height -- imshow still fits the square render to HEIGHT (so the zoom above
# is what sets the content size) while the ~0.9 in of reclaimed width goes to the
# left panel (7.27 -> 8.17 in).  Fig B keeps the original values.
FIGA_WIDTH_RATIOS = (1.30, 1.0)
FIGA_WSPACE = 0.14

# --- Figure B: delivery event ----------------------------------------------
# Delivery event 1 of sol-1 spans t ~ 369-403 min (slew_in/bisector/slew_out).
FIGB_T_START_MIN = 365.0
FIGB_T_END_MIN = 407.0
FIGB_N_SNAPSHOTS = 12    # poses sampled EVENLY in time across [START, END]
FIGB_OUT = FIG_DIR / "figure_06_delivery_orbit.png"
# the panel, and the 3D render ~10% larger.  The taller canvas + adjusted top/bottom
# keep the orientation panel's height equal to Fig A's (6.19 in) while opening ~1.09 in
# of headroom for the role bar.
# The 0.845 in bottom margin keeps the enlarged xlabel and ticks within the canvas,
# holding panel height at (0.8661-0.1041)*8.12 = 6.19 in and headroom at 1.09 in.
FIGB_FIG_HEIGHT = 8.12
FIGB_FIG_TOP = 0.8661
FIGB_FIG_BOTTOM = 0.1041
FIGB_RENDER_ZOOM = 1.25  # render scene ~25% larger (shrinks the view frame -> fills more, trims whitespace)
FIGB_RENDER_UP_SHIFT_FRAC = 0.08  # pan the render up by this * view_half_span (uses top margin -> protects the lower sails)
ROLE_BAR_LABELS = {"slew_in": "slew", "slew_out": "slew", "bisector": "illuminate"}
ROLE_BAR_FONTSIZE = 20   # the <-- label --> role annotations above Fig B
# now sit above the frame here too (FIGB_NUMBER_ABOVE_FRAME).  A NUMBER_FONTSIZE=20
# glyph spans ~0.052 in axes-fraction of the 6.19 in (445.7 pt) panel, so the numbers
# occupy ~1.02-1.072; the bar's own 20 pt text spans +-0.026 about its line.
ROLE_BAR_Y = 1.115       # axes-fraction y of the role-phase arrows (above the numbers)
FIGB_MARKER_MS_CONE = 11
FIGB_MARKER_MS_CLOCK = 10
FIGB_NUMBER_ABOVE_FRAME = True   # snapshot numbers 1..12 above the frame, as in Fig A

# --- render --------------------------------------------------------------
MARS_R_KM = 3396.19          # IAU_MARS equatorial radius (mars_constants)
MARS_TEXTURE_PATH = REPO_ROOT / "data" / "manuscript_figures" / "mars_texture.jpg"
# equirectangular 8192x4096 jpg; per-face UV sampling per sso-stability/visualize_sail.py.
# Mesh built in IAU_MARS body frame then rotated to J2000 via spice.pxform at the
# illumination epoch, so the geography under the sail is real. UV prime meridian is at
# the IMAGE CENTRE (u = lon/2pi + 0.5), per sso-stability/visualize_sail_3d.py.
MARS_MESH_NU = 360           # matplotlib-fallback longitude faces (PyVista path ignores these)
MARS_MESH_NV = 180           # matplotlib-fallback colatitude faces

# --- render backend -------------------------------------------------------
USE_PYVISTA = True           # PyVista (GPU texture, smooth) -> screenshot into the panel; else matplotlib
RENDER_BG = "white"          # PyVista background (blends with the figure)
SAVE_DPI = 300               # matplotlib savefig DPI for the composed figure
# The 3D panel is a raster screenshot, so its crispness is set by the PyVista
# window px, NOT by SAVE_DPI.  Supersample renders it larger; the window px AND
# every in-render label font scale by the SAME factor, so appearance is unchanged
# -- only the pixel count (and thus sharpness in the high-DPI composite) grows.
PYVISTA_SUPERSAMPLE = 1.6
_PV_BASE_PX = 1500
PYVISTA_WINDOW_PX = (int(_PV_BASE_PX * PYVISTA_SUPERSAMPLE), int(_PV_BASE_PX * PYVISTA_SUPERSAMPLE))
PYVISTA_SPHERE_RES = 360     # PyVista sphere theta/phi resolution (GPU-interpolated -> smooth)
PYVISTA_NUMBER_FONT = int(46 * PYVISTA_SUPERSAMPLE)
PYVISTA_TARGET_FONT = int(36 * PYVISTA_SUPERSAMPLE)
PYVISTA_GNOMON_FONT = int(24 * PYVISTA_SUPERSAMPLE)  # Sun-gnomon label font (unused: gnomon removed, SHOW_SUN_GNOMON_3D=False)
TARGET_LAT_DEG = 40.0
TARGET_LON_DEG = 200.0
SHOW_TARGET_MARKER = True
FIGA_SAIL_SIZE_KM = 700.0    # EXAGGERATED for visibility (real sail ~tens of m)
FIGB_SAIL_SIZE_KM = 380.0    # smaller — Fig B samples densely along one arc
NORMAL_ARROW_FACTOR = 1.4    # sail-normal arrow length = factor * sail size
REF_ARROW_KM = 4500.0        # length of the Sun / orbit-normal reference arrows

# --- render camera --------------------------------------------------------
VIEW_ALONG = "orbit_normal"  # camera looks down this axis (out of screen toward viewer): "orbit_normal" | "sun"
VIEW_TILT_DEG = 15.0         # tilt off the exact axis (elevation) for depth
VIEW_AZIMUTH_LEFT_DEG = 0.0  # orbit the camera left about the up axis (0 = orbit normal dead-on, facing the viewer)
ARROW_LEN_FRAC = 0.5         # 3D Sun/orbit-normal arrow length = this * view_half_span (from the anchor)
SHOW_REF_ARROWS_3D = False   # central 3D Sun + orbit-normal arrows in the scene (removed; see Sun corner gnomon)
SHOW_CORNER_INDICATOR = False  # 2D Sun/orbit-normal direction indicator in the panel corner (toggle)
# 3D Sun-direction gnomon, anchored in a screen corner (true 3D arrow along s_mid,
# so it foreshortens correctly as the Sun-line tilts relative to the view).
SHOW_SUN_GNOMON_3D = False
SUN_GNOMON_FRAC = 0.20       # gnomon arrow length = this * view_half_span
SUN_GNOMON_CORNER = (-0.78, -0.70)  # (right, up) offset of the gnomon base, in units of view_half_span
FIGA_VIEW_MARGIN_KM = 120.0  # Fig A: frame = (ring+label) extent + this margin, centred on Mars
FIGA_ZOOM = 0.98             # Fig A: shrink the frame by this factor (smaller = more zoomed-in)
FIGB_VIEW_MARGIN_KM = 150.0  # Fig B: frame = (snapshot+label) extent + this margin, centred on the arc
FIGB_ZOOM = 0.93             # Fig B: shrink the frame by this factor (smaller = more zoomed-in crop)
LABEL_RADIAL_OFFSET_FACTOR = 1.0  # snapshot number sits this * sail_size radially OUTSIDE each sail
SPHERE_CAP_DEG = 92.0        # Fig B: draw Mars as a curved cap this far from the arc direction (no flat cut)

# --- left-panel styling ---------------------------------------------------
# 14->16, Mars 16->20, axis labels 12->16, tick labels ->13 (AXIS_LABEL/TICK below).
# 16->20, tick labels 13->16 (both figures; legend also moved to bottom-right on Fig A).
# 20->25, ticks 16->20.  Kept SHARED across Fig A and Fig B so the paper pair stays
# typographically consistent; Fig A's wider left panel (see FIGA_WIDTH_RATIOS) gives
# the larger text room to breathe.
LEGEND_FONTSIZE = 22
NUMBER_FONTSIZE = 20         # snapshot numbers
MARS_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 25     # x/y axis titles on the orientation panel
TICK_LABEL_FONTSIZE = 20     # numeric tick labels on both panels
NUMBER_Y_DEG_A = 15.0        # Fig A: in-panel fallback height for the snapshot numbers on the
                             # cone (left) axis -- UNUSED while FIGA_NUMBER_ABOVE_FRAME is True
NUMBER_Y_DEG_B = 65.0        # Fig B: higher, to clear the deep cone dip during the delivery arc
NUMBER_ABOVE_FRAME_Y = 1.02  # axes-fraction y for above-frame snapshot numbers (blended transform)
MARKER_MS_CONE = 7           # default snapshot marker size on the cone axis (circles)
MARKER_MS_CLOCK = 6          # default snapshot marker size on the clock axis (squares)
ROLE_SHADE_ALPHA = 0.10      # alpha of the role background shading in the orientation panel

ROLE_COLORS = {
    "cruise": "#3b7dd8",
    "slew_in": "#e8a33d",
    "bisector": "#cc3b3b",
    "slew_out": "#7a3bd8",
}
ROLE_DISPLAY = {"cruise": "station-keep", "slew_in": "slew in",
                "bisector": "illuminate", "slew_out": "slew out"}


# ===========================================================================
# Load
# ===========================================================================

def load_sol(sol: int) -> dict[str, np.ndarray]:
    path = SIM_OUT / f"{CSV_STEM}_sol{sol}_trajectory.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    for key in ("t_s", "et_tdb", "r_x_km", "r_y_km", "r_z_km",
                "s_hat_x", "s_hat_y", "s_hat_z", "h_hat_x", "h_hat_y", "h_hat_z",
                "n_x", "n_y", "n_z", "alpha_cone_deg", "u_transport_deg"):
        out[key] = np.array([float(r[key]) for r in rows])
    out["schedule_role"] = np.array([r["schedule_role"] for r in rows], dtype=object)
    out["r"] = np.column_stack([out["r_x_km"], out["r_y_km"], out["r_z_km"]])
    out["s"] = np.column_stack([out["s_hat_x"], out["s_hat_y"], out["s_hat_z"]])
    out["h"] = np.column_stack([out["h_hat_x"], out["h_hat_y"], out["h_hat_z"]])
    out["n"] = np.column_stack([out["n_x"], out["n_y"], out["n_z"]])
    out["t_min"] = out["t_s"] / 60.0
    print(f"loaded sol {sol}: {len(out['t_s'])} samples, "
          f"t {out['t_min'][0]:.1f}..{out['t_min'][-1]:.1f} min")
    return out


# ===========================================================================
# McInnes Eq 4.7 cone + clock (orbit-normal referenced, instantaneous)
# ===========================================================================

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def mcinnes_cone_clock(n: np.ndarray, s: np.ndarray, h: np.ndarray,
                       sign: int = +1) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """Textbook McInnes 1999 Eq 4.7 cone/clock, plus the in-plane basis (e_A, e_B).

    cone  = acos(n . s)
    e_A   = unit(h - (h.s) s)              (orbit normal projected perp to Sun-line)
    e_B   = sign * unit(e_A x s)           (+1 = McInnes; -1 = archived s x e_A)
    clock = atan2(n . e_B, n . e_A)        (deg)
    Returns (cone_deg, clock_deg, e_A, e_B).
    """
    s = _unit(s)
    n = _unit(n)
    cone = np.degrees(np.arccos(np.clip(np.sum(n * s, axis=-1), -1.0, 1.0)))
    e_A = _unit(h - np.sum(h * s, axis=-1, keepdims=True) * s)
    e_B = sign * _unit(np.cross(e_A, s))
    clock = np.degrees(np.arctan2(np.sum(n * e_B, axis=-1),
                                  np.sum(n * e_A, axis=-1)))
    return cone, clock, e_A, e_B


def clock_for_segment(clock_deg: np.ndarray) -> np.ndarray:
    if UNWRAP_CLOCK:
        return np.degrees(np.unwrap(np.radians(clock_deg)))
    return clock_deg % 360.0


# ===========================================================================
# Orbit chunking (by argument of latitude u advancing 360 deg)
# ===========================================================================

def orbit_bounds(u_deg: np.ndarray, orbit_index_1based: int) -> tuple[int, int]:
    edges = [0]
    anchor = u_deg[0]
    for i in range(len(u_deg)):
        if u_deg[i] >= anchor + 360.0:
            edges.append(i)
            anchor += 360.0
    edges.append(len(u_deg) - 1)
    k = orbit_index_1based - 1
    return edges[k], edges[k + 1]


# ===========================================================================
# Matplotlib fallback renderer
# ===========================================================================

def _sail_quad(center: np.ndarray, n_hat: np.ndarray, roll_ref: np.ndarray,
               size: float, notch: float = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sail face as a NOTCHED pentagon + the triangle filling the notch.

    The asymmetric "dog-ear" corner (+u1+u2) is CUT OUT of the coloured face
    (pentagon) and the white triangle fills that cut.  Face and dog-ear are
    coplanar but ADJACENT (non-overlapping), so matplotlib's one-depth-per-
    collection sort never hides the dog-ear behind the face, so it stays white
    at every orientation. A lifted
    overlapping triangle could not: it sits at a corner, deeper than the face
    centroid on a tilted sail, so the face still sorts in front.

    Roll convention: u1 = roll_ref projected into the sail plane, with roll_ref =
    the orbit normal h (see callers).  This anchors the dog-ear to the SAME
    physical reference the McInnes clock is measured from (clock's e_A is h
    projected perpendicular to the Sun-line).  So the dog-ear's +u1 edge marks
    the orbit-normal direction, and clock=0 (normal leaning toward +e_A, the
    orbit-normal side) corresponds to a specific, repeatable dog-ear pose --
    the clock angle and the visible dog-ear are anchored together, not free.
    u2 = n_hat x u1; the asymmetric corner sits at +u1+u2.
    """
    n_hat = n_hat / np.linalg.norm(n_hat)
    u1 = roll_ref - np.dot(roll_ref, n_hat) * n_hat
    if np.linalg.norm(u1) < 1e-9:
        u1 = np.array([1.0, 0.0, 0.0]) - n_hat[0] * n_hat
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(n_hat, u1)
    h = size / 2.0
    c = center
    A = c - h * u1 - h * u2
    B = c + h * u1 - h * u2
    C = c + h * u1 + h * u2
    D = c - h * u1 + h * u2
    P = C - notch * size * u1   # on the top edge (toward D)
    Q = C - notch * size * u2   # on the right edge (toward B)
    face = np.array([A, B, Q, P, D])   # pentagon (square minus the C corner)
    ear = np.array([C, P, Q])          # triangle filling the notch
    return face, ear, n_hat


_TEX_CACHE: dict = {}


def _load_mars_texture():
    """Equirectangular Mars texture as float HxWx3 in [0,1], cached; None if absent."""
    if "img" in _TEX_CACHE:
        return _TEX_CACHE["img"]
    p = Path(MARS_TEXTURE_PATH)
    img = None
    if p.exists():
        arr = plt.imread(str(p)).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        img = arr
        print(f"  loaded Mars texture {p.name}: {arr.shape[1]}x{arr.shape[0]} px")
    else:
        print(f"  Mars texture not found ({p}); falling back to solid colour")
    _TEX_CACHE["img"] = img
    return img


def _mars_facecolors(uu: np.ndarray, vv: np.ndarray):
    """Per-face RGBA from the texture for a sphere meshed by uu (IAU_MARS east
    longitude) x vv (colatitude).  UV: u = lon/2pi + 0.5 (prime meridian at image
    CENTRE, per sso-stability), v = colat/pi (v=0 north pole = image top)."""
    img = _load_mars_texture()
    if img is None:
        return None
    H, W = img.shape[:2]
    lon_mid = 0.5 * (uu[:-1] + uu[1:])      # face centres, longitude
    colat_mid = 0.5 * (vv[:-1] + vv[1:])    # face centres, colatitude
    px = np.clip(((lon_mid / (2 * np.pi) + 0.5) % 1.0 * (W - 1)).astype(int), 0, W - 1)
    py = np.clip((colat_mid / np.pi * (H - 1)).astype(int), 0, H - 1)
    px_g = np.broadcast_to(px[:, None], (len(px), len(py)))
    py_g = np.broadcast_to(py[None, :], (len(px), len(py)))
    fc = np.ones((len(px), len(py), 4), dtype=np.float32)
    fc[..., :3] = img[py_g, px_g, :3]
    return fc


_KERNELS_LOADED: dict = {}


def _mars_rotation_to_j2000(et: float) -> np.ndarray:
    """3x3 IAU_MARS -> J2000 rotation at ephemeris time et (identity if kernels
    are unavailable, so the figure still renders)."""
    if not _KERNELS_LOADED.get("done"):
        ok = kernels_available()
        if ok:
            load_kernels()
        else:
            print("  SPICE kernels unavailable; Mars geography NOT oriented (identity)")
        _KERNELS_LOADED["done"] = True
        _KERNELS_LOADED["ok"] = ok
    if not _KERNELS_LOADED.get("ok"):
        return np.eye(3)
    return np.asarray(spice.pxform("IAU_MARS", "J2000", float(et)))


def render_matplotlib(ax, traj: dict, seg: slice, snap_idx: list[int],
                       labels: list[str], snap_colors: list[str],
                       sail_size_km: float, view_center: np.ndarray,
                       view_half_span: float, clip_sphere: bool,
                       label_offset_km: float, show_ref_arrows: bool,
                       et_ref: float) -> None:
    normal_len = NORMAL_ARROW_FACTOR * sail_size_km
    ref_len = 0.9 * view_half_span
    r = traj["r"]
    mid = (seg.start + (seg.stop or len(r))) // 2
    s_mid = traj["s"][mid]
    h_mid = traj["h"][mid]

    # camera FIRST (so we can cull Mars's far hemisphere). Looks down VIEW_ALONG.
    axis_vec = {"orbit_normal": h_mid, "sun": s_mid}.get(VIEW_ALONG)
    if axis_vec is not None:
        elev = math.degrees(math.asin(float(np.clip(axis_vec[2], -1.0, 1.0)))) - VIEW_TILT_DEG
        azim = math.degrees(math.atan2(axis_vec[1], axis_vec[0]))
    else:
        elev, azim = 22.0, -60.0
    er, ar = math.radians(elev), math.radians(azim)
    cam_dir = np.array([math.cos(er) * math.cos(ar),
                        math.cos(er) * math.sin(ar), math.sin(er)])

    # Mars: TEXTURED sphere, meshed in IAU_MARS body frame then rotated to J2000
    # at et_ref -> geography under the sail (esp. the illumination pass) is real.
    uu = np.linspace(0, 2 * np.pi, MARS_MESH_NU)   # IAU_MARS east longitude
    vv = np.linspace(0, np.pi, MARS_MESH_NV)       # colatitude (0 = north pole)
    sv = np.sin(vv)
    xb = MARS_R_KM * np.outer(np.cos(uu), sv)
    yb = MARS_R_KM * np.outer(np.sin(uu), sv)
    zb = MARS_R_KM * np.outer(np.ones_like(uu), np.cos(vv))
    Rmat = _mars_rotation_to_j2000(et_ref)
    Pj = np.einsum("ij,uvj->uvi", Rmat, np.stack([xb, yb, zb], axis=-1))
    xs, ys, zs = Pj[..., 0], Pj[..., 1], Pj[..., 2]
    facecolors = _mars_facecolors(uu, vv)
    if facecolors is not None:
        # back-face cull: hide faces whose outward normal points away from camera
        fcen = 0.25 * (Pj[:-1, :-1] + Pj[1:, :-1] + Pj[:-1, 1:] + Pj[1:, 1:])
        fnrm = fcen / np.linalg.norm(fcen, axis=-1, keepdims=True)
        facecolors[(fnrm @ cam_dir) < -0.1, 3] = 0.0
        ax.plot_surface(xs, ys, zs, facecolors=facecolors, shade=False,
                        rstride=1, cstride=1, antialiased=False, zorder=0)
    else:
        sphere_alpha = 1.0 if clip_sphere else 0.55
        ax.plot_surface(xs, ys, zs, color="#a9603f", alpha=sphere_alpha,
                        linewidth=0, rstride=1, cstride=1, shade=True, zorder=0)

    # orbit path over the plotted segment
    ax.plot(r[seg, 0], r[seg, 1], r[seg, 2], color="#444444", lw=1.0, alpha=0.7)

    # reference arrows (Sun-line, orbit normal) from Mars centre — Fig A only.
    if show_ref_arrows:
        ax.quiver(0, 0, 0, *(ref_len * s_mid), color="#d4a017", lw=2.0,
                  arrow_length_ratio=0.12)
        ax.text(*(ref_len * s_mid * 1.05), "Sun (s)", color="#b8860b", fontsize=9)
        ax.quiver(0, 0, 0, *(ref_len * h_mid), color="#2a8c2a", lw=2.0,
                  arrow_length_ratio=0.12)
        ax.text(*(ref_len * h_mid * 1.05), "orbit normal (h)", color="#1f6f1f", fontsize=9)

    # sails at snapshots
    for k, (i, lab, col) in enumerate(zip(snap_idx, labels, snap_colors)):
        c = traj["r"][i]
        n_hat = traj["n"][i]
        roll_ref = traj["h"][i]
        corners, ear, n_hat = _sail_quad(c, n_hat, roll_ref, sail_size_km)
        ax.add_collection3d(Poly3DCollection([corners], facecolor=col,
                            edgecolor="black", alpha=0.9, linewidths=0.8))
        ax.add_collection3d(Poly3DCollection([ear], facecolor="white",
                            edgecolor="black", alpha=0.98, linewidths=0.6))
        ax.quiver(*c, *(normal_len * n_hat), color=col, lw=1.4, arrow_length_ratio=0.3)
        r_hat = c / np.linalg.norm(c)   # number radially OUTSIDE the sail
        ax.text(*(c + label_offset_km * r_hat), lab, color="black",
                fontsize=NUMBER_FONTSIZE, fontweight="bold", ha="center", va="center")

    # zoom box centred on view_center + equal aspect; camera; ditch frame.
    ax.set_box_aspect([1, 1, 1])
    for setlim, c0 in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), view_center):
        setlim(c0 - view_half_span, c0 + view_half_span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.text(*(1.03 * MARS_R_KM * cam_dir), "Mars", color="#ffe9c8",
            fontsize=MARS_LABEL_FONTSIZE, ha="center", va="center", style="italic")


# ===========================================================================
# PyVista render (smooth textured Mars) -> screenshot for the figure panel
# ===========================================================================

_PV_TEX_CACHE: dict = {}


def _pv_texture():
    import pyvista as pv
    if "tex" not in _PV_TEX_CACHE:
        p = Path(MARS_TEXTURE_PATH)
        _PV_TEX_CACHE["tex"] = pv.read_texture(str(p)) if p.exists() else None
    return _PV_TEX_CACHE["tex"]


def _body_xyz(lat_deg: float, lon_deg: float, radius: float = MARS_R_KM) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return radius * np.array([math.cos(la) * math.cos(lo),
                              math.cos(la) * math.sin(lo), math.sin(la)])


def _pv_polygon(pts: np.ndarray):
    import pyvista as pv
    n = len(pts)
    return pv.PolyData(np.asarray(pts, float), faces=np.hstack([[n], np.arange(n)]))


def _camera_basis(cam_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cam_dir, up, cam_right) for the standardised view: look down cam_axis,
    drop VIEW_TILT_DEG in elevation, then orbit VIEW_AZIMUTH_LEFT_DEG about up.
    Shared by the framing (make_figure) and the render so they stay consistent."""
    elev = math.degrees(math.asin(float(np.clip(cam_axis[2], -1, 1)))) - VIEW_TILT_DEG
    azim = math.degrees(math.atan2(cam_axis[1], cam_axis[0]))
    er, ar = math.radians(elev), math.radians(azim)
    cam_dir = np.array([math.cos(er) * math.cos(ar),
                        math.cos(er) * math.sin(ar), math.sin(er)])
    up = np.array([0.0, 0.0, 1.0]) - float(np.dot([0, 0, 1], cam_dir)) * cam_dir
    if np.linalg.norm(up) < 0.1:
        up = np.array([0.0, 1.0, 0.0]) - float(np.dot([0, 1, 0], cam_dir)) * cam_dir
    up /= np.linalg.norm(up)
    a = math.radians(VIEW_AZIMUTH_LEFT_DEG)   # Rodrigues orbit-left about up
    cam_dir = (cam_dir * math.cos(a) + np.cross(up, cam_dir) * math.sin(a)
               + up * float(np.dot(up, cam_dir)) * (1 - math.cos(a)))
    cam_dir /= np.linalg.norm(cam_dir)
    cam_right = np.cross(up, cam_dir)
    cam_right /= np.linalg.norm(cam_right)
    return cam_dir, up, cam_right


def _seamless_mars_mesh(Rmat: np.ndarray, res: int):
    """Textured Mars sphere with the UV seam OPEN at the anti-meridian (no face
    spans the lon 1<->0 wrap), so the texture seam is invisible at any rotation.
    Built in IAU_MARS body frame (u=lon/2pi+0.5, v=1-colat/pi) then rotated to J2000."""
    import pyvista as pv
    n, m = int(res), max(2, int(res) // 2)
    phi = np.linspace(-np.pi, np.pi, n)   # longitude; open seam at +/-pi (anti-meridian)
    th = np.linspace(0.0, np.pi, m)       # colatitude
    PHI, TH = np.meshgrid(phi, th, indexing="ij")            # (n, m)
    X = MARS_R_KM * np.sin(TH) * np.cos(PHI)
    Y = MARS_R_KM * np.sin(TH) * np.sin(PHI)
    Z = MARS_R_KM * np.cos(TH)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    u = (PHI / (2 * np.pi) + 0.5).ravel()
    v = (1.0 - TH / np.pi).ravel()
    ii, jj = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    a = (ii * m + jj).ravel()
    b = ((ii + 1) * m + jj).ravel()
    c = ((ii + 1) * m + jj + 1).ravel()
    d = (ii * m + jj + 1).ravel()
    faces = np.column_stack([np.full(a.size, 4, np.int64), a, b, c, d]).ravel()
    poly = pv.PolyData(pts, faces=faces)
    poly.active_texture_coordinates = np.column_stack([u, v]).astype(np.float32)
    poly.points = (Rmat @ poly.points.T).T
    poly.compute_normals(inplace=True, auto_orient_normals=True)
    return poly


def _add_sun_gnomon_3d(pl, s_mid: np.ndarray, view_center: np.ndarray,
                       view_half_span: float, cam_right: np.ndarray,
                       up: np.ndarray, cam_dir: np.ndarray) -> None:
    """A small TRUE-3D Sun-direction arrow anchored in a screen corner.

    The arrow points along the world Sun-line s_mid, so it foreshortens
    correctly: when the Sun is near the view axis it shows short, when in-plane
    it shows full length, unlike a flat two-dimensional glyph.
    Anchored at a fixed (right, up) screen fraction and pushed toward the camera
    (+cam_dir) so Mars never eats it; parallel projection keeps its size fixed."""
    import pyvista as pv
    hs = float(view_half_span)
    fr, fu = SUN_GNOMON_CORNER
    base = (view_center + cam_right * (fr * hs) + up * (fu * hs)
            + cam_dir * (2.0 * MARS_R_KM))   # toward camera -> in front of Mars
    L = SUN_GNOMON_FRAC * hs
    pl.add_mesh(pv.Arrow(start=tuple(base), direction=tuple(s_mid), scale=L,
                         tip_length=0.26, tip_radius=0.085, shaft_radius=0.030),
                color="#d4a017", ambient=0.9, diffuse=0.2)
    pl.add_point_labels([base + s_mid * (L * 1.18)], ["Sun"], font_size=PYVISTA_GNOMON_FONT,
                        text_color="#d4a017", font_family="times", bold=True,
                        shape=None, show_points=False, always_visible=True)


def _draw_corner_indicator(ax, traj: dict, seg: slice,
                           cam_right: np.ndarray, cam_up: np.ndarray) -> None:
    """Small 2D Sun / orbit-normal direction indicator in the panel corner.
    Each world vector is projected to the screen basis (cam_right, cam_up); the
    2D arrow length is its in-plane magnitude (so a vector pointing into/out of
    the screen, e.g. the orbit normal, correctly shows as a short stub)."""
    mid = (seg.start + (seg.stop or len(traj["r"]))) // 2
    ox, oy, scale = 0.13, 0.15, 0.12
    for vec, name, col in ((traj["s"][mid], "Sun", "#d4a017"),
                           (traj["h"][mid], "orbit normal", "#2a8c2a")):
        sx = float(np.dot(vec, cam_right))
        sy = float(np.dot(vec, cam_up))
        ax.annotate("", xy=(ox + sx * scale, oy + sy * scale), xytext=(ox, oy),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        mag = math.hypot(sx, sy)
        if mag > 0.05:
            ax.text(ox + sx * scale * 1.25, oy + sy * scale * 1.25, name, color=col,
                    fontsize=9, transform=ax.transAxes, ha="center", va="center")


def render_mars_pyvista(traj: dict, seg: slice, snap_idx: list[int],
                        snap_colors: list[str], labels: list[str],
                        sail_size_km: float, view_center: np.ndarray,
                        view_half_span: float, et_ref: float,
                        label_offset_km: float, show_target: bool,
                        cam_axis: np.ndarray, render_up_shift_frac: float = 0.0,
                        show_target_label: bool = True) -> np.ndarray:
    """Render the right-panel scene (textured Mars + sails + arrows + numbers +
    target marker) in PyVista off-screen; return the screenshot (H,W,3)."""
    import pyvista as pv
    normal_len = NORMAL_ARROW_FACTOR * sail_size_km
    r = traj["r"]
    mid = (seg.start + (seg.stop or len(r))) // 2
    h_mid, s_mid = traj["h"][mid], traj["s"][mid]
    Rmat = _mars_rotation_to_j2000(et_ref)

    # --- camera basis FIRST (shared helper): needed up-front to anchor the Sun
    # gnomon IN FRONT of Mars and place the target label.
    cam_dir, up, cam_right = _camera_basis(cam_axis)

    # own the lighting so it matches the Sun-line (physical day/night terminator)
    pl = pv.Plotter(off_screen=True, window_size=list(PYVISTA_WINDOW_PX),
                    lighting="none")
    pl.set_background(RENDER_BG)
    sun_light = pv.Light(position=tuple(s_mid * 1.0e6), focal_point=(0.0, 0.0, 0.0),
                         color="white")
    sun_light.positional = False   # directional: parallel rays from the Sun-line
    sun_light.intensity = 1.15
    pl.add_light(sun_light)

    # Mars: seamless textured sphere, oriented IAU_MARS -> J2000 at et_ref.
    mars = _seamless_mars_mesh(Rmat, PYVISTA_SPHERE_RES)
    tex = _pv_texture()
    if tex is not None:
        pl.add_mesh(mars, texture=tex, smooth_shading=True,
                    ambient=0.28, diffuse=0.95, specular=0.0)
    else:
        pl.add_mesh(mars, color="#a9603f", smooth_shading=True, ambient=0.28, diffuse=0.95)

    # orbit path
    pl.add_mesh(pv.lines_from_points(r[seg]), color="#333333", line_width=2)

    # 3D reference arrows, anchored IN FRONT of Mars (toward camera) at the frame
    # centre so the planet never eats them; they project from the view centre.
    if SHOW_REF_ARROWS_3D:
        anchor = view_center + cam_dir * (1.2 * MARS_R_KM)
        L = ARROW_LEN_FRAC * float(view_half_span)
        for vec, name, acol in ((s_mid, "Sun", "#d4a017"), (h_mid, "orbit normal", "#2a8c2a")):
            pl.add_mesh(pv.Arrow(start=tuple(anchor), direction=tuple(vec), scale=L,
                                 tip_length=0.16, tip_radius=0.032, shaft_radius=0.011),
                        color=acol, ambient=0.85, diffuse=0.25)
            pl.add_point_labels([anchor + vec * L * 1.06], [name], font_size=18,
                                text_color=acol, font_family="times", shape=None,
                                show_points=False, always_visible=True)

    # 3D Sun-direction gnomon in a screen corner (replaces the central arrows)
    if SHOW_SUN_GNOMON_3D:
        _add_sun_gnomon_3d(pl, s_mid, view_center, view_half_span,
                           cam_right, up, cam_dir)

    # target marker (delivery aim point) + label JUST BELOW the blob. The coordinate
    # text label is suppressed when show_target_label is False (Fig B keeps the marker).
    if show_target:
        tgt = Rmat @ _body_xyz(TARGET_LAT_DEG, TARGET_LON_DEG)
        br = 0.28 * sail_size_km
        pl.add_mesh(pv.Sphere(radius=br, center=tgt), color="#19e0ff",
                    ambient=0.85, diffuse=0.2)
        if show_target_label:
            lbl = tgt - up * (br + 0.55 * sail_size_km) + cam_dir * (0.3 * sail_size_km)
            pl.add_point_labels(
                [lbl], [f"({TARGET_LAT_DEG:.0f}°N, {TARGET_LON_DEG:.0f}°E)"],
                font_size=PYVISTA_TARGET_FONT, text_color="#19e0ff", font_family="times",
                bold=True, shape=None, show_points=False, always_visible=True)

    # sails (notched pentagon + white dog-ear), normal arrows, radial numbers
    lab_pts, lab_txt = [], []
    for i, lab, col in zip(snap_idx, labels, snap_colors):
        c, n_hat, roll = r[i], traj["n"][i], traj["h"][i]
        pent, ear, n_hat = _sail_quad(c, n_hat, roll, sail_size_km)
        pl.add_mesh(_pv_polygon(pent), color=col, show_edges=True,
                    edge_color="black", line_width=1.5, ambient=0.65, diffuse=0.5)
        pl.add_mesh(_pv_polygon(ear), color="white", show_edges=True,
                    edge_color="black", line_width=1.0, ambient=0.85, diffuse=0.3)
        pl.add_mesh(pv.Arrow(start=c, direction=n_hat, scale=normal_len),
                    color=col, ambient=0.6, diffuse=0.45)
        lab_pts.append(c + label_offset_km * c / np.linalg.norm(c))   # radially out
        lab_txt.append(lab)
    pl.add_point_labels(lab_pts, lab_txt, font_size=PYVISTA_NUMBER_FONT, text_color="black",
                        font_family="times", bold=True, shape=None, show_points=False,
                        always_visible=True)

    # render_up_shift_frac pans the camera DOWN (-up) so the scene rides UP in the
    # frame: uses the top margin and pulls the lower sails off the bottom edge.
    pl.enable_parallel_projection()
    fp = view_center - up * (render_up_shift_frac * float(view_half_span))
    pl.camera.focal_point = tuple(fp)
    pl.camera.position = tuple(fp + cam_dir * 12.0 * MARS_R_KM)
    pl.camera.up = tuple(up)
    pl.camera.parallel_scale = float(view_half_span) * 1.04

    img = pl.screenshot(return_img=True)
    pl.close()
    return img, cam_right, up


# ===========================================================================
# Figure assembly
# ===========================================================================

def make_figure(traj: dict, seg: slice, snap_idx: list[int],
                title: str, out_png: Path, shade_roles: bool,
                sail_size_km: float, tight_on_snapshots: bool,
                cam_axis: np.ndarray, number_y_deg: float,
                xlim: tuple | None = None, endpoint_ticks: bool = True,
                legend_loc: str | None = "upper right",
                fig_height: float = 7.2, fig_top: float = 0.96,
                fig_bottom: float = 0.10, render_zoom: float = 1.0,
                render_up_shift_frac: float = 0.0,
                show_target_label: bool = True,
                width_ratios: tuple[float, float] = (1.05, 1.0),
                wspace: float = 0.18,
                number_above_frame: bool = False,
                ms_cone: float = MARKER_MS_CONE,
                ms_clock: float = MARKER_MS_CLOCK) -> None:
    cone_all, clock_raw, _, _ = mcinnes_cone_clock(
        traj["n"], traj["s"], traj["h"], sign=CLOCK_SIGN)
    clock_all = np.full_like(clock_raw, np.nan)
    clock_all[seg] = clock_for_segment(clock_raw[seg])

    t = traj["t_min"]
    roles = traj["schedule_role"]
    labels = [str(k + 1) for k in range(len(snap_idx))]
    snap_colors = [ROLE_COLORS.get(roles[i], "#555555") for i in snap_idx]

    fig = plt.figure(figsize=(17.0, fig_height))
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), wspace=wspace,
                          left=0.06, right=0.97, top=fig_top, bottom=fig_bottom)
    ax_o = fig.add_subplot(gs[0, 0])

    # PCHIP-interpolate the plotted curves onto a fine time grid.  The clock is
    # unwrapped BEFORE interpolation (you can't interpolate across a 360->0 jump),
    # then re-wrapped on the fine grid -> the wrap connector spans one fine cell
    # (~sub-pixel) and reads as a truly vertical line, not a slanted segment over
    # one coarse sample.  Markers below stay on the real data points.
    t_seg = t[seg]
    tf = np.linspace(t_seg[0], t_seg[-1], FINE_GRID_N)
    cone_f = PchipInterpolator(t_seg, cone_all[seg])(tf)
    clock_unw = np.degrees(np.unwrap(np.radians(clock_raw[seg])))
    clock_f = PchipInterpolator(t_seg, clock_unw)(tf)
    if not UNWRAP_CLOCK:
        clock_f = clock_f % 360.0

    # ---- orientation panel: cone (left) + clock (right) ----
    color_cone = "#0b5394"
    color_clock = "#cc4125"
    ax_o.plot(tf, cone_f, color=color_cone, lw=1.6, label=r"cone $\alpha$")
    ax_o.set_ylabel(r"cone angle $\alpha$ (degrees)", color=color_cone, fontsize=AXIS_LABEL_FONTSIZE)
    ax_o.tick_params(axis="y", labelcolor=color_cone, labelsize=TICK_LABEL_FONTSIZE)
    ax_o.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax_o.set_ylim(0, 90)
    ax_o.set_xlabel(f"time from sol-{SOL} start (min)", fontsize=AXIS_LABEL_FONTSIZE)

    ax_c = ax_o.twinx()
    ax_c.plot(tf, clock_f, color=color_clock, lw=1.6, label=r"clock $\delta$")
    ax_c.set_ylabel(r"clock angle $\delta$ (degrees)",
                    color=color_clock, fontsize=AXIS_LABEL_FONTSIZE)
    ax_c.tick_params(axis="y", labelcolor=color_clock, labelsize=TICK_LABEL_FONTSIZE)
    if not UNWRAP_CLOCK:
        # clock is a proper azimuth: pin the axis to one full turn so it can't
        # exceed [0,360); the line wraps (vertical jump) at the 360->0 crossing.
        ax_c.set_ylim(0, 360)
        ax_c.set_yticks(CLOCK_YTICKS)

    # Role shading (Figure B); also collect each non-cruise span for the role bar.
    # continuously colour-coded and each band matches the colour of the OSR sketches for
    # that phase in the render -- cruise reads as ROLE_COLORS["cruise"] blue.  Each point
    # is covered exactly once (no compositing of alphas, which would have darkened the
    # already-approved slew/illuminate bands), and the FIRST and LAST run stretch out to
    # the axis limits so the xlim padding leaves no unshaded sliver at either end.
    role_spans = []   # (role, t_left, t_right) for the <-- label --> arrows above
    if shade_roles:
        seg_stop = seg.stop or len(t)
        runs = []     # (role, i_first, i_last) contiguous same-role runs
        i = seg.start
        while i < seg_stop:
            j = i
            while j + 1 < seg_stop and roles[j + 1] == roles[i]:
                j += 1
            runs.append((roles[i], i, j))
            i = j + 1
        for k, (role, i0, i1) in enumerate(runs):
            # true role extent -- what the role-bar arrow must span
            t_left = float(t[i0])
            t_right = float(t[min(i1 + 1, seg_stop - 1)])
            # shaded extent -- stretched at the two ends of the panel only
            shade_left, shade_right = t_left, t_right
            if xlim is not None:
                if k == 0:
                    shade_left = min(t_left, float(xlim[0]))
                if k == len(runs) - 1:
                    shade_right = max(t_right, float(xlim[1]))
            ax_o.axvspan(shade_left, shade_right,
                         color=ROLE_COLORS.get(role, "#999999"),
                         alpha=ROLE_SHADE_ALPHA, lw=0)
            if role != "cruise":
                role_spans.append((role, t_left, t_right))

    # one vertical line per snapshot (1 line : 2 points -- cone + clock).  The number
    # either sits at a common height (number_y_deg) inside the panel, or -- when
    # the blended transform (x in data/min, y in axes fraction) so it tracks the
    # snapshot line at any y-limit.  The same technique as Fig B's role bar; Fig B keeps
    # its numbers inside the panel because that strip is taken by the role bar.
    for k, i in enumerate(snap_idx):
        ax_o.axvline(t[i], color="grey", ls=":", lw=0.9, alpha=0.55, zorder=1)
        ax_o.plot(t[i], cone_all[i], "o", color=snap_colors[k], ms=ms_cone,
                  mec="black", mew=0.6, zorder=5)
        ax_c.plot(t[i], clock_all[i], "s", color=snap_colors[k], ms=ms_clock,
                  mec="black", mew=0.6, zorder=5)
        if number_above_frame:
            ax_o.text(t[i], NUMBER_ABOVE_FRAME_Y, labels[k],
                      transform=ax_o.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=NUMBER_FONTSIZE,
                      fontweight="bold", clip_on=False, zorder=6)
        else:
            ax_o.text(t[i], number_y_deg, labels[k], ha="center", va="center",
                      fontsize=NUMBER_FONTSIZE, fontweight="bold", zorder=6)

    # no vertical gridlines (the only vertical lines are the snapshot ones)
    ax_o.xaxis.grid(False)
    ax_o.yaxis.grid(True, ls=":", alpha=0.25)

    # x-limits: explicit when given, else span the plotted segment.
    if xlim is not None:
        t0, tN = float(xlim[0]), float(xlim[1])
    else:
        t0 = float(np.floor(t[seg.start]))
        tN = float(np.ceil(t[(seg.stop or len(t)) - 1]))
    ax_o.set_xlim(t0, tN)
    if endpoint_ticks:
        # pin a tick at each limit (Fig B: shows the exact sampled-window bounds)
        inter = [tk for tk in ax_o.get_xticks() if t0 < tk < tN]
        ax_o.set_xticks([t0] + inter + [tN])
    else:
        # Fig A: auto ticks within range (leftmost reads 0 given the small left
        # margin); the axis still extends to tN (full-orbit closure).
        ax_o.set_xticks([tk for tk in ax_o.get_xticks() if t0 <= tk <= tN])

    # ---- legend (lines + role colours); loc per caller, or None to omit (Fig B) ----
    if legend_loc is not None:
        handles = [Line2D([0], [0], color=color_cone, lw=1.6, label=r"cone $\alpha$"),
                   Line2D([0], [0], color=color_clock, lw=1.6, label=r"clock $\delta$")]
        if shade_roles:
            roles_present = [r for r in ("cruise", "slew_in", "bisector", "slew_out")
                             if r in set(traj["schedule_role"][snap_idx])]
            handles += [Patch(facecolor=ROLE_COLORS[r], edgecolor="black",
                              alpha=0.85, label=ROLE_DISPLAY[r]) for r in roles_present]
        ax_o.legend(handles=handles, loc=legend_loc, fontsize=LEGEND_FONTSIZE,
                    ncol=2, framealpha=0.92)

    # ---- role-phase bar above the panel (Fig B): a <-- label --> double-headed
    # arrow spanning each non-cruise phase, slew_in/slew_out both reading "slew".
    # Blended transform (x in data/min, y in axes fraction) places it just above the
    # top edge; the centred white-backed label visually breaks the arrow line.
    # role_spans is empty when shade_roles is False (Fig A), so this is a no-op there.
    for role, t_left, t_right in role_spans:
        ax_o.annotate("", xy=(t_right, ROLE_BAR_Y), xytext=(t_left, ROLE_BAR_Y),
                      xycoords=ax_o.get_xaxis_transform(),
                      textcoords=ax_o.get_xaxis_transform(),
                      arrowprops=dict(arrowstyle="<->", color="black", lw=1.6),
                      annotation_clip=False)
        ax_o.text(0.5 * (t_left + t_right), ROLE_BAR_Y, ROLE_BAR_LABELS.get(role, role),
                  transform=ax_o.get_xaxis_transform(), ha="center", va="center",
                  fontsize=ROLE_BAR_FONTSIZE, fontweight="bold",
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
                  clip_on=False, zorder=10)

    # ---- render panel: centre + zoom ----
    label_off = LABEL_RADIAL_OFFSET_FACTOR * sail_size_km
    if tight_on_snapshots:
        # Frame in the ACTUAL screen basis (cam_right, up), not world axes, and
        # pad by each sail's drawn reach (normal arrow 1.4*size, half-diagonal,
        # radial number) so no edge pose clips. view_half_span is guaranteed >=
        # core + reach; FIGB_ZOOM only tightens when that still keeps all visible.
        _, up_b, right_b = _camera_basis(cam_axis)
        pts = traj["r"][snap_idx]
        view_center = pts.mean(axis=0)
        off = pts - view_center
        core = float(max(np.abs(off @ right_b).max(), np.abs(off @ up_b).max()))
        reach = max(label_off, NORMAL_ARROW_FACTOR * sail_size_km) + 0.5 * sail_size_km
        view_half_span = max((core + FIGB_VIEW_MARGIN_KM) * FIGB_ZOOM, core + reach)
    else:
        # Fig A: centre on Mars; frame in the SAME screen basis as Fig B so the
        # full ring, its screen-radial numbers, AND the Mars disk all fit (the
        # A world-axis component maximum can clip top/bottom labels under tilt.
        _, up_a, right_a = _camera_basis(cam_axis)
        view_center = np.zeros(3)
        pts = traj["r"][snap_idx]
        core = float(max(np.abs(pts @ right_a).max(), np.abs(pts @ up_a).max()))
        reach = max(label_off, NORMAL_ARROW_FACTOR * sail_size_km) + 0.5 * sail_size_km
        half = max(core + reach, MARS_R_KM * 1.02)   # never clip the planet
        view_half_span = (half + FIGA_VIEW_MARGIN_KM) * FIGA_ZOOM
    view_half_span /= render_zoom   # >1 shrinks the frame -> render scene appears larger
    print(f"  render framing: core={core:.0f} reach={reach:.0f} "
          f"view_half_span={view_half_span:.0f} km (render_zoom={render_zoom:.2f})")
    # orient Mars at the illumination epoch: bisector midpoint if present (Fig B),
    # else the segment midpoint (Fig A).
    seg_idx = np.arange(seg.start, seg.stop or len(t))
    bis = [i for i in seg_idx if traj["schedule_role"][i] == "bisector"]
    et_ref = float(traj["et_tdb"][bis[len(bis) // 2]] if bis
                   else traj["et_tdb"][seg_idx[len(seg_idx) // 2]])
    if USE_PYVISTA:
        img, cam_right, cam_up = render_mars_pyvista(
            traj, seg, snap_idx, snap_colors, labels, sail_size_km, view_center,
            view_half_span, et_ref, label_offset_km=label_off,
            show_target=SHOW_TARGET_MARKER, cam_axis=cam_axis,
            render_up_shift_frac=render_up_shift_frac,
            show_target_label=show_target_label)
        ax_r = fig.add_subplot(gs[0, 1])
        ax_r.imshow(img)
        ax_r.set_axis_off()
        if SHOW_CORNER_INDICATOR:
            _draw_corner_indicator(ax_r, traj, seg, cam_right, cam_up)
    else:
        ax_r = fig.add_subplot(gs[0, 1], projection="3d")
        render_matplotlib(ax_r, traj, seg, snap_idx, labels, snap_colors,
                           sail_size_km, view_center, view_half_span,
                           clip_sphere=tight_on_snapshots, label_offset_km=label_off,
                           show_ref_arrows=not tight_on_snapshots, et_ref=et_ref)

    fig.savefig(out_png, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_sol(SOL)

    # Cross-check the derived McInnes cone against the stored alpha_cone_deg.
    cone, _, _, _ = mcinnes_cone_clock(traj["n"], traj["s"], traj["h"], sign=CLOCK_SIGN)
    dcone = np.abs(cone - traj["alpha_cone_deg"])
    print(f"cone cross-check vs CSV alpha_cone_deg: max |diff| = {dcone.max():.3e} deg")

    # standardised camera axis (mean orbit normal / Sun over the sol) -> both
    # figures share the viewing angle; only the zoom differs.
    cam_axis = _unit(np.asarray({"orbit_normal": traj["h"], "sun": traj["s"]}
                                .get(VIEW_ALONG, traj["h"])).mean(axis=0))

    # ---- Figure A: typical cruise orbit ----
    a0, a1 = orbit_bounds(traj["u_transport_deg"], FIGA_ORBIT_INDEX)
    segA = slice(a0, a1 + 1)
    rolesA = set(traj["schedule_role"][segA])
    # 9 equal points around the orbit, plot the first 8 (drops the 9th, which
    # coincides with the 1st -> no wrap-around overlap of snapshots 1 and 8).
    snapA = list(np.linspace(a0, a1, FIGA_N_SNAPSHOTS + 1)[:FIGA_N_SNAPSHOTS].round().astype(int))
    # x-axis: a little room left of t=0 (leftmost tick still 0), right edge at the
    # FULL orbit-1 data end so the cone/clock curves run to orbital closure.
    figa_xlim = (-FIGA_LEFT_MARGIN_MIN, float(traj["t_min"][a1]))
    print(f"\nFigure A: orbit {FIGA_ORBIT_INDEX}, idx {a0}..{a1}, "
          f"t {traj['t_min'][a0]:.1f}..{traj['t_min'][a1]:.1f} min, roles={rolesA}, "
          f"xlim {figa_xlim[0]:.1f}..{figa_xlim[1]:.1f}")
    make_figure(traj, segA, snapA,
                f"Figure A — typical station-keeping orbit (sol {SOL}, orbit {FIGA_ORBIT_INDEX}, pure cruise)",
                FIGA_OUT, shade_roles=False, sail_size_km=FIGA_SAIL_SIZE_KM,
                tight_on_snapshots=False, cam_axis=cam_axis,
                number_y_deg=NUMBER_Y_DEG_A, xlim=figa_xlim, endpoint_ticks=False,
                legend_loc="lower right",
                fig_height=FIGA_FIG_HEIGHT, fig_top=FIGA_FIG_TOP,
                fig_bottom=FIGA_FIG_BOTTOM, render_zoom=FIGA_RENDER_ZOOM,
                width_ratios=FIGA_WIDTH_RATIOS, wspace=FIGA_WSPACE,
                number_above_frame=FIGA_NUMBER_ABOVE_FRAME)

    # ---- Figure B: delivery event ----
    t = traj["t_min"]
    b0 = int(np.searchsorted(t, FIGB_T_START_MIN))
    b1 = int(np.searchsorted(t, FIGB_T_END_MIN))
    segB = slice(b0, b1 + 1)
    figb_times = np.linspace(FIGB_T_START_MIN, FIGB_T_END_MIN, FIGB_N_SNAPSHOTS)
    snapB = [int(np.argmin(np.abs(t - tm))) for tm in figb_times]
    rolesB = [traj["schedule_role"][i] for i in snapB]
    # x-limits follow the actual sampled points: first - pad, last + pad.
    samp_t = t[snapB]
    figb_xlim = (float(samp_t.min()) - XLIM_PAD_MIN, float(samp_t.max()) + XLIM_PAD_MIN)
    print(f"\nFigure B: t {FIGB_T_START_MIN}..{FIGB_T_END_MIN} min, idx {b0}..{b1}, "
          f"xlim {figb_xlim[0]:.1f}..{figb_xlim[1]:.1f}")
    print(f"  snapshots (min -> role): "
          + ", ".join(f"{tm:.0f}->{rl}" for tm, rl in zip(figb_times, rolesB)))
    make_figure(traj, segB, snapB,
                f"Figure B — delivery event (sol {SOL}, cruise -> slew in -> bisector -> slew out -> cruise)",
                FIGB_OUT, shade_roles=True, sail_size_km=FIGB_SAIL_SIZE_KM,
                tight_on_snapshots=True, cam_axis=cam_axis,
                number_y_deg=NUMBER_Y_DEG_B, xlim=figb_xlim,
                legend_loc=None,   # no legend; <-- label --> role bar above the panel instead
                fig_height=FIGB_FIG_HEIGHT, fig_top=FIGB_FIG_TOP,
                fig_bottom=FIGB_FIG_BOTTOM, render_zoom=FIGB_RENDER_ZOOM,
                render_up_shift_frac=FIGB_RENDER_UP_SHIFT_FRAC,
                show_target_label=False,   # drop the (40N,200E) text label in Fig B (keep the marker)
                number_above_frame=FIGB_NUMBER_ABOVE_FRAME,
                ms_cone=FIGB_MARKER_MS_CONE, ms_clock=FIGB_MARKER_MS_CLOCK)


if __name__ == "__main__":
    main()
