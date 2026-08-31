"""Render the sail reference-frame diagram used as manuscript Figure 2.

Orientation coordinates follow McInnes (1999), Eq. 4.7, p. 116:
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
from matplotlib import patheffects as pe
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
FIGA_OUT = FIG_DIR / "figA_typical_orbit.png"

# --- Figure B: delivery event ----------------------------------------------
# Delivery event 1 of sol-1 spans t ~ 369-403 min (slew_in/bisector/slew_out).
FIGB_T_START_MIN = 365.0
FIGB_T_END_MIN = 407.0
FIGB_N_SNAPSHOTS = 12    # poses sampled EVENLY in time across [START, END]
FIGB_OUT = FIG_DIR / "figB_delivery.png"

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
SAVE_DPI = 400               # matplotlib savefig DPI for the composed figure (v2: raised 300->400)
# The 3D panel is a raster screenshot, so its crispness is set by the PyVista
# window px, NOT by SAVE_DPI.  Supersample renders it larger; the window px AND
# every in-render label font scale by the SAME factor, so appearance is unchanged
# -- only the pixel count (and thus sharpness in the high-DPI composite) grows.
PYVISTA_SUPERSAMPLE = 2.0     # v2: raised 1.6->2.0 (crisper renders for the higher-DPI composite)
_PV_BASE_PX = 1500
PYVISTA_WINDOW_PX = (int(_PV_BASE_PX * PYVISTA_SUPERSAMPLE), int(_PV_BASE_PX * PYVISTA_SUPERSAMPLE))
PYVISTA_SPHERE_RES = 360     # PyVista sphere theta/phi resolution (GPU-interpolated -> smooth)
PYVISTA_NUMBER_FONT = int(34 * PYVISTA_SUPERSAMPLE)  # snapshot-number font in the render (Times)
PYVISTA_TARGET_FONT = int(26 * PYVISTA_SUPERSAMPLE)  # target-label font in the render (Times)
PYVISTA_GNOMON_FONT = int(24 * PYVISTA_SUPERSAMPLE)  # Sun-gnomon label font in the render (Times)
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
SHOW_SUN_GNOMON_3D = True
SUN_GNOMON_FRAC = 0.20       # gnomon arrow length = this * view_half_span
SUN_GNOMON_CORNER = (-0.78, -0.70)  # (right, up) offset of the gnomon base, in units of view_half_span
FIGA_VIEW_MARGIN_KM = 120.0  # Fig A: frame = (ring+label) extent + this margin, centred on Mars
FIGA_ZOOM = 0.98             # Fig A: shrink the frame by this factor (smaller = more zoomed-in)
FIGB_VIEW_MARGIN_KM = 150.0  # Fig B: frame = (snapshot+label) extent + this margin, centred on the arc
FIGB_ZOOM = 0.93             # Fig B: shrink the frame by this factor (smaller = more zoomed-in crop)
LABEL_RADIAL_OFFSET_FACTOR = 1.0  # snapshot number sits this * sail_size radially OUTSIDE each sail
SPHERE_CAP_DEG = 92.0        # Fig B: draw Mars as a curved cap this far from the arc direction (no flat cut)

# --- left-panel styling ---------------------------------------------------
LEGEND_FONTSIZE = 12
NUMBER_FONTSIZE = 14         # snapshot numbers
MARS_LABEL_FONTSIZE = 16
NUMBER_Y_DEG_A = 15.0        # Fig A: snapshot numbers sit at this height on the cone (left) axis
NUMBER_Y_DEG_B = 65.0        # Fig B: higher, to clear the deep cone dip during the delivery arc

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
                        cam_axis: np.ndarray) -> np.ndarray:
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

    # target marker (delivery aim point) + label JUST BELOW the blob
    if show_target:
        tgt = Rmat @ _body_xyz(TARGET_LAT_DEG, TARGET_LON_DEG)
        br = 0.28 * sail_size_km
        pl.add_mesh(pv.Sphere(radius=br, center=tgt), color="#19e0ff",
                    ambient=0.85, diffuse=0.2)
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

    pl.enable_parallel_projection()
    pl.camera.focal_point = tuple(view_center)
    pl.camera.position = tuple(view_center + cam_dir * 12.0 * MARS_R_KM)
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
                legend_loc: str = "upper right") -> None:
    cone_all, clock_raw, _, _ = mcinnes_cone_clock(
        traj["n"], traj["s"], traj["h"], sign=CLOCK_SIGN)
    clock_all = np.full_like(clock_raw, np.nan)
    clock_all[seg] = clock_for_segment(clock_raw[seg])

    t = traj["t_min"]
    roles = traj["schedule_role"]
    labels = [str(k + 1) for k in range(len(snap_idx))]
    snap_colors = [ROLE_COLORS.get(roles[i], "#555555") for i in snap_idx]

    fig = plt.figure(figsize=(17.0, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18,
                          left=0.06, right=0.97, top=0.96, bottom=0.1)
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
    ax_o.set_ylabel(r"cone angle $\alpha$ (deg)", color=color_cone, fontsize=12)
    ax_o.tick_params(axis="y", labelcolor=color_cone)
    ax_o.set_ylim(0, 90)
    ax_o.set_xlabel(f"time from sol-{SOL} start (min)", fontsize=12)

    ax_c = ax_o.twinx()
    ax_c.plot(tf, clock_f, color=color_clock, lw=1.6, label=r"clock $\delta$")
    ax_c.set_ylabel(r"clock angle $\delta$ (deg, orbit-normal ref)",
                    color=color_clock, fontsize=12)
    ax_c.tick_params(axis="y", labelcolor=color_clock)
    if not UNWRAP_CLOCK:
        # clock is a proper azimuth: pin the axis to one full turn so it can't
        # exceed [0,360); the line wraps (vertical jump) at the 360->0 crossing.
        ax_c.set_ylim(0, 360)
        ax_c.set_yticks(CLOCK_YTICKS)

    # role shading (Figure B)
    if shade_roles:
        seg_start = seg.start
        seg_stop = seg.stop or len(t)
        i = seg_start
        while i < seg_stop:
            j = i
            while j + 1 < seg_stop and roles[j + 1] == roles[i]:
                j += 1
            if roles[i] != "cruise":
                ax_o.axvspan(t[i], t[min(j + 1, seg_stop - 1)],
                             color=ROLE_COLORS.get(roles[i], "#999999"),
                             alpha=0.10, lw=0)
            i = j + 1

    # one vertical line per snapshot (1 line : 2 points -- cone + clock); the
    # number sits at a common height (number_y_deg) on the line.
    for k, i in enumerate(snap_idx):
        ax_o.axvline(t[i], color="grey", ls=":", lw=0.9, alpha=0.55, zorder=1)
        ax_o.plot(t[i], cone_all[i], "o", color=snap_colors[k], ms=7,
                  mec="black", mew=0.6, zorder=5)
        ax_c.plot(t[i], clock_all[i], "s", color=snap_colors[k], ms=6,
                  mec="black", mew=0.6, zorder=5)
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

    # ---- legend (lines + role colours), top-right ----
    handles = [Line2D([0], [0], color=color_cone, lw=1.6, label=r"cone $\alpha$"),
               Line2D([0], [0], color=color_clock, lw=1.6, label=r"clock $\delta$")]
    if shade_roles:
        roles_present = [r for r in ("cruise", "slew_in", "bisector", "slew_out")
                         if r in set(traj["schedule_role"][snap_idx])]
        handles += [Patch(facecolor=ROLE_COLORS[r], edgecolor="black",
                          alpha=0.85, label=ROLE_DISPLAY[r]) for r in roles_present]
    ax_o.legend(handles=handles, loc=legend_loc, fontsize=LEGEND_FONTSIZE,
                ncol=2, framealpha=0.92)

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
            show_target=SHOW_TARGET_MARKER, cam_axis=cam_axis)
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
                number_y_deg=NUMBER_Y_DEG_A, xlim=figa_xlim, endpoint_ticks=False)

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
                legend_loc="upper center")


# ===========================================================================
# ===========================================================================
# Single annotated 3D scene built on the same textured-Mars / sail machinery:
#   - Mars (IAU_MARS, real texture) + spin axis (north pole)
#   - MME2000 inertial triad (where the orbit elements a,e,i,Omega,omega,nu live)
#   - one orbit + the sail (square + dog-ear roll marker) at a representative pose
#   - the sail's defining vectors n_hat (normal), s_hat (Sun-line), h_hat (orbit
#     normal), e_A (clock reference = h projected perp to s)
#   - the McInnes 1999 Eq.4.7 angles: cone alpha = angle(n_hat, s_hat); clock
#     delta = azimuth of n_hat about s_hat, measured from e_A (the orbit-normal
#     projection)
#   - the ground delivery target
ORI_ALT_KM = 500.0            # sail altitude (km) -- ~to scale now (real K=12 ~508 km);
                              # the vectors point OUTWARD (away from Mars) so they clear
ORI_VEC_LEN_KM = 2300.0       # length of the n/s/e_A/e_B vectors drawn from the sail
ORI_ARC_RADIUS_KM = 1500.0    # radius of the cone-angle arc (clock arc = 0.72x)
ORI_SAIL_SIZE_KM = 950.0      # exaggerated sail (real ~tens of m)
ORI_TRIAD_LEN_KM = 4900.0     # X/Y/Z arrows: > Mars R (3396) so they emerge from the disk
ORI_TARGET_RADIUS_KM = 90.0   # ground-target dot radius (small)
ORI_VIEW_AZIM_DEG = 0.0       # rotate camera about the pole (0 = orbit normal dead at viewer)
ORI_VIEW_CENTER_FRAC = 0.52   # view centre = sail_pos * frac (frames Mars + the right-side sail)
ORI_VIEW_HALF_SPAN_KM = 7600.0
ORI_OUT = FIG_DIR / "figure_02_reference_frames.png"

# one PNG per vector pointed at the viewer (Mars/orbit/triad/target dropped). Same
# geometry as figD; only the camera changes -> the object seen from new angles.
ORI_VIEW_SAIL_SPAN_KM = 3000.0
                                # opened up 1850->3000 so the FULL-LENGTH arrows (ORI_PANEL_GEOM_SCALE=1)
                                # fit with margin; combined with the derived panel side below this locks
                                # the thumbnail km/in to the main scene -> arrows render at the SAME size.
ORI_FACING_NUDGE_KM = 750.0     # screen-x nudge (km) for the viewer-facing vector's label
SHOW_NPERP_GUIDE = True         # faint guide along n_hat's projection perp to s_hat
                                # (the clock arc's far end), so the arc attaches to it
# (h_hat) panel -- it is REDUNDANT with the main in-orbit scene, which is itself viewed
# down h_hat; and (b) the e_B panel (e_A is kept -- it is the clock-angle reference, where
# the delta arc starts). Fewer panels -> each is larger.
ORI_VIEWS = [("sun", "s", -1), ("normal", "n", -1), ("eA", "e_A", -1)]  # (suffix, geo key, nudge -1=L/+1=R)
ORI_MAIN_SIDE_IN = 9.0        # v3: absolute side (in) of the square main annotated scene
# actual content -- Mars's projected position in the main render + the cropped row -- so the
# row tucks right under Mars (not under the main axes box, which has whitespace below Mars)
# and panel B's centre aligns with Mars's centre. These knobs feed that derivation:
ORI_SIDE_MARGIN_IN = 0.25     # left/right figure margin (in)
ORI_TITLE_BAND_IN = 0.0
ORI_CAPTION_BAND_IN = 0.0
ORI_MAIN_GAP_IN = 0.04        # gap between Mars (lowest main-scene content) and the row -- "very close"
ORI_PANEL_GAP_IN = 0.06       # gap between adjacent row panels -- "close together"
# v3: panel side DERIVED so the thumbnail km/in EQUALS the main scene's
# (2*HALF_SPAN / MAIN_SIDE == 2*SAIL_SPAN / PANEL_SIDE). With ORI_PANEL_GEOM_SCALE=1 this
# makes the column arrows + arrow-labels render at the SAME on-page size as the main scene
ORI_PANEL_SIDE_IN = ORI_MAIN_SIDE_IN * ORI_VIEW_SAIL_SPAN_KM / ORI_VIEW_HALF_SPAN_KM
ORI_THUMB_LABEL_SCALE = 1.0   # v3: arrow-label fontsize multiplier 0.36->1.0 (match the main scene)
ORI_PANEL_GEOM_SCALE = 1.0    # v3: full-length arrows/arcs (was 0.66) -> same proportions as the main scene
ORI_PANEL_VPAD_FRAC = 0.05    # v3: each panel is cropped (symmetrically about centre) to the box that
                              # holds the sail+vectors+labels in BOTH dimensions (square px kept -> arrows
                              # unchanged); this is the padding around that content box, as a fraction of
                              # the render size. Smaller -> tighter crop / closer packing.
ORI_FACING_LABEL_X = 0.155
                              # "<vec> odot" facing-the-viewer label -- pulled in from the top-right
                              # corner to sit just right of the A/B/C letter box (reads "A  s-hat odot").

_VC = {"n": "#d62728", "s": "#f0a500", "h": "#2a8c2a",
       "eA": "#7a5cff", "eB": "#b048d0", "triad": "#1f3a93", "spin": "#111111",
       "alpha": "#d62728", "delta": "#7a5cff", "target": "#0aa6c2",
       "r": "#1f77b4", "theta": "#ff7f0e"}   # figE: RTN radial / transverse axes

# Right-column thumbnail legend: view name -> (mathtext glyph, color) of the vector
# that faces the viewer in that panel.
VIEW_META = {"sun": (r"$\hat{s}$", _VC["s"]), "normal": (r"$\hat{n}$", _VC["n"]),
             "eA": (r"$e_A$", _VC["eA"]), "orbitnormal": (r"$\hat{h}$", _VC["h"]),
             "eB": (r"$e_B$", _VC["eB"])}


def _project(P, proj):
    """3D world point -> (px, py) in the PyVista screenshot's pixel coords
    (parallel projection; square window). Matches the camera set in the render."""
    d = np.asarray(P, float) - proj["focal"]
    ndc_x = float(np.dot(d, proj["right"])) / proj["scale"]
    ndc_y = float(np.dot(d, proj["up"])) / proj["scale"]
    return (0.5 + 0.5 * ndc_x) * proj["W"], (0.5 - 0.5 * ndc_y) * proj["H"]


def _arc_points(center, u0, u1, radius, n=64):
    """Great-circle (slerp) arc from unit u0 to unit u1, radius about center."""
    u0 = _unit(np.asarray(u0, float)); u1 = _unit(np.asarray(u1, float))
    om = math.acos(float(np.clip(np.dot(u0, u1), -1.0, 1.0)))
    if om < 1e-6:
        return np.array([center + radius * u0])
    ts = np.linspace(0.0, 1.0, n)
    so = math.sin(om)
    return np.array([center + radius * (math.sin((1 - t) * om) * u0
                                        + math.sin(t * om) * u1) / so for t in ts])


def _add_vec(pl, start, vec_unit, length, color, *_ignore):
    """Draw an arrow; return the 3D label anchor (just beyond the tip).
    *_ignore tolerates optional (label, font) positional args; text is now a
    matplotlib overlay (PyVista's font can't draw the hat/Greek glyphs)."""
    import pyvista as pv
    vec_unit = _unit(vec_unit)
    pl.add_mesh(pv.Arrow(start=tuple(start), direction=tuple(vec_unit), scale=length,
                         tip_length=0.13, tip_radius=0.028, shaft_radius=0.009),
                color=color, ambient=0.85, diffuse=0.25)
    return start + vec_unit * length * 1.12


def _orientation_geometry(traj):
    """Single-source the figD geometry so the full figure AND the sail-only
    turntable views (main_orientation_views) draw the IDENTICAL sail + vectors,
    differing only in camera. Picks the rightmost pure-cruise pose, lifts it to
    ORI_ALT_KM, and returns the pose, the 5 defining vectors, the McInnes
    (cone, clock) angles, the cone/clock arcs, the MME triad, the spin pole, and
    the ground target."""
    from reflectors.elements import mme2000_rotation_from_j2000
    try:
        a0, a1 = orbit_bounds(traj["u_transport_deg"], 1)
    except Exception:
        a0, a1 = 0, len(traj["r"]) - 1
    seg = np.arange(a0, a1 + 1)
    h_mean = _unit(traj["h"][seg].mean(axis=0))
    et_mid = float(traj["et_tdb"][seg[len(seg) // 2]])
    pole0 = _unit(_mars_rotation_to_j2000(et_mid) @ np.array([0.0, 0.0, 1.0]))
    up0 = _unit(pole0 - float(np.dot(pole0, h_mean)) * h_mean)
    cright0 = _unit(np.cross(up0, h_mean))
    idx = int(seg[int(np.argmax(traj["r"][seg] @ cright0))])   # rightmost in view
    f_alt = (MARS_R_KM + ORI_ALT_KM) / float(np.linalg.norm(traj["r"][idx]))
    r = traj["r"][idx] * f_alt
    n = traj["n"][idx]; s = traj["s"][idx]; h = traj["h"][idx]
    et_ref = float(traj["et_tdb"][idx])
    cone, clock, e_A, e_B = mcinnes_cone_clock(n, s, h, sign=CLOCK_SIGN)
    cone = float(cone); clock = float(clock) % 360.0
    e_A = _unit(e_A)
    Rmat = _mars_rotation_to_j2000(et_ref)
    pole = _unit(Rmat @ np.array([0.0, 0.0, 1.0]))           # spin axis (IAU_MARS +Z)
    Rmme = np.asarray(mme2000_rotation_from_j2000(et_ref))   # v_mme = Rmme @ v_j2000
    mme_axes = [Rmme[0, :], Rmme[1, :], Rmme[2, :]]          # MME X,Y,Z in J2000
    arc_a = _arc_points(r, n, s, ORI_ARC_RADIUS_KM)          # cone arc (alpha)
    n_perp = _unit(n - float(np.dot(n, s)) * s)
    arc_d = _arc_points(r, e_A, n_perp, ORI_ARC_RADIUS_KM * 0.72)  # clock arc (delta)
    tgt = Rmat @ _body_xyz(TARGET_LAT_DEG, TARGET_LON_DEG)
    return {"idx": idx, "seg": seg, "r": r, "n": n, "s": s, "h": h,
            "e_A": e_A, "e_B": e_B, "cone": cone, "clock": clock,
            "Rmat": Rmat, "pole": pole, "mme_axes": mme_axes, "et_ref": et_ref,
            "orbit_pts": traj["r"][seg] * f_alt,
            "arc_a": arc_a, "arc_d": arc_d, "tgt": tgt}


def _add_nperp_guide(pl, geo, scale=1.0):
    """Faint guide line along n_hat projected into the plane perp to s_hat -- the
    direction the clock (delta) arc sweeps TO. Drawn so the arc visibly attaches to
    a construction direction instead of floating. Toggle: SHOW_NPERP_GUIDE."""
    if not SHOW_NPERP_GUIDE:
        return
    import pyvista as pv
    r = geo["r"]
    n_perp = _unit(geo["n"] - float(np.dot(geo["n"], geo["s"])) * geo["s"])
    end = r + n_perp * (ORI_ARC_RADIUS_KM * 0.72 * 1.12 * scale)
    pl.add_mesh(pv.lines_from_points(np.array([r, end])),
                color=_VC["n"], line_width=2, opacity=0.45)


def _add_sail_vectors_arcs(pl, geo, fnt, scale=1.0):
    """Draw the sail (square + dog-ear roll marker), the 5 defining vectors
    (n,s,h,e_A,e_B), and the cone/clock arcs into the plotter. `scale` shrinks the
    vectors/guide/arcs relative to the fixed-size sail (used by the turntable
    panels so the sail fills more of the frame); figD's main scene uses its own
    inline draw and is unaffected."""
    import pyvista as pv
    r = geo["r"]; n = geo["n"]; s = geo["s"]; h = geo["h"]
    pent, ear, _ = _sail_quad(r, n, h, ORI_SAIL_SIZE_KM)            # sail size unchanged
    pl.add_mesh(_pv_polygon(pent), color="#cfd8e6", show_edges=True, edge_color="black",
                line_width=1.6, ambient=0.6, diffuse=0.5)
    pl.add_mesh(_pv_polygon(ear), color="white", show_edges=True, edge_color="black",
                line_width=1.0, ambient=0.85, diffuse=0.3)
    L = ORI_VEC_LEN_KM * scale
    _add_vec(pl, r, n, L, _VC["n"])
    _add_vec(pl, r, s, L, _VC["s"])
    _add_vec(pl, r, h, L, _VC["h"])
    _add_vec(pl, r, geo["e_A"], L * 0.82, _VC["eA"])
    _add_vec(pl, r, geo["e_B"], L * 0.82, _VC["eB"])
    _add_nperp_guide(pl, geo, scale)
    n_perp = _unit(n - float(np.dot(n, s)) * s)
    arc_a = _arc_points(r, n, s, ORI_ARC_RADIUS_KM * scale)
    arc_d = _arc_points(r, geo["e_A"], n_perp, ORI_ARC_RADIUS_KM * 0.72 * scale)
    for arc, col, lab in ((arc_a, _VC["alpha"], "α"), (arc_d, _VC["delta"], "δ")):
        pl.add_mesh(pv.lines_from_points(arc), color=col, line_width=5)
        pl.add_point_labels([arc[len(arc) // 2]], [lab], font_size=int(fnt * 1.15),
                            text_color=col, font_family="times", bold=True,
                            shape=None, show_points=False, always_visible=True)


def render_orientation_pyvista(traj, idx=None):
    import pyvista as pv

    # Geometry (pose, vectors, angles, arcs, target) single-sourced so the sail-only
    # turntable views (main_orientation_views) show the IDENTICAL object.
    geo = _orientation_geometry(traj)
    r = geo["r"]; n = geo["n"]; s = geo["s"]; h = geo["h"]
    e_A = geo["e_A"]; e_B = geo["e_B"]; cone = geo["cone"]; clock = geo["clock"]
    Rmat = geo["Rmat"]; pole = geo["pole"]; mme_axes = geo["mme_axes"]
    orbit_pts = geo["orbit_pts"]; arc_a = geo["arc_a"]; arc_d = geo["arc_d"]
    tgt = geo["tgt"]

    # the viewer; Mars spin axis (pole) is "up" (vertical). pole is ~perp to h_hat
    # for a near-polar orbit, so this is a clean orbit-face-on view, pole vertical.
    cam_dir = _unit(h)
    up = _unit(pole - float(np.dot(pole, cam_dir)) * cam_dir)
    if ORI_VIEW_AZIM_DEG:                      # optional spin about the pole
        a = math.radians(ORI_VIEW_AZIM_DEG)
        cam_dir = _unit(cam_dir * math.cos(a) + np.cross(up, cam_dir) * math.sin(a)
                        + up * float(np.dot(up, cam_dir)) * (1 - math.cos(a)))
    cam_right = _unit(np.cross(up, cam_dir))
    view_center = r * ORI_VIEW_CENTER_FRAC
    fnt = int(26 * PYVISTA_SUPERSAMPLE)

    pl = pv.Plotter(off_screen=True, window_size=list(PYVISTA_WINDOW_PX), lighting="none")
    pl.set_background(RENDER_BG)
    sun_light = pv.Light(position=tuple(s * 1.0e6), focal_point=(0, 0, 0), color="white")
    sun_light.positional = False; sun_light.intensity = 1.15
    pl.add_light(sun_light)

    # Mars (textured, oriented to et_ref)
    mars = _seamless_mars_mesh(Rmat, PYVISTA_SPHERE_RES)
    tex = _pv_texture()
    if tex is not None:
        pl.add_mesh(mars, texture=tex, smooth_shading=True, ambient=0.30, diffuse=0.95)
    else:
        pl.add_mesh(mars, color="#a9603f", smooth_shading=True, ambient=0.3, diffuse=0.95)

    # Inertial triad at Mars centre: X, Y in the equatorial plane; Z = the Mars
    # spin axis (pole). Arrows extend past Mars R (3396) so they emerge from the
    # disk rather than being buried inside it. (Frame named in the caption.)
    for axis in mme_axes:                      # X, Y, Z arrows (labels via overlay)
        pl.add_mesh(pv.Arrow(start=(0, 0, 0), direction=tuple(_unit(axis)),
                             scale=ORI_TRIAD_LEN_KM, tip_length=0.10, tip_radius=0.020,
                             shaft_radius=0.006), color=_VC["triad"], ambient=0.9)

    # one orbit (schematic altitude) + the sail pose
    pl.add_mesh(pv.lines_from_points(orbit_pts), color="#444444", line_width=3)
    pent, ear, n_hat = _sail_quad(r, n, h, ORI_SAIL_SIZE_KM)
    pl.add_mesh(_pv_polygon(pent), color="#cfd8e6", show_edges=True, edge_color="black",
                line_width=1.6, ambient=0.6, diffuse=0.5)
    pl.add_mesh(_pv_polygon(ear), color="white", show_edges=True, edge_color="black",
                line_width=1.0, ambient=0.85, diffuse=0.3)

    # the defining vectors from the sail centre
    _add_vec(pl, r, n, ORI_VEC_LEN_KM, _VC["n"], "n̂", fnt)
    _add_vec(pl, r, s, ORI_VEC_LEN_KM, _VC["s"], "ŝ", fnt)
    _add_vec(pl, r, h, ORI_VEC_LEN_KM, _VC["h"], "ĥ", fnt)
    _add_vec(pl, r, e_A, ORI_VEC_LEN_KM * 0.82, _VC["eA"], "e_A", fnt)
    _add_vec(pl, r, e_B, ORI_VEC_LEN_KM * 0.82, _VC["eB"])   # second clock-ref axis
    _add_nperp_guide(pl, geo)                                # thin red guide (n proj. perp s)

    # cone arc alpha = angle(n, s)
    arc_a = _arc_points(r, n, s, ORI_ARC_RADIUS_KM)
    pl.add_mesh(pv.lines_from_points(arc_a), color=_VC["alpha"], line_width=5)
    pl.add_point_labels([arc_a[len(arc_a) // 2]], ["α"],
                        font_size=int(fnt * 1.15), text_color=_VC["alpha"],
                        font_family="times", bold=True, shape=None,
                        show_points=False, always_visible=True)

    # clock arc delta = azimuth of n about s, from e_A (in the plane perp to s)
    n_perp = _unit(n - float(np.dot(n, s)) * s)
    arc_d = _arc_points(r, e_A, n_perp, ORI_ARC_RADIUS_KM * 0.72)
    pl.add_mesh(pv.lines_from_points(arc_d), color=_VC["delta"], line_width=5)
    pl.add_point_labels([arc_d[len(arc_d) // 2]], ["δ"],
                        font_size=int(fnt * 1.15), text_color=_VC["delta"],
                        font_family="times", bold=True, shape=None,
                        show_points=False, always_visible=True)

    # ground delivery target
    tgt = Rmat @ _body_xyz(TARGET_LAT_DEG, TARGET_LON_DEG)
    pl.add_mesh(pv.Sphere(radius=ORI_TARGET_RADIUS_KM, center=tuple(tgt)),
                color=_VC["target"], ambient=0.85, diffuse=0.2)

    pl.enable_parallel_projection()
    pl.camera.focal_point = tuple(view_center)
    pl.camera.position = tuple(view_center + cam_dir * 12.0 * MARS_R_KM)
    pl.camera.up = tuple(up)
    pl.camera.parallel_scale = float(ORI_VIEW_HALF_SPAN_KM)
    img = pl.screenshot(return_img=True)
    pl.close()
    print(f"  cone alpha = {cone:.2f} deg, clock delta = {clock:.2f} deg")

    # Crisp text as a matplotlib overlay (PyVista cannot render hat/Greek glyphs):
    # (anchor_3d, mathtext, color, fontsize_pt)
    L = ORI_VEC_LEN_KM
    labels = [
        (r + _unit(n) * L * 1.12, r"$\hat{n}$", _VC["n"], 23),
        (r + _unit(s) * L * 1.12 + cam_right * 320.0, r"$\hat{s}$", _VC["s"], 23),
        (r + _unit(e_A) * L * 0.82 * 1.12, r"$e_A$", _VC["eA"], 19),
        (r + _unit(e_B) * L * 0.82 * 1.12, r"$e_B$", _VC["eB"], 19),
        (arc_a[len(arc_a) // 2], r"$\alpha$", _VC["alpha"], 26),
        (arc_d[len(arc_d) // 2], r"$\delta$", _VC["delta"], 26),
        (tgt - up * 5.5 * ORI_TARGET_RADIUS_KM,
         r"$(40^\circ\mathrm{N},\ 200^\circ\mathrm{E})$", _VC["target"], 18,
         {"fontweight": "normal"}),
        (-pole * MARS_R_KM * 1.06, "South Pole", _VC["spin"], 13),
        (_unit(mme_axes[0]) * ORI_TRIAD_LEN_KM * 1.07, r"$X$", _VC["triad"], 19),
        (_unit(mme_axes[1]) * ORI_TRIAD_LEN_KM * 1.07, r"$Y$", _VC["triad"], 19),
        (_unit(mme_axes[2]) * ORI_TRIAD_LEN_KM * 1.07, r"$Z$", _VC["triad"], 19),
    ]
    proj = {"focal": view_center, "right": cam_right, "up": up,
            "scale": float(ORI_VIEW_HALF_SPAN_KM), "W": img.shape[1], "H": img.shape[0]}
    return img, labels, proj


def main_orientation():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_sol(SOL)
    geo = _orientation_geometry(traj)                 # shared by main scene + thumbnails
    img, labels, proj = render_orientation_pyvista(traj, 0)

    # ----------------------------------------------------------------------------------
    # Render the 3 turntable panels up front. Their cropped size AND Mars's projected
    # position in the main render drive the (derived) figure size + the row placement, so
    # ----------------------------------------------------------------------------------
    panels = [(name, key, nudge) + render_sail_view_pyvista(geo, geo[key],
                                                            facing_key=key, nudge_sign=nudge)
              for (name, key, nudge) in ORI_VIEWS]
    Hpx = float(panels[0][3].shape[0]); Wpx = float(panels[0][3].shape[1])

    # 2D content crop (symmetric about centre; square px kept -> arrows stay main-scene size)
    padx = ORI_PANEL_VPAD_FRAC * Wpx; pady = ORI_PANEL_VPAD_FRAC * Hpx
    ex = ey = 0.0                                       # max half-extent of content from centre (px)
    for (_n, _k, _g, timg, tlabels, tproj) in panels:
        for anchor, txt, color, fs in tlabels:
            px, py = _project(anchor, tproj)
            ex = max(ex, abs(px - Wpx / 2.0)); ey = max(ey, abs(py - Hpx / 2.0))
    ex = min(Wpx / 2.0, ex + padx); ey = min(Hpx / 2.0, ey + pady)
    x_lo, x_hi = Wpx / 2.0 - ex, Wpx / 2.0 + ex
    y_lo, y_hi = Hpx / 2.0 - ey, Hpx / 2.0 + ey
    xkeep = 2.0 * ex / Wpx; ykeep = 2.0 * ey / Hpx
    n_panels = len(panels)
    panel_w_in = ORI_PANEL_SIDE_IN * xkeep
    panel_h_in = ORI_PANEL_SIDE_IN * ykeep             # square px preserved (W==H render)
    row_w_in = n_panels * panel_w_in + (n_panels - 1) * ORI_PANEL_GAP_IN

    # Mars in the main render: centre = projected origin; lowest content = lowest non-white
    # rendered pixel (orbit ring / disc / vectors) unioned with the overlay labels.
    mW, mH = float(proj["W"]), float(proj["H"])
    mars_cx_px = _project(np.zeros(3), proj)[0]
    frac_x = mars_cx_px / mW                            # Mars centre, fraction across the main image
    arr = np.asarray(img)[..., :3]
    rows_with_content = np.where(np.any(arr < 250, axis=2).any(axis=1))[0]
    content_bottom_py = float(rows_with_content.max()) if rows_with_content.size else mH
    for entry in labels:                               # union with overlay labels (South Pole, X/Y/Z, ...)
        _, py = _project(entry[0], proj)
        content_bottom_py = max(content_bottom_py, py + 0.5 * (entry[3] / 72.0) * (mH / ORI_MAIN_SIDE_IN))
    cb_frac = min(1.0, content_bottom_py / mH)          # 0 = top .. 1 = bottom of the main image

    # --- horizontal: row centred on Mars; main scene shifted so Mars lands there ---
    half_left = max(row_w_in / 2.0, ORI_MAIN_SIDE_IN * frac_x)
    half_right = max(row_w_in / 2.0, ORI_MAIN_SIDE_IN * (1.0 - frac_x))
    fig_w_in = half_left + half_right + 2.0 * ORI_SIDE_MARGIN_IN
    x_mars_in = ORI_SIDE_MARGIN_IN + half_left
    main_x_in = x_mars_in - ORI_MAIN_SIDE_IN * frac_x
    x_start_in = x_mars_in - row_w_in / 2.0

    # --- vertical: caption band, row tucked under Mars's lowest content, main scene, title band ---
    main_bottom_in = (ORI_CAPTION_BAND_IN + panel_h_in + ORI_MAIN_GAP_IN
                      - ORI_MAIN_SIDE_IN * (1.0 - cb_frac))
    main_top_in = main_bottom_in + ORI_MAIN_SIDE_IN
    fig_h_in = main_top_in + ORI_TITLE_BAND_IN
    content_bottom_in = main_bottom_in + ORI_MAIN_SIDE_IN * (1.0 - cb_frac)
    row_top_in = content_bottom_in - ORI_MAIN_GAP_IN
    row_y_in = row_top_in - panel_h_in
    print(f"  row crop: xkeep={xkeep:.3f} ykeep={ykeep:.3f}  panel {panel_w_in:.2f}x{panel_h_in:.2f} in")
    print(f"  Mars frac_x={frac_x:.3f} content_bottom_frac={cb_frac:.3f}  fig {fig_w_in:.2f}x{fig_h_in:.2f} in")

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    cx = 0.5                                            # title/caption centred on the full figure

    # --- main annotated scene (drawn first; the row overlaps its lower white whitespace) ---
    ax = fig.add_axes([main_x_in / fig_w_in, main_bottom_in / fig_h_in,
                       ORI_MAIN_SIDE_IN / fig_w_in, ORI_MAIN_SIDE_IN / fig_h_in])
    ax.imshow(img)
    for entry in labels:
        anchor, txt, color, fs = entry[:4]
        extra = dict(entry[4]) if len(entry) > 4 else {}   # optional per-label kwargs (copy)
        weight = extra.pop("fontweight", "bold")           # per-label weight; default bold
        px, py = _project(anchor, proj)
        ax.text(px, py, txt, color=color, fontsize=fs, ha="center", va="center",
                fontweight=weight, zorder=10, **extra)
    ax.set_axis_off()

    # --- bottom row: cropped turntable panels, centred under Mars, tucked just below it.
    # Drawn after the main scene so the panels sit ON TOP of the main render's lower (white)
    # whitespace -> they read as a row hugging Mars. alpha/delta angle labels included. ---
    wf = panel_w_in / fig_w_in; hf = panel_h_in / fig_h_in
    for i, (name, key, nudge, timg, tlabels, tproj) in enumerate(panels):
        x0 = (x_start_in + i * (panel_w_in + ORI_PANEL_GAP_IN)) / fig_w_in
        axt = fig.add_axes([x0, row_y_in / fig_h_in, wf, hf])
        axt.imshow(timg)
        axt.set_aspect("auto")                          # fill the cropped box; pixels stay square
        axt.set_xlim(x_lo, x_hi); axt.set_ylim(y_hi, y_lo)   # crop to content (y inverted)
        for anchor, txt, color, fs in tlabels:          # vectors + alpha/delta (colour-blind aid)
            px, py = _project(anchor, tproj)
            axt.text(px, py, txt, color=color, fontsize=fs * ORI_THUMB_LABEL_SCALE,
                     ha="center", va="center", fontweight="bold", zorder=10)
        axt.set_xticks([]); axt.set_yticks([])
        for sp in axt.spines.values():                  # borderless
            sp.set_visible(False)
        glyph, col = VIEW_META[name]
        axt.text(0.045, 0.955, "ABCDE"[i], transform=axt.transAxes, fontsize=18,
                 fontweight="bold", va="top", ha="left",
                 bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="black", lw=1.0))
        # glyph + odot = "this vector points out of the page, toward you"; sits just
        axt.text(ORI_FACING_LABEL_X, 0.955, glyph + r"$\;\odot$", transform=axt.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="left", color=col)

    fig.savefig(
        ORI_OUT,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"  wrote {ORI_OUT}")


def _sail_vector_labels(geo, cam_right, facing_key=None, nudge_sign=-1, scale=1.0):
    """Overlay label entries (anchor, mathtext, color, fontsize) for the sail's 5
    vectors + the cone/clock arc letters. Used by the turntable views, where h_hat
    of the viewer-facing vector (facing_key) projects onto the sail centre, so it is
    nudged sideways by nudge_sign * cam_right (-1 = left, +1 = right). `scale` matches
    the shrink applied to the drawn vectors/arcs in _add_sail_vectors_arcs."""
    r = geo["r"]; L = ORI_VEC_LEN_KM * scale
    n_perp = _unit(geo["n"] - float(np.dot(geo["n"], geo["s"])) * geo["s"])
    arc_a = _arc_points(r, geo["n"], geo["s"], ORI_ARC_RADIUS_KM * scale)
    arc_d = _arc_points(r, geo["e_A"], n_perp, ORI_ARC_RADIUS_KM * 0.72 * scale)
    base = [
        ("n", r + _unit(geo["n"]) * L * 1.12, r"$\hat{n}$", _VC["n"], 23),
        ("s", r + _unit(geo["s"]) * L * 1.12 + cam_right * 320.0 * scale, r"$\hat{s}$", _VC["s"], 23),
        ("h", r + _unit(geo["h"]) * L * 1.12, r"$\hat{h}$", _VC["h"], 23),
        ("e_A", r + _unit(geo["e_A"]) * L * 0.82 * 1.12, r"$e_A$", _VC["eA"], 19),
        ("e_B", r + _unit(geo["e_B"]) * L * 0.82 * 1.12, r"$e_B$", _VC["eB"], 19),
        ("arc_a", arc_a[len(arc_a) // 2], r"$\alpha$", _VC["alpha"], 26),
        ("arc_d", arc_d[len(arc_d) // 2], r"$\delta$", _VC["delta"], 26),
    ]
    out = []
    for key, anchor, txt, color, fs in base:
        if key == facing_key:
            anchor = anchor + nudge_sign * cam_right * ORI_FACING_NUDGE_KM * scale
        out.append((anchor, txt, color, fs))
    return out


def render_sail_view_pyvista(geo, cam_vec, facing_key=None, nudge_sign=-1):
    """Render ONLY the sail + vectors + arcs (no Mars/orbit/triad/target), viewed
    down cam_vec so that vector points out of the screen toward the viewer. The
    sail and all vectors are frozen in world space (identical to figD's geo); only
    the camera changes -- the viewer sees the same 3D object from a new angle."""
    import pyvista as pv
    r = geo["r"]
    cam_dir = _unit(cam_vec)
    up = geo["pole"] - float(np.dot(geo["pole"], cam_dir)) * cam_dir
    if np.linalg.norm(up) < 1e-3:                  # cam ~along the pole: fall back to h
        up = geo["h"] - float(np.dot(geo["h"], cam_dir)) * cam_dir
    up = _unit(up); cam_right = _unit(np.cross(up, cam_dir))
    fnt = int(26 * PYVISTA_SUPERSAMPLE)
    pl = pv.Plotter(off_screen=True, window_size=list(PYVISTA_WINDOW_PX), lighting="none")
    pl.set_background(RENDER_BG)
    sun_light = pv.Light(position=tuple(_unit(geo["s"]) * 1.0e6),
                         focal_point=tuple(r), color="white")
    sun_light.positional = False; sun_light.intensity = 1.15
    pl.add_light(sun_light)
    _add_sail_vectors_arcs(pl, geo, fnt, scale=ORI_PANEL_GEOM_SCALE)
    pl.enable_parallel_projection()
    pl.camera.focal_point = tuple(r)
    pl.camera.position = tuple(r + cam_dir * 12.0 * MARS_R_KM)
    pl.camera.up = tuple(up)
    pl.camera.parallel_scale = float(ORI_VIEW_SAIL_SPAN_KM)
    img = pl.screenshot(return_img=True)
    pl.close()
    labels = _sail_vector_labels(geo, cam_right, facing_key, nudge_sign,
                                 scale=ORI_PANEL_GEOM_SCALE)
    proj = {"focal": r, "right": cam_right, "up": up,
            "scale": float(ORI_VIEW_SAIL_SPAN_KM), "W": img.shape[1], "H": img.shape[0]}
    return img, labels, proj


def main_orientation_views():
    """Sail-only 'turntable': one PNG per defining vector pointed at the viewer
    to be compiled later as side panels. Separate PNGs for now."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_sol(SOL)
    geo = _orientation_geometry(traj)
    glyph = {"sun": r"$\hat{s}$ (Sun-line)", "normal": r"$\hat{n}$ (sail normal)",
             "eA": r"$e_A$ (clock ref.)", "orbitnormal": r"$\hat{h}$ (orbit normal)",
             "eB": r"$e_B$"}
    for name, key, nudge in ORI_VIEWS:
        img, labels, proj = render_sail_view_pyvista(geo, geo[key],
                                                     facing_key=key, nudge_sign=nudge)
        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img)
        for entry in labels:
            anchor, txt, color, fs = entry[:4]
            px, py = _project(anchor, proj)
            ax.text(px, py, txt, color=color, fontsize=fs, ha="center", va="center",
                    fontweight="bold", zorder=10)
        ax.set_axis_off()
        fig.suptitle(f"Sail orientation — {glyph[name]} toward viewer",
                     fontsize=15, y=0.95)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
        out = FIG_DIR / f"figD_view_{name}_v3.png"
        fig.savefig(out, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  wrote {out}")


# ===========================================================================
# Fig E: the orbit-rotating RTN frame (escape / capture / heliocentric cruise)
# ===========================================================================
# Companion to figD. SAME Mars/orbit/sail pose, but the sail orientation is now
# referenced to the ORBIT-ROTATING RTN frame: r_hat radial-outward, theta_hat the
# in-plane transverse (= h_hat x r_hat, ~ along-track), h_hat the orbit normal --
# the frame the escape (secular_blended.py, mpc_escape.py) and heliocentric-cruise
# (cruise_piecewise.py) steering laws command in. Cone alpha is measured from the
# RADIAL r_hat (NOT the Sun-line s_hat as in figD); clock delta is the azimuth of
# n_hat about r_hat in the (theta_hat, h_hat) plane, from theta_hat:
#   n = cos(alpha) r_hat + sin(alpha) cos(delta) theta_hat + sin(alpha) sin(delta) h_hat
# (McInnes 1999 Eq. 4.7 in the orbit frame).
RTN_VIEW_TILT_DEG = 35.0      # camera tilt off h_hat toward r_hat -> 3/4 view showing all 3 axes
ORI_RTN_OUT = FIG_DIR / "figE_rtn_frame.png"


def render_rtn_pyvista(traj):
    import pyvista as pv
    geo = _orientation_geometry(traj)
    r = geo["r"]; n = geo["n"]; h = geo["h"]; s = geo["s"]
    Rmat = geo["Rmat"]; pole = geo["pole"]; orbit_pts = geo["orbit_pts"]; tgt = geo["tgt"]
    r_hat = _unit(r); h_hat = _unit(h)
    th_hat = _unit(np.cross(h_hat, r_hat))          # in-plane transverse (~ velocity direction)
    n_u = _unit(n)
    cone = math.degrees(math.acos(float(np.clip(np.dot(n_u, r_hat), -1.0, 1.0))))
    n_perp = _unit(n_u - float(np.dot(n_u, r_hat)) * r_hat)
    clock = math.degrees(math.atan2(float(np.dot(n_u, h_hat)),
                                    float(np.dot(n_u, th_hat)))) % 360.0
    arc_a = _arc_points(r, r_hat, n_u, ORI_ARC_RADIUS_KM)               # cone (from r_hat)
    arc_d = _arc_points(r, th_hat, n_perp, ORI_ARC_RADIUS_KM * 0.72)    # clock (about r_hat)

    # 3/4 camera: tilt off h_hat toward r_hat so all three RTN axes are visible;
    # keep the Mars spin pole vertical (as in figD).
    t = math.radians(RTN_VIEW_TILT_DEG)
    cam_dir = _unit(h_hat * math.cos(t) + r_hat * math.sin(t))
    up = pole - float(np.dot(pole, cam_dir)) * cam_dir
    if np.linalg.norm(up) < 1e-2:                   # degenerate -> fall back to h_hat-up
        up = h_hat - float(np.dot(h_hat, cam_dir)) * cam_dir
    up = _unit(up); cam_right = _unit(np.cross(up, cam_dir))
    view_center = r * ORI_VIEW_CENTER_FRAC
    fnt = int(26 * PYVISTA_SUPERSAMPLE)

    pl = pv.Plotter(off_screen=True, window_size=list(PYVISTA_WINDOW_PX), lighting="none")
    pl.set_background(RENDER_BG)
    sun_light = pv.Light(position=tuple(s * 1.0e6), focal_point=(0, 0, 0), color="white")
    sun_light.positional = False; sun_light.intensity = 1.15
    pl.add_light(sun_light)

    mars = _seamless_mars_mesh(Rmat, PYVISTA_SPHERE_RES)
    tex = _pv_texture()
    if tex is not None:
        pl.add_mesh(mars, texture=tex, smooth_shading=True, ambient=0.30, diffuse=0.95)
    else:
        pl.add_mesh(mars, color="#a9603f", smooth_shading=True, ambient=0.3, diffuse=0.95)

    pl.add_mesh(pv.lines_from_points(orbit_pts), color="#444444", line_width=3)
    pent, ear, _ = _sail_quad(r, n, h, ORI_SAIL_SIZE_KM)
    pl.add_mesh(_pv_polygon(pent), color="#cfd8e6", show_edges=True, edge_color="black",
                line_width=1.6, ambient=0.6, diffuse=0.5)
    pl.add_mesh(_pv_polygon(ear), color="white", show_edges=True, edge_color="black",
                line_width=1.0, ambient=0.85, diffuse=0.3)

    # RTN triad (r,theta,h) + sail normal n + Sun-line s (faint, for the figD contrast)
    _add_vec(pl, r, r_hat, ORI_VEC_LEN_KM, _VC["r"])
    _add_vec(pl, r, th_hat, ORI_VEC_LEN_KM, _VC["theta"])
    _add_vec(pl, r, h_hat, ORI_VEC_LEN_KM, _VC["h"])
    _add_vec(pl, r, n_u, ORI_VEC_LEN_KM, _VC["n"])
    _add_vec(pl, r, _unit(s), ORI_VEC_LEN_KM * 0.9, _VC["s"])

    for arc, col, lab in ((arc_a, _VC["alpha"], "α"), (arc_d, _VC["delta"], "δ")):
        pl.add_mesh(pv.lines_from_points(arc), color=col, line_width=5)
        pl.add_point_labels([arc[len(arc) // 2]], [lab], font_size=int(fnt * 1.15),
                            text_color=col, font_family="times", bold=True,
                            shape=None, show_points=False, always_visible=True)

    pl.add_mesh(pv.Sphere(radius=ORI_TARGET_RADIUS_KM, center=tuple(tgt)),
                color=_VC["target"], ambient=0.85, diffuse=0.2)

    pl.enable_parallel_projection()
    pl.camera.focal_point = tuple(view_center)
    pl.camera.position = tuple(view_center + cam_dir * 12.0 * MARS_R_KM)
    pl.camera.up = tuple(up)
    pl.camera.parallel_scale = float(ORI_VIEW_HALF_SPAN_KM)
    img = pl.screenshot(return_img=True)
    pl.close()
    print(f"  RTN cone alpha_r = {cone:.2f} deg, clock delta_r = {clock:.2f} deg")

    L = ORI_VEC_LEN_KM
    labels = [
        (r + r_hat * L * 1.12, r"$\hat{r}$", _VC["r"], 23),
        (r + th_hat * L * 1.12, r"$\hat{\theta}$", _VC["theta"], 23),
        (r + h_hat * L * 1.12, r"$\hat{h}$", _VC["h"], 23),
        (r + n_u * L * 1.12, r"$\hat{n}$", _VC["n"], 23),
        (r + _unit(s) * L * 0.9 * 1.12 + cam_right * 320.0, r"$\hat{s}$", _VC["s"], 20),
        (arc_a[len(arc_a) // 2], r"$\alpha$", _VC["alpha"], 26),
        (arc_d[len(arc_d) // 2], r"$\delta$", _VC["delta"], 26),
        (tgt - up * 5.5 * ORI_TARGET_RADIUS_KM,
         r"$(40^\circ\mathrm{N},\ 200^\circ\mathrm{E})$", _VC["target"], 18, {"fontweight": "normal"}),
        (-pole * MARS_R_KM * 1.06, "South Pole", _VC["spin"], 13),
    ]
    proj = {"focal": view_center, "right": cam_right, "up": up,
            "scale": float(ORI_VIEW_HALF_SPAN_KM), "W": img.shape[1], "H": img.shape[0]}
    return img, labels, proj


def main_rtn():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_sol(SOL)
    img, labels, proj = render_rtn_pyvista(traj)
    fig = plt.figure(figsize=(12.0, 12.5))
    ax = fig.add_axes([0.02, 0.075, 0.96, 0.86])
    ax.imshow(img)
    for entry in labels:
        anchor, txt, color, fs = entry[:4]
        extra = dict(entry[4]) if len(entry) > 4 else {}
        weight = extra.pop("fontweight", "bold")
        px, py = _project(anchor, proj)
        ax.text(px, py, txt, color=color, fontsize=fs, ha="center", va="center",
                fontweight=weight, zorder=10, **extra)
    ax.set_axis_off()
    fig.suptitle("The orbit-rotating RTN frame (escape / capture / heliocentric cruise)",
                 fontsize=16, y=0.965)
    cap = (
        r"RTN basis at the sail: $\hat{r}$ radial-outward, $\hat{\theta}=\hat{h}\times\hat{r}$ "
        r"in-plane transverse ($\approx$ along-track), $\hat{h}$ orbit normal.  "
        r"$\hat{n}$ sail normal; $\hat{s}$ Sun-line." "\n"
        r"Escape / capture / cruise steering: "
        r"$\hat{n}=\cos\alpha\,\hat{r}+\sin\alpha\cos\delta\,\hat{\theta}"
        r"+\sin\alpha\sin\delta\,\hat{h}$  (McInnes 1999 Eq. 4.7 in the orbit frame; "
        r"secular_blended.py / mpc_escape.py / cruise_piecewise.py)." "\n"
        r"Cone $\alpha$ is from the RADIAL $\hat{r}$, clock $\delta$ about $\hat{r}$ from "
        r"$\hat{\theta}$ -- contrast figD, where station-keeping measures cone from the "
        r"Sun-line $\hat{s}$.  Sail/vectors exaggerated; altitude $\approx$500 km."
    )
    fig.text(0.5, 0.038, cap, ha="center", va="center", fontsize=10.5)
    fig.savefig(ORI_RTN_OUT, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  wrote {ORI_RTN_OUT}")


if __name__ == "__main__":
    main_orientation()
