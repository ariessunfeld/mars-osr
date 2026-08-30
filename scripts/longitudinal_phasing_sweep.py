"""Fluence sweep over initial mean anomaly and season.

Each grid point propagates one Mars solar sol with zonal J2 gravity, the Sun as
a third body, and a sun-pointing cruise attitude. Delivery windows and their
irradiance and fluence are evaluated with
:func:`reflectors.visibility.find_delivery_windows`.

Physics included (all via public ``reflectors`` modules):
  Mars obliquity            SPICE IAU_MARS frame (pck00011.tpc)
  Sun distance (1/r²)       reflectors.ephemeris.body_state per sample
  Sun cone half-angle       reflectors.visibility.bisector_normal
  Beam spot size ellipse    reflectors.beam.beam_footprint_semi_axes_km
  Sun angular diameter      reflectors.beam.sun_angular_diameter_rad
  α_max slew constraint     find_delivery_windows alpha_max_rad_s2 filter
  Sail efficiency           reflectors.beam.specular_reflectance via
                            reflectors.sail_designs.make_canonical_sail
  Surface horizon (10°)     find_delivery_windows elev_min_deg
  Sail sunlit (umbra)       reflectors.shadow.in_mars_umbra
  Target above horizon      surface.surface_point_position + great-circle
  Optical propagation       vacuum, fixed transmission = 1.0
  Sub-solar latitude        body_state("SUN", et, frame="IAU_MARS")

Model scope:
  - sun-pointing cruise attitude;
  - one-sol propagation at each grid point, excluding multi-sol secular
    evolution; and
  - vacuum optical propagation, without target BRDF or sail thermal limits.

CLI:
  python scripts/longitudinal_phasing_sweep.py [options]

Defaults use a = 3903.9245 km, LTAN 18 h, target (40°N, 200°E),
σ = 0.018 kg/m², and
α_max = 0.003 deg/s² ≈ 5.236e-5 rad/s².

Output:
  CSV  one row per (M0, epoch) grid point with per-window aggregates
  PNG  2-panel heatmap (fluence + n_windows) over the (M0, season) grid

"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from reflectors.attitude import sun_pointing
from reflectors.dynamics import PropagationOptions, propagate
from reflectors.ephemeris import AU_KM, body_state, utc_to_et
from reflectors.kernels import load_kernels
from reflectors.mars_constants import (
    MARS_SIDEREAL_YEAR_S,
    SECONDS_PER_SOLAR_SOL_S,
)
from reflectors.parallel import CloudpickleMap, configure_multiprocessing_for_spice
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import initial_state_j2000
from reflectors.termination import AltitudeFloor
from reflectors.third_body import sun_third_body
from reflectors.visibility import find_delivery_windows


logger = logging.getLogger(__name__)

# Default design point.
DEFAULT_EPOCH_ANCHOR_UTC = "2028-02-11T12:42:00"   # Mars perihelion 2028
DEFAULT_A_KM = 3903.9245                            # refined a*
DEFAULT_LTAN_H = 18.0                               # dawn-dusk sun-sync
DEFAULT_TARGET_LAT_DEG = 40.0
DEFAULT_TARGET_LON_DEG = 200.0
DEFAULT_SIGMA_KG_PER_M2 = 0.018
DEFAULT_ALPHA_MAX_DEG_PER_S2 = 0.003                # 5.236e-5 rad/s²
DEFAULT_ALTITUDE_FLOOR_KM = 300.0
DEFAULT_ELEV_MIN_DEG = 10.0
DEFAULT_BISECTOR_COS_ALPHA_MIN = 0.1                # half-angle ≤ ~84°
DEFAULT_T_EVAL_CADENCE_S = 60.0
VACUUM_TRANSMISSION = 1.0


@dataclass(frozen=True)
class GridResult:
    """Per-(M0, epoch) result row."""
    # Inputs
    m0_deg: float
    epoch_utc: str
    epoch_et: float
    season_frac_of_mars_year: float        # 0 = perihelion anchor; 0.5 ≈ aphelion
    # Geometry context
    r_mars_sun_au: float
    sub_solar_lat_iau_mars_deg: float
    # Window aggregates
    n_windows: int
    n_windows_pre_filter: int              # before α_max filter
    total_fluence_J_per_m2: float
    peak_irradiance_W_per_m2: float        # max across windows
    mean_window_duration_s: float
    sum_window_duration_s: float
    peak_footprint_semi_major_km: float    # max across windows
    peak_alpha_demand_max_rad_s2: float    # max across windows (info; α_max filter
                                           #   already removed any window > α_max)
    # Bookkeeping
    wall_s: float


def _sub_solar_lat_iau_mars_deg(et: float) -> float:
    """Sun's planetographic latitude in IAU_MARS — Mars seasonal declination.

    Equivalent to L_s-coupled obliquity projection. At Mars-N solstice
    (perihelion proximity → southern summer), sub-sol lat ≈ -25°; at
    Mars-N summer solstice ≈ +25°. Anything in between for general epochs.
    """
    state, _ = body_state("SUN", et, frame="IAU_MARS",
                          abcorr="NONE", observer="MARS")
    rx, ry, rz = float(state[0]), float(state[1]), float(state[2])
    return math.degrees(math.asin(rz / math.sqrt(rx * rx + ry * ry + rz * rz)))


def _mars_sun_distance_au(et: float) -> float:
    state, _ = body_state("MARS", et, frame="J2000",
                          abcorr="NONE", observer="SUN")
    return float(np.linalg.norm(state[:3])) / AU_KM


def evaluate_grid_point(
    m0_deg: float,
    epoch_utc: str,
    *,
    a_km: float,
    ltan_h: float,
    sigma_kg_per_m2: float,
    target_lat_deg: float,
    target_lon_deg: float,
    alpha_max_rad_s2: float,
    elev_min_deg: float,
    bisector_cos_alpha_min: float,
    altitude_floor_km: float,
    atmospheric_transmission: float,
    t_eval_cadence_s: float,
    zonal_degree: int,
    include_sun_third_body: bool,
    epoch_anchor_et: float,
) -> GridResult:
    """Single-sol α=0 propagation + fluence-aggregate sweep at one (M0, epoch).

    Returns a :class:`GridResult` row. Raises on propagation / windowing
    failure (caller handles).
    """
    t0 = time.perf_counter()

    epoch_et = utc_to_et(epoch_utc)
    season_frac = (epoch_et - epoch_anchor_et) / MARS_SIDEREAL_YEAR_S
    season_frac = season_frac - math.floor(season_frac)  # wrap to [0, 1)

    r_sun_au = _mars_sun_distance_au(epoch_et)
    sub_sol_lat_deg = _sub_solar_lat_iau_mars_deg(epoch_et)

    # Sun-sync inclination is determined by a_km via sun_sync_inclination_rad
    # inside initial_state_j2000; argp is pinned to 0 (e=0).
    state0 = initial_state_j2000(
        a_km=a_km,
        ltan_h=ltan_h,
        M0_rad=math.radians(m0_deg),
        epoch_et=epoch_et,
    )

    sail = make_canonical_sail(sigma_kg_per_m2)

    third_bodies = [sun_third_body()] if include_sun_third_body else []
    duration_s = float(SECONDS_PER_SOLAR_SOL_S)
    t_eval = np.arange(0.0, duration_s + 0.1, t_eval_cadence_s)
    if t_eval[-1] < duration_s:
        t_eval = np.append(t_eval, duration_s)

    result = propagate(
        state0_km_kmps=state0,
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et,
        zonal_degree=zonal_degree,
        gravity_degree=0,
        third_bodies=third_bodies,
        solar_sail=sail,
        sail_normal=sun_pointing(),
        altitude_floor=AltitudeFloor.at_km(altitude_floor_km, label="altitude_floor"),
        options=PropagationOptions.fast(),
        t_eval_s=t_eval,
    )

    # Pre-filter window count: call find_delivery_windows once with α_max
    # filter and once without, count both. (Cheap — same propagation result.)
    windows_post = find_delivery_windows(
        result,
        target_lat_deg, target_lon_deg,
        target_elevation_min_deg=elev_min_deg,
        bisector_cos_alpha_min=bisector_cos_alpha_min,
        require_sail_sunlit=True,
        require_sail_above_horizon=True,
        require_bisector_feasible=True,
        sail=sail,
        atmospheric_transmission=atmospheric_transmission,
        alpha_max_rad_s2=alpha_max_rad_s2,
    )
    windows_pre = find_delivery_windows(
        result,
        target_lat_deg, target_lon_deg,
        target_elevation_min_deg=elev_min_deg,
        bisector_cos_alpha_min=bisector_cos_alpha_min,
        require_sail_sunlit=True,
        require_sail_above_horizon=True,
        require_bisector_feasible=True,
        sail=sail,
        atmospheric_transmission=atmospheric_transmission,
        alpha_max_rad_s2=None,   # no α_max filter
    )

    n_w = len(windows_post)
    n_w_pre = len(windows_pre)
    if n_w == 0:
        total_fluence = 0.0
        peak_irr = 0.0
        mean_dur = 0.0
        sum_dur = 0.0
        peak_fp_km = 0.0
        peak_alpha = 0.0
    else:
        total_fluence = float(sum(
            w.fluence_J_per_m2 or 0.0 for w in windows_post
        ))
        peak_irr = float(max(
            (w.peak_irradiance_W_per_m2 or 0.0) for w in windows_post
        ))
        durations = [float(w.duration_s) for w in windows_post]
        mean_dur = float(np.mean(durations))
        sum_dur = float(np.sum(durations))
        peak_fp_km = float(max(
            (w.peak_footprint_semi_major_km or 0.0) for w in windows_post
        ))
        peak_alpha = float(max(
            (w.peak_alpha_demand_rad_s2 or 0.0) for w in windows_post
        ))

    return GridResult(
        m0_deg=float(m0_deg),
        epoch_utc=epoch_utc,
        epoch_et=float(epoch_et),
        season_frac_of_mars_year=float(season_frac),
        r_mars_sun_au=r_sun_au,
        sub_solar_lat_iau_mars_deg=sub_sol_lat_deg,
        n_windows=n_w,
        n_windows_pre_filter=n_w_pre,
        total_fluence_J_per_m2=total_fluence,
        peak_irradiance_W_per_m2=peak_irr,
        mean_window_duration_s=mean_dur,
        sum_window_duration_s=sum_dur,
        peak_footprint_semi_major_km=peak_fp_km,
        peak_alpha_demand_max_rad_s2=peak_alpha,
        wall_s=time.perf_counter() - t0,
    )


def _build_grids(
    m0_count: int,
    season_count: int,
    epoch_anchor_et: float,
    m0_min_deg: float = 0.0,
    m0_max_deg: float = 360.0,
) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
    """Return (m0_grid_deg, season_grid as list of (et, utc))."""
    if m0_count <= 1:
        m0_grid = np.array([m0_min_deg], dtype=float)
    else:
        m0_grid = np.linspace(m0_min_deg, m0_max_deg, m0_count + 1)[:-1]
    if season_count <= 1:
        season_ets = [epoch_anchor_et]
    else:
        season_ets = [
            epoch_anchor_et + k * MARS_SIDEREAL_YEAR_S / season_count
            for k in range(season_count)
        ]
    season_grid = [(float(et), spice.et2utc(float(et), "ISOC", 3))
                   for et in season_ets]
    return m0_grid, season_grid


def _eval_one_grid_point(args: dict) -> GridResult:
    """Worker entrypoint. Pickleable single-arg signature for CloudpickleMap."""
    return evaluate_grid_point(**args)


def render_heatmap(
    results: List[GridResult],
    m0_grid: np.ndarray,
    season_grid: List[Tuple[float, str]],
    out_png: Path,
) -> None:
    """Render a 2-panel heatmap PNG (fluence + n_windows over M0 × season)."""
    n_m0 = len(m0_grid)
    n_season = len(season_grid)
    fluence = np.zeros((n_season, n_m0), dtype=float)
    n_w = np.zeros((n_season, n_m0), dtype=int)
    for r in results:
        i_m0 = int(np.argmin(np.abs(m0_grid - r.m0_deg)))
        i_season = int(np.argmin(np.abs(
            np.array([s[0] for s in season_grid]) - r.epoch_et
        )))
        fluence[i_season, i_m0] = r.total_fluence_J_per_m2
        n_w[i_season, i_m0] = r.n_windows

    season_frac = np.array([
        ((s[0] - season_grid[0][0]) / MARS_SIDEREAL_YEAR_S) % 1.0
        for s in season_grid
    ])

    fig, (ax_flu, ax_nw) = plt.subplots(
        nrows=2, ncols=1, figsize=(13.0, 9.0),
        gridspec_kw=dict(hspace=0.30, left=0.08, right=0.94,
                         top=0.93, bottom=0.07),
    )

    # Build pcolormesh extents (cell-centred → cell-edge boundaries)
    if n_m0 > 1:
        dm0 = float(m0_grid[1] - m0_grid[0])
        m0_edges = np.concatenate([m0_grid - 0.5 * dm0,
                                   [m0_grid[-1] + 0.5 * dm0]])
    else:
        m0_edges = np.array([m0_grid[0] - 5.0, m0_grid[0] + 5.0])
    if n_season > 1:
        dsf = (1.0 / n_season)
        season_edges = np.concatenate([season_frac - 0.5 * dsf,
                                       [season_frac[-1] + 0.5 * dsf]])
    else:
        season_edges = np.array([-0.05, 0.05])

    pcm_flu = ax_flu.pcolormesh(m0_edges, season_edges, fluence,
                                shading="flat", cmap="viridis")
    cbar_flu = fig.colorbar(pcm_flu, ax=ax_flu, pad=0.02)
    cbar_flu.set_label("total fluence (J/m²)")
    ax_flu.set_xlabel("initial M0 (deg)")
    ax_flu.set_ylabel("season fraction past anchor (0 = anchor, 0.5 ≈ aphelion)")
    ax_flu.set_title(
        "Single-sol delivered fluence over (M0, season) at refined "
        "(a*, i*) — α=0 cruise + bisector during open windows",
        fontsize=11,
    )

    # Mark row-max M0 per season
    for i, sf in enumerate(season_frac):
        j_best = int(np.argmax(fluence[i, :]))
        ax_flu.plot(m0_grid[j_best], sf, marker="o",
                    mfc="red", mec="white", ms=7, lw=0)
    # Mark global max
    i_best, j_best = np.unravel_index(int(np.argmax(fluence)), fluence.shape)
    ax_flu.plot(m0_grid[j_best], season_frac[i_best],
                marker="*", mfc="yellow", mec="black", ms=18, lw=0,
                label=f"global max: M0={m0_grid[j_best]:.1f}°, "
                      f"season_frac={season_frac[i_best]:.3f}, "
                      f"fluence={fluence[i_best, j_best]:.2f} J/m²")
    ax_flu.legend(loc="upper right", fontsize=8, framealpha=0.85)

    pcm_nw = ax_nw.pcolormesh(m0_edges, season_edges, n_w,
                              shading="flat", cmap="plasma",
                              vmin=0, vmax=max(int(n_w.max()), 1))
    cbar_nw = fig.colorbar(pcm_nw, ax=ax_nw, pad=0.02)
    cbar_nw.set_label("n_windows (post α_max filter)")
    ax_nw.set_xlabel("initial M0 (deg)")
    ax_nw.set_ylabel("season fraction past anchor")
    ax_nw.set_title("delivery-window count over the same grid", fontsize=11)

    fig.suptitle(
        f"Longitudinal phasing × season fluence sweep — refined "
        f"sun-sync (a* = const, i* = sun-sync(a*))",
        fontsize=12, y=0.99,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_csv(results: List[GridResult], out_csv: Path) -> None:
    if not results:
        raise ValueError("no results to write")
    fieldnames = list(asdict(results[0]).keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--m0-grid", type=int, default=36,
                   help="number of M0 samples on [0°, 360°) (default: 36 = 10°)")
    p.add_argument("--m0-min", type=float, default=0.0,
                   help="M0 lower bound, deg (default: 0)")
    p.add_argument("--m0-max", type=float, default=360.0,
                   help="M0 upper bound, deg (default: 360, exclusive)")
    p.add_argument("--season-grid", type=int, default=12,
                   help="number of season epochs across one Mars year (default: 12)")
    p.add_argument("--epoch-anchor", type=str, default=DEFAULT_EPOCH_ANCHOR_UTC,
                   help=f"season-grid anchor epoch UTC (default: {DEFAULT_EPOCH_ANCHOR_UTC} = perihelion 2028)")
    p.add_argument("--a-km", type=float, default=DEFAULT_A_KM,
                   help=f"semi-major axis km (default: {DEFAULT_A_KM} = refined a*)")
    p.add_argument("--ltan-h", type=float, default=DEFAULT_LTAN_H,
                   help=f"local time of ascending node hours (default: {DEFAULT_LTAN_H})")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_KG_PER_M2,
                   help=f"sail areal density kg/m² (default: {DEFAULT_SIGMA_KG_PER_M2})")
    p.add_argument("--target-lat", type=float, default=DEFAULT_TARGET_LAT_DEG,
                   help=f"target latitude deg (default: {DEFAULT_TARGET_LAT_DEG})")
    p.add_argument("--target-lon", type=float, default=DEFAULT_TARGET_LON_DEG,
                   help=f"target east-positive longitude deg (default: {DEFAULT_TARGET_LON_DEG})")
    p.add_argument("--alpha-max-deg-per-s2", type=float,
                   default=DEFAULT_ALPHA_MAX_DEG_PER_S2,
                   help=f"α_max budget deg/s² (default: {DEFAULT_ALPHA_MAX_DEG_PER_S2})")
    p.add_argument("--altitude-floor-km", type=float,
                   default=DEFAULT_ALTITUDE_FLOOR_KM,
                   help=f"altitude floor termination km (default: {DEFAULT_ALTITUDE_FLOOR_KM})")
    p.add_argument("--elev-min-deg", type=float, default=DEFAULT_ELEV_MIN_DEG,
                   help=f"target elevation gate deg (default: {DEFAULT_ELEV_MIN_DEG})")
    p.add_argument("--bisector-cos-alpha-min", type=float,
                   default=DEFAULT_BISECTOR_COS_ALPHA_MIN,
                   help=f"bisector half-angle gate cos (default: {DEFAULT_BISECTOR_COS_ALPHA_MIN})")
    p.add_argument("--t-eval-cadence-s", type=float,
                   default=DEFAULT_T_EVAL_CADENCE_S,
                   help=f"propagation sampling cadence s (default: {DEFAULT_T_EVAL_CADENCE_S})")
    p.add_argument("--zonal-degree", type=int, default=2, choices=[0, 2],
                   help="0 = pure Kepler; 2 = J2-only (default: 2)")
    p.add_argument("--no-sun-third-body", action="store_true",
                   help="disable Sun third-body (default: included)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel workers via CloudpickleMap (default: 1 = serial)")
    p.add_argument("--output-csv", type=str, default=None,
                   help="output CSV path (default: simulation_outputs/{ts}_phasing_sweep.csv)")
    p.add_argument("--output-png", type=str, default=None,
                   help="output PNG path (default: same as CSV with .png)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sim_out = repo_root / "simulation_outputs"
    sim_out.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M")
    out_csv = (Path(args.output_csv) if args.output_csv
               else sim_out / f"{ts}_phasing_sweep.csv")
    out_png = (Path(args.output_png) if args.output_png
               else out_csv.with_suffix(".png"))

    if args.workers > 1:
        # Fork only after SPICE setup so workers inherit the loaded kernel pool.
        configure_multiprocessing_for_spice()
    load_kernels()

    epoch_anchor_et = utc_to_et(args.epoch_anchor)
    m0_grid, season_grid = _build_grids(
        args.m0_grid, args.season_grid,
        epoch_anchor_et, args.m0_min, args.m0_max,
    )

    alpha_max_rad_s2 = math.radians(args.alpha_max_deg_per_s2)

    print("=" * 78)
    print(f"longitudinal phasing × season fluence sweep")
    print("=" * 78)
    print(f"  M0 grid          = {len(m0_grid)} samples on [{args.m0_min}, "
          f"{args.m0_max}) deg")
    print(f"  season grid      = {len(season_grid)} epochs across 1 Mars year "
          f"from {args.epoch_anchor}")
    print(f"  total grid pts   = {len(m0_grid) * len(season_grid)}")
    print(f"  a_km             = {args.a_km}")
    print(f"  ltan_h           = {args.ltan_h}")
    print(f"  σ                = {args.sigma} kg/m²")
    print(f"  target           = ({args.target_lat}°N, {args.target_lon}°E)")
    print(f"  α_max            = {args.alpha_max_deg_per_s2} deg/s² "
          f"({alpha_max_rad_s2:.3e} rad/s²)")
    print(f"  altitude_floor   = {args.altitude_floor_km} km")
    print(f"  elev_min         = {args.elev_min_deg} deg")
    print(f"  cadence          = {args.t_eval_cadence_s} s")
    print(f"  zonal_degree     = {args.zonal_degree}")
    print(f"  sun_third_body   = {not args.no_sun_third_body}")
    print(f"  optical_path     = vacuum (transmission={VACUUM_TRANSMISSION})")
    print(f"  workers          = {args.workers}")
    print(f"  output csv       = {out_csv}")
    print(f"  output png       = {out_png}")
    print("=" * 78)

    eval_kwargs_common = dict(
        a_km=float(args.a_km),
        ltan_h=float(args.ltan_h),
        sigma_kg_per_m2=float(args.sigma),
        target_lat_deg=float(args.target_lat),
        target_lon_deg=float(args.target_lon),
        alpha_max_rad_s2=float(alpha_max_rad_s2),
        elev_min_deg=float(args.elev_min_deg),
        bisector_cos_alpha_min=float(args.bisector_cos_alpha_min),
        altitude_floor_km=float(args.altitude_floor_km),
        atmospheric_transmission=VACUUM_TRANSMISSION,
        t_eval_cadence_s=float(args.t_eval_cadence_s),
        zonal_degree=int(args.zonal_degree),
        include_sun_third_body=not bool(args.no_sun_third_body),
        epoch_anchor_et=float(epoch_anchor_et),
    )

    grid_pts: List[dict] = []
    for et, utc in season_grid:
        for m0 in m0_grid:
            grid_pts.append({"m0_deg": float(m0), "epoch_utc": utc,
                             **eval_kwargs_common})

    t_run_0 = time.perf_counter()
    if args.workers > 1:
        with CloudpickleMap(n_workers=int(args.workers)) as cp_map:
            results: List[GridResult] = list(
                cp_map(_eval_one_grid_point, grid_pts)
            )
    else:
        results = []
        for k, gp in enumerate(grid_pts):
            r = _eval_one_grid_point(gp)
            results.append(r)
            print(
                f"  [{k+1:>4d}/{len(grid_pts):<4d}] M0={r.m0_deg:6.2f}° "
                f"season_frac={r.season_frac_of_mars_year:.3f} "
                f"r={r.r_mars_sun_au:.4f}AU  "
                f"n_w={r.n_windows}  flu={r.total_fluence_J_per_m2:7.4f}  "
                f"wall={r.wall_s:.2f}s"
            )
    wall_total = time.perf_counter() - t_run_0
    print()
    print(f"  total wall = {wall_total:.1f} s = {wall_total/60:.2f} min")

    write_csv(results, out_csv)
    print(f"  wrote csv  → {out_csv}")
    render_heatmap(results, m0_grid, season_grid, out_png)
    print(f"  wrote png  → {out_png}")

    # Summary
    fluences = np.array([r.total_fluence_J_per_m2 for r in results])
    best_idx = int(np.argmax(fluences))
    best = results[best_idx]
    print()
    print(f"  global max fluence = {best.total_fluence_J_per_m2:.4f} J/m²")
    print(f"  at M0              = {best.m0_deg:.2f}°")
    print(f"  at epoch           = {best.epoch_utc}")
    print(f"  season_frac        = {best.season_frac_of_mars_year:.4f} "
          f"(0 = anchor, 0.5 ≈ aphelion)")
    print(f"  r_mars_sun         = {best.r_mars_sun_au:.4f} AU")
    print(f"  n_windows          = {best.n_windows} "
          f"(pre-α_max filter: {best.n_windows_pre_filter})")
    print(f"  peak_irradiance    = {best.peak_irradiance_W_per_m2:.2e} W/m²")
    print(f"  peak_footprint_km  = {best.peak_footprint_semi_major_km:.4f} "
          f"(semi-major)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
