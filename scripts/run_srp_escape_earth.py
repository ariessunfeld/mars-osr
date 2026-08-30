"""Run SRP-driven Earth escape scenarios with optional atmospheric drag.

The Earth counterpart of ``scripts/run_srp_escape.py`` (Mars). An SRP solar sail
spirals OUT of a low Earth orbit, in forward time, to the Earth Hill sphere
(~1.496e6 km) -- the handoff boundary for a heliocentric solver. Unlike Mars,
the LEO start makes atmospheric DRAG first-order (a ~55 m^2/kg sail at sigma=18),
and the Moon is a first-order third body (it orbits at ~0.26 R_Hill, inside the
escape ceiling).

The default model uses a polar start, Sun and Moon third bodies, Earth
J2 (IAU_EARTH), McInnes SRP, Harris-Priester atmosphere (M&G Table 3.8 +
NRLMSIS-2.1-calibrated >1000 km tail), slew limits (alpha_max=0.003 deg/s^2,
omega_max=0.3 deg/s), and the DRAG-AWARE dE/dt controller (merit =
(a_SRP + a_drag).v_hat).

Comparison modes:
  default      drag dynamics on, drag-aware controller
  --srp_only   drag dynamics on, SRP-only controller
  --no_drag    drag dynamics off

The default resolution uses max_step_s=20 s and
max_step_true_anomaly_deg=0.25 deg to maintain convergence of the
attitude-tracker ODE as `a` grows. Verify convergence by repeating a calculation
with both limits halved.

Usage:
    python scripts/run_srp_escape_earth.py                 # 800 km, drag-aware
    python scripts/run_srp_escape_earth.py 18 --altitude_km 1000 --tag a1000
    python scripts/run_srp_escape_earth.py --srp_only --tag a800_srponly
    python scripts/run_srp_escape_earth.py --max_step_s 10 --max_step_true_anomaly_deg 0.125 --tag a800_fine

Outputs: per-step CSV (orbit + attitude + drag) + summary (+ Hill handoff) under
simulation_outputs/ (gitignored).
Whether a sail escapes depends on the selected altitude and model parameters.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import time
from pathlib import Path

import numpy as np
import spiceypy as spice

from reflectors.atmosphere import HarrisPriester
from reflectors.attitude_control import AttitudeLimits, alpha_command
from reflectors.central_body import earth_central_body
from reflectors.drag import drag_acceleration_from_state, make_drag_force_fn
from reflectors.elements import classical_elements
from reflectors.ephemeris import utc_to_et
from reflectors.escape import propagate_escape
from reflectors.escape_dedot import DEdotParams, DragMeritContext, dedot_steer
from reflectors.gauss import osculating_elements
from reflectors.kernels import load_kernels
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail
from reflectors.shadow import shadow_factor
from reflectors.solar_constants import solar_flux_at
from reflectors.srp import mcinnes_srp_acceleration
from reflectors.surface import earth_equatorial_radius_km
from reflectors.sun_sync import sun_sync_inclination_rad
from reflectors.earth_constants import EARTH_J2, EARTH_SIDEREAL_YEAR_S
from reflectors.third_body import moon_third_body, sun_third_body

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DAY_S = 86400.0
TARGET_CSV_ROWS = 30000
SUN_NAIF_ID = 10
EARTH_NAIF_ID = 399
EPOCH = "2028-01-01T00:00:00"


def earth_polar_state(epoch_et, altitude_km, mu, ltan_hours=None):
    """Circular near-polar (i=90 deg) orbit; LTAN selects the plane vs the Sun.

    ``ltan_hours=None`` selects the noon-midnight construction,
    ``h_hat = earth_pole x sun_hat``
    -> i = 90 deg wrt the Earth equator, Sun IN the orbital plane initially
    (Earth analogue of the Mars polar recipe; decouples
    the J2-SRP secular resonance).

    ``ltan_hours`` given: a polar plane at the requested Local Time of the
    Ascending Node. The ascending node is the Sun's equatorial projection
    (subsolar meridian, = LTAN 12 h) rotated about the Earth pole by
    ``(LTAN - 12) * 15 deg``; ``h_hat = pole x n_asc`` keeps i = 90 deg, and the
    start position is the ascending node. LTAN 12 == noon-midnight (Sun in
    plane); LTAN 6 / 18 == dawn-dusk (Sun PERPENDICULAR to the plane), which
    orients the plane to CONTAIN Earth's heliocentric velocity -> a prograde
    Hill-exit becomes geometrically possible. The two dawn-dusk senses (6 vs
    18) are the same terminator plane with opposite circulation.
    ``ltan_hours=12`` uses the Sun's equatorial projection and node start,
    whereas ``None`` uses the raw Sun direction and subsolar start.
    """
    R = spice.pxform("IAU_EARTH", "J2000", epoch_et)
    pole = R @ np.array([0.0, 0.0, 1.0])
    pole /= np.linalg.norm(pole)
    sun_state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", str(EARTH_NAIF_ID))
    sun_hat = np.asarray(sun_state[:3]) / np.linalg.norm(sun_state[:3])
    a_km = earth_equatorial_radius_km() + altitude_km

    if ltan_hours is None:
        # Noon-midnight construction with the Sun initially in the plane.
        h_hat = np.cross(pole, sun_hat)
        h_hat /= np.linalg.norm(h_hat)
        r_hat = sun_hat - np.dot(sun_hat, h_hat) * h_hat
        r_hat /= np.linalg.norm(r_hat)
        v_hat = np.cross(h_hat, r_hat)
        return np.concatenate([a_km * r_hat, math.sqrt(mu / a_km) * v_hat])

    # General LTAN: rotate the subsolar meridian about the pole to place the
    # ascending node. sun_eq = Sun's projection into the equatorial plane.
    sun_eq = sun_hat - np.dot(sun_hat, pole) * pole
    sun_eq /= np.linalg.norm(sun_eq)
    ang = math.radians((ltan_hours - 12.0) * 15.0)
    # Rodrigues rotation of sun_eq about pole by ang (pole . sun_eq = 0 -> the
    # (1-cos) term drops): n_asc = sun_eq cos + (pole x sun_eq) sin.
    n_asc = sun_eq * math.cos(ang) + np.cross(pole, sun_eq) * math.sin(ang)
    n_asc /= np.linalg.norm(n_asc)
    h_hat = np.cross(pole, n_asc)   # i = 90 deg (h in the equatorial plane)
    h_hat /= np.linalg.norm(h_hat)
    r_hat = n_asc                   # start at the ascending node
    v_hat = np.cross(h_hat, r_hat)
    v_hat /= np.linalg.norm(v_hat)
    return np.concatenate([a_km * r_hat, math.sqrt(mu / a_km) * v_hat])


def earth_sun_sync_ltan_state(epoch_et, altitude_km, ltan_h, mu):
    """Circular TRUE sun-synchronous Earth orbit at a given LTAN.

    True-sun-sync analogue of ``earth_polar_state``'s i=90 dawn-dusk ``--ltan``
    branch: inclination = Brouwer sun-sync inclination for Earth at this altitude
    (~98.6 deg at 800 km; ``sun_sync_inclination_rad`` with Earth mu / R_eq / J2
    and the Earth sidereal year), node at the requested LTAN (same convention as
    the i=90 branch -- ascending node = the Sun's equatorial projection rotated
    about the Earth pole by ``(LTAN-12)*15 deg``). e=0, start at the ascending
    node. Inclination is taken about the TRUE Earth pole (IAU_EARTH, matching
    ``earth_polar_state``); the orbit normal for inclination i with that node is
    ``h = cos(i) pole - sin(i) (pole x n_asc)`` (Vallado classical->Cartesian).
    All in Earth-centred J2000 (the frame ``propagate_escape`` uses). Returns
    ``(state6, i_rad)``.
    """
    R = spice.pxform("IAU_EARTH", "J2000", epoch_et)
    pole = R @ np.array([0.0, 0.0, 1.0])
    pole /= np.linalg.norm(pole)
    sun_state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", str(EARTH_NAIF_ID))
    sun_hat = np.asarray(sun_state[:3]) / np.linalg.norm(sun_state[:3])
    a_km = earth_equatorial_radius_km() + altitude_km
    i_rad = sun_sync_inclination_rad(
        a_km, mu_km3_s2=mu, ref_radius_km=earth_equatorial_radius_km(),
        J2=EARTH_J2, target_period_s=EARTH_SIDEREAL_YEAR_S)
    # ascending node at the requested LTAN (Sun's equatorial projection rotated)
    sun_eq = sun_hat - np.dot(sun_hat, pole) * pole
    sun_eq /= np.linalg.norm(sun_eq)
    ang = math.radians((ltan_h - 12.0) * 15.0)
    n_asc = sun_eq * math.cos(ang) + np.cross(pole, sun_eq) * math.sin(ang)
    n_asc /= np.linalg.norm(n_asc)
    # orbit normal for inclination i_rad with this ascending node
    h_hat = math.cos(i_rad) * pole - math.sin(i_rad) * np.cross(pole, n_asc)
    h_hat /= np.linalg.norm(h_hat)
    r_hat = n_asc
    v_hat = np.cross(h_hat, r_hat)
    v_hat /= np.linalg.norm(v_hat)
    return np.concatenate([a_km * r_hat, math.sqrt(mu / a_km) * v_hat]), i_rad


ap = argparse.ArgumentParser()
ap.add_argument("sigma_g", type=float, nargs="?", default=18.0,
                help="sail loading, g/m^2 (default 18 = canonical design sail)")
ap.add_argument("--altitude_km", type=float, default=800.0,
                help="circular start altitude (km). Sweep {800,1000,2000}.")
ap.add_argument("--ltan", type=float, default=None,
                help="Local Time of Ascending Node (hours) of the initial polar "
                     "plane. None (default) = catalog noon-midnight construction "
                     "(Sun in plane, h=pole x sun). 12 = noon-midnight; 6/18 = "
                     "dawn-dusk (Sun perpendicular to plane -> the plane contains "
                     "Earth's heliocentric velocity -> a prograde Hill-exit becomes "
                     "possible).")
ap.add_argument("--sun-sync", dest="sun_sync", action="store_true",
                help="Use the TRUE Earth sun-sync inclination (~98.6 deg at 800 km) "
                     "instead of i=90 polar, at the --ltan node (requires --ltan). "
                     "The reflector-orbit analogue.")
ap.add_argument("--srp_only", action="store_true",
                help="use an SRP-only, drag-blind controller while retaining "
                     "drag in the dynamics")
ap.add_argument("--no_drag", action="store_true",
                help="disable atmospheric drag entirely (drag-free reference).")
ap.add_argument("--C_d", type=float, default=2.2,
                help="drag coefficient (free-molecular convex body; M&G p.84-85)")
ap.add_argument("--bulge_exponent", type=float, default=6.0,
                help="Harris-Priester diurnal-bulge exponent n (6 polar, 2 low-inc)")
ap.add_argument("--alpha_max_deg_s2", type=float, default=0.003)
ap.add_argument("--omega_max_deg_s", type=float, default=0.3)
ap.add_argument("--max_cone_deg", type=float, default=80.0)
ap.add_argument("--span_days", type=float, default=1500.0,
                help="propagation span cap in days (Earth's deeper well -> long)")
ap.add_argument("--altitude_floor_km", type=float, default=150.0,
                help="re-entry/decay floor above R_eq (km); below it the drag + "
                     "heating model is invalid -> a failed (decayed) escape")
ap.add_argument("--gravity_degree", type=int, default=2, help="2 = J2 (Earth zonal)")
ap.add_argument("--steps_per_orbit", type=int, default=200)
ap.add_argument("--max_step_true_anomaly_deg", type=float, default=0.25,
                help="periapsis Sundman cap (deg); halve for the convergence twin")
ap.add_argument("--max_step_s", type=float, default=20.0,
                help="absolute attitude-resolution step ceiling (s); "
                     "halve (10) for the convergence twin")
ap.add_argument("--epoch", type=str, default=EPOCH,
                help="UTC start epoch of the LEO escape (default 2028-01-01). "
                     "Set to place the escaped (Hill) state at a target date for "
                     "an interplanetary handoff, e.g. 2010-05-26 -> Hill ~2011-10-10.")
ap.add_argument("--energy_gated", action=argparse.BooleanOptionalAction, default=True,
                help="define escape as E>=0 and |r|>=Hill; a bound Hill-radius "
                     "graze is not an escape. --no-energy_gated selects "
                     "radius-only termination.")
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()

sigma_g = args.sigma_g
loading_kg_per_m2 = sigma_g / 1000.0
drag_aware = not args.srp_only
drag_on = not args.no_drag
ctrl = "dragaware" if (drag_aware and drag_on) else ("srponly" if drag_on else "nodrag")
suffix = f"_{args.tag}" if args.tag else ""
if args.ltan is not None:
    _ltbase = f"_ssoLTAN{args.ltan:g}" if args.sun_sync else f"_ltan{args.ltan:g}"
    ltan_tag = _ltbase.replace(".", "p")
else:
    ltan_tag = ""
tag = f"earth_sigma{int(round(sigma_g))}_a{int(round(args.altitude_km))}_{ctrl}{ltan_tag}{suffix}"

load_kernels()
epoch_et = utc_to_et(args.epoch)
earth = earth_central_body()
mu = earth.mu_km3_s2
R_EQ = earth.equatorial_radius_km

if args.sun_sync:
    if args.ltan is None:
        ap.error("--sun-sync requires --ltan")
    state0, _sso_i_rad = earth_sun_sync_ltan_state(
        epoch_et, args.altitude_km, args.ltan, mu)
    print(f"[init] Earth sun-sync LTAN={args.ltan:g}h: i_sso(Earth equator)="
          f"{math.degrees(_sso_i_rad):.3f} deg", flush=True)
else:
    state0 = earth_polar_state(epoch_et, args.altitude_km, mu, ltan_hours=args.ltan)
sail = make_canonical_sail(loading_kg_per_m2)
limits = AttitudeLimits(
    alpha_max_rad_s2=math.radians(args.alpha_max_deg_s2),
    omega_max_rad_s=math.radians(args.omega_max_deg_s),
)

# Density model + drag wiring. Drag DYNAMICS (the RHS hook) and drag AWARENESS
# (the controller merit) are independent: default is both on; --srp_only keeps
# the dynamics but blinds the controller; --no_drag removes drag entirely.
density_model = HarrisPriester(bulge_exponent=args.bulge_exponent)
drag_force_fn = (
    make_drag_force_fn(sail, density_model, central_body=earth, C_d=args.C_d)
    if drag_on else None
)
drag_ctx = (
    DragMeritContext(density_model=density_model, central_body=earth, C_d=args.C_d)
    if (drag_aware and drag_on) else None
)

naive_params = DEdotParams(
    max_cone_rad=math.radians(args.max_cone_deg), rp_warn_km=None, mu_km3_s2=mu,
)


def steering(r, v, s_hat, p_eff, sail_, current_n_hat, et=None):
    """Naive dE/dt-max steering; drag-aware when drag_ctx is set (single toggle)."""
    return dedot_steer(
        r, v, s_hat, p_eff, sail_, current_n_hat=current_n_hat,
        params=naive_params, drag=drag_ctx, et=et,
    ).n_star_j2000


qlaw_shell = QLawParams(a_target_km=earth.hill_radius_km, rp_min_km=R_EQ + 100.0)

print(f"[{tag}] EARTH escape  sigma={sigma_g:.1f} g/m^2  area={sail.area_m2:.0f} m^2 "
      f"mass={sail.mass_kg:.2f} kg  altitude={args.altitude_km:.0f} km", flush=True)
print(f"[{tag}] controller: naive dE/dt-max  drag_aware={drag_aware}  "
      f"drag_dynamics={drag_on}  C_d={args.C_d}  bulge_n={args.bulge_exponent}",
      flush=True)
print(f"[{tag}] bodies: Earth(J{args.gravity_degree}) + Sun + Moon; "
      f"slew alpha_max={args.alpha_max_deg_s2} deg/s^2 omega_max={args.omega_max_deg_s} "
      f"deg/s (strict bound)", flush=True)
print(f"[{tag}] resolution: max_step_s={args.max_step_s} s, "
      f"max_dnu={args.max_step_true_anomaly_deg} deg, Hill={earth.hill_radius_km:.4e} km, "
      f"floor={R_EQ + args.altitude_floor_km:.1f} km", flush=True)

from reflectors.termination import AltitudeFloor

el0_disp = osculating_elements(state0[:3], state0[3:], mu)
print(f"[{tag}] start: a={el0_disp.a_km:.1f} e={el0_disp.e:.4f} "
      f"i={math.degrees(el0_disp.inclination_rad):.2f} deg", flush=True)

# Self-verify the initial-plane orientation vs Earth's heliocentric velocity
# (the cruise-handoff axis). h_hat = r0 x v0 decomposed in Earth's helio triad
# (prograde = v_Earth ; oop = r_E x v_E ; inplane = prograde x oop ~ radial).
# |h.prograde| ~ 0  <=>  plane CONTAINS v_Earth  <=>  prograde Hill-exit possible.
_eE, _ = spice.spkezr("399", epoch_et, "J2000", "NONE", "SUN")
_rE = np.asarray(_eE[:3]); _vE = np.asarray(_eE[3:6])
_prog = _vE / np.linalg.norm(_vE)
_oop = np.cross(_rE, _vE); _oop /= np.linalg.norm(_oop)
_inpl = np.cross(_prog, _oop); _inpl /= np.linalg.norm(_inpl)
_h0 = np.cross(state0[:3], state0[3:]); _h0 /= np.linalg.norm(_h0)
print(f"[{tag}] initial plane (LTAN={args.ltan}): h.prograde={np.dot(_h0,_prog):+.3f} "
      f"h.inplane(radial)={np.dot(_h0,_inpl):+.3f} h.oop={np.dot(_h0,_oop):+.3f}  "
      f"(|h.prograde|~0 => plane contains v_Earth => prograde exit possible)",
      flush=True)

t_span = (0.0, args.span_days * DAY_S)
print(f"[{tag}] propagating ({args.span_days:.0f}-day cap) ...", flush=True)
_t_start = time.perf_counter()


def _progress(t_s, y, step):
    el = osculating_elements(y[:3], y[3:6], mu)
    E = 0.5 * np.dot(y[3:6], y[3:6]) - mu / np.linalg.norm(y[:3])
    wall = time.perf_counter() - _t_start
    print(f"[{tag}] step {step:8d}  day {t_s/DAY_S:8.2f}  a {el.a_km:11.1f}  "
          f"e {el.e:.4f}  rp {el.periapsis_km:10.1f}  ra {el.apoapsis_km:12.1f}  "
          f"E {E:+.5f}  [wall {wall/60.0:.1f} min]", flush=True)


t0 = time.perf_counter()
res = propagate_escape(
    state0, epoch_et, sail, qlaw_shell, limits, t_span,
    gravity_degree=args.gravity_degree,
    central_body=earth,
    third_bodies=(sun_third_body(), moon_third_body()),  # Sun + Moon (cislunar)
    steps_per_orbit=args.steps_per_orbit,
    max_step_true_anomaly_deg=args.max_step_true_anomaly_deg,
    max_step_s=args.max_step_s,
    energy_gated=args.energy_gated,
    altitude_floor=AltitudeFloor(altitude_km=args.altitude_floor_km,
                                 reference_radius_km=R_EQ),
    steering_fn=steering,
    drag_force_fn=drag_force_fn,
    progress_callback=_progress,
)
wall = time.perf_counter() - t0
mu = res.metadata["mu_central_km3_s2"]
print(f"[{tag}] done: {res.termination_reason}, {len(res.t_s)} steps, "
      f"wall {wall/60.0:.1f} min", flush=True)

# --- Per-step CSV ---------------------------------------------------------
n_steps = len(res.t_s)
stride = max(1, n_steps // TARGET_CSV_ROWS)
Path("simulation_outputs").mkdir(exist_ok=True)
csv_path = f"simulation_outputs/escape_{RUN_STAMP}_{tag}_per_step.csv"
header = [
    "day", "t_s", "et",
    "r_x_km", "r_y_km", "r_z_km", "v_x_kmps", "v_y_kmps", "v_z_kmps",
    "a_km", "e", "inc_deg", "raan_deg", "argp_deg", "nu_deg",
    "periapsis_km", "apoapsis_km", "r_mag_km", "alt_km", "energy_km2_s2",
    "n_x", "n_y", "n_z", "omega_mag_rad_s", "alpha_mag_rad_s2",
    "shadow", "srp_accel_mag_km_s2", "drag_accel_mag_km_s2",
    "n_des_x", "n_des_y", "n_des_z", "tracking_angle_deg",
]

with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    for i in range(0, n_steps, stride):
        t = float(res.t_s[i])
        et = epoch_et + t
        orbit = res.orbit_state_km_kmps[i]
        r = orbit[:3]
        v = orbit[3:]
        n = res.attitude_state[i, :3]
        omega = res.attitude_state[i, 3:]

        el = classical_elements(orbit, mu, et)
        r_mag = float(np.linalg.norm(r))
        v_mag = float(np.linalg.norm(v))
        energy = 0.5 * v_mag * v_mag - mu / r_mag

        sun_state, _ = spice.spkezr(str(SUN_NAIF_ID), et, "J2000", "NONE",
                                    str(EARTH_NAIF_ID))
        r_sun = np.asarray(sun_state[:3], dtype=float)
        sat_to_sun = r_sun - r
        r_helio = float(np.linalg.norm(sat_to_sun))
        s_hat = sat_to_sun / r_helio
        shadow = shadow_factor(r, et, EARTH_NAIF_ID, sun_position_j2000_km=r_sun,
                               central_radius_km=R_EQ)
        p_eff = solar_flux_at(r_helio) * shadow
        a_srp = mcinnes_srp_acceleration(n, s_hat, p_eff, sail)
        srp_mag = float(np.linalg.norm(a_srp))
        if drag_force_fn is not None:
            a_drag = drag_acceleration_from_state(r, v, et, n, sail, density_model,
                                                  central_body=earth, C_d=args.C_d)
            drag_mag = float(np.linalg.norm(a_drag))
        else:
            drag_mag = 0.0

        d = dedot_steer(r, v, s_hat, p_eff, sail, current_n_hat=n,
                        params=naive_params, drag=drag_ctx, et=et)
        n_des = d.n_star_j2000
        cos_track = max(-1.0, min(1.0, float(np.dot(n, n_des))))
        tracking_deg = math.degrees(math.acos(cos_track))
        alpha_cmd = alpha_command(n, omega, n_des, limits)

        writer.writerow([
            t / DAY_S, t, et,
            r[0], r[1], r[2], v[0], v[1], v[2],
            el.a_km, el.e, np.degrees(el.inclination_rad),
            np.degrees(el.raan_rad), np.degrees(el.argp_rad), np.degrees(el.nu_rad),
            el.periapsis_km, el.apoapsis_km, r_mag, r_mag - R_EQ, energy,
            n[0], n[1], n[2],
            float(np.linalg.norm(omega)), float(np.linalg.norm(alpha_cmd)),
            int(round(shadow)), srp_mag, drag_mag,
            n_des[0], n_des[1], n_des[2], tracking_deg,
        ])

print(f"[{tag}] per-step CSV ({(n_steps + stride - 1)//stride} rows, "
      f"stride {stride}) -> {csv_path}", flush=True)

# --- Summary -------------------------------------------------------------
el0 = classical_elements(res.orbit_state_km_kmps[0], mu, epoch_et)
elN = classical_elements(res.orbit_state_km_kmps[-1], mu, res.epoch_et)

e_ge0_day = None
for i in range(0, n_steps, 20):
    s = res.orbit_state_km_kmps[i]
    E = 0.5 * np.dot(s[3:], s[3:]) - mu / np.linalg.norm(s[:3])
    if E >= 0:
        e_ge0_day = res.t_s[i] / DAY_S
        break

omega_mags = np.linalg.norm(res.attitude_state[:, 3:], axis=1)
max_omega_frac = float(np.max(omega_mags)) / limits.omega_max_rad_s

# Closest physical approach (the true safety/decay metric). The osculating r_p of
# a near-parabolic end state is a virtual backward-extrapolation; min |r| is what
# the AltitudeFloor event keys on.
r_all_km = np.linalg.norm(res.orbit_state_km_kmps[:, :3], axis=1)
min_r_idx = int(np.argmin(r_all_km))
min_r_km = float(r_all_km[min_r_idx])
min_r_day = float(res.t_s[min_r_idx]) / DAY_S
floor_r_km = R_EQ + args.altitude_floor_km

summary_path = f"simulation_outputs/escape_{RUN_STAMP}_{tag}_summary.txt"
lines = [
    f"SRP EARTH-escape -- sigma={sigma_g:.1f} g/m^2  altitude={args.altitude_km:.0f} km",
    f"controller         : naive dE/dt-max  drag_aware={drag_aware}  drag_dynamics={drag_on}",
    f"epoch              : {args.epoch}",
    f"sail               : area {sail.area_m2:.0f} m^2, mass {sail.mass_kg:.2f} kg "
    f"(A/m {sail.area_m2/sail.mass_kg:.2f} m^2/kg)",
    f"drag               : C_d={args.C_d}, Harris-Priester (bulge n={args.bulge_exponent})"
    + ("" if drag_on else "  [DISABLED]"),
    f"attitude limits    : alpha_max {args.alpha_max_deg_s2:.4f} deg/s^2, "
    f"omega_max {args.omega_max_deg_s:.4f} deg/s",
    f"max |omega| / cap  : {max_omega_frac*100:.4f}%  (strict bound)",
    f"resolution         : max_step_s={args.max_step_s} s, "
    f"max_dnu={args.max_step_true_anomaly_deg} deg",
    f"escape definition  : {'E>=0 AND |r|>=Hill (energy-gated)' if args.energy_gated else 'radius-only (|r|>=Hill)'}",
    f"termination        : {res.termination_reason}",
    f"wall time          : {wall/60.0:.2f} min",
    f"integration steps  : {n_steps}",
    f"a    start / end   : {el0.a_km:.2f} / {elN.a_km:.2f} km",
    f"e    start / end   : {el0.e:.5f} / {elN.e:.5f}",
    f"r_p (osc) start/end: {el0.periapsis_km:.2f} / {elN.periapsis_km:.2f} km "
    f"(osculating a(1-e); VIRTUAL for a hyperbolic end state -- see min actual radius)",
    f"r_a  start / end   : {el0.apoapsis_km:.2f} / {elN.apoapsis_km:.2f} km",
    f"min actual radius  : {min_r_km:.2f} km at day {min_r_day:.1f}  "
    f"(closest approach; floor {floor_r_km:.2f}, margin {min_r_km - floor_r_km:+.1f} km)",
    f"first E>=0         : {('day %.1f' % e_ge0_day) if e_ge0_day is not None else 'never'}",
    f"escaped (Hill)     : {res.escaped}",
]
if res.termination_t_s is not None:
    t_esc = res.termination_t_s
    lines += [f"event time         : {t_esc/DAY_S:.1f} days = {t_esc/DAY_S/365.25:.2f} yr"]
if res.escaped:
    hr = res.termination_orbit_state_km_kmps[:3]
    hv = res.termination_orbit_state_km_kmps[3:]
    lines += [
        "",
        "=== Hill-sphere handoff state (J2000, Earth-centred) ===",
        f"  epoch_et : {res.termination_et:.6f}",
        f"  r_km     : [{hr[0]:.6f}, {hr[1]:.6f}, {hr[2]:.6f}]",
        f"  v_kmps   : [{hv[0]:.9f}, {hv[1]:.9f}, {hv[2]:.9f}]",
        f"  |r|      : {np.linalg.norm(hr):.3f} km",
        f"  |v|      : {np.linalg.norm(hv):.6f} km/s",
        "  (NOTE: Moon is inside the Hill sphere -- forward-validate the handoff "
        "for no recapture, separately.)",
    ]
else:
    lines += ["", "(did not reach the Hill sphere within the span)"]

summary = "\n".join(lines)
with open(summary_path, "w") as fh:
    fh.write(summary + "\n")
print(summary, flush=True)
print(f"[{tag}] summary -> {summary_path}", flush=True)
