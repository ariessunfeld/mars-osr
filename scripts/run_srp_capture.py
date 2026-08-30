"""Generate a Mars capture node via backward-ephemeris escape.

A capture is the time-reverse of an escape. This command runs the escape
propagator with the Sun/Mars ephemeris clocked backward
(``ephemeris_time_direction = -1``). The sail state integrates forward and
spirals outward, while reversing the ephemeris makes the time-reversed path a
physically consistent capture. The Hill-sphere endpoint is the capture node.

The default run starts from a polar 1500 km parking orbit at ``--park_epoch``.
The node epoch is the parking epoch minus the computed spiral duration.

For the handoff velocity, capture is time-reversed escape, so velocity flips.
The node is an outbound state. As a capture handoff it represents the sail
arriving inbound, so the interplanetary leg must deliver Mars-relative
velocity = -v_node. The summary + JSON report BOTH the outbound state and the
inbound (-v) capture-entry velocity, labeled.

The default resolution uses ``max_step_s=20`` and
``max_step_true_anomaly_deg=0.25`` to resolve the attitude dynamics as the orbit
expands. The slew limits are ``alpha_max=0.003 deg/s^2`` and
``omega_max=0.3 deg/s``.

The default configuration is sigma=18 g/m², polar, and 1500 km. Other sails,
planes, and altitudes are supported but are not guaranteed to reach the node.

Usage:
    python scripts/run_srp_capture.py                       # sigma=18 capture node
    python scripts/run_srp_capture.py 10 --tag sig10        # lighter sail
    python scripts/run_srp_capture.py 18 --max_step_s 40 --tag quick  # ~2x faster
    python scripts/run_srp_capture.py 18 --park_epoch 2030-01-01T00:00:00

Controllers: naive (default), blended, and guarded dE/dt-max.
Outputs: per-step CSV (orbit + attitude history) + summary + capture-node JSON
(node state, both velocities, epoch) under simulation_outputs/ (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import spiceypy as spice

from reflectors.attitude_control import AttitudeLimits, alpha_command
from reflectors.dynamics import mars_gm_km3_per_s2
from reflectors.elements import classical_elements
from reflectors.ephemeris import utc_to_et
from reflectors.escape import initial_circular_state, propagate_escape
from reflectors.escape_dedot import (
    BlendedParams,
    DEdotParams,
    blended_steer,
    dedot_steer,
)
from reflectors.gauss import osculating_elements
from reflectors.kernels import load_kernels
from reflectors.mars_constants import MARS_HILL_RADIUS_KM, SECONDS_PER_SOLAR_SOL_S
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail
from reflectors.shadow import shadow_factor
from reflectors.solar_constants import solar_flux_at
from reflectors.srp import mcinnes_srp_acceleration
from reflectors.surface import mars_equatorial_radius_km
from reflectors.termination import AltitudeFloor

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
R_EQ = 3396.19
# Parking-orbit (backward-run START) epoch. Default chosen so the node
# (= park_epoch - T, T ~ 470 sols at sigma=18) lands near 2013-05-31 -- the
# target "arrival at the node" ~600 d after 2011-10-10. The node
# epoch is reported; setting park_epoch = target_node + T pins a requested node
# epoch. Kernel coverage includes this interval with the kernel set fetched by
# scripts/fetch_kernels.py.
DEFAULT_PARK_EPOCH = "2014-09-13T00:00:00"
TARGET_CSV_ROWS = 30000
SUN_NAIF_ID = 10
MARS_NAIF_ID = 499


def polar_state(epoch_et, altitude_km, mu):
    """Circular orbit polar to the Mars equator, Sun in the plane at epoch.

    Construction: h_hat = mars_pole x sun_hat -> i = 90 deg
    wrt the Mars equator, with the Sun lying in the orbital plane initially.
    """
    R = spice.pxform("IAU_MARS", "J2000", epoch_et)
    mars_pole = R @ np.array([0.0, 0.0, 1.0])
    mars_pole /= np.linalg.norm(mars_pole)
    sun_state, _ = spice.spkezr("SUN", epoch_et, "J2000", "NONE", "MARS")
    sun_hat = np.asarray(sun_state[:3]) / np.linalg.norm(sun_state[:3])
    h_hat = np.cross(mars_pole, sun_hat)
    h_hat /= np.linalg.norm(h_hat)
    r_hat = sun_hat - np.dot(sun_hat, h_hat) * h_hat
    r_hat /= np.linalg.norm(r_hat)
    a_km = mars_equatorial_radius_km() + altitude_km
    v_hat = np.cross(h_hat, r_hat)
    return np.concatenate([a_km * r_hat, math.sqrt(mu / a_km) * v_hat])


ap = argparse.ArgumentParser()
ap.add_argument("sigma_g", type=float, nargs="?", default=18.0,
                help="sail loading, g/m^2 (default 18 = canonical design sail)")
ap.add_argument("--plane", choices=["ecliptic", "polar"], default="polar",
                help="default polar (decouples the J2-SRP resonance)")
ap.add_argument("--controller",
                choices=["naive", "blended", "dedot"],
                default="naive",
                help="default naive = reactive no-guard dE/dt-max; "
                     "see module docstring for the other controllers")
ap.add_argument("--altitude_km", type=float, default=1500.0)
ap.add_argument("--alpha_max_deg_s2", type=float, default=0.003)
ap.add_argument("--omega_max_deg_s", type=float, default=0.3)
ap.add_argument("--max_cone_deg", type=float, default=80.0)
ap.add_argument("--r_star_km", type=float, default=3996.19,
                help="blended soft periapsis guard (km); default R_eq+600")
ap.add_argument("--S_0", type=float, default=1.0e7)
ap.add_argument("--k_S", type=float, default=1.0)
ap.add_argument("--w_E", type=float, default=1.0)
ap.add_argument("--rp_warn_km", type=float, default=3896.19,
                help="dedot conditional rp-rate guard (km); default R_eq+500")
ap.add_argument("--span_sols", type=float, default=1002.0,
                help="propagation span in sols; default 1002 = 1.5 Mars years")
ap.add_argument("--altitude_floor_km", type=float, default=300.0)
ap.add_argument("--gravity_degree", type=int, default=2)
ap.add_argument("--steps_per_orbit", type=int, default=200,
                help="RK4 steps per osculating orbit; raise to resolve stiff "
                     "(high-alpha) attitude dynamics")
ap.add_argument("--max_step_true_anomaly_deg", type=float, default=0.25,
                help="cap the true-anomaly advance per step (deg); clusters "
                     "RK4 steps through periapsis (Sundman-style). Default 0.25 "
                     "(converged). None disables.")
ap.add_argument("--max_step_s", type=float, default=20.0,
                help="absolute per-step ceiling (s); holds the attitude-tracker "
                     "ODE resolved as a grows toward escape. Default 20; ~40 "
                     "is a faster exploratory setting. None disables the cap.")
ap.add_argument("--kinematic_attitude", action="store_true",
                help="bypass the slew tracker; sail snaps to the command "
                     "(no slew limits) -- the escaping reference for diagnosis")
ap.add_argument("--lead_s", type=float, default=0.0,
                help="anticipatory feed-forward lead (s); 0 = reactive. "
                     "~700 covers a 180 deg feather flip at "
                     "omega_max=0.3 deg/s. Pre-slews into the correct "
                     "orientation before an imminent transition.")
ap.add_argument("--activate_deg", type=float, default=20.0,
                help="anticipatory lead activation threshold (deg)")
ap.add_argument("--park_epoch", type=str, default=DEFAULT_PARK_EPOCH,
                help="captured parking-orbit epoch = the backward-run START "
                     "(UTC); the node epoch = park_epoch - T is reported")
ap.add_argument("--ephemeris_time_direction", type=int, default=-1,
                choices=[-1, +1],
                help="-1 (default) = clock the Sun/Mars ephemeris BACKWARD to "
                     "generate a capture node; +1 = forward (an ordinary escape, "
                     "for comparison)")
ap.add_argument("--energy_gated", action=argparse.BooleanOptionalAction, default=True,
                help="define capture-node reach as E>=0 and |r|>=Hill; a bound "
                     "Hill-radius graze is not a valid node. --no-energy_gated "
                     "selects radius-only termination.")
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()

sigma_g = args.sigma_g
loading_kg_per_m2 = sigma_g / 1000.0
_dir_tag = "capture" if args.ephemeris_time_direction == -1 else "fwdescape"
suffix = f"_{args.tag}" if args.tag else ""
tag = f"{_dir_tag}_sigma{int(round(sigma_g))}_{args.plane}_{args.controller}{suffix}"

load_kernels()
# epoch_et = the parking-orbit (backward-run START) epoch. The ephemeris is then
# read at et(t) = epoch_et + ephemeris_time_direction * t; with direction -1 the
# Sun walks backward and the node is reached at epoch_et - T.
epoch_et = utc_to_et(args.park_epoch)
mu_mars = mars_gm_km3_per_s2()

if args.plane == "polar":
    state0 = polar_state(epoch_et, args.altitude_km, mu_mars)
else:
    state0 = initial_circular_state(args.altitude_km, epoch_et)

sail = make_canonical_sail(loading_kg_per_m2)
limits = AttitudeLimits(
    alpha_max_rad_s2=math.radians(args.alpha_max_deg_s2),
    omega_max_rad_s=math.radians(args.omega_max_deg_s),
)

blended_params = BlendedParams(
    r_star_km=args.r_star_km, w_E=args.w_E, k_S=args.k_S,
    S_0_km4_s2=args.S_0, max_cone_rad=math.radians(args.max_cone_deg),
    mu_km3_s2=mu_mars,
)
dedot_params = DEdotParams(
    max_cone_rad=math.radians(args.max_cone_deg),
    rp_warn_km=args.rp_warn_km, mu_km3_s2=mu_mars,
)

qlaw_shell = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=R_EQ + 300.0)

if args.controller == "naive":  # reactive no-guard dE/dt-max
    naive_params = DEdotParams(
        max_cone_rad=math.radians(args.max_cone_deg),
        rp_warn_km=None, mu_km3_s2=mu_mars,
    )

    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return dedot_steer(r, v, s_hat, p_eff, sail_,
                           current_n_hat=current_n_hat,
                           params=naive_params).n_star_j2000
elif args.controller == "blended":
    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return blended_steer(r, v, s_hat, p_eff, sail_,
                             current_n_hat=current_n_hat,
                             params=blended_params).n_star_j2000
elif args.controller == "dedot":
    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return dedot_steer(r, v, s_hat, p_eff, sail_,
                           current_n_hat=current_n_hat,
                           params=dedot_params).n_star_j2000


def _diag(r, v, s_hat, p_eff, n):
    """Active-controller diagnostics for the CSV; missing fields -> NaN."""
    if args.controller == "blended":
        b = blended_steer(r, v, s_hat, p_eff, sail,
                          current_n_hat=n, params=blended_params)
        return (b.n_star_j2000, int(b.thrust), math.degrees(b.alpha_rad),
                b.merit_value, b.safety_margin_km4_s2, b.safety_weight)
    # Naive and guarded dE/dt controllers share the same diagnostics.
    diag_params = dedot_params if args.controller == "dedot" else DEdotParams(
        max_cone_rad=math.radians(args.max_cone_deg), rp_warn_km=None,
        mu_km3_s2=mu_mars)
    d = dedot_steer(r, v, s_hat, p_eff, sail,
                    current_n_hat=n, params=diag_params)
    nan = float("nan")
    return (d.n_star_j2000, int(d.thrust), math.degrees(d.alpha_rad),
            nan, nan, nan)


print(f"[{tag}] sigma={sigma_g:.1f} g/m^2  area={sail.area_m2:.0f} m^2  "
      f"mass={sail.mass_kg:.2f} kg  altitude={args.altitude_km:.0f} km  "
      f"plane={args.plane}  controller={args.controller}", flush=True)
print(f"[{tag}] AttitudeLimits: alpha_max={args.alpha_max_deg_s2:.4f} deg/s^2  "
      f"omega_max={args.omega_max_deg_s:.4f} deg/s (strict bound)",
      flush=True)
print(f"[{tag}] resolution: max_step_s={args.max_step_s} s, "
      f"max_dnu={args.max_step_true_anomaly_deg} deg, "
      f"steps/orbit={args.steps_per_orbit} "
      f"(max_step_s is the binding attitude-resolution control)", flush=True)
print(f"[{tag}] anticipatory lead: {args.lead_s:.0f} s "
      f"(activate > {args.activate_deg:.0f} deg)"
      f"{'  [REACTIVE/off]' if args.lead_s <= 0 else ''}", flush=True)
if args.controller == "naive":
    print(f"[{tag}] Naive (reactive no-guard dE/dt-max): "
          f"max_cone={args.max_cone_deg:.1f} deg", flush=True)
elif args.controller == "blended":
    print(f"[{tag}] BlendedParams: r_star={args.r_star_km:.1f} km, k_S={args.k_S}, "
          f"S_0={args.S_0:.1e}, max_cone={args.max_cone_deg:.1f} deg", flush=True)
elif args.controller == "dedot":
    print(f"[{tag}] DEdotParams: rp_warn={args.rp_warn_km:.1f} km, "
          f"max_cone={args.max_cone_deg:.1f} deg", flush=True)

el0_disp = osculating_elements(state0[:3], state0[3:], mu_mars)
print(f"[{tag}] start: a={el0_disp.a_km:.1f} e={el0_disp.e:.4f} "
      f"i={math.degrees(el0_disp.inclination_rad):.2f} deg", flush=True)

t_span = (0.0, args.span_sols * SECONDS_PER_SOLAR_SOL_S)
print(f"[{tag}] propagating ({args.span_sols:.0f}-sol cap) ...", flush=True)
_t_start = time.perf_counter()


def _progress(t_s, y, step):
    el = osculating_elements(y[:3], y[3:6], mu_mars)
    E = 0.5 * np.dot(y[3:6], y[3:6]) - mu_mars / np.linalg.norm(y[:3])
    wall = time.perf_counter() - _t_start
    print(f"[{tag}] step {step:8d}  sol {t_s/SECONDS_PER_SOLAR_SOL_S:8.1f}  "
          f"a {el.a_km:11.1f}  e {el.e:.4f}  rp {el.periapsis_km:10.1f}  "
          f"ra {el.apoapsis_km:12.1f}  E {E:+.4f}  [wall {wall/60.0:.1f} min]",
          flush=True)


t0 = time.perf_counter()
res = propagate_escape(
    state0, epoch_et, sail, qlaw_shell, limits, t_span,
    gravity_degree=args.gravity_degree,
    steps_per_orbit=args.steps_per_orbit,
    max_step_true_anomaly_deg=args.max_step_true_anomaly_deg,
    max_step_s=args.max_step_s,
    altitude_floor=AltitudeFloor.at_km(args.altitude_floor_km),
    steering_fn=steering,
    kinematic_attitude=args.kinematic_attitude,
    anticipatory_lead_s=args.lead_s,
    anticipatory_activate_deg=args.activate_deg,
    ephemeris_time_direction=args.ephemeris_time_direction,
    energy_gated=args.energy_gated,
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
csv_path = f"simulation_outputs/{RUN_STAMP}_{tag}_per_step.csv"
header = [
    "sol", "t_s", "et",
    "r_x_km", "r_y_km", "r_z_km", "v_x_kmps", "v_y_kmps", "v_z_kmps",
    "a_km", "e", "inc_deg", "raan_deg", "argp_deg", "nu_deg",
    "periapsis_km", "apoapsis_km", "r_mag_km", "energy_km2_s2",
    "n_x", "n_y", "n_z", "omega_x", "omega_y", "omega_z",
    "omega_mag_rad_s", "alpha_mag_rad_s2",
    "shadow", "srp_pressure_pa", "srp_accel_mag_km_s2",
    "ctrl_thrust", "ctrl_alpha_deg", "ctrl_merit",
    "S_km4_s2", "w_S",
    "n_des_x", "n_des_y", "n_des_z", "tracking_angle_deg",
]

with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    for i in range(0, n_steps, stride):
        t = float(res.t_s[i])
        et = epoch_et + args.ephemeris_time_direction * t
        orbit = res.orbit_state_km_kmps[i]
        r = orbit[:3]
        v = orbit[3:]
        n = res.attitude_state[i, :3]
        omega = res.attitude_state[i, 3:]

        el = classical_elements(orbit, mu, et)
        r_mag = float(np.linalg.norm(r))
        v_mag = float(np.linalg.norm(v))
        energy = 0.5 * v_mag * v_mag - mu / r_mag

        sun_state, _ = spice.spkezr(
            str(SUN_NAIF_ID), et, "J2000", "NONE", str(MARS_NAIF_ID)
        )
        r_sun = np.asarray(sun_state[:3], dtype=float)
        sat_to_sun = r_sun - r
        r_helio = float(np.linalg.norm(sat_to_sun))
        s_hat = sat_to_sun / r_helio
        shadow = shadow_factor(r, et, MARS_NAIF_ID, sun_position_j2000_km=r_sun)
        p_eff = solar_flux_at(r_helio) * shadow
        a_srp = mcinnes_srp_acceleration(n, s_hat, p_eff, sail)
        srp_mag = float(np.linalg.norm(a_srp))

        n_des, thrust_i, alpha_deg, merit, S_margin, w_S = _diag(r, v, s_hat, p_eff, n)
        alpha_cmd = alpha_command(n, omega, n_des, limits)
        cos_track = max(-1.0, min(1.0, float(np.dot(n, n_des))))
        tracking_deg = math.degrees(math.acos(cos_track))

        writer.writerow([
            t / SECONDS_PER_SOLAR_SOL_S, t, et,
            r[0], r[1], r[2], v[0], v[1], v[2],
            el.a_km, el.e, np.degrees(el.inclination_rad),
            np.degrees(el.raan_rad), np.degrees(el.argp_rad),
            np.degrees(el.nu_rad),
            el.periapsis_km, el.apoapsis_km, r_mag, energy,
            n[0], n[1], n[2], omega[0], omega[1], omega[2],
            float(np.linalg.norm(omega)), float(np.linalg.norm(alpha_cmd)),
            int(round(shadow)), p_eff, srp_mag,
            thrust_i, alpha_deg, merit, S_margin, w_S,
            n_des[0], n_des[1], n_des[2], tracking_deg,
        ])

print(f"[{tag}] per-step CSV ({(n_steps + stride - 1)//stride} rows, "
      f"stride {stride}) -> {csv_path}", flush=True)

# --- Summary -------------------------------------------------------------
el0 = classical_elements(res.orbit_state_km_kmps[0], mu, epoch_et)
elN = classical_elements(res.orbit_state_km_kmps[-1], mu, res.epoch_et)

# first sol where E >= 0
e_ge0_sol = None
for i in range(0, n_steps, 20):
    s = res.orbit_state_km_kmps[i]
    E = 0.5 * np.dot(s[3:], s[3:]) - mu / np.linalg.norm(s[:3])
    if E >= 0:
        e_ge0_sol = res.t_s[i] / SECONDS_PER_SOLAR_SOL_S
        break

omega_mags = np.linalg.norm(res.attitude_state[:, 3:], axis=1)
max_omega_frac = float(np.max(omega_mags)) / limits.omega_max_rad_s

# Closest physical approach: the minimum radius over the integration --
# the true safety metric, distinct from the osculating periapsis a(1-e). For a
# barely-hyperbolic escape end state the osculating r_p is a backward-extrapolated
# "virtual" periapsis the outbound spacecraft never visits (it can read below the
# floor while the actual radius is at the Hill sphere). The AltitudeFloor event
# fires on this radius, so min |r| is the physical closest approach.
r_all_km = np.linalg.norm(res.orbit_state_km_kmps[:, :3], axis=1)
min_r_idx = int(np.argmin(r_all_km))
min_r_km = float(r_all_km[min_r_idx])
min_r_sol = float(res.t_s[min_r_idx]) / SECONDS_PER_SOLAR_SOL_S
floor_r_km = R_EQ + args.altitude_floor_km

_is_capture = args.ephemeris_time_direction == -1
_mode = "Mars CAPTURE node (backward ephemeris)" if _is_capture \
    else "SRP Mars-escape (forward)"
summary_path = f"simulation_outputs/{RUN_STAMP}_{tag}_summary.txt"
lines = [
    f"{_mode} -- sigma={sigma_g:.1f} g/m^2  plane={args.plane}  "
    f"controller={args.controller}",
    f"ephemeris_time_direction : {args.ephemeris_time_direction:+d} "
    f"({'BACKWARD -> capture node' if _is_capture else 'forward -> escape'})",
    f"node/escape definition   : {'E>=0 AND |r|>=Hill (energy-gated)' if args.energy_gated else 'radius-only (|r|>=Hill)'}",
    f"park epoch (run start)   : {args.park_epoch}  (parking orbit)",
    f"altitude start     : {args.altitude_km:.0f} km",
    f"sail               : area {sail.area_m2:.0f} m^2, mass {sail.mass_kg:.2f} kg",
    f"attitude limits    : alpha_max {args.alpha_max_deg_s2:.4f} deg/s^2, "
    f"omega_max {args.omega_max_deg_s:.4f} deg/s",
    f"max |omega| / cap  : {max_omega_frac*100:.4f}%  (strict bound)",
    f"termination        : {res.termination_reason}",
    f"wall time          : {wall/60.0:.2f} min",
    f"integration steps  : {n_steps}",
    f"a    start / end   : {el0.a_km:.2f} / {elN.a_km:.2f} km",
    f"e    start / end   : {el0.e:.5f} / {elN.e:.5f}",
    f"r_p (osc) start/end: {el0.periapsis_km:.2f} / {elN.periapsis_km:.2f} km "
    f"(osculating a(1-e); VIRTUAL for a hyperbolic end state -- see min actual radius)",
    f"r_a  start / end   : {el0.apoapsis_km:.2f} / {elN.apoapsis_km:.2f} km",
    f"min actual radius  : {min_r_km:.2f} km at sol {min_r_sol:.1f}  "
    f"(closest approach; floor {floor_r_km:.2f}, margin {min_r_km - floor_r_km:+.1f} km)",
    f"first E>=0         : {('sol %.1f' % e_ge0_sol) if e_ge0_sol is not None else 'never'}",
    f"reached Hill       : {res.escaped}"
    f"{'  (node found)' if (_is_capture and res.escaped) else ''}",
]
if res.termination_t_s is not None:
    t_esc = res.termination_t_s
    lines += [f"spiral time T      : {t_esc/SECONDS_PER_SOLAR_SOL_S:.1f} sols "
              f"= {t_esc/86400.0:.1f} days"]

node_json = None
if res.escaped:
    hr = res.termination_orbit_state_km_kmps[:3]
    hv = res.termination_orbit_state_km_kmps[3:]
    node_et = res.termination_et  # = park_epoch + dir*T = park_epoch - T (capture)
    node_utc = spice.et2utc(node_et, "C", 3)
    v_inbound = -np.asarray(hv, dtype=float)  # capture-entry velocity (time-reversed)
    _hdr = ("Mars CAPTURE NODE" if _is_capture else "Hill-sphere handoff") \
        + " (J2000, Mars-centred)"
    lines += [
        "",
        f"=== {_hdr} ===",
        f"  node epoch_et : {node_et:.6f}  ({node_utc})",
        f"  r_km          : [{hr[0]:.6f}, {hr[1]:.6f}, {hr[2]:.6f}]",
        f"  |r|           : {np.linalg.norm(hr):.3f} km",
        f"  v_kmps (OUTBOUND, as integrated): "
        f"[{hv[0]:.9f}, {hv[1]:.9f}, {hv[2]:.9f}]",
    ]
    if _is_capture:
        lines += [
            f"  v_kmps (INBOUND capture-entry = -v_outbound): "
            f"[{v_inbound[0]:.9f}, {v_inbound[1]:.9f}, {v_inbound[2]:.9f}]",
            "  (the interplanetary leg delivers the sail to r_km at node epoch "
            "with the INBOUND velocity; capture = time-reverse of this run)",
        ]
    lines += [f"  |v|           : {np.linalg.norm(hv):.6f} km/s"]

    node_json = {
        "kind": "mars_capture_node" if _is_capture else "hill_handoff",
        "ephemeris_time_direction": int(args.ephemeris_time_direction),
        "sigma_g_per_m2": float(sigma_g),
        "plane": args.plane,
        "park_epoch_utc": args.park_epoch,
        "park_epoch_et": float(epoch_et),
        "spiral_time_sols": float(res.termination_t_s / SECONDS_PER_SOLAR_SOL_S),
        "node_epoch_et": float(node_et),
        "node_epoch_utc": node_utc,
        "r_km_j2000_mars": [float(x) for x in hr],
        "v_kmps_outbound_j2000_mars": [float(x) for x in hv],
        "v_kmps_inbound_capture_entry_j2000_mars": [float(x) for x in v_inbound],
        "hill_radius_km": float(np.linalg.norm(hr)),
        "max_omega_frac_of_cap": float(max_omega_frac),
        "min_actual_radius_km": min_r_km,
        "altitude_floor_km": floor_r_km,
    }
else:
    lines += ["", "(did not reach the Hill sphere within the span -- no node)"]

summary = "\n".join(lines)
with open(summary_path, "w") as fh:
    fh.write(summary + "\n")
print(summary, flush=True)
print(f"[{tag}] summary -> {summary_path}", flush=True)

if node_json is not None:
    json_path = f"simulation_outputs/{RUN_STAMP}_{tag}_node.json"
    with open(json_path, "w") as fh:
        json.dump(node_json, fh, indent=2)
    print(f"[{tag}] capture-node JSON -> {json_path}", flush=True)
