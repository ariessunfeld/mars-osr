"""Interplanetary Earth-Hill -> Mars-Hill SRP-sail cruise -- PIECEWISE-RTN IPOPT solve.

This command uses a piecewise-constant RTN tilt-angle command (N nodes x
(phi in-plane, theta
out-of-plane), +/-35 deg) solved as a 6-equality-constraint rendezvous with IPOPT
(reflectors.cruise_solve). The local parameterisation avoids the cone/clock
singularity and gives a well-conditioned terminal Jacobian.

One solve runs per invocation. Supported cases include:

  --perturbers none --target node
  --perturbers earth,moon,mars --target node --warm-from <solution.json>
  --perturbers none --target mars_state --departure earth_exact

Heliocentric J2000 endpoints come from the Earth-escape and Mars-capture files
supplied on the command line. Prints numbers only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import spiceypy as spice

# Reuse endpoint construction from the Fourier-command script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_interplanetary_cruise import (  # noqa: E402
    AU_KM,
    build_problem,
    configure_endpoint_paths,
)

from reflectors.attitude_control import AttitudeLimits  # noqa: E402
from reflectors.cruise_piecewise import (  # noqa: E402
    assert_piecewise_slew_feasible,
    make_piecewise_cruise_defect,
    pack_angles,
    piecewise_cruise_terminal_miss,
    unpack_angles,
)
from reflectors.cruise_solve import angle_box_bounds, solve_piecewise_ipopt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIMOUT = REPO / "simulation_outputs"


def _mars_state_target(p):
    """Mars heliocentric J2000 state at the arrival epoch."""
    st, _ = spice.spkezr("499", p["end_et"], "J2000", "NONE", "10")
    return np.asarray(st, dtype=float)


def _solution_path(tag):
    return SIMOUT / f"cruise_piecewise_{tag}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma-g", type=float, default=18.0)
    ap.add_argument("--N", type=int, default=16, help="number of control nodes")
    ap.add_argument("--target", choices=["node", "mars_state"], default="node")
    ap.add_argument("--perturbers", default="earth,moon,mars",
                    help="'none' | 'mars' | 'earth,moon,mars'")
    ap.add_argument("--departure", default="hill_exit",
                    help="'hill_exit' (escape state) | 'earth_exact' (Earth pos+vel)")
    ap.add_argument("--steps-per-orbit", type=int, default=200)
    ap.add_argument("--angle-bound-deg", type=float, default=35.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--fd-h", type=float, default=1.0e-3)
    ap.add_argument("--tol", type=float, default=1.0e-7)
    ap.add_argument("--warm-from", type=str, default=None,
                    help="JSON solution to warm-start the angle vector from")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--validate", action="store_true",
                    help="after the solve, run the 2-resolution convergence check")
    ap.add_argument("--earth-summary", type=Path, required=True,
                    help="Earth-escape Hill-exit summary produced by "
                         "run_srp_escape_earth.py")
    ap.add_argument("--node-json", type=Path, required=True,
                    help="Mars capture-node JSON produced by run_srp_capture.py")
    args = ap.parse_args()

    configure_endpoint_paths(args.earth_summary, args.node_json)
    p = build_problem(sigma_g=args.sigma_g, perturbers=args.perturbers,
                      departure=args.departure)
    N = args.N
    z0, et0, T_s = p["z0"], p["et0"], p["T_s"]
    ref, sail, cb, tb = p["ref_normal"], p["sail"], p["central_body"], p["third_bodies"]
    r_scale, v_scale = p["r_scale"], p["v_scale"]
    z_tgt = p["zT"] if args.target == "node" else _mars_state_target(p)

    print(f"=== piecewise-RTN cruise solve (sigma={args.sigma_g}, N={N}, "
          f"target={args.target}, perturbers={args.perturbers}, departure={args.departure}) ===")
    print(f"START helio |r|={np.linalg.norm(z0[:3])/AU_KM:.5f} AU  |v|={np.linalg.norm(z0[3:6]):.4f} km/s")
    print(f"TARGET helio |r|={np.linalg.norm(z_tgt[:3])/AU_KM:.5f} AU  |v|={np.linalg.norm(z_tgt[3:6]):.4f} km/s")
    print(f"T = {T_s/86400.0:.4f} d   r_scale={r_scale:.4e} km   v_scale={v_scale:.4f} km/s")

    defect = make_piecewise_cruise_defect(
        z0, z_tgt, et0, T_s, ref, sail, cb, tb,
        N=N, r_scale_km=r_scale, v_scale_kmps=v_scale, steps_per_orbit=args.steps_per_orbit,
    )
    bound = math.radians(args.angle_bound_deg)
    lb, ub = angle_box_bounds(N, bound)

    if args.warm_from:
        warm = json.loads(Path(args.warm_from).read_text())
        x0 = np.asarray(warm["x"], dtype=float)
        if x0.size != 2 * N:
            raise ValueError(f"warm-from has {x0.size} vars; need 2N={2*N}")
        print(f"warm-starting from {args.warm_from} (best_defect_norm={warm.get('best_defect_norm')})")
    else:
        x0 = np.zeros(2 * N)  # flat sunward sail

    # Raw initial miss (diagnostic anchor).
    p0, t0 = unpack_angles(x0, N)
    r0, v0, _ = piecewise_cruise_terminal_miss(
        p0, t0, z0, z_tgt, et0, T_s, ref, sail, cb, tb, steps_per_orbit=args.steps_per_orbit
    )
    print(f"initial (x0) miss: r_miss={r0/AU_KM:.5f} AU ({r0:.1f} km)  v_miss={v0:.5f} km/s")

    x_best, info = solve_piecewise_ipopt(
        defect, 2 * N, N, lb=lb, ub=ub, x0=x0, workers=args.workers,
        fd_h=args.fd_h, max_iter=args.max_iter, tol=args.tol,
    )

    phis, thetas = unpack_angles(x_best, N)
    r_miss, v_miss, z_T = piecewise_cruise_terminal_miss(
        phis, thetas, z0, z_tgt, et0, T_s, ref, sail, cb, tb, steps_per_orbit=args.steps_per_orbit
    )
    worst, budget = assert_piecewise_slew_feasible(phis, thetas, T_s, AttitudeLimits(), N)
    print(f"IPOPT status={info.get('status')} ({info.get('status_msg', b'')!r})")
    print(f"best_defect_norm={info['best_defect_norm']:.4e}")
    print(f"FINAL miss: r_miss={r_miss/AU_KM:.6f} AU ({r_miss:.3f} km)  v_miss={v_miss:.6f} km/s")
    print(f"slew feasible: worst node {math.degrees(worst):.4f} deg vs budget {math.degrees(budget):.1f} deg")
    print(f"phi  [deg] = {np.array2string(np.degrees(phis), precision=2, max_line_width=200)}")
    print(f"theta[deg] = {np.array2string(np.degrees(thetas), precision=2, max_line_width=200)}")

    SIMOUT.mkdir(exist_ok=True)
    out = _solution_path(args.tag)
    out.write_text(json.dumps({
        "tag": args.tag, "sigma_g": args.sigma_g, "N": N, "target": args.target,
        "perturbers": args.perturbers, "departure": args.departure,
        "steps_per_orbit": args.steps_per_orbit, "T_s": T_s, "et0": et0,
        "x": x_best.tolist(), "r_miss_km": r_miss, "v_miss_kmps": v_miss,
        "best_defect_norm": info["best_defect_norm"], "ipopt_status": int(info.get("status", -99)),
    }, indent=2))
    print(f"wrote {out}")

    if args.validate:
        rA, vA, zA = piecewise_cruise_terminal_miss(
            phis, thetas, z0, z_tgt, et0, T_s, ref, sail, cb, tb, steps_per_orbit=200
        )
        rB, vB, zB = piecewise_cruise_terminal_miss(
            phis, thetas, z0, z_tgt, et0, T_s, ref, sail, cb, tb, steps_per_orbit=800
        )
        d_r = float(np.linalg.norm(zA[:3] - zB[:3]))
        d_v = float(np.linalg.norm(zA[3:6] - zB[3:6]))
        print("2-resolution convergence (steps_per_orbit 200 vs 800):")
        print(f"  r_miss 200={rA:.3f} km  800={rB:.3f} km")
        print(f"  |zA-zB| terminal: d_r={d_r:.3f} km ({d_r/AU_KM:.3e} AU)  d_v={d_v:.6f} km/s")


if __name__ == "__main__":
    main()
