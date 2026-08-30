"""Interplanetary Earth-Hill -> Mars-Hill SRP-sail cruise (single-shooting).

Connects an Earth-escape Hill handoff to a Mars capture node by direct single
shooting. The endpoint epochs determine the transit duration. The solver
optimizes a smooth time-Fourier cone/clock sail command
(``reflectors.cruise_command``) to null the terminal miss. The command is
applied kinematically; slew
feasibility is enforced analytically via the optimiser's coefficient box bounds
(``feasible_coeff_boxes``), without a bang-bang tracker.

Modes:
  (default)   Forward feasibility: a hand command -> raw terminal miss.
  --solve     Differential evolution + parallel-FD L-BFGS-B polish.
  --validate  Two-resolution convergence on a converged command + figure.

Prints numbers only; no interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import spiceypy as spice

from reflectors.attitude_control import AttitudeLimits
from reflectors.central_body import sun_central_body
from reflectors.cruise_command import (
    assert_cruise_slew_feasible,
    feasible_coeff_boxes,
    heliocentric_angular_rate,
    heliocentric_orbit_normal,
)
from reflectors.cruise_cost import (
    cruise_terminal_miss,
    make_cruise_terminal_miss_cost,
    propagate_cruise,
)
from reflectors.dynamics import sun_gm_km3_per_s2
from reflectors.kernels import load_kernels
from reflectors.sail_designs import make_canonical_sail
from reflectors.third_body import (
    earth_third_body,
    mars_third_body,
    moon_third_body,
)

REPO = Path(__file__).resolve().parents[1]
SIMOUT = REPO / "simulation_outputs"
NODE_JSON: Path | None = None
EARTH_SUMMARY: Path | None = None
AU_KM = 1.495978707e8


# ---------------------------------------------------------------------------
# Endpoint parsing + heliocentric conversion
# ---------------------------------------------------------------------------


def _parse_earth_start():
    if EARTH_SUMMARY is None:
        raise ValueError("an Earth-escape summary is required")
    text = EARTH_SUMMARY.read_text()
    et = float(re.search(r"epoch_et\s*:\s*([-\d.eE+]+)", text).group(1))
    r = np.array([float(x) for x in re.search(r"r_km\s*:\s*\[([^\]]+)\]", text).group(1).split(",")])
    v = np.array([float(x) for x in re.search(r"v_kmps\s*:\s*\[([^\]]+)\]", text).group(1).split(",")])
    return et, r, v, 399


def _parse_mars_node():
    if NODE_JSON is None:
        raise ValueError("a Mars capture-node JSON file is required")
    d = json.loads(NODE_JSON.read_text())
    et = float(d["node_epoch_et"])
    r = np.array(d["r_km_j2000_mars"], dtype=float)
    v = np.array(d["v_kmps_inbound_capture_entry_j2000_mars"], dtype=float)
    return et, r, v, 499


def _to_heliocentric(et, r_geo, v_geo, planet_naif):
    st, _ = spice.spkezr(str(planet_naif), et, "J2000", "NONE", "10")
    return r_geo + np.asarray(st[:3], float), v_geo + np.asarray(st[3:], float)


def configure_endpoint_paths(earth_summary: str | Path, node_json: str | Path) -> None:
    """Set the two generated handoff artifacts used by the cruise solvers."""
    global EARTH_SUMMARY, NODE_JSON
    EARTH_SUMMARY = Path(earth_summary)
    NODE_JSON = Path(node_json)


def build_problem(sigma_g=18.0, perturbers="earth,moon,mars", departure="hill_exit"):
    """Assemble the fixed-time fixed-endpoint cruise BVP (Sun-centred J2000).

    ``perturbers`` selects the third bodies: ``"earth,moon,mars"`` (default),
    ``"mars"``, or ``"none"`` (point-mass solar gravity only)."""
    load_kernels()
    mu = sun_gm_km3_per_s2()
    s_et, r0g, v0g, s_pl = _parse_earth_start()
    e_et, rTg, vTg, e_pl = _parse_mars_node()
    if departure == "earth_exact":
        # Start at Earth's heliocentric state without escape excess or Hill
        # offset.
        z0 = np.asarray(spice.spkezr("399", s_et, "J2000", "NONE", "10")[0], dtype=float)
    elif departure == "hill_exit":
        z0 = np.concatenate(_to_heliocentric(s_et, r0g, v0g, s_pl))
    else:
        raise ValueError(f"departure must be 'hill_exit' or 'earth_exact', got {departure!r}")
    zT = np.concatenate(_to_heliocentric(e_et, rTg, vTg, e_pl))
    T_s = e_et - s_et
    ref_normal = heliocentric_orbit_normal(z0)
    sail = make_canonical_sail(sigma_g / 1000.0)
    central_body = sun_central_body()
    _pert_map = {
        "none": (),
        "mars": (mars_third_body(),),
        "earth,moon,mars": (earth_third_body(), moon_third_body(), mars_third_body()),
    }
    if perturbers not in _pert_map:
        raise ValueError(f"perturbers must be one of {list(_pert_map)}, got {perturbers!r}")
    third_bodies = _pert_map[perturbers]
    # Scale velocity by v_earth/(2*pi), making position and velocity residuals
    # comparable in AU and year/(2*pi) units.
    v_earth_dep = np.asarray(spice.spkezr("399", s_et, "J2000", "NONE", "10")[0][3:])
    v_scale = float(np.linalg.norm(v_earth_dep)) / (2.0 * math.pi)
    return {
        "mu": mu, "z0": z0, "zT": zT, "et0": s_et, "T_s": T_s,
        "ref_normal": ref_normal, "sail": sail, "central_body": central_body,
        "third_bodies": third_bodies, "r_scale": AU_KM, "v_scale": v_scale,
        "start_et": s_et, "end_et": e_et, "sigma_g": sigma_g,
    }


def _solution_path(order, sigma_g):
    return SIMOUT / f"cruise_solution_K{order}_sigma{int(round(sigma_g))}.json"


def _pad_coeffs(coeffs, from_order, to_order):
    """Embed an order-``from_order`` Fourier coeff vector into order ``to_order``
    by zero-padding the new (higher) harmonics -- so a higher-K solve can warm-
    start from a lower-K solution (the same command, with headroom for more DOF).
    Layout: [a0, ac_1..K, as_1..K, d0, dc_1..K, ds_1..K]."""
    cf = np.asarray(coeffs, dtype=float)
    Kf, Kt = from_order, to_order
    hf = 1 + 2 * Kf
    a0, ac, as_ = cf[0], cf[1:1 + Kf], cf[1 + Kf:1 + 2 * Kf]
    d0, dc, ds = cf[hf], cf[hf + 1:hf + 1 + Kf], cf[hf + 1 + Kf:hf + 1 + 2 * Kf]
    z = lambda v: np.concatenate([v, np.zeros(Kt - Kf)])
    return np.concatenate([[a0], z(ac), z(as_), [d0], z(dc), z(ds)])


# ---------------------------------------------------------------------------
# Optimiser box bounds (slew-feasible by construction)
# ---------------------------------------------------------------------------


# Physical cap on the clock harmonic amplitude. One turn bounds the
# differential-evolution search space; the rate ceiling is enforced separately.
CLOCK_HARMONIC_CAP_RAD = 2.0 * math.pi


def make_bounds(order, T_s, mu, z0, limits, cone_lo=0.2, cone_hi=1.2,
                clock_cap=CLOCK_HARMONIC_CAP_RAD):
    """DE/L-BFGS-B box bounds that keep every command slew-feasible.

    Cone bias in [cone_lo, cone_hi] (interior of [0, pi/2]); cone harmonics in
    +/- cone_box (range-limited for the minimum margin over the bias range so
    the cone never clamps); clock bias in [-pi, pi]; clock harmonics in +/-
    clock_box (the smaller of the rate ceiling and the physical
    ``CLOCK_HARMONIC_CAP_RAD``). A conservative heliocentric frame rate
    is used.
    """
    # Conservative max heliocentric angular rate (fastest at the smallest r;
    # the sail only raises its orbit, so r0 is ~perihelion). 10% radius margin.
    r_min = 0.9 * float(np.linalg.norm(z0[:3]))
    omega_frame_max = math.sqrt(mu / r_min ** 3)
    worst_margin_bias = cone_lo if cone_lo <= (math.pi / 2 - cone_hi) else cone_hi
    cone_box, clock_box_rate = feasible_coeff_boxes(
        order, T_s, limits, worst_margin_bias, omega_frame_max_rad_s=omega_frame_max
    )
    clock_box = min(clock_box_rate, clock_cap)
    half = 1 + 2 * order
    lo = np.empty(2 + 4 * order)
    hi = np.empty(2 + 4 * order)
    lo[0], hi[0] = cone_lo, cone_hi                       # cone bias
    lo[1:half], hi[1:half] = -cone_box, cone_box          # cone harmonics
    lo[half], hi[half] = -math.pi, math.pi                # clock bias
    lo[half + 1:], hi[half + 1:] = -clock_box, clock_box  # clock harmonics
    return list(zip(lo, hi)), omega_frame_max, cone_box, clock_box


# ---------------------------------------------------------------------------
# Forward feasibility
# ---------------------------------------------------------------------------


def forward_feasibility(p, order=1):
    print("=== Forward feasibility (hand commands -> raw terminal miss) ===")
    print(f"T_s = {p['T_s']/86400:.4f} d   r_scale = {p['r_scale']:.3e} km   "
          f"v_scale = {p['v_scale']:.4f} km/s")
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    _, omega_frame_max, cone_box, clock_box = make_bounds(
        order, p["T_s"], p["mu"], p["z0"], limits
    )
    print(f"order K={order}: cone_box={cone_box:.4e} rad  clock_box={clock_box:.4e} rad  "
          f"omega_frame_max={math.degrees(omega_frame_max):.3e} deg/s")
    n = 2 + 4 * order
    # A few hand commands spanning cone/clock; in-plane thrust ~ clock pi/2.
    hands = {
        "cone0.6_clock+pi/2": _hand(order, 0.6, math.pi / 2),
        "cone0.6_clock-pi/2": _hand(order, 0.6, -math.pi / 2),
        "cone0.35_clock+pi/2": _hand(order, 0.35, math.pi / 2),
        "cone0.9_clock0": _hand(order, 0.9, 0.0),
    }
    for label, x in hands.items():
        assert_cruise_slew_feasible(
            x, p["T_s"], limits, omega_frame_max_rad_s=omega_frame_max
        )
        r_miss, v_miss, zT = cruise_terminal_miss(
            x, p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"],
            p["sail"], p["central_body"], p["third_bodies"],
        )
        cost = r_miss / p["r_scale"] + v_miss / p["v_scale"]
        print(f"  {label:22s}  r_miss={r_miss:13.1f} km ({r_miss/AU_KM:7.4f} AU)  "
              f"v_miss={v_miss:9.5f} km/s  cost={cost:.5f}")


def _hand(order, cone_bias, clock_bias):
    x = np.zeros(2 + 4 * order)
    x[0] = cone_bias
    x[1 + 2 * order] = clock_bias
    return x


# ---------------------------------------------------------------------------
# Global DE plus parallel-FD L-BFGS-B polish
# ---------------------------------------------------------------------------


def _seeded_init_population(seeds, bounds, n_pop, rng_scale=0.25):
    """Build a DE initial population concentrated near the seed solutions.

    Known feasible solutions, such as results from a lighter sail, are included
    verbatim plus Gaussian-perturbed copies (scaled by ``rng_scale`` of each
    axis range), filled out with uniform samples, all clipped into bounds. Uses
    a deterministic index-seeded pseudo-random sequence. These optimization
    commands run single-process here, so a fixed-seed NumPy generator is
    sufficient."""
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    span = hi - lo
    rng = np.random.default_rng(12345)
    rows = []
    for s in seeds:
        rows.append(np.clip(np.asarray(s, dtype=float), lo, hi))
    while len(rows) < n_pop:
        base = seeds[len(rows) % len(seeds)] if seeds else lo + 0.5 * span
        if len(rows) < n_pop * 0.7 and seeds:
            cand = np.asarray(base, float) + rng.normal(0, rng_scale, size=len(lo)) * span
        else:
            cand = lo + rng.uniform(0, 1, size=len(lo)) * span
        rows.append(np.clip(cand, lo, hi))
    return np.array(rows)


def solve(p, order=1, workers=8, de_seed=42, de_maxiter=200, de_popsize=15,
          max_step_s=None, seed_coeffs=None, clock_cap=CLOCK_HARMONIC_CAP_RAD):
    from scipy.optimize import differential_evolution, minimize
    from reflectors.parallel import (
        CloudpickleMap,
        configure_multiprocessing_for_spice,
        parallel_fd_jacobian,
    )

    print(f"=== DE + FD-polish solve (K={order}, workers={workers}, "
          f"clock_cap={clock_cap:.2f}, seeded={seed_coeffs is not None}) ===")
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    bounds, omega_frame_max, cone_box, clock_box = make_bounds(
        order, p["T_s"], p["mu"], p["z0"], limits, clock_cap=clock_cap
    )
    cost = make_cruise_terminal_miss_cost(
        p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
        p["central_body"], p["third_bodies"],
        r_scale_km=p["r_scale"], v_scale_kmps=p["v_scale"], w_v=1.0,
        max_step_s=max_step_s,
    )
    # SPICE fork-safety: re-furnish kernels immediately before forking the pool
    # (parallel.py:77-80 pattern), so every worker inherits initialized DAF file
    # handles. The parent advanced the handles via spkezr in build_problem /
    # forward-feasibility, and forking those used handles corrupts concurrent
    # worker reads (SPICE(DAFFRNOTFOUND)).
    configure_multiprocessing_for_spice()
    load_kernels()
    cp_map = CloudpickleMap(n_workers=workers)
    if seed_coeffs is not None:
        init_pop = _seeded_init_population(
            seed_coeffs, bounds, n_pop=de_popsize * (2 + 4 * order)
        )
        de_init = init_pop
    else:
        de_init = "sobol"
    try:
        de = differential_evolution(
            cost, bounds, workers=cp_map, updating="deferred", polish=False,
            init=de_init, seed=de_seed, maxiter=de_maxiter, popsize=de_popsize,
            tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
        )
        print(f"  DE: cost={de.fun:.6e}  nfev={de.nfev}  msg={de.message}")
        res = minimize(
            cost, de.x, method="L-BFGS-B", bounds=bounds,
            jac=lambda xx: parallel_fd_jacobian(
                cost, xx, workers=cp_map, bounds=bounds
            ),
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10},
        )
    finally:
        cp_map.close()
    x = res.x
    print(f"  polish: cost={res.fun:.6e}  nit={res.nit}  msg={res.message}")
    r_miss, v_miss, zT = cruise_terminal_miss(
        x, p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
        p["central_body"], p["third_bodies"], max_step_s=max_step_s,
    )
    b_omega, b_alpha = assert_cruise_slew_feasible(
        x, p["T_s"], limits, omega_frame_max_rad_s=omega_frame_max
    )
    print(f"  converged terminal miss: r_miss={r_miss:.1f} km ({r_miss/AU_KM:.5f} AU)  "
          f"v_miss={v_miss:.5f} km/s")
    print(f"  slew bounds (feasible): |omega|<={math.degrees(b_omega):.3e} deg/s "
          f"({b_omega/limits.omega_max_rad_s:.2e} cap), "
          f"|alpha|<={math.degrees(b_alpha):.3e} deg/s^2 "
          f"({b_alpha/limits.alpha_max_rad_s2:.2e} cap)")
    print(f"  coeffs = {np.array2string(x, precision=8, max_line_width=200)}")
    # Persist for validation, keyed by order and sigma.
    out = _solution_path(order, p["sigma_g"])
    out.write_text(json.dumps({
        "order": order, "sigma_g": p["sigma_g"], "coeffs": x.tolist(),
        "cost": float(res.fun), "r_miss_km": r_miss, "v_miss_kmps": v_miss,
        "T_s": p["T_s"], "et0": p["et0"],
    }, indent=2))
    print(f"  wrote {out}")
    return x, r_miss, v_miss


# ---------------------------------------------------------------------------
# IPOPT single shooting (equality-constrained terminal rendezvous)
# ---------------------------------------------------------------------------


def solve_ipopt(p, order=2, workers=10, clock_cap=2.0 * math.pi, max_iter=400,
                warm_coeffs=None, fd_h=1e-3, w_reg=1e-5, tol=1e-9,
                steps_per_orbit=400, free_departure=False, vinf_max=2.5):
    """Drive the terminal defect to zero with IPOPT (cyipopt).

    Reformulation: instead of minimizing a weighted miss (which
    DE/L-BFGS-B cannot reliably navigate the ill-conditioned ~1-rev single-shooting
    map), solve the rendezvous as 6 EQUALITY CONSTRAINTS g(x)=0 where g is the
    non-dimensional terminal-state defect (positions/AU, velocities/VU). IPOPT's
    interior-point method with a parallel finite-difference constraint Jacobian
    and limited-memory Hessian handles the conditioning. A tiny harmonic
    regularization keeps the (otherwise rank-deficient, N>6) objective well-posed.
    """
    import cyipopt
    from reflectors.cruise_cost import make_cruise_defect, make_free_departure_defect
    from reflectors.parallel import CloudpickleMap, configure_multiprocessing_for_spice

    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    bounds, omega_frame_max, _, _ = make_bounds(
        order, p["T_s"], p["mu"], p["z0"], limits, clock_cap=clock_cap
    )
    n = 2 + 4 * order
    half = 1 + 2 * order
    harm_idx = list(range(1, half)) + list(range(half + 1, n))  # harmonic coeffs

    v_earth = np.asarray(spice.spkezr("399", p["start_et"], "J2000", "NONE", "10")[0][3:])
    if free_departure:
        bounds = list(bounds) + [(-vinf_max, vinf_max)] * 3
        defect = make_free_departure_defect(
            p["z0"][:3], v_earth, p["zT"], p["et0"], p["T_s"], p["sail"],
            p["central_body"], p["third_bodies"], order=order,
            r_scale_km=p["r_scale"], v_scale_kmps=p["v_scale"],
            steps_per_orbit=steps_per_orbit,
        )
    else:
        defect = make_cruise_defect(
            p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
            p["central_body"], p["third_bodies"],
            r_scale_km=p["r_scale"], v_scale_kmps=p["v_scale"],
            steps_per_orbit=steps_per_orbit,
        )
    n_dv = n + (3 if free_departure else 0)
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    configure_multiprocessing_for_spice()
    load_kernels()
    cp_map = CloudpickleMap(n_workers=workers)
    best = {"norm": np.inf, "x": np.asarray(warm_coeffs, dtype=float) if warm_coeffs is not None else None}

    class CruiseNLP:
        def objective(self, x):
            return w_reg * float(np.sum(np.asarray(x)[harm_idx] ** 2))

        def gradient(self, x):
            g = np.zeros(n_dv); g[harm_idx] = 2.0 * w_reg * np.asarray(x)[harm_idx]
            return g

        def constraints(self, x):
            g = defect(np.asarray(x, dtype=float))
            gn = float(np.linalg.norm(g))
            if gn < best["norm"]:  # IPOPT's final iterate need not minimize defect.
                best["norm"] = gn
                best["x"] = np.asarray(x, dtype=float).copy()
            return g

        def jacobian(self, x):
            x = np.asarray(x, dtype=float)
            # Per-variable central FD step: relative for large-magnitude axes
            # (clock harmonics can be O(10-25)), absolute floor for small ones.
            h_i = np.maximum(fd_h, fd_h * np.abs(x))
            pts = []
            for i in range(n_dv):
                xp = x.copy(); xp[i] += h_i[i]; pts.append(xp)
                xm = x.copy(); xm[i] -= h_i[i]; pts.append(xm)
            res = cp_map(defect, pts)  # 2*n_dv entries, each a 6-vector
            J = np.empty((6, n_dv))
            for i in range(n_dv):
                J[:, i] = (np.asarray(res[2 * i]) - np.asarray(res[2 * i + 1])) / (2.0 * h_i[i])
            return J.flatten()  # dense row-major

        def jacobianstructure(self):
            return (np.repeat(np.arange(6), n_dv), np.tile(np.arange(n_dv), 6))

        def intermediate(self, alg_mod, it, obj, inf_pr, inf_du, mu, dn, rg,
                         a_du, a_pr, ls):
            if it % 5 == 0 or inf_pr < 1e-4:
                print(f"  [ipopt] it={it:4d}  obj={obj:.3e}  inf_pr(scaled)={inf_pr:.3e}",
                      flush=True)

    print(f"=== IPOPT solve (K={order}, sigma={p['sigma_g']}, "
          f"clock_cap={clock_cap:.2f}, free_departure={free_departure}) ===")
    if warm_coeffs is not None:
        x0 = np.asarray(warm_coeffs, dtype=float)
        if free_departure and x0.size == n:  # pad with zero v_inf
            x0 = np.concatenate([x0, np.zeros(3)])
    else:
        x0 = 0.5 * (lb + ub)
    x0 = np.clip(x0, lb, ub)
    nlp = cyipopt.Problem(n=n_dv, m=6, problem_obj=CruiseNLP(),
                          lb=lb.tolist(), ub=ub.tolist(),
                          cl=[0.0] * 6, cu=[0.0] * 6)
    nlp.add_option('tol', tol)
    nlp.add_option('constr_viol_tol', 1e-9)
    nlp.add_option('max_iter', int(max_iter))
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('print_level', 0)
    try:
        x_final, info = nlp.solve(x0)
    finally:
        cp_map.close()
    # Retain the lowest-defect iterate because IPOPT returns the final iterate.
    x_sol = best["x"] if best["x"] is not None else x_final
    print(f"  best tracked |g|={best['norm']:.4e} (vs final)")
    coeffs_sol = x_sol[:n]
    if free_departure:
        v_inf = x_sol[n:n + 3]
        z0_used = np.concatenate([p["z0"][:3], v_earth + v_inf])
        ref_used = heliocentric_orbit_normal(z0_used)
        ve_hat = v_earth / np.linalg.norm(v_earth)
        ang = math.degrees(math.acos(np.clip(np.dot(v_inf / np.linalg.norm(v_inf), ve_hat), -1, 1)))
        print(f"  chosen v_inf={np.array2string(v_inf, precision=4)} km/s |v_inf|={np.linalg.norm(v_inf):.4f} "
              f"angle-to-prograde={ang:.1f} deg (escape actual: 1.168 km/s @ 108 deg)")
    else:
        z0_used, ref_used = p["z0"], p["ref_normal"]
    b_omega, b_alpha = assert_cruise_slew_feasible(
        coeffs_sol, p["T_s"], limits, omega_frame_max_rad_s=omega_frame_max
    )
    r_miss, v_miss, _ = cruise_terminal_miss(
        coeffs_sol, z0_used, p["zT"], p["et0"], p["T_s"], ref_used, p["sail"],
        p["central_body"], p["third_bodies"], steps_per_orbit=steps_per_orbit,
    )
    print(f"  IPOPT status={info['status']} ({info.get('status_msg', b'')!r})")
    print(f"  terminal miss: r_miss={r_miss/AU_KM:.6f} AU ({r_miss:.1f} km)  v_miss={v_miss:.6f} km/s")
    print(f"  slew feasible: |omega|/cap={b_omega/limits.omega_max_rad_s:.2e}  "
          f"|alpha|/cap={b_alpha/limits.alpha_max_rad_s2:.2e}")
    print(f"  coeffs = {np.array2string(coeffs_sol, precision=6, max_line_width=200)}")
    tag = "ipopt_freedep" if free_departure else "ipopt"
    out = SIMOUT / (f"cruise_solution_K{order}_sigma{int(round(p['sigma_g']))}"
                    + ("_freedep.json" if free_departure else ".json"))
    rec = {
        "order": order, "sigma_g": p["sigma_g"], "coeffs": np.asarray(coeffs_sol).tolist(),
        "cost": float(r_miss / p["r_scale"] + v_miss / p["v_scale"]),
        "r_miss_km": r_miss, "v_miss_kmps": v_miss, "via": tag,
        "ipopt_status": int(info["status"]),
    }
    if free_departure:
        rec["v_inf_kmps"] = np.asarray(x_sol[n:n + 3]).tolist()
    out.write_text(json.dumps(rec, indent=2))
    print(f"  wrote {out}")
    return x_sol, r_miss, v_miss


# ---------------------------------------------------------------------------
# Free-departure sensitivity solve
# ---------------------------------------------------------------------------


def solve_free_departure(p, order=2, workers=10, vinf_max=2.5, de_maxiter=400,
                         de_popsize=15, clock_cap=8.0 * math.pi):
    """Optimise (cone/clock command + departure v_inf) at the problem's sigma.

    Frees the Earth-relative excess velocity (|component| <= vinf_max) so the
    optimiser selects an injection from the fixed departure position. This
    provides a sensitivity comparison with the fixed-handoff problem."""
    from scipy.optimize import differential_evolution, minimize
    from reflectors.cruise_cost import make_free_departure_cost
    from reflectors.parallel import (
        CloudpickleMap, configure_multiprocessing_for_spice, parallel_fd_jacobian,
    )

    print(f"=== free-departure solve (K={order}, sigma={p['sigma_g']}, "
          f"vinf_max={vinf_max} km/s) ===")
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    cbounds, omega_frame_max, _, _ = make_bounds(
        order, p["T_s"], p["mu"], p["z0"], limits, clock_cap=clock_cap
    )
    bounds = list(cbounds) + [(-vinf_max, vinf_max)] * 3
    r0 = p["z0"][:3]
    v_earth = np.asarray(spice.spkezr("399", p["start_et"], "J2000", "NONE", "10")[0][3:])
    cost = make_free_departure_cost(
        r0, v_earth, p["zT"], p["et0"], p["T_s"], p["sail"], p["central_body"],
        p["third_bodies"], order=order, r_scale_km=p["r_scale"],
        v_scale_kmps=p["v_scale"], w_v=1.0,
    )
    configure_multiprocessing_for_spice()
    load_kernels()
    cp_map = CloudpickleMap(n_workers=workers)
    try:
        de = differential_evolution(
            cost, bounds, workers=cp_map, updating="deferred", polish=False,
            init="sobol", seed=42, maxiter=de_maxiter, popsize=de_popsize,
            tol=1e-9, mutation=(0.5, 1.0), recombination=0.7,
        )
        print(f"  DE: cost={de.fun:.6e}  nfev={de.nfev}  msg={de.message}")
        res = minimize(
            cost, de.x, method="L-BFGS-B", bounds=bounds,
            jac=lambda xx: parallel_fd_jacobian(cost, xx, workers=cp_map, bounds=bounds),
            options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-11},
        )
    finally:
        cp_map.close()
    x = res.x
    n_coeffs = 2 + 4 * order
    v_inf = x[n_coeffs:n_coeffs + 3]
    z0 = np.concatenate([r0, v_earth + v_inf])
    ref = heliocentric_orbit_normal(z0)
    r_miss, v_miss, _ = cruise_terminal_miss(
        x[:n_coeffs], z0, p["zT"], p["et0"], p["T_s"], ref, p["sail"],
        p["central_body"], p["third_bodies"],
    )
    ve_hat = v_earth / np.linalg.norm(v_earth)
    ang = math.degrees(math.acos(np.clip(np.dot(v_inf/np.linalg.norm(v_inf), ve_hat), -1, 1)))
    print(f"  polish cost={res.fun:.6e}")
    print(f"  chosen v_inf = {np.array2string(v_inf, precision=4)} km/s  "
          f"|v_inf|={np.linalg.norm(v_inf):.4f}  angle-to-prograde={ang:.1f} deg")
    print(f"  (escape v_inf: |1.168| km/s at 108 deg from prograde)")
    print(f"  terminal miss: r_miss={r_miss/AU_KM:.6f} AU ({r_miss:.1f} km)  v_miss={v_miss:.5f} km/s")
    out = SIMOUT / f"cruise_free_departure_K{order}_sigma{int(round(p['sigma_g']))}.json"
    out.write_text(json.dumps({
        "order": order, "sigma_g": p["sigma_g"], "coeffs": x[:n_coeffs].tolist(),
        "v_inf_kmps": v_inf.tolist(), "cost": float(res.fun),
        "r_miss_km": r_miss, "v_miss_kmps": v_miss,
    }, indent=2))
    print(f"  wrote {out}")
    return x


# ---------------------------------------------------------------------------
# Beta-homotopy continuation (lighter sail -> target sigma)
# ---------------------------------------------------------------------------


def continuation(start_coeffs, sigmas, order, workers=10,
                 clock_cap=8.0 * math.pi, polish_maxiter=400):
    """Track the cruise solution across decreasing authority (increasing sigma).

    Warm-started L-BFGS-B (parallel-FD) polish at each sigma in ``sigmas``,
    starting from ``start_coeffs`` (a lighter-sail solution). The clock-
    modulation structure carries from the higher-authority light-sail problem
    to the lower-authority sigma=18 case. The ``clock_cap`` bound may be widened
    for the heavier sail; slew feasibility remains enforced.
    """
    from scipy.optimize import minimize
    from reflectors.parallel import (
        CloudpickleMap,
        configure_multiprocessing_for_spice,
        parallel_fd_jacobian,
    )

    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    configure_multiprocessing_for_spice()
    load_kernels()
    cp_map = CloudpickleMap(n_workers=workers)
    x = np.asarray(start_coeffs, dtype=float)
    print(f"=== Beta-homotopy continuation (K={order}) ===")
    try:
        for sg in sigmas:
            p = build_problem(sigma_g=sg)
            bounds, omega_frame_max, _, clock_box = make_bounds(
                order, p["T_s"], p["mu"], p["z0"], limits, clock_cap=clock_cap
            )
            # Clip the warm start into the (possibly wider) bounds.
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            x = np.clip(x, lo, hi)
            cost = make_cruise_terminal_miss_cost(
                p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
                p["central_body"], p["third_bodies"],
                r_scale_km=p["r_scale"], v_scale_kmps=p["v_scale"], w_v=1.0,
            )
            res = minimize(
                cost, x, method="L-BFGS-B", bounds=bounds,
                jac=lambda xx: parallel_fd_jacobian(cost, xx, workers=cp_map, bounds=bounds),
                options={"maxiter": polish_maxiter, "ftol": 1e-14, "gtol": 1e-12},
            )
            x = res.x
            r_miss, v_miss, _ = cruise_terminal_miss(
                x, p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"],
                p["sail"], p["central_body"], p["third_bodies"],
            )
            b_omega, b_alpha = assert_cruise_slew_feasible(
                x, p["T_s"], limits, omega_frame_max_rad_s=omega_frame_max
            )
            print(f"  sigma={sg:5.1f}: cost={res.fun:.6e}  r_miss={r_miss/AU_KM:.6f} AU "
                  f"({r_miss:.1f} km)  v_miss={v_miss:.5f} km/s  "
                  f"|omega|/cap={b_omega/limits.omega_max_rad_s:.2e}  nit={res.nit}")
            out = _solution_path(order, sg)
            out.write_text(json.dumps({
                "order": order, "sigma_g": sg, "coeffs": x.tolist(),
                "cost": float(res.fun), "r_miss_km": r_miss, "v_miss_kmps": v_miss,
                "T_s": p["T_s"], "et0": p["et0"], "via": "continuation",
            }, indent=2))
    finally:
        cp_map.close()
    print(f"  final coeffs (sigma={sigmas[-1]}) = {np.array2string(x, precision=6, max_line_width=200)}")
    return x


# ---------------------------------------------------------------------------
# Two-resolution convergence and trajectory figure
# ---------------------------------------------------------------------------


def validate(p, coeffs, order):
    print(f"=== Validation (K={order}) ===")
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003), omega_max_rad_s=math.radians(0.3)
    )
    _, omega_frame_max, _, _ = make_bounds(order, p["T_s"], p["mu"], p["z0"], limits)
    b_omega, b_alpha = assert_cruise_slew_feasible(
        coeffs, p["T_s"], limits, omega_frame_max_rad_s=omega_frame_max
    )
    print(f"  slew feasible: |omega|<={math.degrees(b_omega):.3e} deg/s "
          f"({b_omega/limits.omega_max_rad_s:.2e} cap), "
          f"|alpha|<={math.degrees(b_alpha):.3e} deg/s^2 "
          f"({b_alpha/limits.alpha_max_rad_s2:.2e} cap)")

    # 2-resolution convergence: default vs finer steps_per_orbit.
    rA, vA, zA = cruise_terminal_miss(
        coeffs, p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
        p["central_body"], p["third_bodies"], steps_per_orbit=200,
    )
    rB, vB, zB = cruise_terminal_miss(
        coeffs, p["z0"], p["zT"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
        p["central_body"], p["third_bodies"], steps_per_orbit=800,
    )
    d_r = float(np.linalg.norm(zA[:3] - zB[:3]))
    d_v = float(np.linalg.norm(zA[3:6] - zB[3:6]))
    print("  2-resolution convergence (steps_per_orbit 200 vs 800):")
    print(f"    r_miss  200={rA:13.1f} km   800={rB:13.1f} km")
    print(f"    v_miss  200={vA:.6f} km/s   800={vB:.6f} km/s")
    print(f"    |zA-zB| terminal: d_r={d_r:.3f} km ({d_r/AU_KM:.3e} AU)  d_v={d_v:.6f} km/s")

    make_cruise_figure(p, coeffs, order)


def make_cruise_figure(p, coeffs, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = propagate_cruise(
        coeffs, p["z0"], p["et0"], p["T_s"], p["ref_normal"], p["sail"],
        p["central_body"], p["third_bodies"], steps_per_orbit=400,
    )
    arc = run.orbit_state_km_kmps[:, :3] / AU_KM

    # Earth + Mars heliocentric orbits over a window bracketing the transit.
    ets = np.linspace(p["start_et"] - 40 * 86400, p["end_et"] + 40 * 86400, 600)
    earth = np.array([spice.spkezr("399", e, "J2000", "NONE", "10")[0][:3] for e in ets]) / AU_KM
    mars = np.array([spice.spkezr("499", e, "J2000", "NONE", "10")[0][:3] for e in ets]) / AU_KM

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(earth[:, 0], earth[:, 1], color="tab:blue", lw=0.8, alpha=0.6, label="Earth orbit")
    ax.plot(mars[:, 0], mars[:, 1], color="tab:red", lw=0.8, alpha=0.6, label="Mars orbit")
    ax.plot(arc[:, 0], arc[:, 1], color="black", lw=1.8, label="SRP-sail cruise")
    ax.scatter([0], [0], color="gold", s=200, marker="*", zorder=5, label="Sun")
    start_date = spice.et2utc(float(p["start_et"]), "ISOC", 0)[:10]
    end_date = spice.et2utc(float(p["end_et"]), "ISOC", 0)[:10]
    ax.scatter(p["z0"][0] / AU_KM, p["z0"][1] / AU_KM, color="tab:blue", s=60,
               zorder=6, edgecolor="k", label=f"Earth escape exit ({start_date})")
    ax.scatter(p["zT"][0] / AU_KM, p["zT"][1] / AU_KM, color="tab:red", s=60,
               zorder=6, edgecolor="k", label=f"Mars capture node ({end_date})")
    ax.scatter(arc[-1, 0], arc[-1, 1], color="black", s=40, marker="x", zorder=6,
               label="cruise terminus")
    ax.set_xlabel("x [AU] (J2000)")
    ax.set_ylabel("y [AU] (J2000)")
    ax.set_title(f"Interplanetary SRP-sail cruise (sigma={int(round(p['sigma_g']))}, "
                 f"K={order}, T={p['T_s'] / 86400.0:.1f} d)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig_dir = SIMOUT / "interplanetary_cruise"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"cruise_heliocentric_sigma{int(round(p['sigma_g']))}_K{order}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", type=int, default=1, help="Fourier order K")
    ap.add_argument("--solve", action="store_true", help="run the DE+FD solve")
    ap.add_argument("--validate", action="store_true",
                    help="2-resolution convergence + figure for the saved solution")
    ap.add_argument("--continue-from-sigma", type=float, default=None,
                    help="beta-homotopy: start from this sigma's saved solution and continue")
    ap.add_argument("--continue-sigmas", type=str, default="8,10,12,14,16,18",
                    help="comma-separated sigma sequence for the continuation")
    ap.add_argument("--seed-from-sigmas", type=str, default=None,
                    help="DE: comma-separated sigmas whose saved solutions seed the population")
    ap.add_argument("--clock-cap", type=float, default=None,
                    help="override the clock harmonic box cap (rad); default 2pi (DE)")
    ap.add_argument("--ipopt", action="store_true",
                    help="IPOPT equality-constrained single-shooting solve")
    ap.add_argument("--warm-from-sigma", type=float, default=None,
                    help="IPOPT: warm-start coeffs from this sigma's saved solution")
    ap.add_argument("--warm-pad-from-order", type=int, default=None,
                    help="IPOPT: read the warm solution at this (lower) order and zero-pad to --order")
    ap.add_argument("--ipopt-max-iter", type=int, default=400)
    ap.add_argument("--free-departure", action="store_true",
                    help="diagnostic: also optimise the departure v_inf")
    ap.add_argument("--vinf-max", type=float, default=2.5,
                    help="free-departure: max |v_inf| component (km/s)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--de-maxiter", type=int, default=200)
    ap.add_argument("--de-popsize", type=int, default=15)
    ap.add_argument("--de-seed", type=int, default=42)
    ap.add_argument("--max-step-s", type=float, default=None)
    ap.add_argument("--sigma-g", type=float, default=18.0)
    ap.add_argument("--perturbers", type=str, default="earth,moon,mars",
                    help="third bodies: 'earth,moon,mars' | 'mars' | 'none' (solar gravity only)")
    ap.add_argument("--departure", type=str, default="hill_exit",
                    help="'hill_exit' (escape state) | 'earth_exact' (Earth pos+vel reference)")
    ap.add_argument("--earth-summary", type=Path, required=True,
                    help="Earth-escape Hill-exit summary produced by "
                         "run_srp_escape_earth.py")
    ap.add_argument("--node-json", type=Path, required=True,
                    help="Mars capture-node JSON produced by run_srp_capture.py")
    args = ap.parse_args()

    configure_endpoint_paths(args.earth_summary, args.node_json)
    SIMOUT.mkdir(exist_ok=True)

    p = build_problem(sigma_g=args.sigma_g, perturbers=args.perturbers,
                      departure=args.departure)
    print(f"START helio |r|={np.linalg.norm(p['z0'][:3])/AU_KM:.5f} AU  "
          f"|v|={np.linalg.norm(p['z0'][3:6]):.4f} km/s")
    print(f"END   helio |r|={np.linalg.norm(p['zT'][:3])/AU_KM:.5f} AU  "
          f"|v|={np.linalg.norm(p['zT'][3:6]):.4f} km/s")
    if args.ipopt:
        warm = None
        if args.warm_from_sigma is not None:
            src_order = args.warm_pad_from_order or args.order
            warm = np.asarray(json.loads(
                _solution_path(src_order, args.warm_from_sigma).read_text())["coeffs"],
                dtype=float)
            if src_order != args.order:
                warm = _pad_coeffs(warm, src_order, args.order)
        solve_ipopt(p, order=args.order, workers=args.workers,
                    max_iter=args.ipopt_max_iter, warm_coeffs=warm,
                    free_departure=args.free_departure, vinf_max=args.vinf_max)
    elif args.free_departure:
        solve_free_departure(p, order=args.order, workers=args.workers,
                             vinf_max=args.vinf_max, de_maxiter=args.de_maxiter,
                             de_popsize=args.de_popsize)
    elif args.continue_from_sigma is not None:
        start = json.loads(_solution_path(args.order, args.continue_from_sigma).read_text())
        sigmas = [float(s) for s in args.continue_sigmas.split(",")]
        continuation(np.asarray(start["coeffs"], dtype=float), sigmas,
                     args.order, workers=args.workers)
    elif args.solve:
        seed_coeffs = None
        if args.seed_from_sigmas:
            seed_coeffs = [
                np.asarray(json.loads(_solution_path(args.order, float(s)).read_text())["coeffs"],
                           dtype=float)
                for s in args.seed_from_sigmas.split(",")
            ]
        clock_cap = args.clock_cap if args.clock_cap is not None else CLOCK_HARMONIC_CAP_RAD
        solve(p, order=args.order, workers=args.workers,
              de_seed=args.de_seed, de_maxiter=args.de_maxiter,
              de_popsize=args.de_popsize, max_step_s=args.max_step_s,
              seed_coeffs=seed_coeffs, clock_cap=clock_cap)
    elif args.validate:
        sol = json.loads(_solution_path(args.order, args.sigma_g).read_text())
        validate(p, np.asarray(sol["coeffs"], dtype=float), args.order)
    else:
        forward_feasibility(p, order=args.order)


if __name__ == "__main__":
    main()
