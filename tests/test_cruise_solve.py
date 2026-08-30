"""Wiring + convergence test for the piecewise-RTN IPOPT cruise solver
(reflectors.cruise_solve).

A SELF-CONSISTENT close: propagate a feasible piecewise command from a synthetic
Sun-centred state to obtain a reachable-by-construction target, then solve the
6-equality NLP from a cold start back to that target. Proves the full stack
(objective + analytic gradient + constraints + parallel-FD Jacobian + cyipopt)
wires together and drives the scaled defect down by orders of magnitude. Uses a
serial map (no worker fork) for determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.central_body import sun_central_body
from reflectors.cruise_command import heliocentric_orbit_normal
from reflectors.cruise_piecewise import (
    make_piecewise_cruise_defect,
    pack_angles,
    piecewise_cruise_terminal_miss,
    unpack_angles,
)
from reflectors.cruise_solve import angle_box_bounds, solve_piecewise_ipopt
from reflectors.dynamics import sun_gm_km3_per_s2
from reflectors.sail_designs import make_canonical_sail

AU_KM = 1.495978707e8

cyipopt = pytest.importorskip("cyipopt")


def _serial_map(fn, items):
    return [fn(x) for x in items]


def test_piecewise_ipopt_closes_self_consistent_target():
    mu = sun_gm_km3_per_s2()
    z0 = np.array([AU_KM, 0.0, 0.0, 0.0, math.sqrt(mu / AU_KM), 0.0])
    ref = heliocentric_orbit_normal(z0)
    sail = make_canonical_sail(0.018)
    cb = sun_central_body()
    tb = ()  # pure two-body Sun + SRP (fast, no SPICE third-body calls)
    et0 = 0.0
    T_s = 120.0 * 86400.0
    N = 4
    spo = 40

    # A feasible seed command -> its terminal state is the reachable target.
    seed = pack_angles(np.full(N, 0.15), np.full(N, 0.05))
    sp, st = unpack_angles(seed, N)
    _, _, z_tgt = piecewise_cruise_terminal_miss(
        sp, st, z0, z0, et0, T_s, ref, sail, cb, tb, steps_per_orbit=spo
    )

    r_scale, v_scale = AU_KM, math.sqrt(mu / AU_KM)
    defect = make_piecewise_cruise_defect(
        z0, z_tgt, et0, T_s, ref, sail, cb, tb,
        N=N, r_scale_km=r_scale, v_scale_kmps=v_scale, steps_per_orbit=spo,
    )
    lb, ub = angle_box_bounds(N)
    x0 = np.zeros(2 * N)
    g0 = float(np.linalg.norm(defect(x0)))  # cold-start defect norm

    x_best, info = solve_piecewise_ipopt(
        defect, 2 * N, N, lb=lb, ub=ub, x0=x0, cp_map=_serial_map,
        max_iter=80, tol=1e-9, constr_viol_tol=1e-9,
    )
    # The stack closed: the best defect dropped by >= 3 orders vs the cold start,
    # and the absolute terminal miss is small (sub-1000 km / sub-m/s on a
    # reachable target).
    assert info["best_defect_norm"] < g0 / 1e3
    phis, thetas = unpack_angles(x_best, N)
    r_miss, v_miss, _ = piecewise_cruise_terminal_miss(
        phis, thetas, z0, z_tgt, et0, T_s, ref, sail, cb, tb, steps_per_orbit=spo
    )
    assert r_miss < 1.0e3
    assert v_miss < 1.0e-3
