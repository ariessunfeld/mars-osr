"""Anticipatory feed-forward lead in ``propagate_escape``.

The bounded-slew tracker reacts to the present steering command, so at a
feather<->thrust / shadow transition the sail begins slewing only at the
transition and spends the slew applying force in the opposing direction. The
anticipatory lead re-evaluates the command ``anticipatory_lead_s`` ahead
(2-body Kepler) and, when a large reorientation is imminent, begins the slew
early so the sail arrives in the correct orientation by the transition.

These tests pin the two contract properties:
  1. Positive signal -- with the lead on, the achieved normal is much closer
     to the post-transition command at the transition time than the reactive
     (lead-off) tracker, i.e. it demonstrably pre-slewed.
  2. REDUCES-TO-BASE -- with the activation threshold above the largest
     command jump (so the gate never fires), the trajectory is bit-identical
     to lead-off. Guards against the lead perturbing smooth arcs.
"""
from __future__ import annotations

import math

import numpy as np

from reflectors.attitude_control import AttitudeLimits
from reflectors.dynamics import mars_gm_km3_per_s2
from reflectors.ephemeris import utc_to_et
from reflectors.escape import propagate_escape
from reflectors.kernels import load_kernels
from reflectors.mars_constants import MARS_HILL_RADIUS_KM, SECONDS_PER_SOLAR_SOL_S
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail

# Two fixed inertial unit normals 90 deg apart -- the scripted command flips
# from N_A to N_B at T_FLIP. 90 deg >> the 20 deg default activation gate.
N_A = np.array([1.0, 0.0, 0.0])
N_B = np.array([0.0, 1.0, 0.0])
T_FLIP_S = 2000.0
LEAD_S = 700.0  # ~ a 90 deg slew time at omega_max=0.3 deg/s


def _scripted_steering(r, v, s_hat, p_eff, sail_, current_n_hat, et):
    """Command flips N_A -> N_B at absolute epoch et = epoch0 + T_FLIP_S.

    epoch0 is captured by the harness via the closure set in each test.
    """
    return N_B if (et - _scripted_steering.epoch0) >= T_FLIP_S else N_A


def _run(lead_s, activate_deg, epoch_et):
    _scripted_steering.epoch0 = epoch_et
    state0 = np.array([4396.0, 0.0, 0.0, 0.0, 3.121, 0.0])  # ~1000 km LMO, circular-ish
    sail = make_canonical_sail(0.018)
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )
    qlaw_shell = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=3696.19)
    return propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits,
        (0.0, 3500.0),
        gravity_degree=0,
        include_sun_third_body=False,
        steering_fn=_scripted_steering,
        anticipatory_lead_s=lead_s,
        anticipatory_activate_deg=activate_deg,
    )


def _normal_at(res, t_target):
    i = int(np.argmin(np.abs(np.asarray(res.t_s) - t_target)))
    n = res.attitude_state[i, :3]
    return n / float(np.linalg.norm(n))


def test_anticipatory_lead_preslews_before_scripted_flip():
    """With lead on, the sail is already swinging toward N_B at the flip."""
    load_kernels()
    epoch_et = utc_to_et("2028-01-01T00:00:00")

    res_reactive = _run(0.0, 20.0, epoch_et)
    res_lead = _run(LEAD_S, 20.0, epoch_et)

    # Angle of the achieved normal to the POST-flip command N_B, evaluated AT
    # the flip instant. Reactive: still ~at N_A (90 deg). Lead: well advanced.
    n_react = _normal_at(res_reactive, T_FLIP_S)
    n_lead = _normal_at(res_lead, T_FLIP_S)
    ang_react = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(n_react, N_B))))))
    ang_lead = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(n_lead, N_B))))))

    # Reactive tracker has not begun the slew at the flip (still near N_A, 90 deg).
    assert ang_react > 70.0, f"reactive should still be near N_A, got {ang_react:.1f} deg"
    # Anticipatory tracker pre-slewed: substantially closer to N_B and
    # ahead of the reactive one by a wide margin.
    assert ang_lead < 35.0, f"lead should be well toward N_B, got {ang_lead:.1f} deg"
    assert ang_react - ang_lead > 40.0, (
        f"lead should lead reactive by a wide margin: "
        f"reactive {ang_react:.1f} vs lead {ang_lead:.1f} deg"
    )


def test_anticipatory_lead_dormant_when_threshold_high_matches_base():
    """activate_deg above the 90 deg jump -> gate never fires -> == base."""
    load_kernels()
    epoch_et = utc_to_et("2028-01-01T00:00:00")

    res_base = _run(0.0, 20.0, epoch_et)
    res_dormant = _run(LEAD_S, 179.0, epoch_et)  # gate never fires (jump is 90 deg)

    a_base = res_base.attitude_state
    a_dorm = res_dormant.attitude_state
    assert a_base.shape == a_dorm.shape
    # Bit-identical attitude history: the dormant lead must not perturb anything.
    assert np.allclose(a_base, a_dorm, atol=1e-12, rtol=0.0)
