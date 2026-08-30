"""Regression tests for ``propagate(ephemeris_time_direction=...)``.

The backward clock (``ephemeris_time_direction=-1``) evaluates every
ephemeris-dependent term (IAU_MARS frame rotation for gravity harmonics, Sun /
moon third-body positions, SRP Sun) at ``epoch_et - t`` instead of
``epoch_et + t``. Positive signal: for time-reversible physics (gravity + third
body, no SRP), a forward run followed by a backward run started from the
velocity-negated end state must retrace to the start state. An incorrect clock
(a term still evaluated forward) puts the Sun / Mars orientation at an
inconsistent epoch and produces a retrace error of many kilometres.

Default (+1) must reproduce forward-only propagation exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from reflectors.dynamics import PropagationOptions, propagate
from reflectors.ephemeris import utc_to_et
from reflectors.kernels import load_kernels
from reflectors.third_body import sun_third_body


EPOCH_UTC = "2028-02-11T12:42:00"
# Circular ~508 km Mars orbit, inclined (so the J2 / IAU_MARS rotation is
# actually exercised, not a degenerate equatorial case).
A_KM = 3903.924477
MU_KM3_S2 = 42828.375816  # MRO120F Mars-system GM (matches gravity_degree=2 model)
_TIGHT = PropagationOptions(method="DOP853", rtol=1e-12, atol=1e-14)


def _circular_state():
    v_circ = np.sqrt(MU_KM3_S2 / A_KM)
    inc = np.radians(93.42)
    r = np.array([A_KM, 0.0, 0.0])
    v = np.array([0.0, v_circ * np.cos(inc), v_circ * np.sin(inc)])
    return np.concatenate([r, v])


def _propagate(state0, t_span, epoch_et, time_dir):
    return propagate(
        state0, t_span,
        epoch_et=epoch_et,
        gravity_degree=2, gravity_order=2,
        third_bodies=[sun_third_body()],
        options=_TIGHT,
        ephemeris_time_direction=time_dir,
    )


def test_backward_clock_retraces_reversible_trajectory():
    """Forward, then backward-from-reversed-end-state, retraces to the start.

    Gravity(2) + Sun third body is time-reversible under (v -> -v, clock
    reversed). The backward run starts at ephemeris ``E0 + T`` (= the forward
    end epoch) with ``time_dir=-1`` so its clock walks E0+T -> E0, matching the
    forward run's ephemeris sequence in reverse.
    """
    load_kernels()
    e0 = utc_to_et(EPOCH_UTC)
    state0 = _circular_state()
    t_final = 3000.0  # partial orbit; long enough for J2 + Sun to bite

    fwd = _propagate(state0, (0.0, t_final), e0, +1)
    end = np.asarray(fwd.state_km_kmps)[-1]
    assert np.isclose(fwd.t_s[-1], t_final)

    rev0 = np.concatenate([end[:3], -end[3:]])
    rev = _propagate(rev0, (0.0, t_final), e0 + t_final, -1)
    back = np.asarray(rev.state_km_kmps)[-1]

    # Retrace: position back to r0, velocity back to -v0.
    pos_err = np.linalg.norm(back[:3] - state0[:3])
    vel_err = np.linalg.norm(back[3:] - (-state0[3:]))
    assert pos_err < 1e-4, f"position retrace error {pos_err:.3e} km too large"
    assert vel_err < 1e-7, f"velocity retrace error {vel_err:.3e} km/s too large"


def test_backward_clock_actually_differs_from_forward():
    """Positive signal that the clock sign matters: running the reversed end
    state with the forward clock does not retrace -- confirming the
    retrace above is due to the ephemeris flip, not a trivial symmetry."""
    load_kernels()
    e0 = utc_to_et(EPOCH_UTC)
    state0 = _circular_state()
    t_final = 3000.0

    fwd = _propagate(state0, (0.0, t_final), e0, +1)
    end = np.asarray(fwd.state_km_kmps)[-1]
    rev0 = np.concatenate([end[:3], -end[3:]])

    forward_clock_result = _propagate(
        rev0, (0.0, t_final), e0 + t_final, +1
    )
    back = np.asarray(forward_clock_result.state_km_kmps)[-1]
    pos_err = np.linalg.norm(back[:3] - state0[:3])
    # Over this short (3000 s) baseline the forward vs backward ephemeris differ
    # only modestly, so the forward clock leaves ~0.8 km of retrace error -- small
    # in absolute terms but ~4 orders of magnitude larger than the correct
    # clock's <1e-4 km retrace. The 0.1 km threshold lies between the
    # two, so it fails if the flip stops taking effect (error -> ~1e-4).
    assert pos_err > 0.1, (
        f"forward-clock retrace error {pos_err:.3e} km unexpectedly small; "
        "the ephemeris_time_direction flip is not taking effect"
    )


def test_forward_default_is_bit_exact():
    """Explicit ephemeris_time_direction=+1 reproduces the default path exactly."""
    load_kernels()
    e0 = utc_to_et(EPOCH_UTC)
    state0 = _circular_state()
    t_final = 3000.0

    default = propagate(
        state0, (0.0, t_final), epoch_et=e0,
        gravity_degree=2, gravity_order=2,
        third_bodies=[sun_third_body()], options=_TIGHT,
    )
    explicit = _propagate(state0, (0.0, t_final), e0, +1)
    a = np.asarray(default.state_km_kmps)[-1]
    b = np.asarray(explicit.state_km_kmps)[-1]
    assert np.array_equal(a, b), "explicit +1 must be bit-exact with the default"


@pytest.mark.parametrize("bad", [0, 2, -2, +2])
def test_invalid_time_direction_raises(bad):
    load_kernels()
    e0 = utc_to_et(EPOCH_UTC)
    with pytest.raises(ValueError, match="ephemeris_time_direction"):
        propagate(
            _circular_state(), (0.0, 100.0), epoch_et=e0,
            gravity_degree=2, third_bodies=[sun_third_body()],
            ephemeris_time_direction=bad,
        )
