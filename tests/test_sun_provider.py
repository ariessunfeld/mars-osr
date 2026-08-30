"""Tests for the injectable Sun provider (reflectors.ephemeris.sun_state_j2000).

The cruise/illumination pipeline routes every Mars->Sun query through
``sun_state_j2000`` so the ablation Sun-models (no-obliquity / circular) apply
consistently. These tests pin:

  1. Exact passthrough when no synthetic model is active.
  2. Positive signals for each synthetic mode (zero_obliquity -> sub-solar
     latitude 0; circular -> constant Sun distance, direction preserved).
  3. Context-manager restore.
  4. FORK inheritance of the process-global override (the parallel-DE / FD pools
     fork to inherit the SPICE pool; the override must ride along).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest
import spiceypy as spice

from reflectors.ephemeris import (
    active_sun_model,
    sun_model,
    sun_state_j2000,
    utc_to_et,
)
from reflectors.mars_constants import MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM

# UTC epochs spanning the Mars year (perihelion-2028 anchor + offsets). Kept as
# strings and converted to ET INSIDE each test -- module-level conversion would
# run at collection time, before conftest furnishes the leapseconds kernel.
_EPOCHS_UTC = [
    "2028-02-11T12:42:00",  # baseline perihelion anchor
    "2028-08-01T00:00:00",  # ~half-Mars-year offset
    "2029-01-01T00:00:00",
]
_OBSERVERS = ["MARS", 499]


# ---------------------------------------------------------------------------
# 1. Bit-exact passthrough
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("utc", _EPOCHS_UTC)
@pytest.mark.parametrize("observer", _OBSERVERS)
def test_passthrough_bit_exact(utc, observer):
    """With no active model, sun_state_j2000 equals the raw SPICE state."""
    et = utc_to_et(utc)
    assert active_sun_model() is None
    got = sun_state_j2000(et, observer)
    ref, _ = spice.spkezr("10", float(et), "J2000", "NONE", str(observer))
    ref = np.asarray(ref, dtype=float)
    assert np.array_equal(got, ref)  # bit-exact, not just close


def test_real_mode_is_passthrough():
    """sun_model('real') is a no-op (bit-exact passthrough, no override)."""
    et = utc_to_et(_EPOCHS_UTC[0])
    with sun_model("real"):
        assert active_sun_model() is None
        got = sun_state_j2000(et, "MARS")
    ref, _ = spice.spkezr("10", float(et), "J2000", "NONE", "MARS")
    assert np.array_equal(got, np.asarray(ref, dtype=float))


# ---------------------------------------------------------------------------
# 2. Synthetic-mode positive signals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("utc", _EPOCHS_UTC)
def test_zero_obliquity_sub_solar_latitude_is_zero(utc):
    """zero_obliquity puts the Sun in Mars' equatorial plane (IAU_MARS z ~ 0),
    preserving the Sun distance."""
    et = utc_to_et(utc)
    real = sun_state_j2000(et, "MARS")
    real_dist = float(np.linalg.norm(real[:3]))
    with sun_model("zero_obliquity"):
        synth = sun_state_j2000(et, "MARS")
    # distance preserved
    assert np.isclose(np.linalg.norm(synth[:3]), real_dist, rtol=1e-12)
    # sub-solar latitude zero: z-component in the Mars body frame ~ 0
    R_j2000_to_body = np.asarray(spice.pxform("J2000", "IAU_MARS", float(et)))
    r_body = R_j2000_to_body @ synth[:3]
    assert abs(r_body[2]) / real_dist < 1e-9
    # real Sun is NOT in the equatorial plane (positive control: obliquity real)
    r_body_real = R_j2000_to_body @ real[:3]
    assert abs(r_body_real[2]) / real_dist > 1e-3


@pytest.mark.parametrize("utc", _EPOCHS_UTC)
def test_circular_holds_distance_keeps_direction(utc):
    """circular sets |r| = a_Mars and keeps the real Sun direction."""
    et = utc_to_et(utc)
    real = sun_state_j2000(et, "MARS")
    real_dir = real[:3] / np.linalg.norm(real[:3])
    with sun_model("circular"):
        synth = sun_state_j2000(et, "MARS")
    assert np.isclose(np.linalg.norm(synth[:3]),
                      MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM, rtol=1e-12)
    synth_dir = synth[:3] / np.linalg.norm(synth[:3])
    assert np.allclose(synth_dir, real_dir, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Context-manager restore + validation
# ---------------------------------------------------------------------------

def test_context_manager_restores():
    et = utc_to_et(_EPOCHS_UTC[0])
    assert active_sun_model() is None
    with sun_model("circular"):
        assert active_sun_model() == {"mode": "circular"}
    assert active_sun_model() is None
    # passthrough restored bit-exactly
    ref, _ = spice.spkezr("10", float(et), "J2000", "NONE", "MARS")
    assert np.array_equal(sun_state_j2000(et, "MARS"), np.asarray(ref, dtype=float))


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        with sun_model("not_a_mode"):
            pass


# ---------------------------------------------------------------------------
# 4. Fork inheritance (parallel-DE workers inherit the override)
# ---------------------------------------------------------------------------

def _child_sun_state(et, observer, q):
    # Runs in a forked child: must see the parent's active override + kernels.
    q.put(np.asarray(sun_state_j2000(et, observer)))


def test_fork_inherits_override():
    """A forked child sees the override set in the parent before the fork --
    the invariant the parallel-DE pools rely on (fork after sun_model is set)."""
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        pytest.skip("fork start method unavailable on this platform")
    et, observer = utc_to_et(_EPOCHS_UTC[0]), "MARS"
    with sun_model("circular"):
        parent_val = np.asarray(sun_state_j2000(et, observer))
        q = ctx.Queue()
        p = ctx.Process(target=_child_sun_state, args=(et, observer, q))
        p.start()
        child_val = q.get(timeout=30)
        p.join(timeout=30)
    assert np.array_equal(child_val, parent_val)
    # and the child saw the SYNTHETIC (circular) Sun, not the real one
    assert np.isclose(np.linalg.norm(child_val[:3]),
                      MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM, rtol=1e-12)
