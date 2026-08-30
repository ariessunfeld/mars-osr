"""Exact same-RHS SPICE sharing and uncached-oracle regressions."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

import reflectors.ephemeris as ephemeris
from reflectors.attitude import sun_pointing
from reflectors.dynamics import PropagationOptions, propagate
from reflectors.gravity import mars_gravity_model
from reflectors.sail_designs import make_canonical_sail
from reflectors.third_body import (
    deimos_third_body,
    phobos_third_body,
    sun_third_body,
)


def test_full_rhs_shares_sun_exactly_and_matches_uncached_oracle(monkeypatch):
    """Three identical Sun requests become one without changing the state.

    Each RHS evaluation requests Sun once for third-body gravity, once for SRP,
    and once for Sun-pointing attitude, plus one request for each Mars moon.
    Thus the uncached path makes five ``spkezr`` calls per RHS while the exact
    cache makes three. Gravity contributes one unique ``pxform`` request.
    """
    original_spkezr = spice.spkezr
    original_pxform = spice.pxform
    raw_state_calls = 0
    raw_rotation_calls = 0

    def counted_spkezr(*args):
        nonlocal raw_state_calls
        raw_state_calls += 1
        return original_spkezr(*args)

    def counted_pxform(*args):
        nonlocal raw_rotation_calls
        raw_rotation_calls += 1
        return original_pxform(*args)

    monkeypatch.setattr(ephemeris.spice, "spkezr", counted_spkezr)
    monkeypatch.setattr(ephemeris.spice, "pxform", counted_pxform)

    model = mars_gravity_model(max_degree=4)
    radius_km = model.ref_radius_km + 400.0
    speed_km_s = float(np.sqrt(model.mu_km3_s2 / radius_km))
    et0 = spice.str2et("2026-06-01T00:00:00")
    sun_state, _ = original_spkezr("10", et0, "J2000", "NONE", "499")
    sun_hat = np.asarray(sun_state[:3], dtype=float)
    sun_hat /= np.linalg.norm(sun_hat)
    velocity_hat = np.cross(sun_hat, np.array([0.0, 0.0, 1.0]))
    velocity_hat /= np.linalg.norm(velocity_hat)
    # Start on Mars's subsolar side so the SRP path reaches (and exercises)
    # the Sun-pointing attitude instead of returning early from the umbra gate.
    state0 = np.concatenate(
        (radius_km * sun_hat, speed_km_s * velocity_hat)
    )
    common = dict(
        epoch_et=et0,
        gravity_degree=4,
        gravity_order=4,
        third_bodies=(
            sun_third_body(),
            phobos_third_body(),
            deimos_third_body(),
        ),
        solar_sail=make_canonical_sail(0.018),
        sail_normal=sun_pointing(),
        options=PropagationOptions.fast(),
        t_eval_s=np.array([0.0, 60.0]),
    )

    raw_state_calls = 0
    raw_rotation_calls = 0
    uncached = propagate(
        state0, (0.0, 60.0), share_ephemeris=False, **common
    )
    uncached_state_calls = raw_state_calls
    uncached_rotation_calls = raw_rotation_calls
    assert uncached_state_calls == 5 * uncached.n_rhs_calls
    assert uncached_rotation_calls == uncached.n_rhs_calls

    raw_state_calls = 0
    raw_rotation_calls = 0
    cached = propagate(
        state0, (0.0, 60.0), share_ephemeris=True, **common
    )

    assert cached.n_rhs_calls == uncached.n_rhs_calls
    assert np.array_equal(cached.t_s, uncached.t_s)
    assert np.array_equal(cached.state_km_kmps, uncached.state_km_kmps)
    assert raw_state_calls == 3 * cached.n_rhs_calls
    assert raw_rotation_calls == cached.n_rhs_calls

    stats = cached.metadata["ephemeris_cache"]
    assert stats["contexts"] == cached.n_rhs_calls
    assert stats["state_requests"] == 5 * cached.n_rhs_calls
    assert stats["state_cache_hits"] == 2 * cached.n_rhs_calls
    assert stats["state_spice_calls"] == 3 * cached.n_rhs_calls
    assert stats["rotation_requests"] == cached.n_rhs_calls
    assert stats["rotation_cache_hits"] == 0
    assert stats["rotation_inverse_hits"] == 0
    assert stats["rotation_spice_calls"] == cached.n_rhs_calls
    assert uncached.metadata["ephemeris_cache"] == {"enabled": False}
