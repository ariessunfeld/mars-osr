"""Drag-aware dE/dt steering.

``dedot_steer(drag=DragMeritContext(...))`` switches the merit to the net energy
rate ``(a_SRP + a_drag).v_hat``; ``drag=None`` selects the SRP-only law.
These tests pin: (1) the toggle COLLAPSES to SRP-only when the drag context's
density is zero (bit-identical search), (2) it actually CHANGES the command when
drag is material (low altitude), (3) it requires the epoch for co-rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.atmosphere import ExponentialAtmosphere, HarrisPriester
from reflectors.central_body import earth_central_body
from reflectors.escape_dedot import DEdotParams, DragMeritContext, dedot_steer
from reflectors.ephemeris import utc_to_et
from reflectors.sail_designs import make_canonical_sail
from reflectors.solar_constants import AU_KM, solar_flux_at

EPOCH = "2028-01-01T00:00:00"


def _setup(alt_km):
    earth = earth_central_body()
    et = utc_to_et(EPOCH)
    mu = earth.mu_km3_s2
    a = earth.equatorial_radius_km + alt_km
    r = np.array([a, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(mu / a), 0.0])  # prograde circular
    s_hat = np.array([0.3, 0.8, 0.5])
    s_hat = s_hat / np.linalg.norm(s_hat)
    P = solar_flux_at(AU_KM)  # ~1 AU SRP
    sail = make_canonical_sail(0.018)
    params = DEdotParams(mu_km3_s2=mu)
    return earth, et, r, v, s_hat, P, sail, params


def test_zero_density_collapses_to_srp_only():
    """Toggle ON but density 0 -> a_drag is identically zero -> the merit and
    the whole search are bit-identical to SRP-only (drag=None)."""
    earth, et, r, v, s_hat, P, sail, params = _setup(400.0)
    srp_only = dedot_steer(r, v, s_hat, P, sail, params=params)
    zero_ctx = DragMeritContext(
        density_model=ExponentialAtmosphere(0.0, 100.0, 0.0),
        central_body=earth, C_d=2.2,
    )
    collapsed = dedot_steer(r, v, s_hat, P, sail, params=params, drag=zero_ctx, et=et)
    assert np.allclose(collapsed.n_star_j2000, srp_only.n_star_j2000, atol=1e-12)
    assert collapsed.thrust == srp_only.thrust


def test_drag_aware_changes_command_at_low_altitude():
    """Where drag is material (700 km, sigma=18), the drag-aware command differs
    from the SRP-only optimum -- the toggle has a real effect."""
    earth, et, r, v, s_hat, P, sail, params = _setup(700.0)
    srp_only = dedot_steer(r, v, s_hat, P, sail, params=params)
    ctx = DragMeritContext(density_model=HarrisPriester(), central_body=earth, C_d=2.2)
    drag_aware = dedot_steer(r, v, s_hat, P, sail, params=params, drag=ctx, et=et)
    cos_ang = float(np.clip(
        np.dot(srp_only.n_star_j2000, drag_aware.n_star_j2000), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_ang))
    assert angle_deg > 1.0, f"drag-aware command barely moved ({angle_deg:.3f} deg)"


def test_drag_aware_sheds_projected_area():
    """The drag-aware orientation presents NO MORE area into the flow than the
    SRP-only one (it tilts to reduce drag, never to increase it)."""
    earth, et, r, v, s_hat, P, sail, params = _setup(600.0)
    v_hat = v / np.linalg.norm(v)
    srp_only = dedot_steer(r, v, s_hat, P, sail, params=params)
    ctx = DragMeritContext(density_model=HarrisPriester(), central_body=earth, C_d=2.2)
    drag_aware = dedot_steer(r, v, s_hat, P, sail, params=params, drag=ctx, et=et)
    proj_srp = abs(float(np.dot(srp_only.n_star_j2000, v_hat)))
    proj_drag = abs(float(np.dot(drag_aware.n_star_j2000, v_hat)))
    assert proj_drag <= proj_srp + 1e-9


def test_drag_aware_requires_epoch():
    earth, et, r, v, s_hat, P, sail, params = _setup(400.0)
    ctx = DragMeritContext(density_model=HarrisPriester(), central_body=earth, C_d=2.2)
    with pytest.raises(ValueError):
        dedot_steer(r, v, s_hat, P, sail, params=params, drag=ctx)  # no et
