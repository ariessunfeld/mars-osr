"""Fast regression tests for optimizer-to-scheduler omega-limit plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import reflectors.optimize as optimize


def _config(*, omega_max_rad_s: float | None) -> optimize.OrbitConfig:
    return optimize.OrbitConfig(
        a_km=3900.0,
        ltan_h=18.0,
        M0_rad=0.0,
        epoch_et=1000.0,
        duration_s=1000.0,
        sail=object(),
        target_lat_deg=40.0,
        target_lon_deg=200.0,
        alpha_max_rad_s2=5.0e-5,
        omega_max_rad_s=omega_max_rad_s,
        initial_state_override_km_kmps=np.array(
            [3900.0, 0.0, 0.0, 0.0, 3.3, 0.0], dtype=float,
        ),
    )


def test_orbit_config_rejects_invalid_omega_limit():
    with pytest.raises(ValueError, match="omega_max_rad_s"):
        _config(omega_max_rad_s=0.0)


def test_handoff_factory_forwards_configured_omega_limit(monkeypatch):
    captured: dict[str, float | None] = {}

    def fake_cruise_to_cruise_slew(*args, **kwargs):
        captured["omega_max_rad_s"] = kwargs["omega_max_rad_s"]
        return (
            lambda _r, _et: np.array([1.0, 0.0, 0.0]),
            SimpleNamespace(t_end_et=1060.0),
        )

    monkeypatch.setattr(
        optimize, "cruise_to_cruise_slew", fake_cruise_to_cruise_slew,
    )
    omega_max = 0.005
    config = _config(omega_max_rad_s=omega_max)
    cruise = lambda _r, _et: np.array([1.0, 0.0, 0.0])
    base_factory = lambda _x, _config: cruise
    factory = optimize.make_handoff_cruise_factory(
        base_factory,
        cruise,
        central_body_gm_km3_s2=42828.0,
    )
    profile = factory(np.zeros(1), config)
    np.testing.assert_array_equal(
        profile(config.initial_state_km_kmps[:3], 1000.0),
        np.array([1.0, 0.0, 0.0]),
    )
    assert captured["omega_max_rad_s"] == omega_max
