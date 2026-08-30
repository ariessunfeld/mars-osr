"""Tests for ``reflectors.atmosphere`` (Harris-Priester + exospheric tail).

Pins the M&G Table 3.8 units conversion, the diurnal bulge bounds
(M&G Eq. 3.103-3.105), and the required >1000 km tail through the
NRLMSIS-2.1-calibrated exospheric scale height (verified against pymsis).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflectors.atmosphere import (
    ExponentialAtmosphere,
    HarrisPriester,
    _bulge_apex_direction,
)
from reflectors.atmosphere_constants import (
    EXOSPHERIC_SCALE_HEIGHT_KM,
    G_PER_KM3_TO_KG_PER_M3,
)


def test_units_conversion_g_per_km3_to_kg_per_m3():
    # 1 g/km^3 = 1e-3 kg / 1e9 m^3 = 1e-12 kg/m^3.
    assert G_PER_KM3_TO_KG_PER_M3 == 1.0e-12


def test_harris_priester_table_value_at_400km():
    # M&G Table 3.8: rho_min(400 km) = 2.249 g/km^3 = 2.249e-12 kg/m^3. Without a
    # Sun direction the model returns the antapex (minimum) density.
    hp = HarrisPriester()
    assert hp.density_kg_m3(400.0) == pytest.approx(2.249e-12, rel=1e-12)


def test_density_monotonic_decreasing_across_table():
    hp = HarrisPriester()
    alts = np.arange(150.0, 1000.0, 25.0)
    rho_min = [hp._interp_segment(a, hp._rho_min) for a in alts]
    rho_max = [hp._interp_segment(a, hp._rho_max) for a in alts]
    assert all(x > y for x, y in zip(rho_min, rho_min[1:]))
    assert all(x > y for x, y in zip(rho_max, rho_max[1:]))


def test_diurnal_bulge_bounds_and_extremes():
    hp = HarrisPriester()
    sun_hat = np.array([1.0, 0.0, 0.0])
    e_b = _bulge_apex_direction(sun_hat, hp.bulge_lag_deg)
    rho_min = hp._interp_segment(400.0, hp._rho_min)
    rho_max = hp._interp_segment(400.0, hp._rho_max)
    # Apex (r_hat == e_b) -> maximum; antapex (r_hat == -e_b) -> minimum.
    assert hp.density_kg_m3(400.0, e_b, sun_hat) == pytest.approx(rho_max, rel=1e-9)
    assert hp.density_kg_m3(400.0, -e_b, sun_hat) == pytest.approx(rho_min, rel=1e-9)
    # Any orientation stays within [min, max].
    for ang in np.linspace(0, math.pi, 12):
        r_hat = math.cos(ang) * e_b + math.sin(ang) * np.array([0, 0, 1.0])
        r_hat = r_hat / np.linalg.norm(r_hat)
        rho = hp.density_kg_m3(400.0, r_hat, sun_hat)
        assert rho_min - 1e-30 <= rho <= rho_max + 1e-30


def test_exospheric_tail_uses_calibrated_scale_height():
    # Above 1000 km the model continues from the 1000 km table value with
    # EXOSPHERIC_SCALE_HEIGHT_KM (NOT the steep heavy-species top-interval H).
    hp = HarrisPriester()
    rho_1000_min = hp._interp_segment(1000.0, hp._rho_min)
    expected_1500 = rho_1000_min * math.exp(-500.0 / EXOSPHERIC_SCALE_HEIGHT_KM)
    assert hp._interp_segment(1500.0, hp._rho_min) == pytest.approx(expected_1500, rel=1e-9)


def test_exponential_atmosphere_basic():
    atm = ExponentialAtmosphere(rho0_kg_m3=1e-12, scale_height_km=50.0, h0_ref_km=500.0)
    assert atm.density_kg_m3(500.0) == pytest.approx(1e-12)
    assert atm.density_kg_m3(550.0) == pytest.approx(1e-12 * math.exp(-1.0))


def test_tail_calibrated_against_nrlmsis():
    """The >1000 km tail tracks NRLMSIS 2.1 within a bounded factor.

    A continuation of the heavy-species scale height falls approximately 100x
    low by 2000 km. pymsis is the independent oracle; skip if unavailable.
    """
    pymsis = pytest.importorskip("pymsis")
    hp = HarrisPriester()
    alts = np.array([1000.0, 1200.0, 1500.0, 2000.0])
    out = pymsis.calculate(
        np.array(["2028-01-01T12:00"], dtype="datetime64[s]"),
        np.array([0.0]), np.array([0.0]), alts,
        np.array([150.0]), np.array([150.0]), np.array([[4.0] * 7]),
        version=2.1,
    )
    msis_noon = np.atleast_1d(np.squeeze(out)[..., 0])
    for k, alt in enumerate(alts):
        hp_max = hp._interp_segment(float(alt), hp._rho_max)
        ratio = hp_max / float(msis_noon[k])
        # The calibrated ratio is approximately 4-7. Heavy-species
        # extrapolation would fall below the asserted range at 2000 km.
        assert 0.2 < ratio < 12.0, f"alt {alt}: HP/NRLMSIS = {ratio:.2f}"
