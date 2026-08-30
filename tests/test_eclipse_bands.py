"""Regression tests for the whole-year eclipse-band calculation."""

from __future__ import annotations

import math

import pytest

from reflectors.eclipse_bands import (
    REFERENCE_EPOCH_UTC,
    EclipseBandSearchConfig,
    SeasonalEclipseBand,
    intersect_seasonal_eclipse_bands,
    keplerian_period_s,
    whole_year_eclipse_free_ltan_band,
)
from reflectors.ephemeris import utc_to_et
from reflectors.gravity import mars_gravity_model
from reflectors.kernels import load_kernels
from reflectors.mars_constants import SECONDS_PER_SOLAR_SOL_S


def _season(
    index: int,
    lower_h: float,
    upper_h: float,
) -> SeasonalEclipseBand:
    return SeasonalEclipseBand(
        season_index=index,
        epoch_et=float(index),
        true_anomaly_deg=10.0 * index,
        equation_of_center_offset_h=0.0,
        effective_ltan_start_h=lower_h,
        effective_ltan_end_h=upper_h,
        reference_ltan_start_h=lower_h,
        reference_ltan_end_h=upper_h,
    )


def test_seasonal_intersection_selects_independent_binding_rows() -> None:
    lower, upper = intersect_seasonal_eclipse_bands(
        (_season(0, 16.0, 20.0), _season(1, 17.0, 21.0), _season(2, 15.0, 19.0))
    )
    assert lower.season_index == 1
    assert upper.season_index == 2
    assert lower.reference_ltan_start_h == 17.0
    assert upper.reference_ltan_end_h == 19.0


def test_empty_whole_year_intersection_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="empty whole-year intersection"):
        intersect_seasonal_eclipse_bands(
            (_season(0, 17.0, 18.0), _season(1, 18.0, 19.0))
        )


def test_keplerian_period_obeys_circular_two_body_identity() -> None:
    semimajor_axis_km = 4408.0
    mu_km3_s2 = 42828.0
    period_s = keplerian_period_s(semimajor_axis_km, mu_km3_s2)
    assert mu_km3_s2 * period_s**2 == pytest.approx(
        4.0 * math.pi**2 * semimajor_axis_km**3,
        rel=2.0e-15,
    )


@pytest.mark.slow
def test_whole_year_method_reproduces_reference_anchor_bands() -> None:
    anchors = (
        (12, 3903.924477, 93.420985, (17.381, 18.376)),
        (11, 4136.808474, 94.204104, (16.742, 19.146)),
        (10, 4407.918053, 95.277366, (16.359, 19.597)),
        (9, 4728.393153, 96.799160, (16.014, 19.941)),
    )
    load_kernels()
    model = mars_gravity_model(max_degree=2)
    epoch_et = utc_to_et(REFERENCE_EPOCH_UTC)
    for family_k, semimajor_axis_km, inclination_deg, expected_band_h in anchors:
        result = whole_year_eclipse_free_ltan_band(
            semimajor_axis_km,
            inclination_deg,
            epoch_et,
            orbital_period_s=SECONDS_PER_SOLAR_SOL_S / family_k,
            config=EclipseBandSearchConfig(),
            mu_km3_s2=float(model.mu_km3_s2),
        )
        assert result.ltan_start_h == pytest.approx(expected_band_h[0], abs=5.1e-4)
        assert result.ltan_end_h == pytest.approx(expected_band_h[1], abs=5.1e-4)
