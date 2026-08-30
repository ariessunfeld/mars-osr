"""Fast arithmetic checks for the single-sourced system-cost assumptions."""

from __future__ import annotations

import pytest

from reflectors.system_costs import (
    ANNUAL_REFLECTOR_COSTS,
    MARS_SURFACE_BATTERY_COSTS,
    STANDALONE_REFLECTOR_COSTS,
    SURFACE_PV_SYSTEM_COSTS,
    BatteryStorageCostAssumptions,
    OrbitingReflectorCostAssumptions,
    SurfacePVSystemCostAssumptions,
)


def test_surface_pv_two_x_balance_of_plant_breakdown() -> None:
    costs = SURFACE_PV_SYSTEM_COSTS
    assert costs.balance_of_plant_to_pv_mass_ratio == 2.0
    assert costs.balance_of_plant_areal_density_kg_m2 == pytest.approx(1.6)
    assert costs.balance_of_plant_procurement_USD_per_m2 == pytest.approx(80.0)
    assert costs.transport_USD_per_m2 == pytest.approx(6000.0)
    assert costs.total_USD_per_m2 == pytest.approx(9330.0)


def test_reflector_scenario_breakdowns() -> None:
    assert STANDALONE_REFLECTOR_COSTS.procurement_USD_per_m2 == pytest.approx(25.0)
    assert STANDALONE_REFLECTOR_COSTS.leo_transport_USD_per_m2 == pytest.approx(12.5)
    assert STANDALONE_REFLECTOR_COSTS.total_USD_per_m2 == pytest.approx(37.5)
    assert ANNUAL_REFLECTOR_COSTS.procurement_USD_per_m2 == pytest.approx(6.3)
    assert ANNUAL_REFLECTOR_COSTS.leo_transport_USD_per_m2 == pytest.approx(4.5)
    assert ANNUAL_REFLECTOR_COSTS.total_USD_per_m2 == pytest.approx(10.8)


def test_battery_cost_breakdown_from_usable_capacity() -> None:
    costs = MARS_SURFACE_BATTERY_COSTS
    assert costs.usable_capacity_fraction == 0.8
    assert costs.nameplate_capacity_MWh(1.0) == pytest.approx(1.25)
    assert costs.mass_kg(1.0) == pytest.approx(1.25e6 / 150.0)
    assert costs.procurement_USD(1.0) == pytest.approx(205_000.0)
    assert costs.transport_USD(1.0) == pytest.approx(20_833_333.333333332)
    assert costs.total_USD(1.0) == pytest.approx(21_038_333.333333332)
    assert (
        costs.mars_surface_transport_USD_per_kg
        == SURFACE_PV_SYSTEM_COSTS.mars_surface_transport_USD_per_kg
    )


@pytest.mark.parametrize(
    "constructor, kwargs",
    [
        (
            SurfacePVSystemCostAssumptions,
            {
                "pv_procurement_USD_per_m2": -1.0,
                "pv_areal_density_kg_m2": 0.8,
                "balance_of_plant_to_pv_mass_ratio": 2.0,
                "balance_of_plant_procurement_USD_per_kg": 50.0,
                "mars_surface_transport_USD_per_kg": 2500.0,
            },
        ),
        (
            OrbitingReflectorCostAssumptions,
            {
                "areal_density_kg_m2": 0.018,
                "procurement_USD_per_kg": -1.0,
                "leo_transport_USD_per_kg": 250.0,
            },
        ),
        (
            BatteryStorageCostAssumptions,
            {
                "usable_capacity_fraction": 0.8,
                "specific_energy_Wh_per_kg": 150.0,
                "procurement_USD_per_kWh": -1.0,
                "mars_surface_transport_USD_per_kg": 2500.0,
            },
        ),
    ],
)
def test_negative_cost_assumptions_are_rejected(constructor, kwargs) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        constructor(**kwargs)


@pytest.mark.parametrize("usable_fraction", [0.0, 1.01])
def test_invalid_battery_usable_fraction_is_rejected(
    usable_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="must lie in"):
        BatteryStorageCostAssumptions(
            usable_capacity_fraction=usable_fraction,
            specific_energy_Wh_per_kg=150.0,
            procurement_USD_per_kWh=164.0,
            mars_surface_transport_USD_per_kg=2500.0,
        )


def test_negative_usable_battery_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="usable_capacity_MWh"):
        MARS_SURFACE_BATTERY_COSTS.total_USD(-1.0)
