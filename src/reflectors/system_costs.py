"""Scenario cost assumptions for surface PV, batteries, and reflectors.

The PV and reflector inputs define configurable scenarios. The battery defaults
are literature-anchored benchmark assumptions, not a detailed Mars power-system
design. Keeping the primitive assumptions here prevents separate analyses from
drifting apart when a mass or transport assumption changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfacePVSystemCostAssumptions:
    """Procurement and delivery assumptions per square metre of PV."""

    pv_procurement_USD_per_m2: float
    pv_areal_density_kg_m2: float
    balance_of_plant_to_pv_mass_ratio: float
    balance_of_plant_procurement_USD_per_kg: float
    mars_surface_transport_USD_per_kg: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def balance_of_plant_areal_density_kg_m2(self) -> float:
        return (
            self.balance_of_plant_to_pv_mass_ratio
            * self.pv_areal_density_kg_m2
        )

    @property
    def balance_of_plant_procurement_USD_per_m2(self) -> float:
        return (
            self.balance_of_plant_areal_density_kg_m2
            * self.balance_of_plant_procurement_USD_per_kg
        )

    @property
    def transport_USD_per_m2(self) -> float:
        # Multiply the transport rate by the panel density before applying the
        # total mass ratio.  Besides being algebraically simpler, this avoids
        # the binary64 roundoff in 0.8 + 1.6 before the multiplication.
        return (
            self.mars_surface_transport_USD_per_kg
            * self.pv_areal_density_kg_m2
            * (1.0 + self.balance_of_plant_to_pv_mass_ratio)
        )

    @property
    def total_USD_per_m2(self) -> float:
        return (
            self.pv_procurement_USD_per_m2
            + self.balance_of_plant_procurement_USD_per_m2
            + self.transport_USD_per_m2
        )


@dataclass(frozen=True)
class BatteryStorageCostAssumptions:
    """Pack mass and cost derived from a supplied usable energy capacity."""

    usable_capacity_fraction: float
    specific_energy_Wh_per_kg: float
    procurement_USD_per_kWh: float
    mars_surface_transport_USD_per_kg: float

    def __post_init__(self) -> None:
        if not 0.0 < self.usable_capacity_fraction <= 1.0:
            raise ValueError("usable_capacity_fraction must lie in (0, 1]")
        if self.specific_energy_Wh_per_kg <= 0.0:
            raise ValueError("specific_energy_Wh_per_kg must be positive")
        if self.procurement_USD_per_kWh < 0.0:
            raise ValueError("procurement_USD_per_kWh must be nonnegative")
        if self.mars_surface_transport_USD_per_kg < 0.0:
            raise ValueError(
                "mars_surface_transport_USD_per_kg must be nonnegative"
            )

    def nameplate_capacity_MWh(self, usable_capacity_MWh: float) -> float:
        if usable_capacity_MWh < 0.0:
            raise ValueError("usable_capacity_MWh must be nonnegative")
        return usable_capacity_MWh / self.usable_capacity_fraction

    def mass_kg(self, usable_capacity_MWh: float) -> float:
        return (
            self.nameplate_capacity_MWh(usable_capacity_MWh)
            * 1.0e6
            / self.specific_energy_Wh_per_kg
        )

    def procurement_USD(self, usable_capacity_MWh: float) -> float:
        return (
            self.nameplate_capacity_MWh(usable_capacity_MWh)
            * 1.0e3
            * self.procurement_USD_per_kWh
        )

    def transport_USD(self, usable_capacity_MWh: float) -> float:
        return (
            self.mass_kg(usable_capacity_MWh)
            * self.mars_surface_transport_USD_per_kg
        )

    def total_USD(self, usable_capacity_MWh: float) -> float:
        return self.procurement_USD(usable_capacity_MWh) + self.transport_USD(
            usable_capacity_MWh
        )


@dataclass(frozen=True)
class OrbitingReflectorCostAssumptions:
    """Procurement and LEO-delivery assumptions per reflector area."""

    areal_density_kg_m2: float
    procurement_USD_per_kg: float
    leo_transport_USD_per_kg: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def procurement_USD_per_m2(self) -> float:
        return self.areal_density_kg_m2 * self.procurement_USD_per_kg

    @property
    def leo_transport_USD_per_m2(self) -> float:
        return self.areal_density_kg_m2 * self.leo_transport_USD_per_kg

    @property
    def total_USD_per_m2(self) -> float:
        return self.procurement_USD_per_m2 + self.leo_transport_USD_per_m2


SURFACE_PV_SYSTEM_COSTS = SurfacePVSystemCostAssumptions(
    pv_procurement_USD_per_m2=3250.0,
    pv_areal_density_kg_m2=0.800,
    balance_of_plant_to_pv_mass_ratio=2.0,
    balance_of_plant_procurement_USD_per_kg=50.0,
    mars_surface_transport_USD_per_kg=2500.0,
)

# The 80% usable fraction makes the installed/nameplate capacity 1.25 times
# the supplied usable capacity, leaving 20% of nameplate energy in reserve.
# NASA (2021), Small Spacecraft Technology State of the Art, Table 3-3 and
# Sec. 3.4, reports 149.2--153.5 Wh/kg for representative Li-ion battery packs
# and gives the pack-level values below the cell-level specific energy:
# https://www.nasa.gov/wp-content/uploads/2021/10/3.soa_power_2021.pdf
# NREL's 2024 ATB gives 164 USD/kWh (2022 USD) as its ex-factory 8-hour LIB
# pack input, the nearest tabulated long-duration case to the supplied needs:
# https://atb.nrel.gov/electricity/2024/commercial_battery_storage
MARS_SURFACE_BATTERY_COSTS = BatteryStorageCostAssumptions(
    usable_capacity_fraction=0.80,
    specific_energy_Wh_per_kg=150.0,
    procurement_USD_per_kWh=164.0,
    mars_surface_transport_USD_per_kg=(
        SURFACE_PV_SYSTEM_COSTS.mars_surface_transport_USD_per_kg
    ),
)

# Reflector scenario with an areal density of 50 g/m2.
STANDALONE_REFLECTOR_COSTS = OrbitingReflectorCostAssumptions(
    areal_density_kg_m2=0.050,
    procurement_USD_per_kg=500.0,
    leo_transport_USD_per_kg=250.0,
)

# Reflector scenario matching the canonical 18 g/m2 sail.
ANNUAL_REFLECTOR_COSTS = OrbitingReflectorCostAssumptions(
    areal_density_kg_m2=0.018,
    procurement_USD_per_kg=350.0,
    leo_transport_USD_per_kg=250.0,
)
