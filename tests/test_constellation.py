"""Verification of the v1 Walker-style constellation generator."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from reflectors.constellation import (
    DEFAULT_FAMILY_SPECS,
    ExplicitShellSpec,
    MANUSCRIPT_V1_ORBIT_MODEL,
    MANUSCRIPT_V1_PROFILE_NAME,
    MRO120F_PROFILE_NAME,
    WalkerSizingConstraints,
    candidate_closest_approach,
    equal_altitude_minimum_distance_km,
    family_stack_altitudes_km,
    generate_walker_constellation,
    generate_walker_constellation_from_shell_specs,
    iter_ring_records,
    iter_satellite_records,
    maximally_fill_altitude_gaps_km,
    maximum_satellites_per_ring,
    mro120f_orbit_model,
    same_ring_chord_spacing_km,
    write_constellation_products,
)


EXPECTED_V1_CONFIGURATIONS = {
    12: (
        [(3, 81, 1)] * 3
        + [(3, 82, 1)] * 9
        + [(3, 83, 1)] * 9
    ),
    11: (
        [(6, 72, 4)] * 7
        + [(6, 71, 4)] * 7
        + [(6, 70, 4)] * 7
    ),
    10: [
        (7, 76, 2),
        (7, 76, 2),
        (7, 76, 2),
        (7, 76, 2),
        (7, 76, 2),
        (7, 76, 2),
        (8, 72, 2),
        (7, 76, 2),
        (7, 75, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 71, 2),
        (8, 70, 2),
        (8, 70, 2),
        (8, 70, 2),
    ],
    9: [
        (7, 93, 6),
        (7, 91, 3),
        (7, 92, 6),
        (7, 92, 6),
        (7, 92, 6),
        (7, 91, 6),
        (7, 91, 6),
        (7, 91, 6),
        (7, 91, 6),
        (7, 90, 6),
        (7, 90, 6),
        (7, 90, 6),
        (7, 90, 6),
        (7, 89, 6),
        (7, 89, 6),
        (7, 89, 6),
        (7, 89, 6),
        (7, 89, 3),
        (10, 62, 5),
        (7, 88, 6),
        (7, 88, 3),
    ],
}


@pytest.fixture(scope="module")
def manuscript_constellation():
    return generate_walker_constellation()


def _radius_unit_vector(
    argument_of_latitude_rad: float,
    inclination_rad: float,
    raan_rad: float,
) -> np.ndarray:
    """Direct circular position, independent of the implementation."""
    cos_u = math.cos(argument_of_latitude_rad)
    sin_u = math.sin(argument_of_latitude_rad)
    cos_i = math.cos(inclination_rad)
    sin_i = math.sin(inclination_rad)
    cos_raan = math.cos(raan_rad)
    sin_raan = math.sin(raan_rad)
    return np.array(
        [
            cos_raan * cos_u - sin_raan * cos_i * sin_u,
            sin_raan * cos_u + cos_raan * cos_i * sin_u,
            sin_i * sin_u,
        ]
    )


def _direct_cartesian_one_orbit_minimum_km(
    semimajor_axis_km: float,
    inclination_rad: float,
    raan_separation_rad: float,
    phase_difference_rad: float,
) -> float:
    """Numerically minimize direct Cartesian distance over common phase."""

    def squared_distance(common_phase_rad: float) -> float:
        first = _radius_unit_vector(common_phase_rad, inclination_rad, 0.0)
        second = _radius_unit_vector(
            common_phase_rad + phase_difference_rad,
            inclination_rad,
            raan_separation_rad,
        )
        displacement = semimajor_axis_km * (second - first)
        return float(np.dot(displacement, displacement))

    phase_grid = np.linspace(0.0, 2.0 * math.pi, 2048, endpoint=False)
    values = np.array([squared_distance(phase) for phase in phase_grid])
    minimum_index = int(np.argmin(values))
    grid_step = 2.0 * math.pi / len(phase_grid)
    center = phase_grid[minimum_index]
    result = minimize_scalar(
        squared_distance,
        bounds=(center - grid_step, center + grid_step),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert result.success
    return math.sqrt(max(0.0, float(result.fun)))


@pytest.mark.parametrize(
    ("semimajor_axis_km", "inclination_deg", "raan_separation_deg", "phase_deg"),
    [
        (3904.0, 93.4, 5.64, 1.0),
        (4408.0, 95.3, 15.97, 127.0),
        (4728.0, 96.8, 38.82, 313.0),
        (4500.0, 90.0, 25.0, 9.0),
    ],
)
def test_closed_form_distance_matches_direct_cartesian_one_orbit_minimization(
    semimajor_axis_km: float,
    inclination_deg: float,
    raan_separation_deg: float,
    phase_deg: float,
) -> None:
    inclination_rad = math.radians(inclination_deg)
    raan_separation_rad = math.radians(raan_separation_deg)
    phase_rad = math.radians(phase_deg)
    analytical = equal_altitude_minimum_distance_km(
        semimajor_axis_km,
        inclination_rad,
        raan_separation_rad,
        phase_rad,
    )
    numerical = _direct_cartesian_one_orbit_minimum_km(
        semimajor_axis_km,
        inclination_rad,
        raan_separation_rad,
        phase_rad,
    )
    assert analytical == pytest.approx(numerical, abs=2e-8)


def test_candidate_reduction_matches_all_explicit_ring_and_satellite_pairs() -> None:
    semimajor_axis_km = 4100.0
    inclination_rad = math.radians(94.0)
    raan_span_rad = math.radians(11.28)
    rings = 3
    satellites_per_ring = 7
    phasing = 1
    analytical = candidate_closest_approach(
        semimajor_axis_km,
        inclination_rad,
        raan_span_rad,
        rings,
        satellites_per_ring,
        phasing,
    )
    assert analytical is not None

    direct_minimum = math.inf
    for first_ring in range(rings):
        first_raan = first_ring * raan_span_rad / (rings - 1)
        for second_ring in range(first_ring + 1, rings):
            second_raan = second_ring * raan_span_rad / (rings - 1)
            raan_difference = second_raan - first_raan
            for first_slot in range(satellites_per_ring):
                first_u = (
                    2.0 * math.pi * first_slot / satellites_per_ring
                    + 2.0
                    * math.pi
                    * phasing
                    * first_ring
                    / (rings * satellites_per_ring)
                )
                for second_slot in range(satellites_per_ring):
                    second_u = (
                        2.0 * math.pi * second_slot / satellites_per_ring
                        + 2.0
                        * math.pi
                        * phasing
                        * second_ring
                        / (rings * satellites_per_ring)
                    )
                    direct_minimum = min(
                        direct_minimum,
                        _direct_cartesian_one_orbit_minimum_km(
                            semimajor_axis_km,
                            inclination_rad,
                            raan_difference,
                            second_u - first_u,
                        ),
                    )
    assert analytical.distance_km == pytest.approx(direct_minimum, abs=2e-8)


def test_same_ring_integer_bound_is_exact_against_neighboring_counts() -> None:
    semimajor_axis_km = 4408.0
    maximum = maximum_satellites_per_ring(semimajor_axis_km, 300.0)
    assert same_ring_chord_spacing_km(semimajor_axis_km, maximum) >= 300.0
    assert same_ring_chord_spacing_km(semimajor_axis_km, maximum + 1) < 300.0


def test_default_bands_center_to_the_v1_intervals() -> None:
    obtained = {
        spec.k_orbits_per_sol: spec.centered_ltan_band_h()
        for spec in DEFAULT_FAMILY_SPECS
    }
    expected = {
        12: (17.624, 18.376),
        11: (16.854, 19.146),
        10: (16.403, 19.597),
        9: (16.059, 19.941),
    }
    for family_k, interval in expected.items():
        assert obtained[family_k] == pytest.approx(interval, abs=1e-12)


def test_continuous_grid_preserves_all_84_centers_and_maximally_fills_gaps() -> None:
    preserved = family_stack_altitudes_km()
    filled = maximally_fill_altitude_gaps_km(preserved, 5.0)
    assert len(preserved) == 84
    assert len(filled) == 185
    assert set(preserved) < set(filled)
    assert filled[0] == 508.0
    assert filled[-1] == 1432.0
    gaps = np.diff(filled)
    assert min(gaps) == pytest.approx(5.0)
    assert sorted(set(gaps)) == pytest.approx([5.0, 6.0, 8.0])
    assert tuple(value for value in filled if 608.0 < value < 741.0) == tuple(
        np.arange(613.0, 734.0, 5.0)
    )
    assert tuple(value for value in filled if 841.0 < value < 1012.0) == tuple(
        np.arange(846.0, 1007.0, 5.0)
    )
    assert tuple(value for value in filled if 1112.0 < value < 1332.0) == tuple(
        np.arange(1117.0, 1328.0, 5.0)
    )


def test_explicit_shell_grid_uses_each_shells_own_band() -> None:
    shell_specs = (
        ExplicitShellSpec(12, 0, 508.0, 17.381, 18.376, "test-band"),
        ExplicitShellSpec(11, 0, 741.0, 16.742, 19.146, "test-band"),
    )
    constraints = WalkerSizingConstraints(shell_count=2)
    constellation = generate_walker_constellation_from_shell_specs(
        shell_specs,
        constraints=constraints,
    )
    assert constellation.shell_layout_name == "explicit-shell-grid"
    assert [shell.altitude_km for shell in constellation.shells] == [508.0, 741.0]
    assert [shell.band_method for shell in constellation.shells] == [
        "test-band",
        "test-band",
    ]
    assert constellation.shells[0].raan_span_deg == pytest.approx(11.28)
    assert constellation.shells[1].raan_span_deg == pytest.approx(34.38)


def test_frozen_profile_reproduces_every_v1_shell_choice(manuscript_constellation) -> None:
    assert manuscript_constellation.orbit_model.name == MANUSCRIPT_V1_PROFILE_NAME
    for family_k, expected in EXPECTED_V1_CONFIGURATIONS.items():
        shells = manuscript_constellation.family_shells(family_k)
        obtained = [
            (shell.rings, shell.satellites_per_ring, shell.phasing)
            for shell in shells
        ]
        assert obtained == expected


def test_frozen_profile_reproduces_v1_counts_and_constraints(manuscript_constellation) -> None:
    family_counts = {
        family_k: sum(
            shell.satellite_count
            for shell in manuscript_constellation.family_shells(family_k)
        )
        for family_k in (9, 10, 11, 12)
    }
    assert family_counts == {9: 13255, 10: 11617, 11: 8946, 12: 5184}
    assert manuscript_constellation.satellite_count == 39002
    assert manuscript_constellation.ring_count == 499
    assert len(manuscript_constellation.shells) == 84
    assert all(
        shell.same_ring_spacing_km >= 300.0
        for shell in manuscript_constellation.shells
    )
    assert all(
        shell.minimum_inter_ring_spacing_km is not None
        and shell.minimum_inter_ring_spacing_km >= 50.0
        for shell in manuscript_constellation.shells
    )


def test_current_mro120f_profile_is_explicitly_not_the_frozen_v1_profile() -> None:
    orbit_model = mro120f_orbit_model()
    assert orbit_model.name == MRO120F_PROFILE_NAME
    assert orbit_model.j2 != MANUSCRIPT_V1_ORBIT_MODEL.j2
    constellation = generate_walker_constellation(orbit_model=orbit_model)
    family_counts = {
        family_k: sum(
            shell.satellite_count for shell in constellation.family_shells(family_k)
        )
        for family_k in (9, 10, 11, 12)
    }
    assert family_counts == {9: 13231, 10: 11696, 11: 8928, 12: 5184}
    assert constellation.satellite_count == 39039


def test_ring_and_satellite_records_obey_the_walker_phase_law(
    manuscript_constellation,
) -> None:
    one_shell = generate_walker_constellation(
        family_specs=(DEFAULT_FAMILY_SPECS[0],),
        constraints=WalkerSizingConstraints(shell_count=1),
    )
    shell = one_shell.shells[0]
    ring_records = list(iter_ring_records(one_shell))
    satellite_records = list(
        iter_satellite_records(
            one_shell,
            initial_argument_of_latitude_offset_deg=7.5,
        )
    )
    assert len(ring_records) == shell.rings
    assert len(satellite_records) == shell.satellite_count
    assert [record["ltan_h"] for record in ring_records] == pytest.approx(
        [17.624, 18.0, 18.376]
    )
    record = next(
        item
        for item in satellite_records
        if item["ring_index"] == 2 and item["slot_index"] == 5
    )
    expected_u_deg = (
        7.5
        + 360.0 * 5 / shell.satellites_per_ring
        + 360.0 * shell.phasing * 2 / (shell.rings * shell.satellites_per_ring)
    ) % 360.0
    assert record["initial_argument_of_latitude_deg"] == pytest.approx(expected_u_deg)
    assert record["initial_mean_anomaly_deg"] == pytest.approx(expected_u_deg)


def test_product_writer_emits_self_describing_complete_inventory(
    tmp_path: Path,
    manuscript_constellation,
) -> None:
    products = write_constellation_products(manuscript_constellation, tmp_path)
    assert set(products) == {"manifest", "shells", "rings", "satellites"}
    manifest = json.loads(products["manifest"].read_text(encoding="utf-8"))
    assert manifest["orbit_model"]["name"] == MANUSCRIPT_V1_PROFILE_NAME
    assert manifest["summary"]["satellite_count"] == 39002
    assert manifest["summary"]["ring_count"] == 499
    assert len(manifest["shells"]) == 84

    with products["shells"].open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 84
    with products["rings"].open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 499
    with products["satellites"].open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 39002

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_constellation_products(manuscript_constellation, tmp_path)
    maximally_fill_altitude_gaps_km,
