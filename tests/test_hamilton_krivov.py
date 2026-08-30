"""Independent checks for the Hamilton-Krivov orbit-averaged oracle.

Primary source: Hamilton & Krivov (1996), Icarus 123, pp. 506-517.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from reflectors.hamilton_krivov import (
    HK96_DEIMOS_C_TIMES_RADIUS_UM,
    HK96_PHOBOS_C_TIMES_RADIUS_UM,
    HK96_PHOBOS_W,
    radiative_parameter,
    radiation_pressure_only_max_eccentricity,
    rp_j2_eccentricity_vector_rhs,
    rp_j2_hamiltonian,
    rp_j2_hamiltonian_from_vector,
    rp_j2_max_eccentricity,
    rp_j2_separatrix_transition,
    rp_j2_stationary_point_bifurcation,
)


def test_deimos_radiation_pressure_curve_matches_equation_17() -> None:
    """Paper p. 509 states that Eq. (17) is the shape of Fig. 2a."""
    expected = {
        20.0: 0.669565654747,
        80.0: 0.190343962266,
        1000.0: 0.015367092668,
    }
    for radius_um, expected_e in expected.items():
        C = radiative_parameter(HK96_DEIMOS_C_TIMES_RADIUS_UM, radius_um)
        assert radiation_pressure_only_max_eccentricity(C) == pytest.approx(
            expected_e, rel=2.0e-9
        )


def test_phobos_separatrix_reproduces_published_jump() -> None:
    """Paper p. 516 reports C=0.01466, e4=0.25, and r_g~331.5 um."""
    transition = rp_j2_separatrix_transition(HK96_PHOBOS_W)
    assert transition.radiative_parameter == pytest.approx(0.01466, abs=5.0e-5)
    assert transition.eccentricity == pytest.approx(0.25, abs=5.0e-4)
    assert transition.grain_radius_um(
        HK96_PHOBOS_C_TIMES_RADIUS_UM
    ) == pytest.approx(331.5, abs=1.0)


def test_phobos_stationary_bifurcation_reproduces_second_landmark() -> None:
    """Paper pp. 516-517 reports C=0.0210, e2=0.180, r_g~232 um."""
    transition = rp_j2_stationary_point_bifurcation(HK96_PHOBOS_W)
    assert transition.radiative_parameter == pytest.approx(0.0210, abs=1.0e-4)
    assert transition.eccentricity == pytest.approx(0.180, abs=2.0e-3)
    assert transition.grain_radius_um(
        HK96_PHOBOS_C_TIMES_RADIUS_UM
    ) == pytest.approx(232.0, abs=2.0)


def test_vector_rhs_is_regular_at_initially_circular_state() -> None:
    C = radiative_parameter(HK96_PHOBOS_C_TIMES_RADIUS_UM, 300.0)
    rhs = rp_j2_eccentricity_vector_rhs(0.0, np.zeros(2), C, HK96_PHOBOS_W)
    np.testing.assert_array_equal(rhs, np.array([0.0, C]))


def test_vector_hamiltonian_matches_polar_form() -> None:
    C = radiative_parameter(HK96_PHOBOS_C_TIMES_RADIUS_UM, 400.0)
    e = 0.23
    phi = 1.7
    xy = e * np.array([math.cos(phi), math.sin(phi)])
    assert rp_j2_hamiltonian_from_vector(xy, C, HK96_PHOBOS_W) == pytest.approx(
        rp_j2_hamiltonian(e, phi, C, HK96_PHOBOS_W), rel=2.0e-15
    )


@pytest.mark.parametrize("radius_um", [100.0, 1000.0])
def test_equation_7_integration_reaches_level_curve_maximum(radius_um: float) -> None:
    """Direct Eq. (7) integration independently reaches the level-curve root."""
    C = radiative_parameter(HK96_PHOBOS_C_TIMES_RADIUS_UM, radius_um)
    lambda_end = 16.0 * math.pi  # eight Mars years in the paper's independent variable
    sample_lambda = np.linspace(0.0, lambda_end, 20_001)
    solution = solve_ivp(
        rp_j2_eccentricity_vector_rhs,
        (0.0, lambda_end),
        np.zeros(2),
        args=(C, HK96_PHOBOS_W),
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
        t_eval=sample_lambda,
    )
    assert solution.success
    eccentricity = np.linalg.norm(solution.y, axis=0)
    expected_max = rp_j2_max_eccentricity(C, HK96_PHOBOS_W)
    assert float(eccentricity.max()) == pytest.approx(expected_max, abs=2.0e-6)

    invariant = np.array(
        [rp_j2_hamiltonian_from_vector(xy, C, HK96_PHOBOS_W) for xy in solution.y.T]
    )
    assert float(np.ptp(invariant)) < 2.0e-10


def test_maximum_eccentricity_has_published_discontinuity() -> None:
    transition = rp_j2_separatrix_transition(HK96_PHOBOS_W)
    below_radius = transition.grain_radius_um(HK96_PHOBOS_C_TIMES_RADIUS_UM) - 1.0
    above_radius = transition.grain_radius_um(HK96_PHOBOS_C_TIMES_RADIUS_UM) + 1.0
    e_small_grain = rp_j2_max_eccentricity(
        radiative_parameter(HK96_PHOBOS_C_TIMES_RADIUS_UM, below_radius),
        HK96_PHOBOS_W,
    )
    e_large_grain = rp_j2_max_eccentricity(
        radiative_parameter(HK96_PHOBOS_C_TIMES_RADIUS_UM, above_radius),
        HK96_PHOBOS_W,
    )
    assert e_small_grain > 0.45
    assert e_large_grain < 0.25
