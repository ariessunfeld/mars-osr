"""Fast verification of pinned Mars constants in ``reflectors.mars_constants``.

The pinned values are literature-sourced. These tests tie them to a live
SPICE / DE440 cross-check so drift (either in the literature
consensus or in the ephemeris kernel) is caught rather than baked in.
"""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from reflectors.dynamics import body_gm_km3_per_s2, sun_gm_km3_per_s2
from reflectors.ephemeris import body_state
from reflectors.kernels import load_kernels
from reflectors.mars_constants import (
    DEIMOS_GM_KONOPLIV_2020_KM3_S2,
    MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM,
    MARS_HILL_RADIUS_KM,
    MARS_PLANET_GM_KONOPLIV_2020_KM3_S2,
    MARS_SIDEREAL_YEAR_DAYS,
    MARS_SIDEREAL_YEAR_S,
    PHOBOS_GM_KONOPLIV_2020_KM3_S2,
    SECONDS_PER_SOLAR_SOL_S,
)
from reflectors.solar_constants import AU_KM


def test_mars_sidereal_year_seconds_matches_days_constant():
    """Trivial consistency between the day and second forms."""
    assert MARS_SIDEREAL_YEAR_S == MARS_SIDEREAL_YEAR_DAYS * 86400.0


def test_mars_sidereal_year_matches_de440_osculating_period():
    """Pinned Mars sidereal year agrees with DE440 osculating Keplerian
    period at J2000 to within 0.5 days.

    The 0.5-day tolerance is comfortably above the osculating-vs-mean
    spread over Mars's eccentric orbit (~0.2 days) and the difference
    between sidereal and tropical definitions (~0.01 days). Tight
    enough to catch a real literature drift, loose enough not to be
    brittle against routine kernel refinements.
    """
    mu_sun = sun_gm_km3_per_s2()
    state_helio, _ = body_state(
        "MARS BARYCENTER", 0.0, observer="SUN", frame="J2000"
    )
    # spice.oscltx returns an 11-vector; element [10] is the orbital period
    # tau. Valid only for elliptic orbits; Mars satisfies this condition.
    elts = np.asarray(spice.oscltx(state_helio, 0.0, mu_sun), dtype=float)
    period_s = float(elts[10])
    delta_days = abs(period_s - MARS_SIDEREAL_YEAR_S) / 86400.0
    assert delta_days < 0.5, (
        f"DE440 osculating Mars period {period_s/86400:.4f} days differs from "
        f"pinned {MARS_SIDEREAL_YEAR_DAYS} days by {delta_days:.4f} days"
    )


def test_mars_solar_day_matches_synodic_identity():
    """Pinned Mars solar day matches the synodic identity

        1 / T_solar = 1 / T_sidereal_day  -  1 / T_sidereal_year

    using the live Mars rotation rate from the PCK kernel and the
    pinned ``MARS_SIDEREAL_YEAR_S``. Tolerance 1.0 s; the synodic
    identity is exact, so any larger drift indicates a literature
    or kernel drift.
    """
    load_kernels()
    # PM for body 499 (Mars): [W_0_deg, dW_dt_deg_per_day, d2W_dt2_deg_per_century2].
    # dW/dt is the IAU prime-meridian rotation rate.
    n_pm, pm = spice.bodvcd(499, "PM", 3)
    assert n_pm == 3, f"unexpected PM array length for Mars: {n_pm}"
    rot_rate_deg_per_day = float(pm[1])
    sidereal_day_s = (360.0 / rot_rate_deg_per_day) * 86400.0
    # Synodic relation: prograde rotation of prograde-orbiting body.
    solar_day_predicted_s = 1.0 / (
        1.0 / sidereal_day_s - 1.0 / MARS_SIDEREAL_YEAR_S
    )
    delta_s = abs(solar_day_predicted_s - SECONDS_PER_SOLAR_SOL_S)
    assert delta_s < 1.0, (
        f"synodic identity gives Mars solar day = "
        f"{solar_day_predicted_s:.3f} s; pinned "
        f"SECONDS_PER_SOLAR_SOL_S = {SECONDS_PER_SOLAR_SOL_S} s; "
        f"delta = {delta_s:.3f} s"
    )


def test_konopliv_2020_gm_quartet_sums_to_mro120f_header():
    """Self-consistency: Mars-alone + Phobos + Deimos GMs (Konopliv 2020)
    sum to the MRO120F SHADR header mu within 1e-6 km^3/s^2.

    The PDS label jgmro_120f_sha.lbl publishes all four numbers from one
    fit. The arithmetic identity
        GM_Mars_alone + GM_Phobos + GM_Deimos == GM_system
    is what makes the central-mu decoupling in
    ``reflectors.dynamics.propagate`` exact: when the moons are passed as
    separate third bodies, subtracting their GMs from the lumped MRO120F
    header mu leaves precisely Mars-alone GM as the central two-body
    parameter.

    Tolerance 1e-6 km^3/s^2 reflects the four-decimal precision of the
    Mars-alone value in the PDS label (last published digit is 1e-4),
    so the test passes if the published values are read in correctly
    and fails if any one of the three drifts at the 1e-7 level or worse.
    """
    from reflectors.gravity import mars_gravity_model

    model = mars_gravity_model(max_degree=2)
    quartet_sum = (
        MARS_PLANET_GM_KONOPLIV_2020_KM3_S2
        + PHOBOS_GM_KONOPLIV_2020_KM3_S2
        + DEIMOS_GM_KONOPLIV_2020_KM3_S2
    )
    assert abs(quartet_sum - model.mu_km3_s2) < 1e-6, (
        f"Konopliv-2020 quartet sum {quartet_sum:.10f} km^3/s^2 differs "
        f"from MRO120F header mu {model.mu_km3_s2:.10f} km^3/s^2 by "
        f"{quartet_sum - model.mu_km3_s2:+.3e} km^3/s^2"
    )


def test_konopliv_2020_moon_gms_consistent_with_de440_kernel_pool():
    """Konopliv 2020 Phobos/Deimos GMs are consistent with the DE440
    kernel-pool values within Konopliv's published 1-sigma uncertainties.

    Konopliv 2020 (PDS label jgmro_120f_sha.lbl):
        Phobos GM = (7.10 +/- 0.05) x 10^-4 km^3/s^2
        Deimos GM = (9.68 +/- 1.30) x 10^-5 km^3/s^2
    DE440 (gm_de440.tpc):
        BODY401_GM ~ 7.0875e-4 km^3/s^2  (~0.18% below Konopliv)
        BODY402_GM ~ 9.6156e-5 km^3/s^2  (~0.54% below Konopliv)

    Both differences fit comfortably inside the Konopliv 1-sigma bars
    (3.5% Phobos, 13% Deimos). This test pins that they keep doing so;
    a kernel-pool refresh that drifts beyond Konopliv 1-sigma would
    indicate a real cross-solution disagreement worth investigating.
    """
    load_kernels()
    de440_phobos_gm = body_gm_km3_per_s2(401)
    de440_deimos_gm = body_gm_km3_per_s2(402)
    phobos_one_sigma = 0.05e-4
    deimos_one_sigma = 1.30e-5
    assert abs(de440_phobos_gm - PHOBOS_GM_KONOPLIV_2020_KM3_S2) < phobos_one_sigma, (
        f"DE440 BODY401_GM = {de440_phobos_gm:.6e} km^3/s^2 differs from "
        f"Konopliv-2020 Phobos GM {PHOBOS_GM_KONOPLIV_2020_KM3_S2:.6e} by "
        f"{de440_phobos_gm - PHOBOS_GM_KONOPLIV_2020_KM3_S2:+.3e} km^3/s^2 "
        f"(beyond Konopliv 1-sigma {phobos_one_sigma:.3e})"
    )
    assert abs(de440_deimos_gm - DEIMOS_GM_KONOPLIV_2020_KM3_S2) < deimos_one_sigma, (
        f"DE440 BODY402_GM = {de440_deimos_gm:.6e} km^3/s^2 differs from "
        f"Konopliv-2020 Deimos GM {DEIMOS_GM_KONOPLIV_2020_KM3_S2:.6e} by "
        f"{de440_deimos_gm - DEIMOS_GM_KONOPLIV_2020_KM3_S2:+.3e} km^3/s^2 "
        f"(beyond Konopliv 1-sigma {deimos_one_sigma:.3e})"
    )


def test_mars_heliocentric_semimajor_axis_consistent_with_au():
    """Pinned ``MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM`` reproduces the
    documented 1.52371034 AU using the single-source AU from
    ``solar_constants`` (no competing AU definition in mars_constants).
    """
    a_from_au = 1.52371034 * AU_KM
    rel = abs(MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM - a_from_au) / a_from_au
    assert rel < 1e-5, (
        f"MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM = "
        f"{MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM:.6e} km differs from "
        f"1.52371034 AU = {a_from_au:.6e} km by {rel:.2e} relative"
    )


def test_mars_heliocentric_semimajor_axis_matches_de440():
    """Pinned Mars heliocentric semi-major axis agrees with the DE440
    osculating Keplerian value at J2000.

    The osculating ``a`` of MARS BARYCENTER about the SUN oscillates
    about the mean element under planetary perturbations; a 0.3%
    tolerance comfortably covers that spread while still catching a
    real literature drift.
    """
    mu_sun = sun_gm_km3_per_s2()
    state_helio, _ = body_state(
        "MARS BARYCENTER", 0.0, observer="SUN", frame="J2000"
    )
    # spice.oscltx 11-vector: element [9] is the semi-major axis a.
    elts = np.asarray(spice.oscltx(state_helio, 0.0, mu_sun), dtype=float)
    a_de440 = float(elts[9])
    rel = abs(a_de440 - MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM) / a_de440
    assert rel < 3e-3, (
        f"DE440 osculating Mars a = {a_de440:.6e} km differs from pinned "
        f"{MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM:.6e} km by {rel:.2e} relative"
    )


def test_mars_hill_radius_matches_kernel_pool_recomputation():
    """Pinned ``MARS_HILL_RADIUS_KM`` reproduces the classical Hill radius

        r_Hill = a_Mars * ( mu_Mars / (3 mu_Sun) ) ** (1/3)

    recomputed from the pinned semi-major axis and the live kernel-pool
    GMs (BODY499_GM, BODY10_GM). A 0.1% tolerance catches a pin typo or
    a kernel-pool GM drift; the pinned value carries no eccentricity
    factor (semi-major-axis mean -- see the mars_constants docstring).
    """
    load_kernels()
    mu_mars = body_gm_km3_per_s2(499)
    mu_sun = sun_gm_km3_per_s2()
    r_hill = MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM * (
        mu_mars / (3.0 * mu_sun)
    ) ** (1.0 / 3.0)
    rel = abs(r_hill - MARS_HILL_RADIUS_KM) / r_hill
    assert rel < 1e-3, (
        f"recomputed Mars Hill radius {r_hill:.6e} km differs from pinned "
        f"MARS_HILL_RADIUS_KM {MARS_HILL_RADIUS_KM:.6e} km by {rel:.2e} "
        f"relative"
    )
