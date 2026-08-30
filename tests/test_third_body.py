"""Fast tests for the third-body gravitational perturbation module.

Organised in four groups:

  1. Identities and limits:
      - a_3b(r=0) = 0 at machine precision (direct and indirect terms
        cancel identically when the sail coincides with Mars centre).
      - Multi-body accelerations sum linearly: from_spice([Sun, Phobos,
        Deimos, Jupiter]) equals the sum of single-body calls to
        machine precision.

  2. Mars / Sun anchors:
      - Sun tidal acceleration at LMO matches the closed-form
        ``2 mu_Sun r_sat / |r_Sun|^3`` estimate (radial-alignment
        geometry) to a few percent.
      - Tidal lobe pattern: a_3b is OUTWARD on both sides of Mars
        along the Mars-Sun line.
      - Precision-headroom regression: at Mars/Sun the direct-form
        sum retains > 1e6 of headroom over the worst individual
        term, sufficient for the direct form (see the module docstring).

  3. Conservative-force property:
      - Central-difference of the perturbation potential
        ``third_body_perturbation_potential`` recovers the analytic
        acceleration ``-grad U_3b`` to within the truncation floor
        (~1e-6 relative). The potential implementation uses the
        analytically cancellation-free factoring described in the
        module docstring, so no high-precision arithmetic is needed.

  4. Propagator plumbing:
      - ``propagate(third_bodies=[sun_third_body()])`` differs from
        the corresponding pure-two-body trajectory by the order of
        magnitude predicted from |a_Sun| * T^2 / 2 over one LMO
        orbit (a few metres).
      - ``propagate(gravity_degree=4, third_bodies=[Sun])`` differs
        from the matching gravity_degree=4 control by a similar
        magnitude, confirming the third-body term composes
        additively with full-harmonic gravity.
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice

from reflectors.dynamics import (
    PropagationOptions,
    mars_gm_km3_per_s2,
    propagate,
)
from reflectors.third_body import (
    deimos_third_body,
    jupiter_third_body,
    phobos_third_body,
    sun_third_body,
    third_body_acceleration,
    third_body_acceleration_from_spice,
    third_body_perturbation_potential,
)


EPOCH_STR = "2026-06-01T00:00:00"
MARS_EQ_R_KM = 3396.19
LMO_ALT_KM = 400.0
R_SAT_KM = MARS_EQ_R_KM + LMO_ALT_KM


# ---------------------------------------------------------------------------
# Group 1 -- identities and limits
# ---------------------------------------------------------------------------


def test_third_body_acceleration_vanishes_at_sail_at_central_body_origin():
    """a_3b(r_sat = 0) = 0 by exact cancellation of direct and indirect terms.

    Eq. 3.37 evaluated at r=0:
      mu_3 * [ (r_3 - 0)/|r_3 - 0|^3 - r_3/|r_3|^3 ]
        = mu_3 * [ r_3/|r_3|^3 - r_3/|r_3|^3 ] = 0.

    Pinned at machine precision for every factory body at a fixed
    epoch. Catches sign or term-omission errors in the acceleration
    code.
    """
    et = spice.str2et(EPOCH_STR)
    for body in (
        sun_third_body(),
        phobos_third_body(),
        deimos_third_body(),
        jupiter_third_body(),
    ):
        state, _ = spice.spkezr(str(body.naif_id), et, "J2000", "NONE", "499")
        r3 = np.asarray(state[:3], dtype=float)
        a = third_body_acceleration(np.zeros(3), r3, body.mu_km3_s2)
        # Floor of cancellation precision: each individual term is
        # mu / |r_3|^2; subtracting them in double gives noise at
        # ~|term| * 1e-16. Pin against that, not against pure zero.
        floor = body.mu_km3_s2 / float(np.dot(r3, r3)) * 1e-15
        assert np.linalg.norm(a) <= floor, (
            f"{body.label}: |a(r=0)|={np.linalg.norm(a):.3e} "
            f"exceeded floor={floor:.3e}"
        )


def test_third_body_acceleration_from_spice_equals_sum_of_individual_calls():
    """Multi-body wrapper is exactly the sum of single-body contributions.

    Pure linearity check at the acceleration level; the propagator-
    level stacking tests below are the trajectory-level analogues.
    """
    et = spice.str2et(EPOCH_STR)
    bodies = [
        sun_third_body(),
        phobos_third_body(),
        deimos_third_body(),
        jupiter_third_body(),
    ]
    r_sat = np.array([R_SAT_KM, 0.0, 0.0])
    a_sum = np.zeros(3)
    for body in bodies:
        state, _ = spice.spkezr(str(body.naif_id), et, "J2000", "NONE", "499")
        a_sum += third_body_acceleration(
            r_sat, np.asarray(state[:3], dtype=float), body.mu_km3_s2
        )
    a_combined = third_body_acceleration_from_spice(r_sat, et, bodies)
    # Floor: same SPICE positions are fetched the same way, same mu's
    # used; should be exactly equal in finite-precision floating point.
    assert np.allclose(a_combined, a_sum, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Group 2 -- Mars / Sun anchors
# ---------------------------------------------------------------------------


def test_sun_tidal_magnitude_at_lmo_matches_closed_form():
    """At Mars LMO the Sun tidal acceleration matches the radial estimate.

    Closed form for a sail along the Mars-Sun line, oriented radially
    relative to Mars centre, in the limit r_sat << r_Sun:

        |a_3b| ~ 2 mu_Sun r_sat / |r_Sun|^3

    For an arbitrary geometry the exact numerical value is within
    O(1) of this estimate; pin a 25% bracket. The test sail is at
    (R+400 km) +x in J2000 at EPOCH_STR, which is not exactly aligned
    with the Sun direction at this epoch, so the exact magnitude
    falls about 4% below the radial closed form. Bracket absorbs
    that geometry without hiding a substantive magnitude error.
    """
    et = spice.str2et(EPOCH_STR)
    sun = sun_third_body()
    state, _ = spice.spkezr("10", et, "J2000", "NONE", "499")
    r_sun = np.asarray(state[:3], dtype=float)
    r_sun_mag = float(np.linalg.norm(r_sun))
    r_sat = np.array([R_SAT_KM, 0.0, 0.0])
    a = third_body_acceleration(r_sat, r_sun, sun.mu_km3_s2)
    a_mag = float(np.linalg.norm(a))
    a_radial_estimate = 2.0 * sun.mu_km3_s2 * R_SAT_KM / r_sun_mag ** 3
    rel = a_mag / a_radial_estimate
    assert 0.75 < rel < 1.25, (
        f"|a_Sun|={a_mag:.3e}, radial closed-form={a_radial_estimate:.3e}, "
        f"ratio={rel:.3f} outside [0.75, 1.25]"
    )
    # Order-of-magnitude pin against drift.
    assert 5e-11 < a_mag < 5e-10, f"|a_Sun|={a_mag:.3e} out of LMO band"


def test_sun_tidal_lobes_outward_on_both_sides_of_mars_sun_line():
    """Tidal pattern: along the Mars-Sun line, a_3b points outward.

    Classic stretch-along-source signature. On the sunward side the
    sail is closer to the Sun than Mars, so net force pulls it
    further sunward (= outward radially). On the anti-sunward side
    the sail is farther, Mars is pulled MORE toward the Sun than
    the sail, so the relative force on the sail is anti-sunward
    (= outward radially again). Both signs of radial-component test
    must come out positive.
    """
    et = spice.str2et(EPOCH_STR)
    sun = sun_third_body()
    state, _ = spice.spkezr("10", et, "J2000", "NONE", "499")
    r_sun = np.asarray(state[:3], dtype=float)
    r_sun_hat = r_sun / np.linalg.norm(r_sun)
    for sign in (+1.0, -1.0):
        r_sat = sign * R_SAT_KM * r_sun_hat
        a = third_body_acceleration(r_sat, r_sun, sun.mu_km3_s2)
        radial = float(np.dot(a, r_sat / np.linalg.norm(r_sat)))
        assert radial > 0.0, (
            f"sign={sign:+.0f} radial component {radial:.3e} expected positive "
            "(tidal lobes point outward on both sides)"
        )


def test_direct_form_precision_headroom_at_mars_sun_geometry():
    """Each individual term in Eq. 3.37 is much larger than the residual
    -- but double-precision evaluation still
    retain plenty of relative precision on the sum.

    Quantitatively for Mars/Sun at LMO:
        |individual term|  ~ mu_Sun / |r_Sun|^2 ~ 2.5e-6 km/s^2
        |residual|         ~ 2 mu_Sun r_sat / |r_Sun|^3 ~ 9e-11 km/s^2
        |residual| / |term| ~ 3.5e-5  (~4.5 digits cancelled)
        double noise / |residual| ~ 1e-16 / 3.5e-5 ~ 3e-12  (~11 digits left)

    Test pins:
        |residual| / max(|term_direct|, |term_indirect|) > 1e-6
    requiring at least six digits of relative precision for the direct form.
    Geometries below this threshold require the Battin/Encke stable form.
    """
    et = spice.str2et(EPOCH_STR)
    sun = sun_third_body()
    state, _ = spice.spkezr("10", et, "J2000", "NONE", "499")
    r_sun = np.asarray(state[:3], dtype=float)
    r_sun_mag = float(np.linalg.norm(r_sun))
    r_sat = np.array([R_SAT_KM, 0.0, 0.0])
    direct_term_mag = sun.mu_km3_s2 / np.linalg.norm(r_sun - r_sat) ** 2
    indirect_term_mag = sun.mu_km3_s2 / r_sun_mag ** 2
    a = third_body_acceleration(r_sat, r_sun, sun.mu_km3_s2)
    headroom = float(np.linalg.norm(a)) / max(direct_term_mag, indirect_term_mag)
    assert headroom > 1e-6, (
        f"precision headroom {headroom:.3e} < 1e-6: switch to Battin "
        "stable form"
    )


# ---------------------------------------------------------------------------
# Group 3 -- conservative-force property
# ---------------------------------------------------------------------------


def test_finite_difference_of_potential_matches_analytic_acceleration():
    """Central-difference of U_3b recovers ``-grad U_3b`` = a_3b.

    Pins the consistency between the analytic acceleration formula
    (Montenbruck Eq. 3.37) and the perturbation potential formula
    (derived in the module docstring as Eq. **). Catches any sign
    or normalization error in either expression.

    The ``third_body_perturbation_potential`` implementation uses
    the analytically cancellation-free form, so plain double-
    precision FD yields agreement well below the central-difference
    truncation floor of ~1e-6 (Richardson O(h^2)). Pin loosely at
    1e-5 to leave headroom for the few cases where the test sail
    geometry is close to a node of d^3U/dr^3.
    """
    et = spice.str2et(EPOCH_STR)
    sun = sun_third_body()
    state, _ = spice.spkezr("10", et, "J2000", "NONE", "499")
    r_sun = np.asarray(state[:3], dtype=float)
    # Off-axis, asymmetric test point: not aligned with Sun direction,
    # not at a symmetry node of the derivatives.
    r_sat = np.array([R_SAT_KM, 500.0, -300.0])
    a_analytic = third_body_acceleration(r_sat, r_sun, sun.mu_km3_s2)
    h = 1e-3  # km
    a_fd = np.zeros(3)
    for i in range(3):
        d = np.zeros(3); d[i] = h
        U_plus = third_body_perturbation_potential(r_sat + d, r_sun, sun.mu_km3_s2)
        U_minus = third_body_perturbation_potential(r_sat - d, r_sun, sun.mu_km3_s2)
        a_fd[i] = -(U_plus - U_minus) / (2.0 * h)
    rel = float(np.linalg.norm(a_fd - a_analytic) / np.linalg.norm(a_analytic))
    assert rel < 1e-5, f"FD vs analytic relative error {rel:.3e} > 1e-5"


# ---------------------------------------------------------------------------
# Group 4 -- propagator plumbing
# ---------------------------------------------------------------------------


def _circular_lmo_initial_state(mu: float, r_mag: float) -> np.ndarray:
    """Construct a 6-vector for a circular orbit in the equatorial plane."""
    v = np.sqrt(mu / r_mag)
    return np.array([r_mag, 0.0, 0.0, 0.0, v, 0.0])


def test_propagate_with_sun_perturbs_two_body_at_expected_magnitude():
    """``third_bodies=[Sun]`` shifts the trajectory by ~|a_Sun| / n^2.

    Naive estimate ``|a_Sun| * T^2 / 2`` overshoots because the sail
    samples the Sun direction from all sides over one orbit and most
    of the perturbation cancels. The leading PERIODIC displacement
    in a Hill-equation analysis scales as

        |delta_r| ~ |a_3b| / n^2

    For Mars LMO at 400 km altitude, n ~ 2 pi / T ~ 9.4e-4 rad/s,
    n^2 ~ 8.8e-7 s^-2, |a_Sun| ~ 1e-10 km/s^2, so the expected
    end-of-orbit residual is ~1.1e-4 km ~ 0.1 m. The exact value depends
    on the Sun-orbit-plane geometry.

    Test: pure two-body baseline vs two-body + Sun-third-body run,
    both at high_accuracy preset. Position difference at end of one
    orbit must lie in [0.1, 10] m. Lower bound rules out "Sun term
    omitted" (without a Sun term the integrator residual
    would be < 1e-3 m); upper bound rules out "Sun term magnitude
    differs by more than an order of magnitude".
    """
    et = spice.str2et(EPOCH_STR)
    mu = mars_gm_km3_per_s2()
    state0 = _circular_lmo_initial_state(mu, R_SAT_KM)
    T = 2.0 * np.pi * np.sqrt(R_SAT_KM ** 3 / mu)
    options = PropagationOptions.high_accuracy()
    base = propagate(
        state0, (0.0, T), epoch_et=et,
        mu_km3_s2=mu, options=options,
    )
    perturbed = propagate(
        state0, (0.0, T), epoch_et=et,
        mu_km3_s2=mu, third_bodies=[sun_third_body()], options=options,
    )
    delta = perturbed.state_km_kmps[-1, :3] - base.state_km_kmps[-1, :3]
    delta_mag_m = float(np.linalg.norm(delta)) * 1000.0  # km -> m
    assert 0.1 < delta_mag_m < 10.0, (
        f"|delta_r| at one orbit = {delta_mag_m:.3f} m, expected 0.1..10 m"
    )


def test_propagate_third_bodies_stacks_additively_with_full_harmonic_gravity():
    """``third_bodies=[Sun]`` stacks with ``gravity_degree=4`` and the
    delta vs the gravity-only control matches the Sun-tide magnitude.

    Confirms (a) the third-body term and the Cunningham gravity term
    are summed at the RHS level (not multiplied or short-circuited),
    and (b) the integrator handles the combined perturbation with
    the same stability as either alone.

    The Hill-equation scale is ``|a_3b|/n^2 ~ 0.1 m``. The position
    difference at the end of one orbit must lie in [0.1, 10] m.
    """
    et = spice.str2et(EPOCH_STR)
    from reflectors.gravity import mars_gravity_model
    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    state0 = _circular_lmo_initial_state(mu, R_SAT_KM)
    T = 2.0 * np.pi * np.sqrt(R_SAT_KM ** 3 / mu)
    options = PropagationOptions.high_accuracy()
    grav_only = propagate(
        state0, (0.0, T), epoch_et=et,
        gravity_degree=4, options=options,
    )
    grav_plus_sun = propagate(
        state0, (0.0, T), epoch_et=et,
        gravity_degree=4, third_bodies=[sun_third_body()], options=options,
    )
    delta = grav_plus_sun.state_km_kmps[-1, :3] - grav_only.state_km_kmps[-1, :3]
    delta_mag_m = float(np.linalg.norm(delta)) * 1000.0
    assert 0.1 < delta_mag_m < 10.0, (
        f"|delta_r| at one orbit = {delta_mag_m:.3f} m, expected 0.1..10 m"
    )
    # Metadata records the third-body spec.
    tb_meta = grav_plus_sun.metadata.get("third_bodies")
    assert tb_meta is not None and tb_meta[0]["label"] == "SUN"
    # Sun is not a Mars-system body, so no central-mu decoupling should
    # occur on this path; the metadata key must be absent.
    assert "central_mu_decouple" not in grav_plus_sun.metadata


def test_propagate_phobos_deimos_third_bodies_decouple_central_mu():
    """``gravity_degree=4 + [Sun, Phobos, Deimos]`` decouples the moons
    from the lumped Mars-system central mu, and the metadata records
    the decoupling exactly.

    Two pinned assertions:

    1. Wiring check (metadata): result.metadata['central_mu_decouple']
       reports naif_ids {401, 402} and mu_subtracted equal to
       (PHOBOS_GM_KONOPLIV_2020 + DEIMOS_GM_KONOPLIV_2020). This passes
       only when the decoupling logic in dynamics.propagate is wired
       through; it fails if the decoupling is removed, gated by
       an incorrect condition, or applied with an inconsistent GM source.

    2. TRAJECTORY SANITY (loose): |delta_r| at one orbit vs the
       gravity-only control sits in [0.1, 10] m -- the same band the
       Sun-only stacking test pins. This bound is intentionally loose:
       the removed central term is purely Keplerian, so its effect at a
       one-orbit baseline is sub-metre
       (~0.18 m along-track phase shift from the ~1e-8 fractional mu
       change). This assertion cannot by itself distinguish the decoupling
       behavior. It is here only as a
       guardrail against gross magnitude regressions in any of the
       perturbations (Sun tide zeroed, integrator blow-up,
       etc.). The metadata pin above guards the decoupling itself; the
       slow-regression envelope test
       (test_full_harmonic_plus_third_bodies_bounded_over_2000_orbits)
       carries the long-baseline trajectory pin, where the Keplerian
       phase shift accumulates to ~km-scale and is visible.
    """
    from reflectors.gravity import mars_gravity_model
    from reflectors.mars_constants import (
        DEIMOS_GM_KONOPLIV_2020_KM3_S2,
        PHOBOS_GM_KONOPLIV_2020_KM3_S2,
    )

    et = spice.str2et(EPOCH_STR)
    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    state0 = _circular_lmo_initial_state(mu, R_SAT_KM)
    T = 2.0 * np.pi * np.sqrt(R_SAT_KM ** 3 / mu)
    options = PropagationOptions.high_accuracy()
    grav_only = propagate(
        state0, (0.0, T), epoch_et=et,
        gravity_degree=4, options=options,
    )
    grav_plus_three = propagate(
        state0, (0.0, T), epoch_et=et,
        gravity_degree=4,
        third_bodies=[sun_third_body(), phobos_third_body(), deimos_third_body()],
        options=options,
    )
    delta = grav_plus_three.state_km_kmps[-1, :3] - grav_only.state_km_kmps[-1, :3]
    delta_mag_m = float(np.linalg.norm(delta)) * 1000.0
    assert 0.1 < delta_mag_m < 10.0, (
        f"|delta_r| at one orbit = {delta_mag_m:.3f} m, expected 0.1..10 m. "
        f"Loose guardrail; the decoupling correctness check is the metadata "
        f"assertion below."
    )
    decouple = grav_plus_three.metadata.get("central_mu_decouple")
    assert decouple is not None, (
        "metadata['central_mu_decouple'] must be populated when Phobos and "
        "Deimos are present alongside gravity"
    )
    assert sorted(decouple["naif_ids"]) == [401, 402], (
        f"unexpected decouple naif_ids: {decouple['naif_ids']}"
    )
    expected_subtraction = (
        PHOBOS_GM_KONOPLIV_2020_KM3_S2 + DEIMOS_GM_KONOPLIV_2020_KM3_S2
    )
    assert abs(decouple["mu_subtracted_km3_s2"] - expected_subtraction) < 1e-12, (
        f"mu_subtracted_km3_s2 = {decouple['mu_subtracted_km3_s2']:.10e} differs "
        f"from expected {expected_subtraction:.10e}"
    )
    # Central mu after decoupling matches Mars-planet-alone (Konopliv 2020).
    expected_central_mu = mu - expected_subtraction
    assert abs(decouple["mu_central_after_decouple_km3_s2"] - expected_central_mu) < 1e-12
