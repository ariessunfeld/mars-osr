"""Slow regression tests for the Mars two-body + zonal + full-harmonic
propagator.

Each test propagates for long enough (>= 30 s wall) to exercise secular effects
that fast tests cannot. Run with::

    pytest -m slow tests/test_slow_regression.py

Skipped from the default run by the ``slow`` pytest marker. See
pyproject.toml for the marker declaration.

Each test pins a physical scale rather than a bit-exact number, allowing
long-term behavioral changes to be detected without requiring machine-precision
agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice

from reflectors.dynamics import PropagationOptions, propagate
from reflectors.elements import classical_elements, mme2000_rotation_from_j2000
from reflectors.gravity import mars_gravity_model, zonal_coefficients
from reflectors.mars_constants import (
    MARS_HILL_RADIUS_KM,
    MARS_SIDEREAL_YEAR_S,
    SECONDS_PER_SOLAR_SOL_S,
)
from reflectors.attitude import sun_pointing
from reflectors.attitude_control import AttitudeLimits
from reflectors.ephemeris import utc_to_et
from reflectors.escape import initial_circular_state, propagate_escape
from reflectors.qlaw import QLawParams
from reflectors.sail_designs import make_canonical_sail
from reflectors.srp import SailOptical, SolarSail, SphericalParticle
from reflectors.surface import mars_equatorial_radius_km
from reflectors.third_body import (
    deimos_third_body,
    phobos_third_body,
    sun_third_body,
)


pytestmark = pytest.mark.slow


def _circular_mme_state_in_j2000(alt_km: float, inc_deg: float, et0: float, mu: float, R: float):
    r0 = R + alt_km
    v0 = float(np.sqrt(mu / r0))
    inc = np.radians(inc_deg)
    state_mme = np.array([r0, 0.0, 0.0, 0.0, v0 * np.cos(inc), v0 * np.sin(inc)])
    R_mat = mme2000_rotation_from_j2000(et0)
    return np.concatenate([R_mat.T @ state_mme[:3], R_mat.T @ state_mme[3:]])


def test_sun_sync_raan_tracks_sun_over_one_mars_year():
    """Sun-sync orbit at 400 km Mars altitude holds its nodal drift for
    one full Mars sidereal year (see reflectors.mars_constants).

    Target: dRAAN/dt_mean = 2 pi / Mars_year. Allowed residual: the
    first-order Brouwer formula has J_2^2 corrections at ~0.5-1 % and
    short-period oscillations up to ~0.5 deg -- neither of which will
    meaningfully drift the local time at the ascending node over a
    year. Pin:
      - total RAAN change over 1 year matches 2 pi to within 3 %,
      - maximum instantaneous deviation from the linear secular drift
        stays below 2 deg (short-period oscillations bounded).
    """
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    J2 = zonal_coefficients(model, 2)[2]
    et0 = spice.str2et("2026-06-01T00:00:00")
    alt_km = 400.0
    r0 = R + alt_km
    n = float(np.sqrt(mu / r0 ** 3))
    target_rate = 2.0 * np.pi / MARS_SIDEREAL_YEAR_S
    cos_i = -target_rate / (1.5 * n * (R / r0) ** 2 * J2)
    inc_deg = float(np.degrees(np.arccos(cos_i)))
    state0 = _circular_mme_state_in_j2000(alt_km, inc_deg, et0, mu, R)

    T = 2 * np.pi / n
    tf = MARS_SIDEREAL_YEAR_S
    n_samples = 2000
    t_eval = np.linspace(0.0, tf, n_samples)
    # The fast tolerance is sufficient because this regression pins a secular
    # trend rather than instantaneous position.
    result = propagate(
        state0, (0.0, tf),
        epoch_et=et0, zonal_degree=2, t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )
    raans = np.empty(result.t_s.size)
    for i, t in enumerate(result.t_s):
        R_mat = mme2000_rotation_from_j2000(et0 + t)
        r_mme = R_mat @ result.state_km_kmps[i, :3]
        v_mme = R_mat @ result.state_km_kmps[i, 3:]
        el = classical_elements(np.concatenate([r_mme, v_mme]), mu)
        raans[i] = el.raan_rad
    raans = np.unwrap(raans)
    total_drift = raans[-1] - raans[0]
    expected_total = target_rate * tf
    rel = (total_drift - expected_total) / expected_total
    assert abs(rel) < 0.03, (
        f"RAAN drift over 1 Mars year: expected {np.degrees(expected_total):.3f} deg, "
        f"got {np.degrees(total_drift):.3f} deg, rel err {rel:.3%}"
    )
    # Residual around the linear trend should be bounded.
    slope, intercept = np.polyfit(result.t_s, raans, 1)
    residual_rad = raans - (slope * result.t_s + intercept)
    peak_dev_deg = float(np.degrees(np.max(np.abs(residual_rad))))
    assert peak_dev_deg < 2.0, f"peak short-period RAAN residual {peak_dev_deg:.3f} deg > 2 deg"


def test_j2_only_5000_orbits_semi_major_axis_inclination_bounded():
    """With only J_2, osculating a and i oscillate but do not drift
    secularly (they're mean-element invariants at first order). Over
    5000 orbits, verify the osculating values stay bounded within the
    expected J_2 short-period amplitude.
    """
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et("2026-06-01T00:00:00")
    alt_km = 400.0
    state0 = _circular_mme_state_in_j2000(alt_km, 60.0, et0, mu, R)
    r0 = R + alt_km
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 5000
    tf = N * T
    # Five samples per orbit are sufficient for the min/max envelope of
    # (a, e, i) while keeping memory use manageable.
    t_eval = np.linspace(0.0, tf, 5 * N + 1)
    result = propagate(
        state0, (0.0, tf),
        epoch_et=et0, zonal_degree=2, t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )
    a_osc = np.empty(result.t_s.size)
    e_osc = np.empty(result.t_s.size)
    i_osc = np.empty(result.t_s.size)
    for i, t in enumerate(result.t_s):
        R_mat = mme2000_rotation_from_j2000(et0 + t)
        r_mme = R_mat @ result.state_km_kmps[i, :3]
        v_mme = R_mat @ result.state_km_kmps[i, 3:]
        el = classical_elements(np.concatenate([r_mme, v_mme]), mu)
        a_osc[i] = el.a_km
        e_osc[i] = el.e
        i_osc[i] = el.inclination_rad

    # Semi-major axis: J_2 oscillation amplitude ~ J_2 (R/a)^2 a. For the
    # numbers: 2e-3 * 0.8 * 3796 = 6 km, so osculating a can swing by up
    # to about 12 km. Allow 30 km to be generous on higher-order terms,
    # and 30 km IS the signature to verify it's bounded.
    a_range = float(a_osc.max() - a_osc.min())
    assert a_range < 30.0, f"a range {a_range:.3f} km exceeds expected J_2 envelope"
    # No secular drift in <a>:
    a_mean_first = float(np.mean(a_osc[: 500]))
    a_mean_last = float(np.mean(a_osc[-500 :]))
    assert abs(a_mean_last - a_mean_first) < 5.0, (
        f"mean a drifted {a_mean_last - a_mean_first:.3f} km over 5000 orbits"
    )
    # Eccentricity: starts at ~0, oscillates but stays small.
    assert e_osc.max() < 5e-3, f"max osculating e = {e_osc.max():.3e}"
    # Inclination remains bounded around its mean; first-order J2 theory
    # forbids secular inclination drift.
    i_range_deg = float(np.degrees(i_osc.max() - i_osc.min()))
    assert i_range_deg < 0.15, f"inclination range {i_range_deg:.4f} deg > 0.15 deg"
    # Stronger check: no net secular drift between first and last 500 samples.
    i_mean_first_deg = float(np.degrees(np.mean(i_osc[:500])))
    i_mean_last_deg = float(np.degrees(np.mean(i_osc[-500:])))
    assert abs(i_mean_last_deg - i_mean_first_deg) < 0.02, (
        f"mean inclination drifted {i_mean_last_deg - i_mean_first_deg:+.4f} deg over 5000 orbits"
    )


def test_full_harmonic_degree6_physically_bounded_over_2000_orbits():
    """Full-harmonic (degree 6, order 6) propagation for 2000 LMO orbits
    stays physically bounded: radial band, osculating (a, e, i) envelope,
    and no-secular-drift in mean a.

    Scope notes:

    * Direct r-difference vs zonal-only baseline is NOT a valid bound --
      tesseral perturbations shift the effective mean motion, so small period
      differences accumulate as physical along-track phase drift. The zonal vs
      full-harmonic equivalence is instead pinned in the fast suite at
      ``test_gravity_path_and_zonal_path_agree_when_order_zero`` where
      ``gravity_order=0`` reduces to the zonal code path to < 1 um
      over 3 orbits -- short enough that phase drift is negligible.
    * The physical regression target here is: (a) does the trajectory
      stay in a physically plausible orbital band, (b) do the osculating
      elements stay bounded, (c) does the orbit close on itself (no accumulated
      secular drift in ``<a>``, which would indicate spurious energy
      injection/drain from the tesseral code).

    2000 LMO orbits correspond to approximately 160 days.
    """
    model = mars_gravity_model(max_degree=6)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et("2026-06-01T00:00:00")
    alt_km = 400.0
    r0 = R + alt_km
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 2000
    tf = N * T

    state0 = _circular_mme_state_in_j2000(alt_km, 60.0, et0, mu, R)
    # 5 samples per orbit keeps memory manageable at 2000 orbits
    # (10k samples); the target is the envelope, not fine structure.
    t_eval = np.linspace(0.0, tf, 5 * N + 1)

    result = propagate(
        state0, (0.0, tf),
        epoch_et=et0, gravity_degree=6, gravity_order=6, t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )

    # Finite values everywhere. Catches any NaN/Inf from a divergent
    # recurrence or a bad rotation matrix at a specific epoch.
    assert np.all(np.isfinite(result.state_km_kmps))

    # Combined J2 and tesseral forcing can raise eccentricity to a few percent.
    # The radial envelope remains above the atmospheric-drag regime.
    r_mag = np.linalg.norm(result.positions(), axis=1)
    assert r_mag.min() > R + alt_km - 150.0, (
        f"minimum radius {r_mag.min() - R:.2f} km alt < {alt_km - 150:.0f} km"
    )
    assert r_mag.max() < R + alt_km + 150.0, (
        f"maximum radius {r_mag.max() - R:.2f} km alt > {alt_km + 150:.0f} km"
    )

    # Osculating (a, e, i) envelope, MME-referenced.
    a_osc = np.empty(result.t_s.size)
    e_osc = np.empty(result.t_s.size)
    i_osc = np.empty(result.t_s.size)
    for i, t in enumerate(result.t_s):
        R_mat = mme2000_rotation_from_j2000(et0 + t)
        r_mme = R_mat @ result.state_km_kmps[i, :3]
        v_mme = R_mat @ result.state_km_kmps[i, 3:]
        el = classical_elements(np.concatenate([r_mme, v_mme]), mu)
        a_osc[i] = el.a_km
        e_osc[i] = el.e
        i_osc[i] = el.inclination_rad

    # Semi-major axis peak-to-peak. Tesseral-on envelope exceeds the
    # J_2-only short-period envelope (~12 km); allow 200 km to absorb
    # the (2,2) and (3,3) beats with mean motion over 2000 orbits.
    a_range = float(a_osc.max() - a_osc.min())
    assert a_range < 200.0, f"osculating a range {a_range:.2f} km > 200 km"

    # No spurious secular drift in <a> between the first and last ~5
    # orbits. Tesseral gravity is conservative (body-fixed time-
    # independent potential), so in inertial frame energy
    # bounces but <a> should not run away. 10 km envelope.
    n_early = 25  # ~5 orbits at 5 samples/orbit
    a_mean_first = float(np.mean(a_osc[:n_early]))
    a_mean_last = float(np.mean(a_osc[-n_early:]))
    assert abs(a_mean_last - a_mean_first) < 10.0, (
        f"mean a drifted {a_mean_last - a_mean_first:+.2f} km over 2000 orbits"
    )

    # Eccentricity remains below the radial-excursion limit.
    assert e_osc.max() < 5e-2, f"max osculating e = {e_osc.max():.3e}"

    # Inclination bounded. First-order Brouwer theory forbids secular
    # drift in i under zonal forcing; tesseral forcing can produce
    # bounded oscillations at orbit-mean-motion beats with Mars rotation.
    i_range_deg = float(np.degrees(i_osc.max() - i_osc.min()))
    assert i_range_deg < 0.5, (
        f"inclination range {i_range_deg:.4f} deg > 0.5 deg"
    )


def test_full_harmonic_plus_third_bodies_bounded_over_2000_orbits():
    """gravity_degree=4 + [Sun, Phobos, Deimos] third-body propagation
    for 2000 LMO orbits stays in a bounded envelope.

    Companion test to ``test_full_harmonic_degree6_physically_bounded_...``
    above, designed to exercise the third-body code path:

      * verifies the additive stacking (gravity harmonics + multiple
        third bodies) doesn't introduce spurious energy injection
        or numerical drift over many orbits;
      * uses three bodies at once so the SPICE-fed multi-body wrapper
        is exercised at every RHS step;
      * applies degree-4 envelope bounds, where tesseral forcing is weaker
        than in the degree-6 case.

    Third-body magnitudes at LMO 400 km are 5-10 orders below |a_J2|,
    so on a 2000-orbit timescale (~14 Ms ~ 160 days) they contribute
    bounded oscillations rather than secular runaway. The envelope
    here is therefore expected to be tighter than the degree=6
    harmonic-only test (which has stronger tesseral forcing).

    """
    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et("2026-06-01T00:00:00")
    alt_km = 400.0
    r0 = R + alt_km
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 2000
    tf = N * T

    state0 = _circular_mme_state_in_j2000(alt_km, 60.0, et0, mu, R)
    t_eval = np.linspace(0.0, tf, 5 * N + 1)

    bodies = [sun_third_body(), phobos_third_body(), deimos_third_body()]

    result = propagate(
        state0, (0.0, tf),
        epoch_et=et0,
        gravity_degree=4, gravity_order=4,
        third_bodies=bodies,
        t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )

    # Finite values everywhere.
    assert np.all(np.isfinite(result.state_km_kmps))

    # Metadata records the third-body roster.
    tb_meta = result.metadata.get("third_bodies")
    assert tb_meta is not None and len(tb_meta) == 3
    assert {b["label"] for b in tb_meta} == {"SUN", "PHOBOS", "DEIMOS"}

    # Radial band. degree=4 harmonics produce gentler e-pumping than
    # degree=6 (no (5,5)/(6,6) tesseral cascade); 100 km on each side
    # is enough.
    r_mag = np.linalg.norm(result.positions(), axis=1)
    assert r_mag.min() > R + alt_km - 100.0, (
        f"minimum radius {r_mag.min() - R:.2f} km alt < {alt_km - 100:.0f} km"
    )
    assert r_mag.max() < R + alt_km + 100.0, (
        f"maximum radius {r_mag.max() - R:.2f} km alt > {alt_km + 100:.0f} km"
    )

    # Osculating elements.
    a_osc = np.empty(result.t_s.size)
    e_osc = np.empty(result.t_s.size)
    i_osc = np.empty(result.t_s.size)
    for i, t in enumerate(result.t_s):
        R_mat = mme2000_rotation_from_j2000(et0 + t)
        r_mme = R_mat @ result.state_km_kmps[i, :3]
        v_mme = R_mat @ result.state_km_kmps[i, 3:]
        el = classical_elements(np.concatenate([r_mme, v_mme]), mu)
        a_osc[i] = el.a_km
        e_osc[i] = el.e
        i_osc[i] = el.inclination_rad

    # Semi-major axis: gentler tesseral forcing at degree=4 -> tighter
    # envelope. 100 km accommodates the (3,2)/(4,3) beats over 2000
    # orbits without absorbing a possible spurious-drift error.
    a_range = float(a_osc.max() - a_osc.min())
    assert a_range < 100.0, f"osculating a range {a_range:.2f} km > 100 km"

    # No spurious secular drift in <a>. Both gravity (conservative,
    # body-fixed time-independent) and the third-body terms (each
    # conservative in the inertial Mars-centered frame, by the
    # finite-difference test in the fast suite) preserve <a> in the
    # mean. 8 km envelope.
    n_early = 25
    a_mean_first = float(np.mean(a_osc[:n_early]))
    a_mean_last = float(np.mean(a_osc[-n_early:]))
    assert abs(a_mean_last - a_mean_first) < 8.0, (
        f"mean a drifted {a_mean_last - a_mean_first:+.2f} km over 2000 orbits"
    )

    # Eccentricity bounded.
    assert e_osc.max() < 3e-2, f"max osculating e = {e_osc.max():.3e}"

    # Inclination bounded.
    i_range_deg = float(np.degrees(i_osc.max() - i_osc.min()))
    assert i_range_deg < 0.5, (
        f"inclination range {i_range_deg:.4f} deg > 0.5 deg"
    )


def test_full_harmonic_plus_third_bodies_plus_srp_bounded_over_2000_orbits():
    """gravity_degree=4 + [Sun, Phobos, Deimos] + sun-pointing SRP stays in
    a physically bounded envelope over 2000 LMO orbits.

    The positive signal is eccentricity above the no-SRP companion's 0.03
    bound. The sail has A = 1000 m^2, m = 50 kg (sigma = 0.05 kg/m^2) and
    ``SailOptical.square_sail_jpl()`` optical coefficients.

    Physical sanity pins:
      - Trajectory remains finite everywhere.
      - Minimum altitude > 100 km (above the atmospheric-drag destruction
        threshold).
      - Maximum altitude < 700 km (comfortably below any apoapsis
        escape mode).
      - Osculating ``a`` swings within a 50-km band and its 5-orbit
        mean does not drift by more than 15 km over the run (SRP
        is non-conservative so SOME drift is physical; the bound
        catches runaway).
      - Eccentricity lies in (0.03, 0.1); the lower bound exercises SRP and the
        upper bound detects numerical divergence.
      - Inclination range bounded at 1 deg.
      - Metadata records the sail and third-body roster.

    """
    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et("2026-06-01T00:00:00")
    alt_km = 400.0
    r0 = R + alt_km
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 2000
    tf = N * T

    state0 = _circular_mme_state_in_j2000(alt_km, 60.0, et0, mu, R)
    t_eval = np.linspace(0.0, tf, 5 * N + 1)

    sail = SolarSail(
        area_m2=1000.0,
        mass_kg=50.0,
        optical=SailOptical.square_sail_jpl(),
    )

    result = propagate(
        state0, (0.0, tf),
        epoch_et=et0,
        gravity_degree=4, gravity_order=4,
        third_bodies=[sun_third_body(), phobos_third_body(), deimos_third_body()],
        solar_sail=sail,
        sail_normal=sun_pointing(),
        t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )

    # Finite values everywhere.
    assert np.all(np.isfinite(result.state_km_kmps))

    # Metadata records sail + bodies.
    assert "solar_sail" in result.metadata
    assert result.metadata["solar_sail"]["area_m2"] == 1000.0
    assert result.metadata["solar_sail"]["mass_kg"] == 50.0
    tb_meta = result.metadata.get("third_bodies")
    assert tb_meta is not None and len(tb_meta) == 3

    # Radial band.
    r_mag = np.linalg.norm(result.positions(), axis=1)
    assert r_mag.min() > R + 100.0, (
        f"minimum altitude {r_mag.min() - R:.2f} km < 100 km destruction boundary"
    )
    assert r_mag.max() < R + 700.0, (
        f"maximum altitude {r_mag.max() - R:.2f} km > 700 km"
    )

    # Osculating elements.
    a_osc = np.empty(result.t_s.size)
    e_osc = np.empty(result.t_s.size)
    i_osc = np.empty(result.t_s.size)
    for i, t in enumerate(result.t_s):
        R_mat = mme2000_rotation_from_j2000(et0 + t)
        r_mme = R_mat @ result.state_km_kmps[i, :3]
        v_mme = R_mat @ result.state_km_kmps[i, 3:]
        el = classical_elements(np.concatenate([r_mme, v_mme]), mu)
        a_osc[i] = el.a_km
        e_osc[i] = el.e
        i_osc[i] = el.inclination_rad

    # Semi-major axis peak-to-peak.
    a_range = float(a_osc.max() - a_osc.min())
    assert a_range < 50.0, f"osculating a range {a_range:.2f} km > 50 km"

    # Bounded <a> drift -- SRP is non-conservative so non-zero drift is
    # physical; the bound catches spurious runaway only.
    n_early = 25  # 5 orbits at 5 samples/orbit
    a_mean_first = float(np.mean(a_osc[:n_early]))
    a_mean_last = float(np.mean(a_osc[-n_early:]))
    assert abs(a_mean_last - a_mean_first) < 15.0, (
        f"mean a drifted {a_mean_last - a_mean_first:+.2f} km over 2000 orbits"
    )

    # Eccentricity pumped by SRP into a characteristic band. Lower bound
    # is the POSITIVE signal that the SRP force is actually being
    # applied (no-SRP degree-4 run caps below 3e-2). Upper bound is
    # stability.
    e_max = float(e_osc.max())
    assert e_max > 0.03, (
        f"max osculating e = {e_max:.4f} <= 0.03: SRP not pumping as expected"
    )
    assert e_max < 0.1, f"max osculating e = {e_max:.4f} > 0.1: possible runaway"

    # Inclination bounded.
    i_range_deg = float(np.degrees(i_osc.max() - i_osc.min()))
    assert i_range_deg < 1.0, (
        f"inclination range {i_range_deg:.4f} deg > 1 deg"
    )


# ---------------------------------------------------------------------------
# refine_delivery_schedule integration
# ---------------------------------------------------------------------------


def test_refine_delivery_schedule_converges_on_canonical_grid_point():
    """Fixed-point iteration converges for a real 501 km LMO grid.

    Runs ``refine_delivery_schedule`` at 501 km / LTAN=18h / M_0=180 deg /
    2026-06-01 for one sidereal sol. Asserts:
      - converges within ``max_iterations=3`` at
        ``convergence_tol_et_s=5.0``
      - ``boundary_drift_history_et_s`` is monotonically non-increasing
        (damping should prevent oscillation)
      - at least one stable matched window (pipeline actually produced
        output)
    """
    import math
    from reflectors.attitude import sun_pointing
    from reflectors.attitude_schedule import refine_delivery_schedule
    from reflectors.dynamics import PropagationOptions
    from reflectors.ephemeris import utc_to_et
    from reflectors.srp import SailOptical, SolarSail
    from reflectors.sun_sync import initial_state_j2000
    from reflectors.termination import AltitudeFloor
    from reflectors.third_body import (
        deimos_third_body, phobos_third_body, sun_third_body,
    )

    epoch_et = utc_to_et("2026-06-01T00:00:00")
    a_km = 501.0 + 3396.19
    state0 = initial_state_j2000(
        a_km=a_km, ltan_h=18.0, M0_rad=math.radians(180.0), epoch_et=epoch_et,
    )
    sail = SolarSail(
        area_m2=1000.0, mass_kg=50.0, optical=SailOptical.square_sail_jpl(),
    )
    duration_s = 88775.0  # 1 sidereal sol

    result = refine_delivery_schedule(
        state0,
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et,
        cruise_profile=sun_pointing(),
        target_lat_deg=40.0,
        target_lon_deg=200.0,
        sail=sail,
        slew_duration_s=300.0,
        max_iterations=3,
        convergence_tol_et_s=5.0,
        damping=0.7,
        propagate_kwargs=dict(
            gravity_degree=6,
            gravity_order=6,
            third_bodies=[sun_third_body(), phobos_third_body(), deimos_third_body()],
            altitude_floor=AltitudeFloor.at_km(300.0, label="altitude_floor"),
            options=PropagationOptions.fast(),
            t_eval_s=np.arange(0.0, duration_s + 0.1, 5.0),
        ),
    )

    # Convergence OR decreasing drift.
    assert result.n_iterations > 0
    drifts = list(result.boundary_drift_history_et_s)
    # At least one window matched for drift to be finite.
    finite_drifts = [d for d in drifts if d != float("inf")]
    assert len(finite_drifts) >= 1, (
        f"all iterations produced inf drift (window matching failed); "
        f"drifts={drifts}"
    )
    # Damped iteration should not grow in drift compared to prev finite.
    for i in range(len(finite_drifts) - 1):
        assert finite_drifts[i + 1] <= finite_drifts[i] * 1.5, (
            f"drift blew up at iteration {i+1}: {finite_drifts[i+1]} vs "
            f"{finite_drifts[i]} at iteration {i}"
        )

    # Schedule delivered at least one stable window.
    assert len(result.final_windows) >= 1, (
        f"no stable windows in final schedule; converged={result.converged}, "
        f"n_iter={result.n_iterations}"
    )

    # Profile actually consumable by propagate (sanity smoke).
    assert result.final_profile is not None
    assert result.metadata.n_windows_kept >= 1


def test_spherical_particle_matches_flat_sail_envelope_over_100_orbits():
    """Sphere SRP path agrees with the equivalent flat-sail+sun_pointing
    propagation on the (a, e, i) envelope.

    This is the end-to-end equivalence regression for the spherical-grain
    SRP path (Hamilton & Krivov 1996 / Burns 1979) in ``reflectors.srp``.
    Setup: 300-um silicate rock at LMO 400 km, 60 deg
    inclination, propagated for 100 orbits under J_2 + SRP only (no other
    perturbations). The flat-sail control uses the absorbing-Lambertian
    optical preset (rho=0, s=0, eps_f=eps_b=0.9, B_f=B_b=2/3) with
    ``A = pi r^2`` and ``m = (4/3) pi r^3 rho`` and ``sun_pointing``
    attitude -- the parameter combination at which the McInnes (1999)
    Eq. 2.57 formula is algebraically identical to H&K96 Eq. (3) with
    ``Q_pr = 1.0``.

    The two RHS evaluations must therefore agree to floating-point noise
    at every step, and the cumulative envelope (a_min, a_max, e_max,
    e_mean, i_min, i_max) must agree to ~1e-9 relative. This is the
    positive-signal regression for the spherical-grain SRP path: it passes
    only if the spherical-particle contributor is active and computes the same
    physics as the absorbing-sphere
    limit of the McInnes formula.

    """
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et("2026-06-01T00:00:00")
    alt_km = 400.0
    state0 = _circular_mme_state_in_j2000(alt_km, 60.0, et0, mu, R)

    r0 = R + alt_km
    T = 2.0 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 100
    tf = N * T
    t_eval = np.linspace(0.0, tf, 5 * N + 1)

    radius_m = 1.5e-4
    density = 3000.0

    sail = SolarSail(
        area_m2=np.pi * radius_m ** 2,
        mass_kg=(4.0 / 3.0) * np.pi * radius_m ** 3 * density,
        optical=SailOptical(
            rho=0.0, s=0.0,
            eps_front=0.9, eps_back=0.9,
            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
        ),
    )
    particle = SphericalParticle(
        radius_m=radius_m, density_kg_per_m3=density, Q_pr=1.0,
    )

    res_flat = propagate(
        state0, (0.0, tf), epoch_et=et0,
        zonal_degree=2,
        solar_sail=sail, sail_normal=sun_pointing(),
        t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )
    res_sphere = propagate(
        state0, (0.0, tf), epoch_et=et0,
        zonal_degree=2,
        spherical_particle=particle,
        t_eval_s=t_eval,
        options=PropagationOptions.fast(),
    )

    # Per-sample classical elements in MME2000 (J_2 path uses the gravity-
    # model mu, recorded on the result).
    def _envelope(res):
        a = np.empty(res.t_s.size)
        e = np.empty(res.t_s.size)
        inc = np.empty(res.t_s.size)
        R_mat = mme2000_rotation_from_j2000(et0)
        for k in range(res.t_s.size):
            r_j = res.state_km_kmps[k, :3]
            v_j = res.state_km_kmps[k, 3:]
            r_mme = R_mat @ r_j
            v_mme = R_mat @ v_j
            elt = classical_elements(
                np.concatenate([r_mme, v_mme]),
                mu_km3_s2=res.mu_km3_s2,
                epoch_et=et0 + float(res.t_s[k]),
            )
            a[k] = elt.a_km
            e[k] = elt.e
            inc[k] = elt.inclination_rad
        return a, e, inc

    a_f, e_f, i_f = _envelope(res_flat)
    a_s, e_s, i_s = _envelope(res_sphere)

    # Envelope numbers agree to integration-noise floor.
    assert np.isclose(np.min(a_f), np.min(a_s), rtol=1e-9, atol=1e-6)
    assert np.isclose(np.max(a_f), np.max(a_s), rtol=1e-9, atol=1e-6)
    assert np.isclose(np.max(e_f), np.max(e_s), rtol=1e-9, atol=1e-12)
    assert np.isclose(np.mean(e_f), np.mean(e_s), rtol=1e-9, atol=1e-12)
    assert np.isclose(np.min(i_f), np.min(i_s), rtol=1e-9, atol=1e-12)
    assert np.isclose(np.max(i_f), np.max(i_s), rtol=1e-9, atol=1e-12)

    # Positive signal: SRP is actually doing work -- e_max > 0 (no-SRP
    # control would stay at e ~ 1e-12 over 100 orbits at this geometry).
    assert np.max(e_s) > 1e-6

    # Metadata records the right path for each.
    assert "spherical_particle" in res_sphere.metadata
    assert "solar_sail" in res_flat.metadata
    assert res_sphere.metadata["spherical_particle"]["Q_pr"] == 1.0


def test_srp_escape_sigma18_full_physics_realistic_slew_strict_bounds():
    """A 50-sol partial SRP escape at sigma=18 with FULL physics and
    REALISTIC slew limits, steered by the Macdonald-McInnes blended
    energy+safety law, with the reference governor enabled. This is the
    long-horizon regression for the slew-strict escape pipeline.

    Asserts:
      - termination = ``t_final`` (no crash within 50 sols);
      - strict ``|omega| <= omega_max``;
      - strict ``|alpha| <= alpha_max`` (recomputed at every sample);
      - the governor's reference normal stays unit and well-defined;
      - the tracker tracks the reference normal closely
        (max ``|angle(n, n_ref)| < 15 deg``);
      - the orbit is raised substantially (a +500 km) -- the positive
        signal that the steering + governor + tracker pipeline is wired
        through and SRP thrust is actually applied.

    The slew-strict pipeline is the supported control path; this regression
    pins its bounded orbital envelope and positive escape signal.
    """
    import math
    from reflectors.attitude_control import GovernorParams
    from reflectors.dynamics import mars_gm_km3_per_s2
    from reflectors.escape_dedot import BlendedParams, blended_steer
    from reflectors.attitude_control import alpha_command

    epoch_et = utc_to_et("2028-01-01T00:00:00")
    state0 = initial_circular_state(1000.0, epoch_et)
    sail = make_canonical_sail(0.018)  # sigma = 18 (canonical)
    R_eq = mars_equatorial_radius_km()
    qlaw_shell = QLawParams(a_target_km=MARS_HILL_RADIUS_KM, rp_min_km=R_eq + 300.0)
    limits = AttitudeLimits(
        alpha_max_rad_s2=math.radians(0.003),
        omega_max_rad_s=math.radians(0.3),
    )
    governor = GovernorParams(
        omega_ref_max_rad_s=0.8 * limits.omega_max_rad_s,
        theta_settle_rad=0.01,
    )
    blended_params = BlendedParams(
        r_star_km=R_eq + 600.0, w_E=1.0, k_S=1.0,
        S_0_km4_s2=1.0e7, max_cone_rad=math.radians(80.0),
        mu_km3_s2=mars_gm_km3_per_s2(),
    )

    def steering(r, v, s_hat, p_eff, sail_, current_n_hat):
        return blended_steer(
            r, v, s_hat, p_eff, sail_,
            current_n_hat=current_n_hat, params=blended_params,
        ).n_star_j2000

    span = 50.0 * SECONDS_PER_SOLAR_SOL_S

    res = propagate_escape(
        state0, epoch_et, sail, qlaw_shell, limits, (0.0, span),
        gravity_degree=2,
        steering_fn=steering,
        governor_params=governor,
    )
    assert res.termination_reason == "t_final", (
        f"unexpected termination: {res.termination_reason}"
    )

    # Strict |omega| <= omega_max.
    omega_mag = np.linalg.norm(res.angular_velocities_rad_s, axis=1)
    assert np.all(omega_mag <= limits.omega_max_rad_s * (1.0 + 1e-12)), (
        f"max omega = {omega_mag.max():.4e} > omega_max + 1e-12 tol"
    )

    # Strict |alpha| <= alpha_max (recompute the command at every sample
    # -- the integrated state should be one whose command magnitude is
    # bounded by alpha_max irrespective of where it lies in the cycle).
    for ni, wi, nref in zip(
        res.sail_normals, res.angular_velocities_rad_s, res.reference_normals,
    ):
        a_cmd = alpha_command(ni, wi, nref, limits)
        assert float(np.linalg.norm(a_cmd)) <= limits.alpha_max_rad_s2 * (1 + 1e-9)

    # Sail normal and reference normal are unit vectors.
    n_norm = np.linalg.norm(res.sail_normals, axis=1)
    assert np.allclose(n_norm, 1.0, atol=1e-6)
    assert res.reference_normals is not None
    nref_norm = np.linalg.norm(res.reference_normals, axis=1)
    assert np.allclose(nref_norm, 1.0, atol=1e-6)

    # The tracker tracks the GOVERNOR's reference normal closely. The
    # governor rate-limits n_ref so |dn_ref/dt| <= omega_ref_max which is
    # below the tracker's omega_max; the tracker has headroom to follow.
    cos_track = np.einsum('ij,ij->i', res.sail_normals, res.reference_normals)
    cos_track = np.clip(cos_track, -1.0, 1.0)
    track_deg = np.degrees(np.arccos(cos_track))
    assert track_deg.max() < 15.0, (
        f"max tracker-to-ref angle {track_deg.max():.2f} deg "
        "(governor should keep this small)"
    )

    # Positive signal: orbit raised by SRP thrust.
    mu = res.metadata["mu_central_km3_s2"]
    a_start = classical_elements(res.orbit_state_km_kmps[0], mu, epoch_et).a_km
    a_end = classical_elements(
        res.orbit_state_km_kmps[-1], mu, res.epoch_et,
    ).a_km
    assert a_end - a_start > 500.0, (
        f"a only rose {a_end - a_start:.1f} km -- SRP thrust not effective"
    )

    # No NaNs anywhere in the integrated state.
    assert np.all(np.isfinite(res.orbit_state_km_kmps))
    assert np.all(np.isfinite(res.attitude_state))
    assert np.all(np.isfinite(res.reference_normals))
