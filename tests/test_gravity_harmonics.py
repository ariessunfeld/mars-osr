"""Fast tests for the Cunningham-1970 full spherical-harmonic Mars gravity.

Organised in four groups:

  1. Numerical identities that MUST hold by construction:
      - Cunningham central-term reproduces two-body at machine precision.
      - Cunningham zonal slice (max_order=0) equals the independent
        scalar-Legendre ``zonal_acceleration_body_fixed`` at machine precision.
      - Frame round-trip: inertial wrapper = R.T @ body_fixed(R @ r).

  2. Anchors against the MRO120F file header and Mars geoid literature:
      - Parsed C_bar / S_bar match Konopliv-table values for (n, m) in
        {(2,0), (2,1), (2,2), (3,1), (3,3)}.
      - Closed-form C_{2,2} contribution at equator prime meridian matches
        ``-9 mu C_{2,2} / r^4`` via the Cunningham path.

  3. Physical properties:
      - Finite-difference of the potential ``V`` via ``_cunningham_V``
        recovers ``-grad U``-type acceleration at ~1e-6 relative (the
        Richardson truncation floor, not the code).
      - Pole safety: inputs at the body-fixed pole do not blow up.
      - Axisymmetry breaker: L dot pole is NOT conserved under tesserals,
        complementary to the existing zonal-only conservation test.

  4. Propagator plumbing:
      - ``propagate(gravity_degree=2, gravity_order=0)`` and
        ``propagate(zonal_degree=2)`` produce the same trajectory.
      - Mutual-exclusion ValueError when both kwargs are set.
      - Metadata reports the chosen path.

All tests use the MRO120F file via ``mars_gravity_model``.
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice

from reflectors.dynamics import (
    PropagationOptions,
    mars_gm_km3_per_s2,
    propagate,
    two_body_acceleration,
)
from reflectors.elements import mme2000_rotation_from_j2000
from reflectors.gravity import (
    CUNNINGHAM_BACKEND_NUMBA,
    CUNNINGHAM_BACKEND_PYTHON,
    _cunningham_V,
    _cunningham_W,
    mars_gravity_acceleration_body_fixed,
    mars_gravity_acceleration_inertial,
    mars_gravity_model,
    normalization_factor,
    normalized_to_unnormalized,
    zonal_acceleration_body_fixed,
    zonal_coefficients,
)


EPOCH_STR = "2026-06-01T00:00:00"


# ---------------------------------------------------------------------------
# 1. Numerical identities
# ---------------------------------------------------------------------------


def test_cunningham_central_term_matches_two_body_to_machine_precision():
    """With max_degree=0 and include_central=True, Cunningham returns
    -mu r/|r|^3 exactly (the (0,0) harmonic with C_{0,0}=1). Verified at
    10 random positions around Mars."""
    model = mars_gravity_model(max_degree=2)
    mu = model.mu_km3_s2
    rng = np.random.default_rng(1)
    max_rel = 0.0
    for _ in range(10):
        r = rng.uniform(-1, 1, 3) * 1e4  # up to 10 Mm scales
        if np.linalg.norm(r) < 1e3:
            continue
        a_cun = mars_gravity_acceleration_body_fixed(
            r, model, max_degree=0, max_order=0, include_central=True
        )
        a_tb = two_body_acceleration(r, mu)
        rel = float(np.linalg.norm(a_cun - a_tb) / np.linalg.norm(a_tb))
        max_rel = max(max_rel, rel)
    # machine precision; 1e-14 is a safety margin.
    assert max_rel < 1e-14, f"central-term rel err {max_rel:.3e}"


@pytest.mark.parametrize("max_deg", [2, 4, 10])
def test_cunningham_zonal_slice_matches_scalar_legendre(max_deg):
    """At max_order=0, the Cunningham path agrees with the independent
    scalar-Legendre zonal code to machine precision. 100 random points.

    This identity is proved analytically by
    ``P_{n+1}'(u) = (n+1) P_n(u) + u P_n'(u)`` -- a Legendre-polynomial
    recurrence. If it ever fails numerically, either a recurrence index
    is off or a sign has flipped in the derivative formulae.
    """
    model = mars_gravity_model(max_degree=max_deg)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    J = zonal_coefficients(model, max_deg)
    rng = np.random.default_rng(2)
    max_rel = 0.0
    for _ in range(100):
        r = rng.uniform(-1, 1, 3) * 1e4
        if np.linalg.norm(r) < 1e3:
            continue
        a_cun = mars_gravity_acceleration_body_fixed(
            r, model, max_degree=max_deg, max_order=0, include_central=False
        )
        a_zon = zonal_acceleration_body_fixed(r, mu, R, J)
        rel = float(np.linalg.norm(a_cun - a_zon) / np.linalg.norm(a_zon))
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-13, (
        f"zonal slice rel err {max_rel:.3e} exceeds 1e-13 at max_deg={max_deg}"
    )


def test_inertial_wrapper_is_consistent_with_body_fixed():
    """At et0, ``mars_gravity_acceleration_inertial`` equals
    R.T @ body_fixed(R @ r) where R = pxform(J2000, IAU_MARS, et0)."""
    model = mars_gravity_model(max_degree=4)
    et = spice.str2et(EPOCH_STR)
    rng = np.random.default_rng(3)
    for _ in range(10):
        r = rng.uniform(-1, 1, 3) * 5e3
        if np.linalg.norm(r) < 1e3:
            continue
        M = np.asarray(spice.pxform("J2000", "IAU_MARS", et))
        a_bf = mars_gravity_acceleration_body_fixed(
            M @ r, model, max_degree=4, max_order=4
        )
        a_in = mars_gravity_acceleration_inertial(
            r, et, model, max_degree=4, max_order=4
        )
        assert np.allclose(a_in, M.T @ a_bf, atol=1e-14)


def test_normalization_factor_round_trip():
    """N_{n,m} converts between normalized and unnormalized consistently:
    building (C_bar, S_bar) -> (C, S) -> (C_bar, S_bar) is identity."""
    rng = np.random.default_rng(4)
    N = 20
    C_bar = np.zeros((N + 1, N + 1))
    S_bar = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        for m in range(n + 1):
            C_bar[n, m] = rng.uniform(-1, 1) * 1e-4
            S_bar[n, m] = rng.uniform(-1, 1) * 1e-4
    C, S = normalized_to_unnormalized(C_bar, S_bar)
    # Explicit inverse
    for n in range(N + 1):
        for m in range(n + 1):
            factor = normalization_factor(n, m)
            assert C[n, m] == pytest.approx(C_bar[n, m] * factor, rel=1e-14)
            assert S[n, m] == pytest.approx(S_bar[n, m] * factor, rel=1e-14)


def test_degree120_sectoral_normalization_remains_representable():
    """The MRO120F corner factor N_120,120 is tiny but nonzero in float64.

    Computing ``sqrt(exp(log_factorial_ratio))`` underflows before the square
    root; applying the one-half multiplier in log space must preserve the
    representable final value.
    """
    factor = normalization_factor(120, 120)
    assert np.isfinite(factor)
    assert factor > 0.0
    assert 1.0e-234 < factor < 2.0e-233


def test_scaled_cunningham_recurrence_matches_literal_form():
    """W_nm = R^n V_nm through degree 60 where both forms are safe.

    This pins the algebraic rescaling used by the compiled acceleration to
    Cunningham's literal Eqs. 14--17 without comparing an implementation to
    itself at the acceleration level.
    """
    model = mars_gravity_model(max_degree=60)
    x, y, z = 3712.0, -445.0, 713.0
    r2 = x * x + y * y + z * z
    V = _cunningham_V(x, y, z, r2, 61, 61)
    W = _cunningham_W(x, y, z, r2, model.ref_radius_km, 61, 61)
    for n in range(62):
        scale = model.ref_radius_km**n
        for m in range(n + 1):
            expected = scale * V[n, m]
            assert W[n, m] == pytest.approx(expected, rel=1e-11, abs=1e-300)


@pytest.mark.parametrize("degree", [6, 40, 60, 120])
def test_compiled_cunningham_matches_python_reference_through_degree120(degree):
    """The Numba implementation and retained Python oracle agree to roundoff.

    Random low-Mars-orbit directions exercise general tesseral geometry; the
    six axes include both exact poles, where latitude/longitude-gradient
    implementations are prone to singularities.  The two backends translate
    the same cited recurrence but execute independently, making this the
    permanent guard for compilation/type-lowering drift.
    """
    model = mars_gravity_model(max_degree=degree)
    rng = np.random.default_rng(20260812 + degree)
    directions = rng.normal(size=(24, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = np.vstack(
        (
            directions,
            np.eye(3),
            -np.eye(3),
        )
    )
    radii_km = np.linspace(3550.0, 5000.0, len(directions))
    max_relative_error = 0.0
    for direction, radius_km in zip(directions, radii_km):
        position = radius_km * direction
        reference = mars_gravity_acceleration_body_fixed(
            position,
            model,
            degree,
            degree,
            backend=CUNNINGHAM_BACKEND_PYTHON,
        )
        compiled = mars_gravity_acceleration_body_fixed(
            position,
            model,
            degree,
            degree,
            backend=CUNNINGHAM_BACKEND_NUMBA,
        )
        scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        max_relative_error = max(
            max_relative_error,
            float(np.linalg.norm(compiled - reference)) / scale,
        )
    assert max_relative_error < 5.0e-12, max_relative_error


def test_compiled_cunningham_include_central_matches_python_reference():
    model = mars_gravity_model(max_degree=40)
    position = np.array([3712.0, -445.0, 713.0])
    reference = mars_gravity_acceleration_body_fixed(
        position,
        model,
        40,
        40,
        include_central=True,
        backend=CUNNINGHAM_BACKEND_PYTHON,
    )
    compiled = mars_gravity_acceleration_body_fixed(
        position,
        model,
        40,
        40,
        include_central=True,
        backend=CUNNINGHAM_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(compiled, reference, rtol=5.0e-13, atol=0.0)


def test_unknown_cunningham_backend_is_rejected():
    model = mars_gravity_model(max_degree=2)
    with pytest.raises(ValueError, match="gravity_backend"):
        mars_gravity_acceleration_body_fixed(
            np.array([3796.0, 0.0, 0.0]),
            model,
            2,
            backend="not-a-backend",
        )


# ---------------------------------------------------------------------------
# 2. Mars coefficient anchors
# ---------------------------------------------------------------------------


def test_mro120f_degree2_order2_matches_konopliv_2016():
    """MRO120F parsed (C_bar, S_bar) for n=2, m=0..2 match the Mars geoid
    fingerprint from Konopliv 2016 Table 3 (the MRO110B predecessor) to
    within the published 4-sigma uncertainty.

    Values pinned here are the publicly-archived MRO120F coefficients
    themselves (the file is the primary source); the literature tie-in
    is a sanity check that the same Mars-gravity model is being loaded.
    """
    model = mars_gravity_model(max_degree=3)
    # Konopliv 2020 (MRO120F) header values reproduced here to the precision
    # quoted in the PDS archive. If MRO120F is replaced with a successor
    # product and these values shift, the test flags the change rather
    # than drifting.
    assert model.C_bar[2, 0] == pytest.approx(-8.75021982e-04, rel=1e-7)
    assert model.C_bar[2, 1] == pytest.approx(+3.75463732e-10, rel=1e-6)
    assert model.S_bar[2, 1] == pytest.approx(+2.20008609e-11, rel=1e-6)
    # Mars's signature tesseral / sectoral: the Tharsis-bulge C_{2,2},
    # S_{2,2} are ~30x Earth's values relative to J_2.
    assert model.C_bar[2, 2] == pytest.approx(-8.46328358e-05, rel=1e-7)
    assert model.S_bar[2, 2] == pytest.approx(+4.89397590e-05, rel=1e-7)
    # Published unnormalized J_2 (Konopliv 2016 table): 1.95664e-3
    J_n = zonal_coefficients(model, 2)
    assert J_n[2] == pytest.approx(1.95664e-3, rel=5e-4)


def test_c22_equatorial_prime_meridian_matches_closed_form():
    """At body-fixed (R, 0, 0) with only the (2,2) harmonic contributing,
    the x-component of the Cunningham acceleration matches the closed
    form -9 mu C_{2,2} / R^4 times R^2 = -9 mu C_{2,2} / R^2.

    Method: compute a(n=2, m<=2) minus a(n=2, m=0). The difference is
    pure (2,1) + (2,2), but (2,1) values are ~10^-10 and contribute
    negligibly, so the difference is dominated by (2,2). Cross-check
    against the closed form to 1%.
    """
    model = mars_gravity_model(max_degree=2)
    R = model.ref_radius_km
    mu = model.mu_km3_s2
    r_bf = np.array([R, 0.0, 0.0])
    a_full = mars_gravity_acceleration_body_fixed(
        r_bf, model, max_degree=2, max_order=2, include_central=False
    )
    a_zonal = mars_gravity_acceleration_body_fixed(
        r_bf, model, max_degree=2, max_order=0, include_central=False
    )
    a_tess = a_full - a_zonal
    # Closed form for (2,2) a_x at (R, 0, 0) from Cunningham pp. 208 + 214:
    C22 = model.C[2, 2]
    a_x_expected = -9.0 * mu * C22 / R ** 2
    # C_{2,2} < 0 for Mars, so a_x_expected > 0 (pulled AWAY from centre
    # along +x because the prime meridian passes through the Tharsis long
    # axis); matches intuition.
    assert a_x_expected > 0.0
    # 1% tolerance absorbs the small (2,1) contribution (|C_{2,1}| ~ 5e-10
    # vs |C_{2,2}| ~ 5e-5, so (2,1) contribution is ~1e-5 relative).
    assert a_tess[0] == pytest.approx(a_x_expected, rel=1e-4)
    # S_{2,2} picks up the y-component. Closed form at (R, 0, 0):
    #   Re(dV/dy) = -Im(V_{3,3})/2 - k Im(V_{3,1}). At z=0, Im parts are 0.
    #   Im(dV/dy) = Re(V_{3,3})/2 + k Re(V_{3,1})
    #     V_{3,3} = 15 R^3 / R^7 = 15/R^4 (real at (R,0,0))
    #     V_{3,1} = -3/(2 R^4)            (real at (R,0,0))
    #     k = (2-2+1)(2-2+2)/2 = 1
    #   Im(dV/dy) = 15/(2 R^4) - 3/(2 R^4) = 12/(2 R^4) = 6/R^4
    # Acceleration y: mu R^2 * [C_{22} * 0 + S_{22} * 6/R^4]
    S22 = model.S[2, 2]
    a_y_expected = 6.0 * mu * S22 / R ** 2
    assert a_tess[1] == pytest.approx(a_y_expected, rel=1e-4)
    # The z-component at z=0 is NOT zero: it comes from the (2,1) harmonic
    # because C_{2,1} is ~5e-10 (real, small but not zero) in MRO120F.
    # Analytical value: at (R, 0, 0) the (2,1) harmonic contributes
    #     a_z = 3 mu C_{2,1} / R^2
    # via V_{3,1}(R, 0, 0) = -3/(2 R^4) (real, Table I) and the m>0
    # derivative dV_{2,1}/dz = -(n-m+1) V_{3,1} = -2 V_{3,1} = 3/R^4.
    # The (2,2) contribution to a_z is identically zero because
    # V_{3,2}(x, y, 0) = 15 * 0 * (x+iy)^2 / r^7 = 0.
    C21 = model.C[2, 1]
    a_z_expected = 3.0 * mu * C21 / R ** 2
    assert a_tess[2] == pytest.approx(a_z_expected, rel=0.01)


# ---------------------------------------------------------------------------
# 3. Physical properties
# ---------------------------------------------------------------------------


def _cunningham_potential(r_bf: np.ndarray, model, max_degree: int, max_order: int) -> float:
    """Scalar potential U(r) = -mu V(r) where V is the Cunningham potential
    (Eq 3 of Cunningham 1970). Used for finite-difference validation.

    Note the sign: Cunningham V is MINUS the potential energy / mu, so the
    physical potential energy per unit mass is U = -mu V. Acceleration is
    a = mu grad V = -grad U, matching the convention used elsewhere.
    """
    x, y, z = float(r_bf[0]), float(r_bf[1]), float(r_bf[2])
    r2 = x * x + y * y + z * z
    n_store = max_degree + 1  # V grid sized to cover (n+1, m+1) accesses, matches main
    m_store = min(max_order + 1, n_store)
    V = _cunningham_V(x, y, z, r2, n_store, m_store)
    R = model.ref_radius_km
    mu = model.mu_km3_s2
    C = model.C
    S = model.S
    val = 0.0
    for n in range(max_degree + 1):
        Rn = R ** n
        for m in range(min(n, max_order) + 1):
            # Re((C_{nm} - i S_{nm}) V_{nm}) = C * Re(V) + S * Im(V)
            val += Rn * (C[n, m] * V[n, m].real + S[n, m] * V[n, m].imag)
    return -mu * val  # U = -mu V


def _scaled_cunningham_disturbing_potential(
    r_bf: np.ndarray,
    model,
    max_degree: int,
    max_order: int,
) -> float:
    """Disturbing potential from scaled W_nm, excluding n=0 and n=1."""
    x, y, z = (float(value) for value in r_bf)
    r2 = x * x + y * y + z * z
    n_store = max_degree + 1
    m_store = min(max_order + 1, n_store)
    W = _cunningham_W(
        x, y, z, r2, model.ref_radius_km, n_store, m_store
    )
    value = 0.0
    for n in range(2, max_degree + 1):
        for m in range(min(n, max_order) + 1):
            value += (
                model.C[n, m] * W[n, m].real
                + model.S[n, m] * W[n, m].imag
            )
    return -model.mu_km3_s2 * value


def test_finite_difference_of_potential_matches_analytic_acceleration():
    """Central-difference -grad U at a random body-fixed point agrees with
    the analytic acceleration from ``mars_gravity_acceleration_body_fixed``
    to ~1e-6 relative at 1 m step size.

    Uses include_central=True so both sides include the (0,0) term and
    the comparison is on the full acceleration rather than the tiny
    perturbation. Tolerance 1e-6 is the floor for central-difference on
    a smooth field at h = 1e-3 km (Richardson O(h^2) error, not code).
    """
    model = mars_gravity_model(max_degree=4)
    r0 = np.array([3796.19, 200.0, 400.0])
    h = 1e-3  # km step
    max_deg, max_ord = 4, 4

    def grad_U(r):
        g = np.zeros(3)
        for i in range(3):
            rp = r.copy(); rp[i] += h
            rm = r.copy(); rm[i] -= h
            g[i] = (
                _cunningham_potential(rp, model, max_deg, max_ord)
                - _cunningham_potential(rm, model, max_deg, max_ord)
            ) / (2.0 * h)
        return g

    a_fd = -grad_U(r0)  # a = -grad U
    a_an = mars_gravity_acceleration_body_fixed(
        r0, model, max_degree=max_deg, max_order=max_ord, include_central=True
    )
    rel = float(np.linalg.norm(a_fd - a_an) / np.linalg.norm(a_an))
    assert rel < 1e-6, f"grad check rel err {rel:.3e}"


def test_degree120_acceleration_matches_scaled_potential_gradient():
    """Full MRO120F is finite and agrees with an independent derivative.

    A five-point stencil differentiates the scalar disturbing potential at a
    representative low-Mars-orbit position.  This is a permanent guard
    against recurrence scaling, sign, and high-degree normalization errors.
    """
    model = mars_gravity_model(max_degree=120)
    r0 = np.array([3712.0, -445.0, 713.0])
    step_km = 0.03

    gradient = np.zeros(3)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step_km
        fm2 = _scaled_cunningham_disturbing_potential(
            r0 - 2.0 * offset, model, 120, 120
        )
        fm1 = _scaled_cunningham_disturbing_potential(
            r0 - offset, model, 120, 120
        )
        fp1 = _scaled_cunningham_disturbing_potential(
            r0 + offset, model, 120, 120
        )
        fp2 = _scaled_cunningham_disturbing_potential(
            r0 + 2.0 * offset, model, 120, 120
        )
        gradient[axis] = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (
            12.0 * step_km
        )

    finite_difference = -gradient
    analytic = mars_gravity_acceleration_body_fixed(
        r0,
        model,
        max_degree=120,
        max_order=120,
        include_central=False,
    )
    assert np.all(np.isfinite(analytic))
    relative_error = float(
        np.linalg.norm(finite_difference - analytic) / np.linalg.norm(analytic)
    )
    assert relative_error < 2.0e-9, relative_error


def test_pole_safety_of_cunningham_path():
    """Acceleration at the body-fixed pole (0, 0, r) is finite and real --
    the Cunningham formulation does not carry the 1/cos(phi) singularity
    that a direct (r, phi, lambda) gradient would. Critical for
    sun-synchronous retrograde Mars orbits (i ~ 93 deg) that fly directly
    over both poles.

    Zonal-only: x and y components are exactly zero by axisymmetry (the
    V_{n,1} sub-chain vanishes at the pole because the (x+iy) factor in
    V_{1,1} is zero, and the m=0 derivatives reference only V_{n+1, 1}).
    z component is the inward attraction.

    With tesserals on, x and y components are in general NONZERO: the
    potential U_{(n,m>0)} vanishes at the pole but its Cartesian gradient
    does not (e.g. grad(x/r^3) at (0,0,R) = (1/R^3, 0, -3/R^2 * ...)).
    What MUST hold is finiteness and physical magnitude -- the test
    confirms both.
    """
    model = mars_gravity_model(max_degree=10)
    r_pole = np.array([0.0, 0.0, 3796.19])
    a_zonal = mars_gravity_acceleration_body_fixed(
        r_pole, model, max_degree=10, max_order=0, include_central=True
    )
    a_full = mars_gravity_acceleration_body_fixed(
        r_pole, model, max_degree=10, max_order=10, include_central=True
    )
    # No NaN / Inf
    assert np.all(np.isfinite(a_zonal))
    assert np.all(np.isfinite(a_full))
    # Zonal-only: transverse components exactly zero.
    assert abs(a_zonal[0]) < 1e-15
    assert abs(a_zonal[1]) < 1e-15
    # Inward attraction along pole
    assert a_zonal[2] < 0.0
    assert a_full[2] < 0.0
    # Radial magnitude close to -mu/R^2 (central + small J_2 perturbation).
    mu = model.mu_km3_s2
    a_central = mu / 3796.19 ** 2
    assert abs(a_full[2] + a_central) / a_central < 0.01
    # Transverse tesseral contributions are tiny on the scale of the
    # central attraction -- the tesseral field magnitudes fall off with
    # altitude, and at 400 km they're well below 0.1% of central gravity.
    transverse = float(np.hypot(a_full[0], a_full[1]))
    assert transverse < 0.001 * a_central


def test_lz_not_conserved_under_tesserals_complementary_to_zonal_case():
    """Under the full spherical-harmonic potential the body-fixed z-axis
    is no longer a symmetry axis, so L dot p_0 (angular momentum about
    Mars's pole at t=0) must drift on per-orbit timescales. This is the
    complementary positive-signal to
    ``test_axial_angular_momentum_conserved_under_zonal_dynamics`` which
    confirms L_z IS conserved under zonals alone.

    Measure: relative drift of L dot p_0 over 5 LMO orbits. With J_2 only
    the drift is <~5e-10 (pole-motion physics). With degree-order 4
    harmonics the drift should be >= ~1e-6 (orders of magnitude larger).
    """
    from reflectors.elements import mme2000_rotation_from_j2000

    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    # 400 km circular orbit, inc 60 deg, constructed in MME-of-date and
    # rotated to J2000 (same pattern as the zonal tests).
    alt = 400.0
    r0 = R + alt
    v0 = float(np.sqrt(mu / r0))
    inc = np.radians(60.0)
    state_mme = np.array([r0, 0.0, 0.0, 0.0, v0 * np.cos(inc), v0 * np.sin(inc)])
    R_mat = mme2000_rotation_from_j2000(et0)
    state0 = np.concatenate([R_mat.T @ state_mme[:3], R_mat.T @ state_mme[3:]])
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 5
    t_eval = np.linspace(0.0, N * T, N * 40 + 1)
    result = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, gravity_degree=4, gravity_order=4, t_eval_s=t_eval,
        options=PropagationOptions.high_accuracy(),
    )
    r = result.positions()
    v = result.velocities()
    h = np.cross(r, v)
    pole0 = np.asarray(spice.pxform("J2000", "IAU_MARS", et0))[2, :]
    pole0 /= np.linalg.norm(pole0)
    Lz = h @ pole0
    rel = float(np.max(np.abs((Lz - Lz[0]) / Lz[0])))
    # Orders of magnitude above the 5e-10 pole-motion floor of the zonal case.
    assert rel > 1e-6, f"expected |dLz/Lz| > 1e-6 under tesserals; got {rel:.3e}"


# ---------------------------------------------------------------------------
# 4. Propagator plumbing
# ---------------------------------------------------------------------------


def test_gravity_path_and_zonal_path_agree_when_order_zero():
    """``propagate(gravity_degree=N, gravity_order=0)`` integrates the same
    dynamics as ``propagate(zonal_degree=N)``; trajectories must match to
    ~1e-10 at high-accuracy tolerance over 3 orbits. Catches wiring
    regressions.
    """
    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    r0 = R + 400.0
    v0 = float(np.sqrt(mu / r0))
    inc = np.radians(45.0)
    R_mat = mme2000_rotation_from_j2000(et0)
    state_mme = np.array([r0, 0.0, 0.0, 0.0, v0 * np.cos(inc), v0 * np.sin(inc)])
    state0 = np.concatenate([R_mat.T @ state_mme[:3], R_mat.T @ state_mme[3:]])
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    N = 3
    t_eval = np.linspace(0.0, N * T, N * 20 + 1)
    res_zonal = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, zonal_degree=4, t_eval_s=t_eval,
        options=PropagationOptions.high_accuracy(),
    )
    res_harm = propagate(
        state0, (0.0, N * T),
        epoch_et=et0, gravity_degree=4, gravity_order=0, t_eval_s=t_eval,
        options=PropagationOptions.high_accuracy(),
    )
    dr = np.linalg.norm(res_zonal.positions() - res_harm.positions(), axis=1)
    max_dr_km = float(dr.max())
    # 3 orbits at high accuracy; agreement should be << 1 mm.
    assert max_dr_km < 1e-6, f"zonal vs gravity(order=0) drift {max_dr_km:.3e} km"


def test_mutual_exclusion_zonal_and_gravity_kwargs():
    """Setting both ``zonal_degree`` and ``gravity_degree`` is rejected."""
    state0 = np.array([3796.19, 0.0, 0.0, 0.0, 3.36, 0.0])
    et0 = spice.str2et(EPOCH_STR)
    with pytest.raises(ValueError, match="mutually exclusive"):
        propagate(
            state0, (0.0, 10.0),
            epoch_et=et0, zonal_degree=2, gravity_degree=2,
        )


def test_gravity_order_bounds_checked():
    """``gravity_order`` must satisfy ``0 <= order <= degree``."""
    state0 = np.array([3796.19, 0.0, 0.0, 0.0, 3.36, 0.0])
    et0 = spice.str2et(EPOCH_STR)
    with pytest.raises(ValueError, match="gravity_order"):
        propagate(
            state0, (0.0, 10.0),
            epoch_et=et0, gravity_degree=2, gravity_order=5,
        )


def test_full_harmonic_propagation_runs_and_reports_metadata():
    """Smoke: gravity_degree=4, gravity_order=4 integrates a one-orbit
    trajectory without crashing, returns bounded values, and records the
    path + degree/order in ``result.metadata``."""
    model = mars_gravity_model(max_degree=4)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    r0 = R + 500.0
    v0 = float(np.sqrt(mu / r0))
    state0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    result = propagate(
        state0, (0.0, T),
        epoch_et=et0, gravity_degree=4, gravity_order=4,
    )
    final = result.state_km_kmps[-1]
    # Position stays within a physically plausible radial band.
    r_final = float(np.linalg.norm(final[:3]))
    assert R + 300.0 < r_final < R + 700.0
    # Metadata recorded.
    assert result.metadata["path"] == "cunningham_full_harmonics"
    assert result.metadata["gravity_degree"] == 4
    assert result.metadata["gravity_order"] == 4
    assert result.metadata["gravity_model"] == "MRO120F"
    assert result.metadata["gravity_backend"] == CUNNINGHAM_BACKEND_NUMBA
    assert result.metadata["gravity_backend_fastmath"] is False


def test_degree40_compiled_and_python_propagations_agree_over_one_orbit():
    """Backend substitution changes a one-orbit state by far below 1 mm.

    The adaptive integrator can take slightly different internal branches
    after roundoff-level force differences, so this pins physical state
    equivalence rather than requiring bit identity.
    """
    model = mars_gravity_model(max_degree=40)
    mu = model.mu_km3_s2
    radius_km = model.ref_radius_km + 400.0
    speed_km_s = float(np.sqrt(mu / radius_km))
    et0 = spice.str2et(EPOCH_STR)
    state0 = np.array(
        [radius_km, 0.0, 0.0, 0.0, speed_km_s * 0.5, speed_km_s * np.sqrt(0.75)]
    )
    period_s = 2.0 * np.pi * np.sqrt(radius_km**3 / mu)
    t_eval_s = np.linspace(0.0, period_s, 41)
    common = dict(
        epoch_et=et0,
        gravity_degree=40,
        gravity_order=40,
        t_eval_s=t_eval_s,
        options=PropagationOptions.high_accuracy(),
    )
    compiled = propagate(
        state0,
        (0.0, period_s),
        gravity_backend=CUNNINGHAM_BACKEND_NUMBA,
        **common,
    )
    reference = propagate(
        state0,
        (0.0, period_s),
        gravity_backend=CUNNINGHAM_BACKEND_PYTHON,
        **common,
    )
    position_difference_km = np.linalg.norm(
        compiled.positions() - reference.positions(), axis=1
    )
    velocity_difference_km_s = np.linalg.norm(
        compiled.velocities() - reference.velocities(), axis=1
    )
    assert float(np.max(position_difference_km)) < 1.0e-6
    assert float(np.max(velocity_difference_km_s)) < 1.0e-9
    assert reference.metadata["gravity_backend"] == CUNNINGHAM_BACKEND_PYTHON


def test_full_harmonic_propagation_defaults_order_to_degree():
    """If ``gravity_order`` is omitted, it defaults to ``gravity_degree``."""
    model = mars_gravity_model(max_degree=3)
    mu = model.mu_km3_s2
    R = model.ref_radius_km
    et0 = spice.str2et(EPOCH_STR)
    r0 = R + 500.0
    v0 = float(np.sqrt(mu / r0))
    state0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    T = 2 * np.pi * np.sqrt(r0 ** 3 / mu)
    result = propagate(
        state0, (0.0, T),
        epoch_et=et0, gravity_degree=3,
    )
    assert result.metadata["gravity_order"] == 3


# ---------------------------------------------------------------------------
# 5. Magnitude ranking: pin Mars's dominant sectoral cascade at LMO altitude
# ---------------------------------------------------------------------------


def _a_partial(r_bf, model, n, m):
    """Return acceleration truncated at degree ``n`` and order ``m``."""
    if n < 2:
        return np.zeros(3)
    return mars_gravity_acceleration_body_fixed(
        r_bf, model, max_degree=n, max_order=min(m, n)
    )


def _single_harmonic_contribution(r_bf, model, n, m):
    """Triangle double-difference isolating a single (n, m) harmonic."""
    if m == 0:
        return _a_partial(r_bf, model, n, 0) - _a_partial(r_bf, model, n - 1, 0)
    return (
        _a_partial(r_bf, model, n, m)
        - _a_partial(r_bf, model, n, m - 1)
        - _a_partial(r_bf, model, n - 1, m)
        + _a_partial(r_bf, model, n - 1, m - 1)
    )


def test_mars_dominant_harmonic_ranking_at_lmo_equator():
    """Mars's characteristic tesseral/sectoral signature at LMO altitude
    (equator, prime meridian, 400 km alt): J_2 is followed by a
    SECTORAL CASCADE (2,2) > (3,3) > (4,4) > (5,5), not a zonal
    sequence. This is the fingerprint of Mars's non-spherical mass
    distribution and distinguishes it from Earth (where J_3, J_4 are
    competitive with tesserals).

    Expected ratios:
      (2, 2) / J_2 = 0.180 (Tharsis-bulge ellipticity)
      (3, 3) / J_2 = 0.102
      (4, 4) / J_2 = 0.031
      (3, 0) / J_2 = 0.014  (J_3)
      (4, 0) / J_2 = 0.008  (J_4)
    """
    model = mars_gravity_model(max_degree=5)
    R = model.ref_radius_km
    alt = 400.0
    r_bf = np.array([R + alt, 0.0, 0.0])
    a_J2 = _single_harmonic_contribution(r_bf, model, 2, 0)
    mag_J2 = float(np.linalg.norm(a_J2))
    ratios = {
        (n, m): float(
            np.linalg.norm(_single_harmonic_contribution(r_bf, model, n, m))
        )
        / mag_J2
        for (n, m) in [(2, 2), (3, 3), (4, 4), (3, 0), (4, 0)]
    }
    # Sectoral cascade, specific values (Konopliv MRO120F fingerprint).
    assert ratios[(2, 2)] == pytest.approx(0.1795, abs=0.005)
    assert ratios[(3, 3)] == pytest.approx(0.1019, abs=0.005)
    assert ratios[(4, 4)] == pytest.approx(0.0312, abs=0.002)
    assert ratios[(3, 0)] == pytest.approx(0.0144, abs=0.002)
    assert ratios[(4, 0)] == pytest.approx(0.0079, abs=0.001)
    # Ordering: sectorals outrank zonals past J_2 at equator LMO.
    assert ratios[(2, 2)] > ratios[(3, 0)]
    assert ratios[(3, 3)] > ratios[(4, 0)]
    # J_2 itself is the reference.
    assert mag_J2 == pytest.approx(6.98e-6, rel=5e-3)  # km/s^2 at 400 km, equator
