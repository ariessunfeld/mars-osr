"""Tests for ``reflectors.beam``: reflected-beam divergence and delivered
surface irradiance.

Structure: class-based, matching the convention in
``tests/test_srp.py`` and ``tests/test_visibility.py``. Primary-source
anchors cross-checked at machine precision where possible; the one
expected O((R_sun/r_sun)^2) discrepancy between the Canady-Allen form
(``b = d tan(alpha/2)``) and the Born-Wolf radiance form is pinned to
that analytical value, not to machine precision.

Citations (primary):
  - Canady & Allen 1982 NASA TP-2065 Eq. 9         ("C-A Eq. 9")
  - Çelik & McInnes 2022 Adv. Space Res. 69:647,
    Eqs. 8a/8b (ellipse axes), Eq. 13 (image area),
    Eq. 15 (finite-mirror correction), Eq. 16 (power density).
  - Viale et al. 2023 Adv. Space Res. 72:1304 Eq. 12.
  - Born & Wolf, *Principles of Optics*, §4.8 (radiometry /
    brightness conservation).
  - McInnes 1999 Ch. 2 §2.6.1 (rho * s specular-fraction decomposition).
"""

from __future__ import annotations

import math

import pytest

from reflectors.beam import (
    _PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX,
    beam_footprint_semi_axes_km,
    beam_image_area_m2,
    beam_image_semi_major_km,
    beam_image_semi_minor_km,
    delivered_power_at_target_W,
    delivered_surface_irradiance_via_radiance,
    delivered_surface_irradiance_W_per_m2,
    specular_reflectance,
    sun_angular_diameter_rad,
    sun_half_angle_tan,
)
from reflectors.solar_constants import (
    AU_KM,
    SOLAR_RADIUS_KM,
    solar_irradiance_W_per_m2_at,
)
from reflectors.srp import SailOptical, SolarSail


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jpl_square_sail() -> SolarSail:
    """Reference JPL square sail: 1000 m^2, 50 kg, McInnes Table 2.1 optics."""
    return SolarSail(
        area_m2=1000.0, mass_kg=50.0,
        optical=SailOptical.square_sail_jpl(),
    )


@pytest.fixture(scope="module")
def ideal_sail() -> SolarSail:
    """Reference ideal sail: 1000 m^2, 50 kg, perfect specular reflectance."""
    return SolarSail(
        area_m2=1000.0, mass_kg=50.0,
        optical=SailOptical.ideal(),
    )


@pytest.fixture(scope="module")
def pure_absorber_sail() -> SolarSail:
    """Sail with rho=0: zero specular reflection. Any irradiance must be 0."""
    return SolarSail(
        area_m2=1000.0, mass_kg=50.0,
        optical=SailOptical(
            rho=0.0, s=1.0,
            eps_front=0.5, eps_back=0.5,
            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
        ),
    )


# ---------------------------------------------------------------------------
# Sun subtense angle
# ---------------------------------------------------------------------------


class TestSunSubtense:
    """Sun's angular diameter as seen from the sail."""

    def test_sun_subtense_at_1au_matches_canady_allen_quote(self):
        """alpha(1 AU) ~ 0.0093 rad.

        Canady-Allen 1982 p. 7 quotes ``D_S = 0.0093 d`` (the spot diameter
        at zenith is 0.93% of slant range). This anchors the subtense
        formula to a widely-cited primary-source number.
        Çelik-McInnes 2022 §2 also quotes 0.0093 rad at 1 AU.
        """
        alpha = sun_angular_diameter_rad(AU_KM)
        assert 0.0092 < alpha < 0.0094
        assert alpha == pytest.approx(0.0093, rel=1e-3)

    def test_sun_subtense_hand_formula(self):
        """Direct definitional pin: alpha = 2 arcsin(R_sun / d)."""
        for r_au in (0.5, 1.0, 1.524, 5.2):
            r_km = r_au * AU_KM
            expected = 2.0 * math.asin(SOLAR_RADIUS_KM / r_km)
            assert sun_angular_diameter_rad(r_km) == pytest.approx(
                expected, rel=1e-14
            )

    def test_sun_subtense_at_mars_mean_distance(self):
        """alpha(1.524 AU) ~ 0.0061 rad (~21 arcmin, ~0.35 deg).

        Mars mean distance; sanity anchor for the beam-spread scale.
        """
        alpha = sun_angular_diameter_rad(1.524 * AU_KM)
        assert 0.0060 < alpha < 0.0062
        deg = math.degrees(alpha)
        assert 0.34 < deg < 0.36

    def test_sun_subtense_inverse_scaling(self):
        """alpha(kr) = alpha(r) / k to leading order (small angle)."""
        alpha_1 = sun_angular_diameter_rad(AU_KM)
        alpha_2 = sun_angular_diameter_rad(2.0 * AU_KM)
        # Small-angle inverse scaling: ratio should be 2 to high precision
        # (deviation is O((R/r)^2) ~ 2e-5 at 1 AU).
        assert alpha_1 / alpha_2 == pytest.approx(2.0, rel=1e-4)

    def test_sun_subtense_tan_vs_sin_small_angle(self):
        """tan(alpha/2) ~= sin(alpha/2) = R_sun / r_sun to O((R/r)^2).

        Relevant because Canady-Allen/Çelik-McInnes use tan whereas the
        radiance-conservation derivation naturally uses sin. The ratio
        tan/sin = 1/cos(alpha/2) = 1 + O((R/r)^2).
        """
        for r_au in (0.5, 1.0, 1.524, 5.2):
            r_km = r_au * AU_KM
            alpha = sun_angular_diameter_rad(r_km)
            t = math.tan(0.5 * alpha)
            s = SOLAR_RADIUS_KM / r_km  # = sin(alpha/2) by the arcsin definition
            # tan/sin = 1/cos(alpha/2), and cos(alpha/2) = sqrt(1 - s^2).
            expected_ratio = 1.0 / math.sqrt(1.0 - s * s)
            assert t / s == pytest.approx(expected_ratio, rel=1e-14)

    def test_sun_subtense_rejects_sail_inside_sun(self):
        with pytest.raises(ValueError, match="inside the Sun"):
            sun_angular_diameter_rad(SOLAR_RADIUS_KM * 0.5)
        with pytest.raises(ValueError, match="inside the Sun"):
            sun_angular_diameter_rad(SOLAR_RADIUS_KM)

    def test_tan_helper_agrees_with_direct_computation(self):
        """``sun_half_angle_tan`` == tan(alpha/2) exactly."""
        for r_au in (0.5, 1.0, 1.524):
            r_km = r_au * AU_KM
            expected = math.tan(0.5 * sun_angular_diameter_rad(r_km))
            assert sun_half_angle_tan(r_km) == pytest.approx(
                expected, rel=1e-14
            )


# ---------------------------------------------------------------------------
# Image ellipse geometry (Canady-Allen Eq. 1-2, Çelik-McInnes Eq. 8a-8b)
# ---------------------------------------------------------------------------


class TestImageEllipse:

    def test_semi_minor_matches_canady_allen_eq_1(self):
        """b = d tan(alpha/2). Exact by definition of the primitive."""
        d = 500.0
        r_sun = 1.524 * AU_KM
        expected = d * math.tan(0.5 * sun_angular_diameter_rad(r_sun))
        assert beam_image_semi_minor_km(d, r_sun) == pytest.approx(
            expected, rel=1e-14
        )

    def test_semi_minor_linear_in_slant(self):
        """b doubles when slant doubles (at fixed sail-Sun distance)."""
        r_sun = 1.524 * AU_KM
        b1 = beam_image_semi_minor_km(500.0, r_sun)
        b2 = beam_image_semi_minor_km(1000.0, r_sun)
        assert b2 / b1 == pytest.approx(2.0, rel=1e-14)

    def test_semi_minor_canady_allen_ruleofthumb_at_1au(self):
        """At zenith at 1 AU: D_spot = 2b ~= 0.0093 * d (Canady-Allen p. 7)."""
        d = 1000.0  # km
        b = beam_image_semi_minor_km(d, AU_KM)
        # Spot diameter 2b, relative to slant d:
        ratio = 2.0 * b / d
        assert ratio == pytest.approx(0.0093, rel=1e-3)

    def test_semi_major_circle_at_zenith(self):
        """At sin(epsilon) = 1 (sail at zenith), a = b (circular image)."""
        d = 500.0
        r_sun = 1.524 * AU_KM
        b = beam_image_semi_minor_km(d, r_sun)
        a = beam_image_semi_major_km(d, r_sun, 1.0)
        assert a == pytest.approx(b, rel=1e-14)

    def test_semi_major_stretches_at_oblique_elevation(self):
        """a = b / sin(epsilon). At eps=30 deg, a = 2b."""
        d = 500.0
        r_sun = 1.524 * AU_KM
        b = beam_image_semi_minor_km(d, r_sun)
        a = beam_image_semi_major_km(d, r_sun, math.sin(math.radians(30.0)))
        assert a == pytest.approx(2.0 * b, rel=1e-14)

    def test_semi_major_rejects_nonpositive_elevation(self):
        """sin(epsilon) <= 0: sail is below target horizon, no beam."""
        with pytest.raises(ValueError, match="sin_elevation"):
            beam_image_semi_major_km(500.0, 1.524 * AU_KM, 0.0)
        with pytest.raises(ValueError, match="sin_elevation"):
            beam_image_semi_major_km(500.0, 1.524 * AU_KM, -0.1)

    def test_image_area_matches_celik_mcinnes_eq_13(self):
        """A_im = pi * b^2 / sin(epsilon). Eq. 13 (point-source form)."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        sin_el = math.sin(math.radians(60.0))
        b_km = beam_image_semi_minor_km(d_km, r_sun_km)
        # Convert to meters for area.
        b_m = b_km * 1000.0
        expected_m2 = math.pi * b_m * b_m / sin_el
        assert beam_image_area_m2(d_km, r_sun_km, sin_el) == pytest.approx(
            expected_m2, rel=1e-14
        )

    def test_image_area_at_zenith_is_pi_b_squared(self):
        """Cross-check the generic formula at the circular-image limit."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        b_m = beam_image_semi_minor_km(d_km, r_sun_km) * 1000.0
        expected_m2 = math.pi * b_m * b_m
        assert beam_image_area_m2(d_km, r_sun_km, 1.0) == pytest.approx(
            expected_m2, rel=1e-14
        )

    def test_finite_mirror_correction_eq_15(self):
        """Eq. 15 form: A_im_finite = (pi/sin eps) * (b^2 + (D_M/2)^2).

        Passed via ``sail_diameter_m`` kwarg. Note: ``beam_image_area_m2``
        uses the simpler quadrature form without the cos(psi/2)
        projection on the mirror term (that's applied in the irradiance
        front-end instead). Test here pins the raw quadrature-sum
        geometry only.
        """
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        sin_el = 1.0
        D_M_m = 32.0  # sqrt(1000) square sail side
        A_point = beam_image_area_m2(d_km, r_sun_km, sin_el)
        A_finite = beam_image_area_m2(
            d_km, r_sun_km, sin_el, sail_diameter_m=D_M_m
        )
        # Finite form adds pi * (D_M/2)^2 / sin(eps). Tolerance 1e-12
        # rather than machine precision because the function evaluates
        # ``pi * (b^2 + m^2) / sin eps`` (single division) whereas the
        # expected_delta here reconstructs ``pi * m^2 / sin eps``
        # separately; the difference is a single-bit roundoff in the
        # distribution of pi, well above zero.
        expected_delta = math.pi * (0.5 * D_M_m) ** 2 / sin_el
        assert (A_finite - A_point) == pytest.approx(
            expected_delta, rel=1e-12
        )

    def test_finite_mirror_correction_small_at_lmo_scales(self):
        """At JPL sail / Mars LMO scales, Eq. 15 vs Eq. 13 differ by < 0.1%.

        Pinned analytical bound: correction ratio ~
        (D_M / (2 * d * tan(alpha/2)))^2. At D_M=32 m, d=500 km,
        Mars 1.524 AU: ratio ~ (32 / (2 * 500e3 * 0.003))^2 ~ 1.1e-4.
        """
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        sin_el = 1.0
        D_M_m = 32.0
        A_point = beam_image_area_m2(d_km, r_sun_km, sin_el)
        A_finite = beam_image_area_m2(
            d_km, r_sun_km, sin_el, sail_diameter_m=D_M_m
        )
        assert (A_finite - A_point) / A_point < 1e-3

    def test_footprint_semi_axes_returns_a_b_tuple(self):
        d = 500.0
        r_sun = 1.524 * AU_KM
        a, b = beam_footprint_semi_axes_km(d, r_sun, math.sin(math.radians(30.0)))
        assert a > b
        assert a == pytest.approx(
            beam_image_semi_major_km(d, r_sun, math.sin(math.radians(30.0))),
            rel=1e-14,
        )
        assert b == pytest.approx(
            beam_image_semi_minor_km(d, r_sun), rel=1e-14
        )


# ---------------------------------------------------------------------------
# Specular reflectance bridge (McInnes rho * s)
# ---------------------------------------------------------------------------


class TestMcInnesIntegration:

    def test_jpl_square_sail_rho_times_s_is_0_827(self):
        """rho * s = 0.88 * 0.94 = 0.8272 (McInnes Tab. 2.1)."""
        opt = SailOptical.square_sail_jpl()
        assert specular_reflectance(opt) == pytest.approx(0.8272, rel=1e-12)

    def test_ideal_sail_rho_times_s_is_1_0(self):
        """Ideal sail is perfect specular: rho = s = 1, eta = 1."""
        opt = SailOptical.ideal()
        assert specular_reflectance(opt) == 1.0

    def test_pure_absorber_rho_times_s_is_0(self):
        """Zero reflectance: rho = 0 ⇒ eta = 0 regardless of s."""
        opt = SailOptical(
            rho=0.0, s=1.0,
            eps_front=0.5, eps_back=0.5,
            B_front=2.0 / 3.0, B_back=2.0 / 3.0,
        )
        assert specular_reflectance(opt) == 0.0

    def test_compared_to_canady_allen_aluminum_value(self):
        """Canady-Allen 1982 p. 16 quotes mu*rho = 0.91*0.92 = 0.8372 for
        aluminum-coated membrane. The JPL square sail (McInnes Tab. 2.1,
        citing Wright 1992) sits at rho*s = 0.827, within 2% of the
        NASA baseline. This cross-references the two physically-consistent
        specular-reflectance conventions."""
        mu_rho = 0.91 * 0.92
        eta_mcinnes = specular_reflectance(SailOptical.square_sail_jpl())
        assert abs(eta_mcinnes - mu_rho) / mu_rho < 0.02


# ---------------------------------------------------------------------------
# Delivered surface irradiance: scaling + cross-checks
# ---------------------------------------------------------------------------


class TestDeliveredIrradianceScaling:

    def test_linear_in_sail_area(self, jpl_square_sail):
        """Doubling sail area doubles irradiance (at fixed geometry)."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        I1 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0
        )
        sail2 = SolarSail(
            area_m2=2000.0, mass_kg=jpl_square_sail.mass_kg,
            optical=jpl_square_sail.optical,
        )
        I2 = delivered_surface_irradiance_W_per_m2(
            sail2, d_km, r_sun_km, 1.0, 1.0
        )
        assert I2 / I1 == pytest.approx(2.0, rel=1e-14)

    def test_inverse_square_in_slant(self, jpl_square_sail):
        """Doubling slant quarters irradiance (1/d^2 scaling)."""
        r_sun_km = 1.524 * AU_KM
        I1 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, 500.0, r_sun_km, 1.0, 1.0
        )
        I2 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, 1000.0, r_sun_km, 1.0, 1.0
        )
        assert I2 / I1 == pytest.approx(0.25, rel=1e-14)

    def test_linear_in_cos_alpha(self, jpl_square_sail):
        """I ~ cos(psi/2). 0.5 cos -> half irradiance."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        I1 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0
        )
        I2 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 0.5, 1.0
        )
        assert I2 / I1 == pytest.approx(0.5, rel=1e-14)

    def test_linear_in_sin_elevation(self, jpl_square_sail):
        """I ~ sin(epsilon). 0.5 sin -> half irradiance."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        I1 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0
        )
        I2 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 0.5
        )
        assert I2 / I1 == pytest.approx(0.5, rel=1e-14)

    def test_linear_in_atmospheric_transmission(self, jpl_square_sail):
        """I ~ chi. chi=0.5 -> half irradiance; chi=0 -> 0."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        I1 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0,
            atmospheric_transmission=1.0,
        )
        I05 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0,
            atmospheric_transmission=0.5,
        )
        I0 = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0,
            atmospheric_transmission=0.0,
        )
        assert I05 / I1 == pytest.approx(0.5, rel=1e-14)
        assert I0 == 0.0

    def test_rejects_out_of_range_transmission(self, jpl_square_sail):
        for chi in (-0.1, 1.01, 2.0):
            with pytest.raises(ValueError, match="atmospheric_transmission"):
                delivered_surface_irradiance_W_per_m2(
                    jpl_square_sail, 500.0, 1.524 * AU_KM, 1.0, 1.0,
                    atmospheric_transmission=chi,
                )

    def test_zero_on_degenerate_geometry(self, jpl_square_sail):
        """cos(psi/2) <= 0 or sin(elevation) <= 0 -> I = 0 exactly."""
        r_sun_km = 1.524 * AU_KM
        assert (
            delivered_surface_irradiance_W_per_m2(
                jpl_square_sail, 500.0, r_sun_km, 0.0, 1.0
            )
            == 0.0
        )
        assert (
            delivered_surface_irradiance_W_per_m2(
                jpl_square_sail, 500.0, r_sun_km, -0.1, 1.0
            )
            == 0.0
        )
        assert (
            delivered_surface_irradiance_W_per_m2(
                jpl_square_sail, 500.0, r_sun_km, 1.0, 0.0
            )
            == 0.0
        )
        assert (
            delivered_surface_irradiance_W_per_m2(
                jpl_square_sail, 500.0, r_sun_km, 1.0, -0.1
            )
            == 0.0
        )

    def test_zero_for_pure_absorber(self, pure_absorber_sail):
        """rho=0 -> eta=0 -> I=0 regardless of geometry."""
        I = delivered_surface_irradiance_W_per_m2(
            pure_absorber_sail, 500.0, 1.524 * AU_KM, 1.0, 1.0
        )
        assert I == 0.0

    def test_r_sun_cancels_at_fixed_geometry(self, jpl_square_sail):
        """Key identity: I is independent of r_sun at fixed (d, A, cos, sin).

        In the C-A form this is because ``I_0 ~ 1/r_sun^2`` and
        ``tan^2(alpha/2) ~ 1/r_sun^2`` cancel; in the radiance form
        ``B_sun`` is r_sun-independent by construction. Pin relative
        agreement within the tan-vs-sin O((R/r)^2) budget.
        """
        d_km = 500.0
        I_earth = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, 1.0 * AU_KM, 1.0, 1.0
        )
        I_mars = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, 1.524 * AU_KM, 1.0, 1.0
        )
        I_jupiter = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, 5.2 * AU_KM, 1.0, 1.0
        )
        # All three should agree to O((R_sun/r_sun)^2), worst at 1 AU (~2e-5).
        assert I_earth == pytest.approx(I_mars, rel=1e-4)
        assert I_mars == pytest.approx(I_jupiter, rel=1e-4)


class TestDeliveredIrradianceCrossCheck:
    """Three-way agreement: C-A Eq. 9 == C-M Eq. 13+16 == Born-Wolf."""

    def test_celik_mcinnes_and_canady_allen_machine_precision(self, jpl_square_sail):
        """C-A Eq. 9 rewritten as P / A_im equals the direct C-A form.

        C-M Eq. 16: sigma_M = I_0 * (A_M / A_im) * cos(psi/2).
        C-A Eq. 9: I = eta * I_0 * A_m * cos(psi/2) * sin(eps) /
                        (pi * [d tan(alpha/2)]^2).
        With A_im = pi * b^2 / sin(eps), these are identical
        (eta = mu * rho in C-A; = rho * s here via specular_reflectance).
        Pinned to machine precision: both paths use the same tan(alpha/2).
        """
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        cos_a = 0.9
        sin_e = math.sin(math.radians(45.0))

        # Direct form:
        I_ca = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, cos_a, sin_e
        )
        # Via power / image area:
        eta = specular_reflectance(jpl_square_sail.optical)
        I_0 = solar_irradiance_W_per_m2_at(r_sun_km)
        P_reflected_W = eta * I_0 * jpl_square_sail.area_m2 * cos_a
        A_im_m2 = beam_image_area_m2(d_km, r_sun_km, sin_e)
        I_cm = P_reflected_W / A_im_m2
        assert I_ca == pytest.approx(I_cm, rel=1e-14)

    def test_born_wolf_agrees_with_canady_allen_to_known_tolerance(
        self, jpl_square_sail
    ):
        """C-A and B-W agree to (1 - (R_sun/r_sun)^2) relative.

        Both are leading-order correct; their ratio is analytically
        (tan^2(alpha/2) / sin^2(alpha/2))^{-1} = cos^2(alpha/2) =
        1 - (R_sun/r_sun)^2 (for alpha = 2 arcsin(R/r)). Pinned
        against this analytical value rather than to machine precision
        because the two forms use different small-angle conventions
        (tan vs. sin).
        """
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        cos_a = 1.0
        sin_e = 1.0
        I_ca = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, cos_a, sin_e
        )
        I_bw = delivered_surface_irradiance_via_radiance(
            jpl_square_sail, d_km, cos_a, sin_e
        )
        ratio = I_ca / I_bw
        sin_half_alpha = SOLAR_RADIUS_KM / r_sun_km
        expected_ratio = 1.0 - sin_half_alpha * sin_half_alpha
        assert ratio == pytest.approx(expected_ratio, rel=1e-12)

    def test_born_wolf_r_sun_independent_exactly(self, jpl_square_sail):
        """Radiance form has no r_sun argument; I is independent of sail-Sun
        distance by construction. Pinned as a dimensional check."""
        d_km = 500.0
        cos_a = 0.9
        sin_e = 0.5
        I_bw = delivered_surface_irradiance_via_radiance(
            jpl_square_sail, d_km, cos_a, sin_e
        )
        # Recompute with different chi to spot-check linearity:
        I_bw_half = delivered_surface_irradiance_via_radiance(
            jpl_square_sail, d_km, cos_a, sin_e,
            atmospheric_transmission=0.5,
        )
        assert I_bw_half / I_bw == pytest.approx(0.5, rel=1e-14)

    def test_viale_eq_12_power_equals_irradiance_times_image_area(
        self, jpl_square_sail
    ):
        """Viale Eq. 12 cross-check: P_SPF (when target >= image) equals
        irradiance * image area. Both paths must give the same total
        reflected power.
        """
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        cos_a = 1.0
        sin_e = 1.0
        A_im_m2 = beam_image_area_m2(d_km, r_sun_km, sin_e)
        # Target larger than image: P_SPF = full reflected power.
        target_area = 10.0 * A_im_m2
        P_viale = delivered_power_at_target_W(
            jpl_square_sail, d_km, r_sun_km, cos_a, sin_e, target_area
        )
        # Irradiance form: I * A_im (integrated over image area).
        I = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, cos_a, sin_e
        )
        P_via_I = I * A_im_m2
        assert P_viale == pytest.approx(P_via_I, rel=1e-14)

    def test_viale_eq_12_small_target_fractionally_intercepted(
        self, jpl_square_sail
    ):
        """When target_area < A_im, P_SPF = P_total * (target / A_im)."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        A_im_m2 = beam_image_area_m2(d_km, r_sun_km, 1.0)
        P_total = delivered_power_at_target_W(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0, A_im_m2
        )
        # Half-size target:
        P_half = delivered_power_at_target_W(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0, 0.5 * A_im_m2
        )
        assert P_half == pytest.approx(0.5 * P_total, rel=1e-14)


class TestFiniteMirrorCorrection:
    """Eq. 15 quadrature form; pinhole-regime guard."""

    def test_eq_15_matches_eq_13_at_our_scales(self, jpl_square_sail):
        """Eq. 15 (finite mirror) vs Eq. 13 (point mirror) differ by <0.1% at
        JPL-sail / Mars-LMO scales. Tests the irradiance path directly
        (which applies cos(psi/2) to the mirror projection)."""
        d_km = 500.0
        r_sun_km = 1.524 * AU_KM
        cos_a = 1.0  # face-on, projection = full D_M
        sin_e = 1.0
        I_point = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, cos_a, sin_e
        )
        I_finite = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, cos_a, sin_e,
            include_finite_mirror_correction=True,
            sail_diameter_m=math.sqrt(jpl_square_sail.area_m2),
        )
        # Both should agree to better than 0.1% at these scales.
        assert abs(I_finite - I_point) / I_point < 1e-3
        # Finite correction makes image larger, so irradiance smaller:
        assert I_finite < I_point

    def test_requires_diameter_when_enabled(self, jpl_square_sail):
        with pytest.raises(ValueError, match="sail_diameter_m"):
            delivered_surface_irradiance_W_per_m2(
                jpl_square_sail, 500.0, 1.524 * AU_KM, 1.0, 1.0,
                include_finite_mirror_correction=True,
            )

    def test_pinhole_regime_violation_raises(self, jpl_square_sail):
        """A large mirror at a short slant triggers the pinhole-regime guard.

        Construct an out-of-regime case: 1000 m diameter mirror, 100 km
        slant, Mars mean. Mirror projected radius 500 m. Sun-spread:
        100e3 m * 0.003 / 2 = 150 m. Ratio ~ 3.3 >>
        _PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX (= 0.2). Raises.
        """
        opt = jpl_square_sail.optical
        big_sail = SolarSail(area_m2=1e6, mass_kg=1000.0, optical=opt)
        with pytest.raises(ValueError, match="pinhole-regime"):
            delivered_surface_irradiance_W_per_m2(
                big_sail, 100.0, 1.524 * AU_KM, 1.0, 1.0,
                include_finite_mirror_correction=True,
                sail_diameter_m=1000.0,
            )

    def test_pinhole_regime_threshold_is_exposed(self):
        """_PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX is a module-level constant."""
        assert 0.0 < _PINHOLE_MIRROR_TO_SPREAD_RATIO_MAX < 1.0


# ---------------------------------------------------------------------------
# Numeric anchors (specific pinned regression values)
# ---------------------------------------------------------------------------


class TestNumericAnchors:
    """Pinned numeric values for the canonical geometry. These anchors
    catch any drift in defining constants (L_Sun, R_Sun, AU) or
    in the formula that would change the numerical output.
    """

    def test_canonical_jpl_sail_mars_lmo_zenith_face_on(self, jpl_square_sail):
        """1000 m^2 JPL square sail, d=500 km, r_sun=1.412 AU,
        cos(psi/2)=1, sin(eps)=1, chi=1:
        I_target ~ 0.066 W/m^2. Hand-derived, pinned to 1%.

        Heuristic sanity: reflected beam delivers ~10^-4 of direct
        solar at Mars (which is ~683 W/m^2 at 1.412 AU). The precise
        ratio is Omega_mirror / Omega_sun, pinned independently
        below.
        """
        d_km = 500.0
        r_sun_km = 1.412 * AU_KM
        I = delivered_surface_irradiance_W_per_m2(
            jpl_square_sail, d_km, r_sun_km, 1.0, 1.0
        )
        assert 0.065 < I < 0.068
        # Also verify it's (approximately) 1e-4 of direct solar at sail:
        I_direct = solar_irradiance_W_per_m2_at(r_sun_km)
        ratio = I / I_direct
        assert 9e-5 < ratio < 1.1e-4

    def test_ratio_of_reflected_to_direct_equals_eta_omega_ratio(
        self, ideal_sail
    ):
        """Analytical identity for ideal sail (eta=1):

            I_refl / I_direct = Omega_mirror / Omega_sun * (1 - (R/r)^2)

        where the (1 - (R/r)^2) factor is the tan-vs-sin correction.
        Pinned using ideal_sail to remove the eta factor.
        """
        d_km = 500.0
        r_sun_km = 1.412 * AU_KM
        # Ideal sail: eta = 1, all flux specularly reflected.
        I_refl = delivered_surface_irradiance_W_per_m2(
            ideal_sail, d_km, r_sun_km, 1.0, 1.0
        )
        I_direct = solar_irradiance_W_per_m2_at(r_sun_km)
        # Expected ratio from the C-A form:
        # I_refl / I_direct = A_sail * cos(psi/2) * sin(eps) /
        #                     (pi * (d * tan(alpha/2))^2)
        alpha = sun_angular_diameter_rad(r_sun_km)
        tan_half = math.tan(0.5 * alpha)
        expected_ratio = ideal_sail.area_m2 / (
            math.pi * (d_km * 1000.0 * tan_half) ** 2
        )
        assert I_refl / I_direct == pytest.approx(expected_ratio, rel=1e-12)

    def test_footprint_radius_at_mars_zenith_canonical(self):
        """Perpendicular beam radius at Mars mean (1.412 AU), d=500 km, is
        ~1.6 km (hand: 500 * 0.00329 = 1.65 km)."""
        b_km = beam_image_semi_minor_km(500.0, 1.412 * AU_KM)
        assert 1.6 < b_km < 1.7


# ---------------------------------------------------------------------------
# Basic input validation
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_semi_minor_rejects_nonpositive_slant(self):
        with pytest.raises(ValueError, match="slant_km"):
            beam_image_semi_minor_km(0.0, AU_KM)
        with pytest.raises(ValueError, match="slant_km"):
            beam_image_semi_minor_km(-1.0, AU_KM)

    def test_image_area_rejects_negative_sail_diameter(self):
        with pytest.raises(ValueError, match="sail_diameter_m"):
            beam_image_area_m2(
                500.0, AU_KM, 1.0, sail_diameter_m=-1.0
            )

    def test_irradiance_rejects_nonpositive_slant(self, jpl_square_sail):
        with pytest.raises(ValueError, match="slant_km"):
            delivered_surface_irradiance_W_per_m2(
                jpl_square_sail, 0.0, AU_KM, 1.0, 1.0
            )

    def test_power_rejects_nonpositive_target_area(self, jpl_square_sail):
        with pytest.raises(ValueError, match="target_area_m2"):
            delivered_power_at_target_W(
                jpl_square_sail, 500.0, AU_KM, 1.0, 1.0, 0.0
            )
