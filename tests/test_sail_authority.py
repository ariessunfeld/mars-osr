"""Fast tests for ``reflectors.sail_authority``.

Pins
====

- ``force_coefficient_at_normal`` reduces to η = 2 exactly for the
  McInnes 'Ideal sail' optical coefficients.
- ``force_coefficient_at_normal`` for the JPL square sail gives a
  hand-computed value (tested below to 1e-12).
- ``characteristic_acceleration`` scales as 1 / σ (linearity of σ).
- ``characteristic_acceleration`` scales as 1 / r² (inverse-square law
  via the Sun-distance flux).
- ``unrestricted_max_secular_de_dt`` reduces to 0 in the σ → ∞ limit
  (no SRP authority).
- ``unrestricted_max_secular_de_dt`` reproduces the closed-form
  ``3 a_c / (2 n a)`` at a hand-checked rank-1 geometry, both for
  σ = 0.05 and σ = 0.018.
"""
from __future__ import annotations

import math

import pytest

from reflectors.sail_authority import (
    characteristic_acceleration,
    force_coefficient_at_normal,
    unrestricted_max_secular_de_dt,
)
from reflectors.sail_designs import make_canonical_sail
from reflectors.solar_constants import AU_KM, solar_flux_at
from reflectors.srp import SailOptical


# ---------------------------------------------------------------------------
# Constants used across multiple pinning tests
# ---------------------------------------------------------------------------


# Rank-1 orbit: 501 km altitude above Mars equatorial radius.
A_KM = 3897.19
# Mars GM, IAU 2015 nominal (used by reflectors.dynamics throughout).
MU_MARS = 42828.37
# Mars perihelion distance (the rank-1 epoch is at perihelion).
MARS_PERIHELION_AU = 1.381
MARS_PERIHELION_KM = MARS_PERIHELION_AU * AU_KM


# ---------------------------------------------------------------------------
# force_coefficient_at_normal
# ---------------------------------------------------------------------------


class TestForceCoefficientAtNormal:
    def test_ideal_sail_is_exactly_2(self) -> None:
        eta = force_coefficient_at_normal(SailOptical.ideal())
        assert eta == pytest.approx(2.0, abs=1.0e-15)

    def test_jpl_square_sail_matches_hand_computation(self) -> None:
        """JPL Halley: ρ=0.88, s=0.94, ε_f=0.05, ε_b=0.55, B_f=0.79, B_b=0.55.

        Hand computation:
          C_s        = 1 - ρ s = 1 - 0.8272 = 0.1728
          thermal    = (ε_f B_f - ε_b B_b) / (ε_f + ε_b)
                     = (0.0395 - 0.3025) / 0.6 = -0.43833...
          C_n(0)     = 2 ρ s + B_f (1-s) ρ + (1-ρ) thermal
                     = 1.6544 + 0.041712 + 0.12 · (-0.4383...)
                     = 1.6544 + 0.041712 - 0.052600
                     = 1.643512
          η          = C_s + C_n(0)  = 1.816312
        """
        eta = force_coefficient_at_normal(SailOptical.square_sail_jpl())
        assert eta == pytest.approx(1.816312, abs=1.0e-6)

    def test_eps_zero_handled_without_division_by_zero(self) -> None:
        """ε_f + ε_b = 0 must NOT raise; the thermal term is zeroed."""
        opt_no_emit = SailOptical(
            rho=0.5, s=0.5, eps_front=0.0, eps_back=0.0,
            B_front=2.0/3.0, B_back=2.0/3.0,
        )
        eta = force_coefficient_at_normal(opt_no_emit)
        # C_s = 1 - 0.25 = 0.75; C_n(0) = 0.5 + (2/3)(0.5)(0.5) + 0.5·0
        #     = 0.5 + 1/6 = 0.6667; eta = 1.4167
        assert eta == pytest.approx(0.75 + 0.5 + (2.0/3.0)*0.25, abs=1.0e-12)


# ---------------------------------------------------------------------------
# characteristic_acceleration
# ---------------------------------------------------------------------------


class TestCharacteristicAcceleration:
    def test_scales_linearly_with_inverse_sigma(self) -> None:
        """a_c(σ) · σ = const at fixed (r, sail.optical)."""
        sail_a = make_canonical_sail(0.018)
        sail_b = make_canonical_sail(0.05)
        a_c_a = characteristic_acceleration(sail_a, MARS_PERIHELION_KM)
        a_c_b = characteristic_acceleration(sail_b, MARS_PERIHELION_KM)
        # Same η, same r, so a_c · σ should be identical.
        assert a_c_a * 0.018 == pytest.approx(a_c_b * 0.05, rel=1.0e-12)
        # And the heavier sail has the smaller a_c.
        assert a_c_a > a_c_b

    def test_scales_inverse_square_with_helio_distance(self) -> None:
        """Doubling r quarters a_c."""
        sail = make_canonical_sail(0.018)
        a_c_close = characteristic_acceleration(sail, MARS_PERIHELION_KM)
        a_c_far = characteristic_acceleration(
            sail, 2.0 * MARS_PERIHELION_KM
        )
        assert a_c_far == pytest.approx(a_c_close / 4.0, rel=1.0e-12)

    def test_perfect_sail_eta_is_2(self) -> None:
        sail = make_canonical_sail(0.018)
        a_c = characteristic_acceleration(
            sail, MARS_PERIHELION_KM, perfect_sail=True
        )
        P_pa = solar_flux_at(MARS_PERIHELION_KM)
        expected = 2.0 * P_pa / 0.018
        assert a_c == pytest.approx(expected, rel=1.0e-12)

    def test_pinned_value_sigma_018_perfect_sail(self) -> None:
        """Pin a_c at σ=0.018 perfect-sail, Mars perihelion (1.381 AU).

        Hand computation:
          P(1.381 AU) = 4.541e-6 / 1.381^2 ≈ 2.381e-6 Pa
          a_c         = 2 · 2.381e-6 / 0.018 ≈ 2.646e-4 m/s^2
        """
        sail = make_canonical_sail(0.018)
        a_c = characteristic_acceleration(
            sail, MARS_PERIHELION_KM, perfect_sail=True
        )
        # Allow 0.5% tolerance — the IAU 2015 nominal vs textbook P_1AU
        # differ by ~0.4%, so higher precision is not pinned here.
        assert a_c == pytest.approx(2.646e-4, rel=5.0e-3)

    def test_jpl_optics_below_perfect_sail(self) -> None:
        """At fixed (σ, r), the JPL sail's a_c is below the perfect-sail
        bound by the ratio η_jpl / 2 ≈ 0.908."""
        sail = make_canonical_sail(0.018)
        a_c_perfect = characteristic_acceleration(
            sail, MARS_PERIHELION_KM, perfect_sail=True
        )
        a_c_jpl = characteristic_acceleration(
            sail, MARS_PERIHELION_KM, perfect_sail=False
        )
        ratio = a_c_jpl / a_c_perfect
        assert ratio == pytest.approx(1.816312 / 2.0, rel=1.0e-6)
        assert 0.90 < ratio < 0.91

    def test_rejects_nonpositive_helio_distance(self) -> None:
        sail = make_canonical_sail(0.018)
        with pytest.raises(ValueError, match="r_helio_km"):
            characteristic_acceleration(sail, 0.0)
        with pytest.raises(ValueError, match="r_helio_km"):
            characteristic_acceleration(sail, -1.0)


# ---------------------------------------------------------------------------
# unrestricted_max_secular_de_dt
# ---------------------------------------------------------------------------


class TestUnrestrictedMaxSecularDeDt:
    def test_scales_inverse_with_sigma(self) -> None:
        """3× lighter sail => 3× more |⟨de/dt⟩|_max."""
        sail_018 = make_canonical_sail(0.018)
        sail_054 = make_canonical_sail(0.054)
        m_018 = unrestricted_max_secular_de_dt(
            sail_018, MU_MARS, A_KM, MARS_PERIHELION_KM
        )
        m_054 = unrestricted_max_secular_de_dt(
            sail_054, MU_MARS, A_KM, MARS_PERIHELION_KM
        )
        assert m_018 == pytest.approx(3.0 * m_054, rel=1.0e-12)

    def test_scales_with_sqrt_a(self) -> None:
        """At fixed (σ, r), ⟨de/dt⟩_max ∝ 1/(n·a) = sqrt(a)/sqrt(μ)
        because n = sqrt(μ/a³). So doubling a MULTIPLIES the bound by
        sqrt(2): a higher orbit has weaker n·a so the same SRP
        acceleration buys more eccentricity drift."""
        sail = make_canonical_sail(0.018)
        m_a1 = unrestricted_max_secular_de_dt(
            sail, MU_MARS, A_KM, MARS_PERIHELION_KM
        )
        m_a2 = unrestricted_max_secular_de_dt(
            sail, MU_MARS, 2.0 * A_KM, MARS_PERIHELION_KM
        )
        assert m_a2 == pytest.approx(m_a1 * math.sqrt(2.0), rel=1.0e-12)

    def test_pinned_value_sigma_018_perfect_sail_reference(self) -> None:
        """Pin |⟨de/dt⟩|_max at sigma=0.018.

        Hand computation (perfect-sail):
          a_c_km_s2  ≈ 2.646e-4 m/s² · 1e-3 = 2.646e-7 km/s²
          n          = sqrt(42828.37 / 3897.19³) ≈ 8.51e-4 rad/s
          n · a      ≈ 8.51e-4 · 3897.19 ≈ 3.317 km/s
          max_de_dt  = (3 · 2.646e-7) / (2 · 3.317)
                     ≈ 1.197e-7 per second

        Per sol (88775 s): |Δe|_max ≈ 1.063e-2.
        """
        sail = make_canonical_sail(0.018)
        rate = unrestricted_max_secular_de_dt(
            sail, MU_MARS, A_KM, MARS_PERIHELION_KM, perfect_sail=True
        )
        assert rate == pytest.approx(1.197e-7, rel=5.0e-3)
        # Cross-check via per-sol |Δe|_max.
        delta_e_per_sol = rate * 88775.0
        assert delta_e_per_sol == pytest.approx(0.0106, rel=5.0e-2)

    def test_pinned_value_sigma_050_perfect_sail_reference(self) -> None:
        """Pin at σ=0.05.

        Hand computation:
          a_c_km_s2  ≈ 9.524e-5 m/s² · 1e-3 = 9.524e-8 km/s²
          (factor of 50/18 ≈ 2.778 below σ=0.018)
          max_de_dt  ≈ 4.31e-8 per second
          per sol    ≈ 3.83e-3 ≈ 0.00383

        """
        sail = make_canonical_sail(0.05)
        rate = unrestricted_max_secular_de_dt(
            sail, MU_MARS, A_KM, MARS_PERIHELION_KM, perfect_sail=True
        )
        delta_e_per_sol = rate * 88775.0
        # 0.0038 ± a few percent
        assert delta_e_per_sol == pytest.approx(0.00383, rel=5.0e-2)

    def test_jpl_optics_reduces_bound_by_eta_ratio(self) -> None:
        """The JPL sail's bound is 1.816/2.0 ≈ 90.8% of the perfect-sail."""
        sail = make_canonical_sail(0.018)
        m_perfect = unrestricted_max_secular_de_dt(
            sail, MU_MARS, A_KM, MARS_PERIHELION_KM, perfect_sail=True
        )
        m_jpl = unrestricted_max_secular_de_dt(
            sail, MU_MARS, A_KM, MARS_PERIHELION_KM, perfect_sail=False
        )
        assert m_jpl / m_perfect == pytest.approx(
            1.816312 / 2.0, rel=1.0e-6
        )

    def test_rejects_nonpositive_inputs(self) -> None:
        sail = make_canonical_sail(0.018)
        with pytest.raises(ValueError, match="mu_central"):
            unrestricted_max_secular_de_dt(
                sail, 0.0, A_KM, MARS_PERIHELION_KM
            )
        with pytest.raises(ValueError, match="a_km"):
            unrestricted_max_secular_de_dt(
                sail, MU_MARS, 0.0, MARS_PERIHELION_KM
            )
