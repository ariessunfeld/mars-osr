"""Fast tests for ``reflectors.sail_designs.make_canonical_sail``."""
from __future__ import annotations

import pytest

from reflectors.sail_designs import make_canonical_sail
from reflectors.srp import SailOptical, SolarSail


class TestMakeCanonicalSailLoading:
    """sigma = m / A round-trips exactly through the factory."""

    @pytest.mark.parametrize("sigma", [0.004, 0.018, 0.05])
    def test_loading_round_trips(self, sigma: float) -> None:
        sail = make_canonical_sail(sigma)
        assert sail.loading_kg_per_m2 == pytest.approx(sigma, abs=0.0)

    def test_default_area_is_1000_m2(self) -> None:
        sail = make_canonical_sail(0.018)
        assert sail.area_m2 == 1000.0

    def test_mass_is_sigma_times_area(self) -> None:
        sail = make_canonical_sail(0.018, area_m2=500.0)
        assert sail.area_m2 == 500.0
        assert sail.mass_kg == pytest.approx(0.018 * 500.0)

    def test_canonical_sigma_50_matches_reference_construction(self) -> None:
        """The canonical factory reproduces the reference construction.

        Pinning this prevents the factory from drifting from
        ``SolarSail(area_m2=1000.0, mass_kg=50.0, ...)``.
        """
        sail = make_canonical_sail(0.05)
        assert sail.area_m2 == 1000.0
        assert sail.mass_kg == 50.0


class TestMakeCanonicalSailOptical:
    """Default optical is JPL square; override threads through."""

    def test_default_optical_is_jpl_square(self) -> None:
        sail = make_canonical_sail(0.018)
        jpl = SailOptical.square_sail_jpl()
        assert sail.optical == jpl

    def test_optical_override_threads_through(self) -> None:
        ideal = SailOptical.ideal()
        sail = make_canonical_sail(0.018, optical=ideal)
        assert sail.optical is ideal


class TestMakeCanonicalSailValidation:
    """sigma is required positional and must be strictly positive."""

    def test_missing_loading_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            make_canonical_sail()  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad_sigma", [0.0, -0.001, -1.0])
    def test_nonpositive_loading_raises(self, bad_sigma: float) -> None:
        with pytest.raises(ValueError, match="loading_kg_per_m2"):
            make_canonical_sail(bad_sigma)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError, match="area_m2"):
            make_canonical_sail(0.018, area_m2=0.0)


class TestMakeCanonicalSailReturnsFrozenSolarSail:
    def test_return_type(self) -> None:
        sail = make_canonical_sail(0.018)
        assert isinstance(sail, SolarSail)

    def test_returned_instance_is_frozen(self) -> None:
        sail = make_canonical_sail(0.018)
        with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
            sail.area_m2 = 2000.0  # type: ignore[misc]
