"""Tests for ``reflectors.earth_gravity`` (J2-only zonal Earth model).

The primary check cross-tests two independent implementations of the same
physics: at n=2 the general zonal recurrence
(``zonal_acceleration_inertial``) must agree with the hand-coded Curtis J2
closed form (``j2_closed_form_inertial``) to machine precision in the
IAU_EARTH frame, exercising the ``body_frame`` parameter on both paths.
"""

from __future__ import annotations

import numpy as np

from reflectors.dynamics import body_gm_km3_per_s2
from reflectors.earth_constants import EARTH_J2
from reflectors.earth_gravity import EarthGravityModel, earth_gravity_model
from reflectors.ephemeris import utc_to_et
from reflectors.gravity import j2_closed_form_inertial, zonal_acceleration_inertial
from reflectors.surface import earth_equatorial_radius_km

EPOCH = "2028-01-01T00:00:00"


def test_earth_gravity_model_fields():
    m = earth_gravity_model()
    assert isinstance(m, EarthGravityModel)
    assert m.max_degree == 2
    assert m.J_by_degree == {2: EARTH_J2}
    assert m.mu_km3_s2 == body_gm_km3_per_s2(399)
    assert m.ref_radius_km == earth_equatorial_radius_km()
    assert m.source == "EGM2008-J2"


def test_earth_gravity_model_requires_degree_two():
    import pytest

    with pytest.raises(ValueError):
        earth_gravity_model(max_degree=1)
    with pytest.raises(ValueError):
        earth_gravity_model(max_degree=4)  # only J2 is available


def test_zonal_matches_j2_closed_form_in_iau_earth():
    """n=2 zonal recurrence == Curtis J2 closed form, IAU_EARTH frame, to
    machine precision. Two independent code paths -> a genuine cross-check."""
    m = earth_gravity_model()
    et = utc_to_et(EPOCH)
    # A few off-equatorial LEO positions (J2 is non-trivial off the equator).
    for r in (
        np.array([5000.0, 0.0, 4000.0]),
        np.array([3000.0, -2500.0, 6000.0]),
        np.array([6800.0, 100.0, 50.0]),
    ):
        a_zonal = zonal_acceleration_inertial(
            r, et, m.mu_km3_s2, m.ref_radius_km, m.J_by_degree,
            body_frame="IAU_EARTH",
        )
        a_curtis = j2_closed_form_inertial(
            r, et, m.mu_km3_s2, m.ref_radius_km, EARTH_J2,
            body_frame="IAU_EARTH",
        )
        rel = np.linalg.norm(a_zonal - a_curtis) / np.linalg.norm(a_curtis)
        assert rel < 1e-12, f"zonal vs J2 closed form rel err {rel:.3e} at {r}"


def test_j2_perturbation_is_nonzero_and_inward_at_equator():
    """Positive signal: for an oblate body (J2>0) the J2 perturbation in the
    equatorial plane points INWARD (toward the centre / the bulge), which a
    zeroed or sign-flipped J2 wiring could not produce. Evaluated body-fixed to
    avoid frame mixing (the equator is the IAU_EARTH z=0 plane)."""
    from reflectors.gravity import zonal_acceleration_body_fixed

    m = earth_gravity_model()
    r_bf = np.array([6800.0, 0.0, 0.0])  # on the equator, body-fixed
    a = zonal_acceleration_body_fixed(
        r_bf, m.mu_km3_s2, m.ref_radius_km, m.J_by_degree
    )
    assert np.linalg.norm(a) > 0.0
    assert a[0] < 0.0  # inward (toward centre) in the equatorial plane
