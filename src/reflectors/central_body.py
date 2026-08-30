"""Central-body configuration for the (body-generic) SRP escape propagator.

This module collects the central-body assumptions used by
``escape.propagate_escape`` in a frozen ``CentralBody`` object, allowing the
same propagator to model escape from Mars or Earth. The Mars factory reads the
same live kernel and constant values as the default configuration, so passing
``central_body=mars_central_body()`` or accepting ``None`` produces the same
Mars dynamics.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from reflectors.dynamics import body_gm_km3_per_s2, mars_gm_km3_per_s2


@dataclass(frozen=True)
class CentralBody:
    """Everything the escape propagator needs to know about its central body.

    Attributes
    ----------
    naif_id
        NAIF integer id (499 Mars, 399 Earth). The observer for every
        ``spkezr`` (Sun + third bodies) and the umbra geometry.
    mu_km3_s2
        Point-mass GM used when ``gravity_degree == 0`` (and as a cross-check
        against the gravity model's mu). Read live from the kernel pool.
    body_frame
        Body-fixed frame name for the zonal gravity rotation (``"IAU_MARS"`` /
        ``"IAU_EARTH"``).
    equatorial_radius_km
        Equatorial radius (km) -- the altitude-floor reference and the umbra
        occulting-disc radius. Read live from the PCK.
    hill_radius_km
        Default Hill-sphere termination radius (the escape handoff boundary).
    gravity_model_factory
        ``factory(max_degree) -> gravity model`` (``mars_gravity_model`` /
        ``earth_gravity_model``). The returned model carries ``.mu_km3_s2`` and
        ``.ref_radius_km``; the escape RHS dispatches on its type
        (``MarsGravityModel`` -> Cunningham, ``EarthGravityModel`` -> zonal).
    label
        Human-readable name (logs, run metadata).
    occults_sun
        Whether this central body can cast a solar-eclipse umbra on the sail
        (the escape RHS gates ``shadow_factor`` on it). ``True`` for an orbited
        planet (Mars, Earth). ``False`` for
        the SUN as central body (interplanetary cruise): with the Sun at the
        frame origin the umbra geometry is degenerate (occulting disc == light-
        source disc at D=0 -> spurious permanent "eclipse"), and physically a
        sail at ~1 AU in interplanetary space is never occulted by the modelled
        bodies, so the umbra is bypassed (always sunlit).
    """

    naif_id: int
    mu_km3_s2: float
    body_frame: str
    equatorial_radius_km: float
    hill_radius_km: float
    gravity_model_factory: Callable[..., Any]
    label: str
    occults_sun: bool = True


def mars_central_body() -> CentralBody:
    """Mars central body using the propagator's default Mars values.

    Reads the same live kernel and constant sources used by the default Mars
    configuration.
    """
    from reflectors.gravity import mars_gravity_model
    from reflectors.mars_constants import MARS_HILL_RADIUS_KM
    from reflectors.surface import mars_equatorial_radius_km

    return CentralBody(
        naif_id=499,
        mu_km3_s2=mars_gm_km3_per_s2(),
        body_frame="IAU_MARS",
        equatorial_radius_km=mars_equatorial_radius_km(),
        hill_radius_km=MARS_HILL_RADIUS_KM,
        gravity_model_factory=mars_gravity_model,
        label="MARS",
    )


def earth_central_body() -> CentralBody:
    """Earth central body for the escape model.

    Earth GM and equatorial radius are read live (BODY399_GM, BODY399_RADII);
    the Hill radius is the literature-pinned ``EARTH_HILL_RADIUS_KM``. Gravity is
    the J2-only zonal ``earth_gravity_model`` in the IAU_EARTH frame.
    """
    from reflectors.earth_constants import EARTH_HILL_RADIUS_KM
    from reflectors.earth_gravity import earth_gravity_model
    from reflectors.surface import earth_equatorial_radius_km

    return CentralBody(
        naif_id=399,
        mu_km3_s2=body_gm_km3_per_s2(399),
        body_frame="IAU_EARTH",
        equatorial_radius_km=earth_equatorial_radius_km(),
        hill_radius_km=EARTH_HILL_RADIUS_KM,
        gravity_model_factory=earth_gravity_model,
        label="EARTH",
    )


# Nominal large heliocentric ceiling for the Sun-as-central cruise. NOT a
# physical termination boundary: the interplanetary cruise always overrides
# ``radius_ceiling`` explicitly (the transfer never escapes the Sun). This
# value only needs to exceed every heliocentric distance the cruise visits
# (Mars aphelion ~2.49e8 km); 1e12 km (~6685 AU) does so by ~4 orders.
SUN_NOMINAL_CEILING_KM = 1.0e12


def _sun_gravity_model_stub(*args, **kwargs):
    """Sentinel gravity factory for the Sun central body.

    The interplanetary cruise runs ``gravity_degree=0`` (point-mass solar
    gravity from ``mu_km3_s2``), so the propagator never builds a gravity
    model for the Sun. Calling this function indicates a modeling error because
    a Sun-centred run must use ``gravity_degree=0`` (no solar zonal-harmonic
    field is wired).
    """
    raise NotImplementedError(
        "sun_central_body has no gravity model: the interplanetary cruise must "
        "run gravity_degree=0 (point-mass solar gravity)."
    )


def sun_central_body() -> CentralBody:
    """Sun central body for interplanetary cruise.

    Point-mass solar gravity only (``gravity_degree=0``); GM read live from the
    kernel pool (BODY10_GM via ``sun_gm_km3_per_s2``), radius from BODY10_RADII
    (``shadow.sun_radius_km`` ~ 695700 km, IAU 2015 nominal). ``occults_sun`` is
    ``False`` -- the Sun cannot eclipse the sail it illuminates, so the umbra is
    bypassed (a 1-AU interplanetary sail is always sunlit). The Hill radius is a
    nominal large ceiling (see ``SUN_NOMINAL_CEILING_KM``); the cruise overrides
    ``radius_ceiling`` so it is never a physical boundary.
    """
    from reflectors.dynamics import sun_gm_km3_per_s2
    from reflectors.shadow import sun_radius_km

    return CentralBody(
        naif_id=10,
        mu_km3_s2=sun_gm_km3_per_s2(),
        body_frame="IAU_SUN",  # unused at gravity_degree=0
        equatorial_radius_km=sun_radius_km(),
        hill_radius_km=SUN_NOMINAL_CEILING_KM,
        gravity_model_factory=_sun_gravity_model_stub,
        label="SUN",
        occults_sun=False,
    )
