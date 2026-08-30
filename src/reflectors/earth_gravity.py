"""Earth zonal gravity model (J2-only by default) for Earth escape.

The Earth-escape model needs only the dominant J2 oblateness
term (the higher zonals and the full tesseral field are negligible against SRP
+ drag + the lunar third body for a spiral that climbs away from low orbit).
Rather than duplicate the MRO120F download/parse machinery of
``MarsGravityModel``, this model carries just ``{n: J_n}`` and reuses the
body-agnostic zonal recurrence ``gravity.zonal_acceleration_inertial`` with
``body_frame="IAU_EARTH"``.

J2 (and any higher zonals) are pinned in ``earth_constants.py`` with citations;
mu and the reference radius are read from the live kernel pool so they cannot
drift from the dynamics used elsewhere.

Sign convention: ``J_by_degree`` holds UNNORMALIZED J_n = -C_{n,0}, matching the
Mars zonal path (``gravity.zonal_coefficients``). Earth's J2 is POSITIVE
(oblate, equatorial bulge), exactly as Mars's is -- so the same recurrence and
sign handling apply unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from reflectors.dynamics import body_gm_km3_per_s2
from reflectors.earth_constants import EARTH_J2
from reflectors.surface import earth_equatorial_radius_km

EARTH_NAIF_ID = 399


@dataclass(frozen=True)
class EarthGravityModel:
    """Zonal Earth gravity model (interface-compatible with the escape RHS).

    Attributes
    ----------
    max_degree
        Highest zonal degree carried (2 = J2-only, the default).
    ref_radius_km
        Reference radius the J_n are defined against (Earth equatorial radius,
        read live from the PCK; EGM2008's 6378.1363 km matches to ~0.3 m).
    mu_km3_s2
        Earth GM (read live from BODY399_GM). For Earth there is NO lumped-moon
        decoupling to undo (unlike Mars/MRO120F): the Moon is a separate third
        body (NAIF 301), so this is already the correct Earth-alone central mu.
    J_by_degree
        Mapping n -> J_n (unnormalized), consumed directly by
        ``gravity.zonal_acceleration_inertial``.
    source
        Provenance string (recorded in run metadata, mirrors
        ``MarsGravityModel.source``).
    """

    max_degree: int
    ref_radius_km: float
    mu_km3_s2: float
    J_by_degree: dict[int, float]
    source: str = "EGM2008-J2"


@lru_cache(maxsize=4)
def earth_gravity_model(max_degree: int = 2) -> EarthGravityModel:
    """Build (and cache) the Earth zonal gravity model.

    This model is J2-only. Earth has a dipole-free field in its centre-of-mass
    frame, so degree 2 is the leading term and the only supported degree.
    """
    if max_degree < 2:
        raise ValueError("max_degree must be >= 2 (J2 is the leading Earth zonal)")
    if max_degree > 2:
        raise ValueError(
            f"earth_gravity_model carries J2 only; requested "
            f"max_degree={max_degree}. Add higher J_n to earth_constants + the "
            "J_by_degree dict below to extend."
        )
    return EarthGravityModel(
        max_degree=max_degree,
        ref_radius_km=earth_equatorial_radius_km(),
        mu_km3_s2=body_gm_km3_per_s2(EARTH_NAIF_ID),
        J_by_degree={2: EARTH_J2},
    )
