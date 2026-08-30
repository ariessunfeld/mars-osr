"""Gauss's variational equations and locally-optimal element-rate maxima.

This module supplies the orbital-mechanics layer the Q-law escape steering
(:mod:`reflectors.qlaw`) is built on:

  1. ``osculating_elements`` -- a fast, SPICE-free extraction of the classical
     osculating elements ``(a, e, i, RAAN, argp, nu)`` from a Cartesian state.
     It returns the same :class:`reflectors.elements.ClassicalElements` frozen
     dataclass that ``elements.classical_elements`` builds, but computes the
     fields with numpy instead of ``spice.oscltx`` so it is cheap enough for
     the propagator RHS hot loop. ``elements.classical_elements`` is kept as
     the SPICE-backed reporting path and as the independent cross-check oracle
     in the test suite.

  2. ``rtn_basis`` -- the radial / transverse / normal orthonormal frame at a
     state, in which thrust accelerations are decomposed for Gauss's equations.

  3. ``gauss_variational_rates`` -- Gauss's form of the variational equations:
     the secular-plus-periodic element rates ``(da/dt ... dnu/dt)`` produced by
     a thrust acceleration with radial / transverse / normal components
     ``(f_r, f_theta, f_h)``.

  4. ``semimajor_axis_rate_max`` / ``eccentricity_rate_max`` -- the
     locally-optimal element-rate maxima ``a_dot_xx`` and ``e_dot_xx``: the
     largest ``|da/dt|`` / ``|de/dt|`` achievable at a given orbit by *any*
     thrust direction at the *most effective* true anomaly, for a fixed thrust
     acceleration magnitude ``f``. These normalise the Q-law proximity
     quotient.

References (primary)
--------------------
Petropoulos, A.E. (2014), *Low-Thrust Trajectories: An overview of the Q-law
and other analytic techniques*, 3rd talk in the "Optimal Control in Space
Mission Design" series, ISAS/JAXA. The "Gauss's Form of the Variational
Equations" slide gives the rate equations reproduced below; the "Optimal
Instantaneous Changes in Elements" slides give ``a_dot_xx``, ``e_dot_xx``.

Vallado, D.A. (2013), *Fundamentals of Astrodynamics and Applications*, 4th
ed., Microcosm Press, Sec. 9.3 ("Variation of Parameters") -- the same Gauss
equations with full derivation; the textbook the rest of this codebase already
cites for classical-element <-> Cartesian conversions
(see ``reflectors.elements``, ``reflectors.sun_sync``).

Conventions
-----------
- State ``(r, v)`` is Cartesian in any inertial frame (the propagator uses
  Mars-centred J2000). All elements are osculating, referenced to that frame.
- Angles in radians. ``a`` is negative for hyperbolic orbits (the standard
  conic convention, matching ``spice.oscltx`` and ``reflectors.elements``).
- The thrust frame is **RTN**: ``r_hat`` radial-outward, ``h_hat`` along the
  orbital angular momentum, ``theta_hat = h_hat x r_hat`` transverse (in-plane,
  in the direction of orbital motion). A thrust acceleration ``f`` decomposes
  as ``f_r = f . r_hat``, ``f_theta = f . theta_hat``, ``f_h = f . h_hat``.
- The true anomaly enters Gauss's equations as ``theta`` in the lecture and as
  ``nu`` in this codebase; they are the same quantity.

Near-circular regularisation
----------------------------
At ``e -> 0`` the true anomaly (angle from periapsis) is undefined and the
scalar ``de/dt`` is non-differentiable. ``osculating_elements`` regularises
this: for ``e < _E_CIRCULAR_TOL`` it reports ``nu = 0`` (i.e. it treats the
current position as periapsis) and ``argp = 0``. With that convention
``gauss_variational_rates`` returns a finite, physically-correct ``de/dt``
(transverse thrust grows ``e``; see the inline derivation). The escape Q-law
targets ``a`` and uses ``de/dt`` only through the periapsis penalty, which is
negligible at the near-circular launch orbit -- so the regularised first
instant carries no weight. RAAN / argp degeneracy at ``i -> 0`` is likewise
guarded; the escape Q-law does not use ``i``, ``RAAN`` or ``argp``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from reflectors.elements import ClassicalElements

logger = logging.getLogger(__name__)

__all__ = [
    "ElementRates",
    "osculating_elements",
    "rtn_basis",
    "gauss_variational_rates",
    "semimajor_axis_rate_max",
    "eccentricity_rate_max",
]


# Eccentricity below which the orbit is treated as circular: the true-anomaly
# origin (periapsis) is regularised to the current position. 1e-9 is far below
# any eccentricity the escape spiral dwells at (it pumps e to O(0.1) within the
# first revs) yet comfortably above float round-off in the e-vector norm.
_E_CIRCULAR_TOL = 1.0e-9

# Node-vector magnitude below which the orbit is treated as equatorial and the
# ascending node (hence RAAN) is regularised. Dimensionless: the node vector
# z_hat x h_hat has magnitude sin(i) times |h_hat|=1, so this is sin(i) ~ 1e-9.
_NODE_TOL = 1.0e-9


@dataclass(frozen=True)
class ElementRates:
    """Time derivatives of the classical elements from Gauss's equations.

    All rates are in SI-consistent units with the rest of the codebase:
    ``da_dt`` in km/s, the four angular rates in rad/s. Produced by
    :func:`gauss_variational_rates` for a given thrust acceleration.
    """

    da_dt_km_s: float
    de_dt_per_s: float
    di_dt_rad_s: float
    draan_dt_rad_s: float
    dargp_dt_rad_s: float
    dnu_dt_rad_s: float


def rtn_basis(
    r_vec_km: np.ndarray, v_vec_kmps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial / transverse / normal orthonormal basis at a Cartesian state.

    Returns ``(r_hat, theta_hat, h_hat)`` as unit 3-vectors in the input
    frame:

    - ``r_hat`` -- radial, from the central body toward the spacecraft.
    - ``h_hat`` -- along the orbital angular momentum ``r x v``.
    - ``theta_hat = h_hat x r_hat`` -- transverse, in the orbit plane and in
      the direction of orbital motion (it carries the positive-``v_theta``
      sense, since ``v . theta_hat = h / r > 0``).

    Raises
    ------
    ValueError
        If ``r`` is the zero vector or ``r`` and ``v`` are parallel (zero
        angular momentum -- a radial/degenerate trajectory with no orbit
        plane).
    """
    r = np.asarray(r_vec_km, dtype=float)
    v = np.asarray(v_vec_kmps, dtype=float)
    r_mag = float(np.linalg.norm(r))
    if r_mag == 0.0:
        raise ValueError("rtn_basis: position vector is zero")
    r_hat = r / r_mag
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag == 0.0:
        raise ValueError(
            "rtn_basis: zero angular momentum (r parallel to v); no orbit "
            "plane is defined"
        )
    h_hat = h_vec / h_mag
    theta_hat = np.cross(h_hat, r_hat)
    return r_hat, theta_hat, h_hat


def osculating_elements(
    r_vec_km: np.ndarray,
    v_vec_kmps: np.ndarray,
    mu_km3_s2: float,
) -> ClassicalElements:
    """Classical osculating elements from a Cartesian state -- SPICE-free.

    A numpy reimplementation of ``reflectors.elements.classical_elements``
    (which wraps ``spice.oscltx``), cheap enough for the propagator RHS hot
    loop. Returns the identical :class:`~reflectors.elements.ClassicalElements`
    frozen dataclass so the two are field-for-field comparable -- the SPICE
    path is the independent cross-check oracle (``tests/test_gauss.py``).

    Construction (Vallado 2013 Sec. 2.5 / 9.x):

    - ``h = r x v``;  specific energy ``E = v^2/2 - mu/r``;
      ``a = -mu / (2 E)`` (negative for hyperbolic).
    - eccentricity vector ``e_vec = (v x h)/mu - r_hat``;  ``e = |e_vec|``.
    - ``p = h^2 / mu`` (semi-latus rectum).
    - ``i = arccos(h_z / h)``.
    - node vector ``n = z_hat x h``;  ``RAAN = atan2(n_y, n_x)``.
    - ``argp`` between ``n`` and ``e_vec`` (quadrant from ``e_z``);
      ``nu`` between ``e_vec`` and ``r`` (quadrant from the sign of ``r . v``).

    Degenerate cases are regularised (see the module docstring): ``e <
    _E_CIRCULAR_TOL`` reports ``nu = argp = 0``; ``|n| < _NODE_TOL`` (equatorial)
    reports ``RAAN = 0`` and measures ``argp`` as the longitude of periapsis.

    Parameters
    ----------
    r_vec_km, v_vec_kmps
        Cartesian position (km) and velocity (km/s), shape (3,).
    mu_km3_s2
        Gravitational parameter of the central body, km^3/s^2.

    Returns
    -------
    ClassicalElements
    """
    r = np.asarray(r_vec_km, dtype=float)
    v = np.asarray(v_vec_kmps, dtype=float)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError(
            f"osculating_elements: r, v must be shape (3,), got "
            f"{r.shape}, {v.shape}"
        )
    if mu_km3_s2 <= 0.0:
        raise ValueError(f"mu_km3_s2 must be > 0, got {mu_km3_s2}")

    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    if r_mag == 0.0:
        raise ValueError("osculating_elements: position vector is zero")

    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag == 0.0:
        raise ValueError(
            "osculating_elements: zero angular momentum (rectilinear "
            "trajectory has no conic elements)"
        )

    energy = 0.5 * v_mag * v_mag - mu_km3_s2 / r_mag
    # a = -mu / 2E; negative for hyperbolic (E > 0), the conic convention.
    a_km = -mu_km3_s2 / (2.0 * energy)

    e_vec = np.cross(v, h_vec) / mu_km3_s2 - r / r_mag
    e = float(np.linalg.norm(e_vec))

    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag)))

    # Node vector n = z_hat x h_vec = (-h_y, h_x, 0).
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n_mag = float(np.linalg.norm(n_vec))
    if n_mag > _NODE_TOL:
        raan = math.atan2(n_vec[1], n_vec[0]) % (2.0 * math.pi)
    else:
        raan = 0.0

    if e > _E_CIRCULAR_TOL:
        # True anomaly: angle between e_vec and r, quadrant from sign(r . v).
        cos_nu = float(np.dot(e_vec, r)) / (e * r_mag)
        nu = math.acos(max(-1.0, min(1.0, cos_nu)))
        if float(np.dot(r, v)) < 0.0:
            nu = 2.0 * math.pi - nu
        if n_mag > _NODE_TOL:
            cos_argp = float(np.dot(n_vec, e_vec)) / (n_mag * e)
            argp = math.acos(max(-1.0, min(1.0, cos_argp)))
            if e_vec[2] < 0.0:
                argp = 2.0 * math.pi - argp
        else:
            # Equatorial: argp degenerates to the longitude of periapsis.
            argp = math.atan2(e_vec[1], e_vec[0]) % (2.0 * math.pi)
    else:
        # Near-circular regularisation: treat the current position as
        # periapsis. nu = argp = 0; see the module docstring.
        nu = 0.0
        argp = 0.0

    if a_km > 0.0:
        period_s = 2.0 * math.pi * math.sqrt(a_km ** 3 / mu_km3_s2)
    else:
        period_s = float("inf")

    return ClassicalElements(
        a_km=float(a_km),
        e=float(e),
        inclination_rad=float(inc),
        raan_rad=float(raan),
        argp_rad=float(argp),
        nu_rad=float(nu),
        period_s=float(period_s),
        mu_km3_s2=float(mu_km3_s2),
    )


def gauss_variational_rates(
    elements: ClassicalElements,
    f_r_km_s2: float,
    f_theta_km_s2: float,
    f_h_km_s2: float,
) -> ElementRates:
    """Classical-element rates from Gauss's variational equations.

    Implements Gauss's form of the variational equations (Petropoulos 2014
    lecture, "Gauss's Form of the Variational Equations" slide; Vallado 2013
    Sec. 9.3) for a thrust acceleration with RTN components
    ``(f_r, f_theta, f_h)`` -- radial-outward, transverse (direction of
    motion), and orbit-normal respectively, each in km/s^2:

        dRAAN/dt = r sin(u) / (h sin i) * f_h
        di/dt    = r cos(u) / h * f_h
        dargp/dt = (1/(e h)) [ -p cos(nu) f_r + (p + r) sin(nu) f_theta ]
                   - r sin(u) cos(i) / (h sin i) * f_h
        da/dt    = (2 a^2 / h) ( e sin(nu) f_r + (p / r) f_theta )
        de/dt    = (1/h) { p sin(nu) f_r
                           + [ (p + r) cos(nu) + r e ] f_theta }
        dnu/dt   = h / r^2 + (1/(e h)) [ p cos(nu) f_r - (p + r) sin(nu) f_theta ]

    with ``u = argp + nu`` the argument of latitude, ``h = sqrt(mu p)``,
    ``p = a (1 - e^2)``, and ``r = p / (1 + e cos nu)``. The ``dnu/dt`` value
    returned includes only the *thrust* contribution to the anomaly drift plus
    the Keplerian ``h/r^2`` term; the central-body two-body motion is carried
    by the Cartesian propagator, so callers integrating Cartesian state should
    not double-count ``dnu/dt``.

    Sign / normalisation: a transverse thrust (``f_theta > 0``, i.e. along the
    velocity) raises ``a`` -- ``da/dt = (2 a^2 / h)(p/r) f_theta > 0`` -- the
    orbit-raising sense the escape spiral exploits.

    Near-circular handling: when ``elements`` was built with the ``e ->
    _E_CIRCULAR_TOL`` regularisation (``nu = 0``), ``da/dt`` and ``de/dt``
    remain finite and correct -- ``de/dt = (1/h)(p + r + r e) f_theta``, the
    physical eccentricity-growth rate under transverse thrust. The ``dargp/dt``
    and ``dnu/dt`` thrust terms carry a ``1/e`` factor and are returned as
    ``0.0`` for ``e < _E_CIRCULAR_TOL`` (argp / nu are not meaningful there and
    the escape Q-law does not use them). The ``f_h`` terms of ``dRAAN/dt`` and
    ``dargp/dt`` carry ``1/sin i`` and are returned as ``0.0`` for
    ``sin i < _NODE_TOL`` (equatorial degeneracy).

    Parameters
    ----------
    elements
        Osculating elements at the state (from :func:`osculating_elements`).
    f_r_km_s2, f_theta_km_s2, f_h_km_s2
        RTN thrust-acceleration components, km/s^2.

    Returns
    -------
    ElementRates
    """
    a = elements.a_km
    e = elements.e
    inc = elements.inclination_rad
    argp = elements.argp_rad
    nu = elements.nu_rad
    mu = elements.mu_km3_s2

    p = a * (1.0 - e * e)
    if p <= 0.0:
        raise ValueError(
            f"gauss_variational_rates: non-positive semi-latus rectum "
            f"p={p} (a={a}, e={e}); state is not a valid conic"
        )
    h = math.sqrt(mu * p)
    r = p / (1.0 + e * math.cos(nu))
    u = argp + nu  # argument of latitude

    sin_nu = math.sin(nu)
    cos_nu = math.cos(nu)
    sin_u = math.sin(u)
    cos_u = math.cos(u)
    sin_i = math.sin(inc)

    # da/dt and de/dt -- no 1/e or 1/sin(i); valid for all e, i (the escape
    # Q-law's primary rates).
    da_dt = (2.0 * a * a / h) * (e * sin_nu * f_r_km_s2 + (p / r) * f_theta_km_s2)
    de_dt = (1.0 / h) * (
        p * sin_nu * f_r_km_s2
        + ((p + r) * cos_nu + r * e) * f_theta_km_s2
    )

    # di/dt -- no singular factor.
    di_dt = (r * cos_u / h) * f_h_km_s2

    # dRAAN/dt and the f_h term of dargp/dt carry 1/sin(i): guard equatorial.
    if sin_i > _NODE_TOL:
        draan_dt = (r * sin_u / (h * sin_i)) * f_h_km_s2
        dargp_fh = -(r * sin_u * math.cos(inc) / (h * sin_i)) * f_h_km_s2
    else:
        draan_dt = 0.0
        dargp_fh = 0.0

    # dargp/dt and dnu/dt thrust terms carry 1/e: guard near-circular.
    if e > _E_CIRCULAR_TOL:
        dargp_dt = (1.0 / (e * h)) * (
            -p * cos_nu * f_r_km_s2 + (p + r) * sin_nu * f_theta_km_s2
        ) + dargp_fh
        dnu_dt = h / (r * r) + (1.0 / (e * h)) * (
            p * cos_nu * f_r_km_s2 - (p + r) * sin_nu * f_theta_km_s2
        )
    else:
        dargp_dt = 0.0
        dnu_dt = h / (r * r)

    return ElementRates(
        da_dt_km_s=float(da_dt),
        de_dt_per_s=float(de_dt),
        di_dt_rad_s=float(di_dt),
        draan_dt_rad_s=float(draan_dt),
        dargp_dt_rad_s=float(dargp_dt),
        dnu_dt_rad_s=float(dnu_dt),
    )


def semimajor_axis_rate_max(
    a_km: float, e: float, f_mag_km_s2: float, mu_km3_s2: float
) -> float:
    """Locally-optimal maximum ``|da/dt|`` -- the Q-law normaliser ``a_dot_xx``.

    The largest ``|da/dt|`` achievable at an orbit ``(a, e)`` by any thrust
    direction at the most effective true anomaly, for a fixed thrust-
    acceleration magnitude ``f`` (Petropoulos 2014, "Optimal Instantaneous
    Changes in Elements" slide):

        a_dot_xx = 2 f sqrt( a^3 (1 + e) / (mu (1 - e)) ).

    Derivation: ``da/dt = (2 a^2 / h)(e sin nu f_r + (p/r) f_theta)``; for
    fixed ``f`` the direction-optimum is ``(2 a^2 f / h) sqrt(e^2 sin^2 nu +
    (1 + e cos nu)^2)`` (using ``p/r = 1 + e cos nu``). That radicand is
    maximised over ``nu`` at ``nu = 0`` (periapsis), giving ``(1 + e)``; with
    ``h = sqrt(mu a (1 - e^2))`` the result reduces to the closed form above.
    The optimum is a purely transverse thrust applied at periapsis.

    Valid for elliptic orbits ``0 <= e < 1``.

    Parameters
    ----------
    a_km
        Semi-major axis (km), must be > 0.
    e
        Eccentricity, ``0 <= e < 1``.
    f_mag_km_s2
        Thrust-acceleration magnitude, km/s^2.
    mu_km3_s2
        Central-body gravitational parameter, km^3/s^2.

    Returns
    -------
    float
        ``a_dot_xx`` in km/s (non-negative).
    """
    if a_km <= 0.0:
        raise ValueError(f"semimajor_axis_rate_max: a_km must be > 0, got {a_km}")
    if not (0.0 <= e < 1.0):
        raise ValueError(
            f"semimajor_axis_rate_max: e must satisfy 0 <= e < 1, got {e}"
        )
    if mu_km3_s2 <= 0.0:
        raise ValueError(f"mu_km3_s2 must be > 0, got {mu_km3_s2}")
    return 2.0 * f_mag_km_s2 * math.sqrt(
        a_km ** 3 * (1.0 + e) / (mu_km3_s2 * (1.0 - e))
    )


def eccentricity_rate_max(
    a_km: float, e: float, f_mag_km_s2: float, mu_km3_s2: float
) -> float:
    """Locally-optimal maximum ``|de/dt|`` -- the Q-law normaliser ``e_dot_xx``.

    The largest ``|de/dt|`` achievable at an orbit ``(a, e)`` by any thrust
    direction at the most effective true anomaly, for a fixed thrust-
    acceleration magnitude ``f`` (Petropoulos 2014, "Optimal Instantaneous
    Changes in Elements" slide):

        e_dot_xx = 2 p f / h,

    with ``p = a (1 - e^2)`` and ``h = sqrt(mu p)``; equivalently
    ``e_dot_xx = 2 f sqrt(p / mu)``. Verified against a brute-force maximum of
    :func:`gauss_variational_rates` over thrust direction and true anomaly in
    ``tests/test_gauss.py``.

    Parameters
    ----------
    a_km, e, f_mag_km_s2, mu_km3_s2
        As for :func:`semimajor_axis_rate_max`.

    Returns
    -------
    float
        ``e_dot_xx`` per second (non-negative).
    """
    if a_km <= 0.0:
        raise ValueError(f"eccentricity_rate_max: a_km must be > 0, got {a_km}")
    if not (0.0 <= e < 1.0):
        raise ValueError(
            f"eccentricity_rate_max: e must satisfy 0 <= e < 1, got {e}"
        )
    if mu_km3_s2 <= 0.0:
        raise ValueError(f"mu_km3_s2 must be > 0, got {mu_km3_s2}")
    p = a_km * (1.0 - e * e)
    return 2.0 * f_mag_km_s2 * math.sqrt(p / mu_km3_s2)
