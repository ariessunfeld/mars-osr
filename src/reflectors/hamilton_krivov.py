"""Orbit-averaged radiation-pressure plus J2 benchmark of Hamilton & Krivov.

This module implements the two-dimensional, orbit-averaged dust model used
for the Deimos and Phobos panels of Fig. 2 in Hamilton & Krivov (1996),
*Icarus* 123, 503-523.  It is a verification oracle for the Cartesian force
model, not a full mission propagator.

The paper assumes that the grain orbit, the planet equator, and the planet's
heliocentric orbit are coplanar and that the planet follows a circular orbit
about the Sun.  The independent variable is solar longitude ``lambda_sun``.
For the Phobos RP+J2 problem, solar tides and electromagnetism are neglected
(``A = L_tilde = 0``), leaving Eqs. (7) and (9)

    dphi/dlambda = C sqrt(1-e^2) cos(phi) / e
                  + W / (1-e^2)^2 - 1,

    de/dlambda   = C sqrt(1-e^2) sin(phi),

    H(e, phi)    = sqrt(1-e^2) + C e cos(phi)
                  + W / (3 (1-e^2)^(3/2)).

Here ``phi`` is longitude of pericenter relative to the Sun, ``C`` is the
radiative parameter, and ``W`` is the oblateness parameter.  The polar
equations are singular at the initially circular state ``e = 0``.  Following
their semicanonical formulation, the equivalent eccentricity-vector system
``x = e cos(phi)``, ``y = e sin(phi)`` is used below:

    dx/dlambda = -g(e) y,
    dy/dlambda =  C sqrt(1-e^2) + g(e) x,
    g(e)        =  W/(1-e^2)^2 - 1.

This form is regular at ``e = 0`` and is obtained algebraically from Eq. (7),
without a numerical regularization parameter.

Primary source
--------------
Hamilton, D. P. & Krivov, A. V. (1996), "Circumplanetary Dust Dynamics:
Effects of Solar Gravity, Radiation Pressure, Planetary Oblateness, and
Electromagnetism," *Icarus* 123, 503-523, especially pp. 506-517,
Eqs. (7), (9), (17), (26), and (29)-(32).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


# Values printed by Hamilton & Krivov (1996), pp. 508 and 513.  These remain
# literature constants rather than being replaced by the current
# MRO120F/SPICE values; the Cartesian comparison reports that distinction.
HK96_DEIMOS_C_TIMES_RADIUS_UM: float = 7.684
HK96_PHOBOS_C_TIMES_RADIUS_UM: float = 4.858
HK96_PHOBOS_W: float = 0.8290


@dataclass(frozen=True)
class RPJ2Transition:
    """One analytically defined transition in the RP+J2 phase portrait."""

    eccentricity: float
    radiative_parameter: float

    def grain_radius_um(self, c_times_radius_um: float) -> float:
        """Return ``r_g`` from the paper's inverse-radius relation ``C r_g``."""
        if c_times_radius_um <= 0.0:
            raise ValueError("c_times_radius_um must be positive")
        return c_times_radius_um / self.radiative_parameter


def _validate_eccentricity(eccentricity: float) -> None:
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError(f"eccentricity must be in [0, 1), got {eccentricity}")


def _validate_rp_j2_parameters(radiative_parameter: float, oblateness_parameter: float) -> None:
    if radiative_parameter <= 0.0:
        raise ValueError("radiative_parameter C must be positive")
    if not (0.0 < oblateness_parameter < 1.0):
        raise ValueError("oblateness_parameter W must lie strictly between 0 and 1")


def radiative_parameter(c_times_radius_um: float, grain_radius_um: float) -> float:
    """Return ``C`` for the paper's relation ``C = (C r_g) / r_g``."""
    if c_times_radius_um <= 0.0 or grain_radius_um <= 0.0:
        raise ValueError("c_times_radius_um and grain_radius_um must be positive")
    return c_times_radius_um / grain_radius_um


def rp_j2_hamiltonian(
    eccentricity: float,
    solar_angle_rad: float,
    radiative_parameter_C: float,
    oblateness_parameter_W: float,
) -> float:
    """Hamilton & Krivov (1996) Eq. (9), specialized to RP+J2."""
    _validate_eccentricity(eccentricity)
    _validate_rp_j2_parameters(radiative_parameter_C, oblateness_parameter_W)
    one_minus_e2 = 1.0 - eccentricity * eccentricity
    return (
        math.sqrt(one_minus_e2)
        + radiative_parameter_C * eccentricity * math.cos(solar_angle_rad)
        + oblateness_parameter_W / (3.0 * one_minus_e2 ** 1.5)
    )


def _circular_level_residual(
    eccentricity: float,
    cos_solar_angle: float,
    radiative_parameter_C: float,
    oblateness_parameter_W: float,
) -> float:
    """Evaluate ``H(e, phi) - H(0)`` without small-e cancellation.

    Direct subtraction loses the ``O(e^2)`` gravity/oblateness terms when
    ``e`` is tiny.  ``sqrt(1-e^2)-1`` is rationalized and
    ``(1-e^2)^(-3/2)-1`` is evaluated with ``expm1(log1p(...))``.
    """
    _validate_eccentricity(eccentricity)
    e2 = eccentricity * eccentricity
    root = math.sqrt(1.0 - e2)
    root_minus_one = -e2 / (1.0 + root)
    inverse_power_minus_one = math.expm1(-1.5 * math.log1p(-e2))
    return (
        root_minus_one
        + radiative_parameter_C * eccentricity * cos_solar_angle
        + (oblateness_parameter_W / 3.0) * inverse_power_minus_one
    )


def rp_j2_eccentricity_vector_rhs(
    lambda_sun_rad: float,
    eccentricity_vector: np.ndarray,
    radiative_parameter_C: float,
    oblateness_parameter_W: float,
) -> np.ndarray:
    """Nonsingular eccentricity-vector form of Hamilton & Krivov Eq. (7).

    ``lambda_sun_rad`` is unused because the circular, orbit-averaged system
    is autonomous; it remains in the signature for ODE integrators.
    """
    del lambda_sun_rad
    _validate_rp_j2_parameters(radiative_parameter_C, oblateness_parameter_W)
    xy = np.asarray(eccentricity_vector, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"eccentricity_vector must have shape (2,), got {xy.shape}")
    x, y = float(xy[0]), float(xy[1])
    e2 = x * x + y * y
    if e2 >= 1.0:
        raise ValueError("eccentricity-vector magnitude must remain below one")
    one_minus_e2 = 1.0 - e2
    g = oblateness_parameter_W / (one_minus_e2 * one_minus_e2) - 1.0
    return np.array(
        [-g * y, radiative_parameter_C * math.sqrt(one_minus_e2) + g * x],
        dtype=float,
    )


def rp_j2_hamiltonian_from_vector(
    eccentricity_vector: np.ndarray,
    radiative_parameter_C: float,
    oblateness_parameter_W: float,
) -> float:
    """Evaluate Eq. (9) directly from ``(e cos(phi), e sin(phi))``."""
    xy = np.asarray(eccentricity_vector, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"eccentricity_vector must have shape (2,), got {xy.shape}")
    x, y = float(xy[0]), float(xy[1])
    e2 = x * x + y * y
    if not (0.0 <= e2 < 1.0):
        raise ValueError("eccentricity-vector magnitude must remain below one")
    one_minus_e2 = 1.0 - e2
    return (
        math.sqrt(one_minus_e2)
        + radiative_parameter_C * x
        + oblateness_parameter_W / (3.0 * one_minus_e2 ** 1.5)
    )


def radiation_pressure_only_max_eccentricity(radiative_parameter_C: float) -> float:
    """Maximum eccentricity from an initially circular orbit, Eq. (17)."""
    if not (0.0 < radiative_parameter_C < 1.0):
        raise ValueError("Eq. (17) requires 0 < radiative_parameter C < 1")
    return 2.0 * radiative_parameter_C / (1.0 + radiative_parameter_C ** 2)


def rp_j2_stationary_point_bifurcation(oblateness_parameter_W: float) -> RPJ2Transition:
    """Return the stationary-point bifurcation from Eqs. (26) and (32).

    This is the 232-micrometer Phobos landmark.  It is distinct from the
    331.5-micrometer discontinuity in the maximum-eccentricity curve.
    """
    if not (0.0 < oblateness_parameter_W < 1.0):
        raise ValueError("oblateness_parameter W must lie strictly between 0 and 1")
    W = oblateness_parameter_W
    e2 = math.sqrt(1.0 + 2.0 * W - math.sqrt(4.0 * W * W + 5.0 * W))
    one_minus_e2 = 1.0 - e2 * e2
    h0_derivative = e2 / math.sqrt(one_minus_e2) * (
        W / (one_minus_e2 * one_minus_e2) - 1.0
    )
    return RPJ2Transition(eccentricity=e2, radiative_parameter=-h0_derivative)


def rp_j2_separatrix_transition(oblateness_parameter_W: float) -> RPJ2Transition:
    """Return the maximum-eccentricity jump from Eqs. (29)-(31).

    At this transition the initially circular level curve passes through the
    saddle on the positive eccentricity-vector axis.  The saddle condition
    eliminates ``C``; the remaining scalar level-curve equation is solved
    between the Eq. (26) minimum and the J2-only equilibrium of Eq. (20).
    """
    stationary = rp_j2_stationary_point_bifurcation(oblateness_parameter_W)
    W = oblateness_parameter_W
    e_j2 = math.sqrt(1.0 - math.sqrt(W))

    def stationary_C(eccentricity: float) -> float:
        one_minus_e2 = 1.0 - eccentricity * eccentricity
        return eccentricity / math.sqrt(one_minus_e2) * (
            1.0 - W / (one_minus_e2 * one_minus_e2)
        )

    def residual(eccentricity: float) -> float:
        C = stationary_C(eccentricity)
        return _circular_level_residual(eccentricity, 1.0, C, W)

    e_critical = brentq(
        residual,
        stationary.eccentricity,
        e_j2 * (1.0 - 1.0e-12),
        xtol=1.0e-14,
        rtol=1.0e-14,
    )
    return RPJ2Transition(
        eccentricity=e_critical,
        radiative_parameter=stationary_C(e_critical),
    )


def rp_j2_max_eccentricity(
    radiative_parameter_C: float,
    oblateness_parameter_W: float,
) -> float:
    """Maximum ``e`` on the initially circular RP+J2 level curve.

    Hamilton & Krivov classify the topology on pp. 514-516.  Above the
    separatrix value of ``C`` (smaller grains), the maximum lies on the
    negative eccentricity-vector axis and is the nonzero root of
    ``H(e, pi) = H(0)``.  Below it (larger grains), the attainable maximum is
    the first positive-axis root of ``H(e, 0) = H(0)``.  At the transition,
    the orbit approaches the saddle asymptotically and the returned value is
    the paper's ``e4``.
    """
    _validate_rp_j2_parameters(radiative_parameter_C, oblateness_parameter_W)
    transition = rp_j2_separatrix_transition(oblateness_parameter_W)
    C = radiative_parameter_C
    W = oblateness_parameter_W
    if math.isclose(C, transition.radiative_parameter, rel_tol=1.0e-12, abs_tol=0.0):
        return transition.eccentricity

    if C > transition.radiative_parameter:
        return brentq(
            lambda e: _circular_level_residual(e, -1.0, C, W),
            1.0e-15,
            1.0 - 1.0e-12,
            xtol=1.0e-14,
            rtol=1.0e-14,
        )

    return brentq(
        lambda e: _circular_level_residual(e, 1.0, C, W),
        1.0e-15,
        transition.eccentricity,
        xtol=1.0e-14,
        rtol=1.0e-14,
    )


__all__ = [
    "HK96_DEIMOS_C_TIMES_RADIUS_UM",
    "HK96_PHOBOS_C_TIMES_RADIUS_UM",
    "HK96_PHOBOS_W",
    "RPJ2Transition",
    "radiative_parameter",
    "radiation_pressure_only_max_eccentricity",
    "rp_j2_eccentricity_vector_rhs",
    "rp_j2_hamiltonian",
    "rp_j2_hamiltonian_from_vector",
    "rp_j2_max_eccentricity",
    "rp_j2_separatrix_transition",
    "rp_j2_stationary_point_bifurcation",
]
