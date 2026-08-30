"""Osculating classical orbital elements for diagnostics.

Propagation uses Cartesian state (see ``reflectors.dynamics``). Classical
elements (a, e, i, Omega, omega, nu) are computed post hoc from Cartesian
states for human-readable reporting and for the analytical J2 drift tests.

Conventions:

    a      semi-major axis (km). Negative for hyperbolic orbits by the
           conic-section convention SPICE uses.
    e      eccentricity (dimensionless).
    i      inclination (rad) -- angle between the orbit plane normal and the
           reference z-axis of the frame in which (r, v) are expressed.
    raan   longitude of ascending node (rad), 0 to 2 pi.
    argp   argument of periapsis (rad), 0 to 2 pi.
    nu     true anomaly (rad), 0 to 2 pi.
    period orbital period (s). ``inf`` for parabolic/hyperbolic.

The frame matters: inclination and RAAN are only physically meaningful if
``(r, v)`` are expressed in a frame whose z-axis is the reference axis of
interest. For orbits around Mars, if ``(r, v)`` is in Mars-centred J2000
(axes parallel to Earth's mean equator / equinox at J2000), the computed
inclination is with respect to J2000 -- NOT Mars's equator. For Mars-equator
elements, transform ``(r, v)`` into a Mars-mean-equator-of-J2000 frame first
(see ``mme2000_rotation_from_j2000``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import spiceypy as spice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassicalElements:
    """Osculating classical elements at a single epoch.

    See module docstring for conventions and frame caveats. ``mu`` and the
    frame used to compute ``(r, v)`` are recorded so downstream callers can
    re-verify.
    """

    a_km: float
    e: float
    inclination_rad: float
    raan_rad: float
    argp_rad: float
    nu_rad: float
    period_s: float
    mu_km3_s2: float

    @property
    def periapsis_km(self) -> float:
        return self.a_km * (1.0 - self.e)

    @property
    def apoapsis_km(self) -> float:
        if self.e >= 1.0:
            return float("inf")
        return self.a_km * (1.0 + self.e)

    @property
    def mean_motion_rad_s(self) -> float:
        if self.e >= 1.0 or self.a_km <= 0.0:
            return float("nan")
        return float(np.sqrt(self.mu_km3_s2 / self.a_km ** 3))

    @property
    def semi_latus_rectum_km(self) -> float:
        return self.a_km * (1.0 - self.e ** 2)


def classical_elements(
    state_km_kmps: np.ndarray,
    mu_km3_s2: float,
    epoch_et: float = 0.0,
) -> ClassicalElements:
    """Compute osculating classical elements from a Cartesian state.

    Thin wrapper around ``spice.oscltx``, which returns the classical six
    plus (true anomaly, semi-major axis, orbital period).
    """
    state = np.asarray(state_km_kmps, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"state must be shape (6,), got {state.shape}")
    elts = np.asarray(spice.oscltx(state, epoch_et, mu_km3_s2), dtype=float)
    rp, ecc, inc, lnode, argp, _m0, _t0, _mu_out, nu, a, tau = elts[:11]
    # spice.oscltx yields tau = 0.0 for non-elliptic orbits.
    period_s = float(tau) if tau > 0.0 else float("inf")
    return ClassicalElements(
        a_km=float(a),
        e=float(ecc),
        inclination_rad=float(inc),
        raan_rad=float(lnode),
        argp_rad=float(argp),
        nu_rad=float(nu),
        period_s=period_s,
        mu_km3_s2=float(mu_km3_s2),
    )


def mean_motion_rad_per_s(a_km: float, mu_km3_s2: float) -> float:
    """Keplerian mean motion ``sqrt(mu / a^3)`` (rad/s). Undefined for a<=0."""
    if a_km <= 0.0:
        return float("nan")
    return float(np.sqrt(mu_km3_s2 / a_km ** 3))


def semi_latus_rectum_km(a_km: float, e: float) -> float:
    """p = a (1 - e^2). Defined for any conic with |e| != 1."""
    return float(a_km * (1.0 - e * e))


def secular_argp_rate_J2_rad_per_s(
    a_km: float,
    e: float,
    inc_rad: float,
    *,
    mu_km3_s2: float | None = None,
    ref_radius_km: float | None = None,
    J2: float | None = None,
) -> float:
    """First-order Brouwer secular rate of argument-of-periapsis under J_2.

    Returns ``dω/dt`` in rad/s for a Keplerian orbit subject only to the
    second zonal harmonic of the central body. The formula is

        dω/dt = (3/4) · J_2 · n · (R/a)^2 · (5 cos^2(i) − 1) / (1 − e^2)^2

    where ``n = sqrt(μ/a^3)`` is the Keplerian mean motion. The sign
    flips through zero at the **critical inclination** ``5 cos^2 i = 1``,
    i.e. ``i ≈ 63.4349°`` (prograde) or ``116.5651°`` (retrograde). At
    Mars sun-sync inclination ``i ≈ 93.2°`` the rate is negative
    (argp regresses), at the K=12 reference orbit roughly ``−1.9 deg/sol``.

    Inputs are interpreted as **mean** elements in the strict Brouwer
    sense; passing osculating elements introduces an O(J_2^2) error in
    the predicted rate, ~4e-6 relative at the reference orbit, which is
    negligible versus the rate itself.

    Parameters
    ----------
    a_km : float
        Semi-major axis (km), > 0.
    e : float
        Eccentricity, ``0 <= e < 1``.
    inc_rad : float
        Inclination (rad).
    mu_km3_s2, ref_radius_km, J2 : float, optional
        Central-body GM, reference radius, and J_2 zonal coefficient.
        Default to the MRO120F gravity-model values via
        ``sun_sync._default_gravity_anchors`` — same source the rest of
        the codebase uses for J_2-driven analytical rates (cf.
        ``sun_sync.sun_sync_inclination_rad``).

    Returns
    -------
    float
        ``dω/dt`` in rad/s. Positive when ``5 cos^2 i > 1`` (i.e. ``|i|``
        below the critical inclination), negative above it.

    References
    ----------
    Brouwer, D. (1959). "Solution of the problem of artificial satellite
    theory without drag." *Astronomical Journal* 64: 378–396, Eq. (40)
    (secular ``dg''/dt`` under J_2).
    """
    if a_km <= 0.0:
        raise ValueError(f"a_km must be > 0, got {a_km}")
    if not (0.0 <= e < 1.0):
        raise ValueError(
            f"e must satisfy 0 <= e < 1 for elliptic orbits, got {e}"
        )
    if mu_km3_s2 is None or ref_radius_km is None or J2 is None:
        from reflectors.sun_sync import _default_gravity_anchors

        default_mu, default_R, default_J2 = _default_gravity_anchors()
        if mu_km3_s2 is None:
            mu_km3_s2 = default_mu
        if ref_radius_km is None:
            ref_radius_km = default_R
        if J2 is None:
            J2 = default_J2

    n = float(np.sqrt(mu_km3_s2 / a_km ** 3))
    cos_i = float(np.cos(inc_rad))
    one_minus_e2 = 1.0 - e * e
    return float(
        0.75
        * J2
        * n
        * (ref_radius_km / a_km) ** 2
        * (5.0 * cos_i * cos_i - 1.0)
        / (one_minus_e2 * one_minus_e2)
    )


# ---------------------------------------------------------------------------
# Reporting helper: rotation into Mars-mean-equator-of-J2000 (MME2000)
# ---------------------------------------------------------------------------


def mme2000_rotation_from_j2000(et: float = 0.0) -> np.ndarray:
    """3x3 rotation matrix from J2000 axes to Mars-mean-equator-of-DATE axes.

    The returned frame is inertial (no spin, unlike IAU_MARS): its z-axis is
    Mars's IAU-defined spin pole direction AT epoch ``et``, its x-axis lies
    along the ascending node of the Mars equator on the J2000 equator.

    Crucially, the pole direction is extracted from the full ``IAU_MARS``
    body-fixed rotation (SPICE ``pxform``), which includes all periodic
    nutation-precession terms of the IAU 2015 model. A naive approach using
    only the polynomial POLE_RA / POLE_DEC terms misses the 50+ Mars
    nutation harmonics and gives a pole biased by ~1.5 deg at 2026-era epochs.

    Note on terminology: strictly "MME2000" refers to the pole at J2000
    epoch (0). The function as implemented returns MME-of-date, which is
    appropriate for comparing simulated orbital elements against the
    analytical J_2 drift theory (which uses the equator w.r.t. which the
    zonal potential is axisymmetric -- i.e. Mars's current equator, not its
    J2000 equator). At et=0 the two frames coincide.
    """
    # Extract Mars spin-pole direction from the full IAU_MARS transform.
    # The body-fixed z-axis in J2000 equals pxform(J2000, IAU_MARS, et).T @ [0,0,1]
    # which equals the third row of pxform(IAU_MARS, J2000, et), i.e. the
    # third column of pxform(J2000, IAU_MARS, et).
    R_j2000_to_bf = np.asarray(spice.pxform("J2000", "IAU_MARS", et), dtype=float)
    pole_in_j2000 = R_j2000_to_bf[2, :].copy()  # row index 2 picks the body-fixed z-axis
    pole_in_j2000 /= np.linalg.norm(pole_in_j2000)

    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), pole_in_j2000)
    x_axis_norm = float(np.linalg.norm(x_axis))
    if x_axis_norm < 1e-12:
        raise RuntimeError("MME node degenerate: Mars pole aligns with J2000 z")
    x_axis /= x_axis_norm
    y_axis = np.cross(pole_in_j2000, x_axis)
    # Rows = MME-of-date axes expressed in J2000.
    return np.vstack([x_axis, y_axis, pole_in_j2000])


def elements_in_mme2000(
    state_j2000_km_kmps: np.ndarray,
    mu_km3_s2: float,
    epoch_et: float = 0.0,
) -> ClassicalElements:
    """Classical elements with inclination / RAAN referenced to Mars's equator.

    The input state is in Mars-centred J2000 and is rotated into MME2000
    (inertial, Mars-equatorial axes), where the elements are computed.
    Inclination is then the
    angle to Mars's equator -- the physically relevant quantity for
    sun-synchronous design.
    """
    R = mme2000_rotation_from_j2000(epoch_et)
    r = R @ state_j2000_km_kmps[:3]
    v = R @ state_j2000_km_kmps[3:]
    state_mme = np.concatenate([r, v])
    return classical_elements(state_mme, mu_km3_s2, epoch_et)


def state_from_classical_mme2000(
    a_km: float,
    e: float,
    inclination_rad: float,
    raan_rad: float,
    argp_rad: float,
    nu_rad: float,
    *,
    mu_km3_s2: float,
    epoch_et: float = 0.0,
) -> np.ndarray:
    """Inverse of ``elements_in_mme2000``: classical six -> Cartesian state.

    Elements are interpreted in MME2000 (Mars-mean-equator of the requested
    epoch, see ``mme2000_rotation_from_j2000``); the returned state is in
    Mars-centred J2000 (the frame ``propagate`` operates in).

    Construction (standard Vallado §2.6.4 / Curtis §4.5):
      1. Build ``(r_pf, v_pf)`` in the perifocal frame from
         ``p = a(1 - e^2)``, ``r = p / (1 + e cos nu)``, and the
         Keplerian velocity along the conic.
      2. Rotate perifocal -> MME2000 by ``R3(-raan) R1(-i) R3(-argp)``.
      3. Rotate MME2000 -> J2000 by the transpose of
         ``mme2000_rotation_from_j2000(epoch_et)``.

    ``argp`` is degenerate for ``e = 0``; callers should set it to 0 by
    convention and use ``nu`` as the angular position from the ascending
    node. For ``e = 0`` circular orbits, mean anomaly ``M`` equals true
    anomaly ``nu`` at the epoch.
    """
    if a_km <= 0.0:
        raise ValueError(f"a_km must be > 0, got {a_km}")
    if not (0.0 <= e < 1.0):
        raise ValueError(f"e must satisfy 0 <= e < 1 for elliptic orbits, got {e}")
    if mu_km3_s2 <= 0.0:
        raise ValueError(f"mu_km3_s2 must be > 0, got {mu_km3_s2}")

    p = a_km * (1.0 - e * e)
    cos_nu = np.cos(nu_rad)
    sin_nu = np.sin(nu_rad)
    r_mag = p / (1.0 + e * cos_nu)
    r_pf = np.array([r_mag * cos_nu, r_mag * sin_nu, 0.0])
    v_pf = np.sqrt(mu_km3_s2 / p) * np.array([-sin_nu, e + cos_nu, 0.0])

    c_raan = np.cos(raan_rad); s_raan = np.sin(raan_rad)
    c_i = np.cos(inclination_rad); s_i = np.sin(inclination_rad)
    c_argp = np.cos(argp_rad); s_argp = np.sin(argp_rad)

    R_z_raan = np.array([
        [c_raan, -s_raan, 0.0],
        [s_raan,  c_raan, 0.0],
        [   0.0,     0.0, 1.0],
    ])
    R_x_i = np.array([
        [1.0, 0.0,  0.0],
        [0.0, c_i, -s_i],
        [0.0, s_i,  c_i],
    ])
    R_z_argp = np.array([
        [c_argp, -s_argp, 0.0],
        [s_argp,  c_argp, 0.0],
        [   0.0,     0.0, 1.0],
    ])
    R_pf_to_mme = R_z_raan @ R_x_i @ R_z_argp

    r_mme = R_pf_to_mme @ r_pf
    v_mme = R_pf_to_mme @ v_pf

    R_j2000_to_mme = mme2000_rotation_from_j2000(epoch_et)
    r_j2000 = R_j2000_to_mme.T @ r_mme
    v_j2000 = R_j2000_to_mme.T @ v_mme

    return np.concatenate([r_j2000, v_j2000])
