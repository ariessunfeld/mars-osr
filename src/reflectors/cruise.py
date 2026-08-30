"""Parametric cruise-attitude family for SRP-driven orbit maintenance.

The module provides parametric cruise-attitude families that expose the sail's
secular orbital-element control authority as optimization variables. Families
range from a fixed cone-and-clock offset to mode-1, mode-2, and mode-3 Fourier
series in argument of latitude.

Formulation
-----------

The sail normal is parameterised as a small angular offset from the
sun direction, phase-locked to the argument-of-latitude ``u`` in the
sail's orbit plane:

    n_hat = cos(beta) * s_hat(r, t) + sin(beta) * offset_dir(r, t)
    offset_dir(r, t) = cos(u + phi_u) * e_A(t) + sin(u + phi_u) * e_B(t)

where

    s_hat(r, t) = unit(r_sun(t) - r)             -- sail-to-Sun direction
    e_A(t)      = unit(n_orb - (n_orb . s_hat) s_hat)   -- orbit normal
                                                   projected perpendicular
                                                   to s_hat
    e_B(t)      = s_hat x e_A                    -- completes the
                                                   orthonormal basis of
                                                   the plane perpendicular
                                                   to s_hat
    u(r)        = argument of latitude of r in the orbit plane

``n_orb`` is the orbit-normal unit vector, FROZEN at closure
construction from the initial state ``(r_0, v_0)``. For a 1-sol
schedule the orbit plane drifts at the sun-sync node-regression rate
(2 pi / Mars sidereal year ~= 0.01 rad/sol at 501 km), which is well
below the ``beta`` resolution any optimiser would use; periodic
reconstruction of ``n_orb`` is not needed at this horizon.

Limits and identities
---------------------

- ``sun_offset(beta=0, phi_u=0, ...)`` returns a callable that is
  bit-for-bit identical to ``sun_pointing()`` at any (r, t): the
  ``sin(beta) * offset_dir`` term vanishes and ``cos(beta) = 1``.
- The rotation axis for the Rodrigues operation is implicit: n_hat
  is computed directly via the linear combination above, which is
  equivalent to rotating s_hat by angle ``beta`` about
  ``offset_dir x s_hat``. This form is numerically cleaner than
  calling a rotation routine.

SRP authority connection
------------------------

The secular d(RAAN)/dt from SRP, to first order, scales as
``<W(u) sin(u)>`` where ``W(u)`` is the cross-track component of the
SRP acceleration at argument-of-latitude ``u``. For this attitude:

    cos(alpha_SRP) = s_hat . n_hat = cos(beta)
    W(u) proportional to sin(2 beta) * sin(u + phi_u)
         (in the plane perpendicular to s_hat, rotating with u)

so <W sin(u)> is maximised at ``phi_u = 0`` and varies as
``sin(2 beta)``. The maximum of this factor occurs at ``beta = 45 deg``;
at ``beta = 10 deg`` its value is ``sin(20 deg) / 1 ≈ 0.342`` of that
maximum.

Harmonic cone-angle extension
-----------------------------

``sun_offset_harmonic`` adds a Fourier-mode-1 modulation of the cone
angle in argument-of-latitude ``u`` while keeping the clock angle
``phi_u`` constant:

    alpha(u) = alpha_0 + alpha_c * cos(u) + alpha_s * sin(u)
    n_hat   = cos(alpha(u)) * s_hat + sin(alpha(u)) * offset_dir(u)

The cos(u) and sin(u) components select the secular Fourier modes of
the (S, T, W) perturbing acceleration that drive ``<dOmega/dt>``,
``<de/dt>``, and ``<di/dt>`` per McInnes 1999 ch. 4 Eqs. 4.14a-f and
4.15a-c. ``alpha_c = alpha_s = 0`` reduces ``sun_offset_harmonic``
bit-for-bit to ``sun_offset(alpha_0, phi_u)``; the constant family is
the DC-only special case.

Harmonic cone-and-clock extension
---------------------------------

``sun_offset_harmonic_full`` adds harmonic modulation of both McInnes
angles: the cone angle alpha and the clock angle delta:

    alpha(u) = alpha_0 + alpha_c * cos(u) + alpha_s * sin(u)
    delta(u) = delta_0 + delta_c * cos(u) + delta_s * sin(u)
    n_hat   = cos(alpha(u)) * s_hat
            + sin(alpha(u)) * [cos(u + delta(u)) * e_A
                              + sin(u + delta(u)) * e_B]

``sun_offset_harmonic`` with constant delta is the
``delta_c = delta_s = 0`` reduction of this family. Justification for
the additional clock-angle freedom is McInnes Eq. 4.15 b/c:

    T = beta_SRP * mu/r^2 * cos^2(alpha) * sin(alpha) * sin(delta)
    W = beta_SRP * mu/r^2 * cos^2(alpha) * sin(alpha) * cos(delta)

With CONSTANT delta, every Fourier mode of ``cos^2(alpha) sin(alpha)``
appears in both T (in-plane) and W (cross-track) at the FIXED ratio
``sin(delta) : cos(delta)``. Harmonic alpha alone cannot selectively
attack ``<de/dt>`` without simultaneously perturbing ``<da/dt>`` and
``<di/dt>``. Time-varying
delta(u) breaks this fixed ratio at each Fourier mode, providing the
structural lever for independent T and W control across modes.

Out of scope
------------

- Dynamic orbit plane tracking (n_orb updated along the propagation).
  Frozen at construction; justified for 1-sol schedules. Refresh
  externally between sols for multi-sol horizons.
- Rigid-body attitude dynamics and actuator sizing.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import spiceypy as spice

from reflectors.attitude import AttitudeCallable
from reflectors.ephemeris import sun_state_j2000


MARS_NAIF_ID = 499
SUN_NAIF_ID = 10


def _orbit_plane_normal_j2000(
    initial_state_km_kmps: np.ndarray,
) -> np.ndarray:
    """Unit orbit-normal vector (J2000) from a 6-vector state.

    Parameters
    ----------
    initial_state_km_kmps
        Sail state vector: positions in km followed by velocities in
        km/s, shape (6,).

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector ``(r x v) / |r x v|`` in J2000 axes, defining the
        orbit plane at the reference epoch.

    Raises
    ------
    ValueError
        If position and velocity are collinear (degenerate orbit).
    """
    r = np.asarray(initial_state_km_kmps[:3], dtype=float)
    v = np.asarray(initial_state_km_kmps[3:6], dtype=float)
    h = np.cross(r, v)
    h_mag = float(np.linalg.norm(h))
    if h_mag == 0.0:
        raise ValueError(
            "orbit plane normal is undefined: r and v are collinear "
            f"(|r x v| = 0). r={r}, v={v}."
        )
    return h / h_mag


def _orbit_plane_basis_j2000(
    initial_state_km_kmps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal basis ``(e_ref, e_ortho, n_orb)`` in J2000.

    ``e_ref`` is the initial position unit vector; ``n_orb`` is the
    orbit normal; ``e_ortho = n_orb x e_ref`` completes the right-
    handed triad. ``(e_ref, e_ortho)`` spans the orbit plane; the
    argument of latitude ``u`` at position ``r`` is
    ``atan2(r . e_ortho, r . e_ref)``, which equals 0 at the initial
    position and advances monotonically with orbital motion.
    """
    r0 = np.asarray(initial_state_km_kmps[:3], dtype=float)
    r0_mag = float(np.linalg.norm(r0))
    if r0_mag == 0.0:
        raise ValueError("initial position vector is zero")
    e_ref = r0 / r0_mag
    n_orb = _orbit_plane_normal_j2000(initial_state_km_kmps)
    e_ortho = np.cross(n_orb, e_ref)
    # e_ortho is unit-norm by construction because n_orb and e_ref are
    # both unit vectors and perpendicular (n_orb . e_ref = 0 by
    # definition of orbit normal).
    return e_ref, e_ortho, n_orb


def _validate_orbit_plane_inputs(
    orbit_plane_normal_j2000: np.ndarray,
    orbit_ref_direction_j2000: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise + validate the frozen orbit-plane basis inputs.

    Returns ``(n_orb, e_ref, e_ortho)`` as unit vectors, with
    ``e_ortho = n_orb x e_ref`` normalised. Shared by both ``sun_offset``
    and ``sun_offset_harmonic`` so the validation contract is identical.
    """
    n_orb = np.asarray(orbit_plane_normal_j2000, dtype=float)
    e_ref = np.asarray(orbit_ref_direction_j2000, dtype=float)
    if float(np.linalg.norm(n_orb)) == 0.0:
        raise ValueError("orbit_plane_normal_j2000 is the zero vector")
    if float(np.linalg.norm(e_ref)) == 0.0:
        raise ValueError("orbit_ref_direction_j2000 is the zero vector")
    n_orb = n_orb / np.linalg.norm(n_orb)
    e_ref = e_ref / np.linalg.norm(e_ref)
    e_ortho = np.cross(n_orb, e_ref)
    e_ortho_mag = float(np.linalg.norm(e_ortho))
    if e_ortho_mag == 0.0:
        raise ValueError(
            "orbit_ref_direction is parallel to orbit_plane_normal; "
            "they must span the orbit plane."
        )
    e_ortho = e_ortho / e_ortho_mag
    return n_orb, e_ref, e_ortho


def _per_call_geometry(
    r_sat_km: np.ndarray,
    et: float,
    *,
    n_orb: np.ndarray,
    e_ref: np.ndarray,
    e_ortho: np.ndarray,
    observer_naif_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-call attitude geometry: ``(s_hat, e_A, e_B, u)``.

    Shared by ``sun_offset`` and ``sun_offset_harmonic`` so both families
    pin the same Sun direction, (e_A, e_B) basis (with degenerate-pole
    fallback), and argument-of-latitude convention.

    ``s_hat`` is the sail-to-Sun unit vector; ``(e_A, e_B)`` is an
    orthonormal basis for the plane perpendicular to ``s_hat`` with
    ``e_A`` chosen as the orbit-normal projection (or a fallback when
    the Sun crosses the orbit pole); ``u = atan2(r . e_ortho,
    r . e_ref)`` is the argument of latitude in the frozen orbit plane.
    """
    state = sun_state_j2000(et, observer_naif_id)
    sat_to_sun = np.asarray(state[:3], dtype=float) - np.asarray(
        r_sat_km, dtype=float
    )
    s_hat = sat_to_sun / float(np.linalg.norm(sat_to_sun))

    # Orbit-normal projected perpendicular to s_hat.
    n_perp = n_orb - float(np.dot(n_orb, s_hat)) * s_hat
    n_perp_mag = float(np.linalg.norm(n_perp))
    if n_perp_mag < 1.0e-12:
        # Orbit normal nearly aligned with Sun direction (sun crossing
        # the orbit pole). Fallback: use any unit vector perpendicular
        # to s_hat. Choose e_ref projected ortho.
        fallback = e_ref - float(np.dot(e_ref, s_hat)) * s_hat
        fb_mag = float(np.linalg.norm(fallback))
        if fb_mag < 1.0e-12:
            # e_ref too is parallel to s_hat (unlikely). Use the
            # x-axis as a last resort.
            fallback = np.array([1.0, 0.0, 0.0])
            fallback = fallback - float(np.dot(fallback, s_hat)) * s_hat
            fb_mag = float(np.linalg.norm(fallback))
        e_A = fallback / fb_mag
    else:
        e_A = n_perp / n_perp_mag
    e_B = np.cross(s_hat, e_A)

    # Argument of latitude in the orbit plane.
    r = np.asarray(r_sat_km, dtype=float)
    u = math.atan2(float(np.dot(r, e_ortho)), float(np.dot(r, e_ref)))

    return s_hat, e_A, e_B, u


def sun_offset(
    beta_rad: float,
    phi_u_rad: float = 0.0,
    *,
    orbit_plane_normal_j2000: np.ndarray,
    orbit_ref_direction_j2000: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
) -> AttitudeCallable:
    """Parametric cruise attitude offset from the Sun by angle beta.

    The sail normal is ``n_hat = cos(beta) * s_hat + sin(beta) *
    offset_dir(u)`` where ``s_hat`` is the Sun direction from the sail
    and ``offset_dir`` is a unit vector in the plane perpendicular to
    ``s_hat``, phase-locked to argument-of-latitude ``u`` with phase
    offset ``phi_u``.

    ``beta = 0`` reproduces ``sun_pointing()`` exactly.

    Parameters
    ----------
    beta_rad
        Offset angle from the Sun direction, radians. Zero = pure
        sun-pointing. Must be in ``[0, pi/2]`` (larger would rotate
        past orthogonal, losing SRP authority).
    phi_u_rad
        Phase offset of the offset-direction rotation in the
        perpendicular-to-Sun plane, radians. 0 means the offset at
        ``u = 0`` lies along the orbit-normal-projected direction;
        pi/2 means it lies in-plane perpendicular to Sun.
    orbit_plane_normal_j2000
        Frozen orbit-normal unit vector ``n_orb``, J2000 axes,
        shape (3,). Typically from
        ``_orbit_plane_normal_j2000(initial_state)``.
    orbit_ref_direction_j2000
        Frozen orbit-plane reference direction ``e_ref``, J2000 axes,
        shape (3,). Usually the initial position direction; sets the
        zero of the argument-of-latitude coordinate.
    observer_naif_id
        NAIF id of the central body for SPICE Sun ephemeris;
        default 499 (Mars).

    Returns
    -------
    AttitudeCallable
        ``(r_sat_km, et) -> n_hat_j2000``. Pure (no state carried);
        safe to reuse across RHS steps.

    Raises
    ------
    ValueError
        If ``beta_rad`` is out of ``[0, pi/2]``; if either frozen
        direction is degenerate.
    """
    if not 0.0 <= beta_rad <= math.pi / 2.0:
        raise ValueError(
            f"beta_rad must be in [0, pi/2], got {beta_rad}"
        )
    n_orb, e_ref, e_ortho = _validate_orbit_plane_inputs(
        orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    )

    cos_beta = math.cos(float(beta_rad))
    sin_beta = math.sin(float(beta_rad))
    phi_u = float(phi_u_rad)

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        s_hat, e_A, e_B, u = _per_call_geometry(
            r_sat_km, et,
            n_orb=n_orb, e_ref=e_ref, e_ortho=e_ortho,
            observer_naif_id=observer_naif_id,
        )

        # beta=0 short circuit (also guards against numerical noise
        # from the e_A / e_B basis when sin(beta) is zero).
        if sin_beta == 0.0:
            return s_hat

        cos_phase = math.cos(u + phi_u)
        sin_phase = math.sin(u + phi_u)
        offset_dir = cos_phase * e_A + sin_phase * e_B

        n_hat = cos_beta * s_hat + sin_beta * offset_dir
        # Numerical cleanup: n_hat SHOULD be unit-norm by construction
        # (s_hat and offset_dir are orthogonal unit vectors), but
        # re-normalise to absorb tiny FP drift.
        return n_hat / float(np.linalg.norm(n_hat))

    return _n_hat


def sun_offset_from_state(
    beta_rad: float,
    phi_u_rad: float = 0.0,
    *,
    initial_state_km_kmps: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
) -> AttitudeCallable:
    """Convenience wrapper: compute the frozen orbit-plane basis from
    an initial state and return a ``sun_offset`` closure.

    Equivalent to calling ``_orbit_plane_basis_j2000`` to get
    ``(e_ref, e_ortho, n_orb)``, then calling ``sun_offset(beta_rad,
    phi_u_rad, orbit_plane_normal_j2000=n_orb,
    orbit_ref_direction_j2000=e_ref, ...)``.

    Parameters
    ----------
    beta_rad, phi_u_rad
        As for ``sun_offset``.
    initial_state_km_kmps
        6-vector sail state at the reference epoch; used only to
        derive the frozen orbit-plane basis.
    observer_naif_id
        As for ``sun_offset``.
    """
    e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(initial_state_km_kmps)
    return sun_offset(
        beta_rad,
        phi_u_rad,
        orbit_plane_normal_j2000=n_orb,
        orbit_ref_direction_j2000=e_ref,
        observer_naif_id=observer_naif_id,
    )


def sun_offset_harmonic(
    alpha_0_rad: float,
    *,
    alpha_c_rad: float = 0.0,
    alpha_s_rad: float = 0.0,
    phi_u_rad: float = 0.0,
    orbit_plane_normal_j2000: np.ndarray,
    orbit_ref_direction_j2000: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
) -> AttitudeCallable:
    """Harmonic-in-u cruise attitude family (Fourier-mode-1 cone angle).

    The sail normal modulates with argument-of-latitude ``u``:

        alpha(u) = alpha_0 + alpha_c * cos(u) + alpha_s * sin(u)
        n_hat   = cos(alpha(u)) * s_hat
                + sin(alpha(u)) * offset_dir(u, phi_u)

    where ``s_hat`` is the Sun direction from the sail and
    ``offset_dir(u, phi_u) = cos(u + phi_u) * e_A + sin(u + phi_u) * e_B``
    spans the plane perpendicular to ``s_hat`` (same basis convention
    as ``sun_offset``).

    ``alpha_c = alpha_s = 0`` reduces bit-for-bit to
    ``sun_offset(alpha_0_rad, phi_u_rad, ...)``.

    Physics anchor: McInnes 1999 ch. 4 Eqs. 4.7 (sail-normal),
    4.14a-f (Lagrange variational equations), 4.15a-c (SRP force in
    S, T, W). Cos(u) and sin(u) Fourier modes of alpha(u) couple
    through cos^k(alpha) sin(alpha) factors in S/T/W to the secular
    averages <da/dt>, <de/dt>, <di/dt>, <dOmega/dt>.

    Parameters
    ----------
    alpha_0_rad
        DC component of the cone angle, radians. Constant-cruise
        special case: alpha_c = alpha_s = 0 reduces to
        ``sun_offset(alpha_0, phi_u)``. Must satisfy
        ``alpha_0 in [0, pi/2]``.
    alpha_c_rad, alpha_s_rad
        Cosine and sine harmonic amplitudes of the cone angle in u,
        radians. Default 0 (constant cruise). Must satisfy the
        non-saturation constraint ``alpha_0 +/- sqrt(alpha_c^2 +
        alpha_s^2) in [0, pi/2]`` so alpha(u) never leaves the valid
        cone-angle range.
    phi_u_rad
        Constant clock-angle offset for ``offset_dir(u)``, radians.
        Same role as in ``sun_offset``.
    orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    observer_naif_id
        As for ``sun_offset``.

    Returns
    -------
    AttitudeCallable
        ``(r_sat_km, et) -> n_hat_j2000``.

    Raises
    ------
    ValueError
        If ``alpha_0`` is outside ``[0, pi/2]``; if
        ``alpha_0 + sqrt(alpha_c^2 + alpha_s^2) > pi/2``; if
        ``alpha_0 - sqrt(alpha_c^2 + alpha_s^2) < 0``; if either
        frozen direction is degenerate.
    """
    alpha_0 = float(alpha_0_rad)
    alpha_c = float(alpha_c_rad)
    alpha_s = float(alpha_s_rad)
    if not 0.0 <= alpha_0 <= math.pi / 2.0:
        raise ValueError(
            f"alpha_0_rad must be in [0, pi/2], got {alpha_0}"
        )
    # Tight bound on alpha(u) extrema: max |alpha_c cos u + alpha_s sin u|
    # = sqrt(alpha_c^2 + alpha_s^2).
    alpha_amp = math.hypot(alpha_c, alpha_s)
    if alpha_0 + alpha_amp > math.pi / 2.0:
        raise ValueError(
            f"alpha_0 + sqrt(alpha_c^2 + alpha_s^2) must be <= pi/2; "
            f"got alpha_0={alpha_0}, alpha_amp={alpha_amp}, "
            f"sum={alpha_0 + alpha_amp}"
        )
    if alpha_0 - alpha_amp < 0.0:
        raise ValueError(
            f"alpha_0 - sqrt(alpha_c^2 + alpha_s^2) must be >= 0; "
            f"got alpha_0={alpha_0}, alpha_amp={alpha_amp}, "
            f"diff={alpha_0 - alpha_amp}"
        )

    n_orb, e_ref, e_ortho = _validate_orbit_plane_inputs(
        orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    )

    phi_u = float(phi_u_rad)

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        s_hat, e_A, e_B, u = _per_call_geometry(
            r_sat_km, et,
            n_orb=n_orb, e_ref=e_ref, e_ortho=e_ortho,
            observer_naif_id=observer_naif_id,
        )

        alpha_u = alpha_0 + alpha_c * math.cos(u) + alpha_s * math.sin(u)
        sin_alpha = math.sin(alpha_u)

        # alpha(u) = 0 short circuit: pure sun-pointing at this u.
        # Matches sun_offset's "beta=0 reduces to sun_pointing"
        # behaviour at the per-sample level.
        if sin_alpha == 0.0:
            return s_hat

        cos_alpha = math.cos(alpha_u)
        cos_phase = math.cos(u + phi_u)
        sin_phase = math.sin(u + phi_u)
        offset_dir = cos_phase * e_A + sin_phase * e_B

        n_hat = cos_alpha * s_hat + sin_alpha * offset_dir
        return n_hat / float(np.linalg.norm(n_hat))

    return _n_hat


def sun_offset_harmonic_from_state(
    alpha_0_rad: float,
    *,
    alpha_c_rad: float = 0.0,
    alpha_s_rad: float = 0.0,
    phi_u_rad: float = 0.0,
    initial_state_km_kmps: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
) -> AttitudeCallable:
    """Convenience wrapper for ``sun_offset_harmonic``.

    Mirrors ``sun_offset_from_state``: computes the frozen orbit-plane
    basis from an initial state, forwards to ``sun_offset_harmonic``.

    Parameters
    ----------
    alpha_0_rad, alpha_c_rad, alpha_s_rad, phi_u_rad
        As for ``sun_offset_harmonic``.
    initial_state_km_kmps
        6-vector sail state at the reference epoch; used only to
        derive the frozen orbit-plane basis.
    observer_naif_id
        As for ``sun_offset_harmonic``.
    """
    e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(initial_state_km_kmps)
    return sun_offset_harmonic(
        alpha_0_rad,
        alpha_c_rad=alpha_c_rad,
        alpha_s_rad=alpha_s_rad,
        phi_u_rad=phi_u_rad,
        orbit_plane_normal_j2000=n_orb,
        orbit_ref_direction_j2000=e_ref,
        observer_naif_id=observer_naif_id,
    )


# Default cap on the harmonic-clock-angle amplitude
# sqrt(delta_c^2 + delta_s^2). Pi/2 keeps the offset_dir rotation rate
# bounded by ~2 * n_orbit (since dδ/du ≤ δ_amp at the Cauchy-Schwarz
# extremum, the term (1 + dδ/du) * n_orbit in |dn̂/dt| stays at most ~2x
# the constant-δ case). Above π/2, |dn̂/dt| and |d²n̂/dt²| can grow
# enough that α_max becomes a binding constraint rather than a
# diagnostic. Pinned here, kwarg-overridable on the function.
DELTA_AMP_MAX_DEFAULT_RAD = math.pi / 2.0


def sun_offset_harmonic_full(
    alpha_0_rad: float,
    *,
    alpha_c_rad: float = 0.0,
    alpha_s_rad: float = 0.0,
    delta_0_rad: float = 0.0,
    delta_c_rad: float = 0.0,
    delta_s_rad: float = 0.0,
    orbit_plane_normal_j2000: np.ndarray,
    orbit_ref_direction_j2000: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
    delta_amp_max_rad: float = DELTA_AMP_MAX_DEFAULT_RAD,
) -> AttitudeCallable:
    """Harmonic-in-u cruise family with BOTH McInnes angles modulated.

    Extends ``sun_offset_harmonic`` by adding a Fourier-mode-1
    modulation of the clock angle delta in addition to the cone angle
    alpha:

        alpha(u) = alpha_0 + alpha_c * cos(u) + alpha_s * sin(u)
        delta(u) = delta_0 + delta_c * cos(u) + delta_s * sin(u)
        n_hat   = cos(alpha(u)) * s_hat
                + sin(alpha(u)) * [cos(u + delta(u)) * e_A
                                  + sin(u + delta(u)) * e_B]

    where ``s_hat`` is the Sun direction from the sail and ``(e_A, e_B)``
    is an orthonormal basis for the plane perpendicular to ``s_hat``
    (same ``_per_call_geometry`` convention as ``sun_offset`` and
    ``sun_offset_harmonic``).

    ``delta_c = delta_s = 0`` reduces bit-for-bit to
    ``sun_offset_harmonic(alpha_0_rad, alpha_c_rad=alpha_c, alpha_s_rad=alpha_s,
    phi_u_rad=delta_0, ...)``. ``alpha_c = alpha_s = delta_c = delta_s = 0``
    further reduces to ``sun_offset(alpha_0, delta_0)``.

    Physics anchor: McInnes 1999 ch. 4 Eqs. 4.7 (sail-normal),
    4.14a-f (Lagrange variational equations), 4.15a-c (SRP force in
    S, T, W components):

        S = beta_SRP * mu/r^2 * cos^3(alpha)
        T = beta_SRP * mu/r^2 * cos^2(alpha) sin(alpha) sin(delta)
        W = beta_SRP * mu/r^2 * cos^2(alpha) sin(alpha) cos(delta)

    Constant delta couples T and W via the shared
    ``cos^2(alpha) sin(alpha)`` factor at fixed ``sin(delta) : cos(delta)``
    ratio; harmonic delta(u) breaks that ratio across Fourier modes,
    enabling independent T and W control -- the structural unlock for
    selectively damping ``<de/dt>`` without driving ``<da/dt>``.

    Parameters
    ----------
    alpha_0_rad
        DC component of the cone angle, radians. Must satisfy
        ``alpha_0 in [0, pi/2]``.
    alpha_c_rad, alpha_s_rad
        Cosine and sine harmonic amplitudes of the cone angle in u,
        radians. Default 0. Must satisfy
        ``alpha_0 +/- sqrt(alpha_c^2 + alpha_s^2) in [0, pi/2]``.
    delta_0_rad
        DC component of the clock angle, radians. Plays the same role
        as ``phi_u_rad`` in ``sun_offset`` / ``sun_offset_harmonic``;
        named ``delta_0`` here to align with McInnes Eq. 4.7
        nomenclature and parallel the ``alpha_*_rad`` triad. Unbounded
        (clock angle is physically periodic in 2 pi; the optimiser
        treats wrap-around cosmetically).
    delta_c_rad, delta_s_rad
        Cosine and sine harmonic amplitudes of the clock angle in u,
        radians. Default 0. Soft amplitude bound
        ``sqrt(delta_c^2 + delta_s^2) <= delta_amp_max_rad`` for
        optimisation tractability (above ~pi/2 the offset_dir rotation
        rate term ``d(delta)/du * n_orbit`` becomes comparable to
        ``n_orbit`` itself and ``alpha_max`` starts binding).
    phi_u handling
        ``sun_offset_harmonic_full`` does NOT take a separate
        ``phi_u_rad`` kwarg: ``delta_0_rad`` is the corresponding ``phi_u``.
        Two names for one quantity in different families would invite
        confusion; this family standardises on ``delta_*``.
    orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    observer_naif_id
        As for ``sun_offset_harmonic``.
    delta_amp_max_rad
        Soft amplitude bound on ``sqrt(delta_c^2 + delta_s^2)``;
        default ``DELTA_AMP_MAX_DEFAULT_RAD = pi/2``.

    Returns
    -------
    AttitudeCallable
        ``(r_sat_km, et) -> n_hat_j2000``.

    Raises
    ------
    ValueError
        If ``alpha_0`` is outside ``[0, pi/2]``; if
        ``alpha_0 +/- sqrt(alpha_c^2 + alpha_s^2)`` leaves
        ``[0, pi/2]``; if
        ``sqrt(delta_c^2 + delta_s^2) > delta_amp_max_rad``; if either
        frozen direction is degenerate.
    """
    alpha_0 = float(alpha_0_rad)
    alpha_c = float(alpha_c_rad)
    alpha_s = float(alpha_s_rad)
    delta_0 = float(delta_0_rad)
    delta_c = float(delta_c_rad)
    delta_s = float(delta_s_rad)
    delta_amp_max = float(delta_amp_max_rad)

    if not 0.0 <= alpha_0 <= math.pi / 2.0:
        raise ValueError(
            f"alpha_0_rad must be in [0, pi/2], got {alpha_0}"
        )
    alpha_amp = math.hypot(alpha_c, alpha_s)
    if alpha_0 + alpha_amp > math.pi / 2.0:
        raise ValueError(
            f"alpha_0 + sqrt(alpha_c^2 + alpha_s^2) must be <= pi/2; "
            f"got alpha_0={alpha_0}, alpha_amp={alpha_amp}, "
            f"sum={alpha_0 + alpha_amp}"
        )
    if alpha_0 - alpha_amp < 0.0:
        raise ValueError(
            f"alpha_0 - sqrt(alpha_c^2 + alpha_s^2) must be >= 0; "
            f"got alpha_0={alpha_0}, alpha_amp={alpha_amp}, "
            f"diff={alpha_0 - alpha_amp}"
        )
    if delta_amp_max < 0.0:
        raise ValueError(
            f"delta_amp_max_rad must be >= 0, got {delta_amp_max}"
        )
    delta_amp = math.hypot(delta_c, delta_s)
    if delta_amp > delta_amp_max:
        raise ValueError(
            f"sqrt(delta_c^2 + delta_s^2) must be <= delta_amp_max_rad; "
            f"got delta_amp={delta_amp}, delta_amp_max={delta_amp_max}"
        )

    n_orb, e_ref, e_ortho = _validate_orbit_plane_inputs(
        orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    )

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        s_hat, e_A, e_B, u = _per_call_geometry(
            r_sat_km, et,
            n_orb=n_orb, e_ref=e_ref, e_ortho=e_ortho,
            observer_naif_id=observer_naif_id,
        )

        cos_u = math.cos(u)
        sin_u = math.sin(u)
        alpha_u = alpha_0 + alpha_c * cos_u + alpha_s * sin_u
        delta_u = delta_0 + delta_c * cos_u + delta_s * sin_u
        sin_alpha = math.sin(alpha_u)

        # alpha(u) = 0 short circuit: pure sun-pointing at this u.
        # Matches the "alpha=0 reduces to sun_pointing"
        # behaviour at the per-sample level.
        if sin_alpha == 0.0:
            return s_hat

        cos_alpha = math.cos(alpha_u)
        cos_phase = math.cos(u + delta_u)
        sin_phase = math.sin(u + delta_u)
        offset_dir = cos_phase * e_A + sin_phase * e_B

        n_hat = cos_alpha * s_hat + sin_alpha * offset_dir
        return n_hat / float(np.linalg.norm(n_hat))

    return _n_hat


def sun_offset_harmonic_full_from_state(
    alpha_0_rad: float,
    *,
    alpha_c_rad: float = 0.0,
    alpha_s_rad: float = 0.0,
    delta_0_rad: float = 0.0,
    delta_c_rad: float = 0.0,
    delta_s_rad: float = 0.0,
    initial_state_km_kmps: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
    delta_amp_max_rad: float = DELTA_AMP_MAX_DEFAULT_RAD,
) -> AttitudeCallable:
    """Convenience wrapper for ``sun_offset_harmonic_full``.

    Mirrors ``sun_offset_harmonic_from_state``: computes the frozen
    orbit-plane basis from an initial state, forwards to
    ``sun_offset_harmonic_full``.

    Parameters
    ----------
    alpha_0_rad, alpha_c_rad, alpha_s_rad, delta_0_rad, delta_c_rad,
    delta_s_rad
        As for ``sun_offset_harmonic_full``.
    initial_state_km_kmps
        6-vector sail state at the reference epoch; used only to
        derive the frozen orbit-plane basis.
    observer_naif_id, delta_amp_max_rad
        As for ``sun_offset_harmonic_full``.
    """
    e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(initial_state_km_kmps)
    return sun_offset_harmonic_full(
        alpha_0_rad,
        alpha_c_rad=alpha_c_rad,
        alpha_s_rad=alpha_s_rad,
        delta_0_rad=delta_0_rad,
        delta_c_rad=delta_c_rad,
        delta_s_rad=delta_s_rad,
        orbit_plane_normal_j2000=n_orb,
        orbit_ref_direction_j2000=e_ref,
        observer_naif_id=observer_naif_id,
        delta_amp_max_rad=delta_amp_max_rad,
    )


# ---------------------------------------------------------------------------
# Mode-2 harmonic family
# ---------------------------------------------------------------------------


def sun_offset_harmonic_full_mode2(
    alpha_0_rad: float,
    *,
    alpha_c1_rad: float = 0.0,
    alpha_s1_rad: float = 0.0,
    alpha_c2_rad: float = 0.0,
    alpha_s2_rad: float = 0.0,
    delta_0_rad: float = 0.0,
    delta_c1_rad: float = 0.0,
    delta_s1_rad: float = 0.0,
    delta_c2_rad: float = 0.0,
    delta_s2_rad: float = 0.0,
    orbit_plane_normal_j2000: np.ndarray,
    orbit_ref_direction_j2000: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
    delta_amp_max_rad: float = DELTA_AMP_MAX_DEFAULT_RAD,
) -> AttitudeCallable:
    """Harmonic-(α, δ) cruise with mode-1 + mode-2 Fourier components.

    Extends ``sun_offset_harmonic_full`` (mode-1) by adding mode-2
    cosine/sine terms in the argument-of-latitude expansion of both
    angles:

        alpha(u) = alpha_0
                 + alpha_c1 cos(u)  + alpha_s1 sin(u)
                 + alpha_c2 cos(2u) + alpha_s2 sin(2u)
        delta(u) = delta_0
                 + delta_c1 cos(u)  + delta_s1 sin(u)
                 + delta_c2 cos(2u) + delta_s2 sin(2u)
        n_hat   = cos(alpha(u)) s_hat
                + sin(alpha(u)) [cos(u + delta(u)) e_A
                                + sin(u + delta(u)) e_B]

    Bit-for-bit reduction: ``alpha_c2 = alpha_s2 = delta_c2 = delta_s2 = 0``
    yields ``sun_offset_harmonic_full(alpha_0, alpha_c=alpha_c1,
    alpha_s=alpha_s1, delta_0=delta_0, delta_c=delta_c1,
    delta_s=delta_s1, ...)`` exactly.

    Physics anchor. McInnes 1999 ch. 4 Eqs. 4.14a-f and 4.15a-c
    establish that the secular averages <da/dt>, <de/dt>, <di/dt>,
    <dOmega/dt> couple to specific Fourier modes of the perturbing
    accelerations (S, T, W). The analytical authority bound at e = 0 captures
    the unrestricted ceiling under any cruise law. Mode-2 adds the
    ``cos(2u), sin(2u)`` Fourier
    components of alpha and delta, accessing higher-frequency content
    of the (S, T, W) integrand and plausibly closing more of the gap
    -- particularly for <da/dt> and <de/dt> where Eq. 4.14a-b carries
    a ``cos(u)`` factor and the integrand's mode-2 content survives
    averaging.

    Cone-angle bound (conservative). The peak alpha-deviation from
    alpha_0 is bounded by the triangle inequality across modes:

        |alpha(u) - alpha_0| <= sqrt(alpha_c1^2 + alpha_s1^2)
                              + sqrt(alpha_c2^2 + alpha_s2^2)

    Equality is achieved when the mode phases align; in general the
    actual peak is below this bound. The condition
    ``alpha_0 +/- (alpha_amp_1 + alpha_amp_2) in [0, pi/2]`` is enforced,
    ensuring that the cone-angle constraint is satisfied at every u. This
    conservative bound gives the optimizer a slightly smaller feasible box
    than the true cone-angle physics permits.

    Delta-amplitude soft cap (consistent with mode-1). Same triangle-
    inequality logic on the delta amplitudes:

        delta_amp_total = sqrt(delta_c1^2 + delta_s1^2)
                        + sqrt(delta_c2^2 + delta_s2^2)
        delta_amp_total <= delta_amp_max_rad

    Default ``delta_amp_max_rad = DELTA_AMP_MAX_DEFAULT_RAD = pi/2``.

    Parameters
    ----------
    alpha_0_rad
        DC component of the cone angle, radians. Must satisfy
        ``alpha_0 in [0, pi/2]``.
    alpha_c1_rad, alpha_s1_rad
        Mode-1 cosine/sine harmonic amplitudes of the cone angle.
    alpha_c2_rad, alpha_s2_rad
        Mode-2 cosine/sine harmonic amplitudes of the cone angle.
    delta_0_rad
        DC component of the clock angle, radians.
    delta_c1_rad, delta_s1_rad
        Mode-1 cosine/sine harmonic amplitudes of the clock angle.
    delta_c2_rad, delta_s2_rad
        Mode-2 cosine/sine harmonic amplitudes of the clock angle.
    orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    observer_naif_id, delta_amp_max_rad
        As for ``sun_offset_harmonic_full``.

    Raises
    ------
    ValueError
        If ``alpha_0`` is outside ``[0, pi/2]``;
        if conservative cone-angle bound puts ``alpha`` outside
        ``[0, pi/2]``;
        if conservative delta-amplitude bound exceeds
        ``delta_amp_max_rad``;
        if either frozen direction is degenerate.
    """
    alpha_0 = float(alpha_0_rad)
    a_c1 = float(alpha_c1_rad); a_s1 = float(alpha_s1_rad)
    a_c2 = float(alpha_c2_rad); a_s2 = float(alpha_s2_rad)
    delta_0 = float(delta_0_rad)
    d_c1 = float(delta_c1_rad); d_s1 = float(delta_s1_rad)
    d_c2 = float(delta_c2_rad); d_s2 = float(delta_s2_rad)
    delta_amp_max = float(delta_amp_max_rad)

    if not 0.0 <= alpha_0 <= math.pi / 2.0:
        raise ValueError(
            f"alpha_0_rad must be in [0, pi/2], got {alpha_0}"
        )
    a_amp1 = math.hypot(a_c1, a_s1)
    a_amp2 = math.hypot(a_c2, a_s2)
    a_amp_total = a_amp1 + a_amp2
    if alpha_0 + a_amp_total > math.pi / 2.0:
        raise ValueError(
            f"alpha_0 + (sqrt(alpha_c1^2+alpha_s1^2) + "
            f"sqrt(alpha_c2^2+alpha_s2^2)) must be <= pi/2; "
            f"got alpha_0={alpha_0}, amp1={a_amp1}, amp2={a_amp2}, "
            f"sum={alpha_0 + a_amp_total}"
        )
    if alpha_0 - a_amp_total < 0.0:
        raise ValueError(
            f"alpha_0 - (sqrt(alpha_c1^2+alpha_s1^2) + "
            f"sqrt(alpha_c2^2+alpha_s2^2)) must be >= 0; "
            f"got alpha_0={alpha_0}, amp1={a_amp1}, amp2={a_amp2}, "
            f"diff={alpha_0 - a_amp_total}"
        )
    if delta_amp_max < 0.0:
        raise ValueError(
            f"delta_amp_max_rad must be >= 0, got {delta_amp_max}"
        )
    d_amp1 = math.hypot(d_c1, d_s1)
    d_amp2 = math.hypot(d_c2, d_s2)
    d_amp_total = d_amp1 + d_amp2
    if d_amp_total > delta_amp_max:
        raise ValueError(
            f"sqrt(delta_c1^2+delta_s1^2) + "
            f"sqrt(delta_c2^2+delta_s2^2) must be <= delta_amp_max_rad; "
            f"got amp1={d_amp1}, amp2={d_amp2}, sum={d_amp_total}, "
            f"delta_amp_max={delta_amp_max}"
        )

    n_orb, e_ref, e_ortho = _validate_orbit_plane_inputs(
        orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    )

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        s_hat, e_A, e_B, u = _per_call_geometry(
            r_sat_km, et,
            n_orb=n_orb, e_ref=e_ref, e_ortho=e_ortho,
            observer_naif_id=observer_naif_id,
        )

        cos_u = math.cos(u)
        sin_u = math.sin(u)
        cos_2u = math.cos(2.0 * u)
        sin_2u = math.sin(2.0 * u)
        alpha_u = (
            alpha_0
            + a_c1 * cos_u + a_s1 * sin_u
            + a_c2 * cos_2u + a_s2 * sin_2u
        )
        delta_u = (
            delta_0
            + d_c1 * cos_u + d_s1 * sin_u
            + d_c2 * cos_2u + d_s2 * sin_2u
        )
        sin_alpha = math.sin(alpha_u)

        if sin_alpha == 0.0:
            return s_hat

        cos_alpha = math.cos(alpha_u)
        cos_phase = math.cos(u + delta_u)
        sin_phase = math.sin(u + delta_u)
        offset_dir = cos_phase * e_A + sin_phase * e_B

        n_hat = cos_alpha * s_hat + sin_alpha * offset_dir
        return n_hat / float(np.linalg.norm(n_hat))

    return _n_hat


def sun_offset_harmonic_full_mode2_from_state(
    alpha_0_rad: float,
    *,
    alpha_c1_rad: float = 0.0,
    alpha_s1_rad: float = 0.0,
    alpha_c2_rad: float = 0.0,
    alpha_s2_rad: float = 0.0,
    delta_0_rad: float = 0.0,
    delta_c1_rad: float = 0.0,
    delta_s1_rad: float = 0.0,
    delta_c2_rad: float = 0.0,
    delta_s2_rad: float = 0.0,
    initial_state_km_kmps: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
    delta_amp_max_rad: float = DELTA_AMP_MAX_DEFAULT_RAD,
) -> AttitudeCallable:
    """Convenience wrapper for ``sun_offset_harmonic_full_mode2``.

    Computes the frozen orbit-plane basis from an initial state and
    forwards to ``sun_offset_harmonic_full_mode2``. Same usage pattern
    as ``sun_offset_harmonic_full_from_state``.
    """
    e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(initial_state_km_kmps)
    return sun_offset_harmonic_full_mode2(
        alpha_0_rad,
        alpha_c1_rad=alpha_c1_rad, alpha_s1_rad=alpha_s1_rad,
        alpha_c2_rad=alpha_c2_rad, alpha_s2_rad=alpha_s2_rad,
        delta_0_rad=delta_0_rad,
        delta_c1_rad=delta_c1_rad, delta_s1_rad=delta_s1_rad,
        delta_c2_rad=delta_c2_rad, delta_s2_rad=delta_s2_rad,
        orbit_plane_normal_j2000=n_orb,
        orbit_ref_direction_j2000=e_ref,
        observer_naif_id=observer_naif_id,
        delta_amp_max_rad=delta_amp_max_rad,
    )


# ---------------------------------------------------------------------------
# Mode-3 harmonic family
# ---------------------------------------------------------------------------


def sun_offset_harmonic_full_mode3(
    alpha_0_rad: float,
    *,
    alpha_c1_rad: float = 0.0,
    alpha_s1_rad: float = 0.0,
    alpha_c2_rad: float = 0.0,
    alpha_s2_rad: float = 0.0,
    alpha_c3_rad: float = 0.0,
    alpha_s3_rad: float = 0.0,
    delta_0_rad: float = 0.0,
    delta_c1_rad: float = 0.0,
    delta_s1_rad: float = 0.0,
    delta_c2_rad: float = 0.0,
    delta_s2_rad: float = 0.0,
    delta_c3_rad: float = 0.0,
    delta_s3_rad: float = 0.0,
    orbit_plane_normal_j2000: np.ndarray,
    orbit_ref_direction_j2000: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
    delta_amp_max_rad: float = DELTA_AMP_MAX_DEFAULT_RAD,
) -> AttitudeCallable:
    """Harmonic-(α, δ) cruise with mode-1 + mode-2 + mode-3 Fourier components.

    Extends ``sun_offset_harmonic_full_mode2`` by adding the
    mode-3 cosine/sine components ``cos(3u), sin(3u)`` to both alpha and
    delta:

        alpha(u) = alpha_0
                 + alpha_c1 cos(u)  + alpha_s1 sin(u)
                 + alpha_c2 cos(2u) + alpha_s2 sin(2u)
                 + alpha_c3 cos(3u) + alpha_s3 sin(3u)
        delta(u) = delta_0
                 + delta_c1 cos(u)  + delta_s1 sin(u)
                 + delta_c2 cos(2u) + delta_s2 sin(2u)
                 + delta_c3 cos(3u) + delta_s3 sin(3u)
        n_hat   = cos(alpha(u)) s_hat
                + sin(alpha(u)) [cos(u + delta(u)) e_A
                                + sin(u + delta(u)) e_B]

    Bit-for-bit reduction: setting all mode-3 amplitudes to zero yields
    ``sun_offset_harmonic_full_mode2`` exactly; setting both mode-2 and
    mode-3 amplitudes to zero yields ``sun_offset_harmonic_full``
    exactly.

    Physics anchor. McInnes 1999 ch. 4 Eqs. 4.14a-f and 4.15a-c. Mode-3
    adds ``cos(3u), sin(3u)`` content,
    accessing the next octave of (S, T, W) Fourier structure. The
    secular averages of orbital-element rates pick up specific Fourier
    modes (e.g., <de/dt> couples to the cos(u) mode of S and the
    cos(u)*sin(u) ~ sin(2u)/2 mode of T per Eq. 4.14b), so mode-3
    primarily extends the cruise law's representational power for terms
    involving cos(2u)*cos(u) ~ (cos(u)+cos(3u))/2 and similar
    triple-products that appear in the cross-terms.

    Cone-angle bound (conservative). Triangle inequality across all
    three modes:

        |alpha(u) - alpha_0| <= sqrt(alpha_c1^2 + alpha_s1^2)
                              + sqrt(alpha_c2^2 + alpha_s2^2)
                              + sqrt(alpha_c3^2 + alpha_s3^2)

    Enforced as
    ``alpha_0 +/- (alpha_amp_1 + alpha_amp_2 + alpha_amp_3) in
    [0, pi/2]``.

    Delta-amplitude soft cap (consistent with mode-2):
    ``alpha_amp_1 + alpha_amp_2 + alpha_amp_3``-style triangle on the
    delta amplitudes:

        delta_amp_total = sqrt(delta_c1^2 + delta_s1^2)
                        + sqrt(delta_c2^2 + delta_s2^2)
                        + sqrt(delta_c3^2 + delta_s3^2)
        delta_amp_total <= delta_amp_max_rad

    Default ``delta_amp_max_rad = DELTA_AMP_MAX_DEFAULT_RAD = pi/2``.

    Parameters
    ----------
    alpha_0_rad
        DC component of the cone angle, radians. Must satisfy
        ``alpha_0 in [0, pi/2]``.
    alpha_c{1,2,3}_rad, alpha_s{1,2,3}_rad
        Cosine/sine amplitudes of the cone angle at orbital harmonics
        1, 2, 3.
    delta_0_rad
        DC component of the clock angle, radians.
    delta_c{1,2,3}_rad, delta_s{1,2,3}_rad
        Cosine/sine amplitudes of the clock angle at orbital harmonics
        1, 2, 3.
    orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    observer_naif_id, delta_amp_max_rad
        As for ``sun_offset_harmonic_full_mode2``.

    Raises
    ------
    ValueError
        If ``alpha_0`` is outside ``[0, pi/2]``;
        if conservative cone-angle bound puts ``alpha`` outside
        ``[0, pi/2]``;
        if conservative delta-amplitude bound exceeds
        ``delta_amp_max_rad``;
        if either frozen direction is degenerate.
    """
    alpha_0 = float(alpha_0_rad)
    a_c1 = float(alpha_c1_rad); a_s1 = float(alpha_s1_rad)
    a_c2 = float(alpha_c2_rad); a_s2 = float(alpha_s2_rad)
    a_c3 = float(alpha_c3_rad); a_s3 = float(alpha_s3_rad)
    delta_0 = float(delta_0_rad)
    d_c1 = float(delta_c1_rad); d_s1 = float(delta_s1_rad)
    d_c2 = float(delta_c2_rad); d_s2 = float(delta_s2_rad)
    d_c3 = float(delta_c3_rad); d_s3 = float(delta_s3_rad)
    delta_amp_max = float(delta_amp_max_rad)

    if not 0.0 <= alpha_0 <= math.pi / 2.0:
        raise ValueError(
            f"alpha_0_rad must be in [0, pi/2], got {alpha_0}"
        )
    a_amp1 = math.hypot(a_c1, a_s1)
    a_amp2 = math.hypot(a_c2, a_s2)
    a_amp3 = math.hypot(a_c3, a_s3)
    a_amp_total = a_amp1 + a_amp2 + a_amp3
    if alpha_0 + a_amp_total > math.pi / 2.0:
        raise ValueError(
            f"alpha_0 + (amp1 + amp2 + amp3) must be <= pi/2; "
            f"got alpha_0={alpha_0}, amp1={a_amp1}, amp2={a_amp2}, "
            f"amp3={a_amp3}, sum={alpha_0 + a_amp_total}"
        )
    if alpha_0 - a_amp_total < 0.0:
        raise ValueError(
            f"alpha_0 - (amp1 + amp2 + amp3) must be >= 0; "
            f"got alpha_0={alpha_0}, amp1={a_amp1}, amp2={a_amp2}, "
            f"amp3={a_amp3}, diff={alpha_0 - a_amp_total}"
        )
    if delta_amp_max < 0.0:
        raise ValueError(
            f"delta_amp_max_rad must be >= 0, got {delta_amp_max}"
        )
    d_amp1 = math.hypot(d_c1, d_s1)
    d_amp2 = math.hypot(d_c2, d_s2)
    d_amp3 = math.hypot(d_c3, d_s3)
    d_amp_total = d_amp1 + d_amp2 + d_amp3
    if d_amp_total > delta_amp_max:
        raise ValueError(
            f"sqrt(delta_c1^2+delta_s1^2) + sqrt(delta_c2^2+delta_s2^2) + "
            f"sqrt(delta_c3^2+delta_s3^2) must be <= delta_amp_max_rad; "
            f"got amp1={d_amp1}, amp2={d_amp2}, amp3={d_amp3}, "
            f"sum={d_amp_total}, delta_amp_max={delta_amp_max}"
        )

    n_orb, e_ref, e_ortho = _validate_orbit_plane_inputs(
        orbit_plane_normal_j2000, orbit_ref_direction_j2000,
    )

    def _n_hat(r_sat_km: np.ndarray, et: float) -> np.ndarray:
        s_hat, e_A, e_B, u = _per_call_geometry(
            r_sat_km, et,
            n_orb=n_orb, e_ref=e_ref, e_ortho=e_ortho,
            observer_naif_id=observer_naif_id,
        )

        cos_u = math.cos(u)
        sin_u = math.sin(u)
        cos_2u = math.cos(2.0 * u)
        sin_2u = math.sin(2.0 * u)
        cos_3u = math.cos(3.0 * u)
        sin_3u = math.sin(3.0 * u)
        alpha_u = (
            alpha_0
            + a_c1 * cos_u + a_s1 * sin_u
            + a_c2 * cos_2u + a_s2 * sin_2u
            + a_c3 * cos_3u + a_s3 * sin_3u
        )
        delta_u = (
            delta_0
            + d_c1 * cos_u + d_s1 * sin_u
            + d_c2 * cos_2u + d_s2 * sin_2u
            + d_c3 * cos_3u + d_s3 * sin_3u
        )
        sin_alpha = math.sin(alpha_u)

        if sin_alpha == 0.0:
            return s_hat

        cos_alpha = math.cos(alpha_u)
        cos_phase = math.cos(u + delta_u)
        sin_phase = math.sin(u + delta_u)
        offset_dir = cos_phase * e_A + sin_phase * e_B

        n_hat = cos_alpha * s_hat + sin_alpha * offset_dir
        return n_hat / float(np.linalg.norm(n_hat))

    return _n_hat


def sun_offset_harmonic_full_mode3_from_state(
    alpha_0_rad: float,
    *,
    alpha_c1_rad: float = 0.0,
    alpha_s1_rad: float = 0.0,
    alpha_c2_rad: float = 0.0,
    alpha_s2_rad: float = 0.0,
    alpha_c3_rad: float = 0.0,
    alpha_s3_rad: float = 0.0,
    delta_0_rad: float = 0.0,
    delta_c1_rad: float = 0.0,
    delta_s1_rad: float = 0.0,
    delta_c2_rad: float = 0.0,
    delta_s2_rad: float = 0.0,
    delta_c3_rad: float = 0.0,
    delta_s3_rad: float = 0.0,
    initial_state_km_kmps: np.ndarray,
    observer_naif_id: int = MARS_NAIF_ID,
    delta_amp_max_rad: float = DELTA_AMP_MAX_DEFAULT_RAD,
) -> AttitudeCallable:
    """Convenience wrapper for ``sun_offset_harmonic_full_mode3``.

    Computes the frozen orbit-plane basis from an initial state and
    forwards to ``sun_offset_harmonic_full_mode3``. Same usage pattern
    as ``sun_offset_harmonic_full_mode2_from_state``.
    """
    e_ref, _e_ortho, n_orb = _orbit_plane_basis_j2000(initial_state_km_kmps)
    return sun_offset_harmonic_full_mode3(
        alpha_0_rad,
        alpha_c1_rad=alpha_c1_rad, alpha_s1_rad=alpha_s1_rad,
        alpha_c2_rad=alpha_c2_rad, alpha_s2_rad=alpha_s2_rad,
        alpha_c3_rad=alpha_c3_rad, alpha_s3_rad=alpha_s3_rad,
        delta_0_rad=delta_0_rad,
        delta_c1_rad=delta_c1_rad, delta_s1_rad=delta_s1_rad,
        delta_c2_rad=delta_c2_rad, delta_s2_rad=delta_s2_rad,
        delta_c3_rad=delta_c3_rad, delta_s3_rad=delta_s3_rad,
        orbit_plane_normal_j2000=n_orb,
        orbit_ref_direction_j2000=e_ref,
        observer_naif_id=observer_naif_id,
        delta_amp_max_rad=delta_amp_max_rad,
    )
