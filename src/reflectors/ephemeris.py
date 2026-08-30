"""Body state vectors from SPICE at arbitrary UTC epochs.

All functions assume the appropriate kernels have been loaded (see
``reflectors.kernels.load_kernels``). State vectors are returned as
``numpy.ndarray`` of shape ``(6,)`` in km and km/s by default, matching
SPICE's ``spkezr`` convention.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional, Union

import numpy as np
import spiceypy as spice

from .mars_constants import MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM


logger = logging.getLogger(__name__)

EpochLike = Union[str, datetime, float]

SUN_NAIF_ID = 10

# Light speed and AU pulled once from SPICE's built-in constants.
C_KM_PER_S = spice.clight()
AU_KM = spice.convrt(1.0, "AU", "KM")


@dataclass
class EphemerisCacheStats:
    """Aggregate counters for exact per-evaluation SPICE sharing.

    These are software-performance diagnostics, not physical outputs. A caller
    may reuse one instance across many short-lived evaluation contexts (the
    propagator does this over every RHS call).
    """

    contexts: int = 0
    state_requests: int = 0
    state_cache_hits: int = 0
    state_spice_calls: int = 0
    rotation_requests: int = 0
    rotation_cache_hits: int = 0
    rotation_inverse_hits: int = 0
    rotation_spice_calls: int = 0

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "enabled": True,
            "contexts": self.contexts,
            "state_requests": self.state_requests,
            "state_cache_hits": self.state_cache_hits,
            "state_spice_calls": self.state_spice_calls,
            "rotation_requests": self.rotation_requests,
            "rotation_cache_hits": self.rotation_cache_hits,
            "rotation_inverse_hits": self.rotation_inverse_hits,
            "rotation_spice_calls": self.rotation_spice_calls,
        }


@dataclass
class _EphemerisEvaluationCache:
    """Values valid only inside one exact dynamics/geometry evaluation."""

    stats: EphemerisCacheStats
    states: dict[
        tuple[str, float, str, str, str], tuple[np.ndarray, float]
    ] = field(default_factory=dict)
    rotations: dict[tuple[str, str, float], np.ndarray] = field(
        default_factory=dict
    )


_ACTIVE_EPHEMERIS_CACHE: ContextVar[
    Optional[_EphemerisEvaluationCache]
] = ContextVar("reflectors_ephemeris_evaluation_cache", default=None)


@contextmanager
def ephemeris_evaluation_context(
    stats: Optional[EphemerisCacheStats] = None,
) -> Iterator[EphemerisCacheStats]:
    """Share exact same-key SPICE results within one evaluation.

    Cache keys contain the unmodified IEEE-754 ET value. There is no time
    rounding or interpolation. The sole derived value is the transpose used
    for the reverse of an already-fetched rotation matrix at the same ET (the
    exact inverse by definition). Nested calls reuse the active context, which
    lets a delivery geometry helper participate in an enclosing dynamics RHS
    context without hiding already-fetched Sun or frame values.
    """
    active = _ACTIVE_EPHEMERIS_CACHE.get()
    if active is not None:
        yield active.stats
        return

    aggregate = EphemerisCacheStats() if stats is None else stats
    aggregate.contexts += 1
    cache = _EphemerisEvaluationCache(stats=aggregate)
    token = _ACTIVE_EPHEMERIS_CACHE.set(cache)
    try:
        yield aggregate
    finally:
        _ACTIVE_EPHEMERIS_CACHE.reset(token)


def spice_state_at_et(
    target: Union[int, str],
    et: float,
    frame: str = "J2000",
    abcorr: str = "NONE",
    observer: Union[int, str] = "SOLAR SYSTEM BARYCENTER",
) -> tuple[np.ndarray, float]:
    """Return a SPICE state, sharing only an identical active-context key."""
    key = (str(target), float(et), str(frame), str(abcorr), str(observer))
    active = _ACTIVE_EPHEMERIS_CACHE.get()
    if active is None:
        state, light_time = spice.spkezr(*key)
        return np.asarray(state, dtype=float), float(light_time)

    active.stats.state_requests += 1
    cached = active.states.get(key)
    if cached is not None:
        active.stats.state_cache_hits += 1
        return cached[0].copy(), cached[1]

    state, light_time = spice.spkezr(*key)
    stored = np.asarray(state, dtype=float).copy()
    light_time_f = float(light_time)
    active.states[key] = (stored, light_time_f)
    active.stats.state_spice_calls += 1
    return stored.copy(), light_time_f


def frame_rotation(from_frame: str, to_frame: str, et: float) -> np.ndarray:
    """Return ``pxform(from_frame, to_frame, et)`` with exact sharing.

    CSPICE ``pxform`` returns a rotation matrix, so the reverse transform at
    the same epoch is its transpose. Tests pin this identity across a sampled
    Mars year.
    """
    key = (str(from_frame), str(to_frame), float(et))
    active = _ACTIVE_EPHEMERIS_CACHE.get()
    if active is None:
        return np.asarray(spice.pxform(*key), dtype=float)

    active.stats.rotation_requests += 1
    cached = active.rotations.get(key)
    if cached is not None:
        active.stats.rotation_cache_hits += 1
        return cached.copy()

    inverse_key = (key[1], key[0], key[2])
    inverse = active.rotations.get(inverse_key)
    if inverse is not None:
        # A rotation's inverse is exactly its transpose. Store the derived key
        # so repeated same-direction requests take the ordinary-hit branch.
        stored = inverse.T.copy()
        active.rotations[key] = stored
        active.stats.rotation_cache_hits += 1
        active.stats.rotation_inverse_hits += 1
        return stored.copy()

    stored = np.asarray(spice.pxform(*key), dtype=float).copy()
    active.rotations[key] = stored
    active.stats.rotation_spice_calls += 1
    return stored.copy()


def utc_to_et(epoch: EpochLike) -> float:
    """Convert a UTC epoch to SPICE ephemeris time (TDB seconds past J2000).

    Accepted inputs:
        - ISO-8601 string (``"2026-04-20T12:00:00"``) or any other format
          that ``spice.str2et`` understands.
        - ``datetime.datetime`` (naive or timezone-aware; naive is treated as UTC).
        - ``float`` already in ET -- returned unchanged.
    """
    if isinstance(epoch, float):
        return epoch
    if isinstance(epoch, datetime):
        # Naive datetimes are interpreted as UTC (matching SPICE convention).
        as_utc = epoch if epoch.tzinfo is None else epoch.astimezone(timezone.utc).replace(tzinfo=None)
        return spice.str2et(as_utc.isoformat())
    if isinstance(epoch, str):
        return spice.str2et(epoch)
    raise TypeError(f"unsupported epoch type: {type(epoch).__name__}")


def body_state(
    target: str,
    epoch: EpochLike,
    observer: str = "SOLAR SYSTEM BARYCENTER",
    frame: str = "J2000",
    abcorr: str = "NONE",
) -> tuple[np.ndarray, float]:
    """Return ``(state_6vec, light_time_s)`` for ``target`` seen from ``observer``.

    ``state_6vec`` has shape (6,): [x, y, z, vx, vy, vz] in km and km/s, in the
    requested inertial frame. Default abcorr is ``NONE`` (geometric state), as
    required for dynamics; use ``LT`` or ``LT+S`` for observation modelling.
    """
    et = utc_to_et(epoch)
    return spice_state_at_et(target, et, frame, abcorr, observer)


def sun_state(
    epoch: EpochLike,
    observer: str = "SOLAR SYSTEM BARYCENTER",
    frame: str = "J2000",
    abcorr: str = "NONE",
) -> tuple[np.ndarray, float]:
    """State of the Sun as seen from ``observer`` (default: SSB)."""
    return body_state("SUN", epoch, observer=observer, frame=frame, abcorr=abcorr)


def mars_state(
    epoch: EpochLike,
    observer: str = "SOLAR SYSTEM BARYCENTER",
    frame: str = "J2000",
    abcorr: str = "NONE",
    center: str = "MARS BARYCENTER",
) -> tuple[np.ndarray, float]:
    """State of Mars as seen from ``observer`` (default: SSB).

    ``center`` picks between ``"MARS BARYCENTER"`` (NAIF ID 4, default -- good
    for heliocentric / solar-system-scale queries) and ``"MARS"`` (NAIF ID 499,
    the planet centre itself -- required for sail dynamics near Mars).
    """
    return body_state(center, epoch, observer=observer, frame=frame, abcorr=abcorr)


def sun_mars_distance_km(epoch: EpochLike) -> float:
    """Scalar Sun-Mars distance (km). Geometric, SSB-independent."""
    state, _ = body_state("MARS BARYCENTER", epoch, observer="SUN")
    return float(np.linalg.norm(state[:3]))


def corotating_heliocentric_triad(
    planet_state6: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed ORTHONORMAL triad co-rotating with a planet's heliocentric
    motion.

    Given a planet's heliocentric state ``[r, v]`` (km, km/s; e.g. from
    ``body_state(planet, et, observer="SUN")``), return unit vectors
    ``(p_hat, e_hat, n_hat)``:

    - ``p_hat = v / |v|``        -- PROGRADE (along the orbital velocity);
    - ``n_hat = unit(r x v)``    -- ORBIT NORMAL (specific angular momentum dir);
    - ``e_hat = n_hat x p_hat``  -- IN-PLANE, completing the right-handed
      orthonormal triad (points roughly Sun-ward / anti-radial, exactly
      perpendicular to both ``p_hat`` and ``n_hat``).

    This is the natural frame for expressing a planetocentric escape/capture
    HAND-OFF state (the hyperbolic excess velocity ``v_inf`` and the Hill-sphere
    exit position) as prograde / in-plane / out-of-plane components that vary
    SMOOTHLY with the planet's orbital phase -- so they can be interpolated
    across phase and rotated back to J2000 at any epoch (the escape-phasing
    characterization + the interplanetary-cruise moving target).

    Note ``p_hat`` (along ``v``) and the radial direction ``r/|r|`` are NOT
    perpendicular for an eccentric orbit (``v`` is perpendicular to ``r`` only at
    the apsides), so a ``{v_hat, r_hat, oop}`` triad is non-orthonormal and
    projecting onto it by dot products is not an orthogonal decomposition. The
    ``{p_hat, e_hat, n_hat}`` triad here is orthonormal by construction and is the
    single source for the decomposition.

    Parameters
    ----------
    planet_state6
        Heliocentric (or any inertial-frame) planet state ``[r, v]``, shape (6,).

    Returns
    -------
    (p_hat, e_hat, n_hat) : tuple of ndarray, each shape (3,)
        Right-handed orthonormal triad: prograde, in-plane, orbit-normal.
    """
    state = np.asarray(planet_state6, dtype=float)
    if state.shape != (6,):
        raise ValueError(
            f"planet_state6 must have shape (6,), got {state.shape}"
        )
    r = state[:3]
    v = state[3:6]
    v_norm = float(np.linalg.norm(v))
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if v_norm == 0.0 or h_norm == 0.0:
        raise ValueError(
            "planet_state6 must have non-zero velocity and angular momentum "
            "(r x v); cannot build the co-rotating triad from a degenerate state"
        )
    p_hat = v / v_norm
    n_hat = h / h_norm
    # n_hat is perpendicular to v (=> to p_hat) by construction, so e_hat is a
    # unit vector; renormalise only as a safeguard against round-off.
    e_hat = np.cross(n_hat, p_hat)
    e_hat = e_hat / float(np.linalg.norm(e_hat))
    return p_hat, e_hat, n_hat


def decompose_into_triad(
    vec: np.ndarray,
    planet_state6: np.ndarray,
) -> tuple[float, float, float]:
    """Components of ``vec`` on a planet's co-rotating heliocentric triad.

    Returns ``(prograde, in_plane, out_of_plane)`` -- the projections of ``vec``
    onto ``(p_hat, e_hat, n_hat)`` from :func:`corotating_heliocentric_triad`.
    Because the triad is orthonormal, ``prograde*p_hat + in_plane*e_hat +
    out_of_plane*n_hat`` reconstructs ``vec`` exactly.

    Parameters
    ----------
    vec
        A 3-vector in the same inertial frame as ``planet_state6`` (e.g. a
        Hill-exit ``v_inf`` or position offset, J2000).
    planet_state6
        The planet's heliocentric state ``[r, v]``, shape (6,).
    """
    w = np.asarray(vec, dtype=float)
    if w.shape != (3,):
        raise ValueError(f"vec must have shape (3,), got {w.shape}")
    p_hat, e_hat, n_hat = corotating_heliocentric_triad(planet_state6)
    return (
        float(np.dot(w, p_hat)),
        float(np.dot(w, e_hat)),
        float(np.dot(w, n_hat)),
    )


def sun_mars_distance_au(epoch: EpochLike) -> float:
    return sun_mars_distance_km(epoch) / AU_KM


# ===========================================================================
# Injectable Sun provider (for ablation studies)
# ===========================================================================
#
# The cruise/illumination pipeline fetches the Mars->Sun state from many
# independent SPICE call-sites (SRP force + shadow, the controller's s_hat, the
# Sun third-body, the bisector pointing law, the umbra/eclipse + window-finder
# checks). "No-obliquity" / "no-eccentricity" ablations must replace
# the real Sun with a synthetic one CONSISTENTLY across all of them. Every such
# call-site is routed through ``sun_state_j2000`` below, which honours a
# process-global override set by the ``sun_model`` context manager.
#
# With no override, calls pass directly through to ``spice.spkezr``.
#
# The override is a MODULE GLOBAL on purpose: the parallel-DE / parallel-FD
# pools fork to inherit the SPICE kernel pool, so a global set before the pool
# forks is inherited by every worker. Callers must enter ``sun_model(...)``
# before creating a pool.

_ACTIVE_SUN_MODEL: Optional[dict] = None
_VALID_SUN_MODES = ("real", "zero_obliquity", "circular")


def sun_state_j2000(et: float, observer: Union[int, str] = "MARS") -> np.ndarray:
    """Observer->Sun state (6,) [km, km/s] in J2000, honouring any active sun model.

    With no active override this returns exactly
    ``spice.spkezr("10", et, "J2000", "NONE", str(observer))[0]``. Under an
    active ``sun_model`` it returns a synthetic Mars->Sun state (see
    ``sun_model``).
    """
    state, _ = spice_state_at_et(
        SUN_NAIF_ID, float(et), "J2000", "NONE", observer
    )
    if _ACTIVE_SUN_MODEL is None:
        return state
    return _synthetic_sun_state(state, float(et))


def _synthetic_sun_state(real_state: np.ndarray, et: float) -> np.ndarray:
    """Map a real observer->Sun state to the active synthetic model."""
    mode = _ACTIVE_SUN_MODEL["mode"]
    r = real_state[:3]
    v = real_state[3:]
    dist = float(np.linalg.norm(r))

    if mode == "zero_obliquity":
        # Force the Sun into Mars' equatorial plane: zero the component along the
        # Mars spin pole, preserving heliocentric longitude AND distance. The
        # sub-solar latitude is then identically 0 (no seasonal declination
        # swing). The pole comes from IAU_MARS->J2000 (independent of the Sun
        # vector), so J2 / the critical inclination / sun-sync are untouched.
        R = frame_rotation("IAU_MARS", "J2000", et)
        pole = R[:, 2]  # Mars +Z (spin axis) expressed in J2000
        r_proj = r - float(np.dot(r, pole)) * pole
        n = float(np.linalg.norm(r_proj))
        if n < 1.0e-9:
            return real_state  # Sun over the pole (unphysical for Mars); no-op
        r_new = r_proj / n * dist
        v_new = v - float(np.dot(v, pole)) * pole  # unused downstream; kept consistent
        return np.concatenate([r_new, v_new])

    if mode == "circular":
        # Put Mars on a circle about the Sun: hold the Sun distance at the
        # orbital semi-major axis (removes the +/-9.3% e=0.0934 seasonal
        # SRP-magnitude swing), keep the real direction. The intra-sol angular-
        # rate variation is sub-dominant and is retained.
        r_new = r / dist * MARS_HELIOCENTRIC_SEMIMAJOR_AXIS_KM
        return np.concatenate([r_new, v])

    raise ValueError(f"unknown sun model mode: {mode!r}")


@contextmanager
def sun_model(mode: str, **params) -> Iterator[None]:
    """Activate a synthetic Sun for the duration of the context.

    ``mode`` is one of:
      - ``"real"``       -- no override; bit-exact real ephemeris (passthrough).
      - ``"zero_obliquity"`` -- Sun rides Mars' equatorial plane (sub-solar lat 0).
      - ``"circular"``   -- Mars on a circle (constant Sun distance = a_Mars).

    Sets a PROCESS GLOBAL -- enter this BEFORE any parallel pool forks so workers
    inherit it. Restores the prior model on exit.
    """
    if mode not in _VALID_SUN_MODES:
        raise ValueError(f"mode must be one of {_VALID_SUN_MODES}, got {mode!r}")
    global _ACTIVE_SUN_MODEL
    prev = _ACTIVE_SUN_MODEL
    _ACTIVE_SUN_MODEL = None if mode == "real" else {"mode": mode, **params}
    logger.info("sun_model activated: mode=%s params=%s", mode, params)
    try:
        yield
    finally:
        _ACTIVE_SUN_MODEL = prev


def active_sun_model() -> Optional[dict]:
    """Return the active synthetic sun-model dict (or None for the real Sun)."""
    return _ACTIVE_SUN_MODEL
