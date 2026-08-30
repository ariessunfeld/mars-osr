"""Mars gravity field: MRO120F (Konopliv 2020) loader, zonal and full
spherical-harmonic acceleration.

Spherical-harmonic gravity model from JPL's MRO120F product:

    Konopliv, A. S. et al. (2020) "Detection of the Chandler wobble of Mars
    from orbiting spacecraft", Geophys. Res. Lett. 47, e2020GL090568.
    Data archive at PDS Geosciences:
        https://pds-geosciences.wustl.edu/mro/mro-m-rss-5-sdp-v1/mrors_1xxx/data/shadr/jgmro_120f_sha.tab

The archived file is an ASCII SHADR table with a one-line header and 14640
(degree, order, C_bar, S_bar, sigma_C, sigma_S) rows. Header values:
    reference radius R_ref = 3396.0 km
    mu (Mars system = Mars + Phobos + Deimos) = 42828.3756640 km^3/s^2
    max degree/order = 120
    coefficients are FULLY NORMALIZED (Kaula 4-pi convention)

The Mars-system mu in the SHADR header decomposes via the PDS label
``jgmro_120f_sha.lbl`` (lines 41-45) into the Konopliv-2020 self-
consistent quartet:

    Mars-system GM (lumped)  = 42828.3756640    km^3/s^2  (header value)
    GM of Mars (planet only) = 42828.3748574    km^3/s^2
    GM of Phobos             =  7.10e-4 +/- 5e-6   km^3/s^2
    GM of Deimos             =  9.68e-5 +/- 1.30e-5 km^3/s^2

with ``Mars + Phobos + Deimos`` reproducing the header to ~1e-7. These
three values are pinned in ``reflectors.mars_constants`` and used by
``reflectors.third_body`` (moon factories) and ``reflectors.dynamics``
(central-mu decoupling when the moons are separate third bodies). Use
those constants rather than re-typing the literals here.

This module:
    - downloads + caches the SHADR file under data/gravity/ on demand;
    - parses into a ``MarsGravityModel`` dataclass that holds mu, R_ref, the
      normalized C_bar / S_bar arrays, and matching UNNORMALIZED C / S arrays
      precomputed once at load time;
    - exposes a cached ``mars_gravity_model(max_degree)`` factory;
    - provides ``zonal_coefficients(model, max_degree)`` (unnormalized J_n)
      and ``zonal_acceleration_body_fixed`` / ``_inertial`` for the zonal
      (m=0) slice using the scalar Legendre polynomial recurrence. These
      remain in the codebase as an independent cross-check implementation.
    - provides ``mars_gravity_acceleration_body_fixed`` / ``_inertial`` for
      the full (zonals + tesserals + sectorals) gravity acceleration via the
      complex-V_{nm} recurrence of Cunningham (1970), *Celest. Mech.*
      2:207-216 -- the primary numerical path once tesseral physics is
      needed. The default backend is an exact-operation-order Numba
      translation with ``fastmath`` disabled; the Python reference recurrence
      remains selectable as an independent oracle. Pole-safe by construction
      (no 1/cos phi singularity).
    - provides ``j2_closed_form_body_fixed`` and its inertial wrapper as a
      hand-coded cross-check of the general recurrence at n=2.

Conventions, spelled out to avoid sign and normalization errors:

    Zonal slice (Curtis convention, physical sign for attractive field):
        U(r, phi) = -(mu/r) * [1 - sum_{n>=2} J_n (R_ref/r)^n P_n(sin phi)]
    with J_n = -C_{n,0} (unnormalized). For Mars J_2 is positive
    (~1.96e-3), which represents mass concentration in the equatorial bulge.

    Zonal acceleration (a = -grad U) in body-fixed Cartesian (x, y, z with z
    along the spin axis, r = sqrt(x^2+y^2+z^2), u = sin phi = z/r):

        a_x^{pert} = (x mu / r^3) * sum_n J_n (R_ref/r)^n * [(n+1) P_n(u) + u P_n'(u)]
        a_y^{pert} = (y mu / r^3) * sum_n J_n (R_ref/r)^n * [(n+1) P_n(u) + u P_n'(u)]
        a_z^{pert} = (mu / r^2)   * sum_n J_n (R_ref/r)^n * [(n+1) u P_n(u) - (1-u^2) P_n'(u)]

    Verification: at (r, 0, 0) (equatorial, +x), J_2 bracket = 3 * P_2(0) + 0
    = 3 * (-1/2) = -3/2, so a_x = -(3/2) * J_2 * mu * R_ref^2 / r^4 (inward
    pull from the bulge, sign matches Curtis eq. 12.30 when rearranged).

    Full-harmonic convention (Cunningham 1970, Eq 1-3): MINUS the potential
    energy per unit mass, divided by mu, is
        V(r, phi, lambda) = sum_{n,m} (R_ref)^n (C_{nm} cos m lambda + S_{nm} sin m lambda)
                                      * P_n^m(sin phi) / r^{n+1}
    where C_{nm}, S_{nm} are UNNORMALIZED and (C_{0,0}, C_{1,0}, C_{1,1},
    S_{1,1}) = (1, 0, 0, 0) for a body with its origin at the centre of
    mass. Zonal relation: J_n = -C_{n,0}. The acceleration on a particle in
    body-fixed coordinates is ``a = mu * grad V``. Note the SIGN: Cunningham
    defines V as minus the conventional potential energy per unit mass, so
    ``+mu grad V`` gives the attractive acceleration (consistent with
    ``a = -mu r/|r|^3`` at n=m=0 with C_{0,0}=1).

    Zonal-slice consistency: the Cunningham formulation at m=0 reduces to
    the scalar Legendre expression above via the Legendre identity
    ``P_{n+1}'(u) = (n+1) P_n(u) + u P_n'(u)``. This is verified to machine
    precision in tests/test_gravity_harmonics.py.

The IAU_MARS body-fixed frame from SPICE has z along Mars's spin pole by
construction, which is what the spherical-harmonic potential requires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from math import lgamma
from pathlib import Path

import numba
from numba import njit
import numpy as np
import requests

from reflectors.ephemeris import frame_rotation
from reflectors.kernels import DATA_DIR

logger = logging.getLogger(__name__)

CUNNINGHAM_BACKEND_NUMBA = "numba"
CUNNINGHAM_BACKEND_PYTHON = "python"
DEFAULT_CUNNINGHAM_BACKEND = CUNNINGHAM_BACKEND_NUMBA
_CUNNINGHAM_BACKENDS = frozenset(
    (CUNNINGHAM_BACKEND_NUMBA, CUNNINGHAM_BACKEND_PYTHON)
)

GRAVITY_DIR = DATA_DIR / "gravity"
MRO120F_URL = (
    "https://pds-geosciences.wustl.edu/mro/mro-m-rss-5-sdp-v1/mrors_1xxx/"
    "data/shadr/jgmro_120f_sha.tab"
)
MRO120F_FILE = GRAVITY_DIR / "jgmro_120f_sha.tab"

# Expected header values from the MRO120F SHADR file. These are NOT a
# parallel source of truth for downstream callers -- the authoritative R_ref
# and mu come from the parsed ``MarsGravityModel`` instance at load time.
# Rather, they are a sanity pin: ``_parse_mro120f`` asserts the header it
# reads matches these to catch a corrupted download, an upstream re-release
# under the same filename, or a parser regression. The reference radius is
# 3396.0 km (a property of the spherical-harmonic fit, NOT the IAU mean
# radius 3396.19 km from pck00011.tpc -- the 190 m offset matters at the
# 1e-4 level of J_2 contributions). mu is the Mars-SYSTEM GM
# (Mars + Phobos + Deimos lumped), distinct from BODY499_GM in gm_de440.tpc
# which refers to Mars-planet-alone (DE440 fit). The Konopliv-2020
# decomposition of this header into Mars-planet / Phobos / Deimos GMs lives
# in ``reflectors.mars_constants``; that is the place to pull the Mars-alone
# value (42828.3748574 km^3/s^2) when calling ``propagate`` against MRO120F
# harmonics with Phobos/Deimos as separate third bodies. All tolerances
# here are tight (absolute 1e-6 km and 1e-4 km^3/s^2) so a genuine
# file-content change trips the assertion immediately rather than
# propagating.
_MRO120F_EXPECTED_R_REF_KM = 3396.0
_MRO120F_EXPECTED_MU_KM3_S2 = 42828.3756639565  # Mars system (Mars+Phobos+Deimos)
_MRO120F_R_REF_TOL_KM = 1e-6
_MRO120F_MU_TOL_KM3_S2 = 1e-4


# ---------------------------------------------------------------------------
# Model data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarsGravityModel:
    """Truncated MRO120F gravity model.

    ``C_bar`` and ``S_bar`` are shape (max_degree+1, max_degree+1) arrays of
    FULLY NORMALIZED coefficients (Kaula 4-pi convention) as read from the
    SHADR file. ``C`` and ``S`` are the matching UNNORMALIZED arrays used by
    the Cunningham-1970 V_{nm} recurrences; they are precomputed once at
    load time via ``normalized_to_unnormalized``. Entries with m > n are
    zero in all four arrays.

    For the zonal-only slice, ``zonal_coefficients(model, max_degree)``
    returns J_n = -C_{n,0} directly.
    """

    max_degree: int
    ref_radius_km: float
    mu_km3_s2: float
    C_bar: np.ndarray
    S_bar: np.ndarray
    C: np.ndarray
    S: np.ndarray
    source: str = "MRO120F"
    # Uncertainty arrays; same shape as C_bar/S_bar. Kept for completeness
    # (retained for uncertainty propagation).
    sigma_C_bar: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_S_bar: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------


def download_mro120f(force: bool = False) -> Path:
    """Download the MRO120F SHADR table to ``data/gravity/`` if not cached.

    The file is small (~900 KB), publicly hosted on the PDS Geosciences node,
    and stored in the unversioned ``data/`` directory.
    """
    if MRO120F_FILE.exists() and not force:
        logger.debug("MRO120F already on disk at %s", MRO120F_FILE)
        return MRO120F_FILE
    GRAVITY_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("downloading MRO120F SHADR from %s", MRO120F_URL)
    with requests.get(MRO120F_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with MRO120F_FILE.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    logger.info("wrote %s (%d bytes)", MRO120F_FILE, MRO120F_FILE.stat().st_size)
    return MRO120F_FILE


def _parse_mro120f(path: Path, max_degree: int) -> MarsGravityModel:
    """Parse the SHADR ASCII table into a ``MarsGravityModel`` truncated to
    ``max_degree``.

    Header line layout (comma-delimited):
        R_ref, GM, ?, n_max, m_max, norm_flag, ?, ?
    Data lines: degree, order, C_bar, S_bar, sigma_C_bar, sigma_S_bar.
    """
    with path.open() as f:
        header = [tok.strip() for tok in f.readline().split(",")]
        R_ref = float(header[0])
        mu = float(header[1])
        nmax_file = int(header[3])
        norm_flag = int(header[5])
        if norm_flag != 1:
            raise ValueError(
                f"MRO120F header reports normalization flag {norm_flag}; "
                "this parser expects fully-normalized coefficients (flag 1)"
            )
        if max_degree > nmax_file:
            raise ValueError(
                f"requested max_degree={max_degree} exceeds file nmax={nmax_file}"
            )
        # Header sanity pin -- see the module-level expected-value block for
        # motivation. Raise explicitly if the downloaded file has drifted.
        if abs(R_ref - _MRO120F_EXPECTED_R_REF_KM) > _MRO120F_R_REF_TOL_KM:
            raise ValueError(
                f"MRO120F header R_ref={R_ref:.6f} km differs from expected "
                f"{_MRO120F_EXPECTED_R_REF_KM:.6f} km by more than "
                f"{_MRO120F_R_REF_TOL_KM} km -- refusing to load a drifted file"
            )
        if abs(mu - _MRO120F_EXPECTED_MU_KM3_S2) > _MRO120F_MU_TOL_KM3_S2:
            raise ValueError(
                f"MRO120F header mu={mu:.10f} km^3/s^2 differs from expected "
                f"{_MRO120F_EXPECTED_MU_KM3_S2:.10f} by more than "
                f"{_MRO120F_MU_TOL_KM3_S2} km^3/s^2 -- refusing to load a drifted file"
            )

        size = max_degree + 1
        C_bar = np.zeros((size, size))
        S_bar = np.zeros((size, size))
        sigma_C_bar = np.zeros((size, size))
        sigma_S_bar = np.zeros((size, size))

        for line in f:
            parts = [p.strip() for p in line.split(",")]
            n = int(parts[0])
            m = int(parts[1])
            if n > max_degree:
                # Remaining rows are degree-sorted, so parsing can stop early.
                break
            if m > max_degree:  # shouldn't occur if n <= max_degree
                continue
            C_bar[n, m] = float(parts[2])
            S_bar[n, m] = float(parts[3])
            sigma_C_bar[n, m] = float(parts[4])
            sigma_S_bar[n, m] = float(parts[5])

    # The Cunningham-1970 recurrences operate on unnormalized (C, S);
    # precompute once here so the hot path does not re-convert per call.
    C, S = normalized_to_unnormalized(C_bar, S_bar)
    # The SHADR file omits the (0, 0) row (and ships (1, 0), (1, 1) as
    # explicit zeros). Cunningham Eq 3 assumes ``C_{0,0} = 1`` by convention
    # so that the (n=0, m=0) harmonic equals the central 1/r potential. This
    # convention is enforced here so
    # ``mars_gravity_acceleration_body_fixed(include_central=True)`` reproduces
    # pure two-body instead of returning
    # zero. (C_bar[0, 0] stays at 0 because that is what the file ships;
    # the unnormalized C[0, 0] = 1 is the Cunningham convention.)
    C[0, 0] = 1.0

    logger.info(
        "parsed MRO120F to degree %d: R_ref=%.3f km, mu=%.8e km^3/s^2",
        max_degree, R_ref, mu,
    )
    return MarsGravityModel(
        max_degree=max_degree,
        ref_radius_km=R_ref,
        mu_km3_s2=mu,
        C_bar=C_bar,
        S_bar=S_bar,
        C=C,
        S=S,
        sigma_C_bar=sigma_C_bar,
        sigma_S_bar=sigma_S_bar,
    )


@lru_cache(maxsize=4)
def mars_gravity_model(max_degree: int = 10) -> MarsGravityModel:
    """Load (and cache) MRO120F truncated to ``max_degree``.

    Downloads the SHADR file on first use; repeated calls reuse the cache.
    Default ``max_degree=10`` is enough for all zonal work up to J10 with
    headroom; request higher degrees when tesseral acceleration lands.
    """
    if max_degree < 2:
        raise ValueError("max_degree must be >= 2 (Mars has no dipole term)")
    path = download_mro120f()
    return _parse_mro120f(path, max_degree)


# ---------------------------------------------------------------------------
# Normalization conversion
# ---------------------------------------------------------------------------


def unnormalized_zonal_from_normalized(c_bar_n0: float, n: int) -> float:
    """Convert a fully-normalized zonal C_bar_{n,0} to unnormalized C_{n,0}.

    Kaula / geodesy convention: C_{n,0} = C_bar_{n,0} * sqrt(2n+1).
    Spot-check: C_bar_{2,0} = -8.750e-4 for Mars (MRO120F) -> C_{2,0} =
    -1.957e-3 -> J_2 = 1.957e-3, matching published tables.
    """
    return float(c_bar_n0) * float(np.sqrt(2 * n + 1))


def normalization_factor(n: int, m: int) -> float:
    """Fully-normalized -> unnormalized conversion factor N_{n,m}.

    Relation: C_{n,m} = N_{n,m} * C_bar_{n,m} and equivalently for S.
    Following the Kaula / 4-pi convention,

        N_{n,0}    = sqrt(2n+1)
        N_{n,m>0}  = sqrt(2 (2n+1) (n-m)! / (n+m)!)

    The factorial ratio is evaluated via lgamma to stay well-behaved to
    very high degree (for n<=120 this is unnecessary in double precision,
    but costs nothing and keeps the numerical path robust).

    Reference: Heiskanen & Moritz (1967), *Physical Geodesy*, §1-14;
    reproduced in Montenbruck & Gill (2000) §3.2.3 Eq. 3.13.
    """
    if m < 0 or n < 0 or m > n:
        raise ValueError(f"invalid (n, m) = ({n}, {m}) -- need 0 <= m <= n")
    if m == 0:
        return float(np.sqrt(2 * n + 1))
    log_fac_ratio = lgamma(n - m + 1) - lgamma(n + m + 1)
    # Apply the square root in log space.  Evaluating exp(log_fac_ratio)
    # first underflows for high-order MRO120F terms (for example n=m=120),
    # even though the final square root remains representable in float64.
    return float(
        np.sqrt(2.0 * (2 * n + 1)) * np.exp(0.5 * log_fac_ratio)
    )


def normalized_to_unnormalized(
    C_bar: np.ndarray, S_bar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert square (C_bar, S_bar) coefficient arrays to unnormalized.

    Applies ``C_{n,m} = N_{n,m} C_bar_{n,m}`` entry-by-entry for n = 0..N and
    m = 0..n; upper-triangular (m > n) entries are left at zero. Shape-
    preserving.
    """
    if C_bar.shape != S_bar.shape or C_bar.ndim != 2 or C_bar.shape[0] != C_bar.shape[1]:
        raise ValueError("C_bar and S_bar must be square matrices of the same shape")
    N = C_bar.shape[0] - 1
    C = np.zeros_like(C_bar)
    S = np.zeros_like(S_bar)
    for n in range(N + 1):
        for m in range(n + 1):
            factor = normalization_factor(n, m)
            C[n, m] = C_bar[n, m] * factor
            S[n, m] = S_bar[n, m] * factor
    return C, S


def zonal_coefficients(model: MarsGravityModel, max_degree: int) -> dict[int, float]:
    """J_n (unnormalized) for n = 2..max_degree.

    J_n = -C_{n,0} = -C_bar_{n,0} * sqrt(2n+1). Returns a dict for explicit
    callability.
    """
    if max_degree > model.max_degree:
        raise ValueError(
            f"requested max_degree={max_degree} exceeds loaded model nmax={model.max_degree}"
        )
    out = {}
    for n in range(2, max_degree + 1):
        c = unnormalized_zonal_from_normalized(model.C_bar[n, 0], n)
        out[n] = -c  # J_n definition
    return out


# ---------------------------------------------------------------------------
# Legendre polynomials and their derivatives (unnormalized)
# ---------------------------------------------------------------------------


def _legendre_P_and_Pprime(u: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays [P_0, ..., P_{n_max}] and [P'_0, ..., P'_{n_max}] at u.

    Uses the standard stable recurrences:
        (n+1) P_{n+1}(u) = (2n+1) u P_n(u) - n P_{n-1}(u)
        n P_n'(u) = (2n-1) (P_{n-1}(u) + u P_{n-1}'(u)) - (n-1) P_{n-2}'(u)

    The derivative recurrence avoids the (1 - u^2) denominator that appears
    in the more commonly quoted identity n P_n' = (2n-1)(u P_{n-1})' - ... ;
    it is regular at u = +-1 (exactly-polar positions).
    """
    P = np.zeros(n_max + 1)
    dP = np.zeros(n_max + 1)
    P[0] = 1.0
    dP[0] = 0.0
    if n_max >= 1:
        P[1] = u
        dP[1] = 1.0
    for n in range(2, n_max + 1):
        P[n] = ((2 * n - 1) * u * P[n - 1] - (n - 1) * P[n - 2]) / n
        dP[n] = ((2 * n - 1) * (P[n - 1] + u * dP[n - 1]) - (n - 1) * dP[n - 2]) / n
    return P, dP


# ---------------------------------------------------------------------------
# Zonal acceleration in the body-fixed frame
# ---------------------------------------------------------------------------


def zonal_acceleration_body_fixed(
    r_bf_km: np.ndarray,
    mu_km3_s2: float,
    ref_radius_km: float,
    J_by_degree: dict[int, float],
) -> np.ndarray:
    """Acceleration from zonal harmonics, body-fixed Cartesian (km/s^2).

    Returns the PERTURBATION acceleration only -- the central two-body term
    is NOT included here (it is provided by
    ``dynamics.two_body_acceleration``). Sum them to get total gravity.

    Inputs:
        r_bf_km: shape (3,), position relative to Mars centre in the Mars
            body-fixed frame (IAU_MARS). The z-axis of this frame is Mars's
            spin axis; the zonal expansion is singular-free in this frame.
        mu_km3_s2: central-body GM. Must match the GM the J_n coefficients
            were fit against for self-consistency.
        ref_radius_km: reference radius R_ref the coefficients are defined
            against (3396.0 km for MRO120F).
        J_by_degree: mapping n -> J_n (unnormalized). Only the keys in this
            dict contribute; missing degrees have zero coefficient.

    Returns: shape (3,) acceleration vector in body-fixed Cartesian km/s^2.
    """
    r = np.asarray(r_bf_km, dtype=float)
    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise ValueError("zonal acceleration undefined at r = 0")
    x, y, z = r
    u = z / r_norm
    n_max = max(J_by_degree.keys())
    P, dP = _legendre_P_and_Pprime(u, n_max)
    R_over_r = ref_radius_km / r_norm

    # Two separate scalar sums: one for the (x, y) component multiplier, one
    # for the z component. Derivation in the module docstring.
    sum_xy = 0.0
    sum_z = 0.0
    R_over_r_n = 1.0  # will be (R/r)^n inside the loop, starting at n=2
    for n in range(2, n_max + 1):
        R_over_r_n_current = R_over_r ** n  # cheap for small n; explicit for clarity
        J = J_by_degree.get(n, 0.0)
        if J == 0.0:
            continue
        # Bracket for x, y: (n+1) P_n + u P_n'
        brk_xy = (n + 1) * P[n] + u * dP[n]
        sum_xy += J * R_over_r_n_current * brk_xy
        # Bracket for z: (n+1) u P_n - (1 - u^2) P_n'
        brk_z = (n + 1) * u * P[n] - (1.0 - u * u) * dP[n]
        sum_z += J * R_over_r_n_current * brk_z

    ax = x * mu_km3_s2 / (r_norm ** 3) * sum_xy
    ay = y * mu_km3_s2 / (r_norm ** 3) * sum_xy
    az = mu_km3_s2 / (r_norm ** 2) * sum_z
    return np.array([ax, ay, az], dtype=float)


def zonal_acceleration_inertial(
    r_j2000_km: np.ndarray,
    et: float,
    mu_km3_s2: float,
    ref_radius_km: float,
    J_by_degree: dict[int, float],
    *,
    body_frame: str = "IAU_MARS",
) -> np.ndarray:
    """Zonal acceleration in J2000 axes.

    Rotates the inertial position into the central body's body-fixed frame
    (``body_frame``; the spin-axis frame in which the zonal expansion is
    singular-free), computes the perturbation acceleration there, then rotates
    the acceleration back to J2000. The inverse rotation is just the transpose
    (rotations are orthogonal).

    ``body_frame`` defaults to ``"IAU_MARS"``; pass ``"IAU_EARTH"`` for
    Earth J2 in the Earth-escape model. The
    zonal recurrence itself is body-agnostic -- only the frame and the
    (mu, ref_radius, J_by_degree) inputs differ.
    """
    M_j2000_to_bf = frame_rotation("J2000", body_frame, et)
    r_bf = M_j2000_to_bf @ np.asarray(r_j2000_km, dtype=float)
    a_bf = zonal_acceleration_body_fixed(r_bf, mu_km3_s2, ref_radius_km, J_by_degree)
    return M_j2000_to_bf.T @ a_bf


# ---------------------------------------------------------------------------
# Independent J2 closed form (cross-check)
# ---------------------------------------------------------------------------


def j2_closed_form_body_fixed(
    r_bf_km: np.ndarray,
    mu_km3_s2: float,
    ref_radius_km: float,
    J2: float,
) -> np.ndarray:
    """Hand-coded J2 acceleration in body-fixed Cartesian (km/s^2).

    From Curtis (2014), "Orbital Mechanics for Engineering Students", eq.
    12.30 -- rearranged to match the sign convention and definitions used
    here. Intended as an independent cross-check for the general zonal
    recurrence at n=2; call sites should prefer ``zonal_acceleration_*``.
    """
    r = np.asarray(r_bf_km, dtype=float)
    r_norm = float(np.linalg.norm(r))
    x, y, z = r
    k = 1.5 * J2 * mu_km3_s2 * ref_radius_km ** 2 / (r_norm ** 5)
    factor_xy = 5.0 * (z / r_norm) ** 2 - 1.0
    factor_z = 5.0 * (z / r_norm) ** 2 - 3.0
    return np.array([k * x * factor_xy, k * y * factor_xy, k * z * factor_z])


def j2_closed_form_inertial(
    r_j2000_km: np.ndarray,
    et: float,
    mu_km3_s2: float,
    ref_radius_km: float,
    J2: float,
    *,
    body_frame: str = "IAU_MARS",
) -> np.ndarray:
    """J2 acceleration in J2000 axes via the closed-form body-fixed formula.

    ``body_frame`` defaults to ``"IAU_MARS"``; pass ``"IAU_EARTH"`` for the
    Earth J2 cross-check.
    """
    M = frame_rotation("J2000", body_frame, et)
    r_bf = M @ np.asarray(r_j2000_km, dtype=float)
    a_bf = j2_closed_form_body_fixed(r_bf, mu_km3_s2, ref_radius_km, J2)
    return M.T @ a_bf


# ---------------------------------------------------------------------------
# Full spherical-harmonic acceleration (Cunningham 1970)
# ---------------------------------------------------------------------------
#
# Reference: Cunningham, L. E. (1970) "On the computation of the spherical
# harmonic terms needed during the numerical integration of the orbital
# motion of an artificial satellite", Celestial Mechanics 2:207-216.
#
# The paper derives solid spherical harmonics
#     V_{n,m}(r) = P_n^m(sin phi) (cos m lambda + i sin m lambda) / r^{n+1}
# as complex-valued functions of body-fixed Cartesian coordinates, with
# simple Cartesian recurrences (Eqs 14, 16, 17) and derivative identities
# (Eqs 18-24, summarised pp. 214-215). The potential is
#     V(r) = Re sum_{n,m} R_eq^n (C_{nm} - i S_{nm}) V_{nm}(r)        (Cun. Eq 3)
# with C_{0,0} = 1, and acceleration = mu * grad V                    (Cun. Eq 29).
# Here (C_{nm}, S_{nm}) are UNNORMALIZED; the converter
# ``normalized_to_unnormalized`` produces them once per model load.
#
# This Cartesian, complex-V formulation is pole-safe (no 1/cos phi
# singularity) -- essential for sun-synchronous Mars orbits at i ~ 93 deg
# that fly directly over both poles.


def _cunningham_V(
    x: float, y: float, z: float, r2: float, n_max: int, m_max: int
) -> np.ndarray:
    """Cunningham solid spherical harmonics V_{n,m} on a (n_max+1, m_max+1) grid.

    Returns a complex128 ndarray ``V`` with
        V[n, m] = P_n^m(sin phi) (cos m lambda + i sin m lambda) / r^{n+1}
    for 0 <= m <= min(n, m_max) and 0 <= n <= n_max. Entries with m > n are
    left at zero. Built via Cunningham Eqs 14-17:

        V_{0,0} = 1/r                                                  (Eq 15)
        V_{n,n} = (2n-1) (x+iy)/r^2 V_{n-1,n-1}                        (Eq 14)
        V_{m+1,m} = (2m+1) z/r^2 V_{m,m}                               (Eq 17)
        V_{n,m} = [(2n-1) z V_{n-1,m} - (n+m-1) V_{n-2,m}] / [(n-m) r^2]
                                                                       (Eq 16, n >= m+2)

    Preconditions: ``n_max >= m_max >= 0`` and r2 > 0.
    """
    if n_max < m_max:
        raise ValueError(f"n_max={n_max} < m_max={m_max}")
    if n_max < 0 or m_max < 0:
        raise ValueError(f"n_max, m_max must be non-negative; got {n_max}, {m_max}")
    r = np.sqrt(r2)
    V = np.zeros((n_max + 1, m_max + 1), dtype=np.complex128)

    # Seed (Cun. Eq 15).
    V[0, 0] = 1.0 / r + 0.0j

    # Zonal chain (m = 0). V_{n,0} is real; complex storage is compatible.
    if n_max >= 1:
        # Eq 17 at n=1, m=0: V_{1,0} = (2*1 - 1) z/r^2 V_{0,0} = z V_{0,0} / r^2
        V[1, 0] = (z / r2) * V[0, 0]
    for n in range(2, n_max + 1):
        # Eq 16: V_{n,0} = [(2n-1) z V_{n-1,0} - (n-1) V_{n-2,0}] / (n r^2)
        V[n, 0] = ((2 * n - 1) * z * V[n - 1, 0] - (n - 1) * V[n - 2, 0]) / (n * r2)

    # For each m = 1..m_max: build diagonal, sub-diagonal, then vertical chain.
    xiy = x + 1j * y
    for m in range(1, m_max + 1):
        # Diagonal (Eq 14).
        V[m, m] = (2 * m - 1) * (xiy / r2) * V[m - 1, m - 1]
        # Sub-diagonal (Eq 17): V_{m+1, m} = (2(m+1)-1) z/r^2 V_{m, m}.
        if m + 1 <= n_max:
            V[m + 1, m] = (2 * (m + 1) - 1) * (z / r2) * V[m, m]
        # Vertical (Eq 16) for n >= m+2.
        for n in range(m + 2, n_max + 1):
            V[n, m] = (
                (2 * n - 1) * z * V[n - 1, m]
                - (n + m - 1) * V[n - 2, m]
            ) / ((n - m) * r2)

    return V


def _cunningham_W(
    x: float,
    y: float,
    z: float,
    r2: float,
    ref_radius: float,
    n_max: int,
    m_max: int,
) -> np.ndarray:
    """Scaled Cunningham harmonics ``W[n,m] = R_ref**n * V[n,m]``.

    This is Cunningham's Eqs. 14--17 after multiplying the degree-``n``
    recurrence by ``R_ref**n``.  The algebraic scaling avoids forming the
    individually extreme factors ``R_ref**n`` and ``V[n,m]``.  Those factors
    overflow and underflow, respectively, near the upper degrees of MRO120F,
    although their physically meaningful product remains representable.

    The recurrence is otherwise identical to :func:`_cunningham_V`:

    ``W[0,0] = 1/r``

    ``W[n,n] = (2n-1) R (x+iy)/r^2 W[n-1,n-1]``

    ``W[m+1,m] = (2m+1) R z/r^2 W[m,m]``

    ``W[n,m] = [(2n-1) R z W[n-1,m]
                 - (n+m-1) R^2 W[n-2,m]] / [(n-m) r^2]``.

    The literal unscaled recurrence remains available in
    :func:`_cunningham_V` as an independent oracle at degrees where both
    forms are numerically safe.
    """
    if n_max < m_max:
        raise ValueError(f"n_max={n_max} < m_max={m_max}")
    if n_max < 0 or m_max < 0:
        raise ValueError(
            f"n_max, m_max must be non-negative; got {n_max}, {m_max}"
        )
    if ref_radius <= 0.0:
        raise ValueError(f"ref_radius must be positive; got {ref_radius}")

    r = np.sqrt(r2)
    W = np.zeros((n_max + 1, m_max + 1), dtype=np.complex128)
    W[0, 0] = 1.0 / r + 0.0j

    radius_over_r2 = ref_radius / r2
    radius_squared_over_r2 = ref_radius * radius_over_r2

    if n_max >= 1:
        W[1, 0] = z * radius_over_r2 * W[0, 0]
    for n in range(2, n_max + 1):
        W[n, 0] = (
            (2 * n - 1) * z * radius_over_r2 * W[n - 1, 0]
            - (n - 1) * radius_squared_over_r2 * W[n - 2, 0]
        ) / n

    xiy = x + 1j * y
    for m in range(1, m_max + 1):
        W[m, m] = (
            (2 * m - 1) * xiy * radius_over_r2 * W[m - 1, m - 1]
        )
        if m + 1 <= n_max:
            W[m + 1, m] = (
                (2 * (m + 1) - 1)
                * z
                * radius_over_r2
                * W[m, m]
            )
        for n in range(m + 2, n_max + 1):
            W[n, m] = (
                (2 * n - 1) * z * radius_over_r2 * W[n - 1, m]
                - (n + m - 1)
                * radius_squared_over_r2
                * W[n - 2, m]
            ) / (n - m)

    return W


@njit(cache=True, fastmath=False)
def _cunningham_acceleration_body_fixed_numba(
    r_bf_km: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    mu_km3_s2: float,
    ref_radius_km: float,
    max_degree: int,
    max_order: int,
    include_central: bool,
) -> np.ndarray:
    """Compiled translation of the scaled Cunningham recurrence and force sum.

    This deliberately preserves the Python reference implementation's
    operation ordering and disables ``fastmath``.  Input validation stays in
    the public wrapper because Numba exceptions in the RHS hot path are both
    slower and less informative.
    """
    x = r_bf_km[0]
    y = r_bf_km[1]
    z = r_bf_km[2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)

    n_store = max_degree + 1
    m_store = min(max_order + 1, n_store)
    W = np.zeros((n_store + 1, m_store + 1), dtype=np.complex128)
    W[0, 0] = 1.0 / r + 0.0j

    radius_over_r2 = ref_radius_km / r2
    radius_squared_over_r2 = ref_radius_km * radius_over_r2
    if n_store >= 1:
        W[1, 0] = z * radius_over_r2 * W[0, 0]
    for n in range(2, n_store + 1):
        W[n, 0] = (
            (2 * n - 1) * z * radius_over_r2 * W[n - 1, 0]
            - (n - 1) * radius_squared_over_r2 * W[n - 2, 0]
        ) / n

    xiy = x + 1j * y
    for m in range(1, m_store + 1):
        W[m, m] = (
            (2 * m - 1) * xiy * radius_over_r2 * W[m - 1, m - 1]
        )
        if m + 1 <= n_store:
            W[m + 1, m] = (
                (2 * (m + 1) - 1)
                * z
                * radius_over_r2
                * W[m, m]
            )
        for n in range(m + 2, n_store + 1):
            W[n, m] = (
                (2 * n - 1) * z * radius_over_r2 * W[n - 1, m]
                - (n + m - 1)
                * radius_squared_over_r2
                * W[n - 2, m]
            ) / (n - m)

    inverse_ref_radius = 1.0 / ref_radius_km
    n_min = 0 if include_central else 2
    ax = 0.0
    ay = 0.0
    az = 0.0
    for n in range(n_min, max_degree + 1):
        m_upper = min(n, max_order)
        for m in range(m_upper + 1):
            Cnm = C[n, m]
            Snm = S[n, m]
            if Cnm == 0.0 and Snm == 0.0:
                continue
            if m == 0:
                Wp1_1 = W[n + 1, 1]
                Wp1_0 = W[n + 1, 0]
                dWdx = -Wp1_1.real * inverse_ref_radius
                dWdy = -Wp1_1.imag * inverse_ref_radius
                dWdz = -(n + 1) * Wp1_0.real * inverse_ref_radius
                ax += mu_km3_s2 * Cnm * dWdx
                ay += mu_km3_s2 * Cnm * dWdy
                az += mu_km3_s2 * Cnm * dWdz
            else:
                k = 0.5 * (n - m + 1) * (n - m + 2)
                Wp1 = W[n + 1, m + 1]
                Wm1 = W[n + 1, m - 1]
                W0 = W[n + 1, m]
                dWdx_re = (
                    -0.5 * Wp1.real + k * Wm1.real
                ) * inverse_ref_radius
                dWdx_im = (
                    -0.5 * Wp1.imag + k * Wm1.imag
                ) * inverse_ref_radius
                dWdy_re = (
                    -0.5 * Wp1.imag - k * Wm1.imag
                ) * inverse_ref_radius
                dWdy_im = (
                    +0.5 * Wp1.real + k * Wm1.real
                ) * inverse_ref_radius
                dWdz_re = (
                    -(n - m + 1) * W0.real * inverse_ref_radius
                )
                dWdz_im = (
                    -(n - m + 1) * W0.imag * inverse_ref_radius
                )
                ax += mu_km3_s2 * (
                    Cnm * dWdx_re + Snm * dWdx_im
                )
                ay += mu_km3_s2 * (
                    Cnm * dWdy_re + Snm * dWdy_im
                )
                az += mu_km3_s2 * (
                    Cnm * dWdz_re + Snm * dWdz_im
                )

    return np.array((ax, ay, az), dtype=np.float64)


def _validate_cunningham_request(
    r_bf_km: np.ndarray,
    model: MarsGravityModel,
    max_degree: int,
    max_order: int | None,
) -> tuple[np.ndarray, int]:
    """Validate and normalize one Cunningham force request."""
    resolved_order = max_degree if max_order is None else max_order
    if resolved_order < 0 or max_degree < 0:
        raise ValueError(
            f"max_degree and max_order must be non-negative; "
            f"got max_degree={max_degree}, max_order={resolved_order}"
        )
    if resolved_order > max_degree:
        raise ValueError(
            f"max_order ({resolved_order}) cannot exceed max_degree ({max_degree})"
        )
    if max_degree > model.max_degree:
        raise ValueError(
            f"requested max_degree={max_degree} exceeds loaded model "
            f"nmax={model.max_degree}"
        )

    r = np.asarray(r_bf_km, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"r_bf_km must be shape (3,); got {r.shape}")
    if float(np.dot(r, r)) == 0.0:
        raise ValueError("gravity acceleration undefined at r = 0")
    return r, int(resolved_order)


def _mars_gravity_acceleration_body_fixed_python(
    r_bf_km: np.ndarray,
    model: MarsGravityModel,
    max_degree: int,
    max_order: int | None = None,
    *,
    include_central: bool = False,
) -> np.ndarray:
    """Reference Python spherical-harmonic acceleration implementation.

    Parameters
    ----------
    r_bf_km
        Position relative to Mars centre in the IAU_MARS body-fixed frame,
        shape (3,), km.
    model
        ``MarsGravityModel`` carrying ``mu_km3_s2``, ``ref_radius_km``, and
        the unnormalized ``C`` / ``S`` coefficient arrays (precomputed at
        load time).
    max_degree
        Include harmonics through this n. Must satisfy
        ``2 <= max_degree <= model.max_degree`` when ``include_central``
        is False; relaxes to ``0 <= max_degree`` when True (though at
        max_degree==0 the recurrence still runs and reduces to the two-body
        term -- use ``reflectors.dynamics.two_body_acceleration`` for
        that case directly; it is cheaper).
    max_order
        Include harmonics through this m. Defaults to ``max_degree``
        (full triangle). Must satisfy ``0 <= max_order <= max_degree``.
        ``max_order = 0`` reduces to the zonal slice -- verified to
        machine precision against ``zonal_acceleration_body_fixed`` in
        the test suite.
    include_central
        If False (default) returns the PERTURBATION acceleration only --
        the (n=0, m=0) central term is skipped so callers can add the
        two-body acceleration separately (the standard split).
        If True, also adds -mu r/|r|^3 via the (0, 0) harmonic with
        C_{0,0} = 1 (unused for Mars where the origin is at the centre
        of mass, but useful for validation against ``-mu r/|r|^3``).

    Returns
    -------
    ndarray, shape (3,)
        Acceleration in body-fixed km/s^2.

    Notes
    -----
    The acceleration is ``a = mu * grad V`` where V is the Cunningham
    potential (his Eq 3, Eq 29); the SIGN is +, not -, because Cunningham
    defines V as minus the conventional potential energy per unit mass.
    At (n=0, m=0) with C_{0,0}=1 this produces ``-mu r/|r|^3``, matching
    ``reflectors.dynamics.two_body_acceleration``.

    Derivative formulas (Cunningham pp. 214-215), with
    k = (n - m + 1)(n - m + 2)/2:

        m = 0:  dV/dx = -Re(V_{n+1,1})
                dV/dy = -Im(V_{n+1,1})
                dV/dz = -(n+1) Re(V_{n+1,0})

        m > 0:  dV/dx = -V_{n+1,m+1}/2 + k V_{n+1,m-1}
                dV/dy = +i V_{n+1,m+1}/2 + i k V_{n+1,m-1}
                dV/dz = -(n-m+1) V_{n+1,m}

    Per-term contribution to acceleration component xi:
        a_xi^{(n,m)} = mu R_eq^n [ C_{nm} Re(dV_{nm}/dxi)
                                 + S_{nm} Im(dV_{nm}/dxi) ]
    which comes from expanding Re((C - iS)(dV/dxi)).

    Zonal-slice consistency (tested to machine precision in
    tests/test_gravity_harmonics.py): at max_order=0 this function agrees
    bit-for-bit-ish with ``zonal_acceleration_body_fixed``, because the
    Cunningham result for m=0 reduces to the scalar Legendre-polynomial
    formula via the identity ``P_{n+1}'(u) = (n+1) P_n(u) + u P_n'(u)``.
    """
    r, max_order = _validate_cunningham_request(
        r_bf_km, model, max_degree, max_order
    )
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    r2 = x * x + y * y + z * z

    # V index bounds. Derivative formulae reference V[n+1, m+1], V[n+1, m],
    # V[n+1, m-1] when contributing harmonic (n, m). Therefore V must be
    # populated up to n_store = max_degree + 1,
    # m_store = min(max_order + 1, n_store).
    n_store = max_degree + 1
    m_store = min(max_order + 1, n_store)
    R_eq = model.ref_radius_km
    W = _cunningham_W(x, y, z, r2, R_eq, n_store, m_store)
    C = model.C
    S = model.S
    mu = model.mu_km3_s2
    inverse_ref_radius = 1.0 / R_eq

    n_min = 0 if include_central else 2  # skip (0,0) central + (1,0/1,1) dipole for perturbation mode
    ax = ay = az = 0.0

    for n in range(n_min, max_degree + 1):
        m_upper = min(n, max_order)
        for m in range(0, m_upper + 1):
            Cnm = float(C[n, m])
            Snm = float(S[n, m])
            if Cnm == 0.0 and Snm == 0.0:
                continue
            if m == 0:
                # V_{n,0} is real; dV/dx and dV/dy project V_{n+1,1} onto
                # Cartesian axes (Cun. p. 214 m=0 case).
                Wp1_1 = W[n + 1, 1]
                Wp1_0 = W[n + 1, 0]
                # R^n dV_nm/dx is expressed through W_{n+1,*}/R.
                dWdx = -Wp1_1.real * inverse_ref_radius
                dWdy = -Wp1_1.imag * inverse_ref_radius
                dWdz = -(n + 1) * Wp1_0.real * inverse_ref_radius
                ax += mu * Cnm * dWdx
                ay += mu * Cnm * dWdy
                az += mu * Cnm * dWdz
            else:
                k = 0.5 * (n - m + 1) * (n - m + 2)
                Wp1 = W[n + 1, m + 1]  # W_{n+1, m+1}
                Wm1 = W[n + 1, m - 1]  # W_{n+1, m-1}
                W0 = W[n + 1, m]       # W_{n+1, m}
                # dV/dx = -V_{n+1,m+1}/2 + k V_{n+1,m-1}
                dWdx_re = (-0.5 * Wp1.real + k * Wm1.real) * inverse_ref_radius
                dWdx_im = (-0.5 * Wp1.imag + k * Wm1.imag) * inverse_ref_radius
                # dV/dy = i V_{n+1,m+1}/2 + i k V_{n+1,m-1}
                #   Re(i w) = -Im(w), Im(i w) = Re(w)
                dWdy_re = (-0.5 * Wp1.imag - k * Wm1.imag) * inverse_ref_radius
                dWdy_im = (+0.5 * Wp1.real + k * Wm1.real) * inverse_ref_radius
                # dV/dz = -(n-m+1) V_{n+1, m}
                dWdz_re = -(n - m + 1) * W0.real * inverse_ref_radius
                dWdz_im = -(n - m + 1) * W0.imag * inverse_ref_radius
                # Re((C - iS) * dV) = C * Re(dV) + S * Im(dV)
                ax += mu * (Cnm * dWdx_re + Snm * dWdx_im)
                ay += mu * (Cnm * dWdy_re + Snm * dWdy_im)
                az += mu * (Cnm * dWdz_re + Snm * dWdz_im)

    return np.array([ax, ay, az], dtype=float)


def cunningham_backend_metadata(backend: str) -> dict[str, object]:
    """Reproducibility metadata for a selected Cunningham implementation."""
    if backend not in _CUNNINGHAM_BACKENDS:
        raise ValueError(
            f"gravity_backend must be one of {sorted(_CUNNINGHAM_BACKENDS)}; "
            f"got {backend!r}"
        )
    if backend == CUNNINGHAM_BACKEND_NUMBA:
        return {
            "gravity_backend": backend,
            "gravity_backend_version": numba.__version__,
            "gravity_backend_fastmath": False,
        }
    return {
        "gravity_backend": backend,
        "gravity_backend_version": "scaled_cunningham_python_reference",
        "gravity_backend_fastmath": False,
    }


def mars_gravity_acceleration_body_fixed(
    r_bf_km: np.ndarray,
    model: MarsGravityModel,
    max_degree: int,
    max_order: int | None = None,
    *,
    include_central: bool = False,
    backend: str = DEFAULT_CUNNINGHAM_BACKEND,
) -> np.ndarray:
    """Spherical-harmonic gravitational acceleration (body-fixed, km/s^2).

    ``backend="numba"`` is the default. It is a compiled,
    ``fastmath=False`` translation of the scaled Cunningham recurrence.
    ``backend="python"`` provides a readable,
    independent regression oracle.  Both consume the same unnormalized
    MRO120F coefficients and apply the same operation ordering.

    All physical conventions and parameter definitions are documented on
    :func:`_mars_gravity_acceleration_body_fixed_python` above.
    """
    cunningham_backend_metadata(backend)
    if backend == CUNNINGHAM_BACKEND_PYTHON:
        return _mars_gravity_acceleration_body_fixed_python(
            r_bf_km,
            model,
            max_degree,
            max_order,
            include_central=include_central,
        )

    r, resolved_order = _validate_cunningham_request(
        r_bf_km, model, max_degree, max_order
    )
    return _cunningham_acceleration_body_fixed_numba(
        r,
        model.C,
        model.S,
        model.mu_km3_s2,
        model.ref_radius_km,
        int(max_degree),
        resolved_order,
        bool(include_central),
    )


def warm_cunningham_backend(
    model: MarsGravityModel,
    max_degree: int,
    max_order: int | None = None,
    *,
    backend: str = DEFAULT_CUNNINGHAM_BACKEND,
) -> None:
    """Compile/initialize a backend before integration or worker forking.

    The Python oracle needs no warmup.  For Numba, one representative call
    compiles the only array/scalar signature used by the propagator; degree,
    order, and coefficient-array shape remain runtime values thereafter.
    """
    cunningham_backend_metadata(backend)
    if backend == CUNNINGHAM_BACKEND_PYTHON:
        return
    test_position_km = np.array(
        [model.ref_radius_km + 400.0, 0.0, 0.0], dtype=float
    )
    mars_gravity_acceleration_body_fixed(
        test_position_km,
        model,
        max_degree,
        max_order,
        include_central=False,
        backend=backend,
    )


def mars_gravity_acceleration_inertial(
    r_j2000_km: np.ndarray,
    et: float,
    model: MarsGravityModel,
    max_degree: int,
    max_order: int | None = None,
    *,
    include_central: bool = False,
    backend: str = DEFAULT_CUNNINGHAM_BACKEND,
) -> np.ndarray:
    """Spherical-harmonic gravity acceleration in J2000 axes (km/s^2).

    Rotates the inertial position into IAU_MARS via ``spice.pxform``, calls
    ``mars_gravity_acceleration_body_fixed``, then rotates the acceleration
    back to J2000. The inverse rotation is the transpose since SPICE
    rotations are orthogonal.
    """
    M_j2000_to_bf = frame_rotation("J2000", "IAU_MARS", et)
    r_bf = M_j2000_to_bf @ np.asarray(r_j2000_km, dtype=float)
    a_bf = mars_gravity_acceleration_body_fixed(
        r_bf,
        model,
        max_degree,
        max_order,
        include_central=include_central,
        backend=backend,
    )
    return M_j2000_to_bf.T @ a_bf
