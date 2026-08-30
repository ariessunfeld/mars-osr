"""Direct dE/dt-maximizing escape steering.

For an SRP-propelled escape, the right local cost to maximize at every
instant is the rate of orbital-energy gain ``dE/dt = F . v / m``,
equivalently the velocity-aligned component of the SRP acceleration
``a_sail . v_hat``. Maximizing this over sail orientation directly tracks
the Edelbaum "continuous tangential thrust" ideal as closely as the SRP
cone and one-sided-sail geometry allow.

Unlike finite-target orbit transfer, escape has no finite semimajor-axis
target. The controller therefore maximizes the local energy-gain rate directly
and uses the acceleration- and rate-limited attitude tracker in
:mod:`reflectors.attitude_control`.

The sail orientation is parameterised by a single cone angle
``alpha`` in the plane of ``s_hat`` (sail-to-Sun) and the orbital velocity:

    n_hat(alpha) = cos(alpha) s_hat + sin(alpha) t_hat

with ``t_hat`` the unit vector orthogonal to ``s_hat`` in the
``(s_hat, v_hat)`` plane, oriented so that ``t_hat . v_hat >= 0``. The
merit function is

    f(alpha) = mcinnes_srp_acceleration(n_hat(alpha), s_hat, P, sail) . v_hat

(km/s^3; only ``argmax`` matters). The search is a coarse grid + golden-
section refinement -- the same 1-D pattern :mod:`reflectors.qlaw` uses for
its McInnes sail-pitch search. ``alpha`` is bounded to a cone half-angle
``max_cone_rad`` from ``s_hat`` so the sail stays well within the
sun-facing hemisphere. If ``f`` is non-positive at every ``alpha`` (no
orientation gives velocity-aligned thrust), the steering FEATHERS:
``n_star`` is set edge-on to ``s_hat`` so ``cos alpha = 0`` and the SRP
force is ~zero whenever the sail catches up to it.

Sign convention. Following :mod:`reflectors.srp`, ``n_hat`` is the SUN-FACING
sail normal: ``cos alpha = n_hat . s_hat > 0`` for an illuminated sail. The
McInnes (1999) closed-form acceleration with this convention is anti-sunward
to leading order; its dot product with velocity is the maximized quantity.

The controller can be supplied as the ``steering_fn`` argument to
:func:`reflectors.escape.propagate_escape`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from reflectors.atmosphere_constants import DEFAULT_DRAG_COEFFICIENT
from reflectors.gauss import (
    gauss_variational_rates,
    osculating_elements,
    rtn_basis,
)
from reflectors.srp import SolarSail, mcinnes_srp_acceleration

logger = logging.getLogger(__name__)

__all__ = [
    "BlendedParams",
    "BlendedSteering",
    "DEdotParams",
    "DEdotSteering",
    "DragMeritContext",
    "blended_steer",
    "dedot_steer",
]


# Tolerance below which ``v_hat`` projected into the plane perpendicular to
# ``s_hat`` has effectively zero length -- velocity is parallel (or anti-
# parallel) to the sun line. In that degenerate case the ``t_hat`` axis is
# undefined; an arbitrary perpendicular lets the search collapse
# to the trivial alpha=0 (sun-facing) optimum.
_V_PERP_TOL = 1.0e-12

# A tiny positive threshold for declaring the best achievable velocity-
# aligned thrust "essentially zero" -- below it the sail feathers rather than
# numerically thrust on round-off.
_FEATHER_MERIT_TOL = 0.0


@dataclass(frozen=True)
class DEdotParams:
    """Tuning for :func:`dedot_steer`.

    Attributes
    ----------
    max_cone_rad
        Cone half-angle (rad) bounding ``|alpha|`` -- the maximum tilt of
        the sail normal away from the Sun line. Default
        ``math.radians(80.0)``; smaller values protect against deep tilts
        that the dE/dt-optimum rarely requires (the unconstrained ideal-sail
        optimum is at ``|alpha| <= 35.26 deg`` for any geometry; see the
        ``arctan(1/sqrt(2))`` derivation in the module test). The bound is
        also a graceful guard against running the McInnes formula too close
        to the edge-on degeneracy ``cos alpha = 0``.
    grid_n
        Number of coarse-grid points across ``[-max_cone_rad, max_cone_rad]``
        (inclusive of both endpoints). Default 20 (same shape as
        :func:`reflectors.qlaw.steer`).
    golden_iters
        Number of golden-section iterations refining the coarse-grid winner.
        Default 30 -- gives ``~6 * (grid_step)`` reduction in the bracket
        width, i.e. sub-microradian convergence at the default grid.
    feather_threshold_km_s2
        If the maximum ``f(alpha) = (a_sail . v_hat)`` over the searched
        ``alpha`` range is less than or equal to this threshold (in km/s^2,
        same units as the SRP acceleration), the steering returns a
        feathered ``n_star``. Default 0.0 -- any positive velocity-aligned
        thrust is taken.
    rp_warn_km
        Periapsis warning radius (km). When set, after picking the
        dE/dt-optimal ``alpha_star`` the steering additionally checks the
        Gauss instantaneous periapsis rate ``drp/dt`` at that orientation;
        if ``rp < rp_warn_km`` AND ``drp/dt < 0``, the steering FEATHERS
        instead of thrusting. This is the conditional rp-rate gate
        so early-mission spiraling is unrestricted,
        but once periapsis has dropped to the warning level, any thrust
        arc that would shrink it further is refused. Default ``None`` --
        the guard is disabled and the controller is purely greedy in
        instantaneous dE/dt. The accompanying ``mu_km3_s2`` is needed to
        compute the Gauss rates.
    mu_km3_s2
        Central gravitational parameter (km^3/s^2) for the periapsis-rate
        guard. Used only when ``rp_warn_km`` is set. Default ``None`` ->
        resolved lazily to ``reflectors.dynamics.mars_gm_km3_per_s2()`` at
        first guarded call. For non-Mars problems pass it explicitly.
    """

    max_cone_rad: float = math.radians(80.0)
    grid_n: int = 20
    golden_iters: int = 30
    feather_threshold_km_s2: float = _FEATHER_MERIT_TOL
    rp_warn_km: Optional[float] = None
    mu_km3_s2: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.max_cone_rad < 0.5 * math.pi):
            raise ValueError(
                f"max_cone_rad must be in (0, pi/2), got {self.max_cone_rad}"
            )
        if self.grid_n < 2:
            raise ValueError(f"grid_n must be >= 2, got {self.grid_n}")
        if self.golden_iters < 0:
            raise ValueError(
                f"golden_iters must be >= 0, got {self.golden_iters}"
            )
        if self.rp_warn_km is not None and self.rp_warn_km <= 0.0:
            raise ValueError(
                f"rp_warn_km must be > 0 if set, got {self.rp_warn_km}"
            )
        if self.mu_km3_s2 is not None and self.mu_km3_s2 <= 0.0:
            raise ValueError(
                f"mu_km3_s2 must be > 0 if set, got {self.mu_km3_s2}"
            )


@dataclass(frozen=True)
class DragMeritContext:
    """Enable drag-aware scoring in :func:`dedot_steer`.

    Passing ``drag=DragMeritContext(...)`` switches the merit from the SRP-only
    ``a_SRP . v_hat`` to the NET energy rate ``(a_SRP + a_drag) . v_hat``, so the
    controller trades SRP energy gain against atmospheric-drag loss -- it tilts
    off broadside-to-velocity (and feathers when drag would dominate) instead of
    greedily chasing the SRP optimum into a high-drag orientation.

    ``drag=None`` selects the SRP-only controller and never enters the drag
    branch. This permits direct comparison of the same controller with
    drag-awareness on or off, while the drag
    DYNAMICS (the ``escape.propagate_escape`` drag hook) unchanged either way.

    Attributes
    ----------
    density_model
        A ``reflectors.atmosphere`` density model (``density_kg_m3(alt, r_hat,
        sun_hat)``).
    central_body
        ``reflectors.central_body.CentralBody`` (equatorial radius -> altitude;
        body frame -> atmosphere co-rotation).
    C_d
        Drag coefficient (default 2.2; ``atmosphere_constants``).

    Drag-aware steering additionally requires the absolute epoch ``et`` (for the
    co-rotating atmosphere); :func:`dedot_steer` raises if ``drag`` is set
    without ``et``.
    """

    density_model: object
    central_body: object
    C_d: float = DEFAULT_DRAG_COEFFICIENT


@dataclass(frozen=True)
class DEdotSteering:
    """Result of :func:`dedot_steer`.

    Attributes
    ----------
    n_star_j2000
        Desired sail normal (sun-facing convention, J2000 axes), shape (3,).
        For a thrusting solution this maximises ``a_sail . v_hat`` within
        the cone bound; for the feathered case it is edge-on to ``s_hat``
        (so the slewed-to-it sail produces ~zero SRP).
    alpha_rad
        Searched cone angle (rad) at the optimum. ``pi/2`` for the feathered
        sentinel.
    thrust
        ``True`` iff a positive velocity-aligned thrust was found within the
        cone. ``False`` -> feathered.
    dEdt_per_unit_mass_km2_s3
        The merit at the chosen orientation, ``a_sail . v_hat`` in km^2/s^3
        (the rate-of-energy-per-unit-mass). 0 for the feathered case.
    """

    n_star_j2000: np.ndarray
    alpha_rad: float
    thrust: bool
    dEdt_per_unit_mass_km2_s3: float


def _sun_velocity_basis(
    s_hat: np.ndarray, v_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(s_hat, t_hat)`` orthonormal in the ``(s_hat, v_hat)`` plane.

    ``t_hat`` is perpendicular to ``s_hat``, lies in the ``(s, v)`` plane,
    and has non-negative projection on ``v_hat``. In the degenerate case
    where ``v_hat`` is parallel or anti-parallel to ``s_hat``, ``t_hat`` is
    an arbitrary unit vector perpendicular to ``s_hat`` (the search then
    collapses to ``alpha=0``, the only orientation that thrusts).
    """
    v_perp = v_hat - float(np.dot(v_hat, s_hat)) * s_hat
    v_perp_norm = float(np.linalg.norm(v_perp))
    if v_perp_norm > _V_PERP_TOL:
        return s_hat, v_perp / v_perp_norm
    # Velocity parallel to sun line: pick any perpendicular axis.
    trial = np.array([1.0, 0.0, 0.0])
    if abs(s_hat[0]) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    t = trial - float(np.dot(trial, s_hat)) * s_hat
    return s_hat, t / float(np.linalg.norm(t))


def _feathered_normal(
    s_hat: np.ndarray, t_hat: np.ndarray, current_n_hat: Optional[np.ndarray]
) -> np.ndarray:
    """Edge-on (``n . s_hat = 0``) sail normal.

    Picks the in-plane edge-on orientation (``t_hat`` or ``-t_hat``) that is
    closer to ``current_n_hat`` -- minimises the slew from the current
    attitude when the steering switches to feather. If ``current_n_hat`` is
    ``None``, defaults to ``+t_hat``.
    """
    if current_n_hat is None:
        return t_hat.copy()
    return t_hat.copy() if float(np.dot(current_n_hat, t_hat)) >= 0.0 else -t_hat


def _feathered_normal_velocity(
    s_basis: np.ndarray,
    t_basis: np.ndarray,
    v_hat: np.ndarray,
    current_n: Optional[np.ndarray],
) -> np.ndarray:
    """In-``(s, v)``-plane sail normal PERPENDICULAR to ``v_hat`` (zero drag
    projected area), sign nearest ``current_n``.

    The drag-aware analogue of :func:`_feathered_normal`: coasting under
    atmospheric drag means presenting minimum area to the FLOW (edge-on to the
    velocity, ``A_proj = A|n.e_v| = 0``), not to the Sun. ``v_hat`` lies in the
    ``(s_basis, t_basis)`` plane by construction, so the in-plane vector
    perpendicular to it is the 90-deg rotation ``(v.t) s - (v.s) t`` (unit, since
    ``s, t`` are orthonormal and ``v_hat`` is a unit in-plane vector).
    """
    vs = float(np.dot(v_hat, s_basis))
    vt = float(np.dot(v_hat, t_basis))
    n_ev = vt * s_basis - vs * t_basis
    nrm = float(np.linalg.norm(n_ev))
    if nrm < _V_PERP_TOL:  # degenerate (v_hat ill-defined in-plane)
        return t_basis.copy()
    n_ev = n_ev / nrm
    if current_n is None:
        return n_ev
    return n_ev if float(np.dot(current_n, n_ev)) >= 0.0 else -n_ev


def dedot_steer(
    r: np.ndarray,
    v: np.ndarray,
    s_hat: np.ndarray,
    P_pa: float,
    sail: SolarSail,
    *,
    current_n_hat: Optional[np.ndarray] = None,
    params: Optional[DEdotParams] = None,
    drag: Optional["DragMeritContext"] = None,
    et: Optional[float] = None,
) -> DEdotSteering:
    """Steering that maximises ``a_sail . v_hat`` over sail tilt ``alpha``.

    With ``drag`` (a :class:`DragMeritContext`) the merit becomes the NET energy
    rate ``(a_SRP + a_drag) . v_hat`` -- the controller prices atmospheric drag
    and tilts off broadside-to-velocity when drag would dominate. ``drag=None``
    (default) is bit-identical to the SRP-only law (the drag branch is skipped).
    Drag-aware mode requires ``et`` (the co-rotating atmosphere needs the epoch).

    Parameters
    ----------
    r, v
        Orbit state (km, km/s, J2000), each shape (3,). Used only for the
        velocity direction; the dE/dt cost is local in state and frame-
        invariant.
    s_hat
        Unit sail-to-Sun direction (J2000), shape (3,).
    P_pa
        Solar radiation pressure at the sail, Pa. May already be
        shadow-gated -- if ``P_pa <= 0`` the steering feathers
        unconditionally (SRP is zero in eclipse, no orientation matters).
    sail
        ``reflectors.srp.SolarSail`` bus (area, mass, optical coefficients).
    current_n_hat
        Current sail normal (for feather-tie-breaking only); does not affect
        the thrust-arc optimum.
    params
        ``DEdotParams``; ``None`` -> defaults.

    Returns
    -------
    DEdotSteering
        See class docstring.
    """
    if params is None:
        params = DEdotParams()

    r_arr = np.asarray(r, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    s = np.asarray(s_hat, dtype=float)
    n_cur = (
        None if current_n_hat is None
        else np.asarray(current_n_hat, dtype=float)
    )

    v_mag = float(np.linalg.norm(v_arr))
    if v_mag < 1.0e-15:
        # No velocity direction defined; trivial fallback.
        return DEdotSteering(
            n_star_j2000=s.copy(),
            alpha_rad=0.0,
            thrust=False,
            dEdt_per_unit_mass_km2_s3=0.0,
        )
    v_hat = v_arr / v_mag
    s_basis, t_basis = _sun_velocity_basis(s, v_hat)

    # Feather posture: edge-on to the SUN (zero SRP) for the SRP-only law, but
    # edge-on to velocity (zero drag) when drag-aware -- coasting under drag
    # means presenting minimum area to the FLOW, not to the Sun. With drag=None
    # this reduces to _feathered_normal (s_basis==s, t_basis==t_hat).
    def _feather_normal() -> np.ndarray:
        if drag is not None:
            return _feathered_normal_velocity(s_basis, t_basis, v_hat, n_cur)
        return _feathered_normal(s_basis, t_basis, n_cur)

    # Eclipse / no-SRP -> feather (any orientation gives zero SRP force; the
    # drag-aware feather still minimises drag).
    if P_pa <= 0.0:
        return DEdotSteering(
            n_star_j2000=_feather_normal(),
            alpha_rad=0.5 * math.pi,
            thrust=False,
            dEdt_per_unit_mass_km2_s3=0.0,
        )

    # Drag-aware merit (the single toggle). Precompute the co-rotating relative
    # velocity and the local density once -- they are constant across the alpha
    # search; only the drag projected area A_proj = A|n.e_v| varies with the
    # candidate orientation. ``drag=None`` -> this block is skipped and the merit
    # below remains the SRP-only ``a_sail . v_hat``.
    _drag_accel = None
    _drag_v_rel = None
    _drag_rho = 0.0
    if drag is not None:
        if et is None:
            raise ValueError(
                "drag-aware dedot_steer requires et (atmosphere co-rotation)"
            )
        from reflectors.drag import atmosphere_relative_velocity, drag_acceleration

        _drag_accel = drag_acceleration
        _drag_v_rel = atmosphere_relative_velocity(
            r_arr, v_arr, et, drag.central_body.body_frame
        )
        _r_mag = float(np.linalg.norm(r_arr))
        _alt_km = _r_mag - drag.central_body.equatorial_radius_km
        # s_hat (sat->Sun) is the diurnal-bulge apex direction (M&G Eq. 3.105).
        _drag_rho = drag.density_model.density_kg_m3(_alt_km, r_arr / _r_mag, s)

    # Periapsis-rate guard activation check. When the
    # current periapsis has dropped below ``rp_warn_km``, the merit is the
    # constrained max: maximise (a_sail . v_hat) over the alphas with
    # Gauss drp/dt(alpha) >= 0. The unconstrained merit (just dE/dt) is
    # recovered when the guard is dormant.
    guard_active = False
    elements = None  # filled below if needed
    r_hat = theta_hat = h_hat = None  # RTN basis if needed
    if params.rp_warn_km is not None:
        elements = osculating_elements(r_arr, v_arr, _resolve_mu(params))
        rp_current = elements.a_km * (1.0 - elements.e)
        if rp_current < params.rp_warn_km:
            guard_active = True
            r_hat, theta_hat, h_hat = rtn_basis(r_arr, v_arr)

    def drp_dt_at(n_hat: np.ndarray, a_sail: np.ndarray) -> float:
        """Instantaneous Gauss drp/dt (km/s) at the given orientation."""
        assert elements is not None and r_hat is not None  # guard_active path
        f_r = float(np.dot(a_sail, r_hat))
        f_t = float(np.dot(a_sail, theta_hat))
        f_h = float(np.dot(a_sail, h_hat))
        rates = gauss_variational_rates(elements, f_r, f_t, f_h)
        return (
            (1.0 - elements.e) * rates.da_dt_km_s
            - elements.a_km * rates.de_dt_per_s
        )

    def merit_pair(alpha: float) -> tuple[float, float]:
        """Return (dE/dt-merit, drp/dt) at this orientation.

        The merit is the velocity-aligned thrust component (km/s^2). The
        drp/dt is computed only when the guard is active; otherwise the
        sentinel ``+inf`` is returned (guarantees the constraint is
        always satisfied -> behaviour identical to the unconstrained
        search).
        """
        n_hat = math.cos(alpha) * s_basis + math.sin(alpha) * t_basis
        a_sail = mcinnes_srp_acceleration(n_hat, s_basis, P_pa, sail)
        merit_val = float(np.dot(a_sail, v_hat))
        if _drag_accel is not None:
            # Net energy rate: subtract the drag work (a_drag . v_hat < 0). The
            # drag projected area depends on this candidate n_hat, so a_drag is
            # re-evaluated per alpha (v_rel, rho are the precomputed constants).
            a_drag = _drag_accel(
                _drag_v_rel, n_hat, _drag_rho, drag.C_d,
                sail.area_m2, sail.mass_kg,
            )
            merit_val += float(np.dot(a_drag, v_hat))
        rp_rate = drp_dt_at(n_hat, a_sail) if guard_active else float("inf")
        return merit_val, rp_rate

    def feasible_merit(alpha: float) -> float:
        """Constrained merit: ``-inf`` if drp/dt < 0 when the guard active."""
        f, rp_rate = merit_pair(alpha)
        return f if rp_rate >= 0.0 else -math.inf

    # Coarse grid (inclusive endpoints).
    A = params.max_cone_rad
    n_grid = params.grid_n
    best_alpha = 0.0
    best_f = -math.inf
    grid_step = 2.0 * A / n_grid
    for i in range(n_grid + 1):
        alpha = -A + i * grid_step
        f = feasible_merit(alpha)
        if f > best_f:
            best_f = f
            best_alpha = alpha

    if best_f == -math.inf:
        # No feasible orientation in the cone -- either no positive dE/dt
        # arc exists or the rp-rate guard forbids every cone direction.
        return DEdotSteering(
            n_star_j2000=_feather_normal(),
            alpha_rad=0.5 * math.pi,
            thrust=False,
            dEdt_per_unit_mass_km2_s3=0.0,
        )

    # Golden-section refinement over one grid-step window around the winner.
    lo = max(-A, best_alpha - grid_step)
    hi = min(A, best_alpha + grid_step)
    inv_gr = (math.sqrt(5.0) - 1.0) / 2.0  # 1/phi
    c = hi - (hi - lo) * inv_gr
    d = lo + (hi - lo) * inv_gr
    fc = feasible_merit(c)
    fd = feasible_merit(d)
    for _ in range(params.golden_iters):
        if fc > fd:
            hi = d
            d = c
            fd = fc
            c = hi - (hi - lo) * inv_gr
            fc = feasible_merit(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + (hi - lo) * inv_gr
            fd = feasible_merit(d)
    alpha_star = 0.5 * (lo + hi)
    f_star = feasible_merit(alpha_star)

    # Compare the refined optimum against the coarse-grid winner -- the
    # refinement can occasionally drift if the bracket has multiple local
    # maxima or crosses a feasibility boundary. Keep the better one.
    if best_f > f_star:
        alpha_star = best_alpha
        f_star = best_f

    if f_star <= params.feather_threshold_km_s2:
        # No constraint-satisfying orientation gives velocity-aligned thrust
        # (feasibility may still allow zero merit; below the threshold means
        # 'not worth thrusting' regardless of feasibility).
        return DEdotSteering(
            n_star_j2000=_feather_normal(),
            alpha_rad=0.5 * math.pi,
            thrust=False,
            dEdt_per_unit_mass_km2_s3=0.0,
        )

    n_star = math.cos(alpha_star) * s_basis + math.sin(alpha_star) * t_basis
    n_star = n_star / float(np.linalg.norm(n_star))
    return DEdotSteering(
        n_star_j2000=n_star,
        alpha_rad=alpha_star,
        thrust=True,
        dEdt_per_unit_mass_km2_s3=f_star * v_mag,
    )


def _resolve_mu(params: DEdotParams) -> float:
    """Lazily resolve ``mu_km3_s2`` -- default to Mars planet GM."""
    if params.mu_km3_s2 is not None:
        return params.mu_km3_s2
    # Imported lazily so the module stays cheap when the guard is disabled.
    from reflectors.dynamics import mars_gm_km3_per_s2
    return mars_gm_km3_per_s2()


# ===========================================================================
# Blended energy + periapsis-safety steering (Macdonald & McInnes 2005)
# ===========================================================================
#
# The greedy dE/dt-only controller above (:func:`dedot_steer`) crashes
# escape spirals at sigma >= 14 because raising orbital energy alone does
# not guarantee periapsis safety: the angular momentum must rise enough
# that r_p = a(1-e) stays above the atmosphere. Macdonald & McInnes 2005
# ("Realistic Earth Escape Strategies for Solar Sailing", JGCD; Strathprints
# 6252) blend an energy-gain controller with a periapsis-safety controller
# via a state-dependent weight, producing near-optimal escape while
# guaranteeing a minimum altitude.
#
# Safety margin (for any conic, two-body):
#
#     S = h^2 - 2 mu r_star - 2 epsilon r_star^2                          (M&M)
#
# where ``epsilon = v^2/2 - mu/r`` is the specific energy, ``h = |r x v|``
# the specific angular momentum, ``mu`` the central-body GM, and
# ``r_star`` the guard radius. The relation comes from the conic identity
# ``h^2 = 2 mu r_p + 2 epsilon r_p^2`` -- S >= 0 iff r_p >= r_star.
#
# The blended objective is
#
#     J(n) = w_E (v . a_s(n)) + w_S(S) (b . a_s(n)),       b = 2 (h x r) - 2 r_star^2 v
#
# with ``w_E`` constant and the safety weight ``w_S(S) = k_S exp(-S/S_0)``
# rising as the orbit approaches the periapsis guard. Inserting the
# objective vector ``c = w_E v + w_S(S) b``, the merit reduces to
# ``c . a_s(n)`` -- a single dot product. Maximising it over sail
# orientations gives the desired ``n*``. The result is "energy-raising
# when safe; angular-momentum-raising when r_p is at risk", with a
# smooth transition between the two regimes.
#
# Reference primary: Macdonald & McInnes, "Realistic Earth Escape
# Strategies for Solar Sailing", J. Guid. Control Dyn., 2005 (Strathprints
# 6252). The merit form here generalises the same idea to any central body.


@dataclass(frozen=True)
class BlendedParams:
    """Parameters for the blended energy + safety steering.

    Attributes
    ----------
    r_star_km
        Soft periapsis guard (km). The safety controller's "target floor"
        -- when ``r_p`` falls toward this value the safety weight rises
        and the controller sacrifices energy gain for h-gain to keep
        periapsis above ``r_star``. Set above the hard atmosphere floor
        (e.g. ``R_eq + 600`` km gives a ~300 km soft margin above the
        ``R_eq + 300 km`` hard altitude floor). Default
        ``MARS_EQUATORIAL_RADIUS + 600 km`` ~ 3996 km.
    w_E
        Weight on the energy-gain term ``v . a_s``. Dimensionless scale,
        default 1.0.
    k_S
        Peak weight on the safety term ``b . a_s`` (taken when ``S = 0``).
        Default 1.0; larger values increase the safety response near the
        boundary.
    S_0_km4_s2
        Soft-guard decay length for the safety weight (units of S, i.e.
        km^4/s^2). ``w_S(S) = k_S exp(-S/S_0)`` is 1.0 at S=0, falls to
        0.37 at S=S_0, and 0.05 at S=3 S_0. Pick so the safety controller
        is active while S is below a few S_0. Default 1.0e7, approximately the
        initial circular-orbit safety-margin scale at LMO.
    max_cone_rad
        Cone half-angle (rad) bounding ``|alpha|`` -- the maximum tilt of
        the sail normal away from the Sun line. Default
        ``math.radians(80.0)`` matching :class:`DEdotParams`.
    grid_n, golden_iters
        1D search grid + golden refinement (same shape as DEdotParams).
    feather_threshold_km_s2
        If the maximum merit over the searched alpha range is below this
        threshold, feather. Default 0.0.
    mu_km3_s2
        Central GM (km^3/s^2). Default None -> resolved lazily to
        :func:`reflectors.dynamics.mars_gm_km3_per_s2` at call time.
    """

    r_star_km: float = 3996.19  # R_eq + 600 km
    w_E: float = 1.0
    k_S: float = 1.0
    S_0_km4_s2: float = 1.0e7
    max_cone_rad: float = math.radians(80.0)
    grid_n: int = 20
    golden_iters: int = 30
    feather_threshold_km_s2: float = 0.0
    mu_km3_s2: Optional[float] = None

    def __post_init__(self) -> None:
        if self.r_star_km <= 0.0:
            raise ValueError(
                f"r_star_km must be > 0, got {self.r_star_km}"
            )
        if self.w_E < 0.0 or self.k_S < 0.0:
            raise ValueError(
                f"weights must be >= 0, got w_E={self.w_E}, k_S={self.k_S}"
            )
        if self.S_0_km4_s2 <= 0.0:
            raise ValueError(
                f"S_0_km4_s2 must be > 0, got {self.S_0_km4_s2}"
            )
        if not (0.0 < self.max_cone_rad < 0.5 * math.pi):
            raise ValueError(
                f"max_cone_rad must be in (0, pi/2), got {self.max_cone_rad}"
            )
        if self.grid_n < 2:
            raise ValueError(f"grid_n must be >= 2, got {self.grid_n}")
        if self.golden_iters < 0:
            raise ValueError(
                f"golden_iters must be >= 0, got {self.golden_iters}"
            )
        if self.mu_km3_s2 is not None and self.mu_km3_s2 <= 0.0:
            raise ValueError(
                f"mu_km3_s2 must be > 0 if set, got {self.mu_km3_s2}"
            )


@dataclass(frozen=True)
class BlendedSteering:
    """Result of :func:`blended_steer`.

    Attributes
    ----------
    n_star_j2000
        Desired sail normal (sun-facing convention, J2000 axes), shape (3,).
    alpha_rad
        Cone angle (rad) at the optimum (the angle between ``n_star`` and
        ``s_hat``). ``pi/2`` for the feathered sentinel.
    thrust
        True iff a positive-merit orientation was found.
    merit_value
        The optimised merit ``c . a_s`` at the chosen orientation
        (km^4/s^3 in mixed units; only the sign + relative comparison is
        meaningful).
    safety_margin_km4_s2
        Current ``S = h^2 - 2 mu r_star - 2 epsilon r_star^2``. Negative
        means osculating periapsis is below ``r_star`` -- the safety
        controller should be in full effect.
    safety_weight
        Current ``w_S = k_S exp(-S/S_0)``. Tracks how much the safety
        term is biasing the controller.
    """

    n_star_j2000: np.ndarray
    alpha_rad: float
    thrust: bool
    merit_value: float
    safety_margin_km4_s2: float
    safety_weight: float


def _resolve_mu_blended(params: BlendedParams) -> float:
    if params.mu_km3_s2 is not None:
        return params.mu_km3_s2
    from reflectors.dynamics import mars_gm_km3_per_s2
    return mars_gm_km3_per_s2()


def blended_steer(
    r: np.ndarray,
    v: np.ndarray,
    s_hat: np.ndarray,
    P_pa: float,
    sail: SolarSail,
    *,
    current_n_hat: Optional[np.ndarray] = None,
    params: Optional[BlendedParams] = None,
    safety_blend_ratio: float = 0.0,
) -> BlendedSteering:
    """Blended energy + periapsis-safety steering (Macdonald & McInnes 2005).

    Maximise ``J(n) = c . a_s(n)`` where
    ``c = w_E v + w_S [2 (h x r) - 2 r_star^2 v]`` and
    ``w_S = k_S exp(-S/S_0) + w_S_anticip``, with the safety margin ``S = h^2 -
    2 mu r_star - 2 epsilon r_star^2``. The sail orientation that
    maximises ``c . a_s`` lies in the plane of ``s_hat`` and the in-plane
    part of ``c``; the search is 1D in that plane, same shape as
    :func:`dedot_steer`.

    Parameters and return mirror :func:`dedot_steer`, with the safety
    state ``(S, w_S)`` exposed in the returned dataclass for monitoring.

    ``safety_blend_ratio`` (default 0.0) is an optional forecast-driven safety
    bias, expressed in dimensionless blend-ratio space: it adds a safety weight
    ``w_S_anticip = safety_blend_ratio * w_E |v| / |b|`` so the safety term
    contributes a fraction ``safety_blend_ratio`` of the energy term's
    magnitude to the merit direction (``b = 2(h x r) - 2 r_star^2 v``). This
    auto-scales across the ``|b| ~ 1e8 >> |v| ~ 3`` mismatch, so the
    gain is O(1). With ``k_S = 0`` (energy-max base) and
    ``safety_blend_ratio = 0`` the controller is pure dE/dt-max (rides the
    boundary). A nonzero bias can respond to a forecast periapsis collapse by
    steering toward
    angular-momentum gain *before* the osculating ``S`` itself drops.
    """
    if params is None:
        params = BlendedParams()
    r_arr = np.asarray(r, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    s = np.asarray(s_hat, dtype=float)
    n_cur = (
        None if current_n_hat is None
        else np.asarray(current_n_hat, dtype=float)
    )

    # Sun side feather (P=0 / eclipse) -- no orientation gives positive force.
    if P_pa <= 0.0:
        # Pick any perpendicular-to-s axis for the feathered normal.
        _, t_hat = _sun_velocity_basis(s, v_arr / max(1e-15, float(np.linalg.norm(v_arr))))
        return BlendedSteering(
            n_star_j2000=_feathered_normal(s, t_hat, n_cur),
            alpha_rad=0.5 * math.pi,
            thrust=False,
            merit_value=0.0,
            safety_margin_km4_s2=0.0,
            safety_weight=0.0,
        )

    # Current orbit state -- specific energy, angular momentum, safety margin.
    mu = _resolve_mu_blended(params)
    r_mag = float(np.linalg.norm(r_arr))
    v_mag2 = float(np.dot(v_arr, v_arr))
    eps = 0.5 * v_mag2 - mu / r_mag
    h_vec = np.cross(r_arr, v_arr)
    h2 = float(np.dot(h_vec, h_vec))
    r_star = params.r_star_km
    S = h2 - 2.0 * mu * r_star - 2.0 * eps * r_star * r_star

    # Reactive safety weight rises as S approaches zero.
    w_S = params.k_S * math.exp(-S / params.S_0_km4_s2) if params.k_S > 0.0 else 0.0
    w_E = params.w_E

    # Objective vector c (J2000). a_s . c is the blended merit (km^4/s^3).
    safety_term = 2.0 * np.cross(h_vec, r_arr) - 2.0 * (r_star ** 2) * v_arr
    # Anticipatory bias (blend-ratio space): add a safety weight that makes the
    # safety term contribute ``safety_blend_ratio`` of the energy term's
    # magnitude to the merit (auto-scaled across |b| >> |v|).
    if safety_blend_ratio > 0.0:
        b_mag = float(np.linalg.norm(safety_term))
        v_mag = float(np.linalg.norm(v_arr))
        if b_mag > _V_PERP_TOL:
            w_S = w_S + safety_blend_ratio * w_E * v_mag / b_mag
    c = w_E * v_arr + w_S * safety_term

    # Search plane: spanned by s_hat and the in-plane part of c.
    c_perp = c - float(np.dot(c, s)) * s
    c_perp_norm = float(np.linalg.norm(c_perp))
    if c_perp_norm > _V_PERP_TOL:
        u_hat = c_perp / c_perp_norm
    else:
        # c is along s_hat (rare); pick arbitrary perpendicular -- the 1D
        # search will collapse to alpha=0 (sun-facing) as the optimum if
        # any thrust at all helps.
        trial = np.array([1.0, 0.0, 0.0])
        if abs(s[0]) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        u_hat = trial - float(np.dot(trial, s)) * s
        u_hat = u_hat / float(np.linalg.norm(u_hat))

    def merit(alpha: float) -> float:
        n_hat = math.cos(alpha) * s + math.sin(alpha) * u_hat
        a_sail = mcinnes_srp_acceleration(n_hat, s, P_pa, sail)
        return float(np.dot(a_sail, c))

    # Coarse 1D grid scan.
    A = params.max_cone_rad
    n_grid = params.grid_n
    best_alpha = 0.0
    best_f = -math.inf
    grid_step = 2.0 * A / n_grid
    for i in range(n_grid + 1):
        alpha = -A + i * grid_step
        f = merit(alpha)
        if f > best_f:
            best_f = f
            best_alpha = alpha

    # Golden-section refinement around the winner.
    lo = max(-A, best_alpha - grid_step)
    hi = min(A, best_alpha + grid_step)
    inv_gr = (math.sqrt(5.0) - 1.0) / 2.0
    cc = hi - (hi - lo) * inv_gr
    dd = lo + (hi - lo) * inv_gr
    fc = merit(cc)
    fd = merit(dd)
    for _ in range(params.golden_iters):
        if fc > fd:
            hi = dd
            dd = cc
            fd = fc
            cc = hi - (hi - lo) * inv_gr
            fc = merit(cc)
        else:
            lo = cc
            cc = dd
            fc = fd
            dd = lo + (hi - lo) * inv_gr
            fd = merit(dd)
    alpha_star = 0.5 * (lo + hi)
    f_star = merit(alpha_star)
    if best_f > f_star:
        alpha_star = best_alpha
        f_star = best_f

    if f_star <= params.feather_threshold_km_s2:
        # No cone direction gives positive merit -> feather.
        _, t_hat_feat = _sun_velocity_basis(
            s,
            v_arr / max(1e-15, float(np.linalg.norm(v_arr))),
        )
        return BlendedSteering(
            n_star_j2000=_feathered_normal(s, t_hat_feat, n_cur),
            alpha_rad=0.5 * math.pi,
            thrust=False,
            merit_value=f_star,
            safety_margin_km4_s2=S,
            safety_weight=w_S,
        )

    n_star = math.cos(alpha_star) * s + math.sin(alpha_star) * u_hat
    n_star = n_star / float(np.linalg.norm(n_star))
    return BlendedSteering(
        n_star_j2000=n_star,
        alpha_rad=alpha_star,
        thrust=True,
        merit_value=f_star,
        safety_margin_km4_s2=S,
        safety_weight=w_S,
    )
