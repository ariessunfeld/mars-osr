"""IPOPT equality-constrained rendezvous solver for the piecewise-RTN cruise.

The interplanetary cruise is solved as a 6-equality-constraint NLP: drive the
SCALED terminal-state defect ``g(x) = 0`` (the piecewise-RTN command's miss to the
target, non-dimensionalised by ``r_scale``/``v_scale``). The objective is the
node-difference smoothness regulariser (``cruise_piecewise.smoothness_*``), which
keeps the otherwise feasibility-only problem well-posed (a unique smooth command
among the feasible set). This mirrors the repository's Fourier ``CruiseNLP``
(``scripts/run_interplanetary_cruise.py``), reusing
``reflectors.parallel.CloudpickleMap`` for the dense parallel finite-difference
constraint Jacobian.

The local, well-conditioned piecewise-RTN parameterisation avoids the
cone/clock singularity that degrades the global-Fourier formulation. ``cyipopt``
is imported lazily so the library imports without it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from reflectors.cruise_piecewise import (
    DEFAULT_SMOOTH_SCALE_RAD,
    DEFAULT_SMOOTH_WEIGHT,
    smoothness_gradient,
    smoothness_objective,
)

DEFAULT_ANGLE_BOUND_RAD = math.radians(35.0)  # +/-35 deg box (reference)


def angle_box_bounds(N: int, angle_bound_rad: float = DEFAULT_ANGLE_BOUND_RAD):
    """Box bounds for the 2N angle decision variables (radians). The piecewise
    command needs no slew-coupled coefficient box (unlike the Fourier path): the
    per-node slew is trivially feasible over a multi-month segment
    (``cruise_piecewise.assert_piecewise_slew_feasible``), so the box is just the
    +/-``angle_bound`` physical authority limit."""
    lo = [-angle_bound_rad] * (2 * N)
    hi = [angle_bound_rad] * (2 * N)
    return lo, hi


class PiecewiseCruiseNLP:
    """cyipopt problem object for the piecewise-RTN cruise rendezvous.

    Parameters
    ----------
    defect
        Picklable closure ``x -> g`` (scaled 6-vector terminal defect), e.g.
        ``cruise_piecewise.make_piecewise_cruise_defect`` (fixed target) or the
        variable-time defect. ``x`` is ``[phi_0..phi_{N-1},
        theta_0..theta_{N-1}, (optional trailing vars)]``.
    n_vars
        Length of the decision vector (``2N`` fixed-time, ``2N+1`` with a duration).
    N
        Number of control nodes (the smoothness regulariser acts on ``x[:2N]``).
    cp_map
        A ``reflectors.parallel.CloudpickleMap`` (or any ``map``-like callable) for
        the parallel finite-difference Jacobian.
    """

    def __init__(
        self,
        defect: Callable[[np.ndarray], np.ndarray],
        n_vars: int,
        N: int,
        cp_map,
        *,
        w_smooth: float = DEFAULT_SMOOTH_WEIGHT,
        smooth_scale_rad: float = DEFAULT_SMOOTH_SCALE_RAD,
        fd_h: float = 1.0e-3,
        early_stop_patience: Optional[int] = None,
        early_stop_rtol: float = 0.02,
        early_stop_min_iters: int = 20,
    ):
        self.defect = defect
        self.n_vars = int(n_vars)
        self.N = int(N)
        self.cp_map = cp_map
        self.w_smooth = float(w_smooth)
        self.smooth_scale_rad = float(smooth_scale_rad)
        self.fd_h = float(fd_h)
        # IPOPT returns the final iterate, which need not minimize the defect;
        # retain the lowest-defect iterate separately.
        self.best = {"norm": np.inf, "x": None}
        # Optional early-stop on a plateau of the scaled primal infeasibility
        # ``inf_pr``. With patience None, ``intermediate`` only reports progress.
        # When enabled, a cold IPOPT solve that is not descending toward
        # feasibility stops early so the caller can try the next initialization.
        self.early_stop_patience = (
            int(early_stop_patience) if early_stop_patience is not None else None
        )
        self.early_stop_rtol = float(early_stop_rtol)
        self.early_stop_min_iters = int(early_stop_min_iters)
        self._es_best = np.inf
        self._es_stall = 0

    # -- objective: node-difference smoothness over the angle blocks -----------
    def objective(self, x):
        return smoothness_objective(
            np.asarray(x), self.N, w_smooth=self.w_smooth, scale_rad=self.smooth_scale_rad
        )

    def gradient(self, x):
        return smoothness_gradient(
            np.asarray(x), self.N, w_smooth=self.w_smooth, scale_rad=self.smooth_scale_rad
        )

    # -- constraints: the scaled terminal defect -------------------------------
    def constraints(self, x):
        g = self.defect(np.asarray(x, dtype=float))
        gn = float(np.linalg.norm(g))
        if gn < self.best["norm"]:
            self.best["norm"] = gn
            self.best["x"] = np.asarray(x, dtype=float).copy()
        return g

    def jacobian(self, x):
        """Dense 6 x n_vars constraint Jacobian by parallel central finite
        differences. ABSOLUTE per-variable step ``fd_h`` (NOT relative): the
        decision vector mixes O(0.5 rad) angles with an O(500-day) duration D, and
        a relative step ``fd_h*|x|`` would perturb D by ~0.5 day -> the Mars target
        moves ~1e6 km over the step, so the D-column captures a large NONLINEAR
        target jump (and can change the RK4 step count) instead of the local
        derivative -> a corrupted Jacobian that stalls IPOPT. With an absolute
        ~1e-3 step, D is
        perturbed by ~1e-3 day (Mars moves ~2000 km, producing a well-scaled local derivative)."""
        x = np.asarray(x, dtype=float)
        h_i = np.full(self.n_vars, self.fd_h)
        pts = []
        for i in range(self.n_vars):
            xp = x.copy(); xp[i] += h_i[i]; pts.append(xp)
            xm = x.copy(); xm[i] -= h_i[i]; pts.append(xm)
        res = self.cp_map(self.defect, pts)  # 2*n_vars entries, each a 6-vector
        J = np.empty((6, self.n_vars))
        for i in range(self.n_vars):
            J[:, i] = (np.asarray(res[2 * i]) - np.asarray(res[2 * i + 1])) / (2.0 * h_i[i])
        return J.flatten()  # dense row-major

    def jacobianstructure(self):
        return (np.repeat(np.arange(6), self.n_vars), np.tile(np.arange(self.n_vars), 6))

    def intermediate(self, alg_mod, it, obj, inf_pr, inf_du, mu, dn, rg, a_du, a_pr, ls):
        if it % 5 == 0 or inf_pr < 1e-4:
            print(f"  [ipopt] it={it:4d}  obj={obj:.3e}  inf_pr(scaled)={inf_pr:.3e}", flush=True)
        if self.early_stop_patience is None:
            return  # default: never abort
        # Plateau detection on the scaled primal infeasibility: count consecutive
        # iterations without a > early_stop_rtol relative drop in the best inf_pr;
        # after a warm-up (early_stop_min_iters) a stall >= patience -> abort by
        # returning False (cyipopt User_Requested_Stop). The best-defect iterate is
        # still recovered via self.best, so an aborted solve reports faithfully.
        if inf_pr < self._es_best * (1.0 - self.early_stop_rtol):
            self._es_best = inf_pr
            self._es_stall = 0
        else:
            self._es_stall += 1
        if it >= self.early_stop_min_iters and self._es_stall >= self.early_stop_patience:
            print(
                f"  [ipopt] EARLY-STOP at it={it}: inf_pr plateaued "
                f"({self._es_stall} iters w/o >{self.early_stop_rtol:.0%} drop; "
                f"best inf_pr(scaled)={self._es_best:.3e})",
                flush=True,
            )
            return False
        return


def _parallel_central_fd_jacobian(defect, x, n_vars, fd_h, cp_map):
    """Dense ``6 x n_vars`` Jacobian of ``defect`` by parallel central finite
    differences with an ABSOLUTE per-variable step ``fd_h`` (see
    ``PiecewiseCruiseNLP.jacobian`` for why absolute, not relative). ``cp_map``
    must be order-preserving (``CloudpickleMap`` wraps ``Pool.map``). Shared by the
    least-squares gradient path; the feasibility path evaluates the equivalent
    Jacobian inline."""
    x = np.asarray(x, dtype=float)
    pts = []
    for i in range(n_vars):
        xp = x.copy(); xp[i] += fd_h; pts.append(xp)
        xm = x.copy(); xm[i] -= fd_h; pts.append(xm)
    res = cp_map(defect, pts)  # 2*n_vars entries, each a 6-vector, IN ORDER
    J = np.empty((6, n_vars))
    for i in range(n_vars):
        J[:, i] = (np.asarray(res[2 * i]) - np.asarray(res[2 * i + 1])) / (2.0 * fd_h)
    return J


class PiecewiseLeastSquaresNLP:
    """cyipopt problem for ``min ||g(x)||^2 (+ optional smoothness)`` over the box
    bounds -- the MINIMISATION (least-squares) formulation, m=0 (bounds only).

    Unlike :class:`PiecewiseCruiseNLP` (equality-constrained feasibility, m=6,
    ``g(x)=0``), the terminal-state MISS is the OBJECTIVE. With no equality
    constraints there is no restoration phase, so IPOPT's line search descends
    ``||g||^2`` monotonically -- robust from a COLD start, where the feasibility
    formulation's restoration can thrash. The Gauss-Newton
    structure is left to IPOPT's limited-memory Hessian; ``gradient = 2 J^T g`` uses
    the same parallel central-FD ``J`` as the feasibility path. ``best`` tracks the
    lowest ``||g||`` iterate seen (IPOPT returns the final one, which may be worse).
    """

    def __init__(
        self,
        defect: Callable[[np.ndarray], np.ndarray],
        n_vars: int,
        N: int,
        cp_map,
        *,
        w_smooth: float = 0.0,
        smooth_scale_rad: float = DEFAULT_SMOOTH_SCALE_RAD,
        fd_h: float = 1.0e-3,
        early_stop_patience: Optional[int] = None,
        early_stop_rtol: float = 0.02,
        early_stop_min_iters: int = 20,
    ):
        self.defect = defect
        self.n_vars = int(n_vars)
        self.N = int(N)
        self.cp_map = cp_map
        self.w_smooth = float(w_smooth)
        self.smooth_scale_rad = float(smooth_scale_rad)
        self.fd_h = float(fd_h)
        self.best = {"norm": np.inf, "x": None}
        self.early_stop_patience = (
            int(early_stop_patience) if early_stop_patience is not None else None
        )
        self.early_stop_rtol = float(early_stop_rtol)
        self.early_stop_min_iters = int(early_stop_min_iters)
        self._es_best = np.inf
        self._es_stall = 0
        self._cache = {"x": None, "g": None}  # avoid re-propagating g for objective->gradient

    def _g(self, x):
        """Scaled 6-vector defect at ``x`` (cached per-x; updates ``best``)."""
        x = np.asarray(x, dtype=float)
        c = self._cache
        if c["x"] is not None and c["g"] is not None and np.array_equal(c["x"], x):
            return c["g"]
        g = np.asarray(self.defect(x), dtype=float)
        gn = float(np.linalg.norm(g))
        if gn < self.best["norm"]:
            self.best["norm"] = gn
            self.best["x"] = x.copy()
        c["x"] = x.copy(); c["g"] = g
        return g

    def objective(self, x):
        g = self._g(x)
        obj = float(g @ g)
        if self.w_smooth:
            obj += smoothness_objective(
                np.asarray(x), self.N, w_smooth=self.w_smooth, scale_rad=self.smooth_scale_rad
            )
        return obj

    def gradient(self, x):
        g = self._g(x)
        J = _parallel_central_fd_jacobian(self.defect, x, self.n_vars, self.fd_h, self.cp_map)
        grad = 2.0 * (J.T @ g)
        if self.w_smooth:
            grad = grad + smoothness_gradient(
                np.asarray(x), self.N, w_smooth=self.w_smooth, scale_rad=self.smooth_scale_rad
            )
        return np.asarray(grad, dtype=float)

    def intermediate(self, alg_mod, it, obj, inf_pr, inf_du, mu, dn, rg, a_du, a_pr, ls):
        if it % 5 == 0 or obj < 1e-8:
            print(f"  [ipopt-lsq] it={it:4d}  obj=||g||^2={obj:.3e}", flush=True)
        if self.early_stop_patience is None:
            return
        # Plateau detection on the OBJECTIVE ||g||^2 (m=0 -> inf_pr is ~0, useless here).
        if obj < self._es_best * (1.0 - self.early_stop_rtol):
            self._es_best = obj
            self._es_stall = 0
        else:
            self._es_stall += 1
        if it >= self.early_stop_min_iters and self._es_stall >= self.early_stop_patience:
            print(
                f"  [ipopt-lsq] EARLY-STOP at it={it}: ||g||^2 plateaued "
                f"({self._es_stall} iters w/o >{self.early_stop_rtol:.0%} drop; "
                f"best ||g||^2={self._es_best:.3e})",
                flush=True,
            )
            return False
        return


def solve_piecewise_ipopt(
    defect: Callable[[np.ndarray], np.ndarray],
    n_vars: int,
    N: int,
    *,
    lb,
    ub,
    x0,
    workers: int = 10,
    cp_map=None,
    w_smooth: float = DEFAULT_SMOOTH_WEIGHT,
    smooth_scale_rad: float = DEFAULT_SMOOTH_SCALE_RAD,
    fd_h: float = 1.0e-3,
    max_iter: int = 400,
    tol: float = 1.0e-7,
    constr_viol_tol: float = 1.0e-6,
    print_level: int = 0,
    early_stop_patience: Optional[int] = None,
    early_stop_rtol: float = 0.02,
    early_stop_min_iters: int = 20,
    formulation: str = "feasibility",
    nlp_scaling_method: Optional[str] = None,
):
    """Solve the cruise rendezvous with IPOPT (cyipopt).

    ``formulation`` selects the problem posed:

    - ``"feasibility"`` (default): the equality-constrained problem
      (``m=6``, ``g(x)=0``, objective = node smoothness). Well-conditioned WARM
      near a feasible point.
    - ``"least_squares"``: ``min ||g(x)||^2 (+ w_smooth*smoothness)`` over the box
      bounds (``m=0``). No restoration phase -> robust COLD. ``nlp_scaling_method``
      (e.g. ``"none"``) is forwarded to IPOPT only here; ``g`` is already
      non-dimensionalised so disabling IPOPT's internal rescaling can help.

    Returns ``(x_best, info)``. ``x_best`` is the lowest-defect iterate seen (not
    necessarily IPOPT's final one). Builds and tears down its own
    ``CloudpickleMap`` (size ``workers``) unless ``cp_map`` is supplied; SPICE
    fork-safety is handled inside the picklable ``defect`` closure
    (``_ensure_worker_kernels``).
    """
    if formulation not in ("feasibility", "least_squares"):
        raise ValueError(f"formulation must be 'feasibility' or 'least_squares', got {formulation!r}")
    import cyipopt
    from reflectors.parallel import (
        CloudpickleMap,
        configure_multiprocessing_for_spice,
    )

    own_map = cp_map is None
    if own_map:
        configure_multiprocessing_for_spice()
        from reflectors.kernels import load_kernels

        load_kernels()  # fresh DAF handles right before forking the pool
        cp_map = CloudpickleMap(n_workers=workers)
    nlp_kwargs = dict(
        w_smooth=w_smooth, smooth_scale_rad=smooth_scale_rad, fd_h=fd_h,
        early_stop_patience=early_stop_patience, early_stop_rtol=early_stop_rtol,
        early_stop_min_iters=early_stop_min_iters,
    )
    if formulation == "least_squares":
        nlp = PiecewiseLeastSquaresNLP(defect, n_vars, N, cp_map, **nlp_kwargs)
        m = 0
    else:
        nlp = PiecewiseCruiseNLP(defect, n_vars, N, cp_map, **nlp_kwargs)
        m = 6
    lb = list(lb); ub = list(ub)
    x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
    try:
        if m == 0:
            problem = cyipopt.Problem(n=n_vars, m=0, problem_obj=nlp, lb=lb, ub=ub)
        else:
            problem = cyipopt.Problem(
                n=n_vars, m=6, problem_obj=nlp, lb=lb, ub=ub, cl=[0.0] * 6, cu=[0.0] * 6
            )
        problem.add_option("tol", float(tol))
        if m > 0:
            problem.add_option("constr_viol_tol", float(constr_viol_tol))
        problem.add_option("max_iter", int(max_iter))
        problem.add_option("hessian_approximation", "limited-memory")
        problem.add_option("mu_strategy", "adaptive")
        if nlp_scaling_method is not None:
            problem.add_option("nlp_scaling_method", str(nlp_scaling_method))
        problem.add_option("print_level", int(print_level))
        x_final, info = problem.solve(x0)
    finally:
        if own_map:
            cp_map.close()
    x_best = nlp.best["x"] if nlp.best["x"] is not None else np.asarray(x_final, dtype=float)
    info = dict(info)
    info["best_defect_norm"] = float(nlp.best["norm"])
    return x_best, info
