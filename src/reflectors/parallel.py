"""Multiprocessing helpers for SPICE-using parallel code.

The motivating problems
-----------------------
1. **SPICE pool inheritance.** Python's ``multiprocessing`` module on
   macOS changed its default start method from ``'fork'`` to ``'spawn'``
   in Python 3.8 (cf.
   https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods,
   "Changed in version 3.8: On macOS, the spawn start method is now the
   default."). Spawned workers start as fresh Python processes that do
   not inherit any of the parent's in-memory state — including the
   SPICE kernel pool loaded by the parent. Workers calling
   ``spice.spkezr(...)`` against an empty pool fail with
   ``SPICE(NOLEAPSECONDS)``/``SPICE(KERNELVARNOTFOUND)`` errors.

   ``configure_multiprocessing_for_spice()`` forces ``'fork'`` so
   children inherit the parent's address space via copy-on-write,
   including the SPICE kernel pool.

2. **Closure pickling.** scipy DE with ``workers > 1`` dispatches the
   cost function via ``multiprocessing.Pool.map``, which serializes
   every task through ``_ForkingPickler``. Local functions (closures
   captured inside another function) cannot be pickled by stdlib pickle.
   ``optimize.optimize_cruise_general`` defines its cost as
   ``def cost_closure(...)`` inside the function body, capturing
   ``config``, ``cruise_factory`` (itself a closure returned by
   ``make_handoff_cruise_factory``), and several scalar weights. Any of
   these is sufficient to break pickling.

   ``CloudpickleMap`` solves this by using ``cloudpickle`` (which
   handles closures) to serialize the function, then dispatching to
   workers via stdlib ``multiprocessing.Pool``. scipy DE accepts a
   ``workers`` argument that is either an int or a map-like callable;
   ``CloudpickleMap`` is the latter.

A note on the macOS fork deprecation guidance
---------------------------------------------
CPython emits ``DeprecationWarning: This process is multi-threaded, use
of fork() may lead to deadlocks in the child`` when ``fork()`` is
invoked from a multi-threaded parent on macOS. The warning is about
fork-after-thread, not fork itself. The command-line scripts in this repository are
single-threaded (the optimizer, propagator, and SPICE wrapper all run
in the main thread of the parent), so the warning does not apply.
SpiceyPy is not known to start background threads during import or kernel
loading.

Known limitation: worker-side history not aggregated to parent
--------------------------------------------------------------
In ``optimize.optimize_cruise_general`` the per-eval ``history`` list
is captured by the cost closure and mutated on every evaluation. Under
fork, each worker mutates its own copy-on-write list; the parent sees
no worker-side appends. ``scipy.optimize.differential_evolution`` runs
its final ``polish=True`` step (L-BFGS-B) in the parent, so the
polish-step evaluations DO reach the parent's ``history``, including
the final optimum. Per-population worker evaluations do not. DE
warm-starting between sols uses ``best_eval`` (the optimum), not the
per-eval ``history``, so this limitation does not affect correctness —
only the completeness of ``history.csv`` for sols optimized with
DE+workers>1. NM-only sols are unaffected (NM is single-process).

Usage
-----
Applications should call ``configure_multiprocessing_for_spice()`` once at
startup, BEFORE creating any ``multiprocessing.Pool`` or invoking
``scipy.optimize.differential_evolution`` with ``workers > 1``. For
parallel scipy DE against any closure-using cost function, pass
``workers=CloudpickleMap(n_workers=N)`` instead of a bare int.

Example::

    from reflectors.parallel import (
        CloudpickleMap, configure_multiprocessing_for_spice,
    )
    from reflectors.kernels import load_kernels
    from scipy.optimize import differential_evolution

    configure_multiprocessing_for_spice()
    load_kernels()

    cp_map = CloudpickleMap(n_workers=8)
    res = differential_evolution(
        cost_function_with_closures,  # closure ok
        bounds=BOUNDS,
        workers=cp_map,
        updating="deferred",  # required for workers != 1
        ...,
    )
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import cloudpickle
import numpy as np


logger = logging.getLogger(__name__)

# Default start method for SPICE-using parallel code. Both macOS and Linux
# support 'fork'; Windows does not. Callers can override per call, but the
# default matches the SPICE-pool-inheritance requirement.
DEFAULT_START_METHOD = "fork"


def configure_multiprocessing_for_spice(
    start_method: str = DEFAULT_START_METHOD,
) -> str:
    """Set the multiprocessing start method so workers inherit the SPICE pool.

    Parameters
    ----------
    start_method
        Start method to set. Must be available on the current platform
        (cf. ``multiprocessing.get_all_start_methods()``). Default is
        ``'fork'``, which inherits the parent's furnished SPICE kernel
        pool via copy-on-write.

    Returns
    -------
    str
        The start method actually set (echoes ``start_method``). Useful
        for caller-side assertion / logging.

    Raises
    ------
    ValueError
        If ``start_method`` is not in
        ``multiprocessing.get_all_start_methods()`` on the current
        platform.
    """
    available = multiprocessing.get_all_start_methods()
    if start_method not in available:
        raise ValueError(
            f"start_method={start_method!r} is not available on this "
            f"platform; available methods are {available}. On macOS and "
            "Linux 'fork' is supported; on Windows only 'spawn' is."
        )
    multiprocessing.set_start_method(start_method, force=True)
    actual = multiprocessing.get_start_method(allow_none=False)
    logger.info(
        "configure_multiprocessing_for_spice: set start method to %r "
        "(was requested: %r; available: %s)",
        actual, start_method, available,
    )
    return actual


# -----------------------------------------------------------------------------
# CloudpickleMap: scipy DE workers callable that handles closures
# -----------------------------------------------------------------------------
#
# Worker globals + helpers are defined at MODULE level so they're picklable
# via stdlib pickle (which is what multiprocessing uses to start the
# Pool's initializer + workers). Only the caller-supplied cost function
# is serialized via cloudpickle; everything else uses stdlib pickle.

_WORKER_FN: Callable[[Any], Any] | None = None


def _cloudpickle_worker_init(pickled_fn_bytes: bytes) -> None:
    """Pool initializer: cloudpickle-deserialize the cost fn into a global.

    Called once per worker at pool startup. Stores the deserialized
    function in the worker's module globals (``_WORKER_FN``) so
    repeated ``_cloudpickle_worker_call`` invocations can reference
    it without re-deserializing.
    """
    global _WORKER_FN
    _WORKER_FN = cloudpickle.loads(pickled_fn_bytes)


def _cloudpickle_worker_call(x: Any) -> Any:
    """Pool worker: invoke the cloudpickle-deserialized fn on a single arg.

    The arg ``x`` is serialized via stdlib pickle (numpy arrays and
    scalar tuples handle this fine — the closure-pickling problem is
    purely on the function side, not the args side).
    """
    if _WORKER_FN is None:
        raise RuntimeError(
            "_cloudpickle_worker_call invoked but _WORKER_FN was never "
            "initialized; ensure the Pool's initializer is "
            "_cloudpickle_worker_init"
        )
    return _WORKER_FN(x)


class CloudpickleMap:
    """Map-like callable for scipy DE workers, using cloudpickle.

    ``scipy.optimize.differential_evolution`` accepts ``workers`` as
    either an integer (in which case scipy's internal
    ``multiprocessing.Pool.map`` is used — fails on closure cost
    functions) OR a map-like callable invoked as
    ``workers(func, iterable) -> list``. This class is the latter.

    Lifecycle
    ---------
    Lazy persistent pool. The first ``__call__`` creates a fork-context
    ``multiprocessing.Pool`` initialized with the cloudpickled cost
    function (workers cache the deserialized fn in their globals).
    Repeated ``__call__``s with the same ``func`` (compared by
    ``id``) reuse the existing pool — zero per-generation overhead.
    If ``func`` changes (e.g. a different DE solver call), the pool
    is rebuilt with the new cloudpickled fn.

    Pool processes are released on ``close()`` / ``__exit__`` /
    ``__del__``. Use as a context manager when possible::

        with CloudpickleMap(n_workers=8) as cp_map:
            res = differential_evolution(
                fn, bounds, workers=cp_map, updating="deferred",
            )

    Parameters
    ----------
    n_workers
        Number of worker processes. Must be >= 1.
    start_method
        Multiprocessing start method. Default ``'fork'`` so workers
        inherit the parent's SPICE kernel pool (call
        ``configure_multiprocessing_for_spice()`` first).

    Examples
    --------
    >>> from scipy.optimize import differential_evolution
    >>> cp_map = CloudpickleMap(n_workers=8)
    >>> try:
    ...     res = differential_evolution(
    ...         fn_with_closures, bounds, workers=cp_map,
    ...         updating="deferred",
    ...     )
    ... finally:
    ...     cp_map.close()
    """

    def __init__(
        self,
        n_workers: int,
        start_method: str = DEFAULT_START_METHOD,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1; got {n_workers}")
        if start_method not in multiprocessing.get_all_start_methods():
            raise ValueError(
                f"start_method={start_method!r} not available on this "
                f"platform; got {multiprocessing.get_all_start_methods()}"
            )
        self.n_workers = int(n_workers)
        self.start_method = start_method
        self._pool: multiprocessing.pool.Pool | None = None
        self._pool_fn_id: int | None = None

    def __call__(
        self, func: Callable[[Any], Any], iterable: Iterable[Any],
    ) -> list:
        items = list(iterable)
        fn_id = id(func)
        if self._pool is None or self._pool_fn_id != fn_id:
            self._destroy_pool()
            ctx = multiprocessing.get_context(self.start_method)
            pickled_fn = cloudpickle.dumps(func)
            self._pool = ctx.Pool(
                processes=self.n_workers,
                initializer=_cloudpickle_worker_init,
                initargs=(pickled_fn,),
            )
            self._pool_fn_id = fn_id
        return self._pool.map(_cloudpickle_worker_call, items)

    def _destroy_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            finally:
                self._pool = None
                self._pool_fn_id = None

    def close(self) -> None:
        """Release worker processes. Idempotent."""
        self._destroy_pool()

    def __enter__(self) -> "CloudpickleMap":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# parallel_fd_jacobian: bounds-aware central finite-difference jacobian
# -----------------------------------------------------------------------------
#
# Companion to ``CloudpickleMap`` for the polish-only warm chain.
# scipy.optimize.minimize(method='L-BFGS-B', jac=callable) computes its
# search direction from gradient evaluations; with ``jac`` returning
# ``parallel_fd_jacobian(cost_fn, x, workers=cp_map)``, the 2N FD
# perturbations are dispatched concurrently across worker processes
# instead of serial-FD in the parent.
#
# Representative polish wall accounting at 60 s cadence:
#   single-thread polish: ~9 min/sol
#   parallel-jac polish on 12 workers: ~35-70 sec/sol


def parallel_fd_jacobian(
    cost_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    h: float = 1e-4,
    workers: Optional["CloudpickleMap"] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    f_at_x: Optional[float] = None,
) -> np.ndarray:
    """Bounds-aware second-order finite-difference jacobian, optionally parallel.

    Computes ∇cost(x) ∈ R^N. For interior axes, uses second-order
    central FD ``(f(x+h·e_i) - f(x-h·e_i)) / (2h)``, O(h²) accurate.
    For axes where ``x[i] ± h`` would leave the bound interval, uses
    second-order one-sided FD evaluated AT x (forward at lower bound,
    backward at upper), still O(h²) accurate:

        forward:  (-3 f(x) + 4 f(x+h) - f(x+2h)) / (2h)
        backward: ( 3 f(x) - 4 f(x-h) + f(x-2h)) / (2h)

    The one-sided formulas need ``f(x)``. Pass it via ``f_at_x`` to
    save a redundant evaluation (the L-BFGS-B caller has just done
    ``fun(x)``); otherwise, ``parallel_fd_jacobian`` evaluates it
    internally as one of the parallel batch entries.

    Parallelism: with ``workers`` (a :class:`CloudpickleMap`), all FD
    cost evaluations are dispatched concurrently — same machinery
    scipy DE uses for its population. With ``workers=None``, runs
    serial.

    Worker-side cost evaluations: under fork, each worker mutates its
    own copy-on-write state; the parent never sees worker-side
    side-effects. For polish-only optimization, this is the right
    behaviour — the parent's ``fun`` calls (separate from ``jac``)
    capture every iterate's cost into history; jac evals are gradient
    samples that the optimizer discards once the gradient is computed.

    Parameters
    ----------
    cost_fn
        Single-arg cost callable, ``f: R^N -> R``. Must be picklable
        (cloudpickle handles closures).
    x
        Point at which to evaluate the gradient, shape ``(N,)``.
    h
        FD step size. Default ``1e-4`` rad (≈ 0.006°), well above the
        propagator's ~1e-6 relative noise floor and small vs typical
        sail dv components (0.01-1.5 rad).
    workers
        Optional ``CloudpickleMap`` for parallel dispatch. ``None``
        (default) → serial.
    bounds
        Optional sequence of ``(lo, hi)`` tuples, length N. When None,
        always uses central FD. When provided, enforces that all
        perturbations stay inside the feasible box; switches to
        one-sided FD near a boundary.
    f_at_x
        Optional pre-computed ``cost_fn(x)``. Used for one-sided FD
        formulas when ``x[i]`` is near a bound; saved one
        evaluation when supplied. If omitted and any axis needs
        one-sided FD, ``cost_fn(x)`` is evaluated as part of the
        parallel batch.

    Returns
    -------
    np.ndarray
        Gradient ``∂cost/∂x_i``, shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``h <= 0``, if ``x`` is not 1-D non-empty, if ``bounds`` is
        provided with length != N, or if any bound interval is too
        narrow for ``2h`` (``(hi - lo) < 2h``).
    """
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    x_input = np.asarray(x, dtype=float)
    if x_input.ndim != 1 or x_input.size == 0:
        raise ValueError(
            f"x must be a non-empty 1-D array, got shape {x_input.shape}"
        )
    x_arr = x_input
    n = x_arr.size

    # Per-axis FD scheme decision: 'central', 'forward', 'backward'.
    # Default central for all when no bounds given.
    schemes: list[str] = ["central"] * n
    if bounds is not None:
        bounds_list = [tuple(b) for b in bounds]
        if len(bounds_list) != n:
            raise ValueError(
                f"bounds length must equal x size, got "
                f"len(bounds)={len(bounds_list)}, x.size={n}"
            )
        for i, (lo, hi) in enumerate(bounds_list):
            if (hi - lo) < 2.0 * h:
                raise ValueError(
                    f"bounds[{i}] interval ({hi - lo}) is narrower than "
                    f"2h ({2.0 * h}); cannot do FD without leaving "
                    f"feasible region. Reduce h."
                )
            xi = x_arr[i]
            # One-sided FD needs x ± 2h to stay feasible (uses f at x,
            # x±h, x±2h). Switch when central FD's x±h would be
            # infeasible, OR when the matching x±2h for one-sided also
            # goes infeasible (requiring the opposite side and x∓2h instead).
            if (xi - 2.0 * h) < lo:
                # Lower-side x-2h infeasible → forward (need x+h, x+2h
                # both ≤ hi, guaranteed by (hi - lo) >= 2h check above).
                schemes[i] = "forward"
            elif (xi + 2.0 * h) > hi:
                schemes[i] = "backward"
            # else: central, x±h and x±2h all feasible.

    # Build perturbation list. One-sided axes are evaluated at
    # x+h, x+2h (forward) or x-h, x-2h (backward). f(x) is needed for
    # the second-order one-sided formulas; add it to the batch
    # exactly once if any axis is one-sided AND f_at_x is None.
    needs_f_at_x = any(s != "central" for s in schemes)
    perturbations: list[np.ndarray] = []
    for i in range(n):
        scheme = schemes[i]
        if scheme == "central":
            xp = x_arr.copy(); xp[i] += h
            xm = x_arr.copy(); xm[i] -= h
            perturbations.extend([xp, xm])
        elif scheme == "forward":
            xp1 = x_arr.copy(); xp1[i] += h
            xp2 = x_arr.copy(); xp2[i] += 2.0 * h
            perturbations.extend([xp1, xp2])
        elif scheme == "backward":
            xm1 = x_arr.copy(); xm1[i] -= h
            xm2 = x_arr.copy(); xm2[i] -= 2.0 * h
            perturbations.extend([xm1, xm2])
    # Resolve f(x): use caller-supplied if present; else (when needed)
    # add x to the batch and read it back after dispatch.
    f0_idx: Optional[int] = None
    if needs_f_at_x and f_at_x is None:
        f0_idx = len(perturbations)
        perturbations.append(x_arr.copy())

    # Dispatch.
    if workers is None:
        results = [cost_fn(xp) for xp in perturbations]
    else:
        results = list(workers(cost_fn, perturbations))

    # Resolve f(x).
    if f_at_x is not None:
        f0 = float(f_at_x)
    elif f0_idx is not None:
        f0 = float(results[f0_idx])
    else:
        f0 = float("nan")  # not needed (all-central path)

    # Assemble gradient.
    grad = np.empty(n, dtype=float)
    for i in range(n):
        scheme = schemes[i]
        a = float(results[2 * i])      # f(x+h) [central/forward] or f(x-h) [backward]
        b = float(results[2 * i + 1])  # f(x-h) [central], f(x+2h) [forward], f(x-2h) [backward]
        if scheme == "central":
            grad[i] = (a - b) / (2.0 * h)
        elif scheme == "forward":
            # ∂f/∂x_i ≈ (-3 f(x) + 4 f(x+h) - f(x+2h)) / (2h)
            grad[i] = (-3.0 * f0 + 4.0 * a - b) / (2.0 * h)
        elif scheme == "backward":
            # ∂f/∂x_i ≈ ( 3 f(x) - 4 f(x-h) + f(x-2h)) / (2h)
            grad[i] = (3.0 * f0 - 4.0 * a + b) / (2.0 * h)

    return grad
