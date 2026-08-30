"""Tests for reflectors.parallel.

Covers:
- ``parallel_fd_jacobian``: bounds-aware second-order FD; serial and
  parallel-via-CloudpickleMap equivalence; analytical-quadratic
  gradient cross-check; bound-clipped one-sided FD; input validation.

The ``CloudpickleMap`` lifecycle and pool reuse are exercised indirectly via
the parallel-versus-serial test case here.

Unit tests use serial execution by default; one two-worker test exercises the
parallel dispatch path.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflectors.parallel import (
    CloudpickleMap,
    configure_multiprocessing_for_spice,
    parallel_fd_jacobian,
)


# -----------------------------------------------------------------------------
# Module-level fixtures (cost functions usable by both serial and parallel
# paths — module-level so they pickle cleanly under fork).
# -----------------------------------------------------------------------------


def _quadratic_cost(x: np.ndarray) -> float:
    """Simple unbounded quadratic: f(x) = sum(x_i^2). ∇f = 2x."""
    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr * arr))


def _shifted_quadratic_cost(x: np.ndarray) -> float:
    """Shifted quadratic: f(x) = sum((x_i - i)^2). ∇f_i = 2(x_i - i)."""
    arr = np.asarray(x, dtype=float)
    targets = np.arange(arr.size, dtype=float)
    return float(np.sum((arr - targets) ** 2))


# -----------------------------------------------------------------------------
# parallel_fd_jacobian: input validation
# -----------------------------------------------------------------------------


class TestParallelFDJacobianInputValidation:
    def test_zero_h_raises(self):
        with pytest.raises(ValueError, match="h must be positive"):
            parallel_fd_jacobian(_quadratic_cost, np.array([0.5]), h=0.0)

    def test_negative_h_raises(self):
        with pytest.raises(ValueError, match="h must be positive"):
            parallel_fd_jacobian(_quadratic_cost, np.array([0.5]), h=-1e-4)

    def test_empty_x_raises(self):
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            parallel_fd_jacobian(_quadratic_cost, np.array([]))

    def test_2d_x_raises(self):
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            parallel_fd_jacobian(_quadratic_cost, np.array([[1.0, 2.0]]))

    def test_bounds_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="bounds length must equal"):
            parallel_fd_jacobian(
                _quadratic_cost, np.array([0.5, 0.5]),
                bounds=[(-1.0, 1.0)],
            )

    def test_bounds_too_narrow_raises(self):
        # h=0.1, interval=0.1 < 2h=0.2
        with pytest.raises(ValueError, match="narrower than"):
            parallel_fd_jacobian(
                _quadratic_cost, np.array([0.0]),
                h=0.1, bounds=[(-0.05, 0.05)],
            )


# -----------------------------------------------------------------------------
# parallel_fd_jacobian: analytical gradient correctness
# -----------------------------------------------------------------------------


class TestParallelFDJacobianAnalytical:
    def test_quadratic_gradient_at_origin(self):
        """∇(sum x_i^2) = 2x; at x=0 gradient is 0."""
        x = np.zeros(5)
        grad = parallel_fd_jacobian(_quadratic_cost, x, h=1e-3)
        # Central FD on a quadratic: bias-free except for floating-point.
        np.testing.assert_allclose(grad, np.zeros(5), atol=1e-9)

    def test_quadratic_gradient_at_nonzero(self):
        """∇(sum x_i^2) at x = (1, 2, 3, -4) is (2, 4, 6, -8)."""
        x = np.array([1.0, 2.0, 3.0, -4.0])
        grad = parallel_fd_jacobian(_quadratic_cost, x, h=1e-3)
        expected = 2.0 * x
        np.testing.assert_allclose(grad, expected, atol=1e-8)

    def test_shifted_quadratic_gradient(self):
        """∇(sum (x_i - i)^2) at x = (0, 1, 2, 3) is (0, 0, 0, 0)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        grad = parallel_fd_jacobian(_shifted_quadratic_cost, x, h=1e-3)
        np.testing.assert_allclose(grad, np.zeros(4), atol=1e-9)

    def test_shifted_quadratic_gradient_offset(self):
        """∇(sum (x_i - i)^2) at x = (0, 0, 0, 0) is (0, -2, -4, -6)."""
        x = np.zeros(4)
        grad = parallel_fd_jacobian(_shifted_quadratic_cost, x, h=1e-3)
        expected = -2.0 * np.arange(4, dtype=float)
        np.testing.assert_allclose(grad, expected, atol=1e-9)


# -----------------------------------------------------------------------------
# parallel_fd_jacobian: bound-clipped one-sided FD
# -----------------------------------------------------------------------------


class TestParallelFDJacobianBoundClipped:
    def test_at_lower_bound_uses_forward_fd(self):
        """At x[i] = lo, central FD would go infeasible — switch to forward.

        Quadratic (x - 0.5)^2: gradient at x=0 is -1.0. With bounds
        [(0.0, 10.0)], x=0 is at the lower bound; forward FD using
        f(x), f(x+h), f(x+2h) should still give -1.0 (analytical
        truncation error 0 for a quadratic).
        """
        cost = lambda x: float((x[0] - 0.5) ** 2)
        x = np.array([0.0])
        grad = parallel_fd_jacobian(
            cost, x, h=1e-3, bounds=[(0.0, 10.0)],
        )
        np.testing.assert_allclose(grad, np.array([-1.0]), atol=1e-8)

    def test_at_upper_bound_uses_backward_fd(self):
        """At x[i] = hi, switch to backward FD.

        f(x) = (x - 0.5)^2. At x = 10, gradient = 2*(10-0.5) = 19.0.
        """
        cost = lambda x: float((x[0] - 0.5) ** 2)
        x = np.array([10.0])
        grad = parallel_fd_jacobian(
            cost, x, h=1e-3, bounds=[(0.0, 10.0)],
        )
        np.testing.assert_allclose(grad, np.array([19.0]), atol=1e-8)

    def test_with_f_at_x_supplied_avoids_extra_eval(self):
        """f_at_x parameter saves one cost evaluation when bound-clipped.

        Verify by checking that the gradient matches whether or not
        f_at_x is supplied (results should be identical).
        """
        cost = lambda x: float((x[0] - 0.5) ** 2)
        x = np.array([0.0])
        # Without f_at_x:
        grad_a = parallel_fd_jacobian(
            cost, x, h=1e-3, bounds=[(0.0, 10.0)],
        )
        # With f_at_x:
        grad_b = parallel_fd_jacobian(
            cost, x, h=1e-3, bounds=[(0.0, 10.0)], f_at_x=cost(x),
        )
        np.testing.assert_allclose(grad_a, grad_b, atol=1e-12)

    def test_mixed_interior_and_clipped(self):
        """A 3-D problem: one axis interior, one at lower, one at upper."""
        cost = lambda x: float(
            (x[0] - 0.5) ** 2 + (x[1] - 0.3) ** 2 + (x[2] - 0.7) ** 2
        )
        x = np.array([0.0, 0.5, 1.0])  # axis 0 at lo=0, axis 2 at hi=1
        grad = parallel_fd_jacobian(
            cost, x, h=1e-3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )
        # ∇f_i = 2(x_i - target)
        expected = np.array([2.0 * (0.0 - 0.5), 2.0 * (0.5 - 0.3), 2.0 * (1.0 - 0.7)])
        np.testing.assert_allclose(grad, expected, atol=1e-8)


# -----------------------------------------------------------------------------
# parallel_fd_jacobian: serial-vs-parallel equivalence
# -----------------------------------------------------------------------------


class TestParallelFDJacobianParallelDispatch:
    def test_serial_matches_parallel_2workers(self):
        """Serial (workers=None) and 2-worker CloudpickleMap give same gradient.

        Uses a small 2-worker pool to keep test wall low. Forces fork
        start method (configure_multiprocessing_for_spice no-op effect
        when already fork; on macOS this is required because default is
        spawn).
        """
        configure_multiprocessing_for_spice()
        x = np.array([0.5, 1.5, -2.0, 3.0])
        grad_serial = parallel_fd_jacobian(
            _shifted_quadratic_cost, x, h=1e-3, workers=None,
        )
        cp = CloudpickleMap(n_workers=2)
        try:
            grad_parallel = parallel_fd_jacobian(
                _shifted_quadratic_cost, x, h=1e-3, workers=cp,
            )
        finally:
            cp.close()
        np.testing.assert_allclose(grad_parallel, grad_serial, atol=1e-12)

    def test_serial_matches_parallel_with_bounds(self):
        """Equivalence holds for bound-clipped paths too."""
        configure_multiprocessing_for_spice()
        x = np.array([0.0, 0.5, 1.0])  # axes 0, 2 are bound-clipped
        cost = _shifted_quadratic_cost
        grad_serial = parallel_fd_jacobian(
            cost, x, h=1e-3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )
        cp = CloudpickleMap(n_workers=2)
        try:
            grad_parallel = parallel_fd_jacobian(
                cost, x, h=1e-3, workers=cp,
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            )
        finally:
            cp.close()
        np.testing.assert_allclose(grad_parallel, grad_serial, atol=1e-12)
