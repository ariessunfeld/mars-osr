"""Unit tests for optional early stopping in ``PiecewiseCruiseNLP``.

The tests exercise the ``intermediate`` callback directly without cyipopt or
propagation. Returning ``False`` stops the solve; returning ``None`` continues.
With ``patience=None``, the callback never aborts.
"""
import numpy as np

from reflectors.cruise_solve import PiecewiseCruiseNLP


def _nlp(**es):
    # defect / cp_map are unused by intermediate(); use trivial stand-ins.
    return PiecewiseCruiseNLP(
        defect=lambda x: np.zeros(6), n_vars=33, N=16, cp_map=map, **es
    )


def _call(nlp, it, inf_pr):
    # intermediate(self, alg_mod, it, obj, inf_pr, inf_du, mu, dn, rg, a_du, a_pr, ls)
    return nlp.intermediate(0, it, 0.0, float(inf_pr), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def test_default_off_never_aborts():
    """patience=None (default) -> intermediate returns None for any sequence."""
    nlp = _nlp()
    assert nlp.early_stop_patience is None
    for it in range(40):
        assert _call(nlp, it, 0.3) is None  # flat plateau, but default never stops


def test_plateau_aborts_after_patience_and_warmup():
    """A flat inf_pr plateau -> abort once it>=min_iters AND stall>=patience."""
    nlp = _nlp(early_stop_patience=5, early_stop_rtol=0.02, early_stop_min_iters=10)
    results = [_call(nlp, it, 0.3) for it in range(11)]
    # No abort during the warm-up even though the plateau started immediately.
    assert all(r is None for r in results[:10])
    # First abort exactly at it=10 (warm-up satisfied, stall has reached patience).
    assert results[10] is False


def test_descending_never_aborts():
    """A geometrically descending inf_pr keeps resetting the stall -> never abort."""
    nlp = _nlp(early_stop_patience=5, early_stop_rtol=0.02, early_stop_min_iters=10)
    for it in range(30):
        assert _call(nlp, it, 0.3 * (0.5 ** it)) is None


def test_slow_but_real_descent_survives_then_plateau_aborts():
    """Descent just above rtol survives; if it then flatlines, abort fires."""
    nlp = _nlp(early_stop_patience=4, early_stop_rtol=0.02, early_stop_min_iters=8)
    val = 1.0
    # 12 iterations each dropping 5% (> rtol) -> stall stays 0, no abort.
    for it in range(12):
        val *= 0.95
        assert _call(nlp, it, val) is None
    # Now flatline: stall climbs; abort once stall>=patience (warm-up already met).
    aborted_at = None
    for it in range(12, 24):
        if _call(nlp, it, val) is False:
            aborted_at = it
            break
    assert aborted_at is not None
