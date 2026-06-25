"""Unit tests for the GP-CATE estimator."""
from __future__ import annotations

import numpy as np
import pytest

from gpcate import GPCATELearner, GPCATEResult


def _synth(n, n0=None, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    tau = 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 2]
    if n0 is None:
        W = (rng.uniform(size=n) < 0.5).astype(int)
    else:
        W = np.ones(n, dtype=int)
        W[rng.choice(n, size=n0, replace=False)] = 0
    Y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(n) * 0.5 + W * tau
    return X, W, Y, tau


def test_fit_predict_shapes():
    X, W, Y, _ = _synth(300, seed=1)
    m = GPCATELearner(random_state=1).fit(X, W, Y)
    Xte = np.random.default_rng(9).standard_normal((40, 4))
    out = m.predict(Xte)
    assert isinstance(out, GPCATEResult)
    assert out.tau.shape == (40,)
    assert out.sd.shape == (40,)
    assert np.all(out.sd > 0)


def test_credible_interval_ordered_and_contains_mean():
    X, W, Y, _ = _synth(300, seed=2)
    m = GPCATELearner(random_state=2).fit(X, W, Y)
    Xte = np.random.default_rng(3).standard_normal((50, 4))
    tau, lo, hi = m.credible_interval(Xte, alpha=0.05)
    assert np.all(lo <= hi)
    assert np.all((lo <= tau) & (tau <= hi))


def test_higher_confidence_is_wider():
    X, W, Y, _ = _synth(300, seed=4)
    m = GPCATELearner(random_state=4).fit(X, W, Y)
    Xte = np.random.default_rng(5).standard_normal((40, 4))
    _, lo90, hi90 = m.credible_interval(Xte, alpha=0.10)
    _, lo99, hi99 = m.credible_interval(Xte, alpha=0.01)
    assert np.mean(hi99 - lo99) > np.mean(hi90 - lo90)


def test_few_placebo_covers_on_average():
    """With a correct model the GP interval should roughly cover."""
    X, W, Y, tau = _synth(530, n0=30, seed=6)
    assert (W == 0).sum() == 30
    m = GPCATELearner(random_state=6).fit(X, W, Y)
    tau_hat, lo, hi = m.credible_interval(X)
    assert np.all(np.isfinite(tau_hat))
    assert np.mean((lo <= tau) & (tau <= hi)) >= 0.8


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GPCATELearner().predict(np.zeros((3, 4)))


def test_degenerate_arm_raises():
    X = np.random.default_rng(0).standard_normal((10, 4))
    W = np.ones(10, dtype=int)              # no control units
    Y = np.random.default_rng(1).standard_normal(10)
    with pytest.raises(ValueError):
        GPCATELearner().fit(X, W, Y)


def _mean_width(learner, Xte):
    _, lo, hi = learner.credible_interval(Xte)
    return float(np.mean(hi - lo))


def test_sparse_control_arm_has_wider_intervals():
    """The few-placebo arm should make intervals wider than a balanced fit."""
    Xb, Wb, Yb, _ = _synth(530, seed=7)                 # balanced
    Xf, Wf, Yf, _ = _synth(530, n0=30, seed=7)          # few-placebo
    Xte = np.random.default_rng(8).standard_normal((60, 4))
    wb = _mean_width(GPCATELearner(random_state=7).fit(Xb, Wb, Yb), Xte)
    wf = _mean_width(GPCATELearner(random_state=7).fit(Xf, Wf, Yf), Xte)
    assert wf > wb
