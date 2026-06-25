"""Unit tests for the pseudo-outcome learner and the meta-learners."""
from __future__ import annotations

import numpy as np
import pytest

from gpcate.datasets import linear_design, stratified_few_placebo
from gpcate.meta_learners import META_LEARNERS
from gpcate.pseudo_outcome import PseudoOutcomeBayesLearner


def _data(seed=0, n0=100):
    rng = np.random.default_rng(seed)
    X, W, Y = stratified_few_placebo(linear_design, n0, 300, rng)
    Xte, _, _, tau = linear_design(400, rng)
    return X, W, Y, Xte, tau


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_pseudo_outcome_fit_predict(alpha):
    X, W, Y, Xte, tau = _data(seed=1)
    m = PseudoOutcomeBayesLearner(alpha=alpha, random_state=1).fit(X, W, Y)
    t, lo, hi = m.credible_interval(Xte, alpha=0.05)
    assert t.shape == lo.shape == hi.shape == (len(Xte),)
    assert np.all(lo <= hi)
    # a sane estimator should be in the right ballpark (true tau ~ 1)
    assert abs(float(np.mean(t)) - 1.0) < 0.6


def test_known_propensity_runs_and_is_sane():
    """The known-propensity mode (Table 2) should fit without estimating a
    propensity model and still produce valid, finite intervals."""
    X, W, Y, Xte, _ = _data(seed=2)
    kn = PseudoOutcomeBayesLearner(
        alpha=1.0, known_propensity=float((W == 1).mean()), random_state=2
    ).fit(X, W, Y)
    t, lo, hi = kn.credible_interval(Xte)
    assert np.all(np.isfinite(t)) and np.all(lo <= hi)
    assert abs(float(np.mean(t)) - 1.0) < 0.6


def test_known_propensity_accepts_per_row_array():
    """A per-row propensity array is accepted (true logistic propensity)."""
    X, W, Y, Xte, _ = _data(seed=5)
    pi = 1.0 / (1.0 + np.exp(-0.4 * X[:, 0]))
    kn = PseudoOutcomeBayesLearner(
        alpha=1.0, known_propensity=pi, random_state=5
    ).fit(X, W, Y)
    t, lo, hi = kn.credible_interval(Xte)
    assert np.all(np.isfinite(t)) and np.all(lo <= hi)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        PseudoOutcomeBayesLearner().predict(np.zeros((3, 4)))


@pytest.mark.parametrize("name", list(META_LEARNERS))
def test_meta_learners_run(name):
    X, W, Y, Xte, tau = _data(seed=3)
    t = META_LEARNERS[name](X, W, Y, Xte, 3)
    assert t.shape == (len(Xte),)
    assert np.all(np.isfinite(t))
