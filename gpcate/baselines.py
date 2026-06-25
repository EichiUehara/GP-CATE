"""Interval-producing baselines: Causal Forest and BART.

The two most common CATE estimators that ship uncertainty intervals,
run on the paper's protocol for the calibration comparison
(Tables 3-5):

* ``causal_forest`` -- the causal-forest DML estimator with honest
  asymptotic confidence intervals (Wager & Athey, 2018; Athey et al.,
  2019), via ``econml.dml.CausalForestDML`` and ``effect_interval``.
* ``bart_slearner`` -- an S-learner BART on ``[X, W]`` (Chipman et al.,
  2010; Hill, 2011); the CATE posterior at ``x`` is the posterior of
  ``f(x, 1) - f(x, 0)`` over the tree-ensemble draws, via ``bartz``.

``econml`` and ``bartz`` are heavy optional dependencies, imported
lazily so that ``import gpcate`` works without them.  Install with
``pip install -e ".[baselines]"``.
"""
from __future__ import annotations

import numpy as np

# BART sampler settings (Appendix B: 60 trees, 200 draws, 100 burn-in).
NDPOST, NSKIP, NTREE = 200, 100, 60


def causal_forest(X, W, Y, Xte, alpha: float = 0.05, n_estimators: int = 500,
                  random_state: int = 0):
    """Causal-forest DML with honest confidence intervals.

    Returns ``(tau, lower, upper)`` for the ``(1 - alpha)`` interval.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=150, max_depth=3),
        model_t=GradientBoostingClassifier(n_estimators=150, max_depth=3),
        discrete_treatment=True,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    cf.fit(Y, W, X=X)
    tau = cf.effect(Xte)
    lo, hi = cf.effect_interval(Xte, alpha=alpha)
    return tau, lo, hi


def bart_slearner(X, W, Y, Xte, alpha: float = 0.05, random_state: int = 0):
    """S-learner BART: one BART on ``[X, W]``; the CATE posterior is the
    posterior of ``f(x, 1) - f(x, 0)`` over the ensemble draws.

    Returns ``(tau, lower, upper)`` for the ``(1 - alpha)`` interval.
    """
    from bartz.BART import gbart

    Xtr = np.column_stack([X, W])
    x1 = np.column_stack([Xte, np.ones(len(Xte))])
    x0 = np.column_stack([Xte, np.zeros(len(Xte))])
    xte = np.vstack([x1, x0])
    m = gbart(
        Xtr.T, Y, x_test=xte.T,
        ndpost=NDPOST, nskip=NSKIP, ntree=NTREE, seed=random_state,
    )
    yt = np.asarray(m.yhat_test)            # (ndpost, 2 * n_test)
    n = len(Xte)
    draws = yt[:, :n] - yt[:, n:]           # (ndpost, n_test) tau draws
    tau = draws.mean(0)
    lo = np.quantile(draws, alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    return tau, lo, hi
