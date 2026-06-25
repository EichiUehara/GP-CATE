"""Point-estimate CATE meta-learners (for the PEHE comparison, Table 5).

S-, T-, X-, and DR-Learner, all sharing the same gradient-boosted
nuisances (depth 3, 200 estimators) and -- for X and DR -- a degree-2
polynomial second stage, so a PEHE difference isolates the
pseudo-outcome rather than the regressor family.  These produce point
estimates only; the interval comparison (GP-CATE vs Causal Forest vs
BART) lives in :mod:`gpcate.baselines` and :mod:`gpcate.gp_cate`.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .basis import polynomial_basis


def _gbr(seed):
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )


def _gbc(seed):
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )


def cross_fit_nuisances(X, W, Y, seed, n_folds=5, clip=0.05):
    """K-fold out-of-fold ``mu0, mu1, pi`` predictions at every row."""
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mu0 = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)
    pi = np.full(n, np.nan)
    for k, (tr, te) in enumerate(kf.split(X)):
        Wtr = W[tr]
        if (Wtr == 0).sum() < 3 or (Wtr == 1).sum() < 3:
            continue
        m0 = _gbr(seed + 10 * k).fit(X[tr][Wtr == 0], Y[tr][Wtr == 0])
        m1 = _gbr(seed + 10 * k + 1).fit(X[tr][Wtr == 1], Y[tr][Wtr == 1])
        pc = _gbc(seed + 10 * k + 2).fit(X[tr], Wtr)
        mu0[te] = m0.predict(X[te])
        mu1[te] = m1.predict(X[te])
        pi[te] = pc.predict_proba(X[te])[:, 1]
    for arr in (mu0, mu1, pi):
        bad = np.isnan(arr)
        if bad.any():
            arr[bad] = np.nanmean(arr)
    pi = np.clip(pi, clip, 1.0 - clip)
    return mu0, mu1, pi


def s_learner(X, W, Y, Xte, seed):
    XW = np.column_stack([X, W])
    m = _gbr(seed).fit(XW, Y)
    t1 = m.predict(np.column_stack([Xte, np.ones(len(Xte))]))
    t0 = m.predict(np.column_stack([Xte, np.zeros(len(Xte))]))
    return t1 - t0


def t_learner(X, W, Y, Xte, seed):
    m0 = _gbr(seed).fit(X[W == 0], Y[W == 0])
    m1 = _gbr(seed + 1).fit(X[W == 1], Y[W == 1])
    return m1.predict(Xte) - m0.predict(Xte)


def x_learner(X, W, Y, Xte, seed):
    """Kunzel et al. (2019): cross-fitted nuisances, per-arm
    pseudo-outcome regressions, propensity-weighted blend."""
    mu0, mu1, _ = cross_fit_nuisances(X, W, Y, seed)
    d1 = Y[W == 1] - mu0[W == 1]
    d0 = mu1[W == 0] - Y[W == 0]
    Phi = polynomial_basis(X)
    Pte = polynomial_basis(Xte)
    r1 = Ridge(alpha=1.0).fit(Phi[W == 1], d1)
    r0 = Ridge(alpha=1.0).fit(Phi[W == 0], d0)
    e = np.clip(
        _gbc(seed).fit(X, W).predict_proba(Xte)[:, 1], 0.05, 0.95
    )
    return e * r0.predict(Pte) + (1.0 - e) * r1.predict(Pte)


def dr_learner(X, W, Y, Xte, seed):
    """Kennedy (2020): cross-fitted DR pseudo-outcome, unweighted ridge
    second stage on the degree-2 polynomial basis."""
    mu0, mu1, pi = cross_fit_nuisances(X, W, Y, seed)
    d_dr = mu1 - mu0 + W * (Y - mu1) / pi - (1 - W) * (Y - mu0) / (1.0 - pi)
    Phi = polynomial_basis(X)
    Pte = polynomial_basis(Xte)
    return Ridge(alpha=1.0).fit(Phi, d_dr).predict(Pte)


META_LEARNERS = {
    "s_learner": s_learner,
    "t_learner": t_learner,
    "x_learner": x_learner,
    "dr_learner": dr_learner,
}
