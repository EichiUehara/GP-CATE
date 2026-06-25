"""Bayesian second-stage regression on a CATE pseudo-outcome.

This module implements the two meta-learner interval estimators the
paper diagnoses:

* ``alpha = 0`` -- a Bayesian regression on the **X-Learner**
  pseudo-outcome ``D_x = Y - mu_0(X)`` (treated) / ``mu_1(X) - Y``
  (control).  This is the natural way to equip the X-Learner with a
  credible interval, and Section 3 shows it under-covers because the
  pseudo-outcome's conditional mean is ``tau - b_0``, displaced by the
  first-order nuisance bias ``b_0``.

* ``alpha = 1`` -- an inverse-variance-weighted regression on the
  **doubly-robust / orthogonal** score ``D_dr`` (Kennedy, 2020).
  Section 4 shows this does not repair the bias in the few-placebo
  regime: efficiency weighting down-weights exactly the control rows
  whose augmentation cancels the nuisance bias.

A convex blend ``D = alpha * D_dr + (1 - alpha) * D_x`` interpolates the
two; its conditional mean is ``tau - (1 - alpha) b_0``.

The second stage is a heteroskedastic Bayesian linear regression on a
feature basis ``Phi(X)``: per-row noise variance ``v(X_i)`` combining
outcome noise, nuisance-imputation noise (the empirical cross-fold
spread of the nuisance predictions), and -- when the propensity is
estimated -- propensity-estimation noise.  Everything is computed
strictly out-of-fold; the K fold posteriors are aggregated into a
consensus posterior whose variance includes the fold-to-fold spread.

Set ``known_propensity`` to a fixed ``p`` (or per-row array) to run the
known-propensity check of Section 4: the propensity carries no
estimation error, so its variance term vanishes and ``pi`` is used
exactly as given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import scipy.linalg as spla
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold

from .basis import _EPS, polynomial_basis


def _default_regressor():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)


def _default_classifier():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)


@dataclass
class _FoldPosterior:
    beta_post: np.ndarray
    Sigma_post: np.ndarray
    p_dim: int


@dataclass
class PseudoOutcomeResult:
    """Output of :meth:`PseudoOutcomeBayesLearner.predict`."""

    tau: np.ndarray
    var: np.ndarray
    per_fold_tau: np.ndarray
    per_fold_var: np.ndarray


@dataclass
class PseudoOutcomeBayesLearner:
    """Heteroskedastic Bayesian regression on a CATE pseudo-outcome.

    Parameters
    ----------
    basis_fn :
        Map ``X -> Phi (n, p)``.  Default: degree-2 polynomial.
    prior_precision :
        Gaussian prior precision ``1/lambda`` for ``beta ~ N(0, lambda I)``.
    n_folds :
        ``K`` for cross-fitting and posterior aggregation.
    propensity_clip :
        Clip an *estimated* ``pi`` to ``[c, 1-c]`` to avoid ``1/0``
        blow-ups in the DR pseudo-outcome.
    alpha :
        Blend weight between the orthogonal DR score (``alpha=1``) and
        the X-Learner pseudo-outcome (``alpha=0``).
    known_propensity :
        If not ``None``, use this propensity (scalar or per-row array)
        instead of estimating it, and drop the propensity-estimation
        variance term.  Used for the known-propensity check of Table 2.
    base_regressor, base_classifier :
        Un-fitted sklearn estimators for the outcome nuisances and the
        propensity model; cloned per fold.
    random_state :
        Seed.
    """

    basis_fn: Callable[[np.ndarray], np.ndarray] = polynomial_basis
    prior_precision: float = 1.0
    n_folds: int = 5
    propensity_clip: float = 0.05
    alpha: float = 1.0
    known_propensity: Optional[object] = None
    base_regressor: object = field(default_factory=_default_regressor)
    base_classifier: object = field(default_factory=_default_classifier)
    random_state: int = 0
    fold_posteriors: List[_FoldPosterior] = field(default_factory=list)
    consensus_beta: Optional[np.ndarray] = None
    consensus_Sigma: Optional[np.ndarray] = None

    def _clone(self, base, seed):
        m = clone(base)
        if "random_state" in m.get_params():
            m.set_params(random_state=seed)
        return m

    def fit(self, X, W, Y) -> "PseudoOutcomeBayesLearner":
        X = np.asarray(X, dtype=float)
        W = np.asarray(W, dtype=int)
        Y = np.asarray(Y, dtype=float)
        n = len(Y)
        a = float(self.alpha)
        known = self.known_propensity is not None

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        splits = list(kf.split(X))
        K = len(splits)

        # --- Pass 1: cross-fitted nuisances ------------------------------- #
        preds0 = np.full((K, n), np.nan)
        preds1 = np.full((K, n), np.nan)
        predspi = np.full((K, n), np.nan)
        fold_of = np.empty(n, dtype=int)
        valid = []
        for k, (tr_idx, te_idx) in enumerate(splits):
            fold_of[te_idx] = k
            Wtr = W[tr_idx]
            if (Wtr == 0).sum() < 3 or (Wtr == 1).sum() < 3:
                continue
            mu0 = self._clone(self.base_regressor, self.random_state + 10 * k)
            mu0.fit(X[tr_idx][Wtr == 0], Y[tr_idx][Wtr == 0])
            mu1 = self._clone(self.base_regressor, self.random_state + 10 * k + 1)
            mu1.fit(X[tr_idx][Wtr == 1], Y[tr_idx][Wtr == 1])
            preds0[k] = mu0.predict(X)
            preds1[k] = mu1.predict(X)
            if known:
                predspi[k] = np.broadcast_to(
                    np.asarray(self.known_propensity, dtype=float), (n,)
                )
            else:
                pi = self._clone(self.base_classifier, self.random_state + 10 * k + 2)
                pi.fit(X[tr_idx], Wtr)
                predspi[k] = pi.predict_proba(X)[:, 1]
            valid.append(k)
        if not valid:
            raise RuntimeError("No valid folds; check per-arm sample sizes.")

        if len(valid) > 1:
            imp_var_0 = np.nanvar(preds0[valid], axis=0, ddof=1)
            imp_var_1 = np.nanvar(preds1[valid], axis=0, ddof=1)
        else:
            imp_var_0 = np.zeros(n)
            imp_var_1 = np.zeros(n)

        rows = np.arange(n)
        mu0_oof = preds0[fold_of, rows]
        mu1_oof = preds1[fold_of, rows]
        pi_oof = predspi[fold_of, rows]
        for arr, src in ((mu0_oof, preds0), (mu1_oof, preds1), (pi_oof, predspi)):
            bad = np.isnan(arr)
            if bad.any():
                arr[bad] = np.nanmean(src[:, bad], axis=0)
        if not known:
            pi_oof = np.clip(pi_oof, self.propensity_clip, 1.0 - self.propensity_clip)

        # --- Outcome-noise variance: OUT-OF-FOLD residuals ---------------- #
        ctrl = W == 0
        treat = W == 1
        sigma2_Y_0 = float(np.var(Y[ctrl] - mu0_oof[ctrl])) if ctrl.any() else 0.0
        sigma2_Y_1 = float(np.var(Y[treat] - mu1_oof[treat])) if treat.any() else 0.0

        # --- Pseudo-outcomes and per-row variance (all out-of-fold) ------- #
        D_dr = (
            mu1_oof - mu0_oof
            + W * (Y - mu1_oof) / pi_oof
            - (1 - W) * (Y - mu0_oof) / (1.0 - pi_oof)
        )
        D_x = np.where(W == 1, Y - mu0_oof, mu1_oof - Y)
        D_blend = a * D_dr + (1.0 - a) * D_x

        c_eff = np.where(
            W == 1, a / pi_oof + (1.0 - a), a / (1.0 - pi_oof) + (1.0 - a)
        )
        outcome_var = c_eff ** 2 * np.where(W == 1, sigma2_Y_1, sigma2_Y_0)

        c_mu1 = a * (1.0 - W / pi_oof) + (1.0 - a) * (1 - W)
        c_mu0 = a * (-1.0 + (1 - W) / (1.0 - pi_oof)) - (1.0 - a) * W
        nuis_var = c_mu1 ** 2 * imp_var_1 + c_mu0 ** 2 * imp_var_0

        if known:
            prop_var = np.zeros(n)
        else:
            mu_oof = np.where(W == 1, mu1_oof, mu0_oof)
            resid = Y - mu_oof
            pi_se_sq = 1.0 / (n * pi_oof * (1.0 - pi_oof) + _EPS)
            prop_var = a ** 2 * resid ** 2 * pi_se_sq

        v_dr = np.maximum(outcome_var + nuis_var + prop_var, _EPS)

        # --- Pass 2: per-fold Bayesian heteroskedastic regression --------- #
        self.fold_posteriors = []
        for k in valid:
            te_idx = splits[k][1]
            Phi = self.basis_fn(X[te_idx])
            p = Phi.shape[1]
            V_inv = 1.0 / v_dr[te_idx]
            XtVinvX = Phi.T @ (Phi * V_inv[:, None])
            XtVinvD = Phi.T @ (V_inv * D_blend[te_idx])
            A = XtVinvX + self.prior_precision * np.eye(p)
            beta_post = spla.solve(A, XtVinvD, assume_a="pos")
            Sigma_post = np.linalg.pinv(A)
            self.fold_posteriors.append(
                _FoldPosterior(beta_post=beta_post, Sigma_post=Sigma_post, p_dim=p)
            )

        # --- Aggregate the K fold posteriors ------------------------------ #
        K_eff = len(self.fold_posteriors)
        betas = np.stack([fp.beta_post for fp in self.fold_posteriors])
        Sigmas = np.stack([fp.Sigma_post for fp in self.fold_posteriors])
        self.consensus_beta = betas.mean(axis=0)
        within = Sigmas.mean(axis=0) / K_eff
        between = np.cov(betas.T, bias=True) if K_eff > 1 else Sigmas.mean(axis=0)
        Sigma_C = within + between
        evals, evecs = np.linalg.eigh(0.5 * (Sigma_C + Sigma_C.T))
        self.consensus_Sigma = evecs @ np.diag(np.maximum(evals, 1e-8)) @ evecs.T
        return self

    def _check_fitted(self) -> None:
        if self.consensus_beta is None:
            raise RuntimeError("Learner is not fitted; call fit() before predict().")

    def predict(self, X_test) -> PseudoOutcomeResult:
        self._check_fitted()
        X_test = np.asarray(X_test, dtype=float)
        Phi = self.basis_fn(X_test)
        K = len(self.fold_posteriors)
        per_fold_tau = np.empty((K, X_test.shape[0]))
        per_fold_var = np.empty((K, X_test.shape[0]))
        for k, fp in enumerate(self.fold_posteriors):
            per_fold_tau[k] = Phi @ fp.beta_post
            per_fold_var[k] = (Phi @ fp.Sigma_post * Phi).sum(axis=1)
        tau = Phi @ self.consensus_beta
        var = (Phi @ self.consensus_Sigma * Phi).sum(axis=1)
        return PseudoOutcomeResult(
            tau=tau, var=var, per_fold_tau=per_fold_tau, per_fold_var=per_fold_var
        )

    def credible_interval(self, X_test, alpha: float = 0.05) -> tuple:
        self._check_fitted()
        out = self.predict(X_test)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        sd = np.sqrt(np.maximum(out.var, _EPS))
        return out.tau, out.tau - z * sd, out.tau + z * sd
