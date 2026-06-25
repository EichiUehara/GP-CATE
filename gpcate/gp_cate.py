"""GP-CATE: calibrated CATE inference in the few-placebo regime.

GP-CATE models each treatment arm's outcome surface with a Gaussian
process and reads the conditional average treatment effect (CATE) off
the difference of the two arm posteriors.  This is the estimator that
the paper's diagnosis (Section 3) and negative result (Section 4)
single out.

Why a Gaussian process, and not a point estimate plus a post-hoc
variance.  The non-orthogonal Bayesian X-Learner (Section 3) and the
orthogonal doubly-robust score (Section 4) both attach uncertainty to
a *point estimate* of the hard placebo nuisance ``mu_0`` -- a
regularised tree fit on the few control units -- and a point estimate's
regularisation bias is invisible to any variance computed after the
fact.  GP-CATE removes the defect at its source: the GP posterior over
the latent function ``mu_w`` is calibrated by construction -- wide
where that arm's units are sparse, narrow where they are dense -- so
the placebo arm's small sample enters the posterior directly as
uncertainty rather than as an unmodelled bias.

For arm ``w`` we fit a GP with the isotropic kernel

    k_w(x, x') = sigma_w^2 * exp(-||x - x'||^2 / (2 * ell_w^2))
                 + eta_w^2 * 1[x = x'],

a constant-times-RBF-plus-white-noise kernel whose hyperparameters
``(sigma_w, ell_w, eta_w)`` are set by marginal likelihood (empirical
Bayes).  The length scale is *isotropic*: per-dimension (ARD) length
scales over-fit a small control arm and worsen coverage (Appendix B).

The CATE posterior is the difference of the two arm posteriors,

    tau(x) | D  ~  N( m_1(x) - m_0(x),  s_1(x)^2 + s_0(x)^2 ),

with ``m_w, s_w`` the GP posterior mean and standard deviation of the
latent function (the observation-noise term removed, so ``s_w`` is
uncertainty about the function, not about a future outcome).  This is a
Bayesian T-learner with GP arms (close to Alaa & van der Schaar, 2017,
without the cross-arm coupling); the paper's contribution is the
analysis that singles it out, not the estimator itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


@dataclass
class GPArm:
    """A fitted GP for one arm, exposing the latent-function posterior."""

    gp: GaussianProcessRegressor
    y_mean: float
    noise: float

    def posterior(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior ``(mean, std)`` over the latent function ``mu_w``.

        The white-noise term is removed, so the returned standard
        deviation is uncertainty about the function, not about a new
        outcome.
        """
        mean, std = self.gp.predict(X, return_std=True)
        fvar = np.maximum(std ** 2 - self.noise, 1e-8)
        return mean + self.y_mean, np.sqrt(fvar)

    @property
    def prior_function_variance(self) -> float:
        """Fitted prior function variance (the ConstantKernel amplitude
        ``sigma_w^2``).  Used by the posterior-contraction diagnostics."""
        return float(self.gp.kernel_.k1.k1.constant_value)


@dataclass
class GPCATEResult:
    """Output of :meth:`GPCATELearner.predict`."""

    tau: np.ndarray      # posterior mean of tau(x)
    lower: np.ndarray    # lower credible bound
    upper: np.ndarray    # upper credible bound
    sd: np.ndarray       # posterior standard deviation of tau(x)


@dataclass
class GPCATELearner:
    """GP-CATE: a Bayesian T-learner with Gaussian-process arms.

    Parameters
    ----------
    amplitude_bounds, lengthscale_bounds, noise_bounds :
        Bounds for the empirical-Bayes optimisation of the kernel
        ``ConstantKernel * RBF + WhiteKernel``.  The RBF length scale is
        isotropic; per-dimension (ARD) length scales over-fit a small
        control arm and are deliberately avoided.
    n_restarts :
        Restarts for the marginal-likelihood optimiser (the paper uses
        two restarts).
    random_state :
        Seed for the optimiser restarts.
    """

    amplitude_bounds: Tuple[float, float] = (1e-2, 1e3)
    lengthscale_bounds: Tuple[float, float] = (1e-1, 1e3)
    noise_bounds: Tuple[float, float] = (1e-3, 1e1)
    n_restarts: int = 2
    random_state: int = 0
    arm0_: Optional[GPArm] = field(default=None)
    arm1_: Optional[GPArm] = field(default=None)

    def _fit_arm(self, X: np.ndarray, Y: np.ndarray) -> GPArm:
        ym = float(Y.mean())
        d = X.shape[1]
        kernel = (
            ConstantKernel(1.0, self.amplitude_bounds)
            * RBF(np.sqrt(d), self.lengthscale_bounds)
            + WhiteKernel(0.25, self.noise_bounds)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
        ).fit(X, Y - ym)
        return GPArm(gp=gp, y_mean=ym, noise=gp.kernel_.k2.noise_level)

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> "GPCATELearner":
        X = np.asarray(X, dtype=float)
        W = np.asarray(W, dtype=int)
        Y = np.asarray(Y, dtype=float)
        if (W == 0).sum() < 2 or (W == 1).sum() < 2:
            raise ValueError("each arm needs at least 2 units")
        self.arm1_ = self._fit_arm(X[W == 1], Y[W == 1])
        self.arm0_ = self._fit_arm(X[W == 0], Y[W == 0])
        return self

    def _check_fitted(self) -> None:
        if self.arm0_ is None:
            raise RuntimeError("GPCATELearner is not fitted; call fit() first.")

    def predict(self, X_test: np.ndarray, alpha: float = 0.05) -> GPCATEResult:
        """Posterior mean and ``(1 - alpha)`` credible interval for tau(x)."""
        self._check_fitted()
        X_test = np.asarray(X_test, dtype=float)
        m1, s1 = self.arm1_.posterior(X_test)
        m0, s0 = self.arm0_.posterior(X_test)
        tau = m1 - m0
        sd = np.sqrt(s1 ** 2 + s0 ** 2)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return GPCATEResult(tau=tau, lower=tau - z * sd, upper=tau + z * sd, sd=sd)

    def credible_interval(self, X_test: np.ndarray, alpha: float = 0.05) -> tuple:
        """Convenience wrapper returning ``(tau, lower, upper)``."""
        out = self.predict(X_test, alpha=alpha)
        return out.tau, out.lower, out.upper
