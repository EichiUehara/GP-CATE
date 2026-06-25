"""Data-generating processes and the IHDP benchmark.

Three designs, exactly as specified in the paper (Section 5 and
Appendix B):

* ``linear``    -- X ~ N(0, I_4); W | X ~ Bernoulli(logistic(0.4 X_0));
                   Y = 0.5 X_0 + 0.3 X_1 + 0.5 xi + W tau(X);
                   tau = 1 + 0.3 X_0 - 0.2 X_2.
* ``non-linear``-- a two-covariate propensity, an interaction-laden
                   response surface, and tau = 1 + sin(1.4 X_0) - 0.3 X_2^2.
* ``IHDP``      -- the semi-synthetic benchmark of Hill (2011): 25 real
                   covariates, 747 units, a simulated response surface
                   with known per-unit tau.

The few-placebo regime ``N0 << N1`` is produced for the synthetic
designs by stratified sampling to ``N0`` controls and ``N1 = 500``
treated; for IHDP by keeping all treated units and sub-sampling the
control arm to ``N0``.

``randomised_design`` draws ``W ~ Bernoulli(p)`` with ``p`` fixed
(``pi == p`` known by design), used for the known-propensity check of
Section 4 / Table 2.
"""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np

# Default location of the committed IHDP CSVs (data/ihdp/ihdp_*.csv).
IHDP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ihdp")


# --------------------------------------------------------------------------- #
# Linear design                                                               #
# --------------------------------------------------------------------------- #
def linear_design(n: int, rng: np.random.Generator):
    """Linear synthetic design.  Returns ``(X, W, Y, tau)``."""
    X = rng.standard_normal((n, 4))
    pi = 1.0 / (1.0 + np.exp(-0.4 * X[:, 0]))
    W = (rng.uniform(size=n) < pi).astype(int)
    tau = 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 2]
    Y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(n) * 0.5 + W * tau
    return X, W, Y, tau


# --------------------------------------------------------------------------- #
# Non-linear design                                                           #
# --------------------------------------------------------------------------- #
def nonlinear_design(n: int, rng: np.random.Generator):
    """Non-linear synthetic design.  Returns ``(X, W, Y, tau)``."""
    X = rng.standard_normal((n, 4))
    pi = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0] + 0.3 * X[:, 1]))
    W = (rng.uniform(size=n) < pi).astype(int)
    tau = 1.0 + np.sin(1.4 * X[:, 0]) - 0.3 * X[:, 2] ** 2
    mu0 = 0.5 * X[:, 0] + 0.4 * X[:, 0] * X[:, 1] - 0.3 * np.sin(1.5 * X[:, 2])
    Y = mu0 + 0.5 * rng.standard_normal(n) + W * tau
    return X, W, Y, tau


# --------------------------------------------------------------------------- #
# Randomised design (known propensity)                                        #
# --------------------------------------------------------------------------- #
def randomised_design(n: int, p: float, rng: np.random.Generator):
    """Randomised unequal allocation ``W ~ Bernoulli(p)``, ``pi == p`` known.

    Same response/CATE surface as the linear design but with the
    treatment assigned at random, so the propensity carries no
    estimation error.  Used for the known-propensity row of Table 2.
    """
    X = rng.standard_normal((n, 4))
    W = (rng.uniform(size=n) < p).astype(int)
    tau = 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 2]
    Y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(n) * 0.5 + W * tau
    return X, W, Y, tau


# --------------------------------------------------------------------------- #
# Stratified few-placebo sampling                                             #
# --------------------------------------------------------------------------- #
def stratified_few_placebo(design, n0: int, n1: int, rng: np.random.Generator):
    """Draw from ``design`` until ``n0`` controls and ``n1`` treated are
    available, then keep the first ``n0`` / ``n1`` of each arm.

    ``design`` is a callable ``(n, rng) -> (X, W, Y, tau)`` such as
    :func:`linear_design` or :func:`nonlinear_design`.  Returns
    ``(X, W, Y)``.
    """
    XL: List[np.ndarray] = []
    WL: List[np.ndarray] = []
    YL: List[np.ndarray] = []
    nh = (0, 0)
    while nh[0] < n0 or nh[1] < n1:
        X, W, Y, _ = design(n0 + n1, rng)
        XL.append(X)
        WL.append(W)
        YL.append(Y)
        nh = (
            sum(int((w == 0).sum()) for w in WL),
            sum(int((w == 1).sum()) for w in WL),
        )
    Xa = np.vstack(XL)
    Wa = np.concatenate(WL)
    Ya = np.concatenate(YL)
    i0 = np.where(Wa == 0)[0][:n0]
    i1 = np.where(Wa == 1)[0][:n1]
    idx = np.concatenate([i0, i1])
    return Xa[idx], Wa[idx], Ya[idx]


# --------------------------------------------------------------------------- #
# IHDP                                                                        #
# --------------------------------------------------------------------------- #
def ihdp_files(data_dir: str = IHDP_DIR) -> List[str]:
    """Sorted list of the committed IHDP replication CSVs."""
    files = sorted(glob.glob(os.path.join(data_dir, "ihdp_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No IHDP CSVs found in {data_dir!r}. Expected ihdp_*.csv files."
        )
    return files


def load_ihdp(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one IHDP replication.

    Columns are ``[W, Y, y_cf, mu0, mu1, X_1, ..., X_25]``.  Covariates
    are standardised (IHDP features have very different scales).
    Returns ``(X, W, Y, tau)`` with ``tau = mu1 - mu0`` the known
    per-unit effect.
    """
    arr = np.loadtxt(path, delimiter=",")
    W = arr[:, 0].astype(int)
    Y = arr[:, 1].astype(float)
    mu0, mu1 = arr[:, 3], arr[:, 4]
    X = arr[:, 5:].astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, W, Y, (mu1 - mu0)


def ihdp_few_placebo_split(X, W, Y, n0: int, rng: np.random.Generator):
    """Keep all treated units; randomly sub-sample ``n0`` controls.

    Returns the sub-sampled ``(X, W, Y)``; coverage is evaluated against
    the known ``tau`` at all original covariate points.
    """
    treat = np.where(W == 1)[0]
    ctrl = np.where(W == 0)[0]
    keep_c = rng.choice(ctrl, size=min(n0, len(ctrl)), replace=False)
    idx = np.concatenate([treat, keep_c])
    return X[idx], W[idx], Y[idx]


# Registry used by the experiment scripts.
DESIGNS = {"linear": linear_design, "nonlinear": nonlinear_design}
