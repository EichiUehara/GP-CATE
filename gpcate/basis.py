"""Feature bases for the pseudo-outcome second-stage regressions.

The diagnosis (Section 3) and the negative result (Section 4) regress a
pseudo-outcome onto a feature basis ``Phi(X)``.  The synthetic designs
use a degree-2 polynomial basis (linear design) or Random Fourier
Features (non-linear design); IHDP uses a linear basis.  Sharing one
basis across the compared estimators isolates the pseudo-outcome from
the regressor family.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def polynomial_basis(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial basis with intercept and pairwise products.

    For ``X`` of shape ``(n, d)``, at ``degree=2`` returns
    ``[1, x_1, ..., x_d, x_1*x_2, ..., x_{d-1}*x_d, x_1^2, ..., x_d^2]``.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    cols = [np.ones(n)]
    cols.extend(X[:, j] for j in range(d))
    if degree >= 2:
        for j in range(d):
            for k in range(j, d):
                cols.append(X[:, j] * X[:, k])
    if degree >= 3:
        for j in range(d):
            cols.append(X[:, j] ** 3)
    if degree >= 4:
        for j in range(d):
            cols.append(X[:, j] ** 4)
    return np.column_stack(cols)


def linear_basis(X: np.ndarray) -> np.ndarray:
    """Intercept-plus-linear basis ``[1, x_1, ..., x_d]`` (used on IHDP)."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def rff_basis(
    X: np.ndarray, n_features: int = 32, bandwidth: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Random Fourier Features approximating an RBF kernel.

    Returns a ``(n, 1 + 2*n_features)`` design matrix with an intercept
    column and sin/cos features at random frequencies -- a
    finite-dimensional non-linear basis that represents smooth
    non-linear functions far better than a low-degree polynomial
    (Rahimi & Recht, 2007).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, n_features)) / bandwidth
    b = rng.uniform(0, 2 * np.pi, size=n_features)
    Z = X @ W + b
    return np.column_stack([np.ones(n), np.cos(Z), np.sin(Z)])
