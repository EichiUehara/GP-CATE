"""Unit tests for the data-generating processes and IHDP loader."""
from __future__ import annotations

import numpy as np
import pytest

from gpcate.datasets import (
    ihdp_few_placebo_split,
    ihdp_files,
    linear_design,
    load_ihdp,
    nonlinear_design,
    randomised_design,
    stratified_few_placebo,
)


@pytest.mark.parametrize("design", [linear_design, nonlinear_design])
def test_design_shapes_and_support(design):
    X, W, Y, tau = design(400, np.random.default_rng(0))
    assert X.shape == (400, 4)
    assert W.shape == Y.shape == tau.shape == (400,)
    assert set(np.unique(W)).issubset({0, 1})
    assert np.all(np.isfinite(Y)) and np.all(np.isfinite(tau))


def test_linear_tau_formula():
    rng = np.random.default_rng(1)
    X, _, _, tau = linear_design(200, rng)
    assert np.allclose(tau, 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 2])


def test_stratified_few_placebo_arm_sizes():
    rng = np.random.default_rng(2)
    X, W, Y = stratified_few_placebo(linear_design, 30, 500, rng)
    assert (W == 0).sum() == 30
    assert (W == 1).sum() == 500
    assert len(X) == 530


def test_randomised_design_marginal_propensity():
    rng = np.random.default_rng(3)
    _, W, _, _ = randomised_design(20000, 0.2, rng)
    assert abs(W.mean() - 0.2) < 0.02


def test_ihdp_loader_and_split():
    files = ihdp_files()
    assert len(files) >= 1
    X, W, Y, tau = load_ihdp(files[0])
    assert X.shape[1] == 25
    assert len(X) == len(W) == len(Y) == len(tau)
    # covariates standardised
    assert np.allclose(X.mean(0), 0.0, atol=1e-6)
    Xs, Ws, Ys = ihdp_few_placebo_split(X, W, Y, 30, np.random.default_rng(0))
    assert (Ws == 0).sum() == 30
    assert (Ws == 1).sum() == (W == 1).sum()
