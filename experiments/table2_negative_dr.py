"""Table 2 -- the negative result (Section 4).

The doubly-robust / orthogonal score does not repair the bias in the
few-placebo regime.  We measure the systematic bias of an
inverse-variance-weighted regression of the DR score (the ``alpha=1``
learner) on the linear design, with the propensity

  * estimated         -- observational linear design (propensity must be
    learned and is extreme, ``1 - pi`` small); and
  * known exactly     -- randomised design ``W ~ Bernoulli(p)``,
    ``p = 500 / (500 + N0)``, so ``pi == p`` carries no estimation
    error and the DR remainder ``R(X) == 0`` exactly.

Bias is the RMSE of the seed-averaged prediction against the true
``tau`` on a fixed 3,000-point test set.  Neither row improves on the
X-Learner pseudo-outcome's 0.146 (Table 1): even with the propensity
known, inverse-variance weighting collapses onto the biased
treated-only estimate.

Paper (Table 2):
    DR, estimated propensity : 0.154 / 0.099 / 0.049  (N0 = 30/100/500)
    DR, propensity known     : 0.174 / 0.086 / 0.033

Usage:  python -m experiments.table2_negative_dr [--quick]
"""
from __future__ import annotations

import numpy as np

from gpcate.datasets import (
    linear_design,
    randomised_design,
    stratified_few_placebo,
)
from gpcate.pseudo_outcome import PseudoOutcomeBayesLearner

from ._common import N_FIXED, N_SEEDS_SYNTH, common_args, n_seeds, save_csv

N0_GRID = [30, 100, 500]


def _bias(predict_fn, Xfix, tau_fix, n_seeds_):
    """RMSE of the seed-averaged prediction against tau."""
    preds = [predict_fn(s, Xfix) for s in range(n_seeds_)]
    P = np.array(preds)
    return float(np.sqrt(np.mean((P.mean(0) - tau_fix) ** 2)))


def run(n_seeds_):
    Xfix, _, _, tau_fix = linear_design(N_FIXED, np.random.default_rng(999))
    rows = []
    for n0 in N0_GRID:
        p_known = 500.0 / (500.0 + n0)

        def predict_estimated(s, Xq, n0=n0):
            rng = np.random.default_rng(s)
            X, W, Y = stratified_few_placebo(linear_design, n0, 500, rng)
            m = PseudoOutcomeBayesLearner(alpha=1.0, n_folds=5, random_state=s).fit(
                X, W, Y
            )
            return m.predict(Xq).tau

        def predict_known(s, Xq, n0=n0, p=p_known):
            rng = np.random.default_rng(1000 + s)
            n = int(round((500 + n0) * 1.2))
            X, W, Y, _ = randomised_design(n, p, rng)
            m = PseudoOutcomeBayesLearner(
                alpha=1.0, n_folds=5, known_propensity=p, random_state=s
            ).fit(X, W, Y)
            return m.predict(Xq).tau

        b_est = _bias(predict_estimated, Xfix, tau_fix, n_seeds_)
        b_known = _bias(predict_known, Xfix, tau_fix, n_seeds_)
        rows.append(dict(N0=n0, bias_estimated=b_est, bias_known=b_known))
        print(
            f"  N0={n0:4d}   DR estimated pi: bias={b_est:.3f}   "
            f"DR known pi: bias={b_known:.3f}",
            flush=True,
        )
    return rows


def main():
    args = common_args(__doc__)
    ns = n_seeds(args, N_SEEDS_SYNTH)
    print(f"=== Table 2: DR-score IVW regression bias (linear design, {ns} seeds) ===")
    print("  paper estimated: 0.154/0.099/0.049 | known: 0.174/0.086/0.033")
    rows = run(ns)
    save_csv(rows, "table2_negative_dr.csv")


if __name__ == "__main__":
    main()
