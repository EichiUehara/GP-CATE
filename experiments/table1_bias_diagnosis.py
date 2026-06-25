"""Table 1 -- the diagnosis (Section 3).

A Bayesian regression on the X-Learner pseudo-outcome under-covers in
the few-placebo regime because the pseudo-outcome's conditional mean is
``tau - b_0``, displaced by the first-order nuisance bias ``b_0``.

We decompose, over seeds, the error of that regression (the
non-orthogonal ``alpha=0`` learner) on the linear design into

  * systematic bias -- RMSE of the seed-averaged prediction against the
    true ``tau`` on a fixed 3,000-point test set;
  * spread          -- sqrt of the mean across-seed prediction variance;

and report the resulting credible-interval coverage on fresh test
points.  The bias is a large fraction of the spread at ``N0 = 30`` (the
interval under-covers), and decays as the placebo arm grows.

Paper (Table 1):
    N0 = 30  : bias 0.146, spread 0.268, coverage 0.88
    N0 = 500 : bias 0.028, spread 0.108, coverage 0.97

Usage:  python -m experiments.table1_bias_diagnosis [--quick]
"""
from __future__ import annotations

import numpy as np

from gpcate.datasets import linear_design, stratified_few_placebo
from gpcate.pseudo_outcome import PseudoOutcomeBayesLearner

from ._common import (
    N_FIXED,
    N_SEEDS_SYNTH,
    N_TEST,
    common_args,
    coverage,
    n_seeds,
    save_csv,
)

N0_GRID = [30, 500]


def run(n_seeds_):
    # Fixed test set for the seed-averaged systematic-bias RMSE.
    Xfix, _, _, tau_fix = linear_design(N_FIXED, np.random.default_rng(999))

    rows = []
    for n0 in N0_GRID:
        preds = []
        covs = []
        for s in range(n_seeds_):
            rng = np.random.default_rng(s)
            X, W, Y = stratified_few_placebo(linear_design, n0, 500, rng)
            Xte, _, _, tau_te = linear_design(N_TEST, rng)

            learner = PseudoOutcomeBayesLearner(alpha=0.0, n_folds=5, random_state=s)
            learner.fit(X, W, Y)
            preds.append(learner.predict(Xfix).tau)
            _, lo, hi = learner.credible_interval(Xte, alpha=0.05)
            covs.append(coverage(lo, hi, tau_te))

        P = np.array(preds)
        bias = float(np.sqrt(np.mean((P.mean(0) - tau_fix) ** 2)))
        spread = float(np.sqrt(np.mean(P.var(0))))
        cov = float(np.mean(covs))
        rows.append(dict(N0=n0, bias=bias, spread=spread, coverage=cov))
        print(
            f"  N0={n0:4d}   bias(RMSE)={bias:.3f}   spread={spread:.3f}   "
            f"coverage={cov:.2f}",
            flush=True,
        )
    return rows


def main():
    args = common_args(__doc__)
    ns = n_seeds(args, N_SEEDS_SYNTH)
    print(f"=== Table 1: X-Learner pseudo-outcome bias diagnosis "
          f"(linear design, {ns} seeds) ===")
    print("  paper: N0=30 bias 0.146 spread 0.268 cov 0.88 | "
          "N0=500 bias 0.028 spread 0.108 cov 0.97")
    rows = run(ns)
    save_csv(rows, "table1_bias_diagnosis.csv")


if __name__ == "__main__":
    main()
