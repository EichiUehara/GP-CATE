"""Table 5 -- point accuracy (Section 5).

Root-PEHE on the linear design (N1 = 500, N0 swept), confirming that
GP-CATE's calibration does not come at the expense of estimation error:
it is the most accurate estimator at N0 >= 100 and competitive at
N0 = 30.  Compares the S/T/X/DR meta-learners (shared gradient-boosted
nuisances), Causal Forest, BART, and GP-CATE.

Paper (Table 5, linear, N0 = 30/100/500):
    S-Learner   0.36 / 0.24 / 0.18      Causal Forest 0.31 / 0.24 / 0.16
    T-Learner   0.46 / 0.37 / 0.27      BART          0.23 / 0.20 / 0.15
    X-Learner   0.58 / 0.23 / 0.11      GP-CATE       0.29 / 0.13 / 0.07
    DR-Learner  0.63 / 0.49 / 0.18

Usage:  python -m experiments.table5_pehe [--quick]
"""
from __future__ import annotations

import numpy as np

from gpcate.baselines import bart_slearner, causal_forest
from gpcate.datasets import linear_design, stratified_few_placebo
from gpcate.gp_cate import GPCATELearner
from gpcate.meta_learners import META_LEARNERS

from ._common import (
    N0_SYNTH,
    N_SEEDS_SYNTH,
    N_TEST,
    common_args,
    n_seeds,
    pehe,
    save_csv,
)


def _gp_cate_point(X, W, Y, Xte, seed):
    return GPCATELearner(random_state=0).fit(X, W, Y).predict(Xte).tau


def _cf_point(X, W, Y, Xte, seed):
    return causal_forest(X, W, Y, Xte)[0]


def _bart_point(X, W, Y, Xte, seed):
    return bart_slearner(X, W, Y, Xte)[0]


POINT_ESTIMATORS = dict(META_LEARNERS)
POINT_ESTIMATORS.update(
    causal_forest=_cf_point, bart=_bart_point, gp_cate=_gp_cate_point
)


def run(n_seeds_):
    rows = []
    for n0 in N0_SYNTH:
        for s in range(n_seeds_):
            rng = np.random.default_rng(s)
            X, W, Y = stratified_few_placebo(linear_design, n0, 500, rng)
            Xte, _, _, tau_te = linear_design(N_TEST, rng)
            rec = {"N0": n0, "seed": s}
            for name, fn in POINT_ESTIMATORS.items():
                rec[f"pehe_{name}"] = pehe(fn(X, W, Y, Xte, s), tau_te)
            rows.append(rec)
        _report(n0, rows)
    return rows


def _report(n0, rows):
    import pandas as pd

    d = pd.DataFrame([r for r in rows if r["N0"] == n0])
    msg = f"  N0={n0:4d}  " + "  ".join(
        f"{m}={d[f'pehe_{m}'].mean():.2f}" for m in POINT_ESTIMATORS
    )
    print(msg, flush=True)


def main():
    args = common_args(__doc__)
    ns = n_seeds(args, N_SEEDS_SYNTH)
    print(f"=== Table 5: root-PEHE on the linear design ({ns} seeds) ===")
    rows = run(ns)
    save_csv(rows, "table5_pehe.csv")


if __name__ == "__main__":
    main()
