"""Tables 3 & 4 -- the main calibration result (Section 5).

GP-CATE is compared against the two most common interval-producing CATE
estimators -- Causal Forest and BART -- across the linear, non-linear,
and IHDP designs in the few-placebo regime.  We report coverage of
nominal-95% intervals (Table 3) and, for IHDP, coverage paired with
mean interval width (Table 4).  No post-hoc calibration is applied.

GP-CATE is the only estimator calibrated throughout (0.94-0.99);
Causal Forest and BART under-cover on the harder designs, most on IHDP.

Paper (Table 3, coverage; columns small/mid/large N0):
    Causal Forest  linear .79/.76/.83  non-lin .59/.61/.63  IHDP .45/.40/.34
    BART           linear .94/.94/.91  non-lin .79/.86/.82  IHDP .59/.74/.77
    GP-CATE        linear .97/.99/.98  non-lin .94/.94/.95  IHDP .96/.96/.95

Usage:  python -m experiments.table3_4_coverage [--quick]
        python -m experiments.table3_4_coverage --only ihdp
"""
from __future__ import annotations

import argparse

import numpy as np

from gpcate.baselines import bart_slearner, causal_forest
from gpcate.datasets import (
    DESIGNS,
    ihdp_few_placebo_split,
    ihdp_files,
    linear_design,
    load_ihdp,
    nonlinear_design,
    stratified_few_placebo,
)
from gpcate.gp_cate import GPCATELearner

from ._common import (
    IHDP_SUBSAMPLES,
    N0_IHDP,
    N0_SYNTH,
    N_SEEDS_SYNTH,
    N_TEST,
    RESULTS_DIR,
    interval_metrics,
    save_csv,
)

DESIGN_FNS = {"linear": linear_design, "nonlinear": nonlinear_design}


def gp_cate(X, W, Y, Xte):
    m = GPCATELearner(random_state=0).fit(X, W, Y)
    return m.credible_interval(Xte, alpha=0.05)


ESTIMATORS = {"gp_cate": gp_cate, "causal_forest": causal_forest, "bart": bart_slearner}


def _eval_estimators(X, W, Y, Xte, tau_te, rec):
    for name, fn in ESTIMATORS.items():
        t, lo, hi = fn(X, W, Y, Xte)
        for k, v in interval_metrics(t, lo, hi, tau_te).items():
            rec[f"{k}_{name}"] = v
    return rec


def run_synthetic(name, n0_grid, n_seeds):
    design = DESIGN_FNS[name]
    rows = []
    for n0 in n0_grid:
        for s in range(n_seeds):
            rng = np.random.default_rng(s)
            X, W, Y = stratified_few_placebo(design, n0, 500, rng)
            Xte, _, _, tau_te = design(N_TEST, rng)
            rows.append(_eval_estimators(X, W, Y, Xte, tau_te,
                                         {"design": name, "N0": n0, "seed": s}))
        _report(name, n0, rows)
    return rows


def run_ihdp(n0_grid, n_sub):
    files = ihdp_files()
    rows = []
    for n0 in n0_grid:
        for rep, f in enumerate(files):
            X, W, Y, tau = load_ihdp(f)
            for s in range(n_sub):
                rng = np.random.default_rng(1000 * rep + s)
                Xs, Ws, Ys = ihdp_few_placebo_split(X, W, Y, n0, rng)
                rows.append(_eval_estimators(Xs, Ws, Ys, X, tau,
                            {"design": "ihdp", "N0": n0, "rep": rep, "sub": s}))
        _report("ihdp", n0, rows)
    return rows


def _report(name, n0, rows):
    import pandas as pd

    d = pd.DataFrame([r for r in rows if r["N0"] == n0 and r["design"] == name])
    msg = f"[{name:9s}] N0={n0:4d}  "
    for est in ESTIMATORS:
        msg += (f"{est}: cov={d[f'cov_{est}'].mean():.2f} "
                f"w={d[f'width_{est}'].mean():.1f}  ")
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["linear", "nonlinear", "ihdp"], default=None)
    args = ap.parse_args()

    ns = 4 if args.quick else N_SEEDS_SYNTH
    n_sub = 2 if args.quick else IHDP_SUBSAMPLES
    designs = [args.only] if args.only else ["linear", "nonlinear", "ihdp"]

    print(f"=== Tables 3 & 4: coverage of nominal-95% intervals "
          f"(GP-CATE vs Causal Forest vs BART) ===")
    rows = []
    for d in designs:
        if d == "ihdp":
            rows += run_ihdp(N0_IHDP, n_sub)
        else:
            rows += run_synthetic(d, N0_SYNTH, ns)
    save_csv(rows, "table3_4_coverage.csv")


if __name__ == "__main__":
    main()
