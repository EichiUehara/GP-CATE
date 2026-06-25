"""GP-CATE diagnostics (Section 5 / Appendix B).

(A) Posterior contraction.  A wide credible interval is honest only if
    the posterior has actually contracted relative to the prior.  Per
    arm we report the fitted prior function variance ``sigma_w^2`` (the
    ConstantKernel amplitude) and the mean posterior function variance
    ``s_w(x)^2``; their ratio is the contraction factor.  Paper: 0.05-0.07
    on the linear and non-linear designs; 0.55 (N0=30) to 0.16 (N0=100)
    on IHDP.

(B) Arm-variance split.  The fraction of the CATE posterior variance
    ``s_1^2 / (s_0^2 + s_1^2)`` from the (well-determined) treated arm.
    Paper: 0.06-0.25 in the few-placebo regime -- the control term
    dominates, so the treated arm may be fitted with a scalable
    approximation when N1 is large.

(C) Treated-arm subsampling.  Re-running GP-CATE with the treated GP
    fitted on a capped random subset leaves coverage unchanged.  Paper:
    coverage 0.97 on the linear design at N0=30 for caps of 500/200/80.

Usage:  python -m experiments.diagnostics [--quick]
"""
from __future__ import annotations

import numpy as np

from gpcate.datasets import (
    ihdp_few_placebo_split,
    ihdp_files,
    linear_design,
    load_ihdp,
    nonlinear_design,
    stratified_few_placebo,
)
from gpcate.gp_cate import GPCATELearner

from ._common import N_TEST, common_args, coverage, n_seeds, save_csv

DESIGN_FNS = {"linear": linear_design, "nonlinear": nonlinear_design}


def _contraction_row(name, n0, X, W, Y, Xte, seed):
    m = GPCATELearner(random_state=seed).fit(X, W, Y)
    s1 = m.arm1_.posterior(Xte)[1]
    s0 = m.arm0_.posterior(Xte)[1]
    pv0 = m.arm0_.prior_function_variance
    pv1 = m.arm1_.prior_function_variance
    ms0 = float(np.mean(s0 ** 2))
    ms1 = float(np.mean(s1 ** 2))
    return dict(
        design=name, N0=n0,
        prior_var0=pv0, post_var0=ms0, contraction0=ms0 / pv0,
        prior_var1=pv1, post_var1=ms1, contraction1=ms1 / pv1,
        treated_var_frac=ms1 / (ms0 + ms1),
    )


def part_ab():
    print("=== (A) posterior contraction + (B) arm-variance split ===")
    rows = []
    for name in ("linear", "nonlinear"):
        design = DESIGN_FNS[name]
        for n0 in (30, 500):
            rng = np.random.default_rng(0)
            X, W, Y = stratified_few_placebo(design, n0, 500, rng)
            Xte, _, _, _ = design(N_TEST, rng)
            rows.append(_contraction_row(name, n0, X, W, Y, Xte, 0))
    files = ihdp_files()
    for n0 in (30, 100):
        accs = []
        for rep, f in enumerate(files):
            X, W, Y, _ = load_ihdp(f)
            rng = np.random.default_rng(rep)
            Xs, Ws, Ys = ihdp_few_placebo_split(X, W, Y, n0, rng)
            accs.append(_contraction_row("ihdp", n0, Xs, Ws, Ys, X, rep))
        rows.append({k: (np.mean([a[k] for a in accs]) if k not in ("design", "N0")
                         else accs[0][k]) for k in accs[0]})
    for r in rows:
        print(f"  {r['design']:>9s} N0={r['N0']:4d}  control contraction="
              f"{r['contraction0']:.2f}  treated contraction="
              f"{r['contraction1']:.2f}  treated share of CATE var="
              f"{r['treated_var_frac']:.2f}", flush=True)
    return rows


def part_c(n_seeds_):
    print("\n=== (C) treated-arm subsampling: coverage vs cap (linear, N0=30) ===")
    rows = []
    for cap in (500, 200, 80):
        covs = []
        for s in range(n_seeds_):
            rng = np.random.default_rng(s)
            X, W, Y = stratified_few_placebo(linear_design, 30, 500, rng)
            Xte, _, _, tau = linear_design(N_TEST, rng)
            tr = np.where(W == 1)[0]
            if len(tr) > cap:
                drop = rng.choice(tr, size=len(tr) - cap, replace=False)
                keep = np.setdiff1d(np.arange(len(Y)), drop)
                Xs, Ws, Ys = X[keep], W[keep], Y[keep]
            else:
                Xs, Ws, Ys = X, W, Y
            m = GPCATELearner(random_state=s).fit(Xs, Ws, Ys)
            _, lo, hi = m.credible_interval(Xte)
            covs.append(coverage(lo, hi, tau))
        cov = float(np.mean(covs))
        rows.append(dict(treated_cap=cap, coverage=cov))
        print(f"  treated cap={cap:4d}   coverage={cov:.3f}", flush=True)
    return rows


def main():
    args = common_args(__doc__)
    ns = n_seeds(args, 15)
    rows_ab = part_ab()
    rows_c = part_c(ns)
    save_csv(rows_ab, "diagnostics_contraction.csv")
    save_csv(rows_c, "diagnostics_subsampling.csv")


if __name__ == "__main__":
    main()
