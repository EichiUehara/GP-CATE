"""Reproduce every numerical result in the paper, in order.

Runs the five experiment scripts back to back and writes their CSVs to
``experiments/results/``.  Pass ``--quick`` for a fast smoke run (few
seeds; numbers approximate but the pipeline is exercised end to end).

Usage:
    python -m experiments.reproduce_all            # full protocol
    python -m experiments.reproduce_all --quick    # fast smoke run
"""
from __future__ import annotations

import argparse
import time

from . import (
    diagnostics,
    table1_bias_diagnosis,
    table2_negative_dr,
    table3_4_coverage,
    table5_pehe,
)
from ._common import N0_IHDP, N0_SYNTH, N_SEEDS_SYNTH, IHDP_SUBSAMPLES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run with few seeds.")
    args = ap.parse_args()
    quick = args.quick
    ns = 4 if quick else N_SEEDS_SYNTH
    n_sub = 2 if quick else IHDP_SUBSAMPLES

    banner = "#" * 72
    steps = [
        ("Table 1 -- diagnosis (X-Learner pseudo-outcome bias)",
         lambda: table1_bias_diagnosis.run(ns)),
        ("Table 2 -- negative result (DR score, estimated & known pi)",
         lambda: table2_negative_dr.run(ns)),
        ("Table 5 -- point accuracy (root-PEHE)",
         lambda: table5_pehe.run(ns)),
        ("Tables 3 & 4 -- coverage (GP-CATE vs Causal Forest vs BART)",
         lambda: (table3_4_coverage.run_synthetic("linear", N0_SYNTH, ns)
                  + table3_4_coverage.run_synthetic("nonlinear", N0_SYNTH, ns)
                  + table3_4_coverage.run_ihdp(N0_IHDP, n_sub))),
        ("Diagnostics -- contraction, arm split, subsampling",
         lambda: (diagnostics.part_ab(), diagnostics.part_c(ns))),
    ]

    t_all = time.time()
    for title, fn in steps:
        print(f"\n{banner}\n# {title}\n{banner}", flush=True)
        t0 = time.time()
        fn()
        print(f"# done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll experiments finished in {time.time() - t_all:.0f}s.")
    print("Results written to experiments/results/.")


if __name__ == "__main__":
    main()
