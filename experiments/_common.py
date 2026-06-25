"""Shared helpers for the reproduction experiments.

Every experiment writes a CSV to ``experiments/results/`` and prints a
table to stdout.  Seed counts and the ``N0`` grid default to the
paper's protocol (Appendix B) and can be reduced with ``--quick`` for a
fast smoke run.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper protocol (Appendix B).
N_SEEDS_SYNTH = 20          # synthetic designs average 20 seeds
N0_SYNTH = [30, 100, 500]   # small / mid / large N0, N1 = 500
N0_IHDP = [30, 60, 100]
IHDP_SUBSAMPLES = 4         # control sub-samples per IHDP replication
N_TEST = 1500               # fresh test points for coverage / PEHE / width
N_FIXED = 3000              # fixed test set for the systematic-bias RMSE


def common_args(description: str) -> argparse.Namespace:
    """Standard CLI: ``--quick`` and ``--seeds`` overrides."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke run: few seeds / reduced grid (numbers approximate).",
    )
    ap.add_argument("--seeds", type=int, default=None, help="override seed count.")
    return ap.parse_args()


def n_seeds(args, full: int) -> int:
    if args.seeds is not None:
        return args.seeds
    return 4 if args.quick else full


def pehe(tau, tau_true) -> float:
    """Root-PEHE: sqrt(mean((tau_hat - tau)^2))."""
    return float(np.sqrt(np.mean((np.asarray(tau) - np.asarray(tau_true)) ** 2)))


def coverage(lo, hi, tau_true) -> float:
    return float(np.mean((lo <= tau_true) & (tau_true <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def interval_metrics(tau, lo, hi, tau_true) -> dict:
    return dict(
        pehe=pehe(tau, tau_true),
        cov=coverage(lo, hi, tau_true),
        width=mean_width(lo, hi),
    )


def save_csv(rows, name: str):
    import pandas as pd

    path = os.path.join(RESULTS_DIR, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n[saved] {path}", flush=True)
    return path
