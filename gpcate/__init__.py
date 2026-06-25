"""GP-CATE: calibrated CATE inference in the few-placebo regime.

Reference implementation for the paper

    "Calibrated Inference for the Conditional Average Treatment Effect
     in the Few-Placebo Regime via Gaussian Processes" (Eichi Uehara).

The package exposes:

* :class:`~gpcate.gp_cate.GPCATELearner` -- GP-CATE, the proposed
  estimator: a Bayesian T-learner with Gaussian-process arms.
* :class:`~gpcate.pseudo_outcome.PseudoOutcomeBayesLearner` -- the
  X-Learner / doubly-robust pseudo-outcome Bayesian regression used to
  establish the diagnosis (Section 3) and the negative result
  (Section 4).
* :mod:`gpcate.meta_learners` -- S/T/X/DR point-estimate meta-learners.
* :mod:`gpcate.baselines` -- Causal Forest and BART interval baselines.
* :mod:`gpcate.datasets` -- the linear, non-linear, and IHDP designs.
"""
from .basis import linear_basis, polynomial_basis, rff_basis
from .gp_cate import GPArm, GPCATELearner, GPCATEResult
from .pseudo_outcome import PseudoOutcomeBayesLearner, PseudoOutcomeResult

__version__ = "1.0.0"

__all__ = [
    "GPCATELearner",
    "GPCATEResult",
    "GPArm",
    "PseudoOutcomeBayesLearner",
    "PseudoOutcomeResult",
    "polynomial_basis",
    "linear_basis",
    "rff_basis",
]
