# IHDP semi-synthetic benchmark

`ihdp_1.csv` ... `ihdp_10.csv` are ten response-surface replications of
the Infant Health and Development Program (IHDP) benchmark introduced by
Hill (2011) for evaluating CATE estimators. Each file has 747 rows and
the columns

```
W, Y, y_cf, mu0, mu1, X_1, ..., X_25
```

where `W` is the (binary) treatment, `Y` the observed outcome, `y_cf`
the counterfactual outcome, `mu0`/`mu1` the noiseless potential-outcome
surfaces, and `X_1..X_25` the 25 real covariates. The per-unit
treatment effect is known: `tau = mu1 - mu0`, which makes coverage
measurable.

The covariates are real (from the IHDP study); the response surface is
simulated (Hill's "setting B" / NPCI simulator). These ten replications
are the standard public release distributed with, e.g., Johansson et
al.'s CFR/`scikit-uplift`-style benchmarks and Shalit et al. (2017).

In this repository the few-placebo regime is induced by keeping all 139
treated units and randomly sub-sampling the control arm to
`N0 in {30, 60, 100}` (see `gpcate.datasets.ihdp_few_placebo_split`).

## References

- J. L. Hill. *Bayesian Nonparametric Modeling for Causal Inference.*
  Journal of Computational and Graphical Statistics, 20(1):217-240, 2011.
- U. Shalit, F. D. Johansson, D. Sontag. *Estimating individual
  treatment effect: generalization bounds and algorithms.* ICML, 2017.
