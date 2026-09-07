# Preconditioned primal--dual DRS experiments

The core package implements equations (58)--(60) and the relaxed relative-error
criterion in Theorem 5 of `Rand_LA_FOM-4`. The metric/preconditioner is

```text
M = diag(I / gamma_x, I / gamma_lambda).
```

`randomized_sketch_descent.drs` provides the public algorithm. Its private
`_linear_solvers` module contains GMRES for the coupled primal--dual system and
CG for the positive-definite Schur complement. Both are warm started and
checked after every inner iteration using
`||epsilon||_M <= sigma ||s||_M`, with
`0 <= sigma < (2 - theta) / 2`.

`problems.py` contains shared problem and reference-solution primitives.
`standard.py` constructs equality-constrained QP and standard-form LP studies.
`run_experiments.py` only handles command dispatch.

All experiments use one entry point. From the repository root:

```bash
.venv/bin/python -m scripts.run_experiments standard
.venv/bin/python -m scripts.run_experiments lp
.venv/bin/python -m scripts.run_experiments inequality-qp
.venv/bin/python -m scripts.run_experiments gamma-sweep
.venv/bin/python -m scripts.run_experiments inequality-gamma-sweep
```

The standard study writes plots and its raw summary table under
`figures/standard/`.

The `lp` study tunes dimensionless preconditioner scales and `(sigma, theta)`
on a proxy LP, transfers the best settings using the singular values of the new
constraint matrix and the RMS objective coefficient, and compares them with
the original fixed baseline. Outputs go to `figures/lp_tuning/`.

The `inequality-qp` study solves `min x.T Q x` subject to
`A x + s = b, s >= 0`. It tunes on `n=100, m=20`, transfers parameters to
`n=500, m=100`, and writes convergence plots plus raw results under
`figures/inequality_qp_tuning/`. It also compares fixed parameters
with progress-based adaptation of `sigma` and `theta`. The adaptive policy
uses only step norms, feasibility, recent contraction, and inner iteration
counts; it does not use the reference optimum.

The `gamma-sweep` study imposes `gamma_x = gamma_lambda = gamma` and compares
outer and total inner iterations across equality QPs, LPs, and slack-form
inequality QPs with moderately conditioned (`kappa(A) = 10`) and
ill-conditioned (`kappa(A) = 100`) constraint matrices. Outputs go
to `figures/gamma_sweep/`. The sweep uses 25 logarithmically spaced
values of `gamma`; its focused inequality-QP plot shows both outer iterations
and total inner CG iterations for the two condition numbers. The focused
`inequality-gamma-sweep` runs this comparison at `n=500, m=100` without
rerunning the other problem classes. Its focused figure also shows normalized
objective error plus feasibility versus outer and cumulative inner iterations
for the common `gamma=1` runs.
