# Section 5: preconditioned primal--dual DRS

This folder implements equations (58)--(60) and the relaxed relative-error
criterion in Theorem 5 of `Rand_LA_FOM-4`. The metric/preconditioner is

```text
M = diag(I / gamma_x, I / gamma_lambda).
```

`primal_dual_drs.py` provides the algorithm for a generic proximal objective.
It supports the two natural methods for equation (58): GMRES on the coupled
primal--dual system and CG on its positive-definite Schur complement. Both are
warm started and checked after every inner iteration using
`||epsilon||_M <= sigma ||s||_M`, with
`0 <= sigma < (2 - theta) / 2`.

`run_experiments.py` constructs reproducible equality-constrained QP and
standard-form LP instances. It varies the two blocks of `M`, `sigma`, and the
under/over-relaxation parameter `theta`, then compares the two inner solvers by
outer iterations and cumulative inner work.

All experiments use one entry point. From the repository root:

```bash
.venv/bin/python -m section5.run_experiments standard
.venv/bin/python -m section5.run_experiments lp
.venv/bin/python -m section5.run_experiments inequality-qp
```

Plots and the raw summary table are written under `figures/section5/`.

The `lp` study first tunes dimensionless preconditioner scales and
`(sigma, theta)` on
a proxy LP, transfers the best settings using the singular values of the new
constraint matrix and the RMS objective coefficient, and compares them with
the original fixed baseline. Outputs go to `figures/section5/lp_tuning/`.

The `inequality-qp` study solves `min x.T Q x` subject to
`A x + s = b, s >= 0`. It tunes on `n=100, m=20`, transfers parameters to
`n=500, m=100`, and writes convergence plots plus raw results under
`figures/section5/inequality_qp_tuning/`. It also compares fixed parameters
with progress-based adaptation of `sigma` and `theta`. The adaptive policy
uses only step norms, feasibility, recent contraction, and inner iteration
counts; it does not use the reference optimum.
