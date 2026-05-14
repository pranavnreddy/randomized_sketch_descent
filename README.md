# randomized_sketch_descent

Experiments on randomized sketch descent and sketched ADMM for
equality-constrained quadratic programs and related linear-system solvers
(randomized Kaczmarz, block Kaczmarz).

## Layout

```
notebooks/   Jupyter notebooks (the original experiments)
scripts/     Standalone, reproducible Python scripts
matlab/      MATLAB prototypes (Kaczmarz, sketch descent, Markowitz)
figures/     Generated plots (git-ignored)
```

## Python environment

Dependencies are listed in `pyproject.toml`. Create the environment with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13
uv pip install -r pyproject.toml
```

Run the notebooks with `uv run jupyter lab`, or run the scripts directly:

```bash
.venv/bin/python scripts/sketched_hyperplane.py        # reproduce the notebook
.venv/bin/python scripts/diagnose_sketched_admm.py     # why the sketched variant fails
.venv/bin/python scripts/sketched_hyperplane_fixed.py  # a correctly-sketched solver
```

Plots are written to `figures/` as PDF.

## Scripts

### `sketched_hyperplane.py`

Reproduces `notebooks/sketched_hyperplane.ipynb`: an equality-constrained QP

```
minimize    (1/2) x' Sigma x
subject to  A x = b
```

solved with ADMM, comparing an **unsketched** y-update (exact projection onto
the hyperplane `{A y = b}`) against a **sketched** one (projection onto
`{y in range(S) : A y = b}` for a fresh random Gaussian sketch `S` each
iteration). Writes suboptimality / feasibility / residual plots to `figures/`.

The unsketched variant converges to the CVXPY optimum; the sketched variant
stalls.

### `diagnose_sketched_admm.py`

Pins down **why** the sketched variant fails, with four experiments:

- **A** — the notebook plots feasibility as `||A x - b||`, but `x` is never
  feasible; the y-iterate is feasible by construction. This is only a
  plotting bug — the real issue is that the iteration does not converge.
- **B** — the sketched projection does *not* fix points that are already
  feasible: feeding the true optimum `x*` back in moves it by ~140% of
  `||x*||`, because `x*` almost surely does not lie in `range(S)`.
- **C** — since `x*` is not a fixed point of the y-update, the whole ADMM has
  no fixed point: started exactly at `x*`, it walks away.
- **D** — the stall is a *bias*, not slow convergence: sweeping the sketch
  size from `m` to `n`, the final residual reaches 0 only when `S` is
  full-rank (i.e. when there is no sketching at all).

**Root cause.** ADMM converges only when the y-update is the proximal operator
of a *fixed* function. The notebook redraws the random subspace `range(S_k)`
every iteration, and — more fundamentally — the optimum `x*` is feasible
(`A x* = b`) but does not lie in `range(S)`, so it is not a common fixed point
of the sketched projections. Contrast sketch-and-project / randomized
Kaczmarz, which converge precisely because the solution set is invariant under
*every* sketched step. The notebook sketches the primal range (`y = S z`)
instead of sketching the constraint rows; a correct sketched projection must
leave every feasible point fixed.

### `sketched_hyperplane_fixed.py`

The fix: sketch the **search direction inside the constraint null space**
rather than the variable's range — i.e. randomized sketch descent (RSD), the
method this repo is named after (`matlab/rsd_markowitz.m`). Starting from a
feasible point, each step moves along `d = S P_S t` with
`P_S = I − (A S)⁺(A S)` the projector onto `null(A S)`, so `A d = 0` and every
iterate stays exactly feasible. At the optimum the KKT condition
`Σ x* = Aᵀλ*` makes the sketched reduced gradient `P_Sᵀ Sᵀ Σ x*` vanish for
*every* `S`, so `x*` is a common fixed point — the property the notebook's
version lacked.

The script compares unsketched ADMM, the broken sketched ADMM, and the fixed
sketched descent on the notebook's problem. It plots **normalized
suboptimality** `(f_k − f*)/f*` (which → 0 for both correct methods) and
feasibility to `figures/`.

## MATLAB

The `matlab/` prototypes are standalone. `random_sketch_descent.m` and
`kaczmarz_recovery.m` expect a `compute_huber.m` helper that is not tracked in
this repo (see `.gitignore`).
