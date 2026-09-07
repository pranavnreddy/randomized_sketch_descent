# Randomized sketch descent

Experiments for the preconditioned primal-dual Douglas--Rachford method in
Section 5 of `Rand_LA_FOM-4`, including equality-constrained quadratic
programs, standard-form linear programs, and slack-form inequality QPs.

## Layout

```text
src/       Installable solver package
scripts/   Reproducible experiment entry points and helpers
tests/     Numerical regression tests for the active solver
figures/   Generated plots and CSV results (git-ignored)
Archive/   Superseded outputs retained for research provenance
```

See [`scripts/README.md`](scripts/README.md) for the method summary, study
descriptions, and output locations.

The reusable API is independent of the experiment scripts:

```python
from randomized_sketch_descent import (
    DRSResult,
    primal_dual_drs,
    quadratic_prox,
)
```

## Environment

Dependencies are declared in `pyproject.toml`. Create the environment with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra experiments
```

The base package depends only on NumPy; the `experiments` extra adds SciPy and
Matplotlib.

## Usage

Run an individual study from the repository root:

```bash
.venv/bin/python -m scripts.run_experiments standard
.venv/bin/python -m scripts.run_experiments lp
.venv/bin/python -m scripts.run_experiments inequality-qp
.venv/bin/python -m scripts.run_experiments gamma-sweep
.venv/bin/python -m scripts.run_experiments inequality-gamma-sweep
```

Use `all` to run every study except the focused inequality gamma sweep. Run
the regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
