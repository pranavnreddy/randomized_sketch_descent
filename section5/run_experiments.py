"""Unified entry point for all Section 5 experiments.

Run ``python -m section5.run_experiments {standard,lp,inequality-qp,all}``.
"""

from __future__ import annotations

import csv
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/randomized-sketch-descent-matplotlib")

import matplotlib
import numpy as np
from scipy.optimize import linprog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .primal_dual_drs import primal_dual_drs, quadratic_prox


@dataclass(frozen=True)
class Problem:
    name: str
    A: np.ndarray
    b: np.ndarray
    prox: object
    objective: object
    optimum: float
    tolerance: float
    max_iterations: int


def make_qp(seed: int = 14) -> Problem:
    rng = np.random.default_rng(seed)
    m, n = 12, 40
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    factor = rng.standard_normal((n, n)) / np.sqrt(n)
    Q = factor.T @ factor + 0.35 * np.eye(n)
    c = rng.standard_normal(n) / np.sqrt(n)
    b = rng.standard_normal(m)
    kkt = np.block([[Q, A.T], [A, np.zeros((m, m))]])
    solution = np.linalg.solve(kkt, np.concatenate((-c, b)))[:n]
    objective = lambda x: float(0.5 * x @ Q @ x + c @ x)
    return Problem("QP", A, b, quadratic_prox(Q, c), objective,
                   objective(solution), 2e-8, 4_000)


def make_lp(seed: int = 27) -> Problem:
    rng = np.random.default_rng(seed)
    m, n = 9, 32
    A = np.vstack((rng.standard_normal((m - 1, n)), np.ones(n)))
    feasible = rng.uniform(0.1, 1.0, n)
    feasible /= feasible.sum()
    b = A @ feasible
    c = rng.standard_normal(n)
    reference = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not reference.success:
        raise RuntimeError(reference.message)
    prox = lambda z, gamma: np.maximum(z - gamma * c, 0.0)
    objective = lambda x: float(c @ x)
    return Problem("LP", A, b, prox, objective, float(reference.fun), 2e-7, 12_000)


def run(problem: Problem, gamma_x: float, gamma_lambda: float, sigma: float,
        theta: float, solver: str):
    return primal_dual_drs(
        problem.prox, problem.A, problem.b, gamma_x=gamma_x,
        gamma_lambda=gamma_lambda, sigma=sigma, theta=theta,
        linear_solver=solver, objective=problem.objective,
        tolerance=problem.tolerance, max_iterations=problem.max_iterations,
    )


def merit(problem: Problem, result) -> np.ndarray:
    scale = max(1.0, abs(problem.optimum))
    objective_error = np.abs(result.objective_values - problem.optimum) / scale
    return np.maximum(objective_error + result.feasibility_norms, 1e-16)


def parameter_sweep(problem: Problem):
    groups = [
        ("M", [
            ("gx=.25, gl=1", .25, 1., .2, 1.),
            ("gx=1, gl=1", 1., 1., .2, 1.),
            ("gx=1, gl=.25", 1., .25, .2, 1.),
        ]),
        ("sigma", [
            ("sigma=.05", .5, 1., .05, 1.),
            ("sigma=.25", .5, 1., .25, 1.),
            ("sigma=.45", .5, 1., .45, 1.),
        ]),
        ("theta", [
            ("theta=.65", .5, 1., .15, .65),
            ("theta=1", .5, 1., .15, 1.),
            ("theta=1.6", .5, 1., .15, 1.6),
        ]),
    ]
    records, plotted = [], []
    for group, choices in groups:
        group_results = []
        for label, gx, gl, sigma, theta in choices:
            result = run(problem, gx, gl, sigma, theta, "schur_cg")
            group_results.append((label, result))
            records.append(record(problem, group, label, gx, gl, sigma, theta,
                                  "schur_cg", result))
        plotted.append((group, group_results))
    return plotted, records


def solver_comparison(problem: Problem):
    results, records = [], []
    for solver in ("schur_cg", "coupled_gmres"):
        result = run(problem, .5, 1., .2, 1., solver)
        results.append((solver, result))
        records.append(record(problem, "solver", solver, .5, 1., .2, 1., solver, result))
    return results, records


def record(problem, group, choice, gx, gl, sigma, theta, solver, result):
    return dict(problem=problem.name, group=group, choice=choice, gamma_x=gx,
                gamma_lambda=gl, sigma=sigma, theta=theta, solver=solver,
                converged=result.converged, outer_iterations=result.iterations,
                inner_iterations=int(result.inner_iterations.sum()),
                final_objective_error=abs(result.objective_values[-1] - problem.optimum),
                final_feasibility=result.feasibility_norms[-1],
                max_relative_error_ratio=float(result.relative_error_ratios.max()))


def plot_parameters(problem: Problem, groups, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, (group, results) in zip(axes, groups):
        for label, result in results:
            axis.semilogy(np.arange(1, result.iterations + 1), merit(problem, result),
                          label=label)
        axis.set_title(group)
        axis.set_xlabel("outer iteration")
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("relative objective error + feasibility")
    fig.suptitle(f"Section 5 {problem.name}: M, inner tolerance, and relaxation")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_solvers(problem: Problem, results, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for label, result in results:
        outer = np.arange(1, result.iterations + 1)
        work = np.cumsum(result.inner_iterations)
        axes[0].semilogy(outer, merit(problem, result), label=label)
        axes[1].semilogy(work, merit(problem, result), label=label)
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("cumulative inner iterations")
    axes[0].set_ylabel("relative objective error + feasibility")
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=8)
    fig.suptitle(f"Section 5 {problem.name}: linear-system methods")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run_standard() -> None:
    output = Path("figures/section5")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for problem in (make_qp(), make_lp()):
        groups, parameter_records = parameter_sweep(problem)
        solvers, solver_records = solver_comparison(problem)
        plot_parameters(problem, groups, output / f"{problem.name.lower()}_parameters.pdf")
        plot_solvers(problem, solvers, output / f"{problem.name.lower()}_solvers.pdf")
        records.extend(parameter_records + solver_records)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote 4 plots and {output / 'results.csv'}")
    for row in records:
        print(f"{row['problem']:>2} {row['group']:<6} {row['choice']:<18} "
              f"outer={row['outer_iterations']:5d} inner={row['inner_iterations']:6d} "
              f"feas={row['final_feasibility']:.2e} objerr={row['final_objective_error']:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study", nargs="?", default="standard",
        choices=("standard", "lp", "inequality-qp", "all"),
    )
    study = parser.parse_args().study
    if study in {"standard", "all"}:
        run_standard()
    if study in {"lp", "all"}:
        from .run_large_lp_tuning import run_study
        run_study()
    if study in {"inequality-qp", "all"}:
        from .run_inequality_qp_tuning import run_study
        run_study()


if __name__ == "__main__":
    main()
