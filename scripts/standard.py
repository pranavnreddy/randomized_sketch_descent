"""Standard parameter and inner-solver comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from randomized_sketch_descent import DRSResult, primal_dual_drs, quadratic_prox

from .experiment_utils import (
    normalized_objective_error,
    plt,
    save_figure,
    write_rows,
)
from .problems import (
    Prox,
    Vector,
    equality_qp_solution,
    nonnegative_linear_prox,
    standard_form_lp_optimum,
)


@dataclass(frozen=True)
class Problem:
    name: str
    matrix: Vector
    rhs: Vector
    prox: Prox
    objective: Callable[[Vector], float]
    optimum: float
    tolerance: float
    max_iterations: int


def make_qp(seed: int = 14) -> Problem:
    rng = np.random.default_rng(seed)
    constraints, variables = 12, 40
    matrix = rng.standard_normal((constraints, variables)) / np.sqrt(variables)
    factor = rng.standard_normal((variables, variables)) / np.sqrt(variables)
    hessian = factor.T @ factor + 0.35 * np.eye(variables)
    linear_term = rng.standard_normal(variables) / np.sqrt(variables)
    rhs = rng.standard_normal(constraints)
    solution = equality_qp_solution(hessian, linear_term, matrix, rhs)

    def objective(vector: Vector) -> float:
        return float(0.5 * vector @ hessian @ vector + linear_term @ vector)

    return Problem(
        "QP",
        matrix,
        rhs,
        quadratic_prox(hessian, linear_term),
        objective,
        objective(solution),
        2e-8,
        4_000,
    )


def make_lp(seed: int = 27) -> Problem:
    rng = np.random.default_rng(seed)
    constraints, variables = 9, 32
    matrix = np.vstack(
        (rng.standard_normal((constraints - 1, variables)), np.ones(variables))
    )
    feasible = rng.uniform(0.1, 1.0, variables)
    feasible /= feasible.sum()
    rhs = matrix @ feasible
    cost = rng.standard_normal(variables)

    def objective(vector: Vector) -> float:
        return float(cost @ vector)

    return Problem(
        "LP",
        matrix,
        rhs,
        nonnegative_linear_prox(cost),
        objective,
        standard_form_lp_optimum(cost, matrix, rhs),
        2e-7,
        12_000,
    )


def solve(
    problem: Problem,
    gamma_x: float,
    gamma_lambda: float,
    sigma: float,
    theta: float,
    solver: str,
) -> DRSResult:
    return primal_dual_drs(
        problem.prox,
        problem.matrix,
        problem.rhs,
        gamma_x=gamma_x,
        gamma_lambda=gamma_lambda,
        sigma=sigma,
        theta=theta,
        linear_solver=solver,
        objective=problem.objective,
        tolerance=problem.tolerance,
        max_iterations=problem.max_iterations,
    )


def merit(problem: Problem, result: DRSResult) -> np.ndarray:
    error = normalized_objective_error(result.objective_values, problem.optimum)
    return np.maximum(error + result.feasibility_norms, 1e-16)


def result_row(problem, group, choice, gx, gl, sigma, theta, solver, result):
    return dict(
        problem=problem.name,
        group=group,
        choice=choice,
        gamma_x=gx,
        gamma_lambda=gl,
        sigma=sigma,
        theta=theta,
        solver=solver,
        converged=result.converged,
        outer_iterations=result.iterations,
        inner_iterations=int(result.inner_iterations.sum()),
        final_objective_error=abs(result.objective_values[-1] - problem.optimum),
        final_feasibility=result.feasibility_norms[-1],
        max_relative_error_ratio=float(result.relative_error_ratios.max()),
    )


def parameter_sweep(problem: Problem):
    groups = [
        (
            "M",
            [
                ("gx=.25, gl=1", 0.25, 1.0, 0.2, 1.0),
                ("gx=1, gl=1", 1.0, 1.0, 0.2, 1.0),
                ("gx=1, gl=.25", 1.0, 0.25, 0.2, 1.0),
            ],
        ),
        (
            "sigma",
            [
                ("sigma=.05", 0.5, 1.0, 0.05, 1.0),
                ("sigma=.25", 0.5, 1.0, 0.25, 1.0),
                ("sigma=.45", 0.5, 1.0, 0.45, 1.0),
            ],
        ),
        (
            "theta",
            [
                ("theta=.65", 0.5, 1.0, 0.15, 0.65),
                ("theta=1", 0.5, 1.0, 0.15, 1.0),
                ("theta=1.6", 0.5, 1.0, 0.15, 1.6),
            ],
        ),
    ]
    rows, plotted = [], []
    for group, choices in groups:
        group_results = []
        for label, gx, gl, sigma, theta in choices:
            result = solve(problem, gx, gl, sigma, theta, "schur_cg")
            group_results.append((label, result))
            rows.append(
                result_row(
                    problem,
                    group,
                    label,
                    gx,
                    gl,
                    sigma,
                    theta,
                    "schur_cg",
                    result,
                )
            )
        plotted.append((group, group_results))
    return plotted, rows


def solver_comparison(problem: Problem):
    results, rows = [], []
    for solver in ("schur_cg", "coupled_gmres"):
        result = solve(problem, 0.5, 1.0, 0.2, 1.0, solver)
        results.append((solver, result))
        rows.append(
            result_row(
                problem, "solver", solver, 0.5, 1.0, 0.2, 1.0, solver, result
            )
        )
    return results, rows


def plot_parameters(problem: Problem, groups, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, (group, results) in zip(axes, groups):
        for label, result in results:
            iterations = np.arange(1, result.iterations + 1)
            axis.semilogy(iterations, merit(problem, result), label=label)
        axis.set_title(group)
        axis.set_xlabel("outer iteration")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("relative objective error + feasibility")
    fig.suptitle(f"Section 5 {problem.name}: M, tolerance, and relaxation")
    save_figure(fig, output)


def plot_solvers(problem: Problem, results, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for label, result in results:
        axes[0].semilogy(
            np.arange(1, result.iterations + 1), merit(problem, result), label=label
        )
        axes[1].semilogy(
            np.cumsum(result.inner_iterations), merit(problem, result), label=label
        )
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("cumulative inner iterations")
    axes[0].set_ylabel("relative objective error + feasibility")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(f"Section 5 {problem.name}: linear-system methods")
    save_figure(fig, output)


def run_study() -> None:
    output = Path("figures/standard")
    rows = []
    for problem in (make_qp(), make_lp()):
        groups, parameter_rows = parameter_sweep(problem)
        solvers, solver_rows = solver_comparison(problem)
        stem = problem.name.lower()
        plot_parameters(problem, groups, output / f"{stem}_parameters.pdf")
        plot_solvers(problem, solvers, output / f"{stem}_solvers.pdf")
        rows.extend(parameter_rows + solver_rows)

    write_rows(output / "results.csv", rows)
    print(f"wrote 4 plots and {output / 'results.csv'}")
    for row in rows:
        print(
            f"{row['problem']:>2} {row['group']:<6} {row['choice']:<18} "
            f"outer={row['outer_iterations']:5d} "
            f"inner={row['inner_iterations']:6d} "
            f"feas={row['final_feasibility']:.2e} "
            f"objerr={row['final_objective_error']:.2e}"
        )
