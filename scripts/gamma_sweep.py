"""Sweep the scalar metric ``gamma_x = gamma_lambda = gamma``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

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
    inequality_qp_optimum,
    nonnegative_linear_prox,
    slack_quadratic_prox,
    standard_form_lp_optimum,
)


GAMMAS = np.logspace(-2, 1, 25)
Condition = Literal["moderately-conditioned", "ill-conditioned"]
CONDITIONS: tuple[Condition, ...] = (
    "moderately-conditioned",
    "ill-conditioned",
)
SINGULAR_LIMITS = {
    "moderately-conditioned": (1.0, 0.1),
    "ill-conditioned": (10.0, 0.1),
}


@dataclass(frozen=True)
class SweepProblem:
    kind: str
    condition: Condition
    dimension: int
    A: np.ndarray
    b: np.ndarray
    prox: Prox
    objective: Callable[[Vector], float]
    optimum: float
    max_iterations: int
    tolerance: float
    matrix_condition: float


def matrix_with_spectrum(
    rng: np.random.Generator,
    m: int,
    n: int,
    condition: Condition,
    rowspace: np.ndarray | None = None,
) -> np.ndarray:
    """Return a full-row-rank matrix with controlled singular values."""
    source = rng.standard_normal((m, n)) if rowspace is None else rowspace
    _, _, vt = np.linalg.svd(source, full_matrices=False)
    high, low = SINGULAR_LIMITS[condition]
    singular_values = np.geomspace(high, low, m)
    return singular_values[:, None] * vt


def make_equality_qp(condition: Condition, seed: int = 41) -> SweepProblem:
    rng = np.random.default_rng(seed)
    m, n = 16, 80
    A = matrix_with_spectrum(rng, m, n, condition)
    factor = rng.standard_normal((n, n)) / np.sqrt(n)
    Q = factor.T @ factor + .3 * np.eye(n)
    c = rng.standard_normal(n) / np.sqrt(n)
    b = A @ rng.standard_normal(n)
    solution = equality_qp_solution(Q, c, A, b)
    objective = lambda x: float(.5 * x @ Q @ x + c @ x)
    return SweepProblem("equality QP", condition, n, A, b,
                        quadratic_prox(Q, c), objective,
                        objective(solution), 8_000, 2e-7, np.linalg.cond(A))


def make_lp(condition: Condition, seed: int = 42) -> SweepProblem:
    rng = np.random.default_rng(seed)
    m, n = 8, 40
    # Preserve a row space containing ones, hence the nonnegative feasible set
    # is bounded, while changing the nonzero singular values of A.
    source = np.vstack((rng.standard_normal((m - 1, n)), np.ones(n)))
    A = matrix_with_spectrum(rng, m, n, condition, source)
    feasible = rng.uniform(.2, 1., n)
    feasible /= feasible.sum()
    b = A @ feasible
    c = rng.standard_normal(n)
    objective = lambda x: float(c @ x)
    return SweepProblem(
        "LP",
        condition,
        n,
        A,
        b,
        nonnegative_linear_prox(c),
        objective,
        standard_form_lp_optimum(c, A, b),
        20_000,
        1e-6,
        np.linalg.cond(A),
    )


def make_inequality_qp(condition: Condition, seed: int = 43) -> SweepProblem:
    rng = np.random.default_rng(seed)
    m, n = 100, 500
    A = matrix_with_spectrum(rng, m, n, condition)
    factor = rng.standard_normal((n, n)) / np.sqrt(n)
    Q = factor.T @ factor + .3 * np.eye(n)
    feasible = rng.standard_normal(n)
    b = A @ feasible + rng.uniform(.05, .2, m)
    B = np.hstack((A, np.eye(m)))
    objective = lambda y: float(y[:n] @ Q @ y[:n])
    return SweepProblem(
        "inequality QP",
        condition,
        n,
        B,
        b,
        slack_quadratic_prox(Q),
        objective,
        inequality_qp_optimum(Q, A, b),
        8_000,
        2e-7,
        np.linalg.cond(A),
    )


def run_problem(
    problem: SweepProblem,
    captured: dict[float, DRSResult] | None = None,
) -> list[dict]:
    rows = []
    for gamma in GAMMAS:
        common = dict(
            problem=problem.kind,
            matrix=problem.condition,
            gamma=float(gamma),
            n=problem.dimension,
            m=problem.A.shape[0],
            matrix_condition=problem.matrix_condition,
        )
        try:
            result = primal_dual_drs(
                problem.prox, problem.A, problem.b,
                gamma_x=float(gamma), gamma_lambda=float(gamma), sigma=.2,
                theta=1., linear_solver="schur_cg", objective=problem.objective,
                tolerance=problem.tolerance,
                max_iterations=problem.max_iterations,
                max_inner_iterations=4 * problem.A.shape[0],
            )
        except RuntimeError as error:
            rows.append(dict(
                **common, converged=False,
                outer_iterations=np.nan, inner_iterations=np.nan,
                mean_inner_iterations=np.nan, objective_error=np.nan,
                feasibility=np.nan, error=str(error),
            ))
            continue
        rows.append(dict(
            **common, converged=result.converged,
            outer_iterations=result.iterations,
            inner_iterations=int(result.inner_iterations.sum()),
            mean_inner_iterations=float(result.inner_iterations.mean()),
            objective_error=abs(result.objective_values[-1] - problem.optimum),
            feasibility=result.feasibility_norms[-1], error="",
        ))
        if captured is not None:
            captured[float(gamma)] = result
    return rows


def plot(rows: list[dict], output: Path) -> None:
    kinds = ("equality QP", "LP", "inequality QP")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for column, kind in enumerate(kinds):
        for condition in CONDITIONS:
            selected = [row for row in rows
                        if row["problem"] == kind and row["matrix"] == condition]
            gamma = np.array([row["gamma"] for row in selected])
            outer = np.array([row["outer_iterations"] for row in selected], dtype=float)
            inner = np.array([row["inner_iterations"] for row in selected], dtype=float)
            axes[0, column].loglog(gamma, outer, "o-", label=condition)
            axes[1, column].loglog(gamma, inner, "o-", label=condition)
            failed = np.array([not row["converged"] for row in selected])
            axes[0, column].scatter(gamma[failed], outer[failed], marker="x",
                                    s=55, color=axes[0, column].lines[-1].get_color())
            axes[1, column].scatter(gamma[failed], inner[failed], marker="x",
                                    s=55, color=axes[1, column].lines[-1].get_color())
        axes[0, column].set_title(kind)
        axes[1, column].set_xlabel(r"shared step size $\gamma$")
        for row in range(2):
            axes[row, column].grid(True, which="both", alpha=.3)
            axes[row, column].legend(fontsize=8)
    axes[0, 0].set_ylabel("outer iterations")
    axes[1, 0].set_ylabel("total inner CG iterations")
    fig.suptitle(r"Section 5 with $\gamma_x=\gamma_\lambda=\gamma$")
    save_figure(fig, output)


def plot_inequality_inner_iterations(rows: list[dict], output: Path,
                                     histories: dict | None = None) -> None:
    """Focused iteration-count and accuracy comparison for the inequality QP."""
    nrows = 2 if histories else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 7 if histories else 4.1))
    axes = np.atleast_2d(axes)
    labels = {
        "moderately-conditioned": r"$\kappa(A)=10$",
        "ill-conditioned": r"$\kappa(A)=100$",
    }
    for condition, label in labels.items():
        selected = [row for row in rows if row["problem"] == "inequality QP"
                    and row["matrix"] == condition]
        gamma = np.array([row["gamma"] for row in selected])
        outer = np.array([row["outer_iterations"] for row in selected], dtype=float)
        inner = np.array([row["inner_iterations"] for row in selected], dtype=float)
        failed = np.array([not row["converged"] for row in selected])
        for axis, values in zip(axes[0], (outer, inner)):
            line, = axis.loglog(gamma, values, "o-", markersize=4,
                                linewidth=1.6, label=label)
            axis.scatter(gamma[failed], values[failed], marker="x", s=55,
                         color=line.get_color(), zorder=3)
    axes[0, 0].set_ylabel("outer iterations")
    axes[0, 1].set_ylabel("total inner CG iterations")
    for axis in axes[0]:
        axis.set_xlabel(r"shared step size $\gamma$")
        axis.grid(True, which="both", alpha=.3)
        axis.legend(title="constraint matrix", fontsize=8)
    if histories:
        for condition, label in labels.items():
            problem, result = histories[condition]
            error = normalized_objective_error(
                result.objective_values, problem.optimum
            )
            accuracy = np.maximum(error + result.feasibility_norms, 1e-16)
            outer = np.arange(1, result.iterations + 1)
            inner = np.cumsum(result.inner_iterations)
            axes[1, 0].semilogy(outer, accuracy, label=label)
            axes[1, 1].semilogy(inner, accuracy, label=label)
        axes[1, 0].set_xlabel("outer iteration")
        axes[1, 1].set_xlabel("cumulative inner CG iterations")
        axes[1, 0].set_ylabel("accuracy")
        axes[1, 1].set_ylabel("accuracy")
        for axis in axes[1]:
            axis.grid(True, which="both", alpha=.3)
            axis.legend(title=r"$\gamma=1$", fontsize=8)
    fig.suptitle(
        r"Inequality QP ($n=500$, $m=100$): iteration counts vs. $\gamma$"
    )
    save_figure(fig, output)


def run_study() -> None:
    output = Path("figures/gamma_sweep")
    makers = (make_equality_qp, make_lp, make_inequality_qp)
    problems = [maker(condition) for maker in makers
                for condition in CONDITIONS]
    rows = [row for problem in problems for row in run_problem(problem)]
    write_rows(output / "results.csv", rows)
    plot(rows, output / "iteration_counts.pdf")
    plot_inequality_inner_iterations(
        rows, output / "inequality_qp_inner_iterations.pdf"
    )
    for kind in ("equality QP", "LP", "inequality QP"):
        for condition in CONDITIONS:
            candidates = [row for row in rows if row["problem"] == kind
                          and row["matrix"] == condition and row["converged"]]
            if not candidates:
                print(f"{kind:<14} {condition:<16} no convergence within cap")
                continue
            best_outer = min(candidates, key=lambda row: row["outer_iterations"])
            best_inner = min(candidates, key=lambda row: row["inner_iterations"])
            print(f"{kind:<14} {condition:<16} "
                  f"outer best gamma={best_outer['gamma']:.3g} "
                  f"({best_outer['outer_iterations']})  "
                  f"inner best gamma={best_inner['gamma']:.3g} "
                  f"({best_inner['inner_iterations']})")
    print(f"wrote {output}")


def run_inequality_study() -> None:
    """Run only the large inequality-QP portion of the gamma sweep."""
    output = Path("figures/gamma_sweep")
    problems = [make_inequality_qp(condition) for condition in CONDITIONS]
    rows = []
    histories = {}
    for problem in problems:
        captured = {}
        rows.extend(run_problem(problem, captured))
        histories[problem.condition] = (problem, captured[1.0])
    write_rows(output / "inequality_qp_results.csv", rows)
    plot_inequality_inner_iterations(
        rows, output / "inequality_qp_inner_iterations.pdf", histories
    )
    for condition in CONDITIONS:
        candidates = [row for row in rows if row["matrix"] == condition
                      and row["converged"]]
        best_outer = min(candidates, key=lambda row: row["outer_iterations"])
        best_inner = min(candidates, key=lambda row: row["inner_iterations"])
        print(f"{condition:<24} outer best gamma={best_outer['gamma']:.3g} "
              f"({best_outer['outer_iterations']})  "
              f"inner best gamma={best_inner['gamma']:.3g} "
              f"({best_inner['inner_iterations']})")
    print(f"wrote focused large-scale results to {output}")
