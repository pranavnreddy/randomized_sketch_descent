"""Run constrained-QP and LP experiments for inexact Douglas--Rachford.

The script prints a compact metrics table and writes convergence plots to
``figures/inexact_dr_qp.pdf`` and ``figures/inexact_dr_lp.pdf``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/randomized-sketch-descent-matplotlib")

import matplotlib
import numpy as np
from scipy.optimize import linprog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inexact_douglas_rachford import inexact_dr, quadratic_prox


@dataclass(frozen=True)
class Metrics:
    problem: str
    converged: bool
    iterations: int
    objective: float
    reference_objective: float
    objective_error: float
    feasibility: float
    solution_error: float
    total_cg_iterations: int
    mean_cg_iterations: float
    max_relative_ratio: float


def _relative_ratio(result) -> np.ndarray:
    denominator = np.maximum(result.step_norms, np.finfo(float).tiny)
    return result.linear_residual_norms / denominator


def run_qp(seed: int = 12):
    rng = np.random.default_rng(seed)
    m, n = 20, 80
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    factor = rng.standard_normal((n, n)) / np.sqrt(n)
    Q = factor.T @ factor + 0.2 * np.eye(n)
    c = rng.standard_normal(n) / np.sqrt(n)
    b = rng.standard_normal(m)
    gamma, sigma = 0.5, 0.7

    result = inexact_dr(
        quadratic_prox(Q, c), A, b, gamma=gamma, sigma=sigma,
        tolerance=1e-8, max_iterations=10_000,
    )
    kkt = np.block([[Q, A.T], [A, np.zeros((m, m))]])
    reference = np.linalg.solve(kkt, np.concatenate((-c, b)))[:n]
    reference_objective = float(0.5 * reference @ Q @ reference + c @ reference)
    objectives = np.einsum("ij,jk,ik->i", result.primal_iterates, Q,
                           result.primal_iterates) / 2 + result.primal_iterates @ c
    objective = float(objectives[-1])
    ratios = _relative_ratio(result)

    metrics = Metrics(
        "quadratic", result.converged, result.iterations, objective,
        reference_objective, abs(objective - reference_objective),
        float(np.linalg.norm(A @ result.primal - b)),
        float(np.linalg.norm(result.primal - reference)),
        int(result.inner_iterations.sum()), float(result.inner_iterations.mean()),
        float(ratios.max()),
    )
    return result, metrics, np.abs(objectives - reference_objective), sigma


def run_lp(seed: int = 21):
    rng = np.random.default_rng(seed)
    m, n = 15, 60
    A = np.vstack((rng.standard_normal((m - 1, n)), np.ones(n)))
    feasible = rng.uniform(0.2, 1.0, n)
    feasible /= feasible.sum()
    b = A @ feasible
    c = rng.standard_normal(n)
    gamma, sigma = 0.5, 0.7

    def prox(z, step):
        return np.maximum(z - step * c, 0.0)

    reference_result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not reference_result.success:
        raise RuntimeError(reference_result.message)
    result = inexact_dr(
        prox, A, b, gamma=gamma, sigma=sigma, tolerance=1e-7,
        max_iterations=20_000,
    )
    objectives = result.primal_iterates @ c
    objective = float(objectives[-1])
    ratios = _relative_ratio(result)

    metrics = Metrics(
        "linear", result.converged, result.iterations, objective,
        float(reference_result.fun), abs(objective - reference_result.fun),
        float(np.linalg.norm(A @ result.primal - b)),
        float(np.linalg.norm(result.primal - reference_result.x)),
        int(result.inner_iterations.sum()), float(result.inner_iterations.mean()),
        float(ratios.max()),
    )
    return result, metrics, np.abs(objectives - reference_result.fun), sigma


def plot_run(result, objective_errors, title: str, path: str) -> None:
    iterations = np.arange(1, result.iterations + 1)
    floor = np.finfo(float).tiny
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].semilogy(iterations, np.maximum(objective_errors, floor))
    axes[0].set_ylabel("absolute objective error")
    axes[1].semilogy(iterations, np.maximum(result.feasibility_norms, floor))
    axes[1].set_ylabel(r"feasibility $\|Ax-b\|$")
    for axis in axes:
        axis.set_xlabel("outer iteration")
        axis.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def print_metrics(metrics: list[Metrics]) -> None:
    header = (
        f"{'problem':<10} {'conv':>5} {'outer':>7} {'objective':>13} "
        f"{'obj error':>11} {'feasibility':>12} {'sol error':>11} "
        f"{'CG total':>9} {'CG mean':>8} {'max e/s':>9}"
    )
    print(header)
    print("-" * len(header))
    for item in metrics:
        print(
            f"{item.problem:<10} {str(item.converged):>5} {item.iterations:7d} "
            f"{item.objective:13.6e} {item.objective_error:11.3e} "
            f"{item.feasibility:12.3e} {item.solution_error:11.3e} "
            f"{item.total_cg_iterations:9d} {item.mean_cg_iterations:8.2f} "
            f"{item.max_relative_ratio:9.3f}"
        )


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    qp_result, qp_metrics, qp_errors, _ = run_qp()
    lp_result, lp_metrics, lp_errors, _ = run_lp()
    plot_run(qp_result, qp_errors, "Inexact DR: constrained quadratic",
             "figures/inexact_dr_qp.pdf")
    plot_run(lp_result, lp_errors, "Inexact DR: linear program",
             "figures/inexact_dr_lp.pdf")
    print_metrics([qp_metrics, lp_metrics])
    print("\nwrote figures/inexact_dr_{qp,lp}.pdf")


if __name__ == "__main__":
    main()
