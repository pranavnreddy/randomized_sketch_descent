"""Inequality-QP study used by :mod:`section5.run_experiments`.

The slack formulation uses the Section 5 variable ``y=(x,s)``, constraint
matrix ``B=[A I]``, and proximal map

    prox_{gamma f}(x,s) = ((I + 2 gamma Q)^-1 x, max(s, 0)).

Dimensionless parameters are tuned on a proxy problem and transferred to a
larger instance using the RMS objective curvature and extreme singular values
of ``B``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .experiment_utils import plot_runs, write_rows
import matplotlib.pyplot as plt

from .primal_dual_drs import DRSResult, primal_dual_drs


@dataclass(frozen=True)
class InequalityQP:
    A: np.ndarray
    b: np.ndarray
    Q: np.ndarray
    B: np.ndarray
    optimum: float
    curvature: float
    s_min: float
    s_max: float


@dataclass(frozen=True)
class Setting:
    label: str
    x_scale: float
    product_scale: float
    sigma: float
    theta: float
    adaptive: bool = False


def make_problem(n: int, seed: int, constraint_ratio: float = .2) -> InequalityQP:
    """Create a feasible dense strongly convex inequality QP."""
    rng = np.random.default_rng(seed)
    m = round(constraint_ratio * n)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    factor = rng.standard_normal((n, n)) / np.sqrt(n)
    Q = factor.T @ factor + .2 * np.eye(n)
    feasible = rng.standard_normal(n)
    slack = rng.uniform(.05, .2, m)
    b = A @ feasible + slack
    B = np.hstack((A, np.eye(m)))

    # scipy's trust-constr reference is independent of the slack DRS method.
    constraint = LinearConstraint(A, -np.inf, b)
    reference = minimize(
        lambda x: float(x @ Q @ x), np.zeros(n),
        jac=lambda x: 2 * Q @ x, hess=lambda _: 2 * Q,
        constraints=constraint, method="trust-constr",
        options={"gtol": 1e-11, "xtol": 1e-12, "maxiter": 2_000},
    )
    if not reference.success:
        raise RuntimeError(reference.message)
    singular_values = np.linalg.svd(B, compute_uv=False)
    return InequalityQP(
        A=A, b=b, Q=Q, B=B, optimum=float(reference.fun),
        curvature=float(np.trace(2 * Q) / n),
        s_min=float(singular_values[-1]), s_max=float(singular_values[0]),
    )


def parameters(problem: InequalityQP, setting: Setting) -> tuple[float, float]:
    gamma_x = setting.x_scale / problem.curvature
    product = setting.product_scale / (problem.s_min * problem.s_max)
    return gamma_x, product / gamma_x


def solve(problem: InequalityQP, setting: Setting, max_iterations: int,
          tolerance: float) -> DRSResult:
    n = problem.Q.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(problem.Q)

    def prox(y: np.ndarray, gamma: float) -> np.ndarray:
        x = eigenvectors @ (
            (eigenvectors.T @ y[:n]) / (1 + 2 * gamma * eigenvalues)
        )
        return np.concatenate((x, np.maximum(y[n:], 0.)))

    gamma_x, gamma_lambda = parameters(problem, setting)
    return primal_dual_drs(
        prox, problem.B, problem.b, gamma_x=gamma_x,
        gamma_lambda=gamma_lambda, sigma=setting.sigma,
        theta=setting.theta, linear_solver="schur_cg",
        adaptive=setting.adaptive,
        objective=lambda y: float(y[:n] @ problem.Q @ y[:n]),
        tolerance=tolerance, max_iterations=max_iterations,
    )


def diagnostics(problem: InequalityQP, result: DRSResult) -> tuple[np.ndarray, np.ndarray]:
    n = problem.Q.shape[0]
    xs = result.primal_iterates[:, :n]
    violations = np.maximum(xs @ problem.A.T - problem.b, 0.)
    inequality = np.linalg.norm(violations, axis=1)
    objective_error = np.abs(result.objective_values - problem.optimum) / max(
        1., abs(problem.optimum)
    )
    merit = np.maximum(objective_error + inequality, 1e-16)
    return merit, inequality


def work_score(problem: InequalityQP, result: DRSResult) -> tuple[float, int]:
    merit, _ = diagnostics(problem, result)
    # Accuracy is compared in half-decade bands; within a band prefer less work.
    return round(float(np.log10(merit[-1])) * 2) / 2, int(result.inner_iterations.sum())


def tune(proxy: InequalityQP):
    preconditioners = [
        Setting(f"x={x:g}, p={p:g}", x, p, .25, 1.)
        for x in (.1, .3, 1.) for p in (.1, 1., 10.)
    ]
    pre_runs = [(s, solve(proxy, s, 3_000, 2e-7)) for s in preconditioners]
    best = min(pre_runs, key=lambda pair: work_score(proxy, pair[1]))[0]
    relaxations = [
        Setting(f"sigma={sigma:g}, theta={theta:g}", best.x_scale,
                best.product_scale, sigma, theta)
        for sigma, theta in ((.05, 1.), (.2, 1.), (.45, 1.),
                             (.1, .7), (.1, 1.3), (.1, 1.7))
    ]
    parameter_runs = [(s, solve(proxy, s, 3_000, 2e-7)) for s in relaxations]
    winners = [s for s, _ in sorted(
        parameter_runs, key=lambda pair: work_score(proxy, pair[1])
    )[:3]]
    return pre_runs, parameter_runs, winners


def result_row(problem_name: str, stage: str, problem: InequalityQP,
               setting: Setting, result: DRSResult) -> dict:
    gamma_x, gamma_lambda = parameters(problem, setting)
    merit, inequality = diagnostics(problem, result)
    return dict(
        problem=problem_name, stage=stage, label=setting.label,
        n=problem.Q.shape[0], m=problem.A.shape[0], gamma_x=gamma_x,
        gamma_lambda=gamma_lambda, x_scale=setting.x_scale,
        product_scale=setting.product_scale, sigma=setting.sigma,
        theta=setting.theta, converged=result.converged,
        adaptive=setting.adaptive,
        outer_iterations=result.iterations,
        inner_iterations=int(result.inner_iterations.sum()),
        objective_error=abs(result.objective_values[-1] - problem.optimum),
        inequality_violation=inequality[-1], final_merit=merit[-1],
    )


def run_study() -> None:
    output = Path("figures/section5/inequality_qp_tuning")
    output.mkdir(parents=True, exist_ok=True)
    proxy = make_problem(100, 311)
    large = make_problem(500, 733)
    pre_runs, parameter_runs, winners = tune(proxy)
    baseline = Setting("fixed baseline", 1., 1., .2, 1.)
    tuned = winners[0]
    adaptive_settings = [
        Setting("adaptive baseline M", 1., 1., .2, 1., True),
        Setting("adaptive tuned M", tuned.x_scale, tuned.product_scale,
                .2, 1., True),
    ]
    large_settings = [baseline, *winners, *adaptive_settings]
    large_runs = [(s, solve(large, s, 12_000, 3e-7)) for s in large_settings]

    plot = lambda runs, problem, title, path: plot_runs(
        runs, lambda result: diagnostics(problem, result)[0], title,
        "relative objective error + inequality violation", path
    )
    plot(pre_runs, proxy, "Inequality QP proxy: preconditioner sweep",
         output / "proxy_preconditioners.pdf")
    plot(parameter_runs, proxy, "Inequality QP proxy: sigma/theta sweep",
         output / "proxy_sigma_theta.pdf")
    plot(large_runs, large, "Large inequality QP (n=500, m=100)",
         output / "large_convergence.pdf")
    plot_adaptation(large_runs, output / "adaptive_parameters.pdf")

    rows = ([result_row("proxy", "preconditioner", proxy, s, r)
             for s, r in pre_runs]
            + [result_row("proxy", "sigma_theta", proxy, s, r)
               for s, r in parameter_runs]
            + [result_row("large", "transfer", large, s, r)
               for s, r in large_runs])
    write_rows(output / "results.csv", rows)

    for setting, result in large_runs:
        gamma_x, gamma_lambda = parameters(large, setting)
        merit, inequality = diagnostics(large, result)
        print(f"{setting.label:<24} gx={gamma_x:.3g} gl={gamma_lambda:.3g} "
              f"outer={result.iterations:5d} inner={result.inner_iterations.sum():7d} "
              f"objerr={abs(result.objective_values[-1]-large.optimum):.2e} "
              f"ineq={inequality[-1]:.2e} merit={merit[-1]:.2e}")
    print(f"wrote {output}")


def plot_adaptation(runs, output: Path) -> None:
    adaptive_runs = [(setting, result) for setting, result in runs
                     if setting.adaptive]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for setting, result in adaptive_runs:
        iterations = np.arange(1, result.iterations + 1)
        axes[0].plot(iterations, result.theta_history, label=setting.label)
        axes[1].plot(iterations, result.sigma_history, label=setting.label)
    axes[0].set_ylabel(r"relaxation $\theta_k$")
    axes[1].set_ylabel(r"inner tolerance $\sigma_k$")
    for axis in axes:
        axis.set_xlabel("outer iteration")
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=8)
    fig.suptitle("Progress-based adaptive parameters")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
