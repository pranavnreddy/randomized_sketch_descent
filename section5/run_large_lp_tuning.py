"""Large-LP study used by :mod:`section5.run_experiments`. 

The preconditioner is parameterized using problem scales instead of raw
numbers.  If ``s_min`` and ``s_max`` are the extreme singular values of A,
we use

    gamma_x = x_scale / rms(c),
    gamma_x * gamma_lambda = product_scale / (s_min * s_max).

The second relation places the transition in ``I + gamma_x*gamma_lambda
A*A.T`` near the geometric center of the spectrum.  A moderate proxy problem
selects the preconditioner and then ``(sigma, theta)``; only the best settings
are carried to the n=500, m=100 instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

from .experiment_utils import plot_runs, write_rows
from .primal_dual_drs import primal_dual_drs


@dataclass(frozen=True)
class LP:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    optimum: float
    s_min: float
    s_max: float
    c_rms: float


@dataclass(frozen=True)
class Setting:
    label: str
    gamma_x: float
    gamma_lambda: float
    sigma: float
    theta: float


def make_lp(n: int, seed: int) -> LP:
    """Make a bounded standard-form LP with m=round(.2*n)."""
    rng = np.random.default_rng(seed)
    m = round(.2 * n)
    # The last row bounds the nonnegative feasible set.  Scaling it to the
    # same row norm as the random rows avoids inserting an artificial outlier.
    A = np.vstack((rng.standard_normal((m - 1, n)), np.ones(n))) / np.sqrt(n)
    feasible = rng.uniform(.2, 1., n)
    feasible /= feasible.sum()
    b = A @ feasible
    c = rng.standard_normal(n)
    reference = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not reference.success:
        raise RuntimeError(reference.message)
    s = np.linalg.svd(A, compute_uv=False)
    return LP(A, b, c, float(reference.fun), float(s[-1]), float(s[0]),
              float(np.linalg.norm(c) / np.sqrt(n)))


def data_setting(lp: LP, x_scale: float, product_scale: float,
                 sigma: float = .2, theta: float = 1.) -> Setting:
    gamma_x = x_scale / lp.c_rms
    product = product_scale / (lp.s_min * lp.s_max)
    gamma_lambda = product / gamma_x
    return Setting(f"x={x_scale:g}, p={product_scale:g}", gamma_x,
                   gamma_lambda, sigma, theta)


def solve(lp: LP, setting: Setting, max_iterations: int, tolerance: float):
    prox = lambda z, gamma: np.maximum(z - gamma * lp.c, 0.)
    objective = lambda x: float(lp.c @ x)
    return primal_dual_drs(
        prox, lp.A, lp.b, gamma_x=setting.gamma_x,
        gamma_lambda=setting.gamma_lambda, sigma=setting.sigma,
        theta=setting.theta, linear_solver="schur_cg", objective=objective,
        tolerance=tolerance, max_iterations=max_iterations,
    )


def merit(lp: LP, result) -> np.ndarray:
    objective_error = np.abs(result.objective_values - lp.optimum) / max(1., abs(lp.optimum))
    return np.maximum(objective_error + result.feasibility_norms, 1e-16)


def score(lp: LP, result) -> tuple[float, int]:
    # Prefer accuracy first; cumulative inner work breaks practically equal
    # ties.  The log score prevents small numerical noise from dominating.
    return (float(np.log10(merit(lp, result)[-1])),
            int(result.inner_iterations.sum()))


def tune(proxy: LP):
    preconditioners = [
        data_setting(proxy, x_scale, product_scale)
        for x_scale in (.25, 1., 4.) for product_scale in (.1, 1., 10.)
    ]
    pre_runs = [(setting, solve(proxy, setting, 4_000, 3e-7))
                for setting in preconditioners]
    best_m, _ = min(pre_runs, key=lambda item: score(proxy, item[1]))

    parameters = [
        Setting(f"sigma={sigma:g}, theta={theta:g}", best_m.gamma_x,
                best_m.gamma_lambda, sigma, theta)
        for sigma, theta in ((.05, 1.), (.2, 1.), (.45, 1.),
                             (.15, .7), (.15, 1.3), (.14, 1.7))
    ]
    parameter_runs = [(setting, solve(proxy, setting, 4_000, 3e-7))
                      for setting in parameters]
    ranked = sorted(parameter_runs, key=lambda item: score(proxy, item[1]))
    return pre_runs, parameter_runs, [item[0] for item in ranked[:3]]


def transfer(setting: Setting, source: LP, target: LP) -> Setting:
    """Transfer dimensionless scale factors from proxy to target data."""
    x_scale = setting.gamma_x * source.c_rms
    product_scale = (setting.gamma_x * setting.gamma_lambda
                     * source.s_min * source.s_max)
    transferred = data_setting(target, x_scale, product_scale,
                               setting.sigma, setting.theta)
    return Setting(setting.label, transferred.gamma_x,
                   transferred.gamma_lambda, setting.sigma, setting.theta)


def row(problem: str, stage: str, lp: LP, setting: Setting, result) -> dict:
    return dict(
        problem=problem, stage=stage, label=setting.label,
        n=lp.A.shape[1], m=lp.A.shape[0], s_min=lp.s_min, s_max=lp.s_max,
        c_rms=lp.c_rms, gamma_x=setting.gamma_x,
        gamma_lambda=setting.gamma_lambda, sigma=setting.sigma,
        theta=setting.theta, converged=result.converged,
        outer_iterations=result.iterations,
        inner_iterations=int(result.inner_iterations.sum()),
        objective_error=abs(result.objective_values[-1] - lp.optimum),
        feasibility=result.feasibility_norms[-1],
        final_merit=merit(lp, result)[-1],
    )


def run_study() -> None:
    output = Path("figures/section5/lp_tuning")
    output.mkdir(parents=True, exist_ok=True)
    proxy = make_lp(150, 101)
    large = make_lp(500, 202)
    pre_runs, parameter_runs, winners = tune(proxy)

    baseline = Setting("fixed baseline", .5, 1., .2, 1.)
    large_settings = [baseline] + [transfer(s, proxy, large) for s in winners]
    large_runs = [(setting, solve(large, setting, 30_000, 3e-7))
                  for setting in large_settings]

    plot = lambda runs, problem, title, path: plot_runs(
        runs, lambda result: merit(problem, result), title,
        "relative objective error + feasibility", path
    )
    plot(pre_runs, proxy, "LP proxy: data-scaled preconditioner sweep",
         output / "proxy_preconditioners.pdf")
    plot(parameter_runs, proxy, "LP proxy: tolerance and relaxation sweep",
         output / "proxy_sigma_theta.pdf")
    plot(large_runs, large, "Large LP (n=500, m=100): tuned transfer",
         output / "large_lp_convergence.pdf")

    rows = ([row("proxy", "preconditioner", proxy, s, r) for s, r in pre_runs]
            + [row("proxy", "sigma_theta", proxy, s, r) for s, r in parameter_runs]
            + [row("large", "transfer", large, s, r) for s, r in large_runs])
    write_rows(output / "results.csv", rows)

    print(f"proxy: n={proxy.A.shape[1]}, m={proxy.A.shape[0]}, "
          f"singular range=[{proxy.s_min:.3g}, {proxy.s_max:.3g}]")
    print(f"large: n={large.A.shape[1]}, m={large.A.shape[0]}, "
          f"singular range=[{large.s_min:.3g}, {large.s_max:.3g}]")
    for setting, result in large_runs:
        print(f"{setting.label:<24} gx={setting.gamma_x:.3g} "
              f"gl={setting.gamma_lambda:.3g} sigma={setting.sigma:g} "
              f"theta={setting.theta:g} outer={result.iterations:5d} "
              f"inner={result.inner_iterations.sum():7d} "
              f"merit={merit(large, result)[-1]:.2e}")
    print(f"wrote {output}")
