
from __future__ import annotations

import os

import numpy as np
import matplotlib
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketched_hyperplane import (
    make_problem,
    unsketched_admm,
)

from sketched_hyperplane_fixed import (
    reference_solution,
)

from kaczmarz_admm import (
    kaczmarz_admm,
    rcd_admm
)

def main():
    A, Sigma, b = make_problem(n=1000, m=200)

    max_iter = 10000000000
    rho = 1
    eps = 1e-8

    start = time.time()
    kacz_costs, kacz_feas, _ = kaczmarz_admm(Sigma, A, b, max_iter, rho, eps)
    end = time.time()
    kacz_time = end - start

    start = time.time()
    rcd_costs, rcd_feas, _ = rcd_admm(Sigma, A, b, max_iter, rho, eps)
    end = time.time()
    rcd_time = end - start

    start = time.time()
    un_costs, un_feas, _ = unsketched_admm(Sigma, A, b, max_iter, rho)
    end = time.time()
    un_time = end - start

    opt_cost = reference_solution(A, Sigma, b)
    print(f"cvxpy optimal cost = {opt_cost:.8g}\n")

    def report(name, costs, feas, time):
        nsub = (costs[-1] - opt_cost) / opt_cost
        print(f"{name:<26} final cost = {costs[-1]:.8g}  "
              f"norm. subopt = {nsub:.3g}  feas = {feas[-1]:.3g}  time = {time:.3f}")

    report("ADMM w/ direct solve", un_costs, un_feas, un_time)
    # report("sketched ADMM (broken)", br_costs, br_feas)
    report("row sketched indirect solve", kacz_costs, kacz_feas, kacz_time)
    # report("column sketched indirect solve", rcd_costs, rcd_feas, rcd_time)

    # ----------------------------------------------------------------
    # Plots: normalized suboptimality (-> 0) and feasibility.
    # ----------------------------------------------------------------
    plot_label = "admm_with_kaczmarz_comparison"
    os.makedirs("figures", exist_ok=True)
    it = np.arange(max_iter)

    series = [
        (f"admm w/ direct solve ({un_time:.3g}s)", un_costs, un_feas),
        # (f"admm w/ column sketched indirect solve ({rcd_time:.3g}s)", rcd_costs, rcd_feas),
        (f"admm w/ row sketched indirect solve ({kacz_time:.3g}s)", kacz_costs, kacz_feas)
    ]

    fig, ax = plt.subplots()
    for label, costs, _ in series:
        # Normalized suboptimality: (f - f*) / f*  ->  0 as the method converges.
        nsub = np.abs(costs - opt_cost) / opt_cost
        ax.semilogy(it, nsub, label=label)
    ax.set_xlabel("Iteration, $k$")
    ax.set_ylabel(r"Normalized suboptimality $(f_k - f^\star)/f^\star$")
    ax.set_title("Normalized suboptimality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"figures/suboptimality_{plot_label}.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, _, feas in series:
        ax.semilogy(it, np.maximum(feas, 1e-18), label=label)
    ax.set_xlabel("Iteration, $k$")
    ax.set_ylabel(r"Feasibility $\|A x_k - b\|$")
    ax.set_title("Feasibility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"figures/feasibility_{plot_label}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote figures/suboptimality_{plot_label}.pdf, figures/feasibility_{plot_label}.pdf")


if __name__ == "__main__":
    main()
