
from __future__ import annotations

import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketched_hyperplane import (
    cg_admm_relative_accuracy,
    make_problem,
    unsketched_admm,
)
from sketched_hyperplane_fixed import reference_solution

def main():
    n = 1000
    m = 200
    A, Sigma, b = make_problem(n=n, m=m)

    max_iter = 1000
    rho = 0.01
    start = time.time()
    cg_rel_costs, cg_rel_feas, _ = cg_admm_relative_accuracy(Sigma, A, b, max_iter, rho)
    end = time.time()
    cg_rel_time = end - start

    start = time.time()
    un_costs, un_feas, _ = unsketched_admm(Sigma, A, b, max_iter, rho)
    end = time.time()
    un_time = end - start

    opt_cost = reference_solution(A, Sigma, b)
    print(f"optimal cost = {opt_cost:.8g}\n")

    def report(name, costs, feas, time):
        nsub = (costs[-1] - opt_cost) / opt_cost
        print(f"{name:<26} final cost = {costs[-1]:.8g}  "
              f"norm. subopt = {nsub:.3g}  feas = {feas[-1]:.3g}  time = {time:.3f}")

    report("ADMM w/ direct solve", un_costs, un_feas, un_time)
    report("ADMM w/ conjugate gradient", cg_rel_costs, cg_rel_feas, cg_rel_time)
    plot_label = "admm_with_cg_and_relative_accuracy_comparison"
    os.makedirs("figures", exist_ok=True)
    series = [
        (f"admm w/ direct solve ({un_time:.3g}s)", un_costs, un_feas),
        (f"admm w/ conjugate gradient & relative accuracy ({cg_rel_time:.3g}s)",
         cg_rel_costs, cg_rel_feas),
    ]

    fig, ax = plt.subplots()
    for label, costs, _ in series:
        nsub = np.abs(costs - opt_cost) / opt_cost
        ax.semilogy(np.arange(len(costs)), nsub, label=label)
    ax.set_xlabel("Iteration, $k$")
    ax.set_ylabel(r"Normalized suboptimality $(f_k - f^\star)/f^\star$")
    ax.set_title("Normalized suboptimality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"figures/suboptimality_{plot_label}.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, _, feas in series:
        ax.semilogy(np.arange(len(feas)), np.maximum(feas, 1e-18), label=label)
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
