"""A correctly-sketched solver for the equality-constrained QP.

    minimize    (1/2) x' Sigma x
    subject to  A x = b

`diagnose_sketched_admm.py` shows why the notebook's sketched ADMM stalls: its
y-update restricts y to range(S) for a fresh random S each iteration, so the
optimum x* (which is feasible but almost surely not in range(S)) is not a fixed
point of the iteration.

The fix is to sketch the *search direction inside the constraint null space*
instead of sketching the variable's range -- i.e. randomized sketch descent
(RSD), the method this repo is named after (`matlab/rsd_markowitz.m`):

  * start from a feasible point x_0  (A x_0 = b);
  * each step draw S in R^{n x p}, form P_S = I - (A S)^+ (A S), the projector
    onto null(A S);
  * the step direction is  d = S P_S t, and  A d = (A S) P_S t = 0,  so every
    iterate stays exactly feasible -- A x_k = b for all k;
  * t solves the sketched, regularized reduced Newton system
        (P_S' S' Sigma S P_S + reg I) t = P_S' S' (Sigma x).

Why this converges where the notebook's version did not: at the optimum
Sigma x* = A' lambda* (KKT), so  S' Sigma x* = (A S)' lambda*  and
P_S' (A S)' = (A S P_S)' = 0.  Hence the reduced gradient P_S' S' Sigma x*
vanishes for *every* S -- x* is a common fixed point of every sketched step.

This script compares, on the notebook's problem:
  * unsketched ADMM         -- exact baseline (from sketched_hyperplane.py)
  * sketched ADMM (broken)  -- the notebook's variant
  * sketched descent (RSD)  -- the fix
and writes normalized-suboptimality / feasibility plots (PDF) to figures/.
"""

from __future__ import annotations

import os

import numpy as np
import cvxpy as cvx
import matplotlib
import math
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sketched_hyperplane import (
    make_problem,
    unsketched_admm,
    sketched_admm,
)


def reference_solution(A, Sigma, b):
    """Optimal cost via CVXPY."""
    n = A.shape[1]
    x = cvx.Variable((n, 1))
    cost = cvx.quad_form(x, Sigma, assume_PSD=True) / 2
    prob = cvx.Problem(cvx.Minimize(cost), [A @ x == b])
    prob.solve()
    return float(np.asarray(cost.value).item())


def feasible_start(A, b):
    """Minimum-norm point with A x = b."""
    return A.T @ np.linalg.solve(A @ A.T, b)


def randomized_sketch_descent(Sigma, A, b, max_iter, reg, seed=0):
    """Feasibility-preserving sketched solver (the fix).

    Returns per-iteration cost and feasibility ||A x - b|| histories.
    """
    n, m = A.shape[1], A.shape[0]
    rng = np.random.default_rng(seed)

    x = feasible_start(A, b)
    costs = np.zeros((max_iter, 1))
    feas = np.zeros((max_iter, 1))

    for k in range(max_iter):
        # Sketch dimension p must exceed m, else null(A S) is trivial.
        p = int(rng.integers(m + 1, n + 1))
        S = rng.standard_normal((n, p))

        AS = A @ S
        P_S = np.eye(p) - np.linalg.pinv(AS) @ AS  # projector onto null(A S)
        SP = S @ P_S

        grad = Sigma @ x
        M = SP.T @ Sigma @ SP + reg * np.eye(p)
        t = np.linalg.solve(M, SP.T @ grad)
        x = x - SP @ t  # A (SP t) = 0  =>  x stays feasible

        costs[k] = x.T @ Sigma @ x / 2
        feas[k] = np.linalg.norm(A @ x - b)
    return costs, feas


def randomized_sketch_descent_indirect_solve(Sigma, A, b, max_iter, reg, indirect_solve_PS, seed=0):
    """Feasibility-preserving sketched solver (the fix).

    Returns per-iteration cost and feasibility ||A x - b|| histories.
    """
    n, m = A.shape[1], A.shape[0]
    rng = np.random.default_rng(seed)

    x = feasible_start(A, b)
    costs = np.zeros((max_iter, 1))
    feas = np.zeros((max_iter, 1))

    for k in range(max_iter):
        # Sketch dimension p must exceed m, else null(A S) is trivial.
        p = int(rng.integers(m + 1, n + 1))
        S = rng.standard_normal((n, p))

        AS = A @ S

        if(indirect_solve_PS):
            P_S = np.zeros(p)
            for _ in range(math.ceil( 2 * math.log(k+1))):
                ind = int(rng.integers(0, m))
                ASi = AS[ind,:]

                P_S = P_S + (descent_dir[ind] - ASi @ P_S) / np.linalg.norm(ASi)**2 * ASi
            P_S = np.eye(p) - P_S
        else:
            P_S = np.eye(p) - np.linalg.pinv(AS) @ AS  # projector onto null(A S)
        SP = S @ P_S

        grad = Sigma @ x
        M = SP.T @ Sigma @ SP + reg * np.eye(p)

        t = np.zeros(shape=(p,))
        descent_dir = SP.T @ grad
        
        for _ in range(math.ceil( 2 * math.log(k+1))):
            ind = int(rng.integers(0, p-1))
            Mi = M[ind,:]

            t = t + (descent_dir[ind] - Mi @ t) / np.linalg.norm(Mi)**2 * Mi
        # t = np.linalg.solve(M, SP.T @ grad)
        d = SP @ t
        d = np.reshape(d, (n,-1))
        x = x - d  # A (SP t) = 0  =>  x stays feasible

        costs[k] = x.T @ Sigma @ x / 2
        feas[k] = np.linalg.norm(A @ x - b)
    return costs, feas


def main():
    A, Sigma, b = make_problem()
    max_iter = 400
    rho = 1.0
    reg = 1e-2
    indirect_solve_PS = True

    opt_cost = reference_solution(A, Sigma, b)
    print(f"cvxpy optimal cost = {opt_cost:.8g}\n")

    start = time.time()
    un_costs, un_feas, _ = unsketched_admm(Sigma, A, b, max_iter, rho)
    end = time.time()
    un_time = end - start
    # br_costs, br_feas, _ = sketched_admm(Sigma, A, b, max_iter, rho)
    start = time.time()
    rsd_costs, rsd_feas = randomized_sketch_descent(Sigma, A, b, max_iter, reg)
    end = time.time()
    rsd_time = end - start

    start = time.time()
    rsdi_costs, rsdi_feas = randomized_sketch_descent_indirect_solve(Sigma, A, b, max_iter, reg, indirect_solve_PS)
    end = time.time()
    rsdi_time = end - start

    def report(name, costs, feas, time):
        nsub = (costs[-1, 0] - opt_cost) / opt_cost
        print(f"{name:<26} final cost = {costs[-1, 0]:.8g}  "
              f"norm. subopt = {nsub:.3g}  feas = {feas[-1, 0]:.3g}  time = {time:.3f}")

    report("unsketched ADMM", un_costs, un_feas, un_time)
    # report("sketched ADMM (broken)", br_costs, br_feas)
    report("sketched descent (RSD, direct solve)", rsd_costs, rsd_feas, rsd_time)
    report("sketched descent (RSD, indirect solve)", rsdi_costs, rsdi_feas, rsdi_time)

    # ----------------------------------------------------------------
    # Plots: normalized suboptimality (-> 0) and feasibility.
    # ----------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    it = np.arange(max_iter)

    series = [
        (f"unsketched admm  {un_time:.3g}s", un_costs, un_feas),
        (f"sketched descent (RSD, direct solve {rsd_time:.3g}s)", rsd_costs, rsd_feas),
        (f"sketched descent (RSD, indirect solve {rsdi_time:.3g}s)", rsdi_costs, rsdi_feas)
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
    fig.savefig("figures/suboptimality_log_indirect_both.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, _, feas in series:
        ax.semilogy(it, np.maximum(feas, 1e-18), label=label)
    ax.set_xlabel("Iteration, $k$")
    ax.set_ylabel(r"Feasibility $\|A x_k - b\|$")
    ax.set_title("Feasibility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig("figures/feasibility_log_indirect_both.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nwrote figures/suboptimality_log_indirect.pdf, figures/feasibility_log_indirect.pdf")


if __name__ == "__main__":
    main()
