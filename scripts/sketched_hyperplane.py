"""Reproduce notebooks/sketched_hyperplane.ipynb.

Equality-constrained QP solved with ADMM:

    minimize    (1/2) x' Sigma x
    subject to  A x = b

Two ADMM variants are compared:

* ``unsketched_admm`` -- the y-update is the exact Euclidean projection of
  ``x + u`` onto the hyperplane ``{y : A y = b}``.
* ``sketched_admm``   -- the y-update projects onto ``{y in range(S) : A y = b}``
  for a fresh random Gaussian sketch ``S`` drawn every iteration.

Running this script prints the final iterates and writes the three diagnostic
plots (suboptimality, feasibility, ADMM residual) to ``figures/``.
"""

from __future__ import annotations

import os

import numpy as np
import cvxpy as cvx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Problem data (matches the notebook; seeded here for reproducibility)
# --------------------------------------------------------------------------
SEED = 0
N = 1000
M = 100


def make_problem(seed: int = SEED, n: int = N, m: int = M):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(m, n))
    Sigma = rng.standard_normal(size=(n, n)) / np.sqrt(n)
    Sigma = Sigma.T @ Sigma + np.eye(n)
    b = rng.standard_normal(size=(m,))
    return A, Sigma, b


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------
def project_onto_hyperplane(A, x, b):
    """Euclidean projection of x onto {y : A y = b} via the minimum-norm solution."""
    sol = x + np.linalg.lstsq(A, b - A @ x, rcond=None)[0]
    return sol


def project_onto_sketched_hyperplane(A, x, b, sketch_size, rng=None):
    """Projection of x onto {y in range(S) : A y = b} for a random sketch S.

    Writes y = S z and solves the KKT system of
        minimize (1/2) ||S z - x||^2  subject to  A S z = b.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = np.shape(A)[1]
    m = np.shape(A)[0]
    S = rng.standard_normal(size=(n, sketch_size))
    big_mat = np.block([[S.T @ S, -S.T @ A.T], [A @ S, np.zeros((m, m))]])
    sol = np.linalg.lstsq(big_mat, np.block([[S.T @ x], [b]]), rcond=None)[0]
    return S @ sol[0:sketch_size]


# --------------------------------------------------------------------------
# ADMM variants
# --------------------------------------------------------------------------
def unsketched_admm(Sigma, A, b, max_iter, rho):
    n = np.shape(A)[1]
    costs = np.zeros((max_iter,))
    feas = np.zeros((max_iter,))
    admm_res = np.zeros((max_iter,))

    x = np.zeros((n,))
    y = np.zeros((n,))
    u = np.zeros((n,))

    for i in range(max_iter):
        x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
        y = project_onto_hyperplane(A, x + u, b)
        u = x - y + u

        costs[i] = x.T @ Sigma @ x / 2
        feas[i] = np.linalg.norm(A @ y - b)
        admm_res[i] = np.linalg.norm(x - y)
        if(admm_res[i] < 10e-9):
            break
    return costs, feas, admm_res


def sketched_admm(Sigma, A, b, max_iter, rho, seed=SEED):
    n = np.shape(A)[1]
    m = np.shape(A)[0]
    rng = np.random.default_rng(seed)

    costs = np.zeros((max_iter,))
    feas = np.zeros((max_iter,))
    admm_res = np.zeros((max_iter,))

    x = np.zeros((n,))
    y = np.zeros((n,))
    u = np.zeros((n,))

    for i in range(max_iter):
        sketch_size = rng.integers(m, n)
        x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
        y = project_onto_sketched_hyperplane(A, x + u, b, sketch_size, rng=rng)
        u = x - y + u

        costs[i] = x.T @ Sigma @ x / 2
        # Notebook measures feasibility of x here; we report both x and y.
        feas[i] = np.linalg.norm(A @ x - b)
        admm_res[i] = np.linalg.norm(x - y)
    return costs, feas, admm_res


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    A, Sigma, b = make_problem()
    n = A.shape[1]
    max_iter = 1000
    rho = 1.0

    print(f"cond(Sigma) = {np.linalg.cond(Sigma):.4g}")
    print(f"cond(A)     = {np.linalg.cond(A):.4g}")

    costs, feas, admm_res = unsketched_admm(Sigma, A, b, max_iter, rho)
    print("\nunsketched ADMM (final):")
    print(f"  cost = {costs[-1, 0]:.6g}")
    print(f"  feas = {feas[-1, 0]:.6g}")
    print(f"  res  = {admm_res[-1, 0]:.6g}")

    s_costs, s_feas, s_res = sketched_admm(Sigma, A, b, max_iter, rho)
    print("\nsketched ADMM (final):")
    print(f"  cost = {s_costs[-1, 0]:.6g}")
    print(f"  feas = {s_feas[-1, 0]:.6g}")
    print(f"  res  = {s_res[-1, 0]:.6g}")

    # Reference optimum via CVXPY.
    x = cvx.Variable((n, 1))
    cost = cvx.quad_form(x, Sigma, assume_PSD=True) / 2
    prob = cvx.Problem(cvx.Minimize(cost), [A @ x == b])
    prob.solve()
    opt_cost = float(np.asarray(cost.value).item())
    print(f"\ncvxpy optimal cost = {opt_cost:.6g}")

    # Normalized suboptimality: (f - f*) / f*  ->  0 as the method converges.
    sub = (costs - opt_cost) / opt_cost
    s_sub = (s_costs - opt_cost) / opt_cost

    os.makedirs("figures", exist_ok=True)
    it = np.arange(max_iter)

    for name, series_u, series_s, ylabel in [
        ("suboptimality", sub, s_sub,
         r"Normalized suboptimality $(f_k - f^\star)/f^\star$"),
        ("feasibility", feas, s_feas, r"Feasibility $\|A x_k - b\|$"),
        ("admm_residual", admm_res, s_res, r"ADMM residual $\|x_k - y_k\|$"),
    ]:
        fig, ax = plt.subplots()
        ax.semilogy(it, np.abs(series_u), label="unsketched admm")
        ax.semilogy(it, np.abs(series_s), label="sketched admm")
        ax.set_xlabel("Iteration, $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("$")[0].strip())
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    print("\nwrote figures/{suboptimality,feasibility,admm_residual}.pdf")


if __name__ == "__main__":
    main()
