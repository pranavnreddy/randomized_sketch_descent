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

import numpy as np
import cvxpy as cvx
import math

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


def randomized_sketch_descent_indirect_solve(Sigma, A, b, max_iter, reg, seed=0):
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