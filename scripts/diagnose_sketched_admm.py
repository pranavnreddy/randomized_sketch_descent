"""Why does the sketched ADMM in sketched_hyperplane.ipynb fail?

The sketched y-update projects ``x + u`` onto ``{y in range(S) : A y = b}`` for
a *fresh* random Gaussian sketch ``S`` every iteration.  This script isolates
the consequences with four small experiments:

  A. The reported "feasibility" uses ``A x - b``; ``x`` is never feasible.  The
     y-iterate ``y`` is exactly feasible -- but the algorithm still does not
     converge, so this is not just a plotting bug.
  B. The sketched projection does NOT fix points that are already feasible:
     feeding the true optimum ``x*`` back in returns something far from ``x*``.
     Hence ``x*`` is not a fixed point of the y-update.
  C. Because ``x*`` is not a fixed point, the whole ADMM has no fixed point.
     Starting it exactly at the optimum, it immediately walks away.
  D. The stall is a bias, not slow convergence: sweeping the sketch size from
     m up to n, the final residual only reaches 0 when the sketch is
     full-dimensional (range(S) = R^n), i.e. when there is no sketching at all.
"""

from __future__ import annotations

import numpy as np

from sketched_hyperplane import (
    make_problem,
    project_onto_hyperplane,
    project_onto_sketched_hyperplane,
)


def reference_solution(A, Sigma, b):
    m, n = A.shape
    kkt = np.block([[Sigma, A.T], [A, np.zeros((m, m))]])
    x = np.linalg.solve(kkt, np.concatenate((np.zeros(n), b)))[:n, None]
    return x, float((x.T @ Sigma @ x / 2).item())


def exp_A_feasibility_of_y(A, Sigma, b, max_iter=300, rho=1.0):
    print("=" * 72)
    print("A. feasibility of x vs y inside sketched ADMM")
    print("=" * 72)
    n, m = A.shape[1], A.shape[0]
    rng = np.random.default_rng(0)
    x = np.zeros((n, 1))
    y = np.zeros((n, 1))
    u = np.zeros((n, 1))
    fx = fy = res = 0.0
    for _ in range(max_iter):
        sketch_size = int(rng.integers(m, n))
        x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
        y = project_onto_sketched_hyperplane(A, x + u, b, sketch_size, rng=rng)
        u = x - y + u
        fx = np.linalg.norm(A @ x - b)
        fy = np.linalg.norm(A @ y - b)
        res = np.linalg.norm(x - y)
    print(f"  ||A x - b|| = {fx:.4g}   (what the notebook plots)")
    print(f"  ||A y - b|| = {fy:.4g}   (y is feasible by construction)")
    print(f"  ||x - y||   = {res:.4g}   (ADMM residual -- does NOT vanish)")
    print("  => y is feasible every step, yet x and y never agree:")
    print("     the algorithm itself stalls; the A x term is only a plotting bug.\n")


def exp_B_optimum_not_fixed(A, Sigma, b, x_opt, trials=200):
    print("=" * 72)
    print("B. the sketched projection does not fix the optimum x*")
    print("=" * 72)
    n, m = A.shape[1], A.shape[0]
    print(f"  ||A x* - b||                       = {np.linalg.norm(A @ x_opt - b):.3g}  (x* feasible)")
    # Exact projection of an already-feasible point returns it unchanged.
    p_exact = project_onto_hyperplane(A, x_opt, b)
    print(f"  ||proj_hyperplane(x*) - x*||        = {np.linalg.norm(p_exact - x_opt):.3g}  (exact: x* is fixed)")
    # Sketched projection of the same feasible point: moves it a lot.
    rng = np.random.default_rng(1)
    dists = []
    for _ in range(trials):
        sketch_size = int(rng.integers(m, n))
        p = project_onto_sketched_hyperplane(A, x_opt, b, sketch_size, rng=rng)
        dists.append(np.linalg.norm(p - x_opt))
    dists = np.array(dists)
    print(f"  ||proj_sketched(x*) - x*||  mean    = {dists.mean():.3g}  +/- {dists.std():.2g}")
    print(f"                              min/max = {dists.min():.3g} / {dists.max():.3g}")
    print(f"  relative move: {dists.mean() / np.linalg.norm(x_opt):.1%} of ||x*||")
    print("  => x* lies in {A y = b} but (almost surely) NOT in range(S),")
    print("     so the sketched projection pushes it off the optimum every time.\n")


def exp_C_no_fixed_point(A, Sigma, b, x_opt, max_iter=200, rho=1.0):
    print("=" * 72)
    print("C. ADMM started exactly at the optimum walks away")
    print("=" * 72)
    n, m = A.shape[1], A.shape[0]
    # ADMM fixed point for this scaling: rho*Sigma*x = -u, x = y.
    x = x_opt.copy()
    y = x_opt.copy()
    u = -rho * Sigma @ x_opt
    rng = np.random.default_rng(2)

    # Sanity: the *unsketched* update leaves (x*, y*, u*) put.
    x_un = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
    y_un = project_onto_hyperplane(A, x_un + u, b)
    print(f"  unsketched update from optimum: ||x-x*|| = {np.linalg.norm(x_un - x_opt):.3g} "
          f"(stays -- x* is a fixed point)")

    drift = []
    for _ in range(max_iter):
        sketch_size = int(rng.integers(m, n))
        x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
        y = project_onto_sketched_hyperplane(A, x + u, b, sketch_size, rng=rng)
        u = x - y + u
        drift.append(np.linalg.norm(x - x_opt))
    print("  sketched update from optimum:")
    for k in (1, 2, 5, 20, max_iter):
        print(f"    ||x-x*|| after {k:>4} step(s) = {drift[k - 1]:.3g}")
    print("  => the sketched scheme has no fixed point; it cannot stay at x*.\n")


def exp_D_bias_vs_sketch_size(A, Sigma, b, x_opt, opt_cost, max_iter=400, rho=1.0):
    print("=" * 72)
    print("D. the stall is bias, not slow convergence: sweep the sketch size")
    print("=" * 72)
    n, m = A.shape[1], A.shape[0]
    sizes = [m, 200, 400, 600, 800, 950, n]
    print(f"  {'sketch_size':>12} {'final ||x-y||':>16} {'final suboptimality':>22}")
    for s in sizes:
        rng = np.random.default_rng(3)
        x = np.zeros((n, 1))
        y = np.zeros((n, 1))
        u = np.zeros((n, 1))
        for _ in range(max_iter):
            x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
            y = project_onto_sketched_hyperplane(A, x + u, b, s, rng=rng)
            u = x - y + u
        res = np.linalg.norm(x - y)
        sub = float((x.T @ Sigma @ x / 2).item()) - opt_cost
        tag = "  <- full rank: range(S)=R^n, no sketching" if s == n else ""
        print(f"  {s:>12} {res:>16.4g} {sub:>22.4g}{tag}")
    print("  => the residual shrinks to 0 only when S is square/full-rank.")
    print("     Any genuine sketch (s < n) leaves a non-vanishing bias floor.\n")


def main():
    A, Sigma, b = make_problem()
    x_opt, opt_cost = reference_solution(A, Sigma, b)
    print(f"reference optimum: cost = {opt_cost:.6g}, ||x*|| = {np.linalg.norm(x_opt):.4g}\n")

    exp_A_feasibility_of_y(A, Sigma, b)
    exp_B_optimum_not_fixed(A, Sigma, b, x_opt)
    exp_C_no_fixed_point(A, Sigma, b, x_opt)
    exp_D_bias_vs_sketch_size(A, Sigma, b, x_opt, opt_cost)

    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print(
        """The sketched y-update restricts y to range(S) for a fresh random S each
iteration: y = argmin ||S z - (x+u)|| s.t. A S z = b.

ADMM converges only when the y-update is the proximal operator of a FIXED
function g.  Here:

  * g changes every iteration (a new random subspace range(S_k)), so the
    standard ADMM convergence theorem does not apply; and

  * more fundamentally, the optimal y* = x* is feasible (A x* = b) but almost
    surely x* not in range(S), so x* is NOT a fixed point of the sketched
    projection.  With no common fixed point the iteration cannot converge --
    it only reaches a random "noise floor" (experiments B, C, D).

This is the opposite of sketch-and-project / randomized Kaczmarz, which work
because the solution set {x : A x = b} is invariant under EVERY sketched
step.  The notebook sketches the wrong space: it sketches the primal range
(y = S z) instead of sketching the constraint rows.  A correct sketched
projection must keep every feasible-optimal point fixed -- e.g. sketch the
constraints,  y = (x+u) - (A_S)^T (A_S A_S^T)^-1 (A_S (x+u) - b_S),  which
leaves {A y = b} invariant.

(Secondary, cosmetic bug: sketched_admm logs feasibility as ||A x - b||
instead of ||A y - b||, so the plotted curve is even worse than the real
iterate behaviour.)"""
    )


if __name__ == "__main__":
    main()
