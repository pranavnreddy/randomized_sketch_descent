from __future__ import annotations

import numpy as np

def randomized_kaczmarz(A, b, x0=None, tol=1e-8, max_iter=10000, seed=0):
    """Approximately solve ``A x = b`` by randomized Kaczmarz."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    row_norms_sq = np.sum(A**2, axis=1)
    if np.any(row_norms_sq == 0):
        raise ValueError("A contains a zero row")
    probabilities = row_norms_sq / row_norms_sq.sum()

    for k in range(max_iter):
        i = rng.choice(m, p=probabilities)
        row = A[i]
        x += ((b[i] - row @ x) / row_norms_sq[i]) * row
        if k % n == 0:
            if np.linalg.norm(A @ x - b) < tol:
                break
    return x

def randomized_coordinate_descent(A, b, x0=None, tol=1e-8, max_iter=10000, seed=0):
    """Approximately minimize ``||A x - b||^2 / 2`` by coordinate descent."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    residual = A @ x - b
    col_norm_sq = np.sum(A * A, axis=0)
    if np.any(col_norm_sq == 0):
        raise ValueError("A contains a zero column")
    probabilities = col_norm_sq / col_norm_sq.sum()

    for k in range(max_iter):
        j = rng.choice(n, p=probabilities)
        column = A[:, j]
        delta = -(column @ residual) / col_norm_sq[j]
        x[j] += delta
        residual += delta * column
        if k % n == 0 and np.linalg.norm(residual) < tol:
            break
    return x

def kaczmarz_admm(Sigma, A, b, max_iter, rho, eps, seed=0, max_inner_solve_iter=10):
    n, m = A.shape[1], A.shape[0]
    costs = np.zeros((max_iter,))
    feas = np.zeros((max_iter,))
    admm_res = np.zeros((max_iter,))

    x = np.zeros((n,))
    y = np.zeros((n,))
    nu = np.zeros((m,))
    u = np.zeros((n,))
    AAT = A @ A.T

    for i in range(max_iter):
        x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
        nu = randomized_kaczmarz(AAT, b - A @ (x+u), x0=nu, tol=eps, seed=seed, max_iter=max_inner_solve_iter)
        y = x+u + A.T @ nu
        u = x - y + u

        costs[i] = x.T @ Sigma @ x / 2
        feas[i] = np.linalg.norm(A @ y - b)
        admm_res[i] = np.linalg.norm(x - y)
    return costs, feas, admm_res

def rcd_admm(Sigma, A, b, max_iter, rho, eps, seed=0, max_inner_solve_iter=10000):
    n, m = A.shape[1], A.shape[0]
    costs = np.zeros((max_iter,))
    feas = np.zeros((max_iter,))
    admm_res = np.zeros((max_iter,))

    x = np.zeros((n,))
    y = np.zeros((n,))
    u = np.zeros((n,))
    lam = np.zeros((m,))
    AAT = A @ A.T

    for i in range(max_iter):
        x = np.linalg.solve(rho * Sigma + np.eye(n), y - u)
        lam = randomized_coordinate_descent(AAT, b - A @ (x+u), x0=lam, tol=eps, seed=seed, max_iter=max_inner_solve_iter)
        y = x+u + A.T @ lam
        u = x - y + u

        costs[i] = x.T @ Sigma @ x / 2
        feas[i] = np.linalg.norm(A @ y - b)
        admm_res[i] = np.linalg.norm(x - y)
    return costs, feas, admm_res
