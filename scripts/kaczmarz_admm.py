from __future__ import annotations

import numpy as np

def solve_kaczmarz(A, b, tol=1e-8, max_iter=10000, randomize=False, seed=0):
    rng = np.random.default_rng(seed)
    def calc_err(X):
        return np.max(np.abs(A @ X - b))

    n, m = A.shape[1], A.shape[0]
    X = np.zeros(n)
    
    k = 0
    if randomize:
        p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2
    else:
        p = range(m)
    while(k < max_iter):
        if randomize:
            i = rng.choice(range(m), p=p)
        else:
            i = k % m
        ai = A[i,:]
        Xnew = X + (b[i] - ai @ X) / np.linalg.norm(ai)**2 * ai
        err = np.max(np.abs(A @ Xnew - b))

        if err < tol:
            break
        X = Xnew
        k += 1

        if(k % 100 == 0):
            print(err)

    return X

import numpy as np

def randomized_kaczmarz(A, b, x0=None, tol=1e-8, max_iter=10000, seed=0):
    """
    Solve Ax = b using the Randomized Kaczmarz method.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Coefficient matrix
    b : ndarray, shape (m,)
        Right-hand side vector
    x0 : ndarray, optional
        Initial guess
    max_iters : int
        Maximum number of iterations
    tol : float
        Residual tolerance
    random_state : int or None
        RNG seed

    Returns
    -------
    x : ndarray
        Approximate solution
    history : list
        Residual norm history
    """
    rng = np.random.default_rng(seed)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    m, n = A.shape

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0

    # Row selection probabilities
    row_norms_sq = np.sum(A**2, axis=1)

    if np.any(row_norms_sq == 0):
        raise ValueError("Matrix contains zero rows.")

    probabilities = row_norms_sq / np.sum(row_norms_sq)

    for k in range(max_iter):
        # Randomly choose a row
        i = rng.choice(m, p=probabilities)

        a_i = A[i]

        # Kaczmarz update
        residual = b[i] - np.dot(a_i, x)

        x = x + (residual / row_norms_sq[i]) * a_i

        # Track residual norm occasionally
        if k % n == 0:
            full_residual = np.linalg.norm(A @ x - b)
            if full_residual < tol:
                break
    return x

import numpy as np

def randomized_coordinate_descent(A, b, x0=None, tol=1e-8, max_iter=10000, seed=0):
    """
    Randomized coordinate descent for least squares:

        min_x 0.5 * ||Ax - b||^2

    Parameters
    ----------
    A : ndarray (m, n)
    b : ndarray (m,)
    max_iter : int
    tol : float
    random_state : int or None

    Returns
    -------
    x : ndarray (n,)
    """

    rng = np.random.default_rng(seed)

    m, n = A.shape

    # Accept either a 1-D (m,) or column (m, 1) right-hand side.
    b = np.asarray(b, dtype=float).reshape(-1)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0
    # Residual r = Ax - b
    r = -b.copy()

    # Squared column norms
    col_norm_sq = np.sum(A * A, axis=0)

    # Sampling probabilities
    probs = col_norm_sq / np.sum(col_norm_sq)

    for k in range(max_iter):
        # Randomly choose coordinate
        j = rng.choice(n, p=probs)

        if col_norm_sq[j] == 0:
            continue

        a_j = A[:, j]

        # Coordinate update
        delta = -(a_j @ r) / col_norm_sq[j]

        x[j] += delta

        # Efficient residual update
        r += delta * a_j

        # Convergence test every so often
        if k % n == 0:
            full_residual = np.linalg.norm(A @ x - b)
            if full_residual < tol:
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