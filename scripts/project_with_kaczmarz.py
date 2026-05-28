import numpy as np
import time

from sketched_hyperplane import (
    project_onto_hyperplane
)
from kaczmarz_admm import (
    randomized_kaczmarz,
    randomized_coordinate_descent
)

def main():
    n = 1000
    m = 100
    rng = np.random.default_rng(int(time.time() * 10) % 10)
    A = rng.standard_normal(size=(m, n))
    x0 = rng.standard_normal(size=(n, 1))
    b = rng.standard_normal(size=(m, 1))
    eps = 1e-6

    num_trials = 100

    proj_time = proj_kkt_time = rcd_time = kacz_time = rcd_kkt_time = kacz_kkt_time = 0
    rcd_err = kacz_err = rcd_kkt_err = kacz_kkt_err = 0

    for _ in range(num_trials):
        A = rng.standard_normal(size=(m, n))
        x0 = rng.standard_normal(size=(n, 1))
        b = rng.standard_normal(size=(m, 1))
        # Rebuild the KKT matrix for *this* trial's A (it was previously built
        # once outside the loop from a stale A, so the KKT-based solves were
        # comparing against a projection from a different problem).
        big_mat = np.block([[np.eye(n), -A.T], [A, np.zeros((m, m))]])
        start = time.time()
        proj_lam = np.linalg.solve(A @ A.T, b - A @ x0)
        proj_x = x0 + A.T @ proj_lam
        end = time.time()
        proj_time += end - start

        start = time.time()
        proj_y = project_onto_hyperplane(A, x0, b)
        end = time.time()
        proj_kkt_time += end - start

        start = time.time()
        rcd_lam = randomized_coordinate_descent(A @ A.T, b - A @ x0, tol=eps)
        rcd_lam = rcd_lam[:, np.newaxis]
        rcd_y = x0 + A.T @ rcd_lam
        end = time.time()
        rcd_time += end - start
        rcd_err += np.linalg.norm(proj_y - rcd_y) / np.linalg.norm(proj_y)

        start = time.time()
        kacz_lam = randomized_kaczmarz(A @ A.T, b - A @ x0, tol=eps)
        kacz_lam = kacz_lam[:, np.newaxis]
        kacz_y = x0 + A.T @ kacz_lam
        end = time.time()
        kacz_time += end - start
        kacz_err += np.linalg.norm(proj_y - kacz_y) / np.linalg.norm(proj_y)

        start = time.time()
        rcd_kkt_sol = randomized_coordinate_descent(big_mat, np.block([[x0], [b]]), tol=eps)
        rcd_kkt_y = rcd_kkt_sol[0:n]
        end = time.time()
        rcd_kkt_time += end - start
        rcd_kkt_err += np.linalg.norm(np.squeeze(proj_y) - rcd_kkt_y) / np.linalg.norm(proj_y)

        start = time.time()
        kacz_kkt_sol = randomized_kaczmarz(big_mat, np.block([[x0], [b]]), tol=eps)
        kacz_kkt_y = kacz_kkt_sol[0:n]
        end = time.time()
        kacz_kkt_time += end - start
        kacz_kkt_err += np.linalg.norm(np.squeeze(proj_y) - kacz_kkt_y) / np.linalg.norm(proj_y)
    
    proj_time /= num_trials 
    proj_kkt_time /= num_trials 
    rcd_time /= num_trials 
    kacz_time /= num_trials
    rcd_kkt_time /= num_trials
    kacz_kkt_time /= num_trials

    rcd_err /= num_trials
    kacz_err /= num_trials 
    rcd_kkt_err /= num_trials 
    kacz_kkt_err /= num_trials

    print(f"proj time: {proj_time:.4g}s")
    print(f"proj kkt time: {proj_kkt_time:.4g}s")
    print(f"rcd rel err: {rcd_err:.6g} ({rcd_time:.4g}s)")
    print(f"kacz rel err: {kacz_err:.6g} ({kacz_time:.4g}s)")
    print(f"rcd kkt rel err: {rcd_kkt_err:.6g} ({rcd_kkt_time:.4g}s)")
    print(f"kacz kkt rel err: {kacz_kkt_err:.6g} ({kacz_kkt_time:.4g}s)")

if __name__ == "__main__":
    main()
