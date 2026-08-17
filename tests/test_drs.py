"""Regression tests for both Douglas--Rachford formulations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import LinearConstraint, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inexact_douglas_rachford import (  # noqa: E402
    inexact_dr,
    quadratic_prox as reduced_quadratic_prox,
)
from kaczmarz_admm import randomized_coordinate_descent, randomized_kaczmarz  # noqa: E402
from sketched_hyperplane_fixed import reference_solution  # noqa: E402
from section5 import primal_dual_drs, quadratic_prox  # noqa: E402


def qp_data(seed: int = 4, m: int = 4, n: int = 11):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    factor = rng.standard_normal((n, n))
    Q = factor.T @ factor + np.eye(n)
    c = rng.standard_normal(n)
    b = rng.standard_normal(m)
    kkt = np.block([[Q, A.T], [A, np.zeros((m, m))]])
    solution = np.linalg.solve(kkt, np.r_[-c, b])[:n]
    return A, b, Q, c, solution


def lp_data(seed: int = 7, m: int = 4, n: int = 12):
    rng = np.random.default_rng(seed)
    A = np.vstack((rng.standard_normal((m - 1, n)), np.ones(n)))
    feasible = rng.uniform(.2, 1., n)
    feasible /= feasible.sum()
    b = A @ feasible
    c = rng.standard_normal(n)
    reference = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not reference.success:
        raise RuntimeError(reference.message)
    prox = lambda z, gamma: np.maximum(z - gamma * c, 0.)
    return A, b, c, prox, reference


class ReducedSystemDRTests(unittest.TestCase):
    def test_qp(self):
        A, b, Q, c, reference = qp_data(seed=12, m=5, n=18)
        sigma = .6
        result = inexact_dr(
            reduced_quadratic_prox(Q, c), A, b, gamma=.15, sigma=sigma,
            tolerance=1e-9, max_iterations=5_000,
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.primal, reference, atol=2e-8, rtol=2e-8)
        self.assertLess(np.linalg.norm(A @ result.primal - b), 2e-8)
        self.assertTrue(np.all(
            result.linear_residual_norms**2
            <= sigma * result.step_norms**2 + 1e-24
        ))

    def test_lp(self):
        A, b, c, prox, reference = lp_data(seed=21, m=6, n=20)
        sigma = .5
        result = inexact_dr(
            prox, A, b, gamma=.5, sigma=sigma, tolerance=1e-7,
            max_iterations=20_000,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(c @ result.primal - reference.fun), 2e-6)
        self.assertLess(np.linalg.norm(A @ result.primal - b), 2e-7)
        self.assertGreaterEqual(float(result.primal.min()), -1e-12)
        self.assertTrue(np.all(
            result.linear_residual_norms**2
            <= sigma * result.step_norms**2 + 1e-24
        ))


class Section5DRTests(unittest.TestCase):
    def test_qp_with_each_linear_solver(self):
        A, b, Q, c, reference = qp_data()
        for solver in ("schur_cg", "coupled_gmres"):
            with self.subTest(solver=solver):
                result = primal_dual_drs(
                    quadratic_prox(Q, c), A, b, gamma_x=.15,
                    gamma_lambda=.4, sigma=.2, theta=1.2,
                    linear_solver=solver, tolerance=1e-9,
                    max_iterations=5_000,
                )
                self.assertTrue(result.converged)
                np.testing.assert_allclose(
                    result.primal, reference, atol=3e-8, rtol=3e-8
                )
                self.assertTrue(np.all(result.relative_error_ratios <= .2 + 1e-12))

    def test_lp(self):
        A, b, c, prox, reference = lp_data()
        result = primal_dual_drs(
            prox, A, b, gamma_x=.4, gamma_lambda=.8, sigma=.3,
            theta=.8, tolerance=1e-7, max_iterations=10_000,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(c @ result.primal - reference.fun), 3e-6)
        self.assertLess(np.linalg.norm(A @ result.primal - b), 3e-7)
        self.assertTrue(np.all(result.relative_error_ratios <= .3 + 1e-12))

    def test_invalid_parameters(self):
        identity = np.eye(2)
        zero = np.zeros(2)
        invalid = (("theta", 0.), ("theta", 2.), ("sigma", -.1),
                   ("sigma", .5), ("gamma_x", 0.), ("gamma_lambda", 0.))
        for name, value in invalid:
            kwargs = {name: value}
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                primal_dual_drs(lambda x, _: x, identity, zero, **kwargs)

    def test_adaptive_parameters_remain_admissible(self):
        A, b, Q, c, _ = qp_data()
        result = primal_dual_drs(
            quadratic_prox(Q, c), A, b, gamma_x=.15, gamma_lambda=.4,
            sigma=.2, theta=1., adaptive=True, adaptation_interval=5,
            tolerance=1e-8, max_iterations=1_000,
        )
        self.assertTrue(result.converged)
        self.assertTrue(np.all(result.theta_history > 0))
        self.assertTrue(np.all(result.theta_history < 2))
        self.assertTrue(np.all(result.sigma_history >= 0))
        self.assertTrue(np.all(
            result.sigma_history < (2 - result.theta_history) / 2
        ))

    def test_inequality_qp_slack_formulation(self):
        rng = np.random.default_rng(18)
        m, n = 3, 8
        A = rng.standard_normal((m, n))
        factor = rng.standard_normal((n, n))
        Q = factor.T @ factor + .5 * np.eye(n)
        feasible = rng.standard_normal(n)
        b = A @ feasible + rng.uniform(.1, .3, m)
        B = np.hstack((A, np.eye(m)))
        eigenvalues, eigenvectors = np.linalg.eigh(Q)

        def prox(y, gamma):
            x = eigenvectors @ (
                (eigenvectors.T @ y[:n]) / (1 + 2 * gamma * eigenvalues)
            )
            return np.r_[x, np.maximum(y[n:], 0.)]

        reference = minimize(
            lambda x: x @ Q @ x, np.zeros(n), jac=lambda x: 2 * Q @ x,
            hess=lambda _: 2 * Q,
            constraints=LinearConstraint(A, -np.inf, b), method="trust-constr",
            options={"gtol": 1e-11},
        )
        result = primal_dual_drs(
            prox, B, b, gamma_x=.15, gamma_lambda=.8, sigma=.2,
            theta=1.3, tolerance=1e-9, max_iterations=5_000,
        )
        x, slack = result.primal[:n], result.primal[n:]
        self.assertTrue(result.converged)
        self.assertGreaterEqual(slack.min(), -1e-12)
        self.assertLess(np.linalg.norm(A @ x + slack - b), 2e-8)
        self.assertLess(abs(float(x @ Q @ x) - reference.fun), 5e-6)


class RandomizedLinearSolverTests(unittest.TestCase):
    def test_kaczmarz_and_coordinate_descent(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((8, 8)) + 4 * np.eye(8)
        reference = rng.standard_normal(8)
        b = A @ reference

        kaczmarz = randomized_kaczmarz(A, b, tol=1e-9,
                                       max_iter=100_000, seed=2)
        coordinate = randomized_coordinate_descent(
            A, b, tol=1e-9, max_iter=100_000, seed=2
        )
        np.testing.assert_allclose(kaczmarz, reference, atol=2e-8)
        np.testing.assert_allclose(coordinate, reference, atol=2e-8)

    def test_qp_reference_cost_uses_kkt_system(self):
        A, b, Q, _, _ = qp_data()
        m, n = A.shape
        kkt = np.block([[Q, A.T], [A, np.zeros((m, m))]])
        solution = np.linalg.solve(kkt, np.r_[np.zeros(n), b])[:n]
        expected = float(solution @ Q @ solution / 2)
        self.assertAlmostEqual(reference_solution(A, Q, b), expected, places=11)


if __name__ == "__main__":
    unittest.main()
