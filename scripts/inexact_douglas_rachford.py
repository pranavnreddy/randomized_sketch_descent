"""Inexact Douglas--Rachford splitting for equality-constrained problems.

This module implements the algorithm on pages 4--12 of ``Rand_LA_FOM-2``
for

    minimize f(x)  subject to A x = b.

The KKT inclusion is split into

    F(x, multiplier) = (partial f(x), 0),
    G(x, multiplier) = (A.T @ multiplier, b - A @ x).

The resolvent of ``G`` requires a solve with
``H = I + gamma**2 * A @ A.T``.  Conjugate gradients is stopped as soon as
the linear-system residual satisfies the relative-error condition

    ||e_k||^2 <= sigma * ||s_k||^2,

where ``s_k`` is the proposed Douglas--Rachford update.  Thus the inner
accuracy adapts to the size of the outer step instead of using a fixed
tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]
ProximalOperator = Callable[[Vector, float], ArrayLike]


@dataclass(frozen=True)
class InexactDRResult:
    """Result and convergence diagnostics from :func:`inexact_dr`."""

    primal: Vector
    multiplier: Vector
    fixed_point_primal: Vector
    converged: bool
    iterations: int
    step_norms: Vector
    feasibility_norms: Vector
    linear_residual_norms: Vector
    relative_error_bounds: Vector
    inner_iterations: NDArray[np.int_]
    primal_iterates: NDArray[np.float64]
    update_norms: Vector


def _as_vector(value: ArrayLike, length: int, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def inexact_dr(
    prox_f: ProximalOperator,
    A: ArrayLike,
    b: ArrayLike,
    *,
    gamma: float = 1.0,
    sigma: float = 0.7,
    x0: ArrayLike | None = None,
    multiplier0: ArrayLike | None = None,
    max_iterations: int = 1_000,
    max_inner_iterations: int | None = None,
    tolerance: float = 1e-8,
) -> InexactDRResult:
    """Run relative-error inexact Douglas--Rachford splitting.

    Parameters
    ----------
    prox_f:
        Callable implementing ``prox_{gamma f}(z)`` as ``prox_f(z, gamma)``.
    A, b:
        Equality constraint ``A @ x = b``. ``A`` has shape ``(m, n)``.
    gamma:
        Positive Douglas--Rachford stepsize.
    sigma:
        Relative inner-error parameter.  It must lie in ``[0, 1)``;
        ``sigma=0`` requests the full (up to roundoff) CG solve.
    x0, multiplier0:
        Initial Douglas--Rachford variables.  They default to zero.
    max_iterations:
        Maximum number of outer iterations.
    max_inner_iterations:
        Maximum CG steps in each outer iteration.  Defaults to ``m``.
    tolerance:
        Stop when both the DR step norm and primal feasibility are no larger
        than this value.

    Notes
    -----
    The returned ``primal`` is the proximal point ``v`` from the slides.
    ``fixed_point_primal`` is the first component of the DRS fixed-point
    iterate; these are generally not the same away from convergence.
    """
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"A must be two-dimensional, got ndim={matrix.ndim}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite values")
    m, n = matrix.shape
    rhs_b = _as_vector(b, m, "b")

    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and positive")
    if not 0 <= sigma < 1:
        raise ValueError("sigma must satisfy 0 <= sigma < 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    inner_limit = m if max_inner_iterations is None else max_inner_iterations
    if inner_limit < 1:
        raise ValueError("max_inner_iterations must be positive")

    x = np.zeros(n) if x0 is None else _as_vector(x0, n, "x0")
    multiplier = (
        np.zeros(m)
        if multiplier0 is None
        else _as_vector(multiplier0, m, "multiplier0")
    )

    step_history: list[float] = []
    feasibility_history: list[float] = []
    residual_history: list[float] = []
    bound_history: list[float] = []
    inner_history: list[int] = []
    primal_history: list[Vector] = []
    update_history: list[float] = []
    primal = _as_vector(prox_f(x.copy(), gamma), n, "prox_f result")
    converged = False

    # Applying H without explicitly forming A @ A.T also supports wide A.
    def apply_h(v: Vector) -> Vector:
        return v + gamma**2 * (matrix @ (matrix.T @ v))

    for _ in range(max_iterations):
        # Reduced form of the resolvent equation from page 5:
        # (I + gamma^2 A A.T) nu = lambda - gamma (b - A x).
        reduced_rhs = multiplier - gamma * (rhs_b - matrix @ x)

        # Warm-start at the current outer multiplier.  At every CG iterate we
        # can form the tentative resolvent and test the slide's criterion.
        nu = multiplier.copy()
        residual = reduced_rhs - apply_h(nu)
        direction = residual.copy()
        residual_sq = float(residual @ residual)

        accepted = False
        inner_steps = 0
        for inner_steps in range(inner_limit + 1):
            xi = x - gamma * (matrix.T @ nu)
            # 2*xi - x = xi - gamma*A.T@nu.
            reflected_x = xi - gamma * (matrix.T @ nu)
            v = _as_vector(prox_f(reflected_x, gamma), n,
                           "prox_f result")
            w_minus_nu = -gamma * (rhs_b - matrix @ xi)
            primal_step = v - xi
            # The relative-error scale on slides 9--11 is formed from
            # (v-xi, nu-lambda), not from the eventual inexact dual update.
            step_norm = float(np.hypot(np.linalg.norm(primal_step),
                                       np.linalg.norm(nu - multiplier)))
            update_norm = float(np.hypot(np.linalg.norm(primal_step),
                                         np.linalg.norm(w_minus_nu)))
            residual_norm = float(np.sqrt(max(residual_sq, 0.0)))
            # The slides state ||e_k||^2 <= sigma ||s_k||^2.  Comparing
            # unsquared norms therefore uses sqrt(sigma), not sigma.
            relative_bound = np.sqrt(sigma) * step_norm

            roundoff_floor = 10 * np.finfo(float).eps * max(
                1.0, float(np.linalg.norm(reduced_rhs))
            )
            if residual_norm <= max(relative_bound, roundoff_floor):
                accepted = True
                break
            if inner_steps == inner_limit:
                break

            h_direction = apply_h(direction)
            curvature = float(direction @ h_direction)
            if curvature <= 0:
                raise RuntimeError("CG encountered nonpositive curvature")
            alpha = residual_sq / curvature
            nu += alpha * direction
            next_residual = residual - alpha * h_direction
            next_residual_sq = float(next_residual @ next_residual)
            if residual_sq == 0:
                residual = next_residual
                residual_sq = next_residual_sq
                continue
            direction = next_residual + (next_residual_sq / residual_sq) * direction
            residual = next_residual
            residual_sq = next_residual_sq

        if not accepted:
            raise RuntimeError(
                "inner CG did not meet the relative-error condition within "
                f"{inner_limit} iterations (residual={residual_norm:.3e}, "
                f"bound={relative_bound:.3e})"
            )

        # DRS update in page 8: z_next = z + (v, w) - (xi, nu).
        x += primal_step
        multiplier += w_minus_nu

        feasibility = float(np.linalg.norm(matrix @ v - rhs_b))
        step_history.append(step_norm)
        feasibility_history.append(feasibility)
        residual_history.append(residual_norm)
        bound_history.append(relative_bound)
        inner_history.append(inner_steps)
        primal_history.append(v.copy())
        update_history.append(update_norm)

        if update_norm <= tolerance and feasibility <= tolerance:
            converged = True
            break

    return InexactDRResult(
        primal=v.copy(),
        multiplier=multiplier.copy(),
        fixed_point_primal=x.copy(),
        converged=converged,
        iterations=len(step_history),
        step_norms=np.asarray(step_history),
        feasibility_norms=np.asarray(feasibility_history),
        linear_residual_norms=np.asarray(residual_history),
        relative_error_bounds=np.asarray(bound_history),
        inner_iterations=np.asarray(inner_history, dtype=int),
        primal_iterates=np.asarray(primal_history),
        update_norms=np.asarray(update_history),
    )


def quadratic_prox(hessian: ArrayLike, linear_term: ArrayLike | None = None) -> ProximalOperator:
    """Build the proximal operator of ``0.5*x.T@Q@x + c.T@x``.

    This convenience helper is useful for the quadratic programs elsewhere in
    this repository.  ``Q`` must be symmetric positive semidefinite.
    """
    q = np.asarray(hessian, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("hessian must be a square matrix")
    if not np.allclose(q, q.T):
        raise ValueError("hessian must be symmetric")
    n = q.shape[0]
    c = np.zeros(n) if linear_term is None else _as_vector(linear_term, n, "linear_term")

    def prox(z: Vector, gamma: float) -> Vector:
        return np.linalg.solve(np.eye(n) + gamma * q, z - gamma * c)

    return prox
