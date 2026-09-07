"""Warm-started iterative solvers with caller-defined stopping tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]
State = TypeVar("State")
Operator = Callable[[Vector], Vector]
Assess = Callable[[Vector], tuple[bool, State]]


@dataclass(frozen=True)
class InnerSolve(Generic[State]):
    accepted: bool
    iterations: int
    state: State


def schur_cg(
    operator: Operator,
    rhs: Vector,
    initial: Vector,
    assemble: Callable[[Vector], Vector],
    assess: Assess[State],
    max_iterations: int,
) -> InnerSolve[State]:
    """Run conjugate gradients until ``assess`` accepts an assembled state."""
    iterate = initial.copy()
    residual = rhs - operator(iterate)
    direction = residual.copy()
    residual_squared = float(residual @ residual)

    for iteration in range(max_iterations + 1):
        accepted, state = assess(assemble(iterate))
        if accepted or iteration == max_iterations:
            return InnerSolve(accepted, iteration, state)

        image = operator(direction)
        alpha = residual_squared / float(direction @ image)
        iterate += alpha * direction
        next_residual = residual - alpha * image
        next_squared = float(next_residual @ next_residual)
        direction = next_residual + (next_squared / residual_squared) * direction
        residual, residual_squared = next_residual, next_squared

    raise AssertionError("unreachable")


def coupled_gmres(
    operator: Operator,
    rhs: Vector,
    initial: Vector,
    assess: Assess[State],
    max_iterations: int,
) -> InnerSolve[State]:
    """Run unrestarted GMRES until ``assess`` accepts an iterate."""
    initial = initial.copy()
    residual = rhs - operator(initial)
    beta = float(np.linalg.norm(residual))
    accepted, state = assess(initial)
    if accepted or beta <= np.finfo(float).tiny:
        return InnerSolve(accepted, 0, state)

    basis: list[Vector] = [residual / beta]
    hessenberg = np.zeros((max_iterations + 1, max_iterations))
    target = np.zeros(max_iterations + 1)
    target[0] = beta
    breakdown_tolerance = 100 * np.finfo(float).eps

    for column in range(max_iterations):
        image = operator(basis[column])
        for row in range(column + 1):
            hessenberg[row, column] = basis[row] @ image
            image -= hessenberg[row, column] * basis[row]
        hessenberg[column + 1, column] = np.linalg.norm(image)
        if hessenberg[column + 1, column] > breakdown_tolerance:
            basis.append(image / hessenberg[column + 1, column])

        coefficients, *_ = np.linalg.lstsq(
            hessenberg[: column + 2, : column + 1],
            target[: column + 2],
            rcond=None,
        )
        iterate = initial + np.column_stack(basis[: column + 1]) @ coefficients
        accepted, state = assess(iterate)
        iteration = column + 1
        if accepted or hessenberg[iteration, column] <= breakdown_tolerance:
            return InnerSolve(accepted, iteration, state)

    return InnerSolve(accepted, max_iterations, state)
