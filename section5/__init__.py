"""Section 5 preconditioned primal--dual Douglas--Rachford method."""

from .primal_dual_drs import DRSResult, primal_dual_drs, quadratic_prox

__all__ = ["DRSResult", "primal_dual_drs", "quadratic_prox"]
