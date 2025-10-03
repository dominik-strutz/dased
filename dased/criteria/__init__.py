"""
Optimization criteria for DAS layout design.

This package exposes classes such as :class:`~dased.criteria.eig.EIGCriterion`
and :class:`~dased.criteria.raysense.RaySensitivity` used by the
optimization routines in :mod:`~dased.optimisation`.
"""

from .eig import EIGCriterion
from .raysense import RaySensitivity, get_gaussian_prior

__all__ = ["EIGCriterion", "RaySensitivity", "get_gaussian_prior"]