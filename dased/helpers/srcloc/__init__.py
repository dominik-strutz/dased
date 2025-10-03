"""
Source-localization helper utilities for DAS applications.

This subpackage provides forward operators, likelihood helpers and posterior
tools used in Bayesian source localization. Notable exported symbols include
:class:`~dased.helpers.srcloc.ForwardHomogeneous`, :class:`~dased.helpers.srcloc.DataLikelihood`, and
:func:`~dased.helpers.srcloc.calculate_posterior`.

External types commonly referenced are :class:`torch.Tensor`, :class:`xarray.DataArray`,
and :class:`geopandas.GeoDataFrame`.
"""

from .data_likelihood import DataLikelihood
from .forward import ForwardBase, ForwardHomogeneous, ForwardLayeredLookup
from .magnitude_relation import MagnitudeRelation
from .distributions import SurfaceField_Distribution, PolygonUniform, IndependentNormal
from .posterior import calculate_posterior

__all__ = [
    "ForwardBase",
    "ForwardHomogeneous",
    "ForwardLayeredLookup",
    "MagnitudeRelation",
    "DataLikelihood",
    "SurfaceField_Distribution",
    "PolygonUniform",
    "IndependentNormal",
    "calculate_posterior",
]
