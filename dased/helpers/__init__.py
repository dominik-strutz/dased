"""
Helper utilities for the DASED project.

Exposes subpackages such as :mod:`~dased.helpers.srcloc` which contains
source-localization helpers. External types referenced in docstrings use
Sphinx roles (for example :class:`torch.Tensor`, :class:`xarray.DataArray`).
"""

from . import srcloc

__all__ = ["srcloc"]
