"""
DASED Package
-------------

This package provides the core functionality for the DASED project, including 
modules for layout management, criteria evaluation, optimization, and helper
utilities for specialized applications.

Core Modules:
    - layout: DAS layout management and visualization
    - criteria: Evaluation criteria for optimization  
    - optimisation: Optimization algorithms and tools
    - helpers: Specialized helper modules for advanced applications

Helper Modules:
    - helpers.srcloc: Source localization tools using Bayesian inference
"""

__version__ = "0.1.0"

from . import criteria
from . import helpers

__all__ = ["criteria", "helpers"]
