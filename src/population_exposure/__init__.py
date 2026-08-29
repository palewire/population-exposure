"""Assign population values to hazard rows, features, and raster cells."""

from population_exposure.assignment import assign_population
from population_exposure.raster import RasterAssignment

__all__ = ["RasterAssignment", "assign_population"]
