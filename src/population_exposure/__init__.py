"""Assign population values to hazard rows, features, and raster cells."""

from population_exposure import populations
from population_exposure._errors import (
    CrsMismatchError,
    MissingPopulationDataError,
    PartialCoverageError,
)
from population_exposure.assignment import assign_population
from population_exposure.raster import RasterAssignment

__all__ = [
    "CrsMismatchError",
    "MissingPopulationDataError",
    "PartialCoverageError",
    "RasterAssignment",
    "assign_population",
    "populations",
]
