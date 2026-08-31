"""Errors raised when hazard and population inputs cannot be combined safely."""

from __future__ import annotations


class CrsMismatchError(ValueError):
    """Raised when hazard and population coordinate systems do not match.

    The package requires matching coordinate systems by default. This error
    subclasses ``ValueError``, so existing handlers that catch ``ValueError``
    still work.

    Examples:
        >>> import population_exposure as pe
        >>> issubclass(pe.CrsMismatchError, ValueError)
        True
    """


class PartialCoverageError(ValueError):
    """Raised when a hazard lies outside the population raster's outline.

    The package requires vector features and hazard rasters to sit entirely
    inside the population raster's outer edge. This is about where the grid
    reaches, not about whether it holds values there. This error subclasses
    ``ValueError``, so existing handlers that catch ``ValueError`` still work.

    Examples:
        >>> import population_exposure as pe
        >>> issubclass(pe.PartialCoverageError, ValueError)
        True
    """


class MissingPopulationDataError(ValueError):
    """Raised when the population raster holds no values where a hazard sits.

    A no-data cell records that the source has nothing to say about a place.
    It is not a count of zero people, so the package refuses to report one.
    This error subclasses ``ValueError``, so existing handlers that catch
    ``ValueError`` still work.

    Examples:
        >>> import population_exposure as pe
        >>> issubclass(pe.MissingPopulationDataError, ValueError)
        True
    """
