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
    """Raised when a hazard geometry lies outside the population raster.

    The package rejects vector features that are not completely covered and
    raster hazards with no spatial overlap. This error subclasses
    ``ValueError``, so existing handlers that catch ``ValueError`` still work.

    Examples:
        >>> import population_exposure as pe
        >>> issubclass(pe.PartialCoverageError, ValueError)
        True
    """
