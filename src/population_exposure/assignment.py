"""Population assignment dispatch for tabular and spatial hazard data."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.io import DatasetReader

from population_exposure._validation import (
    normalize_columns,
    numeric_values,
    require_columns,
    require_complete_keys,
    require_unique_keys,
)
from population_exposure.raster import (
    RasterAssignment,
    RasterSource,
    assign_raster_population,
)
from population_exposure.vector import assign_vector_population

if TYPE_CHECKING:
    from collections.abc import Sequence

_VECTOR_SUFFIXES = frozenset({".geojson", ".gpkg", ".json", ".shp"})
_RASTER_SUFFIXES = frozenset({".tif", ".tiff"})


@overload
def assign_population(
    hazard: gpd.GeoDataFrame,
    population: RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    antimeridian: Literal["error", "split"] = "error",
    allow_overlaps: bool = False,
    allow_reprojection: bool = False,
    allow_partial_coverage: bool = False,
    allow_missing_population_data: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float | None = None,
) -> gpd.GeoDataFrame: ...


@overload
def assign_population(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    antimeridian: Literal["error", "split"] = "error",
    allow_overlaps: bool = False,
    allow_reprojection: bool = False,
    allow_partial_coverage: bool = False,
    allow_missing_population_data: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float | None = None,
) -> pd.DataFrame: ...


@overload
def assign_population(
    hazard: DatasetReader,
    population: RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    antimeridian: Literal["error", "split"] = "error",
    allow_overlaps: bool = False,
    allow_reprojection: bool = False,
    allow_partial_coverage: bool = False,
    allow_missing_population_data: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float | None = None,
) -> RasterAssignment: ...


@overload
def assign_population(
    hazard: str | PathLike[str],
    population: RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    antimeridian: Literal["error", "split"] = "error",
    allow_overlaps: bool = False,
    allow_reprojection: bool = False,
    allow_partial_coverage: bool = False,
    allow_missing_population_data: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float | None = None,
) -> gpd.GeoDataFrame | RasterAssignment: ...


def assign_population(
    hazard: pd.DataFrame | gpd.GeoDataFrame | str | PathLike[str] | DatasetReader,
    population: pd.DataFrame | RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    antimeridian: Literal["error", "split"] = "error",
    allow_overlaps: bool = False,
    allow_reprojection: bool = False,
    allow_partial_coverage: bool = False,
    allow_missing_population_data: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float | None = None,
) -> pd.DataFrame | gpd.GeoDataFrame | RasterAssignment:
    """Return hazards with an estimated population represented by a source/year.

    Spatial hazard and population inputs must share one coordinate system, and
    spatial hazards must sit inside the population raster. Table assignment
    uses exact keys and has no coordinate system. Each spatial rule can be
    relaxed with an explicit opt-in.

    The result is not a count of observed people, exact households, or
    event-time presence. Vector allocation uses the covered share of each
    population cell; raster allocation uses coverage-weighted sum resampling;
    table allocation uses an exact key join. A finer output grid does not add
    demographic detail.

    Two separate facts are reported about spatial hazards. Coverage says how
    much of the hazard sits inside the population raster's outer edge, and is
    required to be complete by default. Data support says how much of it has
    real population values rather than no-data. Partial data support is allowed
    and reported, because coastlines are made of it; only a hazard with no
    values at all raises, and that too can be allowed explicitly.

    Args:
        hazard: A pandas table, GeoDataFrame, vector file path, GeoTIFF path,
            or open Rasterio reader.
        population: A pandas table for table hazards. For maps and rasters, a
            population-count GeoTIFF path, an open Rasterio reader, or a
            catalog selection such as ``"worldpop-global-1km:2020"``.
        cell_columns: One key column, or a sequence of key columns, used only
            for table assignment.
        population_column: Name of the population column to append.
        antimeridian: For geographic vector hazards, ``"error"`` keeps the safe
            default and rejects unsplit seam crossings. ``"split"`` interprets
            those edges as the shorter route and normalizes temporary assignment
            geometry. Returned geometry is unchanged.
        allow_overlaps: True to allow overlapping vector polygons. It applies
            only to vector hazards.
        allow_reprojection: True to transform vector geometry to the population
            coordinate system, or to warp a population raster to the hazard
            coordinate system and grid. It applies only to vector and raster
            hazards.
        allow_partial_coverage: True to allow a hazard that reaches outside the
            population raster, and to report how much of it was covered. The
            share is not the share of population captured and must not scale a
            partial total. Vector features report an approximate physical
            surface-area share; raster hazards report a footprint-area share on
            the population grid. It applies to vector and raster hazards.
        allow_missing_population_data: True to allow a hazard the population
            raster holds no values for anywhere. Vector features get ``NaN``
            rather than a number, because no-data is not a count of zero
            people. Partly missing data needs no opt-in; it is always allowed
            and always reported. It applies to vector and raster hazards.
        hazard_band: A one-based band number for multiband hazard rasters.
        conservation_tolerance: The allowed relative difference between the
            population covered by the hazard footprint and the population
            aligned to the hazard grid. It measures the arithmetic of
            regridding only; it is not a completeness or uncertainty measure.
            It applies only to raster hazards and defaults to ``1e-6``, or
            ``1e-3`` when reprojection is used.

    Returns:
        pandas.DataFrame | geopandas.GeoDataFrame | RasterAssignment: The
        hazard input with the estimated population represented by the
        selected source and reference year assigned, matching the input
        type.

    Raises:
        population_exposure.CrsMismatchError: If the coordinate systems
            differ and reprojection was not allowed.
        population_exposure.MissingPopulationDataError: If the population
            raster has no values where the hazard sits and missing data was
            not allowed.
        population_exposure.PartialCoverageError: If the hazard reaches
            outside the population raster and partial coverage was not
            allowed.
        TypeError: If the hazard and population types cannot be combined.
        ValueError: If an option does not apply to the hazard type, or an input
            cannot be used.

    Examples:
        >>> import pandas as pd
        >>> import population_exposure as pe
        >>> hazard = pd.DataFrame({"cell": ["A"]})
        >>> population = pd.DataFrame({"cell": ["A"], "population": [10.0]})
        >>> # Illustrative values: no external source or reference year.
        >>> pe.assign_population(hazard, population, cell_columns="cell")
          cell  population
        0    A        10.0
    """
    _validate_common_options(
        population_column=population_column,
        antimeridian=antimeridian,
        allow_overlaps=allow_overlaps,
        allow_reprojection=allow_reprojection,
        allow_partial_coverage=allow_partial_coverage,
        allow_missing_population_data=allow_missing_population_data,
        hazard_band=hazard_band,
        conservation_tolerance=conservation_tolerance,
    )

    if isinstance(hazard, gpd.GeoDataFrame):
        if isinstance(population, pd.DataFrame):
            raise TypeError("Vector hazards require a population raster.")
        if hazard_band is not None:
            raise ValueError("hazard_band applies only to raster hazards.")
        return assign_vector_population(
            hazard,
            population,
            population_column=population_column,
            antimeridian=antimeridian,
            allow_overlaps=allow_overlaps,
            allow_reprojection=allow_reprojection,
            allow_partial_coverage=allow_partial_coverage,
            allow_missing_population_data=allow_missing_population_data,
        )

    if isinstance(hazard, DatasetReader):
        if isinstance(population, pd.DataFrame):
            raise TypeError("Raster hazards require a population raster.")
        _reject_vector_only_options(
            antimeridian=antimeridian,
            allow_overlaps=allow_overlaps,
        )
        return assign_raster_population(
            hazard,
            population,
            population_column=population_column,
            hazard_band=hazard_band,
            conservation_tolerance=conservation_tolerance,
            allow_reprojection=allow_reprojection,
            allow_partial_coverage=allow_partial_coverage,
            allow_missing_population_data=allow_missing_population_data,
        )

    if isinstance(hazard, (str, PathLike)):
        path = _existing_path(hazard, parameter="hazard")
        suffix = path.suffix.lower()
        if suffix in _VECTOR_SUFFIXES:
            if isinstance(population, pd.DataFrame):
                raise TypeError("Vector hazards require a population raster.")
            if hazard_band is not None:
                raise ValueError("hazard_band applies only to raster hazards.")
            return assign_vector_population(
                path,
                population,
                population_column=population_column,
                antimeridian=antimeridian,
                allow_overlaps=allow_overlaps,
                allow_reprojection=allow_reprojection,
                allow_partial_coverage=allow_partial_coverage,
                allow_missing_population_data=allow_missing_population_data,
            )
        if suffix in _RASTER_SUFFIXES:
            if isinstance(population, pd.DataFrame):
                raise TypeError("Raster hazards require a population raster.")
            _reject_vector_only_options(
                antimeridian=antimeridian,
                allow_overlaps=allow_overlaps,
            )
            return assign_raster_population(
                path,
                population,
                population_column=population_column,
                hazard_band=hazard_band,
                conservation_tolerance=conservation_tolerance,
                allow_reprojection=allow_reprojection,
                allow_partial_coverage=allow_partial_coverage,
                allow_missing_population_data=allow_missing_population_data,
            )
        supported = ", ".join(sorted(_VECTOR_SUFFIXES | _RASTER_SUFFIXES))
        raise ValueError(
            f"hazard path must use a supported extension ({supported}); got {suffix!r}."
        )

    if isinstance(hazard, pd.DataFrame):
        if not isinstance(population, pd.DataFrame):
            raise TypeError(
                "population must be a pandas DataFrame for tabular hazards."
            )
        _reject_vector_only_options(
            antimeridian=antimeridian,
            allow_overlaps=allow_overlaps,
        )
        _reject_spatial_only_options(
            allow_partial_coverage=allow_partial_coverage,
            allow_missing_population_data=allow_missing_population_data,
        )
        if allow_reprojection:
            raise ValueError(
                "allow_reprojection applies only to vector and raster hazards."
            )
        if hazard_band is not None:
            raise ValueError("hazard_band applies only to raster hazards.")
        return _assign_tabular_population(
            hazard,
            population,
            cell_columns=cell_columns,
            population_column=population_column,
        )

    raise TypeError(
        "hazard must be a pandas DataFrame, GeoDataFrame, supported vector path, "
        "GeoTIFF path, or open Rasterio DatasetReader."
    )


def _assign_tabular_population(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    cell_columns: str | Sequence[str],
    population_column: str,
) -> pd.DataFrame:
    """Assign population to tabular hazard rows by exact keys."""
    cells = normalize_columns(cell_columns, parameter="cell_columns")
    if population_column in cells:
        raise ValueError("population_column cannot also be a cell column.")

    require_columns(hazard, cells, frame_name="hazard")
    require_columns(population, (*cells, population_column), frame_name="population")
    _require_new_population_column(hazard, population_column)

    require_complete_keys(hazard, cells, frame_name="hazard")
    require_complete_keys(population, cells, frame_name="population")
    require_unique_keys(hazard, cells, frame_name="hazard")
    require_unique_keys(population, cells, frame_name="population")

    population_values = numeric_values(
        population,
        population_column,
        frame_name="population",
    )
    if not np.isfinite(population_values).all():
        raise ValueError("Population values must be finite.")
    if (population_values < 0).any():
        raise ValueError("Population values must be non-negative.")

    cell_aliases = tuple(f"__cell_{index}" for index in range(len(cells)))
    hazard_keys = hazard.loc[:, list(cells)].rename(
        columns=dict(zip(cells, cell_aliases, strict=True))
    )
    hazard_keys["__row_order"] = np.arange(len(hazard), dtype=np.int64)

    population_work = population.loc[:, list(cells)].rename(
        columns=dict(zip(cells, cell_aliases, strict=True))
    )
    population_work["__population"] = population_values

    assigned = hazard_keys.merge(
        population_work,
        how="left",
        on=list(cell_aliases),
        sort=False,
        validate="one_to_one",
        indicator="__match",
    ).sort_values("__row_order", kind="stable")
    unmatched = assigned["__match"].eq("left_only")
    if unmatched.any():
        count = int(unmatched.sum())
        noun = "row" if count == 1 else "rows"
        raise ValueError(
            f"{count} hazard {noun} did not match a population cell exactly."
        )

    result = hazard.copy(deep=False)
    result[population_column] = assigned["__population"].to_numpy(dtype=float)
    return result


def _validate_common_options(
    *,
    population_column: str,
    antimeridian: Literal["error", "split"],
    allow_overlaps: bool,
    allow_reprojection: bool,
    allow_partial_coverage: bool,
    allow_missing_population_data: bool,
    hazard_band: int | None,
    conservation_tolerance: float | None,
) -> None:
    """Validate options shared by all input types.

    Args:
        population_column: Name of the population column to append.
        antimeridian: How vector antimeridian crossings should be handled.
        allow_overlaps: True to allow overlapping vector polygons.
        allow_reprojection: True to allow automatic reprojection.
        allow_partial_coverage: True to allow partly covered hazards.
        allow_missing_population_data: True to allow hazards the population
            raster has no values for.
        hazard_band: A one-based band number, or None.
        conservation_tolerance: An explicit allowed relative difference, or
            None to use the default for the situation.

    Returns:
        None.

    Raises:
        TypeError: If a flag is not a boolean, or a band number is not an
            integer.
        ValueError: If the column name or tolerance cannot be used.

    Examples:
        >>> _validate_common_options(
        ...     population_column="population",
        ...     antimeridian="error",
        ...     allow_overlaps=False,
        ...     allow_reprojection=False,
        ...     allow_partial_coverage=False,
        ...     allow_missing_population_data=False,
        ...     hazard_band=None,
        ...     conservation_tolerance=None,
        ... )
    """
    if not isinstance(population_column, str) or not population_column:
        raise ValueError("population_column must be a non-empty column name.")
    if not isinstance(antimeridian, str) or antimeridian not in {"error", "split"}:
        raise ValueError("antimeridian must be 'error' or 'split'.")
    for name, value in (
        ("allow_overlaps", allow_overlaps),
        ("allow_reprojection", allow_reprojection),
        ("allow_partial_coverage", allow_partial_coverage),
        ("allow_missing_population_data", allow_missing_population_data),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    if hazard_band is not None and (
        isinstance(hazard_band, bool) or not isinstance(hazard_band, int)
    ):
        raise TypeError("hazard_band must be an integer band number or None.")
    if conservation_tolerance is not None and (
        isinstance(conservation_tolerance, bool)
        or not isinstance(conservation_tolerance, (int, float))
        or not np.isfinite(conservation_tolerance)
        or conservation_tolerance < 0
    ):
        raise ValueError("conservation_tolerance must be finite and non-negative.")


def _reject_vector_only_options(
    *,
    antimeridian: Literal["error", "split"],
    allow_overlaps: bool,
) -> None:
    """Reject vector-only options passed with another hazard type.

    Args:
        antimeridian: How vector antimeridian crossings should be handled.
        allow_overlaps: True to allow overlapping vector polygons.

    Returns:
        None.

    Raises:
        ValueError: If the option is turned on.

    Examples:
        >>> _reject_vector_only_options(
        ...     antimeridian="error",
        ...     allow_overlaps=False,
        ... )
    """
    if antimeridian != "error":
        raise ValueError("antimeridian applies only to vector hazards.")
    if allow_overlaps:
        raise ValueError("allow_overlaps applies only to vector hazards.")


def _reject_spatial_only_options(
    *,
    allow_partial_coverage: bool,
    allow_missing_population_data: bool,
) -> None:
    """Reject raster-backed options passed with a table hazard.

    Args:
        allow_partial_coverage: True to allow partly covered hazards.
        allow_missing_population_data: True to allow hazards the population
            raster has no values for.

    Returns:
        None.

    Raises:
        ValueError: If either option is turned on.

    Examples:
        >>> _reject_spatial_only_options(
        ...     allow_partial_coverage=False,
        ...     allow_missing_population_data=False,
        ... )
    """
    if allow_partial_coverage:
        raise ValueError(
            "allow_partial_coverage applies only to vector and raster hazards."
        )
    if allow_missing_population_data:
        raise ValueError(
            "allow_missing_population_data applies only to vector and raster hazards."
        )


def _existing_path(value: str | PathLike[str], *, parameter: str) -> Path:
    """Return a validated local file path."""
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{parameter} path does not exist or is not a file: {path}.")
    return path


def _require_new_population_column(
    hazard: pd.DataFrame,
    population_column: str,
) -> None:
    """Require assignment to append rather than overwrite a hazard column."""
    if population_column in hazard.columns:
        raise ValueError(
            f"hazard already has a column named {population_column!r}; "
            "choose a different population_column."
        )
