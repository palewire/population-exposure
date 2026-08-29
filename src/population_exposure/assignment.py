"""Population assignment dispatch for tabular and spatial hazard data."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, overload

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
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> gpd.GeoDataFrame: ...


@overload
def assign_population(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> pd.DataFrame: ...


@overload
def assign_population(
    hazard: DatasetReader,
    population: RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> RasterAssignment: ...


@overload
def assign_population(
    hazard: str | PathLike[str],
    population: RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> gpd.GeoDataFrame | RasterAssignment: ...


def assign_population(
    hazard: pd.DataFrame | gpd.GeoDataFrame | str | PathLike[str] | DatasetReader,
    population: pd.DataFrame | RasterSource,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> pd.DataFrame | gpd.GeoDataFrame | RasterAssignment:
    """Return hazard rows, features, or cells with assigned population."""
    _validate_common_options(
        population_column=population_column,
        allow_overlaps=allow_overlaps,
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
            allow_overlaps=allow_overlaps,
        )

    if isinstance(hazard, DatasetReader):
        if isinstance(population, pd.DataFrame):
            raise TypeError("Raster hazards require a population raster.")
        if allow_overlaps:
            raise ValueError("allow_overlaps applies only to vector hazards.")
        return assign_raster_population(
            hazard,
            population,
            population_column=population_column,
            hazard_band=hazard_band,
            conservation_tolerance=conservation_tolerance,
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
                allow_overlaps=allow_overlaps,
            )
        if suffix in _RASTER_SUFFIXES:
            if isinstance(population, pd.DataFrame):
                raise TypeError("Raster hazards require a population raster.")
            if allow_overlaps:
                raise ValueError("allow_overlaps applies only to vector hazards.")
            return assign_raster_population(
                path,
                population,
                population_column=population_column,
                hazard_band=hazard_band,
                conservation_tolerance=conservation_tolerance,
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
        if allow_overlaps:
            raise ValueError("allow_overlaps applies only to vector hazards.")
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
    allow_overlaps: bool,
    hazard_band: int | None,
    conservation_tolerance: float,
) -> None:
    """Validate options shared by all input types."""
    if not isinstance(population_column, str) or not population_column:
        raise ValueError("population_column must be a non-empty column name.")
    if not isinstance(allow_overlaps, bool):
        raise TypeError("allow_overlaps must be a boolean.")
    if hazard_band is not None and (
        isinstance(hazard_band, bool) or not isinstance(hazard_band, int)
    ):
        raise TypeError("hazard_band must be an integer band number or None.")
    if (
        isinstance(conservation_tolerance, bool)
        or not isinstance(conservation_tolerance, (int, float))
        or not np.isfinite(conservation_tolerance)
        or conservation_tolerance < 0
    ):
        raise ValueError("conservation_tolerance must be finite and non-negative.")


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
