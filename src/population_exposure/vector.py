"""Population assignment for vector hazards."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import shapely
from exactextract import exact_extract
from pyogrio.errors import DataSourceError

from population_exposure.raster import (
    RasterSource,
    normalize_raster_source,
    open_raster,
    validate_population_raster,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

_ROW_ID = "__population_exposure_row__"
_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})


def assign_vector_population(
    hazard: gpd.GeoDataFrame | Path,
    population: RasterSource,
    *,
    population_column: str,
    allow_overlaps: bool,
) -> gpd.GeoDataFrame:
    """Assign coverage-aware population sums to polygon features."""
    source = _load_vector(hazard)
    _validate_vector(source, population_column=population_column)
    if not allow_overlaps:
        _reject_overlaps(source)

    population_source = normalize_raster_source(population, parameter="population")
    with open_raster(population_source, parameter="population") as population_reader:
        validate_population_raster(population_reader)
        working = source.to_crs(population_reader.crs)
        working[_ROW_ID] = np.arange(len(working), dtype=np.int64)
        summary = cast(
            "pd.DataFrame",
            exact_extract(
                population_reader,
                working,
                ["sum", "count"],
                include_cols=_ROW_ID,
                output="pandas",
                strategy="feature-sequential",
            ),
        )
        totals = _ordered_totals(summary, expected_rows=len(working))
        population_crs = population_reader.crs

    result = cast("gpd.GeoDataFrame", source.copy(deep=True))
    result[population_column] = totals
    result.attrs = {
        **source.attrs,
        "population_assignment": {
            "method": "exactextract_sum",
            "population_crs": population_crs.to_string(),
            "population_band": 1,
            "overlaps_allowed": allow_overlaps,
        },
    }
    return result


def _load_vector(hazard: gpd.GeoDataFrame | Path) -> gpd.GeoDataFrame:
    """Load a vector path or copy a caller-owned frame."""
    if isinstance(hazard, gpd.GeoDataFrame):
        return cast("gpd.GeoDataFrame", hazard.copy(deep=True))
    try:
        return cast("gpd.GeoDataFrame", gpd.read_file(hazard, engine="pyogrio"))
    except DataSourceError as error:
        raise ValueError(f"hazard vector could not be read: {hazard}.") from error


def _validate_vector(
    hazard: gpd.GeoDataFrame,
    *,
    population_column: str,
) -> None:
    """Validate polygon geometry and output-column safety."""
    if population_column in hazard.columns:
        raise ValueError(
            f"hazard already has a column named {population_column!r}; "
            "choose a different population_column."
        )
    if hazard.crs is None:
        raise ValueError("hazard vector must define a CRS.")
    if hazard.empty:
        raise ValueError("hazard vector must contain at least one polygon feature.")
    if hazard.geometry.isna().any():
        raise ValueError("hazard vector contains missing geometry.")
    if hazard.geometry.is_empty.any():
        raise ValueError("hazard vector contains empty geometry.")
    if not hazard.geometry.is_valid.all():
        raise ValueError("hazard vector contains invalid geometry.")
    non_polygon = ~hazard.geom_type.isin(_POLYGON_TYPES)
    if non_polygon.any():
        found = ", ".join(sorted(hazard.loc[non_polygon].geom_type.unique()))
        raise ValueError(
            "hazard vector must contain only Polygon or MultiPolygon geometry; "
            f"found {found}."
        )


def _reject_overlaps(hazard: gpd.GeoDataFrame) -> None:
    """Reject feature pairs whose interiors share positive area."""
    geometries = hazard.geometry.to_numpy()
    pairs = shapely.STRtree(geometries).query(geometries, predicate="intersects")
    for left, right in zip(pairs[0], pairs[1], strict=True):
        if left >= right:
            continue
        intersection = shapely.intersection(geometries[left], geometries[right])
        if shapely.area(intersection) > 0:
            raise ValueError(
                "hazard vector contains overlapping polygons at row positions "
                f"{left} and {right}; pass allow_overlaps=True to calculate "
                "independent, non-additive feature totals."
            )


def _ordered_totals(summary: pd.DataFrame, *, expected_rows: int) -> np.ndarray:
    """Return exactextract totals in original feature order."""
    required = {_ROW_ID, "sum", "count"}
    if not required.issubset(summary.columns) or len(summary) != expected_rows:
        raise RuntimeError("Exactextract returned an unexpected vector result.")
    ordered = summary.sort_values(_ROW_ID, kind="stable")
    row_ids = ordered[_ROW_ID].to_numpy(dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(expected_rows, dtype=np.int64)):
        raise RuntimeError("Exactextract did not return every vector feature once.")
    coverage = ordered["count"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(coverage).all() or (coverage <= 0).any():
        raise ValueError(
            "Every hazard polygon must overlap at least one valid population cell."
        )
    totals = ordered["sum"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(totals).all():
        raise RuntimeError("Exactextract returned a non-finite population total.")
    if (totals < 0).any():
        raise RuntimeError("Exactextract returned a negative population total.")
    return totals
