"""Population assignment for vector hazards."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import shapely
from exactextract import exact_extract
from pyogrio.errors import DataSourceError

from population_exposure._crs import (
    boundary_tolerance,
    require_matching_crs,
    transform_geometries,
)
from population_exposure._errors import PartialCoverageError
from population_exposure.raster import (
    RasterSource,
    normalize_raster_source,
    open_raster,
    raster_footprint,
    validate_population_raster,
    validate_raster_grid,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry

_ROW_ID = "__population_exposure_row__"
_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})

COVERAGE_FRACTION_COLUMN = "population_coverage_fraction"
COVERAGE_COMPLETE_COLUMN = "population_coverage_complete"
_COVERAGE_COLUMNS = (COVERAGE_FRACTION_COLUMN, COVERAGE_COMPLETE_COLUMN)

# A feature counts as fully covered when its share is within this much of one.
# The allowance absorbs floating-point noise, not a real sliver of missing area.
COVERAGE_TOLERANCE = 1e-9
_REPORTED_ROWS = 5


def assign_vector_population(
    hazard: gpd.GeoDataFrame | Path,
    population: RasterSource,
    *,
    population_column: str,
    allow_overlaps: bool,
    allow_reprojection: bool,
    allow_partial_coverage: bool,
) -> gpd.GeoDataFrame:
    """Assign coverage-aware population sums to polygon features.

    Args:
        hazard: A GeoDataFrame of polygons, or a path to a vector file.
        population: A population-count raster path, an open Rasterio reader, or
            a catalog selection.
        population_column: Name of the population column to append.
        allow_overlaps: True to allow overlapping polygons.
        allow_reprojection: True to transform the hazard onto the population
            coordinate system automatically.
        allow_partial_coverage: True to allow features that reach outside the
            population raster, and to report how much of each was covered.

    Returns:
        geopandas.GeoDataFrame: The hazard features with population appended,
        plus coverage columns when partial coverage is allowed.

    Raises:
        population_exposure.CrsMismatchError: If the coordinate systems
            differ and reprojection was not allowed.
        population_exposure.PartialCoverageError: If a feature reaches outside
            the population raster and partial coverage was not allowed.
        ValueError: If the hazard geometry or column names cannot be used.

    Examples:
        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
        >>> assign_vector_population(  # doctest: +SKIP
        ...     hazard,
        ...     "population.tif",
        ...     population_column="population",
        ...     allow_overlaps=False,
        ...     allow_reprojection=False,
        ...     allow_partial_coverage=False,
        ... )
    """
    source = _load_vector(hazard)
    _validate_vector(
        source,
        population_column=population_column,
        allow_partial_coverage=allow_partial_coverage,
    )
    if not allow_overlaps:
        _reject_overlaps(source)

    from population_exposure.populations._api import (
        metadata_for_reader,
        resolve_for_assignment,
    )

    resolved_population = resolve_for_assignment(population)
    population_source = normalize_raster_source(
        resolved_population.source,
        parameter="population",
    )
    with open_raster(population_source, parameter="population") as population_reader:
        validate_raster_grid(population_reader, name="population")
        reprojecting = require_matching_crs(
            source.crs,
            population_reader.crs,
            hazard_kind="vector",
            allow_reprojection=allow_reprojection,
        )
        population_total = validate_population_raster(population_reader)
        population_metadata = metadata_for_reader(
            resolved_population,
            population_reader,
            total=population_total,
        )
        geometries = _geometries_on_population_grid(
            source,
            population_reader,
            reprojecting=reprojecting,
        )
        coverage = _coverage_fractions(geometries, population_reader)
        _require_coverage(
            coverage,
            index=source.index,
            population=population_reader,
            geometries=geometries,
            allow_partial_coverage=allow_partial_coverage,
        )
        working = gpd.GeoDataFrame(
            {_ROW_ID: np.arange(len(geometries), dtype=np.int64)},
            geometry=geometries,
            crs=population_reader.crs,
        )
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

    result = source
    result[population_column] = totals
    if allow_partial_coverage:
        result[COVERAGE_FRACTION_COLUMN] = coverage
        result[COVERAGE_COMPLETE_COLUMN] = coverage >= 1.0 - COVERAGE_TOLERANCE
    result.attrs = {
        **source.attrs,
        "population_assignment": {
            "method": "exactextract_sum",
            "population_crs": population_crs.to_string(),
            "population_band": 1,
            "overlaps_allowed": allow_overlaps,
            "reprojected": reprojecting,
            "partial_coverage_allowed": allow_partial_coverage,
        },
        "population_source": population_metadata,
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
    allow_partial_coverage: bool,
) -> None:
    """Validate polygon geometry and output-column safety.

    Args:
        hazard: The loaded hazard features.
        population_column: Name of the population column to append.
        allow_partial_coverage: True when coverage columns will be added.

    Returns:
        None.

    Raises:
        ValueError: If an output column name is taken, the coordinate system is
            missing, or the geometry cannot be used.

    Examples:
        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
        >>> _validate_vector(
        ...     hazard,
        ...     population_column="population",
        ...     allow_partial_coverage=False,
        ... )
    """
    if population_column in hazard.columns:
        raise ValueError(
            f"hazard already has a column named {population_column!r}; "
            "choose a different population_column."
        )
    if allow_partial_coverage:
        taken = [name for name in _COVERAGE_COLUMNS if name in hazard.columns]
        if taken:
            names = ", ".join(repr(name) for name in taken)
            raise ValueError(
                f"hazard already has a column named {names}; rename it before "
                "using allow_partial_coverage=True, which adds the "
                f"{COVERAGE_FRACTION_COLUMN!r} and {COVERAGE_COMPLETE_COLUMN!r} "
                "columns."
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


def _geometries_on_population_grid(
    hazard: gpd.GeoDataFrame,
    population: DatasetReader,
    *,
    reprojecting: bool,
) -> list[BaseGeometry]:
    """Return hazard geometry expressed in the population coordinate system.

    Args:
        hazard: The validated hazard features.
        population: The open population raster.
        reprojecting: True when the coordinate systems differ and the caller
            allowed automatic reprojection.

    Returns:
        list[shapely.geometry.base.BaseGeometry]: One geometry per hazard
        feature, in hazard row order.

    Examples:
        >>> import geopandas as gpd
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _geometries_on_population_grid(hazard, raster, reprojecting=False)
    """
    geometries = list(hazard.geometry.to_numpy())
    if not reprojecting:
        return cast("list[BaseGeometry]", geometries)
    return transform_geometries(
        geometries,
        source_crs=hazard.crs,
        target_crs=population.crs,
        tolerance=boundary_tolerance(population),
    )


def _coverage_fractions(
    geometries: list[BaseGeometry],
    population: DatasetReader,
) -> np.ndarray:
    """Return each feature's share of area inside the population raster.

    The share is measured on the population raster's own grid plane, after any
    reprojection, so it matches the area the raster can actually supply. It is
    a share of the raster's support, not a share of the Earth's surface.

    Args:
        geometries: Hazard geometry already expressed in the population
            coordinate system.
        population: The open population raster.

    Returns:
        numpy.ndarray: One share between 0 and 1 for each feature, in hazard
        row order.

    Examples:
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _coverage_fractions([box(0, 0, 1, 1)], raster)
        array([1.])
    """
    footprint = raster_footprint(population)
    values = np.asarray(geometries, dtype=object)
    areas = shapely.area(values)
    inside = shapely.area(shapely.intersection(values, footprint))
    fractions = np.divide(
        inside,
        areas,
        out=np.zeros(len(values), dtype=float),
        where=areas > 0,
    )
    return np.clip(fractions, 0.0, 1.0)


def _require_coverage(
    coverage: np.ndarray,
    *,
    index: pd.Index,
    population: DatasetReader,
    geometries: list[BaseGeometry],
    allow_partial_coverage: bool,
) -> None:
    """Require every feature to sit inside the population raster's footprint.

    Completeness is measured against the raster's outer footprint. No-data
    cells inside that footprint, such as ocean or empty land, still count as
    covered, so ordinary coastal polygons are not rejected.

    Args:
        coverage: Each feature's share of area inside the raster.
        index: The hazard index, used to name the affected features.
        population: The open population raster.
        geometries: Hazard geometry in the population coordinate system.
        allow_partial_coverage: True when the caller accepted partial results.

    Returns:
        None.

    Raises:
        population_exposure.PartialCoverageError: If a feature is entirely
            outside the raster, or if a feature is partly outside it and
            partial coverage was not allowed.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _require_coverage(
        ...         np.array([1.0]),
        ...         index=pd.Index([0]),
        ...         population=raster,
        ...         geometries=[box(0, 0, 1, 1)],
        ...         allow_partial_coverage=False,
        ...     )
    """
    outside = coverage <= 0.0
    if outside.any():
        count = int(outside.sum())
        noun = "feature" if count == 1 else "features"
        verb = "falls" if count == 1 else "fall"
        raise PartialCoverageError(
            f"{count} hazard {noun} {verb} entirely outside the population "
            f"raster: {_describe_rows(index, outside, coverage)}. "
            f"{_wrapping_note(population, geometries, outside)}"
            "Use a population raster that covers the hazard, or revise the "
            "geometry."
        )
    if allow_partial_coverage:
        return
    partial = coverage < 1.0 - COVERAGE_TOLERANCE
    if not partial.any():
        return
    count = int(partial.sum())
    noun = "feature" if count == 1 else "features"
    verb = "reaches" if count == 1 else "reach"
    raise PartialCoverageError(
        f"{count} hazard {noun} {verb} outside the population raster: "
        f"{_describe_rows(index, partial, coverage)}. Complete coverage is "
        "required by default because the returned population would otherwise "
        "leave out everyone in the part of the feature the raster does not "
        f"reach. {_wrapping_note(population, geometries, partial)}"
        "Clip or revise the geometry so it fits inside the raster, or opt in "
        "with pe.assign_population(hazard, population, "
        "allow_partial_coverage=True), which returns the partial total plus "
        f"the {COVERAGE_FRACTION_COLUMN!r} and {COVERAGE_COMPLETE_COLUMN!r} "
        "columns."
    )


def _describe_rows(
    index: pd.Index,
    selected: np.ndarray,
    coverage: np.ndarray,
) -> str:
    """Name the affected hazard rows and how much of each was covered.

    Args:
        index: The hazard index.
        selected: A boolean mask of the affected rows.
        coverage: Each feature's share of area inside the raster.

    Returns:
        str: A readable list of row labels and covered shares, shortened when
        many rows are affected.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> _describe_rows(
        ...     pd.Index(["a", "b"]),
        ...     np.array([False, True]),
        ...     np.array([1.0, 0.5]),
        ... )
        "'b' covered 50.0%"
    """
    positions = np.flatnonzero(selected)
    shown = positions[:_REPORTED_ROWS]
    described = ", ".join(
        f"{index[position]!r} covered {coverage[position]:.1%}" for position in shown
    )
    remaining = len(positions) - len(shown)
    if remaining > 0:
        described = f"{described}, and {remaining} more"
    return described


def _wrapping_note(
    population: DatasetReader,
    geometries: list[BaseGeometry],
    selected: np.ndarray,
) -> str:
    """Explain longitude wrapping when geometry sits beyond a lon/lat raster.

    Args:
        population: The open population raster.
        geometries: Hazard geometry in the population coordinate system.
        selected: A boolean mask of the affected rows.

    Returns:
        str: A sentence about longitude wrapping, or an empty string when the
        population raster does not use longitude and latitude or the affected
        geometry stays inside its longitude range.

    Examples:
        >>> import numpy as np
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _wrapping_note(raster, [box(0, 0, 1, 1)], np.array([True]))
        ''
    """
    crs = population.crs
    if crs is None or not crs.is_geographic:  # pragma: no cover
        return ""
    left = population.bounds.left
    right = population.bounds.right
    for position in np.flatnonzero(selected):
        minimum, _, maximum, _ = geometries[position].bounds
        if minimum < left or maximum > right:
            return (
                "Some longitudes fall outside the raster's "
                f"{left:g} to {right:g} range. Longitudes are never wrapped "
                "automatically, so shift them into that range or split the "
                "geometry at the antimeridian. "
            )
    return ""


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
