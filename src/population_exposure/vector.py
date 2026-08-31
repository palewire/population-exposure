"""Population assignment for vector hazards."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import shapely
from exactextract import exact_extract
from pyogrio.errors import DataSourceError
from pyproj import CRS, Geod

from population_exposure._crs import (
    as_crs,
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
_SURFACE_AREA_CRS = CRS.from_epsg(4326)
_SURFACE_AREA_TOLERANCE = 1e-7
_WGS84_GEOD = Geod(ellps="WGS84")
_MAX_GEODESIC_SEGMENT_DEGREES = 0.1
_MAX_GEODESIC_RING_VERTICES = 100_000
_WGS84_ECCENTRICITY = math.sqrt(1 - (_WGS84_GEOD.b / _WGS84_GEOD.a) ** 2)
_WGS84_HALF_SURFACE_AREA = (
    math.pi
    * _WGS84_GEOD.a**2
    * (
        1
        + (1 - _WGS84_ECCENTRICITY**2)
        * math.atanh(_WGS84_ECCENTRICITY)
        / _WGS84_ECCENTRICITY
    )
)
_HALF_SURFACE_TOLERANCE = 1e-12


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
            population raster, and to report each feature's physical
            surface-area share that was covered. This fraction is not the share
            of population captured and must not scale a partial total.

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
        totals = _ordered_totals(
            summary,
            expected_rows=len(working),
            spatial_coverage=coverage,
        )
        surface_coverage = (
            _surface_coverage_fractions(
                geometries,
                population_reader,
                population_reader.crs,
            )
            if allow_partial_coverage
            else None
        )
        population_crs = population_reader.crs

    result = source
    result[population_column] = totals
    if allow_partial_coverage:
        assert surface_coverage is not None
        result[COVERAGE_FRACTION_COLUMN] = surface_coverage
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


def _surface_coverage_fractions(
    geometries: list[BaseGeometry],
    population: DatasetReader,
    population_crs: object,
) -> np.ndarray:
    """Return each feature's covered share of physical Earth-surface area.

    The strict coverage rule uses the population grid's own plane. This
    separate calculation reports a projection-independent physical-area share
    without changing that rule.

    Args:
        geometries: Hazard geometry in the population raster's coordinate
            system.
        population: The open population raster.
        population_crs: The population raster's coordinate system.

    Returns:
        numpy.ndarray: One share between 0 and 1 for each feature, measured on
        the WGS84 ellipsoid.

    Raises:
        ValueError: If a polygon covers half or more of the Earth, which the
            geodesic area routine cannot measure unambiguously; is empty; has a
            non-finite ring; or needs more than 100,000 vertices in one ring.

    Examples:
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _surface_coverage_fractions([box(0, 0, 1, 1)], raster, raster.crs)
        array([1.])
    """
    footprint = raster_footprint(population)
    covered = [
        _polygonal_geometry(geometry.intersection(footprint)) for geometry in geometries
    ]
    geographic_geometries = _geographic_geometries(geometries, population_crs)
    geographic_covered = _geographic_geometries(covered, population_crs)
    full_areas = np.asarray(
        [_geodesic_area(geometry) for geometry in geographic_geometries],
        dtype=float,
    )
    covered_areas = np.asarray(
        [_geodesic_area(geometry) for geometry in geographic_covered],
        dtype=float,
    )
    fractions = covered_areas / full_areas
    return np.clip(fractions, 0.0, 1.0)


def _geographic_geometries(
    geometries: list[BaseGeometry],
    source_crs: object,
) -> list[BaseGeometry]:
    """Return polygon geometries in WGS84 longitude and latitude.

    Args:
        geometries: Polygon or multipolygon geometry in ``source_crs``.
        source_crs: The coordinate system of the geometries.

    Returns:
        list[shapely.geometry.base.BaseGeometry]: The input geometry in WGS84
        longitude and latitude.

    Examples:
        >>> from shapely.geometry import box
        >>> _geographic_geometries([box(0, 0, 1, 1)], "EPSG:4326")[0].geom_type
        'Polygon'
    """
    if as_crs(source_crs, parameter="population") == _SURFACE_AREA_CRS:
        return geometries
    return transform_geometries(
        geometries,
        source_crs=source_crs,
        target_crs=_SURFACE_AREA_CRS,
        tolerance=_SURFACE_AREA_TOLERANCE,
    )


def _polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Return the polygonal parts of an area intersection.

    Args:
        geometry: Geometry created by an intersection.

    Returns:
        shapely.geometry.base.BaseGeometry: A polygon or multipolygon.

    Raises:
        RuntimeError: If the intersection has no polygonal area.

    Examples:
        >>> from shapely.geometry import box
        >>> _polygonal_geometry(box(0, 0, 1, 1)).geom_type
        'Polygon'
    """
    if isinstance(geometry, (shapely.Polygon, shapely.MultiPolygon)):
        return geometry
    parts = [
        part
        for part in shapely.get_parts(geometry)
        if isinstance(part, (shapely.Polygon, shapely.MultiPolygon))
    ]
    if not parts:
        raise RuntimeError("Population footprint intersection has no area.")
    polygonal = shapely.union_all(parts)
    if not isinstance(polygonal, (shapely.Polygon, shapely.MultiPolygon)):
        raise RuntimeError("Population footprint intersection has no polygonal area.")
    return polygonal


def _geodesic_area(geometry: BaseGeometry) -> float:
    """Return a polygon or multipolygon's physical area on the WGS84 ellipsoid.

    Args:
        geometry: Polygon or multipolygon in WGS84 longitude and latitude.

    Returns:
        float: The positive physical area in square meters.

    Raises:
        ValueError: If a polygon covers half or more of the Earth, which the
            geodesic area routine cannot measure unambiguously; is empty; has a
            non-finite ring; or needs more than 100,000 vertices in one ring.

    Examples:
        >>> from shapely.geometry import box
        >>> _geodesic_area(box(0, 0, 1, 1)) > 0
        True
    """
    normalized = shapely.orient_polygons(_polygonal_geometry(geometry))
    densified = _densify_geographic_geometry(normalized)
    polygons = (
        densified.geoms if isinstance(densified, shapely.MultiPolygon) else (densified,)
    )
    areas: list[float] = []
    for polygon in polygons:
        area, _ = _WGS84_GEOD.geometry_area_perimeter(polygon)
        if (
            not np.isfinite(area)
            or area <= 0
            or area >= _WGS84_HALF_SURFACE_AREA * (1 - _HALF_SURFACE_TOLERANCE)
        ):
            raise ValueError(
                "population coverage cannot be measured for a polygon that "
                "covers half or more of the Earth. Split it into smaller "
                "polygons before assignment."
            )
        areas.append(area)
    return float(sum(areas))


def _densify_geographic_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Split geographic-coordinate edges before calculating geodesic area.

    The input's edges are straight in longitude and latitude, while
    ``Geod.geometry_area_perimeter`` follows geodesics between consecutive
    coordinates. Splitting every edge into at most 0.1-degree pieces keeps that
    difference below 2.5e-7 relative error for latitude-band areas while
    limiting a world-spanning rectangular ring to 10,801 vertices.

    Args:
        geometry: Polygon or multipolygon in WGS84 longitude and latitude.

    Returns:
        shapely.geometry.base.BaseGeometry: Geometry with geographic edges
        split into short straight pieces.

    Raises:
        ValueError: If the geometry or a ring is empty, has non-finite
            coordinates, or would require more than 100,000 vertices.
        RuntimeError: If densification does not produce polygonal geometry.

    Examples:
        >>> from shapely.geometry import box
        >>> len(_densify_geographic_geometry(box(0, 0, 1, 1)).exterior.coords)
        41
    """
    if geometry.is_empty:
        raise ValueError(
            "population coverage cannot be measured for an empty WGS84 polygon."
        )
    polygons = (
        geometry.geoms if isinstance(geometry, shapely.MultiPolygon) else (geometry,)
    )
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            vertices = _densified_ring_vertex_count(ring)
            if vertices > _MAX_GEODESIC_RING_VERTICES:
                raise ValueError(
                    "population coverage cannot be measured because a polygon "
                    "ring would need more than "
                    f"{_MAX_GEODESIC_RING_VERTICES:,} vertices."
                )
    densified = shapely.segmentize(geometry, _MAX_GEODESIC_SEGMENT_DEGREES)
    if not isinstance(densified, (shapely.Polygon, shapely.MultiPolygon)):
        raise RuntimeError(
            "Geographic boundary densification did not produce polygonal geometry."
        )
    return densified


def _densified_ring_vertex_count(ring: shapely.LinearRing) -> int:
    """Return the vertex count after splitting one geographic-coordinate ring.

    Args:
        ring: Closed WGS84 longitude-latitude ring.

    Returns:
        int: The number of vertices after the 0.1-degree splitting limit.

    Raises:
        ValueError: If the ring is empty or has non-finite coordinates.

    Examples:
        >>> from shapely.geometry import box
        >>> _densified_ring_vertex_count(box(0, 0, 1, 1).exterior)
        41
    """
    coordinates = np.asarray(ring.coords, dtype=float)
    if coordinates.ndim != 2 or len(coordinates) < 2:
        raise ValueError(
            "population coverage cannot be measured for an empty WGS84 ring."
        )
    coordinates = coordinates[:, :2]
    if not np.isfinite(coordinates).all():
        raise ValueError(
            "population coverage cannot be measured for non-finite WGS84 coordinates."
        )
    differences = np.diff(coordinates, axis=0)
    lengths = np.hypot(differences[:, 0], differences[:, 1])
    segments = np.maximum(
        np.ceil(lengths / _MAX_GEODESIC_SEGMENT_DEGREES),
        1,
    )
    return int(segments.sum()) + 1


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


def _ordered_totals(
    summary: pd.DataFrame,
    *,
    expected_rows: int,
    spatial_coverage: np.ndarray,
) -> np.ndarray:
    """Return exactextract totals in original feature order.

    Args:
        summary: ExactExtract output containing one row per hazard feature.
        expected_rows: Number of requested hazard features.
        spatial_coverage: Each feature's footprint coverage, already checked to
            be greater than zero.

    Returns:
        numpy.ndarray: One finite, non-negative population total per feature.

    Raises:
        RuntimeError: If ExactExtract returns an incomplete or unusable result.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> _ordered_totals(
        ...     pd.DataFrame(
        ...         {
        ...             _ROW_ID: [0],
        ...             "sum": [2.0],
        ...             "count": [1.0],
        ...         }
        ...     ),
        ...     expected_rows=1,
        ...     spatial_coverage=np.array([1.0]),
        ... )
        array([2.])
    """
    required = {_ROW_ID, "sum", "count"}
    if not required.issubset(summary.columns) or len(summary) != expected_rows:
        raise RuntimeError("Exactextract returned an unexpected vector result.")
    if (
        spatial_coverage.shape != (expected_rows,)
        or not np.isfinite(spatial_coverage).all()
        or (spatial_coverage <= 0).any()
    ):
        raise RuntimeError("Vector coverage must be checked before population totals.")
    ordered = summary.sort_values(_ROW_ID, kind="stable")
    row_ids = ordered[_ROW_ID].to_numpy(dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(expected_rows, dtype=np.int64)):
        raise RuntimeError("Exactextract did not return every vector feature once.")
    valid_cell_count = ordered["count"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(valid_cell_count).all() or (valid_cell_count < 0).any():
        raise RuntimeError("Exactextract returned an invalid valid-cell count.")
    totals = ordered["sum"].to_numpy(dtype=float, na_value=np.nan)
    no_valid_cells = valid_cell_count == 0
    totals = np.where(no_valid_cells, 0.0, totals)
    if not np.isfinite(totals).all():
        raise RuntimeError("Exactextract returned a non-finite population total.")
    if (totals < 0).any():
        raise RuntimeError("Exactextract returned a negative population total.")
    return totals
