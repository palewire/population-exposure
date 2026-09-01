"""Population assignment for vector hazards."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import geopandas as gpd
import numpy as np
import shapely
from exactextract import exact_extract
from pyogrio.errors import DataSourceError
from pyproj import CRS, Geod

from population_exposure._crs import (
    as_crs,
    boundary_tolerance,
    reject_wrapped_geometries,
    require_matching_crs,
    split_wrapped_geometries,
    transform_geometries,
)
from population_exposure._errors import MissingPopulationDataError, PartialCoverageError
from population_exposure.raster import (
    COVERAGE_TOLERANCE,
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
AntimeridianMode: TypeAlias = Literal["error", "split"]

COVERAGE_FRACTION_COLUMN = "population_coverage_fraction"
COVERAGE_COMPLETE_COLUMN = "population_coverage_complete"
DATA_FRACTION_COLUMN = "population_data_fraction"
DATA_COMPLETE_COLUMN = "population_data_complete"
_COVERAGE_COLUMNS = (COVERAGE_FRACTION_COLUMN, COVERAGE_COMPLETE_COLUMN)
_DATA_COLUMNS = (DATA_FRACTION_COLUMN, DATA_COMPLETE_COLUMN)

# The largest share below one, used to keep an incomplete share from rounding
# up to a complete-looking 1.0.
_JUST_BELOW_ONE = float(np.nextafter(1.0, 0.0))

_REPORTED_ROWS = 5
_SURFACE_AREA_CRS = CRS.from_epsg(4326)
_SURFACE_AREA_TOLERANCE = 1e-7
_WGS84_GEOD = Geod(ellps="WGS84")
_MAX_GEODESIC_SEGMENT_DEGREES = 0.1
# Limit only the temporary vertices created while splitting long geographic edges.
_MAX_ADDED_GEODESIC_VERTICES = 100_000
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
    antimeridian: AntimeridianMode,
    allow_overlaps: bool,
    allow_reprojection: bool,
    allow_partial_coverage: bool,
    allow_missing_population_data: bool,
) -> gpd.GeoDataFrame:
    """Assign estimated source/year population to polygon features.

    Each population cell contributes according to the share of its area covered
    by a feature. The result represents the selected source and reference year;
    it is not a count of observed people or event-time presence.

    Two separate facts are reported for every feature. Coverage says how much
    of the feature sits inside the population raster's outer edge, and must be
    complete unless partial coverage is allowed. Data support says how much of
    it has real population values rather than no-data; partial support is
    allowed and reported, and only a feature with no values at all raises.

    Args:
        hazard: A GeoDataFrame of polygons, or a path to a vector file.
        population: A population-count raster path, an open Rasterio reader, or
            a catalog selection.
        population_column: Name of the population column to append.
        antimeridian: ``"error"`` to reject wrapped boundaries, or ``"split"``
            to normalize them for assignment.
        allow_overlaps: True to allow overlapping polygons.
        allow_reprojection: True to transform the hazard geometry to the
            population coordinate system automatically.
        allow_partial_coverage: True to allow features that reach outside the
            population raster, and to report each feature's approximate physical
            surface-area share that was covered. This fraction is not the share
            of population captured and must not scale a partial total.
        allow_missing_population_data: True to return ``NaN`` instead of raising
            for a feature the population raster has no values for.

    Returns:
        geopandas.GeoDataFrame: The hazard features with population appended,
        plus the two data-support columns, plus the two coverage columns when
        partial coverage is allowed.

    Raises:
        population_exposure.CrsMismatchError: If the coordinate systems
            differ and reprojection was not allowed.
        population_exposure.MissingPopulationDataError: If the population
            raster has no values under a feature and missing data was not
            allowed.
        population_exposure.PartialCoverageError: If a feature reaches outside
            the population raster and partial coverage was not allowed.
        RuntimeError: If partial coverage is requested and a feature's
            intersection with the population raster has no polygonal area.
        ValueError: If the hazard geometry or column names cannot be used.

    Examples:
        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
        >>> assign_vector_population(  # doctest: +SKIP
        ...     hazard,
        ...     "population.tif",
        ...     population_column="population",
        ...     antimeridian="error",
        ...     allow_overlaps=False,
        ...     allow_reprojection=False,
        ...     allow_partial_coverage=False,
        ...     allow_missing_population_data=False,
        ... )
    """
    source = _load_vector(hazard)
    _validate_vector(
        source,
        population_column=population_column,
        antimeridian=antimeridian,
        allow_partial_coverage=allow_partial_coverage,
    )
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
        geometries = _geometries_on_population_grid(
            source,
            population_reader,
            reprojecting=reprojecting,
            antimeridian=antimeridian,
        )
        if (antimeridian == "split" or reprojecting) and not shapely.is_valid(
            np.asarray(geometries, dtype=object)
        ).all():
            raise ValueError("hazard vector contains invalid geometry.")
        if not allow_overlaps:
            _reject_overlaps(geometries)
        population_total = validate_population_raster(population_reader)
        population_metadata = metadata_for_reader(
            resolved_population,
            population_reader,
            total=population_total,
        )
        coverage = _coverage_fractions(geometries, population_reader)
        covered = coverage >= 1.0 - COVERAGE_TOLERANCE
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
        totals, valid_cells = _ordered_totals(
            summary,
            expected_rows=len(working),
            spatial_coverage=coverage,
        )
        data = _data_fractions(valid_cells, geometries, population_reader)
        _require_population_data(
            data,
            index=source.index,
            allow_missing_population_data=allow_missing_population_data,
        )
        surface_coverage = (
            _surface_coverage_fractions(
                geometries,
                population_reader,
                population_reader.crs,
                covered=covered,
            )
            if allow_partial_coverage
            else None
        )
        population_crs = population_reader.crs

    result = source
    result[population_column] = totals
    result[DATA_FRACTION_COLUMN] = data
    result[DATA_COMPLETE_COLUMN] = data == 1.0
    if allow_partial_coverage:
        assert surface_coverage is not None
        result[COVERAGE_FRACTION_COLUMN] = surface_coverage
        result[COVERAGE_COMPLETE_COLUMN] = covered
    result.attrs = {
        **source.attrs,
        "population_assignment": {
            "method": "exactextract_sum",
            "population_crs": population_crs.to_string(),
            "population_band": 1,
            "overlaps_allowed": allow_overlaps,
            "reprojected": reprojecting,
            "partial_coverage_allowed": allow_partial_coverage,
            "missing_population_data_allowed": allow_missing_population_data,
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
    antimeridian: AntimeridianMode,
    allow_partial_coverage: bool,
) -> None:
    """Validate polygon geometry and output-column safety.

    Args:
        hazard: The loaded hazard features.
        population_column: Name of the population column to append.
        antimeridian: How wrapped geographic boundaries should be handled.
        allow_partial_coverage: True when coverage columns will be added
            alongside the data-support columns.

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
        ...     antimeridian="error",
        ...     allow_partial_coverage=False,
        ... )
    """
    reserved = _DATA_COLUMNS + (_COVERAGE_COLUMNS if allow_partial_coverage else ())
    if population_column in reserved:
        raise ValueError(
            f"population_column cannot be {population_column!r}; assignment "
            f"adds that column itself. Assignment adds "
            f"{', '.join(repr(name) for name in reserved)}."
        )
    if population_column in hazard.columns:
        raise ValueError(
            f"hazard already has a column named {population_column!r}; "
            "choose a different population_column."
        )
    taken = [name for name in reserved if name in hazard.columns]
    if taken:
        names = ", ".join(repr(name) for name in taken)
        raise ValueError(
            f"hazard already has a column named {names}; rename it before "
            "assignment, which adds the "
            f"{', '.join(repr(name) for name in reserved)} columns."
        )
    if hazard.crs is None:
        raise ValueError("hazard vector must define a CRS.")
    if hazard.empty:
        raise ValueError("hazard vector must contain at least one polygon feature.")
    if hazard.geometry.isna().any():
        raise ValueError("hazard vector contains missing geometry.")
    if hazard.geometry.is_empty.any():
        raise ValueError("hazard vector contains empty geometry.")
    if antimeridian == "error" and not hazard.geometry.is_valid.all():
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
    antimeridian: AntimeridianMode,
) -> list[BaseGeometry]:
    """Return hazard geometry expressed in the population coordinate system.

    Args:
        hazard: The validated hazard features.
        population: The open population raster.
        reprojecting: True when the coordinate systems differ and the caller
            allowed automatic reprojection.
        antimeridian: How to handle wrapped geographic polygon boundaries.

    Returns:
        list[shapely.geometry.base.BaseGeometry]: One geometry per hazard
        feature, in hazard row order.

    Examples:
        >>> import geopandas as gpd
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _geometries_on_population_grid(
        ...         hazard, raster, reprojecting=False, antimeridian="error"
        ...     )
    """
    geometries = list(hazard.geometry.to_numpy())
    if antimeridian == "error":
        reject_wrapped_geometries(geometries, hazard_crs=hazard.crs)
    else:
        hazard_crs = as_crs(hazard.crs, parameter="hazard")
        population_crs = as_crs(population.crs, parameter="population")
        target_bounds = (
            (population.bounds.left, population.bounds.right)
            if hazard_crs == population_crs and population_crs.is_geographic
            else None
        )
        geometries = split_wrapped_geometries(
            geometries,
            hazard_crs=hazard_crs,
            target_longitude_bounds=target_bounds,
        )
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
    *,
    covered: np.ndarray,
) -> np.ndarray:
    """Return each feature's approximate covered share of physical Earth-surface area.

    The strict coverage rule uses the population grid's own plane. This
    separate calculation reports a projection-independent physical-area share
    without changing that rule. A feature the rule counts as fully covered is
    reported as exactly 1, so the share and the completeness flag agree.

    Args:
        geometries: Hazard geometry in the population raster's coordinate
            system.
        population: The open population raster.
        population_crs: The population raster's coordinate system.
        covered: True for each feature the strict rule counts as fully inside
            the population raster.

    Returns:
        numpy.ndarray: One share between 0 and 1 for each feature, measured on
        the WGS84 ellipsoid. It is exactly 1 for a covered feature and strictly
        less than 1 for any other.

    Raises:
        RuntimeError: If a feature's intersection with the population raster
            has no polygonal area.
        ValueError: If a polygon covers half or more of the Earth, which the
            geodesic area routine cannot measure unambiguously; has a
            non-positive or non-finite geodesic area; is empty; has a
            non-finite ring; or needs more than 100,000 added vertices while
            splitting long geographic edges.

    Examples:
        >>> import numpy as np
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _surface_coverage_fractions(
        ...         [box(0, 0, 1, 1)],
        ...         raster,
        ...         raster.crs,
        ...         covered=np.array([True]),
        ...     )
        array([1.])
    """
    footprint = raster_footprint(population)
    covered_geometries = [
        _polygonal_geometry(geometry.intersection(footprint)) for geometry in geometries
    ]
    geographic_geometries = _geographic_geometries(geometries, population_crs)
    geographic_covered = _geographic_geometries(covered_geometries, population_crs)
    full_areas = np.asarray(
        [_geodesic_area(geometry) for geometry in geographic_geometries],
        dtype=float,
    )
    covered_areas = np.asarray(
        [_geodesic_area(geometry) for geometry in geographic_covered],
        dtype=float,
    )
    fractions = np.clip(covered_areas / full_areas, 0.0, 1.0)
    # The geodesic measurement is approximate, so a barely incomplete feature
    # can measure a full share. Keep it strictly below one, so a fraction of
    # exactly 1 always means the strict rule counted the feature as covered.
    return np.where(covered, 1.0, np.minimum(fractions, _JUST_BELOW_ONE))


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
    if geometry.is_empty:
        raise RuntimeError("Population footprint intersection has no area.")
    parts = [
        part
        for part in shapely.get_parts(geometry)
        if isinstance(part, (shapely.Polygon, shapely.MultiPolygon))
    ]
    if not parts:
        raise RuntimeError("Population footprint intersection has no polygonal area.")
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
        RuntimeError: If the input geometry has no polygonal area.
        ValueError: If a polygon covers half or more of the Earth, which the
            geodesic area routine cannot measure unambiguously; has a
            non-positive or non-finite geodesic area; is empty; has a
            non-finite ring; or needs more than 100,000 added vertices while
            splitting long geographic edges.

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
        if not np.isfinite(area):
            raise ValueError(
                "population coverage cannot be measured because a polygon "
                "component has a non-finite geodesic area."
            )
        if abs(area) >= _WGS84_HALF_SURFACE_AREA * (1 - _HALF_SURFACE_TOLERANCE):
            raise ValueError(
                "population coverage cannot be measured for a polygon that "
                "covers half or more of the Earth. Split it into smaller "
                "polygons before assignment."
            )
        if area <= 0:
            raise ValueError(
                "population coverage cannot be measured because a polygon "
                "component has a non-positive geodesic area."
            )
        areas.append(area)
    return float(sum(areas))


def _densify_geographic_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Split geographic-coordinate edges before calculating geodesic area.

    The input's edges are straight in longitude and latitude, while
    ``Geod.geometry_area_perimeter`` follows geodesics between consecutive
    coordinates. Splitting every edge into at most 0.1-degree pieces keeps that
    difference below 1e-6 relative error for latitude-band areas while
    limiting a world-spanning rectangular ring to 10,801 vertices.

    Args:
        geometry: Polygon or multipolygon in WGS84 longitude and latitude.

    Returns:
        shapely.geometry.base.BaseGeometry: Geometry with geographic edges
        split into short straight pieces.

    Raises:
        ValueError: If the geometry or a ring is empty, has non-finite
            coordinates, or would require more than 100,000 added vertices.
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
    added_vertices = sum(
        _added_geodesic_ring_vertex_count(ring)
        for polygon in polygons
        for ring in (polygon.exterior, *polygon.interiors)
    )
    if added_vertices > _MAX_ADDED_GEODESIC_VERTICES:
        raise ValueError(
            "population coverage cannot be measured because densifying "
            f"geographic boundaries would add {added_vertices:,} vertices, "
            "exceeding the allowed budget of "
            f"{_MAX_ADDED_GEODESIC_VERTICES:,}. Simplify unusually long edges "
            "or split the work into smaller polygons."
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
        int: The number of vertices Shapely will produce after the 0.1-degree
            splitting limit.

    Raises:
        ValueError: If the ring is empty or has non-finite coordinates.

    Examples:
        >>> from shapely.geometry import box
        >>> _densified_ring_vertex_count(box(0, 0, 1, 1).exterior)
        41
    """
    segments = _geodesic_ring_segment_counts(ring)
    return int(segments.sum()) + 1


def _added_geodesic_ring_vertex_count(ring: shapely.LinearRing) -> int:
    """Return how many vertices splitting one ring would add.

    Args:
        ring: Closed WGS84 longitude-latitude ring.

    Returns:
        int: The number of additional vertices in Shapely's densified ring.

    Raises:
        ValueError: If the ring is empty or has non-finite coordinates.

    Examples:
        >>> from shapely.geometry import box
        >>> _added_geodesic_ring_vertex_count(box(0, 0, 1, 1).exterior)
        36
    """
    segments = _geodesic_ring_segment_counts(ring)
    return int(np.maximum(segments - 1, 0).sum())


def _geodesic_ring_segment_counts(ring: shapely.LinearRing) -> np.ndarray:
    """Return the pieces needed to split each edge of one ring.

    Args:
        ring: Closed WGS84 longitude-latitude ring.

    Returns:
        numpy.ndarray: The number of 0.1-degree pieces for each edge.

    Raises:
        ValueError: If the ring is empty or has non-finite coordinates.

    Examples:
        >>> from shapely.geometry import box
        >>> _geodesic_ring_segment_counts(box(0, 0, 1, 1).exterior).sum()
        np.float64(40.0)
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
    return np.ceil(lengths / _MAX_GEODESIC_SEGMENT_DEGREES)


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
    *,
    measure: str = "covered",
) -> str:
    """Name the affected hazard rows and how much of each was covered.

    Args:
        index: The hazard index.
        selected: A boolean mask of the affected rows.
        coverage: Each feature's share of area inside the raster.
        measure: The word used to describe the share, such as ``"covered"``.

    Returns:
        str: A readable list of row labels and shares, shortened when many rows
        are affected.

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
        f"{index[position]!r} {measure} {coverage[position]:.1%}" for position in shown
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


def _reject_overlaps(geometries: list[BaseGeometry]) -> None:
    """Reject feature pairs whose interiors share positive area."""
    values = np.asarray(geometries, dtype=object)
    pairs = shapely.STRtree(values).query(values, predicate="intersects")
    for left, right in zip(pairs[0], pairs[1], strict=True):
        if left >= right:
            continue
        intersection = shapely.intersection(values[left], values[right])
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactextract totals and valid-cell counts in original feature order.

    A feature with no valid cells gets ``NaN`` rather than zero. No-data records
    that the source has nothing to say about a place, so it cannot stand in for
    a count of zero people.

    Args:
        summary: ExactExtract output containing one row per hazard feature.
        expected_rows: Number of requested hazard features.
        spatial_coverage: Each feature's footprint coverage, already checked to
            be greater than zero.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: One non-negative population total
        per feature, which is ``NaN`` where no valid cells were found, and the
        matching valid-cell count, measured in whole and partial cells.

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
        (array([2.]), array([1.]))
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
    no_valid_cells = valid_cell_count <= 0
    totals = np.where(no_valid_cells, np.nan, totals)
    if not np.isfinite(totals[~no_valid_cells]).all():
        raise RuntimeError("Exactextract returned a non-finite population total.")
    if (totals[~no_valid_cells] < 0).any():
        raise RuntimeError("Exactextract returned a negative population total.")
    return totals, valid_cell_count


def _data_fractions(
    valid_cells: np.ndarray,
    geometries: list[BaseGeometry],
    population: DatasetReader,
) -> np.ndarray:
    """Return each feature's share of area holding real population values.

    No-data cells and area outside the raster both count as unsupported. The
    share is valid source-cell area measured in the population raster's own
    coordinate plane, so it can be compared directly with the grid-plane
    coverage the strict default tests. It is not physical Earth-surface area,
    unlike ``population_coverage_fraction``, and on a longitude-latitude raster
    the two differ materially away from the equator. It is also not a share of
    population, so it must never scale or extrapolate a partial total.

    Args:
        valid_cells: Each feature's valid-cell count from ExactExtract,
            measured in whole and partial cells.
        geometries: Hazard geometry already expressed in the population
            coordinate system.
        population: The open population raster.

    Returns:
        numpy.ndarray: One share between 0 and 1 for each feature, in hazard
        row order. It is exactly 1 when every part of the feature has values.

    Examples:
        >>> import numpy as np
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _data_fractions(np.array([1.0]), [box(0, 0, 1, 1)], raster)
        array([1.])
    """
    cell_area = abs(population.transform.determinant)
    areas = shapely.area(np.asarray(geometries, dtype=object))
    supported = valid_cells * cell_area
    fractions = np.divide(
        supported,
        areas,
        out=np.zeros(len(areas), dtype=float),
        where=areas > 0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    # Judge completeness against one cell, not against the feature. A share
    # measured relative to a feature spanning a billion cells would round a
    # whole missing cell away; a gap this much smaller than a single cell is
    # floating-point noise at any size. A feature with no valid cells at all is
    # never complete, however small it is.
    complete = (valid_cells > 0) & (areas - supported <= COVERAGE_TOLERANCE * cell_area)
    # Report exactly 1 when complete and strictly less than 1 otherwise, so the
    # share and the completeness flag can never disagree.
    return np.where(complete, 1.0, np.minimum(fractions, _JUST_BELOW_ONE))


def _require_population_data(
    data: np.ndarray,
    *,
    index: pd.Index,
    allow_missing_population_data: bool,
) -> None:
    """Require the population raster to hold values under every feature.

    Args:
        data: Each feature's share of area holding real population values.
        index: The hazard index, used to name the affected features.
        allow_missing_population_data: True when the caller accepted ``NaN``
            for features the raster has nothing to say about.

    Returns:
        None.

    Raises:
        population_exposure.MissingPopulationDataError: If the raster has no
            values under a feature and missing data was not allowed.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> _require_population_data(
        ...     np.array([1.0]),
        ...     index=pd.Index([0]),
        ...     allow_missing_population_data=False,
        ... )
    """
    if allow_missing_population_data:
        return
    missing = data <= 0.0
    if not missing.any():
        return
    count = int(missing.sum())
    noun = "feature" if count == 1 else "features"
    verb = "has" if count == 1 else "have"
    raise MissingPopulationDataError(
        f"{count} hazard {noun} {verb} no population values anywhere the raster "
        f"covers {'it' if count == 1 else 'them'}: "
        f"{_describe_rows(index, missing, data, measure='has population data for')}. "
        "Every cell the raster supplies there is no-data. No-data records that "
        "the population source has nothing to say about a place, so it is not "
        "evidence that nobody lives there and is not reported as zero. Use a "
        "population raster with values there, or opt in with "
        "pe.assign_population(hazard, population, "
        "allow_missing_population_data=True), which returns NaN for those "
        f"features alongside the {DATA_FRACTION_COLUMN!r} and "
        f"{DATA_COMPLETE_COLUMN!r} columns."
    )
