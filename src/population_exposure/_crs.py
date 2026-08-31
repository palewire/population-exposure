"""Coordinate-system checks and boundary transformation shared by hazard types.

Hazard and population inputs must use the same coordinate system by default.
This module holds the one place where that rule is checked, where the error
that explains it is written, and where geometry is transformed accurately when
a caller explicitly opts in to automatic reprojection.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
from pyproj import CRS, Transformer

from population_exposure._errors import CrsMismatchError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry

HazardKind = Literal["vector", "raster"]

_MAX_REFINEMENTS = 16
_CELL_FRACTION = 0.1
_HALF_TURN_DEGREES = 180.0

_MANUAL_GUIDANCE: dict[HazardKind, str] = {
    "vector": (
        "Transform the hazard yourself with "
        'hazard.to_crs("{population}") before assignment'
    ),
    "raster": (
        "Warp one raster to the other's coordinate system yourself with "
        "rasterio.warp.reproject or gdalwarp before assignment"
    ),
}
_AUTOMATIC_GUIDANCE: dict[HazardKind, str] = {
    "vector": (
        "which adds enough points along every boundary to keep curves accurate "
        "when the projection changes"
    ),
    "raster": (
        "which warps population onto the hazard grid with count-preserving sum "
        "resampling and reports the resulting difference"
    ),
}


def as_crs(value: object, *, parameter: str) -> CRS:
    """Return a comparable coordinate system for a hazard or population input.

    Args:
        value: A coordinate system from GeoPandas, Rasterio, PROJ, or any
            string PROJ accepts, such as ``"EPSG:4326"``.
        parameter: The input name used in the error message, such as
            ``"hazard"``.

    Returns:
        pyproj.CRS: The same coordinate system in one comparable form.

    Raises:
        ValueError: If the value is missing or cannot be read as a coordinate
            system.

    Examples:
        >>> as_crs("EPSG:4326", parameter="hazard").to_epsg()
        4326
    """
    if value is None:
        raise ValueError(f"{parameter} must define a CRS.")
    try:
        return CRS.from_user_input(value)
    except Exception as error:
        raise ValueError(f"{parameter} CRS could not be read: {value!r}.") from error


def require_matching_crs(
    hazard_crs: object,
    population_crs: object,
    *,
    hazard_kind: HazardKind,
    allow_reprojection: bool,
) -> bool:
    """Check the coordinate systems of a hazard and a population raster.

    Args:
        hazard_crs: The hazard coordinate system.
        population_crs: The population raster coordinate system.
        hazard_kind: Either ``"vector"`` or ``"raster"``. It selects the manual
            guidance included in the error message.
        allow_reprojection: True when the caller explicitly asked for automatic
            reprojection.

    Returns:
        bool: True when the inputs must be reprojected, False when they already
        share one coordinate system.

    Raises:
        CrsMismatchError: If the coordinate systems differ and the caller did
            not ask for automatic reprojection.

    Examples:
        >>> require_matching_crs(
        ...     "EPSG:4326",
        ...     "EPSG:4326",
        ...     hazard_kind="vector",
        ...     allow_reprojection=False,
        ... )
        False
    """
    hazard = as_crs(hazard_crs, parameter="hazard")
    population = as_crs(population_crs, parameter="population")
    if hazard == population:
        return False
    if allow_reprojection:
        return True
    hazard_name = _crs_name(hazard)
    population_name = _crs_name(population)
    manual = _MANUAL_GUIDANCE[hazard_kind].format(population=population_name)
    automatic = _AUTOMATIC_GUIDANCE[hazard_kind]
    raise CrsMismatchError(
        "hazard and population coordinate systems do not match: hazard uses "
        f"{hazard_name} and population uses {population_name}. Matching "
        "coordinate systems are required by default because reprojection moves "
        "boundaries and can shift population into or out of the result. "
        f"{manual}, or opt in to automatic reprojection with "
        "pe.assign_population(hazard, population, allow_reprojection=True), "
        f"{automatic}."
    )


def boundary_tolerance(population: DatasetReader) -> float:
    """Return how far a transformed boundary may stray from the true curve.

    The allowance is a tenth of the shorter side of one population cell, so
    accuracy is measured against the grid the population is read from rather
    than a fixed number of points.

    Args:
        population: An open population raster.

    Returns:
        float: The allowed distance, in the population raster's own units.

    Raises:
        ValueError: If the population raster cell size is not usable.

    Examples:
        >>> import rasterio
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     boundary_tolerance(raster)
        0.1
    """
    transform = population.transform
    column = math.hypot(transform.a, transform.d)
    row = math.hypot(transform.b, transform.e)
    smaller = min(column, row)
    if not math.isfinite(smaller) or smaller <= 0:  # pragma: no cover
        raise ValueError("population raster must define a positive cell size.")
    return _CELL_FRACTION * smaller


def transform_geometries(
    geometries: Iterable[BaseGeometry],
    *,
    source_crs: object,
    target_crs: object,
    tolerance: float,
) -> list[BaseGeometry]:
    """Transform polygons to another coordinate system without losing curves.

    A straight line in one coordinate system is usually a curve in another.
    Transforming only the original corners would cut those curves off. This
    function adds points along every boundary until the transformed edge sits
    within ``tolerance`` of the true curve, then transforms the result.

    Args:
        geometries: ``Polygon`` or ``MultiPolygon`` geometry in the source
            coordinate system.
        source_crs: The coordinate system the geometry is written in.
        target_crs: The coordinate system to transform into.
        tolerance: The allowed distance between the transformed boundary and
            the true curve, in target coordinate system units.

    Returns:
        list[shapely.geometry.base.BaseGeometry]: The transformed geometry, in
        the same order as the input.

    Raises:
        CrsMismatchError: If a boundary cannot be represented in the target
            coordinate system, or if it crosses the antimeridian and would
            otherwise wrap around the world.

    Examples:
        >>> from shapely.geometry import box
        >>> transformed = transform_geometries(
        ...     [box(-1, -1, 1, 1)],
        ...     source_crs="EPSG:4326",
        ...     target_crs="ESRI:54009",
        ...     tolerance=100.0,
        ... )
        >>> transformed[0].geom_type
        'Polygon'
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    source = as_crs(source_crs, parameter="source")
    target = as_crs(target_crs, parameter="target")
    transformer = Transformer.from_crs(source, target, always_xy=True)
    geographic = bool(target.is_geographic)
    return [
        _transform_geometry(
            geometry,
            transformer=transformer,
            tolerance=tolerance,
            geographic=geographic,
        )
        for geometry in geometries
    ]


def _crs_name(crs: CRS) -> str:
    """Return a short, readable name for a coordinate system.

    Args:
        crs: The coordinate system to name.

    Returns:
        str: An authority code such as ``"EPSG:4326"`` when one is known,
        otherwise the coordinate system's own name.

    Examples:
        >>> _crs_name(CRS.from_user_input("EPSG:3857"))
        'EPSG:3857'
    """
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0]}:{authority[1]}"
    return crs.name


def _transform_geometry(
    geometry: BaseGeometry,
    *,
    transformer: Transformer,
    tolerance: float,
    geographic: bool,
) -> BaseGeometry:
    """Transform one polygon or multipolygon.

    Args:
        geometry: The ``Polygon`` or ``MultiPolygon`` to transform.
        transformer: The prepared PROJ transformer.
        tolerance: The allowed distance from the true curve, in target units.
        geographic: True when the target coordinate system uses longitude and
            latitude.

    Returns:
        shapely.geometry.base.BaseGeometry: The transformed geometry.

    Raises:
        CrsMismatchError: If the geometry cannot be transformed accurately.

    Examples:
        >>> from pyproj import Transformer
        >>> from shapely.geometry import box
        >>> transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        >>> _transform_geometry(
        ...     box(0, 0, 1, 1),
        ...     transformer=transformer,
        ...     tolerance=1.0,
        ...     geographic=False,
        ... ).geom_type
        'Polygon'
    """
    if isinstance(geometry, shapely.MultiPolygon):
        return shapely.MultiPolygon(
            [
                _transform_polygon(
                    part,
                    transformer=transformer,
                    tolerance=tolerance,
                    geographic=geographic,
                )
                for part in geometry.geoms
            ]
        )
    if isinstance(geometry, shapely.Polygon):
        return _transform_polygon(
            geometry,
            transformer=transformer,
            tolerance=tolerance,
            geographic=geographic,
        )
    raise CrsMismatchError(  # pragma: no cover
        "Only Polygon and MultiPolygon geometry can be reprojected; got "
        f"{geometry.geom_type}."
    )


def _transform_polygon(
    polygon: shapely.Polygon,
    *,
    transformer: Transformer,
    tolerance: float,
    geographic: bool,
) -> shapely.Polygon:
    """Transform a polygon's outer boundary and every hole it contains.

    Args:
        polygon: The polygon to transform.
        transformer: The prepared PROJ transformer.
        tolerance: The allowed distance from the true curve, in target units.
        geographic: True when the target coordinate system uses longitude and
            latitude.

    Returns:
        shapely.Polygon: The transformed polygon, holes included.

    Raises:
        CrsMismatchError: If a boundary cannot be transformed accurately.

    Examples:
        >>> from pyproj import Transformer
        >>> from shapely.geometry import box
        >>> transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        >>> _transform_polygon(
        ...     box(0, 0, 1, 1),
        ...     transformer=transformer,
        ...     tolerance=1.0,
        ...     geographic=False,
        ... ).is_valid
        True
    """
    shell = _transform_ring(
        np.asarray(polygon.exterior.coords, dtype=float),
        transformer=transformer,
        tolerance=tolerance,
        geographic=geographic,
    )
    holes = [
        _transform_ring(
            np.asarray(interior.coords, dtype=float),
            transformer=transformer,
            tolerance=tolerance,
            geographic=geographic,
        )
        for interior in polygon.interiors
    ]
    return shapely.Polygon(shell, holes)


def _transform_ring(
    source_points: np.ndarray,
    *,
    transformer: Transformer,
    tolerance: float,
    geographic: bool,
) -> np.ndarray:
    """Transform one closed ring, adding points where the boundary curves.

    Each pass transforms the middle of every remaining segment and compares it
    with the middle of the transformed segment. Where the two are further apart
    than ``tolerance``, the transformed middle is kept and the segment is split
    again. Straight-through transforms finish on the first pass.

    Args:
        source_points: An ``(n, 2)`` array of ring coordinates. The first and
            last points are the same.
        transformer: The prepared PROJ transformer.
        tolerance: The allowed distance from the true curve, in target units.
        geographic: True when the target coordinate system uses longitude and
            latitude.

    Returns:
        numpy.ndarray: An ``(m, 2)`` array of transformed ring coordinates,
        where ``m`` is at least ``n``.

    Raises:
        CrsMismatchError: If the ring cannot be represented within
            ``tolerance``, or if it wraps around the antimeridian.

    Examples:
        >>> from pyproj import Transformer
        >>> transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        >>> ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        >>> _transform_ring(
        ...     ring, transformer=transformer, tolerance=1.0, geographic=True
        ... ).shape
        (4, 2)
    """
    target_points = _transform_points(source_points, transformer)
    for _ in range(_MAX_REFINEMENTS):
        middles = 0.5 * (source_points[:-1] + source_points[1:])
        transformed_middles = _transform_points(middles, transformer)
        chord_middles = 0.5 * (target_points[:-1] + target_points[1:])
        offsets = transformed_middles - chord_middles
        deviation = np.hypot(offsets[:, 0], offsets[:, 1])
        needed = deviation > tolerance
        if not needed.any():
            break
        positions = np.flatnonzero(needed) + 1
        source_points = np.insert(source_points, positions, middles[needed], axis=0)
        target_points = np.insert(
            target_points, positions, transformed_middles[needed], axis=0
        )
    else:
        _reject_wrapped(target_points, geographic=geographic)
        raise CrsMismatchError(
            "hazard boundary could not be transformed accurately enough for "
            "the population grid. Transform the hazard yourself, or simplify "
            "geometry that touches the edge of a projection."
        )
    _reject_wrapped(target_points, geographic=geographic)
    return target_points


def _transform_points(points: np.ndarray, transformer: Transformer) -> np.ndarray:
    """Transform an array of coordinates and reject unusable results.

    Args:
        points: An ``(n, 2)`` array of source coordinates.
        transformer: The prepared PROJ transformer.

    Returns:
        numpy.ndarray: An ``(n, 2)`` array of transformed coordinates.

    Raises:
        CrsMismatchError: If any point falls outside the target coordinate
            system's usable area.

    Examples:
        >>> from pyproj import Transformer
        >>> transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        >>> _transform_points(np.array([[0.0, 0.0]]), transformer).round(3)
        array([[0., 0.]])
    """
    x_values, y_values = transformer.transform(points[:, 0], points[:, 1])
    transformed = np.column_stack(
        [np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)]
    )
    if not np.isfinite(transformed).all():
        raise CrsMismatchError(
            "hazard geometry reaches outside the area the population "
            "coordinate system can represent. Clip the hazard to that area, or "
            "transform it yourself."
        )
    return transformed


def _reject_wrapped(points: np.ndarray, *, geographic: bool) -> None:
    """Reject a boundary that jumps across the antimeridian.

    Args:
        points: An ``(n, 2)`` array of transformed ring coordinates.
        geographic: True when the coordinates are longitude and latitude.

    Returns:
        None.

    Raises:
        CrsMismatchError: If two neighboring points are more than 180 degrees
            of longitude apart, which means the boundary wrapped around the
            world instead of crossing the antimeridian.

    Examples:
        >>> _reject_wrapped(np.array([[0.0, 0.0], [1.0, 1.0]]), geographic=True)
    """
    if not geographic or len(points) < 2:
        return
    steps = np.abs(np.diff(points[:, 0]))
    if (steps > _HALF_TURN_DEGREES).any():
        raise CrsMismatchError(
            "hazard geometry crosses the antimeridian once transformed, so its "
            "boundary would wrap the long way around the world. Split the "
            "geometry at 180 degrees longitude before assignment."
        )
