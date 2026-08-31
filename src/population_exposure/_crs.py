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
    half_turn = _longitude_half_turn(target, parameter="target")
    return [
        _transform_geometry(
            geometry,
            transformer=transformer,
            tolerance=tolerance,
            half_turn=half_turn,
        )
        for geometry in geometries
    ]


def reject_wrapped_geometries(
    geometries: Iterable[BaseGeometry],
    *,
    hazard_crs: object,
) -> None:
    """Reject polygon rings with an unsplit antimeridian edge.

    Args:
        geometries: ``Polygon`` or ``MultiPolygon`` hazard geometry.
        hazard_crs: The coordinate system of the hazard geometries.

    Returns:
        None.

    Raises:
        ValueError: If a geographic polygon ring has neighboring longitudes
            more than half a turn apart, or its longitude unit is unavailable.

    Examples:
        >>> from shapely.geometry import box
        >>> reject_wrapped_geometries(
        ...     [box(0, 0, 1, 1)],
        ...     hazard_crs="EPSG:4326",
        ... )
    """
    crs = as_crs(hazard_crs, parameter="hazard")
    half_turn = _longitude_half_turn(crs, parameter="hazard")
    if half_turn is None:
        return
    for geometry in geometries:
        polygons = (
            geometry.geoms if isinstance(geometry, shapely.MultiPolygon) else [geometry]
        )
        for polygon in polygons:
            _reject_wrapped(
                np.asarray(polygon.exterior.coords, dtype=float),
                half_turn=half_turn,
                error_type=ValueError,
            )
            for interior in polygon.interiors:
                _reject_wrapped(
                    np.asarray(interior.coords, dtype=float),
                    half_turn=half_turn,
                    error_type=ValueError,
                )


def _longitude_half_turn(crs: CRS, *, parameter: str) -> float | None:
    """Return half a turn in a geographic CRS's longitude unit.

    Args:
        crs: The parsed coordinate system.
        parameter: Input name used in error messages.

    Returns:
        float | None: Half a turn in longitude coordinate units, or None for a
            non-geographic coordinate system.

    Raises:
        ValueError: If a geographic CRS does not identify one usable longitude
            axis and angular unit.

    Examples:
        >>> _longitude_half_turn(CRS.from_epsg(4326), parameter="hazard")
        180.0
        >>> _longitude_half_turn(CRS.from_epsg(4807), parameter="hazard")
        200.0
    """
    if not crs.is_geographic:
        return None
    longitude_axes = [
        axis
        for axis in crs.axis_info
        if isinstance(axis.direction, str)
        and axis.direction.lower() in {"east", "west"}
    ]
    if len(longitude_axes) != 1:
        raise ValueError(
            f"{parameter} geographic CRS must define exactly one east-west "
            "longitude axis so antimeridian crossings can be checked."
        )
    radians_per_unit = longitude_axes[0].unit_conversion_factor
    if (
        radians_per_unit is None
        or not math.isfinite(radians_per_unit)
        or radians_per_unit <= 0
    ):
        raise ValueError(
            f"{parameter} geographic CRS longitude axis must define a positive "
            "finite angular unit conversion so antimeridian crossings can be checked."
        )
    half_turn = math.pi / radians_per_unit
    if not math.isfinite(half_turn) or half_turn <= 0:  # pragma: no cover
        raise ValueError(
            f"{parameter} geographic CRS longitude angular unit does not produce "
            "a usable half-turn value."
        )
    return half_turn


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
    half_turn: float | None,
) -> BaseGeometry:
    """Transform one polygon or multipolygon.

    Args:
        geometry: The ``Polygon`` or ``MultiPolygon`` to transform.
        transformer: The prepared PROJ transformer.
        tolerance: The allowed distance from the true curve, in target units.
        half_turn: Half a turn in target longitude units, or None when the
            target coordinate system is non-geographic.

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
        ...     half_turn=None,
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
                    half_turn=half_turn,
                )
                for part in geometry.geoms
            ]
        )
    if isinstance(geometry, shapely.Polygon):
        return _transform_polygon(
            geometry,
            transformer=transformer,
            tolerance=tolerance,
            half_turn=half_turn,
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
    half_turn: float | None,
) -> shapely.Polygon:
    """Transform a polygon's outer boundary and every hole it contains.

    Args:
        polygon: The polygon to transform.
        transformer: The prepared PROJ transformer.
        tolerance: The allowed distance from the true curve, in target units.
        half_turn: Half a turn in target longitude units, or None when the
            target coordinate system is non-geographic.

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
        ...     half_turn=None,
        ... ).is_valid
        True
    """
    shell = _transform_ring(
        np.asarray(polygon.exterior.coords, dtype=float),
        transformer=transformer,
        tolerance=tolerance,
        half_turn=half_turn,
    )
    holes = [
        _transform_ring(
            np.asarray(interior.coords, dtype=float),
            transformer=transformer,
            tolerance=tolerance,
            half_turn=half_turn,
        )
        for interior in polygon.interiors
    ]
    return shapely.Polygon(shell, holes)


def _transform_ring(
    source_points: np.ndarray,
    *,
    transformer: Transformer,
    tolerance: float,
    half_turn: float | None,
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
        half_turn: Half a turn in target longitude units, or None when the
            target coordinate system is non-geographic.

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
        ...     ring, transformer=transformer, tolerance=1.0, half_turn=180.0
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
        _reject_wrapped(
            target_points,
            half_turn=half_turn,
            error_type=CrsMismatchError,
        )
        raise CrsMismatchError(
            "hazard boundary could not be transformed accurately enough for "
            "the population grid. Transform the hazard yourself, or simplify "
            "geometry that touches the edge of a projection."
        )
    _reject_wrapped(
        target_points,
        half_turn=half_turn,
        error_type=CrsMismatchError,
    )
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


def _reject_wrapped(
    points: np.ndarray,
    *,
    half_turn: float | None,
    error_type: type[ValueError],
) -> None:
    """Reject a boundary that jumps across the antimeridian.

    Args:
        points: An ``(n, 2)`` array of transformed ring coordinates.
        half_turn: Half a turn in longitude coordinate units, or None when the
            coordinates are non-geographic.
        error_type: ValueError subclass to raise for a wrapped boundary.

    Returns:
        None.

    Raises:
        ValueError: If two neighboring points are more than half a turn apart,
            which means the boundary wrapped around the world instead of
            crossing the antimeridian.

    Examples:
        >>> _reject_wrapped(
        ...     np.array([[0.0, 0.0], [1.0, 1.0]]),
        ...     half_turn=180.0,
        ...     error_type=ValueError,
        ... )
    """
    if half_turn is None or len(points) < 2:
        return
    steps = np.abs(np.diff(points[:, 0]))
    if (steps > math.nextafter(half_turn, math.inf)).any():
        raise error_type(
            "hazard geometry has an unsplit boundary edge crossing the "
            "antimeridian, so it would wrap the long way around the world. "
            "Split the geometry at the antimeridian before assignment."
        )
