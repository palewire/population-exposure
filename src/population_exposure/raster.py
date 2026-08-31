"""Population assignment for raster hazards."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

import geopandas as gpd
import numpy as np
import rasterio
from exactextract import exact_extract
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import Polygon

from population_exposure._crs import (
    boundary_tolerance,
    require_matching_crs,
    transform_geometries,
)
from population_exposure._errors import PartialCoverageError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from rasterio import CRS, Affine
    from rasterio.coords import BoundingBox
    from shapely.geometry.base import BaseGeometry

RasterSource: TypeAlias = str | PathLike[str] | DatasetReader
_DEFAULT_BLOCK_SHAPE = (256, 256)

# Aligning two rasters that already share a coordinate system is arithmetic, so
# only floating-point noise is expected. Warping between coordinate systems
# carries GDAL's own small allocation difference, measured near 1e-4.
SAME_CRS_CONSERVATION_TOLERANCE = 1e-6
CROSS_CRS_CONSERVATION_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class RasterAssignment:
    """Lazy, window-readable hazard and aligned population raster cells."""

    shape: tuple[int, int]
    crs: CRS
    transform: Affine
    bounds: BoundingBox
    hazard_band: int
    attrs: Mapping[str, object]
    _hazard: RasterSource = field(repr=False)
    _population: RasterSource = field(repr=False)
    _block_shape: tuple[int, int] = field(repr=False)

    def read(
        self,
        window: Window | None = None,
    ) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
        """Read hazard and aligned population values for the same cells."""
        with open_raster(self._hazard, parameter="hazard") as hazard:
            hazard_values = hazard.read(self.hazard_band, window=window, masked=True)
        with open_raster(self._population, parameter="population") as population:
            with _aligned_population_reader(
                population,
                crs=self.crs,
                transform=self.transform,
                shape=self.shape,
            ) as aligned:
                population_values = aligned.read(1, window=window, masked=True)
        return hazard_values, population_values

    def iter_blocks(
        self,
    ) -> Iterator[tuple[Window, np.ma.MaskedArray, np.ma.MaskedArray]]:
        """Yield aligned hazard and population arrays one bounded window at a time."""
        with (
            open_raster(self._hazard, parameter="hazard") as hazard,
            open_raster(self._population, parameter="population") as population,
            _aligned_population_reader(
                population,
                crs=self.crs,
                transform=self.transform,
                shape=self.shape,
            ) as aligned,
        ):
            for window in _windows(self.shape, self._block_shape):
                yield (
                    window,
                    hazard.read(self.hazard_band, window=window, masked=True),
                    aligned.read(1, window=window, masked=True),
                )


def assign_raster_population(
    hazard: RasterSource,
    population: RasterSource,
    *,
    population_column: str,
    hazard_band: int | None,
    conservation_tolerance: float | None,
    allow_reprojection: bool,
) -> RasterAssignment:
    """Validate rasters and return a lazy population-aligned result.

    Args:
        hazard: A hazard GeoTIFF path or open Rasterio reader.
        population: A population-count raster path, an open Rasterio reader, or
            a catalog selection.
        population_column: Name recorded for the aligned population values.
        hazard_band: A one-based hazard band number, or None for a one-band
            hazard raster.
        conservation_tolerance: The allowed relative difference between the
            covered and aligned population totals, or None to use the default
            for the situation.
        allow_reprojection: True to warp population from another coordinate
            system onto the hazard grid automatically.

    Returns:
        RasterAssignment: A lazy result that reads hazard and aligned
        population cells together.

    Raises:
        population_exposure.CrsMismatchError: If the coordinate systems
            differ and reprojection was not allowed.
        population_exposure.PartialCoverageError: If the hazard raster lies
            entirely outside the population raster.
        ValueError: If a raster cannot be used, or population is not conserved
            within the allowed difference.

    Examples:
        >>> assign_raster_population(  # doctest: +SKIP
        ...     "hazard.tif",
        ...     "population.tif",
        ...     population_column="population",
        ...     hazard_band=None,
        ...     conservation_tolerance=None,
        ...     allow_reprojection=False,
        ... )
    """
    hazard_source = normalize_raster_source(hazard, parameter="hazard")

    with open_raster(hazard_source, parameter="hazard") as hazard_reader:
        selected_band = _select_hazard_band(hazard_reader, hazard_band)
        validate_raster_grid(hazard_reader, name="hazard")
        hazard_crs = hazard_reader.crs
        assert hazard_crs is not None
        shape = hazard_reader.shape
        transform = hazard_reader.transform
        bounds = hazard_reader.bounds
        block_shape = _safe_block_shape(
            hazard_reader.block_shapes[selected_band - 1],
            shape,
        )
        footprint = raster_footprint(hazard_reader)

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
            hazard_crs,
            population_reader.crs,
            hazard_kind="raster",
            allow_reprojection=allow_reprojection,
        )
        source_total = validate_population_raster(population_reader)
        population_metadata = metadata_for_reader(
            resolved_population,
            population_reader,
            total=source_total,
        )
        population_footprint = _footprint_on_population_grid(
            footprint,
            population=population_reader,
            footprint_crs=hazard_crs,
            reprojecting=reprojecting,
        )
        _require_raster_overlap(population_footprint, population_reader)
        expected_total = _population_in_footprint(
            population_reader, population_footprint
        )
        aligned_total = _aligned_population_total(
            population_reader,
            crs=hazard_crs,
            transform=transform,
            shape=shape,
            block_shape=block_shape,
        )

    tolerance = resolve_conservation_tolerance(
        conservation_tolerance,
        reprojecting=reprojecting,
    )
    difference = abs(aligned_total - expected_total)
    scale = max(1.0, abs(expected_total))
    relative_difference = difference / scale
    allowed_difference = tolerance * scale
    if difference > allowed_difference:
        raise ValueError(
            "Population was not conserved while aligning to the hazard grid: "
            f"expected {expected_total:.12g}, got {aligned_total:.12g}, "
            f"difference {difference:.12g} exceeds {allowed_difference:.12g} "
            f"(relative difference {relative_difference:.3g}, allowed "
            f"{tolerance:.3g})."
            + (
                " Warping between coordinate systems has a small difference of "
                "its own, which grows on coarse population grids. Raise "
                "conservation_tolerance if this difference is acceptable for "
                "your analysis."
                if reprojecting
                else ""
            )
        )

    attrs: Mapping[str, object] = MappingProxyType(
        {
            "population_assignment": "raster_sum_resampling",
            "population_name": population_column,
            "population_source_total": source_total,
            "population_covered_total": expected_total,
            "population_aligned_total": aligned_total,
            "population_conservation_tolerance": tolerance,
            "population_conservation_relative_difference": relative_difference,
            "population_reprojected": reprojecting,
            "population_source": population_metadata,
        }
    )
    return RasterAssignment(
        shape=shape,
        crs=hazard_crs,
        transform=transform,
        bounds=bounds,
        hazard_band=selected_band,
        attrs=attrs,
        _hazard=hazard_source,
        _population=population_source,
        _block_shape=block_shape,
    )


def resolve_conservation_tolerance(
    conservation_tolerance: float | None,
    *,
    reprojecting: bool,
) -> float:
    """Return the allowed relative difference for a conservation check.

    Same-grid alignment is arithmetic, so it holds to a very small difference.
    Warping between coordinate systems does not, because GDAL allocates source
    cells to destination cells approximately. Measured cross-system differences
    are near ``1e-4`` on ordinary grids, so the default allows ``1e-3``, which
    still catches the far larger differences a wrong footprint produces.

    Args:
        conservation_tolerance: An explicit allowance, or None to choose the
            default for the situation.
        reprojecting: True when population is warped from another coordinate
            system.

    Returns:
        float: The relative difference allowed.

    Examples:
        >>> resolve_conservation_tolerance(None, reprojecting=False)
        1e-06
        >>> resolve_conservation_tolerance(None, reprojecting=True)
        0.001
        >>> resolve_conservation_tolerance(0.5, reprojecting=True)
        0.5
    """
    if conservation_tolerance is not None:
        return float(conservation_tolerance)
    if reprojecting:
        return CROSS_CRS_CONSERVATION_TOLERANCE
    return SAME_CRS_CONSERVATION_TOLERANCE


def normalize_raster_source(source: RasterSource, *, parameter: str) -> RasterSource:
    """Return a checked local raster source without opening it permanently."""
    if isinstance(source, DatasetReader):
        if source.closed:
            raise ValueError(f"{parameter} raster reader is closed.")
        return source
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(
                f"{parameter} raster path does not exist or is not a file: {path}."
            )
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"{parameter} raster must be a GeoTIFF (.tif or .tiff).")
        return path
    raise TypeError(
        f"{parameter} must be a GeoTIFF path or open Rasterio DatasetReader."
    )


@contextmanager
def open_raster(
    source: RasterSource,
    *,
    parameter: str,
) -> Iterator[DatasetReader]:
    """Open path inputs while leaving caller-owned readers open."""
    if isinstance(source, DatasetReader):
        if source.closed:
            raise ValueError(f"{parameter} raster reader is closed.")
        yield source
        return
    try:
        with rasterio.open(source) as reader:
            yield reader
    except RasterioIOError as error:
        raise ValueError(
            f"{parameter} raster could not be opened: {source}."
        ) from error


def validate_population_raster(population: DatasetReader) -> float:
    """Validate a one-band population-count raster and return its total."""
    validate_raster_grid(population, name="population")
    if population.count != 1:
        raise ValueError(
            "population raster must contain exactly one count band; "
            f"found {population.count}."
        )
    nodata = population.nodata
    if nodata is not None and not (np.isfinite(nodata) or np.isnan(nodata)):
        raise ValueError("population raster nodata must be finite or NaN.")
    _validate_count_metadata(population)

    total = 0.0
    valid_cells = 0
    for _, window in population.block_windows(1):
        values = population.read(1, window=window, masked=True)
        valid = np.asarray(values.compressed(), dtype=float)
        if not np.isfinite(valid).all():
            raise ValueError("Population raster values must be finite outside nodata.")
        if (valid < 0).any():
            raise ValueError("Population raster values must be non-negative.")
        total += float(valid.sum(dtype=np.float64))
        valid_cells += valid.size
    if valid_cells == 0:
        raise ValueError(
            "population raster must contain at least one valid count cell."
        )
    return total


def validate_raster_grid(dataset: DatasetReader, *, name: str) -> None:
    """Validate raster georeferencing and dimensions.

    Args:
        dataset: An open raster.
        name: The input name used in error messages, such as ``"hazard"``.

    Returns:
        None.

    Raises:
        ValueError: If the raster has no coordinate system, no georeferencing
            transform, or unusable dimensions or bounds.

    Examples:
        >>> import rasterio
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     validate_raster_grid(raster, name="population")
    """
    if dataset.crs is None:
        raise ValueError(f"{name} raster must define a CRS.")
    if dataset.width <= 0 or dataset.height <= 0:  # pragma: no cover
        raise ValueError(f"{name} raster dimensions must be positive.")
    coefficients = np.asarray(tuple(dataset.transform)[:6], dtype=float)
    if (  # pragma: no cover
        not np.isfinite(coefficients).all() or dataset.transform.determinant == 0
    ):
        raise ValueError(f"{name} raster must define a finite, invertible transform.")
    if dataset.transform.is_identity:
        raise ValueError(f"{name} raster must define a georeferencing transform.")
    bounds = np.asarray(dataset.bounds, dtype=float)
    if not np.isfinite(bounds).all():  # pragma: no cover
        raise ValueError(f"{name} raster bounds must be finite.")
    if dataset.bounds.left >= dataset.bounds.right:  # pragma: no cover
        raise ValueError(f"{name} raster bounds must have positive width.")
    if dataset.bounds.bottom >= dataset.bounds.top:  # pragma: no cover
        raise ValueError(f"{name} raster bounds must have positive height.")


def _validate_count_metadata(population: DatasetReader) -> None:
    """Reject metadata that explicitly describes density rather than counts."""
    tags = {
        str(key).lower(): str(value).lower()
        for key, value in population.tags(1).items()
    }
    semantics = tags.get("population_semantics")
    if semantics is not None and semantics != "count":
        raise ValueError(
            "population raster metadata must declare population_semantics=count."
        )
    metadata = " ".join(
        value
        for key, value in tags.items()
        if key in {"description", "long_name", "unit", "units"}
    )
    unit = population.units[0] if population.units else None
    if unit:
        metadata = f"{metadata} {unit.lower()}"
    density_markers = (
        "density",
        "people/km",
        "persons/km",
        "population/km",
        "people per square",
        "persons per square",
        "population per square",
        "km-2",
        "km^-2",
    )
    if any(marker in metadata for marker in density_markers):
        raise ValueError(
            "population raster metadata describes density; provide population counts."
        )


def _select_hazard_band(
    hazard: DatasetReader,
    requested: int | None,
) -> int:
    """Select one hazard band without guessing for multiband rasters."""
    if hazard.count < 1:  # pragma: no cover
        raise ValueError("hazard raster must contain at least one band.")
    if requested is None:
        if hazard.count != 1:
            raise ValueError(
                "hazard raster has multiple bands; select one with hazard_band."
            )
        return 1
    if requested < 1 or requested > hazard.count:
        raise ValueError(
            f"hazard_band must be between 1 and {hazard.count}; got {requested}."
        )
    return requested


def raster_footprint(dataset: DatasetReader) -> Polygon:
    """Return the exact outer grid footprint, including rotated grids.

    Args:
        dataset: An open raster.

    Returns:
        shapely.geometry.Polygon: The outer edge of the grid, in the raster's
        own coordinate system.

    Examples:
        >>> import rasterio
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     raster_footprint(raster).geom_type
        'Polygon'
    """
    transform = dataset.transform
    return Polygon(
        [
            transform @ (0, 0),
            transform @ (dataset.width, 0),
            transform @ (dataset.width, dataset.height),
            transform @ (0, dataset.height),
        ]
    )


def _footprint_on_population_grid(
    footprint: Polygon,
    *,
    population: DatasetReader,
    footprint_crs: CRS,
    reprojecting: bool,
) -> BaseGeometry:
    """Express a hazard raster footprint on the population raster's grid.

    Args:
        footprint: The hazard grid footprint, in the hazard coordinate system.
        population: The open population raster.
        footprint_crs: The hazard coordinate system.
        reprojecting: True when the coordinate systems differ and the caller
            allowed automatic reprojection.

    Returns:
        shapely.geometry.base.BaseGeometry: The hazard footprint in the
        population raster's coordinate system.

    Examples:
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _footprint_on_population_grid(
        ...         box(0, 0, 1, 1),
        ...         population=raster,
        ...         footprint_crs=raster.crs,
        ...         reprojecting=False,
        ...     ).geom_type
        'Polygon'
    """
    if not reprojecting:
        return footprint
    return transform_geometries(
        [footprint],
        source_crs=footprint_crs,
        target_crs=population.crs,
        tolerance=boundary_tolerance(population),
    )[0]


def _require_raster_overlap(
    footprint: BaseGeometry,
    population: DatasetReader,
) -> None:
    """Require the hazard and population raster footprints to share area.

    Args:
        footprint: Hazard footprint in the population coordinate system.
        population: The open population raster.

    Returns:
        None.

    Raises:
        population_exposure.PartialCoverageError: If the raster footprints do
            not share any area.

    Examples:
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _require_raster_overlap(box(0, 0, 1, 1), raster)
    """
    overlap = footprint.intersection(raster_footprint(population))
    if not overlap.is_empty and overlap.area > 0:
        return
    hazard_bounds = ", ".join(f"{value:g}" for value in footprint.bounds)
    population_bounds = ", ".join(f"{value:g}" for value in population.bounds)
    raise PartialCoverageError(
        "hazard raster lies entirely outside the population raster. "
        f"Hazard bounds on the population grid: ({hazard_bounds}); "
        f"population bounds: ({population_bounds}). Use a population raster "
        "that overlaps the hazard raster."
    )


def _population_in_footprint(
    population: DatasetReader,
    footprint: BaseGeometry,
) -> float:
    """Calculate the exact population covered by the hazard footprint.

    Args:
        population: The open population raster.
        footprint: The hazard grid footprint, in the population coordinate
            system.

    Returns:
        float: The population covered by the footprint.

    Examples:
        >>> import rasterio
        >>> from shapely.geometry import box
        >>> with rasterio.open("population.tif") as raster:  # doctest: +SKIP
        ...     _population_in_footprint(raster, box(0, 0, 1, 1))
        10.0
    """
    feature = gpd.GeoDataFrame(geometry=[footprint], crs=population.crs)
    summary = exact_extract(population, feature, "sum", output="pandas")
    value = float(summary.loc[0, "sum"])
    if not np.isfinite(value):  # pragma: no cover
        return 0.0
    return value


@contextmanager
def _aligned_population_reader(
    population: DatasetReader,
    *,
    crs: CRS,
    transform: Affine,
    shape: tuple[int, int],
) -> Iterator[WarpedVRT]:
    """Yield a population raster virtually aligned with the hazard grid."""
    with WarpedVRT(
        population,
        crs=crs,
        transform=transform,
        height=shape[0],
        width=shape[1],
        src_nodata=population.nodata,
        nodata=np.nan,
        resampling=Resampling.sum,
        dtype="float64",
        warp_mem_limit=64,
    ) as aligned:
        yield aligned


def _aligned_population_total(
    population: DatasetReader,
    *,
    crs: CRS,
    transform: Affine,
    shape: tuple[int, int],
    block_shape: tuple[int, int],
) -> float:
    """Sum aligned population in bounded windows."""
    total = 0.0
    with _aligned_population_reader(
        population,
        crs=crs,
        transform=transform,
        shape=shape,
    ) as aligned:
        for window in _windows(shape, block_shape):
            values = aligned.read(1, window=window, masked=True)
            valid = np.asarray(values.compressed(), dtype=float)
            if not np.isfinite(valid).all():  # pragma: no cover
                raise ValueError(
                    "Aligned population contains non-finite values outside nodata."
                )
            if (valid < 0).any():  # pragma: no cover
                raise ValueError("Aligned population contains negative values.")
            total += float(valid.sum(dtype=np.float64))
    return total


def _safe_block_shape(
    block_shape: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Cap unusually large source blocks for bounded result reads."""
    rows = min(block_shape[0], _DEFAULT_BLOCK_SHAPE[0], shape[0])
    columns = min(block_shape[1], _DEFAULT_BLOCK_SHAPE[1], shape[1])
    return rows, columns


def _windows(
    shape: tuple[int, int],
    block_shape: tuple[int, int],
) -> Iterator[Window]:
    """Yield top-to-bottom, left-to-right windows."""
    height, width = shape
    block_height, block_width = block_shape
    for row in range(0, height, block_height):
        for column in range(0, width, block_width):
            yield Window.from_slices(
                (row, min(row + block_height, height)),
                (column, min(column + block_width, width)),
            )
