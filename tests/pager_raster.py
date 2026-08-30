"""Helpers for reproducing the USGS PAGER Ridgecrest raster exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from defusedxml import ElementTree
from rasterio.transform import Affine, from_origin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from population_exposure import RasterAssignment

PAGER_GRID_URL = (
    "https://earthquake.usgs.gov/product/losspager/ci38457511/us/1562429005114/grid.xml"
)
PAGER_EXPOSURES_URL = (
    "https://earthquake.usgs.gov/product/losspager/ci38457511/us/"
    "1562429005114/json/exposures.json"
)
PAGER_XML_URL = (
    "https://earthquake.usgs.gov/product/losspager/ci38457511/us/"
    "1562429005114/pager.xml"
)
PAGER_EVENT_ID = "ci38457511"
PAGER_PRODUCT_TIMESTAMP = "1562429005114"
PAGER_GRID_SHA256 = "bc412154745d00442bb9ad0ffbfdfd3e168acbf28646183d1a88d96f711fe063"
PAGER_EXPOSURES_SHA256 = (
    "a4987b1990db0ca536c2c97175340cbe486d9362d37b7d5fe7cd3353d590f227"
)
PAGER_XML_SHA256 = "58e92b0d31383dd38203d457e6773667864ee95be969a92bd96f4cb5b4796639"
_PAGER_NAMESPACE = "http://earthquake.usgs.gov/eqcenter/shakemap"
_ARC_MINUTE = 1.0 / 60.0
_COORDINATE_ROUNDING_TOLERANCE = 0.000051
MMI_BAND_BOUNDARIES = tuple(0.5 + index for index in range(11))
PUBLISHED_EXPOSURE = (0, 10642442, 17902502, 21545813, 601968, 1721, 44915, 200, 0, 0)


@dataclass(frozen=True, slots=True)
class PagerGrid:
    """Parsed PAGER intensity grid and its raster georeferencing.

    Args:
        mmi: Two-dimensional MMI values in north-to-south row order.
        transform: Pixel-corner transform for the MMI array.
        event_id: USGS event identifier from the XML.
        process_timestamp: ShakeMap process timestamp from the XML.

    Returns:
        A validated immutable description of one PAGER grid.

    Examples:
        >>> grid = parse_pager_grid(Path("grid.xml"))
        >>> grid.mmi.shape
        (547, 671)
    """

    mmi: np.ndarray
    transform: Affine
    event_id: str
    process_timestamp: str


def parse_pager_grid(path: Path) -> PagerGrid:
    """Parse and validate the PAGER XML grid as a north-up raster.

    Args:
        path: Local copy of the official PAGER ``grid.xml`` artifact.

    Returns:
        Parsed MMI values and a transform whose pixels are centered on the
        listed LON/LAT coordinates.

    Raises:
        ValueError: If the event, grid shape, field, spacing, or row order is
            not the expected PAGER representation.

    Examples:
        >>> grid = parse_pager_grid(Path("grid.xml"))
        >>> grid.transform.a == 1 / 60
        True
    """
    root = ElementTree.parse(path).getroot()
    specification = root.find(f"{{{_PAGER_NAMESPACE}}}grid_specification")
    if specification is None:
        raise ValueError("PAGER grid is missing grid_specification.")
    attributes = specification.attrib
    nlon = _required_integer(attributes, "nlon")
    nlat = _required_integer(attributes, "nlat")
    if nlon <= 0 or nlat <= 0:
        raise ValueError("PAGER grid dimensions must be positive.")

    event_id = root.attrib.get("event_id")
    if event_id != PAGER_EVENT_ID:
        raise ValueError(
            f"PAGER grid event_id must be {PAGER_EVENT_ID!r}; got {event_id!r}."
        )
    process_timestamp = root.attrib.get("process_timestamp")
    if process_timestamp is None:
        raise ValueError("PAGER grid is missing process_timestamp.")

    fields = root.findall(f"{{{_PAGER_NAMESPACE}}}grid_field")
    mmi_field = next(
        (field for field in fields if field.attrib.get("name") == "MMI"), None
    )
    if mmi_field is None:
        raise ValueError("PAGER grid is missing its MMI field.")
    mmi_index = _required_integer(mmi_field.attrib, "index") - 1

    grid_data = root.find(f"{{{_PAGER_NAMESPACE}}}grid_data")
    if grid_data is None or grid_data.text is None:
        raise ValueError("PAGER grid is missing grid_data.")
    rows = [
        [float(value) for value in line.split()]
        for line in grid_data.text.splitlines()
        if line.strip()
    ]
    if len(rows) != nlon * nlat:
        raise ValueError(
            f"PAGER grid contains {len(rows)} rows; expected {nlon * nlat}."
        )
    values = np.asarray(rows, dtype=np.float64)
    if mmi_index < 0 or mmi_index >= values.shape[1]:
        raise ValueError("PAGER MMI field index is outside the data rows.")
    values = values.reshape(nlat, nlon, values.shape[1])
    longitudes = values[0, :, 0]
    latitudes = values[:, 0, 1]
    _validate_coordinates(longitudes, latitudes, attributes)
    expected_longitudes = np.round(longitudes / _ARC_MINUTE) * _ARC_MINUTE
    expected_latitudes = np.round(latitudes / _ARC_MINUTE) * _ARC_MINUTE
    if not np.allclose(
        values[:, :, 0],
        expected_longitudes[None, :],
        rtol=0,
        atol=_COORDINATE_ROUNDING_TOLERANCE,
    ) or not np.allclose(
        values[:, :, 1],
        expected_latitudes[:, None],
        rtol=0,
        atol=_COORDINATE_ROUNDING_TOLERANCE,
    ):
        raise ValueError("PAGER coordinate rows or columns are not regular.")
    mmi = values[:, :, mmi_index].astype(np.float32)
    if not np.isfinite(mmi).all():
        raise ValueError("PAGER MMI values must be finite.")
    left = float(np.round(longitudes[0] / _ARC_MINUTE) * _ARC_MINUTE - _ARC_MINUTE / 2)
    top = float(np.round(latitudes[0] / _ARC_MINUTE) * _ARC_MINUTE + _ARC_MINUTE / 2)
    return PagerGrid(
        mmi=mmi,
        transform=from_origin(left, top, _ARC_MINUTE, _ARC_MINUTE),
        event_id=event_id,
        process_timestamp=process_timestamp,
    )


def write_pager_geotiff(xml_path: Path, output_path: Path) -> Path:
    """Convert one PAGER XML grid to a deterministic MMI GeoTIFF.

    Args:
        xml_path: Local copy of the official PAGER ``grid.xml`` artifact.
        output_path: Destination GeoTIFF path, which may not be a directory.

    Returns:
        The destination path after writing the one-band float32 raster.

    Raises:
        ValueError: If the source XML is invalid or the output is a directory.

    Examples:
        >>> write_pager_geotiff(Path("grid.xml"), Path("mmi.tif"))
        PosixPath('mmi.tif')
    """
    if output_path.is_dir():
        raise ValueError(f"PAGER GeoTIFF output is a directory: {output_path}.")
    grid = parse_pager_grid(xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=grid.mmi.shape[1],
        height=grid.mmi.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=grid.transform,
        nodata=-9999.0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=3,
    ) as dataset:
        dataset.write(grid.mmi, 1)
        dataset.update_tags(
            event_id=grid.event_id,
            process_timestamp=grid.process_timestamp,
            source_grid=PAGER_GRID_URL,
            grid_field="MMI",
            coordinate_semantics="listed coordinates are pixel centers",
            spacing_degrees=f"{_ARC_MINUTE:.17g}",
            row_order="north_to_south",
            nodata_semantics="no nodata in source",
        )
    return output_path


def aggregate_pager_exposure(
    assignment: RasterAssignment,
) -> tuple[float, ...]:
    """Sum assigned population into PAGER's half-open MMI bands.

    Args:
        assignment: Raster assignment pairing the PAGER MMI grid with a
            population-count raster.

    Returns:
        Ten totals indexed by MMI 1 through 10.

    Raises:
        ValueError: If a valid hazard cell has an MMI outside the PAGER range.

    Examples:
        >>> # ``assignment`` is produced by ``assign_population``.
        >>> len(aggregate_pager_exposure(assignment))
        10
    """
    totals = np.zeros(10, dtype=np.float64)
    for _, hazard, population in assignment.iter_blocks():
        hazard_mask = np.ma.getmaskarray(hazard)
        population_mask = np.ma.getmaskarray(population)
        valid = ~(hazard_mask | population_mask)
        hazard_values = np.asarray(hazard.data, dtype=np.float64)
        population_values = np.asarray(population.data, dtype=np.float64)
        if np.any(valid & ((hazard_values < 0.5) | (hazard_values >= 10.5))):
            raise ValueError("PAGER MMI values must be in [0.5, 10.5).")
        band = np.digitize(hazard_values, MMI_BAND_BOUNDARIES, right=False)
        for mmi in range(1, 11):
            selected = valid & (band == mmi)
            totals[mmi - 1] += population_values[selected].sum(dtype=np.float64)
    return tuple(float(value) for value in totals)


def _required_integer(attributes: Mapping[str, str], name: str) -> int:
    """Read one required positive integer XML attribute.

    Args:
        attributes: XML attribute mapping.
        name: Required attribute name.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the attribute is absent or not an integer.

    Examples:
        >>> _required_integer({"nlon": "2"}, "nlon")
        2
    """
    value = attributes.get(name)
    if value is None:
        raise ValueError(f"PAGER grid attribute {name!r} is required.")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"PAGER grid attribute {name!r} must be an integer."
        ) from error


def _validate_coordinates(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    attributes: Mapping[str, str],
) -> None:
    """Validate rounded one-arc-minute coordinates and north-to-south rows.

    Args:
        longitudes: First-row longitude centers.
        latitudes: First-column latitude centers.
        attributes: PAGER grid specification attributes.

    Returns:
        None.

    Raises:
        ValueError: If spacing metadata, coordinate regularity, or orientation
            does not match the expected PAGER grid.

    Examples:
        >>> _validate_coordinates(np.array([0.0, 1 / 60]), np.array([1 / 60, 0.0]), {})
    """
    for name in ("nominal_lon_spacing", "nominal_lat_spacing"):
        nominal = float(attributes.get(name, "nan"))
        if not np.isclose(
            nominal,
            _ARC_MINUTE,
            rtol=0,
            atol=_COORDINATE_ROUNDING_TOLERANCE,
        ):
            raise ValueError(f"PAGER {name} is not one arc minute: {nominal!r}.")
    expected_longitudes = np.round(longitudes / _ARC_MINUTE) * _ARC_MINUTE
    expected_latitudes = np.round(latitudes / _ARC_MINUTE) * _ARC_MINUTE
    if not np.allclose(
        longitudes,
        expected_longitudes,
        rtol=0,
        atol=_COORDINATE_ROUNDING_TOLERANCE,
    ) or not np.allclose(
        latitudes,
        expected_latitudes,
        rtol=0,
        atol=_COORDINATE_ROUNDING_TOLERANCE,
    ):
        raise ValueError("PAGER coordinates are not rounded one-arc-minute centers.")
    if not np.all(np.diff(longitudes) > 0):
        raise ValueError("PAGER longitudes must increase from west to east.")
    if not np.all(np.diff(latitudes) < 0):
        raise ValueError("PAGER latitudes must decrease from north to south.")
