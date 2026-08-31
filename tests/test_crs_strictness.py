"""Tests for strict coordinate-system matching and opt-in reprojection."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import MultiPolygon, Polygon, box

import population_exposure as pe
from population_exposure import populations
from population_exposure.raster import CROSS_CRS_CONSERVATION_TOLERANCE
from tests.test_populations import use_tiny_source

if TYPE_CHECKING:
    from pathlib import Path

MOLLWEIDE = "ESRI:54009"
MOLLWEIDE_RADIUS = CRS.from_user_input(MOLLWEIDE).ellipsoid.semi_major_metre

# Accuracy promised for opted-in reprojection, declared before measuring it.
# Adding points along a boundary cannot be exact, so this allows a little room
# while staying far below the error a corners-only transform produces.
REPROJECTION_TOLERANCE = 1e-3
# The error a corners-only transform produces for the same fixture.
VERTEX_ONLY_ERROR = 2e-2


def mollweide_box_area(degrees: float) -> float:
    """Return the exact Mollweide area of a square longitude/latitude box.

    Mollweide is an equal-area projection on a sphere, so a box running from
    ``-degrees`` to ``degrees`` in both directions covers
    ``radius ** 2 * longitude_span * (sin(top) - sin(bottom))``.

    Args:
        degrees: Half the width and height of the box, in degrees.

    Returns:
        float: The projected area in square meters.

    Examples:
        >>> round(mollweide_box_area(20) / 1e12, 3)
        19427.043
    """
    radians = math.radians(degrees)
    return MOLLWEIDE_RADIUS**2 * (2 * radians) * (2 * math.sin(radians))


def write_mollweide_population(
    path: Path,
    *,
    degrees: float,
    size: int,
    margin: float = 0.25,
) -> tuple[Path, float]:
    """Write a uniform Mollweide population raster around a lon/lat box.

    Every cell holds one person, so the population inside any polygon is that
    polygon's projected area divided by the area of one cell.

    Args:
        path: Where to write the raster.
        degrees: Half the width and height of the covered box, in degrees.
        size: The number of rows and columns.
        margin: Extra space around the box, as a share of its size.

    Returns:
        tuple[pathlib.Path, float]: The raster path and the area of one cell in
        square meters.

    Examples:
        >>> write_mollweide_population(  # doctest: +SKIP
        ...     tmp_path / "population.tif", degrees=20, size=240
        ... )
    """
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", MOLLWEIDE, -degrees, -degrees, degrees, degrees
    )
    padding = margin * max(right - left, top - bottom)
    left, bottom = left - padding, bottom - padding
    right, top = right + padding, top + padding
    width = (right - left) / size
    height = (top - bottom) / size
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float64",
        crs=MOLLWEIDE,
        transform=from_origin(left, top, width, height),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.ones((size, size), dtype="float64"), 1)
    return path, width * height


def write_population(
    path: Path,
    *,
    crs: str = "EPSG:3857",
    transform=None,
) -> Path:
    """Write the standard tiny two-by-two population raster.

    Args:
        path: Where to write the raster.
        crs: The coordinate system to record.
        transform: An explicit affine transform, or None for one-unit cells
            anchored at the origin.

    Returns:
        pathlib.Path: The raster path.

    Examples:
        >>> write_population(tmp_path / "population.tif")  # doctest: +SKIP
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform or from_origin(0, 2, 1, 1),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]]), 1)
    return path


def write_hazard_raster(
    path: Path,
    *,
    bounds: tuple[float, float, float, float],
    crs: str,
    size: int,
) -> Path:
    """Write a tiny one-band hazard raster covering explicit bounds.

    Args:
        path: Where to write the raster.
        bounds: The left, bottom, right, and top edges.
        crs: The coordinate system to record.
        size: The number of rows and columns.

    Returns:
        pathlib.Path: The raster path.

    Examples:
        >>> write_hazard_raster(  # doctest: +SKIP
        ...     tmp_path / "hazard.tif",
        ...     bounds=(0, 0, 2, 2),
        ...     crs="EPSG:4326",
        ...     size=2,
        ... )
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="int16",
        crs=crs,
        transform=from_bounds(*bounds, size, size),
        nodata=-32768,
    ) as dataset:
        dataset.write(np.ones((size, size), dtype=np.int16), 1)
    return path


def test_vector_crs_mismatch_raises_and_explains_both_choices(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif", crs="EPSG:4326")
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

    with pytest.raises(pe.CrsMismatchError) as caught:
        pe.assign_population(hazard, population)

    message = str(caught.value)
    assert "hazard uses EPSG:3857" in message
    assert "population uses EPSG:4326" in message
    assert "required by default" in message
    assert 'hazard.to_crs("EPSG:4326")' in message
    assert "allow_reprojection=True" in message


def test_raster_crs_mismatch_raises_and_explains_both_choices(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif", crs="EPSG:4326")
    hazard = write_hazard_raster(
        tmp_path / "hazard.tif",
        bounds=transform_bounds("EPSG:4326", "EPSG:3857", 0, 0, 2, 2),
        crs="EPSG:3857",
        size=2,
    )

    with pytest.raises(pe.CrsMismatchError) as caught:
        pe.assign_population(hazard, population)

    message = str(caught.value)
    assert "hazard uses EPSG:3857" in message
    assert "population uses EPSG:4326" in message
    assert "required by default" in message
    assert "rasterio.warp.reproject or gdalwarp" in message
    assert "allow_reprojection=True" in message


def test_cross_crs_outside_raster_requires_opt_in_then_raises_coverage_error(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif", crs="EPSG:4326")
    hazard = write_hazard_raster(
        tmp_path / "hazard.tif",
        bounds=transform_bounds("EPSG:4326", "EPSG:3857", 10, 10, 12, 12),
        crs="EPSG:3857",
        size=2,
    )

    with pytest.raises(pe.CrsMismatchError):
        pe.assign_population(hazard, population)
    with pytest.raises(pe.PartialCoverageError, match="entirely outside"):
        pe.assign_population(hazard, population, allow_reprojection=True)


def test_matching_coordinate_systems_need_no_flag(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")
    hazard_raster = write_hazard_raster(
        tmp_path / "hazard.tif",
        bounds=(0, 0, 2, 2),
        crs="EPSG:3857",
        size=2,
    )

    vector_result = pe.assign_population(hazard, population)
    raster_result = pe.assign_population(hazard_raster, population)

    assert vector_result["population"].item() == pytest.approx(10.0)
    assert vector_result.attrs["population_assignment"]["reprojected"] is False
    assert raster_result.attrs["population_reprojected"] is False
    assert raster_result.attrs["population_conservation_tolerance"] == 1e-6


def test_manual_transformation_is_an_equal_alternative(tmp_path: Path) -> None:
    """The guidance in the error message has to actually work."""
    population, cell_area = write_mollweide_population(
        tmp_path / "population.tif", degrees=20, size=240
    )
    hazard = gpd.GeoDataFrame(geometry=[box(-20, -20, 20, 20)], crs="EPSG:4326")

    manual = hazard.copy()
    manual["geometry"] = manual.geometry.segmentize(max_segment_length=0.1)
    manual = manual.to_crs(MOLLWEIDE)

    result = pe.assign_population(manual, population)
    expected = mollweide_box_area(20) / cell_area

    assert result["population"].item() == pytest.approx(
        expected, rel=REPROJECTION_TOLERANCE
    )


def test_opted_in_vector_reprojection_keeps_curved_boundaries(
    tmp_path: Path,
) -> None:
    """A lon/lat box becomes a curved shape in Mollweide, and must stay curved.

    Mollweide is equal-area, so the population inside the box is known in
    advance: its projected area divided by the area of one cell. Transforming
    only the four corners cuts the curve off and undercounts badly.
    """
    population, cell_area = write_mollweide_population(
        tmp_path / "population.tif", degrees=20, size=240
    )
    hazard = gpd.GeoDataFrame(geometry=[box(-20, -20, 20, 20)], crs="EPSG:4326")
    expected = mollweide_box_area(20) / cell_area

    result = pe.assign_population(hazard, population, allow_reprojection=True)

    assert result["population"].item() == pytest.approx(
        expected, rel=REPROJECTION_TOLERANCE
    )
    assert result.crs == hazard.crs
    assert result.attrs["population_assignment"]["reprojected"] is True

    corners_only = pe.assign_population(hazard.to_crs(MOLLWEIDE), population)
    shortfall = (expected - corners_only["population"].item()) / expected
    assert shortfall > VERTEX_ONLY_ERROR


def test_opted_in_reprojection_handles_multipolygon_and_holes(
    tmp_path: Path,
) -> None:
    population, cell_area = write_mollweide_population(
        tmp_path / "population.tif", degrees=30, size=300
    )
    ring = Polygon(
        box(-20, -20, 20, 20).exterior.coords,
        [tuple(box(-10, -10, 10, 10).exterior.coords)],
    )
    island = box(-28, 22, -22, 28)
    hazard = gpd.GeoDataFrame(
        geometry=[MultiPolygon([ring, island])],
        crs="EPSG:4326",
    )
    expected = (
        mollweide_box_area(20)
        - mollweide_box_area(10)
        + _mollweide_area(-28, 22, -22, 28)
    ) / cell_area

    result = pe.assign_population(hazard, population, allow_reprojection=True)

    assert result["population"].item() == pytest.approx(
        expected, rel=REPROJECTION_TOLERANCE
    )


def _mollweide_area(left: float, bottom: float, right: float, top: float) -> float:
    """Return the exact Mollweide area of any longitude/latitude box.

    Args:
        left: The western edge in degrees.
        bottom: The southern edge in degrees.
        right: The eastern edge in degrees.
        top: The northern edge in degrees.

    Returns:
        float: The projected area in square meters.

    Examples:
        >>> round(_mollweide_area(-1, -1, 1, 1) / 1e9, 3)
        49318.451
    """
    span = math.radians(right - left)
    height = math.sin(math.radians(top)) - math.sin(math.radians(bottom))
    return MOLLWEIDE_RADIUS**2 * span * height


def test_reprojection_rejects_geometry_that_wraps_the_antimeridian(
    tmp_path: Path,
) -> None:
    population = write_population(
        tmp_path / "population.tif",
        crs="EPSG:4326",
        transform=from_bounds(-180, -10, 180, 10, 2, 2),
    )
    crossing = transform_bounds("EPSG:4326", "EPSG:3857", 170, -5, 179.9, 5)
    hazard = gpd.GeoDataFrame(
        geometry=[box(crossing[0], crossing[1], crossing[2] * 1.02, crossing[3])],
        crs="EPSG:3857",
    )

    with pytest.raises(pe.CrsMismatchError, match="antimeridian"):
        pe.assign_population(hazard, population, allow_reprojection=True)


def test_reprojection_rejects_geometry_outside_the_projection(
    tmp_path: Path,
) -> None:
    """Mollweide only reaches about 18,000 km east and west of its center."""
    population = write_population(
        tmp_path / "population.tif",
        crs="EPSG:4326",
        transform=from_bounds(-180, -10, 180, 10, 2, 2),
    )
    hazard = gpd.GeoDataFrame(
        geometry=[box(2.0e7, -1.0e6, 3.0e7, 1.0e6)],
        crs=MOLLWEIDE,
    )

    with pytest.raises(pe.CrsMismatchError, match="outside the area"):
        pe.assign_population(hazard, population, allow_reprojection=True)


def test_opted_in_raster_reprojection_uses_the_transformed_footprint(
    tmp_path: Path,
) -> None:
    """The hazard footprint curves in Mollweide, and the check must see that."""
    population, cell_area = write_mollweide_population(
        tmp_path / "population.tif", degrees=20, size=400
    )
    hazard = write_hazard_raster(
        tmp_path / "hazard.tif",
        bounds=(-20, -20, 20, 20),
        crs="EPSG:4326",
        size=40,
    )
    expected = mollweide_box_area(20) / cell_area

    result = pe.assign_population(hazard, population, allow_reprojection=True)

    covered = result.attrs["population_covered_total"]
    relative = result.attrs["population_conservation_relative_difference"]
    assert covered == pytest.approx(expected, rel=REPROJECTION_TOLERANCE)
    assert result.attrs["population_reprojected"] is True
    assert result.attrs["population_conservation_tolerance"] == (
        CROSS_CRS_CONSERVATION_TOLERANCE
    )
    assert 0 < relative <= CROSS_CRS_CONSERVATION_TOLERANCE
    assert result.attrs["population_aligned_total"] == pytest.approx(
        covered, rel=CROSS_CRS_CONSERVATION_TOLERANCE
    )


def test_corners_only_footprint_would_fail_the_conservation_check(
    tmp_path: Path,
) -> None:
    """Prove the fixture above catches the corners-only defect it replaced."""
    population, cell_area = write_mollweide_population(
        tmp_path / "population.tif", degrees=20, size=400
    )
    footprint = box(-20, -20, 20, 20)
    corners_only = gpd.GeoDataFrame(geometry=[footprint], crs="EPSG:4326").to_crs(
        MOLLWEIDE
    )
    expected = mollweide_box_area(20) / cell_area

    with rasterio.open(population) as reader:
        from exactextract import exact_extract

        covered = float(
            exact_extract(reader, corners_only, "sum", output="pandas").loc[0, "sum"]
        )

    shortfall = (expected - covered) / expected
    assert shortfall > CROSS_CRS_CONSERVATION_TOLERANCE * 10


def test_conservation_failure_names_the_relative_difference(
    tmp_path: Path,
) -> None:
    population, _ = write_mollweide_population(
        tmp_path / "population.tif", degrees=20, size=400
    )
    hazard = write_hazard_raster(
        tmp_path / "hazard.tif",
        bounds=(-20, -20, 20, 20),
        crs="EPSG:4326",
        size=40,
    )

    with pytest.raises(ValueError, match="relative difference") as caught:
        pe.assign_population(
            hazard,
            population,
            allow_reprojection=True,
            conservation_tolerance=0.0,
        )

    assert "Raise conservation_tolerance" in str(caught.value)


def test_dispatch_forwards_the_flag_for_vector_and_raster_paths(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif", crs="EPSG:4326")
    vector_path = tmp_path / "hazard.geojson"
    bounds = transform_bounds("EPSG:4326", "EPSG:3857", 0.2, 0.2, 1.8, 1.8)
    gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:3857").to_file(
        vector_path, driver="GeoJSON"
    )
    raster_path = write_hazard_raster(
        tmp_path / "hazard.tif",
        bounds=transform_bounds("EPSG:4326", "EPSG:3857", 0, 0, 2, 2),
        crs="EPSG:3857",
        size=2,
    )

    for hazard in (vector_path, raster_path):
        with pytest.raises(pe.CrsMismatchError):
            pe.assign_population(hazard, population)

    assert (
        pe.assign_population(vector_path, population, allow_reprojection=True)[
            "population"
        ].item()
        > 0
    )
    assert (
        pe.assign_population(
            raster_path,
            population,
            allow_reprojection=True,
        ).attrs["population_reprojected"]
        is True
    )


def test_dispatch_forwards_the_flag_for_registered_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    monkeypatch.setenv("POPULATION_EXPOSURE_CACHE_DIR", str(tmp_path / "cache"))
    write_population(tmp_path / "worldpop-global-1km-2020.tif", crs="EPSG:4326")
    populations.register(
        "worldpop-global-1km:2020",
        tmp_path / "worldpop-global-1km-2020.tif",
    )
    bounds = transform_bounds("EPSG:4326", "EPSG:3857", 0.2, 0.2, 1.8, 1.8)
    hazard = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:3857")

    with pytest.raises(pe.CrsMismatchError):
        pe.assign_population(hazard, "worldpop-global-1km:2020")

    result = pe.assign_population(
        hazard,
        "worldpop-global-1km:2020",
        allow_reprojection=True,
    )

    assert result["population"].item() > 0
    assert result.attrs["population_source"]["selection"] == (
        "worldpop-global-1km:2020"
    )


def test_reprojection_does_not_apply_to_tables() -> None:
    import pandas as pd

    hazard = pd.DataFrame({"cell": ["A"]})
    population = pd.DataFrame({"cell": ["A"], "population": [1.0]})

    with pytest.raises(ValueError, match="only to vector and raster hazards"):
        pe.assign_population(
            hazard,
            population,
            cell_columns="cell",
            allow_reprojection=True,
        )


def test_flags_must_be_booleans(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

    with pytest.raises(TypeError, match="allow_reprojection must be a boolean"):
        pe.assign_population(hazard, population, allow_reprojection="yes")
