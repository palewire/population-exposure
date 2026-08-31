"""Tests for vector population assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, Polygon, box

import population_exposure as pe
from population_exposure import assign_population
from population_exposure.vector import _geodesic_area, _ordered_totals

if TYPE_CHECKING:
    from pathlib import Path


def write_population(
    path: Path,
    *,
    values: np.ndarray | None = None,
    crs: str | None = "EPSG:3857",
    transform=None,
    nodata: float | None = -9999,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a tiny population raster."""
    data = (
        np.array([[1.0, 2.0], [3.0, 4.0]])
        if values is None
        else np.asarray(values, dtype=float)
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform or from_origin(0, 2, 1, 1),
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)
        if tags:
            dataset.update_tags(1, **tags)
    return path


def polygon_frame() -> gpd.GeoDataFrame:
    """Return two non-overlapping polygons with a custom index."""
    return gpd.GeoDataFrame(
        {
            "risk": ["high", "low"],
            "notes": [pd.NA, "edge"],
        },
        geometry=[box(0, 0, 1.5, 2), box(1.5, 0, 2, 2)],
        crs="EPSG:3857",
        index=pd.Index([8, 3], name="feature"),
    )


def test_vector_assignment_preserves_features_and_fractional_coverage(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = polygon_frame()
    original = hazard.copy(deep=True)

    result = assign_population(hazard, population)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.index.equals(hazard.index)
    assert result.columns.tolist() == ["risk", "notes", "geometry", "population"]
    assert result["population"].tolist() == [7.0, 3.0]
    assert result.crs == hazard.crs
    assert_geodataframe_equal(result.drop(columns="population"), original)
    assert_geodataframe_equal(hazard, original)
    assert result.attrs["population_assignment"] == {
        "method": "exactextract_sum",
        "population_crs": "EPSG:3857",
        "population_band": 1,
        "overlaps_allowed": False,
        "reprojected": False,
        "partial_coverage_allowed": False,
    }


def test_exactextract_fractional_coverage_golden(tmp_path: Path) -> None:
    """Check ExactExtract's published fractional coverage through the public API.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    # Oracle: https://github.com/isciences/exactextractr/blob/4a3d69a1e2e7936e5bd806fe22c29bf13ee042a3/tests/testthat/test_coverage_fraction.R#L20-L37
    # and https://github.com/isciences/exactextract/blob/3b4926bd0237a6ab20b7f69d1dfc960625c3b3af/test/test_operation.cpp#L97-L117
    # This is a wrapper conformance golden; ExactExtract's upstream tests
    # validate its algorithm.
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    transform = from_origin(0, 3, 1, 1)
    population = write_population(
        tmp_path / "population.tif",
        values=values,
        transform=transform,
    )
    hazard = gpd.GeoDataFrame(
        geometry=[box(0.5, 0.5, 2.5, 2.5)],
        crs="EPSG:3857",
    )

    corner_total = 0.25 * (1 + 3 + 7 + 9)
    edge_total = 0.5 * (2 + 4 + 6 + 8)
    center_total = 5.0
    assert corner_total + edge_total + center_total == 20.0

    result = pe.assign_population(hazard, population)

    # This is far below one person and only accommodates floating-point addition.
    assert result["population"].item() == pytest.approx(20.0, rel=0, abs=1e-9)
    assert result.attrs["population_assignment"]["method"] == "exactextract_sum"


def test_opted_in_reprojection_does_not_mutate_the_input(tmp_path: Path) -> None:
    bounds = transform_bounds("EPSG:4326", "EPSG:3857", 0, 0, 2, 2)
    population = write_population(
        tmp_path / "population.tif",
        transform=from_bounds(*bounds, 2, 2),
    )
    hazard = gpd.GeoDataFrame(
        {"risk": ["west", "east"]},
        geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2)],
        crs="EPSG:4326",
    )
    original = hazard.copy(deep=True)

    result = assign_population(hazard, population, allow_reprojection=True)

    assert result["population"].tolist() == pytest.approx([4.0, 6.0])
    assert result.crs == hazard.crs
    assert result.attrs["population_assignment"]["reprojected"] is True
    assert_geodataframe_equal(hazard, original)


@pytest.mark.parametrize(
    ("suffix", "driver"),
    [
        (".geojson", "GeoJSON"),
        (".gpkg", "GPKG"),
        (".shp", "ESRI Shapefile"),
    ],
)
def test_common_vector_paths_are_supported(
    tmp_path: Path,
    suffix: str,
    driver: str,
) -> None:
    vector_path = tmp_path / f"hazard{suffix}"
    polygon_frame().reset_index(drop=True).to_file(vector_path, driver=driver)
    population = write_population(tmp_path / "population.tif")

    result = assign_population(vector_path, population)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result["population"].tolist() == [7.0, 3.0]


def test_population_nodata_is_excluded(tmp_path: Path) -> None:
    population = write_population(
        tmp_path / "population.tif",
        values=np.array([[1.0, -9999.0], [3.0, 4.0]]),
    )
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

    result = assign_population(hazard, population)

    assert result["population"].item() == 8.0


def test_overlapping_polygons_fail_by_default_and_can_be_allowed(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(
        {"name": ["all", "right"]},
        geometry=[box(0, 0, 2, 2), box(1, 0, 2, 2)],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="overlapping polygons"):
        assign_population(hazard, population)

    result = assign_population(hazard, population, allow_overlaps=True)

    assert result["population"].tolist() == [10.0, 6.0]
    assert result.attrs["population_assignment"]["overlaps_allowed"] is True


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (None, "missing geometry"),
        (Polygon(), "empty geometry"),
        (Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]), "invalid geometry"),
        (LineString([(0, 0), (1, 1)]), "only Polygon or MultiPolygon"),
    ],
)
def test_bad_vector_geometry_fails(
    tmp_path: Path,
    geometry,
    message: str,
) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:3857")

    with pytest.raises(ValueError, match=message):
        assign_population(hazard, population)


def test_empty_vector_and_missing_crs_fail(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    missing_crs = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])

    with pytest.raises(ValueError, match="at least one polygon"):
        assign_population(empty, population)
    with pytest.raises(ValueError, match="must define a CRS"):
        assign_population(missing_crs, population)


def test_polygon_outside_population_coverage_fails(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(10, 10, 11, 11)], crs="EPSG:3857")

    with pytest.raises(
        pe.PartialCoverageError, match="entirely outside the population raster"
    ):
        assign_population(hazard, population)


def test_existing_population_column_fails(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = polygon_frame()
    hazard["people"] = 0

    with pytest.raises(ValueError, match="already has a column"):
        assign_population(hazard, population, population_column="people")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([[1.0, np.inf], [3.0, 4.0]]), "must be finite"),
        (np.array([[1.0, -0.1], [3.0, 4.0]]), "must be non-negative"),
    ],
)
def test_invalid_population_values_fail(
    tmp_path: Path,
    values: np.ndarray,
    message: str,
) -> None:
    population = write_population(tmp_path / "population.tif", values=values)

    with pytest.raises(ValueError, match=message):
        assign_population(polygon_frame(), population)


def test_density_metadata_fails(tmp_path: Path) -> None:
    population = write_population(
        tmp_path / "population.tif",
        tags={"units": "people per square kilometer"},
    )

    with pytest.raises(ValueError, match="describes density"):
        assign_population(polygon_frame(), population)


def test_population_reader_remains_open(tmp_path: Path) -> None:
    population_path = write_population(tmp_path / "population.tif")

    with rasterio.open(population_path) as population:
        result = assign_population(polygon_frame(), population)
        assert not population.closed
        assert result["population"].sum() == 10.0
        assert not population.closed


def test_vector_rejects_tabular_population_and_raster_band(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    tabular_population = pd.DataFrame({"population": [1]})

    with pytest.raises(TypeError, match="require a population raster"):
        assign_population(polygon_frame(), tabular_population)
    with pytest.raises(ValueError, match="hazard_band applies only to raster"):
        assign_population(polygon_frame(), population, hazard_band=1)


def test_vector_path_rejects_tabular_population_and_raster_band(
    tmp_path: Path,
) -> None:
    hazard_path = tmp_path / "hazard.geojson"
    polygon_frame().to_file(hazard_path, driver="GeoJSON")
    population = write_population(tmp_path / "population.tif")
    tabular_population = pd.DataFrame({"population": [1]})

    with pytest.raises(TypeError, match="require a population raster"):
        assign_population(hazard_path, tabular_population)
    with pytest.raises(ValueError, match="hazard_band applies only to raster"):
        assign_population(hazard_path, population, hazard_band=1)


def test_unreadable_vector_fails_clearly(tmp_path: Path) -> None:
    hazard_path = tmp_path / "hazard.geojson"
    hazard_path.write_text("not geojson")
    population = write_population(tmp_path / "population.tif")

    with pytest.raises(ValueError, match="hazard vector could not be read"):
        assign_population(hazard_path, population)


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        (pd.DataFrame({"sum": [1.0]}), "unexpected vector result"),
        (
            pd.DataFrame(
                {
                    "__population_exposure_row__": [0, 1],
                    "sum": [1.0, 1.0],
                    "count": [1.0, 1.0],
                }
            ),
            "unexpected vector result",
        ),
        (
            pd.DataFrame(
                {
                    "__population_exposure_row__": [1],
                    "sum": [1.0],
                    "count": [1.0],
                }
            ),
            "did not return every",
        ),
        (
            pd.DataFrame(
                {
                    "__population_exposure_row__": [0],
                    "sum": [1.0],
                    "count": [np.inf],
                }
            ),
            "invalid valid-cell count",
        ),
        (
            pd.DataFrame(
                {
                    "__population_exposure_row__": [0],
                    "sum": [np.inf],
                    "count": [1.0],
                }
            ),
            "non-finite population",
        ),
        (
            pd.DataFrame(
                {
                    "__population_exposure_row__": [0],
                    "sum": [-1.0],
                    "count": [1.0],
                }
            ),
            "negative population",
        ),
    ],
)
def test_unexpected_exactextract_results_fail(
    summary: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises((RuntimeError, ValueError), match=message):
        _ordered_totals(
            summary,
            expected_rows=1,
            spatial_coverage=np.array([1.0]),
        )


def test_no_valid_cells_return_zero_after_spatial_coverage_check() -> None:
    summary = pd.DataFrame(
        {
            "__population_exposure_row__": [0],
            "sum": [np.nan],
            "count": [0.0],
        }
    )

    totals = _ordered_totals(
        summary,
        expected_rows=1,
        spatial_coverage=np.array([0.5]),
    )

    np.testing.assert_array_equal(totals, [0.0])


def test_geodesic_area_rejects_a_half_earth_polygon() -> None:
    with pytest.raises(ValueError, match="half or more"):
        _geodesic_area(box(0, -45, 180, 45))
