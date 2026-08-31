"""Tests for strict population coverage of vector hazards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box

import population_exposure as pe

if TYPE_CHECKING:
    from pathlib import Path

FRACTION = "population_coverage_fraction"
COMPLETE = "population_coverage_complete"


def write_population(
    path: Path,
    *,
    values: np.ndarray | None = None,
    crs: str = "EPSG:3857",
    transform=None,
) -> Path:
    """Write a tiny population raster covering the unit square at the origin.

    Args:
        path: Where to write the raster.
        values: The cell values, or None for the standard two-by-two grid.
        crs: The coordinate system to record.
        transform: An explicit affine transform, or None for one-unit cells.

    Returns:
        pathlib.Path: The raster path.

    Examples:
        >>> write_population(tmp_path / "population.tif")  # doctest: +SKIP
    """
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
        transform=transform or from_origin(0, data.shape[0], 1, 1),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def test_feature_inside_the_raster_succeeds_by_default(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(0.25, 0.25, 1.75, 1.75)], crs="EPSG:3857")

    result = pe.assign_population(hazard, population)

    assert result["population"].item() == pytest.approx(5.625)
    assert FRACTION not in result.columns
    assert COMPLETE not in result.columns


@pytest.mark.parametrize(
    ("geometry", "share"),
    [
        (box(1, 0, 3, 2), "50.0%"),
        (box(0, 0, 20, 2), "10.0%"),
    ],
    ids=["half-outside", "ninety-percent-outside"],
)
def test_features_reaching_outside_raise_by_default(
    tmp_path: Path,
    geometry,
    share: str,
) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(
        {"name": ["coast"]},
        geometry=[geometry],
        crs="EPSG:3857",
        index=pd.Index(["zone-a"], name="zone"),
    )

    with pytest.raises(pe.PartialCoverageError) as caught:
        pe.assign_population(hazard, population)

    message = str(caught.value)
    assert "'zone-a'" in message
    assert share in message
    assert "leave out everyone" in message
    assert "Clip or revise the geometry" in message
    assert "allow_partial_coverage=True" in message


def test_feature_entirely_outside_still_raises(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(10, 10, 11, 11)], crs="EPSG:3857")

    for allow in (False, True):
        with pytest.raises(pe.PartialCoverageError, match="entirely outside"):
            pe.assign_population(hazard, population, allow_partial_coverage=allow)


def test_opting_in_returns_the_partial_total_and_its_coverage(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(1, 0, 3, 2)], crs="EPSG:3857")

    result = pe.assign_population(hazard, population, allow_partial_coverage=True)

    assert result["population"].item() == pytest.approx(6.0)
    assert result[FRACTION].item() == pytest.approx(0.5)
    assert bool(result[COMPLETE].item()) is False
    assert result.attrs["population_assignment"]["partial_coverage_allowed"] is True


def test_mixed_batch_reports_coverage_for_every_row(tmp_path: Path) -> None:
    population = write_population(
        tmp_path / "population.tif",
        values=np.ones((4, 4)),
    )
    hazard = gpd.GeoDataFrame(
        {"name": ["inside", "half", "mostly-out"]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 6, 4), box(3.8, 0, 13.8, 1)],
        crs="EPSG:3857",
        index=pd.Index([7, 8, 9]),
    )

    strict = pytest.raises(pe.PartialCoverageError)
    with strict as caught:
        pe.assign_population(hazard, population)
    assert "8" in str(caught.value)
    assert "9" in str(caught.value)

    result = pe.assign_population(hazard, population, allow_partial_coverage=True)

    assert result.index.equals(hazard.index)
    assert result["population"].tolist() == pytest.approx([1.0, 4.0, 0.2])
    assert result[FRACTION].tolist() == pytest.approx([1.0, 0.5, 0.02])
    assert result[COMPLETE].tolist() == [True, False, False]


def test_no_data_inside_the_footprint_stays_covered(tmp_path: Path) -> None:
    """Ocean and empty land are no-data, and must not look like missing area."""
    population = write_population(
        tmp_path / "population.tif",
        values=np.array([[-9999.0, 2.0], [3.0, 4.0]]),
    )
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

    result = pe.assign_population(hazard, population)

    assert result["population"].item() == pytest.approx(9.0)


def test_longitudes_past_the_raster_edge_are_named_in_the_error(
    tmp_path: Path,
) -> None:
    population = write_population(
        tmp_path / "population.tif",
        transform=from_bounds(-180, -10, 180, 10, 2, 2),
        crs="EPSG:4326",
    )
    hazard = gpd.GeoDataFrame(geometry=[box(170, -5, 190, 5)], crs="EPSG:4326")

    with pytest.raises(pe.PartialCoverageError) as caught:
        pe.assign_population(hazard, population)

    message = str(caught.value)
    assert "-180 to 180 range" in message
    assert "antimeridian" in message

    opted_in = pe.assign_population(hazard, population, allow_partial_coverage=True)

    assert opted_in[FRACTION].item() == pytest.approx(0.5)
    assert bool(opted_in[COMPLETE].item()) is False


def test_coverage_is_measured_after_opted_in_reprojection(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif", crs="EPSG:4326")
    inside = transform_bounds("EPSG:4326", "EPSG:3857", 0.25, 0.25, 1.75, 1.75)
    outside = transform_bounds("EPSG:4326", "EPSG:3857", 1, 0, 3, 2)
    hazard = gpd.GeoDataFrame(geometry=[box(*inside)], crs="EPSG:3857")
    reaching = gpd.GeoDataFrame(geometry=[box(*outside)], crs="EPSG:3857")

    result = pe.assign_population(hazard, population, allow_reprojection=True)

    assert result["population"].item() == pytest.approx(5.625, rel=1e-6)

    with pytest.raises(pe.PartialCoverageError, match="reaches outside"):
        pe.assign_population(reaching, population, allow_reprojection=True)


def test_existing_coverage_columns_are_rejected(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(
        {FRACTION: [0.0]},
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="rename it before"):
        pe.assign_population(hazard, population, allow_partial_coverage=True)

    assert pe.assign_population(hazard, population)["population"].item() == 10.0


def test_dispatch_forwards_partial_coverage_for_vector_paths(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif")
    vector_path = tmp_path / "hazard.geojson"
    gpd.GeoDataFrame(geometry=[box(1, 0, 3, 2)], crs="EPSG:3857").to_file(
        vector_path, driver="GeoJSON"
    )

    with pytest.raises(pe.PartialCoverageError):
        pe.assign_population(vector_path, population)

    result = pe.assign_population(
        vector_path,
        population,
        allow_partial_coverage=True,
    )

    assert result["population"].item() == pytest.approx(6.0)
    assert result[FRACTION].item() == pytest.approx(0.5)


def test_partial_coverage_does_not_apply_to_other_hazard_types(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard_raster = tmp_path / "hazard.tif"
    with rasterio.open(
        hazard_raster,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=from_origin(0, 2, 1, 1),
        nodata=-32768,
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype=np.int16), 1)
    table = pd.DataFrame({"cell": ["A"]})
    table_population = pd.DataFrame({"cell": ["A"], "population": [1.0]})

    with pytest.raises(ValueError, match="only to vector hazards"):
        pe.assign_population(hazard_raster, population, allow_partial_coverage=True)

    with pytest.raises(ValueError, match="only to vector hazards"):
        pe.assign_population(
            table,
            table_population,
            cell_columns="cell",
            allow_partial_coverage=True,
        )


def test_partial_coverage_flag_must_be_a_boolean(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

    with pytest.raises(TypeError, match="allow_partial_coverage must be a boolean"):
        pe.assign_population(hazard, population, allow_partial_coverage=1)


def test_many_partial_features_are_summarized(tmp_path: Path) -> None:
    population = write_population(tmp_path / "population.tif")
    geometries = [box(index * 0.2, 0, index * 0.2 + 4, 2) for index in range(7)]
    hazard = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    with pytest.raises(pe.PartialCoverageError) as caught:
        pe.assign_population(hazard, population, allow_overlaps=True)

    assert "and 2 more" in str(caught.value)
