"""Tests for the shared coordinate-system helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from population_exposure._crs import (
    _crs_name,
    as_crs,
    boundary_tolerance,
    require_matching_crs,
    transform_geometries,
)
from population_exposure._errors import CrsMismatchError

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_coordinate_system_is_named() -> None:
    with pytest.raises(ValueError, match="hazard must define a CRS"):
        as_crs(None, parameter="hazard")


def test_unreadable_coordinate_system_is_named() -> None:
    with pytest.raises(ValueError, match="population CRS could not be read"):
        as_crs("not-a-coordinate-system", parameter="population")


def test_geopandas_and_rasterio_coordinate_systems_compare_equal() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

    assert (
        require_matching_crs(
            frame.crs,
            "EPSG:4326",
            hazard_kind="vector",
            allow_reprojection=False,
        )
        is False
    )


def test_coordinate_systems_without_a_code_use_their_name() -> None:
    custom = CRS.from_proj4("+proj=moll +lon_0=90 +datum=WGS84 +units=m +no_defs")

    assert _crs_name(custom) == custom.name
    assert ":" not in _crs_name(custom)


def test_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tolerance must be finite and positive"):
        transform_geometries(
            [box(0, 0, 1, 1)],
            source_crs="EPSG:4326",
            target_crs="EPSG:3857",
            tolerance=0.0,
        )


def test_impossible_accuracy_fails_loudly() -> None:
    """Asking for more accuracy than points can deliver is an error, not a guess."""
    with pytest.raises(CrsMismatchError, match="accurately enough"):
        transform_geometries(
            [box(-40, -40, 40, 40)],
            source_crs="EPSG:4326",
            target_crs="ESRI:54009",
            tolerance=1e-9,
        )


def test_boundary_allowance_is_a_tenth_of_the_shorter_cell_side(
    tmp_path: Path,
) -> None:
    """The promised accuracy has to match what the code actually allows."""
    path = tmp_path / "population.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float64",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 5.0, 2.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.ones((2, 2)), 1)

    with rasterio.open(path) as dataset:
        assert boundary_tolerance(dataset) == pytest.approx(0.2)
