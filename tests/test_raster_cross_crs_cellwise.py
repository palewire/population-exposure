"""Independent cellwise validation for cross-CRS raster assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import transform as transform_geometry

import population_exposure as pe

if TYPE_CHECKING:
    from pathlib import Path


_FIXTURE_TOTAL = 100.0
_ORACLE_ABSOLUTE_TOLERANCE = 1e-6
_CELLWISE_ABSOLUTE_TOLERANCE = _FIXTURE_TOTAL * 2e-4
_TOTAL_ABSOLUTE_TOLERANCE = _FIXTURE_TOTAL * 1e-8
_GOLDEN_DESTINATION_COUNTS = np.array(
    [
        [6.401646559479, 10.760132396940, 10.888750310061],
        [16.943197525947, 27.393784907209, 27.612488300363],
    ]
)


def test_cross_crs_cells_match_independent_area_oracle(tmp_path: Path) -> None:
    """Match every warped cell against an explicit area-intersection oracle.

    The source is a 2-by-2 EPSG:4326 grid whose values are counts for whole
    source cells, not densities. The destination is a 2-by-3 EPSG:3035 grid;
    EPSG:3035 is a non-cylindrical Lambert azimuthal equal-area projection.
    Its projected top-edge midpoint sits more than 100 meters from the chord
    between the projected corners, confirming that this transform is curved.
    Longitude/latitude axis order is fixed with ``always_xy=True``. Geographic
    source edges are segmented every 0.005 degree before projection, and
    boundary-only contact contributes zero area. The destination covers every
    source cell completely. Both rasters declare nodata sentinels, but this
    fixture deliberately contains no nodata so it isolates cell allocation
    from missing-data behavior.

    For each destination cell ``d``, the oracle calculates
    ``sum(count_s * area(projected_s intersect d) / area(projected_s))``. It
    uses only PyProj and Shapely and is fixed before ``assign_population`` runs;
    it does not call package raster helpers or any raster resampler. Halving
    the edge segment length changes each golden value by less than 1e-6 person.
    The 0.02-person cell tolerance is 0.02% of the 100-person fixture and more
    than 300 times smaller than the least error caused by the checked spatial
    swap. Full coverage permits the tighter 1e-6-person aggregate tolerance.

    Args:
        tmp_path: Temporary directory where the two tiny rasters are written.

    Returns:
        None.

    Examples:
        Run this case alone with
        ``uv run pytest tests/test_raster_cross_crs_cellwise.py -q``.
    """
    source_counts = np.array([[11.0, 17.0], [29.0, 43.0]])
    source_transform = from_origin(9.5, 52.5, 0.5, 0.5)
    destination_shape = (2, 3)
    destination_transform = from_origin(
        4_280_000.0,
        3_270_000.0,
        80_000.0 / destination_shape[1],
        60_000.0,
    )
    project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    projected_top = np.array(
        [project.transform(longitude, 52.5) for longitude in (9.5, 10.0, 10.5)]
    )
    projected_chord_midpoint = (projected_top[0] + projected_top[2]) / 2
    assert np.linalg.norm(projected_top[1] - projected_chord_midpoint) > 100.0

    projected_source_cells = []
    for source_row, source_column in np.ndindex(source_counts.shape):
        source_west, source_north = source_transform @ (
            source_column,
            source_row,
        )
        source_east, source_south = source_transform @ (
            source_column + 1,
            source_row + 1,
        )
        source_cell = box(
            source_west,
            source_south,
            source_east,
            source_north,
        ).segmentize(0.005)
        projected_source_cells.append(
            (
                source_row,
                source_column,
                transform_geometry(project.transform, source_cell),
            )
        )

    expected_counts = np.zeros(destination_shape)
    source_coverage = np.zeros(source_counts.shape)
    for destination_row, destination_column in np.ndindex(destination_shape):
        destination_west, destination_north = destination_transform @ (
            destination_column,
            destination_row,
        )
        destination_east, destination_south = destination_transform @ (
            destination_column + 1,
            destination_row + 1,
        )
        destination_cell = box(
            destination_west,
            destination_south,
            destination_east,
            destination_north,
        )
        for source_row, source_column, source_cell in projected_source_cells:
            covered_fraction = (
                source_cell.intersection(destination_cell).area / source_cell.area
            )
            expected_counts[destination_row, destination_column] += (
                source_counts[source_row, source_column] * covered_fraction
            )
            source_coverage[source_row, source_column] += covered_fraction

    np.testing.assert_allclose(
        expected_counts,
        _GOLDEN_DESTINATION_COUNTS,
        rtol=0.0,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        source_coverage,
        np.ones(source_counts.shape),
        rtol=0.0,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )
    assert expected_counts.sum() == pytest.approx(
        _FIXTURE_TOTAL,
        rel=0.0,
        abs=_TOTAL_ABSOLUTE_TOLERANCE,
    )

    total_preserving_swap = np.flip(expected_counts)
    assert total_preserving_swap.sum() == pytest.approx(
        expected_counts.sum(),
        rel=0.0,
        abs=_TOTAL_ABSOLUTE_TOLERANCE,
    )
    assert not np.allclose(
        total_preserving_swap,
        expected_counts,
        rtol=0.0,
        atol=_CELLWISE_ABSOLUTE_TOLERANCE,
    )

    population_path = tmp_path / "population.tif"
    with rasterio.open(
        population_path,
        "w",
        driver="GTiff",
        height=source_counts.shape[0],
        width=source_counts.shape[1],
        count=1,
        dtype=source_counts.dtype,
        crs="EPSG:4326",
        transform=source_transform,
        nodata=-9999.0,
    ) as population:
        population.write(source_counts, 1)
        population.update_tags(1, population_semantics="count")

    hazard_path = tmp_path / "hazard.tif"
    with rasterio.open(
        hazard_path,
        "w",
        driver="GTiff",
        height=destination_shape[0],
        width=destination_shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=destination_transform,
        nodata=255,
    ) as hazard:
        hazard.write(
            np.arange(np.prod(destination_shape), dtype=np.uint8).reshape(
                destination_shape
            ),
            1,
        )

    result = pe.assign_population(
        hazard_path,
        population_path,
        allow_reprojection=True,
    )
    _, assigned_counts = result.read()

    assert result.attrs["population_reprojected"] is True
    assert not np.ma.getmaskarray(assigned_counts).any()
    np.testing.assert_allclose(
        assigned_counts,
        expected_counts,
        rtol=0.0,
        atol=_CELLWISE_ABSOLUTE_TOLERANCE,
    )
    assert assigned_counts.sum() == pytest.approx(
        expected_counts.sum(),
        rel=0.0,
        abs=_TOTAL_ABSOLUTE_TOLERANCE,
    )
