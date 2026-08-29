"""Tests for raster population assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds

from population_exposure import RasterAssignment, assign_population
from population_exposure import raster as raster_module
from population_exposure.raster import normalize_raster_source

if TYPE_CHECKING:
    from pathlib import Path


def write_raster(
    path: Path,
    values: np.ndarray,
    *,
    crs: str | None = "EPSG:3857",
    transform=None,
    nodata: float | int | None = -9999,
    tags: dict[str, str] | None = None,
    unit: str | None = None,
) -> Path:
    """Write a tiny one- or multiband GeoTIFF."""
    data = np.asarray(values)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform or from_origin(0, data.shape[1], 1, 1),
        nodata=nodata,
    ) as dataset:
        dataset.write(data)
        if tags:
            dataset.update_tags(1, **tags)
        if unit:
            dataset.set_band_unit(1, unit)
    return path


def base_rasters(tmp_path: Path) -> tuple[Path, Path]:
    """Write matching tiny hazard and population rasters."""
    hazard = write_raster(
        tmp_path / "hazard.tif",
        np.array([[10, 20], [30, 40]], dtype=np.int16),
        nodata=-32768,
    )
    population = write_raster(
        tmp_path / "population.tif",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    return hazard, population


def test_same_grid_returns_lazy_aligned_result(tmp_path: Path) -> None:
    hazard, population = base_rasters(tmp_path)

    result = assign_population(hazard, population)

    assert isinstance(result, RasterAssignment)
    assert result.shape == (2, 2)
    assert result.crs.to_string() == "EPSG:3857"
    assert result.hazard_band == 1
    hazard_values, population_values = result.read()
    np.testing.assert_array_equal(hazard_values, [[10, 20], [30, 40]])
    np.testing.assert_array_equal(population_values, [[1.0, 2.0], [3.0, 4.0]])
    assert {
        key: value for key, value in result.attrs.items() if key != "population_source"
    } == {
        "population_assignment": "raster_sum_resampling",
        "population_name": "population",
        "population_source_total": 10.0,
        "population_covered_total": 10.0,
        "population_aligned_total": 10.0,
        "population_conservation_tolerance": 1e-6,
    }
    assert result.attrs["population_source"]["source_id"] == "custom"
    assert len(result.attrs["population_source"]["local_sha256"]) == 64


def test_different_crs_and_resolution_preserve_counts(tmp_path: Path) -> None:
    population = write_raster(
        tmp_path / "population.tif",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    )
    bounds = transform_bounds("EPSG:4326", "EPSG:3857", 0, 0, 2, 2)
    hazard = write_raster(
        tmp_path / "hazard.tif",
        np.arange(16, dtype=np.int16).reshape(4, 4),
        transform=from_bounds(*bounds, 4, 4),
        nodata=-32768,
    )

    result = assign_population(hazard, population)
    hazard_values, aligned = result.read()

    assert hazard_values.shape == aligned.shape == (4, 4)
    assert float(aligned.sum()) == pytest.approx(10.0, rel=1e-12)
    assert result.attrs["population_covered_total"] == pytest.approx(10.0)
    assert result.attrs["population_aligned_total"] == pytest.approx(10.0)


def test_partial_extent_and_nodata_are_accounted_for(tmp_path: Path) -> None:
    population = write_raster(
        tmp_path / "population.tif",
        np.array([[1.0, -9999.0], [3.0, 4.0]]),
    )
    hazard = write_raster(
        tmp_path / "hazard.tif",
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16),
        transform=from_origin(0, 2, 1, 1),
        nodata=-32768,
    )

    result = assign_population(hazard, population)
    _, aligned = result.read()

    assert float(aligned.sum()) == 8.0
    assert aligned.mask[:, 2].all()
    assert result.attrs["population_source_total"] == 8.0
    assert result.attrs["population_covered_total"] == 8.0


def test_fractional_partial_extent_is_conserved(tmp_path: Path) -> None:
    population = write_raster(
        tmp_path / "population.tif",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    hazard = write_raster(
        tmp_path / "hazard.tif",
        np.ones((2, 3), dtype=np.uint8),
        transform=from_origin(0, 2, 0.5, 1),
        nodata=255,
    )

    result = assign_population(hazard, population)
    _, aligned = result.read()

    assert float(aligned.sum()) == pytest.approx(7.0)
    assert result.attrs["population_covered_total"] == pytest.approx(7.0)


def test_multiband_hazard_requires_explicit_band(tmp_path: Path) -> None:
    population = write_raster(
        tmp_path / "population.tif",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    hazard = write_raster(
        tmp_path / "hazard.tif",
        np.array(
            [
                [[1, 2], [3, 4]],
                [[10, 20], [30, 40]],
            ],
            dtype=np.int16,
        ),
        nodata=-32768,
    )

    with pytest.raises(ValueError, match="multiple bands"):
        assign_population(hazard, population)

    result = assign_population(hazard, population, hazard_band=2)

    assert result.hazard_band == 2
    np.testing.assert_array_equal(result.read()[0], [[10, 20], [30, 40]])


def test_invalid_hazard_band_fails(tmp_path: Path) -> None:
    hazard, population = base_rasters(tmp_path)

    with pytest.raises(ValueError, match="between 1 and 1"):
        assign_population(hazard, population, hazard_band=2)


def test_missing_crs_and_transform_fail(tmp_path: Path) -> None:
    population = write_raster(
        tmp_path / "population.tif",
        np.ones((2, 2)),
    )
    no_crs = write_raster(
        tmp_path / "no-crs.tif",
        np.ones((2, 2)),
        crs=None,
    )
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        identity = write_raster(
            tmp_path / "identity.tif",
            np.ones((2, 2)),
            transform=rasterio.Affine.identity(),
        )

    with pytest.raises(ValueError, match="must define a CRS"):
        assign_population(no_crs, population)
    with pytest.raises(ValueError, match="georeferencing transform"):
        assign_population(identity, population)


def test_population_missing_crs_and_invalid_nodata_fail(tmp_path: Path) -> None:
    hazard, _ = base_rasters(tmp_path)
    missing_crs = write_raster(
        tmp_path / "missing-crs.tif",
        np.ones((2, 2)),
        crs=None,
    )
    invalid_nodata = write_raster(
        tmp_path / "invalid-nodata.tif",
        np.ones((2, 2)),
        nodata=np.inf,
    )

    with pytest.raises(ValueError, match="population raster must define a CRS"):
        assign_population(hazard, missing_crs)
    with pytest.raises(ValueError, match="nodata must be finite or NaN"):
        assign_population(hazard, invalid_nodata)


def test_population_must_have_one_band(tmp_path: Path) -> None:
    hazard, _ = base_rasters(tmp_path)
    population = write_raster(
        tmp_path / "population-multiband.tif",
        np.ones((2, 2, 2)),
    )

    with pytest.raises(ValueError, match="exactly one count band"):
        assign_population(hazard, population)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([[1.0, np.inf], [3.0, 4.0]]), "must be finite"),
        (np.array([[1.0, -0.1], [3.0, 4.0]]), "must be non-negative"),
        (np.full((2, 2), -9999.0), "at least one valid count cell"),
    ],
)
def test_invalid_population_cells_fail(
    tmp_path: Path,
    values: np.ndarray,
    message: str,
) -> None:
    hazard, _ = base_rasters(tmp_path)
    population = write_raster(tmp_path / "bad-population.tif", values)

    with pytest.raises(ValueError, match=message):
        assign_population(hazard, population)


def test_density_metadata_is_rejected(tmp_path: Path) -> None:
    hazard, _ = base_rasters(tmp_path)
    population = write_raster(
        tmp_path / "density.tif",
        np.ones((2, 2)),
        tags={"population_semantics": "density"},
    )

    with pytest.raises(ValueError, match="population_semantics=count"):
        assign_population(hazard, population)


def test_density_band_unit_is_rejected(tmp_path: Path) -> None:
    hazard, _ = base_rasters(tmp_path)
    population = write_raster(
        tmp_path / "density-unit.tif",
        np.ones((2, 2)),
        unit="people/km2",
    )

    with pytest.raises(ValueError, match="describes density"):
        assign_population(hazard, population)


def test_reader_inputs_remain_open(tmp_path: Path) -> None:
    hazard_path, population_path = base_rasters(tmp_path)

    with (
        rasterio.open(hazard_path) as hazard,
        rasterio.open(population_path) as population,
    ):
        result = assign_population(hazard, population)
        assert not hazard.closed
        assert not population.closed
        result.read()
        assert not hazard.closed
        assert not population.closed


def test_path_inputs_remain_unchanged(tmp_path: Path) -> None:
    hazard, population = base_rasters(tmp_path)
    hazard_before = hazard.read_bytes()
    population_before = population.read_bytes()

    result = assign_population(hazard, population)
    result.read()

    assert hazard.read_bytes() == hazard_before
    assert population.read_bytes() == population_before


def test_iter_blocks_bounds_memory_without_a_cell_dataframe(tmp_path: Path) -> None:
    shape = (300, 300)
    hazard = write_raster(
        tmp_path / "hazard.tif",
        np.ones(shape, dtype=np.uint8),
        nodata=255,
    )
    population = write_raster(
        tmp_path / "population.tif",
        np.ones(shape, dtype=np.float32),
    )

    result = assign_population(hazard, population)
    blocks = list(result.iter_blocks())

    assert len(blocks) > 1
    assert sum(values.size for _, values, _ in blocks) == 90_000
    assert max(values.size for _, values, _ in blocks) <= 256 * 256
    assert max(values.size for _, _, values in blocks) <= 256 * 256


def test_missing_and_unsupported_paths_fail(tmp_path: Path) -> None:
    _, population = base_rasters(tmp_path)
    unsupported = tmp_path / "hazard.csv"
    unsupported.write_text("not,a,map\n")

    with pytest.raises(ValueError, match="does not exist"):
        assign_population(tmp_path / "missing.tif", population)
    with pytest.raises(ValueError, match="supported extension"):
        assign_population(unsupported, population)


def test_custom_population_name_is_recorded(tmp_path: Path) -> None:
    hazard, population = base_rasters(tmp_path)

    result = assign_population(hazard, population, population_column="people")

    assert result.attrs["population_name"] == "people"


def test_raster_rejects_tabular_population_and_vector_option(
    tmp_path: Path,
) -> None:
    hazard, population = base_rasters(tmp_path)
    tabular_population = pd.DataFrame({"population": [1]})

    with pytest.raises(TypeError, match="population raster"):
        assign_population(hazard, tabular_population)
    with pytest.raises(ValueError, match="allow_overlaps applies only to vector"):
        assign_population(hazard, population, allow_overlaps=True)

    with rasterio.open(hazard) as hazard_reader:
        with pytest.raises(TypeError, match="population raster"):
            assign_population(
                hazard_reader,
                tabular_population,
            )
        with pytest.raises(ValueError, match="allow_overlaps applies only to vector"):
            assign_population(hazard_reader, population, allow_overlaps=True)


def test_bad_population_sources_fail(tmp_path: Path) -> None:
    hazard, _ = base_rasters(tmp_path)
    unsupported = tmp_path / "population.csv"
    unsupported.write_text("population\n1\n")
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_text("not a raster")

    with pytest.raises(ValueError, match="does not exist"):
        assign_population(hazard, tmp_path / "missing.tif")
    with pytest.raises(ValueError, match="must be a GeoTIFF"):
        assign_population(hazard, unsupported)
    with pytest.raises(TypeError, match="GeoTIFF path or open Rasterio"):
        assign_population(hazard, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="could not be opened"):
        assign_population(corrupt, hazard)


def test_low_level_raster_source_validation_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        normalize_raster_source(tmp_path / "missing.tif", parameter="population")
    with pytest.raises(TypeError, match="GeoTIFF path or open Rasterio"):
        normalize_raster_source(object(), parameter="population")  # type: ignore[arg-type]


def test_closed_readers_fail_without_being_reopened(tmp_path: Path) -> None:
    hazard_path, population_path = base_rasters(tmp_path)
    hazard = rasterio.open(hazard_path)
    hazard.close()

    with pytest.raises(ValueError, match="hazard raster reader is closed"):
        assign_population(hazard, population_path)

    with (
        rasterio.open(hazard_path) as open_hazard,
        rasterio.open(population_path) as population,
    ):
        result = assign_population(open_hazard, population)
        population.close()
        with pytest.raises(ValueError, match="population raster reader is closed"):
            result.read()


def test_conservation_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hazard, population = base_rasters(tmp_path)
    monkeypatch.setattr(
        raster_module,
        "_aligned_population_total",
        lambda *args, **kwargs: 0.0,
    )

    with pytest.raises(ValueError, match="Population was not conserved"):
        assign_population(hazard, population)
