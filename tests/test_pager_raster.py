"""Golden and opt-in live tests for the USGS PAGER Ridgecrest raster."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from population_exposure import assign_population
from population_exposure.populations import register
from population_exposure.populations._http import download_file, sha256_file
from tests.pager_raster import (
    PAGER_EVENT_ID,
    PAGER_EXPOSURES_SHA256,
    PAGER_EXPOSURES_URL,
    PAGER_GRID_SHA256,
    PAGER_GRID_URL,
    PAGER_PRODUCT_TIMESTAMP,
    PAGER_XML_SHA256,
    PAGER_XML_URL,
    PUBLISHED_EXPOSURE,
    _required_integer,
    aggregate_pager_exposure,
    parse_pager_grid,
    write_pager_geotiff,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pager_ridgecrest"
FIXTURE_PATH = FIXTURE_DIR / "mmi.tif"
METADATA_PATH = FIXTURE_DIR / "metadata.json"


def test_pager_fixture_records_the_authoritative_grid() -> None:
    """Keep the derived hazard fixture tied to the exact PAGER product."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["event_id"] == PAGER_EVENT_ID
    assert metadata["product_timestamp"] == PAGER_PRODUCT_TIMESTAMP
    assert metadata["source_artifacts"]["grid"]["url"] == PAGER_GRID_URL
    assert metadata["source_artifacts"]["grid"]["sha256"] == PAGER_GRID_SHA256
    assert metadata["source_artifacts"]["exposures"]["url"] == PAGER_EXPOSURES_URL
    assert metadata["source_artifacts"]["exposures"]["sha256"] == PAGER_EXPOSURES_SHA256
    assert metadata["source_artifacts"]["pager"]["url"] == PAGER_XML_URL
    assert metadata["source_artifacts"]["pager"]["sha256"] == PAGER_XML_SHA256
    assert metadata["published_exposure"] == list(PUBLISHED_EXPOSURE)
    assert sha256_file(FIXTURE_PATH) == metadata["fixture"]["sha256"]

    with rasterio.open(FIXTURE_PATH) as dataset:
        assert (dataset.width, dataset.height) == (671, 547)
        assert dataset.crs.to_string() == "EPSG:4326"
        assert tuple(dataset.transform)[:6] == pytest.approx(
            tuple(metadata["grid"]["transform"])
        )
        assert dataset.transform.a == pytest.approx(1 / 60)
        assert dataset.transform.e == pytest.approx(-1 / 60)
        assert dataset.tags()["row_order"] == "north_to_south"
        assert dataset.read(1, window=((0, 1), (0, 1)))[0, 0] == pytest.approx(2)


def test_pager_bands_use_half_open_boundaries_and_sum_resampling(
    tmp_path: Path,
) -> None:
    """Pin PAGER intervals and coverage-weighted population alignment."""
    hazard_path = tmp_path / "hazard.tif"
    population_path = tmp_path / "population.tif"
    hazard = np.array(
        [[0.5, 1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5, 9.5]],
        dtype=np.float32,
    )
    population = np.arange(1, 41, dtype=np.float32).reshape(4, 10)
    _write_raster(hazard_path, hazard, from_origin(0, 2, 1, 1))
    _write_raster(
        population_path,
        population,
        from_origin(0, 2, 0.5, 0.5),
        population=True,
    )

    result = assign_population(hazard_path, population_path)

    assert aggregate_pager_exposure(result) == pytest.approx(
        (26, 34, 42, 50, 58, 106, 114, 122, 130, 138)
    )
    assert result.attrs["population_aligned_total"] == pytest.approx(820)


def test_pager_xml_parser_preserves_center_coordinates_and_row_order(
    tmp_path: Path,
) -> None:
    """Pin XML center-to-corner conversion and north-to-south ordering."""
    xml_path = tmp_path / "grid.xml"
    xml_path.write_text(
        """<?xml version="1.0"?>
<shakemap_grid xmlns="http://earthquake.usgs.gov/eqcenter/shakemap"
 event_id="ci38457511" process_timestamp="2019-07-06T15:13:28">
 <grid_specification nlon="2" nlat="2"
  nominal_lon_spacing="0.0167" nominal_lat_spacing="0.0167"/>
 <grid_field index="1" name="LON"/>
 <grid_field index="2" name="LAT"/>
 <grid_field index="3" name="MMI"/>
 <grid_data>
  -0.0167 0.0167 5.5
  0.0000 0.0167 6.5
  -0.0167 0.0000 7.5
  0.0000 0.0000 8.5
 </grid_data>
</shakemap_grid>
""",
        encoding="utf-8",
    )

    parsed = parse_pager_grid(xml_path)

    np.testing.assert_array_equal(parsed.mmi, [[5.5, 6.5], [7.5, 8.5]])
    assert tuple(parsed.transform)[:6] == pytest.approx(
        (1 / 60, 0, -0.025, 0, -1 / 60, 0.025)
    )


def test_required_pager_attribute_errors_are_explicit() -> None:
    """Report missing and malformed XML integer attributes distinctly."""
    with pytest.raises(ValueError, match="is required"):
        _required_integer({}, "nlon")
    with pytest.raises(ValueError, match="must be an integer"):
        _required_integer({"nlon": "two"}, "nlon")


@pytest.mark.live
@pytest.mark.timeout(7200)
def test_pager_raster_reproduces_published_exposure(
    tmp_path: Path,
) -> None:
    """Compare the PAGER bands using an explicitly supplied LandScan 2017 file."""
    licensed_path = os.environ.get("PAGER_LANDSCAN_2017_PATH")
    if not licensed_path:
        pytest.skip(
            "Set PAGER_LANDSCAN_2017_PATH to a licensed LandScan Global 2017 "
            "GeoTIFF for the opt-in reproduction."
        )
    population_path = Path(licensed_path).expanduser()
    if not population_path.is_file():
        pytest.fail(f"PAGER_LANDSCAN_2017_PATH is not a file: {population_path}")

    grid_xml = tmp_path / "grid.xml"
    download_file(
        PAGER_GRID_URL,
        grid_xml,
        headers=None,
        max_bytes=50_000_000,
        exact_bytes=None,
        publisher_checksum=None,
    )
    assert sha256_file(grid_xml) == PAGER_GRID_SHA256
    grid = parse_pager_grid(grid_xml)
    assert grid.event_id == PAGER_EVENT_ID
    assert grid.process_timestamp == "2019-07-06T15:13:28"
    hazard_path = write_pager_geotiff(grid_xml, tmp_path / "pager-mmi.tif")
    registered = register(
        "landscan-global:2017",
        population_path,
        cache_dir=tmp_path / "cache",
    )

    result = assign_population(hazard_path, registered)
    observed = aggregate_pager_exposure(result)
    expected = np.asarray(PUBLISHED_EXPOSURE, dtype=np.float64)
    differences = np.abs(np.asarray(observed) - expected)
    tolerance = 0.5
    assert np.max(differences) <= tolerance, (
        "PAGER reproduction exceeded the measured integer-count rounding "
        f"tolerance of {tolerance} people: observed={observed}, "
        f"expected={tuple(expected)}, differences={tuple(differences)}."
    )
    assert observed[5] + observed[6] + observed[7] == pytest.approx(
        46836, abs=tolerance
    )


def _write_raster(
    path: Path,
    values: np.ndarray,
    transform,
    *,
    population: bool = False,
) -> None:
    """Write one small EPSG:4326 count or hazard raster for a test.

    Args:
        path: Destination GeoTIFF path.
        values: Two-dimensional values to write.
        transform: Pixel-corner transform for the values.
        population: Whether to mark the raster as population counts.

    Returns:
        None.

    Examples:
        >>> _write_raster(Path("example.tif"), np.ones((1, 1)), from_origin(0, 1, 1, 1))
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)
        if population:
            dataset.update_tags(population_semantics="count")
