"""Offline golden validation for GPWv4 tabular population assignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from population_exposure import assign_population
from population_exposure.populations._http import sha256_file

FIXTURE_DIRECTORY = Path(__file__).parent / "data" / "gpwv4_r11_iceland_tabular"
COORDINATES = ("longitude", "latitude")


def _population_frame(path: Path, decimal_places: int) -> pd.DataFrame:
    """Return independently generated coordinate and population rows from a raster.

    Args:
        path: Fixture 30-arc-second population GeoTIFF.
        decimal_places: Published decimal precision used for coordinate keys.

    Returns:
        DataFrame containing exact raster-center keys and population values.

    Examples:
        >>> _population_frame(Path("population.tif"), 12).columns.tolist()
        ['longitude', 'latitude', 'population']
    """
    with rasterio.open(path) as source:
        values = source.read(1, masked=True)
        valid = ~np.ma.getmaskarray(values).ravel()
        rows, columns = np.indices(source.shape)
        longitudes, latitudes = rasterio.transform.xy(
            source.transform,
            rows.ravel()[valid],
            columns.ravel()[valid],
            offset="center",
        )
    return pd.DataFrame(
        {
            "longitude": np.round(np.asarray(longitudes), decimal_places),
            "latitude": np.round(np.asarray(latitudes), decimal_places),
            "population": np.asarray(values.data, dtype=np.float32)
            .ravel()[valid]
            .astype(float),
        }
    )


@pytest.mark.component
def test_gpwv4_iceland_tabular_cross_resolution_golden() -> None:
    """Match shuffled GPW cells and reproduce each published one-degree total."""
    metadata = json.loads(
        (FIXTURE_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )
    population_path = FIXTURE_DIRECTORY / "population_30_sec.tif"
    coarse_path = FIXTURE_DIRECTORY / "population_1_deg.tif"
    hazard_path = FIXTURE_DIRECTORY / "hazard_cells.csv"

    for filename, expected_hash in metadata["fixture_files"].items():
        assert sha256_file(FIXTURE_DIRECTORY / filename) == expected_hash
    assert metadata["purpose"] == (
        "Exact-coordinate tabular population assignment validation; "
        "not a hazard or exposure result."
    )
    assert metadata["source_grid"]["crs"] == "EPSG:4326"
    assert metadata["source_grid"]["dtype"] == "float32"
    assert metadata["source_grid"]["cells_per_coarse_side"] == 120
    assert "not a hazard or exposure result" in metadata["purpose"]

    with rasterio.open(population_path) as fine:
        assert list(fine.shape) == metadata["fixture_grid"]["fine_shape"]
        assert fine.crs.to_string() == metadata["source_grid"]["crs"]
        assert fine.dtypes == (metadata["source_grid"]["dtype"],)
        assert list(fine.transform)[:6] == pytest.approx(
            metadata["fixture_grid"]["fine_transform"],
            rel=0,
            abs=1e-12,
        )
        assert list(fine.bounds) == pytest.approx(
            metadata["fixture_grid"]["fine_bounds"],
            rel=0,
            abs=1e-12,
        )
        assert fine.nodata == metadata["fixture_grid"]["fine_nodata"]
    with rasterio.open(coarse_path) as coarse:
        assert list(coarse.shape) == metadata["fixture_grid"]["coarse_shape"]
        assert coarse.crs.to_string() == metadata["source_grid"]["crs"]
        assert coarse.dtypes == (metadata["source_grid"]["dtype"],)
        assert list(coarse.transform)[:6] == pytest.approx(
            metadata["fixture_grid"]["coarse_transform"],
            rel=0,
            abs=1e-12,
        )
        assert list(coarse.bounds) == pytest.approx(
            metadata["fixture_grid"]["coarse_bounds"],
            rel=0,
            abs=1e-12,
        )
        assert coarse.nodata == metadata["fixture_grid"]["coarse_nodata"]

    hazard = pd.read_csv(hazard_path)
    population = _population_frame(
        population_path,
        metadata["fixture_grid"]["coordinate_decimal_places"],
    )
    assert len(hazard) == metadata["fixture_grid"]["hazard_rows"] == len(population)
    with rasterio.open(population_path) as source:
        assert (
            np.ma.getmaskarray(source.read(1, masked=True)).sum()
            == metadata["fixture_grid"]["fine_masked_cells"]
        )
    assert not hazard.duplicated(list(COORDINATES)).any()
    assert not population.duplicated(list(COORDINATES)).any()
    assert hazard["parent_1_degree_cell"].nunique() == 4

    expected_values = (
        population.set_index(list(COORDINATES))["population"]
        .reindex(pd.MultiIndex.from_frame(hazard.loc[:, list(COORDINATES)]))
        .to_numpy()
    )
    assert np.isfinite(expected_values).all()
    assert not np.isnan(expected_values).any()
    shuffled_population = population.sample(frac=1, random_state=20260831)
    result = assign_population(hazard, shuffled_population)

    assert result.index.equals(hazard.index)
    assert result.columns.tolist() == [*hazard.columns, "population"]
    pd.testing.assert_frame_equal(result.drop(columns="population"), hazard)
    np.testing.assert_array_equal(result["population"].to_numpy(), expected_values)

    with rasterio.open(coarse_path) as coarse:
        published = coarse.read(1, masked=True)
        assert not np.ma.getmaskarray(published).any()
        assert published.dtype == np.dtype(np.float32)
    for oracle in metadata["published_oracle"]["cells"]:
        parent = oracle["parent_1_degree_cell"]
        assigned = result.loc[
            result["parent_1_degree_cell"].eq(parent), "population"
        ].sum()
        coarse_row = (
            int(parent.split("-")[1]) - metadata["fixture_grid"]["coarse_window"][1]
        )
        coarse_column = (
            int(parent.split("-")[3]) - metadata["fixture_grid"]["coarse_window"][0]
        )
        assert float(published[coarse_row, coarse_column]) == pytest.approx(
            oracle["published_population"],
            rel=0,
            abs=0,
        )
        assert assigned == pytest.approx(
            oracle["published_population"],
            rel=0,
            abs=oracle["absolute_tolerance"],
        )
