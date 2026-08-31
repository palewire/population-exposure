"""Internal property-based regression checks for population assignment.

These generated examples prove only that the package continues to meet its own
row-order and exact-key behavior. They do not compare against external data,
published methods, or population estimates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from hypothesis import given, settings
from hypothesis import strategies as st
from rasterio.transform import from_origin
from shapely.geometry import box

import population_exposure as pe
from population_exposure import assign_population

WEIGHTS = st.floats(
    min_value=0,
    max_value=1_000_000,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
COUNTS = st.floats(
    min_value=0,
    max_value=10_000,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
NODATA = -9999.0


@settings(deadline=None)
@given(st.lists(WEIGHTS, max_size=30), st.integers(min_value=0, max_value=2**32 - 1))
def test_assignment_preserves_rows_and_matches_keys(
    weights: list[float],
    seed: int,
) -> None:
    cells = list(range(len(weights)))
    hazard = pd.DataFrame(
        {
            "cell": cells[::-1],
            "hazard": [None if cell % 2 else cell for cell in cells],
        }
    )
    population = pd.DataFrame({"cell": cells, "population": weights}).sample(
        frac=1,
        random_state=seed,
    )

    result = assign_population(hazard, population, cell_columns="cell")

    assert result["cell"].tolist() == cells[::-1]
    pd.testing.assert_series_equal(result["hazard"], hazard["hazard"])
    assert result["population"].tolist() == list(reversed(weights))


@settings(max_examples=40, deadline=None)
@given(
    st.lists(COUNTS, min_size=4, max_size=4),
    st.lists(st.booleans(), min_size=4, max_size=4),
)
def test_no_data_is_never_reported_as_a_number(
    counts: list[float],
    present: list[bool],
) -> None:
    """No-data never becomes a count, and every real value is added up."""
    values = np.where(present, counts, NODATA).reshape(2, 2)
    if not any(present):
        # A raster with nothing in it at all is rejected earlier.
        return
    hazard = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "population.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float64",
            crs="EPSG:3857",
            transform=from_origin(0, 2, 1, 1),
            nodata=NODATA,
        ) as dataset:
            dataset.write(values, 1)

        result = pe.assign_population(
            hazard,
            path,
            allow_missing_population_data=True,
        )

    expected = sum(count for count, keep in zip(counts, present, strict=True) if keep)
    assert result["population"].item() == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert result["population_data_fraction"].item() == pytest.approx(sum(present) / 4)
    assert bool(result["population_data_complete"].item()) is all(present)
