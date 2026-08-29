"""Property tests for general exposure rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from population_exposure import ExposureBands, calculate_exposure

BANDS = ExposureBands.from_breaks(
    [-1, 0, 1],
    ids=("very_low", "low", "high", "very_high"),
)
FINITE_HAZARDS = st.floats(
    min_value=-1_000,
    max_value=1_000,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
WEIGHTS = st.floats(
    min_value=0,
    max_value=1_000_000,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
ROWS = st.lists(st.tuples(FINITE_HAZARDS, WEIGHTS), max_size=20)


def frames(
    rows: list[tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = len(rows)
    hazard = pd.DataFrame(
        {
            "longitude": np.arange(size, dtype=float),
            "latitude": np.zeros(size, dtype=float),
            "hazard": [row[0] for row in rows],
        }
    )
    population = pd.DataFrame(
        {
            "longitude": np.arange(size, dtype=float),
            "latitude": np.zeros(size, dtype=float),
            "population": [row[1] for row in rows],
        }
    )
    return hazard, population


@given(ROWS)
def test_conserves_represented_population(
    rows: list[tuple[float, float]],
) -> None:
    hazard, population = frames(rows)

    result = calculate_exposure(
        hazard,
        population,
        bands=BANDS,
        hazard_column="hazard",
    )

    expected = sum(row[1] for row in rows)
    np.testing.assert_allclose(result["population_total"].sum(), expected)
    np.testing.assert_allclose(
        result["represented_population"].unique(),
        [expected],
    )
    if expected > 0:
        np.testing.assert_allclose(
            result["population_fraction"].sum(),
            1.0,
        )
    else:
        assert result["population_fraction"].isna().all()


@given(ROWS, st.integers(min_value=0, max_value=2**32 - 2))
def test_row_permutations_do_not_change_results(
    rows: list[tuple[float, float]],
    seed: int,
) -> None:
    hazard, population = frames(rows)
    expected = calculate_exposure(
        hazard,
        population,
        bands=BANDS,
        hazard_column="hazard",
    )

    actual = calculate_exposure(
        hazard.sample(frac=1, random_state=seed).reset_index(drop=True),
        population.sample(frac=1, random_state=seed + 1).reset_index(drop=True),
        bands=BANDS,
        hazard_column="hazard",
    )

    pd.testing.assert_frame_equal(actual, expected)


@given(st.lists(FINITE_HAZARDS, max_size=30))
def test_every_finite_value_is_classified_once(values: list[float]) -> None:
    rows = [(value, 1.0) for value in values]
    hazard, population = frames(rows)

    result = calculate_exposure(
        hazard,
        population,
        bands=BANDS,
        hazard_column="hazard",
    )

    expected = np.bincount(
        np.searchsorted([-1, 0, 1], values, side="right"),
        minlength=4,
    )
    np.testing.assert_array_equal(result["population_total"], expected)
    assert result["population_total"].sum() == len(values)


@given(
    st.lists(
        st.tuples(FINITE_HAZARDS, WEIGHTS, st.integers(min_value=0, max_value=4)),
        max_size=20,
    )
)
def test_partitioned_group_totals_agree_with_global_totals(
    rows: list[tuple[float, float, int]],
) -> None:
    pairs = [(hazard, weight) for hazard, weight, _group in rows]
    hazard, population = frames(pairs)
    population["group"] = [group for _hazard, _weight, group in rows]

    global_result = calculate_exposure(
        hazard,
        population,
        bands=BANDS,
        hazard_column="hazard",
    )
    grouped_result = calculate_exposure(
        hazard,
        population,
        bands=BANDS,
        hazard_column="hazard",
        group_by="group",
    )

    grouped_totals = (
        grouped_result.groupby("band_order", sort=True)["population_total"]
        .sum()
        .reindex(range(4), fill_value=0)
    )
    np.testing.assert_allclose(
        grouped_totals.to_numpy(),
        global_result["population_total"].to_numpy(),
    )
