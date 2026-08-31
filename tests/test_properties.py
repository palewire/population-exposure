"""Internal property-based regression checks for population assignment.

These generated examples prove only that the package continues to meet its own
row-order and exact-key behavior. They do not compare against external data,
published methods, or population estimates.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from population_exposure import assign_population

WEIGHTS = st.floats(
    min_value=0,
    max_value=1_000_000,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


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
