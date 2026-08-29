"""Population assignment for tabular hazard data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from population_exposure._validation import (
    normalize_columns,
    numeric_values,
    require_columns,
    require_complete_keys,
    require_unique_keys,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def assign_population(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
) -> pd.DataFrame:
    """Return the hazard rows with population assigned by exact cell keys."""
    if not isinstance(hazard, pd.DataFrame):
        raise TypeError("hazard must be a pandas DataFrame.")
    if not isinstance(population, pd.DataFrame):
        raise TypeError("population must be a pandas DataFrame.")
    if not isinstance(population_column, str) or not population_column:
        raise ValueError("population_column must be a non-empty column name.")

    cells = normalize_columns(cell_columns, parameter="cell_columns")
    if population_column in cells:
        raise ValueError("population_column cannot also be a cell column.")

    require_columns(hazard, cells, frame_name="hazard")
    require_columns(population, (*cells, population_column), frame_name="population")
    if population_column in hazard.columns:
        raise ValueError(
            f"hazard already has a column named {population_column!r}; "
            "choose a different population_column."
        )

    require_complete_keys(hazard, cells, frame_name="hazard")
    require_complete_keys(population, cells, frame_name="population")
    require_unique_keys(hazard, cells, frame_name="hazard")
    require_unique_keys(population, cells, frame_name="population")

    population_values = numeric_values(
        population,
        population_column,
        frame_name="population",
    )
    if not np.isfinite(population_values).all():
        raise ValueError("Population values must be finite.")
    if (population_values < 0).any():
        raise ValueError("Population values must be non-negative.")

    cell_aliases = tuple(f"__cell_{index}" for index in range(len(cells)))
    hazard_keys = hazard.loc[:, list(cells)].rename(
        columns=dict(zip(cells, cell_aliases, strict=True))
    )
    hazard_keys["__row_order"] = np.arange(len(hazard), dtype=np.int64)

    population_work = population.loc[:, list(cells)].rename(
        columns=dict(zip(cells, cell_aliases, strict=True))
    )
    population_work["__population"] = population_values

    assigned = hazard_keys.merge(
        population_work,
        how="left",
        on=list(cell_aliases),
        sort=False,
        validate="one_to_one",
        indicator="__match",
    ).sort_values("__row_order", kind="stable")
    unmatched = assigned["__match"].eq("left_only")
    if unmatched.any():
        count = int(unmatched.sum())
        noun = "row" if count == 1 else "rows"
        raise ValueError(
            f"{count} hazard {noun} did not match a population cell exactly."
        )

    result = hazard.copy(deep=True)
    result[population_column] = assigned["__population"].to_numpy(dtype=float)
    return result
