"""Population exposure calculation."""

from __future__ import annotations

import math
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
from population_exposure.bands import ExposureBands

if TYPE_CHECKING:
    from collections.abc import Sequence

_RESULT_COLUMNS = (
    "band_id",
    "band_order",
    "band_label",
    "lower_bound",
    "upper_bound",
    "population_total",
    "population_fraction",
    "represented_population",
)


def calculate_exposure(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    bands: ExposureBands,
    hazard_column: str,
    population_column: str = "population",
    cell_columns: Sequence[str] = ("longitude", "latitude"),
    group_by: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join hazard values to population weights and total each exposure band."""
    if not isinstance(hazard, pd.DataFrame):
        raise TypeError("hazard must be a pandas DataFrame.")
    if not isinstance(population, pd.DataFrame):
        raise TypeError("population must be a pandas DataFrame.")
    if not isinstance(bands, ExposureBands):
        raise TypeError("bands must be an ExposureBands instance.")
    if not isinstance(hazard_column, str) or not hazard_column:
        raise ValueError("hazard_column must be a non-empty column name.")
    if not isinstance(population_column, str) or not population_column:
        raise ValueError("population_column must be a non-empty column name.")

    cells = normalize_columns(cell_columns, parameter="cell_columns")
    groups = (
        () if group_by is None else normalize_columns(group_by, parameter="group_by")
    )
    _validate_column_roles(
        cells,
        groups,
        hazard_column=hazard_column,
        population_column=population_column,
    )

    require_columns(hazard, (*cells, hazard_column), frame_name="hazard")
    require_columns(
        population,
        (*cells, *groups, population_column),
        frame_name="population",
    )
    require_complete_keys(hazard, cells, frame_name="hazard")
    require_complete_keys(population, (*cells, *groups), frame_name="population")
    require_unique_keys(hazard, cells, frame_name="hazard")
    require_unique_keys(population, (*cells, *groups), frame_name="population")

    hazard_values = numeric_values(hazard, hazard_column, frame_name="hazard")
    population_values = numeric_values(
        population,
        population_column,
        frame_name="population",
    )
    if not np.isfinite(population_values).all():
        raise ValueError("Population weights must be finite.")
    if (population_values < 0).any():
        raise ValueError("Population weights must be non-negative.")

    cell_aliases = tuple(f"__cell_{index}" for index in range(len(cells)))
    group_aliases = tuple(f"__group_{index}" for index in range(len(groups)))
    hazard_work = hazard.loc[:, [*cells, hazard_column]].rename(
        columns={
            **dict(zip(cells, cell_aliases, strict=True)),
            hazard_column: "__hazard",
        }
    )
    population_work = population.loc[
        :,
        [*cells, *groups, population_column],
    ].rename(
        columns={
            **dict(zip(cells, cell_aliases, strict=True)),
            **dict(zip(groups, group_aliases, strict=True)),
            population_column: "__population",
        }
    )
    hazard_work["__hazard"] = hazard_values
    population_work["__population"] = population_values

    merged = population_work.merge(
        hazard_work,
        how="left",
        on=list(cell_aliases),
        sort=False,
        validate="many_to_one",
        indicator="__match",
    )
    unmatched = merged["__match"].eq("left_only")
    if unmatched.any():
        count = int(unmatched.sum())
        noun = "row" if count == 1 else "rows"
        raise ValueError(
            f"{count} population {noun} did not match a hazard cell exactly."
        )

    band_frame = _band_frame(bands)
    output = _empty_output_grid(
        population_work,
        group_aliases=group_aliases,
        band_frame=band_frame,
    )

    finite_hazard = np.isfinite(merged["__hazard"].to_numpy(dtype=float))
    represented = merged.loc[finite_hazard, [*group_aliases, "__population"]].copy()
    represented["__band_order"] = np.searchsorted(
        [band.upper_bound for band in bands.bands[:-1]],
        merged.loc[finite_hazard, "__hazard"].to_numpy(dtype=float),
        side="right",
    )

    if groups:
        totals = (
            represented.groupby(
                [*group_aliases, "__band_order"],
                sort=False,
                observed=True,
            )["__population"]
            .agg(math.fsum)
            .rename("population_total")
            .reset_index()
        )
        represented_totals = (
            represented.groupby(
                list(group_aliases),
                sort=False,
                observed=True,
            )["__population"]
            .agg(math.fsum)
            .rename("represented_population")
            .reset_index()
        )
        output = output.merge(
            totals,
            how="left",
            on=[*group_aliases, "__band_order"],
            sort=False,
        ).merge(
            represented_totals,
            how="left",
            on=list(group_aliases),
            sort=False,
        )
    else:
        totals = np.array(
            [
                math.fsum(
                    represented.loc[
                        represented["__band_order"].eq(band_order),
                        "__population",
                    ]
                )
                for band_order in range(len(bands.bands))
            ],
            dtype=float,
        )
        output["population_total"] = totals
        output["represented_population"] = math.fsum(represented["__population"])

    output["population_total"] = output["population_total"].fillna(0.0).astype(float)
    output["represented_population"] = (
        output["represented_population"].fillna(0.0).astype(float)
    )
    fractions = pd.Series(
        pd.array([pd.NA] * len(output), dtype="Float64"),
        index=output.index,
    )
    positive = output["represented_population"] > 0
    fractions.loc[positive] = (
        output.loc[positive, "population_total"]
        / output.loc[positive, "represented_population"]
    )
    output["population_fraction"] = fractions

    output = output.rename(
        columns={
            **dict(zip(group_aliases, groups, strict=True)),
            "__band_order": "band_order",
        }
    )
    result = output.loc[:, [*groups, *_RESULT_COLUMNS]].reset_index(drop=True)
    assert isinstance(result, pd.DataFrame)
    return result


def _validate_column_roles(
    cells: tuple[str, ...],
    groups: tuple[str, ...],
    *,
    hazard_column: str,
    population_column: str,
) -> None:
    overlap = set(cells) & set(groups)
    if overlap:
        raise ValueError("Group columns cannot also be cell columns.")
    if hazard_column in cells:
        raise ValueError("hazard_column cannot also be a cell column.")
    if population_column in (*cells, *groups):
        raise ValueError("population_column cannot also be a cell or group column.")
    if hazard_column == population_column:
        raise ValueError("hazard_column and population_column must be different.")
    reserved_groups = set(groups) & set(_RESULT_COLUMNS)
    if reserved_groups:
        raise ValueError("Group columns cannot use result column names.")


def _band_frame(bands: ExposureBands) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__band_order": range(len(bands.bands)),
            "band_id": [band.id for band in bands.bands],
            "band_label": [band.label for band in bands.bands],
            "lower_bound": [band.lower_bound for band in bands.bands],
            "upper_bound": [band.upper_bound for band in bands.bands],
        }
    )


def _empty_output_grid(
    population: pd.DataFrame,
    *,
    group_aliases: tuple[str, ...],
    band_frame: pd.DataFrame,
) -> pd.DataFrame:
    if not group_aliases:
        return band_frame.copy()

    observed_groups = population.loc[:, list(group_aliases)].drop_duplicates()
    try:
        observed_groups = observed_groups.sort_values(
            list(group_aliases),
            kind="stable",
        )
    except TypeError:
        sort_columns: list[str] = []
        for index, group_alias in enumerate(group_aliases):
            sort_column = f"__sort_{index}"
            sort_columns.append(sort_column)
            observed_groups[sort_column] = observed_groups[group_alias].map(
                _stable_group_key
            )
        observed_groups = observed_groups.sort_values(
            sort_columns,
            kind="stable",
        ).drop(columns=sort_columns)
    observed_groups = observed_groups.reset_index(drop=True)
    observed_groups["__cross"] = 1
    grid_bands = band_frame.copy()
    grid_bands["__cross"] = 1
    output = observed_groups.merge(grid_bands, on="__cross", sort=False).drop(
        columns="__cross"
    )
    assert isinstance(output, pd.DataFrame)
    return output


def _stable_group_key(value: object) -> str:
    """Return a comparable key for mixed-type group values."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"
