"""Example-based tests for population assignment."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import population_exposure
from population_exposure import assign_population


def hazard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [12.0, 10.0, 11.0],
            "latitude": [20.0, 20.0, 20.0],
            "severity": ["warning", np.nan, "watch"],
            "county": ["South", "North", "North"],
        },
        index=pd.Index([7, 3, 5], name="source_row"),
    )


def population_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [10.0, 11.0, 12.0, 13.0],
            "latitude": [20.0, 20.0, 20.0, 20.0],
            "population": [100.0, 200.25, 300.5, 999.0],
            "unused": ["a", "b", "c", "d"],
        }
    )


def test_public_api_is_limited_to_documented_symbols() -> None:
    assert population_exposure.__all__ == [
        "CrsMismatchError",
        "PartialCoverageError",
        "RasterAssignment",
        "assign_population",
        "populations",
    ]
    assert population_exposure.assign_population is assign_population
    assert population_exposure.populations is not None


def test_tabular_parameters_remain_in_the_public_signature() -> None:
    parameters = inspect.signature(assign_population).parameters

    assert list(parameters)[:4] == [
        "hazard",
        "population",
        "cell_columns",
        "population_column",
    ]
    assert parameters["cell_columns"].default == ("longitude", "latitude")
    assert parameters["cell_columns"].annotation == "str | Sequence[str]"
    assert parameters["population_column"].default == "population"
    assert "gpd.GeoDataFrame" in parameters["hazard"].annotation


def test_assigns_population_and_preserves_input_structure() -> None:
    hazard = hazard_frame()

    result = assign_population(hazard, population_frame())

    assert result.columns.tolist() == [
        "longitude",
        "latitude",
        "severity",
        "county",
        "population",
    ]
    assert result.index.equals(hazard.index)
    assert result["population"].tolist() == [300.5, 100.0, 200.25]
    pd.testing.assert_frame_equal(result.drop(columns="population"), hazard)
    assert result["severity"].isna().tolist() == [False, True, False]


def test_custom_keys_and_population_column() -> None:
    hazard = pd.DataFrame(
        {
            "grid_x": [2, 1],
            "grid_y": [4, 3],
            "category": ["high", "low"],
        }
    )
    population = pd.DataFrame(
        {
            "grid_x": [1, 2],
            "grid_y": [3, 4],
            "people": [1.125, 2.75],
        }
    )

    result = assign_population(
        hazard,
        population,
        cell_columns=("grid_x", "grid_y"),
        population_column="people",
    )

    assert result.to_dict("records") == [
        {"grid_x": 2, "grid_y": 4, "category": "high", "people": 2.75},
        {"grid_x": 1, "grid_y": 3, "category": "low", "people": 1.125},
    ]
    assert result["people"].dtype == float


def test_single_string_cell_column_and_sparse_input() -> None:
    hazard = pd.DataFrame({"cell": ["C", "A"], "value": [np.inf, np.nan]})
    population = pd.DataFrame({"cell": ["A", "B", "C"], "population": [10, 20, 30]})

    result = assign_population(hazard, population, cell_columns="cell")

    assert result["cell"].tolist() == ["C", "A"]
    assert result["value"].isna().tolist() == [False, True]
    assert result["population"].tolist() == [30.0, 10.0]


def test_inputs_are_not_mutated() -> None:
    hazard = hazard_frame()
    population = population_frame()
    original_hazard = hazard.copy(deep=True)
    original_population = population.copy(deep=True)

    assign_population(hazard, population)

    pd.testing.assert_frame_equal(hazard, original_hazard)
    pd.testing.assert_frame_equal(population, original_population)


def test_population_order_does_not_change_output() -> None:
    hazard = hazard_frame()
    population = population_frame()

    forward = assign_population(hazard, population)
    reversed_population = assign_population(
        hazard,
        population.iloc[::-1].reset_index(drop=True),
    )

    pd.testing.assert_frame_equal(forward, reversed_population)


def test_empty_hazard_returns_empty_float_population_column() -> None:
    hazard = hazard_frame().iloc[:0]

    result = assign_population(hazard, population_frame())

    assert result.empty
    assert result.columns.tolist() == [*hazard.columns, "population"]
    assert result["population"].dtype == float
    assert result.index.equals(hazard.index)


def test_empty_hazard_and_population_use_an_empty_numeric_array() -> None:
    hazard = hazard_frame().iloc[:0]
    population = population_frame().iloc[:0]

    result = assign_population(hazard, population)

    assert result.empty
    assert result["population"].dtype == float


@pytest.mark.parametrize(
    ("target", "column", "message"),
    [
        ("hazard", "longitude", "hazard is missing required columns: 'longitude'"),
        (
            "population",
            "latitude",
            "population is missing required columns: 'latitude'",
        ),
        (
            "population",
            "population",
            "population is missing required columns: 'population'",
        ),
    ],
)
def test_missing_columns_fail(target: str, column: str, message: str) -> None:
    hazard = hazard_frame()
    population = population_frame()
    selected = hazard if target == "hazard" else population
    selected.drop(columns=column, inplace=True)

    with pytest.raises(ValueError, match=message):
        assign_population(hazard, population)


@pytest.mark.parametrize("target", ["hazard", "population"])
def test_null_keys_fail(target: str) -> None:
    hazard = hazard_frame()
    population = population_frame()
    selected = hazard if target == "hazard" else population
    selected.loc[selected.index[0], "longitude"] = np.nan

    with pytest.raises(ValueError, match=rf"{target} has null values in key columns"):
        assign_population(hazard, population)


@pytest.mark.parametrize("target", ["hazard", "population"])
def test_duplicate_keys_fail(target: str) -> None:
    hazard = hazard_frame()
    population = population_frame()
    selected = hazard if target == "hazard" else population
    duplicate = selected.iloc[[0]].copy()
    selected.loc[selected.index[-1], ["longitude", "latitude"]] = duplicate.loc[
        duplicate.index[0], ["longitude", "latitude"]
    ].to_numpy()

    with pytest.raises(ValueError, match=rf"{target} has duplicate rows"):
        assign_population(hazard, population)


def test_unmatched_hazard_cells_fail() -> None:
    hazard = hazard_frame()
    hazard.loc[7, "longitude"] = 99.0
    hazard.loc[5, "longitude"] = 98.0

    with pytest.raises(
        ValueError,
        match="2 hazard rows did not match a population cell exactly",
    ):
        assign_population(hazard, population_frame())


def test_single_unmatched_hazard_cell_uses_singular_message() -> None:
    hazard = hazard_frame()
    hazard.loc[7, "longitude"] = 99.0

    with pytest.raises(
        ValueError,
        match="1 hazard row did not match a population cell exactly",
    ):
        assign_population(hazard, population_frame())


def test_cell_keys_are_not_rounded() -> None:
    hazard = hazard_frame().iloc[[0]].copy()
    population = population_frame()
    hazard.iloc[0, hazard.columns.get_loc("longitude")] += 1e-12

    with pytest.raises(ValueError, match="did not match"):
        assign_population(hazard, population)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_population_values_must_be_finite(value: float) -> None:
    population = population_frame()
    population.loc[0, "population"] = value

    with pytest.raises(ValueError, match="Population values must be finite"):
        assign_population(hazard_frame(), population)


def test_population_values_must_be_non_negative() -> None:
    population = population_frame()
    population.loc[0, "population"] = -0.01

    with pytest.raises(ValueError, match="Population values must be non-negative"):
        assign_population(hazard_frame(), population)


@pytest.mark.parametrize("value", [["bad"] * 4, [True] * 4])
def test_population_values_must_be_numeric(value: list[object]) -> None:
    population = population_frame()
    population["population"] = value

    with pytest.raises(
        ValueError,
        match="population column 'population' must be numeric",
    ):
        assign_population(hazard_frame(), population)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cell_columns": ()}, "cell_columns must name at least one column"),
        ({"cell_columns": ("",)}, "non-empty column names"),
        ({"cell_columns": ("x", "x")}, "duplicate column names"),
        (
            {"cell_columns": ("population",)},
            "population_column cannot also be a cell column",
        ),
        ({"population_column": ""}, "population_column must be a non-empty"),
        ({"allow_overlaps": True}, "allow_overlaps applies only to vector"),
        ({"hazard_band": 1}, "hazard_band applies only to raster"),
        (
            {"conservation_tolerance": -1},
            "conservation_tolerance must be finite and non-negative",
        ),
    ],
)
def test_invalid_column_options_fail(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assign_population(hazard_frame(), population_frame(), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"allow_overlaps": 1}, TypeError, "allow_overlaps must be a boolean"),
        ({"hazard_band": True}, TypeError, "hazard_band must be an integer"),
        ({"hazard_band": "1"}, TypeError, "hazard_band must be an integer"),
        (
            {"conservation_tolerance": np.inf},
            ValueError,
            "conservation_tolerance must be finite",
        ),
        (
            {"conservation_tolerance": True},
            ValueError,
            "conservation_tolerance must be finite",
        ),
    ],
)
def test_invalid_shared_options_fail(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        assign_population(hazard_frame(), population_frame(), **kwargs)


def test_existing_output_column_fails() -> None:
    hazard = hazard_frame()
    hazard["population"] = 0

    with pytest.raises(ValueError, match="hazard already has a column named"):
        assign_population(hazard, population_frame())


def test_public_parameters_require_data_frames() -> None:
    with pytest.raises(TypeError, match="hazard must be a pandas DataFrame"):
        assign_population([], population_frame())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="population must be a pandas DataFrame"):
        assign_population(hazard_frame(), [])  # type: ignore[arg-type]
