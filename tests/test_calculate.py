"""Example-based tests for exposure calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import population_exposure
from population_exposure import ExposureBand, ExposureBands, calculate_exposure


@pytest.fixture
def bands() -> ExposureBands:
    return ExposureBands.from_breaks(
        [-2, 2],
        ids=("below", "near", "above"),
        labels=("Below", "Near", "Above"),
    )


def frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": np.arange(len(values), dtype=float),
            "latitude": np.zeros(len(values), dtype=float),
            "value": values,
        }
    )


def populations(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": np.arange(len(values), dtype=float),
            "latitude": np.zeros(len(values), dtype=float),
            "population": values,
        }
    )


def test_public_api_is_limited_to_documented_symbols() -> None:
    assert population_exposure.__all__ == [
        "ExposureBand",
        "ExposureBands",
        "calculate_exposure",
    ]
    assert population_exposure.ExposureBand is ExposureBand


def test_boundaries_enter_the_higher_band(bands: ExposureBands) -> None:
    result = calculate_exposure(
        frame([-3, -2, 1, 2, 8]),
        populations([1, 2, 3, 4, 5]),
        bands=bands,
        hazard_column="value",
    )

    assert result.columns.tolist() == [
        "band_id",
        "band_order",
        "band_label",
        "lower_bound",
        "upper_bound",
        "population_total",
        "population_fraction",
        "represented_population",
    ]
    assert result["band_id"].tolist() == ["below", "near", "above"]
    assert result["band_order"].tolist() == [0, 1, 2]
    assert result["population_total"].tolist() == [1.0, 5.0, 9.0]
    assert result["represented_population"].tolist() == [15.0, 15.0, 15.0]
    np.testing.assert_allclose(
        result["population_fraction"].to_numpy(dtype=float),
        [1 / 15, 5 / 15, 9 / 15],
    )


def test_non_finite_hazards_are_not_represented(bands: ExposureBands) -> None:
    result = calculate_exposure(
        frame([-np.inf, np.nan, -2, 2, np.inf]),
        populations([1, 2, 3, 4, 5]),
        bands=bands,
        hazard_column="value",
    )

    assert result["population_total"].tolist() == [0.0, 3.0, 4.0]
    assert result["represented_population"].tolist() == [7.0, 7.0, 7.0]


def test_zero_population_has_nullable_fractions(bands: ExposureBands) -> None:
    result = calculate_exposure(
        frame([-3, 0, 3]),
        populations([0, 0, 0]),
        bands=bands,
        hazard_column="value",
    )

    assert result["population_total"].tolist() == [0.0, 0.0, 0.0]
    assert result["represented_population"].tolist() == [0.0, 0.0, 0.0]
    assert str(result["population_fraction"].dtype) == "Float64"
    assert result["population_fraction"].isna().all()


def test_empty_global_input_still_emits_every_band(bands: ExposureBands) -> None:
    hazard = pd.DataFrame(
        {
            "longitude": pd.Series(dtype=float),
            "latitude": pd.Series(dtype=float),
            "value": pd.Series(dtype=float),
        }
    )
    population = pd.DataFrame(
        {
            "longitude": pd.Series(dtype=float),
            "latitude": pd.Series(dtype=float),
            "population": pd.Series(dtype=float),
        }
    )

    result = calculate_exposure(
        hazard,
        population,
        bands=bands,
        hazard_column="value",
    )

    assert result["band_id"].tolist() == ["below", "near", "above"]
    assert result["population_total"].eq(0).all()
    assert result["population_fraction"].isna().all()


def test_extra_hazard_rows_are_harmless(bands: ExposureBands) -> None:
    result = calculate_exposure(
        frame([-3, 0, 3]),
        populations([2, 4]),
        bands=bands,
        hazard_column="value",
    )

    assert result["population_total"].tolist() == [2.0, 4.0, 0.0]


def test_grouped_output_is_sorted_and_emits_empty_bands(
    bands: ExposureBands,
) -> None:
    population = populations([2, 4, 8])
    population["region"] = ["west", "east", "west"]

    result = calculate_exposure(
        frame([-3, 0, 3]),
        population,
        bands=bands,
        hazard_column="value",
        group_by="region",
    )

    assert result[["region", "band_id"]].to_records(index=False).tolist() == [
        ("east", "below"),
        ("east", "near"),
        ("east", "above"),
        ("west", "below"),
        ("west", "near"),
        ("west", "above"),
    ]
    assert result["population_total"].tolist() == [0, 4, 0, 2, 0, 8]
    assert result["represented_population"].tolist() == [4, 4, 4, 10, 10, 10]


def test_mixed_type_groups_have_deterministic_order(bands: ExposureBands) -> None:
    population = populations([2, 4])
    population["region"] = pd.Series([1, "east"], dtype=object)

    result = calculate_exposure(
        frame([-3, 3]),
        population,
        bands=bands,
        hazard_column="value",
        group_by="region",
    )

    assert result["region"].drop_duplicates().tolist() == [1, "east"]


def test_overlapping_groups_are_calculated_independently(
    bands: ExposureBands,
) -> None:
    hazard = frame([-3, 3])
    population = pd.DataFrame(
        {
            "longitude": [0.0, 0.0, 1.0],
            "latitude": [0.0, 0.0, 0.0],
            "region": ["all", "west", "all"],
            "population": [2.0, 2.0, 5.0],
        }
    )

    result = calculate_exposure(
        hazard,
        population,
        bands=bands,
        hazard_column="value",
        group_by="region",
    )

    totals = result.pivot(index="region", columns="band_id", values="population_total")
    assert totals.loc["all", "below"] == 2
    assert totals.loc["all", "above"] == 5
    assert totals.loc["west", "below"] == 2
    assert totals.loc["west", "above"] == 0


def test_multiple_groups_and_custom_columns(bands: ExposureBands) -> None:
    hazard = pd.DataFrame({"x": [1, 2], "y": [3, 4], "score": [-5, 5]})
    population = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
            "country": ["B", "A"],
            "class": [2, 1],
            "people": [7, 9],
        }
    )

    result = calculate_exposure(
        hazard,
        population,
        bands=bands,
        hazard_column="score",
        population_column="people",
        cell_columns=("x", "y"),
        group_by=("country", "class"),
    )

    assert result.columns[:2].tolist() == ["country", "class"]
    assert result[["country", "class"]].drop_duplicates().to_records(
        index=False
    ).tolist() == [
        ("A", 1),
        ("B", 2),
    ]
    assert result["population_total"].sum() == 16


def test_inputs_are_not_mutated(bands: ExposureBands) -> None:
    hazard = frame([-3, 3])
    population = populations([2, 5])
    original_hazard = hazard.copy(deep=True)
    original_population = population.copy(deep=True)

    calculate_exposure(
        hazard,
        population,
        bands=bands,
        hazard_column="value",
    )

    pd.testing.assert_frame_equal(hazard, original_hazard)
    pd.testing.assert_frame_equal(population, original_population)


def test_float_totals_do_not_depend_on_row_order(bands: ExposureBands) -> None:
    hazard = frame([3, 3, 3])
    population = populations([1e16, 1, 1])

    forward = calculate_exposure(
        hazard,
        population,
        bands=bands,
        hazard_column="value",
    )
    reversed_result = calculate_exposure(
        hazard.iloc[::-1],
        population.iloc[::-1],
        bands=bands,
        hazard_column="value",
    )

    pd.testing.assert_frame_equal(forward, reversed_result)
    assert forward.loc[2, "population_total"] == 1.0000000000000002e16


@pytest.mark.parametrize(
    ("target", "column", "message"),
    [
        ("hazard", "value", "hazard column 'value' must be numeric"),
        ("population", "population", "population column 'population' must be numeric"),
    ],
)
def test_value_columns_must_be_numeric(
    bands: ExposureBands,
    target: str,
    column: str,
    message: str,
) -> None:
    hazard = frame([1])
    population = populations([1])
    selected = hazard if target == "hazard" else population
    selected[column] = ["bad"]

    with pytest.raises(ValueError, match=message):
        calculate_exposure(
            hazard,
            population,
            bands=bands,
            hazard_column="value",
        )


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf])
def test_population_weights_must_be_finite(
    bands: ExposureBands,
    weight: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        calculate_exposure(
            frame([1]),
            populations([weight]),
            bands=bands,
            hazard_column="value",
        )


def test_population_weights_must_be_non_negative(bands: ExposureBands) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_exposure(
            frame([1]),
            populations([-1]),
            bands=bands,
            hazard_column="value",
        )


@pytest.mark.parametrize(
    ("target", "columns", "message"),
    [
        ("hazard", ("longitude",), "hazard has null values"),
        ("population", ("latitude",), "population has null values"),
        ("population", ("region",), "population has null values"),
    ],
)
def test_null_keys_fail(
    bands: ExposureBands,
    target: str,
    columns: tuple[str, ...],
    message: str,
) -> None:
    hazard = frame([1])
    population = populations([1])
    population["region"] = "north"
    selected = hazard if target == "hazard" else population
    selected.loc[0, columns[0]] = None

    with pytest.raises(ValueError, match=message):
        calculate_exposure(
            hazard,
            population,
            bands=bands,
            hazard_column="value",
            group_by="region",
        )


def test_duplicate_hazard_cells_fail(bands: ExposureBands) -> None:
    hazard = pd.concat([frame([1]), frame([2])], ignore_index=True)

    with pytest.raises(ValueError, match="hazard has duplicate rows"):
        calculate_exposure(
            hazard,
            populations([1]),
            bands=bands,
            hazard_column="value",
        )


def test_duplicate_population_cells_within_a_group_fail(
    bands: ExposureBands,
) -> None:
    population = pd.concat([populations([1]), populations([2])], ignore_index=True)
    population["region"] = "north"

    with pytest.raises(ValueError, match="population has duplicate rows"):
        calculate_exposure(
            frame([1]),
            population,
            bands=bands,
            hazard_column="value",
            group_by="region",
        )


def test_unmatched_population_cells_fail(bands: ExposureBands) -> None:
    with pytest.raises(ValueError, match="1 population row did not match"):
        calculate_exposure(
            frame([1]),
            populations([1, 2]),
            bands=bands,
            hazard_column="value",
        )


def test_cell_keys_are_not_rounded(bands: ExposureBands) -> None:
    hazard = frame([1])
    population = populations([1])
    population.loc[0, "longitude"] += 1e-12

    with pytest.raises(ValueError, match="did not match"):
        calculate_exposure(
            hazard,
            population,
            bands=bands,
            hazard_column="value",
        )


@pytest.mark.parametrize(
    ("hazard", "population", "message"),
    [
        (
            frame([1]).drop(columns="value"),
            populations([1]),
            "hazard is missing required columns: 'value'",
        ),
        (
            frame([1]),
            populations([1]).drop(columns="population"),
            "population is missing required columns: 'population'",
        ),
    ],
)
def test_missing_columns_fail(
    bands: ExposureBands,
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_exposure(
            hazard,
            population,
            bands=bands,
            hazard_column="value",
        )


def test_group_columns_must_come_from_population(bands: ExposureBands) -> None:
    hazard = frame([1])
    hazard["region"] = "north"

    with pytest.raises(
        ValueError,
        match="population is missing required columns: 'region'",
    ):
        calculate_exposure(
            hazard,
            populations([1]),
            bands=bands,
            hazard_column="value",
            group_by="region",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cell_columns": ()}, "cell_columns must name"),
        ({"cell_columns": ("",)}, "non-empty column names"),
        ({"cell_columns": ("x", "x")}, "duplicate column"),
        ({"group_by": []}, "group_by must name"),
        ({"group_by": ("region", "region")}, "duplicate column"),
        ({"cell_columns": ("longitude",), "group_by": "longitude"}, "cannot also"),
        ({"cell_columns": ("value",)}, "hazard_column cannot also"),
        ({"population_column": "longitude"}, "population_column cannot also"),
        ({"population_column": "value"}, "must be different"),
        ({"group_by": "band_id"}, "result column names"),
    ],
)
def test_invalid_column_roles_fail(
    bands: ExposureBands,
    kwargs: dict[str, object],
    message: str,
) -> None:
    hazard = frame([1])
    hazard["band_id"] = "hazard-group"
    population = populations([1])
    population["region"] = "north"
    population["band_id"] = "reserved"
    population["value"] = 1

    with pytest.raises(ValueError, match=message):
        calculate_exposure(
            hazard,
            population,
            bands=bands,
            hazard_column="value",
            **kwargs,
        )


def test_public_parameters_require_expected_types(bands: ExposureBands) -> None:
    with pytest.raises(TypeError, match="hazard must"):
        calculate_exposure(
            [],  # type: ignore[arg-type]
            populations([1]),
            bands=bands,
            hazard_column="value",
        )
    with pytest.raises(TypeError, match="population must"):
        calculate_exposure(
            frame([1]),
            [],  # type: ignore[arg-type]
            bands=bands,
            hazard_column="value",
        )
    with pytest.raises(TypeError, match="ExposureBands"):
        calculate_exposure(
            frame([1]),
            populations([1]),
            bands="bad",  # type: ignore[arg-type]
            hazard_column="value",
        )
    with pytest.raises(ValueError, match="hazard_column must"):
        calculate_exposure(
            frame([1]),
            populations([1]),
            bands=bands,
            hazard_column="",
        )
    with pytest.raises(ValueError, match="population_column must"):
        calculate_exposure(
            frame([1]),
            populations([1]),
            bands=bands,
            hazard_column="value",
            population_column="",
        )
