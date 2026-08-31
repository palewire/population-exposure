"""Offline golden coverage for a verified GHSL tabular cell join."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from population_exposure import assign_population
from scripts.regenerate_ghsl_tabular_golden import _safe_relative_path

FIXTURE_DIRECTORY = Path(__file__).parent / "data" / "ghsl_aruba_tabular"


@pytest.mark.parametrize(
    "path",
    [
        r"cells\..\..\outside.csv",
        "cells/C:/outside.csv",
        r"cells\C:\outside.csv",
        "/outside.csv",
    ],
)
def test_ghsl_regeneration_rejects_unsafe_paths(path: str) -> None:
    """Keep ZIP and workbook path safety checks resistant to bypasses.

    Args:
        path: Unsafe path form supplied by the parametrized test.

    Returns:
        None.

    Examples:
        >>> test_ghsl_regeneration_rejects_unsafe_paths("/outside.csv")
    """
    with pytest.raises(ValueError, match="unsafe path"):
        _safe_relative_path(path, description="test path")


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deliberately reordered GHSL hazard and population cell tables.

    Args:
        None.

    Returns:
        Hazard table and reverse-ordered population table sharing exact keys.

    Examples:
        >>> hazard, population = _tables()
        >>> len(hazard) == len(population)
        True
    """
    cells = pd.read_csv(FIXTURE_DIRECTORY / "cells.csv")
    hazard = cells.loc[:, ["longitude", "latitude", "smod_class", "degurba_l1"]]
    population = cells.loc[:, ["longitude", "latitude", "population"]].iloc[::-1]
    return hazard, population


@pytest.mark.component
def test_ghsl_aruba_exact_coordinate_join_matches_recorded_reproduction() -> None:
    """Pin source cell order, class mapping, and measured workbook differences.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_ghsl_aruba_exact_coordinate_join_matches_recorded_reproduction()
    """
    metadata = json.loads(
        (FIXTURE_DIRECTORY / "metadata.json").read_text(encoding="utf-8")
    )
    cells_path = FIXTURE_DIRECTORY / "cells.csv"
    assert (
        hashlib.sha256(cells_path.read_bytes()).hexdigest()
        == metadata["fixture_files"]["cells.csv"]
    )
    hazard, population = _tables()

    assigned = assign_population(hazard, population)

    assert len(assigned) == metadata["grid"]["selected_cells"] == 216
    np.testing.assert_array_equal(
        assigned.loc[:, ["longitude", "latitude"]].to_numpy(),
        hazard.loc[:, ["longitude", "latitude"]].to_numpy(),
    )
    np.testing.assert_array_equal(
        assigned["degurba_l1"].to_numpy(), hazard["degurba_l1"].to_numpy()
    )
    totals = assigned.groupby("degurba_l1", sort=False)["population"].sum()
    for category, expected in metadata["reproduced"]["aruba_global_smod"].items():
        assert totals[category] == pytest.approx(expected, abs=1e-9)
        assert totals[category] - metadata["workbook"]["aruba"][category] == (
            pytest.approx(
                metadata["differences"]["aruba_global_smod_minus_workbook"][category],
                abs=1e-9,
            )
        )
    assert assigned.loc[assigned["smod_class"].eq(30), "degurba_l1"].eq("UC").all()
    assert (
        assigned.loc[assigned["smod_class"].isin((21, 22, 23)), "degurba_l1"]
        .eq("UCL")
        .all()
    )
    assert (
        assigned.loc[assigned["smod_class"].isin((11, 12, 13)), "degurba_l1"]
        .eq("RUR")
        .all()
    )


@pytest.mark.component
def test_ghsl_aruba_join_rejects_missing_and_nonfinite_population() -> None:
    """Keep the fixture's exact-match and finite-value failure paths covered.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_ghsl_aruba_join_rejects_missing_and_nonfinite_population()
    """
    hazard, population = _tables()
    with pytest.raises(ValueError, match="1 hazard row did not match"):
        assign_population(hazard, population.iloc[1:])
    nonfinite = population.copy()
    nonfinite.iloc[0, nonfinite.columns.get_loc("population")] = np.nan
    with pytest.raises(ValueError, match="Population values must be finite"):
        assign_population(hazard, nonfinite)
