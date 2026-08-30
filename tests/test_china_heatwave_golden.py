"""Offline golden validation for the published China heatwave result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from population_exposure import assign_population
from population_exposure.populations._http import DownloadResult
from validation.china_heatwave_2019 import reproduce as reproduction
from validation.china_heatwave_2019.method import (
    count_heatwave_days,
    heatwave_anomaly,
)
from validation.china_heatwave_2019.reproduce import _build_rows, _exposure

GOLDEN_DIRECTORY = Path(__file__).parent / "fixtures" / "china_heatwave_2019"


def test_counts_every_day_in_qualifying_runs() -> None:
    """Count full qualifying runs without counting shorter hot spells.

    Args:
        None.

    Returns:
        None.

    Examples:
        Pytest runs this check as part of the offline suite.
    """
    exceeded = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, True, True],
            [True, False, True],
        ],
        dtype=np.bool_,
    )

    result = count_heatwave_days(exceeded)

    assert result.tolist() == [3, 0, 4]


def test_heatwave_anomaly_uses_one_cellwise_warm_season_threshold() -> None:
    """Apply one pooled baseline threshold independently to each cell.

    Args:
        None.

    Returns:
        None.

    Examples:
        Pytest runs this check as part of the offline suite.
    """
    baseline = np.array(
        [
            [[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]]],
            [[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]]],
        ]
    )
    target = np.array([[[4.0, 40.0]], [[4.0, 40.0]], [[4.0, 40.0]]])

    result = heatwave_anomaly(baseline, target, percentile=50)

    assert result.threshold.tolist() == [[2.0, 20.0]]
    assert result.baseline_days_by_year.tolist() == [[[0, 0]], [[0, 0]]]
    assert result.baseline_days.tolist() == [[0.0, 0.0]]
    assert result.target_days.tolist() == [[3, 3]]
    assert result.additional_days.tolist() == [[3.0, 3.0]]


def test_baseline_exposure_uses_each_baseline_years_population() -> None:
    """Multiply annual heatwave days before averaging baseline exposure.

    Args:
        None.

    Returns:
        None.

    Examples:
        Pytest runs this check as part of the offline suite.
    """
    baseline_population = np.arange(1, 21, dtype=np.float64).reshape(20, 1, 1)
    population = np.concatenate([baseline_population, np.array([[[100.0]]])])
    baseline_days_by_year = np.ones((20, 1, 1), dtype=np.int16)
    rows = _build_rows(
        np.array([1.0]),
        np.array([2.0]),
        population,
        np.array([[True]]),
        baseline_days_by_year,
        np.array([[1.0]]),
        np.array([[3]], dtype=np.int16),
        np.array([[2.0]]),
    )

    result = _exposure(rows)

    assert rows["baseline_heatwave_person_days"].item() == 10.5
    assert result == pytest.approx((289.5, 100.0, 2.895, 300.0, 10.5))


@pytest.mark.parametrize(
    ("population_value", "mask_value", "message"),
    [
        (1.0, False, "selected no cells"),
        (0.0, True, "population total must be finite and positive"),
    ],
)
def test_empty_or_zero_population_selection_fails_clearly(
    population_value: float,
    mask_value: bool,
    message: str,
) -> None:
    """Reject selections that cannot produce a per-person result.

    Args:
        population_value: Population assigned to every test year.
        mask_value: Whether the single test cell is selected.
        message: Expected validation-error text.

    Returns:
        None.

    Examples:
        Pytest runs both empty-mask and zero-population cases.
    """
    population = np.full((21, 1, 1), population_value, dtype=np.float64)

    with pytest.raises(ValueError, match=message):
        _build_rows(
            np.array([1.0]),
            np.array([2.0]),
            population,
            np.array([[mask_value]]),
            np.ones((20, 1, 1), dtype=np.int16),
            np.array([[1.0]]),
            np.array([[3]], dtype=np.int16),
            np.array([[2.0]]),
        )


def test_invalid_cached_download_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace a bad owned cache file with verified source bytes.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest attribute replacement helper.

    Returns:
        None.

    Examples:
        Pytest runs this check without making a network request.
    """
    destination = tmp_path / "source.bin"
    destination.write_bytes(b"bad")
    expected = b"verified"
    expected_sha256 = hashlib.sha256(expected).hexdigest()

    def fake_download(url, partial_path, **kwargs):
        assert not destination.exists()
        partial_path.write_bytes(expected)
        return DownloadResult(size=len(expected), sha256=expected_sha256)

    monkeypatch.setattr(reproduction, "download_file", fake_download)

    result = reproduction._download(
        "https://example.test/source.bin",
        destination,
        expected_size=len(expected),
        expected_sha256=expected_sha256,
    )

    assert result == destination
    assert result.read_bytes() == expected


def test_invalid_cached_country_raster_is_reextracted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace a bad extracted raster from the verified source archive.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest attribute replacement helper.

    Returns:
        None.

    Examples:
        Pytest runs this check with a tiny local ZIP archive.
    """
    expected = b"replacement raster"
    archive = tmp_path / "countries.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(reproduction.COUNTRY_RASTER_FILENAME, expected)
    destination = tmp_path / reproduction.COUNTRY_RASTER_FILENAME
    destination.write_bytes(b"bad")
    monkeypatch.setattr(reproduction, "COUNTRY_RASTER_SIZE", len(expected))
    monkeypatch.setattr(
        reproduction,
        "COUNTRY_RASTER_SHA256",
        hashlib.sha256(expected).hexdigest(),
    )

    result = reproduction._extract_country_raster(archive, destination)

    assert result == destination
    assert result.read_bytes() == expected


@pytest.mark.integration
def test_china_2019_heatwave_exposure_matches_regenerated_golden() -> None:
    """Assign golden population by exact coordinates and aggregate exposure.

    Args:
        None.

    Returns:
        None.

    Examples:
        Pytest runs this network-independent check during ``make verify``.
    """
    metadata_path = GOLDEN_DIRECTORY / "golden.json"
    cells_path = GOLDEN_DIRECTORY / "cells.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cells_bytes = cells_path.read_bytes()
    cells = pd.read_csv(cells_path)

    method = metadata["method"]
    assert method["warm_season"] == {
        "days_per_year": 153,
        "end": "09-30",
        "start": "05-01",
    }
    threshold = method["threshold"]
    assert {
        key: threshold[key]
        for key in (
            "baseline_years",
            "comparison",
            "percentile",
            "quantile_method",
            "sample",
        )
    } == {
        "baseline_years": [1986, 2005],
        "comparison": "strictly greater than",
        "percentile": 92.5,
        "quantile_method": "linear",
        "sample": "all 3060 baseline warm-season daily maxima per cell",
    }
    assert threshold["inclusive_comparison_sensitivity"]["comparison"] == (
        "greater than or equal to"
    )
    assert method["heatwave"] == {
        "counting": "all days in each qualifying run",
        "minimum_consecutive_days": 3,
        "run_boundary": "each May-September warm season resets",
    }
    assert method["population"]["record"] == "Zenodo 3768003"
    assert method["population"]["age_band_lower_bound"] == 65
    assert method["population"]["baseline_years"] == [1986, 2005]
    assert method["population"]["target_year"] == 2019
    extractions = method["temperature"]["extractions"]
    assert len(extractions) == 21
    assert [
        int(Path(extraction["file"]).stem.split("_")[-1]) for extraction in extractions
    ] == [*range(1986, 2006), 2019]
    assert all(extraction["size_bytes"] > 0 for extraction in extractions)
    assert all(len(extraction["sha256"]) == 64 for extraction in extractions)
    assert method["geography"]["selection"] == "UN M49 country code 156"
    assert method["geography"]["finite_population_cells"] == len(cells)
    assert (
        abs(
            method["geography"]["mask_cells"]
            - method["geography"]["reference_mainland_grid_cells"]
        )
        / method["geography"]["reference_mainland_grid_cells"]
        <= method["geography"]["reference_grid_cell_relative_tolerance"]
    )
    assert hashlib.sha256(cells_bytes).hexdigest() == metadata["fixture"]["sha256"]
    assert len(cells) == metadata["fixture"]["rows"]
    assert cells.columns.tolist() == metadata["fixture"]["columns"]
    assert cells.notna().all(axis=None)
    assert not cells[["longitude", "latitude"]].duplicated().any()
    assert (cells["population_65_plus_2019"] >= 0).all()
    assert np.allclose(cells["longitude"] * 2, np.round(cells["longitude"] * 2))
    assert np.allclose(cells["latitude"] * 2, np.round(cells["latitude"] * 2))
    assert cells["heatwave_days_2019"].between(0, 153).all()
    assert np.allclose(
        cells["heatwave_days_2019"],
        np.round(cells["heatwave_days_2019"]),
    )
    assert cells["baseline_heatwave_days"].between(0, 153).all()
    assert np.allclose(
        cells["baseline_heatwave_days"] * 20,
        np.round(cells["baseline_heatwave_days"] * 20),
    )
    np.testing.assert_allclose(
        cells["additional_heatwave_days"],
        cells["heatwave_days_2019"] - cells["baseline_heatwave_days"],
        rtol=0,
        atol=1e-12,
    )
    assert (cells["baseline_heatwave_person_days"] >= 0).all()

    hazard = cells[
        [
            "longitude",
            "latitude",
            "heatwave_days_2019",
            "baseline_heatwave_days",
            "additional_heatwave_days",
            "baseline_heatwave_person_days",
            "additional_heatwave_person_days",
        ]
    ]
    population = cells[["longitude", "latitude", "population_65_plus_2019"]].iloc[::-1]
    assigned = assign_population(
        hazard,
        population,
        population_column="population_65_plus_2019",
    )
    additional_person_days = assigned["heatwave_days_2019"].to_numpy(
        dtype=np.float64
    ) * assigned["population_65_plus_2019"].to_numpy(dtype=np.float64) - assigned[
        "baseline_heatwave_person_days"
    ].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        additional_person_days,
        assigned["additional_heatwave_person_days"].to_numpy(dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
    target_person_days = float(
        np.sum(
            assigned["heatwave_days_2019"].to_numpy(dtype=np.float64)
            * assigned["population_65_plus_2019"].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    )
    baseline_person_days = float(
        np.sum(
            assigned["baseline_heatwave_person_days"].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    )
    person_days = float(np.sum(additional_person_days, dtype=np.float64))
    assert person_days == pytest.approx(
        target_person_days - baseline_person_days,
        abs=1e-6,
    )
    total_population = float(
        np.sum(
            assigned["population_65_plus_2019"].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    )
    days_per_person = person_days / total_population

    reproduced = metadata["reproduction"]
    publication = metadata["publication"]
    assert person_days == pytest.approx(
        reproduced["additional_person_days"],
        abs=1e-3,
    )
    assert total_population == pytest.approx(
        reproduced["population_65_plus"],
        abs=1e-3,
    )
    assert target_person_days == pytest.approx(
        reproduced["heatwave_person_days_2019"],
        abs=1e-3,
    )
    assert baseline_person_days == pytest.approx(
        reproduced["mean_annual_heatwave_person_days_1986_2005"],
        abs=1e-3,
    )
    assert days_per_person == pytest.approx(
        reproduced["additional_days_per_person"],
        abs=1e-12,
    )
    matches_reported_days = (
        abs(days_per_person - publication["reported_additional_days_per_person"])
        <= publication["reported_additional_days_per_person_rounding_tolerance"]
    )
    assert (
        matches_reported_days
        is metadata["publication_comparison"][
            "method_result_within_reported_days_per_person_tolerance"
        ]
    )
    assert (
        abs(person_days - publication["reported_additional_person_days"])
        <= publication["reported_additional_person_days_rounding_tolerance"]
    ) is metadata["publication_comparison"][
        "method_result_within_reported_person_days_tolerance"
    ]
    assert (
        abs(target_person_days - publication["reported_additional_person_days"])
        <= publication["reported_additional_person_days_rounding_tolerance"]
    ) is metadata["publication_comparison"][
        "absolute_2019_within_reported_person_days_tolerance"
    ]
    expected_status = (
        "matches reported additional exposure"
        if metadata["publication_comparison"][
            "method_result_within_reported_person_days_tolerance"
        ]
        and metadata["publication_comparison"][
            "method_result_within_reported_days_per_person_tolerance"
        ]
        else "outside reported additional exposure tolerance"
    )
    assert metadata["publication_comparison"]["status"] == expected_status
    assert expected_status == "outside reported additional exposure tolerance"
    assert not metadata["series_diagnostics"]["year_2000"][
        "method_within_reported_tolerance"
    ]
