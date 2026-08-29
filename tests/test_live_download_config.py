"""Offline tests that keep the real-download workflow bounded and opt-in."""

from __future__ import annotations

from pathlib import Path

import pytest

from population_exposure import populations
from tests.live_downloads import (
    CHAMBERS,
    DOWNLOADABLE_PROVIDERS,
    GHSL,
    GPW,
    LANDSCAN,
    SCHEDULED_PROVIDERS,
    WORLDPOP,
    download_failure_phase,
    providers_for_run,
    selection_for_provider,
)


def test_scheduled_live_coverage_is_the_small_anonymous_subset() -> None:
    assert providers_for_run("scheduled") == (WORLDPOP, GHSL)
    assert CHAMBERS not in SCHEDULED_PROVIDERS
    assert GPW not in SCHEDULED_PROVIDERS
    assert LANDSCAN not in DOWNLOADABLE_PROVIDERS


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (WORLDPOP, (WORLDPOP,)),
        (GHSL, (GHSL,)),
        (GPW, (GPW,)),
        (CHAMBERS, (CHAMBERS,)),
        ("all", DOWNLOADABLE_PROVIDERS),
    ],
)
def test_manual_live_choices_cover_each_downloadable_provider(
    choice: str,
    expected: tuple[str, ...],
) -> None:
    assert providers_for_run(choice) == expected


def test_live_provider_choice_rejects_unknown_or_manual_sources() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        providers_for_run(LANDSCAN)


def test_live_selections_use_current_catalog_years_without_urls() -> None:
    source_years = {
        source.source_id: max(source.supported_years) for source in populations.list()
    }

    for source_id in DOWNLOADABLE_PROVIDERS:
        assert (
            selection_for_provider(source_id)
            == f"{source_id}:{source_years[source_id]}"
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Publisher md5 checksum verification failed", "checksum/size verification"),
        (
            "Server returned an invalid Content-Length header",
            "checksum/size verification",
        ),
        ("Official archive must contain exactly one GeoTIFF", "archive extraction"),
        ("'ghsl' requires CRS ESRI:54009", "raster validation"),
        ("Could not download the official file", "acquisition"),
    ],
)
def test_live_error_labels_identify_the_failed_download_stage(
    message: str,
    expected: str,
) -> None:
    assert download_failure_phase(ValueError(message)) == expected


def test_live_workflow_is_manual_or_monthly_and_never_pr_ci() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "live-downloads.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert 'cron: "17 5 1 * *"' in workflow
    assert "pull_request:" not in workflow
    assert "\n  push:" not in workflow
    assert "${{ inputs.provider || 'scheduled' }}" in workflow
    assert "EARTHDATA_TOKEN: ${{ secrets.EARTHDATA_TOKEN }}" in workflow
    assert "tests/test_live_downloads.py" in workflow
    for choice in ("scheduled", *DOWNLOADABLE_PROVIDERS, "all"):
        assert f"          - {choice}" in workflow
