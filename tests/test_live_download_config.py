"""Offline tests that keep the real-download workflow bounded and opt-in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from population_exposure import populations
from tests.live_downloads import (
    CHAMBERS,
    DOWNLOADABLE_PROVIDERS,
    GHSL,
    GPW,
    LANDSCAN,
    SCHEDULED_PROVIDERS,
    WORLDPOP,
    compare_gpw_fine_to_coarse,
    download_failure_phase,
    gpw_coarse_oracle,
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


def test_gpw_oracle_uses_the_matching_official_count_product() -> None:
    selection = selection_for_provider(GPW)
    fine = populations.info(selection)
    oracle = gpw_coarse_oracle(selection)

    assert oracle.selection == selection
    assert oracle.official_url == fine.official_url.replace(
        f"_{fine.year}_30_sec_tif.zip",
        f"_{fine.year}_1_deg_tif.zip",
    )
    assert oracle.archive_member == fine.expected_filename.replace(
        "_30_sec.tif",
        "_1_deg.tif",
    )
    assert "unwpp" not in oracle.official_url.lower()


def test_gpw_oracle_rejects_a_non_gpw_selection() -> None:
    with pytest.raises(ValueError, match="not a GPW population-count"):
        gpw_coarse_oracle(selection_for_provider(WORLDPOP))


def test_gpw_parity_sums_exactly_aligned_tiny_fine_cells(
    tmp_path: Path,
) -> None:
    fine_path = _write_count_grid(
        tmp_path / "fine.tif",
        np.arange(1, 17, dtype=np.float32).reshape(4, 4),
        from_origin(-2, 2, 1, 1),
    )
    coarse_path = _write_count_grid(
        tmp_path / "coarse.tif",
        np.array([[14, 22], [46, 54]], dtype=np.float32),
        from_origin(-2, 2, 2, 2),
    )

    with rasterio.open(fine_path) as fine, rasterio.open(coarse_path) as coarse:
        parity = compare_gpw_fine_to_coarse(fine, coarse)

    assert parity.compared_cells == 4
    assert parity.maximum_absolute_difference == 0
    assert parity.maximum_tolerance_normalized_difference == 0
    assert parity.maximum_ulp_normalized_difference == 0
    assert parity.aggregate_difference == 0


def test_gpw_parity_allows_official_float32_publisher_quantization(
    tmp_path: Path,
) -> None:
    fine_path = _write_count_grid(
        tmp_path / "fine.tif",
        np.array([[33_554_432, 2], [0, 0]], dtype=np.float32),
        from_origin(-2, 2, 1, 1),
    )
    coarse_path = _write_count_grid(
        tmp_path / "coarse.tif",
        np.array([[33_554_432]], dtype=np.float32),
        from_origin(-2, 2, 2, 2),
    )

    with rasterio.open(fine_path) as fine, rasterio.open(coarse_path) as coarse:
        parity = compare_gpw_fine_to_coarse(fine, coarse)

    assert parity.maximum_absolute_difference == 2
    assert parity.maximum_tolerance >= 2
    assert parity.maximum_tolerance_normalized_difference <= 1
    assert parity.maximum_ulp_normalized_difference == pytest.approx(0.5)
    assert parity.aggregate_difference <= parity.aggregate_tolerance


def test_gpw_parity_rejects_a_material_coarse_count_mismatch(
    tmp_path: Path,
) -> None:
    fine_path = _write_count_grid(
        tmp_path / "fine.tif",
        np.array([[33_554_432, 2], [0, 0]], dtype=np.float32),
        from_origin(-2, 2, 1, 1),
    )
    coarse_path = _write_count_grid(
        tmp_path / "coarse.tif",
        np.array([[33_554_428]], dtype=np.float32),
        from_origin(-2, 2, 2, 2),
    )

    with rasterio.open(fine_path) as fine, rasterio.open(coarse_path) as coarse:
        parity = compare_gpw_fine_to_coarse(fine, coarse)

    assert parity.maximum_tolerance_normalized_difference > 1
    assert parity.aggregate_difference > parity.aggregate_tolerance


def test_gpw_parity_rejects_grids_without_a_shared_origin(tmp_path: Path) -> None:
    fine_path = _write_count_grid(
        tmp_path / "fine.tif",
        np.ones((4, 4), dtype=np.float32),
        from_origin(-2, 2, 1, 1),
    )
    coarse_path = _write_count_grid(
        tmp_path / "coarse.tif",
        np.ones((2, 2), dtype=np.float32),
        from_origin(-1, 2, 2, 2),
    )

    with rasterio.open(fine_path) as fine, rasterio.open(coarse_path) as coarse:
        with pytest.raises(ValueError, match="same origin"):
            compare_gpw_fine_to_coarse(fine, coarse)


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
    assert "tests/test_live_downloads.py" in workflow
    assert "uv python install 3.13" in workflow
    assert "timeout-minutes: 300" in workflow
    for choice in ("scheduled", *DOWNLOADABLE_PROVIDERS):
        assert f"          - {choice}" in workflow
    assert "          - all" not in workflow
    before_test, test_step = workflow.split(
        "      - name: Download, validate, receipt, and reuse offline",
        maxsplit=1,
    )
    assert "EARTHDATA_TOKEN:" not in before_test
    assert "EARTHDATA_TOKEN:" in test_step
    assert "secrets.EARTHDATA_TOKEN" in test_step
    assert "inputs.provider == 'gpwv4-r11-count'" in test_step


def _write_count_grid(path: Path, values: np.ndarray, transform) -> Path:
    """Write a one-band count GeoTIFF used only for alignment unit tests."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)
    return path
