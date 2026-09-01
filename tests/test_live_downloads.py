"""Opt-in tests that exercise complete official population-provider downloads."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import rasterio

from population_exposure import populations
from population_exposure.populations import _api
from population_exposure.populations._cache import (
    _json_safe,
    cache_entry,
    verified_receipt,
)
from population_exposure.populations._http import download_file
from population_exposure.populations._raster import validate_catalog_raster
from population_exposure.populations._sources import SOURCES
from tests.live_downloads import (
    GPW,
    compare_gpw_fine_to_coarse,
    download_failure_phase,
    gpw_coarse_oracle,
    providers_for_run,
    selection_for_provider,
)

if TYPE_CHECKING:
    from pathlib import Path


def _fail(source_id: str, selection: str, phase: str, error: Exception) -> None:
    pytest.fail(f"{source_id} ({selection}) {phase} failed: {error}")


@pytest.mark.live
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("source_id", providers_for_run())
def test_official_provider_download_is_validated_receipted_and_reused_offline(
    source_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download one complete provider artifact into a fresh cache via the public API."""
    selection = selection_for_provider(source_id)
    selected = populations.info(selection)
    source = SOURCES[source_id]
    cache_dir = tmp_path / "cache"
    token = os.environ.get("EARTHDATA_TOKEN") if source_id == GPW else None

    if source_id == GPW and not token:
        pytest.fail(
            "gpwv4-r11-count requires the EARTHDATA_TOKEN secret for this manual "
            "live run; configure the secret and dispatch the workflow again."
        )

    try:
        downloaded = populations.download(
            selection,
            cache_dir=cache_dir,
            earthdata_token=token,
        )
    except (OSError, ValueError) as error:
        _fail(source_id, selection, download_failure_phase(error), error)

    try:
        observed = validate_catalog_raster(
            downloaded,
            source,
            selected.year,
            require_year_marker=False,
        )
    except (OSError, ValueError) as error:
        _fail(source_id, selection, "raster validation", error)

    entry = cache_entry(cache_dir, source, selected.year)
    receipt = verified_receipt(downloaded, entry.receipt, selection=selection)
    if receipt is None:
        pytest.fail(f"{source_id} ({selection}) receipt validation failed.")
    assert downloaded == entry.raster, (
        f"{source_id} ({selection}) cache path was unexpected."
    )
    assert receipt["official_url"] == selected.official_url
    assert receipt["local_size_bytes"] == downloaded.stat().st_size
    assert receipt["observed"] == _json_safe(observed)

    if source_id == GPW:
        _assert_gpw_parity(downloaded, selection, token, tmp_path)

    monkeypatch.setattr(
        _api,
        "download_file",
        lambda *args, **kwargs: pytest.fail(
            f"{source_id} ({selection}) cache reuse attempted a network download."
        ),
    )
    try:
        cached = populations.download(selection, cache_dir=cache_dir, offline=True)
    except (OSError, ValueError) as error:
        _fail(source_id, selection, "offline cache reuse", error)
    assert cached == downloaded, (
        f"{source_id} ({selection}) offline cache reuse changed paths."
    )


def _assert_gpw_parity(
    fine_path: Path,
    selection: str,
    earthdata_token: str | None,
    temporary_directory: Path,
) -> None:
    """Check official 30-arc-second counts against CIESIN's one-degree counts."""
    if earthdata_token is None:  # pragma: no cover - guarded by the live test
        raise ValueError("GPW parity requires an Earthdata token.")
    oracle = gpw_coarse_oracle(selection)
    archive_path = temporary_directory / "gpw-coarse.zip"
    coarse_path = temporary_directory / oracle.archive_member
    try:
        download_file(
            oracle.official_url,
            archive_path,
            headers=_api._authentication_headers(SOURCES[GPW], earthdata_token),
            max_bytes=10_000_000,
            exact_bytes=None,
            publisher_checksum=None,
        )
        _api._extract_member(archive_path, coarse_path, oracle.archive_member)
    except (OSError, ValueError) as error:
        _fail(GPW, selection, "official CIESIN coarse-oracle acquisition", error)

    try:
        with rasterio.open(fine_path) as fine, rasterio.open(coarse_path) as coarse:
            parity = compare_gpw_fine_to_coarse(fine, coarse)
    except (OSError, ValueError) as error:
        _fail(GPW, selection, "official CIESIN coarse-oracle alignment", error)

    assert parity.compared_cells > 0, f"{GPW} ({selection}) had no comparable cells."
    assert parity.maximum_tolerance_normalized_difference <= 1.0, (
        f"{GPW} ({selection}) fine counts exceeded CIESIN's float32 precision "
        f"allowance: compared cells={parity.compared_cells}, "
        f"worst absolute difference={parity.maximum_absolute_difference:.12g} people, "
        f"maximum tolerance={parity.maximum_tolerance:.12g} people, "
        f"maximum tolerance-normalized difference="
        f"{parity.maximum_tolerance_normalized_difference:.12g}, "
        f"maximum ULP-normalized difference="
        f"{parity.maximum_ulp_normalized_difference:.12g}."
    )
    assert parity.aggregate_difference <= parity.aggregate_tolerance, (
        f"{GPW} ({selection}) did not conserve the global population total: "
        f"compared cells={parity.compared_cells}, "
        f"aggregate difference={parity.aggregate_difference:.12g} people, "
        f"aggregate tolerance={parity.aggregate_tolerance:.12g} people."
    )
