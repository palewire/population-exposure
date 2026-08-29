"""Opt-in tests that exercise complete official population-provider downloads."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from population_exposure import populations
from population_exposure.populations import _api
from population_exposure.populations._cache import cache_entry, verified_receipt
from population_exposure.populations._raster import validate_catalog_raster
from population_exposure.populations._sources import SOURCES
from tests.live_downloads import (
    GPW,
    download_failure_phase,
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
    assert receipt["observed"] == observed

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
