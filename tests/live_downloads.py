"""Shared configuration for opt-in real population-provider tests."""

from __future__ import annotations

import os

from population_exposure import populations

WORLDPOP = "worldpop-global-1km"
GHSL = "ghsl-r2023a-mollweide-1km"
GPW = "gpwv4-r11-count"
CHAMBERS = "chambers-hybrid"
LANDSCAN = "landscan-global"

SCHEDULED_PROVIDERS = (WORLDPOP, GHSL)
DOWNLOADABLE_PROVIDERS = (WORLDPOP, GHSL, GPW, CHAMBERS)
MANUAL_CHOICES = ("scheduled", *DOWNLOADABLE_PROVIDERS, "all")
_LIVE_PROVIDERS_ENV = "POPULATION_EXPOSURE_LIVE_PROVIDERS"


def providers_for_run(value: str | None = None) -> tuple[str, ...]:
    """Return the approved source IDs selected for one live test run."""
    choice = (
        value if value is not None else os.environ.get(_LIVE_PROVIDERS_ENV, "scheduled")
    )
    normalized = choice.strip().lower()
    if normalized == "scheduled":
        return SCHEDULED_PROVIDERS
    if normalized == "all":
        return DOWNLOADABLE_PROVIDERS
    if normalized in DOWNLOADABLE_PROVIDERS:
        return (normalized,)
    choices = ", ".join(MANUAL_CHOICES)
    raise ValueError(
        f"{_LIVE_PROVIDERS_ENV} must be one of: {choices}; got {choice!r}."
    )


def selection_for_provider(source_id: str) -> str:
    """Choose the latest catalog-supported year without copying provider URLs."""
    sources = {source.source_id: source for source in populations.list()}
    source = sources.get(source_id)
    if source is None or source_id not in DOWNLOADABLE_PROVIDERS:
        raise ValueError(f"{source_id!r} is not an approved live-download provider.")
    return f"{source.source_id}:{max(source.supported_years)}"


def download_failure_phase(error: Exception) -> str:
    """Classify package errors so an external-provider failure is actionable."""
    message = str(error).lower()
    if any(
        marker in message
        for marker in (
            "checksum",
            "content-length",
            "download size",
            "byte safety limit",
            "verified size",
            "expected exactly",
        )
    ):
        return "checksum/size verification"
    if any(marker in message for marker in ("archive", "zip", "extraction")):
        return "archive extraction"
    if any(
        marker in message
        for marker in (
            "population file is not a readable geotiff",
            "requires crs",
            "requires width",
            "requires height",
            "requires pixel size",
            "requires bounds",
            "requires nodata",
            "population total",
        )
    ):
        return "raster validation"
    return "acquisition"
