"""Platform cache layout and adjacent JSON receipts."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_cache_path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from population_exposure.populations._models import SelectionInfo
    from population_exposure.populations._sources import SourceSpec

CACHE_ENV = "POPULATION_EXPOSURE_CACHE_DIR"
OFFLINE_ENV = "POPULATION_EXPOSURE_OFFLINE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """All paths used to install one selected annual raster."""

    directory: Path
    raster: Path
    receipt: Path
    partial: Path
    lock: Path


def cache_root(override: str | os.PathLike[str] | None) -> Path:
    """Return an explicit, environment, or platform-appropriate cache root."""
    if override is not None:
        return Path(override).expanduser()
    configured = os.environ.get(CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return user_cache_path("population-exposure", appauthor=False)


def cache_entry(root: Path, source: SourceSpec, year: int) -> CacheEntry:
    """Return the grouped source/release/year cache paths."""
    directory = source_cache_directory(root, source) / str(year)
    raster = directory / source.filename_template.format(year=year)
    return CacheEntry(
        directory=directory,
        raster=raster,
        receipt=raster.with_suffix(f"{raster.suffix}.json"),
        partial=directory / f"{raster.name}.partial",
        lock=directory / ".lock",
    )


def source_cache_directory(root: Path, source: SourceSpec) -> Path:
    """Return the directory shared by every year of one source release."""
    release = re.sub(r"[^a-z0-9]+", "-", source.release.lower()).strip("-")
    return root / source.source_id / release


def offline_enabled(value: bool | None) -> bool:
    """Resolve an explicit flag or strict environment setting."""
    if value is not None:
        if not isinstance(value, bool):
            raise TypeError("offline must be a boolean or None.")
        return value
    configured = os.environ.get(OFFLINE_ENV, "").strip().lower()
    if configured in _TRUTHY:
        return True
    if configured in _FALSY:
        return False
    raise ValueError(
        f"{OFFLINE_ENV} must be one of: 1, 0, true, false, yes, no, on, off."
    )


def verified_receipt(
    raster: Path,
    receipt_path: Path,
    *,
    selection: str | None,
) -> dict[str, object] | None:
    """Return receipt data only when it still identifies unchanged local bytes."""
    if not raster.is_file() or not receipt_path.is_file():
        return None
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        return None
    typed_data: dict[str, object] = dict(data)
    if selection is not None and typed_data.get("selection") != selection:
        return None
    sha256 = typed_data.get("local_sha256")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        return None
    try:
        stat = raster.stat()
    except OSError:  # pragma: no cover - file removed between checks
        return None
    if typed_data.get("local_size_bytes") != stat.st_size:
        return None
    if typed_data.get("_verified_mtime_ns") != stat.st_mtime_ns:
        return None
    if typed_data.get("_verified_ctime_ns") != stat.st_ctime_ns:
        return None
    if typed_data.get("_verified_device") != stat.st_dev:
        return None
    if typed_data.get("_verified_inode") != stat.st_ino:
        return None
    return typed_data


def write_receipt(
    raster: Path,
    receipt_path: Path,
    *,
    info: SelectionInfo,
    sha256: str,
    observed: Mapping[str, object],
    processing_note: str,
    source_file_sha256: str | None = None,
) -> dict[str, object]:
    """Write one standards-compliant receipt beside a validated raster."""
    stat = raster.stat()
    data: dict[str, object] = {
        "selection": info.selection,
        "source_id": info.source_id,
        "release": info.release,
        "year": info.year,
        "publisher": info.publisher,
        "official_url": info.official_url,
        "landing_page": info.landing_page,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "local_path": str(raster.resolve()),
        "local_sha256": sha256,
        "local_size_bytes": stat.st_size,
        "_verified_mtime_ns": stat.st_mtime_ns,
        "_verified_ctime_ns": stat.st_ctime_ns,
        "_verified_device": stat.st_dev,
        "_verified_inode": stat.st_ino,
        "observed": dict(observed),
        "units": info.units,
        "population_meaning": info.meaning,
        "license": info.license,
        "citation": info.citation,
        "doi": info.doi,
        "processing_note": processing_note,
    }
    if source_file_sha256 is not None:
        data["source_file_sha256"] = source_file_sha256
    safe_value = _json_safe(data)
    if not isinstance(safe_value, dict):  # pragma: no cover
        raise TypeError("Receipt data must be a dictionary.")
    safe_data: dict[str, object] = dict(safe_value)
    temporary = receipt_path.with_suffix(f"{receipt_path.suffix}.partial")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(safe_data, output, allow_nan=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(receipt_path)
    return safe_data


def assignment_metadata(receipt: Mapping[str, object]) -> dict[str, object]:
    """Return receipt facts intended for assignment result attrs."""
    keys = (
        "selection",
        "source_id",
        "release",
        "year",
        "doi",
        "citation",
        "license",
        "units",
        "population_meaning",
        "local_path",
        "local_sha256",
        "processing_note",
        "observed",
    )
    return {key: receipt[key] for key in keys if key in receipt}


def _json_safe(value: object) -> object:
    """Convert nested values to strict JSON-compatible values."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
