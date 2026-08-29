"""Public population catalog API and assignment integration helpers."""

from __future__ import annotations

import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from zipfile import BadZipFile, ZipFile

from filelock import FileLock, Timeout
from rasterio.io import DatasetReader

from population_exposure.populations._cache import (
    assignment_metadata,
    cache_entry,
    cache_root,
    offline_enabled,
    source_cache_directory,
    verified_receipt,
    write_receipt,
)
from population_exposure.populations._chambers import derive_chambers_year
from population_exposure.populations._http import download_file, sha256_file
from population_exposure.populations._raster import (
    observed_raster_facts,
    validate_catalog_raster,
)
from population_exposure.populations._selection import (
    looks_like_selection,
    parse_selection,
)
from population_exposure.populations._sources import SOURCES

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from population_exposure.populations._models import SelectionInfo, SourceInfo
    from population_exposure.populations._sources import SourceSpec
    from population_exposure.raster import RasterSource

_EARTHDATA_TOKEN_ENV = "EARTHDATA_TOKEN"  # noqa: S105
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_REGISTERED_BYTES = 10_000_000_000
_MAX_EXTRACTED_BYTES = 10_000_000_000
_LOCK_TIMEOUT_SECONDS = 300
_CHAMBERS_FILENAME = "demographics_hybrid_1950_2020_15_min.nc"


@dataclass(frozen=True, slots=True)
class ResolvedPopulation:
    """A local or caller-owned raster plus source facts for result attrs."""

    source: RasterSource
    metadata: Mapping[str, object]


def list() -> tuple[SourceInfo, ...]:
    """Return the five built-in population sources in stable order."""
    return tuple(source.source_info() for source in SOURCES.values())


def info(selection: str) -> SelectionInfo:
    """Return license, citation, size, and acquisition facts without downloading."""
    source, year = parse_selection(selection)
    return source.selection_info(year)


def download(
    selection: str,
    *,
    cache_dir: str | PathLike[str] | None = None,
    refresh: bool = False,
    offline: bool | None = None,
    earthdata_token: str | None = None,
) -> Path:
    """Return a verified cached raster for one explicit catalog selection."""
    if not isinstance(refresh, bool):
        raise TypeError("refresh must be a boolean.")
    source, year = parse_selection(selection)
    selected = source.selection_info(year)
    entry = cache_entry(cache_root(cache_dir), source, year)
    is_offline = offline_enabled(offline)

    cached = verified_receipt(entry.raster, entry.receipt, selection=selection)
    if cached is not None and not refresh:
        return entry.raster
    if is_offline and (source.delivery != "chambers" or refresh):
        raise ValueError(_offline_message(selected))

    entry.directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(entry.lock, timeout=_LOCK_TIMEOUT_SECONDS):
            cached = verified_receipt(
                entry.raster,
                entry.receipt,
                selection=selection,
            )
            if cached is not None and not refresh:
                return entry.raster
            if source.delivery == "manual":
                if cached is not None:
                    raise ValueError(
                        f"{selection!r} is manually registered and cannot be "
                        "refreshed automatically. Register a newly acquired ORNL "
                        "file to replace it; the verified cached file was retained."
                    )
                raise ValueError(_manual_message(selected))
            if refresh:
                _remove_if_file(entry.partial)
                _remove_if_file(entry.partial.with_suffix(".zip.partial"))
            if source.delivery == "chambers":
                return _download_chambers(
                    source,
                    selected,
                    entry.raster,
                    entry.receipt,
                    entry.partial,
                    root=cache_root(cache_dir),
                    refresh=refresh,
                    offline=is_offline,
                    earthdata_token=earthdata_token,
                )
            return _download_raster(
                source,
                selected,
                entry.raster,
                entry.receipt,
                entry.partial,
                earthdata_token=earthdata_token,
            )
    except Timeout as error:
        raise ValueError(
            f"Timed out waiting for another process to finish {selection!r}."
        ) from error


def register(
    selection: str,
    path: str | PathLike[str],
    *,
    cache_dir: str | PathLike[str] | None = None,
) -> Path:
    """Copy, validate, and receipt a caller-owned annual population GeoTIFF."""
    source, year = parse_selection(selection)
    selected = source.selection_info(year)
    original = Path(path).expanduser()
    if not original.is_file():
        raise ValueError(
            f"Population file to register does not exist or is not a file: {original}."
        )
    entry = cache_entry(cache_root(cache_dir), source, year)
    entry.directory.mkdir(parents=True, exist_ok=True)

    try:
        with FileLock(entry.lock, timeout=_LOCK_TIMEOUT_SECONDS):
            validate_catalog_raster(
                original,
                source,
                year,
                require_year_marker=True,
            )
            same_file = _same_file(original, entry.raster)
            if not same_file:
                _remove_if_file(entry.partial)
                try:
                    sha256 = _copy_bounded(
                        original,
                        entry.partial,
                        max_bytes=_MAX_REGISTERED_BYTES,
                    )
                    observed = validate_catalog_raster(
                        entry.partial,
                        source,
                        year,
                        require_year_marker=False,
                    )
                except (OSError, ValueError):
                    _remove_if_file(entry.partial)
                    raise
                _install_with_receipt(
                    entry.partial,
                    entry.raster,
                    entry.receipt,
                    info=selected,
                    sha256=sha256,
                    observed=observed,
                    processing_note=(
                        "Copied from a caller-owned file after source, year, grid, "
                        "and population-count validation; the original file was not "
                        "modified."
                    ),
                )
            else:
                observed = validate_catalog_raster(
                    entry.raster,
                    source,
                    year,
                    require_year_marker=True,
                )
                sha256 = sha256_file(entry.raster)
                write_receipt(
                    entry.raster,
                    entry.receipt,
                    info=selected,
                    sha256=sha256,
                    observed=observed,
                    processing_note=(
                        "Copied from a caller-owned file after source, year, grid, "
                        "and population-count validation; the original file was not "
                        "modified."
                    ),
                )
            return entry.raster
    except Timeout as error:
        raise ValueError(
            f"Timed out waiting for another process to finish {selection!r}."
        ) from error


def resolve_for_assignment(population: RasterSource) -> ResolvedPopulation:
    """Resolve exact catalog strings while preserving paths and open readers."""
    if isinstance(population, DatasetReader):
        return ResolvedPopulation(population, MappingProxyType({}))
    if isinstance(population, os.PathLike):
        path = Path(population).expanduser()
        if not path.is_file():
            raise ValueError(
                f"population raster path does not exist or is not a file: {path}."
            )
        return ResolvedPopulation(path, _metadata_from_adjacent_receipt(path))
    if isinstance(population, str):
        path = Path(population).expanduser()
        if path.is_file():
            return ResolvedPopulation(path, _metadata_from_adjacent_receipt(path))
        if looks_like_selection(population):
            downloaded = download(population)
            return ResolvedPopulation(
                downloaded,
                _metadata_from_adjacent_receipt(downloaded),
            )
        raise ValueError(
            f"population raster path does not exist or is not a file: {path}."
        )
    raise TypeError("population must be a GeoTIFF path or open Rasterio DatasetReader.")


def metadata_for_reader(
    resolved: ResolvedPopulation,
    reader: DatasetReader,
    *,
    total: float,
) -> Mapping[str, object]:
    """Return catalog receipt facts or observed facts for a custom source."""
    if resolved.metadata:
        return dict(resolved.metadata)
    observed = observed_raster_facts(reader, total=total)
    metadata: dict[str, object] = {
        "source_id": "custom",
        "units": observed.get("units"),
        "population_meaning": "caller-supplied",
        "observed": observed,
        "processing_note": (
            "Caller-supplied population-count raster; no publisher license or "
            "citation was inferred."
        ),
    }
    path = _reader_local_path(resolved.source, reader)
    if path is not None:
        metadata["local_path"] = str(path.resolve())
        metadata["local_sha256"] = sha256_file(path)
    return metadata


def _download_raster(
    source: SourceSpec,
    selected: SelectionInfo,
    raster_path: Path,
    receipt_path: Path,
    partial_path: Path,
    *,
    earthdata_token: str | None,
) -> Path:
    """Download a direct GeoTIFF or one exact GeoTIFF from a publisher ZIP."""
    headers = _authentication_headers(source, earthdata_token)
    if source.delivery == "geotiff":
        download_file(
            selected.official_url,
            partial_path,
            headers=headers,
            max_bytes=source.max_download_bytes,
            exact_bytes=source.exact_download_bytes,
            publisher_checksum=source.publisher_checksum,
        )
        try:
            observed = validate_catalog_raster(
                partial_path,
                source,
                selected.year,
                require_year_marker=False,
            )
            sha256 = sha256_file(partial_path)
        except (OSError, ValueError):
            _remove_if_file(partial_path)
            raise
        processing = (
            "Downloaded anonymously from the official publisher route without "
            "transforming the GeoTIFF."
        )
    else:
        archive_path = partial_path.with_suffix(".zip.partial")
        download_file(
            selected.official_url,
            archive_path,
            headers=headers,
            max_bytes=source.max_download_bytes,
            exact_bytes=source.exact_download_bytes,
            publisher_checksum=source.publisher_checksum,
        )
        member = source.archive_member(selected.year)
        assert member is not None
        try:
            _extract_member(archive_path, partial_path, member)
            observed = validate_catalog_raster(
                partial_path,
                source,
                selected.year,
                require_year_marker=False,
            )
            sha256 = sha256_file(partial_path)
        except (BadZipFile, OSError, ValueError):
            _remove_if_file(partial_path)
            _remove_if_file(archive_path)
            raise
        _remove_if_file(archive_path)
        if source.acquisition == "earthdata":
            processing = (
                "Downloaded from the official Earthdata route with a transient "
                "user-owned token, then extracted the exact population-count "
                "GeoTIFF; the token was not stored."
            )
        else:
            processing = (
                "Downloaded the official publisher archive and extracted the exact "
                "population-count GeoTIFF without changing cell values."
            )

    _install_with_receipt(
        partial_path,
        raster_path,
        receipt_path,
        info=selected,
        sha256=sha256,
        observed=observed,
        processing_note=processing,
    )
    return raster_path


def _download_chambers(
    source: SourceSpec,
    selected: SelectionInfo,
    raster_path: Path,
    receipt_path: Path,
    partial_path: Path,
    *,
    root: Path,
    refresh: bool,
    offline: bool,
    earthdata_token: str | None,
) -> Path:
    """Reuse one verified multi-year source and derive one annual total."""
    shared = source_cache_directory(root, source) / "source"
    source_path = shared / _CHAMBERS_FILENAME
    source_receipt_path = source_path.with_suffix(f"{source_path.suffix}.json")
    source_partial = shared / f"{_CHAMBERS_FILENAME}.partial"
    source_lock = shared / ".lock"
    shared.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(source_lock, timeout=_LOCK_TIMEOUT_SECONDS):
            cached_source = verified_receipt(
                source_path,
                source_receipt_path,
                selection=None,
            )
            if refresh:
                _remove_if_file(source_partial)
                cached_source = None
            if cached_source is None:
                if offline:
                    raise ValueError(_offline_message(selected))
                result = download_file(
                    selected.official_url,
                    source_partial,
                    headers=_authentication_headers(source, earthdata_token),
                    max_bytes=source.max_download_bytes,
                    exact_bytes=source.exact_download_bytes,
                    publisher_checksum=source.publisher_checksum,
                )
                cached_source = _install_with_receipt(
                    source_partial,
                    source_path,
                    source_receipt_path,
                    info=selected,
                    sha256=result.sha256,
                    observed={
                        "format": "NetCDF-4",
                        "variable": "demographic_totals",
                        "year_coverage": "1950-2020",
                        "age_bands": 21,
                    },
                    processing_note=(
                        "Downloaded once from immutable Zenodo record 6011021 for "
                        "reuse across requested annual derivations."
                    ),
                )
    except Timeout as error:
        raise ValueError(
            "Timed out waiting for another process to finish the Chambers source."
        ) from error

    try:
        derive_chambers_year(source_path, partial_path, selected.year)
        observed = validate_catalog_raster(
            partial_path,
            source,
            selected.year,
            require_year_marker=True,
        )
        sha256 = sha256_file(partial_path)
    except (OSError, ValueError):
        _remove_if_file(partial_path)
        raise
    source_sha256 = cached_source.get("local_sha256")
    assert isinstance(source_sha256, str)
    _install_with_receipt(
        partial_path,
        raster_path,
        receipt_path,
        info=selected,
        sha256=sha256,
        observed=observed,
        source_file_sha256=source_sha256,
        processing_note=(
            "Derived from the one cached NetCDF-4 source by summing the selected "
            "year's 21 five-year age bands in bounded windows."
        ),
    )
    return raster_path


def _install_with_receipt(
    staged_path: Path,
    final_path: Path,
    receipt_path: Path,
    *,
    info: SelectionInfo,
    sha256: str,
    observed: Mapping[str, object],
    processing_note: str,
    source_file_sha256: str | None = None,
) -> dict[str, object]:
    """Install new bytes and roll back if their receipt cannot be persisted."""
    file_backup = final_path.with_name(f"{final_path.name}.backup")
    receipt_backup = receipt_path.with_name(f"{receipt_path.name}.backup")
    _remove_if_file(file_backup)
    _remove_if_file(receipt_backup)
    file_was_backed_up = False
    receipt_was_backed_up = False
    try:
        if final_path.is_file():
            final_path.replace(file_backup)
            file_was_backed_up = True
        if receipt_path.is_file():
            receipt_path.replace(receipt_backup)
            receipt_was_backed_up = True
        staged_path.replace(final_path)
        receipt = write_receipt(
            final_path,
            receipt_path,
            info=info,
            sha256=sha256,
            observed=observed,
            processing_note=processing_note,
            source_file_sha256=source_file_sha256,
        )
    except OSError:
        _remove_if_file(final_path)
        _remove_if_file(receipt_path)
        _remove_if_file(receipt_path.with_suffix(f"{receipt_path.suffix}.partial"))
        if file_was_backed_up:
            file_backup.replace(final_path)
        if receipt_was_backed_up:
            receipt_backup.replace(receipt_path)
        raise
    _remove_if_file(file_backup)
    _remove_if_file(receipt_backup)
    return receipt


def _authentication_headers(
    source: SourceSpec,
    earthdata_token: str | None,
) -> dict[str, str] | None:
    """Return a transient authorization header only for the Earthdata source."""
    if source.acquisition != "earthdata":
        return None
    token = earthdata_token or os.environ.get(_EARTHDATA_TOKEN_ENV)
    if token is None or not token.strip():
        raise ValueError(
            "GPWv4 requires your Earthdata token. Pass earthdata_token=... or set "
            "EARTHDATA_TOKEN, or download the official 30 arc-second population "
            "count GeoTIFF and call populations.register(). The token is used only "
            "for the request and is never stored or logged."
        )
    return {"Authorization": f"Bearer {token}"}


def _extract_member(archive_path: Path, output_path: Path, expected: str) -> None:
    """Copy one exact archive member without extracting arbitrary paths."""
    with ZipFile(archive_path) as archive:
        matches = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).name == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Official archive must contain exactly one {expected!r} GeoTIFF."
            )
        member = matches[0]
        if member.file_size > _MAX_EXTRACTED_BYTES:
            raise ValueError("Archived GeoTIFF exceeds the extraction safety limit.")
        with archive.open(member) as source, output_path.open("wb") as output:
            while chunk := source.read(_COPY_CHUNK_SIZE):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())


def _copy_bounded(source: Path, destination: Path, *, max_bytes: int) -> str:
    """Copy a local file without mutating it and enforce a size limit."""
    if source.stat().st_size > max_bytes:
        raise ValueError(
            f"Population file exceeds the {max_bytes}-byte registration safety limit."
        )
    with source.open("rb") as input_file, destination.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=_COPY_CHUNK_SIZE)
        output_file.flush()
        os.fsync(output_file.fileno())
    return sha256_file(destination)


def _metadata_from_adjacent_receipt(path: Path) -> Mapping[str, object]:
    """Load trusted catalog attrs only from a valid adjacent receipt."""
    receipt_path = path.with_suffix(f"{path.suffix}.json")
    receipt = verified_receipt(path, receipt_path, selection=None)
    if receipt is None:
        return MappingProxyType({})
    return MappingProxyType(assignment_metadata(receipt))


def _reader_local_path(
    source: RasterSource,
    reader: DatasetReader,
) -> Path | None:
    """Return a real local path when one can be observed."""
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
    else:
        path = Path(reader.name)
    return path if path.is_file() else None


def _same_file(left: Path, right: Path) -> bool:
    """Return whether two existing paths name the same file."""
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def _remove_if_file(path: Path) -> None:
    """Remove one known cache file when present."""
    with suppress(FileNotFoundError):
        path.unlink()


def _offline_message(selected: SelectionInfo) -> str:
    """Return an actionable no-network cache miss."""
    return (
        f"Offline mode made no network request, but {selected.selection!r} has no "
        "verified cached or registered raster. Run populations.download() online "
        "or populations.register() first."
    )


def _manual_message(selected: SelectionInfo) -> str:
    """Return the official LandScan manual acquisition instructions."""
    return (
        f"{selected.selection!r} requires manual acquisition from the official ORNL "
        f"LandScan portal at {selected.landing_page}. Complete ORNL registration and "
        "license acceptance, extract the selected year's GeoTIFF, then call "
        f"populations.register({selected.selection!r}, path). The package does not "
        "automate the form or use mirrors."
    )
