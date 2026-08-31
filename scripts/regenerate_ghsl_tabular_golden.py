"""Regenerate the offline GHSL Aruba tabular golden fixture.

This maintainer-only script downloads about 520 MB on a cold cache. It
deliberately records that the global 30-arc-second grid is not the same as
GHS-COUNTRY-STATS: the latter splits settlement clusters at GADM 4.1 country
borders before classifying them.

Example:
    uv run --group test python scripts/regenerate_ghsl_tabular_golden.py \
        --accept-download \
        tests/data/ghsl_aruba_tabular
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import numpy as np
import rasterio
from defusedxml import ElementTree  # deptry: ignore[DEP004]
from platformdirs import user_cache_path
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

if TYPE_CHECKING:
    from collections.abc import Iterator

SMOD_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss/V2-0/"
    "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip"
)
POPULATION_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/"
    "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip"
)
COUNTRY_STATS_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_COUNTRY_STATS_MT_GLOBE_R2024A/V1-0/"
    "GHS_COUNTRY_STATS_MT_GLOBE_R2024A.zip"
)
GADM_ARUBA_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ABW_0.json"
SMOD_MEMBER = "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif"
POPULATION_MEMBER = "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
WORKBOOK_MEMBER = "GHS-COUNTRY-STATS_MT_GLOBE_R2024_V1_0.xlsx"
COUNTRY_STATS_ARCHIVE = "GHS_COUNTRY_STATS_MT_GLOBE_R2024A.zip"
CACHE_DIRECTORY_NAME = "ghsl-tabular-golden"
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_BLOCK_BYTES = 1024 * 1024
GRID_TOLERANCE = 1e-12
WORKBOOK_COLUMNS = (
    "GADM_ID",
    "GADM_ISO",
    "GADM_NAME",
    "DEGURBA_L1",
    "1975",
    "1980",
    "1985",
    "1990",
    "1995",
    "2000",
    "2005",
    "2010",
    "2015",
    "2020",
    "2025",
    "2030",
)
WORKBOOK_COLUMN_LETTERS = tuple(
    chr(ord("A") + index) for index in range(len(WORKBOOK_COLUMNS))
)
L1_CODES = {
    "UC": (30,),
    "UCL": (21, 22, 23),
    "RUR": (11, 12, 13),
}
EXCLUDED_SMOD_CLASSES = frozenset({-200, 10})


@dataclass(frozen=True, slots=True)
class Source:
    """Describe one checksum-pinned source download.

    Args:
        filename: Cache filename.
        url: HTTPS location provided by the source publisher.
        sha256: Expected SHA-256 digest for the complete response.
        bytes: Expected response size in bytes.

    Returns:
        An immutable source description.

    Examples:
        >>> Source("source.zip", "https://example.test/source.zip", "a" * 64, 1)
        Source(filename='source.zip', url='https://example.test/source.zip', sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', bytes=1)
    """

    filename: str
    url: str
    sha256: str
    bytes: int


SOURCES = (
    Source(
        "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip",
        SMOD_URL,
        "7cf02eb4d0c08e5f987dc899dfcf26be4d77e21aed72a3358cef2ded2d0f75e2",  # pragma: allowlist secret
        33775811,
    ),
    Source(
        "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip",
        POPULATION_URL,
        "579fb7477b33d9be61e9562b170ea108a670b85ef6fe23b61a22d17200929636",  # pragma: allowlist secret
        482351880,
    ),
    Source(
        "GHS_COUNTRY_STATS_MT_GLOBE_R2024A.zip",
        COUNTRY_STATS_URL,
        "bdc5069a6cf9c5fe001d6b12c7d6b1dc1fccf9d063db1406d90a6507cce8a4be",  # pragma: allowlist secret
        1546681,
    ),
    Source(
        "gadm41_ABW_0.json",
        GADM_ARUBA_URL,
        "f9532884df370f11b75d8a5652b90644f9ef6c644623759c9bbcae9768c41c6a",  # pragma: allowlist secret
        2495,
    ),
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Args:
        path: Existing file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Examples:
        >>> sha256(Path("source.bin"))
        '...'
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(DOWNLOAD_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_source(source: Source, cache_directory: Path) -> Path:
    """Return one verified source, downloading it into the shared cache if needed.

    Args:
        source: Pinned source identity and expected byte count.
        cache_directory: Shared directory used across worktrees.

    Returns:
        Path to the verified complete source file.

    Raises:
        ValueError: If the URL is not HTTPS or a response exceeds its limit.

    Examples:
        >>> cached_source(SOURCES[0], Path("/tmp/cache"))  # doctest: +SKIP
        PosixPath('/tmp/cache/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip')
    """
    if urlsplit(source.url).scheme != "https":
        raise ValueError(f"Source URL must use HTTPS: {source.url}")
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination = cache_directory / source.filename
    if destination.is_file() and _matches_source(destination, source):
        return destination
    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(  # noqa: S310 - HTTPS scheme checked above.
        source.url,
        headers={"User-Agent": "population-exposure"},
    )
    try:
        with (
            urllib.request.urlopen(  # noqa: S310 - URL scheme is checked above.
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response,
            partial.open("xb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length is None or int(content_length) != source.bytes:
                raise ValueError(
                    f"Unexpected Content-Length for {source.url}: {content_length!r}."
                )
            copied = 0
            while block := response.read(DOWNLOAD_BLOCK_BYTES):
                copied += len(block)
                if copied > source.bytes:
                    raise ValueError(f"Source exceeded byte limit: {source.url}")
                output.write(block)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if not _matches_source(partial, source):
        partial.unlink(missing_ok=True)
        raise ValueError(f"Downloaded source failed validation: {source.url}")
    partial.replace(destination)
    return destination


def _matches_source(path: Path, source: Source) -> bool:
    """Return whether a file matches its exact expected size and checksum.

    Args:
        path: Candidate cached file.
        source: Source identity against which to check the file.

    Returns:
        True only when the file size and SHA-256 both match.

    Examples:
        >>> _matches_source(Path("missing.bin"), SOURCES[0])
        False
    """
    return (
        path.is_file()
        and path.stat().st_size == source.bytes
        and sha256(path) == source.sha256
    )


def extract_member(archive_path: Path, member: str, destination: Path) -> Path:
    """Safely stream one named archive member to a destination path.

    Args:
        archive_path: Verified ZIP archive containing the source member.
        member: Exact, relative POSIX archive member name to extract.
        destination: New local path to receive the member's bytes.

    Returns:
        Destination path after extraction.

    Raises:
        ValueError: If the member name is unsafe or absent.

    Examples:
        >>> extract_member(
        ...     Path("source.zip"), "grid.tif", Path("grid.tif")
        ... )  # doctest: +SKIP
        PosixPath('grid.tif')
    """
    _safe_relative_path(member, description="Archive member")
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            raise ValueError(f"Archive lacks required member: {member}") from error
        if info.is_dir() or info.file_size <= 0:
            raise ValueError(f"Archive member is not a non-empty file: {member}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=DOWNLOAD_BLOCK_BYTES)
    return destination


def _safe_relative_path(value: str, *, description: str) -> PurePosixPath:
    """Return a normalized relative path after rejecting unsafe components.

    Args:
        value: Path value using either slash style.
        description: Human-readable source of the path for error messages.

    Returns:
        Normalized, relative POSIX path.

    Raises:
        ValueError: If the value is absolute, traverses a parent, or has a drive
            prefix.

    Examples:
        >>> _safe_relative_path(r"folder\\grid.tif", description="member")
        PurePosixPath('folder/grid.tif')
    """
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"{description} has an unsafe path: {value!r}")
    return path


def workbook_totals(archive_path: Path) -> dict[str, dict[str, float]]:
    """Read the POP_L1 2020 totals from the official workbook without Excel.

    Args:
        archive_path: Verified COUNTRY-STATS ZIP archive.

    Returns:
        Mapping from GADM ISO code to its UC, UCL, and RUR population totals.

    Raises:
        ValueError: If the workbook or its POP_L1 schema is unexpected.

    Examples:
        >>> workbook_totals(Path("country-stats.zip"))  # doctest: +SKIP
        {'ABW': {'UC': 56903.19754754787, 'UCL': 45177.75497597072, 'RUR': 4504.047416000278}}
    """
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relation = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            workbook_info = archive.getinfo(WORKBOOK_MEMBER)
        except KeyError as error:
            raise ValueError(
                f"Archive lacks required member: {WORKBOOK_MEMBER}"
            ) from error
        if workbook_info.file_size > 1_000_000:
            raise ValueError("Workbook exceeds its expected extraction limit.")
        workbook_bytes = archive.read(workbook_info)
    with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", namespace)
        try:
            pop_l1 = next(sheet for sheet in sheets if sheet.attrib["name"] == "POP_L1")
        except StopIteration as error:
            raise ValueError("Workbook does not contain POP_L1.") from error
        relationship_id = pop_l1.attrib[f"{{{relation}}}id"]
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        target = next(
            (
                item.attrib["Target"]
                for item in relationships
                if item.attrib["Id"] == relationship_id
            ),
            None,
        )
        if target is None:
            raise ValueError("Workbook POP_L1 relationship is absent.")
        target_path = _safe_relative_path(
            target,
            description="Workbook POP_L1 relationship",
        )
        strings = _shared_strings(archive, namespace)
        sheet = ElementTree.fromstring(archive.read(f"xl/{target_path}"))
    rows = list(_worksheet_rows(sheet, strings, namespace))
    if not rows or _workbook_header(rows[0]) != WORKBOOK_COLUMNS:
        raise ValueError("Workbook POP_L1 columns changed.")
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows[1:]:
        category = row.get("D")
        if category in L1_CODES:
            totals[row["B"]][category] += float(row["N"])
    return {country: dict(values) for country, values in totals.items()}


def _workbook_header(row: dict[str, str]) -> tuple[str, ...]:
    """Return the POP_L1 header in canonical spreadsheet column order."""
    return tuple(row.get(column, "") for column in WORKBOOK_COLUMN_LETTERS)


def _shared_strings(archive: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    """Read an XLSX shared-string table.

    Args:
        archive: Open workbook archive.
        namespace: Spreadsheet XML namespace mapping.

    Returns:
        Decoded shared-string values by zero-based index.

    Examples:
        >>> _shared_strings  # doctest: +ELLIPSIS
        <function _shared_strings at ...>
    """
    shared = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(item.itertext()) for item in shared.findall("m:si", namespace)]


def _worksheet_rows(
    sheet: ElementTree.Element,
    strings: list[str],
    namespace: dict[str, str],
) -> Iterator[dict[str, str]]:
    """Yield worksheet rows keyed by their spreadsheet column letters.

    Args:
        sheet: Parsed worksheet XML root.
        strings: Shared-string values used by the worksheet.
        namespace: Spreadsheet XML namespace mapping.

    Yields:
        One mapping of column letters to decoded cell values per row.

    Examples:
        >>> list(_worksheet_rows(ElementTree.Element("sheet"), [], {}))
        []
    """
    for row in sheet.findall(".//m:sheetData/m:row", namespace):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", namespace):
            value = cell.findtext("m:v", default="", namespaces=namespace)
            if cell.attrib.get("t") == "s":
                value = strings[int(value)]
            column = "".join(
                character for character in cell.attrib["r"] if character.isalpha()
            )
            values[column] = value
        yield values


def global_totals(smod_path: Path, population_path: Path) -> dict[str, float]:
    """Sum the full WGS84 grids by the documented SMOD L1 mapping.

    Args:
        smod_path: Extracted GHS-SMOD 2020 WGS84 30-arc-second raster.
        population_path: Extracted GHS-POP 2020 WGS84 30-arc-second raster.

    Returns:
        UC, UCL, and RUR totals from the global SMOD classification.

    Raises:
        ValueError: If the grids are not the expected aligned source grid.

    Examples:
        >>> global_totals(Path("smod.tif"), Path("population.tif"))  # doctest: +SKIP
        {'UC': ..., 'UCL': ..., 'RUR': ...}
    """
    totals: dict[str, float] = defaultdict(float)
    known_codes = set(L1_CODES["UC"] + L1_CODES["UCL"] + L1_CODES["RUR"])
    with rasterio.open(smod_path) as smod, rasterio.open(population_path) as population:
        _validate_grids(smod, population)
        for _, window in smod.block_windows(1):
            classes = smod.read(1, window=window)
            values = population.read(1, window=window)
            if not np.isfinite(values).all():
                raise ValueError("GHS-POP contains non-finite values.")
            unexpected_codes = set(np.unique(classes)) - known_codes - {-200, 0, 10}
            if unexpected_codes:
                raise ValueError(
                    f"SMOD contains unexpected class values: {unexpected_codes}"
                )
            if np.any(values[classes == 0] != 0):
                raise ValueError("Unclassified SMOD cells have non-zero population.")
            for category, codes in L1_CODES.items():
                totals[category] += float(
                    values[np.isin(classes, codes)].sum(dtype=np.float64)
                )
    return dict(totals)


def build_fixture(
    output_directory: Path,
    sources: dict[str, Path],
) -> None:
    """Create the compact Aruba cell tables and measured source metadata.

    Args:
        output_directory: New directory to receive fixture files.
        sources: Mapping of source filename to verified local path.

    Returns:
        None. Writes CSV and JSON files to output_directory.

    Raises:
        ValueError: If source metadata, GADM selection, or grid values differ.

    Examples:
        >>> build_fixture(Path("fixture"), {})  # doctest: +SKIP
    """
    workbook = workbook_totals(sources[COUNTRY_STATS_ARCHIVE])
    if workbook["ABW"] != {
        "UC": 56903.19754754787,
        "UCL": 45177.75497597072,
        "RUR": 4504.047416000278,
    }:
        raise ValueError("Official Aruba POP_L1 values changed.")
    gadm = json.loads(sources["gadm41_ABW_0.json"].read_text(encoding="utf-8"))
    features = gadm.get("features")
    if not isinstance(features, list) or len(features) != 1:
        raise ValueError("GADM Aruba source must contain exactly one feature.")
    feature = features[0]
    if feature["properties"] != {"GID_0": "ABW", "COUNTRY": "Aruba"}:
        raise ValueError("GADM source is not Aruba.")
    geometry = feature["geometry"]
    output_directory.mkdir(parents=True, exist_ok=False)
    cells_path = output_directory / "cells.csv"
    smod_archive = sources["GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip"]
    population_archive = sources["GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip"]
    with tempfile.TemporaryDirectory(
        prefix="population-exposure-ghsl-", dir=smod_archive.parent
    ) as temporary:
        temporary_path = Path(temporary)
        smod_path = extract_member(
            smod_archive, SMOD_MEMBER, temporary_path / "smod.tif"
        )
        population_path = extract_member(
            population_archive, POPULATION_MEMBER, temporary_path / "population.tif"
        )
        direct_global = global_totals(smod_path, population_path)
        rows, fixture_totals, grid = _aruba_rows(smod_path, population_path, geometry)
    _write_rows(cells_path, rows)
    workbook_global = {
        category: sum(country[category] for country in workbook.values())
        for category in L1_CODES
    }
    metadata = {
        "fixture": "GHSL R2023A 2020 Aruba exact-coordinate tabular cell join",
        "fixture_files": {"cells.csv": sha256(cells_path)},
        "grid": grid,
        "methodology": {
            "cell_selection": (
                "Centers of WGS84 30 arc-second cells within GADM 4.1 Aruba, "
                "using rasterio.geometry_mask(all_touched=False)."
            ),
            "coordinate_definition": (
                "longitude and latitude are SMOD pixel centers serialized with "
                "Python repr(float); the same canonical values key both tables."
            ),
            "l1_mapping": {
                "UC": [30],
                "UCL": [21, 22, 23],
                "RUR": [11, 12, 13],
                "excluded": [0, 10, -200],
            },
            "scope": (
                "SMOD is a settlement classification, not a hazard. This checks "
                "the public tabular cell-join method."
            ),
            "workbook_difference": (
                "GHS-COUNTRY-STATS creates clusters separately within GADM 4.1 "
                "country boundaries; it is not a direct sum of global SMOD cells."
            ),
        },
        "reproduced": {
            "aruba_global_smod": fixture_totals,
            "global_smod": direct_global,
        },
        "workbook": {"aruba": workbook["ABW"], "global": workbook_global},
        "differences": {
            "aruba_global_smod_minus_workbook": {
                category: fixture_totals[category] - workbook["ABW"][category]
                for category in L1_CODES
            },
            "global_smod_minus_workbook": {
                category: direct_global[category] - workbook_global[category]
                for category in L1_CODES
            },
        },
        "sources": {
            source.filename: {
                "bytes": source.bytes,
                "sha256": source.sha256,
                "url": source.url,
            }
            for source in SOURCES
        },
        "citations": {
            "country_stats_methodology": COUNTRY_STATS_URL,
            "ghsl_data_package_2023": (
                "https://human-settlement.emergency.copernicus.eu/"
                "documents/GHSL_Data_Package_2023.pdf"
            ),
            "smod_product": (
                "https://human-settlement.emergency.copernicus.eu/data/"
                "ghs_smod2023.json"
            ),
        },
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aruba_rows(
    smod_path: Path,
    population_path: Path,
    geometry: dict[str, object],
) -> tuple[list[dict[str, str]], dict[str, float], dict[str, object]]:
    """Return canonical table rows and totals for the GADM Aruba selection.

    Args:
        smod_path: Extracted SMOD source raster.
        population_path: Extracted population source raster.
        geometry: One GADM 4.1 Aruba GeoJSON geometry.

    Returns:
        Rows for cells.csv, category totals, and validated grid details.

    Raises:
        ValueError: If a selected cell has an unsupported source class.

    Examples:
        >>> _aruba_rows(Path("smod.tif"), Path("population.tif"), {})  # doctest: +SKIP
        ([], {}, {})
    """
    code_categories = {
        code: category for category, codes in L1_CODES.items() for code in codes
    }
    with rasterio.open(smod_path) as smod, rasterio.open(population_path) as population:
        _validate_grids(smod, population)
        bounds = rasterio.coords.BoundingBox(*geometry_bounds(geometry))
        window = (
            from_bounds(*bounds, transform=smod.transform)
            .round_offsets()
            .round_lengths()
            .intersection(Window(0, 0, smod.width, smod.height))
        )
        classes = smod.read(1, window=window)
        values = population.read(1, window=window)
        transform = smod.window_transform(window)
        included = geometry_mask(
            [geometry],
            out_shape=classes.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )
        rows: list[dict[str, str]] = []
        totals: dict[str, float] = defaultdict(float)
        for row, column in np.argwhere(included):
            value = float(values[row, column])
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "Aruba GHS-POP values must be finite and non-negative."
                )
            code = int(classes[row, column])
            category = _category_for_smod(code, value, code_categories)
            if category is None:
                continue
            longitude, latitude = transform * (int(column) + 0.5, int(row) + 0.5)
            rows.append(
                {
                    "longitude": repr(longitude),
                    "latitude": repr(latitude),
                    "smod_class": str(code),
                    "degurba_l1": category,
                    "population": repr(value),
                }
            )
            totals[category] += value
        grid = {
            "crs": smod.crs.to_string(),
            "height": smod.height,
            "width": smod.width,
            "transform": list(smod.transform)[:6],
            "selected_cells": len(rows),
        }
    return rows, dict(totals), grid


def _category_for_smod(
    code: int, value: float, code_categories: dict[int, str]
) -> str | None:
    """Return the fixture category, skip excluded classes, or reject unknowns."""
    category = code_categories.get(code)
    if category is not None:
        return category
    if code == 0:
        if value != 0:
            raise ValueError("Unclassified Aruba SMOD cells have non-zero population.")
        return None
    if code in EXCLUDED_SMOD_CLASSES:
        return None
    raise ValueError(f"Unexpected SMOD class in Aruba: {code}")


def geometry_bounds(geometry: dict[str, object]) -> tuple[float, float, float, float]:
    """Return the bounds of a GeoJSON Polygon or MultiPolygon geometry.

    Args:
        geometry: GeoJSON geometry containing numeric coordinate pairs.

    Returns:
        West, south, east, and north coordinate bounds.

    Raises:
        ValueError: If the geometry is not a Polygon or MultiPolygon.

    Examples:
        >>> geometry_bounds({"type": "Polygon", "coordinates": [[[1, 2], [3, 4]]]})
        (1.0, 2.0, 3.0, 4.0)
    """
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError(f"Unsupported GADM geometry type: {geometry_type!r}")
    pairs = [pair for polygon in polygons for ring in polygon for pair in ring]
    if not pairs:
        raise ValueError("GADM geometry has no coordinates.")
    longitudes = [float(pair[0]) for pair in pairs]
    latitudes = [float(pair[1]) for pair in pairs]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def _validate_grids(
    smod: rasterio.io.DatasetReader, population: rasterio.io.DatasetReader
) -> None:
    """Require the two source rasters to represent the same nominal 30ss grid.

    Args:
        smod: Open SMOD raster.
        population: Open population raster.

    Returns:
        None.

    Raises:
        ValueError: If source dimensions, CRS, values, or transform disagree.

    Examples:
        >>> _validate_grids  # doctest: +ELLIPSIS
        <function _validate_grids at ...>
    """
    if (
        (smod.width, smod.height) != (43202, 21384)
        or (population.width, population.height) != (43202, 21384)
        or smod.crs.to_string() != "EPSG:4326"
        or population.crs.to_string() != "EPSG:4326"
        or smod.nodata != -200
        or population.nodata is not None
        or not np.allclose(
            tuple(smod.transform)[:6],
            tuple(population.transform)[:6],
            rtol=0,
            atol=GRID_TOLERANCE,
        )
    ):
        raise ValueError("GHSL sources are not the expected aligned WGS84 30ss grid.")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write deterministic fixture rows in source raster order.

    Args:
        path: New CSV file to create.
        rows: Canonical selected-cell records.

    Returns:
        None.

    Examples:
        >>> _write_rows(Path("cells.csv"), [])  # doctest: +SKIP
    """
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "longitude",
                "latitude",
                "smod_class",
                "degurba_l1",
                "population",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run a checked GHSL fixture regeneration after explicit acknowledgement.

    Args:
        None.

    Returns:
        None. Downloads verified sources and creates the requested fixture.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-download",
        action="store_true",
        help="Confirm the approximately 520 MB download on a cold shared cache.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="New directory that will receive the generated fixture.",
    )
    arguments = parser.parse_args()
    if not arguments.accept_download:
        parser.error(
            "--accept-download is required because this downloads about 520 MB."
        )
    if arguments.output_directory.exists():
        parser.error(f"Output directory already exists: {arguments.output_directory}")
    cache_directory = (
        user_cache_path("population-exposure", appauthor=False) / CACHE_DIRECTORY_NAME
    )
    paths = {
        source.filename: cached_source(source, cache_directory) for source in SOURCES
    }
    build_fixture(arguments.output_directory, paths)
    print(f"Wrote GHSL tabular golden fixture to {arguments.output_directory}")


if __name__ == "__main__":
    main()
