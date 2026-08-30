"""Regenerate the China 2019 heatwave exposure golden artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np
import pandas as pd
import rasterio
from platformdirs import user_cache_path
from rasterio.windows import from_bounds

from population_exposure import assign_population
from population_exposure.populations._http import download_file, sha256_file
from validation.china_heatwave_2019.method import (
    count_heatwave_days,
    heatwave_anomaly,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

BASELINE_YEARS = tuple(range(1986, 2006))
TARGET_YEAR = 2019
WARM_SEASON_MONTHS = tuple(range(5, 10))
PERCENTILE = 92.5
MINIMUM_RUN_DAYS = 3
CHINA_UN_M49 = 156
REFERENCE_MAINLAND_GRID_CELLS = 3_829
REFERENCE_MAINLAND_GRID_RELATIVE_TOLERANCE = 0.05
PUBLISHED_PERSON_DAYS = 2_200_000_000.0
PUBLISHED_PERSON_DAYS_TOLERANCE = 5_000_000.0
PUBLISHED_DAYS_PER_PERSON = 13.0
PUBLISHED_DAYS_PER_PERSON_TOLERANCE = 0.5
PUBLISHED_2000_PERSON_DAYS = 71_800_000.0
PUBLISHED_2000_PERSON_DAYS_TOLERANCE = 50_000.0

POPULATION_FILENAME = "demographics_1950_2020.nc"
POPULATION_URL = (
    f"https://zenodo.org/api/records/3768003/files/{POPULATION_FILENAME}/content"
)
POPULATION_SIZE = 1_660_395_967
POPULATION_MD5 = "5f24e8b2088ea0127f495ea364725df5"  # pragma: allowlist secret
POPULATION_SHA256 = "baa951326f2975d5d6dabfb2555ef6fcc86347137766c56f739344990fb3ad07"  # pragma: allowlist secret

COUNTRY_ARCHIVE_FILENAME = "gpw-v4-national-identifier-grid-rev11_30_sec_tif.zip"
COUNTRY_RASTER_FILENAME = "gpw_v4_national_identifier_grid_rev11_30_sec.tif"
COUNTRY_URL = (
    "https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/"
    "sedac-root/downloads/data/gpw-v4/gpw-v4-national-identifier-grid-rev11/"
    f"{COUNTRY_ARCHIVE_FILENAME}"
)
COUNTRY_ARCHIVE_SIZE = 12_548_937
COUNTRY_ARCHIVE_SHA256 = "878a19e79569bc81385af20c9b38837e66e3805d0b62d83d59298337a8cf1aa5"  # pragma: allowlist secret
COUNTRY_RASTER_SIZE = 38_236_733
COUNTRY_RASTER_SHA256 = "71294115eead3a45ac02514f96cc859766fc4113d03437ad0afcb7e7fff9f19f"  # pragma: allowlist secret
COUNTRY_RASTER_RESOLUTION = 1 / 120
COUNTRY_RASTER_NODATA = -32_768
ASSIGNMENT_GRID_RESOLUTION = 0.5
COUNTRY_CELLS_PER_ASSIGNMENT_SIDE = 60

ERA5_DATASET = "derived-era5-single-levels-daily-statistics"
ERA5_VARIABLE = "t2m"
ERA5_AREA = (55, 72, 17, 136)
ERA5_YEAR_BATCHES = tuple((year,) for year in (*BASELINE_YEARS, TARGET_YEAR))
ERA5_MAX_BATCH_BYTES = 100_000_000
ERA5_SOURCE_GRID_RESOLUTION = 0.25
ERA5_TEMPERATURE_RANGE_K = (150.0, 400.0)

GOLDEN_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "china_heatwave_2019"
)
GOLDEN_CELLS = GOLDEN_DIRECTORY / "cells.csv"
GOLDEN_METADATA = GOLDEN_DIRECTORY / "golden.json"


@dataclass(frozen=True, slots=True)
class CountryMaskAssignment:
    """Country assignment derived from aligned source-cell footprints.

    Args:
        values: Boolean assignment to mainland China for each target cell.
        source_cells_per_target_side: Native GPW cells along each target-cell side.
        tied_target_cells: Non-empty target cells excluded for tied country areas.
        china_tied_target_cells: Excluded ties where China shared the largest area.

    Returns:
        An immutable description of the aligned majority-area assignment.

    Examples:
        >>> assignment = CountryMaskAssignment(
        ...     values=np.array([[True]]),
        ...     source_cells_per_target_side=60,
        ...     tied_target_cells=0,
        ...     china_tied_target_cells=0,
        ... )
        >>> assignment.values.item()
        True
    """

    values: NDArray[np.bool_]
    source_cells_per_target_side: int
    tied_target_cells: int
    china_tied_target_cells: int


def _md5(path: Path) -> str:
    """Return the lowercase MD5 digest used by the Zenodo record.

    Args:
        path: File to hash.

    Returns:
        The file's lowercase hexadecimal MD5 digest.

    Examples:
        >>> len(_md5(Path(__file__)))
        32
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    bearer_token: str | None = None,
    authentication_required: bool = False,
) -> Path:
    """Download one immutable source file and verify its published identity.

    Args:
        url: HTTPS source URL.
        destination: Local cache destination.
        expected_size: Exact expected byte count.
        expected_sha256: Optional expected SHA-256 digest.
        expected_md5: Optional expected MD5 digest.
        bearer_token: Optional transient Earthdata token.
        authentication_required: Whether an empty bearer token is an error.

    Returns:
        The verified cache path.

    Raises:
        ValueError: If an existing or downloaded file fails verification.
        OSError: If the request or local file operation fails.

    Examples:
        This function is used with the immutable URLs declared in this module.
    """
    if destination.is_file():
        try:
            _verify_file(
                destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                expected_md5=expected_md5,
            )
        except ValueError:
            destination.unlink()
        else:
            return destination

    if authentication_required and (bearer_token is None or not bearer_token.strip()):
        raise ValueError(
            "The GPW country mask requires EARTHDATA_TOKEN in the environment."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.partial")
    headers = (
        {"Authorization": "Bearer " + bearer_token.strip()}
        if bearer_token is not None
        else None
    )
    publisher_checksum = (
        f"md5:{expected_md5}"
        if expected_md5 is not None
        else (f"sha256:{expected_sha256}" if expected_sha256 is not None else None)
    )
    try:
        result = download_file(
            url,
            partial,
            headers=headers,
            max_bytes=expected_size,
            exact_bytes=expected_size,
            publisher_checksum=publisher_checksum,
        )
        if expected_sha256 is not None and result.sha256 != expected_sha256:
            raise ValueError(f"{destination.name} failed SHA-256 verification.")
    except (OSError, ValueError):
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str | None,
    expected_md5: str | None,
) -> None:
    """Verify one source file against exact size and digest records.

    Args:
        path: File to verify.
        expected_size: Exact expected byte count.
        expected_sha256: Optional expected SHA-256 digest.
        expected_md5: Optional expected MD5 digest.

    Returns:
        None.

    Raises:
        ValueError: If size or either configured digest differs.

    Examples:
        This function verifies downloads before any source values are read.
    """
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path.name} must contain {expected_size} bytes; "
            f"found {path.stat().st_size}."
        )
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"{path.name} failed SHA-256 verification.")
    if expected_md5 is not None and _md5(path) != expected_md5:
        raise ValueError(f"{path.name} failed the publisher's MD5 verification.")


def _population_grid(
    source_path: Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Read the original source's terminal 65+ band for all required years.

    Args:
        source_path: Verified Chambers 2020 NetCDF file.

    Returns:
        Descending latitudes, ascending longitudes, and population counts
        ordered by the 20 baseline years followed by 2019.

    Raises:
        ValueError: If the published source dimensions have changed.

    Examples:
        The returned coordinates align exactly with the report's 0.5-degree
        ERA5 analysis grid.
    """
    import xarray as xr  # deptry: ignore[DEP004]

    with xr.open_dataset(source_path, engine="netcdf4") as dataset:
        ages = tuple(int(value) for value in dataset.age_band_lower_bound.values)
        if ages != tuple(range(0, 70, 5)):
            raise ValueError(
                "The original Chambers source must end with one cumulative 65+ band."
            )
        population = (
            dataset.demographic_totals.sel(
                year=[*BASELINE_YEARS, TARGET_YEAR],
                age_band_lower_bound=65,
                latitude=slice(55.0, 17.0),
                longitude=slice(72.0, 136.0),
            )
            .load()
            .transpose("year", "latitude", "longitude")
        )
        latitudes = np.asarray(population.latitude.values, dtype=np.float64)
        longitudes = np.asarray(population.longitude.values, dtype=np.float64)
        values = np.asarray(population.values, dtype=np.float64)
    expected_shape = (len(BASELINE_YEARS) + 1, latitudes.size, longitudes.size)
    if values.shape != expected_shape:
        raise ValueError("The Chambers regional population grid has changed shape.")
    return latitudes, longitudes, values


def _china_mask(
    country_raster: Path,
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
) -> CountryMaskAssignment:
    """Assign each target-cell footprint to its majority-area GPW country.

    Args:
        country_raster: Native 30-arc-second GPWv4 national identifier raster.
        latitudes: Descending 0.5-degree target-cell center latitudes.
        longitudes: Ascending 0.5-degree target-cell center longitudes.

    Returns:
        Mainland-China assignments and footprint-alignment diagnostics.

    Raises:
        ValueError: If either grid has unexpected geometry or no cells are
            assigned to China.

    Examples:
        The national mask intentionally excludes separately coded Hong Kong
        (344) and Taiwan (158), matching the published national denominator.
    """
    if (
        latitudes.ndim != 1
        or longitudes.ndim != 1
        or latitudes.size == 0
        or longitudes.size == 0
    ):
        raise ValueError("Target country coordinates must be non-empty 1D arrays.")
    if not np.allclose(
        np.diff(latitudes),
        -ASSIGNMENT_GRID_RESOLUTION,
        rtol=0,
        atol=1e-12,
    ) or not np.allclose(
        np.diff(longitudes),
        ASSIGNMENT_GRID_RESOLUTION,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Target country coordinates must use a regular 0.5-degree grid."
        )

    left = float(longitudes[0] - ASSIGNMENT_GRID_RESOLUTION / 2)
    right = float(longitudes[-1] + ASSIGNMENT_GRID_RESOLUTION / 2)
    top = float(latitudes[0] + ASSIGNMENT_GRID_RESOLUTION / 2)
    bottom = float(latitudes[-1] - ASSIGNMENT_GRID_RESOLUTION / 2)
    with rasterio.open(country_raster) as dataset:
        expected_transform = rasterio.Affine(
            COUNTRY_RASTER_RESOLUTION,
            0,
            -180,
            0,
            -COUNTRY_RASTER_RESOLUTION,
            90,
        )
        expected_shape = (
            round(180 / COUNTRY_RASTER_RESOLUTION),
            round(360 / COUNTRY_RASTER_RESOLUTION),
        )
        if (
            dataset.shape != expected_shape
            or dataset.crs != rasterio.CRS.from_epsg(4326)
            or not np.allclose(
                tuple(dataset.transform)[:6],
                tuple(expected_transform)[:6],
                rtol=0,
                atol=1e-12,
            )
            or dataset.nodata != COUNTRY_RASTER_NODATA
        ):
            raise ValueError(
                "GPW national identifiers must use the published global "
                "30-arc-second grid."
            )
        source_cells_per_side = round(
            ASSIGNMENT_GRID_RESOLUTION / COUNTRY_RASTER_RESOLUTION
        )
        if source_cells_per_side != COUNTRY_CELLS_PER_ASSIGNMENT_SIDE:
            raise ValueError("GPW and target-cell resolutions do not align exactly.")
        raw_window = from_bounds(
            left,
            bottom,
            right,
            top,
            dataset.transform,
        )
        window = raw_window.round_offsets().round_lengths()
        if not np.allclose(
            (
                raw_window.col_off,
                raw_window.row_off,
                raw_window.width,
                raw_window.height,
            ),
            (window.col_off, window.row_off, window.width, window.height),
            rtol=0,
            atol=1e-9,
        ) or not np.allclose(
            dataset.window_bounds(window),
            (left, bottom, right, top),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Target cell footprints do not align with GPW pixels.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting the shape on a NumPy array has been deprecated",
                category=DeprecationWarning,
            )
            identifiers = dataset.read(1, window=window)
        expected_shape = (
            latitudes.size * source_cells_per_side,
            longitudes.size * source_cells_per_side,
        )
        if identifiers.shape != expected_shape:
            raise ValueError(
                f"Aligned GPW window has shape {identifiers.shape}; "
                f"expected {expected_shape}."
            )
        row_offsets = int(window.row_off) + np.arange(identifiers.shape[0])
        source_tops = dataset.transform.f + row_offsets * dataset.transform.e
        source_bottoms = source_tops + dataset.transform.e

    row_areas = np.sin(np.deg2rad(source_tops)) - np.sin(np.deg2rad(source_bottoms))
    blocks = identifiers.reshape(
        latitudes.size,
        source_cells_per_side,
        longitudes.size,
        source_cells_per_side,
    )
    block_row_areas = row_areas.reshape(latitudes.size, source_cells_per_side)
    country_codes = np.unique(identifiers)
    country_codes = country_codes[country_codes != COUNTRY_RASTER_NODATA]
    if country_codes.size == 0:
        raise ValueError("Aligned GPW country window contains no country identifiers.")
    country_areas = np.stack(
        [
            np.einsum(
                "abcd,ab->ac",
                blocks == country_code,
                block_row_areas,
                dtype=np.float64,
                optimize=True,
            )
            for country_code in country_codes
        ]
    )
    maximum_area = np.max(country_areas, axis=0)
    tied = (np.sum(country_areas == maximum_area, axis=0) > 1) & (maximum_area > 0)
    tied_count = int(np.count_nonzero(tied))
    china_indexes = np.flatnonzero(country_codes == CHINA_UN_M49)
    if china_indexes.size != 1:
        raise ValueError("Aligned GPW country window must contain China code 156.")
    china_area = country_areas[int(china_indexes[0])]
    china_tied = tied & (china_area == maximum_area)
    mask = (china_area == maximum_area) & ~tied & (maximum_area > 0)
    if not mask.any():
        raise ValueError("The GPW country grid contains no UN M49 156 cells.")
    return CountryMaskAssignment(
        values=mask,
        source_cells_per_target_side=source_cells_per_side,
        tied_target_cells=tied_count,
        china_tied_target_cells=int(np.count_nonzero(china_tied)),
    )


def _extract_country_raster(archive_path: Path, destination: Path) -> Path:
    """Extract the exact country raster from its verified SEDAC archive.

    Args:
        archive_path: Verified SEDAC ZIP archive.
        destination: Cache path for the GeoTIFF.

    Returns:
        The extracted GeoTIFF path.

    Raises:
        ValueError: If the archive does not contain one exact raster member.

    Examples:
        The extraction ignores the accompanying shapefile and lookup tables.
    """
    if destination.is_file():
        try:
            _verify_file(
                destination,
                expected_size=COUNTRY_RASTER_SIZE,
                expected_sha256=COUNTRY_RASTER_SHA256,
                expected_md5=None,
            )
        except ValueError:
            destination.unlink()
        else:
            return destination

    with ZipFile(archive_path) as archive:
        matches = [
            member
            for member in archive.infolist()
            if Path(member.filename).name == COUNTRY_RASTER_FILENAME
        ]
        if len(matches) != 1:
            raise ValueError(
                f"GPW archive must contain exactly one {COUNTRY_RASTER_FILENAME}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(matches[0]) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
    try:
        _verify_file(
            destination,
            expected_size=COUNTRY_RASTER_SIZE,
            expected_sha256=COUNTRY_RASTER_SHA256,
            expected_md5=None,
        )
    except ValueError:
        destination.unlink()
        raise
    return destination


def _expected_era5_digests() -> dict[str, str]:
    """Read normalized annual ERA5 digests from the committed golden metadata.

    Args:
        None.

    Returns:
        Expected normalized temperature digest keyed by cached filename.

    Raises:
        ValueError: If a recorded digest has an invalid shape.

    Examples:
        The first regeneration that introduces normalized digests returns an
        empty mapping and writes the manifest for future cache checks.
    """
    if not GOLDEN_METADATA.is_file():
        return {}
    payload = json.loads(GOLDEN_METADATA.read_text(encoding="utf-8"))
    extractions = (
        payload.get("method", {})
        .get("temperature", {})
        .get(
            "extractions",
            [],
        )
    )
    if not isinstance(extractions, list):
        raise ValueError("Golden ERA5 extractions metadata must be a list.")
    expected: dict[str, str] = {}
    for extraction in extractions:
        if not isinstance(extraction, dict):
            raise ValueError("Golden ERA5 extraction entries must be dictionaries.")
        filename = extraction.get("file")
        digest = extraction.get("normalized_t2m_sha256")
        if digest is None:
            continue
        if (
            not isinstance(filename, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Golden ERA5 normalized digest metadata is invalid.")
        expected[filename] = digest
    return expected


def _era5_expected_coordinates() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the exact source-grid coordinates requested from CDS.

    Args:
        None.

    Returns:
        Descending latitude and ascending longitude arrays.

    Examples:
        >>> latitude, longitude = _era5_expected_coordinates()
        >>> (latitude[0], longitude[0])
        (np.float64(55.0), np.float64(72.0))
    """
    north, west, south, east = ERA5_AREA
    latitude_count = round((north - south) / ERA5_SOURCE_GRID_RESOLUTION) + 1
    longitude_count = round((east - west) / ERA5_SOURCE_GRID_RESOLUTION) + 1
    return (
        np.linspace(north, south, latitude_count, dtype=np.float64),
        np.linspace(west, east, longitude_count, dtype=np.float64),
    )


def _read_era5_source(
    source_path: Path,
    years: tuple[int, ...],
    *,
    expected_digest: str | None,
) -> tuple[dict[int, NDArray[np.float32]], str]:
    """Validate and read one official ERA5 daily-statistics NetCDF response.

    Args:
        source_path: Cached annual CDS response.
        years: Exact years expected in the response.
        expected_digest: Optional committed normalized-value digest.

    Returns:
        Daily maximum arrays keyed by year and their normalized digest.

    Raises:
        ValueError: If schema, coordinates, dates, values, or digest differ.
        OSError: If the NetCDF response cannot be read.

    Examples:
        Regeneration uses this before trusting either cached or new bytes.
    """
    import xarray as xr  # deptry: ignore[DEP004]

    expected_latitudes, expected_longitudes = _era5_expected_coordinates()
    expected_times = np.concatenate(
        [
            pd.date_range(
                f"{year}-05-01",
                f"{year}-09-30",
                freq="D",
            ).to_numpy()
            for year in years
        ]
    )
    with xr.open_dataset(source_path, engine="netcdf4") as dataset:
        if tuple(dataset.data_vars) != (ERA5_VARIABLE,):
            raise ValueError(
                f"ERA5 source must contain only {ERA5_VARIABLE!r} as a data variable."
            )
        temperature = dataset[ERA5_VARIABLE]
        if temperature.dims != ("valid_time", "latitude", "longitude"):
            raise ValueError("ERA5 temperature dimensions or order are invalid.")
        if temperature.dtype != np.dtype(np.float32):
            raise ValueError("ERA5 daily maximum temperature must use float32.")
        if temperature.attrs.get("units") != "K":
            raise ValueError("ERA5 daily maximum temperature must use kelvin.")
        if not np.array_equal(dataset.latitude.values, expected_latitudes):
            raise ValueError("ERA5 source latitude coordinates are invalid.")
        if not np.array_equal(dataset.longitude.values, expected_longitudes):
            raise ValueError("ERA5 source longitude coordinates are invalid.")
        if not np.array_equal(dataset.valid_time.values, expected_times):
            raise ValueError("ERA5 source must contain each requested warm-season day.")
        values = np.asarray(temperature.values, dtype=np.float32)

    expected_shape = (
        len(years) * 153,
        expected_latitudes.size,
        expected_longitudes.size,
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"ERA5 source has shape {values.shape}; expected {expected_shape}."
        )
    minimum, maximum = ERA5_TEMPERATURE_RANGE_K
    if (
        not np.isfinite(values).all()
        or np.any(values < minimum)
        or np.any(values > maximum)
    ):
        raise ValueError(
            f"ERA5 temperatures must be finite and within {minimum}-{maximum} K."
        )
    normalized_digest = _array_digest([values])
    if expected_digest is not None and normalized_digest != expected_digest:
        raise ValueError("ERA5 normalized temperature digest does not match golden.")
    values_by_year = {
        year: values[index * 153 : (index + 1) * 153]
        for index, year in enumerate(years)
    }
    return values_by_year, normalized_digest


def _retrieve_era5_source(destination: Path, years: tuple[int, ...]) -> None:
    """Retrieve one bounded ERA5 daily-statistics response from CDS.

    Args:
        destination: Partial NetCDF path written by the CDS client.
        years: Exact years to request.

    Returns:
        None.

    Examples:
        Tests replace this network boundary with a local fixture writer.
    """
    import cdsapi  # deptry: ignore[DEP004]

    request = {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": [str(year) for year in years],
        "month": [f"{month:02d}" for month in WARM_SEASON_MONTHS],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": list(ERA5_AREA),
        "data_format": "netcdf",
    }
    cdsapi.Client().retrieve(ERA5_DATASET, request, str(destination))


def _ensure_era5_source(
    destination: Path,
    years: tuple[int, ...],
    *,
    expected_digest: str | None,
) -> Path:
    """Return one validated cache file, repairing invalid owned bytes.

    Args:
        destination: Final annual NetCDF cache path.
        years: Exact years expected in the response.
        expected_digest: Optional committed normalized-value digest.

    Returns:
        The validated final cache path.

    Raises:
        ValueError: If a fresh CDS response fails size or content validation.
        OSError: If retrieval or local file operations fail.

    Examples:
        Parseable cache corruption is removed before one fresh retrieval.
    """
    if destination.is_file():
        try:
            _read_era5_source(
                destination,
                years,
                expected_digest=expected_digest,
            )
        except (OSError, ValueError):
            destination.unlink()
        else:
            return destination

    partial = destination.with_suffix(".nc.partial")
    partial.unlink(missing_ok=True)
    _retrieve_era5_source(partial, years)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise ValueError("CDS did not create a non-empty ERA5 response.")
    if partial.stat().st_size > ERA5_MAX_BATCH_BYTES:
        partial.unlink()
        raise ValueError(f"CDS response exceeded {ERA5_MAX_BATCH_BYTES} bytes.")
    try:
        _read_era5_source(
            partial,
            years,
            expected_digest=expected_digest,
        )
    except (OSError, ValueError):
        partial.unlink()
        raise
    partial.replace(destination)
    return destination


def _era5_sources(cache_directory: Path) -> tuple[Path, ...]:
    """Return validated bounded official ERA5 daily-statistics extracts.

    Args:
        cache_directory: Directory for reusable NetCDF responses.

    Returns:
        One validated cache path for each declared year batch.

    Raises:
        ValueError: If a cached and refreshed response both fail validation.
        OSError: If CDS retrieval or local file operations fail.

    Examples:
        A configured ``~/.cdsapirc`` is required only for missing or invalid
        batches.
    """
    cache_directory.mkdir(parents=True, exist_ok=True)
    expected_digests = _expected_era5_digests()
    paths: list[Path] = []
    for years in ERA5_YEAR_BATCHES:
        first_year = years[0]
        last_year = years[-1]
        destination = cache_directory / (
            f"era5_daily_maximum_{first_year}_{last_year}.nc"
        )
        paths.append(
            _ensure_era5_source(
                destination,
                years,
                expected_digest=expected_digests.get(destination.name),
            )
        )
    return tuple(paths)


def _era5_arrays(
    source_paths: Sequence[Path],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    dict[str, str],
]:
    """Read baseline and target daily maxima from validated ERA5 extracts.

    Args:
        source_paths: Bounded CDS daily-statistics NetCDF responses.
        latitudes: Exact 0.5-degree target latitudes.
        longitudes: Exact 0.5-degree target longitudes.

    Returns:
        Baseline values, 2019 values, and normalized annual digests.

    Raises:
        ValueError: If required years or target coordinates are missing.
        OSError: If a source cannot be read.

    Examples:
        Every returned year has the report's 153-day warm season.
    """
    values_by_year: dict[int, NDArray[np.float32]] = {}
    normalized_digests: dict[str, str] = {}
    expected_digests = _expected_era5_digests()
    for source_path, years in zip(source_paths, ERA5_YEAR_BATCHES, strict=True):
        source_values, normalized_digest = _read_era5_source(
            source_path,
            years,
            expected_digest=expected_digests.get(source_path.name),
        )
        values_by_year.update(source_values)
        normalized_digests[source_path.name] = normalized_digest

    required_years = (*BASELINE_YEARS, TARGET_YEAR)
    if tuple(sorted(values_by_year)) != required_years:
        raise ValueError("ERA5 extracts do not contain exactly the required years.")
    expected_latitudes, expected_longitudes = _era5_expected_coordinates()
    latitude_indexes = np.searchsorted(-expected_latitudes, -latitudes)
    longitude_indexes = np.searchsorted(expected_longitudes, longitudes)
    if not np.array_equal(expected_latitudes[latitude_indexes], latitudes):
        raise ValueError("Target latitudes do not align with ERA5 coordinates.")
    if not np.array_equal(expected_longitudes[longitude_indexes], longitudes):
        raise ValueError("Target longitudes do not align with ERA5 coordinates.")
    selected_by_year = {
        year: values[:, latitude_indexes][:, :, longitude_indexes]
        for year, values in values_by_year.items()
    }
    baseline = np.stack([selected_by_year[year] for year in BASELINE_YEARS])
    return baseline, selected_by_year[TARGET_YEAR], normalized_digests


def _array_digest(arrays: Sequence[NDArray[np.floating]]) -> str:
    """Hash normalized little-endian float32 arrays in declared order.

    Args:
        arrays: Ordered temperature arrays to identify.

    Returns:
        Lowercase SHA-256 digest of array shapes and values.

    Examples:
        >>> len(_array_digest([np.array([1.0], dtype=np.float32)]))
        64
    """
    digest = hashlib.sha256()
    for array in arrays:
        normalized = np.asarray(array, dtype="<f4")
        digest.update(json.dumps(normalized.shape).encode())
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _positive_population_total(
    values: NDArray[np.float64],
    *,
    context: str,
) -> float:
    """Return a finite, positive population total.

    Args:
        values: Population counts to sum.
        context: Short description included in validation errors.

    Returns:
        The float64 population total.

    Raises:
        ValueError: If no positive finite population is present.

    Examples:
        >>> _positive_population_total(
        ...     np.array([1.0, 2.0]),
        ...     context="Example",
        ... )
        3.0
    """
    total = float(np.sum(values, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{context} population total must be finite and positive.")
    return total


def _inclusive_threshold_sensitivity(
    baseline_daily_maximum: NDArray[np.float32],
    target_daily_maximum: NDArray[np.float32],
    threshold: NDArray[np.float64],
    population_by_year: NDArray[np.float64],
    china_mask: NDArray[np.bool_],
) -> dict[str, float | str]:
    """Quantify the appendix table's inclusive-threshold interpretation.

    Args:
        baseline_daily_maximum: ERA5 warm seasons ordered by baseline year.
        target_daily_maximum: ERA5 warm season for 2019.
        threshold: Per-cell 92.5th-percentile threshold.
        population_by_year: Population for each baseline year followed by 2019.
        china_mask: Replacement mainland China mask.

    Returns:
        Exposure totals obtained with ``temperature >= threshold``.

    Raises:
        ValueError: If population and climate arrays do not share one grid.

    Examples:
        The generated metadata keeps this result separate from the controlling
        strict comparison used by the paper's methods sentence.
    """
    inclusive_baseline_days = np.stack(
        [
            count_heatwave_days(
                year >= threshold,
                minimum_run_days=MINIMUM_RUN_DAYS,
            )
            for year in baseline_daily_maximum
        ]
    )
    inclusive_target_days = count_heatwave_days(
        target_daily_maximum >= threshold,
        minimum_run_days=MINIMUM_RUN_DAYS,
    )
    expected_population_shape = (
        len(BASELINE_YEARS) + 1,
        *inclusive_target_days.shape,
    )
    if population_by_year.shape != expected_population_shape:
        raise ValueError("Population and climate sensitivity grids must align.")
    selected = china_mask & np.isfinite(population_by_year).all(axis=0)
    baseline_person_days = np.mean(
        inclusive_baseline_days.astype(np.float64) * population_by_year[:-1],
        axis=0,
        dtype=np.float64,
    )
    target_person_days = (
        inclusive_target_days.astype(np.float64) * population_by_year[-1]
    )
    additional_person_days = float(
        np.sum(
            target_person_days[selected] - baseline_person_days[selected],
            dtype=np.float64,
        )
    )
    population = _positive_population_total(
        population_by_year[-1][selected],
        context="Inclusive threshold sensitivity",
    )
    return {
        "comparison": "greater than or equal to",
        "additional_person_days": additional_person_days,
        "additional_days_per_person": additional_person_days / population,
        "heatwave_person_days_2019": float(
            np.sum(target_person_days[selected], dtype=np.float64)
        ),
    }


def _series_diagnostics(
    baseline_days_by_year: NDArray[np.int16],
    baseline_days: NDArray[np.float64],
    target_days: NDArray[np.int16],
    population_by_year: NDArray[np.float64],
    china_mask: NDArray[np.bool_],
) -> dict[str, object]:
    """Calculate independent checks on the paper's reported time series.

    Args:
        baseline_days_by_year: Heatwave-day grids for 1986 through 2005.
        baseline_days: Mean annual baseline heatwave-day grid.
        target_days: Heatwave-day grid for 2019.
        population_by_year: Population for each baseline year followed by 2019.
        china_mask: Replacement mainland China mask.

    Returns:
        The 2000 cross-check and current-population-weighted 2019 anomaly.

    Raises:
        ValueError: If year and population axes do not match the method.

    Examples:
        The 2000 result can be compared with the paper's reported 71.8 million
        person-days without changing the controlling calculation.
    """
    if baseline_days_by_year.shape[0] != len(BASELINE_YEARS):
        raise ValueError("Series diagnostics require all baseline years.")
    if population_by_year.shape[0] != len(BASELINE_YEARS) + 1:
        raise ValueError("Series diagnostics require baseline and target population.")
    selected = china_mask & np.isfinite(population_by_year).all(axis=0)
    _positive_population_total(
        population_by_year[-1][selected],
        context="Series diagnostics",
    )
    annual_baseline_exposure = (
        baseline_days_by_year.astype(np.float64) * population_by_year[:-1]
    )
    mean_baseline_exposure = np.mean(
        annual_baseline_exposure,
        axis=0,
        dtype=np.float64,
    )
    index_2000 = BASELINE_YEARS.index(2000)
    method_result_2000 = float(
        np.sum(
            annual_baseline_exposure[index_2000][selected]
            - mean_baseline_exposure[selected],
            dtype=np.float64,
        )
    )
    current_population_result_2000 = float(
        np.sum(
            (baseline_days_by_year[index_2000].astype(np.float64) - baseline_days)[
                selected
            ]
            * population_by_year[index_2000][selected],
            dtype=np.float64,
        )
    )
    current_population_result_2019 = float(
        np.sum(
            (target_days.astype(np.float64) - baseline_days)[selected]
            * population_by_year[-1][selected],
            dtype=np.float64,
        )
    )
    return {
        "year_2000": {
            "reported_additional_person_days": PUBLISHED_2000_PERSON_DAYS,
            "reported_rounding_tolerance": PUBLISHED_2000_PERSON_DAYS_TOLERANCE,
            "method_additional_person_days": method_result_2000,
            "method_difference_from_reported": (
                method_result_2000 - PUBLISHED_2000_PERSON_DAYS
            ),
            "method_within_reported_tolerance": (
                abs(method_result_2000 - PUBLISHED_2000_PERSON_DAYS)
                <= PUBLISHED_2000_PERSON_DAYS_TOLERANCE
            ),
            "current_population_weighted_day_anomaly_person_days": (
                current_population_result_2000
            ),
        },
        "year_2019": {
            "current_population_weighted_day_anomaly_person_days": (
                current_population_result_2019
            ),
        },
    }


def _build_rows(
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
    population_by_year: NDArray[np.float64],
    china_mask: NDArray[np.bool_],
    baseline_days_by_year: NDArray[np.int16],
    baseline_days: NDArray[np.float64],
    target_days: NDArray[np.int16],
    additional_days: NDArray[np.float64],
) -> pd.DataFrame:
    """Build the reviewable cell-level fixture used by the offline test.

    Args:
        latitudes: Descending grid latitudes.
        longitudes: Ascending grid longitudes.
        population_by_year: Population aged 65+ for the 20 baseline years,
            followed by 2019.
        china_mask: Cells assigned to mainland China by GPW.
        baseline_days_by_year: Heatwave-day counts for each baseline year.
        baseline_days: Mean annual 1986-2005 heatwave days by cell.
        target_days: 2019 heatwave days by cell.
        additional_days: 2019 heatwave days minus the baseline mean.

    Returns:
        One row per finite-population China grid cell.

    Raises:
        ValueError: If the input arrays do not share one grid or population
            values are invalid.

    Examples:
        The output keeps exact 0.5-degree coordinate keys for package assignment.
    """
    grid_shape = (latitudes.size, longitudes.size)
    arrays = (china_mask, baseline_days, target_days, additional_days)
    if any(array.shape != grid_shape for array in arrays):
        raise ValueError("Population, mask, and heatwave arrays must share one grid.")
    expected_population_shape = (len(BASELINE_YEARS) + 1, *grid_shape)
    if population_by_year.shape != expected_population_shape:
        raise ValueError(
            "Population must contain the 20 baseline years followed by 2019."
        )
    expected_baseline_shape = (len(BASELINE_YEARS), *grid_shape)
    if baseline_days_by_year.shape != expected_baseline_shape:
        raise ValueError("Heatwave days must contain all 20 baseline years.")
    finite_population = np.isfinite(population_by_year).all(axis=0)
    selected = china_mask & finite_population
    if not np.any(selected):
        raise ValueError("China mask selected no cells with complete population.")
    if np.any(population_by_year[:, selected] < 0):
        raise ValueError("Selected population values must be non-negative.")

    baseline_person_days = np.mean(
        baseline_days_by_year.astype(np.float64) * population_by_year[:-1],
        axis=0,
        dtype=np.float64,
    )
    target_population = population_by_year[-1]
    _positive_population_total(
        target_population[selected],
        context="Selected 2019",
    )
    additional_person_days = (
        target_days.astype(np.float64) * target_population - baseline_person_days
    )
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    rows = pd.DataFrame(
        {
            "longitude": longitude_grid[selected],
            "latitude": latitude_grid[selected],
            "heatwave_days_2019": target_days[selected],
            "baseline_heatwave_days": baseline_days[selected],
            "additional_heatwave_days": additional_days[selected],
            "population_65_plus_2019": target_population[selected],
            "baseline_heatwave_person_days": baseline_person_days[selected],
            "additional_heatwave_person_days": additional_person_days[selected],
        }
    )
    return rows.sort_values(
        ["latitude", "longitude"], ascending=[False, True]
    ).reset_index(drop=True)


def _exposure(rows: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """Assign population through the public API and aggregate exposure.

    Args:
        rows: Golden cell inputs produced by :func:`_build_rows`.

    Returns:
        Additional person-days, total older population, days per person, 2019
        person-days, and mean annual baseline person-days.

    Raises:
        ValueError: If the serialized per-cell result differs from the
            assignment-based calculation.

    Examples:
        Population rows are reversed so assignment cannot pass by row position.
    """
    hazard = rows[
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
    population = rows[["longitude", "latitude", "population_65_plus_2019"]].iloc[::-1]
    assigned = assign_population(
        hazard,
        population,
        population_column="population_65_plus_2019",
    )
    target_person_days_by_cell = assigned["heatwave_days_2019"].to_numpy(
        dtype=np.float64
    ) * assigned["population_65_plus_2019"].to_numpy(dtype=np.float64)
    baseline_person_days_by_cell = assigned["baseline_heatwave_person_days"].to_numpy(
        dtype=np.float64
    )
    additional_person_days_by_cell = (
        target_person_days_by_cell - baseline_person_days_by_cell
    )
    if not np.allclose(
        additional_person_days_by_cell,
        assigned["additional_heatwave_person_days"].to_numpy(dtype=np.float64),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("Per-cell exposure values do not match assigned population.")
    person_days = float(
        np.sum(
            additional_person_days_by_cell,
            dtype=np.float64,
        )
    )
    total_population = _positive_population_total(
        assigned["population_65_plus_2019"].to_numpy(dtype=np.float64),
        context="Assigned 2019",
    )
    target_person_days = float(np.sum(target_person_days_by_cell, dtype=np.float64))
    baseline_person_days = float(np.sum(baseline_person_days_by_cell, dtype=np.float64))
    return (
        person_days,
        total_population,
        person_days / total_population,
        target_person_days,
        baseline_person_days,
    )


def _metadata(
    rows: pd.DataFrame,
    *,
    temperature_digest: str,
    temperature_sources: Sequence[dict[str, object]],
    inclusive_threshold_sensitivity: dict[str, float | str],
    series_diagnostics: dict[str, object],
    country_assignment: CountryMaskAssignment,
    cells_sha256: str,
    person_days: float,
    total_population: float,
    days_per_person: float,
    target_person_days: float,
    baseline_person_days: float,
) -> dict[str, object]:
    """Build stable scientific metadata for the generated fixture.

    Args:
        rows: Cell-level golden inputs.
        temperature_digest: Digest over baseline and target daily maxima.
        temperature_sources: Cached CDS response identities.
        inclusive_threshold_sensitivity: Result under the appendix table's
            conflicting inclusive comparison.
        series_diagnostics: Cross-checks on the paper's reported time series.
        country_assignment: Aligned majority-area China assignment.
        cells_sha256: Digest of the serialized CSV.
        person_days: Reproduced additional heatwave person-days.
        total_population: Selected 2019 population aged 65+.
        days_per_person: Reproduced additional heatwave days per older person.
        target_person_days: Total 2019 heatwave person-days.
        baseline_person_days: Mean annual 1986-2005 heatwave person-days.

    Returns:
        Strict JSON-compatible metadata.

    Examples:
        The result records both publication precision and regenerated values.
    """
    person_days_match = (
        abs(person_days - PUBLISHED_PERSON_DAYS) <= PUBLISHED_PERSON_DAYS_TOLERANCE
    )
    days_per_person_match = (
        abs(days_per_person - PUBLISHED_DAYS_PER_PERSON)
        <= PUBLISHED_DAYS_PER_PERSON_TOLERANCE
    )
    absolute_2019_match = (
        abs(target_person_days - PUBLISHED_PERSON_DAYS)
        <= PUBLISHED_PERSON_DAYS_TOLERANCE
    )
    return {
        "publication": {
            "citation": (
                "Cai W, Zhang C, Suen HP, et al. The 2020 China report of "
                "the Lancet Countdown on health and climate change. Lancet "
                "Public Health. 2021;6:e131-e147."
            ),
            "doi": "10.1016/S2468-2667(20)30256-5",
            "pmcid": "PMC7966675",
            "supplement_url": (
                "https://pmc.ncbi.nlm.nih.gov/articles/instance/7966675/bin/mmc2.pdf"
            ),
            "reported_additional_person_days": PUBLISHED_PERSON_DAYS,
            "reported_additional_person_days_rounding_tolerance": (
                PUBLISHED_PERSON_DAYS_TOLERANCE
            ),
            "reported_additional_days_per_person": PUBLISHED_DAYS_PER_PERSON,
            "reported_additional_days_per_person_rounding_tolerance": (
                PUBLISHED_DAYS_PER_PERSON_TOLERANCE
            ),
        },
        "method": {
            "temperature": {
                "source": "ECMWF ERA5 hourly data on single levels",
                "source_doi": "10.24381/cds.adbb2d47",
                "citation": (
                    "Copernicus Climate Change Service. ERA5 hourly data on "
                    "single levels from 1940 to present."
                ),
                "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
                "access_product": ERA5_DATASET,
                "access_product_doi": "10.24381/cds.4991cf48",
                "asset": ERA5_VARIABLE,
                "product_type": "reanalysis",
                "daily_statistic": "daily_maximum",
                "frequency": "1_hourly",
                "time_zone": "utc+00:00",
                "request_area_north_west_south_east": list(ERA5_AREA),
                "derived_daily_max_sha256": temperature_digest,
                "extractions": list(temperature_sources),
            },
            "grid": {
                "crs": "EPSG:4326",
                "spacing_degrees": 0.5,
                "coordinate_rule": (
                    "select every exact 0.5-degree ERA5 grid point matching "
                    "the original Chambers grid"
                ),
            },
            "warm_season": {
                "start": "05-01",
                "end": "09-30",
                "days_per_year": 153,
            },
            "threshold": {
                "baseline_years": [BASELINE_YEARS[0], BASELINE_YEARS[-1]],
                "percentile": PERCENTILE,
                "sample": "all 3060 baseline warm-season daily maxima per cell",
                "quantile_method": "linear",
                "comparison": "strictly greater than",
                "inclusive_comparison_sensitivity": (inclusive_threshold_sensitivity),
            },
            "heatwave": {
                "minimum_consecutive_days": MINIMUM_RUN_DAYS,
                "counting": "all days in each qualifying run",
                "run_boundary": "each May-September warm season resets",
            },
            "population": {
                "source": "Chambers hybrid gridded demographic data",
                "record": "Zenodo 3768003",
                "doi": "10.5281/zenodo.3768003",
                "citation": (
                    "Chambers J. Hybrid gridded demographic data for the "
                    "world, 1950-2020. Zenodo; 2020."
                ),
                "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
                "file": POPULATION_FILENAME,
                "size_bytes": POPULATION_SIZE,
                "publisher_md5": POPULATION_MD5,
                "sha256": POPULATION_SHA256,
                "variable": "demographic_totals",
                "baseline_years": [BASELINE_YEARS[0], BASELINE_YEARS[-1]],
                "target_year": TARGET_YEAR,
                "age_band_lower_bound": 65,
                "age_band_meaning": "terminal cumulative band: age 65 and older",
            },
            "geography": {
                "source": "GPWv4 National Identifier Grid, Revision 11",
                "doi": "10.7927/H4TD9VDP",
                "citation": (
                    "CIESIN, Columbia University. Gridded Population of the "
                    "World, Version 4: National Identifier Grid, Revision 11. "
                    "NASA SEDAC; 2018."
                ),
                "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
                "file": COUNTRY_ARCHIVE_FILENAME,
                "size_bytes": COUNTRY_ARCHIVE_SIZE,
                "sha256": COUNTRY_ARCHIVE_SHA256,
                "raster_sha256": COUNTRY_RASTER_SHA256,
                "resolution": "30 arc-seconds",
                "selection": "UN M49 country code 156",
                "publisher_cell_rule": (
                    "source pixels use the input country covering the majority "
                    "of land area"
                ),
                "target_cell_rule": (
                    "align each centered 0.5-degree footprint with its 60 by 60 "
                    "native GPW pixels, sum spherical pixel area by country, "
                    "exclude nodata, and exclude cells without one unique "
                    "largest country area"
                ),
                "role": (
                    "traceable replacement for the unnamed original China "
                    "administrative mask"
                ),
                "caveat": (
                    "SEDAC states that this grid represents input-data coverage "
                    "and is not an official country boundary"
                ),
                "regional_bounds": [72.0, 17.0, 136.0, 55.0],
                "mask_cells": int(np.count_nonzero(country_assignment.values)),
                "finite_population_cells": len(rows),
                "source_cells_per_target_side": (
                    country_assignment.source_cells_per_target_side
                ),
                "source_cells_per_target_cell": (
                    country_assignment.source_cells_per_target_side**2
                ),
                "tied_target_cells": country_assignment.tied_target_cells,
                "china_tied_target_cells": (country_assignment.china_tied_target_cells),
                "reference_mainland_grid_cells": REFERENCE_MAINLAND_GRID_CELLS,
                "reference_grid_cell_relative_tolerance": (
                    REFERENCE_MAINLAND_GRID_RELATIVE_TOLERANCE
                ),
            },
            "baseline_population_handling": (
                "each 1986-2005 heatwave-day grid is multiplied by its same-year "
                "population grid before the 20 annual exposure grids are averaged"
            ),
            "metric": (
                "sum(2019 heatwave days * 2019 population age 65+) - mean("
                "sum(annual heatwave days * same-year population age 65+) for "
                "1986-2005)"
            ),
        },
        "fixture": {
            "file": GOLDEN_CELLS.name,
            "sha256": cells_sha256,
            "rows": len(rows),
            "columns": list(rows.columns),
            "legal_note": (
                "Compact derived cell values from cited CC BY 4.0 sources; "
                "no source raster or NetCDF is redistributed."
            ),
        },
        "reproduction": {
            "additional_person_days": person_days,
            "published_difference_person_days": person_days - PUBLISHED_PERSON_DAYS,
            "population_65_plus": total_population,
            "additional_days_per_person": days_per_person,
            "heatwave_person_days_2019": target_person_days,
            "mean_annual_heatwave_person_days_1986_2005": baseline_person_days,
        },
        "publication_comparison": {
            "status": (
                "matches reported additional exposure"
                if person_days_match and days_per_person_match
                else "outside reported additional exposure tolerance"
            ),
            "method_result_within_reported_person_days_tolerance": person_days_match,
            "method_result_within_reported_days_per_person_tolerance": (
                days_per_person_match
            ),
            "absolute_2019_within_reported_person_days_tolerance": (
                absolute_2019_match
            ),
        },
        "series_diagnostics": series_diagnostics,
    }


def reproduce(cache_directory: Path, *, write_golden: bool) -> dict[str, object]:
    """Run the bounded public-data reproduction.

    Args:
        cache_directory: Persistent cache for source and derived data.
        write_golden: Whether to replace the committed golden files.

    Returns:
        Generated metadata, whether or not committed outputs are written.

    Raises:
        ValueError: If a source identity, grid, or generated value is invalid.
        OSError: If acquisition or output persistence fails.

    Examples:
        Run through ``make regenerate-china-heatwave`` to install the locked
        validation dependencies.
    """
    source_directory = cache_directory / "source"
    population_path = _download(
        POPULATION_URL,
        source_directory / POPULATION_FILENAME,
        expected_size=POPULATION_SIZE,
        expected_sha256=POPULATION_SHA256,
        expected_md5=POPULATION_MD5,
    )
    country_archive = _download(
        COUNTRY_URL,
        source_directory / COUNTRY_ARCHIVE_FILENAME,
        expected_size=COUNTRY_ARCHIVE_SIZE,
        expected_sha256=COUNTRY_ARCHIVE_SHA256,
        bearer_token=os.environ.get("EARTHDATA_TOKEN"),
        authentication_required=True,
    )
    country_raster = _extract_country_raster(
        country_archive,
        source_directory / COUNTRY_RASTER_FILENAME,
    )
    latitudes, longitudes, population_by_year = _population_grid(population_path)
    country_assignment = _china_mask(country_raster, latitudes, longitudes)
    china_mask = country_assignment.values
    mask_cells = int(np.count_nonzero(china_mask))
    relative_mask_difference = (
        abs(mask_cells - REFERENCE_MAINLAND_GRID_CELLS) / REFERENCE_MAINLAND_GRID_CELLS
    )
    if relative_mask_difference > REFERENCE_MAINLAND_GRID_RELATIVE_TOLERANCE:
        raise ValueError(
            f"Replacement China mask has {mask_cells} cells, outside the "
            f"{REFERENCE_MAINLAND_GRID_CELLS} +/- 5% plausibility range."
        )
    era5_sources = _era5_sources(cache_directory / "era5-daily-maximum")
    baseline, target, era5_normalized_digests = _era5_arrays(
        era5_sources,
        latitudes,
        longitudes,
    )
    anomaly = heatwave_anomaly(
        baseline,
        target,
        percentile=PERCENTILE,
        minimum_run_days=MINIMUM_RUN_DAYS,
    )
    inclusive_sensitivity = _inclusive_threshold_sensitivity(
        baseline,
        target,
        anomaly.threshold,
        population_by_year,
        china_mask,
    )
    series_diagnostics = _series_diagnostics(
        anomaly.baseline_days_by_year,
        anomaly.baseline_days,
        anomaly.target_days,
        population_by_year,
        china_mask,
    )
    rows = _build_rows(
        latitudes,
        longitudes,
        population_by_year,
        china_mask,
        anomaly.baseline_days_by_year,
        anomaly.baseline_days,
        anomaly.target_days,
        anomaly.additional_days,
    )
    (
        person_days,
        total_population,
        days_per_person,
        target_person_days,
        baseline_person_days,
    ) = _exposure(rows)
    inclusive_person_days = inclusive_sensitivity["additional_person_days"]
    if not isinstance(inclusive_person_days, float):  # pragma: no cover
        raise TypeError("Inclusive sensitivity person-days must be numeric.")
    inclusive_sensitivity["additional_person_days_difference_from_strict"] = (
        inclusive_person_days - person_days
    )

    candidate_directory = cache_directory / "candidate"
    candidate_directory.mkdir(parents=True, exist_ok=True)
    candidate_cells = candidate_directory / GOLDEN_CELLS.name
    rows.to_csv(
        candidate_cells,
        index=False,
        float_format="%.10f",
        lineterminator="\n",
    )
    metadata = _metadata(
        rows,
        temperature_digest=_array_digest([*baseline, target]),
        temperature_sources=[
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "normalized_t2m_sha256": era5_normalized_digests[path.name],
            }
            for path in era5_sources
        ],
        inclusive_threshold_sensitivity=inclusive_sensitivity,
        series_diagnostics=series_diagnostics,
        country_assignment=country_assignment,
        cells_sha256=sha256_file(candidate_cells),
        person_days=person_days,
        total_population=total_population,
        days_per_person=days_per_person,
        target_person_days=target_person_days,
        baseline_person_days=baseline_person_days,
    )
    candidate_metadata = candidate_directory / GOLDEN_METADATA.name
    candidate_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if write_golden:
        GOLDEN_DIRECTORY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_cells, GOLDEN_CELLS)
        shutil.copy2(candidate_metadata, GOLDEN_METADATA)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """Parse options, reproduce the result, and print the numeric comparison.

    Args:
        argv: Optional command arguments, excluding the executable name.

    Returns:
        Zero after a successful reproduction.

    Examples:
        ``main(["--write-golden"])`` regenerates committed artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=user_cache_path("population-exposure", appauthor=False)
        / "validations"
        / "china-heatwave-2019",
    )
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="replace the committed cells.csv and golden.json after validation",
    )
    args = parser.parse_args(argv)
    metadata = reproduce(args.cache_dir.expanduser(), write_golden=args.write_golden)
    summary = {key: metadata[key] for key in ("reproduction", "publication_comparison")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
