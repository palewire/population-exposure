"""Regenerate the China 2019 heatwave exposure golden artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np
import pandas as pd
import rasterio
from platformdirs import user_cache_path

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

COUNTRY_ARCHIVE_FILENAME = "gpw-v4-national-identifier-grid-rev11_30_min_tif.zip"
COUNTRY_RASTER_FILENAME = "gpw_v4_national_identifier_grid_rev11_30_min.tif"
COUNTRY_URL = (
    "https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/"
    "sedac-root/downloads/data/gpw-v4/gpw-v4-national-identifier-grid-rev11/"
    f"{COUNTRY_ARCHIVE_FILENAME}"
)
COUNTRY_ARCHIVE_SIZE = 95_190
COUNTRY_ARCHIVE_SHA256 = "b84c0c57918fec1df004c9aedc045ced3cdf9950e17e0272b332f13c3980bcd0"  # pragma: allowlist secret
COUNTRY_RASTER_SIZE = 37_453
COUNTRY_RASTER_SHA256 = "58a52cd474946294f3b98edd6276288b7a73cdb17006ee2b6eb4975c924ea5f6"  # pragma: allowlist secret

ERA5_DATASET = "derived-era5-single-levels-daily-statistics"
ERA5_VARIABLE = "t2m"
ERA5_AREA = (55, 72, 17, 136)
ERA5_YEAR_BATCHES = tuple((year,) for year in (*BASELINE_YEARS, TARGET_YEAR))
ERA5_MAX_BATCH_BYTES = 100_000_000

GOLDEN_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "china_heatwave_2019"
)
GOLDEN_CELLS = GOLDEN_DIRECTORY / "cells.csv"
GOLDEN_METADATA = GOLDEN_DIRECTORY / "golden.json"


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
) -> NDArray[np.bool_]:
    """Identify the report's national China cells with the GPW country grid.

    Args:
        country_raster: GPWv4 Revision 11 30-minute national identifier raster.
        latitudes: Population-grid latitudes.
        longitudes: Population-grid longitudes in zero-to-360 notation.

    Returns:
        A boolean grid selecting UN M49 code 156, mainland China.

    Raises:
        ValueError: If no cells match the expected China code.

    Examples:
        The national mask intentionally excludes separately coded Hong Kong
        (344) and Taiwan (158), matching the published national denominator.
    """
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    signed_longitudes = np.where(
        longitude_grid > 180,
        longitude_grid - 360,
        longitude_grid,
    )
    points = zip(
        signed_longitudes.ravel(),
        latitude_grid.ravel(),
        strict=True,
    )
    with rasterio.open(country_raster) as dataset:
        identifiers = np.fromiter(
            (sample[0] for sample in dataset.sample(points)),
            dtype=np.int16,
            count=longitude_grid.size,
        ).reshape(longitude_grid.shape)
    mask = identifiers == CHINA_UN_M49
    if not mask.any():
        raise ValueError("The GPW country grid contains no UN M49 156 cells.")
    return mask


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


def _era5_sources(cache_directory: Path) -> tuple[Path, ...]:
    """Return bounded official ERA5 daily-statistics extracts.

    Args:
        cache_directory: Directory for reusable NetCDF responses.

    Returns:
        One cached NetCDF path for each declared year batch.

    Raises:
        ValueError: If CDS does not create a non-empty NetCDF response.
        OSError: If a CDS request or local file operation fails.

    Examples:
        A configured ``~/.cdsapirc`` is required only for missing batches.
    """
    import cdsapi  # deptry: ignore[DEP004]

    cache_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for years in ERA5_YEAR_BATCHES:
        first_year = years[0]
        last_year = years[-1]
        destination = cache_directory / (
            f"era5_daily_maximum_{first_year}_{last_year}.nc"
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            partial = destination.with_suffix(".nc.partial")
            partial.unlink(missing_ok=True)
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
            cdsapi.Client().retrieve(ERA5_DATASET, request, str(partial))
            if not partial.is_file() or partial.stat().st_size == 0:
                raise ValueError("CDS did not create a non-empty ERA5 response.")
            if partial.stat().st_size > ERA5_MAX_BATCH_BYTES:
                partial.unlink()
                raise ValueError(f"CDS response exceeded {ERA5_MAX_BATCH_BYTES} bytes.")
            partial.replace(destination)
        paths.append(destination)
    return tuple(paths)


def _era5_arrays(
    source_paths: Sequence[Path],
    latitudes: NDArray[np.float64],
    longitudes: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Read baseline and target daily maxima from official ERA5 extracts.

    Args:
        source_paths: Bounded CDS daily-statistics NetCDF responses.
        latitudes: Exact 0.5-degree target latitudes.
        longitudes: Exact 0.5-degree target longitudes.

    Returns:
        Baseline values ordered by year, day, latitude, and longitude, followed
        by 2019 values ordered by day, latitude, and longitude.

    Raises:
        ValueError: If source metadata, dates, shapes, or values are unexpected.

    Examples:
        Every returned year has the report's 153-day warm season.
    """
    import xarray as xr  # deptry: ignore[DEP004]

    values_by_year: dict[int, NDArray[np.float32]] = {}
    for source_path in source_paths:
        with xr.open_dataset(source_path, engine="netcdf4") as dataset:
            if ERA5_VARIABLE not in dataset:
                raise ValueError(f"ERA5 source must contain {ERA5_VARIABLE!r}.")
            temperature = dataset[ERA5_VARIABLE]
            if temperature.attrs.get("units") != "K":
                raise ValueError("ERA5 daily maximum temperature must use kelvin.")
            selected = (
                temperature.sel(latitude=latitudes, longitude=longitudes)
                .transpose("valid_time", "latitude", "longitude")
                .load()
            )
            available_years = np.unique(selected.valid_time.dt.year.values)
            for year_value in available_years:
                year = int(year_value)
                year_data = selected.sel(valid_time=str(year))
                expected_times = pd.date_range(
                    f"{year}-05-01",
                    f"{year}-09-30",
                    freq="D",
                ).to_numpy()
                if not np.array_equal(year_data.valid_time.values, expected_times):
                    raise ValueError(
                        f"ERA5 {year} must contain each May-September UTC day."
                    )
                values_by_year[year] = np.asarray(
                    year_data.values,
                    dtype=np.float32,
                )

    required_years = (*BASELINE_YEARS, TARGET_YEAR)
    if tuple(sorted(values_by_year)) != required_years:
        raise ValueError("ERA5 extracts do not contain exactly the required years.")
    expected_shape = (153, latitudes.size, longitudes.size)
    for year, values in values_by_year.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"ERA5 {year} has shape {values.shape}; expected {expected_shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"ERA5 {year} contains non-finite values.")
    baseline = np.stack([values_by_year[year] for year in BASELINE_YEARS])
    return baseline, values_by_year[TARGET_YEAR]


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
    mask_cells: int,
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
        mask_cells: GPW country-code cells inside the bounded extraction.
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
                "resolution": "30 arc-minutes",
                "selection": "UN M49 country code 156",
                "publisher_cell_rule": (
                    "aggregated pixels use the input country covering the "
                    "majority of land area"
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
                "mask_cells": mask_cells,
                "finite_population_cells": len(rows),
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
    china_mask = _china_mask(country_raster, latitudes, longitudes)
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
    baseline, target = _era5_arrays(
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
            }
            for path in era5_sources
        ],
        inclusive_threshold_sensitivity=inclusive_sensitivity,
        series_diagnostics=series_diagnostics,
        mask_cells=mask_cells,
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
