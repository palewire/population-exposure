"""Memory-bounded annual extraction from the Chambers NetCDF source."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from rasterio.io import DatasetReader

_VARIABLE = "demographic_totals"
_AGE_BANDS = tuple(range(0, 101, 5))
_BLOCK_SIZE = 256


def derive_chambers_year(source_path: Path, output_path: Path, year: int) -> None:
    """Derive one annual total without loading the full source cube."""
    try:
        with _open_source(source_path) as source:
            _derive_from_reader(source, output_path, year)
    except RasterioIOError as error:
        raise ValueError(
            "Chambers source is not the expected NetCDF-4 demographic_totals cube."
        ) from error


def _open_source(source_path: Path) -> DatasetReader:
    """Open the one published NetCDF variable through GDAL."""
    subdataset = f'NETCDF:"{source_path}":{_VARIABLE}'
    return cast("DatasetReader", rasterio.open(subdataset))


def _derive_from_reader(
    source: DatasetReader,
    output_path: Path,
    year: int,
) -> None:
    """Sum a selected year's 21 age bands in bounded windows."""
    if source.width != 1_440 or source.height != 721:
        raise ValueError(
            "Chambers demographic_totals must use the published 1440 by 721 grid."
        )
    if source.crs != rasterio.CRS.from_epsg(4326):
        raise ValueError("Chambers demographic_totals must use WGS 84 coordinates.")
    resolution = (abs(source.transform.a), abs(source.transform.e))
    if not np.allclose(resolution, (0.25, 0.25), rtol=0, atol=1e-12):
        raise ValueError(
            "Chambers demographic_totals must use 0.25 degree grid spacing."
        )

    bands = _bands_for_year(source, year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=source.width,
        height=source.height,
        count=1,
        dtype="float64",
        crs=source.crs,
        transform=source.transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=_BLOCK_SIZE,
        blockysize=_BLOCK_SIZE,
        BIGTIFF="IF_NEEDED",
    ) as output:
        for window in _windows(source.shape):
            total = np.zeros((int(window.height), int(window.width)), dtype=np.float64)
            valid_any = np.zeros(total.shape, dtype=bool)
            for band in bands:
                values = source.read(band, window=window, masked=True)
                mask = np.ma.getmaskarray(values)
                valid = np.asarray(values.data[~mask], dtype=np.float64)
                if not np.isfinite(valid).all() or (valid < 0).any():
                    raise ValueError(
                        "Chambers demographic_totals contains invalid population "
                        "values."
                    )
                total += np.asarray(values.filled(0), dtype=np.float64)
                valid_any |= ~mask
            total[~valid_any] = np.nan
            output.write(total, 1, window=window)
        output.update_tags(
            1,
            population_semantics="count",
            units="population count per cell",
            year=str(year),
            source_variable=_VARIABLE,
            processing="sum of 21 five-year age bands",
        )


def _bands_for_year(source: DatasetReader, year: int) -> tuple[int, ...]:
    """Find and verify the 21 published age bands for one year."""
    selected: list[tuple[int, int]] = []
    for band in range(1, source.count + 1):
        tags = source.tags(band)
        band_year = _integer_tag(tags.get("NETCDF_DIM_year"))
        age = _integer_tag(tags.get("NETCDF_DIM_age_band_lower_bound"))
        if band_year == year and age is not None:
            selected.append((age, band))
    selected.sort()
    ages = tuple(age for age, _ in selected)
    if ages != _AGE_BANDS:
        raise ValueError(
            f"Chambers source does not contain all 21 age bands for year {year}."
        )
    return tuple(band for _, band in selected)


def _integer_tag(value: str | None) -> int | None:
    """Parse GDAL NetCDF dimension tags such as ``{2020}`` or ``2020``."""
    if value is None:
        return None
    normalized = value.strip().strip("{}")
    try:
        number = float(normalized)
    except ValueError:
        return None
    if not number.is_integer():
        return None
    return int(number)


def _windows(shape: tuple[int, int]) -> Iterator[Window]:
    """Yield small top-to-bottom, left-to-right output windows."""
    height, width = shape
    for row in range(0, height, _BLOCK_SIZE):
        for column in range(0, width, _BLOCK_SIZE):
            yield Window.from_slices(
                (row, min(row + _BLOCK_SIZE, height)),
                (column, min(column + _BLOCK_SIZE, width)),
            )
