"""Memory-bounded annual extraction from the Chambers NetCDF source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

if TYPE_CHECKING:
    from pathlib import Path

_VARIABLE = "demographic_totals"
_LATITUDE = "latitude"
_LONGITUDE = "longitude"
_AGE_BAND = "age_band_lower_bound"
_YEAR = "year"
_SOURCE_SHAPE = (720, 1_440, 14, 71)
_AGE_BANDS = np.arange(0, 70, 5, dtype=np.int64)
_YEARS = np.arange(1950, 2021, dtype=np.int64)
_LATITUDES = np.arange(360, -360, -1, dtype=np.float64) / 4
_LONGITUDES = np.arange(1_440, dtype=np.float64) / 4
_BLOCK_SIZE = 256
_TRANSFORM = from_origin(-180, 90, 0.25, 0.25)


def derive_chambers_year(source_path: Path, output_path: Path, year: int) -> None:
    """Derive one annual total without loading the full source cube.

    Args:
        source_path: Path to the published Chambers NetCDF-4 file.
        output_path: Destination path for the derived GeoTIFF.
        year: Annual population total to derive, from 1950 through 2020.

    Returns:
        None. The derived single-band GeoTIFF is written to ``output_path``.

    Raises:
        ValueError: If the source is not the expected Chambers NetCDF-4 cube.

    Examples:
        >>> derive_chambers_year(Path("source.nc"), Path("chambers-2020.tif"), 2020)
    """
    try:
        with h5py.File(source_path, "r") as source:
            _derive_from_file(source, output_path, year)
    except OSError as error:
        raise ValueError(
            "Chambers source is not the expected NetCDF-4 demographic_totals cube."
        ) from error


def _derive_from_file(source: h5py.File, output_path: Path, year: int) -> None:
    """Sum one published year's demographic bands in bounded row windows.

    Args:
        source: Open HDF5 file containing the published Chambers datasets.
        output_path: Destination path for the derived GeoTIFF.
        year: Annual population total to derive.

    Returns:
        None. The derived GeoTIFF is written to ``output_path``.

    Raises:
        ValueError: If the published dimensions, coordinates, or values differ
            from the documented Chambers source.

    Examples:
        >>> with h5py.File("source.nc") as source:
        ...     _derive_from_file(source, Path("chambers-2020.tif"), 2020)
    """
    totals = _dataset(source, _VARIABLE)
    if totals.shape != _SOURCE_SHAPE or totals.dtype != np.dtype(np.float32):
        raise ValueError(
            "Chambers demographic_totals must be a float32 720 by 1440 by 14 by 71 "
            "cube."
        )
    _validate_coordinates(source)
    year_indexes = np.flatnonzero(_dataset_values(source, _YEAR) == year)
    if not year_indexes.size:
        raise ValueError(f"Chambers source does not contain requested year {year}.")
    year_index = int(year_indexes[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=_SOURCE_SHAPE[1],
        height=_SOURCE_SHAPE[0],
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=_TRANSFORM,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=_BLOCK_SIZE,
        blockysize=_BLOCK_SIZE,
        BIGTIFF="IF_NEEDED",
    ) as output:
        for window in _windows():
            output.write(
                _window_total(totals, window, year_index),
                1,
                window=window,
            )
        output.update_tags(
            1,
            population_semantics="count",
            units="population count per cell",
            year=str(year),
            source_variable=_VARIABLE,
            processing="sum of 14 five-year age bands",
        )


def _dataset(source: h5py.File, name: str) -> h5py.Dataset:
    """Return one required dataset from the Chambers source.

    Args:
        source: Open HDF5 file containing the published Chambers datasets.
        name: Name of the required dataset.

    Returns:
        The named HDF5 dataset.

    Raises:
        ValueError: If the named dataset is missing or is a group.

    Examples:
        >>> with h5py.File("source.nc") as source:
        ...     demographic_totals = _dataset(source, "demographic_totals")
    """
    dataset = source.get(name)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Chambers source is missing the {name!r} dataset.")
    return dataset


def _dataset_values(source: h5py.File, name: str) -> np.ndarray:
    """Read one small, required coordinate dataset from the Chambers source.

    Args:
        source: Open HDF5 file containing the published Chambers datasets.
        name: Name of the coordinate dataset.

    Returns:
        Coordinate values as a NumPy array.

    Examples:
        >>> with h5py.File("source.nc") as source:
        ...     years = _dataset_values(source, "year")
    """
    return np.asarray(_dataset(source, name))


def _validate_coordinates(source: h5py.File) -> None:
    """Require the documented coordinate axes and requested year.

    Args:
        source: Open HDF5 file containing the published Chambers datasets.

    Returns:
        None.

    Raises:
        ValueError: If a coordinate axis differs from the published source.

    Examples:
        >>> with h5py.File("source.nc") as source:
        ...     _validate_coordinates(source)
    """
    expected_axes = {
        _LATITUDE: _LATITUDES,
        _LONGITUDE: _LONGITUDES,
        _AGE_BAND: _AGE_BANDS,
        _YEAR: _YEARS,
    }
    for name, expected in expected_axes.items():
        values = _dataset_values(source, name)
        if not np.array_equal(values, expected):
            raise ValueError(
                f"Chambers {name} coordinates differ from the published source."
            )


def _window_total(
    source: h5py.Dataset,
    window: Window,
    year_index: int,
) -> np.ndarray:
    """Sum one annual source window and shift longitudes to -180 through 180.

    Args:
        source: Four-dimensional Chambers demographic totals dataset.
        window: Row and column window in the derived GeoTIFF grid.
        year_index: Zero-based index of the requested year in the source.

    Returns:
        A float64 population-total array for ``window``.

    Raises:
        ValueError: If a source value is infinite or negative.

    Examples:
        >>> totals = _window_total(source, Window(0, 0, 1440, 256), 70)
    """
    row_start = int(window.row_off)
    row_stop = row_start + int(window.height)
    values = np.asarray(
        source[row_start:row_stop, :, :, year_index],
        dtype=np.float64,
    )
    valid = ~np.isnan(values)
    finite = values[valid]
    if not np.isfinite(finite).all() or (finite < 0).any():
        raise ValueError(
            "Chambers demographic_totals contains invalid population values."
        )
    total = np.nansum(values, axis=2, dtype=np.float64)
    total[~valid.any(axis=2)] = np.nan
    return np.roll(total, -_SOURCE_SHAPE[1] // 2, axis=1)


def _windows() -> tuple[Window, ...]:
    """Return bounded output windows that cover the published global grid.

    Args:
        None.

    Returns:
        Row-major windows no larger than 256 by 256 cells.

    Examples:
        >>> _windows()[0]
        Window(col_off=0, row_off=0, width=256, height=256)
    """
    height, width = _SOURCE_SHAPE[:2]
    windows: list[Window] = []
    for row in range(0, height, _BLOCK_SIZE):
        for column in range(0, width, _BLOCK_SIZE):
            window = Window.from_slices(
                (row, min(row + _BLOCK_SIZE, height)),
                (column, min(column + _BLOCK_SIZE, width)),
            )
            assert isinstance(window, Window)
            windows.append(window)
    return tuple(windows)
